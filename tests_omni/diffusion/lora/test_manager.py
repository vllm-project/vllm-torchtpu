# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU tests for diffusion PEFT and distilled LoRA in vllm-omni."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA
from vllm_omni.diffusion.lora.loader import LoraLoaderMixin
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.request import LoRARequest

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

MAX_REL_L2 = 2.0e-2
MAX_ABS = 5.0e-2
LORA_RANK = 2
LORA_SCALE = 0.5


class _TinyLoRATransformer(nn.Module):
    packed_modules_mapping = {"to_qkv": ["to_q", "to_k", "to_v"]}
    stacked_params_mapping = [
        (".to_qkv", ".to_q", "q"),
        (".to_qkv", ".to_k", "k"),
        (".to_qkv", ".to_v", "v"),
    ]

    def __init__(self, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.to_qkv = linear.QKVParallelLinear(
            hidden_size=8,
            head_size=4,
            total_num_heads=2,
            bias=False,
            params_dtype=dtype,
        )
        self.to_out = linear.RowParallelLinear(
            input_size=24,
            output_size=8,
            bias=False,
            params_dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.to_qkv(x)
        out, _ = self.to_out(qkv)
        return out


class _TinyLoRAPipeline(nn.Module):
    def __init__(self, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.transformer = _TinyLoRATransformer(dtype=dtype)


@pytest.fixture
def single_rank_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make vLLM's parallel linear layers see a 1-rank tensor-parallel group."""
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)


@dataclass(frozen=True)
class _LoraTensors:
    """CPU fp32 input, base weights, and LoRA A/B factors of the tiny model."""

    x: torch.Tensor
    qkv_weight: torch.Tensor
    out_weight: torch.Tensor
    q_a: torch.Tensor
    q_b: torch.Tensor
    k_a: torch.Tensor
    k_b: torch.Tensor
    v_a: torch.Tensor
    v_b: torch.Tensor
    out_a: torch.Tensor
    out_b: torch.Tensor

    @property
    def qkv_delta(self) -> torch.Tensor:
        return torch.cat(
            [self.q_b @ self.q_a, self.k_b @ self.k_a, self.v_b @ self.v_a],
            dim=0,
        )

    @property
    def out_delta(self) -> torch.Tensor:
        return self.out_b @ self.out_a

    @property
    def base_qkv_ref(self) -> torch.Tensor:
        return self.x @ self.qkv_weight.T

    @property
    def ref_base(self) -> torch.Tensor:
        return self.base_qkv_ref @ self.out_weight.T

    def lora_weights(self, prefix: str) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}.to_q.lora_A.weight": self.q_a,
            f"{prefix}.to_q.lora_B.weight": self.q_b,
            f"{prefix}.to_k.lora_A.weight": self.k_a,
            f"{prefix}.to_k.lora_B.weight": self.k_b,
            f"{prefix}.to_v.lora_A.weight": self.v_a,
            f"{prefix}.to_v.lora_B.weight": self.v_b,
            f"{prefix}.to_out.lora_A.weight": self.out_a,
            f"{prefix}.to_out.lora_B.weight": self.out_b,
        }


def _random_lora_tensors() -> _LoraTensors:
    # Keyword arguments evaluate left to right, which fixes the RNG draw order.
    return _LoraTensors(
        x=torch.randn(2, 8, dtype=torch.float32),
        qkv_weight=torch.randn(24, 8, dtype=torch.float32) * 0.1,
        out_weight=torch.randn(8, 24, dtype=torch.float32) * 0.1,
        q_a=torch.randn(LORA_RANK, 8, dtype=torch.float32) * 0.2,
        q_b=torch.randn(8, LORA_RANK, dtype=torch.float32) * 0.2,
        k_a=torch.randn(LORA_RANK, 8, dtype=torch.float32) * 0.2,
        k_b=torch.randn(8, LORA_RANK, dtype=torch.float32) * 0.2,
        v_a=torch.randn(LORA_RANK, 8, dtype=torch.float32) * 0.2,
        v_b=torch.randn(8, LORA_RANK, dtype=torch.float32) * 0.2,
        out_a=torch.randn(LORA_RANK, 24, dtype=torch.float32) * 0.2,
        out_b=torch.randn(8, LORA_RANK, dtype=torch.float32) * 0.2,
    )


def _tpu_pipeline(tensors: _LoraTensors) -> _TinyLoRAPipeline:
    """bf16 tiny pipeline on TPU with the base weights loaded."""
    pipeline = _TinyLoRAPipeline(dtype=torch.bfloat16)
    with torch.no_grad():
        pipeline.transformer.to_qkv.weight.copy_(tensors.qkv_weight.to(torch.bfloat16))
        pipeline.transformer.to_out.weight.copy_(tensors.out_weight.to(torch.bfloat16))
    return pipeline.to(device=torch.device("tpu"))


@pytest.mark.usefixtures("single_rank_tp")
def test_diffusion_lora_peft(tmp_path: Path) -> None:
    """Diffusion LoRA (PEFT & Distill): PEFT adapter.

    - TPU path: Build a tiny bf16 diffusion transformer (QKVParallelLinear +
      RowParallelLinear with stacked_params_mapping) on TPU and activate a PEFT
      adapter through DiffusionLoRAManager.
    - Setup check: _active_adapter_id, wrapped DiffusionBaseLinearLayerWithLoRA
      layers, and TPU lora_a_stacked.
    - Outcome check: Compare adapted TPU forward and post-deactivate base TPU
      forward against exact CPU fp32 reference via assert_parity.
    """
    torch.manual_seed(42)
    tpu_device = torch.device("tpu")
    tensors = _random_lora_tensors()

    with set_current_vllm_config(VllmConfig()):
        pipeline = _tpu_pipeline(tensors)
        x_tpu = tensors.x.to(dtype=torch.bfloat16, device=tpu_device)

        adapted_qkv_ref = tensors.base_qkv_ref + LORA_SCALE * (
            tensors.x @ tensors.qkv_delta.T
        )
        ref_adapted = adapted_qkv_ref @ tensors.out_weight.T + LORA_SCALE * (
            adapted_qkv_ref @ tensors.out_delta.T
        )

        adapter_dir = tmp_path / "peft_lora"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            tensors.lora_weights("base_model.model.transformer"),
            str(adapter_dir / "adapter_model.safetensors"),
        )
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": LORA_RANK,
                    "lora_alpha": LORA_RANK,
                    "target_modules": ["to_q", "to_k", "to_v", "to_out"],
                }
            ),
            encoding="utf-8",
        )
        lora_request = LoRARequest(
            lora_name="tiny_peft",
            lora_int_id=1,
            lora_path=str(adapter_dir),
        )
        manager = DiffusionLoRAManager(
            pipeline=pipeline,
            device=tpu_device,
            dtype=torch.bfloat16,
            max_cached_adapters=1,
        )
        manager.set_active_adapter(lora_request, lora_scale=LORA_SCALE)

        assert manager._active_adapter_id == 1, (
            f"Expected active adapter id 1, got {manager._active_adapter_id}"
        )
        assert isinstance(
            pipeline.transformer.to_qkv, DiffusionBaseLinearLayerWithLoRA
        ), (
            f"Expected to_qkv wrapped in DiffusionBaseLinearLayerWithLoRA, "
            f"got {type(pipeline.transformer.to_qkv).__name__}"
        )
        assert isinstance(
            pipeline.transformer.to_out, DiffusionBaseLinearLayerWithLoRA
        ), (
            f"Expected to_out wrapped in DiffusionBaseLinearLayerWithLoRA, "
            f"got {type(pipeline.transformer.to_out).__name__}"
        )
        lora_a_device = pipeline.transformer.to_qkv.lora_a_stacked[0].device.type
        assert lora_a_device == "tpu", (
            f"peft: to_qkv lora_a_stacked on {lora_a_device}, expected tpu"
        )

        with torch.inference_mode():
            tpu_adapted = pipeline.transformer(x_tpu)
        assert_parity(
            tpu_adapted,
            ref_adapted,
            name="diffusion_lora_peft_adapted",
            max_rel_l2=MAX_REL_L2,
            max_abs=MAX_ABS,
        )

        manager.set_active_adapter(None)
        with torch.inference_mode():
            tpu_base = pipeline.transformer(x_tpu)
        assert_parity(
            tpu_base,
            tensors.ref_base,
            name="diffusion_lora_peft_base",
            max_rel_l2=MAX_REL_L2,
            max_abs=MAX_ABS,
        )


@pytest.mark.usefixtures("single_rank_tp")
def test_diffusion_lora_distill() -> None:
    """Diffusion LoRA (PEFT & Distill): distilled LoRA fusion.

    - TPU path: Build a tiny bf16 diffusion transformer (QKVParallelLinear +
      RowParallelLinear with stacked_params_mapping) on TPU and fuse a LoRA
      state dict through LoraLoaderMixin.load_lora_into_module.
    - Setup check: loaded LoRA keys and TPU parameter placement.
    - Outcome check: Compare fused TPU forward and post-unload base TPU forward
      against exact CPU fp32 reference via assert_parity.
    """
    torch.manual_seed(42)
    tpu_device = torch.device("tpu")
    tensors = _random_lora_tensors()

    with set_current_vllm_config(VllmConfig()):
        pipeline = _tpu_pipeline(tensors)
        x_tpu = tensors.x.to(dtype=torch.bfloat16, device=tpu_device)

        adapted_qkv_ref = tensors.base_qkv_ref + (tensors.x @ tensors.qkv_delta.T)
        ref_adapted = adapted_qkv_ref @ (tensors.out_weight + tensors.out_delta).T

        state_dict = tensors.lora_weights("transformer")
        loaded_keys = LoraLoaderMixin.load_lora_into_module(
            state_dict, pipeline.transformer, prefix="transformer"
        )
        assert loaded_keys == set(state_dict.keys()), (
            f"Expected {set(state_dict.keys())}, loaded {loaded_keys}"
        )
        qkv_device = pipeline.transformer.to_qkv.weight.device.type
        assert qkv_device == "tpu", (
            f"distill: to_qkv weight on {qkv_device} after fusion, expected tpu"
        )

        with torch.inference_mode():
            tpu_fused = pipeline.transformer(x_tpu)
        assert_parity(
            tpu_fused,
            ref_adapted,
            name="diffusion_lora_distill_fused",
            max_rel_l2=MAX_REL_L2,
            max_abs=MAX_ABS,
        )

        unloaded_keys = LoraLoaderMixin.unload_module_lora(
            state_dict, pipeline.transformer, prefix="transformer"
        )
        assert unloaded_keys == set(state_dict.keys()), (
            f"Expected {set(state_dict.keys())}, unloaded {unloaded_keys}"
        )
        with torch.inference_mode():
            tpu_unloaded = pipeline.transformer(x_tpu)
        assert_parity(
            tpu_unloaded,
            tensors.ref_base,
            name="diffusion_lora_distill_unloaded",
            max_rel_l2=MAX_REL_L2,
            max_abs=MAX_ABS,
        )
