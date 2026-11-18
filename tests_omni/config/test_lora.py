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
"""TPU tests for autoregressive LoRA config, request routing, and worker manager."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from vllm import SamplingParams
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm_omni.config.lora import LoRAConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.lora.request import LoRARequest
from vllm_omni.worker.omni_connector_validation import validate_worker_omni_connector

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from vllm_torchtpu.lora.lora_manager import TPUWorkerLoRAManager

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MAX_REL_L2 = 2.0e-2
MAX_ABS = 5.0e-2


def _write_tiny_qwen2_and_lora(
    root: Path,
) -> tuple[Path, Path, torch.Tensor, torch.Tensor]:
    model_dir = root / "tiny_qwen2"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    lora_dir = root / "tiny_lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    lora_a = torch.randn(4, 16, dtype=torch.bfloat16) * 0.1
    lora_b = torch.randn(16, 4, dtype=torch.bfloat16) * 0.1
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": lora_a,
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": lora_b,
        },
        str(lora_dir / "adapter_model.safetensors"),
    )
    (lora_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 4,
                "lora_alpha": 4,
                "target_modules": ["q_proj"],
                "peft_type": "LORA",
            }
        ),
        encoding="utf-8",
    )
    return model_dir, lora_dir, lora_a, lora_b


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "OmniEngineArgs resolves OmniTpuPlatform.get_omni_ar_worker_cls() "
        "('TPUWorker') which fails validate_worker_omni_connector (tracked by "
        "vllm-torchtpu PR #2065)"
    ),
)
def test_autoregressive_lora(tmp_path: Path) -> None:
    """Autoregressive LoRA.

    - TPU path: Build a tiny Qwen2 config + PEFT adapter, resolve
      OmniEngineArgs(worker_type="ar", requires_full_payload_input=True,
      enable_lora=True) -> create_engine_config(), route LoRARequest via
      build_engine_core_request_from_tokens, and apply TPUWorkerLoRAManager.
    - Setup check: Assert LoRAConfig resolution and connector validation on
      vllm_config.parallel_config.worker_cls.
    - Outcome check: Assert OmniEngineCoreRequest.lora_request matches,
      activate the adapter on TPU, and verify adapted output via assert_parity.
    """
    torch.manual_seed(42)
    model_dir, lora_dir, lora_a, lora_b = _write_tiny_qwen2_and_lora(tmp_path)

    engine_args = OmniEngineArgs(
        model=str(model_dir),
        worker_type="ar",
        requires_full_payload_input=True,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=16,
    )
    vllm_config = engine_args.create_engine_config()
    assert isinstance(vllm_config.lora_config, LoRAConfig), (
        f"Expected LoRAConfig, got {type(vllm_config.lora_config).__name__}"
    )
    validate_worker_omni_connector(
        vllm_config.parallel_config.worker_cls, required=True
    )

    lora_req = LoRARequest(
        lora_name="tiny_ar_lora",
        lora_int_id=1,
        lora_path=str(lora_dir),
    )
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params_with_lora = SimpleNamespace(
        max_tokens=1,
        min_tokens=0,
        lora_request=lora_req,
        clone=lambda: sampling_params,
    )
    core_req = build_engine_core_request_from_tokens(
        request_id="req-ar-lora",
        prompt={"prompt_token_ids": [1, 2, 3]},
        params=sampling_params_with_lora,  # type: ignore[arg-type]
    )
    assert core_req.lora_request == lora_req, (
        f"Expected lora_request={lora_req}, got {core_req.lora_request}"
    )

    tpu_device = torch.device("tpu")
    linear_layer = ColumnParallelLinear(
        16, 16, bias=False, params_dtype=torch.bfloat16
    ).to(tpu_device)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].self_attn = torch.nn.Module()
    model.layers[0].self_attn.q_proj = linear_layer

    manager = TPUWorkerLoRAManager(
        vllm_config=vllm_config,
        device=tpu_device,
        embedding_modules={},
        max_num_tokens=16,
    )
    manager.create_lora_manager(model)
    manager.add_adapter(lora_req)

    x_cpu = torch.randn(2, 16, dtype=torch.float32)
    ref_out = x_cpu @ linear_layer.weight.cpu().float().T + (
        (x_cpu @ lora_a.float().T) @ lora_b.float().T
    )
    tpu_out, _ = model.layers[0].self_attn.q_proj(
        x_cpu.to(dtype=torch.bfloat16, device=tpu_device)
    )
    assert_parity(
        tpu_out,
        ref_out,
        name="autoregressive_lora",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
