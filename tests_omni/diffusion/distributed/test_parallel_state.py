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
"""TPU tests for diffusion tensor parallelism in parallel_state."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from diffusers.models.transformers.transformer_wan import (
    WanTransformer3DModel as RefWanTransformer,
)
from torch_tpu._internal.utils import hardware
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    load_wan_weights_with_optional_gate,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    DistributedRMSNorm,
    WanTransformer3DModel,
)
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.platforms import current_omni_platform

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_all_loaded
from tests_omni.tiny_models import tiny_diffusion_dir
from vllm_torchtpu.layers.adapter.linear import TpuRowParallelLinear

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

TP_SIZE = 2
REL_L2_RATIO = 1.5
REL_L2_FLOOR = 1e-2

_TINY_WAN_CONFIG: dict[str, Any] = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 4,
    "attention_head_dim": 128,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 64,
    "freq_dim": 256,
    "ffn_dim": 256,
    "num_layers": 2,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "rope_max_seq_len": 256,
}


def _rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    diff = actual.detach().cpu().double() - reference.detach().cpu().double()
    return float(diff.norm() / reference.detach().cpu().double().norm())


def _assert_bf16_parity(
    actual: torch.Tensor,
    ref_fp32: torch.Tensor,
    ref_bf16: torch.Tensor,
    *,
    name: str,
) -> None:
    assert actual.shape == ref_fp32.shape, (
        f"{name}: shape {tuple(actual.shape)} != expected {tuple(ref_fp32.shape)}"
    )
    assert bool(torch.isfinite(actual).all()), f"{name}: output has non-finite values"
    cpu_bf16_err = _rel_l2(ref_bf16, ref_fp32)
    limit = max(REL_L2_RATIO * cpu_bf16_err, REL_L2_FLOOR)
    tpu_err = _rel_l2(actual, ref_fp32)
    assert tpu_err <= limit, (
        f"{name}: relative L2 error {tpu_err:.4e} exceeds limit {limit:.4e} "
        f"(cpu_bf16_err={cpu_bf16_err:.4e})"
    )


def _cast_inputs(
    inputs: dict[str, torch.Tensor], device: torch.device | str, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    return {k: v.to(device=device, dtype=dtype) for k, v in inputs.items()}


def _tp_worker(rank: int, world_size: int, master_port: int, work_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    device = current_omni_platform.get_torch_device(rank)
    current_omni_platform.set_device(device)

    od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(),
        dtype=torch.bfloat16,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=world_size),
    )
    vllm_config = create_diffusion_vllm_config(device, od_config)
    current_omni_platform.init_diffusion_worker_vllm_config(vllm_config)

    with (
        set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config),
        set_current_diffusion_config(od_config),
    ):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="env://",
            local_rank=rank,
        )
        try:
            initialize_model_parallel(tensor_parallel_size=world_size)
            payload = torch.load(Path(work_dir) / "payload.pt", weights_only=True)
            with set_default_torch_dtype(torch.bfloat16):
                dit = WanTransformer3DModel(**_TINY_WAN_CONFIG)
            parent = torch.nn.Module()
            parent.transformer = dit
            loaded = load_wan_weights_with_optional_gate(
                parent,
                ((f"transformer.{k}", v) for k, v in payload["state_dict"].items()),
            )
            assert_all_loaded({k.removeprefix("transformer.") for k in loaded}, dit)
            dit = dit.to(device=device).eval()

            tpu_inputs = _cast_inputs(payload["inputs"], device, torch.bfloat16)
            with torch.no_grad():
                out = dit(**tpu_inputs, return_dict=False)[0]

            block0 = dit.blocks[0]
            torch.save(
                {
                    "tp_world_size": get_tensor_model_parallel_world_size(),
                    "is_tpu_row_linear": type(block0.ffn.net_2) is TpuRowParallelLinear,
                    "is_dist_rms_norm": type(block0.attn1.norm_q) is DistributedRMSNorm,
                    "qkv_rows": int(block0.attn1.to_qkv.weight.shape[0]),
                    "ffn_down_cols": int(block0.ffn.net_2.weight.shape[1]),
                    "output": out.detach().cpu().float(),
                },
                Path(work_dir) / f"rank_{rank}.pt",
            )
        finally:
            destroy_distributed_env()


def test_diffusion_tensor_parallelism(tmp_path: Path) -> None:
    """Diffusion Tensor Parallelism: TP=2 Wan DiT parity against TP=1 CPU fp32.

    - TPU path: spawns 2 workers on tpu_dist with tensor_parallel_size=2 and runs a
      2-layer WanTransformer3DModel (QKVParallelLinear, ColumnParallelGELU,
      TpuRowParallelLinear, DistributedRMSNorm) in bf16 on (1, 16, 2, 8, 8) latents.
    - Setup check: get_tensor_model_parallel_world_size() == 2, ffn.net_2 is
      TpuRowParallelLinear, attn1.norm_q is DistributedRMSNorm, and QKV/FFN weights
      are half-sharded per rank.
    - Outcome check: rank-0 and rank-1 outputs match bitwise and stay within
      max(REL_L2_RATIO * cpu_bf16_err, REL_L2_FLOOR) of the TP=1 CPU fp32 output.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= TP_SIZE, (
        f"Diffusion TP test requires {TP_SIZE} TPU devices, found {device_count}"
    )

    # CPU generator only: torch.manual_seed() also seeds, and so opens, the TPU.
    torch.default_generator.manual_seed(0)
    ref_fp32_model = RefWanTransformer(**_TINY_WAN_CONFIG).eval()
    ref_bf16_model = copy.deepcopy(ref_fp32_model).to(torch.bfloat16).eval()

    gen = torch.Generator().manual_seed(7)
    inputs = {
        "hidden_states": torch.randn(1, 16, 2, 8, 8, generator=gen),
        "timestep": torch.tensor([900.0]),
        "encoder_hidden_states": torch.randn(1, 16, 64, generator=gen),
    }

    with torch.no_grad():
        ref_fp32 = ref_fp32_model(**inputs, return_dict=False)[0].float()
        ref_bf16 = ref_bf16_model(
            **_cast_inputs(inputs, "cpu", torch.bfloat16), return_dict=False
        )[0].float()

    torch.save(
        {"state_dict": ref_fp32_model.state_dict(), "inputs": inputs},
        tmp_path / "payload.pt",
    )

    master_port = get_open_port()
    try:
        torch.multiprocessing.spawn(
            _tp_worker,
            args=(TP_SIZE, master_port, str(tmp_path)),
            nprocs=TP_SIZE,
        )
    finally:
        wait_for_tpu_release()

    rank0 = torch.load(tmp_path / "rank_0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank_1.pt", weights_only=True)

    expected_qkv_rows = (
        3
        * (_TINY_WAN_CONFIG["num_attention_heads"] // TP_SIZE)
        * _TINY_WAN_CONFIG["attention_head_dim"]
    )
    expected_ffn_cols = _TINY_WAN_CONFIG["ffn_dim"] // TP_SIZE
    for rank_idx, res in enumerate((rank0, rank1)):
        assert res["tp_world_size"] == TP_SIZE, (
            f"rank {rank_idx}: expected tp_world_size={TP_SIZE}, "
            f"got {res['tp_world_size']}"
        )
        assert res["is_tpu_row_linear"] is True, (
            f"rank {rank_idx}: expected ffn.net_2 to be TpuRowParallelLinear"
        )
        assert res["is_dist_rms_norm"] is True, (
            f"rank {rank_idx}: expected attn1.norm_q to be DistributedRMSNorm"
        )
        assert res["qkv_rows"] == expected_qkv_rows, (
            f"rank {rank_idx}: expected QKV weight rows {expected_qkv_rows}, "
            f"got {res['qkv_rows']}"
        )
        assert res["ffn_down_cols"] == expected_ffn_cols, (
            f"rank {rank_idx}: expected FFN net_2 cols {expected_ffn_cols}, "
            f"got {res['ffn_down_cols']}"
        )

    assert torch.equal(rank0["output"], rank1["output"]), (
        "TP=2 rank-0 and rank-1 outputs differed after TpuRowParallelLinear all-reduce"
    )
    _assert_bf16_parity(rank0["output"], ref_fp32, ref_bf16, name="Wan TP=2")
