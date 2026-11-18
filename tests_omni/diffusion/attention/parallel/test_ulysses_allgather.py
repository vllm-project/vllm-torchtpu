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
"""TPU tests for composed 2D Ulysses x AllGather-KV sequence parallelism."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch_tpu._internal.utils import hardware
from vllm.utils.network_utils import get_open_port
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.ulysses_allgather import (
    UlyssesAllGatherKVParallelAttention,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.platforms import current_omni_platform

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

ULYSSES_DEGREE = 2
ALLGATHER_DEGREE = 2
WORLD_SIZE = ULYSSES_DEGREE * ALLGATHER_DEGREE
IMG_SEQ_LEN = 64
JOINT_SEQ_LEN = 8
NUM_HEADS = 4
HEAD_DIM = 128
REL_L2_RATIO = 1.5
REL_L2_FLOOR = 1e-2


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


def _reference_joint_attention(
    q_img: torch.Tensor,
    k_img: torch.Tensor,
    v_img: torch.Tensor,
    jq: torch.Tensor,
    jk: torch.Tensor,
    jv: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.cat([jq.to(dtype), q_img.to(dtype)], dim=1)
    k = torch.cat([jk.to(dtype), k_img.to(dtype)], dim=1)
    v = torch.cat([jv.to(dtype), v_img.to(dtype)], dim=1)
    out = (
        F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=HEAD_DIM**-0.5,
        )
        .transpose(1, 2)
        .float()
    )
    return out[:, JOINT_SEQ_LEN:], out[:, :JOINT_SEQ_LEN]


def _ulysses_allgather_worker(
    rank: int, world_size: int, master_port: int, work_dir: str
) -> None:
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
        parallel_config=DiffusionParallelConfig(
            ulysses_degree=ULYSSES_DEGREE,
            allgather_degree=ALLGATHER_DEGREE,
            ring_degree=1,
            sequence_parallel_size=world_size,
        ),
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
            initialize_model_parallel(
                ulysses_degree=ULYSSES_DEGREE,
                allgather_degree=ALLGATHER_DEGREE,
                ring_degree=1,
                sequence_parallel_size=world_size,
            )
            attn = Attention(
                num_heads=NUM_HEADS,
                head_size=HEAD_DIM,
                causal=False,
                softmax_scale=HEAD_DIM**-0.5,
            )
            payload = torch.load(Path(work_dir) / "payload.pt", weights_only=True)
            local_len = IMG_SEQ_LEN // world_size
            start = rank * local_len
            end = start + local_len
            q_local = payload["q"][:, start:end].to(device, torch.bfloat16).contiguous()
            k_local = payload["k"][:, start:end].to(device, torch.bfloat16).contiguous()
            v_local = payload["v"][:, start:end].to(device, torch.bfloat16).contiguous()
            jq = payload["jq"].to(device, torch.bfloat16).contiguous()
            jk = payload["jk"].to(device, torch.bfloat16).contiguous()
            jv = payload["jv"].to(device, torch.bfloat16).contiguous()

            metadata = AttentionMetadata(
                joint_query=jq,
                joint_key=jk,
                joint_value=jv,
                joint_strategy="front",
            )
            with torch.no_grad():
                out = attn(q_local, k_local, v_local, metadata)

            out_cpu = out.detach().cpu().float()
            sp_group = get_sp_group()
            torch.save(
                {
                    "ulysses_world_size": sp_group.ulysses_world_size,
                    "allgather_world_size": sp_group.allgather_world_size,
                    "is_ulysses_allgather": type(attn.parallel_strategy)
                    is UlyssesAllGatherKVParallelAttention,
                    "strategy_name": attn.parallel_strategy.name,
                    "out_shape": tuple(out_cpu.shape),
                    "img_out": out_cpu[:, JOINT_SEQ_LEN:],
                    "joint_out": out_cpu[:, :JOINT_SEQ_LEN],
                },
                Path(work_dir) / f"rank_{rank}.pt",
            )
        finally:
            destroy_distributed_env()


def test_ulysses_allgather_kv_sequence_parallelism(tmp_path: Path) -> None:
    """Ulysses x AllGather-KV Sequence Parallelism: 2x2 orthogonal 2D SP parity.

    - TPU path: spawns 4 workers on tpu_dist with ulysses_degree=2 and
      allgather_degree=2 (Ulysses [0,1]/[2,3], AllGather [0,2]/[1,3]) and runs
      Attention in bf16 on (1, 16, 4, 128) image shards + (1, 8, 4, 128) joint Q/K/V.
    - Setup check: ulysses_world_size == 2, allgather_world_size == 2, and
      parallel_strategy is UlyssesAllGatherKVParallelAttention ("ulysses_allgather_kv").
    - Outcome check: concatenated 4-rank image shards and joint outputs match
      unsharded CPU fp32 attention within max(REL_L2_RATIO * cpu_bf16_err,
      REL_L2_FLOOR).
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= WORLD_SIZE, (
        f"Ulysses x AllGather-KV test requires {WORLD_SIZE} TPU devices, "
        f"found {device_count}"
    )

    gen = torch.Generator().manual_seed(41)
    q = torch.randn(1, IMG_SEQ_LEN, NUM_HEADS, HEAD_DIM, generator=gen)
    k = torch.randn(1, IMG_SEQ_LEN, NUM_HEADS, HEAD_DIM, generator=gen)
    v = torch.randn(1, IMG_SEQ_LEN, NUM_HEADS, HEAD_DIM, generator=gen)
    jq = torch.randn(1, JOINT_SEQ_LEN, NUM_HEADS, HEAD_DIM, generator=gen)
    jk = torch.randn(1, JOINT_SEQ_LEN, NUM_HEADS, HEAD_DIM, generator=gen)
    jv = torch.randn(1, JOINT_SEQ_LEN, NUM_HEADS, HEAD_DIM, generator=gen)

    ref_img_fp32, ref_joint_fp32 = _reference_joint_attention(
        q, k, v, jq, jk, jv, dtype=torch.float32
    )
    ref_img_bf16, ref_joint_bf16 = _reference_joint_attention(
        q, k, v, jq, jk, jv, dtype=torch.bfloat16
    )

    torch.save(
        {"q": q, "k": k, "v": v, "jq": jq, "jk": jk, "jv": jv},
        tmp_path / "payload.pt",
    )
    master_port = get_open_port()
    try:
        torch.multiprocessing.spawn(
            _ulysses_allgather_worker,
            args=(WORLD_SIZE, master_port, str(tmp_path)),
            nprocs=WORLD_SIZE,
        )
    finally:
        wait_for_tpu_release()

    results = [
        torch.load(tmp_path / f"rank_{r}.pt", weights_only=True)
        for r in range(WORLD_SIZE)
    ]
    local_img_len = IMG_SEQ_LEN // WORLD_SIZE
    expected_out_shape = (1, JOINT_SEQ_LEN + local_img_len, NUM_HEADS, HEAD_DIM)
    for r, res in enumerate(results):
        assert res["ulysses_world_size"] == ULYSSES_DEGREE, (
            f"rank {r}: expected ulysses_world_size={ULYSSES_DEGREE}, "
            f"got {res['ulysses_world_size']}"
        )
        assert res["allgather_world_size"] == ALLGATHER_DEGREE, (
            f"rank {r}: expected allgather_world_size={ALLGATHER_DEGREE}, "
            f"got {res['allgather_world_size']}"
        )
        assert res["is_ulysses_allgather"] is True, (
            f"rank {r}: expected UlyssesAllGatherKVParallelAttention strategy"
        )
        assert res["strategy_name"] == "ulysses_allgather_kv", (
            f"rank {r}: expected strategy_name='ulysses_allgather_kv', "
            f"got {res['strategy_name']!r}"
        )
        assert res["out_shape"] == expected_out_shape, (
            f"rank {r}: expected output shape {expected_out_shape}, "
            f"got {res['out_shape']}"
        )

    tpu_img = torch.cat([res["img_out"] for res in results], dim=1)
    _assert_bf16_parity(
        tpu_img, ref_img_fp32, ref_img_bf16, name="Ulysses x AllGather-KV image"
    )
    for r, res in enumerate(results):
        _assert_bf16_parity(
            res["joint_out"],
            ref_joint_fp32,
            ref_joint_bf16,
            name=f"Ulysses x AllGather-KV joint rank {r}",
        )
