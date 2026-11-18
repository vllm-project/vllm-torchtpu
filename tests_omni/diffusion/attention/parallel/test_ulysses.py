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
"""TPU tests for Ulysses sequence-parallel attention."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch.multiprocessing import ProcessRaisedException
from torch_tpu._internal.utils import hardware
from vllm.utils.network_utils import get_open_port
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    set_forward_context,
)
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
HEAD_DIM = 128
REL_L2_RATIO = 1.5
REL_L2_FLOOR = 1e-2


@dataclass(frozen=True)
class UlyssesCase:
    case_id: str
    ulysses_mode: str
    ulysses_a2a_permute: bool
    mask_sp_padding: bool
    num_heads: int
    num_kv_heads: int
    shard_lens: tuple[int, int]
    valid_seq_len: int
    joint_len: int


JOINT_CASES = [
    pytest.param(
        UlyssesCase(
            case_id="strict",
            ulysses_mode="strict",
            ulysses_a2a_permute=False,
            mask_sp_padding=False,
            num_heads=4,
            num_kv_heads=4,
            shard_lens=(32, 32),
            valid_seq_len=64,
            joint_len=8,
        ),
        id="strict",
    ),
    pytest.param(
        UlyssesCase(
            case_id="advanced_uaa",
            ulysses_mode="advanced_uaa",
            ulysses_a2a_permute=False,
            mask_sp_padding=False,
            num_heads=6,
            num_kv_heads=3,
            shard_lens=(32, 48),
            valid_seq_len=80,
            joint_len=8,
        ),
        id="advanced_uaa",
        marks=pytest.mark.xfail(
            strict=True,
            raises=ProcessRaisedException,
            reason=(
                "worker RuntimeError: UAA's all_to_all_single over uneven shards "
                "has asymmetric input buffer sizes, which torch_tpu rejects"
            ),
        ),
    ),
]
IMAGE_ONLY_CASES = [
    pytest.param(
        UlyssesCase(
            case_id="ulysses_a2a_permute",
            ulysses_mode="strict",
            ulysses_a2a_permute=True,
            mask_sp_padding=False,
            num_heads=4,
            num_kv_heads=4,
            shard_lens=(32, 32),
            valid_seq_len=64,
            joint_len=0,
        ),
        id="ulysses_a2a_permute",
        marks=pytest.mark.xfail(
            strict=True,
            raises=ProcessRaisedException,
            reason="ulysses_a2a_permute requires NVIDIA libnccl.so and CUDA symm_mem",
        ),
    ),
    pytest.param(
        UlyssesCase(
            case_id="mask_sp_padding",
            ulysses_mode="strict",
            ulysses_a2a_permute=False,
            mask_sp_padding=True,
            num_heads=4,
            num_kv_heads=4,
            shard_lens=(32, 32),
            valid_seq_len=61,
            joint_len=0,
        ),
        id="mask_sp_padding",
        marks=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason=(
                "2D SP key padding mask falls back to TpuSDPAImpl broadcast mask "
                "which produces NaNs in bf16 on torch_tpu"
            ),
        ),
    ),
]


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


def _reference_attention(
    q_img: torch.Tensor,
    k_img: torch.Tensor,
    v_img: torch.Tensor,
    joint: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    q, k, v = (t.to(dtype) for t in (q_img, k_img, v_img))
    joint_len = 0
    if joint is not None:
        jq, jk, jv = (t.to(dtype) for t in joint)
        joint_len = int(jq.shape[1])
        q = torch.cat([jq, q], dim=1)
        k = torch.cat([jk, k], dim=1)
        v = torch.cat([jv, v], dim=1)
    gqa = q.shape[2] // k.shape[2]
    if gqa > 1:
        k = k.repeat_interleave(gqa, dim=2)
        v = v.repeat_interleave(gqa, dim=2)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=HEAD_DIM**-0.5,
    ).transpose(1, 2)
    out_f32 = out.float()
    if joint_len > 0:
        return out_f32[:, joint_len:], out_f32[:, :joint_len]
    return out_f32, None


def _ulysses_worker(
    rank: int, case: UlyssesCase, master_port: int, work_dir: str
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(ULYSSES_DEGREE)

    device = current_omni_platform.get_torch_device(rank)
    current_omni_platform.set_device(device)

    od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(),
        dtype=torch.bfloat16,
        parallel_config=DiffusionParallelConfig(
            ulysses_degree=ULYSSES_DEGREE,
            sequence_parallel_size=ULYSSES_DEGREE,
            ulysses_mode=case.ulysses_mode,
            ulysses_a2a_permute=case.ulysses_a2a_permute,
            mask_sp_padding=case.mask_sp_padding,
        ),
    )
    vllm_config = create_diffusion_vllm_config(device, od_config)
    current_omni_platform.init_diffusion_worker_vllm_config(vllm_config)

    with (
        set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config),
        set_current_diffusion_config(od_config),
    ):
        init_distributed_environment(
            world_size=ULYSSES_DEGREE,
            rank=rank,
            distributed_init_method="env://",
            local_rank=rank,
        )
        try:
            initialize_model_parallel(
                ulysses_degree=ULYSSES_DEGREE,
                sequence_parallel_size=ULYSSES_DEGREE,
            )
            attn = Attention(
                num_heads=case.num_heads,
                head_size=HEAD_DIM,
                causal=False,
                softmax_scale=HEAD_DIM**-0.5,
                num_kv_heads=case.num_kv_heads,
            )
            payload = torch.load(Path(work_dir) / "payload.pt", weights_only=True)
            start = sum(case.shard_lens[:rank])
            end = start + case.shard_lens[rank]
            q_local = payload["q"][:, start:end].to(device, torch.bfloat16).contiguous()
            k_local = payload["k"][:, start:end].to(device, torch.bfloat16).contiguous()
            v_local = payload["v"][:, start:end].to(device, torch.bfloat16).contiguous()

            total_padded = sum(case.shard_lens)
            pad_size = total_padded - case.valid_seq_len
            ctx = get_forward_context()
            ctx.sp_original_seq_len = case.valid_seq_len
            ctx.sp_padding_size = pad_size

            attn_mask = None
            if case.mask_sp_padding and pad_size > 0:
                attn_mask = torch.ones(1, total_padded, dtype=torch.bool, device=device)
                attn_mask[:, case.valid_seq_len :] = False

            joint_q = joint_k = joint_v = None
            if case.joint_len > 0:
                joint_q = payload["jq"].to(device, torch.bfloat16).contiguous()
                joint_k = payload["jk"].to(device, torch.bfloat16).contiguous()
                joint_v = payload["jv"].to(device, torch.bfloat16).contiguous()

            metadata = AttentionMetadata(
                attn_mask=attn_mask,
                joint_query=joint_q,
                joint_key=joint_k,
                joint_value=joint_v,
                joint_strategy="front",
            )
            with torch.no_grad():
                out = attn(q_local, k_local, v_local, metadata)

            out_cpu = out.detach().cpu().float()
            joint_out = out_cpu[:, : case.joint_len] if case.joint_len > 0 else None
            img_out = out_cpu[:, case.joint_len :]
            sp_group = get_sp_group()
            torch.save(
                {
                    "ulysses_world_size": sp_group.ulysses_world_size,
                    "is_ulysses": type(attn.parallel_strategy)
                    is UlyssesParallelAttention,
                    "local_img_shape": tuple(img_out.shape),
                    "img_out": img_out,
                    "joint_out": joint_out,
                },
                Path(work_dir) / f"rank_{rank}.pt",
            )
        finally:
            destroy_distributed_env()


def _valid_qkv(
    case: UlyssesCase, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(1, case.valid_seq_len, case.num_heads, HEAD_DIM, generator=gen)
    k = torch.randn(1, case.valid_seq_len, case.num_kv_heads, HEAD_DIM, generator=gen)
    v = torch.randn(1, case.valid_seq_len, case.num_kv_heads, HEAD_DIM, generator=gen)
    return q, k, v


def _padded_payload(
    case: UlyssesCase, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Pad the image Q/K/V to sum(shard_lens) tokens for the workers."""
    pad = (0, 0, 0, 0, 0, sum(case.shard_lens) - case.valid_seq_len)
    return {"q": F.pad(q, pad), "k": F.pad(k, pad), "v": F.pad(v, pad)}


def _run_ulysses_workers(
    case: UlyssesCase, payload: dict[str, torch.Tensor], work_dir: Path
) -> list[dict[str, Any]]:
    torch.save(payload, work_dir / "payload.pt")
    master_port = get_open_port()
    try:
        torch.multiprocessing.spawn(
            _ulysses_worker,
            args=(case, master_port, str(work_dir)),
            nprocs=ULYSSES_DEGREE,
        )
    finally:
        wait_for_tpu_release()
    return [
        torch.load(work_dir / f"rank_{r}.pt", weights_only=True)
        for r in range(ULYSSES_DEGREE)
    ]


def _assert_rank_results(case: UlyssesCase, results: list[dict[str, Any]]) -> None:
    for r, res in enumerate(results):
        expected_shape = (1, case.shard_lens[r], case.num_heads, HEAD_DIM)
        assert res["ulysses_world_size"] == ULYSSES_DEGREE, (
            f"{case.case_id} rank {r}: expected ulysses_world_size={ULYSSES_DEGREE}, "
            f"got {res['ulysses_world_size']}"
        )
        assert res["is_ulysses"] is True, (
            f"{case.case_id} rank {r}: expected UlyssesParallelAttention strategy"
        )
        assert res["local_img_shape"] == expected_shape, (
            f"{case.case_id} rank {r}: expected shard shape {expected_shape}, "
            f"got {res['local_img_shape']}"
        )


@pytest.mark.parametrize("case", JOINT_CASES)
def test_ulysses_sequence_parallelism(tmp_path: Path, case: UlyssesCase) -> None:
    """Ulysses Sequence Parallelism: joint attention, strict and advanced_uaa.

    - TPU path: spawns 2 workers on tpu_dist and runs vllm_omni Attention with
      UlyssesParallelAttention on image shards plus front joint tokens, in strict
      mode and advanced_uaa (uneven shards + non-divisible 6Q/3KV GQA heads).
    - Setup check: get_sp_group().ulysses_world_size == 2, parallel_strategy is
      UlyssesParallelAttention, and each rank produces its local shard shape.
    - Outcome check: concatenated unpadded image shards and joint tokens match
      unsharded CPU fp32 attention within max(REL_L2_RATIO * cpu_bf16_err,
      REL_L2_FLOOR); both ranks hold the same joint output.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= ULYSSES_DEGREE, (
        f"Ulysses test requires {ULYSSES_DEGREE} TPU devices, found {device_count}"
    )

    gen = torch.Generator().manual_seed(23)
    q_valid, k_valid, v_valid = _valid_qkv(case, gen)
    jq = torch.randn(1, case.joint_len, case.num_heads, HEAD_DIM, generator=gen)
    jk = torch.randn(1, case.joint_len, case.num_kv_heads, HEAD_DIM, generator=gen)
    jv = torch.randn(1, case.joint_len, case.num_kv_heads, HEAD_DIM, generator=gen)
    ref_img_fp32, ref_joint_fp32 = _reference_attention(
        q_valid, k_valid, v_valid, (jq, jk, jv), dtype=torch.float32
    )
    ref_img_bf16, ref_joint_bf16 = _reference_attention(
        q_valid, k_valid, v_valid, (jq, jk, jv), dtype=torch.bfloat16
    )

    payload = _padded_payload(case, q_valid, k_valid, v_valid)
    payload.update({"jq": jq, "jk": jk, "jv": jv})
    results = _run_ulysses_workers(case, payload, tmp_path)

    _assert_rank_results(case, results)
    tpu_img = torch.cat([res["img_out"] for res in results], dim=1)[
        :, : case.valid_seq_len
    ]
    _assert_bf16_parity(
        tpu_img, ref_img_fp32, ref_img_bf16, name=f"{case.case_id} image"
    )
    assert torch.equal(results[0]["joint_out"], results[1]["joint_out"]), (
        f"{case.case_id}: rank-0 and rank-1 joint outputs differed after all_gather"
    )
    _assert_bf16_parity(
        results[0]["joint_out"],
        ref_joint_fp32,
        ref_joint_bf16,
        name=f"{case.case_id} joint",
    )


@pytest.mark.parametrize("case", IMAGE_ONLY_CASES)
def test_ulysses_sequence_parallelism_image_only(
    tmp_path: Path, case: UlyssesCase
) -> None:
    """Ulysses Sequence Parallelism: image-only, a2a_permute and pad mask.

    - TPU path: spawns 2 workers on tpu_dist and runs vllm_omni Attention with
      UlyssesParallelAttention on image shards without joint tokens, with
      ulysses_a2a_permute and with mask_sp_padding.
    - Setup check: get_sp_group().ulysses_world_size == 2, parallel_strategy is
      UlyssesParallelAttention, and each rank produces its local shard shape.
    - Outcome check: concatenated unpadded image shards match unsharded CPU fp32
      attention within max(REL_L2_RATIO * cpu_bf16_err, REL_L2_FLOOR).
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= ULYSSES_DEGREE, (
        f"Ulysses test requires {ULYSSES_DEGREE} TPU devices, found {device_count}"
    )

    gen = torch.Generator().manual_seed(23)
    q_valid, k_valid, v_valid = _valid_qkv(case, gen)
    ref_img_fp32, _ = _reference_attention(
        q_valid, k_valid, v_valid, None, dtype=torch.float32
    )
    ref_img_bf16, _ = _reference_attention(
        q_valid, k_valid, v_valid, None, dtype=torch.bfloat16
    )

    payload = _padded_payload(case, q_valid, k_valid, v_valid)
    results = _run_ulysses_workers(case, payload, tmp_path)

    _assert_rank_results(case, results)
    tpu_img = torch.cat([res["img_out"] for res in results], dim=1)[
        :, : case.valid_seq_len
    ]
    _assert_bf16_parity(
        tpu_img, ref_img_fp32, ref_img_bf16, name=f"{case.case_id} image"
    )
