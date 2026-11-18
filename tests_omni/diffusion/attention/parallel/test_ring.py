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
"""TPU tests for Ring Attention sequence parallelism."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch_tpu._internal.utils import hardware
from vllm.utils.network_utils import get_open_port
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
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

RING_DEGREE = 2
SEQ_LEN = 64
NUM_HEADS = 4
HEAD_DIM = 128
REL_L2_RATIO = 1.5
REL_L2_FLOOR = 1e-2
WORKER_TIMEOUT_S = 120
RING_HANG_REASON = (
    "Ring attention hangs on TPU: RingComm batch_isend_irecv never completes in "
    "torch_tpu ProcessGroupTpu.experimental_send"
)


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


def _ring_worker(rank: int, world_size: int, master_port: int, work_dir: str) -> None:
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
            ring_degree=world_size,
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
                ring_degree=world_size,
                sequence_parallel_size=world_size,
            )
            attn = Attention(
                num_heads=NUM_HEADS,
                head_size=HEAD_DIM,
                causal=False,
                softmax_scale=HEAD_DIM**-0.5,
            )
            payload = torch.load(Path(work_dir) / "payload.pt", weights_only=True)
            local_len = SEQ_LEN // world_size
            start = rank * local_len
            end = start + local_len
            q_local = payload["q"][:, start:end].to(device, torch.bfloat16).contiguous()
            k_local = payload["k"][:, start:end].to(device, torch.bfloat16).contiguous()
            v_local = payload["v"][:, start:end].to(device, torch.bfloat16).contiguous()

            with torch.no_grad():
                out = attn(q_local, k_local, v_local)

            sp_group = get_sp_group()
            torch.save(
                {
                    "ring_world_size": sp_group.ring_world_size,
                    "use_ring": attn.use_ring,
                    "is_ring_strategy": type(attn.parallel_strategy)
                    is RingParallelAttention,
                    "output": out.detach().cpu().float(),
                },
                Path(work_dir) / f"rank_{rank}.pt",
            )
        finally:
            destroy_distributed_env()


@pytest.mark.xfail(strict=True, raises=TimeoutError, reason=RING_HANG_REASON)
def test_ring_attention_sequence_parallelism(tmp_path: Path) -> None:
    """Ring Attention Sequence Parallelism: ring_degree=2 attention parity.

    - TPU path: spawns 2 workers on tpu_dist with ring_degree=2 and runs Attention
      (RingParallelAttention + ring_pytorch_attn_func) on (1, 32, 4, 128) bf16 shards.
    - Setup check: get_sp_group().ring_world_size == 2, attn.use_ring is True, and
      attn.parallel_strategy is RingParallelAttention.
    - Outcome check: concatenated sequence shards match unsharded CPU fp32 SDPA
      within max(REL_L2_RATIO * cpu_bf16_err, REL_L2_FLOOR).
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= RING_DEGREE, (
        f"Ring SP test requires {RING_DEGREE} TPU devices, found {device_count}"
    )

    gen = torch.Generator().manual_seed(31)
    q = torch.randn(1, SEQ_LEN, NUM_HEADS, HEAD_DIM, generator=gen)
    k = torch.randn(1, SEQ_LEN, NUM_HEADS, HEAD_DIM, generator=gen)
    v = torch.randn(1, SEQ_LEN, NUM_HEADS, HEAD_DIM, generator=gen)

    ref_fp32 = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=HEAD_DIM**-0.5,
    ).transpose(1, 2)
    ref_bf16 = (
        F.scaled_dot_product_attention(
            q.to(torch.bfloat16).transpose(1, 2),
            k.to(torch.bfloat16).transpose(1, 2),
            v.to(torch.bfloat16).transpose(1, 2),
            scale=HEAD_DIM**-0.5,
        )
        .transpose(1, 2)
        .float()
    )

    torch.save({"q": q, "k": k, "v": v}, tmp_path / "payload.pt")
    master_port = get_open_port()
    workers = torch.multiprocessing.start_processes(
        _ring_worker,
        args=(RING_DEGREE, master_port, str(tmp_path)),
        nprocs=RING_DEGREE,
        join=False,
    )
    deadline = time.monotonic() + WORKER_TIMEOUT_S
    try:
        while not workers.join(timeout=max(deadline - time.monotonic(), 0.0)):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Ring SP={RING_DEGREE}: ring workers still running after "
                    f"{WORKER_TIMEOUT_S}s"
                )
    finally:
        # A hung rank keeps its chip, and ranks that finished wait for it in
        # libtpu's stop-session barrier, so kill them all before the release wait.
        for proc in workers.processes:
            proc.kill()
            proc.join()
        wait_for_tpu_release()

    results = [
        torch.load(tmp_path / f"rank_{r}.pt", weights_only=True)
        for r in range(RING_DEGREE)
    ]
    for r, res in enumerate(results):
        assert res["ring_world_size"] == RING_DEGREE, (
            f"rank {r}: expected ring_world_size={RING_DEGREE}, "
            f"got {res['ring_world_size']}"
        )
        assert res["use_ring"] is True, f"rank {r}: expected attn.use_ring is True"
        assert res["is_ring_strategy"] is True, (
            f"rank {r}: expected RingParallelAttention strategy"
        )

    tpu_out = torch.cat([res["output"] for res in results], dim=1)
    _assert_bf16_parity(tpu_out, ref_fp32, ref_bf16, name="Ring SP=2")
