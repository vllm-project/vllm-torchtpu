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
"""TPU tests for classifier-free guidance (CFG) parallelism."""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch.multiprocessing import ProcessRaisedException
from torch_tpu._internal.utils import hardware
from vllm.utils.network_utils import get_open_port
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_classifier_free_guidance_world_size,
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

REL_L2_RATIO = 1.5
REL_L2_FLOOR = 1e-2
WORKER_TIMEOUT_S = 120


class _CountingTinyDiT(torch.nn.Module):
    """Tiny conv + self-attention DiT that counts local forward invocations."""

    def __init__(self, in_channels: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        self.in_proj = torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.qkv = torch.nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.final_proj = torch.nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor]:
        self.forward_calls += 1
        bsz, _, height, width = x.shape
        tokens = self.in_proj(x).flatten(2).transpose(1, 2) + cond
        normed = self.norm(tokens)
        q, k, v = self.qkv(normed).chunk(3, dim=-1)
        seq_len = height * width
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = torch.matmul(torch.softmax(scores, dim=-1), v)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)
        out = (tokens + self.out_proj(attn)).transpose(1, 2)
        return (self.final_proj(out.view(bsz, self.hidden_dim, height, width)),)


class _CfgPipeline(CFGParallelMixin):
    """Minimal CFGParallelMixin pipeline supporting 2-branch and 3-branch CFG."""

    def __init__(self, transformer: _CountingTinyDiT) -> None:
        self.transformer = transformer

    def combine_multi_branch_cfg_noise(
        self,
        predictions: list[torch.Tensor | tuple[torch.Tensor, ...]],
        true_cfg_scale: float | dict[str, float],
        cfg_normalize: bool = False,
    ) -> torch.Tensor:
        assert isinstance(true_cfg_scale, dict), (
            f"expected dict true_cfg_scale for multi-branch CFG, "
            f"got {type(true_cfg_scale).__name__}"
        )
        pos, ref, uncond = (
            predictions[0],
            predictions[1],
            predictions[2],
        )
        assert isinstance(pos, torch.Tensor), (
            f"expected pos prediction Tensor, got {type(pos).__name__}"
        )
        assert isinstance(ref, torch.Tensor), (
            f"expected ref prediction Tensor, got {type(ref).__name__}"
        )
        assert isinstance(uncond, torch.Tensor), (
            f"expected uncond prediction Tensor, got {type(uncond).__name__}"
        )
        combined = (
            uncond
            + true_cfg_scale["image"] * (ref - uncond)
            + true_cfg_scale["text"] * (pos - ref)
        )
        if cfg_normalize:
            combined = self.cfg_normalize_function(pos, combined)
        return combined


def _run_two_branch(
    pipeline: _CfgPipeline, branches: list[dict[str, torch.Tensor]]
) -> torch.Tensor:
    out = pipeline.predict_noise_maybe_with_cfg(
        do_true_cfg=True,
        true_cfg_scale=5.0,
        positive_kwargs=branches[0],
        negative_kwargs=branches[1],
        cfg_normalize=True,
    )
    assert isinstance(out, torch.Tensor), (
        f"two-branch CFG expected Tensor output, got {type(out).__name__}"
    )
    return out


def _run_three_branch(
    pipeline: _CfgPipeline, branches: list[dict[str, torch.Tensor]]
) -> torch.Tensor:
    out = pipeline.predict_noise_with_multi_branch_cfg(
        do_true_cfg=True,
        true_cfg_scale={"text": 5.0, "image": 2.0},
        branches_kwargs=branches,
        cfg_normalize=True,
    )
    assert isinstance(out, torch.Tensor), (
        f"three-branch CFG expected Tensor output, got {type(out).__name__}"
    )
    return out


@dataclass(frozen=True)
class CfgCase:
    case_id: str
    cfg_parallel_size: int
    n_branches: int
    expected_calls: tuple[int, ...]
    run_cfg: Callable[[_CfgPipeline, list[dict[str, torch.Tensor]]], torch.Tensor]


CFG_CASES = [
    pytest.param(
        CfgCase("two_branch_cfg2", 2, 2, (1, 1), _run_two_branch),
        id="two_branch_cfg2",
    ),
    pytest.param(
        CfgCase("three_branch_cfg2", 2, 3, (2, 1), _run_three_branch),
        id="three_branch_cfg2",
        marks=pytest.mark.xfail(
            strict=True,
            raises=TimeoutError,
            reason=(
                "rank 0 hangs in the CFG all_gather: 3 branches on 2 ranks run 2 vs 1 "
                "DiT forwards, so each rank compiles a different program around the "
                "collective"
            ),
        ),
    ),
    pytest.param(
        CfgCase("three_branch_cfg3", 3, 3, (1, 1, 1), _run_three_branch),
        id="three_branch_cfg3",
        marks=pytest.mark.xfail(
            strict=True,
            raises=ProcessRaisedException,
            reason=(
                "torch_tpu hardware.get_tpu_topology(3) raises RuntimeError "
                "because 3-device single-host TPU topologies are unsupported"
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


def _cast_branches(
    branches: list[dict[str, torch.Tensor]],
    device: torch.device | str,
    dtype: torch.dtype,
) -> list[dict[str, torch.Tensor]]:
    return [
        {k: v.to(device=device, dtype=dtype) for k, v in branch.items()}
        for branch in branches
    ]


def _cfg_worker(
    rank: int,
    case: CfgCase,
    master_port: int,
    work_dir: str,
) -> None:
    world_size = case.cfg_parallel_size
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
        parallel_config=DiffusionParallelConfig(cfg_parallel_size=world_size),
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
            initialize_model_parallel(cfg_parallel_size=world_size)
            payload = torch.load(Path(work_dir) / "payload.pt", weights_only=True)
            dit = _CountingTinyDiT()
            dit.load_state_dict(payload["state_dict"])
            dit = dit.to(device=device, dtype=torch.bfloat16).eval()

            pipeline = _CfgPipeline(dit)
            tpu_branches = _cast_branches(payload["branches"], device, torch.bfloat16)
            with torch.no_grad():
                pred = case.run_cfg(pipeline, tpu_branches)

            torch.save(
                {
                    "cfg_world_size": get_classifier_free_guidance_world_size(),
                    "forward_calls": dit.forward_calls,
                    "output": pred.detach().cpu().float(),
                },
                Path(work_dir) / f"rank_{rank}.pt",
            )
        finally:
            destroy_distributed_env()


@pytest.mark.parametrize("case", CFG_CASES)
def test_cfg_parallelism(tmp_path: Path, case: CfgCase) -> None:
    """CFG Parallelism: 2-branch and 3-branch CFG dispatch and all-gather on TPU.

    - TPU path: spawns cfg_parallel_size workers on tpu_dist and runs
      CFGParallelMixin (predict_noise_maybe_with_cfg for 2 branches and
      predict_noise_with_multi_branch_cfg for 3 branches) on a bf16 DiT.
    - Setup check: get_classifier_free_guidance_world_size() == cfg_parallel_size
      and per-rank DiT forward counts match the round-robin branch dispatch.
    - Outcome check: all ranks produce bitwise-identical combined predictions
      within max(REL_L2_RATIO * cpu_bf16_err, REL_L2_FLOOR) of sequential CPU fp32.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= case.cfg_parallel_size, (
        f"{case.case_id} requires {case.cfg_parallel_size} TPU devices, "
        f"found {device_count}"
    )

    # CPU generator only: torch.manual_seed() also seeds, and so opens, the TPU.
    torch.default_generator.manual_seed(0)
    dit_fp32 = _CountingTinyDiT().eval()
    dit_bf16 = copy.deepcopy(dit_fp32).to(torch.bfloat16).eval()

    gen = torch.Generator().manual_seed(11)
    shared_latents = torch.randn(2, 4, 8, 8, generator=gen)
    branches = [
        {"x": shared_latents.clone(), "cond": torch.randn(2, 64, 128, generator=gen)}
        for _ in range(case.n_branches)
    ]

    with torch.no_grad():
        ref_fp32 = case.run_cfg(_CfgPipeline(dit_fp32), branches).float()
        ref_bf16 = case.run_cfg(
            _CfgPipeline(dit_bf16), _cast_branches(branches, "cpu", torch.bfloat16)
        ).float()

    torch.save(
        {"state_dict": dit_fp32.state_dict(), "branches": branches},
        tmp_path / "payload.pt",
    )

    master_port = get_open_port()
    workers = torch.multiprocessing.start_processes(
        _cfg_worker,
        args=(case, master_port, str(tmp_path)),
        nprocs=case.cfg_parallel_size,
        join=False,
    )
    deadline = time.monotonic() + WORKER_TIMEOUT_S
    try:
        while not workers.join(timeout=max(deadline - time.monotonic(), 0.0)):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{case.case_id}: CFG workers still running after "
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
        for r in range(case.cfg_parallel_size)
    ]
    for r, res in enumerate(results):
        assert res["cfg_world_size"] == case.cfg_parallel_size, (
            f"{case.case_id} rank {r}: expected cfg_world_size="
            f"{case.cfg_parallel_size}, got {res['cfg_world_size']}"
        )
    actual_calls = tuple(res["forward_calls"] for res in results)
    assert actual_calls == case.expected_calls, (
        f"{case.case_id}: expected per-rank forward calls {case.expected_calls}, "
        f"got {actual_calls}"
    )
    for r in range(1, case.cfg_parallel_size):
        assert torch.equal(results[0]["output"], results[r]["output"]), (
            f"{case.case_id}: rank 0 and rank {r} CFG outputs differed after all_gather"
        )

    _assert_bf16_parity(results[0]["output"], ref_fp32, ref_bf16, name=case.case_id)
