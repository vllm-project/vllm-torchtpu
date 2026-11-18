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
"""Multi-rank TPU tests for diffusion Hybrid Sharded Data Parallel (HSDP)."""

from __future__ import annotations

import os
from typing import Any

import portpicker
import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import DTensor
from torch_tpu._internal.utils import hardware
from vllm_omni.diffusion.distributed.hsdp import (
    HSDPInferenceConfig,
    apply_hsdp_to_model,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_fs_group,
    get_hsdp_replicate_group,
    init_distributed_environment,
    initialize_model_parallel,
)

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

HIDDEN_DIM = 128
INTERMEDIATE_DIM = 256
NUM_BLOCKS = 2
BATCH_SIZE = 2
SEQ_LEN = 16
HSDP_SEED = 53
BF16_ERR_RATIO = 2.0
HSDP_REL_L2_FLOOR = 1.5e-2
HSDP_ABS_FLOOR = 2.5e-2


class _TinyDiTBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM, bias=False)
        self.proj_out = nn.Linear(INTERMEDIATE_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(torch.nn.functional.silu(self.proj_in(x)))


class _TinyHSDPDiT(nn.Module):
    """Small multi-block DiT module exercising HSDP block sharding and ignore rules."""

    @staticmethod
    def _is_block(name: str, _module: nn.Module) -> bool:
        return name.startswith("blocks.") and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_block]
    _hsdp_ignored_modules = ["time_embed"]

    def __init__(self) -> None:
        super().__init__()
        self.time_embed = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.blocks = nn.ModuleList([_TinyDiTBlock() for _ in range(NUM_BLOCKS)])
        self.out_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = x + self.time_embed(t_emb)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


def _build_seeded_dit(dtype: torch.dtype) -> _TinyHSDPDiT:
    cpu_rng = torch.get_rng_state()
    try:
        # CPU generator only: torch.manual_seed() also seeds, and so opens, the TPU.
        torch.default_generator.manual_seed(HSDP_SEED)
        model = _TinyHSDPDiT()
        with torch.no_grad():
            for param in model.parameters():
                param.mul_(0.25)
    finally:
        torch.set_rng_state(cpu_rng)
    return model.eval().to(dtype=dtype)


def _run_cpu_reference(
    x_cpu: torch.Tensor, t_cpu: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    model = _build_seeded_dit(dtype)
    with torch.no_grad():
        out = model(x_cpu.to(dtype=dtype), t_cpu.to(dtype=dtype))
    return out.float()


def _error_metrics(
    actual: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float]:
    diff = (actual.double() - reference.double()).flatten()
    ref_flat = reference.double().flatten()
    return float(diff.norm() / ref_flat.norm()), float(diff.abs().max())


def _hsdp_worker(
    rank: int,
    world_size: int,
    replicate_size: int,
    shard_size: int,
    master_port: int,
    x_cpu: torch.Tensor,
    t_cpu: torch.Tensor,
    result_queue: Any,
) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        OmniTpuPlatform.set_device(0)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{master_port}",
            local_rank=rank,
        )
        initialize_model_parallel(use_hsdp=True, fully_shard_degree=shard_size)

        target_device = torch.device("tpu")
        model = _build_seeded_dit(torch.bfloat16)
        apply_hsdp_to_model(
            model,
            HSDPInferenceConfig(
                enabled=True,
                hsdp_replicate_size=replicate_size,
                hsdp_shard_size=shard_size,
                param_dtype=torch.bfloat16,
                reshard_after_forward=True,
            ),
            target_device=target_device,
        )

        block_param = model.blocks[0].proj_in.weight
        ignored_param = model.time_embed.weight
        rep_world_size = (
            get_hsdp_replicate_group().world_size if replicate_size > 1 else 1
        )
        local_shape = (
            tuple(block_param._local_tensor.shape)
            if isinstance(block_param, DTensor)
            else tuple(block_param.shape)
        )
        local_dev = (
            block_param._local_tensor.device.type
            if isinstance(block_param, DTensor)
            else block_param.device.type
        )
        setup_info = (
            get_fs_group().world_size,
            rep_world_size,
            isinstance(block_param, DTensor),
            local_shape,
            local_dev,
            isinstance(ignored_param, DTensor),
            ignored_param.device.type,
        )

        x_tpu = x_cpu.to(device=target_device, dtype=torch.bfloat16)
        t_tpu = t_cpu.to(device=target_device, dtype=torch.bfloat16)
        with torch.no_grad():
            out1 = model(x_tpu, t_tpu)
            out2 = model(x_tpu, t_tpu)

        torch.accelerator.synchronize()
        keep_outputs = rank % shard_size == 0
        # numpy pickles by value. A queued CPU tensor sends a storage fd that the
        # parent can only fetch while this process is alive.
        outputs = (
            (out1.detach().cpu().float().numpy(), out2.detach().cpu().float().numpy())
            if keep_outputs
            else None
        )
        result_queue.put(("ok", rank, setup_info, outputs))
    except Exception as exc:
        result_queue.put(("error", rank, exc, None))
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize(
    "replicate_size,shard_size",
    [
        (1, 2),
        (2, 2),
    ],
    ids=["rep1_shard2", "rep2_shard2"],
)
@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "FSDP2 fully_shard moves the CPU-resident params (HSDP defers weight "
        "placement to it) to the TPU with `param.data = param.to(device)`; "
        "Tensor.set_data rejects the CPU-to-TPU change (incompatible tensor type)."
    ),
)
def test_hybrid_sharded_data_parallel(replicate_size: int, shard_size: int) -> None:
    """Hybrid Sharded Data Parallel (HSDP): FSDP2 DiT sharding on TPU.

    - TPU path: apply ``apply_hsdp_to_model`` to a multi-block DiT across spawned
      TPU ranks under ``(replicate=1, shard=2)`` and ``(replicate=2, shard=2)``.
    - Setup check: assert ``get_fs_group().world_size == shard_size``, ``DTensor``
      local shard shapes on sharded blocks, and unsharded ``tpu`` placement on
      ignored modules.
    - Outcome check: run two ``torch.no_grad()`` forwards for reshard replay,
      check replica parity when ``replicate_size > 1``, and assert parity vs CPU fp32.
    """
    world_size = replicate_size * shard_size
    case_name = f"hsdp_rep{replicate_size}_shard{shard_size}"
    device_count = hardware.get_tpu_device_count()
    assert device_count >= world_size, (
        f"{case_name}: requires {world_size} TPU devices, found {device_count}"
    )

    gen = torch.Generator(device="cpu").manual_seed(61)
    x_cpu = torch.randn(
        (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), generator=gen, dtype=torch.float32
    )
    t_cpu = torch.randn((BATCH_SIZE, 1, HIDDEN_DIM), generator=gen, dtype=torch.float32)

    ref_fp32 = _run_cpu_reference(x_cpu, t_cpu, torch.float32)
    ref_bf16 = _run_cpu_reference(x_cpu, t_cpu, torch.bfloat16)
    cpu_bf16_rel_l2, cpu_bf16_max_abs = _error_metrics(ref_bf16, ref_fp32)

    master_port = portpicker.pick_unused_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.SimpleQueue()
    procs = [
        ctx.Process(
            target=_hsdp_worker,
            args=(
                rank,
                world_size,
                replicate_size,
                shard_size,
                master_port,
                x_cpu,
                t_cpu,
                result_queue,
            ),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    # Read before join: put() blocks until a payload larger than the pipe is read.
    records = [result_queue.get() for _ in range(world_size)]
    for proc in procs:
        proc.join()
    wait_for_tpu_release()

    for status, _rank, payload, _out in records:
        if status == "error":
            raise payload
    records.sort(key=lambda item: item[1])

    expected_setup = (
        shard_size,
        replicate_size,
        True,
        (INTERMEDIATE_DIM // shard_size, HIDDEN_DIM),
        "tpu",
        False,
        "tpu",
    )
    for _, rank, setup_info, _ in records:
        assert setup_info == expected_setup, (
            f"{case_name} rank {rank}: setup {setup_info} != {expected_setup}"
        )

    rank0_pair = records[0][3]
    assert rank0_pair is not None, f"{case_name}: rank 0 outputs missing"
    rank0_out1, rank0_out2 = map(torch.from_numpy, rank0_pair)
    assert_parity(
        rank0_out2,
        rank0_out1,
        name=f"{case_name}_reshard_replay",
        max_rel_l2=0.0,
        max_abs=0.0,
    )

    # Each later replica's leader rank must reproduce replica 0 exactly.
    for _, rank, _, rep_pair in records[shard_size::shard_size]:
        assert rep_pair is not None, f"{case_name}: rank {rank} outputs missing"
        assert_parity(
            torch.from_numpy(rep_pair[0]),
            rank0_out1,
            name=f"{case_name}_replica_match_rank{rank}",
            max_rel_l2=0.0,
            max_abs=0.0,
        )

    assert_parity(
        rank0_out2,
        ref_fp32,
        name=case_name,
        max_rel_l2=max(BF16_ERR_RATIO * cpu_bf16_rel_l2, HSDP_REL_L2_FLOOR),
        max_abs=max(BF16_ERR_RATIO * cpu_bf16_max_abs, HSDP_ABS_FLOOR),
    )
