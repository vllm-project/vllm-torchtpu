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
"""Multi-rank TPU tests for distributed VAE parallel decode and encode."""

from __future__ import annotations

import contextlib
import os
import queue
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import portpicker
import pytest
import torch
import torch.multiprocessing as mp
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from torch_tpu._internal.utils import hardware
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import (
    DistributedAutoencoderKL,
)
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
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

VAE_WORLD_SIZE = 2
VAE_SEED = 17
BF16_ERR_RATIO = 2.0
VAE_REL_L2_FLOOR = 2.5e-2
VAE_ABS_FLOOR = 5.0e-2
WORKER_TIMEOUT_S = 120

SPATIAL_SHARD_P2P_UNSUPPORTED = (
    "Wan spatial-shard VAE decode exchanges halos with "
    "torch.distributed.batch_isend_irecv, and torch_tpu fails the send: "
    "PjRtBuffer::CopyToRemoteDevice INVALID_ARGUMENT (remote and local sizes "
    "differ)."
)
MAX_ALL_REDUCE_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "DistributedVaeExecutor._sync_errors all_reduces an int64 status with "
        "ReduceOp.MAX, which XLA fails to compile on TPU (only Sum all-reduce lowers)"
    ),
)


@dataclass(frozen=True)
class _VaeParallelCase:
    case_id: str
    op_kind: str
    parallel_mode: str
    input_shape: tuple[int, ...]
    expected_tile_distributed: bool
    expected_rank1_broadcast: bool


@contextlib.contextmanager
def _cpu_seed(seed: int) -> Iterator[None]:
    """Temporarily seed the CPU RNG; ``torch.manual_seed`` would also open the TPU."""
    rng_state = torch.get_rng_state()
    try:
        torch.default_generator.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(rng_state)


def _build_wan_vae(dtype: torch.dtype, *, distributed: bool = True) -> AutoencoderKLWan:
    cls = DistributedAutoencoderKLWan if distributed else AutoencoderKLWan
    with _cpu_seed(VAE_SEED):
        vae = cls(
            base_dim=32,
            z_dim=16,
            dim_mult=[1, 2, 2, 2],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )
    vae.enable_tiling(
        tile_sample_min_height=64,
        tile_sample_min_width=64,
        tile_sample_stride_height=48,
        tile_sample_stride_width=48,
    )
    return vae.eval().to(dtype=dtype)


def _build_kl_vae(dtype: torch.dtype, *, distributed: bool = True) -> AutoencoderKL:
    cls = DistributedAutoencoderKL if distributed else AutoencoderKL
    with _cpu_seed(VAE_SEED):
        vae = cls(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(32, 64),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=16,
        )
    return vae.eval().to(dtype=dtype)


def _run_vae_single(
    op_kind: str, x_cpu: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    inp = x_cpu.to(dtype=dtype)
    with torch.no_grad():
        if op_kind == "wan_decode":
            vae = _build_wan_vae(dtype, distributed=False)
            return vae.tiled_decode(inp, return_dict=False)[0].float()
        if op_kind == "wan_encode":
            vae = _build_wan_vae(dtype, distributed=False)
            return vae.encode(inp).latent_dist.mean.float()
        vae_kl = _build_kl_vae(dtype, distributed=False)
        return vae_kl.decode(inp, return_dict=False)[0].float()


def _cpu_ref_worker(op_kind: str, x_cpu: torch.Tensor, result_queue: Any) -> None:
    """Compute CPU fp32 and bf16 references in a child so PID 11 never loads PJRT."""
    os.environ["TPU_VISIBLE_DEVICES"] = ""
    try:
        ref_fp32 = _run_vae_single(op_kind, x_cpu, torch.float32)
        ref_bf16 = _run_vae_single(op_kind, x_cpu, torch.bfloat16)
        # numpy pickles by value. A queued CPU tensor sends a storage fd that the
        # parent can only fetch while this process is alive.
        result_queue.put(("ok", (ref_fp32.numpy(), ref_bf16.numpy())))
    except Exception as exc:
        result_queue.put(("error", exc))


def _error_metrics(
    actual: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float]:
    diff = (actual.double() - reference.double()).flatten()
    ref_flat = reference.double().flatten()
    return float(diff.norm() / ref_flat.norm()), float(diff.abs().max())


def _vae_worker(
    rank: int,
    master_port: int,
    case: _VaeParallelCase,
    x_cpu: torch.Tensor,
    result_queue: Any,
) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(VAE_WORLD_SIZE)

        OmniTpuPlatform.set_device(0)
        init_distributed_environment(
            world_size=VAE_WORLD_SIZE,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{master_port}",
            local_rank=rank,
        )
        initialize_model_parallel(ulysses_degree=VAE_WORLD_SIZE)

        device = torch.device("tpu")
        if case.op_kind in ("wan_decode", "wan_encode"):
            vae: DistributedAutoencoderKLWan | DistributedAutoencoderKL = (
                _build_wan_vae(torch.bfloat16).to(device)
            )
        else:
            vae = _build_kl_vae(torch.bfloat16).to(device)
        vae.init_distributed()
        vae.set_parallel_size(VAE_WORLD_SIZE, mode=case.parallel_mode)

        setup_info = (
            vae.distributed_executor.parallel_size,
            vae.distributed_executor.parallel_mode,
            vae.is_distributed_enabled(),
        )

        x_tpu = x_cpu.to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            if case.op_kind == "wan_encode":
                out = vae.encode(x_tpu).latent_dist.mean
            else:
                out = vae.decode(x_tpu, return_dict=False)[0]

        torch.accelerator.synchronize()
        out_shape = tuple(out.shape)
        out_cpu = out.detach().cpu().float().numpy() if rank == 0 else None
        result_queue.put(("ok", rank, setup_info, (out_shape, out_cpu)))
    except Exception as exc:
        result_queue.put(("error", rank, exc, None))
    finally:
        destroy_distributed_env()


CASES = (
    pytest.param(
        _VaeParallelCase(
            case_id="wan_tile_decode",
            op_kind="wan_decode",
            parallel_mode="tile",
            input_shape=(1, 16, 2, 14, 14),
            expected_tile_distributed=True,
            expected_rank1_broadcast=False,
        ),
        id="wan_tile_decode",
        marks=MAX_ALL_REDUCE_XFAIL,
    ),
    pytest.param(
        _VaeParallelCase(
            case_id="wan_tile_encode",
            op_kind="wan_encode",
            parallel_mode="tile",
            input_shape=(1, 3, 5, 112, 112),
            expected_tile_distributed=True,
            expected_rank1_broadcast=True,
        ),
        id="wan_tile_encode",
        marks=MAX_ALL_REDUCE_XFAIL,
    ),
    pytest.param(
        _VaeParallelCase(
            case_id="kl_batch_decode",
            op_kind="kl_decode",
            parallel_mode="batch",
            input_shape=(2, 4, 16, 16),
            expected_tile_distributed=False,
            expected_rank1_broadcast=True,
        ),
        id="kl_batch_decode",
        marks=MAX_ALL_REDUCE_XFAIL,
    ),
    pytest.param(
        _VaeParallelCase(
            case_id="wan_spatial_shard_width",
            op_kind="wan_decode",
            parallel_mode="spatial_shard_width",
            input_shape=(1, 16, 2, 14, 14),
            expected_tile_distributed=True,
            expected_rank1_broadcast=False,
        ),
        id="wan_spatial_shard_width",
        marks=pytest.mark.xfail(
            strict=True,
            raises=RuntimeError,
            reason=SPATIAL_SHARD_P2P_UNSUPPORTED,
        ),
    ),
)


@pytest.mark.parametrize("case", CASES)
def test_vae_parallel_decode_encode(case: _VaeParallelCase) -> None:
    """VAE parallel decode / encode: multi-rank tile and batch execution on TPU.

    - TPU path: run ``DistributedAutoencoderKLWan`` (``tile`` decode and encode)
      and ``DistributedAutoencoderKL`` (``batch`` decode) across 2 spawned TPU ranks.
    - Setup check: assert ``parallel_size == 2``, ``parallel_mode``, and
      ``is_distributed_enabled()`` on both ranks, plus expected rank-1 broadcast shape.
    - Outcome check: compare rank-0 TPU bf16 output against CPU fp32 reference via
      ``assert_parity`` scaled by CPU bf16 error.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= VAE_WORLD_SIZE, (
        f"Test requires {VAE_WORLD_SIZE} TPU devices, found {device_count}"
    )

    gen = torch.Generator(device="cpu").manual_seed(29)
    x_cpu = torch.randn(case.input_shape, generator=gen, dtype=torch.float32) * 0.5

    ctx = mp.get_context("spawn")
    ref_queue = ctx.SimpleQueue()
    ref_proc = ctx.Process(
        target=_cpu_ref_worker, args=(case.op_kind, x_cpu, ref_queue)
    )
    ref_proc.start()
    ref_status, ref_payload = ref_queue.get()
    ref_proc.join()
    if ref_status == "error":
        raise ref_payload
    ref_fp32, ref_bf16 = map(torch.from_numpy, ref_payload)
    cpu_bf16_rel_l2, cpu_bf16_max_abs = _error_metrics(ref_bf16, ref_fp32)

    master_port = portpicker.pick_unused_port()
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_vae_worker,
            args=(rank, master_port, case, x_cpu, result_queue),
        )
        for rank in range(VAE_WORLD_SIZE)
    ]
    for proc in procs:
        proc.start()
    deadline = time.monotonic() + WORKER_TIMEOUT_S
    try:
        # Read before join: a worker cannot exit until its queued payload is read.
        records = [
            result_queue.get(timeout=max(deadline - time.monotonic(), 0.0))
            for _ in range(VAE_WORLD_SIZE)
        ]
        for proc in procs:
            proc.join(timeout=max(deadline - time.monotonic(), 0.0))
    except queue.Empty:
        raise TimeoutError(
            f"{case.case_id}: VAE workers sent no result within {WORKER_TIMEOUT_S}s"
        ) from None
    finally:
        # A hung rank keeps its chip, so kill every worker before the release wait.
        for proc in procs:
            proc.kill()
            proc.join()
        wait_for_tpu_release()

    for status, _rank, payload, _out in records:
        if status == "error":
            raise payload
    records.sort(key=lambda item: item[1])

    expected_setup = (
        VAE_WORLD_SIZE,
        case.parallel_mode,
        case.expected_tile_distributed,
    )
    for _, rank, setup_info, _ in records:
        assert setup_info == expected_setup, (
            f"{case.case_id} rank {rank}: setup {setup_info} != {expected_setup}"
        )

    rank0_shape, rank0_out = records[0][3]
    rank1_shape, _ = records[1][3]
    expected_rank1_shape = rank0_shape if case.expected_rank1_broadcast else (0,)
    assert rank1_shape == expected_rank1_shape, (
        f"{case.case_id}: rank 1 output shape {rank1_shape} != {expected_rank1_shape}"
    )
    assert rank0_out is not None, f"{case.case_id}: rank 0 output missing"

    assert_parity(
        torch.from_numpy(rank0_out),
        ref_fp32,
        name=case.case_id,
        max_rel_l2=max(BF16_ERR_RATIO * cpu_bf16_rel_l2, VAE_REL_L2_FLOOR),
        max_abs=max(BF16_ERR_RATIO * cpu_bf16_max_abs, VAE_ABS_FLOOR),
    )
