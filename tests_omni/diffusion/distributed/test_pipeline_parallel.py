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
"""Multi-rank TPU tests for diffusion pipeline parallelism."""

from __future__ import annotations

import os
import queue
import time
from typing import Any

import portpicker
import pytest
import torch
import torch.multiprocessing as mp
import vllm.distributed.parallel_state as vllm_parallel_state
from torch_tpu._internal.utils import hardware
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.sequence import IntermediateTensors
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_pp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.pipeline_parallel import PipelineParallelMixin

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

PP_WORLD_SIZE = 2
NUM_LAYERS = 4
HIDDEN_DIM = 64
BATCH_SIZE = 2
SEQ_LEN = 16
BASE_SEED = 100
CFG_SCALE = 4.0
BF16_ERR_RATIO = 2.0
PP_REL_L2_FLOOR = 1.5e-2
PP_ABS_FLOOR = 2.5e-2
WORKER_TIMEOUT_S = 120

PP_P2P_UNSUPPORTED = (
    "Diffusion pipeline parallelism stage hand-off calls "
    "torch.distributed.isend/irecv (group_coordinator.py), which hangs for "
    "1800 s and segfaults in torch_tpu ProcessGroupTpu.experimental_send."
)


class _StepScheduler:
    def step(
        self,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        latents: torch.Tensor,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        del t, return_dict
        return (latents - 0.1 * noise_pred,)


class _ToyPPPipeline(PipelineParallelMixin, CFGParallelMixin):
    """4-layer Linear DiT stand-in partitioned via vLLM ``make_layers``."""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.transformer = torch.nn.Module()
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers,
            lambda prefix: torch.nn.Linear(dim, dim, bias=False),
            prefix="layers",
        )
        for idx, layer in enumerate(self.layers):
            if not isinstance(layer, PPMissingLayer):
                gen = torch.Generator(device="cpu").manual_seed(seed + idx)
                weight = torch.randn((dim, dim), generator=gen, dtype=torch.float32)
                layer.weight.data.copy_(weight * 0.1)
        self.layers.to(device=device, dtype=dtype)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], dim
        )
        self.scheduler = _StepScheduler()

    def predict_noise(
        self,
        x: torch.Tensor | None = None,
        cond_bias: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        **_kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            x = intermediate_tensors["hidden_states"]
        assert x is not None, "PipelineParallelMixin.predict_noise received x=None"
        assert cond_bias is not None, (
            "PipelineParallelMixin.predict_noise received cond_bias=None"
        )
        if get_pp_group().is_first_rank:
            x = x + cond_bias
        for idx in range(self.start_layer, self.end_layer):
            x = torch.tanh(self.layers[idx](x))
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": x})
        return x


def _cpu_reference_denoise(
    latents_init: torch.Tensor,
    pos_bias: torch.Tensor,
    neg_bias: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = []
    for idx in range(NUM_LAYERS):
        gen = torch.Generator(device="cpu").manual_seed(BASE_SEED + idx)
        w = torch.randn((HIDDEN_DIM, HIDDEN_DIM), generator=gen, dtype=torch.float32)
        weights.append((w * 0.1).to(dtype=dtype))

    latents = latents_init.to(dtype=dtype)
    pos_b = pos_bias.to(dtype=dtype)
    neg_b = neg_bias.to(dtype=dtype)
    for _ in range(2):
        pos = latents + pos_b
        neg = latents + neg_b
        for w in weights:
            pos = torch.tanh(pos @ w.T)
            neg = torch.tanh(neg @ w.T)
        combined = neg + CFG_SCALE * (pos - neg)
        pos_norm = torch.norm(pos, dim=-1, keepdim=True)
        comb_norm = torch.norm(combined, dim=-1, keepdim=True)
        noise_pred = combined * (pos_norm / (comb_norm + 1e-8))
        latents = latents - 0.1 * noise_pred
    return latents.float()


def _error_metrics(
    actual: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float]:
    diff = (actual.double() - reference.double()).flatten()
    ref_flat = reference.double().flatten()
    return float(diff.norm() / ref_flat.norm()), float(diff.abs().max())


def _pp_worker(
    rank: int,
    master_port: int,
    partition_env: str | None,
    latents_init: torch.Tensor,
    pos_bias: torch.Tensor,
    neg_bias: torch.Tensor,
    result_queue: Any,
) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(PP_WORLD_SIZE)
        if partition_env is not None:
            os.environ["VLLM_PP_LAYER_PARTITION"] = partition_env
        else:
            os.environ.pop("VLLM_PP_LAYER_PARTITION", None)

        OmniTpuPlatform.set_device(0)
        init_distributed_environment(
            world_size=PP_WORLD_SIZE,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{master_port}",
            local_rank=rank,
        )
        initialize_model_parallel(pipeline_parallel_size=PP_WORLD_SIZE)

        device = torch.device("tpu")
        pipe = _ToyPPPipeline(
            num_layers=NUM_LAYERS,
            dim=HIDDEN_DIM,
            seed=BASE_SEED,
            device=device,
            dtype=torch.bfloat16,
        )
        missing_count = sum(isinstance(m, PPMissingLayer) for m in pipe.layers)
        setup_info = (
            get_pp_group().world_size,
            (pipe.start_layer, pipe.end_layer),
            missing_count,
        )

        latents: Any = latents_init.to(device=device, dtype=torch.bfloat16)
        pos_b = pos_bias.to(device=device, dtype=torch.bfloat16)
        neg_b = neg_bias.to(device=device, dtype=torch.bfloat16)
        timestep = torch.tensor(1.0, device=device, dtype=torch.bfloat16)

        with torch.inference_mode():
            for _ in range(2):
                noise_pred = pipe.predict_noise_maybe_with_cfg(
                    do_true_cfg=True,
                    true_cfg_scale=CFG_SCALE,
                    positive_kwargs={"x": latents, "cond_bias": pos_b},
                    negative_kwargs={"x": latents, "cond_bias": neg_b},
                    cfg_normalize=True,
                )
                latents = pipe.scheduler_step_maybe_with_cfg(
                    noise_pred, timestep, latents, do_true_cfg=True
                )
            resolved = torch.as_tensor(latents)
            pipe._sync_pp_send()

        torch.accelerator.synchronize()
        out_cpu = resolved.detach().cpu().float().numpy() if rank == 0 else None
        result_queue.put(("ok", rank, setup_info, out_cpu))
    except Exception as exc:
        result_queue.put(("error", rank, exc, None))
    finally:
        destroy_distributed_env()
        vllm_parallel_state._PP = None


@pytest.mark.xfail(strict=True, raises=TimeoutError, reason=PP_P2P_UNSUPPORTED)
@pytest.mark.parametrize(
    "partition_env,expected_ranges,expected_missing",
    [
        (None, ((0, 2), (2, 4)), (2, 2)),
        ("1,3", ((0, 1), (1, 4)), (3, 1)),
    ],
    ids=["even_split", "custom_partition"],
)
def test_diffusion_pipeline_parallelism(
    partition_env: str | None,
    expected_ranges: tuple[tuple[int, int], tuple[int, int]],
    expected_missing: tuple[int, int],
) -> None:
    """Diffusion Pipeline Parallelism: 2-stage PipelineParallelMixin on TPU.

    - TPU path: run a 4-block ``PipelineParallelMixin`` DiT across 2 spawned TPU
      ranks (even split and ``VLLM_PP_LAYER_PARTITION="1,3"``) over ``tpu_dist``.
    - Setup check: assert ``get_pp_group().world_size == 2`` and ``make_layers``
      active / ``PPMissingLayer`` block counts on both ranks.
    - Outcome check: compare rank-0 denoised latents against CPU fp32 reference via
      ``assert_parity`` scaled by CPU bf16 error.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= PP_WORLD_SIZE, (
        f"Test requires {PP_WORLD_SIZE} TPU devices, found {device_count}"
    )

    gen = torch.Generator(device="cpu").manual_seed(7)
    latents_init = torch.randn(
        (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), generator=gen, dtype=torch.float32
    )
    pos_bias = torch.randn(
        (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), generator=gen, dtype=torch.float32
    )
    neg_bias = torch.randn(
        (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), generator=gen, dtype=torch.float32
    )

    ref_fp32 = _cpu_reference_denoise(latents_init, pos_bias, neg_bias, torch.float32)
    ref_bf16 = _cpu_reference_denoise(latents_init, pos_bias, neg_bias, torch.bfloat16)
    cpu_bf16_rel_l2, cpu_bf16_max_abs = _error_metrics(ref_bf16, ref_fp32)

    master_port = portpicker.pick_unused_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_pp_worker,
            args=(
                rank,
                master_port,
                partition_env,
                latents_init,
                pos_bias,
                neg_bias,
                result_queue,
            ),
        )
        for rank in range(PP_WORLD_SIZE)
    ]
    for proc in procs:
        proc.start()
    deadline = time.monotonic() + WORKER_TIMEOUT_S
    try:
        # Read before join: a worker's queued tensor storage dies with the worker.
        records = [
            result_queue.get(timeout=max(deadline - time.monotonic(), 0.0))
            for _ in range(PP_WORLD_SIZE)
        ]
        for proc in procs:
            proc.join(timeout=max(deadline - time.monotonic(), 0.0))
    except queue.Empty:
        raise TimeoutError(
            f"PP=2 partition={partition_env}: workers sent no result within "
            f"{WORKER_TIMEOUT_S}s"
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

    pp_sizes = tuple(rec[2][0] for rec in records)
    layer_ranges = tuple(rec[2][1] for rec in records)
    missing_counts = tuple(rec[2][2] for rec in records)
    assert pp_sizes == (PP_WORLD_SIZE, PP_WORLD_SIZE), (
        f"PP world sizes {pp_sizes} != {(PP_WORLD_SIZE, PP_WORLD_SIZE)}"
    )
    assert layer_ranges == expected_ranges, (
        f"PP layer ranges {layer_ranges} != {expected_ranges}"
    )
    assert missing_counts == expected_missing, (
        f"PPMissingLayer counts {missing_counts} != {expected_missing}"
    )

    rank0_out = records[0][3]
    assert rank0_out is not None, "Rank 0 did not return denoised latents"
    assert_parity(
        torch.from_numpy(rank0_out),
        ref_fp32,
        name=f"PP=2 partition={partition_env}",
        max_rel_l2=max(BF16_ERR_RATIO * cpu_bf16_rel_l2, PP_REL_L2_FLOOR),
        max_abs=max(BF16_ERR_RATIO * cpu_bf16_max_abs, PP_ABS_FLOOR),
    )
