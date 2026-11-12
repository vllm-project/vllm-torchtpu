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
"""Reconcile per-worker compact-mamba block-count overrides.

Workers size the KV pool from their OWN measured free HBM, so their block
counts never agree exactly. The spread that matters is relative: a few tens
of MiB of allocator jitter is a couple of blocks when a block is megabytes
(Kimi-K3: ~14 MiB/block, spreads of 1-3 observed at TP=32) but hundreds of
blocks when a block is ~140 KiB (Kimi-Linear-48B, where 8 workers landed
551348..551506 -- a spread of 158 against an absolute tolerance of 4).
Accept whichever tolerance is larger, so the check still catches a worker
that genuinely disagrees about the budget, and pin the MINIMUM so no worker
is asked for more blocks than it measured.

A rank that compiles the largest token bucket during the memory profile
holds more HBM afterwards than a rank that loads the same bucket, from the
profile cache or, under TPU_PRECOMPILE_PEER_REPLAY, from a peer. That spread
is a fixed number of bytes, and no block tolerance fits it across block
sizes and pool sizes. Which ranks compile depends on the state of the
caches, not on a flag: peer replay with a cold cache leaves one compiling
rank, and a partly warm cache (e.g. after an interrupted start) can do the
same without it. So when the executor has each worker's KV budget in bytes,
which it does after `determine_available_memory`, a pipeline stage whose
budgets differ by at most `_NUM_BLOCKS_OVERRIDE_BYTES_TOL` is accepted too.
This makes 1 GiB the limit for every compact-mamba pool sized after the
profile run. Accepting a spread never over-commits a worker, because the
minimum is pinned. A worker short by more than 1 GiB, e.g. one sharing its
chip with a leftover process, still fails start-up.

Shared by every executor that collects `get_num_gpu_blocks_override` from
its workers (multiproc and Ray multi-host); build 155 failed on the Ray
path's own copy of this logic, which still demanded exact agreement.
"""

from typing import Any, cast

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_NUM_BLOCKS_OVERRIDE_TOL = 4
_NUM_BLOCKS_OVERRIDE_REL_TOL = 0.005
# Compile residue measured with TP8 and EP, the compact-mamba layout and an
# 8192-token profile bucket: 0.68 GiB on Qwen3.8-Flash-Next-FP8 (463 blocks at
# 1.5 MiB per block) and 0.11 GiB on Qwen3.5-397B-A17B-FP8 (30 blocks at
# 3.75 MiB per block). 1 GiB leaves about 0.3 GiB of margin over the larger.
# With every rank compiling that bucket, all 8 Qwen3.8 ranks held the same
# 0.68 GiB. The residue depends on the model, not only on the bucket size, and
# can also change with `max_num_batched_tokens` and with torch_tpu or libtpu
# updates: a budget spread just over 1 GiB more likely means it grew than that
# a worker shares its chip. The INFO line below logs the budget spread.
_NUM_BLOCKS_OVERRIDE_BYTES_TOL = 1 << 30

_GIB = 1 << 30


def reconcile_num_gpu_blocks_override(
    worker_overrides: list[int | None],
    workers_per_stage: int | None = None,
    budgets: list[int] | None = None,
    peer_replay: bool = False,
) -> int | None:
    """Return the agreed block-count override, or None if no worker set one.

    Raises ValueError when the spread across workers exceeds the absolute and
    the relative block tolerance and, given `budgets`, the budgets of the
    workers that set an override differ by more than the byte tolerance: that
    is a sizing bug or an outlier worker, not jitter.

    `budgets` holds, per worker and in the same order as `worker_overrides`,
    the KV budget in bytes that the worker sized its pool from.
    `peer_replay` only adds the TPU_PRECOMPILE_PEER_REPLAY=0 workaround to
    the error.

    `workers_per_stage` splits `worker_overrides` (in rank order) into
    pipeline stages. Stages hold different layers and so measure different
    counts; agreement is checked within each stage and the smallest stage
    value is pinned, since vLLM sizes every stage's pool to it. A stage
    that set no override, because it holds no attention or no mamba layers,
    is rejected when other stages did: vLLM would apply their value to it
    too, against a capacity it never measured.
    """
    if budgets is not None and len(budgets) != len(worker_overrides):
        raise ValueError(
            f"[kv-sizing] {len(budgets)} KV budgets for {len(worker_overrides)} workers"
        )
    if workers_per_stage and 0 < workers_per_stage < len(worker_overrides):
        stage_values = [
            reconcile_num_gpu_blocks_override(
                worker_overrides[start : start + workers_per_stage],
                budgets=None
                if budgets is None
                else budgets[start : start + workers_per_stage],
                peer_replay=peer_replay,
            )
            for start in range(0, len(worker_overrides), workers_per_stage)
        ]
        if all(value is None for value in stage_values):
            return None
        missing = [i for i, value in enumerate(stage_values) if value is None]
        if missing:
            raise ValueError(
                f"[kv-sizing] pipeline stage(s) {missing} set no compact-mamba "
                "attention block count while the other stages did (per-stage "
                f"values: {stage_values}). vLLM applies one block count to "
                "every stage, and a stage that did not size its own pool has "
                "no measured capacity for it. Partition the layers so every "
                "stage holds both attention and mamba layers, or pin "
                "--num-gpu-blocks-override."
            )
        # None was rejected above; the filter only narrows the type.
        return min(value for value in stage_values if value is not None)
    overrides = [ovr for ovr in worker_overrides if ovr is not None]
    if not overrides:
        return None

    lo, hi = min(overrides), max(overrides)
    tol = max(_NUM_BLOCKS_OVERRIDE_TOL, int(lo * _NUM_BLOCKS_OVERRIDE_REL_TOL))
    within_bytes = False
    bytes_note = ""
    if budgets is not None:
        set_budgets = [
            budget
            for budget, ovr in zip(budgets, worker_overrides, strict=True)
            if ovr is not None
        ]
        budget_spread = max(set_budgets) - min(set_budgets)
        within_bytes = budget_spread <= _NUM_BLOCKS_OVERRIDE_BYTES_TOL
        bytes_note = (
            f", budget spread {budget_spread / _GIB:.2f} GiB against "
            f"{_NUM_BLOCKS_OVERRIDE_BYTES_TOL / _GIB:.2f} GiB"
        )
    if hi - lo > tol and not within_bytes:
        compile_note = ""
        if budgets is not None:
            compile_note = (
                " A rank that compiled the largest token bucket during the "
                "memory profile, while the others loaded it, holds more HBM "
                "than they do; if that rank is the outlier, the compile left "
                "more than the byte tolerance."
            )
            if peer_replay:
                compile_note += (
                    " Under TPU_PRECOMPILE_PEER_REPLAY only one rank compiles "
                    "each bucket, and TPU_PRECOMPILE_PEER_REPLAY=0 makes "
                    "every rank compile it."
                )
        raise ValueError(
            "[kv-sizing] workers disagree on the compact-mamba "
            f"attention block count by {hi - lo} blocks "
            f"(min={lo}, max={hi}, tolerance={tol}{bytes_note}). Each worker "
            "sizes the pool from its own measured free HBM, so a "
            "spread this large means the workers do not have "
            "comparable HBM budgets -- check the per-worker "
            "'Compact-mamba KV cache:' lines "
            "(KVCacheManager.size_compact_mamba_kv_cache) "
            f"for the outlier.{compile_note} To start anyway, give every "
            "worker the same size with --num-gpu-blocks-override or "
            "--kv-cache-memory-bytes. "
            f"Per-worker values: {sorted(overrides)}"
        )
    logger.info(
        "[kv-sizing] compact-mamba attention blocks across %d "
        "workers: min=%d max=%d spread=%d (tolerance=%d%s); "
        "pinning num_gpu_blocks_override=%d",
        len(overrides),
        lo,
        hi,
        hi - lo,
        tol,
        bytes_note,
        lo,
    )
    return lo


def adopt_worker_num_gpu_blocks_override(
    executor: Any, budgets: list[int] | None = None
) -> None:
    """Copy the workers' agreed `num_gpu_blocks_override` to the engine config.

    The workers' hybrid KV sizing sets the override on their own cache_config:
    the uniform layout in `get_kv_cache_spec` (the workers then skip their
    profile run), the compact-mamba layout in `determine_available_memory`,
    after the profile run. Workers are separate processes, so every executor
    must copy the value to the engine-side config after both calls, or vLLM
    derives its own block count, which does not match the per-layer TPU
    allocation. A no-op once the engine has a value (the user's
    --num-gpu-blocks-override, or one copied by the first call). Shared so
    the executors cannot drift apart again.

    `budgets` is the per-worker list `determine_available_memory` returned:
    the bytes each worker gave to `size_compact_mamba_kv_cache`. Given it,
    a budget spread up to `_NUM_BLOCKS_OVERRIDE_BYTES_TOL` is accepted (see
    the module docstring). The call after `get_kv_cache_spec` has no budgets
    and keeps the block tolerances.
    """
    cache_config = executor.vllm_config.cache_config
    if cache_config.num_gpu_blocks_override is not None:
        return
    pc = executor.vllm_config.parallel_config
    agreed = reconcile_num_gpu_blocks_override(
        cast(list, executor.collective_rpc("get_num_gpu_blocks_override")),
        workers_per_stage=pc.world_size // pc.pipeline_parallel_size,
        budgets=budgets,
        # Only picks the workarounds the error names; the owner/waiter split
        # behind peer replay is installed only under TPU_PARALLEL_PRECOMPILE.
        peer_replay=envs.TPU_PRECOMPILE_PEER_REPLAY and envs.TPU_PARALLEL_PRECOMPILE,
    )
    if agreed is not None:
        cache_config.num_gpu_blocks_override = agreed
