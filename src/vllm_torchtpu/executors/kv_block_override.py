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

Shared by every executor that collects `get_num_gpu_blocks_override` from
its workers (multiproc and Ray multi-host); build 155 failed on the Ray
path's own copy of this logic, which still demanded exact agreement.
"""

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_NUM_BLOCKS_OVERRIDE_TOL = 4
_NUM_BLOCKS_OVERRIDE_REL_TOL = 0.005


def reconcile_num_gpu_blocks_override(
    worker_overrides: list[int | None], workers_per_stage: int | None = None
) -> int | None:
    """Return the agreed block-count override, or None if no worker set one.

    Raises ValueError when the spread across workers exceeds both the
    absolute and the relative tolerance -- that is a sizing bug, not jitter.

    `workers_per_stage` splits `worker_overrides` (in rank order) into
    pipeline stages. Stages hold different layers and so measure different
    counts; agreement is checked within each stage and the smallest stage
    value is pinned, since vLLM sizes every stage's pool to it. A stage
    that set no override, because it holds no attention or no mamba layers,
    is rejected when other stages did: vLLM would apply their value to it
    too, against a capacity it never measured.
    """
    if workers_per_stage and 0 < workers_per_stage < len(worker_overrides):
        stage_values = [
            reconcile_num_gpu_blocks_override(
                worker_overrides[start : start + workers_per_stage]
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
        return min(stage_values)
    overrides = [ovr for ovr in worker_overrides if ovr is not None]
    if not overrides:
        return None

    lo, hi = min(overrides), max(overrides)
    tol = max(_NUM_BLOCKS_OVERRIDE_TOL, int(lo * _NUM_BLOCKS_OVERRIDE_REL_TOL))
    if hi - lo > tol:
        raise ValueError(
            "[kv-sizing] workers disagree on the compact-mamba "
            f"attention block count by {hi - lo} blocks "
            f"(min={lo}, max={hi}, tolerance={tol}). Each worker "
            "sizes the pool from its own measured free HBM, so a "
            "spread this large means the workers do not have "
            "comparable HBM budgets -- check the per-worker "
            "'Compact-mamba KV cache:' lines in "
            "tpu_runner._maybe_set_compact_mamba_num_blocks_"
            f"override for the outlier. Per-worker values: "
            f"{sorted(overrides)}"
        )
    logger.info(
        "[kv-sizing] compact-mamba attention blocks across %d "
        "workers: min=%d max=%d spread=%d (tolerance=%d); "
        "pinning num_gpu_blocks_override=%d",
        len(overrides),
        lo,
        hi,
        hi - lo,
        tol,
        lo,
    )
    return lo
