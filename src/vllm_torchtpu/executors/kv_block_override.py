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

from typing import Optional

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_NUM_BLOCKS_OVERRIDE_TOL = 4
_NUM_BLOCKS_OVERRIDE_REL_TOL = 0.005


def reconcile_num_gpu_blocks_override(
        worker_overrides: list[Optional[int]]) -> Optional[int]:
    """Return the agreed block-count override, or None if no worker set one.

    Raises ValueError when the spread across workers exceeds both the
    absolute and the relative tolerance -- that is a sizing bug, not jitter.
    """
    overrides = [ovr for ovr in worker_overrides if ovr is not None]
    if not overrides:
        return None

    lo, hi = min(overrides), max(overrides)
    tol = max(_NUM_BLOCKS_OVERRIDE_TOL, int(lo * _NUM_BLOCKS_OVERRIDE_REL_TOL))
    if hi - lo > tol:
        raise ValueError("[kv-sizing] workers disagree on the compact-mamba "
                         f"attention block count by {hi - lo} blocks "
                         f"(min={lo}, max={hi}, tolerance={tol}). Each worker "
                         "sizes the pool from its own measured free HBM, so a "
                         "spread this large means the workers do not have "
                         "comparable HBM budgets -- check the per-worker "
                         "'Compact-mamba KV cache:' lines in "
                         "tpu_runner._maybe_set_compact_mamba_num_blocks_"
                         f"override for the outlier. Per-worker values: "
                         f"{sorted(overrides)}")
    logger.info(
        "[kv-sizing] compact-mamba attention blocks across %d "
        "workers: min=%d max=%d spread=%d (tolerance=%d); "
        "pinning num_gpu_blocks_override=%d", len(overrides), lo, hi, hi - lo,
        tol, lo)
    return lo
