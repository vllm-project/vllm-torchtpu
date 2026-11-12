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
"""Direct tests for the per-worker block-count reconciliation.

`tests/executors/test_tpu_multiproc_executor.py` already drives this function
through both executors, so the realistic scenarios (Kimi-K3 at TP=32,
Kimi-Linear-48B at TP=8) live there. This file pins the decision rule itself:

    tol = max(4, int(lo * 0.005))        accept when hi - lo <= tol, return lo

The two tolerances swap over at lo=1000, and which one wins is the whole point
of the function, so the boundary is worth a test that does not depend on any
executor plumbing.
"""

import pytest

from vllm_torchtpu.executors.kv_block_override import (
    _NUM_BLOCKS_OVERRIDE_REL_TOL,
    _NUM_BLOCKS_OVERRIDE_TOL,
    reconcile_num_gpu_blocks_override,
)


def test_no_worker_set_an_override():
    """Uniform sizing was skipped on every worker, so vLLM keeps its own
    num_blocks computation. None, not 0 and not an exception."""
    assert reconcile_num_gpu_blocks_override([None, None, None]) is None


def test_empty_worker_list():
    """An executor with no workers to poll must not raise on min() of an
    empty sequence."""
    assert reconcile_num_gpu_blocks_override([]) is None


# 5428 carries no meaning of its own. It is the block count the sibling
# executor test has used since it was written, kept here so the two files
# read as the same scenario. What does matter is its magnitude: at a block
# count in the thousands the relative tolerance dominates
# (int(5428 * 0.005) == 27), so the small spreads below are inside it. The
# same spreads against a three-digit block count would raise, which is what
# the absolute-floor tests further down cover.


def test_single_worker_agrees_with_itself():
    """One worker cannot disagree with anyone, so its own count is agreed."""
    assert reconcile_num_gpu_blocks_override([5428]) == 5428


def test_identical_overrides_pass_through():
    """The ordinary case: every worker measured the same budget."""
    assert reconcile_num_gpu_blocks_override([5428, 5428, 5428]) == 5428


def test_workers_without_an_override_are_ignored():
    """Only some workers run compact-mamba sizing. The Nones drop out and the
    workers that did measure still decide the budget."""
    assert reconcile_num_gpu_blocks_override([None, 5428, 5430, None]) == 5428


def test_minimum_wins_not_the_average_or_the_maximum():
    """The pool is sized once for every worker, so the smallest measured
    budget is the only safe pick: a larger one asks some worker for HBM it
    does not have."""
    assert reconcile_num_gpu_blocks_override([5440, 5428, 5433]) == 5428


# ---------------------------------------------------------------------------
# The absolute floor, for block counts below 1000.
# ---------------------------------------------------------------------------


def test_absolute_tolerance_accepts_a_spread_exactly_at_the_floor():
    """At lo=100 the relative tolerance rounds to 0, so the floor of 4 is what
    applies. A spread of exactly 4 is inside it."""
    assert reconcile_num_gpu_blocks_override([100, 104, 102]) == 100


def test_absolute_tolerance_rejects_one_block_past_the_floor():
    """One block past the floor is the whole point of having a floor: at this
    scale the relative tolerance is 0, so 4 is the entire allowance."""
    with pytest.raises(ValueError, match="workers disagree"):
        reconcile_num_gpu_blocks_override([100, 105])


# ---------------------------------------------------------------------------
# The relative tolerance, which only starts to matter at lo=1000.
# ---------------------------------------------------------------------------


def test_relative_tolerance_takes_over_at_one_thousand_blocks():
    """int(1000 * 0.005) == 5, the first block count where the relative
    tolerance is wider than the floor of 4. A spread of 5 is accepted here and
    would be rejected at 999."""
    assert reconcile_num_gpu_blocks_override([1000, 1005]) == 1000
    with pytest.raises(ValueError, match="workers disagree"):
        reconcile_num_gpu_blocks_override([999, 1004])


def test_relative_tolerance_scales_with_the_block_count():
    """At lo=100000 the tolerance is 500 blocks. The same 500-block spread is
    a hard failure at small block counts, which is exactly the behaviour the
    Kimi-Linear-48B fix needed."""
    assert reconcile_num_gpu_blocks_override([100_000, 100_500]) == 100_000
    with pytest.raises(ValueError, match="workers disagree"):
        reconcile_num_gpu_blocks_override([100_000, 100_501])


def test_tolerance_constants_match_the_documented_rule():
    """The boundaries above are hand-computed from these two constants; if
    either is retuned, the expected values in this file must be recomputed."""
    assert _NUM_BLOCKS_OVERRIDE_TOL == 4
    assert _NUM_BLOCKS_OVERRIDE_REL_TOL == 0.005


# ---------------------------------------------------------------------------
# The failure message is the only debugging aid an operator gets: the engine
# dies during start-up, before any per-worker log is easy to correlate.
# ---------------------------------------------------------------------------


def test_error_names_the_spread_the_bounds_and_the_tolerance():
    """6456 vs 8000 is a 24% disagreement, far outside any tolerance. The
    numbers in the message are what tell an operator whether they are looking
    at measurement jitter or at one worker with a different HBM budget."""
    with pytest.raises(ValueError) as exc:
        reconcile_num_gpu_blocks_override([6456, 8000, 6456])

    message = str(exc.value)
    assert "by 1544 blocks" in message
    assert "min=6456" in message
    assert "max=8000" in message
    assert "tolerance=32" in message


def test_error_lists_the_per_worker_values_sorted():
    """Sorted, so the outlier is at one end instead of buried at whichever
    rank happened to report it."""
    with pytest.raises(ValueError) as exc:
        reconcile_num_gpu_blocks_override([6456, 8000, 6457, None])

    assert "[6456, 6457, 8000]" in str(exc.value)
