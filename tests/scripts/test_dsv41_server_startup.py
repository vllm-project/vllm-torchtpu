# SPDX-License-Identifier: Apache-2.0
"""Startup summaries preserve parallel rank and incomplete-run boundaries."""

import datetime
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench/server_steps"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from summarize_startup import assert_no_compilation, summarize_startup  # noqa: E402

pytestmark = pytest.mark.cpu_test


def test_compilation_audit_excludes_startup_and_handles_new_year():
    start = datetime.datetime(2026, 12, 31, 23, 59, 58).timestamp() * 1e9
    intervals = [[int(start), int(start + 5e9)]]
    assert_no_compilation("12-31 23:58:00 Compiling FX graph", intervals)
    with pytest.raises(ValueError, match="Compilation in formal"):
        assert_no_compilation("01-01 00:00:01 Compiling FX graph", intervals)
    with pytest.raises(ValueError, match="Cannot timestamp"):
        assert_no_compilation("Compiling FX graph", intervals)


def test_parallel_ranks_are_not_summed():
    log = "\n".join(
        f"(EngineCore_DP{rank} pid=1) init engine (warmup) took {total} s "
        f"(compilation: {compile_time} s)"
        for rank, total, compile_time in [(0, 20, 12), (1, 23, 15)]
    )
    result = summarize_startup(log)
    assert result["complete_ranks"] == 2
    assert result["max_engine_init_seconds"] == 23
    assert result["max_reported_compilation_seconds"] == 15


def test_partial_startup_and_cache_events_are_not_success():
    result = summarize_startup(
        "(Worker_DP0_EP0 pid=1) Saved compiled artifact to cache\n"
        "(Worker_DP0_EP0 pid=1) Loading compiled artifact from cache\n"
        "(Worker_DP1_EP1 pid=2) Starting torch.compile\n"
    )
    assert result["complete_ranks"] == 0
    assert result["max_engine_init_seconds"] is None
    assert result["ranks"][0]["artifact_saves"] == 1
    assert result["ranks"][0]["artifact_load_attempts"] == 1
