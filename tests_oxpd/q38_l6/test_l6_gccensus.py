# SPDX-License-Identifier: Apache-2.0
"""q38 L6 gen-2 heap type census (core/l6_gccensus.py): host-only tests; run in a CDK CPU pod."""

import gc
import inspect
import os
from types import SimpleNamespace as NS

import pytest

from vllm_torchtpu.core import l6_gccensus as cz


@pytest.fixture
def cen(monkeypatch):
    for k in (cz.ENV, cz.MS_ENV, cz.MAX_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(cz.ENV, "1")
    cz._resolve()
    lines = []
    monkeypatch.setattr(cz.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    yield monkeypatch, lines
    if cz._callback in gc.callbacks:
        gc.callbacks.remove(cz._callback)
    for k in (cz.ENV, cz.MS_ENV, cz.MAX_ENV):
        monkeypatch.delenv(k, raising=False)
    cz._resolve()


def _clock(mp, seq):
    it = iter(seq)
    mp.setattr(cz.time, "perf_counter", lambda: next(it))


def test_flag_parse_and_defaults(monkeypatch):
    for k in (cz.ENV, cz.MS_ENV, cz.MAX_ENV):
        monkeypatch.delenv(k, raising=False)
    cz._resolve()
    assert not cz.ON and cz.MS == 1000 and cz.MAX == 4
    monkeypatch.setenv(cz.ENV, "1")
    monkeypatch.setenv(cz.MS_ENV, "250")
    monkeypatch.setenv(cz.MAX_ENV, "2")
    cz._resolve()
    assert cz.ON and cz.MS == 250 and cz.MAX == 2
    cz._resolve()


def test_nothing_before_warmup(cen):
    mp, lines = cen
    cz.on_step(NS(warmup_done=False))
    assert not cz._state["installed"] and lines == [] and cz._callback not in gc.callbacks


def test_warm_census_once_and_callback_installed(cen):
    mp, lines = cen
    r = NS(warmup_done=True)
    cz.on_step(r)
    cz.on_step(r)
    assert cz._callback in gc.callbacks
    assert len(lines) == 1 and "event=warm" in lines[0] and "total=" in lines[0] and "top=" in lines[0], lines


def test_callback_marks_only_slow_gen2(cen):
    mp, lines = cen
    _clock(mp, [0.0, 0.5, 10.0, 12.0, 20.0, 22.0])  # gen0 0.5 s, gen1 2 s, gen2 2 s
    cz._callback("start", {"generation": 0}); cz._callback("stop", {"generation": 0})
    assert cz._state["pending"] is None
    cz._callback("start", {"generation": 1}); cz._callback("stop", {"generation": 1})
    assert cz._state["pending"] is None
    cz._callback("start", {"generation": 2}); cz._callback("stop", {"generation": 2})
    assert cz._state["pending"] is not None and abs(cz._state["pending"] - 2000.0) < 1e-6


def test_fast_gen2_is_not_marked(cen):
    mp, lines = cen
    mp.setenv(cz.MS_ENV, "1000")
    cz._resolve()
    _clock(mp, [0.0, 0.999])
    cz._callback("start", {"generation": 2}); cz._callback("stop", {"generation": 2})
    assert cz._state["pending"] is None


def test_callback_does_no_census(cen):
    """The callback runs inside a collection: it must only stamp and flag, never walk the heap or log."""
    src = inspect.getsource(cz._callback)
    assert "get_objects" not in src and "census(" not in src and "logger" not in src


def test_pending_gen2_census_runs_next_step_and_cap_bounds(cen):
    mp, lines = cen
    mp.setenv(cz.MAX_ENV, "2")
    cz._resolve()
    r = NS(warmup_done=True)
    cz.on_step(r)                              # warm census (1 of 2)
    cz._state["pending"] = 1500.0
    cz.on_step(r)                              # gen2 census (2 of 2)
    cz._state["pending"] = 3000.0
    cz.on_step(r)                              # capped
    assert [("event=warm" in l, "event=gen2" in l) for l in lines] == [(True, False), (False, True)], lines
    assert "dur_ms=1500" in lines[1]


def test_census_counts_types(cen):
    mp, lines = cen

    class Q38CensusProbe:
        pass

    # Freeze what is already there (as the worker's warm-up freeze_gc_heap() does), so the heap that earlier tests
    # in this process left in generation 2 cannot crowd the probe out of the top 15 (in the full P1 run after
    # test_l6_gcadmit.py, generation 2 held 868,162 objects with 165,190 functions: dev j-44c9b389).
    gc.collect()
    gc.freeze()
    try:
        keep = [Q38CensusProbe() for _ in range(3000)]
        gc.collect(); gc.collect()
        line = cz.census("warm", 0.0)
    finally:
        gc.unfreeze()  # test isolation only
    assert "Q38CensusProbe:3000" in line.split("top=")[1], line
    total = int(line.split("total=")[1].split()[0])
    assert total >= 3000
    del keep


def test_runner_hook_after_update_states_and_guarded():
    from vllm_torchtpu.runner import tpu_runner

    src = inspect.getsource(tpu_runner.TPUModelRunner.execute_model)
    i_upd = src.index("self._update_states(scheduler_output)")
    i_hook = src.index("_l6_gccensus.on_step(self)")
    i_mark = src.index('host_phases.mark("update_states")')
    assert i_upd < i_hook < i_mark
    assert "if _l6_gccensus.ON:" in src[i_upd:i_hook]


def test_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    for k in (cz.ENV, cz.MS_ENV, cz.MAX_ENV):
        assert k in envs.environment_variables and k in tpu_compiler._TPU_COMPILE_ENV_IGNORED
