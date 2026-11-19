# SPDX-License-Identifier: Apache-2.0
"""q38 L5 SAB: step-synchronous whole-step ON / OFF windows inside one served run (served_ab.py + the driver hook)."""

import os
import pathlib
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv
from vllm_torchtpu.megakernel_runtime import served_ab as sab


def _runner(step, warm=True):
    """A batch that changes every step (so each step's fingerprint is distinct)."""
    ids = [f"req-{(step + j) % 97}" for j in range(4)]
    return SimpleNamespace(warmup_done=warm, input_batch=SimpleNamespace(
        req_ids=ids, num_computed_tokens_cpu=np.array([1000 + step * 4 + j for j in range(4)], dtype=np.int64)))


def _ranks(tmp_path, world, period, confirm, errors):
    out = []
    for r in range(world):
        out.append(sab.ServedAB(period, confirm, r, world, str(tmp_path)))
    return out


@pytest.fixture
def logs(monkeypatch):
    rec = {"error": [], "info": []}
    monkeypatch.setattr(sab.logger, "error", lambda fmt, *a: rec["error"].append(fmt % a))
    monkeypatch.setattr(sab.logger, "info", lambda fmt, *a: rec["info"].append(fmt % a))
    return rec


def test_imported_from_staged_tree():
    """Proof guard (L2's staging lesson): with Q38_STAGED_ROOT set, vllm_torchtpu must import from under it, not from an
    installed copy. Skipped when unset (the proof sets it and requires an exact pass count, so a skip fails the proof)."""
    root = os.environ.get("Q38_STAGED_ROOT")
    if not root:
        pytest.skip("Q38_STAGED_ROOT not set")
    import vllm_torchtpu

    f = os.path.realpath(vllm_torchtpu.__file__)
    assert f.startswith(os.path.realpath(root) + os.sep), f
    assert os.path.realpath(sab.__file__).startswith(os.path.realpath(root) + os.sep), sab.__file__


def test_parse():
    assert sab.parse("") is None and sab.parse("  ") is None
    assert sab.parse("whole:4000") == (4000, 64) and sab.parse("whole:4000:32") == (4000, 32)
    for bad in ("whole", "whole:100:64", "whole:10:0", "seg:4000", "whole:x", "whole:4000:64:1"):
        with pytest.raises(ValueError):
            sab.parse(bad)


def test_from_env_arms_with_rank_world_and_store(logs, tmp_path):
    a = sab.from_env("whole:10:3", 2, 8, str(tmp_path / "s"))
    assert isinstance(a, sab.ServedAB) and (a.period, a.confirm, a.rank, a.world) == (10, 3, 2, 8)
    assert os.path.isdir(tmp_path / "s") and sab.from_env("", 0, 1, str(tmp_path)) is None
    assert any("q38 SAB armed on rank 2 of 8" in m for m in logs["info"])


def test_invalid_spec_is_off_with_one_error(logs, tmp_path):
    assert sab.from_env("whole:10:9", 0, 1, str(tmp_path)) is None
    assert len(logs["error"]) == 1 and "q38 SAB off" in logs["error"][0]


def test_single_rank_windows(logs, tmp_path):
    a = sab.ServedAB(10, 3, 0, 1, str(tmp_path))
    on = []
    for s in range(40):
        a.tick(_runner(s))
        on.append(a.whole_on())
    # windows of 10 calls: 0 ON, 1 OFF after 3 confirm calls, 2 ON, 3 OFF after 3
    want = [True] * 10 + [True] * 3 + [False] * 7 + [True] * 10 + [True] * 3 + [False] * 7
    assert on == want
    assert sum("OFF from calls 13" in m for m in logs["info"]) == 1 and not logs["error"]


def test_warmup_calls_do_not_count(logs, tmp_path):
    a = sab.ServedAB(10, 3, 0, 1, str(tmp_path))
    for s in range(25):
        a.tick(_runner(s, warm=False))
        assert a.whole_on()
    assert a.n == 0
    seq = []
    for s in range(20):
        a.tick(_runner(s))
        seq.append(a.whole_on())
    assert seq == [True] * 13 + [False] * 7


def _lockstep(ranks, steps, skew=None):
    """Every rank ticks once per global step (rank `skew` ticks one extra time before step 0: one step ahead)."""
    if skew is not None:
        ranks[skew].tick(_runner(-1))
    arms = []
    for s in range(steps):
        row = []
        for a in ranks:
            a.tick(_runner(s))
            row.append(a.whole_on())
        arms.append(row)
    return arms


def test_eight_ranks_in_lockstep_flip_together(logs, tmp_path):
    ranks = _ranks(tmp_path, 8, 20, 4, logs)
    arms = _lockstep(ranks, 80)
    assert all(len(set(r)) == 1 for r in arms), "ranks disagree on a step"
    col = [r[0] for r in arms]
    assert col == ([True] * 24 + [False] * 16) * 2
    assert not logs["error"]


def test_a_desynchronised_rank_disables_every_rank_and_none_flips(logs, tmp_path):
    """Rank 3 counts one call more than the others (out of step by one). At the OFF window's confirmation every rank
    sees unequal fingerprints: all keep whole-step ON for the rest of the run, and each logs exactly one ERROR."""
    ranks = _ranks(tmp_path, 8, 20, 4, logs)
    arms = _lockstep(ranks, 120, skew=3)
    assert all(all(r) for r in arms), "a rank flipped to OFF while out of step"
    assert all(a.disabled for a in ranks)
    per_rank = [sum(f"rank {r} DISABLED" in m for m in logs["error"]) for r in range(8)]
    assert per_rank == [1] * 8, per_rank


def test_a_missing_rank_entry_disables(logs, tmp_path):
    ranks = _ranks(tmp_path, 8, 20, 4, logs)
    arms = _lockstep(ranks[:7], 60)  # rank 7 never writes its entries
    assert all(all(r) for r in arms) and all(a.disabled for a in ranks[:7])
    assert len(logs["error"]) == 7 and all("missing or unequal" in m for m in logs["error"])


def test_store_write_failure_disables(logs, tmp_path):
    a = sab.ServedAB(10, 3, 0, 1, str(tmp_path / "x"))
    os.rmdir(tmp_path / "x")
    (tmp_path / "x").write_text("not a dir")
    a.tick(_runner(0))
    assert a.disabled and a.whole_on() and len(logs["error"]) == 1


# ------------------------------------------------------------------ the driver hook (a minimal driver, real try_forward)
class _Stop(BaseException):
    pass


def _driver(monkeypatch, ab):
    d = drv.Qwen38SegmentDriver.__new__(drv.Qwen38SegmentDriver)
    d.program, d.whole, d.buckets = object(), False, (8, 16, 32, 48, 64)
    d.whole_program, d.whole_above, d.whole_dry = object(), "stock", False
    d.whole_max_rows, d.whole_cap, d.stats, d.counters, d.seq, d.mk_calls = 64, 262144, {}, {}, 0, 0
    d.sab = ab
    notes = []

    def note(key):
        notes.append(key)
        if key.startswith("mk_b"):
            raise _Stop(key)
    d._note = note
    monkeypatch.setattr(drv.MegakernelDriver, "_inspect_step_detail", staticmethod(lambda *a: (True, 4, drv.T_VERIFY, "")))
    return d, notes


def _step(d, runner):
    try:
        return d.try_forward(runner, torch.zeros(16, dtype=torch.int32), torch.zeros(16, dtype=torch.int32), None, None)
    except _Stop as e:
        return ("MK", str(e))


def test_hook_off_window_routes_to_stock_and_counts(monkeypatch, logs, tmp_path):
    ab = sab.ServedAB(10, 3, 0, 1, str(tmp_path))
    d, notes = _driver(monkeypatch, ab)
    res = [_step(d, _runner(s)) for s in range(20)]
    assert res[:13] == [("MK", "mk_b16")] * 13
    assert res[13:] == [(False, None)] * 7
    assert notes.count("ab_off_b16") == 7
    assert d.counters.get("drafts_stock") == 7 * 4 * (drv.T_VERIFY - 1)


def test_hook_absent_leaves_the_route_unchanged(monkeypatch, tmp_path):
    d, notes = _driver(monkeypatch, None)
    assert all(_step(d, _runner(s)) == ("MK", "mk_b16") for s in range(30))
    assert "ab_off_b16" not in notes


def test_env_registered_parsed_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert "VLLM_TPU_SERVED_AB" in envs.environment_variables and "VLLM_TPU_SERVED_AB" in _TPU_COMPILE_ENV_IGNORED
    old = os.environ.pop("VLLM_TPU_SERVED_AB", None)
    try:
        assert envs.environment_variables["VLLM_TPU_SERVED_AB"]() == ""
        os.environ["VLLM_TPU_SERVED_AB"] = " whole:4000:64 "
        assert envs.environment_variables["VLLM_TPU_SERVED_AB"]() == "whole:4000:64"
    finally:
        os.environ.pop("VLLM_TPU_SERVED_AB", None)
        if old is not None:
            os.environ["VLLM_TPU_SERVED_AB"] = old


def test_only_host_code_reads_the_flag():
    """Static: the flag is read in envs.py / the compile-key ignore list / the driver's attach code; served_ab is imported
    only by the driver; no model, kernel or layer module mentions either."""
    import vllm_torchtpu

    root = pathlib.Path(vllm_torchtpu.__file__).parent
    readers, importers = set(), set()
    for p in root.rglob("*.py"):
        text = p.read_text(errors="ignore")
        rel = str(p.relative_to(root))
        if "VLLM_TPU_SERVED_AB" in text:
            readers.add(rel)
        if re.search(r"(import served_ab|served_ab import|megakernel_runtime\.served_ab)", text):
            importers.add(rel)
    # q38 L6-HOST (composed tree q38-l6-sab): core/l6_ab.py reads the flag on the host, at process
    # init, only to check that its own A/B classes are balanced inside SAB's windows (L5 balance rule).
    assert readers == {"envs.py", "compilation/tpu_compiler.py", "megakernel_runtime/qwen38_driver.py",
                       "megakernel_runtime/served_ab.py", "core/l6_ab.py"}, readers
    # q38 L6-HOST: core/l6_ab.py imports served_ab only to parse the spec with SAB's own parse().
    assert importers == {"megakernel_runtime/qwen38_driver.py", "core/l6_ab.py"}, importers
