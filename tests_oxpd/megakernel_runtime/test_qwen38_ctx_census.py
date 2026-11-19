# SPDX-License-Identifier: Apache-2.0
"""q38 L5 CTXCENSUS: the log-only served context census of the Qwen3.8 driver (VLLM_TPU_Q38_CTX_CENSUS=N).

A minimal driver runs the real `try_forward` on fake runners: every N-th real whole-step call (and every N-th step
sent to stock by `wholecap`) logs, on TP rank 0 only, one line with the bucket, the request count and the slots'
host-side computed-token counts in slot order. With the flag off nothing is logged and every decision is unchanged.
"""

import contextlib
import os
import pathlib
import re
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv

ENV = "VLLM_TPU_Q38_CTX_CENSUS"
WHOLE_OUT = ("whole-step output",)
CAP = 100_000
LINE = re.compile(r"^q38 CTXCENSUS b(\d+) n=(\d+) path=(whole|stock_ctx) kv=([0-9,]*) calls=(\d+)$")


class _NoDeviceRead:
    """Stands in for the device `context_lens`: any use of it (a device read / host sync) raises."""

    def __getattr__(self, name):
        raise AssertionError(f"device tensor touched: .{name}")

    def __len__(self):
        raise AssertionError("device tensor touched: len()")


class _BadList(np.ndarray):
    def tolist(self):
        raise RuntimeError("census read failed")


def _runner(kv, warm=True, junk=7_777_777):
    """A runner whose first len(kv) slots are live; the slots past n_req hold junk the census must not log."""
    arr = np.array(list(kv) + [junk] * 4, dtype=np.int32)
    return SimpleNamespace(warmup_done=warm, n_req=len(kv), input_batch=SimpleNamespace(num_computed_tokens_cpu=arr))


def _driver(monkeypatch, rank=0):
    d = drv.Qwen38SegmentDriver.__new__(drv.Qwen38SegmentDriver)
    d.program, d.whole, d.buckets = object(), False, (8, 16, 32, 48, 64, 128)
    d.whole_program = lambda **kw: WHOLE_OUT
    d.whole_above, d.whole_dry, d.whole_max_rows, d.whole_cap = "stock", False, 64, CAP
    d.stats, d.counters, d.seq, d.mk_calls, d.whole_calls = {}, {}, 0, 10, 0
    d.segx_left = d.h0check_left = d.check_left = d.wholecheck_left = 0
    d.shadow = d.whole_shadow = None
    d.warm = d.sync = False
    d._ctxc = None
    d._contexts = lambda runner, tag: contextlib.nullcontext()
    # STACK-D v3 (v3c) driver: L5 SAB off, and L1-D07B1's host check of the shared GDN metadata on the first whole
    # calls (a no-op stand-in; the real check is a no-op with VLLM_TPU_MK_GDN_META_DEDUP off).
    d.sab = None
    d.whole_module = SimpleNamespace(check_gdn_meta_shared=lambda: None)
    monkeypatch.setattr(
        drv.MegakernelDriver, "_inspect_step_detail", staticmethod(lambda *a: (True, a[2].n_req, drv.T_VERIFY, ""))
    )
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(ps, "get_tp_group", lambda: SimpleNamespace(rank_in_group=rank))
    return d


@pytest.fixture
def logs(monkeypatch):
    rec = {"info": [], "warning": []}
    monkeypatch.setattr(drv.logger, "info", lambda fmt, *a: rec["info"].append(fmt % a if a else fmt))
    monkeypatch.setattr(drv.logger, "warning", lambda fmt, *a: rec["warning"].append(fmt % a if a else fmt))
    return rec


def _census(rec):
    return [m for m in rec["info"] if m.startswith("q38 CTXCENSUS b")]


def _step(d, runner, rows):
    kw = {"context_lens": _NoDeviceRead()}
    return d.try_forward(runner, torch.zeros(rows, dtype=torch.int32), torch.zeros(rows, dtype=torch.int32), None, None, kw)


# A served-shaped sequence: whole steps at b16 / b32 / b48 (unsorted contexts), steps past `wholecap` (stock_ctx),
# a step above `wholerows` (row cap, not a context fallback), warm-up and dummy (<= 64 tokens) steps.
def _sequence():
    seq = []
    for i in range(12):
        seq.append((16, [9000 - 100 * i, 120 + i, 70_000, 4097]))
        seq.append((32, [5 * i + 300, 99_000, 65, 77_000, 1200, 33_333 + i, 800, 81]))
        if i % 2 == 0:
            seq.append((16, [CAP, 2000 + i, 500]))  # CAP + T_VERIFY > wholecap: stock_ctx
        if i % 3 == 0:
            seq.append((48, [61_000, 1_000 + i, 15_000, 400, 2, 90_000, 7, 8, 9, 10, 11]))
    seq.append((128, [3000] * 20))  # above wholerows 64: stock_above_whole_b128
    return seq


def _drive(monkeypatch, value, rank=0, warm=True):
    rec = {"info": [], "warning": []}
    with patch.dict(os.environ):
        if value is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = value
        with monkeypatch.context() as m:
            m.setattr(drv.logger, "info", lambda fmt, *a: rec["info"].append(fmt % a if a else fmt))
            m.setattr(drv.logger, "warning", lambda fmt, *a: rec["warning"].append(fmt % a if a else fmt))
            d = _driver(m, rank)
            seq = _sequence()
            res = [_step(d, _runner(kv, warm=warm), rows) for rows, kv in seq]
            res += [_step(d, _runner([10, 20, 30, 40], warm=warm), 16)]  # dummy-sized contexts
            res += [_step(d, _runner([5000, 6000, 7000], warm=False), 16)]  # warm-up step
    return d, res, rec, seq


def test_imported_from_staged_tree():
    """Proof guard (L2's staging lesson): with Q38_STAGED_ROOT set, vllm_torchtpu and the driver must import from under
    it. Skipped when unset (the proof sets it and requires an exact pass count with 0 skipped)."""
    import vllm_torchtpu

    print("vllm_torchtpu", vllm_torchtpu.__file__, "qwen38_driver", drv.__file__)
    root = os.environ.get("Q38_STAGED_ROOT")
    if not root:
        pytest.skip("Q38_STAGED_ROOT not set")
    want = os.path.realpath(root) + os.sep
    assert os.path.realpath(vllm_torchtpu.__file__).startswith(want), vllm_torchtpu.__file__
    assert os.path.realpath(drv.__file__).startswith(want), drv.__file__


def test_env_registered_parsed_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert ENV in envs.environment_variables and ENV in _TPU_COMPILE_ENV_IGNORED
    getter = envs.environment_variables[ENV]
    with patch.dict(os.environ):
        os.environ.pop(ENV, None)
        assert getter() == 0
        os.environ[ENV] = "200"
        assert getter() == 200
        os.environ[ENV] = "-1"
        with pytest.raises(ValueError):
            getter()


def _hash_with(value):
    from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            gpu_memory_utilization=0.9, num_gpu_blocks=1024, num_gpu_blocks_override=None,
            enable_prefix_caching=False, mamba_cache_mode="none",
        ),
        model_config=SimpleNamespace(max_model_len=4096),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=16),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        speculative_config=None,
    )
    with patch("vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files", return_value=[]):
        with patch.dict(os.environ):
            if value is None:
                os.environ.pop(ENV, None)
            else:
                os.environ[ENV] = value
            return compute_tpu_compilation_hash(cfg)


def test_compile_hash_identical_with_and_without_the_flag():
    off = _hash_with(None)
    assert off == _hash_with("0") == _hash_with("1") == _hash_with("200")


def test_off_logs_nothing_and_no_decision_changes(monkeypatch):
    """Unset, 0 and N = 3 (rank 0 and rank 5): identical return values, step-mix notes, draft counters and whole-step
    call counts on the same fake inputs; unset / 0 log no census line at all."""
    runs = {v: _drive(monkeypatch, v) for v in (None, "0", "3")}
    runs["3r5"] = _drive(monkeypatch, "3", rank=5)
    base_d, base_res, _, seq = runs[None]
    # 28 real whole steps + the dummy-sized and the warm-up step (whole too, not counted by the census)
    assert base_res.count((True, WHOLE_OUT)) == 30 and base_d.stats.get("stock_whole_ctx") == 6
    assert base_d.stats.get("stock_above_whole_b128") == 1
    for key, (d, res, rec, _) in runs.items():
        assert res == base_res, key
        assert d.stats == base_d.stats and d.counters == base_d.counters and d.whole_calls == base_d.whole_calls, key
    for key in (None, "0"):
        rec = runs[key][2]
        assert not [m for m in rec["info"] + rec["warning"] if "CTXCENSUS" in m], key
    assert _census(runs["3"][2])


def test_every_third_call_on_rank0_in_slot_order(monkeypatch):
    d, res, rec, seq = _drive(monkeypatch, "3")
    lines = [LINE.match(m) for m in _census(rec)]
    assert all(lines), _census(rec)
    whole = [(r, kv) for r, kv in seq if r <= 64 and max(kv) + drv.T_VERIFY <= CAP]
    ctx = [(r, kv) for r, kv in seq if r <= 64 and max(kv) + drv.T_VERIFY > CAP]
    assert len(whole) == 28 and len(ctx) == 6
    for path, steps in (("whole", whole), ("stock_ctx", ctx)):
        got = [m for m in lines if m.group(3) == path]
        want = [(i, steps[i - 1]) for i in range(3, len(steps) + 1, 3)]
        assert len(got) == len(want) == len(steps) // 3, (path, len(got))
        for m, (i, (rows, kv)) in zip(got, want):
            assert int(m.group(1)) == rows and int(m.group(2)) == len(kv), m.group(0)
            assert [int(x) for x in m.group(4).split(",")] == kv, m.group(0)  # slot order, no junk slots
            assert int(m.group(5)) == i
    # unsorted contexts really occur in the logged lines (a sorted list would differ)
    assert any([int(x) for x in m.group(4).split(",")] != sorted(int(x) for x in m.group(4).split(",")) for m in lines)
    armed = [m for m in rec["info"] if m.startswith("q38 CTXCENSUS armed")]
    assert len(armed) == 1 and "every 3 real calls" in armed[0] and "rank 0" in armed[0]
    assert not rec["warning"]


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5, 6, 7])
def test_other_ranks_log_no_census_line(monkeypatch, rank):
    d, res, rec, _ = _drive(monkeypatch, "3", rank=rank)
    assert not _census(rec)
    assert len([m for m in rec["info"] if m.startswith("q38 CTXCENSUS armed")]) == 1


def test_warmup_and_dummy_steps_are_not_counted(monkeypatch):
    """Warm-up steps (warmup_done False) and dummy-sized steps (no context > 64) do not count or log."""
    d, res, rec, _ = _drive(monkeypatch, "1", warm=False)
    assert not _census(rec) and d.whole_calls > 0
    d, res, rec, seq = _drive(monkeypatch, "1")
    n_real = sum(1 for r, kv in seq if r <= 64)
    assert len(_census(rec)) == n_real  # the dummy-sized and the warm-up step at the end are not logged


def test_a_census_error_turns_the_census_off_and_never_the_megakernel(monkeypatch, logs):
    """The census's own read (`.tolist()`, used by nothing else on the step) raises: one warning, no line, and every
    step still runs the whole-step program (an escaping error would hit try_forward's fail-closed handler)."""
    monkeypatch.setenv(ENV, "1")
    d = _driver(monkeypatch)
    runner = _runner([5000, 6000])
    runner.input_batch.num_computed_tokens_cpu = runner.input_batch.num_computed_tokens_cpu.view(_BadList)
    out = [_step(d, runner, 16) for _ in range(4)]
    assert out == [(True, WHOLE_OUT)] * 4 and d.whole_program is not None and d.program is not None
    assert len([m for m in logs["warning"] if "CTXCENSUS off after an error" in m]) == 1
    assert not _census(logs)


def test_only_host_code_reads_the_flag():
    """Static: the flag appears only in envs.py, the compile-key ignore list and the driver; in the driver only in
    the census helpers (not in a traced or program-building method)."""
    import vllm_torchtpu

    root = pathlib.Path(vllm_torchtpu.__file__).parent
    readers = {str(p.relative_to(root)) for p in root.rglob("*.py") if ENV in p.read_text(errors="ignore")}
    assert readers == {"envs.py", "compilation/tpu_compiler.py", "megakernel_runtime/qwen38_driver.py"}, readers
    src = pathlib.Path(drv.__file__).read_text()
    assert src.count("envs.VLLM_TPU_Q38_CTX_CENSUS") == 1
    body = src[src.index("    def _arm_ctx_census"): src.index("    def try_forward")]
    assert "envs.VLLM_TPU_Q38_CTX_CENSUS" in body
