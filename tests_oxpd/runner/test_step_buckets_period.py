# SPDX-License-Identifier: Apache-2.0
"""q38 L5 STEPBUCKETS2: per-bucket host step periods (VLLM_TPU_LOG_STEP_BUCKETS_PERIOD)."""

import os
import re
import subprocess
import sys
from unittest.mock import patch

import pytest

from vllm_torchtpu.runner import step_buckets

LINE2 = re.compile(r"^STEPBUCKETS2 steps=(\d+) periods=(\d+) span_us=(\d+) trim:(\d+)/(\d+)(.*)$")


def _fresh(monkeypatch, every, period, ticks_ns):
    """Reset the module state; the host clock returns ticks_ns in order."""
    monkeypatch.setattr(step_buckets, "_EVERY", every)
    monkeypatch.setattr(step_buckets, "_PERIOD", period)
    monkeypatch.setattr(step_buckets, "_counts", {})
    monkeypatch.setattr(step_buckets, "_steps", 0)
    monkeypatch.setattr(step_buckets, "_per", {})
    monkeypatch.setattr(step_buckets, "_trim", [0, 0])
    monkeypatch.setattr(step_buckets, "_last", [None, None, None])
    it = iter(ticks_ns)
    monkeypatch.setattr(step_buckets, "_clock", lambda: next(it))
    lines = []
    monkeypatch.setattr(step_buckets.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    return lines


def _parse2(line):
    m = LINE2.match(line)
    assert m, line
    per = {}
    for kv in m.group(6).split():
        b, v = kv.split(":")
        n, us = v.split("/")
        per[int(b)] = (int(n), int(us))
    return {"steps": int(m.group(1)), "periods": int(m.group(2)), "span_us": int(m.group(3)),
            "trim": (int(m.group(4)), int(m.group(5))), "per": per}


def test_period_is_booked_to_the_earlier_calls_bucket(monkeypatch):
    # calls at t = 0, 9, 17, 30, 40.5, 41 ms with buckets 16, 16, 32, 16, 48, 16
    ms = [0.0, 9.0, 17.0, 30.0, 40.5, 41.0]
    lines = _fresh(monkeypatch, 6, True, [int(x * 1e6) for x in ms])
    for b in (16, 16, 32, 16, 48, 16):
        step_buckets.count(b)
    assert lines[0] == "STEPBUCKETS steps=6 16:4 32:1 48:1"
    d = _parse2(lines[1])
    # periods booked to the earlier call: 0..9 -> 16, 9..17 -> 16, 17..30 -> 32, 30..40.5 -> 16, 40.5..41 -> 48
    assert d["per"] == {16: (3, 27500), 32: (1, 13000), 48: (1, 500)}
    assert d["steps"] == 6 and d["periods"] == 5 and d["trim"] == (0, 0)
    assert d["span_us"] == 41000


def test_bucket_sums_plus_trim_equal_the_span(monkeypatch):
    """Conservation: every interval between consecutive calls is booked exactly once (ns state exact; the
    printed microseconds within one floor per bucket)."""
    import random

    rnd = random.Random(7)
    t, ticks, buckets = 0, [], []
    for i in range(4000):
        t += rnd.choice([rnd.randint(5_000_000, 40_000_000), rnd.randint(51_000_000, 70_000_000) if i % 397 == 0 else 9_000_000])
        ticks.append(t)
        buckets.append(rnd.choice([16, 32, 48, 64, 96, 128, 160, 256]))
    lines = _fresh(monkeypatch, 1000, True, ticks)
    for b in buckets:
        step_buckets.count(b)
    total_ns = sum(v[1] for v in step_buckets._per.values()) + step_buckets._trim[1]
    assert total_ns == ticks[-1] - ticks[0]
    assert sum(v[0] for v in step_buckets._per.values()) + step_buckets._trim[0] == len(ticks) - 1
    assert step_buckets._trim[0] > 0
    last = _parse2([x for x in lines if x.startswith("STEPBUCKETS2")][-1])
    booked = sum(us for _, us in last["per"].values()) + last["trim"][1]
    assert 0 <= last["span_us"] - booked <= len(last["per"]) + 1
    assert last["periods"] == last["steps"] - 1 == 3999


def test_long_periods_are_trimmed_not_booked(monkeypatch):
    lines = _fresh(monkeypatch, 3, True, [0, 60_000_000, 70_000_000])
    for b in (16, 32, 16):
        step_buckets.count(b)
    d = _parse2(lines[1])
    assert d["trim"] == (1, 60000) and d["per"] == {32: (1, 10000)}


def test_flag_off_keeps_the_original_line_only(monkeypatch):
    lines = _fresh(monkeypatch, 3, False, [])  # the clock must not be read: an empty iterator would raise
    for n in (3200, 2048, 3200, 6400, 3200, 3200):
        step_buckets.count(n)
    assert lines == ["STEPBUCKETS steps=3 2048:1 3200:2", "STEPBUCKETS steps=6 2048:1 3200:4 6400:1"]
    assert step_buckets._per == {} and step_buckets._trim == [0, 0]


def test_period_without_counts_is_a_no_op(monkeypatch):
    lines = _fresh(monkeypatch, 0, True, [])
    step_buckets.count(16)
    assert lines == [] and step_buckets._steps == 0 and step_buckets._per == {}


def test_env_flag_is_registered_parses_and_is_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    name = "VLLM_TPU_LOG_STEP_BUCKETS_PERIOD"
    assert name in envs.environment_variables and name in _TPU_COMPILE_ENV_IGNORED
    getter = envs.environment_variables[name]
    with patch.dict(os.environ):
        os.environ.pop(name, None)
        assert getter() is False
        os.environ[name] = "1"
        assert getter() is True


def test_module_reads_the_flag_at_import():
    """A fresh interpreter with the env set gets _PERIOD True (the module-level read the runner relies on)."""
    code = "from vllm_torchtpu.runner import step_buckets as s; print(s._PERIOD, s._EVERY)"
    env = dict(os.environ, VLLM_TPU_LOG_STEP_BUCKETS="2000", VLLM_TPU_LOG_STEP_BUCKETS_PERIOD="1")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "True 2000"


def test_only_host_modules_read_the_flag():
    """Static: the flag appears only in envs.py, the compile-key ignore list and step_buckets.py, and step_buckets is
    imported only by the runner (never by a model, kernel or layer module that a trace could reach)."""
    import pathlib

    import vllm_torchtpu

    root = pathlib.Path(vllm_torchtpu.__file__).parent
    readers, importers = set(), set()
    for p in root.rglob("*.py"):
        text = p.read_text(errors="ignore")
        rel = str(p.relative_to(root))
        if "VLLM_TPU_LOG_STEP_BUCKETS_PERIOD" in text:
            readers.add(rel)
        if re.search(r"(from vllm_torchtpu\.runner import step_buckets|import vllm_torchtpu\.runner\.step_buckets|from \. import step_buckets)", text):
            importers.add(rel)
    assert readers == {"envs.py", "compilation/tpu_compiler.py", "runner/step_buckets.py"}, readers
    assert importers == {"runner/tpu_runner.py"}, importers
