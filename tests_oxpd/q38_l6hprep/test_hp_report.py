# SPDX-License-Identifier: Apache-2.0
"""q38 L6-INSTR T-1(ii) HP-REPORT: host_phases._report groups the phases by
sample count and takes one percentile call per group. The HOSTPHASES line must
be byte-identical to the per-phase report for the same samples, and each
statistic bitwise equal."""

import copy

import numpy as np
import pytest

import hp_report_ref as ref
from vllm_torchtpu.runner import host_phases


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)


def _new_line(monkeypatch, samples, every, rank=5, steps=4000):
    log = _Log()
    monkeypatch.setattr(host_phases, "logger", log)
    monkeypatch.setattr(host_phases, "_EVERY", every)
    monkeypatch.setattr(host_phases, "_rank", rank)
    monkeypatch.setattr(host_phases, "_steps", steps)
    monkeypatch.setattr(host_phases, "_samples", samples)
    host_phases._report()
    assert host_phases._samples == {}, "the report must clear the window"
    assert len(log.lines) == 1
    return log.lines[0]


def _ref_line(samples, every, rank=5, steps=4000):
    log = _Log()
    ref.report_ref(samples, every, rank, steps, log)
    return log.lines[0]


def test_known_window_exact_line(monkeypatch):
    samples = {"a": [1000, 2000, 3000], "b": [5000], "c": [100, 200, 300, 400]}
    want = ("HOSTPHASES rank=3 steps=8 per-step host us (p50/p90/mean-per-step) "
            "total_mean=3 a=2/3/2 b=5/5/1 c=0/0/0")
    assert _ref_line(copy.deepcopy(samples), 4, rank=3, steps=8) == want
    assert _new_line(monkeypatch, samples, 4, rank=3, steps=8) == want


@pytest.mark.parametrize("seed", range(40))
def test_line_is_byte_identical_over_random_ragged_windows(monkeypatch, seed):
    rng = np.random.default_rng(1000 + seed)
    every = int(rng.choice([1, 3, 64, 200]))
    samples = ref.window(rng, every=every)
    want = _ref_line(copy.deepcopy(samples), every)
    got = _new_line(monkeypatch, samples, every)
    assert got == want, (got, want)


@pytest.mark.parametrize("seed", range(20))
def test_each_statistic_is_bitwise_the_per_phase_value(seed):
    rng = np.random.default_rng(2000 + seed)
    samples = ref.window(rng, every=int(rng.choice([2, 64, 200])))
    want = ref.stats_ref(samples)
    got = host_phases._phase_stats(samples)
    assert list(got) != [] and set(got) == set(want)
    for name in samples:
        g = tuple(float(x).hex() for x in got[name])
        w = tuple(float(x).hex() for x in want[name])
        assert g == w, (name, len(samples[name]), g, w)


def test_names_keep_their_window_order(monkeypatch):
    # Interleaved lengths: grouping must not reorder the line.
    samples = {"z": [10] * 4, "y": [20] * 2, "x": [30] * 4, "w": [40] * 1, "v": [50] * 2}
    line = _new_line(monkeypatch, copy.deepcopy(samples), 4)
    assert [kv.split("=")[0] for kv in line.split()[-5:]] == ["z", "y", "x", "w", "v"]
    assert line == _ref_line(samples, 4)


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import os

    import vllm_torchtpu

    want = os.environ.get("Q38_L6HPREP_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6HPREP_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
