# SPDX-License-Identifier: Apache-2.0
from vllm_torchtpu.runner import step_buckets


def test_counts_steps_per_bucket_and_reports_every_n(monkeypatch):
    monkeypatch.setattr(step_buckets, "_EVERY", 3)
    monkeypatch.setattr(step_buckets, "_counts", {})
    monkeypatch.setattr(step_buckets, "_steps", 0)
    lines = []
    monkeypatch.setattr(step_buckets.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    for n in (3200, 2048, 3200, 6400, 3200, 3200):
        step_buckets.count(n)
    assert step_buckets._counts == {2048: 1, 3200: 4, 6400: 1}
    assert lines == ["STEPBUCKETS steps=3 2048:1 3200:2", "STEPBUCKETS steps=6 2048:1 3200:4 6400:1"]


def test_off_by_default_is_a_no_op(monkeypatch):
    monkeypatch.setattr(step_buckets, "_EVERY", 0)
    monkeypatch.setattr(step_buckets, "_counts", {})
    monkeypatch.setattr(step_buckets, "_steps", 0)
    step_buckets.count(3200)
    assert step_buckets._counts == {} and step_buckets._steps == 0
