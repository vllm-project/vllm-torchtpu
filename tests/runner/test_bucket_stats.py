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

from vllm_torchtpu.runner import bucket_stats
from vllm_torchtpu.runner.bucket_stats import BucketStats


def _capture(monkeypatch):
    lines = []
    monkeypatch.setattr(
        bucket_stats.logger, "info", lambda fmt, *a: lines.append(fmt % a)
    )
    return lines


def test_logs_cumulative_counts_and_valid_share(monkeypatch):
    lines = _capture(monkeypatch)
    stats = BucketStats(enabled=True, interval_s=1e9)
    stats.record(64, 48)
    stats.record(64, 16)
    stats.record(8192, 8192)
    stats.log()
    assert lines == [
        "Target-forward token buckets (cumulative, 3 steps): 64=2(50%) 8192=1(100%)"
    ]
    stats.log()  # nothing new since the last line
    assert len(lines) == 1


def test_disabled_rank_records_nothing(monkeypatch):
    lines = _capture(monkeypatch)
    stats = BucketStats(enabled=False, interval_s=0)
    stats.record(64, 1)
    stats.log()
    assert lines == []


def test_logs_on_interval(monkeypatch):
    lines = _capture(monkeypatch)
    stats = BucketStats(enabled=True, interval_s=0)
    stats.record(16, 4)
    assert lines == ["Target-forward token buckets (cumulative, 1 steps): 16=1(25%)"]
