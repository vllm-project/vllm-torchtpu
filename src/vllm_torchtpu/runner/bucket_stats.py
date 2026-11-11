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
"""Count which padded token buckets the target forward runs at.

Every bucket is a compiled program that stays resident on the host (~0.75 GiB
per backbone bucket on MiniMax-M3.1 TP8), so a bucket real traffic never
hits is pure cost. The log line gives, per bucket, the step count and the
share of its rows that were real tokens, cumulative since startup.
"""

from __future__ import annotations

import time
from collections import Counter

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class BucketStats:
    """Cumulative per-bucket step and valid-token counts, logged periodically."""

    def __init__(self, enabled: bool, interval_s: float = 60.0):
        self._enabled = enabled
        self._interval_s = interval_s
        self._steps: Counter[int] = Counter()
        self._valid: Counter[int] = Counter()
        self._last_log = time.monotonic()
        self._logged_steps = 0

    def record(self, padded_tokens: int, valid_tokens: int) -> None:
        if not self._enabled:
            return
        self._steps[padded_tokens] += 1
        self._valid[padded_tokens] += valid_tokens
        now = time.monotonic()
        if now - self._last_log >= self._interval_s:
            self._last_log = now
            self.log()

    def log(self) -> None:
        total = sum(self._steps.values())
        if not self._enabled or total == self._logged_steps:
            return
        self._logged_steps = total
        logger.info(
            "Target-forward token buckets (cumulative, %d steps): %s",
            total,
            " ".join(
                f"{b}={n}({100 * self._valid[b] / (b * n):.0f}%)"
                for b, n in sorted(self._steps.items())
            ),
        )
