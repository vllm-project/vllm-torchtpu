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
"""Per-target-forward metrics. Token counts exclude padding and cached history."""

from dataclasses import dataclass

TOKEN_BUCKETS = (1, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
TIME_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10)

# All metrics have rank and bucket labels. bucket is the local compiled shape;
# global token counts describe the logical batch, repeated on each PP/PCP rank.
FORWARD_METRICS = {
    "valid_tokens": (
        "Effective global query tokens in one target forward.",
        TOKEN_BUCKETS,
    ),
    "capacity_tokens": ("Global padded capacity in one target forward.", TOKEN_BUCKETS),
    "utilization_ratio": (
        "Global effective tokens divided by padded capacity.",
        (0.125, 0.25, 0.5, 0.75, 0.875, 1),
    ),
    "request_min_tokens": (
        "Smallest scheduled query span in this forward.",
        TOKEN_BUCKETS,
    ),
    "request_max_tokens": (
        "Largest scheduled query span in this forward.",
        TOKEN_BUCKETS,
    ),
    "requests": (
        "Number of real requests in this forward.",
        (1, 2, 4, 8, 16, 32, 64, 128),
    ),
    "host_issue_seconds": (
        "Host duration of the forward_model call; not device execution time.",
        TIME_BUCKETS,
    ),
    "host_gap_seconds": (
        "Previous forward_model return to next call, "
        "including other work and idleness.",
        TIME_BUCKETS,
    ),
    "completion_span_seconds": (
        "Background-observed start-to-end Event completion span; "
        "approximate, not device timestamps.",
        TIME_BUCKETS,
    ),
    "completion_gap_seconds": (
        "Previous end Event observation to next start Event observation; "
        "includes MTP, communication and idle time.",
        TIME_BUCKETS,
    ),
    "observer_lag_seconds": (
        "Delay from recording the start Event to the collector beginning to wait.",
        TIME_BUCKETS,
    ),
}

ForwardSample = tuple[str, str, str, float]


@dataclass(frozen=True)
class ForwardBatch:
    bucket: int
    capacity: int
    request_tokens: tuple[int, ...]

    def samples(self, rank: str) -> list[ForwardSample]:
        assert self.bucket > 0 and self.capacity > 0 and self.request_tokens
        assert min(self.request_tokens) > 0
        valid = sum(self.request_tokens)
        assert valid <= self.capacity
        values = {
            "valid_tokens": valid,
            "capacity_tokens": self.capacity,
            "utilization_ratio": valid / self.capacity,
            "request_min_tokens": min(self.request_tokens),
            "request_max_tokens": max(self.request_tokens),
            "requests": len(self.request_tokens),
        }
        return [
            (rank, str(self.bucket), name, float(value))
            for name, value in values.items()
        ]
