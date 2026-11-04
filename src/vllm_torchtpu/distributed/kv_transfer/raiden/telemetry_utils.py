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
"""Telemetry utility functions for TPU Raiden metrics."""

import re

# Metric name prefix
RAIDEN_METRIC_PREFIX = "tpu_raiden_"

_LABELED_METRIC_KEY_REGEX = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?")
_LABEL_PAIR_REGEX = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
)
_UNESCAPE_PATTERN = re.compile(r'\\([\\n"tr])')


def _unescape_match(match: re.Match) -> str:
    escaped = match.group(1)
    return {"n": "\n", "t": "\t", "r": "\r"}.get(escaped, escaped)


def parse_raiden_metric_key(raw_key: str) -> tuple[str, dict[str, str]]:
    """Parses Prometheus text-formatted keys emitted by Raiden telemetry, e.g.:

    tpu_raiden_sent_bytes_total{direction="push"} into base name and labels
    dict.
    """
    match = _LABELED_METRIC_KEY_REGEX.fullmatch(raw_key.strip())
    if not match:
        return raw_key.strip(), {}

    base_name = match.group(1)
    raw_labels = match.group(2)
    if not raw_labels:
        return base_name, {}

    labels = {}
    for pair_match in _LABEL_PAIR_REGEX.finditer(raw_labels):
        k, raw_val = pair_match.group(1), pair_match.group(2)
        labels[k] = _UNESCAPE_PATTERN.sub(_unescape_match, raw_val)
    return base_name, labels


def normalize_raiden_metric_name(name: str) -> str:
    """Ensures a Raiden metric name starts with RAIDEN_METRIC_PREFIX."""
    return (
        name
        if name.startswith(RAIDEN_METRIC_PREFIX)
        else f"{RAIDEN_METRIC_PREFIX}{name}"
    )
