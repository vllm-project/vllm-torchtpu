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
"""Options of the TPU model runner, from ``additional_config["tpu_runner"]``.

    --additional-config '{"tpu_runner": {"stage_timing": false}}'

The options only steer host-side telemetry and never change a compiled
program, so the section stays out of the compile-cache key
(``HASH_IGNORED_ADDITIONAL_CONFIG_KEYS``). Parse once at init with :func:`get`.
"""

from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import config_section

SECTION = "tpu_runner"


@dataclass(frozen=True)
class RunnerOptions:
    stage_timing: bool = True
    """Record each step's per-stage device and host timings
    (``RunnerStageTiming``) for the KV connector's transfer statistics. It takes
    effect only when the KV connector has a ``TpuKVConnectorStats`` sink
    (``TPUConnector`` or ``TPURaidenConnector``, alone or inside a
    multi-connector), because nothing else reads the timings. On by default, so
    those connectors' statistics are unchanged; set it to false to save the
    per-stage event records, which cost about 0.4-0.5 ms per decode step at
    batch 1-2 under P/D."""


def parse(raw: Any) -> RunnerOptions:
    """The ``tpu_runner`` section as options; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``1`` for a bool, say), raises ``ValueError`` (``config_section.parse``).
    """
    return config_section.parse(RunnerOptions, SECTION, raw)


def get(vllm_config: Any) -> RunnerOptions:
    """The engine's runner options."""
    return config_section.get(RunnerOptions, SECTION, vllm_config)
