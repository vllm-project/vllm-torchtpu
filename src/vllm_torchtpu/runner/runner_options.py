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

The options only steer the host (telemetry and process-wide interpreter
settings) and never change a compiled program, so the section stays out of
the compile-cache key
(``HASH_IGNORED_ADDITIONAL_CONFIG_KEYS``). Parse once at init with :func:`get`.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

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

    gc_freeze: bool = False
    """After 500 steps, run ``gc.collect()`` and ``gc.freeze()`` and raise the
    gen-0 collection threshold to 50000, so Python's cyclic GC runs rarely in
    the decode loop (``runner.host_trims``). A trade-off for small-batch
    decode, off by default: it applies to every thread of the process, and
    objects created after the freeze are collected less often."""

    switch_interval_us: int = 0
    """Python's thread switch interval (``sys.setswitchinterval``) in
    microseconds, so a background thread (output, connector or submit thread)
    cannot hold the GIL for long while the step loop waits on it. 0 keeps
    Python's 5 ms default. A trade-off for small-batch decode: it applies to
    every thread of the process, and a shorter interval adds switching
    overhead to busy threads."""


def parse(raw: Any) -> RunnerOptions:
    """The ``tpu_runner`` section as options; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``1`` for a bool, say), raises ``ValueError``.
    """
    if raw is None:
        return RunnerOptions()
    if not isinstance(raw, dict):
        raise ValueError(
            f"additional_config.{SECTION} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    fields = {f.name: f.type for f in dataclasses.fields(RunnerOptions)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(
            f"additional_config.{SECTION} has unknown keys {unknown}; "
            f"valid keys: {sorted(fields)}"
        )
    for name, value in raw.items():
        expected = fields[name]
        if type(value) is not expected:
            kind = expected.__name__ if isinstance(expected, type) else expected
            raise ValueError(
                f"additional_config.{SECTION}.{name} must be a {kind}, got {value!r}"
            )
    options = RunnerOptions(**raw)
    if options.switch_interval_us < 0:
        raise ValueError(
            f"additional_config.{SECTION}.switch_interval_us must be >= 0, "
            f"got {options.switch_interval_us}"
        )
    return options


def get(vllm_config: Any) -> RunnerOptions:
    """The engine's runner options."""
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        return RunnerOptions()
    return parse(additional.get(SECTION))
