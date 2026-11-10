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
"""Options of the TPU worker, from ``additional_config["tpu_worker"]``.

    --additional-config '{"tpu_worker": {"numa_bind": true}}'

Every option defaults to main's behaviour without it. The options only place
the worker process on the host and never change a compiled program, so the
section stays out of the compile-cache key
(``HASH_IGNORED_ADDITIONAL_CONFIG_KEYS``). Parse once at init with :func:`get`.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

SECTION = "tpu_worker"


@dataclass(frozen=True)
class WorkerOptions:
    numa_bind: bool = False
    """Before the TPU runtime starts, narrow each worker's CPU set to the NUMA
    node its chip is attached to and prefer that node's memory
    (``worker.numa_bind``). The inherited CPU set is only ever narrowed, and a
    rank that does not map onto a chip is left alone. Off by default."""


def parse(raw: Any) -> WorkerOptions:
    """The ``tpu_worker`` section as options; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``1`` for a bool, say), raises ``ValueError``.
    """
    if raw is None:
        return WorkerOptions()
    if not isinstance(raw, dict):
        raise ValueError(
            f"additional_config.{SECTION} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    fields = {f.name: f.type for f in dataclasses.fields(WorkerOptions)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(
            f"additional_config.{SECTION} has unknown keys {unknown}; "
            f"valid keys: {sorted(fields)}"
        )
    for name, value in raw.items():
        want = fields[name]
        assert isinstance(want, type)  # the annotations are classes, not strings
        if type(value) is not want:
            raise ValueError(
                f"additional_config.{SECTION}.{name} must be a "
                f"{want.__name__}, got {value!r}"
            )
    return WorkerOptions(**raw)


def get(vllm_config: Any) -> WorkerOptions:
    """The engine's worker options."""
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        return WorkerOptions()
    return parse(additional.get(SECTION))
