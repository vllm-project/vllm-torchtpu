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
"""Options of the TPU schedulers, from ``additional_config["tpu_scheduler"]``.

    --additional-config '{"tpu_scheduler": {"promote_warmest": false}}'

Every option defaults to main's behaviour without it. The options only change
which request a scheduler admits next and never change a compiled program,
so the section stays out of the compile-cache key
(``HASH_IGNORED_ADDITIONAL_CONFIG_KEYS``). Parse once at init with :func:`get`.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

SECTION = "tpu_scheduler"


@dataclass(frozen=True)
class SchedulerOptions:
    promote_warmest: bool = True
    """Promote the warmest request of each waiting queue to its head once per
    step (``PrefixAwareScheduler``). False admits in FCFS order. The IX TP8
    points ran false; #1 ran the default."""


def parse(raw: Any) -> SchedulerOptions:
    """The ``tpu_scheduler`` section as options; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``0`` for a bool, say), raises ``ValueError``.
    """
    if raw is None:
        return SchedulerOptions()
    if not isinstance(raw, dict):
        raise ValueError(
            f"additional_config.{SECTION} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    fields = {f.name: f.type for f in dataclasses.fields(SchedulerOptions)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(
            f"additional_config.{SECTION} has unknown keys {unknown}; "
            f"valid keys: {sorted(fields)}"
        )
    for name, value in raw.items():
        field_type = fields[name]
        if type(value) is not field_type:
            name_of_type = getattr(field_type, "__name__", str(field_type))
            raise ValueError(
                f"additional_config.{SECTION}.{name} must be a "
                f"{name_of_type}, got {value!r}"
            )
    return SchedulerOptions(**raw)


def get(vllm_config: Any) -> SchedulerOptions:
    """The engine's scheduler options."""
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        return SchedulerOptions()
    return parse(additional.get(SECTION))
