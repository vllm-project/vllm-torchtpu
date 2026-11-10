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
"""Options of the TPU collectives, from ``additional_config["tpu_collectives"]``.

    --additional-config '{"tpu_collectives": {"small_all_reduce": true}}'

Every option defaults to main's behaviour without it. The options change
compiled programs, so the section stays in the compile-cache key. Parse once
at init with :func:`get` and keep the result.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

SECTION = "tpu_collectives"


@dataclass(frozen=True)
class CollectivesOptions:
    small_all_reduce: bool = False
    """Arm the Pallas ICI small collectives over the expert-parallel mesh: the
    TP all-reduce of a small bf16 block (at most
    ``small_all_reduce_op.MAX_ROWS`` rows) and the DP MoE combine of a small
    block (at most ``small_all_reduce_op.COMBINE_MAX_ROWS`` rows per engine).
    Larger blocks keep XLA's collectives. Needs an expert-parallel group.
    Off by default."""


def parse(raw: Any) -> CollectivesOptions:
    """The ``tpu_collectives`` section as options; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``1`` for a bool, say), raises ``ValueError``.
    """
    if raw is None:
        return CollectivesOptions()
    if not isinstance(raw, dict):
        raise ValueError(
            f"additional_config.{SECTION} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    fields = {f.name: f.type for f in dataclasses.fields(CollectivesOptions)}
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
    return CollectivesOptions(**raw)


def get(vllm_config: Any) -> CollectivesOptions:
    """The engine's collectives options."""
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        return CollectivesOptions()
    return parse(additional.get(SECTION))
