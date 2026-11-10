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
"""Options of the TPU MoE layers, from ``additional_config["tpu_moe"]``.

    --additional-config '{"tpu_moe": {"lowtok_fused_fp8_dispatch": false}}'

An option here only changes a path that a runtime rule has already selected,
so it never turns a path on for a model by itself. The options change compiled
programs, so the section stays in the compile-cache key. Parse once at init
with :func:`get` and keep the result.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

SECTION = "tpu_moe"


@dataclass(frozen=True)
class MoeOptions:
    lowtok_fused_fp8_dispatch: bool = True
    """Where the fused few-token collectives run (``moe_lowtok_fused_op``,
    armed by its DP x TP layout rule), push each engine's rows to its DP peers
    quantized per row to FP8 with their scale instead of as bf16, which halves
    the dispatch bytes. The expert pass quantizes the same rows the same way,
    so the expert inputs are unchanged. On by default, as every DP x TP decode
    leg ran it; it changes nothing where the fused collectives are not armed.
    ``false`` is the ablation arm that checks the end-to-end acceptance length
    on TPU."""


def parse(raw: Any) -> MoeOptions:
    """The ``tpu_moe`` section as options; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``1`` for a bool, say), raises ``ValueError``.
    """
    if raw is None:
        return MoeOptions()
    if not isinstance(raw, dict):
        raise ValueError(
            f"additional_config.{SECTION} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    fields = {f.name: f.type for f in dataclasses.fields(MoeOptions)}
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
    return MoeOptions(**raw)


def get(vllm_config: Any) -> MoeOptions:
    """The engine's MoE options."""
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        return MoeOptions()
    return parse(additional.get(SECTION))
