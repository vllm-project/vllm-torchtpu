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
"""Options of the pipeline chunk scheduler, from
``additional_config["tpu_pp_scheduler"]``.

    --additional-config '{"tpu_pp_scheduler": {"prefix_aware": true}}'

The platform installs ``TpuPipelineChunkScheduler`` on every engine with more
than one pipeline stage that names no other scheduler class, so every option
defaults to that scheduler's behaviour without it: an engine that sets none
of them schedules exactly as before. The options only change which request
runs next and how prefill work is priced when a step's chunks are sized; none
changes a compiled program, so the section stays out of the compile-cache key
(``HASH_IGNORED_ADDITIONAL_CONFIG_KEYS``). Parse once at init with
:func:`get`.

The scheduler's chunk sizing itself is main's lone-prompt and equal-cost
chunking (``TPU_PP_LONE_PROMPT_CHUNKS``, ``TPU_PP_EQUAL_COST_CHUNKS``);
``quadratic_cost`` refines it and needs the equal-cost cost model.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

SECTION = "tpu_pp_scheduler"


@dataclass(frozen=True)
class PPSchedulerOptions:
    paw_aging_tok_per_s: float = 0.0
    """Uncached prompt tokens credited per second a request has waited, in the
    ``prefix_aware`` orders (waiting-queue sort, running-prefill order,
    warmest-first promotion). 0 is no aging. Only meaningful with
    ``prefix_aware``, so a positive value without it is refused. The IX
    prefill legs ran 3000."""

    prefix_aware: bool = False
    """Borrow ``PrefixAwareScheduler``'s admission on a pipeline: the
    waiting-queue sort (on a KV producer), shortest-remaining-prefill-first
    order of running prefills, warmest-first promotion (unless
    ``tpu_scheduler.promote_warmest`` is false), the cross-step prefix-hit
    memo, divergent hybrid hits resolved through the KV connector, the
    trimmed token-id resend, packed prompt ids, and the connector's
    completion poll at the start of each step. The IX prefill legs ran
    true."""

    quadratic_cost: bool = False
    """Price prefill work by its attention cost as well as its token count,
    on the equal-cost step cost model: lone-prompt chunking's spread counts
    each pending request's (chunk, prefix) pairs and a waiting request's
    prefix hit, and the ``prefix_aware`` orders rank by effective tokens.
    Needs ``TPU_PP_EQUAL_COST_CHUNKS`` (on by default); the scheduler refuses
    it otherwise. The IX prefill legs ran true."""


def parse(raw: Any) -> PPSchedulerOptions:
    """The ``tpu_pp_scheduler`` section as options; None gives the defaults.

    Strict: an unknown key, or a value of the wrong type (``0`` for a bool,
    ``true`` for a number), raises ``ValueError``. A float option also takes a
    JSON integer.
    """
    if raw is None:
        return PPSchedulerOptions()
    if not isinstance(raw, dict):
        raise ValueError(
            f"additional_config.{SECTION} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    fields = {f.name: f.type for f in dataclasses.fields(PPSchedulerOptions)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(
            f"additional_config.{SECTION} has unknown keys {unknown}; "
            f"valid keys: {sorted(fields)}"
        )
    values: dict[str, Any] = {}
    for name, value in raw.items():
        field_type = fields[name]
        ok = type(value) is field_type or (field_type is float and type(value) is int)
        if not ok:
            name_of_type = getattr(field_type, "__name__", str(field_type))
            raise ValueError(
                f"additional_config.{SECTION}.{name} must be a "
                f"{name_of_type}, got {value!r}"
            )
        values[name] = float(value) if field_type is float else value
    options = PPSchedulerOptions(**values)
    aging = options.paw_aging_tok_per_s
    if not math.isfinite(aging) or aging < 0:
        raise ValueError(
            f"additional_config.{SECTION}.paw_aging_tok_per_s must be a "
            f"finite number >= 0, got {aging!r}"
        )
    if aging > 0 and not options.prefix_aware:
        raise ValueError(
            f"additional_config.{SECTION}.paw_aging_tok_per_s ages the "
            "prefix_aware orders; set prefix_aware to true or leave it at 0"
        )
    return options


def get(vllm_config: Any) -> PPSchedulerOptions:
    """The engine's pipeline chunk scheduler options."""
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        return PPSchedulerOptions()
    return parse(additional.get(SECTION))
