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

from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import config_section

SECTION = "tpu_pp_scheduler"


@dataclass(frozen=True)
class PPSchedulerOptions:
    paw_aging_tok_per_s: int = 3000
    """Uncached prompt tokens credited per second a request has waited, in the
    ``prefix_aware`` orders (waiting-queue sort, running-prefill order,
    warmest-first promotion), so a large prompt is overtaken for a bounded
    time; 0 turns the aging off. Read only with ``prefix_aware``, so a
    positive value set without it is refused. The default is the value the
    validation entries' prefill legs run (they also set it)."""

    prefix_aware: bool = False
    """Run ``PrefixAwareScheduler``'s admission on a pipeline: the
    waiting-queue sort (on a KV producer), shortest-remaining-prefill-first
    order of running prefills, warmest-first promotion (unless
    ``tpu_scheduler.promote_warmest`` is false), the cross-step prefix-hit
    memo, divergent hybrid hits resolved through the KV connector, the
    trimmed token-id resend, packed prompt ids, and the connector's
    completion poll at the start of each step. The validation entries'
    prefill legs run true."""

    quadratic_cost: bool = False
    """Price prefill work by its attention cost as well as its token count,
    on the equal-cost step cost model: lone-prompt chunking's spread counts
    each pending request's (chunk, prefix) pairs and a waiting request's
    prefix hit, and the ``prefix_aware`` orders rank by effective tokens.
    Needs ``TPU_PP_EQUAL_COST_CHUNKS`` (on by default); the scheduler refuses
    it otherwise. The validation entries' prefill legs run true."""


def parse(raw: Any) -> PPSchedulerOptions:
    """The ``tpu_pp_scheduler`` section as options; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``0`` for a bool, ``true`` or ``3000.0`` for a whole number), raises
    ``ValueError`` (``config_section.parse``), and so does a negative aging
    rate or a positive one set without ``prefix_aware``.
    """
    options = config_section.parse(PPSchedulerOptions, SECTION, raw)
    aging = options.paw_aging_tok_per_s
    if aging < 0:
        raise ValueError(
            f"additional_config.{SECTION}.paw_aging_tok_per_s must be >= 0, "
            f"got {aging!r}"
        )
    set_here = isinstance(raw, dict) and "paw_aging_tok_per_s" in raw
    if not options.prefix_aware and aging > 0 and set_here:
        raise ValueError(
            f"additional_config.{SECTION}.paw_aging_tok_per_s ages the "
            "prefix_aware orders; set prefix_aware to true or leave it at 0"
        )
    return options


def get(vllm_config: Any) -> PPSchedulerOptions:
    """The engine's pipeline chunk scheduler options (``parse``, so the
    checks above apply too)."""
    additional = vllm_config.additional_config
    if not isinstance(additional, dict):
        return PPSchedulerOptions()
    return parse(additional.get(SECTION))
