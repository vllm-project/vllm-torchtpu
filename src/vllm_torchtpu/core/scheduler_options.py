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

    --additional-config '{"tpu_scheduler": {"kv_prefetch": true}}'

Every option defaults to main's behaviour without it. The options only change
which request a scheduler admits next and never change a compiled program,
so the section stays out of the compile-cache key
(``HASH_IGNORED_ADDITIONAL_CONFIG_KEYS``). Parse once at init with :func:`get`.
"""

import dataclasses
import typing
from dataclasses import dataclass
from typing import Any

SECTION = "tpu_scheduler"


@dataclass(frozen=True)
class SchedulerOptions:
    promote_warmest: bool = True
    """Promote the warmest request of each waiting queue to its head once per
    step (``PrefixAwareScheduler``). False admits in FCFS order. The TP8
    decode legs of the validation entries run false; p01's runs the
    default."""

    kv_prefetch: bool = False
    """Start remote KV loads for the head of ``waiting`` while the running set
    is full (``PrefixAwareScheduler`` on a KV consumer with a connector;
    refused elsewhere), holding at most 8 requests and keeping 512 blocks for
    the running set. Every decode leg of the validation entries sets it."""

    decode_inflight_throttle: bool = False
    """Throttle remote loads (``PrefixAwareScheduler`` on a KV consumer with
    decode context parallelism > 1; refused elsewhere): refuse a new load
    while 16 are in flight, and otherwise reserve the largest of vLLM's own
    async-load reservation, 128 blocks and 8 blocks x (running + in flight).
    The validation entries' DCP 4 decode legs set it."""

    prefix_aware_levers: bool = False
    """Run ``PrefixAwareScheduler``'s decode-leg levers on an engine that is
    not a KV consumer (no KV transfer, a KV producer, or ``kv_both``):
    decoding requests ahead of partial prefills, warmest-first promotion of
    both waiting queues (once per step), the cross-step prefix-hit memo,
    packed prompt ids, the trimmed token-id resend, and divergent hybrid hits
    resolved through a KV connector that has
    ``resolve_divergent_local_hit_tokens`` and claims
    ``supports_divergent_local_hybrid_hits``. A KV consumer runs the levers
    anyway and keeps vLLM's lookup for divergent hits, so there it adds
    nothing. The cost is host-side: the memo's prefix-cache change log, at
    most 65536 entries."""

    hybrid_state_retention: bool = False
    """Keep the GDN state snapshots a later turn of a conversation hits on
    (``core/hybrid_retention.py``) in ``PrefixAwareScheduler``, on any engine
    serving a Qwen3.5 MoE model with prefix caching on. Its prefill
    checkpoint stops need ``--mamba-cache-mode align``, under which vLLM
    splits prefill chunks at Mamba blocks. It pins up to 360 Mamba blocks of
    HBM, which an allocation that runs short evicts. Elsewhere (another
    model, prefix caching off) it logs that retention is off and changes
    nothing. The pipeline chunk scheduler (PP > 1) does not read it and keeps
    snapshots on a KV producer only."""


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
    hints = typing.get_type_hints(SchedulerOptions)
    fields = {f.name: hints[f.name] for f in dataclasses.fields(SchedulerOptions)}
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
    additional = vllm_config.additional_config
    if not isinstance(additional, dict):
        return SchedulerOptions()
    return parse(additional.get(SECTION))
