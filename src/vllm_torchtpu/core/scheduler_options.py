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

from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import config_section

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


def parse(raw: Any) -> SchedulerOptions:
    """The ``tpu_scheduler`` section as options; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``0`` for a bool, say), raises ``ValueError`` (``config_section.parse``).
    """
    return config_section.parse(SchedulerOptions, SECTION, raw)


def get(vllm_config: Any) -> SchedulerOptions:
    """The engine's scheduler options (``config_section.get``)."""
    return config_section.get(SchedulerOptions, SECTION, vllm_config)
