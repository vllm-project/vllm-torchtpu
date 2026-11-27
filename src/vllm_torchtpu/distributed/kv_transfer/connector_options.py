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
"""Opt-in options of the TPU KV connectors in ``kv_connector_extra_config``.

Each connector reads its options with ``read_options`` into one of the frozen
dataclasses below, whose fields are the option names, types and defaults. A
connector's ``kv_connector_extra_config`` also holds settings that are not
options (``connectors``, ``cpu_bytes_to_use``, ...); their own readers take
them. Every field's annotation must be a class, not a string, so this module
does not use ``from __future__ import annotations``.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True)
class RaidenTransferOptions:
    """``TPURaidenConnector``'s options.

    Attributes:
        async_complete: On a KV producer, issue the post-transfer
            ``complete_request_blocks`` RPC on a background thread, which
            retries a failed call.
        async_register: On a KV producer, register a step's request blocks on
            a side thread that starts once the forward is dispatched. Only the
            default admission path does; DSv4, GLM and Kimi-K3 admission
            register synchronously.
        late_poll: On a KV consumer with async scheduling, report loads that
            land during a step with that step's output.
    """

    async_complete: bool = False
    async_register: bool = False
    late_poll: bool = False


@dataclasses.dataclass(frozen=True)
class RaidenOffloadingOptions:
    """``TPURaidenOffloadingConnector``'s options.

    Attributes:
        host_tier_recency: The scheduler refreshes a request's host-tier keys
            on every lookup and store, probes partial tails only above the
            local hit, offers FullAttention chunks for storing from chunk 0
            again, and launches fence-acknowledged stores largest first.
        max_async_loads: Host-tier loads in flight before a request with a
            host-tier hit is deferred to a later step; 0 is no cap.
    """

    host_tier_recency: bool = False
    max_async_loads: int = 0


@dataclasses.dataclass(frozen=True)
class MultiConnectorOptions:
    """``TPUMultiConnector``'s options.

    Attributes:
        producer_divergent_hits: On a KV producer, report divergent local
            hybrid hits when any child can complete them, instead of
            upstream's every-child rule.
    """

    producer_divergent_hits: bool = False


# The options of each connector, by the class name --kv-transfer-config gives.
CONNECTOR_OPTIONS: dict[str, type] = {
    "TPUMultiConnector": MultiConnectorOptions,
    "TPURaidenConnector": RaidenTransferOptions,
    "TPURaidenOffloadingConnector": RaidenOffloadingOptions,
}


def option_names(options_cls: type) -> frozenset[str]:
    """The option names of ``options_cls``."""
    return frozenset(f.name for f in dataclasses.fields(options_cls))


def read_options[T](options_cls: type[T], extra_config: Mapping[str, Any] | None) -> T:
    """The options of ``options_cls`` in ``extra_config``; defaults for the rest.

    Strict: a value whose type is not exactly the field's (``"1"`` or ``1``
    for a bool, a negative int) raises ``ValueError``, and so does another
    connector's option, which this connector would otherwise ignore. Keys
    that are no connector's option are left to their own readers.
    """
    extra_config = extra_config or {}
    own = option_names(options_cls)
    reader = next(n for n, cls in CONNECTOR_OPTIONS.items() if cls is options_cls)
    for name, other_cls in CONNECTOR_OPTIONS.items():
        misplaced = sorted((option_names(other_cls) - own) & extra_config.keys())
        if misplaced:
            raise ValueError(
                f"{reader}'s kv_connector_extra_config sets {misplaced}: "
                f"options of {name}, which {reader} does not read"
            )
    values: dict[str, Any] = {}
    # dataclasses.fields is typed for DataclassInstance, which a type
    # parameter cannot name; every caller passes a frozen dataclass.
    # pyrefly: ignore[bad-argument-type]
    for field in dataclasses.fields(options_cls):
        if field.name not in extra_config:
            continue
        value = extra_config[field.name]
        want = field.type
        assert isinstance(want, type)  # the annotations are classes, not strings
        if type(value) is not want or (want is int and value < 0):
            kind = "true or false" if want is bool else "a non-negative integer"
            raise ValueError(
                f"kv_connector_extra_config[{field.name!r}] must be {kind}, "
                f"got {value!r}"
            )
        values[field.name] = value
    return options_cls(**values)
