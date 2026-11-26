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

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

EXPERT_FAMILIES = ("expert_gu", "expert_gus", "expert_down", "expert_ds")
PREFILL_GROUP_BANK_SIZES = (3,) + (8,) * 11 + (1,)
SHARED_BANK_SIZES = (43, 49)


def bank_name(family: str, bank: int) -> str:
    return f"{family}#bank{bank}"


def bank_of_slot(bank_sizes: Sequence[int], slot: int) -> tuple[int, int]:
    start = 0
    for bank, size in enumerate(bank_sizes):
        if slot < start + size:
            return bank, slot - start
        start += size
    raise ValueError(f"MoE slot {slot} is beyond the {start} banked slots")


def group_bank(bank_sizes: Sequence[int], group: int) -> int:
    start = sum(PREFILL_GROUP_BANK_SIZES[:group])
    last = start + PREFILL_GROUP_BANK_SIZES[group] - 1
    bank, _ = bank_of_slot(bank_sizes, start)
    if bank_of_slot(bank_sizes, last)[0] != bank:
        raise ValueError(f"prefill group {group} straddles banks {tuple(bank_sizes)}")
    return bank


def bank_expert_parts(
    iterator: Iterator[tuple[str, str, Any]], bank_sizes: Sequence[int]
) -> Iterator[tuple[str, str, Any]]:
    state: dict[str, list[Any]] = {}
    for group, name, value in iterator:
        family, marker, suffix = name.partition("@")
        if family not in EXPERT_FAMILIES:
            yield group, name, value
            continue
        part, parts = map(int, suffix.split("/")) if marker else (0, 1)
        entry = state.setdefault(family, [0, 0, 0, None])
        expected_part, bank, used, buffer = entry
        if part != expected_part:
            raise ValueError(f"Out-of-order expert checkpoint part: {name}")
        offset = 0
        while offset < value.shape[0]:
            if bank == len(bank_sizes):
                raise ValueError(f"Too many expert layers for {family}")
            capacity = bank_sizes[bank]
            if buffer is None:
                buffer = np.empty((capacity, *value.shape[1:]), dtype=value.dtype)
            count = min(capacity - used, value.shape[0] - offset)
            buffer[used : used + count] = value[offset : offset + count]
            used += count
            offset += count
            if used == capacity:
                yield group, bank_name(family, bank), buffer
                bank, used, buffer = bank + 1, 0, None
        state[family] = [part + 1, bank, used, buffer]
        if part + 1 == parts and (bank != len(bank_sizes) or used):
            raise ValueError(f"Incomplete expert checkpoint: {family}")
    if set(state) != set(EXPERT_FAMILIES) or any(
        bank != len(bank_sizes) or used for _, bank, used, _ in state.values()
    ):
        raise ValueError("Missing expert checkpoint parts")
