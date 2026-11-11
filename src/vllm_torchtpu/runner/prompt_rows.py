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
"""Prompt token-id lists backed by a lazily materialized `int32` NumPy array."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


class PromptTokenIds(list):
    """A `list[int]` carrying a read-only `np.int32` mirror invalidated on mutation."""

    __slots__ = ("_array",)

    def __init__(self, ids: Iterable[int] = (), array: np.ndarray | None = None):
        super().__init__(ids)
        if array is not None and len(array) != len(self):
            raise ValueError(f"array of {len(array)} ids for {len(self)} tokens")
        self._array = None if array is None else _read_only(array)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        if self._array is None:
            self._array = _read_only(
                np.fromiter(list.__iter__(self), dtype=np.int32, count=len(self))
            )
        array = self._array
        if dtype is not None and array.dtype != dtype:
            return array.astype(dtype)
        return array.copy() if copy else array

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (list(self),)

    def _changed(self) -> None:
        self._array = None

    def __setitem__(self, key: Any, value: Any) -> None:
        self._changed()
        super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        self._changed()
        super().__delitem__(key)

    def __iadd__(self, other: Iterable[int]) -> PromptTokenIds:
        self._changed()
        return super().__iadd__(other)

    def __imul__(self, n: Any) -> PromptTokenIds:
        self._changed()
        return super().__imul__(n)

    def append(self, value: int) -> None:
        self._changed()
        super().append(value)

    def extend(self, values: Iterable[int]) -> None:
        self._changed()
        super().extend(values)

    def insert(self, index: Any, value: int) -> None:
        self._changed()
        super().insert(index, value)

    def pop(self, index: Any = -1) -> int:
        self._changed()
        return super().pop(index)

    def remove(self, value: int) -> None:
        self._changed()
        super().remove(value)

    def clear(self) -> None:
        self._changed()
        super().clear()

    def sort(self, *args: Any, **kwargs: Any) -> None:
        self._changed()
        super().sort(*args, **kwargs)

    def reverse(self) -> None:
        self._changed()
        super().reverse()


def cache_prompt_rows(
    requests: Mapping[str, Any], input_batch: Any, req_ids: Iterable[str]
) -> None:
    """Wrap newly added text requests' `prompt_token_ids` in lazy `PromptTokenIds`."""
    for req_id in req_ids:
        req_state = requests.get(req_id)
        if req_state is None or getattr(req_state, "mm_features", None):
            continue
        prompt = req_state.prompt_token_ids
        if prompt is None or isinstance(prompt, PromptTokenIds):
            continue
        if input_batch.req_id_to_index.get(req_id) is None:
            continue
        req_state.prompt_token_ids = PromptTokenIds(prompt)
