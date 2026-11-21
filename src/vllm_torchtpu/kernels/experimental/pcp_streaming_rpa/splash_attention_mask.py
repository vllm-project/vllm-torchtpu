# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

"""Mini-mask creation library.

Source provenance:
  Ported from https://github.com/openxla/tokamax
  (`tokamax/_src/ops/splash_attention/splash_attention_mask.py`),
  commit `5dff06a967d941abc5035edff68766e22a315430` (2026-10-20).
  Local modifications vs upstream Tokamax:
  - Renamed `_ComputableMask` to `ComputableMask` for direct type checks in
    `splash_attention_mask_info`.
  - Pruned unused mask classes and helpers (`LocalMask`, `ChunkedCausalMask`,
    `LogicalOr`, `LogicalAnd`, `Mask.__or__`, `Mask.__and__`,
    `make_causal_mask`, `make_local_attention_mask`,
    `make_chunk_attention_mask`, `make_random_mask`).
  - Removed legacy mypy error-suppression directive and applied repository
    formatting and lint cleanup.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np


class Mask:
    """A base class for splash attention masks."""

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def __getitem__(self, idx) -> np.ndarray:
        raise NotImplementedError

    def __bool__(self) -> bool:
        raise NotImplementedError(
            "Conversion to bool is unsupported. Could be caused by using logical"
            " instead of bitwise operations on masks."
        )


class ComputableMask(Mask):
    """Superclass for all masks that can be computed inside the kernel.

    Uses a callable object.

    This subclass is designed to be used with Splash Attention.
    It allows the mask logic to be computed on-the-fly or fused into the attention
    kernel, avoiding the memory cost of materializing the full
    (sequence_length, sequence_length) boolean mask array, which can be excessive
    for long sequences.

    Attributes:
      _shape: Shape of the 2-dim mask: (q_seq_len, kv_seq_len).
      offset: Offset of q start wrt kv. A positive offset shifts the bottom
        triangle upward, a negative one shifts it downward. A negative offset
        makes the first 'offset' rows of the attention matrix all 0s which leads
        to undefined softmax.
      q_sequence: Indices of Q sequence. q_sequence is reused across __getitem__
        calls which is important for compile-time performance.
      mask_function: Function used by the SplashAttention kernel to compute the
        mask rather than loading it.
    """

    _shape: tuple[int, int]
    q_sequence: np.ndarray
    mask_function: Callable[..., Any]

    def __init__(
        self,
        shape: tuple[int, int],
        mask_function: Callable[..., Any],
        shard_count: int = 1,
    ):
        self._shape = shape
        self.mask_function = mask_function
        q_seq_len = self.shape[0]

        if q_seq_len % (shard_count * shard_count) != 0:
            raise ValueError(
                f"Shard count squared ({shard_count * shard_count}) must"
                f" divide Q seq_len ({self.shape[0]}) evenly."
            )

        self.q_sequence = np.arange(q_seq_len, dtype=np.int32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __getitem__(self, idx) -> np.ndarray:
        if len(idx) != 2:
            raise NotImplementedError(f"Unsupported slice: {idx}")

        q_slice, kv_slice = idx
        if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
            raise NotImplementedError(f"Unsupported slice: {idx}")

        q_slice = _fill_slice(q_slice, self.shape[0])
        kv_slice = _fill_slice(kv_slice, self.shape[1])

        rows = self.q_sequence[q_slice]
        cols = np.arange(kv_slice.start, kv_slice.stop)

        return self.mask_function(rows[:, None], cols[None, :])

    def __eq__(self, other: object):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()


class CausalMask(ComputableMask):
    """Lazy causal mask, prevents the model from attending to future tokens.

    Attributes:
      offset: Offset of q start wrt kv. A positive offset shifts the bottom
        triangle upward, a negative one shifts it downward. A negative offset
        makes the first 'offset' rows of the attention matrix all 0s which leads
        to undefined softmax.
    """

    offset: int

    def __init__(
        self,
        shape: tuple[int, int],
        offset: int = 0,
        shard_count: int = 1,
    ):
        self.offset = offset

        def causal_mask_function(q_ids, kv_ids):
            # When evaluating the mask in process_mask we typically work with numpy
            # array views.
            # Avoid the addition when possible to avoid instantiating an actual array.
            if self.offset == 0:
                return q_ids >= kv_ids
            else:
                return q_ids + self.offset >= kv_ids

        mask_function = causal_mask_function

        super().__init__(
            shape=shape,
            mask_function=mask_function,
            shard_count=shard_count,
        )

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.shape == other.shape
            and self.offset == other.offset
            and np.array_equal(self.q_sequence, other.q_sequence)
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self.shape,
                self.offset,
                self.q_sequence.tobytes() if self.q_sequence is not None else None,
            )
        )


@dataclasses.dataclass(slots=True)
class NumpyMask(Mask):
    """A mask backed by a dense numpy array."""

    array: np.ndarray

    def __post_init__(self):
        if self.array.ndim != 2:
            raise ValueError("Expected a 2-dim array")

        if self.array.dtype != np.bool_:
            raise ValueError("Mask must be a boolean array")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def __getitem__(self, idx) -> np.ndarray:
        return self.array[idx]

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented

        return np.array_equal(self.array, other.array, equal_nan=True)

    def __hash__(self):
        return hash((type(self), self.array.tobytes()))


def _fill_slice(inp_slice: slice, size: int) -> slice:
    assert inp_slice.step is None or inp_slice.step == 1
    start = 0 if inp_slice.start is None else inp_slice.start
    stop = size if inp_slice.stop is None else inp_slice.stop
    assert start >= 0
    assert stop <= size
    return slice(start, stop, None)


@dataclasses.dataclass(frozen=True, slots=True)
class FullMask(Mask):
    """Lazy full mask, allows all tokens to attend to all other tokens."""

    # TODO: Transform FullMask into a _ComputableMask.

    _shape: tuple[int, int]

    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            raise ValueError(f"Unsupported shape type: {type(self.shape)}")

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __getitem__(self, idx) -> np.ndarray:
        if len(idx) != 2:
            raise NotImplementedError(f"Unsupported slice: {idx}")
        i, j = idx
        if not isinstance(i, slice) or not isinstance(j, slice):
            raise NotImplementedError(f"Unsupported slice: {idx}")
        i = _fill_slice(i, self.shape[0])
        j = _fill_slice(j, self.shape[1])
        return np.ones((i.stop - i.start, j.stop - j.start), dtype=np.bool_)

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.shape == other.shape

    def __hash__(self):
        return hash((type(self), self.shape))
