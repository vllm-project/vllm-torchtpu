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
"""Packed per-token bitmasks: token ``t`` is bit ``t % 32`` of ``int32`` word
``t // 32``, so a row over a vocabulary of ``V`` ids has ``cdiv(V, 32)``
words."""

import torch


def unpack_token_bitmask(
    bitmask: torch.Tensor, shifts: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """Bool ``[rows, num_tokens]`` mask, ``True`` where the token's bit is set.

    ``bitmask`` is ``[rows, cdiv(vocab_size, 32)]`` ``int32`` and ``shifts`` is
    ``torch.arange(32, dtype=torch.int32)`` on ``bitmask.device``.
    ``num_tokens`` must be at most ``32 * bitmask.shape[1]``.
    """
    # (rows, words, 1) >> (1, 1, 32) broadcasts to (rows, words, 32).
    bits = torch.bitwise_right_shift(bitmask[:, :, None], shifts[None, None, :]) & 1
    return (bits != 0).reshape(bitmask.shape[0], -1)[:, :num_tokens]
