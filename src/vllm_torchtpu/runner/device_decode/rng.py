# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Counter-based sampling uniforms keyed by (seed, position, token id).

Ported from the noise of vLLM's Model Runner V2 sampler, `murmur3_hash32` and
`murmur3_uniform32` in vllm/v1/worker/gpu/sample/gumbel.py (v0.33.0), and
bitwise equal to it. The hash is MurmurHash3_x86_32 with seed 0 of the 16-byte
key (seed low word, seed high word, position, token id). The uniform adds the
hash's two 16-bit halves, each exact in float32, so their sum is
(hash + 0.5) * 2**-32 rounded once. A draw depends only on its key: not on the
batch, its order or padding, the step history, or which path sampled it.

The hash runs on int32, whose multiplications wrap. Right shifts of int32 are
arithmetic, so masks make them logical.
"""

import torch

# MurmurHash3_x86_32's multipliers and addend, as signed int32.
_MIX_MUL_1 = -862048943  # 0xCC9E2D51
_MIX_MUL_2 = 461845907  # 0x1B873593
_MIX_ADD = -430675100  # 0xE6546B64
_FMIX_MUL_1 = -2048144789  # 0x85EBCA6B
_FMIX_MUL_2 = -1028477387  # 0xC2B2AE35
_KEY_BYTES = 16


def _shr(x: torch.Tensor, shift: int) -> torch.Tensor:
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def _rotl(x: torch.Tensor, shift: int) -> torch.Tensor:
    return (x << shift) | _shr(x, 32 - shift)


def _scramble(key: torch.Tensor) -> torch.Tensor:
    return _rotl(key * _MIX_MUL_1, 15) * _MIX_MUL_2


def _mix(h: torch.Tensor, scrambled_key: torch.Tensor) -> torch.Tensor:
    return _rotl(h ^ scrambled_key, 13) * 5 + _MIX_ADD


def _fmix(h: torch.Tensor) -> torch.Tensor:
    h = h ^ _shr(h, 16)
    h = h * _FMIX_MUL_1
    h = h ^ _shr(h, 13)
    h = h * _FMIX_MUL_2
    return h ^ _shr(h, 16)


def _uniform_from_bits(bits: torch.Tensor) -> torch.Tensor:
    hi16 = _shr(bits, 16).to(torch.float32)
    lo16 = (bits & 0xFFFF).to(torch.float32)
    return hi16 * 2.0**-16 + (lo16 + 0.5) * 2.0**-32


def counter_uniform_impl(
    seed_lo: torch.Tensor, seed_hi: torch.Tensor, pos: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """Float32 uniforms [rows, vocab_size] in (0, 1], one per (row, token id).

    `seed_lo` and `seed_hi` [rows] are the int32 bit patterns of the low and
    high words of each row's 64-bit seed; `pos` [rows] int32 is the position
    the row samples at. Equal to vLLM's `murmur3_uniform32(seed, pos, token)`.
    Plain torch ops, which the decode head and the stock path's noise program
    compile into their own programs.
    """
    for name, words in (("seed_lo", seed_lo), ("seed_hi", seed_hi), ("pos", pos)):
        if words.dtype != torch.int32:
            # Wider integers would not wrap at 32 bits.
            raise TypeError(
                f"counter_uniform_impl: {name} must be int32, not {words.dtype}"
            )
    h = _mix(torch.zeros_like(seed_lo), _scramble(seed_lo))
    h = _mix(h, _scramble(seed_hi))
    h = _mix(h, _scramble(pos))
    tokens = torch.arange(vocab_size, dtype=torch.int32, device=pos.device)
    h = _mix(h.unsqueeze(1), _scramble(tokens).unsqueeze(0))
    return _uniform_from_bits(_fmix(h ^ _KEY_BYTES))


def counter_uniform_for_rows_impl(
    seed_lo_slots: torch.Tensor,
    seed_hi_slots: torch.Tensor,
    rows_slot: torch.Tensor,
    rows_pos: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """`counter_uniform_impl` for rows whose seed words sit in slot tensors.

    `seed_lo_slots` / `seed_hi_slots` are [num_slots] int32; `rows_slot` and
    `rows_pos` [rows] int32 give each row's slot and position.
    """
    return counter_uniform_impl(
        torch.index_select(seed_lo_slots, 0, rows_slot),
        torch.index_select(seed_hi_slots, 0, rows_slot),
        rows_pos,
        vocab_size,
    )
