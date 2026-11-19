# Copyright 2025 Google LLC
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
"""Uniform draws a compiled graph can take: a counter-based hash.

An eager `torch.rand` on a `torch.Generator` is its own program, and its
output is written to HBM and read back by the sampler. A graph cannot take
the generator as an input, but it can take a device int32 counter. Each draw
hashes (counter, element index) with two keyed rounds of the murmur3 32-bit
finalizer. The finalizer is a bijection of the 32-bit words, so the words of
one draw are distinct, and each counter value keys an independent-looking
draw. The top 24 bits become the float32 uniform in [0, 1), the resolution
`torch.rand` has. Every TP rank holds the same counter and runs the same
programs, so the ranks draw the same uniforms.
"""

import torch

_FMIX_C1 = -2048144789  # 0x85EBCA6B as int32
_FMIX_C2 = -1028477387  # 0xC2B2AE35 as int32
_GOLDEN = -1640531527  # 0x9E3779B9 as int32
_KEY2_SALT = 0x5BD1E995


def _fmix32(x: torch.Tensor) -> torch.Tensor:
    """murmur3 fmix32 on int32 words (wrapping multiply, logical shifts)."""
    x = x ^ ((x >> 16) & 0xFFFF)
    x = x * _FMIX_C1
    x = x ^ ((x >> 13) & 0x7FFFF)
    x = x * _FMIX_C2
    x = x ^ ((x >> 16) & 0xFFFF)
    return x


def counter_uniform(shape: tuple[int, ...], counter: torch.Tensor) -> torch.Tensor:
    """float32 uniforms in [0, 1) of ``shape`` for one int32 counter value
    (a 0-d tensor)."""
    numel = 1
    for d in shape:
        numel *= d
    k1 = _fmix32(counter * _GOLDEN)
    k2 = _fmix32(counter ^ _KEY2_SALT)
    idx = torch.arange(numel, dtype=torch.int32, device=counter.device)
    h = _fmix32(_fmix32(idx ^ k1) + k2)
    u = ((h >> 8) & 0xFFFFFF).to(torch.float32) * (1.0 / 16777216.0)
    return u.view(shape)


def initial_counter(seed: int) -> int:
    """The counter's start value for a seed, as an int32."""
    x = (int(seed) * 0x9E3779B9 + 0x7F4A7C15) & 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x
