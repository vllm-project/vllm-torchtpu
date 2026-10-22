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
"""Integer exact oracle: binary significands, exact addition and ties-even division."""

import random

import numpy as np


def decode(bits):
    sign = -1 if bits & 0x8000 else 1
    e = (bits >> 7) & 255
    m = bits & 127
    if e == 255:
        raise ValueError("nonfinite")
    return sign * (m if e == 0 else 128 + m), (-133 if e == 0 else e - 134)


def round_bits(n, e):
    if not n:
        return 0
    sign = 0x8000 if n < 0 else 0
    n = abs(n)
    unbiased = n.bit_length() - 1 + e
    quantum = max(unbiased - 7, -133)
    shift = quantum - e
    if shift > 0:
        q, r = divmod(n, 1 << shift)
        half = 1 << (shift - 1)
        q += r > half or (r == half and q % 2 == 1)
    else:
        q = n << (-shift)
    if q == 0:
        return sign
    if q >= 256:
        q >>= 1
        quantum += 1
    if quantum + 7 > 127:
        return sign | 0x7F80
    if quantum == -133 and q < 128:
        return sign | q
    return sign | ((quantum + 134) << 7) | (q - 128)


def fma_bits(a, b, c):
    an, ae = decode(a)
    bn, be = decode(b)
    cn, ce = decode(c)
    pe = ae + be
    e = min(pe, ce)
    n = (an * bn << (pe - e)) + (cn << (ce - e))
    if n == 0 and an * bn == 0 and cn == 0:
        return 0x8000 if ((a ^ b) & c & 0x8000) else 0
    return round_bits(n, e)


def corpus():
    rng = random.Random(1964)
    cases = []
    # Exact midpoint at1: product1/256 and c1. Both tie parities/signs.
    for a in [0x3B80, 0xBB80, 0x3B81, 0xBB81, 0x3B7F, 0xBB7F]:
        for b in [0x3F80, 0xBF80, 0x3F01, 0xBF01]:
            for c in [0x3F80, 0xBF80, 0x3F81, 0xBF81, 0, 0x8000]:
                cases.append((a, b, c))
    for _ in range(16000):
        a = (
            (rng.randrange(2) << 15)
            | (rng.randrange(95, 151) << 7)
            | rng.randrange(128)
        )
        b = (
            (rng.randrange(2) << 15)
            | (rng.randrange(95, 128) << 7)
            | rng.randrange(128)
        )
        c = (
            (rng.randrange(2) << 15)
            | (rng.randrange(95, 151) << 7)
            | rng.randrange(128)
        )
        cases.append((a, b, c))
    boundaries = [
        (a, b, c)
        for a in [0, 0x8000, 1, 0x7F, 0x80, 0x7F7F, 0xFF7F]
        for b in [0, 0x3F00, 0x3F80, 0xBF80]
        for c in [0, 0x8000, 1, 0x80, 0x7F7F]
    ]
    return cases, boundaries


def product_bits(a, b):
    an, ae = decode(a)
    bn, be = decode(b)
    return round_bits(an * bn, ae + be)


def rope_cases():
    rng = random.Random(1964001)
    rows = []
    for _ in range(8192):
        a = (
            (rng.randrange(2) << 15)
            | (rng.randrange(112, 137) << 7)
            | rng.randrange(128)
        )
        b = (
            (rng.randrange(2) << 15)
            | (rng.randrange(112, 137) << 7)
            | rng.randrange(128)
        )
        c = (
            (rng.randrange(2) << 15)
            | (rng.randrange(112, 127) << 7)
            | rng.randrange(128)
        )
        s = (
            (rng.randrange(2) << 15)
            | (rng.randrange(112, 127) << 7)
            | rng.randrange(128)
        )
        rows.append((a, b, c, s))
    return rows


def rope_bits(row, *, target):
    a, b, c, s = row
    first = fma_bits(a, c, product_bits(b, s) ^ 0x8000)
    if target == "inner_b_cos":
        second = fma_bits(a, s, product_bits(b, c))
    elif target == "inner_a_sin":
        second = fma_bits(b, c, product_bits(a, s))
    else:
        raise ValueError(target)
    return first, second


def fp32_value(bits):
    """Decode BF16 as a NumPy FP32 value without production rounding code."""
    return np.asarray(int(bits) << 16, dtype=np.uint32).view(np.float32)[()]


def bf16_bits_from_fp32(value):
    """Round a finite FP32 value via its integer significand, ties to even."""
    bits = int(np.asarray(value, dtype=np.float32).view(np.uint32))
    sign = -1 if bits >> 31 else 1
    exponent, fraction = (bits >> 23) & 255, bits & ((1 << 23) - 1)
    if exponent == 255:
        raise ValueError("oracle expects finite FP32 intermediates")
    if not exponent and not fraction:
        return (bits >> 16) & 0x8000
    return round_bits(
        sign * (fraction if exponent == 0 else (1 << 23) + fraction),
        -149 if exponent == 0 else exponent - 150,
    )


def fp32_multiply_add_bits(a, b, c):
    """Separate NumPy FP32 multiply and add, followed by integer BF16 RNE."""
    product = np.multiply(fp32_value(a), fp32_value(b), dtype=np.float32)
    total = np.add(product, fp32_value(c), dtype=np.float32)
    return bf16_bits_from_fp32(total)


def fp32_rope_bits(row):
    """Independent index-RoPE operation order; no fused single-round oracle."""
    a, b, c, s = row
    # Inner products round to BF16 before the separate outer FP32 operations.
    bs = bf16_bits_from_fp32(
        np.multiply(fp32_value(b), fp32_value(s), dtype=np.float32)
    )
    bc = bf16_bits_from_fp32(
        np.multiply(fp32_value(b), fp32_value(c), dtype=np.float32)
    )
    return (
        fp32_multiply_add_bits(a, c, bs ^ 0x8000),
        fp32_multiply_add_bits(a, s, bc),
    )
