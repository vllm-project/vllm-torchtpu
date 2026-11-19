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
"""q38 L2-UNPACK (X-UA-UNPACK): the union tile body reads its ring slot with full-vreg strided loads.

`VLLM_TPU_MK_UNION_UNPACK` (default 0 = off; read by `qwen38_wholestep_call`, registered, so it keys the compile cache)
selects the indexed block option `union_unpack`, served by `qsa_full_lean`'s lean class (independent of X-UA-ISSUE's
`union_issue` and the D08 S1 options; it composes with all of them). With it on, `union_attn.attend_request`'s tile body
takes its FP8 K / V `[TILE, 256]` from `unpack_kv` instead of `qsa_attn.block._unpack_kv`.

Why: a ring slot is `[TILE, 1, 4, 256]` FP8, each token's 1 KiB row packing `[K0, V0, K1, V1]` (pack row `c` in byte `c`
of one 32-bit word per lane). `_unpack_kv` views the slot as `[TILE, 256]` u32 and loads it whole; that view has one
sublane per token, so the load lowers to 512 single-sublane partial loads (2 per token) plus their sublane re-assembly
(vcombine / vrot.slane) and a u32 -> u16 -> u8 chain per byte (M: L1-D24 census j-eb8c3c4c, D14 union tile loop: vld 821,
512 of them partial, all at block.py:65; relayout 1443; scratch/dhunt7/hunt_4.md). The stock RPAd kernel reads the same
packed rows as 128-lane rows with a sublane stride (batched_rpa kernel.py `strided_load_bkv` + utils.py `strided_load`):
the u32 view flattened to `[2 TILE, 128]` (row `2 t + h` = token `t`, lanes `128 h ...`), lane half `h` of every token is
`pl.ds(h, TILE, stride=2)`, 8 tokens per full-vreg load: 2 x 32 = 64 loads per slot. The slot view is annotated
`no_hazard_no_deps` as batched_rpa does (the DMA wait that precedes the read stays a barrier).

Bytes: the served pool is unpaired (`QWEN4_EXP_PACK_MAIN_KV=0`), so every QSA layer's head pair is pair 0: K = byte 0,
V = byte 1. `unpack_kv` takes only the static offset 0 (asserted); the whole-step call refuses a pool layout with a non-zero
main slot when the option is on (`qwen38_wholestep_call._uaunpack_slots`). The split is batched_rpa's binary tree
(`utils.convert_to_target_bitwidth`, kv_slot 0): the u32 words truncated to u16 (bytes 0-1), K = the low byte, V = the low
byte of the u16 pairs shifted right by 8 (through their u32 view).

Exactness: the same FP8 bytes in the same `[TILE, 256]` order as `_unpack_kv(slot, 0)`, so every later op is unchanged and
the outputs are bitwise (tests/megakernel/test_l2_unpack.py). TPU interpret mode has no ref bitcast / reshape / annotate
(its loads take only NDIndexers), so there the slot is loaded whole and the same strided row selection runs on the value
(`jax.lax.slice_in_dim` with the same start / size / stride as the device `pl.ds`); the byte split is shared.

Credit: the strided read and the byte tree are the stock batched_rpa kernel's (kernels/experimental/batched_rpa
kernel.py:36-80, utils.py:71-90 and :113-152), restated here so the interpret path shares the index arithmetic.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.qsa_attn.block import TILE

F8 = jnp.float8_e4m3fn
LANES = 128  # lanes per vreg row
HALVES = 256 // LANES  # 128-lane rows per token in the u32 view (the strided load's sublane stride)


def check_offset(off: Any) -> None:
    """X-UA-UNPACK reads head pair 0 (K byte 0, V byte 1): only the static offset 0 is accepted."""
    assert isinstance(off, int) and not isinstance(off, bool) and off == 0, (
        f"X-UA-UNPACK reads head pair 0 only (unpaired pool, PACK_MAIN_KV=0); got offset {off!r}"
    )


def ring_words(slot_ref: Any, interpret: bool) -> jax.Array:
    """`[TILE, 256]` u32 words of a `[TILE, 1, 4, 256]` FP8 slot (token `t`, lane `l`: pack rows 0-3 in bytes 0-3),
    read as `HALVES` sublane-strided loads of the slot's 128-lane rows."""
    if interpret:
        raw = slot_ref[...]
        flat = pltpu.bitcast(raw.reshape(-1, raw.shape[-1]), jnp.uint32).reshape(-1, LANES)

        def read(start: int, size: int, stride: int) -> jax.Array:
            return jax.lax.slice_in_dim(flat, start, start + (size - 1) * stride + 1, stride)

    else:
        words_ref = slot_ref.bitcast(jnp.uint32).reshape(-1, slot_ref.shape[-1])
        # As batched_rpa: no memory dependency between the ring's DMA-in and these loads beyond the slot's wait.
        words_ref = pltpu.annotate(words_ref, no_hazard_no_deps=True)
        flat_ref = words_ref.reshape(-1, LANES)

        def read(start: int, size: int, stride: int) -> jax.Array:
            return flat_ref[pl.ds(start, size, stride)]

    # Lane half h of every token: flat rows h, h + HALVES, ... (one full-vreg load per 8 tokens).
    halves = [read(h, TILE, HALVES) for h in range(HALVES)]
    return jnp.concatenate(halves, axis=1)


def unpack_kv(slot_ref: Any, off: Any, interpret: bool) -> tuple[jax.Array, jax.Array]:
    """FP8 K / V `[TILE, 256]` of head pair 0 from a `[TILE, 1, 4, 256]` slot: `qsa_attn.block._unpack_kv(slot, 0)`'s
    values, read with full-vreg strided loads (module docstring)."""
    check_offset(off)
    words = ring_words(slot_ref, interpret)
    pair = words.astype(jnp.uint16)  # bytes 0-1: (K, V) of head pair 0
    k8 = pltpu.bitcast(pair.astype(jnp.uint8), F8)  # byte 0
    hi = pltpu.bitcast(pltpu.bitcast(pair, jnp.uint32) >> 8, jnp.uint16)
    v8 = pltpu.bitcast(hi.astype(jnp.uint8), F8)  # byte 1
    return k8, v8
