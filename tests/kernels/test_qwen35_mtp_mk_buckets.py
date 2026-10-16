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
"""MTP draft megakernel buckets at a running cap whose steps arrive padded (CPU).

With 25 to 28 requests the runner pads a draft loop step to 32 rows and a draft
step 0 to 128 rows; the draft megakernel keeps the buckets that hold the padded
steps, and the padded rows leave the live rows' outputs and the KV pool unchanged.
"""

from __future__ import annotations

import dataclasses
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from qwen35_megakernel_test_utils import (
    jnp_leaves,
    make_tiny_backbone,
    setup_8_cpu_devices,
)

from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K
from vllm_torchtpu.kernels.qwen35_megakernel import mtp_adapter as MA
from vllm_torchtpu.kernels.qwen35_megakernel import pool_io
from vllm_torchtpu.kernels.qwen35_megakernel.vllm_adapter import pack_weights

# COMPILE_SIZES_DECODE of the AgentX decode leg: the runner's token paddings.
PADDINGS = (4, 8, 16, 24, 32, 48, 64, 96, 128, 256)
LOOP = MA._DEFAULT_LOOP_BUCKETS
VERIFY = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)


def test_covering_buckets_keep_the_padded_step():
    assert MA._padded(28, PADDINGS) == 32
    assert MA._padded(112, PADDINGS) == 128
    assert MA._padded(300, PADDINGS) == 300
    # cap 28: 28 requests arrive as a 32-row loop step
    assert MA._covering_buckets(LOOP, 28, PADDINGS) == (4, 8, 12, 16, 20, 24, 28, 32)
    # cap 24 pads to 24: the list is what the running cap alone gives
    assert MA._covering_buckets(LOOP, 24, PADDINGS) == (4, 8, 12, 16, 20, 24)
    assert MA._covering_buckets(VERIFY, 96, PADDINGS) == VERIFY
    assert MA._covering_buckets(VERIFY, 112, PADDINGS) == VERIFY
    # no paddings known: the cap itself
    assert MA._covering_buckets(LOOP, 28, ()) == (4, 8, 12, 16, 20, 24, 28)
    # a cap between two buckets keeps the next one
    assert MA._covering_buckets((4, 8, 16), 10, ()) == (4, 8, 16)
    assert MA._covering_buckets((8, 16), 2, ()) == (8,)


def _bare_mtp(max_reqs: int, paddings=PADDINGS) -> MA.Qwen35MtpMegakernel:
    mk = object.__new__(MA.Qwen35MtpMegakernel)
    mk.bound = True
    mk.layer_name = "mtp.layers.0.self_attn.attn"
    mk.tp_rank = 0
    mk._paddings = tuple(paddings)
    mk.loop_buckets = MA._covering_buckets(LOOP, max_reqs, mk._paddings)
    mk.verify_buckets = MA._covering_buckets(VERIFY, 4 * max_reqs, mk._paddings)
    mk._step_mix, mk._step_mix_t0, mk._step_mix_period_s = {}, time.monotonic(), 1e9
    return mk


def _md(mk, knr: int):
    return {
        mk.layer_name: SimpleNamespace(
            seq_lens=torch.zeros(knr, dtype=torch.int32),
            query_start_loc=torch.zeros(knr + 1, dtype=torch.int32),
            block_tables=torch.zeros((knr, 8), dtype=torch.int32),
        )
    }


def test_use_for_step_takes_padded_steps_at_cap_28():
    mk = _bare_mtp(28)
    md = _md(mk, 28)
    assert mk.use_for_step(md, 32, step0=False)  # 28 requests padded to 32 rows
    assert not mk.use_for_step(md, 48, step0=False)
    assert mk.use_for_step(md, 128, step0=True)  # 28 x 4 = 112 rows padded to 128
    assert not mk.use_for_step(md, 129, step0=True)
    md7 = _md(mk, 7)
    assert mk.use_for_step(md7, 32, step0=True)  # 7 x 4 = 28 rows padded to 32
    assert not mk.use_for_step(md7, 48, step0=True)
    md16 = _md(mk, 16)
    assert not mk.use_for_step(md16, 80, step0=True)  # 64 rows pad to 64, not 80
    assert {k for k in mk._step_mix} == {
        "mk_loop_n32",
        "xla_loop_n48",
        "mk_s0_n128",
        "xla_s0_n129",
        "mk_s0_n32",
        "xla_s0_n48",
        "xla_s0_n80",
    }


def test_use_for_step_cap_24_unchanged():
    mk = _bare_mtp(24)
    assert mk.loop_buckets[-1] == 24
    md = _md(mk, 24)
    assert mk.use_for_step(md, 24, step0=False)
    assert not mk.use_for_step(md, 32, step0=False)
    assert mk.use_for_step(md, 96, step0=True)
    assert not mk.use_for_step(md, 128, step0=True)


def test_padded_rows_leave_live_rows_and_pool_unchanged():
    """The draft step-0 operands of 2 requests in a 2-sequence bucket and in a
    4-sequence bucket whose rows past the requests' 8 hold other values: same
    output rows, same pool."""
    c8 = dataclasses.replace(
        K.TINY_EP,
        R=1,
        B=8,
        T=4,
        L=1,
        att_every=1,
        NH=1,
        NKV=1,
        KH=1,
        VH=2,
        I=128,
        IS=128,
        E=128,
        TOPK=2,
        e_slots=128,
        paged_kv=True,
        num_pages=40,
        page_size=128,
        max_pages_per_seq=8,
        ctx=1024,
    )
    c16 = dataclasses.replace(c8, B=16)
    tiles = dataclasses.replace(K.default_tiles(c8), kv_blk=512, kv_sub=256)
    torch.manual_seed(0)
    w = jnp_leaves(pack_weights(make_tiny_backbone(c8, e_slots=c8.EW), c8, 0, tiles))
    mesh = Mesh(np.array(setup_8_cpu_devices()[:1]), ("tp",))
    skip = frozenset({"experts"})

    rng = np.random.default_rng(0)
    pool_u8 = rng.integers(
        0, 0x40, (c8.num_pages, 2, 64, 4, c8.page_size), dtype=np.uint8
    )
    pool_u8 |= (rng.integers(0, 2, pool_u8.shape, dtype=np.uint8) << 7).astype(np.uint8)
    pool0 = jax.lax.bitcast_convert_type(jnp.asarray(pool_u8), K.FP8)
    block_tables = jnp.asarray([list(range(8, 16)), list(range(16, 24))], jnp.int32)
    seq_lens = jnp.asarray([304, 704], jnp.int32)
    qsl = jnp.asarray([0, 4, 8], jnp.int32)
    gather = jnp.asarray([3, 7], jnp.int32)
    hid = jax.random.normal(jax.random.PRNGKey(1), (16, c8.H), jnp.float32).astype(
        jnp.bfloat16
    )

    outs = {}
    for c, rows in ((c8, 8), (c16, 16)):
        ops = MA._step_operands(c, hid[:rows], block_tables, seq_lens, qsl, gather)
        pos, h, pt, live, moe_rows = ops
        fn, _, _, _ = pool_io.make_flat_attn_step(
            mesh, c, tiles, interpret=True, skip=skip
        )
        out, pool = fn(pos, h, *w, pool0, pt, live, moe_rows)
        outs[rows] = (
            np.asarray(out[:8], np.float32),
            np.asarray(jax.lax.bitcast_convert_type(pool, jnp.uint8)),
            np.asarray(moe_rows),
        )
    (o8, p8, m8), (o16, p16, m16) = outs[8], outs[16]
    assert m8.tolist() == [0, 0, 0, 1, 0, 0, 0, 1]
    assert m16.tolist() == m8.tolist() + [0] * 8
    assert np.isfinite(o8).all()
    np.testing.assert_allclose(o16, o8, rtol=2e-2, atol=2e-2)
    assert np.array_equal(p16, p8)
    assert not np.array_equal(p8, pool_u8)
