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
"""The kernel against the model reference on TPU v7x-8 (skipped elsewhere).

A tiny two-layer geometry (one KV head per rank, as the checkpoint at TP=8)
runs one paged decode step with RoPE from a per-row table at an M-RoPE
position 7 below the KV index, and the ungated attention, bf16 KV pages and
dense FFN through the whole model reference, at the page sizes vLLM picks:
128 tokens, and 512 when its SMEM page-table bound raises the block size. The
in-kernel KV write (`fused_pool_io`) matches the scatter program bitwise, and
the whole-step kernel (`TokenIO`: runner tensors in, greedy tokens out) leaves
the same pages and returns the greedy tokens of the hidden step's output, ties
going to the lowest id.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest  # pyrefly: ignore[missing-import]
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.core import CacheExec, ExecSpec, cache, planner, pool
from vllm_torchtpu.megakernel.core import token_io as tio_mod
from vllm_torchtpu.megakernel.core.modules.gqa.decls import ROPE_TABLE_INPUT
from vllm_torchtpu.megakernel.core.ops import BF16, F32
from vllm_torchtpu.megakernel.core.weights.synthetic import (
    pack_inputs,
    random_logical_inputs,
)

from .. import model
from .tiny import tiny_geom, tiny_tiles

R = 8

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu" or jax.device_count() < R,
    reason="needs a TPU v7x-8",
)


def _rope_table(g: object, pos: jax.Array, mr: int) -> jax.Array:
    """`(2, MR, ROT // 2)` cos / sin of each row's position, bf16-rounded like
    vLLM's rotary cache."""
    half = g.ROT // 2
    inv = g.theta ** (-jnp.arange(half, dtype=F32) / half)
    ang = pos.astype(F32)[:, None] * inv
    cs = jnp.stack([jnp.cos(ang), jnp.sin(ang)]).astype(BF16).astype(F32)
    return jnp.pad(cs, ((0, 0), (0, mr - pos.shape[0]), (0, 0)))


def _check_close(got: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> None:
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("page_size", "kv_blk"),
    [(128, 256), (512, 1024), (512, 256)],
    ids=["p128_blk2pages", "p512_blk2pages", "p512_blk_rounds_up_to_1page"],
)
def test_kernel_matches_model_reference(page_size: int, kv_blk: int) -> None:
    mdef = model.register()
    g = tiny_geom(R=R)
    ex = ExecSpec(
        R=R,
        B=2,
        T=1,
        ctx=4 * page_size,
        tiles=tiny_tiles(kv_blk=kv_blk),
        # 64 MiB pools: XLA stages a pool small enough for VMEM in a copy, which
        # an in-kernel KV write would miss (the serving pools are ~1.2 GB).
        cache=CacheExec(
            paged_kv=True,
            page_size=page_size,
            num_pages=(64 << 20) // (256 * page_size * 2),
            max_pages_per_seq=4,
        ),
    )
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    _, hid, w_l, st_l = random_logical_inputs(
        jax.random.PRNGKey(0), g, ex, model_def=mdef
    )
    # The new tokens land past the first page: slot 7 of page 1, and the last
    # page's slot `page_size - 5`.
    pos = jnp.array([page_size + 7, 4 * page_size - 5], jnp.int32)
    rope_cs = _rope_table(g, pos - 7, ex.MR)
    ref_out, ref_st = mdef.reference_step(
        g, ex, pos, hid, w_l, st_l, extra={ROPE_TABLE_INPUT: rope_cs}
    )

    lay = dataclasses.replace(mdef.pool_layout(g, R=R), page_size=page_size)
    spec = mdef.to_model_spec(g, R=R).with_layout(lay)
    rope_k = jax.device_put(jnp.tile(rope_cs, (R, 1, 1)), NamedSharding(mesh, P("tp")))

    def run(fused: bool) -> tuple[jax.Array, dict]:
        pos_k, hid_k, w_k, st_k = pack_inputs(
            pos,
            hid,
            w_l,
            st_l,
            g,
            ex,
            mesh=mesh,
            build_default_pool=True,
            model_def=mdef,
        )
        step = cache.make_pooled_step(
            mesh, spec, ex, donate_pools=fused, fused_pool_io=fused
        )
        return step(pos_k, hid_k, w_k, st_k, rope_k)

    out, st_out = run(fused=False)
    # The in-kernel KV write (one 128-lane block RMW per layer) leaves the same
    # pools and hidden states as the scatter program.
    out_f, st_f = run(fused=True)
    np.testing.assert_array_equal(np.asarray(out_f), np.asarray(out))
    for a, b in zip(st_f[pool.POOL_STATE_NAME], st_out[pool.POOL_STATE_NAME]):
        np.testing.assert_array_equal(
            np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16)
        )

    _check_token_step(
        g, ex, spec, mesh, pos, hid, w_l, st_l, out_f, st_f[pool.POOL_STATE_NAME]
    )

    out = np.asarray(out, np.float32).reshape(R, ex.M, g.H)[:, : ex.B]
    ref = np.asarray(ref_out, np.float32)
    for r in range(R):
        _check_close(out[r], ref, atol=4e-2, rtol=2e-2)
    cos = (out[0] * ref).sum() / np.linalg.norm(out[0]) / np.linalg.norm(ref)
    assert cos >= 0.9999, cos

    pt, _, _, _ = pool.allocate_test_layout(lay, ex)
    kv = pool.pool_to_states(lay, ex, st_out[pool.POOL_STATE_NAME], pt)
    for k in ("kc", "vc"):
        got = np.asarray(kv[k], np.float32)
        _check_close(got, np.asarray(ref_st[k], np.float32), atol=3.5e-2, rtol=1.5e-2)


def _flat_weights(spec: object, ex: ExecSpec, w: dict) -> list:
    names = planner.plan(spec, cache.with_cache(ex, paged_kv=True)).w_names
    return [x for n in names for x in (w[n] if isinstance(w[n], tuple) else (w[n],))]


def _check_token_step(g, ex, spec, mesh, pos, hid, w_l, st_l, hid_out, pools_ref):
    """The token step on the same inputs: embedding rows hold `hid`, the RoPE
    cache rows hold the table, and the LM head ties two ids across ranks (row
    0) and across tiles of one rank (row 1) above every other logit, with a
    larger one past the vocabulary."""
    shard = 304
    vocab = R * shard - 5
    tio = tio_mod.TokenIO(
        vocab=vocab,
        shard=shard,
        tile=tio_mod.head_tile(g.H, shard, ex.tiles.bank),
        rope_rows=4 * ex.ctx,
        rope_cols=g.ROT,
        rope_dtype=F32,
    )
    assert tio.n_tiles == 2
    ex_t = dataclasses.replace(cache.with_cache(ex, fused_pool_io=True), tokens=tio)
    h = np.asarray(hid_out, np.float32).reshape(R, ex.M, g.H)[0, : ex.B]
    key = jax.random.PRNGKey(5)
    ids = np.array([7, 5 * shard + 9], np.int32)
    emb = jax.random.normal(key, (R * shard, g.H), F32).astype(BF16)
    emb = emb.at[ids].set(hid[: ex.B].astype(BF16))
    head = np.array(jax.random.normal(key, (R * shard, g.H), F32) * 0.05)
    ties = ([3, shard + 1], [2 * shard + 5, 2 * shard + tio.tile + 7])
    for b, rows in enumerate(ties):
        head[rows] = 4 * h[b] / np.linalg.norm(h[b])
    head[R * shard - 2] = 8 * h[0] / np.linalg.norm(h[0])  # padded vocab
    head = jnp.asarray(head, BF16)
    logits = (jnp.asarray(h, BF16).astype(F32) @ head.astype(F32).T)[:, :vocab]
    want = np.asarray(jnp.argmax(logits.astype(BF16).astype(F32), axis=-1))
    np.testing.assert_array_equal(want, [t[0] for t in ties])

    half = g.ROT // 2
    inv = g.theta ** (-jnp.arange(half, dtype=F32) / half)
    ang = jnp.arange(tio.rope_rows, dtype=F32)[:, None] * inv
    rope_cache = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], -1)
    _, _, w_k, st_k = pack_inputs(
        pos,
        hid,
        w_l,
        st_l,
        g,
        ex,
        mesh=mesh,
        build_default_pool=True,
        model_def=model.Qwen3VLText,
    )
    w_k = {
        **w_k,
        tio_mod.EMBED_WEIGHT: emb,
        tio_mod.HEAD_WEIGHT: jnp.concatenate(
            [
                tio_mod.pack_head(head[r * shard : (r + 1) * shard], tio)
                for r in range(R)
            ]
        ),
    }
    # The runner's padding: 3 more token rows, 6 sequences of 6 page slots.
    positions = jnp.pad(jnp.tile(pos - 7, (3, 1)), ((0, 0), (0, 3)))
    tables = np.zeros((ex.NS + 6, 6), np.int32)
    tables[: ex.NS, :4] = np.asarray(st_k["page_table"][: ex.NS])
    seq_lens = np.zeros((ex.NS + 6,), np.int32)
    seq_lens[: ex.NS] = np.asarray(pos) + 1
    step, in_sp, _, _ = cache.make_flat_pooled_step(mesh, spec, ex_t)
    args = [
        jnp.pad(jnp.asarray(ids), (0, 3)),
        positions.astype(jnp.int32),
        *_flat_weights(spec, ex_t, w_k),
        *st_k[pool.POOL_STATE_NAME],
        jnp.asarray(tables.reshape(-1)),
        rope_cache,
        jnp.asarray(seq_lens),
    ]
    outs = step(
        *[jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, in_sp)]
    )
    np.testing.assert_array_equal(np.asarray(outs[0]), want)
    for a, b in zip(outs[1:], pools_ref):
        np.testing.assert_array_equal(
            np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16)
        )
