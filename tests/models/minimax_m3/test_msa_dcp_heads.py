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
"""`msa_dcp_heads.attend_and_merge` (DCP 8 over 4 KV heads, 4-bit caches)
on 8 host devices against every rank's `msa_dcp.msa_attend_dcp` merged in
f32 (interpret mode). Needs 8 host devices:
``XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest ...``."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention,
    msa_dcp,
    msa_dcp_heads,
    msa_fp4,
)

PAGE, HEAD_DIM, TOPK, DCP, HEADS, Q_PER_KV = 128, 128, 16, 8, 4, 16
REPLICAS = DCP // HEADS
HL = HEADS * Q_PER_KV // DCP  # query heads per rank
SM_SCALE = HEAD_DIM**-0.5


def _step(rng, q_lens, ctx_lens, pages_per_seq, num_tokens):
    num_seqs = len(q_lens)
    max_num_seqs = num_seqs + 1
    num_real = sum(q_lens)
    num_pages = max_num_seqs * pages_per_seq + 1
    block_tables = rng.permutation(np.arange(1, num_pages))[
        : max_num_seqs * pages_per_seq
    ].astype(np.int32)
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[:num_seqs] = np.array(ctx_lens) + np.array(q_lens)
    cu = np.full(max_num_seqs + 1, num_real, np.int32)
    cu[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    dist = np.array([0, 0, num_seqs], np.int32)
    meta = tuple(jnp.asarray(a) for a in (seq_lens, block_tables, cu, dist))

    kvs, scs, cands = [], [], []
    x = [
        jnp.asarray(rng.standard_normal((num_tokens, HEADS, HEAD_DIM)), jnp.bfloat16)
        for _ in range(3)
    ]
    ik = jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM)), jnp.bfloat16)
    for rank in range(DCP):
        kv = msa_fp4.pack_rows(
            jnp.asarray(
                rng.integers(0, 16, (num_pages, PAGE, HEADS * 2 * HEAD_DIM)), jnp.uint8
            )
        )
        index = msa_fp4.pack_rows(
            jnp.asarray(rng.integers(0, 16, (num_pages, PAGE, HEAD_DIM)), jnp.uint8),
            jnp.asarray(rng.uniform(0.5, 1.5, (num_pages, HEAD_DIM)), jnp.float32),
        )
        sc = jnp.asarray(
            rng.uniform(0.5, 1.5, (num_pages, msa_fp4.scale_rows(HEADS), PAGE)),
            jnp.float32,
        )
        kv, _, sc, cand = msa_dcp.msa_write_and_candidates_dcp(
            x[0],
            x[1],
            x[2],
            ik,
            kv,
            index,
            *meta,
            topk=TOPK,
            init_blocks=1,
            local_blocks=1,
            rank=rank,
            dcp=DCP,
            scale_cache=sc,
            interpret=True,
        )
        kvs.append(kv), scs.append(sc), cands.append(cand)
    q = [
        jnp.asarray(rng.standard_normal((num_tokens, HL, HEAD_DIM)), jnp.bfloat16)
        for _ in range(DCP)
    ]
    return q, kvs, scs, cands, meta, num_real


def _reference(q, kvs, scs, cands, meta):
    q_all = jnp.concatenate(q, axis=1)
    cand_all = jnp.stack(cands)
    outs, lses = [], []
    for rank in range(DCP):
        out, lse = msa_dcp.msa_attend_dcp(
            q_all,
            kvs[rank],
            cand_all,
            *meta[:3],
            sm_scale=SM_SCALE,
            topk=TOPK,
            rank=rank,
            dcp=DCP,
            scale_cache=scs[rank],
            interpret=True,
        )
        outs.append(out.astype(jnp.float32)), lses.append(lse)
    lses = jnp.stack(lses)  # [dcp, H, T]
    w = jnp.nan_to_num(jnp.exp(lses - jax.nn.logsumexp(lses, axis=0)))
    merged = jnp.sum(w[..., None] * jnp.stack(outs), axis=0)  # [H, T, D]
    merged = msa_fp4.unrotate_output(merged).transpose(1, 0, 2)
    return [merged[:, r * HL : (r + 1) * HL] for r in range(DCP)]


def _spmd(q, kvs, scs, cands, meta, **kw):
    mesh = Mesh(np.asarray(jax.devices()[:DCP]), ("dcp",))
    body = functools.partial(
        msa_dcp_heads.attend_and_merge,
        axis="dcp",
        dcp=DCP,
        kv_head_replicas=REPLICAS,
        sm_scale=SM_SCALE,
        topk=TOPK,
        rs_out_precision=False,
        interpret=True,
        **kw,
    )
    sharded, rep = P("dcp"), P()
    f = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(sharded,) * 5 + (rep,) * 4,
        out_specs=sharded,
        check_vma=False,
    )
    ranks = jnp.arange(DCP, dtype=jnp.int32)
    out = f(ranks, *(jnp.concatenate(a) for a in (q, kvs, scs, cands)), *meta)
    return [
        o.reshape(-1, HL, HEAD_DIM) for o in np.split(np.asarray(out, np.float32), DCP)
    ]


@pytest.mark.parametrize(
    "q_lens,ctx_lens,num_tokens,min_tokens,max_call,width",
    [
        # Two decodes and two prefills: the prefills (9 + 4 logical blocks)
        # take two rounds of a 12-block buffer.
        ([8, 8, 200, 100], [3000, 5000, 8800, 3900], 320, 64, 2048, 12),
        # Kernel calls of 48 rows: several windows per round, the last one
        # clamped back over rows already attended.
        ([8, 8, 200, 100], [3000, 5000, 8800, 3900], 320, 64, 48, 12),
        # An odd block-table width: the replica-split buffer pads to 12.
        ([8, 8, 200, 100], [3000, 5000, 8800, 3900], 320, 64, 2048, 11),
        # A decode after a prefill: the decode rows are no prefix of the step.
        ([8, 200, 8, 100], [3000, 8800, 5000, 3900], 320, 64, 2048, 12),
        # 384 tokens: the replicas take alternate 64-token blocks
        # (`_interleave`), over two rounds and over 48-row windows.
        ([8, 8, 200, 100], [3000, 5000, 8800, 3900], 384, 64, 2048, 12),
        ([8, 8, 200, 100], [3000, 5000, 8800, 3900], 384, 64, 48, 12),
        # The same step below the head-parallel size: DCP attention only.
        ([8, 8, 200, 100], [3000, 5000, 8800, 3900], 320, 1 << 20, 2048, 12),
    ],
)
def test_attend_and_merge_matches_dcp_reference(
    q_lens, ctx_lens, num_tokens, min_tokens, max_call, width, monkeypatch
):
    if len(jax.devices()) < DCP:
        pytest.skip(
            "needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)"
        )
    monkeypatch.setattr(msa_attention, "MAX_TOKENS_PER_CALL", max_call)
    rng = np.random.default_rng(0)
    step = _step(rng, q_lens, ctx_lens, width, num_tokens)
    *args, num_real = step
    want = _reference(*args)
    got = _spmd(*args, decode_rows=8, head_parallel_min_tokens=min_tokens)
    for r in range(DCP):
        np.testing.assert_allclose(
            got[r][:num_real], np.asarray(want[r])[:num_real], rtol=2e-2, atol=2e-2
        )


@pytest.mark.parametrize("live,lanes", [(0, 1), (5, 1), (11, 1), (12, 1), (5, 0)])
def test_exchange_head_pages(live, lanes):
    """`lanes = 0`: the served (lane, dcp) mesh, with a lane axis of size 1."""
    if len(jax.devices()) < DCP:
        pytest.skip(
            "needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)"
        )
    rng = np.random.default_rng(live)
    num_pages, buffer, width = 20, 12, 256
    caches = rng.integers(0, 256, (DCP, num_pages, 2, 4, HEADS * width), np.uint8)
    pages = rng.integers(0, num_pages, (DCP, buffer)).astype(np.int32)
    devices = np.asarray(jax.devices()[:DCP])
    mesh = Mesh(devices, ("dcp",)) if lanes else Mesh(devices[None], ("lane", "dcp"))
    body = functools.partial(
        msa_dcp_heads.exchange_head_pages,
        axis="dcp",
        dcp=DCP,
        replicas=REPLICAS,
        head_width=width,
        interpret=True,
    )
    f = jax.shard_map(
        lambda c, p, n, r: body(c, p[0], n, r)[None],
        mesh=mesh,
        in_specs=(P("dcp"), P("dcp"), P(), P("dcp")),
        out_specs=P("dcp"),
        check_vma=False,
    )
    got = np.asarray(
        f(
            jnp.asarray(caches.reshape(DCP * num_pages, 2, 4, -1)),
            jnp.asarray(pages),
            jnp.full((1,), live, jnp.int32),
            jnp.arange(DCP, dtype=jnp.int32),
        )
    )
    part = buffer // REPLICAS
    for rank in range(DCP):
        head = rank // REPLICAS
        for j in range(REPLICAS):
            n = int(msa_dcp_heads.live_part(live, j, REPLICAS, part))
            for src in range(DCP):
                want = caches[src, pages[src, j * part : j * part + n]]
                want = want[..., head * width : (head + 1) * width]
                np.testing.assert_array_equal(got[rank, j, src, :n], want)


def test_swap_with_row_crosses_bits():
    """A pair swaps 16-bit rows and a 32-bit extra bit for bit (-inf, NaN and
    a subnormal included)."""
    if len(jax.devices()) < 2:
        pytest.skip("needs 2 devices")
    mesh = Mesh(np.asarray(jax.devices()[:2]), ("dcp",))
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((2 * 8, 16, 128)), jnp.bfloat16)
    extra = rng.standard_normal((2 * 16, 40)).astype(np.float32)
    extra[0, 0], extra[17, 3], extra[5, 39] = -np.inf, np.nan, 1e-42
    got_x, got_extra = jax.shard_map(
        lambda a, b: msa_dcp._swap_with_row(a, jnp.asarray(b), "dcp"),
        mesh=mesh,
        in_specs=(P("dcp"), P("dcp")),
        out_specs=(P("dcp"), P("dcp")),
        check_vma=False,
    )(x, jnp.asarray(extra))
    halves = [slice(0, 8), slice(8, 16)], [slice(0, 16), slice(16, 32)]
    for rank, peer in ((0, 1), (1, 0)):
        np.testing.assert_array_equal(
            np.asarray(got_x)[halves[0][rank]].view(np.uint16),
            np.asarray(x)[halves[0][peer]].view(np.uint16),
        )
        np.testing.assert_array_equal(
            np.asarray(got_extra)[halves[1][rank]].view(np.uint32),
            extra[halves[1][peer]].view(np.uint32),
        )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_weigh_partials_matches_xla(dtype):
    """`msa_dcp_heads.weigh_partials` weighs as the XLA formula does (up to
    one rounding of the result)."""
    rng = np.random.default_rng(4)
    out = jnp.asarray(rng.standard_normal((64, 96, 128)), jnp.bfloat16)
    lses = rng.standard_normal((DCP, 96, 64)).astype(np.float32) * 4
    lses[:, 3, 5] = -np.inf  # a token no rank attended for head 5
    lses[2, 7, 1] = -np.inf
    lses = jnp.asarray(lses)
    lse = lses[3]
    got = msa_dcp_heads.weigh_partials(out, lse, lses, dtype, interpret=True)
    weight = jnp.exp(lse - jax.nn.logsumexp(lses, axis=0)).T[..., None]
    want = jnp.where(weight > 0, out.astype(jnp.float32) * weight, 0.0).astype(dtype)
    tol = 1e-5 if dtype == jnp.float32 else 1e-2
    np.testing.assert_allclose(
        np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=tol, atol=1e-6
    )


@pytest.mark.parametrize("num_tokens", [512, 1500])
def test_unrotate_to_rows_matches_unrotate_output(num_tokens):
    """Bit for bit, including a partial last block of tokens."""
    rng = np.random.default_rng(num_tokens)
    x = jnp.asarray(rng.standard_normal((HL, num_tokens, HEAD_DIM)), jnp.bfloat16)
    got = msa_dcp_heads.unrotate_to_rows(x, interpret=True)
    want = msa_fp4.unrotate_output(x.transpose(1, 0, 2)).reshape(num_tokens, -1)
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )


def test_pair_first_reduce_scatter_matches_psum_scatter():
    """The two-exchange reduce-scatter of a DCP group of 4 keeps block r on
    rank r, summed over the group as `psum_scatter` sums it (f32 sums)."""
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("dcp",))
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal((4 * 16, 24, 128)), jnp.float32)
    ranks = jnp.arange(4, dtype=jnp.int32)

    def body(fn):
        return jax.shard_map(
            fn, mesh=mesh, in_specs=(P("dcp"), P("dcp")), out_specs=P("dcp"),
            check_vma=False,
        )(x, ranks)

    got = body(lambda a, r: msa_dcp_heads._pair_first_reduce_scatter(a, "dcp", r[0]))
    want = body(
        lambda a, r: jax.lax.psum_scatter(a, "dcp", scatter_dimension=0, tiled=True)
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
