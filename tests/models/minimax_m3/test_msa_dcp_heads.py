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

from tests.models.minimax_m3.packing import pack_index_fp4
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_attention,
    msa_dcp,
    msa_dcp_heads,
    msa_fp4,
    msa_quantize,
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
        index = pack_index_fp4(
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
    q_all = msa_fp4.rotate_queries(jnp.concatenate(q, axis=1))  # the op rotates q
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


def test_block_all_to_all_matches_xla():
    # Interpret mode cannot carry semaphores out of a call.
    if jax.default_backend() != "tpu" or len(jax.devices()) < DCP:
        pytest.skip("needs 8 TPU devices")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1 << 16, (DCP * DCP, 3, 16, 128), np.uint16)
    mesh = Mesh(np.asarray(jax.devices()[:DCP]), ("dcp",))

    def body(x, rank):
        started = msa_dcp_heads.start_block_all_to_all(x, rank, axis="dcp", dcp=DCP)
        got = msa_dcp_heads.finish_block_exchange(started, DCP)
        return got, jax.lax.all_to_all(x, "dcp", 0, 0, tiled=True)

    f = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(P("dcp"), P("dcp")),
        out_specs=(P("dcp"), P("dcp")),
        check_vma=False,
    )
    got, want = f(jnp.asarray(x), jnp.arange(DCP, dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_block_all_gather_matches_xla():
    # Interpret mode cannot carry semaphores out of a call.
    if jax.default_backend() != "tpu" or len(jax.devices()) < DCP:
        pytest.skip("needs 8 TPU devices")
    lanes, ranks = 2, DCP // 2
    w = np.random.default_rng(1).integers(-100, 100, (DCP * 64, 512), np.int8)
    mesh = Mesh(np.asarray(jax.devices()[:DCP]).reshape(lanes, ranks), ("lane", "dcp"))
    both = P(("lane", "dcp"))

    def body(w, index):
        started = msa_dcp_heads.start_block_all_gather(
            w, index, axes=("lane", "dcp"), shape=(lanes, ranks)
        )
        got = msa_dcp_heads.finish_block_exchange(started, DCP)
        return got[None], jax.lax.all_gather(w, ("lane", "dcp"))[None]

    f = jax.shard_map(
        body, mesh=mesh, in_specs=(both, both), out_specs=(both, both), check_vma=False
    )
    got, want = f(jnp.asarray(w), jnp.arange(DCP, dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


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


def _replica_halves(x, rank):
    """`x [T, ...]`'s rows as the replicas take them: alternate 64-token
    blocks, ``(rank % 2's, the other replica's)``."""
    blocks = x.reshape(-1, 2, 64, *x.shape[1:])
    return blocks[:, rank % 2].reshape(-1, *x.shape[1:]), blocks[
        :, 1 - rank % 2
    ].reshape(-1, *x.shape[1:])


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("num_tokens", [384, 2432])
def test_rotate_replica_rows_matches_rotate_queries(num_tokens, rank):
    """Bit for bit, including a partial last grid step (2432 tokens)."""
    rng = np.random.default_rng(num_tokens + rank)
    q = jnp.asarray(rng.standard_normal((num_tokens, HL, HEAD_DIM)), jnp.bfloat16)
    got = msa_dcp_heads.rotate_replica_rows(q, jnp.int32(rank), interpret=True)
    want = _replica_halves(msa_fp4.rotate_queries(q), rank)
    for g, w in zip(got, want):
        assert g.shape == (HL, num_tokens // 2, HEAD_DIM)
        np.testing.assert_array_equal(
            np.asarray(g).view(np.uint16),
            np.asarray(w.transpose(1, 0, 2)).view(np.uint16),
        )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("num_tokens", [384, 2432])
def test_unrotate_replica_rows_matches_unrotate_output(num_tokens, rank):
    """Bit for bit: own rows from the first heads of a head-major output
    twice as wide, the partner's from `other`, back in step order."""
    rng = np.random.default_rng(num_tokens + rank)
    out = jnp.asarray(
        rng.standard_normal((2 * HL, num_tokens // 2, HEAD_DIM)), jnp.bfloat16
    )
    other = jnp.asarray(
        rng.standard_normal((HL, num_tokens // 2, HEAD_DIM)), jnp.bfloat16
    )
    got = msa_dcp_heads.unrotate_replica_rows(
        out, other, jnp.int32(rank), interpret=True
    )
    halves = (out[:HL], other) if rank % 2 == 0 else (other, out[:HL])
    rows = jnp.stack(halves).reshape(2, HL, -1, 64, HEAD_DIM)
    rows = rows.transpose(1, 2, 0, 3, 4).reshape(HL, num_tokens, HEAD_DIM)
    want = msa_fp4.unrotate_output(rows.transpose(1, 0, 2)).reshape(num_tokens, -1)
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )


def test_pair_first_merge_matches_lse_merge():
    """The two-exchange LSE merge of a DCP group of 4 keeps head block r on
    rank r, equal to the log-sum-exp-weighted sum of every rank's partial
    (a -inf LSE: a rank that attended nothing for that head and token)."""
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("dcp",))
    rng = np.random.default_rng(7)
    heads, tokens = 4, 24
    out = rng.standard_normal((4, 4 * heads, tokens, 128)).astype(np.float32)
    lse = (rng.standard_normal((4, 4 * heads, tokens)) * 4).astype(np.float32)
    lse[1, 3, 5] = -np.inf
    lse[:, 6, 2] = -np.inf  # no rank attended
    out[1, 3, 5] = 0.0
    out[:, 6, 2] = 0.0
    ranks = jnp.arange(4, dtype=jnp.int32)
    got = jax.shard_map(
        lambda o, lg, r: msa_dcp_heads._pair_first_merge(o, lg, "dcp", r[0], True),
        mesh=mesh,
        in_specs=(P("dcp"), P("dcp"), P("dcp")),
        out_specs=P("dcp"),
        check_vma=False,
    )(
        jnp.asarray(out.reshape(-1, tokens, 128)),
        jnp.asarray(lse.reshape(-1, tokens)),
        ranks,
    )
    w = np.nan_to_num(np.exp(lse - np.logaddexp.reduce(lse, axis=0)))
    want = (w[..., None] * out).sum(0).reshape(4, heads, tokens, 128)
    np.testing.assert_allclose(
        np.asarray(got).reshape(4, heads, tokens, 128), want, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("head_calls", [True, False])
def test_dcp4_decode_merge_matches_dcp_reference(head_calls, monkeypatch):
    """c64's layout (DCP 4 over 2 KV heads), decode rows only: the pair-first
    merge, with one kernel call per KV head (`_pair_first_merge_by_head`) or
    one call for both."""
    mod = globals()
    for name, value in {"DCP": 4, "HEADS": 2, "REPLICAS": 2, "HL": 8}.items():
        monkeypatch.setitem(mod, name, value)
    if len(jax.devices()) < DCP:
        pytest.skip(
            "needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)"
        )
    monkeypatch.setattr(msa_dcp_heads, "PAIR_FIRST_HEAD_CALLS", head_calls)
    rng = np.random.default_rng(1)
    *args, num_real = _step(rng, [8, 8, 8], [3000, 5000, 8800], 12, 32)
    want = _reference(*args)
    got = _spmd(*args, decode_rows=8)
    for r in range(DCP):
        np.testing.assert_allclose(
            got[r][:num_real], np.asarray(want[r])[:num_real], rtol=2e-2, atol=2e-2
        )


@pytest.mark.parametrize("num_tokens", [2112, 2047, 384])
def test_gather_group_kv_matches_group_rows(num_tokens):
    """The group's K/V and index queries gathered in the program (DCP 4 over
    2 KV heads) are every head's rows: lane-dense rows (2112 tokens, no
    whole 8-bit tiles to quantize into), 2047 tokens that do not split over
    the replicas, decode-sized rows."""
    dcp, heads = 4, 2
    if len(jax.devices()) < dcp:
        pytest.skip(
            "needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)"
        )
    rng = np.random.default_rng(num_tokens)
    group = [
        jnp.asarray(rng.standard_normal((num_tokens, heads, HEAD_DIM)), jnp.bfloat16)
        for _ in range(3)
    ]  # k, v, index_q of the group's heads
    own = [jnp.repeat(x, dcp // heads, axis=1) for x in group]  # rank r: head r // 2
    mesh = Mesh(np.asarray(jax.devices()[:dcp]), ("dcp",))

    def body(rank, *rows):
        got = msa_dcp_heads.gather_group_kv(
            *(x.transpose(1, 0, 2) for x in rows),
            rank,
            axis="dcp",
            dcp=dcp,
            kv_head_replicas=dcp // heads,
            interpret=True,
        )
        return tuple(x[None] for x in got)

    f = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(P("dcp"),) * 4,
        out_specs=(P("dcp"),) * 3,
        check_vma=False,
    )
    got = f(jnp.arange(dcp, dtype=jnp.int32), *(x.transpose(1, 0, 2) for x in own))
    for r in range(dcp):
        for g, w in zip(got, group):
            np.testing.assert_array_equal(np.asarray(g[r]), np.asarray(w))


@pytest.mark.parametrize("num_tokens", [2048, 2304])
def test_gather_group_kv_quantizes_the_write_rows(num_tokens):
    """A step that quantizes its K/V in a kernel gathers them quantized (DCP 4
    over 2 KV heads): bit for bit the rows and scales `quantize_kv_rows` makes
    of the group's bf16 heads (what the write stores); quarters of 256 and
    288 tokens (unpack blocks of 256 and 32)."""
    dcp, heads = 4, 2
    if len(jax.devices()) < dcp:
        pytest.skip(
            "needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)"
        )
    rng = np.random.default_rng(num_tokens)
    group = [
        jnp.asarray(rng.standard_normal((num_tokens, heads, HEAD_DIM)), jnp.bfloat16)
        for _ in range(3)
    ]  # k, v, index_q of the group's heads
    own = [jnp.repeat(x, dcp // heads, axis=1) for x in group]  # rank r: head r // 2
    mesh = Mesh(np.asarray(jax.devices()[:dcp]), ("dcp",))

    def body(rank, k, v, index_q):
        k, v, index_q = msa_dcp_heads.gather_group_kv(
            k.transpose(1, 0, 2),
            v.transpose(1, 0, 2),
            index_q.transpose(1, 0, 2),
            rank,
            axis="dcp",
            dcp=dcp,
            kv_head_replicas=dcp // heads,
            interpret=True,
        )
        kv = (k.rows, k.scales) if v is None else (k, v)
        return tuple(x[None] for x in kv), index_q[None]

    f = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(P("dcp"),) * 4,
        out_specs=((P("dcp"), P("dcp")), P("dcp")),
        check_vma=False,
    )
    ranks = jnp.arange(dcp, dtype=jnp.int32)
    (a, b), index_q = f(ranks, *(x.transpose(1, 0, 2) for x in own))
    rows, scales = msa_quantize.quantize_kv_rows(group[0], group[1], interpret=True)
    for r in range(dcp):
        np.testing.assert_array_equal(np.asarray(index_q[r]), np.asarray(group[2]))
        np.testing.assert_array_equal(np.asarray(a[r]), np.asarray(rows))
        np.testing.assert_array_equal(np.asarray(b[r]), np.asarray(scales))


def test_fp4_targets_take_quantized_kv():
    """Pre-quantized K/V (`msa.QuantizedKV`) make the write targets bf16 K/V
    do, bit for bit."""
    rng = np.random.default_rng(5)
    num_tokens, heads, pages = 2048, 2, 4
    k, v = (
        jnp.asarray(rng.standard_normal((num_tokens, heads, HEAD_DIM)), jnp.bfloat16)
        for _ in range(2)
    )
    index_k = jnp.asarray(rng.standard_normal((num_tokens, HEAD_DIM)), jnp.bfloat16)
    caches = (
        jnp.zeros((pages, 16, 4, heads * 2 * HEAD_DIM), jnp.uint8),
        jnp.zeros((pages, 17, 4, HEAD_DIM), jnp.uint8),
        jnp.zeros((pages, 8, PAGE), jnp.float32),
    )
    want = msa.fp4_targets(*caches, k, v, index_k, interpret=True)
    kv = msa.QuantizedKV(*msa_quantize.quantize_kv_rows(k, v, interpret=True))
    got = msa.fp4_targets(*caches, kv, None, index_k, interpret=True)
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
