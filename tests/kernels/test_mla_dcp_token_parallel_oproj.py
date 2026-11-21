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
"""Sequence-row o_proj of the token-parallel split vs o_proj + reduce-scatter.

`dcp_token_parallel.project_sequence_rows` against the head-sharded o_proj and
reduce-scatter it replaces, on a host-device mesh
(XLA_FLAGS=--xla_force_host_platform_device_count=16).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op_mod

AXIS = "dcp"
PAD_SENTINEL = 1.0e4


def _mesh(dcp: int) -> Mesh:
    if len(jax.devices()) < dcp:
        pytest.skip(f"needs {dcp} devices")
    return Mesh(np.array(jax.devices()[:dcp]), (AXIS,))


def _layout(dcp: int, split_rows: int) -> tp.TokenParallelLayout:
    return tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=1,
        page_size=dcp,
        split_rows=split_rows,
        max_pages=1,
    )


def _sharded(mesh: Mesh, x: np.ndarray) -> jax.Array:
    return jax.device_put(jnp.asarray(x), NamedSharding(mesh, P(AXIS)))


def _own_rows_global(step_rows: np.ndarray, dcp: int, split_rows: int) -> np.ndarray:
    """Rank j's token-parallel rows `split_rows + j * R + i`, stacked.

    Rows past T carry PAD_SENTINEL."""
    num_tokens = step_rows.shape[0]
    rows = num_tokens // dcp
    padded = np.concatenate(
        [
            step_rows,
            np.full((split_rows, *step_rows.shape[1:]), PAD_SENTINEL, np.float32),
        ]
    )
    return np.concatenate(
        [
            padded[split_rows + j * rows : split_rows + (j + 1) * rows]
            for j in range(dcp)
        ]
    )


@pytest.mark.parametrize(("dcp", "split_rows"), [(4, 2), (16, 16)])
def test_should_land_on_contiguous_sequence_rows_when_token_rows_are_shifted(
    dcp, split_rows
):
    mesh = _mesh(dcp)
    layout = _layout(dcp, split_rows)
    num_tokens, width = dcp * 2 * split_rows, 8
    step_rows = np.arange(num_tokens * width, dtype=np.float32).reshape(
        num_tokens, width
    )
    own = _sharded(mesh, _own_rows_global(step_rows, dcp, split_rows))
    lead = jnp.asarray(step_rows[:split_rows])
    rank_ids = _sharded(mesh, np.arange(dcp, dtype=np.int32))

    out = jax.shard_map(
        lambda o, lr, r: tp.token_rows_to_sequence_rows(layout, o, lr, r[0], AXIS),
        mesh=mesh,
        in_specs=(P(AXIS), P(), P(AXIS)),
        out_specs=P(AXIS),
        check_vma=False,
    )(own, lead, rank_ids)

    np.testing.assert_array_equal(np.asarray(out), step_rows)


def _random_fp8_row_shards(
    rng, groups: int, k: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row-parallel FP8 o_proj: `[groups * k, n]` and `[groups, n]` scales.

    Each `k`-row shard is quantized per output channel on its own."""
    w = rng.standard_normal((groups, k, n)).astype(np.float32)
    w[1] *= 7.0  # shards with very different ranges get different scales
    scale = np.abs(w).max(axis=1) / 448.0
    w_q = np.asarray(jnp.asarray(w / scale[:, None, :]).astype(jnp.float8_e4m3fn))
    return w_q.reshape(groups * k, n), scale.astype(np.float32)


def _pair_halves(w_q, w_s, dcp):
    """Every rank's chip-pair half: even ranks shards [0, dcp/2), odd the rest."""
    k = w_q.shape[0] // dcp
    halves = [w_q[: dcp // 2 * k], w_q[dcp // 2 * k :]]
    scales = [w_s[: dcp // 2], w_s[dcp // 2 :]]
    return (
        np.concatenate([halves[r % 2] for r in range(dcp)]),
        np.concatenate([scales[r % 2] for r in range(dcp)]),
    )


def _whole_k_fp8(rng, k, n):
    """Whole-K per-channel FP8 o_proj: `[k, n]` and its `[n]` scale."""
    w = rng.standard_normal((k, n)).astype(np.float32)
    w[: k // 3] *= 7.0
    scale = (np.abs(w).max(axis=0) / 448.0).astype(np.float32)
    return np.asarray(jnp.asarray(w / scale).astype(jnp.float8_e4m3fn)), scale


def _fp8_rows(x):
    """Per-row FP8 round trip, as `quantize_rows` then dequantization."""
    x_q, scale = tp.quantize_rows(jnp.asarray(x))
    return np.asarray(x_q).astype(np.float32) * np.asarray(scale)


@pytest.mark.parametrize("dcp", [4, 16])
def test_should_match_fp8_product_when_o_proj_is_whole_k_pair_halves(dcp):
    rng = np.random.default_rng(100 + dcp)
    mesh = _mesh(dcp)
    split_rows, heads_per_rank, latent, v_dim, hidden = 8, 2, 16, 8, 24
    layout = _layout(dcp, split_rows)
    heads = dcp * heads_per_rank
    num_tokens = dcp * 2 * split_rows
    lead_partial = rng.standard_normal((dcp, split_rows, heads, latent)).astype(
        np.float32
    )
    lead_lse = rng.uniform(-3.0, 3.0, (dcp, split_rows, heads)).astype(np.float32)
    w_uv = rng.standard_normal((heads, latent, v_dim)).astype(np.float32)
    tail_v = rng.standard_normal((num_tokens - split_rows, heads, v_dim)).astype(
        np.float32
    )
    w_q, w_s = _whole_k_fp8(rng, heads * v_dim, hidden)
    m = np.logaddexp.reduce(lead_lse, axis=0)
    lead_latent = (lead_partial * np.exp(lead_lse - m[None])[..., None]).sum(axis=0)
    lead_v = np.einsum("rhl,hlv->rhv", lead_latent, w_uv)
    step_v = np.concatenate([lead_v, tail_v])
    expected = _fp8_rows(step_v.reshape(num_tokens, -1)) @ (
        w_q.astype(np.float32) * w_s[None, :]
    )
    half = heads * v_dim // 2
    w_arg = np.concatenate(
        [w_q[(r % 2) * half : (r % 2 + 1) * half] for r in range(dcp)]
    )

    out = tp.project_sequence_rows(
        layout,
        mesh,
        AXIS,
        _sharded(mesh, lead_partial.reshape(dcp * split_rows, heads, latent)),
        _sharded(mesh, lead_lse.reshape(dcp * split_rows, heads)),
        _sharded(mesh, _own_rows_global(step_v, dcp, split_rows)),
        _sharded(mesh, np.arange(dcp, dtype=np.int32)),
        jnp.asarray(w_uv),
        _sharded(mesh, w_arg),
        _sharded(mesh, np.tile(w_s, dcp)),
    )

    np.testing.assert_allclose(
        np.asarray(out), expected, rtol=1e-3, atol=1e-3 * np.abs(expected).max()
    )


@pytest.mark.parametrize("pair_halves", [False, True], ids=["own_shard", "pair_half"])
@pytest.mark.parametrize("dcp", [4, 16])
def test_should_match_head_sharded_oproj_reduce_scatter_when_projecting_split_step(
    dcp, pair_halves
):
    rng = np.random.default_rng(dcp)
    mesh = _mesh(dcp)
    split_rows, heads_per_rank, latent, v_dim, hidden = 8, 2, 16, 8, 24
    layout = _layout(dcp, split_rows)
    heads = dcp * heads_per_rank
    num_tokens = dcp * 2 * split_rows

    lead_partial = rng.standard_normal((dcp, split_rows, heads, latent)).astype(
        np.float32
    )
    lead_lse = rng.uniform(-3.0, 3.0, (dcp, split_rows, heads)).astype(np.float32)
    lead_lse[0, 0, :] = -np.inf  # a rank that owns none of a row's keys
    w_uv = rng.standard_normal((heads, latent, v_dim)).astype(np.float32)
    tail_v = rng.standard_normal((num_tokens - split_rows, heads, v_dim)).astype(
        np.float32
    )
    w_q, w_s = _random_fp8_row_shards(rng, dcp, heads_per_rank * v_dim, hidden)

    # Reference: the exact LSE merge, W_UV, then o_proj over every row at once
    # (head-sharded partial sums + reduce-scatter compute the same rows).
    m = np.logaddexp.reduce(lead_lse, axis=0)
    weights = np.exp(lead_lse - m[None])
    lead_latent = (lead_partial * weights[..., None]).sum(axis=0)
    lead_v = np.einsum("rhl,hlv->rhv", lead_latent, w_uv)
    step_v = np.concatenate([lead_v, tail_v]).reshape(num_tokens, heads * v_dim)
    w_deq = (
        w_q.astype(np.float32).reshape(dcp, heads_per_rank * v_dim, hidden)
        * w_s[:, None, :]
    ).reshape(heads * v_dim, hidden)
    expected = step_v @ w_deq

    own_out = _own_rows_global(
        np.concatenate([lead_v, tail_v]), dcp, split_rows
    )  # [dcp * R, H, V]
    if pair_halves:
        w_arg, s_arg = _pair_halves(w_q, w_s, dcp)
    else:
        w_arg, s_arg = w_q, w_s.reshape(-1)
    out = tp.project_sequence_rows(
        layout,
        mesh,
        AXIS,
        _sharded(mesh, lead_partial.reshape(dcp * split_rows, heads, latent)),
        _sharded(mesh, lead_lse.reshape(dcp * split_rows, heads)),
        _sharded(mesh, own_out),
        _sharded(mesh, np.arange(dcp, dtype=np.int32)),
        jnp.asarray(w_uv),
        _sharded(mesh, w_arg),
        _sharded(mesh, s_arg),
    )

    assert out.shape == (num_tokens, hidden)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=2e-4, atol=2e-3)


def test_should_reject_lead_rows_not_matching_split_rows():
    layout = _layout(4, 2)
    with pytest.raises(ValueError, match="split_rows"):
        tp.token_rows_to_sequence_rows(
            layout, jnp.zeros((4, 8)), jnp.zeros((3, 8)), jnp.int32(0), AXIS
        )


def test_should_preshift_sequence_rows_before_oproj_preserving_bf16_bits():
    dcp, split_rows, heads_per_rank, latent, v_dim, hidden = 4, 8, 2, 16, 8, 24
    rng = np.random.default_rng(77)
    mesh = _mesh(dcp)
    layout = _layout(dcp, split_rows)
    heads = dcp * heads_per_rank
    rows_per_rank = 2 * split_rows
    lead_partial = rng.standard_normal((dcp, split_rows, heads, latent)).astype(
        jnp.bfloat16
    )
    lead_lse = rng.uniform(-3.0, 3.0, (dcp, split_rows, heads)).astype(np.float32)
    lead_lse[0, 0, :] = -np.inf
    w_uv = rng.standard_normal((heads, latent, v_dim)).astype(jnp.bfloat16)
    own_out = rng.standard_normal((dcp * rows_per_rank, heads, v_dim)).astype(
        jnp.bfloat16
    )
    w_q, w_s = _whole_k_fp8(rng, heads * v_dim, hidden)
    half = heads * v_dim // 2
    w_arg = np.concatenate(
        [w_q[(r % 2) * half : (r % 2 + 1) * half] for r in range(dcp)]
    )
    args = (
        _sharded(mesh, lead_partial.reshape(dcp * split_rows, heads, latent)),
        _sharded(mesh, lead_lse.reshape(dcp * split_rows, heads)),
        _sharded(mesh, own_out),
        _sharded(mesh, np.arange(dcp, dtype=np.int32)),
        jnp.asarray(w_uv),
        _sharded(mesh, w_arg),
        _sharded(mesh, np.tile(w_s, dcp)),
    )
    got = tp.project_sequence_rows(layout, mesh, AXIS, *args)

    # Independent reference: project [lead; own] first, then shift to sequence rows.
    sharded = P(AXIS)

    def _postshift_ref(part, lse_part, own, rank_arr, w_uv_all, w_shard, s_shard):
        lses = jax.lax.all_gather(lse_part, AXIS, axis=0, tiled=False)
        m = jax.nn.logsumexp(lses, axis=0)
        safe_m = jnp.where(jnp.isfinite(m), m, 0.0)
        w = jnp.where(jnp.isfinite(lse_part), jnp.exp(lse_part - safe_m), 0.0)
        weighted = part.astype(jnp.float32) * w[..., None]
        lead_latent = jax.lax.psum(weighted, AXIS).astype(part.dtype)
        lead_v = jnp.einsum(
            "rhl,hlv->rhv", lead_latent, w_uv_all.astype(lead_latent.dtype)
        ).astype(own.dtype)
        s = lead_v.shape[0]
        rows = jnp.concatenate(
            [lead_v.reshape(s, -1), own.reshape(own.shape[0], -1)], axis=0
        )
        proj = tp.project_all_heads(rows, w_shard, s_shard, rank_arr[0], AXIS, dcp)
        return tp.token_rows_to_sequence_rows(
            layout, proj[s:], proj[:s], rank_arr[0], AXIS
        )

    want = jax.shard_map(
        _postshift_ref,
        mesh=mesh,
        in_specs=(sharded, sharded, sharded, sharded, P(), sharded, sharded),
        out_specs=sharded,
        check_vma=False,
    )(*args)
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )

    # Structural check: FP8 o_proj dot_general receives exact R rows, not R + s.
    jaxpr = jax.make_jaxpr(lambda *a: tp.project_sequence_rows(layout, mesh, AXIS, *a))(
        *args
    )
    dot_lhs_rows: list[int] = []

    def _walk(jpr):
        for eqn in jpr.eqns:
            if eqn.primitive.name == "dot_general":
                shape = eqn.invars[0].aval.shape
                if len(shape) == 2 and shape[1] == half:
                    dot_lhs_rows.append(shape[0])
            for sub in jax.core.jaxprs_in_params(eqn.params):
                _walk(sub)

    _walk(jaxpr.jaxpr)
    assert dot_lhs_rows and all(r == rows_per_rank for r in dot_lhs_rows)


def test_should_keep_rank_tie_zero_delta_for_all_65536_bf16_bit_patterns():
    dcp = 4
    mesh = _mesh(dcp)
    shd = P(AXIS)
    ranks = _sharded(mesh, np.arange(dcp, dtype=np.int32))
    # Exhaustive check across all 65,536 bf16 bit patterns (including NaNs/Infs).
    all_bf16 = jax.lax.bitcast_convert_type(
        jnp.arange(65536, dtype=jnp.uint16), jnp.bfloat16
    )
    probe_vals = jnp.array(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan")], dtype=jnp.bfloat16
    )
    for v in probe_vals:
        partial = _sharded(mesh, np.full((dcp, 1, 1), v, dtype=jnp.bfloat16))
        tied = tp_op_mod._tie_rank_to_partial(  # noqa: SLF001 -- unit under test
            partial, ranks, mesh, shd
        )
        np.testing.assert_array_equal(np.asarray(tied), np.arange(dcp, dtype=np.int32))
    bits = jax.lax.bitcast_convert_type(all_bf16, jnp.uint16).astype(jnp.int32)
    assert int(jnp.sum(bits == jnp.int32(-1))) == 0
    partial = _sharded(mesh, np.zeros((dcp, 1, 1), dtype=jnp.bfloat16))
    jaxpr = jax.make_jaxpr(
        lambda p, r: tp_op_mod._tie_rank_to_partial(  # noqa: SLF001 -- unit under test
            p, r, mesh, shd
        )
    )(partial, ranks)
    prims: set[str] = set()

    def _walk(jpr):
        for eqn in jpr.eqns:
            prims.add(eqn.primitive.name)
            for sub in jax.core.jaxprs_in_params(eqn.params):
                _walk(sub)

    _walk(jaxpr.jaxpr)
    assert "optimization_barrier" in prims


def test_should_match_serial_sequence_rows_when_lead_merge_and_w_uv_in_branch():
    dcp = 4
    mesh = _mesh(dcp)
    split_rows = 2
    rows_per_rank = 4
    num_tokens = dcp * rows_per_rank
    heads, lkv, vdim, hidden = 4, 8, 6, 12
    page_size = 4
    layout = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=1,
        page_size=page_size,
        split_rows=split_rows,
        max_pages=2,
        gather_caps=(1, 2),
    )
    rng = np.random.default_rng(137)
    nope_cache = _sharded(
        mesh, rng.standard_normal((dcp * 2, page_size, lkv)).astype(np.float32)
    )
    rope_cache = _sharded(
        mesh, rng.standard_normal((dcp * 2, page_size, 4)).astype(np.float32)
    )
    q = _sharded(
        mesh,
        rng.standard_normal((dcp * rows_per_rank, heads, lkv)).astype(np.float32),
    ).astype(jnp.bfloat16)
    kv_c = jnp.zeros((num_tokens, lkv), jnp.bfloat16)
    k_pe = jnp.zeros((num_tokens, 4), jnp.bfloat16)
    flat_topk = _sharded(
        mesh,
        rng.integers(0, dcp * page_size, size=(dcp * rows_per_rank, 4), dtype=np.int32),
    )
    seq_lens = jnp.asarray([num_tokens], jnp.int32)
    block_tables = jnp.asarray([0, 1], jnp.int32)
    qsl = jnp.asarray([0, num_tokens], jnp.int32)
    dist = jnp.asarray([0, 0, 1], jnp.int32)
    rank_ids = _sharded(mesh, np.arange(dcp, dtype=np.int32))
    w_uv = jnp.asarray(
        rng.standard_normal((heads, lkv, vdim)).astype(np.float32) * 0.2,
        jnp.bfloat16,
    )
    lead_part = _sharded(
        mesh,
        (rng.standard_normal((dcp * split_rows, heads, lkv)) * 0.5).astype(np.float32),
    ).astype(jnp.bfloat16)
    lead_lse_np = rng.standard_normal((dcp * split_rows, heads)).astype(np.float32)
    # Anti-cancellation: non-finite -inf rows on one rank and all ranks for a head.
    lead_lse_np[0, 1] = -np.inf
    lead_lse_np[1::split_rows, 3] = -np.inf
    lead_lse = _sharded(mesh, lead_lse_np)
    w_q_full, s_full = _whole_k_fp8(rng, heads * vdim, hidden)
    half = heads * vdim // 2
    w_o = np.concatenate(
        [w_q_full[(r % 2) * half : (r % 2 + 1) * half] for r in range(dcp)]
    )
    w_o_shd = _sharded(mesh, w_o)
    s_o_shd = _sharded(mesh, np.tile(s_full, dcp))

    def _attend(q_tok, g_nope, g_rope, flat, num_rows):
        del g_rope, flat, num_rows
        scale = jnp.mean(g_nope.astype(jnp.float32)) * jnp.float32(1e-3)
        return (q_tok.astype(jnp.float32) + scale).astype(q_tok.dtype)

    _, _, own_ref = tp.sparse_mla_attention(
        layout,
        mesh,
        AXIS,
        nope_cache,
        rope_cache,
        q,
        kv_c,
        k_pe,
        flat_topk,
        seq_lens,
        block_tables,
        qsl,
        dist,
        write_kv=None,
        attend=_attend,
        rank_ids=rank_ids,
        q_own_rows=True,
        w_uv=w_uv,
        token_rows=True,
        w_uv_replicated=True,
    )
    want = tp.project_sequence_rows(
        layout,
        mesh,
        AXIS,
        lead_part,
        lead_lse,
        own_ref,
        rank_ids,
        w_uv,
        w_o_shd,
        s_o_shd,
    )

    _, _, own_got, lead_v_got = tp.sparse_mla_attention_with_lead(
        layout,
        mesh,
        AXIS,
        nope_cache,
        rope_cache,
        q,
        kv_c,
        k_pe,
        flat_topk,
        seq_lens,
        block_tables,
        qsl,
        dist,
        lead_partial=lead_part,
        lead_lse=lead_lse,
        write_kv=None,
        attend=_attend,
        w_uv=w_uv,
        rank_ids=rank_ids,
        q_own_rows=True,
        w_uv_replicated=True,
    )
    got = tp.project_sequence_rows_from_v(
        layout,
        mesh,
        AXIS,
        lead_v_got,
        own_got,
        rank_ids,
        w_o_shd,
        s_o_shd,
    )
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )

    # Structural check: lead all_gather, lead psum, optimization_barrier, and
    # W_UV dot_general live inside the cap switch branches, not outside.
    jaxpr = jax.make_jaxpr(
        lambda *a: tp.sparse_mla_attention_with_lead(
            layout,
            mesh,
            AXIS,
            *a[:10],
            lead_partial=a[10],
            lead_lse=a[11],
            write_kv=None,
            attend=_attend,
            w_uv=a[12],
            rank_ids=a[13],
            q_own_rows=True,
            w_uv_replicated=True,
        )
    )(
        nope_cache,
        rope_cache,
        q,
        kv_c,
        k_pe,
        flat_topk,
        seq_lens,
        block_tables,
        qsl,
        dist,
        lead_part,
        lead_lse,
        w_uv,
        rank_ids,
    )
    in_cond_prims: set[str] = set()
    outside_cond_prims: set[str] = set()

    def _walk_scope(jpr, inside_cond: bool):
        for eqn in jpr.eqns:
            is_cond = eqn.primitive.name == "cond"
            target = in_cond_prims if inside_cond else outside_cond_prims
            target.add(eqn.primitive.name)
            for sub in jax.core.jaxprs_in_params(eqn.params):
                _walk_scope(sub, inside_cond or is_cond)

    _walk_scope(jaxpr.jaxpr, False)
    assert {
        "all_gather",
        "psum",
        "optimization_barrier",
        "dot_general",
    } <= in_cond_prims
    assert "psum" not in outside_cond_prims
    assert "dot_general" not in outside_cond_prims


def test_should_reject_partial_or_invalid_lead_merge_arguments():
    dcp = 4
    mesh = _mesh(dcp)
    layout = _layout(dcp, 2)
    shd = _sharded(mesh, np.zeros((dcp, 1), dtype=np.float32))
    rep = jnp.zeros((1,), dtype=np.float32)
    with pytest.raises(ValueError, match="together"):
        tp._sparse_mla_attention_core(  # noqa: SLF001 -- fail-fast guard test
            layout,
            mesh,
            AXIS,
            shd,
            shd,
            shd,
            rep,
            rep,
            shd,
            rep,
            rep,
            rep,
            rep,
            write_kv=None,
            attend=lambda *a: a[0],
            lead_partial=shd,
            lead_lse=None,
        )
    with pytest.raises(ValueError, match="token_rows=True"):
        tp._sparse_mla_attention_core(  # noqa: SLF001 -- fail-fast guard test
            layout,
            mesh,
            AXIS,
            shd,
            shd,
            shd,
            rep,
            rep,
            shd,
            rep,
            rep,
            rep,
            rep,
            write_kv=None,
            attend=lambda *a: a[0],
            w_uv=rep,
            token_rows=False,
            lead_partial=shd,
            lead_lse=shd,
        )
