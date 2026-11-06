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
"""Bitwise identity of the non-EP MoE token-movement choices
(envs.TPU_MOE_NONEP_GATHER / _PERMUTE) against the served path.

The CPU tests cover the pure-JAX choices (column- and row-chunked XLA gathers,
the two_sorts and counting permutations); the TPU tests run fused_moe_func end
to end at the PP8xTP1 prefill shape (512 experts, top-10, hidden 2560,
intermediate 640, per-channel fp8 weights) and add the SparseCore gather.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.core import fused_moe_gmm
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    auto_gather_cols,
    chunked_gather,
    column_chunked_gather,
    prepare_routed_gmm_inputs,
)

HIDDEN = 2560
EXPERTS = 512
TOPK = 10
INTER = 640


def _bits(x: jax.Array) -> np.ndarray:
    """Raw bit pattern, so NaN payloads and signed zeros compare exactly."""
    a = np.asarray(x)
    return a.view(np.uint16) if a.dtype.itemsize == 2 else a.view(np.uint32)


def _hidden(rows: int, cols: int, seed: int = 0) -> jax.Array:
    x = np.asarray(
        jax.random.normal(jax.random.key(seed), (rows, cols), jnp.float32).astype(
            jnp.bfloat16
        )
    ).copy()
    # A gather must copy bits, special values included.
    x[0, 0], x[1, 1], x[2, 2] = np.nan, -0.0, np.inf
    return jnp.asarray(x)


def _routing(tokens: int, seed: int = 1) -> tuple[jax.Array, jax.Array]:
    """Distinct top-k expert ids per token and softmax-like weights."""
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(EXPERTS, TOPK, replace=False) for _ in range(tokens)])
    w = rng.random((tokens, TOPK)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)
    return jnp.asarray(ids, jnp.int32), jnp.asarray(w, jnp.bfloat16)


@pytest.mark.parametrize(
    "rows,expected",
    [(2048, 2560), (3200, 2560), (4096, 1280), (6400, 1280), (8192, 640), (16384, 256)],
)
def test_auto_gather_cols_keeps_source_table_in_measured_vmem_size(rows, expected):
    cols = auto_gather_cols(rows, HIDDEN, 2)
    assert cols == expected
    assert HIDDEN % cols == 0 and cols % 128 == 0
    assert rows * cols * 2 <= fused_moe_gmm._NONEP_GATHER_SOURCE_BYTES


@pytest.mark.parametrize("cols", [128, 640, 1024, 1280, 2560, 4096])
def test_column_chunked_gather_is_bitwise_plain_gather(cols):
    x = _hidden(640, HIDDEN)
    idx = jnp.asarray(np.random.default_rng(2).integers(0, 640, 6400), jnp.int32)
    np.testing.assert_array_equal(
        _bits(column_chunked_gather(x, idx, cols)), _bits(x[idx])
    )


@pytest.mark.parametrize("chunk", [1000, 3200, 6400, 16384])
def test_chunked_gather_is_bitwise_plain_gather(chunk):
    x = _hidden(640, HIDDEN)
    idx = jnp.asarray(np.random.default_rng(3).integers(0, 640, 6400), jnp.int32)
    np.testing.assert_array_equal(_bits(chunked_gather(x, idx, chunk)), _bits(x[idx]))


_GATHERS = [("xla", 0, 16384), ("xla_colchunk", 640, 16384), ("xla_colchunk", 1024, 16384),
            ("xla_chunked", 0, 1000)]
_PERMUTES = ["sorts", "two_sorts", "counting"]


@pytest.mark.parametrize("skip_padded", [False, True])
@pytest.mark.parametrize("tokens", [640, 1001])
def test_nonep_permute_and_gather_choices_match_served_path(tokens, skip_padded):
    """Every (permute, gather) pair returns bitwise the served outputs: the
    gathered rows, group sizes, inverse permutation and sorted token ids."""
    x = _hidden(tokens, HIDDEN)
    ids, w = _routing(tokens)
    if skip_padded:
        # Padded tokens keep their ids but carry zero weights.
        w = w.at[tokens - 7 :].set(0)

    def run(permute, gather, cols, chunk):
        return prepare_routed_gmm_inputs(
            x, ids, w, local_num_experts=EXPERTS, topk=TOPK, use_ep=False,
            use_sparse_core=True, skip_padded_tokens=skip_padded,
            nonep_permute=permute, nonep_gather=gather, nonep_gather_cols=cols,
            nonep_gather_chunk=chunk,
        )

    ref = run("sorts", "xla", 0, 16384)
    assert ref[0].shape == (tokens * TOPK, HIDDEN)
    for permute, (gather, cols, chunk) in itertools.product(_PERMUTES, _GATHERS):
        got = run(permute, gather, cols, chunk)
        for name, a, b in zip(
            ("x", "group_sizes", "revert", "weights", "valid", "token_ids"), got, ref
        ):
            np.testing.assert_array_equal(
                _bits(a) if a.dtype == jnp.bfloat16 else np.asarray(a),
                _bits(b) if b.dtype == jnp.bfloat16 else np.asarray(b),
                err_msg=f"{permute}/{gather}/{cols}/{chunk}: {name}",
            )


def test_nonep_choices_change_only_the_non_ep_program():
    """The rows are bitwise equal either way, so check the traced program: the
    choices rewrite the non-EP path and leave the EP path's program as is."""
    tokens = 640
    x = _hidden(tokens, HIDDEN)
    ids, w = _routing(tokens)

    def jaxpr(use_ep, **kw):
        fn = lambda a, b, c: prepare_routed_gmm_inputs(  # noqa: E731
            a, b, c, local_num_experts=EXPERTS, topk=TOPK, use_ep=use_ep,
            use_sparse_core=False, **kw,
        )
        return str(jax.make_jaxpr(fn)(x, ids, w))

    chosen = dict(nonep_permute="two_sorts", nonep_gather="xla_colchunk", nonep_gather_cols=128)
    assert jaxpr(True, **chosen) == jaxpr(True)
    non_ep = jaxpr(False, **chosen)
    assert non_ep != jaxpr(False)
    assert "concatenate" in non_ep and "concatenate" not in jaxpr(False)


def _ref_gmm(calls, tiles=None, hoists=None):
    """Pure-JAX stand-in for gmm_wrapper that records its n_window (and
    tile_m); every output column is computed independently, as the kernel's
    are."""
    tiles = [] if tiles is None else tiles
    hoists = [] if hoists is None else hoists

    def gmm(lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset, zero_initialize=False,
            fuse_act=None, preferred_element_type=None, rhs_quant_dtype=None, n_window=None,
            tile_m=0, hoist_lhs_quant=False):
        calls.append(n_window)
        tiles.append(tile_m)
        hoists.append(hoist_lhs_quant)
        r = (rhs.astype(jnp.float32) * rhs_scale[:, 0]).astype(jnp.bfloat16)
        if n_window is not None:
            r = r[..., n_window[0] : n_window[0] + n_window[1]]
        out = jax.lax.ragged_dot(lhs, r, group_sizes, preferred_element_type=jnp.float32)
        if fuse_act == "silu":
            h = out.shape[1] // 2
            out = jax.nn.silu(out[:, :h]) * out[:, h:]
        return out.astype(preferred_element_type or lhs.dtype)

    return gmm


def _ref_combine(rows, revert, weights, valid, reduce_group_size):
    """Destination-major weighted sum in f32, per column."""
    t = revert.shape[0] // reduce_group_size
    picked = rows[revert].astype(jnp.float32).reshape(t, reduce_group_size, -1)
    w = (weights.astype(jnp.float32) * valid).reshape(t, reduce_group_size, 1)
    return (picked * w).sum(axis=1)


@pytest.mark.parametrize("split", [2, 4])
def test_combine_split_matches_one_gmm_and_one_combine(monkeypatch, split):
    """TPU_MOE_NONEP_COMBINE_SPLIT: per-column-block GMM2 + combine gives the
    unsplit output bitwise, and each block's GMM2 reads its own columns."""
    experts, hidden, inter, topk, tokens = 16, 512, 128, 4, 640  # 2,560 routes: the prefill (multi-sort) path
    k1, k2, k3 = jax.random.split(jax.random.key(3), 3)
    w1, s1 = _fp8_per_channel(k1, (experts, hidden, 2 * inter))
    w2, s2 = _fp8_per_channel(k2, (experts, inter, hidden))
    x = (jax.random.normal(k3, (tokens, hidden), jnp.float32) / 4).astype(jnp.bfloat16)
    rng = np.random.default_rng(5)
    ids = jnp.asarray(np.stack([rng.choice(experts, topk, replace=False) for _ in range(tokens)]), jnp.int32)
    w = jnp.asarray(rng.random((tokens, topk)), jnp.bfloat16)
    calls = []
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", _ref_gmm(calls))
    monkeypatch.setattr(fused_moe_gmm, "ragged_gather_reduce_v3", _ref_combine)
    monkeypatch.setattr(fused_moe_gmm, "_local_ragged_combine_eligible", lambda *a, **k: True)
    xs, gs, rev, wf, valid, tok = prepare_routed_gmm_inputs(
        x, ids, w, local_num_experts=experts, topk=topk, use_ep=False, use_sparse_core=True
    )

    def run(n):
        return fused_moe_gmm.moe_gmm(
            xs, w1, s1, None, w2, s2, None, gs, rev, wf, valid, tok, activation="silu",
            num_tokens=tokens, topk=topk, use_ep=False, use_sparse_core=True,
            nonep_combine_split=n,
        )

    ref = run(1)
    assert calls == [None, None]
    calls.clear()
    got = run(split)
    width = hidden // split
    assert calls == [None] + [(j * width, width) for j in range(split)]
    assert got.shape == ref.shape == (tokens, hidden)
    np.testing.assert_array_equal(_bits(got), _bits(ref))
    assert np.asarray(ref, np.float32).std() > 0


def test_tiling_with_tile_m_overrides_only_the_m_tiling(monkeypatch):
    from vllm_torchtpu.kernels.megablox.gmm_v2 import Dimensions, TileSizes

    default = TileSizes(tile_m=512, tile_k=2560, tile_n=640, bucket_base=128)
    monkeypatch.setattr(fused_moe_gmm, "calculate_tiling", lambda *a, **k: default)
    fused_moe_gmm.tiling_with_tile_m.cache_clear()
    dims = Dimensions(size_m=64000, size_k=2560, size_n=1280, size_group=512,
                      size_lhs_group=512, size_lhs_sublane=16)
    t = lambda tm, d=dims: fused_moe_gmm.tiling_with_tile_m(tm)(d, None, None, 0, "silu")  # noqa: E731
    assert t(1024) == TileSizes(1024, 2560, 640, 256)
    assert t(2048) == TileSizes(2048, 2560, 640, 512)
    assert t(256) == TileSizes(256, 2560, 640, 128)
    assert t(512) == default
    # small steps never get a tile larger than their (bucket-aligned) row count
    small = Dimensions(size_m=300, size_k=2560, size_n=1280, size_group=512,
                       size_lhs_group=512, size_lhs_sublane=16)
    assert t(2048, small).tile_m == 384
    # the factory returns one function per value: a stable static argument
    assert fused_moe_gmm.tiling_with_tile_m(1024) is fused_moe_gmm.tiling_with_tile_m(1024)


@pytest.mark.parametrize("use_ep", [False, True])
def test_gmm_tile_m_and_hoist_reach_both_gmms_on_the_non_ep_path_only(monkeypatch, use_ep):
    experts, hidden, inter, topk, tokens = 16, 512, 128, 4, 640
    k1, k2, k3 = jax.random.split(jax.random.key(4), 3)
    w1, s1 = _fp8_per_channel(k1, (experts, hidden, 2 * inter))
    w2, s2 = _fp8_per_channel(k2, (experts, inter, hidden))
    x = (jax.random.normal(k3, (tokens, hidden), jnp.float32) / 4).astype(jnp.bfloat16)
    rng = np.random.default_rng(6)
    ids = jnp.asarray(np.stack([rng.choice(experts, topk, replace=False) for _ in range(tokens)]), jnp.int32)
    w = jnp.asarray(rng.random((tokens, topk)), jnp.bfloat16)
    calls, tiles, hoists = [], [], []
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", _ref_gmm(calls, tiles, hoists))
    xs, gs, rev, wf, valid, tok = prepare_routed_gmm_inputs(
        x, ids, w, local_num_experts=experts, topk=topk, use_ep=False, use_sparse_core=False
    )
    fused_moe_gmm.moe_gmm(
        xs, w1, s1, None, w2, s2, None, gs, rev, wf, valid, tok, activation="silu",
        num_tokens=tokens, topk=topk, use_ep=use_ep, use_sparse_core=False, nonep_gmm_tile_m=1024,
        nonep_gmm_hoist_quant=True,
    )
    assert tiles == ([0, 0] if use_ep else [1024, 1024])
    assert hoists == ([False, False] if use_ep else [True, True])


# --------------------------------------------------------------------------- TPU


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _fp8_per_channel(key, shape):
    """Per-output-channel fp8 weight [E, K, N] and its f32 scale [E, 1, 1, N],
    the layout fp8.py's requantization hands the GMM kernel."""
    w = jax.random.normal(key, shape, jnp.float32) / 16
    amax = jnp.max(jnp.abs(w), axis=1, keepdims=True)
    scale = amax / float(jnp.finfo(jnp.float8_e4m3fn).max)
    q = (w / scale).astype(jnp.float8_e4m3fn)
    return q, scale[:, :, None, :].astype(jnp.float32)


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [3200, 6400])
def test_fused_moe_func_nonep_choices_bitwise_on_tpu(tokens):
    _require_tpu()
    k1, k2, k3 = jax.random.split(jax.random.key(7), 3)
    w1, s1 = _fp8_per_channel(k1, (EXPERTS, HIDDEN, 2 * INTER))
    w2, s2 = _fp8_per_channel(k2, (EXPERTS, INTER, HIDDEN))
    x = (jax.random.normal(k3, (tokens, HIDDEN), jnp.float32) / 4).astype(jnp.bfloat16)
    ids, w = _routing(tokens)
    threshold = fused_moe_gmm.resolve_onehot_permute_threshold()

    def run(route_ids=ids, **kw):
        return fused_moe_gmm.fused_moe_func(
            x, w1, w2, s1, s2, None, None, w, route_ids, topk=TOPK, activation="silu",
            use_ep=False, use_sparse_core=True,
            onehot_moe_permute_threshold=threshold, **kw,
        )

    out = run()
    ref = _bits(out)
    assert np.isfinite(np.asarray(out, np.float32)).all()
    for kw in (
        dict(nonep_gather="xla_colchunk"),
        dict(nonep_gather="xla_colchunk", nonep_gather_cols=640),
        dict(nonep_gather="sc"),
        dict(nonep_gather="xla_chunked"),
        dict(nonep_permute="two_sorts"),
        dict(nonep_permute="counting"),
        dict(nonep_permute="two_sorts", nonep_gather="xla_colchunk"),
        dict(nonep_combine_split=2),
        dict(nonep_combine_split=4),
        dict(nonep_combine_split=2, nonep_gather="xla_colchunk"),
        dict(nonep_gmm_tile_m=256),
        dict(nonep_gmm_tile_m=1024),
        dict(nonep_gmm_tile_m=2048),
        dict(nonep_gmm_hoist_quant=True),
        dict(nonep_gather="xla_colchunk", nonep_combine_split=2, nonep_permute="two_sorts",
             nonep_gmm_tile_m=1024),
    ):
        np.testing.assert_array_equal(_bits(run(**kw)), ref, err_msg=str(kw))
    # The comparison must be able to fail: moving token 0's first route to an
    # expert it does not use changes its output row.
    unused = int(np.setdiff1d(np.arange(EXPERTS), np.asarray(ids[0]))[0])
    assert (_bits(run(route_ids=ids.at[0, 0].set(unused))) != ref).any()


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [3200, 6400])
def test_gmm_v2_n_window_is_a_bitwise_column_slice_on_tpu(tokens):
    """gmm_v2(n_window=(start, width)) returns exactly columns [start, start +
    width) of the full output, and a different window does not."""
    _require_tpu()
    from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2

    k1, k2 = jax.random.split(jax.random.key(11))
    w2, s2 = _fp8_per_channel(k1, (EXPERTS, INTER, HIDDEN))
    lhs = (jax.random.normal(k2, (tokens * TOPK, INTER), jnp.float32) / 4).astype(jnp.bfloat16)
    ids, _ = _routing(tokens)
    gs = jnp.bincount(ids.reshape(-1), length=EXPERTS).astype(jnp.int32)
    kw = dict(rhs_scale=s2, group_offset=jnp.array([0], jnp.int32), zero_initialize=False,
              preferred_element_type=jnp.bfloat16)
    full = _bits(gmm_v2(lhs, w2, gs, **kw))
    halves = [_bits(gmm_v2(lhs, w2, gs, n_window=(st, 1280), **kw)) for st in (0, 1280)]
    np.testing.assert_array_equal(np.concatenate(halves, axis=1), full)
    quarter = _bits(gmm_v2(lhs, w2, gs, n_window=(640, 640), **kw))
    np.testing.assert_array_equal(quarter, full[:, 640:1280])
    assert (halves[1] != full[:, :1280]).any()


@pytest.mark.parametrize("scopes,level,on", [(0, 2, False), (1, 2, False), (2, 2, True), (3, 2, True)])
def test_gmm_v2_trace_scopes_gate(scopes, level, on):
    """Phase trace scopes exist only when asked for; off is a no-op context."""
    import contextlib

    from vllm_torchtpu.kernels.megablox.gmm_v2 import GmmConfigs, trace_scope

    cfgs = GmmConfigs(tiles=None, dims=None, lhs_cfgs=None, rhs_cfgs=None, out_dtype=None,
                      acc_dtype=None, zero_init=False, fuse_act=None, trace_scopes=scopes)
    scope = trace_scope(cfgs, "gmm_mxu", level)
    assert isinstance(scope, contextlib.nullcontext) is (not on)
    with scope:
        pass


@pytest.mark.parametrize("value,expected", [(None, ""), ("skip_math", "skip_math"), (" half_cols ", "half_cols")])
def test_sc_combine_bench_knob_is_read_from_the_environment(monkeypatch, value, expected):
    """TPU_SC_COMBINE_BENCH (bench-only attribution) is unset by default."""
    from vllm_torchtpu.kernels.sparse_core import ragged_gather_reduce_v3 as v3

    if value is None:
        monkeypatch.delenv("TPU_SC_COMBINE_BENCH", raising=False)
    else:
        monkeypatch.setenv("TPU_SC_COMBINE_BENCH", value)
    assert v3._bench_mode() == expected
