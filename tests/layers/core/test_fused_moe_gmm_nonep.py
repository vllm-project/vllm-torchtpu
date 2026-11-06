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


def _ref_gmm(calls, tiles=None, hoists=None, splits=None, epvs=None):
    """Pure-JAX stand-in for gmm_wrapper that records its n_window (and
    tile_m); every output column is computed independently, as the kernel's
    are."""
    tiles = [] if tiles is None else tiles
    hoists = [] if hoists is None else hoists
    splits = [] if splits is None else splits
    epvs = [] if epvs is None else epvs

    def gmm(lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset, zero_initialize=False,
            fuse_act=None, preferred_element_type=None, rhs_quant_dtype=None, n_window=None,
            tile_m=0, hoist_lhs_quant=False, rhs_split=1, experts_per_visit=1):
        calls.append(n_window)
        epvs.append(experts_per_visit)
        tiles.append(tile_m)
        hoists.append(hoist_lhs_quant)
        splits.append(rhs_split)
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
    calls, tiles, hoists, splits = [], [], [], []
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", _ref_gmm(calls, tiles, hoists, splits))
    xs, gs, rev, wf, valid, tok = prepare_routed_gmm_inputs(
        x, ids, w, local_num_experts=experts, topk=topk, use_ep=False, use_sparse_core=False
    )
    fused_moe_gmm.moe_gmm(
        xs, w1, s1, None, w2, s2, None, gs, rev, wf, valid, tok, activation="silu",
        num_tokens=tokens, topk=topk, use_ep=use_ep, use_sparse_core=False, nonep_gmm_tile_m=1024,
        nonep_gmm_hoist_quant=True, nonep_gmm2_rhs_split=4,
    )
    assert tiles == ([0, 0] if use_ep else [1024, 1024])
    assert hoists == ([False, False] if use_ep else [True, True])
    # the weight-DMA split applies to gmm2 only (gmm1 already has gate/up DMAs)
    assert splits == ([1, 1] if use_ep else [1, 4])


@pytest.mark.parametrize("use_ep", [False, True])
@pytest.mark.parametrize("split", [1, 2])
def test_experts_per_visit_reaches_every_gmm_on_the_non_ep_path_only(monkeypatch, use_ep, split):
    experts, hidden, inter, topk, tokens = 16, 512, 128, 4, 640
    k1, k2, k3 = jax.random.split(jax.random.key(5), 3)
    w1, s1 = _fp8_per_channel(k1, (experts, hidden, 2 * inter))
    w2, s2 = _fp8_per_channel(k2, (experts, inter, hidden))
    x = (jax.random.normal(k3, (tokens, hidden), jnp.float32) / 4).astype(jnp.bfloat16)
    rng = np.random.default_rng(7)
    ids = jnp.asarray(np.stack([rng.choice(experts, topk, replace=False) for _ in range(tokens)]), jnp.int32)
    w = jnp.asarray(rng.random((tokens, topk)), jnp.bfloat16)
    calls, epvs = [], []
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", _ref_gmm(calls, epvs=epvs))
    monkeypatch.setattr(fused_moe_gmm, "_local_ragged_combine_eligible", lambda *a, **k: True)
    monkeypatch.setattr(fused_moe_gmm, "ragged_gather_reduce_v3",
                        lambda rows, rev, wts, valid, reduce_group_size: _ref_combine(
                            rows, rev, wts, valid, reduce_group_size))
    xs, gs, rev, wf, valid, tok = prepare_routed_gmm_inputs(
        x, ids, w, local_num_experts=experts, topk=topk, use_ep=False, use_sparse_core=False
    )
    fused_moe_gmm.moe_gmm(
        xs, w1, s1, None, w2, s2, None, gs, rev, wf, valid, tok, activation="silu",
        num_tokens=tokens, topk=topk, use_ep=use_ep, use_sparse_core=False,
        nonep_combine_split=split, nonep_gmm_experts_per_visit=4,
    )
    # gmm1, then gmm2 once per column block when the combine is split
    assert len(epvs) == 1 + split
    assert epvs == [1] * (1 + split) if use_ep else epvs == [4] * (1 + split)


class _FakeRef:
    """Minimal ref stand-in: .at[idx] is a view (here a numpy slice)."""

    def __init__(self, a):
        self.a = a

    @property
    def at(self):
        return _FakeAt(self.a)


class _FakeAt:
    def __init__(self, a):
        self.a = a

    def __getitem__(self, idx):
        return _FakeRef(self.a[idx])


def test_weights_ref_select_takes_one_expert_of_the_block():
    from vllm_torchtpu.kernels.megablox.gmm_v2 import FusedWeightsRef, WeightsRef

    e, k, n = 4, 8, 6
    w = np.arange(e * k * 2 * n).reshape(e, k, 2 * n)
    sc = np.arange(e * 2 * 2 * n).reshape(e, 2, 1, 2 * n)
    b = np.arange(e * 2 * n).reshape(e, 1, 2 * n)
    gate = WeightsRef(weight=_FakeRef(w[..., :n]), scale=_FakeRef(sc[..., :n]), bias=_FakeRef(b[..., :n]))
    up = WeightsRef(weight=_FakeRef(w[..., n:]), scale=_FakeRef(sc[..., n:]), bias=None)
    for i in range(e):
        one = WeightsRef(weight=_FakeRef(w), scale=None, bias=None).select(i)
        np.testing.assert_array_equal(one.weight.a, w[i])
        assert one.scale is None and one.bias is None
        f = FusedWeightsRef(gate=gate, up=up).select(i)
        np.testing.assert_array_equal(f.gate.weight.a, w[i, :, :n])
        np.testing.assert_array_equal(f.up.weight.a, w[i, :, n:])
        np.testing.assert_array_equal(f.gate.scale.a, sc[i, ..., :n])
        np.testing.assert_array_equal(f.gate.bias.a, b[i, ..., :n])
        assert f.up.bias is None


@pytest.mark.parametrize("tile_m,bucket_base", [(512, 128), (256, 64), (128, 32)])
def test_sub_expert_window_covers_the_rows_inside_the_tile(tile_m, bucket_base):
    """Every sub-expert row range [lo, hi) of a tile gets a sublane-aligned
    window [start, start + bucket) that covers it, stays inside the tile, and
    uses the smallest bucket that covers [lo-rounded-down, hi) when it fits."""
    from vllm_torchtpu.kernels.megablox.gmm_v2 import sub_expert_window

    sub = 16
    for lo in range(0, tile_m):
        for hi in range(lo + 1, tile_m + 1, 3):
            idx, start = sub_expert_window(lo, hi, sublane=sub, bucket_base=bucket_base, tile_m=tile_m)
            bucket = bucket_base * (idx + 1)
            assert 0 <= idx < tile_m // bucket_base
            assert start % sub == 0 and 0 <= start <= lo
            assert start + bucket >= hi and start + bucket <= tile_m
            assert bucket - bucket_base < hi - (lo // sub) * sub
    # traced values give the same window
    idx, start = sub_expert_window(jnp.int32(130), jnp.int32(260), sublane=sub,
                                   bucket_base=bucket_base, tile_m=tile_m)
    assert (int(idx), int(start)) == sub_expert_window(130, 260, sublane=sub,
                                                       bucket_base=bucket_base, tile_m=tile_m)


def test_check_experts_per_visit_rejects_what_the_kernel_cannot_do():
    from vllm_torchtpu.kernels.megablox.gmm_v2 import check_experts_per_visit

    ok = dict(size_group=512, size_lhs_group=512, size_k=640, tile_k=640, rhs_split=1)
    for e in (2, 4):
        check_experts_per_visit(e, **ok)
    for e, bad in ((3, {}), (8, {}), (2, dict(size_group=511, size_lhs_group=511)),
                   (2, dict(size_lhs_group=1024)), (2, dict(tile_k=320)), (2, dict(rhs_split=2))):
        with pytest.raises(ValueError):
            check_experts_per_visit(e, **{**ok, **bad})


@pytest.mark.parametrize("topk", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20])
def test_dense_runs_reproduce_the_all_valid_route_to_token_map(topk):
    """The combine's run table: in an all-valid block route r belongs to
    token r // topk. Every chunk's runs cover its 16 lanes in order, each run
    is one token's maximal stretch of consecutive lanes, and the tokens are
    the true ones for every chunk of a 64-token block."""
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import dense_runs

    n_phase, tok_per_period, table = dense_runs(topk)
    assert n_phase * 16 == tok_per_period * topk  # one period, in chunks and in tokens
    assert len(table) == n_phase
    for chunk in range(64 * topk // 16):
        period, phase = divmod(chunk, n_phase)
        lanes = []
        for tok, first, length in table[phase]:
            assert length >= 1
            for lane in range(first, first + length):
                lanes.append(lane)
                assert period * tok_per_period + tok == (chunk * 16 + lane) // topk
        assert lanes == list(range(16))
        toks = [t for t, _, _ in table[phase]]
        assert toks == sorted(set(toks))  # maximal runs: one per token, in order
    if topk == 10:
        assert table == (
            ((0, 0, 10), (1, 10, 6)),
            ((1, 0, 4), (2, 4, 10), (3, 14, 2)),
            ((3, 0, 8), (4, 8, 8)),
            ((4, 0, 2), (5, 2, 10), (6, 12, 4)),
            ((6, 0, 6), (7, 6, 10)),
        )


def test_dense_combine_mode_is_read_and_checked(monkeypatch):
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import _dense_mode

    for value, expected in (("", ""), ("reg", "reg"), (" addupdate ", "addupdate")):
        monkeypatch.setenv("TPU_SC_COMBINE_DENSE", value)
        assert _dense_mode() == expected
    monkeypatch.delenv("TPU_SC_COMBINE_DENSE")
    assert _dense_mode() == ""
    monkeypatch.setenv("TPU_SC_COMBINE_DENSE", "registers")
    with pytest.raises(ValueError):
        _dense_mode()


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


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [3200, 6400])
@pytest.mark.parametrize("epv", [2, 4])
def test_gmm_v2_experts_per_visit_is_bitwise_on_tpu(tokens, epv):
    """Both served GMM shapes (gmm1 fused SiLU k 2560, gmm2 k 640), Qwen3.8
    routing sizes: E consecutive experts per visit give bitwise the one-expert
    output, including signed zeros."""
    _require_tpu()
    from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2

    experts, hidden, inter, topk = 512, 2560, 640, 10
    rng = np.random.default_rng(tokens + epv)
    rows = tokens * topk
    sizes = rng.multinomial(rows, rng.dirichlet(np.ones(experts)))
    sizes[rng.integers(0, experts, 8)] = 0
    sizes[0] += rows - sizes.sum()
    gs = jnp.asarray(sizes, jnp.int32)
    k1, k2, k3 = jax.random.split(jax.random.key(tokens), 3)
    w1, s1 = _fp8_per_channel(k1, (experts, hidden, 2 * inter))
    w2, s2 = _fp8_per_channel(k2, (experts, inter, hidden))
    x = (jax.random.normal(k3, (rows, hidden), jnp.float32) / 4).astype(jnp.bfloat16)
    for lhs, w, s, act in ((x, w1, s1, "silu"), (x[:, :inter], w2, s2, None)):
        outs = [gmm_v2(lhs, w, gs, rhs_scale=s, zero_initialize=False, fuse_act=act,
                       preferred_element_type=jnp.bfloat16, experts_per_visit=e)
                for e in (1, epv)]
        a, b = (np.asarray(o).view(np.uint16) for o in outs)
        assert np.array_equal(a, b), (act, int((a != b).sum()))


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



def test_split_part_maps_columns_to_one_part():
    from vllm_torchtpu.kernels.megablox.gmm_v2 import split_part

    assert split_part(0, 256, 640) == (0, 0, 256)
    assert split_part(512, 640, 640) == (0, 512, 640)
    assert split_part(640, 896, 640) == (1, 0, 256)
    assert split_part(2304, 2560, 640) == (3, 384, 640)
    for bad in ((512, 768), (0, 0)):
        with pytest.raises(ValueError):
            split_part(*bad, 640)


@pytest.mark.parametrize("nsplit", [2, 4, 5])
def test_split_weights_ref_blocks_equal_the_whole_tile(nsplit):
    """Every mxu-wide block read from the parts equals the same block of the
    whole tile, and the concatenation of the parts is the tile."""
    from vllm_torchtpu.kernels.megablox.gmm_v2 import SplitWeightsRef, WeightsRef

    k, n, mxu = 640, 2560, 256
    tile = np.arange(k * n, dtype=np.int32).reshape(k, n)
    w = n // nsplit
    ref = SplitWeightsRef(parts=tuple(WeightsRef(weight=tile[:, j * w:(j + 1) * w], scale=None, bias=None)
                                      for j in range(nsplit)))
    np.testing.assert_array_equal(np.asarray(ref.get_weight()), tile)
    for n0 in range(0, n, mxu):
        n1 = min(n, n0 + mxu)
        if n0 // w != (n1 - 1) // w:
            continue   # the kernel requires blocks inside one part (lane-aligned parts, mxu-aligned chunks)
        for k0, k1 in ((0, 512), (512, 640)):
            np.testing.assert_array_equal(ref.weight_block(k0, k1, n0, n1), tile[k0:k1, n0:n1])


def test_sc_padded_gather_matches_sc_and_keeps_padded_rows(monkeypatch):
    """TPU_MOE_NONEP_GATHER=sc_padded: the gathered rows keep the kernel's
    padded row blocks (no trimming copy), and the MoE output is bitwise the
    trimmed path's, because no GMM visit or combine index reaches the extra rows."""
    experts, hidden, inter, topk, tokens = 16, 512, 128, 4, 640
    k1, k2, k3 = jax.random.split(jax.random.key(9), 3)
    w1, s1 = _fp8_per_channel(k1, (experts, hidden, 2 * inter))
    w2, s2 = _fp8_per_channel(k2, (experts, inter, hidden))
    x = (jax.random.normal(k3, (tokens, hidden), jnp.float32) / 4).astype(jnp.bfloat16)
    rng = np.random.default_rng(8)
    ids = jnp.asarray(np.stack([rng.choice(experts, topk, replace=False) for _ in range(tokens)]), jnp.int32)
    w = jnp.asarray(rng.random((tokens, topk)), jnp.bfloat16)

    def fake_sc_gather(src, idx, start, end, trim_rows=True):
        # the kernel writes whole 2,048-row blocks; padded slots gather row 0
        pad = -idx.shape[0] % 2048
        out = src[jnp.pad(idx, (0, pad))]
        return out[: idx.shape[0]] if trim_rows else out

    monkeypatch.setattr(fused_moe_gmm, "ragged_gather_v2", fake_sc_gather)
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", _ref_gmm([]))
    padded = prepare_routed_gmm_inputs(x, ids, w, local_num_experts=experts, topk=topk, use_ep=False,
                                       use_sparse_core=True, nonep_gather="sc_padded")[0]
    assert padded.shape[0] == 4096 and tokens * topk == 2560

    def run(gather):
        return fused_moe_gmm.fused_moe_func.__wrapped__(
            x, w1, w2, s1, s2, None, None, w, ids, topk=topk, activation="silu",
            use_ep=False, use_sparse_core=True, nonep_gather=gather)

    ref = _bits(run("xla"))
    np.testing.assert_array_equal(_bits(run("sc")), ref)
    np.testing.assert_array_equal(_bits(run("sc_padded")), ref)
