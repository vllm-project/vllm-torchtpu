# SPDX-License-Identifier: Apache-2.0
"""Opt-in forward post-processing kernel checks; run this file explicitly."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import fused_greedy_head as kernel
from vllm_torchtpu.kernels.attention_preprocessing import qa_rope, rope_gather
from vllm_torchtpu.kernels.deepseek_v41.cache_write import write_cache_rows_banked
from vllm_torchtpu.kernels.markov_sampling import sample_slot
from vllm_torchtpu.kernels.shared_record_attention import shared_record_attention
from vllm_torchtpu.runner import utils

RANK, HEAD_DIM, HIDDEN = 1280, 512, 5120


def _inputs(tokens, positions, cache_rows=4096, hidden=HIDDEN):
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.normal(size=(tokens, hidden)), jnp.bfloat16)
    # Zero row (scale 1 branch) and a tiny row near the FP8 underflow edge.
    x = x.at[0].set(0).at[1].multiply(1e-6)
    weight = jnp.asarray(rng.normal(size=(hidden, RANK + HEAD_DIM)), jnp.float8_e4m3fn)
    scale = jnp.asarray(rng.uniform(0.01, 0.1, RANK + HEAD_DIM), jnp.float32)
    gamma_qr = jnp.asarray(rng.normal(size=(RANK,)), jnp.bfloat16)
    gamma_kv = jnp.asarray(rng.normal(size=(HEAD_DIM,)), jnp.bfloat16)
    angles = rng.uniform(-3, 3, size=(cache_rows, 32))
    cache = jnp.asarray(
        np.concatenate((np.cos(angles), np.sin(angles)), 1), jnp.float32
    )
    rows = rope_gather.gather_rows(cache, jnp.asarray(positions, jnp.int32))
    return x, weight, scale, gamma_qr, gamma_kv, rows


def _positions(tokens, kind):
    if kind == "contiguous":
        return np.arange(tokens) + 17
    if kind == "jumps":  # request boundaries
        half = tokens // 2
        return np.concatenate([np.arange(half) + 3000, np.arange(tokens - half)])
    positions = np.arange(tokens) + 4096 - tokens // 2  # past the table end
    positions[:4] = -1
    return positions


def _requires_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("Requires a TPU backend")


@pytest.mark.parametrize("tokens", [288, 240, 2048, 37])
def test_kv_layers_match_single_layer_calls(tokens):
    """One multi-layer call equals each layer's own KV-only call, bitwise."""
    _requires_tpu()
    x, weight, scale, gamma_qr, gamma_kv, rows = _inputs(
        tokens, _positions(tokens, "jumps")
    )
    rng = np.random.default_rng(23)
    weights = [weight] + [
        jnp.asarray(rng.normal(size=weight.shape), jnp.float8_e4m3fn) for _ in range(2)
    ]
    scales = [scale] + [
        jnp.asarray(rng.uniform(0.01, 0.1, scale.shape), jnp.float32) for _ in range(2)
    ]
    gammas = [gamma_kv] + [
        jnp.asarray(rng.normal(size=gamma_kv.shape), jnp.bfloat16) for _ in range(2)
    ]
    fused = jax.jit(qa_rope.fused_kv_norm_rope_layers)(
        x, tuple(weights), tuple(scales), tuple(gammas), rows
    )
    single = jax.jit(functools.partial(qa_rope.fused_qa_kv_norm_rope, emit_qr=False))
    assert len(fused) == 3
    for w, s, g, got in zip(weights, scales, gammas, fused, strict=True):
        _, ref = single(x, w, s, gamma_qr, g, rows)
        assert got.shape == ref.shape and got.dtype == ref.dtype
        np.testing.assert_array_equal(
            np.asarray(got).view(np.uint16), np.asarray(ref).view(np.uint16)
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("rows", [4096, 4097, 99328])
@pytest.mark.parametrize("seed", range(3))
def test_paired_rows_match_gather_rows(rows, seed):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-3, 3, size=(rows, 32))
    table = np.concatenate((np.cos(angles), np.sin(angles)), 1).astype(np.float32)
    table[0, 0] = -0.0  # sign of zero must survive the select
    # Out-of-range positions on both sides exercise the clamp.
    positions = rng.integers(-5, rows + 5, 288).astype(np.int32)
    positions[:3] = (0, rows - 1, rows)
    pairs = utils._pair_rows(torch.from_numpy(table)).numpy()
    assert pairs.shape == ((rows + 1) // 2, 128)
    ref = rope_gather.gather_rows(jnp.asarray(table), jnp.asarray(positions))
    got = rope_gather.gather_paired_rows(jnp.asarray(pairs), jnp.asarray(positions))
    assert got.dtype == ref.dtype and got.shape == ref.shape
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint32), np.asarray(ref).view(np.uint32)
    )


@pytest.mark.parametrize(
    "rows,tokens,seed",
    [
        (rows, tokens, rows + tokens)
        for rows in (4096, 4097, 99328)
        for tokens in (1, 288, 300)
    ]
    + [(98816, 288, seed) for seed in range(3)],
)
def test_sc_paired_rows_match_gather_rows(rows, tokens, seed):
    """The SparseCore row fetch returns the same bits as the XLA gather."""
    import jax

    if jax.default_backend() != "tpu":
        pytest.skip("Requires a TPU backend")
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-3, 3, size=(rows, 32))
    table = np.concatenate((np.cos(angles), np.sin(angles)), 1).astype(np.float32)
    table[0, 0] = -0.0
    positions = rng.integers(-5, rows + 5, tokens).astype(np.int32)
    positions[: min(3, tokens)] = (0, rows - 1, rows)[: min(3, tokens)]
    pairs = utils._pair_rows(torch.from_numpy(table)).numpy()
    ref = rope_gather.gather_rows(jnp.asarray(table), jnp.asarray(positions))
    got = jax.jit(rope_gather.gather_paired_rows_sc)(
        jnp.asarray(pairs), jnp.asarray(positions)
    )
    assert got.dtype == ref.dtype and got.shape == ref.shape
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint32), np.asarray(ref).view(np.uint32)
    )


@pytest.fixture
def sparsecore_device():
    if jax.default_backend() != "tpu":
        pytest.skip("Row writes require a TPU with SparseCore")
    if pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("Row writes require SparseCore")


def _random_words(rng, shape):
    return np.asarray(rng.integers(-(2**31), 2**31, shape), np.int32)


def _expected(cache, rows, words):
    expected = np.array(cache)
    for row, word in zip(rows, words):
        if 0 <= row < expected.shape[0]:
            expected[row] = word
    return expected


@pytest.mark.parametrize("lo_rows,hi_rows", [(8192, 8192), (8192, 3000), (512, 7000)])
@pytest.mark.parametrize("count", [1, 37, 864, 4096])
def test_banked_row_writes_match_per_bank_writes(
    sparsecore_device, lo_rows, hi_rows, count
):
    # Rows address [lo; hi]; out-of-range and negative rows are dropped, and
    # each bank must equal what one write_cache_rows per bank leaves.
    rng = np.random.default_rng(count + lo_rows + hi_rows)
    total = lo_rows + hi_rows
    lo = _random_words(rng, (lo_rows, 128))
    hi = _random_words(rng, (hi_rows, 128))
    rows = rng.permutation(total)[: min(count, total)].astype(np.int32)
    rows[3::11] = total + rows[3::11]
    rows[::7] = -1
    words = _random_words(rng, (rows.shape[0], 128))
    got_lo, got_hi = write_cache_rows_banked(
        jnp.asarray(lo), jnp.asarray(hi), jnp.asarray(rows), jnp.asarray(words)
    )
    lo_part = np.where((rows >= 0) & (rows < lo_rows), rows, -1).astype(np.int32)
    hi_part = np.where(rows >= lo_rows, rows - lo_rows, -1).astype(np.int32)
    np.testing.assert_array_equal(got_lo, _expected(lo, lo_part, words))
    np.testing.assert_array_equal(got_hi, _expected(hi, hi_part, words))


def test_banked_row_writes_reject_incompatible_shapes():
    good = jnp.zeros((8, 128), jnp.int32)
    with pytest.raises(ValueError, match="INT32"):
        write_cache_rows_banked(
            good,
            jnp.zeros((8, 64), jnp.int32),
            jnp.zeros(1, jnp.int32),
            jnp.zeros((1, 128), jnp.int32),
        )
    # Two distinct owners: both are donated, and one buffer cannot be
    # donated twice.
    lo, hi = write_cache_rows_banked(
        jnp.zeros((8, 128), jnp.int32),
        jnp.zeros((8, 128), jnp.int32),
        jnp.zeros(0, jnp.int32),
        jnp.zeros((0, 128), jnp.int32),
    )
    np.testing.assert_array_equal(lo, 0)
    np.testing.assert_array_equal(hi, 0)


@pytest.mark.parametrize("lo_rows,hi_rows", [(0, 8), (8, 0), (0, 0)])
@pytest.mark.parametrize("count", [0, 1])
def test_banked_row_writes_reject_empty_owners_before_device_query(
    monkeypatch, lo_rows, hi_rows, count
):
    def unexpected_device_query():
        pytest.fail("Empty owners must fail before inspecting TPU hardware")

    monkeypatch.setattr(pltpu, "get_tpu_info", unexpected_device_query)
    with pytest.raises(ValueError, match="non-empty owners"):
        write_cache_rows_banked(
            jnp.zeros((lo_rows, 128), jnp.int32),
            jnp.zeros((hi_rows, 128), jnp.int32),
            jnp.zeros(count, jnp.int32),
            jnp.zeros((count, 128), jnp.int32),
        )


@pytest.fixture
def cpu_jax_backend():
    """Keep the marked lookup/combine checks on CPU."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.mark.usefixtures("cpu_jax_backend")
@pytest.mark.parametrize("onehot", [False, True])
def test_combine_arrivals_isolates_nan_on_unrouted_padded_tokens(onehot):
    """NaN arrival rows stay isolated to padded rows and zero out via row_routes."""
    from vllm_torchtpu.kernels.fused_moe.v2.layer import _combine_arrivals

    # Use t_local=8 so combine_step_tokens(t_local) is 0 and the XLA epilogue runs
    # on CPU without invoking the Pallas TPU combine kernel.
    t_local, topk, lane_blocks, lanes, tile_m = 8, 4, 2, 128, 128
    recv_rows = 32
    arrivals = np.ones((recv_rows, lane_blocks, lanes), dtype=np.float32)
    arrival_scales = np.ones((4, tile_m), dtype=np.float32)

    # Sentinel pairs receive `pos = mirror_pos =` their rank among sentinel pairs
    # within the routing block (`[0, block)`). Poison the entire range `[0, 16)`
    # that the 4 padded tokens (`4 * topk = 16` pairs) index into.
    sentinel_ranks = np.arange(4 * topk, dtype=np.int32).reshape(4, topk)
    arrivals[: 4 * topk] = np.nan
    arrival_scales[0, : 4 * topk] = np.nan

    pos = np.full((t_local, topk), 16, dtype=np.int32)
    mirror_pos = np.full((t_local, topk), 16, dtype=np.int32)
    topk_weights = np.full((t_local, topk), 0.25, dtype=np.float32)

    # Mark rows 4..7 as padded: pos and mirror_pos span [0, 16), topk_weights=0.0.
    pos[4:] = sentinel_ranks
    mirror_pos[4:] = sentinel_ranks
    topk_weights[4:] = 0.0
    row_routes = jnp.asarray([[True]] * 4 + [[False]] * 4)

    combined = _combine_arrivals(
        jnp.asarray(arrivals),
        jnp.asarray(arrival_scales),
        jnp.asarray(pos),
        jnp.asarray(mirror_pos),
        jnp.asarray(topk_weights),
        jnp.bfloat16,
        onehot_scale_lookup=onehot,
    )
    out = np.asarray(jnp.where(row_routes, combined, jnp.zeros_like(combined)))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:4].astype(np.float32), 1.0)
    np.testing.assert_array_equal(
        out[4:].astype(np.float32), np.zeros((4, lane_blocks * lanes), np.float32)
    )


@pytest.mark.usefixtures("cpu_jax_backend")
@pytest.mark.parametrize("onehot", [False, True])
@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("lookups,sublanes", [(720, 264), (32, 4), (7, 1)])
def test_row_scale_lookup_matches_gather_bit_for_bit(
    onehot, enable_x64, lookups, sublanes
):
    """The one-hot lookup returns the gather's bits, NaN payloads included,
    and a NaN reaches only the lookups that read it."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.fused_moe.v2.layer import _lookup_row_scales

    rng = np.random.default_rng(lookups + sublanes)
    bits = rng.integers(0, 2**32, (sublanes, 128), dtype=np.uint64).astype(np.uint32)
    poison = rng.random(bits.shape) < 0.05
    payload = rng.integers(1, 1 << 22, poison.sum()).astype(np.uint32)
    sign = rng.integers(0, 2, poison.sum()).astype(np.uint32) << 31
    bits[poison] = sign | np.uint32(0x7F800000) | payload
    flat = bits.reshape(-1)
    for pattern in (0x7F800000, 0xFF800000, 0x80000000, 0x00000001):
        flat[rng.integers(0, flat.size)] = pattern
    table = jnp.asarray(bits.view(np.float32))
    scl = rng.integers(0, sublanes * 128, lookups).astype(np.int32)
    nan_at = np.flatnonzero(np.isnan(table.reshape(-1)))
    if nan_at.size:
        scl[::5] = rng.choice(nan_at, len(scl[::5]))
    scl = jnp.asarray(scl)
    expected = np.asarray(table[jnp.right_shift(scl, 7), scl & 127]).view(np.uint32)

    lookup = jax.jit(_lookup_row_scales, static_argnames=("shift", "onehot"))

    with jax.enable_x64(enable_x64):
        lowered = lookup.lower(table, scl, 7, onehot=onehot)
        assert ("stablehlo.dot_general" in lowered.as_text()) == onehot
        got = np.asarray(lowered.compile()(table, scl))
    np.testing.assert_array_equal(got.view(np.uint32), expected)


def _reference_first_argmax(logits):
    """Serving semantics: BF16 round, a row with NaN takes its first NaN,
    otherwise the first column holding the maximum (-0 == +0)."""
    import numpy as np

    rounded = np.asarray(
        torch.from_numpy(logits).to(torch.bfloat16).float().numpy(), np.float64
    )
    out = []
    for row in rounded:
        nan = np.flatnonzero(np.isnan(row))
        out.append(nan[0] if nan.size else int(np.flatnonzero(row == row.max())[0]))
    return np.asarray(out)


@pytest.mark.cpu_test
def test_packed_key_argmax_matches_first_index_semantics():
    """The packed int32 key must reproduce the two-pass argmax it replaced:
    a wrong order of keys, a lost tie rule or a missed NaN would pick a
    different token for real greedy decoding."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(0)
    rows, tile, tiles = 16, 128, 3
    logits = rng.standard_normal((rows, tile * tiles)).astype(np.float32)
    logits[0, [5, 300]] = 7.0  # tie across tiles: first tile wins
    logits[1, [40, 41]] = 7.0  # tie inside one tile: first column wins
    logits[2, 200] = np.nan  # NaN beats every finite value
    logits[2, 3] = np.inf
    logits[3, [10, 250]] = np.nan  # earliest NaN across tiles
    logits[4] = -np.inf
    logits[4, 77] = -0.0
    logits[4, 20] = 0.0  # -0 and +0 tie, first column wins
    logits[5] = -1.0
    logits[5, 9] = -3.0e-8  # all negative: largest (closest to 0) wins
    logits[6, 30] = 5.0
    logits[6, 31] = 5.0001  # equal after BF16 rounding: first column wins
    logits[7] = -np.inf  # all -inf: column 0

    blocks = jnp.asarray(logits).reshape(rows, tiles, tile).transpose(1, 0, 2)
    packed = jax.vmap(kernel.packed_tile_argmax)(blocks)
    # Fold tiles in vocab order as the kernel does, from garbage initial state.
    key = jnp.full((rows,), 0x7FFFFFFF, jnp.int32)
    ids = jnp.full((rows,), -1, jnp.int32)
    for n in range(tiles):
        key, ids = kernel.merge_tile(key, ids, packed[n], n * tile, n == 0)
    got = np.asarray(ids)
    np.testing.assert_array_equal(got, _reference_first_argmax(logits))
    assert got[:8].tolist() == [5, 40, 200, 10, 20, 9, 30, 0]


@pytest.mark.parametrize("pipelined", [False, True])
@pytest.mark.parametrize("slot", [0, 4])
def test_slot_selection_preserves_ties_nans_and_last_vocabulary_entry(slot, pipelined):
    if jax.default_backend() != "tpu":
        pytest.skip("Compiled MXU kernel requires TPU")
    x = jnp.zeros((48, 256), jnp.bfloat16)
    weight = jnp.zeros((129280, 256), jnp.bfloat16)
    base = jnp.full((240, 129280), -10, jnp.bfloat16)
    values = jnp.full((48, 129280), -10, jnp.bfloat16)
    expected = np.arange(48, dtype=np.int64) * 2600
    expected[-1] = 129279
    values = values.at[jnp.arange(48), jnp.asarray(expected)].set(10)
    values = values.at[0, :].set(-jnp.inf)
    expected[0] = 0
    values = values.at[1, 16383].set(jnp.nan).at[1, 16384].set(jnp.nan)
    expected[1] = 16383
    values = values.at[2, :].set(-1).at[2, 16383].set(-0.0).at[2, 16384].set(0.0)
    expected[2] = 16383
    base = base.at[slot * 48 : (slot + 1) * 48].set(values)
    actual = jax.jit(functools.partial(sample_slot, slot=slot, pipelined=pipelined))(
        x, weight, base
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("pipelined", [False, True])
def test_projection_changes_the_selected_token(pipelined):
    if jax.default_backend() != "tpu":
        pytest.skip("Compiled MXU kernel requires TPU")
    x = jnp.ones((48, 256), jnp.bfloat16)
    weight = jnp.zeros((129280, 256), jnp.bfloat16).at[129279].set(1)
    base = jnp.zeros((240, 129280), jnp.bfloat16).at[:, 1].set(100)
    actual = jax.jit(functools.partial(sample_slot, slot=4, pipelined=pipelined))(
        x, weight, base
    )
    np.testing.assert_array_equal(actual, np.full(48, 129279))


def test_pipelined_kernel_matches_original_over_five_slots():
    if jax.default_backend() != "tpu":
        pytest.skip("Compiled MXU kernel requires TPU")
    k = jax.random.split(jax.random.key(3), 4)
    base = (jax.random.normal(k[0], (240, 129280)) * 2).astype(jnp.bfloat16)
    emb = (jax.random.normal(k[1], (129280, 256)) * 0.05).astype(jnp.bfloat16)
    weight = (jax.random.normal(k[2], (129280, 256)) * 0.05).astype(jnp.bfloat16)
    anchors = jax.random.randint(k[3], (48,), 0, 129280, jnp.int32)

    @functools.partial(jax.jit, static_argnames=("pipelined",))
    def tokens(*, pipelined):
        previous, out = anchors, []
        for slot in range(5):
            previous = sample_slot(
                emb[previous], weight, base, slot=slot, pipelined=pipelined
            )
            out.append(previous)
        return jnp.stack(out, axis=1)

    np.testing.assert_array_equal(
        np.asarray(tokens(pipelined=True)), np.asarray(tokens(pipelined=False))
    )


def _pieces(x):
    return jnp.stack(
        [
            x[..., start + parity : start + 256 : 2]
            for start in (0, 256)
            for parity in (0, 1)
        ]
    )


def test_row_valid_zeroes_rows_and_keeps_the_rest_bitwise():
    if jax.default_backend() != "tpu":
        pytest.skip("Compiled MXU kernel requires TPU")
    rng = np.random.default_rng(5)
    q = jnp.asarray(rng.normal(size=(4, 320, 512)), jnp.bfloat16)
    kv = jnp.asarray(rng.normal(size=(4, 192, 512)), jnp.bfloat16)
    valid = jnp.arange(192)[None, :] < jnp.asarray([192, 100, 1, 50])[:, None]
    kv = jnp.where(valid[..., None], kv, 0)
    sink = jnp.linspace(-1, 1, 320, dtype=jnp.float32)[:, None]
    mask = jnp.pad(valid.astype(jnp.int32), ((0, 0), (0, 64)))[:, None, :]
    tokens = jnp.asarray(rng.integers(0, 2, size=(4 * 5,)), jnp.int32)
    run = jax.jit(shared_record_attention)
    plain = run(mask, _pieces(q), _pieces(kv), sink)
    fused = run(mask, _pieces(q), _pieces(kv), sink, tokens)
    rows = jnp.repeat(tokens.reshape(4, 5), 64, axis=1)[:, :, None]
    expected = jnp.where(rows != 0, plain, jnp.zeros((), plain.dtype))
    np.testing.assert_array_equal(
        np.asarray(fused).view(np.uint16), np.asarray(expected).view(np.uint16)
    )
