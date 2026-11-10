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
"""In-place MSA cache row writes (interpret mode) against the XLA scatter."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from tests.models.minimax_m3.packing import CACHE_DTYPES, PACKING, pack
from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_write

PAGE = 128
NUM_PAGES = 12
DROPPED = 2**30


def _batch(q_lens, ctx_lens, rng, pad_tokens):
    """Pages and positions of every token of a ragged batch, as msa_forward
    computes them; padding tokens are dropped."""
    perm = rng.permutation(np.arange(1, NUM_PAGES))
    pages, pos, used = [], [], 0
    for q, c in zip(q_lens, ctx_lens):
        n = -(-(q + c) // PAGE)
        table = perm[used : used + n]
        used += n
        p = np.arange(c, c + q)
        pages.append(table[p // PAGE])
        pos.append(p)
    pages = np.concatenate(pages + [np.full(pad_tokens, DROPPED)]).astype(np.int32)
    pos = np.concatenate(pos + [np.zeros(pad_tokens, np.int64)]).astype(np.int32)
    return jnp.asarray(pages), jnp.asarray(pos)


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,pad_tokens",
    [
        ([1, 1, 1], [5, 130, 255], 2),  # decode: mid-page, page start, last slot
        ([300], [70], 0),  # prefill from mid-page across three pages
        ([1, 200, 3], [127, 0, 125], 4),  # mixed, a run ending at a page edge
        ([], [], 3),  # nothing to write
    ],
)
def test_matches_xla_scatter(q_lens, ctx_lens, pad_tokens, width, dtype):
    rng = np.random.default_rng(sum(q_lens) + width)
    num_tokens = sum(q_lens) + pad_tokens
    cache = pack(
        rng.standard_normal((NUM_PAGES, PAGE, width)).astype(np.float32), dtype
    )
    rows = jnp.asarray(rng.standard_normal((num_tokens, width)) * 300).astype(
        jnp.bfloat16
    )
    pages, pos = _batch(q_lens, ctx_lens, rng, pad_tokens)

    expected = msa.write_rows_xla(cache, rows, pages, pos)
    got = msa.write_rows(
        cache, rows, pages, pos, max_num_seqs=max(len(q_lens), 1), interpret=True
    )
    np.testing.assert_array_equal(
        np.asarray(got.astype(jnp.float32)), np.asarray(expected.astype(jnp.float32))
    )


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_lowers_for_tpu(dtype):
    """Pallas -> Mosaic lowering and verification run without a TPU."""
    tokens, width = 256, 256
    args = (
        jax.ShapeDtypeStruct(
            (4096, PAGE // PACKING[dtype], PACKING[dtype], width), dtype
        ),
        jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = (
            jax.jit(functools.partial(msa.write_rows, max_num_seqs=16))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    text = lowered.as_text()
    assert "tpu_custom_call" in text
    assert "minimax_m3_msa_write_rows" in text


def test_fp4_rows_land_in_their_nibbles():
    """bits=4 on a uint8 4-bit cache: runs across and within pages, the rest
    of every page (other slots' nibbles) unchanged."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

    rng = np.random.default_rng(3)
    num_pages, page, width = 6, 128, 256
    before = jnp.asarray(rng.integers(0, 16, (num_pages, page, width)), jnp.uint8)
    cache = msa_fp4.pack_rows(before)
    # Request A: slots 120..135 over pages 2 -> 4; request B: 5 tokens in page 1; one dropped token.
    pages = np.array([2] * 8 + [4] * 8 + [1] * 5 + [99], np.int32)
    slots = np.array(list(range(120, 128)) + list(range(8)) + list(range(40, 45)) + [0], np.int32)
    rows = jnp.asarray(rng.integers(0, 16, (len(pages), width)), jnp.uint8)
    out = msa_write.write_rows(
        cache, rows, jnp.asarray(pages), jnp.asarray(slots),
        max_runs=msa_write.max_write_runs(len(pages), 3, page), bits=4, interpret=True,
    )
    expected = np.asarray(before).copy()
    for t, (p, s) in enumerate(zip(pages, slots)):
        if p < num_pages:
            expected[p, s] = np.asarray(rows[t])
    np.testing.assert_array_equal(msa_fp4.unpack_rows(out), expected)


# --- Traced grid (one step per run) against the static grid of max_runs steps.
# Real cache shapes: 4-bit DCP4 (two KV heads per rank: K/V rows of 512
# nibbles, index rows with folded scales, a scale page) and fp8 DCP2.

_GRID_PAGES = 48


def _runs_batch(rng, run_lens, pad_tokens):
    """Pages and slots of runs on distinct pages (a run may start mid-page),
    then dropped padding tokens."""
    perm = rng.permutation(_GRID_PAGES)
    pages, slots = [], []
    for i, n in enumerate(run_lens):
        lo = int(rng.integers(0, PAGE - n + 1))
        pages.append(np.full(n, perm[i]))
        slots.append(np.arange(lo, lo + n))
    pages.append(np.full(pad_tokens, DROPPED))
    slots.append(np.zeros(pad_tokens, np.int64))
    return (
        jnp.asarray(np.concatenate(pages).astype(np.int32)),
        jnp.asarray(np.concatenate(slots).astype(np.int32)),
    )


def _grid_case(case, num_tokens, rng):
    """(run lengths, padding tokens, max_runs) of a batch of `num_tokens`."""
    if case == "none":  # this rank owns no token of the step
        return [], num_tokens, msa_write.max_write_runs(num_tokens, 8, PAGE)
    if num_tokens <= 64:  # decode-like: 1-token runs
        lens = [1] * (num_tokens // 2) + [3, 5]
    else:  # mixed: decode runs, then prefill runs of up to a page
        lens = [1] * 9 + [8, 100, 128, 128, 77]
        lens += [128] * ((num_tokens - sum(lens)) // 128 - 1)
    pad = num_tokens - sum(lens)
    assert pad >= 0
    if case == "full":  # as many runs as the grid bound
        return lens, pad, len(lens)
    return lens, pad, len(lens) + 7  # some idle steps on the static grid


def _write_all(config, inputs, pages, slots, max_runs, **kw):
    """Every cache write of one layer's step, as uint8 / uint32 views."""
    mw = msa_write
    if config == "int4":
        kv, idx, sc, kv_rows, idx_rows, idx_scales, sc_values = inputs
        kv = mw.write_rows(kv, kv_rows, pages, slots, max_runs=max_runs, bits=4, interpret=True, **kw)
        idx = mw.write_rows(
            idx, idx_rows, pages, slots, max_runs=max_runs, bits=4, scales=idx_scales,
            interpret=True, **kw,
        )
        sc = mw.write_slot_scales(sc, sc_values, pages, slots, max_runs=max_runs, interpret=True, **kw)
        return [np.asarray(kv), np.asarray(idx), np.asarray(jax.lax.bitcast_convert_type(sc, jnp.uint32))]
    kv, idx, kv_rows, idx_rows = inputs
    kv = mw.write_rows(kv, kv_rows, pages, slots, max_runs=max_runs, interpret=True, **kw)
    idx = mw.write_rows(idx, idx_rows, pages, slots, max_runs=max_runs, interpret=True, **kw)
    return [np.asarray(jax.lax.bitcast_convert_type(c, jnp.uint8)) for c in (kv, idx)]


def _grid_inputs(config, num_tokens, rng):
    p = _GRID_PAGES
    if config == "int4":
        u8 = lambda *s: jnp.asarray(rng.integers(0, 256, s), jnp.uint8)  # noqa: E731
        nib = lambda *s: jnp.asarray(rng.integers(0, 16, s), jnp.uint8)  # noqa: E731
        f32 = lambda *s: jnp.asarray(rng.standard_normal(s), jnp.float32)  # noqa: E731
        return (
            u8(p, 16, 4, 512), u8(p, 17, 4, 128), f32(p, 8, 128),
            nib(num_tokens, 512), nib(num_tokens, 128), f32(num_tokens), f32(num_tokens, 4),
        )
    f8 = lambda *s: jnp.asarray(rng.standard_normal(s) * 50, jnp.float8_e4m3fn)  # noqa: E731
    return f8(p, 32, 4, 256), f8(p, 32, 4, 128), f8(num_tokens, 256), f8(num_tokens, 128)


def _check_grid(config, num_tokens, case, seed=0):
    rng = np.random.default_rng(seed + num_tokens)
    lens, pad, max_runs = _grid_case(case, num_tokens, rng)
    pages, slots = _runs_batch(rng, lens, pad)
    inputs = _grid_inputs(config, num_tokens, rng)
    static = _write_all(config, inputs, pages, slots, max_runs, traced_grid=False)
    traced = _write_all(config, inputs, pages, slots, max_runs, traced_grid=True)
    for s, t in zip(static, traced):
        np.testing.assert_array_equal(t, s)
    if case == "none":  # nothing written: every cache bit-identical to its input
        caches = inputs[:3] if config == "int4" else inputs[:2]
        for c, t in zip(caches, traced):
            np.testing.assert_array_equal(t.view(np.uint8).ravel(), np.asarray(
                jax.lax.bitcast_convert_type(c, jnp.uint8)).ravel())
    return static, traced


@pytest.mark.parametrize("config", ["int4", "fp8"])
@pytest.mark.parametrize("num_tokens", [40, 640])  # by-token and by-page paths
@pytest.mark.parametrize("case", ["some", "none", "full"])
def test_traced_grid_matches_static_grid(config, num_tokens, case):
    _check_grid(config, num_tokens, case)


@pytest.mark.parametrize("num_tokens", [40, 640])
def test_traced_grid_check_catches_a_dropped_run(num_tokens, monkeypatch):
    """Fault injection: a traced grid one step short (the last run unwritten)
    must fail the comparison."""
    mw = msa_write
    real = mw._grid
    monkeypatch.setattr(
        mw, "_grid",
        lambda n, m, traced: real(n, m, traced) if not traced else (jnp.maximum(n - 1, 1),),
    )
    jax.clear_caches()
    try:
        with pytest.raises(AssertionError):
            _check_grid("int4", num_tokens, "full")
    finally:
        jax.clear_caches()


# --- One kernel for every cache of a step (`write_fp4_caches`, `write_rows_many`)
# against the per-cache writes and against a numpy oracle.


def _oracle(config, inputs, pages, slots):
    """Expected caches as uint8 / uint32 views: each valid token's row (and
    scales) at its page and slot, everything else as before."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

    pages, slots = np.asarray(pages), np.asarray(slots)
    live = [(t, p, s) for t, (p, s) in enumerate(zip(pages, slots)) if p < _GRID_PAGES]
    if config == "int4":
        kv, idx, sc, kv_rows, idx_rows, idx_scales, sc_values = inputs
        kv_n = np.asarray(msa_fp4.unpack_rows(kv)).copy()
        idx_n = np.asarray(msa_fp4.unpack_rows(idx)).copy()
        idx_s = np.asarray(jax.lax.bitcast_convert_type(msa_fp4.folded_scales(idx), jnp.uint32)).copy()
        sc_b = np.asarray(jax.lax.bitcast_convert_type(sc, jnp.uint32)).copy()
        idx_scale_bits = np.asarray(jax.lax.bitcast_convert_type(idx_scales, jnp.uint32))
        sc_bits = np.asarray(jax.lax.bitcast_convert_type(sc_values, jnp.uint32))
        n = sc_bits.shape[1]
        for t, p, s in live:
            kv_n[p, s] = np.asarray(kv_rows[t])
            idx_n[p, s] = np.asarray(idx_rows[t])
            idx_s[p, s] = idx_scale_bits[t]
            sc_b[p, :n, s] = sc_bits[t]
        return kv_n, idx_n, idx_s, sc_b
    out = []
    for cache, rows in ((inputs[0], inputs[2]), (inputs[1], inputs[3])):
        c = np.asarray(jax.lax.bitcast_convert_type(cache, jnp.uint8)).copy()
        r = np.asarray(jax.lax.bitcast_convert_type(rows, jnp.uint8))
        for t, p, s in live:
            c[p, s // 4, s % 4] = r[t]
        out.append(c)
    return out


def _as_oracle(config, caches):
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

    if config == "int4":
        kv, idx, sc = caches
        return (
            np.asarray(msa_fp4.unpack_rows(kv)),
            np.asarray(msa_fp4.unpack_rows(idx)),
            np.asarray(jax.lax.bitcast_convert_type(msa_fp4.folded_scales(idx), jnp.uint32)),
            np.asarray(jax.lax.bitcast_convert_type(sc, jnp.uint32)),
        )
    return [np.asarray(jax.lax.bitcast_convert_type(c, jnp.uint8)) for c in caches]


def _write_fused(config, inputs, pages, slots, max_runs, **kw):
    if config == "int4":
        out = msa_write.write_fp4_caches(*inputs, pages, slots, max_runs=max_runs, interpret=True, **kw)
        return [np.asarray(out[0]), np.asarray(out[1]),
                np.asarray(jax.lax.bitcast_convert_type(out[2], jnp.uint32))], out
    out = msa_write.write_rows_many(inputs[:2], inputs[2:], pages, slots, max_runs=max_runs, interpret=True, **kw)
    return [np.asarray(jax.lax.bitcast_convert_type(c, jnp.uint8)) for c in out], out


def _check_fused(config, num_tokens, case, seed=0):
    rng = np.random.default_rng(seed + num_tokens)
    lens, pad, max_runs = _grid_case(case, num_tokens, rng)
    pages, slots = _runs_batch(rng, lens, pad)
    inputs = _grid_inputs(config, num_tokens, rng)
    fused, caches = _write_fused(config, inputs, pages, slots, max_runs)
    for got, want in zip(_as_oracle(config, caches), _oracle(config, inputs, pages, slots)):
        np.testing.assert_array_equal(got, want)
    for traced in (True, False):
        separate = _write_all(config, inputs, pages, slots, max_runs, traced_grid=traced)
        for f, s in zip(fused, separate):
            np.testing.assert_array_equal(f, s)


@pytest.mark.parametrize("config", ["int4", "fp8"])
@pytest.mark.parametrize("num_tokens", [40, 640])
@pytest.mark.parametrize("case", ["some", "none", "full"])
def test_fused_write_matches_oracle_and_separate_writes(config, num_tokens, case):
    _check_fused(config, num_tokens, case)


@pytest.mark.parametrize("num_tokens", [40, 640])
def test_fused_oracle_catches_a_misplaced_scale(num_tokens, monkeypatch):
    """Fault injection: K/V scales one lane off in the scale page must fail."""
    for name in ("_slot_scales_by_token", "_slot_scales_by_page"):
        real = getattr(msa_write, name)
        monkeypatch.setattr(
            msa_write, name,
            functools.partial(lambda lo, hi, *a, real, **k: real(lo + 1, hi + 1, *a, **k), real=real),
        )
    jax.clear_caches()
    try:
        with pytest.raises(AssertionError):
            _check_fused("int4", num_tokens, "some")
    finally:
        jax.clear_caches()
