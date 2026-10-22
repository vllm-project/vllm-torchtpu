# SPDX-License-Identifier: Apache-2.0
"""QSA decode routed to the fused sparse GQA decode kernel.

``VLLM_TPU_QSA_FUSED_SPARSE_DECODE`` swaps batched RPA for the fused kernel in
the decode route and the decode prefix of mixed steps. Neither batched RPA nor
the fused kernel (it reuses RPA's buffers and manual DMA pipeline) runs in a
Pallas interpret mode, so on CPU both kernels are replaced by decode-only
references with identical math. That checks everything around the kernels:
routing, virtual page metadata, query/gate layouts, K/V scales, paired-page
head offsets, idle-row zeroing and request chunking. The kernel itself is
lowered to Mosaic for TPU7x, and compared with RPA on hardware by the
``tpu_test`` cases of ``test_sparse_gqa_decode.py``.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.mesh import AbstractDevice, AbstractMesh
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from test_qwen4_exp_paired_main_kv import reference_rpa

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16
from vllm_torchtpu.kernels.sparse_gqa_decode import decode

FLAG = "VLLM_TPU_QSA_FUSED_SPARSE_DECODE"
RATIO, WIDTH, Q_HEADS = 4, 256, 3
TPU7X = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def reference_fused_kernel(
    *,
    cu_q_lens,
    compressed_query_offsets,
    q_lens,
    pages,
    metadata,
    query,
    unused_new_kv,
    cache,
    gate,
    cfgs,
    policy,
):
    """``fused_kernel.decode_kernel`` semantics: gated, read-only decode."""
    del cu_q_lens, unused_new_kv, policy
    kv_heads, batch, per_kv, _, dim = query.shape
    kv_lens = compressed_query_offsets + q_lens
    pages_per_seq = len(pages) // batch
    assert pages_per_seq == cfgs.serve.pages_per_seq
    assert cache.shape[1] == cfgs.serve.page_size
    combined = cache.reshape(*cache.shape[:2], -1, dim)
    assert combined.shape[2] >= 2 * kv_heads
    # Every live request owns ceil(kv_len / bkv) schedule tasks.
    tiles = (kv_lens + cfgs.bkv_sz - 1) // cfgs.bkv_sz
    expected_steps = (jnp.sum(tiles) + cfgs.batch_size - 1) // cfgs.batch_size
    out = jnp.zeros(query.shape, jnp.float32)
    for seq in range(batch):
        ids = pages[seq * pages_per_seq : (seq + 1) * pages_per_seq]
        rows = combined[ids].reshape(pages_per_seq * cache.shape[1], -1, dim)
        key = rows[:, 0 : 2 * kv_heads : 2].astype(jnp.float32) * cfgs.serve.scale_k
        value = rows[:, 1 : 2 * kv_heads : 2].astype(jnp.float32) * cfgs.serve.scale_v
        visible = jnp.arange(len(rows)) < kv_lens[seq]
        q = query[:, seq, :, 0]
        score = jnp.einsum("hqd,thd->hqt", q, key) * cfgs.model.sm_scale
        score = jnp.where(visible[None, None], score, -jnp.inf)
        prob = jax.nn.softmax(score, axis=-1)
        prob = jnp.where(jnp.isnan(prob), 0, prob)
        result = jnp.einsum("hqt,thd->hqd", prob, value)
        g = gate[seq].astype(jnp.float32).reshape(kv_heads, per_kv, dim)
        gated = bf16(
            bf16(result).astype(jnp.float32)
            * bf16(jax.nn.sigmoid(g)).astype(jnp.float32)
        ).astype(jnp.float32)
        out = out.at[:, seq, :, 0].set(jnp.where(kv_lens[seq] > 0, gated, 0))
    out = jnp.where(metadata.actual_steps[0] == expected_steps, out, jnp.nan)
    return out, cache


@pytest.fixture
def kernels(monkeypatch):
    """CPU stand-ins for both decode kernels; records which one ran."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper

    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    calls = []

    def rpa(*args, **kwargs):
        calls.append(("rpa", args[0].shape, args[1].shape))
        return reference_rpa(*args, **kwargs)

    def fused(**kwargs):
        calls.append(("fused", kwargs["query"].shape, kwargs["cache"].shape))
        return reference_fused_kernel(**kwargs)

    monkeypatch.setattr(wrapper, "ragged_paged_attention", rpa)
    monkeypatch.setattr(decode, "decode_kernel", fused)
    return calls


def _cache(rng, pages, page, fp8=True):
    packing = 4 if fp8 else 2
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    # Paired FP8 rows: layer slot s holds K/V at head slots 2s and 2s + 1.
    return jnp.asarray(
        rng.normal(0, 0.5, (pages, page, 4 // packing, packing, WIDTH)), dtype
    )


def _rows(rng, lengths, columns, groups, pages):
    """Per-token rows: selected groups (top-k within budget), tables, Q, gate."""
    lengths = np.asarray(lengths, np.int32)
    blocks = np.full((len(lengths), groups), -1, np.int32)
    for row, length in enumerate(lengths):
        count = min(length // RATIO, groups)
        if count:
            blocks[row, :count] = rng.choice(length // RATIO, count, replace=False)
    table = rng.integers(0, pages, (len(lengths), columns)).astype(np.int32)
    q = jnp.asarray(rng.normal(0, 0.7, (len(lengths), Q_HEADS, WIDTH)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 2, q.shape), jnp.bfloat16)
    return q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table)


def _verify_step(rng, contexts, tokens_per_request, rows, groups, page, columns):
    """An MTP verify step: K+1 rows per request, idle requests, padding rows."""
    starts, lengths = [0], []
    for context in contexts:
        n = tokens_per_request if context else 0
        lengths += [context + i + 1 for i in range(n)]
        starts.append(starts[-1] + n)
    lengths += [0] * (rows - len(lengths))
    pages = 2 * columns
    q, gate, blocks, lengths, table = _rows(rng, lengths, columns, groups, pages)
    starts = jnp.asarray(starts, jnp.int32)
    return (q, gate, blocks, lengths, table, _cache(rng, pages, page)), starts


def _attend(args, starts, *, fused, **options):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(FLAG, "1" if fused else "0")
        return jax.jit(
            functools.partial(
                qsa_rpa.qsa_attention,
                starts=starts,
                kv_heads=1,
                ratio=RATIO,
                k_scale=0.375,
                v_scale=2.5,
                **options,
            )
        )(*args)


@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize(
    "contexts,rows,groups,page,columns",
    [
        # Short contexts: tail-only, partial tail and whole groups.
        ([0, 1, 6, 9, 40, 0], 24, 16, 128, 2),
        # Contexts beyond the top-k budget: selected groups plus ring tail.
        ([300, 129, 0, 255, 77], 20, 16, 128, 3),
    ],
)
def test_verify_step_matches_rpa(
    kernels, offset, contexts, rows, groups, page, columns
):
    rng = np.random.default_rng(sum(contexts) + offset)
    args, starts = _verify_step(rng, contexts, 4, rows, groups, page, columns)
    options = dict(max_query_len=4)
    if offset:
        options["kv_head_offset"] = offset
    expected, cache = _attend(args, starts, fused=False, **options)
    assert [c[0] for c in kernels] == ["rpa"]
    actual, fused_cache = _attend(args, starts, fused=True, **options)
    assert [c[0] for c in kernels] == ["rpa", "fused"]
    # The fused kernel sees the paired row as offset + 1 KV heads and zero
    # queries ahead of this layer's three heads.
    kv_heads, batch, per_kv = kernels[-1][1][:3]
    assert (kv_heads, per_kv) == (1 + offset, Q_HEADS)
    assert batch == rows
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[np.asarray(args[3]) == 0], 0)
    assert np.abs(expected[np.asarray(args[3]) > 0]).max() > 0
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(fused_cache).view(np.uint8), np.asarray(args[5]).view(np.uint8)
    )
    assert cache.shape == fused_cache.shape


def test_paired_layers_read_their_own_heads(kernels):
    rng = np.random.default_rng(5)
    args, starts = _verify_step(rng, [50, 0, 130], 2, 8, 16, 128, 2)
    first, _ = _attend(args, starts, fused=True, max_query_len=2)
    second, _ = _attend(args, starts, fused=True, max_query_len=2, kv_head_offset=1)
    live = np.asarray(args[3]) > 0
    assert not np.array_equal(np.asarray(first)[live], np.asarray(second)[live])
    # Swapping the paired slots swaps the layers.
    cache = args[5]
    swapped = jnp.concatenate((cache[:, :, :, 2:], cache[:, :, :, :2]), axis=3)
    moved, _ = _attend(
        (*args[:5], swapped), starts, fused=True, max_query_len=2, kv_head_offset=1
    )
    np.testing.assert_array_equal(np.asarray(moved), np.asarray(first))


def test_large_verify_batch_chunks_like_rpa(kernels):
    """More rows than one fused launch holds: the batch runs in chunks."""
    rng = np.random.default_rng(9)
    contexts = list(rng.integers(0, 2200, 40))
    args, starts = _verify_step(rng, contexts, 4, 160, 512, 128, 18)
    expected, _ = _attend(args, starts, fused=False, max_query_len=4)
    actual, _ = _attend(args, starts, fused=True, max_query_len=4)
    chunk = kernels[-1][1][1]
    assert chunk < 160
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("offset", [0, 1])
def test_mixed_step_decode_prefix_matches_rpa(kernels, offset):
    """Short-request prefix on the decode kernel, the rest on sparse prefill."""
    rng = np.random.default_rng(13 + offset)
    page, columns, groups = 512, 2, 16
    # Requests: three decode rows, a 2-token verify request, a 21-token chunk.
    contexts = [100, 0, 600, 31]
    spans = [1, 1, 2, 21]
    starts, lengths = [0], []
    for context, span in zip(contexts, spans):
        lengths += [context + i + 1 for i in range(span)]
        starts.append(starts[-1] + span)
    lengths += [0] * (32 - len(lengths))
    q, gate, blocks, lengths, table = _rows(rng, lengths, columns, groups, 6)
    args = (q, gate, blocks, lengths, table, _cache(rng, 6, page))
    starts = jnp.asarray(starts, jnp.int32)
    options = dict(max_query_len=2)
    if offset:
        options["kv_head_offset"] = offset
    expected, _ = _attend(args, starts, fused=False, **options)
    actual, _ = _attend(args, starts, fused=True, **options)
    assert [c[0] for c in kernels] == ["rpa", "fused"]
    # Decode capacity: one row per request times max_query_len.
    assert kernels[-1][1][1] == 8
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)


def test_flag_defaults_off(kernels, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    rng = np.random.default_rng(1)
    args, starts = _verify_step(rng, [20, 9], 1, 4, 16, 128, 1)
    jax.jit(
        functools.partial(qsa_rpa.qsa_attention, starts=starts, kv_heads=1, ratio=RATIO)
    )(*args)
    assert [c[0] for c in kernels] == ["rpa"]


@pytest.mark.parametrize("case", ["chip", "dtype"])
def test_ineligible_inputs_fall_back_to_rpa(kernels, monkeypatch, case):
    rng = np.random.default_rng(2)
    args, starts = _verify_step(rng, [20, 9], 1, 4, 16, 128, 1)
    if case == "chip":
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    else:
        args = (args[0].astype(jnp.float32), *args[1:])
    _attend(args, starts, fused=True)
    assert [c[0] for c in kernels] == ["rpa"]


def _lower(fn, *specs):
    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((), (), abstract_device=device)):
        return (
            jax.jit(fn, donate_argnums=(5,))
            .trace(*specs)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )


@pytest.mark.parametrize("fp8", [True, False])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize(
    "rows,requests,page,columns",
    [(8, 8, 128, 64), (1024, 256, 128, 513), (64, 16, 1664, 40)],
)
def test_decode_route_lowers_fused_kernel_for_tpu7x(
    monkeypatch, fp8, offset, rows, requests, page, columns
):
    """TP8 geometry: three Q heads and one KV head per device, d = 256,
    top-512 groups of four tokens; MTP verify rows (K = 3) at B = 256."""
    if offset and not fp8:
        pytest.skip("Paired main KV pages are FP8-only")
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    monkeypatch.setenv(FLAG, "1")
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    packing = 4 if fp8 else 2
    shape = jax.ShapeDtypeStruct
    specs = (
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, 512), jnp.int32),
        shape((rows,), jnp.int32),
        shape((rows, columns), jnp.int32),
        shape((4096, page, 1, packing, WIDTH), dtype),
        shape((requests + 1,), jnp.int32),
    )
    options = dict(
        kv_heads=1,
        ratio=RATIO,
        k_scale=0.375 if fp8 else 1.0,
        v_scale=2.5 if fp8 else 1.0,
        max_query_len=rows // requests,
        **({"kv_head_offset": offset} if offset else {}),
    )
    # A fresh callable per lowering: the flag is read at trace time.
    text = _lower(functools.partial(qsa_rpa.qsa_attention, **options), *specs)
    assert "fused_qsa_decode_rpa_pipeline" in text
    assert "qsa_decode_schedule" in text
    monkeypatch.setenv(FLAG, "0")
    text = _lower(functools.partial(qsa_rpa.qsa_attention, **options), *specs)
    assert "fused_qsa_decode" not in text


@pytest.mark.parametrize("main_slot", [0, 1])
def test_pooled_mixed_step_lowers_fused_prefix(monkeypatch, main_slot):
    """The pooled QSA layer (mixed step, paired main KV) lowers with the
    fused decode prefix ahead of the sparse prefill."""
    from test_qwen4_exp_pool_inplace import _lower_pooled_step

    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    monkeypatch.setenv(FLAG, "1")
    text = _lower_pooled_step(
        8, kv_heads=1, q_heads=Q_HEADS, main_slot=main_slot
    ).as_text()
    assert "fused_qsa_decode_rpa_pipeline" in text
