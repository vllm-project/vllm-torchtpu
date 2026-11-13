# SPDX-License-Identifier: Apache-2.0
"""In-row compressed keys (QWEN4_EXP_INROW_COMPRESSED_K).

With one FP8 KV head per device a main-KV token row holds K and V in bytes 0
and 1 of each 32-bit lane word; bytes 2 and 3 -- the row's second BF16 half --
hold one compressed key. Key ``g`` of a table entry of ``span`` pages of
tokens sits at row ``g % (span * B / ratio)`` of the entry's page. Writes of
either kind must keep the other's bytes, reads must find the keys there, and
selection and attention must match a layer whose keys have pages of their own.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa as q
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import InRowKeyReader

RATIO = 4
W = 256


def on_tpu() -> bool:
    # Called inside tests only: probing the backend at import claims the TPU
    # (tests/conftest.py refuses that).
    return jax.default_backend() == "tpu"


@pytest.fixture
def interpret(monkeypatch):
    """The Pallas writers and readers in interpret mode, off TPU."""
    if on_tpu():
        yield
        return
    original = pl.pallas_call
    monkeypatch.setattr(
        pl, "pallas_call", lambda *a, **k: original(*a, **{**k, "interpret": True})
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    yield


def _pool(pages, block):
    return jnp.zeros((pages, block, 1, 4, W), jnp.float8_e4m3fn)


def test_slot_codes_are_distinct_from_packed_and_merged():
    for span in range(1, 8):
        slot = q.inrow_compressed_slot(span)
        assert q.is_inrow(slot) and q.inrow_span(slot) == span
        assert slot > 8  # past every packed slot
    assert not q.is_inrow(0) and not q.is_inrow(7)
    assert not q.is_inrow(q.merged_compressed_slot(64, 2))
    with pytest.raises(AssertionError):
        q.inrow_keys_per_entry(q.inrow_compressed_slot(5), 512, RATIO)  # 640 keys > 512 rows


@pytest.mark.parametrize("block,span", [(512, 4), (512, 1), (1664, 1), (128, 2)])
def test_kernel_pages_and_expansion_agree(block, span):
    slot = q.inrow_compressed_slot(span)
    pool = _pool(9, block)
    rows = 128 // RATIO
    per_page = block // rows
    per_entry = span * block // RATIO // rows
    columns = jnp.asarray([3, 0, 8, 5], jnp.int32)
    pages = np.asarray(q.compressed_kernel_pages(columns, pool, slot, RATIO)).reshape(4, per_entry)
    np.testing.assert_array_equal(
        pages, np.asarray(columns)[:, None] * per_page + np.arange(per_entry)
    )
    expansion, stride, offset = q._compressed_expansion(per_entry, per_page, slot)
    np.testing.assert_array_equal(
        pages, np.asarray(columns)[:, None] * stride + offset + np.arange(expansion)
    )
    # An entry's keys stay inside its own page.
    assert per_entry <= per_page


def test_pairs_follow_the_table():
    block, span, tokens = 128, 2, 600
    slot = q.inrow_compressed_slot(span)
    pool = _pool(12, block)
    logical = jnp.arange(tokens, dtype=jnp.int32) + 37
    complete = (logical % RATIO) == RATIO - 1
    table = jnp.asarray([[4, 11, 2, 7]], jnp.int32)
    keys = jnp.asarray(np.random.default_rng(0).standard_normal((tokens, 128)), jnp.bfloat16)
    pair, write, mask, values = q.compressed_pairs(
        pool, keys, complete, jnp.zeros((tokens,), jnp.int32), logical,
        jnp.asarray([0, tokens], jnp.int32), jnp.asarray([37], jnp.int32), table, RATIO, slot,
    )
    entry = span * block
    want = np.asarray(table)[0, np.asarray(logical) // entry] * block + (np.asarray(logical) % entry) // RATIO
    np.testing.assert_array_equal(np.asarray(pair), want)
    np.testing.assert_array_equal(np.asarray(write), np.asarray(complete))
    assert not np.asarray(mask[:, 0]).any() and np.asarray(mask[:, 1]).all()
    np.testing.assert_array_equal(
        np.asarray(values[:, 1, :128], np.float32), np.asarray(keys, np.float32)
    )
    # One writer per row: the rows written are distinct.
    rows = np.asarray(pair)[np.asarray(write)]
    assert len(np.unique(rows)) == len(rows)


def _read_keys(pool, page, rows, variant):
    """Keys at rows ``[0, rows)`` of pool page ``page`` through the reader."""
    os.environ["VLLM_TPU_QSA_INROW_READ"] = variant
    try:
        cache = pool.reshape(-1, 32, 4, W)
        per_page = pool.shape[1] // 32
        out = []
        for k in range(-(-rows // 32)):
            def kernel(c_ref, o_ref):
                keys, _ = InRowKeyReader().read(c_ref, 32, W)
                o_ref[...] = keys

            # A [32, 4, W] window, as StreamIndex hands the reader one
            # sequence's KV block.
            window = jax.lax.dynamic_index_in_dim(cache, page * per_page + k, keepdims=False)
            out.append(pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((32, W), jnp.bfloat16),
            )(window))
        return jnp.concatenate(out)[:rows]
    finally:
        os.environ.pop("VLLM_TPU_QSA_INROW_READ", None)


@pytest.mark.parametrize("variant", ["u32", "bf16"])
@pytest.mark.parametrize("tokens", [64, 256])
def test_writes_keep_each_other_and_read_back(interpret, variant, tokens):
    """Main KV (paired slot-0 writer) then keys (half-masked writer) then more
    main KV: each keeps the other's bytes, and the reader returns the keys."""
    block, span = 128, 2
    rng = np.random.default_rng(tokens)
    slot = q.inrow_compressed_slot(span)
    table = jnp.asarray([[3, 5]], jnp.int32)
    logical = jnp.arange(tokens, dtype=jnp.int32)
    k = jnp.asarray(rng.standard_normal((tokens, 1, W)), jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((tokens, 1, W)), jnp.bfloat16)
    rows = q.pack_rows(k, v, fp8=True)
    entry = span * block
    page = table[0, logical // entry]
    row = logical % block  # a DCP-free stand-in: every token in its entry's page
    half = tokens // 2
    first = jnp.arange(tokens) < half
    pool = q.write_main_cache(_pool(8, block), rows, page, row, first, slot=0)
    keys = jnp.asarray(rng.standard_normal((tokens, W)), jnp.bfloat16)
    pool = q.write_compressed(
        pool, keys, (logical % RATIO) == RATIO - 1, jnp.zeros((tokens,), jnp.int32), logical,
        jnp.asarray([0, tokens], jnp.int32), jnp.asarray([0], jnp.int32), table, RATIO, slot,
    )
    pool = q.write_main_cache(pool, rows, page, row, ~first, slot=0)
    # Main KV: the plain writer's values in sublanes 0-1 of every token row.
    plain = q.write_main_cache(_pool(8, block), rows, page, row, jnp.ones((tokens,), bool))
    got, want = np.asarray(pool.astype(jnp.float32)), np.asarray(plain.astype(jnp.float32))
    np.testing.assert_array_equal(got[..., 0:2, :], want[..., 0:2, :])
    # Keys: one per RATIO tokens, rows from 0 of page 3.
    n_keys = tokens // RATIO
    read = _read_keys(pool, 3, n_keys, variant)
    np.testing.assert_array_equal(
        np.asarray(read, np.float32), np.asarray(keys[RATIO - 1 :: RATIO], np.float32)
    )


def _lower_inrow_step(tokens, compressed_slot, main_slot):
    from jax._src.mesh import AbstractDevice, AbstractMesh

    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import pooled_qsa_step

    requests = 3
    rng = np.random.default_rng(tokens)

    def rand(shape):
        return jnp.asarray(rng.normal(0, 0.3, shape), jnp.bfloat16)

    pool = jnp.zeros((12, 1664, 1, 4, 256), jnp.float8_e4m3fn)
    args = (
        rand((tokens, 3 * 256 * 2)), rand((tokens, 256)), rand((tokens, 256)),
        rand((tokens, 5 * 128)), rand((256,)), rand((256,)), rand((128,)), rand((128,)),
        jnp.zeros((tokens,), jnp.int32), rand((4096, 128)), pool,
        jnp.array([0, 1, 3, tokens - 1], jnp.int32), jnp.full((requests,), 100, jnp.int32),
        jnp.zeros((requests * 32,), jnp.int32), jnp.ones((requests, 2), jnp.int32),
        jnp.ones((requests, 2), jnp.int32),
    )
    options = dict(
        compress_ratio=4, token_topk=2048, norm_eps=1e-6, rotary_dim=64, k_scale=0.375,
        v_scale=2.5, compressed_slot=compressed_slot, main_slot=main_slot,
    )
    step = jax.jit(pooled_qsa_step, donate_argnums=(10,), static_argnames=tuple(options))
    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((), (), abstract_device=device)):
        return step.trace(*args, **options).lower(lowering_platforms=("tpu",))


@pytest.mark.parametrize("variant", ["u32", "bf16"])
@pytest.mark.parametrize("tokens", [8, 128])
def test_inrow_pooled_step_lowers(variant, tokens, monkeypatch):
    """The real pooled QSA step with in-row keys (one KV head, slot-0 main KV)
    lowers for TPU: writes, the in-row reader and the attention."""
    monkeypatch.setenv("VLLM_TPU_QSA_INROW_READ", variant)
    text = _lower_inrow_step(tokens, q.inrow_compressed_slot(1), 0).as_text()
    assert "tpu_custom_call" in text


@pytest.mark.parametrize("variant", ["u32", "bf16"])
@pytest.mark.parametrize("seq", [700, 5000])
def test_inrow_selection_matches_plain_pages(variant, seq, monkeypatch):
    """select_blocks over in-row keys picks what it picks over the same keys
    in pages of their own (one query per request, a decode step)."""
    if not on_tpu():
        pytest.skip("StreamIndex top-k runs on TPU only")
    monkeypatch.setenv("VLLM_TPU_QSA_INROW_READ", variant)
    block, span, req_n = 512, 4, 2
    rng = np.random.default_rng(seq)
    entry = span * block
    n_entries = -(-seq // entry)
    n_plain = -(-seq // block)
    keys = jnp.asarray(rng.standard_normal((req_n, seq, 128)) * 0.5, jnp.bfloat16)
    tokens = req_n * seq
    flat_keys = keys.reshape(tokens, 128)
    logical = jnp.tile(jnp.arange(seq, dtype=jnp.int32), req_n)
    req = jnp.repeat(jnp.arange(req_n, dtype=jnp.int32), seq)
    starts = jnp.asarray([0, seq, 2 * seq], jnp.int32)
    context = jnp.zeros((req_n,), jnp.int32)
    complete = (logical % RATIO) == RATIO - 1
    # Plain: compressed pages of their own (slot 0, one table entry per page).
    plain_table = jnp.asarray(1 + np.arange(req_n * n_plain).reshape(req_n, n_plain), jnp.int32)
    plain = q.write_compressed(_pool(1 + req_n * n_plain, block), flat_keys, complete, req,
                               logical, starts, context, plain_table, RATIO, 0)
    slot = q.inrow_compressed_slot(span)
    inrow_table = jnp.asarray(1 + np.arange(req_n * n_entries).reshape(req_n, n_entries), jnp.int32)
    inrow = _pool(1 + req_n * n_entries, block)
    # Random main KV first, so the reader must ignore bytes 0-1.
    noise = jnp.asarray(rng.integers(0, 256, inrow.shape, dtype=np.uint8))
    inrow = jax.lax.bitcast_convert_type(noise, jnp.float8_e4m3fn)
    inrow = q.write_compressed(inrow, flat_keys, complete, req, logical, starts, context,
                               inrow_table, RATIO, slot)
    query = jnp.asarray(rng.standard_normal((req_n, 1, 128)), jnp.bfloat16)
    lengths = jnp.full((req_n,), seq, jnp.int32)
    one = jnp.arange(req_n, dtype=jnp.int32)
    q_starts = jnp.arange(req_n + 1, dtype=jnp.int32)
    ctx = jnp.full((req_n,), seq - 1, jnp.int32)
    kw = dict(ratio=RATIO, block_topk=16)
    a = q.select_blocks(query, plain, lengths, plain_table, one, q_starts, ctx, slot=0, **kw)
    b = q.select_blocks(query, inrow, lengths, inrow_table, one, q_starts, ctx, slot=slot, **kw)
    np.testing.assert_array_equal(np.sort(np.asarray(a), -1), np.sort(np.asarray(b), -1))
