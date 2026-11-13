# SPDX-License-Identifier: Apache-2.0
"""Host-side checks of VLLM_TPU_QSA_UNION_LEAN_GLUE,
VLLM_TPU_QSA_UNION_CHAINED_CALLS and VLLM_TPU_QSA_INDEXER_META_SELECT; the
kernels themselves are compared on TPU in
tests/kernels/qwen4_exp/dcp2/test_dcp2_attn_glue_flags_tpu.py.

- The union decode attention's calls, with the batched RPA replaced by a fake
  that writes rows the way the kernel does (rows of the call's sequences; the
  rest keep their buffer's values): every flag setting gives the flag-off
  output and LSE; with VLLM_TPU_QSA_UNION_LEAN_GLUE a step one call covers is
  one call over its own rows, with VLLM_TPU_QSA_UNION_CHAINED_CALLS each of
  several calls runs over the step's rows on the previous call's buffers, and
  padded request slots schedule nothing.
- The selection mask built from word rows equals the one built from the
  padded layout, and the kernel metadata of a call without word rows is what
  it was without the field.
- The indexer schedule selected instead of computed in a conditional: the same
  integers for decode, mixed, chunked and empty passes.
"""

import dataclasses
import types

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import metadata as idx_meta
from vllm_torchtpu.kernels.experimental.batched_rpa import configs, kernel, wrapper
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa, qsa_union_meta

UNION_ENV = {
    "VLLM_TPU_QSA_DECODE_RPA_UNION": "640",
    "VLLM_TPU_QSA_DECODE_RPA_LANES": "4",
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": "1",
}
PAGES = 2080  # union page slots a request (union_page_slots(512, 4, 640))
HEAD_DIM, HEADS = 192, 12


def _fake_rpa(calls):
    """The batched RPA's row contract: each row of the call's sequences gets a
    value from its query, its sequence's kv length and selection words; every
    other row keeps what its buffer held (the query, or under ``carry`` an
    earlier call's output). Buffers are in the kernel's layout, as the wrapper
    returns them with return_carry."""

    def rpa(
        query,
        keys,
        values,
        cache,
        kv_lens,
        pages,
        cu,
        dist,
        *,
        selection_bits,
        carry=None,
        return_carry=False,
        **kw,
    ):
        rows, heads, dim = query.shape
        calls.append(
            dict(
                rows=rows,
                keys=keys.shape[0],
                cu=np.asarray(cu),
                kv_lens=np.asarray(kv_lens),
                word_rows=kw.get("selection_word_rows", False),
                carried=carry is not None,
            )
        )
        if carry is None:
            buf = query.reshape(1, rows, heads, 1, dim)
            lse_buf = jnp.full((1, rows, heads, 1, 128), -jnp.inf, jnp.float32)
        else:
            buf, lse_buf = carry
        view = buf.reshape(rows, heads, dim)
        row = jnp.arange(rows)
        seq = jnp.clip(jnp.searchsorted(cu[1:], row, side="right"), 0, len(kv_lens) - 1)
        inside = (row >= cu[0]) & (row < cu[-1])
        words = jnp.sum(selection_bits % 7, axis=1).astype(jnp.float32)
        bias = (kv_lens[seq].astype(jnp.float32) + words[seq])[:, None, None]
        view = jnp.where(inside[:, None, None], view * 2 + bias, view)
        lse_rows = (view[..., 0] + bias[..., 0])[None, :, :, None, None]
        lse_buf = jnp.where(inside[None, :, None, None, None], lse_rows, lse_buf)
        buf = view.reshape(1, rows, heads, 1, dim)
        if return_carry:
            return buf, cache, lse_buf
        out, lse = wrapper.finish_carry((buf, lse_buf), query.shape, 1, query.dtype)
        return out, cache, lse

    return rpa


def _step(rng, query_lens, slots, rows, first=0):
    """``q`` and union metadata for a step of ``rows`` rows whose requests
    hold ``query_lens`` rows from row ``first``, in ``slots`` request slots."""
    lens = np.zeros(slots, np.int32)
    lens[: len(query_lens)] = query_lens
    starts = (first + np.concatenate([[0], np.cumsum(lens)])).astype(np.int32)
    meta = np.zeros((slots, 16), np.int32)
    meta[: len(query_lens), 0] = 1
    meta[: len(query_lens), 1] = rng.integers(1, 9000, len(query_lens))
    meta[: len(query_lens), 2:6] = rng.integers(0, 3, (len(query_lens), 4))
    pages = rng.integers(0, 4096, (slots, PAGES)).astype(np.int32)
    sel = rng.integers(0, 2**31 - 1, (slots, PAGES * 4)).astype(np.int32)
    q = jnp.asarray(rng.normal(size=(rows, HEADS, HEAD_DIM)), jnp.bfloat16)
    starts, pages, sel, meta = map(jnp.asarray, (starts, pages, sel, meta))
    live = qsa_union_meta.union_row_live(meta, starts, rows)
    return q, (starts, pages, sel, meta, live)


CASES = {
    # rows, live requests' rows, request slots the call sees, first row
    "16 rows, 4 of 32 slots": (16, [4] * 4, 32, 0),
    "64 rows, an empty request, 4 rows past the last": (
        64,
        [4] * 9 + [0] + [4] * 6,
        32,
        0,
    ),
    "32 rows from row 4": (32, [4] * 5 + [3, 2], 32, 4),
    "128 rows": (128, [4] * 32, 32, 0),
    "192 rows, two calls": (192, [4] * 48, 64, 0),
    "256 rows, second call partly empty": (256, [4] * 40 + [2, 3] + [4] * 10, 64, 0),
    "384 rows, three calls": (384, [4] * 90 + [3] * 2, 96, 0),
}
FLAGS = {  # VLLM_TPU_QSA_UNION_LEAN_GLUE, VLLM_TPU_QSA_UNION_CHAINED_CALLS
    "off": ("0", "0"),
    "lean": ("1", "0"),
    "chained": ("0", "1"),
    "both": ("1", "1"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_union_lean_glue_same_rows_one_buffer_over_the_step(monkeypatch, case):
    _tpu7x_info(monkeypatch)
    rows, query_lens, slots, first = CASES[case]
    for key, value in UNION_ENV.items():
        monkeypatch.setenv(key, value)
    rng = np.random.default_rng(len(query_lens) + rows)
    q, meta = _step(rng, query_lens, slots, rows, first)
    cache = jnp.zeros((8, 512, 1, 4, 256), jnp.float8_e4m3fn)
    dcp = (jnp.zeros((1,), jnp.int32), 4)
    results, calls = {}, {}
    for name, (lean, chained) in FLAGS.items():
        monkeypatch.setenv("VLLM_TPU_QSA_UNION_LEAN_GLUE", lean)
        monkeypatch.setenv("VLLM_TPU_QSA_UNION_CHAINED_CALLS", chained)
        calls[name] = []
        monkeypatch.setattr(wrapper, "ragged_paged_attention", _fake_rpa(calls[name]))
        out, lse, _ = qsa_rpa.union_decode_attention(
            q,
            None,
            None,
            None,
            cache,
            None,
            ratio=4,
            k_scale=0.375,
            v_scale=2.5,
            kv_head_offset=0,
            max_query_len=4,
            dcp=dcp,
            read_only=True,
            metadata=meta,
        )
        results[name] = (np.asarray(out), np.asarray(lse))
    for name in FLAGS:
        for i, what in ((0, "out"), (1, "lse")):
            assert np.array_equal(
                results["off"][i].view(np.uint32), results[name][i].view(np.uint32)
            ), f"{case}: {what} differs with {name}"
    assert results["off"][0].shape == (rows, HEADS, HEAD_DIM)
    assert np.any(np.asarray(meta[4])), "no live row"
    chunks = slots // 32
    assert all(c["rows"] == 128 and not c["word_rows"] for c in calls["off"])
    assert all(c["word_rows"] for f in ("lean", "both") for c in calls[f])
    starts = np.asarray(meta[0])
    live = len(query_lens)
    if chunks == 1:
        # One call: the chained flag alone leaves the sliced call.
        assert [c["rows"] for c in calls["chained"]] == [128]
        for name in ("lean", "both"):
            (call,) = calls[name]
            assert call["rows"] == call["keys"] == rows and not call["carried"]
            assert np.array_equal(call["cu"], np.clip(starts, 0, rows))
            # Request slots past the live requests schedule nothing.
            assert np.all(np.diff(call["cu"])[live:] == 0)
            assert np.all(call["kv_lens"][live:] == 0)
        return
    assert [c["rows"] for c in calls["lean"]] == [128] * chunks
    for name in ("chained", "both"):
        assert [c["rows"] for c in calls[name]] == [rows] * chunks
        assert [c["carried"] for c in calls[name]] == [False] + [True] * (chunks - 1)
        cu = [c["cu"] for c in calls[name]]
        for k in range(chunks):
            edges = starts[32 * k : 32 * k + 33]
            want = np.clip(edges, edges[0], min(edges[0] + 128, rows))
            assert np.array_equal(cu[k], want), (name, k)
        # Every live row is in exactly one call; padded slots schedule nothing.
        lens = np.concatenate([np.diff(x) for x in cu])
        assert lens.sum() == sum(query_lens) and np.all(lens[live:] == 0)


def _tpu7x_info(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def test_selection_mask_from_word_rows_is_the_same(monkeypatch):
    _tpu7x_info(monkeypatch)
    rng = np.random.default_rng(3)
    batch, bkv, tq = 4, 640, 16
    words = rng.integers(0, 2**31 - 1, (batch, bkv // 128, 128)).astype(np.int32)
    step_meta = types.SimpleNamespace(query_base=jnp.asarray([0, 4, 8, 12], jnp.int32))
    masks = {}
    for word_rows in (False, True):
        cfgs = types.SimpleNamespace(
            aligned_num_q_heads_per_kv_head=12,
            serve=types.SimpleNamespace(selection_word_rows=word_rows),
        )
        sel = jnp.asarray(words[:, :, None, :] if word_rows else words)
        masks[word_rows] = kernel.generate_selection_mask(
            (batch, 1, tq, bkv),
            bq_start=0,
            step_meta=step_meta,
            sel_vref=sel,
            cfgs=cfgs,
        )
    for a, b in zip(masks[False], masks[True]):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_word_rows_change_only_the_flagged_kernels(monkeypatch):
    _tpu7x_info(monkeypatch)
    serve = configs.ServingConfigs(
        num_seqs=32,
        page_size=512,
        total_q_tokens=16,
        num_page_indices=32 * PAGES,
        dtype_q=jnp.float32,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.float32,
        decode_query_size=4,
        return_lse=True,
        skip_kv_update=True,
        kv_slot=0,
        compact_kv_schedule=True,
        selection_mask=True,
        read_only_cache=True,
    )
    model = configs.ModelConfigs(
        num_q_heads=12,
        num_kv_heads=1,
        head_dim=256,
        sliding_window=None,
        sm_scale=1.0,
        soft_cap=None,
        mask_value=-1e30,
    )
    block = configs.BlockSizes(bq_sz=4, bq_c_sz=4, bkv_sz=640, batch_size=4)

    def cfgs(serve):
        return configs.RpaConfigs(
            block=block,
            model=model,
            serve=serve,
            vmem_limit_bytes=64 << 20,
            mode=configs.RpaCase.DECODE,
        )

    off, on = cfgs(serve), cfgs(dataclasses.replace(serve, selection_word_rows=True))
    assert off.selection_vmem_shape == (4, 5, 128)
    assert on.selection_vmem_shape == (4, 5, 1, 128)
    meta_off, meta_on = kernel.get_kernel_metadata(off), kernel.get_kernel_metadata(on)
    assert not any("selection_word_rows" in key for key in meta_off)
    assert meta_on.pop("serve.selection_word_rows") == 1
    assert meta_on == meta_off
    assert kernel.get_kernel_name(on) == kernel.get_kernel_name(off)


INDEXER_PASSES = {
    # seq lens, cu_q_lens step, start, end, kwargs
    "decode batched": (
        [9000, 37, 1500, 0, 12000, 800, 0, 0],
        1,
        0,
        6,
        dict(bq_sz=1, bkv_sz=1024, static_q_len=1, seq_batch_size=4),
    ),
    "decode tail": (
        [9000, 37, 1500, 0, 12000, 800, 0, 0],
        1,
        4,
        6,
        dict(bq_sz=1, bkv_sz=1024, static_q_len=1, seq_batch_size=1),
    ),
    "mixed": (
        [9000, 37, 1500, 300, 12000, 800, 0, 0],
        4,
        0,
        6,
        dict(bq_sz=4, bkv_sz=1024, static_q_len=None, seq_batch_size=1),
    ),
    "chunked": (
        [9000, 37, 1500, 300, 12000, 800, 0, 0],
        4,
        0,
        6,
        dict(
            bq_sz=4,
            bkv_sz=1024,
            static_q_len=None,
            seq_batch_size=1,
            chunk_token_start=(0, 8, 16),
            chunk_tokens=8,
        ),
    ),
    "empty": (
        [9000, 37, 1500, 300, 12000, 800, 0, 0],
        4,
        6,
        6,
        dict(bq_sz=4, bkv_sz=1024, static_q_len=None, seq_batch_size=1),
    ),
}


@pytest.mark.parametrize("name", sorted(INDEXER_PASSES))
def test_indexer_schedule_selected_is_the_same(monkeypatch, name):
    lens, step, start, end, kw = INDEXER_PASSES[name]
    seq_lens = jnp.asarray(lens, jnp.int32)
    q_lens = np.where(np.arange(len(lens)) < 6, step, 0)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
    got = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_META_SELECT", flag)
        got[flag] = idx_meta.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu,
            start_seq_idx=jnp.asarray(start),
            end_seq_idx=jnp.asarray(end),
            pages_per_seq=64,
            page_size=256,
            max_num_tokens=32,
            compression_ratio=4,
            **kw,
        )
    for field in ("num_steps", "batch_tile_idx", "bq_idx", "bkv_idx"):
        a, b = (
            np.asarray(getattr(got["0"], field)),
            np.asarray(getattr(got["1"], field)),
        )
        assert a.dtype == b.dtype and np.array_equal(a, b), f"{name}: {field} differs"
    if start >= end:
        assert not np.any(np.asarray(got["1"].num_steps))
        assert not np.any(np.asarray(got["1"].batch_tile_idx))
    else:
        assert int(np.sum(np.asarray(got["1"].num_steps))) > 0
