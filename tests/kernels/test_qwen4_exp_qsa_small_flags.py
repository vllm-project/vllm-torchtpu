# SPDX-License-Identifier: Apache-2.0
"""Exact QSA decode flags: VLLM_TPU_RPA_SCHEDULE_HBM, VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY,
VLLM_TPU_QSA_INDEXER_TAIL_SKIP and VLLM_TPU_QSA_INDEXER_DENSE_STEP.

None of them may change a value. On CPU:

- the shared-stream decode indexer selects the same blocks with the tail skip
  on as off (real StreamIndex scoring kernel in interpret mode; interpret VMEM
  starts as NaN, so a skipped page that held a live key would show);
- the decode RPA call carries ``kv_skip_group`` / ``schedule_in_hbm`` exactly
  when its flag is set, and the default call is unchanged;
- every flag lowers for TPU7x on the served decode routes, the flagged kernels
  are the ones that change, and with every flag off the lowered program is
  the one without them.

Bit-for-bit kernel results on hardware are in test_qwen4_exp_qsa_small_flags_tpu.py.
"""

import functools
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from test_qwen4_exp_shared_stream_indexer import (
    BLOCK_TOPK,
    CASES,
    RATIO,
    ReferenceTopK,
    _interpret_pallas_call,
    _tpu7x_info,
    inputs,
)

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa

streamindex = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)
bref_override = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.bref_override"
)

TAIL = "VLLM_TPU_QSA_INDEXER_TAIL_SKIP"
DENSE = "VLLM_TPU_QSA_INDEXER_DENSE_STEP"
SKIP = "VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY"
HBM = "VLLM_TPU_RPA_SCHEDULE_HBM"


def _copy_in_page_groups(self, src_ref, grid_indices):
    """Interpret-mode page DMA honouring the tail skip's page groups.

    Interpret mode cannot DMA from the flattened HBM view, so whole pages are
    copied, as test_qwen4_exp_shared_stream_indexer does, but only in the
    groups ``_for_groups`` issues; the kernel's own ``wait_in`` then waits for
    exactly those groups.
    """
    cache_kv_hbm, page_indices_ref, meta_ref = src_ref[:3]
    p_id = grid_indices[0] + bref_override._step_offset(meta_ref, self.chunk_idx)
    seq = meta_ref.batch_tile_idx[p_id]
    kv_p_start = meta_ref.bkv_idx[p_id] * self.bkv_p
    slot = self.current_copy_in_slot
    vmem_dst = self.window_ref.at[slot]
    ps = self.page_size_per_kv_packing

    for batch_idx in range(self.seq_batch_size):

        def pages(first_page, count, batch_idx=batch_idx):
            for i in range(first_page, first_page + count):
                page = self.page_id(page_indices_ref, seq + batch_idx, kv_p_start, i)
                page = jnp.minimum(page, cache_kv_hbm.shape[0] - 1)
                pltpu.make_async_copy(
                    cache_kv_hbm.at[page],
                    vmem_dst.at[batch_idx, pl.ds(i * ps, ps)],
                    self.sem_recvs.at[slot],
                ).start()

        self._for_groups(src_ref, p_id, pages)


@pytest.fixture
def interpret_groups(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x_info)
    monkeypatch.setattr(streamindex.pl, "pallas_call", _interpret_pallas_call())
    monkeypatch.setattr(
        bref_override.StreamIndexKVBufferedRef, "copy_in", _copy_in_page_groups
    )
    monkeypatch.setattr(
        pooled_qsa,
        "streamindex_topk",
        functools.partial(streamindex.streamindex_topk, topk_backend=ReferenceTopK()),
    )


@pytest.mark.parametrize("pool_table", [False, True])
@pytest.mark.parametrize("kv_block,tail", [(256, 1), (256, 2), (1024, 2), (1024, 3)])
@pytest.mark.parametrize("case", sorted(CASES))
def test_tail_skip_keeps_the_selection(
    monkeypatch, interpret_groups, case, kv_block, tail, pool_table
):
    """The tail skip fetches only the page groups holding keys of the block's
    sequence; the skipped pages (NaN in interpret VMEM) are all masked."""
    q_lens, context, tokens = CASES[case]
    args = inputs(len(case), q_lens, context, tokens, "integer")
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", str(kv_block))
    monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_MIXED_ONLY", "1")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE", str(int(pool_table)))

    def run():
        return np.asarray(
            pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=BLOCK_TOPK)
        )

    expected = run()
    monkeypatch.setenv(TAIL, str(tail))
    np.testing.assert_array_equal(run(), expected)


def test_tail_skip_page_groups():
    """``_groups_with_keys``: every group that holds a key of the sequence is
    issued, at least one, never more than the block has."""
    import types

    bkv_p, group, keys_per_page = 13, 4, 8
    bkv_sz = bkv_p * keys_per_page
    ref = types.SimpleNamespace(
        bkv_p=bkv_p, tail_skip=(group, keys_per_page, bkv_sz, 2, 1, 1)
    )
    ref._num_groups = functools.partial(
        bref_override.StreamIndexKVBufferedRef._num_groups, ref
    )
    meta = types.SimpleNamespace(
        batch_tile_idx=jnp.zeros((1,), jnp.int32), bkv_idx=jnp.zeros((1,), jnp.int32)
    )
    for seq_len in range(0, 4 * 3 * bkv_sz, 7):
        for block in range(3):
            meta.bkv_idx = jnp.full((1,), block, jnp.int32)
            count = int(
                bref_override.StreamIndexKVBufferedRef._groups_with_keys(
                    ref,
                    (None, None, meta, jnp.asarray([seq_len], jnp.int32),
                     jnp.zeros((1,), jnp.int32)),
                    0,
                )
            )
            keys = (seq_len >> 2) - block * bkv_sz
            need = -(-max(keys, 0) // (group * keys_per_page))
            assert count == min(max(need, 1), -(-bkv_p // group)), (seq_len, block)


# --- decode RPA plumbing -------------------------------------------------------

SERVED_RPA_ENV = {
    "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_SLOT_READ": "1",
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_RPA_BKV": "512",
    "VLLM_TPU_QSA_DECODE_RPA_LANES": "4",
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": "128",
    "VLLM_TPU_QSA_FUSED_SPARSE_DECODE": "0",
}


@pytest.mark.parametrize("skip,hbm", [(0, 0), (16, 0), (0, 1), (16, 1)])
def test_rpa_flags_reach_the_kernel_call(monkeypatch, skip, hbm):
    from test_qwen4_exp_paired_main_kv import reference_rpa
    from test_qwen4_exp_fused_sparse_decode import RATIO as R, TPU7X, _verify_step

    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper

    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)

    calls = []

    def rpa(queries, keys, values, kv_cache, *args, **kwargs):
        calls.append(
            {k: kwargs[k] for k in ("kv_skip_group", "schedule_in_hbm") if k in kwargs}
        )
        slot = kwargs.pop("kv_slot", None)
        kwargs.pop("kv_skip_group", None)
        kwargs.pop("schedule_in_hbm", None)
        view = kv_cache
        if slot is not None:
            rows = kv_cache.reshape(*kv_cache.shape[:2], -1, kv_cache.shape[-1])
            pair = rows[:, :, 2 * slot : 2 * slot + 2]
            view = pair.reshape(*kv_cache.shape[:2], 1, 2, kv_cache.shape[-1])
        out, _ = reference_rpa(queries, keys, values, view, *args, **kwargs)
        return out, kv_cache

    monkeypatch.setattr(wrapper, "ragged_paged_attention", rpa)
    for k, v in SERVED_RPA_ENV.items():
        monkeypatch.setenv(k, v)
    rng = np.random.default_rng(5)
    args, starts = _verify_step(rng, [300, 0, 41, 129, 7], 4, 20, 16, 128, 3)

    def attend():
        return np.asarray(
            jax.jit(
                functools.partial(
                    qsa_rpa.qsa_attention, starts=starts, kv_heads=1, ratio=R,
                    k_scale=0.375, v_scale=2.5, max_query_len=4,
                )
            )(*args)[0]
        )

    monkeypatch.setenv(SKIP, "0")
    monkeypatch.setenv(HBM, "0")
    expected = attend()
    assert calls and all(c == {} for c in calls)
    calls.clear()
    monkeypatch.setenv(SKIP, str(skip))
    monkeypatch.setenv(HBM, str(hbm))
    np.testing.assert_array_equal(attend(), expected)
    want = {
        **({"kv_skip_group": skip} if skip else {}),
        **({"schedule_in_hbm": True} if hbm else {}),
    }
    assert calls and all(c == want for c in calls)


def test_skip_group_needs_the_compact_schedule(monkeypatch):
    from test_qwen4_exp_fused_sparse_decode import TPU7X

    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)

    with pytest.raises(ValueError, match="kv_skip_group needs compact_kv_schedule"):
        wrapper.ragged_paged_attention(
            jnp.zeros((8, 4, 256), jnp.float32),
            jnp.zeros((8, 1, 256), jnp.float8_e4m3fn),
            jnp.zeros((8, 1, 256), jnp.float8_e4m3fn),
            jnp.zeros((64, 4, 1, 4, 256), jnp.float8_e4m3fn),
            jnp.zeros((8,), jnp.int32),
            jnp.zeros((64,), jnp.int32),
            jnp.zeros((9,), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            skip_kv_update=True,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
            kv_skip_group=16,
        )


# --- TPU7x lowering ------------------------------------------------------------


def _lower_rpa(monkeypatch, env, rows=128, requests=32, columns=230):
    from test_qwen4_exp_fused_sparse_decode import Q_HEADS, TPU7X, WIDTH, _lower

    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    for k, v in {**SERVED_RPA_ENV, SKIP: "0", HBM: "0", **env}.items():
        monkeypatch.setenv(k, v)
    shape = jax.ShapeDtypeStruct
    specs = (
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, 512), jnp.int32),
        shape((rows,), jnp.int32),
        shape((rows, columns), jnp.int32),
        shape((4096, 512, 1, 4, WIDTH), jnp.float8_e4m3fn),
        shape((requests + 1,), jnp.int32),
    )
    options = dict(
        kv_heads=1, ratio=4, k_scale=0.375, v_scale=2.5,
        max_query_len=rows // requests, kv_head_offset=1,
    )
    return _lower(functools.partial(qsa_rpa.qsa_attention, **options), *specs)


def test_rpa_flags_lower_for_tpu7x(monkeypatch):
    """Served verify geometry (32 x 4 rows, paired FP8 slot 1, compact 512-key
    tiles of 4 lanes): each flag lowers and changes the program; off is the
    program without the flags."""
    off = _lower_rpa(monkeypatch, {})
    assert "kv_skip_group" not in off
    skip = _lower_rpa(monkeypatch, {SKIP: "16"})
    hbm = _lower_rpa(monkeypatch, {HBM: "1"})
    both = _lower_rpa(monkeypatch, {SKIP: "16", HBM: "1"})
    assert "kv_skip_group" in skip and "kv_skip_group" in both
    assert len({off, skip, hbm, both}) == 4
    assert _lower_rpa(monkeypatch, {}) == off


def _lower_indexer(monkeypatch, env, rows=128, requests=32, columns=512):
    from test_qwen4_exp_fused_sparse_decode import TPU7X

    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    served = {
        "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
        "VLLM_TPU_QSA_DECODE_KV_BLOCK": "3328",
        "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK": "1024",
        "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": "1",
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": "1",
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": "1",
        "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY": "1",
        "VLLM_TPU_QSA_INDEXER_HOIST_META": "1",
        "VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES": "1",
        TAIL: "0",
        DENSE: "0",
    }
    for k, v in {**served, **env}.items():
        monkeypatch.setenv(k, v)
    shape = jax.ShapeDtypeStruct
    specs = dict(
        query=shape((rows, 4, 128), jnp.bfloat16),
        pool=shape((4096, 512, 1, 4, 256), jnp.float8_e4m3fn),
        lengths=shape((rows,), jnp.int32),
        table=shape((requests, columns), jnp.int32),
        req=shape((rows,), jnp.int32),
        starts=shape((requests + 1,), jnp.int32),
        context=shape((requests,), jnp.int32),
    )
    names = tuple(specs)

    def fn(*a):
        return pooled_qsa.select_blocks(
            **dict(zip(names, a)), ratio=4, block_topk=128, slot=3, max_query_len=4
        )

    device = jax.sharding.AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    with jax.sharding.use_abstract_mesh(mesh):
        return (
            jax.jit(fn).trace(*specs.values()).lower(lowering_platforms=("tpu",)).as_text()
        )


def test_indexer_flags_lower_for_tpu7x(monkeypatch):
    """Served replicated decode indexer (4 index heads x 128, FP8 pool read as
    BF16 keys, 3,328-key blocks on the pool-page table): the dense step and
    the tail skip lower and change the scoring kernel; off is unchanged."""
    off = _lower_indexer(monkeypatch, {})
    tail = _lower_indexer(monkeypatch, {TAIL: "8"})
    dense = _lower_indexer(monkeypatch, {DENSE: "1"})
    both = _lower_indexer(monkeypatch, {TAIL: "8", DENSE: "1"})
    assert len({off, tail, dense, both}) == 4
    assert _lower_indexer(monkeypatch, {}) == off


def test_tail_skip_check_catches_a_skipped_key(monkeypatch, interpret_groups):
    """Positive control: fetching only each block's first page group (dropping
    live keys) changes the selection, so the check above can fail."""
    q_lens, context, tokens = CASES["verify_k3"]
    args = inputs(3, q_lens, context, tokens, "integer")
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", "1024")
    monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_MIXED_ONLY", "1")

    def run():
        return np.asarray(
            pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=BLOCK_TOPK)
        )

    expected = run()
    monkeypatch.setenv(TAIL, "2")
    monkeypatch.setattr(
        bref_override.StreamIndexKVBufferedRef,
        "_groups_with_keys",
        lambda self, src_ref, p_id: jnp.int32(1),
    )
    assert not np.array_equal(run(), expected)
