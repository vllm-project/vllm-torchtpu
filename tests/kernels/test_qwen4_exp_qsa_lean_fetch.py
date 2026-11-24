# SPDX-License-Identifier: Apache-2.0
"""QSA decode with ``VLLM_TPU_QSA_DECODE_LEAN_FETCH``.

Batched RPA's HEAD_ALONG_SUBLANE cache fetch then loads one schedule word per
page (its size) instead of three: a tile's pages are consecutive page-table
entries landing at ``i * page_size``. The DMAs are unchanged. Batched RPA does
not run in Pallas interpret mode, so on CPU this checks the schedule invariant
the lean fetch relies on, the routing and the TPU7x lowering; outputs are
compared bit for bit on hardware by scripts/bench_qsa_rpa_decode.py
(``test_lean_fetch``).
"""

import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from test_qwen4_exp_fused_sparse_decode import (  # noqa: F401
    FLAG,
    Q_HEADS,
    RATIO,
    TPU7X,
    WIDTH,
    _lower,
    _verify_step,
    kernels,
)

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

LEAN = "VLLM_TPU_QSA_DECODE_LEAN_FETCH"


@pytest.mark.parametrize("lean", [False, True])
def test_lean_fetch_reaches_rpa_and_matches(
    monkeypatch,
    kernels,  # noqa: F811
    lean,
):
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper

    seen = []
    reference = wrapper.ragged_paged_attention

    def rpa(*args, **kwargs):
        seen.append(kwargs.get("lean_kv_fetch", False))
        return reference(*args, **kwargs)

    monkeypatch.setattr(wrapper, "ragged_paged_attention", rpa)
    rng = np.random.default_rng(5)
    args, starts = _verify_step(rng, [300, 0, 41, 129, 7], 4, 20, 16, 128, 3)
    outs = []
    for value in ("0", "1" if lean else "0"):
        monkeypatch.setenv(FLAG, "0")
        monkeypatch.setenv(LEAN, value)
        out, _ = jax.jit(
            functools.partial(
                qsa_rpa.qsa_attention,
                starts=starts,
                kv_heads=1,
                ratio=RATIO,
                k_scale=0.375,
                v_scale=2.5,
                max_query_len=4,
            )
        )(*args)
        outs.append(np.asarray(out))
    assert seen == [False, lean]
    np.testing.assert_array_equal(outs[0], outs[1])


def test_schedule_lands_tile_pages_consecutively(monkeypatch):
    """The decode schedule's cache entries for a tile are page-table entries
    p0 + i landing at i * page_size: what the lean fetch reconstructs."""
    from jax.experimental import pallas as pl

    from vllm_torchtpu.kernels.experimental.batched_rpa import (
        configs,
        schedule,
    )

    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    seqs, pages_per_seq, page = 6, 129, RATIO
    kv_lens = jnp.asarray([513, 0, 4, 2, 300, 257], jnp.int32)
    cfgs = configs.RpaConfigs(
        block=configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8),
        model=configs.ModelConfigs(
            num_q_heads=6, num_kv_heads=2, head_dim=WIDTH, mask_value=-1e30
        ),
        serve=configs.ServingConfigs(
            num_seqs=seqs,
            page_size=page,
            total_q_tokens=seqs,
            num_page_indices=seqs * pages_per_seq,
            dtype_q=jnp.float32,
            dtype_kv=jnp.float8_e4m3fn,
            dtype_out=jnp.float32,
            skip_kv_update=True,
            lean_kv_fetch=True,
        ),
        vmem_limit_bytes=64 * 1024 * 1024,
        mode=configs.RpaCase.DECODE,
    )
    q_lens = (kv_lens > 0).astype(jnp.int32)
    cu = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(q_lens)))
    sched = schedule.generate_rpa_metadata(
        cu,
        kv_lens - q_lens,
        kv_lens - q_lens,
        q_lens,
        jnp.full((3,), seqs, jnp.int32),
        cfgs=cfgs,
    )
    entries = np.asarray(sched.dma_kv_cache.data).reshape(-1, cfgs.bkv_p_cache, 3)
    steps = int(np.asarray(sched.actual_steps)[0]) * cfgs.batch_size
    assert steps > 0
    for tile in entries[:steps]:
        if tile[0, 2] == 0:  # idle lane
            continue
        p0 = tile[0, 0]
        i = np.arange(cfgs.bkv_p_cache)
        np.testing.assert_array_equal(
            tile[:, 0], np.minimum(p0 + i, seqs * pages_per_seq - 1)
        )
        np.testing.assert_array_equal(tile[:, 1], i * page)


@pytest.mark.parametrize("offset", [0, 1])
def test_lean_fetch_lowers_for_tpu7x(monkeypatch, offset):
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    monkeypatch.setenv(FLAG, "0")
    monkeypatch.setenv(LEAN, "1")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE", "1")
    rows, columns = 32, 64
    shape = jax.ShapeDtypeStruct
    specs = (
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, 512), jnp.int32),
        shape((rows,), jnp.int32),
        shape((rows, columns), jnp.int32),
        shape((4096, 512, 1, 4, WIDTH), jnp.float8_e4m3fn),
        shape((rows // 4 + 1,), jnp.int32),
    )
    options = dict(kv_heads=1, ratio=RATIO, k_scale=0.375, v_scale=2.5)
    if offset:
        options["kv_head_offset"] = offset
    text = _lower(
        functools.partial(qsa_rpa.qsa_attention, max_query_len=4, **options), *specs
    )
    assert 'kernel_name = "RPAd' in text
    assert 'kernel_name = "RPAm' not in text


@pytest.mark.parametrize("offset", [0, 1])
def test_fast_fetch_lowers_for_tpu7x(monkeypatch, offset):
    """VLLM_TPU_QSA_DECODE_FAST_FETCH: the full-tile fetch lowers to Mosaic."""
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    for name, value in (
        (FLAG, "0"),
        (LEAN, "1"),
        ("VLLM_TPU_QSA_DECODE_FAST_FETCH", "1"),
        ("VLLM_TPU_QSA_DECODE_SLOT_READ", "1"),
        ("VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE", "1"),
    ):
        monkeypatch.setenv(name, value)
    rows, columns = 32, 64
    shape = jax.ShapeDtypeStruct
    specs = (
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, 512), jnp.int32),
        shape((rows,), jnp.int32),
        shape((rows, columns), jnp.int32),
        shape((4096, 512, 1, 4, WIDTH), jnp.float8_e4m3fn),
        shape((rows // 4 + 1,), jnp.int32),
    )
    options = dict(kv_heads=1, ratio=RATIO, k_scale=0.375, v_scale=2.5)
    if offset:
        options["kv_head_offset"] = offset
    text = _lower(
        functools.partial(qsa_rpa.qsa_attention, max_query_len=4, **options), *specs
    )
    assert 'kernel_name = "RPAd' in text


class _Ref:
    """A ref stand-in that only records the index path a DMA is built from."""

    shape = (1, 1, 1, 1, 1)

    def __init__(self, path=()):
        self.path = path

    @property
    def at(self):
        return self

    def __getitem__(self, index):
        return _Ref(self.path + (index,))

    def reshape(self, *shape):
        return self


class _Smem:
    """Flat SMEM words that refuse a read past the end, as hardware would not."""

    def __init__(self, values):
        self.values = np.asarray(values, np.int32).reshape(-1)
        self.reads = []

    @property
    def at(self):
        return self

    def __getitem__(self, pos):
        pos = int(pos)
        self.reads.append(pos)
        if not 0 <= pos < self.values.size:
            raise IndexError(f"SMEM read at {pos} of {self.values.size} words")
        return jnp.int32(self.values[pos])


def _bounds(index):
    from jax.experimental import pallas as pl

    if isinstance(index, pl.Slice):
        return int(index.start), int(index.size)
    return int(index)


@pytest.mark.parametrize("compact,lean", [(True, False), (True, True), (False, False)])
def test_cache_fetch_reads_only_the_schedule_it_has(monkeypatch, compact, lean):
    """VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE keeps one KV-cache entry per
    lane-tile. The HEAD_ALONG_SUBLANE fetch must take the compact path whether
    or not lean_kv_fetch is on: the per-page path reads bkv_p_cache entries
    per lane, past the end of the compact schedule. Without the compact
    schedule the per-page path is unchanged."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import (
        bref_override,
        configs,
        schedule,
    )

    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    seqs, pages_per_seq, page = 6, 129, RATIO
    cfgs = configs.RpaConfigs(
        block=configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8),
        model=configs.ModelConfigs(
            num_q_heads=6, num_kv_heads=2, head_dim=WIDTH, mask_value=-1e30
        ),
        serve=configs.ServingConfigs(
            num_seqs=seqs,
            page_size=page,
            total_q_tokens=seqs,
            num_page_indices=seqs * pages_per_seq,
            dtype_q=jnp.float32,
            dtype_kv=jnp.float8_e4m3fn,
            dtype_out=jnp.float32,
            skip_kv_update=True,
            lean_kv_fetch=lean,
            compact_kv_schedule=compact,
        ),
        vmem_limit_bytes=64 * 1024 * 1024,
        mode=configs.RpaCase.DECODE,
    )
    steps, lanes, pages = 3, cfgs.batch_size, cfgs.bkv_p_cache
    entries = cfgs.kv_schedule_entries
    assert entries == (1 if compact else pages) and pages > 1
    # Lane b of the last step: tile starting at page-table entry 7 * b, b * 31
    # tokens (lane 0 idle, the other lanes span several pages).
    tokens = np.arange(lanes) * 31
    p0 = np.arange(lanes) * 7
    cache = np.zeros((steps, lanes, entries, 3), np.int32)
    i = np.arange(entries)
    cache[-1, :, :, 0] = p0[:, None] + i
    cache[-1, :, :, 1] = i * page
    cache[-1, :, :, 2] = (
        tokens[:, None] if compact else np.clip(tokens[:, None] - i * page, 0, page)
    )
    dma_kv_cache = _Smem(cache)
    page_table = _Smem(np.arange(seqs * pages_per_seq) * 3)
    new_struct = schedule.HeadAlongSublaneDmaNew
    sched = types.SimpleNamespace(
        dma_kv_cache=schedule.SmemWrapper(dma_kv_cache, cache.shape),
        dma_kv_new=schedule.SmemArrayOfStructs(
            _Smem(np.zeros(steps * lanes * cfgs.bkv_p_new * new_struct.num_fields())),
            (steps, lanes, cfgs.bkv_p_new),
            new_struct,
            new_struct.num_fields(),
        ),
    )
    sched.get_dma_kv_cache = functools.partial(
        schedule.RpaSchedule.get_dma_kv_cache, sched
    )
    copies = []

    class _Copy:
        def __init__(self, src, dst, sem):
            if src.path[0] == "cache":
                lane, offset = dst.path[-1]
                copies.append((lane, _bounds(offset), _bounds(src.path[-1])))

        def start(self):
            pass

    monkeypatch.setattr(bref_override.pltpu, "make_async_copy", _Copy)
    ref = types.SimpleNamespace(
        cfgs=cfgs,
        src_ref=(_Ref(("cache",)), _Ref(("new",)), sched, page_table),
        sem_recvs=_Ref(("sem",)),
        window_ref=_Ref(("window",)),
    )
    bref_override.KVBufferedRefHeadAlongSublane.copy_in(ref, steps - 1, 0, steps)

    # Every cache-schedule read is an entry of the last step's lanes.
    first = (steps - 1) * lanes * entries * 3
    assert all(first <= pos < cache.size for pos in dma_kv_cache.reads)
    want = []
    for b in range(lanes):
        for j in range(pages):
            sz = int(np.clip(tokens[b] - j * page, 0, page))
            p = min(p0[b] + j, seqs * pages_per_seq - 1)
            want.append((b, (j * page, sz), (p * 3 * page, sz)))
    assert copies == want
