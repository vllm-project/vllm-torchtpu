# SPDX-License-Identifier: Apache-2.0
"""q38 RPAD-HIGHROWS F4: ``VLLM_TPU_QSA_DECODE_RPA_PT_PREFIX``.

The union-route RPA decode kernel takes its page table in HBM instead of as a
scalar prefetch and copies into SMEM only each sequence's used prefix
(``kernel.pt_prefix_bounds``). Batched RPA does not run in Pallas interpret
mode, so on CPU this checks:

- the coverage invariant the change rests on: on the schedule the union route
  generates (the schedule kernel in interpret mode, TPU7x info), every
  page-table index the compact KV fetch can read, for real lanes, idle lanes
  and no-op steps, with and without SKIP_EMPTY, lies inside the copied words,
  so it reads the value the full copy holds; a one-tile-short copy and a copy
  without sequence 0's first tile break it (positive controls);
- routing of the key to the RPA call, the wrapper guard, the kernel name and
  metadata (flag-off programs keep their metadata);
- the compile key.

Outputs are compared bit for bit on hardware (the F4 microbench).
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.batched_rpa import configs, kernel, schedule

KEY = "VLLM_TPU_QSA_DECODE_RPA_PT_PREFIX"
TPU7X = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
SEQS, ROWS_PER_SEQ, PPS, PAGE, BKV, LANES = 32, 3, 2080, 4, 640, 4
BKV_P = BKV // PAGE


def _cfgs(**serve_kw):
    return configs.RpaConfigs(
        block=configs.BlockSizes(bq_sz=ROWS_PER_SEQ, bq_c_sz=ROWS_PER_SEQ, bkv_sz=BKV, batch_size=LANES),
        model=configs.ModelConfigs(num_q_heads=12, num_kv_heads=1, head_dim=256, mask_value=-1e30),
        serve=configs.ServingConfigs(
            num_seqs=SEQS, page_size=PAGE, total_q_tokens=SEQS * ROWS_PER_SEQ,
            num_page_indices=SEQS * PPS, dtype_q=jnp.float32, dtype_kv=jnp.float8_e4m3fn,
            dtype_out=jnp.float32, skip_kv_update=True, lean_kv_fetch=True, fast_kv_fetch=True,
            compact_kv_schedule=True, decode_query_size=ROWS_PER_SEQ, read_only_cache=True,
            **serve_kw,
        ),
        vmem_limit_bytes=64 * 1024 * 1024,
        mode=configs.RpaCase.DECODE,
    )


def _kv_lens(case):
    rng = np.random.default_rng(17 + case)
    # Union tokens of a k2 verify request on one DCP rank: about 4 x 210-243
    # groups (real-selection proxy), with exact tile multiples, a 1-token tail,
    # empty and padded request slots, and a request at the 13-tile cap.
    lens = rng.integers(4 * 120, 4 * 330, SEQS).astype(np.int32)
    lens[1], lens[2], lens[3], lens[4] = BKV, 2 * BKV, 1, 4 * 513 * 1
    lens[5] = PPS * PAGE  # the whole 13-tile row
    lens[-3:] = 0  # padded request slots
    lens[6] = 0  # a live request owning no group here (its rows stay; no KV tile is scheduled)
    if case == 1:
        lens[0] = 0  # sequence 0 empty: idle lanes index its first tile
    # Tiles pack four to a step in task order; make the last step hold one
    # tile, so three idle lanes (page index 0, no tokens) are scheduled.
    while sum(-(-int(x) // BKV) for x in lens) % LANES != 1:
        lens[7] += BKV
    return lens


def _schedule(lens, monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    orig = pl.pallas_call
    monkeypatch.setattr(pl, "pallas_call", lambda *a, **k: orig(*a, **{**k, "interpret": True}))
    q_lens = np.full(SEQS, ROWS_PER_SEQ, np.int32)
    q_lens[-3:] = 0  # padded request slots hold no rows
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    kv = jnp.asarray(lens)
    cfgs = _cfgs()
    # The wrapper's schedule config: the selection path's lens, and the fields
    # that only change the fetch / mask cleared (one schedule serves them all).
    sched_cfgs = dataclasses.replace(cfgs, serve=dataclasses.replace(
        cfgs.serve, kv_slot=None, kv_skip_group=0, selection_mask=False,
        selection_word_rows=False, read_only_cache=False, page_table_prefix=False))
    sched = schedule.generate_rpa_metadata(
        jnp.asarray(cu), kv, kv, jnp.zeros_like(kv), jnp.full((3,), SEQS, jnp.int32), cfgs=sched_cfgs,
    )
    steps = int(np.asarray(sched.actual_steps)[0])
    ent = np.asarray(sched.dma_kv_cache.data).reshape(-1, LANES, cfgs.kv_schedule_entries, 3)[:steps, :, 0, :]
    return cfgs, steps, ent


def _reads(ent, skip):
    """Page-table indices the compact fetch (bref_override, compact branch)
    reads: every scheduled lane with its tokens, and every lane again as a
    no-op (tokens 0; a no-op step reads a chunk's step-0 entries, a subset)."""
    last = SEQS * PPS - 1
    out = set()
    for tokens_zero in (False, True):
        for step in ent:
            for p0, _, tokens in step:
                tokens = 0 if tokens_zero else int(tokens)
                if not skip:
                    idx = range(BKV_P)
                else:
                    idx = [i for lo in range(0, BKV_P, skip) if tokens > lo * PAGE
                           for i in range(lo, min(lo + skip, BKV_P))]
                out.update(int(p0) if i == 0 else min(int(p0) + i, last) for i in idx)
    return out


def _copied(lens, *, short=False, no_first=False):
    words = set()
    for s, k_len in enumerate(lens):
        if short or no_first:
            n = (int(k_len) + BKV - 1) // BKV * BKV_P
            n = max(n - BKV_P, 0) if short else n
            if s == 0 and not no_first:
                n = max(n, BKV_P)
            lo, hi = s * PPS // kernel.PT_ALIGN * kernel.PT_ALIGN, -(-(s * PPS + min(n, PPS)) // kernel.PT_ALIGN) * kernel.PT_ALIGN
            hi = min(hi, SEQS * PPS)
        else:
            lo, hi = kernel.pt_prefix_bounds(
                s, int(k_len), pages_per_seq=PPS, bkv_sz=BKV, bkv_p=BKV_P, num_page_indices=SEQS * PPS,
            )
            lo, hi = int(lo), int(hi)
        words.update(range(lo, hi))
    return words


@pytest.mark.parametrize("case", [0, 1])
@pytest.mark.parametrize("skip", [0, 32])
def test_every_fetched_page_index_is_copied(monkeypatch, case, skip):
    lens = _kv_lens(case)
    cfgs, steps, ent = _schedule(lens, monkeypatch)
    assert steps > 0 and cfgs.bkv_p == BKV_P
    idle = sum(1 for step in ent for p0, _, tokens in step if p0 == 0 and tokens == 0)
    assert idle >= LANES - 1, idle  # the last step's idle lanes are in the schedule
    reads = _reads(ent, skip)
    copied = _copied(lens)
    missing = sorted(reads - copied)
    assert not missing, missing[:8]
    # The copy is a prefix: far fewer words than the whole [32, 2080] table.
    assert len(copied) < 0.5 * SEQS * PPS


@pytest.mark.parametrize("skip", [0, 32])
def test_positive_controls_break_coverage(monkeypatch, skip):
    lens = _kv_lens(1)
    _, _, ent = _schedule(lens, monkeypatch)
    reads = _reads(ent, skip)
    assert reads - _copied(lens, short=True), "a one-tile-short copy must miss fetched indices"
    if not skip:  # idle lanes only read page index 0 onward without the skip groups
        assert reads - _copied(lens, no_first=True), "sequence 0's first tile is read by idle lanes"


def test_union_route_passes_the_key_only_when_set_and_read_only(monkeypatch):
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

    for k, v in {"VLLM_TPU_QSA_DECODE_RPA_UNION": "640", "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
                 "VLLM_TPU_QSA_DECODE_SLOT_READ": "1"}.items():
        monkeypatch.setenv(k, v)
    seen = []

    def spy(query, k, v, cache, kv_lens, pages, cu, dist, **kw):
        seen.append(kw.get("page_table_prefix"))
        return jnp.zeros(query.shape, jnp.float32), cache, jnp.zeros(query.shape[:2], jnp.float32)

    monkeypatch.setattr(wrapper, "ragged_paged_attention", spy)
    rows = SEQS * ROWS_PER_SEQ
    starts = jnp.arange(SEQS + 1, dtype=jnp.int32) * ROWS_PER_SEQ
    meta = (starts, jnp.zeros((SEQS, PPS), jnp.int32), jnp.zeros((SEQS, PPS * PAGE), jnp.int32),
            jnp.zeros((SEQS, 16), jnp.int32), jnp.ones((rows,), bool))
    cache = jnp.zeros((2, 512, 1, 4, 256), jnp.float8_e4m3fn)
    q = jnp.zeros((rows, 12, 256), jnp.bfloat16)
    got = {}
    for value, ro in ((None, True), ("0", True), ("1", True), ("1", False)):
        monkeypatch.delenv(KEY, raising=False)
        if value is not None:
            monkeypatch.setenv(KEY, value)
        seen.clear()
        jax.eval_shape(lambda q: qsa_rpa.union_decode_attention(
            q, None, None, None, cache, starts, ratio=4, k_scale=1.0, v_scale=1.0, kv_head_offset=1,
            max_query_len=ROWS_PER_SEQ, dcp=(jnp.zeros((1,), jnp.int32), 4), read_only=ro, metadata=meta), q)
        got[(value, ro)] = list(seen)
    assert got[(None, True)] == [None] and got[("0", True)] == [None]
    assert got[("1", True)] == [True]
    assert got[("1", False)] == [None]


def test_wrapper_guard_and_kernel_identity():
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper

    off, on = _cfgs(), _cfgs(page_table_prefix=True)
    assert kernel.get_kernel_name(on) == kernel.get_kernel_name(off) + "-pp"
    assert "serve.page_table_prefix" not in kernel.get_kernel_metadata(off)
    assert kernel.get_kernel_metadata(on)["serve.page_table_prefix"] in (True, 1)
    import inspect

    src = inspect.getsource(wrapper.ragged_paged_attention)
    assert 'raise ValueError("page_table_prefix needs decode_only and read_only_cache")' in src
    assert "page_table_prefix=False" in inspect.getsource(wrapper)  # the schedule ignores it


def test_key_is_registered_and_compile_hashed(monkeypatch):
    from vllm_torchtpu.compilation import tpu_compiler as tc

    assert KEY in envs.environment_variables
    assert KEY not in tc._TPU_COMPILE_ENV_IGNORED
    monkeypatch.delenv(KEY, raising=False)
    base = tc._tpu_compile_env_factors()
    monkeypatch.setenv(KEY, "1")
    on = tc._tpu_compile_env_factors()
    assert base != on
