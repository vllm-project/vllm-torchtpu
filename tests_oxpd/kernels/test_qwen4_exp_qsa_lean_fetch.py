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
def test_lean_fetch_reaches_rpa_and_matches(monkeypatch, kernels, lean):
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
                qsa_rpa.qsa_attention, starts=starts, kv_heads=1, ratio=RATIO,
                k_scale=0.375, v_scale=2.5, max_query_len=4,
            )
        )(*args)
        outs.append(np.asarray(out))
    assert seen == [False, lean]
    np.testing.assert_array_equal(outs[0], outs[1])


def test_schedule_lands_tile_pages_consecutively(monkeypatch):
    """The decode schedule's cache entries for a tile are page-table entries
    p0 + i landing at i * page_size: what the lean fetch reconstructs."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import (
        configs,
        schedule,
    )
    from jax.experimental import pallas as pl

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
            num_seqs=seqs, page_size=page, total_q_tokens=seqs,
            num_page_indices=seqs * pages_per_seq, dtype_q=jnp.float32,
            dtype_kv=jnp.float8_e4m3fn, dtype_out=jnp.float32,
            skip_kv_update=True, lean_kv_fetch=True,
        ),
        vmem_limit_bytes=64 * 1024 * 1024,
        mode=configs.RpaCase.DECODE,
    )
    q_lens = (kv_lens > 0).astype(jnp.int32)
    cu = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(q_lens)))
    sched = schedule.generate_rpa_metadata(
        cu, kv_lens - q_lens, kv_lens - q_lens, q_lens,
        jnp.full((3,), seqs, jnp.int32), cfgs=cfgs,
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
