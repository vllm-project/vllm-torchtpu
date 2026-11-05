# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the main-KV DCP-2 equality harness (CPU, Pallas interpret)."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_TESTS = os.path.abspath(os.path.join(HERE, "..", ".."))
for path in (HERE, KERNEL_TESTS):
    if path not in sys.path:
        sys.path.insert(0, path)


ON_TPU = os.environ.get("DCP2_HARNESS_DEVICE", "cpu").lower() == "tpu"


@pytest.fixture(autouse=True)
def cpu_device():
    """CPU mode runs on the CPU backend even on a TPU host (the pod pytest
    recipe sets JAX_PLATFORMS=tpu,cpu): default-precision TPU matmuls round
    through bf16, far outside the 2-ulp bar the CPU oracles are held to."""
    if ON_TPU:
        yield
        return
    import jax

    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.fixture
def interpret(monkeypatch):
    """Pallas kernels in the legacy interpret evaluator on CPU (TPU7x info);
    a no-op under DCP2_HARNESS_DEVICE=tpu (real Mosaic kernels)."""
    if ON_TPU:
        return
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def reference_rpa_lse(queries, keys, values, kv_cache, kv_lens, page_indices,
                      cu_q_lens, distribution, *, sm_scale, k_scale, v_scale,
                      out_dtype, kv_layout, skip_kv_update, return_lse=False, **_):
    """test_qwen4_exp_paired_main_kv.reference_rpa (decode-only
    HEAD_ALONG_SUBLANE RPA; combined head 2h = K, 2h + 1 = V) plus the
    ``return_lse`` output (out, cache, lse [rows, heads], natural log)."""
    import jax
    import jax.numpy as jnp

    assert skip_kv_update
    kv_heads = keys.shape[1]
    per_kv = queries.shape[1] // kv_heads
    pages_per_seq = len(page_indices) // len(kv_lens)
    page_size = kv_cache.shape[1]
    combined = kv_cache.reshape(*kv_cache.shape[:2], -1, kv_cache.shape[-1])
    out = jnp.zeros(queries.shape, jnp.float32)
    lse_out = jnp.full(queries.shape[:2], -jnp.inf, jnp.float32)
    for seq in range(len(kv_lens)):
        pages = page_indices[seq * pages_per_seq : (seq + 1) * pages_per_seq]
        cache = combined[pages].reshape(pages_per_seq * page_size, -1, combined.shape[-1])
        key = cache[:, 0 : 2 * kv_heads : 2].astype(jnp.float32) * k_scale
        value = cache[:, 1 : 2 * kv_heads : 2].astype(jnp.float32) * v_scale
        visible = jnp.arange(len(cache)) < kv_lens[seq]
        row = jnp.minimum(cu_q_lens[seq], len(queries) - 1)
        q = queries[row].astype(jnp.float32).reshape(kv_heads, per_kv, -1)
        score = jnp.einsum("hqd,thd->hqt", q, key) * sm_scale
        score = jnp.where(visible[None, None], score, -jnp.inf)
        prob = jax.nn.softmax(score, axis=-1)
        prob = jnp.where(jnp.isnan(prob), 0, prob)
        result = jnp.einsum("hqt,thd->hqd", prob, value).reshape(-1, q.shape[-1])
        lse = jax.nn.logsumexp(score, axis=-1).reshape(-1)
        live = cu_q_lens[seq + 1] > cu_q_lens[seq]
        out = out.at[row].set(jnp.where(live, result, out[row]))
        lse_out = lse_out.at[row].set(jnp.where(live, lse, lse_out[row]))
    if return_lse:
        return out.astype(out_dtype), kv_cache, lse_out
    return out.astype(out_dtype), kv_cache


@pytest.fixture
def rpa_stand_in(monkeypatch):
    """Batched RPA cannot run in interpret mode: the decode-only reference the
    existing QSA tests use, extended with ``return_lse`` (set6's contract:
    ``wrapper.ragged_paged_attention(..., return_lse=True)`` -> out, cache, lse)."""
    if ON_TPU:
        return
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper

    monkeypatch.setattr(wrapper, "ragged_paged_attention", reference_rpa_lse)
