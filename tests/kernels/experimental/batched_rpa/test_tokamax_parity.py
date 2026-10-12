# SPDX-License-Identifier: Apache-2.0
"""tokamax's batched RPA against the in-tree kernel it replaces, on TPU.

`PallasAttentionBackendImpl._build_rpa_kernel` hands the CUSTOM backend's
batched RPA to tokamax's operator, so the two must agree on every case that
routing lets through: HEAD_ALONG_SUBLANE pools, causal masks, sliding windows,
soft caps, fp8 KV with static scales and speculative-verify windows. The
in-tree kernel is itself checked against the reference in correctness_test.py.

`distribution` follows the runner's convention, `[decodes, decodes, total]`:
vLLM never fills the PREFILL bucket, so a prompt with no cached tokens rides in
the mixed bucket. The PREFILL bucket itself is checked separately, against the
reference, at the end.
"""

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, utils, wrapper

pytestmark = pytest.mark.tpu_test


@pytest.fixture
def tpu():
    """Skips off TPU. A fixture rather than a skipif: the conftest refuses test
    modules that initialize a JAX backend while being collected."""
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernels need a TPU.")


def _inputs(seq_lens, distribution, *, kv_dtype, page_size=16, num_pages=64):
    """Batched RPA inputs for `seq_lens` = [(q_len, kv_len), ...]."""
    rng = np.random.default_rng(0)

    def rand(shape, dtype):
        return jnp.asarray(rng.random(size=shape, dtype=np.float32)).astype(dtype)

    num_q_heads, num_kv_heads, head_dim = 8, 2, 128
    max_tokens, max_seqs = 128, 8
    packing = utils.get_dtype_packing(kv_dtype)
    heads_x2 = utils.align_to(num_kv_heads * 2, packing)
    pages_per_seq = max(utils.align_to(kv, page_size) // page_size for _, kv in seq_lens)

    q = rand((max_tokens, num_q_heads, head_dim), jnp.bfloat16)
    k = rand((max_tokens, num_kv_heads, head_dim), kv_dtype)
    v = rand((max_tokens, num_kv_heads, head_dim), kv_dtype)
    kv_cache = rand((num_pages, page_size, heads_x2 // packing, packing, head_dim), kv_dtype)

    cu_q_lens, kv_lens, page_indices, next_page = [0], [], [], 0
    for q_len, kv_len in seq_lens:
        cu_q_lens.append(cu_q_lens[-1] + q_len)
        kv_lens.append(kv_len)
        n = utils.align_to(kv_len, page_size) // page_size
        pages = list(range(next_page, next_page + n))
        next_page += n
        page_indices.extend(pages + [0] * (pages_per_seq - n))
    pad = max_seqs - len(seq_lens)
    page_indices.extend([0] * (pad * pages_per_seq))
    return (
        q,
        k,
        v,
        kv_cache,
        jnp.asarray(kv_lens + [0] * pad, jnp.int32),
        jnp.asarray(page_indices, jnp.int32),
        jnp.asarray(cu_q_lens + [cu_q_lens[-1]] * pad, jnp.int32),
        jnp.asarray(distribution, jnp.int32),
    )


_DECODE = ([(1, 40), (1, 17), (1, 64)], [3, 3, 3])
_PREFILL = ([(24, 24), (40, 40)], [0, 0, 2])
_MIXED = ([(1, 33), (16, 16), (12, 50)], [1, 1, 3])


@pytest.mark.parametrize(
    "case, kwargs, kv_dtype",
    [
        (_DECODE, {}, jnp.bfloat16),
        (_PREFILL, {}, jnp.bfloat16),
        (_MIXED, {}, jnp.bfloat16),
        (_MIXED, {"sliding_window": 16}, jnp.bfloat16),
        (_MIXED, {"soft_cap": 30.0}, jnp.bfloat16),
        (_MIXED, {"k_scale": 0.5, "v_scale": 0.25}, jnp.float8_e4m3fn),
        (([(2, 40), (2, 21)], [2, 2, 2]), {"decode_query_size": 2}, jnp.bfloat16),
    ],
    ids=["decode", "prefill", "mixed", "sliding-window", "soft-cap", "fp8-kv", "spec-verify"],
)
def test_tokamax_matches_in_tree(tpu, case, kwargs, kv_dtype):
    tokamax_rpa = pytest.importorskip(
        "tokamax._src.ops.experimental.batched_rpa.pallas_mosaic_tpu"
    )
    seq_lens, distribution = case
    args = _inputs(seq_lens, distribution, kv_dtype=kv_dtype)
    kwargs = dict(sm_scale=128**-0.5, **kwargs)
    num_tokens = sum(q_len for q_len, _ in seq_lens)

    want_out, want_kv = wrapper.ragged_paged_attention(
        *jax.tree.map(jnp.copy, args),
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        **kwargs,
    )
    got_out, got_kv = tokamax_rpa.PallasTpuBatchedRpa()(
        *jax.tree.map(jnp.copy, args), **kwargs
    )

    np.testing.assert_allclose(
        np.asarray(got_out[:num_tokens], np.float32),
        np.asarray(want_out[:num_tokens], np.float32),
        atol=2e-2,
        rtol=2e-2,
    )
    assert got_kv.shape == want_kv.shape
    np.testing.assert_array_equal(
        np.asarray(got_kv.astype(jnp.float32)), np.asarray(want_kv.astype(jnp.float32))
    )


def _reference():
    """`ref_ragged_paged_attention` from the in-tree kernel's correctness test."""
    path = pathlib.Path(__file__).with_name("correctness_test.py")
    spec = importlib.util.spec_from_file_location("_batched_rpa_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ref_ragged_paged_attention


@pytest.mark.parametrize(
    "impl",
    [
        pytest.param(
            "in-tree",
            marks=pytest.mark.xfail(
                strict=True,
                reason="The in-tree kernel gets prefill-only sequences in the "
                "PREFILL bucket wrong (40% of elements off on TPU v7x, "
                "2026-10-10). The runner never fills that bucket, so serving "
                "is unaffected; tokamax handles it.",
            ),
        ),
        "tokamax",
    ],
)
def test_prefill_bucket_against_reference(tpu, impl):
    """The PREFILL bucket (`distribution[0] < distribution[1]`), which vLLM
    never fills, held against the reference for each kernel on its own."""
    tokamax_rpa = pytest.importorskip(
        "tokamax._src.ops.experimental.batched_rpa.pallas_mosaic_tpu"
    )
    seq_lens = [(24, 24), (40, 40)]
    args = _inputs(seq_lens, [0, 2, 2], kv_dtype=jnp.bfloat16)
    num_tokens = sum(q_len for q_len, _ in seq_lens)
    kwargs = dict(sm_scale=128**-0.5)

    want, _ = _reference()(*jax.tree.map(jnp.copy, args), **kwargs)
    if impl == "in-tree":
        got, _ = wrapper.ragged_paged_attention(
            *jax.tree.map(jnp.copy, args),
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
            **kwargs,
        )
    else:
        got, _ = tokamax_rpa.PallasTpuBatchedRpa()(*jax.tree.map(jnp.copy, args), **kwargs)
    # correctness_test.py's tolerance for 16-bit KV.
    np.testing.assert_allclose(
        np.asarray(got[:num_tokens], np.float32),
        np.asarray(want[:num_tokens], np.float32),
        atol=0.2,
        rtol=0.2,
    )
