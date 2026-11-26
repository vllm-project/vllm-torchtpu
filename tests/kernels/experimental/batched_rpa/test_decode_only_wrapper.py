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
"""Tests for batched RPA static decode_only=True wrapper path."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_TPU_7X_INFO = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _find_pallas_eqns(jaxpr):
    eqns = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            eqns.append(eqn)
        for val in eqn.params.values():
            if hasattr(val, "jaxpr"):
                eqns.extend(_find_pallas_eqns(val.jaxpr))
    return eqns


def _make_dummy_inputs(kv_layout: configs.KVLayout):
    max_tokens = 16
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128
    num_pages = 16
    page_size = 128
    max_seqs = 8
    pages_per_seq = 2

    q = jnp.zeros((max_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16)
    k = jnp.zeros((max_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    v = jnp.zeros((max_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    kv_shape = wrapper.get_kv_cache_shape(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        jnp.bfloat16,
        kv_layout=kv_layout,
        chip_version=pltpu.ChipVersion.TPU_7X,
    )
    kv_cache = jnp.zeros(kv_shape, dtype=jnp.bfloat16)
    kv_lens = jnp.ones((max_seqs,), dtype=jnp.int32)
    page_indices = jnp.zeros((max_seqs * pages_per_seq,), dtype=jnp.int32)
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32)
    distribution = jnp.array([max_seqs, max_seqs, max_seqs], dtype=jnp.int32)
    return q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution


@pytest.mark.parametrize(
    "kv_layout",
    [configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE],
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_wrapper_decode_only_omits_mixed_and_preserves_aliasing(kv_layout, return_lse):
    """decode_only=True statically omits RPAm and preserves input-output aliases."""
    inputs = _make_dummy_inputs(kv_layout=kv_layout)
    with mock.patch.object(pltpu, "get_tpu_info", return_value=_TPU_7X_INFO):
        mixed = jax.make_jaxpr(
            lambda *args: wrapper.ragged_paged_attention(
                *args, kv_layout=kv_layout, return_lse=return_lse, decode_only=False
            )
        )(*inputs)
        decode = jax.make_jaxpr(
            lambda *args: wrapper.ragged_paged_attention(
                *args, kv_layout=kv_layout, return_lse=return_lse, decode_only=True
            )
        )(*inputs)

    mixed_calls = [
        (e.params.get("name"), e.params.get("input_output_aliases"))
        for e in _find_pallas_eqns(mixed.jaxpr)
    ]
    decode_calls = [
        (e.params.get("name"), e.params.get("input_output_aliases"))
        for e in _find_pallas_eqns(decode.jaxpr)
    ]

    assert len(mixed_calls) == 4
    assert mixed_calls[0][0] == "rpa_metadata_schedule"
    assert mixed_calls[1][0].startswith("RPAd-")
    assert mixed_calls[2][0] == "rpa_metadata_schedule"
    assert mixed_calls[3][0].startswith("RPAm-")

    assert len(decode_calls) == 2
    assert decode_calls[0][0] == "rpa_metadata_schedule"
    assert decode_calls[1][0].startswith("RPAd-")
    assert decode_calls[1][1] == mixed_calls[1][1]
    assert len(decode_calls[1][1]) == (3 if return_lse else 2)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "kv_layout",
    [configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE],
)
@pytest.mark.parametrize("decode_query_size", [1, 4])
def test_tpu_decode_only_matches_mixed_bit_for_bit(kv_layout, decode_query_size):
    """On TPU, decode_only=True matches decode_only=False on pure-decode batches."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU backend")

    rng = np.random.default_rng(0)
    active_seqs = 6
    max_seqs = 8
    page_size = 128
    pages_per_seq = 2
    num_pages = max_seqs * pages_per_seq
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128
    active_tokens = active_seqs * decode_query_size
    max_tokens = active_tokens + 8  # trailing padded token slots

    q = jnp.asarray(
        rng.normal(size=(max_tokens, num_q_heads, head_dim)), dtype=jnp.bfloat16
    )
    k = jnp.asarray(
        rng.normal(size=(max_tokens, num_kv_heads, head_dim)), dtype=jnp.bfloat16
    )
    v = jnp.asarray(
        rng.normal(size=(max_tokens, num_kv_heads, head_dim)), dtype=jnp.bfloat16
    )
    kv_shape = wrapper.get_kv_cache_shape(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        jnp.bfloat16,
        kv_layout=kv_layout,
    )
    kv_init = jnp.asarray(rng.normal(size=kv_shape), dtype=jnp.bfloat16)
    kv_lens_np = np.zeros((max_seqs,), dtype=np.int32)
    kv_lens_np[:active_seqs] = rng.integers(
        decode_query_size + 1, page_size * pages_per_seq, size=(active_seqs,)
    )
    kv_lens = jnp.asarray(kv_lens_np, dtype=jnp.int32)
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)
    q_lens = [decode_query_size] * active_seqs + [0] * (max_seqs - active_seqs)
    cu_q_lens = jnp.asarray([0, *np.cumsum(q_lens).tolist()], dtype=jnp.int32)
    distribution = jnp.array([active_seqs, active_seqs, active_seqs], dtype=jnp.int32)

    out_mixed, kv_mixed = wrapper.ragged_paged_attention(
        q,
        k,
        v,
        kv_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=head_dim**-0.5,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        decode_only=False,
    )
    out_decode, kv_decode = wrapper.ragged_paged_attention(
        q,
        k,
        v,
        kv_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=head_dim**-0.5,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        decode_only=True,
    )
    out_mixed, kv_mixed, out_decode, kv_decode = jax.block_until_ready(
        (out_mixed, kv_mixed, out_decode, kv_decode)
    )
    out_decode_np = np.asarray(out_decode[:active_tokens].astype(jnp.float32))
    assert np.any(out_decode_np != 0.0), "decode output unexpectedly all zeros"
    np.testing.assert_array_equal(
        out_decode_np,
        np.asarray(out_mixed[:active_tokens].astype(jnp.float32)),
    )
    np.testing.assert_array_equal(
        np.asarray(kv_decode.astype(jnp.float32)),
        np.asarray(kv_mixed.astype(jnp.float32)),
    )
