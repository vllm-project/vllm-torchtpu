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
"""Tests for skipping dead VMEM init and schedule flush on empty batched RPA steps."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    schedule,
    utils,
    wrapper,
)

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


def _make_dummy_inputs(
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
):
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


def _to_seq_along_lane(kv_cache: jax.Array, num_kv_heads: int) -> jax.Array:
    num_pages, page_size, _, kv_packing, head_dim = kv_cache.shape
    num_kv_heads_x2 = utils.align_to(num_kv_heads * 2, kv_packing)
    return kv_cache.reshape(
        num_pages,
        page_size,
        num_kv_heads_x2,
        head_dim // kv_packing,
        kv_packing,
    ).transpose(0, 2, 3, 4, 1)


def _to_head_along_sublane(kv_cache: jax.Array, num_kv_heads: int) -> jax.Array:
    num_pages, _, head_dim_packed, kv_packing, page_size = kv_cache.shape
    num_kv_heads_x2 = utils.align_to(num_kv_heads * 2, kv_packing)
    return kv_cache.transpose(0, 4, 1, 2, 3).reshape(
        num_pages,
        page_size,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        head_dim_packed * kv_packing,
    )


def test_rpa_kernel_guards_vmem_init_on_actual_steps():
    """VMEM scratch and Q/KV initialization must be gated by actual_steps > 0."""
    inputs = _make_dummy_inputs()
    with mock.patch.object(pltpu, "get_tpu_info", return_value=_TPU_7X_INFO):
        closed = jax.make_jaxpr(wrapper.ragged_paged_attention)(*inputs)

    rpa_eqns = [
        eqn
        for eqn in _find_pallas_eqns(closed.jaxpr)
        if eqn.params.get("name", "").startswith(("RPAd-", "RPAm-"))
    ]
    assert len(rpa_eqns) == 2

    for eqn in rpa_eqns:
        inner = eqn.params["jaxpr"]
        scoped = [e for e in inner.eqns if e.primitive.name == "run_scoped"]
        assert len(scoped) == 1
        scoped_eqns = scoped[0].params["jaxpr"].eqns

        # No unconditional VMEM stores/writes may execute outside the init cond
        # or the schedule loop.
        top_prims = {e.primitive.name for e in scoped_eqns}
        assert "swap" not in top_prims, (
            f"{eqn.params['name']} has unconditional store in run_scoped: {top_prims}"
        )
        assert "addupdate" not in top_prims

        conds = [e for e in scoped_eqns if e.primitive.name == "cond"]
        assert len(conds) >= 1
        false_branch, true_branch = conds[0].params["branches"]
        assert not any(e.primitive.name == "swap" for e in false_branch.jaxpr.eqns)
        # All 5 VMEM init stores (m, l, acc, q_ref_flat, kv_ref_flat) live in the
        # `actual_steps > 0` branch.
        true_stores = sum(
            1 for e in true_branch.jaxpr.eqns if e.primitive.name == "swap"
        )
        assert true_stores == 5


def test_rpa_metadata_schedule_empty_count_writes_only_actual_steps():
    """When count == 0, schedule flush must only DMA actual_steps to HBM."""
    inputs = _make_dummy_inputs()
    with mock.patch.object(pltpu, "get_tpu_info", return_value=_TPU_7X_INFO):
        closed = jax.make_jaxpr(wrapper.ragged_paged_attention)(*inputs)

    sched_eqns = [
        eqn
        for eqn in _find_pallas_eqns(closed.jaxpr)
        if eqn.params.get("name") == "rpa_metadata_schedule"
    ]
    assert len(sched_eqns) == 2

    for eqn in sched_eqns:
        conds = [e for e in eqn.params["jaxpr"].eqns if e.primitive.name == "cond"]
        assert len(conds) >= 1
        false_branch, true_branch = conds[-1].params["branches"]
        false_prims = [e.primitive.name for e in false_branch.jaxpr.eqns]
        true_prims = [e.primitive.name for e in true_branch.jaxpr.eqns]
        assert "while" not in false_prims
        assert false_prims.count("dma_start") == 1
        assert false_prims.count("dma_wait") == 1
        assert true_prims.count("dma_start") > 1
        assert true_prims.count("dma_wait") == true_prims.count("dma_start")


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "kv_layout",
    [configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE],
)
@pytest.mark.parametrize("decode_query_size", [1, 4])
def test_tpu_pure_decode_matches_reference(kv_layout, decode_query_size):
    """Pure-decode steps (empty RPAm) match reference with padding and spec-decode."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU backend")

    from tests.kernels.experimental.batched_rpa.correctness_test import (
        ref_ragged_paged_attention,
    )

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
    ref_kv_shape = wrapper.get_kv_cache_shape(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        jnp.bfloat16,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    kv_ref_init = jnp.asarray(rng.normal(size=ref_kv_shape), dtype=jnp.bfloat16)
    kv_rpa_init = (
        _to_seq_along_lane(kv_ref_init, num_kv_heads)
        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE
        else kv_ref_init
    )

    kv_lens_np = np.zeros((max_seqs,), dtype=np.int32)
    kv_lens_np[:active_seqs] = rng.integers(
        decode_query_size + 1, page_size * pages_per_seq, size=(active_seqs,)
    )
    kv_lens = jnp.asarray(kv_lens_np, dtype=jnp.int32)
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)
    q_lens = [decode_query_size] * active_seqs + [0] * (max_seqs - active_seqs)
    cu_q_lens = jnp.asarray([0, *np.cumsum(q_lens).tolist()], dtype=jnp.int32)
    distribution = jnp.array([active_seqs, active_seqs, active_seqs], dtype=jnp.int32)
    sm_scale = head_dim**-0.5

    ref_out, ref_kv = ref_ragged_paged_attention(
        q,
        k,
        v,
        kv_ref_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
    )
    out, kv_out = wrapper.ragged_paged_attention(
        q,
        k,
        v,
        kv_rpa_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
    )
    out, kv_out = jax.block_until_ready((out, kv_out))
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        kv_out = _to_head_along_sublane(kv_out, num_kv_heads)

    active_out = np.asarray(out[:active_tokens].astype(jnp.float32))
    assert np.any(active_out != 0.0), "decode output unexpectedly all zeros"
    np.testing.assert_allclose(
        active_out,
        np.asarray(ref_out.astype(jnp.float32)),
        atol=5e-2,
        rtol=5e-2,
    )
    np.testing.assert_array_equal(
        np.asarray(kv_out.astype(jnp.float32)),
        np.asarray(ref_kv.astype(jnp.float32)),
    )


@pytest.mark.tpu_test
def test_tpu_schedule_exact_chunk_boundary_matches_reference():
    """Exact max_steps_ub * batch_size boundary (count==0, hbm_offset>0) matches ref."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU backend")

    from tests.kernels.experimental.batched_rpa.correctness_test import (
        ref_ragged_paged_attention,
    )

    num_seqs = 128
    page_size = 128
    num_q_heads = 8
    num_kv_heads = 1
    head_dim = 128
    sm_scale = head_dim**-0.5
    decode_blocks = configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=2048, batch_size=8)

    # Probe max_steps_ub on the live TPU so scheduled entries land exactly on
    # max_steps_ub * batch_size, leaving loop_carry.count == 0 and hbm_offset > 0.
    probe_cfgs = configs.RpaConfigs(
        block=decode_blocks,
        model=configs.ModelConfigs(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sm_scale=sm_scale,
            mask_value=float(jnp.finfo(jnp.bfloat16).min),
        ),
        serve=configs.ServingConfigs(
            num_seqs=num_seqs,
            page_size=page_size,
            total_q_tokens=num_seqs,
            num_page_indices=num_seqs * 16,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16,
        ),
        vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes),
        mode=configs.RpaCase.DECODE,
    )
    target_entries = probe_cfgs.max_steps_ub * decode_blocks.batch_size
    assert target_entries % num_seqs == 0
    kv_blocks_per_seq = target_entries // num_seqs
    pages_per_seq = kv_blocks_per_seq * (decode_blocks.bkv_sz // page_size)
    # Recompute with the exact num_page_indices to confirm max_steps_ub is unchanged.
    exact_cfgs = configs.RpaConfigs(
        block=decode_blocks,
        model=probe_cfgs.model,
        serve=configs.ServingConfigs(
            num_seqs=num_seqs,
            page_size=page_size,
            total_q_tokens=num_seqs,
            num_page_indices=num_seqs * pages_per_seq,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16,
        ),
        vmem_limit_bytes=probe_cfgs.vmem_limit_bytes,
        mode=configs.RpaCase.DECODE,
    )
    assert exact_cfgs.max_steps_ub == probe_cfgs.max_steps_ub

    # Share read-only prefix pages across sequences, with a unique last page per
    # sequence so 1-token KV writebacks never alias across sequences.
    num_shared_prefix_pages = pages_per_seq - 1
    num_pages = num_shared_prefix_pages + num_seqs
    page_table = np.empty((num_seqs, pages_per_seq), dtype=np.int32)
    page_table[:, :num_shared_prefix_pages] = np.arange(
        num_shared_prefix_pages, dtype=np.int32
    )
    page_table[:, -1] = num_shared_prefix_pages + np.arange(num_seqs, dtype=np.int32)
    page_indices = jnp.asarray(page_table.ravel(), dtype=jnp.int32)

    kv_len_per_seq = kv_blocks_per_seq * decode_blocks.bkv_sz
    kv_lens = jnp.full((num_seqs,), kv_len_per_seq, dtype=jnp.int32)
    cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)
    distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)

    # Confirm the schedule hits the exact chunk boundary (actual_steps == max_steps_ub).
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    kv_cache_lens = kv_lens - q_lens
    sched = schedule.generate_rpa_metadata(
        cu_q_lens,
        kv_cache_lens,
        kv_cache_lens,
        q_lens,
        distribution,
        page_indices,
        cfgs=exact_cfgs,
    )
    assert int(jax.block_until_ready(sched.actual_steps)[0]) == exact_cfgs.max_steps_ub

    rng = np.random.default_rng(1)
    q = jnp.asarray(
        rng.normal(size=(num_seqs, num_q_heads, head_dim)), dtype=jnp.bfloat16
    )
    k = jnp.asarray(
        rng.normal(size=(num_seqs, num_kv_heads, head_dim)), dtype=jnp.bfloat16
    )
    v = jnp.asarray(
        rng.normal(size=(num_seqs, num_kv_heads, head_dim)), dtype=jnp.bfloat16
    )
    kv_shape = wrapper.get_kv_cache_shape(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        jnp.bfloat16,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    kv_init = jnp.asarray(rng.normal(size=kv_shape), dtype=jnp.bfloat16)

    ref_out, ref_kv = ref_ragged_paged_attention(
        q,
        k,
        v,
        kv_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
    )
    out, kv_out = wrapper.ragged_paged_attention(
        q,
        k,
        v,
        kv_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        decode_block_sizes=decode_blocks,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    out, kv_out = jax.block_until_ready((out, kv_out))

    out_np = np.asarray(out.astype(jnp.float32))
    assert np.any(out_np != 0.0), "boundary output unexpectedly all zeros"
    np.testing.assert_allclose(
        out_np,
        np.asarray(ref_out.astype(jnp.float32)),
        atol=5e-2,
        rtol=5e-2,
    )
    np.testing.assert_array_equal(
        np.asarray(kv_out.astype(jnp.float32)),
        np.asarray(ref_kv.astype(jnp.float32)),
    )
