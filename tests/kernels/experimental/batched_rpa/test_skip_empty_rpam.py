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
from jax.extend import core as jax_core

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    schedule,
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
        scoped_jaxpr = scoped[0].params["jaxpr"]
        scoped_eqns = scoped_jaxpr.eqns

        # No unconditional VMEM stores may execute outside the init cond or loop.
        top_prims = {e.primitive.name for e in scoped_eqns}
        assert "swap" not in top_prims, (
            f"{eqn.params['name']} has unconditional store in run_scoped: {top_prims}"
        )
        assert "addupdate" not in top_prims

        conds = [e for e in scoped_eqns if e.primitive.name == "cond"]
        assert len(conds) >= 1

        # Verify the init cond predicate is strictly `actual_steps > 0`, where
        # `actual_steps` is loaded from the scalar SMEM ref `actual_steps_ref[0]`.
        scoped_producer = {e.outvars[0]: e for e in scoped_eqns if len(e.outvars) == 1}
        pred_eqn = scoped_producer[conds[0].invars[0]]
        if pred_eqn.primitive.name == "convert_element_type":
            pred_eqn = scoped_producer[pred_eqn.invars[0]]
        assert pred_eqn.primitive.name == "gt", (
            f"{eqn.params['name']} init cond predicate is {pred_eqn.primitive.name}, "
            "expected 'gt' (actual_steps > 0)"
        )
        lhs, rhs = pred_eqn.invars
        assert isinstance(rhs, jax_core.Literal) and rhs.val == 0
        constvar_to_outer = dict(zip(scoped_jaxpr.constvars, scoped[0].invars))
        inner_producer = {e.outvars[0]: e for e in inner.eqns if len(e.outvars) == 1}
        load_eqn = inner_producer[constvar_to_outer[lhs]]
        assert load_eqn.primitive.name == "get"
        assert load_eqn.invars[0].aval.shape == (1,)

        false_branch, true_branch = conds[0].params["branches"]
        assert not any(e.primitive.name == "swap" for e in false_branch.jaxpr.eqns)
        assert any(e.primitive.name == "swap" for e in true_branch.jaxpr.eqns)


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
def test_tpu_schedule_exact_chunk_boundary_matches_closed_form():
    """Exact max_steps_ub * batch_size boundary (count==0, hbm_offset>0) is exact.

    Uses uniform attention (K = 0) over block-indicator V channels where channel
    `d` holds `float(B)` only in KV block `d % B` (0 elsewhere). The exact output
    is 1.0 on every channel, while dropping any KV block shifts its channels to
    0.0 (a 1.0 delta, 20x larger than `atol`) and repeating a block shifts them
    to ~2B/(B+1).
    """
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU backend")

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
    assert 1 <= kv_blocks_per_seq <= head_dim
    pages_per_kv_block = decode_blocks.bkv_sz // page_size
    pages_per_seq = kv_blocks_per_seq * pages_per_kv_block
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

    # Build uniform-attention inputs (K = 0) with block-indicator V channels.
    # For HEAD_ALONG_SUBLANE with num_kv_heads = 1 and bf16 (kv_packing = 2),
    # kv_cache has shape [num_pages, page_size, 1, 2, head_dim] where slot 0 is K
    # and slot 1 is V.
    kv_shape = wrapper.get_kv_cache_shape(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        jnp.bfloat16,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    kv_init_np = np.zeros(kv_shape, dtype=np.float32)
    dims = np.arange(head_dim)
    for page_pos in range(num_shared_prefix_pages):
        block_idx = page_pos // pages_per_kv_block
        v_pattern = np.where(
            dims % kv_blocks_per_seq == block_idx, float(kv_blocks_per_seq), 0.0
        )
        kv_init_np[page_pos, :, 0, 1, :] = v_pattern

    last_block_idx = kv_blocks_per_seq - 1
    last_v_pattern = np.where(
        dims % kv_blocks_per_seq == last_block_idx, float(kv_blocks_per_seq), 0.0
    )
    # Fill the unique last pages with the last block's V pattern, except the
    # final token slot (page_size - 1) which carries a -1.0 sentinel to verify
    # the new-token KV writeback overwrites it.
    kv_init_np[num_shared_prefix_pages:, : page_size - 1, 0, 1, :] = last_v_pattern
    kv_init_np[num_shared_prefix_pages:, page_size - 1, 0, 1, :] = -1.0

    q = jnp.ones((num_seqs, num_q_heads, head_dim), dtype=jnp.bfloat16)
    k = jnp.zeros((num_seqs, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    v = jnp.broadcast_to(
        jnp.asarray(last_v_pattern, dtype=jnp.bfloat16)[None, None, :],
        (num_seqs, num_kv_heads, head_dim),
    )
    kv_init = jnp.asarray(kv_init_np, dtype=jnp.bfloat16)

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

    atol = 5e-2
    expected_out = np.ones((num_seqs, num_q_heads, head_dim), dtype=np.float32)
    assert np.abs(expected_out).mean() > 10.0 * atol
    np.testing.assert_allclose(
        np.asarray(out.astype(jnp.float32)),
        expected_out,
        atol=atol,
        rtol=0.0,
    )
    # Verify every sequence's newly written token overwrote the -1.0 sentinel.
    written_v = np.asarray(
        kv_out[num_shared_prefix_pages:, page_size - 1, 0, 1, :].astype(jnp.float32)
    )
    expected_v = np.broadcast_to(last_v_pattern, (num_seqs, head_dim))
    np.testing.assert_array_equal(written_v, expected_v)
