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
"""Tests for batched RPA empty-launch skipping, output fence, and interior tiles."""

import ast
import functools
import inspect
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    flash_attention,
    kernel,
    schedule,
    wrapper,
)

_DECODE_BLOCKS = configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=128, batch_size=4)
_PREFILL_BLOCKS = configs.BlockSizes(bq_sz=64, bq_c_sz=32, bkv_sz=64, batch_size=2)
_NUM_Q_HEADS = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 128
_PAGE_SIZE = 16
_PAGES_PER_SEQ = 16
_MAX_SEQS = 8
_VMEM_BUDGET = 1 << 26


def _make_rpa_configs(
    mode: configs.RpaCase,
    *,
    sliding_window: int | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
    skip_empty_launch: bool = True,
    fence_output: bool = True,
    unmasked_interior: bool = True,
    total_q_tokens: int = 64,
) -> configs.RpaConfigs:
    return configs.RpaConfigs(
        block=_DECODE_BLOCKS if mode == configs.RpaCase.DECODE else _PREFILL_BLOCKS,
        model=configs.ModelConfigs(
            num_q_heads=_NUM_Q_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            sliding_window=sliding_window,
            sm_scale=1.0 / (_HEAD_DIM**0.5),
            soft_cap=None,
            mask_value=float(jnp.finfo(jnp.bfloat16).min),
        ),
        serve=configs.ServingConfigs(
            num_seqs=_MAX_SEQS,
            page_size=_PAGE_SIZE,
            total_q_tokens=total_q_tokens,
            num_page_indices=_MAX_SEQS * _PAGES_PER_SEQ,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16,
            attention_scope=attention_scope,
            skip_empty_launch=skip_empty_launch,
            fence_output=fence_output,
            unmasked_interior=unmasked_interior,
        ),
        mode=mode,
        vmem_limit_bytes=_VMEM_BUDGET,
    )


@pytest.mark.parametrize(
    "empty_mode,distribution,q_len_per_seq",
    [
        (configs.RpaCase.MIXED, [4, 4, 4], 1),
        (configs.RpaCase.DECODE, [0, 0, 2], 16),
    ],
)
def test_schedule_skip_empty_launch_interpret(
    empty_mode: configs.RpaCase,
    distribution: list[int],
    q_len_per_seq: int,
):
    n_active = distribution[2]
    q_lens = np.array([q_len_per_seq] * n_active + [0] * (_MAX_SEQS - n_active))
    interpret_params = pltpu.InterpretParams()
    tpu7x = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=tpu7x)

    with (
        jax.default_device(jax.devices("cpu")[0]),
        jax.sharding.use_abstract_mesh(mesh),
    ):
        cu_q_lens = jnp.asarray(
            np.concatenate([[0], np.cumsum(q_lens)]), dtype=jnp.int32
        )
        kv_cache_lens = jnp.asarray(
            [32] * n_active + [0] * (_MAX_SEQS - n_active), dtype=jnp.int32
        )
        kv_new_lens = jnp.asarray(q_lens, dtype=jnp.int32)
        page_indices = jnp.arange(_MAX_SEQS * _PAGES_PER_SEQ, dtype=jnp.int32)
        dist = jnp.asarray(distribution, dtype=jnp.int32)

        sched_skip = schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_cache_lens,
            kv_cache_lens,
            kv_new_lens,
            dist,
            page_indices,
            _make_rpa_configs(empty_mode, skip_empty_launch=True),
            interpret=interpret_params,
        )
        sched_full = schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_cache_lens,
            kv_cache_lens,
            kv_new_lens,
            dist,
            page_indices,
            _make_rpa_configs(empty_mode, skip_empty_launch=False),
            interpret=interpret_params,
        )
        assert int(sched_skip.actual_steps[0]) == 0
        assert int(sched_full.actual_steps[0]) == 0

        active_mode = (
            configs.RpaCase.DECODE
            if empty_mode == configs.RpaCase.MIXED
            else configs.RpaCase.MIXED
        )
        active_skip = schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_cache_lens,
            kv_cache_lens,
            kv_new_lens,
            dist,
            page_indices,
            _make_rpa_configs(active_mode, skip_empty_launch=True),
            interpret=interpret_params,
        )
        active_full = schedule.generate_rpa_metadata(
            cu_q_lens,
            kv_cache_lens,
            kv_cache_lens,
            kv_new_lens,
            dist,
            page_indices,
            _make_rpa_configs(active_mode, skip_empty_launch=False),
            interpret=interpret_params,
        )
        assert int(active_skip.actual_steps[0]) > 0
        for leaf_a, leaf_b in zip(
            jax.tree_util.tree_leaves(active_skip),
            jax.tree_util.tree_leaves(active_full),
        ):
            np.testing.assert_array_equal(np.asarray(leaf_a), np.asarray(leaf_b))


def test_step_needs_mask_and_hoisted_interior_branch():
    # Ensure flash_attention_qk_softmax contains no value-carrying lax.cond
    # barrier inside the unrolled bq_c loop, and _attend_and_store keeps exact
    # software-pipelined PV + output store structure.
    qk_tree = ast.parse(
        textwrap.dedent(inspect.getsource(flash_attention.flash_attention_qk_softmax))
    )
    cond_calls = [
        node
        for node in ast.walk(qk_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cond"
    ]
    assert len(cond_calls) == 0

    attend_tree = ast.parse(
        textwrap.dedent(inspect.getsource(kernel._attend_and_store))
    )
    pv_calls = [
        node
        for node in ast.walk(attend_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "flash_attention_pv"
    ]
    store_calls = [
        node
        for node in ast.walk(attend_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "calculate_and_store_out"
    ]
    # Two calls in _attend_and_store: software-pipelined chunk loop + epilogue.
    assert len(pv_calls) == 2
    assert len(store_calls) == 1

    cfgs = _make_rpa_configs(configs.RpaCase.MIXED, sliding_window=256)
    s_idx = schedule.SmemWrapper(
        data=jnp.asarray([0, 0, 0, -1], dtype=jnp.int32), shape=(2, 2)
    )

    class _FakeSchedule:
        def __init__(self, s):
            self.s_idx = s

    fake_sched = _FakeSchedule(s_idx)
    # bkv_sz=64, bq_sz=64, sliding_window=256:
    # interior requires offset <= -63 and offset > 63 - 256 = -193.
    interior_meta = kernel.ComputeMetadata(
        causal_offset=[jnp.int32(-64), jnp.int32(-128)]
    )
    assert not bool(
        kernel.step_needs_mask(
            jnp.int32(0),
            fake_sched,  # type: ignore[arg-type]
            interior_meta,
            cfgs=cfgs,
        )
    )

    # Diagonal step: offset == 0 > -63 -> needs mask.
    diag_meta = kernel.ComputeMetadata(causal_offset=[jnp.int32(0), jnp.int32(-128)])
    assert bool(
        kernel.step_needs_mask(
            jnp.int32(0),
            fake_sched,  # type: ignore[arg-type]
            diag_meta,
            cfgs=cfgs,
        )
    )

    # Sliding-window boundary: offset == -193 <= 63 - 256 -> needs mask.
    sw_edge_meta = kernel.ComputeMetadata(
        causal_offset=[jnp.int32(-193), jnp.int32(-128)]
    )
    assert bool(
        kernel.step_needs_mask(
            jnp.int32(0),
            fake_sched,  # type: ignore[arg-type]
            sw_edge_meta,
            cfgs=cfgs,
        )
    )

    # Padding lane (step=1 has s_idx == -1 on lane 1) -> needs mask.
    assert bool(
        kernel.step_needs_mask(
            jnp.int32(1),
            fake_sched,  # type: ignore[arg-type]
            interior_meta,
            cfgs=cfgs,
        )
    )


def test_flash_attention_qk_softmax_conditional_mask_parity():
    cfgs = _make_rpa_configs(configs.RpaCase.MIXED)
    key = jax.random.key(0)
    kq, kk = jax.random.split(key)
    tq = cfgs.bq_c_sz * cfgs.aligned_num_q_heads_per_kv_head
    q = jax.random.normal(
        kq, (cfgs.batch_size, _NUM_KV_HEADS, tq, _HEAD_DIM), dtype=jnp.bfloat16
    )
    k = jax.random.normal(
        kk,
        (cfgs.batch_size, _NUM_KV_HEADS, cfgs.bkv_sz, _HEAD_DIM),
        dtype=jnp.bfloat16,
    )
    m_prev = jnp.full((_NUM_KV_HEADS, tq, 128), -jnp.inf, dtype=jnp.bfloat16)
    l_prev = jnp.zeros((_NUM_KV_HEADS, tq, 128), dtype=jnp.bfloat16)
    is_last_k = schedule.SmemWrapper(
        data=jnp.asarray([0, 0], dtype=jnp.int32), shape=(1, cfgs.batch_size)
    )

    interior_meta = kernel.ComputeMetadata(
        causal_offset=[jnp.int32(-128), jnp.int32(-128)]
    )
    mask_fn = functools.partial(
        kernel.generate_mask,
        shape=(cfgs.batch_size, _NUM_KV_HEADS, tq, cfgs.bkv_sz),
        bq_start=0,
        step_meta=interior_meta,
        cfgs=cfgs,
    )
    out_masked = jax.jit(
        lambda: flash_attention.flash_attention_qk_softmax(
            jnp.int32(0),
            q,
            k,
            m_prev,
            l_prev,
            is_last_k,
            custom_mask=mask_fn(),
            cfgs=cfgs,
            bq_start=0,
        )
    )()
    out_unmasked = jax.jit(
        lambda: flash_attention.flash_attention_qk_softmax(
            jnp.int32(0),
            q,
            k,
            m_prev,
            l_prev,
            is_last_k,
            custom_mask=None,
            cfgs=cfgs,
            bq_start=0,
        )
    )()
    np.testing.assert_allclose(
        np.asarray(out_masked[0], dtype=np.float32),
        np.asarray(out_unmasked[0], dtype=np.float32),
        rtol=2e-2,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(out_masked[3], dtype=np.float32),
        np.asarray(out_unmasked[3], dtype=np.float32),
        rtol=2e-2,
        atol=2e-3,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "distribution,q_lens,sliding_window",
    [
        ([4, 4, 4], [1, 1, 1, 1, 0, 0, 0, 0], None),
        ([0, 0, 2], [96, 160, 0, 0, 0, 0, 0, 0], None),
        ([1, 1, 3], [1, 192, 160, 0, 0, 0, 0, 0], 128),
    ],
)
def test_tpu_batched_rpa_optimizations_parity(
    distribution: list[int],
    q_lens: list[int],
    sliding_window: int | None,
):
    total_tokens = max(sum(q_lens), 8)
    key = jax.random.key(42)
    kq, kk, kv, kc = jax.random.split(key, 4)
    queries = jax.random.normal(
        kq, (total_tokens, _NUM_Q_HEADS, _HEAD_DIM), dtype=jnp.bfloat16
    )
    keys = jax.random.normal(
        kk, (total_tokens, _NUM_KV_HEADS, _HEAD_DIM), dtype=jnp.bfloat16
    )
    values = jax.random.normal(
        kv, (total_tokens, _NUM_KV_HEADS, _HEAD_DIM), dtype=jnp.bfloat16
    )
    cache_shape = wrapper.get_kv_cache_shape(
        _MAX_SEQS * _PAGES_PER_SEQ,
        _PAGE_SIZE,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        jnp.bfloat16,
    )
    kv_cache_init = jax.random.normal(kc, cache_shape, dtype=jnp.bfloat16)

    q_arr = np.asarray(q_lens, dtype=np.int32)
    cu_q_lens = jnp.asarray(np.concatenate([[0], np.cumsum(q_arr)]), dtype=jnp.int32)
    kv_lens = jnp.asarray(
        [64 + q if q > 0 else 0 for q in q_lens],
        dtype=jnp.int32,
    )
    page_indices = jnp.arange(_MAX_SEQS * _PAGES_PER_SEQ, dtype=jnp.int32)
    dist = jnp.asarray(distribution, dtype=jnp.int32)

    out_ref, cache_ref = wrapper.ragged_paged_attention(
        queries,
        keys,
        values,
        kv_cache_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        dist,
        sliding_window=sliding_window,
        decode_block_sizes=_DECODE_BLOCKS,
        prefill_block_sizes=_PREFILL_BLOCKS,
        skip_empty_launch=False,
        fence_output=False,
        unmasked_interior=False,
    )
    out_opt, cache_opt = wrapper.ragged_paged_attention(
        queries,
        keys,
        values,
        kv_cache_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        dist,
        sliding_window=sliding_window,
        decode_block_sizes=_DECODE_BLOCKS,
        prefill_block_sizes=_PREFILL_BLOCKS,
        skip_empty_launch=True,
        fence_output=True,
        unmasked_interior=True,
    )
    valid_tokens = sum(q_lens)
    np.testing.assert_allclose(
        np.asarray(out_opt[:valid_tokens], dtype=np.float32),
        np.asarray(out_ref[:valid_tokens], dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_array_equal(np.asarray(cache_opt), np.asarray(cache_ref))
