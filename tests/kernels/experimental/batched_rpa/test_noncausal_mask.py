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
"""CPU tests for batched RPA's non-causal QK mask and KV-block bound.

The full kernel needs a TPU; these cover the pure-JAX pieces that decide which
KV positions each query sees.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, kernel, schedule

NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
BQ_SZ = 16
BKV_SZ = 32


def _make_cfgs(
    causal: bool,
    *,
    mode: configs.RpaCase = configs.RpaCase.DECODE,
    sliding_window: int | None = None,
    cp_group_size: int | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
) -> configs.RpaConfigs:
    return configs.RpaConfigs(
        block=configs.BlockSizes(
            bq_sz=BQ_SZ, bq_c_sz=BQ_SZ, bkv_sz=BKV_SZ, batch_size=2
        ),
        model=configs.ModelConfigs(
            num_q_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=128,
            mask_value=-1e9,
            sliding_window=sliding_window,
            causal=causal,
        ),
        serve=configs.ServingConfigs(
            num_seqs=2,
            page_size=16,
            total_q_tokens=2 * BQ_SZ,
            num_page_indices=64,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.float32,
            decode_query_size=BQ_SZ,
            cp_group_size=cp_group_size,
            attention_scope=attention_scope,
        ),
        mode=mode,
        vmem_limit_bytes=64 * 1024 * 1024,
    )


# (q_base, k_id, total_kv_len) per batch element: a full tile, then the last
# tile of a sequence whose KV ends mid-tile.
_SEQS = [(48, 32, 64), (100, 96, 107)]


def _expected_mask(cfgs, bq_start, rows, causal):
    heads = cfgs.aligned_num_q_heads_per_kv_head
    out = []
    for q_base, k_id, total in _SEQS:
        q_pos = q_base + bq_start + np.arange(rows)[:, None] // heads
        k_pos = k_id + np.arange(BKV_SZ)[None, :]
        m = (q_pos >= k_pos) if causal else (k_pos < total)
        out.append(np.broadcast_to(m, (NUM_KV_HEADS, rows, BKV_SZ)))
    return out


def _step_meta(causal):
    return kernel.ComputeMetadata(
        causal_offset=[jnp.int32(k_id - q_base) for q_base, k_id, _ in _SEQS],
        kv_len_in_tile=(
            None if causal else [jnp.int32(total - k_id) for _, k_id, total in _SEQS]
        ),
    )


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize("bq_start", [0, 8])
def test_generate_mask(causal, bq_start):
    cfgs = _make_cfgs(causal)
    rows = (BQ_SZ - bq_start) * cfgs.aligned_num_q_heads_per_kv_head
    masks = kernel.generate_mask(
        (2, NUM_KV_HEADS, rows, BKV_SZ),
        bq_start=bq_start,
        step_meta=_step_meta(causal),
        cfgs=cfgs,
    )
    for actual, expected in zip(masks, _expected_mask(cfgs, bq_start, rows, causal)):
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_noncausal_mask_drops_stale_tile_tail():
    # The second sequence has 11 valid KV tokens in its last tile; every query,
    # including the earliest, must see exactly those and nothing past them.
    cfgs = _make_cfgs(causal=False)
    rows = BQ_SZ * cfgs.aligned_num_q_heads_per_kv_head
    mask = np.asarray(
        kernel.generate_mask(
            (2, NUM_KV_HEADS, rows, BKV_SZ),
            bq_start=0,
            step_meta=_step_meta(causal=False),
            cfgs=cfgs,
        )[1]
    )
    assert mask[:, :, :11].all()
    assert not mask[:, :, 11:].any()


@pytest.mark.parametrize(
    "q_offset,q_idx,q_sz_task,num_k",
    [
        (0, 0, 16, 8),  # prefill, first Q tile
        (64, 1, 16, 8),  # prefill, mid-sequence Q tile
        (240, 0, 16, 8),  # decode: queries end the sequence
        (100, 0, 7, 4),  # partial Q tile
    ],
)
def test_end_k_idx(q_offset, q_idx, q_sz_task, num_k):
    kw = dict(q_offset=q_offset, q_idx=q_idx, q_sz_task=q_sz_task, num_k=num_k)
    causal = int(schedule.get_end_k_idx(**kw, cfgs=_make_cfgs(True)))
    expected_causal = min(
        num_k, (q_offset + q_idx * BQ_SZ + q_sz_task - 1) // BKV_SZ + 1
    )
    assert causal == expected_causal
    assert int(schedule.get_end_k_idx(**kw, cfgs=_make_cfgs(False))) == num_k


def test_decode_end_k_idx_unchanged_by_noncausal():
    # Decode queries are the last tokens of their sequence, so the causal bound
    # already reaches the last KV block: non-causal schedules no extra steps.
    total_kv_len, q_len = 250, 16
    num_k = -(-total_kv_len // BKV_SZ)
    kw = dict(q_offset=total_kv_len - q_len, q_idx=0, q_sz_task=q_len, num_k=num_k)
    assert int(schedule.get_end_k_idx(**kw, cfgs=_make_cfgs(True))) == num_k
    assert int(schedule.get_end_k_idx(**kw, cfgs=_make_cfgs(False))) == num_k


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sliding_window=128),
        dict(cp_group_size=2),
        dict(attention_scope=configs.AttentionScope.CACHE_ONLY),
        dict(attention_scope=configs.AttentionScope.NEW_TOKENS_ONLY),
    ],
    ids=["sliding_window", "cp", "cache_only", "new_tokens_only"],
)
def test_noncausal_rejects_unsupported_features(kwargs):
    with pytest.raises(NotImplementedError):
        _make_cfgs(causal=False, **kwargs).validate_non_causal()
    _make_cfgs(causal=True, **kwargs).validate_non_causal()


def test_schedule_multiplier_doubles_only_for_noncausal_mixed():
    base = _make_cfgs(True).max_schedule_size_multiplier
    assert _make_cfgs(False).max_schedule_size_multiplier == base
    mixed = configs.RpaCase.MIXED
    assert _make_cfgs(True, mode=mixed).max_schedule_size_multiplier == base
    assert _make_cfgs(False, mode=mixed).max_schedule_size_multiplier == 2 * base


def test_kernel_name_marks_noncausal():
    causal = _make_cfgs(True)
    noncausal = dataclasses.replace(
        causal, model=dataclasses.replace(causal.model, causal=False)
    )
    assert not kernel.get_kernel_name(causal).endswith("-nc")
    assert kernel.get_kernel_name(noncausal).endswith("-nc")
