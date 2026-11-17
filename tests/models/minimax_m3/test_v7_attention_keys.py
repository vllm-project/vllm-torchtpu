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
"""The v7 MiniMax attention keys reach their module flags, accept the served
values, and leave every flag at its default when unset."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3 import kernel_toggles
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse_r6 as r6
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_dcp_heads
from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

# The cg line's c48 / c64 MSA configs (/tmp/t10diff/cg_envs.txt E48_16 / E64_16).
C48 = {
    "minimax_msa_lse_r6": True,
    "minimax_msa_prefill_page_cache": 512,
    "minimax_msa_prefill_page_cache_tile_pages": 64,
    "minimax_msa_prefill_widths": "2/4/6/8/12/16/20/24",
    "minimax_msa_prefill_pages_per_step": 32,
    "minimax_msa_prefill_adaptive": "30:8:64",
    "minimax_msa_page_cache_sort": "packed",
}
C64 = {
    "minimax_msa_lse_r6": True,
    "minimax_msa_prefill_page_cache": 512,
    "minimax_msa_prefill_page_cache_tile_pages": 192,
    "minimax_msa_prefill_block_q": 32,
    "minimax_msa_prefill_widths": "2/4/6/8/12",
    "minimax_msa_decode_one_call": True,
    "minimax_msa_decode_work_list_kernel": True,
    "minimax_dcp_hp_prefill_config": "16:16::1024",
    "minimax_msa_page_cache_sort": "packed",
    "minimax_dcp_heads_attend": True,
}


def _defaults():
    return (
        r6.PREFILL_ADAPTIVE,
        r6.PAGE_CACHE_SORT,
        r6.PAGE_CACHE_WIDTHS,
        msa_dcp_heads.HP_PREFILL_CONFIG,
        msa_ops.MERGE_A2A_MAX_TOKENS,
        msa_ops.QCAND_GATHER_MAX_TOKENS,
    )


def test_unset_keys_keep_the_defaults():
    kernel_toggles.apply({})
    assert _defaults() == (None, "", (), None, 0, 0)
    assert not msa_ops._packed_merge(8) and not msa_ops._packed_gather(8)


def test_cg_c48_config():
    try:
        kernel_toggles.apply(C48)
        assert r6.PREFILL_STEP_WIDTHS == (2, 4, 6, 8, 12, 16, 20, 24)
        assert r6.PREFILL_PAGES_PER_STEP == 32
        assert r6.PAGE_CACHE_TILE_PAGES == 64
        assert r6.PREFILL_ADAPTIVE == (30, 8, 64)
        assert r6.PAGE_CACHE_SORT == "packed"
    finally:
        kernel_toggles.apply({})


def test_cg_c64_config():
    try:
        kernel_toggles.apply(C64)
        assert r6.PREFILL_BLOCK_Q == 32
        assert r6.PAGE_CACHE_TILE_PAGES == 192
        assert r6.PREFILL_STEP_WIDTHS == (2, 4, 6, 8, 12)
        assert msa_dcp_heads.HP_PREFILL_CONFIG == dict(
            block_q=16, pages_per_step=16, page_cache_slots=1024
        )
        assert msa_ops.DCP_HEADS_ATTEND is True
    finally:
        kernel_toggles.apply({})
    assert _defaults() == (None, "", (), None, 0, 0)


def test_packed_collective_keys():
    try:
        kernel_toggles.apply(
            {
                "minimax_dcp_merge_a2a_max_tokens": 512,
                "minimax_dcp_qcand_gather_max_tokens": 512,
            }
        )
        assert msa_ops._packed_merge(512) and not msa_ops._packed_merge(513)
        assert msa_ops._packed_gather(1) and not msa_ops._packed_gather(1024)
        with pytest.raises(ValueError, match="minimax_dcp_merge_a2a_max_tokens"):
            kernel_toggles.apply({"minimax_dcp_merge_a2a_max_tokens": -1})
    finally:
        kernel_toggles.apply({})
    assert msa_ops.MERGE_A2A_MAX_TOKENS == msa_ops.QCAND_GATHER_MAX_TOKENS == 0


def test_bad_values_are_errors():
    try:
        with pytest.raises(ValueError, match="minimax_msa_page_cache_sort"):
            kernel_toggles.apply({"minimax_msa_page_cache_sort": "fast"})
        with pytest.raises(ValueError, match="minimax_dcp_hp_prefill_config"):
            kernel_toggles.apply({"minimax_dcp_hp_prefill_config": "16:16"})
        with pytest.raises(ValueError, match="minimax_msa_page_cache_widths"):
            kernel_toggles.apply({"minimax_msa_page_cache_widths": "64/x"})
    finally:
        kernel_toggles.apply({})


# --- the c64 prefill tile config runs (32-token tiles, 192 cached pages) ---

S, D, HG, TOPK, PAGES = 128, 128, 16, 16, 64


def _inputs(seed, t=128):
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

    rng = np.random.default_rng(seed)
    blocks = np.stack(
        [rng.choice(PAGES - 1, size=TOPK, replace=False) for _ in range(t)]
    )[:, None]
    scores = -np.sort(-rng.standard_normal((t, 1, TOPK)), axis=-1)
    cand = np.concatenate([scores, blocks.astype(np.float64)], axis=-1)
    cand = msa_dcp.to_slots(jnp.asarray(cand.reshape(t, -1), jnp.float32), TOPK)[None]
    q = jnp.asarray(rng.standard_normal((t, HG, D)) * 0.5, jnp.bfloat16)
    kv = msa_fp4.pack_rows(
        jnp.asarray(rng.integers(0, 16, size=(PAGES, S, 2 * D)), jnp.uint8)
    )
    sc = jnp.asarray(
        rng.uniform(0.01, 0.03, size=(PAGES, msa_fp4.scale_rows(1), S)), jnp.float32
    )
    return (
        q,
        kv,
        cand,
        jnp.asarray([PAGES * S], jnp.int32),
        jnp.asarray(rng.permutation(PAGES)[None], jnp.int32),
        jnp.asarray([0, t], jnp.int32),
        sc,
    )


def _attend(args):
    q, kv, cand, seq_lens, block_tables, cu_q_lens, sc = args
    out, lse = msa_dcp.msa_attend_dcp(
        q, kv, cand, seq_lens, block_tables, cu_q_lens,
        sm_scale=1 / np.sqrt(D), topk=TOPK, rank=0, dcp=1, scale_cache=sc,
        interpret=True,
    )
    return np.asarray(out.astype(jnp.float32)), np.asarray(lse)


def test_c64_prefill_tiles_match_the_default_config(monkeypatch):
    """32-token tiles reorder the sums (not bitwise): same values to bf16
    output rounding; the packed page cache table changes nothing."""
    monkeypatch.setattr(msa_dcp, "_MIN_CANDIDATE_TOKENS", 64)
    monkeypatch.setattr(msa_dcp, "DCP_SMALL_BATCH_TOKENS", 64)
    monkeypatch.setattr(msa_attention_lse, "R6_KERNEL", True)
    args = _inputs(0)
    jax.clear_caches()
    want_out, want_lse = _attend(args)
    monkeypatch.setattr(r6, "PREFILL_PAGE_CACHE_SLOTS", 512)
    monkeypatch.setattr(r6, "PAGE_CACHE_TILE_PAGES", 192)
    monkeypatch.setattr(r6, "PREFILL_BLOCK_Q", 32)
    monkeypatch.setattr(r6, "PREFILL_STEP_WIDTHS", (2, 4, 6, 8, 12))
    monkeypatch.setattr(r6, "PAGE_CACHE_SORT", "packed")
    jax.clear_caches()
    out, lse = _attend(args)
    np.testing.assert_allclose(lse, want_lse, rtol=0, atol=1e-4)
    np.testing.assert_allclose(out, want_out, rtol=2e-2, atol=2e-2)
    monkeypatch.setattr(r6, "PAGE_CACHE_SORT", "")
    jax.clear_caches()
    out2, lse2 = _attend(args)
    np.testing.assert_array_equal(out2, out)
    np.testing.assert_array_equal(lse2, lse)
