# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for DeepSeek-V4.1 DSpark K=5 port and 1+K Engram prefetch."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.deepseek_v41.dspark import DSparkDeepseekV41ForCausalLM
from vllm_torchtpu.models.vllm.deepseek_v41.engram import TpuNgramHashState
from vllm_torchtpu.models.vllm.deepseek_v41.model import _gather_aux_hidden_states
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncPreResults
from vllm_torchtpu.spec_decode.dspark import DSparkProposer


class _FakeHFConfig:
    def __init__(self, d):
        self._d = dict(d)
        for k, v in d.items():
            setattr(self, k, v)

    def to_dict(self):
        return dict(self._d)


def _make_dspark_proposer(max_reqs: int):
    hf_dict = {
        "model_type": "deepseek_v41",
        "dspark_noise_token_id": 128799,
        "dspark_block_size": 5,
        "dspark_target_layer_ids": [37, 38, 39],
    }
    spec_cfg = SimpleNamespace(
        num_speculative_tokens=5,
        draft_tensor_parallel_size=8,
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
        dspark_draft_topk=None,
        draft_model_config=SimpleNamespace(hf_config=_FakeHFConfig(hf_dict)),
    )
    vllm_cfg = SimpleNamespace(
        speculative_config=spec_cfg,
        model_config=SimpleNamespace(max_model_len=204800),
        compilation_config=SimpleNamespace(
            static_forward_context={"model.layers.0.self_attn": mock.MagicMock()},
            fast_moe_cold_start=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=8,
            is_moe_model=True,
            use_sequence_parallel_moe=False,
        ),
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        num_reqs_max_model_len=max_reqs,
        num_reqs_most_model_len=None,
        num_tokens_paddings=[
            16,
            32,
            64,
            128,
            192,
            256,
            384,
            512,
            768,
            1024,
            1536,
            2048,
            4096,
        ],
    )
    return DSparkProposer(runner=runner, vllm_config=vllm_cfg)


def test_dspark_v41_model_registry_and_weight_remap():
    from vllm.model_executor.models import ModelRegistry

    register_models()
    assert "DSparkV41DraftModel" in ModelRegistry.get_supported_archs()

    dummy = object.__new__(DSparkDeepseekV41ForCausalLM)
    assert (
        dummy._remap_dspark_name("mtp.0.main_proj.weight") == "model.main_proj.weight"
    )
    assert (
        dummy._remap_dspark_name("mtp.0.markov_head.embed.weight")
        == "model.markov_head.markov_w1.weight"
    )
    assert (
        dummy._remap_dspark_name("mtp.0.markov_head.head.weight")
        == "model.markov_head.markov_w2.weight"
    )
    assert (
        dummy._remap_dspark_name("mtp.2.attn.wq_b.weight")
        == "model.layers.2.attn.wq_b.weight"
    )
    assert dummy._remap_dspark_name("mtp.0.confidence_head.weight") is None


def test_dspark_tp8_draft_token_paddings():
    p64 = _make_dspark_proposer(64)
    assert p64._aux_layer_shift == 0
    assert p64._get_draft_token_paddings() == [16, 40, 80, 160, 240, 320]
    assert p64._get_padded_len(8 * 5) == 40

    p256 = _make_dspark_proposer(256)
    buckets = p256._get_draft_token_paddings()
    assert buckets == [16, 40, 80, 160, 240, 320, 400, 640, 1280]
    assert all(b % 8 == 0 for b in buckets)
    assert p256._get_padded_len(8 * 5) == 40
    assert p256._get_padded_len(32 * 5) == 160
    assert p256._get_padded_len(64 * 5) == 320
    assert p256._get_padded_len(128 * 5) == 640
    assert p256._get_padded_len(256 * 5) == 1280


def test_dflash_update_draft_kv_cache_skips_empty_aux_list():
    proposer = _make_dspark_proposer(64)
    proposer._tpu_precompute_and_update_kv_cache = mock.MagicMock()
    chunk_none = SimpleNamespace(aux_hidden_states=None)
    chunk_empty = SimpleNamespace(aux_hidden_states=[])
    proposer._update_draft_kv_cache_from_target(chunk_none)
    proposer._update_draft_kv_cache_from_target(chunk_empty)
    proposer._tpu_precompute_and_update_kv_cache.assert_not_called()


def test_gather_aux_hidden_states_batches_tp_all_gather():
    t0 = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    t1 = t0 + 100.0
    t2 = t0 + 200.0

    # When not token-sharded, returns as-is without calling tp_all_gather.
    out_unsharded = _gather_aux_hidden_states([t0, t1, t2], token_sharded_mhc=False)
    assert len(out_unsharded) == 3
    assert torch.equal(out_unsharded[0], t0)

    # When token-sharded, batches into a single tp_all_gather call and splits back.
    with mock.patch(
        "vllm_torchtpu.models.vllm.deepseek_v41.model.tp_all_gather",
        side_effect=lambda x, dim=0: torch.cat([x, x + 1000.0], dim=dim),
    ) as mock_ag:
        out_sharded = _gather_aux_hidden_states([t0, t1, t2], token_sharded_mhc=True)
        assert mock_ag.call_count == 1
        assert len(out_sharded) == 3
        for idx, base in enumerate((t0, t1, t2)):
            expected = torch.cat([base, base + 1000.0], dim=0)
            assert torch.equal(out_sharded[idx], expected)


def test_engram_cpu_hashes_1_plus_k_async_overlay_matches_ground_truth():
    # Build a lightweight TpuNgramHashState without tokenizer IO.
    hs = object.__new__(TpuNgramHashState)
    hs.layout = SimpleNamespace(max_ngram_size=4)
    hs.pad_id = 0
    hs._token_map_np = np.arange(1000, dtype=np.int64)
    hs._mult_np = np.array(
        [[3, 5, 7, 11], [13, 17, 19, 23]], dtype=np.int64
    )  # 2 engram layers, max_ngram=4
    hs._primes_np = np.array(
        [
            [[101, 103], [107, 109], [113, 127]],
            [[131, 137], [139, 149], [151, 157]],
        ],
        dtype=np.int64,
    )  # [2, 3, 2]
    hs._offsets_np = np.array(
        [[0, 200, 400, 600, 800, 1000], [10, 210, 410, 610, 810, 1010]],
        dtype=np.int64,
    )

    # Ground-truth token history for request 0:
    # - Prompt tokens at 0..9: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # - Step N-1 verified 5 drafts, accepted 3 drafts + 1 bonus (L_acc = 4):
    #   sampled_ids = [20, 21, 22, 23] at positions 10..13 (2 drafts rejected)
    # - Step N verifies 1+K = 6 tokens: [23, 31, 32, 33, 34, 35] at pos 13..18
    gt_tokens = np.zeros((2, 64), dtype=np.int32)
    gt_tokens[0, :10] = np.arange(10, 20, dtype=np.int32)
    gt_tokens[0, 10:13] = [20, 21, 22]
    gt_tokens[0, 13:19] = [23, 31, 32, 33, 34, 35]

    gt_req_idx = np.zeros(6, dtype=np.int32)
    gt_pos = np.array([13, 14, 15, 16, 17, 18], dtype=np.int32)
    expected_hashes = hs.cpu_hashes(gt_tokens, gt_req_idx, gt_pos)

    # Now simulate async scheduling state before _modify_prev_results:
    # - token_ids only has prompt tokens 0..9 committed (10..15 hold placeholders)
    # - num_tokens_no_spec was optimistically advanced to 10 + 6 = 16
    # - optimistic positions_np for Step N are [15, 16, 17, 18, 19, 20]
    async_tokens = np.zeros((2, 64), dtype=np.int32)
    async_tokens[0, :10] = np.arange(10, 20, dtype=np.int32)
    async_tokens[0, 10:16] = 999  # stale placeholders

    optimistic_pos = np.array([15, 16, 17, 18, 19, 20], dtype=np.int32)
    n_placeholder = 6
    end_idx_no_spec = 16
    sampled_r = np.array([20, 21, 22, 23], dtype=np.int64)
    l_acc = len(sampled_r)
    n_rej = n_placeholder - l_acc  # 2
    start_idx = end_idx_no_spec - n_placeholder  # 10
    src = np.array([23, 31, 32, 33, 34, 35], dtype=np.int64)

    span_start_by_req = np.array([start_idx, 1000], dtype=np.int32)
    span_tokens = np.zeros((2, 14), dtype=np.int32)
    span_tokens[0, : l_acc - 1] = sampled_r[:-1]
    span_tokens[0, l_acc - 1 : l_acc - 1 + len(src)] = src

    actual_hashes = hs.cpu_hashes(
        async_tokens,
        gt_req_idx,
        optimistic_pos - n_rej,
        span_start_by_req=span_start_by_req,
        span_tokens=span_tokens,
    )
    np.testing.assert_array_equal(actual_hashes, expected_hashes)


def test_async_pre_results_wait_for_next_tokens_cpu():
    t = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
    ev = mock.MagicMock()
    res = AsyncPreResults(
        req_ids=["r0"],
        next_tokens_tpu=t,
        request_seq_lens=[],
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={"r0": 0},
        next_tokens_cpu=t,
        next_tokens_ready_event=ev,
    )
    out = res.wait_for_next_tokens_cpu()
    assert torch.equal(out, t)
    ev.synchronize.assert_called_once()
    # Second call should not re-synchronize.
    res.wait_for_next_tokens_cpu()
    ev.synchronize.assert_called_once()


def test_leader_group_bounds_ignores_draft_layers():
    from vllm_torchtpu.models.vllm.deepseek_v41.attention import (
        VllmDeepseekV41MLAAttention,
    )

    ctx = {}
    for lid in range(43):
        m = object.__new__(VllmDeepseekV41MLAAttention)
        object.__setattr__(m, "prefix", f"model.layers.{lid}.attn")
        object.__setattr__(m, "non_causal_multi_token_decode", lid >= 40)
        object.__setattr__(m, "indexer", object() if lid in (0, 20, 24, 36) else None)
        object.__setattr__(m, "_static_forward_context", ctx)
        ctx[m.prefix] = m

    first_leader, max_lid = ctx["model.layers.20.attn"]._leader_group_bounds(
        ced_decoder=True
    )
    assert first_leader.prefix == "model.layers.20.attn"
    assert max_lid == 36
