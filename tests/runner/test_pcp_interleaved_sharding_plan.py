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
"""Unit tests for `InterleavedPcpShardingPlan` and unified `PcpManager`."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from tests.runner.runner_input_test_utils import (
    _make_runner,
    _scheduler_output,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_logits_indices,
    build_pcp_rank_major_token_order,
    pcp_local_token_counts,
)
from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator
from vllm_torchtpu.runner.pcp import (
    PcpDraftLayout,
    PcpManager,
    ensure_host_token_buffer_capacity,
    validate_pcp_manager_layers,
)
from vllm_torchtpu.runner.pcp.head_tail_sharding_plan import HeadTailPcpShardingPlan
from vllm_torchtpu.runner.pcp.interleaved_sharding_plan import (
    InterleavedPcpShardingPlan,
    attention_kv_page_size,
    interleaved_max_local_tokens,
)
from vllm_torchtpu.runner.pcp.manager import PcpShardingPlanName
from vllm_torchtpu.runner.pcp.sharding_plan import round_up_to_bucket
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _vllm_config(*, pcp_size=4, dcp_size=1, tp_size=2, interleave_size=4, **overrides):
    model_kw = dict(
        architecture="Qwen3MoeForCausalLM",
        use_mla=False,
        hf_text_config=SimpleNamespace(),
        max_model_len=32768,
        is_multimodal_model=False,
        multimodal_config=None,
        runner_type="generate",
    )
    parallel_kw = dict(
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        tensor_parallel_size=tp_size,
        cp_kv_cache_interleave_size=interleave_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        is_moe_model=True,
        enable_expert_parallel=True,
    )
    for key in list(overrides):
        if key in model_kw:
            model_kw[key] = overrides.pop(key)
        elif key in parallel_kw:
            parallel_kw[key] = overrides.pop(key)
    model_config = overrides.pop("model_config", SimpleNamespace(**model_kw))
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(**parallel_kw),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=16384, max_num_seqs=16, async_scheduling=False
        ),
        cache_config=SimpleNamespace(block_size=128),
        speculative_config=overrides.pop("speculative_config", None),
        lora_config=overrides.pop("lora_config", None),
        kv_transfer_config=overrides.pop("kv_transfer_config", None),
    )


def _attach_interleaved_manager(
    runner, *, pcp_size, pcp_rank, interleave_size, max_num_batched_tokens=None
):
    runner.parallel_config.prefill_context_parallel_size = pcp_size
    runner.parallel_config.cp_kv_cache_interleave_size = interleave_size
    budget = (
        max(runner.num_tokens_paddings) * pcp_size
        if max_num_batched_tokens is None
        else max_num_batched_tokens
    )
    runner.pcp_manager = PcpManager(
        InterleavedPcpShardingPlan(
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
            max_num_batched_tokens=budget,
            page_size=runner.block_size,
        )
    )


@pytest.mark.parametrize(
    ("cfg_kw", "kv_layout", "expected_err", "match"),
    [
        ({"dcp_size": 2}, "NHD", NotImplementedError, "DCP"),
        ({"interleave_size": 0}, "NHD", ValueError, "interleave_size > 0"),
        ({"interleave_size": 3}, "NHD", ValueError, "must divide 512"),
        ({"interleave_size": 4}, "HND", ValueError, "multiple of 128"),
        (
            {"enable_expert_parallel": False},
            "NHD",
            NotImplementedError,
            "expert-parallel",
        ),
        (
            {
                "is_multimodal_model": True,
                "multimodal_config": SimpleNamespace(language_model_only=False),
            },
            "NHD",
            NotImplementedError,
            "language-model-only",
        ),
        (
            {
                "architecture": "Qwen3_5MoeForConditionalGeneration",
                "interleave_size": 4,
            },
            "NHD",
            ValueError,
            "multiple of 16",
        ),
        (
            {
                "speculative_config": SimpleNamespace(
                    method="eagle", num_speculative_tokens=1
                )
            },
            "NHD",
            NotImplementedError,
            "method=mtp",
        ),
        (
            {
                "speculative_config": SimpleNamespace(
                    method="mtp", num_speculative_tokens=2
                )
            },
            "NHD",
            NotImplementedError,
            "num_speculative_tokens=1",
        ),
        ({"runner_type": "pooling"}, "NHD", NotImplementedError, "pooling"),
        (
            {"use_mla": True, "hf_text_config": SimpleNamespace()},
            "NHD",
            NotImplementedError,
            "dense MLA",
        ),
        (
            {
                "pcp_size": 2,
                "dcp_size": 4,
                "tp_size": 4,
                "use_mla": True,
                "hf_text_config": SimpleNamespace(index_topk=2048),
            },
            "NHD",
            NotImplementedError,
            "decode_context_parallel_size ==",
        ),
        (
            {
                "pcp_size": 2,
                "dcp_size": 8,
                "tp_size": 4,
                "use_mla": True,
                "hf_text_config": SimpleNamespace(index_topk=2048),
                "is_multimodal_model": True,
                "multimodal_config": SimpleNamespace(language_model_only=True),
            },
            "NHD",
            NotImplementedError,
            "multimodal",
        ),
        (
            {
                "pcp_size": 2,
                "dcp_size": 8,
                "tp_size": 4,
                "use_mla": True,
                "hf_text_config": SimpleNamespace(index_topk=2048),
                "kv_transfer_config": object(),
            },
            "NHD",
            NotImplementedError,
            "KV",
        ),
        (
            {
                "pcp_size": 2,
                "dcp_size": 8,
                "tp_size": 4,
                "use_mla": True,
                "hf_text_config": SimpleNamespace(index_topk=2048),
                "pipeline_parallel_size": 2,
            },
            "NHD",
            NotImplementedError,
            "pipeline",
        ),
        (
            {
                "pcp_size": 2,
                "dcp_size": 8,
                "tp_size": 4,
                "use_mla": True,
                "hf_text_config": SimpleNamespace(index_topk=2048),
                "speculative_config": SimpleNamespace(
                    method="mtp", num_speculative_tokens=1
                ),
            },
            "NHD",
            NotImplementedError,
            "speculative",
        ),
    ],
)
def test_pcp_manager_static_validation_rejects_invalid_configs(
    cfg_kw, kv_layout, expected_err, match
):
    with pytest.raises(expected_err, match=match):
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_config(**cfg_kw), kv_cache_layout=kv_layout
        )


@pytest.mark.parametrize("kv_role", [None, "kv_producer", "kv_consumer", "kv_both"])
def test_pcp_manager_accepts_interleaved_features_and_selects_plans(kv_role):
    disabled = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(pcp_size=1), kv_cache_layout="NHD"
    )
    assert disabled.enabled is False
    with pytest.raises(NotImplementedError, match="> 1"):
        PcpManager.from_vllm_config(_vllm_config(pcp_size=1))

    kv_transfer_cfg = (
        None
        if kv_role is None
        else SimpleNamespace(
            kv_role=kv_role,
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
        )
    )
    cfg = _vllm_config(
        architecture="Qwen3_5MoeForConditionalGeneration",
        interleave_size=16,
        is_multimodal_model=True,
        multimodal_config=SimpleNamespace(language_model_only=True),
        pipeline_parallel_size=2,
        data_parallel_size=2,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        kv_transfer_config=kv_transfer_cfg,
    )
    PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(model_config=None), kv_cache_layout="NHD"
    )
    PcpStaticSupportValidator.validate_platform_config(cfg, kv_cache_layout="NHD")
    mgr = PcpManager.from_vllm_config(cfg)
    assert mgr.plan_name is PcpShardingPlanName.INTERLEAVED
    assert isinstance(mgr.sharding_plan, InterleavedPcpShardingPlan)
    assert mgr.uses_pcp_metadata is False
    assert mgr.uses_pcp_streaming is True
    assert mgr.requires_backend_preinit is True
    assert mgr.backend_preinit_world_size == 4

    ht_cfg = _vllm_config(
        pcp_size=2,
        dcp_size=8,
        tp_size=4,
        use_mla=True,
        hf_text_config=SimpleNamespace(index_topk=2048),
    )
    PcpStaticSupportValidator.validate_platform_config(ht_cfg, kv_cache_layout="HND")
    ht_mgr = PcpManager.from_vllm_config(ht_cfg)
    assert ht_mgr.plan_name is PcpShardingPlanName.HEADTAIL
    assert isinstance(ht_mgr.sharding_plan, HeadTailPcpShardingPlan)
    assert ht_mgr.uses_pcp_metadata is True
    assert ht_mgr.uses_pcp_streaming is False
    assert ht_mgr.draft_layout() is None
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        ht_mgr.padded_num_tokens(
            runner=None,
            scheduler_output=None,
            start_index=0,
            num_scheduled_tokens_per_req=[1],
        )


def test_validate_pcp_manager_layers_interleaved():
    from vllm.model_executor.layers.attention import Attention

    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    attn_layer = Attention.__new__(Attention)
    attn_layer.impl = PallasAttentionBackendImpl.__new__(PallasAttentionBackendImpl)
    gdn_layer = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
    validate_pcp_manager_layers(
        {"attn": attn_layer, "gdn": gdn_layer}, PcpShardingPlanName.INTERLEAVED
    )

    class _UnsupportedLayer:
        supports_tpu_pcp_manager = True

    with pytest.raises(NotImplementedError, match="_UnsupportedLayer"):
        validate_pcp_manager_layers(
            {"attn": attn_layer, "mla": _UnsupportedLayer()},
            PcpShardingPlanName.INTERLEAVED,
        )


def test_interleaved_page_size_and_max_local_tokens():
    assert interleaved_max_local_tokens(16384, pcp_size=4, interleave_size=4) == 4096
    assert interleaved_max_local_tokens(2049, pcp_size=4, interleave_size=128) == 1024

    plan = InterleavedPcpShardingPlan(
        pcp_size=4, pcp_rank=0, interleave_size=4, max_num_batched_tokens=16384
    )
    attn = FullAttentionSpec(
        block_size=128, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
    )
    mamba = MambaSpec(shapes=((4, 8),), dtypes=(torch.bfloat16,), block_size=512)
    kv_cfg = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(["a"], UniformTypeKVCacheSpecs(128, {"a": attn})),
            KVCacheGroupSpec(["m"], mamba),
        ]
    )
    assert attention_kv_page_size(kv_cfg) == 128
    plan.configure_kv_cache(kv_cfg)
    assert plan.page_size == 128
    with pytest.raises(ValueError, match="multiple of cp_kv_cache_interleave_size"):
        plan.set_page_size(6)


@pytest.mark.parametrize(
    ("pcp_size", "computed", "prompts", "scheduled", "interleave", "uses_mrope"),
    [
        (4, [0], [37], [37], 4, False),
        (4, [20, 31, 44, 15, 28, 39], [16] * 6, [1] * 6, 4, False),
        (4, [40, 8], [32, 48], [1, 23], 4, True),
        (4, [0, 12], [260, 300], [260, 270], 128, False),
        (2, [16], [32], [16], 16, False),
        (4, [0, 0], [1024, 512], [1024, 512], 16, False),
        (8, [0] * 32, [256] * 32, [256] * 32, 256, False),
    ],
)
def test_interleaved_plan_packs_tokens_positions_and_logits_across_ranks(
    pcp_size, computed, prompts, scheduled, interleave, uses_mrope
):
    block_size = max(16, interleave)
    paddings = [16, 32, 256, 512, 1024]
    total_tokens = sum(scheduled)
    owner_starts = np.concatenate(([0], np.cumsum(scheduled[:-1], dtype=np.int64)))
    local_counts = pcp_local_token_counts(
        scheduled,
        pcp_size,
        interleave,
        token_owner_start_offsets_per_req=owner_starts,
    )
    expected_local_padded = round_up_to_bucket(paddings, int(local_counts.max()))
    global_padded = expected_local_padded * pcp_size
    packed_to_req, req_to_packed = build_pcp_rank_major_token_order(
        scheduled,
        pcp_size=pcp_size,
        interleave_size=interleave,
        padded_num_tokens=global_padded,
        token_owner_start_offsets_per_req=owner_starts,
    )
    expected_logits = build_pcp_logits_indices(
        scheduled,
        pcp_size=pcp_size,
        interleave_size=interleave,
        padded_num_tokens=global_padded,
        token_owner_start_offsets_per_req=owner_starts,
    )

    owner_hits = np.zeros(total_tokens, dtype=np.int32)
    for rank in range(pcp_size):
        mgr = _make_runner(
            num_computed_tokens=computed,
            prompt_tokens=prompts,
            scheduled_tokens=scheduled,
            token_paddings=paddings,
            uses_mrope=uses_mrope,
            pcp_size=pcp_size,
            max_num_reqs=max(8, len(scheduled)),
            block_size=block_size,
        )
        _attach_interleaved_manager(
            mgr,
            pcp_size=pcp_size,
            pcp_rank=rank,
            interleave_size=interleave,
            max_num_batched_tokens=total_tokens,
        )
        so = _scheduler_output(scheduled)
        assert (
            mgr.pcp_manager.padded_num_tokens(
                runner=mgr,
                scheduler_output=so,
                start_index=0,
                num_scheduled_tokens_per_req=np.asarray(scheduled, dtype=np.int32),
            )
            == expected_local_padded
        )

        mgr_attn, mgr_logits, padded_num_reqs, *_ = TPUModelRunner._prepare_inputs(
            mgr, so, 0, len(scheduled)
        )
        md = mgr_attn["layer.0"]
        assert md.use_pcp_streaming is True
        plan = mgr.pcp_manager.sharding_plan
        local_start = rank * expected_local_padded
        local_end = local_start + expected_local_padded
        assert plan.token_slice == slice(local_start, local_end)
        assert plan.local_padded_num_tokens == expected_local_padded
        assert plan.local_num_tokens == int(local_counts[rank])
        assert plan.local_padded_num_tokens <= mgr.pcp_manager.max_local_num_tokens

        local_map = packed_to_req[local_start:local_end]
        np.testing.assert_array_equal(
            mgr.pcp_manager.local_to_request_major_token_indices, local_map
        )

        req_major_ids = np.concatenate(
            [
                mgr.input_batch.token_ids_cpu[i, c : c + s]
                for i, (c, s) in enumerate(zip(computed, scheduled))
            ]
        )
        expected_ids = np.where(
            local_map >= 0, req_major_ids[np.maximum(local_map, 0)], 0
        )
        np.testing.assert_array_equal(mgr.input_ids.numpy(), expected_ids)

        if uses_mrope:
            base_pos = np.arange(total_tokens, dtype=np.int32)
            for row_idx, offset in enumerate((0, 1000, 2000)):
                row_vals = base_pos + offset
                expected_pos = np.where(
                    local_map >= 0, row_vals[np.maximum(local_map, 0)], 0
                )
                np.testing.assert_array_equal(
                    mgr.position_ids[row_idx].numpy(), expected_pos
                )
        else:
            req_major_pos = np.concatenate(
                [
                    np.arange(c, c + s, dtype=np.int32)
                    for c, s in zip(computed, scheduled)
                ]
            )
            expected_pos = np.where(
                local_map >= 0, req_major_pos[np.maximum(local_map, 0)], 0
            )
            np.testing.assert_array_equal(mgr.position_ids.numpy(), expected_pos)

        padded_logits = np.full(padded_num_reqs, -1, dtype=np.int32)
        padded_logits[: len(scheduled)] = expected_logits
        np.testing.assert_array_equal(mgr_logits.numpy(), padded_logits)

        for u in range(total_tokens):
            local_idx = plan.local_index_for_request_major_token(u)
            if local_start <= int(req_to_packed[u]) < local_end:
                assert local_idx == int(req_to_packed[u]) - local_start
                owner_hits[u] += 1
            else:
                assert local_idx is None

    np.testing.assert_array_equal(owner_hits, np.ones(total_tokens, dtype=np.int32))


def test_padded_num_tokens_honors_dp_bucket_and_rejects_overflow():
    r = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=[20, 5],
        scheduled_tokens=[20, 5],
        token_paddings=[16, 32, 64],
        pcp_size=2,
        block_size=16,
    )
    _attach_interleaved_manager(r, pcp_size=2, pcp_rank=0, interleave_size=16)
    counts = np.asarray([20, 5], dtype=np.int32)
    so = _scheduler_output([20, 5])

    assert (
        r.pcp_manager.padded_num_tokens(
            runner=r,
            scheduler_output=so,
            start_index=0,
            num_scheduled_tokens_per_req=counts,
        )
        == 16
    )
    r._dp_target_bucket = 64
    assert (
        r.pcp_manager.padded_num_tokens(
            runner=r,
            scheduler_output=so,
            start_index=0,
            num_scheduled_tokens_per_req=counts,
        )
        == 64
    )

    overflow = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[1024],
        scheduled_tokens=[1024],
        token_paddings=[128],
        pcp_size=4,
        block_size=16,
    )
    _attach_interleaved_manager(overflow, pcp_size=4, pcp_rank=0, interleave_size=16)
    with pytest.raises(ValueError, match="compile bucket"):
        overflow.pcp_manager.padded_num_tokens(
            runner=overflow,
            scheduler_output=_scheduler_output([1024]),
            start_index=0,
            num_scheduled_tokens_per_req=np.asarray([1024], dtype=np.int32),
        )


@pytest.mark.parametrize(
    ("q_lens", "pcp_size", "interleave_size", "padded_num_tokens", "owner_starts"),
    [
        ([2, 3], 2, 1, 6, [0, 2]),
        ([3, 3], 2, 2, 8, [0, 3]),
        ([2, 5, 1], 4, 2, 16, [0, 2, 7]),
        ([1], 4, 2, 8, [5]),
    ],
)
def test_pcp_draft_layout_localizes_tokens_and_gathers_uniquely(
    q_lens, pcp_size, interleave_size, padded_num_tokens, owner_starts
):
    packed_to_request, request_to_packed = build_pcp_rank_major_token_order(
        q_lens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens=padded_num_tokens,
        token_owner_start_offsets_per_req=owner_starts,
    )
    total_num_tokens = sum(q_lens)
    padded_req_to_packed = np.full(padded_num_tokens, -1, dtype=request_to_packed.dtype)
    padded_req_to_packed[:total_num_tokens] = request_to_packed

    # Append a padded trailing gather entry to verify num_valid_gathers masking.
    request_gathers = torch.tensor(
        list(np.cumsum(q_lens, dtype=np.int64) - 1) + [0],
        dtype=torch.int64,
    )
    request_tokens = torch.arange(total_num_tokens, dtype=torch.int64)
    local_padded = padded_num_tokens // pcp_size
    owner_counts = torch.zeros(len(q_lens), dtype=torch.int64)

    for rank in range(pcp_size):
        local_start = rank * local_padded
        local_end = local_start + local_padded
        local_to_req = packed_to_request[local_start:local_end]
        layout = PcpDraftLayout(
            local_to_request_major=local_to_req,
            padded_request_to_packed=padded_req_to_packed,
            local_start=local_start,
            local_end=local_end,
        )
        local_tokens, local_gathers = layout.localize_token_tensor_and_gather_indices(
            request_tokens,
            request_gathers,
            num_valid_gathers=len(q_lens),
        )
        assert int(local_gathers[-1]) == -1
        if not np.any(local_to_req >= 0):
            torch.testing.assert_close(local_tokens, torch.zeros_like(local_tokens))
            torch.testing.assert_close(
                local_gathers, torch.full_like(local_gathers, -1)
            )
        owner_counts += local_gathers[:-1].ge(0)

    torch.testing.assert_close(owner_counts, torch.ones_like(owner_counts))


def test_pcp_draft_layout_snapshot_and_owner_masked_reduce():
    r = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=[17, 25],
        scheduled_tokens=[17, 25],
        token_paddings=[32],
        pcp_size=2,
        block_size=16,
    )
    r.num_reqs_max_model_len = 1
    _attach_interleaved_manager(r, pcp_size=2, pcp_rank=0, interleave_size=16)
    so = _scheduler_output([17, 25])

    TPUModelRunner._prepare_inputs(r, so, 0, 0)
    first_draft = r.pcp_manager.draft_layout()
    assert first_draft is not None
    assert first_draft.padded_request_to_packed.shape == (64,)
    first_local_copy = first_draft.local_to_request_major.copy()

    TPUModelRunner._prepare_inputs(r, so, 1, 0)
    second_draft = r.pcp_manager.draft_layout()
    assert second_draft is not None
    assert first_draft is not second_draft
    np.testing.assert_array_equal(first_draft.local_to_request_major, first_local_copy)

    local_tensor = torch.tensor([[99.0], [20.0], [7.0]])
    owner_mask = torch.tensor([False, True, False])
    reduced_inputs: list[torch.Tensor] = []

    def fake_reduce(masked_tensor: torch.Tensor) -> torch.Tensor:
        reduced_inputs.append(masked_tensor.clone())
        remote_owner = torch.zeros_like(masked_tensor)
        remote_owner[0, 0] = 10.0
        return masked_tensor + remote_owner

    with patch(
        "vllm_torchtpu.runner.pcp.sharding_plan.all_reduce_sum", side_effect=fake_reduce
    ):
        result = first_draft.aggregate_request_aligned_tensor(local_tensor, owner_mask)

    torch.testing.assert_close(reduced_inputs[0], torch.tensor([[0.0], [20.0], [0.0]]))
    torch.testing.assert_close(result, torch.tensor([[10.0], [20.0], [0.0]]))


def test_interleaved_logits_selection_and_dummy_step():
    scheduled = [9, 6, 1]
    hidden_per_rank = []
    managers = []
    for rank in range(4):
        r = _make_runner(
            num_computed_tokens=[0, 4, 20],
            prompt_tokens=[9, 10, 16],
            scheduled_tokens=scheduled,
            token_paddings=[512, 1024],
            pcp_size=4,
            block_size=16,
        )
        _attach_interleaved_manager(r, pcp_size=4, pcp_rank=rank, interleave_size=4)
        TPUModelRunner._prepare_inputs(r, _scheduler_output(scheduled), 0, 3)
        hidden_per_rank.append(r.input_ids.to(torch.float32).unsqueeze(-1))
        managers.append(r.pcp_manager)

    with patch(
        "vllm_torchtpu.runner.pcp.sharding_plan.all_reduce_sum", side_effect=lambda x: x
    ):
        summed = sum(
            m.maybe_select_logits_hidden_states(h)
            for m, h in zip(managers, hidden_per_rank)
        )
    assert summed[:3, 0].to(torch.int32).tolist() == [108, 209, 320]

    managers[0].prepare_dummy(num_tokens=512, num_reqs=8, max_num_tokens=1024)
    assert managers[0].sharding_plan.global_padded_num_tokens == 2048
    assert managers[0].sharding_plan.local_padded_num_tokens == 512
    assert managers[0].sharding_plan.token_slice == slice(0, 512)
    assert managers[0].uses_pcp_streaming is True


def test_interleaved_prepare_real_rejects_boundary_and_encoder_inputs():
    bad_boundary = _make_runner(
        num_computed_tokens=[6],
        prompt_tokens=[8],
        scheduled_tokens=[4],
        token_paddings=[512],
        pcp_size=4,
        block_size=16,
    )
    _attach_interleaved_manager(bad_boundary, pcp_size=4, pcp_rank=0, interleave_size=4)
    with pytest.raises(NotImplementedError, match="boundary-crossing"):
        TPUModelRunner._prepare_inputs(bad_boundary, _scheduler_output([4]), 0, 1)

    mm_runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[8],
        scheduled_tokens=[8],
        token_paddings=[512],
        pcp_size=4,
        block_size=16,
    )
    _attach_interleaved_manager(mm_runner, pcp_size=4, pcp_rank=0, interleave_size=4)
    so = _scheduler_output([8])
    so.scheduled_encoder_inputs = {"req0": [0]}
    with pytest.raises(NotImplementedError, match="encoder inputs"):
        TPUModelRunner._prepare_inputs(mm_runner, so, 0, 1)


def test_backbone_token_paddings_filters_and_rejects_compile_size_hole():
    r = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[8],
        scheduled_tokens=[8],
        token_paddings=[16, 64, 512, 1024, 2048, 4096],
        pcp_size=4,
        block_size=16,
    )
    _attach_interleaved_manager(
        r, pcp_size=4, pcp_rank=0, interleave_size=4, max_num_batched_tokens=4096
    )
    assert TPUModelRunner._backbone_token_paddings(r) == [512, 1024]

    r.num_tokens_paddings = [512, 1500]
    with pytest.raises(ValueError, match="compile_sizes"):
        TPUModelRunner._backbone_token_paddings(r)


def test_ensure_host_token_buffer_capacity_grows_all_staging_buffers():
    r = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[8],
        scheduled_tokens=[8],
        token_paddings=[16],
        uses_mrope=True,
        pcp_size=1,
        block_size=16,
    )
    r.supports_mm_inputs = True
    r.is_mm_embed_cpu = torch.zeros(16, dtype=torch.bool)
    r.input_ids_cpu[:4] = torch.tensor([11, 22, 33, 44], dtype=torch.int32)
    r.positions_cpu[:4] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    r.is_mm_embed_cpu[2] = True
    r.mrope_positions.cpu[:, :4] = 7

    ensure_host_token_buffer_capacity(r, 64)
    assert r.input_ids_cpu.shape == (64,)
    assert r.positions_cpu.shape == (64,)
    assert r.positions_np.shape == (64,)
    assert r.is_mm_embed_cpu.shape == (64,)
    assert r.mrope_positions.cpu.shape == (3, 65)
    assert r.arange_np.shape == (64,)
    assert r.input_ids_cpu[:4].tolist() == [11, 22, 33, 44]
    assert r.positions_np[:4].tolist() == [1, 2, 3, 4]
    assert bool(r.is_mm_embed_cpu[2]) is True
    assert torch.all(r.mrope_positions.cpu[:, :4] == 7)


def test_mrope_second_chunk_shifts_positions_and_guards_overflow():
    scheduled = [5, 7]
    r = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=[5, 7],
        scheduled_tokens=scheduled,
        token_paddings=[16],
        uses_mrope=True,
        pcp_size=1,
        block_size=16,
    )
    r.num_reqs_max_model_len = 1
    TPUModelRunner._prepare_inputs(r, _scheduler_output(scheduled), 1, 0)
    expected = torch.stack(
        [
            torch.arange(5, 12, dtype=torch.int32),
            torch.arange(1005, 1012, dtype=torch.int32),
            torch.arange(2005, 2012, dtype=torch.int32),
        ]
    )
    assert torch.equal(r.position_ids[:, :7], expected)

    r.mrope_positions = r._make_buffer(3, 8, dtype=torch.int32)
    r._calc_mrope_positions = lambda _so: None
    with pytest.raises(ValueError, match="mrope_positions buffer"):
        TPUModelRunner._prepare_inputs(r, _scheduler_output(scheduled), 1, 0)


def test_init_token_buckets_and_host_buffers_narrows_paddings_and_presizes_buffers():
    compile_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    r = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[2049],
        scheduled_tokens=[2049],
        token_paddings=[16],
        uses_mrope=True,
        pcp_size=4,
        block_size=128,
    )
    r.supports_mm_inputs = True
    r.scheduler_config.max_num_batched_tokens = 2049
    r.vllm_config.compilation_config.compile_sizes = list(compile_sizes)
    r._backbone_token_paddings = TPUModelRunner._backbone_token_paddings.__get__(r)
    _attach_interleaved_manager(
        r, pcp_size=4, pcp_rank=0, interleave_size=128, max_num_batched_tokens=2049
    )

    TPUModelRunner._init_token_buckets_and_host_buffers(r, r.vllm_config)

    assert r.num_tokens_paddings == [512, 1024]
    assert r.max_num_tokens == 1024
    assert r._global_num_tokens_paddings == compile_sizes
    assert r.vllm_config.compilation_config.compile_sizes == [512, 1024]
    assert r.input_ids_cpu.shape == (4096,)
    assert r.positions_cpu.shape == (4096,)
    assert r.mrope_positions.cpu.shape == (3, 4097)
    assert r.is_mm_embed_cpu.shape == (4096,)
    assert r.arange_np.shape == (4096,)
    assert r.query_start_loc_cpu.shape == (1025,)
    assert r.seq_lens_cpu.shape == (1024,)

    def fill_mrope_positions(so):
        n = int(so.total_num_scheduled_tokens)
        cols = torch.arange(n, dtype=torch.int32)
        r.mrope_positions.cpu[0, :n] = cols
        r.mrope_positions.cpu[1, :n] = cols + 1000
        r.mrope_positions.cpu[2, :n] = cols + 2000

    r._calc_mrope_positions = fill_mrope_positions
    orig_ids = r.input_ids_cpu

    # Small step must round to local 512 (not unnarrowed 16).
    TPUModelRunner._prepare_inputs(r, _scheduler_output([8]), 0, 1)
    assert r.pcp_manager.sharding_plan.local_padded_num_tokens == 512

    # Full 2049-token step (> max_num_tokens=1024) fits pre-sized host buffers.
    TPUModelRunner._prepare_inputs(r, _scheduler_output([2049]), 0, 1)
    assert r.input_ids_cpu is orig_ids
    assert r.pcp_manager.sharding_plan.local_padded_num_tokens == 1024


def test_phased_profiler_uses_global_token_buckets_when_local_paddings_narrowed():
    class _Stop(Exception):
        pass

    def stop():
        raise _Stop

    r = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[1500],
        scheduled_tokens=[1500],
        token_paddings=[512, 1024],
        pcp_size=4,
        block_size=16,
    )
    r.num_tokens_paddings = [512, 1024]
    r._global_num_tokens_paddings = [16, 64, 512, 1024, 2048, 4096]
    r.execute_model_state = None
    r.batch_counter = 0
    r.phase_based_profiler = MagicMock()
    r._update_states = lambda _so: None
    r._reorder_batch_for_rpa = lambda _so: (0, 0)
    r._count_input_chunks = lambda _so: (1, 1)
    r._any_decode = lambda _so: False
    r._dp_local_prompt_logprob_rows = lambda _so: 0
    r._dp_coordinated_step = lambda *_a, **_kw: (None, 1)
    r._readmit_trimmed_drafts = lambda *_a: None
    r._is_async_drafter = False
    r._greedy_top_tokens_this_step = stop

    with patch("vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", lambda: False):
        with pytest.raises(_Stop):
            TPUModelRunner.execute_model(r, _scheduler_output([1500]))
        stats = r.phase_based_profiler.step.call_args.args[0]
        assert stats["padded_total_num_scheduled_tokens"] == 2048

        # Custom per-rank compile_sizes cap below global step budget: clamp cleanly.
        r._global_num_tokens_paddings = [512, 1024]
        with pytest.raises(_Stop):
            TPUModelRunner.execute_model(r, _scheduler_output([1500]))
        clamped = r.phase_based_profiler.step.call_args.args[0]
        assert clamped["padded_total_num_scheduled_tokens"] == 1024
