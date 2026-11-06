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
"""Unit tests for `InterleavedPcpShardingPlan` and `PcpManager` validation."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.cp_mla_attention import dcp_spans_pcp
from vllm_torchtpu.layers.core.pcp_sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
    ensure_host_token_buffer_capacity,
)
from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator
from vllm_torchtpu.runner.pcp import PcpManager, validate_pcp_manager_layers
from vllm_torchtpu.runner.pcp.head_tail_sharding_plan import HeadTailPcpShardingPlan
from vllm_torchtpu.runner.pcp.interleaved_sharding_plan import (
    InterleavedPcpShardingPlan,
    attention_kv_page_size,
    interleaved_max_local_tokens,
)
from vllm_torchtpu.runner.pcp.manager import PcpShardingPlanName
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _vllm_config(*, pcp_size=4, dcp_size=1, tp_size=2, interleave_size=4, **overrides):
    model_kw = dict(
        architecture="Qwen3MoeForCausalLM",
        use_mla=False,
        hf_text_config=SimpleNamespace(),
        max_model_len=32768,
        is_multimodal_model=False,
        multimodal_config=None,
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
    for key, val in list(overrides.items()):
        if key in model_kw:
            model_kw[key] = overrides.pop(key)
        elif key in parallel_kw:
            parallel_kw[key] = overrides.pop(key)
    return SimpleNamespace(
        model_config=SimpleNamespace(**model_kw),
        parallel_config=SimpleNamespace(**parallel_kw),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=16384, max_num_seqs=16, async_scheduling=False
        ),
        cache_config=SimpleNamespace(block_size=128),
        speculative_config=overrides.pop("speculative_config", None),
        lora_config=overrides.pop("lora_config", None),
        kv_transfer_config=None,
    )


def _attach_interleaved_manager(
    runner, *, pcp_size, pcp_rank, interleave_size, max_num_batched_tokens=None
):
    runner.parallel_config.prefill_context_parallel_size = pcp_size
    runner.parallel_config.cp_kv_cache_interleave_size = interleave_size
    runner.sequence_layout_planner = None
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
    ("plan_env", "cfg_kw", "kv_layout", "expected_err", "match"),
    [
        ("zigzag", {}, "NHD", ValueError, "Unsupported TPU_PCP_SHARDING_PLAN"),
        ("interleaved", {"pcp_size": 1}, "NHD", NotImplementedError, "> 1"),
        ("interleaved", {"dcp_size": 2}, "NHD", NotImplementedError, "decode_context"),
        ("interleaved", {"interleave_size": 3}, "NHD", ValueError, "must divide 512"),
        ("interleaved", {"interleave_size": 4}, "HND", ValueError, "multiple of 128"),
        (
            "interleaved",
            {"enable_expert_parallel": False},
            "NHD",
            NotImplementedError,
            "expert-parallel",
        ),
        (
            "interleaved",
            {
                "is_multimodal_model": True,
                "multimodal_config": SimpleNamespace(language_model_only=False),
            },
            "NHD",
            NotImplementedError,
            "language-model-only",
        ),
        (
            "interleaved",
            {
                "architecture": "Qwen3_5MoeForConditionalGeneration",
                "interleave_size": 4,
            },
            "NHD",
            NotImplementedError,
            ">= 16",
        ),
        (
            "interleaved",
            {
                "speculative_config": SimpleNamespace(
                    method="mtp", num_speculative_tokens=1
                )
            },
            "NHD",
            NotImplementedError,
            "speculative",
        ),
        ("headtail", {}, "NHD", NotImplementedError, "sparse MLA"),
        (
            "headtail",
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
            "headtail",
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
    ],
)
def test_pcp_manager_static_validation_rejects_invalid_configs(
    monkeypatch, plan_env, cfg_kw, kv_layout, expected_err, match
):
    monkeypatch.setattr(envs, "USE_TPU_PCP_MANAGER", True)
    monkeypatch.setattr(envs, "TPU_PCP_SHARDING_PLAN", plan_env)
    with pytest.raises(expected_err, match=match):
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_config(**cfg_kw), kv_cache_layout=kv_layout
        )


def test_pcp_manager_accepts_language_model_only_and_selects_plans(monkeypatch):
    monkeypatch.setattr(envs, "USE_TPU_PCP_MANAGER", True)
    monkeypatch.setattr(envs, "TPU_PCP_SHARDING_PLAN", "interleaved")
    assert dcp_spans_pcp() is False
    cfg = _vllm_config(
        architecture="Qwen3_5MoeForConditionalGeneration",
        interleave_size=16,
        is_multimodal_model=True,
        multimodal_config=SimpleNamespace(language_model_only=True),
    )
    PcpStaticSupportValidator.validate_platform_config(cfg, kv_cache_layout="NHD")
    mgr = PcpManager.from_vllm_config(cfg)
    assert isinstance(mgr.sharding_plan, InterleavedPcpShardingPlan)
    assert mgr.sequence_layout_descriptor == PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR
    assert mgr.requires_backend_preinit is True

    monkeypatch.setattr(envs, "TPU_PCP_SHARDING_PLAN", "headtail")
    assert dcp_spans_pcp() is True
    ht_cfg = _vllm_config(
        pcp_size=2,
        dcp_size=8,
        tp_size=4,
        use_mla=True,
        hf_text_config=SimpleNamespace(index_topk=2048),
    )
    assert isinstance(
        PcpManager.from_vllm_config(ht_cfg).sharding_plan, HeadTailPcpShardingPlan
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
        supports_interleaved_pcp_manager = True

    with pytest.raises(NotImplementedError, match="_UnsupportedLayer"):
        validate_pcp_manager_layers(
            {"attn": attn_layer, "mla": _UnsupportedLayer()},
            PcpShardingPlanName.INTERLEAVED,
        )
    assert (
        inspect.signature(validate_pcp_manager_layers).parameters["plan"].default
        is inspect.Parameter.empty
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
    ("computed", "prompts", "scheduled", "interleave", "uses_mrope"),
    [
        ([0], [37], [37], 4, False),
        ([20, 31, 44, 15, 28, 39], [16] * 6, [1] * 6, 4, False),
        ([40, 8], [32, 48], [1, 23], 4, True),
        ([0, 12], [260, 300], [260, 270], 128, False),
    ],
)
def test_interleaved_plan_matches_legacy_sequence_layout(
    computed, prompts, scheduled, interleave, uses_mrope
):
    pcp_size = 4
    block_size = max(16, interleave)
    paddings = [512, 1024]
    total_tokens = sum(scheduled)
    for rank in range(pcp_size):
        ref = _make_runner(
            num_computed_tokens=computed,
            prompt_tokens=prompts,
            scheduled_tokens=scheduled,
            token_paddings=paddings,
            uses_mrope=uses_mrope,
            pcp_size=pcp_size,
            block_size=block_size,
        )
        ref.parallel_config.cp_kv_cache_interleave_size = interleave
        ref.sequence_layout_planner.eligibility.interleave_size = interleave

        mgr = _make_runner(
            num_computed_tokens=computed,
            prompt_tokens=prompts,
            scheduled_tokens=scheduled,
            token_paddings=paddings,
            uses_mrope=uses_mrope,
            pcp_size=pcp_size,
            block_size=block_size,
        )
        _attach_interleaved_manager(
            mgr,
            pcp_size=pcp_size,
            pcp_rank=rank,
            interleave_size=interleave,
            max_num_batched_tokens=total_tokens,
        )

        with (
            patch(
                "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_rank",
                return_value=rank,
            ),
            patch(
                "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_world_size",
                return_value=pcp_size,
            ),
        ):
            ref_attn, ref_logits, *_ = TPUModelRunner._prepare_inputs(
                ref, _scheduler_output(scheduled), 0, len(scheduled)
            )
        mgr_attn, mgr_logits, *_ = TPUModelRunner._prepare_inputs(
            mgr, _scheduler_output(scheduled), 0, len(scheduled)
        )

        assert torch.equal(mgr.input_ids, ref.input_ids)
        assert torch.equal(mgr.position_ids, ref.position_ids)
        assert torch.equal(mgr_logits, ref_logits)
        md_mgr, md_ref = mgr_attn["layer.0"], ref_attn["layer.0"]
        assert torch.equal(md_mgr.seq_lens, md_ref.seq_lens)
        assert torch.equal(md_mgr.query_start_loc, md_ref.query_start_loc)
        assert (
            md_mgr.sequence_layout_kind,
            md_mgr.sequence_layout_protocol,
        ) == ("partial", "pcp_streaming")
        plan = mgr.pcp_manager.sharding_plan
        assert plan.local_padded_num_tokens <= mgr.pcp_manager.max_local_num_tokens
        assert mgr._last_sequence_layout_plan is plan.sequence_layout_plan
        assert plan.sequence_layout_plan is not None
        ref_plan = ref._last_sequence_layout_plan
        assert [
            plan.local_index_for_request_major_token(u) for u in range(total_tokens)
        ] == [
            ref_plan.local_index_for_request_major_token(u) for u in range(total_tokens)
        ]


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
    assert managers[0].sharding_plan.sequence_layout_plan is not None


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
    with pytest.raises(NotImplementedError, match="boundary"):
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
