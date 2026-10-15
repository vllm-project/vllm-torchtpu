# SPDX-License-Identifier: Apache-2.0
"""Admitted topology, physical DP slots and checkpoint ownership contracts."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config
from vllm_torchtpu.models.vllm.qwen4_exp.parallel import PLEParallel


def config(tp=1, dp=8):
    text = q4exp_text_config()
    return NS(
        model_config=NS(
            hf_text_config=text,
            hf_config=NS(
                quantization_config={
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                    "activation_scheme": "dynamic",
                }
            ),
            quantization="fp8",
            dtype=torch.bfloat16,
            max_model_len=8192,
        ),
        parallel_config=NS(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            data_parallel_size_local=dp,
            nnodes=1,
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_dbo=False,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
            expert_placement_strategy="linear",
            enable_elastic_ep=False,
        ),
        cache_config=NS(
            cache_dtype="auto",
            block_size=128,
            enable_prefix_caching=False,
            kv_cache_memory_bytes=None,
        ),
        scheduler_config=NS(
            max_num_batched_tokens=128,
            max_num_seqs=64,
            async_scheduling=False,
            disable_hybrid_kv_cache_manager=False,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
        additional_config={},
    )


@pytest.mark.parametrize("tp,dp", [(1, 8), (2, 4), (4, 2), (8, 1)])
@pytest.mark.parametrize("across", [False, True])
@pytest.mark.parametrize("enable_ep", [False, True])
def test_admitted_topologies_and_engram_modes(tp, dp, across, enable_ep):
    cfg = config(tp, dp)
    cfg.parallel_config.enable_expert_parallel = enable_ep
    if not enable_ep and tp == 1:
        with pytest.raises(ValueError, match="EP off.*TP>=2"):
            validate_config(cfg)
        return
    validate_config(cfg)  # Missing embedding_across_dp keeps the upstream default.
    cfg.additional_config = {
        "qwen4_exp": {
            "engram_config": {
                "cpu_offload": False,
                "embedding_across_dp": across,
                "dp_shared_memory": False,
            }
        }
    }
    validate_config(cfg)


@pytest.mark.parametrize("tp,dp", [(2, 2), (4, 4)])
@pytest.mark.parametrize("across", [False, True])
def test_topologies_requiring_fewer_or_more_than_eight_devices_are_rejected(
    tp, dp, across
):
    cfg = config(tp, dp)
    cfg.additional_config = {
        "qwen4_exp": {"engram_config": {"embedding_across_dp": across}}
    }
    with pytest.raises(ValueError, match="TP8/DP1"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "flag", ["cpu_offload", "embedding_across_dp", "dp_shared_memory"]
)
@pytest.mark.parametrize("value", ["false", 0, None])
def test_engram_flags_reject_truthiness(flag, value):
    cfg = config()
    cfg.additional_config = {"qwen4_exp": {"engram_config": {flag: value}}}
    with pytest.raises(ValueError, match=f"{flag} must be bool"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 2),
        ("data_parallel_size", 1),
        ("expert_placement_strategy", "round_robin"),
        ("enable_elastic_ep", True),
        ("enable_eplb", True),
        ("enable_dbo", True),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
    ],
)
def test_unsupported_parallel_modes_are_rejected(field, value):
    cfg = config()
    setattr(cfg.parallel_config, field, value)
    with pytest.raises(ValueError, match="requires TP8"):
        validate_config(cfg)


@pytest.mark.parametrize("tp,dp", [(2, 4), (4, 2)])
def test_model_admission_precedes_tpu_sequence_parallel_normalization(tp, dp):
    cfg = config(tp, dp)
    # vLLM derives this property for TP*DP EP before platform_activation
    # disables it. It is not a user-requested model capability.
    cfg.parallel_config.use_sequence_parallel_moe = True
    validate_config(cfg)


def test_shared_host_requires_explicit_offload():
    cfg = config()
    cfg.additional_config = {"qwen4_exp": {"engram_config": {"dp_shared_memory": True}}}
    with pytest.raises(ValueError, match="dp_shared_memory requires cpu_offload"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "shared,across", [(False, False), (False, True), (True, False)]
)
def test_host_configuration_keeps_ownership_modes_explicit(monkeypatch, shared, across):
    from vllm_torchtpu.models.vllm.qwen4_exp import config as admission

    monkeypatch.setattr(admission, "host_available_memory", lambda: 2 * 1024**4)
    cfg = config()
    cfg.additional_config = {
        "qwen4_exp": {
            "engram_config": dict(
                cpu_offload=True, dp_shared_memory=shared, embedding_across_dp=across
            )
        }
    }
    validate_config(cfg)


@pytest.mark.parametrize("invalid", ["across", "remote", "singleton"])
def test_shared_host_rejects_ambiguous_or_nonlocal_participants(invalid):
    cfg = config(8, 1) if invalid == "singleton" else config()
    cfg.additional_config = {
        "qwen4_exp": {
            "engram_config": dict(
                cpu_offload=True,
                dp_shared_memory=True,
                embedding_across_dp=invalid == "across",
            )
        }
    }
    if invalid == "remote":
        cfg.parallel_config.data_parallel_size_local = 4
    with pytest.raises(ValueError, match="shared PLE requires"):
        validate_config(cfg)


def test_private_host_replicas_are_checked_against_host_capacity(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp import config as admission

    monkeypatch.setattr(admission, "host_available_memory", lambda: 850 * 1024**3)
    cfg = config()
    cfg.additional_config = {"qwen4_exp": {"engram_config": {"cpu_offload": True}}}
    with pytest.raises(ValueError, match="host PLE capacity"):
        validate_config(cfg)


@pytest.mark.parametrize("across", [False, True])
def test_private_host_offload_rejects_unvalidated_multihost(monkeypatch, across):
    from vllm_torchtpu.models.vllm.qwen4_exp import config as admission

    monkeypatch.setattr(admission, "host_available_memory", lambda: 2 * 1024**4)
    cfg = config()
    cfg.parallel_config.nnodes = 2
    cfg.parallel_config.data_parallel_size_local = 4
    cfg.additional_config = {
        "qwen4_exp": {
            "engram_config": dict(cpu_offload=True, embedding_across_dp=across)
        }
    }
    with pytest.raises(ValueError, match="host PLE requires.*one host"):
        validate_config(cfg)


def test_ple_borrowed_groups_and_physical_slots():
    counts = [3, 0, 7, 1]
    slot = max(counts)
    ids = [torch.full((slot, 16), rank, dtype=torch.int32) for rank in range(4)]
    gathered = torch.cat(ids)
    raw = (
        torch.arange(4 * slot * 2560)
        .remainder(256)
        .to(torch.uint8)
        .view(torch.int8)
        .reshape(4 * slot, 2560)
    )
    for rank in range(4):
        tp = NS(world_size=1, rank_in_group=0, all_reduce=Mock(side_effect=lambda x: x))
        dp = NS(
            world_size=4,
            rank_in_group=rank,
            all_gather=Mock(return_value=gathered),
            all_reduce=Mock(return_value=raw),
        )
        etp = PLEParallel(tp, dp, embedding_across_dp=True)
        assert (etp.rank, etp.world_size) == (rank, 4)
        assert etp.gather_ids(ids[rank]) is gathered
        actual = etp.combine(torch.zeros_like(raw), slot)
        torch.testing.assert_close(
            actual, raw[rank * slot : (rank + 1) * slot], rtol=0, atol=0
        )
        assert actual.shape == (slot, 2560)  # Includes a completely idle DP.
        dp.all_gather.assert_called_once_with(ids[rank], dim=0)
        independent = PLEParallel(tp, dp, embedding_across_dp=False)
        assert independent.gather_ids(ids[rank]) is ids[rank]
        assert independent.combine(raw[:slot], slot).shape == (slot, 2560)
        assert independent.world_size == 1
    # DP1 must borrow TP for either configuration value, not EP by size.
    tp = NS(world_size=8, rank_in_group=7)
    dp = NS(world_size=1, rank_in_group=0)
    assert PLEParallel(tp, dp, embedding_across_dp=True).group is tp


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("across", [False, True])
def test_ple_owner_ranges_cover_each_table_once(tp, across):
    dp = 8 // tp
    rows = 8192
    replicas = [[] for _ in range(1 if across else dp)]
    for rank in range(8):
        parallel = PLEParallel(
            NS(world_size=tp, rank_in_group=rank % tp),
            NS(world_size=dp, rank_in_group=rank // tp),
            embedding_across_dp=across,
        )
        expected_rank, expected_size = (rank, 8) if across else (rank % tp, tp)
        assert (parallel.rank, parallel.world_size) == (expected_rank, expected_size)
        start = parallel.rank * rows // parallel.world_size
        stop = (parallel.rank + 1) * rows // parallel.world_size
        replicas[0 if across else rank // tp].extend(range(start, stop))
    assert all(owned_rows == list(range(rows)) for owned_rows in replicas)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("model_config", "dtype", torch.float32),
        ("model_config", "quantization", "nvfp4"),
        ("model_config", "max_model_len", 127),
        ("scheduler_config", "max_num_batched_tokens", 8193),
        ("scheduler_config", "max_num_seqs", 0),
        ("cache_config", "enable_prefix_caching", True),
        ("cache_config", "cache_dtype", "fp8_per_token_head"),
        ("cache_config", "block_size", 256),
    ],
)
def test_dp8_does_not_expand_other_admission_limits(monkeypatch, section, field, value):
    if field == "enable_prefix_caching":
        # Prefix reuse is admitted only with unified pool ownership.
        monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    cfg = config()
    setattr(getattr(cfg, section), field, value)
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("tokens", [129, 8192])
def test_dp8_accepts_admitted_prefill_token_budgets(tokens):
    cfg = config()
    cfg.scheduler_config.max_num_batched_tokens = tokens
    validate_config(cfg)


@pytest.mark.parametrize("tp,dp", [(1, 8), (2, 4), (4, 2), (8, 1)])
def test_non_spec_async_uses_the_same_admitted_topologies(tp, dp):
    cfg = config(tp, dp)
    cfg.scheduler_config.async_scheduling = True
    validate_config(cfg)


@pytest.mark.parametrize("architecture", ["Qwen4ExpForCausalLM", "Qwen3ForCausalLM"])
@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("across", [False, True])
def test_dp_collectives_follow_model_communication_not_only_ep(
    architecture, ep, across
):
    from vllm_torchtpu.models.parallel_requirements import requires_dp_collectives

    cfg = NS(
        parallel_config=NS(enable_expert_parallel=ep),
        model_config=NS(architecture=architecture),
        additional_config={
            "qwen4_exp": {"engram_config": {"embedding_across_dp": across}}
        },
    )
    assert requires_dp_collectives(cfg) is (
        ep or (architecture == "Qwen4ExpForCausalLM" and across)
    )


@pytest.mark.parametrize("tp", [2, 4])
def test_dp_shared_ple_keeps_one_world_with_ep_off(tp):
    from vllm_torchtpu.worker.tpu_rank_binding import get_tpu_worker_binding

    assignments = []
    for rank in range(8):
        cfg = NS(
            world_size=tp,
            local_world_size=tp,
            prefill_context_parallel_size=1,
            data_parallel_size=8 // tp,
            data_parallel_size_local=8 // tp,
            data_parallel_index=rank // tp,
            data_parallel_rank_local=rank // tp,
            enable_expert_parallel=False,
        )
        binding = get_tpu_worker_binding(
            cfg, rank % tp, rank % tp, env={}, dp_collectives=True
        )
        assignments.append(binding.native_local_rank)
        assert binding.rank == rank and binding.local_rank == rank
        assert binding.world_size == binding.local_world_size == 8
        assert binding.init_rank == rank % tp and binding.init_world_size == tp
        independent = get_tpu_worker_binding(cfg, rank % tp, rank % tp, env={})
        assert independent.world_size == tp and independent.rank == rank % tp
        assert independent.native_local_rank == rank
    assert assignments == list(range(8))


def test_admission_uses_model_relations_instead_of_checkpoint_literals():
    cfg = config()
    cfg.model_config.hf_text_config = q4exp_text_config(
        hidden_size=1536,
        hc_count=2,
        hc_lowrank=96,
        num_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=384,
        shared_expert_intermediate_size=256,
        vocab_size=65536,
        eos_token_id=7,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        linear_num_key_heads=8,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=3,
        indexer_n_heads=2,
        indexer_head_dim=64,
        indexer_budget=1024,
        indexer_compress_ratio=8,
        ple_embed_dim=1024,
        ple_conv_kernel_size=3,
        ple_layer_ids=[3],
        num_hidden_layers=8,
        layer_types=["linear_attention", "full_attention"] * 4,
    )
    validate_config(cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("moe_intermediate_size", 641),
        ("linear_num_key_heads", 15),
        ("num_attention_heads", 25),
        ("indexer_compress_ratio", 3),
        ("indexer_budget", 2049),
        ("ple_embed_dim", 2559),
    ],
)
def test_admission_rejects_inconsistent_model_geometry(field, value):
    cfg = config(2, 4)
    cfg.parallel_config.enable_expert_parallel = False
    setattr(cfg.model_config.hf_text_config, field, value)
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("slots", [65, 128, 256])
def test_request_capacity_is_not_limited_to_bringup_slots(slots):
    cfg = config()
    cfg.scheduler_config.max_num_seqs = slots
    validate_config(cfg)
