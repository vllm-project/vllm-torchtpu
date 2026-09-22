# SPDX-License-Identifier: Apache-2.0
"""Admitted topology, physical DP slots and checkpoint ownership contracts."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config
from vllm_torchtpu.models.vllm.qwen4_exp.parallel import PLEParallel


def config(tp=1, dp=8):
    text = NS(hidden_size=2560,
              num_hidden_layers=48,
              hc_count=4,
              hc_lowrank=320,
              vocab_size=248320,
              num_attention_heads=24,
              num_key_value_heads=2,
              head_dim=256,
              num_experts=512,
              num_experts_per_tok=10,
              moe_intermediate_size=640,
              shared_expert_intermediate_size=640,
              linear_num_key_heads=16,
              linear_num_value_heads=48,
              linear_key_head_dim=128,
              linear_value_head_dim=128,
              linear_conv_kernel_dim=4,
              output_gate_type='sigmoid',
              indexer_n_heads=4,
              indexer_kv_heads=1,
              indexer_head_dim=128,
              indexer_budget=2048,
              indexer_compress_ratio=4,
              ple_layer_ids=[2],
              ple_embed_dim=2560,
              ple_conv_kernel_size=4,
              ngram_size=3,
              heads_per_ngram=8,
              ngram_vocab_size_base=20000000,
              make_ngram_vocab_size_divisible_by=128,
              eos_token_id=248044,
              rms_norm_eps=1e-6,
              hidden_act='silu',
              partial_rotary_factor=.25,
              rope_parameters={'rope_theta': 10000000},
              layer_types=[
                  'full_attention' if i % 4 == 3 else 'linear_attention'
                  for i in range(48)
              ])
    return NS(model_config=NS(hf_text_config=text,
                              hf_config=NS(
                                  quantization_config={
                                      'quant_method': 'fp8',
                                      'weight_block_size': [128, 128],
                                      'activation_scheme': 'dynamic'
                                  }),
                              quantization='fp8',
                              dtype=torch.bfloat16,
                              max_model_len=8192),
              parallel_config=NS(tensor_parallel_size=tp,
                                 data_parallel_size=dp,
                                 enable_expert_parallel=True,
                                 pipeline_parallel_size=1,
                                 prefill_context_parallel_size=1,
                                 decode_context_parallel_size=1,
                                 enable_dbo=False,
                                 enable_eplb=False,
                                 use_sequence_parallel_moe=False,
                                 expert_placement_strategy='linear',
                                 enable_elastic_ep=False),
              cache_config=NS(cache_dtype='auto',
                              block_size=128,
                              enable_prefix_caching=False,
                              kv_cache_memory_bytes=None),
              scheduler_config=NS(max_num_batched_tokens=128,
                                  max_num_seqs=64,
                                  async_scheduling=False,
                                  disable_hybrid_kv_cache_manager=False),
              speculative_config=None,
              kv_transfer_config=None,
              lora_config=None,
              additional_config={})


@pytest.mark.parametrize('tp,dp', [(1, 8), (2, 4), (4, 2), (8, 1)])
@pytest.mark.parametrize('across', [False, True])
def test_admitted_topologies_and_engram_modes(tp, dp, across):
    cfg = config(tp, dp)
    validate_config(
        cfg)  # Missing embedding_across_dp keeps the upstream default.
    cfg.additional_config = {
        'qwen4_exp': {
            'engram_config': {
                'cpu_offload': False,
                'embedding_across_dp': across,
                'dp_shared_memory': False
            }
        }
    }
    validate_config(cfg)


@pytest.mark.parametrize('tp,dp', [(2, 2), (4, 4)])
@pytest.mark.parametrize('across', [False, True])
def test_topologies_requiring_fewer_or_more_than_eight_devices_are_rejected(
        tp, dp, across):
    cfg = config(tp, dp)
    cfg.additional_config = {
        'qwen4_exp': {
            'engram_config': {
                'embedding_across_dp': across
            }
        }
    }
    with pytest.raises(ValueError, match='TP8/DP1'):
        validate_config(cfg)


@pytest.mark.parametrize(
    'flag', ['cpu_offload', 'embedding_across_dp', 'dp_shared_memory'])
@pytest.mark.parametrize('value', ['false', 0, None])
def test_engram_flags_reject_truthiness(flag, value):
    cfg = config()
    cfg.additional_config = {'qwen4_exp': {'engram_config': {flag: value}}}
    with pytest.raises(ValueError, match=f'{flag} must be bool'):
        validate_config(cfg)


@pytest.mark.parametrize('field,value',
                         [('tensor_parallel_size', 2),
                          ('data_parallel_size', 1),
                          ('enable_expert_parallel', False),
                          ('expert_placement_strategy', 'round_robin'),
                          ('enable_elastic_ep', True), ('enable_eplb', True),
                          ('enable_dbo', True), ('pipeline_parallel_size', 2),
                          ('prefill_context_parallel_size', 2),
                          ('decode_context_parallel_size', 2)])
def test_unsupported_parallel_modes_are_rejected(field, value):
    cfg = config()
    setattr(cfg.parallel_config, field, value)
    with pytest.raises(ValueError, match='requires TP8'):
        validate_config(cfg)


@pytest.mark.parametrize('tp,dp', [(2, 4), (4, 2)])
def test_model_admission_precedes_tpu_sequence_parallel_normalization(tp, dp):
    cfg = config(tp, dp)
    # vLLM derives this property for TP*DP EP before platform_activation
    # disables it. It is not a user-requested model capability.
    cfg.parallel_config.use_sequence_parallel_moe = True
    validate_config(cfg)


@pytest.mark.parametrize('flag', ['cpu_offload', 'dp_shared_memory'])
def test_host_flags_are_not_resident_sharding_switches(flag):
    cfg = config()
    cfg.additional_config = {'qwen4_exp': {'engram_config': {flag: True}}}
    with pytest.raises(ValueError, match='host offload/shared memory'):
        validate_config(cfg)


def test_ple_borrowed_groups_and_physical_slots():
    counts = [3, 0, 7, 1]
    slot = max(counts)
    ids = [
        torch.full((slot, 16), rank, dtype=torch.int32) for rank in range(4)
    ]
    gathered = torch.cat(ids)
    raw = torch.arange(4 * slot * 2560).remainder(256).to(torch.uint8).view(
        torch.int8).reshape(4 * slot, 2560)
    for rank in range(4):
        tp = NS(world_size=1,
                rank_in_group=0,
                all_reduce=Mock(side_effect=lambda x: x))
        dp = NS(world_size=4,
                rank_in_group=rank,
                all_gather=Mock(return_value=gathered),
                all_reduce=Mock(return_value=raw))
        etp = PLEParallel(tp, dp, embedding_across_dp=True)
        assert (etp.rank, etp.world_size) == (rank, 4)
        assert etp.gather_ids(ids[rank]) is gathered
        actual = etp.combine(torch.zeros_like(raw), slot)
        torch.testing.assert_close(actual,
                                   raw[rank * slot:(rank + 1) * slot],
                                   rtol=0,
                                   atol=0)
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


@pytest.mark.parametrize('tp', [1, 2, 4, 8])
@pytest.mark.parametrize('across', [False, True])
def test_ple_owner_ranges_cover_each_table_once(tp, across):
    dp = 8 // tp
    rows = 8192
    replicas = [[] for _ in range(1 if across else dp)]
    for rank in range(8):
        parallel = PLEParallel(NS(world_size=tp, rank_in_group=rank % tp),
                               NS(world_size=dp, rank_in_group=rank // tp),
                               embedding_across_dp=across)
        expected_rank, expected_size = ((rank, 8) if across else
                                        (rank % tp, tp))
        assert (parallel.rank, parallel.world_size) == (expected_rank,
                                                        expected_size)
        start = parallel.rank * rows // parallel.world_size
        stop = (parallel.rank + 1) * rows // parallel.world_size
        replicas[0 if across else rank // tp].extend(range(start, stop))
    assert all(owned_rows == list(range(rows)) for owned_rows in replicas)


@pytest.mark.parametrize('tp,dp,rank,across',
                         [(1, 8, 7, True), (1, 8, 7, False), (2, 4, 7, True),
                          (2, 4, 7, False), (4, 2, 7, True), (4, 2, 7, False),
                          (8, 1, 7, False)])
def test_model_uses_true_tp_ep_and_embedding_geometry(monkeypatch, tp, dp,
                                                      rank, across):
    from vllm_torchtpu.models.vllm.qwen4_exp import model
    cfg = config(tp, dp)
    cfg.additional_config = {
        'qwen4_exp': {
            'engram_config': {
                'embedding_across_dp': across
            }
        }
    }
    cfg.device_config = NS(device='cpu')
    cfg.compilation_config = NS(static_forward_context={})
    monkeypatch.setattr(model, 'get_tp_group',
                        lambda: NS(world_size=tp, rank_in_group=rank % tp))
    monkeypatch.setattr(model, 'get_dp_group',
                        lambda: NS(world_size=dp, rank_in_group=rank // tp))
    monkeypatch.setattr(model, 'get_ep_group',
                        lambda: NS(world_size=8, rank_in_group=rank))
    qsa, ple, moe = Mock(side_effect=lambda **kw: nn.Identity()), Mock(
        side_effect=lambda **kw: nn.Identity()), Mock(
            side_effect=lambda **kw: nn.Identity())
    monkeypatch.setattr(model, 'Qwen4ExpQSA', qsa)
    monkeypatch.setattr(model, 'Qwen4ExpPLE', ple)
    monkeypatch.setattr(model, 'Qwen4ExpSparseMoeBlock', moe)
    monkeypatch.setattr(model, 'Qwen4ExpGatedDeltaNetAttention',
                        lambda *a, **kw: nn.Identity())
    monkeypatch.setattr(model, 'GatedResidual', lambda *a, **kw: nn.Identity())
    model.Qwen4ExpDecoderLayer(cfg, 3, 'model.layers.3')
    assert qsa.call_args.kwargs['tp_size'] == tp
    assert qsa.call_args.kwargs['tp_rank'] == rank % tp
    assert moe.call_args.kwargs == {'ep_rank': rank, 'ep_size': 8}
    model.Qwen4ExpDecoderLayer(cfg, 1, 'model.layers.1')
    parallel = ple.call_args.kwargs['parallel']
    assert (parallel.rank, parallel.world_size) == ((rank, 8) if across else
                                                    (rank % tp, tp))


@pytest.mark.parametrize('rank', [0, 7])
def test_model_loader_accepts_only_owned_experts_and_rejects_missing_owned(
        rank):
    from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpForCausalLM
    module = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    nn.Module.__init__(module)
    expert = NS(ep_rank=rank,
                _loaded=set(),
                _pending={},
                load_weights=Mock(),
                finalize_weights=Mock())
    layer = NS(mlp=expert,
               ple=None,
               layer_type='full_attention',
               self_attn=NS(finalize_weights=Mock()))
    module.model = NS(layers=[layer])
    names = [
        f'experts.{e}.{p}.{field}' for e in range(rank * 64, (rank + 1) * 64)
        for p in ('gate_proj', 'up_proj', 'down_proj')
        for field in ('weight', 'weight_scale_inv')
    ]
    names += [
        'gate.weight', 'shared_expert_gate.weight',
        'shared_expert.gate_proj.weight', 'shared_expert.up_proj.weight',
        'shared_expert.down_proj.weight'
    ]
    weights = [(f'model.language_model.layers.0.mlp.{name}', torch.empty(0))
               for name in names]
    assert module.load_weights(iter(weights)) == set()
    expert.finalize_weights.assert_called_once()
    with pytest.raises(ValueError, match='expert checkpoint: 1 missing'):
        module.load_weights(iter(weights[1:]))


@pytest.mark.parametrize('section,field,value', [
    ('model_config', 'dtype', torch.float32),
    ('model_config', 'quantization', 'nvfp4'),
    ('model_config', 'max_model_len', 127),
    ('model_config', 'max_model_len', 8193),
    ('scheduler_config', 'max_num_batched_tokens', 129),
    ('scheduler_config', 'max_num_seqs', 65),
    ('scheduler_config', 'async_scheduling', True),
    ('cache_config', 'enable_prefix_caching', True),
    ('cache_config', 'cache_dtype', 'fp8'),
    ('cache_config', 'block_size', 256),
])
def test_dp8_does_not_expand_other_admission_limits(section, field, value):
    cfg = config()
    setattr(getattr(cfg, section), field, value)
    with pytest.raises(ValueError):
        validate_config(cfg)
