# SPDX-License-Identifier: Apache-2.0
"""Host preparation must satisfy vLLM's bucket and persistent-cache contracts."""
import copy
import dataclasses
import pickle
from types import SimpleNamespace as NS

import pytest
import torch


def bare_config(config_type):
    # Exercise real dataclass hashing/serialization without downloading a model,
    # selecting a device, or running the unrelated model-admission hooks.
    cfg = object.__new__(config_type)
    for field in dataclasses.fields(config_type):
        setattr(cfg, field.name, None)
    return cfg


def launch_config(monkeypatch, instance):
    from vllm.config import ModelConfig, VllmConfig

    from vllm_torchtpu.models.vllm.qwen4_exp import config

    monkeypatch.setattr(config.Qwen4ExpForCausalLMConfig,
                        'verify_and_update_config', lambda cfg: None)
    monkeypatch.setattr(config, 'validate_config', lambda cfg: None)
    model = bare_config(ModelConfig)
    model.model, model.revision, model.max_model_len = 'fixture', None, 8192
    cfg = NS(model_config=model,
             instance_id=instance,
             parallel_config=NS(data_parallel_size=8, tensor_parallel_size=1),
             scheduler_config=NS(max_num_seqs=8),
             cache_config=NS(cache_dtype='fp8_per_token_head',
                             num_gpu_blocks_override=None),
             additional_config={
                 'qwen4_exp': {
                     'engram_config': {
                         'cpu_offload': True,
                         'dp_shared_memory': True,
                         'embedding_across_dp': False
                     }
                 }
             })
    config.TPUQwen4ExpConfig.verify_and_update_config(cfg)
    hashed = bare_config(VllmConfig)
    hashed.model_config = model
    hashed.additional_config = cfg.additional_config
    hashed.observability_config = NS(compute_hash=lambda: 'observability')
    return cfg, hashed


def test_launch_identity_keeps_compile_cache_stable_and_storage_distinct(
        monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.ple_host import shared_ple_options

    first, first_hash = launch_config(monkeypatch, 'service-a')
    second, second_hash = launch_config(monkeypatch, 'service-b')
    assert first_hash.compute_hash() == second_hash.compute_hash()
    a = shared_ple_options(first.model_config, 'ple', 0, 1)
    b = shared_ple_options(second.model_config, 'ple', 0, 1)
    assert a['key'] != b['key'] and a['identity'] != b['identity']
    before = second_hash.compute_hash()
    second_hash.additional_config['qwen4_exp']['engram_config'].update(
        dp_shared_memory=False, embedding_across_dp=True)
    assert second_hash.compute_hash() != before


@pytest.mark.parametrize('transport', ['deepcopy', 'pickle', 'cloudpickle'])
def test_launch_identity_survives_config_transport_and_dp_collapse(
        monkeypatch, transport):
    import cloudpickle

    from vllm_torchtpu.models.vllm.qwen4_exp.ple_host import shared_ple_options

    original, hashed = launch_config(monkeypatch, 'service-a')
    if transport == 'deepcopy':
        restored = copy.deepcopy(original)
    else:
        codec = pickle if transport == 'pickle' else cloudpickle
        restored = codec.loads(codec.dumps(original))
    restored.instance_id += '_dp3'
    restored.parallel_config.data_parallel_size = 1
    options = shared_ple_options(restored.model_config, 'ple', 0, 1)
    assert options['participants'] == 8
    assert options['identity']['instance'] == 'service-a'
    before = hashed.compute_hash()
    hashed.model_config = restored.model_config
    assert hashed.compute_hash() == before


@pytest.mark.parametrize('dp', [1, 2, 4, 8])
def test_host_input_retains_dp_axis_in_real_vllm_bucket_specialization(
        monkeypatch, dp):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from vllm.compilation.piecewise_backend import create_concrete_args
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.config import CompilationMode

    from vllm_torchtpu.models.vllm.qwen4_exp import model
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import Qwen4ExpPLE

    # Retain the actual model's compile decorator/argument annotations while
    # replacing only weight construction and compiler initialization.
    monkeypatch.setattr(model, 'VocabParallelEmbedding',
                        lambda *a: torch.nn.Identity())
    monkeypatch.setattr(model, 'GatedResidual',
                        lambda *a, **kw: torch.nn.Identity())
    monkeypatch.setattr(model, 'hc_config', lambda text: None)
    monkeypatch.setattr(TorchCompileWithNoGuardsWrapper, '__init__',
                        lambda *a, **kw: None)
    cfg = NS(model_config=NS(
        hf_text_config=NS(vocab_size=8, hidden_size=2, num_hidden_layers=0)),
             compilation_config=NS(mode=CompilationMode.VLLM_COMPILE))
    backbone = model.Qwen4ExpBackbone(cfg=cfg, vllm_config=cfg)
    token_count = 4
    ids = torch.zeros(token_count, dtype=torch.int32, device='cpu')
    rows = torch.arange(dp * token_count * 2560,
                        device='cpu').reshape(dp * token_count,
                                              2560).to(torch.int16)
    owner = NS(
        _host_ids=lambda *a: torch.zeros(
            (dp * token_count, 16), dtype=torch.int32, device='cpu'),
        ple_embedding=NS(ngram_embedding=NS(lookup_host=lambda ids: rows)))
    prepared = Qwen4ExpPLE.prepare_host(owner, ids, None, None, None)
    axis = backbone._dynamic_arg_dims['ple_raw_embeddings']
    shape_env = ShapeEnv()
    shape = list(prepared.shape)
    shape[axis] = shape_env.create_unbacked_symint()
    with FakeTensorMode(shape_env=shape_env):
        fake = torch.empty(shape, dtype=prepared.dtype, device='cpu')
    graph = torch.fx.Graph()
    arg = graph.placeholder('ple_raw_embeddings')
    arg.meta['example_value'] = fake
    graph.output(arg)
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    for bucket in (1, 7, 16, 128):
        concrete, = create_concrete_args(module, bucket)
        # This last DP rank ran beyond the incorrectly specialized T-row
        # argument before the static DP axis was preserved.
        local = concrete.reshape(-1, 2560).narrow(0, (dp - 1) * bucket, bucket)
        assert local.shape == (bucket, 2560)
        assert concrete.shape == (dp, bucket, 2560)
    assert torch.equal(prepared.reshape(-1, 2560), rows)
