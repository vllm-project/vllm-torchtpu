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
"""Host preparation must satisfy vLLM's bucket and persistent-cache contracts."""

import copy
import dataclasses
import pickle
from types import SimpleNamespace as NS
from typing import cast

import pytest
import torch


def bare_config(config_type):
    # Exercise real dataclass hashing/serialization without downloading a model,
    # selecting a device, or running the unrelated model-admission hooks.
    cfg = object.__new__(config_type)
    for field in dataclasses.fields(config_type):
        setattr(cfg, field.name, None)
    return cfg


def launch_config(monkeypatch, instance, *, offload=True, engram=None, backend=None):
    from transformers import PretrainedConfig
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.config.aux_output import AuxOutputConfig

    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import config

    monkeypatch.setattr(
        config.Qwen4ExpForCausalLMConfig, "verify_and_update_config", lambda cfg: None
    )
    monkeypatch.setattr(config, "validate_config", lambda cfg: None)
    # Allocation capacity is unrelated to launch identity; retain real metadata.
    monkeypatch.setattr(config, "native_cache_capacity", lambda cfg, specs=None: 1)
    model = bare_config(ModelConfig)
    model.model, model.revision, model.max_model_len = "fixture", None, 8192
    model.hf_text_config = PretrainedConfig(**vars(q4exp_text_config()))
    model.dtype = torch.bfloat16
    cfg = NS(
        model_config=model,
        engram_config=engram,
        instance_id=instance,
        parallel_config=NS(data_parallel_size=8, tensor_parallel_size=1),
        scheduler_config=NS(max_num_seqs=8),
        compilation_config=NS(splitting_ops=None),
        cache_config=CacheConfig(
            block_size=128, enable_prefix_caching=False, mamba_cache_mode="none"
        ),
        additional_config={
            "qwen4_exp": {
                "engram_config": {
                    "cpu_offload": offload,
                    "dp_shared_memory": offload,
                    "embedding_across_dp": False,
                }
            }
        },
    )
    if backend is not None:
        cfg.additional_config["qwen4_exp"]["ple_host_backend"] = backend
    config.TPUQwen4ExpConfig.verify_and_update_config(cfg)
    hashed = bare_config(VllmConfig)
    hashed.aux_output_config = AuxOutputConfig()
    hashed.model_config = model
    hashed.engram_config = cfg.engram_config
    hashed.additional_config = cfg.additional_config
    hashed.observability_config = NS(compute_hash=lambda: "observability")
    return cfg, hashed


def test_launch_identity_keeps_compile_cache_stable_and_storage_distinct(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.ple_host import shared_ple_options

    first, first_hash = launch_config(monkeypatch, "service-a")
    second, second_hash = launch_config(monkeypatch, "service-b")
    assert first_hash.compute_hash() == second_hash.compute_hash()
    a = shared_ple_options(first.engram_config.host_launch, "ple", 0, 1)
    b = shared_ple_options(second.engram_config.host_launch, "ple", 0, 1)
    assert a.key != b.key and a.identity != b.identity
    before = second_hash.compute_hash()
    second_hash.additional_config["qwen4_exp"]["engram_config"].update(
        dp_shared_memory=False, embedding_across_dp=True
    )
    assert second_hash.compute_hash() != before


@pytest.mark.parametrize("transport", ["deepcopy", "pickle", "cloudpickle"])
def test_launch_identity_survives_config_transport_and_dp_collapse(
    monkeypatch, transport
):
    import cloudpickle

    from vllm_torchtpu.models.vllm.qwen4_exp.ple_host import shared_ple_options

    original, hashed = launch_config(monkeypatch, "service-a")
    if transport == "deepcopy":
        restored = copy.deepcopy(original)
    else:
        codec = pickle if transport == "pickle" else cloudpickle
        restored = codec.loads(codec.dumps(original))
    restored.instance_id += "_dp3"
    restored.parallel_config.data_parallel_size = 1
    options = shared_ple_options(restored.engram_config.host_launch, "ple", 0, 1)
    assert options.participants == 8
    assert options.identity["instance"] == "service-a"
    before = hashed.compute_hash()
    hashed.model_config = restored.model_config
    assert hashed.compute_hash() == before


@pytest.mark.parametrize("dp", [1, 2, 8])
@pytest.mark.parametrize("fp8", [False, True])
def test_host_boundary_preserves_bits_dp_axis_and_current_ids(monkeypatch, dp, fp8):
    from vllm.compilation.backends import split_graph
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbeddingShardIndices,
    )

    from vllm_torchtpu.models.vllm.qwen4_exp.ple_offload import host_lookup
    from vllm_torchtpu.models.vllm.qwen4_exp.ple_storage import HostPLETable

    # This test compiles a CPU FX graph; Dynamo must not acquire a TPU stream.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    width, heads, start, rows = 24, 3, 17, 11
    bits = (torch.arange(rows * width, device="cpu") * 73 + 159).reshape(rows, width)
    bits = bits.to(torch.int8 if fp8 else torch.int16)
    # Small host-table fixture: exercise the actual byte lookup and custom-op
    # boundary; constructor/shard loading have separate component coverage.
    embedding = HostPLETable.__new__(HostPLETable)
    torch.nn.Module.__init__(embedding)
    embedding._finalized = True
    embedding.storage_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    embedding.shard_indices = VocabParallelEmbeddingShardIndices(
        start,
        start + rows,
        0,
        0,
        start,
        start + rows,
        0,
        0,
    )
    embedding.embedding_dim = width
    embedding._shared_table = None
    embedding._gather_workers = 1
    embedding._gather_pool = None
    embedding.weight = torch.nn.Parameter(
        bits.view(embedding.storage_dtype), requires_grad=False
    )
    context = NS(
        no_compile_layers={
            "ple": NS(ple_embedding=NS(ngram_embedding=embedding)),
        }
    )
    boundaries = []

    def backend(graph, inputs):
        split, pieces = split_graph(graph, ["qwen4_exp::host_lookup"])
        boundaries.append(sum(piece.is_splitting_graph for piece in pieces))
        return split.forward

    def forward(tokens):
        ids = tokens[:, None].expand(-1, heads).repeat(dp, 1)
        return host_lookup(ids, tokens.shape[0], heads * width, dp, fp8, "ple")

    compiled = torch.compile(forward, backend=backend, fullgraph=True)
    try:
        for count in (7, 8, 16, 128):
            for shift in (0, 5):
                tokens = (
                    (torch.arange(count, device="cpu", dtype=torch.int32) + shift)
                    % (rows + 4)
                    + start
                    - 2
                )
                tokens[0] = -1
                torch._dynamo.mark_dynamic(tokens, 0)
                expected = torch.zeros(
                    (dp, count, heads, width), dtype=bits.dtype, device="cpu"
                )
                for i, token in enumerate(tokens.tolist()):
                    if start <= token < start + rows:
                        expected[:, i, :, :] = bits[token - start]
                with override_forward_context(cast(ForwardContext, context)):
                    actual = compiled(tokens)
                assert actual.shape == (dp, count, heads * width)
                assert actual.dtype == bits.dtype
                assert torch.equal(actual, expected.flatten(2))
        assert boundaries and all(count == 1 for count in boundaries)
    finally:
        torch._dynamo.reset()
        embedding.close()


@pytest.mark.parametrize("offload", [False, True])
def test_model_declares_host_boundary_only_for_offload(monkeypatch, offload):
    cfg, _ = launch_config(monkeypatch, "compile-boundary", offload=offload)
    assert cfg.compilation_config.splitting_ops == (
        ["qwen4_exp::host_lookup"] if offload else None
    )


@pytest.mark.parametrize(
    "cpu,across,shared,thp",
    [
        (False, True, False, False),
        (True, False, False, True),
        (True, False, None, False),
    ],
)
def test_typed_launch_preserves_existing_engram_policy_hash(
    monkeypatch, cpu, across, shared, thp
):
    from vllm.config.engram import EngramConfig

    from vllm_torchtpu.models.vllm.qwen4_exp.config import (
        Qwen4ExpEngramConfig,
        TPUQwen4ExpConfig,
    )

    original = EngramConfig(
        cpu_offload=cpu,
        embedding_across_dp=across,
        dp_shared_memory=shared,
        use_thp=thp,
    )
    cfg, hashed = launch_config(monkeypatch, "origin", engram=original)
    actual = cfg.engram_config
    assert isinstance(actual, Qwen4ExpEngramConfig)
    assert (
        actual.cpu_offload,
        actual.embedding_across_dp,
        actual.dp_shared_memory,
        actual.use_thp,
    ) == (cpu, across, shared, thp)
    assert actual.compute_hash() == original.compute_hash()
    descriptor = actual.host_launch
    assert (
        descriptor is not None
        and descriptor.instance == "origin"
        and descriptor.participants == 8
    )
    cfg.instance_id = "origin_dp3"
    cfg.parallel_config.data_parallel_size = 1
    TPUQwen4ExpConfig.verify_and_update_config(cfg)
    assert cfg.engram_config is actual and cfg.engram_config.host_launch is descriptor


def test_disabled_offload_keeps_absent_engram_configuration(monkeypatch):
    cfg, _ = launch_config(monkeypatch, "disabled", offload=False)
    assert cfg.engram_config is None


def test_native_boundary_and_cache_identity(monkeypatch):
    _, original = launch_config(monkeypatch, "original")
    cfg, native = launch_config(monkeypatch, "native", backend="native_dma")
    assert cfg.compilation_config.splitting_ops == ["qwen4_exp::host_lookup"]
    assert original.compute_hash() != native.compute_hash()


@pytest.mark.parametrize("configured", [None, "torch", "native_dma"])
@pytest.mark.parametrize("override", [None, "torch", "native_dma"])
def test_host_backend_env_resolves_before_hashing_and_transport(
    monkeypatch, configured, override
):
    name = "VLLM_TPU_QWEN38_FLASH_PLE_HOST_BACKEND"
    monkeypatch.delenv(name, raising=False)
    if override is not None:
        monkeypatch.setenv(name, override)
    cfg, hashed = launch_config(monkeypatch, "override", backend=configured)
    expected = override if override is not None else configured
    assert cfg.additional_config["qwen4_exp"].get("ple_host_backend") == expected
    # Workers receive the selected backend even without the driver's env.
    monkeypatch.delenv(name, raising=False)
    restored = pickle.loads(pickle.dumps(cfg))
    reference, reference_hash = launch_config(monkeypatch, "explicit", backend=expected)
    assert restored.additional_config == reference.additional_config
    assert hashed.compute_hash() == reference_hash.compute_hash()


@pytest.mark.parametrize("value", ["", "0", "invalid"])
def test_host_backend_env_rejects_invalid_selection(monkeypatch, value):
    monkeypatch.setenv("VLLM_TPU_QWEN38_FLASH_PLE_HOST_BACKEND", value)
    with pytest.raises(ValueError, match="VLLM_TPU_QWEN38_FLASH_PLE_HOST_BACKEND"):
        launch_config(monkeypatch, "invalid")


@pytest.mark.parametrize("backend", ["torch", "native_dma"])
@pytest.mark.parametrize("offload", [False, True])
def test_host_backend_env_preserves_offload_admission_and_dp_sharding(
    monkeypatch, backend, offload
):
    from tests.models.vllm.test_qwen4_exp_parallel import config
    from vllm_torchtpu.models.vllm.qwen4_exp.config import (
        Qwen4ExpForCausalLMConfig,
        TPUQwen4ExpConfig,
    )

    monkeypatch.setenv("VLLM_TPU_QWEN38_FLASH_PLE_HOST_BACKEND", backend)
    monkeypatch.setattr(
        Qwen4ExpForCausalLMConfig, "verify_and_update_config", lambda cfg: None
    )
    cfg = config()
    cfg.compilation_config = NS(splitting_ops=None)
    engram = dict(cpu_offload=offload, embedding_across_dp=True, dp_shared_memory=False)
    cfg.additional_config = {"qwen4_exp": {"engram_config": engram.copy()}}
    if backend == "native_dma" and not offload:
        with pytest.raises(ValueError, match="one host-offloaded PLE layer"):
            TPUQwen4ExpConfig.verify_and_update_config(cfg)
    else:
        TPUQwen4ExpConfig.verify_and_update_config(cfg)
        assert cfg.compilation_config.splitting_ops == (
            ["qwen4_exp::host_lookup"] if offload else None
        )
    assert cfg.additional_config["qwen4_exp"]["ple_host_backend"] == backend
    assert cfg.additional_config["qwen4_exp"]["engram_config"] == engram


@pytest.mark.parametrize("backend", ["raw_dma", "raw_dma_overlap", "native_dma"])
@pytest.mark.parametrize("offload,layers", [(False, [2]), (True, [1, 2]), (True, [2])])
def test_host_backend_admission(backend, offload, layers):
    from tests.models.vllm.test_qwen4_exp_parallel import config
    from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config

    cfg = config()
    cfg.model_config.hf_text_config.ple_layer_ids = layers
    cfg.additional_config = {
        "qwen4_exp": {
            "ple_host_backend": backend,
            "engram_config": {"cpu_offload": offload},
        }
    }
    if backend != "native_dma":
        with pytest.raises(ValueError, match="not a valid PLEHostBackend"):
            validate_config(cfg)
    elif offload and len(layers) == 1:
        validate_config(cfg)
    else:
        with pytest.raises(ValueError, match="one host-offloaded PLE layer"):
            validate_config(cfg)


@pytest.mark.parametrize("across", [False, True])
def test_native_host_dma_supports_dp_shards(across):
    from tests.models.vllm.test_qwen4_exp_parallel import config
    from vllm_torchtpu.models.vllm.qwen4_exp.config import validate_config

    cfg = config()
    cfg.additional_config = {
        "qwen4_exp": {
            "ple_host_backend": "native_dma",
            "engram_config": {"cpu_offload": True, "embedding_across_dp": across},
        }
    }
    validate_config(cfg)
