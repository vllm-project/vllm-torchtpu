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


def launch_config(monkeypatch, instance, *, offload=True):
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
    config.TPUQwen4ExpConfig.verify_and_update_config(cfg)
    hashed = bare_config(VllmConfig)
    hashed.aux_output_config = AuxOutputConfig()
    hashed.model_config = model
    hashed.additional_config = cfg.additional_config
    hashed.observability_config = NS(compute_hash=lambda: "observability")
    return cfg, hashed


def test_launch_identity_keeps_compile_cache_stable_and_storage_distinct(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.ple_host import shared_ple_options

    first, first_hash = launch_config(monkeypatch, "service-a")
    second, second_hash = launch_config(monkeypatch, "service-b")
    assert first_hash.compute_hash() == second_hash.compute_hash()
    a = shared_ple_options(first.model_config, "ple", 0, 1)
    b = shared_ple_options(second.model_config, "ple", 0, 1)
    assert a["key"] != b["key"] and a["identity"] != b["identity"]
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
    options = shared_ple_options(restored.model_config, "ple", 0, 1)
    assert options["participants"] == 8
    assert options["identity"]["instance"] == "service-a"
    before = hashed.compute_hash()
    hashed.model_config = restored.model_config
    assert hashed.compute_hash() == before


@pytest.mark.parametrize("dp", [1, 2, 8])
@pytest.mark.parametrize("fp8", [False, True])
def test_host_boundary_preserves_bits_dp_axis_and_current_ids(monkeypatch, dp, fp8):
    from vllm.compilation.backends import split_graph
    from vllm.forward_context import override_forward_context

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
    embedding.shard_indices = NS(
        org_vocab_start_index=start, org_vocab_end_index=start + rows
    )
    embedding.embedding_dim = width
    embedding._shared_table = None
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
                with override_forward_context(context):
                    actual = compiled(tokens)
                assert actual.shape == (dp, count, heads * width)
                assert actual.dtype == bits.dtype
                assert torch.equal(actual, expected.flatten(2))
        assert boundaries and all(count == 1 for count in boundaries)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("offload", [False, True])
def test_model_declares_host_boundary_only_for_offload(monkeypatch, offload):
    cfg, _ = launch_config(monkeypatch, "compile-boundary", offload=offload)
    assert cfg.compilation_config.splitting_ops == (
        ["qwen4_exp::host_lookup"] if offload else None
    )


def test_overlap_adds_its_split_points_only_when_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_PLE_HOST_OVERLAP", "1")
    cfg, _ = launch_config(monkeypatch, "compile-boundary-overlap", offload=True)
    assert cfg.compilation_config.splitting_ops == [
        "qwen4_exp::host_lookup",
        "qwen4_exp::host_lookup_start",
        "qwen4_exp::host_lookup_finish",
    ]
