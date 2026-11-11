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
"""Model-owned immutable buffers must not alias mutable attention state."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn
from vllm.v1.kv_cache_interface import MambaSpec

from tests.models.vllm.test_qwen4_exp_parallel import config
from tests.runner.test_qwen4_exp_pool import pool_config
from vllm_torchtpu.models.vllm.qwen4_exp.config import (
    native_cache_specs,
    validate_config,
)
from vllm_torchtpu.models.vllm.qwen4_exp.mtp import (
    remap_mtp_weights,
)
from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_plan_includes_draft_and_ple_checkpoints(k):
    cfg = pool_config()
    cfg.cache_config.enable_prefix_caching = False
    cfg.speculative_config = NS(num_speculative_tokens=k)
    native = native_cache_specs(cfg)
    layout, pooled = pooled_cache_specs(cfg)
    assert set(pooled) == {n for n in native if not n.endswith(".compressed_k")}
    draft = "mtp.layers.48.self_attn"
    assert {draft, draft + ".compressed_k", draft + ".raw_ring"} <= native.keys()
    for name, spec in pooled.items():
        assert spec.page_size_bytes == layout.page_bytes
        if isinstance(spec, MambaSpec):
            assert spec.num_speculative_blocks == k
            assert (
                spec.max_num_blocks_per_req(cfg, cfg.model_config.max_model_len)
                == k + 1
            )
            if name.endswith(".ple"):
                text = cfg.model_config.hf_text_config
                assert spec.shapes == (
                    (
                        (text.ple_conv_kernel_size - 1) * text.ngram_size,
                        text.hidden_size * text.hc_count,
                    ),
                )
        elif name.endswith("raw_ring"):
            assert spec.block_size == 8
    assert len([n for n in pooled if n.endswith("raw_ring")]) == 13


@pytest.mark.parametrize("k", [1, 2, 4, 5])
@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("tp,dp,ep", [(1, 8, True), (2, 4, False)])
def test_mtp_admission(k, prefix, tp, dp, ep):
    cfg = config(tp, dp)
    cfg.parallel_config.enable_expert_parallel = ep
    cfg.cache_config.enable_prefix_caching = prefix
    cfg.cache_config.mamba_cache_mode = "align" if prefix else "none"
    cfg.speculative_config = NS(
        method="mtp", num_speculative_tokens=k, draft_tensor_parallel_size=tp
    )
    validate_config(cfg)
    cfg.speculative_config.method = "eagle3"
    with pytest.raises(ValueError, match="Qwen4Exp MTP requires"):
        validate_config(cfg)


@pytest.mark.parametrize("draft_tp", [None, 1, 2])
def test_mtp_requires_matching_target_and_draft_tp(draft_tp):
    cfg = config(tp=2, dp=4)
    cfg.speculative_config = NS(
        method="mtp", num_speculative_tokens=2, draft_tensor_parallel_size=draft_tp
    )
    if draft_tp == 1:
        with pytest.raises(ValueError, match="draft_tensor_parallel_size.*target"):
            validate_config(cfg)
    else:
        validate_config(cfg)


def test_validation_rpc_rejects_prefix_cached_engine_before_mutating_sampler(tmp_path):
    from tests.runner.qwen4_exp_mtp_validation import Qwen4ExpMTPValidationWorker

    sampler = NS(synthetic_mode=False, synthetic_conditional_rates=None)
    worker = Qwen4ExpMTPValidationWorker()
    worker.model_runner = NS(
        cache_config=NS(enable_prefix_caching=True), rejection_sampler=sampler
    )
    with pytest.raises(ValueError, match="disposable.*prefix caching disabled"):
        worker.validate_qwen4_exp_mtp(str(tmp_path))
    assert not sampler.synthetic_mode
    assert sampler.synthetic_conditional_rates is None
    assert not list(tmp_path.iterdir())


def test_mtp_checkpoint_namespace():
    weight = torch.empty(1)
    mapped = list(
        remap_mtp_weights(
            [
                ("mtp.layers.0.mlp.experts.12.up_proj.weight", weight),
                ("model.language_model.mtp.fc_hidden.weight", weight),
                ("model.layers.0.self_attn.q_proj.weight", weight),
            ]
        )
    )
    assert [name for name, _ in mapped] == [
        "model.layers.0.mlp.experts.12.up_proj.weight",
        "model.fc_hidden.weight",
    ]
    assert all(value is weight for _, value in mapped)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("steps", [1, 4])
def test_mtp_projection_normalizes_all_streams_and_shares_branch_weight(
    monkeypatch, compiled, steps
):
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    # Run the production forward with transparent decoder/mixer, isolating
    # the native fusion math from attention caches and distributed MoE.
    tokens, h, branches = 7, 16, 4
    torch.random.default_generator.manual_seed(72)

    class Norm(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(width) * 0.2)

        def forward(self, x):
            z = x.float()
            return (
                z
                * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)
                * (1 + self.weight)
            ).to(x.dtype)

    observed = {}

    class Decoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.self_attn = NS(prefix="mtp.qsa", config=NS(hc_count=branches))

        def forward(
            self,
            hidden,
            positions,
            ids,
            starts,
            ngrams,
            context,
            valid,
        ):
            observed.update(context=context, valid=valid)
            return hidden

    decoder = Decoder()
    mixer = NS(mix=lambda hidden: (hidden.view(tokens, branches, h).sum(1), None))

    obj = NS(
        hidden_size=h,
        hc_count=branches,
        attention_prefix="mtp.qsa",
        embed_tokens=nn.Embedding(32, h),
        pre_fc_norm_embedding=Norm(h),
        pre_fc_norm_hidden=Norm(h * branches),
        fc_embedding=nn.Linear(h, h, bias=False),
        fc_hidden=nn.Linear(h, h, bias=False),
        layers=[decoder],
        hyper_connection_mixer=mixer,
        num_mtp_layers=1,
    )
    md = NS(
        query_start_loc=torch.tensor([0, tokens]),
        seq_lens=torch.tensor([tokens]),
        block_tables=torch.tensor([1]),
    )
    context = NS(attn_metadata={"mtp.qsa": md, "mtp.qsa.compressed_k": md})
    monkeypatch.setattr(mtp, "get_forward_context", lambda: context)
    obj.embed_input_ids = obj.embed_tokens
    ids = torch.arange(tokens)
    hidden = torch.randn(tokens, h * branches)

    def forward(values):
        return mtp.Qwen4ExpMultiTokenPredictor.forward(obj, ids, ids, values)

    if compiled:
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    for _ in range(steps):
        logits_hidden, carry = forward(hidden)
        assert observed["context"].tolist() == [0]
        assert observed["valid"].all()
        x = hidden.double()
        n = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        n *= 1 + obj.pre_fc_norm_hidden.weight.double()
        e = obj.embed_tokens(ids).double()
        e = e / torch.sqrt(e.square().mean(-1, keepdim=True) + 1e-6)
        e *= 1 + obj.pre_fc_norm_embedding.weight.double()
        reference = torch.einsum(
            "tbh,oh->tbo", n.reshape(tokens, branches, h), obj.fc_hidden.weight.double()
        )
        reference += (e @ obj.fc_embedding.weight.double().T)[:, None, :]
        torch.testing.assert_close(
            carry.double(), reference.flatten(1), atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            logits_hidden.double(), reference.sum(1), atol=1e-6, rtol=1e-5
        )

        hidden = carry.detach()


def test_proposer_keeps_carry_and_lm_head_widths_separate():
    from tests.spec_decode.test_eagle3 import _make_proposer

    class Draft(nn.Module):
        config = NS(hidden_size=16)

    proposer = _make_proposer(method="mtp")
    proposer.draft_model = Draft()
    proposer._mtp_hc_mult = 4
    assert proposer._draft_hidden_size() == 16
    assert proposer._draft_input_hidden_size() == 64
    assert proposer._draft_combine_input_size() == 64


def test_mtp_fresh_quant_config_keeps_shared_expert_bf16(monkeypatch):
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
    )

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    quant = VllmFp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "modules_to_not_convert": [
                f"mtp.layers.0.mlp.shared_expert.{projection}"
                for projection in ("gate_proj", "up_proj", "down_proj")
            ],
        }
    )
    text = NS(
        num_hidden_layers=48,
        ple_layer_ids=[2],
        rope_parameters={"rope_theta": 10000},
        mtp={"rope_theta": 10000000},
    )
    cfg = NS(speculative_config=NS(draft_model_config=NS(hf_text_config=text)))
    monkeypatch.setattr(mtp, "get_tpu_quantization_config", lambda _: quant)
    draft = mtp.make_mtp_config(cfg)
    assert draft.model_config.hf_text_config.ple_layer_ids == []
    assert text.ple_layer_ids == [2]
    for projection in ("gate_up_proj", "down_proj"):
        assert is_layer_skipped(
            f"mtp.layers.48.mlp.shared_expert.{projection}",
            draft.quant_config.ignored_layers,
            draft.quant_config.packed_modules_mapping,
        )
    assert not is_layer_skipped(
        "mtp.layers.48.mlp.experts",
        draft.quant_config.ignored_layers,
        draft.quant_config.packed_modules_mapping,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_dummy_keeps_rpa_and_checkpoint_distributions_as_distinct_views(enabled):
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache

    cache = object.__new__(Qwen4ExpPooledCache)
    cfg = pool_config()
    cfg.cache_config.enable_prefix_caching = False
    cfg.speculative_config = NS(num_speculative_tokens=1) if enabled else None
    _, specs = pooled_cache_specs(cfg)
    state_spec = specs["model.layers.0.linear_attn"]
    cache.groups = [NS(kv_cache_spec=state_spec, layer_names=["gdn"])]
    cache.runner = NS(device="cpu", vllm_config=cfg, max_model_len=128)
    cache.pages = 32
    metadata = AttentionMetadata(
        input_positions=torch.arange(16),
        block_tables=torch.zeros(2, dtype=torch.int32),
        seq_lens=torch.tensor([16, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 16, 16], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    built = cache.dummy_metadata(metadata, 1)["gdn"]
    rpa = built.request_distribution
    mamba = built.mamba_request_distribution
    if not enabled:
        assert mamba is None
        assert rpa is metadata.request_distribution
        return
    assert rpa is not mamba
    assert rpa.untyped_storage().data_ptr() == mamba.untyped_storage().data_ptr()
    assert rpa.storage_offset() == 0 and mamba.storage_offset() == 3
    mamba[:2] = 1
    assert rpa.tolist() == [0, 0, 1]
    assert mamba.tolist() == [1, 1, 1]


def test_warmup_rejects_nonfinite_live_hidden_but_ignores_padding():
    from tests.runner.qwen4_exp_mtp_validation import validate_live_hidden

    hidden = torch.tensor([[1.0, 2.0], [float("nan"), float("inf")]])
    validate_live_hidden(hidden, 1)
    with pytest.raises(RuntimeError, match="non-finite target hidden"):
        validate_live_hidden(hidden, 2)


@pytest.mark.parametrize("state_name", ["linear_attn", "ple"])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_dummy_state_metadata_matches_serving_storage_contract(k, state_name):
    from vllm_torchtpu.layers.core.attention_metadata import (
        AttentionMetadata,
        _staged_walk_products,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache

    cache = object.__new__(Qwen4ExpPooledCache)
    cfg = pool_config()
    cfg.cache_config.enable_prefix_caching = False
    cfg.speculative_config = NS(num_speculative_tokens=k)
    _, specs = pooled_cache_specs(cfg)
    spec = next(spec for name, spec in specs.items() if name.endswith("." + state_name))
    columns = 1 + spec.num_speculative_blocks
    assert columns == k + 1
    cache.groups = [NS(kv_cache_spec=spec, layer_names=["state"])]
    cache.runner = NS(device="cpu", vllm_config=cfg, max_model_len=128)
    cache.pages = 64
    metadata = AttentionMetadata(
        input_positions=torch.arange(16),
        block_tables=torch.zeros(4 * columns, dtype=torch.int32),
        seq_lens=torch.tensor([8, 8, 0, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8, 16, 16, 16], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
    )
    dummy = cache.dummy_metadata(metadata, 2)["state"]
    serving = _staged_walk_products(
        dummy.block_tables,
        metadata.seq_lens,
        ((0, 4, columns),),
        ((0, columns, spec.block_size, None),),
        k + 1,
    )
    assert dummy.mamba_slot_read_offsets.shape == (cache.pages,)
    assert torch.count_nonzero(dummy.mamba_slot_read_offsets) == 0
    captured = (dummy.block_tables, dummy.mamba_state_indices)
    if columns > 1:
        captured += (dummy.mamba_ckpt_indices,)
    else:
        assert dummy.mamba_ckpt_indices is None
        serving = serving[:2]
    for actual, expected in zip(captured, serving):
        torch.testing.assert_close(actual, expected)
        assert actual.stride() == expected.stride()
    # AOT compilation can specialize aliases across graph inputs. The native
    # walk produces independent tables, state indices and checkpoint gathers.
    for values in (captured, serving):
        assert len({x.untyped_storage().data_ptr() for x in values}) == len(values)


def test_mtp_constructs_target_tpu_decoder(monkeypatch, vllm_config_context):
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.model_executor.layers import linear

    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    cfg.model_config = NS(
        hf_text_config=q4exp_text_config(
            hidden_size=16, vocab_size=32, hc_lowrank=4, tie_word_embeddings=False
        ),
        dtype=torch.float32,
        head_dtype=None,
    )
    monkeypatch.setattr(mtp, "make_mtp_config", lambda _: cfg)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )

    class Decoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.prefix = kwargs["prefix"] if "prefix" in kwargs else args[2]
            assert kwargs["layer_type"] == "indexed_attention"
            assert kwargs["hc_prefill_tile"] == 0
            self.mlp = NS(
                experts=nn.Identity(),
                n_logical_experts=8,
                n_physical_experts=8,
                n_local_physical_experts=1,
                n_routed_experts=8,
                n_redundant_experts=0,
            )

    from vllm.model_executor.layers import vocab_parallel_embedding as vocab

    monkeypatch.setattr(vocab, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mtp, "Qwen4ExpDecoderLayer", Decoder)
    with torch.device("cpu"):
        draft = mtp.Qwen4ExpMTP(vllm_config=cfg)
    assert isinstance(draft.model.layers[0], Decoder)
    assert draft.model.hyper_connection_mixer.prefill_full_tile == 0
    assert draft.model.layers[0].prefix == "mtp.layers.48"
    assert not draft.model.layers[0]._forward_pre_hooks
    assert not draft.model.layers[0]._forward_hooks
    loaded = {}

    def load(module, weights, layers, preloaded, *, block_size):
        parameters = dict(module.named_parameters())
        loaded.update(preloaded=preloaded, parameters=parameters)
        assert block_size == 128
        return set(parameters)

    monkeypatch.setattr(mtp, "load_qwen4_exp_weights", load)
    monkeypatch.setattr(mtp, "pooled_cache_specs", lambda _: (NS(block_size=128), {}))
    assert draft.load_weights([]) == set(dict(draft.named_parameters()))
    assert loaded["preloaded"] == {
        "model.embed_tokens.weight",
        "lm_head.weight",
    }
    with torch.no_grad():
        draft.model.embed_tokens.weight.normal_()
    ids = torch.tensor([0, 3, 7])
    torch.testing.assert_close(
        draft.embed_input_ids(ids), draft.model.embed_tokens(ids)
    )


@pytest.mark.parametrize(
    "prefix",
    ["", "model.", "language_model.", "model.language_model."],
)
def test_mtp_mapper_filters_target_and_preserves_weight_payload(prefix):
    weight = torch.randn(2, 2)
    mapped = list(
        remap_mtp_weights(
            [
                (prefix + "mtp.fc_hidden.weight", weight),
                (prefix + "layers.0.mlp.gate.weight", torch.zeros(1)),
                (prefix + "embed_tokens.weight", torch.zeros(1)),
                ("lm_head.weight", torch.zeros(1)),
            ]
        )
    )
    assert len(mapped) == 1
    assert mapped[0][0] == "model.fc_hidden.weight"
    assert mapped[0][1] is weight


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_mtp_admission_with_model_owned_pd(monkeypatch, k, role):
    for name in ("QWEN35", "KIMIK3", "GLM", "DSV4"):
        monkeypatch.setenv(f"TPU_RAIDEN_{name}_ADMISSION", "0")
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    cfg = config(tp=1, dp=8)
    cfg.speculative_config = NS(
        method="mtp", num_speculative_tokens=k, draft_tensor_parallel_size=1
    )
    cfg.kv_transfer_config = NS(
        kv_connector="Qwen4ExpRaidenConnector",
        kv_connector_module_path=None,
        kv_role=role,
    )
    cfg.cache_config.cache_dtype = "fp8_e4m3"
    cfg.cache_config.mamba_ssm_cache_dtype = "bfloat16"
    validate_config(cfg)
    cfg.kv_transfer_config.kv_role = "kv_both"
    with pytest.raises(ValueError, match="producer or consumer role"):
        validate_config(cfg)
