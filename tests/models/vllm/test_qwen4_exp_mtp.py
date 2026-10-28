# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp MTP admission, cache ownership, checkpoint names and rollback."""

from types import SimpleNamespace as NS

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tests.models.vllm.test_qwen4_exp_parallel import config
from vllm_torchtpu.models.vllm.qwen4_exp.config import (
    max_qsa_speculative_tokens,
    mtp_layer_prefixes,
    native_cache_specs,
    validate_config,
)

# Distinct non-expert MTP tensors of Qwen/Qwen3.8-Flash-Next-FP8
# (model.safetensors.index.json); routed experts follow
# mtp.layers.0.mlp.experts.<e>.{gate,up,down}_proj.{weight,weight_scale_inv}.
CHECKPOINT_MTP_TENSORS = [
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]


def mtp_config(k=3, tp=8):
    cfg = config(tp=tp, dp=1)
    text = cfg.model_config.hf_text_config
    text.mtp_num_hidden_layers = 1
    text.mtp_use_dedicated_embeddings = False
    text.mtp = {
        "hybrid": True,
        "layer_types": ["full_attention"],
        "num_hidden_layers": 1,
    }
    cfg.cache_config.mamba_cache_mode = "align"
    cfg.speculative_config = spec_config(k)
    return cfg


def spec_config(k):
    return NS(
        method="mtp",
        num_speculative_tokens=k,
        draft_tensor_parallel_size=None,
        use_eagle=lambda: True,
        use_eagle_block_drop=lambda: False,
    )


@pytest.fixture
def pooled(monkeypatch):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")


@pytest.mark.usefixtures("pooled")
@pytest.mark.parametrize("k", [1, 3, 5])
def test_admits_native_mtp_on_unified_pool(k):
    validate_config(mtp_config(k))


@pytest.mark.usefixtures("pooled")
@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda c: setattr(c.speculative_config, "method", "eagle3"), "method='mtp'"),
        (lambda c: setattr(c.speculative_config, "num_speculative_tokens", 6), "QSA"),
        (
            lambda c: setattr(c.speculative_config, "draft_tensor_parallel_size", 1),
            "draft_tensor_parallel_size",
        ),
        (lambda c: setattr(c.parallel_config, "data_parallel_size", 2), "data_par"),
        (
            lambda c: setattr(
                c.model_config.hf_text_config, "mtp_use_dedicated_embeddings", True
            ),
            "shares the target embeddings",
        ),
        (
            lambda c: setattr(
                c.model_config.hf_text_config, "mtp_num_hidden_layers", 2
            ),
            "exactly one",
        ),
        (lambda c: setattr(c.cache_config, "mamba_cache_mode", "all"), "mamba_cache"),
    ],
)
def test_rejects_unsupported_mtp(mutate, error):
    cfg = mtp_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=error):
        validate_config(cfg)


def test_rejects_mtp_without_unified_pool(monkeypatch):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    cfg = mtp_config()
    cfg.cache_config.enable_prefix_caching = False
    with pytest.raises(ValueError, match="unified block pool"):
        validate_config(cfg)


def test_ring_bound_matches_compression_geometry():
    cfg = mtp_config()
    # Eight ring rows, compression ratio four: windows of up to six tokens.
    assert max_qsa_speculative_tokens(cfg.model_config.hf_text_config) == 5


def test_specs_add_mtp_qsa_owner_and_state_checkpoints():
    from tests.runner.test_qwen4_exp_pool import pool_config

    cfg = pool_config(tp=8)
    cfg.model_config.hf_text_config.mtp_num_hidden_layers = 1
    plain_specs = native_cache_specs(cfg)
    cfg.speculative_config = spec_config(3)
    specs = native_cache_specs(cfg)
    assert mtp_layer_prefixes(cfg) == ["mtp.layers.48"]
    base = "mtp.layers.48.self_attn"
    assert {base, base + ".compressed_k", base + ".raw_ring"} <= specs.keys()
    assert specs[base] == specs["model.layers.3.self_attn"]
    assert specs[base + ".raw_ring"] == specs["model.layers.3.self_attn.raw_ring"]
    assert specs["model.layers.0.linear_attn"].num_speculative_blocks == 3
    assert specs["model.layers.1.ple"].num_speculative_blocks == 3

    assert not any(name.startswith("mtp.") for name in plain_specs)
    assert plain_specs["model.layers.0.linear_attn"].num_speculative_blocks == 0
    assert plain_specs["model.layers.1.ple"].num_speculative_blocks == 0


@pytest.mark.parametrize("prefix", [False, True])
def test_pool_plan_covers_mtp_layers_with_uniform_pages(prefix):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    from vllm.v1.kv_cache_interface import MambaSpec

    from tests.runner.test_qwen4_exp_pool import pool_config
    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

    cfg = pool_config(tp=8)
    cfg.cache_config.enable_prefix_caching = prefix
    cfg.cache_config.mamba_cache_mode = "align" if prefix else "none"
    cfg.model_config.hf_text_config.mtp_num_hidden_layers = 1
    cfg.speculative_config = spec_config(3)
    layout, specs = pooled_cache_specs(cfg)
    assert {spec.page_size_bytes for spec in specs.values()} == {layout.page_bytes}
    groups = get_kv_cache_groups(cfg, specs)
    names = {name for group in groups for name in group.layer_names}
    assert names == specs.keys()
    for group in groups:
        if isinstance(group.kv_cache_spec, MambaSpec):
            assert group.kv_cache_spec.num_speculative_blocks == 3


def test_checkpoint_names_map_onto_draft_modules():
    from vllm_torchtpu.models.vllm.qwen4_exp.mtp import mtp_checkpoint_name

    assert mtp_checkpoint_name("model.language_model.embed_tokens.weight") is None
    assert mtp_checkpoint_name("lm_head.weight") is None
    assert mtp_checkpoint_name("model.language_model.layers.3.mlp.gate.weight") is None
    mapped = [mtp_checkpoint_name(name) for name in CHECKPOINT_MTP_TENSORS]
    assert all(name.startswith("model.") for name in mapped)
    assert mapped[0] == "model.fc_embedding.weight"
    assert (
        mtp_checkpoint_name("mtp.layers.0.mlp.experts.7.up_proj.weight_scale_inv")
        == "model.layers.0.mlp.experts.7.up_proj.weight_scale_inv"
    )


def test_draft_quant_config_renames_only_mtp_layers():
    from vllm_torchtpu.models.vllm.qwen4_exp.mtp import draft_quant_config

    ignored = [
        "lm_head",
        "model.layers.3.mlp.gate",
        "mtp.fc_embedding",
        "mtp.layers.0.mlp.shared_expert.gate_proj",
        "mtp.layers.0.self_attn.q_proj",
    ]
    target = NS(ignored_layers=ignored, weight_block_size=[128, 128])
    draft = draft_quant_config(target, 48)
    assert target.ignored_layers is ignored and ignored[3].startswith("mtp.layers.0.")
    assert draft.ignored_layers == [
        "lm_head",
        "model.layers.3.mlp.gate",
        "mtp.fc_embedding",
        "mtp.layers.48.mlp.shared_expert.gate_proj",
        "mtp.layers.48.self_attn.q_proj",
    ]
    assert draft.weight_block_size == [128, 128]
    assert draft_quant_config(None, 48) is None


@pytest.fixture
def cpu_vllm_config():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


@pytest.mark.usefixtures("cpu_vllm_config")
def test_predictor_fuses_upstream_inputs_and_returns_both_streams(monkeypatch):
    from torch import nn

    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    text = q4exp_text_config(
        hidden_size=8, hc_count=4, hc_lowrank=4, vocab_size=16, num_hidden_layers=2
    )
    seen = {}

    class Layer(nn.Module):
        def __init__(self, cfg, index, prefix, *, layer_type):
            super().__init__()
            assert (index, prefix, layer_type) == (2, "mtp.layers.2", "full_attention")
            self.self_attn = NS(prefix=prefix + ".self_attn")

        def forward(
            self,
            hidden,
            positions,
            input_ids,
            starts,
            ngram,
            slots,
            context,
            valid,
            main,
            compressed,
        ):
            seen.update(context=context, valid=valid, positions=positions)
            return hidden * 2

    monkeypatch.setattr(mtp, "Qwen4ExpDecoderLayer", Layer)
    monkeypatch.setattr(mtp, "VocabParallelEmbedding", nn.Embedding)
    for module in ("parameter", "layers.linear"):
        for name, value in (
            ("get_tensor_model_parallel_rank", 0),
            ("get_tensor_model_parallel_world_size", 1),
        ):
            monkeypatch.setattr(
                f"vllm.model_executor.{module}.{name}",
                lambda value=value: value,
                raising=False,
            )
    cfg = NS(model_config=NS(hf_text_config=text), additional_config={})
    with torch.device("cpu"):
        predictor = mtp.Qwen4ExpMultiTokenPredictor(cfg, ["mtp.layers.2"])
    # The draft serves every token bucket from one graph.
    assert predictor.layers[0].self_attn.single_graph
    predictor = predictor.to(torch.bfloat16)
    for p in predictor.parameters():
        p.data.normal_(0, 0.2)
    ids = torch.tensor([3, 5, 7])
    target = torch.randn(3, 32).to(torch.bfloat16)

    def rms(x, w):
        x = x.float()
        normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        return normed * (1 + w.float())

    emb = rms(predictor.embed_tokens(ids), predictor.pre_fc_norm_embedding.weight)
    emb = emb @ predictor.fc_embedding.weight.float().T
    # One RMSNorm over all hc*H elements, then the shared per-stream fc.
    hid = rms(target, predictor.pre_fc_norm_hidden.weight).reshape(3, 4, 8)
    hid = hid @ predictor.fc_hidden.weight.float().T
    expected = (hid + emb[:, None]).reshape(3, 32)
    fused = predictor.fuse_inputs(ids, target)
    torch.testing.assert_close(fused.float(), expected, rtol=3e-2, atol=3e-2)
    expected = fused

    metadata = NS(
        query_start_loc=torch.tensor([0, 2, 2, 2]),
        seq_lens=torch.tensor([9, 5, 0]),
        block_tables=torch.zeros(3, dtype=torch.int32),
    )
    monkeypatch.setattr(
        mtp,
        "get_forward_context",
        lambda: NS(attn_metadata={"mtp.layers.2.self_attn": metadata}),
    )
    sample, multi = predictor(ids, torch.tensor([7, 8, 0]), target, None, 0)
    torch.testing.assert_close(multi, expected * 2)
    assert sample.shape == (3, 8)
    torch.testing.assert_close(
        sample, predictor.hyper_connection_mixer.mix(expected * 2)[0]
    )
    assert seen["context"].tolist() == [7, 5, 0]
    assert seen["valid"].tolist() == [True, True, False]
    assert seen["positions"].dtype == torch.int64


def test_ple_checkpoint_addressing_follows_read_offsets():
    from vllm_torchtpu.models.vllm.qwen4_exp.model import ple_speculative_state

    assert ple_speculative_state(NS(mamba_ckpt_indices=None)) is None
    offsets = torch.zeros(32, dtype=torch.int32)
    offsets[10], offsets[20] = 2, 3
    read, ckpt, windowed = ple_speculative_state(
        NS(
            mamba_ckpt_indices=torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23], [5, 6, 7, 8], [0, 0, 0, 0]]
            ),
            mamba_state_indices=torch.tensor([10, 20, 5, 0]),
            mamba_slot_read_offsets=offsets,
            mamba_request_distribution=torch.tensor([2, 2, 3]),
        )
    )
    assert read.tolist() == [12, 23, 5, 0]
    assert ckpt.dtype == torch.int32
    assert windowed.tolist() == [True, True, False, False]


def _short_conv_reference(history, x, state_len):
    """Window carried after appending ``x`` to ``history`` (oldest first)."""
    return np.concatenate([history, x])[-state_len:]


@pytest.mark.parametrize("length", [1, 2, 4])
def test_per_position_ple_windows_match_truncated_chunks(length):
    from vllm_torchtpu.kernels.qwen4_exp.pool import speculative_history_windows

    rng = np.random.default_rng(length)
    state_len, channels, window = 9, 3, 4
    history = rng.normal(size=(2, state_len, channels)).astype(np.float32)
    x = rng.normal(size=(length + 3, channels)).astype(np.float32)
    starts = jnp.asarray([0, length, length + 3], jnp.int32)
    windows = np.asarray(
        speculative_history_windows(
            jnp.asarray(x),
            jnp.asarray(history),
            starts,
            jnp.asarray([True, False]),
            window,
        )
    )
    for t in range(min(length, window)):
        np.testing.assert_array_equal(
            windows[0, t], _short_conv_reference(history[0], x[: t + 1], state_len)
        )
    # Without initial state the pre-chunk history reads as zeros.
    for t in range(3):
        expected = _short_conv_reference(
            np.zeros_like(history[1]), x[length : length + t + 1], state_len
        )
        np.testing.assert_array_equal(windows[1, t], expected)


def test_ple_rollback_selects_last_accepted_window():
    """Selecting checkpoint ``accepted`` equals never seeing rejected drafts."""
    from vllm_torchtpu.kernels.qwen4_exp.pool import speculative_history_windows

    rng = np.random.default_rng(7)
    state_len, channels, k = 9, 2, 3
    truth = rng.normal(size=(64, channels)).astype(np.float32)
    history = np.zeros((state_len, channels), np.float32)
    clean = np.zeros((state_len, channels), np.float32)
    context, initial = 0, False
    while context + k + 1 < len(truth):
        accepted = int(rng.integers(0, k + 1))
        chunk = truth[context : context + k + 1].copy()
        chunk[accepted + 1 :] += 100.0  # rejected drafts
        windows = np.asarray(
            speculative_history_windows(
                jnp.asarray(chunk),
                jnp.asarray(history[None]),
                jnp.asarray([0, k + 1], jnp.int32),
                jnp.asarray([initial]),
                k + 1,
            )
        )
        history = windows[0, accepted]
        clean = _short_conv_reference(
            clean, truth[context : context + accepted + 1], state_len
        )
        np.testing.assert_array_equal(history, clean)
        context += accepted + 1
        initial = True


def test_ngram_context_reads_device_verified_tokens():
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import (
        merge_recent_ngram_context,
    )

    # Previous step: request at row 1 had context 10 (tokens 8, 9 before it)
    # and verified [b, d1, d2, d3] = [50, 51, 52, 53] at positions 10..13.
    tokens = torch.zeros((3, 6), dtype=torch.int64)
    tokens[1] = torch.tensor([8, 9, 50, 51, 52, 53])
    recent_context = torch.tensor([0, 10, 0])
    lengths = torch.tensor([0, 4, 0])
    # The host store still holds placeholders (-1) past position 10.
    for accepted, expected in [(0, [9, 50]), (1, [50, 51]), (2, [51, 52])]:
        context = torch.tensor([10 + accepted + 1, 4])
        host = torch.tensor([[-1, -1], [2, 3]])
        if accepted == 0:
            host[0] = torch.tensor([9, -1])
        merged = merge_recent_ngram_context(
            host, tokens, recent_context, lengths, torch.tensor([1, -1]), context
        )
        assert merged[0].tolist() == expected
        assert merged[1].tolist() == [2, 3]  # no previous record: host value


@pytest.mark.usefixtures("cpu_vllm_config")
def test_draft_loads_every_checkpoint_mtp_tensor(monkeypatch):
    """Real decoder layer/QSA/HC modules; only the MoE block is stubbed."""
    from torch import nn
    from vllm.config import CompilationMode

    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import model, mtp

    text = q4exp_text_config(
        hidden_size=16,
        hc_count=4,
        hc_lowrank=4,
        vocab_size=32,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        indexer_n_heads=1,
        indexer_head_dim=16,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        mtp_num_hidden_layers=1,
    )

    class MoE(nn.Module):
        def __init__(self, vllm_config, prefix):
            super().__init__()
            self.prefix = prefix

    loaded_moe = []

    class Loader:
        def __init__(self, module):
            self.module = module

        def load_weights(self, weights, mapper):
            names = [name for name, _ in weights]
            loaded_moe.extend(f"{self.module.prefix}.{name}" for name in names)
            return set()

    monkeypatch.setattr(model, "Qwen3NextSparseMoeBlock", MoE)
    monkeypatch.setattr(
        model, "get_tp_group", lambda: NS(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(mtp, "AutoWeightsLoader", Loader)
    monkeypatch.setattr(mtp, "verify_moe_checkpoint", lambda *args: None)
    monkeypatch.setattr(mtp, "VocabParallelEmbedding", nn.Embedding)
    monkeypatch.setattr(mtp, "ParallelLMHead", lambda v, h, prefix: nn.Embedding(v, h))
    monkeypatch.setattr(mtp, "LogitsProcessor", lambda vocab: nn.Identity())
    for module in ("parameter", "layers.linear"):
        for name, value in (
            ("get_tensor_model_parallel_rank", 0),
            ("get_tensor_model_parallel_world_size", 1),
        ):
            monkeypatch.setattr(
                f"vllm.model_executor.{module}.{name}",
                lambda value=value: value,
                raising=False,
            )
    context = {}
    cfg = NS(
        model_config=NS(hf_text_config=text, max_model_len=256),
        cache_config=NS(cache_dtype="auto"),
        device_config=NS(device="cpu"),
        compilation_config=NS(
            mode=CompilationMode.NONE, static_forward_context=context
        ),
        additional_config={},
        quant_config=NS(ignored_layers=["mtp.layers.0.mlp.gate"]),
        speculative_config=spec_config(3),
    )
    monkeypatch.setattr(model, "use_unified_pool", lambda: True)
    with torch.device("cpu"):
        draft = mtp.Qwen4ExpMTP(vllm_config=cfg, prefix="")
    attention = draft.model.layers[0].self_attn
    assert context["mtp.layers.2.self_attn"] is attention
    assert attention.prefix == "mtp.layers.2.self_attn"
    assert draft.model.layers[0].ple is None
    assert draft.model.layers[0].layer_type == "full_attention"

    parameters = dict(draft.named_parameters())
    qsa_shapes = {
        "q_proj.weight": (2 * 4 * 32, 16),
        "k_proj.weight": (32, 16),
        "v_proj.weight": (32, 16),
        "o_proj.weight": (16, 4 * 32),
        "q_norm.weight": (32,),
        "k_norm.weight": (32,),
        "indexer.index_qk_proj.weight": (2 * 16, 16),
        "indexer.q_layernorm.weight": (16,),
        "indexer.k_layernorm.weight": (16,),
    }

    def tensor(name):
        relative = name.removeprefix("mtp.layers.0.self_attn.")
        if relative in qsa_shapes:
            shape = qsa_shapes[relative]
        elif ".mlp." in name:
            shape = (1,)
        else:
            shape = parameters[mtp.mtp_checkpoint_name(name)].shape
        return torch.randn(shape).to(torch.bfloat16)

    checkpoint = [
        ("model.language_model.embed_tokens.weight", torch.zeros(32, 16)),
        ("lm_head.weight", torch.zeros(32, 16)),
        ("model.language_model.layers.1.mlp.gate.weight", torch.zeros(1)),
    ] + [(name, tensor(name)) for name in CHECKPOINT_MTP_TENSORS]
    draft.load_weights(iter(checkpoint))
    assert sorted(loaded_moe) == sorted(
        "mtp.layers.2.mlp." + name.removeprefix("mtp.layers.0.mlp.")
        for name in CHECKPOINT_MTP_TENSORS
        if ".mlp." in name
    )
    torch.testing.assert_close(
        draft.model.fc_hidden.weight,
        dict(checkpoint)["mtp.fc_hidden.weight"],
    )
    with pytest.raises(ValueError, match="Incomplete Qwen4Exp MTP checkpoint"):
        draft.load_weights(
            iter(
                (name, tensor(name))
                for name in CHECKPOINT_MTP_TENSORS
                if name != "mtp.pre_fc_norm_hidden.weight"
            )
        )


def test_draft_drops_token_count_dependent_hc_tiles():
    from vllm_torchtpu.models.vllm.qwen4_exp.mtp import draft_additional_config

    target = {
        "qwen4_exp": {
            "engram_config": {"cpu_offload": True},
            "hc_prefill_full_tile": 128,
            "hc_prefill_norm_tile": 128,
            "hc_prefill_up_tile": 128,
        },
        "other": 1,
    }
    draft = draft_additional_config(target)
    assert draft == {"qwen4_exp": {"engram_config": {"cpu_offload": True}}, "other": 1}
    assert target["qwen4_exp"]["hc_prefill_full_tile"] == 128


@pytest.mark.usefixtures("cpu_vllm_config")
def test_draft_rows_match_all_token_step_at_those_rows(monkeypatch):
    """Row-selected step 0 equals the all-token step at the selected rows.

    Real decoder layer and hyper-connections; the MoE and the QSA attention
    are row-wise stand-ins. The stand-in attention also checks that the
    rows step still hands every token to QSA (its cache writes).
    """
    from torch import nn
    from vllm.config import CompilationMode

    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import model, mtp

    text = q4exp_text_config(
        hidden_size=16,
        hc_count=4,
        hc_lowrank=4,
        vocab_size=32,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        indexer_n_heads=1,
        indexer_head_dim=16,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        mtp_num_hidden_layers=1,
    )

    class MoE(nn.Module):
        def __init__(self, vllm_config, prefix):
            super().__init__()
            self.prefix = prefix

        def forward(self, x):
            return torch.tanh(x.float() * 1.5).to(x.dtype)

    monkeypatch.setattr(model, "Qwen3NextSparseMoeBlock", MoE)
    monkeypatch.setattr(
        model, "get_tp_group", lambda: NS(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(mtp, "VocabParallelEmbedding", nn.Embedding)
    monkeypatch.setattr(mtp, "ParallelLMHead", lambda v, h, prefix: nn.Embedding(v, h))
    monkeypatch.setattr(mtp, "LogitsProcessor", lambda vocab: nn.Identity())
    for module in ("parameter", "layers.linear"):
        for name, value in (
            ("get_tensor_model_parallel_rank", 0),
            ("get_tensor_model_parallel_world_size", 1),
        ):
            monkeypatch.setattr(
                f"vllm.model_executor.{module}.{name}",
                lambda value=value: value,
                raising=False,
            )
    cfg = NS(
        model_config=NS(hf_text_config=text, max_model_len=256),
        cache_config=NS(cache_dtype="auto"),
        device_config=NS(device="cpu"),
        compilation_config=NS(mode=CompilationMode.NONE, static_forward_context={}),
        additional_config={},
        quant_config=None,
        speculative_config=spec_config(3),
    )
    monkeypatch.setattr(model, "use_unified_pool", lambda: True)
    torch.manual_seed(0)
    with torch.device("cpu"):
        draft = mtp.Qwen4ExpMTP(vllm_config=cfg, prefix="")
    draft = draft.to(torch.bfloat16)
    for p in draft.parameters():
        p.data.normal_(0, 0.2)
    attention = draft.model.layers[0].self_attn
    attended = []

    def forward_with_indices(hidden, positions, metadata, rows=None, reduce=True):
        attended.append((len(hidden), None if rows is None else rows.tolist()))
        output = torch.sin(hidden.float() * 3 + positions[:, None].float() / 7)
        output = output.to(hidden.dtype)
        if rows is not None:
            keep = (rows >= 0)[:, None]
            output = torch.where(keep, output[rows.clamp_min(0)], 0)
        return output, None

    monkeypatch.setattr(attention, "forward_with_indices", forward_with_indices)
    metadata = NS(
        query_start_loc=torch.tensor([0, 3, 7, 7]),
        seq_lens=torch.tensor([12, 9, 0]),
        block_tables=torch.zeros(3, dtype=torch.int32),
    )
    monkeypatch.setattr(
        mtp,
        "get_forward_context",
        lambda: NS(attn_metadata={"mtp.layers.2.self_attn": metadata}),
    )
    ids = torch.tensor([3, 5, 7, 1, 2, 9, 4, 0])
    positions = torch.tensor([9, 10, 11, 5, 6, 7, 8, 0])
    target = torch.randn(8, 64).to(torch.bfloat16)
    sample, multi = draft(ids, positions, target)
    rows = torch.tensor([2, 6, 6, -1])
    row_sample, row_multi = draft(ids, positions, target, draft_rows=rows)
    assert attended == [(8, None), (8, [2, 6, 6, -1])]
    assert row_sample.shape == (4, 16) and row_multi.shape == (4, 64)
    picked = rows.clamp_min(0)[:3]
    torch.testing.assert_close(row_multi[:3], multi[picked], rtol=0, atol=0)
    torch.testing.assert_close(row_sample[:3], sample[picked], rtol=0, atol=0)
    assert torch.isfinite(row_multi[3].float()).all()


def _draft_rows_trace_inputs(dims, *, tokens, rows, buckets):
    """Trace a draft-step-0-shaped forward as support_torch_compile does
    (dims marked dynamic, dynamic=False, guards dropped) and return each
    bucket's concrete placeholder shapes, built the way vLLM's piecewise
    backend builds them (create_concrete_args)."""
    import torch._dynamo
    from vllm.compilation.piecewise_backend import create_concrete_args

    def step0(input_ids, positions, hidden_states, draft_rows):
        hidden = hidden_states + (input_ids + positions)[:, None].to(hidden_states)
        picked = hidden[draft_rows.clamp_min(0)]
        return torch.where((draft_rows >= 0)[:, None], picked, 0)

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    args = {
        "input_ids": torch.zeros(tokens, dtype=torch.int64),
        "positions": torch.zeros(tokens, dtype=torch.int64),
        "hidden_states": torch.zeros(tokens, 8),
        "draft_rows": torch.full((rows,), -1, dtype=torch.int64),
    }
    for name, dim in dims.items():
        torch._dynamo.mark_dynamic(args[name], dim)
    torch._dynamo.reset()
    torch.compile(step0, backend=backend, fullgraph=True, dynamic=False)(**args)
    (graph,) = graphs
    shapes = {}
    for size in buckets:
        concrete = [
            tuple(a.shape)
            for a in create_concrete_args(graph, size)
            if isinstance(a, torch.Tensor)
        ]
        shapes[size] = concrete
    return shapes


def test_draft_rows_static_across_token_buckets():
    """vLLM compiles the draft once per token bucket by setting every trace
    symbol to the bucket size. draft_rows must therefore not be dynamic, or
    bucket 256's program expects s64[256] rows while a step passes the
    request-sized ones (RuntimeProgramInputMismatch on TPU)."""
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    assert "draft_rows" not in mtp.QWEN4_EXP_MTP_DYNAMIC_ARG_DIMS
    # Trace at tokens == rows: the worst case, duck sizing would share a
    # symbol between the two if draft_rows were dynamic.
    shapes = _draft_rows_trace_inputs(
        mtp.QWEN4_EXP_MTP_DYNAMIC_ARG_DIMS, tokens=32, rows=32, buckets=(64, 256)
    )
    for size, concrete in shapes.items():
        assert (size,) in concrete  # input_ids / positions follow the bucket
        assert (size, 8) in concrete
        assert concrete[-1] == (32,)  # draft_rows stays at its fixed length
    # The pre-fix dims reproduce the failure: draft_rows takes the bucket size.
    broken = _draft_rows_trace_inputs(
        {**mtp.QWEN4_EXP_MTP_DYNAMIC_ARG_DIMS, "draft_rows": 0},
        tokens=64,
        rows=32,
        buckets=(256,),
    )
    assert broken[256][-1] == (256,)


def test_megakernel_paths_keep_the_sink_layer_on_attend():
    """VLLM_TPU_QSA_DRAFT_SHARED_SELECTION: the draft reads the selection the target's last QSA layer
    writes in `attend`. `select_for_kernel` returns no selection, so it refuses the sink layer, and
    the served megakernel paths never take it: in-kernel attention is gated on
    `segment_call.attention_rows`, which neither MK segment object defines (default 0 -> attend)."""
    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

    with pytest.raises(NotImplementedError, match="shared-selection"):
        Qwen4ExpQSA.select_for_kernel(NS(_selection_sink=torch.zeros(1, 1)), None, None, None)
    try:
        from vllm_torchtpu.megakernel_runtime.qwen38_driver import Qwen38Segments
    except Exception:  # noqa: BLE001 - optional runtime deps
        pytest.skip("megakernel runtime unavailable")
    assert not hasattr(Qwen38Segments, "attention_rows")
