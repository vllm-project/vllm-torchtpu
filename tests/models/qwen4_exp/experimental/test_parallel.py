# SPDX-License-Identifier: Apache-2.0
"""Admitted topology and checkpoint ownership contracts."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tests.models.qwen4_exp.experimental.qwen38_test_utils import (
    pin_plan,
    q4exp_text_config,
)
from vllm_torchtpu.models.qwen4_exp.experimental.config import validate_config
from vllm_torchtpu.models.qwen4_exp.experimental.parallel import PLEParallel


@pytest.fixture(autouse=True)
def plan_features(monkeypatch):
    """Every plan feature off (PLE replicated, on HBM); tests edit the dict."""
    return pin_plan(monkeypatch)


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
@pytest.mark.parametrize("enable_ep", [False, True])
def test_admitted_topologies(tp, dp, enable_ep):
    cfg = config(tp, dp)
    cfg.parallel_config.enable_expert_parallel = enable_ep
    validate_config(cfg)


@pytest.mark.parametrize("tp,dp", [(1, 1), (2, 1), (4, 1), (2, 2), (4, 4)])
@pytest.mark.parametrize("enable_ep", [False, True])
def test_topology_admission_follows_geometry_not_device_count(tp, dp, enable_ep):
    cfg = config(tp, dp)
    cfg.parallel_config.enable_expert_parallel = enable_ep
    validate_config(cfg)


@pytest.mark.parametrize("tp,dp,message", [(1, 3, "experts"), (16, 1, "QSA")])
def test_topology_rejects_nonintegral_expert_or_head_shards(tp, dp, message):
    cfg = config(tp, dp)
    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


@pytest.mark.parametrize(
    "field,value",
    [
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
    with pytest.raises(ValueError, match="requires PP/CP1"):
        validate_config(cfg)


@pytest.mark.parametrize("tp,dp", [(2, 4), (4, 2)])
def test_model_admission_precedes_tpu_sequence_parallel_normalization(tp, dp):
    cfg = config(tp, dp)
    # vLLM derives this property for TP*DP EP before platform_activation
    # disables it. It is not a user-requested model capability.
    cfg.parallel_config.use_sequence_parallel_moe = True
    validate_config(cfg)


def test_host_configuration_is_admitted(monkeypatch, plan_features):
    from vllm_torchtpu.models.qwen4_exp.experimental import config as admission

    monkeypatch.setattr(admission, "host_available_memory", lambda: 2 * 1024**4)
    cfg = config()
    plan_features["PLE_CPU_OFFLOAD"] = True
    validate_config(cfg)


def test_private_host_replicas_are_checked_against_host_capacity(
    monkeypatch, plan_features
):
    from vllm_torchtpu.models.qwen4_exp.experimental import config as admission

    monkeypatch.setattr(admission, "host_available_memory", lambda: 850 * 1024**3)
    cfg = config()
    plan_features["PLE_CPU_OFFLOAD"] = True
    with pytest.raises(ValueError, match="host PLE capacity"):
        validate_config(cfg)


def test_private_host_offload_rejects_unvalidated_multihost(monkeypatch, plan_features):
    from vllm_torchtpu.models.qwen4_exp.experimental import config as admission

    monkeypatch.setattr(admission, "host_available_memory", lambda: 2 * 1024**4)
    cfg = config()
    cfg.parallel_config.nnodes = 2
    cfg.parallel_config.data_parallel_size_local = 4
    plan_features["PLE_CPU_OFFLOAD"] = True
    with pytest.raises(ValueError, match="host PLE requires.*one host"):
        validate_config(cfg)


def test_ple_borrows_the_tp_group():
    raw = torch.arange(3 * 2560).remainder(256).to(torch.uint8).view(torch.int8)
    raw = raw.reshape(3, 2560)
    tp = NS(world_size=8, rank_in_group=7, all_reduce=Mock(side_effect=lambda x: x))
    parallel = PLEParallel(tp)
    assert parallel.group is tp
    assert (parallel.rank, parallel.world_size) == (7, 8)
    assert parallel.combine(raw) is raw
    tp.all_reduce.assert_called_once_with(raw)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_ple_owner_ranges_cover_the_table_once(tp):
    rows = 8192
    owned = []
    for rank in range(tp):
        parallel = PLEParallel(NS(world_size=tp, rank_in_group=rank))
        start = parallel.rank * rows // parallel.world_size
        stop = (parallel.rank + 1) * rows // parallel.world_size
        owned.extend(range(start, stop))
    assert owned == list(range(rows))


@pytest.mark.parametrize("tp,dp,rank", [(1, 8, 7), (2, 4, 7), (4, 2, 7), (8, 1, 7)])
@pytest.mark.parametrize("enable_ep", [False, True])
def test_model_uses_true_tp_ep_and_embedding_geometry(
    monkeypatch, tp, dp, rank, enable_ep
):
    from vllm_torchtpu.models.qwen4_exp.experimental import model

    cfg = config(tp, dp)
    cfg.parallel_config.enable_expert_parallel = enable_ep
    cfg.device_config = NS(device="cpu")
    cfg.compilation_config = NS(static_forward_context={})
    cfg.quant_config = object()
    monkeypatch.setattr(
        model, "get_tp_group", lambda: NS(world_size=tp, rank_in_group=rank % tp)
    )
    qsa, ple, moe = (
        Mock(side_effect=lambda **kw: nn.Identity()),
        Mock(side_effect=lambda **kw: nn.Identity()),
        Mock(side_effect=lambda **kw: nn.Identity()),
    )
    monkeypatch.setattr(model, "Qwen4ExpQSA", qsa)
    monkeypatch.setattr(model, "Qwen4ExpPooledPLE", ple)
    monkeypatch.setattr(model, "Qwen3NextSparseMoeBlock", moe)
    gdn = Mock(side_effect=lambda *a, **kw: nn.Identity())
    monkeypatch.setattr(model, "Qwen4ExpGatedDeltaNetAttention", gdn)
    monkeypatch.setattr(model, "GatedResidual", lambda *a, **kw: nn.Identity())
    model.Qwen4ExpDecoderLayer(cfg, 3, "model.layers.3")
    assert qsa.call_args.kwargs["tp_size"] == tp
    assert qsa.call_args.kwargs["tp_rank"] == rank % tp
    assert moe.call_args.kwargs == {"vllm_config": cfg, "prefix": "model.layers.3.mlp"}
    model.Qwen4ExpDecoderLayer(cfg, 1, "model.layers.1")
    assert gdn.call_args.kwargs["vllm_config"] is cfg
    assert gdn.call_args.kwargs["vllm_config"].quant_config is cfg.quant_config
    parallel = ple.call_args.kwargs["parallel"]
    assert (parallel.rank, parallel.world_size) == (rank % tp, tp)


@pytest.mark.parametrize("excluded", [True, False])
@pytest.mark.parametrize(
    "projection,checkpoint_names",
    [
        ("in_proj_qkvz", ["in_proj_qkv", "in_proj_z"]),
        ("in_proj_ba", ["in_proj_b", "in_proj_a"]),
        ("out_proj", ["out_proj"]),
    ],
)
def test_gdn_quantization_follows_checkpoint_exclusions(
    monkeypatch, projection, checkpoint_names, excluded
):
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization import fp8 as upstream_fp8
    from vllm.model_executor.model_loader.utils import configure_quant_config

    from vllm_torchtpu.layers.adapter.quantization.fp8 import (
        VllmFp8Config,
        VllmFp8LinearMethodTPU,
    )
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmUnquantizedLinearMethod,
    )
    from vllm_torchtpu.models.qwen4_exp.experimental.model import Qwen4ExpForCausalLM

    # Preserve the upstream function modified by the existing TPU FP8 method.
    monkeypatch.setattr(
        upstream_fp8, "init_fp8_linear_kernel", upstream_fp8.init_fp8_linear_kernel
    )
    quant = VllmFp8Config.from_config(
        {
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [
                f"model.language_model.layers.0.linear_attn.{name}"
                for name in checkpoint_names
            ]
            if excluded
            else [],
        }
    )
    configure_quant_config(quant, Qwen4ExpForCausalLM)
    quant.vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    layer = LinearBase.__new__(LinearBase)
    nn.Module.__init__(layer)
    layer.input_size = layer.output_size = 128
    method = quant.get_quant_method(
        layer, prefix=f"model.layers.0.linear_attn.{projection}"
    )
    expected = VllmUnquantizedLinearMethod if excluded else VllmFp8LinearMethodTPU
    assert isinstance(method, expected)


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
        ("shared_expert_intermediate_size", 641),
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


@pytest.fixture
def upstream_moe(monkeypatch):
    """Build real upstream layers on CPU without initializing device groups."""
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.fused_moe import layer as fused_moe
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.model_executor.models import qwen3_next
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
    from vllm_torchtpu.models.qwen4_exp.experimental.model import Qwen4ExpForCausalLM

    def build(
        tp,
        enable_ep,
        rank,
        renormalize=True,
        shared_fp8=False,
        world_size=8,
    ):
        cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
        text = q4exp_text_config(
            hidden_size=256,
            num_experts=8,
            num_experts_per_tok=2,
            norm_topk_prob=renormalize,
        )
        cfg.model_config = NS(hf_text_config=text, dtype=torch.bfloat16)
        cfg.device_config.device = torch.device("cpu")
        cfg.parallel_config.tensor_parallel_size = tp
        cfg.parallel_config.data_parallel_size = world_size // tp
        cfg.parallel_config.enable_expert_parallel = enable_ep
        quant = VllmFp8Config.from_config(
            {
                "weight_block_size": [128, 128],
                "modules_to_not_convert": [
                    f"model.language_model.layers.0.mlp.{name}"
                    for name in (
                        "gate",
                        "shared_expert_gate",
                        "shared_expert.gate_proj",
                        "shared_expert.up_proj",
                        "shared_expert.down_proj",
                    )
                    if not shared_fp8 or not name.startswith("shared_expert.")
                ],
            }
        )
        configure_quant_config(quant, Qwen4ExpForCausalLM)
        quant.vllm_config = cfg
        cfg.quant_config = quant
        parallel = FusedMoEParallelConfig(
            tp_size=1 if enable_ep else tp,
            pcp_size=1,
            dp_size=world_size // tp,
            ep_size=world_size if enable_ep else 1,
            tp_rank=0 if enable_ep else rank % tp,
            pcp_rank=0,
            dp_rank=rank // tp,
            ep_rank=rank if enable_ep else 0,
            sp_size=1,
            use_ep=enable_ep,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        )
        monkeypatch.setattr(
            parallel_state, "_TP", NS(world_size=tp, rank_in_group=rank % tp)
        )
        monkeypatch.setattr(
            qwen3_next,
            "get_ep_group",
            lambda: NS(device_group=NS(size=lambda: world_size)),
        )
        monkeypatch.setattr(
            fused_moe, "make_parallel_config", lambda **kwargs: parallel
        )
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with set_current_vllm_config(cfg), torch.device("cpu"):
                return Qwen3NextSparseMoeBlock(cfg, "model.layers.0.mlp")
        finally:
            torch.set_default_dtype(previous_dtype)

    return build


@pytest.mark.parametrize(
    "tp,enable_ep,rank",
    [
        (1, True, 7),
        (2, True, 7),
        (4, True, 0),
        (8, True, 7),
        (1, False, 0),
        (2, False, 1),
        (4, False, 1),
        (8, False, 7),
    ],
)
@pytest.mark.parametrize("renormalize", [False, True])
def test_moe_uses_upstream_quantization_and_checkpoint_exclusions(
    upstream_moe, tp, enable_ep, rank, renormalize
):
    """Catch lost quant exclusions, geometry, or routing options at integration."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmUnquantizedLinearMethod,
    )

    block = upstream_moe(tp, enable_ep, rank, renormalize)
    experts = block.experts.routed_experts
    assert isinstance(block, Qwen3NextSparseMoeBlock)
    assert type(block).forward is Qwen3NextSparseMoeBlock.forward
    assert isinstance(experts.quant_method, VllmFp8MoEMethodTPU)
    assert type(block.gate.quant_method) is UnquantizedLinearMethod
    assert type(block.shared_expert_gate.quant_method) is UnquantizedLinearMethod
    assert isinstance(
        block.shared_expert.gate_up_proj.quant_method, VllmUnquantizedLinearMethod
    )
    assert isinstance(
        block.shared_expert.down_proj.quant_method, VllmUnquantizedLinearMethod
    )
    assert block.gate.weight.dtype == torch.bfloat16
    assert block.shared_expert.gate_up_proj.weight.dtype == torch.bfloat16
    assert experts.renormalize == renormalize
    assert experts.moe_config.experts_per_token == 2
    width = 640 if enable_ep else 640 // tp
    count = 1 if enable_ep else 8
    assert experts.w13_weight.shape == (count, 2 * width, 256)
    assert experts.w2_weight.shape == (count, 256, width)
    assert block.shared_expert.gate_up_proj.weight.shape == (2 * 640 // tp, 256)


@pytest.mark.tpu_test
@pytest.mark.parametrize("enable_ep", [False, True])
def test_tpu_upstream_moe_block128_checkpoint_accuracy(
    upstream_moe, monkeypatch, enable_ep
):
    """Existing block-128 FP8 mode retains BF16 activations on this TPU."""
    import torch.nn.functional as F
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "128")
    monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn")
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "0")
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "0")
    # One rank exercises both ordinary routing and the EP interface without
    # bootstrapping a distributed deployment.
    block = upstream_moe(1, enable_ep, 0, world_size=1)
    cfg = block.experts.routed_experts.quant_method.quant_config.vllm_config
    weights = []
    reference_weights = {}
    for expert in range(8):
        for projection, value in [
            ("gate", 0.5 + expert / 8),
            ("up", 1.0),
            ("down", 1.0),
        ]:
            shape = (256, 640) if projection == "down" else (640, 256)
            name = f"experts.{expert}.{projection}_proj"
            weights.append(
                (name + ".weight", torch.full(shape, value, dtype=torch.float8_e4m3fn))
            )
            weights.append(
                (
                    name + ".weight_scale_inv",
                    torch.full(
                        (shape[0] // 128, shape[1] // 128), 1 / 256, dtype=torch.float32
                    ),
                )
            )
            reference_weights[expert, projection] = torch.full(
                shape, value / 256, dtype=torch.bfloat16
            )
    for name, shape, value in [
        ("gate", (8, 256), 0.0),
        ("shared_expert_gate", (1, 256), 1 / 256),
        ("shared_expert.gate_proj", (640, 256), 1 / 256),
        ("shared_expert.up_proj", (640, 256), 1 / 256),
        ("shared_expert.down_proj", (256, 640), 1 / 256),
    ]:
        weight = torch.full(shape, value, dtype=torch.bfloat16)
        if name == "gate":
            weight[:, 0] = torch.arange(8, dtype=torch.bfloat16) / 8
        reference_weights[name] = weight
        weights.append((name + ".weight", weight))
    mapper = WeightsMapper(
        orig_to_new_stacked={
            "shared_expert.gate_proj": ("shared_expert.gate_up_proj", 0),
            "shared_expert.up_proj": ("shared_expert.gate_up_proj", 1),
        }
    )
    AutoWeightsLoader(block).load_weights(iter(weights), mapper=mapper)

    x = (
        torch.linspace(0.25, 1, 16, dtype=torch.bfloat16)[:, None]
        .expand(16, 256)
        .contiguous()
    )
    probabilities, ids = torch.softmax(
        F.linear(x, reference_weights["gate"]).float(), -1
    ).topk(2, -1)
    probabilities = (probabilities / probabilities.sum(-1, keepdim=True)).bfloat16()
    expected = torch.zeros_like(x)
    for expert in range(8):
        hidden = F.silu(F.linear(x, reference_weights[expert, "gate"])) * F.linear(
            x, reference_weights[expert, "up"]
        )
        output = F.linear(hidden, reference_weights[expert, "down"])
        expected += output * torch.where(ids == expert, probabilities, 0).sum(
            -1, keepdim=True
        )
    hidden = F.silu(F.linear(x, reference_weights["shared_expert.gate_proj"])) * (
        F.linear(x, reference_weights["shared_expert.up_proj"])
    )
    expected += F.linear(hidden, reference_weights["shared_expert.down_proj"]) * (
        torch.sigmoid(F.linear(x, reference_weights["shared_expert_gate"]))
    )

    with set_current_vllm_config(cfg), torch.no_grad():
        block = block.to("tpu")
        for child in block.modules():
            method = getattr(child, "quant_method", None)
            if method is not None:
                method.process_weights_after_loading(child)
        device_input = x.to("tpu")
        with set_forward_context(None, cfg, num_tokens=x.shape[0]):
            eager = block(device_input).cpu().float()
            compiled = (
                torch.compile(block, backend="tpu", fullgraph=True, dynamic=False)(
                    device_input
                )
                .cpu()
                .float()
            )
    torch.testing.assert_close(eager, compiled, rtol=0, atol=0)
    expected = expected.float()
    for actual in (eager, compiled):
        assert torch.isfinite(actual).all()
        nrmse = torch.linalg.vector_norm(actual - expected) / (
            torch.linalg.vector_norm(expected)
        )
        cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        assert nrmse < 0.01, nrmse.item()
        assert cosine > 0.999, cosine.item()


@pytest.mark.parametrize("prefix", ["model.language_model.", "model."])
@pytest.mark.parametrize(
    "enable_ep,shared_fp8,missing_projection",
    [
        (False, False, None),
        (True, False, None),
        (True, True, "gate"),
        (True, True, "up"),
        (True, True, "down"),
    ],
)
def test_model_streams_expert_and_shared_weights_through_upstream_loaders(
    upstream_moe, prefix, enable_ep, shared_fp8, missing_projection
):
    """Checkpoint name mapping must preserve expert IDs and shared gate/up halves."""
    from vllm_torchtpu.models.qwen4_exp.experimental.model import Qwen4ExpForCausalLM

    tp = 1 if shared_fp8 else 2
    block = upstream_moe(tp, enable_ep, 7, shared_fp8=shared_fp8)
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = block
    layer.ple = None
    layer.layer_type = "full_attention"
    layer.self_attn = nn.Module()
    layer.self_attn.finalize_weights = Mock()
    model.model.layers = nn.ModuleList([layer])
    weights = []
    for expert in range(8):
        for projection, value in [("gate", 1), ("up", 2), ("down", 3)]:
            shape = (256, 640) if projection == "down" else (640, 256)
            name = f"experts.{expert}.{projection}_proj"
            weights.append(
                (
                    f"{name}.weight_scale_inv",
                    torch.full(
                        (shape[0] // 128, shape[1] // 128),
                        expert + value,
                        dtype=torch.float32,
                    ),
                )
            )
            weights.append(
                (f"{name}.weight", torch.full(shape, value, dtype=torch.float8_e4m3fn))
            )
    for name, shape, value in [
        ("gate", (8, 256), 4),
        ("shared_expert_gate", (1, 256), 5),
        ("shared_expert.gate_proj", (640, 256), 6),
        ("shared_expert.up_proj", (640, 256), 7),
        ("shared_expert.down_proj", (256, 640), 8),
    ]:
        quantized = shared_fp8 and name.startswith("shared_expert.")
        dtype = torch.float8_e4m3fn if quantized else torch.bfloat16
        weights.append((name + ".weight", torch.full(shape, value, dtype=dtype)))
        if quantized:
            weights.append(
                (
                    name + ".weight_scale_inv",
                    torch.full(
                        (shape[0] // 128, shape[1] // 128),
                        value / 256,
                        dtype=torch.float32,
                    ),
                )
            )
    # Reverse order places scales after values and reverses the shared halves.
    source = [
        (prefix + "layers.0.mlp." + name, value) for name, value in reversed(weights)
    ]
    if enable_ep:
        source = [
            (name, value)
            for name, value in source
            if ".experts." not in name or ".experts.7." in name
        ]
    model.load_weights(iter(source))
    experts = block.experts.routed_experts
    expert_ids = [7] if enable_ep else list(range(8))
    logical_width = 640 if enable_ep else 320
    padded_width = experts.w2_weight.shape[-1]
    for local, expert_id in enumerate(expert_ids):
        scale_half = experts.w13_weight_scale_inv.shape[1] // 2
        for half, value in [(0, 1), (1, 2)]:
            offset = half * padded_width
            assert torch.all(
                experts.w13_weight[local, offset : offset + logical_width].float()
                == value
            )
            assert torch.all(
                experts.w13_weight_scale_inv[
                    local, half * scale_half : (half + 1) * scale_half
                ]
                == expert_id + value
            )
        assert torch.all(experts.w2_weight[local, :, :logical_width].float() == 3)
        assert torch.all(experts.w2_weight_scale_inv[local] == expert_id + 3)
    assert torch.all(block.gate.weight == 4)
    assert torch.all(block.shared_expert_gate.weight == 5)
    shared_width = 640 // tp
    assert torch.all(
        block.shared_expert.gate_up_proj.weight[:shared_width].float() == 6
    )
    assert torch.all(
        block.shared_expert.gate_up_proj.weight[shared_width:].float() == 7
    )
    assert torch.all(block.shared_expert.down_proj.weight.float() == 8)
    if shared_fp8:
        for projection in (
            block.shared_expert.gate_up_proj,
            block.shared_expert.down_proj,
        ):
            assert torch.all(projection.weight_scale_inv > 0)
            torch.testing.assert_close(
                projection.weight_scale, projection.weight_scale_inv, rtol=0, atol=0
            )
    layer.self_attn.finalize_weights.assert_called_once()
    missing_name = prefix + "layers.0.mlp.experts.7.gate_proj.weight_scale_inv"
    incomplete = ((name, value) for name, value in source if name != missing_name)
    with pytest.raises(ValueError, match="expert checkpoint: 1 missing"):
        model.load_weights(incomplete)
    if shared_fp8:
        layer.mlp = upstream_moe(tp, enable_ep, 7, shared_fp8=True)
        missing_name = (
            prefix
            + "layers.0.mlp.shared_expert."
            + missing_projection
            + "_proj.weight_scale_inv"
        )
        incomplete = ((name, value) for name, value in source if name != missing_name)
        with pytest.raises(
            ValueError,
            match="shared_expert." + missing_projection + "_proj.weight_scale",
        ):
            model.load_weights(incomplete)


def test_decoder_zeroes_unwritten_padding_before_standard_moe():
    """NaN GDN padding must not reach the generic router or EP dispatch."""
    from vllm_torchtpu.models.qwen4_exp.experimental.model import (
        Qwen4ExpDecoderLayer,
    )

    class Residual(nn.Module):
        def mix(self, hidden):
            return hidden, None

        def combine(self, block, residual):
            return block

        def combine_mix(self, block, residual, next_hc):
            return next_hc.mix(self.combine(block, residual))

    layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    layer.ple = None
    layer.layer_type = "linear_attention"
    # GDN takes (block, step metadata cache) and returns the block.
    layer.linear_attn = lambda block, step_metadata_cache=None: block
    layer.attn_hyper_connection = Residual()
    layer.mlp_hyper_connection = Residual()
    layer.mlp = nn.Identity()
    observed = []
    layer.mlp.register_forward_pre_hook(lambda _, args: observed.append(args[0]))
    hidden = torch.tensor(
        [[1.0, -2.0], [float("nan"), float("nan")]], dtype=torch.bfloat16
    )
    valid = torch.tensor([True, False])
    output = layer(hidden, None, None, None, None, None, None, valid, None, None)
    expected = torch.tensor([[1.0, -2.0], [0.0, 0.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("slots", [65, 128, 256])
def test_request_capacity_is_not_limited_to_bringup_slots(slots):
    cfg = config()
    cfg.scheduler_config.max_num_seqs = slots
    validate_config(cfg)


def test_reuses_upstream_moe_without_precision_override(upstream_moe):
    from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner

    block = upstream_moe(1, True, 0)
    assert isinstance(block.experts, VllmMoERunner)
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

    assert type(block) is Qwen3NextSparseMoeBlock
    assert "combine_dtype" not in vars(block.experts)
    assert block.experts.do_naive_dispatch_combine
