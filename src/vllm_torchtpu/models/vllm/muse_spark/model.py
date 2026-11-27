# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Muse Spark text model on TPU, with DP attention and TP or DP×TP EP experts.

Upstream source: ``vllm/models/muse_spark/nvidia/model.py``
(``MuseSpark1xAttention``, ``MuseSpark1xMoE``, ``MuseSpark1xDecoderLayer``,
``MuseSpark1xModel`` and ``MuseSpark1xForCausalLM``). The residual gate and the
logit soft cap are imported unchanged from
``vllm/models/muse_spark/common/layers.py``.

TPU changes:

- Modules keep the checkpoint's names and layout (separate ``qkv_proj`` and
  ``gate_proj``, ``mlp``, ``*_residual_gate``), so weights load without
  upstream's renames or q/k row interleaving; RoPE is NeoX-style on the
  checkpoint's ``[even | odd]`` q/k rows instead.
- Attention applies the q/k norm, vLLM's FP32 rotary embedding and the sigmoid
  output gate separately instead of the fused CUDA qk-norm-RoPE-KV and
  o-norm-gate kernels.
- Routing uses ``moe_routing.select_experts`` on FP32 router logits from a
  scoped ``Precision.HIGHEST`` matmul (``fp32_matmul``). Experts are the
  shared ``RoutedExperts`` with the shared NVFP4 MoE method (native FP4 with
  FP32 block scales), run through ``fused_moe_func``'s rows mode, which returns
  one row per token and expert so the per-expert post-norm runs before the
  weighted sum.
  TP partials are all-reduced before the post-norm; DP attention with EP
  gathers tokens over DP, reduce-scatters the FP32 sums over DP and
  all-reduces over TP.
- RMSNorms are unfused FP32 (``layers.MuseRMSNorm``).
- No sequence parallelism, MegaMoE, fused residual-norm boundaries, PP,
  context parallelism, EPLB or LoRA.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, RoutedExperts
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.models.muse_spark.common.layers import (
    MulTanhSoftCap,
    MuseSpark1xNormPreservingResidualConnection,
)
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.custom_ops.fp32_matmul_op import (
    fp32_matmul,
    prebuild_fp32_matmul_op,
)
from vllm_torchtpu.layers.adapter.fused_moe import (
    TpuMoEActivationMixin,
    fused_moe_gmm,
    prebuild_fused_moe_kernel,
)
from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4MoEMethod
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.moe_reference_parity import (
    SILU_BF16_KEY,
    get_moe_reference_parity,
)

from .layers import (
    MuseRMSNorm,
    MuseRMSNormWithInputScale,
    attention_layout,
    selected_expert_outputs,
)
from .weights import selected_packed_expert_outputs

if TYPE_CHECKING:
    from vllm.transformers_utils.configs.muse_spark import MuseSparkTextConfig

    from vllm_torchtpu.layers.adapter.quantization.configs import TpuRoutedExperts


class MuseRoutedExperts(RoutedExperts):
    """The shared routed experts, loaded from the checkpoint's stacked tensors.

    The checkpoint stores each projection as one [experts, ...] tensor. Each
    expert's slice goes to the shared weight loader, which keeps this rank's
    TP or EP share; the shared MoE quantization method then lays the weights
    out for gmm_v2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # _expert_rows reads BF16 or native FP4 experts and their block scales.
        if not isinstance(
            self.quant_method, (VllmUnquantizedFusedMoEMethod, VllmNvfp4MoEMethod)
        ):
            raise ValueError(
                f"Muse TPU experts run BF16 or NVFP4 weights, but {self.layer_name} "
                f"uses {type(self.quant_method).__name__}; QUANTIZE_ON_LOAD_PREFIXES "
                "must not select the experts"
            )
        if isinstance(self.quant_method, VllmNvfp4MoEMethod):
            # _expert_rows runs gmm_v2 rows mode, never a fused EP op, so keep
            # the experts' block-16 scales instead of requantizing for one.
            self.quant_method.allow_fused_ep = False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        for name, tensor in weights:
            w13 = name.startswith("gate_up_proj")
            if not w13 and not name.startswith("down_proj"):
                raise ValueError(f"Unexpected Muse expert tensor: {name}")
            suffix = name.removeprefix("gate_up_proj" if w13 else "down_proj")
            param_name = ("w13" if w13 else "w2") + (suffix or "_weight")
            param = getattr(self, param_name, None)
            if param is None:
                raise ValueError(
                    f"Muse experts have no {param_name} for {name}; NVFP4 experts "
                    "must not be listed in modules_to_not_convert"
                )
            if w13:
                # The gate and up halves load as the w1 and w3 shards.
                halves = (
                    tensor.chunk(2, dim=1) if tensor.dim() == 3 else tensor.unbind(1)
                )
                shards = list(zip(("w1", "w3"), halves))
            else:
                shards = [("w2", tensor)]
            for shard_id, part in shards:
                for expert_id, value in enumerate(part.unbind(0)):
                    param.weight_loader(
                        param=param,
                        loaded_weight=value,
                        weight_name=param_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
            loaded.add(param_name)
        return loaded

    def initialize_dummy_quantized_weights(self, seed: int) -> None:
        """Finite, non-zero NVFP4 experts for ``--load-format dummy``.

        The dummy loader's packed bytes and block scales do not decode to
        usable experts. After the NVFP4 MoE method lays the experts out
        (native FP4 [E, K, N] and FP32 block scales), use random +/-0.5 FP4
        values and a fan-in block scale, generated on CPU so no full-size
        random tensor lands in HBM.
        """
        layer = cast("TpuRoutedExperts", self)
        if layer.w13_weight.dtype != torch.float4_e2m1fn_x2:
            return
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for prefix in ("w13", "w2"):
            weight = getattr(layer, f"{prefix}_weight")
            values = torch.randint(
                0, 256, weight.shape, dtype=torch.uint8, generator=generator
            )
            values = values.bitwise_and_(0x88).bitwise_or_(0x11)
            weight.data.copy_(values.view(torch.float4_e2m1fn_x2))
            getattr(layer, f"{prefix}_weight_scale").data.fill_(weight.shape[1] ** -0.5)


def _expert_rows(
    experts: RoutedExperts, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    """Unweighted rows [tokens, top_k, hidden] from this rank's experts."""
    layer = cast("TpuRoutedExperts", experts)
    if x.device.type == "tpu":
        method = cast(TpuMoEActivationMixin, layer.quant_method)
        # NVFP4 experts are native FP4 with FP32 block scales; BF16 has none.
        scaled = layer.w13_weight.dtype == torch.float4_e2m1fn_x2
        return fused_moe_gmm(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale if scaled else None,
            layer.w2_weight_scale if scaled else None,
            None,
            None,
            weights,
            ids,
            layer._experts_start,
            topk=ids.shape[-1],
            activation="silu",
            topk_weights_float32=method._tpu_topk_weights_float32,
            return_rows=True,
        )
    # The CPU reference reads the checkpoint layout: packed uint8 rows with
    # E4M3 block scales and per-expert global scales.
    packed = layer.w13_weight.dtype == torch.uint8
    owned = None
    if layer._experts_start is not None:
        local_ids = ids - layer._experts_start
        owned = (local_ids >= 0) & (local_ids < layer.w13_weight.shape[0])
        ids = local_ids.clamp(0, layer.w13_weight.shape[0] - 1)
    if packed:
        rows = selected_packed_expert_outputs(
            x,
            ids,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale.view(torch.uint8),
            layer.w2_weight_scale.view(torch.uint8),
            layer.w13_weight_scale_2,
            layer.w2_weight_scale_2,
        )
    else:
        rows = selected_expert_outputs(x, ids, layer.w13_weight, layer.w2_weight)
    return rows if owned is None else rows.masked_fill(~owned[..., None], 0)


class MuseMoE(nn.Module):
    def __init__(
        self,
        config,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        use_ep: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.use_ep = use_ep
        self.dp_group = get_dp_group() if use_ep else None
        self.dp_size = self.dp_group.world_size if self.dp_group is not None else 1
        self.top_k = config.num_experts_per_tok
        self.multiplier = config.output_multiplier
        dense = VllmUnquantizedConfig()
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=dense,
            prefix=f"{prefix}.gate",
        )
        # route() reads the router weight directly, without a quantized matmul.
        if not isinstance(self.gate.quant_method, VllmUnquantizedLinearMethod):
            raise ValueError(
                f"Muse TPU reads the {prefix}.gate router weight in FP32; "
                "QUANTIZE_ON_LOAD_PREFIXES must not select it"
            )
        self.gate.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_local_experts, dtype=torch.float32),
        )
        # Read by moe_routing.select_experts.
        self.gate.routed_scaling_factor = config.routed_scaling_factor
        # torch's Module.__setattr__ stub only accepts modules and tensors.
        # pyrefly: ignore[bad-argument-type]
        self.gate.use_grouped_topk = False
        hidden = config.moe_hidden_size
        self.pre_expert_proj = ColumnParallelLinear(
            config.hidden_size,
            hidden,
            bias=False,
            gather_output=True,
            quant_config=dense,
            prefix=f"{prefix}.pre_expert_proj",
        )
        self.post_expert_proj = ColumnParallelLinear(
            hidden,
            config.hidden_size,
            bias=False,
            gather_output=True,
            quant_config=dense,
            prefix=f"{prefix}.post_expert_proj",
        )
        self.pre_expert_norm = MuseRMSNorm(
            hidden, config.rms_norm_eps, zero_centered=True
        )
        self.experts = FusedMoEFactory(
            num_experts=config.num_local_experts,
            top_k=self.top_k,
            hidden_size=hidden,
            intermediate_size=config.moe_intermediate_size,
            params_dtype=torch.bfloat16,
            renormalize=True,
            quant_config=quant_config or dense,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            routed_experts_cls=MuseRoutedExperts,
        ).routed_experts
        self.post_expert_norm = MuseRMSNormWithInputScale(
            hidden, config.post_norm_eps, zero_centered=True
        )
        # Build the router and rows ops outside torch.compile tracing.
        prebuild_fp32_matmul_op()
        parity = get_moe_reference_parity(get_current_vllm_config_or_none())
        if parity.silu_bf16:
            raise ValueError(
                "Muse TPU expert rows run SiLU in FP32 between the two GEMMs; "
                f"additional_config[{SILU_BF16_KEY!r}] is not supported"
            )
        prebuild_fused_moe_kernel(
            topk=self.top_k,
            activation="silu",
            use_ep=use_ep,
            topk_weights_float32=parity.topk_weights_float32,
            return_rows=True,
        )

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FP32 routing weights and int32 expert ids for each token."""
        # Expert selection is sensitive to logit rounding; TPU FP32 matmuls
        # otherwise run as BF16 passes.
        logits = fp32_matmul(x, cast(torch.Tensor, self.gate.weight))
        return moe_routing.select_experts(
            x.float(),
            logits * self.multiplier,
            topk=self.top_k,
            renormalize=True,
            scoring_fn="sigmoid",
            layer=self.gate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, ids = self.route(x)
        x, _ = self.pre_expert_proj(x)
        x = self.pre_expert_norm(x)
        if self.dp_size > 1:
            assert self.dp_group is not None
            # Gather once per TP lane so replicated tokens are counted only once.
            x = self.dp_group.all_gather(x, dim=0)
            weights = self.dp_group.all_gather(weights, dim=0)
            ids = self.dp_group.all_gather(ids, dim=0)
        rows = _expert_rows(self.experts, x, weights, ids)
        # Normalize the sum of TP partials, not individual shards.
        if self.tp_size > 1 and not self.use_ep:
            rows = tensor_model_parallel_all_reduce(rows.contiguous())
        rows = self.post_expert_norm(rows, out_dtype=torch.float32)
        result = (rows * weights.unsqueeze(-1)).sum(dim=1)
        if self.dp_size > 1:
            assert self.dp_group is not None
            # Return this DP engine's rows before the remaining TP reduction.
            result = self.dp_group.reduce_scatter(result.contiguous(), dim=0)
        if self.use_ep and self.tp_size > 1:
            # Under EP each TP rank holds different experts; add their sums.
            result = tensor_model_parallel_all_reduce(result.contiguous())
        result = result.to(x.dtype)
        result, _ = self.post_expert_proj(result)
        return result


class MuseAttention(nn.Module):
    def __init__(self, config, vllm_config: VllmConfig, layer_idx: int, prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        dense = VllmUnquantizedConfig()
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=dense,
            prefix=f"{prefix}.qkv_proj",
        )
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
            quant_config=dense,
            prefix=f"{prefix}.gate_proj",
        )
        # The flipped [in, out] layout makes a 1-token forward copy a transposed
        # weight every step; the vLLM layout does not, and longer forwards
        # compile the same either way.
        setattr(self.gate_proj, KEEP_VLLM_LAYOUT_ATTR, True)
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=dense,
            prefix=f"{prefix}.o_proj",
        )
        self.qk_norm = MuseRMSNorm(
            config.head_dim, config.rms_norm_eps, has_weight=False
        )
        self.o_norm = MuseRMSNorm(
            config.head_dim, config.rms_norm_eps, has_weight=False
        )
        use_rope, window = attention_layout(config, layer_idx)
        self.rotary_emb = (
            get_rope(
                config.head_dim,
                vllm_config.model_config.max_model_len,
                is_neox_style=True,
                rope_parameters={
                    **config.rope_parameters,
                    "rope_theta": config.layer_rope_theta[layer_idx],
                },
                dtype=torch.float32,
            )
            if use_rope
            else None
        )
        scale = config.head_dim**-0.5 * config.qk_scale_factor
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            per_layer_sliding_window=window,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        q, k = self.qk_norm(q), self.qk_norm(k)
        if self.rotary_emb is not None:
            # FP32 inputs keep the FP32 cos/sin table; round once afterwards.
            q, k = self.rotary_emb(positions, q.float(), k.float())
            q, k = q.to(x.dtype), cast(torch.Tensor, k).to(x.dtype)
        out = self.attn(q.flatten(1), k.flatten(1), v)
        out = out.reshape(-1, self.num_heads, self.head_dim)
        out = self.o_norm(out, out_dtype=torch.float32)
        gate, _ = self.gate_proj(x)
        out = out * gate.reshape_as(out).float().sigmoid()
        return self.o_proj(out.to(x.dtype).flatten(1))[0]


class MuseDecoderLayer(nn.Module):
    def __init__(self, config, vllm_config: VllmConfig, layer_idx: int, prefix: str):
        super().__init__()
        self.dtype = vllm_config.model_config.dtype
        self.self_attn = MuseAttention(
            config, vllm_config, layer_idx, f"{prefix}.self_attn"
        )
        self.mlp = MuseMoE(
            config,
            f"{prefix}.mlp",
            quant_config=vllm_config.quant_config,
            use_ep=vllm_config.parallel_config.enable_expert_parallel,
        )
        self.input_layernorm = MuseRMSNorm(
            config.hidden_size, config.rms_norm_eps, zero_centered=True
        )
        self.pre_feedforward_layernorm = MuseRMSNorm(
            config.hidden_size, config.rms_norm_eps, zero_centered=True
        )
        self.post_attn_norm = MuseRMSNorm(
            config.hidden_size, config.post_norm_eps, has_weight=False
        )
        self.post_feedforward_layernorm = MuseRMSNormWithInputScale(
            config.hidden_size, config.post_norm_eps, zero_centered=True
        )
        self.post_attention_residual_gate = MuseSpark1xNormPreservingResidualConnection(
            config.hidden_size, temperature=config.residual_gate_temperature
        )
        self.post_feedforward_residual_gate = (
            MuseSpark1xNormPreservingResidualConnection(
                config.hidden_size, temperature=config.residual_gate_temperature
            )
        )

    def forward(self, positions: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
        x = self.input_layernorm(stream, out_dtype=self.dtype)
        branch = self.self_attn(positions, x)
        # Keep post-norm values in FP32 through the residual blend.
        stream = self.post_attention_residual_gate(
            stream, self.post_attn_norm(branch, out_dtype=stream.dtype)
        )
        x = self.pre_feedforward_layernorm(stream, out_dtype=self.dtype)
        return self.post_feedforward_residual_gate(
            stream, self.post_feedforward_layernorm(self.mlp(x), out_dtype=stream.dtype)
        )


@support_torch_compile
class MuseModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        # The compile wrapper keys the on-disk compile cache on self.prefix.
        self.prefix = prefix
        config = vllm_config.model_config.hf_text_config
        self.residual_dtype = torch.float32
        self.dtype = cast(torch.dtype, vllm_config.model_config.dtype)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.tok_embeddings_norm = MuseRMSNorm(
            config.hidden_size, config.rms_norm_eps, has_weight=False
        )
        self.layers = nn.ModuleList(
            [
                MuseDecoderLayer(config, vllm_config, i, f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = MuseRMSNorm(config.hidden_size, config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_embeddings_norm(
            self.embed_tokens(input_ids), out_dtype=self.residual_dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            x = self.embed_input_ids(input_ids)
        else:
            x = inputs_embeds
        x = x.to(self.residual_dtype)
        for layer in self.layers:
            x = layer(positions, x)
        return self.norm(x, out_dtype=self.dtype)


class _MuseModelComponents(nn.Module):
    language_model: MuseModel


class MuseSparkForCausalLM(nn.Module):
    # The text-only model skips the checkpoint's vision and audio tensors.
    ignored_weight_prefixes: tuple[str, ...] = (
        "model.vision_tower.",
        "model.vision_adapter.",
        "model.vision_projection.",
        "model.audio_tower.",
    )
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".mlp.experts.post_expert_norm.": ".mlp.post_expert_norm."},
        orig_to_new_stacked={
            f".self_attn.{p}_proj.": (".self_attn.qkv_proj.", p) for p in "qkv"
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = config = cast(
            "MuseSparkTextConfig", vllm_config.model_config.hf_text_config
        )
        parallel = vllm_config.parallel_config
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.enable_eplb
        ):
            raise ValueError(
                "Muse Spark TPU supports TP, DP attention and EP; "
                "PP, context parallelism and EPLB are not supported"
            )
        self.use_ep = parallel.enable_expert_parallel
        if parallel.data_parallel_size > 1 and not self.use_ep:
            raise ValueError("Muse DP attention requires expert parallelism")
        if self.use_ep and parallel.expert_placement_strategy != "linear":
            raise ValueError("Muse EP requires linear expert placement")
        if any(kind != "sparse" for kind in config.mlp_layer_types):
            raise ValueError("Muse TPU expects routed MoE layers")
        if config.tie_word_embeddings:
            raise ValueError("Muse TPU expects an untied output projection")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("Muse TPU requires --dtype bfloat16")
        quant = vllm_config.model_config.quantization
        if quant not in (None, "modelopt_fp4"):
            raise ValueError(
                f"Muse TPU supports BF16 and NVFP4 checkpoints, got {quant}"
            )
        self._dummy_load = vllm_config.load_config.load_format == "dummy"
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_rank = get_ep_group().rank_in_group if self.use_ep else 0
        self.ep_size = get_ep_group().world_size if self.use_ep else 1
        expert_tp = 1 if self.use_ep else self.tp_size
        if self.use_ep and config.num_local_experts % self.ep_size:
            raise ValueError("Muse expert count must be divisible by EP")
        # NVFP4 block scales cover 16 values along K, and the down projection
        # contracts over the TP-sliced intermediate dimension.
        if quant == "modelopt_fp4" and config.moe_intermediate_size % (16 * expert_tp):
            raise ValueError("Muse NVFP4 TP partitions must align to 16-value blocks")
        model_prefix = (
            f"{prefix}.model.language_model" if prefix else "model.language_model"
        )
        self.model = _MuseModelComponents()
        self.model.language_model = MuseModel(
            vllm_config=vllm_config, prefix=model_prefix
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.multiplier = config.output_multiplier
        self.soft_cap = (
            None
            if config.final_logit_softcapping is None
            else MulTanhSoftCap(self.multiplier, config.final_logit_softcapping)
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(
            self.lm_head,
            hidden_states.to(cast(torch.Tensor, self.lm_head.weight).dtype),
        )
        if logits is None:
            return None
        if self.soft_cap is not None:
            return self.soft_cap(logits)
        return logits.float() * self.multiplier

    def process_weights_after_loading(self) -> None:
        if not self._dummy_load:
            return
        experts = (m for m in self.modules() if isinstance(m, MuseRoutedExperts))
        for layer, module in enumerate(experts):
            module.initialize_dummy_quantized_weights(
                1234 + 104729 * (layer * self.ep_size + self.ep_rank) + self.tp_rank
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def bf16_dense(weights: Iterable[tuple[str, torch.Tensor]]):
            for name, tensor in weights:
                # Vision and audio tensors are skipped before the BF16 check.
                if name.startswith(self.ignored_weight_prefixes):
                    continue
                if ".experts." not in name and (
                    name.endswith(("_scale", "_scale_2")) or tensor.dtype == torch.uint8
                ):
                    raise ValueError(
                        f"Muse TPU runs dense projections in BF16, but {name} is "
                        "quantized; list the module in the checkpoint's "
                        "modules_to_not_convert and store it in BF16"
                    )
                yield name, tensor

        loader = AutoWeightsLoader(self)
        return loader.load_weights(bf16_dense(weights), mapper=self.hf_to_vllm_mapper)
