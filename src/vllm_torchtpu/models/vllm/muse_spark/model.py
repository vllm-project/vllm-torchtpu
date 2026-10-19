# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark text inference with DP attention and TP or DP×TP EP experts."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4Config
from vllm_torchtpu.layers.adapter.quantization.unquantized import VllmUnquantizedConfig
from vllm_torchtpu.logger import init_logger

from .layers import (
    MuseResidual,
    MuseRMSNorm,
    MuseRotaryEmbedding,
    attention_layout,
    route_experts,
    selected_expert_outputs,
)
from .weights import (
    decode_expert_weight,
    expert_ep_slice,
    expert_tp_slice,
    selected_packed_expert_outputs,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from .audio import AutoEncoder
    from .config import MuseSparkTextConfig
    from .vision import NaViT, NaViTAdapter


class MuseExperts(nn.Module):
    gate_up_proj: nn.Parameter
    down_proj: nn.Parameter
    gate_up_proj_weight_scale: nn.Parameter
    down_proj_weight_scale: nn.Parameter
    gate_up_proj_weight_scale_2: nn.Parameter
    down_proj_weight_scale_2: nn.Parameter
    experts_start: torch.Tensor | None
    post_expert_norm: MuseRMSNorm

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        packed: bool = False,
        use_ep: bool = False,
    ):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        expert_tp = 1 if use_ep else tp
        if intermediate_size % expert_tp:
            raise ValueError("Muse intermediate dimension must be divisible by TP")
        local = intermediate_size // expert_tp
        self.packed = packed
        self.use_ep = use_ep
        if use_ep:
            ep_group = get_ep_group()
            if num_experts % ep_group.world_size:
                raise ValueError("Muse expert count must be divisible by EP")
            local_experts = num_experts // ep_group.world_size
            start = ep_group.rank_in_group * local_experts
        else:
            local_experts, start = num_experts, 0
        self.register_buffer(
            "experts_start",
            torch.tensor([start], dtype=torch.int32) if use_ep else None,
            persistent=False,
        )
        from .expert_matmul import get_grouped_experts_op

        self._grouped_experts = get_grouped_experts_op(num_experts if use_ep else None)
        num_experts = local_experts
        if packed:
            if hidden_size % 16 or local % 16:
                raise ValueError("Packed Muse expert dimensions must align to block 16")
            for name, rows, columns in (
                ("gate_up_proj", 2 * local, hidden_size),
                ("down_proj", hidden_size, local),
            ):
                self.register_parameter(
                    name,
                    nn.Parameter(
                        torch.zeros(num_experts, columns // 2, rows, dtype=torch.uint8),
                        requires_grad=False,
                    ),
                )
                self.register_parameter(
                    name + "_weight_scale",
                    nn.Parameter(
                        torch.empty(
                            num_experts, columns // 16, rows, dtype=torch.uint8
                        ),
                        requires_grad=False,
                    ),
                )
                self.register_parameter(
                    name + "_weight_scale_2",
                    nn.Parameter(
                        torch.empty(
                            (num_experts, 2)
                            if name == "gate_up_proj"
                            else (num_experts,),
                            dtype=torch.float32,
                        ),
                        requires_grad=False,
                    ),
                )
            return
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * local), requires_grad=False
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, local, hidden_size), requires_grad=False
        )

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        if x.device.type == "tpu":
            return self._grouped_experts(
                x,
                ids,
                self.gate_up_proj,
                self.down_proj,
                self.gate_up_proj_weight_scale if self.packed else None,
                self.down_proj_weight_scale if self.packed else None,
                self.gate_up_proj_weight_scale_2 if self.packed else None,
                self.down_proj_weight_scale_2 if self.packed else None,
                self.experts_start,
            )
        owned = None
        if self.use_ep:
            assert self.experts_start is not None
            local_ids = ids - self.experts_start
            owned = (local_ids >= 0) & (local_ids < self.gate_up_proj.shape[0])
            ids = local_ids.clamp(0, self.gate_up_proj.shape[0] - 1)
        if self.packed:
            rows = selected_packed_expert_outputs(
                x,
                ids,
                self.gate_up_proj.transpose(1, 2),
                self.down_proj.transpose(1, 2),
                self.gate_up_proj_weight_scale.transpose(1, 2),
                self.down_proj_weight_scale.transpose(1, 2),
                self.gate_up_proj_weight_scale_2,
                self.down_proj_weight_scale_2,
            )
        else:
            rows = selected_expert_outputs(x, ids, self.gate_up_proj, self.down_proj)
        return rows if owned is None else rows.masked_fill(~owned[..., None], 0)


class MuseMoE(nn.Module):
    def __init__(self, config, prefix: str, packed: bool = False, use_ep: bool = False):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.use_ep = use_ep
        self.dp_group = get_dp_group() if use_ep else None
        self.dp_size = self.dp_group.world_size if self.dp_group is not None else 1
        self.top_k = config.num_experts_per_tok
        self.multiplier = config.output_multiplier
        self.scoring = "sigmoid"
        self.division_eps = 1e-15
        self.route_scale = config.routed_scaling_factor
        dense = VllmUnquantizedConfig()
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=dense,
            prefix=f"{prefix}.gate",
        )
        self.gate.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_local_experts, dtype=torch.float32),
        )
        hidden = config.moe_hidden_size
        self.pre_expert_proj = ColumnParallelLinear(
            config.hidden_size,
            hidden,
            bias=False,
            gather_output=True,
            quant_config=dense,
            prefix=f"{prefix}.pre_expert_proj",
        )
        self.post_expert_proj = RowParallelLinear(
            hidden,
            config.hidden_size,
            bias=False,
            input_is_parallel=False,
            quant_config=dense,
            prefix=f"{prefix}.post_expert_proj",
        )
        self.pre_expert_norm = MuseRMSNorm(
            hidden, config.rms_norm_eps, zero_centered=True
        )
        self.experts = MuseExperts(
            hidden,
            config.moe_intermediate_size,
            config.num_local_experts,
            packed=packed,
            use_ep=use_ep,
        )
        self.experts.post_expert_norm = MuseRMSNorm(
            hidden, config.post_norm_eps, zero_centered=True, weight_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.gate(x.float())
        weights, ids = route_experts(
            logits,
            cast(torch.Tensor, self.gate.e_score_correction_bias),
            top_k=self.top_k,
            multiplier=self.multiplier,
            scoring=self.scoring,
            eps=self.division_eps,
            route_scale=self.route_scale,
        )
        x, _ = self.pre_expert_proj(x)
        x = self.pre_expert_norm(x)
        if self.dp_size > 1:
            assert self.dp_group is not None
            # Gather once per TP lane so replicated tokens are counted only once.
            x = self.dp_group.all_gather(x, dim=0)
            weights = self.dp_group.all_gather(weights, dim=0)
            ids = self.dp_group.all_gather(ids, dim=0)
        rows = self.experts(x, ids)
        # Normalize the sum of TP partials, not individual shards.
        if self.tp_size > 1 and not self.use_ep:
            rows = tensor_model_parallel_all_reduce(rows.contiguous())
        rows = self.experts.post_expert_norm(rows, out_dtype=torch.float32)
        result = (rows * weights.unsqueeze(-1)).sum(dim=1)
        if self.dp_size > 1:
            assert self.dp_group is not None
            # Return this DP engine's rows before the remaining TP reduction.
            result = self.dp_group.reduce_scatter(result.contiguous(), dim=0)
        if self.use_ep and self.tp_size > 1:
            # Normalize whole experts before the weighted FP32 reduction.
            result = tensor_model_parallel_all_reduce(result.contiguous())
        result = result.to(x.dtype)
        result, _ = self.post_expert_proj(result)
        return result


class MuseAttention(nn.Module):
    def __init__(self, config, vllm_config: VllmConfig, layer_idx: int, prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        # Pad explicitly: the native 64-wide cache packs K/V differently.
        self.attention_head_dim = max(128, self.head_dim)
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
            MuseRotaryEmbedding(
                config.head_dim,
                vllm_config.model_config.max_model_len,
                config.layer_rope_theta[layer_idx],
            )
            if use_rope
            else None
        )
        scale = config.head_dim**-0.5 * config.qk_scale_factor
        self.attn = Attention(
            self.num_heads,
            self.attention_head_dim,
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
            q, k = self.rotary_emb(q, positions), self.rotary_emb(k, positions)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        if self.attention_head_dim != self.head_dim:
            padding = (0, self.attention_head_dim - self.head_dim)
            q = torch.nn.functional.pad(q, padding)
            k = torch.nn.functional.pad(k, padding)
            v = torch.nn.functional.pad(v, padding)
        out = self.attn(q.flatten(1), k.flatten(1), v.flatten(1))
        out = out.reshape(-1, self.num_heads, self.attention_head_dim)[
            ..., : self.head_dim
        ]
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
        moe_prefix = f"{prefix}.mlp"
        packed = (
            vllm_config.model_config.quantization == "modelopt_fp4"
            and cast(dict[str, Any], vllm_config.additional_config).get(
                "muse_packed_experts", get_tensor_model_parallel_world_size() <= 16
            )
            and not cast(VllmNvfp4Config, vllm_config.quant_config).is_layer_excluded(
                moe_prefix + ".experts"
            )
        )
        self.mlp = MuseMoE(
            config,
            moe_prefix,
            packed=packed,
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
        self.post_feedforward_layernorm = MuseRMSNorm(
            config.hidden_size,
            config.post_norm_eps,
            zero_centered=True,
            weight_first=True,
        )
        self.post_attention_residual_gate = MuseResidual(
            config.hidden_size, config.residual_gate_temperature
        )
        self.post_feedforward_residual_gate = MuseResidual(
            config.hidden_size, config.residual_gate_temperature
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
    vision_tower: "NaViT"
    vision_adapter: "NaViTAdapter"
    vision_projection: RowParallelLinear
    audio_tower: "AutoEncoder"


class MuseSparkForCausalLM(nn.Module):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

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
        if quant == "modelopt_fp4":
            quant_config = cast(VllmNvfp4Config, vllm_config.quant_config)
            quant_config.exclude_modules = list(
                vllm_config.model_config.hf_config.quantization_config[
                    "modules_to_not_convert"
                ]
            )
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_rank = get_ep_group().rank_in_group if self.use_ep else 0
        self.ep_size = get_ep_group().world_size if self.use_ep else 1
        expert_tp = 1 if self.use_ep else self.tp_size
        if self.use_ep and config.num_local_experts % self.ep_size:
            raise ValueError("Muse expert count must be divisible by EP")
        if quant == "modelopt_fp4" and config.moe_intermediate_size % (16 * expert_tp):
            raise ValueError("Muse NVFP4 TP partitions must align to 16-value blocks")
        torch.set_float32_matmul_precision("highest")
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
            params_dtype=torch.bfloat16,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.logits_processor.head_dtype = cast(torch.Tensor, self.lm_head.weight).dtype
        self.multiplier = config.output_multiplier
        self.soft_cap = config.final_logit_softcapping
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
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
        logits = logits.float() * self.multiplier
        if self.soft_cap is not None:
            logits = self.soft_cap * torch.tanh(logits / self.soft_cap)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = {
            name: param
            for name, param in self.named_parameters()
            if name.startswith(("model.language_model.", "lm_head."))
        }
        buffers = dict(self.named_buffers())
        pending: dict[str, dict[str, torch.Tensor]] = {}
        loaded: set[str] = set()
        qkv_parts: dict[str, set[str]] = {}
        seen: set[str] = set()
        for name, tensor in weights:
            if name.startswith(
                (
                    "model.vision_tower.",
                    "model.vision_adapter.",
                    "model.vision_projection.",
                    "model.audio_tower.",
                )
            ):
                continue
            if name in seen:
                raise ValueError(f"Duplicate Muse checkpoint tensor: {name}")
            seen.add(name)
            if ".experts." in name and name.rsplit(".", 1)[-1].startswith(
                ("gate_up_proj", "down_proj")
            ):
                module_name, suffix = name.rsplit(".", 1)
                if suffix in ("gate_up_proj_input_scale", "down_proj_input_scale"):
                    continue  # BF16 activations do not consume NVFP4 activation scales.
                local = (
                    expert_ep_slice(tensor, self.ep_rank, self.ep_size)
                    if self.use_ep
                    else expert_tp_slice(tensor, suffix, self.tp_rank, self.tp_size)
                )
                if (
                    suffix in ("gate_up_proj", "down_proj")
                    and tensor.dtype == torch.bfloat16
                ):
                    local = local.transpose(1, 2).contiguous()
                    if tuple(local.shape) != tuple(params[name].shape):
                        raise ValueError(f"Muse expert shape mismatch: {name}")
                    params[name].data.copy_(local)
                    loaded.add(name)
                    continue
                group = pending.setdefault(module_name, {})
                group[suffix] = local
                for weight_name in ("gate_up_proj", "down_proj"):
                    components = (
                        weight_name,
                        weight_name + "_weight_scale",
                        weight_name + "_weight_scale_2",
                    )
                    if all(k in group for k in components):
                        full_name = module_name + "." + weight_name
                        if params[full_name].dtype == torch.uint8:
                            # Transpose to K-major on CPU once, before device transfer.
                            packed = group[weight_name].transpose(1, 2).contiguous()
                            block_scale = group[weight_name + "_weight_scale"]
                            if block_scale.dtype != torch.float8_e4m3fn:
                                raise ValueError(
                                    f"Muse NVFP4 block scales must be E4M3: {full_name}"
                                )
                            scales = (
                                block_scale.view(torch.uint8)
                                .transpose(1, 2)
                                .contiguous()
                            )
                            global_scale = group[weight_name + "_weight_scale_2"]
                            scale_name = full_name + "_weight_scale"
                            global_name = full_name + "_weight_scale_2"
                            if (
                                packed.shape != params[full_name].shape
                                or scales.shape != params[scale_name].shape
                                or global_scale.shape != params[global_name].shape
                            ):
                                raise ValueError(
                                    f"Muse packed expert shape mismatch: {full_name}"
                                )
                            params[full_name].data.copy_(packed)
                            params[scale_name].data.copy_(scales)
                            params[global_name].data.copy_(global_scale)
                            loaded.update((full_name, scale_name, global_name))
                            for key in components:
                                del group[key]
                            continue
                        decoded = decode_expert_weight(group, weight_name)
                        if tuple(decoded.shape) != tuple(params[full_name].shape):
                            raise ValueError(f"Muse expert shape mismatch: {full_name}")
                        params[full_name].data.copy_(decoded)
                        loaded.add(full_name)
                        if self.tp_rank == 0 and weight_name == "down_proj":
                            logger.info("Decoded Muse NVFP4 experts: %s", module_name)
                        for key in components:
                            del group[key]
                        del decoded
                if not group:
                    del pending[module_name]
                continue
            mapped = name
            shard = None
            for projection, shard_id in (
                ("q_proj", "q"),
                ("k_proj", "k"),
                ("v_proj", "v"),
            ):
                if f".self_attn.{projection}." in name:
                    mapped = name.replace(f".{projection}.", ".qkv_proj.")
                    shard = shard_id
                    break
            param = params.get(mapped, buffers.get(mapped))
            if param is None:
                raise ValueError(f"Unexpected Muse text tensor: {name}")
            loader = getattr(param, "weight_loader", None)
            if shard is not None:
                assert loader is not None
                loader(param, tensor, shard)
                qkv_parts.setdefault(mapped, set()).add(shard)
                if qkv_parts[mapped] == {"q", "k", "v"}:
                    loaded.add(mapped)
            elif loader is not None:
                loader(param, tensor)
                loaded.add(mapped)
            else:
                if tuple(param.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"Muse weight shape mismatch: {name}: "
                        f"{tensor.shape} != {param.shape}"
                    )
                param.data.copy_(tensor)
                loaded.add(mapped)
        required = set(params) | {
            n for n in buffers if n.endswith(".gate.e_score_correction_bias")
        }
        missing = required - loaded
        if pending or missing:
            raise ValueError(
                f"Incomplete Muse checkpoint: missing={sorted(missing)}, "
                f"unmatched expert components={list(pending)}"
            )
        logger.info(
            "Loaded %d Muse text tensors on TP rank %d/%d, EP rank %d/%d; "
            "experts use %s, "
            "compute is BF16",
            len(loaded),
            self.tp_rank,
            self.tp_size,
            self.ep_rank,
            self.ep_size,
            "EP" if self.use_ep else "TP",
        )
        return loaded
