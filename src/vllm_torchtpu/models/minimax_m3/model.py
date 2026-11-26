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
"""MiniMax M3 (text) on TPU."""

import contextlib
from collections.abc import Iterable, Iterator

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed import pallas_all_reduce
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3 import norm_hoist, quant, sp_o_proj
from vllm_torchtpu.models.minimax_m3.attention import MiniMaxM3SparseAttention
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse, sp_norm
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP, MiniMaxM3MoE
from vllm_torchtpu.models.minimax_m3.ops.sp_norm import get_add_norm_quant_op
from vllm_torchtpu.models.minimax_m3.quant import CT_TO_MODELOPT_EXPERT_SUFFIX

logger = init_logger(__name__)


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, layer_id: int, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.self_attn = MiniMaxM3SparseAttention(
            config,
            quant_config,
            vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
            layer_id=layer_id,
        )
        # Dense layers name their FFN `mlp`, MoE layers `block_sparse_moe`.
        self.is_moe_layer = bool(config.moe_layer_freq[layer_id])
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3MoE(
                config, quant_config, prefix=f"{prefix}.block_sparse_moe"
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config,
                config.dense_intermediate_size,
                quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > 1 and envs.TPU_MINIMAX_M3_SP_FUSED_NORM:
            self._add_norm_quant = get_add_norm_quant_op(config.rms_norm_eps)

    @contextlib.contextmanager
    def _layer_owned_reductions(self) -> Iterator[None]:
        """While tracing a sequence-parallel step, o_proj and the FFN return
        per-rank partial sums and the layer reduces them; every other trace
        keeps their own all-reduces, so decode programs are unchanged."""
        o_proj = self.self_attn.o_proj
        if self.is_moe_layer:
            cfg = self.block_sparse_moe.experts.moe_config
            saved = (o_proj.reduce_results, cfg.skip_final_all_reduce)
            o_proj.reduce_results, cfg.skip_final_all_reduce = False, True
        else:
            down = self.mlp.down_proj
            saved = (o_proj.reduce_results, down.reduce_results)
            o_proj.reduce_results = down.reduce_results = False
        try:
            yield
        finally:
            if self.is_moe_layer:
                o_proj.reduce_results, cfg.skip_final_all_reduce = saved
            else:
                o_proj.reduce_results, down.reduce_results = saved

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        index_reuse: dict | None = None,
        attn_rs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual-deferred layer. `index_reuse` (MSA index reuse on) is the
        step's shared selection dict (`MiniMaxM3SparseAttention.forward`);
        ``attn_rs`` (large steps, TPU_MINIMAX_M3_SP_MODE=attn) reduce-scatters
        the attention output instead of all-reducing it."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif envs.TPU_MINIMAX_M3_FP8_CAST_NORM and quant.fp8_norm_applies(
            self.self_attn.qkv_proj, hidden_states.shape[0]
        ):
            hidden_states, residual = quant.fused_add_rms_norm_e4m3(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.variance_epsilon,
            )
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if attn_rs:
            with self._layer_owned_reductions():
                return self._attn_rs(positions, hidden_states, residual)
        if index_reuse is None:
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states, index_reuse)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        return ffn(hidden_states), residual

    def _attn_rs(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each rank gets the new residual stream of its token shard,
        normalizes it and all-gathers it as the FFN input; the attention
        partial rides the FFN all-reduce. The returned residual is the
        pre-attention one, so the next fused add + norm sees
        ``ffn + attn + residual``."""
        attn = self.self_attn(positions, hidden_states)
        shard = _reduce_scatter_rows(attn + residual / self.tp_size)
        normed = _all_gather_rows(
            self.post_attention_layernorm(shard), envs.TPU_MINIMAX_M3_SP_FP8_GATHER
        )
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        out = ffn(normed) + attn
        return tensor_model_parallel_all_reduce(out), residual

    def forward_sp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        shard: torch.Tensor | None,
        update: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Sequence-parallel layer: the residual stream stays a token shard.

        ``shard`` is this rank's rows of the residual stream (None for the
        first layer, whose ``hidden_states`` is the full embedding) and
        ``update`` the previous layer's not yet added reduce-scattered FFN
        output (None when already in ``shard``). Norms run on the shard,
        all-gathers feed attention and the FFN, and their partial outputs are
        reduce-scattered back onto the shard. Returns the residual shard
        entering this layer (None for the first), the shard after attention
        and this layer's FFN update.
        """
        with self._layer_owned_reductions():
            return self._forward_sp(positions, hidden_states, shard, update)

    def _norm_gather(
        self,
        norm: GemmaRMSNorm,
        shard: torch.Tensor,
        update: torch.Tensor | None,
        fp8: bool,
        e4m3_out: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """(shard + update, all-gathered norm(shard + update)): one fused pass
        over the shard when it feeds an e4m3 gather and an update is pending.
        ``e4m3_out`` returns the gathered e4m3 rows and their scales."""
        fused = envs.TPU_MINIMAX_M3_SP_FUSED_NORM and _fits_sp_norm(shard)
        if update is not None and fp8 and fused:
            weight = norm_hoist.gemma_weight(norm)
            shard, q, scale = self._add_norm_quant(shard, update, weight)
        else:
            if update is not None:
                shard = shard + update
            normed = norm(shard)
            if not fp8:
                return shard, tensor_model_parallel_all_gather(normed, 0)
            q, scale = _quantize_rows(normed)
        q, scale = _gather_rows_e4m3(q, scale)
        if e4m3_out:
            return shard, (q, scale)
        return shard, (q.float() * scale).to(shard.dtype)

    def _forward_sp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        shard: torch.Tensor | None,
        update: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        layer_input = None
        o_proj = self.self_attn.o_proj
        if shard is None:
            normed = self.input_layernorm(hidden_states)
            if sp_o_proj.applies(o_proj):
                attn = self.self_attn(positions, normed, project=False)
                rows = sp_o_proj.local_rows(hidden_states)
                shard = sp_o_proj.sp_o_proj(o_proj, attn) + rows
            else:
                attn = self.self_attn(positions, normed)
                shard = _reduce_scatter_rows(attn + hidden_states / self.tp_size)
        else:
            shard, normed = self._norm_gather(
                self.input_layernorm,
                shard,
                update,
                envs.TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER,
                envs.TPU_MINIMAX_M3_SP_E4M3_QKV,
            )
            layer_input = shard
            if sp_o_proj.applies(o_proj):
                attn = self.self_attn(positions, normed, project=False)
                update = sp_o_proj.sp_o_proj(o_proj, attn)
            else:
                attn = self.self_attn(positions, normed)
                update = _reduce_scatter_rows(attn)
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        shard, normed = self._norm_gather(
            self.post_attention_layernorm,
            shard,
            update,
            envs.TPU_MINIMAX_M3_SP_FP8_GATHER,
            envs.TPU_MINIMAX_M3_SP_E4M3_QKV,
        )
        return layer_input, shard, _reduce_scatter_rows(ffn(normed))


def _sp_mode() -> str:
    return envs.TPU_MINIMAX_M3_SP_MODE if envs.TPU_MINIMAX_M3_SP_MIN_TOKENS else "off"


def _sp_step(num_tokens: int, tp_size: int) -> bool:
    """Whether a step of ``num_tokens`` runs the sequence-parallel path.

    Only inside a warmup trace, which gives each side of this branch its own
    trace (compilation/shape_variants.py); eager forwards and any other trace
    keep the plain path.
    """
    return bool(
        torch.compiler.is_compiling()
        and shape_variants.aot_cache_tag()
        and num_tokens >= envs.TPU_MINIMAX_M3_SP_MIN_TOKENS
        and num_tokens % tp_size == 0
    )


def _reduce_scatter_rows(x: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter token rows over TP (rank r gets rows [r T/TP, (r + 1)
    T/TP) of the sum). TPU_MINIMAX_M3_SP_QRS (int8-payload
    qar_reduce_scatter) or TPU_MINIMAX_M3_SP_HIER_RS (hierrs_tc) sends them
    through the armed row reduce-scatter op when it is armed for the TP group."""
    if envs.TPU_MINIMAX_M3_SP_QRS or envs.TPU_MINIMAX_M3_SP_HIER_RS:
        out = pallas_all_reduce.reduce_scatter(get_tp_group().ranks, x)
        if out is not None:
            return out
    return tensor_model_parallel_reduce_scatter(x, 0)


_FP8_MAX = 448.0


def _all_gather_rows(x: torch.Tensor, fp8: bool) -> torch.Tensor:
    """All-gather token rows over TP; ``fp8`` sends each bf16 row as e4m3 with
    a per-row f32 scale (gathered natively: a bitcast to a wider carrier costs
    an 8-bit relayout per call)."""
    if not fp8:
        return tensor_model_parallel_all_gather(x, 0)
    q, scale = _gather_rows_e4m3(*_quantize_rows(x))
    return (q.float() * scale).to(x.dtype)


def _fits_sp_norm(shard: torch.Tensor) -> bool:
    """Whether `sp_norm.add_norm_quant` tiles this shard's rows."""
    rows = shard.shape[0]
    return rows <= sp_norm.BLOCK_ROWS or rows % sp_norm.BLOCK_ROWS == 0


def _quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 rows of ``x`` and their per-row f32 scales ``amax / 448``."""
    scale = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / _FP8_MAX
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def _gather_rows_e4m3(
    q: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # TPU_TP_PALLAS_SMALL_ALL_GATHER: the f32 row scales (a few KiB) in one
    # Pallas call (None when it is not armed: XLA's all-gather as before).
    scales = None
    if envs.TPU_TP_PALLAS_SMALL_ALL_GATHER:
        scales = pallas_all_reduce.all_gather_small(get_tp_group().ranks, scale)
    if scales is None:
        scales = tensor_model_parallel_all_gather(scale, 0)
    return tensor_model_parallel_all_gather(q, 0), scales


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0}
)
class MiniMaxM3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        # Keys this model's compile-artifact cache directory.
        self.prefix = prefix
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleList(
            MiniMaxM3DecoderLayer(vllm_config, idx, prefix=f"{prefix}.layers.{idx}")
            for idx in range(config.num_hidden_layers)
        )
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.sp_mode = _sp_mode() if self.tp_size > 1 else "off"
        if self.sp_mode != "off":
            logger.info(
                "MiniMax-M3 sequence parallelism %s on steps of >= %d tokens "
                "(multiple of TP %d): fp8 gather %s, fp8 attn gather %s, fused "
                "norm %s, e4m3 qkv %s, a2a o_proj %s (e4m3 %s)",
                self.sp_mode,
                envs.TPU_MINIMAX_M3_SP_MIN_TOKENS,
                self.tp_size,
                envs.TPU_MINIMAX_M3_SP_FP8_GATHER,
                envs.TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER,
                envs.TPU_MINIMAX_M3_SP_FUSED_NORM,
                envs.TPU_MINIMAX_M3_SP_E4M3_QKV,
                envs.TPU_MINIMAX_M3_SP_A2A_OPROJ,
                envs.TPU_MINIMAX_M3_SP_A2A_E4M3,
            )
        # Residual-stream entries of these layers (k = output of layer k-1,
        # residual added, before the next norm) are also returned; set by
        # the DSpark drafter before the first trace.
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        # MSA index reuse (`index_reuse`): the layers share one dict per step;
        # only when some layer's op runs it (else the calls are unchanged).
        self.index_reuse = any(
            layer.self_attn.index_reuse_full is not None for layer in self.layers
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        aux_hidden_states = []
        if 0 in self.aux_hidden_state_layers:
            aux_hidden_states.append(inputs_embeds)
        # Sequence parallelism (TPU_MINIMAX_M3_SP_MODE) only on large steps,
        # MSA index reuse only on decode steps: a step takes at most one.
        sp_mode = getattr(self, "sp_mode", "off")
        sp_step = sp_mode != "off" and _sp_step(inputs_embeds.shape[0], self.tp_size)
        if sp_step and sp_mode == "full":
            shard, update = None, None
            for idx, layer in enumerate(self.layers):
                # Layer idx materializes its input residual (entry idx).
                layer_input, shard, update = layer.forward_sp(
                    positions, inputs_embeds, shard, update
                )
                if idx > 0 and idx in self.aux_hidden_state_layers:
                    aux_hidden_states.append(
                        tensor_model_parallel_all_gather(layer_input, 0)
                    )
            shard = shard + update
            if len(self.layers) in self.aux_hidden_state_layers:
                aux_hidden_states.append(tensor_model_parallel_all_gather(shard, 0))
            hidden_states = tensor_model_parallel_all_gather(self.norm(shard), 0)
        else:
            hidden_states, residual = inputs_embeds, None
            if sp_step:
                kwargs = {"attn_rs": True}
            elif getattr(self, "index_reuse", False):
                kwargs = {"index_reuse": {}}
            else:
                kwargs = {}
            for idx, layer in enumerate(self.layers):
                hidden_states, residual = layer(
                    positions, hidden_states, residual, **kwargs
                )
                if idx + 1 in self.aux_hidden_state_layers:
                    aux_hidden_states.append(hidden_states + residual)
            hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # (param, checkpoint shard, shard id). Leading dots keep `.q_proj`
        # from matching `index_q_proj`.
        stacked_params_mapping: list[tuple[str, str, int | str]] = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_proj", ".index_q_proj", "index_q"),
            (".qkv_proj", ".index_k_proj", "index_k"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            name = name.replace(".weight_scale_inv", ".weight_scale")
            if ".experts." in name:
                for ct_suffix, modelopt_suffix in CT_TO_MODELOPT_EXPERT_SUFFIX.items():
                    name = name.replace(ct_suffix, modelopt_suffix)
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    raise ValueError(f"Unexpected expert weight {name}")
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        self._configure_max_free()
        return loaded_params

    def _configure_max_free(self) -> None:
        """TPU_MINIMAX_M3_MSA_MAX_FREE=1: bounds every layer's MSA scores from
        the loaded q / k norms and turns the max-free softmax on under the
        kernels' limit (`msa_attention_lse.configure_max_free`). Off: no-op."""
        if not envs.TPU_MINIMAX_M3_MSA_MAX_FREE:
            return
        bounds = [
            layer.self_attn.score_bound()
            for layer in self.layers
            if isinstance(getattr(layer, "self_attn", None), MiniMaxM3SparseAttention)
        ]
        if not bounds:
            return
        msa_attention_lse.configure_max_free(max(bounds))  # logs the bound


class MiniMaxM3SparseForCausalLM(nn.Module):
    """Text-only MiniMax M3; also serves the vision-language architecture."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.": "",
            "vision_tower.": None,
            "multi_modal_projector.": None,
            "patch_merge_mlp.": None,
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.model = MiniMaxM3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        from vllm_torchtpu import envs as tpu_envs

        if tpu_envs.TPU_FP8_LM_HEAD:
            from vllm_torchtpu.layers.adapter.quantization.fp8_lm_head import (
                Fp8LMHeadMethod,
            )

            # A head_dtype override reads lm_head.weight unscaled.
            if getattr(self.logits_processor, "head_dtype", None) not in (
                None,
                torch.bfloat16,
            ):
                raise NotImplementedError(
                    "TPU_FP8_LM_HEAD does not support a head_dtype override."
                )
            self.lm_head.quant_method = Fp8LMHeadMethod()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        num_layers = len(self.model.layers)
        invalid = [idx for idx in layers if not 0 <= idx <= num_layers]
        if invalid:
            raise ValueError(f"Invalid MiniMax M3 aux hidden-state layers: {invalid}")
        self.model.aux_hidden_state_layers = tuple(layers)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if intermediate_tensors is not None:
            raise NotImplementedError("MiniMax M3 on TPU does not support PP")
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def process_weights_after_loading(self) -> None:
        """Model-level post-load hook (vLLM calls it after the per-layer
        quant finalize): TPU_MINIMAX_M3_NORM_WEIGHT_HOIST."""
        if envs.TPU_MINIMAX_M3_NORM_WEIGHT_HOIST:
            norm_hoist.hoist(self.model)

    def compute_logits_shard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's vocab shard of ``compute_logits`` (pad columns
        included), without the all-gather."""
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
