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
from vllm import ir
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

from vllm_torchtpu import envs, model_loader_patches
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed import pallas_all_reduce
from vllm_torchtpu.layers.adapter.quantization.fp8_lm_head import Fp8LMHeadMethod
from vllm_torchtpu.models.minimax_m3.attention import MiniMaxM3SparseAttention
from vllm_torchtpu.models.minimax_m3.expert_placement import ExpertPlacement
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP, MiniMaxM3MoE
from vllm_torchtpu.models.minimax_m3.quant import (
    CT_TO_MODELOPT_EXPERT_SUFFIX,
    MiniMaxM3Mxfp8LinearMethod,
)
from vllm_torchtpu.utils import fresh_buffer

# Per-routed-expert rows outside the experts: they follow the expert placement.
_PER_EXPERT_ROWS = (".block_sparse_moe.gate.weight", ".e_score_correction_bias")


class FoldedGemmaRMSNorm(GemmaRMSNorm):
    """GemmaRMSNorm that applies ``1 + w`` folded to f32 once after loading
    (`fold_weight`), instead of converting the weight on every forward."""

    def fold_weight(self) -> None:
        self.folded_weight = fresh_buffer(self.weight.float() + 1.0)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``weight``: a copy of ``folded_weight`` placed for this call (an
        all-reduce carried it into VMEM), else None."""
        weight = self.folded_weight if weight is None else weight
        if residual is None:
            return ir.ops.rms_norm(x, weight, self.variance_epsilon)
        return ir.ops.fused_add_rms_norm(x, residual, weight, self.variance_epsilon)


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
        self.input_layernorm = FoldedGemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = FoldedGemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.tp_size = get_tensor_model_parallel_world_size()

    @contextlib.contextmanager
    def _layer_owned_reductions(self, attention: bool = True) -> Iterator[None]:
        """o_proj and the FFN return per-rank partial sums and the layer
        reduces them: in a sequence-parallel step both, and with
        ``attention=False`` (a step whose FFN all-reduce carries the next QKV
        weight) the FFN's alone. Other traces keep their own all-reduces."""
        o_proj = self.self_attn.o_proj
        if self.is_moe_layer:
            cfg = self.block_sparse_moe.experts.moe_config
            saved = (o_proj.reduce_results, cfg.skip_final_all_reduce)
            cfg.skip_final_all_reduce = True
        else:
            down = self.mlp.down_proj
            saved = (o_proj.reduce_results, down.reduce_results)
            down.reduce_results = False
        if attention:
            o_proj.reduce_results = False
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
        attn_rs: bool = False,
        carried: tuple[torch.Tensor, torch.Tensor] | None = None,
        carry: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """Residual-deferred layer; ``attn_rs`` (large steps) reduce-scatters
        the attention output instead of all-reducing it.

        ``carry`` (not with ``attn_rs``): the next layer's (input norm, QKV)
        weights (`MiniMaxM3Model._qkv_carries`), which the FFN all-reduce
        copies into VMEM while it runs; the copies are returned third, for
        the next layer's ``carried`` (None when nothing is carried)."""
        norm_weight, qkv_weight = carried or (None, None)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual, weight=norm_weight
            )
        if not attn_rs:
            hidden_states = self.self_attn(positions, hidden_states, qkv_weight)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
            if carry is None:
                return ffn(hidden_states), residual, None
            with self._layer_owned_reductions(attention=False):
                partial = ffn(hidden_states)
            reduced = pallas_all_reduce.all_reduce_carrying(
                get_tp_group().ranks, partial, carry
            )
            if reduced is None:
                raise RuntimeError("the FFN output is not a tensor the op reduces")
            out, copies = reduced
            return out, residual, copies
        assert carry is None
        with self._layer_owned_reductions():
            return *self._attn_rs(positions, hidden_states, residual), None

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
        shard = tensor_model_parallel_reduce_scatter(attn + residual / self.tp_size, 0)
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
    ) -> torch.Tensor:
        """Sequence-parallel layer: the residual stream stays a token shard.

        ``shard`` is this rank's rows of the residual stream (None for the
        first layer, whose ``hidden_states`` is the full embedding). Norms run
        on the shard, all-gathers feed attention and the FFN, and their
        partial outputs are reduce-scattered back onto the shard.
        """
        with self._layer_owned_reductions():
            return self._forward_sp(positions, hidden_states, shard)

    def _forward_sp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        shard: torch.Tensor | None,
    ) -> torch.Tensor:
        if shard is None:
            attn = self.self_attn(positions, self.input_layernorm(hidden_states))
            shard = tensor_model_parallel_reduce_scatter(
                attn + hidden_states / self.tp_size, 0
            )
        else:
            normed = _all_gather_rows(
                self.input_layernorm(shard), envs.TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER
            )
            attn = self.self_attn(positions, normed)
            shard = shard + tensor_model_parallel_reduce_scatter(attn, 0)
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        normed = _all_gather_rows(
            self.post_attention_layernorm(shard), envs.TPU_MINIMAX_M3_SP_FP8_GATHER
        )
        return shard + tensor_model_parallel_reduce_scatter(ffn(normed), 0)


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


_FP8_MAX = 448.0


def _all_gather_rows(x: torch.Tensor, fp8: bool) -> torch.Tensor:
    """All-gather token rows over TP; ``fp8`` sends each bf16 row as e4m3 with
    a per-row f32 scale (gathered natively: a bitcast to a wider carrier costs
    an 8-bit relayout per call)."""
    if not fp8:
        return tensor_model_parallel_all_gather(x, 0)
    scale = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / _FP8_MAX
    q = tensor_model_parallel_all_gather((x.float() / scale).to(torch.float8_e4m3fn), 0)
    scale = tensor_model_parallel_all_gather(scale, 0)
    return (q.float() * scale).to(x.dtype)


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0}
)
class MiniMaxM3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        placement = ExpertPlacement(
            vllm_config.model_config.hf_text_config.num_local_experts
        )
        # The EP weight filter keeps the checkpoint experts of this rank's blocks.
        model_loader_patches.set_expert_slot_map(
            placement.owner_slots if placement else None
        )
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype
        # Keys this model's compile-artifact cache directory.
        self.prefix = prefix
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleList(
            MiniMaxM3DecoderLayer(vllm_config, idx, prefix=f"{prefix}.layers.{idx}")
            for idx in range(config.num_hidden_layers)
        )
        self.norm = FoldedGemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.sp_mode = _sp_mode() if self.tp_size > 1 else "off"
        # Residual-stream entries of these layers (k = output of layer k-1,
        # residual added, before the next norm) are also returned; set by
        # the DSpark drafter before the first trace.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

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
        sp_step = self.sp_mode != "off" and _sp_step(
            inputs_embeds.shape[0], self.tp_size
        )
        if sp_step and self.sp_mode == "full":
            shard = None
            for idx, layer in enumerate(self.layers):
                shard = layer.forward_sp(positions, inputs_embeds, shard)
                if idx + 1 in self.aux_hidden_state_layers:
                    aux_hidden_states.append(tensor_model_parallel_all_gather(shard, 0))
            hidden_states = tensor_model_parallel_all_gather(self.norm(shard), 0)
        else:
            hidden_states, residual, carried = inputs_embeds, None, None
            carries = self._qkv_carries(sp_step)
            for idx, layer in enumerate(self.layers):
                hidden_states, residual, carried = layer(
                    positions,
                    hidden_states,
                    residual,
                    sp_step,
                    carried,
                    carries[idx],
                )
                if idx + 1 in self.aux_hidden_state_layers:
                    aux_hidden_states.append(hidden_states + residual)
            hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _qkv_carries(
        self, sp_step: bool
    ) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        """Per layer, the next layer's (input norm, QKV) weights for its FFN
        all-reduce to carry into VMEM (`MiniMaxM3DecoderLayer.forward`), or
        None.

        `fp8_linear` reads the QKV weight from VMEM; XLA's own prefetch of it
        cannot start before the FFN all-reduce (the MoE kernel holds VMEM
        until then) and lands as a copy just ahead of the QKV projection.
        Only when every step the op carries for reads the copy: a step that
        reads the weight from HBM would pay the copy for nothing. The op
        takes bf16 activations, as `fp8_linear` does. The norm weight, which
        the next layer reads first, rides along so that this trace takes its
        weights in the order the traces without a carry do (sequence-parallel
        steps): traces hand compiled buckets to one another only when their
        inputs match (compilation/shape_variants.py)."""
        none = [None] * len(self.layers)
        rows = pallas_all_reduce.carry_rows(get_tp_group().ranks)
        if sp_step or not rows or self.dtype != torch.bfloat16:
            return none
        carries = []
        for layer in self.layers[1:]:
            qkv = layer.self_attn.qkv_proj
            method = qkv.quant_method
            fp8 = isinstance(method, MiniMaxM3Mxfp8LinearMethod)
            weight = method.vmem_weight(qkv, rows) if fp8 else None
            norm = layer.input_layernorm.folded_weight
            carries.append(None if weight is None else (norm, weight))
        return carries + [None]

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
        placement = ExpertPlacement(self.config.num_local_experts)
        for name, loaded_weight in weights:
            name = name.replace(".weight_scale_inv", ".weight_scale")
            if name.endswith(_PER_EXPERT_ROWS):
                loaded_weight = placement.to_slots(name, loaded_weight)
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
                    for physical in placement.physical_experts(name, expert_id):
                        param.weight_loader(
                            param,
                            loaded_weight,
                            name,
                            shard_id=shard_id,
                            expert_id=physical,
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
        return loaded_params


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
        if envs.TPU_FP8_LM_HEAD:
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

    def compute_logits_shard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's vocab shard of ``compute_logits`` (pad columns
        included), without the all-gather."""
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )

    def process_weights_after_loading(self) -> None:
        """Fold the Gemma norms' ``1 + w`` once (vLLM's model-level hook, after
        real and dummy loads alike)."""
        for module in self.model.modules():
            if isinstance(module, (FoldedGemmaRMSNorm, MiniMaxM3SparseAttention)):
                module.fold_weight()
