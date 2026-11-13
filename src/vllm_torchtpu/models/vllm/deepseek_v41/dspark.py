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
"""TPU implementation of the DeepSeek-V4.1 DSpark speculative decoding draft model."""

from __future__ import annotations
import vllm_torchtpu.envs as envs

import os

try:
    import jax
except ImportError:
    pass

import re
import typing
from collections.abc import Callable, Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.offloader import NoopOffloader, set_offloader

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.tp_collectives import (
    tp_all_gather,
    tp_shard_slice,
    tp_token_shard_ok,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_flash_mhc_op import (  # noqa: E501
    get_mhc_final_collapse_norm_v41_op,
)
from vllm_torchtpu.models.vllm.deepseek_v4.model import _HostStagedParams
from vllm_torchtpu.models.vllm.deepseek_v41.model import (
    DeepseekV41DecoderLayer,
    _clear_stage_caches,
)

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _init_dspark_qmm_ops() -> tuple[Callable, Callable]:
    import jax
    import jax.numpy as jnp
    from torch_tpu._internal import pallas

    from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax

    def _zero_cat_qmm_jax(
        h0: jax.Array,
        h1: jax.Array,
        h2: jax.Array,
        w_q: jax.Array,
        w_s: jax.Array,
    ) -> jax.Array:
        orig_m = h0.shape[0]
        m_pad = (orig_m + 7) & ~7
        x_cat = jnp.concatenate([h0, h1, h2], axis=-1)
        if m_pad > orig_m:
            x_cat = jnp.pad(x_cat, ((0, m_pad - orig_m), (0, 0)))
        out = _quantized_matmul_jax(x_cat, w_q, w_s)
        return out[:orig_m] if m_pad > orig_m else out

    def _aligned_qmm_jax(
        x: jax.Array,
        w_q: jax.Array,
        w_s: jax.Array,
    ) -> jax.Array:
        orig_m = x.shape[0]
        m_pad = (orig_m + 7) & ~7
        x_pad = jnp.pad(x, ((0, m_pad - orig_m), (0, 0))) if m_pad > orig_m else x
        out = _quantized_matmul_jax(x_pad, w_q, w_s)
        return out[:orig_m] if m_pad > orig_m else out

    zero_cat_op = pallas.jax_op("pallas::dspark_zero_cat_qmm_kernel", _zero_cat_qmm_jax)
    aligned_qmm_op = pallas.jax_op(
        "pallas::dspark_aligned_qmm_kernel", _aligned_qmm_jax
    )

    def _fake_zero_cat_qmm(
        h0: torch.Tensor,
        h1: torch.Tensor,
        h2: torch.Tensor,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
    ) -> torch.Tensor:
        del h1, h2, w_s
        return torch.empty(
            (h0.shape[0], w_q.shape[-1]), dtype=h0.dtype, device=h0.device
        )

    def _fake_aligned_qmm(
        x: torch.Tensor,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
    ) -> torch.Tensor:
        del w_s
        return torch.empty((x.shape[0], w_q.shape[-1]), dtype=x.dtype, device=x.device)

    zero_cat_op.register_fake(_fake_zero_cat_qmm)
    aligned_qmm_op.register_fake(_fake_aligned_qmm)
    return zero_cat_op, aligned_qmm_op


try:
    _dspark_zero_cat_qmm_op, _dspark_aligned_qmm_op = _init_dspark_qmm_ops()
except ImportError:
    _dspark_zero_cat_qmm_op = None
    _dspark_aligned_qmm_op = None


class DSparkDeepseekV41Model(nn.Module):
    """DeepSeek-V4.1 DSpark 3-layer parallel draft transformer on TPU."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        set_offloader(NoopOffloader())
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = (
            getattr(config, "n_mtp_layers", None)
            or getattr(config, "num_nextn_predict_layers", None)
            or 3
        )

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        self.register_buffer(
            "tp_rank_idx",
            torch.tensor([tp_rank], dtype=torch.int64),
            persistent=False,
        )
        init_pre_mix = torch.zeros((config.hc_mult,), dtype=torch.float32)
        init_pre_mix[0] = 1.0
        self.register_buffer("init_pre_mix", init_pre_mix, persistent=False)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

        _ROPE_DICT.clear()
        # Draft layers use the smaller DSpark expert set.
        moe_overrides = {
            "n_routed_experts": config.dspark_n_routed_experts,
            "num_experts_per_tok": config.dspark_num_experts_per_tok,
        }
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(
                    vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    topk_indices_buffer=self.topk_indices_buffer,
                    engram_layout=None,
                    moe_overrides=moe_overrides,
                )
                for i in range(self.num_dspark_layers)
            ]
        )
        for layer in self.layers:
            layer.attn.non_causal_multi_token_decode = True

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self._fused_ctx_wkv_weight: torch.Tensor | None = None
        self._fused_ctx_wkv_scale: torch.Tensor | None = None
        self._main_proj_w: torch.Tensor | None = None
        self._main_proj_s: torch.Tensor | None = None
        self._main_proj_bf16_splits: tuple[torch.Tensor, ...] | None = None

        from vllm_torchtpu.models.vllm.deepseek_v41 import attention as _attn_mod
        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            get_vllm_model_wrapper_context,
        )

        _get_swa_store_op = getattr(_attn_mod, "_get_dspark_swa_store_op", None)
        if (
            callable(_get_swa_store_op)
            and get_vllm_model_wrapper_context().mesh is not None
        ):
            for _ps in (16, 32, 64, 128):
                _ = _get_swa_store_op(_ps)

    def process_weights_after_loading(self) -> None:
        from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
        from vllm_torchtpu.utils import synchronize_tensors

        mp_w = getattr(self.main_proj, "weight", None)
        mp_s = getattr(self.main_proj, "weight_scale", None)
        n_chunks = len(self.target_layer_ids)
        if mp_w is not None and n_chunks == 3:
            w_data = mp_w.data
            if mp_s is not None:
                if getattr(self.main_proj, KEEP_VLLM_LAYOUT_ATTR, False):
                    w_data = w_data.transpose(0, 1).contiguous()
                self._main_proj_w = w_data
                self._main_proj_s = mp_s.data.contiguous()
                if w_data.device.type == "tpu":
                    synchronize_tensors(
                        [self._main_proj_w, self._main_proj_s], wait=True
                    )
            elif w_data.dtype in (torch.bfloat16, torch.float16, torch.float32):
                if w_data.shape == (self.hidden_size, self.hidden_size * n_chunks):
                    splits = tuple(
                        c.contiguous() for c in w_data.split(self.hidden_size, dim=1)
                    )
                    self._main_proj_bf16_splits = splits
                    if w_data.device.type == "tpu":
                        synchronize_tensors(list(splits), wait=True)

        slices_w: list[torch.Tensor] = []
        slices_s: list[torch.Tensor] = []
        use_quant = False
        for layer in self.layers:
            attn = layer.attn
            attn.process_weights_after_loading(torch.bfloat16)
            if hasattr(attn, "wkv_weight") and hasattr(attn, "wkv_weight_scale"):
                use_quant = True
                slices_w.append(attn.wkv_weight)
                slices_s.append(attn.wkv_weight_scale)
            else:
                f_wqa = getattr(attn, "fused_wqa_wkv", None)
                w = getattr(f_wqa, "weight", None)
                if w is None or w.dtype not in (
                    torch.bfloat16,
                    torch.float16,
                    torch.float32,
                ):
                    self._fused_ctx_wkv_weight = None
                    self._fused_ctx_wkv_scale = None
                    return
                if getattr(f_wqa, KEEP_VLLM_LAYOUT_ATTR, False):
                    if w.shape[1] != attn.q_lora_rank + attn.head_dim:
                        self._fused_ctx_wkv_weight = None
                        self._fused_ctx_wkv_scale = None
                        return
                    slices_w.append(w[:, attn.q_lora_rank :].transpose(0, 1))
                else:
                    if w.shape[0] != attn.q_lora_rank + attn.head_dim:
                        self._fused_ctx_wkv_weight = None
                        self._fused_ctx_wkv_scale = None
                        return
                    slices_w.append(w[attn.q_lora_rank :])
        if (
            use_quant
            and len(slices_w) == len(self.layers)
            and len(slices_s) == len(self.layers)
        ):
            self._fused_ctx_wkv_weight = torch.cat(slices_w, dim=1).contiguous()
            self._fused_ctx_wkv_scale = torch.cat(slices_s, dim=-1).contiguous()
            if self._fused_ctx_wkv_weight.device.type == "tpu":
                synchronize_tensors(
                    [self._fused_ctx_wkv_weight, self._fused_ctx_wkv_scale],
                    wait=True,
                )
        elif slices_w and not use_quant and len(slices_w) == len(self.layers):
            self._fused_ctx_wkv_weight = torch.cat(slices_w, dim=0).contiguous()
            self._fused_ctx_wkv_scale = None
            if self._fused_ctx_wkv_weight.device.type == "tpu":
                synchronize_tensors([self._fused_ctx_wkv_weight], wait=True)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.layers)

    def combine_hidden_states(
        self,
        aux_hidden_states: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    ) -> torch.Tensor:
        mp_w = self._main_proj_w
        mp_s = self._main_proj_s
        if (
            mp_w is not None
            and mp_s is not None
            and _dspark_zero_cat_qmm_op is not None
        ):
            if isinstance(aux_hidden_states, (list, tuple)):
                h0, h1, h2 = aux_hidden_states
            else:
                h0, h1, h2 = aux_hidden_states.split(self.hidden_size, dim=-1)
            proj = _dspark_zero_cat_qmm_op(h0, h1, h2, mp_w, mp_s)
            return self.main_norm(proj)

        bf16_splits = self._main_proj_bf16_splits
        if bf16_splits is not None:
            w0, w1, w2 = bf16_splits
            if isinstance(aux_hidden_states, (list, tuple)):
                h0, h1, h2 = aux_hidden_states
            else:
                h0, h1, h2 = aux_hidden_states.split(self.hidden_size, dim=-1)
            proj = (
                torch.nn.functional.linear(h0, w0)
                + torch.nn.functional.linear(h1, w1)
                + torch.nn.functional.linear(h2, w2)
            )
            return self.main_norm(proj)

        if isinstance(aux_hidden_states, (list, tuple)):
            aux_hidden_states = torch.cat(aux_hidden_states, dim=-1)
        proj = self.main_proj(aux_hidden_states)
        if isinstance(proj, tuple):
            proj = proj[0]
        return self.main_norm(proj)

    def precompute_and_store_context_kv(
        self,
        target_hidden_states: torch.Tensor
        | tuple[torch.Tensor, ...]
        | list[torch.Tensor],
        positions: torch.Tensor,
        metadata: tuple,
    ) -> torch.Tensor:
        if positions.ndim > 1:
            positions = positions.flatten()
        tp_size = get_tensor_model_parallel_world_size()
        total_tokens = positions.shape[0]
        if tp_size > 1:
            _ = torch.ops.aten.alias.default(self.tp_rank_idx)
        use_token_shard = bool(
            envs.VLLM_DSV41_DSPARK_SHARDED_CTX_KV
        ) and tp_token_shard_ok(total_tokens, tp_size)
        if use_token_shard:
            tp_rank = get_tensor_model_parallel_rank()
            if isinstance(target_hidden_states, (list, tuple)):
                sharded_inputs: torch.Tensor | tuple[torch.Tensor, ...] = tuple(
                    tp_shard_slice(h, tp_size, tp_rank, self.tp_rank_idx)
                    for h in target_hidden_states
                )
            else:
                sharded_inputs = tp_shard_slice(
                    target_hidden_states, tp_size, tp_rank, self.tp_rank_idx
                )
            main_x = self.combine_hidden_states(sharded_inputs)
        else:
            main_x = self.combine_hidden_states(target_hidden_states)

        fused_wkv = self._fused_ctx_wkv_weight
        fused_scale = self._fused_ctx_wkv_scale
        if fused_wkv is not None:
            if fused_scale is not None and _dspark_aligned_qmm_op is not None:
                kv_all = _dspark_aligned_qmm_op(main_x, fused_wkv, fused_scale).view(
                    main_x.shape[0], len(self.layers), -1
                )
            elif fused_scale is None:
                kv_all = torch.nn.functional.linear(main_x, fused_wkv).view(
                    main_x.shape[0], len(self.layers), -1
                )
            else:
                kv_all = None
            if kv_all is not None:
                normed_layers = [
                    layer.attn.kv_norm(kv_all[:, layer_idx])
                    for layer_idx, layer in enumerate(self.layers)
                ]
                if use_token_shard:
                    kv_packed = tp_all_gather(
                        torch.cat(normed_layers, dim=-1), dim=0
                    ).view(total_tokens, len(self.layers), -1)
                    for layer_idx, layer in enumerate(self.layers):
                        layer.attn.store_context_kv(
                            kv_packed[:, layer_idx], positions, metadata[layer_idx]
                        )
                else:
                    for layer_idx, layer in enumerate(self.layers):
                        layer.attn.store_context_kv(
                            normed_layers[layer_idx], positions, metadata[layer_idx]
                        )
                return main_x
        if use_token_shard:
            main_x = tp_all_gather(main_x, dim=0)
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.attn
            kv = attn.project_wkv(main_x)
            kv = attn.kv_norm(kv)
            attn.store_context_kv(kv, positions, metadata[layer_idx])
        return main_x

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1 and envs.VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS >= 0:
            _ = torch.ops.aten.alias.default(self.tp_rank_idx)
        positions = torch.ops.aten.alias.default(positions)
        num_tokens = positions.shape[0]
        token_sharded_mhc = tp_token_shard_ok(num_tokens, tp_size)

        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        if token_sharded_mhc:
            tp_rank = get_tensor_model_parallel_rank()
            inputs_embeds = tp_shard_slice(
                inputs_embeds, tp_size, tp_rank, self.tp_rank_idx
            )
        pre_mix = inputs_embeds[:, :1].to(torch.float32) * 0.0 + self.init_pre_mix
        hidden_states = inputs_embeds.repeat(1, self.hc_mult).view(
            -1, self.hc_mult, inputs_embeds.shape[-1]
        )

        previous_x: torch.Tensor | None = None
        previous_post_mix: torch.Tensor | None = None
        previous_res_mix: torch.Tensor | None = None

        for layer in self.layers:
            (
                previous_x,
                hidden_states,
                pre_mix,
                previous_post_mix,
                previous_res_mix,
            ) = layer(
                positions,
                hidden_states,
                pre_mix,
                previous_x=previous_x,
                previous_post_mix=previous_post_mix,
                previous_res_mix=previous_res_mix,
                engram_hashes=None,
                token_sharded=token_sharded_mhc,
                tp_rank_idx=self.tp_rank_idx,
            )
        _clear_stage_caches(self.layers)

        out = get_mhc_final_collapse_norm_v41_op(self.rms_norm_eps)(
            previous_x,
            hidden_states,
            previous_post_mix,
            previous_res_mix,
            pre_mix,
            self.norm.weight,
        )
        if token_sharded_mhc:
            out = tp_all_gather(out, dim=0)
        return out


class DSparkDeepseekV41ForCausalLM(nn.Module):
    """vLLM model-registry entry for DeepSeek-V4.1 DSpark (`DSparkV41DraftModel`)."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DSparkDeepseekV41Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self._tp_markov_meta: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = (
            None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(
        self,
        aux_hidden_states: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    ) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_attn_causal(self) -> list[bool]:
        return self.model.get_draft_attn_causal()

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def tpu_precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
        positions: torch.Tensor,
        metadata: tuple,
    ) -> torch.Tensor:
        return self.model.precompute_and_store_context_kv(
            hidden_states, positions, metadata
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def _remap_dspark_name(self, name: str) -> str | None:
        if name.endswith(".bias_vl"):
            return None
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)
        if rest.startswith("confidence_head."):
            return None
        if rest.startswith("markov_head.embed."):
            return "model.markov_head.markov_w1." + rest.removeprefix(
                "markov_head.embed."
            )
        if rest.startswith("markov_head.head."):
            return "model.markov_head.markov_w2." + rest.removeprefix(
                "markov_head.head."
            )
        if rest.startswith(("main_proj.", "main_norm.", "norm.", "markov_head.")):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.dspark_n_routed_experts,
        )
        if getattr(envs, "VLLM_DSV41_MODULO_EXPERT_BALANCE", False):
            from vllm.distributed import get_ep_group

            ep_size = get_ep_group().world_size
            num_experts = self.config.dspark_n_routed_experts
            if ep_size > 1 and num_experts % ep_size == 0:
                g_local = num_experts // ep_size
                expert_mapping = [
                    (pname, wname, (eid % ep_size) * g_local + (eid // ep_size), sid)
                    for pname, wname, eid, sid in expert_mapping
                ]
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        staged = _HostStagedParams()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        try:
            for orig_name, loaded_weight in weights:
                mapped = self._remap_dspark_name(orig_name)
                if mapped is None:
                    continue
                name = mapped

                if name.endswith(".scale"):
                    suffix = (
                        expert_scale_suffix
                        if _EXPERT_SCALE_RE.search(name)
                        else ".weight_scale_inv"
                    )
                    name = name.removesuffix(".scale") + suffix
                if ".shared_experts.w2" in name:
                    name = name.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )

                if ".experts." in name:
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for (
                        param_name,
                        weight_name,
                        expert_id,
                        expert_shard_id,
                    ) in expert_mapping:
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if name_mapped not in params_dict:
                            continue
                        param = staged.stage(name_mapped, params_dict[name_mapped])
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            loaded_params.add(name_mapped)
                            break
                    continue

                is_layer_param = name.startswith("model.layers.")
                for (
                    param_name,
                    weight_name,
                    stacked_shard_id,
                ) in stacked_params_mapping:
                    if not is_layer_param or weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        break
                    param = staged.stage(name, params_dict[name])
                    param.weight_loader(param, loaded_weight, stacked_shard_id)
                    loaded_params.add(name)
                    break
                else:
                    if "attn_sink" in name:
                        if name not in params_dict:
                            continue
                        narrow = loaded_weight[head_start:head_end]
                        param = staged.stage(name, params_dict[name])
                        param.data[: narrow.shape[0]].copy_(narrow)
                        loaded_params.add(name)
                        continue
                    if name.endswith(".ffn.gate.bias"):
                        name = name.replace(
                            ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                        )
                    if name not in params_dict:
                        continue
                    param = staged.stage(name, params_dict[name])
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
        finally:
            staged.flush_all()

        for layer in self.model.layers:
            layer._ensure_mhc_buffers(force=True)

        logger.info(
            "DSparkDeepseekV41ForCausalLM loaded %d parameters",
            len(loaded_params),
        )
        return loaded_params

    def init_tp_markov_meta(self) -> None:
        from vllm_torchtpu.utils import synchronize_tensors

        lm_head = self.lm_head
        markov_w2 = getattr(self.model.markov_head, "markov_w2", None)
        shard_indices = getattr(lm_head, "shard_indices", None)
        tp_size = int(getattr(lm_head, "tp_size", 1))
        if (
            lm_head is None
            or markov_w2 is None
            or shard_indices is None
            or tp_size <= 1
        ):
            self._tp_markov_meta = None
            self._tp_lm_w_pad = None
            return
        dev = lm_head.weight.device
        shard_cols = int(lm_head.weight.shape[0])
        shard_pad = ((shard_cols + 127) // 128) * 128
        start_idx = int(shard_indices.org_vocab_start_index)
        end_idx = int(shard_indices.org_vocab_end_index)
        valid_vocab = end_idx - start_idx
        w2_raw = markov_w2.weight[start_idx:end_idx]
        w2_pad = (
            torch.cat(
                (
                    w2_raw,
                    torch.zeros(
                        (shard_pad - valid_vocab, w2_raw.shape[1]),
                        dtype=w2_raw.dtype,
                        device=w2_raw.device,
                    ),
                ),
                dim=0,
            )
            if valid_vocab < shard_pad
            else w2_raw.clone().contiguous()
        ).to(device=dev)
        pad_mask = (
            torch.arange(shard_pad, dtype=torch.int32, device=dev) >= valid_vocab
        ).view(1, shard_pad)
        pad_bias = torch.where(
            pad_mask,
            torch.full((1, shard_pad), float("-inf"), dtype=w2_raw.dtype, device=dev),
            torch.zeros((1, shard_pad), dtype=w2_raw.dtype, device=dev),
        )
        lm_w = lm_head.weight
        lm_w_pad: torch.Tensor | None = None
        if (
            getattr(lm_head, "weight_scale", None) is None
            and lm_w.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and lm_w.ndim == 2
        ):
            lm_w_pad = (
                torch.cat(
                    (
                        lm_w,
                        torch.zeros(
                            (shard_pad - shard_cols, lm_w.shape[1]),
                            dtype=lm_w.dtype,
                            device=dev,
                        ),
                    ),
                    dim=0,
                ).contiguous()
                if shard_pad > shard_cols
                else lm_w.contiguous()
            )
        start_t = torch.tensor([start_idx], dtype=torch.int32, device=dev)
        if dev.type == "tpu":
            sync_list = [w2_pad, pad_bias, start_t]
            if lm_w_pad is not None:
                sync_list.append(lm_w_pad)
            synchronize_tensors(sync_list, wait=True)
        self._tp_markov_meta = (w2_pad, pad_bias, start_t)
        self._tp_lm_w_pad = lm_w_pad

    def sample_draft_tokens(
        self,
        pred_hidden: torch.Tensor,
        prev_init: torch.Tensor,
        num_pred: int,
    ) -> torch.Tensor:
        padded_num_reqs = prev_init.shape[0]
        lm_head = self.lm_head
        markov_w2 = getattr(self.model.markov_head, "markov_w2", None)
        tp_size = int(getattr(lm_head, "tp_size", 1))
        meta = self._tp_markov_meta
        pred_kcb = (
            pred_hidden.view(padded_num_reqs, num_pred, -1)
            .transpose(0, 1)
            .contiguous()
            .view(num_pred * padded_num_reqs, -1)
        )
        if lm_head is None or markov_w2 is None or tp_size <= 1 or meta is None:
            logits_kbv = self.compute_draft_logits(pred_kcb).view(
                num_pred, padded_num_reqs, -1
            )
            draft_tokens = []
            prev = prev_init
            for idx in range(num_pred):
                bias = self.markov_bias(self.markov_embed(prev))
                step = self.map_draft_to_target((logits_kbv[idx] + bias).argmax(dim=-1))
                draft_tokens.append(step)
                prev = step
            return torch.stack(draft_tokens, dim=1)

        shard_cols = int(lm_head.weight.shape[0])
        w2_pad, pad_bias, start_t = meta
        lm_w_pad = getattr(self, "_tp_lm_w_pad", None)
        shard_pad = int(w2_pad.shape[0])
        if lm_w_pad is not None:
            logits_kbv = torch.nn.functional.linear(pred_kcb, lm_w_pad).view(
                num_pred, padded_num_reqs, shard_pad
            )
        else:
            logits_local = self.logits_processor._get_logits(
                pred_kcb, lm_head, None, skip_gather=True
            )
            assert logits_local is not None
            if shard_pad > shard_cols:
                logits_local = torch.cat(
                    (
                        logits_local,
                        torch.zeros(
                            (logits_local.shape[0], shard_pad - shard_cols),
                            dtype=logits_local.dtype,
                            device=logits_local.device,
                        ),
                    ),
                    dim=-1,
                )
            logits_kbv = logits_local.view(num_pred, padded_num_reqs, shard_pad)

        draft_tokens = []
        prev = prev_init
        for idx in range(num_pred):
            bias_local = torch.nn.functional.linear(self.markov_embed(prev), w2_pad)
            local_val, local_idx = (logits_kbv[idx] + bias_local + pad_bias).max(
                dim=-1, keepdim=True
            )
            packed_local = torch.cat(
                [
                    local_val.to(torch.float32).view(torch.int32),
                    local_idx.to(torch.int32) + start_t,
                ],
                dim=-1,
            )
            packed_all = tp_all_gather(packed_local, dim=0).view(
                tp_size, padded_num_reqs, 2
            )
            all_vals = packed_all[..., 0].view(torch.float32)
            all_idxs = packed_all[..., 1]
            best_shard = all_vals.argmax(dim=0, keepdim=True)
            step = all_idxs.gather(0, best_shard).squeeze(0)
            draft_tokens.append(step)
            prev = step
        return torch.stack(draft_tokens, dim=1)

    def process_weights_after_loading(self) -> None:
        for layer in self.model.layers:
            layer._ensure_mhc_buffers()
            ffn_mod = getattr(layer, "ffn", None)
            if ffn_mod is not None and hasattr(
                ffn_mod, "process_weights_after_loading"
            ):
                ffn_mod.process_weights_after_loading()
        self.model.process_weights_after_loading()
        self.init_tp_markov_meta()
