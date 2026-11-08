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
"""ERNIE 5.5 (``Ernie5_5_MoeForCausalLM``) on TPU: DeepSeek-V4 backbone pieces + ERNIE
blocks.

43 decoder layers on hc_mult = 4 mHC streams: 37 HCA layers (``ErnieHCAAttention``: DSv4
compress-128 + SWA attention with ERNIE's gate and VHA) and 6 MLA-DSA layers
(``ErnieDSAAttention``), a dense SiTU MLP at layer 0 and the multi-router latent MoE
elsewhere; hc_head collapse, final norm, untied lm_head + MultiMax.

mHC uses the DeepSeek-V4 TPU ops, whose pre-mix RMS factor is rsqrt(mean + eps); ERNIE's
is 1 / (sqrt(mean) + eps) (vLLM-ERNIE mhc_kernels.py:84). With eps = 1e-6 over 16384
stream values the two differ by ~1e-7 relative, below the bf16 output resolution
(measured against a float64 reference of the ERNIE formula).
"""

import re
import typing
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor import (
    VllmDeepseekCompressor,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op import (
    MHCOps,
    get_mhc_ops,
    get_mhc_post_op,
)
from vllm_torchtpu.models.vllm.deepseek_v4.layers import mhc_collapse_head
from vllm_torchtpu.models.vllm.deepseek_v4.model import _HostStagedParams
from vllm_torchtpu.models.vllm.ernie5_5.layers import (
    ErnieHCAAttention,
    ErnieMLP,
    ErnieMoE,
    multimax,
)

logger = init_logger(__name__)


def _dsa_attention_cls():
    """MLA-DSA attention for compress_ratios[i] <= 0 (layers 7, 14, ..., 42)."""
    from vllm_torchtpu.models.vllm.ernie5_5.dsa import ErnieDSAAttention

    return ErnieDSAAttention


class Ernie5_5DecoderLayer(nn.Module):
    """One layer: mHC pre -> attn_norm -> attention -> fused mHC post/pre -> ffn_norm ->
    FFN (post deferred)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        m = re.search(r"layers\.(\d+)", prefix)
        self.layer_idx = int(m.group(1)) if m else -1
        self.hidden_size, self.rms_norm_eps = config.hidden_size, config.rms_norm_eps
        if config.is_dsa_layer(self.layer_idx):
            self.attn = _dsa_attention_cls()(
                vllm_config,
                prefix=f"{prefix}.attn",
                topk_indices_buffer=topk_indices_buffer,
            )
        else:
            self.attn = ErnieHCAAttention(vllm_config, prefix=f"{prefix}.attn")
        if self.layer_idx < config.first_k_dense_replace:
            self.ffn = ErnieMLP(
                config,
                config.intermediate_size,
                vllm_config.quant_config,
                prefix=f"{prefix}.ffn",
            )
        else:
            self.ffn = ErnieMoE(
                config, vllm_config.quant_config, prefix=f"{prefix}.ffn"
            )
        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps = (
            config.hc_mult,
            config.hc_sinkhorn_iters,
            config.hc_eps,
        )
        self.hc_post_alpha = 2.0
        mix, dim = (2 + self.hc_mult) * self.hc_mult, self.hc_mult * self.hidden_size
        for kind in ("attn", "ffn"):
            setattr(
                self,
                f"hc_{kind}_fn",
                nn.Parameter(
                    torch.empty(mix, dim, dtype=torch.float32), requires_grad=False
                ),
            )
            setattr(
                self,
                f"hc_{kind}_base",
                nn.Parameter(
                    torch.empty(mix, dtype=torch.float32), requires_grad=False
                ),
            )
            setattr(
                self,
                f"hc_{kind}_scale",
                nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False),
            )

    @property
    def mhc_ops(self) -> MHCOps:
        ops = self.__dict__.get("_mhc_ops_instance")
        if ops is None:
            ops = get_mhc_ops(
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
            )
            object.__setattr__(self, "_mhc_ops_instance", ops)
        return ops

    def forward(self, x, positions, post_mix=None, res_mix=None, residual=None):
        if residual is None:
            residual = x
            post_mix, res_mix, x = self.mhc_ops.pre(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
        else:
            residual, post_mix, res_mix, x = self.mhc_ops.fused(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
            )
        x = self.attn(positions, self.attn_norm(x))
        residual, post_mix, res_mix, x = self.mhc_ops.fused(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        x = self.ffn(self.ffn_norm(x))
        return x, residual, post_mix, res_mix


class Ernie5_5Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.hc_mult, self.hc_eps, self.rms_norm_eps = (
            config.hc_mult,
            config.hc_eps,
            config.rms_norm_eps,
        )
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        # one [max_num_batched_tokens, index_topk + window] row table shared by the 6
        # DSA layers
        from vllm_torchtpu.models.vllm.ernie5_5.dsa import make_topk_buffer

        self.topk_indices_buffer = make_topk_buffer(vllm_config)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Ernie5_5DecoderLayer(
                vllm_config, prefix=prefix, topk_indices_buffer=self.topk_indices_buffer
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = (
            RMSNorm(config.hidden_size, self.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

    @property
    def mhc_post_op(self):
        op = self.__dict__.get("_mhc_post_op_instance")
        if op is None:
            op = get_mhc_post_op()
            object.__setattr__(self, "_mhc_post_op_instance", op)
        return op

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size, dtype, device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size), dtype=dtype
                )
            }
        )

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if get_pp_group().is_first_rank:
            h = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            # The first mHC pre sees the embedding broadcast into hc_mult identical
            # streams (vLLM-ERNIE mhc_pre_broadcast_tilelang: residual_out = x
            # repeated).
            h = h.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            # non-first PP ranks always receive intermediate_tensors
            # pyrefly: ignore[unsupported-operation]
            h = typing.cast(torch.Tensor, intermediate_tensors["hidden_states"])
        residual, post_mix, res_mix = None, None, None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            h, residual, post_mix, res_mix = layer(
                h, positions, post_mix, res_mix, residual
            )
        if post_mix is not None:
            h = self.mhc_post_op(h, residual, post_mix, res_mix)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": h})
        h = mhc_collapse_head(
            h,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        return self.norm(h)


# Checkpoint name -> module name, in order, first match wins (vLLM-ERNIE
# model.py:588-638 patterns, renamed to the DeepSeek-V4 TPU attention's parameters).
# Each pattern spans whole dotted segments.
_SUBSTR = [
    (".self_attn.core_attention.softmax_offset", ".attn.attn_sink"),
    (".self_attn.core_attention.", ".attn."),
    (".self_attn.gate_proj.", ".attn.o_gate."),
    (".self_attn.vha_postmix_", ".attn.vha."),
    (".self_attn.q_layernorm.", ".attn.q_norm."),
    (".self_attn.kv_layernorm.", ".attn.kv_norm."),
    (".self_attn.linear_q_up_proj.", ".attn.wq_b."),
    (".self_attn.linear_o_group_proj_scale_inv", ".attn.wo_a.weight_scale_inv"),
    (".self_attn.linear_o_group_proj", ".attn.wo_a.weight"),
    (".self_attn.o_proj.", ".attn.wo_b."),
    (".self_attn.", ".attn."),
    (".block_sparse_moe.", ".ffn."),
    (".mlp.", ".ffn."),
    (".input_layernorm.", ".attn_norm."),
    (".post_attention_layernorm.", ".ffn_norm."),
    (".self_attention_hyper_connection.mapping_proj.weight", ".hc_attn_fn"),
    (".self_attention_hyper_connection.bias", ".hc_attn_base"),
    (".mlp_hyper_connection.mapping_proj.weight", ".hc_ffn_fn"),
    (".mlp_hyper_connection.bias", ".hc_ffn_base"),
]
# (checkpoint substring, fused param substring, shard id)
_STACKED = [
    (".attn.linear_q_down_proj.", ".attn.fused_wqa_wkv.", 0),
    (".attn.linear_kv_proj.", ".attn.fused_wqa_wkv.", 1),
    (".attn.q_a_proj.", ".attn.fused_qkv_a_proj.", 0),
    (".attn.kv_a_proj_with_mqa.", ".attn.fused_qkv_a_proj.", 1),
    (".compressor.linear_wkv.", ".compressor.fused_wkv_wgate.", 0),
    (".compressor.linear_wgate.", ".compressor.fused_wkv_wgate.", 1),
    (".indexer.wk.", ".indexer.wk_weights_proj.", 0),
    (".indexer.weights_proj.", ".indexer.wk_weights_proj.", 1),
    (".shared_experts.w1.", ".shared_experts.gate_up_proj.", 0),
    (".shared_experts.w3.", ".shared_experts.gate_up_proj.", 1),
    (".ffn.gate_proj.", ".ffn.gate_up_proj.", 0),
    (".ffn.up_proj.", ".ffn.gate_up_proj.", 1),
]
_HC_ALPHA = {"alpha_pre": 0, "alpha_post": 1, "alpha_res": 2}


def ernie_name(name: str) -> tuple[str, int | None]:
    """(module parameter name, shard id or None) of a checkpoint tensor name."""
    name = name.replace("model.mhc_contract.", "model.").replace(
        "model.lm_head.", "lm_head."
    )
    m = re.search(
        r"\.(self_attention|mlp)_hyper_connection\.(alpha_pre|alpha_post|alpha_res)$",
        name,
    )
    if m:
        kind = "attn" if m.group(1) == "self_attention" else "ffn"
        return name[: m.start()] + f".hc_{kind}_scale", _HC_ALPHA[m.group(2)]
    for a, b in _SUBSTR:
        if a in name:
            name = name.replace(a, b, 1)
            break
    # after the block_sparse_moe -> ffn rename above (one _SUBSTR entry per name), so a
    # separate step
    name = name.replace(".shared_experts.w2.", ".shared_experts.down_proj.")
    if name.endswith(".ffn.gate.weight_1"):
        return name[: -len("_1")], 1
    if name.endswith(".ffn.gate.weight"):
        return name, 0
    for a, b, sid in _STACKED:
        if a in name:
            return name.replace(a, b, 1), sid
    return name, None


class Ernie5_5ForCausalLM(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = Ernie5_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.lm_head.register_buffer(
                "multimax_ranges", torch.zeros(4, dtype=torch.float32)
            )
            self.lm_head.register_buffer(
                "multimax_ts", torch.zeros(4, dtype=torch.float32)
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        return multimax(logits, self.lm_head.multimax_ranges, self.lm_head.multimax_ts)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Checkpoint names -> module params (ernie_name); experts through the fused-MoE
        expert mapping; mHC mixers transposed ([in, out] in the checkpoint); per-head
        sinks narrowed to this rank; the ECHO / MTP layer (layers.num_hidden_layers)
        skipped. Writes are staged on host and flushed once per parameter (DSv4 TPU).
        Checkpoint tensors that map to no parameter are kept in
        ``self.unmapped_checkpoint_tensors`` and logged."""
        params = dict(self.named_parameters())
        bufs = dict(self.named_buffers())
        loaded: set[str] = set()
        expert_mapping = self.get_expert_mapping()
        tp, rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        n_layers = self.config.num_hidden_layers
        staged = _HostStagedParams()
        unmapped: list[str] = []
        try:
            for ckpt_name, w in weights:
                m = re.match(r"model\.layers\.(\d+)\.", ckpt_name)
                if m and int(m.group(1)) >= n_layers:
                    continue  # ECHO / MTP layer: speculative decoding only
                name, shard = ernie_name(ckpt_name)
                if name.startswith("lm_head.multimax_"):
                    bufs[name].copy_(w.float())
                    loaded.add(name)
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if ".experts." in name:
                    # a parameter exists for it (under EP the expert may live on
                    # another rank)
                    hit = False
                    for (
                        param_name,
                        weight_name,
                        expert_id,
                        expert_shard,
                    ) in expert_mapping:
                        if weight_name not in name:
                            continue
                        mapped = name.replace(weight_name, param_name)
                        if mapped not in params:
                            continue
                        hit = True
                        p = staged.stage(mapped, params[mapped])
                        if p.weight_loader(
                            p,
                            w,
                            mapped,
                            shard_id=expert_shard,
                            expert_id=expert_id,
                            return_success=True,
                        ):
                            loaded.add(mapped)
                            break
                    if not hit:
                        unmapped.append(ckpt_name)
                    continue
                if name not in params:
                    unmapped.append(ckpt_name)
                    continue
                p = staged.stage(name, params[name])
                leaf = name.rpartition(".")[2]
                if leaf.startswith("hc_") and leaf.endswith("_fn"):
                    p.data.copy_(w.t())
                elif leaf in (
                    "hc_attn_scale",
                    "hc_ffn_scale",
                ):  # alpha_pre / post / res -> rows 0 / 1 / 2
                    p.data[shard : shard + 1].copy_(w)
                elif name.endswith(".ffn.gate.weight"):
                    rows = w.shape[0]
                    p.data[shard * rows : (shard + 1) * rows].copy_(w)
                elif leaf == "attn_sink":
                    n = w.shape[0] // tp
                    p.data[:n].copy_(w[rank * n : (rank + 1) * n])
                elif shard is not None:
                    p.weight_loader(p, w, shard)
                else:
                    getattr(p, "weight_loader", default_weight_loader)(p, w)
                loaded.add(name)
        finally:
            staged.flush_all()
        self.unmapped_checkpoint_tensors = unmapped
        if unmapped:
            logger.warning(
                "ERNIE load_weights: %d checkpoint tensors map to no parameter, "
                "e.g. %s",
                len(unmapped),
                unmapped[:5],
            )
        for module in self.modules():
            if isinstance(module, VllmDeepseekCompressor):
                module.transpose_wkv_wgate()
        return loaded
