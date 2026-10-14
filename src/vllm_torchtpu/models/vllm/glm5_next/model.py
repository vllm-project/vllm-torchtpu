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
"""Native GLM-5.3-Flash (``glm5_next``) TPU model implementation.

45 decoder layers: 34 KDA linear-attention and 11 NoPE-MLA full-attention,
every one of them wrapped in Manifold-Harmonized Coupling (mHC) with
``hc_mult=4`` residual streams. Layers 0-2 are dense MLPs, 3-44 are MoE.

Two pieces of the released architecture are deliberately absent here:

* the k-pool compressed DSA indexer -- the 11 full-attention layers run as
  dense MLA, which is numerically fine but quadratic at long context;
* the MTP layer (checkpoint index 45) and the vision tower, whose weights are
  skipped at load time.
"""

from __future__ import annotations

import sys
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.model_executor.offloader import NoopOffloader, set_offloader
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op import (  # noqa: E501
    MHCOps,
    get_mhc_ops,
    get_mhc_post_op,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.deepseek_v4.model import _HostStagedParams
from vllm_torchtpu.models.vllm.glm5_next.attention import (
    Glm5NextLinearAttention,
    Glm5NextMLA,
    glm5_next_kda_state_dtype,
)
from vllm_torchtpu.models.vllm.glm5_next.indexer import dense_dsa_fallback
from vllm_torchtpu.models.vllm.glm5_next.layers import Glm5NextMLP
from vllm_torchtpu.models.vllm.glm5_next.moe import Glm5NextMoE
from vllm_torchtpu.models.vllm.glm5_next.sp import TokenShards, sp_min_tokens, use_sp

logger = init_logger(__name__)

_SP_TWINS = (
    (".self_attn.q_b_proj.", ".self_attn.q_b_proj_sp."),
    (".self_attn.o_proj.", ".self_attn.o_proj_sp."),
)

# GLM's HyperConnection multiplies the post gate by 2 (``2 * sigmoid(...)``);
# the shared mHC kernel takes that factor as a constant.
HC_POST_MULT = 2.0


class Glm5NextDecoderLayer(nn.Module):
    """One GLM-5.3 decoder layer: mHC seam, attention, mHC seam, feed-forward."""

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        if config.layer_types[self.layer_idx] == "linear_attention":
            self.self_attn = Glm5NextLinearAttention(
                config, vllm_config, prefix=f"{prefix}.self_attn"
            )
        else:
            self.self_attn = Glm5NextMLA(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )

        if self.layer_idx < config.first_k_dense_replace:
            self.mlp = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=config.swiglu_limit,
                quant_config=vllm_config.quant_config,
                reduce_results=False,  # reduced in `forward`, like the MoE
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMoE(vllm_config, prefix=f"{prefix}.mlp")

        self.input_layernorm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, self.rms_norm_eps)

        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        # Checkpoint stores `fn` in bf16 and the rest in fp32; the gate math is
        # fp32 throughout, so declare all of them fp32 and let the loader cast.
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )

    @property
    def mhc_ops(self) -> MHCOps:
        """This layer's mHC Pallas ops, built once outside the trace.

        Identical math to ``Glm5NextTextHyperConnection``: sigmoid pre gates
        with ``+hc_eps``, ``2 * sigmoid`` post gates, softmax + Sinkhorn-Knopp
        on the H x H combine matrix. See ``get_mhc_ops`` for why this cannot be
        built inside a compiled forward.
        """
        ops = self.__dict__.get("_mhc_ops_instance")
        if ops is None:
            ops = get_mhc_ops(
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                HC_POST_MULT,
                self.hc_sinkhorn_iters,
            )
            object.__setattr__(self, "_mhc_ops_instance", ops)
        return ops

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collapse the residual streams into this sublayer's input."""
        post_mix, res_mix, layer_input = self.mhc_ops.pre(x, hc_fn, hc_scale, hc_base)
        return layer_input, post_mix, res_mix

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        sp: TokenShards | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defer each post into the next pre so the pair is one kernel.

        `x` arrives as the previous sublayer's raw output and the caller
        carries `(residual, post_mix, res_mix)` -- the recombine that output
        still owes. The chain is opened by the layer-0 attention pre, which has
        no preceding post, and closed by the caller.

        With `sp`, everything here is this rank's token shard; the sublayers
        gather their input and scatter their output (see `sp.py`).
        """
        if residual is None:
            residual = x
            x, post_mix, res_mix = self.hc_pre(
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

        x = self.input_layernorm(x)
        x = self.self_attn(positions, x, sequence_parallel=sp is not None)

        residual, post_mix, res_mix, x = self.mhc_ops.fused(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        x = self.post_attention_layernorm(x)
        if sp is not None:
            x = sp.reduce_scatter(self.mlp(sp.all_gather(x)))
        else:
            x = tensor_model_parallel_all_reduce(self.mlp(x))
        return x, residual, post_mix, res_mix


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    }
)
class Glm5NextModel(nn.Module):
    """GLM-5.3-Flash backbone: embedding, 45 mHC decoder layers, final norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        set_offloader(NoopOffloader())
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.hc_mult = config.hc_mult

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        # One candidate-list buffer shared by all 11 DSA layers: only one is
        # live at a time, and the sparse MLA kernel reads it straight after the
        # indexer writes it. Width is `index_topk` plus the up-to-3 tail tokens
        # `index_kpool_always_select_tail` appends, so 2051 rather than 2048 --
        # which is why these layers miss `matches_glm52_tpu7x_profile`'s
        # hardcoded 2048-wide masked-dense fast path and take the general
        # sparse kernel.
        self.topk_indices_buffer = None
        if not dense_dsa_fallback() and getattr(config, "index_topk", 0):
            width = config.index_topk + (
                config.index_kpool - 1
                if getattr(config, "index_kpool_always_select_tail", False)
                else 0
            )
            # Pad to a multiple of 256. The sparse MLA kernel needs a multiple
            # of 4 at minimum, but its SparseCore gather runs in batches of
            # 16384 indices: at 2052 a 64-token prefill chunk (131328 indices)
            # is padded to 147456 = 64 * 2304 and then sliced back, which is a
            # full copy of the gathered KV (~110 ms of an 8K prefill). At 2304
            # nothing is padded or sliced, and the SparseCore cache layout's
            # divisibility rule is met as well. Rows are left-packed, so the
            # extra columns are -1 and masked out.
            width = -(-width // 256) * 256
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                width,
                dtype=torch.int32,
                device=vllm_config.device_config.device,
            )
        self.layers = nn.ModuleList(
            [
                Glm5NextDecoderLayer(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    topk_indices_buffer=self.topk_indices_buffer,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.sp = TokenShards() if sp_min_tokens() else None
        if self.sp is not None:
            for layer in self.layers:
                # KDA (Kimi's layer) all-gathers / reduce-scatters through
                # `prefill_group` when called with `sequence_parallel=True`.
                layer.self_attn.prefill_group = self.sp

    @property
    def mhc_post_op(self):
        """Settles the post that the last decoder layer deferred.

        Carries no gate constants, so this shares the layers' op cache entry
        rather than registering a new one.
        """
        op = self.__dict__.get("_mhc_post_op_instance")
        if op is None:
            op = get_mhc_post_op()
            object.__setattr__(self, "_mhc_post_op_instance", op)
        return op

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            assert input_ids is not None
            hidden_states = self.embed_input_ids(input_ids)
        # A large step keeps the residual stream token-sharded (`sp.py`).
        # Branching on the token count is fine here: the backbone is traced
        # per bucket group.
        sp = (
            self.sp
            if (self.sp is not None and use_sp(hidden_states.shape[0]))
            else None
        )
        if sp is not None:
            hidden_states = sp.local_tokens(hidden_states)
        # One embedding broadcast into hc_mult parallel residual streams.
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual, post_mix, res_mix = None, None, None
        for layer in islice(self.layers, 0, len(self.layers)):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                post_mix,
                res_mix,
                residual,
                sp,
            )

        if post_mix is not None:
            hidden_states = self.mhc_post_op(hidden_states, residual, post_mix, res_mix)

        # Glm5NextTextHyperHead: an unweighted mean over the streams. DeepSeek
        # -V4 learns gates for this collapse; GLM does not, and the checkpoint
        # correspondingly ships no `hc_head_*` tensors.
        hidden_states = self.norm(hidden_states.mean(dim=-2))
        if sp is not None:
            hidden_states = sp.all_gather(hidden_states)
        return hidden_states


class Glm5NextForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """GLM-5.3-Flash text model for TPU."""

    # `model.language_model.` -> `model.`; the vision tower keeps its
    # `model.visual.` prefix and falls out as an unknown parameter below.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )

    # The checkpoint's `modules_to_not_convert` is written per unfused tensor,
    # one entry per layer. It excludes every KDA projection (those layers are
    # bf16) but knows nothing about the fused modules this implementation
    # builds. `is_layer_skipped` expands a fused name through this mapping and
    # checks the shards, so the exclusions land on the right layers without
    # editing the checkpoint's list: the 34 KDA layers' `fused_qkvb_proj` and
    # `fused_fa_ga_proj` stay bf16, while MLA's `fused_qkv_a_proj` and every
    # `gate_up_proj` remain block-FP8.
    packed_modules_mapping = {
        "fused_qkvb_proj": ["q_proj", "k_proj", "v_proj", "b_proj"],
        "fused_fa_ga_proj": ["f_a_proj", "g_a_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # (param substring, checkpoint substring, shard id). Order matters only in
    # that the attention entries must not be shadowed by the MLP ones; the
    # dotted forms keep `q_proj` from matching `q_a_proj` and so on.
    stacked_params_mapping = (
        # KDA: q/k/v/b projections share one column-parallel matmul.
        (".self_attn.fused_qkvb_proj.", ".self_attn.q_proj.", 0),
        (".self_attn.fused_qkvb_proj.", ".self_attn.k_proj.", 1),
        (".self_attn.fused_qkvb_proj.", ".self_attn.v_proj.", 2),
        (".self_attn.fused_qkvb_proj.", ".self_attn.b_proj.", 3),
        # KDA: the two rank-`head_dim` gate down-projections, replicated.
        (".self_attn.fused_fa_ga_proj.", ".self_attn.f_a_proj.", 0),
        (".self_attn.fused_fa_ga_proj.", ".self_attn.g_a_proj.", 1),
        # MLA: the q and kv down-projections.
        (".self_attn.fused_qkv_a_proj.", ".self_attn.q_a_proj.", 0),
        (".self_attn.fused_qkv_a_proj.", ".self_attn.kv_a_proj_with_mqa.", 1),
        # Dense and shared-expert MLPs. Routed experts use the same tensor
        # names, so this rule is skipped for anything under `.experts.`.
        (".gate_up_proj.", ".gate_proj.", 0),
        (".gate_up_proj.", ".up_proj.", 1),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # 45 layers x 4 residual streams overruns CPython's default 1000-frame
        # limit during AOT capture, same as DeepSeek-V4's 61.
        sys.setrecursionlimit(50000)
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = Glm5NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("The TPU GLM-5.3 model does not support PP")
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return glm5_next_kda_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        config = vllm_config.model_config.hf_text_config
        kda_config = config.linear_attn_config
        local_heads = (
            kda_config["num_heads"] // vllm_config.parallel_config.tensor_parallel_size
        )
        return (
            (
                kda_config["short_conv_kernel_size"] - 1,
                3,
                local_heads,
                kda_config["head_dim"],
            ),
            (local_heads, kda_config["head_dim"], kda_config["head_dim"]),
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint into stacked, expert and plain parameters.

        Anything with no home in ``params_dict`` is dropped on the floor: that
        is the vision tower (``model.visual.``) and the MTP layer (checkpoint
        index 45, one past ``num_hidden_layers``), neither of which this model
        builds.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_mapping = self.get_expert_mapping()
        # Slice writes into an already-materialized device tensor allocate a
        # fresh buffer per call on this backend; stage them on the host and
        # flush each parameter with one whole-tensor copy. See
        # `_HostStagedParams`.
        staged = _HostStagedParams()

        try:
            for name, loaded_weight in self.hf_to_vllm_mapper.apply(weights):
                if name.startswith("model.visual."):
                    continue

                for param_name, weight_name, shard_id in self.stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    if ".experts." in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    if mapped not in params_dict:
                        continue
                    param = staged.stage(mapped, params_dict[mapped])
                    param.weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(mapped)
                    break
                else:
                    if ".experts." in name:
                        for (
                            param_name,
                            weight_name,
                            expert_id,
                            expert_shard_id,
                        ) in expert_mapping:
                            if weight_name not in name:
                                continue
                            mapped = name.replace(weight_name, param_name)
                            if mapped not in params_dict:
                                continue
                            param = staged.stage(mapped, params_dict[mapped])
                            weight_loader = typing.cast(
                                Callable[..., bool], param.weight_loader
                            )
                            if weight_loader(
                                param,
                                loaded_weight,
                                mapped,
                                shard_id=expert_shard_id,
                                expert_id=expert_id,
                                return_success=True,
                            ):
                                loaded_params.add(mapped)
                                break
                        continue
                    if name not in params_dict:
                        continue
                    param = staged.stage(name, params_dict[name])
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    # The sequence-parallel twins (`Glm5NextMLA`) take the
                    # same checkpoint tensors, whole.
                    for tp_name, sp_name in _SP_TWINS:
                        twin = name.replace(tp_name, sp_name)
                        if tp_name in name and twin in params_dict:
                            param = staged.stage(twin, params_dict[twin])
                            getattr(param, "weight_loader", default_weight_loader)(
                                param, loaded_weight
                            )
                            loaded_params.add(twin)
        finally:
            staged.flush_all()

        return loaded_params


class Glm5NextForConditionalGeneration(Glm5NextForCausalLM):
    """The released architecture name, text-only for now.

    GLM-5.3-Flash is a VLM, but the vision tower is not ported yet, so this is
    the text model under the checkpoint's own ``architectures`` entry. Image
    and video inputs are not accepted; ``model.visual.*`` is skipped at load.
    """
