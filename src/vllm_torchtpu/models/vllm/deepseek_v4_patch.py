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
"""Context manager to patch DeepSeek V4 model registry and classes during loading."""

# TODO(patemotter): replace this patching with a dedicated DSv4 model.py
# registered in `register_models()`, the way `kimi_k3` is.

import sys
from contextlib import contextmanager

import torch
from vllm.config import VllmConfig

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.deepseek_v4_requant import \
    attention_requantizer_generator

logger = init_logger(__name__)

_PATCHED = False


@contextmanager
def _maybe_patch_for_deepseek_v4(vllm_config: VllmConfig):
    architectures = getattr(vllm_config.model_config.hf_config,
                            "architectures", None) or []
    logger.info(f"Patcher checking architectures: {architectures}")

    is_ds_v4 = any("DeepseekV4ForCausalLM" in arch for arch in architectures)
    if not is_ds_v4:
        yield
        return

    # Re-entering would wrap `load_weights` a second time and run the
    # requantizer twice, feeding already-dequantized scales back through it.
    global _PATCHED
    if _PATCHED:
        yield
        return

    logger.info("Applying DeepSeek V4 TPU model patches...")

    # Bypasses a Dynamo json serialization bug during metrics logging.
    try:
        from torch._dynamo import utils as dynamo_utils
        dynamo_utils._get_dynamo_config_for_logging = lambda: {}
    except ImportError:
        pass

    # Patch vllm.distributed.parallel_state to reflect the TPU sharding strategy (TP vs EP)
    import vllm.model_executor.layers.fused_moe.config as fmc

    parallel_config = vllm_config.parallel_config
    ep_size = (parallel_config.world_size if getattr(
        parallel_config, "enable_expert_parallel", False) else 1)

    _orig_make = fmc.FusedMoEParallelConfig.make

    def patched_make(*args, **kwargs):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else -1
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if ep_size > 1:
            moe_tp_size = max(1, world_size // ep_size)
            ret_rank = rank // moe_tp_size if rank >= 0 else 0

            return fmc.FusedMoEParallelConfig(
                tp_size=1,
                tp_rank=0,
                pcp_size=1,
                pcp_rank=0,
                dp_size=1,
                dp_rank=0,
                ep_size=ep_size,
                ep_rank=ret_rank,
                sp_size=1,
                use_ep=True,
                all2all_backend="allgather_reducescatter",
                enable_eplb=False,
            )
        return _orig_make(*args, **kwargs)

    fmc.FusedMoEParallelConfig.make = patched_make

    # 1. Patch the model registry to resolve to the AMD (ROCm) variant.
    # The AMD variant is more extensible and allows us to swap attention/MHC classes.
    import vllm.models.deepseek_v4 as ds_v4

    def patch_load_weights_for_class(cls):
        _orig_load_weights = cls.load_weights

        def patched_load_weights(self, weights):

            self.model.has_tilelang = False
            for idx, l in enumerate(self.model.layers):  # noqa: E741
                l.layer_idx = idx
                l.has_tilelang = False
                if hasattr(l, "ffn"):
                    l.ffn.layer_idx = idx
                    if hasattr(l.ffn, "experts"):
                        l.ffn.experts.layer_idx = idx
                        # Quant methods read layer_idx off the RoutedExperts
                        # submodule, not just the owning MoERunner.
                        routed_experts_for_idx = getattr(
                            l.ffn.experts, "routed_experts", None)
                        if routed_experts_for_idx is not None:
                            routed_experts_for_idx.layer_idx = idx
                        # Hash-MoE layers route by gate.tid2eid[token_id], and
                        # TPU routing sees RoutedExperts as `layer`, so the
                        # table has to be reachable from there too.
                        tid2eid = getattr(getattr(l.ffn, "gate", None),
                                          "tid2eid", None)
                        if tid2eid is not None:
                            runner = l.ffn.experts
                            for tgt in (runner,
                                        getattr(runner, "routed_experts",
                                                None)):
                                if tgt is not None:
                                    object.__setattr__(tgt,
                                                       "hash_indices_table",
                                                       tid2eid)

            from vllm.models.deepseek_v4.amd.model import \
                _make_deepseek_v4_weights_mapper
            quant_config = getattr(vllm_config, "quant_config", None)
            expert_dtype = getattr(quant_config, "expert_dtype", "fp4")
            is_fp4_experts = (expert_dtype == "fp4")
            orig_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)

            outer_self = self
            prev_mapper = getattr(self, "hf_to_vllm_mapper", None)

            class WrappedMapper:

                def apply(self, w):
                    for name, loaded_weight in orig_mapper.apply(w):
                        if name == "model.embed.weight":
                            name = "model.embed_tokens.weight"
                        if not is_fp4_experts and ".experts." in name and name.endswith(
                                "weight_scale"):
                            name = name + "_inv"
                        if "hc_head.hc_" in name:
                            name = name.replace("hc_head.hc_", "hc_head_")

                        # Map shared experts from HF (gate_proj/up_proj/down_proj) to vLLM (w1/w3/w2)
                        # so they are matched by stacked_params_mapping and loaded by the normal loader
                        if ".shared_experts.gate_proj" in name:
                            name = name.replace(".shared_experts.gate_proj",
                                                ".shared_experts.w1")
                        elif ".shared_experts.up_proj" in name:
                            name = name.replace(".shared_experts.up_proj",
                                                ".shared_experts.w3")
                        elif ".shared_experts.down_proj" in name:
                            name = name.replace(".shared_experts.down_proj",
                                                ".shared_experts.w2")

                        if "layers." in name:
                            try:
                                layer_idx = int(
                                    name.split("layers.")[1].split(".")[0])
                                if layer_idx >= outer_self.config.num_hidden_layers:
                                    continue
                            except (ValueError, IndexError):
                                pass
                        # Map MHC layer parameters from HF (nested) to vLLM (flat)
                        if ".attn_hc.base" in name:
                            name = name.replace(".attn_hc.base",
                                                ".hc_attn_base")
                        elif ".attn_hc.fn" in name:
                            name = name.replace(".attn_hc.fn", ".hc_attn_fn")
                        elif ".attn_hc.weight_scale_inv" in name:
                            name = name.replace(".attn_hc.weight_scale_inv",
                                                ".hc_attn_scale")
                        elif ".ffn_hc.base" in name:
                            name = name.replace(".ffn_hc.base", ".hc_ffn_base")
                        elif ".ffn_hc.fn" in name:
                            name = name.replace(".ffn_hc.fn", ".hc_ffn_fn")
                        elif ".ffn_hc.weight_scale_inv" in name:
                            name = name.replace(".ffn_hc.weight_scale_inv",
                                                ".hc_ffn_scale")

                        # Map layernorms
                        if ".input_layernorm" in name:
                            name = name.replace(".input_layernorm",
                                                ".attn_norm")
                        elif ".post_attention_layernorm" in name:
                            name = name.replace(".post_attention_layernorm",
                                                ".ffn_norm")

                        # Map mlp to ffn
                        if ".mlp." in name:
                            name = name.replace(".mlp.", ".ffn.")

                        # Map attention parameters
                        # 1. Indexer Compressor mappings (HF -> vLLM)
                        if ".self_attn.compressor.indexer.gate_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.gate_proj",
                                ".attn.indexer.compressor.wgate")
                        elif ".self_attn.compressor.indexer.kv_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.kv_proj",
                                ".attn.indexer.compressor.wkv")
                        elif ".self_attn.compressor.indexer.kv_norm" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.kv_norm",
                                ".attn.indexer.compressor.norm")
                        elif ".self_attn.compressor.indexer.position_bias" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.position_bias",
                                ".attn.indexer.compressor.ape")

                        # 2. Indexer main mappings (HF -> vLLM)
                        elif ".self_attn.compressor.indexer.q_b_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.q_b_proj",
                                ".attn.indexer.wq_b")
                        elif ".self_attn.compressor.indexer.scorer.weights_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.scorer.weights_proj",
                                ".attn.indexer.weights_proj")

                        # 3. Attention Compressor mappings (HF -> vLLM)
                        elif ".self_attn.compressor.gate_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.gate_proj",
                                ".attn.compressor.wgate")
                        elif ".self_attn.compressor.kv_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.kv_proj",
                                ".attn.compressor.wkv")
                        elif ".self_attn.compressor.kv_norm" in name:
                            name = name.replace(
                                ".self_attn.compressor.kv_norm",
                                ".attn.compressor.norm")
                        elif ".self_attn.compressor.position_bias" in name:
                            name = name.replace(
                                ".self_attn.compressor.position_bias",
                                ".attn.compressor.ape")
                        elif ".self_attn.q_a_proj" in name:
                            name = name.replace(".self_attn.q_a_proj",
                                                ".attn.wq_a")
                        elif ".self_attn.q_a_norm" in name:
                            name = name.replace(".self_attn.q_a_norm",
                                                ".attn.q_norm")
                        elif ".self_attn.q_b_proj" in name:
                            name = name.replace(".self_attn.q_b_proj",
                                                ".attn.wq_b")
                        elif ".self_attn.kv_proj" in name:
                            name = name.replace(".self_attn.kv_proj",
                                                ".attn.wkv")
                        elif ".self_attn.kv_norm" in name:
                            name = name.replace(".self_attn.kv_norm",
                                                ".attn.kv_norm")
                        elif ".self_attn.o_a_proj" in name:
                            name = name.replace(".self_attn.o_a_proj",
                                                ".attn.wo_a")
                        elif ".self_attn.o_b_proj" in name:
                            name = name.replace(".self_attn.o_b_proj",
                                                ".attn.wo_b")
                        elif ".self_attn.sinks" in name:
                            name = name.replace(".self_attn.sinks",
                                                ".attn.attn_sink")

                        if ".experts." not in name and name.endswith(
                                "weight_scale_inv"):
                            name = name.replace("weight_scale_inv",
                                                "weight_scale")

                        yield name, loaded_weight

            try:
                # Update TP status of all parameters since new ones (like weight_scale)
                # are registered during weight creation which runs AFTER linear layer __init__.
                for name, module in self.model.named_modules():
                    if hasattr(module, "update_param_tp_status"):
                        module.update_param_tp_status()

                mapper = WrappedMapper()

                def mapped_weights_generator(weights_iter):
                    for name, loaded_weight in weights_iter:
                        for mapped_name, mapped_weight in mapper.apply([
                            (name, loaded_weight)
                        ]):
                            yield mapped_name, mapped_weight

                weights = attention_requantizer_generator(
                    mapped_weights_generator(weights))

                class IdentityMapper:

                    def apply(self, w):

                        def mapper_generator(w_iter):
                            for name, loaded_weight in w_iter:
                                if ".shared_experts.w2" in name:
                                    name = name.replace(
                                        ".shared_experts.w2",
                                        ".shared_experts.down_proj")
                                yield name, loaded_weight

                        return mapper_generator(w)

                self.hf_to_vllm_mapper = IdentityMapper()

                res = _orig_load_weights(self, weights)

                # No manual process_weights_after_loading for MoE layers:
                # vLLM's generic walker finds them, and a second call would
                # re-process already-packed weights.

                import torch.distributed as dist
                if dist.is_initialized():
                    logger.info(
                        "Aligning workers at barrier after _orig_load_weights")
                    dist.barrier()

                from torch_tpu._internal import sync
                sync.synchronize(wait=True)
                import gc
                gc.collect()

                return res
            finally:
                self.hf_to_vllm_mapper = prev_mapper

        cls.load_weights = patched_load_weights

    # Patch the standard CUDA/CPU variant, then the AMD one it resolves to.
    _CudaDeepseekV4ForCausalLM = ds_v4.DeepseekV4ForCausalLM
    patch_load_weights_for_class(_CudaDeepseekV4ForCausalLM)

    from vllm.models.deepseek_v4.amd.model import \
        DeepseekV4ForCausalLM as _AmdDeepseekV4ForCausalLM
    ds_v4.DeepseekV4ForCausalLM = _AmdDeepseekV4ForCausalLM
    patch_load_weights_for_class(_AmdDeepseekV4ForCausalLM)

    def patched_compute_logits(self, hidden_states: torch.Tensor):
        return self.logits_processor(self.lm_head, hidden_states)

    _AmdDeepseekV4ForCausalLM.compute_logits = patched_compute_logits

    # 1.5. Patch DeepseekV4DecoderLayer and DeepseekV4Model __init__ for clean support_torch_compile wrapping
    import torch.nn as nn
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.model_executor.layers.vocab_parallel_embedding import \
        VocabParallelEmbedding
    from vllm.model_executor.models.utils import PPMissingLayer, make_layers
    from vllm.models.deepseek_v4.amd.model import (DeepseekV4DecoderLayer,
                                                   DeepseekV4Model)
    from vllm.platforms import current_platform

    _orig_decoder_layer_init = DeepseekV4DecoderLayer.__init__

    def patched_decoder_layer_init(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
        **kwargs,
    ):
        _orig_decoder_layer_init(
            self,
            vllm_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        # Distinct per layer, so each gets its own compile cache directory.
        self.prefix = prefix

    DeepseekV4DecoderLayer.__init__ = patched_decoder_layer_init

    def patched_model_init(self,
                           *,
                           vllm_config: VllmConfig,
                           prefix: str = "",
                           **kwargs):
        import torch.nn as nn
        from vllm.model_executor.layers.layernorm import RMSNorm
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        aux_stream_list = None
        self.device = current_platform.device_type
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        from vllm.model_executor.layers.mhc import HCHeadOp
        self.hc_head_op = HCHeadOp()
        self.vllm_config = vllm_config
        self.head = DeepseekV4Head(
            self.hc_head_op,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.norm,
            self.rms_norm_eps,
            self.hc_eps,
        )

    class DeepseekV4Head(nn.Module):

        def __init__(
            self,
            hc_head_op,
            hc_head_fn,
            hc_head_scale,
            hc_head_base,
            norm,
            rms_norm_eps: float,
            hc_eps: float,
        ):
            super().__init__()
            # One head per model, so a literal is a stable compile cache key.
            self.prefix = "model.head"
            self.hc_head_op = hc_head_op
            self.hc_head_fn = hc_head_fn
            self.hc_head_scale = hc_head_scale
            self.hc_head_base = hc_head_base
            self.norm = norm
            self.rms_norm_eps = rms_norm_eps
            self.hc_eps = hc_eps

        def forward(
            self,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None = None,
            post_mix: torch.Tensor | None = None,
            res_mix: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if residual is not None and post_mix is not None and res_mix is not None:
                mixed_residual = torch.einsum(
                    "...ij,...ih->...jh",
                    res_mix.to(torch.float32),
                    residual.to(torch.float32),
                )
                post_term = post_mix.to(
                    torch.float32) * hidden_states.unsqueeze(-2).to(
                        torch.float32)
                hidden_states = (mixed_residual + post_term).to(residual.dtype)
            hidden_states = self.hc_head_op(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_scale,
                self.hc_head_base,
                self.rms_norm_eps,
                self.hc_eps,
            )
            return self.norm(hidden_states)

    def patched_model_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):
        from itertools import islice

        from vllm.distributed.parallel_state import get_pp_group
        from vllm.sequence import IntermediateTensors

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(
                1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        residual, post_mix, res_mix = None, None, None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # The last layer's hc_post is folded into `self.head` so it lands in
        # the head's compiled graph rather than the per-layer one.
        return self.head(
            hidden_states,
            residual,
            post_mix,
            res_mix,
        )

    DeepseekV4Model.__init__ = patched_model_init
    DeepseekV4Model.forward = patched_model_forward

    # Tracing this model exceeds the default 1000 frames.
    sys.setrecursionlimit(50000)

    # Layer-level and head graph mode.
    from vllm.compilation.decorators import support_torch_compile
    if not hasattr(DeepseekV4DecoderLayer, "__vllm_torch_compile__"):
        support_torch_compile(
            DeepseekV4DecoderLayer,
            dynamic_arg_dims={
                "x": 0,
                "positions": 0,
                "residual": 0,
                "post_mix": 0,
                "res_mix": 0,
                "input_ids": 0,
            },
        )
    if not hasattr(DeepseekV4Head, "__vllm_torch_compile__"):
        support_torch_compile(
            DeepseekV4Head,
            dynamic_arg_dims={
                "hidden_states": 0,
                "residual": 0,
                "post_mix": 0,
                "res_mix": 0,
            },
        )

    # 2. Clear cached resolved architectures so the AMD model is picked up.
    from vllm.model_executor.model_loader import utils as _ml_utils
    from vllm.model_executor.models.registry import _try_load_model_cls
    _ml_utils._MODEL_ARCH_BY_HASH.clear()
    _try_load_model_cls.cache_clear()

    # 3. Swap the DeepseekV4ROCMAiterMLAAttention class with our TPU implementation.
    from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_attention import \
        patch_deepseek_v4_mla_cls
    patch_deepseek_v4_mla_cls()

    # 4. Mock CUDA device capability, Stream, and Event allocation since they are unavailable on TPU.
    from vllm_torchtpu.models.vllm.deepseek_v4_stubs import patch_cuda_stubs

    # The patches stay active for the process lifetime (see the `finally`
    # below), so the returned originals are deliberately not kept.
    patch_cuda_stubs()

    # 5. Patch HCHeadOp.forward_native with a TPU-safe fp32 implementation.
    from vllm.model_executor.layers.mhc import HCHeadOp

    def mhc_hc_head_native(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        # hc_fn/hc_scale/hc_base are float32, so this projection must run in
        # fp32; casting activations to bf16 first is a dtype mismatch.
        residual_flat = hidden_states.flatten(-2).float()
        residual_norm = residual_flat * torch.rsqrt(
            residual_flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps)
        pre_mix = torch.nn.functional.linear(residual_norm, hc_fn)
        pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
        return torch.sum(pre_mix.unsqueeze(-1) * hidden_states.float(),
                         dim=-2).to(hidden_states.dtype)

    import safetensors.torch  # pyrefly: ignore
    safetensors.torch._TYPES["F8_E8M0"] = torch.uint8

    HCHeadOp.forward_native = mhc_hc_head_native

    # Latched only after every patch step succeeded: a mid-patch exception
    # must not leave later loads silently accepting a half-patched process.
    _PATCHED = True

    try:
        yield
    finally:
        logger.info(
            "DeepSeek V4 TPU model patches kept active for compilation and worker execution."
        )
