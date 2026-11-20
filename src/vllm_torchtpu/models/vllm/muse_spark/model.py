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
"""TPU text-model subclasses for Meta Muse Spark Open on upstream vLLM v0.33.0.

Reuses upstream ``vllm.models.muse_spark`` (``amd/model.py`` and
``common/{layers,moe,omni}.py``) and overrides only the TPU-specific pieces:
1. Pure-PyTorch ``o_norm`` + gated sigmoid output path and FP32 NEOX RoPE in
   :class:`TPUMuseSparkAttention` / :class:`TPUMuseSparkDecoderLayer` so cached
   keys keep HF ``[even | odd]`` channel order for downstream TPU kernels.
2. :class:`TPUMuseSparkMoE` wired to :func:`build_post_norm_moe` after
   per-layer quantization finalization, with TP-sharded ``pooler``/``unpooler``.
3. Non-interleaving ``load_weights`` and model-level
   ``process_weights_after_loading`` in :class:`TPUMuseSparkForCausalLM`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import copy
from typing import Any, cast

import torch
from torch import nn

from vllm_torchtpu.models.vllm.muse_spark import compat

compat.apply()

from vllm.config import VllmConfig  # noqa: E402
from vllm.distributed import (  # noqa: E402
    get_ep_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger  # noqa: E402
from vllm.model_executor.layers.fused_moe import (  # noqa: E402
    FusedMoEFactory,
    GateLinear,
)
from vllm.model_executor.layers.linear import (  # noqa: E402
    ColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.rotary_embedding import (  # noqa: E402
    RotaryEmbedding,
    get_rope,
)
from vllm.model_executor.models.llama import LlamaModel  # noqa: E402
from vllm.model_executor.models.utils import (  # noqa: E402
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
)
from vllm.models.muse_spark.amd import model as up  # noqa: E402
from vllm.models.muse_spark.common.layers import (  # noqa: E402
    MuseSpark1xQKVOGateParallelLinear,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
    materialize_residual_path_constants,
)
from vllm.models.muse_spark.common.moe import (  # noqa: E402
    MuseSpark1xRouter,
    MuseSparkConsolidatedRoutedExperts,
    MuseSparkMoEConfig,
)
from vllm.models.muse_spark.common.omni import OMNI_WEIGHTS_MAPPER  # noqa: E402
from vllm.sequence import IntermediateTensors  # noqa: E402
from vllm.transformers_utils.configs.muse_spark import (  # noqa: E402
    MuseSparkDSparkConfig,
)

from vllm_torchtpu.layers.adapter.custom_ops.post_norm_moe_op import (  # noqa: E402
    PostNormMoeStatic,
    build_post_norm_moe,
)
from vllm_torchtpu.layers.adapter.linear_common import (  # noqa: E402
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config  # noqa: E402
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import (  # noqa: E402
    VllmNvfp4Config,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (  # noqa: E402
    VllmUnquantizedConfig,
    VllmUnquantizedLinearMethod,
)

logger = init_logger(__name__)

_TEXT_ONLY_SKIP_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
    "model.audio_tower.",
    "model.audio_projection.",
    "model.perception",
)


class TPUMuseSparkAttention(up.MuseSpark1xAttention):
    """MuseSpark1x attention using TPU FP32 NEOX RoPE and native o_norm/gate."""

    qkv_proj: Any
    o_proj: Any
    pre_qkv_proj_norm: Any
    qk_norm: Any
    rotary_emb: Any
    attn: Any
    q_size: int
    kv_size: int
    head_dim: int
    num_kv_heads: int
    is_sequence_parallel_output: bool
    causal: bool
    fuse_o_norm_gate: bool
    fused_qk_norm_rope_kv_insert: bool

    def _qk_norm_rope_kv_insert(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Adapted from upstream vllm.models.muse_spark.amd.model:598-651
        # (Apache-2.0): drops the ROCm fused KV insert branch and rotates
        # q.float()/k.float() through self.rotary_emb so TPU RoPE preserves
        # the FP32 cos/sin cache (`rotary_embedding.py` matches cache dtype to
        # query dtype).
        if self.use_o_gate:
            q, k, v, og = qkv.split(
                [self.q_size, self.kv_size, self.kv_size, self.q_size], dim=-1
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            og = None

        if self.qk_norm is not None:
            q = self.qk_norm(
                q.reshape(-1, self.head_dim), out_dtype=torch.float32
            ).reshape(-1, self.q_size)
            k = self.qk_norm(
                k.reshape(-1, self.head_dim), out_dtype=torch.float32
            ).reshape(-1, self.kv_size)
            q = q.to(v.dtype)
            k = k.to(v.dtype)

        if self.use_attn_temperature_tuning:
            q = (q * self._get_attn_temperature_scale(positions)).to(q.dtype)

        if self.rotary_emb is not None:
            q32, k32 = self.rotary_emb(positions, q.float(), k.float())
            assert k32 is not None
            q = q32.to(v.dtype)
            k = k32.to(v.dtype)
        return q, k, v, og


class TPUMuseSparkMoE(nn.Module):
    """TPU Muse Spark compressed MoE block backed by ``post_norm_moe_op``."""

    router_w_kmajor: torch.Tensor | None

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.layer_idx = layer_idx
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.moe_hidden_size = int(config.moe_hidden_size)
        self.intermediate_size = int(config.moe_intermediate_size)
        self.top_k = int(config.num_experts_per_tok)
        self.n_routed_experts = int(config.num_local_experts)
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts
        self.n_shared_experts = int(config.num_shared_experts)

        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_size = get_ep_group().device_group.size()
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.route_scale = float(config.routed_scaling_factor)
        self.router_output_multiplier = float(config.output_multiplier)
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.post_norm_eps = float(config.post_norm_eps)
        self.router_eps = 1e-15

        self.router = GateLinear(
            self.hidden_size,
            self.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.router",
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )
        self.register_buffer("router_w_kmajor", None, persistent=False)

        self.moe_router = MuseSpark1xRouter(
            top_k=self.top_k,
            num_experts=self.n_routed_experts,
            expert_bias=self.expert_bias,
            route_scale=self.route_scale,
            router_output_multiplier=self.router_output_multiplier,
            router_score_normalization="sigmoid",
            router_score_division_eps=self.router_eps,
        )

        # TP-sharded pooler/unpooler with gather_output=True and TPU canonical
        # (k, n) layout; avoids replicating 8192x4096 weights on every TP rank.
        params_dtype = cast(torch.dtype, vllm_config.model_config.dtype)
        self.pooler = ColumnParallelLinear(
            self.hidden_size,
            self.moe_hidden_size,
            bias=False,
            gather_output=True,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.pooler",
        )
        self.pooler.quant_method = VllmUnquantizedLinearMethod()

        self.unpooler = ColumnParallelLinear(
            self.moe_hidden_size,
            self.hidden_size,
            bias=False,
            gather_output=True,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.unpooler",
        )
        self.unpooler.quant_method = VllmUnquantizedLinearMethod()

        self.pre_expert_norm = MuseSpark1xRMSNorm(
            hidden_size=self.moe_hidden_size,
            eps=self.rms_norm_eps,
            has_weight=True,
            is_post_norm=False,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
            output_norm_gain_center_type="zero",
        )
        self.post_expert_norm = MuseSpark1xRMSNormWithInputScale(
            hidden_size=self.moe_hidden_size,
            eps=self.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=None,
            output_norm_gain_center_type="zero",
        )

        self.experts = FusedMoEFactory(
            num_experts=self.n_routed_experts,
            top_k=self.top_k,
            hidden_size=self.moe_hidden_size,
            intermediate_size=self.intermediate_size,
            renormalize=True,
            router=self.moe_router,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            is_sequence_parallel=False,
            router_logits_dtype=torch.float32,
            scoring_func="sigmoid",
            e_score_correction_bias=self.expert_bias,
            routed_scaling_factor=self.route_scale,
            routed_experts_cls=MuseSparkConsolidatedRoutedExperts,
            moe_config_cls=MuseSparkMoEConfig,
        )

        mpc = self.experts.routed_experts.moe_config.moe_parallel_config
        if mpc.use_ep and int(mpc.ep_size) != self.tp_size:
            raise NotImplementedError(
                f"Muse Spark MoE requires ep_size == tp_size (got ep_size="
                f"{mpc.ep_size}, tp_size={self.tp_size}): the combine step "
                "all-reduces over the TP group only."
            )

        self._impl: (
            Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor] | None
        ) = None

    @property
    def pre_projected(self) -> bool:
        return self.tp_size > 1

    def bind_fused_op(self) -> None:
        """Finalize K-major router weights and bind ``PostNormMoe`` post-PWAL."""
        rw = cast(torch.Tensor, self.router.weight).detach().float()
        if not getattr(self.router, WEIGHT_FLIPPED_ATTR, False):
            rw = rw.t()
        self.router_w_kmajor = rw.contiguous()
        if self.router_w_kmajor.device.type == "tpu":
            from vllm_torchtpu.utils import synchronize_tensors

            synchronize_tensors([self.router_w_kmajor])

        pre_w: torch.Tensor | None = None
        if not self.pre_projected:
            pw = cast(torch.Tensor, self.pooler.weight).detach()
            if not getattr(self.pooler, WEIGHT_FLIPPED_ATTR, False):
                pw = pw.t()
            pre_w = pw.contiguous()

        routed = self.experts.routed_experts
        mpc = routed.moe_config.moe_parallel_config
        static = PostNormMoeStatic(
            topk=self.top_k,
            output_multiplier=self.router_output_multiplier,
            routed_scaling_factor=self.route_scale,
            rms_norm_eps=self.rms_norm_eps,
            post_norm_eps=self.post_norm_eps,
            router_eps=self.router_eps,
            activation="silu",
            use_ep=bool(mpc.use_ep),
            pre_projected=self.pre_projected,
        )
        assert self.pre_expert_norm.weight is not None
        assert self.post_expert_norm.weight is not None
        self._impl = build_post_norm_moe(
            routed,
            static=static,
            router_w_kmajor=self.router_w_kmajor,
            router_bias=self.expert_bias,
            pre_w=pre_w,
            pre_norm_w=self.pre_expert_norm.weight,
            post_norm_w=self.post_expert_norm.weight,
            hidden_size=self.hidden_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._impl is None:
            raise RuntimeError(
                "Muse MoE has no implementation bound (the model-level "
                "process_weights_after_loading hook did not run)."
            )
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, orig_shape[-1])
        z_pre = self.pooler(x)[0] if self.pre_projected else None
        partial = self._impl(x, z_pre)
        if self.ep_size > 1:
            partial = tensor_model_parallel_all_reduce(partial)
        out, _ = self.unpooler(partial.to(x.dtype))
        return out.view(orig_shape)


class TPUMuseSparkDecoderLayer(up.MuseSpark1xDecoderLayer):
    """MuseSpark1x decoder layer wired to :class:`TPUMuseSparkMoE` and NEOX RoPE."""

    moe_cls = TPUMuseSparkMoE  # type: ignore[assignment]
    # pyrefly: ignore[bad-override-mutable-attribute]
    self_attn: TPUMuseSparkAttention

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        attn = self.self_attn
        if isinstance(
            attn.qkv_proj, MuseSpark1xQKVOGateParallelLinear
        ) and not isinstance(attn.qkv_proj.quant_method, UnquantizedLinearMethod):
            raise NotImplementedError(
                "Muse Spark fused QKVO-gate projection (`self_attn.qkv_proj`) "
                "must remain unquantized on TPU: upstream "
                "MuseSpark1xQKVOGateParallelLinear.load_weights does not "
                "support 0-D quantization scales. Exclude `self_attn` via "
                "quantization_config."
            )
        attn.__class__ = TPUMuseSparkAttention
        attn.fuse_o_norm_gate = False
        attn.fused_qk_norm_rope_kv_insert = False
        attn.attn.kv_cache_update_bypass = False

        config = vllm_config.model_config.hf_text_config
        is_draft = isinstance(config, MuseSparkDSparkConfig)
        if attn.rotary_emb is not None:
            if is_draft:
                old_re = attn.rotary_emb
                attn.rotary_emb = RotaryEmbedding(
                    head_size=old_re.head_size,
                    rotary_dim=old_re.rotary_dim,
                    max_position_embeddings=old_re.max_position_embeddings,
                    base=old_re.base,
                    is_neox_style=True,
                    dtype=torch.float32,
                )
            else:
                rope_theta = config.layer_rope_theta[self.layer_idx]
                attn.rotary_emb = get_rope(
                    head_size=attn.head_dim,
                    max_position=vllm_config.model_config.max_model_len,
                    is_neox_style=True,
                    rope_parameters={
                        "rope_theta": rope_theta,
                        "rope_type": "default",
                    },
                    dtype=torch.float32,
                )


class TPUMuseSparkModel(up.MuseSpark1xModel):
    """Re-enable ``LlamaModel``'s ``@support_torch_compile`` wrapper on TPU.

    ``LlamaModel`` already carries ``@support_torch_compile`` with dynamic batch
    symbols; ``MuseSpark1xModel`` sets ``_ignore_compile_vllm = True`` via
    ``@ignore_torch_compile``. Resetting ``_ignore_compile_vllm = False``
    restores compilation without double-wrapping ``TorchCompileWithNoGuardsWrapper``.

    Also restores ``LlamaModel.hf_to_vllm_mapper`` (matching upstream
    ``nvidia/model.py``) so dense layers stack ``gate_proj``/``up_proj`` into
    ``LlamaMLP.gate_up_proj``; ``amd/model.py`` drops those entries only for
    Quark per-expert keys, whereas HF/ModelOpt checkpoints use consolidated
    ``experts.gate_up_proj``.
    """

    _ignore_compile_vllm = False
    hf_to_vllm_mapper = LlamaModel.hf_to_vllm_mapper

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[up.MuseSpark1xDecoderLayer] = TPUMuseSparkDecoderLayer,
    ) -> None:
        # Expose prefix before LlamaModel's TorchCompileWithNoGuardsWrapper
        # __init__ runs so TPU compile_prefix cache isolation is deterministic.
        object.__setattr__(self, "prefix", prefix)
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=layer_type,
        )

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        res = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        if isinstance(res, tuple):
            output, aux_states = res
            return output, [s.to(self.default_dtype) for s in aux_states]
        return res


class TPUMuseSparkForCausalLM(up.MuseSpark1xForCausalLM):
    """TPU Muse Spark text model reusing upstream ``MuseSpark1xForCausalLM``."""

    # pyrefly: ignore[bad-override-mutable-attribute]
    hf_to_vllm_mapper: WeightsMapper = cast(
        WeightsMapper, up.MuseSpark1xForCausalLM.hf_to_vllm_mapper
    )
    model_cls = TPUMuseSparkModel
    decoder_layer_cls = TPUMuseSparkDecoderLayer
    supported_quantization_configs = (
        VllmNvfp4Config,
        VllmFp8Config,
        VllmUnquantizedConfig,
    )
    supported_quantization_description = (
        "ModelOpt NVFP4, per-tensor FP8, or BF16 (tpu-unquantized)"
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj", "gate_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[up.MuseSpark1xDecoderLayer] | None = None,
    ) -> None:
        if vllm_config.quant_config is not None and not isinstance(
            vllm_config.quant_config, self.supported_quantization_configs
        ):
            raise NotImplementedError(
                "MuseSpark1x on TPU only supports "
                f"{self.supported_quantization_description} checkpoints; "
                f"got quantization {vllm_config.quant_config.get_name()!r}."
            )
        if (
            isinstance(vllm_config.quant_config, VllmFp8Config)
            and getattr(vllm_config.quant_config, "weight_block_size", None) is not None
        ):
            raise NotImplementedError(
                "MuseSpark1x on TPU does not support blockwise FP8 "
                f"(weight_block_size={vllm_config.quant_config.weight_block_size!r}): "
                "upstream MuseSparkConsolidatedRoutedExperts only loads "
                "per-tensor FP8 or ModelOpt NVFP4 scales."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def set_moe_parameters(self) -> None:
        self.moe_layers = []
        example_moe: TPUMuseSparkMoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.feed_forward, TPUMuseSparkMoE):
                example_moe = layer.feed_forward
                self.moe_layers.append(layer.feed_forward.experts)

        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
        else:
            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        del num_physical_experts, num_local_physical_experts
        raise NotImplementedError("EPLB is not supported for Muse Spark on TPU.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights in HF/NEOX channel order without upstream GPT-J interleave."""
        loaded = AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
        n_gates = materialize_residual_path_constants(self)
        logger.info(
            "Materialized residual-path constants for %d gated residual connections",
            n_gates,
        )
        return loaded

    def process_weights_after_loading(self) -> None:
        """Model-level post-load hook executed after per-layer quant finalization."""
        for module in self.modules():
            if isinstance(module, TPUMuseSparkMoE):
                module.bind_fused_op()
        compat.sync_residual_constants(self)


class MuseSparkForConditionalGeneration(TPUMuseSparkForCausalLM):
    """Text-only TPU entry point for ``MuseSparkForConditionalGeneration``.

    Unwraps ``hf_config.text_config`` and skips vision/audio tower tensors so
    the released multimodal checkpoint serves text-only cleanly before multimodal
    support is layered on top.
    """

    hf_to_vllm_mapper = TPUMuseSparkForCausalLM.hf_to_vllm_mapper | OMNI_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        hf_config = vllm_config.model_config.hf_config
        text_vllm_config = copy(vllm_config)
        if hasattr(hf_config, "text_config"):
            text_vllm_config.model_config = copy(vllm_config.model_config)
            text_vllm_config.model_config.hf_config = hf_config.text_config
        super().__init__(vllm_config=text_vllm_config, prefix=prefix)

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        return name.startswith(_TEXT_ONLY_SKIP_PREFIXES)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered = (
            (name, tensor)
            for name, tensor in weights
            if not name.startswith(_TEXT_ONLY_SKIP_PREFIXES)
        )
        return super().load_weights(filtered)


TPUMuseSparkTextOnlyForConditionalGeneration = MuseSparkForConditionalGeneration

__all__ = [
    "MuseSparkForConditionalGeneration",
    "TPUMuseSparkAttention",
    "TPUMuseSparkDecoderLayer",
    "TPUMuseSparkForCausalLM",
    "TPUMuseSparkMoE",
    "TPUMuseSparkModel",
    "TPUMuseSparkTextOnlyForConditionalGeneration",
]
