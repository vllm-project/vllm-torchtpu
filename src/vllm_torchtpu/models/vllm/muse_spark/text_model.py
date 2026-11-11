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
"""Muse Spark Open text decoder and causal LM for vLLM on TorchTPU."""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import islice
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    RoutedExperts,
    UnquantizedFusedMoEMethod,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import ModelOptQuantConfigBase
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.custom_ops import muse_moe_tpu
from vllm_torchtpu.layers.adapter.custom_ops.batch_invariant_ops import (
    det_centered_rms_norm,
    det_input_scaled_rms_norm,
    det_input_scaled_rms_norm_residual_gate,
    det_o_norm_sigmoid_gate,
    det_qkv_qk_norm_rope,
    det_rms_norm_residual_gate,
    det_rms_norm_scaled,
    prepare_det_ops,
)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

from .attention_compat import install_muse_hd64_attention
from .batch_invariant import (
    batch_invariant_enabled,
    det_lm_head_logits,
    det_projection,
    pad_lm_head_weight,
    rms_normalize,
)
from .configuration import (
    MuseSparkConfig,
    MuseSparkDSparkConfig,
    MuseSparkTextConfig,
)
from .det_collectives import (
    det_all_reduce,
    det_attn_post_sandwich,
    det_dense_post_sandwich,
    det_moe_post_sandwich,
    prepare_det_collectives,
    tp_position_block,
    tp_shard_index,
)

logger = init_logger(__name__)


def get_muse_spark_text_config(config: Any) -> MuseSparkTextConfig:
    """Return the text subconfig from a multimodal or text Muse Spark config."""
    if isinstance(config, MuseSparkTextConfig):
        return config
    return config.get_text_config()


def normalize_muse_spark_hf_quant_config(hf_config: Any) -> None:
    """Ensure ModelOpt NVFP4 exclude_modules and modules_to_not_convert populate ignore.

    vLLM 0.29's ModelOptQuantConfigBase.from_config reads config['ignore'] when
    there is no top level 'quantization' subdict, whereas the Muse Spark NVFP4
    config.json stores exclusions under 'exclude_modules' and
    'modules_to_not_convert'. Merging them into 'ignore' before quantization
    config construction preserves all BF16 exclusions.
    """
    for cfg in (hf_config, getattr(hf_config, "text_config", None)):
        if cfg is None:
            continue
        qcfg = getattr(cfg, "quantization_config", None)
        if not isinstance(qcfg, dict):
            continue
        if qcfg.get("quant_method") == "modelopt" and "quantization" not in qcfg:
            ignore = list(qcfg.get("ignore") or [])
            seen = set(ignore)
            for key in ("exclude_modules", "modules_to_not_convert"):
                for item in qcfg.get(key) or []:
                    if item not in seen:
                        ignore.append(item)
                        seen.add(item)
            if ignore:
                qcfg["ignore"] = ignore


def normalize_muse_spark_quant_config(
    quant_config: QuantizationConfig | None,
    hf_config: Any,
    mapper: WeightsMapper,
    packed_mapping: dict[str, list[str]],
) -> None:
    """Propagate checkpoint exclusions and packed module mappings to quant_config."""
    if quant_config is None:
        return
    quant_config.packed_modules_mapping.update(packed_mapping)
    # The exclusion list each supported quant config consults.
    if isinstance(quant_config, ModelOptQuantConfigBase):
        target = quant_config.exclude_modules
    elif isinstance(quant_config, Fp8Config):
        target = quant_config.ignored_layers
    else:
        return
    qcfg = getattr(hf_config, "quantization_config", None)
    if not isinstance(qcfg, dict):
        text_cfg = getattr(hf_config, "text_config", None)
        qcfg = getattr(text_cfg, "quantization_config", None) if text_cfg else None
    raw_excludes: list[str] = []
    if isinstance(qcfg, dict):
        for key in ("ignore", "exclude_modules", "modules_to_not_convert", "exclude"):
            for item in qcfg.get(key) or []:
                if item not in raw_excludes:
                    raw_excludes.append(item)
    if not raw_excludes:
        return
    rename_mapper = mapper.get_rename_mapper()
    mapped_excludes = rename_mapper.apply_list(raw_excludes)
    seen = set(target)
    for item in raw_excludes + mapped_excludes:
        if item is not None and item not in seen:
            target.append(item)
            seen.add(item)


# MuseSparkDSparkModel always builds these blocks, so a draft config that
# turns one off describes a different architecture.
_DSPARK_ALWAYS_ON_FLAGS = (
    "use_qk_norm",
    "use_attn_o_gate",
    "use_attn_o_norm",
    "use_residual_gate",
)


def _reject_unsupported_dspark_config(hf_cfg: Any) -> None:
    """Raise for Muse DSpark draft settings MuseSparkDSparkModel cannot run.

    Keys a config does not set keep the model's behavior.
    """
    disabled = [f for f in _DSPARK_ALWAYS_ON_FLAGS if getattr(hf_cfg, f, True) is False]
    if disabled:
        raise ValueError(
            f"Muse Spark DSpark draft sets {disabled} to False; the TPU DSpark "
            "model only implements these blocks enabled."
        )


def validate_megakernel_config(vllm_config: VllmConfig) -> None:
    """Reject setups the VLLM_TPU_MUSE_MEGAKERNEL=1 decode path does not support.

    The megakernel decodes owned requests outside vLLM's forward: a pure
    megakernel step returns before the DP lockstep collective, the PP stage
    handoff and the KV connector save, and only the DSpark drafter is fused
    into it.
    """
    parallel_cfg = vllm_config.parallel_config
    if parallel_cfg.tensor_parallel_size != 8:
        raise NotImplementedError(
            "VLLM_TPU_MUSE_MEGAKERNEL=1 requires tensor_parallel_size=8 "
            f"on TPU, got {parallel_cfg.tensor_parallel_size}."
        )
    if parallel_cfg.data_parallel_size > 1:
        raise NotImplementedError(
            "VLLM_TPU_MUSE_MEGAKERNEL=1 does not support data_parallel_size > 1 "
            f"(got {parallel_cfg.data_parallel_size}): megakernel-only steps skip "
            "the DP lockstep collective."
        )
    if parallel_cfg.pipeline_parallel_size > 1:
        raise NotImplementedError(
            "VLLM_TPU_MUSE_MEGAKERNEL=1 does not support pipeline_parallel_size "
            f"> 1 (got {parallel_cfg.pipeline_parallel_size})."
        )
    if vllm_config.kv_transfer_config is not None:
        raise NotImplementedError(
            "VLLM_TPU_MUSE_MEGAKERNEL=1 does not support KV connectors "
            "(--kv-transfer-config): megakernel-decoded KV is not saved through "
            "the connector."
        )
    spec_cfg = vllm_config.speculative_config
    if spec_cfg is not None and spec_cfg.method != "dspark":
        raise NotImplementedError(
            "VLLM_TPU_MUSE_MEGAKERNEL=1 supports only the dspark speculative "
            f"method, got {spec_cfg.method!r}."
        )
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.tpu_info import get_chip_version

    chip = get_chip_version()
    if chip != pltpu.ChipVersion.TPU_7X:
        raise NotImplementedError(
            f"VLLM_TPU_MUSE_MEGAKERNEL=1 supports only TPU v7x, got {chip.value}."
        )


class TPUMuseSparkConfig(VerifyAndUpdateConfig):
    """MODELS_CONFIG_MAP hook for Muse Spark Open on TPU."""

    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None:
        normalize_muse_spark_hf_quant_config(model_config.hf_config)
        normalize_muse_spark_hf_quant_config(model_config.model_arch_config)

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        normalize_muse_spark_hf_quant_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        if cache_config is not None and not cache_config.user_specified_block_size:
            logger.info_once(
                "Muse Spark: overriding KV cache block_size %s -> 128 "
                "(pass --block-size to choose another value).",
                cache_config.block_size,
            )
            cache_config.block_size = 128  # type: ignore[assignment]
            cache_config.user_specified_block_size = True

        if envs.VLLM_TPU_MUSE_MEGAKERNEL:
            validate_megakernel_config(vllm_config)
            sched_cfg = vllm_config.scheduler_config
            if sched_cfg.async_scheduling is not False:
                logger.info_once(
                    "VLLM_TPU_MUSE_MEGAKERNEL=1: disabling async_scheduling for "
                    "synchronous decode slot ownership."
                )
                sched_cfg.async_scheduling = False

        spec_config = vllm_config.speculative_config
        if (
            spec_config is not None
            and spec_config.method == "dspark"
            and spec_config.draft_model_config is not None
        ):
            draft_cfg = spec_config.draft_model_config
            hf_cfg = draft_cfg.hf_config
            target_hf = vllm_config.model_config.hf_config
            is_muse_target = isinstance(
                target_hf,
                (MuseSparkConfig, MuseSparkTextConfig, MuseSparkDSparkConfig),
            )
            if hf_cfg is not None and (
                is_muse_target or isinstance(hf_cfg, MuseSparkDSparkConfig)
            ):
                _reject_unsupported_dspark_config(hf_cfg)
                cast(Any, hf_cfg).model_type = "muse_spark_dspark"
                hf_cfg.architectures = ["MuseSparkDSparkModel"]
                # vLLM gives a dspark draft the target's quantization (and
                # copies its quantization_config); the Muse draft is bf16.
                # A quantized draft checkpoint still fails loudly at load:
                # its scale tensors have no matching bf16 parameter.
                draft_cfg.quantization = None
                draft_cfg.quantization_config = None
                spec_config.update_arch_()


class MuseSparkRMSNorm(nn.Module):
    """RMSNorm computed in fp32 with optional plain weight scale."""

    def __init__(
        self,
        dim: int | None = None,
        eps: float = 1e-6,
        with_scale: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.with_scale = bool(with_scale)
        if self.with_scale:
            assert dim is not None
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return rms_normalize(x, self.eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.with_scale and batch_invariant_enabled(hidden_states):
            return det_rms_norm_scaled(
                hidden_states, self.eps, scale=1.0, out_dtype=hidden_states.dtype
            )
        out = self._norm(hidden_states.float())
        if self.with_scale and self.weight is not None:
            out = out * self.weight.float()
        return out.type_as(hidden_states)


class MuseSparkCenteredRMSNorm(nn.Module):
    """Zero centered RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return rms_normalize(x, self.eps)

    def forward(
        self, x: torch.Tensor, *, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        target_dtype = out_dtype if out_dtype is not None else x.dtype
        if batch_invariant_enabled(x):
            return det_centered_rms_norm(
                x, self.weight, self.eps, out_dtype=target_dtype
            )
        out = self._norm(x.float()) * (1.0 + self.weight.float())
        return out.to(target_dtype)


class MuseSparkInputScaledRMSNorm(nn.Module):
    """InputScaledRMSNorm: scale by (1 + weight), round to input dtype, then RMS."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return rms_normalize(x, self.eps)

    def forward(
        self, x: torch.Tensor, *, round_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        target_dtype = round_dtype if round_dtype is not None else x.dtype
        if batch_invariant_enabled(x) and target_dtype == x.dtype:
            return det_input_scaled_rms_norm(
                x, self.weight, self.eps, out_dtype=target_dtype
            )
        scaled = (x.float() * (1.0 + self.weight.float())).to(target_dtype)
        out = self._norm(scaled.float())
        return out.to(target_dtype)


def compute_residual_gate_alpha_beta(
    gate: torch.Tensor,
    temperature: float = 0.3,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (alpha, beta) fp32 residual gate coefficients from gate parameter."""
    w = gate.float() / float(temperature)
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=float(eps)))
    return alpha, beta


class MuseSparkResidualGate(nn.Module):
    """Learned channelwise residual gate with precomputed fp32 (alpha, beta)."""

    alpha: torch.Tensor | None
    beta: torch.Tensor | None

    def __init__(
        self,
        config: MuseSparkTextConfig | MuseSparkDSparkConfig,
        residual_gate_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(config.hidden_size, dtype=torch.float32))
        self.temperature = float(config.residual_gate_temperature)
        self.eps = float(residual_gate_eps)
        self.residual_fp32 = config.residual_dtype == "fp32"
        self.register_buffer("alpha", None, persistent=False)
        self.register_buffer("beta", None, persistent=False)

    def process_weights_after_loading(self) -> None:
        with torch.no_grad():
            alpha, beta = compute_residual_gate_alpha_beta(
                self.gate, self.temperature, self.eps
            )
            alpha_buf = torch.empty_like(alpha)
            beta_buf = torch.empty_like(beta)
            alpha_buf.copy_(alpha)
            beta_buf.copy_(beta)
            self.register_buffer("alpha", alpha_buf, persistent=False)
            self.register_buffer("beta", beta_buf, persistent=False)
            if alpha_buf.device.type == "tpu":
                synchronize_tensors([alpha_buf, beta_buf])

    def forward(
        self, residual: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.alpha is not None and self.beta is not None:
            alpha, beta = self.alpha, self.beta
        else:
            alpha, beta = compute_residual_gate_alpha_beta(
                self.gate, self.temperature, self.eps
            )
        out = alpha * residual.float() + beta * hidden_states.float()
        if self.residual_fp32:
            return out
        return out.to(hidden_states.dtype)


class MuseSparkAttention(nn.Module):
    """Muse Spark GQA with per head QK/O norms, output sigmoid gate, and SWA/NoPE."""

    def __init__(
        self,
        config: MuseSparkTextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = extract_layer_index(prefix)

        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.total_num_heads}) must be divisible "
                f"by tensor_parallel_size ({tp_size})"
            )
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"num_key_value_heads ({self.total_num_kv_heads}) must be "
                    f"divisible by tensor_parallel_size ({tp_size})"
                )
        else:
            if tp_size % self.total_num_kv_heads != 0:
                raise ValueError(
                    f"tensor_parallel_size ({tp_size}) must be divisible by "
                    f"num_key_value_heads ({self.total_num_kv_heads})"
                )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = float(self.head_dim**-0.5)
        self.qk_scale_factor = float(config.qk_scale_factor)

        self.is_sliding = config.layer_types[self.layer_idx] == "sliding_attention"
        window = config.sliding_window
        self.sliding_window = int(window) if self.is_sliding and window else None
        layer_theta = config.layer_rope_theta[self.layer_idx]
        self.use_rope = bool(layer_theta and layer_theta > 0)

        self.qk_norm = MuseSparkRMSNorm(eps=config.rms_norm_eps, with_scale=False)
        self.o_norm = MuseSparkRMSNorm(eps=config.rms_norm_eps, with_scale=False)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bool(config.attention_bias),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        # Reduced by det_all_reduce in forward (row invariant).
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.o_proj",
        )
        if config.attention_bias:
            raise ValueError("Muse Spark attention has no bias")

        if self.use_rope:
            rope_params = dict(config.rope_parameters)
            rope_params["rope_theta"] = float(layer_theta)
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=rope_params,
                is_neox_style=True,
                dtype=torch.float32,
            )
        else:
            self.rotary_emb = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=None,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn",
        )
        install_muse_hd64_attention(self.attn)

    def forward_attn_only(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv_w = self.qkv_proj.weight
        if (
            isinstance(qkv_w, torch.Tensor)
            and batch_invariant_enabled(hidden_states)
            and getattr(self.qkv_proj, "bias", None) is None
            and qkv_w.dtype == torch.bfloat16
        ):
            flipped = bool(getattr(self.qkv_proj, WEIGHT_FLIPPED_ATTR, False))
            cs_cache = (
                self.rotary_emb.cos_sin_cache if self.rotary_emb is not None else None
            )
            q, k, v = det_qkv_qk_norm_rope(
                hidden_states,
                qkv_w,
                cs_cache,
                positions if self.rotary_emb is not None else None,
                self.qk_norm.eps,
                self.qk_scale_factor,
                self.q_size,
                self.kv_size,
                self.head_dim,
                rhs_nk=not flipped,
            )
            return self.attn(q, k, v)
        qkv = det_projection(self.qkv_proj, hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if batch_invariant_enabled(q):
            q = det_rms_norm_scaled(
                q.reshape(-1, self.head_dim),
                self.qk_norm.eps,
                scale=self.qk_scale_factor,
                out_dtype=v.dtype,
            ).reshape(-1, self.q_size)
            k = det_rms_norm_scaled(
                k.reshape(-1, self.head_dim),
                self.qk_norm.eps,
                scale=1.0,
                out_dtype=v.dtype,
            ).reshape(-1, self.kv_size)
        else:
            q = (
                self.qk_norm(q.reshape(-1, self.head_dim)).reshape(-1, self.q_size)
                * self.qk_scale_factor
            ).to(v.dtype)
            k = (
                self.qk_norm(k.reshape(-1, self.head_dim))
                .reshape(-1, self.kv_size)
                .to(v.dtype)
            )
        if self.rotary_emb is not None:
            q_f32, k_f32 = self.rotary_emb(positions, q.float(), k.float())
            q, k = q_f32.to(v.dtype), k_f32.to(v.dtype)
        return self.attn(q, k, v)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = self.forward_attn_only(positions, hidden_states)
        gate = det_projection(self.gate_proj, hidden_states)
        if batch_invariant_enabled(attn_output):
            attn_output = det_o_norm_sigmoid_gate(
                attn_output, gate, self.o_norm.eps, self.head_dim
            )
        else:
            attn_output = self.o_norm(attn_output.reshape(-1, self.head_dim)).reshape(
                -1, self.q_size
            )
            attn_output = attn_output * torch.sigmoid(gate)
        output = det_projection(self.o_proj, attn_output)
        output = det_all_reduce(output)
        return output


class MuseSparkMLP(nn.Module):
    """Dense SwiGLU MLP fallback when mlp_layer_types[i] is not sparse."""

    def __init__(
        self,
        config: MuseSparkTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation {config.hidden_act!r}")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def _moe_combine(m_fp32: torch.Tensor, act_dtype: torch.dtype) -> torch.Tensor:
    """Fixed-order EP sum of the MoE partials in fp32."""
    return det_all_reduce(m_fp32, out_dtype=act_dtype)


class MuseSparkMoE(nn.Module):
    """Muse Spark compressed bottleneck MoE with per expert InputScaledRMSNorm."""

    def __init__(
        self,
        config: MuseSparkTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.moe_hidden_size = int(config.moe_hidden_size)
        self.moe_intermediate_size = int(config.moe_intermediate_size)
        self.num_experts = int(config.num_local_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.output_multiplier = float(config.output_multiplier)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.post_norm_eps = float(config.post_norm_eps)
        self.router_eps = 1e-15

        self.tp_size = get_tensor_model_parallel_world_size()
        vllm_config = get_current_vllm_config_or_none()
        self.use_ep = bool(
            vllm_config is not None
            and vllm_config.parallel_config.enable_expert_parallel
        )
        if self.tp_size > 1 and not self.use_ep:
            raise ValueError(
                "MuseSparkMoE requires --enable-expert-parallel when "
                f"tensor_parallel_size ({self.tp_size}) > 1 because the per expert "
                "post norm requires each expert's complete 4096 wide output."
            )

        self.gate = GateLinear(
            input_size=self.hidden_size,
            output_size=self.num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        bias_param = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.gate.e_score_correction_bias = bias_param
        self.router_w_kmajor: torch.Tensor | None
        self.register_buffer("router_w_kmajor", None, persistent=False)
        params_dtype = config.dtype if isinstance(config.dtype, torch.dtype) else None

        self.pre_expert_proj = ColumnParallelLinear(
            self.hidden_size,
            self.moe_hidden_size,
            bias=False,
            gather_output=True,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.pre_expert_proj",
        )
        self.pre_expert_proj.quant_method = VllmUnquantizedLinearMethod()
        self.pre_expert_norm = MuseSparkCenteredRMSNorm(
            self.moe_hidden_size, eps=self.rms_norm_eps
        )

        self.experts = FusedMoEFactory(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.moe_hidden_size,
            intermediate_size=self.moe_intermediate_size,
            activation=config.hidden_act,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            e_score_correction_bias=bias_param,
            routed_scaling_factor=self.routed_scaling_factor,
            router_logits_dtype=torch.float32,
        )
        self.post_expert_norm = MuseSparkInputScaledRMSNorm(
            self.moe_hidden_size, eps=self.post_norm_eps
        )
        self.post_expert_proj = ColumnParallelLinear(
            self.moe_hidden_size,
            self.hidden_size,
            bias=False,
            gather_output=True,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.post_expert_proj",
        )
        self.post_expert_proj.quant_method = VllmUnquantizedLinearMethod()
        self._fused_moe_args: muse_moe_tpu.MuseFusedMoeArgs | None | bool = False
        if muse_moe_tpu.muse_moe_impl() == muse_moe_tpu.MOE_IMPL_EAGER:
            self._check_eager_supported()

    def _check_eager_supported(self) -> None:
        """Raise unless the eager MoE can read the routed expert weights.

        `_forward_eager` multiplies `w13_weight`/`w2_weight` as dense tensors and
        never reads block scales, so only unquantized routed experts are valid.
        """
        method = self.experts.routed_experts.quant_method
        if not isinstance(method, UnquantizedFusedMoEMethod):
            # TODO: dequantize NVFP4/FP8 experts in the eager MoE if a
            # non-fused reference path for quantized checkpoints is needed.
            raise NotImplementedError(
                f"{self.prefix}: the eager Muse Spark MoE supports only "
                f"unquantized routed experts, got {type(method).__name__}. "
                "Use VLLM_TPU_MUSE_MOE_IMPL=fused for quantized checkpoints."
            )

    def process_weights_after_loading(self) -> None:
        self.pre_expert_proj.quant_method.process_weights_after_loading(
            self.pre_expert_proj
        )
        self.post_expert_proj.quant_method.process_weights_after_loading(
            self.post_expert_proj
        )
        with torch.no_grad():
            gate_w = self.gate.weight
            assert isinstance(gate_w, torch.Tensor)
            if gate_w.shape == (self.num_experts, self.hidden_size):
                router_w = gate_w.float().t().contiguous()
                gate_w = nn.Parameter(router_w, requires_grad=False)
                self.gate.weight = gate_w
                if router_w.device.type == "tpu":
                    synchronize_tensors([gate_w])
            self.router_w_kmajor = gate_w
        self._fused_moe_args = False
        if gate_w.device.type == "tpu":
            self._fused_moe()

    def _fused_moe(self) -> muse_moe_tpu.MuseFusedMoeArgs | None:
        fused = self.__dict__.get("_fused_moe_args", False)
        if fused is False:
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "Muse Spark fused MoE op was not resolved before tracing; "
                    "process_weights_after_loading must run first."
                )
            fused = None
            if (
                self.gate.weight.device.type == "tpu"
                and muse_moe_tpu.muse_moe_impl() == muse_moe_tpu.MOE_IMPL_FUSED
            ):
                if self.router_w_kmajor is None:
                    self.process_weights_after_loading()
                    return self.__dict__.get("_fused_moe_args")
                fused = muse_moe_tpu.MuseFusedMoeArgs(self)
                if fused.op is None:
                    # Offloaded experts are served by an op installed later
                    # (megakernel shared store); anything else is a slow fallback.
                    w13 = cast(RoutedExperts, self.experts.routed_experts).w13_weight
                    assert isinstance(w13, torch.Tensor)
                    offloaded = w13.numel() == 0
                    log = logger.info_once if offloaded else logger.warning_once
                    log("Muse Spark fused MoE disabled: %s", fused.unsupported)
                    fused = None
            self.__dict__["_fused_moe_args"] = fused
        return fused  # type: ignore[return-value]

    def _forward_eager(self, x_flat: torch.Tensor) -> torch.Tensor:
        self._check_eager_supported()
        if isinstance(self.router_w_kmajor, torch.Tensor):
            router_logits = torch.matmul(x_flat.float(), self.router_w_kmajor)
        else:
            gate_w = self.gate.weight
            assert isinstance(gate_w, torch.Tensor)
            router_logits = F.linear(x_flat.float(), gate_w.float())
        router_logits = router_logits * self.output_multiplier
        scores = torch.sigmoid(router_logits)
        bias = self.gate.e_score_correction_bias
        assert isinstance(bias, torch.Tensor)
        scores_for_choice = scores + bias.float().unsqueeze(0)
        _, topk_ids = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
        topk_weights = scores.gather(1, topk_ids)
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + self.router_eps
        )
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor

        z = det_projection(self.pre_expert_proj, x_flat)
        z = self.pre_expert_norm(z, out_dtype=x_flat.dtype)

        routed = getattr(self.experts, "routed_experts", self.experts)
        w13 = routed.w13_weight
        w2 = routed.w2_weight
        assert isinstance(w13, torch.Tensor) and isinstance(w2, torch.Tensor)
        is_kmajor = (
            getattr(routed.quant_method, "_tpu_activation_str", None) is not None
        )
        local_num_experts = w13.shape[0]
        experts_start = getattr(routed, "_experts_start", None)
        start_id = int(experts_start.item()) if experts_start is not None else 0

        mixed_fp32 = torch.zeros(
            (z.shape[0], self.moe_hidden_size),
            dtype=torch.float32,
            device=z.device,
        )
        for local_e in range(local_num_experts):
            global_e = start_id + local_e if self.use_ep else local_e
            tok_idx, k_pos = torch.where(topk_ids == global_e)
            if tok_idx.numel() == 0:
                continue
            cur = z[tok_idx]
            if is_kmajor:
                gate_up = torch.matmul(
                    cur.float(),
                    w13[
                        local_e,
                        : self.moe_hidden_size,
                        : 2 * self.moe_intermediate_size,
                    ].float(),
                ).to(cur.dtype)
            else:
                gate_up = F.linear(cur, w13[local_e])
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
            if is_kmajor:
                down = torch.matmul(
                    act.float(),
                    w2[
                        local_e, : self.moe_intermediate_size, : self.moe_hidden_size
                    ].float(),
                ).to(cur.dtype)
            else:
                down = F.linear(act, w2[local_e])
            y_e = self.post_expert_norm(down, round_dtype=cur.dtype)
            weighted = y_e.float() * topk_weights[tok_idx, k_pos, None]
            mixed_fp32.index_add_(0, tok_idx, weighted)

        if self.tp_size > 1:
            mixed = _moe_combine(mixed_fp32, x_flat.dtype)
        else:
            mixed = mixed_fp32.to(x_flat.dtype)
        out = det_projection(self.post_expert_proj, mixed)
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x_flat = hidden_states.view(-1, self.hidden_size)
        fused = self._fused_moe()
        if fused is None:
            out = self._forward_eager(x_flat)
            return out.view(orig_shape)
        z_pre = (
            det_projection(self.pre_expert_proj, x_flat).contiguous()
            if fused.pre_projected
            else None
        )
        m_fp32 = fused(x_flat.contiguous(), z_pre)
        if self.tp_size > 1:
            m_act = _moe_combine(m_fp32, x_flat.dtype)
        else:
            m_act = m_fp32.to(x_flat.dtype)
        out = det_projection(self.post_expert_proj, m_act)
        return out.view(orig_shape)


class MuseSparkDecoderLayer(nn.Module):
    """Single Muse Spark decoder layer with sandwich norms and residual gates."""

    def __init__(
        self,
        config: MuseSparkTextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = int(config.hidden_size)

        self.self_attn = MuseSparkAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        if config.mlp_layer_types[self.layer_idx] == "sparse":
            self.mlp: nn.Module = MuseSparkMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = MuseSparkMLP(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = MuseSparkCenteredRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MuseSparkRMSNorm(
            self.hidden_size, eps=config.post_norm_eps, with_scale=False
        )
        self.pre_feedforward_layernorm = MuseSparkCenteredRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = MuseSparkInputScaledRMSNorm(
            self.hidden_size, eps=config.post_norm_eps
        )
        self.post_attention_residual_gate = MuseSparkResidualGate(config)
        self.post_feedforward_residual_gate = MuseSparkResidualGate(config)
        self._tp_shard_idx: int | None = None
        self._in_norm_w_s: torch.Tensor | None = None
        self._post_w_s: torch.Tensor | None = None

    def _act_dtype(self, fallback: torch.dtype) -> torch.dtype:
        w = getattr(self.self_attn.qkv_proj, "weight", None)
        if w is not None and w.dtype in (torch.bfloat16, torch.float16, torch.float32):
            return w.dtype
        return torch.bfloat16 if fallback == torch.float32 else fallback

    def process_weights_after_loading(self) -> None:
        if self.post_attention_residual_gate.alpha is None:
            self.post_attention_residual_gate.process_weights_after_loading()
        if self.post_feedforward_residual_gate.alpha is None:
            self.post_feedforward_residual_gate.process_weights_after_loading()
        tp_size = get_tensor_model_parallel_world_size()
        if (
            tp_size > 1
            and self.input_layernorm.weight.device.type == "tpu"
            and self.hidden_size % (tp_size * 128) == 0
            and self.post_attention_residual_gate.alpha is not None
            and self.post_attention_residual_gate.beta is not None
            and self.post_feedforward_residual_gate.alpha is not None
            and self.post_feedforward_residual_gate.beta is not None
        ):
            shard = tp_shard_index()
            self._tp_shard_idx = shard
            hs = self.hidden_size // tp_size
            s0, s1 = shard * hs, (shard + 1) * hs
            with torch.no_grad():
                self._in_norm_w_s = (
                    self.input_layernorm.weight.data[s0:s1].reshape(1, hs).contiguous()
                )
                self._pre_ffn_w_s = (
                    self.pre_feedforward_layernorm.weight.data[s0:s1]
                    .reshape(1, hs)
                    .contiguous()
                )
                self._post_ffn_w_s = (
                    self.post_feedforward_layernorm.weight.data[s0:s1]
                    .reshape(1, hs)
                    .contiguous()
                )
                self._attn_alpha_s = (
                    self.post_attention_residual_gate.alpha.data[s0:s1]
                    .reshape(1, hs)
                    .contiguous()
                )
                self._attn_beta_s = (
                    self.post_attention_residual_gate.beta.data[s0:s1]
                    .reshape(1, hs)
                    .contiguous()
                )
                self._ffn_alpha_s = (
                    self.post_feedforward_residual_gate.alpha.data[s0:s1]
                    .reshape(1, hs)
                    .contiguous()
                )
                self._ffn_beta_s = (
                    self.post_feedforward_residual_gate.beta.data[s0:s1]
                    .reshape(1, hs)
                    .contiguous()
                )
                self._post_w_s = None
                if isinstance(self.mlp, MuseSparkMoE):
                    post_proj_w = self.mlp.post_expert_proj.weight
                    assert isinstance(post_proj_w, torch.Tensor)
                    pw = post_proj_w.data
                    flipped = bool(
                        getattr(self.mlp.post_expert_proj, WEIGHT_FLIPPED_ATTR, False)
                    )
                    blk = tp_position_block(pw, 1 if flipped else 0)
                    if blk is not pw:
                        self._post_w_s = blk
                        synchronize_tensors([blk])
                synchronize_tensors(
                    [
                        self._in_norm_w_s,
                        self._pre_ffn_w_s,
                        self._post_ffn_w_s,
                        self._attn_alpha_s,
                        self._attn_beta_s,
                        self._ffn_alpha_s,
                        self._ffn_beta_s,
                    ]
                )

    def forward_sharded(
        self,
        positions: torch.Tensor,
        h_cur: torch.Tensor,
        res_s: torch.Tensor,
        next_layer: MuseSparkDecoderLayer | None,
        *,
        gather_hid: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        attn_out = self.self_attn.forward_attn_only(
            positions=positions, hidden_states=h_cur
        ).reshape(-1, self.self_attn.q_size)
        gate_w = self.self_attn.gate_proj.weight
        o_w = self.self_attn.o_proj.weight
        assert isinstance(gate_w, torch.Tensor) and isinstance(o_w, torch.Tensor)
        gate_nk = not bool(
            getattr(self.self_attn.gate_proj, WEIGHT_FLIPPED_ATTR, False)
        )
        o_nk = not bool(getattr(self.self_attn.o_proj, WEIGHT_FLIPPED_ATTR, False))

        if isinstance(self.mlp, MuseSparkMoE):
            fused = self.mlp._fused_moe()
            if fused is not None and fused.pre_projected:
                pre_w = self.mlp.pre_expert_proj.weight
                assert isinstance(pre_w, torch.Tensor)
                pre_nk = not bool(
                    getattr(self.mlp.pre_expert_proj, WEIGHT_FLIPPED_ATTR, False)
                )
                res_s, h_ffn, z_pre = det_attn_post_sandwich(
                    attn_out,
                    h_cur,
                    gate_w,
                    o_w,
                    res_s,
                    self._attn_alpha_s,
                    self._attn_beta_s,
                    self._pre_ffn_w_s,
                    o_eps=self.self_attn.o_norm.eps,
                    post_attn_eps=self.post_attention_layernorm.eps,
                    pre_ffn_eps=self.pre_feedforward_layernorm.eps,
                    gate_rhs_nk=gate_nk,
                    o_rhs_nk=o_nk,
                    pre_exp_w=pre_w,
                    pre_rhs_nk=pre_nk,
                    head_dim=self.self_attn.head_dim,
                )
                m_fp32 = fused(h_ffn, z_pre)
                post_w = self._post_w_s
                if post_w is None:
                    post_w = self.mlp.post_expert_proj.weight
                assert isinstance(post_w, torch.Tensor)
                post_nk = not bool(
                    getattr(self.mlp.post_expert_proj, WEIGHT_FLIPPED_ATTR, False)
                )
                next_w_s = next_layer._in_norm_w_s if next_layer is not None else None
                next_eps = (
                    next_layer.input_layernorm.eps
                    if next_layer is not None
                    else self.input_layernorm.eps
                )
                out = det_moe_post_sandwich(
                    m_fp32,
                    post_w,
                    res_s,
                    self._post_ffn_w_s,
                    self._ffn_alpha_s,
                    self._ffn_beta_s,
                    next_w_s,
                    post_ffn_eps=self.post_feedforward_layernorm.eps,
                    next_in_eps=next_eps,
                    post_rhs_nk=post_nk,
                    gather_hid=(gather_hid or next_w_s is None),
                )
                if next_w_s is None:
                    return h_ffn, res_s, out
                if gather_hid:
                    res_s, h_next, hid_full = out
                    return h_next, res_s, hid_full
                res_s, h_next = out
                return h_next, res_s, None

        res_s, h_ffn = det_attn_post_sandwich(
            attn_out,
            h_cur,
            gate_w,
            o_w,
            res_s,
            self._attn_alpha_s,
            self._attn_beta_s,
            self._pre_ffn_w_s,
            o_eps=self.self_attn.o_norm.eps,
            post_attn_eps=self.post_attention_layernorm.eps,
            pre_ffn_eps=self.pre_feedforward_layernorm.eps,
            gate_rhs_nk=gate_nk,
            o_rhs_nk=o_nk,
            head_dim=self.self_attn.head_dim,
        )
        ff_out = self.mlp(h_ffn).reshape(-1, self.hidden_size)
        shard = (
            self._tp_shard_idx if self._tp_shard_idx is not None else tp_shard_index()
        )
        hs = self.hidden_size // get_tensor_model_parallel_world_size()
        ff_s = ff_out[:, shard * hs : (shard + 1) * hs].contiguous()
        if next_layer is None:
            # Last layer (fused MoE off): there is no next input_layernorm. Use
            # this layer's weight as a stand-in and keep only the gathered
            # residual, as the fused branch does for its final layer.
            next_w_s, next_eps, gather_hid = (
                self._in_norm_w_s,
                self.input_layernorm.eps,
                True,
            )
        else:
            next_w_s, next_eps = next_layer._in_norm_w_s, next_layer.input_layernorm.eps
        assert next_w_s is not None
        out_d = det_dense_post_sandwich(
            ff_s,
            res_s,
            self._post_ffn_w_s,
            self._ffn_alpha_s,
            self._ffn_beta_s,
            next_w_s,
            post_ffn_eps=self.post_feedforward_layernorm.eps,
            next_in_eps=next_eps,
            gather_hid=gather_hid,
        )
        if gather_hid:
            res_s, h_next, hid_full = out_d
            return h_next, res_s, hid_full
        res_s, h_next = out_d
        return h_next, res_s, None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        act_dtype = self._act_dtype(hidden_states.dtype)

        residual = hidden_states
        h = self.input_layernorm(residual, out_dtype=act_dtype)
        attn_out = self.self_attn(positions=positions, hidden_states=h)
        if (
            batch_invariant_enabled(attn_out)
            and self.post_attention_residual_gate.alpha is not None
            and self.post_attention_residual_gate.beta is not None
        ):
            out_dt = (
                torch.float32
                if self.post_attention_residual_gate.residual_fp32
                else attn_out.dtype
            )
            hidden_states = det_rms_norm_residual_gate(
                residual,
                attn_out,
                self.post_attention_residual_gate.alpha,
                self.post_attention_residual_gate.beta,
                self.post_attention_layernorm.eps,
                out_dtype=out_dt,
            )
        else:
            attn_out = self.post_attention_layernorm(attn_out)
            hidden_states = self.post_attention_residual_gate(residual, attn_out)

        residual = hidden_states
        h = self.pre_feedforward_layernorm(residual, out_dtype=act_dtype)
        ff_out = self.mlp(h)
        if (
            batch_invariant_enabled(ff_out)
            and ff_out.dtype == act_dtype
            and self.post_feedforward_residual_gate.alpha is not None
            and self.post_feedforward_residual_gate.beta is not None
        ):
            out_dt = (
                torch.float32
                if self.post_feedforward_residual_gate.residual_fp32
                else ff_out.dtype
            )
            hidden_states = det_input_scaled_rms_norm_residual_gate(
                residual,
                ff_out,
                self.post_feedforward_layernorm.weight,
                self.post_feedforward_residual_gate.alpha,
                self.post_feedforward_residual_gate.beta,
                self.post_feedforward_layernorm.eps,
                out_dtype=out_dt,
            )
        else:
            ff_out = self.post_feedforward_layernorm(ff_out, round_dtype=act_dtype)
            hidden_states = self.post_feedforward_residual_gate(residual, ff_out)
        return hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    }
)
class MuseSparkTextModel(nn.Module, EagleModelMixin):
    """Muse Spark text backbone."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = get_muse_spark_text_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.residual_fp32 = config.residual_dtype == "fp32"

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.embed_norm = MuseSparkRMSNorm(eps=config.rms_norm_eps, with_scale=False)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MuseSparkDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = MuseSparkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # The fp32 residual stream starts from the unrounded embedding RMS.
        if self.residual_fp32:
            return self.embed_norm._norm(self.embed_tokens(input_ids).float())
        return self.embed_norm(self.embed_tokens(input_ids))

    def _act_dtype(self, fallback: torch.dtype) -> torch.dtype:
        w = getattr(self.embed_tokens, "weight", None)
        if w is not None and w.dtype in (torch.bfloat16, torch.float16, torch.float32):
            return w.dtype
        return torch.bfloat16 if fallback == torch.float32 else fallback

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(input_ids)
            act_dtype = self._act_dtype(hidden_states.dtype)
            if self.residual_fp32:
                hidden_states = hidden_states.float()
        else:
            assert intermediate_tensors is not None
            raw_hs = intermediate_tensors["hidden_states"]
            assert isinstance(raw_hs, torch.Tensor)
            hidden_states = raw_hs
            act_dtype = self._act_dtype(hidden_states.dtype)

        aux_hidden_states: list[torch.Tensor] = []
        if self.start_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states.to(act_dtype))

        first_layer = cast(MuseSparkDecoderLayer, self.layers[self.start_layer])
        orig_shape = hidden_states.shape
        x2 = hidden_states.reshape(-1, self.config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        # det_collectives works for any TP > 1, but TP=8 (v7x) is the only size
        # validated against the bit-exact eval baselines; others use the plain
        # per-layer loop below.
        if (
            batch_invariant_enabled(x2)
            and not torch.compiler.is_compiling()
            and self.residual_fp32
            and tp_size == 8
            and first_layer._in_norm_w_s is not None
            and get_pp_group().is_last_rank
        ):
            shard = (
                first_layer._tp_shard_idx
                if first_layer._tp_shard_idx is not None
                else tp_shard_index()
            )
            hs = self.config.hidden_size // tp_size
            h_cur = first_layer.input_layernorm(x2, out_dtype=act_dtype)
            res_s = x2[:, shard * hs : (shard + 1) * hs].contiguous()
            hid_full: torch.Tensor | None = None
            for layer_idx in range(self.start_layer, self.end_layer):
                layer = cast(MuseSparkDecoderLayer, self.layers[layer_idx])
                next_layer = (
                    cast(MuseSparkDecoderLayer, self.layers[layer_idx + 1])
                    if (layer_idx + 1) < self.end_layer
                    else None
                )
                need_aux = (layer_idx + 1) in self.aux_hidden_state_layers
                need_final = next_layer is None
                h_cur, res_s, hid_full = layer.forward_sharded(
                    positions,
                    h_cur,
                    res_s,
                    next_layer,
                    gather_hid=(need_aux or need_final),
                )
                if need_aux and hid_full is not None:
                    aux_hidden_states.append(hid_full.reshape(orig_shape).to(act_dtype))
            assert hid_full is not None
            hidden_states = hid_full.reshape(orig_shape)
        else:
            for layer_idx, raw_layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer),
                start=self.start_layer,
            ):
                layer = cast(MuseSparkDecoderLayer, raw_layer)
                hidden_states = layer(positions, hidden_states)
                if (layer_idx + 1) in self.aux_hidden_state_layers:
                    aux_hidden_states.append(hidden_states.to(act_dtype))

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states).to(act_dtype)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


_FUSED_EXPERT_RE = re.compile(
    r"^(?P<prefix>(?:language_model\.)?model\.layers\.\d+\.mlp\.experts)\."
    r"(?P<proj>gate_up_proj|down_proj)"
    r"(?:[._](?P<suffix>weight_scale_2|weight_scale|input_scale|weight))?$"
)


def _resolve_expert_param(
    params_dict: dict[str, nn.Parameter],
    experts_prefix: str,
    attr: str,
) -> tuple[str, nn.Parameter] | None:
    for candidate in (
        f"{experts_prefix}.routed_experts.{attr}",
        f"{experts_prefix}.{attr}",
    ):
        if candidate in params_dict:
            return candidate, params_dict[candidate]
    return None


def _load_fused_3d_expert_tensor(
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    routed_by_prefix: dict[str, RoutedExperts],
    loaded_params: set[str],
) -> bool:
    """Load a fused 3D expert tensor one local EP expert at a time."""
    m = _FUSED_EXPERT_RE.match(name)
    if m is None:
        return False
    experts_prefix = m.group("prefix")
    proj = m.group("proj")
    suffix = m.group("suffix") or "weight"
    routed = routed_by_prefix.get(experts_prefix)
    if routed is None:
        return True

    is_gate_up = proj == "gate_up_proj"
    base_prefix = "w13" if is_gate_up else "w2"
    if suffix == "weight":
        attr_candidates = [f"{base_prefix}_weight"]
    elif suffix == "weight_scale":
        attr_candidates = [
            f"{base_prefix}_weight_scale",
            f"{base_prefix}_weight_scale_inv",
        ]
    elif suffix == "weight_scale_2":
        attr_candidates = [f"{base_prefix}_weight_scale_2"]
    elif suffix == "input_scale":
        attr_candidates = [f"{base_prefix}_input_scale"]
    else:
        return False

    resolved = None
    for attr in attr_candidates:
        resolved = _resolve_expert_param(params_dict, experts_prefix, attr)
        if resolved is not None:
            break
    if resolved is None:
        return True

    param_name, param = resolved
    param_loader = cast(Any, param).weight_loader
    num_global_experts = int(
        getattr(routed, "global_num_experts", loaded_weight.shape[0])
    )

    if suffix in ("weight", "weight_scale"):
        for expert_id in range(num_global_experts):
            if routed._map_global_expert_id_to_local_expert_id(expert_id) == -1:
                continue
            expert_tensor = loaded_weight[expert_id]
            if is_gate_up:
                half = expert_tensor.shape[0] // 2
                param_loader(
                    param,
                    expert_tensor[:half],
                    param_name,
                    shard_id="w1",
                    expert_id=expert_id,
                )
                param_loader(
                    param,
                    expert_tensor[half:],
                    param_name,
                    shard_id="w3",
                    expert_id=expert_id,
                )
            else:
                param_loader(
                    param,
                    expert_tensor,
                    param_name,
                    shard_id="w2",
                    expert_id=expert_id,
                )
        loaded_params.add(param_name)
        # Also mark input_scale loaded when weight lands in case a W4A16
        # checkpoint omits input_scale tensors.
        if suffix == "weight":
            in_scale = _resolve_expert_param(
                params_dict, experts_prefix, f"{base_prefix}_input_scale"
            )
            if in_scale is not None:
                loaded_params.add(in_scale[0])
        return True

    if suffix in ("weight_scale_2", "input_scale"):
        for expert_id in range(num_global_experts):
            local_id = routed._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                continue
            if loaded_weight.ndim == 0:
                val = loaded_weight
            elif loaded_weight.shape[0] == 1 and num_global_experts > 1:
                val = loaded_weight[0]
            else:
                val = loaded_weight[expert_id]
            if is_gate_up:
                if val.ndim >= 1 and val.numel() >= 2:
                    v1 = val.reshape(-1)[0]
                    v3 = val.reshape(-1)[1]
                else:
                    v1 = v3 = val.reshape(())
                param_loader(
                    param,
                    v1,
                    param_name,
                    shard_id="w1",
                    expert_id=expert_id,
                )
                param_loader(
                    param,
                    v3,
                    param_name,
                    shard_id="w3",
                    expert_id=expert_id,
                )
                if (
                    suffix == "input_scale"
                    and param.data.ndim == 2
                    and param.data.shape[1] == 2
                ):
                    param.data[local_id, 0] = v1.to(
                        device=param.data.device, dtype=param.data.dtype
                    )
                    param.data[local_id, 1] = v3.to(
                        device=param.data.device, dtype=param.data.dtype
                    )
            else:
                v2 = val.reshape(())
                param_loader(
                    param,
                    v2,
                    param_name,
                    shard_id="w2",
                    expert_id=expert_id,
                )
        loaded_params.add(param_name)
        return True

    return False


def load_muse_spark_text_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    mapper: WeightsMapper,
    num_experts: int,
    tie_word_embeddings: bool = False,
    ignore_prefixes: tuple[str, ...] = (),
) -> set[str]:
    """Load Muse Spark text and MoE weights (both fused 3D and per expert)."""
    params_dict = dict(model.named_parameters())
    routed_by_prefix: dict[str, RoutedExperts] = {}
    for mod_name, mod in model.named_modules():
        if isinstance(mod, RoutedExperts):
            routed_by_prefix[mod_name] = mod
            if mod_name.endswith(".routed_experts"):
                routed_by_prefix[mod_name.removesuffix(".routed_experts")] = mod

    expert_params_mapping = (
        fused_moe_make_expert_params_mapping(
            model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )
        + fused_moe_make_expert_params_mapping(
            model,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=num_experts,
        )
        if num_experts > 0
        else []
    )
    loaded_experts: set[str] = set()
    # Checkpoint layers past a num_hidden_layers override (truncated parity runs)
    # have no module; skip them instead of failing the load.
    inner_model = cast(Any, model).model
    num_layers = len(inner_model.layers)

    def _ordinary_weights():
        for name, loaded_weight in mapper.apply(weights):
            if ignore_prefixes and name.startswith(ignore_prefixes):
                continue
            layer_match = re.match(r"model\.layers\.(\d+)\.", name)
            if layer_match and int(layer_match.group(1)) >= num_layers:
                continue
            if _load_fused_3d_expert_tensor(
                name, loaded_weight, params_dict, routed_by_prefix, loaded_experts
            ):
                continue
            for (
                param_name,
                weight_name,
                expert_id,
                expert_shard_id,
            ) in expert_params_mapping:
                if weight_name not in name:
                    continue
                candidate = name.replace(weight_name, param_name)
                if candidate not in params_dict:
                    continue
                parameter = params_dict[candidate]
                cast(Any, parameter).weight_loader(
                    parameter,
                    loaded_weight,
                    candidate,
                    expert_id=expert_id,
                    shard_id=expert_shard_id,
                )
                loaded_experts.add(candidate)
                break
            else:
                yield name, loaded_weight

    loader = AutoWeightsLoader(model)
    skip_mapper = (
        WeightsMapper(orig_to_new_prefix={"lm_head.": None})
        if tie_word_embeddings
        else None
    )
    ordinary_loaded = loader.load_weights(_ordinary_weights(), mapper=skip_mapper)
    return loaded_experts | ordinary_loaded


def init_muse_spark_moe_metadata(
    model: Any, text_model: MuseSparkTextModel, config: MuseSparkTextConfig
) -> None:
    """Populate the attributes the MixtureOfExperts contract requires."""
    moe_mlps: list[MuseSparkMoE] = [
        mlp
        for layer in islice(
            text_model.layers, text_model.start_layer, text_model.end_layer
        )
        if isinstance((mlp := getattr(layer, "mlp", None)), MuseSparkMoE)
    ]
    model.expert_weights = []
    model.moe_mlp_layers = moe_mlps
    model.moe_layers = [mlp.experts for mlp in moe_mlps]
    model.num_moe_layers = len(moe_mlps)
    model.num_expert_groups = 1
    model.num_logical_experts = int(config.num_local_experts)
    model.num_physical_experts = model.num_logical_experts
    local = model.num_logical_experts
    if moe_mlps:
        routed = getattr(moe_mlps[0].experts, "routed_experts", moe_mlps[0].experts)
        local = int(getattr(routed, "local_num_experts", local))
    model.num_local_physical_experts = local
    model.num_routed_experts = model.num_logical_experts
    model.num_shared_experts = 0
    model.num_redundant_experts = 0


def muse_spark_norm_eps(root: nn.Module) -> tuple[float, ...]:
    """Distinct eps values of the Muse norms (one batch invariant op each)."""
    norms = (MuseSparkRMSNorm, MuseSparkCenteredRMSNorm, MuseSparkInputScaledRMSNorm)
    return tuple(sorted({m.eps for m in root.modules() if isinstance(m, norms)}))


def process_muse_spark_text_modules_after_loading(root: nn.Module) -> None:
    """Run post load weight preparation on residual gates and MoE blocks."""
    for module in root.modules():
        if isinstance(module, (MuseSparkResidualGate, MuseSparkMoE)):
            module.process_weights_after_loading()
    for module in root.modules():
        if isinstance(module, MuseSparkDecoderLayer):
            module.process_weights_after_loading()


class MuseSparkForCausalLM(
    nn.Module,
    MixtureOfExperts,
    SupportsQuant,
    SupportsPP,
    SupportsEagle3,
):
    """Text only Muse Spark causal language model for vLLM on TorchTPU."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".mlp.experts.post_expert_norm": ".mlp.post_expert_norm",
        },
        orig_to_new_prefix={
            "model.rotary_emb.": None,
            "model.language_model.rotary_emb.": None,
            "model.language_model.": "model.",
            "language_model.": "model.",
            "model.vision_tower.": None,
            "vision_tower.": None,
            "model.vision_adapter.": None,
            "vision_adapter.": None,
            "model.vision_projection.": None,
            "vision_projection.": None,
            "model.perception_emb_norm.": None,
            "perception_emb_norm.": None,
            "model.audio_tower.": None,
            "audio_tower.": None,
        },
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "w13_weight": ["gate_up_proj", "gate_proj", "up_proj"],
        "w2_weight": ["down_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = get_muse_spark_text_config(config)
        quant_config = vllm_config.quant_config
        assert self.hf_to_vllm_mapper is not None
        normalize_muse_spark_quant_config(
            quant_config,
            config,
            self.hf_to_vllm_mapper,
            self.packed_modules_mapping,
        )
        self.config = config
        self.text_config = text_config
        self.quant_config = quant_config

        self.model = MuseSparkTextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if text_config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.output_multiplier = float(text_config.output_multiplier)
        self.final_logit_softcapping = (
            float(text_config.final_logit_softcapping)
            if text_config.final_logit_softcapping is not None
            else None
        )
        self.logits_processor = LogitsProcessor(text_config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        init_muse_spark_moe_metadata(self, self.model, text_config)

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None:
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts

    def get_language_model(self) -> Any:
        # Spec decode shares `get_language_model().model.embed_tokens` and
        # `.lm_head` with drafts, so this wrapper (not the backbone) is the LM.
        return self

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...] | list[int]) -> None:
        """Tag aux layers as 1-indexed ids: `i` captures the output of layer `i - 1`.

        `0` captures the embedding output. Spec decode callers (dflash, eagle3)
        already pass `target_layer_id + 1`, so the ids are stored as given.
        """
        num_layers = len(self.model.layers)
        invalid = [lyr for lyr in layers if lyr < 0 or lyr > num_layers]
        if invalid:
            raise ValueError(
                f"Invalid Muse Spark aux hidden-state layers {invalid}: expected "
                f"1-indexed ids in [0, {num_layers}]."
            )
        self.model.aux_hidden_state_layers = tuple(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.get_eagle3_default_aux_hidden_state_layers()

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        if num_layers == MuseSparkDSparkConfig.DEFAULT_TARGET_NUM_HIDDEN_LAYERS:
            # The released DSpark draft's target_layer_ids, 1-indexed.
            return tuple(i + 1 for i in MuseSparkDSparkConfig.DEFAULT_TARGET_LAYER_IDS)
        return (2, num_layers // 2, num_layers)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.text_config.num_local_experts,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        del kwargs
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        padded = self._det_lm_head()
        if padded is not None:
            shard_vocab = int(
                getattr(self.lm_head, "_muse_shard_vocab", padded.shape[0])
            )
            logits = det_lm_head_logits(
                hidden_states,
                padded,
                shard_vocab,
                self.logits_processor.org_vocab_size,
                int(getattr(self.lm_head, "tp_size", 1)),
            )
        else:
            logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        logits = logits.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return logits

    def process_weights_after_loading(self) -> None:
        process_muse_spark_text_modules_after_loading(self)
        self._det_lm_head()

    def _det_lm_head(self) -> torch.Tensor | None:
        """Lane aligned lm_head shard for the batch invariant head, built in place.

        Pads this rank's vocab shard with zero rows up to the lane multiple and
        replaces head.weight in place, so HBM holds only the padded shard instead
        of the original shard plus a padded copy.
        """
        head = self.lm_head
        weight = getattr(head, "weight", None)
        if not (
            isinstance(head, ParallelLMHead)
            and weight is not None
            and isinstance(weight, torch.Tensor)
            and weight.dim() == 2
            and weight.dtype == torch.bfloat16
            and batch_invariant_enabled(weight)
        ):
            return None
        head_any: Any = head
        if getattr(head, "_muse_det_prepared", False):
            head_w = head.weight
            assert isinstance(head_w, torch.Tensor)
            return head_w
        if torch.compiler.is_compiling():
            raise RuntimeError("Muse batch invariant lm_head built during tracing")
        head_any._muse_shard_vocab = int(weight.shape[0])
        with torch.no_grad():
            padded = pad_lm_head_weight(weight.data).contiguous()
            if padded is not weight.data:
                head.weight = nn.Parameter(padded, requires_grad=False)
                del weight
            head_w = head.weight
            assert isinstance(head_w, torch.Tensor)
            synchronize_tensors([head_w])
        head_any._muse_det_prepared = True
        qkv_shapes = tuple(
            sorted(
                {
                    (m.q_size, m.kv_size, m.head_dim)
                    for m in self.modules()
                    if isinstance(m, MuseSparkAttention)
                }
            )
        )
        prepare_det_collectives(
            rms_norm_eps=float(self.text_config.rms_norm_eps),
            post_norm_eps=float(self.text_config.post_norm_eps),
            head_dim=int(self.text_config.head_dim),
        )
        prepare_det_ops(
            muse_spark_norm_eps(self),
            qk_scales=(float(self.text_config.qk_scale_factor),),
            qkv_shapes=qkv_shapes,
        )
        return head_w

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        assert self.hf_to_vllm_mapper is not None
        loaded = load_muse_spark_text_weights(
            self,
            weights,
            mapper=self.hf_to_vllm_mapper,
            num_experts=self.text_config.num_local_experts,
            tie_word_embeddings=bool(self.text_config.tie_word_embeddings),
        )
        # vLLM's loader calls process_weights_after_loading() again after the
        # routed quant methods finalize w13/w2; MuseSparkMoE then drops the fused
        # op resolved here and rebuilds it on the processed tensors.
        self.process_weights_after_loading()
        return loaded
