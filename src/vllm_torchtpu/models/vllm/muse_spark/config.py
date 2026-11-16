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
"""Lightweight ``MODELS_CONFIG_MAP`` hook for Muse Spark Open on TPU.

Kept free of heavy model/kernel imports so ``register_models()`` remains lazy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


def normalize_muse_spark_hf_quant_config(hf_config: Any) -> None:
    """Union ModelOpt ``exclude_modules`` and ``modules_to_not_convert``.

    Upstream ``ModelOptNvFp4Config`` reads ``config.get("ignore",
    config.get("exclude_modules", []))`` and ignores ``modules_to_not_convert``
    when ``exclude_modules`` is also present.
    """
    if hf_config is None:
        return
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


class TPUMuseSparkConfig(VerifyAndUpdateConfig):
    """Validate parallelism/runtime flags and apply TPU defaults for Muse Spark."""

    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None:
        normalize_muse_spark_hf_quant_config(model_config.hf_config)
        normalize_muse_spark_hf_quant_config(
            getattr(model_config, "model_arch_config", None)
        )

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        import vllm_torchtpu.envs as envs
        from vllm_torchtpu.logger import init_logger

        logger = init_logger(__name__)
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

        pc = vllm_config.parallel_config
        if pc.data_parallel_size > 1:
            raise NotImplementedError(
                "Muse Spark on TPU requires data_parallel_size=1: vLLM flattens "
                "expert parallelism across DP x TP while the MoE combine reduces "
                "over TP only."
            )
        if pc.prefill_context_parallel_size > 1 or pc.decode_context_parallel_size > 1:
            raise NotImplementedError(
                "Muse Spark on TPU does not support context parallelism (PCP/DCP > 1)."
            )
        if pc.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Muse Spark on TPU requires pipeline_parallel_size=1."
            )
        if pc.tensor_parallel_size > 1 and not pc.enable_expert_parallel:
            raise NotImplementedError(
                "Muse Spark on TPU requires --enable-expert-parallel when "
                "tensor_parallel_size > 1: the nonlinear per-expert post-norm "
                "requires each expert's full output before combining."
            )
        if pc.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Muse Spark on TPU does not support sequence-parallel MoE."
            )
        if pc.enable_eplb:
            raise NotImplementedError(
                "Muse Spark on TPU does not support EPLB expert rebalancing."
            )
        if vllm_config.lora_config is not None:
            raise NotImplementedError("Muse Spark on TPU does not support LoRA.")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Muse Spark speculative decoding on TPU lands in the DSpark "
                "follow-up PR."
            )
        if cache_config is not None and cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "bf16",
        ):
            raise NotImplementedError(
                f"Muse Spark on TPU currently supports auto/bfloat16 KV cache "
                f"(got cache_dtype={cache_config.cache_dtype!r})."
            )
        if bool(envs.TPU_EXPERT_PLACEMENT.strip()):
            raise NotImplementedError(
                "Muse Spark on TPU does not support TPU_EXPERT_PLACEMENT: the "
                "fused post-norm MoE op routes directly from router_w_kmajor."
            )
