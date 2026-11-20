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
"""Lightweight ``MODELS_CONFIG_MAP`` hook for Muse Spark Open + DSpark on TPU.

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


def _verify_and_repair_dspark_config(vllm_config: VllmConfig) -> None:
    """Validate DSpark speculative config and repair unlisted draft configs."""
    import json
    from pathlib import Path

    from vllm.transformers_utils.configs.muse_spark import MuseSparkDSparkConfig

    spec_config = vllm_config.speculative_config
    if spec_config is None:
        return
    if spec_config.method != "dspark":
        raise NotImplementedError(
            "Muse Spark on TPU only supports speculative_config.method='dspark' "
            f"(got {spec_config.method!r})."
        )
    pc = vllm_config.parallel_config
    dtp = spec_config.draft_tensor_parallel_size
    if dtp not in (None, pc.tensor_parallel_size):
        raise NotImplementedError(
            "Muse Spark DSpark on TPU requires draft_tensor_parallel_size to "
            f"match target tensor_parallel_size (got draft_tp={dtp}, "
            f"target_tp={pc.tensor_parallel_size})."
        )

    draft_mc = spec_config.draft_model_config
    if draft_mc is None:
        return
    draft_hf = draft_mc.hf_config
    had_own_quant = bool(getattr(draft_hf, "quantization_config", None))
    if not isinstance(draft_hf, MuseSparkDSparkConfig):
        # Prefer the raw config.json when available so an unrelated AutoConfig
        # class (e.g. Qwen3Config) does not strip or default Muse fields such
        # as sliding_window or residual_dtype before re-parsing.
        raw: dict[str, Any] | None = None
        local_cfg = Path(str(draft_mc.model)) / "config.json"
        if local_cfg.is_file():
            raw = json.loads(local_cfg.read_text(encoding="utf-8"))
        if raw is None:
            raw = (
                draft_hf.to_dict()
                if hasattr(draft_hf, "to_dict")
                else dict(vars(draft_hf))
            )
        had_own_quant = bool(raw.get("quantization_config"))
        for drop_key in ("model_type", "architectures"):
            raw.pop(drop_key, None)
        if not had_own_quant:
            raw.pop("quantization_config", None)
        draft_hf = MuseSparkDSparkConfig(**raw)
        draft_mc.hf_config = draft_hf

    if not had_own_quant:
        # The draft ModelConfig is built with speculative_config.quantization
        # (the user's draft weight quantization), but upstream may overwrite
        # draft_model_config.quantization with the target's. Restore the
        # user's value and only drop quantization the user did not request.
        draft_mc.quantization = spec_config.quantization  # type: ignore[assignment]
        if not spec_config.quantization and hasattr(draft_hf, "quantization_config"):
            draft_hf.quantization_config = None

    if getattr(draft_hf, "global_attn_cfg", None) is None and getattr(
        draft_hf, "sliding_window", None
    ):
        draft_hf.global_attn_cfg = [int(draft_hf.sliding_window)]

    draft_hf.architectures = ["MuseSparkDSparkModel"]
    spec_config.update_arch_()
    if getattr(spec_config, "num_speculative_tokens", None) is None:
        block = getattr(draft_hf, "block_size", None) or getattr(
            draft_hf, "dspark_block_size", None
        )
        if block is not None:
            spec_config.num_speculative_tokens = int(block)


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
        _verify_and_repair_dspark_config(vllm_config)
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
        mm_cfg = getattr(vllm_config.model_config, "multimodal_config", None)
        if mm_cfg is not None and getattr(mm_cfg, "mm_encoder_tp_mode", None) == "data":
            raise NotImplementedError(
                "Muse Spark on TPU does not support mm_encoder_tp_mode='data'; "
                "use the default mm_encoder_tp_mode='weights'."
            )
        sc = getattr(vllm_config, "scheduler_config", None)
        if sc is not None and getattr(sc, "disable_chunked_mm_input", False):
            raise NotImplementedError(
                "Muse Spark on TPU does not support --disable-chunked-mm-input."
            )
