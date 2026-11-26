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
"""Config checks for the TPU Qwen4Exp text model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vllm.model_executor.models.config import Qwen4ExpForCausalLMConfig

from vllm_torchtpu.block_major_pool import unified_block_major_enabled
from vllm_torchtpu.gdn_pool_layout import unified_kv_layout_enabled_for_architecture
from vllm_torchtpu.models.qwen4_exp.qsa import dense_max_model_len

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _cfg_get(cfg: object, key: str) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key)
    return getattr(cfg, key, None)


class TPUQwen4ExpConfig(Qwen4ExpForCausalLMConfig):
    """Upstream Qwen4Exp checks, plus the features the TPU model lacks yet."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Qwen4ExpForCausalLMConfig.verify_and_update_config(vllm_config)
        model_config = vllm_config.model_config
        text_config = model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        spec_config = vllm_config.speculative_config
        # Each PLE layer keeps its state in a unified-pool block of its GDN
        # layer's spec (see Qwen4ExpPLELayer; transformers admits PLE only on
        # linear_attention layers).
        has_ple = bool(text_config.ple_layer_ids)
        has_sparse_qsa = (
            text_config.indexer_budget is not None
            and text_config.indexer_compress_ratio is not None
            and model_config.max_model_len > dense_max_model_len(text_config)
            and (
                "indexed_attention" in (text_config.layer_types or ())
                or spec_config is not None
            )
        )
        if has_ple and not unified_kv_layout_enabled_for_architecture(
            model_config.architecture
        ):
            raise NotImplementedError(
                "Qwen4Exp n-gram PLE keeps its state in the unified KV pool; "
                "unset TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0."
            )
        if (has_ple or has_sparse_qsa) and unified_block_major_enabled(vllm_config):
            raise NotImplementedError(
                "Qwen4Exp PLE/QSA does not support block-major unified KV pools; "
                "unset VLLM_TPU_BLOCK_MAJOR_KV."
            )
        if has_ple and vllm_config.kv_transfer_config is not None:
            raise NotImplementedError(
                "Qwen4Exp n-gram PLE does not support KV transfer yet."
            )
        mamba_cache_mode = vllm_config.cache_config.mamba_cache_mode
        if has_ple and mamba_cache_mode not in (None, "none", "align"):
            raise NotImplementedError(
                "Qwen4Exp n-gram PLE does not support "
                f"mamba_cache_mode={mamba_cache_mode!r}."
            )
        # transformers (>=5.18, see pyproject.toml) labels the checkpoint's
        # full_attention (QSA) layers indexed_attention.
        unsupported_layer_types = sorted(
            set(text_config.layer_types or ())
            - {"linear_attention", "indexed_attention"}
        )
        if unsupported_layer_types:
            raise NotImplementedError(
                f"Qwen4Exp layer types {unsupported_layer_types} are not "
                "supported on TPU yet."
            )
        if spec_config is not None:
            if spec_config.method != "mtp":
                raise NotImplementedError(
                    "Qwen4Exp on TPU supports only method='mtp' speculative "
                    f"decoding, got {spec_config.method!r}."
                )
            draft_tp = spec_config.draft_tensor_parallel_size
            if (
                draft_tp is not None
                and draft_tp != parallel_config.tensor_parallel_size
            ):
                raise NotImplementedError(
                    "Qwen4Exp MTP on TPU requires draft_tensor_parallel_size "
                    "== tensor_parallel_size "
                    f"({parallel_config.tensor_parallel_size}), got {draft_tp}."
                )
            draft_model_config = spec_config.draft_model_config
            draft_hf_config = (
                draft_model_config.hf_config if draft_model_config is not None else None
            )
            draft_text_config = (
                getattr(draft_model_config, "hf_text_config", None) or text_config
            )
            if draft_model_config is not None and getattr(
                draft_model_config, "quantization", None
            ) != getattr(model_config, "quantization", None):
                raise NotImplementedError(
                    "Qwen4Exp MTP on TPU requires draft quantization "
                    f"({getattr(draft_model_config, 'quantization', None)!r}) "
                    "to match target quantization "
                    f"({getattr(model_config, 'quantization', None)!r})."
                )
            mtp_cfg = getattr(draft_text_config, "mtp", None)
            layer_counts = [
                int(v)
                for v in (
                    getattr(draft_text_config, "mtp_num_hidden_layers", None),
                    getattr(draft_text_config, "num_nextn_predict_layers", None),
                    _cfg_get(mtp_cfg, "num_hidden_layers")
                    if mtp_cfg is not None
                    else None,
                )
                if v is not None
            ]
            if not layer_counts or any(n != 1 for n in layer_counts):
                raise NotImplementedError(
                    "Qwen4Exp MTP on TPU supports only 1 MTP hidden layer, "
                    f"got {layer_counts or [0]}."
                )
            if getattr(draft_text_config, "mtp_use_dedicated_embeddings", False):
                raise NotImplementedError(
                    "Qwen4Exp MTP on TPU does not support "
                    "mtp_use_dedicated_embeddings=True."
                )
            if (
                getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
                or getattr(draft_text_config, "index_share_for_mtp_iteration", False)
                or getattr(text_config, "index_share_for_mtp_iteration", False)
                or (
                    mtp_cfg is not None
                    and bool(_cfg_get(mtp_cfg, "index_share_for_mtp_iteration"))
                )
            ):
                raise NotImplementedError(
                    "Qwen4Exp MTP on TPU does not support "
                    "index_share_for_mtp_iteration=True yet."
                )
            if mtp_cfg is not None:
                mtp_layer_types_seq = tuple(_cfg_get(mtp_cfg, "layer_types") or ())
                if len(mtp_layer_types_seq) > 1:
                    raise NotImplementedError(
                        "Qwen4Exp MTP on TPU supports only 1 MTP layer, got "
                        f"layer_types={list(mtp_layer_types_seq)}."
                    )
                bad_mtp_types = sorted(
                    set(mtp_layer_types_seq) - {"full_attention", "indexed_attention"}
                )
                if bad_mtp_types:
                    raise NotImplementedError(
                        f"Qwen4Exp MTP layer types {bad_mtp_types} are not "
                        "supported on TPU yet."
                    )
                if _cfg_get(mtp_cfg, "mtp_use_hidden_state_from_layer") is not None:
                    raise NotImplementedError(
                        "Qwen4Exp MTP on TPU requires "
                        "mtp_use_hidden_state_from_layer=None."
                    )
                mtp_rope = _cfg_get(mtp_cfg, "rope_theta")
                rope_params = getattr(text_config, "rope_parameters", None)
                target_rope = (
                    _cfg_get(rope_params, "rope_theta")
                    if rope_params is not None
                    else getattr(text_config, "rope_theta", None)
                )
                if (
                    mtp_rope is not None
                    and target_rope is not None
                    and float(mtp_rope) != float(target_rope)
                ):
                    raise NotImplementedError(
                        f"Qwen4Exp MTP rope_theta ({mtp_rope}) must match "
                        f"target rope_theta ({target_rope})."
                    )
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Qwen4Exp on TPU requires pipeline_parallel_size=1."
            )
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp hyper-connections do not support sequence-parallel "
                "MoE; unset TPU_MOE_SEQUENCE_PARALLEL."
            )
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp on TPU does not support context parallelism yet."
            )
