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
"""The Qwen3.5 config hook: TPU serving defaults for Qwen3.5 engines.

``TPUQwen35MoeConfig`` is registered in vLLM's ``MODELS_CONFIG_MAP`` for the
architectures in ``settings.ARCHITECTURES``, so vLLM calls it from
``VllmConfig.__post_init__`` (before the platform's
``check_and_update_config``) for those models only. It keeps vLLM's own
Qwen3.5 checks, validates ``additional_config["qwen3_5"]``, and applies
the TPU defaults below. Each default fills a value only when the user left it
unset, so an explicit command-line value always wins.

- ``kv_cache_memory_bytes``: a shape-calibrated budget, except on a
  ``kv_producer`` (prefill) leg, which stays on ``gpu_memory_utilization``.
- ``prefix_match_unit``: 128 tokens on a ``kv_producer`` leg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.qwen3_5 import settings

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# Baseline expert-slot count per EP rank (64 primary + 16 replicated slots)
# used as the reference point when computing per-rank HBM KV capacity credit.
_QWEN35_SAFE_KV_SLOTS = 80


def _expert_slots_and_layers(model_config, parallel_config) -> tuple[int, int]:
    """(slots, layers): the expert slots each EP rank holds and the MoE layers
    that hold them. Main's placement (TPU_EXPERT_PLACEMENT) only permutes
    experts, so every rank holds num_experts / EP of them in every layer."""
    text = model_config.hf_text_config
    ep = int(parallel_config.tensor_parallel_size) * int(
        parallel_config.data_parallel_size
    )
    return int(text.num_experts) // max(ep, 1), int(text.num_hidden_layers)


def _expert_slot_kv_credit(model_config, parallel_config) -> tuple[int, object]:
    """(credit, slots): HBM bytes per device freed (positive) or taken
    (negative) by the expert placement holding fewer (more) slots per rank
    than the placement the Qwen3.5 safe_kv budgets were measured with: one FP8
    expert (gate|up and down, 3 x hidden x moe_intermediate bytes) per slot
    per placed MoE layer; and the slot count (for logging)."""
    try:
        text = model_config.hf_text_config
        hidden = int(text.hidden_size)
        inter = int(text.moe_intermediate_size)
        slots, layers = _expert_slots_and_layers(model_config, parallel_config)
    except Exception as e:  # noqa: BLE001 - never block startup on the credit
        logger.warning("expert-slot KV credit unavailable (%s); using 0.", e)
        return 0, "?"
    return (_QWEN35_SAFE_KV_SLOTS - slots) * layers * 3 * hidden * inter, slots


def _apply_prefix_match_unit(vllm_config: VllmConfig) -> None:
    # Only a prefill (kv_producer) leg: its prefix hits can then land on any
    # 128-token boundary inside its large blocks (2176 tokens at the IX
    # points). A decode leg and a mixed engine keep vLLM's default.
    kv_cfg = vllm_config.kv_transfer_config
    if kv_cfg is None or kv_cfg.kv_role != "kv_producer":
        return
    cache_config = vllm_config.cache_config
    if cache_config.prefix_match_unit is None:
        cache_config.prefix_match_unit = 128


def _apply_kv_budget(vllm_config: VllmConfig) -> None:
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    parallel_config = vllm_config.parallel_config
    dcp_size = int(getattr(parallel_config, "decode_context_parallel_size", 1) or 1)
    # Calibrated for 80 expert slots per rank (64 primary + 16 replicated slots)
    # while reserving ~2.5 GiB of HBM headroom for runtime activations
    # and program-load scratch buffers under DCP=1 and DCP>1.
    if dcp_size == 1 and scheduler_config.max_num_seqs > 16:
        safe_kv = 32_500_000_000
    else:
        safe_kv = 33_500_000_000
    credit, slots = _expert_slot_kv_credit(vllm_config.model_config, parallel_config)
    safe_kv += credit
    # Apply the shape-calibrated HBM KV memory budget unless the caller
    # passed --kv-cache-memory-bytes.
    requested = cache_config.kv_cache_memory_bytes
    keep = requested is not None
    # The budget is calibrated for a mixed engine. A disagg prefill leg
    # (PCP shards, prefill-sized activations) OOMs at it, so leave that
    # leg on gpu_memory_utilization.
    kv_cfg = vllm_config.kv_transfer_config
    if kv_cfg is not None and kv_cfg.kv_role == "kv_producer":
        logger.info(
            "Qwen3.5-397B kv_cache_memory_bytes left to the prefill "
            "leg (requested %s; mixed-engine auto would be %d).",
            requested,
            safe_kv,
        )
        keep = True
    elif keep or requested != safe_kv:
        logger.warning(
            "Qwen3.5-397B kv_cache_memory_bytes %s: requested %s, "
            "auto %d (slots=%s, dcp=%d, seqs=%d).",
            "kept (explicit)" if keep else "set to auto",
            requested,
            safe_kv,
            slots,
            dcp_size,
            scheduler_config.max_num_seqs,
        )
    if not keep:
        cache_config.kv_cache_memory_bytes = safe_kv


def apply_serving_defaults(vllm_config: VllmConfig) -> None:
    """Fill the Qwen3.5 TPU serving defaults the user did not set."""
    _apply_prefix_match_unit(vllm_config)
    _apply_kv_budget(vllm_config)


class TPUQwen35MoeConfig(Qwen3_5ForConditionalGenerationConfig):
    """vLLM's Qwen3.5 checks, the Qwen3.5 settings, and the TPU defaults."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        # Fails at engine start on an unknown setting or a wrongly typed value.
        settings.get(vllm_config)
        apply_serving_defaults(vllm_config)
