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
Qwen3.5 checks, validates ``additional_config["qwen3_5"]``, refuses a
``qwen3_5`` or ``tpu_*`` section of ``additional_config`` that no reader in
this build claims (``additional_config_sections``), and applies the TPU
defaults below. Engines of other architectures get none of this.

Each default fills a value only when the user left it unset, so an explicit
command-line value always wins, and only on the engines it was measured on:
Qwen3.5-397B-A17B on TPU7x, in the roles and layouts of the validation
entries (``validation/``). Every other engine keeps main's behaviour.

- ``kv_cache_memory_bytes``: a calibrated budget on a decode
  (``kv_consumer``) leg at TP8 + DCP4 with expert parallelism, no pipeline
  or prefill context parallelism, and the fused EP MoE v3 kernel
  (``MOE_FUSED_EP_KERNEL_IMPL=v3``): the entries' TP8 decode legs. Every
  other engine stays on ``gpu_memory_utilization``, among them a leg with
  fused EP v2 or the grouped-matmul MoE (their activations were never
  measured beside this budget), the DP4 x TP2 decode layout (its entry passes
  ``--kv-cache-memory-bytes``) and main's nightly TP1 x DP8 configs.
- ``prefix_match_unit``: 128 tokens on a prefill (``kv_producer``) leg
  without prefill or decode context parallelism (the entries' PP8 prefill
  legs). Every other engine keeps the unit unset, so a PCP prefill leg
  resolves its hash block size from its KV cache groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax.experimental.pallas import tpu as pltpu
from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig

from vllm_torchtpu import additional_config_sections, envs, tpu_info
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.qwen3_5 import settings

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# Baseline expert-slot count per EP rank (64 primary + 16 replicated slots)
# used as the reference point when computing per-rank HBM KV capacity credit.
_QWEN35_SAFE_KV_SLOTS = 80

# The defaults were measured for this checkpoint shape (Qwen3.5-397B-A17B's
# text config) on TPU7x devices.
_CALIBRATED_SHAPE = {"num_experts": 512, "hidden_size": 4096, "num_hidden_layers": 60}
_CALIBRATED_CHIP = pltpu.ChipVersion.TPU_7X.value
# The decode layouts (tensor, data, decode context parallel size) whose legs
# ran on the KV budget, with expert parallelism over their 8 ranks: the
# validation entries' TP8 + DCP4 decode legs.
_CALIBRATED_DECODE_LAYOUTS = frozenset({(8, 1, 4)})
# The MoE kernel those legs ran (MOE_FUSED_EP_KERNEL_IMPL): its scratch is
# part of the HBM the budget leaves for activations.
_CALIBRATED_MOE_KERNEL = "v3"
# The budget at 80 expert slots per rank (64 primary + 16 replicated slots),
# reserving about 2.5 GiB of HBM for runtime activations and program-load
# scratch buffers.
_CALIBRATED_KV_BYTES = 33_500_000_000


def _chip_family() -> str | None:
    """This host's TPU chip (the ``pltpu.ChipVersion`` value, "7x" for
    TPU7x), or None when its accelerator type is unset or unrecognized.
    Never claims the chips."""
    chip_version = tpu_info.find_chip_version()
    return None if chip_version is None else chip_version.value


def _is_calibrated(model_config) -> bool:
    """Whether the defaults were measured for this checkpoint and device:
    the calibrated checkpoint shape on a TPU7x device. Any other shape, a
    smaller device or an unknown device keeps main's behaviour."""
    text = getattr(model_config, "hf_text_config", None)
    for key, value in _CALIBRATED_SHAPE.items():
        if getattr(text, key, None) != value:
            return False
    return _chip_family() == _CALIBRATED_CHIP


def _expert_slot_kv_credit(model_config, parallel_config) -> tuple[int, int]:
    """(credit, slots): HBM bytes per device freed (positive) or taken
    (negative) by the expert placement holding fewer (more) slots per rank
    than the placement the Qwen3.5 safe_kv budgets were measured with, one FP8
    expert (gate|up and down, 3 x hidden x moe_intermediate bytes) per slot
    per MoE layer of the target; and a placed layer's slot count (for
    logging).

    Every target layer holds num_experts / EP slots per rank, or the slots
    per rank of a placement table (TPU_EXPERT_PLACEMENT) that places it, its
    primaries and copy slots; table entries past the target's layers (a
    drafter's) are not target layers. A table with copy slots also budgets
    the drafter's MoE layer for its copy slots, whether or not the table
    places it: the measured budgets did (every TP8 decode leg of the record
    runs logged that budget), and it leaves copies x 3 x hidden x
    moe_intermediate bytes (48 MiB for 4 copies at Qwen3.5-397B-A17B's shape)
    unspent where the drafter's layer is not placed."""
    from vllm_torchtpu.layers.adapter import expert_placement

    text = model_config.hf_text_config
    ep = parallel_config.tensor_parallel_size * parallel_config.data_parallel_size
    unplaced = text.num_experts // ep
    placement = expert_placement.current()
    placed = unplaced if placement is None else placement.slots_per_rank
    slot_layers = 0 if placement is None else -placement.copies_per_rank
    for layer in range(text.num_hidden_layers):
        is_placed = placement is not None and layer in placement.layers
        slot_layers += _QWEN35_SAFE_KV_SLOTS - (placed if is_placed else unplaced)
    slot_bytes = 3 * text.hidden_size * text.moe_intermediate_size
    return slot_layers * slot_bytes, placed


def _apply_prefix_match_unit(vllm_config: VllmConfig, calibrated: bool) -> None:
    # Only a prefill (kv_producer) leg: its prefix hits can then land on any
    # 128-token boundary inside its large blocks (2176 tokens on the
    # validation entries' prefill legs). A decode leg and a mixed engine keep
    # vLLM's default.
    kv_cfg = vllm_config.kv_transfer_config
    if kv_cfg is None or kv_cfg.kv_role != "kv_producer":
        return
    cache_config = vllm_config.cache_config
    if cache_config.prefix_match_unit is not None:
        return
    # The measured prefill legs ran PP8 without context parallelism. With the
    # unit unset, a PCP or DCP leg resolves its hash block size from its KV
    # cache groups (the platform's PCP and DCP block-size patches).
    parallel_config = vllm_config.parallel_config
    pcp = parallel_config.prefill_context_parallel_size
    dcp = parallel_config.decode_context_parallel_size
    if pcp != 1 or dcp != 1 or not calibrated:
        logger.info(
            "Qwen3.5 prefix_match_unit left unset on this prefill leg: the "
            "128-token unit was measured for Qwen3.5-397B-A17B on TPU7x "
            "without context parallelism (this leg: calibrated checkpoint and "
            "device=%s, pcp=%d, dcp=%d).",
            calibrated,
            pcp,
            dcp,
        )
        return
    cache_config.prefix_match_unit = 128


def _moe_kernel() -> str | None:
    """The fused EP MoE implementation this process runs (``v2`` when the
    switch is on without an implementation), or None without fused EP."""
    if not envs.USE_MOE_FUSED_EP_KERNEL:
        return None
    return envs.MOE_FUSED_EP_KERNEL_IMPL or "v2"


def _is_measured_kv_layout(vllm_config: VllmConfig) -> bool:
    """Whether the engine's role, layout and MoE kernel are those the KV
    budget was measured on: a decode (kv_consumer) leg at TP8 + DCP4 with
    expert parallelism, no pipeline or prefill context parallelism, and fused
    EP MoE v3. There the expert-slot credit counts each rank's slots and
    layers exactly; PP splits the layers across stages and vLLM folds PCP into
    the EP size."""
    kv_cfg = vllm_config.kv_transfer_config
    parallel_config = vllm_config.parallel_config
    layout = (
        parallel_config.tensor_parallel_size,
        parallel_config.data_parallel_size,
        parallel_config.decode_context_parallel_size,
    )
    return (
        kv_cfg is not None
        and kv_cfg.kv_role == "kv_consumer"
        and layout in _CALIBRATED_DECODE_LAYOUTS
        and parallel_config.enable_expert_parallel
        and parallel_config.pipeline_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
        and _moe_kernel() == _CALIBRATED_MOE_KERNEL
    )


def _apply_kv_budget(vllm_config: VllmConfig, calibrated: bool) -> None:
    cache_config = vllm_config.cache_config
    if cache_config.kv_cache_memory_bytes is not None:
        return
    parallel_config = vllm_config.parallel_config
    if not calibrated or not _is_measured_kv_layout(vllm_config):
        kv_cfg = vllm_config.kv_transfer_config
        logger.info(
            "Qwen3.5 kv_cache_memory_bytes left to gpu_memory_utilization: the "
            "calibrated budget was measured for Qwen3.5-397B-A17B decode "
            "(kv_consumer) legs on TPU7x at TP8 + DCP4 with expert "
            "parallelism, no PP or PCP and fused EP MoE v3 (this engine: "
            "calibrated checkpoint and device=%s, kv_role=%s, tp=%d, dp=%d, "
            "dcp=%d, expert_parallel=%s, pp=%d, pcp=%d, fused_ep=%s).",
            calibrated,
            None if kv_cfg is None else kv_cfg.kv_role,
            parallel_config.tensor_parallel_size,
            parallel_config.data_parallel_size,
            parallel_config.decode_context_parallel_size,
            parallel_config.enable_expert_parallel,
            parallel_config.pipeline_parallel_size,
            parallel_config.prefill_context_parallel_size,
            _moe_kernel(),
        )
        return
    credit, slots = _expert_slot_kv_credit(vllm_config.model_config, parallel_config)
    safe_kv = _CALIBRATED_KV_BYTES + credit
    logger.info(
        "Qwen3.5-397B kv_cache_memory_bytes set to auto: %d (slots=%d, dcp=%d, "
        "seqs=%d).",
        safe_kv,
        slots,
        parallel_config.decode_context_parallel_size,
        vllm_config.scheduler_config.max_num_seqs,
    )
    cache_config.kv_cache_memory_bytes = safe_kv


def apply_serving_defaults(vllm_config: VllmConfig) -> None:
    """Fill the Qwen3.5 TPU serving defaults the user did not set."""
    calibrated = _is_calibrated(vllm_config.model_config)
    _apply_prefix_match_unit(vllm_config, calibrated)
    _apply_kv_budget(vllm_config, calibrated)


class TPUQwen35MoeConfig(Qwen3_5ForConditionalGenerationConfig):
    """vLLM's Qwen3.5 checks, the Qwen3.5 settings, and the TPU defaults."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        # Fails at engine start on a qwen3_5 or tpu_* section this build does
        # not read, so a Qwen3.5 deployment never runs silently without it.
        additional_config_sections.check(vllm_config.additional_config)
        # Fails at engine start on an unknown setting or a wrongly typed value.
        settings.get(vllm_config)
        apply_serving_defaults(vllm_config)
