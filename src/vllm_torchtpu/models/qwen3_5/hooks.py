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
command-line value always wins. Every default below but the kernel page
applies only on the engines it was measured on: Qwen3.5-397B-A17B on TPU7x,
in the roles and layouts of the validation entries (``validation/``). The
kernel page applies to either leg of a KV-transfer pair of that checkpoint
shape on any chip, because it only makes the two legs page alike. Every other
engine keeps main's behaviour.

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
- ``additional_config["tpu_kv_transfer"]["kernel_page_tokens"]``: 128, for a
  KV producer or consumer of the calibrated checkpoint shape with a
  seq-along-lane KV cache layout and no prefill context parallelism, so the
  prefill and decode legs of a pair page their seq-along-lane caches alike
  (see ``attention.kv_transfer_kernel_page``). A PCP prefill engine, an
  engine on another layout (whose page the transfer does not carry), an
  engine whose only connector offloads (``kv_both``, no peer) and other
  checkpoints keep vLLM's page choice, as on main.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from jax.experimental.pallas import tpu as pltpu
from vllm import envs as vllm_envs
from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu import additional_config_sections, envs, tpu_info
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.layers.adapter.attention import (
    KV_LAYOUT_BY_VLLM_LAYOUT,
    KV_TRANSFER_SECTION,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.qwen3_5 import settings

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


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
# The expert slots per rank (64 primary + 16 replicated) the budget was
# measured with; _expert_slot_kv_credit adjusts it for every slot fewer or more.
_CALIBRATED_EXPERT_SLOTS = 80
# The budget at those slots, reserving about 2.5 GiB of HBM for runtime
# activations and program-load scratch buffers.
_CALIBRATED_KV_BYTES = 33_500_000_000

# The batched-RPA kernel page both legs of a KV-transfer pair of the calibrated
# checkpoint run at: every block either leg sizes (2304 tokens at a TP1
# prefill, 640 at a TP8 decode) is a multiple of it.
_KV_TRANSFER_PAGE_TOKENS = 128


def _chip_family() -> str | None:
    """This host's TPU chip (the ``pltpu.ChipVersion`` value, "7x" for
    TPU7x), or None when its accelerator type is unset or unrecognized.
    Never claims the chips."""
    chip_version = tpu_info.find_chip_version()
    return None if chip_version is None else chip_version.value


def _has_calibrated_shape(model_config) -> bool:
    """Whether the checkpoint has the calibrated shape (Qwen3.5-397B-A17B's
    text config), whatever the device."""
    text = model_config.hf_text_config
    return all(getattr(text, k, None) == v for k, v in _CALIBRATED_SHAPE.items())


def _is_calibrated(model_config) -> bool:
    """Whether the defaults were measured for this checkpoint and device:
    the calibrated checkpoint shape on a TPU7x device. Any other shape, a
    smaller device or an unknown device keeps main's behaviour."""
    return _has_calibrated_shape(model_config) and _chip_family() == _CALIBRATED_CHIP


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
    places it: the measured budgets did (every TP8 decode leg of the measured
    runs logged that budget), and it leaves copies x 3 x hidden x
    moe_intermediate bytes (48 MiB for 4 copies at Qwen3.5-397B-A17B's shape)
    unspent where the drafter's layer is not placed."""
    # Imported here, not at the module top: every vLLM process imports this
    # module through the general plugin (register_models), and only the
    # measured decode layout's budget reads the placement.
    from vllm_torchtpu.layers.adapter import expert_placement

    text = model_config.hf_text_config
    ep = parallel_config.tensor_parallel_size * parallel_config.data_parallel_size
    unplaced = text.num_experts // ep
    placement = expert_placement.current()
    placed = unplaced if placement is None else placement.slots_per_rank
    slot_layers = 0 if placement is None else -placement.copies_per_rank
    for layer in range(text.num_hidden_layers):
        is_placed = placement is not None and layer in placement.layers
        slot_layers += _CALIBRATED_EXPERT_SLOTS - (placed if is_placed else unplaced)
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


# vLLM's legacy names for two KV cache layouts, which VLLM_KV_CACHE_LAYOUT
# accepts. vLLM's own parser (vllm.config.cache) is private, so the hook reads
# names itself; test_raiden_pd_kernel_page checks it against vLLM's parser for
# every name vLLM accepts.
_KV_LAYOUT_ALIASES = {"NHD": "LBNHC", "HND": "LBHNC"}


def _kv_layout(name: str) -> KVCacheLayout:
    """The KV cache layout a name gives, as vLLM reads it: a layout's own name
    or one of vLLM's legacy aliases."""
    try:
        return KVCacheLayout[_KV_LAYOUT_ALIASES.get(name, name)]
    except KeyError:
        valid = [m.name for m in KVCacheLayout] + list(_KV_LAYOUT_ALIASES)
        raise ValueError(
            f"Unknown KV cache layout {name!r}. Valid layouts: {valid}"
        ) from None


def _kv_layout_is_seq_along_lane(vllm_config: VllmConfig) -> bool:
    """Whether the engine's KV cache will be seq-along-lane, the only layout
    whose kernel page the KV transfer carries.

    The platform resolves the layout after this hook runs
    (``resolve_kv_cache_layout``): a layout already on the cache config,
    else ``VLLM_KV_CACHE_LAYOUT``, else the KV connector's required layout,
    else the attention backend's first one. The TPU connectors require no
    layout and every TPU attention backend lists a head-along-sublane layout
    first, so without an explicit layout the cache is head-along-sublane.
    vLLM refuses a ``VLLM_KV_CACHE_LAYOUT`` outside its choices when the
    variable is read.
    """
    name = vllm_config.cache_config.kv_cache_layout or vllm_envs.VLLM_KV_CACHE_LAYOUT
    if name is None:
        return False
    layout = _kv_layout(name)
    return (
        KV_LAYOUT_BY_VLLM_LAYOUT.get(layout)
        is batched_rpa_configs.KVLayout.SEQ_ALONG_LANE
    )


def _apply_kv_transfer_page(vllm_config: VllmConfig) -> None:
    # Only a producer and a consumer exchange pages with a peer. An engine whose
    # only connector offloads (kv_both) has none, and its page at 128 against
    # vLLM's choice was never measured. (vLLM's MultiConnector gives each child
    # a shallow copy of the engine config, so the offloading child of a prefill
    # engine still sees the section written for the engine.)
    transfer = vllm_config.kv_transfer_config
    if transfer is None or transfer.kv_role not in ("kv_producer", "kv_consumer"):
        return
    if vllm_config.parallel_config.prefill_context_parallel_size > 1:
        return
    if not _has_calibrated_shape(vllm_config.model_config):
        return
    # On another layout the page would not apply, and the platform would
    # warn about a setting the user never wrote.
    if not _kv_layout_is_seq_along_lane(vllm_config):
        return
    additional_config = cast("dict[str, Any]", vllm_config.additional_config)
    # An explicit section, valid or not, is the user's: the platform validates it.
    if KV_TRANSFER_SECTION in additional_config:
        return
    additional_config[KV_TRANSFER_SECTION] = {
        "kernel_page_tokens": _KV_TRANSFER_PAGE_TOKENS
    }


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


def is_measured_target(model_config) -> bool:
    """Whether this engine serves the checkpoint and device the Qwen3.5 PD
    stack's kernel levers were measured on: a Qwen3.5 architecture
    (``settings.is_qwen3_5``) with the calibrated checkpoint shape on TPU7x
    (``_is_calibrated``, Qwen3.5-397B-A17B). The kernel numerics below and the
    few-token MoE kernel (``moe_lowtok_scores.enabled``) apply only there.

    Any other Qwen3.5 checkpoint (main's CI model, Qwen3.5-35B-A3B, among
    them) or device keeps main's arithmetic and code: those levers change the
    last bits of the forward pass, which is not a free choice where they were
    not measured."""
    if not settings.is_qwen3_5(model_config):
        return False
    if _is_calibrated(model_config):
        return True
    logger.info_once(
        "Qwen3.5 kernel levers (reciprocal-scale quantize, tuned and streamed "
        "FP8 MoE grouped matmul, two-sort non-EP MoE routing, few-token MoE "
        "kernel) left at main's arithmetic and code: they were measured for "
        "Qwen3.5-397B-A17B on TPU7x only."
    )
    return False


def apply_kernel_numerics(model_config) -> bool:
    """Qwen3.5's kernel numerics and code paths where they were measured
    (``is_measured_target``), in every process that loads the model.

    The Qwen3.5 PD stack was tuned and evaluated with the quantize helpers'
    reciprocal scale (kernels/quantized_matmul/util.RECIPROCAL_SCALE), the
    grouped matmul's hand-streamed whole-expert slabs
    (kernels/megablox/gmm_v2.RHS_STREAM) and its tuned FP8 MoE tiling and
    rhs-scale hoist (gmm_v2.TUNED_FP8_MOE), all off by default, and the
    two-sort routing metadata of the non-EP grouped-matmul MoE where an
    argsort builds it (layers/core/fused_moe_gmm.NONEP_PERMUTE, "sorts" by
    default; same metadata, one sort fewer); select them before the model's
    kernels are traced. Other models keep main's arithmetic and code. Returns
    False without a model config and for any model other than the measured
    target: the patch registry marks a patch done by its target, not by
    model, so a later load of the measured target in the same process still
    turns them on. Returns True once they are on.
    """
    if not is_measured_target(model_config):
        return False
    # Imported where the rule applies, not at the module top: every vLLM
    # process imports this module through the general plugin
    # (register_models), and only the measured target needs these kernels.
    from vllm_torchtpu.kernels.megablox import gmm_v2
    from vllm_torchtpu.kernels.quantized_matmul import util
    from vllm_torchtpu.layers.core import fused_moe_gmm

    util.set_reciprocal_scale(True)
    gmm_v2.set_rhs_stream(True)
    gmm_v2.set_tuned_fp8_moe(True)
    fused_moe_gmm.set_nonep_permute("two_sorts")
    return True


def mtp_draft_path(vllm_config: VllmConfig) -> bool:
    """Qwen3.5's draft-path rule (spec_decode/draft_path_rules.py): a Qwen3.5
    target with an MTP draft runs greedy top tokens, fast token substitution
    and the fused MTP draft loop by default, and batch-1 async decode runs the
    runner's decode carry. Like every default in this file, only for the
    checkpoint shape and device they were validated on (_is_calibrated); any
    other Qwen3.5 engine keeps main's opt-in keys. Read once at engine and
    drafter start, not per step."""
    spec = vllm_config.speculative_config
    model_config = vllm_config.model_config
    if (
        spec is None
        or spec.method != "mtp"
        or not settings.is_qwen3_5(model_config)
        or not _has_calibrated_shape(model_config)
    ):
        return False
    chip = _chip_family()
    if chip is None:
        logger.warning_once(
            "Qwen3.5 MTP: this checkpoint's draft-path defaults were validated "
            "on TPU7x, but this host's TPU chip is unknown (is "
            "TPU_ACCELERATOR_TYPE set?), so they stay off."
        )
    return chip == _CALIBRATED_CHIP


def apply_serving_defaults(vllm_config: VllmConfig) -> None:
    """Fill the Qwen3.5 TPU serving defaults the user did not set."""
    calibrated = _is_calibrated(vllm_config.model_config)
    _apply_prefix_match_unit(vllm_config, calibrated)
    _apply_kv_budget(vllm_config, calibrated)
    _apply_kv_transfer_page(vllm_config)


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
