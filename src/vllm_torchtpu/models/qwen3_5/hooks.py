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

- ``prefix_match_unit``: 128 tokens on a prefill (``kv_producer``) leg
  without prefill or decode context parallelism (the entries' PP8 prefill
  legs). Every other engine keeps the unit unset, so a PCP prefill leg
  resolves its hash block size from its KV cache groups.
- ``additional_config["tpu_kv_transfer"]["kernel_page_tokens"]``: 128, for an
  engine of the calibrated checkpoint shape with a KV transfer config and
  without prefill context parallelism, so the prefill and decode legs of a
  pair page their seq-along-lane caches alike (see
  ``attention.kv_transfer_kernel_page``). A PCP prefill engine and other
  checkpoints keep vLLM's page choice, as on main.
- ``tpu_pp_scheduler.hybrid_state_retention``: true on a prefill
  (``kv_producer``) leg with more than one pipeline stage, without context
  parallelism, whose pipeline chunk scheduler the platform installs (the
  entries' PP8 prefill legs). Every other engine keeps it false.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from jax.experimental.pallas import tpu as pltpu
from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig

from vllm_torchtpu import additional_config_sections, tpu_info
from vllm_torchtpu.core.scheduler_utils import kv_roles
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.qwen3_5 import settings

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# The defaults were measured for this checkpoint shape (Qwen3.5-397B-A17B's
# text config) on TPU7x devices.
_CALIBRATED_SHAPE = {"num_experts": 512, "hidden_size": 4096, "num_hidden_layers": 60}
_CALIBRATED_CHIP = pltpu.ChipVersion.TPU_7X.value

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


def has_calibrated_shape(model_config) -> bool:
    """Whether the checkpoint has the calibrated shape (Qwen3.5-397B-A17B's
    text config), whatever the device."""
    text = getattr(model_config, "hf_text_config", None)
    return all(getattr(text, k, None) == v for k, v in _CALIBRATED_SHAPE.items())


def _is_calibrated(model_config) -> bool:
    """Whether the defaults were measured for this checkpoint and device:
    the calibrated checkpoint shape on a TPU7x device. Any other shape, a
    smaller device or an unknown device keeps main's behaviour."""
    text = getattr(model_config, "hf_text_config", None)
    for key, value in _CALIBRATED_SHAPE.items():
        if getattr(text, key, None) != value:
            return False
    return _chip_family() == _CALIBRATED_CHIP


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


def _apply_kv_transfer_page(vllm_config: VllmConfig) -> None:
    if vllm_config.kv_transfer_config is None:
        return
    if vllm_config.parallel_config.prefill_context_parallel_size > 1:
        return
    if not has_calibrated_shape(vllm_config.model_config):
        return
    from vllm_torchtpu.layers.adapter.attention import KV_TRANSFER_SECTION

    additional_config = cast("dict[str, Any]", vllm_config.additional_config)
    # An explicit section, valid or not, is the user's: the platform validates it.
    if KV_TRANSFER_SECTION in additional_config:
        return
    additional_config[KV_TRANSFER_SECTION] = {
        "kernel_page_tokens": _KV_TRANSFER_PAGE_TOKENS
    }


def _apply_hybrid_state_retention(vllm_config: VllmConfig, calibrated: bool) -> None:
    # Only the measured prefill legs: a kv_producer pipeline without context
    # parallelism that gets the pipeline chunk scheduler, which keeps the
    # snapshots. Every other engine keeps the option's default, false.
    if not kv_roles(vllm_config.kv_transfer_config).producer or not calibrated:
        return
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.pipeline_parallel_size <= 1
        or parallel_config.prefill_context_parallel_size != 1
        or parallel_config.decode_context_parallel_size != 1
    ):
        return
    # Lazy: the scheduler modules import vLLM's scheduler.
    from vllm_torchtpu.core import pp_chunk_scheduler, pp_scheduler_options

    if pp_chunk_scheduler.pipeline_chunks_off_reason(vllm_config) is not None:
        return
    additional = vllm_config.additional_config
    if not isinstance(additional, dict):
        return
    section = additional.get(pp_scheduler_options.SECTION, {})
    # An explicit value wins; a malformed section fails in the parser.
    if not isinstance(section, dict) or "hybrid_state_retention" in section:
        return
    additional[pp_scheduler_options.SECTION] = {
        **section,
        "hybrid_state_retention": True,
    }


def apply_serving_defaults(vllm_config: VllmConfig) -> None:
    """Fill the Qwen3.5 TPU serving defaults the user did not set."""
    calibrated = _is_calibrated(vllm_config.model_config)
    _apply_prefix_match_unit(vllm_config, calibrated)
    _apply_kv_transfer_page(vllm_config)
    _apply_hybrid_state_retention(vllm_config, calibrated)


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
