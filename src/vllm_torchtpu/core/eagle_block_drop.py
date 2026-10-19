# SPDX-License-Identifier: Apache-2.0
"""Configures upstream EAGLE prefix-cache block retention."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_torchtpu import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def apply_eagle_block_drop_override(vllm_config: VllmConfig) -> None:
    """Propagates VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP into SpeculativeConfig."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None and envs.VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP:
        speculative_config.disable_eagle_block_drop = True
