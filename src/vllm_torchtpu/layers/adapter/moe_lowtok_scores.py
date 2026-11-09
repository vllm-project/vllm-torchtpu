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
"""TPU_MOE_LOWTOK_FROM_SCORES: the few-token kernel's scores entry.

For a small step the adaptive fused EP op runs the few-token kernel with the
router's top-k and, under a placement with copy slots, the choice of one
holder per hit copied expert inside the kernel
(kernels/moe_lowtok/expert_parallel.py, ``moe_lowtok_scores_ep``). This
module decides at weight loading whether a layer can be served that way,
builds the layer's per-rank holder tables, and supplies them to the op on
every call. Every bucket passes the same operands, so the model graph does
not depend on which path a bucket takes.

TPU_MOE_LOWTOK_SCORES_STRICT=1 refuses to start when a layer cannot be
served; otherwise the refusal is logged and the layer keeps its other paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch

from vllm_torchtpu import envs, expert_placement
from vllm_torchtpu.kernels.moe_lowtok.kernel import (
    LANES,
    can_use_moe_lowtok,
    hot_onehot,
    hot_tables,
    pack_tables,
)
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The layer attribute holding its scores-entry operands once armed.
SCORES_OPERANDS_ATTR = "_lowtok_scores_operands"
_JAX_DTYPES = {torch.float8_e4m3fn: jnp.float8_e4m3fn, torch.bfloat16: jnp.bfloat16}


@dataclass(frozen=True)
class ScoresTables:
    """One rank's holder tables for one layer (host arrays)."""

    tables: np.ndarray  # the packed scalar tables (pack_tables)
    primary: np.ndarray  # [LANES] the rank holding each expert's primary
    copy: np.ndarray  # [LANES] the rank holding its copy, -1 without one
    hot_onehot: np.ndarray  # [LANES, HOT_SLOTS] the copied experts, hottest first
    has_copies: bool


def build_scores_tables(layer_name: str, ep_rank: int, ep_size: int) -> ScoresTables:
    """The tables of a placed layer with copies, or of the contiguous
    placement (rank r holds experts r * LANES / ep_size onwards, no copies)
    for an unplaced layer, in the router's (virtual) expert order."""
    n = expert_placement.layer_index(layer_name) if expert_placement.enabled() else None
    if n is not None and expert_placement.has_layer(n):
        gid, primary, copy = expert_placement.holder_tables(n, ep_rank)
        order = expert_placement.hot_order(n)
    else:
        per_rank = LANES // ep_size
        gid = list(range(ep_rank * per_rank, (ep_rank + 1) * per_rank))
        primary = [e // per_rank for e in range(LANES)]
        copy = [-1] * LANES
        order = []
    hot, hot_primary, hot_copy = hot_tables(primary, copy, order)
    return ScoresTables(
        tables=pack_tables(ep_rank, gid, hot, hot_primary, hot_copy),
        primary=np.asarray(primary, np.int32),
        copy=np.asarray(copy, np.int32),
        hot_onehot=hot_onehot(hot),
        has_copies=bool(order),
    )


def _shape(tensor: torch.Tensor) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        tuple(tensor.shape), _JAX_DTYPES.get(tensor.dtype, jnp.float32)
    )


def scores_entry_refusal(layer, activation: str, adaptive: bool) -> str | None:
    """Why the scores entry cannot serve this layer, or None. The routing
    terms the adaptive op itself checks at prebuild (softmax scoring, no
    expert-selection bias, grouping, custom router or routed scaling) are
    not repeated here."""
    if not envs.TPU_MOE_LOWTOK_MAX_TOKENS:
        return "TPU_MOE_LOWTOK_MAX_TOKENS unset"
    if not adaptive:
        return "it runs in the adaptive or v3 fused EP op (MOE_FUSED_EP_KERNEL_IMPL)"
    if not layer.moe_config.moe_parallel_config.use_ep:
        return "no expert parallelism"
    if int(layer.global_num_experts) != LANES:
        return f"{layer.global_num_experts} experts, not {LANES}"
    if not layer.renormalize:
        return "no renormalization of the top-k weights"
    hidden = layer.w13_weight.shape[1]
    if not can_use_moe_lowtok(
        jax.ShapeDtypeStruct((1, hidden), jnp.bfloat16),
        _shape(layer.w13_weight),
        _shape(layer.w2_weight),
        _shape(layer.w13_weight_scale_inv),
        _shape(layer.w2_weight_scale_inv),
        getattr(layer, "w13_bias", None),
        getattr(layer, "w2_bias", None),
        activation=activation,
    ):
        return "the kernel does not support this layer's weights or activation"
    return None


def arm(layer, activation: str, adaptive: bool) -> ScoresTables | None:
    """Build the layer's scores-entry tables and return them, or log why the
    layer is refused (raising under TPU_MOE_LOWTOK_SCORES_STRICT) and return
    None. A no-op returning None without TPU_MOE_LOWTOK_FROM_SCORES."""
    if not envs.TPU_MOE_LOWTOK_FROM_SCORES:
        return None
    refusal = scores_entry_refusal(layer, activation, adaptive)
    if refusal is not None:
        if envs.TPU_MOE_LOWTOK_SCORES_STRICT:
            raise RuntimeError(
                f"few-token MoE scores entry refused: {refusal} "
                "(TPU_MOE_LOWTOK_SCORES_STRICT)"
            )
        logger.warning_once("few-token MoE scores entry refused: %s", refusal)
        return None
    config = layer.moe_config
    tables = build_scores_tables(
        getattr(layer, "layer_name", ""), int(config.ep_rank), int(config.ep_size)
    )
    device = layer.w13_weight.device

    def tensor(array, dtype):
        return torch.tensor(array, dtype=dtype, device=device)

    # Leading axis 1 (or LANES for the one-hot): each rank's own block of an
    # operand the op shards over the EP axis.
    operands = (
        tensor(tables.tables.reshape(1, -1), torch.int32),
        tensor(tables.primary.reshape(1, -1), torch.int32),
        tensor(tables.copy.reshape(1, -1), torch.int32),
        tensor(tables.hot_onehot, torch.bfloat16),
    )
    if tables.has_copies:
        operands += (
            layer.w13_copy_weight,
            layer.w2_copy_weight,
            layer.w13_copy_weight_scale_inv,
            layer.w2_copy_weight_scale_inv,
        )
    else:
        # Never read without copies; the op takes the primaries in their place.
        operands += (
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
        )
    setattr(layer, SCORES_OPERANDS_ATTR, operands)
    logger.info_once(
        "few-token MoE scores entry armed (cap %d rows, copies %s)",
        envs.TPU_MOE_LOWTOK_MAX_TOKENS,
        tables.has_copies,
    )
    return tables


def operands(layer, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The scores-entry operands the adaptive op takes after its own: this
    rank's padded-row mask, then the layer's tables and copy weights. Empty
    for a layer that is not armed."""
    tables = getattr(layer, SCORES_OPERANDS_ATTR, None)
    if tables is None:
        return ()
    padding = token_padding.local_padding_rows(
        hidden_states.shape[0], hidden_states.device
    )
    return (padding, *tables)
