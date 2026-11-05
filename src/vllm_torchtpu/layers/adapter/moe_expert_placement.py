# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer expert placement for linear expert parallelism, applied at load.

Under linear EP, rank r computes experts [r * E/R, (r + 1) * E/R). When a
layer's popular experts share a rank, that rank gets most of the routed rows
and every other rank waits for it in the MoE all-reduce.
``TPU_MOE_EXPERT_PLACEMENT`` names a table that assigns each layer's experts
to ranks so the expected rows per rank are equal. The table is applied by
renumbering the experts while the checkpoint loads:

- expert e of layer L is loaded as expert ``new_id(L, e)``, so linear EP puts
  it on the rank the table names;
- the router's output rows (``mlp.gate.weight``) and its score-correction
  bias are loaded in the same new order, so the router returns new ids.

Every token picks the same experts with the same weights; nothing is added at
run time. The EP weight filter renumbers before it decides which experts a
rank reads.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import moe_placement_glm52
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM

logger = init_logger(__name__)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.mlp\.")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
_ROUTER_SUFFIXES = (".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")

# Table name -> (ranks, per-layer string of each expert's rank in hex).
_TABLES = {
    "glm52-ep16-prefill": (moe_placement_glm52.RANKS, moe_placement_glm52.LAYERS)
}


# The EP weight filter renumbers only while DeepseekV2ForCausalLM.load_weights
# loads a model it checked against the table, the one load that renumbers the
# experts. The filter refuses to run before that load (a model of another
# class) and leaves the names of later loads, such as a drafter's, as they are.
_model_accepted = False
_loading_accepted_model = False


class Renumbering(NamedTuple):
    new_of_old: list[int]  # new id of each checkpoint expert
    old_of_new: list[int]  # checkpoint expert of each new id


def _load_error(message: str) -> ValueError:
    return ValueError(
        f"TPU_MOE_EXPERT_PLACEMENT={envs.TPU_MOE_EXPERT_PLACEMENT}: {message}; "
        "unset TPU_MOE_EXPERT_PLACEMENT for this model or parallel configuration"
    )


@functools.cache
def placement() -> dict[int, Renumbering]:
    """Each table layer's renumbering; empty when ``TPU_MOE_EXPERT_PLACEMENT``
    is unset."""
    name = envs.TPU_MOE_EXPERT_PLACEMENT
    if not name:
        return {}
    if name not in _TABLES:
        raise _load_error(f"no such table; known: {sorted(_TABLES)}")
    ranks, layers = _TABLES[name]
    return {layer: _renumbering(cores, ranks) for layer, cores in layers.items()}


def _renumbering(cores: str, ranks: int) -> Renumbering:
    rank_of = [int(c, 16) for c in cores]
    per_rank = len(rank_of) // ranks
    old_of_new = [e for r in range(ranks) for e, c in enumerate(rank_of) if c == r]
    if len(rank_of) % ranks or any(rank_of.count(r) != per_rank for r in range(ranks)):
        raise _load_error(
            f"placement table does not give each of {ranks} ranks {per_rank} experts"
        )
    new_of_old = [0] * len(rank_of)
    for new, old in enumerate(old_of_new):
        new_of_old[old] = new
    return Renumbering(new_of_old, old_of_new)


def _layer(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def renamed(name: str) -> str:
    """The checkpoint tensor name with its expert id renumbered."""
    layer = _layer(name)
    match = _EXPERT_RE.search(name)
    if layer not in placement() or match is None:
        return name
    new = placement()[layer].new_of_old[int(match.group(1))]
    return f"{name[: match.start()]}.experts.{new}.{name[match.end() :]}"


def reordered(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """The router's rows (or bias entries) in new id order."""
    layer = _layer(name)
    if layer not in placement() or not name.endswith(_ROUTER_SUFFIXES):
        return tensor
    order = torch.tensor(placement()[layer].old_of_new, device=tensor.device)
    return tensor.index_select(0, order)


def renumbered_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """The checkpoint's tensors under their renumbered names, the router's in
    new id order."""
    for name, tensor in weights:
        yield renamed(name), reordered(name, tensor)


def _check_layers(model: DeepseekV2ForCausalLM) -> None:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE

    ranks = _TABLES[envs.TPU_MOE_EXPERT_PLACEMENT][0]
    if max(placement()) >= len(model.model.layers):
        raise _load_error(
            f"placement table names layers up to {max(placement())}; the model "
            f"has {len(model.model.layers)}"
        )
    for index, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp, DeepseekV2MoE):
            if index in placement():
                raise _load_error(
                    f"placement table names layer {index}, which has no MoE"
                )
            continue
        if index not in placement():
            raise _load_error(f"placement table has no entry for MoE layer {index}")
        config = layer.mlp.experts.moe_config
        if (
            not config.moe_parallel_config.use_ep
            or config.ep_size != ranks
            or config.num_experts != len(placement()[index].new_of_old)
        ):
            raise _load_error(
                f"placement table needs linear EP over {ranks} ranks with "
                f"{len(placement()[index].new_of_old)} experts; layer {index} has "
                f"ep_size {config.ep_size} and {config.num_experts} experts"
            )


def install() -> None:
    """Renumber experts while DeepSeek-V2-family checkpoints (GLM-5.2) load,
    and make the EP weight filter keep each rank's renumbered experts."""
    if not envs.TPU_MOE_EXPERT_PLACEMENT:
        return
    placement()
    from vllm.model_executor.models import deepseek_v2

    from vllm_torchtpu import model_loader_patches

    model_class = deepseek_v2.DeepseekV2ForCausalLM
    if getattr(model_class, "_tpu_expert_placement", False):
        return
    upstream_load_weights = model_class.load_weights
    upstream_should_skip = model_loader_patches._should_skip

    def load_weights(self, weights):
        global _model_accepted, _loading_accepted_model
        _check_layers(self)
        _model_accepted = _loading_accepted_model = True
        try:
            return upstream_load_weights(self, renumbered_weights(weights))
        finally:
            _loading_accepted_model = False

    def should_skip(name: str, local_expert_ids: set[int]) -> bool:
        if not _model_accepted:
            raise _load_error(
                "the model being loaded is not a DeepseekV2ForCausalLM, the "
                "only model class that applies the table"
            )
        if _loading_accepted_model:
            name = renamed(name)
        return upstream_should_skip(name, local_expert_ids)

    model_class.load_weights = load_weights
    model_class._tpu_expert_placement = True
    model_loader_patches._should_skip = should_skip
    logger.info(
        "Applied TPU patch: expert placement %s on %d MoE layers.",
        envs.TPU_MOE_EXPERT_PLACEMENT,
        len(placement()),
    )
