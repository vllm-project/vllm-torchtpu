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
"""Torch bridge for candidate-based non-greedy verify sampling
(``TPU_CANDIDATE_SAMPLING``, ``layers.core.candidate_sampling``).

The runner keeps each rank's logits shard instead of all-gathering the
vocabulary, and one program per verify chunk picks the bonus tokens and the
per-slot accept/recover decisions; ``assemble_rejection_output`` turns those
into the rejection sampler's output. Every rank of the TP group runs the same
program on the same random draws, so the decisions are replicated.
"""

import functools
from typing import NamedTuple

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

from vllm_torchtpu.layers.core.candidate_sampling import candidate_verify
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

TP_AXIS = "tp"
# Candidates each vocabulary shard contributes per row.
CANDIDATES_PER_SHARD = 64
# Rows whose top_k leaves at least 16 candidates of slack per shard; exactness
# is checked per step regardless (ties at the cutoff fall back in-program).
MAX_TOP_K = 48
NUM_OPERANDS = 15


class VocabShardLogits(NamedTuple):
    """This rank's [rows, vocab shard] logits, standing in for full logits."""

    local: torch.Tensor


@functools.cache
def get_tp_mesh():
    """A one-axis mesh over this engine's TP group, partition = TP rank."""
    from torch_tpu._internal.pallas import get_pallas_mesh
    from vllm.distributed.parallel_state import get_tp_group

    from vllm_torchtpu.distributed.mesh_utils import get_pipeline_stage

    group = get_tp_group()
    _, stages = get_pipeline_stage()
    if stages > 1:
        raise NotImplementedError("candidate sampling under pipeline parallelism")
    mesh = get_pallas_mesh(
        axis_names=(TP_AXIS,),
        mesh_shape=(int(group.world_size),),
        process_group=group.device_group,
    )
    logger.info(
        "Candidate sampling mesh | tp=%d ranks=%s devices=%s",
        group.world_size,
        tuple(group.ranks),
        tuple(int(d.id) for d in mesh.devices.flat),
    )
    return mesh


@functools.cache
def _get_op(vocab_size: int, synthetic: bool):
    mesh = get_tp_mesh()
    run = functools.partial(
        candidate_verify,
        mesh=mesh,
        axis=TP_AXIS,
        kc=CANDIDATES_PER_SHARD,
        vocab_size=vocab_size,
        synthetic=synthetic,
    )

    # torch_tpu infers static argnums from annotations: every operand is jax.Array.
    def fn(
        bonus_logits: jax.Array,
        target_logits: jax.Array,
        bounds: jax.Array,
        draft: jax.Array,
        temps_b: jax.Array,
        top_k_b: jax.Array,
        top_p_b: jax.Array,
        temps_t: jax.Array,
        top_k_t: jax.Array,
        top_p_t: jax.Array,
        bonus_u: jax.Array,
        accept_u: jax.Array,
        recover_u: jax.Array,
        rates: jax.Array,
        seed: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return run(
            bonus_logits,
            target_logits,
            bounds,
            draft,
            temps_b,
            top_k_b,
            top_p_b,
            temps_t,
            top_k_t,
            top_p_t,
            bonus_u,
            accept_u,
            recover_u,
            rates,
            seed,
        )

    spec = PartitionSpec(TP_AXIS)
    op = pallas.jax_op(
        f"pallas::candidate_verify_v{vocab_size}_tp{mesh.size}_s{int(synthetic)}",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * NUM_OPERANDS,
        output_partition_specs=(spec,) * 6,
    )
    logger.info(
        "Candidate sampling armed | vocab=%d tp=%d candidates per shard=%d "
        "max top_k=%d synthetic=%s",
        vocab_size,
        mesh.size,
        CANDIDATES_PER_SHARD,
        MAX_TOP_K,
        synthetic,
    )
    return op


def candidate_width(tp_size: int) -> int:
    return CANDIDATES_PER_SHARD * tp_size


def candidate_verify_decisions(
    bonus_logits: torch.Tensor,
    target_logits: torch.Tensor,
    vocab_bounds: torch.Tensor,
    draft_token_ids: torch.Tensor,
    temps_b: torch.Tensor,
    top_k_b: torch.Tensor,
    top_p_b: torch.Tensor,
    temps_t: torch.Tensor,
    top_k_t: torch.Tensor,
    top_p_t: torch.Tensor,
    bonus_u: torch.Tensor,
    accept_u: torch.Tensor,
    recover_u: torch.Tensor,
    rates_t: torch.Tensor,
    seed: torch.Tensor,
    *,
    vocab_size: int,
    synthetic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bonus ids [B], accepted [T] (int32), recovered ids [T] and the
    candidates-exact flag [] for one verify chunk, from this rank's shard of
    the bonus rows [B, Vl] and verified rows [T, Vl]."""
    op = _get_op(vocab_size, synthetic)
    operands = (
        bonus_logits,
        target_logits,
        vocab_bounds,
        draft_token_ids,
        temps_b,
        top_k_b,
        top_p_b,
        temps_t,
        top_k_t,
        top_p_t,
        bonus_u,
        accept_u,
        recover_u,
        rates_t,
        seed,
    )
    outs = op(*(t.reshape(1, *t.shape) for t in operands))
    bonus, accepted, recovered, exact = (o[0] for o in outs[:4])
    return bonus, accepted, recovered, exact
