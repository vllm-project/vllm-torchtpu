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
"""The LM head split by vocabulary rows over the expert-parallel group.

Each rank keeps ``vocab / ep`` rows of the head. Per step the selected hidden
rows of every rank are all-gathered, multiplied against the local slice, and an
all-to-all returns each rank its own rows over the full vocabulary, so a rank
reads ``1 / ep`` of the head's weight instead of all of it.

The collectives run at one fixed row count on every rank, the most rows any
logits call can pass (``logits_rows``); the op pads to it and trims inside its
own program, so ranks calling it with different row counts still pair.
"""

import bisect
import copy

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from vllm.config import SpeculativeConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _refuse(reason: str) -> NotImplementedError:
    return NotImplementedError(
        f"TPU_SHARDED_LM_HEAD: {reason}. Unset TPU_SHARDED_LM_HEAD to serve "
        "this configuration."
    )


def request_buckets(max_num_reqs: int) -> list[int]:
    """The runner's request buckets while the head is sharded: exactly one.

    A smaller bucket padded and trimmed the rows around the op in the torch
    graph, and above one row tile the compiler read the op's output at the
    wrong vocabulary positions. The op now pads inside its own program, but
    only the one-bucket shape has run on TPU.
    """
    return [max_num_reqs]


def logits_rows(
    max_num_reqs: int,
    num_tokens_paddings: list[int],
    speculative_config: SpeculativeConfig | None,
) -> int:
    """The most rows a logits call can pass: max_num_reqs, or under speculative
    decoding the token bucket holding max_num_reqs * (1 + k) verify rows."""
    if speculative_config is None:
        return max_num_reqs
    verify_rows = max_num_reqs * (1 + speculative_config.num_speculative_tokens)
    bucket = min(verify_rows, num_tokens_paddings[-1])
    return num_tokens_paddings[bisect.bisect_left(num_tokens_paddings, bucket)]


def find_head(model: torch.nn.Module) -> ParallelLMHead:
    """The model's untied vocabulary projection."""
    heads = [m for m in model.modules() if isinstance(m, ParallelLMHead)]
    if not heads:
        raise _refuse("the model has no ParallelLMHead on this rank")
    head = heads[0]
    for module in model.modules():
        if (
            module is not head
            and isinstance(module, VocabParallelEmbedding)
            and module.weight is head.weight
        ):
            raise _refuse("the LM head is tied to the token embedding")
    return head


def head_fn(mesh: jax.sharding.Mesh, rows: int):
    """Pad to `rows`, gather, multiply the local vocabulary slice, exchange."""

    def local(x: jax.Array, w: jax.Array) -> jax.Array:
        num_rows = x.shape[0]
        x_padded = jnp.pad(x, ((0, rows - num_rows), (0, 0)))
        gathered = jax.lax.all_gather(x_padded, EP_AXIS_NAME, axis=0, tiled=True)
        logits = jnp.dot(gathered, w.T, preferred_element_type=jnp.float32)
        logits = jax.lax.all_to_all(
            logits.astype(x.dtype),
            EP_AXIS_NAME,
            split_axis=0,
            concat_axis=1,
            tiled=True,
        )
        return logits[:num_rows]

    def fn(x: jax.Array, w: jax.Array) -> jax.Array:
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(EP_AXIS_NAME), P(EP_AXIS_NAME)),
            out_specs=P(EP_AXIS_NAME),
            check_vma=False,
        )(x, w)

    return fn


def build_op(mesh: jax.sharding.Mesh, rows: int):
    ep_size = mesh.shape[EP_AXIS_NAME]
    op = sharded_jax_op(
        "pallas::sharded_lm_head",
        head_fn(mesh, rows),
        mesh=mesh,
        input_partition_specs=(P(EP_AXIS_NAME), P(EP_AXIS_NAME)),
        output_partition_specs=P(EP_AXIS_NAME),
    )

    def _fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x.new_empty((x.shape[0], w.shape[0] * ep_size))

    op.register_fake(_fake)
    return op


class ShardedLMHeadMethod:
    """`quant_method` for the head: project at most `rows` rows."""

    def __init__(self, op, rows: int):
        self.op = op
        self.rows = rows

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_rows = x.shape[0]
        if num_rows > self.rows:
            raise ValueError(
                f"sharded LM head: {num_rows} rows exceed the fixed {self.rows}"
            )
        out = self.op(x, layer.weight_slice)
        if bias is not None:
            out = out + bias
        return out

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor):
        raise NotImplementedError("The sharded LM head is not an embedding.")


def _draft_shares_collectives(drafter) -> bool:
    """Whether an idle DP rank replays every draft logits call of a busy one.

    Eagle3/MTP drafts that join the EP-DP lockstep run the draft LM head
    num_speculative_tokens times per chunk on busy and idle ranks alike, at
    rows no larger than `logits_rows`. Any other draft would call the head's
    collectives on busy ranks only.
    """
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    return isinstance(drafter, Eagle3Proposer) and drafter._dp_lockstep_sharded()


def _unshare_draft_head(draft_model: torch.nn.Module, head: ParallelLMHead) -> None:
    """Point every draft reference to `head` at a full-weight copy of it."""
    parents = [
        (parent, name)
        for parent in draft_model.modules()
        for name, child in parent.named_children()
        if child is head
    ]
    if parents:
        full_head = copy.deepcopy(head)
        for parent, name in parents:
            setattr(parent, name, full_head)
        logger.info("sharded LM head: the draft keeps a full-weight copy")


def install(model: torch.nn.Module, rows: int, drafter=None) -> None:
    """Replace the model's LM head with the sharded projection.

    Call after the drafter has loaded. A draft that shares the head keeps
    sharing it when its logits calls pair across ranks, and otherwise gets a
    full-weight copy; a draft that copied the head already holds the full
    weight.
    """
    head = find_head(model)
    if head.tp_size > 1:
        raise _refuse(
            f"the LM head is already split over {head.tp_size} tensor-parallel "
            "ranks; run with tensor_parallel_size=1 and data parallelism"
        )
    if head.weight.element_size() < 2:
        raise _refuse(
            f"the LM head weight is {head.weight.dtype}; it must be unquantized"
        )
    mesh = build_ep_mesh()
    if mesh is None:
        raise _refuse("no expert-parallel group; pass --enable-expert-parallel")
    ep_size = mesh.shape[EP_AXIS_NAME]
    weight = head.weight.data
    vocab, hidden = weight.shape
    if vocab % ep_size:
        raise _refuse(f"the padded vocabulary {vocab} does not split {ep_size} ways")
    if drafter is not None and not _draft_shares_collectives(drafter):
        _unshare_draft_head(drafter.draft_model, head)
    part = vocab // ep_size
    index = ep_mesh_index()
    # Sliced on the host: a device-side slice is a lazy strided view whose
    # compiled program is cached without the slice bounds, so ranks slicing
    # different windows would load one another's executables.
    weight_cpu = weight.cpu()
    local = weight_cpu[index * part : (index + 1) * part].contiguous()
    head.weight_slice = torch.nn.Parameter(local.to(weight.device), requires_grad=False)
    # A one-row stand-in keeps `head.weight` for shape reads without the
    # full vocabulary.
    head.weight = torch.nn.Parameter(
        weight_cpu[:1].clone().to(weight.device), requires_grad=False
    )
    head.quant_method = ShardedLMHeadMethod(build_op(mesh, rows), rows)
    logger.info(
        "sharded LM head: %d x %d split %d ways, rows %d-%d here, %d collective rows",
        vocab,
        hidden,
        ep_size,
        index * part,
        (index + 1) * part,
        rows,
    )
