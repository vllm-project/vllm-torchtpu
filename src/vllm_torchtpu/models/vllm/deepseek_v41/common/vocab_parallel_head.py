# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-parallel LM head over the EP ranks.

Without it every rank holds the whole head weight [V, D] and scores its own
rows against it, so every rank reads the whole weight from HBM at every call.
With it, rank i holds only vocab rows [i*V/W, (i+1)*V/W):

1. the rows of all W ranks are gathered on every rank ([W*T, D]);
2. each rank scores all of them against its 1/W of the vocab ([W*T, V/W]);
3. an all-to-all sends each rank its own T rows over every vocab slice, which
   are laid side by side into [T, V].

Each logit is the same dot product over D as before, accumulated in FP32 and
rounded once to BF16; only finished values move between ranks. Every rank must
call the head with the same row count, since the collectives pair across ranks.

``shard_lm_head`` splits the head whenever it can serve it and otherwise keeps
the whole head, with one log line naming the reason (``whole_head_reason``).
"""

from functools import cache

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


def make_head(mesh, world: int, vocab: int):
    """The SPMD head over ``mesh``: (rows [W*T, D] bf16, weight [V, D] bf16),
    both sharded on axis 0, -> logits [W*T, V] bf16 sharded on axis 0."""
    assert vocab % world == 0, f"vocab {vocab} is not divisible by {world} ranks"
    shard = vocab // world

    def local(x, w):
        rows, width = x.shape
        # The gather of every rank's rows, written as an all-to-all of W
        # copies: serving offloads all-gathers to SparseCore, and the
        # Engram HBM lookup (engram_hbm.py) found that offload unsafe on
        # multi-rank decode, while the all-to-all form is not affected.
        copies = jnp.broadcast_to(x[None], (world, rows, width))
        every = jax.lax.all_to_all(
            copies.reshape(world * rows, width), EP_AXIS_NAME, 0, 0, tiled=True
        )
        part = jax.lax.dot_general(
            every,
            w,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        # [owner rank, T, V/W] -> [vocab slice, T, V/W] -> [T, V].
        mine = jax.lax.all_to_all(
            part.reshape(world, rows, shard), EP_AXIS_NAME, 0, 0, tiled=True
        )
        return jnp.transpose(mine, (1, 0, 2)).reshape(rows, vocab)

    def head(x: jax.Array, w: jax.Array) -> jax.Array:
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(EP_AXIS_NAME, None), P(EP_AXIS_NAME, None)),
            out_specs=P(EP_AXIS_NAME, None),
            check_vma=False,
        )(x, w)

    return head


@cache
def build_head_op(world: int, vocab: int, width: int):
    """Torch op for ``make_head`` over the EP mesh, shared by every row count
    and by the target and draft heads. Builds the mesh, a collective, so it is
    called at load time and never first inside a trace."""
    from torch_tpu._internal.pallas import jax_op

    from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh

    mesh = build_ep_mesh()
    if mesh is None:
        raise RuntimeError("The vocab-parallel LM head needs an expert-parallel mesh")
    op = jax_op(
        f"pallas::dsv41_vocab_parallel_head_w{world}_v{vocab}_d{width}",
        make_head(mesh, world, vocab),
        mesh=mesh,
        input_partition_specs=(P(EP_AXIS_NAME, None), P(EP_AXIS_NAME, None)),
        output_partition_specs=P(EP_AXIS_NAME, None),
    )

    def fake(x, w):
        return x.new_empty((x.shape[0], vocab), dtype=torch.bfloat16)

    op.register_fake(fake)
    return op


def _logits_processor_reason(lm_head, logits_processor) -> str | None:
    """Why the logits processor does more than return the head's matmul."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    weight = lm_head.weight
    if logits_processor.logits_as_input:
        return "the logits processor takes logits as input"
    if logits_processor.soft_cap is not None:
        return f"the logits are soft-capped at {logits_processor.soft_cap}"
    if logits_processor.scale != 1.0:
        return f"the logits are scaled by {logits_processor.scale}"
    if logits_processor.org_vocab_size != weight.shape[0]:
        return (
            f"the head is padded from {logits_processor.org_vocab_size} to "
            f"{weight.shape[0]} vocab rows"
        )
    if lm_head.bias is not None:
        return "the head has a bias"
    if not isinstance(
        lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
    ):
        return "the head is quantized"
    if logits_processor.head_dtype not in (None, weight.dtype):
        return f"the head computes in {logits_processor.head_dtype}"
    return None


def _fused_head_reason(weight, vllm_config) -> str | None:
    """Which fused kernel that reads the whole vocab serves this deployment:
    the fused greedy head (as ``TPUModelRunner._init_fused_greedy_head``
    builds it) or the DSpark draft sampler (48 requests of 5 draft rows,
    ``DSparkDeepseekV41ForCausalLM._sample_decode``)."""
    from vllm_torchtpu.layers.adapter.sample.fused_greedy_head import (
        FusedGreedyHead,
    )

    spec = vllm_config.speculative_config
    parallel_config = vllm_config.parallel_config
    if spec is None or spec.method != "dspark":
        return None
    if parallel_config.tensor_parallel_size != 1:
        return None
    max_reqs = vllm_config.scheduler_config.max_num_seqs
    k = spec.num_speculative_tokens
    rows = max_reqs * (k + 1)
    if (
        envs.TPU_GREEDY_HEAD_BACKEND != "logits"
        and parallel_config.pipeline_parallel_size == 1
        and vllm_config.lora_config is None
        and FusedGreedyHead.is_supported(weight, rows)
    ):
        return f"the fused greedy head reads the whole vocab for {rows} rows"
    if (max_reqs, k) == (48, 5) and tuple(weight.shape) == (129280, 5120):
        return "the fused DSpark draft sampler reads the whole vocab"
    return None


def whole_head_reason(lm_head, logits_processor, vllm_config) -> str | None:
    """Why ``lm_head`` must stay whole on this deployment, or None to split it."""
    from vllm_torchtpu.distributed.ep_mesh import ep_device_ids, ep_mesh_index

    parallel_config = vllm_config.parallel_config
    if not (
        parallel_config.data_parallel_size > 1
        and parallel_config.enable_expert_parallel
    ):
        return "its collectives need DP lockstep (DP > 1 with expert parallelism)"
    if envs.TPU_GREEDY_HEAD_BACKEND == "pallas":
        return "TPU_GREEDY_HEAD_BACKEND=pallas needs the whole vocab on every rank"
    reason = _logits_processor_reason(lm_head, logits_processor)
    if reason is not None:
        return reason
    reason = _fused_head_reason(lm_head.weight, vllm_config)
    if reason is not None:
        return reason
    ids, rank = ep_device_ids(), ep_mesh_index()
    if ids is None or rank is None or len(ids) < 2:
        return "it needs EP over 2 or more ranks"
    weight = lm_head.weight
    if weight.shape[0] % len(ids) or weight.dtype != torch.bfloat16:
        return (
            f"it needs a BF16 vocab divisible by {len(ids)}; got "
            f"{tuple(weight.shape)} {weight.dtype}"
        )
    world = torch.distributed.get_world_size()
    if len(ids) != world:
        return (
            f"its mesh spans the whole world of {world} ranks, and the EP "
            f"group has {len(ids)}"
        )
    return None


def shard_lm_head(lm_head: torch.nn.Module, logits_processor, vllm_config) -> bool:
    """Keep only this rank's vocab slice of ``lm_head.weight`` and attach the
    head op where the head can be split; returns whether it was split."""
    from vllm_torchtpu.distributed.ep_mesh import ep_device_ids, ep_mesh_index

    reason = whole_head_reason(lm_head, logits_processor, vllm_config)
    if reason is not None:
        logger.info("LM head kept whole (not vocab-parallel): %s", reason)
        return False
    ids, rank = ep_device_ids(), ep_mesh_index()
    assert ids is not None and rank is not None
    world = len(ids)
    weight = lm_head.weight
    vocab, width = weight.shape
    shard = vocab // world
    op = build_head_op(world, vocab, width)
    piece = weight.data[rank * shard : (rank + 1) * shard].clone()
    synchronize_tensors(piece)
    lm_head.weight = torch.nn.Parameter(piece, requires_grad=False)
    lm_head.vocab_parallel_op = op
    lm_head.vocab_parallel_vocab = vocab
    logger.info(
        "Vocab-parallel LM head: rank %d holds vocab rows [%d, %d) of %d",
        rank,
        rank * shard,
        (rank + 1) * shard,
        vocab,
    )
    return True


def is_vocab_parallel(lm_head: torch.nn.Module | None) -> bool:
    return getattr(lm_head, "vocab_parallel_op", None) is not None


def logits(lm_head: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """[T, D] rows of this rank -> [T, V] BF16 logits over the whole vocab."""
    return lm_head.vocab_parallel_op(hidden.to(torch.bfloat16), lm_head.weight)
