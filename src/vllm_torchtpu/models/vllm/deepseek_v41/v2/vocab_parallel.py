# SPDX-License-Identifier: Apache-2.0
"""Torch ops for the vocabulary-parallel V4.1 heads.

With `TPU_DEEPSEEK_V41_VOCAB_PARALLEL_HEADS`, each EP rank keeps a
``shard``-row slice of the shared LM head (and of the DSpark Markov head) and
runs `kernels/vocab_parallel_head.py` over the EP mesh. The ops hold
collectives, so every rank must call them in the same order with equal shapes:
the runner pads the target's logits rows to a DP-wide count
(`TPUModelRunner._dp_logits_rows`), and DSpark's draft rows are DP-wide already.
"""

from functools import partial

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import jax_op

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import vocab_parallel_head as vph
from vllm_torchtpu.kernels.logits_head import is_supported_slice
from vllm_torchtpu.kernels.markov_sampling import supports_group
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_ROWS = P(vph.AXIS, None)
_HEADS: "VocabParallelHeads | None" = None


def _target_logits_op(mesh, world: int, vocab_size: int):
    # torch_tpu's jax_op needs every argument annotated.
    def logits(hidden: jax.Array, weight_shard: jax.Array) -> jax.Array:
        # Two chunks: the first chunk's exchange overlaps the second's matmul.
        local = partial(vph.target_logits, world=world, vocab_size=vocab_size, chunks=2)
        return jax.shard_map(
            local, mesh=mesh, in_specs=(_ROWS,) * 2, out_specs=_ROWS, check_vma=False
        )(hidden, weight_shard)

    return logits


def _dspark_samples_op(mesh, world: int, vocab_size: int):
    def local(hidden, anchors, markov_embed, markov_shard, weight_shard, rank):
        return vph.dspark_samples(
            hidden,
            anchors[:, 0],
            markov_embed,
            markov_shard,
            weight_shard,
            rank,
            world=world,
            vocab_size=vocab_size,
        )

    def samples(
        hidden: jax.Array,
        anchors: jax.Array,
        markov_embed: jax.Array,
        markov_shard: jax.Array,
        weight_shard: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        return jax.shard_map(
            local, mesh=mesh, in_specs=(_ROWS,) * 6, out_specs=_ROWS, check_vma=False
        )(hidden, anchors, markov_embed, markov_shard, weight_shard, rank)

    return samples


class VocabParallelHeads:
    """This rank's head slices and the EP-mesh ops that use them."""

    def __init__(self, lm_weight: torch.Tensor):
        from vllm_torchtpu.distributed.ep_mesh import (
            build_ep_mesh,
            ep_mesh_index,
            get_ep_group,
        )

        group = get_ep_group()
        mesh = build_ep_mesh(axis_name=vph.AXIS)
        rank = ep_mesh_index()
        if group is None or mesh is None or rank is None:
            raise RuntimeError("Vocab-parallel heads need an expert-parallel group")
        self.world = int(group.world_size)
        self.rank = rank
        self.vocab_size, self.width = lm_weight.shape
        self.shard = vph.shard_rows(self.vocab_size, self.world)
        self._mesh = mesh
        self.lm_slice = self._slice(lm_weight, self.shard)
        self._rank_index = torch.tensor(
            [[rank]], dtype=torch.int32, device=lm_weight.device
        )
        self._target_op = jax_op(
            "pallas::dsv41_vocab_parallel_logits",
            _target_logits_op(mesh, self.world, self.vocab_size),
            mesh=mesh,
            input_partition_specs=(_ROWS,) * 2,
            output_partition_specs=_ROWS,
        )
        vocab = self.vocab_size
        self._target_op.register_fake(
            lambda hidden, weight: hidden.new_empty((hidden.shape[0], vocab))
        )
        self.markov_slice: torch.Tensor | None = None
        self._samples_op = None
        logger.info(
            "[DSV41] Vocab-parallel LM head: %d ranks, rank %d holds vocab rows "
            "[%d, %d) of %d",
            self.world,
            rank,
            rank * self.shard,
            min((rank + 1) * self.shard, self.vocab_size),
            self.vocab_size,
        )

    def reload(self, lm_weight: torch.Tensor) -> None:
        """Refresh the LM head slice after the target's weights reload."""
        self.lm_slice.copy_(self._slice(lm_weight, self.shard))

    def _slice(self, weight: torch.Tensor, rows_out: int) -> torch.Tensor:
        """This rank's ``shard`` rows of a [vocab, width] head, zero-padded to
        ``rows_out`` rows."""
        start = self.rank * self.shard
        rows = weight[start : start + self.shard]
        out = torch.zeros(
            (rows_out, weight.shape[1]), dtype=weight.dtype, device=weight.device
        )
        out[: rows.shape[0]].copy_(rows)
        return out

    def supports_target_rows(self, rows: int) -> bool:
        return is_supported_slice(self.world * rows, self.width)

    def target_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """[R, hidden] -> [R, vocab], R equal on every rank."""
        return self._target_op(hidden, self.lm_slice)

    def enable_dspark(self, markov_head: torch.Tensor) -> None:
        """Slice the DSpark Markov head and build the draft sampler op; on a
        later call (a weight reload), refresh the slice."""
        assert markov_head.shape[0] == self.vocab_size
        if not envs.TPU_DEEPSEEK_V41_DSPARK_PALLAS_LOGITS:
            raise ValueError(
                "TPU_DEEPSEEK_V41_VOCAB_PARALLEL_HEADS=1 shards the Pallas DSpark "
                "head: it needs TPU_DEEPSEEK_V41_DSPARK_PALLAS_LOGITS=1"
            )
        markov_slice = self._slice(markov_head, vph.markov_rows(self.shard))
        if self.markov_slice is not None:
            self.markov_slice.copy_(markov_slice)
            return
        self.markov_slice = markov_slice

        self._samples_op = jax_op(
            "pallas::dsv41_vocab_parallel_dspark_samples",
            _dspark_samples_op(self._mesh, self.world, self.vocab_size),
            mesh=self._mesh,
            input_partition_specs=(_ROWS,) * 6,
            output_partition_specs=_ROWS,
        )
        self._samples_op.register_fake(
            lambda hidden, anchors, *_: anchors.new_empty(
                (anchors.shape[0], hidden.shape[0] // anchors.shape[0]),
                dtype=torch.int32,
            )
        )
        logger.info("[DSV41] Vocab-parallel DSpark draft head and Markov sampler")

    def supports_dspark(self, num_reqs: int, block_size: int) -> bool:
        return self._samples_op is not None and is_supported_slice(
            self.world * num_reqs * block_size, self.width
        )

    @staticmethod
    def supports_world(world: int, vocab_size: int) -> bool:
        """Whether one Markov kernel tile's key packing covers a slice."""
        return supports_group(vph.markov_rows(vph.shard_rows(vocab_size, world)))

    def dspark_samples(
        self, hidden: torch.Tensor, anchors: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """Slot-major normed [block * n, hidden] rows and [n] anchors ->
        [n, block] int32 greedy draft tokens. ``markov_embed`` is the full
        Markov embedding: each rank passes it as its own block of the mesh."""
        assert self._samples_op is not None and self.markov_slice is not None
        return self._samples_op(
            hidden,
            anchors.to(torch.int32).reshape(-1, 1),
            markov_embed,
            self.markov_slice,
            self.lm_slice,
            self._rank_index,
        )


def build(lm_weight: torch.Tensor) -> VocabParallelHeads:
    """Create this process's heads once, after the target's weights load."""
    global _HEADS
    assert _HEADS is None, "vocab-parallel heads are built once per process"
    _HEADS = VocabParallelHeads(lm_weight)
    return _HEADS


def heads() -> VocabParallelHeads | None:
    return _HEADS
