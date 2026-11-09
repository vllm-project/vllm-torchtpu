# SPDX-License-Identifier: Apache-2.0
"""Fixed-envelope slot-major DSpark sampling and final mHC collapse."""

import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.logits_head import logits_head
from vllm_torchtpu.kernels.markov_sampling import sample_slot
from vllm_torchtpu.models.vllm.deepseek_v41.common.mhc import _collapse_flat
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh


def samples(
    logits: jax.Array,
    embedding: jax.Array,
    markov_weight: jax.Array,
    anchors: jax.Array,
    *,
    pipelined: bool,
) -> jax.Array:
    previous = anchors
    tokens = []
    for slot in range(logits.shape[0] // anchors.shape[0]):
        previous = sample_slot(
            embedding[previous], markov_weight, logits, slot=slot, pipelined=pipelined
        )
        tokens.append(previous)
    return jnp.stack(tokens, axis=1).astype(jnp.int64)


def initialize_samples():
    return _initialize_samples(envs.TPU_DSPARK_PIPELINED_MARKOV)


@functools.cache
def _initialize_samples(pipelined: bool):
    def sample(
        logits: jax.Array,
        embedding: jax.Array,
        markov_weight: jax.Array,
        anchors: jax.Array,
    ) -> jax.Array:
        return samples(logits, embedding, markov_weight, anchors, pipelined=pipelined)

    op = pallas.jax_op(
        f"pallas::dspark_slot_major_samples_{int(pipelined)}",
        sample,
        mesh=None,
        input_partition_specs=cast(Any, (P(),) * 4),
    )
    op.register_fake(
        lambda logits, e, mw, anchors: anchors.new_empty(
            (anchors.shape[0], logits.shape[0] // anchors.shape[0])
        )
    )
    return op


def initialize_logits_head():
    op = pallas.jax_op(
        "pallas::dspark_logits_head",
        logits_head,
        mesh=None,
        input_partition_specs=cast(Any, (P(),) * 2),
    )
    op.register_fake(
        lambda hidden, weight: hidden.new_empty((hidden.shape[0], weight.shape[0]))
    )
    return op


def collapse_flat(flat: jax.Array, coeff: jax.Array) -> jax.Array:
    return _collapse_flat(flat, coeff).astype(flat.dtype)


def initialize_collapse():
    op = pallas.jax_op(
        "pallas::dspark_flat_collapse",
        collapse_flat,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(Any, (P(), P())),
    )
    op.register_fake(
        lambda flat, coeff: flat.new_empty(
            (flat.shape[0], flat.shape[1] // coeff.shape[1])
        )
    )
    return op
