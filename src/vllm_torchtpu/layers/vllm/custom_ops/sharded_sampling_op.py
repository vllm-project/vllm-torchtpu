# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""TP batch-sharded sampling custom op."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from torch_tpu._internal.distributed import tpu_distributed

from vllm_torchtpu.layers.common.binary_search import topk_mask, topp_mask

TP_AXIS_NAME = "tp"
_MASKED_LOGIT_VALUE = -1e12


def _sample(logits: jax.Array, temperatures: jax.Array, u: jax.Array,
            top_k: jax.Array, top_p: jax.Array) -> jax.Array:
    safe_temperatures = jnp.where(temperatures == 0.0, 1.0, temperatures)
    scaled_logits = logits / safe_temperatures
    masked_logits = lax.cond(
        jnp.any(top_k > 0), lambda logits_: jnp.where(
            top_k > 0,
            topk_mask(logits_, jnp.squeeze(top_k, -1), _MASKED_LOGIT_VALUE),
            logits_), lambda logits_: logits_, scaled_logits)
    masked_logits = lax.cond(
        jnp.any(top_p < 1.0), lambda logits_: jnp.where(
            top_p < 1.0,
            topp_mask(logits_, jnp.squeeze(top_p, -1), _MASKED_LOGIT_VALUE),
            logits_), lambda logits_: logits_, masked_logits)
    u = jnp.clip(u, jnp.finfo(u.dtype).tiny, 1.0 - jnp.finfo(u.dtype).eps)
    noisy_logits = masked_logits - jnp.log(-jnp.log(u))
    return jnp.argmax(jnp.where(temperatures <= 1e-5, logits, noisy_logits),
                      axis=-1).astype(jnp.int32)


def _gather_owned_logits(local_logits: jax.Array, tp_size: int,
                         rank: jax.Array, vocab_rank: jax.Array):
    batch_size, local_vocab_size = local_logits.shape
    slots = (batch_size + tp_size - 1) // tp_size
    payload = math.gcd(local_vocab_size, 128)
    local_tiles = local_vocab_size // payload
    padded = jnp.pad(local_logits, ((0, slots * tp_size - batch_size), (0, 0)))
    operand = padded.reshape(slots, tp_size, local_tiles, payload).transpose(
        1, 0, 2, 3).reshape(tp_size * slots * local_tiles, payload)

    owners = jnp.arange(tp_size, dtype=jnp.int32)
    rows_per_owner = jnp.maximum(
        (batch_size + tp_size - 1 - owners) // tp_size, 0)
    input_offsets = owners * slots * local_tiles
    send_sizes = rows_per_owner * local_tiles
    output_offsets = jnp.full((tp_size, ),
                              vocab_rank * slots * local_tiles,
                              dtype=jnp.int32)
    recv_sizes = jnp.full((tp_size, ),
                          rows_per_owner[rank] * local_tiles,
                          dtype=jnp.int32)
    gathered = lax.ragged_all_to_all(
        operand,
        jnp.zeros((tp_size * slots * local_tiles, payload),
                  dtype=local_logits.dtype),
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=TP_AXIS_NAME,
    )
    owner_logits = gathered.reshape(tp_size, slots, local_tiles,
                                    payload).transpose(1, 0, 2, 3).reshape(
                                        slots, tp_size * local_vocab_size)
    row_ids = rank + jnp.arange(slots, dtype=jnp.int32) * tp_size
    return owner_logits, row_ids, row_ids < batch_size


def _tp_rank(tp_size: int) -> jax.Array:
    ranks = jnp.arange(tp_size, dtype=jnp.int32)
    ranks = lax.all_to_all(ranks,
                           axis_name=TP_AXIS_NAME,
                           split_axis=0,
                           concat_axis=0,
                           tiled=True)
    return ranks[0]


def _run_sharded_sample(local_logits: jax.Array, temperatures: jax.Array,
                        u: jax.Array, top_k: jax.Array, top_p: jax.Array, *,
                        mesh: Mesh, vocab_size: int,
                        vocab_ranks: tuple[int, ...]) -> jax.Array:
    tp_size = mesh.shape[TP_AXIS_NAME]
    batch_size = local_logits.shape[0]

    def per_device(local_logits_, temperatures_, u_, top_k_, top_p_):
        rank = _tp_rank(tp_size)
        owner_logits, row_ids, valid = _gather_owned_logits(
            local_logits_, tp_size, rank,
            jnp.asarray(vocab_ranks)[rank])

        def owned_rows(tensor):
            return tensor[jnp.minimum(row_ids, tensor.shape[0] - 1)]

        operands = (owner_logits[:, :vocab_size], owned_rows(temperatures_),
                    owned_rows(u_), owned_rows(top_k_), owned_rows(top_p_))

        def sample_owned(values):
            all_greedy = jnp.all(
                jnp.where(valid[:, None], values[1] <= 1e-5, True))
            return lax.cond(
                all_greedy, lambda tensors: jnp.argmax(tensors[0], axis=-1).
                astype(jnp.int32), lambda tensors: _sample(*tensors), values)

        owner_tokens = lax.cond(
            jnp.any(valid), sample_owned, lambda values: jnp.zeros(
                (values[0].shape[0], ), dtype=jnp.int32), operands)
        output_rows = jnp.where(valid, row_ids, batch_size)
        local_tokens = jnp.zeros(
            (batch_size, ), dtype=jnp.int32).at[output_rows].add(jnp.where(
                valid, owner_tokens, 0),
                                                                 mode="drop")
        return lax.psum(local_tokens, TP_AXIS_NAME)

    return jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(P(None, TP_AXIS_NAME), P(), P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )(local_logits, temperatures, u, top_k, top_p)


def build_sharded_sampling_op(vocab_size: int) -> object:
    devices = {int(device.id): device for device in jax.devices()}
    physical_device_ids = tuple(
        int(value) for value in tpu_distributed.all_global_device_ids())
    local_device_id = torch.tensor([tpu_distributed.global_device_id()],
                                   dtype=torch.int32,
                                   device="tpu")
    gathered_device_ids = [
        torch.empty_like(local_device_id)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(gathered_device_ids, local_device_id)
    torch_rank_by_device_id = {
        int(value.cpu().item()): rank
        for rank, value in enumerate(gathered_device_ids)
    }
    vocab_ranks = tuple(torch_rank_by_device_id[device_id]
                        for device_id in physical_device_ids)
    mesh = Mesh(
        np.asarray([devices[device_id] for device_id in physical_device_ids]),
        (TP_AXIS_NAME, ))

    def sharded_sample(local_logits: jax.Array, temperatures: jax.Array,
                       u: jax.Array, top_k: jax.Array,
                       top_p: jax.Array) -> jax.Array:
        return _run_sharded_sample(local_logits,
                                   temperatures,
                                   u,
                                   top_k,
                                   top_p,
                                   mesh=mesh,
                                   vocab_size=vocab_size,
                                   vocab_ranks=vocab_ranks)

    return pallas.jax_op(
        f"pallas::tp_sharded_sample_v{vocab_size}_tp{mesh.size}",
        sharded_sample,
        mesh=mesh,
        input_partition_specs=(P(None, TP_AXIS_NAME), P(), P(), P(), P()),
    )
