# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch ops for device-requested Host Engram gathers over the DP ICI mesh.

Every rank owns a head shard of each Host Engram table and one native channel
per table (``csrc/engram_host_channel.cc``). In-graph:

* ``gather_hashes``: all-gather the DP ranks' padded hashes over ICI.
* ``start``: select this owner's rows and publish them to its Host worker.
  ``hidden`` passes through an optimization barrier so the request is issued
  before the computation that consumes ``hidden``.
* ``finish``: at the consuming layer, wait for this request, publish the
  retained rows to a fresh output and all-to-all them back to their tokens.
"""

from functools import cache
from typing import cast

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import get_pallas_mesh, jax_op
from torch_tpu._internal.pallas.pallas import JaxCallable

from vllm_torchtpu.kernels import host_channel

AXIS = "engram_dp"
# The gathered row: 256 FP8 bytes then 8 E8M0 scale bytes.
ROW_WORDS = 66
# Scale words of an invalid row: four E8M0 255 bytes each, which dequantize to 0.
_INVALID_ROW = (0,) * 64 + (-1, -1)


def round_pages(words: int) -> int:
    return -(-words // host_channel.PAGE_WORDS) * host_channel.PAGE_WORDS


def _local_rows(hashes: jax.Array, interval: jax.Array, owners: int):
    """This owner's row ids (invalid -> dummy row 0) and validity, token-major."""
    heads = -(-hashes.shape[1] // owners)
    head_start, start, count = interval.reshape(-1)
    padded = jnp.pad(
        hashes, ((0, 0), (0, heads * owners - hashes.shape[1])), constant_values=-1
    )
    selected = lax.dynamic_slice(padded, (0, head_start), (hashes.shape[0], heads))
    valid = (selected >= start) & (selected < start + count)
    return jnp.where(valid, selected - start, 0).reshape(-1), valid


def _start(hashes, interval, step, dependency, host_ids, host_request, hidden):
    rows, _ = _local_rows(hashes, interval, lax.axis_size(AXIS))
    payload = jnp.concatenate((jnp.array([rows.size], jnp.int32), dependency, rows))
    payload = jnp.pad(payload, (0, round_pages(payload.size) - payload.size))
    if payload.size > host_ids.size:
        raise ValueError(
            f"Engram request of {rows.size} rows exceeds the channel capacity of "
            f"{host_ids.size - 2} (max_num_batched_tokens x DP x local heads)"
        )
    # Positive and never repeated across consecutive requests; INT32_MIN
    # is the stop answer.
    generation = jnp.where(step[0] == jnp.iinfo(jnp.int32).max, 1, step[0] + 1)
    page = jnp.broadcast_to(generation, (host_channel.PAGE_WORDS,))
    receipt, host_ids, host_request = host_channel.request(
        payload, page, host_ids, host_request
    )
    receipt, hidden = lax.optimization_barrier((receipt, hidden))
    return (
        receipt,
        jax.device_put(host_ids, jax.memory.Space.Host),
        jax.device_put(host_request, jax.memory.Space.Host),
        hidden,
    )


def _finish(hashes, interval, receipt, host_response, retained, hidden):
    receipt, hidden = lax.optimization_barrier((receipt, hidden))
    owners = lax.axis_size(AXIS)
    rows, valid = _local_rows(hashes, interval, owners)
    status = host_channel.receive(receipt, host_response)
    packed = host_channel.publish(status, retained, round_pages(rows.size * ROW_WORDS))[
        : rows.size * ROW_WORDS
    ]
    packed = jnp.where(
        valid.reshape(-1, 1),
        packed.reshape(-1, ROW_WORDS),
        jnp.array(_INVALID_ROW, jnp.int32),
    ).reshape(*valid.shape, ROW_WORDS)
    rows = lax.all_to_all(packed, AXIS, split_axis=0, concat_axis=1, tiled=True)
    return lax.optimization_barrier((rows[:, : hashes.shape[1]], hidden))


class _HostCallable(JaxCallable):
    """Trace placeholders with their memory kinds; Host pages are pinned_host."""

    placements: tuple[NamedSharding, ...]

    def _get_jax_placeholders(self, norm_args, norm_kwargs=None):
        values, keywords = super()._get_jax_placeholders(norm_args, norm_kwargs)
        return [
            jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=s)
            for v, s in zip(values, self.placements, strict=True)
        ], keywords


# Argument indices of the pinned Host pages; they must be graph inputs.
_HOST_ARGS = {"start": (4, 5), "finish": (3,)}


@cache
def initialize() -> None:
    """Register ``torch.ops.engram_host.*`` once torch.distributed is up."""
    mesh = get_pallas_mesh(axis_names=(AXIS,))
    shard, replicated = P(AXIS), P()

    def callable_(name, body, in_specs, out_specs, donated=()):
        host = _HOST_ARGS[name]
        placements = tuple(
            NamedSharding(
                mesh.abstract_mesh,
                spec,
                memory_kind="pinned_host" if i in host else "device",
            )
            for i, spec in enumerate(in_specs)
        )
        mapped = jax.shard_map(
            body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
        )
        call = _HostCallable(
            name=f"engram_host_{name}",
            trace_key=f"engram_host_{name}",
            jit_fn=jax.jit(mapped, in_shardings=placements, donate_argnums=donated),
            mesh=mesh,
            input_partition_specs=in_specs,
            output_partition_specs=out_specs,
            donate_argnums=list(donated),
        )
        call.placements = placements
        return call

    start = callable_(
        "start", _start, (replicated,) + (shard,) * 6, (shard,) * 4, donated=(4, 5)
    )
    finish = callable_("finish", _finish, (replicated,) + (shard,) * 5, (shard,) * 2)

    @torch.library.custom_op("engram_host::start", mutates_args=())
    def start_op(
        hashes: torch.Tensor,
        interval: torch.Tensor,
        step: torch.Tensor,
        dependency: torch.Tensor,
        host_ids: torch.Tensor,
        host_request: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = start(
            hashes, interval, step, dependency, host_ids, host_request, hidden
        )
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], outputs
        )

    @start_op.register_fake
    def _(hashes, interval, step, dependency, host_ids, host_request, hidden):
        return (
            torch.empty_like(host_request),
            torch.empty_like(host_ids),
            torch.empty_like(host_request),
            torch.empty_like(hidden),
        )

    @torch.library.custom_op("engram_host::finish", mutates_args=())
    def finish_op(
        hashes: torch.Tensor,
        interval: torch.Tensor,
        receipt: torch.Tensor,
        host_response: torch.Tensor,
        retained: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = finish(hashes, interval, receipt, host_response, retained, hidden)
        return cast(tuple[torch.Tensor, torch.Tensor], outputs)

    @finish_op.register_fake
    def _(hashes, interval, receipt, host_response, retained, hidden):
        tokens = hashes.shape[0] // mesh.size
        return (
            retained.new_empty((tokens, hashes.shape[1], ROW_WORDS)),
            torch.empty_like(hidden),
        )

    def gather(hashes: jax.Array) -> jax.Array:
        return jax.shard_map(
            lambda x: lax.all_gather(x, AXIS, axis=0, tiled=True),
            mesh=mesh,
            in_specs=shard,
            out_specs=replicated,
            check_vma=False,
        )(hashes)

    # TorchTPU accepts PartitionSpec; its annotation only permits plain tuples.
    # pyrefly: ignore[no-matching-overload]
    gather_op = jax_op(
        "engram_host::gather_hashes",
        gather,
        mesh=mesh,
        input_partition_specs=(shard,),
        output_partition_specs=(replicated,),
    )

    # The default fake kernel rejects vLLM's symbolic token axis.
    @gather_op.register_fake
    def _(hashes):
        return hashes.new_empty((hashes.shape[0] * mesh.size, *hashes.shape[1:]))

    _mark_host_pages(
        {
            torch.ops.engram_host.start.default: _HOST_ARGS["start"],
            torch.ops.engram_host.finish.default: _HOST_ARGS["finish"],
        }
    )


def _mark_host_pages(host_args: dict) -> None:
    """Keep the Host pages pinned_host graph inputs through AOTAutograd.

    TorchTPU marks only ``ao.reload`` sources; AOTAutograd's fake inputs lose
    the PJRT memory kind of the channel's Host views.
    """
    from torch_tpu._internal.compile import _backend

    mark = _backend._mark_pinned_host_inputs

    def mark_with_channel_pages(graph_module, example_inputs):
        mark(graph_module, example_inputs)
        placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
        for node in graph_module.graph.nodes:
            for index in host_args.get(node.target, ()):
                page = node.args[index]
                if page not in placeholders:
                    raise ValueError("Engram Host channel pages must be graph inputs")
                example_inputs[placeholders.index(page)]._is_pinned_host = True

    _backend._mark_pinned_host_inputs = mark_with_channel_pages


def gather_hashes(hashes: torch.Tensor) -> torch.Tensor:
    return torch.ops.engram_host.gather_hashes(hashes)
