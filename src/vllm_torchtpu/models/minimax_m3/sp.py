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
"""Sequence-parallel residual stream for long MiniMax M3 steps.

With TPU_MINIMAX_M3_SP_MIN_TOKENS set, a step of at least that many tokens
(and a multiple of the TP size) keeps the residual stream split by token
across the TP group. Per layer:

    x = all_gather(norm(h + r))          # attention input, every token
    h = reduce_scatter(attn_partial(x))  # this rank's tokens
    x = all_gather(norm(h + r))          # FFN input, every token
    h = reduce_scatter(ffn_partial(x))

An all-reduce moves the bytes of a reduce-scatter plus an all-gather, so the
collectives alone cost the same in bf16; what shrinks is the residual add and
both RMSNorms (1/TP of the tokens each) and, with an fp8 gather
(TPU_MINIMAX_M3_SP_GATHER / _ATTN_GATHER = fp8), half of every all-gather's
bytes. The embedding's all-reduce becomes a reduce-scatter of the
vocab-parallel partials (exact: one rank holds each token's row), and the
output norm runs on the local tokens before one all-gather.

The branch is decided from the token count while tracing, which leaves a shape
guard, so the backbone warmup gives each side its own trace
(`compilation.shape_variants`). Both sides read the same parameters and
buffers in the same order, so a bucket handed between the traces keeps its
input signature.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, get_ep_group
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

FP8_MAX = 448.0
_SCALE_FLOOR = 1e-12


@dataclasses.dataclass(frozen=True)
class _Ops:
    n: int
    reduce_scatter: Any
    # The bf16 reduce-scatter (the embedding's, which is exact).
    reduce_scatter_exact: Any
    # The Pallas all-reduce's rank tensor when that op is armed, else None.
    rank: torch.Tensor | None
    all_gather: Any
    all_gather_fp8: Any


_OPS: _Ops | None = None


def active(num_tokens: int, min_tokens: int, tp_size: int) -> bool:
    """Whether a step of `num_tokens` takes the sequence-parallel stream."""
    if min_tokens <= 0 or tp_size <= 1 or _OPS is None:
        return False
    # Only the backbone warmup gives each side of this branch its own trace
    # (`shape_variants.aot_cache_tag` is set there); any other compile of the
    # model keeps the all-reduce stream.
    if torch.compiler.is_compiling() and shape_variants.aot_cache_tag() is None:
        return False
    return num_tokens >= min_tokens and num_tokens % tp_size == 0


def _jax_op(name: str, body, mesh, fake, with_rank: bool = False, body_mesh=None):
    """A torch op running `body` per rank. with_rank: the op also takes the
    Pallas all-reduce's rank operand ([1, 1] int32, unused by `body`).

    body_mesh: a mesh over the same devices in the same order with more axes
    (`_chip_mesh`) for `body`'s collectives; torch_tpu still sees `mesh`."""
    axis = mesh.axis_names[0]
    specs = (P(axis), P(axis)) if with_rank else (P(axis),)
    inner = mesh if body_mesh is None else body_mesh
    inner_spec = P(tuple(inner.axis_names), None)
    inner_specs = (inner_spec,) * len(specs)

    def per_rank(x, *_rank):
        return body(x)

    sharded = jax.shard_map(
        per_rank,
        mesh=inner,
        in_specs=inner_specs,
        out_specs=inner_spec,
        check_vma=False,
    )

    def put(args):
        return (
            jax.device_put(a, NamedSharding(inner, s)) for a, s in zip(args, inner_specs)
        )

    def call(*args):
        out = sharded(*put(args))
        if body_mesh is not None:
            out = jax.device_put(out, NamedSharding(mesh, P(axis)))
        return out

    # torch_tpu reads the operand types from these annotations.
    if with_rank:

        def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
            return call(x, rank)

    else:

        def fn(x: jax.Array) -> jax.Array:
            return call(x)

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, *_rank: fake(x))
    return op


def _chip_mesh(mesh):
    """`mesh` as a (chip, core) mesh, the same devices in the same order, when
    each chip's cores sit next to each other in it; else None."""
    import numpy as np
    from jax.sharding import Mesh

    from vllm_torchtpu.distributed.chip_topology import get_chip_topology

    topo = get_chip_topology()
    if topo is None or topo.cores_per_chip < 2:
        return None
    cpc = topo.cores_per_chip
    chip_of = {d: i for i, ids in enumerate(topo.chips) for d in ids}
    devices = list(mesh.devices.flat)
    if len(devices) % cpc:
        return None
    for i in range(0, len(devices), cpc):
        if len({chip_of.get(int(d.id)) for d in devices[i : i + cpc]}) != 1:
            return None
    return Mesh(
        np.asarray(devices).reshape(len(devices) // cpc, cpc),
        axis_names=("chip", "core"),
    )


def _make_ops(mesh, rank: torch.Tensor | None, chip_mesh=None) -> _Ops:
    n = int(mesh.size)
    scatter_fp8 = envs.TPU_MINIMAX_M3_SP_SCATTER == "fp8" and chip_mesh is None
    if chip_mesh is not None:
        # Reduce over the cores of a chip first (half the bytes stay on the
        # chip), then over the chips; the gather undoes it in reverse, so the
        # token slices come back in order.
        def rs(x):
            x = lax.psum_scatter(x, "core", scatter_dimension=0, tiled=True)
            return lax.psum_scatter(x, "chip", scatter_dimension=0, tiled=True)

        def ag(x):
            x = lax.all_gather(x, "chip", axis=0, tiled=True)
            return lax.all_gather(x, "core", axis=0, tiled=True)

    else:
        axis = mesh.axis_names[0]

        def rs(x):
            return lax.psum_scatter(x, axis, scatter_dimension=0, tiled=True)

        def ag(x):
            return lax.all_gather(x, axis, axis=0, tiled=True)

    axis1 = mesh.axis_names[0]
    wire = {"u8": jnp.uint8, "s8": jnp.int8, "f8": None}[envs.TPU_MINIMAX_M3_SP_FP8_WIRE]

    def ag8(x):
        xf = x.astype(jnp.float32)
        scale = jnp.maximum(jnp.max(jnp.abs(xf), axis=-1, keepdims=True), _SCALE_FLOOR)
        scale = scale / FP8_MAX
        q = (xf / scale).astype(jnp.float8_e4m3fn)
        if wire is None:
            q = ag(q)
        else:
            q = lax.bitcast_convert_type(ag(lax.bitcast_convert_type(q, wire)), q.dtype)
        scale = ag(scale)
        return (q.astype(jnp.float32) * scale).astype(x.dtype)

    def _fp8_rows(xf):
        scale = jnp.maximum(jnp.max(jnp.abs(xf), axis=-1, keepdims=True), _SCALE_FLOOR)
        scale = scale / FP8_MAX
        return (xf / scale).astype(jnp.float8_e4m3fn), scale

    def rs8(x):
        # [T, H] partial -> [n, T / n, H]: piece j goes to rank j, which sums
        # the n pieces it receives (the psum_scatter's slice order).
        t, h = x.shape
        q, scale = _fp8_rows(x.astype(jnp.float32).reshape(n, t // n, h))
        q = lax.bitcast_convert_type(
            lax.all_to_all(
                lax.bitcast_convert_type(q, jnp.int8), axis1, 0, 0, tiled=True
            ),
            jnp.float8_e4m3fn,
        )
        scale = lax.all_to_all(scale, axis1, 0, 0, tiled=True)
        return jnp.sum(q.astype(jnp.float32) * scale, axis=0).astype(x.dtype)

    def fake_rs(x):
        return x.new_empty((x.shape[0] // n, *x.shape[1:]))

    def fake_ag(x):
        return x.new_empty((x.shape[0] * n, *x.shape[1:]))

    def rs_op(name, body):
        return _jax_op(
            name, body, mesh, fake_rs, with_rank=rank is not None, body_mesh=chip_mesh
        )

    exact = rs_op("minimax_sp::reduce_scatter", rs)
    return _Ops(
        n=n,
        reduce_scatter=rs_op("minimax_sp::reduce_scatter_fp8", rs8)
        if scatter_fp8
        else exact,
        reduce_scatter_exact=exact,
        rank=rank,
        all_gather=_jax_op(
            "minimax_sp::all_gather", ag, mesh, fake_ag, body_mesh=chip_mesh
        ),
        all_gather_fp8=_jax_op(
            "minimax_sp::all_gather_fp8", ag8, mesh, fake_ag, body_mesh=chip_mesh
        ),
    )


def arm() -> None:
    """Build the stream's collectives over the TP group. Collective over the
    TP group; runs once at the end of TPUModelRunner.load_model, outside any
    trace (the mesh is built from a gather of device ids)."""
    global _OPS
    tp = parallel_state.get_tp_group()
    if tp.world_size == 1:
        return
    ep = get_ep_group()
    tp_ranks = tuple(int(r) for r in tp.ranks)
    if ep is None or tp_ranks != tuple(int(r) for r in ep.ranks):
        raise ValueError(
            "TPU_MINIMAX_M3_SP_MIN_TOKENS needs an EP group over the TP ranks "
            f"(TP {tp_ranks}, EP {None if ep is None else tuple(ep.ranks)})"
        )
    if len(tp_ranks) != torch.distributed.get_world_size():
        raise ValueError("TPU_MINIMAX_M3_SP_MIN_TOKENS needs TP over the whole world")
    mesh = build_ep_mesh()
    if mesh is None:
        raise ValueError("TPU_MINIMAX_M3_SP_MIN_TOKENS: no EP mesh")
    # The all-reduce stream lifts the Pallas all-reduce's rank tensor as a
    # graph input (it is that op's operand); the reduce-scatter takes it too,
    # so both traces of the backbone have the same inputs and a token bucket
    # can be handed between them (`shape_variants`).
    from vllm_torchtpu.distributed import pallas_all_reduce

    armed = pallas_all_reduce._ARMED
    rank = armed.rank if armed is not None and armed.ranks == tp_ranks else None
    chip_mesh = None
    if envs.TPU_MINIMAX_M3_SP_HIERARCHICAL:
        chip_mesh = _chip_mesh(mesh)
        if chip_mesh is None:
            logger.warning(
                "TPU_MINIMAX_M3_SP_HIERARCHICAL: the TP mesh does not hold each "
                "chip's cores together; flat collectives."
            )
    _OPS = _make_ops(mesh, rank, chip_mesh)
    logger.info(
        "TPU_MINIMAX_M3_SP_MIN_TOKENS: sequence-parallel collectives armed over "
        "%d ranks (TPU_MINIMAX_M3_SP_HIERARCHICAL=%d, "
        "TPU_MINIMAX_M3_SP_SCATTER=%s, TPU_MINIMAX_M3_SP_FP8_WIRE=%s, mesh %s).",
        _OPS.n,
        int(chip_mesh is not None),
        envs.TPU_MINIMAX_M3_SP_SCATTER,
        envs.TPU_MINIMAX_M3_SP_FP8_WIRE,
        dict((mesh if chip_mesh is None else chip_mesh).shape),
    )


def reduce_scatter(x: torch.Tensor, exact: bool = False) -> torch.Tensor:
    """A 1/TP token slice of the TP sum of `x`: [T, H] -> [T / tp, H]. The
    slices are in mesh order, the order `all_gather` puts them back in.
    exact: always the bf16 reduce-scatter (TPU_MINIMAX_M3_SP_SCATTER ignored)."""
    assert _OPS is not None
    op = _OPS.reduce_scatter_exact if exact else _OPS.reduce_scatter
    if _OPS.rank is not None:
        return op(x, _OPS.rank)
    return op(x)


def all_gather(x: torch.Tensor, fp8: bool, num_tokens: int) -> torch.Tensor:
    """Every rank's slice, in mesh order: [T / tp, H] -> [T, H] in x.dtype.

    fp8: each row goes over the wire as e4m3 with one f32 scale (amax / 448)
    and is expanded back after the gather.

    num_tokens: T, the step's token count (`input_ids.shape[0]`). The gather's
    output is viewed with it so the traced size is the step's own symbol, not
    tp * (T // tp): DCP code that splits the token dim (`gather_group_kv` at
    DCP 8) cannot hand a derived size to a torch collective.
    """
    assert _OPS is not None
    out = (_OPS.all_gather_fp8 if fp8 else _OPS.all_gather)(x)
    return out.view(num_tokens, *x.shape[1:])


def embed_partial(embed: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """The TPU VocabParallelEmbedding forward without its TP all-reduce: rows
    of tokens outside this rank's vocab shard are zero."""
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    masked, mask = vpe.get_masked_input_and_mask(
        input_ids,
        embed.org_vocab_start_index,
        embed.org_vocab_end_index,
        embed.num_org_vocab_padding,
        embed.added_vocab_start_index,
        embed.added_vocab_end_index,
    )
    out = embed.quant_method.embedding(embed, masked.long())
    out.masked_fill_(mask.unsqueeze(-1), 0)
    return out
