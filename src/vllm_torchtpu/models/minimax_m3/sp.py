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


def _jax_op(name: str, body, mesh, fake):
    axis = mesh.axis_names[0]
    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=(P(axis),), out_specs=P(axis), check_vma=False
    )

    # torch_tpu reads the operand types from this annotation.
    def fn(x: jax.Array) -> jax.Array:
        return sharded(jax.device_put(x, NamedSharding(mesh, P(axis))))

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=(P(axis),),
        output_partition_specs=P(axis),
    )
    op.register_fake(fake)
    return op


def _make_ops(mesh) -> _Ops:
    axis = mesh.axis_names[0]
    n = int(mesh.size)

    def rs(x):
        return lax.psum_scatter(x, axis, scatter_dimension=0, tiled=True)

    def ag(x):
        return lax.all_gather(x, axis, axis=0, tiled=True)

    def ag8(x):
        xf = x.astype(jnp.float32)
        scale = jnp.maximum(jnp.max(jnp.abs(xf), axis=-1, keepdims=True), _SCALE_FLOOR)
        scale = scale / FP8_MAX
        q = (xf / scale).astype(jnp.float8_e4m3fn)
        q = lax.all_gather(lax.bitcast_convert_type(q, jnp.uint8), axis, axis=0, tiled=True)
        q = lax.bitcast_convert_type(q, jnp.float8_e4m3fn)
        scale = lax.all_gather(scale, axis, axis=0, tiled=True)
        return (q.astype(jnp.float32) * scale).astype(x.dtype)

    def fake_rs(x):
        return x.new_empty((x.shape[0] // n, *x.shape[1:]))

    def fake_ag(x):
        return x.new_empty((x.shape[0] * n, *x.shape[1:]))

    return _Ops(
        n=n,
        reduce_scatter=_jax_op("minimax_sp::reduce_scatter", rs, mesh, fake_rs),
        all_gather=_jax_op("minimax_sp::all_gather", ag, mesh, fake_ag),
        all_gather_fp8=_jax_op("minimax_sp::all_gather_fp8", ag8, mesh, fake_ag),
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
    _OPS = _make_ops(mesh)
    logger.info(
        "TPU_MINIMAX_M3_SP_MIN_TOKENS: sequence-parallel collectives armed over "
        "%d ranks.",
        _OPS.n,
    )


def reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """A 1/TP token slice of the TP sum of `x`: [T, H] -> [T / tp, H]. The
    slices are in mesh order, the order `all_gather` puts them back in."""
    assert _OPS is not None
    return _OPS.reduce_scatter(x)


def all_gather(x: torch.Tensor, fp8: bool) -> torch.Tensor:
    """Every rank's slice, in mesh order: [T / tp, H] -> [T, H] in x.dtype.

    fp8: each row goes over the wire as e4m3 with one f32 scale (amax / 448)
    and is expanded back after the gather.
    """
    assert _OPS is not None
    return (_OPS.all_gather_fp8 if fp8 else _OPS.all_gather)(x)


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
