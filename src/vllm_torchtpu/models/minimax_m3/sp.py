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
"""Sequence-parallel (SP) steps of MiniMax M3 (TPU_MINIMAX_M3_SP_MODE).

A step of at least TPU_MINIMAX_M3_SP_MIN_TOKENS tokens that is a multiple of
the TP size runs one of two layouts (`model.MiniMaxM3DecoderLayer`):

"full": the residual stream stays a 1/TP token shard on every rank. Per layer

    x = all_gather(norm(shard))          # attention input, every token
    shard += reduce_scatter(attn_partial(x))
    x = all_gather(norm(shard))          # FFN input, every token
    shard += reduce_scatter(ffn_partial(x))

"attn": the attention partial plus the residual is reduce-scattered, the FFN
input all-gathered from the normalized shard, and the attention partial rides
the FFN all-reduce.

A reduce-scatter plus an all-gather moves the bytes of one all-reduce; what
shrinks is the residual add and both RMSNorms (1/TP of the tokens) and, with
TPU_MINIMAX_M3_SP_FP8_GATHER / _FP8_ATTN_GATHER, half of every all-gather
(e4m3 rows with one f32 scale each; lossy).

The collectives are JAX ops over the EP mesh, armed once after load (`arm`):
torch_tpu cannot compile torch's reduce-scatter on the symbolic token dim
("SymIntArrayRef expected to contain only concrete integers"). The fp8 payload
goes over the wire as int8 (a uint8 payload hangs the SparseCore-offloaded
all-gather; int8 does not). The reduce-scatter also takes the Pallas
all-reduce's rank operand, unused, so the SP trace lifts the same graph inputs
as the plain trace and `compilation.shape_variants` can hand buckets between
them. With TPU_MOE_COMBINE_HIER_RS armed for the TP group the FFN (and with
TPU_MINIMAX_M3_SP_ATTN_HIER_RS the attention) reduce-scatters of "full" steps
run on hierrs_tc (`distributed.hier_reduce_scatter`).

The branch is taken only inside the backbone warmup traces, which give each
side its own trace (the token-count comparison leaves a shape guard).
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.distributed import parallel_state

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MODES = ("off", "attn", "full")
FP8_MAX = 448.0
_SCALE_FLOOR = 1e-12


@dataclasses.dataclass(frozen=True)
class SpConfig:
    """`mode` "off" / "attn" / "full", on steps of at least `min_tokens`
    tokens (multiple of TP); `fp8_gather` / `fp8_attn_gather`: e4m3 FFN /
    attention input all-gathers; `ffn_hier_rs` / `attn_hier_rs`: "full"
    reduce-scatters through hierrs_tc when it is armed for TP."""

    mode: str = "off"
    min_tokens: int = 0
    fp8_gather: bool = False
    fp8_attn_gather: bool = False
    ffn_hier_rs: bool = False
    attn_hier_rs: bool = False

    @property
    def on(self) -> bool:
        return self.mode != "off" and self.min_tokens > 0


OFF = SpConfig()


def config_from_env() -> SpConfig:
    """The TPU_MINIMAX_M3_SP_* (and hierrs) env vars, checked."""
    mode = envs.TPU_MINIMAX_M3_SP_MODE
    min_tokens = envs.TPU_MINIMAX_M3_SP_MIN_TOKENS
    hier = envs.TPU_MOE_COMBINE_HIER_RS
    attn_hier = envs.TPU_MINIMAX_M3_SP_ATTN_HIER_RS
    if attn_hier and not hier:
        raise ValueError(
            "TPU_MINIMAX_M3_SP_ATTN_HIER_RS=1 needs TPU_MOE_COMBINE_HIER_RS=1."
        )
    if mode == "off" or min_tokens == 0:
        return OFF
    ffn_hier = hier and mode == "full"
    return SpConfig(
        mode,
        min_tokens,
        envs.TPU_MINIMAX_M3_SP_FP8_GATHER,
        envs.TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER,
        ffn_hier,
        ffn_hier and attn_hier,
    )


@dataclasses.dataclass(frozen=True)
class _Ops:
    n: int
    reduce_scatter: Any
    # The Pallas all-reduce's rank tensor when that op is armed, else None.
    rank: torch.Tensor | None
    all_gather: Any
    all_gather_fp8: Any
    # hierrs_tc lines up with these ops (its row blocks go by EP rank, ours
    # by mesh index; the same when the mesh is rank-ordered).
    hier_ok: bool = False


_OPS: _Ops | None = None


def armed() -> bool:
    return _OPS is not None


def step(config: SpConfig, num_tokens: int, tp_size: int) -> bool:
    """Whether a step of `num_tokens` takes the sequence-parallel path.

    Only inside a backbone warmup trace (`shape_variants.aot_cache_tag` set
    there), which gives each side of this branch its own trace; eager
    forwards and any other compile keep the plain path.
    """
    return bool(
        config.on
        and tp_size > 1
        and _OPS is not None
        and torch.compiler.is_compiling()
        and shape_variants.aot_cache_tag()
        and num_tokens >= config.min_tokens
        and num_tokens % tp_size == 0
    )


# ---------------------------------------------------------------------------
# Per-rank JAX bodies (also used by the CPU tests)
# ---------------------------------------------------------------------------


def fp8_rows(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """[T, H] -> (e4m3 [T, H], f32 scale [T, 1]); scale = row amax / 448."""
    xf = x.astype(jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(xf), axis=-1, keepdims=True), _SCALE_FLOOR)
    scale = scale / FP8_MAX
    return (xf / scale).astype(jnp.float8_e4m3fn), scale


def from_fp8_rows(q: jax.Array, scale: jax.Array, dtype) -> jax.Array:
    return (q.astype(jnp.float32) * scale).astype(dtype)


def make_bodies(axis: str):
    """(reduce_scatter, all_gather, all_gather_fp8) per-rank bodies over the
    mesh axis `axis`."""

    def rs(x):
        return lax.psum_scatter(x, axis, scatter_dimension=0, tiled=True)

    def ag(x):
        return lax.all_gather(x, axis, axis=0, tiled=True)

    def ag8(x):
        q, scale = fp8_rows(x)
        q = lax.bitcast_convert_type(
            ag(lax.bitcast_convert_type(q, jnp.int8)), jnp.float8_e4m3fn
        )
        return from_fp8_rows(q, ag(scale), x.dtype)

    return rs, ag, ag8


def _jax_op(name: str, body, mesh, fake, with_rank: bool = False):
    """A torch op running `body` per rank over `mesh`. with_rank: the op also
    takes the Pallas all-reduce's rank operand ([1, 1] int32, unused)."""
    from torch_tpu._internal import pallas

    axis = mesh.axis_names[0]
    specs = (P(axis), P(axis)) if with_rank else (P(axis),)

    def per_rank(x, *_rank):
        return body(x)

    sharded = jax.shard_map(
        per_rank, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    def call(*args):
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # torch_tpu reads the operand types (and the single output) from these
    # annotations.
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


def make_ops(mesh, rank: torch.Tensor | None, hier_ok: bool = False) -> _Ops:
    n = int(mesh.size)
    rs, ag, ag8 = make_bodies(mesh.axis_names[0])

    def fake_rs(x):
        return x.new_empty((x.shape[0] // n, *x.shape[1:]))

    def fake_ag(x):
        return x.new_empty((x.shape[0] * n, *x.shape[1:]))

    return _Ops(
        n=n,
        reduce_scatter=_jax_op(
            "minimax_sp::reduce_scatter", rs, mesh, fake_rs, with_rank=rank is not None
        ),
        rank=rank,
        all_gather=_jax_op("minimax_sp::all_gather", ag, mesh, fake_ag),
        all_gather_fp8=_jax_op("minimax_sp::all_gather_fp8", ag8, mesh, fake_ag),
        hier_ok=hier_ok,
    )


def arm(config: SpConfig | None = None) -> bool:
    """Build the SP collectives over the TP group; returns whether armed.
    Collective over the TP group: runs once at the end of
    TPUModelRunner.load_model (after the Pallas all-reduce), outside any
    trace."""
    global _OPS
    config = config_from_env() if config is None else config
    tp = parallel_state.get_tp_group()
    if not config.on or tp.world_size == 1:
        return False
    from vllm_torchtpu.distributed import hier_reduce_scatter, pallas_all_reduce
    from vllm_torchtpu.distributed.ep_mesh import (
        build_ep_mesh,
        ep_rank_order,
        get_ep_group,
    )

    ep = get_ep_group()
    tp_ranks = tuple(int(r) for r in tp.ranks)
    if ep is None or tp_ranks != tuple(int(r) for r in ep.ranks):
        raise ValueError(
            "TPU_MINIMAX_M3_SP_MODE needs an EP group over the TP ranks "
            f"(TP {tp_ranks}, EP {None if ep is None else tuple(ep.ranks)})"
        )
    if len(tp_ranks) != torch.distributed.get_world_size():
        raise ValueError("TPU_MINIMAX_M3_SP_MODE needs TP over the whole world")
    mesh = build_ep_mesh()
    if mesh is None:
        raise ValueError("TPU_MINIMAX_M3_SP_MODE: no EP mesh")
    armed_ar = pallas_all_reduce._ARMED
    rank = (
        armed_ar.rank if armed_ar is not None and armed_ar.ranks == tp_ranks else None
    )
    hier_ok = False
    if config.ffn_hier_rs and hier_reduce_scatter.is_armed_for_tp():
        order = ep_rank_order()
        hier_ok = order is None or tuple(order) == tuple(range(len(order)))
        if not hier_ok:
            logger.warning(
                "TPU_MOE_COMBINE_HIER_RS: the EP mesh is not rank-ordered (%s), "
                "so hierrs_tc's row blocks would not match the SP gathers; the "
                "SP reduce-scatters stay XLA collectives.",
                order,
            )
    _OPS = make_ops(mesh, rank, hier_ok)
    logger.info(
        "MiniMax M3 sequence parallelism armed over %d ranks: "
        "TPU_MINIMAX_M3_SP_MODE=%s TPU_MINIMAX_M3_SP_MIN_TOKENS=%d "
        "TPU_MINIMAX_M3_SP_FP8_GATHER=%d TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER=%d "
        "hierrs reduce-scatter ffn=%d attn=%d (rank operand %s).",
        _OPS.n,
        config.mode,
        config.min_tokens,
        int(config.fp8_gather),
        int(config.fp8_attn_gather),
        int(config.ffn_hier_rs and hier_ok),
        int(config.attn_hier_rs and hier_ok),
        "on" if rank is not None else "off",
    )
    return True


def reduce_scatter(x: torch.Tensor, hier: bool = False) -> torch.Tensor:
    """This rank's 1/TP token slice of the TP sum of `x`: [T, H] -> [T/tp, H],
    in mesh order (the order `all_gather` puts them back in). hier: on
    hierrs_tc when it is armed for TP and lines up (else the same as off)."""
    assert _OPS is not None
    if hier and _OPS.hier_ok:
        from vllm_torchtpu.distributed import hier_reduce_scatter as hrs

        hier_op = hrs._ARMED
        # A call the kernel cannot take keeps this module's reduce-scatter
        # (not the group's torch collective, which cannot take the symbolic
        # token dim on torch_tpu).
        if (
            hier_op is not None
            and hier_op.scope == "tp"
            and hrs._ineligible_reason(len(hier_op.ranks), x) is None
        ):
            return hier_op.op(x)
    if _OPS.rank is not None:
        return _OPS.reduce_scatter(x, _OPS.rank)
    return _OPS.reduce_scatter(x)


def all_gather(x: torch.Tensor, fp8: bool, num_tokens) -> torch.Tensor:
    """Every rank's slice, in mesh order: [T/tp, H] -> [T, H] in x.dtype.

    fp8: each row goes over the wire as e4m3 (int8 bits) with one f32 scale.
    num_tokens: T, the step's token count symbol (`input_ids.shape[0]`). The
    output is viewed with it so the traced size is the step's own symbol, not
    tp * (T // tp): DCP code that splits the token dim (`gather_group_kv` at
    DCP 8) cannot hand a derived size to a torch collective.
    """
    assert _OPS is not None
    out = (_OPS.all_gather_fp8 if fp8 else _OPS.all_gather)(x)
    return out.view(num_tokens, *x.shape[1:])
