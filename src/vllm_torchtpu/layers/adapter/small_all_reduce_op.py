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
"""Small TP all-reduces and DP MoE combines through the Pallas kernels in
``kernels/experimental/small_collectives``.

``prebuild`` builds both ops over the expert-parallel mesh at model load. The
device communicator then hands a 2-D bf16 block of at most ``MAX_ROWS`` rows
(an all-reduce) or ``COMBINE_MAX_ROWS`` rows per engine (a DP combine) to a
kernel and anything else to vLLM's collective.

The choice reads the token count, so in torch it is taken only in a trace that
the backbone warmup forks per token bucket (``compilation.shape_variants``).
Both sides of the fork lift the same rank buffer, as the warmup requires to
hand a bucket's executable from one trace to the other.

A trace compiled outside that warmup -- the MTP drafter's step-0 and loop-step
programs -- cannot fork: vLLM compiles its one FX graph for every bucket, and a
token-count branch would freeze at the bucket traced first. With
TPU_SMALL_ALL_REDUCE_DRAFT those traces call a second op that takes the choice
inside JAX instead, where each bucket is exported at its concrete shape: the
kernel up to the row cap, XLA's all-reduce (``psum``) above it. Without the
flag, and in eager mode, they keep vLLM's collective.

With TPU_SMALL_ALL_REDUCE_VALID_ROWS a backbone block that has the two-shot
kernel's fold path (``all_reduce.fold_geometry``) is handed the step's
valid-row count from the token-padding mask: up to ``fold_rows`` valid rows
are folded into one row tile and only that moves, padding rows coming out +0;
more valid rows run the plain two-shot. Valid rows are bitwise the plain
kernel's either way. The count is the same on every TP rank (the ranks share
the step's tokens). The drafter traces keep the sized op: the mask describes
the target step's rows.

With TPU_SMALL_ALL_REDUCE_CHIP_PAIRS=<rows> every kernel all-reduce of at
least <rows> rows (up to the row cap) runs the two-shot by chip pairs
(``all_reduce`` form "pairs"): each chip adds its two cores' blocks first and
sends one piece over ICI per chip. Such a block skips the valid-rows
all-reduce. At prebuild every rank checks that mesh indices 2i and 2i + 1 are
the two cores of one chip, and the pairs form runs once at the row cap on
known blocks; any rank's failure raises on every rank.

With TPU_FUSED_OPROJ_AR a row-parallel per-channel FP8 layer served by the
weight-streaming kernel (TPU_FP8_LINEAR_PALLAS) whose output this all-reduce
would sum -- attention's o_proj -- takes ``row_parallel_or`` instead: one kernel
for the matmul and the all-reduce (kernels/quantized_matmul/
fp8_oproj_all_reduce.py), bitwise the two-call path, for blocks of at most
16 rows inside the warmup. Everything else keeps the layer's own forward.
"""

import faulthandler

import jax
import torch
from jax.sharding import NamedSharding, PartitionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    get_ep_group,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.experimental.moe_tp_decode.all_reduce import (
    check_chip_pairs,
)
from vllm_torchtpu.kernels.experimental.small_collectives import (
    all_reduce,
    reduce_scatter,
)
from vllm_torchtpu.kernels.experimental.small_collectives.common import LANES
from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.linear_common import (
    FP8_PALLAS_N_ATTR,
    FP8_PALLAS_WEIGHT,
    KEEP_VLLM_LAYOUT_ATTR,
    PREQUANT_ATTR,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Largest TP all-reduce, in rows, the kernel takes: at H=4096 bf16 on 8 cores
# two-shot beats XLA through 96 rows, loses from 128 (j-e6f25e61, j-de6349e8).
MAX_ROWS = 96
# Largest DP combine, in rows per engine, the reduce-scatter takes: it beats
# XLA through 24 rows, ties at 32 and loses at 64 (j-de6349e8).
COMBINE_MAX_ROWS = 24
# A wrong rank or mesh hangs the chip-pair self-test: dump every stack and exit.
SELFTEST_TIMEOUT_S = 300


def max_rows() -> int:
    """The all-reduce row cap: TPU_SMALL_ALL_REDUCE_MAX_ROWS, else MAX_ROWS."""
    return envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS or MAX_ROWS


_AR_OP = None
# The all-reduce for traces outside the backbone warmup: the row choice is
# made per exported bucket inside JAX (TPU_SMALL_ALL_REDUCE_DRAFT).
_AR_SIZED_OP = None
_RS_OP = None
# The valid-rows all-reduce (TPU_SMALL_ALL_REDUCE_VALID_ROWS) and the bf16 row
# tile its fold path folds into.
_AR_VALID_OP = None
_VALID_TILE_ROWS = 0
# This core's mesh index as an int32 [1] device tensor: the kernels take it as
# data (see small_collectives/common.py).
_RANK = None
_CORES = 0
_MESH_RANKS: tuple[int, ...] = ()
# Blocks of at least this many rows take the chip-pair two-shot (0: none).
_PAIR_ROWS = 0
# Whether the DP group is the mesh, so its combine may take the kernel.
_DP_COMBINE = False
# The fused row-parallel FP8 matmul + all-reduce (TPU_FUSED_OPROJ_AR); built only
# when the TP group is the mesh (the only group it sums over).
_MM_AR_OP = None


def _build_all_reduce(mesh, psum_above: int | None = None, pair_rows: int = 0):
    """The all-reduce op over ``mesh``. With ``psum_above``, a block of more
    rows than that is summed by XLA's all-reduce instead of the kernel: the
    block's rows are concrete when JAX exports each bucket, so the choice
    leaves no guard on the caller's trace. Blocks of at least ``pair_rows``
    rows take the chip-pair two-shot (0: none)."""
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank):
        if psum_above is not None and block.shape[0] > psum_above:
            return jax.lax.psum(block, EP_AXIS_NAME)
        return all_reduce.small_all_reduce(
            block, num_cores=n, axis=EP_AXIS_NAME, rank=rank, pair_rows=pair_rows
        )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )(x, rank)

    def fake(x, rank):
        return torch.empty_like(x)

    return sharded_jax_op(
        "pallas::small_all_reduce"
        if psum_above is None
        else "pallas::small_all_reduce_or_psum",
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec),
        output_partition_specs=spec,
        fake=fake,
    )


def _build_valid_rows_all_reduce(mesh, tile_rows: int):
    """The all-reduce op over ``mesh`` that moves only a block's valid rows:
    fn(x [rows, hidden], n_valid int32 [1], rank int32 [1]) per core."""
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, n_valid, rank):
        return all_reduce.small_all_reduce_valid_rows(
            block, n_valid, rank, num_cores=n, axis=EP_AXIS_NAME, tile_rows=tile_rows
        )

    def fn(x: jax.Array, n_valid: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )(x, n_valid, rank)

    def fake(x, n_valid, rank):
        return torch.empty_like(x)

    return sharded_jax_op(
        "pallas::small_all_reduce_valid_rows",
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec, spec),
        output_partition_specs=spec,
        fake=fake,
    )


def _build_matmul_all_reduce(mesh, waves: bool):
    """The fused op over ``mesh``: fn(x [rows, k] bf16, w [k, n] fp8, w_s [n] f32,
    rank int32 [1]) per core -> the sum over the cores of x @ w * scales, [rows, n]."""
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(x, w, w_s, rank):
        return fp8_oproj_all_reduce.fp8_matmul_all_reduce(
            x, w, w_s, rank, num_cores=n, axis=EP_AXIS_NAME, waves=waves
        )

    def fn(x: jax.Array, w: jax.Array, w_s: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 4, out_specs=spec, check_vma=False
        )(x, w, w_s, rank)

    def fake(x, w, w_s, rank):
        return x.new_empty((x.shape[0], w.shape[1]))

    return sharded_jax_op(
        "pallas::fp8_matmul_all_reduce",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 4,
        output_partition_specs=spec,
        fake=fake,
    )


def _build_reduce_scatter(mesh):
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank):
        return reduce_scatter.small_reduce_scatter(
            block, num_cores=n, axis=EP_AXIS_NAME, rank=rank
        )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )(x, rank)

    def fake(x, rank):
        return x.new_empty((x.shape[0] // n, x.shape[1]))

    return sharded_jax_op(
        "pallas::small_reduce_scatter",
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec),
        output_partition_specs=spec,
        fake=fake,
    )


def _gather(obj) -> list:
    """all_gather_object over the EP group's CPU group, in group rank order."""
    group = get_ep_group()
    assert group is not None
    seen = [None] * int(group.world_size)
    torch.distributed.all_gather_object(seen, obj, group=group.cpu_group)
    return seen


def _chip_places(mesh, mesh_index: int) -> list[tuple[tuple[int, ...], int | None]]:
    """(chip coords, core on chip) at each index of ``mesh``. Its devices
    carry ids only (get_pallas_mesh builds them from the process group), so
    every rank reports its own JAX device, the one TPU device its process
    addresses, and it must be the mesh's device at the rank's index.
    Collective."""
    device = jax.local_devices()[0]
    mine = dict(
        mesh_index=mesh_index,
        id=int(device.id),
        coords=tuple(getattr(device, "coords", None) or ()),
        core=getattr(device, "core_on_chip", None),
    )
    seen = sorted(_gather(mine), key=lambda s: s["mesh_index"])
    ids = [int(d.id) for d in mesh.devices.flat]
    if [(s["mesh_index"], s["id"]) for s in seen] != list(enumerate(ids)):
        raise RuntimeError(
            f"small all-reduce: the ranks' JAX devices {seen} are not the "
            f"mesh's devices {ids} in mesh order"
        )
    return [(s["coords"], s["core"]) for s in seen]


def _selftest_pairs(mesh, mesh_index: int, rows: int, hidden: int) -> bool:
    """Whether the chip-pair two-shot, run once through ``mesh`` and the rank
    operand on all_reduce.selftest_blocks, gives this rank the exact sum.
    Collective: every rank makes the same call."""
    from torch_tpu._internal import pallas

    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)
    sharding = NamedSharding(mesh, spec)

    def body(block, rank):
        return all_reduce.small_all_reduce(
            block, num_cores=n, axis=EP_AXIS_NAME, rank=rank, form="pairs"
        )

    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
    )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return sharded(jax.device_put(x, sharding), jax.device_put(rank, sharding))

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::small_all_reduce_pairs_selftest",
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec),
        output_partition_specs=spec,
    )
    op.register_fake(lambda x, rank: torch.empty_like(x))
    blocks = all_reduce.selftest_blocks(n, rows, hidden)
    mine = torch.from_numpy(blocks[mesh_index]).to(torch.bfloat16).to(_RANK.device)
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        out = op(mine, _RANK).to(torch.float32).cpu().numpy()
    finally:
        faulthandler.cancel_dump_traceback_later()
    return all_reduce.selftest_ok(blocks, out)


def _arm_chip_pairs(mesh, mesh_index: int, hidden: int) -> int:
    """TPU_SMALL_ALL_REDUCE_CHIP_PAIRS: the rows from which blocks take the
    chip-pair two-shot, once the mesh pairs every chip's two cores and the
    form gives every rank the exact sum at the row cap. Collective; raises on
    every rank when any rank fails."""
    pair_rows = envs.TPU_SMALL_ALL_REDUCE_CHIP_PAIRS
    if hidden <= 0 or hidden % (_CORES * LANES):
        raise ValueError(
            f"TPU_SMALL_ALL_REDUCE_CHIP_PAIRS: hidden size {hidden} is not a "
            f"multiple of {_CORES} cores x {LANES} lanes"
        )
    places = _chip_places(mesh, mesh_index)
    check_chip_pairs(places)
    rows = max_rows()
    ok = _selftest_pairs(mesh, mesh_index, rows, hidden)
    seen = _gather(dict(mesh_index=mesh_index, ok=ok))
    if not all(s["ok"] for s in seen):
        raise RuntimeError(
            "small all-reduce: the chip-pair self-test at "
            f"{rows} x {hidden} failed (per rank): {seen}"
        )
    ids = [int(d.id) for d in mesh.devices.flat]
    logger.info(
        "Small all-reduce by chip pairs from %d rows; self-test at %d x %d "
        "passed; pairs (mesh indices: device ids @ chip coords): %s",
        pair_rows,
        rows,
        hidden,
        "; ".join(
            f"{i},{i + 1}: {ids[i]},{ids[i + 1]} @ {places[i][0]}"
            for i in range(0, len(ids), 2)
        ),
    )
    return pair_rows


def prebuild(hidden_size: int = 0) -> None:
    """Build the ops and this core's rank tensor, outside any traced region:
    the mesh comes from a collective. A no-op unless TPU_SMALL_ALL_REDUCE.
    hidden_size is the model's, the width of the chip-pair self-test."""
    global _AR_OP, _AR_SIZED_OP, _RS_OP, _RANK, _CORES, _MESH_RANKS, _DP_COMBINE
    global _AR_VALID_OP, _VALID_TILE_ROWS, _MM_AR_OP, _PAIR_ROWS
    if not envs.TPU_SMALL_ALL_REDUCE or _AR_OP is not None:
        return
    mesh = build_ep_mesh()
    if mesh is None and not envs.is_set("TPU_SMALL_ALL_REDUCE"):
        # On by default; without an EP mesh every all-reduce stays on XLA's.
        logger.info_once("Small collectives off: no expert-parallel mesh.")
        return
    if mesh is None:
        raise NotImplementedError(
            "TPU_SMALL_ALL_REDUCE=1 builds its kernels over the expert-parallel "
            "mesh, and this deployment has no expert-parallel group (a dense "
            "model or one core). Unset TPU_SMALL_ALL_REDUCE."
        )
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    _CORES = mesh.shape[EP_AXIS_NAME]
    ep_group = get_ep_group()
    assert ep_group is not None  # build_ep_mesh() returned a mesh over it
    _MESH_RANKS = tuple(int(r) for r in ep_group.ranks)
    mesh_index = ep_mesh_index()
    assert mesh_index is not None
    _RANK = torch.tensor([mesh_index], dtype=torch.int32, device="tpu")
    if envs.TPU_SMALL_ALL_REDUCE_CHIP_PAIRS:
        _PAIR_ROWS = _arm_chip_pairs(mesh, mesh_index, hidden_size)
    _AR_OP = _build_all_reduce(mesh, pair_rows=_PAIR_ROWS)
    if envs.TPU_SMALL_ALL_REDUCE_DRAFT:
        _AR_SIZED_OP = _build_all_reduce(
            mesh, psum_above=max_rows(), pair_rows=_PAIR_ROWS
        )
    _RS_OP = _build_reduce_scatter(mesh)
    if envs.TPU_SMALL_ALL_REDUCE_VALID_ROWS:
        from jax.experimental.pallas import tpu as pltpu

        info = pltpu.get_tpu_info()
        _VALID_TILE_ROWS = int(info.get_sublane_tiling(jax.numpy.bfloat16))
        _AR_VALID_OP = _build_valid_rows_all_reduce(mesh, _VALID_TILE_ROWS)
    # The mesh is rank-ordered (ep_mesh.build_ep_mesh), so when the DP group
    # is the mesh, DP rank r's rows go to mesh core r.
    _DP_COMBINE = tuple(int(r) for r in get_dp_group().ranks) == _MESH_RANKS
    tp_all_reduce = tuple(int(r) for r in get_tp_group().ranks) == _MESH_RANKS
    if envs.TPU_FUSED_OPROJ_AR and tp_all_reduce:
        _MM_AR_OP = _build_matmul_all_reduce(mesh, envs.TPU_FUSED_OPROJ_AR_WAVES)
    logger.info(
        "Small collectives armed | cores=%d | max_rows=%d | "
        "combine_max_rows=%d | one_shot_max_bytes=%d | tp_all_reduce=%s | "
        "dp_combine=%s | outside_warmup=%s | valid_rows=%s | fused_oproj_ar=%s | "
        "chip_pairs=%s",
        _CORES,
        max_rows(),
        COMBINE_MAX_ROWS,
        all_reduce.ONE_SHOT_MAX_BYTES,
        tp_all_reduce,
        _DP_COMBINE,
        "kernel to max_rows, psum above (MTP drafter)"
        if _AR_SIZED_OP is not None
        else "vLLM's collectives (MTP drafter, eager)",
        f"fold into {_VALID_TILE_ROWS}-row tiles (needs the token-padding mask)"
        if _AR_VALID_OP is not None
        else "off",
        _MM_AR_OP is not None,
        f"from {_PAIR_ROWS} rows" if _PAIR_ROWS else "off",
    )


def _fits(x: torch.Tensor, width_multiple: int, in_warmup: bool = True) -> bool:
    """Static checks: whether this call site can take a kernel at all."""
    return (
        torch.compiler.is_compiling()
        and shape_variants.in_warmup() == in_warmup
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[1] % width_multiple == 0
    )


def _padding_mask() -> torch.Tensor | None:
    """The token-padding mask the valid-rows all-reduce reads, or None."""
    if _AR_VALID_OP is None or token_padding._padding_state is None:
        return None
    return token_padding._padding_state.local_padding_mask


def retain(out: torch.Tensor) -> torch.Tensor:
    """``out`` plus a zero read from the rank buffer (and, with the valid-rows
    all-reduce, the padding mask), for a token-count branch that skips the
    kernel while its other side calls it: the warmup hands a bucket over only
    between traces that lift the same buffers. A read whose result is dead
    would be eliminated, so the zero rides on ``out``."""
    if _RANK is None or not shape_variants.in_warmup():
        return out
    value = _RANK[0].to(out.dtype)
    mask = _padding_mask()
    if mask is not None:
        value = value + mask[0].to(out.dtype)
    return out + (value - value)


def _valid_rows(rows: int, mask: torch.Tensor) -> torch.Tensor:
    """int32 [1], the valid rows of this block: the mask marks rows
    [n_valid, padded) as padding, the same on every TP rank."""
    padding = mask[:rows].to(torch.int32).sum()
    return (rows - padding).reshape(1).to(torch.int32)


def _small_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """A block of at most max_rows rows inside the warmup: the valid-rows
    all-reduce when armed and the block has a fold path and is below the
    chip-pair rows, else the kernel _AR_OP picks (with the mask read, so every
    bucket lifts the same buffers)."""
    assert _AR_OP is not None, "init_small_all_reduce builds the ops first"
    mask = _padding_mask()
    if mask is None:
        return _AR_OP(x, _RANK)
    rows, hidden = x.shape
    # The cap rule (envs.bucket_gate_edge): `rows >= _PAIR_ROWS` with the
    # caller's `rows <= max_rows()` would pin the traced token count when the
    # pair rows are the row cap (64 and 64: ConstraintViolationError).
    pairs = _PAIR_ROWS > 0 and rows > envs.bucket_gate_edge(_PAIR_ROWS)
    if (
        not pairs
        and all_reduce.fold_geometry(rows, hidden, _CORES, _VALID_TILE_ROWS) is not None
    ):
        assert _AR_VALID_OP is not None
        return _AR_VALID_OP(x, _valid_rows(rows, mask), _RANK)
    return _AR_OP(retain(x), _RANK)


def all_reduce_or(x: torch.Tensor, ranks, fallback) -> torch.Tensor:
    """The sum of ``x`` over the group ``ranks``: the kernel for a small block
    when the group is the mesh, ``fallback(x)`` otherwise."""
    if _AR_OP is None or tuple(ranks) != _MESH_RANKS:
        return fallback(x)
    if _AR_SIZED_OP is not None and _fits(x, _CORES * LANES, in_warmup=False):
        # A trace outside the warmup: the op picks kernel or psum per bucket.
        return _AR_SIZED_OP(x, _RANK)
    if not _fits(x, _CORES * LANES):
        return fallback(x)
    if x.shape[0] <= max_rows():
        return _small_all_reduce(x)
    if envs.TPU_FUSED_ADD_RMSNORM:
        # The zero rides on the all-reduce INPUT, where XLA fuses it into the
        # producer's epilogue (sum of zeros is zero). On the output it cannot
        # fuse into the Pallas RMSNorm that reads it, and costs a full pass
        # (plus a relayout to the kernel's operand tiling).
        return fallback(retain(x))
    return retain(fallback(x))


def dp_combine_or(x: torch.Tensor, fallback) -> torch.Tensor:
    """The DP MoE combine of ``x`` ([dp * rows, hidden], block r for DP rank
    r): the kernel for a small block when the DP group is the mesh,
    ``fallback(x)`` otherwise."""
    if _RS_OP is None or not _DP_COMBINE or not _fits(x, LANES):
        return fallback(x)
    if x.shape[0] <= _CORES * COMBINE_MAX_ROWS:
        return _RS_OP(x, _RANK)
    return retain(fallback(x))


def _fused_layer_weights(layer):
    """(w [k, n] fp8, w_s [n] f32) when ``layer`` is a per-channel FP8 layer on the
    weight-streaming kernel with the kernel's own weight (no padded copy, no
    prequant slot, no bias) whose output is all-reduced over the TP group; else None."""
    if (
        getattr(layer, FP8_PALLAS_N_ATTR, None) is None
        or getattr(layer, FP8_PALLAS_WEIGHT, None) is not None
        or getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False)
        or getattr(layer, PREQUANT_ATTR, None) is not None
        or getattr(layer, "bias", None) is not None
        or not getattr(layer, "reduce_results", False)
        or getattr(layer, "tp_size", 1) != _CORES
    ):
        return None
    w, w_s = layer.weight, getattr(layer, "weight_scale", None)
    if w is None or w_s is None or w.dim() != 2 or w_s.dim() != 1:
        return None
    if w.dtype != torch.float8_e4m3fn or w_s.shape[0] != w.shape[1]:
        return None
    return w, w_s


def row_parallel_or(x: torch.Tensor, layer) -> torch.Tensor:
    """``layer(x)[0]`` for a row-parallel layer: the fused FP8 matmul + all-reduce
    kernel for a decode block it supports (TPU_FUSED_OPROJ_AR), the layer's own
    matmul and all-reduce otherwise. Both sides read the rank buffer (and the
    padding mask), so the warmup's two traces lift the same buffers."""
    if _MM_AR_OP is not None and _fits(x, LANES):
        weights = _fused_layer_weights(layer)
        if weights is not None:
            w, w_s = weights
            rows, k = x.shape
            if k == w.shape[0] and fp8_oproj_all_reduce.supports(
                rows, k, w.shape[1], _CORES
            ):
                return _MM_AR_OP(retain(x), w, w_s, _RANK)
    return layer(x)[0]
