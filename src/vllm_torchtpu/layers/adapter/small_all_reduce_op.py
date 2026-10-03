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
"""Small TP/DP/EP/DCP collectives through the Pallas kernels in
``kernels/experimental/small_collectives``.

``prebuild`` builds the four ops (``small_all_reduce``, ``small_reduce_scatter``,
``small_all_gather``, ``small_all_to_all``) over the expert-parallel mesh at
model load. The device communicator and GLM-5.2 / DeepSeek-V4 decode call sites
then hand a 2-D bf16 block of at most ``MAX_ROWS`` / ``ALL_GATHER_MAX_ROWS`` /
``ALL_TO_ALL_MAX_ROWS`` rows (or ``COMBINE_MAX_ROWS`` rows per engine for
reduce-scatter) to a Pallas kernel and anything else to vLLM's collective.

The choice reads the token count, so it is taken only in a trace that the
backbone warmup forks per token bucket (``compilation.shape_variants``). Any
other trace -- the MTP drafter, eager mode, the profiling run -- keeps vLLM's
collective. Both sides of the fork lift the same rank buffer, as the warmup
requires to hand a bucket's executable from one trace to the other.
"""

import os

import jax
import torch
from jax.sharding import PartitionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants

try:
    from vllm_torchtpu.distributed.ep_mesh import (
        EP_AXIS_NAME,
        build_ep_mesh,
        ep_mesh_index,
        get_ep_group,
    )
except ImportError:  # pragma: no cover - CPU unit-test environment without vllm
    EP_AXIS_NAME = "d"
    build_ep_mesh = None  # type: ignore[assignment]
    ep_mesh_index = None  # type: ignore[assignment]
    get_ep_group = None  # type: ignore[assignment]

try:
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
except ImportError:  # pragma: no cover - CPU unit-test environment without torch_tpu
    sharded_jax_op = None  # type: ignore[assignment]

from vllm_torchtpu.kernels.experimental.small_collectives import (
    all_gather,
    all_reduce,
    all_to_all,
    reduce_scatter,
)
from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    round_up,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Largest TP all-reduce, in rows, the kernel takes: at H=4096 bf16 on 8 cores
# two-shot beats XLA through 96 rows, loses from 128 (j-e6f25e61, j-de6349e8).
MAX_ROWS = 96
# Largest DP combine, in rows per engine, the reduce-scatter takes: measured
# crossover between 40 and 48 (j-a8d66cde, j-e6f25e61, j-de6349e8). The win
# relies on XLA lowering psum_scatter to a full all-reduce when rows per engine
# is not a multiple of 16, which depends on libtpu: re-measure on a bump.
COMBINE_MAX_ROWS = 40
# Largest per-rank row count for small_all_gather and small_all_to_all in decode
# and small-chunk prefill (b/568788608, b/568770670, b/568775832).
ALL_GATHER_MAX_ROWS = 96
ALL_TO_ALL_MAX_ROWS = 96

_AR_OP = None
_RS_OP = None
_AG_OP = None
_A2A_OP = None
# This core's mesh index as an int32 [1] device tensor: the kernels take it as
# data (see small_collectives/common.py).
_RANK = None
_CORES = 0
_MESH_RANKS: tuple[int, ...] = ()
# Whether the DP group is the mesh, so its combine may take the kernel.
_DP_COMBINE = False


def _is_enabled() -> bool:
    """Whether small Pallas collectives are enabled via environment flags."""
    if getattr(envs, "TPU_SMALL_ALL_REDUCE", False):
        return True
    if getattr(envs, "TPU_SMALL_COLLECTIVES", False):
        return True
    val = os.getenv("TPU_SMALL_COLLECTIVES", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _build_all_reduce(mesh):
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank):
        return all_reduce.small_all_reduce(
            block, num_cores=n, axis=EP_AXIS_NAME, rank=rank
        )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )(x, rank)

    def fake(x, rank):
        return torch.empty_like(x)

    return sharded_jax_op(
        "pallas::small_all_reduce",
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec),
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


def _build_all_gather(mesh):
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank):
        return all_gather.small_all_gather(
            block, num_cores=n, axis=EP_AXIS_NAME, rank=rank
        )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )(x, rank)

    def fake(x, rank):
        return x.new_empty((n * x.shape[0], x.shape[1]))

    return sharded_jax_op(
        "pallas::small_all_gather",
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec),
        output_partition_specs=spec,
        fake=fake,
    )


def _build_all_to_all(mesh):
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank):
        s_ax = 0 if block.shape[0] % n == 0 else 1
        return all_to_all.small_all_to_all(
            block,
            num_cores=n,
            axis=EP_AXIS_NAME,
            split_axis=s_ax,
            concat_axis=s_ax,
            rank=rank,
        )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )(x, rank)

    def fake(x, rank):
        return torch.empty_like(x)

    return sharded_jax_op(
        "pallas::small_all_to_all",
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec),
        output_partition_specs=spec,
        fake=fake,
    )


def prebuild() -> None:
    """Build the ops and this core's rank tensor, outside any traced region:
    the mesh comes from a collective. A no-op unless TPU_SMALL_ALL_REDUCE or
    TPU_SMALL_COLLECTIVES is enabled."""
    global _AR_OP, _RS_OP, _AG_OP, _A2A_OP, _RANK, _CORES, _MESH_RANKS, _DP_COMBINE
    if not _is_enabled() or _AR_OP is not None:
        return
    mesh = build_ep_mesh()
    if mesh is None:
        raise NotImplementedError(
            "TPU_SMALL_ALL_REDUCE=1 builds its kernels over the expert-parallel "
            "mesh, and this deployment has no expert-parallel group (a dense "
            "model or one core). Unset TPU_SMALL_ALL_REDUCE."
        )
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    _CORES = mesh.shape[EP_AXIS_NAME]
    _MESH_RANKS = tuple(int(r) for r in get_ep_group().ranks)
    _RANK = torch.tensor([ep_mesh_index()], dtype=torch.int32, device="tpu")
    _AR_OP = _build_all_reduce(mesh)
    _RS_OP = _build_reduce_scatter(mesh)
    _AG_OP = _build_all_gather(mesh)
    _A2A_OP = _build_all_to_all(mesh)
    # The mesh is rank-ordered (ep_mesh.build_ep_mesh), so when the DP group
    # is the mesh, DP rank r's rows go to mesh core r.
    _DP_COMBINE = tuple(int(r) for r in get_dp_group().ranks) == _MESH_RANKS
    tp_all_reduce = tuple(int(r) for r in get_tp_group().ranks) == _MESH_RANKS
    logger.info(
        "Small collectives armed | cores=%d | max_rows=%d | "
        "combine_max_rows=%d | all_gather_max_rows=%d | "
        "all_to_all_max_rows=%d | one_shot_max_bytes=%d | tp_all_reduce=%s | "
        "dp_combine=%s | "
        "backbone warmup traces only; the MTP drafter and eager steps keep "
        "vLLM's collectives",
        _CORES,
        MAX_ROWS,
        COMBINE_MAX_ROWS,
        ALL_GATHER_MAX_ROWS,
        ALL_TO_ALL_MAX_ROWS,
        all_reduce.ONE_SHOT_MAX_BYTES,
        tp_all_reduce,
        _DP_COMBINE,
    )


def _fits(x: torch.Tensor, width_multiple: int) -> bool:
    """Static checks: whether this call site can take a kernel at all."""
    return (
        torch.compiler.is_compiling()
        and shape_variants.in_warmup()
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[1] % width_multiple == 0
    )


def _all_reduce_width_multiple(x: torch.Tensor) -> int:
    """Width multiple required for ``small_all_reduce`` on ``x``.

    - ``oneshot`` (payload <= ``ONE_SHOT_MAX_BYTES``) and multi-host meshes
      (``_CORES > 8``, e.g. 16 cores on ``2x2x2`` or 32 cores on ``2x2x4``
      where ``all_reduce.small_all_reduce`` automatically pads columns to
      ``_CORES * LANES`` or runs ``hierarchical``) only require ``LANES = 128``
      alignment, unlocking ``H = 6144`` on 32 cores (``2x2x4``).
    - Single-host ``_CORES <= 8`` in ``twoshot`` mode requires
      ``_CORES * LANES`` so unaligned widths fall back untouched.
    """
    if x.dim() == 2 and x.dtype == torch.bfloat16:
        form = all_reduce.form_for(x.shape[0], x.shape[1], x.element_size())
        if form == "oneshot" or _CORES > 8:
            return LANES
    return max(_CORES, 1) * LANES


def retain(out: torch.Tensor) -> torch.Tensor:
    """``out`` plus a zero read from the rank buffer, for a token-count branch
    that skips the kernel while its other side calls it: the warmup hands a
    bucket over only between traces that lift the same buffers. A read whose
    result is dead would be eliminated, so the zero rides on ``out``."""
    if _RANK is None or not shape_variants.in_warmup():
        return out
    value = _RANK[0].to(out.dtype)
    return out + (value - value)


def all_reduce_or(x: torch.Tensor, ranks, fallback) -> torch.Tensor:
    """The sum of ``x`` over the group ``ranks``: the kernel for a small block
    when the group is the mesh, ``fallback(x)`` otherwise."""
    if _AR_OP is None or tuple(ranks) != _MESH_RANKS:
        return fallback(x)
    if not _fits(x, _all_reduce_width_multiple(x)):
        return fallback(x)
    if x.shape[0] <= MAX_ROWS:
        return _AR_OP(x, _RANK)
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


def maybe_small_all_gather(
    x: torch.Tensor,
    ranks=None,
    fallback=None,
) -> torch.Tensor:
    """All-gather ``x`` ([rows, hidden]) across the mesh when ``rows <= ALL_GATHER_MAX_ROWS``,
    or call ``fallback(x)`` otherwise."""
    if fallback is None:
        raise ValueError("maybe_small_all_gather requires a fallback callable")
    if _AG_OP is None or (ranks is not None and tuple(ranks) != _MESH_RANKS):
        return fallback(x)
    if not _fits(x, LANES):
        return fallback(x)
    if x.shape[0] <= ALL_GATHER_MAX_ROWS:
        return _AG_OP(x, _RANK)
    return retain(fallback(x))


all_gather_or = maybe_small_all_gather


def maybe_small_all_to_all(
    x: torch.Tensor,
    ranks=None,
    fallback=None,
) -> torch.Tensor:
    """All-to-all ``x`` ([rows, hidden]) across the mesh when ``rows <= ALL_TO_ALL_MAX_ROWS``,
    or call ``fallback(x)`` otherwise."""
    if fallback is None:
        raise ValueError("maybe_small_all_to_all requires a fallback callable")
    if _A2A_OP is None or (ranks is not None and tuple(ranks) != _MESH_RANKS):
        return fallback(x)
    if not _fits(x, LANES):
        return fallback(x)
    if _CORES > 0 and (x.shape[0] % _CORES != 0 and x.shape[1] % _CORES != 0):
        return fallback(x)
    if x.shape[0] <= ALL_TO_ALL_MAX_ROWS:
        return _A2A_OP(x, _RANK)
    return retain(fallback(x))


all_to_all_or = maybe_small_all_to_all


def maybe_small_reduce_scatter(
    x: torch.Tensor,
    ranks=None,
    fallback=None,
) -> torch.Tensor:
    """Reduce-scatter ``x`` ([cores * rows, hidden]) across the mesh when
    ``rows <= COMBINE_MAX_ROWS``, or call ``fallback(x)`` otherwise."""
    if fallback is None:
        raise ValueError("maybe_small_reduce_scatter requires a fallback callable")
    if _RS_OP is None or (ranks is not None and tuple(ranks) != _MESH_RANKS):
        return fallback(x)
    if not _fits(x, LANES) or (_CORES > 0 and x.shape[0] % _CORES != 0):
        return fallback(x)
    if x.shape[0] <= _CORES * COMBINE_MAX_ROWS:
        return _RS_OP(x, _RANK)
    return retain(fallback(x))


reduce_scatter_or = maybe_small_reduce_scatter


def fused_q_head_all_gather(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    ranks=None,
    fallback=None,
    *,
    gather_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused single-launch Q-head all-gather for MLA / DCP decode (b/568770670).

    Concatenates ``q_nope`` (e.g. ``[T, H_local, 512]`` or ``[T, cols_nope]``)
    and ``q_rope`` (e.g. ``[T, H_local, 64]`` or ``[T, cols_rope]``) along the
    last dimension before a single ``maybe_small_all_gather`` launch, then
    splits the gathered tensor back into ``(q_nope_gathered, q_rope_gathered)``.
    Replaces 2 separate XLA all-gather launches (36.0 us on 2x2x2) with 1
    Pallas all-gather launch (~13.3 us).
    """
    if q_nope.shape[:-1] != q_rope.shape[:-1]:
        raise ValueError(
            f"q_nope {tuple(q_nope.shape)} and q_rope {tuple(q_rope.shape)} "
            "must share leading dimensions"
        )
    if q_nope.dtype != q_rope.dtype:
        raise ValueError(
            f"q_nope dtype {q_nope.dtype} and q_rope dtype {q_rope.dtype} must match"
        )
    if gather_dim is None:
        gather_dim = 1 if q_nope.dim() == 3 else 0

    nope_dim = q_nope.shape[-1]
    rope_dim = q_rope.shape[-1]
    merged_last = nope_dim + rope_dim
    merged = torch.cat([q_nope, q_rope], dim=-1)

    if q_nope.dim() == 3 and gather_dim == 1:
        num_tokens, local_heads, _ = q_nope.shape
        flat_cols = local_heads * merged_last
        padded_cols = round_up(flat_cols, LANES)
        merged_2d = merged.reshape(num_tokens, flat_cols)
        if padded_cols != flat_cols:
            merged_2d = torch.nn.functional.pad(merged_2d, (0, padded_cols - flat_cols))

        def _fb_2d(t_2d: torch.Tensor) -> torch.Tensor:
            if fallback is not None:
                return fallback(t_2d)
            raise RuntimeError("No fallback provided for fused_q_head_all_gather")

        gathered_2d = maybe_small_all_gather(merged_2d, ranks, _fb_2d)
        if padded_cols != flat_cols:
            gathered_2d = gathered_2d[:, :flat_cols]
        n = gathered_2d.shape[0] // num_tokens
        gathered_3d = (
            gathered_2d.reshape(n, num_tokens, local_heads, merged_last)
            .transpose(0, 1)
            .reshape(num_tokens, n * local_heads, merged_last)
        )
        out_nope, out_rope = gathered_3d.split([nope_dim, rope_dim], dim=-1)
        return out_nope.contiguous(), out_rope.contiguous()

    if q_nope.dim() == 2 and gather_dim == 0:
        padded_cols = round_up(merged_last, LANES)
        merged_2d = merged
        if padded_cols != merged_last:
            merged_2d = torch.nn.functional.pad(
                merged_2d, (0, padded_cols - merged_last)
            )

        def _fb_2d(t_2d: torch.Tensor) -> torch.Tensor:
            if fallback is not None:
                return fallback(t_2d)
            raise RuntimeError("No fallback provided for fused_q_head_all_gather")

        gathered_2d = maybe_small_all_gather(merged_2d, ranks, _fb_2d)
        if padded_cols != merged_last:
            gathered_2d = gathered_2d[:, :merged_last]
        out_nope, out_rope = gathered_2d.split([nope_dim, rope_dim], dim=-1)
        return out_nope.contiguous(), out_rope.contiguous()

    raise ValueError(
        f"Unsupported shape {tuple(q_nope.shape)} with gather_dim={gather_dim}"
    )


def fused_topk_scores_indices_all_gather(
    scores: torch.Tensor,
    indices: torch.Tensor,
    ranks=None,
    fallback=None,
    *,
    gather_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused single-launch TopK candidate scores + indices all-gather for DSA
    Indexer decode (b/568775832).

    Packs ``scores`` (``[T, k_scores]``, ``bfloat16`` or ``float32``) and
    ``indices`` (``[T, k_idx]``, ``int32``, ``bfloat16``, or ``float32``) into a
    single 2-D ``bfloat16`` payload padded to ``LANES = 128`` columns, executes
    one ``maybe_small_all_gather``, and unpacks the exact original dtypes and
    shapes. Cuts multi-stage Indexer AG overhead from 123.0 us to ~13.5 us on
    2x2x2.
    """
    if scores.dim() != 2 or indices.dim() != 2:
        raise ValueError(
            f"scores {tuple(scores.shape)} and indices {tuple(indices.shape)} must be 2D"
        )
    if scores.shape[0] != indices.shape[0]:
        raise ValueError(
            f"scores rows {scores.shape[0]} != indices rows {indices.shape[0]}"
        )
    if fallback is None:
        raise ValueError(
            "fused_topk_scores_indices_all_gather requires a fallback callable"
        )

    num_tokens = scores.shape[0]
    scores_bf16 = (
        scores
        if scores.dtype == torch.bfloat16
        else scores.contiguous().view(torch.bfloat16)
    )
    indices_bf16 = (
        indices
        if indices.dtype == torch.bfloat16
        else indices.contiguous().view(torch.bfloat16)
    )
    s_cols = scores_bf16.shape[1]
    i_cols = indices_bf16.shape[1]
    raw_cols = s_cols + i_cols
    padded_cols = round_up(raw_cols, LANES)

    packed = torch.cat([scores_bf16, indices_bf16], dim=1)
    if padded_cols != raw_cols:
        packed = torch.nn.functional.pad(packed, (0, padded_cols - raw_cols))

    can_arm = (
        _AG_OP is not None
        and (ranks is None or tuple(ranks) == _MESH_RANKS)
        and _fits(packed, LANES)
    )
    if can_arm and num_tokens <= ALL_GATHER_MAX_ROWS:
        gathered_packed = _AG_OP(packed, _RANK)
        used_retain = False
    else:
        gathered_packed = fallback(packed)
        used_retain = can_arm and num_tokens > ALL_GATHER_MAX_ROWS

    unpadded = gathered_packed[:, :raw_cols]
    g_scores_bf16, g_indices_bf16 = unpadded.split([s_cols, i_cols], dim=1)

    g_scores = (
        g_scores_bf16
        if scores.dtype == torch.bfloat16
        else g_scores_bf16.contiguous().view(scores.dtype)
    )
    g_indices = (
        g_indices_bf16
        if indices.dtype == torch.bfloat16
        else g_indices_bf16.contiguous().view(indices.dtype)
    )
    if used_retain:
        g_scores = retain(g_scores)

    if gather_dim == 1:
        n = g_scores.shape[0] // num_tokens
        g_scores = (
            g_scores.reshape(n, num_tokens, scores.shape[1])
            .transpose(0, 1)
            .reshape(num_tokens, n * scores.shape[1])
            .contiguous()
        )
        g_indices = (
            g_indices.reshape(n, num_tokens, indices.shape[1])
            .transpose(0, 1)
            .reshape(num_tokens, n * indices.shape[1])
            .contiguous()
        )
    return g_scores, g_indices


def maybe_lse_all_to_all(
    lse: torch.Tensor,
    ranks=None,
    fallback=None,
) -> torch.Tensor:
    """LSE attention all-to-all helper for DCP MLA decode (b/568788608).

    Supports both ``bfloat16`` and ``float32`` 2-D LSE tensors ``[rows, heads]``
    (automatically bit-viewing ``float32`` as ``2 * heads`` ``bfloat16`` lanes
    for the zero-arithmetic DMA ``small_all_to_all`` kernel).
    """
    if fallback is None:
        raise ValueError("maybe_lse_all_to_all requires a fallback callable")
    if lse.dim() != 2 or lse.dtype not in (torch.bfloat16, torch.float32):
        return fallback(lse)

    is_f32 = lse.dtype == torch.float32
    lse_bf16 = lse.contiguous().view(torch.bfloat16) if is_f32 else lse
    raw_cols = lse_bf16.shape[1]
    padded_cols = round_up(raw_cols, LANES)
    work = (
        torch.nn.functional.pad(lse_bf16, (0, padded_cols - raw_cols))
        if padded_cols != raw_cols
        else lse_bf16
    )

    can_arm = (
        _A2A_OP is not None
        and (ranks is None or tuple(ranks) == _MESH_RANKS)
        and _fits(work, LANES)
        and (_CORES == 0 or work.shape[0] % _CORES == 0 or work.shape[1] % _CORES == 0)
    )
    if not can_arm:
        return fallback(lse)
    if work.shape[0] <= ALL_TO_ALL_MAX_ROWS:
        out_padded = _A2A_OP(work, _RANK)
        out_bf16 = (
            out_padded[:, :raw_cols].contiguous()
            if padded_cols != raw_cols
            else out_padded
        )
        return out_bf16.view(torch.float32) if is_f32 else out_bf16
    return retain(fallback(lse))


def maybe_post_attn_reduce_scatter(
    x: torch.Tensor,
    ranks=None,
    fallback=None,
    *,
    scatter_dim: int = 0,
) -> torch.Tensor:
    """Post-attention reduce-scatter helper for MLA / DCP decode (b/568555797).

    Supports 2-D ``[cores * rows, hidden]`` (``scatter_dim=0``) or 3-D
    ``[rows, cores * local_heads, head_dim]`` (``scatter_dim=1``).
    """
    if fallback is None:
        raise ValueError("maybe_post_attn_reduce_scatter requires a fallback callable")
    if x.dim() == 2 and scatter_dim == 0:
        return maybe_small_reduce_scatter(x, ranks, fallback)
    if x.dim() == 3 and scatter_dim == 1 and _CORES > 0:
        num_tokens, total_heads, head_dim = x.shape
        if total_heads % _CORES != 0:
            return fallback(x)
        local_heads = total_heads // _CORES
        flat_cols = local_heads * head_dim
        if flat_cols % LANES != 0:
            return fallback(x)
        x_2d = (
            x.reshape(num_tokens, _CORES, local_heads, head_dim)
            .transpose(0, 1)
            .reshape(_CORES * num_tokens, flat_cols)
        )
        if (
            _RS_OP is None
            or (ranks is not None and tuple(ranks) != _MESH_RANKS)
            or not _fits(x_2d, LANES)
        ):
            return fallback(x)
        if num_tokens <= COMBINE_MAX_ROWS:
            out_2d = _RS_OP(x_2d, _RANK)
            return out_2d.reshape(num_tokens, local_heads, head_dim)
        return retain(fallback(x))
    return fallback(x)


def maybe_moe_dispatch_all_gather(
    hidden_states: torch.Tensor,
    ranks=None,
    fallback=None,
) -> torch.Tensor:
    """MoE pre-FFN dispatch all-gather helper (``moe-dispatch-ag``, b/568788608)."""
    return maybe_small_all_gather(hidden_states, ranks, fallback)

