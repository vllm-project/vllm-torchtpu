# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-IDXFILL (VLLM_TPU_QSA_PREFILL_QPROJ_FILL): the QSA walk ``q_proj``
of a ragged prefill step as row slices that run inside the prefill indexer's
SparseCore top-k windows.

With the no-cond prefill indexer (VLLM_TPU_QSA_PREFILL_INDEXER_NO_COND) a QSA
layer's chunked indexer pins window g -- chunk g's SparseCore top-k, from its
async update to its done -- with the XLA scheduling group ``base + g`` (the
layer's own static range, ``config.layer_scheduling_group_base(prefix)``),
which holds chunk g + 1's TensorCore scoring and the done. The SparseCore call
outlasts the scoring, so the TensorCore waits in every window, while the whole
``q_proj`` matmul lands untagged after the windows. Here the matmul runs as
``chunks - 2`` row slices and slice g carries the group ``base + g``: the
scheduler emits it inside window g. Each slice is the dot ``F.linear`` lowers
to in torch_tpu (stablehlo.dot_general, contracting the last dims of input and
weight, the global float32 matmul precision; csrc/ops/linear/
linear_aten_kernels.cc BuildLinearShlo), so every output row is the same dot.

It lives under layers/adapter/custom_ops so the compile-cache key hashes it.
The op is built at import: Dynamo cannot trace its construction from inside a
compiled forward.
"""

import jax
import jax.numpy as jnp
import torch
from jax.experimental import xla_metadata
from torch_tpu._internal import pallas

# torch.get_float32_matmul_precision() names (torch_tpu _loader.py) -> the
# stablehlo precision torch_tpu gives F.linear.
_PRECISION = {
    "medium": jax.lax.Precision.DEFAULT,
    "high": jax.lax.Precision.HIGH,
    "highest": jax.lax.Precision.HIGHEST,
}

# Row alignment of the slice starts (see qproj_fill_jax).
_ALIGN = 16

# (rows, slices, group base, precision) already logged by this process.
_LOGGED: set = set()


def indexer_chunk(rows: int, chunk_override: int) -> int | None:
    """``pooled_qsa.prefill_indexer_chunk(rows)`` with the env read once at layer
    build (``chunk_override`` = VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK): the forward
    runs inside Dynamo's fullgraph capture, so no env reads or imports there."""
    if chunk_override and chunk_override % 8 == 0 and rows % chunk_override == 0:
        return chunk_override
    return 512 if rows % 512 == 0 else None


def fill_slices(rows: int, slots: int, max_query_len: int, chunk_override: int) -> int:
    """Row slices for a step of ``rows`` tokens, or 0 for no fill.

    Mirrors pooled_qsa's choice of the chunked ragged prefill indexer: the
    ragged path (``rows > slots * max_query_len``), its token chunk and
    ``rows // chunk`` chunks (streamindex_topk), of which windows
    ``0 .. chunks - 2`` carry groups. The fill takes windows ``0 .. chunks - 3``
    and leaves the last grouped window to the last chunk's larger scoring.
    Fewer than 3 chunks: no fill.
    """
    if rows <= slots * max_query_len:
        return 0
    chunk = indexer_chunk(rows, chunk_override)
    if not chunk:
        return 0
    chunks = rows // chunk
    return chunks - 2 if chunks >= 3 else 0


def matmul_precision() -> str:
    """torch_tpu's global float32 matmul precision name (read at layer build)."""
    return str(torch.get_float32_matmul_precision())


def _log_engaged(rows: int, slices: int, group_base: int, precision: str) -> None:
    """Engagement line (rule 13), trace time only: one per traced shape, layer
    group base and process (a compile-cache hit prints nothing)."""
    key = (rows, slices, group_base, precision)
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info(
        "q38 PSTEP-IDXFILL QPROJ-FILL ENGAGED: %d rows, %d slices, groups %d..%d, precision %s"
        " (VLLM_TPU_QSA_PREFILL_QPROJ_FILL)",
        rows, slices, group_base, group_base + slices - 1, precision,
    )


def qproj_fill_jax(
    hidden: jax.Array,
    weight: jax.Array,
    *,
    group_base: int,
    slices: int,
    precision: str,
) -> jax.Array:
    """``hidden @ weight.T`` as ``slices`` row slices, slice g in scheduling
    group ``group_base + g``; the slices are concatenated in row order.

    Slice starts are rounded down to 16 rows (one bf16 (8, 128)(2, 1) tile of
    the [rows, N] result), so XLA assembles the slices in place (each dot fused
    with a dynamic-update-slice into the result) instead of a concatenate copy;
    6,400 rows / 8 slices are 800-row slices either way."""
    rows = hidden.shape[0]
    prec = _PRECISION[precision]
    bounds = [(rows * g // slices) // _ALIGN * _ALIGN for g in range(slices)] + [rows]
    parts = []
    for g in range(slices):
        part = jax.lax.dot_general(
            hidden[bounds[g] : bounds[g + 1]],
            weight,
            (((hidden.ndim - 1,), (weight.ndim - 1,)), ((), ())),
            precision=(prec, prec),
        )
        parts.append(
            xla_metadata.set_xla_metadata(part, _scheduling_group_id=group_base + g)
        )
    _log_engaged(rows, slices, group_base, precision)
    return jnp.concatenate(parts, axis=0)


QPROJ_FILL_OP = pallas.jax_op("qwen4_exp::qsa_qproj_fill", qproj_fill_jax)


@QPROJ_FILL_OP.register_fake
def _fake_qproj_fill(hidden, weight, *, group_base, slices, precision):
    del group_base, slices, precision
    return hidden.new_empty((hidden.shape[0], weight.shape[0]))


def qproj_fill(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    slots: int,
    max_query_len: int,
    *,
    group_base: int,
    chunk_override: int,
    precision: str,
) -> torch.Tensor | None:
    """The filled ``q_proj`` of a step, or None when the step does not take the
    chunked prefill indexer with at least 3 chunks (the caller keeps
    ``q_proj``). Static per graph: the row count, slots and the layer's base.
    No imports, env reads or logging here (inside the Dynamo capture)."""
    slices = fill_slices(int(hidden.shape[0]), slots, max_query_len, chunk_override)
    if slices < 1:
        return None
    return QPROJ_FILL_OP(
        hidden, weight, group_base=group_base, slices=slices, precision=precision
    )
