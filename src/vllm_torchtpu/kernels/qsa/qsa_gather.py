"""SparseCore gather of QSA KV-cache rows (SC -> HBM stage of sparse QSA).

Every token of the QSA cache occupies one 128-word (512-byte) int32 row's worth
of fp8 K and V per KV head (head_dim 256), packed in 4-token blocks; see
`qsa_cache_layout` for the byte layout. The gather is a pure row copy (in
`sparse_qsa`, one `kv_heads * 2` KB row per compression group): the gathered
row *is* the output row, so the SparseCore issues indirect HBM reads and
streams the rows straight back to HBM for the TensorCore attention kernel.
"""

from typing import Any, cast

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp

# Minimum gather-row granularity in int32 words (one 512 B cache row).
ROW_WORDS = 128
# `pl.Indirect` gathers issued per pipeline step (keeps multiple gather DMAs
# in flight), by gathered row size.
NUM_STREAMS = 8  # 512 B rows.
GROUP_NUM_STREAMS = 4  # Larger rows (e.g. a 2 KB compression group).
# Upper bound on the rows a subcore handles per outer step (bounds its index
# buffer).
MAX_ROWS_PER_STEP = 2048


# Budget for the gather buffers (num_streams x _BUFFER_COUNT slots x
# SIMD-width rows) in per-subcore TileSpmem: half of TPU7x's 512 KiB, leaving
# room for the index buffers and compiler scratch.
GATHER_BUFFER_BYTES = 256 * 1024
# Gather buffer slots per stream (default double buffering). Deeper buffering
# (`pl.Buffered(>= 3)`) does not lower with `pl.Indirect` on SparseCore.
_BUFFER_COUNT = 2


def _cdiv(a: int, b: int) -> int:
  return (a + b - 1) // b


def _num_streams(row_words: int, simd_lanes: int) -> int:
  """Stream count for `row_words`-word rows, capped by the buffer budget."""
  preferred = NUM_STREAMS if row_words <= ROW_WORDS else GROUP_NUM_STREAMS
  per_stream = _BUFFER_COUNT * simd_lanes * row_words * 4
  return max(1, min(preferred, GATHER_BUFFER_BYTES // per_stream))


@jax.jit
def qsa_gather(cache: jax.Array, indices: jax.Array) -> jax.Array:
  """SparseCore gather of QSA cache rows.

  Args:
    cache: int32 array whose last dim (a multiple of 128 words) is one gather
      row; leading dims are flattened. E.g. `sparse_qsa` passes
      (num_tokens // ratio, ratio * kv_heads * 128): one row per compression
      group.
    indices: (N,) int32 row indices into the flattened cache.

  Returns:
    (N, cache.shape[-1]) int32. Row i is cache row `indices[i]`.
  """
  assert indices.ndim == 1, "Indices must be 1D."
  assert cache.dtype == jnp.int32, cache.dtype
  row_words = cache.shape[-1]
  assert row_words % ROW_WORDS == 0, cache.shape

  # The kernel takes its operands untiled (`use_tc_tiling_on_sc=False`), i.e.
  # row-major, so flattening the leading dims is free.
  cache = cache.reshape(-1, row_words)
  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "SparseCore info is missing."
  out_size = indices.size
  num_cores = sc_info.num_cores * sc_info.num_subcores
  # Rows per `pl.Indirect` gather: one SIMD-width subchunk.
  row_subchunk_size = sc_info.num_lanes
  num_streams = _num_streams(row_words, row_subchunk_size)

  # Rows per subcore per outer step. Fix the number of rounds first (the
  # fewest that respect MAX_ROWS_PER_STEP), then size the chunks to fill
  # them, in whole `num_streams * SIMD width` units. Every subcore then runs
  # the same number of chunks and the padding is under one unit per chunk;
  # capping the chunk first instead can leave a last round that only a few
  # subcores work on.
  unit = row_subchunk_size * num_streams
  rounds = max(_cdiv(out_size, num_cores * MAX_ROWS_PER_STEP), 1)
  row_chunk_size = _cdiv(_cdiv(out_size, num_cores * rounds), unit) * unit
  num_row_subchunks = row_chunk_size // row_subchunk_size
  block_size = row_chunk_size * num_cores
  num_blocks = _cdiv(out_size, block_size)
  out_pad_size = num_blocks * block_size - out_size
  # Write exactly `out_size` rows when that is whole subchunks (the pad rows
  # are skipped in-kernel); otherwise write the padded size and slice.
  num_out_rows = (
      out_size if out_size % row_subchunk_size == 0 else out_size + out_pad_size
  )
  if out_pad_size:
    # Spread the padding to avoid hotspots.
    num_tokens = cache.shape[0]
    pad = (jnp.arange(out_pad_size, dtype=indices.dtype) * 104729) % num_tokens
    indices = jnp.concatenate([indices, pad])

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  mesh_axes = (vector_mesh.core_axis_name, vector_mesh.subcore_axis_name)

  def kernel(cache_hbm_ref, indices_hbm_ref, out_hbm_ref, sem):
    core_index = lax.axis_index(mesh_axes)

    def outer_pipeline(idx_ref):
      b = pl.program_id(0)
      out_row_base = (b * num_cores + core_index) * num_row_subchunks

      # Subchunk handled by stream `s` at inner step `r`. `num_streams`
      # independent `pl.Indirect` gathers run concurrently per step, keeping
      # several gather DMAs in flight to raise effective read bandwidth.
      def subchunk(r, s):
        return r * num_streams + s

      def idx_window(r, s):
        return idx_ref[
            pl.ds(subchunk(r, s) * row_subchunk_size, row_subchunk_size)
        ]

      def _body(*gathered):
        r = pl.program_id(0)
        # No vector work: DMA each gather buffer straight to HBM rather than
        # routing it through a pipeline output buffer, which would cost a
        # VMEM->VMEM copy.
        # Subchunks at or past `num_out_rows` (index padding) are gathered but
        # not written. When `out_size` is whole subchunks, the output then
        # holds exactly `out_size` rows, so the caller needs no slice (an HBM
        # relayout copy on the TensorCore).
        copies = []
        for s in range(num_streams):
          out_row = (out_row_base + subchunk(r, s)) * row_subchunk_size
          copy = pltpu.make_async_copy(
              gathered[s],
              out_hbm_ref.at[pl.ds(out_row, row_subchunk_size)],
              sem.at[s],
          )
          in_range = out_row < num_out_rows
          pl.when(in_range)(copy.start)
          copies.append((in_range, copy))
        for in_range, copy in copies:
          pl.when(in_range)(copy.wait)

      # Stream `s` gathers the `row_subchunk_size` cache rows named by its
      # index window, double-buffered.
      in_specs = tuple(
          pl.BlockSpec(
              (pl.Indirect(row_subchunk_size), row_words),
              lambda r, s=s: (idx_window(r, s), 0),
          )
          for s in range(num_streams)
      )

      # Skip work chunks made entirely of index padding.
      @pl.when(out_row_base * row_subchunk_size < out_size)
      def _run_gather():
        pltpu.emit_pipeline(
            _body,
            grid=(num_row_subchunks // num_streams,),
            in_specs=in_specs,
        )(*([cache_hbm_ref] * num_streams))

    idx_spec = pl.BlockSpec(
        (row_chunk_size,), lambda b: (b * num_cores + core_index,)
    )
    pltpu.emit_pipeline(
        outer_pipeline, grid=(num_blocks,), in_specs=(idx_spec,)
    )(indices_hbm_ref)

  out = cast(
      Any,
      pl.kernel(
          kernel,
          out_type=jax.ShapeDtypeStruct((num_out_rows, row_words), jnp.int32),
          # One DMA semaphore per stream for the gather-buffer -> HBM copies.
          scratch_types=(pltpu.SemaphoreType.DMA((num_streams,)),),
          compiler_params=pltpu.CompilerParams(
              use_tc_tiling_on_sc=False,
              needs_layout_passes=True,
              disable_bounds_checks=True,
          ),
          mesh=vector_mesh,
          name="sc_qsa_gather",
      )(cache, indices),
  )
  return out if num_out_rows == out_size else out[:out_size]
