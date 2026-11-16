"""TPU DMA size vs. HBM bandwidth microbenchmark.

Copies a (ROWS, H) f32 array HBM -> VMEM -> HBM with a triple-buffered manual
DMA pipeline. All arms move exactly the same bytes; only the DMA size differs:

  * Baseline "block": one DMA per tile (TILE rows, i.e. 512 x 16KB = 8MB).
  * Experiment sweep: each row (16KB) is split into 16KB / S chunks and every
    chunk is its own DMA, for S in {2KB, 4KB, 8KB, 16KB}:
      - 16KB: 1 DMA per row   (512 DMAs per tile).
      -  8KB: 2 DMAs per row  (1024 DMAs per tile).
      -  4KB: 4 DMAs per row  (2048 DMAs per tile).
      -  2KB: 8 DMAs per row  (4096 DMAs per tile).

Every arm is verified bitwise against the input, timed on-device via the JAX
profiler, and compared against the block baseline.
"""

import glob
import os
import shutil
import statistics
import tempfile

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

H = 4096  # Row width in f32 elements (16KB per row).
TILE = 512  # Rows per pipeline stage (8MB per tile).
ROWS = 20480  # Total rows (~335MB array, 40 tiles).
NB = 3  # Number of VMEM buffers (triple buffering).

_BYTES_PER_ELEM = 4  # f32.
ROW_BYTES = H * _BYTES_PER_ELEM  # 16KB.

BLOCK_MODE = "block"
# Experiment arm: DMA sizes (KB) to sweep. 16KB == one DMA per row.
DMA_SIZES_KB = (2, 4, 8, 16)

# Maximum allowed slowdown of each sweep arm relative to the block baseline.
# `None` means the arm is only reported, not asserted.
MAX_SLOWDOWN_VS_BLOCK = {2: None, 4: None, 8: None, 16: 1.1}

_WARMUP_REPS = 5
_TIMED_REPS = 20


def _chunk_elems(dma_kb):
  """Returns the number of f32 elements per DMA for a `dma_kb` DMA size."""
  dma_bytes = dma_kb * 1024
  if dma_bytes > ROW_BYTES or ROW_BYTES % dma_bytes:
    raise ValueError(
        f"DMA size {dma_kb}KB must evenly divide the {ROW_BYTES}B row."
    )
  return dma_bytes // _BYTES_PER_ELEM


def _mode_name(mode):
  return BLOCK_MODE if mode == BLOCK_MODE else f"{mode}KB"


def _mode_desc(mode):
  if mode == BLOCK_MODE:
    return f"Block DMA (1x {TILE * ROW_BYTES // (1 << 20)}MB)"
  n = TILE * (H // _chunk_elems(mode))
  return f"{mode:>2}KB DMA ({n}x {mode}KB)"


def get_device_time_ms(fn, x, reps=_TIMED_REPS):
  """Returns median on-device duration (ms) of `fn(x)` on TPU:0."""
  for _ in range(_WARMUP_REPS):
    fn(x).block_until_ready()

  trace_dir = tempfile.mkdtemp()
  try:
    with jax.profiler.trace(trace_dir):
      out = None
      for _ in range(reps):
        out = fn(x)
      out.block_until_ready()

    pbs = glob.glob(f"{trace_dir}/**/*.xplane.pb", recursive=True)
    plane = jax.profiler.ProfileData.from_file(
        max(pbs, key=os.path.getmtime)
    ).find_plane_with_name("/device:TPU:0")

    durs = [
        stat[1]
        for line in plane.lines
        if line.name == "XLA Modules"
        for e in line.events
        for stat in e.stats
        if stat[0] == "device_duration_ps"
    ]
    if not durs:
      raise RuntimeError("No device_duration_ps found in profile.")
    return statistics.median(durs) / 1e9
  finally:
    shutil.rmtree(trace_dir, ignore_errors=True)


def get_kernel(mode):
  """Builds the triple-buffered HBM->VMEM->HBM copy kernel for `mode`.

  Args:
    mode: `BLOCK_MODE` for one DMA per tile, or an int DMA size in KB (one of
      `DMA_SIZES_KB`) to split every row into `16KB / mode` DMAs.
  """
  if mode == BLOCK_MODE:
    w = None
  elif mode in DMA_SIZES_KB:
    w = _chunk_elems(mode)  # Elements per DMA.
  else:
    raise ValueError(f"Unknown mode: {mode}")
  chunks_per_row = None if w is None else H // w

  def _kern(in_ref, out_ref, vmem, sem_r, sem_w):
    num_tiles = in_ref.shape[0] // TILE

    def start_r(i, slot):
      if w is None:
        pltpu.make_async_copy(
            in_ref.at[pl.ds(i * TILE, TILE)], vmem.at[slot], sem_r.at[slot]
        ).start()
        return

      # Fully unrolled at trace time so all DMA offsets are static ints.
      for k in range(TILE * chunks_per_row):
        j, c = divmod(k, chunks_per_row)
        pltpu.make_async_copy(
            in_ref.at[pl.ds(i * TILE + j, 1), pl.ds(c * w, w)],
            vmem.at[slot, pl.ds(j, 1), pl.ds(c * w, w)],
            sem_r.at[slot],
        ).start()

    def wait_r(slot):
      if w is None:
        pltpu.make_async_copy(
            in_ref.at[pl.ds(0, TILE)], vmem.at[slot], sem_r.at[slot]
        ).wait()
        return
      canon = pltpu.make_async_copy(
          in_ref.at[pl.ds(0, 1), pl.ds(0, w)],
          vmem.at[slot, pl.ds(0, 1), pl.ds(0, w)],
          sem_r.at[slot],
      )

      for _ in range(TILE * chunks_per_row):
        canon.wait()

    def start_w(i, slot):
      if w is None:
        pltpu.make_async_copy(
            vmem.at[slot], out_ref.at[pl.ds(i * TILE, TILE)], sem_w.at[slot]
        ).start()
        return

      # Fully unrolled at trace time so all DMA offsets are static ints.
      for k in range(TILE * chunks_per_row):
        j, c = divmod(k, chunks_per_row)
        pltpu.make_async_copy(
            vmem.at[slot, pl.ds(j, 1), pl.ds(c * w, w)],
            out_ref.at[pl.ds(i * TILE + j, 1), pl.ds(c * w, w)],
            sem_w.at[slot],
        ).start()

    def wait_w(slot):
      if w is None:
        pltpu.make_async_copy(
            vmem.at[slot], out_ref.at[pl.ds(0, TILE)], sem_w.at[slot]
        ).wait()
        return
      canon = pltpu.make_async_copy(
          vmem.at[slot, pl.ds(0, 1), pl.ds(0, w)],
          out_ref.at[pl.ds(0, 1), pl.ds(0, w)],
          sem_w.at[slot],
      )

      for _ in range(TILE * chunks_per_row):
        canon.wait()

    # Prologue: prefetch the first NB - 1 tiles.
    for k in range(NB - 1):

      @pl.when(k < num_tiles)
      def _(k=k):
        start_r(jnp.int32(k), k)

    # Steady state: wait read, issue write, prefetch tile i + NB - 1.
    @pl.loop(0, num_tiles)
    def _(i):
      slot = i % NB
      wait_r(slot)
      start_w(i, slot)
      nxt = i + (NB - 1)

      @pl.when(nxt < num_tiles)
      def _():
        nslot = nxt % NB

        # The slot is reused: wait for its previous write to drain first.
        @pl.when(nxt >= NB)
        def _():
          wait_w(nslot)

        start_r(nxt, nslot)

    # Epilogue: drain the last NB outstanding writes.
    for k in range(NB):

      @pl.when(num_tiles - NB + k >= 0)
      def _(k=k):
        wait_w((num_tiles - NB + k) % NB)

  return _kern


def build_copy_fn(mode, x):
  """Returns a compiled copy function for `mode` specialized to `x`."""
  kern = get_kernel(mode)

  @jax.jit
  def run(x):
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=(
            pltpu.VMEM((NB, TILE, H), jnp.float32),
            pltpu.SemaphoreType.DMA((NB,)),
            pltpu.SemaphoreType.DMA((NB,)),
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"dma_{_mode_name(mode)}",
    )(x)

  return run.lower(x).compile()


class DmaSizeBandwidthTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.x = jax.random.normal(
        jax.random.PRNGKey(0), (ROWS, H), dtype=jnp.float32
    )
    cls.x_np = np.asarray(cls.x)
    cls.results = {}  # mode -> (ms, gbps), shared across test methods.

  def _run_mode(self, mode):
    """Verifies and benchmarks `mode`; results are cached per mode."""
    if mode in self.results:
      return self.results[mode]
    fn = build_copy_fn(mode, self.x)
    out = np.asarray(jax.device_get(fn(self.x)))
    np.testing.assert_array_equal(
        out,
        self.x_np,
        err_msg=f"Bitwise verification failed for {_mode_name(mode)}",
    )
    ms = get_device_time_ms(fn, self.x)
    gbps = (2 * ROWS * H * _BYTES_PER_ELEM) / (ms * 1e-3) / 1e9
    self.results[mode] = (ms, gbps)
    return ms, gbps

  def test_block_copy_is_bitwise_exact(self):
    ms, gbps = self._run_mode(BLOCK_MODE)
    logging.info("%s: %.3f ms | %.1f GB/s", _mode_desc(BLOCK_MODE), ms, gbps)

  @parameterized.named_parameters((f"{kb}kb", kb) for kb in DMA_SIZES_KB)
  def test_sweep_copy_is_bitwise_exact(self, dma_kb):
    ms, gbps = self._run_mode(dma_kb)
    logging.info("%s: %.3f ms | %.1f GB/s", _mode_desc(dma_kb), ms, gbps)

  def test_dma_size_sweep_vs_block(self):
    logging.info("%d rows, H=%d, Tile=%d", ROWS, H, TILE)
    ms_block, gbps_block = self._run_mode(BLOCK_MODE)
    logging.info(
        "%-24s: %.3f ms | %6.1f GB/s | 1.000x",
        _mode_desc(BLOCK_MODE),
        ms_block,
        gbps_block,
    )
    ratios = {}
    for kb in DMA_SIZES_KB:
      ms, gbps = self._run_mode(kb)
      ratios[kb] = ms / ms_block
      logging.info(
          "%-24s: %.3f ms | %6.1f GB/s | %.3fx",
          _mode_desc(kb),
          ms,
          gbps,
          ratios[kb],
      )
    for kb, ratio in ratios.items():
      limit = MAX_SLOWDOWN_VS_BLOCK.get(kb)
      if limit is not None:
        with self.subTest(dma_kb=kb):
          self.assertLessEqual(
              ratio, limit, f"{kb}KB DMAs are {ratio:.3f}x slower than block"
          )


if __name__ == "__main__":
  absltest.main()