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
# ==============================================================================
"""In-kernel token-gather assembly for the fused kernel (`fuse_tg_assemble`).

The same copy as `dispatch.tg_assemble`'s vector kernel -- the landed sender
lists moved run by run to their slab offsets in a zeroed image with 8-row
aligned tiles, a dynamic lane rotate, a sublane select and a masked
read-modify-write, then clipped to [0, T - 1] -- run in the fused kernel's
prologue instead of as its own pallas_call ahead of it. The prologue has
already started the first weight refills, so the assembly runs under the
weight stream rather than in front of it.

The token image ends in SCALAR memory (one VMEM -> SMEM copy), where the tile
gather reads each slab row's token directly, so no per-expert HBM -> SMEM
gather window is needed. The fp8 activation-scale image stays in VMEM as
f32, where the tile's scale window is a VMEM -> VMEM copy.

Bit patterns are those of the separate kernel: the token image and scale bit
image are built by the same per-run copy and clip, so the fused kernel
gathers the same tokens and scales.
"""

import os

from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_moe import host as fused_host

LANES = 128
SUBLANES = 8
# VMEM rows in front of and behind the landed lists, so the aligned 16-row
# source window of any tile stays inside the scratch.
FRONT = SUBLANES
BACK = 2 * SUBLANES
# Run-table fields (dispatch.tg_assemble.assemble_runs); the table arrives
# flat, field-major: field f of run r is word f * n_runs + r.
RUN_SRC = 0
RUN_N = 1
RUN_DST = 2
RUN_SRC1 = 3
N_RUN_FIELDS = 4
# The SMEM the token image may take. The XLA callee share of SMEM is about
# 600 KiB and the kernel's tables need the rest; decode-sized slabs are well
# under this (t_local = 48 at EP = 32: 64 KiB).
MAX_IMAGE_SMEM_BYTES = 256 * 1024
# Opt-in (bit-exact): clip the token values inside the per-run
# copy instead of a separate pass over the image afterwards (the image is
# zero-cleared first, and clip(0) = 0, so the result is the same).
_TG_CLIP_FUSE = os.environ.get("DEDUP_MOE_TG_CLIP_FUSE", "0") == "1"


def image_rows(ragged_rows_alloc, tile_m):
  """Rows of the [rows, 128] token image (dispatch.tg_assemble.gather_rows)."""
  n = ragged_rows_alloc + fused_host.token_gather_window_rows(tile_m)
  return fused_host.align_up(
      fused_host.align_up(n, LANES) // LANES, SUBLANES
  )


def scale_image_rows(ragged_rows_alloc):
  """Rows of the f32 activation-scale image, padded to whole tiles."""
  return fused_host.align_up(
      fused_host.act_scale_slab_rows(ragged_rows_alloc), SUBLANES
  )


def check_fits(ragged_rows_alloc, tile_m):
  """Refuses a slab whose token image does not fit the SMEM share."""
  nbytes = image_rows(ragged_rows_alloc, tile_m) * LANES * 4
  if nbytes > MAX_IMAGE_SMEM_BYTES:
    raise ValueError(
        f"fuse_tg_assemble keeps the token image in SMEM: {nbytes / 1024:.0f}"
        f" KiB for ragged_rows_alloc={ragged_rows_alloc} is over the"
        f" {MAX_IMAGE_SMEM_BYTES // 1024} KiB share. Use the separate"
        " tg_assemble kernel at this token bucket."
    )


def scratch_shapes(*, land_rows, tg_rows, scale_rows):
  """VMEM / SMEM / semaphore scratch, in the order `assemble` takes them."""
  out = [
      pltpu.VMEM((FRONT + land_rows + BACK, LANES), jnp.int32),  # land_vm
      pltpu.VMEM((tg_rows, LANES), jnp.int32),  # tg_vm
  ]
  if scale_rows:
    out.append(pltpu.VMEM((scale_rows, LANES), jnp.float32))  # scale_vm
  out += [
      pltpu.SMEM((tg_rows, LANES), jnp.int32),  # tg_sm
      pltpu.SemaphoreType.DMA((2,)),  # tg_sems
  ]
  return out


def _sublanes_from(win, k):
  """win[k : k + 8] of a [16, 128] window for a dynamic k in [0, 8]."""
  out = win[0:SUBLANES]
  for j in range(1, SUBLANES + 1):
    out = jnp.where(k == j, win[j : j + SUBLANES], out)
  return out


def _copy_run(land_vm, img_vm, src, n, dst, max_tok=None):
  """img[dst + i] = land[src + i] (as img's element type) for i < n.

  dispatch.tg_assemble._copy_run_vector, with the copied int32 words
  bitcast to the image's element type (f32 for the scale image) so the scale
  bits land unchanged.

  Args:
    land_vm: [rows, 128] i32 landed lists (padded by FRONT rows).
    img_vm: [rows, 128] image.
    src: First source word, in land_vm's padded coordinates.
    n: Words to copy.
    dst: First destination word.
    max_tok: Optional clip bound `n_tokens - 1` for the copied values.
  """
  delta = src - dst
  q = jnp.right_shift(delta, 7)
  rem = jnp.bitwise_and(delta, LANES - 1)
  t_first = jnp.right_shift(dst, 10)
  t_last = jnp.right_shift(dst + n - 1, 10)
  sub = lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
  lane = lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 1)

  def tile(ti, carry):
    r0 = pl.multiple_of(ti * SUBLANES, SUBLANES)
    q0 = r0 + q
    qa = pl.multiple_of(jnp.bitwise_and(q0, -SUBLANES), SUBLANES)
    sh = q0 - qa
    win = land_vm[pl.ds(qa, 2 * SUBLANES), :]
    a = _sublanes_from(win, sh)
    b = _sublanes_from(win, sh + 1)
    shift = LANES - rem
    vals = jnp.where(
        lane < LANES - rem, pltpu.roll(a, shift, 1), pltpu.roll(b, shift, 1)
    )
    if img_vm.dtype != vals.dtype:
      vals = lax.bitcast_convert_type(vals, img_vm.dtype)
    if max_tok is not None:
      vals = jnp.clip(vals, 0, max_tok)
    w = (r0 + sub) * LANES + lane
    live = (w >= dst) & (w < dst + n)
    cur = img_vm[pl.ds(r0, SUBLANES), :]
    img_vm[pl.ds(r0, SUBLANES), :] = jnp.where(live, vals, cur)
    return carry

  lax.fori_loop(t_first, t_last + 1, tile, 0)


def _copy_run_pair(
    land_vm,
    tg_vm,
    scale_vm,
    src0,
    src1,
    n,
    dst,
    max_tok=None,
    at_most_two_tiles=False,
):
  """Fused `_copy_run` for `tg_vm` and `scale_vm` sharing loop/mask/roll setup.

  `src1 - src0` is always a multiple of `LANES = 128` (`tg_assemble.py`), so
  `rem`, `shift`, `t_first`, `t_last`, `w`, and `live` are identical for both.
  """
  delta0 = src0 - dst
  delta1 = src1 - dst
  q0_base = jnp.right_shift(delta0, 7)
  q1_base = jnp.right_shift(delta1, 7)
  rem = jnp.bitwise_and(delta0, LANES - 1)
  t_first = jnp.right_shift(dst, 10)
  t_last = jnp.right_shift(dst + n - 1, 10)
  sub = lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
  lane = lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 1)
  shift = LANES - rem
  lane_keep = lane < shift

  def _gather_shifted(q_base, r0):
    q0 = r0 + q_base
    qa = pl.multiple_of(jnp.bitwise_and(q0, -SUBLANES), SUBLANES)
    sh = q0 - qa
    win = land_vm[pl.ds(qa, 2 * SUBLANES), :]
    a = _sublanes_from(win, sh)
    b = _sublanes_from(win, sh + 1)
    return jnp.where(
        lane_keep, pltpu.roll(a, shift, 1), pltpu.roll(b, shift, 1)
    )

  def tile(ti, carry):
    r0 = pl.multiple_of(ti * SUBLANES, SUBLANES)
    w = (r0 + sub) * LANES + lane
    live = (w >= dst) & (w < dst + n)

    vals0 = _gather_shifted(q0_base, r0)
    if max_tok is not None:
      vals0 = jnp.clip(vals0, 0, max_tok)
    cur0 = tg_vm[pl.ds(r0, SUBLANES), :]
    tg_vm[pl.ds(r0, SUBLANES), :] = jnp.where(live, vals0, cur0)

    vals1 = lax.bitcast_convert_type(_gather_shifted(q1_base, r0), scale_vm.dtype)
    cur1 = scale_vm[pl.ds(r0, SUBLANES), :]
    scale_vm[pl.ds(r0, SUBLANES), :] = jnp.where(live, vals1, cur1)
    return carry

  if at_most_two_tiles:
    tile(t_first, 0)

    @pl.when(t_last > t_first)
    def _():
      tile(t_last, 0)

  else:
    lax.fori_loop(t_first, t_last + 1, tile, 0)


def _for_tiles(rows, body):
  def step(i, carry):
    body(pl.multiple_of(i * SUBLANES, SUBLANES))
    return carry

  lax.fori_loop(0, rows // SUBLANES, step, 0)


def assemble(
    *,
    ep,
    n_runs,
    n_tokens,
    runs_sm,
    meta_sm,
    land_hbm,
    tg_sem,
    land_vm,
    tg_vm,
    scale_vm,
    tg_sm,
    tg_sems,
    fuse_copy=False,
    wait_to_smem=True,
    fast_short_runs=False,
):
  """Builds the token image in `tg_sm` (and the scale image in `scale_vm`).

  Args:
    ep: Expert-parallel width.
    n_runs: Runs in the run table (ep * g_local).
    n_tokens: Tokens in the slab; tokens are clipped to [0, n_tokens - 1].
    runs_sm: Flat field-major [4 * n_runs] i32 run table (SMEM).
    meta_sm: [1 + ep] i32: rows to wait for, then rows landed per source.
    land_hbm: [ep * src_rows, 128] i32 landed lists.
    tg_sem: The start kernel's list semaphore, or None for complete lists.
    land_vm: VMEM staging for the lists.
    tg_vm: VMEM token image.
    scale_vm: VMEM f32 scale image, or None (unquantized rows).
    tg_sm: SMEM token image the tile gather reads.
    tg_sems: Two DMA semaphores.
  """
  src_rows = land_hbm.shape[0] // ep
  if tg_sem is not None:
    rows = meta_sm[0]

    @pl.when(rows > 0)
    def _():
      pltpu.make_async_copy(
          land_hbm.at[pl.ds(0, rows)], land_hbm.at[pl.ds(0, rows)], tg_sem
      ).wait()

  copies = [
      pltpu.make_async_copy(
          land_hbm.at[pl.ds(s * src_rows, meta_sm[1 + s])],
          land_vm.at[pl.ds(FRONT + s * src_rows, meta_sm[1 + s])],
          tg_sems.at[0],
      )
      for s in range(ep)
  ]
  for c in copies:
    c.start()

  def clear(img):
    zero = jnp.zeros((SUBLANES, LANES), img.dtype)

    def body(r0):
      img[pl.ds(r0, SUBLANES), :] = zero

    _for_tiles(img.shape[0], body)

  if (
      fuse_copy
      and fast_short_runs
      and scale_vm is not None
      and tg_vm.shape == scale_vm.shape
  ):
    zero_i = jnp.zeros((SUBLANES, LANES), tg_vm.dtype)
    zero_f = jnp.zeros((SUBLANES, LANES), scale_vm.dtype)

    def clear_both(r0):
      tg_vm[pl.ds(r0, SUBLANES), :] = zero_i
      scale_vm[pl.ds(r0, SUBLANES), :] = zero_f

    _for_tiles(tg_vm.shape[0], clear_both)
  else:
    clear(tg_vm)
    if scale_vm is not None:
      clear(scale_vm)
  for c in copies:
    c.wait()

  # With two planes per entry (token + scale) plus a 1-row header, one source's
  # list capacity in entries is at most `((src_rows - 1) // 2) * LANES`. When
  # that fits in one `SUBLANES * LANES = 1024` tile (`t_local <= 96`), any run
  # spans at most 2 tiles (`t_last <= t_first + 1`).
  at_most_two_tiles = (
      fast_short_runs
      and scale_vm is not None
      and (src_rows - 1) // 2 <= SUBLANES
  )

  def run(r, carry):
    n = runs_sm[RUN_N * n_runs + r]

    @pl.when(n > 0)
    def _():
      dst = runs_sm[RUN_DST * n_runs + r]
      src0 = runs_sm[RUN_SRC * n_runs + r] + FRONT * LANES
      max_tok = max(n_tokens - 1, 0) if _TG_CLIP_FUSE else None
      if fuse_copy and scale_vm is not None:
        _copy_run_pair(
            land_vm,
            tg_vm,
            scale_vm,
            src0,
            runs_sm[RUN_SRC1 * n_runs + r] + FRONT * LANES,
            n,
            dst,
            max_tok=max_tok,
            at_most_two_tiles=at_most_two_tiles,
        )
      else:
        _copy_run(
            land_vm,
            tg_vm,
            src0,
            n,
            dst,
            max_tok=max_tok,
        )
        if scale_vm is not None:
          _copy_run(
              land_vm,
              scale_vm,
              runs_sm[RUN_SRC1 * n_runs + r] + FRONT * LANES,
              n,
              dst,
          )

    return carry

  lax.fori_loop(0, n_runs, run, 0)

  if not _TG_CLIP_FUSE:

    def clip(r0):
      tg_vm[pl.ds(r0, SUBLANES), :] = jnp.clip(
          tg_vm[pl.ds(r0, SUBLANES), :], 0, n_tokens - 1
      )

    _for_tiles(tg_vm.shape[0], clip)
  to_smem = pltpu.make_async_copy(tg_vm, tg_sm, tg_sems.at[1])
  to_smem.start()
  if wait_to_smem:
    to_smem.wait()
  return to_smem


def token_at(tg_sm, row):
  """The token of slab row `row` in the SMEM image."""
  return tg_sm[jnp.right_shift(row, 7), jnp.bitwise_and(row, LANES - 1)]

