# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
#
# Adapted from a code-reviewed TPU diffusion kernel implementation.
# Changes from the original:
# - Removed pareto_block_q(), rank_block_candidates(), _MEASURED_OVERRIDES,
#   the residual* kernel modes, and unused TpuSpec.peak_bf16_flops /
#   hbm_bw_bytes.
# - Typed spec_from_tpu_info(info: Any) instead of suppressing pyrefly,
#   populated TpuSpec.num_lanes from get_tpu_info(), and cached detect_spec().
# - v7 fixed-m VMEM fraction 0.6, which caps block_q at 4096 for head_dim 128.
# - Only v7 counts as calibrated in _KNOWN_GENERATIONS (was v6 and v7).
# - Added vmem_footprint_bytes(), a VMEM upper bound fitted on v7x, and cap
#   block_q so it fits physical VMEM.
# - block_kv is the whole K/V extent when that is shorter than
#   safe_kv_block, and sequences up to one MXU tile use one
#   full-extent block instead of raising.

"""Dynamic, generation-portable block-size autotuner for splash attention.

Given a sequence length and the TPU hardware spec, this derives the
``(block_q, block_kv, block_kv_compute)`` for the custom splash kernel
(``kernel.flash_attention_bshd``) with no on-device sweep. The spec is read at
runtime from ``pltpu.get_tpu_info`` (VMEM, MXU edge, generation),
so v8/v9 are handled automatically; the few compiler-calibrated constants are
multiples of the MXU edge.

Why a cost model and not a single formula
-----------------------------------------
The kernel is MXU-bound on QK^T (contraction K = head_dim = 128, an irreducible
~60%-of-peak ceiling). Block sizes only change how close to that ceiling it
runs. A v7 sweep across resolutions (per chiplet, fixed-m) showed the winning
``block_q`` is the one that minimizes the *padded* query rows
``ceil(seq / block_q) * block_q`` (a near-empty final q-block wastes a whole
tile of compute), among block_q large enough to fill the MXU N dimension and
small enough to fit VMEM.

So block_q is chosen by a tiny closed-form search (no device), not a constant:
the resolution, frame count, and generation all flow through.

Derivation of the other knobs
-----------------------------
* ``block_kv`` = ``block_kv_compute`` = ``safe_kv_block`` (= 8 * mxu_dim = 2048
  on v7): the K/V DMA tile and the MXU compute tile (QK^T's M dimension). Taking
  the whole cliff-safe K/V block as one compute tile maximises MXU fill; 2048
  beat 1024 at every swept resolution and never tripped the Mosaic cliff.
* ``block_q`` (above): cost-model search over the VMEM-feasible range, choosing
  the minimum padded-q tile; ties broken toward the larger block_q (fewer grid
  rows). The VMEM ceiling is the f32 QK^T score tile,
  ``block_kv_compute * block_q * 4 + operands <= usable_fraction * VMEM``,
  and the whole kernel's footprint (``vmem_footprint_bytes``) must fit
  physical VMEM.

Asymmetric Q / K/V extents
--------------------------
In cross-attention the query sequence length differs from the K/V
sequence length. Every entry point takes an optional ``kv_seq_len`` for
this; the K/V block clamps to the K/V extent and the grid cost uses both
extents.
"""

import dataclasses
import functools
import logging
import math
from typing import Any

_logger = logging.getLogger(__name__)


# Structural constants as multiples of the MXU edge, so they scale with the
# hardware. They reproduce the measured v7 optima (mxu_dim 256 ->
# safe_kv_block 2048). Re-verify per generation via a sweep.
_SAFE_KV_BLOCK_MULTIPLE = 8
_DEFAULT_USABLE_VMEM_FRACTION = 0.93
# VMEM kept free under vmem_footprint_bytes(): Mosaic reserves some for
# itself (64 MiB on v7x reports 63.94 MiB usable).
_VMEM_RESERVE_BYTES = 2**20

# Fixed-m drops the VPU softmax rescale that hid DMA behind compute, so its
# score tile must double-buffer to hide DMA -> fits <= VMEM/2 (fraction ~0.5).
# Per-generation, sweep-pinned like safe_kv_block; no entry = normal fraction.
_FIXED_M_KERNEL_MODES = frozenset({"fixedm"})
# v7x: fixed-m measured fastest at block_q 3072-4096 at 32K and 75K tokens
# (40 heads x 128); 0.6 caps block_q at 4096 for head_dim 128. The v6 entry
# is inherited from the original implementation and not measured here.
_FIXED_M_VMEM_FRACTION: dict[int, float] = {6: 0.5, 7: 0.6}

# Generations whose block sizes were measured on hardware (no "uncalibrated"
# warning). Only v7x has been.
_KNOWN_GENERATIONS = frozenset({7})


@dataclasses.dataclass(frozen=True)
class TpuSpec:
    """Hardware parameters that drive the block-size derivation.

    The first group is queried from ``get_tpu_info``; the second is calibrated
    (structural multiples of ``mxu_dim``).

    Attributes:
      name: Human-readable device name.
      generation: TPU generation integer (7, 8, ...).
      vmem_bytes: Vector memory per core (the Pallas/Mosaic kernel budget).
      mxu_dim: MXU systolic array edge length (the multiple-of rule).
      safe_kv_block: K/V block size (DMA + MXU compute tile); largest that does
        not trip the Mosaic cliff.
      usable_vmem_fraction: Fraction of vmem_bytes the score tile + operands may
        occupy, leaving headroom for Mosaic scratch and double buffering.
      num_lanes: Vector lane count (128 on v7+), determining activation layouts.
    """

    name: str
    generation: int
    vmem_bytes: int
    mxu_dim: int
    safe_kv_block: int
    usable_vmem_fraction: float = _DEFAULT_USABLE_VMEM_FRACTION
    num_lanes: int = 128


# Known-good v7 spec, used for tests and as the fallback off a TPU host.
V7_SPEC = TpuSpec(
    name="v7",
    generation=7,
    vmem_bytes=64 * 2**20,
    mxu_dim=256,
    safe_kv_block=2048,
)


def spec_from_tpu_info(info: Any) -> TpuSpec:
    """Builds a TpuSpec from a ``get_tpu_info()`` result.

    Args:
      info: The object returned by
        ``jax.experimental.pallas.tpu.get_tpu_info``; must expose generation,
        mxu_column_size, vmem_capacity_bytes, and num_lanes.

    Returns:
      A TpuSpec with device-queried fields plus the calibrated K/V block.
    """
    mxu_dim = int(info.mxu_column_size)
    generation = int(info.generation)
    return TpuSpec(
        name=f"v{generation}",
        generation=generation,
        vmem_bytes=int(info.vmem_capacity_bytes),
        mxu_dim=mxu_dim,
        safe_kv_block=_SAFE_KV_BLOCK_MULTIPLE * mxu_dim,
        num_lanes=int(info.num_lanes),
    )


@functools.cache
def detect_spec() -> TpuSpec:
    """Returns the TpuSpec for the attached accelerator.

    Queries the live TPU via ``get_tpu_info`` when one is present, otherwise
    falls back to ``V7_SPEC`` (e.g. on a CPU host). Unknown generations get the
    structural-multiple defaults with a warning to re-run the sweep.

    Returns:
      The detected (or fallback) TpuSpec.
    """
    import jax

    devices = jax.devices()
    if not devices or devices[0].platform != "tpu":
        _logger.warning("No TPU detected; falling back to %s spec.", V7_SPEC.name)
        return V7_SPEC

    from jax.experimental.pallas import tpu as pltpu

    spec = spec_from_tpu_info(pltpu.get_tpu_info())
    if spec.generation not in _KNOWN_GENERATIONS:
        _logger.warning(
            "TPU generation %d is not explicitly calibrated; using structural "
            "defaults (safe_kv_block=%d, mxu_dim=%d, vmem=%dMiB). Run the splash "
            "block sweep to verify.",
            spec.generation,
            spec.safe_kv_block,
            spec.mxu_dim,
            spec.vmem_bytes // 2**20,
        )
    return spec


def score_tile_bytes(
    block_q: int,
    block_kv_compute: int,
    head_dim: int,
    block_kv: int,
    bytes_per_element: int = 2,
) -> int:
    """Estimates the VMEM footprint of one kernel step in bytes.

    The f32 QK^T score tile dominates; the rest is the double-buffered q/k/v
    tiles, the f32 output accumulator, and the max/sum scratch.

    Args:
      block_q: Query block size.
      block_kv_compute: MXU compute block size (score tile rows).
      head_dim: Per-head feature dimension.
      block_kv: Key/value memory block size (double buffered).
      bytes_per_element: Activation element size (bf16 = 2).

    Returns:
      The estimated resident VMEM footprint in bytes.
    """
    score_f32 = block_kv_compute * block_q * 4
    qkv_2x = 2 * (block_q + 2 * block_kv) * head_dim * bytes_per_element
    output_f32 = block_q * head_dim * 4
    max_sum_scratch = 2 * 8 * block_q * 4
    return score_f32 + qkv_2x + output_f32 + max_sum_scratch


def vmem_footprint_bytes(
    block_q: int,
    block_kv_compute: int,
    head_dim: int,
    block_kv: int,
) -> int:
    """Upper bound on the VMEM one kernel instance allocates, in bytes.

    Unlike ``score_tile_bytes`` (the performance budget), this has to cover
    every buffer Mosaic allocates. Fitted to the VMEM Mosaic reports on v7x
    (head_dim 64 and 128, block_kv 512 to 2048, block_q 2048 to 18944). Per
    query row: the f32 score tile (``4 * block_kv_compute``, held in
    register spill slots) plus the double-buffered bf16 q and output blocks
    and the f32 o, m and l scratch (at most ``13 * head_dim``). Plus the
    double-buffered bf16 K and V blocks. It is 0.2 to 2.7 MiB above every
    allocation Mosaic reported.

    Args:
      block_q: Query block size.
      block_kv_compute: MXU compute block size (score tile rows).
      head_dim: Per-head feature dimension.
      block_kv: Key/value memory block size.

    Returns:
      The VMEM upper bound in bytes.
    """
    per_query_row = 4 * block_kv_compute + 13 * head_dim
    return block_q * per_query_row + 8 * block_kv * head_dim


def _vmem_max_block_q(
    seq_len: int,
    block_kv_compute: int,
    block_kv: int,
    head_dim: int,
    bytes_per_element: int,
    spec: TpuSpec,
) -> int:
    """Largest block_q (multiple of mxu_dim, <= seq) that fits VMEM.

    The score tile has to fit the performance budget (``score_tile_bytes``
    within ``usable_vmem_fraction``) and the whole kernel has to fit physical
    VMEM (``vmem_footprint_bytes``, less ``_VMEM_RESERVE_BYTES``).
    """
    budget = int(spec.vmem_bytes * spec.usable_vmem_fraction)
    hard_limit = spec.vmem_bytes - _VMEM_RESERVE_BYTES
    best = spec.mxu_dim
    block_q = spec.mxu_dim
    while block_q <= seq_len:
        fits = (
            score_tile_bytes(
                block_q, block_kv_compute, head_dim, block_kv, bytes_per_element
            )
            <= budget
            and vmem_footprint_bytes(block_q, block_kv_compute, head_dim, block_kv)
            <= hard_limit
        )
        if not fits:
            break
        best = block_q
        block_q += spec.mxu_dim
    return best


def tile_grid_cost(
    seq_len: int,
    kv_seq_len: int,
    block_q: int,
    block_kv: int,
    spec: TpuSpec,
) -> int:
    """Model cost of a (block_q, block_kv) tile grid (lower is better).

    Total cost = number of grid cells times per-cell cost, where a cell's
    cost is its padded MXU work (``bq * bkv`` score elements) plus a fixed
    per-cell overhead (DMA setup, pipeline barriers, scratch init) calibrated
    as ``mxu_dim * safe_kv_block`` score-element-equivalents:

        ceil(seq/bq) * ceil(kv_seq/bkv) * (bq * bkv + mxu_dim * safe_kv_block)

    At a fixed ``block_kv`` this reduces to the original 1D model
    ``ceil(seq/bq) * (bq + mxu_dim)`` (scaled by the constant
    ``ceil(kv/bkv) * bkv``), so within-family picks are unchanged. Across
    K/V families it reproduces the measured v7 kernel ranking at seq 75600:
    6912/2048 < 5888/2048 < 8448/1024 (45.4 < 46.3 < 46.9 ms measured).

    Args:
      seq_len: Unpadded query sequence length.
      kv_seq_len: Unpadded key/value sequence length.
      block_q: Candidate query block size.
      block_kv: Candidate key/value block size.
      spec: Target hardware spec (per-cell overhead calibration).

    Returns:
      The model cost (lower is better).
    """
    cells = math.ceil(seq_len / block_q) * math.ceil(kv_seq_len / block_kv)
    cell_overhead = spec.mxu_dim * spec.safe_kv_block
    return cells * (block_q * block_kv + cell_overhead)


def _search_block_q(
    seq_len: int,
    block_kv_compute: int,
    block_kv: int,
    head_dim: int,
    bytes_per_element: int,
    spec: TpuSpec,
    kv_seq_len: int | None = None,
) -> int:
    """Cost-model search for block_q over the VMEM-feasible range.

    Minimises ``tile_grid_cost`` (padding + per-cell overhead) over every
    multiple of the MXU edge that fits VMEM and yields grid height >= 2. Ties
    break toward fewer padded rows, then the larger block_q. The cost itself
    penalises tiny block_q, so no explicit lower bound is needed.

    Args:
      seq_len: Unpadded query sequence length.
      block_kv_compute: Chosen MXU compute block (sets the VMEM ceiling).
      block_kv: Chosen K/V block.
      head_dim: Per-head feature dimension.
      bytes_per_element: Activation element size in bytes.
      spec: Target hardware spec.
      kv_seq_len: Unpadded key/value sequence length; ``None`` means
        self-attention with ``kv_seq_len == seq_len``.

    Returns:
      The chosen block_q (a multiple of mxu_dim, giving grid height >= 2).
    """
    kv_seq = kv_seq_len if kv_seq_len is not None else seq_len
    vmem_max = _vmem_max_block_q(
        seq_len, block_kv_compute, block_kv, head_dim, bytes_per_element, spec
    )
    candidates = [
        bq
        for bq in range(spec.mxu_dim, vmem_max + 1, spec.mxu_dim)
        if math.ceil(seq_len / bq) >= 2
    ]
    if not candidates:  # seq <= mxu_dim: single full-extent block.
        candidates = [seq_len]
    # Tie-break at equal model cost: prefer the LARGER block_q — the cost
    # already accounts for padding and per-cell overhead, and at equal
    # cost fewer, larger tiles mean fewer DMA issues (measured at the
    # ring shard 27900: cost-tied 4864 beat 2816 by 5.7%).
    return min(
        candidates,
        key=lambda bq: (
            tile_grid_cost(seq_len, kv_seq, bq, block_kv, spec),
            -bq,
        ),
    )


def autotune_block_sizes(
    seq_len: int,
    head_dim: int = 128,
    bytes_per_element: int = 2,
    spec: TpuSpec | None = None,
    kv_seq_len: int | None = None,
    kernel_mode: str = "online",
) -> dict[str, int]:
    """Derives custom-splash block sizes for a sequence length.

    See the module docstring for the derivation. The result is independent of the
    context-parallel shard count: sharding changes how many heads a core owns (the
    parallel grid dimension), not the per-tile economics this tunes.

    Args:
      seq_len: Unpadded query sequence length.
      head_dim: Per-head feature dimension (the fixed QK^T contraction).
      bytes_per_element: Activation element size in bytes (bf16 = 2).
      spec: Target hardware spec. Auto-detected from the live TPU when omitted.
      kv_seq_len: Unpadded key/value sequence length when it differs from
        ``seq_len`` (e.g. cross-attention). ``None`` means self-attention with
        ``kv_seq_len == seq_len``.
      kernel_mode: Which kernel variant consumes the blocks (``online`` or
        ``fixedm``).

    Returns:
      A mapping with block_q, block_kv and block_kv_compute.
    """
    if spec is None:
        spec = detect_spec()
    kv_seq = kv_seq_len if kv_seq_len is not None else seq_len

    # Fixed-m tightens the block_q VMEM budget to the double-buffer ceiling
    # (~VMEM/2; see _FIXED_M_VMEM_FRACTION), keeping it out of the DMA cliff.
    # min() only lowers the caller's fraction; no entry for the gen is a no-op.
    if kernel_mode in _FIXED_M_KERNEL_MODES:
        fixed_m_fraction = _FIXED_M_VMEM_FRACTION.get(spec.generation)
        if fixed_m_fraction is not None:
            spec = dataclasses.replace(
                spec,
                usable_vmem_fraction=min(spec.usable_vmem_fraction, fixed_m_fraction),
            )

    # K/V block and MXU compute tile: the whole cliff-safe block, clamped to
    # the K/V extent (not the query extent — they differ under 2D CP). A
    # shorter K/V is one full-extent block, which is legal at any length.
    block_kv = min(spec.safe_kv_block, kv_seq)
    block_kv_compute = block_kv

    # block_q: cost-model search over the VMEM-feasible range.
    block_q = _search_block_q(
        seq_len,
        block_kv_compute,
        block_kv,
        head_dim,
        bytes_per_element,
        spec,
        kv_seq_len=kv_seq,
    )

    return {
        "block_q": block_q,
        "block_kv": block_kv,
        "block_kv_compute": block_kv_compute,
    }
