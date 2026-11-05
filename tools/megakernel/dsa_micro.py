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
"""Kimi-K3.1 DSA decode microbenchmarks (``bench dsa-micro``).

Each case is a JAX body over the 32-rank ``tp`` mesh, registered as a torch op
exactly as the decode ops are, so a body may issue remote DMAs and semaphore
signals between ranks. :func:`time_detail` times a body by the device-loop slope:
the body runs ``N1`` and ``N2`` chained iterations inside one ``fori_loop``
call each, and the per-iteration time is ``(T(N2) - T(N1)) / (N2 - N1)`` of the
slowest rank, the median over repeats. Launch, compile and readback cost are
the same in both calls and cancel. Every SMEM operand (a table a kernel reads
as scalars) has at least eight entries along its last axis (:func:`smem_table`).

Calibration cases (any later case is compared against them):

* ``copy``: one VMEM -> HBM DMA of ``R x 640`` BF16 per iteration (the
  kernel first fills the VMEM source with a small-integer pattern).
* ``barrier``: ``collectives32.barrier()`` per iteration, one signal to each of
  the 31 peers and a wait.

``select`` is the decode's exact top-``SELECT_K`` over the 32 DCP ranks' local
scores ``[R, T]`` (FP32, in VMEM), as a radix-histogram select (``--variant
radix4|radix8``: ``b``-bit digits, ``32 / b`` rounds):

* Keys are the order-preserving uint32 of the scores (:func:`order_keys`). Round
  ``j`` counts, per row and digit value ``d``, the local keys at or above
  ``prefix + d << shift`` with VPU compares summed over lane tiles (no
  scatter); the group sums the ``[R, 2^b]`` counts and every rank takes the same
  digit (:func:`radix_digit`). After the last round the threshold key ``tau``
  and the quota ``q`` of keys equal to it are known. ``--histogram packed``
  instead counts each digit value among the prefix's keys in 8-bit counters,
  four per int32 lane, and sums them from the top (:func:`cumulative_counts`).
* Ties: one more exchange of per-local-page counts of keys equal to ``tau``;
  each rank takes its ties in global token order (:func:`tie_offsets`), so the
  selection equals ``dsa_reference.select_topk``.
* Rows of at most ``SELECT_K`` tokens select every token, without rounds.
* ``--tie-skip on`` (tag ``_a``): the last round's chosen count is the count at
  or above ``tau``, the same on every rank; the tie round runs only when it
  exceeds ``SELECT_K`` on some row, and otherwise every key at or above
  ``tau`` is selected, exactly ``select_topk``'s set. A second output ``stats
  [V, 128]`` holds per virtual row whether its ties overflow the quota, whether
  the round ran and that count, which the check compares with
  :func:`tie_round_needed` and records (``tie_rows``, ``tie_round_ran``).
* ``--row-sum before`` (tag ``_e``, below eight rows): a row's virtual rows'
  counts are summed before the exchange into one ``[8, 128]`` packet of the
  ``R x 2^b`` counts (:func:`row_packet`), and copied back to them after it.
* ``--scores``: ``normal``, ``ties`` (rounded to few values), ``exact``
  (:func:`exact_tie_scores`: threshold ties that exactly fill the quota or
  overflow it by one, spread over ranks and pages) or ``indexer`` (the
  ``score`` case's ``bf16q`` kernel output on every rank, gathered once a
  point).
* ``--exchange flat`` sends every count packet to the 31 peers; ``host`` sums
  the 8 lanes of a host first, then the 4 host sums over the lane's peers.
  Packets are remote DMAs into per-round-parity inboxes with byte-counted
  waits; the only barrier is one at kernel entry. ``none`` (no exchange, no
  barrier) and ``handshake`` (the entry barrier only) reuse the local counts:
  timing only, never checked, as is ``--ties off`` (the rounds without the tie
  step).
* ``--exchange merge`` (R <= 8, as the GPU DCP indexer's merge): each rank's
  candidates are its valid local tokens or, above ``SELECT_K`` of them, its
  local top-k (the rounds without an exchange; local ties to the lower index,
  their round only when they overflow), ``min(T, SELECT_K)`` int32 keys and
  global token ids a row (:func:`merge_candidates_model`). Above ``SELECT_K``
  each virtual row compacts a plane's selected lanes into 128-lane rows of its
  row's candidates (:func:`_merge_compact`). One all-gather writes each rank's
  block into its slot of every peer's inbox (31 remote DMAs, byte-counted
  waits; one inbox, as the entry barrier separates calls). Every rank then runs
  the rounds over a row's ``RANKS min(T, SELECT_K)`` gathered keys and, on rows
  whose ties overflow the quota, a second select over the tied candidates'
  negated ids for the quota's lowest ids (:func:`merge_select_model`); it marks
  its own candidates above the threshold and its tied ones among those ids.
  Exact: a token of the global top-k is in its owner's local top-k. With
  ``--ties off`` (timing only) the kernel skips the all-gather, the ids'
  select and the local tie round: the rounds run over the peers' stale slots.

``gather`` reads each row's ``S`` selected local tokens' latent rows out of a
rank's paged pool (``[blocks, 128, packing, 640]``, BF16 or FP8 pages) into
VMEM, for ``R`` rows whose blocks are scattered over the pool:

* ``--variant row|tile|page``: one DMA per selected token (the sublane row that
  holds it: a 1-D slice of one token does not lower, as Mosaic slices a packed
  dimension only in whole sublane rows), one per distinct 8-sublane-row tile
  holding a selected token (the cache commit kernel's tile), or one per distinct
  block. The lists are built on the host (ascending, deduplicated per row) and
  read by the kernel as scalars with the block table.
* ``--pattern uniform|recent``: random positions, or mostly the newest tokens
  plus a few sinks (:func:`select_tokens`).
* The DMAs of all rows go through one ring of ``--inflight`` VMEM slots: the
  first ``inflight`` start at once, each later one after the wait for the DMA
  that used its slot. ``--wait-group`` slots share a semaphore and one
  byte-counted wait. The kernel ends with one dependent read of the last slot.
* ``--verify on`` also runs a kernel that consumes each slot after its wait,
  adding the slot's 32-bit words, weighted by the sublane row's position in the
  slot and by the DMA's number (odd weights), to the row's accumulator; the
  host recomputes them from the pool bits.

``compact`` turns the select's output (its int32 mask words ``[words, V,
128]``, DMA'd into VMEM) into the gather's DMA lists in SMEM, per virtual row
(:class:`CompactLists`):

* ``--variant row|tile|page``: one code ``block << 8 | first sublane row`` per
  distinct sublane row, tile or block holding a selected token (row and tile
  lists per virtual row, page lists per row on its first part), the first
  ``--capacity`` of them, each with its token bitmask, plus the counts of
  selected tokens, sublane rows, tiles and pages.
* Per block of eight virtual rows, per 128-token plane, selector products on
  the MXU give each sublane row's and tile's token count and the plane's token
  and unit bits (:func:`_compact_selectors`). The running sums of the planes'
  unit counts give each list position its plane, and a rank-select of that
  plane's unit bits its unit (:func:`compact_model`).
* Each block's ``[3, 8, C]`` tile of codes, bitmasks and counts moves to one of
  two SMEM slots (``--smem-path dma``: one DMA; ``hbm``: through an HBM
  buffer), waited for by the next block. The timed body copies the last block's
  SMEM chunk to HBM, the checked one every block's, which the check compares
  with :func:`compact_expected`.
* The selections differ per rank (``--pattern``, ``--selected``); ``--row-mix
  on`` cycles rows of S, no, one and every token.

``fold`` is K3's MLA decode attend (``mla_dcp.DcpAttention``: QK, the FP32
online softmax, PV, then ``o = O / l`` and ``lse``) of ``R`` rows' 96 absorbed
query heads over each row's gathered latent rows, staged contiguously in HBM
(:func:`fold_stage`, :func:`_fold_kernel`):

* ``--variant row|tile|page``: every token of each distinct sublane row, tile
  or block holding a selected token, folded under the selection's lane mask
  (32 tokens a SMEM word); ``dense``: the selected tokens only; ``none``: no
  tokens (each row's fixed cost). The selections are the same on every rank
  (``--pattern``, ``--selected``, ``--row-mix``); the pool and the queries
  differ.
* The rows' streams go through one ring of ``FOLD_DEPTH`` VMEM slots of
  ``--fold-tokens`` N tokens (a step), and each row's query through two slots.
  Each slot's lanes 576:640, NaN when staged, are zeroed before use. BF16 pages
  take BF16 queries; FP8 pages the decode's FP8 query wire.
* ``--fold-compaction onehot|copy`` (``tile``, ``page``): each chunk's
  selected tokens move into a dense VMEM buffer first (one-hot dots of 128
  dense rows, or one token at a time), which is then folded in steps of N.
* ``--fold-compute off`` streams the chunks without folding them (timing
  only). The timed body writes the last row's result to HBM, the checked one
  every row's, which the check compares with :func:`fold_reference`
  (``dsa_reference.sparse_mla_attention``) within ``FOLD_TOLERANCE``, and
  whose ``lse`` must be within :func:`fold_exact_lse_tolerance` of
  :func:`fold_exact_lse` (the exact log-sum-exp of the kernel's own values).

``score`` is the indexer's scoring of a rank's ``T`` local index keys for
``R`` rows: ``s[r, t] = 2^(e_t - 127) sum_h w[r, h] ReLU(q[r, h] . k_t)``, the
keys ``k_t`` FP8 in index blocks (``dsa_cache``) in HBM, ``e_t`` their
exponent bytes:

* The blocks stream through a ring of VMEM slots, ``--prefetch`` of them in
  flight while one is scored (1: double buffered).
* Per block, MXU dots of the ``[32 R, 128]`` queries (``--variant bf16q``: FP8
  values held in BF16, an exact dot against the FP8 keys; ``fp8q``: FP8, not
  exact) with the keys, ``--dot-tokens`` keys and ``--dot-groups`` row groups
  a dot. The query rows go in groups of eight rows (or all of them below
  eight), head-major inside a group (:func:`score_query_rows`), so that the head
  sum is a sum of aligned ``[8, 128]`` tiles, then (below eight rows) rolls
  over sublanes (:func:`head_sum_model`). The scale multiplies the sum (exact:
  a positive power of two), formed from the exponent bytes in the kernel.
* The scores ``[R_pad, T]`` FP32 (``R_pad``: R rounded up to 8) stay in VMEM,
  the select's input. ``--writeback slice`` (timed by default) copies only the
  last 128 tokens' scores to HBM; ``full`` copies all of them, the body the
  check reads.
* ``--layout select`` keeps them in the select's own layout instead, ``[planes,
  V, 128]`` (:func:`to_virtual`): ``V`` = 8 and ``g = 8 / R`` parts a row below
  eight rows, ``V = R`` and ``g = 1`` from eight, ``planes = T / (g 128)``. At
  R = 2, 4 the query rows put each row's heads on its virtual rows' sublanes
  (``score_query_rows(rows, "select")``) and the select's ``group_sum`` gives
  each of them the row's total; R = 1 keeps the rolls by 1, 2, 4, and R >= 8
  has none. Below eight rows each 128-key column is one read-modify-write of
  its plane's ``[8, 128]`` tile, on the sublanes of its part
  (:func:`virtual_place`). The timed body copies the last plane to HBM.
* ``--emit keys`` stores the select's int32 keys (:func:`signed_keys`, the
  inverse :func:`scores_of_keys`) instead of FP32 scores: ``INT32_MIN`` past
  each row's local tokens (``--lengths``, as the select's:
  ``local_counts(length, T, block_tokens)[rank]``) and on padded rows. The
  counts come in as an SMEM ``meta`` operand (R entries, at least eight).
* ``--score-pages per-row`` gives each row its own ``T / block_tokens`` blocks,
  scattered over a pool of ``R T / block_tokens`` by :func:`block_table`
  (``shared``, the default: every row scores the same blocks). One ring
  streams the pages (the block table, in stream order, read from SMEM); a
  page's dots are its row's ``[32, 128]`` heads with ``--dot-tokens`` of the
  page's keys each.
  ``--pages-per-step K`` (1, 2, 4, 8) scores K pages a loop step
  (:func:`score_stream`): K consecutive blocks of one row (row-major), or at
  K = 8 from eight rows one block of rows ``8 t .. 8 t + 7`` (block-major). The
  ring holds ``(D + 1) K`` slots, ``D = ceil(prefetch / K)`` steps in flight;
  a step issues its K dots per dot's keys before their head sums. Row-major
  pages put each total on its row's sublane (rolls by 1, 2, 4 and a
  read-modify-write); block-major steps combine the eight rows' totals into
  one tile (:func:`combine_rows`) and store it. ``--score-work compute-only``
  (the ring filled once, no DMA in the loop) and ``dma-only`` (pages streamed,
  not scored) are per-row timing-only arms, never checked.
* ``--score-pages decode`` times the decode kernel's own scorer
  (``dsa_decode.IndexScorer``, :func:`_decode_score_body`) over the per-row
  pages: local tables, every row at position ``RANKS T - 1`` (each rank holds
  ``T`` local tokens a row; the new token, on rank 31, is the row's last local
  key), primed and called as the decode layer calls it
  (:func:`decode_call_rows`), the scores in its ``[planes, V, 128]`` layout.
  It takes ``--layout select --emit scores``, one page a step, full work and
  ``bf16q`` (``--variant fp8q`` is refused); its ring, dots and row groups are
  the scorer's own, and its record's ``prefetch`` is the ring's
  ``dsa_decode.RING_DEPTH`` pages in flight. The check bounds each row by
  ``dsa_decode.local_length``.
* Host side: ``score_plans(rows, tokens, args)`` and ``_build_score(rows,
  tokens, args)`` (``--layout``, ``--emit``, ``--score-pages``,
  ``--pages-per-step``, ``--score-work``, ``--lengths``); ``_score_setup(rows,
  tokens, variant, plan, lengths=())``, whose :class:`ScorePlan` carries
  ``layout``, ``emit``, ``pages``, ``pages_per_step`` and ``work``; the check
  ``_score_check(rows, inputs, bound, extra, tag, *, layout="padded",
  emit="scores", counts=None)``.

``--n2`` defaults to 72 iterations, and to 264 for a case whose slowest rank
comes out under 20 us per iteration (its 72-iteration span is too short against
launch jitter).
"""

from __future__ import annotations

import argparse
import functools
import itertools
import re
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from tools.megakernel import harness

AXIS = "tp"
RANKS = 32
LATENT_LANES = 640
DEFAULT_N2 = 72
SHORT_CALL_US = 20.0
LONG_N2 = 264
SPREAD_FRACTION = 0.05
SELECT_K = 2048
SELECT_SEED = 31
SELECT_BITS = {"radix4": 4, "radix8": 8}
SELECT_EXCHANGES = ("flat", "host", "none", "handshake", "merge")
TIMING_ONLY = ("none", "handshake")
HOST_LANES = 8
LANES = 128
SUBLANES = 8
BIN_CHUNK = 16  # digit values counted per pass over the keys
SELECT_HISTOGRAMS = ("cumulative", "packed")
SELECT_ROW_SUMS = ("after", "before")
FIELD_PLANES = 255  # planes an 8-bit packed counter may count before widening
SELECT_VMEM_LIMIT = 48 * 2**20
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
GATHER_VARIANTS = ("row", "tile", "page")
GATHER_PATTERNS = ("uniform", "recent")
GATHER_DTYPES = ("bf16", "fp8")
GATHER_SELECTED = (16, 64, 256, 2048)
GATHER_INFLIGHT = 64
GATHER_SEED = 37
GATHER_VMEM_LIMIT = 56 * 2**20
POOL_ROWS = 128  # sublane rows of a pool block
TILE_ROWS = 8  # the cache commit kernel's native tile
GATHER_SLOT_ROWS = {"row": 1, "tile": TILE_ROWS, "page": POOL_ROWS}
BALANCED_SELECTED = SELECT_K // RANKS
RECENT_TOKENS = 128  # the last 4096 global positions, as local tokens
SINK_TOKENS = 128
RECENT_SHARE = 0.75
SINK_SHARE = 0.05
SCORE_VARIANTS = ("bf16q", "fp8q")
SCORE_BLOCK_TOKENS = (256, 512)
SCORE_WRITEBACKS = ("slice", "full")
SCORE_LAYOUTS = ("padded", "select")
SCORE_EMITS = ("scores", "keys")
SCORE_PAGES = ("shared", "per-row", "decode")
SCORE_PAGES_PER_STEP = (1, 2, 4, 8)
SCORE_WORKS = ("full", "compute-only", "dma-only")
SCORE_DOT_TOKENS = 128
SCORE_DOT_GROUPS = 1
SCORE_PREFETCH = 1
SCORE_SEED = 41
SCORE_EXPONENTS = 4  # key magnitudes 2^k, k uniform in [-4, 4] per token
SCORE_TOLERANCE = 1e-5  # bf16q: FP32 reassociation only
# bf16q keys, relative to the valid tokens only: 2 x the largest error seen, one
# significant digit.
SCORE_KEYS_TOLERANCE = 2e-5
FP8_SCORE_BOUND = 0.1  # fp8q: catches layout errors (order one), not precision
SCORE_REFERENCE_TOKENS = 4096  # keys per CPU reference einsum
GLOBAL_RECALL_ROWS = 2  # up to these rows every rank gathers every rank's scores
SCORE_VMEM_LIMIT = 48 * 2**20
COMPACT_SEED = 43
COMPACT_CAPACITY = 128
COMPACT_VMEM_LIMIT = 48 * 2**20
COUNT_FIELDS = ("selected", "rows", "tiles", "pages")
SMEM_PATHS = ("dma", "hbm")
COMPACT_UNROLL = 8  # mask words (phase 1) and planes (phase 3) per loop step
FOLD_SEED = 47
FOLD_HEADS = 96  # 32 DCP members x 3 head slots (mla_dcp.head_rows)
FOLD_TOKENS = 256
FOLD_DEPTH = 4  # ring slots, as mla_dcp.BUFFERS
FOLD_VMEM_LIMIT = 48 * 2**20
FOLD_SMEM_WORDS = 32768  # a guard on one call's SMEM tables, not a measured limit
FOLD_DENSE_TOKENS = 4096  # the guard on a row's compacted buffer
FOLD_VARIANTS = ("row", "tile", "page", "dense", "none")
FOLD_COMPACTIONS = ("none", "onehot", "copy")
WORD_TOKENS = 32  # stream tokens per mask word
FOLD_CHECK_ROWS = 8
FOLD_CHECK_TOKENS = 32768  # selected tokens of a call up to which every row is checked
META_FIELDS = 6
STEP_FIRST = 1
STEP_LAST = 2
STEP_FOLD = 4
STEP_COMPACT = 8
STEP_DENSE = 16
STEP_EMPTY = 32
NAN_BITS = {"bf16": 0x7FC0, "fp8": 0x7F}
# 2 x the kernel-dtype model's largest error, one significant digit: (o, lse).
FOLD_TOLERANCE = {"bf16": (3e-2, 3e-2), "fp8": (3e-1, 4e-1)}
# The kernel's lse against fold_exact_lse, by the same rule, at least 1e-4.
# Interpreted on CPU the kernel's dots are exact, so the model's deviation sets
# it for both dtypes. On TPU the FP8 x FP8 MXU dots are not exact, so the FP8
# bound comes from the TPU kernel's own deviation; BF16 dots are exact there.
FOLD_EXACT_LSE_TOLERANCE = {"bf16": 1e-4, "fp8": 1e-4}
FOLD_EXACT_LSE_TOLERANCE_TPU = {"bf16": 1e-4, "fp8": 2e-1}


@dataclass
class Case:
    """One result record (``--json``): ``tokens`` is 0 for a case that does not
    depend on the local token count, ``check`` None for one with no check and
    ``"invalid"`` for a timing-only ablation."""

    case: str
    rows: int
    tokens: int
    variant: str
    us: float
    check: bool | str | None


@dataclass
class Setup:
    """A case at one (rows, tokens) point. ``body(*operands, iterations)`` runs
    the iterations chained through a ``fori_loop`` (iteration k + 1 consumes a
    value or buffer iteration k produced); ``operands`` are this rank's torch
    tensors for ``in_specs``. ``check`` receives this rank's outputs of one
    call of the body that ran ``--n1`` chained iterations (so it must hold
    after any number of them) and says whether they are right on this rank
    (:func:`reduce_check` collects the ranks' verdicts); ``invalid`` marks a
    timing-only ablation whose outputs are wrong by design. ``check_body`` (same
    operands and specs) is the body whose outputs ``check`` reads when the
    timed body does not produce checkable ones; ``extra`` is added to the
    case's JSON record (a check may add what it measured to it)."""

    body: Callable[..., Any]
    operands: tuple
    in_specs: tuple
    out_specs: tuple
    variant: str = ""
    check: Callable[[Sequence[Any]], bool] | None = None
    invalid: bool = False
    check_body: Callable[..., Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseDef:
    """``build(rows, tokens, args)``: the case's setups at one point."""

    build: Callable[[int, int, Any], list[Setup]]
    per_tokens: bool


def slopes(samples: Sequence[Sequence[tuple[float, float]]], n1: int, n2: int):
    """Per-iteration microseconds of ``(T(n1), T(n2))`` seconds samples:
    ``samples[rank][repeat]`` in, the same nesting out."""
    return [[1e6 * (t2 - t1) / (n2 - n1) for t1, t2 in rank] for rank in samples]


def slowest_rank_median(
    samples: Sequence[Sequence[tuple[float, float]]], n1: int, n2: int
) -> float:
    """The median over repeats of the slowest rank's slope in each repeat."""
    per_rank = slopes(samples, n1, n2)
    return statistics.median(
        max(rank[i] for rank in per_rank) for i in range(len(per_rank[0]))
    )


def check_lengths(n1: int, n2: int, repeats: int) -> None:
    if not 1 <= n1 < n2 or repeats < 1:
        raise ValueError(
            f"need 1 <= n1 < n2 and repeats >= 1, got n1={n1}, n2={n2}, "
            f"repeats={repeats}"
        )


_OPS: dict[str, tuple[tuple, Callable[..., Any]]] = {}


def _op(name, body, operands, in_specs, out_specs, iterations):
    """The torch op running ``body`` for ``iterations``, registered once per
    (``name``, ``iterations``). ``name`` identifies the body, specs and operand
    shapes: reusing it for another raises. Characters a torch op name does not
    allow become ``_``."""
    op_name = f"pallas::dsa_micro_{re.sub(r'[^0-9A-Za-z_]', '_', name)}_n{iterations}"
    signature = (
        body,
        tuple(in_specs),
        tuple(out_specs),
        tuple((tuple(o.shape), str(o.dtype)) for o in operands),
    )
    if op_name not in _OPS:
        from vllm_torchtpu.megakernel import bridge

        mesh, _ = bridge.kernel_mesh(AXIS, RANKS)
        op = bridge.register_op(
            op_name,
            mesh,
            lambda *operands: body(*operands, iterations),
            in_specs,
            out_specs,
        )
        _OPS[op_name] = (signature, op)
    elif _OPS[op_name][0] != signature:
        raise ValueError(
            f"{op_name} was registered for another body, specs or operand "
            "shapes; give each case point its own name"
        )
    return _OPS[op_name][1]


def call_body(name, body, operands, in_specs, out_specs, iterations):
    """One call of ``body`` for ``iterations``: its outputs."""
    return _op(name, body, operands, in_specs, out_specs, iterations)(*operands)


def local_verdict(check, outputs) -> str | None:
    """None if ``check(outputs)`` holds, else why not. A check that raises fails
    here instead of leaving its peers waiting in the gather."""
    try:
        return None if check(outputs) else "check returned False"
    except Exception as exc:  # noqa: BLE001 -- reported through the gather
        return f"{type(exc).__name__}: {exc}"


def failing_ranks(verdicts: Sequence[tuple[int, str | None]]) -> dict[int, str]:
    """``{kernel rank: why}`` of the ranks whose verdict is not None."""
    return {rank: why for rank, why in sorted(verdicts) if why is not None}


def reduce_check(check, outputs) -> dict[int, str]:
    """Run ``check`` on every rank's outputs: the failing kernel ranks (empty
    when all pass), the same on every rank."""
    import torch

    from vllm_torchtpu.megakernel import bridge

    mine = (bridge.kernel_mesh(AXIS, RANKS)[1], local_verdict(check, outputs))
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, mine)
    return failing_ranks(gathered)


def _first(outputs):
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


@dataclass
class Timing:
    """``us``: the slowest rank's slope in each repeat, median over repeats;
    ``rank_us``: each rank's median slope, sorted (the noise floor of ``us``)."""

    us: float
    rank_us: list[float]

    @property
    def rank_median(self) -> float:
        return self.rank_us[len(self.rank_us) // 2]

    @property
    def spread_flag(self) -> bool:
        """The headline is probably noise-inflated (see :func:`time_detail`): the
        slowest rank's median slope is well above the median rank's, or the
        headline is well above that slowest rank's own median (different ranks
        spiked in different repeats)."""
        slowest = self.rank_us[-1]
        return (
            slowest - self.rank_median > SPREAD_FRACTION * self.us
            or self.us - slowest > SPREAD_FRACTION * slowest
        )


def time_detail(
    name: str,
    body: Callable[..., Any],
    operands: Sequence[Any],
    in_specs: tuple,
    out_specs: tuple,
    *,
    n1: int,
    n2: int,
    repeats: int,
) -> Timing:
    """The :class:`Timing` of ``body``: microseconds per iteration (the slowest
    rank's median slope) and every rank's median slope.

    ``body(*operands, iterations)`` runs under the 32-rank mesh registration
    (``bridge.register_op``), so it may use collective ids, semaphores and
    remote DMAs, and must chain its iterations through a ``fori_loop``. Both
    loop lengths are compiled and run once untimed first; each timed call
    follows a process-group barrier and ends with one host read of the first
    output. The slowest rank is found from every rank's times, so every rank
    returns the same value.

    The result is biased upward: host launch, readback and sync jitter inflate
    the maximum over 32 independent ranks, so its noise is about
    jitter / (n2 - n1) per slope. The median rank is what to compare against.

    The ``fori_loop`` is outside the ``pallas_call``, so every iteration pays
    one ``pallas_call`` launch, as do the ``copy`` and ``barrier`` calibrations
    (``copy`` also includes its in-kernel fill of the VMEM source). A body that
    loops inside one kernel is not comparable to them. A body with remote DMAs
    carries its own entry barrier, inbox parity and exit handshake: iteration
    k + 1 must not race iteration k's semaphores or buffers.
    """
    import torch

    from vllm_torchtpu.megakernel import bridge

    check_lengths(n1, n2, repeats)
    lengths = (n1, n2)
    ops = {n: _op(name, body, operands, in_specs, out_specs, n) for n in lengths}
    for n in lengths:
        bridge.materialize(_first(ops[n](*operands)))
    samples = []
    for _ in range(repeats):
        times = []
        for n in lengths:
            harness.sync()
            started = time.perf_counter()
            bridge.materialize(_first(ops[n](*operands)))
            times.append(time.perf_counter() - started)
        samples.append(tuple(times))
    import os

    if os.environ.get(
        "DSA_MICRO_TRACE"
    ):  # an xprof trace of one n1-iteration call (bench prefill's _trace)
        from tools.megakernel import bench

        harness.sync()
        with bench._trace(os.environ["DSA_MICRO_TRACE"], name, True):
            bridge.materialize(_first(ops[n1](*operands)))
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, samples)
    timing = Timing(
        slowest_rank_median(gathered, n1, n2),
        sorted(statistics.median(rank) for rank in slopes(gathered, n1, n2)),
    )
    spread = "/".join(
        f"{value:.2f}"
        for value in (timing.rank_us[0], timing.rank_median, timing.rank_us[-1])
    )
    last = [round(1e3 * t, 3) for t in samples[-1]]
    harness.log(
        f"{name}: {timing.us:.2f} us per iteration (N {n1}/{n2}; each rank's "
        f"median slope min/median/max {spread}"
        + (", SPREAD FLAGGED" if timing.spread_flag else "")
        + f"; this rank's T(N) in the last repeat {last} ms)"
    )
    return timing


def measure(
    name: str, setup: Setup, n1: int, n2: int, repeats: int
) -> tuple[int, Timing]:
    """``(n2 used, timing)``; ``n2`` 0: the default, or the long one for a call
    whose slowest rank's own median slope is under ``SHORT_CALL_US`` (the
    headline, a maximum over ranks of each repeat, can be inflated by spikes on
    different ranks in different repeats and is not what decides)."""

    def timed(n):
        return time_detail(
            name,
            setup.body,
            setup.operands,
            setup.in_specs,
            setup.out_specs,
            n1=n1,
            n2=n,
            repeats=repeats,
        )

    if n2:
        return n2, timed(n2)
    timing = timed(DEFAULT_N2)
    if timing.rank_us[-1] < SHORT_CALL_US:
        return LONG_N2, timed(LONG_N2)
    return DEFAULT_N2, timing


def _rank_rng(seed: int):
    from vllm_torchtpu.megakernel import bridge

    return np.random.default_rng([seed, bridge.kernel_mesh(AXIS, RANKS)[1]])


def _upload_bits(bits, dtype):
    """The ``[1, ...]`` device tensor of torch ``dtype`` whose bit pattern is
    ``bits`` (uint8 FP8 or uint16 BF16 words): ``bridge.upload`` cannot wrap an
    FP8 array."""
    import torch

    return torch.from_numpy(bits).view(dtype).unsqueeze(0).to("tpu")


def smem_table(values):
    """``values`` as an SMEM operand: zero-padded to at least eight entries
    along its last axis (an SMEM operand of one entry slows every call). The
    kernels read only their own entries."""
    values = np.asarray(values)
    pad = max(0, SUBLANES - values.shape[-1])
    return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, pad)])


def _carry(rows: int):
    """The per-rank buffer a calibration body carries from iteration to
    iteration: random BF16 ``[1, rows, 640]``, as a latent row block."""
    import ml_dtypes

    from vllm_torchtpu.megakernel import bridge

    values = _rank_rng(rows).standard_normal((rows, LATENT_LANES))
    return bridge.upload(values.astype(ml_dtypes.bfloat16))


def _copy_pattern(rows: int):
    """The rows the ``copy`` kernel writes (small integers: exact in BF16)."""
    import ml_dtypes

    row, lane = np.indices((rows, LATENT_LANES))
    return ((5 * row + lane) & 127).astype(ml_dtypes.bfloat16)


def _build_copy(rows: int, tokens: int, args=None) -> list[Setup]:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import PartitionSpec as P

    def kernel(_, out, rows_vmem, sem):
        shape = (rows, LATENT_LANES)
        row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        rows_vmem[...] = (
            ((5 * row + lane) & 127).astype(jnp.float32).astype(jnp.bfloat16)
        )
        copy = pltpu.make_async_copy(rows_vmem, out.at[0], sem)
        copy.start()
        copy.wait()

    def body(carry, iterations):
        # The aliased carry orders the calls; the kernel only overwrites it.
        call = pl.pallas_call(
            kernel,
            name="dsa_micro_copy",
            out_shape=jax.ShapeDtypeStruct(carry.shape, carry.dtype),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((rows, LATENT_LANES), jnp.bfloat16),
                pltpu.SemaphoreType.DMA,
            ],
            input_output_aliases={0: 0},
        )
        return (jax.lax.fori_loop(0, iterations, lambda _, c: call(c), carry),)

    expected = _copy_pattern(rows).astype(np.float32)

    def check(outputs):
        return bool(np.array_equal(_first(outputs)[0].float().cpu().numpy(), expected))

    return [Setup(body, (_carry(rows),), (P(AXIS),), (P(AXIS),), check=check)]


def _build_barrier(rows: int, tokens: int, args=None) -> list[Setup]:
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel.common import collectives32

    def kernel(_, out):
        collectives32.barrier()

    def body(carry, iterations):
        # The aliased carry orders the calls; the kernel never touches it.
        call = pl.pallas_call(
            kernel,
            name="dsa_micro_barrier",
            out_shape=jax.ShapeDtypeStruct(carry.shape, carry.dtype),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            input_output_aliases={0: 0},
            compiler_params=pltpu.CompilerParams(collective_id=0),
        )
        return (jax.lax.fori_loop(0, iterations, lambda _, c: call(c), carry),)

    return [Setup(body, (_carry(rows),), (P(AXIS),), (P(AXIS),))]


def order_keys(scores):
    """The order-preserving uint32 keys of FP32 ``scores`` (a higher key is a
    higher score), as ``sparsecore_topk`` orders them."""
    bits = np.ascontiguousarray(scores, np.float32).view(np.uint32)
    flip = np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
    return bits ^ flip


def global_token_index(local, rank, page_tokens: int):
    """The global token index of local token ``local`` of DCP rank ``rank``,
    whose j-th local page is global page ``RANKS * j + rank``."""
    from tools.megakernel import dsa_reference

    return dsa_reference.global_token(local, rank, page_tokens, RANKS)


def local_counts(length: int, tokens: int, page_tokens: int):
    """``[RANKS]``: each rank's local candidates in a context of ``length``
    tokens (a prefix of its ``tokens`` local slots)."""
    index = global_token_index(
        np.arange(tokens)[None], np.arange(RANKS)[:, None], page_tokens
    )
    return (index < length).sum(axis=1)


def radix_digit(counts_ge, k: int, upper: int) -> tuple[int, int]:
    """One round's decision from ``counts_ge[d]``, the group's count of keys at
    or above the round's ``d``-th threshold (non-increasing, ``counts_ge[0] >=
    k``): the digit, the last threshold still holding ``k`` keys, and the count
    at or above the next threshold up (``upper``, the previous round's, when the
    digit is the top one). After the last round that count is the number of
    keys above the threshold key."""
    counts_ge = np.asarray(counts_ge)
    digit = int((counts_ge >= k).sum()) - 1
    if digit + 1 < len(counts_ge):
        upper = int(counts_ge[digit + 1])
    return digit, upper


def cumulative_counts(equal, above: int):
    """A round's ``counts_ge`` (:func:`radix_digit`) from ``equal[d]``, the
    group's count of the prefix's keys with digit ``d``, and ``above``, its
    count of keys above the prefix (the previous round's ``upper``)."""
    return np.cumsum(np.asarray(equal)[::-1])[::-1] + above


def tie_offsets(ties, rank: int):
    """``[pages]``: from every rank's per-local-page counts of threshold ties
    (``ties [RANKS, pages]``), the ties ahead of rank ``rank``'s page ``j`` in
    global token order other than its own earlier pages: every other rank's on
    pages before ``j`` and the lower ranks' on page ``j``. A tie at local index
    ``t`` is taken when this plus the rank's own ties before ``t`` is under the
    quota."""
    ties = np.asarray(ties)
    others = ties.sum(axis=0) - ties[rank]
    return np.cumsum(others) - others + ties[:rank].sum(axis=0)


def radix_select_model(
    keys,
    counts,
    k: int,
    bits: int,
    page_tokens: int,
    histogram: str = "cumulative",
    tie_skip: bool = False,
):
    """The select kernel's arithmetic for one row: ``keys [RANKS, T]`` uint32,
    ``counts [RANKS]`` each rank's valid local tokens; the selected mask
    ``[RANKS, T]``. ``tie_skip``: ``(mask, need)``, ``need`` whether the row
    needed the tie round (the count at or above the threshold key, the last
    round's chosen count, exceeds ``k``); without it every key at or above the
    threshold is selected."""
    keys = np.asarray(keys).astype(np.int64)
    valid = np.arange(keys.shape[1])[None] < np.asarray(counts)[:, None]
    if valid.sum() <= k:
        return (valid, False) if tie_skip else valid
    base, upper, at = 0, 0, 0
    for step in range(32 // bits):
        shift = 32 - bits * (step + 1)
        if histogram == "packed":
            digit = (keys - base) >> shift
            inside = valid & (keys >= base) & (digit < 2**bits)
            equal = np.bincount(digit[inside], minlength=2**bits)
            counts_ge = cumulative_counts(equal, upper)
        else:
            thresholds = base + (np.arange(2**bits, dtype=np.int64) << shift)
            counts_ge = [int((valid & (keys >= t)).sum()) for t in thresholds]
        digit, upper = radix_digit(counts_ge, k, upper)
        at = int(counts_ge[digit])
        base += digit << shift
    if tie_skip and at == k:
        return valid & (keys >= base), False
    tie = valid & (keys == base)
    page_ties = tie.reshape(RANKS, -1, page_tokens).sum(axis=2)
    ahead = np.cumsum(tie, axis=1) - tie
    selected = valid & (keys > base)
    for rank in range(RANKS):
        offsets = np.repeat(tie_offsets(page_ties, rank), page_tokens)
        selected[rank] |= tie[rank] & (offsets + ahead[rank] < k - upper)
    return (selected, True) if tie_skip else selected


def tie_round_needed(values, lengths, k: int = SELECT_K):
    """Per row of global ``values [R, RANKS T]`` (global token order): whether
    the k-th largest valid value ties the (k+1)-th, i.e. count(>= tau) > k,
    the only case the tie round must run."""
    out = []
    for row, length in zip(values, lengths):
        if length <= k:
            out.append(False)
            continue
        ordered = np.sort(row[:length])[::-1]
        out.append(bool(ordered[k - 1] == ordered[k]))
    return np.array(out)


# (ties filling the quota, extra ties) by row % 4
EXACT_TIES = ((7, 0), (7, 1), (100, 0), (100, 1))


def exact_tie_scores(rows: int, tokens: int, page_tokens: int):
    """Global scores ``[R, RANKS T]`` whose threshold ties exactly fill the
    quota (rows 0, 2 mod 4: the round is skipped) or overflow it by one (rows
    1, 3: it runs), at random global positions so they span ranks and pages;
    every other value is distinct."""
    size = RANKS * tokens
    if size <= SELECT_K:
        raise ValueError(f"select: {size} tokens cannot hold the exact tie rows")
    rng = np.random.default_rng([SELECT_SEED, rows, tokens, page_tokens, 1])
    out = np.empty((rows, size), np.float32)
    for r in range(rows):
        ties, extra = EXACT_TIES[r % 4]
        order = rng.permutation(size)
        above = SELECT_K - ties
        out[r, order[:above]] = 2 + rng.permutation(above) / size
        out[r, order[above : above + ties + extra]] = 1.0
        rest = order[above + ties + extra :]
        out[r, rest] = -rng.permutation(len(rest)) / size
    return out


def merge_candidates_model(
    values, lengths, rank: int, k: int = SELECT_K, page_tokens: int = 256
):
    """Rank ``rank``'s ``--exchange merge`` candidates from its local scores
    ``values [R, T]`` in rows of global ``lengths``: ``(keys uint32 [R, min(T,
    k)], ids int32 [R, min(T, k)])``, the :func:`order_keys` key and the
    :func:`global_token_index` of every valid local token when the rank holds
    at most ``k`` of them, else of its local top-``k`` (ties to the lower local,
    so global, index), in local token order; key 0 and id -1 pad a row."""
    values = np.asarray(values, np.float32)
    rows, tokens = values.shape
    width = min(tokens, k)
    keys = np.zeros((rows, width), np.uint32)
    ids = np.full((rows, width), -1, np.int32)
    index = global_token_index(np.arange(tokens), rank, page_tokens)
    for r, length in enumerate(lengths):
        key = order_keys(values[r])
        order = np.lexsort((np.arange(tokens), -key.astype(np.int64)))
        top = np.sort(order[index[order] < length][:k])
        keys[r, : len(top)], ids[r, : len(top)] = key[top], index[top]
    return keys, ids


def merge_select_model(gathered_keys, gathered_ids, lengths, k=SELECT_K, size=None):
    """The global top-``k`` of every rank's merge candidates (``gathered_keys``,
    ``gathered_ids [RANKS, R, C]``, :func:`merge_candidates_model`) as an int8
    mask ``[R, size]`` over global token indices (``size`` defaults to ``RANKS
    C``, the context when every token is a candidate): the keys above the
    ``k``-th largest ``tau`` and, of those equal to it, the lowest ids up to the
    quota (every candidate in a row of at most ``k`` tokens). Exact: a token of
    the global top-``k`` has fewer than ``k`` tokens above it, so it is in its
    owner's local top-``k``."""
    keys = np.asarray(gathered_keys).astype(np.int64)
    ids = np.asarray(gathered_ids)
    ranks, rows, width = keys.shape
    mask = np.zeros((rows, size or ranks * width), np.int8)
    for r in range(rows):
        key, index = keys[:, r].reshape(-1), ids[:, r].reshape(-1)
        real = index >= 0
        order = np.lexsort((index[real], -key[real]))[: min(k, lengths[r])]
        mask[r, index[real][order]] = 1
    return mask


def row_packet(counts, rows: int, bins: int):
    """``--row-sum before``'s packet: the rows' summed counts ``[rows, bins]``
    at flat positions ``r bins + d`` of one ``[8, 128]`` tile (``rows bins <=
    1024``)."""
    if rows * bins > SUBLANES * LANES:
        raise ValueError(f"select: {rows} rows of {bins} counts exceed one packet")
    flat = np.zeros(SUBLANES * LANES, np.int64)
    flat[: rows * bins] = np.asarray(counts).reshape(-1)
    return flat.reshape(SUBLANES, LANES)


def row_unpacket(packet, rows: int, bins: int):
    return np.asarray(packet).reshape(-1)[: rows * bins].reshape(rows, bins)


def padded_rows(rows: int) -> int:
    """``rows`` rounded up to whole tiles of eight sublanes."""
    return -(-rows // SUBLANES) * SUBLANES


def tile_group_rows(rows: int, case: str) -> int:
    """The rows that share one tile of a case's row-padded buffers: eight, or
    all of them below eight (``rows`` must divide 8 or be a multiple of 8)."""
    if rows >= SUBLANES and rows % SUBLANES == 0:
        return SUBLANES
    if 0 < rows < SUBLANES and SUBLANES % rows == 0:
        return rows
    raise ValueError(f"{case}: {rows} rows must divide 8 or be a multiple of 8")


def virtual_groups(rows: int, tokens: int) -> int:
    """The parts each row splits into in the select kernel's layout: the kernel
    holds ``padded_rows(rows)`` virtual rows of whole 128-token lane rows, so
    below eight rows a row splits into ``8 / rows`` contiguous parts."""
    groups = SUBLANES // tile_group_rows(rows, "select")
    if tokens % (groups * LANES):
        raise ValueError(
            f"select: {tokens} local tokens do not split into {groups} parts of "
            f"whole {LANES}-token lane rows"
        )
    return groups


def to_virtual(values, groups: int):
    """``[R, T]`` -> ``[planes, R * groups, 128]``: virtual row ``r * groups +
    g`` holds row r's tokens ``[g T / groups, (g + 1) T / groups)``, 128 a
    plane."""
    rows, tokens = values.shape
    planes = tokens // (groups * LANES)
    return (
        values.reshape(rows, groups, planes, LANES)
        .transpose(2, 0, 1, 3)
        .reshape(planes, rows * groups, LANES)
    )


def from_virtual(values, rows: int):
    planes, vrows, _ = values.shape
    groups = vrows // rows
    return (
        values.reshape(planes, rows, groups, LANES)
        .transpose(1, 2, 0, 3)
        .reshape(rows, groups * planes * LANES)
    )


def unpack_mask(words, rows: int, tokens: int):
    """The select kernel's output (int32 ``[words, V, 128]``; byte ``i`` of word
    ``w`` is plane ``4 w + i``'s 0/1 mask) -> the bool mask ``[R, T]``."""
    planes = tokens // (virtual_groups(rows, tokens) * LANES)
    words = np.ascontiguousarray(words, np.int32)
    by_plane = words.view(np.uint8).reshape(*words.shape, 4).transpose(0, 3, 1, 2)
    return from_virtual(by_plane.reshape(-1, *words.shape[1:])[:planes], rows) != 0


def select_pages(tokens: int, page_tokens: int) -> int:
    """The pages in ``tokens`` local tokens: at most ``LANES``, a power of two,
    of ``page_tokens`` tokens, a power-of-two number of ``LANES``-token lane
    rows (the kernel's page shift and dense packets rely on both)."""
    pages = tokens // page_tokens
    lane_rows = page_tokens // LANES
    if (
        tokens % page_tokens
        or page_tokens % LANES
        or pages & (pages - 1)
        or lane_rows & (lane_rows - 1)
    ):
        raise ValueError(
            f"select: {tokens} local tokens must be a power of two number of whole "
            f"{page_tokens}-token pages, which must be a power of two multiple of "
            f"{LANES} tokens"
        )
    if pages > LANES:
        raise ValueError(
            f"select: {tokens} local tokens are {pages} pages of {page_tokens}, "
            f"at most {LANES}"
        )
    return pages


def _token_ids(s, at, me, groups: int, part: int, page_tokens: int):
    """The global token index (:func:`global_token_index`) of lane ``at`` of
    plane ``s`` on each virtual row of the select's ``[8, 128]`` layout, on
    rank ``me``."""
    import jax
    import jax.numpy as jnp

    sub = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
    local = (sub % groups) * part + s * LANES + at
    page_bits = page_tokens.bit_length() - 1
    page = jax.lax.shift_right_logical(local, page_bits)
    return ((page * RANKS + me) << page_bits) | (local & (page_tokens - 1))


def _lane_compact(selected, values, strict):
    """Each sublane row's ``selected`` lanes of ``values`` moved to its first
    lanes in lane order, by shifts of 1, 2, 4, ... 64 lanes (low bit first, so
    no two lanes meet), and each moved lane's distance from its source
    (``strict``: ``dsa_select._strict_lanes``)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    lane = jax.lax.broadcasted_iota(jnp.int32, values.shape, 1)
    before = jnp.dot(
        selected.astype(jnp.int32).astype(jnp.bfloat16),
        strict,
        preferred_element_type=jnp.float32,
    ).astype(jnp.int32)
    shift = jnp.where(selected, lane - before, 0)
    live = selected
    step = 1
    while step < LANES:
        move = live & ((shift & step) != 0)
        arrive = pltpu.roll(move.astype(jnp.int32), LANES - step, 1) != 0
        values = jnp.where(arrive, pltpu.roll(values, LANES - step, 1), values)
        shift = jnp.where(arrive, pltpu.roll(shift, LANES - step, 1), shift)
        live = arrive | (live & jnp.logical_not(move))
        step *= 2
    return values, shift


def _lane_rotate(x, amount):
    """Each sublane row of ``x`` rotated up by its own ``amount`` (0 to 127)
    lanes."""
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    step = 1
    while step < LANES:
        x = jnp.where((amount & step) != 0, pltpu.roll(x, step, 1), x)
        step *= 2
    return x


def _merge_compact(
    keys,
    mask,
    slot,
    table,
    flush,
    lane_rows,
    sem,
    token_id,
    *,
    groups,
    planes,
    cand_planes,
):
    """Each row's selected local tokens (``mask``: the select's mask words,
    ``keys``: its int32 keys, both over ``planes`` planes of 8 virtual rows) as
    ``--exchange merge`` candidates in ``slot [2 C, 8, 128]``: keys in planes
    ``[0, C)``, global ids (``token_id(plane, lanes)``) in ``[C, 2 C)``, row
    ``r``'s lane-row ``L`` (128 candidates) at plane ``L // groups``, sublane
    ``r groups + L % groups``; INT32_MIN keys and -1 ids pad a row's last.

    Each virtual row compacts a plane's selected lanes (:func:`_lane_compact`)
    and appends them to its 128-lane accumulator (:func:`_lane_rotate`); a full
    accumulator goes to its row's next lane-row. A first pass marks, per plane,
    the virtual rows whose accumulator fills there (``table`` [8, 128] in VMEM,
    moved to SMEM ``flush``, which the second pass reads as scalars);
    ``lane_rows`` (SMEM, 8) counts each row's lane-rows. The partly filled
    accumulators of a row's virtual rows are packed end to end last."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels.dsa_select import (
        _group_exclusive,
        _group_sum,
        _strict_lanes,
    )

    shape = (SUBLANES, LANES)
    sub = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    words = planes // 4
    zero = jnp.zeros(shape, jnp.int32)

    def chosen(word, i):
        return (jax.lax.shift_right_logical(word, 8 * i) & 255) != 0

    def count(selected):
        return jnp.broadcast_to(
            jnp.sum(selected.astype(jnp.int32), axis=1, keepdims=True), shape
        )

    slot[pl.ds(0, cand_planes)] = jnp.full((cand_planes, *shape), INT32_MIN, jnp.int32)
    slot[pl.ds(cand_planes, cand_planes)] = jnp.full(
        (cand_planes, *shape), -1, jnp.int32
    )

    def flags(w, carry):
        fill, bits = carry
        word = mask[w]
        for i in range(4):
            s = 4 * w + i
            n = count(chosen(word, i))
            full = jnp.sum(
                jnp.where(fill + n >= LANES, jnp.left_shift(1, sub), 0),
                axis=0,
                keepdims=True,
            )
            here = (sub == jax.lax.shift_right_logical(s, 7)) & (lane == (s & 127))
            bits = jnp.where(here, full, bits)
            fill = (fill + n) & (LANES - 1)
        return fill, bits

    _, bits = jax.lax.fori_loop(0, words, flags, (zero, zero))
    table[...] = bits
    move = pltpu.make_async_copy(table, flush, sem)
    move.start()
    for r in range(SUBLANES):
        lane_rows[r] = 0
    move.wait()

    def put(row, source, key_row, id_row):
        """Virtual row ``source``'s ``key_row``/``id_row`` as row ``row``'s next
        lane-row."""
        at = lane_rows[row]
        plane = jax.lax.shift_right_logical(at, groups.bit_length() - 1)
        target = row * groups + (at & (groups - 1))
        if groups > 1:
            key_row = jnp.broadcast_to(key_row[source : source + 1], shape)
            id_row = jnp.broadcast_to(id_row[source : source + 1], shape)
        here = sub == target
        slot[plane] = jnp.where(here, key_row, slot[plane])
        slot[cand_planes + plane] = jnp.where(here, id_row, slot[cand_planes + plane])
        lane_rows[row] = at + 1

    strict = _strict_lanes()

    def append(w, carry):
        acc_key, acc_id, fill = carry
        word = mask[w]
        for i in range(4):
            s = 4 * w + i
            selected = chosen(word, i)
            n = count(selected)
            moved, shift = _lane_compact(selected, keys[s], strict)
            moved_id = token_id(s, lane + shift)
            moved, moved_id = _lane_rotate(moved, fill), _lane_rotate(moved_id, fill)
            end = fill + n
            low = (lane >= fill) & (lane < end)
            acc_key = jnp.where(low, moved, acc_key)
            acc_id = jnp.where(low, moved_id, acc_id)
            full = flush[jax.lax.shift_right_logical(s, 7), s & 127]

            @pl.when(full != 0)
            def _(acc_key=acc_key, acc_id=acc_id, full=full):
                for v in range(SUBLANES):
                    pl.when((jax.lax.shift_right_logical(full, v) & 1) != 0)(
                        lambda v=v: put(v // groups, v, acc_key, acc_id)
                    )

            over = end >= LANES
            acc_key = jnp.where(over, moved, acc_key)
            acc_id = jnp.where(over, moved_id, acc_id)
            fill = end & (LANES - 1)
        return acc_key, acc_id, fill

    acc_key, acc_id, fill = jax.lax.fori_loop(0, words, append, (zero, zero, zero))
    # The partly filled rows of a row's virtual rows, end to end.
    ahead = _group_exclusive(fill, groups)
    total = _group_sum(fill, groups)
    for t in range(groups):
        amount = (ahead - LANES * t) & (LANES - 1)
        at = LANES * t + lane
        mine = (at >= ahead) & (at < ahead + fill)
        tail = []
        for acc, pad in ((acc_key, INT32_MIN), (acc_id, -1)):
            part = _group_sum(jnp.where(mine, _lane_rotate(acc, amount), 0), groups)
            tail.append(jnp.where(at < total, part, pad))
        for row in range(SUBLANES // groups):
            pl.when(lane_rows[row] < cand_planes * groups)(
                lambda row=row, tail=tail: put(row, row * groups, *tail)
            )


def _select_kernel(
    rows,
    tokens,
    bits,
    exchange,
    ties,
    page_tokens,
    histogram,
    tie_skip=False,
    row_sum_order="after",
):
    """The select ``pallas_call`` kernel body and its scratch shapes.
    ``tie_skip``: the tie round runs only when a row's ties overflow its quota,
    and a second output ``stats [V, 128]`` holds per virtual row lane 0
    ``need``, lane 1 whether the round ran, lane 2 the count at or above the
    threshold key. ``row_sum_order="before"`` (below eight rows): each row's
    counts are summed over its virtual rows before the exchange, in one ``[8,
    128]`` packet (:func:`row_packet`). ``exchange="merge"`` (below nine rows):
    the local top-k candidates of every rank in one all-gather, then the select
    over them on every rank (the module docstring)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.common import collectives32
    from vllm_torchtpu.megakernel.kimi_k3.kernels.dsa_select import (
        _exchange_sum,
        _group_exclusive,
        _group_sum,
        _inbox_exchange,
        _key,
        _lane_exclusive,
        _lane_suffix,
        _packet_shape,
        _roll,
        _row_pieces,
        _select_stages,
        _strict_lanes,
    )

    merge = exchange == "merge"
    if merge and (rows > SUBLANES or tie_skip):
        raise ValueError("select: the merge runs below nine rows, without the tie skip")
    groups = virtual_groups(rows, tokens)
    vrows = rows * groups
    part = tokens // groups
    planes = part // LANES
    words = -(-planes // 4)
    bins = 2**bits
    bin_lanes = max(bins, LANES)
    # Digit values per pass over the keys (packed: four per counter).
    chunk = min(bins, BIN_CHUNK * (4 if histogram == "packed" else 1))
    pages = tokens // page_tokens
    page_shift = (page_tokens // LANES).bit_length() - 1
    if row_sum_order not in SELECT_ROW_SUMS:
        raise ValueError(
            f"select: row sum {row_sum_order!r} is not one of {SELECT_ROW_SUMS}"
        )
    if tie_skip and not ties:
        raise ValueError("select: the tie skip needs the tie round (ties on)")
    # At eight rows and more a row is one virtual row: there is nothing to sum.
    sum_first = row_sum_order == "before" and groups > 1
    pieces = _row_pieces(rows, bins) if sum_first else []
    hist_packet = (SUBLANES, LANES) if sum_first else _packet_shape(vrows, bins)
    tie_packet = _packet_shape(vrows, pages)
    stages = _select_stages(exchange, RANKS)
    # The merge's local select runs its tie round only when ties overflow too.
    gate = tie_skip or merge
    # Merge: a row's candidates, all its local tokens or its local top-k.
    cand_planes = min(tokens, SELECT_K) // (groups * LANES)
    compact = merge and tokens > SELECT_K

    wires = []
    for slots, _ in stages:
        wires.append(pltpu.VMEM((2, slots, *hist_packet), jnp.int32))
        wires.append(pltpu.SemaphoreType.DMA((2, 2)))
    if ties:
        for slots, _ in stages:
            wires.append(pltpu.VMEM((1, slots, *tie_packet), jnp.int32))
            wires.append(pltpu.SemaphoreType.DMA((2, 1)))
    scratch = [
        pltpu.VMEM((planes, vrows, LANES), jnp.float32),  # scores
        pltpu.VMEM((4 * words, vrows, LANES), jnp.int32),  # keys
        pltpu.VMEM((words, vrows, LANES), jnp.int32),  # mask words
        pltpu.VMEM((vrows, bin_lanes), jnp.int32),  # local counts
        pltpu.VMEM((8 if merge else 6 if gate else 5, vrows, LANES), jnp.int32),
        pltpu.SemaphoreType.DMA((2,)),
        *([pltpu.VMEM((vrows, LANES), jnp.int32)] if tie_skip else []),  # stats
        *wires,
    ]
    if merge:
        scratch += [
            pltpu.VMEM((1, RANKS, 2 * cand_planes, vrows, LANES), jnp.int32),
            pltpu.SemaphoreType.DMA((2, 1)),
        ]
    if compact:
        scratch += [
            pltpu.VMEM((SUBLANES, LANES), jnp.int32),  # flush bits per plane
            pltpu.SMEM((SUBLANES, LANES), jnp.int32),
            pltpu.SMEM((SUBLANES,), jnp.int32),  # lane-rows per row
        ]

    def row_sum(x):
        return jnp.sum(x, axis=1, keepdims=True)

    def pack(x, width, shape):
        if width >= LANES:
            return x
        block = shape[0]
        packet = x[:block]
        for f in range(1, vrows // block):
            packet = packet + pltpu.roll(x[f * block : (f + 1) * block], f * width, 1)
        return packet

    def unpack(packet, width, shape):
        if width >= LANES:
            return packet
        block = shape[0]
        keep = jax.lax.broadcasted_iota(jnp.int32, packet.shape, 1) < width
        parts = [
            jnp.where(keep, pltpu.roll(packet, (LANES - f * width) % LANES, 1), 0)
            for f in range(vrows // block)
        ]
        return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)

    def group_exclusive(x):
        """Sum over a row's earlier virtual rows."""
        return _group_exclusive(x, groups)

    def to_row_packet(local):
        """Each row's counts, summed on all its virtual rows of ``local [8,
        bin_lanes]``, from its first one into its place in the packet."""
        sub = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 1)
        packet = jnp.zeros((SUBLANES, LANES), jnp.int32)
        for r, c, s, offset, width in pieces:
            piece = local[:, c * LANES : (c + 1) * LANES]
            if width < LANES:
                piece = jnp.where(lane < width, piece, 0)
            piece = _roll(_roll(piece, s - r * groups, 0), offset, 1)
            packet = packet + jnp.where(sub == s, piece, 0)
        return packet

    def from_row_packet(packet):
        """The summed packet's counts of each row, on all its virtual rows."""
        sub = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 1)
        chunks = [jnp.zeros((SUBLANES, LANES), jnp.int32)] * (bin_lanes // LANES)
        for r, c, s, offset, width in pieces:
            piece = _roll(_roll(packet, r * groups - s, 0), -offset, 1)
            if width < LANES:
                piece = jnp.where(lane < width, piece, 0)
            chunks[c] = jnp.where(sub == r * groups, piece, chunks[c])
        # Each row's counts sit on its first virtual row: the group sum copies them.
        chunks = [_group_sum(x, groups) for x in chunks]
        return chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=1)

    def kernel(meta, scores_hbm, *refs):
        # The aliased inputs, the outputs (the same buffers), then the scratch.
        outputs = 2 if tie_skip else 1
        mask_hbm, stats_hbm = refs[outputs], (refs[outputs + 1] if tie_skip else None)
        scores, keys, mask, hist, state, sem, *wire_refs = refs[2 * outputs :]
        stats = wire_refs.pop(0) if tie_skip else None
        merge_refs = wire_refs[2 * len(stages) * (2 if ties else 1) :]
        me = jax.lax.axis_index(AXIS)
        lane = jax.lax.broadcasted_iota(jnp.int32, (vrows, LANES), 1)
        sub = jax.lax.broadcasted_iota(jnp.int32, (vrows, LANES), 0)
        part_of = sub % groups
        load = pltpu.make_async_copy(scores_hbm.at[0], scores, sem.at[0])
        load.start()
        if exchange != "none":
            # Peers write into this kernel's inboxes: all must have entered it.
            collectives32.barrier()

        count_ref, early_ref, base_ref, upper_ref, offset_ref = (
            state.at[i] for i in range(5)
        )
        at_ref = state.at[5] if gate else None
        with jax.named_scope("rows"):
            state[...] = jnp.zeros(state.shape, jnp.int32)
            hist[...] = jnp.zeros(hist.shape, jnp.int32)
            sub8 = jax.lax.broadcasted_iota(jnp.int32, (8, LANES), 0)
            active = False
            for v in range(vrows):
                r, g = divmod(v, groups)
                block = pl.ds(8 * (v // 8), 8)
                count = jnp.clip(meta[0, 0, r] - g * part, 0, part)
                count_ref[block, :] = jnp.where(
                    sub8 == v % 8, count, count_ref[block, :]
                )
                # The merge's local select: rows of at most k local tokens.
                field = 0 if merge else 1
                early = (meta[0, field, r] <= SELECT_K).astype(jnp.int32)
                early_ref[block, :] = jnp.where(
                    sub8 == v % 8, early, early_ref[block, :]
                )
                if g == 0:
                    active = jnp.logical_or(active, meta[0, field, r] > SELECT_K)
            if 4 * words > planes:
                keys[pl.ds(planes, 4 * words - planes)] = jnp.full(
                    (4 * words - planes, vrows, LANES), INT32_MIN, jnp.int32
                )
        load.wait()

        def to_keys(s, _):
            key = _key(pltpu.bitcast(scores[s], jnp.int32))
            valid = s * LANES + lane < count_ref[...]
            keys[s] = jnp.where(valid, key, INT32_MIN)

        with jax.named_scope("keys"):
            jax.lax.fori_loop(0, planes, to_keys, None)

        def reduce(stage_wires, packet, parity, lower_too):
            """``(sum over ranks, sum over lower ranks)`` of ``packet``."""
            return _exchange_sum(
                me,
                stages,
                stage_wires,
                packet,
                parity,
                lower_too,
                lambda slots, stride: (me // stride) % slots,
            )

        hist_wires = [
            (wire_refs[2 * i], wire_refs[2 * i + 1]) for i in range(len(stages))
        ]
        tie_wires = [
            (wire_refs[2 * (len(stages) + i)], wire_refs[2 * (len(stages) + i) + 1])
            for i in range(len(stages) if ties else 0)
        ]

        def read_keys(s, block):
            return keys[s, block, :]

        def count_pass(shift, read=read_keys, planes=planes, base_ref=base_ref):
            bin_lane = jax.lax.broadcasted_iota(jnp.int32, (8, bin_lanes), 1)
            unroll = 2 if planes % 2 == 0 else 1

            def per_chunk(c, _):
                def per_block(b, _):
                    block = pl.ds(pl.multiple_of(8 * b, 8), 8)
                    base = base_ref[block, :]
                    thresholds = [
                        (base | ((c * chunk + d) << shift)) ^ INT32_MIN
                        for d in range(chunk)
                    ]

                    def step(s, acc):
                        x = read(s, block)
                        return tuple(
                            a + (x >= t).astype(jnp.int32)
                            for a, t in zip(acc, thresholds)
                        )

                    zero = jnp.zeros((8, LANES), jnp.int32)
                    acc = jax.lax.fori_loop(
                        0, planes, step, (zero,) * chunk, unroll=unroll
                    )
                    h = hist[block, :]
                    for d in range(chunk):
                        h = jnp.where(bin_lane == c * chunk + d, row_sum(acc[d]), h)
                    hist[block, :] = h

                jax.lax.fori_loop(0, vrows // 8, per_block, None)

            jax.lax.fori_loop(0, bins // chunk, per_chunk, None)

        def packed_pass(shift, read=read_keys, planes=planes, base_ref=base_ref):
            """Counts of each digit value (not cumulative) among the keys of
            the round's prefix, as four 8-bit counters per int32 lane, one
            compare-and-add per key and four digit values."""
            bin_lane = jax.lax.broadcasted_iota(jnp.int32, (8, bin_lanes), 1)
            spans = -(-planes // FIELD_PLANES)
            span = -(-planes // spans)

            def per_chunk(c, _):
                def per_block(b, _):
                    block = pl.ds(pl.multiple_of(8 * b, 8), 8)
                    base = base_ref[block, :] ^ INT32_MIN
                    first = c * chunk // 4
                    targets = [first + g for g in range(chunk // 4)]
                    mine = (bin_lane >= c * chunk) & (bin_lane < (c + 1) * chunk)

                    def per_span(p, h):
                        def step(s, acc):
                            # Keys outside the prefix give digits >= 2^b (or < 0).
                            digit = jax.lax.shift_right_logical(
                                read(s, block) - base, shift
                            )
                            group = jax.lax.shift_right_logical(digit, 2)
                            one = jnp.left_shift(1, (digit & 3) << 3)
                            return tuple(
                                a + jnp.where(group == t, one, 0)
                                for a, t in zip(acc, targets)
                            )

                        zero = jnp.zeros((8, LANES), jnp.int32)
                        acc = jax.lax.fori_loop(
                            p * span,
                            jnp.minimum((p + 1) * span, planes),
                            step,
                            (zero,) * (chunk // 4),
                        )
                        for g in range(chunk // 4):
                            for f in range(4):
                                field = jax.lax.shift_right_logical(acc[g], 8 * f) & 255
                                h = jnp.where(
                                    bin_lane == c * chunk + 4 * g + f,
                                    h + row_sum(field),
                                    h,
                                )
                        return h

                    h = jnp.where(mine, 0, hist[block, :])
                    hist[block, :] = jax.lax.fori_loop(0, spans, per_span, h)

                jax.lax.fori_loop(0, vrows // 8, per_block, None)

            jax.lax.fori_loop(0, bins // chunk, per_chunk, None)

        def rounds(read, planes, base_ref, upper_ref, at_ref, target=SELECT_K):
            """A radix select's rounds over ``planes`` planes of ``read``'s
            keys, for the ``target``-th largest (per virtual row)."""
            jax.lax.fori_loop(
                0,
                32 // bits,
                functools.partial(
                    one_round,
                    read=read,
                    planes=planes,
                    base_ref=base_ref,
                    upper_ref=upper_ref,
                    at_ref=at_ref,
                    target=target,
                ),
                None,
            )

        def one_round(
            j,
            _,
            read=read_keys,
            planes=planes,
            base_ref=base_ref,
            upper_ref=upper_ref,
            at_ref=at_ref,
            target=SELECT_K,
        ):
            shift = 32 - bits * (j + 1)
            with jax.named_scope("count"):
                if histogram == "packed":
                    packed_pass(shift, read, planes, base_ref)
                else:
                    count_pass(shift, read, planes, base_ref)
            with jax.named_scope("exchange"):
                if sum_first:
                    packet = to_row_packet(_group_sum(hist[...], groups))
                else:
                    packet = pack(hist[...], bins, hist_packet)
                total, _ = reduce(hist_wires, packet, j % 2, False)
            with jax.named_scope("decide"):
                if sum_first:
                    counts = from_row_packet(total)
                else:
                    counts = unpack(total, bins, hist_packet)
                    if groups > 1:
                        counts = _group_sum(counts, groups)
                in_bins = jax.lax.broadcasted_iota(jnp.int32, counts.shape, 1) < bins
                if histogram == "packed":
                    # Cumulative from the top, plus the keys above the prefix.
                    above = upper_ref[...][:, :1]
                    counts = jnp.where(in_bins, _lane_suffix(counts) + above, 0)
                reach = row_sum(jnp.where(in_bins & (counts >= target), 1, 0))
                below = jnp.max(
                    jnp.where(in_bins & (counts < target), counts, -1),
                    axis=1,
                    keepdims=True,
                )
                below = jnp.broadcast_to(below, (vrows, LANES))
                upper_ref[...] = jnp.where(below >= 0, below, upper_ref[...])
                base_ref[...] = base_ref[...] | ((reach - 1) << shift)
                if at_ref is not None:
                    # The chosen digit's count: after the last round, count(>= tau).
                    at = jnp.min(
                        jnp.where(in_bins & (counts >= target), counts, INT32_MAX),
                        axis=1,
                        keepdims=True,
                    )
                    at_ref[...] = jnp.broadcast_to(at, (vrows, LANES))

        def page_of(s):
            return jax.lax.shift_right_logical(part_of * planes + s, page_shift)

        def need():
            """1 on the rows whose threshold ties overflow the quota, the only
            ones the tie round must run for: the same on every rank, as it comes
            from summed counts."""
            return jnp.where((at_ref[...] != SELECT_K) & (early_ref[...] == 0), 1, 0)

        def tie_round():
            with jax.named_scope("ties"):
                tau = base_ref[...] ^ INT32_MIN

                def tie_pages(s, counts):
                    tie = (keys[s] == tau).astype(jnp.int32)
                    return counts + jnp.where(lane == page_of(s), row_sum(tie), 0)

                own = jax.lax.fori_loop(
                    0, planes, tie_pages, jnp.zeros((vrows, LANES), jnp.int32)
                )
                total, lower = reduce(tie_wires, pack(own, pages, tie_packet), 0, True)
                total, lower = (unpack(x, pages, tie_packet) for x in (total, lower))
                if groups > 1:
                    total, lower, own = (
                        _group_sum(total, groups),
                        _group_sum(lower, groups),
                        _group_sum(own, groups),
                    )
                offset_ref[...] = _lane_exclusive(total - own) + lower

        @pl.when(active)
        def _():
            if merge and not compact:
                return  # every local token is a candidate
            jax.lax.fori_loop(0, 32 // bits, one_round, None)
            if not ties:
                return
            if gate:
                pl.when(jnp.max(need()) > 0)(tie_round)
            else:
                tie_round()

        def select_all(with_ties):
            tau = base_ref[...] ^ INT32_MIN
            quota = SELECT_K - upper_ref[...]
            early = early_ref[...] != 0
            before = jnp.zeros((vrows, LANES), jnp.int32)
            if with_ties and groups > 1:
                row_ties = jnp.zeros((vrows, LANES), jnp.int32)
                for s in range(planes):
                    row_ties = row_ties + row_sum((keys[s] == tau).astype(jnp.int32))
                before = group_exclusive(row_ties)
            strict = _strict_lanes()

            def select_word(w, before):
                word = jnp.zeros((vrows, LANES), jnp.int32)
                for i in range(4):
                    s = 4 * w + i
                    x = keys[s]
                    if with_ties:
                        tie = x == tau
                        tie_count = tie.astype(jnp.int32)
                        inner = jnp.dot(
                            tie_count.astype(jnp.bfloat16),
                            strict,
                            preferred_element_type=jnp.float32,
                        ).astype(jnp.int32)
                        ahead = row_sum(
                            jnp.where(lane == page_of(s), offset_ref[...], 0)
                        )
                        chosen = (x > tau) | (tie & (ahead + before + inner < quota))
                        before = before + row_sum(tie_count)
                    else:
                        chosen = x >= tau
                    valid = (s * LANES + lane < count_ref[...]).astype(jnp.int32)
                    chosen = jnp.where(early, valid, chosen.astype(jnp.int32))
                    word = word | (chosen << (8 * i))
                mask[w] = word
                return before

            jax.lax.fori_loop(0, words, select_word, before)

        def token_id(s, at):
            return _token_ids(s, at, me, groups, part, page_tokens)

        def merge_select():
            """This rank's candidates into its inbox slot, one all-gather, the
            select over every rank's candidates (and on rows whose ties overflow
            the quota, the quota's lowest global ids among the tied ones), then
            this rank's mask."""
            inbox, inbox_sems, *compact_refs = merge_refs
            slot = inbox.at[0, me]
            with jax.named_scope("candidates"):
                if compact:
                    _merge_compact(
                        keys,
                        mask,
                        slot,
                        *compact_refs,
                        sem.at[0],
                        token_id,
                        groups=groups,
                        planes=planes,
                        cand_planes=cand_planes,
                    )
                else:

                    def own(s, _):
                        slot[s] = keys[s]
                        valid = s * LANES + lane < count_ref[...]
                        slot[cand_planes + s] = jnp.where(valid, token_id(s, lane), -1)

                    jax.lax.fori_loop(0, planes, own, None)
            if ties:  # else timing only: the select runs over stale peer slots
                with jax.named_scope("gather"):
                    _inbox_exchange(
                        me, inbox, inbox_sems, None, 0, RANKS, 1, (me // 1) % RANKS
                    )
            plane_bits = cand_planes.bit_length() - 1
            gathered = RANKS * cand_planes

            def candidate(s, half):
                rank = jax.lax.shift_right_logical(s, plane_bits)
                return inbox[0, rank, half * cand_planes + (s & (cand_planes - 1))]

            base2_ref, upper2_ref = state.at[6], state.at[7]
            active = False
            for v in range(vrows):
                length = meta[0, 1, v // groups]
                early_ref[...] = jnp.where(
                    sub == v, (length <= SELECT_K).astype(jnp.int32), early_ref[...]
                )
                active = jnp.logical_or(active, length > SELECT_K)
            for ref in (base_ref, upper_ref, at_ref):
                ref[...] = jnp.zeros((vrows, LANES), jnp.int32)
            with jax.named_scope("merge"):
                pl.when(active)(
                    lambda: rounds(
                        lambda s, _: candidate(s, 0),
                        gathered,
                        base_ref,
                        upper_ref,
                        at_ref,
                    )
                )
                tie_rows = need() != 0
                tau = base_ref[...] ^ INT32_MIN

                def lowest_ids():
                    # Higher keys for lower ids: the quota-th largest picks them.
                    quota = (SELECT_K - upper_ref[...])[:, :1]
                    base2_ref[...] = jnp.zeros((vrows, LANES), jnp.int32)
                    upper2_ref[...] = jnp.zeros((vrows, LANES), jnp.int32)
                    rounds(
                        lambda s, _: jnp.where(
                            candidate(s, 0) == tau, -candidate(s, 1), INT32_MIN
                        ),
                        gathered,
                        base2_ref,
                        upper2_ref,
                        None,
                        quota,
                    )

                if ties:
                    pl.when(jnp.max(tie_rows.astype(jnp.int32)) > 0)(lowest_ids)
            with jax.named_scope("mark"):
                lowest = base2_ref[...] ^ INT32_MIN
                early = early_ref[...] != 0

                def mark_word(w, _):
                    local = mask[w] if compact else None
                    word = jnp.zeros((vrows, LANES), jnp.int32)
                    for i in range(4):
                        s = 4 * w + i
                        x = keys[s]
                        valid = s * LANES + lane < count_ref[...]
                        picked = valid
                        if compact:
                            picked = (
                                jax.lax.shift_right_logical(local, 8 * i) & 255
                            ) != 0
                        taken = jnp.logical_not(tie_rows) | (
                            -token_id(s, lane) >= lowest
                        )
                        chosen = picked & ((x > tau) | ((x == tau) & taken))
                        chosen = jnp.where(
                            early, valid.astype(jnp.int32), chosen.astype(jnp.int32)
                        )
                        word = word | (chosen << (8 * i))
                    mask[w] = word

                jax.lax.fori_loop(0, words, mark_word, None)

        with jax.named_scope("select"):
            if merge and not compact:
                pass
            elif gate:
                # Without the round every tie is taken: exactly the quota.
                ran = jnp.max(need()) > 0
                pl.when(ran)(lambda: select_all(True))
                pl.when(jnp.logical_not(ran))(lambda: select_all(False))
            else:
                select_all(ties)
        if merge:
            merge_select()
        store = pltpu.make_async_copy(mask, mask_hbm.at[0], sem.at[1])
        store.start()
        if tie_skip:
            needed = need()
            any_row = jnp.broadcast_to(
                jnp.max(jnp.max(needed, axis=1, keepdims=True), axis=0, keepdims=True),
                (vrows, LANES),
            )
            stats[...] = jnp.where(
                lane == 0,
                needed,
                jnp.where(lane == 1, any_row, jnp.where(lane == 2, at_ref[...], 0)),
            )
            out = pltpu.make_async_copy(stats, stats_hbm.at[0], sem.at[0])
            out.start()
            out.wait()
        store.wait()

    return kernel, scratch, words, vrows, planes


SELECT_REFERENCES: dict[tuple, Any] = {}


def _select_lengths(rows: int, tokens: int, lengths: Sequence[int]):
    """Each row's context: ``lengths`` cycled over the rows (0: every slot of
    every rank), at most ``RANKS * tokens``."""
    full = RANKS * tokens
    picked = [lengths[r % len(lengths)] if lengths else 0 for r in range(rows)]
    return [min(length, full) if length > 0 else full for length in picked]


INDEXER_SCORES: dict[tuple[int, int, int], Any] = {}
EXACT_SCORES: dict[tuple[int, int, int], Any] = {}
INDEXER_SCORE_LIMIT = 8 * 32768  # R T kernel scores a point gathers to every rank


def _indexer_scores(rows: int, tokens: int, page_tokens: int):
    """``[RANKS, R, T]``: every rank's ``bf16q`` score kernel output (shared
    keys, :func:`_score_inputs`' seeds, index blocks of ``page_tokens``), from
    one call of the full write-back body on each rank, cached per point."""
    key = (rows, tokens, page_tokens)
    if key not in INDEXER_SCORES:
        import torch

        from vllm_torchtpu.megakernel import bridge

        if page_tokens not in SCORE_BLOCK_TOKENS:
            raise ValueError(
                f"select: indexer scores need index blocks of {SCORE_BLOCK_TOKENS} "
                f"tokens, not {page_tokens}"
            )
        if rows * tokens > INDEXER_SCORE_LIMIT:
            raise ValueError(
                f"select: indexer scores of {rows} x {tokens} tokens exceed "
                f"{INDEXER_SCORE_LIMIT}"
            )
        plan = ScorePlan(rows, tokens, page_tokens, 128, 1, 16, True)
        setup = _score_setup(rows, tokens, "bf16q", plan)
        outputs = call_body(
            f"select_indexer_r{rows}_t{tokens}_{setup.variant}",
            setup.body,
            setup.operands,
            setup.in_specs,
            setup.out_specs,
            1,
        )
        mine = _first(outputs)[0].cpu().numpy()[:rows]
        error = float(score_errors(mine, _score_inputs(rows, tokens).reference).max())
        # The global top-k needs every rank's kernel scores; regenerating them
        # costs 32 times the score math.
        gathered = [None] * torch.distributed.get_world_size()
        rank = bridge.kernel_mesh(AXIS, RANKS)[1]
        torch.distributed.all_gather_object(gathered, (rank, mine, error))
        worst = max(e for _, _, e in gathered)
        harness.log(
            f"select: indexer scores R={rows} T={tokens}: max |s - ref| / max |ref| "
            f"{worst:.3g} over {len(gathered)} ranks"
        )
        if not worst <= SCORE_TOLERANCE:
            raise RuntimeError(
                f"select: the indexer scores miss the reference by {worst:.3g}"
            )
        INDEXER_SCORES.clear()
        INDEXER_SCORES[key] = np.stack(
            [m for _, m, _ in sorted(gathered, key=lambda g: g[0])]
        )
    return INDEXER_SCORES[key]


def _select_scores(
    rows: int, tokens: int, rank: int, data: str, page_tokens: int = 256
):
    """Rank ``rank``'s local scores ``[R, T]``: random normal, or (``ties``)
    rounded to few values per row so that many tie, within and across ranks
    and at the threshold (every second row of four: all equal); ``exact``: its
    slice of :func:`exact_tie_scores`; ``indexer``: its score kernel output
    (:func:`_indexer_scores`, filled by the setup)."""
    if data == "exact":
        key = (rows, tokens, page_tokens)
        if key not in EXACT_SCORES:
            EXACT_SCORES.clear()
            EXACT_SCORES[key] = exact_tie_scores(rows, tokens, page_tokens)
        index = global_token_index(np.arange(tokens), rank, page_tokens)
        return EXACT_SCORES[key][:, index]
    if data == "indexer":
        return INDEXER_SCORES[rows, tokens, page_tokens][rank]
    values = np.random.default_rng([SELECT_SEED, rows, tokens, rank]).standard_normal(
        (rows, tokens)
    )
    if data == "ties":
        pattern = np.arange(rows)[:, None] % 4
        values = np.where(pattern == 1, 0.75, np.round(4 * values) / 4 + 0.125)
        values = np.where(pattern == 2, np.round(values) + 0.5, values)
    return values.astype(np.float32)


def _select_reference(rows, tokens, data, lengths, page_tokens):
    """``(global scores [R, RANKS T], wanted mask [R, RANKS T], lengths)``:
    every rank's scores at their global token index and
    ``dsa_reference.select_topk``'s selection, cached per input."""
    key = (rows, tokens, data, tuple(lengths), page_tokens)
    if key not in SELECT_REFERENCES:
        import jax
        import jax.numpy as jnp

        from tools.megakernel import dsa_reference as ref

        values = np.zeros((rows, RANKS * tokens), np.float32)
        local = np.arange(tokens)
        for rank in range(RANKS):
            index = global_token_index(local, rank, page_tokens)
            values[:, index] = _select_scores(rows, tokens, rank, data, page_tokens)
        with jax.default_device(jax.devices("cpu")[0]):
            chosen = np.asarray(
                ref.select_topk(jnp.asarray(values), jnp.asarray(lengths), SELECT_K)
            )
        wanted = np.zeros(values.shape, bool)
        for r in range(rows):
            wanted[r, chosen[r][chosen[r] >= 0]] = True
        SELECT_REFERENCES.clear()
        SELECT_REFERENCES[key] = (values, wanted, np.asarray(lengths))
    return SELECT_REFERENCES[key]


def _select_row_stats(selected, rank, rows, tokens, data, lengths, page_tokens):
    """This rank's part of every row's verdict: per row ``(equal to the
    reference here, selected, lowest selected score, highest unselected valid
    score, selected ties, highest selected tie index, lowest unselected tie
    index)``, ties being the scores equal to the reference's lowest selected
    one."""
    values, wanted, length = _select_reference(rows, tokens, data, lengths, page_tokens)
    index = global_token_index(np.arange(tokens), rank, page_tokens)
    mine, want = values[:, index], wanted[:, index]
    valid = index[None] < length[:, None]
    stats = []
    for r in range(rows):
        tau = values[r][wanted[r]].min() if wanted[r].any() else -np.inf
        tie = valid[r] & (mine[r] == tau)
        picked, skipped = selected[r] & tie, ~selected[r] & tie
        stats.append(
            (
                bool(np.array_equal(selected[r], want[r])),
                int(selected[r].sum()),
                float(mine[r][selected[r]].min(initial=np.inf)),
                float(mine[r][valid[r] & ~selected[r]].max(initial=-np.inf)),
                int(picked.sum()),
                int(index[picked].max(initial=-1)),
                int(index[skipped].min(initial=RANKS * tokens)),
            )
        )
    return stats


def select_verdicts(stats, values, wanted, lengths, k: int = SELECT_K):
    """Per row ``(passed, line)`` from every rank's :func:`_select_row_stats`:
    the group selects ``min(k, length)`` tokens, every selected score is at
    least every unselected one, the ties at the threshold fill the quota with
    the lowest global indices, and every rank's mask equals the reference's."""
    verdicts = []
    for r, length in enumerate(lengths):
        parts = [rank_stats[r] for rank_stats in stats]
        equal = all(s[0] for s in parts)
        selected = sum(s[1] for s in parts)
        ordered = min(s[2] for s in parts) >= max(s[3] for s in parts)
        valid = np.arange(values.shape[1]) < length
        tau = values[r][wanted[r]].min() if wanted[r].any() else -np.inf
        quota = int(wanted[r].sum() - (valid & (values[r] > tau)).sum())
        tied = int((valid & (values[r] == tau)).sum())
        taken = sum(s[4] for s in parts)
        lowest = max(s[5] for s in parts) < min(s[6] for s in parts)
        passed = (
            equal
            and selected == min(k, length)
            and ordered
            and taken == quota
            and lowest
        )
        verdicts.append(
            (
                passed,
                f"row {r}: {'PASS' if passed else 'FAIL'} (selected {selected} of "
                f"{length}; selected >= unselected {ordered}; {tied} tied at the "
                f"threshold, {taken} taken of quota {quota}, lowest indices "
                f"{lowest}; equal to select_topk on every rank {equal})",
            )
        )
    return verdicts


def select_tie_stats(stats, values, wanted, lengths, rows: int, k: int = SELECT_K):
    """``(passed, rows needing the round, whether it ran, line)`` from the tie
    skip kernel's ``stats [V, 128]`` (lane 0 ``need``, lane 1 whether the round
    ran, lane 2 the count at or above the threshold key; alike on a row's
    virtual rows): ``need`` is :func:`tie_round_needed`, the round ran when any
    row needed it, and on rows longer than ``k`` the count is the reference's
    valid values at or above its lowest selected one."""
    stats = np.asarray(stats)
    groups = len(stats) // rows
    need, ran, at = (stats[:, lane].reshape(rows, groups) for lane in range(3))
    alike = all((x == x[:, :1]).all() for x in (need, ran, at))
    need, ran, at = need[:, 0] != 0, ran[:, 0] != 0, at[:, 0]
    wanted_need = tie_round_needed(values, lengths, k)
    long = np.asarray(lengths) > k
    counts = np.array(
        [
            int((values[r][:n] >= values[r][wanted[r]].min()).sum()) if n > k else 0
            for r, n in enumerate(lengths)
        ]
    )
    passed = bool(
        alike
        and np.array_equal(need, wanted_need)
        and (ran == wanted_need.any()).all()
        and np.array_equal(at[long], counts[long])
    )
    line = (
        f"{int(need.sum())} of {rows} rows needed the tie round (reference "
        f"{int(wanted_need.sum())}), the round ran {bool(ran.any())}"
    )
    if not passed:
        line += (
            f"; rows {np.flatnonzero(need != wanted_need).tolist()[:8]} differ, "
            f"counts at tau {at[long][:4].tolist()} for {counts[long][:4].tolist()}, "
            f"virtual rows alike {alike}"
        )
    return passed, int(need.sum()), bool(ran.any()), line


def _select_check(
    rows, tokens, data, lengths, page_tokens, tag, extra=None, tie_skip=False
):
    """Each rank's verdict on its own mask. A row passes on a rank when its mask
    equals the reference's there, or, if every rank's does, when the group's
    conditions (:func:`select_verdicts`) hold. Those sum over ranks, so every
    rank's row statistics are gathered here. ``tie_skip``: the kernel's second
    output must also hold :func:`select_tie_stats`, whose rows needing the round
    (``tie_rows``) and ``tie_round_ran`` go into ``extra``."""

    def check(outputs):
        import torch

        from vllm_torchtpu.megakernel import bridge

        rank = bridge.kernel_mesh(AXIS, RANKS)[1]
        try:
            selected = unpack_mask(_first(outputs)[0].cpu().numpy(), rows, tokens)
            mine = _select_row_stats(
                selected, rank, rows, tokens, data, lengths, page_tokens
            )
            kernel_stats = outputs[1][0].cpu().numpy() if tie_skip else None
        except Exception as exc:  # noqa: BLE001 -- reported after the gather
            mine = f"{type(exc).__name__}: {exc}"
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, (rank, mine))
        if isinstance(mine, str):
            raise RuntimeError(mine)
        stats = [s for _, s in sorted(gathered)]
        if any(isinstance(s, str) for s in stats):
            # A peer has no statistics to judge the group by; it fails itself.
            return all(row[0] for row in mine)
        values, wanted, length = _select_reference(
            rows, tokens, data, lengths, page_tokens
        )
        verdicts = select_verdicts(stats, values, wanted, length)
        for passed, line in verdicts:
            if rows <= 8 or not passed:
                harness.log(f"{tag} check {line}")
        ok = all(
            passed if all(s[r][0] for s in stats) else mine[r][0]
            for r, (passed, _) in enumerate(verdicts)
        )
        if tie_skip:
            passed, tie_rows, ran, line = select_tie_stats(
                kernel_stats, values, wanted, length, rows
            )
            if extra is not None:
                extra["tie_rows"], extra["tie_round_ran"] = tie_rows, ran
            harness.log(f"{tag} check tie round {'PASS' if passed else 'FAIL'}: {line}")
            ok = ok and passed
        return ok

    return check


HOST_ORDER: list[tuple[int, str, tuple]] = []


def _host_order_checked() -> None:
    """The ``host`` exchange takes kernel rank ``8 h + lane`` to be lane
    ``lane`` of host ``h``: check it once from each worker's Slurm node and log
    the mapping with the devices' chip coordinates."""
    if HOST_ORDER:
        return
    import os

    import torch

    from vllm_torchtpu.megakernel import bridge

    mesh, rank = bridge.kernel_mesh(AXIS, RANKS)
    device = mesh.devices.flat[rank]
    mine = (
        rank,
        os.environ.get("SLURM_NODEID", "?"),
        (*getattr(device, "coords", ()), getattr(device, "core_on_chip", -1)),
    )
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, mine)
    HOST_ORDER.extend(sorted(gathered))
    harness.log(
        "select: kernel rank (Slurm node, chip x/y/z/core): "
        + ", ".join(
            f"{r} ({node}, {'/'.join(map(str, c))})" for r, node, c in HOST_ORDER
        )
    )
    nodes = [node for _, node, _ in HOST_ORDER]
    if any(nodes[r] != nodes[HOST_LANES * (r // HOST_LANES)] for r in range(RANKS)) or (
        len(set(nodes)) != RANKS // HOST_LANES
    ):
        raise RuntimeError(f"select: kernel ranks are not host-major: nodes {nodes}")


def _select_setup(
    rows,
    tokens,
    variant,
    exchange,
    ties,
    histogram,
    args,
    tie_skip=False,
    row_sum="after",
) -> Setup:
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel import bridge

    page_tokens = args.page_tokens
    select_pages(tokens, page_tokens)
    merge = exchange == "merge"
    if merge and rows > SUBLANES:
        raise ValueError(f"select: --exchange merge runs at R <= 8, not R = {rows}")
    if merge and tie_skip:
        raise ValueError(
            "select: --tie-skip has no meaning for --exchange merge, which has no "
            "tie round to skip"
        )
    if tie_skip and not ties:
        raise ValueError("select: --tie-skip on needs the tie round (--ties on)")
    if row_sum not in SELECT_ROW_SUMS:
        raise ValueError(f"select: row sum {row_sum!r} is not one of {SELECT_ROW_SUMS}")
    if virtual_groups(rows, tokens) == 1 or merge:
        row_sum = "after"  # a row is one virtual row, or no counts are exchanged
    if exchange == "host":
        _host_order_checked()
    if args.scores == "indexer":
        _indexer_scores(rows, tokens, page_tokens)
    kernel, scratch, words, vrows, planes = _select_kernel(
        rows,
        tokens,
        SELECT_BITS[variant],
        exchange,
        ties,
        page_tokens,
        histogram,
        tie_skip=tie_skip,
        row_sum_order=row_sum,
    )
    tag = f"{variant}_{exchange}" + ("" if ties else "_noties")
    tag += "_packed" if histogram == "packed" else ""
    tag += f"_p{page_tokens}" if page_tokens != 256 else ""
    tag += ("_a" if tie_skip else "") + ("_e" if row_sum == "before" else "")
    rank = bridge.kernel_mesh(AXIS, RANKS)[1]
    lengths = _select_lengths(rows, tokens, args.lengths)
    counts = [local_counts(length, tokens, page_tokens)[rank] for length in lengths]
    scores = to_virtual(
        _select_scores(rows, tokens, rank, args.scores, page_tokens),
        virtual_groups(rows, tokens),
    )
    meta = smem_table(np.asarray([counts, lengths], np.int32))
    carried = [np.zeros((words, vrows, LANES), np.int32)]
    if tie_skip:
        carried.append(np.zeros((vrows, LANES), np.int32))  # stats

    def body(meta, scores, *carry):
        *carry, iterations = carry
        shapes = [jax.ShapeDtypeStruct(c.shape, c.dtype) for c in carry]
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        call = pl.pallas_call(
            kernel,
            name=f"dsa_micro_select_{tag}",
            out_shape=shapes if tie_skip else shapes[0],
            in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), hbm, *[hbm] * len(carry)],
            out_specs=[hbm] * len(carry) if tie_skip else hbm,
            scratch_shapes=scratch,
            input_output_aliases={2 + i: i for i in range(len(carry))},
            compiler_params=pltpu.CompilerParams(
                collective_id=0 if exchange != "none" else None,
                vmem_limit_bytes=SELECT_VMEM_LIMIT,
            ),
        )
        # The aliased mask orders the calls; each call overwrites it.
        if not tie_skip:
            loop = jax.lax.fori_loop(
                0, iterations, lambda _, m: call(meta, scores, m), carry[0]
            )
            return (loop,)
        return tuple(
            jax.lax.fori_loop(
                0, iterations, lambda _, c: tuple(call(meta, scores, *c)), tuple(carry)
            )
        )

    invalid = exchange in TIMING_ONLY or not ties
    extra = {
        "exchange": exchange,
        "histogram": histogram,
        "ties": "on" if ties else "off",
        "select_page_tokens": page_tokens,
        "tie_skip": "on" if tie_skip else "off",
        "row_sum": row_sum,
    }
    if merge:
        width = min(tokens, SELECT_K)
        # The global select's candidates per row, and the 8-byte (key, id)
        # candidates a rank receives from its peers per call; none are
        # gathered without the tie step (timing only).
        extra["merge_candidates"] = RANKS * width if ties else None
        extra["merge_bytes"] = (RANKS - 1) * rows * width * 8 if ties else None
    return Setup(
        body,
        (bridge.upload(meta), bridge.upload(scores), *map(bridge.upload, carried)),
        (P(AXIS),) * (2 + len(carried)),
        (P(AXIS),) * len(carried),
        variant=tag,
        check=None
        if invalid
        else _select_check(
            rows,
            tokens,
            args.scores,
            lengths,
            page_tokens,
            f"select_r{rows}_t{tokens}_{tag}",
            extra,
            tie_skip,
        ),
        invalid=invalid,
        extra=extra,
    )


def _build_select(rows: int, tokens: int, args) -> list[Setup]:
    """The distinct select setups at one point, one per ``--histogram`` x
    ``--variant`` x ``--exchange`` x ``--ties`` x ``--tie-skip`` x
    ``--row-sum``. The variant tag is ``{radix4|radix8}_{exchange}`` plus
    ``_noties`` (``--ties off``), ``_packed`` (``--histogram packed``),
    ``_p{N}`` (``--page-tokens N`` other than 256), ``_a`` (``--tie-skip on``)
    and ``_e`` (``--row-sum before`` below eight rows; from eight rows it is the
    default setup); the record's ``extra`` fields carry the same settings.
    ``--exchange merge`` (R <= 8) adds no suffix; its ``extra`` adds
    ``merge_candidates`` (per row) and ``merge_bytes`` (received per rank per
    call)."""
    variants = [v for v in args.variant if v in SELECT_BITS] or list(SELECT_BITS)
    setups = {}
    for histogram, variant, exchange, ties, tie_skip, row_sum in itertools.product(
        args.histogram,
        variants,
        args.exchange,
        args.ties,
        getattr(args, "tie_skip", ["off"]),
        getattr(args, "row_sum", ["after"]),
    ):
        setup = _select_setup(
            rows,
            tokens,
            variant,
            exchange,
            ties == "on",
            histogram,
            args,
            tie_skip == "on",
            row_sum,
        )
        setups.setdefault(setup.variant, setup)
    return list(setups.values())


def _select_rounds_form(rows: int, tokens: int, bits: int, rounds: int):
    """``select_rounds``' form: ``dsa_select.select_form`` under the installed
    KernelConfig (``--set``: pack16, digit bits, stop and count options), at
    eight rows ``bits`` a digit when nonzero; with ``rounds`` the stop after
    any round (capped there), else the config's own stop."""
    import dataclasses

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    form = dsa_select.select_form(rows, tokens)
    if bits and bits != form.bits:
        if rows != 8:
            raise ValueError(
                f"select_rounds: {rows} rows take {form.bits}-bit digits, not {bits}"
            )
        form = dataclasses.replace(form, bits=bits)
    return dataclasses.replace(form, stop="any") if rounds else form


def select_rounds_call(
    rows: int, tokens: int, page_tokens: int, form, rounds: int, interpret=None
):
    """``(call, mask shape)``: one ``select_rounds`` call, ``call(meta, scores,
    mask)`` on this rank's ``meta [1, 1, >= rows]`` positions (SMEM), ``scores
    [1, planes, rows, 128]`` FP32 and the aliased ``mask [1, words, rows, 128]``
    int32 (HBM), under a ``shard_map`` over the 32-rank ``"tp"`` group: an
    entry barrier, the scores' load, ``dsa_select.TopkSelect.run`` (``rounds``:
    its ``rounds_cap``) and the mask's store. ``interpret``: Pallas TPU
    interpret mode (the CPU check)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.common import collectives32
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode, dsa_select

    shapes = dsa_select.scratch_shapes(rows, tokens, form)
    planes, vrows, _ = dsa_decode.score_shape(tokens, rows)

    def kernel(
        meta_ref,
        scores_hbm,
        _,
        mask_hbm,
        scores_vmem,
        mask,
        bounds,
        state,
        histogram,
        sem,
        smem,
        *wires,
    ):
        load = pltpu.make_async_copy(scores_hbm.at[0], scores_vmem, sem.at[0])
        load.start()
        collectives32.barrier()  # peers write into this call's inboxes
        load.wait()
        member = jax.lax.axis_index(AXIS)
        select = dsa_select.TopkSelect(
            dsa_select.SelectScratch(mask, bounds, state, histogram, smem, wires),
            batch=rows,
            local_tokens=tokens,
            position_ref=meta_ref.at[0, 0],
            owner=member,
            member=member,
            base=0,
            group=RANKS,
            page_tokens=page_tokens,
            form=form,
            rounds_cap=rounds,
        )
        select.run(scores_vmem)
        store = pltpu.make_async_copy(mask, mask_hbm.at[0], sem.at[1])
        store.start()
        store.wait()

    mask_shape = (1, *shapes["mask"][0])
    call = pl.pallas_call(
        kernel,
        name=f"dsa_micro_select_rounds_b{form.bits}_{form.stop}",
        out_shape=jax.ShapeDtypeStruct(mask_shape, jnp.int32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            *[pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[
            pltpu.VMEM((planes, vrows, LANES), jnp.float32),
            *(
                pltpu.VMEM(*shapes[name])
                for name in ("mask", "bounds", "state", "histogram")
            ),
            pltpu.SemaphoreType.DMA((2,)),
            *dsa_select.scoped_scratch(),
            *dsa_select.persistent_scratch(rows, tokens, page_tokens, RANKS, form),
        ],
        input_output_aliases={2: 0},
        compiler_params=pltpu.CompilerParams(
            collective_id=0, vmem_limit_bytes=SELECT_VMEM_LIMIT
        ),
        interpret=interpret,
    )
    return call, mask_shape


def _select_rounds_values(rows: int, tokens: int, rank: int):
    """``rank``'s ``select_rounds`` scores ``[rows, tokens]``, random normal."""
    return (
        np.random.default_rng([SELECT_SEED, rows, rank])
        .standard_normal((rows, tokens))
        .astype(np.float32)
    )


def select_rounds_operands(rows: int, tokens: int, rank: int, lengths):
    """``select_rounds``' numpy operands of ``rank``: ``(meta [1, >= 8]
    positions, scores [planes, rows, 128])``, random normal scores."""
    positions = np.asarray(lengths, np.int32) - 1
    return smem_table(positions[None, :]), to_virtual(
        _select_rounds_values(rows, tokens, rank), 1
    )


def select_rounds_differing_rows(words, rank: int, rows, tokens, lengths, page_tokens):
    """The rows whose ``select_rounds`` mask words (``[words, rows, 128]`` int32)
    on ``rank`` differ from ``dsa_reference.decode_selection``, every rank's
    scores regenerated here."""
    import jax

    from tools.megakernel import dsa_reference
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    local = []
    for r in range(RANKS):
        values = _select_rounds_values(rows, tokens, r)
        counts = [int(local_counts(n, tokens, page_tokens)[r]) for n in lengths]
        row_values = np.full((rows, tokens), np.nan, np.float32)
        for b, n in enumerate(counts):
            row_values[b, :n] = values[b, :n]
        local.append(row_values)
    with jax.default_device(jax.devices("cpu")[0]):
        want, _ = dsa_reference.decode_selection(
            local, [n - 1 for n in lengths], page_tokens=page_tokens, group=RANKS
        )
    got = dsa_select.unpack_mask(words, rows, tokens)
    return np.flatnonzero((got != want[rank]).any(axis=1))


def _select_rounds_check(rows, tokens, lengths, page_tokens, tag):
    """Each rank's verdict on its own mask against the reference."""

    def check(outputs):
        from vllm_torchtpu.megakernel import bridge

        rank = bridge.kernel_mesh(AXIS, RANKS)[1]
        words = _first(outputs)[0].cpu().numpy()
        bad = select_rounds_differing_rows(
            words, rank, rows, tokens, lengths, page_tokens
        )
        if len(bad):
            harness.log(f"{tag} check FAIL on rank {rank}: rows {bad[:8].tolist()}")
        return len(bad) == 0

    return check


def _build_select_rounds(rows: int, tokens: int, args) -> list[Setup]:
    """The decode select itself (``dsa_select.TopkSelect``, the 32-rank group,
    one entry barrier a call, :func:`select_rounds_call`) per ``--digit-bits``
    x ``--rounds`` under ``--set``: ``--rounds 0`` the config's rounds, ``N`` exactly N
    rounds of the stop form (the stop check computed, not obeyed:
    ``rounds_cap``, timing only). Each call reloads this rank's scores (random
    normal, rows of ``--lengths``) from HBM. ``--rounds 0`` is checked: every
    rank's mask against ``dsa_reference.decode_selection``."""
    import jax
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_select

    page_tokens = args.page_tokens
    select_pages(tokens, page_tokens)
    if rows < 8:
        raise ValueError("select_rounds: eight rows and more (one virtual row a row)")
    rank = bridge.kernel_mesh(AXIS, RANKS)[1]
    lengths = _select_lengths(rows, tokens, args.lengths)
    meta, scores = select_rounds_operands(rows, tokens, rank, lengths)
    setups = []
    for bits, rounds in itertools.product(args.digit_bits, args.rounds):
        form = _select_rounds_form(rows, tokens, bits, rounds)
        call, mask_shape = select_rounds_call(rows, tokens, page_tokens, form, rounds)

        def body(meta, scores, mask, iterations, call=call):
            return (
                jax.lax.fori_loop(
                    0, iterations, lambda _, m: call(meta, scores, m), mask
                ),
            )

        packet = dsa_select._histogram_packet(rows, 1, form)
        setups.append(
            Setup(
                body,
                (
                    bridge.upload(meta),
                    bridge.upload(scores),
                    bridge.upload(np.zeros(mask_shape[1:], np.int32)),
                ),
                (P(AXIS),) * 3,
                (P(AXIS),),
                variant=f"bits{form.bits}_rounds{rounds or form.rounds}_"
                f"{form.histogram}" + ("_stop" if rounds else ""),
                check=None
                if rounds
                else _select_rounds_check(
                    rows,
                    tokens,
                    lengths,
                    page_tokens,
                    f"select_rounds_r{rows}_t{tokens}_bits{form.bits}",
                ),
                invalid=bool(rounds),
                extra={
                    "bits": form.bits,
                    "rounds": rounds or form.rounds,
                    "stop": form.stop,
                    "wire": form.wire,
                    "exchange": form.exchange,
                    "histogram": form.histogram,
                    "packet_bytes": 4 * packet[0] * packet[1],
                    "select_page_tokens": page_tokens,
                    "lengths": lengths[:1] + lengths[-1:],
                },
            )
        )
    return setups


@dataclass(frozen=True)
class PoolGeometry:
    """A unified pool ``[blocks, 128, packing, 640]``: ``packing`` tokens share
    each 32-bit word of a 2560-byte physical row (BF16 two, FP8 four)."""

    name: str
    packing: int
    itemsize: int

    @property
    def block_tokens(self) -> int:
        return POOL_ROWS * self.packing

    @property
    def token_bytes(self) -> int:
        """The bytes of one token's latent row."""
        return LATENT_LANES * self.itemsize

    @property
    def row_bytes(self) -> int:
        """The bytes of one sublane row, holding ``packing`` tokens."""
        return LATENT_LANES * self.packing * self.itemsize


GEOMETRY = {"bf16": PoolGeometry("bf16", 2, 2), "fp8": PoolGeometry("fp8", 4, 1)}


def locate(tokens, geometry: PoolGeometry):
    """``(block index, sublane row, packing slot)`` of local tokens ``tokens``
    (arrays) of a sequence whose block ``e`` holds tokens ``[e B, (e + 1) B)``,
    ``B`` the block's token count: token ``t`` of a block at ``[t // packing, t %
    packing]``."""
    tokens = np.asarray(tokens)
    within = tokens % geometry.block_tokens
    return (
        tokens // geometry.block_tokens,
        within // geometry.packing,
        within % geometry.packing,
    )


def recent_window(count: int) -> int:
    """The local tokens the ``recent`` pattern's recent share is drawn from: the
    last 4096 global positions are the rank's last ``RECENT_TOKENS`` tokens when
    it holds its balanced share of the selection, and grow in proportion to
    ``count`` beyond that (a rank holding all of it owns every page)."""
    return RECENT_TOKENS * max(1, count // BALANCED_SELECTED)


def select_tokens(rng, tokens: int, count: int, pattern: str):
    """``count`` distinct local tokens of ``tokens``, ascending. ``uniform``:
    random positions. ``recent``: ``RECENT_SHARE`` in the last
    :func:`recent_window`, ``SINK_SHARE`` in the first ``SINK_TOKENS``, the rest
    uniform over what is left."""
    if not 0 < count <= tokens:
        raise ValueError(f"gather: cannot select {count} of {tokens} tokens")
    taken = np.zeros(tokens, bool)
    chosen = []

    def take(candidates, wanted):
        free = candidates[~taken[candidates]]
        picked = rng.choice(free, min(wanted, free.size), replace=False)
        taken[picked] = True
        chosen.append(picked)

    if pattern == "recent":
        window = min(tokens, recent_window(count))
        take(np.arange(tokens - window, tokens), round(RECENT_SHARE * count))
        take(np.arange(min(tokens, SINK_TOKENS)), round(SINK_SHARE * count))
    elif pattern != "uniform":
        raise ValueError(f"gather: unknown pattern {pattern!r}")
    take(np.arange(tokens), count - sum(len(c) for c in chosen))
    return np.sort(np.concatenate(chosen))


def gather_units(selected, variant: str, geometry: PoolGeometry):
    """``(block index, first sublane row)`` of the DMAs that gather one row's
    ``selected`` tokens (ascending): ``row`` one per token, for the sublane row
    holding it; ``tile`` one per distinct tile of ``TILE_ROWS`` sublane rows
    holding any; ``page`` one per distinct block (first row 0)."""
    block, sublane, _ = locate(selected, geometry)
    if variant == "row":
        return block, sublane
    if variant == "tile":
        tiles = np.unique(block * (POOL_ROWS // TILE_ROWS) + sublane // TILE_ROWS)
        return tiles // (POOL_ROWS // TILE_ROWS), TILE_ROWS * (
            tiles % (POOL_ROWS // TILE_ROWS)
        )
    if variant == "page":
        blocks = np.unique(block)
        return blocks, np.zeros_like(blocks)
    raise ValueError(f"gather: unknown variant {variant!r}")


def pack_mask(mask, rows: int, tokens: int):
    """The select kernel's output words of the bool mask ``[R, T]``: byte ``i``
    of word ``w`` is plane ``4 w + i``'s 0/1 mask (the inverse of
    :func:`unpack_mask`)."""
    planes = to_virtual(np.asarray(mask, np.uint8), virtual_groups(rows, tokens))
    padded = np.zeros((-(-len(planes) // 4) * 4, *planes.shape[1:]), np.uint8)
    padded[: len(planes)] = planes
    words = padded.reshape(-1, 4, *planes.shape[1:]).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(words).view(np.int32)[..., 0]


def unit_tokens(variant: str, geometry: PoolGeometry) -> int:
    """The tokens of one DMA unit: a sublane row, a tile of ``TILE_ROWS``
    sublane rows, or a block."""
    return {"row": 1, "tile": TILE_ROWS, "page": POOL_ROWS}[variant] * geometry.packing


def distinct_units(selected, variant: str, geometry: PoolGeometry):
    """``(block index, first sublane row, token bitmask)`` of each distinct unit
    holding a selected token, ascending: one DMA each. Unlike
    :func:`gather_units` (one ``row`` DMA per token) rows deduplicate. Bit
    ``j`` of a bitmask is the unit's ``j``-th token; pages carry none (the fold
    reads the select's mask)."""
    size = unit_tokens(variant, geometry)
    selected = np.asarray(selected, np.int64)
    units, inverse = np.unique(selected // size, return_inverse=True)
    bits = np.zeros(len(units), np.int64)
    if variant != "page":
        np.bitwise_or.at(bits, inverse, np.int64(1) << (selected % size))
    first_token = units * size
    return (
        first_token // geometry.block_tokens,
        (first_token % geometry.block_tokens) // geometry.packing,
        bits,
    )


def pair_counts(selected, geometry: PoolGeometry):
    """One row's ``(selected tokens, distinct sublane rows, tiles, pages)``."""
    return (
        len(selected),
        *(len(distinct_units(selected, v, geometry)[0]) for v in GATHER_VARIANTS),
    )


def _int32(values):
    return (np.asarray(values, np.int64) & 0xFFFFFFFF).astype(np.uint32).view(np.int32)


@dataclass
class CompactLists:
    """A rank's compact output, per virtual row ``v = r groups + g``:
    ``codes`` and ``bitmasks`` ``[V, C]`` (code ``block << 8 | first sublane
    row``, relative to the row; 0 past the list) and ``counts [V, 4]``
    (:data:`COUNT_FIELDS`: part g's selected tokens, sublane rows and tiles;
    the row's pages on its first part, 0 on the others). ``row`` and
    ``tile`` lists are per part, ``page`` lists per row on its first part;
    each holds its first C units, the counts all of them. Code 0 is a real
    unit (block 0, sublane row 0): a list's length is its count, never the
    number of nonzero codes."""

    codes: Any
    bitmasks: Any
    counts: Any


def compact_expected(selections, rows, tokens, variant, geometry, capacity):
    """The :class:`CompactLists` of every row's ``selections`` (ascending local
    tokens), from :func:`distinct_units`."""
    groups = virtual_groups(rows, tokens)
    part = tokens // groups
    vrows = padded_rows(rows) if groups > 1 else rows
    codes = np.zeros((vrows, capacity), np.int64)
    bitmasks = np.zeros_like(codes)
    counts = np.zeros((vrows, len(COUNT_FIELDS)), np.int64)
    for r, selected in enumerate(selections):
        selected = np.asarray(selected, np.int64)
        for g in range(groups):
            v = r * groups + g
            mine = selected[(selected >= g * part) & (selected < (g + 1) * part)]
            counts[v, :3] = pair_counts(mine, geometry)[:3]
            if g == 0:
                counts[v, 3] = len(distinct_units(selected, "page", geometry)[0])
            if variant == "page" and g:
                continue
            block, first, bits = distinct_units(
                selected if variant == "page" else mine, variant, geometry
            )
            n = min(len(block), capacity)
            codes[v, :n] = (block[:n] << 8) | first[:n]
            bitmasks[v, :n] = bits[:n]
    return CompactLists(_int32(codes), _int32(bitmasks), _int32(counts))


def _compact_lanes(variant: str, geometry: PoolGeometry):
    """``(rows, tiles, first, units)``: a 128-token plane's sublane rows and
    tiles, and its units of ``variant`` (lanes ``[first, first + units)`` of the
    ``units`` selector product, :func:`_compact_selectors`; none for ``page``,
    whose units span planes)."""
    rows = LANES // geometry.packing
    tiles = rows // TILE_ROWS
    first, units = {"row": (0, rows), "tile": (rows, tiles), "page": (0, 0)}[variant]
    return rows, tiles, first, units


def _compact_selectors(row, col, variant: str, geometry: PoolGeometry, xp):
    """The compact kernel's ``[128, 128]`` selector matrices (integers, exact as
    BF16 dot operands; ``row``, ``col`` their index grids, ``xp`` numpy or
    ``jax.numpy``):

    * ``units``: a plane's 0/1 token mask times it counts each sublane row's
      tokens in lanes ``[0, rows)``, each tile's in the next ``tiles``
      (:func:`_compact_lanes`), and holds its tokens as eight 16-bit halves in
      the eight lanes after them (half ``h``: tokens ``[16 h, 16 h + 16)``).
    * ``bits``: the variant's 0/1 unit flags (lanes of that product) times it
      are its unit bits as 16-bit halves in lanes 0-3 (zero for ``page``).

    Every product is a sum of distinct powers of two below 2^16 (or a count),
    exact in an FP32 accumulator."""
    rows, tiles, first, units = _compact_lanes(variant, geometry)
    words = rows + tiles
    packing_shift = geometry.packing.bit_length() - 1
    tile_shift = (TILE_ROWS * geometry.packing).bit_length() - 1
    unit_matrix = xp.where((col < rows) & ((row >> packing_shift) == col), 1, 0)
    in_tiles = (col >= rows) & (col < words) & ((row >> tile_shift) == col - rows)
    unit_matrix = xp.where(in_tiles, 1, unit_matrix)
    in_words = (col >= words) & (col < words + 8) & ((row >> 4) == col - words)
    unit_matrix = xp.where(in_words, xp.left_shift(1, row & 15), unit_matrix)
    unit = row - first
    in_units = (unit >= 0) & (unit < units) & ((unit >> 4) == col)
    bit_matrix = xp.where(in_units, xp.left_shift(1, unit & 15), 0)
    return unit_matrix, bit_matrix


def _popcount(x, srl):
    """The set bits of each 32-bit word of ``x`` (SWAR adds; ``srl``: a logical
    right shift)."""
    x = x - (srl(x, 1) & 0x55555555)
    x = (x & 0x33333333) + (srl(x, 2) & 0x33333333)
    x = (x + srl(x, 4)) & 0x0F0F0F0F
    x = x + srl(x, 8)
    x = x + srl(x, 16)
    return x & 0x3F


def _rank_select(words, k, srl, xp):
    """The index of the ``k``-th set bit (from 0) of the bit string ``words``
    (bit ``i`` of word ``w`` is its bit ``32 w + i``; any value when ``k`` is
    not below its popcount): the word by running popcounts, then halves of 16,
    8, 4, 2 and 1 bits."""
    x, index, below = words[0], 0, 0
    seen = _popcount(x, srl)
    for w, word in enumerate(words[1:], 1):
        later = k >= seen
        x = xp.where(later, word, x)
        below = xp.where(later, seen, below)
        index = xp.where(later, 32 * w, index)
        seen = seen + _popcount(word, srl)
    k = k - below
    for width in (16, 8, 4, 2, 1):
        low = x & ((1 << width) - 1)
        ones = _popcount(low, srl)
        high = k >= ones
        k = xp.where(high, k - ones, k)
        index = xp.where(high, index + width, index)
        x = xp.where(high, srl(x, width), low)
    return index


def _unit_code(token, geometry: PoolGeometry):
    """The code ``block << 8 | first sublane row`` of the unit whose first
    row-local token is ``token``."""
    block_shift = geometry.block_tokens.bit_length() - 1
    packing_shift = geometry.packing.bit_length() - 1
    first = (token & (geometry.block_tokens - 1)) >> packing_shift
    return ((token >> block_shift) << 8) | first


def _unit_bitmask(token_words, unit, size: int, srl, xp):
    """Unit ``unit``'s ``size`` token bits, out of its plane's four 32-bit
    token words (token ``l`` is bit ``l % 32`` of word ``l // 32``): the word by
    three selects, then a shift (no lane gather)."""
    first = unit * size
    word = token_words[0]
    for w in range(1, 4):
        word = xp.where(srl(first, 5) == w, token_words[w], word)
    if size == 32:
        return word
    return srl(word, first & 31) & ((1 << size) - 1)


def compact_model(words, rows, tokens, variant, geometry, capacity):
    """The compact kernel's algorithm on the host, phase by phase (its spec):
    the select's mask ``words`` (int32 ``[words, V, 128]``) -> the
    :class:`CompactLists`. Per block of eight virtual rows:

    1. Per plane ``s`` of the rows' parts: the 0/1 token mask ``m_s [8, 128]``
       (byte ``s % 4`` of word ``s // 4``) times the ``units`` selector
       (:func:`_compact_selectors`) gives each sublane row's and tile's token
       count and the plane's token bits, the variant's unit flags (counts > 0)
       times ``bits`` its unit bits, and their row sum its unit count ``c_s``.
       The selected tokens, sublane rows and tiles add up over the planes, and
       the pages any plane touches are flagged.
    2. The plane bases: the exclusive running sums of ``c_s``.
    3. Per list position ``p``: the plane holding it (``base_s <= p < base_s +
       c_s``) and that plane's fields; its unit by a rank-select of the
       plane's unit bits; the unit's code and its token bits by a shift of the
       plane's token words. Positions past the total are 0.
    4. Pages: the flags summed over a row's parts (``group_sum``); the row's
       page count on its first part, and for ``page`` its list there, by the
       same rank-select over the flags' four words."""
    srl = np.right_shift
    groups = virtual_groups(rows, tokens)
    vrows, part = rows * groups, tokens // groups
    planes = part // LANES
    size = unit_tokens(variant, geometry)
    row_lanes, tile_lanes, first_lane, units = _compact_lanes(variant, geometry)
    word_lane = row_lanes + tile_lanes
    unit_words = -(-units // 32)
    unit_matrix, bit_matrix = (
        m.astype(np.float32)
        for m in _compact_selectors(*np.indices((LANES, LANES)), variant, geometry, np)
    )
    page_shift = geometry.block_tokens.bit_length() - 1
    lane = np.arange(LANES)[None]
    position = np.arange(capacity)[None]
    part_of = np.arange(SUBLANES)[:, None] % groups
    first_part = part_of == 0
    words = np.asarray(words).astype(np.int64) & 0xFFFFFFFF

    def paired(x, lane0, count):
        """``count`` 32-bit words from the 16-bit halves in lanes from ``lane0``."""
        return [
            x[:, lane0 + 2 * w : lane0 + 2 * w + 1]
            | (x[:, lane0 + 2 * w + 1 : lane0 + 2 * w + 2] << 16)
            for w in range(count)
        ]

    out = [np.zeros((vrows, capacity), np.int64) for _ in range(2)]
    counts = np.zeros((vrows, len(COUNT_FIELDS)), np.int64)
    for b in range(vrows // SUBLANES):
        block = slice(SUBLANES * b, SUBLANES * (b + 1))
        # Phase 1.
        tallies = np.zeros((SUBLANES, 3), np.int64)
        page_any = np.zeros((SUBLANES, LANES), bool)
        fields = []
        for s in range(planes):
            m = srl(words[s // 4, block], 8 * (s % 4)) & 0xFF
            y = (m.astype(np.float32) @ unit_matrix).astype(np.int64)
            hit = y > 0
            tallies[:, 0] += m.sum(axis=1)
            tallies[:, 1] += hit[:, :row_lanes].sum(axis=1)
            tallies[:, 2] += hit[:, row_lanes:word_lane].sum(axis=1)
            page = (part_of * part + LANES * s) >> page_shift
            page_any |= (lane == page) & m.any(axis=1, keepdims=True)
            if units:
                flags = hit & (lane >= first_lane) & (lane < first_lane + units)
                z = (flags.astype(np.float32) @ bit_matrix).astype(np.int64)
                fields.append(
                    (
                        paired(z, 0, unit_words),
                        paired(y, word_lane, 4),
                        flags.sum(axis=1, keepdims=True),
                    )
                )
        if units:
            # Phase 2.
            unit_counts = np.concatenate([f[2] for f in fields], axis=1)
            bases = np.cumsum(unit_counts, axis=1) - unit_counts
            total = unit_counts.sum(axis=1, keepdims=True)
            # Phase 3.
            plane = np.zeros((SUBLANES, capacity), np.int64)
            base = np.zeros_like(plane)
            unit_bits = [np.zeros_like(plane)] * unit_words
            token_bits = [np.zeros_like(plane)] * 4
            for s, (plane_units, plane_tokens, _) in enumerate(fields):
                count, start = unit_counts[:, s : s + 1], bases[:, s : s + 1]
                inside = (start <= position) & (position < start + count)
                plane += start + count <= position
                base = np.where(inside, start, base)
                unit_bits = [
                    np.where(inside, f, a) for f, a in zip(plane_units, unit_bits)
                ]
                token_bits = [
                    np.where(inside, f, a) for f, a in zip(plane_tokens, token_bits)
                ]
            unit = _rank_select(unit_bits, position - base, srl, np)
            token = part_of * part + (plane * (LANES // size) + unit) * size
            valid = position < total
            out[0][block] = np.where(valid, _unit_code(token, geometry), 0)
            bitmask = _unit_bitmask(token_bits, unit, size, srl, np)
            out[1][block] = np.where(valid, bitmask, 0)
        # Phase 4.
        flags = page_any.reshape(-1, groups, LANES).sum(axis=1, keepdims=True)
        flags = np.repeat(flags, groups, axis=1).reshape(SUBLANES, LANES) > 0
        pages = flags.sum(axis=1, keepdims=True)
        counts[block, :3] = tallies
        counts[block, 3:] = np.where(first_part, pages, 0)
        if variant == "page":
            y = (flags.astype(np.float32) @ unit_matrix).astype(np.int64)
            page = _rank_select(paired(y, word_lane, 4), position, srl, np)
            listed = first_part & (position < pages)
            out[0][block] = np.where(
                listed, _unit_code(page << page_shift, geometry), 0
            )
    return CompactLists(_int32(out[0]), _int32(out[1]), _int32(counts))


def dma_codes(units):
    """The DMA list of ``units`` (per row, ``(block index, first sublane row)``
    arrays) as int32 codes ``row << 16 | block index << 8 | first row``, in row
    order."""
    codes = []
    for row, (block, first) in enumerate(units):
        if row >= 2**15 or block.max(initial=0) >= 256 or first.max(initial=0) >= 256:
            raise ValueError("gather: row, block index or sublane row out of range")
        codes.append((row << 16) | (block.astype(np.int64) << 8) | first)
    return np.concatenate(codes).astype(np.int32)


def block_table(rows: int, per_row: int, seed: int):
    """``[rows, per_row]`` distinct pool blocks: each row's pages scattered over
    a pool of ``rows * per_row`` blocks."""
    return (
        np.random.default_rng([seed, rows, per_row])
        .permutation(rows * per_row)
        .reshape(rows, per_row)
        .astype(np.int32)
    )


def ring_slots(dmas: int, inflight: int) -> int:
    """The slots of the gather ring: ``inflight`` (a power of two), or fewer when
    the whole list fits."""
    if inflight < 1 or inflight & (inflight - 1):
        raise ValueError(f"gather: in-flight cap {inflight} must be a power of two")
    return min(inflight, 1 << max(0, dmas - 1).bit_length())


def unit_words(pool_bits, blocks, first, slot_rows: int, geometry: PoolGeometry):
    """``[n, 640]`` uint32: the sum over the ``slot_rows`` sublane rows from
    ``first`` of block ``blocks`` of the rows' 32-bit words, ``packing`` tokens'
    bits to a word, token 0 lowest (the kernel's int32 view of the pool dtype),
    each weighted by ``2 j + 1`` for its position ``j`` in the unit."""
    rows = first[:, None] + np.arange(slot_rows)
    units = pool_bits[blocks[:, None], rows].astype(np.uint32)
    shift = np.arange(geometry.packing, dtype=np.uint32) * (8 * geometry.itemsize)
    words = (units << shift[None, None, :, None]).sum(axis=2, dtype=np.uint32)
    weight = 2 * np.arange(slot_rows, dtype=np.uint32) + 1
    return (words * weight[None, :, None]).sum(axis=1, dtype=np.uint32)


def gather_accumulators(
    pool_bits, table, units, slot_rows: int, rows_padded: int, geometry
):
    """``[rows_padded, 640]`` int32, the verify kernel's expected output: row
    ``r`` sums, over its DMAs ``n`` (numbered from 0 over all rows in the DMA
    list), ``2 n + 1`` times the DMA's :func:`unit_words`, modulo 2**32. The
    weights are odd so that no bit of a word drops out of the product."""
    out = np.zeros((rows_padded, LATENT_LANES), np.uint32)
    ordinal = 0
    for row, (block, first) in enumerate(units):
        weights = 2 * np.arange(ordinal, ordinal + len(block), dtype=np.uint32) + 1
        words = unit_words(pool_bits, table[row][block], first, slot_rows, geometry)
        out[row] = (words * weights[:, None]).sum(axis=0, dtype=np.uint32)
        ordinal += len(block)
    return out.view(np.int32)


@dataclass
class GatherPool:
    """A rank's pool on device and its bits on the host: ``max_rows`` sequences
    of ``tokens`` tokens, their blocks scattered by ``table``."""

    device: Any
    bits: Any
    table: Any


GATHER_POOLS: dict[tuple, GatherPool] = {}


def _pool_bits(kv_dtype: str, tokens: int, max_rows: int, rank: int):
    """``(bits, table)``: rank ``rank``'s pool on the host, ``max_rows``
    sequences of ``tokens`` local tokens, ``bits [blocks, 128, packing, 640]``
    (the uint16 / uint8 words of random normal BF16 / FP8 values) and the
    :func:`block_table` scattering each sequence's blocks over it."""
    import ml_dtypes

    geometry = GEOMETRY[kv_dtype]
    if tokens % geometry.block_tokens:
        raise ValueError(
            f"pool: {tokens} local tokens must be whole {kv_dtype} blocks of "
            f"{geometry.block_tokens}"
        )
    per_row = tokens // geometry.block_tokens
    shape = (max_rows * per_row, POOL_ROWS, geometry.packing, LATENT_LANES)
    rng = np.random.default_rng([GATHER_SEED + geometry.packing, rank])
    dtype = {"bf16": ml_dtypes.bfloat16, "fp8": ml_dtypes.float8_e4m3fn}[kv_dtype]
    bits = np.empty(shape, np.uint8 if kv_dtype == "fp8" else np.uint16)
    flat = bits.reshape(-1)
    for start in range(0, flat.size, 2**24):
        part = rng.standard_normal(min(2**24, flat.size - start), np.float32)
        flat[start : start + part.size] = part.astype(dtype).view(flat.dtype)
    return bits, block_table(max_rows, per_row, GATHER_SEED)


def _gather_pool(kv_dtype: str, tokens: int, max_rows: int) -> GatherPool:
    import torch

    from vllm_torchtpu.megakernel import bridge

    key = (kv_dtype, tokens, max_rows)
    if key not in GATHER_POOLS:
        for old in [k for k in GATHER_POOLS if k[1:] != key[1:]]:
            del GATHER_POOLS[old]
        rank = bridge.kernel_mesh(AXIS, RANKS)[1]
        bits, table = _pool_bits(kv_dtype, tokens, max_rows, rank)
        view = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[kv_dtype]
        GATHER_POOLS[key] = GatherPool(_upload_bits(bits, view), bits, table)
    return GATHER_POOLS[key]


def _gather_selection(rows: int, tokens: int, count: int, pattern: str):
    """Each row's selected local tokens, the same on every rank (and for any
    ``rows`` at least the row's index)."""
    return [
        select_tokens(
            np.random.default_rng(
                [GATHER_SEED, tokens, count, GATHER_PATTERNS.index(pattern), row]
            ),
            tokens,
            count,
            pattern,
        )
        for row in range(rows)
    ]


def row_selections(seed, counts, tokens: int, pattern: str, suffix=()):
    """Row ``r``'s ``counts[r]`` selected local tokens (:func:`select_tokens`,
    ascending), drawn from ``default_rng([*seed, r, *suffix])``; a row of no
    tokens draws nothing."""
    return [
        select_tokens(np.random.default_rng([*seed, r, *suffix]), tokens, n, pattern)
        if n
        else np.zeros(0, np.int64)
        for r, n in enumerate(counts)
    ]


def row_mix_counts(count: int, tokens: int, rows: int, row_mix: bool) -> list[int]:
    """Each row's selected tokens: ``count``, or with ``row_mix`` ``count``,
    none, one and every token, cycled over the rows."""
    if not row_mix:
        return [count] * rows
    return [[count, 0, 1, tokens][r % 4] for r in range(rows)]


def _gather_kernel(variant, geometry, per_row, dmas, slots, group, verify):
    """The gather ``pallas_call`` kernel and its scratch shapes. The DMAs go into
    a ring of ``slots`` VMEM slots, ``group`` consecutive ones sharing a
    semaphore: the first ``slots`` DMAs start at once, then each group of
    ``group`` starts after one byte-counted wait for the group that last used its
    slots, and the last groups are waited at the end. ``verify`` consumes each
    slot after its wait (the slot's words, weighted by the sublane row's position
    in the slot and by the DMA's number, into the row's accumulator); otherwise
    the kernel only reads one word of the last slot into the carry."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    slot_rows = GATHER_SLOT_ROWS[variant]
    dtype = jnp.bfloat16 if geometry.name == "bf16" else jnp.float8_e4m3fn
    groups = slots // group
    chunks, tail = divmod(dmas, group)
    scratch = [
        pltpu.VMEM((slots, slot_rows, geometry.packing, LATENT_LANES), dtype),
        pltpu.SemaphoreType.DMA((groups,)),
    ]

    def kernel(table, codes, pool, carry, acc, ring, sems):
        acc[...] = jnp.zeros(acc.shape, jnp.int32) if verify else carry[...]

        def source(i):
            code = codes[0, i]
            block = table[0, (code >> 16) * per_row + ((code >> 8) & 255)]
            if variant == "page":
                return pool.at[0, block]
            first = code & 255
            if variant == "tile":
                first = pl.multiple_of(first, TILE_ROWS)
            return pool.at[0, block, pl.ds(first, slot_rows)]

        def start(i):
            slot = i & (slots - 1)
            pltpu.make_async_copy(
                source(i), ring.at[slot], sems.at[slot // group]
            ).start()

        def words(slot):
            return ring.at[slot].bitcast(jnp.int32)

        def consume(i):
            row = codes[0, i] >> 16
            view = words(i & (slots - 1))
            total = jnp.zeros((1, LATENT_LANES), jnp.int32)
            for first in range(0, slot_rows, TILE_ROWS):
                part = view[pl.ds(first, min(TILE_ROWS, slot_rows))]
                position = first + jax.lax.broadcasted_iota(jnp.int32, part.shape, 0)
                total += jnp.sum(part * (2 * position + 1), axis=0)
            acc[0, pl.ds(row, 1), :] += (2 * i + 1) * total

        def start_chunk(chunk, size):
            for k in range(size):
                start(chunk * group + k)

        def finish_chunk(chunk, size):
            g = chunk & (groups - 1)
            span = ring.at[pl.ds(g * group, size)]
            pltpu.make_async_copy(span, span, sems.at[g]).wait()
            if verify:
                for k in range(size):
                    consume(chunk * group + k)

        head = min(groups, chunks)

        @pl.loop(0, head)
        def _(c):
            start_chunk(c, group)

        @pl.loop(head, chunks)
        def _(c):
            finish_chunk(c - groups, group)
            start_chunk(c, group)

        if tail:
            if chunks >= groups:
                finish_chunk(chunks - groups, group)
            start_chunk(chunks, tail)

        @pl.loop(max(0, chunks + (tail > 0) - groups), chunks)
        def _(c):
            finish_chunk(c, group)

        if tail:
            finish_chunk(chunks, tail)

        if not verify:
            acc[0, pl.ds(0, 1), pl.ds(0, LANES)] += words((dmas - 1) & (slots - 1))[
                0, :, pl.ds(0, LANES)
            ]

    return kernel, scratch


def _gather_body(variant, geometry, per_row, dmas, slots, group, verify, tag):
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    kernel, scratch = _gather_kernel(
        variant, geometry, per_row, dmas, slots, group, verify
    )

    def body(table, codes, pool, carry, iterations):
        call = pl.pallas_call(
            kernel,
            name=f"dsa_micro_gather_{tag}",
            out_shape=jax.ShapeDtypeStruct(carry.shape, carry.dtype),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=scratch,
            input_output_aliases={3: 0},
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=GATHER_VMEM_LIMIT),
        )
        # The aliased carry orders the calls; each reads it and adds one word.
        return (
            jax.lax.fori_loop(
                0, iterations, lambda _, c: call(table, codes, pool, c), carry
            ),
        )

    return body


def _gather_setup(
    rows, tokens, variant, pattern, count, kv_dtype, inflight, group, args
):
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel import bridge

    geometry = GEOMETRY[kv_dtype]
    pool = _gather_pool(kv_dtype, tokens, max(args.rows))
    per_row = tokens // geometry.block_tokens
    units = [
        gather_units(selected, variant, geometry)
        for selected in _gather_selection(rows, tokens, count, pattern)
    ]
    codes = dma_codes(units)
    dmas = len(codes)
    slots = ring_slots(dmas, inflight)
    slot_rows = GATHER_SLOT_ROWS[variant]
    tag = f"{variant}_{pattern}_s{count}_{kv_dtype}"
    tag += f"_i{inflight}" if inflight != GATHER_INFLIGHT else ""
    if group < 1 or group & (group - 1):
        raise ValueError(f"gather: wait group {group} must be a power of two")
    group = min(group, slots)
    tag += f"_g{group}" if group != 1 else ""
    rows_padded = padded_rows(rows)
    operands = (
        bridge.upload(smem_table(pool.table.reshape(-1))),
        bridge.upload(smem_table(codes)),
        pool.device,
        bridge.upload(np.zeros((rows_padded, LATENT_LANES), np.int32)),
    )

    def build(verify):
        return _gather_body(variant, geometry, per_row, dmas, slots, group, verify, tag)

    def check(outputs):
        wanted = gather_accumulators(
            pool.bits, pool.table, units, slot_rows, rows_padded, geometry
        )
        got = _first(outputs)[0].cpu().numpy()
        bad = np.flatnonzero((got != wanted).any(axis=1))
        harness.log(
            f"gather_r{rows}_t{tokens}_{tag} check "
            + ("PASS" if not bad.size else f"FAIL (rows {bad.tolist()[:8]})")
            + f" ({dmas} DMAs, {slots} slots)"
        )
        return not bad.size

    return Setup(
        build(False),
        operands,
        (P(AXIS),) * 4,
        (P(AXIS),),
        variant=tag,
        check=check if args.verify == "on" else None,
        check_body=build(True) if args.verify == "on" else None,
        extra={
            "granularity": variant,
            "selected": count,
            "pattern": pattern,
            "kv_dtype": kv_dtype,
            "inflight": inflight,
            "wait_group": group,
            "slots": slots,
            "dmas": dmas,
            "moved_bytes": dmas * slot_rows * geometry.row_bytes,
            "useful_bytes": rows * count * geometry.token_bytes,
        },
    )


def _build_gather(rows: int, tokens: int, args) -> list[Setup]:
    """The distinct gather setups at one point over the flags' grid. The variant
    tag is ``{row|tile|page}_{uniform|recent}_s{selected}_{bf16|fp8}`` plus
    ``_i{N}`` (``--inflight N`` other than 64) and ``_g{N}`` (a wait group other
    than 1, after capping it to the ring)."""
    variants = [v for v in args.variant if v in GATHER_VARIANTS] or list(
        GATHER_VARIANTS
    )
    setups = {}
    for kv_dtype in args.kv_dtype:
        for pattern in args.pattern:
            for count in args.selected:
                for variant in variants:
                    for inflight in args.inflight:
                        for group in args.wait_group:
                            setup = _gather_setup(
                                rows,
                                tokens,
                                variant,
                                pattern,
                                count,
                                kv_dtype,
                                inflight,
                                group,
                                args,
                            )
                            # A wait group wider than the ring is the ring.
                            setups.setdefault(setup.variant, setup)
    return list(setups.values())


def _smem_move(stage, smem_slot, bounce_slot, sem, path, start=True):
    """The DMA that lands the VMEM tile ``stage`` in ``smem_slot``: ``dma``
    copies it directly, ``hbm`` through ``bounce_slot`` in HBM (that first hop
    is started and waited here). Started when ``start`` is set; either way the
    last hop's descriptor is returned for the caller's wait, so a later loop
    iteration rebuilds it with ``start=False``."""
    from jax.experimental.pallas import tpu as pltpu

    source = stage
    if path == "hbm":
        if start:
            hop = pltpu.make_async_copy(stage, bounce_slot, sem)
            hop.start()
            hop.wait()
        source = bounce_slot
    move = pltpu.make_async_copy(source, smem_slot, sem)
    if start:
        move.start()
    return move


def _compact_kernel(rows, tokens, variant, geometry, capacity, smem_path, full):
    """The compact ``pallas_call`` kernel and its scratch shapes; its algorithm
    is :func:`compact_model`'s. The mask words are DMA'd into VMEM at entry.
    Per block of eight virtual rows (a ``fori_loop``): phases 1-2 store each
    plane's fields lane-broadcast in ``[fields, planes, 8, 128]`` (unit words,
    four token words, unit count, base), phase 3 scans them per list position,
    phase 4 sums the page flags over the row's parts; the codes, bitmasks and
    counts (lanes 0-3) go into the staging tile ``[3, 8, C]``, which moves to
    SMEM slot ``block % 2`` (:func:`_smem_move`). A block waits for the
    previous block's move before it overwrites the staging tile. ``full``
    copies every block's SMEM chunk to ``out[block]`` (the checked body);
    otherwise only the last block's."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels.dsa_select import _group_sum

    srl = jax.lax.shift_right_logical
    groups = virtual_groups(rows, tokens)
    vrows, part = rows * groups, tokens // groups
    planes = part // LANES
    blocks = vrows // SUBLANES
    size = unit_tokens(variant, geometry)
    row_lanes, tile_lanes, first_lane, units = _compact_lanes(variant, geometry)
    word_lane = row_lanes + tile_lanes
    unit_words = -(-units // 32)
    count_field, base_field = unit_words + 4, unit_words + 5
    page_shift = geometry.block_tokens.bit_length() - 1
    out_vregs = capacity // LANES
    fields_shape = (unit_words + 6, planes if units else 1, SUBLANES, LANES)
    scratch = [
        pltpu.VMEM((-(-planes // 4), vrows, LANES), jnp.int32),  # mask words
        pltpu.VMEM(fields_shape, jnp.int32),  # per-plane fields
        pltpu.VMEM((3, SUBLANES, capacity), jnp.int32),  # staging tile
        pltpu.SMEM((2, 3, SUBLANES, capacity), jnp.int32),
        pltpu.SemaphoreType.DMA((4,)),
    ]

    def row_sum(x):
        return jnp.sum(x, axis=1, keepdims=True)

    def broadcast(x):
        return jnp.broadcast_to(row_sum(x), x.shape)

    def paired(x, lane0, lanes):
        """Lanes ``lane0`` and ``lane0 + 1`` of ``x`` (16-bit halves) as one
        32-bit word, on every lane."""
        high = jnp.where(lanes == lane0 + 1, jnp.left_shift(x, 16), 0)
        return broadcast(jnp.where(lanes == lane0, x, high))

    def kernel(words_hbm, *refs):
        if smem_path == "hbm":
            _, _, out, bounce, mask, fields, stage, smem, sems = refs
        else:
            _, out, mask, fields, stage, smem, sems = refs
            bounce = None
        load = pltpu.make_async_copy(words_hbm.at[0], mask, sems.at[0])
        load.start()
        index = (
            jax.lax.broadcasted_iota(jnp.int32, (LANES, LANES), 0),
            jax.lax.broadcasted_iota(jnp.int32, (LANES, LANES), 1),
        )
        unit_matrix, bit_matrix = (
            m.astype(jnp.float32).astype(jnp.bfloat16)
            for m in _compact_selectors(*index, variant, geometry, jnp)
        )
        lane = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 1)
        part_of = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0) % groups
        first_part = part_of == 0
        zero = jnp.zeros((SUBLANES, LANES), jnp.int32)

        def dot(x, matrix):
            # 0/1 masks become BF16 only as dot operands.
            return jnp.dot(
                x.astype(jnp.bfloat16), matrix, preferred_element_type=jnp.float32
            ).astype(jnp.int32)

        def move(b, start):
            slot = b % 2
            return _smem_move(
                stage,
                smem.at[slot],
                None if bounce is None else bounce.at[0, slot],
                sems.at[1 + slot],
                smem_path,
                start,
            )

        def drain(b):
            copy = pltpu.make_async_copy(smem.at[b % 2], out.at[0, b], sems.at[3])
            copy.start()
            copy.wait()

        def planes_of_word(w, n, carry, rows8):
            """Phases 1-2 for planes ``4 w .. 4 w + n - 1``, one dot for all."""
            tallies, pages, running = carry
            word = mask[w, rows8, :]
            m = jnp.concatenate([srl(word, 8 * i) & 255 for i in range(n)], axis=0)
            lanes = jax.lax.broadcasted_iota(jnp.int32, m.shape, 1)
            y = dot(m, unit_matrix)
            hit = y > 0
            row_hit = jnp.where(hit & (lanes < row_lanes), 1, 0)
            in_tiles = (lanes >= row_lanes) & (lanes < word_lane)
            tile_hit = jnp.where(hit & in_tiles, 1, 0)
            if units:
                in_units = (lanes >= first_lane) & (lanes < first_lane + units)
                flags = jnp.where(hit & in_units, 1, 0)
                z = dot(flags, bit_matrix)
                plane_fields = [
                    *(paired(z, 2 * k, lanes) for k in range(unit_words)),
                    *(paired(y, word_lane + 2 * k, lanes) for k in range(4)),
                ]
                count = broadcast(flags)
                touched = count > 0
            else:
                touched = broadcast(m) > 0
            for i in range(n):
                s = 4 * w + i
                tile = slice(SUBLANES * i, SUBLANES * (i + 1))
                tallies = (
                    tallies[0] + m[tile],
                    tallies[1] + row_hit[tile],
                    tallies[2] + tile_hit[tile],
                )
                page = srl(part_of * part + LANES * s, page_shift)
                pages = jnp.where((lane == page) & touched[tile], 1, pages)
                if units:
                    for f, value in enumerate(plane_fields):
                        fields[f, s] = value[tile]
                    fields[count_field, s] = count[tile]
                    fields[base_field, s] = running
                    running = running + count[tile]
            return tallies, pages, running

        def list_positions(total):
            """Phase 3: each list position's unit code and token bits."""
            codes, bitmasks = [], []
            for q in range(out_vregs):
                position = q * LANES + lane

                def scan(s, acc, position=position):
                    plane, base, *picked = acc
                    count, start = fields[count_field, s], fields[base_field, s]
                    inside = (start <= position) & (position < start + count)
                    plane = plane + jnp.where(start + count <= position, 1, 0)
                    base = jnp.where(inside, start, base)
                    picked = [
                        jnp.where(inside, fields[f, s], a) for f, a in enumerate(picked)
                    ]
                    return (plane, base, *picked)

                plane, base, *picked = jax.lax.fori_loop(
                    0,
                    planes,
                    scan,
                    (zero,) * (2 + unit_words + 4),
                    unroll=COMPACT_UNROLL,
                )
                unit = _rank_select(picked[:unit_words], position - base, srl, jnp)
                token = part_of * part + (plane * (LANES // size) + unit) * size
                valid = position < total
                codes.append(jnp.where(valid, _unit_code(token, geometry), 0))
                bitmask = _unit_bitmask(picked[unit_words:], unit, size, srl, jnp)
                bitmasks.append(jnp.where(valid, bitmask, 0))
            return codes, bitmasks

        def page_list(flags, row_pages):
            """Phase 4's list: the row's flagged pages, on its first part."""
            page_lanes = dot(flags, unit_matrix)
            page_words = [paired(page_lanes, word_lane + 2 * k, lane) for k in range(4)]
            codes = []
            for q in range(out_vregs):
                position = q * LANES + lane
                page = _rank_select(page_words, position, srl, jnp)
                code = _unit_code(jnp.left_shift(page, page_shift), geometry)
                listed = first_part & (position < row_pages)
                codes.append(jnp.where(listed, code, 0))
            return codes, [zero] * out_vregs

        load.wait()

        def per_block(b, _):
            rows8 = pl.ds(pl.multiple_of(SUBLANES * b, SUBLANES), SUBLANES)
            carry = ((zero,) * 3, zero, zero)
            full_words, tail = divmod(planes, 4)
            # Unrolled so that consecutive words' dependent dot -> compare ->
            # dot -> lane-sum chains overlap.
            carry = jax.lax.fori_loop(
                0,
                full_words,
                lambda w, c: planes_of_word(w, 4, c, rows8),
                carry,
                unroll=COMPACT_UNROLL,
            )
            if tail:
                carry = planes_of_word(full_words, tail, carry, rows8)
            tallies, pages, total = carry
            if groups > 1:
                pages = _group_sum(pages, groups)
            flags = jnp.where(pages > 0, 1, 0)
            row_pages = broadcast(flags)
            if units:
                codes, bitmasks = list_positions(total)
            else:
                codes, bitmasks = page_list(flags, row_pages)
            selected, sublane_rows, tiles = (broadcast(t) for t in tallies)
            counts = jnp.where(first_part & (lane == 3), row_pages, 0)
            counts = jnp.where(lane == 2, tiles, counts)
            counts = jnp.where(lane == 1, sublane_rows, counts)
            counts = jnp.where(lane == 0, selected, counts)

            @pl.when(b > 0)
            def _():
                move(b - 1, False).wait()
                if full:
                    drain(b - 1)

            for q in range(out_vregs):
                lanes = pl.ds(q * LANES, LANES)
                stage[0, :, lanes] = codes[q]
                stage[1, :, lanes] = bitmasks[q]
                stage[2, :, lanes] = counts if q == 0 else zero
            move(b, True)

        jax.lax.fori_loop(0, blocks, per_block, None)
        move(blocks - 1, False).wait()
        drain(blocks - 1)

    return kernel, scratch


def _compact_selection(rows, tokens, count, pattern, row_mix):
    """Each row's selected local tokens (:func:`row_mix_counts` of them),
    different on every rank."""
    from vllm_torchtpu.megakernel import bridge

    rank = bridge.kernel_mesh(AXIS, RANKS)[1]
    return row_selections(
        [COMPACT_SEED, tokens, count, GATHER_PATTERNS.index(pattern)],
        row_mix_counts(count, tokens, rows, row_mix),
        tokens,
        pattern,
        suffix=(rank,),
    )


def _compact_setup(
    rows, tokens, variant, pattern, count, kv_dtype, capacity, smem_path, args
) -> Setup:
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel import bridge

    geometry = GEOMETRY[kv_dtype]
    if not 0 < count <= tokens:
        raise ValueError(f"compact: cannot select {count} of {tokens} tokens")
    if capacity < LANES or capacity % LANES or capacity > 4 * LANES:
        raise ValueError(
            f"compact: capacity {capacity} must be a multiple of {LANES} up to "
            f"{4 * LANES}"
        )
    groups = virtual_groups(rows, tokens)
    if tokens > LANES * geometry.block_tokens:
        raise ValueError(
            f"compact: {tokens} local tokens are more than {LANES} {kv_dtype} pages"
        )
    row_mix = args.row_mix == "on"
    tag = f"{variant}_{pattern}_s{count}_{kv_dtype}"
    tag += f"_c{capacity}" if capacity != COMPACT_CAPACITY else ""
    tag += "_hbm" if smem_path == "hbm" else ""
    tag += "_mix" if row_mix else ""
    selections = _compact_selection(rows, tokens, count, pattern, row_mix)
    mask = np.zeros((rows, tokens), bool)
    for r, selected in enumerate(selections):
        mask[r, selected] = True
    expected = compact_expected(selections, rows, tokens, variant, geometry, capacity)
    vrows = rows * groups
    units = expected.counts[:, 1 + GATHER_VARIANTS.index(variant)].astype(np.int64)
    listed, overflow = (
        int(np.minimum(units, capacity).sum()),
        int((units > capacity).sum()),
    )
    carry = [np.zeros((vrows // SUBLANES, 3, SUBLANES, capacity), np.int32)]
    if smem_path == "hbm":
        carry.append(np.zeros((2, 3, SUBLANES, capacity), np.int32))

    def build(full):
        kernel, scratch = _compact_kernel(
            rows, tokens, variant, geometry, capacity, smem_path, full
        )
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)

        def body(words, *state):
            *carried, iterations = state
            call = pl.pallas_call(
                kernel,
                name=f"dsa_micro_compact_{tag}" + ("_full" if full else ""),
                out_shape=tuple(
                    jax.ShapeDtypeStruct(c.shape, c.dtype) for c in carried
                ),
                in_specs=[hbm] * (1 + len(carried)),
                out_specs=tuple(hbm for _ in carried),
                scratch_shapes=scratch,
                input_output_aliases={1 + i: i for i in range(len(carried))},
                compiler_params=pltpu.CompilerParams(
                    vmem_limit_bytes=COMPACT_VMEM_LIMIT
                ),
            )
            # The aliased outputs order the calls; each call only overwrites them.
            return tuple(
                jax.lax.fori_loop(
                    0, iterations, lambda _, c: tuple(call(words, *c)), tuple(carried)
                )
            )

        return body

    padded_counts = np.zeros((vrows, capacity), np.int32)
    padded_counts[:, : len(COUNT_FIELDS)] = expected.counts
    wanted = {
        "codes": expected.codes,
        "bitmasks": expected.bitmasks,
        "counts": padded_counts,
    }

    def check(outputs):
        got = _first(outputs)[0].cpu().numpy()
        lists = got.transpose(1, 0, 2, 3).reshape(3, vrows, capacity)
        bad = {
            name: np.flatnonzero((got_list != want).any(axis=1))
            for got_list, (name, want) in zip(lists, wanted.items())
        }
        failed = [name for name, at in bad.items() if at.size]
        first_rows = sorted({int(v) for name in failed for v in bad[name]})[:8]
        harness.log(
            f"compact_r{rows}_t{tokens}_{tag} check "
            + (
                f"FAIL ({', '.join(failed)}; virtual rows {first_rows})"
                if failed
                else "PASS"
            )
            + f" ({listed} codes, {overflow} lists over capacity)"
        )
        return not failed

    return Setup(
        build(False),
        (
            bridge.upload(pack_mask(mask, rows, tokens)),
            *(bridge.upload(c) for c in carry),
        ),
        (P(AXIS),) * (1 + len(carry)),
        (P(AXIS),) * len(carry),
        variant=tag,
        check=check,
        check_body=build(True),
        extra={
            "granularity": variant,
            "selected": count,
            "pattern": pattern,
            "kv_dtype": kv_dtype,
            "capacity": capacity,
            "smem_path": smem_path,
            "row_mix": args.row_mix,
            "codes": listed,
            "overflow": overflow,
        },
    )


def _build_compact(rows: int, tokens: int, args) -> list[Setup]:
    """The distinct compact setups at one point, over ``--kv-dtype`` x
    ``--pattern`` x ``--selected`` x ``--variant`` (row, tile, page: the lists'
    unit) x ``--capacity`` x ``--smem-path``. The variant tag is
    ``{row|tile|page}_{uniform|recent}_s{S}_{bf16|fp8}`` plus ``_c{C}``
    (``--capacity C`` other than 128), ``_hbm`` (``--smem-path hbm``) and
    ``_mix`` (``--row-mix on``)."""
    variants = [v for v in args.variant if v in GATHER_VARIANTS] or list(
        GATHER_VARIANTS
    )
    setups = {}
    for kv_dtype in args.kv_dtype:
        for pattern in args.pattern:
            for count in args.selected:
                for variant in variants:
                    for capacity in args.capacity:
                        for smem_path in args.smem_path:
                            setup = _compact_setup(
                                rows,
                                tokens,
                                variant,
                                pattern,
                                count,
                                kv_dtype,
                                capacity,
                                smem_path,
                                args,
                            )
                            setups.setdefault(setup.variant, setup)
    return list(setups.values())


def absorbed_queries(q_nope, q_rope, w_uk):
    """``[R, 96, 640]`` FP32 absorbed queries as the decode kernel attends with
    them: ``[q_nope @ W_UK^T | q_rope | 0] * 192^-0.5`` (``dsa_reference.
    sparse_mla_attention``'s query), from ``q_nope [R, 96, 128]``, ``q_rope
    [R, 96, 64]`` and ``w_uk [96, 512, 128]``."""
    absorbed = np.einsum(
        "rhd,hld->rhl", q_nope.astype(np.float64), w_uk.astype(np.float64)
    )
    out = np.zeros((*q_nope.shape[:2], LATENT_LANES), np.float64)
    out[..., :512], out[..., 512:576] = absorbed, q_rope
    return (out * (128 + 64) ** -0.5).astype(np.float32)


def fp8_query_packet(query):
    """``[H, 640]`` uint8: the decode kernel's FP8 query wire (``mla_dcp.
    _pack_fp8_query``) of FP32 queries: BF16-rounded values over a per-head
    scale ``amax / 448`` as E4M3 bytes in lanes 0:576, the FP32 scale's four
    bytes repeated over lanes 576:640."""
    import ml_dtypes

    q = np.asarray(query, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)
    amax = np.abs(q).max(axis=1, keepdims=True)
    scale = np.where(amax > 0, amax / np.float32(448.0), np.float32(1.0)).astype(
        np.float32
    )
    out = (q / scale).astype(ml_dtypes.float8_e4m3fn).view(np.uint8).copy()
    out[:, 576:] = np.tile(scale.view(np.uint8).reshape(-1, 4), 16)
    return out


def fold_query_values(query, kv_dtype: str):
    """``(values [H, 640], scale [H, 1])`` FP32: what the fold kernel's QK dot
    takes of the FP32 ``query``: its BF16 rounding with scale one, or the
    :func:`fp8_query_packet` codes (lanes 576:640 cleared) and scale."""
    import ml_dtypes

    query = np.asarray(query, np.float32)
    if kv_dtype == "bf16":
        values = query.astype(ml_dtypes.bfloat16).astype(np.float32)
        return values, np.ones((len(query), 1), np.float32)
    packet = fp8_query_packet(query)
    values = packet.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    values[:, 576:] = 0
    return values, np.ascontiguousarray(packet[:, 576:580]).view(np.float32)


def fold_reference(q_nope, q_rope, w_uk, latents, selected):
    """``(o [96, 512], lse [96])`` in FP64: ``dsa_reference.sparse_mla_attention``
    over exactly ``selected`` (its last token passed as the always-included new
    row), ``lse`` the log-sum-exp of the same absorbed logits; ``(0, NEG)`` when
    nothing is selected. ``latents [T, 640]`` FP32 values of the cache dtype."""
    import jax
    import jax.numpy as jnp

    from tools.megakernel import dsa_reference as ref
    from vllm_torchtpu.megakernel.kimi_k3.kernels.mla_dcp import NEG as NEG_LSE

    if not len(selected):
        return np.zeros((FOLD_HEADS, 512)), np.full(FOLD_HEADS, NEG_LSE)
    with jax.default_device(jax.devices("cpu")[0]):
        o = ref.sparse_mla_attention(
            jnp.asarray(q_nope),
            jnp.asarray(q_rope),
            jnp.asarray(w_uk),
            jnp.asarray(latents),
            jnp.asarray(selected[:-1], jnp.int32),
            jnp.asarray(latents[selected[-1]]),
        )
    query = absorbed_queries(q_nope[None], q_rope[None], w_uk)[0].astype(np.float64)
    logits = query[:, :576] @ latents[selected, :576].astype(np.float64).T
    top = logits.max(axis=1, keepdims=True)
    lse = top[:, 0] + np.log(np.exp(logits - top).sum(axis=1))
    return np.asarray(o, np.float64), lse


def fold_model(query, scale, stream, mask, step, kv_dtype, *, rescale=True):
    """``(o [96, 512], lse [96])``: K3's fold arithmetic (``mla_dcp.
    DcpAttention._scores``, ``_fold_part``, ``_pack_partials``) over one row's
    token stream, ``step`` tokens a step. ``query [96, 640]`` FP32 holds the
    kernel's query values (:func:`fold_query_values`) and ``scale [96, 1]``
    the FP8 query scale (ones for BF16); ``stream [N, 640]`` FP32 the
    stream's latent values, lanes 576:640 zero; ``mask [N]`` the tokens
    folded. Per step: ``s = fp32(q c^T) scale`` (``-inf`` where masked), the
    online softmax in FP32, ``P`` BF16, or E4M3 of ``256 p`` (and ``1 / 256``
    after ``P c``); then ``o = O / l``, ``lse = m + log l``, or ``(0, NEG)``
    with nothing folded. Casts round to nearest even (``ml_dtypes``).
    ``rescale=False`` never rescales the running sums (a wrong fold, for
    negative controls)."""
    import ml_dtypes

    from vllm_torchtpu.megakernel.kimi_k3.kernels.mla_dcp import LATENT, NEG

    query = np.asarray(query, np.float64)
    scale = np.asarray(scale, np.float32).reshape(-1, 1)
    stream = np.asarray(stream, np.float64)
    mask = np.asarray(mask, bool)
    heads = len(query)
    maximum = np.full((heads, 1), -np.inf, np.float32)
    total = np.zeros((heads, 1), np.float32)
    acc = np.zeros((heads, LATENT), np.float32)
    for first in range(0, len(stream), step):
        tokens = stream[first : first + step]
        score = (query @ tokens.T).astype(np.float32) * scale
        score = np.where(mask[None, first : first + step], score, -np.inf)
        new = np.maximum(maximum, score.max(axis=1, keepdims=True))
        alpha = np.exp(maximum - new) if rescale else np.ones_like(new)
        p = np.exp(score - new)
        total = total * alpha + p.sum(axis=1, keepdims=True, dtype=np.float32)
        maximum = new
        if kv_dtype == "fp8":
            high = (p * np.float32(256)).astype(ml_dtypes.float8_e4m3fn)
            weighted = (high.astype(np.float64) @ tokens[:, :LATENT]).astype(
                np.float32
            ) * np.float32(1 / 256)
        else:
            high = p.astype(ml_dtypes.bfloat16).astype(np.float64)
            weighted = (high @ tokens[:, :LATENT]).astype(np.float32)
        acc = acc * alpha + weighted
    has = total > 0
    safe = np.where(has, total, np.float32(1))
    o = np.where(has, acc / safe, np.float32(0))
    lse = np.where(has, maximum + np.log(safe), np.float32(NEG))
    return o, lse[:, 0]


def fold_errors(o, lse, o_ref, lse_ref):
    """Per row (the last two axes of ``o``, the last of ``lse``), in FP64:
    ``(max |o - o_ref| / max |o_ref|, max |lse - lse_ref|)``."""
    o, o_ref = np.asarray(o, np.float64), np.asarray(o_ref, np.float64)
    lse, lse_ref = np.asarray(lse, np.float64), np.asarray(lse_ref, np.float64)
    return (
        np.abs(o - o_ref).max(axis=(-2, -1)) / np.abs(o_ref).max(axis=(-2, -1)),
        np.abs(lse - lse_ref).max(axis=-1),
    )


def fold_exact_lse(query, scale, latents):
    """``[H]`` FP64: the log-sum-exp over every row of ``latents [n, 640]``
    (cache-dtype values) of the logits ``(query c^T) scale`` from the
    kernel's query values and scale (:func:`fold_query_values`). No
    probability rounding enters ``lse``, so :func:`fold_model` matches this
    to FP32 accuracy."""
    query = np.asarray(query, np.float64)[:, :576]
    latents = np.asarray(latents, np.float64)[:, :576]
    logits = (query @ latents.T) * np.asarray(scale, np.float64).reshape(-1, 1)
    top = logits.max(axis=1, keepdims=True)
    return top[:, 0] + np.log(np.exp(logits - top).sum(axis=1))


def fold_exact_lse_tolerance(kv_dtype: str) -> float:
    """The exact-lse bound for the kernel where it runs: on TPU, or
    interpreted (no TPU backend: a Pallas TPU kernel runs only interpreted
    there)."""
    import jax

    on_tpu = jax.default_backend() == "tpu"
    return (FOLD_EXACT_LSE_TOLERANCE_TPU if on_tpu else FOLD_EXACT_LSE_TOLERANCE)[
        kv_dtype
    ]


def latent_values(bits, table, row: int, tokens, kv_dtype: str):
    """``[n, 640]`` FP32: the cache-dtype values of row ``row``'s local
    ``tokens`` in the pool (``bits`` and ``table`` of :func:`_pool_bits`);
    token -1 (padding) is zero."""
    import ml_dtypes

    geometry = GEOMETRY[kv_dtype]
    dtype = {"bf16": ml_dtypes.bfloat16, "fp8": ml_dtypes.float8_e4m3fn}[kv_dtype]
    tokens = np.asarray(tokens, np.int64)
    out = np.zeros((len(tokens), LATENT_LANES), np.float32)
    real = tokens >= 0
    block, sublane, slot = locate(tokens[real], geometry)
    out[real] = bits[table[row][block], sublane, slot].view(dtype).astype(np.float32)
    return out


def fold_stream(selected, variant: str, geometry: PoolGeometry, fold_tokens: int):
    """One row's token stream through the fold, as local tokens padded with -1
    to whole ``fold_tokens`` steps: every token of each distinct sublane row
    (``row``), tile (``tile``) or block (``page``) holding a selected token,
    ascending, or the selected tokens (``dense``)."""
    size = 1 if variant == "dense" else unit_tokens(variant, geometry)
    units = np.unique(np.asarray(selected, np.int64) // size)
    stream = (units[:, None] * size + np.arange(size)).reshape(-1)
    padded = np.full(-(-len(stream) // fold_tokens) * fold_tokens, -1, np.int64)
    padded[: len(stream)] = stream
    return padded


class FoldTablesTooLarge(ValueError):
    """A fold point whose SMEM tables exceed ``FOLD_SMEM_WORDS``."""


@dataclass
class FoldStage:
    """A rank's fold inputs (:func:`fold_stage`): ``words`` int32 ``[n, 640]``
    (the rows' token streams back to back, ``packing`` tokens a word), the
    SMEM tables ``meta`` (``META_FIELDS`` int32 per step), ``masks`` and
    ``copy_tokens``, each row's ``streams`` (local tokens, -1 padding) and
    ``selections``, the step and chunk counts and ``folded_tokens``, the
    tokens the fold works on (padding included)."""

    words: Any
    meta: Any
    masks: Any
    copy_tokens: Any
    streams: list
    selections: list
    steps: int
    chunks: int
    folded_tokens: int

    @property
    def smem_words(self) -> int:
        return len(self.meta) + len(self.masks) + len(self.copy_tokens)


def _check_fold_tokens(fold_tokens: int) -> None:
    if fold_tokens < LANES or fold_tokens % LANES:
        raise ValueError(
            f"fold: {fold_tokens} tokens a step must be a positive multiple of {LANES}"
        )


def _stage_words(streams, bits, table, kv_dtype: str):
    """``[n / packing, 640]`` int32: the streams' tokens, stream token ``i``
    in word row ``i // packing`` from bit ``8 itemsize (i % packing)``; lanes
    576:640 hold NaN bits (the kernel zeroes them), padding is zero."""
    geometry = GEOMETRY[kv_dtype]
    tokens = np.concatenate(streams) if streams else np.zeros(0, np.int64)
    if not len(tokens):
        return np.zeros((SUBLANES, LATENT_LANES), np.int32)
    row = np.concatenate([np.full(len(s), r) for r, s in enumerate(streams)])
    staged = np.zeros((len(tokens), LATENT_LANES), bits.dtype)
    real = tokens >= 0
    block, sublane, slot = locate(tokens[real], geometry)
    staged[real] = bits[table[row[real], block], sublane, slot]
    staged[real, 576:] = NAN_BITS[kv_dtype]
    packed = staged.reshape(-1, geometry.packing, LATENT_LANES).transpose(0, 2, 1)
    return np.ascontiguousarray(packed).view(np.int32).reshape(-1, LATENT_LANES)


def fold_stage(
    rows,
    tokens,
    count,
    pattern,
    kv_dtype,
    variant,
    fold_tokens,
    compaction,
    row_mix,
    bits,
    table,
) -> FoldStage:
    """The fold's HBM staging and SMEM tables (the same on every rank but for
    the pool ``bits``). Row ``r`` selects :func:`row_mix_counts` local tokens
    (:func:`row_selections`, seeded by T, S, the pattern and r); ``none``
    selects none. Its stream is :func:`fold_stream`. Steps, ``META_FIELDS``
    fields each (row, ``STEP_*`` flags, chunk or -1, mask-word offset or
    dense step, selected or valid tokens, ``copy_tokens`` offset):

    * a FOLD step per ``fold_tokens`` chunk (each holds a selected token,
      padding only in a row's last);
    * with a compaction (``tile``, ``page``): a COMPACT step per chunk, then
      ``ceil(S_r / N)`` DENSE steps over the compacted buffer;
    * a row with nothing selected: one EMPTY step.

    A row's first step is FIRST, its last LAST. ``masks``: ``N / 32`` words
    per FOLD or COMPACT step, bit ``b`` of word ``w`` its token ``32 w + b``;
    ``copy_tokens``: each COMPACT step's selected tokens' offsets in its chunk
    (``copy``)."""
    geometry = GEOMETRY[kv_dtype]
    _check_fold_tokens(fold_tokens)
    if variant not in FOLD_VARIANTS or compaction not in FOLD_COMPACTIONS:
        raise ValueError(f"fold: unknown variant {variant!r} or compaction")
    if variant == "page" and fold_tokens % geometry.block_tokens:
        raise ValueError(
            f"fold: page steps of {fold_tokens} tokens must hold whole {kv_dtype} "
            f"pages of {geometry.block_tokens}"
        )
    compact = compaction != "none"
    if compact and variant not in ("tile", "page"):
        raise ValueError(f"fold: {variant} streams take no compaction")
    if variant == "none":
        selections = [np.zeros(0, np.int64)] * rows
    else:
        selections = row_selections(
            [FOLD_SEED, tokens, count, GATHER_PATTERNS.index(pattern)],
            row_mix_counts(count, tokens, rows, row_mix),
            tokens,
            pattern,
        )
    if compact and max(map(len, selections)) > FOLD_DENSE_TOKENS:
        raise ValueError(
            f"fold: the dense buffer holds at most {FOLD_DENSE_TOKENS} tokens a row"
        )
    meta, masks, copies, streams = [], [], [], []
    chunks = folded = 0
    bit = np.int64(1) << np.arange(WORD_TOKENS, dtype=np.int64)
    for r, selected in enumerate(selections):
        if not len(selected):
            streams.append(np.zeros(0, np.int64))
            meta.append([r, STEP_FIRST | STEP_LAST | STEP_EMPTY, -1, 0, 0, 0])
            continue
        stream = fold_stream(selected, variant, geometry, fold_tokens)
        streams.append(stream)
        chosen = np.isin(stream, selected).reshape(-1, fold_tokens)
        steps = []
        for part in chosen:
            picked = np.flatnonzero(part)
            if not compact:
                # K3's fold starts from m = -inf: an all-masked step is NaN.
                assert len(picked), "a fold step without a selected token"
            words = (part.reshape(-1, WORD_TOKENS) * bit).sum(axis=1)
            steps.append(
                [
                    r,
                    STEP_COMPACT if compact else STEP_FOLD,
                    chunks,
                    len(masks),
                    len(picked),
                    len(copies),
                ]
            )
            masks.extend(words.tolist())
            if compaction == "copy":
                copies.extend(picked.tolist())
            chunks += 1
        if compact:
            dense_steps = -(-len(selected) // fold_tokens)
            for j in range(dense_steps):
                valid = min(fold_tokens, len(selected) - fold_tokens * j)
                steps.append([r, STEP_DENSE, -1, j, valid, 0])
            folded += dense_steps * fold_tokens
        else:
            folded += len(stream)
        steps[0][1] |= STEP_FIRST
        steps[-1][1] |= STEP_LAST
        meta.extend(steps)
    stage = FoldStage(
        _stage_words(streams, bits, table, kv_dtype),
        _int32(np.asarray(meta, np.int64).reshape(-1)),
        _int32(masks if masks else [0]),
        _int32(copies if copies else [0]),
        streams,
        selections,
        len(meta),
        chunks,
        folded,
    )
    if stage.smem_words > FOLD_SMEM_WORDS:
        raise FoldTablesTooLarge(
            f"{stage.smem_words} words of SMEM tables, over the guard of "
            f"{FOLD_SMEM_WORDS}"
        )
    return stage


FOLD_POOLS: dict[tuple, tuple] = {}
FOLD_INPUTS: dict[int, tuple] = {}


def _fold_pool(kv_dtype: str, tokens: int, max_rows: int):
    """This rank's :func:`_pool_bits`, on the host only (the fold stages its
    streams instead of reading the pool)."""
    from vllm_torchtpu.megakernel import bridge

    key = (kv_dtype, tokens, max_rows)
    if key not in FOLD_POOLS:
        for old in [k for k in FOLD_POOLS if k[1:] != key[1:]]:
            del FOLD_POOLS[old]
        rank = bridge.kernel_mesh(AXIS, RANKS)[1]
        FOLD_POOLS[key] = _pool_bits(kv_dtype, tokens, max_rows, rank)
    return FOLD_POOLS[key]


def _fold_inputs(max_rows: int):
    """``(q_nope [R, 96, 128], q_rope [R, 96, 64], w_uk [96, 512, 128],
    queries [R, 96, 640])`` FP32 of this rank: random normal (``w_uk`` over
    sqrt(128), so the logits' standard deviation is about 1.7), row ``r`` the
    same for any ``R`` above it; ``queries`` are :func:`absorbed_queries`."""
    if max_rows not in FOLD_INPUTS:
        rng = _rank_rng(FOLD_SEED)
        w_uk = rng.standard_normal((FOLD_HEADS, 512, 128)) / np.sqrt(128)
        q_nope = np.zeros((max_rows, FOLD_HEADS, 128), np.float32)
        q_rope = np.zeros((max_rows, FOLD_HEADS, 64), np.float32)
        for r in range(max_rows):
            q_nope[r] = rng.standard_normal((FOLD_HEADS, 128))
            q_rope[r] = rng.standard_normal((FOLD_HEADS, 64))
        w_uk = w_uk.astype(np.float32)
        FOLD_INPUTS.clear()
        FOLD_INPUTS[max_rows] = (
            q_nope,
            q_rope,
            w_uk,
            absorbed_queries(q_nope, q_rope, w_uk),
        )
    return FOLD_INPUTS[max_rows]


def fold_tag(
    variant, pattern, count, kv_dtype, fold_tokens, compaction, compute, row_mix
):
    """The fold case's variant tag (see :func:`_build_fold`)."""
    if variant == "none":
        return f"none_{kv_dtype}"
    tag = f"{variant}_{pattern}_s{count}_{kv_dtype}_k{fold_tokens}"
    tag += f"_{compaction}" if compaction != "none" else ""
    tag += "_nocompute" if compute == "off" else ""
    return tag + ("_mix" if row_mix else "")


@dataclass(frozen=True)
class FoldPlan:
    """A fold kernel: ``rows`` rows in ``steps`` steps (:func:`fold_stage`),
    ``chunks`` chunks of ``fold_tokens`` staged tokens (``words_rows`` int32
    word rows), ``mask_words`` and ``copy_tokens`` SMEM entries, a dense
    buffer of ``dense_tokens`` tokens (0 without a compaction). ``compute``
    off only streams the chunks; ``full`` writes every row's result to HBM,
    else only the last row's."""

    rows: int
    kv_dtype: str
    variant: str
    fold_tokens: int
    compaction: str
    compute: bool
    steps: int
    chunks: int
    words_rows: int
    mask_words: int
    copy_tokens: int
    dense_tokens: int
    full: bool


def _fold_kernel(plan: FoldPlan):
    """The fold ``pallas_call`` kernel and its scratch shapes: K3's DCP attend
    arithmetic (``mla_dcp.DcpAttention._scores``, ``_fold_part``,
    ``_pack_partials``) over staged token streams, the same body for every
    form. A ``fori_loop`` runs the ``meta`` steps (SMEM, :func:`fold_stage`):

    * FIRST: wait for the row's query (two slots, the next row's started
      now), unpack the FP8 wire, reset ``m = -inf``, ``l = 0``, ``O = 0`` and
      the dense count.
    * FOLD, COMPACT: wait for the step's chunk (a ring of ``FOLD_DEPTH``
      slots, ``FOLD_DEPTH - 1`` ahead), start the one ``FOLD_DEPTH - 1``
      later into the slot the step before used, zero lanes 576:640 by a
      masked store and build the ``[1, N]`` lane mask from ``N / 32`` SMEM
      words. FOLD folds the chunk; COMPACT moves its selected tokens into the
      dense buffer, ``onehot`` by one-hot dots of 128 dense rows (positions
      from an exclusive prefix of the mask, a 0/1 dot with a triangular
      matrix), ``copy`` one token at a time from ``copy_tokens``.
    * DENSE: the fold over ``N`` tokens of the dense buffer.
    * LAST: ``o = O / l``, ``lse = m + log l`` (``(0, NEG)`` when ``l = 0``)
      into the result, ``o`` in lanes 0:512 and ``lse`` in lane 512, DMA'd to
      ``out[r]`` (every row when ``full``, else the last).

    EMPTY steps run FIRST and LAST only."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels.mla_dcp import LATENT, NEG

    geometry = GEOMETRY[plan.kv_dtype]
    fp8 = plan.kv_dtype == "fp8"
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    packing, n = geometry.packing, plan.fold_tokens
    chunk_rows = n // packing
    tile = TILE_ROWS * packing  # tokens of an aligned 8-word-row tile
    vregs = n // LANES
    pad_lanes = LATENT_LANES - LATENT
    compact = plan.compaction != "none"
    srl = jax.lax.shift_right_logical
    scratch = [
        pltpu.VMEM((FOLD_DEPTH, chunk_rows, LATENT_LANES), jnp.int32),  # ring
        pltpu.SemaphoreType.DMA((FOLD_DEPTH,)),
        pltpu.VMEM((2, FOLD_HEADS, LATENT_LANES), jnp.uint8 if fp8 else jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.VMEM((FOLD_HEADS, LATENT_LANES), dtype),  # the unpacked FP8 query
        pltpu.VMEM((FOLD_HEADS, 1), jnp.float32),  # its scale
        pltpu.VMEM((FOLD_HEADS, 1), jnp.float32),  # running maximum
        pltpu.VMEM((FOLD_HEADS, 1), jnp.float32),  # denominator
        pltpu.VMEM((FOLD_HEADS, LATENT), jnp.float32),  # accumulator
        pltpu.VMEM((FOLD_HEADS, LATENT_LANES), jnp.float32),  # row result
        pltpu.SemaphoreType.DMA,
    ]
    if compact:
        scratch += [
            pltpu.VMEM((plan.dense_tokens // packing, LATENT_LANES), jnp.int32),
            pltpu.SMEM((1,), jnp.int32),  # the row's dense count
        ]

    def kernel(
        meta,
        masks,
        copies,
        queries,
        words,
        _,
        out,
        ring,
        ring_sems,
        query_slots,
        query_sems,
        query_fp8,
        query_scale,
        maximum,
        denominator,
        accumulator,
        result,
        result_sem,
        *dense_refs,
    ):
        dense, count_ref = dense_refs if compact else (None, None)

        def field(i, f):
            return meta[0, META_FIELDS * i + f]

        def chunk_copy(c):
            slot = c % FOLD_DEPTH
            source = words.at[0, pl.ds(pl.multiple_of(c * chunk_rows, 8), chunk_rows)]
            return pltpu.make_async_copy(source, ring.at[slot], ring_sems.at[slot])

        def query_copy(r):
            slot = r % 2
            return pltpu.make_async_copy(
                queries.at[0, r], query_slots.at[slot], query_sems.at[slot]
            )

        def result_copy(r):
            return pltpu.make_async_copy(result, out.at[0, r], result_sem)

        for c in range(min(FOLD_DEPTH - 1, plan.chunks)):
            chunk_copy(c).start()
        query_copy(0).start()
        if compact:
            # Dense steps read the rows past a row's count under a zero
            # probability, and zero times a stale NaN is NaN.
            @pl.loop(0, plan.dense_tokens // tile)
            def _(t):
                dense[pl.ds(pl.multiple_of(t * TILE_ROWS, TILE_ROWS), TILE_ROWS)] = (
                    jnp.zeros((TILE_ROWS, LATENT_LANES), jnp.int32)
                )

        def loader(words_ref):
            def load(lanes):
                return pltpu.bitcast(words_ref[:, lanes], dtype)

            return load

        def lane_mask(offset):
            """The step's mask as ``N / 128`` ``[1, 128]`` int32 0/1 parts."""
            lane = jax.lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
            parts = []
            for v in range(vregs):
                bits = jnp.zeros((1, LANES), jnp.int32)
                for j in range(LANES // WORD_TOKENS):
                    word = masks[0, offset + 4 * v + j]
                    bits = jnp.where((lane >> 5) == j, word, bits)
                parts.append(srl(bits, lane & 31) & 1)
            return parts

        def query(r):
            return query_fp8[...] if fp8 else query_slots[r % 2]

        def fold(load, r, mask):
            """``_scores`` then ``_fold_part``; ``mask [1, N]`` marks the
            tokens folded."""
            score = jax.lax.dot_general(
                query(r),
                load(pl.ds(0, LATENT_LANES)),
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            if fp8:
                score = score * query_scale[...]
            score = jnp.where(mask, score, -jnp.inf)
            previous = maximum[...]
            new_maximum = jnp.maximum(previous, jnp.max(score, axis=1, keepdims=True))
            scale = jnp.exp(previous - new_maximum)
            probabilities = jnp.exp(score - new_maximum)
            denominator[...] = denominator[...] * scale + jnp.sum(
                probabilities, axis=1, keepdims=True
            )
            maximum[...] = new_maximum
            high = (
                (probabilities * 256).astype(jnp.float8_e4m3fn)
                if fp8
                else probabilities.astype(jnp.bfloat16)
            )
            weighted = jax.lax.dot_general(
                high,
                load(pl.ds(0, LATENT)),
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            if fp8:
                weighted = weighted * (1.0 / 256)
            accumulator[...] = accumulator[...] * scale + weighted

        def first(r):
            query_copy(r).wait()

            @pl.when(r + 1 < plan.rows)
            def _():
                query_copy(r + 1).start()

            if fp8:
                # mla_dcp._unpack_fp8_query: the scale by a lane reduction.
                packet = query_slots[r % 2].astype(jnp.uint32)
                lanes = jax.lax.broadcasted_iota(jnp.int32, packet.shape, 1)
                shifted = packet << (((lanes - 576) % 4) * 8)
                scale_bits = jnp.sum(
                    jnp.where((lanes >= 576) & (lanes < 580), shifted, jnp.uint32(0)),
                    axis=1,
                    keepdims=True,
                )
                query_scale[...] = jax.lax.bitcast_convert_type(scale_bits, jnp.float32)
                payload = jnp.where(lanes < 576, packet, jnp.uint32(0)).astype(
                    jnp.uint8
                )
                query_fp8[...] = jax.lax.bitcast_convert_type(
                    payload, jnp.float8_e4m3fn
                )
            maximum[...] = jnp.full(maximum.shape, -jnp.inf, jnp.float32)
            denominator[...] = jnp.zeros(denominator.shape, jnp.float32)
            accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)
            if compact:
                count_ref[0] = 0

        def last(r):
            total = denominator[...]
            has = total > 0
            safe = jnp.where(has, total, 1.0)
            if plan.full:

                @pl.when(r > 0)
                def _():
                    result_copy(r - 1).wait()

            result[:, pl.ds(0, LATENT)] = jnp.where(has, accumulator[...] / safe, 0.0)
            lse = jnp.where(has, maximum[...] + jnp.log(safe), NEG)
            pad = jax.lax.broadcasted_iota(jnp.int32, (FOLD_HEADS, pad_lanes), 1)
            result[:, pl.ds(LATENT, pad_lanes)] = jnp.where(pad == 0, lse, 0.0)
            if plan.full:
                result_copy(r).start()
            else:

                @pl.when(r == plan.rows - 1)
                def _():
                    result_copy(r).start()

        def onehot(load, parts, k):
            """The chunk's ``k`` selected tokens into dense rows ``[count,
            count + k)``: per 128 dense rows from the tile-aligned window
            holding ``count``, ``E [128, N] @ tokens`` (one nonzero product a
            row, exact), merged into the rows it owns."""
            count = count_ref[0]
            upper = jnp.where(
                jax.lax.broadcasted_iota(jnp.int32, (LANES, LANES), 0)
                < jax.lax.broadcasted_iota(jnp.int32, (LANES, LANES), 1),
                1.0,
                0.0,
            ).astype(jnp.bfloat16)
            stacked = -(-vregs // 16) * 16
            sublane = jax.lax.broadcasted_iota(jnp.int32, (stacked, LANES), 0)
            mask_rows = jnp.zeros((stacked, LANES), jnp.int32)
            for v in range(vregs):
                mask_rows = jnp.where(sublane == v, parts[v], mask_rows)
            # Each 128-token part's exclusive prefix, one 0/1 dot for all.
            prefix = jnp.dot(
                mask_rows.astype(jnp.float32).astype(jnp.bfloat16),
                upper,
                preferred_element_type=jnp.float32,
            ).astype(jnp.int32)
            positions, carry = [], count
            for v in range(vregs):
                positions.append(carry + prefix[v : v + 1])
                carry = carry + jnp.sum(parts[v], axis=1, keepdims=True)
            position = jnp.concatenate(positions, axis=1)
            selected = jnp.concatenate(parts, axis=1) != 0
            window0 = count // tile * tile
            passes = jnp.where(k > 0, (count - window0 + k + LANES - 1) // LANES, 0)

            @pl.loop(0, passes)
            def _(p):
                window = window0 + LANES * p
                row = window + jax.lax.broadcasted_iota(jnp.int32, (LANES, n), 0)
                one_hot = jnp.where((row == position) & selected, 1.0, 0.0)
                moved = jax.lax.dot_general(
                    one_hot.astype(jnp.bfloat16),
                    load(pl.ds(0, LATENT_LANES)),
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                ).astype(dtype)
                rows = pl.ds(
                    pl.multiple_of(window // packing, TILE_ROWS), LANES // packing
                )
                old = pltpu.bitcast(dense[rows, :], dtype)
                target = window + jax.lax.broadcasted_iota(
                    jnp.int32, (LANES, LATENT_LANES), 0
                )
                mine = (target >= count) & (target < count + k)
                dense[rows, :] = pltpu.bitcast(jnp.where(mine, moved, old), jnp.int32)

            count_ref[0] = count + k

        def copy(slot, k, offset):
            """The chunk's ``k`` selected tokens (``copy_tokens``) into the
            dense rows one by one: each token's row out of its aligned tile
            (a sum over sublanes of one nonzero), written into the dense tile
            holding the row's count."""
            sublane = jax.lax.broadcasted_iota(jnp.int32, (tile, LATENT_LANES), 0)

            def one(j, count):
                t = copies[0, offset + j]
                source = pl.ds(
                    pl.multiple_of(t // tile * TILE_ROWS, TILE_ROWS), TILE_ROWS
                )
                x = pltpu.bitcast(slot[source, :], dtype).astype(jnp.float32)
                row = jnp.sum(
                    jnp.where(sublane == t % tile, x, 0.0), axis=0, keepdims=True
                )
                target = pl.ds(
                    pl.multiple_of(count // tile * TILE_ROWS, TILE_ROWS), TILE_ROWS
                )
                block = pltpu.bitcast(dense[target, :], dtype)
                block = jnp.where(sublane == count % tile, row.astype(dtype), block)
                dense[target, :] = pltpu.bitcast(block, jnp.int32)
                return count + 1

            count_ref[0] = jax.lax.fori_loop(0, k, one, count_ref[0])

        def step(i, _):
            r, flags = field(i, 0), field(i, 1)

            @pl.when((flags & STEP_FIRST) != 0)
            def _():
                first(r)

            @pl.when((flags & (STEP_FOLD | STEP_COMPACT)) != 0)
            def _():
                c = field(i, 2)
                chunk_copy(c).wait()

                @pl.when(c + FOLD_DEPTH - 1 < plan.chunks)
                def _():
                    chunk_copy(c + FOLD_DEPTH - 1).start()

                slot = ring.at[c % FOLD_DEPTH]
                pad = jax.lax.broadcasted_iota(jnp.int32, (chunk_rows, pad_lanes), 1)
                pltpu.store(
                    slot.at[:, pl.ds(LATENT, pad_lanes)],
                    jnp.zeros((chunk_rows, pad_lanes), jnp.int32),
                    mask=pad >= 64,
                )
                if not plan.compute:
                    return
                parts = lane_mask(field(i, 3))

                @pl.when((flags & STEP_FOLD) != 0)
                def _():
                    fold(loader(slot), r, jnp.concatenate(parts, axis=1) != 0)

                if plan.compaction == "onehot":

                    @pl.when((flags & STEP_COMPACT) != 0)
                    def _():
                        onehot(loader(slot), parts, field(i, 4))

                elif plan.compaction == "copy":

                    @pl.when((flags & STEP_COMPACT) != 0)
                    def _():
                        copy(slot, field(i, 4), field(i, 5))

            if compact and plan.compute:

                @pl.when((flags & STEP_DENSE) != 0)
                def _():
                    first_row = pl.multiple_of(field(i, 3) * chunk_rows, TILE_ROWS)
                    valid = jax.lax.broadcasted_iota(jnp.int32, (1, n), 1) < field(i, 4)
                    fold(loader(dense.at[pl.ds(first_row, chunk_rows)]), r, valid)

            @pl.when((flags & STEP_LAST) != 0)
            def _():
                last(r)

        jax.lax.fori_loop(0, plan.steps, step, None)
        result_copy(plan.rows - 1).wait()

    return kernel, scratch


def _fold_body(plan: FoldPlan, tag: str):
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    kernel, scratch = _fold_kernel(plan)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def body(meta, masks, copies, queries, words, out, iterations):
        call = pl.pallas_call(
            kernel,
            name=f"dsa_micro_fold_{tag}" + ("_full" if plan.full else ""),
            out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
            in_specs=[smem, smem, smem, hbm, hbm, hbm],
            out_specs=hbm,
            scratch_shapes=scratch,
            input_output_aliases={5: 0},
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=FOLD_VMEM_LIMIT),
        )
        # The aliased output orders the calls; each call only overwrites it.
        return (
            jax.lax.fori_loop(
                0,
                iterations,
                lambda _, o: call(meta, masks, copies, queries, words, o),
                out,
            ),
        )

    return body


def _fold_check(
    stage, count, kv_dtype, fold_tokens, compaction, inputs, pool, extra, tag
):
    """Each rank's verdict on its own results: every checked row (all of them
    up to ``FOLD_CHECK_TOKENS`` selected tokens a call, else the first
    ``FOLD_CHECK_ROWS`` and the last) within ``FOLD_TOLERANCE`` of
    :func:`fold_reference` and its ``lse`` within
    :func:`fold_exact_lse_tolerance` of :func:`fold_exact_lse`, rows with
    nothing selected exactly ``(0, NEG)`` and lanes 513:640 zero. The o error
    against :func:`fold_model` over the same stream and step is recorded, not
    gated."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels.mla_dcp import LATENT, NEG

    q_nope, q_rope, w_uk, queries = inputs
    bits, table = pool
    rows = len(stage.selections)
    checked = list(range(rows))
    if rows * count > FOLD_CHECK_TOKENS:
        checked = sorted({*range(min(FOLD_CHECK_ROWS, rows)), rows - 1})
    bound_o, bound_lse = FOLD_TOLERANCE[kv_dtype]
    bound_exact = fold_exact_lse_tolerance(kv_dtype)
    names = ("max_error", "lse_error", "exact_lse_error", "model_error")

    def row_errors(got, r):
        """``(o error, lse error, exact lse error, model o error)`` of row
        ``r``."""
        o, lse = got[r, :, :LATENT], got[r, :, LATENT]
        selected = stage.selections[r]
        if not len(selected):
            exact = not o.any() and bool((lse == np.float32(NEG)).all())
            return (0.0,) * len(names) if exact else (np.inf,) * len(names)
        latents = latent_values(bits, table, r, selected, kv_dtype)
        o_ref, lse_ref = fold_reference(
            q_nope[r], q_rope[r], w_uk, latents, np.arange(len(selected))
        )
        o_error, lse_error = fold_errors(o, lse, o_ref, lse_ref)
        stream = selected if compaction != "none" else stage.streams[r]
        values = latent_values(bits, table, r, stream, kv_dtype)
        values[:, 576:] = 0
        q, scale = fold_query_values(queries[r], kv_dtype)
        model = fold_model(
            q, scale, values, np.isin(stream, selected), fold_tokens, kv_dtype
        )
        exact_error = np.abs(lse - fold_exact_lse(q, scale, latents)).max()
        return tuple(
            float(e)
            for e in (o_error, lse_error, exact_error, fold_errors(o, lse, *model)[0])
        )

    def check(outputs):
        import torch

        from vllm_torchtpu.megakernel import bridge

        rank = bridge.kernel_mesh(AXIS, RANKS)[1]
        try:
            got = _first(outputs)[0].cpu().numpy()
            errors = np.array([row_errors(got, r) for r in checked])
            within = (
                (errors[:, 0] <= bound_o)
                & (errors[:, 1] <= bound_lse)
                & (errors[:, 2] <= bound_exact)
            )
            ok = bool(within.all()) and not got[:, :, LATENT + 1 :].any()
            mine = (*(float(e) for e in errors.max(axis=0)), ok)
        except Exception as exc:  # noqa: BLE001 -- reported after the gather
            mine = f"{type(exc).__name__}: {exc}"
        # The record holds the largest errors over every rank's checked rows.
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, (rank, mine))
        parts = [m for _, m in gathered if not isinstance(m, str)]
        broken = [f"rank {r}: {m}" for r, m in gathered if isinstance(m, str)]
        if parts:
            for k, name in enumerate(names):
                extra[name] = float(np.max([p[k] for p in parts]))
            verdict = "PASS" if all(p[-1] for p in parts) and not broken else "FAIL"
            harness.log(
                f"{tag} check {verdict}: max |o - ref| / max |ref| "
                f"{extra['max_error']:.3g} (bound {bound_o:g}), max |lse - ref| "
                f"{extra['lse_error']:.3g} (bound {bound_lse:g}), max |lse - exact| "
                f"{extra['exact_lse_error']:.3g} (bound {bound_exact:g}), against "
                f"fold_model {extra['model_error']:.3g}; {len(checked)} of {rows} "
                f"rows checked on "
                f"{len(parts)} ranks" + (f"; {'; '.join(broken)}" if broken else "")
            )
        if isinstance(mine, str):
            raise RuntimeError(mine)
        return mine[-1]

    return check


def _fold_setup(
    rows,
    tokens,
    variant,
    pattern,
    count,
    kv_dtype,
    fold_tokens,
    compaction,
    compute,
    args,
) -> Setup:
    import ml_dtypes
    import torch
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel import bridge

    _check_fold_tokens(fold_tokens)
    if (
        variant not in FOLD_VARIANTS
        or compaction not in FOLD_COMPACTIONS
        or compute not in ("on", "off")
    ):
        raise ValueError(
            f"fold: unknown variant {variant!r}, compaction {compaction!r} or "
            f"compute {compute!r}"
        )
    if variant != "none" and not 0 < count <= tokens:
        raise ValueError(f"fold: cannot select {count} of {tokens} tokens")
    if variant == "none":
        compute = "on"  # nothing to fold either way
    row_mix = args.row_mix == "on"
    tag = fold_tag(
        variant, pattern, count, kv_dtype, fold_tokens, compaction, compute, row_mix
    )
    pool = _fold_pool(kv_dtype, tokens, max(args.rows))
    stage = fold_stage(
        rows, tokens, count, pattern, kv_dtype, variant, fold_tokens, compaction,
        row_mix, *pool,
    )  # fmt: skip
    inputs = _fold_inputs(max(args.rows))
    queries = inputs[3][:rows]
    if kv_dtype == "fp8":
        packets = np.stack([fp8_query_packet(q) for q in queries])
        query_operand = _upload_bits(packets, torch.uint8)
    else:
        query_operand = bridge.upload(queries.astype(ml_dtypes.bfloat16))
    longest = max(map(len, stage.selections))
    dense_tokens = 0
    if compaction != "none":
        # The one-hot window's last pass overhangs the count by < 128 rows.
        dense_tokens = -(-longest // fold_tokens) * fold_tokens + LANES
    plan = FoldPlan(
        rows,
        kv_dtype,
        variant,
        fold_tokens,
        compaction,
        compute == "on",
        stage.steps,
        stage.chunks,
        len(stage.words),
        len(stage.masks),
        len(stage.copy_tokens),
        dense_tokens,
        False,
    )
    extra = {
        "granularity": variant,
        "selected": 0 if variant == "none" else count,
        "pattern": pattern,
        "kv_dtype": kv_dtype,
        "fold_tokens": fold_tokens,
        "compaction": compaction,
        "compute": compute,
        "row_mix": args.row_mix,
        "steps": stage.steps,
        "chunks": stage.chunks,
        "folded_tokens": stage.folded_tokens,
        "selected_tokens": int(sum(map(len, stage.selections))),
    }
    invalid = compute == "off"
    return Setup(
        _fold_body(plan, tag),
        (
            bridge.upload(smem_table(stage.meta)),
            bridge.upload(smem_table(stage.masks)),
            bridge.upload(smem_table(stage.copy_tokens)),
            query_operand,
            bridge.upload(stage.words),
            bridge.upload(np.zeros((rows, FOLD_HEADS, LATENT_LANES), np.float32)),
        ),
        (P(AXIS),) * 6,
        (P(AXIS),),
        variant=tag,
        check=None
        if invalid
        else _fold_check(
            stage,
            count,
            kv_dtype,
            fold_tokens,
            compaction,
            inputs,
            pool,
            extra,
            f"fold_r{rows}_t{tokens}_{tag}",
        ),
        invalid=invalid,
        check_body=None if invalid else _fold_body(replace(plan, full=True), tag),
        extra=extra,
    )


def _build_fold(rows: int, tokens: int, args) -> list[Setup]:
    """The distinct fold setups at one point, over ``--kv-dtype`` x
    ``--pattern`` x ``--selected`` x ``--variant`` (row, tile, page, dense,
    none) x ``--fold-tokens`` x ``--fold-compaction`` (tile and page only) x
    ``--fold-compute``, skipping S above T, ``page`` steps that are not whole
    pages, and points whose SMEM tables exceed ``FOLD_SMEM_WORDS`` (logged).
    The variant tag is ``{row|tile|page|dense}_{uniform|recent}_s{S}_{bf16|
    fp8}_k{N}`` plus ``_onehot`` / ``_copy`` (the compaction), ``_nocompute``
    (``--fold-compute off``) and ``_mix`` (``--row-mix on``); ``none_{bf16|
    fp8}`` for ``none``, which depends on none of them."""
    variants = [v for v in args.variant if v in FOLD_VARIANTS] or list(FOLD_VARIANTS)
    setups = {}
    grid = itertools.product(
        args.kv_dtype,
        args.pattern,
        args.selected,
        variants,
        args.fold_tokens,
        args.fold_compaction,
        args.fold_compute,
    )
    for kv_dtype, pattern, count, variant, fold_tokens, compaction, compute in grid:
        if compaction != "none" and variant not in ("tile", "page"):
            continue
        if variant != "none" and count > tokens:
            continue
        if variant == "page" and fold_tokens % GEOMETRY[kv_dtype].block_tokens:
            continue
        row_mix = args.row_mix == "on"
        knobs = (variant, pattern, count, kv_dtype, fold_tokens, compaction, compute)
        tag = fold_tag(*knobs, row_mix)
        if tag in setups:
            continue
        try:
            setups[tag] = _fold_setup(
                rows, tokens, variant, pattern, count, kv_dtype, fold_tokens,
                compaction, compute, args,
            )  # fmt: skip
        except FoldTablesTooLarge as exc:
            harness.log(f"fold R={rows} T={tokens} {tag}: skipped ({exc})")
    return list(setups.values())


def score_query_rows(rows: int, layout: str = "padded"):
    """``[R, 32]``: the kernel's query row of row ``r``'s head ``h``. Rows go
    in groups of ``G`` (:func:`tile_group_rows`), group ``g`` taking query rows
    ``[32 G g, 32 G (g + 1))`` head-major: ``(r, h)`` at ``32 G g + G h + r %
    G``, so that each 8-row tile of a group's logits holds ``8 / G`` heads of
    its ``G`` rows. ``layout="select"`` at R = 2, 4 puts ``(r, h)`` at ``8 (h
    // g) + r g + h % g``, ``g = 8 / R``: tile ``t`` holds heads ``g t .. g t +
    g - 1`` of every row, row ``r`` on its virtual rows' sublanes ``r g .. r g
    + g - 1`` (at R = 1 and R >= 8 both layouts are the same)."""
    from vllm_torchtpu.megakernel.kimi_k3.dsa import INDEX_HEADS

    group = tile_group_rows(rows, "score")
    row = np.arange(rows)[:, None]
    head = np.arange(INDEX_HEADS)[None]
    if layout == "select" and 1 < rows < SUBLANES:
        parts = SUBLANES // rows
        return SUBLANES * (head // parts) + parts * row + head % parts
    return INDEX_HEADS * group * (row // group) + group * head + row % group


def head_sum_model(logits, weights, rows: int, layout: str = "padded"):
    """The kernel's head sum: ``logits [32 R, n]`` and ``weights [32 R]`` in
    :func:`score_query_rows` order -> ``[R_pad, n]``, ``sum_h w ReLU(logit)``
    per row. A group's ``8 / G``-head tiles are summed, then rolls over
    sublanes by ``G, 2 G, ...`` add the heads that share a tile: ``padded``
    keeps row ``r`` on sublane ``r`` (padded rows zero). ``select`` below eight
    rows puts row ``r``'s total on each of its sublanes ``r g .. r g + g - 1``:
    at R = 2, 4 by the select's ``group_sum`` (rolls by 1 .. g / 2 within
    blocks of ``g`` sublanes), at R = 1 by the rolls by 1, 2, 4."""
    group = tile_group_rows(rows, "score")
    span = len(logits) // (rows // group)
    sub = np.arange(SUBLANES)[:, None]
    out = np.zeros((padded_rows(rows), logits.shape[1]), np.float32)
    for g in range(rows // group):
        part = slice(g * span, (g + 1) * span)
        terms = np.maximum(logits[part], 0) * weights[part, None]
        total = terms.reshape(-1, 8, logits.shape[1]).sum(axis=0)
        if layout == "select" and 1 < rows < SUBLANES:
            step = 1
            while step < SUBLANES // rows:
                mate = np.where(
                    sub & step, np.roll(total, step, 0), np.roll(total, 8 - step, 0)
                )
                total, step = total + mate, 2 * step
        else:
            shift = group
            while shift < 8:
                total = total + np.roll(total, shift, axis=0)
                shift *= 2
        kept = 8 if layout == "select" else min(8, rows)
        out[8 * g : 8 * g + kept] = total[:kept]
    return out


def signed_keys(scores):
    """The score kernel's int32 keys of FP32 ``scores`` (``--emit keys``): the
    select's signed form of :func:`order_keys`, a higher key a higher score."""
    return (order_keys(scores) ^ np.uint32(0x80000000)).view(np.int32)


def scores_of_keys(keys):
    """The FP32 scores of :func:`signed_keys`' keys (the same bit map, an
    involution on the int32 bits)."""
    keys = np.ascontiguousarray(keys, np.int32)
    return (keys ^ ((keys >> 31) & 0x7FFFFFFF)).view(np.float32)


def virtual_place(column, rows: int, tokens: int):
    """``(plane, part)`` of the 128-token lane column ``column`` in the
    select's layout (:func:`to_virtual`): row ``r``'s keys of that column are
    virtual row ``r g + part`` of plane ``plane``."""
    planes = tokens // (virtual_groups(rows, tokens) * LANES)
    return column % planes, column // planes


def block_major(rows: int, pages_per_step: int) -> bool:
    """Whether per-row pages go block-major: eight pages a step from eight
    rows, one block of a group of eight rows."""
    return pages_per_step == SUBLANES and rows >= SUBLANES


def score_stream(rows: int, blocks: int, pages_per_step: int):
    """``(row, block)`` of each per-row stream entry, ``pages_per_step`` K
    entries a loop step: row-major (K consecutive blocks of one row), or
    block-major (:func:`block_major`: step ``t blocks + b`` is block ``b`` of
    rows ``8 t .. 8 t + 7``)."""
    entry = np.arange(rows * blocks)
    if block_major(rows, pages_per_step):
        step, page = entry // SUBLANES, entry % SUBLANES
        return SUBLANES * (step // blocks) + page, step % blocks
    return entry // blocks, entry % blocks


def combine_rows(tiles, sublane, roll, where):
    """Eight ``[8, 128]`` tiles -> one, sublane ``r`` holding the sum of
    ``tiles[r]``'s sublanes. Three pairwise levels (bit 1, 2, 4 of the
    sublane): a pair's first tile keeps the sublanes where the bit is clear,
    the second those where it is set, and each adds its mate across that bit
    (``roll`` by the bit, either direction)."""
    step = 1
    while len(tiles) > 1:
        low = (sublane & step) == 0
        tiles = [
            where(low, a, b) + roll(where(low, b, a), step)
            for a, b in zip(tiles[::2], tiles[1::2])
        ]
        step *= 2
    return tiles[0]


def score_keys(rng, tokens: int):
    """``[T, 128]`` BF16 index keys, random normal times ``2^k`` per token,
    ``k`` uniform in ``[-SCORE_EXPONENTS, SCORE_EXPONENTS]``, so that tokens
    get different scales."""
    import ml_dtypes

    from vllm_torchtpu.megakernel.kimi_k3.dsa import INDEX_DIM

    magnitude = 2.0 ** rng.integers(-SCORE_EXPONENTS, SCORE_EXPONENTS + 1, (tokens, 1))
    return (rng.standard_normal((tokens, INDEX_DIM)) * magnitude).astype(
        ml_dtypes.bfloat16
    )


def pack_index_blocks(key_bytes, exponent, block_tokens: int):
    """``[blocks, *index_block_shape(block_tokens)]`` uint8 of local keys
    (``dsa_cache.quantize_index_keys``' ``key_bytes [T, 128]`` and ``exponent
    [T]``), laid out as the ``dsa_cache`` module docstring says."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    tokens = len(key_bytes)
    if tokens % block_tokens:
        raise ValueError(
            f"score: {tokens} local tokens must be whole blocks of {block_tokens}"
        )
    blocks = tokens // block_tokens
    out = np.zeros((blocks, *dsa_cache.index_block_shape(block_tokens)), np.uint8)
    out[:, :block_tokens] = np.reshape(key_bytes, (blocks, block_tokens, -1))
    out[:, block_tokens : block_tokens + block_tokens // LANES] = np.reshape(
        exponent, (blocks, -1, LANES)
    )
    return out


def score_errors(got, wanted, valid=None):
    """Per row: ``max |got - wanted| / max |wanted|`` (in FP64), over the
    tokens ``valid`` marks (default: all; a row with none: 0)."""
    got, wanted = np.asarray(got, np.float64), np.asarray(wanted, np.float64)
    if valid is None:
        return np.abs(got - wanted).max(axis=1) / np.abs(wanted).max(axis=1)
    diff = np.where(valid, np.abs(got - wanted), 0).max(axis=1, initial=0)
    scale = np.where(valid, np.abs(wanted), 0).max(axis=1, initial=0)
    return diff / np.where(scale > 0, scale, 1)


def topk_recall(got, wanted, k: int):
    """Per row: the share of ``wanted``'s top ``k`` tokens among ``got``'s
    (1 when a row has at most ``k`` tokens)."""
    if got.shape[1] <= k:
        return np.ones(len(got))
    top_got = np.argpartition(-got, k, axis=1)[:, :k]
    top_wanted = np.argpartition(-wanted, k, axis=1)[:, :k]
    return np.array(
        [len(np.intersect1d(a, b)) / k for a, b in zip(top_got, top_wanted)]
    )


def valid_topk_recall(got, wanted, k: int):
    """:func:`topk_recall` per row over the tokens whose wanted score is
    finite (masked tokens are ``-inf``)."""
    return np.array(
        [
            topk_recall(g[None, np.isfinite(w)], w[None, np.isfinite(w)], k)[0]
            for g, w in zip(got, wanted)
        ]
    )


@dataclass
class ScoreInputs:
    """A rank's scoring inputs: ``queries [R, 32, 128]`` FP32 holding FP8
    values (``dsa_cache.quantize_index_queries``), head weights ``weights [R,
    32]``, the keys' ``key_bytes [T, 128]`` and ``exponent [T]`` (per-row
    pages: ``[R, T, 128]`` and ``[R, T]``, each row's own keys), and
    ``dsa_reference.scores`` of them ``reference [R, T]``."""

    queries: Any
    weights: Any
    key_bytes: Any
    exponent: Any
    reference: Any


SCORE_INPUTS: dict[tuple[int, int, str], ScoreInputs] = {}


def _score_inputs(rows: int, tokens: int, pages: str = "shared") -> ScoreInputs:
    if (rows, tokens, pages) not in SCORE_INPUTS:
        import jax
        import jax.numpy as jnp

        from tools.megakernel import dsa_reference as ref
        from vllm_torchtpu.megakernel import bridge
        from vllm_torchtpu.megakernel.kimi_k3.dsa import INDEX_DIM, INDEX_HEADS
        from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

        rank = bridge.kernel_mesh(AXIS, RANKS)[1]
        shared = pages == "shared"
        seed = [SCORE_SEED, rows, tokens, rank] + ([] if shared else [1])
        rng = np.random.default_rng(seed)
        keys = [score_keys(rng, tokens) for _ in range(1 if shared else rows)]
        raw = rng.standard_normal((rows, INDEX_HEADS, INDEX_DIM)).astype(np.float32)
        weights = rng.standard_normal((rows, INDEX_HEADS)).astype(np.float32)
        key_bytes, exponent, reference = [], [], []
        with jax.default_device(jax.devices("cpu")[0]):
            queries = dsa_cache.quantize_index_queries(jnp.asarray(raw))[0]
            for r, own in enumerate(keys):
                kb, ex = dsa_cache.quantize_index_keys(jnp.asarray(own))
                dequantized = dsa_cache.dequantize_index_keys(kb, ex)
                scored = slice(None) if shared else slice(r, r + 1)
                reference.append(
                    np.concatenate(
                        [
                            np.asarray(
                                ref.scores(
                                    queries[scored],
                                    jnp.asarray(weights[scored]),
                                    dequantized[start : start + SCORE_REFERENCE_TOKENS],
                                )
                            )
                            for start in range(0, tokens, SCORE_REFERENCE_TOKENS)
                        ],
                        axis=1,
                    )
                )
                key_bytes.append(np.asarray(kb))
                exponent.append(np.asarray(ex))
        SCORE_INPUTS.clear()
        SCORE_INPUTS[(rows, tokens, pages)] = ScoreInputs(
            np.asarray(queries),
            weights,
            key_bytes[0] if shared else np.stack(key_bytes),
            exponent[0] if shared else np.stack(exponent),
            np.concatenate(reference, axis=0),
        )
    return SCORE_INPUTS[(rows, tokens, pages)]


@dataclass(frozen=True)
class ScorePlan:
    """A score kernel: ``rows`` rows over ``tokens`` local keys in blocks of
    ``block_tokens``; one MXU dot per ``dot_tokens`` keys and ``dot_groups``
    row groups; ``prefetch`` blocks in flight while one is scored (1: double
    buffered); ``full`` copies every score to HBM, else the last 128 tokens'
    (``padded``) or the last plane (``select``). ``layout``: the score buffer,
    ``[R_pad, T]`` (``padded``) or the select's virtual rows ``[planes, V,
    128]`` (``select``); ``emit``: FP32 ``scores`` or the select's int32
    ``keys``; ``pages``: every row scores one ``shared`` key set, or its own
    blocks (``per-row``), ``pages_per_step`` of them a loop step
    (:func:`score_stream`), or through the decode kernel's scorer
    (``decode``); ``work``: ``full``, or the per-row timing-only arms
    ``compute-only`` (no DMA: the ring's pages scored again) and ``dma-only``
    (pages streamed, not scored)."""

    rows: int
    tokens: int
    block_tokens: int
    dot_tokens: int
    dot_groups: int
    prefetch: int
    full: bool
    layout: str = "padded"
    emit: str = "scores"
    pages: str = "shared"
    pages_per_step: int = 1
    work: str = "full"
    sequence_rows: int = 1
    decode_calls: str = "chunks"

    @property
    def blocks(self) -> int:
        return self.tokens // self.block_tokens

    @property
    def stream(self) -> int:
        """The index blocks one call streams: one key set's, or every row's."""
        return self.blocks * (1 if self.pages == "shared" else self.rows)

    @property
    def steps(self) -> int:
        return self.stream // self.pages_per_step

    @property
    def in_flight(self) -> int:
        """Steps whose pages are in flight while one is scored: ``prefetch``
        pages, rounded up to whole steps."""
        return min(-(-self.prefetch // self.pages_per_step), self.steps)

    @property
    def slots(self) -> int:
        return min(self.in_flight + 1, self.steps) * self.pages_per_step

    @property
    def block_major(self) -> bool:
        return self.pages == "per-row" and block_major(self.rows, self.pages_per_step)


def _score_kernel(plan: ScorePlan):
    """The score ``pallas_call`` kernel and its scratch shapes. A ring of
    ``plan.slots`` VMEM slots streams the blocks, each refilled with the block
    ``prefetch`` ahead once its own is scored; per ``dot_tokens`` keys and
    ``dot_groups`` row groups one MXU dot, then per group and 128 keys the head
    sum (:func:`head_sum_model`) times the keys' scales, stored into the score
    buffer.

    ``select`` below eight rows stores each 128-key column on its plane by one
    read-modify-write of the plane's ``[8, 128]`` tile, on the sublanes of the
    column's part (:func:`virtual_place`). ``keys`` turns each total into the
    select's int32 key, ``INT32_MIN`` past its row's count (SMEM ``meta``,
    broadcast per sublane at entry). ``per-row`` streams the block table's
    pages in :func:`score_stream` order, one ring for the whole call, K =
    ``pages_per_step`` pages a loop step: the step starts the pages of the
    step ``in_flight`` ahead into the previous step's slots, waits for its own
    K, then per dot's keys issues the K pages' dots (each its row's 32 heads)
    before their K head sums. Row-major steps put each total on its row's
    sublane by a read-modify-write; block-major steps combine the eight rows'
    totals into one tile (:func:`combine_rows`) and store it."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.dsa import INDEX_DIM, INDEX_HEADS
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache
    from vllm_torchtpu.megakernel.kimi_k3.kernels.dsa_select import _group_sum, _key

    rows, tokens, block_tokens = plan.rows, plan.tokens, plan.block_tokens
    dot_tokens, dot_groups, slot_count = plan.dot_tokens, plan.dot_groups, plan.slots
    per_step, in_flight, steps = plan.pages_per_step, plan.in_flight, plan.steps
    group = tile_group_rows(rows, "score")
    groups = rows // group
    span = INDEX_HEADS * group  # query rows of a group
    scale_rows = dsa_cache.SCALE_ROWS
    select, keyed = plan.layout == "select", plan.emit == "keys"
    per_row = plan.pages == "per-row"
    padded = padded_rows(rows)
    parts = virtual_groups(rows, tokens) if select else 1
    planes = tokens // (parts * LANES)
    plane_bits = planes.bit_length() - 1
    shape = (planes, padded, LANES) if select else (padded, tokens)
    scratch = [
        pltpu.VMEM((slot_count, block_tokens + scale_rows, INDEX_DIM), jnp.uint8),
        pltpu.VMEM(shape, jnp.int32 if keyed else jnp.float32),
        pltpu.SemaphoreType.DMA((slot_count + 1,)),
    ]
    if keyed and (not per_row or plan.block_major):
        scratch.append(pltpu.VMEM((padded // SUBLANES, SUBLANES, LANES), jnp.int32))

    def owner(v):
        """The row of virtual row ``v`` (None: a padded row)."""
        return v // parts if select else (v if v < rows else None)

    def tree_sum(terms):
        while len(terms) > 1:
            terms = [
                terms[i] + terms[i + 1] if i + 1 < len(terms) else terms[i]
                for i in range(0, len(terms), 2)
            ]
        return terms[0]

    def kernel(*refs):
        meta = refs[0] if keyed else None
        table = refs[keyed] if per_row else None
        queries, weights, pool, _, out, slots, scores, sems, *rest = refs[
            keyed + per_row :
        ]

        def fetch(i):
            slot = i % slot_count
            source = pool.at[0, table[0, i]] if per_row else pool.at[0, i]
            return pltpu.make_async_copy(source, slots.at[slot], sems.at[slot])

        if plan.work == "compute-only":
            # Fill the ring once: every step scores the pages left in it.
            for i in range(slot_count):
                fetch(i).start()
            for i in range(slot_count):
                fetch(i).wait()
        else:
            for i in range(in_flight * per_step):
                fetch(i).start()
        sublane = jax.lax.broadcasted_iota(jnp.int32, (8, LANES), 0)
        if keyed:
            lane = jax.lax.broadcasted_iota(jnp.int32, (8, LANES), 1)
        if rest:
            for tile in range(padded // SUBLANES):
                count = jnp.zeros((SUBLANES, LANES), jnp.int32)
                for s in range(SUBLANES):
                    row = owner(SUBLANES * tile + s)
                    if row is not None:
                        count = jnp.where(sublane == s, meta[0, row], count)
                rest[0][tile] = count
        if per_row and not select and rows < SUBLANES:
            # Only the rows' own sublanes are stored: the padded ones keep this.
            scores[...] = jnp.full(shape, INT32_MIN if keyed else 0, scores.dtype)

        def emitted(value, column, count):
            if not keyed:
                return value
            key = _key(pltpu.bitcast(value, jnp.int32))
            return jnp.where(column + lane < count, key, INT32_MIN)

        def on_plane(col):
            """The plane and part of 128-key column ``col`` below eight rows."""
            return col & (planes - 1), col >> plane_bits

        def put(value, tile, column, col):
            """``value``: 128 keys of tile ``tile``'s eight virtual rows."""
            if not select:
                scores[
                    pl.ds(tile * 8, 8), pl.ds(pl.multiple_of(column, LANES), LANES)
                ] = value
            elif rows >= SUBLANES:
                scores[col, pl.ds(tile * 8, 8), :] = value
            else:
                plane, part = on_plane(col)
                scores[plane] = jnp.where(
                    (sublane & (parts - 1)) == part, value, scores[plane]
                )

        def put_row(value, row, column, col):
            """``value``: row ``row``'s 128 keys on every sublane."""
            if select and rows < SUBLANES:
                plane, part = on_plane(col)
                index, mine = (plane,), sublane == row * parts + part
            else:
                tile = pl.ds(pl.multiple_of((row >> 3) * 8, 8), 8)
                index = (
                    (col, tile, slice(None))
                    if select
                    else (tile, pl.ds(pl.multiple_of(column, LANES), LANES))
                )
                mine = sublane == (row & 7)
            scores[index] = jnp.where(mine, value, scores[index])

        def score_block(i, b):
            """Stream entry ``i``: local block ``b`` of every row."""

            # The slot of entry i - 1, scored in the previous iteration.
            @pl.when(i + plan.prefetch < plan.stream)
            def _():
                fetch(i + plan.prefetch).start()

            fetch(i).wait()
            slot = i % slot_count
            exponents = slots[slot, pl.ds(block_tokens, scale_rows), :].astype(
                jnp.int32
            )
            for first in range(0, block_tokens, dot_tokens):
                keys = pltpu.bitcast(
                    slots[slot, pl.ds(first, dot_tokens), :], jnp.float8_e4m3fn
                )
                for g0 in range(0, groups, dot_groups):
                    count = min(dot_groups, groups - g0)
                    logits = jax.lax.dot_general(
                        queries[0, pl.ds(g0 * span, count * span), :],
                        keys,
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    for g in range(count):
                        for c in range(dot_tokens // LANES):
                            terms = [
                                jnp.maximum(
                                    logits[
                                        g * span + 8 * t : g * span + 8 * t + 8,
                                        c * LANES : (c + 1) * LANES,
                                    ],
                                    0.0,
                                )
                                * weights[0, pl.ds((g0 + g) * span + 8 * t, 8), :]
                                for t in range(span // 8)
                            ]
                            total = tree_sum(terms)
                            if select and 1 < rows < SUBLANES:
                                total = _group_sum(total, parts)
                            else:
                                shift = group
                                while shift < 8:
                                    total = total + pltpu.roll(total, shift, 0)
                                    shift *= 2
                            if group < 8 and not select:
                                total = jnp.where(sublane < rows, total, 0.0)
                            k = (first + c * LANES) // LANES
                            scale = pltpu.bitcast(
                                exponents[k : k + 1] << 23, jnp.float32
                            )
                            column = b * block_tokens + first + c * LANES
                            col = b * (block_tokens // LANES) + k
                            value = emitted(
                                total * scale,
                                column,
                                rest[0][g0 + g] if keyed else None,
                            )
                            put(value, g0 + g, column, col)

        def score_step(j, pages, tile):
            """Loop step ``j``: stream entries ``j K .. j K + K - 1``, page
            ``p`` local block ``pages[p][1]`` of row ``pages[p][0]``; ``tile``:
            the rows' tile (block-major), else None."""
            if plan.work != "compute-only":
                # Into the slots of step j - 1, scored in the previous iteration.
                @pl.when(j + in_flight < steps)
                def _():
                    for p in range(per_step):
                        fetch((j + in_flight) * per_step + p).start()

                for p in range(per_step):
                    fetch(j * per_step + p).wait()
            if plan.work == "dma-only":
                return
            base = (j * per_step) % slot_count
            exponents = [
                slots[base + p, pl.ds(block_tokens, scale_rows), :].astype(jnp.int32)
                for p in range(per_step)
            ]
            heads = [pl.multiple_of(row * INDEX_HEADS, INDEX_HEADS) for row, _ in pages]
            for first in range(0, block_tokens, dot_tokens):
                logits = [
                    jax.lax.dot_general(
                        queries[0, pl.ds(heads[p], INDEX_HEADS), :],
                        pltpu.bitcast(
                            slots[base + p, pl.ds(first, dot_tokens), :],
                            jnp.float8_e4m3fn,
                        ),
                        (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    for p in range(per_step)
                ]
                for c in range(dot_tokens // LANES):
                    k = (first + c * LANES) // LANES
                    totals = []
                    for p, (row, b) in enumerate(pages):
                        terms = [
                            jnp.maximum(
                                logits[p][
                                    8 * t : 8 * t + 8, c * LANES : (c + 1) * LANES
                                ],
                                0.0,
                            )
                            * weights[
                                0, pl.ds(pl.multiple_of(heads[p] + 8 * t, 8), 8), :
                            ]
                            for t in range(INDEX_HEADS // 8)
                        ]
                        total = tree_sum(terms)
                        scale = pltpu.bitcast(
                            exponents[p][k : k + 1] << 23, jnp.float32
                        )
                        if tile is not None:
                            totals.append(total * scale)
                            continue
                        for shift in (1, 2, 4):
                            total = total + pltpu.roll(total, shift, 0)
                        column = b * block_tokens + first + c * LANES
                        col = b * (block_tokens // LANES) + k
                        value = emitted(
                            total * scale, column, meta[0, row] if keyed else None
                        )
                        put_row(value, row, column, col)
                    if tile is None:
                        continue
                    b = pages[0][1]
                    column = b * block_tokens + first + c * LANES
                    col = b * (block_tokens // LANES) + k
                    total = combine_rows(
                        totals, sublane, lambda x, s: pltpu.roll(x, s, 0), jnp.where
                    )
                    value = emitted(total, column, rest[0][tile] if keyed else None)
                    rows8 = pl.ds(pl.multiple_of(tile * 8, 8), 8)
                    if select:
                        scores[col, rows8, :] = value
                    else:
                        scores[rows8, pl.ds(pl.multiple_of(column, LANES), LANES)] = (
                            value
                        )

        if plan.block_major:

            def one_tile(tile, _):
                def one_block(b, _):
                    pages = [(SUBLANES * tile + p, b) for p in range(SUBLANES)]
                    score_step(tile * plan.blocks + b, pages, tile)

                jax.lax.fori_loop(0, plan.blocks, one_block, None)

            jax.lax.fori_loop(0, rows // SUBLANES, one_tile, None)
        elif per_row:
            row_steps = plan.blocks // per_step

            def one_row(row, _):
                def one_step(s, _):
                    pages = [(row, s * per_step + p) for p in range(per_step)]
                    score_step(row * row_steps + s, pages, None)

                jax.lax.fori_loop(0, row_steps, one_step, None)

            jax.lax.fori_loop(0, rows, one_row, None)
        else:
            jax.lax.fori_loop(0, plan.blocks, lambda b, _: score_block(b, b), None)
        if plan.full:
            source, target = scores, out.at[0]
        elif select:
            source, target = scores.at[planes - 1], out.at[0, planes - 1]
        else:
            last = pl.ds(tokens - LANES, LANES)
            source, target = scores.at[:, last], out.at[0, :, last]
        store = pltpu.make_async_copy(source, target, sems.at[slot_count])
        store.start()
        store.wait()

    return kernel, scratch


def _score_body(plan: ScorePlan, tag: str):
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    kernel, scratch = _score_kernel(plan)
    tables = (plan.emit == "keys") + (plan.pages == "per-row")

    def body(*args):
        *inputs, out, iterations = args
        call = pl.pallas_call(
            kernel,
            name=f"dsa_micro_score_{tag}",
            out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
            in_specs=[
                *[pl.BlockSpec(memory_space=pltpu.SMEM)] * tables,
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch,
            input_output_aliases={len(inputs): 0},
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=SCORE_VMEM_LIMIT),
        )
        # The aliased scores order the calls; each call only overwrites them.
        return (jax.lax.fori_loop(0, iterations, lambda _, o: call(*inputs, o), out),)

    return body


def decode_call_rows(batch: int, calls: str = "chunks") -> int:
    """Rows a decode layer's scorer call scores: B2's two one-row groups, 8 from
    eight rows (B8's group, the 8-row DCP chunks), else the batch. At B2 the
    decode layer calls once a group from a Python loop (two traced scorer
    bodies); ``dsa_decode.score_calls`` rolls the two calls into one body."""
    if (
        calls == "whole"
    ):  # the decode stack's score_and_select: every row of a pass in one call
        return batch
    return 1 if batch == 2 else min(batch, 8)


def _decode_score_body(plan: ScorePlan, tag: str, rank: int):
    """``--score-pages decode``: ``dsa_decode.IndexScorer`` on rank ``rank``
    over the local tables (SMEM) at the rows' positions (SMEM), primed and
    called as the decode layer calls it (:func:`decode_call_rows`); the scores'
    last plane (``full``: all of them) is copied to HBM."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.config import active
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    rows, block = plan.rows, plan.block_tokens
    shapes = dsa_decode.scratch_shapes(
        rows, plan.tokens, block, False, sequence_rows=plan.sequence_rows
    )
    planes = shapes["scores"][0][0]
    ring_count = shapes["ring"][0][0]

    def kernel(
        table, positions, cache, queries, weights, new_values, _, out, scores, ring,
        *scoped,
    ):  # fmt: skip
        scratch = dsa_decode.IndexScratch(
            cache, None, ring, queries, weights, None, scores, *scoped
        )
        width = plan.sequence_rows
        # A verify sequence's first row streams its pages (DcpAttention's
        # counts for cached verify rows); the others none.
        counts = [
            dsa_decode.local_pages(positions[r] + width - 1, rank, block, RANKS)
            if r % width == 0
            else jnp.int32(0)
            for r in range(rows)
        ]
        scorer = dsa_decode.IndexScorer(
            scratch, position_ref=positions, batch=rows, active=positions[0] >= 0,
            table_ref=table, pages=plan.blocks, layer=0, page_tokens=block,
            group=RANKS, global_pages=False, owner=rank, slots=3,
            uint8_packets=False, new_value_ref=new_values, counts=counts,
            sequence_rows=width, skip=active().dcp_skip,
        )  # fmt: skip
        dsa_decode.score_calls(scorer, rows, decode_call_rows(rows, plan.decode_calls))
        source, target = (
            (scores, out) if plan.full else (scores.at[planes - 1], out.at[planes - 1])
        )
        # The query head's semaphore, which this kernel leaves unused.
        copy = pltpu.make_async_copy(source, target, scoped[-1].at[ring_count])
        copy.start()
        copy.wait()

    def body(*args):
        *inputs, out, iterations = args
        # This rank's block of each operand, once, outside the timed loop.
        inputs = [x.reshape(x.shape[1:]) for x in inputs]
        call = pl.pallas_call(
            kernel,
            name=f"dsa_micro_score_{tag}",
            out_shape=jax.ShapeDtypeStruct(out.shape[1:], out.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.HBM),
                    *[pl.BlockSpec(memory_space=pltpu.VMEM)] * 3,
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=(
                    pltpu.VMEM(*shapes["scores"]),
                    pltpu.VMEM(*shapes["ring"]),
                    *dsa_decode.scoped_scratch(rows, False, ring_count),
                ),
            ),
            input_output_aliases={len(inputs): 0},
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=SCORE_VMEM_LIMIT),
        )
        # The aliased scores order the calls; each call only overwrites them.
        scores = jax.lax.fori_loop(
            0, iterations, lambda _, o: call(*inputs, o), out.reshape(out.shape[1:])
        )
        return (scores.reshape(out.shape),)

    return body


def decode_reference_scores(setup: Setup, rank: int):
    """``dsa_decode.score_rows`` on a ``--score-pages decode`` setup's own
    operands (this rank's ``[1, ...]`` blocks, host or interpreted arrays) in
    the decode layer's calls: the scores ``[planes, V, 128]`` the decode form
    writes (tests)."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    table, positions, cache, queries, weights, new_values, _ = (
        x[0] for x in setup.operands
    )
    rows = len(queries)
    return dsa_decode.score_rows(
        cache,
        table[: len(cache)].reshape(rows, -1),  # one pool block a table entry
        positions[:rows],
        queries,
        weights,
        new_values,
        layer=0,
        page_tokens=setup.extra["block_tokens"],
        group=RANKS,
        owner=rank,
        chunk=decode_call_rows(rows),
    )


def _score_check(
    rows,
    inputs: ScoreInputs,
    bound: float,
    extra,
    tag,
    *,
    layout="padded",
    emit="scores",
    counts=None,
):
    """Each rank's verdict on its own scores (the error metric within
    ``bound``, padded rows zero). Every rank's metrics are gathered, inside the
    check, for the log and the record (``extra``): the largest error, the lowest
    local top-k recall and, for up to ``GLOBAL_RECALL_ROWS`` rows, the recall of
    the top k of all ranks' tokens, which needs every rank's scores.
    ``layout="select"`` reads ``[planes, V, 128]`` (:func:`from_virtual`),
    ``layout="decode"`` the decode scorer's (``dsa_decode.scores_by_row``).
    ``emit="keys"`` compares the keys of each row's first ``counts[r]`` tokens
    (default: all) as scores (:func:`scores_of_keys`); every later token and
    every padded row must hold ``INT32_MIN``. ``emit="scores"`` with
    ``counts`` compares each row's first ``counts[r]`` scores only."""

    def check(outputs):
        import torch

        from vllm_torchtpu.megakernel import bridge
        from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

        rank = bridge.kernel_mesh(AXIS, RANKS)[1]
        try:
            got = _first(outputs)[0].cpu().numpy()
            if layout == "select":
                values, rest = from_virtual(got, rows), got[:0]
            elif layout == "decode":
                values, rest = dsa_decode.scores_by_row(got, rows), got[:0]
            else:
                values, rest = got[:rows], got[rows:]
            if emit == "keys":
                tokens = values.shape[1]
                limit = np.full(rows, tokens) if counts is None else np.asarray(counts)
                valid = np.arange(tokens)[None] < limit[:, None]
                rest_ok = (rest == INT32_MIN).all() and (
                    values[~valid] == INT32_MIN
                ).all()
                scores = np.where(valid, scores_of_keys(values), -np.inf)
                wanted = np.where(valid, inputs.reference, -np.inf)
            elif counts is None:
                valid, rest_ok = None, not rest.any()
                scores, wanted = values, inputs.reference
            else:
                valid = np.arange(values.shape[1])[None] < np.asarray(counts)[:, None]
                rest_ok = not rest.any()
                scores = values
                wanted = np.where(valid, inputs.reference, -np.inf)
            mine = (
                score_errors(scores, inputs.reference, valid),
                valid_topk_recall(scores, wanted, SELECT_K),
                bool(rest_ok),
                (scores, wanted) if rows <= GLOBAL_RECALL_ROWS else None,
            )
        except Exception as exc:  # noqa: BLE001 -- reported after the gather
            mine = f"{type(exc).__name__}: {exc}"
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, (rank, mine))
        parts = sorted(
            ((r, m) for r, m in gathered if not isinstance(m, str)),
            key=lambda p: p[0],
        )
        errors = [f"rank {r}: {m}" for r, m in gathered if isinstance(m, str)]
        if parts:
            worst = max(parts, key=lambda p: np.nan_to_num(p[1][0], nan=np.inf).max())
            extra["max_error"] = float(np.max(worst[1][0]))
            extra["recall_local"] = float(min(p[1][1].min() for p in parts))
            line = (
                f"max |s - ref| / max |ref| {extra['max_error']:.3g} (rank "
                f"{worst[0]}), local top-{SELECT_K} recall min "
                f"{extra['recall_local']:.4f}"
            )
            if len(parts) == RANKS and parts[0][1][3] is not None:
                every = [
                    np.concatenate(x, axis=1) for x in zip(*(p[1][3] for p in parts))
                ]
                extra["recall_global"] = float(
                    valid_topk_recall(*every, SELECT_K).min()
                )
                line += f", global {extra['recall_global']:.4f}"
            harness.log(
                f"{tag} check: {line}" + (f"; {'; '.join(errors)}" if errors else "")
            )
        if isinstance(mine, str):
            raise RuntimeError(mine)
        return bool(mine[0].max() <= bound and mine[2])

    return check


def _score_setup(rows, tokens, variant, plan: ScorePlan, lengths=()) -> Setup:
    import jax
    import ml_dtypes
    import torch
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3.dsa import INDEX_DIM, INDEX_HEADS
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    select, keyed = plan.layout == "select", plan.emit == "keys"
    per_row, decode = plan.pages == "per-row", plan.pages == "decode"
    rank = bridge.kernel_mesh(AXIS, RANKS)[1]
    inputs = _score_inputs(rows, tokens, "per-row" if decode else plan.pages)
    if per_row or decode:
        own = block_table(rows, plan.blocks, SCORE_SEED)
        table = own[score_stream(rows, plan.blocks, plan.pages_per_step)]
        blocks = pack_index_blocks(
            inputs.key_bytes.reshape(-1, INDEX_DIM),
            inputs.exponent.reshape(-1),
            plan.block_tokens,
        )
        pool = np.empty_like(blocks)
        pool[own.reshape(-1)] = blocks
        index = np.arange(INDEX_HEADS * rows).reshape(rows, INDEX_HEADS)
    else:
        pool = pack_index_blocks(inputs.key_bytes, inputs.exponent, plan.block_tokens)
        index = score_query_rows(rows, plan.layout)
    queries = np.zeros((INDEX_HEADS * rows, INDEX_DIM), np.float32)
    queries[index] = inputs.queries
    weights = np.zeros((INDEX_HEADS * rows, LANES), np.float32)
    weights[index] = inputs.weights[..., None]
    if decode:  # the scorer's [R, 32, 128]
        queries = queries.reshape(rows, INDEX_HEADS, INDEX_DIM)
        weights = weights.reshape(rows, INDEX_HEADS, LANES)
    if variant == "fp8q":
        queries = _upload_bits(
            queries.astype(ml_dtypes.float8_e4m3fn).view(np.uint8),
            torch.float8_e4m3fn,
        )
    else:
        queries = bridge.upload(queries.astype(ml_dtypes.bfloat16))
    padded = padded_rows(rows)
    groups = rows // tile_group_rows(rows, "score")
    if decode:
        shape = dsa_decode.score_shape(tokens, rows)
    elif select:
        planes = tokens // (virtual_groups(rows, tokens) * LANES)
        shape = (planes, padded, LANES)
    else:
        shape = (padded, tokens)
    tag = f"{variant}_b{plan.block_tokens}" + ("_full" if plan.full else "")
    tag += f"_d{plan.dot_tokens}" if plan.dot_tokens != SCORE_DOT_TOKENS else ""
    if plan.dot_groups != min(SCORE_DOT_GROUPS, groups):
        tag += f"_g{plan.dot_groups}"
    tag += (
        f"_p{plan.prefetch}" if plan.prefetch != SCORE_PREFETCH and not decode else ""
    )
    tag += ("_vsel" if select else "") + ("_keys" if keyed else "")
    tag += "_pr" if per_row else ""
    tag += f"_k{plan.pages_per_step}" if plan.pages_per_step != 1 else ""
    tag += {"compute-only": "_conly", "dma-only": "_donly"}.get(plan.work, "")
    tag += "_dec" if decode else ""
    tag += f"_w{plan.sequence_rows}" if plan.sequence_rows > 1 else ""
    tag += "_whole" if plan.decode_calls == "whole" else ""
    extra = {
        "block_tokens": plan.block_tokens,
        "writeback": "full" if plan.full else "slice",
        "dot_tokens": plan.dot_tokens,
        "dot_groups": plan.dot_groups,
        "prefetch": plan.prefetch,
        "index_bytes": int(pool.nbytes),
        "mxu_flops": 2 * rows * INDEX_HEADS * tokens * INDEX_DIM,
        "score_vmem_bytes": int(np.prod(shape)) * 4,
        "layout": plan.layout,
        "emit": plan.emit,
        "score_pages": plan.pages,
        "pages_per_step": plan.pages_per_step,
        "score_work": plan.work,
        "sequence_rows": plan.sequence_rows,
    }
    if per_row:
        extra["page_order"] = "block-major" if plan.block_major else "row-major"
    tables, counts = [], None
    if keyed:
        contexts = _select_lengths(rows, tokens, lengths)
        local = {
            n: local_counts(n, tokens, plan.block_tokens)[rank] for n in set(contexts)
        }
        counts = np.asarray([local[n] for n in contexts], np.int32)
        tables.append(counts)
    if per_row:
        tables.append(table)
    if decode:
        extra["page_order"] = "decode"
        # Every rank holds T local tokens of each row; the new token is on rank 31.
        position = RANKS * tokens - 1
        with jax.default_device(jax.devices("cpu")[0]):
            length = int(
                dsa_decode.local_length(position, rank, plan.block_tokens, RANKS)
            )
        counts = [length] * rows
        # Verify sequences: rows at consecutive positions, the last row's the
        # one above (timing only).
        width = plan.sequence_rows
        positions = np.asarray(
            [position - width + 1 + r % width for r in range(rows)], np.int32
        )
        tables += [own.reshape(-1), positions]
    tables = [bridge.upload(smem_table(t)) for t in tables]
    if decode:
        # The new token's handoff (dsa_cache.index_key_rows): the row's last
        # local key, which on rank 31 is the new token's slot.
        new_values = np.zeros((rows, 3, INDEX_DIM), ml_dtypes.bfloat16)
        last = inputs.key_bytes[:, tokens - 1].view(ml_dtypes.float8_e4m3fn)
        new_values[:, 0] = last.astype(ml_dtypes.bfloat16)
        exponent = inputs.exponent[:, tokens - 1, None].astype(np.float32)
        new_values[:, 1] = exponent.astype(ml_dtypes.bfloat16)
        operands = (
            *tables,
            bridge.upload(pool[:, None]),
            queries,
            bridge.upload(weights),
            bridge.upload(new_values),
            bridge.upload(np.zeros(shape, np.float32)),
        )
    else:
        operands = (
            *tables,
            queries,
            bridge.upload(weights),
            bridge.upload(pool),
            bridge.upload(np.zeros(shape, np.int32 if keyed else np.float32)),
        )
    if variant == "bf16q":
        bound = SCORE_KEYS_TOLERANCE if keyed else SCORE_TOLERANCE
    else:
        bound = FP8_SCORE_BOUND
    invalid = plan.work != "full" or plan.sequence_rows > 1

    def make_body(form: ScorePlan, name: str):
        return (
            _decode_score_body(form, name, rank) if decode else _score_body(form, name)
        )

    return Setup(
        make_body(plan, tag),
        operands,
        (P(AXIS),) * len(operands),
        (P(AXIS),),
        variant=tag,
        check=None
        if invalid
        else _score_check(
            rows,
            inputs,
            bound,
            extra,
            f"score_r{rows}_t{tokens}_{tag}",
            layout="decode" if decode else plan.layout,
            emit=plan.emit,
            counts=counts,
        ),
        invalid=invalid,
        check_body=None
        if plan.full or invalid
        else make_body(replace(plan, full=True), f"{tag}_full"),
        extra=extra,
    )


def score_plans(rows: int, tokens: int, args) -> list[ScorePlan]:
    """The distinct kernels the flags ask for at one point: dots of every
    ``--dot-tokens`` that divides the block (each must divide one of the
    blocks), ``--dot-groups`` capped at the rows' groups (one with per-row
    pages: a page's dot is one row's), ``--prefetch`` capped at the streamed
    blocks, for every ``--layout``, ``--emit`` and ``--score-pages``; per-row
    pages also for every ``--pages-per-step`` (row-major steps must divide a
    row's blocks) and ``--score-work`` (shared pages: one page a step, full
    work). Decode pages take the decode scorer's own form: ``select``,
    ``scores``, one page a step, full work, its ring (prefetch: its
    ``RING_DEPTH`` pages in flight) and its dots (128 keys, one row's)."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels.dsa_decode import RING_DEPTH

    if any(tokens % block for block in args.block_tokens):
        raise ValueError(
            f"score: {tokens} local tokens must be whole blocks of {args.block_tokens}"
        )
    if min(args.dot_tokens + args.dot_groups + args.prefetch) < 1:
        raise ValueError(
            "score: --dot-tokens, --dot-groups and --prefetch must be positive"
        )
    unfit = [
        width
        for width in args.dot_tokens
        if width % LANES or all(block % width for block in args.block_tokens)
    ]
    if unfit:
        raise ValueError(
            f"score: keys a dot {unfit} are not whole {LANES}-key rows dividing "
            f"any block of {args.block_tokens}"
        )
    for name, values, allowed in (
        ("--layout", args.layout, SCORE_LAYOUTS),
        ("--emit", args.emit, SCORE_EMITS),
        ("--score-pages", args.score_pages, SCORE_PAGES),
        ("--pages-per-step", args.pages_per_step, SCORE_PAGES_PER_STEP),
        ("--score-work", args.score_work, SCORE_WORKS),
    ):
        if not values or set(values) - set(allowed):
            raise ValueError(f"score: {name} {values} not from {list(allowed)}")
    groups = rows // tile_group_rows(rows, "score")
    if "select" in args.layout:
        planes = tokens // (virtual_groups(rows, tokens) * LANES)
        if planes & (planes - 1):
            raise ValueError(
                f"score: --layout select needs a power of two number of "
                f"{LANES}-token planes, not {planes} ({tokens} local tokens)"
            )
    plans = {}
    for block in args.block_tokens:
        for dot_tokens in args.dot_tokens:
            if block % dot_tokens:
                continue
            for dot_groups, prefetch in itertools.product(
                args.dot_groups, args.prefetch
            ):
                for layout, emit, pages, per_step, work in itertools.product(
                    args.layout,
                    args.emit,
                    args.score_pages,
                    args.pages_per_step,
                    args.score_work,
                ):
                    per_row, decode = pages == "per-row", pages == "decode"
                    if decode:
                        form = (layout, emit, per_step, work)
                        if form != ("select", "scores", 1, "full"):
                            raise ValueError(
                                "score: --score-pages decode times the decode scorer: "
                                "--layout select --emit scores, one page a step, "
                                "full work"
                            )
                    elif not per_row:
                        if work != "full":
                            continue
                        per_step = 1
                    elif (tokens // block) % per_step and not block_major(
                        rows, per_step
                    ):
                        raise ValueError(
                            f"score: --pages-per-step {per_step} must divide a "
                            f"row's {tokens // block} blocks of {block} tokens "
                            f"(block-major {SUBLANES} needs {SUBLANES} rows or more)"
                        )
                    stream = tokens // block * (rows if per_row else 1)
                    plan = ScorePlan(
                        rows,
                        tokens,
                        block,
                        SCORE_DOT_TOKENS if decode else dot_tokens,
                        1 if per_row or decode else min(dot_groups, groups),
                        RING_DEPTH if decode else min(prefetch, stream),
                        args.writeback == "full",
                        layout,
                        emit,
                        pages,
                        per_step,
                        work,
                    )
                    for width in (
                        getattr(args, "sequence_rows", [1]) if decode else (1,)
                    ):
                        if rows % width:
                            raise ValueError(
                                f"score: {rows} rows are not whole sequences of {width}"
                            )
                        plans.setdefault(
                            replace(
                                plan,
                                sequence_rows=width,
                                decode_calls=getattr(args, "decode_calls", "chunks"),
                            ),
                            None,
                        )
    return list(plans)


def _build_score(rows: int, tokens: int, args) -> list[Setup]:
    """The score setups at one point, one per distinct plan x ``--variant``. The
    variant tag is ``{bf16q|fp8q}_b{block tokens}`` plus ``_full``
    (``--writeback full``), ``_d{N}`` (keys a dot other than 128), ``_g{N}`` (row
    groups a dot other than one), ``_p{N}`` (``--prefetch N`` other than 1),
    ``_vsel`` (``--layout select``), ``_keys`` (``--emit keys``), ``_pr``
    (``--score-pages per-row``), ``_k{K}`` (``--pages-per-step K`` other than
    1), ``_conly`` / ``_donly`` (``--score-work compute-only`` /
    ``dma-only``, timing only) and ``_dec`` (``--score-pages decode``, ``bf16q``
    only: the decode scorer's queries; no ``_p``, its ring is fixed).
    ``--lengths`` sets the rows' contexts of ``--emit keys``."""
    asked = [v for v in args.variant if v in SCORE_VARIANTS]
    if "fp8q" in asked and "decode" in args.score_pages:
        raise ValueError(
            "score: --score-pages decode times the decode scorer: --variant bf16q "
            "(its queries are FP8 values held in BF16)"
        )
    variants = asked or list(SCORE_VARIANTS)
    return [
        _score_setup(rows, tokens, variant, plan, lengths=args.lengths)
        for plan in score_plans(rows, tokens, args)
        for variant in variants
        if plan.pages != "decode" or variant == "bf16q"
    ]


CASES = {
    "copy": CaseDef(_build_copy, per_tokens=False),
    "barrier": CaseDef(_build_barrier, per_tokens=False),
    "select": CaseDef(_build_select, per_tokens=True),
    "select_rounds": CaseDef(_build_select_rounds, per_tokens=True),
    "gather": CaseDef(_build_gather, per_tokens=True),
    "compact": CaseDef(_build_compact, per_tokens=True),
    "fold": CaseDef(_build_fold, per_tokens=True),
    "score": CaseDef(_build_score, per_tokens=True),
}


def _ints(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected comma separated integers"
        ) from None


def _case_names(text: str) -> list[str]:
    names = [part for part in text.split(",") if part]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown case {unknown}; choose from {sorted(CASES)}"
        )
    return names


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r}: expected a positive integer")
    return value


def _positive_ints(text: str) -> list[int]:
    values = _ints(text)
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected comma separated positive integers"
        )
    return values


def _int_choices(allowed: Sequence[int]) -> Callable[[str], list[int]]:
    parse = _choices([str(value) for value in allowed])
    return lambda text: [int(value) for value in parse(text)]


def _choices(allowed: Sequence[str]) -> Callable[[str], list[str]]:
    def parse(text: str) -> list[str]:
        names = [part for part in text.split(",") if part]
        unknown = [name for name in names if name not in allowed]
        if unknown or not names:
            raise argparse.ArgumentTypeError(
                f"{text!r}: choose comma separated values from {list(allowed)}"
            )
        return names

    return parse


def add_arguments(parser) -> None:
    parser.add_argument(
        "--cases",
        type=_case_names,
        default=["copy", "barrier"],
        help=f"comma separated cases, from {', '.join(sorted(CASES))}",
    )
    parser.add_argument(
        "--rows", type=_positive_ints, default=[1, 8, 32, 64], help="decode rows R"
    )
    parser.add_argument(
        "--tokens",
        type=_positive_ints,
        default=[1024, 4096, 8192, 32768],
        help="local tokens per rank T = L / 32 (cases that depend on it)",
    )
    parser.add_argument(
        "--n1", type=_positive, default=8, help="iterations of the short loop"
    )
    parser.add_argument(
        "--n2",
        type=int,
        default=0,
        help="iterations of the long loop (default 72; 264 for a case under 20 us)",
    )
    parser.add_argument("--repeats", type=_positive, default=5)
    parser.add_argument(
        "--variant",
        type=_choices((*SELECT_BITS, *FOLD_VARIANTS, *SCORE_VARIANTS)),
        default=[],
        help="select: digit widths (radix4, radix8); gather, compact: DMA "
        "granularity (row, tile, page); fold: the token stream (row, tile, page, "
        "dense, none); score: query dtype (bf16q, fp8q); a case ignores the "
        "others' names, and none given means all of its own",
    )
    select = parser.add_argument_group("select")
    select.add_argument(
        "--exchange",
        type=_choices(SELECT_EXCHANGES),
        default=["flat", "host"],
        help="select: count exchanges; none and handshake are timing only; merge "
        "(R <= 8) all-gathers every rank's local top-k candidates once, and a grid "
        "with merge and --rows above 8 stops with an error",
    )
    select.add_argument(
        "--ties",
        type=_choices(("on", "off")),
        default=["on"],
        help="select: off runs the rounds without the tie step (timing only); for "
        "merge it also skips the all-gather, the tied ids' select and the local tie "
        "round",
    )
    select.add_argument(
        "--histogram",
        type=_choices(SELECT_HISTOGRAMS),
        default=["cumulative"],
        help="select: per digit value, compare-and-sum keys at or above it "
        "(cumulative), or count it in 8-bit counters packed four to a lane",
    )
    select.add_argument(
        "--tie-skip",
        type=_choices(("off", "on")),
        default=["off"],
        help="select: on runs the tie round only when a row's ties at the "
        "threshold overflow its quota; with --ties off or --exchange merge in the "
        "grid the run stops with an error (the fold grid skips its invalid points "
        "instead)",
    )
    select.add_argument(
        "--row-sum",
        type=_choices(SELECT_ROW_SUMS),
        default=["after"],
        help="select, below eight rows: sum a row's virtual rows' counts after "
        "the exchange, or before it (one 4 KiB packet)",
    )
    select.add_argument("--page-tokens", type=_positive, default=256)
    select.add_argument(
        "--digit-bits",
        type=_ints,
        default=[0],
        help="select_rounds: digit widths at eight rows (0: the config's form)",
    )
    select.add_argument(
        "--rounds",
        type=_ints,
        default=[0],
        help="select_rounds: 0 the config's rounds, N exactly N rounds of the stop "
        "form (timing only)",
    )
    select.add_argument(
        "--scores",
        choices=("normal", "ties", "exact", "indexer"),
        default="normal",
        help="select: random normal scores, rounded ones with many ties, ones "
        "whose threshold ties fill or overflow the quota (exact), or the score "
        "kernel's (indexer)",
    )
    select.add_argument(
        "--lengths",
        type=_ints,
        default=[],
        help="select, score --emit keys: per-row contexts, cycled over the rows "
        "(default and 0: every slot of every rank)",
    )
    gather = parser.add_argument_group("gather")
    gather.add_argument(
        "--pattern",
        type=_choices(GATHER_PATTERNS),
        default=list(GATHER_PATTERNS),
        help="gather, compact, fold: selected token positions (uniform, recent)",
    )
    gather.add_argument(
        "--selected",
        type=_positive_ints,
        default=list(GATHER_SELECTED),
        help="gather, compact, fold: selected local tokens per row S",
    )
    gather.add_argument(
        "--kv-dtype",
        type=_choices(GATHER_DTYPES),
        default=list(GATHER_DTYPES),
        help="gather, compact, fold: pool pages (bf16: 2 tokens per sublane row, "
        "fp8: 4)",
    )
    gather.add_argument(
        "--inflight",
        type=_positive_ints,
        default=[GATHER_INFLIGHT],
        help="gather: DMAs in flight before the first wait (powers of two)",
    )
    gather.add_argument(
        "--wait-group",
        type=_positive_ints,
        default=[1],
        help="gather: consecutive ring slots sharing a semaphore, waited with "
        "one byte-counted wait (powers of two)",
    )
    gather.add_argument(
        "--verify",
        choices=("on", "off"),
        default="on",
        help="gather: also run a consuming kernel and check the gathered words",
    )
    compact = parser.add_argument_group("compact")
    compact.add_argument(
        "--capacity",
        type=_positive_ints,
        default=[COMPACT_CAPACITY],
        help="compact: list positions per virtual row C (multiples of 128, at "
        "most 512)",
    )
    compact.add_argument(
        "--smem-path",
        type=_choices(SMEM_PATHS),
        default=["dma"],
        help="compact: each block's lists go VMEM -> SMEM by one DMA (dma) or "
        "through an HBM buffer (hbm)",
    )
    compact.add_argument(
        "--row-mix",
        choices=("on", "off"),
        default="off",
        help="compact, fold: rows select S, no, one and every token, cycled "
        "(off: every row S)",
    )
    fold = parser.add_argument_group("fold")
    fold.add_argument(
        "--fold-tokens",
        type=_positive_ints,
        default=[FOLD_TOKENS],
        help="fold: tokens a step N (multiples of 128; page: whole pages)",
    )
    fold.add_argument(
        "--fold-compaction",
        type=_choices(FOLD_COMPACTIONS),
        default=["none"],
        help="fold: tile and page streams fold masked (none) or move their "
        "selected tokens into a dense buffer first, by one-hot dots (onehot) or "
        "one token at a time (copy)",
    )
    fold.add_argument(
        "--fold-compute",
        type=_choices(("on", "off")),
        default=["on"],
        help="fold: off streams the chunks without folding them (timing only)",
    )
    score = parser.add_argument_group("score")
    score.add_argument(
        "--block-tokens",
        type=_int_choices(SCORE_BLOCK_TOKENS),
        default=list(SCORE_BLOCK_TOKENS),
        help="score: tokens per index block (256: BF16 pages, 512: FP8 pages)",
    )
    score.add_argument(
        "--writeback",
        choices=SCORE_WRITEBACKS,
        default="slice",
        help="score: the timed body copies the last 128 tokens' scores to HBM "
        "(slice) or all of them (full, the checked body)",
    )
    score.add_argument(
        "--dot-tokens",
        type=_positive_ints,
        default=[SCORE_DOT_TOKENS],
        help="score: keys per MXU dot (multiples of 128 dividing the block)",
    )
    score.add_argument(
        "--dot-groups",
        type=_positive_ints,
        default=[SCORE_DOT_GROUPS],
        help="score: groups of eight rows per MXU dot",
    )
    score.add_argument(
        "--prefetch",
        type=_positive_ints,
        default=[SCORE_PREFETCH],
        help="score: index blocks in flight while one is scored (1: double buffered)",
    )
    score.add_argument(
        "--layout",
        type=_choices(SCORE_LAYOUTS),
        default=["padded"],
        help="score: the score buffer: [R_pad, T] (padded) or the select's virtual "
        "rows [planes, V, 128] (select)",
    )
    score.add_argument(
        "--emit",
        type=_choices(SCORE_EMITS),
        default=["scores"],
        help="score: FP32 scores, or the select's int32 keys, INT32_MIN past each "
        "row's local tokens (keys, --lengths)",
    )
    score.add_argument(
        "--score-pages",
        type=_choices(SCORE_PAGES),
        default=["shared"],
        help="score: every row scores one key set (shared), or its own blocks "
        "through a block table (per-row), or through the decode kernel's scorer "
        "(decode: --layout select --emit scores, bf16q)",
    )
    score.add_argument(
        "--pages-per-step",
        type=_int_choices(SCORE_PAGES_PER_STEP),
        default=[1],
        help="score --score-pages per-row: pages a loop step scores, consecutive "
        "blocks of one row, or at 8 from eight rows one block of eight rows",
    )
    score.add_argument(
        "--sequence-rows",
        type=lambda text: [int(v) for v in text.split(",") if v],
        default=[1],
        help="score --score-pages decode: rows of a verify sequence (sharing its "
        "first row's pages; above 1 timing only)",
    )
    score.add_argument(
        "--decode-calls",
        choices=("chunks", "whole"),
        default="chunks",
        help="score --score-pages decode: the scorer's calls, 8-row chunks (B2: one "
        "row) or every row in one call, as the decode stack's score_and_select "
        "makes them",
    )
    score.add_argument(
        "--score-work",
        type=_choices(SCORE_WORKS),
        default=["full"],
        help="score --score-pages per-row: compute-only scores the pages left in "
        "the ring without DMAs, dma-only streams them unscored (timing only)",
    )


def run(args, records: harness.Records) -> None:
    from vllm_torchtpu.megakernel import bridge

    try:
        check_lengths(args.n1, args.n2 or DEFAULT_N2, args.repeats)
    except ValueError as exc:
        harness.fail(str(exc))
    bridge.kernel_mesh(AXIS, RANKS)
    results = []
    failures = []
    for name in args.cases:
        definition = CASES[name]
        for tokens in args.tokens if definition.per_tokens else [0]:
            for rows in args.rows:
                for setup in definition.build(rows, tokens, args):
                    tag = f"{name}_r{rows}" + (f"_t{tokens}" if tokens else "")
                    tag += f"_{setup.variant}" if setup.variant else ""
                    harness.sync()
                    n2, timing = measure(tag, setup, args.n1, args.n2, args.repeats)
                    check, failed = ("invalid" if setup.invalid else None), {}
                    if setup.check is not None:
                        outputs = call_body(
                            tag + ("_check" if setup.check_body else ""),
                            setup.check_body or setup.body,
                            setup.operands,
                            setup.in_specs,
                            setup.out_specs,
                            args.n1,
                        )
                        failed = reduce_check(setup.check, outputs)
                        check = not failed
                    case = Case(
                        name, rows, tokens, setup.variant, round(timing.us, 3), check
                    )
                    records.add(
                        **asdict(case),
                        **setup.extra,
                        n1=args.n1,
                        n2=n2,
                        repeats=args.repeats,
                        rank_us_min=round(timing.rank_us[0], 3),
                        rank_us_median=round(timing.rank_median, 3),
                        rank_us_max=round(timing.rank_us[-1], 3),
                        spread_flag=timing.spread_flag,
                        failed_ranks=sorted(failed),
                    )
                    results.append(case)
                    if failed:
                        failures.append((case, failed))
                    harness.log(
                        f"{name} R={rows}"
                        + (f" T={tokens}" if tokens else "")
                        + (f" {setup.variant}" if setup.variant else "")
                        + f": {timing.us:.2f} us per iteration, check {check}"
                        + (f", failed on ranks {sorted(failed)}" if failed else "")
                    )
    harness.log(
        "summary: "
        + "; ".join(
            f"{c.case}@R{c.rows}"
            + (f"/T{c.tokens}" if c.tokens else "")
            + f"={c.us:.2f}us"
            for c in results
        )
    )
    if failures:
        harness.fail(
            "check failed: "
            + "; ".join(
                f"{c.case} R={c.rows} T={c.tokens} on ranks {failed}"
                for c, failed in failures
            )
        )
