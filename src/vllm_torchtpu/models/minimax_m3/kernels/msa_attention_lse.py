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
"""`msa_attention` that also returns each row's log-sum-exp.

Decode context parallelism (`msa_dcp`) attends each token over the share of
its selected pages one rank holds and merges the ranks' partial outputs by
their log-sum-exps. The kernel is `msa_attention._kernel_v2` with one more
output; the work list, page unpacking and masking are shared with it.
"""

import functools
import logging
import math
import os
import re
import types

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import tight_vmem
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention, msa_fp4
from vllm_torchtpu.models.minimax_m3.kernels import page_table as page_table_lib

logger = init_logger(__name__)

# Page counts a kernel step may run at besides the full group (`_kernel`).
# 24 serves 32-page groups (32, 48: 64-page ones): a step costs per score
# column, filled or not, and a 17-24-page remainder is common there
# (standalone 4-bit 8-token tiles at served-c48-like dispersion: 2.00 -> 1.83
# ms, j-492a1bb9).
_STEP_WIDTHS = (4, 8, 16, 24, 32, 48)
# With TPU_MINIMAX_M3_MSA_PREFILL_WIDTH12=1, prefill-sized calls
# (`msa_dcp.msa_attend_dcp`) also run 9-12-page groups at a 12-page width
# instead of the full 16: a quarter fewer score columns for those groups.
PREFILL_STEP_WIDTHS = (
    (4, 8, 12)
    if os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_WIDTH12", "0") == "1"
    else _STEP_WIDTHS
)
# TPU_MINIMAX_M3_MSA_PREFILL_WIDTHS=<w,w,...> (e.g. "2,4,6,8,12"; "/" also
# separates, for env lists that are themselves comma-separated) sets the
# prefill step widths outright (below 16, ascending); unset keeps the above.
if os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_WIDTHS"):
    PREFILL_STEP_WIDTHS = tuple(
        sorted(
            int(w)
            for w in os.environ["TPU_MINIMAX_M3_MSA_PREFILL_WIDTHS"]
            .replace("/", ",")
            .split(",")
        )
    )
# With TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE=<slots> (0 = off), prefill-sized
# calls on 4-bit caches copy each distinct page of a call once into a VMEM
# cache of that many slots (~20 KB each) instead of once per tile holding it
# (`_page_cache`). A call with more distinct pages, or a tile union over
# `PAGE_CACHE_TILE_PAGES`, streams per group as before, in the same kernel.
PREFILL_PAGE_CACHE_SLOTS = int(
    os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE", "0") or 0
)
# Tile unions past this many pages make a call stream (only this many entries of
# each union are sorted). At DCP 2 a 16-token tile's union is ~20 pages and ~5%
# of tiles pass 32, so nearly every 2,048-token call has one:
# TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_TILE_PAGES=64 keeps those calls cached.
PAGE_CACHE_TILE_PAGES = int(
    os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_TILE_PAGES", "32") or 32
)
# With TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_BF16=1 (and a page cache), the
# cached 4-bit pages are unpacked to bf16 once, after the cache fill, into a
# second VMEM cache (64 KB a slot), so a group reads bf16 rows instead of
# unpacking each page again for every tile that holds it (~17 tiles a page).
PREFILL_PAGE_CACHE_BF16 = (
    os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_BF16", "0") == "1"
)
# With TPU_MINIMAX_M3_MSA_PREFILL_ROW_SPLIT=<n> (1 = off), prefill-sized calls
# run each group's scores, softmax and p @ v in `n` independent parts of the
# tile's query heads, so one part's softmax can be scheduled beside the next
# part's QK matmul. Rows are independent: the results are bitwise the same.
# With TPU_MINIMAX_M3_MSA_PREFILL_BLOCK_Q=8 (default 16), prefill-sized calls
# tile 8 tokens instead of 16: half the rows a group, each over a smaller page
# union. Not bitwise: a row's scores then sit in other columns of the step, so
# sums over them may round in another order.
PREFILL_BLOCK_Q = int(os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_BLOCK_Q", "0") or 0)
# TPU_MINIMAX_M3_MSA_PREFILL_PAGES_PER_STEP=<n> (default 16): pages a prefill
# group holds (a tile's union runs in ceil(union / n) groups). Not bitwise
# when it changes how a tile's pages split into groups.
PREFILL_PAGES_PER_STEP = int(
    os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_PAGES_PER_STEP", "0") or 0
)
PREFILL_ROW_SPLIT = int(os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_ROW_SPLIT", "1") or 1)
if PREFILL_PAGE_CACHE_SLOTS:
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE=%d: 4-bit MSA prefill attention "
        "page cache of %d slots.",
        PREFILL_PAGE_CACHE_SLOTS,
        PREFILL_PAGE_CACHE_SLOTS,
    )
if PAGE_CACHE_TILE_PAGES != 32:
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_TILE_PAGES=%d: MSA prefill page cache "
        "sorts %d entries of each tile union.",
        PAGE_CACHE_TILE_PAGES,
        PAGE_CACHE_TILE_PAGES,
    )
if PREFILL_PAGE_CACHE_BF16:
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_BF16=1: MSA prefill attention page "
        "cache unpacked to bf16 once per call."
    )
if PREFILL_BLOCK_Q:
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_BLOCK_Q=%d: MSA prefill attention tiles of %d tokens.",
        PREFILL_BLOCK_Q,
        PREFILL_BLOCK_Q,
    )
if PREFILL_PAGES_PER_STEP:
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGES_PER_STEP=%d: MSA prefill attention groups of "
        "%d pages.",
        PREFILL_PAGES_PER_STEP,
        PREFILL_PAGES_PER_STEP,
    )
if PREFILL_ROW_SPLIT != 1:
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_ROW_SPLIT=%d: MSA prefill attention groups run "
        "in %d parts of the query heads.",
        PREFILL_ROW_SPLIT,
        PREFILL_ROW_SPLIT,
    )
if os.environ.get("TPU_MINIMAX_M3_MSA_PREFILL_WIDTHS"):
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_WIDTHS: MSA prefill attention step widths %s.",
        PREFILL_STEP_WIDTHS,
    )
elif PREFILL_STEP_WIDTHS != _STEP_WIDTHS:
    logger.info(
        "TPU_MINIMAX_M3_MSA_PREFILL_WIDTH12=1: MSA prefill attention step widths %s.",
        PREFILL_STEP_WIDTHS,
    )
# Tiles per grid step: a step runs the groups of this many tiles in a loop.
# One step per group paid a per-step cost worth ~2-3% of the call (pool48
# selections, 1-2 groups a tile; bench j-caec1354).
TILES_PER_STEP = 4
# Groups whose pages are in flight ahead of the running one: 4-bit groups
# (a page and a scale page each) gain from 2 (bench j-caec1354: c64 shared
# 2206 -> 2026, forced 2150 -> 2075 us per call), 8/16-bit ones do not.
_FP4_PREFETCH = 2
# Page copies a group's start loop issues per iteration (`_kernel`).
_START_UNROLL = 4
_LOG2_E = math.log2(math.e)
_LN_2 = math.log(2.0)


def _fp4_rows(kv_ref) -> jax.Array:
    """Unscaled bf16 rows ``[S, 2D]`` of a 4-bit page, as `msa_attention._fp4_page`."""
    words = kv_ref.bitcast(jnp.uint32)[0, :, 0, :]
    return pltpu.bitcast(words, jnp.int4).astype(jnp.bfloat16)


def _fp4_scales(scale_ref, k_row, v_row):
    """K / V scales of a page as `msa_attention._fp4_page` reads them."""
    if isinstance(k_row, int):
        page = scale_ref[0]
        sk, sv = page[k_row : k_row + 1], page[v_row : v_row + 1]
    else:
        sk = scale_ref[0, pl.ds(k_row, 1), :]
        sv = scale_ref[0, pl.ds(v_row, 1), :]
    return jnp.where(jnp.isfinite(sk), sk, 0.0), jnp.where(jnp.isfinite(sv), sv, 0.0)


def _fp4_ints(kv_ref) -> jax.Array:
    """Int4 rows ``[S, 2D]`` of a 4-bit page in slot order (unscaled)."""
    words = kv_ref.bitcast(jnp.uint32)[0, :, 0, :]
    return pltpu.bitcast(words, jnp.int4)


# `prefill_precision_levers` "pv8": V enters the MXU as its exact int4 values
# in fp8, and the probabilities carry the V scale times _PV8_UNIT (lifting
# p * scale out of fp8's subnormal range), capped so that p * scale * unit
# stays within fp8's range (V scales above _FP8_MAX / _PV8_UNIT = 7, |v| ~ 50,
# would saturate); the output divides the unit out.
_PV8_UNIT = 64.0
_FP8_MAX = 448.0
PREFILL_PRECISION_ENV = "TPU_MINIMAX_M3_MSA_PREFILL_PRECISION"
_PRECISION_LEVERS = ("qk8", "pv8")
# TPU_MINIMAX_M3_MSA_PREFILL_FP8_OPT levers of 8-bit (fp8) pages on the
# transposed prefill path (`_kernel.attend_t`), default off, not bitwise:
# "exp16": p = exp2(s - m) in bf16 (p enters p @ v in bf16 anyway; l sums the
# rounded p in f32). Standalone fp8 calls on served DCP 2 selections, block_q
# 32 and 32-page groups: -2.6% (8,192 tokens), out rel-rms 3e-3 (j-662afb0e).
FP8_OPT_ENV = "TPU_MINIMAX_M3_MSA_PREFILL_FP8_OPT"
# "bias": the per-score cmp + select page mask becomes a lane-row add (0 or
# -inf per token, from the page's hit / limit) on fully visible or unseen
# pages; only a page some token sees in part (its local page) or an empty
# slot keeps the select, under a scalar branch. Bitwise (finite pages).
# "lanemask": the page mask compared once per 128-lane vreg of rows (4 heads
# x 32 tokens at block_q 32) and reused by every head group (bitwise).
# "vt": each page's V rows transposed as they unpack, so p @ v runs as
# v^T [D, keys] @ p [keys, rows] with no transposed operand in the dot.
# "lmxu": the softmax row sums come out of the p @ v dot (8 rows of ones
# under v^T) instead of a VALU add per score vreg; l sums the bf16 p the dot
# multiplies.
# "mdot": no per-score running max or subtraction. The group's shift m comes
# from a small dot of the queries with the group's first page (its keys past
# the visible slots zeroed): m = max(m_prev, max of those raw scores +
# _MDOT_HEADROOM), rounded to bf16, and the main QK dot runs on a 256-wide
# contraction whose extra column is 1 for every key and -m for every row, so
# it yields s - m directly. Exact softmax for any m; f32 / bf16 keep p in
# range while the true row max stays within ~2^7 of m.
# "msub": the same first-page shift, subtracted per score on the VALU (128-wide
# contraction); only the per-score running max goes.
_FP8_LEVERS = ("exp16", "bias", "lanemask", "vt", "lmxu", "mdot", "msub")
_MDOT_HEADROOM = 16.0


def fp8_prefill_levers(spec: str | None = None) -> frozenset:
    """Levers of ``envs.TPU_MINIMAX_M3_MSA_PREFILL_FP8_OPT`` (or `spec`); read
    when the program is traced."""
    if spec is None:
        from vllm_torchtpu import envs

        spec = envs.TPU_MINIMAX_M3_MSA_PREFILL_FP8_OPT
    levers = frozenset(x.strip() for x in spec.replace("/", ",").split(",") if x.strip())
    bad = levers - set(_FP8_LEVERS)
    if bad:
        raise ValueError(f"{FP8_OPT_ENV}: unknown levers {sorted(bad)}")
    if levers:
        _proof(FP8_OPT_ENV, f"fp8 MSA prefill attention levers {sorted(levers)}")
    return levers


def prefill_precision_levers(spec: str | None = None) -> frozenset:
    """Levers of ``envs.TPU_MINIMAX_M3_MSA_PREFILL_PRECISION`` (or `spec`) for
    4-bit prefill calls with the page cache; read when the program is traced."""
    if spec is None:
        from vllm_torchtpu import envs

        spec = envs.TPU_MINIMAX_M3_MSA_PREFILL_PRECISION
    levers = frozenset(x.strip() for x in spec.replace("/", ",").split(",") if x.strip())
    bad = levers - set(_PRECISION_LEVERS)
    if bad:
        raise ValueError(f"{PREFILL_PRECISION_ENV}: unknown levers {sorted(bad)}")
    if levers:
        _proof(PREFILL_PRECISION_ENV, f"MSA prefill attention levers {sorted(levers)}")
    return levers


# Max-free softmax (TPU_MINIMAX_M3_MSA_MAX_FREE=1, default off): q and k come
# out of per-head RMS norms (weight 1 + w; partial RoPE and the 4-bit Hadamard
# rotation keep norms), so |q . k| * sm_scale <= sqrt(D) * max|1 + w_q| *
# max|1 + w_k| (times the cache's key rounding), a bound the model computes
# from its loaded weights (`MiniMaxM3SparseAttention.score_bound`) and hands
# to `configure_max_free`. Under `_MAX_FREE_SCORE_BOUND` the kernels take
# p = exp2(s) against 0 instead of a running max: p never overflows (exp(75)
# leaves e^13 of f32 range for a tile's sum and p @ v), a seen row's largest p
# stays a normal float, and the max reduction, the rescale of l / acc and the
# m scratch traffic all go. Read when a program is traced (after load).
MAX_FREE_ENV = "TPU_MINIMAX_M3_MSA_MAX_FREE"
MXU_MASK_ENV = "TPU_MINIMAX_M3_MSA_MXU_MASK"
_MAX_FREE_SCORE_BOUND = 75.0
MAX_FREE = False
# With max-free (and TPU_MINIMAX_M3_MSA_MXU_MASK=1), `_kernel.attend_t` takes
# each token's mask from the QK matmul: the keys carry a head_dim-wide column
# block that is -_MASK_BIAS where a token does not see the key, the queries a
# one-hot of their token there. Needs max-free: against a running max a row
# that sees nothing would exponentiate -_MASK_BIAS - _MASK_VALUE.
MXU_MASK = False
_MASK_BIAS = 1e30
# Key norm of the cached keys over the exact ones', per KV cache dtype
# (`key_rounding`): e4m3 rounds at most 1/16; int4 rows of a Hadamard-rotated
# vector carry ~12 % rounding, taken as 1.25 (a heuristic margin: the worst
# case, every other element half a step off, is 1.476).
KEY_ROUNDING = {"int4": 1.25, "fp8": 1.0625, "bf16": 1.0}
# fp8 queries ("qk8" of `prefill_precision_levers`) round q by at most 1/16.
_QK8_ROUNDING = 1.0625


def score_bound(
    q_norm_weight, k_norm_weight, head_dim: int, key_rounding: float = 1.0
) -> float:
    """``|q . k| * head_dim**-0.5`` bound (natural log units) for per-head
    Gemma RMS norms of weights `q_norm_weight` / `k_norm_weight` (applied as
    ``1 + w``): ``sqrt(D) * max|1 + w_q| * max|1 + w_k| * key_rounding``.
    Takes numpy arrays, torch tensors or jax arrays."""
    import numpy as np

    def gain(w):
        if hasattr(w, "detach"):  # torch
            w = w.detach().float().cpu().numpy()
        return float(np.max(np.abs(1.0 + np.asarray(w, np.float32))))

    return head_dim**0.5 * gain(q_norm_weight) * gain(k_norm_weight) * key_rounding


def configure_max_free(
    bound: float, enabled: bool | None = None, mxu_mask: bool | None = None
) -> bool:
    """Sets `MAX_FREE` (and `MXU_MASK`) for a loaded model whose largest score
    bound (natural log units, every layer, after its keys' cache rounding) is
    `bound`, and returns `MAX_FREE`. `enabled` / `mxu_mask` default to
    ``envs.TPU_MINIMAX_M3_MSA_MAX_FREE`` / ``..._MXU_MASK``; a bound past
    `_MAX_FREE_SCORE_BOUND` refuses (keeps the running max)."""
    global MAX_FREE, MXU_MASK
    if enabled is None or mxu_mask is None:
        from vllm_torchtpu import envs

        if enabled is None:
            enabled = envs.TPU_MINIMAX_M3_MSA_MAX_FREE
        if mxu_mask is None:
            mxu_mask = envs.TPU_MINIMAX_M3_MSA_MXU_MASK
    MAX_FREE = bool(enabled) and bound <= _MAX_FREE_SCORE_BOUND
    MXU_MASK = MAX_FREE and bool(mxu_mask)
    if enabled:
        if MAX_FREE:
            msg = (
                f"{MAX_FREE_ENV}=1: MSA score bound {bound:.1f} <= "
                f"{_MAX_FREE_SCORE_BOUND:.1f}, max-free softmax on (MXU mask {MXU_MASK})."
            )
        else:
            msg = (
                f"{MAX_FREE_ENV}=1 refused: MSA score bound {bound:.1f} > "
                f"{_MAX_FREE_SCORE_BOUND:.1f}; keeping the running max."
            )
        logger.warning(msg)
        print(msg, flush=True)  # workers may log nowhere
    return MAX_FREE


K1_ENV = "TPU_MINIMAX_M3_MSA_PREFILL_K1"
# K1 levers ("+"-separated; "1" = `_K1_DEFAULT`):
#   split<N>: 4-bit max-free prefill groups wider than N pages run their
#     softmax in ceil(W / N) equal page chunks, the chunks' p @ v and row
#     sums added in order: shorter live ranges than one [rows, W * S] score
#     slab (fewer spills). Same values; bitwise on TPU7x (bench k1 b1/b2).
#   w56: head-parallel prefill calls whose TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG
#     gives no widths run 64-page groups at widths 4/8/16/24/32/40/48/56/64
#     instead of 4/8/16/24/32/48/64 (`msa_dcp_heads.hp_prefill_kernel_kwargs`):
#     fewer empty score columns for 33-56-page unions. Bitwise (b2).
_K1_DEFAULT = "split32+w56"
K1_HP_WIDTHS = (4, 8, 16, 24, 32, 40, 48, 56)


def k1_levers(spec: str | None = None) -> frozenset:
    """TPU_MINIMAX_M3_MSA_PREFILL_K1 (or `spec`; read when a program is
    traced): empty when unset or "0", else ``{"k1", <levers>}``."""
    if spec is None:
        spec = os.environ.get(K1_ENV, "0")
    spec = spec.strip()
    if spec in ("", "0"):
        return frozenset()
    if spec == "1":
        spec = _K1_DEFAULT
    levers = {x.strip() for x in spec.split("+") if x.strip()}
    for x in levers:
        if x != "w56" and not re.fullmatch(r"split\d+", x):
            raise ValueError(f"{K1_ENV}={spec!r}: unknown lever {x!r}")
    _proof(K1_ENV, f"MSA prefill attention K1 levers {sorted(levers)}")
    return frozenset({"k1"} | levers)


K8_ENV = "TPU_MINIMAX_M3_MSA_PREFILL_K8"
# K8 levers ("+"-separated; "1" = `_K8_DEFAULT`): glue around the
# head-parallel MSA prefill attention (`msa_dcp_heads.attend_and_merge`).
# TPU7x bench (k8 mb1, served c48 T8192 DCP4 shapes, attend_and_merge per
# layer, 60 instances, all bitwise): merge -39 us, scat -5 us, cmp -3 us,
# merge+scat -41 us (of 2381 us); "1" = merge+scat.
#   merge: the decode rows' outputs go into the head-parallel output by one
#     in-place scatter (sorted, unique indices; the non-decode slots point
#     past the end and are dropped) instead of a `lax.cond` between an
#     update-slice and a scatter, whose branches copy the whole [T, Hl, D]
#     output (twice per layer: a relayout and the branch result). Same
#     rows written with the same values: bitwise.
#   cmp: `_page_cache` (packed modes) finds the distinct keys and each
#     entry's slot by compare-and-count against the one key sort instead of
#     two more sorts. The table is a function of the keys only: bitwise.
#   mergedus: like merge, as an in-place update-slice of the leading rows
#     plus a scatter that drops every row when the decode rows lead.
#   scat: like cmp, by two scatters (the distinct keys to their slots, the
#     sorted entries' slots back to their positions) instead of the two sorts.
_K8_DEFAULT = "merge+scat"
_K8_LEVERS = ("merge", "mergedus", "cmp", "scat")


def k8_levers(spec: str | None = None) -> frozenset:
    """TPU_MINIMAX_M3_MSA_PREFILL_K8 (or `spec`; read when a program is
    traced): empty when unset or "0", else the levers."""
    if spec is None:
        spec = os.environ.get(K8_ENV, "0")
    spec = spec.strip()
    if spec in ("", "0"):
        return frozenset()
    if spec == "1":
        spec = _K8_DEFAULT
    levers = {x.strip() for x in spec.split("+") if x.strip()}
    for x in levers:
        if x not in _K8_LEVERS:
            raise ValueError(f"{K8_ENV}={spec!r}: unknown lever {x!r}")
    _proof(K8_ENV, f"MSA prefill attention glue K8 levers {sorted(levers)}")
    return frozenset(levers)


def max_free_levers() -> frozenset:
    """``{"maxfree"[, "mxumask"]}`` per the configured state (empty when off),
    for a kernel's `precision`; read when the program is traced."""
    if not MAX_FREE:
        return frozenset()
    _proof(MAX_FREE_ENV, "MSA attention softmax with no running max")
    if MXU_MASK:
        _proof(MXU_MASK_ENV, "MSA prefill attention mask from the QK matmul")
        return frozenset(("maxfree", "mxumask"))
    return frozenset(("maxfree",))


def _packed_sort_fits(n: int, cache_size: int, key_bound: int) -> bool:
    """Whether `_page_cache`'s packed sorts of `n` entries fit 31 bits."""
    pos_bits = max(1, (n - 1).bit_length())
    slot_bits = (cache_size + 1).bit_length()
    return (key_bound + 1) << pos_bits < 1 << 31 and pos_bits + slot_bits <= 31


def _page_cache(
    pages: jax.Array,
    num_tiles: int,
    cache_size: int,
    width: int = 0,
    mode: str = "",
    key_bound: int = 0,
) -> jax.Array:
    """A call's unions (`msa_attention._work_list`: `num_tiles` rows, ``-1``
    padded) -> ``[unions' cache slots | distinct keys (cache_size) | count |
    fits]``: each union entry's slot among the call's distinct page keys
    (``-1`` for padding; the key itself when they do not all fit), the first
    `cache_size` distinct keys ascending, their count, and whether they all
    fit. Only the first `width` (default `PAGE_CACHE_TILE_PAGES`) of each
    union are sorted (a call with a wider union streams). Sorts only, no
    gather or scatter.

    `mode` (TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT) changes only how the sorts
    run: "unstable" drops their stability (equal keys get equal slots in any
    order); "packed" sorts one int32 per entry, the key (``< key_bound``,
    padding as ``key_bound``) above the entry position and the position above
    ``slot + 1`` (clamped at ``cache_size + 1``: past it the call streams and
    the slots are not used); "packed_search" finds the distinct keys by a
    binary search of the slot counts instead of a sort; "packed_batched"
    runs the packed distinct-key sort and the packed inverse-permutation sort
    (independent of each other) as one batched sort of two rows, so the call
    waits for two sort latencies instead of three. Callers check
    `_packed_sort_fits` for the packed modes. The table is the same bit for
    bit."""
    rows = pages.reshape(num_tiles, -1)
    width = min(width or PAGE_CACHE_TILE_PAGES, rows.shape[1])
    head = rows[:, :width].reshape(-1)
    packed_modes = ("packed", "packed_search", "packed_batched")
    if mode in packed_modes:
        n = head.shape[0]
        pos_bits = max(1, (n - 1).bit_length())
        slot_bits = (cache_size + 1).bit_length()
        no_page = key_bound
        key = jnp.where(head < 0, no_page, head)
        position = jnp.arange(n, dtype=jnp.int32)
        packed = jax.lax.sort((key << pos_bits) | position, is_stable=False)
        key = packed >> pos_bits
        position = packed & ((1 << pos_bits) - 1)
    else:
        no_page = msa_attention._NO_PAGE
        key = jnp.where(head < 0, no_page, head)
        position = jnp.arange(key.shape[0], dtype=jnp.int32)
        key, position = jax.lax.sort(
            (key, position), num_keys=1, is_stable=mode != "unstable"
        )
    new = jnp.concatenate([key[:1] != no_page, key[1:] != key[:-1]])
    new = new & (key != no_page)
    slot = jnp.cumsum(new, dtype=jnp.int32) - 1
    num_distinct = slot[-1] + 1
    by_position = None
    if mode == "packed_batched":
        # Both rows are non-negative int32 below 2^31 (`_packed_sort_fits`).
        both = jax.lax.sort(
            jnp.stack(
                [
                    jnp.where(new, key, no_page),
                    (position << slot_bits)
                    | jnp.minimum(slot + 1, cache_size + 1),
                ]
            ),
            dimension=1,
            is_stable=False,
        )
        distinct = both[0, :cache_size]
        distinct = jnp.pad(
            distinct,
            (0, cache_size - distinct.shape[0]),
            constant_values=no_page,
        )
        by_position = both[1]
    elif mode == "packed_search":
        # Slot counts ascend: the j-th distinct key is the first entry whose
        # slot reaches j.
        nth = jnp.arange(cache_size, dtype=jnp.int32)
        at = jnp.searchsorted(slot, nth, side="left", method="scan")
        distinct = jnp.where(
            nth < num_distinct, key[jnp.minimum(at, key.shape[0] - 1)], no_page
        )
    else:
        distinct = jax.lax.sort(
            jnp.where(new, key, no_page), is_stable=mode == ""
        )[:cache_size]
        distinct = jnp.pad(
            distinct,
            (0, cache_size - distinct.shape[0]),
            constant_values=no_page,
        )
    k8 = k8_levers() & {"cmp", "scat"} if mode in packed_modes else frozenset()
    if k8:
        # K8 (`k8_levers`): the distinct keys and each entry's slot from the
        # one sort above. `slot` ascends over the sorted entries and is -1
        # only before the first key, i.e. only when there is none (padding
        # sorts last); the slots are used only when every key fits.
        n = key.shape[0]
        if "cmp" in k8:
            # The j-th distinct key is the first sorted entry of slot j:
            # after count(slot < j) entries.
            nth = jnp.arange(cache_size, dtype=jnp.int32)
            at = jnp.sum(
                (slot[None, :] < nth[:, None]).astype(jnp.int32), axis=1
            )
            distinct = jnp.where(
                nth < num_distinct, key[jnp.minimum(at, n - 1)], no_page
            )
            # An entry's slot is the count of distinct keys below its key
            # (`no_page` padding counts nowhere: it is above every key).
            k = jnp.where(head < 0, no_page, head)
            slot = jnp.sum(
                (distinct[None, :] < k[:, None]).astype(jnp.int32), axis=1
            )
        else:
            # Each distinct key's first entry writes it at its slot (slots past
            # the table and repeated keys write past the end: dropped); each
            # sorted entry writes its slot at its position (a permutation).
            dst = jnp.where(new, slot, cache_size + jnp.arange(n, dtype=jnp.int32))
            distinct = jnp.full((cache_size,), no_page, jnp.int32).at[dst].set(
                key, mode="drop", unique_indices=True
            )
            slot = jnp.zeros((n,), jnp.int32).at[position].set(
                slot, unique_indices=True
            )
    elif mode in packed_modes:
        if by_position is None:
            by_position = jax.lax.sort(
                (position << slot_bits) | jnp.minimum(slot + 1, cache_size + 1),
                is_stable=False,
            )
        slot = (by_position & ((1 << slot_bits) - 1)) - 1
    else:
        _, slot = jax.lax.sort(
            (position, slot), num_keys=1, is_stable=mode != "unstable"
        )
    slot = jnp.where(head < 0, -1, slot).reshape(num_tiles, width)
    slot = jnp.pad(slot, ((0, 0), (0, rows.shape[1] - width)), constant_values=-1)
    fits = (num_distinct <= cache_size) & jnp.all(rows[:, width:] < 0)
    return jnp.concatenate(
        [
            jnp.where(fits, slot.reshape(-1), pages),
            jnp.where(distinct == no_page, 0, distinct),
            jnp.minimum(num_distinct, cache_size)[None],
            fits.astype(jnp.int32)[None],
        ]
    ).astype(jnp.int32)


def _page_cache_tables(
    unions: jax.Array,
    num_tiles: int,
    cache_size: int,
    key_bound: int,
    tile_pages: int = 0,
) -> jax.Array:
    """`_page_cache` of each call's unions ``[calls, union entries]``, the
    first `tile_pages` (default `PAGE_CACHE_TILE_PAGES`) of each tile union
    sorted, with the sort mode of TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT and, with
    TPU_MINIMAX_M3_MSA_PAGE_CACHE_WIDTHS, the narrowest table width that holds
    every call's widest tile union (a `lax.switch` in the same program; a
    union never passes the width its branch sorts, so every branch's table
    is the full width's)."""
    from vllm_torchtpu import envs

    full = min(tile_pages or PAGE_CACHE_TILE_PAGES, unions.shape[1] // num_tiles)
    mode = envs.TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT.strip().lower()
    if mode not in ("", "unstable", "packed", "packed_search", "packed_batched"):
        raise ValueError(f"TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT={mode!r}")
    spec = envs.TPU_MINIMAX_M3_MSA_PAGE_CACHE_WIDTHS.replace(",", "/")
    widths = sorted({int(w) for w in spec.split("/") if w.strip()})
    widths = [w for w in widths if 0 < w < full] + [full]
    if mode in ("packed", "packed_search", "packed_batched") and not _packed_sort_fits(
        num_tiles * full, cache_size, key_bound
    ):
        mode = "unstable"
    if mode:
        _proof(
            "TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT",
            f"MSA prefill page cache table sorts {mode}",
        )
    if len(widths) > 1:
        _proof(
            "TPU_MINIMAX_M3_MSA_PAGE_CACHE_WIDTHS",
            f"MSA prefill page cache table widths {widths} by widest union",
        )

    def build(w):
        return jax.vmap(
            functools.partial(
                _page_cache,
                num_tiles=num_tiles,
                cache_size=cache_size,
                width=w,
                mode=mode,
                key_bound=key_bound,
            )
        )

    if len(widths) == 1:
        return build(full)(unions)
    rows = unions.reshape(unions.shape[0], num_tiles, -1)
    # Unions are -1 past their pages: a width holds every union when nothing
    # past it is a page.
    branch = sum(
        jnp.any(rows[:, :, w:] >= 0).astype(jnp.int32) for w in widths[:-1]
    )
    return jax.lax.switch(branch, [build(w) for w in widths], unions)


def _kernel(
    # Scalar prefetch
    work_page_ref,  # tile page key unions (`msa_attention._work_list`)
    work_meta_ref,  # [groups] union row << _ROW_SHIFT | last and first bits
    work_count_ref,  # [groups] pages in the group (its first slots)
    page_table_ref,  # page key -> physical page
    head_ref,  # [2] KV head of the cache rows, block of query heads
    tile_offset_ref,  # [1] the call's first tile (read by the index maps only)
    # In
    q_ref,  # [H, Bq, D]
    selected_ref,  # [Bq, K] page keys, -1 padded
    local_page_ref,  # [Bq, 1]
    local_limit_ref,  # [Bq, 1] visible slots of the local page
    kv_hbm_ref,  # [pages, S / p, p, G * 2D] the whole cache, in HBM
    *refs,  # [scale pages (4-bit)], [aliased out, lse], out, lse, m, l, acc,
    # page buffers, semaphores, [scale buffers, semaphores (4-bit)]
    sm_scale: float,
    aliased: bool = False,
    fp4_rows: tuple[int, int] | str | None = None,
    block_start_ref=None,
    tiles_per_step: int = 1,
    groups_per_tile: int = 1,
    transposed: bool = False,
    step_widths: tuple[int, ...] = _STEP_WIDTHS,
    cache_ref=None,
    pages_per_step: int = msa_attention.DEFAULT_PAGES_PER_STEP,
    bf16_cache: bool = False,
    row_split: int = 1,
    tiles_per_head: int = 0,
    precision: frozenset = frozenset(),
    pair: bool = False,
):
    """`msa_attention._kernel_v2` that also writes ``lse = m + log(l)``
    (``-inf`` for a row that sees no slot; its output is 0). Scores, the
    running max ``m`` and ``l`` are in log2 units (scaled by ``sm_scale *
    log2(e)``): exp2 needs no multiply per score. 4-bit caches fold the scale
    into the K scales of the score columns; others into the queries, which
    then round to bf16 once more before the QK dot (an XLA multiply of the
    whole head-major query outside the kernel was not fused, ~40 us per layer
    in the c4 twin, j-18e7ec75).

    Each group copies a later group's pages into a free slot of the page
    ring itself, one loop iteration per page it holds: unrolled
    per-slot copies run predicated, so every step paid all slots' address
    arithmetic and checks (~800 scalar bundles of a ~2,200-bundle 4-page
    step in the LLO).

    A tile of one group (its first and last) runs a plain softmax that
    keeps nothing in the m / l / acc scratch, and a tile with no page only
    writes its zero output.

    A 4-bit cache (`fp4_rows`: static K / V scale-page rows, or ``"head"``
    for rows ``2k, 2k + 1`` of the cache-row KV head ``k``, or ``("head",
    row0)`` for rows ``row0 + 2k, row0 + 2k + 1``) copies each page's scale
    page beside it.

    Grid step ``b`` runs the groups ``block_start_ref[b] ..
    block_start_ref[b + 1] - 1`` of its `tiles_per_step` tiles in a loop
    (the last entry is the call's group count); the query, selection and
    output blocks hold those tiles.

    With `tiles_per_head`, the call's tiles run over every KV head
    (`_one_call_heads`): a group's cache-row KV head is its tile //
    `tiles_per_head`, not `head_ref[0]`."""
    fp4 = fp4_rows is not None
    # `bf16_cache`: the last scratch is the cache's pages unpacked to bf16,
    # [cache_size, S, 2D]; filled after the cache, read when it is in use.
    bfc_ref = refs[-1] if bf16_cache else None
    if bf16_cache:
        refs = refs[:-1]
    tail = refs[len(refs) - (9 if fp4 else 7) :]
    o_ref, lse_ref, m_ref, l_ref, acc_ref, buf_ref, sem_ref = tail[:7]
    scale_hbm_ref = refs[0] if fp4 else None
    scale_buf_ref, scale_sem_ref = tail[7:] if fp4 else (None, None)
    # `cache_ref` (4-bit prefill, `PREFILL_PAGE_CACHE_SLOTS`): ``[distinct page
    # keys (cache_size) | count | fits]``; the unions then hold cache slots.
    cached = cache_ref is not None
    ring = sem_ref.shape[0] - (1 if cached else 0)  # prefetched groups + running
    if cached:
        cache_size = cache_ref.shape[0] - 2
        num_cached = cache_ref[cache_size]
        use_cache = cache_ref[cache_size + 1] != 0
    num_groups = block_start_ref[pl.num_programs(0)]
    # `precision` levers (4-bit pages, `prefill_precision_levers`).
    qk8 = fp4 and "qk8" in precision
    pv8 = fp4 and "pv8" in precision
    exp16_t = not fp4 and "exp16" in precision
    bias_t = not fp4 and "bias" in precision
    lanemask_t = not fp4 and "lanemask" in precision
    lmxu_t = not fp4 and "lmxu" in precision
    vt_t = not fp4 and ("vt" in precision or lmxu_t)
    out_unit = 1.0 / _PV8_UNIT if pv8 else 1.0
    # Max-free softmax (`max_free_levers`): p = exp2(s) against 0, no m.
    # "pv8" calls keep the running max: their p enters an fp8 dot (max 448).
    max_free = "maxfree" in precision and not pv8
    # mdot / msub shift by the first page's max; max-free has no max to shift.
    mdot_t = not fp4 and "mdot" in precision and not max_free
    msub_t = not fp4 and "msub" in precision and not mdot_t and not max_free
    # A first-page shift may sit above a row's max: l can be below 1.
    mshift_t = mdot_t or msub_t
    mxu_mask_on = max_free and "mxumask" in precision
    # K1 levers (`k1_levers`) of 4-bit max-free calls.
    k1_split = next(
        (int(x[5:]) for x in precision if x.startswith("split")), 0
    ) if (fp4 and max_free and not pv8 and "k1" in precision) else 0
    # The least softmax sum a tile divides by: a row that saw no slot has 0
    # (and a zero output). Against a true max a seen row's sum is >= 1;
    # against max-free's 0 it is only positive.
    denom_floor = float(jnp.finfo(jnp.float32).tiny) if max_free else 1.0

    def group_head(w):
        if not tiles_per_head:
            return head_ref[0]
        tile = msa_attention._meta_tile(work_meta_ref[w], groups_per_tile)
        return tile // tiles_per_head

    def group(
        w,
        mode="full",
        sink=None,
        q_ref=q_ref,
        o_ref=o_ref,
        lse_ref=lse_ref,
        selected_ref=selected_ref,
        local_page_ref=local_page_ref,
        local_limit_ref=local_limit_ref,
    ):
        store_out = None
        if tiles_per_step > 1:
            tile = msa_attention._meta_tile(work_meta_ref[w], groups_per_tile)
            bq = q_ref.shape[1] // tiles_per_step
            start = tile % tiles_per_step * bq
            rows = pl.ds(pl.multiple_of(start, bq), bq)
            q_ref = q_ref.at[:, rows, :]
            packed = _merged_store_rows(bq, o_ref.shape[0], o_ref.dtype)
            if packed > 1:

                def store_out(out, part=0, parts=1, o_rows=o_ref, start=start):
                    """`_store_unaligned_rows` of query-head part `part` of
                    `parts` (all heads by default)."""
                    n = o_rows.shape[0] // parts
                    ref = o_rows if parts == 1 else o_rows.at[pl.ds(part * n, n)]
                    _store_unaligned_rows(ref, start, bq, packed, out)

            o_ref = o_ref.at[:, rows, :]
            selected_ref = selected_ref.at[rows, :]
            local_page_ref = local_page_ref.at[rows, :]
            local_limit_ref = local_limit_ref.at[rows, :]
            lse_ref = lse_ref.at[:, pl.ds(tile % tiles_per_step, 1)]
        meta = work_meta_ref[w]
        count = work_count_ref[w]
        num_heads, block_q, head_dim = q_ref.shape
        rows_per_page, packing, row_width = buf_ref.shape[-3:]
        page_size = rows_per_page * (msa_fp4.TOKENS_PER_WORD if fp4 else packing)
        half = w % ring
        first = (meta & msa_attention._FIRST_BIT) != 0
        last = (meta & msa_attention._LAST_BIT) != 0

        def stream_at(half, g):
            return (half * pages_per_step + g,) if cached else (half, g)

        def page_copies(at, sem_at, page, kv_head, n=1):
            """Copies of ``n`` pages from `page` on into buffer ``at``, each
            signalling semaphore ``sem_at``. Streaming without a page cache,
            every copy of a ring slot signals its one semaphore, so a wait sized
            to ``n`` pages waits for any ``n`` of them."""
            src = kv_hbm_ref.at[
                pl.ds(page, n), :, :, pl.ds(kv_head * row_width, row_width)
            ]
            copies = [pltpu.make_async_copy(src, buf_ref.at[at], sem_ref.at[sem_at])]
            if fp4:
                copies.append(
                    pltpu.make_async_copy(
                        scale_hbm_ref.at[pl.ds(page, n)],
                        scale_buf_ref.at[at],
                        scale_sem_ref.at[sem_at],
                    )
                )
            return copies

        def stream_cached():
            """Per-group streaming beside a page cache (flat buffers, one
            semaphore per slot)."""

            def start_group(group, half):
                base = msa_attention._slot(work_meta_ref[group], pages_per_step, 0)
                kv_head = group_head(group)

                def start(g, carry):
                    page = page_table_lib.load(
                        page_table_ref, work_page_ref[base + g]
                    )
                    for copy in page_copies(
                        stream_at(half, g), (half, g), page, kv_head
                    ):
                        copy.start()
                    return carry

                jax.lax.fori_loop(0, work_count_ref[group], start, 0)

            @pl.when(w == 0)
            def _first():
                for d in range(ring - 1):
                    pl.when(d < num_groups)(functools.partial(start_group, d, d))

            @pl.when(w + ring - 1 < num_groups)
            def _next():
                start_group(w + ring - 1, (w + ring - 1) % ring)

            def wait(g, carry):
                # The source only sizes the wait.
                for copy in page_copies(stream_at(half, g), (half, g), 0, 0):
                    copy.wait()
                return carry

            jax.lax.fori_loop(0, count, wait, 0)

        def stream():
            if cached:
                stream_cached()
                return

            def start_group(group, half):
                base = msa_attention._slot(work_meta_ref[group], pages_per_step, 0)
                kv_head = group_head(group)

                def start(g, queue=0):
                    page = page_table_lib.load(
                        page_table_ref, work_page_ref[base + g]
                    )
                    for copy in page_copies(
                        (half, pl.ds(g, 1), 0), (half, 0), page, kv_head
                    ):
                        copy.start(priority=queue)

                # Four page copies per iteration: their address chains (two
                # dependent SMEM loads each) overlap.
                group_count = work_count_ref[group]

                def start4(i, carry):
                    for j in range(_START_UNROLL):  # alternate the two DMA queues
                        start(_START_UNROLL * i + j, j % 2)
                    return carry

                jax.lax.fori_loop(0, group_count // _START_UNROLL, start4, 0)
                done = group_count - group_count % _START_UNROLL

                def start1(g, carry):
                    start(g)
                    return carry

                jax.lax.fori_loop(done, group_count, start1, 0)

            @pl.when(w == 0)
            def _first():
                for d in range(ring - 1):
                    pl.when(d < num_groups)(functools.partial(start_group, d, d))

            @pl.when(w + ring - 1 < num_groups)
            def _next():
                start_group(w + ring - 1, (w + ring - 1) % ring)

            # One wait per set bit of `count` (the source only sizes the wait).
            n = 1
            while n <= pages_per_step:
                for copy in page_copies((half, pl.ds(0, n), 0), (half, 0), 0, 0, n):
                    pl.when((count & n) != 0)(copy.wait)
                n *= 2

        if not cached:
            if mode == "full":  # `pair` kernels copy and wait outside
                stream()

            def group_page(g):
                """Key, page ref and scale page ref of the group's slot `g`."""
                page = work_page_ref[msa_attention._slot(meta, pages_per_step, g)]
                scale_ref = scale_buf_ref.at[half, g] if fp4 else None
                return page, buf_ref.at[half, g], scale_ref

        else:
            # Each distinct page of the call is copied once, at its first
            # group, into the cache instead of once per tile that holds it.
            pl.when(jnp.logical_not(use_cache))(stream)
            cache_sem = (ring, 0)  # past the streaming semaphores

            @pl.when(use_cache & (w == 0))
            def _fill():
                def start(i, carry):
                    for copy in page_copies(
                        (i,),
                        cache_sem,
                        page_table_lib.load(page_table_ref, cache_ref[i]),
                        head_ref[0],
                    ):
                        copy.start()
                    return carry

                def wait_one(i, carry):
                    for copy in page_copies((i,), cache_sem, 0, 0):
                        copy.wait()
                    return carry

                jax.lax.fori_loop(0, num_cached, start, 0)
                jax.lax.fori_loop(0, num_cached, wait_one, 0)
                if bf16_cache:

                    def unpack(i, carry):
                        bfc_ref[i] = _fp4_rows(buf_ref.at[i])
                        return carry

                    jax.lax.fori_loop(0, num_cached, unpack, 0)

            def group_page(g):
                """Key, page ref and scale page ref of the group's slot `g`."""
                entry = work_page_ref[msa_attention._slot(meta, pages_per_step, g)]
                slot = jnp.maximum(entry, 0)
                key = jnp.where(entry >= 0, cache_ref[slot], -1)
                page = jnp.where(use_cache, key, entry)
                buffer = jnp.where(use_cache, slot, half * pages_per_step + g)
                scale_ref = scale_buf_ref.at[buffer] if fp4 else None
                return page, buf_ref.at[buffer], scale_ref

        @pl.when(first & jnp.logical_not(last))
        def _init():
            if not max_free:
                m_ref[...] = jnp.full_like(m_ref, msa_attention._MASK_VALUE)
            l_ref[...] = jnp.zeros_like(l_ref)
            acc_ref[...] = jnp.zeros_like(acc_ref)

        def put(out):  # [H * Bq, D] or [H, Bq, D]
            out = out.reshape(num_heads, block_q, head_dim)
            if sink is not None:  # `_pair_loop` stores two tiles' rows at once
                sink.append(out)
                return
            if store_out is None:
                o_ref[...] = out.astype(o_ref.dtype)
            else:
                store_out(out)

        def write(out, lse, part=0, parts=1):
            """Rows of query-head part `part` of `parts` (all heads by default)."""
            if parts == 1:
                put(out)
                lse_ref[...] = lse.T.reshape(lse_ref.shape)  # lane-dense row
                return
            n = num_heads // parts
            out = out.reshape(n, block_q, head_dim)
            if store_out is None:
                o_ref[part * n : (part + 1) * n] = out.astype(o_ref.dtype)
            else:
                store_out(out, part, parts)
            lanes = n * block_q
            lse_ref[..., pl.ds(part * lanes, lanes)] = lse.T.reshape(
                lse_ref.shape[:-1] + (lanes,)
            )

        def finish(m, denom, acc, part=0, parts=1):
            """Output and lse from a tile's softmax state (log2 units, ``m``
            finite). A row that saw no slot has ``denom = 0`` and ``acc = 0``:
            its output is 0 and its lse ``m + log(0) = -inf`` with no select (a
            seen row has ``denom >= denom_floor``; ``m`` is 0 with max-free)."""
            write(
                acc * (out_unit / jnp.maximum(denom, denom_floor)),
                m * _LN_2 + jnp.log(denom),
                part,
                parts,
            )

        def attend_t(width, single):
            """`attend` for 8/16-bit caches on transposed scores ``s^T [keys,
            rows]``: the max and sum over keys are sublane reductions, the
            output accumulates as ``acc^T [D, rows]`` across the p @ v
            contraction, and ``m``, ``l`` are lane vectors over rows (fewer
            spills and pops: bench j-9081937a, c4 pool48 -5.4%). 4-bit pages
            keep `attend`, whose per-key scales are lane vectors there, and
            so do 8-token (decode) tiles: c4 b3 decode +0.4 ms per step
            transposed (twin j-fa6121d0 vs glue-only j-2abe5011)."""
            row_slots = (
                jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
                if packing == 4
                else msa_attention._bitcast_slots(rows_per_page, packing)
            )  # [S, 1]
            sel_t = selected_ref[...].T  # [K, Bq]
            local_page = local_page_ref[...].T  # [1, Bq]
            local_limit = local_limit_ref[...].T
            keys, values, masks, limits, fixes = [], [], [], [], []
            # The MXU mask replaces the "bias" / "lanemask" / select masks.
            mxu_mask = mxu_mask_on and block_q <= head_dim
            bias_on = bias_t and not mxu_mask
            mdot_seen = None
            lane_reps = (
                128 // block_q
                if 128 % block_q == 0 and num_heads % (128 // block_q) == 0
                and num_heads > 128 // block_q
                else 0
            )
            for g in range(width):
                page, kv_ref, scale_ref = group_page(g)
                hit = jnp.max((sel_t == page).astype(jnp.int32), axis=0, keepdims=True)
                limit = jnp.where(local_page == page, local_limit, page_size)
                limit = jnp.where(page >= 0, limit * hit, 0)  # [1, Bq]
                seen = jnp.max(limit)
                if g == 0:
                    mdot_seen = seen
                if bias_on:
                    # A token sees all of the page or none of it, unless the
                    # page is its local one; an empty slot (page -1) holds
                    # stale bytes that may not be finite.
                    part = jnp.max(
                        ((limit > 0) & (limit < page_size)).astype(jnp.int32)
                    )
                    limits.append(limit)
                    fixes.append((part > 0) | (page < 0))
                if packing == 4 and mxu_mask:  # an additive mask needs finite keys
                    rows = kv_ref[0].reshape(page_size, -1).astype(jnp.bfloat16)
                    rows = jnp.where(row_slots < seen, rows, 0)
                    values.append(rows[:, head_dim:])
                elif packing == 4:
                    rows = kv_ref[0].reshape(page_size, -1).astype(jnp.bfloat16)
                    values.append(jnp.where(row_slots < seen, rows[:, head_dim:], 0))
                else:
                    rows = msa_attention._page_rows(kv_ref, seen)  # zeroed past seen
                    values.append(rows[:, head_dim:])
                if mxu_mask:
                    # [K | A]: A[key, t] = -_MASK_BIAS where token t does not
                    # see the key; lanes past Bq meet zero query columns.
                    padded = jnp.pad(limit, ((0, 0), (0, head_dim - block_q)))
                    bias = jnp.where(row_slots < padded, 0.0, -_MASK_BIAS)
                    keys.append(
                        jnp.concatenate([rows[:, :head_dim], bias.astype(rows.dtype)], 1)
                    )
                    continue
                keys.append(rows[:, :head_dim])
                if lanemask_t and lane_reps:
                    # One 128-lane mask: every 128-lane chunk of rows holds the
                    # same tokens (h * Bq + t, Bq dividing 128).
                    masks.append(row_slots < jnp.tile(limit, (1, lane_reps)))  # [S, 128]
                elif not bias_on:
                    # Rows are head-major (h * Bq + t): every head shares a token's mask.
                    masks.append(row_slots < jnp.tile(limit, (1, num_heads)))  # [S, rows]
            k = jnp.concatenate(keys, axis=0)
            if vt_t:
                v = jnp.concatenate([x.T for x in values], axis=1)  # [D, W * S]
                if lmxu_t:
                    v = jnp.concatenate([v, jnp.ones((8, v.shape[1]), v.dtype)], axis=0)
            else:
                v = jnp.concatenate(values, axis=0)
            q = q_ref[...].reshape(num_heads * block_q, head_dim)
            q = (q.astype(jnp.float32) * (sm_scale * _LOG2_E)).astype(q.dtype)
            if mxu_mask:  # [q | onehot(token)]: row h * Bq + t is token t
                shape = (num_heads * block_q, head_dim)
                row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
                col = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
                onehot = jnp.where(col == row % block_q, 1.0, 0.0).astype(q.dtype)
                q = jnp.concatenate([q, onehot], axis=1)
            if max_free:
                m_prev = 0.0
            else:
                m_prev = m_ref[...] if not single else msa_attention._MASK_VALUE
            if mshift_t and packing == 4:
                # Raw scores of the first page (finite: slots no token sees zeroed).
                k0 = jnp.where(row_slots < mdot_seen, keys[0], 0).astype(k.dtype)
                s0 = jax.lax.dot_general(
                    k0, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
                )  # [S, rows]
                m_new = jnp.maximum(
                    m_prev, jnp.max(s0, axis=0, keepdims=True) + _MDOT_HEADROOM
                )
                # The shift passes through the dot in bf16: keep that exact value.
                m_new = m_new.astype(jnp.bfloat16).astype(jnp.float32)
            if mdot_t and packing == 4:
                lane = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
                k_aug = jnp.broadcast_to((lane == 0).astype(k.dtype), (k.shape[0], 128))
                q_aug = jnp.where(lane == 0, -m_new.T, 0.0).astype(q.dtype)  # [rows, 128]
                s = jax.lax.dot_general(
                    jnp.concatenate([k, k_aug], axis=1),
                    jnp.concatenate([q, q_aug], axis=1),
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )  # [W * S, rows], already s - m_new
            else:
                s = jax.lax.dot_general(
                    k, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
                )  # [W * S, rows]
            if mxu_mask:
                pass
            elif bias_on:
                slabs = []
                for g in range(width):
                    limit = jnp.tile(limits[g], (1, num_heads))  # [1, rows]
                    s_g = s[g * page_size : (g + 1) * page_size] + jnp.where(
                        limit > 0, 0.0, -jnp.inf
                    )

                    def _select(x, limit=limit):
                        return jnp.where(row_slots < limit, x, -jnp.inf)

                    slabs.append(jax.lax.cond(fixes[g], _select, lambda x: x, s_g))
                s = jnp.concatenate(slabs, axis=0)
            elif lanemask_t and lane_reps:
                mask = jnp.concatenate(masks, axis=0)  # [W * S, 128]
                s = jnp.concatenate(
                    [
                        jnp.where(mask, s[:, c : c + 128], -jnp.inf)
                        for c in range(0, s.shape[1], 128)
                    ],
                    axis=1,
                )
            else:
                mask = jnp.concatenate(masks, axis=0)
                s = jnp.where(mask, s, -jnp.inf)
            if mshift_t and packing == 4:
                if msub_t:
                    s = s - m_new
                p = jnp.exp2(s.astype(jnp.bfloat16) if exp16_t else s)
                if not lmxu_t:
                    l_cur = jnp.sum(p.astype(jnp.float32), axis=0, keepdims=True)
                shifted = None
            elif max_free:
                m_prev = m_new = 0.0
                shifted = s
            else:
                m_new = jnp.maximum(m_prev, jnp.max(s, axis=0, keepdims=True))
                shifted = s - m_new
            if shifted is None:
                pass
            elif exp16_t:
                p = jnp.exp2(shifted.astype(jnp.bfloat16))
                if not lmxu_t:
                    l_cur = jnp.sum(p.astype(jnp.float32), axis=0, keepdims=True)
            else:
                p = jnp.exp2(shifted)
                if not lmxu_t:
                    l_cur = jnp.sum(p, axis=0, keepdims=True)
            pv = jax.lax.dot_general(
                v,
                p.astype(v.dtype),
                (((1 if vt_t else 0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )  # [D, rows] ([D + 8, rows] with lmxu: row D is l)
            if lmxu_t:
                l_cur = pv[head_dim : head_dim + 1]
                pv = pv[:head_dim]
            if single:
                finish_t(m_new, l_cur, pv)
                return
            if max_free:
                l_new = l_ref[...] + l_cur
                acc_new = acc_ref[...] + pv
            else:
                alpha = jnp.exp2(m_prev - m_new)
                l_new = alpha * l_ref[...] + l_cur
                acc_new = alpha * acc_ref[...] + pv

            @pl.when(last)
            def _finalize():
                finish_t(m_new, l_new, acc_new)

            @pl.when(jnp.logical_not(last))
            def _carry():
                if not max_free:
                    m_ref[...] = m_new
                l_ref[...] = l_new
                acc_ref[...] = acc_new

        def finish_t(m, denom, acc_t):
            """`finish` for `attend_t`'s lane-vector state."""
            if mshift_t and packing == 4:
                inv = 1.0 / jnp.where(denom > 0, denom, 1.0)
            else:
                inv = 1.0 / jnp.maximum(denom, denom_floor)
            put((acc_t * inv).T)
            lse_ref[...] = (m * _LN_2 + jnp.log(denom)).reshape(lse_ref.shape)

        def attend(width, single, from_bf16=False):
            """One softmax update over the group's first `width` slots; a
            `single` group (the tile's only one) needs no carried state.
            `from_bf16`: 4-bit pages read from the unpacked bf16 cache.
            The tile's query heads run in `row_split` independent parts."""
            # int4 and fp8 pages unpack in slot order; bf16 in the bitcast's order.
            row_slots = (
                jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
                if fp4 or packing == 4
                else msa_attention._bitcast_slots(rows_per_page, packing)
            )  # [S, 1]
            slots = row_slots.T  # [1, S]
            selected = selected_ref[...]
            local_page = local_page_ref[...]
            local_limit = local_limit_ref[...]
            keys, values, masks, k_scales, v_scales = [], [], [], [], []
            for g in range(width):
                if from_bf16:
                    entry = work_page_ref[msa_attention._slot(meta, pages_per_step, g)]
                    slot = jnp.maximum(entry, 0)
                    page = jnp.where(entry >= 0, cache_ref[slot], -1)
                    kv_ref, scale_ref = None, scale_buf_ref.at[slot]
                else:
                    page, kv_ref, scale_ref = group_page(g)
                is_selected = jnp.max(
                    (selected == page).astype(jnp.int32), axis=1, keepdims=True
                )
                limit = jnp.where(local_page == page, local_limit, page_size)
                limit = jnp.where(page >= 0, limit * is_selected, 0)  # [Bq, 1]
                seen = jnp.max(limit)
                if fp4:
                    # Stale or unwritten slots hold finite int4 values and
                    # scales that `_fp4_page` reads as 0 when non-finite, so
                    # their p = 0 contributes nothing.
                    if fp4_rows == "head":
                        k_row = 2 * group_head(w)
                        v_row = k_row + 1
                    elif isinstance(fp4_rows, tuple) and fp4_rows[0] == "head":
                        # ("head", row0): a shared scale page's rows from row0.
                        k_row = fp4_rows[1] + 2 * group_head(w)
                        v_row = k_row + 1
                    else:
                        k_row, v_row = fp4_rows
                    if from_bf16:
                        rows = bfc_ref[slot]
                        sk, sv = _fp4_scales(scale_ref, k_row, v_row)
                    elif qk8 or pv8:
                        rows = _fp4_ints(kv_ref)  # int4 [S, 2D]
                        sk, sv = _fp4_scales(scale_ref, k_row, v_row)
                    else:
                        rows, sk, sv = msa_attention._fp4_page(
                            kv_ref, scale_ref, k_row, v_row
                        )
                    k_int, v_int = rows[:, :head_dim], rows[:, head_dim:]
                    # int4 values are exact in fp8; scales apply around the dots.
                    values.append(
                        v_int.astype(jnp.float8_e4m3fn if pv8 else jnp.bfloat16)
                    )
                    keys.append(
                        k_int.astype(jnp.float8_e4m3fn if qk8 else jnp.bfloat16)
                    )
                    k_scales.append(sk)
                    v_scales.append(sv)
                elif packing == 4:
                    # The packed [S / 4, 4, 2D] page read as [S, 2D] rows: slot
                    # r * 4 + j at row r * 4 + j, and a cheaper relayout than the
                    # 32-bit words' bitcast (16 vs 30 bundles a page, AOT).
                    rows = kv_ref[0].reshape(page_size, -1).astype(jnp.bfloat16)
                    values.append(jnp.where(row_slots < seen, rows[:, head_dim:], 0))
                else:
                    rows = msa_attention._page_rows(kv_ref, seen)
                    values.append(rows[:, head_dim:])
                if not fp4:
                    keys.append(rows[:, :head_dim])
                masks.append(slots < limit)  # [Bq, S]
            k = jnp.concatenate(keys, axis=0)
            v = jnp.concatenate(values, axis=0)
            token_mask = jnp.concatenate(masks, axis=1)
            scale = sm_scale * _LOG2_E
            if fp4:
                k_scale = jnp.concatenate(k_scales, axis=1) * scale
                v_scale = jnp.concatenate(v_scales, axis=1)
                if pv8:
                    v_scale = jnp.minimum(v_scale * _PV8_UNIT, _FP8_MAX)
            part_heads = num_heads // row_split
            part_rows = part_heads * block_q

            for part in range(row_split):
                heads = slice(part * part_heads, (part + 1) * part_heads)
                rows_at = pl.ds(part * part_rows, part_rows)
                if k1_split and width > k1_split:
                    q = q_ref[heads].reshape(part_rows, head_dim)
                    if qk8:
                        q = q.astype(jnp.float8_e4m3fn)
                    pv = p_sum = None
                    parts = -(-width // k1_split)
                    chunk = -(-width // parts)
                    for g0 in range(0, width, chunk):
                        g1 = min(width, g0 + chunk)
                        k_c = jnp.concatenate(keys[g0:g1], axis=0)
                        v_c = jnp.concatenate(values[g0:g1], axis=0)
                        cols = slice(g0 * page_size, g1 * page_size)
                        s_c = jax.lax.dot_general(
                            q, k_c, (((1,), (1,)), ((), ())),
                            preferred_element_type=jnp.float32,
                        )
                        s_c = s_c * k_scale[:, cols]
                        tm = jnp.concatenate([token_mask[:, cols]] * part_heads, axis=0)
                        p_c = jnp.exp2(jnp.where(tm, s_c, -jnp.inf))
                        for j in range(g1 - g0):
                            piece = p_c[:, j * page_size : (j + 1) * page_size]
                            p_sum = piece if p_sum is None else p_sum + piece
                        pv_c = jax.lax.dot_general(
                            (p_c * v_scale[:, cols]).astype(v_c.dtype),
                            v_c,
                            (((1,), (0,)), ((), ())),
                            preferred_element_type=jnp.float32,
                        )
                        pv = pv_c if pv is None else pv + pv_c
                    l_cur = jnp.sum(p_sum, axis=1, keepdims=True)
                    if single:
                        finish(0.0, l_cur, pv, part, row_split)
                        continue
                    l_new = l_ref[rows_at] + l_cur
                    acc_new = acc_ref[rows_at] + pv
                    pl.when(last)(
                        functools.partial(finish, 0.0, l_new, acc_new, part, row_split)
                    )

                    def _carry_k1(rows_at=rows_at, l_new=l_new, acc_new=acc_new):
                        l_ref[rows_at] = l_new
                        acc_ref[rows_at] = acc_new

                    pl.when(jnp.logical_not(last))(_carry_k1)
                    continue
                mask = jnp.concatenate([token_mask] * part_heads, axis=0)
                q = q_ref[heads].reshape(part_rows, head_dim)
                if not fp4:
                    q = (q.astype(jnp.float32) * scale).astype(q.dtype)
                elif qk8:
                    q = q.astype(jnp.float8_e4m3fn)
                s = jax.lax.dot_general(
                    q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
                )
                if fp4:  # x = int4 * scale: K scales the score columns, V scales p
                    s = s * k_scale
                s = jnp.where(mask, s, -jnp.inf)

                # The running max starts at a finite floor (`_MASK_VALUE`), so a
                # row with no visible slot gets p = exp2(-inf) = 0, not exp2(nan).
                if max_free:
                    m_prev = m_new = 0.0
                    p = jnp.exp2(s)
                else:
                    m_prev = (
                        m_ref[rows_at] if not single else msa_attention._MASK_VALUE
                    )
                    m_new = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
                    p = jnp.exp2(s - m_new)
                l_cur = jnp.sum(p, axis=1, keepdims=True)
                p_v = p * v_scale if fp4 else p
                pv = jax.lax.dot_general(
                    p_v.astype(v.dtype),
                    v,
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                if single:
                    finish(m_new, l_cur, pv, part, row_split)
                    continue
                if max_free:
                    l_new = l_ref[rows_at] + l_cur
                    acc_new = acc_ref[rows_at] + pv
                else:
                    alpha = jnp.exp2(m_prev - m_new)
                    l_new = alpha * l_ref[rows_at] + l_cur
                    acc_new = alpha * acc_ref[rows_at] + pv

                pl.when(last)(
                    functools.partial(
                        finish, m_new, l_new, acc_new, part, row_split
                    )
                )

                def _carry(rows_at=rows_at, m_new=m_new, l_new=l_new, acc_new=acc_new):
                    if not max_free:
                        m_ref[rows_at] = m_new
                    l_ref[rows_at] = l_new
                    acc_ref[rows_at] = acc_new

                pl.when(jnp.logical_not(last))(_carry)

        # Slots past the group's pages hold stale bytes: their limits are 0, so
        # their scores are masked and their value rows zeroed.
        # Run at the narrowest width holding the group's pages (`_STEP_WIDTHS`;
        # groups are 16, 32 or 64 pages by call). A step costs per score column,
        # filled or not (~2.3 us at 16 pages, ~1.1 us at 4, twin standalone
        # j-952e8a64), plus ~0.7 us whatever its width (j-aec3278d).
        def run_widths(count=count, first=first, last=last):
            single = first & last
            widths = [n for n in step_widths if n < pages_per_step] + [pages_per_step]
            for i, width in enumerate(widths):
                fits = count > (widths[i - 1] if i else 0)
                if width < pages_per_step:
                    fits = fits & (count <= width)
                body = attend_t if transposed else attend
                if bf16_cache:  # the cache in use reads bf16 pages, a streaming call unpacks
                    for from_bf16, when in ((True, use_cache), (False, jnp.logical_not(use_cache))):
                        pl.when(fits & single & when)(
                            functools.partial(attend, width, True, from_bf16)
                        )
                        pl.when(fits & jnp.logical_not(single) & when)(
                            functools.partial(attend, width, False, from_bf16)
                        )
                    continue
                pl.when(fits & single)(functools.partial(body, width, True))
                pl.when(fits & jnp.logical_not(single))(
                    functools.partial(body, width, False)
                )

        def empty():  # a tile of padding tokens: its only group holds no page
            write(
                jnp.zeros((num_heads * block_q, head_dim), jnp.float32),
                jnp.full((num_heads * block_q, 1), -jnp.inf, jnp.float32),
            )

        if mode != "full":
            return types.SimpleNamespace(
                attend=attend, count=count, single=first & last,
                run_widths=run_widths, empty=empty,
            )
        run_widths()
        pl.when(count == 0)(empty)

    def loop(w, carry):
        group(w)
        return carry

    b = pl.program_id(0)
    if not pair:
        jax.lax.fori_loop(block_start_ref[b], block_start_ref[b + 1], loop, 0)
        return
    _pair_loop(
        group, b, block_start_ref, num_groups, tiles_per_step, work_page_ref,
        work_meta_ref, work_count_ref, page_table_ref, kv_hbm_ref, buf_ref, sem_ref,
        scale_hbm_ref, scale_buf_ref, scale_sem_ref, pages_per_step, step_widths,
        group_head, o_ref,
    )


def _pair_loop(
    group, b, block_start_ref, num_groups, tiles_per_step, work_page_ref,
    work_meta_ref, work_count_ref, page_table_ref, kv_hbm_ref, buf_ref, sem_ref,
    scale_hbm_ref, scale_buf_ref, scale_sem_ref, pages_per_step, step_widths,
    group_head, o_ref,
):
    """`_kernel`'s group loop with `pair` (decode, streamed pages): a grid
    step whose tiles hold one group each runs them two at a time, both
    tiles' softmax in one straight-line body at the wider tile's step width,
    so their dependency chains (MXU push / pop, reductions, exp) interleave
    instead of running back to back (one 8-token tile's body is latency
    bound, ~2.5 ops per bundle); the pair's rows (an even tile's and the
    next) fill whole packed output tiles, so they store at once instead of
    each merging into a shared one. Other steps run their groups one by one.
    Pages stream through a ring of 4 group slots, group ``g`` in slot
    ``g % 4``: before the groups ``[w, w + n)`` run, groups ``w + 2 ..
    w + n + 1`` start (2 groups in flight ahead of the running ones)."""
    ring = sem_ref.shape[0]
    row_width = buf_ref.shape[-1]
    fp4 = scale_hbm_ref is not None

    def copies(slot, sem_slot, page, kv_head, n=1, g=0):
        src = kv_hbm_ref.at[pl.ds(page, n), :, :, pl.ds(kv_head * row_width, row_width)]
        out = [
            pltpu.make_async_copy(
                src, buf_ref.at[slot, pl.ds(g, n), 0], sem_ref.at[sem_slot, 0]
            )
        ]
        if fp4:
            out.append(
                pltpu.make_async_copy(
                    scale_hbm_ref.at[pl.ds(page, n)],
                    scale_buf_ref.at[slot, pl.ds(g, n), 0],
                    scale_sem_ref.at[sem_slot, 0],
                )
            )
        return out

    def start(group_id):
        slot = group_id % ring
        base = msa_attention._slot(work_meta_ref[group_id], pages_per_step, 0)
        kv_head = group_head(group_id)

        def one(g, queue=0):
            page = page_table_lib.load(page_table_ref, work_page_ref[base + g])
            for copy in copies(slot, slot, page, kv_head, g=g):
                copy.start(priority=queue)

        count = work_count_ref[group_id]

        def start4(i, carry):
            for j in range(_START_UNROLL):
                one(_START_UNROLL * i + j, j % 2)
            return carry

        jax.lax.fori_loop(0, count // _START_UNROLL, start4, 0)

        def start1(g, carry):
            one(g)
            return carry

        jax.lax.fori_loop(count - count % _START_UNROLL, count, start1, 0)

    def start_if(group_id):
        pl.when(group_id < num_groups)(functools.partial(start, group_id))

    def wait(group_id):
        slot = group_id % ring
        count = work_count_ref[group_id]
        n = 1
        while n <= pages_per_step:
            for copy in copies(slot, slot, 0, 0, n):
                pl.when((count & n) != 0)(copy.wait)
            n *= 2

    lo, hi = block_start_ref[b], block_start_ref[b + 1]

    @pl.when(lo == 0)
    def _first():
        start_if(0)
        start_if(1)

    def one_by_one(w, carry):
        start_if(w + 2)
        wait(w)
        g = group(w, mode="setup")
        g.run_widths()
        pl.when(g.count == 0)(g.empty)
        return carry

    widths = [n for n in step_widths if n < pages_per_step] + [pages_per_step]

    def two_at_a_time(i, carry):
        w = lo + 2 * i
        start_if(w + 2)
        start_if(w + 3)
        wait(w)
        wait(w + 1)
        outs = ([], [])
        ga = group(w, mode="setup", sink=outs[0])
        gb = group(w + 1, mode="setup", sink=outs[1])
        count = jnp.maximum(ga.count, gb.count)
        block_q = o_ref.shape[1] // tiles_per_step
        rows = pl.ds(pl.multiple_of(2 * i * block_q, 2 * block_q), 2 * block_q)

        def store():
            o_ref[:, rows, :] = jnp.concatenate(
                [outs[0].pop(), outs[1].pop()], axis=1
            ).astype(o_ref.dtype)
        for k, width in enumerate(widths):
            fits = count > (widths[k - 1] if k else 0)
            if width < pages_per_step:
                fits = fits & (count <= width)

            def both(width=width):
                ga.attend(width, True)
                gb.attend(width, True)
                store()

            pl.when(fits)(both)

        @pl.when(count == 0)
        def _empty():
            ga.empty()
            gb.empty()
            store()

        return carry

    if tiles_per_step % 2:  # no pairs in a step
        jax.lax.fori_loop(lo, hi, one_by_one, 0)
        return
    paired = hi - lo == tiles_per_step

    @pl.when(paired)
    def _pairs():
        jax.lax.fori_loop(0, tiles_per_step // 2, two_at_a_time, 0)

    @pl.when(jnp.logical_not(paired))
    def _singles():
        jax.lax.fori_loop(lo, hi, one_by_one, 0)


def _merged_store_rows(block_q: int, num_heads: int, dtype) -> int:
    """The packed output window (`_store_unaligned_rows`) a tile's output is
    merged into, or 1 for a plain store: a window is the output dtype's
    sublane tile (16 rows for bf16). Blocks of 16 heads lower their 8-row
    stores (decode tiles); 8 heads did not (c64 head-parallel, j-d318070b)."""
    rows = 8 * 4 // jnp.dtype(dtype).itemsize
    return rows if block_q % rows and num_heads % rows else 1


def _store_unaligned_rows(o_ref, start, rows, packed, out):
    """Writes ``out [H, rows, D]`` (f32) to rows ``start ..`` of ``o_ref``
    when `rows` is not a multiple of the output's packed sublane tile
    (`packed`): a store at a run-time offset inside a packed tile does not
    lower (Mosaic E2003), so the tile holding the rows is read, merged and
    written whole. Tiles of a grid step run in order, so the rest of the
    packed tile holds an earlier tile's rows or rows a later one writes."""
    base = pl.multiple_of(start // packed * packed, packed)
    window = o_ref.at[:, pl.ds(base, packed), :]
    copies = jnp.concatenate([out.astype(jnp.float32)] * (packed // rows), axis=1)
    row = jax.lax.broadcasted_iota(jnp.int32, copies.shape, 1) + base
    mine = (row >= start) & (row < start + rows)
    window[...] = jnp.where(mine, copies, window[...].astype(jnp.float32)).astype(
        o_ref.dtype
    )


def _attention_call(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    head,
    tile_offset,
    out,
    work,
    *,
    q_per_head,
    sm_scale,
    block_q,
    pages_per_step,
    interpret,
    scale_cache=None,
    scale_rows=None,
    step_widths=_STEP_WIDTHS,
    cache_size=0,
    cache_bf16=False,
    row_split=1,
    tiles_per_head=0,
    precision=frozenset(),
    pair=False,
):
    """One kernel call over the tiles ``tile_offset + [0, tiles)`` of the
    head-major ``q [G * Hg, T, D]``, ``selected [G, T, K]``, ``local_page``
    and ``local_limit [T, 1]``, for query heads ``[b * Hg, (b + 1) * Hg)``
    attending cache-row KV head ``k``, ``head = (k, b)`` (``int32[2]``) with
    ``tile_offset`` (``int32[1]``): scalar-prefetch inputs, not statics, so
    one kernel serves every head and chunk, and no call slices its inputs.

    ``work = (packed, tiles, tiles_per_step[, offsets])``: the call's
    `msa_attention._work_list` as `_packed_work` (or `_packed_work_kernel`,
    whose segment `offsets` it also gives) packs it.

    With `tiles_per_head`, the call's tiles are every KV head's
    (`_one_call_heads`): grid step ``b``'s tiles are head ``b * tiles_per_step
    // tiles_per_head``'s, and `head` and `tile_offset` are unused.

    Writes those heads' and tokens' rows of ``(out [G * Hg, T, D], lse
    [G, T / Bq, 1, Hg * Bq])`` (a tile's lse a lane-dense row, ``h * Bq +
    t``): into `out` in place when given (an earlier call's
    result), else into new arrays whose other rows are unset. A 4-bit
    `kv_cache` takes its `scale_cache` and `scale_rows` (`_kernel`'s
    `fp4_rows`)."""
    num_q_heads, num_tokens, head_dim = q.shape
    num_heads = q_per_head
    _, rows_per_page, packing, _ = kv_cache.shape
    # One KV head's [K | V] slice of each row.
    row_width = 2 * head_dim
    topk = selected.shape[2]
    packed, tiles, tiles_per_step = work[:3]
    groups_per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)
    num_steps = tiles // tiles_per_step
    max_groups = tiles * groups_per_tile
    union_len = max_groups * pages_per_step
    offsets = (
        work[3]
        if len(work) > 3
        else (0, union_len, union_len + max_groups, union_len + 2 * max_groups)
    )

    def kernel(packed_ref, *refs, **kwargs):
        return _kernel(
            packed_ref.at[pl.ds(offsets[0], union_len)],
            packed_ref.at[pl.ds(offsets[1], max_groups)],
            packed_ref.at[pl.ds(offsets[2], max_groups)],
            *refs,
            block_start_ref=packed_ref.at[pl.ds(offsets[3], num_steps + 1)],
            cache_ref=(
                packed_ref.at[
                    pl.ds(union_len + 2 * max_groups + num_steps + 1, cache_size + 2)
                ]
                if cache_size
                else None
            ),
            pages_per_step=pages_per_step,
            **kwargs,
        )

    steps_per_head = tiles_per_head // tiles_per_step if tiles_per_head else 0

    def tile_block(b, packed, table, heads, offset):
        if steps_per_head:
            return (b % steps_per_head, 0)
        return (offset[0] // tiles_per_step + b, 0)

    def head_major_block(w, *prefetch):
        if steps_per_head:
            return (w // steps_per_head, *tile_block(w, *prefetch))
        return (prefetch[2][1], *tile_block(w, *prefetch))

    def lse_block(w, *prefetch):
        return (*head_major_block(w, *prefetch), 0)

    aliased = out is not None
    in_place = (pl.BlockSpec(memory_space=pl.ANY),) * 2 if aliased else ()
    fp4 = scale_rows is not None
    transposed = not fp4 and block_q >= msa_attention.DEFAULT_BLOCK_Q
    scale_inputs = (scale_cache,) if fp4 else ()
    ring = 1 + (_FP4_PREFETCH if fp4 else 1)
    # `pair` (decode, streamed): two running groups + two in flight.
    pair = bool(pair) and not cache_size and not transposed and row_split == 1
    if pair:
        ring = 4
    # Page buffers: a ring of groups, or (page cache) one flat array of
    # max(cache_size, ring groups) pages; the cache's semaphore row follows.
    if cache_size:
        buffers = (max(cache_size, ring * pages_per_step),)
        semaphores = (ring + 1, pages_per_step)
    else:
        # Streaming only: one semaphore per ring slot (`page_copies`).
        buffers = (ring, pages_per_step)
        semaphores = (ring, 1)
    scale_scratch = (
        (
            pltpu.VMEM(buffers + (1,) + scale_cache.shape[1:], jnp.float32),
            pltpu.SemaphoreType.DMA(semaphores),
        )
        if fp4
        else ()
    )
    bf16_cache = bool(cache_size and cache_bf16 and fp4)
    bf16_scratch = (
        (pltpu.VMEM((cache_size, rows_per_page * 8, row_width), jnp.bfloat16),)
        if bf16_cache
        else ()
    )
    if row_split < 1 or num_heads % row_split or (num_heads // row_split * block_q) % 128:
        row_split = 1
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(num_steps,),
        in_specs=[
            pl.BlockSpec(
                (num_heads, tiles_per_step * block_q, head_dim), head_major_block
            ),
            pl.BlockSpec((None, tiles_per_step * block_q, topk), head_major_block),
            pl.BlockSpec((tiles_per_step * block_q, 1), tile_block),
            pl.BlockSpec((tiles_per_step * block_q, 1), tile_block),
            pl.BlockSpec(memory_space=pl.ANY),
            *(pl.BlockSpec(memory_space=pl.ANY) for _ in scale_inputs),
            *in_place,
        ],
        out_specs=[
            pl.BlockSpec(
                (num_heads, tiles_per_step * block_q, head_dim), head_major_block
            ),
            pl.BlockSpec((1, tiles_per_step, 1, num_heads * block_q), lse_block),
        ],
        scratch_shapes=[
            # m, l, acc: rows on sublanes for `attend`, lanes for `attend_t`.
            *(
                [
                    pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
                    pltpu.VMEM((num_heads * block_q, 1), jnp.float32),
                    pltpu.VMEM((num_heads * block_q, head_dim), jnp.float32),
                ]
                if not transposed
                else [
                    pltpu.VMEM((1, num_heads * block_q), jnp.float32),
                    pltpu.VMEM((1, num_heads * block_q), jnp.float32),
                    pltpu.VMEM((head_dim, num_heads * block_q), jnp.float32),
                ]
            ),
            pltpu.VMEM(
                buffers + (1, rows_per_page, packing, row_width), kv_cache.dtype
            ),
            pltpu.SemaphoreType.DMA(semaphores),
            *scale_scratch,
            *bf16_scratch,
        ],
    )
    num_inputs = 4 + 5 + len(scale_inputs)
    return tight_vmem.pallas_call(
        functools.partial(
            kernel,
            sm_scale=sm_scale,
            aliased=aliased,
            tiles_per_step=tiles_per_step,
            groups_per_tile=groups_per_tile,
            transposed=transposed,
            step_widths=step_widths,
            bf16_cache=bf16_cache,
            row_split=row_split,
            tiles_per_head=tiles_per_head,
            precision=(precision if fp4 else precision & frozenset(_FP8_LEVERS))
            | max_free_levers()
            | (k1_levers() if fp4 else frozenset()),
            pair=pair,
            **({"fp4_rows": scale_rows} if fp4 else {}),
        ),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((num_q_heads, num_tokens, head_dim), q.dtype),
            jax.ShapeDtypeStruct(
                (
                    num_q_heads // num_heads,
                    num_tokens // block_q,
                    1,
                    num_heads * block_q,
                ),
                jnp.float32,
            ),
        ],
        input_output_aliases=({num_inputs: 0, num_inputs + 1: 1} if aliased else {}),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            # The bf16 page cache (64 KB a slot) sits beside the 4-bit one.
            vmem_limit_bytes=(
                min(
                    63 * 1024 * 1024,
                    msa_attention._VMEM_LIMIT_BYTES + cache_size * rows_per_page * 8 * row_width * 2,
                )
                if bf16_cache
                else msa_attention._VMEM_LIMIT_BYTES
            ),
            # Every copy's page comes from the page table, so the checks only
            # add scalar work to each step (as the ragged paged attention).
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
        interpret=interpret,
        name="minimax_m3_msa_sparse_attention_lse",
    )(
        packed,
        page_table,
        head,
        tile_offset,
        q,
        selected,
        local_page,
        local_limit,
        kv_cache,
        *scale_inputs,
        *(out if aliased else ()),
    )


def _tiles_per_step(num_tiles: int, rows: int = 1, align: int = 1) -> int:
    """The most tiles (up to `TILES_PER_STEP`) a grid step can take evenly;
    with `align`, only counts whose `rows`-row tiles fill whole `align`-row
    output windows (`_store_unaligned_rows`), or one tile."""
    return next(
        n
        for n in range(TILES_PER_STEP, 0, -1)
        if num_tiles % n == 0 and (n == 1 or n * rows % align == 0)
    )


def _packed_work(selected, block_q, pages_per_step, tiles_per_step=None, align=1):
    """`msa_attention._work_list` of each call's ``selected [C, T, K]`` as
    one int32 row per call, ``[unions | meta words | counts | step
    starts]``, and the tiles each grid step takes: a call then takes one
    slice of its work list, not one per array (each an XLA op of ~1.3 us,
    x 4 calls x G per layer). Step ``b``'s first group is its
    ``tiles_per_step * b``-th tile's; the last entry is the group count."""
    pages, meta, ends, counts = jax.vmap(
        functools.partial(
            msa_attention._work_list, block_q=block_q, pages_per_step=pages_per_step
        )
    )(selected)
    num_tiles = selected.shape[1] // block_q
    if tiles_per_step is None:
        tiles_per_step = _tiles_per_step(num_tiles, block_q, align)
    groups_per_tile = msa_attention._groups_per_tile(
        block_q, selected.shape[2], pages_per_step
    )
    live = jnp.arange(meta.shape[1], dtype=jnp.int32)[None] < ends[:, None]
    tile = jnp.where(live, msa_attention._meta_tile(meta, groups_per_tile), num_tiles)
    bounds = jnp.arange(0, num_tiles + 1, tiles_per_step, dtype=jnp.int32)
    starts = jnp.sum(tile[:, None, :] < bounds[None, :, None], axis=2, dtype=jnp.int32)
    return jnp.concatenate([pages, meta, counts, starts], axis=1), tiles_per_step


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def _work_offsets(num_tiles, block_q, topk, pages_per_step, tiles_per_step):
    """`_packed_work_kernel`'s row: the unions, meta words, counts and step
    starts each at a 128-aligned offset; returns the offsets and the length."""
    groups_per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)
    union_len = num_tiles * groups_per_tile * pages_per_step
    groups = _round_up(num_tiles * groups_per_tile, 128)
    meta = _round_up(union_len, 128)
    starts = meta + 2 * groups
    length = starts + _round_up(num_tiles // tiles_per_step + 1, 128)
    return (0, meta, meta + groups, starts), length


def _bitonic_sort_rows(x):
    """Each row of ``x [R, n]`` (``n`` a power of two) sorted ascending by a
    bitonic network on the lanes: log2(n) * (log2(n) + 1) / 2 compare-exchange
    stages, each two lane rolls and a min / max per vreg, for all rows at
    once (an XLA sort is two ~5 us ops per call at c64)."""
    n = x.shape[1]
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    k = 2
    while k <= n:
        j = k // 2
        while j >= 1:
            lower = (lane & j) == 0
            partner = jnp.where(
                lower, pltpu.roll(x, n - j, axis=1), pltpu.roll(x, j, axis=1)
            )
            keep_min = lower == ((lane & k) == 0)
            x = jnp.where(
                keep_min, jnp.minimum(x, partner), jnp.maximum(x, partner)
            )
            j //= 2
        k *= 2
    return x


def _dense_union_row(x):
    """`msa_attention._dense_unions` of one tile ``x [1, n]`` (`_NO_PAGE`
    for none) on the VPU: [n, n] compares of the keys on lanes against the
    same keys on sublanes (one transpose). Few tiles: its latency is a few
    dozen vreg ops, below the bitonic network's ~50 dependent stages."""
    n = x.shape[1]
    no_page = msa_attention._NO_PAGE
    i32 = jnp.int32
    sub = jax.lax.broadcasted_iota(i32, (n, n), 0)
    lane = jax.lax.broadcasted_iota(i32, (n, n), 1)
    xc = jnp.transpose(jnp.broadcast_to(x, (8, n)))[:, 0:1]  # [n, 1]
    eq = (xc == x).astype(i32)  # [i, j]
    seen_row = jnp.max(jnp.where(sub < lane, eq, 0), axis=0, keepdims=True)
    seen_col = jnp.max(jnp.where(sub > lane, eq, 0), axis=1, keepdims=True)
    first_row = jnp.where((x != no_page) & (seen_row == 0), 1, 0)
    first_col = (xc != no_page) & (seen_col == 0)
    # Distinct pages below x_i: x_i's slot in the ascending union.
    place_col = jnp.sum(jnp.where(x < xc, first_row, 0), axis=1, keepdims=True)
    return jnp.min(
        jnp.where(first_col & (place_col == lane), xc, no_page),
        axis=0,
        keepdims=True,
    )


# Calls of at most this many tiles build unions by `_dense_union_row`, more
# by `_bitonic_sort_rows` (bench j-2ab6fa8c / j-10aec1e9: dense -2..-4 us at
# 1-6 8-token tiles, +3..+5 at 12-16; bitonic ~ XLA at 6-24 tiles).
_DENSE_UNION_MAX_TILES = 6


def _work_list_kernel(
    rows_ref, o_ref, *, block_q, pages_per_step, tiles_per_step, offsets
):
    """`_packed_work` of one call in VMEM: ``rows_ref [tiles, Bq * K]`` each
    tile's page keys (``-1`` none) -> ``o_ref [1, L]`` at `offsets`.

    The unions are `msa_attention._sorted_unions` with both sorts as
    `_bitonic_sort_rows` over every tile at once: the same ascending
    distinct pages. The group words follow `msa_attention._work_list` from
    the tiles' page counts by [tiles, groups] compares. One kernel replaces
    the work list's ~12 small XLA fusions (~4 us per layer in the c16 decode
    bench, each a fixed launch cost on a few vregs) and its two sorts."""
    num_tiles, n = rows_ref.shape
    no_page = msa_attention._NO_PAGE
    groups_per_tile = msa_attention._groups_per_tile(
        block_q, n // block_q, pages_per_step
    )
    union_len = groups_per_tile * pages_per_step
    i32 = jnp.int32
    tiles_p = _round_up(num_tiles, 8)
    o_ref[...] = jnp.full(o_ref.shape, -1, i32)
    if num_tiles <= _DENSE_UNION_MAX_TILES:
        row_sub = jax.lax.broadcasted_iota(i32, (tiles_p, n), 0)
        union = jnp.full((tiles_p, n), no_page, i32)
        for t in range(num_tiles):
            x = rows_ref[t : t + 1, :]
            union = jnp.where(
                row_sub == t, _dense_union_row(jnp.where(x < 0, no_page, x)), union
            )
    else:
        x = rows_ref[...]
        if tiles_p > num_tiles:
            x = jnp.concatenate(
                [x, jnp.full((tiles_p - num_tiles, n), -1, i32)], axis=0
            )
        x = _bitonic_sort_rows(jnp.where(x < 0, no_page, x))
        lane = jax.lax.broadcasted_iota(i32, x.shape, 1)
        dup = (lane > 0) & (x == pltpu.roll(x, 1, axis=1))
        union = _bitonic_sort_rows(jnp.where(dup, no_page, x))
    pages_col = jnp.sum((union != no_page).astype(i32), axis=1, keepdims=True)
    union = jnp.where(union == no_page, -1, union)
    for t in range(num_tiles):
        start = offsets[0] + t * union_len
        o_ref[:, start : start + n] = union[t : t + 1, :]

    tile_sub = jax.lax.broadcasted_iota(i32, (tiles_p, 1), 0)
    real = tile_sub < num_tiles
    groups_col = jnp.where(
        real, jnp.maximum(-(-pages_col // pages_per_step), 1), 0
    )  # [tiles_p, 1]
    total = jnp.sum(groups_col, axis=0, keepdims=True)  # [1, 1]
    # ends / starts of each tile's groups, as a column.
    tri = jax.lax.broadcasted_iota(i32, (tiles_p, tiles_p), 1)
    tri_sub = jax.lax.broadcasted_iota(i32, (tiles_p, tiles_p), 0)
    groups_row = jnp.sum(
        jnp.where(tri_sub == tri, groups_col, 0), axis=0, keepdims=True
    )  # [1, tiles_p]
    ends_col = jnp.sum(
        jnp.where(tri <= tri_sub, groups_row, 0), axis=1, keepdims=True
    )
    starts_col = ends_col - groups_col

    max_groups = num_tiles * groups_per_tile
    width = _round_up(max_groups, 128)
    w = jax.lax.broadcasted_iota(i32, (tiles_p, width), 1)
    in_start = (starts_col <= w) & real
    tile = jnp.sum(
        jnp.where(in_start & (tile_sub >= 1), 1, 0), axis=0, keepdims=True
    )
    start = jnp.max(jnp.where(in_start, starts_col, 0), axis=0, keepdims=True)
    end = jnp.min(
        jnp.where((ends_col > w) & real, ends_col, total), axis=0, keepdims=True
    )
    tile_pages = jnp.max(
        jnp.where(in_start & (ends_col > w), pages_col, 0), axis=0, keepdims=True
    )
    w = w[0:1]
    group = jnp.clip(w - start, 0, groups_per_tile - 1)
    count = jnp.clip(tile_pages - group * pages_per_step, 0, pages_per_step)
    meta = (
        ((tile * groups_per_tile + group) << msa_attention._ROW_SHIFT)
        | jnp.where(w == start, msa_attention._FIRST_BIT, 0)
        | jnp.where(w + 1 == end, msa_attention._LAST_BIT, 0)
    )
    o_ref[:, offsets[1] : offsets[1] + width] = meta
    o_ref[:, offsets[2] : offsets[2] + width] = count
    # Step b's first group: the groups of the tiles before b * tiles_per_step.
    steps = o_ref.shape[1] - offsets[3]
    bound = jax.lax.broadcasted_iota(i32, (tiles_p, steps), 1) * tiles_per_step
    o_ref[:, offsets[3] :] = jnp.sum(
        jnp.where(tile_sub < bound, groups_col, 0), axis=0, keepdims=True
    )


# `_packed_work_kernel` serves calls of at most this many tiles (its union
# stores are unrolled; c64 DCP 8 in one call: 4 heads x 32 16-token tiles).
_WORK_KERNEL_MAX_TILES = 128


def _packed_work_kernel(
    selected, block_q, pages_per_step, tiles_per_step=None, interpret=False
):
    """`_packed_work` as one Pallas kernel (`_work_list_kernel`): ``selected
    [C, T, K]`` -> ``(packed [C, L], tiles_per_step, offsets)``."""
    num_calls, num_tokens, topk = selected.shape
    num_tiles = num_tokens // block_q
    if tiles_per_step is None:
        tiles_per_step = _tiles_per_step(num_tiles)
    offsets, length = _work_offsets(
        num_tiles, block_q, topk, pages_per_step, tiles_per_step
    )
    rows = selected.reshape(num_calls, num_tiles, block_q * topk)
    packed = pl.pallas_call(
        functools.partial(
            _work_list_kernel,
            block_q=block_q,
            pages_per_step=pages_per_step,
            tiles_per_step=tiles_per_step,
            offsets=offsets,
        ),
        grid=(num_calls,),
        in_specs=[
            pl.BlockSpec((None, num_tiles, block_q * topk), lambda c: (c, 0, 0))
        ],
        out_specs=pl.BlockSpec((None, 1, length), lambda c: (c, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((num_calls, 1, length), jnp.int32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="minimax_m3_msa_decode_work_list",
    )(rows)
    return packed.reshape(num_calls, length), tiles_per_step, offsets


def _env_flag(name: str) -> bool:
    """A default-off switch, read when the calling program is traced."""
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


_PROVEN: set[str] = set()


def _proof(flag: str, detail: str) -> None:
    """Once per process: name the default-off path a traced program took."""
    if flag not in _PROVEN:
        _PROVEN.add(flag)
        logging.getLogger(__name__).warning("%s=1: %s", flag, detail)
        print(f"{flag}=1: {detail}", flush=True)  # workers may log nowhere


DECODE_OPT_ENV = "TPU_MINIMAX_M3_MSA_DECODE_OPT"


def decode_opt() -> dict:
    """`TPU_MINIMAX_M3_MSA_DECODE_OPT` (read when the program is traced) for
    decode batches of 8-token tiles with wide groups (`msa_dcp`, more than
    128 tokens), items separated by "," or "+" (inside comma-separated env
    lists): ``widths=<w/w/...>`` the step widths below
    the full group; ``pps=<n>`` pages per group; ``pair``: every decode call
    (`_chunked_call`, any batch) runs its one-group tiles two at a time
    (`_pair_loop`). Empty = off (the defaults)."""
    spec = os.environ.get(DECODE_OPT_ENV, "").strip()
    opt = {}
    for item in filter(None, (x.strip() for x in spec.replace("+", ",").split(","))):
        key, _, val = item.partition("=")
        if key == "widths" and val:
            opt["step_widths"] = tuple(sorted(int(w) for w in val.split("/")))
        elif key == "pps" and val.isdigit() and int(val) >= 4:
            opt["pages_per_step"] = int(val)
        elif key == "pair" and not val:
            opt["pair"] = True
        else:
            raise ValueError(f"{DECODE_OPT_ENV}: unknown item {item!r} in {spec!r}")
    if opt:
        _proof(DECODE_OPT_ENV, f"MSA decode tiles run {opt}")
    return opt


def decode_pair() -> bool:
    """`TPU_MINIMAX_M3_MSA_DECODE_OPT` holds ``pair``."""
    spec = os.environ.get(DECODE_OPT_ENV, "")
    return "pair" in (x.strip() for x in spec.replace("+", ",").split(","))


# 4-bit prefill calls with a page cache pick their tiles per call from the
# width of their selections (`adaptive_prefill`).
ADAPTIVE_ENV = "TPU_MINIMAX_M3_MSA_PREFILL_ADAPTIVE"


def adaptive_prefill() -> tuple[int, dict] | None:
    """``envs.TPU_MINIMAX_M3_MSA_PREFILL_ADAPTIVE`` = "<threshold>:<block_q>:
    <tile_pages>[:<pages_per_step>[:<widths w.w.w>]]" -> (threshold, the wide
    configuration's kernel kwargs), or None (unset). The optional fields give
    the wide configuration its own pages per step and step widths ("." between
    widths; unset = the configured ones). Read when the calling program is
    traced, through `envs` (part of the compile-cache key). A spec of more
    than one tier (`adaptive_tiers`) gives its first tier."""
    tiers = adaptive_tiers()
    return tiers[0][:2] if tiers else None


def adaptive_tiers() -> list[tuple[int, dict, str]]:
    """``envs.TPU_MINIMAX_M3_MSA_PREFILL_ADAPTIVE`` as tiers, ";" between
    them: [(threshold, kernel kwargs, stat)], thresholds increasing within a
    stat. A call runs the last tier whose threshold its selections exceed,
    or the configured tiles if none. Each tier is "[<stat>]<threshold>:
    <block_q>:<tile_pages>[:<pages_per_step>[:<w.w...>]]"; the stat is the
    mean distinct pages per narrow query tile (no prefix, `selection_width`)
    or "x" for the widest sampled tile's distinct pages (`selection_width`
    max), so the calls with the widest tiles, which set the slowest KV head,
    can take their own configuration. Empty: []."""
    from vllm_torchtpu import envs

    spec = envs.TPU_MINIMAX_M3_MSA_PREFILL_ADAPTIVE.strip()
    if not spec:
        return []
    want = "[x]<threshold>:<block_q>:<tile_pages>[:<pages_per_step>[:<w.w...>]]"
    tiers = []
    for tier in spec.split(";"):
        fields = tier.strip().replace("/", ":").split(":")
        stat = "max" if fields[0].startswith("x") else "mean"
        fields[0] = fields[0].removeprefix("x")
        if not 3 <= len(fields) <= 5:
            raise ValueError(f"{ADAPTIVE_ENV}={spec!r}: want {want} per tier")
        parts = [int(x) for x in fields[:4]]
        if min(parts) <= 0:
            raise ValueError(f"{ADAPTIVE_ENV}={spec!r}: want {want} per tier")
        kw = dict(block_q=parts[1], page_cache_tile_pages=parts[2])
        if len(parts) > 3:
            kw["pages_per_step"] = parts[3]
        if len(fields) > 4:
            kw["step_widths"] = tuple(sorted(int(w) for w in fields[4].split(".")))
        if tiers and stat == tiers[-1][2] and parts[0] <= tiers[-1][0]:
            raise ValueError(f"{ADAPTIVE_ENV}={spec!r}: thresholds must increase")
        tiers.append((parts[0], kw, stat))
    return tiers


def selection_width(
    selected: jax.Array, block_q: int, max_tiles: int = 64
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(distinct pages summed over the sampled tiles, sampled tiles holding
    any page, the most distinct pages of one sampled tile) of ``selected [T,
    G, K]`` (page keys, ``-1`` padded) in `block_q`-token tiles of each KV
    head; int32 scalars. Sampling as `wide_selection`."""
    num_tokens, num_heads, topk = selected.shape
    pad = -num_tokens % block_q
    keys = jnp.pad(selected, ((0, pad), (0, 0), (0, 0)), constant_values=-1)
    keys = keys.transpose(1, 0, 2).reshape(num_heads, -1, block_q * topk)
    keys = keys[:, :: max(1, keys.shape[1] // max_tiles)]
    width = keys.shape[-1]
    earlier = jnp.arange(width)[None, :] < jnp.arange(width)[:, None]
    seen = jnp.any((keys[..., :, None] == keys[..., None, :]) & earlier, axis=-1)
    per_tile = jnp.sum((keys >= 0) & ~seen, axis=-1, dtype=jnp.int32)
    tiles = jnp.sum(jnp.any(keys >= 0, axis=-1), dtype=jnp.int32)
    return jnp.sum(per_tile), tiles, jnp.max(per_tile)


def adaptive_tier(
    selected: jax.Array, block_q: int, tiers: list[tuple[int, dict, str]]
) -> jax.Array:
    """Index into [configured tiles, *tiers] of the configuration a call
    with page keys ``selected [T, G, K]`` runs (`adaptive_tiers`): int32."""
    distinct, tiles, widest = selection_width(selected, block_q)
    index = jnp.int32(0)
    for i, (threshold, _, stat) in enumerate(tiers):
        over = widest > threshold if stat == "max" else distinct > threshold * tiles
        index = jnp.where(over, jnp.int32(i + 1), index)
    return index


def wide_selection(
    selected: jax.Array, block_q: int, threshold: int, max_tiles: int = 64
) -> jax.Array:
    """Whether ``selected [T, G, K]`` (page keys, ``-1`` padded) holds more
    than `threshold` distinct pages per `block_q`-token tile of a KV head on
    average, over the tiles that hold any page: a bool scalar.

    The union width sets what a tile costs: a wide union pays for its rows
    over more pages (smaller tiles narrow it), a narrow one for the fixed
    cost per tile (larger tiles amortize it). Counted by comparing each key
    with the tile's earlier keys (no sort), on every n-th tile only (about
    `max_tiles` per head): every tile of an 8k call cost ~76 us a call on
    TPU v7x, while the sampled mean picks the same configuration on 99% of
    real c48 calls."""
    num_tokens, num_heads, topk = selected.shape
    pad = -num_tokens % block_q
    keys = jnp.pad(selected, ((0, pad), (0, 0), (0, 0)), constant_values=-1)
    keys = keys.transpose(1, 0, 2).reshape(num_heads, -1, block_q * topk)
    keys = keys[:, :: max(1, keys.shape[1] // max_tiles)]
    width = keys.shape[-1]
    earlier = jnp.arange(width)[None, :] < jnp.arange(width)[:, None]
    seen = jnp.any((keys[..., :, None] == keys[..., None, :]) & earlier, axis=-1)
    distinct = jnp.sum((keys >= 0) & ~seen, dtype=jnp.int32)
    tiles = jnp.sum(jnp.any(keys >= 0, axis=-1), dtype=jnp.int32)
    return distinct > threshold * tiles


# Decode work list by `_packed_work_kernel` instead of XLA (8-token tiles).
WORK_LIST_KERNEL_ENV = "TPU_MINIMAX_M3_MSA_DECODE_WORK_LIST_KERNEL"
# Every KV head of a decode batch in one kernel call (`_one_call_heads`).
ONE_CALL_ENV = "TPU_MINIMAX_M3_MSA_DECODE_ONE_CALL"
# Decode batches: at most this many tokens per KV head (64 requests x 8).
_DECODE_MAX_TOKENS = 512


def _chunked_call(
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    first_head,
    *,
    sm_scale,
    block_q,
    pages_per_step,
    interpret,
    scale_cache=None,
    scale_rows=None,
    head_major=False,
    step_widths=_STEP_WIDTHS,
    page_cache_slots=0,
    page_cache_bf16=False,
    row_split=1,
    page_cache_tile_pages=0,
    q_head_major=False,
    prefill_precision=None,
):
    """`_attention_call` per KV head and chunk of at most
    `MAX_TOKENS_PER_CALL` tokens (the work list lives in SMEM).

    With `head_major`, ``(out [G * Hg, T, D], lse [G * Hg, T])`` come back
    head-major, as the kernel writes them, with no transpose.

    ``q [T, G * Hg, D]``: cache-row KV head ``first_head + h`` owns query
    heads ``[h * Hg, (h + 1) * Hg)``; ``selected [T, G, K]``. Every call
    reads the whole head-major query and writes its head's and chunk's rows
    of one output in place, so no call needs a slice, transpose or stacking
    of its own (~47 us per 2,048-token chunk under the former `lax.map` in
    the c4 prefill twin, j-3c651507, and ~43 us more per head at DCP 8).
    The calls after the first run as one `fori_loop`, so the kernel is
    emitted twice per call site (the first call creates the output) rather
    than once per head and chunk. With `q_head_major`, q comes ``[G * Hg, T,
    D]``, as the kernel reads it."""
    if not q_head_major:
        q = q.transpose(1, 0, 2)
    num_q_heads, num_tokens, _ = q.shape
    num_heads = selected.shape[1]
    if num_tokens == 0:
        if head_major:
            return q, jnp.zeros((num_q_heads, 0), jnp.float32)
        return q.transpose(1, 0, 2), jnp.zeros((0, num_q_heads), jnp.float32)
    call = functools.partial(
        _attention_call,
        q_per_head=num_q_heads // num_heads,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_rows=scale_rows,
        step_widths=step_widths,
        row_split=row_split,
    )
    cache_size = page_cache_slots if scale_cache is not None else 0
    if cache_size:
        call = functools.partial(
            call,
            cache_size=cache_size,
            cache_bf16=page_cache_bf16,
            precision=prefill_precision_levers(prefill_precision),
        )
    elif scale_cache is None and prefill_precision:
        # 8-bit pages: `prefill_precision` names TPU_MINIMAX_M3_MSA_PREFILL_FP8_OPT
        # levers (`msa_dcp.msa_attend_dcp`).
        call = functools.partial(
            call, precision=fp8_prefill_levers(prefill_precision)
        )
    limit = msa_attention.MAX_TOKENS_PER_CALL
    step = min(num_tokens, max(block_q, limit - limit % block_q))
    num_chunks = -(-num_tokens // step)
    pad = num_chunks * step - num_tokens
    q = jnp.pad(q, ((0, 0), (0, pad), (0, 0)))
    selected = jnp.pad(
        selected, ((0, pad), (0, 0), (0, 0)), constant_values=-1
    ).transpose(1, 0, 2)
    local_page = jnp.pad(local_page, (0, pad), constant_values=-1)[:, None]
    local_limit = jnp.pad(local_limit, (0, pad))[:, None]

    num_tiles = step // block_q
    # Tiles of a grid step fill whole packed output windows (`_merged_store_rows`).
    align = _merged_store_rows(block_q, num_q_heads // num_heads, q.dtype)
    tiles_per_step = _tiles_per_step(num_tiles, block_q, align)
    decode = num_chunks == 1 and num_tokens <= _DECODE_MAX_TOKENS and not cache_size
    if decode and decode_pair():
        _proof(DECODE_OPT_ENV, "MSA decode tiles run two at a time (pair)")
        call = functools.partial(call, pair=True)
    if decode and num_heads > 1 and _env_flag(ONE_CALL_ENV):
        _proof(ONE_CALL_ENV, f"{num_heads} KV heads in one MSA decode call")
        return _one_call_heads(
            call, q, kv_cache, selected, local_page, local_limit, page_table,
            first_head, num_tokens=num_tokens, block_q=block_q,
            pages_per_step=pages_per_step, head_major=head_major,
            interpret=interpret, align=align,
        )
    keys = selected.reshape(num_heads * num_chunks, step, selected.shape[2])
    offsets = None
    if (
        decode
        and num_tiles <= _DENSE_UNION_MAX_TILES
        and _env_flag(WORK_LIST_KERNEL_ENV)
    ):
        _proof(WORK_LIST_KERNEL_ENV, "MSA decode work list by one Pallas kernel")
        packed, tps, offsets = _packed_work_kernel(
            keys, block_q, pages_per_step, tiles_per_step=tiles_per_step,
            interpret=interpret,
        )
        assert tps == tiles_per_step
    else:
        # Every call's work list in one batched pass ahead of the loop: built
        # per call inside it, each call paid its own two union sorts and glue
        # (~9 us per call, x G x 4 calls per layer).
        packed, _ = _packed_work(
            keys, block_q, pages_per_step, tiles_per_step=tiles_per_step
        )
    if cache_size:
        union_len = (step // block_q) * msa_attention._groups_per_tile(
            block_q, selected.shape[2], pages_per_step
        ) * pages_per_step
        table = _page_cache_tables(
            packed[:, :union_len],
            num_tiles=step // block_q,
            cache_size=cache_size,
            key_bound=page_table_lib.num_keys(page_table),
            tile_pages=page_cache_tile_pages,
        )
        packed = jnp.concatenate(
            [table[:, :union_len], packed[:, union_len:], table[:, union_len:]], axis=1
        )

    def head_chunk(i, out):
        h, c = i // num_chunks, i % num_chunks
        offset = jnp.full((1,), c * (step // block_q), jnp.int32)
        heads = jnp.stack([first_head + h, h]).astype(jnp.int32)
        return call(
            q,
            kv_cache,
            selected,
            local_page,
            local_limit,
            page_table,
            heads,
            offset,
            out,
            (packed[i], num_tiles, tiles_per_step)
            + ((offsets,) if offsets is not None else ()),
        )

    out = head_chunk(0, None)
    if num_heads * num_chunks > 1:
        out = jax.lax.fori_loop(1, num_heads * num_chunks, head_chunk, out)
    return _finish(out, num_tokens, block_q, head_major)


def _finish(out, num_tokens, block_q, head_major):
    out, lse = out
    num_q_heads = out.shape[0]
    # lse [head blocks, tiles, 1, Hg * Bq] -> [T, G * Hg] (or head-major)
    num_blocks, num_tiles = lse.shape[:2]
    if head_major:
        lse = lse.reshape(num_blocks, num_tiles, -1, block_q).transpose(0, 2, 1, 3)
        lse = lse.reshape(num_q_heads, num_tiles * block_q)
        return out[:, :num_tokens], lse[:, :num_tokens]
    lse = lse.reshape(num_blocks, num_tiles, -1, block_q).transpose(1, 3, 0, 2)
    lse = lse.reshape(num_tiles * block_q, num_q_heads)
    return out.transpose(1, 0, 2)[:num_tokens], lse[:num_tokens]


def _one_call_heads(
    call,
    q,
    kv_cache,
    selected,
    local_page,
    local_limit,
    page_table,
    first_head,
    *,
    num_tokens,
    block_q,
    pages_per_step,
    head_major,
    interpret=False,
    align=1,
):
    """`_chunked_call` of a decode batch (one chunk) as one kernel call over
    every KV head's tiles (head-major ``q [G * Hg, T, D]``, ``selected [G, T,
    K]``): the per-head calls of the `fori_loop` each paid a call's fixed
    cost (pipeline fill and drain, the first group's exposed page copies,
    the loop step). Tile ``h * tiles + j`` is head ``h``'s tile ``j``: no
    tile spans two heads, so its union, groups and softmax order are the
    per-head call's and the outputs are the same bits."""
    num_heads, step, topk = selected.shape
    tiles = step // block_q
    tiles_per_step = _tiles_per_step(tiles, block_q, align)
    keys = selected.reshape(1, num_heads * step, topk)
    offsets = None
    if num_heads * tiles <= _WORK_KERNEL_MAX_TILES and _env_flag(WORK_LIST_KERNEL_ENV):
        _proof(WORK_LIST_KERNEL_ENV, "MSA decode work list by one Pallas kernel")
        packed, tps, offsets = _packed_work_kernel(
            keys, block_q, pages_per_step, tiles_per_step=tiles_per_step,
            interpret=interpret,
        )
        assert tps == tiles_per_step
    else:
        packed, _ = _packed_work(
            keys, block_q, pages_per_step, tiles_per_step=tiles_per_step
        )
    heads = jnp.stack([jnp.asarray(first_head), jnp.asarray(0)]).astype(jnp.int32)
    if not isinstance(first_head, int) or first_head:
        raise NotImplementedError("one call serves cache-row KV heads 0..G-1")
    out = call(
        q,
        kv_cache,
        selected,
        local_page,
        local_limit,
        page_table,
        heads,
        jnp.zeros((1,), jnp.int32),
        None,
        (packed[0], num_heads * tiles, tiles_per_step)
        + ((offsets,) if offsets is not None else ()),
        tiles_per_head=tiles,
    )
    return _finish(out, num_tokens, block_q, head_major)


def _pad_tokens(q, selected, local_page, local_limit, block_q, q_head_major=False):
    pad = -selected.shape[0] % block_q
    if pad:
        widths = [(0, 0)] * q.ndim
        widths[1 if q_head_major else 0] = (0, pad)
        q = jnp.pad(q, widths)
        selected = jnp.pad(
            selected, ((0, pad),) + ((0, 0),) * (selected.ndim - 1), constant_values=-1
        )
        local_page = jnp.pad(local_page, (0, pad), constant_values=-1)
        local_limit = jnp.pad(local_limit, (0, pad))
    return q, selected, local_page, local_limit


def _check_rows(kv_cache, head_dim, head=None):
    num_heads = kv_cache.shape[3] // (2 * head_dim)
    if kv_cache.shape[3] % (2 * head_dim) or (
        head is not None and not 0 <= head < num_heads
    ):
        raise ValueError(
            f"Cache rows hold [K | V] of width {2 * head_dim} per KV head; got "
            f"{kv_cache.shape} for head {head}."
        )
    return num_heads


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "block_q",
        "pages_per_step",
        "head",
        "k_scale_row",
        "v_scale_row",
        "interpret",
        "head_major",
        "step_widths",
        "page_cache_slots",
        "page_cache_bf16",
        "row_split",
        "page_cache_tile_pages",
        "q_head_major",
        "prefill_precision",
    ),
)
def msa_sparse_attention_lse(
    q: jax.Array,
    kv_cache: jax.Array,
    selected: jax.Array,
    local_page: jax.Array,
    local_limit: jax.Array,
    page_table: jax.Array,
    *,
    sm_scale: float,
    block_q: int = msa_attention.DEFAULT_BLOCK_Q,
    pages_per_step: int = msa_attention.DEFAULT_PAGES_PER_STEP,
    head: int = 0,
    scale_cache: jax.Array | None = None,
    k_scale_row: int = 0,
    v_scale_row: int = 1,
    interpret: bool = False,
    head_major: bool = False,
    step_widths: tuple[int, ...] = _STEP_WIDTHS,
    page_cache_slots: int = 0,
    page_cache_bf16: bool = False,
    row_split: int = 1,
    page_cache_tile_pages: int = 0,
    q_head_major: bool = False,
    prefill_precision: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """`msa_attention.msa_sparse_attention` (kernel v2, page keys translated
    by `page_table`) returning ``(out [T, H, D], lse [T, H] f32)``.

    A token with no selected page (or a padding token) gets a zero output
    and ``lse = -inf``, so it carries no weight in a log-sum-exp merge.
    Cache rows may hold several KV heads, ``[K_0 | V_0 | K_1 | V_1 ...]``;
    `head` picks the one these queries attend (a 2D-wide block of each page).
    A 4-bit cache takes its `scale_cache` (`msa_attention.msa_sparse_attention`).
    With `head_major`, the results come back head-major: ``out [H, T, D]``,
    ``lse [H, T]``; with `q_head_major`, q comes head-major, ``[H, T, D]``.
    """
    num_tokens, head_dim = selected.shape[0], q.shape[2]
    scale_cache, row0 = msa_fp4.split_scales(scale_cache)
    k_scale_row, v_scale_row = k_scale_row + row0, v_scale_row + row0
    msa_attention.check_cache(kv_cache, scale_cache)
    _check_rows(kv_cache, head_dim, head)
    q, selected, local_page, local_limit = _pad_tokens(
        q, selected, local_page, local_limit, block_q, q_head_major
    )
    out, lse = _chunked_call(
        q,
        kv_cache,
        selected[:, None],
        local_page,
        local_limit,
        page_table,
        head,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_rows=None if scale_cache is None else (k_scale_row, v_scale_row),
        head_major=head_major,
        step_widths=step_widths,
        page_cache_slots=page_cache_slots,
        page_cache_bf16=page_cache_bf16,
        row_split=row_split,
        page_cache_tile_pages=page_cache_tile_pages,
        q_head_major=q_head_major,
        prefill_precision=prefill_precision,
    )
    if head_major:
        return out[:, :num_tokens], lse[:, :num_tokens]
    return out[:num_tokens], lse[:num_tokens]


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "block_q",
        "pages_per_step",
        "interpret",
        "head_major",
        "step_widths",
        "page_cache_slots",
        "page_cache_bf16",
        "row_split",
        "page_cache_tile_pages",
        "q_head_major",
        "prefill_precision",
    ),
)
def msa_sparse_attention_lse_heads(
    q: jax.Array,
    kv_cache: jax.Array,
    selected: jax.Array,
    local_page: jax.Array,
    local_limit: jax.Array,
    page_table: jax.Array,
    *,
    sm_scale: float,
    block_q: int = msa_attention.DEFAULT_BLOCK_Q,
    pages_per_step: int = msa_attention.DEFAULT_PAGES_PER_STEP,
    scale_cache: jax.Array | None = None,
    interpret: bool = False,
    head_major: bool = False,
    step_widths: tuple[int, ...] = _STEP_WIDTHS,
    page_cache_slots: int = 0,
    page_cache_bf16: bool = False,
    row_split: int = 1,
    page_cache_tile_pages: int = 0,
    q_head_major: bool = False,
    prefill_precision: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """`msa_sparse_attention_lse` for every KV head of the cache rows at once.

    ``q [T, G * Hg, D]`` (KV head ``h`` owns query heads ``[h * Hg,
    (h + 1) * Hg)``), ``selected [T, G, K]``. The heads run in one loop over
    a single kernel whose KV head is an input, so the program carries one
    copy of the kernel instead of one per head. Returns ``(out
    [T, G * Hg, D], lse [T, G * Hg] f32)``.

    With a 4-bit cache, `scale_cache` ``[pages, R, S]`` holds head ``h``'s K / V
    scales in rows ``2h`` / ``2h + 1``. `head_major` and `q_head_major` as
    `msa_sparse_attention_lse`.
    """
    num_tokens, num_heads = selected.shape[:2]
    num_q_heads, head_dim = q.shape[0 if q_head_major else 1], q.shape[2]
    scale_cache, row0 = msa_fp4.split_scales(scale_cache)
    msa_attention.check_cache(kv_cache, scale_cache)
    if scale_cache is not None and scale_cache.shape[1] < row0 + 2 * num_heads:
        raise ValueError(
            f"scale pages {scale_cache.shape} miss K/V rows of {num_heads} heads."
        )
    if _check_rows(kv_cache, head_dim) < num_heads or num_q_heads % num_heads:
        raise ValueError(
            f"{num_heads} KV heads of {num_q_heads} query heads do not fit cache "
            f"rows {kv_cache.shape}."
        )
    q, selected, local_page, local_limit = _pad_tokens(
        q, selected, local_page, local_limit, block_q, q_head_major
    )
    out, lse = _chunked_call(
        q,
        kv_cache,
        selected,
        local_page,
        local_limit,
        page_table,
        0,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_rows=(
            None if scale_cache is None else "head" if row0 == 0 else ("head", row0)
        ),
        head_major=head_major,
        step_widths=step_widths,
        page_cache_slots=page_cache_slots,
        page_cache_bf16=page_cache_bf16,
        row_split=row_split,
        page_cache_tile_pages=page_cache_tile_pages,
        q_head_major=q_head_major,
        prefill_precision=prefill_precision,
    )
    if head_major:
        return out[:, :num_tokens], lse[:, :num_tokens]
    return out[:num_tokens], lse[:num_tokens]
