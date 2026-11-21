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
"""Environment variables that apply only to experimental Qwen3.8-Flash.

Every flag has an owner and a removal date. Flags here must not change the
behavior of any other model; shared code never reads this module directly.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm_torchtpu.envs import (
    env_bool,
    env_float,
    env_int,
    env_nonnegative_int,
    env_optional_int,
    env_str,
    env_with_choices,
)

if TYPE_CHECKING:
    VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL: bool = False
    TPU_GDN_CHUNK_PAIR: bool = False
    TPU_GDN_KERNEL_ZERO_OUT: bool = False
    TPU_GDN_RAW_BA: bool = False
    TPU_GDN_RAW_QKV: bool = False
    VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR: bool = False
    VLLM_TPU_GDN_HOIST_METADATA: bool = False
    VLLM_TPU_GDN_MIXED_TILE_POW2: bool = False
    MOE_ADAPTIVE_2D_CHANNEL_SCALES: bool = False
    VLLM_TPU_MAMBA_TAIL_FOLD: bool = False
    VLLM_TPU_MAMBA_TAIL_FOLD_DCP: bool = False
    VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY: bool = False
    VLLM_TPU_MAMBA_TAIL_FOLD_UNIT: int = 0
    QWEN4_EXP_MAX_POOL_BYTES: int | None = None
    QWEN4_EXP_PACK_COMPRESSED_K: bool = False
    QWEN4_EXP_PACK_MAIN_KV: bool = False
    QWEN4_EXP_POOL_GRAPH_COPIES: int = 0
    QWEN4_EXP_POOL_HEADROOM_BYTES: int = 6 * 1024**3
    QWEN4_EXP_POOL_SIZING: str = "hbm"
    QWEN4_EXP_PREFILL_SP: bool = False
    QWEN4_EXP_PREFILL_SP_MIN_TOKENS: int = 1024
    QWEN4_EXP_PREFILL_SP_MOE: bool = False
    QWEN4_EXP_PREFILL_SP_PAD_HC: bool = False
    QWEN4_EXP_PREFILL_SP_SHARED_LOCAL: bool = False
    TPU_QWEN4_HC_DECODE_FUSED: bool = False
    TPU_QWEN4_HC_PREFILL_NORM: str = "vpu"
    VLLM_TPU_HC_PREFILL_COMBINE_MIX: bool = False
    VLLM_TPU_HC_PREFILL_INJECT: bool = False
    VLLM_TPU_HC_PREFILL_UP_T: bool = False
    VLLM_TPU_HC_PREFILL_WIDE_TILE: int = 0
    VLLM_TPU_MEGAKERNEL: str = ""
    VLLM_TPU_MK_HOST_PLE: bool = False
    VLLM_TPU_MK_PLE_TRIM: bool = False
    VLLM_TPU_MK_WHOLE_L47: bool = False
    VLLM_TPU_PLE_CONV_HISTORY_FAST: bool = False
    VLLM_TPU_PLE_FUSED_GATE: bool = False
    VLLM_TPU_PLE_HOST_FAST_LOOKUP: bool = False
    VLLM_TPU_PLE_HOST_GATHER_WORKERS: int = 0
    VLLM_TPU_PLE_HOST_OVERLAP: bool = False
    VLLM_TPU_PLE_HOST_OVERLAP_CHECK: int = 0
    VLLM_TPU_PLE_HOST_TIMING: bool = False
    VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS: int = 8
    VLLM_TPU_QSA_BF16_REDUCE_PRECISION: bool = False
    VLLM_TPU_QSA_COMPRESSED_DCP: bool = False
    VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT: bool = False
    VLLM_TPU_QSA_COMPRESSED_RUN_WRITE: bool = False
    VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS: int = 512
    VLLM_TPU_QSA_DCP_GATE_IN_COMBINE: bool = False
    VLLM_TPU_QSA_DCP_HEAD_EXCHANGE: bool = False
    VLLM_TPU_QSA_DCP_META_PARTITION: bool = False
    VLLM_TPU_QSA_DCP_PACK_LISTS: bool = False
    VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK: int = 1024
    VLLM_TPU_QSA_DCP_PREFILL_QUERY_TILE: int = 256
    VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR: bool = False
    VLLM_TPU_QSA_DCP_Q_GATHER: bool = False
    VLLM_TPU_QSA_DCP_SHARD_MERGE: bool = False
    VLLM_TPU_QSA_DCP_SPLIT_LSE: bool = False
    VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE: bool = False
    VLLM_TPU_QSA_DECODE_FAST_FETCH: bool = False
    VLLM_TPU_QSA_DECODE_KV_BLOCK: int = 1024
    VLLM_TPU_QSA_DECODE_LEAN_FETCH: bool = False
    VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA: bool = False
    VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE: bool = False
    VLLM_TPU_QSA_DECODE_QUERY_BLOCK: int = 0
    VLLM_TPU_QSA_DECODE_RPA_BKV: int = 256
    VLLM_TPU_QSA_DECODE_RPA_LANES: int = 8
    VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES: int = 0
    VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE: bool = False
    VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK: int = 256
    VLLM_TPU_QSA_DECODE_SLOT_READ: bool = False
    VLLM_TPU_QSA_DECODE_TOKEN_SHARD: bool = False
    VLLM_TPU_QSA_DRAFT_SHARED_SELECTION: bool = False
    VLLM_TPU_QSA_FUSED_DCP_BLOCKS: bool = False
    VLLM_TPU_QSA_FUSED_KV_WRITE: bool = False
    VLLM_TPU_QSA_FUSED_SPARSE_DECODE: bool = False
    VLLM_TPU_QSA_INDEXER_HOIST_META: bool = False
    VLLM_TPU_QSA_INDEXER_MIXED_ONLY: bool = False
    VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY: bool = False
    VLLM_TPU_QSA_INDEXER_SHARD: bool = False
    VLLM_TPU_QSA_INDEXER_SHARD_A2A_MIN_TOKENS: int = 0
    VLLM_TPU_QSA_INDEXER_SHARD_SKIP_SP: bool = False
    VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES: bool = False
    VLLM_TPU_QSA_INDEXER_TOKEN_SHARD: bool = False
    VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT: bool = False
    VLLM_TPU_QSA_MAIN_KV_DCP: int = 0
    VLLM_TPU_QSA_MAIN_RUN_WRITE: bool = False
    VLLM_TPU_QSA_MAIN_RUN_WRITE_MIN_TOKENS: int = 512
    VLLM_TPU_QSA_MAIN_RUN_WRITE_TILE: int = 512
    VLLM_TPU_QSA_MTP_KV_DCP: bool = False
    VLLM_TPU_QSA_PREFILL_BF16_MXU: bool = False
    VLLM_TPU_QSA_PREFILL_BITSET_KERNEL: bool = True
    VLLM_TPU_QSA_PREFILL_BITSET_OWNED: bool = False
    VLLM_TPU_QSA_PREFILL_INDEXER_BKV: int = 256
    VLLM_TPU_QSA_PREFILL_INDEXER_BQ: int = 128
    VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK: int = 0
    VLLM_TPU_QSA_PREFILL_LEAN: bool = False
    VLLM_TPU_QSA_PREFILL_LIVE_MASK_PADDED: bool = False
    VLLM_TPU_QSA_PREFILL_PAGE_COPIES: bool = False
    VLLM_TPU_QSA_PREFILL_PREFIX_PAGES: bool = False
    VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_ALPHA: float = 0.0
    VLLM_TPU_QSA_PREFILL_UNION_CAP: int = 12288
    VLLM_TPU_QSA_PREFILL_UNION_COMPACT_COPY: str = "vmem"
    VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS: int = 0
    VLLM_TPU_QSA_PREFILL_UNION_GATHERED: str = ""
    VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS: int = 4
    VLLM_TPU_QSA_PREFILL_UNION_GATHER_GROWTH: float = 1.0
    VLLM_TPU_QSA_PREFILL_UNION_KV_TILE: int = 1024
    VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_LISTED: str = ""
    VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT: int = 32768
    VLLM_TPU_QSA_PREFILL_UNION_QK16: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_WALK: int = 0
    VLLM_TPU_QSA_PREFILL_VPU: str = ""
    VLLM_TPU_QSA_PREFIX_SLICE: bool = False
    VLLM_TPU_QSA_RING_DMA_BATCH: int = 0
    VLLM_TPU_QSA_RING_SELECT: bool = False
    VLLM_TPU_QSA_TOPK_STATS: int = 0
    VLLM_TPU_QSA_VISIBLE_BYPASS: bool = False
    VLLM_TPU_QWEN4_FUSED_PREP: bool = False
    VLLM_TPU_QWEN4_FUSED_PREP_CHECK: str = ""
    VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL: bool = False
    VLLM_TPU_QWEN4_NGRAM_WINDOW: str = ""
    VLLM_TPU_SHAPE_TUNED: bool = False

# Owner: @DRZDONG. Remove: when experimental/ replaces models/vllm/qwen4_exp
# or by 2027-03-31, whichever comes first.
environment_variables: dict[str, Callable[[], Any]] = {
    # Temporary migration selector: serve the Qwen4Exp architectures with
    # models/qwen38_flash/experimental instead of models/vllm/qwen4_exp.
    "VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL": env_bool(
        "VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL", default=False
    ),
    # Model-only flags moved from vllm_torchtpu/envs.py (hill-climb toggles).
    # Owner: @DRZDONG. Remove each one once its optimization self-selects at
    # runtime or is dropped; review by 2027-03-31.
    "QWEN4_EXP_MAX_POOL_BYTES": env_optional_int("QWEN4_EXP_MAX_POOL_BYTES"),
    "QWEN4_EXP_PACK_COMPRESSED_K": env_bool(
        "QWEN4_EXP_PACK_COMPRESSED_K", default=False
    ),
    "QWEN4_EXP_PACK_MAIN_KV": env_bool("QWEN4_EXP_PACK_MAIN_KV", default=False),
    "QWEN4_EXP_POOL_GRAPH_COPIES": env_int("QWEN4_EXP_POOL_GRAPH_COPIES", 0),
    "QWEN4_EXP_POOL_HEADROOM_BYTES": env_int(
        "QWEN4_EXP_POOL_HEADROOM_BYTES", 6 * 1024**3
    ),
    "QWEN4_EXP_POOL_SIZING": env_str("QWEN4_EXP_POOL_SIZING", "hbm"),
    # qwen4_exp: shard prefill-sized steps across TP ranks between attention
    # blocks (reduce-scatter instead of all-reduce), so the hyper-connection
    # streams, PLE gate and residual math run on 1/TP of the tokens. Steps
    # below QWEN4_EXP_PREFILL_SP_MIN_TOKENS keep the replicated TP path.
    "QWEN4_EXP_PREFILL_SP": env_bool("QWEN4_EXP_PREFILL_SP"),
    "QWEN4_EXP_PREFILL_SP_MIN_TOKENS": env_int("QWEN4_EXP_PREFILL_SP_MIN_TOKENS", 1024),
    # With QWEN4_EXP_PREFILL_SP=1: the adaptive fused EP MoE takes each
    # rank's own rows on sequence-parallel steps (no replica split/restore)
    # and the shared expert's TP sum is reduce-scattered, not all-reduced.
    "QWEN4_EXP_PREFILL_SP_MOE": env_bool("QWEN4_EXP_PREFILL_SP_MOE"),
    # With QWEN4_EXP_PREFILL_SP=1: the sequence-parallel graph pads each
    # rank's shard to the HC kernels' row tile, so buckets whose shard does
    # not divide it (1024 and 3072 at TP8) take sequence parallelism too.
    "QWEN4_EXP_PREFILL_SP_PAD_HC": env_bool("QWEN4_EXP_PREFILL_SP_PAD_HC"),
    # With QWEN4_EXP_PREFILL_SP_MOE=1: the shared expert of sequence-parallel steps runs
    # on
    # this rank's own rows with a replicated copy of its weights (loaded next to the TP
    # slices,
    # which decode keeps), so it needs no all-gather or reduce-scatter. Costs one full
    # shared
    # expert of HBM per rank.
    "QWEN4_EXP_PREFILL_SP_SHARED_LOCAL": env_bool("QWEN4_EXP_PREFILL_SP_SHARED_LOCAL"),
    # qwen4_exp: run each hyper-connection mix (per-branch norm, low-rank
    # gate, branch mean and the combine's injection gate) at <= 512 tokens as
    # one Pallas call, with the attn combine fused ahead of the mlp mix.
    "TPU_QWEN4_HC_DECODE_FUSED": env_bool("TPU_QWEN4_HC_DECODE_FUSED"),
    # Prefill HC full mix: RMS-norm variant of the kernel's norm loop. vpu (the
    # default, unchanged program); rows64 (64-row loop steps, bitwise); gram /
    # gram64 (each branch's sum of squares as the diagonal of x x^T on the MXU;
    # not bitwise, only the f32 summation order of the variance changes);
    # bf16ep / bf16ep64 (the epilogue n * w, n + . in bf16 arithmetic, bitwise:
    # a bf16 x bf16 product is exact in f32 and the bf16 sum rounds the same).
    "TPU_QWEN4_HC_PREFILL_NORM": env_with_choices(
        "TPU_QWEN4_HC_PREFILL_NORM",
        "vpu",
        ["vpu", "rows64", "gram", "gram64", "bf16ep", "bf16ep64"],
        case_sensitive=False,
    ),
    # Prefill HC combine_mix (attn -> mlp): the gated residual runs as the
    # prologue of the next HC's full-mix kernel, which writes the combined hyper
    # input once instead of a separate merge kernel writing it and the mix
    # reading it back (bit-identical).
    "VLLM_TPU_HC_PREFILL_COMBINE_MIX": env_bool("VLLM_TPU_HC_PREFILL_COMBINE_MIX"),
    # Prefill HC full mix: project the normalized input onto the combine's gate
    # weight in the kernel and return that [T, hc] projection instead of writing
    # the normalized [T, hc * H] input for an XLA projection to read back.
    "VLLM_TPU_HC_PREFILL_INJECT": env_bool("VLLM_TPU_HC_PREFILL_INJECT"),
    # Prefill HC full mix: take the up projection as its [low rank, width]
    # transpose. The checkpoint weight is held column-major, so the transpose
    # is a bitcast and the per-call relayout copy before the kernel goes away
    # (bit-identical).
    "VLLM_TPU_HC_PREFILL_UP_T": env_bool("VLLM_TPU_HC_PREFILL_UP_T"),
    # Prefill HC full mix: token tile for calls whose rows divide two such tiles
    # (0 = always hc_prefill_full_tile). Rows are independent, so the tile only
    # changes the schedule (bit-identical); 256 is ~25 us/call faster than 128
    # at 1024 rows.
    "VLLM_TPU_HC_PREFILL_WIDE_TILE": env_nonnegative_int(
        "VLLM_TPU_HC_PREFILL_WIDE_TILE", 0
    ),
    # Enable and configure the fused Pallas megakernel ("1"/"true" for defaults,
    # "ep"/"tp" for mode, or "mode=tp,buckets=8:16:32,verify_buckets=4:8:16,ab=1").
    "VLLM_TPU_MEGAKERNEL": env_str("VLLM_TPU_MEGAKERNEL", ""),
    # Qwen3.8 megakernel with PLE CPU offload: the segment / whole-step programs take
    # the
    # PLE layer's rows from the host lookup instead of refusing to run (the kernels
    # never
    # read the PLE table; only apply_ple before the PLE layer's head group does).
    "VLLM_TPU_MK_HOST_PLE": env_bool("VLLM_TPU_MK_HOST_PLE"),
    # Qwen4Exp PLE decode trims, same bytes: on steps of <= 8 tokens the head
    # groups' n-gram rows ride the token-embedding all-reduce as integer-valued
    # BF16 lanes (one collective instead of two), and the n-gram hash constants
    # are trace-time Python ints instead of int64 buffers.
    "VLLM_TPU_MK_PLE_TRIM": env_bool("VLLM_TPU_MK_PLE_TRIM"),
    # Whole-step megakernel: also run the last QSA layer (model layer 47) in the
    # whole-step
    # call, as a one-layer kernel after layers 3..46, its step tokens written with
    # theirs;
    # the stock L47 attention core and last segment kernel are skipped.
    "VLLM_TPU_MK_WHOLE_L47": env_bool("VLLM_TPU_MK_WHOLE_L47"),
    # qwen4_exp PLE short conv (request-owned history): skip the history read
    # for the zero-distance tap and gather only each request's first
    # `distance` history rows for the others, instead of one history row per
    # token per tap (bit-identical).
    "VLLM_TPU_PLE_CONV_HISTORY_FAST": env_bool("VLLM_TPU_PLE_CONV_HISTORY_FAST"),
    # Qwen4-exp resident PLE table: tokens per gather in the raw embedding
    # lookup (a multiple of 8). 8 is the originally validated chunk; an
    # 8192-token prefill step then runs 1024 sequential gathers.
    # qwen4_exp PLE: one Pallas kernel for the per-stream normalized key .
    # query gate dot (norm weights, BF16 products, FP32 reduction) instead of
    # two grouped RMS norms and a Torch product/sum. Reduction order differs.
    "VLLM_TPU_PLE_FUSED_GATE": env_bool("VLLM_TPU_PLE_FUSED_GATE"),
    # Host PLE (CPU offload): gather into two reused, pinned-if-possible
    # buffers and copy them to the device without staging (ple_storage.py).
    "VLLM_TPU_PLE_HOST_FAST_LOOKUP": env_bool("VLLM_TPU_PLE_HOST_FAST_LOOKUP"),
    # Host-offloaded PLE table: split lookups of at least 2 * 16384 rows over
    # up to this many threads (disjoint output ranges; 0/1 = serial). Prefill
    # chunks only; the gathered rows are identical.
    "VLLM_TPU_PLE_HOST_GATHER_WORKERS": env_nonnegative_int(
        "VLLM_TPU_PLE_HOST_GATHER_WORKERS", 0
    ),
    # Host PLE: start the ids' copy before the decoder layers that precede the
    # PLE layer and finish the lookup just before it (ple_offload.py).
    "VLLM_TPU_PLE_HOST_OVERLAP": env_bool("VLLM_TPU_PLE_HOST_OVERLAP"),
    # Verify the first N overlapped ids copies against a blocking copy.
    "VLLM_TPU_PLE_HOST_OVERLAP_CHECK": env_nonnegative_int(
        "VLLM_TPU_PLE_HOST_OVERLAP_CHECK", 0
    ),
    # Log the host PLE lookup's D2H / gather / H2D host time every 1000 calls.
    "VLLM_TPU_PLE_HOST_TIMING": env_bool("VLLM_TPU_PLE_HOST_TIMING"),
    "VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS": env_nonnegative_int(
        "VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS", 8
    ),
    # Qwen4-exp QSA: round to BF16 with lax.reduce_precision instead of the
    # integer bit trick. Same RNE; the boundary stays observable to XLA and
    # fuses into neighbouring ops instead of running as its own fusion.
    "VLLM_TPU_QSA_BF16_REDUCE_PRECISION": env_bool(
        "VLLM_TPU_QSA_BF16_REDUCE_PRECISION", default=False
    ),
    # With VLLM_TPU_QSA_MAIN_KV_DCP and VLLM_TPU_QSA_MTP_KV_DCP: compressed keys
    # split the same way; each rank scores its owned groups and the pair merges
    # the local top-k lists into the exact global selection.
    "VLLM_TPU_QSA_COMPRESSED_DCP": env_bool("VLLM_TPU_QSA_COMPRESSED_DCP"),
    # Compressed-key writes: take each pair's two key rows by selecting over
    # statically shifted copies of the keys (offsets within two compression
    # groups of the token) instead of a gather; a lax.cond falls back to the
    # gather when a written row's offset is out of range (bit-identical pool).
    "VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT": env_bool(
        "VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT"
    ),
    # Qwen4-exp pooled QSA: write compressed keys with the run-based writer
    # (kernels/qwen4_exp/compressed_write.py): one DMA per run of consecutive
    # pair rows, all in flight, instead of per-tile / per-row serial writes.
    "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE": env_bool(
        "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE", default=False
    ),
    # With VLLM_TPU_QSA_COMPRESSED_RUN_WRITE: steps with fewer key rows keep the
    # tiled writer (the run writer is slower on decode-sized steps).
    "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS": env_nonnegative_int(
        "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS", 512
    ),
    # Main-KV DCP: pass the whole q/gate row to the pair combine, which slices
    # the gate half itself, instead of slicing it on the Torch side
    # (a standalone strided slice plus relayout; bit-identical).
    "VLLM_TPU_QSA_DCP_GATE_IN_COMBINE": env_bool("VLLM_TPU_QSA_DCP_GATE_IN_COMBINE"),
    # With VLLM_TPU_QSA_MAIN_KV_DCP: the pair exchanges each rank's heads of
    # the partial attention (all-to-all) instead of all-gathering every
    # partial; half the bytes, bit-identical.
    "VLLM_TPU_QSA_DCP_HEAD_EXCHANGE": env_bool("VLLM_TPU_QSA_DCP_HEAD_EXCHANGE"),
    # Decode DCP metadata: the owned-first order of an already ascending
    # selection by a stable partition instead of a [rows, budget] sort.
    "VLLM_TPU_QSA_DCP_META_PARTITION": env_bool("VLLM_TPU_QSA_DCP_META_PARTITION"),
    # Compressed main-KV DCP: all-gather the local top-k values (f32, as int32
    # bits) and group ids (int32) as one stacked tensor instead of two
    # gathers; same bits.
    "VLLM_TPU_QSA_DCP_PACK_LISTS": env_bool("VLLM_TPU_QSA_DCP_PACK_LISTS"),
    # Main-KV DCP prefill (owned-page walk): query block / subtile; 0 keeps the tile
    # table's.
    "VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK": env_nonnegative_int(
        "VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK", 1024
    ),
    "VLLM_TPU_QSA_DCP_PREFILL_QUERY_TILE": env_nonnegative_int(
        "VLLM_TPU_QSA_DCP_PREFILL_QUERY_TILE", 256
    ),
    # Compressed main-KV DCP ops: all-gather the pair's q/gate rank-major
    # (dim 0) and join the heads inside the step, instead of a dim -1 gather
    # whose head join runs as a standalone copy (bit-identical).
    "VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR": env_bool("VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR"),
    # Qwen4-exp token-shard DCP prefill: all-gather only the pair's q heads,
    # not their gates (the DCP attention is ungated; each rank gates its own
    # heads); half the bytes, bit-identical.
    "VLLM_TPU_QSA_DCP_Q_GATHER": env_bool("VLLM_TPU_QSA_DCP_Q_GATHER"),
    # With VLLM_TPU_QSA_COMPRESSED_DCP on the prefill token-shard path: each
    # rank merges its own 1/TP of the rows from its pair's lists and one TP
    # all-gather shares the merged selection, instead of every rank merging
    # every row of every pair's raw lists (bit-identical).
    "VLLM_TPU_QSA_DCP_SHARD_MERGE": env_bool("VLLM_TPU_QSA_DCP_SHARD_MERGE"),
    # Main-KV DCP attention: route the partial output and its lse as two arrays
    # instead of one [.., D + 1] channel (a padded f32 concat plus the slices
    # that undo it at every 8K prefill step); same values.
    "VLLM_TPU_QSA_DCP_SPLIT_LSE": env_bool("VLLM_TPU_QSA_DCP_SPLIT_LSE"),
    # Qwen4-exp QSA decode/verify attention on RPA: compact KV schedule (one
    # entry per lane-tile instead of one per 4-token page), which lets the
    # decode KV block grow past 256 tokens within the SMEM schedule budget.
    # The RPA body has a fixed cost per step, so fewer, larger steps help.
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": env_bool(
        "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE", default=False
    ),
    # Qwen4-exp QSA decode/verify attention on RPA (implies the lean fetch):
    # full KV tiles fetch with static page sizes and page ids read at static
    # offsets, cutting the scalar work per DMA that bounds the decode kernel.
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": env_bool(
        "VLLM_TPU_QSA_DECODE_FAST_FETCH", default=False
    ),
    # Compressed keys per KV block of the shared-stream pass above (a multiple
    # of 128). Only read when VLLM_TPU_QSA_DECODE_QUERY_BLOCK > 0.
    "VLLM_TPU_QSA_DECODE_KV_BLOCK": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_KV_BLOCK", 1024
    ),
    # Qwen4-exp QSA decode/verify attention on RPA: fetch each KV page with
    # one scalar load instead of three (the decode KV fetch issues one DMA
    # per 4-token group and is scalar-issue bound). Same DMAs and results.
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": env_bool(
        "VLLM_TPU_QSA_DECODE_LEAN_FETCH", default=False
    ),
    # With the pool-page table: when the decode KV block holds whole pool
    # pages (VLLM_TPU_QSA_DECODE_KV_BLOCK a multiple of 13 kernel pages x 32
    # keys, e.g. 3328), DMA each pool page's consecutive kernel pages at once
    # instead of one 16 KiB descriptor per kernel page. Same bytes and order.
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": env_bool(
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA", default=False
    ),
    # Qwen4-exp shared-stream decode indexer: pass StreamIndex the pool-page
    # table and expand each entry into its kernel pages inside the KV reader.
    # The scalar-prefetched table shrinks by the kernel pages per pool page
    # (13 at 1664-token pages), so a full-batch verify step scores in one
    # query window instead of several serial ones. Same pages, same order.
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": env_bool(
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE", default=False
    ),
    # Qwen4-exp QSA decode/verify indexer: score all query tokens of a
    # request (e.g. the K+1 MTP verify tokens) in one query block of this many
    # tokens, sharing one stream over the request's compressed keys instead of
    # re-streaming it once per token. Each token keeps its per-token scoring
    # expression, so top-k selections match the per-token path. 0 (default)
    # keeps the per-token path; 4 or 8 covers MTP K <= 3 or K <= 7 per block.
    "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_QUERY_BLOCK", 0
    ),
    # Decode KV tokens per RPA step and lanes (requests) per step with the
    # compact schedule; only read when it is on.
    "VLLM_TPU_QSA_DECODE_RPA_BKV": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_RPA_BKV", 256
    ),
    "VLLM_TPU_QSA_DECODE_RPA_LANES": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_RPA_LANES", 8
    ),
    # Qwen4-exp QSA decode/verify attention with the compact schedule: query
    # rows per RPA call (0 keeps the 64-row SMEM bound of the per-page
    # schedule). Larger steps otherwise run as a scan of 64-row calls.
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES", 0
    ),
    # Qwen4-exp QSA decode/verify attention: present every head slot of the
    # main KV row (zero queries for the other layer's head) so all QSA layers
    # share one decode kernel configuration and XLA computes the decode
    # schedule once per step; RPA also skips its empty mixed pass.
    "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE": env_bool(
        "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE", default=False
    ),
    # Compressed keys per KV block for the one-token requests of that pass on
    # steps of at most one token per request (MTP draft rows, plain decode);
    # verify steps keep 256. Exact selection at any block size. Only read
    # when VLLM_TPU_QSA_DECODE_QUERY_BLOCK > 0.
    "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK", 256
    ),
    # Qwen4-exp QSA decode/verify attention on RPA with FP8 main KV rows:
    # each layer attends its own head slot of the packed row (one KV head,
    # only its K/V pair unpacked) instead of unpacking every slot and, for a
    # paired page's second layer, presenting both heads with zero queries.
    "VLLM_TPU_QSA_DECODE_SLOT_READ": env_bool(
        "VLLM_TPU_QSA_DECODE_SLOT_READ", default=False
    ),
    # Qwen4-exp pooled QSA with VLLM_TPU_QSA_COMPRESSED_DCP, the replicated
    # (non-SP) backbone, i.e. decode / verify and the small prefill buckets:
    # each DCP pair indexes one contiguous 1/(TP / DCP) of the step's rows
    # (token_shard_step) and one TP all-gather brings every pair's local lists
    # for the pair merge, instead of every pair indexing every row. The
    # selection is the replicated indexer's.
    "VLLM_TPU_QSA_DECODE_TOKEN_SHARD": env_bool(
        "VLLM_TPU_QSA_DECODE_TOKEN_SHARD", default=False
    ),
    "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION": env_bool(
        "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION"
    ),
    # With VLLM_TPU_QSA_COMPRESSED_DCP: merge a pair's local top-k lists into the
    # global selection with one TensorCore Pallas kernel (radix select, tie
    # cut, bitonic sort) instead of the SparseCore top-k + keep mask + sort
    # (bit-identical).
    "VLLM_TPU_QSA_FUSED_DCP_BLOCKS": env_bool("VLLM_TPU_QSA_FUSED_DCP_BLOCKS"),
    # Qwen4-exp pooled QSA decode/verify steps: write main KV, compressed
    # indexer keys and the raw ring/tags of a layer with one Pallas kernel
    # whose row DMAs are all in flight together, and read the ring with every
    # request's DMA outstanding. Prefill-sized steps keep the tiled writers.
    "VLLM_TPU_QSA_FUSED_KV_WRITE": env_bool(
        "VLLM_TPU_QSA_FUSED_KV_WRITE", default=False
    ),
    # Qwen4-exp QSA decode/verify attention: run the decode rows (and the
    # decode prefix of mixed steps) on the fused sparse GQA decode kernel
    # (kernels/sparse_gqa_decode) instead of batched RPA. Inputs outside the
    # kernel's static envelope (non-TPU7x, unsupported shapes) still use RPA.
    "VLLM_TPU_QSA_FUSED_SPARSE_DECODE": env_bool(
        "VLLM_TPU_QSA_FUSED_SPARSE_DECODE", default=False
    ),
    # Qwen4-exp QSA indexer: build StreamIndex's step-level metadata (and the
    # pass schedules of decode / verify calls) outside the early-exit cond, so
    # XLA computes them once per step instead of once per QSA layer.
    "VLLM_TPU_QSA_INDEXER_HOIST_META": env_bool(
        "VLLM_TPU_QSA_INDEXER_HOIST_META", default=False
    ),
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": env_bool(
        "VLLM_TPU_QSA_INDEXER_MIXED_ONLY", default=False
    ),
    # Qwen4-exp shared-stream decode indexer: SparseCore top-k copies only
    # each score row's live prefix (one subcore per row) instead of every
    # subcore copying its whole slice of a row sized for max_model_len.
    # Same winners; selection time follows the context, not the table width.
    "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY": env_bool(
        "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY", default=False
    ),
    # Qwen4-exp pooled QSA: split the lightning indexer across the TP group.
    # Each rank scores an interleaved 1/TP of every request's compressed keys,
    # the local top-k candidates are gathered, and every rank merges the exact
    # global top-k. Off: every rank scores every key (replicated indexer).
    "VLLM_TPU_QSA_INDEXER_SHARD": env_bool("VLLM_TPU_QSA_INDEXER_SHARD", default=False),
    # With VLLM_TPU_QSA_INDEXER_SHARD: steps of at least this many tokens merge
    # a 1/TP token block per rank after an all-to-all and all-gather the merged
    # blocks, instead of all-gathering every rank's candidates. 0: never.
    "VLLM_TPU_QSA_INDEXER_SHARD_A2A_MIN_TOKENS": env_nonnegative_int(
        "VLLM_TPU_QSA_INDEXER_SHARD_A2A_MIN_TOKENS", 0
    ),
    # With VLLM_TPU_QSA_INDEXER_SHARD: the prefill sequence-parallel backbone
    # (QWEN4_EXP_PREFILL_SP, steps of at least its minimum token count) keeps
    # the replicated indexer; only the smaller steps shard it. The prefill
    # candidate exchange costs more than the sharded scoring saves. The two
    # backbones are separate graphs with disjoint buckets, so the choice is
    # static per graph.
    "VLLM_TPU_QSA_INDEXER_SHARD_SKIP_SP": env_bool(
        "VLLM_TPU_QSA_INDEXER_SHARD_SKIP_SP", default=False
    ),
    # Qwen4-exp decode / verify indexer: StreamIndex pads each token's score
    # rows to whole (8, 128) int32 tiles, so its HBM scores reach the
    # SparseCore top-k as a bitcast instead of a relayout copy (~8 us per QSA
    # layer at c48). The padding lies past every row length (same selection).
    "VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES": env_bool(
        "VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES", default=False
    ),
    # Qwen4-exp pooled QSA, prefill sequence-parallel backbone only: each TP
    # rank runs the indexer on one contiguous 1/TP of the gathered step's
    # tokens and the selected blocks are all-gathered, instead of every rank
    # indexing every token. The selection is the replicated indexer's. Takes
    # precedence over VLLM_TPU_QSA_INDEXER_SHARD in that graph.
    "VLLM_TPU_QSA_INDEXER_TOKEN_SHARD": env_bool(
        "VLLM_TPU_QSA_INDEXER_TOKEN_SHARD", default=False
    ),
    # QSA indexer cache updates: build each token's compression-group members
    # from static shifts and the ring / tag rows from a one-hot product instead
    # of per-token row gathers (same values; ~3 ms of gathers per 8K prefill).
    "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT": env_bool(
        "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT", default=False
    ),
    # Main KV of the target QSA layers split over TP rank pairs that share a
    # KV head (2; 0/1 = off): each rank stores and attends half the tokens,
    # the pair's partial attention is LSE-combined.
    "VLLM_TPU_QSA_MAIN_KV_DCP": env_nonnegative_int("VLLM_TPU_QSA_MAIN_KV_DCP", 0),
    # Qwen4-exp QSA: run-based main KV writer for prefill-sized steps
    # (kernels/qwen4_exp/main_write.py): one DMA per run of consecutive rows
    # (power-of-two pieces), all in flight, instead of per-row DMAs in every
    # 128-token tile that is not page aligned and fully valid.
    "VLLM_TPU_QSA_MAIN_RUN_WRITE": env_bool(
        "VLLM_TPU_QSA_MAIN_RUN_WRITE", default=False
    ),
    # With VLLM_TPU_QSA_MAIN_RUN_WRITE: steps with fewer rows keep the tiled writers.
    "VLLM_TPU_QSA_MAIN_RUN_WRITE_MIN_TOKENS": env_nonnegative_int(
        "VLLM_TPU_QSA_MAIN_RUN_WRITE_MIN_TOKENS", 512
    ),
    # Rows per grid step of the run writer (a power of two); runs never cross a
    # tile, and a tile's DMAs all complete before the next tile starts.
    "VLLM_TPU_QSA_MAIN_RUN_WRITE_TILE": env_nonnegative_int(
        "VLLM_TPU_QSA_MAIN_RUN_WRITE_TILE", 512
    ),
    # With VLLM_TPU_QSA_MAIN_KV_DCP: the MTP draft layers' main KV split the
    # same way (otherwise the draft keeps whole pages).
    "VLLM_TPU_QSA_MTP_KV_DCP": env_bool("VLLM_TPU_QSA_MTP_KV_DCP"),
    # Qwen4-exp QSA prefill attention: feed the MXU BF16 operands (exact
    # queries and keys, probabilities split into two BF16 terms) instead of
    # FP32 operands, which Mosaic multiplies at full FP32 precision.
    "VLLM_TPU_QSA_PREFILL_BF16_MXU": env_bool(
        "VLLM_TPU_QSA_PREFILL_BF16_MXU", default=False
    ),
    # Qwen4-exp QSA prefill: build the per-query selection bitset with a
    # compare-and-accumulate Pallas kernel. Off falls back to the vmapped
    # scatter-add, which XLA:TPU lowers to a global sort of every selection.
    "VLLM_TPU_QSA_PREFILL_BITSET_KERNEL": env_bool(
        "VLLM_TPU_QSA_PREFILL_BITSET_KERNEL", default=True
    ),
    # Main-KV DCP sparse prefill: the selection bitset kernel emits this rank's
    # words only, instead of a global bitset then an owned-word gather
    # (bit-identical).
    "VLLM_TPU_QSA_PREFILL_BITSET_OWNED": env_bool("VLLM_TPU_QSA_PREFILL_BITSET_OWNED"),
    "VLLM_TPU_QSA_PREFILL_INDEXER_BKV": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_INDEXER_BKV", 256
    ),
    # Qwen4-exp ragged prefill indexer tiles: queries per block of the prefill
    # passes and compressed keys per KV block (a multiple of 256). Defaults
    # 128 / 256 are the served tiles. Each score keeps its own expression, so
    # the winners do not depend on the tiling. BKV=1024 also aligns the score
    # row width to whole sublane tiles, so the SparseCore's 1-D view of the
    # scores needs no relayout copy before the top-k.
    "VLLM_TPU_QSA_PREFILL_INDEXER_BQ": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_INDEXER_BQ", 128
    ),
    # Qwen4-exp prefill indexer: the same bounded SparseCore score copy for
    # prefill chunks (rows early in a prompt see a short context).
    # Qwen4-exp ragged prefill indexer: score/select in chunks of this many
    # tokens, overlapping chunk g's SparseCore top-k with chunk g+1's scoring.
    # 0 (default) keeps 512-token chunks for steps that are a multiple of 512
    # and one unchunked call otherwise. A value that does not divide the step
    # (or is not a multiple of 8) falls back to the default. Per-row scores
    # and winners do not depend on the chunking.
    "VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK", 0
    ),
    # qwen4_exp QSA sparse prefill: skip the FP8 key finite/written guard,
    # which the score mask select already covers (bit-identical output).
    "VLLM_TPU_QSA_PREFILL_LEAN": env_bool("VLLM_TPU_QSA_PREFILL_LEAN"),
    # Sparse prefill with lse (main-KV DCP): mask dead rows only in the padded
    # layouts; the aligned layout's kernel rows are already 0 / -inf there
    # (bit-identical, drops an f32 [T, heads, D] select per call).
    "VLLM_TPU_QSA_PREFILL_LIVE_MASK_PADDED": env_bool(
        "VLLM_TPU_QSA_PREFILL_LIVE_MASK_PADDED"
    ),
    # qwen4_exp QSA sparse prefill: fetch each KV tile with one DMA per
    # page-sized run instead of one per 128-row selection segment, when KV
    # tiles and pages nest (TP8: 1024-row tiles over 512-row pages; DCP: one
    # page per tile). Same rows, bit-identical.
    "VLLM_TPU_QSA_PREFILL_PAGE_COPIES": env_bool("VLLM_TPU_QSA_PREFILL_PAGE_COPIES"),
    # qwen4_exp QSA sparse prefill: when a step's selections and causal lengths
    # fit a page-table prefix, build the selection metadata (bitset, active
    # tiles) over that prefix instead of max_model_len (bit-identical).
    "VLLM_TPU_QSA_PREFILL_PREFIX_PAGES": env_bool("VLLM_TPU_QSA_PREFILL_PREFIX_PAGES"),
    "VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY": env_bool(
        "VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY", default=False
    ),
    "VLLM_TPU_QSA_PREFILL_UNION_ALPHA": env_float(
        "VLLM_TPU_QSA_PREFILL_UNION_ALPHA", 0.0
    ),
    "VLLM_TPU_QSA_PREFILL_UNION_CAP": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_UNION_CAP", 12288
    ),
    # COMPACT_COPY: how the compact pool is copied ("vmem": whole pages staged
    # through VMEM; "pipe": a Pallas pipeline of 32-bit words; "dma": HBM-to-HBM
    # DMAs, slow). All byte-exact.
    "VLLM_TPU_QSA_PREFILL_UNION_COMPACT_COPY": env_str(
        "VLLM_TPU_QSA_PREFILL_UNION_COMPACT_COPY", "vmem"
    ),
    # COMPACT_REQS (N > 0; with GATHERED): the SparseCore takes at most 16 GiB
    # per tensor, so the gather reads a per-step compact pool of the first N
    # requests' pages (whole-page copies); a step with more requests takes the
    # listed walk on the full pool (bitwise). 0: the gather reads the pool.
    "VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS", 0
    ),
    # GATHERED ("", "reference", "sparsecore"; with LISTED): the walk-free
    # kernel on chunked SparseCore-gathered union tiles.
    "VLLM_TPU_QSA_PREFILL_UNION_GATHERED": env_str(
        "VLLM_TPU_QSA_PREFILL_UNION_GATHERED", ""
    ),
    "VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS": env_int(
        "VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS", 4
    ),
    # GATHER_GROWTH (>= 1; 1 = even chunks): chunk k of the gathered walk holds
    # about growth**k as many query groups as chunk 0. Bitwise.
    "VLLM_TPU_QSA_PREFILL_UNION_GATHER_GROWTH": env_float(
        "VLLM_TPU_QSA_PREFILL_UNION_GATHER_GROWTH", 1.0
    ),
    # KV tile of the union walk: union groups per tile times 4 (256-2048).
    "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE", 1024
    ),
    # LAZY_SHARE (with SHARE_BITS): walk on group-aligned rows; only a fallback
    # builds the block-sparse layout.
    "VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE": env_bool(
        "VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE"
    ),
    # Union walk options (all off by default; bitwise unless noted):
    # LISTED: issue the walk's copies from a union id list instead of scanning
    # the union bitset; "sparsecore" (union_sc_list, TPU) or "reference" (XLA).
    # SHARE_BITS: a cap fallback reuses the walk plan's selection bitset.
    # ALPHA: also fall back when a group's union exceeds this fraction of its
    # visible groups (0 = off). QK16: QK as one BF16 MXU pass (exact operands,
    # power-of-two score scales only).
    "VLLM_TPU_QSA_PREFILL_UNION_LISTED": env_str(
        "VLLM_TPU_QSA_PREFILL_UNION_LISTED", ""
    ),
    "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", 32768
    ),
    "VLLM_TPU_QSA_PREFILL_UNION_QK16": env_bool("VLLM_TPU_QSA_PREFILL_UNION_QK16"),
    # RUN_TABLES (with LISTED; parked): the walk issued from SparseCore run tables.
    "VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES": env_bool(
        "VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES"
    ),
    # SC_BITSET: the union plan's per-row bitset built on the SparseCore (the
    # same bits), off the TensorCore.
    "VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET": env_bool(
        "VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET"
    ),
    "VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS": env_bool(
        "VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS"
    ),
    # Qwen4-exp QSA prefill: union-walk attention (qsa_union_dense.
    # union_walk_prefill) with query groups of this many rows (a divisor of 128,
    # at least 8; 0, the default, keeps the block-sparse kernel). Each group of
    # rows attends only the union of its rows' selected 4-token groups, DMA'd
    # straight from the paged pool, instead of every KV tile any row of a
    # 256-row query block selected. Taken for D256 steps (24/2 or a TP rank's
    # 3/1, 6/1, 12/1 query/KV heads) on an FP8 pool of 128-aligned (1, 4, 256)
    # pages, at least 1024 tokens and a multiple of the group, without DCP, a KV
    # head offset or LSE output. Each step falls back to the block-sparse kernel
    # when its longest row is below VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT
    # tokens or a group's union exceeds VLLM_TPU_QSA_PREFILL_UNION_CAP 4-token
    # groups. Not bitwise against the block-sparse kernel (union-order online
    # softmax, equal precision); accuracy gated.
    "VLLM_TPU_QSA_PREFILL_UNION_WALK": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_UNION_WALK", 0
    ),
    # qwen4_exp QSA sparse prefill: cut the per-score vector (VPU) work of the
    # block-sparse kernel's inner loop (block_sparse_gqa_prefill, with or
    # without DCP). "" / "0" = off; "1" = the default set (mask,qk16,pv16); or
    # a comma list of: mask (always on: per-segment selection bits, causal and
    # written guards only on the diagonal KV tile; exact), qk16 (one BF16 QK
    # MXU pass; exact when the score scale is a power of two), pv16 (BF16
    # probabilities into the PV dot; bit-identical on v7x, where Mosaic
    # already runs the FP32 dot as one BF16 pass).
    # Precedence: VLLM_TPU_QSA_PREFILL_UNION_WALK replaces the block-sparse
    # kernel where it applies; this flag acts on every remaining call (DCP,
    # kv_head_offset layers, short contexts, small steps, the union walk's
    # shared fallback).
    "VLLM_TPU_QSA_PREFILL_VPU": env_str("VLLM_TPU_QSA_PREFILL_VPU", ""),
    # Mixed QSA steps: move the decode prefix off the prefill rows with
    # contiguous dynamic slices instead of row gathers (same values; prefill
    # chunks spent ~3 ms/8K on the gathers).
    "VLLM_TPU_QSA_PREFIX_SLICE": env_bool("VLLM_TPU_QSA_PREFIX_SLICE", default=False),
    # Qwen4-exp QSA ring/tag state I/O: concurrent row DMAs per tile. 0 uses
    # the measured table (pool pages without an entry, e.g. 512-token FP8
    # pages, fall back to one serial DMA per request); N > 0 uses the largest
    # power of two <= N that divides the request rows.
    "VLLM_TPU_QSA_RING_DMA_BATCH": env_nonnegative_int(
        "VLLM_TPU_QSA_RING_DMA_BATCH", 0
    ),
    # Qwen4-exp pooled QSA: request row r owns ring row r, so the indexer
    # update's whole-row ring/tag gathers and final scatter become selects.
    "VLLM_TPU_QSA_RING_SELECT": env_bool("VLLM_TPU_QSA_RING_SELECT", default=False),
    # Debug only: every N QSA selections, print top-k statistics (run lengths
    # of selected groups, overlap of a request's verify rows) from a host
    # callback. 0 (default) adds nothing to the program.
    "VLLM_TPU_QSA_TOPK_STATS": env_nonnegative_int("VLLM_TPU_QSA_TOPK_STATS", 0),
    # qwen4_exp QSA (unsharded indexer path): when every row's completed groups
    # fit the top-k budget, take all visible groups and skip the StreamIndex
    # scoring call (the sharded path already does). Same selected set; the
    # order of the selected groups may differ from the kernel's.
    "VLLM_TPU_QSA_VISIBLE_BYPASS": env_bool("VLLM_TPU_QSA_VISIBLE_BYPASS"),
    # Qwen4Exp: run the adapter's per-step input preparation (n-gram context,
    # speculative record, slots) as one compiled program instead of ~20 eager
    # launches; models/vllm/qwen4_exp/inputs.py fused_step_inputs.
    "VLLM_TPU_QWEN4_FUSED_PREP": env_bool("VLLM_TPU_QWEN4_FUSED_PREP"),
    # "check": compare the fused input preparation with the eager path on every
    # live step (counts logged every 500 checks); "fault": its positive control.
    "VLLM_TPU_QWEN4_FUSED_PREP_CHECK": lambda: os.getenv(
        "VLLM_TPU_QWEN4_FUSED_PREP_CHECK", ""
    ),
    # The fused program returns the updated speculative record instead of
    # mutating its inputs (no eager copy-back per record tensor after it).
    "VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL": env_bool(
        "VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL"
    ),
    # Qwen4-exp QSA indexer on decode / verify steps: score every request in the
    # mixed pass and drop the two one-token decode passes (empty on verify
    # steps). Bypasses VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK on one-token steps.
    # Qwen4-exp MTP: the draft's QSA layer attends the target's last QSA layer
    # selection (the request's last verified row) instead of scoring its own
    # indexer: a selection-sharing approximation, gated on real AL.
    # Qwen4Exp n-gram context: "1" = H2D only each row's window of the host
    # token store that the device context can address
    # ([host_ctx - K - ngram + 1, host_ctx), K = spec tokens) instead of the
    # whole [max_num_seqs, max_model_len] table every step (16 MiB at c1, ~2 ms
    # of transfer); "check" = both paths and log mismatches for the first 200
    # steps, then as "1" (models/vllm/qwen4_exp/inputs.py).
    "VLLM_TPU_QWEN4_NGRAM_WINDOW": lambda: os.getenv("VLLM_TPU_QWEN4_NGRAM_WINDOW", ""),
    # Per-shape tile tables (kernels/qwen4_exp/shape_tuned.py): kernels look up
    # measured tiles by their static shape bucket (decode rows, prefill chunk
    # tokens) and override the env-var tiles. Off keeps every env tile as set.
    "VLLM_TPU_SHAPE_TUNED": env_bool("VLLM_TPU_SHAPE_TUNED", default=False),
    # Qwen4Exp unified pool: finish a prefill in one chunk instead of
    # splitting off the tokens past its last Mamba block boundary, and write
    # the GDN / conv / PLE state at that boundary from inside the step
    # (core/mamba_tail_fold.py). Engine core and workers.
    "VLLM_TPU_MAMBA_TAIL_FOLD": env_bool("VLLM_TPU_MAMBA_TAIL_FOLD", default=False),
    # Ignored (hard-off): the fold under the Qwen4Exp main-KV DCP gave wrong
    # outputs (za6 GSM8K c32 0.26, c48 0.008). The fold turns itself off under
    # any context parallelism.
    "VLLM_TPU_MAMBA_TAIL_FOLD_DCP": env_bool(
        "VLLM_TPU_MAMBA_TAIL_FOLD_DCP", default=False
    ),
    # With VLLM_TPU_MAMBA_TAIL_FOLD (and speculative decoding): also fold a
    # prompt's last chunk when it holds no Mamba block boundary, e.g. a prefix
    # hit at 82944 and a 473-token rest that the fine unit would split 256 +
    # 217. The checkpoint goes to the request's first speculative scratch
    # block and is moved to its own block before the next forward
    # (core/mamba_tail_fold.py).
    "VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY": env_bool(
        "VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY", default=False
    ),
    # With VLLM_TPU_MAMBA_TAIL_FOLD: the granularity (tokens) of the internal
    # prefill checkpoint and of prefix-cache hits. 0 = the Mamba block (512,
    # unchanged). A finer unit (e.g. 256) that divides the Mamba block and is a
    # multiple of the largest GDN prefill tile (128, 256 with TPU_GDN_CHUNK_PAIR)
    # checkpoints the prompt nearer its end, so the next turn's prefix hit
    # stops at most unit - 1 tokens short of the prior prompt instead of up
    # to 511. Also sets --prefix-match-unit to the unit when it is unset.
    "VLLM_TPU_MAMBA_TAIL_FOLD_UNIT": env_int("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", 0),
    # Give the adaptive fused MoE per-channel weight scales [E, N] buffers of
    # their own at load time (vllm_adapter.cache_channel_scales_2d); squeezing
    # the GMM [E, 1, 1, N] scales in the hot path relayouts both scales of
    # every MoE layer on every forward.
    "MOE_ADAPTIVE_2D_CHANNEL_SCALES": env_bool("MOE_ADAPTIVE_2D_CHANNEL_SCALES"),
    # GDN flags of the model-owned GDN op/kernel copy (ops/gdn_attention_op.py,
    # kernels/gdn_v3/), moved from envs.py with it.
    # GDN chunked prefill: each grid step takes two chunks of the same
    # sequence. The state-independent intra-chunk work (gates, K K^T, the
    # triangular inverse, U/W, masked Q K^T) runs once for both chunks batched
    # over heads, then the two state updates run in order. Same ops per chunk,
    # so bit-identical. A sequence's odd last chunk still costs a full pair
    # (its empty half is masked).
    "TPU_GDN_CHUNK_PAIR": env_bool("TPU_GDN_CHUNK_PAIR", default=False),
    # GDN v3: the first kernel of a call zero-fills the output rows no kernel
    # writes (bucket padding) instead of XLA zero-initializing the whole
    # output buffer first. Bit-identical.
    "TPU_GDN_KERNEL_ZERO_OUT": env_bool("TPU_GDN_KERNEL_ZERO_OUT", default=False),
    # GDN v3: the same for the b / a gate inputs ([tokens, n_v] as is, no FP32
    # cast and 128-lane pad in XLA). GDN only (not KDA).
    "TPU_GDN_RAW_BA": env_bool("TPU_GDN_RAW_BA", default=False),
    # GDN v3: hand the kernel the projection's qkv [tokens, dim] as is (BF16)
    # instead of an FP32 [tokens, 1, dim] copy. The kernel DMAs a tile-aligned
    # row block and casts / places the rows in VMEM. Same values (BF16 -> FP32
    # is exact), so bit-identical; removes an XLA convert + relayout copy per
    # layer.
    "TPU_GDN_RAW_QKV": env_bool("TPU_GDN_RAW_QKV", default=False),
    # Hand the pooled GDN kernel a load-time float32 copy of the conv weight
    # in its `[kernel_size, 1, dim]` order. Otherwise every GDN layer swaps
    # and casts the bf16 `[dim, 1, kernel_size]` weight on every forward.
    "VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR": env_bool("VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR"),
    # Compute the pooled GDN step metadata (read-offset gather, int32 casts,
    # PER_SEQ roll) once per step and KV-cache group instead of in every GDN
    # layer: each layer's op is a separate program, so XLA cannot share them.
    # Same integers; spec decoding on the plain unified pool only.
    "VLLM_TPU_GDN_HOIST_METADATA": env_bool("VLLM_TPU_GDN_HOIST_METADATA"),
    # GDN v3 static tiling: round the mixed tile down to a power of two so a
    # 24/40-row bucket stays divisible by the triangular solve's block, and
    # assert that divisibility (PR #2821 ece3d83b5). Dynamic tiling
    # (TPU_ENABLE_GDN_DYNAMIC_TILING) already picks powers of two.
    "VLLM_TPU_GDN_MIXED_TILE_POW2": env_bool("VLLM_TPU_GDN_MIXED_TILE_POW2"),
}


# Read only by pool sizing, which reaches the compiled graphs through
# num_gpu_blocks; the selector picks the model, which the key already covers.
_COMPILE_KEY_IGNORED = frozenset(
    {
        "VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL",
        "QWEN4_EXP_MAX_POOL_BYTES",
        "QWEN4_EXP_POOL_SIZING",
        "QWEN4_EXP_POOL_GRAPH_COPIES",
        "QWEN4_EXP_POOL_HEADROOM_BYTES",
    }
)


def compile_key_factors() -> dict[str, Any]:
    """The settings that can change a compiled graph (compile cache key)."""
    return {
        name: getter()
        for name, getter in environment_variables.items()
        if name not in _COMPILE_KEY_IGNORED
    }


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())
