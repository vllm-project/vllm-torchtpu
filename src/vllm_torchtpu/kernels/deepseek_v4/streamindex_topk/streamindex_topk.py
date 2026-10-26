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
"""TPU-Friendly StreamIndex Top-K kernel."""

import dataclasses
import functools
import logging
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    BITS_NEG_INF,
    sparsecore_topk,
    sparsecore_topk_dcp_merge,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    bref_override,
    config,
    dcp_sc_compact,
    metadata,
)

try:
    from vllm_torchtpu.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

MlaCase = config.MlaCase
KVLayout = config.KVLayout
DEFAULT_BUFFER_COUNT = config.DEFAULT_BUFFER_COUNT
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
# K-IDXM group maxima: 16 lane tiles (2048 columns) per packed 128-lane gm row.
GM_TILES = 16
GM_LEVELS = 4  # log2(GM_TILES)
GM_F32 = __import__("os").environ.get("TPU_DSV4_GM_INT", "0") != "1"
# Test-only: TPU_KIDXM_FAULT=gm_bad (positive control: odd tiles' pair max skipped)
# or gm_flat (the v1 flat roll form, a timing arm).
_GM_FAULT = __import__("os").environ.get("TPU_KIDXM_FAULT", "")


def _monotone_key(bits):
    """f32 bits -> int32 key ordered as the floats (the SparseCore top-k's)."""
    return jnp.bitwise_xor(
        bits, jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(0x7FFFFFFF))
    )
DCP_AXIS_NAME = "dcp"

cp_local_to_global = metadata.cp_local_to_global
cp_local_length = metadata.cp_local_length
cp_owner_rank = metadata.cp_owner_rank
cp_global_to_local = metadata.cp_global_to_local
cp_rank_as_data = metadata.cp_rank_as_data


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


@dataclasses.dataclass(frozen=True)
class _BlockedPrefillConfig:
    """Multi-token (prefill / mixed) scoring layout and tiling configuration.

    The head reduction runs over blocks of `tok_blk` tokens and `lane_chunk` KV
    positions. Heads are dealt round-robin into one group per MXU (head `h` goes
    to group `h % G`), and the cached queries are ordered `(head_group,
    token_block, head, token)`: each chunk's `q @ k` is one matmul per head
    group, which the compiler spreads over the MXUs, and every MXU emits its rows
    in the exact order the per-head accumulation consumes them. The
    lane-broadcast indexer weights are cached once per q block instead of being
    rebuilt every step.
    """

    enabled: bool
    tok_blk: int
    num_tok_blks: int
    num_head_groups: int
    heads_per_group: int
    rows_per_group_tok_blk: int
    rows_per_group: int
    lane_chunk: int
    num_lane_chunks: int
    tiles_per_chunk: int
    strided_score_store: bool
    score_pitch: int
    bf16_reduce: bool
    wb_dtype: Any

    @classmethod
    def create(
        cls,
        *,
        bq_sz: int,
        bkv_sz: int,
        num_q_heads: int,
        q_dtype: Any,
        seq_batch_size: int,
    ) -> "_BlockedPrefillConfig":
        """Computes the blocked-prefill layout parameters for a kernel pass."""
        tok_blk = math.gcd(bq_sz, 32)
        num_tok_blks = bq_sz // tok_blk
        num_head_groups = math.gcd(num_q_heads, pltpu.get_tpu_info().num_mxus)
        heads_per_group = num_q_heads // num_head_groups

        lane_chunk = math.gcd(bkv_sz, 256)
        num_lane_chunks = bkv_sz // lane_chunk
        tiles_per_chunk = lane_chunk // 128

        q_sublane_tiling = 8 * get_dtype_packing(q_dtype)
        enabled = seq_batch_size == 1 and bq_sz > 1 and tok_blk % q_sublane_tiling == 0

        # Scores are written straight into the [bq, nsub, 128] block with strided
        # stores when every token owns whole (8, 128) tiles of it.
        num_sublanes_bkv = bkv_sz // 128
        strided_score_store = num_sublanes_bkv % 8 == 0
        # Rows between consecutive tokens of the VMEM score block. The sublanes of
        # one store must fall in distinct VMEM banks (row mod 32), so with a pitch
        # of nsub rows, a multiple of 16 for the usual block sizes, every strided
        # store splits into 2-row pieces (1-row at nsub = 32), and a bundle holding
        # two stores cannot also pop the MXU. A pitch of 8 mod 16 rows lets each
        # store cover 4 rows; the padding rows are never copied out.
        score_pitch = num_sublanes_bkv
        if enabled and strided_score_store and score_pitch % 16 == 0:
            score_pitch += 8

        # Reduced-precision head reduction (used for FP8 queries):
        # relu(q @ k) is rounded to bf16 and the weights are held in bf16, so each
        # weighted term is an exact bf16 x bf16 product accumulated in f32. The
        # compiler lowers that pattern to one multiply-accumulate per packed bf16
        # register, which halves the VPU work of the reduction; see
        # `reduce_token_block`. It needs whole packed (16, 128) tiles per token
        # block.
        bf16_reduce = get_dtype_bitwidth(q_dtype) == 8 and tok_blk % 16 == 0
        wb_dtype = jnp.bfloat16 if bf16_reduce else jnp.float32

        return cls(
            enabled=enabled,
            tok_blk=tok_blk,
            num_tok_blks=num_tok_blks,
            num_head_groups=num_head_groups,
            heads_per_group=heads_per_group,
            rows_per_group_tok_blk=heads_per_group * tok_blk,
            rows_per_group=heads_per_group * bq_sz,
            lane_chunk=lane_chunk,
            num_lane_chunks=num_lane_chunks,
            tiles_per_chunk=tiles_per_chunk,
            strided_score_store=strided_score_store,
            score_pitch=score_pitch,
            bf16_reduce=bf16_reduce,
            wb_dtype=wb_dtype,
        )

    def cache_row(self, h: int, tb: int) -> int:
        """First `q_hm_ref` row of head `h` in token block `tb`."""
        g, hg = h % self.num_head_groups, h // self.num_head_groups
        return (
            g * self.rows_per_group
            + tb * self.rows_per_group_tok_blk
            + hg * self.tok_blk
        )


def kernel(
    # Prefetch
    seq_lens_ref,  # [max_num_seqs]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [3] (start_seq, end_seq, chunk_token_start)
    metadata_ref,  # MetadataRef
    cp_rank_ref,  # [1]
    # Input
    page_indices_hbm_ref,  # [max_num_seqs * pages_per_seq]
    q_hbm_ref,
    indexer_weights_hbm_ref,
    cache_kv_hbm_ref,
    *score_refs,
    seq_batch_size: int,
    **kwargs,
):
    """`score_pipeline` over an SMEM copy of this pass's page-table rows.

    A scalar prefetch would copy the whole table into SMEM before the kernel
    starts, ~4 us at 64 sequences x 1024 pages; a pass reads only the rows of
    its own sequences (and of the padding ones that fill its last tile).
    """
    max_num_seqs = seq_lens_ref.shape[0]
    pages_per_seq = page_indices_hbm_ref.shape[0] // max_num_seqs
    start_seq = start_end_seq_idx_ref[0]
    end_seq = jnp.minimum(start_end_seq_idx_ref[1] + seq_batch_size - 1, max_num_seqs)

    def body(page_indices_ref, sem):
        # SMEM is tiled by 128 words, so a window has to start on a tile; a table
        # whose rows are not whole tiles is copied whole.
        if pages_per_seq % 128 == 0:
            rows = pl.ds(
                start_seq * pages_per_seq,
                jnp.maximum(end_seq - start_seq, 0) * pages_per_seq,
            )
        else:
            rows = pl.ds(0, page_indices_hbm_ref.shape[0])
        copy = pltpu.make_async_copy(
            page_indices_hbm_ref.at[rows], page_indices_ref.at[rows], sem
        )
        copy.start()
        copy.wait()
        score_pipeline(
            seq_lens_ref,
            page_indices_ref,
            cu_q_lens_ref,
            start_end_seq_idx_ref,
            metadata_ref,
            cp_rank_ref,
            q_hbm_ref,
            indexer_weights_hbm_ref,
            cache_kv_hbm_ref,
            *score_refs,
            seq_batch_size=seq_batch_size,
            **kwargs,
        )

    pl.run_scoped(
        body,
        pltpu.SMEM(page_indices_hbm_ref.shape, page_indices_hbm_ref.dtype),
        pltpu.SemaphoreType.DMA(()),
    )


def score_pipeline(
    # Prefetch
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [3] (start_seq, end_seq, chunk_token_start)
    metadata_ref,  # MetadataRef
    cp_rank_ref,  # [1]
    # Input
    q_hbm_ref,  # Shape: [max_num_tokens, num_q_heads, head_dim], Memory: HBM
    indexer_weights_hbm_ref,  # Shape: [max_num_tokens, num_q_heads], Memory: HBM
    cache_kv_hbm_ref,
    # HEAD_ALONG_SUBLANE:
    #   [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    # SEQ_ALONG_LANE: [total_num_pages, kv_sublane_groups, kv_packing, page_size]
    # HBM scores, [chunk_tokens, num_sublanes_total, 128]: (out,), or
    # (donated_in, out) after the first pass. Only the last is ever read.
    *score_refs,
    compression_ratio: int,
    static_q_len: int | None,
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
    buffer_count: int = 3,
    case: MlaCase = MlaCase.MIXED,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    cp_size: int = 1,
    interleave_c: int = 1,
    chunk_idx: int = 0,
    verify_bd: bool = False,
    group_max: bool = False,
    gm_min_kv: int = 0,
):
    """Executes the n-buffered Pallas scoring pipeline for StreamIndex Top-K.

    group_max (K-IDXM): the multi-token (blocked) path also writes, per token,
    the maximum monotone key of every 16 consecutive score columns, packed 128
    groups per 2048 columns (see `sparsecore_topk_pf`). The last two refs are
    then (scores, group maxima)."""
    scores_hbm_ref = score_refs[-2] if group_max else score_refs[-1]
    gm_hbm_ref = score_refs[-1] if group_max else None
    buffer_cnt = buffer_count
    seq_along_lane = kv_layout == KVLayout.SEQ_ALONG_LANE

    _, num_q_heads, head_dim = q_hbm_ref.shape

    if seq_along_lane:
        # [total_num_pages, kv_sublane_groups, kv_packing, page_size] where the
        # leading `head_dim` sublane rows hold the FP8 key bytes and the next row
        # holds the per-token UE8M0 scale.
        (_, kv_sublane_groups, kv_packing, page_size) = cache_kv_hbm_ref.shape
        kv_head_dim_groups = head_dim // kv_packing
        page_size_per_kv_packing = page_size // kv_packing
    else:
        (
            _,
            page_size_per_kv_packing,
            kv_packing,
            _,
        ) = cache_kv_hbm_ref.shape
        page_size = page_size_per_kv_packing * kv_packing
        kv_sublane_groups = None
        kv_head_dim_groups = None

    max_num_seqs = seq_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    pages_per_seq = num_page_indices // max_num_seqs

    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_p * page_size
    num_sublanes_bkv = bkv_sz // 128
    shift_comp = (compression_ratio - 1).bit_length()

    # The schedule holds every chunk's steps back to back, so this chunk starts
    # where the chunks before it end and runs `num_steps[chunk_idx]` steps.
    step_offset = sum(metadata_ref.num_steps[m] for m in range(chunk_idx))

    def seq_tile(p_id):
        """Sequence tile scheduled at pipeline step `p_id`."""
        return metadata_ref.batch_tile_idx[step_offset + p_id]

    def bq_block(p_id):
        """Query block scheduled at pipeline step `p_id`."""
        return metadata_ref.bq_idx[step_offset + p_id]

    def bkv_block(p_id):
        """KV block scheduled at pipeline step `p_id`."""
        return metadata_ref.bkv_idx[step_offset + p_id]

    prefill_cfg = _BlockedPrefillConfig.create(
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        num_q_heads=num_q_heads,
        q_dtype=q_hbm_ref.dtype,
        seq_batch_size=seq_batch_size,
    )

    q_spec = pl.BlockSpec(
        (pl.BoundedSlice(seq_batch_size * bq_sz), num_q_heads, head_dim),
        lambda p_id: (seq_tile(p_id), bq_block(p_id), 0),
        pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
    )
    weights_spec = pl.BlockSpec(
        (pl.BoundedSlice(seq_batch_size * bq_sz), num_q_heads),
        lambda p_id: (seq_tile(p_id), bq_block(p_id), 0),
        pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
    )
    if seq_along_lane:
        kv_spec = pl.BlockSpec(
            (
                pl.BoundedSlice(seq_batch_size),
                bkv_p * kv_sublane_groups,
                kv_packing,
                page_size,
            ),
            lambda p_id: (seq_tile(p_id), 0, 0, bkv_block(p_id)),
            pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
        )
    else:
        kv_spec = pl.BlockSpec(
            (
                pl.BoundedSlice(seq_batch_size),
                bkv_sz_per_kv_packing,
                kv_packing,
                cache_kv_hbm_ref.shape[-1],
            ),
            lambda p_id: (seq_tile(p_id), 0, bkv_block(p_id), 0),
            pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
        )
    out_spec = pl.BlockSpec(
        (pl.BoundedSlice(seq_batch_size * bq_sz), prefill_cfg.score_pitch, 128),
        lambda p_id: (seq_tile(p_id), bq_block(p_id), bkv_block(p_id), 0),
        pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
    )

    q_alloc = bref_override.StreamIndexQBufferedRef.input(
        spec=q_spec,
        dtype_or_type=q_hbm_ref,
        buffer_count=buffer_cnt,
        use_lookahead=True,
        bq_sz=bq_sz,
        seq_batch_size=seq_batch_size,
        chunk_tokens=scores_hbm_ref.shape[0],
        chunk_idx=chunk_idx,
    )
    weights_alloc = bref_override.StreamIndexQBufferedRef.input(
        spec=weights_spec,
        dtype_or_type=indexer_weights_hbm_ref,
        buffer_count=buffer_cnt,
        use_lookahead=True,
        bq_sz=bq_sz,
        seq_batch_size=seq_batch_size,
        chunk_tokens=scores_hbm_ref.shape[0],
        chunk_idx=chunk_idx,
    )
    if seq_along_lane:
        kv_alloc = bref_override.StreamIndexKVSeqAlongLaneBufferedRef.input(
            spec=kv_spec,
            dtype_or_type=cache_kv_hbm_ref,
            buffer_count=buffer_cnt,
            use_lookahead=True,
            bkv_p=bkv_p,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
        )
    else:
        kv_alloc = bref_override.StreamIndexKVBufferedRef.input(
            spec=kv_spec,
            dtype_or_type=cache_kv_hbm_ref,
            buffer_count=buffer_cnt,
            use_lookahead=True,
            bkv_p=bkv_p,
            page_size_per_kv_packing=page_size_per_kv_packing,
            kv_packing=kv_packing,
            pages_per_seq=pages_per_seq,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
        )
    o_alloc = bref_override.StreamIndexOBufferedRef.output(
        spec=out_spec,
        dtype_or_type=scores_hbm_ref,
        buffer_count=2,
        use_lookahead=False,
        bq_sz=bq_sz,
        num_sublanes=num_sublanes_bkv,
        seq_batch_size=seq_batch_size,
        chunk_idx=chunk_idx,
    )
    gm_rows_bkv = num_sublanes_bkv // GM_TILES
    g_alloc = None
    if group_max:
        assert num_sublanes_bkv % GM_TILES == 0, (bkv_sz, GM_TILES)
        g_spec = pl.BlockSpec(
            (pl.BoundedSlice(seq_batch_size * bq_sz), gm_rows_bkv, 128),
            lambda p_id: (seq_tile(p_id), bq_block(p_id), bkv_block(p_id), 0),
            pipeline_mode=pl.Buffered(buffer_count=2, use_lookahead=False),
        )
        g_alloc = bref_override.StreamIndexOBufferedRef.output(
            spec=g_spec,
            dtype_or_type=gm_hbm_ref,
            buffer_count=2,
            use_lookahead=False,
            bq_sz=bq_sz,
            num_sublanes=gm_rows_bkv,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
        )

    # ------------------------------------------------------------------------
    # SEQ_ALONG_LANE block access.
    #
    # The VMEM block is u8[seq_batch, bkv_p * kv_sublane_groups, kv_packing,
    # page_size]: the block's pages back to back, each exactly as it is laid out
    # in HBM, so every page arrives in one contiguous copy. In a page, packed
    # group g holds key dims [4g, 4g + 4) of every token, one byte per dim, and
    # group `kv_head_dim_groups` holds the UE8M0 scale in byte 0. Viewed as u32
    # with tp = page_size // 128 lane tiles a page, row
    # `(page * kv_sublane_groups + g) * tp + j` is group g of the page's lane
    # tile j, so a sublane-strided load with stride tp yields the
    # [head_dim // 4, 128] words of one lane tile. Bitcasting those words to FP8
    # gives [head_dim, 128] with key dim 4g + i taken from byte i -- exactly the
    # MXU-native packed layout -- so no relayout is needed on the key path.
    # (Pages used to land in lane windows of a [groups, kv_packing, bkv_sz]
    # block, which made that stride a whole block row, bkv_sz // 128 tiles. On
    # tpu7x the verify pass ran 20-23% longer that way.)
    # ------------------------------------------------------------------------
    num_lane_tiles_bkv = bkv_sz // 128
    tiles_per_page = page_size // 128

    def seq_along_lane_block_rows(bkv_vmem):
        return bkv_vmem.bitcast(jnp.uint32).reshape(
            seq_batch_size, bkv_p * kv_sublane_groups * tiles_per_page, 128
        )

    def seq_along_lane_row(group, lane_tile_idx):
        """u32 row of `group` of block lane tile `lane_tile_idx`."""
        page, tile = divmod(lane_tile_idx, tiles_per_page)
        return (page * kv_sublane_groups + group) * tiles_per_page + tile

    def seq_along_lane_key_tile(rows, batch_idx, lane_tile_idx):
        """[head_dim, 128] FP8 keys of lane tile `lane_tile_idx`."""
        words = rows[
            batch_idx,
            pl.ds(
                seq_along_lane_row(0, lane_tile_idx),
                kv_head_dim_groups,
                stride=tiles_per_page,
            ),
            :,
        ]
        return pltpu.bitcast(words, jnp.float8_e4m3fn)

    def seq_along_lane_scale_tile(rows, batch_idx, lane_tile_idx):
        """[1, 128] f32 dequantization scales of lane tile `lane_tile_idx`."""
        row = seq_along_lane_row(kv_head_dim_groups, lane_tile_idx)
        words = rows[batch_idx, pl.ds(row, 1), :]
        # UE8M0 holds a bare biased exponent, which is where f32 keeps its own:
        # moving byte 0 into the f32 exponent field yields 2^(e - 127) exactly
        # for every code the quantizer writes. The codes it never writes, 0 and
        # 0xFF, decode to 0 and +inf instead of e8m0's 2^-127 and NaN; selecting
        # those out is ~4% of SAL decode at batch 32.
        return pltpu.bitcast((words & 0xFF) << 23, jnp.float32)

    def transpose_to_head_major(bq_rows, n_tok):
        """[n_tok*H, D] head-minor rows -> [H*n_tok, D] head-major rows."""
        return bq_rows.transpose(1, 0, 2).reshape(num_q_heads * n_tok, head_dim)

    # Tokens per head in the cached head-major q block of the per-step path.
    # Whole 8-row sublane groups keep each head's [tokens, bkv] slice of q @ k
    # vreg-aligned: at the 7-token verify window a head straddles two groups,
    # and realigning every slice costs more VPU work than the reduction itself.
    # The zero pad rows are scored and dropped.
    hm_tok = align_to(bq_sz, 8)

    # Block-diagonal verify scoring (`verify_bd`, TPU_STREAMIDX_VERIFY_BD). The
    # contraction is head_dim = 128 deep, but the MXU pads it to 256 anyway, so
    # stacking two lane tiles' keys on the contraction ([keys_a; keys_b]) against
    # a block-diagonal q -- row (h, half, t) holds q[t, h] in contraction half
    # `half`, zeros in the other -- scores both tiles in the passes one took.
    # Each head then owns 2 * hm_tok = 16 rows: one packed (16, 128) bf16 tile
    # per lane tile, so the head reduction runs in packed bf16 (relu(q.k) and
    # the weights rounded to bf16, products accumulated in f32; the numerics of
    # `reduce_token_block`'s bf16 reduce), one multiply-accumulate per packed
    # register instead of a max, a multiply and an add per f32 register.
    use_verify_bd = (
        verify_bd
        and seq_along_lane
        and not prefill_cfg.enabled
        and seq_batch_size == 1
        and bq_sz > 1
        and get_dtype_bitwidth(q_hbm_ref.dtype) == 8
        and 2 * hm_tok == 8 * get_dtype_packing(jnp.bfloat16)
        and 2 * head_dim <= 256
        and (num_lane_tiles_bkv // 2) % 2 == 0
    )

    def pad_tokens(bq_rows, n_tok):
        """[n_tok, H, D] -> [hm_tok, H, D], zero rows appended."""
        if n_tok == hm_tok:
            return bq_rows
        pad = jnp.zeros((hm_tok - n_tok, *bq_rows.shape[1:]), bq_rows.dtype)
        return jnp.concatenate([bq_rows, pad], axis=0)

    def reduce_heads(bq_rows, bkv, bq_weights, head_major=False):
        """relu(q @ k) weighted-summed over heads.

        Args:
          bq_rows: [n_tok * num_q_heads, head_dim] queries, head-minor.
          bkv: the packed key block for this step.
          bq_weights: [n_tok, num_q_heads] indexer weights.

        Returns:
          [n_tok, bkv_sz] f32 weighted sum over heads.
        """
        n_tok = bq_weights.shape[0]
        w = bq_weights.astype(jnp.float32)

        def qk(rows):
            if seq_along_lane:
                # bkv is [head_dim, bkv_sz]: the MXU-native contraction layout.
                return jnp.einsum(
                    "nd,dm->nm", rows, bkv, preferred_element_type=jnp.float32
                )
            # bkv is [bkv_sz, head_dim].
            return jnp.einsum(
                "nd,md->nm", rows, bkv, preferred_element_type=jnp.float32
            )

        if n_tok == 1:
            # Decode. With a single query token there is no token axis to trade
            # against the head axis, so the plain vectorized form is cheapest.
            st = qk(bq_rows).reshape(n_tok, num_q_heads, -1)
            st = jnp.maximum(st, 0.0) * w[:, :, None]
            return st.sum(axis=1)

        # Prefill. Transpose to head-major if needed.
        if head_major:
            # The cached block carries `hm_tok` tokens per head.
            rows_hm, rows_tok = bq_rows, hm_tok
            w = jnp.pad(w, ((0, rows_tok - n_tok), (0, 0)))
        else:
            rows_hm, rows_tok = transpose_to_head_major(bq_rows, n_tok), n_tok
        st = qk(rows_hm).reshape(num_q_heads, rows_tok, -1)
        acc = jnp.maximum(st[0], 0.0) * w[:, 0:1]
        # Loop over heads 1..H-1, accumulating the sum of relu(st[h] * w[:, h]).
        for h in range(1, num_q_heads):
            relu_st = jnp.maximum(st[h], 0.0)  # relu
            acc = acc + relu_st * w[:, h : h + 1]  # fused multiply-add.
        return acc[:n_tok]

    def compute_scores(
        bq_vec,
        bkv_vec,
        scale_val_vec,
        bq_weights_vec,
        q_is_head_major=False,
        num_sublanes=num_sublanes_bkv,
    ):
        assert len(bq_vec) == seq_batch_size
        assert len(bkv_vec) == seq_batch_size
        assert len(scale_val_vec) == seq_batch_size
        assert len(bq_weights_vec) == seq_batch_size
        ret = []

        for batch_idx in range(seq_batch_size):
            bq = bq_vec[batch_idx].reshape(-1, head_dim)
            bkv = bkv_vec[batch_idx]
            scale_val = scale_val_vec[batch_idx]
            bq_weights = bq_weights_vec[batch_idx]

            s_summed = reduce_heads(bq, bkv, bq_weights, head_major=q_is_head_major)
            # Positions a token cannot see are left unmasked, as in
            # `blocked_prefill_step`: the top-k reads each row only up to its
            # `row_lengths` entry, the causal and end-of-sequence limit.
            s_summed = s_summed * scale_val
            ret.append(s_summed.reshape(-1, num_sublanes, 128))
        return jnp.concatenate(ret, axis=0)

    def build_verify_bd_caches(q_vmem, weights_vmem, q_bd_ref, wb_ref):
        """Caches the block-diagonal q and lane-broadcast bf16 weights of a q block.

        Args:
          q_vmem: [bq, H, D] query block.
          weights_vmem: [bq, H] indexer weights.
          q_bd_ref: [H * 2 * hm_tok, 2 * D] block-diagonal q, rows (head, half,
            token); the token axis is padded to hm_tok with zero rows.
          wb_ref: [H, 2 * hm_tok, 128] bf16 weights of each (head, half, token)
            row, broadcast along lanes; zero on the padding rows.
        """
        # Built in f32 (exact for FP8 values), cast back to the q dtype.
        qf = pad_tokens(q_vmem[...], bq_sz).astype(jnp.float32)
        qhm = qf.transpose(1, 0, 2)  # [H, hm_tok, D]
        z = jnp.zeros_like(qhm)
        q_bd = jnp.concatenate(
            [
                jnp.concatenate([qhm, z], axis=-1),
                jnp.concatenate([z, qhm], axis=-1),
            ],
            axis=1,
        )
        q_bd_ref[...] = q_bd.reshape(q_bd_ref.shape).astype(q_bd_ref.dtype)
        w = weights_vmem[...].astype(jnp.float32)
        w = jnp.pad(w, ((0, hm_tok - w.shape[0]), (0, 0)))  # [hm_tok, H]
        for h in range(num_q_heads):
            col = w[:, h : h + 1]
            wb_ref[h] = jnp.broadcast_to(
                jnp.concatenate([col, col], axis=0), (2 * hm_tok, 128)
            ).astype(wb_ref.dtype)

    def step_body(
        q_vmem, weights_vmem, bkv_vmem, scores_vmem, gm_vmem=None, q_hm_ref=None,
        wb_ref=None,
    ):
        bkv_idx = bkv_block(pl.program_id(0))

        if wb_ref is not None:
            # Verify-bd: the caches depend only on the q block (see below).
            @pl.when(bkv_idx == 0)
            def _():
                build_verify_bd_caches(q_vmem, weights_vmem, q_hm_ref, wb_ref)

            bq_vec = None
        elif q_hm_ref is not None:
            # The head-major reordering depends only on the q block, but the grid
            # iterates kv innermost (`bkv_idx = step % num_bkv`), so bq_idx is held
            # fixed for a whole run of steps. Reordering once per q block instead of
            # once per step avoids redoing identical work num_bkv times; `bkv_idx
            # == 0` marks exactly the first step of each new q block.
            @pl.when(bkv_idx == 0)
            def _():
                q_hm_ref[...] = transpose_to_head_major(
                    pad_tokens(q_vmem[...], bq_sz), hm_tok
                )

            bq_vec = [q_hm_ref[...]]
        else:
            q_data = q_vmem[...].reshape(seq_batch_size, -1, head_dim)
            bq_vec = [q_data[batch_idx] for batch_idx in range(seq_batch_size)]
        w_data = weights_vmem[...].reshape(seq_batch_size, -1, num_q_heads)
        bq_weights_vec = [w_data[batch_idx] for batch_idx in range(seq_batch_size)]

        def score_verify_bd(num_tiles):
            """Verify-bd scoring of the block's first `num_tiles` lane tiles."""
            rows = seq_along_lane_block_rows(bkv_vmem)
            half_n = num_tiles // 2
            # Two lane tiles per contraction half per matmul (N = 256 lanes), the
            # heads reduced inside each chunk and the weight tile reloaded from
            # VMEM per chunk, as in `blocked_prefill_step`: in that shape the
            # compiler fuses each packed bf16 term into one multiply-accumulate.
            # One matmul over the whole block, with a single weight register
            # reused across every lane tile, does not fuse: the relu is packed,
            # unpacked and multiplied and added in f32, 2.5x the VALU work.
            chunk = math.gcd(half_n, 2)
            halves = ([], [])
            for c in range(half_n // chunk):
                tiles = (
                    range(c * chunk, (c + 1) * chunk),
                    range(half_n + c * chunk, half_n + (c + 1) * chunk),
                )
                # Contraction half 0 scores lane tiles [0, n/2), half 1 [n/2, n).
                rhs = jnp.concatenate(
                    [
                        jnp.concatenate(
                            [seq_along_lane_key_tile(rows, 0, j) for j in t], axis=1
                        )
                        for t in tiles
                    ],
                    axis=0,
                )
                st = jnp.einsum(
                    "nd,dm->nm",
                    q_hm_ref[...],
                    rhs,
                    preferred_element_type=jnp.float32,
                ).reshape(num_q_heads, 2 * hm_tok, chunk * 128)
                acc = None
                for h in range(num_q_heads):
                    w_h = wb_ref[h]
                    if chunk > 1:
                        w_h = jnp.concatenate([w_h] * chunk, axis=1)
                    # As in `reduce_token_block`: rounding before the relu equals
                    # rounding after it, and the chain starts from a product and
                    # adds one unpacked bf16 x bf16 product per head.
                    r = jnp.maximum(st[h].astype(jnp.bfloat16), 0)
                    term = r.astype(jnp.float32) * w_h.astype(jnp.float32)
                    acc = term if acc is None else acc + term
                assert acc is not None
                # Rows [0, bq) of the head slice score half 0's tiles, rows
                # [hm_tok, hm_tok + bq) half 1's.
                for half, t in enumerate(tiles):
                    scale = jnp.concatenate(
                        [seq_along_lane_scale_tile(rows, 0, j) for j in t], axis=1
                    )
                    r0 = half * hm_tok
                    halves[half].append(acc[r0 : r0 + bq_sz] * scale)
            s = jnp.concatenate(halves[0] + halves[1], axis=1)
            scores_vmem[:, :num_tiles] = pltpu.bitcast(
                s.reshape(bq_sz, num_tiles, 128), jnp.int32
            )

        def score(num_tiles):
            """Scores the block's first `num_tiles` lane tiles."""
            if wb_ref is not None:
                score_verify_bd(num_tiles)
                return
            bkvs = []
            bkv_scales = []
            for batch_idx in range(seq_batch_size):
                if seq_along_lane:
                    # Keys and scales come straight out of the packed block in their
                    # MXU and lane layouts respectively; see
                    # `seq_along_lane_block_rows`.
                    rows = seq_along_lane_block_rows(bkv_vmem)
                    lane_tiles = range(num_tiles)
                    fp8_val = jnp.concatenate(
                        [
                            seq_along_lane_key_tile(rows, batch_idx, lane_tile_idx)
                            for lane_tile_idx in lane_tiles
                        ],
                        axis=1,
                    )
                    scale_val = jnp.concatenate(
                        [
                            seq_along_lane_scale_tile(rows, batch_idx, lane_tile_idx)
                            for lane_tile_idx in lane_tiles
                        ],
                        axis=1,
                    )
                    bkvs.append(fp8_val)
                    bkv_scales.append(scale_val)
                else:
                    bkv = bkv_vmem[batch_idx, :bkv_sz_per_kv_packing][...]
                    flat_bkv = bkv.reshape(-1, bkv.shape[-1])
                    fp8_val = pltpu.bitcast(flat_bkv[:, :head_dim], jnp.float8_e4m3fn)
                    scale_val = pltpu.bitcast(
                        flat_bkv[:, head_dim : head_dim + 1].T, jnp.float8_e8m0fnu
                    ).astype(jnp.float32)
                    bkvs.append(fp8_val.reshape(bkv_sz, head_dim))
                    bkv_scales.append(scale_val)

            scores = compute_scores(
                bq_vec,
                bkvs,
                bkv_scales,
                bq_weights_vec,
                q_is_head_major=q_hm_ref is not None,
                num_sublanes=num_tiles,
            )
            scores_vmem[:, :num_tiles] = pltpu.bitcast(scores, jnp.int32)

        # A sequence's last KV block is usually part empty, and positions past
        # its compressed length are never read (the top-k stops at
        # `row_lengths`). When they fill at least half the block, only the
        # first half is scored. The scored positions' arithmetic is unchanged.
        half_tiles = num_lane_tiles_bkv // 2
        if (
            seq_along_lane
            and seq_batch_size == 1
            and cp_size == 1
            and half_tiles % 8 == 0
        ):
            kv_len = seq_lens_ref[seq_tile(pl.program_id(0))] >> shift_comp
            past_half = kv_len - bkv_idx * bkv_sz > half_tiles * 128
            pl.when(past_half)(functools.partial(score, num_lane_tiles_bkv))
            pl.when(jnp.logical_not(past_half))(functools.partial(score, half_tiles))
        else:
            score(num_lane_tiles_bkv)

    def blocked_prefill_step(
        q_vmem, weights_vmem, bkv_vmem, scores_vmem, gm_vmem=None, q_hm_ref=None,
        wb_ref=None, q_half_ref=None,
    ):
        """Scores one (q block, kv block) step of the multi-token path.

        Args:
          q_vmem: [bq, H, D] query block.
          weights_vmem: [bq, H] indexer weights.
          bkv_vmem: the packed key block of this step.
          scores_vmem: [bq, score_pitch, 128] i32 output block, of which the
            first nsub rows of each token are copied out.
          q_hm_ref: [bq * H, D] cache of the q block in (head group, token block,
            head, token) row order.
          wb_ref: [num_tok_blks, H, tok_blk, 128] cache of the indexer weights,
            broadcast along lanes; bf16 for FP8 queries, f32 otherwise.
          q_half_ref: [H * tok_blk // 2, D] cache of the block's first half of
            tokens in (head group, head, token) row order, or None.
        """
        bkv_idx = bkv_block(pl.program_id(0))
        tok_blk = prefill_cfg.tok_blk
        num_tok_blks = prefill_cfg.num_tok_blks
        num_head_groups = prefill_cfg.num_head_groups
        heads_per_group = prefill_cfg.heads_per_group
        rows_per_group_tok_blk = prefill_cfg.rows_per_group_tok_blk
        rows_per_group = prefill_cfg.rows_per_group
        lane_chunk = prefill_cfg.lane_chunk
        num_lane_chunks = prefill_cfg.num_lane_chunks
        tiles_per_chunk = prefill_cfg.tiles_per_chunk
        score_pitch = prefill_cfg.score_pitch

        # The caches depend only on the q block, but the grid iterates kv
        # innermost (`bkv_idx = step % num_bkv`), so bq_idx is held fixed for a
        # whole run of steps. Rebuilding them once per q block instead of once per
        # step avoids redoing identical work num_bkv times; `bkv_idx == 0` marks
        # exactly the first step of each new q block.
        @pl.when(bkv_idx == 0)
        def _():
            w = weights_vmem[...].astype(jnp.float32)
            for tb in range(num_tok_blks):
                t0 = tb * tok_blk
                q_rows = transpose_to_head_major(q_vmem[pl.ds(t0, tok_blk)], tok_blk)
                for h in range(num_q_heads):
                    q_hm_ref[pl.ds(prefill_cfg.cache_row(h, tb), tok_blk)] = q_rows[
                        h * tok_blk : (h + 1) * tok_blk
                    ]
                    if q_half_ref is not None:
                        g, hg = h % num_head_groups, h // num_head_groups
                        q_half_ref[
                            pl.ds(
                                (g * heads_per_group + hg) * (tok_blk // 2),
                                tok_blk // 2,
                            )
                        ] = q_rows[h * tok_blk : h * tok_blk + tok_blk // 2]
                    wb_ref[tb, h] = jnp.broadcast_to(
                        w[t0 : t0 + tok_blk, h : h + 1], (tok_blk, 128)
                    ).astype(wb_ref.dtype)

        # Positions a token cannot see are left unmasked. The SparseCore top-k
        # reads each score row only up to its `row_lengths` entry, which is
        # exactly the causal and end-of-sequence limit (`_effective_row_lengths`);
        # that same bound already keeps it off the KV blocks the schedule skips,
        # so masking here would only cost a pass over every diagonal block.

        if seq_along_lane:
            rows = seq_along_lane_block_rows(bkv_vmem)

            def load_chunk(c):
                tiles = range(c * tiles_per_chunk, (c + 1) * tiles_per_chunk)
                keys = jnp.concatenate(
                    [
                        seq_along_lane_key_tile(rows, 0, lane_tile_idx)
                        for lane_tile_idx in tiles
                    ],
                    1,
                )
                scales = jnp.concatenate(
                    [
                        seq_along_lane_scale_tile(rows, 0, lane_tile_idx)
                        for lane_tile_idx in tiles
                    ],
                    1,
                )
                return keys, scales

            def qk(q_rows, keys):
                # keys is [head_dim, lane_chunk]: the MXU-native contraction layout.
                return jnp.einsum(
                    "nd,dm->nm", q_rows, keys, preferred_element_type=jnp.float32
                )

        else:
            groups_per_chunk = lane_chunk // kv_packing

            def load_chunk(c):
                blk = bkv_vmem[0, pl.ds(c * groups_per_chunk, groups_per_chunk)]
                flat = blk.reshape(lane_chunk, blk.shape[-1])
                keys = pltpu.bitcast(flat[:, :head_dim], jnp.float8_e4m3fn)
                scales = pltpu.bitcast(
                    flat[:, head_dim : head_dim + 1].T, jnp.float8_e8m0fnu
                ).astype(jnp.float32)
                return keys, scales

            def qk(q_rows, keys):
                # keys is [lane_chunk, head_dim].
                return jnp.einsum(
                    "nd,md->nm", q_rows, keys, preferred_element_type=jnp.float32
                )

        def reduce_token_block(sts, tb, rows):
            """sum_h relu(st_h) * w_h over the first `rows` rows of token block `tb`.

            Args:
              sts: per head group, the [rows_per_group, lane_chunk] q @ k block.
              tb: token block index.
              rows: leading rows (tokens) of the block to reduce.
            """
            acc = None
            for i in range(heads_per_group):
                # Consecutive heads come from different groups, and so from different
                # MXUs, so the accumulation drains every MXU at the rate it produces
                # results.
                for g, st in enumerate(sts):
                    # The half-block cache holds `rows` rows per head.
                    r0 = (
                        i * rows
                        if rows < tok_blk
                        else tb * rows_per_group_tok_blk + i * tok_blk
                    )
                    st_h = st[r0 : r0 + rows]
                    w_h = wb_ref[tb, i * num_head_groups + g][:rows]
                    if tiles_per_chunk > 1:
                        w_h = jnp.concatenate([w_h] * tiles_per_chunk, axis=1)
                    if prefill_cfg.bf16_reduce:
                        # Rounding before the relu is the same as rounding after
                        # it. The chain has to start from a product and add one
                        # unpacked bf16 x bf16 product per head for the compiler
                        # to fuse each packed pair of links into a single
                        # multiply-accumulate.
                        r = jnp.maximum(st_h.astype(jnp.bfloat16), 0)
                        term = r.astype(jnp.float32) * w_h.astype(jnp.float32)
                    else:
                        term = jnp.maximum(st_h, 0.0) * w_h
                    acc = term if acc is None else acc + term
            assert acc is not None
            return acc

        # Row `t * score_pitch + j` holds lane tile j of token t.
        scores_rows = (
            scores_vmem.reshape(bq_sz * score_pitch, 128)
            if prefill_cfg.strided_score_store
            else None
        )

        def score_block(rows, with_gm=True):
            """Scores every token block, reducing only its first `rows` rows."""
            chunk_scores = []
            gm_packed = [None] * num_tok_blks
            if gm_vmem is not None:
                assert tiles_per_chunk * num_lane_chunks % GM_TILES == 0
                gm_lane = jax.lax.broadcasted_iota(jnp.int32, (rows, 128), 1) & (
                    GM_TILES - 1
                )
            for c in range(num_lane_chunks):
                keys, scales = load_chunk(c)
                # Reload the cached rows for every chunk instead of keeping the whole
                # q block live in registers across chunks. Each head group is its
                # own matmul so that the groups can run on different MXUs.
                if rows < tok_blk:
                    group_rows = heads_per_group * rows
                    sts = [
                        qk(q_half_ref[pl.ds(g * group_rows, group_rows)], keys)
                        for g in range(num_head_groups)
                    ]
                else:
                    sts = [
                        qk(q_hm_ref[pl.ds(g * rows_per_group, rows_per_group)], keys)
                        for g in range(num_head_groups)
                    ]
                tb_scores = []
                for tb in range(num_tok_blks):
                    acc = reduce_token_block(sts, tb, rows) * scales
                    if gm_vmem is not None and with_gm:
                        for jj in range(tiles_per_chunk):
                            tile = c * tiles_per_chunk + jj
                            if _GM_FAULT == "gm_zero":  # timing-only removal arm
                                if tile % GM_TILES == GM_TILES - 1:
                                    gm_vmem[pl.ds(tb * tok_blk, rows), tile // GM_TILES] = (
                                        jnp.zeros((rows, 128), jnp.int32))
                                continue
                            if GM_F32:
                                # Max in f32 (the order of the keys for every non-NaN
                                # score; -0 is canonicalized to +0 at the end so the
                                # group key is never below a +0 member's key).
                                key = acc[:, jj * 128 : (jj + 1) * 128]
                            else:
                                key = _monotone_key(
                                    pltpu.bitcast(acc[:, jj * 128 : (jj + 1) * 128], jnp.int32)
                                )
                            if _GM_FAULT == "gm_flat":
                                # v1 form: 4 lane rolls per tile + a roll/select pack.
                                i = tile % GM_TILES
                                for sh in (1, 2, 4, 8):
                                    key = jnp.maximum(key, pltpu.roll(key, 128 - sh, 1))
                                moved = pltpu.roll(key, i, 1) if i else key
                                gm_packed[tb] = (
                                    moved if gm_packed[tb] is None
                                    else jnp.where(gm_lane == i, moved, gm_packed[tb])
                                )
                                if i == GM_TILES - 1:
                                    gm_vmem[pl.ds(tb * tok_blk, rows), tile // GM_TILES] = gm_packed[tb]
                                    gm_packed[tb] = None
                                continue
                            # Tree: combine tile groups pairwise; a level-l value holds,
                            # at lane 2^l m + s, tile s's max over lanes [2^l m, 2^l m + 2^l).
                            stack = gm_packed[tb] or []
                            stack.append((0, key))
                            while len(stack) >= 2 and stack[-1][0] == stack[-2][0]:
                                (lv, y), (_, x) = stack.pop(), stack.pop()
                                w = 1 << lv
                                x = jnp.maximum(x, pltpu.roll(x, 128 - w, 1))
                                if not (_GM_FAULT == "gm_bad" and lv == 0):
                                    y = jnp.maximum(y, pltpu.roll(y, 128 - w, 1))
                                stack.append(
                                    (lv + 1, jnp.where((gm_lane & (2 * w - 1)) < w, x,
                                                       pltpu.roll(y, w, 1)))
                                )
                            if stack[-1][0] == GM_LEVELS:
                                top = stack.pop()[1]
                                if GM_F32:
                                    top = _monotone_key(
                                        pltpu.bitcast(jnp.where(top == 0, 0.0, top), jnp.int32)
                                    )
                                gm_vmem[pl.ds(tb * tok_blk, rows), tile // GM_TILES] = top
                            gm_packed[tb] = stack
                    if scores_rows is None:
                        tb_scores.append(acc)
                        continue
                    for jj in range(tiles_per_chunk):
                        lane_tile = c * tiles_per_chunk + jj
                        scores_rows[
                            pl.ds(
                                tb * tok_blk * score_pitch + lane_tile,
                                rows,
                                stride=score_pitch,
                            )
                        ] = pltpu.bitcast(acc[:, jj * 128 : (jj + 1) * 128], jnp.int32)
                if scores_rows is None:
                    chunk_scores.append(jnp.concatenate(tb_scores, axis=0))

            if scores_rows is None:
                s_summed = jnp.concatenate(chunk_scores, axis=1)
                scores_vmem[...] = pltpu.bitcast(
                    s_summed.reshape(bq_sz, num_sublanes_bkv, 128), jnp.int32
                )

        # A q block's rows past its sequence (or chunk) are padding: they are
        # never copied out. When at most half the block is real, as in the last
        # block of a 100-token chunk, only the first packed 16-row half is
        # reduced. The real rows' arithmetic is unchanged.
        half = tok_blk // 2
        if q_half_ref is not None:
            assert scores_rows is not None
            _, valid_rows = bref_override.token_window(
                (None, cu_q_lens_ref, metadata_ref, start_end_seq_idx_ref),
                pl.program_id(0),
                scores_hbm_ref.shape[0],
                bq_sz=bq_sz,
                seq_batch_size=seq_batch_size,
                chunk_idx=chunk_idx,
            )
            if gm_vmem is not None and gm_min_kv > 0:
                gm_on = (seq_lens_ref[seq_tile(pl.program_id(0))] >> shift_comp) >= gm_min_kv
                for vr, vh in ((valid_rows <= half, half), (valid_rows > half, tok_blk)):
                    pl.when(vr & gm_on)(functools.partial(score_block, vh, True))
                    pl.when(vr & jnp.logical_not(gm_on))(
                        functools.partial(score_block, vh, False))
            else:
                pl.when(valid_rows <= half)(functools.partial(score_block, half))
                pl.when(valid_rows > half)(functools.partial(score_block, tok_blk))
        else:
            score_block(tok_blk)

    def _run_all(q_hm_ref=None, wb_ref=None, q_half_ref=None):
        if prefill_cfg.enabled:
            body = functools.partial(
                blocked_prefill_step,
                q_hm_ref=q_hm_ref,
                wb_ref=wb_ref,
                q_half_ref=q_half_ref,
            )
        else:
            body = functools.partial(step_body, q_hm_ref=q_hm_ref, wb_ref=wb_ref)

        allocs = (q_alloc, weights_alloc, kv_alloc, o_alloc) + (
            (g_alloc,) if group_max else ()
        )

        @pl.with_scoped(final_allocs=allocs)
        def _run_pipeline(final_allocs):
            q_buf, weights_buf, kv_buf, o_buf = final_allocs[:4]
            window_refs = (cu_q_lens_ref, metadata_ref, start_end_seq_idx_ref)
            if group_max:
                g_buf = final_allocs[4]
                pipeline_fn = pltpu.emit_pipeline(
                    body,
                    grid=(metadata_ref.num_steps[chunk_idx],),
                    in_specs=(q_buf.spec, weights_buf.spec, kv_buf.spec),
                    out_specs=(o_buf.spec, g_buf.spec),
                )
                pipeline_fn(
                    (q_hbm_ref, *window_refs),
                    (indexer_weights_hbm_ref, *window_refs),
                    (cache_kv_hbm_ref, page_indices_ref, metadata_ref),
                    (scores_hbm_ref, *window_refs),
                    (gm_hbm_ref, *window_refs),
                    allocations=(q_buf, weights_buf, kv_buf, o_buf, g_buf),
                )
                return
            pipeline_fn = pltpu.emit_pipeline(
                body,
                grid=(metadata_ref.num_steps[chunk_idx],),
                in_specs=(q_buf.spec, weights_buf.spec, kv_buf.spec),
                out_specs=o_buf.spec,
            )
            pipeline_fn(
                (q_hbm_ref, *window_refs),
                (indexer_weights_hbm_ref, *window_refs),
                (cache_kv_hbm_ref, page_indices_ref, metadata_ref),
                (scores_hbm_ref, *window_refs),
                allocations=(q_buf, weights_buf, kv_buf, o_buf),
            )

        _run_pipeline()

    # Only the multi-token path reorders q, and only a single sequence per tile
    # keeps the cached block unambiguous. A single 32-token block also keeps a
    # cache of its first 16 tokens, for blocks with at most 16 real rows.
    half_block_cache = (
        prefill_cfg.enabled
        and prefill_cfg.num_tok_blks == 1
        and prefill_cfg.bf16_reduce
        and prefill_cfg.strided_score_store
        and prefill_cfg.tok_blk == 32
    )
    if prefill_cfg.enabled:
        pl.run_scoped(
            _run_all,
            pltpu.VMEM((num_q_heads * bq_sz, head_dim), q_hbm_ref.dtype),
            pltpu.VMEM(
                (
                    prefill_cfg.num_tok_blks,
                    num_q_heads,
                    prefill_cfg.tok_blk,
                    128,
                ),
                prefill_cfg.wb_dtype,
            ),
            *(
                (pltpu.VMEM((num_q_heads * bq_sz // 2, head_dim), q_hbm_ref.dtype),)
                if half_block_cache
                else ()
            ),
        )
    elif use_verify_bd:
        pl.run_scoped(
            lambda q_bd_ref, wb_ref: _run_all(q_hm_ref=q_bd_ref, wb_ref=wb_ref),
            pltpu.VMEM((num_q_heads * 2 * hm_tok, 2 * head_dim), q_hbm_ref.dtype),
            pltpu.VMEM((num_q_heads, 2 * hm_tok, 128), jnp.bfloat16),
        )
    elif seq_batch_size == 1 and bq_sz > 1:
        pl.run_scoped(
            _run_all,
            pltpu.VMEM((num_q_heads * hm_tok, head_dim), q_hbm_ref.dtype),
        )
    else:
        _run_all()

    ### ------- Kernel end ------- ###


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    _, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    return q


def prepare_index_weights(
    index_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads],
    q_dtype,
):
    _, actual_num_q_heads = index_weights.shape
    index_weights = index_weights.astype(jnp.float32)
    num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q_dtype))
    index_weights = jnp.pad(
        index_weights,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
        ),
        constant_values=0,
    )
    return index_weights


def prepare_outputs(out):
    if out.ndim == 3:
        out = out.reshape(out.shape[0], -1)
    return out


# --------------------------------------------------------------------------
# KV cache layout helpers
# --------------------------------------------------------------------------
#
# The index cache stores, per compressed KV token, `head_dim` FP8 (e4m3) bytes
# followed by a single UE8M0 (e8m0) scale byte.
#
# HEAD_ALONG_SUBLANE puts the token on the sublane dimension, so the record
# must be padded out to a multiple of the 128-lane vector register width:
# `align_to(head_dim + 1, 128)`. For head_dim=128 that wastes 127 of every 256
# bytes of HBM traffic.
#
# SEQ_ALONG_LANE puts the token on the *lane* dimension, so the record only has
# to be padded out to a multiple of the sublane packing (4 for 8-bit data):
# `head_dim + kv_packing`. For head_dim=128 that is 132 bytes, a 1.94x
# reduction in HBM bytes per token.

# 8-bit KV data packs 4 values into every 32-bit sublane word.
KV_PACKING = 4

DEFAULT_INDEXER_KV_CACHE_MODE = "sal"
_INDEXER_KV_LAYOUT_FALLBACK_LOGGED: set[tuple[str, int | str]] = set()


def indexer_kv_layout(
    block_size: int,
    mode: str | None = None,
) -> KVLayout:
    """Returns the single-source `KVLayout` for an indexer K cache of `block_size`.

    `sal` requires `block_size` to be a positive multiple of 128 and is not
    supported by Raiden GLM Stage-3 admission (`build_glm_mla_pool_manifest` /
    `lower_glm_row_spans` require packed-row `HEAD_ALONG_SUBLANE` geometry).
    When `sal` is selected by default and Raiden GLM admission is enabled, or
    when `block_size < 128` or `block_size % 128 != 0`, this helper selects
    `KVLayout.HEAD_ALONG_SUBLANE` (logging a one-time warning) so cache
    allocation and op execution always agree on the layout. Explicitly
    requesting `sal` while Raiden GLM admission is enabled raises `ValueError`.
    """
    try:
        from vllm_torchtpu import envs

        raiden_glm_admission = envs.use_raiden_glm_admission()
        explicit_env = envs.is_indexer_kv_cache_mode_explicit()
        env_mode = envs.VLLM_TPU_INDEXER_KV_CACHE_MODE
    except ImportError:
        raiden_glm_admission = False
        explicit_env = False
        env_mode = DEFAULT_INDEXER_KV_CACHE_MODE

    if mode is None:
        raw_mode = env_mode
        explicit_mode = explicit_env
    else:
        raw_mode = mode
        explicit_mode = True
    resolved = raw_mode.strip().lower()
    if resolved not in ("xla", "has", "sal"):
        raise ValueError(
            f"Invalid VLLM_TPU_INDEXER_KV_CACHE_MODE={raw_mode!r}; "
            "expected one of ('xla', 'has', 'sal')."
        )
    if resolved == "sal":
        if raiden_glm_admission:
            if explicit_mode:
                raise ValueError(
                    "VLLM_TPU_INDEXER_KV_CACHE_MODE='sal' is incompatible with "
                    "Raiden GLM admission (TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 and "
                    "TPU_RAIDEN_GLM_ADMISSION=1), which requires packed-row "
                    "HEAD_ALONG_SUBLANE layout for dsa.idx. Set "
                    "VLLM_TPU_INDEXER_KV_CACHE_MODE=has (or unset it to allow "
                    "automatic fallback)."
                )
            key: tuple[str, int | str] = (resolved, "raiden_glm_admission")
            if key not in _INDEXER_KV_LAYOUT_FALLBACK_LOGGED:
                _INDEXER_KV_LAYOUT_FALLBACK_LOGGED.add(key)
                logger.warning(
                    "Default VLLM_TPU_INDEXER_KV_CACHE_MODE='sal' is incompatible "
                    "with Raiden GLM admission (which requires packed-row "
                    "HEAD_ALONG_SUBLANE layout for dsa.idx); falling back to "
                    "HEAD_ALONG_SUBLANE layout with 'has' writer. Set "
                    "VLLM_TPU_INDEXER_KV_CACHE_MODE=has to silence this warning.",
                )
            return KVLayout.HEAD_ALONG_SUBLANE
        if block_size < 128 or block_size % 128 != 0:
            key = (resolved, int(block_size))
            if key not in _INDEXER_KV_LAYOUT_FALLBACK_LOGGED:
                _INDEXER_KV_LAYOUT_FALLBACK_LOGGED.add(key)
                logger.warning(
                    "VLLM_TPU_INDEXER_KV_CACHE_MODE=%r requires block_size to be a "
                    "positive multiple of 128 (got block_size=%d); falling back to "
                    "HEAD_ALONG_SUBLANE layout with 'has' writer.",
                    resolved,
                    block_size,
                )
            return KVLayout.HEAD_ALONG_SUBLANE
        return KVLayout.SEQ_ALONG_LANE
    return KVLayout.HEAD_ALONG_SUBLANE


def get_kv_cache_shape(
    total_num_pages: int,
    page_size: int,
    actual_head_dim: int,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
) -> tuple[int, ...]:
    """Returns the packed uint8 index KV cache shape for a given layout."""
    kv_layout = KVLayout.parse(kv_layout)
    if actual_head_dim % 64 != 0:
        unpacked = actual_head_dim - max(1, actual_head_dim // 128)
        if unpacked > 0 and unpacked % 64 == 0:
            actual_head_dim = unpacked
    head_dim = align_to(actual_head_dim, 128)
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if page_size % 128 != 0:
            raise ValueError(
                "SEQ_ALONG_LANE requires page_size to be a multiple of the 128-lane"
                f" register width, got {page_size=}."
            )
        return (
            total_num_pages,
            head_dim // KV_PACKING + 1,
            KV_PACKING,
            page_size,
        )
    if page_size % KV_PACKING != 0:
        raise ValueError(
            "HEAD_ALONG_SUBLANE requires page_size to be a multiple of"
            f" {KV_PACKING}, got {page_size=}."
        )
    # `head_dim // 128` scale bytes are reserved to mirror the compressor's
    # record format, then the record is padded to the lane width.
    record_width = head_dim + head_dim // 128
    return (
        total_num_pages,
        page_size // KV_PACKING,
        KV_PACKING,
        align_to(record_width, 128),
    )


def convert_cache_to_seq_along_lane(
    cache_kv: jax.Array,  # uint8[pages, page_size // 4, 4, width]
    actual_head_dim: int,
) -> jax.Array:  # uint8[pages, head_dim // 4 + 1, 4, page_size]
    """Repacks a HEAD_ALONG_SUBLANE index cache into the SEQ_ALONG_LANE layout.

    This is a host-side (XLA) conversion intended for tests and for one-off
    migration of an existing cache; production callers should write the cache in
    the target layout directly.
    """
    total_num_pages, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    head_dim = align_to(actual_head_dim, 128)

    # [pages, page_size, width] -> keys and the single scale byte per token.
    flat = cache_kv.reshape(total_num_pages, page_size, -1)
    keys = flat[:, :, :head_dim]  # [pages, page_size, head_dim]
    scales = flat[:, :, head_dim : head_dim + 1]  # [pages, page_size, 1]
    # Pad the scale record out to a full packed sublane group.
    scales = jnp.pad(scales, ((0, 0), (0, 0), (0, KV_PACKING - 1)), constant_values=0)
    rows = jnp.concatenate([keys, scales], axis=-1)
    # [pages, page_size, head_dim + 4] -> [pages, head_dim + 4, page_size]
    rows = rows.swapaxes(1, 2)
    return rows.reshape(
        total_num_pages,
        (head_dim + KV_PACKING) // KV_PACKING,
        KV_PACKING,
        page_size,
    )


def _token_seq_ids(
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    num_tokens: int,
    max_num_seqs: int,
) -> jax.Array:  # i32[num_tokens]
    """Sequence of each token row (rows past the batch get `distribution[2]`)."""
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_mask = token_ids[:, None] >= cu_q_lens[None, 1 : max_num_seqs + 1]
    seq_mask = jnp.where(
        jnp.arange(max_num_seqs)[None, :] < distribution[2], seq_mask, False
    )
    return jnp.sum(seq_mask, axis=1)


# Rows per score-returning SparseCore top-k call of a kv-split share.
_KV_SPLIT_SC_MAX_ROWS = 1024


def kv_split_window(
    seq_lens: jax.Array,  # i32[max_num_seqs]
    rank: jax.Array | int,
    *,
    kv_split: int,
    compression_ratio: int,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """(first row, row count) of rank `rank`'s share of each sequence's
    compressed positions: contiguous whole pages, the same count per rank.
    (Ported from patemotter fdcc70a0d.)"""
    kv_len = seq_lens // compression_ratio
    pages = cdiv(kv_len, page_size)
    per_rank = cdiv(pages, kv_split)
    start = jnp.minimum(rank * per_rank, pages) * page_size
    return start, jnp.clip(kv_len - start, 0, per_rank * page_size)


def merge_kv_split_topk(
    shares: jax.Array,  # i32[kv_split, T, 2 * k]: (indices, score bits) per rank
    k: int,
) -> jax.Array:  # i32[T, k]
    """The unsplit top-k from every rank's kv_split share top-k.

    A global winner is beaten by fewer than k positions, so by fewer than k
    on its own rank, and survives there. Ranks own increasing column ranges,
    so the rank-major candidate list is in column order and the top-k over it
    breaks ties toward the lowest column, as the unsplit top-k does.
    (After patemotter fdcc70a0d / 7898b6033.)
    """
    num_ranks, num_tokens, _ = shares.shape
    cand = shares.reshape(num_ranks, num_tokens, 2, k).transpose(2, 1, 0, 3)
    cand = cand.reshape(2, num_tokens, num_ranks * k)
    cand_idx, cand_bits = cand[0], cand[1]
    valid = cand_idx >= 0
    num_valid = jnp.sum(valid, axis=1, dtype=jnp.int32)
    # Rows with at most k candidates keep them all (below), and are skipped by
    # the SparseCore top-k (row length 0). Selecting k of a row padded with -inf ties is its slow path (1.3 ms
    # for 256 rows at 2k context vs 0.14 ms at 100k, tpu7x one core).
    long_row = num_valid > k
    slots = sparsecore_topk(
        cand_bits,
        k,
        row_lengths=jnp.where(long_row, num_ranks * k, 0).astype(jnp.int32),
        write_empty_rows=True,
    )
    # Empty slots gather distinct columns: an all-equal index (every slot -1
    # -> column 0) made this gather ~10x slower on tpu7x (1.3 vs 0.1 ms).
    lane = jnp.arange(k, dtype=jnp.int32)[None, :]
    picked = jnp.take_along_axis(cand_idx, jnp.where(slots >= 0, slots, lane), axis=1)
    from_sc = jnp.where(slots >= 0, picked, -1)
    # A row with at most k visible positions selects all of them: columns
    # 0 .. num_valid-1 in order, as the unsplit top-k (and its early-exit fast
    # path) returns. No gather needed.
    j = jnp.arange(k, dtype=jnp.int32)[None, :]
    compact = jnp.where(j < num_valid[:, None], j, -1)
    return jnp.where(long_row[:, None], from_sc, compact)


def token_window_metadata(
    seq_lens: jax.Array,  # i32[S]
    page_indices: jax.Array,  # i32[S * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[S + 1]
    distribution: jax.Array,  # i32[3]
    offset: jax.Array,  # i32 scalar: first row of the window
    *,
    window: int,
    compression_ratio: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """(seq_lens, page_indices, cu_q_lens, distribution) of the token rows
    `[offset, offset + window)` of a batch, as a batch of its own.

    Sequences are renumbered from the first one with a row in the window (its
    page-table row moves with it). A sequence cut by the window keeps its rows'
    positions: its length drops by the rows past the window. The length the
    scoring passes see is then rounded up to a whole compressed block (capped
    at the real length) so the block a cut row may select -- the whole batch's
    row lengths allow a row its own open block when the sequence continues --
    is scored; the passes leave positions unmasked and the row lengths (from
    the whole batch, `row_window`) cut every row exactly.

    The caller guarantees no cut inside the verify/decode segments (window >=
    their rows, windows aligned to multiples of `window`).
    """
    S = seq_lens.shape[0]
    pps = page_indices.shape[0] // S
    ids = jnp.arange(S, dtype=jnp.int32)
    live = ids < distribution[2]
    s0 = jnp.sum(live & (cu_q_lens[1:] <= offset)).astype(jnp.int32)
    src = jnp.minimum(s0 + ids, S - 1)
    # Sequences with a row in the window: from s0 up to the last one that
    # starts before the window's end (no zero-row sequence is handed over).
    s1 = jnp.sum(live & (cu_q_lens[:-1] < offset + window)).astype(jnp.int32)
    n_live = jnp.maximum(s1 - s0, 0)
    j_live = ids < n_live
    cu_full = cu_q_lens[jnp.minimum(s0 + jnp.arange(S + 1, dtype=jnp.int32), S)]
    cu_w = jnp.clip(cu_full - offset, 0, window)
    cu_w = jnp.where(
        jnp.arange(S + 1) <= n_live, cu_w, cu_w[jnp.minimum(n_live, S)]
    ).astype(jnp.int32)
    end = cu_q_lens[src + 1]
    after = jnp.maximum(end - (offset + window), 0)
    exact = seq_lens[src] - after
    c = compression_ratio
    padded = jnp.minimum(seq_lens[src], (exact + c - 1) // c * c)
    seq_w = jnp.where(j_live, padded, 0).astype(seq_lens.dtype)
    pages_w = page_indices.reshape(S, pps)[src].reshape(-1)
    dist_w = jnp.stack(
        [
            jnp.clip(distribution[0] - s0, 0, n_live),
            jnp.clip(distribution[1] - s0, 0, n_live),
            n_live,
        ]
    ).astype(distribution.dtype)
    return seq_w, pages_w, cu_w, dist_w


def _effective_row_lengths(
    seq_lens: jax.Array,  # i32[max_num_seqs]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    num_tokens: int,
    num_positions: int,
    compression_ratio: int,
    cp_rank: jax.Array | int = 0,
    cp_size: int = 1,
    interleave_c: int = 1,
    kv_window: tuple[jax.Array, jax.Array] | None = None,
    strict_causal: bool = False,
) -> jax.Array:  # i32[num_tokens]
    """Visible compressed KV positions per token row of the score matrix.

    With `strict_causal`, a token sees the (token_pos + 1) // compression_ratio
    complete groups at or before it. Without it, it also sees its own group
    whenever that group completes later in the same forward, i.e. the
    compressed entry of up to compression_ratio - 1 tokens after it.

    Under context parallelism the score matrix is rank-local, so the row length
    is the number of globally-visible positions that *this* rank owns.
    """
    max_num_seqs = seq_lens.shape[0]
    num_seqs = distribution[2]
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_mask = token_ids[:, None] >= cu_q_lens[None, 1 : max_num_seqs + 1]
    seq_mask = jnp.where(jnp.arange(max_num_seqs)[None, :] < num_seqs, seq_mask, False)
    seq_ids = jnp.sum(seq_mask, axis=1)

    q_start = cu_q_lens[seq_ids]
    q_len = cu_q_lens[seq_ids + 1] - q_start
    seq_len = seq_lens[seq_ids]
    token_pos = seq_len - q_len + (token_ids - q_start)
    kv_len = seq_len // compression_ratio
    if strict_causal:
        visible = jnp.minimum(kv_len, (token_pos + 1) // compression_ratio)
    else:
        visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
    visible = cp_local_length(visible, cp_rank, cp_size, interleave_c)
    if kv_window is not None:
        win_start, win_len = kv_window
        visible = jnp.clip(visible - win_start[seq_ids], 0, win_len[seq_ids])
    valid = token_ids < cu_q_lens[num_seqs]
    return jnp.where(valid, jnp.minimum(visible, num_positions), 0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "compression_ratio",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "buffer_count",
        "vmem_limit_bytes",
        "decode_req_batch_size",
        "enable_early_exit",
        "kv_layout",
        "chunk_tokens",
        "cp_size",
        "interleave_size",
        "return_scores",
        "uniform_q_len",
        "sc_flat_scores",
        "sc_topk_coop_max_rows",
        "kv_split",
        "row_window_tokens",
        "verify_bd",
        "strict_causal",
        "chunk_sched_groups",
        "group_max",
        "group_max_min_kv",
    ),
)
def streamindex_topk(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    indexer_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads]
    # D = align_to(actual_head_dim, 128); W = align_to(D + D // 128, 128)
    # HEAD_ALONG_SUBLANE: uint8[pages, page_size // 4, 4, W]
    # SEQ_ALONG_LANE:     uint8[pages, D // 4 + 1, 4, page_size]
    cache_kv: jax.Array,
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    buffer_count: tuple[int, int, int] | int | None = DEFAULT_BUFFER_COUNT,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
    chunk_tokens: int | None = None,
    cp_size: int = 1,
    cp_rank: jax.Array | int = 0,
    interleave_size: int = 1,
    return_scores: bool = False,
    uniform_q_len: int | None = None,
    sc_flat_scores: bool = False,
    sc_topk_coop_max_rows: int = 0,
    kv_split: int = 1,
    kv_split_rank: jax.Array | int = 0,
    row_window: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None,
    row_window_tokens: int = 0,
    verify_bd: bool | None = None,
    strict_causal: bool = False,
    chunk_sched_groups: bool = True,
    group_max: bool = False,
    group_max_min_kv: int = 0,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """StreamIndex Top-K retrieval.

    row_window (TPU_DSV4_INDEXER_TOKEN_SPLIT_ABOVE): `q` holds rows
    `[offset, offset + T)` of a `row_window_tokens`-row batch and the metadata
    is that window's (`token_window_metadata`); `row_window = (seq_lens,
    cu_q_lens, distribution, offset)` of the whole batch sets each row's
    visible length exactly as the whole batch's top-k does.

    kv_split > 1 (ported from patemotter fdcc70a0d): score only rank
    `kv_split_rank`'s contiguous, page-aligned 1/kv_split share of every
    sequence's compressed positions of a cache every rank holds whole, and
    return that share's top-k as GLOBAL columns with their score bits
    (unfilled slots: index -1, score BITS_NEG_INF). A top-k over all ranks'
    candidates (rank-major) is the unsplit top-k.

    Args:
      q: concatenated all sequences' queries.
      indexer_weights: concatenated all sequences' indexer weights.
      cache_kv: the current kv cache, packed as uint8. Its shape depends on
        `kv_layout`. With `D = align_to(actual_head_dim, 128)`: *
        `HEAD_ALONG_SUBLANE`: `[total_num_pages, page_size // 4, 4, align_to(D + D
        // 128, 128)]`. Tokens sit on the sublane dimension; each token's lane
        record holds its `D` FP8 key bytes followed by `D // 128` UE8M0 scale
        bytes, padded out to a multiple of the 128-lane register width. For
        `D=128` that is 129 useful bytes stored in 256. * `SEQ_ALONG_LANE`:
        `[total_num_pages, D // 4 + 1, 4, page_size]`. Tokens sit on the lane
        dimension; sublane rows `[0, D)` hold the FP8 key bytes, row `D` holds the
        per-token UE8M0 scale, and rows `D+1 .. D+3` are padding that completes
        the final packed sublane group. For `D=128` that is 132 bytes per token.
      seq_lens: the length of each sequence in the kv cache (uncompressed).
      page_indices: flattened page indices look-up table by (seq_id, page_id).
      cu_q_lens: the cumulative sum of the effective query lengths. Similar to
        kv_lens, only the first num_seqs+1 values are valid.
      distribution: (i, j, k) represents that sequences[0:i] are decode-only,
        sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
        k is also the total number of sequences.
      k: Number of top-K elements to retrieve.
      compression_ratio: KV cache compression ratio.
      num_kv_pages_per_block: number of kv pages to be processed in one block in
        the pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
      num_queries_per_block: number of queries to be processed in one block in the
        pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
      buffer_count: buffer count for the pallas kernel. This is a tuple of
        (decode, prefill, mixed) cases or a single integer. Defaults to (4, 3, 3).
      vmem_limit_bytes: the vmem limit for the pallas kernel.
      enable_early_exit: whether to enable early exit using jax.lax.cond when k >=
        kv_len for all sequences in the batch. Defaults to False.
      decode_req_batch_size: maximum decode batch size per iteration.
      kv_layout: memory layout of `cache_kv`. `SEQ_ALONG_LANE` removes the lane
        padding that per-token quantization forces on `HEAD_ALONG_SUBLANE` and is
        therefore substantially faster whenever the kernel is HBM bound.
      chunk_tokens: size, in tokens, of each piece to cut the flat token axis
        into. Each piece scores into its own buffer and runs its own SparseCore
        top-k, and consecutive pieces share an XLA scheduling group so chunk i's
        top-k overlaps chunk i+1's scoring. None, or any size that does not divide
        `max_num_tokens`, runs a single unpipelined pass. Defaults to None: the
        size that wins depends on the workload and the KV layout, so the caller
        picks it.
      cp_size: number of context-parallel ranks the compressed KV cache is sharded
        over. 1 (default) means no sharding and the whole CP path compiles away.
        When > 1, `cache_kv` / `page_indices` describe only this rank's shard, and
        the returned top-k is *local* -- correct only after the cross-rank merge
        in `streamindex_topk_dcp`.
      cp_rank: this rank's index in the CP group. Traced, so one compiled program
        serves every rank.
      interleave_size: CP chunk-interleave width in uncompressed tokens. Must be a
        multiple of `compression_ratio`.
      uniform_q_len: the query length every sequence in
        `[distribution[0], distribution[1])` has -- speculative-decoding verify
        windows, for instance. That segment is then scored by its own pass with
        a static q length and the (decode, prefill, mixed) block sizes' prefill
        entry, instead of the decode passes, which assume one query token. None
        keeps the segment on the decode passes.
      sc_flat_scores: hand the SparseCore top-k the scoring kernel's
        [T, S, 128] buffer as is, with no padding rows. Its bytes already are
        the flat row-major rows the top-k reads, so XLA passes it through as a
        bitcast; the default reshapes it to [T, S * 128] and pads the rows to
        the top-k's wave size, which XLA lowers as a relayout, a pad and a
        SparseCore data-format conversion of the whole (padded) buffer. Same
        selection either way.
      sc_topk_coop_max_rows: `sparsecore_topk`'s `coop_max_rows`.
      verify_bd: score the `uniform_q_len` pass (windows of <= 8 tokens) with a
        block-diagonal q and a packed bf16 head reduction (see `score_pipeline`).
        Not bit-exact: relu(q.k) and the weights are rounded to bf16, as in the
        mixed pass. None reads TPU_STREAMIDX_VERIFY_BD.
      strict_causal: a token selects only among the compressed groups that are
        complete at its own position, (pos + 1) // compression_ratio of them.
        False keeps the previous bound, pos // compression_ratio + 1, under
        which a multi-token forward (chunked prefill, speculative verify) lets
        a token see its own group's entry when that group completes later in
        the same forward.
      return_scores: also return the score of each selected position, which is
        what makes a cross-rank merge possible. A local top-k over a sharded KV
        axis is not the global top-k, so the merge needs the scores the same way
        the DCP attention kernels need an LSE output.

    Returns:
      Top-K indices (rank-local when `cp_size > 1`, global compressed positions
      when `cp_size == 1`), or `(indices, scores)` if `return_scores`. Unfilled
      slots are `-1` with score `-inf`.
    """
    # Scale factors for the FP8 index cache format are packed directly inside
    # `cache_kv`, keeping HBM transactions fused.

    kv_layout = KVLayout.parse(kv_layout)
    if verify_bd is None:
        from vllm_torchtpu import envs

        verify_bd = envs.TPU_STREAMIDX_VERIFY_BD

    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
    if cp_size > 1:
        if interleave_size % compression_ratio != 0:
            raise ValueError(
                f"interleave_size ({interleave_size}) must be a multiple of "
                f"compression_ratio ({compression_ratio}) for the CP chunk "
                "boundary to fall on a compressed-row boundary."
            )
        if enable_early_exit:
            raise NotImplementedError(
                "enable_early_exit is not supported with cp_size > 1."
            )
    if enable_early_exit and return_scores:
        raise NotImplementedError(
            "return_scores is not supported with enable_early_exit."
        )
    interleave_c = interleave_size // compression_ratio if cp_size > 1 else 1

    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )
    if compression_ratio < 1 or (compression_ratio & (compression_ratio - 1)) != 0:
        raise ValueError("compression_ratio must be a power of 2.")

    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block

    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
    else:
        num_queries_per_blocks = num_queries_per_block

    if buffer_count is None:
        buffer_counts = list(DEFAULT_BUFFER_COUNT)
    elif isinstance(buffer_count, int):
        buffer_counts = [buffer_count for _ in range(3)]
    else:
        buffer_counts = list(buffer_count)

    if len(buffer_counts) != 3:
        raise ValueError(
            f"buffer_count must be a 3-tuple or a single integer, got {buffer_count}"
        )

    if chunk_tokens is not None and chunk_tokens % decode_req_batch_size:
        raise ValueError(
            f"chunk_tokens ({chunk_tokens}) must be a multiple of"
            f" decode_req_batch_size ({decode_req_batch_size}), or a cut can"
            " fall inside a decode block and split it across two chunks."
        )

    max_num_seqs = seq_lens.shape[0]

    original_dtype = q.dtype
    actual_head_dim = q.shape[-1]

    prepared_indexer_weights = prepare_index_weights(indexer_weights, original_dtype)
    q = prepare_q_inputs(q)
    head_dim = q.shape[-1]

    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        total_num_pages, kv_sublane_groups, kv_packing, page_size = cache_kv.shape
        expected_shape = get_kv_cache_shape(
            total_num_pages, page_size, actual_head_dim, kv_layout
        )
        if cache_kv.shape != expected_shape:
            raise ValueError(
                "SEQ_ALONG_LANE expects cache_kv of shape"
                f" {expected_shape}, got {cache_kv.shape}."
            )
        if kv_sublane_groups * kv_packing != head_dim + kv_packing:
            raise ValueError(
                f"Inconsistent SEQ_ALONG_LANE cache: {kv_sublane_groups=},"
                f" {kv_packing=} does not match {head_dim=}."
            )
    else:
        _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
        page_size = page_size_per_kv_packing * kv_packing
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    kv_window = None
    global_seq_lens = seq_lens
    if kv_split > 1:
        if cp_size > 1 or enable_early_exit:
            raise NotImplementedError("kv_split needs cp_size == 1, no early exit.")
        return_scores = True
        kv_window = kv_split_window(
            seq_lens,
            kv_split_rank,
            kv_split=kv_split,
            compression_ratio=compression_ratio,
            page_size=page_size,
        )
        # The passes see each sequence as its window alone: the page table row
        # starts at the window's first page, the length is the window's. The
        # window length is held at >= the sequence's query length so the
        # passes' query positions stay non-negative; the extra columns are cut
        # by the row lengths (clipped to the window) before the top-k.
        first_page = kv_window[0] // page_size
        cols = jnp.arange(pages_per_seq, dtype=jnp.int32)[None, :] + first_page[:, None]
        page_indices = jnp.take_along_axis(
            page_indices.reshape(max_num_seqs, pages_per_seq),
            jnp.minimum(cols, pages_per_seq - 1),
            axis=1,
        ).reshape(-1)
        q_lens_per_seq = cu_q_lens[1 : max_num_seqs + 1] - cu_q_lens[:max_num_seqs]
        seq_lens = jnp.maximum(
            kv_window[1] * compression_ratio, jnp.maximum(q_lens_per_seq, 0)
        ).astype(seq_lens.dtype)

    for bkv_p in num_kv_pages_per_blocks:
        bkv_sz = page_size * bkv_p
        if bkv_sz % 128 != 0:
            raise ValueError(
                f"bkv_sz ({page_size} * {bkv_p} = {bkv_sz}) must be a multiple of 128."
            )

    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * page_size // 128
        for bkv_p in num_kv_pages_per_blocks
    )

    def _block_sizes(num_queries_per_block, bkv_p, static_q_len):
        """Query and KV block sizes a pass runs with."""
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz = page_size * bkv_p
        return bq_sz, bkv_sz

    def run_scores_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        scores_init,
        seq_lens,
        page_indices,
        cu_q_lens,
        start_seq_idx,
        end_seq_idx,
        static_q_len,
        num_kv_pages_per_block,
        num_queries_per_block,
        buffer_count,
        seq_batch_size,
        out_dtype,
        pass_meta,
        case=MlaCase.MIXED,
        chunk_token_start=None,
        chunk_tokens=None,
        scheduling_group_id=None,
        chunk_idx=0,
    ):
        max_num_tokens, num_q_heads, head_dim = q.shape
        # Only support batching for decode sequences.
        # TODO: support batching for decode sequences with speculative decoding
        # enabled, e.g. static_q_len = gamma + 1.
        if seq_batch_size > 1:
            assert static_q_len == 1

        chunked = chunk_token_start is not None
        if not chunked:
            chunk_tokens = max_num_tokens

        bkv_p = num_kv_pages_per_block
        bq_sz, bkv_sz = _block_sizes(num_queries_per_block, bkv_p, static_q_len)

        hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
        # Passes 2 and 3 donate the previous pass's HBM buffer so its rows survive
        # into this output. Only pass 1 has no donor, so Pallas allocates it there;
        # one buffer per chunk either way.
        aliased = scores_init is not None
        in_specs = [
            hbm_spec,  # page_indices
            hbm_spec,  # q
            hbm_spec,  # prepared_indexer_weights
            hbm_spec,  # cache_kv
        ]
        if aliased:
            # scores_init (+ gm_init), aliased to the outputs
            in_specs += [hbm_spec] * (2 if group_max else 1)
        out_specs = (hbm_spec, hbm_spec) if group_max else hbm_spec

        scratch_shapes = []

        chunk_start = 0 if chunk_token_start is None else chunk_token_start
        scalar_prefetches = (
            seq_lens,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx, chunk_start], jnp.int32),
            pass_meta,
            jnp.asarray([cp_rank], jnp.int32),
        )

        num_scalar_prefetch = len(scalar_prefetches)
        num_flat_prefetches = len(jax.tree_util.tree_leaves(scalar_prefetches))
        # The aliased `scores_init` is the 5th operand after the flat prefetches.
        input_output_aliases = {num_flat_prefetches + 4: 0} if aliased else {}
        if aliased and group_max:
            input_output_aliases[num_flat_prefetches + 5] = 1

        layout_tag = "" if kv_layout == KVLayout.HEAD_ALONG_SUBLANE else "-sal"
        scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}{layout_tag}"
        if chunked:
            # The chunk count is part of the kernel's identity, so a sweep over chunk
            # sizes shows up in the profile instead of silently reusing one shape.
            scope_name += f"-c{max_num_tokens // chunk_tokens}"
        pallas_kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    kernel,
                    compression_ratio=compression_ratio,
                    static_q_len=static_q_len,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    seq_batch_size=seq_batch_size,
                    buffer_count=buffer_count,
                    case=case,
                    kv_layout=kv_layout,
                    cp_size=cp_size,
                    interleave_c=interleave_c,
                    chunk_idx=chunk_idx,
                    verify_bd=verify_bd and case == MlaCase.PREFILL,
                    group_max=group_max,
                    gm_min_kv=group_max_min_kv if group_max else 0,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=num_scalar_prefetch,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=(1,),
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=(
                    jax.ShapeDtypeStruct(
                        shape=(chunk_tokens, num_sublanes_total, 128),
                        dtype=out_dtype,
                    ),
                    jax.ShapeDtypeStruct(
                        shape=(chunk_tokens, num_sublanes_total // GM_TILES, 128),
                        dtype=jnp.int32,
                    ),
                )
                if group_max
                else jax.ShapeDtypeStruct(
                    shape=(chunk_tokens, num_sublanes_total, 128),
                    dtype=out_dtype,
                ),
                input_output_aliases=input_output_aliases,
                name=scope_name,
            )
        )
        scores = pallas_kernel(
            *scalar_prefetches,
            page_indices,
            q,
            prepared_indexer_weights,
            cache_kv,
            *((tuple(scores_init) if group_max else (scores_init,)) if aliased else ()),
        )
        if scheduling_group_id is not None:
            scores = jax.tree.map(
                lambda x: xla_metadata.set_xla_metadata(
                    x, _scheduling_group_id=scheduling_group_id
                ),
                scores,
            )
        return scores

    def _pass_specs(decode_batch_end):
        """`run_scores_kernel` kwargs for the passes over the batch."""
        decode_cfg = dict(
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            buffer_count=buffer_counts[0],
            static_q_len=1,
            case=MlaCase.DECODE,
        )
        batched_decode = (
            ()
            if max_num_seqs < decode_req_batch_size
            else (
                dict(
                    start_seq_idx=jnp.array(0),
                    end_seq_idx=decode_batch_end,
                    seq_batch_size=decode_req_batch_size,
                    **decode_cfg,
                ),
            )
        )
        uniform = (
            ()
            if uniform_q_len is None
            else (
                dict(
                    num_kv_pages_per_block=num_kv_pages_per_blocks[1],
                    num_queries_per_block=num_queries_per_blocks[1],
                    buffer_count=buffer_counts[1],
                    start_seq_idx=distribution[0],
                    end_seq_idx=distribution[1],
                    static_q_len=uniform_q_len,
                    seq_batch_size=1,
                    case=MlaCase.PREFILL,
                ),
            )
        )
        # Passes write disjoint token rows, so their order is free. The decode
        # passes go last: with speculative decoding they are usually empty, and
        # only a pass that receives the buffer can be skipped (see
        # `_scores_for_chunk`).
        return (
            *uniform,
            dict(
                num_kv_pages_per_block=num_kv_pages_per_blocks[2],
                num_queries_per_block=num_queries_per_blocks[2],
                buffer_count=buffer_counts[2],
                start_seq_idx=distribution[1],
                end_seq_idx=distribution[2],
                static_q_len=None,
                seq_batch_size=1,
                case=MlaCase.MIXED,
            ),
            *batched_decode,
            # Handle num_decode_seqs % decode_req_batch_size != 0 case.
            dict(
                start_seq_idx=decode_batch_end,
                end_seq_idx=distribution[0 if uniform else 1],
                seq_batch_size=1,
                **decode_cfg,
            ),
        )

    def _pass_metadata(spec, chunk_starts, chunk_tokens):
        """One schedule per pass, holding every chunk's steps back to back."""
        bq_sz, bkv_sz = _block_sizes(
            spec["num_queries_per_block"],
            spec["num_kv_pages_per_block"],
            spec["static_q_len"],
        )
        return metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=spec["start_seq_idx"],
            end_seq_idx=spec["end_seq_idx"],
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            pages_per_seq=pages_per_seq,
            compression_ratio=compression_ratio,
            static_q_len=spec["static_q_len"],
            seq_batch_size=spec["seq_batch_size"],
            page_size=page_size,
            max_num_tokens=q.shape[0],
            chunk_token_start=chunk_starts,
            chunk_tokens=chunk_tokens,
            cp_rank=cp_rank,
            cp_size=cp_size,
            interleave_c=interleave_c,
        )

    def _scores_for_chunk(
        passes,
        chunk_idx,
        chunk_token_start,
        chunk_tokens,
        out_dtype,
        scheduling_group_id,
    ):
        """Scores for one window of the flat token axis, in a private buffer."""
        # `scores=None`: no -inf pre-fill. It would write all of HBM scores, and
        # XLA would common the identical per-chunk fills into one buffer then clone
        # it back out (~370us on 1GB of scores). `row_lengths` bounds the reads.
        scores = None
        if chunk_token_start is not None:
            # Pipelined pieces: the mixed pass goes first so that a piece holding
            # no verify-window rows can skip the uniform pass (below) - every
            # piece would otherwise launch it, empty, for its page-table copy.
            passes = tuple(sorted(passes, key=lambda sp: sp[0]["case"] != MlaCase.MIXED))
        for spec, pass_meta in passes:

            def run(scores_init, spec=spec, pass_meta=pass_meta):
                return run_scores_kernel(
                    q,
                    prepared_indexer_weights,
                    cache_kv,
                    scores_init,
                    seq_lens,
                    page_indices,
                    cu_q_lens,
                    pass_meta=pass_meta,
                    chunk_idx=chunk_idx,
                    chunk_token_start=chunk_token_start,
                    chunk_tokens=chunk_tokens,
                    out_dtype=out_dtype,
                    scheduling_group_id=scheduling_group_id,
                    **spec,
                )

            if (
                scores is not None
                and chunk_token_start is not None
                and spec["case"] == MlaCase.PREFILL
            ):
                scores = jax.lax.cond(
                    pass_meta.num_steps[chunk_idx] > 0,
                    run,
                    lambda scores_init: scores_init,
                    scores,
                )
            elif scores is not None and spec["case"] == MlaCase.DECODE:
                # An empty pass still pays its launch and the page-table copy
                # into SMEM; skipping it leaves the buffer as the pass would.
                scores = jax.lax.cond(
                    spec["end_seq_idx"] > spec["start_seq_idx"],
                    run,
                    lambda scores_init: scores_init,
                    scores,
                )
            else:
                scores = run(scores)
        assert scores is not None
        return scores

    def _common_path(_):
        # Raw f32 bits, not f32. The SparseCore top-k compares bits, and this is
        # the dtype it wants to read, so nothing has to reformat HBM scores.
        out_dtype = jnp.int32
        max_num_tokens = q.shape[0]

        # TODO: we shall sort the sequences by length, so that multiple decode
        # sequences in one batch have similar lengths to reduce waste of compute.
        # With the same batch size, the longest sequence will determine number of
        # blocks to run computation for.
        decode_batch_end = (
            jnp.array(0)
            if max_num_seqs < decode_req_batch_size
            else distribution[0] // decode_req_batch_size * decode_req_batch_size
        )

        num_positions = max(num_sublanes_total * 128, k)
        if row_window is None:
            eff_row_lengths = _effective_row_lengths(
                seq_lens=global_seq_lens,
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                num_tokens=max_num_tokens,
                num_positions=num_positions,
                compression_ratio=compression_ratio,
                cp_rank=cp_rank,
                cp_size=cp_size,
                interleave_c=interleave_c,
                kv_window=kv_window,
                strict_causal=strict_causal,
            )
        else:
            if cp_size > 1 or kv_window is not None:
                raise NotImplementedError("row_window needs cp_size == 1, no kv_split.")
            full_seq_lens, full_cu, full_dist, row_offset = row_window
            eff_row_lengths = jax.lax.dynamic_slice(
                _effective_row_lengths(
                    seq_lens=full_seq_lens,
                    cu_q_lens=full_cu,
                    distribution=full_dist,
                    num_tokens=row_window_tokens,
                    num_positions=num_positions,
                    compression_ratio=compression_ratio,
                    strict_causal=strict_causal,
                ),
                (row_offset,),
                (max_num_tokens,),
            )

        split_chunks = False
        call_chunk_tokens = chunk_tokens
        if (
            kv_window is not None
            and chunk_tokens is None
            and max_num_tokens > _KV_SPLIT_SC_MAX_ROWS
            and max_num_tokens % _KV_SPLIT_SC_MAX_ROWS == 0
        ):
            # A share returns score bits, and the score-returning SparseCore
            # top-k over 2048 rows does not fit SparseCore memory (2109439 >
            # 2097151 words, twin j-7aed76d2): score and select in row chunks.
            # Without scheduling-group annotations: in the served graph they
            # failed to compile ("annotation groups with gaps", twin j-6706060c).
            call_chunk_tokens = _KV_SPLIT_SC_MAX_ROWS
            split_chunks = True

        if call_chunk_tokens is None or max_num_tokens % call_chunk_tokens:
            num_chunks, chunk_len = 1, max_num_tokens
        else:
            num_chunks, chunk_len = (
                max_num_tokens // call_chunk_tokens,
                call_chunk_tokens,
            )

        # Ids count from 1, so the 0 is never a real group: a single chunk has no
        # pair to annotate and takes the `None` branch below both times.
        group_base = 0
        if not chunk_sched_groups:
            # TPU_STREAMIDX_CHUNK_SCHED_GROUPS=0: pieces without annotations.
            split_chunks = True
        if num_chunks > 1 and not split_chunks:
            group_base = config.reserve_scheduling_group_ids(num_chunks - 1)

        chunk_starts = (
            (None,)
            if num_chunks == 1
            else tuple(m * chunk_len for m in range(num_chunks))
        )
        passes = tuple(
            (spec, _pass_metadata(spec, chunk_starts, chunk_len))
            for spec in _pass_specs(decode_batch_end)
        )
        topk_idxs = []
        topk_scores_list = []
        for m in range(num_chunks):
            # Group g holds chunk g's SparseCore top-k and chunk g+1's TensorCore
            # scoring, so the scheduler emits `sc(g).start, tc(g+1), sc(g).done`.
            # Chunk 0's scoring and the last top-k have no partner.
            tc_group = None if (m == 0 or split_chunks) else group_base + m - 1
            sc_group = (
                None if (m == num_chunks - 1 or split_chunks) else group_base + m
            )

            start = m * chunk_len
            scores = _scores_for_chunk(
                passes,
                m,
                chunk_starts[m],
                chunk_len,
                out_dtype,
                tc_group,
            )
            gm = None
            if group_max:
                scores, gm = scores
            # A kv-split share takes the by-length pass, never the cooperative
            # one (lowc cccc1c5b7: coop shares came back wrong next to the merge).
            topk_kwargs = dict(
                coop_max_rows=0 if kv_window is not None else sc_topk_coop_max_rows
            )
            if sc_flat_scores and scores.shape[1] * 128 >= k:
                topk_kwargs["pad_rows"] = False
            else:
                scores = scores.reshape(chunk_len, -1)
            if scores.ndim == 2 and scores.shape[1] < k:
                scores = jnp.pad(
                    scores,
                    ((0, 0), (0, k - scores.shape[1])),
                    constant_values=BITS_NEG_INF,
                )

            if gm is not None:
                # K-IDXM: long rows of the blocked (mixed) pass are selected from
                # their group maxima; every other row as before. Same selection.
                assert not return_scores
                if scores.ndim == 2:  # sc_flat_scores off: the same bytes, viewed 3D
                    scores = scores.reshape(scores.shape[0], -1, 128)
                from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk_pf import (
                    sparsecore_topk_pf,
                )

                rows = start + jnp.arange(chunk_len, dtype=jnp.int32)
                eligible = (rows >= cu_q_lens[distribution[1]]) & (
                    rows < cu_q_lens[distribution[2]]
                )
                chunk_topk_idxs = sparsecore_topk_pf(
                    scores,
                    gm,
                    k,
                    eff_row_lengths[start : start + chunk_len],
                    eligible=eligible,
                    min_len=max(group_max_min_kv, 16 * k + 16),
                    scheduling_group_id=sc_group,
                    **topk_kwargs,
                )
            elif return_scores:
                chunk_topk_idxs, chunk_topk_scores = sparsecore_topk(
                    scores,
                    k,
                    row_lengths=eff_row_lengths[start : start + chunk_len],
                    write_empty_rows=True,
                    scheduling_group_id=sc_group,
                    return_scores=True,
                    **topk_kwargs,
                )
                topk_scores_list.append(chunk_topk_scores)
            else:
                chunk_topk_idxs = sparsecore_topk(
                    scores,
                    k,
                    row_lengths=eff_row_lengths[start : start + chunk_len],
                    write_empty_rows=True,
                    scheduling_group_id=sc_group,
                    return_scores=False,
                    **topk_kwargs,
                )
            topk_idxs.append(chunk_topk_idxs)

        out = topk_idxs[0] if len(topk_idxs) == 1 else jnp.concatenate(topk_idxs)
        out = out[:max_num_tokens, :k]
        if kv_window is not None:
            seq_ids = _token_seq_ids(
                cu_q_lens, distribution, max_num_tokens, max_num_seqs
            )
            first_row = jnp.pad(kv_window[0], (0, 1))[seq_ids]
            out = jnp.where(out >= 0, out + first_row[:, None], -1)

        if cp_size == 1 and not return_scores:
            return out

        # Columns in `out` are rank-local compressed positions. Even when
        # `cp_size > 1`, `streamindex_topk` only returns rank-local indices (callers
        # like `streamindex_topk_dcp` resolve winners directly against their local
        # candidate list without converting to global coordinates).
        if not return_scores:
            return out

        scores_out = (
            topk_scores_list[0]
            if len(topk_scores_list) == 1
            else jnp.concatenate(topk_scores_list)
        )
        scores_out = scores_out[:max_num_tokens, :k]
        if kv_window is not None:
            scores_out = jnp.where(out >= 0, scores_out, BITS_NEG_INF)
        return out, scores_out

    def _fast_path(_):
        token_idx = jnp.arange(q.shape[0])
        seq_idx = jnp.minimum(
            jnp.searchsorted(cu_q_lens[1:], token_idx, side="right"),
            seq_lens.shape[0] - 1,
        )
        seq_len = seq_lens[seq_idx]
        q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
        q_start = cu_q_lens[seq_idx]
        q_abs_pos = (seq_len - q_len) + (token_idx - q_start)
        max_valid_idx = jnp.minimum(
            seq_len // compression_ratio - 1,
            (q_abs_pos + 1) // compression_ratio - 1
            if strict_causal
            else q_abs_pos // compression_ratio,
        )
        max_valid_idx = jnp.where(token_idx < cu_q_lens[-1], max_valid_idx, -1)
        s_idx = jnp.arange(k, dtype=jnp.int32)[None, :]
        return jnp.where(s_idx <= max_valid_idx[:, None], s_idx, -1)

    if not enable_early_exit:
        return _common_path(None)
    return jax.lax.cond(
        jnp.max(seq_lens) // compression_ratio <= k,
        _fast_path,
        _common_path,
        None,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "mesh",
        "k",
        "compression_ratio",
        "dcp_size",
        "interleave_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "buffer_count",
        "vmem_limit_bytes",
        "decode_req_batch_size",
        "kv_layout",
        "chunk_tokens",
        "dcp_axis_name",
        "a2a_merge",
    ),
)
def streamindex_topk_dcp(
    q: jax.Array,  # [padded_num_tokens, num_q_heads, head_dim], replicated
    indexer_weights: jax.Array,  # [padded_num_tokens, num_q_heads], replicated
    cache_kv: jax.Array,  # sharded on axis 0 over the DCP axis
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * virtual_pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    mesh,
    k: int,
    compression_ratio: int,
    dcp_size: int,
    interleave_size: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    buffer_count: tuple[int, int, int] | int | None = DEFAULT_BUFFER_COUNT,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
    chunk_tokens: int | None = None,
    dcp_axis_name: str = DCP_AXIS_NAME,
    a2a_merge: bool = False,
) -> jax.Array:
    """Exact global top-k over a DCP-sharded KV cache, delivered rank-local.

    Architecture:
      1. Local Scoring: Every rank scores the replicated query tokens against its
         own local KV cache shard via `streamindex_topk`, producing candidate
         indices and scores of shape `[padded_num_tokens, k]`. Because DCP
         partitions the KV cache, a true global top-k winner is beaten by at most
         `k - 1` positions globally, and therefore by at most `k - 1` positions
         on its owner rank. Thus, every true global winner is guaranteed to
         survive into its owner's local candidate list.
      2. Candidate All-Gather: A single `all_gather` along the candidate axis
         (axis 1) exchanges only the candidate *scores*:
         `[T, k] -> [T, dcp_size * k]`. The local candidate indices stay home.
      3. Fused SparseCore Merge, Owner Filter & Prefix Compaction: Each rank runs
         `sparsecore_topk_dcp_merge` over the gathered `[T, dcp_size * k]` scores
         and its pre-gather `idxs[T, k]`. Inside a single SparseCore pass, the
         kernel finds the exact global top-k threshold across all `dcp_size * k`
         scores, scans only `[dcp_rank * k, (dcp_rank + 1) * k)` during emission,
         resolves surviving slots directly to `idxs`, and left-packs them into a
         `-1`-padded prefix of width `k`. The ranks' outputs form an exact
         partition of the global top-k.

    By gathering candidates along axis 1 upfront rather than partitioning tokens
    over an `all_to_all`, all communication is consolidated into a single upfront
    collective phase, avoiding the barrier latency of a multi-phase pipeline and
    preserving the token axis `T` unpartitioned.

    `a2a_merge` makes the opposite trade for steps 2-3: an `all_to_all` hands
    rank `j` every rank's candidate scores for token block `j`, it runs
    `sparsecore_topk` over only that `T / dcp_size` block, and an
    `all_gather(axis=0)` of the int32 slots restores the token axis. Each rank
    then receives `T * k` scores plus `T * k` slots instead of
    `T * dcp_size * k` scores, and merges `T / dcp_size` rows instead of `T`, at
    the cost of a second, dependent collective and of the fused merge: the
    owner filter and compaction run afterwards in
    `dcp_sc_compact.pack_owned_winners`, since a slot still names its source
    rank and that rank's own list, so the indices never travel on either path.

    Args:
      q: replicated query tokens, in natural request-major order
        `[padded_num_tokens, num_q_heads, head_dim]`.
      indexer_weights: replicated query weights `[padded_num_tokens,
        num_q_heads]`.
      cache_kv: this rank's shard of the compressed KV cache.
      seq_lens: the length of each sequence in the kv cache (uncompressed).
      page_indices: replicated *virtual* page ordinals, resolved against each
        rank's own shard by `local = virtual % num_local_pages`.
      cu_q_lens: the cumulative sum of the effective query lengths.
      distribution: (i, j, k) partition indices.
      mesh: mesh containing `dcp_axis_name`.
      k: Number of top-K elements to retrieve.
      compression_ratio: KV cache compression ratio.
      dcp_size: number of context-parallel ranks.
      interleave_size: chunk-interleave width in uncompressed tokens.
      num_kv_pages_per_block: number of kv pages per block in pallas kernel.
      num_queries_per_block: number of queries per block in pallas kernel.
      buffer_count: buffer count for pallas kernel.
      vmem_limit_bytes: vmem limit for pallas kernel.
      decode_req_batch_size: maximum decode batch size per iteration.
      kv_layout: physical layout of `cache_kv`, a `KVLayout` or its string value.
      chunk_tokens: forwarded to `streamindex_topk` for the per-rank local scoring
        pass; see there for the semantics. Chunking cuts the replicated
        query-token axis, which is orthogonal to the KV axis DCP shards, so the
        two compose.
      dcp_axis_name: name of the DCP mesh axis.
      a2a_merge: merge over the token axis with an `all_to_all` instead of over
        the candidate axis with an `all_gather`; see above. Same result.

    Returns:
      i32[padded_num_tokens, k] of this rank's own *local* cache indices for
      every token, packed into a prefix and `-1` padded, sharded over
      `dcp_axis_name` (global leading dim `dcp_size * padded_num_tokens`).
      Callers derive per-token counts from the `-1` padding; a row that is all
      `-1` means this rank owns none of that token's top-k, which the attention
      side must mask out of the LSE merge rather than attend over. Rows are
      `k` wide but hold `k / dcp_size` entries on average, so a caller that
      gathers the full width pays `dcp_size`x more than it needs to; the
      attention side should bound its gather by the per-row count.
    """
    if dcp_size <= 1:
        raise ValueError(
            f"streamindex_topk_dcp requires dcp_size > 1, got {dcp_size}; "
            "use streamindex_topk for the unsharded case."
        )
    if dcp_axis_name not in mesh.axis_names:
        raise ValueError(f"mesh {mesh.axis_names} has no {dcp_axis_name!r} axis.")
    if mesh.shape[dcp_axis_name] != dcp_size:
        raise ValueError(
            f"dcp_size={dcp_size} does not match mesh axis"
            f" {dcp_axis_name!r}={mesh.shape[dcp_axis_name]}."
        )
    if q.shape[0] % dcp_size != 0:
        raise ValueError(
            f"padded_num_tokens={q.shape[0]} must be divisible by dcp_size={dcp_size}."
        )

    def _local(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
        dcp_rank = cp_rank_as_data(dcp_axis_name, dcp_size)
        local_page_indices = jnp.mod(page_indices, jnp.int32(cache_kv.shape[0]))

        idxs, scores = streamindex_topk(
            q,
            indexer_weights,
            cache_kv,
            seq_lens,
            local_page_indices,
            cu_q_lens,
            distribution,
            k=k,
            compression_ratio=compression_ratio,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            buffer_count=buffer_count,
            vmem_limit_bytes=vmem_limit_bytes,
            decode_req_batch_size=decode_req_batch_size,
            enable_early_exit=False,
            kv_layout=kv_layout,
            chunk_tokens=chunk_tokens,
            cp_size=dcp_size,
            cp_rank=dcp_rank,
            interleave_size=interleave_size,
            return_scores=True,
        )

        if a2a_merge:
            # Stage 1. Rank `j` collects every rank's scores for token block
            # `j`: [T, k] -> [T / dcp, dcp * k], rank-major like the gather.
            scores = lax.all_to_all(
                scores, dcp_axis_name, split_axis=0, concat_axis=1, tiled=True
            )
            # Stage 2. This block's exact global top-k. Full width as the row
            # length: -inf entries (unfilled local slots, padding tokens) are
            # never selected, so they need no masking.
            slots = sparsecore_topk(
                scores,
                k,
                row_lengths=jnp.full(
                    (scores.shape[0],), scores.shape[1], dtype=jnp.int32
                ),
                write_empty_rows=True,
            )
            # Stage 3. Every rank's merged block back, in token order: [T, k]
            # slots. A slot names its source rank and that rank's own list, so
            # the owner filter resolves it against this rank's `idxs` as is.
            slots = lax.all_gather(slots, dcp_axis_name, axis=0, tiled=True)
            return dcp_sc_compact.pack_owned_winners(slots, idxs, dcp_rank)

        # Stage 1. All-gather the scores; the indices stay home.
        scores = lax.all_gather(
            scores,
            dcp_axis_name,
            axis=1,
            tiled=True,
        )

        # Stage 2. Fused SparseCore global top-k threshold selection, owner
        # filtering, local index resolution, and prefix compaction.
        return sparsecore_topk_dcp_merge(
            scores,
            idxs,
            k,
            dcp_rank=dcp_rank,
            dcp_size=dcp_size,
        )

    replicated = P()
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            replicated,  # q
            replicated,  # indexer_weights
            P(dcp_axis_name),  # cache_kv
            replicated,  # seq_lens
            replicated,  # page_indices
            replicated,  # cu_q_lens
            replicated,  # distribution
        ),
        out_specs=P(dcp_axis_name),
        check_vma=False,
    )(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )
