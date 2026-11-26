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
    rank_major_merge_fits,
    sparsecore_topk,
    sparsecore_topk_dcp_merge,
)
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk_dcp import sparsecore_topk_dcp
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    bref_override,
    config,
    dcp_sc_compact,
    metadata,
)
from vllm_torchtpu.kernels.experimental.small_collectives import (
    all_gather as small_collectives_all_gather,
)

_small_all_gather = small_collectives_all_gather.small_all_gather

try:
    from vllm_torchtpu.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

MlaCase = config.MlaCase
KVLayout = config.KVLayout
DEFAULT_BUFFER_COUNT = config.DEFAULT_BUFFER_COUNT
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
DCP_AXIS_NAME = "dcp"
_DECODE_PACK_BUFFER_COUNT = 6
SMALL_AG_MAX_TOKENS = 8

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
        # Avoid multiple-of-8 pitch on strided score stores: padding by +1 gives
        # an odd row stride so full vector-width stores avoid splitting, cutting
        # padding from 8 rows down to 1 row per token.
        score_pitch = num_sublanes_bkv
        if enabled and strided_score_store and score_pitch % 8 == 0:
            score_pitch += 1

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
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [3] (start_seq, end_seq, chunk_token_start)
    metadata_ref,  # MetadataRef
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
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
    buffer_count: int = 3,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    chunk_idx: int = 0,
    decode_pack: bool = False,
    decode_page_major: bool = False,
):
    """Executes the n-buffered Pallas scoring pipeline for StreamIndex Top-K."""
    scores_hbm_ref = score_refs[-1]
    buffer_cnt = buffer_count
    seq_along_lane = kv_layout == KVLayout.SEQ_ALONG_LANE

    _, num_q_heads, head_dim = q_hbm_ref.shape

    u32_sal = (
        seq_along_lane
        and cache_kv_hbm_ref.ndim == 3
        and cache_kv_hbm_ref.dtype == jnp.uint32
    )
    if seq_along_lane:
        if u32_sal:
            (_, kv_sublane_groups, page_size) = cache_kv_hbm_ref.shape
            kv_packing = 4
        else:
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

    q_packing = get_dtype_packing(q_hbm_ref.dtype)
    decode_pack = (
        decode_pack
        and seq_along_lane
        and not prefill_cfg.enabled
        and seq_batch_size == 1
        and bq_sz == q_packing
        and 2 * bq_sz == 8
        and num_sublanes_bkv % 2 == 0
    )
    page_major = (
        decode_page_major
        and seq_along_lane
        and (u32_sal or not prefill_cfg.enabled)
        and seq_batch_size == 1
        and page_size % 128 == 0
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
    if u32_sal and page_major:
        assert kv_sublane_groups is not None
        # Avoid 3D flat u32 [1, 33, bkv_sz] or u8 bitcast+reshape on prefill:
        # 4D u32 [1, bkv_p, 33, page_size] copies each page contiguously and
        # slices u32[32, 128] with stride=1 without memref_reshape errors.
        kv_spec = pl.BlockSpec(
            (
                pl.BoundedSlice(seq_batch_size),
                bkv_p,
                kv_sublane_groups,
                page_size,
            ),
            lambda p_id: (seq_tile(p_id), bkv_block(p_id), 0, 0),
            pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
        )
    elif u32_sal:
        assert kv_sublane_groups is not None
        kv_spec = pl.BlockSpec(
            (
                pl.BoundedSlice(seq_batch_size),
                kv_sublane_groups,
                bkv_sz,
            ),
            lambda p_id: (seq_tile(p_id), 0, bkv_block(p_id)),
            pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
        )
    elif seq_along_lane and page_major:
        assert kv_sublane_groups is not None
        kv_spec = pl.BlockSpec(
            (
                pl.BoundedSlice(seq_batch_size),
                bkv_p * kv_sublane_groups,
                kv_packing,
                page_size,
            ),
            lambda p_id: (seq_tile(p_id), bkv_block(p_id), 0, 0),
            pipeline_mode=pl.Buffered(buffer_count=buffer_cnt, use_lookahead=True),
        )
    elif seq_along_lane:
        kv_spec = pl.BlockSpec(
            (
                pl.BoundedSlice(seq_batch_size),
                kv_sublane_groups,
                kv_packing,
                bkv_sz,
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
            page_major_groups=(kv_sublane_groups or 0) if page_major else 0,
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

    # ------------------------------------------------------------------------
    # SEQ_ALONG_LANE block access.
    #
    # The VMEM block is u8[seq_batch, kv_sublane_groups, kv_packing, bkv_sz]:
    # packed group g holds key dims [4g, 4g + 4) of every token, one byte per
    # dim, and group `kv_head_dim_groups` holds the UE8M0 scale in byte 0.
    # Viewed as u32, row `g * nt + j` (nt = bkv_sz // 128) is group g of lane
    # tile j, so a sublane-strided load of rows `j, j + nt, ...` yields the
    # [head_dim // 4, 128] words of lane tile j. Bitcasting those words to FP8
    # gives [head_dim, 128] with key dim 4g + i taken from byte i -- exactly the
    # MXU-native packed layout -- so no relayout is needed on the key path.
    # ------------------------------------------------------------------------
    num_lane_tiles_bkv = bkv_sz // 128
    tiles_per_page = page_size // 128

    def seq_along_lane_block_rows(bkv_vmem):
        if u32_sal:
            return bkv_vmem
        assert kv_sublane_groups is not None
        return bkv_vmem.bitcast(jnp.uint32).reshape(
            seq_batch_size, kv_sublane_groups * num_lane_tiles_bkv, 128
        )

    def _tile_row0(lane_tile_idx):
        if page_major:
            assert kv_sublane_groups is not None
            p, j = divmod(lane_tile_idx, tiles_per_page)
            return p * kv_sublane_groups * tiles_per_page + j, tiles_per_page
        return lane_tile_idx, num_lane_tiles_bkv

    def seq_along_lane_key_tile(rows, batch_idx, lane_tile_idx):
        """[head_dim, 128] FP8 keys of lane tile `lane_tile_idx`."""
        if u32_sal:
            if page_major:
                p, j = divmod(lane_tile_idx, tiles_per_page)
                words = rows[
                    batch_idx,
                    p,
                    pl.ds(0, kv_head_dim_groups),
                    pl.ds(j * 128, 128),
                ]
            else:
                words = rows[
                    batch_idx,
                    pl.ds(0, kv_head_dim_groups),
                    pl.ds(lane_tile_idx * 128, 128),
                ]
            return pltpu.bitcast(words, jnp.float8_e4m3fn)
        row0, stride = _tile_row0(lane_tile_idx)
        words = rows[
            batch_idx,
            pl.ds(row0, kv_head_dim_groups, stride=stride),
            :,
        ]
        return pltpu.bitcast(words, jnp.float8_e4m3fn)

    def seq_along_lane_scale_tile(rows, batch_idx, lane_tile_idx):
        """[1, 128] f32 dequantization scales of lane tile `lane_tile_idx`."""
        if u32_sal:
            if page_major:
                p, j = divmod(lane_tile_idx, tiles_per_page)
                words = rows[
                    batch_idx,
                    p,
                    pl.ds(kv_head_dim_groups, 1),
                    pl.ds(j * 128, 128),
                ]
            else:
                words = rows[
                    batch_idx,
                    pl.ds(kv_head_dim_groups, 1),
                    pl.ds(lane_tile_idx * 128, 128),
                ]
            return pltpu.bitcast((words & 0xFF) << 23, jnp.float32)
        row0, stride = _tile_row0(lane_tile_idx)
        row = row0 + kv_head_dim_groups * stride
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
            ret.append(s_summed.reshape(-1, num_sublanes_bkv, 128))
        return jnp.concatenate(ret, axis=0)

    def step_body(q_vmem, weights_vmem, bkv_vmem, scores_vmem, q_hm_ref=None):
        bkv_idx = bkv_block(pl.program_id(0))

        if q_hm_ref is not None:
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

        bkvs = []
        bkv_scales = []
        for batch_idx in range(seq_batch_size):
            if seq_along_lane:
                # Keys and scales come straight out of the packed block in their MXU
                # and lane layouts respectively; see `seq_along_lane_block_rows`.
                rows = seq_along_lane_block_rows(bkv_vmem)
                lane_tiles = range(num_lane_tiles_bkv)
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
        )
        scores_vmem[...] = pltpu.bitcast(scores, jnp.int32)

    def packed_decode_step(
        q_vmem, weights_vmem, bkv_vmem, scores_vmem, q_hm_ref, q2w_ref, q2_ref, w2_ref
    ):
        """Block-diagonal 4-query packed scoring step on SEQ_ALONG_LANE KV."""
        half, tiles_half, rows8 = bkv_sz // 2, num_lane_tiles_bkv // 2, 2 * bq_sz

        @pl.when(bkv_block(pl.program_id(0)) == 0)
        def _():
            q_hm_ref[...] = transpose_to_head_major(q_vmem[...], bq_sz)
            words = pltpu.bitcast(q_hm_ref[...], jnp.uint32)
            lo = (words & 0xFF) | ((words & 0xFF00) << 8)
            hi = ((words >> 16) & 0xFF) | ((words >> 8) & 0xFF0000)
            for s, shift in ((0, 0), (1, 8)):
                q2w_ref[s, pl.ds(0, num_q_heads, stride=2), :] = lo << shift
                q2w_ref[s, pl.ds(1, num_q_heads, stride=2), :] = hi << shift
            q2_ref[...] = pltpu.bitcast(
                jnp.concatenate([q2w_ref[0], q2w_ref[1]], axis=1), q_hbm_ref.dtype
            )
            w = weights_vmem[...].reshape(bq_sz, num_q_heads).astype(jnp.float32)
            for r in range(rows8):
                w2_ref[pl.ds(r, 1), :] = w[r // 2 : r // 2 + 1, :]

        rows = seq_along_lane_block_rows(bkv_vmem)
        keys = [seq_along_lane_key_tile(rows, 0, j) for j in range(num_lane_tiles_bkv)]
        scls = [
            seq_along_lane_scale_tile(rows, 0, j) for j in range(num_lane_tiles_bkv)
        ]
        k2 = jnp.concatenate(
            [
                jnp.concatenate(keys[:tiles_half], 1),
                jnp.concatenate(keys[tiles_half:], 1),
            ],
            axis=0,
        )
        st = jnp.einsum(
            "nd,dm->nm", q2_ref[...], k2, preferred_element_type=jnp.float32
        ).reshape(num_q_heads, rows8, half)
        w2 = w2_ref[...]
        acc = jnp.maximum(st[0], 0.0) * w2[:, 0:1]
        for h in range(1, num_q_heads):
            acc = acc + jnp.maximum(st[h], 0.0) * w2[:, h : h + 1]
        sel_hi = (lax.broadcasted_iota(jnp.int32, acc.shape, 0) & 1) == 1
        acc *= jnp.where(
            sel_hi,
            jnp.concatenate(scls[tiles_half:], 1),
            jnp.concatenate(scls[:tiles_half], 1),
        )
        scores_rows = scores_vmem.reshape(bq_sz * num_sublanes_bkv, 128)
        for jj in range(tiles_half):
            scores_rows[pl.ds(jj, rows8, stride=tiles_half)] = pltpu.bitcast(
                acc[:, jj * 128 : (jj + 1) * 128], jnp.int32
            )

    def blocked_prefill_step(
        q_vmem, weights_vmem, bkv_vmem, scores_vmem, q_hm_ref, wb_ref
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

        def reduce_token_block(sts, tb):
            """sum_h relu(st_h) * w_h over the rows of token block `tb`.

            Args:
              sts: per head group, the [rows_per_group, lane_chunk] q @ k block.
              tb: token block index.
            """
            acc = None
            for i in range(heads_per_group):
                # Consecutive heads come from different groups, and so from different
                # MXUs, so the accumulation drains every MXU at the rate it produces
                # results.
                for g, st in enumerate(sts):
                    r0 = tb * rows_per_group_tok_blk + i * tok_blk
                    st_h = st[r0 : r0 + tok_blk]
                    w_h = wb_ref[tb, i * num_head_groups + g]
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
        chunk_scores = []
        for c in range(num_lane_chunks):
            keys, scales = load_chunk(c)
            # Reload the cached rows for every chunk instead of keeping the whole q
            # block live in registers across chunks. Each head group is its own
            # matmul so that the groups can run on different MXUs.
            sts = [
                qk(q_hm_ref[pl.ds(g * rows_per_group, rows_per_group)], keys)
                for g in range(num_head_groups)
            ]
            tb_scores = []
            for tb in range(num_tok_blks):
                acc = reduce_token_block(sts, tb) * scales
                if scores_rows is None:
                    tb_scores.append(acc)
                    continue
                for jj in range(tiles_per_chunk):
                    lane_tile = c * tiles_per_chunk + jj
                    scores_rows[
                        pl.ds(
                            tb * tok_blk * score_pitch + lane_tile,
                            tok_blk,
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

    def _run_all(q_hm_ref=None, wb_ref=None, q2_ref=None, w2_ref=None):
        if prefill_cfg.enabled:
            body = functools.partial(
                blocked_prefill_step, q_hm_ref=q_hm_ref, wb_ref=wb_ref
            )
        elif decode_pack:
            body = functools.partial(
                packed_decode_step,
                q_hm_ref=q_hm_ref,
                q2w_ref=wb_ref,
                q2_ref=q2_ref,
                w2_ref=w2_ref,
            )
        else:
            body = functools.partial(step_body, q_hm_ref=q_hm_ref)

        @pl.with_scoped(final_allocs=(q_alloc, weights_alloc, kv_alloc, o_alloc))
        def _run_pipeline(final_allocs):
            q_buf, weights_buf, kv_buf, o_buf = final_allocs
            pipeline_fn = pltpu.emit_pipeline(
                body,
                grid=(metadata_ref.num_steps[chunk_idx],),
                in_specs=(q_buf.spec, weights_buf.spec, kv_buf.spec),
                out_specs=o_buf.spec,
            )
            window_refs = (cu_q_lens_ref, metadata_ref, start_end_seq_idx_ref)
            pipeline_fn(
                (q_hbm_ref, *window_refs),
                (indexer_weights_hbm_ref, *window_refs),
                (cache_kv_hbm_ref, page_indices_ref, metadata_ref),
                (scores_hbm_ref, *window_refs),
                allocations=(q_buf, weights_buf, kv_buf, o_buf),
            )

        _run_pipeline()

    # Only the multi-token path reorders q, and only a single sequence per tile
    # keeps the cached block unambiguous.
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
        )
    elif decode_pack:
        pl.run_scoped(
            _run_all,
            pltpu.VMEM((num_q_heads * bq_sz, head_dim), q_hbm_ref.dtype),
            pltpu.VMEM((2, 2 * num_q_heads, head_dim), jnp.uint32),
            pltpu.VMEM((8 * num_q_heads, 2 * head_dim), q_hbm_ref.dtype),
            pltpu.VMEM((2 * bq_sz, num_q_heads), jnp.float32),
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
    *,
    raiden_glm_admission: bool = False,
    explicit_mode: bool | None = None,
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
    if mode is None:
        raw_mode = DEFAULT_INDEXER_KV_CACHE_MODE
        is_explicit = bool(explicit_mode)
    else:
        raw_mode = mode
        is_explicit = True if explicit_mode is None else bool(explicit_mode)
    resolved = raw_mode.strip().lower()
    if resolved not in ("xla", "has", "sal"):
        raise ValueError(
            f"Invalid VLLM_TPU_INDEXER_KV_CACHE_MODE={raw_mode!r}; "
            "expected one of ('xla', 'has', 'sal')."
        )
    if resolved == "sal":
        if raiden_glm_admission:
            if is_explicit:
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


_SMALL_EFFECTIVE_ROW_GATHER_MAX_TOKENS = 4


def _effective_global_visible(
    seq_lens: jax.Array,  # i32[max_num_seqs]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    num_tokens: int,
    compression_ratio: int,
) -> tuple[jax.Array, jax.Array]:
    """Return global `(visible, valid)` per token before CP sharding/clamp."""
    max_num_seqs = seq_lens.shape[0]
    num_seqs = distribution[2]
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    # Avoid replacing dynamic gathers on tiny decode buckets (num_tokens <= 4):
    # on 16-TC DCP decode the [4, max_num_seqs] reduction adds per-step latency,
    # whereas b >= 8 and prefill gain from gather-free VPU math.
    if num_tokens <= _SMALL_EFFECTIVE_ROW_GATHER_MAX_TOKENS:
        seq_mask = token_ids[:, None] >= cu_q_lens[None, 1 : max_num_seqs + 1]
        seq_mask = jnp.where(
            jnp.arange(max_num_seqs)[None, :] < num_seqs, seq_mask, False
        )
        seq_ids = jnp.sum(seq_mask, axis=1)
        q_start = cu_q_lens[seq_ids]
        q_len = cu_q_lens[seq_ids + 1] - q_start
        seq_len = seq_lens[seq_ids]
        token_pos = seq_len - q_len + (token_ids - q_start)
        kv_len = seq_len // compression_ratio
        visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
        valid = (token_ids >= cu_q_lens[0]) & (token_ids < cu_q_lens[num_seqs])
        return visible, valid

    seq_idx = jnp.arange(max_num_seqs, dtype=jnp.int32)
    q_starts = cu_q_lens[:max_num_seqs]
    q_ends = cu_q_lens[1 : max_num_seqs + 1]
    # Avoid dynamic 1D gathers on cu_q_lens/seq_lens for b >= 8 and prefill:
    # XLA lowers those as sequential gather fusions, whereas a masked int32
    # reduction over [num_tokens, max_num_seqs] stays on the VPU (~6x faster).
    in_seq = (
        (seq_idx[None, :] < num_seqs)
        & (token_ids[:, None] >= q_starts[None, :])
        & (token_ids[:, None] < q_ends[None, :])
    )
    if compression_ratio == 1:
        pos_off = (seq_lens - q_ends + jnp.int32(1))[None, :]
        cand = jnp.minimum(seq_lens[None, :], token_ids[:, None] + pos_off)
        visible = jnp.sum(jnp.where(in_seq, cand, jnp.int32(0)), axis=1)
        return visible, visible > jnp.int32(0)
    pos_off = seq_lens - q_ends
    off = jnp.sum(jnp.where(in_seq, pos_off[None, :], jnp.int32(0)), axis=1)
    seq_len = jnp.sum(jnp.where(in_seq, seq_lens[None, :], jnp.int32(0)), axis=1)
    token_pos = token_ids + off
    kv_len = seq_len // compression_ratio
    visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
    valid = jnp.any(in_seq, axis=1)
    return visible, valid


def _clamp_local_visible(
    visible: jax.Array,
    valid: jax.Array,
    *,
    num_positions: int,
    compression_ratio: int,
    num_tokens: int,
    cp_rank: jax.Array | int,
    cp_size: int,
    interleave_c: int,
) -> jax.Array:
    local = cp_local_length(visible, cp_rank, cp_size, interleave_c)
    clamped = jnp.minimum(local, jnp.int32(num_positions))
    if compression_ratio == 1 and num_tokens > _SMALL_EFFECTIVE_ROW_GATHER_MAX_TOKENS:
        return clamped
    return jnp.where(valid, clamped, jnp.int32(0))


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
) -> jax.Array:  # i32[num_tokens]
    """Visible compressed KV positions per token row of the score matrix.

    A token sees only the (token_pos + 1) // compression_ratio groups complete
    at or before it. In a multi-token forward (chunked prefill, speculative
    verify) its own group can already be in the cache, completed by a later
    token of the same forward, but that entry covers tokens after it.

    Under context parallelism the score matrix is rank-local, so the row length
    is the number of globally-visible positions that *this* rank owns.
    """
    visible, valid = _effective_global_visible(
        seq_lens, cu_q_lens, distribution, num_tokens, compression_ratio
    )
    return _clamp_local_visible(
        visible,
        valid,
        num_positions=num_positions,
        compression_ratio=compression_ratio,
        num_tokens=num_tokens,
        cp_rank=cp_rank,
        cp_size=cp_size,
        interleave_c=interleave_c,
    )


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
        "skip_empty_decode",
        "skip_empty_decode_chunked",
        "decode_pack",
        "decode_page_major",
        "shared_meta",
        "dcp_axis_name",
        "decode_as_mixed",
        "coop_v2",
        "double_slice_budget",
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
    skip_empty_decode: bool = False,
    skip_empty_decode_chunked: bool = False,
    final_sc_tie: tuple = (),
    decode_pack: bool = False,
    decode_page_major: bool | None = None,
    shared_meta: bool = False,
    dcp_axis_name: str | None = None,
    decode_as_mixed: bool = False,
    coop_v2: bool = False,
    double_slice_budget: bool = False,
) -> jax.Array | tuple[Any, ...]:
    """Retrieve exact top-K KV indices (and optional scores) from FP8 index-K cache."""
    kv_layout = KVLayout.parse(kv_layout)

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
    if dcp_axis_name is not None and (cp_size == 1 or return_scores):
        raise ValueError("dcp_axis_name needs cp_size > 1 and return_scores=False.")
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

    if decode_as_mixed:
        zero = jnp.zeros((), jnp.int32)
        distribution = jnp.stack([zero, zero, distribution[2].astype(jnp.int32)])
    elif uniform_q_len is None:
        q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        one_token = (q_lens == 1) & (jnp.arange(max_num_seqs) < distribution[1])
        num_one_token = jnp.sum(jnp.cumprod(one_token.astype(jnp.int32)))
        distribution = jnp.stack(
            [
                jnp.minimum(distribution[0], num_one_token),
                num_one_token,
                distribution[2],
            ]
        ).astype(jnp.int32)

    original_dtype = q.dtype
    actual_head_dim = q.shape[-1]

    prepared_indexer_weights = prepare_index_weights(indexer_weights, original_dtype)
    q = prepare_q_inputs(q)
    head_dim = q.shape[-1]

    u32_sal_in = (
        kv_layout == KVLayout.SEQ_ALONG_LANE
        and cache_kv.ndim == 3
        and cache_kv.dtype == jnp.uint32
    )
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if u32_sal_in:
            total_num_pages, kv_sublane_groups, page_size = cache_kv.shape
            kv_packing = 4
            expected_u8 = get_kv_cache_shape(
                total_num_pages, page_size, actual_head_dim, kv_layout
            )
            expected_u32 = (expected_u8[0], expected_u8[1], expected_u8[3])
            if cache_kv.shape != expected_u32:
                raise ValueError(
                    "SEQ_ALONG_LANE packed u32 expects cache_kv of shape"
                    f" {expected_u32}, got {cache_kv.shape}."
                )
        else:
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

    def _pack_flags(case, seq_batch_size, bq_sz):
        ok = (
            case == MlaCase.MIXED
            and kv_layout == KVLayout.SEQ_ALONG_LANE
            and seq_batch_size == 1
            and bq_sz == get_dtype_packing(q.dtype)
        )
        page_major = bool(
            (decode_pack if decode_page_major is None else decode_page_major)
            and case == MlaCase.MIXED
            and kv_layout == KVLayout.SEQ_ALONG_LANE
            and seq_batch_size == 1
            and (
                (u32_sal_in and page_size % 128 == 0)
                or bq_sz == get_dtype_packing(q.dtype)
            )
        )
        return ok and decode_pack and 2 * bq_sz == 8, page_major

    def _start_end(start_seq_idx, end_seq_idx, chunk_token_start):
        return jnp.array(
            [start_seq_idx, end_seq_idx, chunk_token_start or 0], jnp.int32
        )

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
        start_end=None,
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
            hbm_spec,  # q
            hbm_spec,  # prepared_indexer_weights
            hbm_spec,  # cache_kv
        ]
        if aliased:
            in_specs.append(hbm_spec)  # scores_init, aliased to out
        out_specs = hbm_spec

        scratch_shapes = []

        if start_end is None:
            start_end = _start_end(start_seq_idx, end_seq_idx, chunk_token_start)
        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            start_end,
            pass_meta,
        )

        num_scalar_prefetch = len(scalar_prefetches)
        num_flat_prefetches = len(jax.tree_util.tree_leaves(scalar_prefetches))
        # The aliased `scores_init` is the 4th operand after the flat prefetches.
        input_output_aliases = {num_flat_prefetches + 3: 0} if aliased else {}

        layout_tag = "" if kv_layout == KVLayout.HEAD_ALONG_SUBLANE else "-sal"
        pack, pack_page_major = _pack_flags(case, seq_batch_size, bq_sz)
        if pack:
            buffer_count = max(buffer_count, _DECODE_PACK_BUFFER_COUNT)
            layout_tag += "-pk"
        if pack_page_major:
            layout_tag += "-pm"
        scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}{layout_tag}"
        if chunked:
            # The chunk count is part of the kernel's identity, so a sweep over chunk
            # sizes shows up in the profile instead of silently reusing one shape.
            scope_name += f"-c{max_num_tokens // chunk_tokens}"
        pallas_kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    kernel,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    seq_batch_size=seq_batch_size,
                    buffer_count=buffer_count,
                    kv_layout=kv_layout,
                    chunk_idx=chunk_idx,
                    decode_pack=pack,
                    decode_page_major=pack_page_major,
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
                out_shape=jax.ShapeDtypeStruct(
                    shape=(chunk_tokens, num_sublanes_total, 128),
                    dtype=out_dtype,
                ),
                input_output_aliases=input_output_aliases,
                name=scope_name,
            )
        )
        scores = pallas_kernel(
            *scalar_prefetches,
            q,
            prepared_indexer_weights,
            cache_kv,
            *((scores_init,) if aliased else ()),
        )
        if scheduling_group_id is not None:
            scores = xla_metadata.set_xla_metadata(
                scores, _scheduling_group_id=scheduling_group_id
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
        return (
            *batched_decode,
            # Handle num_decode_seqs % decode_req_batch_size != 0 case.
            dict(
                start_seq_idx=decode_batch_end,
                end_seq_idx=distribution[0 if uniform else 1],
                seq_batch_size=1,
                **decode_cfg,
            ),
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
        for spec, pass_meta, start_end in passes:
            if callable(pass_meta):
                pass_meta = pass_meta()
            scores = run_scores_kernel(
                q,
                prepared_indexer_weights,
                cache_kv,
                scores,
                seq_lens,
                page_indices,
                cu_q_lens,
                pass_meta=pass_meta,
                chunk_idx=chunk_idx,
                chunk_token_start=chunk_token_start,
                chunk_tokens=chunk_tokens,
                out_dtype=out_dtype,
                scheduling_group_id=scheduling_group_id,
                start_end=start_end,
                **spec,
            )
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
        if max_num_tokens > _SMALL_EFFECTIVE_ROW_GATHER_MAX_TOKENS:
            global_visible, global_valid = _effective_global_visible(
                seq_lens, cu_q_lens, distribution, max_num_tokens, compression_ratio
            )
            eff_row_lengths = _clamp_local_visible(
                global_visible,
                global_valid,
                num_positions=num_positions,
                compression_ratio=compression_ratio,
                num_tokens=max_num_tokens,
                cp_rank=cp_rank,
                cp_size=cp_size,
                interleave_c=interleave_c,
            )
            # Rank 0 owns the first interleave block of every row, so its row
            # lengths bound every rank's, and every rank computes the same (only
            # `dcp_axis_name` reads them).
            row_bounds = _clamp_local_visible(
                global_visible,
                global_valid,
                num_positions=num_positions,
                compression_ratio=compression_ratio,
                num_tokens=max_num_tokens,
                cp_rank=0,
                cp_size=cp_size,
                interleave_c=interleave_c,
            )
        else:
            eff_row_lengths = _effective_row_lengths(
                seq_lens=seq_lens,
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                num_tokens=max_num_tokens,
                num_positions=num_positions,
                compression_ratio=compression_ratio,
                cp_rank=cp_rank,
                cp_size=cp_size,
                interleave_c=interleave_c,
            )
            row_bounds = _effective_row_lengths(
                seq_lens=seq_lens,
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                num_tokens=max_num_tokens,
                num_positions=num_positions,
                compression_ratio=compression_ratio,
                cp_rank=0,
                cp_size=cp_size,
                interleave_c=interleave_c,
            )

        if chunk_tokens is None or max_num_tokens % chunk_tokens:
            num_chunks, chunk_len = 1, max_num_tokens
        else:
            num_chunks, chunk_len = max_num_tokens // chunk_tokens, chunk_tokens

        # Ids count from 1, so the 0 is never a real group: a single chunk has no
        # pair to annotate and takes the `None` branch below both times.
        use_sched_groups = num_chunks > 1 and dcp_axis_name is None
        group_base = (
            config.reserve_scheduling_group_ids(num_chunks - 1)
            if use_sched_groups
            else 0
        )

        chunk_starts = (
            (None,)
            if num_chunks == 1
            else tuple(m * chunk_len for m in range(num_chunks))
        )
        row_bounds = (
            _effective_row_lengths(
                seq_lens=seq_lens,
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                num_tokens=max_num_tokens,
                num_positions=num_positions,
                compression_ratio=compression_ratio,
                cp_rank=0,
                cp_size=cp_size,
                interleave_c=interleave_c,
            )
            if dcp_axis_name is not None
            else eff_row_lengths
        )
        lazy_decode_meta = shared_meta and skip_empty_decode and num_chunks == 1

        def _meta(spec):
            if lazy_decode_meta and spec["case"] != MlaCase.MIXED:
                return functools.partial(_pass_metadata, spec, chunk_starts, chunk_len)
            return _pass_metadata(spec, chunk_starts, chunk_len)

        passes = tuple(
            (spec, _meta(spec))
            for spec in _pass_specs(decode_batch_end)
            if not decode_as_mixed or spec["case"] == MlaCase.MIXED
        )
        topk_idxs: list[jax.Array] = []
        topk_scores_list: list[jax.Array] = []
        tied: tuple[Any, ...] = ()
        for m in range(num_chunks):
            # Group g holds chunk g's SparseCore top-k and chunk g+1's TensorCore
            # scoring, so the scheduler emits `sc(g).start, tc(g+1), sc(g).done`.
            # Chunk 0's scoring and the last top-k have no partner.
            tc_group = group_base + m - 1 if use_sched_groups and m > 0 else None
            sc_group = (
                group_base + m if use_sched_groups and m < num_chunks - 1 else None
            )

            start = m * chunk_len
            chunk_passes = tuple(
                (
                    spec,
                    meta,
                    _start_end(
                        spec["start_seq_idx"], spec["end_seq_idx"], chunk_starts[m]
                    )
                    if shared_meta and not callable(meta)
                    else None,
                )
                for spec, meta in passes
            )
            scores_for = functools.partial(
                _scores_for_chunk,
                chunk_idx=m,
                chunk_token_start=chunk_starts[m],
                chunk_tokens=chunk_len,
                out_dtype=out_dtype,
                scheduling_group_id=tc_group,
            )
            mixed_only = tuple(p for p in chunk_passes if p[0]["case"] == MlaCase.MIXED)
            if decode_as_mixed:
                scores = scores_for(chunk_passes)
            elif (
                skip_empty_decode_chunked
                and num_chunks > 1
                and m * chunk_len >= max_num_seqs
            ):
                scores = scores_for(mixed_only)
            elif (skip_empty_decode and num_chunks == 1) or (
                skip_empty_decode_chunked and num_chunks > 1 and m == 0
            ):
                scores = jax.lax.cond(
                    distribution[1] > 0,
                    lambda f=scores_for, p=chunk_passes: f(p),
                    lambda f=scores_for, p=mixed_only: f(p),
                )
            else:
                scores = scores_for(chunk_passes)
            scores = scores.reshape(chunk_len, -1)
            if scores.shape[1] < k:
                scores = jnp.pad(
                    scores,
                    ((0, 0), (0, k - scores.shape[1])),
                    constant_values=BITS_NEG_INF,
                )
            if final_sc_tie and m == num_chunks - 1:
                _, tied = lax.optimization_barrier(
                    (scores[:1, :128], tuple(final_sc_tie))
                )

            if dcp_axis_name is not None:
                chunk_topk_idxs = sparsecore_topk_dcp(
                    scores,
                    k,
                    eff_row_lengths[start : start + chunk_len],
                    row_bounds[start : start + chunk_len],
                    dcp_size=cp_size,
                    axis_name=dcp_axis_name,
                    scheduling_group_id=sc_group,
                    after=topk_idxs[-1] if topk_idxs else None,
                )
            elif return_scores:
                chunk_topk_idxs, chunk_topk_scores = sparsecore_topk(
                    scores,
                    k,
                    row_lengths=eff_row_lengths[start : start + chunk_len],
                    write_empty_rows=True,
                    scheduling_group_id=sc_group,
                    return_scores=True,
                    coop_v2=coop_v2,
                    double_slice_budget=double_slice_budget,
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
                    coop_v2=coop_v2,
                    double_slice_budget=double_slice_budget,
                )
            topk_idxs.append(chunk_topk_idxs)

        out = topk_idxs[0] if len(topk_idxs) == 1 else jnp.concatenate(topk_idxs)
        out = out[:max_num_tokens, :k]

        extra = (tied,) if final_sc_tie else ()
        if not return_scores:
            return (out, *extra) if extra else out

        scores_out = (
            topk_scores_list[0]
            if len(topk_scores_list) == 1
            else jnp.concatenate(topk_scores_list)
        )
        scores_out = scores_out[:max_num_tokens, :k]
        return (out, scores_out, *extra)

    def _fast_path(_):
        token_idx = jnp.arange(q.shape[0])
        # Avoid default binary-search searchsorted: lowers to a gather
        # while-loop on TPU (~640 us vs ~10 us at 1024 tokens).
        seq_idx = jnp.minimum(
            jnp.searchsorted(
                cu_q_lens[1:], token_idx, side="right", method="compare_all"
            ),
            seq_lens.shape[0] - 1,
        )
        seq_len = seq_lens[seq_idx]
        q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
        q_start = cu_q_lens[seq_idx]
        q_abs_pos = (seq_len - q_len) + (token_idx - q_start)
        max_valid_idx = jnp.minimum(
            seq_len // compression_ratio - 1,
            q_abs_pos // compression_ratio,
        )
        valid_tok = (token_idx >= cu_q_lens[0]) & (token_idx < cu_q_lens[-1])
        max_valid_idx = jnp.where(valid_tok, max_valid_idx, -1)
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
        "hist_merge",
        "skip_empty_decode",
        "skip_empty_decode_chunked",
        "overlap_fn",
        "overlap_pin_max_tokens",
        "overlap_rank",
        "flat_merge",
        "small_all_gather",
        "decode_pack",
        "shared_meta",
        "decode_as_mixed",
        "coop_v2",
        "double_slice_budget",
        "coop_merge",
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
    mesh: jax.sharding.Mesh,
    k: int,
    compression_ratio: int,
    dcp_size: int,
    interleave_size: int = 1,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    buffer_count: tuple[int, int, int] | int | None = DEFAULT_BUFFER_COUNT,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    kv_layout: KVLayout | str = KVLayout.HEAD_ALONG_SUBLANE,
    chunk_tokens: int | None = None,
    dcp_axis_name: str = DCP_AXIS_NAME,
    a2a_merge: bool = False,
    hist_merge: bool = False,
    skip_empty_decode: bool = False,
    skip_empty_decode_chunked: bool = False,
    dcp_rank_ids: jax.Array | None = None,
    dcp_lane_ids: jax.Array | None = None,
    overlap_fn: Any = None,
    overlap_args: tuple = (),
    overlap_pin_max_tokens: int | None = None,
    overlap_rank: bool = False,
    flat_merge: bool = False,
    small_all_gather: bool = False,
    decode_pack: bool = False,
    shared_meta: bool = False,
    decode_as_mixed: bool = False,
    coop_v2: bool = False,
    double_slice_budget: bool = False,
    coop_merge: bool = False,
) -> jax.Array | tuple[jax.Array, Any]:
    """Global top-k over an interleaved DCP indexer cache in one shard_map."""
    if dcp_size < 2:
        raise ValueError(f"streamindex_topk_dcp requires dcp_size >= 2, got {dcp_size}")
    if mesh.shape.get(dcp_axis_name) != dcp_size:
        raise ValueError(
            f"mesh axis {dcp_axis_name!r} has size {mesh.shape.get(dcp_axis_name)}, "
            f"expected dcp_size={dcp_size}"
        )
    if page_indices.ndim == 2:
        page_indices = page_indices.reshape(-1)
    if a2a_merge and hist_merge:
        raise ValueError("a2a_merge and hist_merge are mutually exclusive.")
    if a2a_merge and q.shape[0] % dcp_size != 0:
        raise ValueError(
            f"padded_num_tokens={q.shape[0]} must be divisible by dcp_size={dcp_size}."
        )
    flat_merge = (
        bool(flat_merge) and not a2a_merge and rank_major_merge_fits(q.shape[0])
    )
    small_all_gather = (
        bool(small_all_gather)
        and not a2a_merge
        and flat_merge
        and q.shape[0] <= SMALL_AG_MAX_TOKENS
    )
    lane_axes = tuple(ax for ax in mesh.axis_names if ax != dcp_axis_name)
    if small_all_gather and len(lane_axes) > 1:
        raise ValueError(
            f"small_all_gather takes at most one axis beside {dcp_axis_name!r}, "
            f"got {mesh.axis_names}"
        )
    if overlap_fn is not None and not overlap_args:
        raise ValueError("overlap_fn requires at least one overlap_args array.")
    pin = (
        overlap_fn is not None
        and (overlap_pin_max_tokens is None or q.shape[0] <= overlap_pin_max_tokens)
        and (
            chunk_tokens is None
            or q.shape[0] % chunk_tokens != 0
            or q.shape[0] == chunk_tokens
        )
    )

    def _local(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        *extra,
    ):
        n_rank, n_lane = int(dcp_rank_ids is not None), int(dcp_lane_ids is not None)
        rank_ids, lane_ids = extra[:n_rank], extra[n_rank : n_rank + n_lane]
        ov_args = extra[n_rank + n_lane :]
        dcp_rank = (
            rank_ids[0][0] if rank_ids else cp_rank_as_data(dcp_axis_name, dcp_size)
        )
        lane_axis = lane_axes[0] if lane_axes else None
        lane = (
            lane_ids[0]
            if (lane_axis is not None and lane_ids)
            else jnp.zeros((1,), jnp.int32)
            if (lane_axis is not None and mesh.shape[lane_axis] == 1)
            else None
        )
        if lane_ids:
            dcp_rank = dcp_rank + jnp.int32(0) * lane_ids[0][0]
        local_page_indices = jnp.mod(page_indices, jnp.int32(cache_kv.shape[0]))

        local = streamindex_topk(
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
            return_scores=not hist_merge,
            skip_empty_decode=skip_empty_decode,
            skip_empty_decode_chunked=skip_empty_decode_chunked,
            final_sc_tie=tuple(ov_args[:1]) if pin else (),
            decode_pack=decode_pack,
            shared_meta=shared_meta,
            dcp_axis_name=dcp_axis_name if hist_merge else None,
            decode_as_mixed=decode_as_mixed,
            coop_v2=coop_v2,
            double_slice_budget=double_slice_budget,
        )
        scores = None
        if pin:
            idxs, *rest, tied = local
            scores = rest[0] if rest else None
            ov_in = (*tied, *ov_args[1:])
        else:
            idxs, scores = (local, None) if hist_merge else local
            ov_in = tuple(ov_args)
        if flat_merge and scores is not None:
            scores = scores.reshape(-1)

        ov = None
        if overlap_fn is not None:
            if overlap_rank:
                if lane_axis is not None and lane is None:
                    raise ValueError(
                        "overlap_rank under lane axis > 1 needs dcp_lane_ids"
                    )
                ov = overlap_fn(
                    *ov_in, jnp.reshape(dcp_rank, (1,)).astype(jnp.int32), lane
                )
            else:
                ov = overlap_fn(*ov_in)
            if pin and not hist_merge:
                scores, ov = lax.optimization_barrier((scores, ov))

        def _ret(topk):
            return topk if overlap_fn is None else (topk, ov)

        if hist_merge:
            return _ret(idxs)

        assert scores is not None
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
                coop_v2=coop_v2,
                double_slice_budget=double_slice_budget,
            )
            assert isinstance(slots, jax.Array)
            slots = lax.all_gather(slots, dcp_axis_name, axis=0, tiled=True)
            return _ret(dcp_sc_compact.pack_owned_winners(slots, idxs, dcp_rank))

        if small_all_gather:
            scores = _small_all_gather(
                scores,
                num_cores=dcp_size,
                axis=dcp_axis_name,
                rank=jnp.reshape(dcp_rank, (1,)),
                lane_axis=lane_axis,
                lane=lane,
                hbm_operand=True,
            )
        else:
            scores = lax.all_gather(
                scores, dcp_axis_name, axis=0 if flat_merge else 1, tiled=True
            )

        return _ret(
            sparsecore_topk_dcp_merge(
                scores,
                idxs,
                k,
                dcp_rank=dcp_rank,
                dcp_size=dcp_size,
                rank_major=flat_merge,
                coop_merge=coop_merge,
            )
        )

    replicated = P()
    ov_leaves = tuple(overlap_args) if overlap_fn is not None else ()
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            replicated,
            replicated,
            P(dcp_axis_name),
            replicated,
            replicated,
            replicated,
            replicated,
            *((P(dcp_axis_name),) if dcp_rank_ids is not None else ()),
            *((P(dcp_axis_name),) if dcp_lane_ids is not None else ()),
            *(replicated for _ in ov_leaves),
        ),
        out_specs=(
            P(dcp_axis_name) if overlap_fn is None else (P(dcp_axis_name), replicated)
        ),
        check_vma=False,
    )(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        *((dcp_rank_ids,) if dcp_rank_ids is not None else ()),
        *((dcp_lane_ids,) if dcp_lane_ids is not None else ()),
        *ov_leaves,
    )
