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
):
    """Executes the n-buffered Pallas scoring pipeline for StreamIndex Top-K."""
    scores_hbm_ref = score_refs[-1]
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
                bkv_p,
                kv_sublane_groups,
                kv_packing,
                page_size,
            ),
            lambda p_id: (seq_tile(p_id), bkv_block(p_id), 0, 0, 0),
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
            compression_ratio=compression_ratio,
            cp_size=cp_size,
            interleave_c=interleave_c,
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
    # The VMEM block is u8[seq_batch, bkv_p, kv_sublane_groups, kv_packing,
    # page_size]: each page keeps its own HBM layout, so it lands with one
    # contiguous DMA. Packed group g holds key dims [4g, 4g + 4) of every token,
    # one byte per dim, and group `kv_head_dim_groups` holds the UE8M0 scale in
    # byte 0. Viewed as u32, row `(page * kv_sublane_groups + g) * nt + j` (nt =
    # page_size // 128) is group g of lane tile j of that page, so a
    # sublane-strided load of the page's rows `j, j + nt, ...` yields the
    # [head_dim // 4, 128] words of lane tile j. Bitcasting those words to FP8
    # gives [head_dim, 128] with key dim 4g + i taken from byte i -- exactly the
    # MXU-native packed layout -- so no relayout is needed on the key path.
    # ------------------------------------------------------------------------
    num_lane_tiles_page = page_size // 128

    def seq_along_lane_block_rows(bkv_vmem):
        assert kv_sublane_groups is not None
        return bkv_vmem.bitcast(jnp.uint32).reshape(
            seq_batch_size, bkv_p * kv_sublane_groups * num_lane_tiles_page, 128
        )

    def seq_along_lane_row(lane_tile_idx, group):
        """u32 row holding packed group `group` of block lane tile `lane_tile_idx`."""
        assert kv_sublane_groups is not None
        page, tile = divmod(lane_tile_idx, num_lane_tiles_page)
        return (page * kv_sublane_groups + group) * num_lane_tiles_page + tile

    def seq_along_lane_key_tile(rows, batch_idx, lane_tile_idx):
        """[head_dim, 128] FP8 keys of lane tile `lane_tile_idx`."""
        words = rows[
            batch_idx,
            pl.ds(
                seq_along_lane_row(lane_tile_idx, 0),
                kv_head_dim_groups,
                stride=num_lane_tiles_page,
            ),
            :,
        ]
        return pltpu.bitcast(words, jnp.float8_e4m3fn)

    def seq_along_lane_scale_tile(rows, batch_idx, lane_tile_idx):
        """[1, 128] f32 dequantization scales of lane tile `lane_tile_idx`."""
        row = seq_along_lane_row(lane_tile_idx, kv_head_dim_groups)
        words = rows[batch_idx, pl.ds(row, 1), :]
        # UE8M0 holds a bare biased exponent, which is where f32 keeps its own:
        # moving byte 0 into the f32 exponent field yields 2^(e - 127) exactly
        # for every code the quantizer writes. The codes it never writes, 0 and
        # 0xFF, decode to 0 and +inf instead of e8m0's 2^-127 and NaN; selecting
        # those out is ~4% of SAL decode at batch 32.
        return pltpu.bitcast((words & 0xFF) << 23, jnp.float32)

    def load_key_chunk(bkv_vmem, batch_idx, chunk_idx, lane_chunk):
        """Keys and [1, lane_chunk] f32 scales of one lane chunk of a sequence.

        The keys are [head_dim, lane_chunk] for SEQ_ALONG_LANE and
        [lane_chunk, head_dim] for HEAD_ALONG_SUBLANE; `qk` takes either.
        """
        if seq_along_lane:
            rows = seq_along_lane_block_rows(bkv_vmem)
            tiles_per_chunk = lane_chunk // 128
            tiles = range(
                chunk_idx * tiles_per_chunk, (chunk_idx + 1) * tiles_per_chunk
            )
            keys = jnp.concatenate(
                [seq_along_lane_key_tile(rows, batch_idx, t) for t in tiles], axis=1
            )
            scales = jnp.concatenate(
                [seq_along_lane_scale_tile(rows, batch_idx, t) for t in tiles], axis=1
            )
            return keys, scales
        groups_per_chunk = lane_chunk // kv_packing
        blk = bkv_vmem[batch_idx, pl.ds(chunk_idx * groups_per_chunk, groups_per_chunk)]
        flat = blk.reshape(lane_chunk, blk.shape[-1])
        keys = pltpu.bitcast(flat[:, :head_dim], jnp.float8_e4m3fn)
        scales = pltpu.bitcast(
            flat[:, head_dim : head_dim + 1].T, jnp.float8_e8m0fnu
        ).astype(jnp.float32)
        return keys, scales

    def qk(q_rows, keys):
        """[rows, lane_chunk] f32 q @ k for the keys of `load_key_chunk`."""
        if seq_along_lane:
            # keys is [head_dim, lane_chunk]: the MXU-native contraction layout.
            return jnp.einsum(
                "nd,dm->nm", q_rows, keys, preferred_element_type=jnp.float32
            )
        return jnp.einsum("nd,md->nm", q_rows, keys, preferred_element_type=jnp.float32)

    def transpose_to_head_major(bq_rows, n_tok):
        """[n_tok*H, D] head-minor rows -> [H*n_tok, D] head-major rows."""
        return bq_rows.transpose(1, 0, 2).reshape(num_q_heads * n_tok, head_dim)

    def step_body(q_vmem, weights_vmem, bkv_vmem, scores_vmem):
        """Scores one step of the decode and small-q-block paths.

        Each sequence's [bq * H, D] queries keep their token-major order, so the
        head reduction adds whole vregs, and the keys are taken one lane chunk at
        a time to bound the live [bq * H, lane_chunk] product. Positions a token
        cannot see are left unmasked; see `blocked_prefill_step`.
        """
        rows_per_seq = bq_sz * num_q_heads
        lane_chunk = math.gcd(bkv_sz, max(128, min(1024, 32768 // rows_per_seq)))
        tiles_per_chunk = lane_chunk // 128
        q = q_vmem[...].reshape(seq_batch_size, rows_per_seq, head_dim)
        w = (
            weights_vmem[...]
            .astype(jnp.float32)
            .reshape(seq_batch_size, rows_per_seq, 1)
        )
        for batch_idx in range(seq_batch_size):
            for c in range(bkv_sz // lane_chunk):
                keys, scales = load_key_chunk(bkv_vmem, batch_idx, c, lane_chunk)
                st = jnp.maximum(qk(q[batch_idx], keys), 0.0) * w[batch_idx]
                s = st.reshape(bq_sz, num_q_heads, lane_chunk).sum(axis=1) * scales
                scores_vmem[
                    pl.ds(batch_idx * bq_sz, bq_sz),
                    pl.ds(c * tiles_per_chunk, tiles_per_chunk),
                ] = pltpu.bitcast(s.reshape(bq_sz, tiles_per_chunk, 128), jnp.int32)

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
            keys, scales = load_key_chunk(bkv_vmem, 0, c, lane_chunk)
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

    def _run_all(q_hm_ref=None, wb_ref=None):
        if prefill_cfg.enabled:
            body = functools.partial(
                blocked_prefill_step, q_hm_ref=q_hm_ref, wb_ref=wb_ref
            )
        else:
            body = step_body

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
                (
                    cache_kv_hbm_ref,
                    page_indices_ref,
                    metadata_ref,
                    seq_lens_ref,
                    cp_rank_ref,
                ),
                (scores_hbm_ref, *window_refs),
                allocations=(q_buf, weights_buf, kv_buf, o_buf),
            )

        _run_pipeline()

    # Only the blocked multi-token path caches a reordered q block.
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

    Under context parallelism the score matrix is rank-local, so the row length
    is the number of globally-visible positions that *this* rank owns.
    """
    max_num_seqs = seq_lens.shape[0]
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
    q_ends = cu_q_lens[None, 1 : max_num_seqs + 1]
    # One-hot [num_tokens, max_num_seqs] owner of each token. Summing fields over
    # it, instead of gathering them by sequence id, lets XLA fuse the whole
    # computation into a few kernels, which matters at decode latency.
    owner = (
        (token_ids >= cu_q_lens[None, :max_num_seqs])
        & (token_ids < q_ends)
        & (jnp.arange(max_num_seqs)[None, :] < distribution[2])
    )
    q_end = jnp.sum(jnp.where(owner, q_ends, 0), axis=1)
    seq_len = jnp.sum(jnp.where(owner, seq_lens[None, :], 0), axis=1)
    token_pos = seq_len - (q_end - token_ids[:, 0])
    kv_len = seq_len // compression_ratio
    visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
    visible = cp_local_length(visible, cp_rank, cp_size, interleave_c)
    return jnp.where(jnp.any(owner, axis=1), jnp.minimum(visible, num_positions), 0)


def _short_prefill_end(
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    max_q_len: int,
) -> jax.Array:  # i32[]
    """End of the leading prefill-only sequences with at most `max_q_len` tokens.

    These are scored in their own pass with a `max_q_len` q block. Any prefill-only
    sequence from the first longer one on goes to the mixed pass, which handles any
    length, so no caller-provided `distribution` can leave tokens unscored.
    """
    max_num_seqs = cu_q_lens.shape[0] - 1
    seq_ids = jnp.arange(max_num_seqs, dtype=jnp.int32)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    # The first sequence after the decodes that is too long for the pass, or
    # distribution[1] if that comes first.
    too_long = (seq_ids >= distribution[0]) & (q_lens > max_q_len)
    return jnp.min(jnp.where(too_long, seq_ids, distribution[1]))


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
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """StreamIndex Top-K retrieval.

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
      distribution: (i, j, k) sequence boundaries. sequences[0:i] are decodes
        and must have exactly one query token each; k is the total number of
        sequences. sequences[i:k] are split between two passes. The prefill pass
        takes the run of sequences starting at i that each have at most
        `num_queries_per_block[1]` tokens, such as speculative-decoding verify
        windows, and stops at the first longer sequence or at j, whichever comes
        first. The mixed pass takes the rest, of any length, so a short sequence
        after a long one goes there too. `(i, k, k)` lets the short sequences
        right after the decodes skip the mixed pass's large q block, and
        `(i, i, k)` sends everything after the decodes to the mixed pass. When
        the prefill and mixed passes have the same block sizes and buffer count,
        the prefill pass is skipped and j is ignored.
      k: Number of top-K elements to retrieve.
      compression_ratio: KV cache compression ratio.
      num_kv_pages_per_block: number of kv pages to be processed in one block in
        the pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
      num_queries_per_block: number of queries to be processed in one block in the
        pallas kernel. This is a tuple of (decode, prefill, mixed) cases. The
        prefill entry is also the longest prefill-only sequence that gets its own
        pass; see `distribution`.
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
        # Every pass after the first donates the previous pass's HBM buffer so its
        # rows survive into this output. Only the first has no donor, so Pallas
        # allocates it there; one buffer per chunk either way.
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

        chunk_start = 0 if chunk_token_start is None else chunk_token_start
        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx, chunk_start], jnp.int32),
            pass_meta,
            jnp.asarray([cp_rank], jnp.int32),
        )

        num_scalar_prefetch = len(scalar_prefetches)
        num_flat_prefetches = len(jax.tree_util.tree_leaves(scalar_prefetches))
        # The aliased `scores_init` is the 4th operand after the flat prefetches.
        input_output_aliases = {num_flat_prefetches + 3: 0} if aliased else {}

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

    def _case_blocks(case):
        """The block-size kwargs of `case`'s pass, from the per-case tuples."""
        return dict(
            num_kv_pages_per_block=num_kv_pages_per_blocks[case.value],
            num_queries_per_block=num_queries_per_blocks[case.value],
            buffer_count=buffer_counts[case.value],
        )

    # Short prefill-only sequences, e.g. speculative-decoding verify windows, get
    # a pass of their own in which each fits one q block: a 4-token window in the
    # mixed pass's 128-row q block would spend 97% of its work on padding. With
    # the same block sizes as the mixed pass, that pass would only add an empty
    # kernel launch and its schedule (8-15us) to every call, so it is skipped.
    use_short_prefill_pass = _case_blocks(MlaCase.PREFILL) != _case_blocks(
        MlaCase.MIXED
    )
    max_short_q_len = num_queries_per_blocks[MlaCase.PREFILL.value]

    def _pass_specs(decode_batch_end, short_end):
        """`run_scores_kernel` kwargs for the passes over the batch, in order."""

        def spec(case, start_seq_idx, end_seq_idx, static_q_len, seq_batch_size=1):
            return dict(
                case=case,
                start_seq_idx=start_seq_idx,
                end_seq_idx=end_seq_idx,
                static_q_len=static_q_len,
                seq_batch_size=seq_batch_size,
                **_case_blocks(case),
            )

        specs = []
        if max_num_seqs >= decode_req_batch_size:
            specs.append(
                spec(
                    MlaCase.DECODE,
                    jnp.array(0),
                    decode_batch_end,
                    static_q_len=1,
                    seq_batch_size=decode_req_batch_size,
                )
            )
        # The decodes past the last full group of `decode_req_batch_size`.
        specs.append(
            spec(MlaCase.DECODE, decode_batch_end, distribution[0], static_q_len=1)
        )
        if use_short_prefill_pass:
            # `static_q_len` is the one q block, which bounds the schedule by the
            # number of sequences instead of tokens.
            specs.append(
                spec(
                    MlaCase.PREFILL,
                    distribution[0],
                    short_end,
                    static_q_len=max_short_q_len,
                )
            )
        specs.append(spec(MlaCase.MIXED, short_end, distribution[2], static_q_len=None))
        return specs

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
        for spec, pass_meta in passes:
            # A pass with a `static_q_len` holds sequences of at most that many
            # tokens, behind only sequences no longer, so its tokens all fall in
            # the first `max_num_seqs * static_q_len`. A chunk past them would
            # launch it empty, which costs 2-9us a pass, so it is not launched.
            if (
                chunk_token_start is not None
                and spec["static_q_len"] is not None
                and chunk_token_start >= max_num_seqs * spec["static_q_len"]
            ):
                continue
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

        short_end = (
            _short_prefill_end(cu_q_lens, distribution, max_q_len=max_short_q_len)
            if use_short_prefill_pass
            else distribution[0]
        )

        num_positions = max(num_sublanes_total * 128, k)
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

        if chunk_tokens is None or max_num_tokens % chunk_tokens:
            num_chunks, chunk_len = 1, max_num_tokens
        else:
            num_chunks, chunk_len = max_num_tokens // chunk_tokens, chunk_tokens

        # Ids count from 1, so the 0 is never a real group: a single chunk has no
        # pair to annotate and takes the `None` branch below both times.
        group_base = 0
        if num_chunks > 1:
            group_base = config.reserve_scheduling_group_ids(num_chunks - 1)

        chunk_starts = (
            (None,)
            if num_chunks == 1
            else tuple(m * chunk_len for m in range(num_chunks))
        )
        pass_specs = _pass_specs(decode_batch_end, short_end)

        def with_schedules(specs):
            return tuple(
                (spec, _pass_metadata(spec, chunk_starts, chunk_len)) for spec in specs
            )

        # During speculative decoding every sequence is usually a short verify
        # window, which leaves the prefill pass as the only one with work. A
        # single chunk then branches around the other passes, skipping their
        # schedules and launches, which dominate a small batch's latency.
        branch_on_short_only = num_chunks == 1 and use_short_prefill_pass
        passes = () if branch_on_short_only else with_schedules(pass_specs)

        def score_chunk(m, scheduling_group_id):
            def score(scheduled):
                return _scores_for_chunk(
                    scheduled,
                    m,
                    chunk_starts[m],
                    chunk_len,
                    out_dtype,
                    scheduling_group_id,
                )

            if not branch_on_short_only:
                return score(passes)
            # Each branch builds its own schedules, so the short-only one skips
            # the others'.
            short_specs = [s for s in pass_specs if s["case"] == MlaCase.PREFILL]
            return lax.cond(
                (distribution[0] == 0) & (short_end == distribution[2]),
                lambda: score(with_schedules(short_specs)),
                lambda: score(with_schedules(pass_specs)),
            )

        topk_idxs = []
        topk_scores_list = []
        for m in range(num_chunks):
            # Group g holds chunk g's SparseCore top-k and chunk g+1's TensorCore
            # scoring, so the scheduler emits `sc(g).start, tc(g+1), sc(g).done`.
            # Chunk 0's scoring and the last top-k have no partner.
            tc_group = None if m == 0 else group_base + m - 1
            sc_group = None if m == num_chunks - 1 else group_base + m

            start = m * chunk_len
            scores = score_chunk(m, tc_group)
            scores = scores.reshape(chunk_len, -1)
            if scores.shape[1] < k:
                scores = jnp.pad(
                    scores,
                    ((0, 0), (0, k - scores.shape[1])),
                    constant_values=BITS_NEG_INF,
                )

            if return_scores:
                chunk_topk_idxs, chunk_topk_scores = sparsecore_topk(
                    scores,
                    k,
                    row_lengths=eff_row_lengths[start : start + chunk_len],
                    write_empty_rows=True,
                    scheduling_group_id=sc_group,
                    return_scores=True,
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
                )
            topk_idxs.append(chunk_topk_idxs)

        out = topk_idxs[0] if len(topk_idxs) == 1 else jnp.concatenate(topk_idxs)
        out = out[:max_num_tokens, :k]

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
            q_abs_pos // compression_ratio,
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
