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
"""The layer's build-time settings, as one record with its constraints.

Nothing inside the package reads the environment. A caller constructs a
Config directly (the serving adapter builds one from its own settings).
"""
import dataclasses

import jax.numpy as jnp

from .device import VMEM_FRACTION
from .layout import LANES, ROW_BLOCK, SUBLANES

# How an expert's result rows travel to the token's shard: fp8 rows with one
# scale per row (half the bytes), or bf16 rows.
RESULT_ROWS = ("fp8", "bf16")
TOKEN_ROWS = ("fp8", "bf16")
INTERMEDIATES = ("fp8", "bf16")
# The element type the fp8 scaling products are formed in (the row
# quantizer, the intermediate, the result epilogue); the rest of the vector
# math runs on the matmuls' float32 accumulators.
ELEMENTWISE_DTYPES = ("float32", "bfloat16")
ROW_SCALE_DTYPES = (jnp.float32, jnp.bfloat16)
# Result buffer slots and the row-fetch lookahead.
RESULT_SLOTS_DEFAULT = 3
ROW_FETCH_AHEAD_DEFAULT = 2
# Weight buffer slots, and the experts ahead of compute a refill is issued.
WEIGHT_SLOTS_DEFAULT = 3
WEIGHT_PREFETCH_DEFAULT = 2
# Routing blocks of 128 pairs the two tables kernels handle per grid step:
# up to 8 blocks, 1024 pairs, a tuning choice (fewer, larger steps against
# the step's VMEM). A step reads whole tiles of the routing message (1024
# pairs), so smaller counts do not lower the token padding; the layer
# takes the largest count that pads no further than the smallest does.
TABLES_BLOCKS_PER_STEP_MAX = 8
TABLES_BLOCKS_PER_STEP_AUTOMATIC = 0
STREAM_BLOCK_AUTOMATIC = 0
STREAM_ROWS_DEFAULT = 1024
# The share of the chip's VMEM one kernel may plan for (device.py's).
VMEM_FRACTION_DEFAULT = VMEM_FRACTION
# The programs' geometry, each a field so a sweep can set it per shape.
# Rows per index-driven DMA of the sparse-core gather behind the combine,
# and its DMAs in flight per subcore.
GATHER_BLOCK_ROWS_DEFAULT = 16
GATHER_RING_DEFAULT = 6
# The automatic row fetch (Config.sorted_rows = None): the routed rows are
# sorted into expert order when a shard holds at least this many experts or
# at most this many tokens (sorted_rows_by_shape).
SORTED_ROWS_EXPERTS_PER_SHARD = 32
SORTED_ROWS_TOKENS_PER_SHARD = 256
# The most routed pairs one chunk of the sparse-core row-tables scatter
# carries, and its chunks in flight per subcore.
SCATTER_CHUNK_PAIRS_DEFAULT = 640
SCATTER_RING_DEFAULT = 4
# Tokens per grid step of the combine's weighted sum, and its token loop's
# unroll.
COMBINE_TOKENS_PER_TILE_DEFAULT = 128
COMBINE_UNROLL_DEFAULT = 32
# Columns of the intermediate the FFN kernel activates and requantizes per
# pass, so the live bf16 intermediate stays small.
ACTIVATION_COLUMNS_PER_PASS_DEFAULT = 1024
# The fewest words of a row table the shard-tables kernel zeroes per grid
# step.
TABLES_ZERO_WORDS_DEFAULT = 8192
# The DMA queue the expert weight refills issue on: 1 is the queue off the
# in-order one the token row fetches use, 0 the same queue.
WEIGHT_DMA_PRIORITY_DEFAULT = 1
WEIGHT_DMA_PRIORITIES = (0, 1)
# Settings tuned per shape, keyed by (hidden, inter, experts per shard,
# top_k): the fields tuned off the defaults for that shape and nothing else.
# The layer consults it when the caller passes no Config; a caller's own
# Config is taken as it is.
TUNED_BY_SHAPE = {}


@dataclasses.dataclass(frozen=True)
class Config:
    """The layer's build-time settings.

    Attributes:
      result_slots: buffers a tile's result rows are staged in. A tile computes
        into one while earlier tiles' commits drain out of the others.
      row_fetch_ahead: tiles ahead of compute a tile's rows are fetched,
        into the slot its results will be staged in. Smaller than
        result_slots so that slot is free.
      weight_slots: the most buffers an expert's weight matrices are
        streamed into. The layer takes the largest count up to it whose VMEM
        estimate fits the chip's budget, so a wide expert runs at two.
      weight_prefetch: experts ahead of compute a weight refill is issued.
        Smaller than weight_slots so the refill's slot is not a live one.
      tables_blocks_per_step: routing blocks the tables kernels handle per
        grid step, each 128 pairs. The pallas mode pads a shard's tokens so
        its pairs are a whole number of steps, so 0 (the default) lets the
        layer take, per call, the largest count up to
        TABLES_BLOCKS_PER_STEP_MAX that pads no further than the gather's
        own unit does.
      stream_block: columns of the intermediate per streamed weight block.
        0 chooses for the shape: whole-expert weight slots when they fit
        the VMEM budget, else the widest block that fits. A count forces
        the streamed build: the weight slots hold column blocks of one
        expert, a group of stream_rows token rows keeps its float32
        running sum in VMEM, and the intermediate is rounded per
        activation block, which has to divide the stream block.
      stream_rows: token rows per group of a streamed expert (the rows
        whose running sum is resident), a multiple of the tile height.
      vmem_fraction: the share of the chip's VMEM the kernels plan for
        (the weight-slot fit, the streaming choice, each program's limit).
      activation_block: how each matmul's input is rounded to fp8: 0 is
        one scale per token row over the whole width; a multiple of 128
        that divides hidden and inter is one scale per that many values
        of a row, for the token rows and the intermediate alike. With
        four-bit weights the block has
        to be the weight block.
      token_block: how the token rows are rounded to fp8 on their way into
        the up projection, when it differs from activation_block: None
        (the default) follows activation_block, except that a streamed
        expert (fp8 or bf16 weights) takes 0, the layer's choice; 0 is one
        scale per row
        over the whole width; a multiple of 128 dividing hidden is one
        scale per that many values. With activation_block set, a streamed
        expert rounds its intermediate per block (it has to), while the
        token rows need no blocked scale: token_block 0 leaves the up
        projection one plain matmul with a single rescale, where a blocked
        contraction rescales its accumulator once per block on the vector
        unit. With four-bit weights a nonzero token block has to be the
        weight block.
      intermediate: the rows between the two matmuls: "fp8" requantizes
        the activated rows to fp8 with one scale per row (or per
        activation block) and runs the down matmul in fp8; "bf16" keeps
        them in bfloat16, no requantization, and runs the down matmul in
        bfloat16 against a per-expert bfloat16 copy of w2 (whole-expert
        builds with fp8 or bf16 weights).
      result_rows: "fp8" sends each expert's result rows between shards as
        fp8 with one scale per row; "bf16" sends bf16 rows, twice the
        bytes and no third rounding.
      token_rows: "fp8" quantizes the token rows to fp8 with one scale per
        row (or per token block) before the dispatch, the largest of the
        kernel's roundings; "bf16" dispatches them as they are, twice the
        bytes and no rounding, and the up matmul runs in bf16: as the
        weights are with bf16 weights, or on a per-expert bf16 copy of w1
        with fp8 weights (whole-expert builds; not the block-scaled
        forms).
      elementwise_dtype: the element type the fp8 scaling products are
        formed in (the row quantizer, the intermediate, the result
        epilogue). "bfloat16", the default, is a packed bf16 multiply that
        leaves a second rounding to the compiler, so two compilations can
        differ by an fp8 step on a few of the values, and is the faster
        form. "float32" rounds each product once, so every
        compilation, the eager reference and the Mosaic kernels agree to
        the bit. The scales, bias and activation run on the float32
        accumulators either way.
      combine_dtype: the element type of the weighted sum over a token's
        selected experts at the end of the layer: "float32", or "bfloat16"
        for the vector unit's packed rate at one bf16 rounding per term.
      row_scale_dtype: rounding of each row's quantization scale, float32
        or bfloat16. Whether bfloat16 is acceptable is a model-quality
        question, decided by evaluation.
      bounds_checks: a diagnostic build with Mosaic's bounds checks on every
        dynamic index. Slower, changes no value.
      gather_block_rows: rows per index-driven DMA of the sparse-core
        gather behind the combine, a whole number of 8-row blocks.
      gather_ring: the gather's DMAs in flight per subcore; the layer
        halves it while the ring's staging (block rows by row bytes by
        ring) exceeds the default's at 4096 lanes.
      scatter_chunk_pairs: the most routed pairs per chunk of the
        sparse-core row-tables scatter, a multiple of 128; the layer takes
        the largest count up to it that divides a subcore's run.
      scatter_ring: the scatter's chunks in flight per subcore.
      combine_tokens_per_tile: tokens per grid step of the combine's
        weighted sum, a whole number of 8-row blocks; the routing message
        pads the tokens to a multiple of it.
      combine_unroll: the weighted sum's token loop unroll, dividing
        combine_tokens_per_tile.
      activation_columns_per_pass: columns of the intermediate the FFN
        kernel activates and requantizes per pass, a multiple of 128.
      tables_zero_words: the fewest words of a row table the shard-tables
        kernel zeroes per grid step, a whole number of 1024-word tiles.
      weight_dma_priority: the DMA queue the expert weight refills issue
        on, 1 (off the token rows' in-order queue) or 0.
      sorted_rows: how the FFN kernel fetches each expert's input rows.
        False: one copy per row from the token buffer, whose rows are in
        token order. The copies overlap the result push.
        True: a sparse-core gather sorts the routed rows into expert
        order first, and the kernel fetches each tile as one copy. Whole-expert builds only: a streamed build keeps
        the per-row fetch, and so does a width the gather cannot move or
        a table whose sort program is over the sparse core's program
        size. None (the default): sorted_rows_by_shape decides from the
        shard's experts and tokens.
    """
    result_slots: int = RESULT_SLOTS_DEFAULT
    row_fetch_ahead: int = ROW_FETCH_AHEAD_DEFAULT
    weight_slots: int = WEIGHT_SLOTS_DEFAULT
    weight_prefetch: int = WEIGHT_PREFETCH_DEFAULT
    tables_blocks_per_step: int = TABLES_BLOCKS_PER_STEP_AUTOMATIC
    stream_block: int = STREAM_BLOCK_AUTOMATIC
    stream_rows: int = STREAM_ROWS_DEFAULT
    vmem_fraction: float = VMEM_FRACTION_DEFAULT
    activation_block: int = 0
    token_block: object = None
    result_rows: str = "fp8"
    token_rows: str = "fp8"
    intermediate: str = "fp8"
    elementwise_dtype: str = "bfloat16"
    combine_dtype: str = "float32"
    row_scale_dtype: object = jnp.float32
    bounds_checks: bool = False
    gather_block_rows: int = GATHER_BLOCK_ROWS_DEFAULT
    gather_ring: int = GATHER_RING_DEFAULT
    scatter_chunk_pairs: int = SCATTER_CHUNK_PAIRS_DEFAULT
    scatter_ring: int = SCATTER_RING_DEFAULT
    combine_tokens_per_tile: int = COMBINE_TOKENS_PER_TILE_DEFAULT
    combine_unroll: int = COMBINE_UNROLL_DEFAULT
    activation_columns_per_pass: int = ACTIVATION_COLUMNS_PER_PASS_DEFAULT
    tables_zero_words: int = TABLES_ZERO_WORDS_DEFAULT
    weight_dma_priority: int = WEIGHT_DMA_PRIORITY_DEFAULT
    sorted_rows: object = None

    @property
    def token_dtype(self):
        """The token rows' element type on the wire."""
        return jnp.bfloat16 if self.token_rows == "bf16" else jnp.float8_e4m3fn

    @property
    def token_rows_block(self):
        """The token rows' rounding block: token_block, or activation_block
        when token_block is None."""
        return (self.activation_block
                if self.token_block is None else self.token_block)

    @classmethod
    def for_shape(cls, *, hidden, inter, experts_per_shard, top_k):
        """The record for a layer shape: the defaults, with the fields
        TUNED_BY_SHAPE holds for (hidden, inter, experts_per_shard, top_k)
        on top."""
        tuned = TUNED_BY_SHAPE.get((hidden, inter, experts_per_shard, top_k),
                                   {})
        return cls(**tuned)

    @property
    def elementwise(self):
        """The element type as a jnp dtype."""
        return jnp.float32 if self.elementwise_dtype == "float32" else jnp.bfloat16

    @property
    def combine(self):
        """The combine's element type as a jnp dtype."""
        return jnp.float32 if self.combine_dtype == "float32" else jnp.bfloat16

    def __post_init__(self):
        if not any(self.sorted_rows is v for v in (None, True, False)):
            raise ValueError(f"sorted_rows {self.sorted_rows!r}: True, False "
                             "or None for the choice by shape")
        if self.result_slots < 2:
            raise ValueError(f"result_slots {self.result_slots}: one slot "
                             "computes while another drains, so at least 2")
        if not 1 <= self.row_fetch_ahead < self.result_slots:
            raise ValueError(
                f"row_fetch_ahead {self.row_fetch_ahead} has to be between 1 "
                f"and result_slots - 1 = {self.result_slots - 1}: a "
                "tile's rows land in the slot its results are staged in")
        if self.weight_slots < 2:
            raise ValueError(f"weight_slots {self.weight_slots}: one slot is "
                             "read while another fills, so at least 2")
        if not 1 <= self.weight_prefetch < self.weight_slots:
            raise ValueError(
                f"weight_prefetch {self.weight_prefetch} has to be between 1 "
                f"and weight_slots - 1 = {self.weight_slots - 1}: the refill "
                "must not land in a slot an earlier expert still reads")
        if self.tables_blocks_per_step < 0:
            raise ValueError(f"tables_blocks_per_step "
                             f"{self.tables_blocks_per_step}: a count of "
                             "routing blocks, or 0 for the layer's choice")
        if self.stream_block < 0 or self.stream_block % LANES:
            raise ValueError(f"stream_block {self.stream_block}: 0 (the "
                             f"layer's choice) or a multiple of {LANES}")
        if not 0.0 < self.vmem_fraction <= 1.0:
            raise ValueError(f"vmem_fraction {self.vmem_fraction}: a share of "
                             "the chip's VMEM, above 0 and at most 1")
        if self.stream_rows < ROW_BLOCK or self.stream_rows % ROW_BLOCK:
            raise ValueError(f"stream_rows {self.stream_rows}: a whole number "
                             f"of {ROW_BLOCK}-row blocks")
        if self.combine_dtype not in ELEMENTWISE_DTYPES:
            raise ValueError(f"combine_dtype {self.combine_dtype!r} is not "
                             f"one of {ELEMENTWISE_DTYPES}")
        if self.elementwise_dtype not in ELEMENTWISE_DTYPES:
            raise ValueError(f"elementwise_dtype {self.elementwise_dtype!r} "
                             f"is not one of {ELEMENTWISE_DTYPES}")
        if self.intermediate not in INTERMEDIATES:
            raise ValueError(
                f"intermediate {self.intermediate!r} is not one of "
                f"{INTERMEDIATES}")
        if self.result_rows not in RESULT_ROWS:
            raise ValueError(f"result_rows {self.result_rows!r} is not one of "
                             f"{RESULT_ROWS}")
        if self.token_rows not in TOKEN_ROWS:
            raise ValueError(f"token_rows {self.token_rows!r} is not one of "
                             f"{TOKEN_ROWS}")
        if self.activation_block < 0 or self.activation_block % LANES:
            raise ValueError(f"activation_block {self.activation_block}: 0 "
                             f"(one scale per row) or a multiple of {LANES}")
        if self.token_block is not None and (
                not isinstance(self.token_block, int) or self.token_block < 0
                or self.token_block % LANES):
            raise ValueError(f"token_block {self.token_block}: None (the "
                             f"activation block), 0 (one scale per row) or a "
                             f"multiple of {LANES}")
        dtypes = [jnp.dtype(d) for d in ROW_SCALE_DTYPES]
        if jnp.dtype(self.row_scale_dtype) not in dtypes:
            raise ValueError(f"row_scale_dtype {self.row_scale_dtype} is not "
                             f"one of {[d.name for d in dtypes]}")
        if self.gather_block_rows <= 0 or self.gather_block_rows % SUBLANES:
            raise ValueError(f"gather_block_rows {self.gather_block_rows}: a "
                             f"whole number of {SUBLANES}-row blocks")
        if self.gather_ring < 1:
            raise ValueError(f"gather_ring {self.gather_ring}: at least one "
                             "gather DMA in flight")
        if self.scatter_chunk_pairs <= 0 or self.scatter_chunk_pairs % LANES:
            raise ValueError(
                f"scatter_chunk_pairs {self.scatter_chunk_pairs}: "
                f"a multiple of {LANES}")
        if self.scatter_ring < 1:
            raise ValueError(f"scatter_ring {self.scatter_ring}: at least one "
                             "scatter chunk in flight")
        if (self.combine_tokens_per_tile <= 0
                or self.combine_tokens_per_tile % SUBLANES):
            raise ValueError(f"combine_tokens_per_tile "
                             f"{self.combine_tokens_per_tile}: a whole number "
                             f"of {SUBLANES}-row blocks")
        if (self.combine_unroll < 1
                or self.combine_tokens_per_tile % self.combine_unroll):
            raise ValueError(f"combine_unroll {self.combine_unroll}: at least "
                             "1, dividing combine_tokens_per_tile "
                             f"{self.combine_tokens_per_tile}")
        if (self.activation_columns_per_pass <= 0
                or self.activation_columns_per_pass % LANES):
            raise ValueError(f"activation_columns_per_pass "
                             f"{self.activation_columns_per_pass}: a multiple "
                             f"of {LANES}")
        if (self.tables_zero_words <= 0
                or self.tables_zero_words % (SUBLANES * LANES)):
            raise ValueError(f"tables_zero_words {self.tables_zero_words}: a "
                             f"whole number of {SUBLANES * LANES}-word tiles")
        if self.weight_dma_priority not in WEIGHT_DMA_PRIORITIES:
            raise ValueError(f"weight_dma_priority {self.weight_dma_priority} "
                             f"is not one of {WEIGHT_DMA_PRIORITIES}")


def sorted_rows_by_shape(experts_per_shard, tokens_per_shard):
    """The row fetch for a shard's shape when Config.sorted_rows is None:
    sorted when the shard holds SORTED_ROWS_EXPERTS_PER_SHARD experts or
    more, or SORTED_ROWS_TOKENS_PER_SHARD tokens or fewer; the per-row
    fetch otherwise: with many experts per shard or few tokens the sort
    pays for itself, with few experts and many tokens the per-row copies
    hide under the result push and the sort is exposed."""
    return (experts_per_shard >= SORTED_ROWS_EXPERTS_PER_SHARD
            or tokens_per_shard <= SORTED_ROWS_TOKENS_PER_SHARD)
