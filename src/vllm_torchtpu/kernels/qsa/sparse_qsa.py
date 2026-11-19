"""QSA sparse GQA attention: SparseCore gather -> HBM -> TensorCore attention.

The leading `batch` axis of every input is a flat list of *query tokens*, not
requests: each query token carries its own selected top-k groups
(`blocks[b]`), its own causal context length (`lengths[b]`) and its own
page-table row (`table[b]`), and attends only to its own selected KVs. So the
same kernel serves:

  * decode: one query token per request;
  * spec-decoding verify: the `1 + num_draft` tokens of a request, each with
    its own top-k selection and `lengths` incrementing by one per token;
  * prefill / chunked prefill: every prompt token of a request, each with its
    own top-k selection and causal `lengths` (position + 1).

Query tokens of the same request simply repeat that request's page-table row.
Below, "query" (or "row") means one entry of the `batch` axis.

Stage 1 (`qsa_gather`, SparseCore): for every query, gather the cache rows of
its selected groups -- the top-k groups plus the partial tail group, one
contiguous `RATIO` (4)-token row of all KV heads per group (requires a
power-of-2 `page_size >= 4`) -- into a dense HBM buffer
`int32[chunk_size, kv_slots * kv_heads, 128]` (one per chunk of `chunk_size`
queries).

Stage 2 (`_qsa_attention_body`, TensorCore): an `emit_pipeline` streams the
gathered rows into multi-buffered VMEM; fp8 K/V byte-plane extraction (fed to
the MXU as fp8 weights against bf16 q / p), single-query softmax attention
(each query token against its own gathered tokens), then the sigmoid output
gate.

Requires `dim == 256` and an fp8 (e4m3fn) cache. Applied independently per
query token `b`, with query head `h` attending to KV head
`h // (heads // kv_heads)`:

  out[b] = softmax(q[b] k_b^T / sqrt(dim)) v_b * sigmoid(gate[b]),

where `k_b` / `v_b` are the tokens selected for query `b`, zeroed for queries
with `lengths[b] == 0`.

Cache layout: `int32[num_pages, page_size * kv_heads, 128]` (512 bytes per
token per KV head), packed in 4-token blocks whose byte planes bitcast natively
to token-ordered fp8 tiles; every gathered group (4 tokens, page-aligned) is one
contiguous `kv_heads * 2` KB row holding all KV heads. See `qsa_cache_layout`
for the byte layout and `qsa_cache_layout.pack_kv_cache` to build it.

Token list per query (`(budget + 1) * ratio` slots, padded to `kv_slots` in
HBM and `num_slots` in VMEM):
the `count = min(lengths // ratio, budget)` selected groups first, then the
`lengths % ratio` tokens of the tail group. The valid tokens are therefore
always a prefix of length `kv_len = count * ratio + lengths % ratio`, so the
attention kernel only needs a per-query length mask (causality across the
query tokens of one request is encoded in their per-query `lengths` /
`blocks`; no extra causal mask is applied).
"""

import functools
import math

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.qsa import qsa_cache_layout
from vllm_torchtpu.kernels.qsa import qsa_gather

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

HEAD_DIM = qsa_cache_layout.HEAD_DIM
LANES = qsa_cache_layout.LANES
ROW_WORDS = qsa_cache_layout.ROW_WORDS
assert ROW_WORDS == qsa_gather.ROW_WORDS, (ROW_WORDS, qsa_gather.ROW_WORDS)
CACHE_DTYPE = qsa_cache_layout.CACHE_DTYPE
# Tokens per cache block: one int32 word packs this many fp8 bytes.
TOKEN_BLOCK = qsa_cache_layout.TOKEN_BLOCK
# Tokens per compression group (gathered as one contiguous `kv_heads * 2` KB
# cache row). Fixed for this model.
RATIO = 4
# Per-query token slots in VMEM are padded to this multiple, keeping the
# gathered block lane-dense for the [heads, T] score tile.
TOKEN_ALIGN = 128
# Per-query token slots in HBM are padded to this multiple (one int32
# sublane tile), so the gather output reshapes for free.
GATHER_ROW_ALIGN = 8


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def align_to(x, a):
  return cdiv(x, a) * a


def _largest_divisor(x: int, cap: int) -> int:
  """Largest divisor of ``x`` that is <= ``cap``."""
  for candidate in range(min(x, cap), 0, -1):
    if x % candidate == 0:
      return candidate
  return 1


# ---------------------------------------------------------------------------
# Gather index build (TensorCore Pallas kernel).
# ---------------------------------------------------------------------------


def _is_pow2(x: int) -> bool:
  return x > 0 and x & (x - 1) == 0


# Queries per index-build grid step (one vreg of sublanes).
_LOOKUP_ROWS = 8


def build_group_gather_indices(
    blocks: jax.Array,  # i32[batch, budget]
    lengths: jax.Array,  # i32[batch]
    table: jax.Array,  # i32[batch, pages_per_seq]
    *,
    page_size: int,
    total_num_pages: int,
    num_slots: int,
) -> tuple[jax.Array, jax.Array]:
  """Group-row indices `i32[batch, num_slots // RATIO]` and `kv_len`.

  One index per compression group into the cache viewed as
  `[num_tokens // RATIO, RATIO * kv_heads * 128]`: with a power-of-2
  `page_size >= RATIO` a group's `RATIO` tokens are one 4-token block of one
  page, i.e. one contiguous `kv_heads * 2` KB row.

  Slot order matches `qsa_ref.ref_sparse_qsa_decode`: the selected groups,
  then the tail group, so the valid tokens form a prefix of length `kv_len`.
  The tail group is gathered whole; its tokens past `kv_len` are masked by the
  attention kernel like any other invalid slot. Invalid groups are spread over
  the cache to avoid gather hotspots.
  """
  assert _is_pow2(page_size) and page_size >= RATIO, page_size
  batch, budget = blocks.shape
  pages_per_seq = table.shape[1]
  assert num_slots % RATIO == 0, num_slots
  num_group_slots = num_slots // RATIO
  assert num_group_slots >= budget + 1, (num_group_slots, budget)

  # Pad to whole vregs (8 queries x 128 columns); `blocks` to the output
  # width so every column chunk reads it. Padded queries have length 0.
  padded_batch = align_to(batch, _LOOKUP_ROWS)
  pad_rows = padded_batch - batch
  lengths = lengths.astype(jnp.int32)
  blocks = jnp.pad(
      blocks.astype(jnp.int32),
      ((0, pad_rows), (0, align_to(num_group_slots, LANES) - budget)),
  )
  table = jnp.pad(
      table.astype(jnp.int32),
      ((0, pad_rows), (0, align_to(pages_per_seq, LANES) - pages_per_seq)),
  )

  row_block = lambda cols: pl.BlockSpec((_LOOKUP_ROWS, cols), lambda i: (i, 0))
  group_rows = pl.pallas_call(
      functools.partial(
          _group_rows_kernel,
          budget=budget,
          page_size=page_size,
          num_group_rows=total_num_pages * page_size // RATIO,
      ),
      grid=(padded_batch // _LOOKUP_ROWS,),
      in_specs=[
          row_block(1),
          row_block(blocks.shape[1]),
          row_block(table.shape[1]),
      ],
      out_specs=row_block(num_group_slots),
      out_shape=jax.ShapeDtypeStruct(
          (padded_batch, num_group_slots), jnp.int32
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          disable_bounds_checks=True,
      ),
      name="qsa_group_rows",
  )(jnp.pad(lengths, (0, pad_rows))[:, None], blocks, table)

  kv_len = jnp.minimum(lengths // RATIO, budget) * RATIO + lengths % RATIO
  return group_rows[:batch], kv_len


def _group_rows_kernel(
    lengths_ref,  # i32[8, 1]
    blocks_ref,  # i32[8, align_to(num_group_slots, 128)]
    table_ref,  # i32[8, padded_pages]
    out_ref,  # i32[8, num_group_slots]
    *,
    budget: int,
    page_size: int,
    num_group_rows: int,
):
  """Group-row indices of 8 queries (one vreg of sublanes).

  Per 128-column chunk of group slots: tail-group insertion, validity, page
  lookup and the scattered rows of invalid slots -- the whole index build in
  one launch instead of a chain of small XLA ops. Index math is shifts / masks
  (power-of-2 `page_size`, `RATIO == 4`).
  """
  rows, num_group_slots = out_ref.shape
  groups_per_page = page_size // RATIO
  log_gpp = groups_per_page.bit_length() - 1
  # Invalid slots read scattered rows (avoids gather hotspots): a power-of-2
  # mask instead of `% num_group_rows`, which stays in range.
  scatter_mask = (1 << (num_group_rows.bit_length() - 1)) - 1

  shape = (rows, LANES)
  row_iota = lax.broadcasted_iota(jnp.int32, shape, 0)
  lane_iota = lax.broadcasted_iota(jnp.int32, shape, 1)
  element = (pl.program_id(0) * rows + row_iota) * num_group_slots + lane_iota
  length = jnp.broadcast_to(lengths_ref[...], shape)
  complete = lax.shift_right_logical(length, 2)  # length // RATIO
  count = jnp.minimum(complete, budget)
  has_tail = (length & (RATIO - 1)) > 0

  for c0 in range(0, num_group_slots, LANES):
    col = lane_iota + c0
    # Selected groups, then the tail group. Invalid slots may hold any id
    # (e.g. -1): their lookup below is harmless and the result is replaced.
    group = jnp.where(col == count, complete, blocks_ref[:, c0 : c0 + LANES])
    valid = (col < count) | ((col == count) & has_tail)
    # physical = table[r, logical]: one 128-lane gather per table chunk.
    # Logical pages past the table match no chunk (or its zero padding).
    logical = lax.shift_right_logical(group, log_gpp)
    page_chunk = lax.shift_right_logical(logical, 7)  # logical // LANES
    page_lane = logical & (LANES - 1)
    physical = jnp.zeros(shape, jnp.int32)
    for t in range(table_ref.shape[1] // LANES):
      gathered = jnp.take_along_axis(
          table_ref[:, t * LANES : (t + 1) * LANES], page_lane, axis=1
      )
      physical = jnp.where(page_chunk == t, gathered, physical)
    group_row = lax.shift_left(physical, log_gpp) | (
        group & (groups_per_page - 1)
    )
    scattered = ((element + c0) * 104729) & scatter_mask
    width = min(LANES, num_group_slots - c0)
    out_ref[:, c0 : c0 + width] = jnp.where(valid, group_row, scattered)[
        :, :width
    ]


# ---------------------------------------------------------------------------
# TensorCore attention kernel.
# ---------------------------------------------------------------------------


def _qsa_attention_body(
    q_ref,  # bf16[batch_size, heads, 256]
    gate_ref,  # bf16[batch_size, heads, 256]
    kv_ref,  # int32[batch_size, num_slots * kv_heads, 128]; see below
    o_ref,  # bf16[batch_size, heads, 256]
    *,
    kv_lens_ref,  # SMEM i32[batch]
    start_seq_idx: int,
    kv_heads: int,
    kv_slots: int,
    num_bufs: int,
    sm_scale: float,
):
  """Attention for one grid step of `batch_size` queries (VMEM blocks).

  `kv_ref` rows `>= kv_slots * kv_heads` are not filled by the pipeline.
  """
  batch_size, num_rows, _ = kv_ref.shape
  num_slots = num_rows // kv_heads
  group = q_ref.shape[1] // kv_heads  # query heads per KV head
  block_rows = TOKEN_BLOCK * kv_heads  # rows per 4-token block
  q_dtype = q_ref.dtype

  def load_kv(buf, h):
    """Gathered rows -> fp8 (k, v) of KV head h, each [num_slots, 256]."""

    def plane(p):
      # Rows block_rows * b + TOKEN_BLOCK * h + p hold plane p of KV head h of
      # block b's tokens; the bitcast unpacks byte j of each word to token
      # TOKEN_BLOCK * b + j.
      words = buf[
          pl.ds(
              TOKEN_BLOCK * h + p,
              num_slots // TOKEN_BLOCK,
              stride=block_rows,
          ),
          :,
      ]
      return pltpu.bitcast(words, CACHE_DTYPE)  # fp8[num_slots, 128]

    k = jnp.concatenate([plane(0), plane(1)], axis=1)
    v = jnp.concatenate([plane(2), plane(3)], axis=1)
    return k, v

  # `lax.dot_general` (not `jnp.einsum`, which type-promotes) so that a bf16
  # LHS can meet an fp8 K / V operand unconverted.
  dot = functools.partial(lax.dot_general, preferred_element_type=jnp.float32)
  nt = (((1,), (1,)), ((), ()))  # a[n, d] @ b[m, d]^T
  nn = (((1,), (0,)), ((), ()))  # a[n, m] @ b[m, d]

  def step1_qk_softmax(q, k, kv_len):
    s = dot(q, k, nt)
    s *= sm_scale
    col = lax.broadcasted_iota(jnp.int32, s.shape, 1)
    s = jnp.where(col < kv_len, s, jnp.finfo(s.dtype).min)
    m = jnp.max(s, axis=1, keepdims=True)
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=1, keepdims=True)
    return p, l

  def step2_pv(p, v, l, gate, kv_len):
    p = p.astype(jnp.bfloat16)
    o = dot(p, v, nn)
    o = o * pltpu.reciprocal(l, approx=True)
    o = o * jax.nn.sigmoid(gate.astype(jnp.float32))
    # Queries with lengths == 0 attend to nothing (their q / gate may be
    # garbage): emit zeros.
    return jnp.where(kv_len > 0, o, 0.0).astype(q_dtype)

  # Slots past kv_len get p == 0, so PV needs their V finite (0 * NaN = NaN).
  # Gathered rows [kv_len, kv_slots) are cache bytes, finite by contract (the
  # cache is zero-initialized and never written with NaN). Rows >= kv_slots
  # are never written by the pipeline's DMA and start as stale VMEM (possibly
  # fp8 NaN): clear them on each buffer slot's first use (step i uses slot
  # i % num_bufs); they stay zero afterwards.
  if kv_slots < num_slots:

    @pl.when(pl.program_id(0) < num_bufs)
    def _clear_unfilled_rows():
      kv_ref[:, kv_slots * kv_heads :, :] = jnp.zeros(
          (batch_size, (num_slots - kv_slots) * kv_heads, ROW_WORDS),
          jnp.int32,
      )

  base = start_seq_idx + pl.program_id(0) * batch_size
  kv_lens = [kv_lens_ref[base + i] for i in range(batch_size)]

  # Software-pipeline the (query, KV head) pairs: QK/softmax of one pair
  # overlaps PV of the previous one, interleaving MXU and VPU/EUP work.
  prev = None
  for i in range(batch_size):
    for h in range(kv_heads):
      heads = slice(h * group, (h + 1) * group)
      k, v = load_kv(kv_ref.at[i], h)
      p, l = step1_qk_softmax(q_ref[i, heads], k, kv_lens[i])
      if prev is not None:
        dst, args = prev
        o_ref[dst] = step2_pv(*args)
      prev = ((i, heads), (p, v, l, gate_ref[i, heads], kv_lens[i]))
  dst, args = prev
  o_ref[dst] = step2_pv(*args)


def _run_attention_kernel(
    q: jax.Array,  # bf16[batch, heads, 256]
    gate: jax.Array,  # bf16[batch, heads, 256]
    gathered: jax.Array,  # int32[chunk_size, kv_slots * kv_heads, 128]
    kv_lens: jax.Array,  # i32[batch]
    prev_out: jax.Array | None,  # bf16[batch, heads, 256], earlier chunks
    *,
    kv_heads: int,
    num_slots: int,  # VMEM token slots per query (>= kv_slots)
    start_seq_idx: int,
    end_seq_idx: int,
    batch_size: int,
    num_bufs: int,
    sm_scale: float,
    vmem_limit_bytes: int,
) -> jax.Array:
  """Attention over queries [start_seq_idx, end_seq_idx).

  Writes those rows of the output. `prev_out` (the previous chunk's output) is
  aliased to the output so earlier chunks' rows carry over; the first chunk
  passes None and gets a fresh buffer, so q is never copied for aliasing.

  An `emit_pipeline` over grid steps of `batch_size` queries streams q, gate
  and the gathered rows into `num_bufs`-deep VMEM buffers and writes the
  output back double-buffered. The K/V block spans `num_slots * kv_heads`
  rows, but only the `kv_slots * kv_heads` gathered rows are copied (the
  pipeline clips blocks to the array); `_qsa_attention_body` clears the rest
  once per buffer slot.
  """
  _, num_q_heads, head_dim = q.shape
  chunk_size, kv_rows, _ = gathered.shape
  assert kv_rows % kv_heads == 0, (kv_rows, kv_heads)
  kv_slots = kv_rows // kv_heads
  assert kv_slots <= num_slots, (kv_slots, num_slots)
  assert chunk_size == end_seq_idx - start_seq_idx
  assert chunk_size % batch_size == 0
  has_prev_out = prev_out is not None

  def row_spec(**kwargs):
    # q / gate / output rows of grid step i, i.e. of its `batch_size` queries.
    return pl.BlockSpec(
        (pl.Element(batch_size), num_q_heads, head_dim),
        lambda i: (start_seq_idx + i * batch_size, 0, 0),
        **kwargs,
    )

  in_spec = row_spec(pipeline_mode=pl.Buffered(num_bufs))
  # The gathered rows of those queries (indexed relative to the chunk).
  kv_spec = pl.BlockSpec(
      (batch_size, num_slots * kv_heads, ROW_WORDS),
      lambda i: (i, 0, 0),
      pipeline_mode=pl.Buffered(num_bufs),
  )

  def kernel(kv_lens_ref, q_hbm, gate_hbm, kv_hbm, *refs):
    o_hbm = refs[-1]  # refs[0] is prev_out (aliased to o_hbm) if present.
    pltpu.emit_pipeline(
        functools.partial(
            _qsa_attention_body,
            kv_lens_ref=kv_lens_ref,
            start_seq_idx=start_seq_idx,
            kv_heads=kv_heads,
            kv_slots=kv_slots,
            num_bufs=num_bufs,
            sm_scale=sm_scale,
        ),
        grid=(chunk_size // batch_size,),
        in_specs=[in_spec, in_spec, kv_spec],
        out_specs=row_spec(),
    )(q_hbm, gate_hbm, kv_hbm, o_hbm)

  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  operands = [q, gate, gathered] + ([prev_out] if has_prev_out else [])
  scope_name = (
      f"SparseQSA-t_{num_slots}-kv_{kv_heads}-bz_{batch_size}-cz_{chunk_size}"
  )
  kernel = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[hbm] * len(operands),
          out_specs=hbm,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
      # Operand indices count the scalar prefetch; prev_out is last.
      input_output_aliases={4: 0} if has_prev_out else {},
      name=scope_name,
  )
  return jax.named_scope(scope_name)(kernel)(kv_lens, *operands)


@functools.partial(
    jax.jit,
    static_argnames=(
        "ratio",
        "kv_heads",
        "chunk_size",
        "attention_kernel_batch_size",
        "attention_num_bufs",
        "vmem_limit_bytes",
    ),
)
def sparse_qsa(
    q: jax.Array,  # bf16[batch, heads, 256]
    gate: jax.Array,  # bf16[batch, heads, 256]
    blocks: jax.Array,  # i32[batch, budget]
    lengths: jax.Array,  # i32[batch]
    table: jax.Array,  # i32[batch, pages_per_seq]
    cache: jax.Array,  # int32[num_pages, page_size * kv_heads, 128]
    *,
    ratio: int,
    kv_heads: int = 1,
    chunk_size: int = 64,
    attention_kernel_batch_size: int = 8,
    attention_num_bufs: int = 3,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
  """QSA sparse GQA attention, one independent query token per `batch` row.

  Works for decode, spec-decoding verify and (chunked) prefill alike: flatten
  all query tokens onto the `batch` axis, give each its own top-k `blocks`,
  causal `lengths` (its position + 1) and its request's page-table row. Each
  query token attends only to its own selected KVs (plus its tail group).

  Args:
    q: queries, bf16[batch, heads, 256], one row per query token.
    gate: output gate logits, bf16[batch, heads, 256].
    blocks: selected group ids per query token, i32[batch, budget]; only the
      first `min(lengths // ratio, budget)` entries are used.
    lengths: causal context length (tokens visible to that query token,
      including itself) per query token, i32[batch].
    table: page table of each query token's request, i32[batch,
      pages_per_seq]; tokens of the same request repeat the same row.
    cache: packed fp8 K/V cache, int32[num_pages, page_size * kv_heads, 128]
      (see `qsa_cache_layout` / `qsa_cache_layout.pack_kv_cache`).
    ratio: tokens per group; must be `RATIO` (4).
    kv_heads: number of KV heads; query head h uses KV head `h // (heads //
      kv_heads)`. The groups (`blocks`) are shared by all KV heads.
    chunk_size: queries per gather + attention launch pair; bounds the
      gathered HBM buffer to `chunk_size * kv_slots * kv_heads * 512` bytes.
    attention_kernel_batch_size: queries per attention grid step.
    attention_num_bufs: input (K/V, q, gate) buffering depth of the attention
      kernel; batches are fetched `attention_num_bufs - 1` grid steps ahead.
    vmem_limit_bytes: VMEM limit for the attention kernel.

  Returns:
    bf16[batch, heads, 256].
  """
  batch, num_q_heads, head_dim = q.shape
  assert head_dim == HEAD_DIM, head_dim
  assert gate.shape == q.shape
  assert kv_heads >= 1 and num_q_heads % kv_heads == 0, (num_q_heads, kv_heads)
  # bf16 sublane packing of each KV head's query-head group.
  assert (num_q_heads // kv_heads) % 2 == 0, (num_q_heads, kv_heads)
  assert cache.ndim == 3 and cache.shape[-1] == ROW_WORDS, cache.shape
  total_num_pages, page_rows, _ = cache.shape
  assert page_rows % kv_heads == 0, (cache.shape, kv_heads)
  page_size = page_rows // kv_heads
  # Gather one contiguous row per group (kv_heads * 2 KB): 4x fewer, longer
  # DMAs than one row per token. Requires page-aligned groups.
  assert ratio == RATIO, ratio
  assert _is_pow2(page_size) and page_size >= RATIO, page_size
  budget = blocks.shape[1]
  # Token slots per query: the budget groups plus the tail group. Two
  # paddings of the same count:
  #   kv_slots  -- HBM gather buffer; tight (GATHER_ROW_ALIGN) so at most one
  #                always-masked group is gathered and DMA'd.
  #   num_slots -- VMEM compute tile; lane-dense (TOKEN_ALIGN). The attention
  #                kernel zero-fills rows [kv_slots, num_slots).
  # For budget=512: kv_slots=2056, num_slots=2176, skipping 30 groups/query.
  slots = (budget + 1) * ratio
  kv_slots = align_to(slots, GATHER_ROW_ALIGN)
  num_slots = align_to(slots, TOKEN_ALIGN)
  assert kv_slots <= num_slots, (kv_slots, num_slots)

  flat_indices, kv_lens = build_group_gather_indices(
      blocks,
      lengths,
      table,
      page_size=page_size,
      total_num_pages=total_num_pages,
      num_slots=kv_slots,
  )
  flat_indices = flat_indices.reshape(-1)
  gather_cache = cache.reshape(-1, ratio * kv_heads * ROW_WORDS)
  index_rows = kv_slots // ratio  # gather indices per query

  sm_scale = 1.0 / math.sqrt(head_dim)

  out = None
  for start in range(0, batch, chunk_size):
    end = min(start + chunk_size, batch)
    chunk_size = end - start
    gathered = qsa_gather.qsa_gather(
        gather_cache,
        flat_indices[start * index_rows : end * index_rows],
    ).reshape(chunk_size, kv_slots * kv_heads, ROW_WORDS)
    out = _run_attention_kernel(
        q,
        gate,
        gathered,
        kv_lens,
        out,
        kv_heads=kv_heads,
        num_slots=num_slots,
        start_seq_idx=start,
        end_seq_idx=end,
        batch_size=_largest_divisor(chunk_size, attention_kernel_batch_size),
        num_bufs=attention_num_bufs,
        sm_scale=sm_scale,
        vmem_limit_bytes=vmem_limit_bytes,
    )
  return out
