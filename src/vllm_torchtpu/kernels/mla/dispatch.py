# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Routing among MLA ragged paged attention kernels.

The router owns eligibility, selection and composition; each kernel package
under `kernels/mla` owns only its implementation and does not import this
module. The sparse kernel is the default path, which the sparse MLA layer
calls directly. The layer enters this module only through its opt-in branch,
after `applies` accepts the call's static shapes.

The current routing policy is the opt-in GLM-5.2 TPU7x masked-dense
experiment: the exact profile, the calibrated cost selection and the
device-side choice between the sparse and masked-dense kernels live here, so
the sparse path, the runner and the attention metadata carry no routing state.

Each step runs one of three routes, selected on device:

    analytic, bitmap   masked-dense for every request, or a sparse prefix of
                       one-token decodes and over-limit requests followed by a
                       masked-dense suffix
    sparse             the sparse kernel's stages for the whole step

The runner may order requests as `[decode][longest KV ... shortest KV]`, but
nothing here relies on that: each masked-dense route derives its split from
device tensors and is not selected unless the sparse requests form a prefix.
"""

import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.mla.masked_dense.cost_model import (
    GLM52_TPU7X_PREFILL_COST_MODEL,
    KV_BLOCK_SIZE,
    QUERY_BLOCK_SIZE,
    MaskedDensePrefillCostModel,
)
from vllm_torchtpu.kernels.mla.masked_dense.kernel import (
    masked_dense_ragged_paged_attention,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_kernel

# This profile is deliberately exact. The opt-in is process-wide, while the
# sparse MLA wrapper is generic to vLLM's DeepSeekV2-style model family. An
# unrecognized sparse model must therefore retain the original gather kernel.
GLM52_NUM_HEADS = 64
GLM52_NOPE_DIM = 512
GLM52_ROPE_DIM = 64
GLM52_INDEX_TOPK = 2048
GLM52_PAGE_SIZE = 1024

MASKED_DENSE_ANALYTIC_MAX_KV_LEN = 2048
MASKED_DENSE_MAX_KV_LEN = 6144
MASKED_DENSE_KV_BLOCK = 1024
MASKED_DENSE_KV_PAGES_PER_BLOCK = (1, 1, 1)
MASKED_DENSE_QUERIES_PER_BLOCK = (1, 32, 32)
MASKED_DENSE_PREFILL_COST_MODEL = GLM52_TPU7X_PREFILL_COST_MODEL
# Keep decode-sized compiled buckets on the original sparse graph. The
# masked-dense serving target uses at least 512 query rows; below that point a
# dynamic dispatcher would add branch and compilation cost to the common
# decode path while serving no targeted prefill workload.
MASKED_DENSE_MIN_TOKEN_BUCKET = 512
# A 1,024-token TPU7x sweep found a split with 256 masked-dense suffix rows
# slower than sparse-only, while 512 suffix rows was faster.
MASKED_DENSE_MIN_SUFFIX_TOKENS = 512


def _resolve_masked_dense_limit(name: str, default: int) -> int:
    value = getattr(envs, name)
    if value is None:
        return default
    if value < 0 or value % MASKED_DENSE_KV_BLOCK != 0:
        raise ValueError(
            f"{name}={value} must be 0 (disabled) or a positive multiple of "
            f"{MASKED_DENSE_KV_BLOCK}"
        )
    return value


# Process-static configuration: servers set the environment before importing
# model code. Tests may override these module attributes directly.
MASKED_DENSE_ANALYTIC_LIMIT = _resolve_masked_dense_limit(
    "TPU_MLA_MASKED_DENSE_ANALYTIC_MAX_KV_LEN", MASKED_DENSE_ANALYTIC_MAX_KV_LEN
)
MASKED_DENSE_LIMIT = _resolve_masked_dense_limit(
    "TPU_MLA_MASKED_DENSE_MAX_KV_LEN", MASKED_DENSE_MAX_KV_LEN
)


def _align_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def matches_glm52_tpu7x_profile(
    q: jax.Array | jax.ShapeDtypeStruct,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    topk_indices: jax.Array,
) -> bool:
    """Whether static operands match the calibrated GLM-5.2 kernel profile."""
    return bool(
        q.ndim == 3
        and q.shape[1:] == (GLM52_NUM_HEADS, GLM52_NOPE_DIM + GLM52_ROPE_DIM)
        and topk_indices.ndim == 2
        and topk_indices.shape[-1] == GLM52_INDEX_TOPK
        and kv_cache_nope.ndim == 4
        and kv_cache_nope.shape[1] == GLM52_PAGE_SIZE
        and kv_cache_rope.ndim == 4
        and kv_cache_rope.shape[1] * kv_cache_rope.shape[2] == GLM52_PAGE_SIZE
    )


def applies(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    topk_indices: jax.Array,
    cache_layout: str,
) -> bool:
    """Whether a sparse MLA call may take the masked-dense branch.

    Decided at trace time from static shapes and process-wide settings, so a
    call that does not apply compiles exactly the original sparse graph.
    """
    query = jax.ShapeDtypeStruct(
        (*ql_nope.shape[:-1], ql_nope.shape[-1] + q_pe.shape[-1]), ql_nope.dtype
    )
    return bool(
        envs.TPU_MLA_MASKED_DENSE_ENABLED
        and cache_layout == sparse_kernel.TENSORCORE_CACHE_LAYOUT
        and query.shape[0] >= MASKED_DENSE_MIN_TOKEN_BUCKET
        and matches_glm52_tpu7x_profile(
            query, kv_cache_nope, kv_cache_rope, topk_indices
        )
    )


def split_point(
    seq_lens: jax.Array,
    request_distribution: jax.Array,
    limit: int,
) -> tuple[jax.Array, jax.Array]:
    """Splits a step into a sparse prefix and a masked-dense suffix.

    The prefix holds every one-token decode and every request whose KV length
    exceeds `limit`. Returns `(split_seq, contiguous)`: the split is usable
    only when `contiguous`, i.e. those requests are exactly `[0, split_seq)`.
    """
    seq_ids = jnp.arange(seq_lens.shape[0])
    active = seq_ids < request_distribution[2]
    in_prefix = active & ((seq_ids < request_distribution[0]) | (seq_lens > limit))
    split_seq = jnp.sum(in_prefix).astype(jnp.int32)
    contiguous = jnp.all(~active | (in_prefix == (seq_ids < split_seq)))
    return split_seq, contiguous


def select_tier(
    seq_lens: jax.Array,
    request_distribution: jax.Array,
    query_start_loc: jax.Array,
    limits: tuple[int, int],
    cost_model: MaskedDensePrefillCostModel | None = None,
) -> jax.Array:
    """Selects 0 (analytic), 1 (bitmap) or 2 (sparse) for one step.

    `limits` is (analytic, bitmap); zero disables a tier. A masked-dense tier
    is a candidate when its split is contiguous, leaves at least one request
    for the masked-dense kernel and, if it has a sparse prefix, at least
    `MASKED_DENSE_MIN_SUFFIX_TOKENS` masked-dense tokens. The cheapest
    candidate under the calibrated cost model wins.
    """
    if cost_model is None:
        cost_model = MASKED_DENSE_PREFILL_COST_MODEL
    unavailable = jnp.iinfo(jnp.int32).max
    max_num_seqs = seq_lens.shape[0]
    num_seqs = request_distribution[2]
    seq_ids = jnp.arange(max_num_seqs)
    active = seq_ids < num_seqs
    q_lens = query_start_loc[1 : max_num_seqs + 1] - query_start_loc[:max_num_seqs]
    kv_blocks = (jnp.where(active, seq_lens, 0) + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    q_blocks = (q_lens + QUERY_BLOCK_SIZE - 1) // QUERY_BLOCK_SIZE
    num_tokens = query_start_loc[num_seqs]

    def masked_dense_cost(limit, fixed_ns, token_ns, bitmap):
        if limit == 0:
            return jnp.array(unavailable, jnp.int32)
        split_seq, contiguous = split_point(seq_lens, request_distribution, limit)
        split_token = query_start_loc[split_seq]
        suffix_tokens = num_tokens - split_token
        suffix = active & (seq_ids >= split_seq)
        suffix_grid = jnp.sum(jnp.where(suffix, q_blocks * kv_blocks, 0))
        # The CSA bitmap is generated for every query row, prefix included.
        dense_tokens = num_tokens if bitmap else suffix_tokens
        cost = (
            fixed_ns + token_ns * dense_tokens + cost_model.q_kv_block_ns * suffix_grid
        )
        cost = cost + jnp.where(
            split_seq > 0,
            cost_model.sparse_fixed_ns + cost_model.sparse_token_ns * split_token,
            0,
        )
        valid = (
            contiguous
            & (split_seq < num_seqs)
            & ((split_seq == 0) | (suffix_tokens >= MASKED_DENSE_MIN_SUFFIX_TOKENS))
        )
        return jnp.where(valid, cost, unavailable)

    analytic_limit, bitmap_limit = limits
    sparse_cost = cost_model.sparse_fixed_ns + cost_model.sparse_token_ns * num_tokens
    costs = jnp.stack(
        (
            masked_dense_cost(
                analytic_limit,
                cost_model.analytic_fixed_ns,
                cost_model.analytic_token_ns,
                bitmap=False,
            ),
            masked_dense_cost(
                bitmap_limit,
                cost_model.bitmap_fixed_ns,
                cost_model.bitmap_token_ns,
                bitmap=True,
            ),
            sparse_cost,
        )
    )
    return jnp.argmin(costs).astype(jnp.int32)


def _sparse_prefix(q, split_token, run_chunk):
    """Sparse attention for query rows `[0, split_token)`, in place.

    `run_chunk` must preserve rows at or after `token_end`: they still hold
    queries for the masked-dense suffix.

    A split always leaves at least `MASKED_DENSE_MIN_SUFFIX_TOKENS` suffix
    rows, so chunks starting in that tail are not emitted. Earlier chunks are
    skipped on device once they start past the split.
    """
    num_rows = q.shape[0]
    chunk_size = sparse_kernel.DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE
    for start_pos in range(0, num_rows - MASKED_DENSE_MIN_SUFFIX_TOKENS, chunk_size):
        q = jax.lax.cond(
            start_pos < split_token,
            functools.partial(run_chunk, start_pos=start_pos, token_end=split_token),
            lambda x: x,
            q,
        )
    return q


def attention_from_parts(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    topk_indices: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    sm_scale: float,
    k_scale: float,
) -> jax.Array:
    """MLA attention for a call `applies` accepted.

    Returns `[tokens, heads, ql_nope.shape[-1]]`. Each route concatenates the
    query parts and crops its own output: under the torch-tpu lowering an array
    crossing a `lax.switch`/`lax.cond` boundary is copied, so the full-width
    query and output stay inside the routes.
    """
    num_rows, num_heads, latent_width = ql_nope.shape
    chunk_size = sparse_kernel.DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE

    def query():
        return sparse_kernel.prepare_q_inputs(jnp.concatenate([ql_nope, q_pe], axis=-1))

    def crop(q):
        return q[:, :num_heads, :latent_width]

    # Query-independent, so it is prepared once outside the conditionals and
    # layers that share these indexer outputs can share it too.
    kv_lens, physical_indices = sparse_kernel.prepare_gather_indices(
        topk_indices,
        block_tables,
        query_start_loc,
        page_size=kv_cache_nope.shape[1],
        total_num_pages=kv_cache_nope.shape[0],
    )

    def sparse_chunk(q, start_pos, token_end, *, preserve_suffix=False):
        return sparse_kernel.sparse_attention_chunk(
            q,
            kv_cache_nope,
            kv_cache_rope,
            kv_lens,
            physical_indices,
            None,
            start_pos=start_pos,
            end_pos=min(start_pos + chunk_size, num_rows),
            token_end=token_end,
            sm_scale=sm_scale,
            k_scale=k_scale,
            preserve_suffix=preserve_suffix,
        )

    def masked_dense(q, max_kv_len, sequence_start=None):
        return masked_dense_ragged_paged_attention(
            q,
            kv_cache_nope,
            kv_cache_rope,
            seq_lens,
            topk_indices,
            block_tables,
            query_start_loc,
            request_distribution,
            sm_scale=sm_scale,
            k_scale=k_scale,
            max_kv_len=max_kv_len,
            num_kv_pages_per_block=MASKED_DENSE_KV_PAGES_PER_BLOCK,
            num_queries_per_block=MASKED_DENSE_QUERIES_PER_BLOCK,
            sequence_start=sequence_start,
        )

    def sparse_route():
        q = query()
        token_end = query_start_loc[request_distribution[2]]
        for start_pos in range(0, num_rows, chunk_size):
            q = sparse_chunk(q, start_pos, token_end)
        return crop(q)

    def masked_dense_route(limit, max_kv_len):
        split_seq, _ = split_point(seq_lens, request_distribution, limit)

        def whole(_):
            return crop(masked_dense(query(), max_kv_len))

        def split(_):
            q = _sparse_prefix(
                query(),
                query_start_loc[split_seq],
                functools.partial(sparse_chunk, preserve_suffix=True),
            )
            return crop(masked_dense(q, max_kv_len, split_seq))

        return jax.lax.cond(split_seq > 0, split, whole, None)

    page_span = block_tables.shape[0] // seq_lens.shape[0] * kv_cache_nope.shape[1]
    branches, tier_limits = [], []
    for slot, limit in enumerate((MASKED_DENSE_ANALYTIC_LIMIT, MASKED_DENSE_LIMIT)):
        max_kv_len = min(_align_to(limit, MASKED_DENSE_KV_BLOCK), page_span)
        if max_kv_len == 0 or (slot == 0 and max_kv_len > topk_indices.shape[-1]):
            tier_limits.append(0)
        else:
            tier_limits.append(limit)
            branches.append(functools.partial(masked_dense_route, limit, max_kv_len))
    branches.append(sparse_route)

    if len(branches) == 1:
        return sparse_route()
    tier = select_tier(
        seq_lens, request_distribution, query_start_loc, tuple(tier_limits)
    )
    if len(branches) == 2:
        kept = 0 if tier_limits[0] else 1
        return jax.lax.cond(tier == kept, branches[0], branches[1])
    return jax.lax.switch(tier, branches)


def groups_requests(
    attn_module,
    kv_cache_specs,
    page_size: int,
    decode_context_parallel_size: int,
) -> bool:
    """Whether the runner should order a sparse MLA layer's requests by length.

    The static counterpart of `applies`, evaluated from the layer's
    configuration when its KV cache is allocated. Ordering by KV length only
    helps this routing, so other models keep their request order. The kernel
    query is the `kv_lora_rank`-wide latent plus the rope part.
    """
    indexer = getattr(attn_module, "indexer", None)
    topk = getattr(indexer, "topk_tokens", getattr(attn_module, "topk_tokens", None))
    if not isinstance(topk, int):
        buf = getattr(
            attn_module,
            "topk_indices_buffer",
            getattr(getattr(attn_module, "impl", None), "topk_indices_buffer", None),
        )
        if buf is not None and getattr(buf, "ndim", 0) >= 2:
            topk = int(buf.shape[-1])
    topk_matches = (
        topk == GLM52_INDEX_TOPK if isinstance(topk, int) else indexer is None
    )
    return bool(
        decode_context_parallel_size == 1
        and all(
            spec.layout.value == sparse_kernel.TENSORCORE_CACHE_LAYOUT
            for spec in kv_cache_specs
        )
        and attn_module.num_heads == GLM52_NUM_HEADS
        and attn_module.kv_lora_rank == GLM52_NOPE_DIM
        and attn_module.qk_rope_head_dim == GLM52_ROPE_DIM
        and topk_matches
        and page_size == GLM52_PAGE_SIZE
    )


def update_runner_profile(
    runner,
    attn_module,
    kv_cache_specs,
    page_size: int,
) -> None:
    """Update runner's static eligibility for masked-dense request ordering."""
    if getattr(runner, "_masked_dense_groups_requests", None) is False:
        return
    dcp_size = getattr(
        getattr(runner, "parallel_config", None),
        "decode_context_parallel_size",
        1,
    )
    runner._masked_dense_groups_requests = groups_requests(
        attn_module, kv_cache_specs, page_size, dcp_size
    )


def order_prefills_by_kv_len(
    runner,
    start: int,
    end: int,
    scheduler_output,
) -> None:
    """Order requests `[start, end)` by KV length, longest first.

    Used only with the opt-in masked-dense MLA routing, which can run a
    sparse prefix of long requests and a masked-dense suffix of short ones.
    The routing re-derives and validates any split on device, so this order
    affects performance only. Ties keep their current order, and
    `swap_states` moves every request-associated buffer with its request.
    """
    tail: list[str] = []
    kv_lens: dict[str, int] = {}
    position: dict[str, int] = {}
    for i in range(start, end):
        req_id = runner.input_batch.req_ids[i]
        assert req_id is not None
        tail.append(req_id)
        kv_lens[req_id] = int(runner.input_batch.num_computed_tokens_cpu[i]) + int(
            scheduler_output.num_scheduled_tokens[req_id]
        )
        position[req_id] = i
    for dst, req_id in enumerate(
        sorted(tail, key=kv_lens.__getitem__, reverse=True), start
    ):
        src = position[req_id]
        if src != dst:
            displaced = runner.input_batch.req_ids[dst]
            assert displaced is not None
            runner.input_batch.swap_states(dst, src)
            position[displaced] = src
            position[req_id] = dst


def maybe_order_prefills_by_kv_len(
    runner,
    num_decode: int,
    num_reqs: int,
    scheduler_output,
    spec_decode_tokens,
) -> None:
    """Order non-speculative prefill requests by KV length when enabled."""
    if (
        not spec_decode_tokens
        and getattr(runner, "_masked_dense_groups_requests", None)
        and envs.TPU_MLA_MASKED_DENSE_ENABLED
    ):
        order_prefills_by_kv_len(runner, num_decode, num_reqs, scheduler_output)
