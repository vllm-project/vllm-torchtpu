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
"""PCP history attention on the batched RPA kernel (TPU_PCP_HISTORY_RPA=1).

The streaming PCP kernel attends the cached prefix ("history") by circulating
one 128-token page per online-softmax update around the PCP ring. This module
replaces that history pass with the batched RPA prefill kernel (RPAm), which
updates once per ``bkv`` (1024) tokens, pipelines its DMAs and has the native
fp8 PV path:

1. The PCP current pass (unchanged) leaves the online state ``(m, l, acc)``
   of every local query row over the request's current chunk and the partial
   history-boundary virtual page.
2. Every rank gathers all ranks' Q (one all-gather over the PCP axis).
3. For each source rank ``s``, RPAm attends ``Q_s`` to this rank's local share
   of the full history virtual pages (``CACHE_ONLY``, no causal mask, f32 LSE).
   Local page ``j`` of request ``i`` is ``block_tables[i, j] mod
   local_blocks``, the same page the streaming history pass reads.
4. One all-to-all returns every partial to the rank that owns its queries.
5. The partials and the current state are merged by their log-sum-exp.

History positions all precede the request's queries, so the order in which
KV is visited is irrelevant and no causal mask is needed in step 3.
"""

import functools
import os
from collections.abc import Callable

import jax
import jax.numpy as jnp


def _rank_token_count_before(position, rank, *, pcp_size, interleave_size):
    cycle = pcp_size * interleave_size
    full_cycles = position // cycle
    cycle_offset = position - full_cycles * cycle
    partial = jnp.clip(cycle_offset - rank * interleave_size, 0, interleave_size)
    return full_cycles * interleave_size + partial


def history_rpa_metadata(
    kv_lens,
    cu_q_lens,
    distribution,
    *,
    pcp_size: int,
    interleave_size: int,
    page_size: int,
):
    """Per-source-rank RPAm metadata for the history pass.

    Returns:
        cu_q_lens_src: [pcp_size, max_num_seqs + 1]. Source rank ``s``'s
            rank-local compact Q rows of each request, from row 0.
        kv_lens_src: [pcp_size, max_num_seqs]. History tokens this rank holds
            for each request (identical on every rank: each full virtual page
            holds ``page_size`` tokens per rank) plus that source's local
            query count, as RPAm's ``kv_lens`` counts the queries too.
        history_local_tokens: [max_num_seqs]. History tokens per rank.
    """
    max_num_seqs = kv_lens.shape[0]
    kv_lens = jnp.asarray(kv_lens, jnp.int32)
    cu_q_lens = jnp.asarray(cu_q_lens, jnp.int32)
    q_len_raw = cu_q_lens[1:] - cu_q_lens[:-1]
    active_num_reqs = jnp.clip(distribution[2], 0, max_num_seqs)
    req_active = jnp.logical_and(
        jnp.arange(max_num_seqs, dtype=jnp.int32) < active_num_reqs, q_len_raw > 0
    )
    q_len = jnp.where(req_active, q_len_raw, 0)
    history = jnp.where(req_active, kv_lens - q_len, 0)
    virtual_page = page_size * pcp_size
    # The partial last virtual page is the streaming current pass's
    # history-boundary group; only whole virtual pages belong here.
    history_local_tokens = (history // virtual_page) * page_size

    owner_start = cu_q_lens[:-1][:, None]
    ranks = jnp.arange(pcp_size, dtype=jnp.int32)[None, :]
    count = functools.partial(
        _rank_token_count_before, pcp_size=pcp_size, interleave_size=interleave_size
    )
    local_counts = count(owner_start + q_len[:, None], ranks) - count(
        owner_start, ranks
    )
    local_counts = jnp.where(req_active[:, None], local_counts, 0)  # [S, pcp]
    cu_q_lens_src = jnp.concatenate(
        (
            jnp.zeros((pcp_size, 1), jnp.int32),
            jnp.cumsum(local_counts.T, axis=1, dtype=jnp.int32),
        ),
        axis=1,
    )
    kv_lens_src = jnp.where(
        local_counts.T > 0, history_local_tokens[None, :] + local_counts.T, 0
    )
    return cu_q_lens_src, kv_lens_src.astype(jnp.int32), history_local_tokens


def merge_current_state_with_partials(m_state, l_state, acc_state, parts_o, parts_lse):
    """LSE merge of the PCP current-pass state with history partials.

    Args:
        m_state, l_state: [kv_heads, T, q_per_kv_padded, lanes] f32 running
            max and sum of the current pass (lane 0 is used).
        acc_state: [kv_heads, T, q_per_kv_padded, head_dim] f32 unnormalized
            accumulator of the current pass.
        parts_o: [P, T, num_q_heads, head_dim] normalized history partials.
        parts_lse: [P, T, num_q_heads] f32 LSE of each partial; -inf where a
            partial attended nothing.

    Returns:
        [T, kv_heads, q_per_kv_padded, head_dim] f32 merged output. Padded
        heads (q_per_kv_padded > q_per_kv) carry the current state only.
    """
    kv_heads, tokens, q_pad, head_dim = acc_state.shape
    num_parts, _, num_q_heads, _ = parts_o.shape
    q_per_kv = num_q_heads // kv_heads
    m = jnp.transpose(m_state[..., 0], (1, 0, 2)).astype(jnp.float32)  # [T,kv,qp]
    l = jnp.transpose(l_state[..., 0], (1, 0, 2)).astype(jnp.float32)
    acc = jnp.transpose(acc_state, (1, 0, 2, 3)).astype(jnp.float32)
    l_pos = l > 0
    lse_c = jnp.where(l_pos, m + jnp.log(jnp.where(l_pos, l, 1.0)), -jnp.inf)
    out_c = acc / jnp.where(l_pos, l, 1.0)[..., None]

    o = parts_o.reshape(num_parts, tokens, kv_heads, q_per_kv, head_dim).astype(
        jnp.float32
    )
    lse = parts_lse.reshape(num_parts, tokens, kv_heads, q_per_kv).astype(jnp.float32)
    if q_pad != q_per_kv:
        pad = ((0, 0), (0, 0), (0, 0), (0, q_pad - q_per_kv))
        o = jnp.pad(o, pad + ((0, 0),))
        lse = jnp.pad(lse, pad, constant_values=-jnp.inf)

    big = jnp.maximum(lse_c, jnp.max(lse, axis=0))
    big = jnp.where(jnp.isfinite(big), big, 0.0)
    w_c = jnp.exp(lse_c - big)  # exp(-inf) == 0
    w = jnp.exp(lse - big[None])
    # A partial that attended nothing has lse -inf and an undefined o (RPAm
    # leaves the aliased Q rows there).
    o = jnp.where((w > 0)[..., None], o, 0.0)
    num = w_c[..., None] * out_c + jnp.sum(w[..., None] * o, axis=0)
    den = w_c + jnp.sum(w, axis=0)
    return num / jnp.where(den > 0, den, 1.0)[..., None]


def rpa_q_no_alias_enabled() -> bool:
    """TPU_PCP_RPA_Q_NO_ALIAS=1 (default off): the PCP history / current RPAm calls on
    the merge-kernel path write a fresh o instead of aliasing q, and skip their empty
    DECODE call (their distribution is (0, 0, n)). The own and current calls share one
    relayouted q, so with aliasing XLA copied it once per layer (copy.N.q_hbm_ref).
    Rows a call does not write are undefined instead of q; the merge kernel zeroes
    every row whose partial attended nothing or that is not valid."""
    return os.environ.get("TPU_PCP_RPA_Q_NO_ALIAS", "0") == "1"


def rpa_q_no_alias_kwargs() -> dict:
    """For a call on the compact-LSE path: only the merge kernel's zeroing is relied on."""
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import merge_kernel

    if rpa_q_no_alias_enabled() and merge_kernel.merge_kernel_enabled():
        return {"alias_q": False, "skip_decode": True}
    return {}


def batched_rpa_history_fn(
    *,
    sm_scale: float,
    k_scale: float | None,
    v_scale: float | None,
    kv_layout,
    raw: bool = False,
    lse_compact: bool | None = None,
) -> Callable:
    """The production per-source history call: RPAm CACHE_ONLY, f32 LSE.

    With ``raw`` the call returns RPAm's own layouts (o [Hkv, T, G // p,
    p, D], lse [Hkv, T, G]) for ``merge_partials_native``. ``lse_compact``
    overrides TPU_PCP_LSE_COMPACT (False: the [Hkv, T, G] LSE even when the
    switch is on; the MTP final-rows subset call takes it).
    """
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs as rpa_configs
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as rpa_wrapper

    if kv_layout != rpa_configs.KVLayout.SEQ_ALONG_LANE:
        raise NotImplementedError(
            "TPU_PCP_HISTORY_RPA supports the SEQ_ALONG_LANE (HND) cache only: "
            "the PCP HEAD_ALONG_SUBLANE cache omits RPA's bank-conflict stride."
        )

    # TPU_PCP_HISTORY_RPA_BLOCKS="bq,bq_c,bkv,batch": the history calls' own
    # prefill tiles (CACHE_ONLY has no causal mask, so it can take other tiles
    # than the causal prefill TPU_RPA_PREFILL_BLOCKS serves). Empty: the
    # wrapper's choice (which honours TPU_RPA_PREFILL_BLOCKS).
    blocks_env = os.environ.get("TPU_PCP_HISTORY_RPA_BLOCKS", "").strip()
    blocks = None
    if blocks_env:
        bq, bq_c, bkv, batch = (int(x) for x in blocks_env.split(","))
        blocks = rpa_configs.BlockSizes(bq_sz=bq, bq_c_sz=bq_c, bkv_sz=bkv, batch_size=batch)

    compact = raw and (lse_compact_enabled() if lse_compact is None else bool(lse_compact))
    # TPU_PCP_RPA_Q_NO_ALIAS: the merge-kernel path (compact LSE) takes a fresh o (q stays
    # read-only, so the own and current calls can share one q) and no empty DECODE call.
    no_alias = rpa_q_no_alias_kwargs() if compact else {}

    def fn(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, num_seqs,
           schedule=None, schedule_only=False):
        # RPA stages new K/V in the dtype it is given and DMAs it into the
        # cache-dtype VMEM window, so it must already be in the cache dtype.
        # CACHE_ONLY without a KV update never reads it.
        # schedule_only: return this call's MIXED schedule leaves (to share
        # across layers); schedule: such leaves, used instead of building one.
        k = k.astype(kv_cache.dtype)
        v = v.astype(kv_cache.dtype)
        distribution = jnp.stack(
            (jnp.int32(0), jnp.int32(0), jnp.asarray(num_seqs, jnp.int32))
        )
        res = rpa_wrapper.ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_layout=kv_layout,
            attention_scope=rpa_configs.AttentionScope.CACHE_ONLY,
            return_lse=True,
            lse_dtype=jnp.float32,
            lse_compact=compact,
            skip_kv_update=True,
            prefill_block_sizes=blocks,
            raw_outputs=raw,
            schedule_only=schedule_only,
            mixed_schedule_leaves=schedule,
            **no_alias,
        )
        if schedule_only:
            return res
        o, kv_cache, lse = res
        if raw and not compact:
            return o, kv_cache, raw_lse(lse)
        return o, kv_cache, lse

    return fn


def history_partials_meta(
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    local_num_blocks: int,
    pcp_size: int,
    interleave_size: int,
    page_size: int,
):
    """Metadata of the history calls (shared by a forward pass's layers)."""
    max_num_seqs = kv_lens.shape[0]
    if page_indices.ndim == 1:
        block_tables = page_indices.reshape(max_num_seqs, -1)
    else:
        block_tables = page_indices
    local_page_indices = jnp.mod(
        block_tables.astype(jnp.int32), jnp.int32(local_num_blocks)
    ).reshape(-1)
    cu_q_lens_src, kv_lens_src, _ = history_rpa_metadata(
        kv_lens,
        cu_q_lens,
        distribution,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        page_size=page_size,
    )
    num_seqs = jnp.clip(distribution[2], 0, max_num_seqs).astype(jnp.int32)
    return dict(
        local_page_indices=local_page_indices,
        cu_q_lens_src=cu_q_lens_src,
        kv_lens_src=kv_lens_src,
        num_seqs=num_seqs,
    )


def mtp_final_rows_history_requested() -> bool:
    """TPU_PCP_MTP_FINAL_ROWS_HISTORY=1 (default off): the prefill-only MTP
    draft layer's history pass runs only for the query row at each request's
    last chunk position (``final_rows_meta``). The draft reads only that row's
    output (eagle3 gathers the last-token rows), and the KV write-back and the
    current pass are unchanged, so the draft layer's KV is unchanged."""
    return os.environ.get("TPU_PCP_MTP_FINAL_ROWS_HISTORY", "0").strip() == "1"


def final_rows_meta(kv_lens, cu_q_lens, distribution, cu_q_lens_src, kv_lens_src, *,
                    pcp_size: int, interleave_size: int):
    """History metadata for one query row per request: its last chunk position.

    Query rows are owned by their offset in the packed step stream (as in
    ``history_rpa_metadata``): request ``i``'s last row, stream offset
    ``cu_q_lens[i + 1] - 1``, lives on rank ``(offset // interleave) %
    pcp_size``, where it is the request's last local row,
    ``cu_q_lens_src[s, i + 1] - 1`` (a rank's rows of a request are its owned
    offsets in order).

    Returns a dict of [pcp_size, S] arrays (S = max_num_seqs): ``idx`` local
    rows of the source's final rows, compacted to the front in request order
    (0 past ``count``), ``valid`` (row j < count), ``cu`` [P, S + 1] and
    ``kv`` (RPAm cu_q_lens / kv_lens with one query per such request).
    """
    max_num_seqs = kv_lens.shape[0]
    kv_lens = jnp.asarray(kv_lens, jnp.int32)
    cu_q_lens = jnp.asarray(cu_q_lens, jnp.int32)
    q_len = cu_q_lens[1:] - cu_q_lens[:-1]
    seqs = jnp.arange(max_num_seqs, dtype=jnp.int32)
    active = jnp.logical_and(seqs < jnp.clip(distribution[2], 0, max_num_seqs), q_len > 0)
    owner = ((cu_q_lens[1:] - 1) // interleave_size) % pcp_size  # [S]
    ranks = jnp.arange(pcp_size, dtype=jnp.int32)[:, None]
    has = jnp.logical_and(active[None, :], owner[None, :] == ranks)  # [P, S]
    local_counts = cu_q_lens_src[:, 1:] - cu_q_lens_src[:, :-1]
    rows = cu_q_lens_src[:, 1:] - 1
    order = jnp.argsort(jnp.logical_not(has), axis=1, stable=True)
    count = jnp.sum(has, axis=1, dtype=jnp.int32)  # [P]
    valid = seqs[None, :] < count[:, None]
    idx = jnp.where(valid, jnp.take_along_axis(rows, order, axis=1), 0)
    cu = jnp.concatenate(
        (jnp.zeros((pcp_size, 1), jnp.int32), jnp.cumsum(has.astype(jnp.int32), axis=1)),
        axis=1,
    )
    kv = jnp.where(has, kv_lens_src - local_counts + 1, 0).astype(jnp.int32)
    return dict(idx=idx.astype(jnp.int32), valid=valid, cu=cu.astype(jnp.int32), kv=kv)


def _scatter_final_rows(o_sub, lse_sub, idx, valid, num_rows, *, lse_out_compact: bool):
    """Subset partials -> full-row partials: o 0 and lse -inf on every other row.

    Standard layouts ([S, H, D], [S, H]) or RPAm raw ([Hkv, S, G // p, p, D],
    [Hkv, S, Gp]); with ``lse_out_compact`` the raw LSE is returned lane-dense
    [Hkv, T * Gp // 128, 128], as TPU_PCP_LSE_COMPACT's calls write it.
    """
    rows = jnp.where(valid, idx, num_rows)  # out of range -> dropped
    if o_sub.ndim == 3:
        o = jnp.zeros((num_rows,) + o_sub.shape[1:], o_sub.dtype).at[rows].set(o_sub, mode="drop")
        lse = jnp.full((num_rows,) + lse_sub.shape[1:], -jnp.inf, jnp.float32).at[rows].set(
            lse_sub.astype(jnp.float32), mode="drop")
        return o, lse
    o = jnp.zeros(o_sub.shape[:1] + (num_rows,) + o_sub.shape[2:], o_sub.dtype).at[:, rows].set(
        o_sub, mode="drop")
    lse = jnp.full(lse_sub.shape[:1] + (num_rows,) + lse_sub.shape[2:], -jnp.inf, jnp.float32).at[
        :, rows].set(lse_sub.astype(jnp.float32), mode="drop")
    if lse_out_compact:
        lse = lse.reshape(lse.shape[0], num_rows * lse.shape[2] // 128, 128)
    return o, lse


def _pick_schedule(schedules, idx):
    """``schedule=`` kwargs with source ``idx``'s leaves from per-source
    stacked schedule leaves (none when nothing was precomputed)."""
    if schedules is None:
        return {}
    return dict(schedule=tuple(leaf[idx] for leaf in schedules))


def pcp_history_partials(
    q_local,
    k_local,
    v_local,
    kv_cache_local,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    pcp_size: int,
    interleave_size: int,
    page_size: int,
    pcp_axis_name: str,
    history_fn: Callable,
    rank=None,
    meta=None,
    schedules=None,
    final_rows: bool = False,
    lse_out_compact: bool = False,
):
    """History partials for this rank's queries, inside the PCP shard_map body.

    With ``final_rows`` (TPU_PCP_MTP_FINAL_ROWS_HISTORY, the MTP draft layer)
    only each request's last-position row is attended (``final_rows_meta``);
    every other row's partials are o 0 / lse -inf, which the merges ignore.
    ``history_fn`` must then return the non-compact LSE; ``lse_out_compact``
    re-packs the scattered LSE for a compact-LSE merge.

    ``meta`` (``history_partials_meta``) and ``schedules`` (per-source
    stacked MIXED schedule leaves, source = whose queries) may come
    precomputed once per forward pass (TPU_PCP_RPA_HOIST_META).

    Args:
        q_local: [T, num_q_heads, head_dim] this rank's compact Q rows.
        k_local, v_local: [T, kv_heads, head_dim]; only their shapes matter
            (the history call neither writes nor attends new tokens).
        kv_cache_local: this rank's KV cache shard.
        kv_lens, page_indices, cu_q_lens, distribution: replicated vLLM
            metadata, as given to the streaming kernel.
        history_fn: ``(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens,
            num_seqs) -> (o, kv_cache, lse)`` over this rank's local pages.

    Returns:
        (parts_o [P, T, H, D], parts_lse [P, T, H] f32, kv_cache), where part
        j is rank j's history pages attended by this rank's queries.
    """
    if meta is None:
        with jax.named_scope("pcp_hist_meta"):
            meta = history_partials_meta(
                kv_lens, page_indices, cu_q_lens, distribution,
                local_num_blocks=jax.tree.leaves(kv_cache_local)[0].shape[0],
                pcp_size=pcp_size, interleave_size=interleave_size,
                page_size=page_size,
            )
    local_page_indices = meta["local_page_indices"]
    cu_q_lens_src = meta["cu_q_lens_src"]
    kv_lens_src = meta["kv_lens_src"]
    num_seqs = meta["num_seqs"]

    if final_rows:
        return _final_rows_history_partials(
            q_local, k_local, v_local, kv_cache_local, kv_lens, cu_q_lens, distribution,
            cu_q_lens_src, kv_lens_src, local_page_indices, num_seqs,
            pcp_size=pcp_size, interleave_size=interleave_size,
            pcp_axis_name=pcp_axis_name, history_fn=history_fn, rank=rank,
            lse_out_compact=lse_out_compact,
        )

    if pcp_size == 2 and pair_exchange_enabled():
        return _pcp2_history_partials(
            q_local, k_local, v_local, kv_cache_local, kv_lens_src, cu_q_lens_src,
            local_page_indices, num_seqs, pcp_axis_name=pcp_axis_name,
            history_fn=history_fn, rank=rank, schedules=schedules,
        )

    with jax.named_scope("pcp_hist_q_allgather"):
        q_all = jax.lax.all_gather(q_local, pcp_axis_name)  # [P, T, H, D]

    parts_o, parts_lse = [], []
    kv = kv_cache_local
    for src in range(pcp_size):
        with jax.named_scope(f"pcp_hist_rpa_src{src}"):
            # The cache is threaded through every call, so each RPAm call
            # aliases it in place rather than XLA copying it.
            o, kv, lse = history_fn(
                q_all[src],
                k_local,
                v_local,
                kv,
                kv_lens_src[src],
                local_page_indices,
                cu_q_lens_src[src],
                num_seqs,
                **_pick_schedule(schedules, src),
            )
        parts_o.append(o)
        parts_lse.append(lse.astype(jnp.float32))

    with jax.named_scope("pcp_hist_o_return"):
        # parts[s] answers rank s's queries; after the all-to-all, row j
        # holds rank j's partial for this rank's queries.
        parts_o = jax.lax.all_to_all(
            jnp.stack(parts_o), pcp_axis_name, split_axis=0, concat_axis=0
        )
        parts_lse = jax.lax.all_to_all(
            jnp.stack(parts_lse), pcp_axis_name, split_axis=0, concat_axis=0
        )
    return parts_o, parts_lse, kv


def _final_rows_history_partials(
    q_local, k_local, v_local, kv_cache_local, kv_lens, cu_q_lens, distribution,
    cu_q_lens_src, kv_lens_src, local_page_indices, num_seqs, *, pcp_size,
    interleave_size, pcp_axis_name, history_fn, rank, lse_out_compact,
):
    """``pcp_history_partials`` for each request's last-position row only.

    Same routing as the full pass (pair exchange on PCP2, else all-gather /
    all-to-all), on [S]-row query subsets; the partials are scattered back to
    the local rows. Returns ([P] partial o, [P] lse, kv_cache) like the full
    pass, with every non-final row at o 0 / lse -inf.
    """
    fr = final_rows_meta(
        kv_lens, cu_q_lens, distribution, cu_q_lens_src, kv_lens_src,
        pcp_size=pcp_size, interleave_size=interleave_size,
    )
    me = rank if rank is not None else jax.lax.axis_index(pcp_axis_name)
    lt = q_local.shape[0]
    idx_me, valid_me = fr["idx"][me], fr["valid"][me]
    s_rows = idx_me.shape[0]
    with jax.named_scope("pcp_mtp_final_rows_gather"):
        q_sub = jnp.take(q_local, idx_me, axis=0)
    k_sub = jnp.zeros((s_rows,) + k_local.shape[1:], k_local.dtype)
    v_sub = jnp.zeros((s_rows,) + v_local.shape[1:], v_local.dtype)

    def call(q, kv_in, src):
        return history_fn(q, k_sub, v_sub, kv_in, fr["kv"][src], local_page_indices,
                          fr["cu"][src], num_seqs)

    if pcp_size == 2 and pair_exchange_enabled():
        swap = [(0, 1), (1, 0)]
        with jax.named_scope("pcp_hist_q_swap"):
            q_peer = jax.lax.ppermute(q_sub, pcp_axis_name, swap)
        with jax.named_scope("pcp_hist_rpa_own"):
            o_own, kv, lse_own = call(q_sub, kv_cache_local, me)
        with jax.named_scope("pcp_hist_rpa_peer"):
            o_pq, kv, lse_pq = call(q_peer, kv, 1 - me)
        with jax.named_scope("pcp_hist_o_swap"):
            o_back = jax.lax.ppermute(o_pq, pcp_axis_name, swap)
            lse_back = jax.lax.ppermute(lse_pq.astype(jnp.float32), pcp_axis_name, swap)
        subs = [(o_own, lse_own), (o_back, lse_back)]
    else:
        with jax.named_scope("pcp_hist_q_allgather"):
            q_all = jax.lax.all_gather(q_sub, pcp_axis_name)
        parts_o, parts_lse, kv = [], [], kv_cache_local
        for src in range(pcp_size):
            with jax.named_scope(f"pcp_hist_rpa_src{src}"):
                o, kv, lse = call(q_all[src], kv, src)
            parts_o.append(o)
            parts_lse.append(lse.astype(jnp.float32))
        with jax.named_scope("pcp_hist_o_return"):
            parts_o = jax.lax.all_to_all(jnp.stack(parts_o), pcp_axis_name, split_axis=0, concat_axis=0)
            parts_lse = jax.lax.all_to_all(jnp.stack(parts_lse), pcp_axis_name, split_axis=0, concat_axis=0)
        subs = [(parts_o[j], parts_lse[j]) for j in range(pcp_size)]
    outs_o, outs_lse = [], []
    with jax.named_scope("pcp_mtp_final_rows_scatter"):
        for o_sub, lse_sub in subs:
            o, lse = _scatter_final_rows(o_sub, lse_sub, idx_me, valid_me, lt,
                                         lse_out_compact=lse_out_compact)
            outs_o.append(o)
            outs_lse.append(lse)
    return outs_o, outs_lse, kv


def peer_first_enabled() -> bool:
    """TPU_PCP_RPA_PEER_FIRST=1 (with the pair exchange): run the peer's history
    call first, so its partial travels back during the own call."""
    return os.environ.get("TPU_PCP_RPA_PEER_FIRST", "0") == "1"


def _group_flag(name: str, *groups: str) -> bool:
    """``name``=1/0 decides; unset, any of the group switches set to 1 enables."""
    v = os.environ.get(name, "")
    if v in ("0", "1"):
        return v == "1"
    return any(os.environ.get(g, "0") == "1" for g in groups)


def meta_compare_all_enabled() -> bool:
    """TPU_PCP_META_COMPARE_ALL (group TPU_PCP_GLUE2): the current pass's
    slot-to-request lookup uses searchsorted(method="compare_all") instead of
    the default "scan" (a while loop over the 16,896 slots, ~0.25 ms per
    forward on gcsq B's xprof per the GDN agent). Exact either way."""
    return _group_flag("TPU_PCP_META_COMPARE_ALL", "TPU_PCP_GLUE2")


def boundary_after_write_enabled() -> bool:
    """TPU_PCP_BOUNDARY_AFTER_WRITE=1 (all-RPA, default off until measured):
    write the step's fresh KV into the cache first, then read the boundary
    pages from the written cache. The write only fills positions >= the
    request's history length, and the pool takes only boundary slots below it,
    so the values are the same; but now the boundary read depends on the write
    by data, nothing else reads the pre-write cache, and XLA has no reason to
    copy the pool (glb2r j-30f1b9aa: 4 full-pool copies despite the
    read-before-write barrier)."""
    return os.environ.get("TPU_PCP_BOUNDARY_AFTER_WRITE", "0") == "1"


def pool_kernel_raw_sources_enabled() -> bool:
    """TPU_PCP_POOL_KERNEL_RAW_SOURCES=1 (with the pool kernel): pass the raw
    all-gather outputs ([P, A, B, C, LT_pad] fresh KV, [P * S, A, B, C, page]
    boundary pages) instead of their token-last flattenings, so XLA does not
    relayout the kernel's inputs (gla3c: copy.fresh_hbm + reshape, ~10 ms per round)."""
    return os.environ.get("TPU_PCP_POOL_KERNEL_RAW_SOURCES", "0") == "1"


def cur_pool_kernel_enabled() -> bool:
    """TPU_PCP_CUR_POOL_KERNEL=1 (all-RPA, default off until measured): build
    the current pass's page pool with the run-copy Pallas kernel
    (pool_kernel.build_pool) instead of element-wise gathers plus relayouts.
    Any alignment; falls back to the element-wise build if a page needs more
    runs than the kernel's static bound."""
    return _group_flag("TPU_PCP_CUR_POOL_KERNEL", "TPU_PCP_GLUE2")


def kv_read_before_write_enabled() -> bool:
    """TPU_PCP_KV_READ_BEFORE_WRITE=1 (default off until measured): order the
    current pass's boundary-page read of the KV cache before the in-place
    write-back with an optimization barrier, so XLA never needs a defensive
    full-pool copy (all-RPA path)."""
    return os.environ.get("TPU_PCP_KV_READ_BEFORE_WRITE", "0") == "1"


def pair_exchange_enabled() -> bool:
    """TPU_PCP_RPA_PAIR_EXCHANGE=1 (default off until measured): at PCP2 the history pass swaps Q
    and the peer's partial with one ppermute each, instead of all-gathering Q
    and all-to-all-ing a stacked [P, ...] copy of the partials."""
    return os.environ.get("TPU_PCP_RPA_PAIR_EXCHANGE", "0") == "1"


def q_pack_outside_enabled() -> bool:
    """TPU_PCP_Q_PACK_OUTSIDE=1 (all-RPA PCP2, default off): the entry packs q into RPAm's
    kv-major [Hkv, T, G/p, p, D] before the shard_map and the pair exchange swaps that packed
    buffer, so XLA folds every q permutation (with TPU_QK_NORM_ROPE_Q_PACKED none is left).
    Inside the shard_map q is unpacked back to [T, H, D] for the unchanged calls: their own
    pack (batched_rpa prepare_inputs) folds against it. Pure layout, values identical."""
    return os.environ.get("TPU_PCP_Q_PACK_OUTSIDE", "0") == "1"


def q_pack(q, num_kv_heads: int, packing: int):
    """[T, H, D] -> RPAm's packed kv-major [Hkv, T, G / p, p, D] (prepare_inputs' order)."""
    t, h, d = q.shape
    qp = q.reshape(t, num_kv_heads, h // num_kv_heads // packing, packing, d).swapaxes(0, 1)
    if os.environ.get("TPU_PCP_Q_PACK_MUTANT", "0") == "1":  # tests only: kv heads swapped
        qp = qp[::-1]
    return qp


def q_unpack(qp):
    """``q_pack``'s inverse: [Hkv, T, G / p, p, D] -> [T, H, D]."""
    hkv, t, g2, p, d = qp.shape
    return qp.swapaxes(0, 1).reshape(t, hkv * g2 * p, d)


def _pcp2_history_partials(
    q_local, k_local, v_local, kv_cache_local, kv_lens_src, cu_q_lens_src,
    local_page_indices, num_seqs, *, pcp_axis_name, history_fn, rank=None,
    schedules=None,
):
    """PCP2 history partials with pairwise exchanges (no stacked copies).

    Returns ([own, from_peer] partial outputs, matching LSEs, kv_cache): own is
    this rank's queries over its local pages; from_peer is this rank's queries
    over the peer's pages, computed on the peer and sent back.
    """
    # The rank comes in as data (the wrapper's replicated selection). Under the
    # torch_tpu export, lax.axis_index lowers to partition-id, which the SPMD
    # partitioner refuses (DRA2 run03 j-fdeba0e9). Standalone JAX may use it.
    me = rank if rank is not None else jax.lax.axis_index(pcp_axis_name)
    swap = [(0, 1), (1, 0)]
    with jax.named_scope("pcp_hist_q_swap"):
        if q_pack_outside_enabled() and q_local.dtype.itemsize == 2:
            # Swap the packed buffer (q_local is the entry's unpack of it, so pack folds away)
            # and unpack on arrival; the peer call's own pack folds against that unpack.
            q_peer = q_unpack(jax.lax.ppermute(q_pack(q_local, k_local.shape[1], 2), pcp_axis_name, swap))
        else:
            q_peer = jax.lax.ppermute(q_local, pcp_axis_name, swap)
    def own(kv_in):
        with jax.named_scope("pcp_hist_rpa_own"):
            return history_fn(
                q_local, k_local, v_local, kv_in, kv_lens_src[me],
                local_page_indices, cu_q_lens_src[me], num_seqs,
                **_pick_schedule(schedules, me),
            )

    def peer(kv_in):
        with jax.named_scope("pcp_hist_rpa_peer"):
            return history_fn(
                q_peer, k_local, v_local, kv_in, kv_lens_src[1 - me],
                local_page_indices, cu_q_lens_src[1 - me], num_seqs,
                **_pick_schedule(schedules, 1 - me),
            )

    def send_back(o_pq, lse_pq):
        with jax.named_scope("pcp_hist_o_swap"):
            return (jax.lax.ppermute(o_pq, pcp_axis_name, swap),
                    jax.lax.ppermute(lse_pq.astype(jnp.float32), pcp_axis_name, swap))

    if peer_first_enabled():
        # The peer's partial is sent back while this rank's own history call
        # runs, instead of after both calls with nothing left to overlap.
        o_pq, kv, lse_pq = peer(kv_cache_local)
        o_back, lse_back = send_back(o_pq, lse_pq)
        o_own, kv, lse_own = own(kv)
    else:
        o_own, kv, lse_own = own(kv_cache_local)
        o_pq, kv, lse_pq = peer(kv)
        o_back, lse_back = send_back(o_pq, lse_pq)
    return [o_own, o_back], [lse_own.astype(jnp.float32), lse_back], kv


def pcp_history_attention(
    q_local,
    k_local,
    v_local,
    kv_cache_local,
    m_state,
    l_state,
    acc_state,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    pcp_size: int,
    interleave_size: int,
    page_size: int,
    pcp_axis_name: str,
    history_fn: Callable,
    rank=None,
):
    """History partials merged with the streaming current pass's state.

    Returns:
        (output [T, kv_heads, q_per_kv_padded, head_dim] f32, kv_cache).
    """
    parts_o, parts_lse, kv = pcp_history_partials(
        q_local,
        k_local,
        v_local,
        kv_cache_local,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        page_size=page_size,
        pcp_axis_name=pcp_axis_name,
        history_fn=history_fn,
        rank=rank,
    )
    if isinstance(parts_o, list):
        parts_o, parts_lse = jnp.stack(parts_o), jnp.stack(parts_lse)
    with jax.named_scope("pcp_hist_merge"):
        out = merge_current_state_with_partials(
            m_state, l_state, acc_state, parts_o, parts_lse
        )
    return out, kv


def merge_partials(parts_o, parts_lse):
    """LSE merge of normalized partials.

    Args:
        parts_o: [P, T, H, D]; parts_lse: [P, T, H] f32 (-inf: attended
            nothing, its o is ignored).

    Returns:
        [T, H, D] f32.
    """
    lse = parts_lse.astype(jnp.float32)
    big = jnp.max(lse, axis=0)
    big = jnp.where(jnp.isfinite(big), big, 0.0)
    w = jnp.exp(lse - big[None])
    o = jnp.where((w > 0)[..., None], parts_o.astype(jnp.float32), 0.0)
    num = jnp.sum(w[..., None] * o, axis=0)
    den = jnp.sum(w, axis=0)
    return num / jnp.where(den > 0, den, 1.0)[..., None]


def merge_partials_fused(parts_o, parts_lse, out_dtype):
    """LSE merge of a list of partials in one elementwise pass.

    Same result as ``merge_partials`` but takes the partials as separate
    arrays (no stacked copy), keeps the weights in f32 and writes the output
    directly in ``out_dtype``, so XLA can emit a single fusion that reads each
    partial once.

    Args:
        parts_o: sequence of [T, H, D] normalized partials (any float dtype).
        parts_lse: sequence of [T, H] LSEs (-inf: attended nothing).
        out_dtype: dtype of the merged output.

    Returns:
        [T, H, D] in ``out_dtype``.
    """
    lses = [x.astype(jnp.float32) for x in parts_lse]
    big = functools.reduce(jnp.maximum, lses)
    big = jnp.where(jnp.isfinite(big), big, 0.0)
    num = None
    den = None
    for o, lse in zip(parts_o, lses):
        w = jnp.exp(lse - big)
        term = jnp.where((w > 0)[..., None], w[..., None] * o.astype(jnp.float32), 0.0)
        num = term if num is None else num + term
        den = w if den is None else den + w
    return (num / jnp.where(den > 0, den, 1.0)[..., None]).astype(out_dtype)


def raw_merge_enabled() -> bool:
    """TPU_PCP_RPA_RAW_MERGE=1 (all-RPA only, default off until measured): the
    RPAm calls hand back their own output/LSE buffers and one fused merge
    reads them in place, applies the valid-row mask and writes [T, H, D].
    TPU_PCP_MERGE_KERNEL=1 implies it (the Pallas merge reads the same raw
    buffers and also writes invalid rows as zeros)."""
    return (
        os.environ.get("TPU_PCP_RPA_RAW_MERGE", "0") == "1"
        or os.environ.get("TPU_PCP_MERGE_KERNEL", "0") == "1"
    )


def lse_compact_enabled() -> bool:
    """TPU_PCP_LSE_COMPACT=1 (with TPU_PCP_MERGE_KERNEL; default off): the RPAm
    calls write a lane-dense LSE [Hkv, T * Gp // 128, 128] themselves
    (batched_rpa lse_compact), so neither the 128-lane LSE buffer's -inf init
    nor its lane-0 extraction runs, and the merge kernel reads it as is."""
    return (
        os.environ.get("TPU_PCP_LSE_COMPACT", "0") == "1"
        and os.environ.get("TPU_PCP_MERGE_KERNEL", "0") == "1"
    )


def raw_lse(lse_raw):
    """[Hkv, T, Gp, lanes] RPAm LSE buffer -> [Hkv, T, Gp] (lane 0).

    With TPU_RPA_LSE_LANE0_BARRIER the buffer is read behind an optimization
    barrier, as batched_rpa's standard_from_raw does: otherwise layout
    assignment relayouts the whole 128-lane buffer to slice lane 0 (AOT tpu7x,
    4 layers, TPU_PCP_MERGE_KERNEL: 4 f32[2,4352,16,128] copies). The barrier is
    an identity."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as rpa_wrapper

    if rpa_wrapper.lse_lane0_barrier_enabled():
        lse_raw = jax.lax.optimization_barrier(lse_raw)
    return lse_raw[..., 0].astype(jnp.float32)


def raw_from_standard(o, lse, num_kv_heads, q_packing):
    """Standard ([T, H, D], [T, H]) partial -> the raw layouts
    ([Hkv, T, G // q_packing, q_packing, D], [Hkv, T, G]); the inverse of
    batched_rpa's prepare_outputs + LSE extraction when G % q_packing == 0
    and D needs no lane padding."""
    t, h, d = o.shape
    g = h // num_kv_heads
    o_raw = jnp.swapaxes(o.reshape(t, num_kv_heads, g // q_packing, q_packing, d), 0, 1)
    lse_raw = jnp.swapaxes(lse.reshape(t, num_kv_heads, g), 0, 1)
    return o_raw, lse_raw


def merge_partials_native(parts_o, parts_lse, row_valid, out_dtype):
    """LSE merge of partials in RPAm's own layout, in one elementwise pass.

    Args:
        parts_o: sequence of [Hkv, T, G // p, p, D] normalized partials.
        parts_lse: sequence of [Hkv, T, G] LSEs (-inf: attended nothing).
        row_valid: [T] bool; invalid rows are written as zeros.
        out_dtype: dtype of the merged output.

    Returns:
        [T, Hkv * G, D] in ``out_dtype`` (head h = kv * G + g, as RPAm's
        prepare_inputs packs it).
    """
    hkv, t, g2, pk, d = parts_o[0].shape
    lses = [x.astype(jnp.float32).reshape(hkv, t, g2, pk) for x in parts_lse]
    big = functools.reduce(jnp.maximum, lses)
    big = jnp.where(jnp.isfinite(big), big, 0.0)
    num = None
    den = None
    for o, lse in zip(parts_o, lses):
        w = jnp.exp(lse - big)
        term = jnp.where((w > 0)[..., None], w[..., None] * o.astype(jnp.float32), 0.0)
        num = term if num is None else num + term
        den = w if den is None else den + w
    out = num / jnp.where(den > 0, den, 1.0)[..., None]  # [Hkv, T, G2, p, D]
    out = jnp.swapaxes(out.reshape(hkv, t, g2 * pk, d), 0, 1)  # [T, Hkv, G, D]
    out = jnp.where(row_valid[:, None, None, None], out, 0.0)
    return out.reshape(t, hkv * g2 * pk, d).astype(out_dtype)


def _flat_tokens(x):
    """[P, A, B, C, N] -> [A, B, C, P * N] (token axis last, rank-major)."""
    return jnp.moveaxis(x, 0, 3).reshape(x.shape[1:4] + (x.shape[0] * x.shape[4],))


def meta_onehot_enabled() -> bool:
    """TPU_PCP_META_ONEHOT=1 (default off): the index metadata replaces its slow s32 gathers /
    scatters over small per-request (or per-page run) tables with one-hot select-reduces:
    the query row mapping's per-row request lookups, current_rpa_meta's local_starts[req, rank]
    and pool_runs' per-page run compaction. Integer selects with exactly one (or no) selected
    term, so the results are identical; a gather from a 32-entry table took ~40-90 us on TPU."""
    return os.environ.get("TPU_PCP_META_ONEHOT", "0") == "1"


def onehot_take(table, idx):
    """``table[idx]`` for a small 1-D integer table and in-range ``idx``, without a gather."""
    oh = idx[..., None] == jnp.arange(table.shape[0], dtype=idx.dtype)
    return jnp.sum(jnp.where(oh, table, jnp.zeros((), table.dtype)), axis=-1, dtype=table.dtype)


def onehot_take2(table, i, j):
    """``table[i, j]`` for a small 2-D integer table and in-range 1-D ``i``, ``j``."""
    rows = jnp.sum(
        jnp.where((i[:, None] == jnp.arange(table.shape[0], dtype=i.dtype))[..., None], table[None], 0),
        axis=1, dtype=table.dtype,
    )  # [N, table.shape[1]]
    return jnp.sum(jnp.where(j[:, None] == jnp.arange(table.shape[1], dtype=j.dtype), rows, 0), axis=1,
                   dtype=table.dtype)


def current_rpa_meta(
    kv_lens,
    cu_q_lens,
    distribution,
    block_tables_mod,
    local_positions,
    local_requests,
    local_valid,
    *,
    lt_pad: int,
    pcp_size: int,
    interleave_size: int,
    page_size: int,
):
    """The index (metadata-only) half of ``current_rpa_inputs``.

    Depends only on the step's metadata, so layers of a forward pass can
    share it (TPU_PCP_RPA_HOIST_META).

    Returns:
        dict: ``page_ids`` [S] boundary page per request, ``is_b`` /
        ``valid`` / ``b_take`` / ``c_take`` [NTP * page] pool-slot selectors
        and gather indices, and the RPA call's ``kv_lens``, ``cu_q_lens``,
        ``page_indices`` (flat) and ``num_seqs``.
    """
    if page_major_fresh_kv_enabled() and page_size % interleave_size != 0:
        # Page-major fresh KV: the pool kernel reads fresh runs from one-page rows. A
        # fresh run stays inside one interleave chunk of the rank-major columns (the
        # next token is on another rank), so no run crosses a row when the page is a
        # multiple of the interleave; other shapes are refused rather than split.
        raise ValueError(
            f"TPU_PCP_PAGE_MAJOR_FRESH_KV needs page_size ({page_size}) to be a multiple of "
            f"the interleave ({interleave_size}).")
    max_seqs = kv_lens.shape[0]
    lt = local_positions.shape[0]
    vpage = pcp_size * page_size
    count = functools.partial(
        _rank_token_count_before, pcp_size=pcp_size, interleave_size=interleave_size
    )

    kv_lens = jnp.asarray(kv_lens, jnp.int32)
    cu = jnp.asarray(cu_q_lens, jnp.int32)
    q_len_raw = cu[1:] - cu[:-1]
    active = jnp.logical_and(
        jnp.arange(max_seqs, dtype=jnp.int32) < jnp.clip(distribution[2], 0, max_seqs),
        q_len_raw > 0,
    )
    q_len = jnp.where(active, q_len_raw, 0)
    hist = jnp.where(active, kv_lens - q_len, 0)
    h_full = (hist // vpage) * vpage
    bnd = hist - h_full
    ranks = jnp.arange(pcp_size, dtype=jnp.int32)[None, :]
    local_counts = jnp.where(
        active[:, None],
        count(cu[:-1][:, None] + q_len[:, None], ranks) - count(cu[:-1][:, None], ranks),
        0,
    )
    local_starts = jnp.cumsum(local_counts, axis=0) - local_counts  # [S, P]
    boundary_page = jnp.clip(hist // vpage, 0, block_tables_mod.shape[1] - 1)
    page_ids = jnp.take_along_axis(block_tables_mod, boundary_page[:, None], axis=1)[:, 0]

    # Pool layout: request i owns pages [pstart_i, pstart_i + npages_i).
    seg_len = bnd + q_len
    npages = jnp.where(active, (seg_len + page_size - 1) // page_size, 0)
    pstart = jnp.cumsum(npages) - npages
    ntp = -(-(pcp_size * lt) // page_size) + 2 * max_seqs
    slots = jnp.arange(ntp * page_size, dtype=jnp.int32)
    slot_page = slots // page_size
    search_kw = dict(method="compare_all") if meta_compare_all_enabled() else {}
    req = jnp.clip(
        jnp.searchsorted(pstart, slot_page, side="right", **search_kw) - 1, 0, max_seqs - 1
    )
    r = slots - pstart[req] * page_size
    valid = jnp.logical_and(active[req], jnp.logical_and(r >= 0, r < seg_len[req]))
    is_b = r < bnd[req]
    # Boundary token: global position h_full + r on rank (pos // il) % P.
    bpos = h_full[req] + r
    brank = (bpos // interleave_size) % pcp_size
    boff = count(bpos, brank) - count(h_full[req], brank)
    bidx = (brank * max_seqs + req) * page_size + jnp.clip(boff, 0, page_size - 1)
    # Fresh token: batch-flat owner coordinate cu_i + offset picks the rank.
    coord = cu[:-1][req] + (r - bnd[req])
    crank = (coord // interleave_size) % pcp_size
    lstart = onehot_take2(local_starts, req, crank) if meta_onehot_enabled() else local_starts[req, crank]
    crow = lstart + count(coord, crank) - count(cu[:-1][req], crank)
    cidx = crank * lt_pad + jnp.clip(crow, 0, lt_pad - 1)

    # One RPA sequence per run of consecutive local rows (<= one interleave).
    prev_pos = jnp.concatenate((jnp.full((1,), -2, jnp.int32), local_positions[:-1]))
    prev_req = jnp.concatenate((jnp.full((1,), -1, jnp.int32), local_requests[:-1]))
    starts = jnp.logical_and(
        local_valid,
        jnp.logical_or(prev_req != local_requests, prev_pos + 1 != local_positions),
    )
    nch_max = lt // interleave_size + 2 * max_seqs + 1
    n_valid = jnp.sum(local_valid.astype(jnp.int32))
    nch = jnp.sum(starts.astype(jnp.int32))
    chunk_rows = jnp.nonzero(starts, size=nch_max, fill_value=lt)[0].astype(jnp.int32)
    j = jnp.arange(nch_max, dtype=jnp.int32)
    first = jnp.where(j < nch, chunk_rows, n_valid)
    cu_ch = jnp.concatenate((first, n_valid[None])).astype(jnp.int32)
    last = jnp.clip(cu_ch[1:] - 1, 0, lt - 1)
    ch_req = jnp.clip(local_requests[jnp.clip(first, 0, lt - 1)], 0, max_seqs - 1)
    ch_kv = bnd[ch_req] + (local_positions[last] - hist[ch_req]) + 1
    ch_kv = jnp.where(j < nch, ch_kv, 0).astype(jnp.int32)
    width = -(-(vpage + pcp_size * lt) // page_size) + 1
    ch_pages = jnp.clip(pstart[ch_req][:, None] + jnp.arange(width)[None, :], 0, ntp - 1)
    out = dict(
        page_ids=page_ids.astype(jnp.int32),
        is_b=is_b,
        valid=valid,
        b_take=jnp.where(is_b, bidx, 0).astype(jnp.int32),
        c_take=jnp.where(is_b, 0, cidx).astype(jnp.int32),
        kv_lens=ch_kv,
        cu_q_lens=cu_ch,
        page_indices=ch_pages.reshape(-1).astype(jnp.int32),
        num_seqs=nch.astype(jnp.int32),
    )
    if cur_pool_kernel_enabled():
        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import pool_kernel

        out["pool_runs"], out["pool_overflow"] = pool_kernel.pool_runs(
            out["is_b"], out["valid"], out["b_take"], out["c_take"], ntp=ntp,
            page_size=page_size,
            k_max=pool_kernel.max_runs_per_page(page_size, interleave_size),
            onehot=meta_onehot_enabled(),
        )
    return out


def page_major_fresh_kv_enabled() -> bool:
    """TPU_PCP_PAGE_MAJOR_FRESH_KV=1 (all-RPA, default off): the packed fresh KV is
    laid out page-major [LT_pad / page, A, B, C, page] before the all-gather, so
    the KV write kernel and the pool kernel copy whole contiguous pages instead of
    token-last strided slices. Needs the pool kernel with raw sources, the direct
    capture write and the boundary-after-write order. Pure data movement: the cache
    and the pool are byte-identical to the token-last path."""
    return os.environ.get("TPU_PCP_PAGE_MAJOR_FRESH_KV", "0") == "1"


def kv_gather_rows_enabled() -> bool:
    """TPU_PCP_KV_GATHER_ROWS=1 (all-RPA, default off): the fresh-KV and boundary-page
    all-gathers move [N, A, B * C, page] rows instead of [N, A, B, C, page] pages. Row
    b * C + c keeps the packed fp8 cache's byte order (the C values of a column share one
    32-bit word), so the gather can run in the cache's own layout instead of a B-packed
    one, which needed a relayout copy before and after each all-gather. Pure data
    movement: the gathered values are identical."""
    return os.environ.get("TPU_PCP_KV_GATHER_ROWS", "0") == "1"


def _all_gather_rows(x, axis_name):
    """``jax.lax.all_gather(x, axis_name)`` for x = [N, A, B, C, page], through merged rows."""
    n, a, b, c, pg = x.shape
    g = jax.lax.all_gather(x.reshape(n, a, b * c, pg), axis_name)
    return g.reshape((g.shape[0], n, a, b, c, pg))


def to_page_major(packed, page_size):
    """[A, B, C, N] token-last -> [N / page, A, B, C, page] (a permutation)."""
    a, b, c, n = packed.shape
    return jnp.transpose(packed.reshape(a, b, c, n // page_size, page_size), (3, 0, 1, 2, 4))


def from_page_major_rows(g_pm, page_size):
    """[P, NPG, A, B, C, page] -> the raw all-gather layout [P, A, B, C, NPG * page]."""
    p, npg, a, b, c, pg = g_pm.shape
    return jnp.transpose(g_pm, (0, 2, 3, 4, 1, 5)).reshape(p, a, b, c, npg * pg)


def current_rpa_build(meta, packed_current_kv, kv_cache_local, *, pcp_size, page_size, pcp_axis_name,
                      write_after_gather=None, raw_write_source=False):
    """The data half of ``current_rpa_inputs``: gather all ranks' fresh KV and
    the boundary pages and fill the temporary page pool from ``meta``.

    ``write_after_gather(gathered) -> written cache`` (TPU_PCP_BOUNDARY_AFTER_WRITE):
    the cache write-back runs first and the boundary pages are read from its
    result; the returned ``cache`` is then the written cache. With
    ``raw_write_source`` it receives the raw all-gather [P, A, B, C, LT_pad]
    instead of the token-last ``gathered`` (TPU_PCP_CAPTURE_DIRECT + raw sources).
    """
    max_seqs = meta["page_ids"].shape[0]
    ntp = meta["is_b"].shape[0] // page_size
    pm = page_major_fresh_kv_enabled()
    if pm and not (write_after_gather is not None and raw_write_source and "pool_runs" in meta
                   and pool_kernel_raw_sources_enabled()):
        raise ValueError(
            "TPU_PCP_PAGE_MAJOR_FRESH_KV needs TPU_PCP_BOUNDARY_AFTER_WRITE, TPU_PCP_CAPTURE_DIRECT, "
            "TPU_PCP_CUR_POOL_KERNEL and TPU_PCP_POOL_KERNEL_RAW_SOURCES."
        )
    with jax.named_scope("pcp_cur_kv_allgather"):
        if pm:
            # Page rows [P * NPG, A, B, C, page]: source column s is row s // page,
            # lane s % page, the same flat rank-major column as the token-last stack.
            pm_kv = to_page_major(packed_current_kv, page_size)
            g_pm = (
                _all_gather_rows(pm_kv, pcp_axis_name)
                if kv_gather_rows_enabled()
                else jax.lax.all_gather(pm_kv, pcp_axis_name)
            )
            g_raw = g_pm.reshape((-1,) + g_pm.shape[2:])
            gathered = None  # only the (rare) pool-overflow fallback needs it; built there
        else:
            g_raw = jax.lax.all_gather(packed_current_kv, pcp_axis_name)  # [P, A, B, C, LT_pad]
            gathered = _flat_tokens(g_raw)
    if write_after_gather is not None:
        with jax.named_scope("pcp_cur_writeback"):
            kv_cache_local = write_after_gather(g_raw if raw_write_source else gathered)
    with jax.named_scope("pcp_cur_kv_allgather"):
        bpages = jnp.take(kv_cache_local, meta["page_ids"], axis=0)  # [S, A, B, C, page]
        if write_after_gather is not None:
            # The written cache now feeds the boundary gather and the history
            # RPA calls, which alias the cache in place. Order the gather
            # before them by data, or XLA copies the pool for one of the two
            # (gla2c j-99f11fa5: 8 full-pool copies).
            bpages, kv_cache_local = jax.lax.optimization_barrier(
                (bpages, kv_cache_local)
            )
        elif kv_read_before_write_enabled():
            # The boundary-page read must finish before the in-place cache
            # write-back. Nothing in the data flow orders the two, so the
            # scheduler may put the read after the write; XLA then copies the
            # whole (shared KV/GDN) pool twice per layer (gatec j-06c302b8:
            # 8 x f8[47668,4,64,4,128] copies, 64 ms per step). Passing the
            # cache through a barrier together with the read makes the write
            # depend on the read.
            bpages, kv_cache_local = jax.lax.optimization_barrier(
                (bpages, kv_cache_local)
            )
        ball = (
            _all_gather_rows(bpages, pcp_axis_name)
            if kv_gather_rows_enabled()
            else jax.lax.all_gather(bpages, pcp_axis_name)
        ).reshape(
            (pcp_size * max_seqs,) + bpages.shape[1:]
        )  # [P * S, A, B, C, page], index rank * S + request
        bflat = jnp.moveaxis(ball, 0, 3).reshape(
            ball.shape[1:4] + (pcp_size * max_seqs * page_size,)
        )
    with jax.named_scope("pcp_cur_pool_build"):
        if "pool_runs" in meta:
            from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import pool_kernel

            pool = jax.lax.cond(
                meta["pool_overflow"],
                lambda: _element_pool_tokens(
                    meta,
                    _flat_tokens(from_page_major_rows(g_pm, page_size)) if pm else gathered,
                    bflat, ntp=ntp, page_size=page_size,
                ),
                lambda: (
                    # TPU_PCP_POOL_KERNEL_RAW_SOURCES: hand the kernel the
                    # all-gather outputs as row stacks (no token-last relayout).
                    pool_kernel.build_pool(meta["pool_runs"], g_raw, ball, page_size=page_size)
                    if pool_kernel_raw_sources_enabled()
                    else pool_kernel.build_pool(
                        meta["pool_runs"], gathered, bflat, page_size=page_size
                    )
                ),
            )
        elif row_gather_enabled():
            # Token-major rows: each gather index fetches one token's whole
            # [A, B, C] row instead of single lanes of the token-last layout.
            g_rows = _token_major(gathered)  # [P * LT_pad, A, B, C]
            b_rows = jnp.moveaxis(ball, -1, 1).reshape(
                (pcp_size * max_seqs * page_size,) + ball.shape[1:4]
            )
            sel = meta["is_b"][:, None, None, None]
            tok_rows = jnp.where(
                sel,
                jnp.take(b_rows, meta["b_take"], axis=0),
                jnp.take(g_rows, meta["c_take"], axis=0),
            )
            tok_rows = jnp.where(
                meta["valid"][:, None, None, None], tok_rows, jnp.zeros((), tok_rows.dtype)
            )
            pool = jnp.moveaxis(
                tok_rows.reshape((ntp, page_size) + tok_rows.shape[1:]), 1, -1
            )  # [NTP, A, B, C, page]
        else:
            pool = _element_pool_tokens(meta, gathered, bflat, ntp=ntp, page_size=page_size)
    return dict(
        pool=pool,
        kv_lens=meta["kv_lens"],
        cu_q_lens=meta["cu_q_lens"],
        page_indices=meta["page_indices"],
        num_seqs=meta["num_seqs"],
        gathered=gathered,
        g_raw=g_raw,
        cache=kv_cache_local,
    )


def _element_pool_tokens(meta, gathered, bflat, *, ntp, page_size):
    """The default element-wise pool: every slot gathered from the boundary
    pages or the gathered fresh KV, invalid slots zero."""
    tok = jnp.where(
        meta["is_b"],
        jnp.take(bflat, meta["b_take"], axis=-1),
        jnp.take(gathered, meta["c_take"], axis=-1),
    )
    tok = jnp.where(meta["valid"], tok, jnp.zeros((), tok.dtype))
    return jnp.moveaxis(tok.reshape(tok.shape[:3] + (ntp, page_size)), 3, 0)  # [NTP, A, B, C, page]


def current_rpa_inputs(
    packed_current_kv,
    kv_cache_local,
    kv_lens,
    cu_q_lens,
    distribution,
    block_tables_mod,
    local_positions,
    local_requests,
    local_valid,
    *,
    pcp_size: int,
    interleave_size: int,
    page_size: int,
    pcp_axis_name: str,
):
    """Build the causal current-chunk attention problem for batched RPA.

    Every rank gathers all ranks' fresh KV and the history-boundary pages, and
    lays out, per request, the global-order tokens ``[H_full, H + q)``
    (boundary partial virtual page + current chunk) on a small temporary page
    pool. Each run of consecutive local query rows (one interleave chunk) is
    one RPA sequence whose KV is its request's pool pages up to its last row,
    so RPA's own causal mask (queries are the last q_len KV positions) is the
    request's causal mask.

    Args:
        packed_current_kv: [A, B, C, LT_pad] this rank's fresh KV in the
            cache's packed layout (token axis last), rank-major rows.
        kv_cache_local: [num_blocks, A, B, C, page] this rank's cache.
        block_tables_mod: [S, W] block table reduced mod num_blocks.
        local_positions, local_requests, local_valid: [LT] request-absolute
            position, request index and validity of this rank's Q rows.

    Returns:
        dict with ``pool`` [NTP, A, B, C, page], ``kv_lens``, ``cu_q_lens``,
        ``page_indices`` (flat), ``num_seqs`` for the RPA call, and
        ``gathered`` [A, B, C, P * LT_pad] (all fresh KV, rank-major).
    """
    meta = current_rpa_meta(
        kv_lens, cu_q_lens, distribution, block_tables_mod,
        local_positions, local_requests, local_valid,
        lt_pad=packed_current_kv.shape[-1], pcp_size=pcp_size,
        interleave_size=interleave_size, page_size=page_size,
    )
    return current_rpa_build(
        meta, packed_current_kv, kv_cache_local, pcp_size=pcp_size,
        page_size=page_size, pcp_axis_name=pcp_axis_name,
    )


def capture_index(capture_offsets, owner_ranks, row_valid, cache_rank, *, lt, lt_pad):
    """The index half of ``captured_kv_from_gathered``: (flat source column
    in the padded rank-major stream, has-a-source) per capture slot."""
    g_rows = capture_offsets.shape[0]
    rows = jnp.arange(g_rows, dtype=jnp.int32)
    owned = jnp.logical_and(row_valid, owner_ranks == cache_rank)
    dst = jnp.where(owned, capture_offsets, g_rows)
    src_of_dst = jnp.full((g_rows,), -1, jnp.int32).at[dst].set(rows, mode="drop")
    has = src_of_dst >= 0
    src = jnp.where(has, src_of_dst, 0)
    flat = (src // lt) * lt_pad + src % lt
    return flat.astype(jnp.int32), has


def capture_direct_enabled() -> bool:
    """TPU_PCP_CAPTURE_DIRECT=1 (all-RPA, default off until measured): write
    the step's fresh KV into the cache straight from the gathered stream, with
    the write-back segments split wherever the gathered column breaks, instead
    of first gathering a dense capture buffer (gla3c: the capture gather and its
    relayouts, ~18-25 ms per round)."""
    return os.environ.get("TPU_PCP_CAPTURE_DIRECT", "0") == "1"


_DWS_END_BITS = 14


def dws_cummax_enabled() -> bool:
    """TPU_PCP_DWS_CUMMAX=1 (default off): ``direct_write_segments`` takes each
    capture row's segment end from one packed (segment id, end) int32 key
    propagated by cummax, and its destination from one gather of dst0 - src0,
    instead of three s32[R] gathers from the [D] descriptor tables (census #5a:
    ~3.7 ms/round of gathers). Same int32 results; needs R < 2**14 and
    D < 2**17 (static shapes; otherwise the original path runs)."""
    return os.environ.get("TPU_PCP_DWS_CUMMAX", "0") == "1"


def direct_write_segments(descriptors, num_segments, flat, has, row_width=None):
    """Re-express capture-buffer write-back segments in gathered-stream columns.

    descriptors [3, D] = (capture row start, cache slot start, length) over the
    dense capture buffer, the first ``num_segments`` valid and in row order;
    capture row k holds gathered column ``flat[k]`` when ``has[k]``. Returns
    ([3, R] descriptors whose sources are gathered columns, count), R = number
    of capture rows: each original segment is split where ``flat`` stops
    advancing by one, so every copy stays contiguous on both sides. With
    ``row_width`` (a row-stacked source, e.g. the raw all-gather [P, ..., W])
    segments also break where the column crosses a multiple of ``row_width``.
    """
    src0, dst0, ln = descriptors[0], descriptors[1], descriptors[2]
    d = src0.shape[0]
    r = flat.shape[0]
    rows = jnp.arange(r, dtype=jnp.int32)
    seg_ok = jnp.logical_and(jnp.arange(d, dtype=jnp.int32) < num_segments, ln > 0)
    if dws_cummax_enabled() and r < (1 << _DWS_END_BITS) and d < (1 << (31 - _DWS_END_BITS)):
        # One packed (segment id, segment end) key per segment start, propagated by
        # cummax: the id sits in the high bits, so the running max is the latest
        # started segment (the same sid as the start_at cummax below) and its low
        # bits are that segment's end. end is clipped to [0, r], which keeps
        # rows < end exact for every row in [0, r). dst = (dst0 - src0)[sid] + rows
        # is the same int32 value as dst0[sid] + rows - src0[sid] (two's complement).
        ids = jnp.arange(d, dtype=jnp.int32)
        end = jnp.clip(src0 + ln, 0, r)
        key = jnp.left_shift(ids, _DWS_END_BITS) | end
        key_at = jnp.full((r,), -1, jnp.int32).at[jnp.where(seg_ok, src0, r)].set(
            key, mode="drop"
        )
        ckey = jax.lax.cummax(key_at, axis=0)
        sid = jnp.right_shift(ckey, _DWS_END_BITS)  # arithmetic: -1 stays -1
        sidc = jnp.clip(sid, 0, d - 1)
        in_seg = jnp.logical_and(sid >= 0, rows < (ckey & ((1 << _DWS_END_BITS) - 1)))
        dst = (dst0 - src0)[sidc] + rows
    else:
        start_at = jnp.full((r,), -1, jnp.int32).at[jnp.where(seg_ok, src0, r)].set(
            jnp.arange(d, dtype=jnp.int32), mode="drop"
        )
        sid = jax.lax.cummax(start_at, axis=0)
        sidc = jnp.clip(sid, 0, d - 1)
        in_seg = jnp.logical_and(sid >= 0, rows < src0[sidc] + ln[sidc])
        dst = dst0[sidc] + rows - src0[sidc]
    ok = jnp.logical_and(in_seg, has)
    shift = lambda x, fill: jnp.concatenate((jnp.full((1,), fill, x.dtype), x[:-1]))
    start = jnp.logical_and(
        ok,
        jnp.logical_not(shift(ok, False))
        | (sid != shift(sid, -7))
        | (dst != shift(dst, -7) + 1)
        | (flat != shift(flat, -7) + 1)
        | (False if row_width is None else (flat % row_width == 0)),
    )
    new_id = jnp.cumsum(start.astype(jnp.int32)) - 1
    tgt = jnp.where(start, new_id, r)
    out_src = jnp.zeros((r,), jnp.int32).at[tgt].set(flat.astype(jnp.int32), mode="drop")
    out_dst = jnp.zeros((r,), jnp.int32).at[tgt].set(dst.astype(jnp.int32), mode="drop")
    out_len = jnp.zeros((r,), jnp.int32).at[jnp.where(ok, new_id, r)].add(1, mode="drop")
    return jnp.stack((out_src, out_dst, out_len), axis=0), jnp.sum(start.astype(jnp.int32))


def row_gather_enabled() -> bool:
    """TPU_PCP_RPA_ROW_GATHER=1 (all-RPA, default off until measured): the
    current-pool build and the KV capture gather whole token rows of a
    token-major copy instead of single lanes of the token-last layout."""
    return os.environ.get("TPU_PCP_RPA_ROW_GATHER", "0") == "1"


def _token_major(x):
    """[A, B, C, N] token-last -> [N, A, B, C] token-major."""
    return jnp.moveaxis(x, -1, 0)


def capture_apply(gathered, flat, has):
    if row_gather_enabled():
        rows = jnp.take(_token_major(gathered), flat, axis=0)  # [N, A, B, C]
        rows = jnp.where(has[:, None, None, None], rows, jnp.zeros((), rows.dtype))
        return jnp.moveaxis(rows, 0, -1)
    buf = jnp.take(gathered, flat, axis=-1)
    return jnp.where(has, buf, jnp.zeros((), buf.dtype))


def captured_kv_from_gathered(gathered, capture_offsets, owner_ranks, row_valid, cache_rank, lt):
    """Dense capture buffer for this cache rank from all ranks' fresh KV.

    Same order as the streaming kernel's ring capture: row g of the rank-major
    stream goes to ``capture_offsets[g]`` when ``cache_rank`` owns it.

    Args:
        gathered: [A, B, C, P * LT_pad] rank-major fresh KV (padded per rank).
        capture_offsets, owner_ranks, row_valid: [P * LT] writeback mapping.
        lt: local (unpadded) rows per rank.

    Returns:
        [A, B, C, P * LT] capture buffer (unused tail zero).
    """
    g_rows = capture_offsets.shape[0]
    lt_pad = gathered.shape[-1] // (g_rows // lt)
    flat, has = capture_index(
        capture_offsets, owner_ranks, row_valid, cache_rank, lt=lt, lt_pad=lt_pad
    )
    return capture_apply(gathered, flat, has)


def batched_rpa_current_fn(
    *,
    sm_scale: float,
    k_scale: float | None,
    v_scale: float | None,
    kv_layout,
    q_block: int,
    raw: bool = False,
) -> Callable:
    """RPAm causal over the temporary pool (FULL scope, no KV update, f32 LSE).

    With ``raw`` it returns the kernel's own layouts, as the history call.
    """
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs as rpa_configs
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as rpa_wrapper

    if kv_layout != rpa_configs.KVLayout.SEQ_ALONG_LANE:
        raise NotImplementedError("TPU_PCP_CURRENT_RPA supports the HND cache only.")
    compact = raw and lse_compact_enabled()
    no_alias = rpa_q_no_alias_kwargs() if compact else {}

    def fn(q, k, v, pool, kv_lens, page_indices, cu_q_lens, num_seqs,
           schedule=None, schedule_only=False):
        k = k.astype(pool.dtype)
        v = v.astype(pool.dtype)
        distribution = jnp.stack(
            (jnp.int32(0), jnp.int32(0), jnp.asarray(num_seqs, jnp.int32))
        )
        # Sequences are at most one interleave (q_block) long, so a q tile of
        # that size wastes no rows.
        blocks = rpa_configs.BlockSizes(
            bq_sz=q_block, bq_c_sz=16, bkv_sz=1024, batch_size=1
        )
        res = rpa_wrapper.ragged_paged_attention(
            q,
            k,
            v,
            pool,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_layout=kv_layout,
            return_lse=True,
            lse_dtype=jnp.float32,
            lse_compact=compact,
            skip_kv_update=True,
            prefill_block_sizes=blocks,
            raw_outputs=raw,
            schedule_only=schedule_only,
            mixed_schedule_leaves=schedule,
            **no_alias,
        )
        if schedule_only:
            return res
        o, _, lse = res
        if raw and not compact:
            return o, raw_lse(lse)
        return o, lse

    return fn


def pcp_attention_all_rpa(
    q_local,
    k_local,
    v_local,
    packed_current_kv,
    kv_cache_local,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    local_rows,
    writeback,
    *,
    pcp_size: int,
    interleave_size: int,
    page_size: int,
    pcp_axis_name: str,
    current_fn: Callable,
    history_fn: Callable,
    write_fn: Callable,
    raw: bool = False,
    meta=None,
    write_direct_fn: Callable | None = None,
    final_rows: bool = False,
    history_fn_sub: Callable | None = None,
    o_token_major: bool = False,
):
    """Current chunk and history both on batched RPA, merged by LSE.

    ``final_rows`` (TPU_PCP_MTP_FINAL_ROWS_HISTORY, MTP draft layer only):
    the history pass runs for each request's last-position row only
    (``history_fn_sub``, which returns the non-compact LSE); the current
    pass, the capture and the cache write-back are unchanged.

    With ``raw``, ``current_fn``/``history_fn`` return RPAm's own layouts
    (o [Hkv, T, G // p, p, D], lse [Hkv, T, G]) and the merge reads them in
    place and zeroes the invalid rows (``merge_partials_native``).

    ``meta`` (``pcp_all_rpa_meta``, computed once per forward pass) replaces
    every metadata-only computation and schedule build of this call.

    Args:
        local_rows: (positions, requests, valid) of this rank's Q rows.
        writeback: (capture_offsets, owner_ranks, row_valid, cache_rank) over
            the rank-major stream, from the streaming wrapper's mapping.
        write_fn: ``(kv_cache, captured) -> kv_cache`` cache write-back.

    Returns:
        (output [T, H, D] f32, kv_cache).
    """
    lt = q_local.shape[0]
    if final_rows and meta is not None:
        raise ValueError("TPU_PCP_MTP_FINAL_ROWS_HISTORY needs the per-call (non-hoisted) metadata.")
    if meta is None:
        max_num_seqs = kv_lens.shape[0]
        block_tables = (
            page_indices.reshape(max_num_seqs, -1)
            if page_indices.ndim == 1 else page_indices
        )
        local_num_blocks = jax.tree.leaves(kv_cache_local)[0].shape[0]
        block_tables_mod = jnp.mod(
            block_tables.astype(jnp.int32), jnp.int32(local_num_blocks)
        )
        with jax.named_scope("pcp_cur_inputs"):
            cur_meta = current_rpa_meta(
                kv_lens, cu_q_lens, distribution, block_tables_mod, *local_rows,
                lt_pad=packed_current_kv.shape[-1], pcp_size=pcp_size,
                interleave_size=interleave_size, page_size=page_size,
            )
        hist_meta = sched = cap = None
    else:
        cur_meta, hist_meta, cap, sched = meta["cur"], meta["hist"], meta["cap"], meta["sched"]
    def _capture(gathered):
        if cap is None:
            return captured_kv_from_gathered(gathered, *writeback, lt)
        return capture_apply(gathered, *cap)

    direct = capture_direct_enabled()
    if direct and write_direct_fn is None:
        raise ValueError("TPU_PCP_CAPTURE_DIRECT needs a write_direct_fn.")

    def _write(cache, gathered):
        if direct:
            # TPU_PCP_CAPTURE_DIRECT: segments straight from the gathered stream.
            flat, has = cap if cap is not None else capture_index(
                *writeback, lt=lt, lt_pad=packed_current_kv.shape[-1]
            )
            return write_direct_fn(cache, gathered, flat, has)
        return write_fn(cache, _capture(gathered))

    after = boundary_after_write_enabled()
    # Direct write + raw pool sources: the write kernel reads the raw
    # all-gather row stack too, so nothing needs the token-last relayout.
    raw_src = direct and pool_kernel_raw_sources_enabled()
    with jax.named_scope("pcp_cur_inputs"):
        cur = current_rpa_build(
            cur_meta, packed_current_kv, kv_cache_local, pcp_size=pcp_size,
            page_size=page_size, pcp_axis_name=pcp_axis_name,
            write_after_gather=(
                (lambda g: _write(kv_cache_local, g)) if after else None
            ),
            raw_write_source=raw_src,
        )
    with jax.named_scope("pcp_cur_rpa"):
        o_cur, lse_cur = current_fn(
            q_local,
            k_local,
            v_local,
            cur["pool"],
            cur["kv_lens"],
            cur["page_indices"],
            cur["cu_q_lens"],
            cur["num_seqs"],
            **({} if sched is None else dict(schedule=sched["cur"])),
        )
    if after:
        kv = cur["cache"]  # already written, before the boundary read
    else:
        with jax.named_scope("pcp_cur_writeback"):
            # cur["cache"] is kv_cache_local itself unless the read-before-write
            # barrier is on, in which case it is the barrier's output.
            kv = _write(cur["cache"], cur["g_raw"] if raw_src else cur["gathered"])
    parts_o, parts_lse, kv = pcp_history_partials(
        q_local,
        k_local,
        v_local,
        kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        page_size=page_size,
        pcp_axis_name=pcp_axis_name,
        history_fn=(history_fn_sub or history_fn) if final_rows else history_fn,
        rank=writeback[3],
        meta=hist_meta,
        schedules=None if sched is None else sched["hist"],
        final_rows=final_rows,
        lse_out_compact=bool(final_rows and raw and lse_compact_enabled()),
    )
    with jax.named_scope("pcp_all_merge"):
        if not isinstance(parts_o, list):
            parts_o = [parts_o[j] for j in range(parts_o.shape[0])]
            parts_lse = [parts_lse[j] for j in range(parts_lse.shape[0])]
        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import merge_kernel

        all_o = [o_cur] + parts_o
        all_lse = [lse_cur] + parts_lse
        g = all_o[0].shape[2] * all_o[0].shape[3] if raw else None
        compact = raw and lse_compact_enabled()
        if raw and merge_kernel.merge_kernel_enabled() and merge_kernel.supported(all_o):
            # TPU_PCP_MERGE_KERNEL: one Pallas pass over the raw partials.
            with jax.named_scope("pcp_lse_compact"):
                lds = [
                    x.astype(jnp.float32) if compact else merge_kernel.lane_dense(x.astype(jnp.float32))
                    for x in all_lse
                ]
                mask = merge_kernel.row_mask_lane_dense(local_rows[2], g)
            out = merge_kernel.merge_partials_kernel(
                all_o, lds, mask, q_local.dtype, **({"token_major": True} if o_token_major else {})
            )
        elif raw:
            if compact:  # lane-dense [Hkv, T * Gp // 128, 128] -> [Hkv, T, Gp]
                all_lse = [x.reshape(x.shape[0], all_o[0].shape[1], g) for x in all_lse]
            out = merge_partials_native(all_o, all_lse, local_rows[2], q_local.dtype)
        else:
            out = merge_partials_fused(
                [o_cur] + parts_o, [lse_cur] + parts_lse, q_local.dtype
            )
        if o_token_major and out.ndim == 3:  # TPU_PCP_O_TOKEN_MAJOR without the merge kernel
            out = out.reshape(out.shape[0], -1)
    return out, kv


def hoist_meta_enabled() -> bool:
    """TPU_PCP_RPA_HOIST_META=1 (all-RPA only, default off until measured):
    the metadata-only work and the RPA schedule builds of the all-RPA path
    run once per forward pass (``pcp_all_rpa_meta``) instead of per layer.
    Unset, it follows the glue-trims group switch TPU_PCP_GLUE_TRIMS (same
    rule as batched_rpa.wrapper.glue_flag; no import, it runs in traced code)."""
    v = os.environ.get("TPU_PCP_RPA_HOIST_META", "")
    if v in ("0", "1"):
        return v == "1"
    return os.environ.get("TPU_PCP_GLUE_TRIMS", "0") == "1"


def pcp_all_rpa_meta(
    q_local,
    k_local,
    kv_cache_local,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    local_rows,
    writeback,
    *,
    lt_pad: int,
    pcp_size: int,
    interleave_size: int,
    page_size: int,
    current_fn: Callable,
    history_fn: Callable,
):
    """Everything ``pcp_attention_all_rpa`` computes from metadata alone.

    Inside the PCP shard_map body. ``q_local``, ``k_local`` and
    ``kv_cache_local`` supply shapes and dtypes only (the schedule builds
    read no tensor data). Returns the ``meta`` dict for
    ``pcp_attention_all_rpa``.
    """
    max_num_seqs = kv_lens.shape[0]
    block_tables = (
        page_indices.reshape(max_num_seqs, -1) if page_indices.ndim == 1 else page_indices
    )
    local_num_blocks = jax.tree.leaves(kv_cache_local)[0].shape[0]
    block_tables_mod = jnp.mod(block_tables.astype(jnp.int32), jnp.int32(local_num_blocks))
    lt = q_local.shape[0]
    cur = current_rpa_meta(
        kv_lens, cu_q_lens, distribution, block_tables_mod, *local_rows,
        lt_pad=lt_pad, pcp_size=pcp_size, interleave_size=interleave_size,
        page_size=page_size,
    )
    hist = history_partials_meta(
        kv_lens, page_indices, cu_q_lens, distribution,
        local_num_blocks=local_num_blocks, pcp_size=pcp_size,
        interleave_size=interleave_size, page_size=page_size,
    )
    cap = capture_index(*writeback, lt=lt, lt_pad=lt_pad)
    ntp = cur["is_b"].shape[0] // page_size
    pool_shape = jnp.zeros((ntp,) + kv_cache_local.shape[1:], kv_cache_local.dtype)
    sched_cur = current_fn(
        q_local, k_local, k_local, pool_shape, cur["kv_lens"], cur["page_indices"],
        cur["cu_q_lens"], cur["num_seqs"], schedule_only=True,
    )
    per_src = [
        history_fn(
            q_local, k_local, k_local, kv_cache_local, hist["kv_lens_src"][src],
            hist["local_page_indices"], hist["cu_q_lens_src"][src], hist["num_seqs"],
            schedule_only=True,
        )
        for src in range(pcp_size)
    ]
    sched_hist = tuple(jnp.stack(leaves) for leaves in zip(*per_src))
    return dict(cur=cur, hist=hist, cap=cap, sched=dict(cur=tuple(sched_cur), hist=sched_hist))
