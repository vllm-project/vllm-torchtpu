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
"""Server-free 8-rank replay & verification for SparseMLA + SWA + compact_records.

Covers atomic subop isolation, verification, and benchmarking for:
  - `--mode precompact`: cache `compact_records` at 8 source layers
  - `--mode token_parallel`: All-to-All `q` to `[T/8, 64, 512]`
  - `--mode union_gather`: per-128-token-chunk unique KV row union

Supported `--subop` modes:
  - `all`: Full `sparse_fp8_attention_jax` (SWA + barrier + compact + SparseMLA)
  - `swa_only`: `_attend_window` (sliding-window union gather + band-mask attention)
  - `compact_records_only`: `compact_records` (`int32[T, 512]` sort & `-1` tail pad)
  - `sc_gather_only`: `gather_page_ids` + `csa_gather` across all 128-token chunks
  - `mla_tc_only`: TensorCore `_attention_kernel` only (pre-gathered KV buffers)
"""

import argparse
import functools
import pathlib
import statistics
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common  # noqa: E402

SUBOPS = (
    "all",
    "swa_only",
    "compact_records_only",
    "sc_gather_only",
    "mla_tc_only",
)
MODES = (
    "baseline",
    "precompact",
    "token_parallel",
    "union_gather",
)
COMPRESSED_LAYERS = 38
REUSE_INDEX_LAYERS = 30


def _build_query_start_loc(torch, tokens: int, num_seqs: int, q_len: int = 0) -> Any:
    """Build `query_start_loc` (`int32[num_seqs + 1]`).

    When `q_len > 0` (Decode `q_len=1` or DSpark Verify `q_len=6`), each of the
    `num_seqs` requests has `q_len` query tokens (`[0, q_len, 2*q_len, ...]`).
    When `q_len == 0` (prefill chunk), leaves 2 tail pad tokens and uses
    non-multiples of 64 when `num_seqs > 1` so 64-token SWA tiles straddle
    request boundaries and exercise the `uncovered` patch path in `_attend_window`.
    """
    if q_len > 0:
        if num_seqs * q_len > tokens:
            raise ValueError(
                f"num_seqs={num_seqs} * q_len={q_len} exceeds tokens={tokens}"
            )
        return torch.arange(num_seqs + 1, dtype=torch.int32) * q_len
    valid_tokens = max(1, tokens - 2)
    if num_seqs == 1:
        return torch.tensor([0, valid_tokens], dtype=torch.int32)
    base = valid_tokens // num_seqs
    starts = [0]
    for s in range(num_seqs - 1):
        delta = (-7 if s % 2 == 0 else 7) if base > 16 else 0
        nxt = min(
            valid_tokens - (num_seqs - 1 - s),
            max(starts[-1] + 1, (s + 1) * base + delta),
        )
        starts.append(nxt)
    starts.append(valid_tokens)
    return torch.tensor(starts, dtype=torch.int32)


def _build_swa_inputs(
    torch,
    rng: Any,
    tokens: int,
    window_size: int,
    query_start_loc: Any,
    q_len: int = 0,
) -> tuple[Any, Any, Any]:
    """Build `(swa_pool, swa_records, swa_slots)` matching `v2/attention.py`.

    Each SWA record is 512 BF16 elements = 256 int32 words = 2 word-rows of 128.
    Within each sequence, `swa_records[t, j]` shifts by 1 column per token so
    `_window_union` covers intra-request tiles and only patches cross-request
    boundary tokens.
    """
    num_seqs = int(query_start_loc.numel() - 1)
    valid_tokens = int(query_start_loc[-1].item())
    swa_num_records = tokens + num_seqs * (window_size + 16) + 64
    # Finite normal BF16 values packed in int32[swa_num_records * 2, 128].
    vals = (
        torch.randn((swa_num_records, 512), generator=rng, dtype=torch.float32) * 0.25
    )
    vals_bf16 = vals.to(torch.bfloat16)
    swa_pool = vals_bf16.view(torch.int32).reshape(swa_num_records * 2, 128)

    swa_records = torch.full((tokens, window_size), -1, dtype=torch.int32)
    swa_slots = torch.full((tokens,), -1, dtype=torch.int32)
    w_offs = torch.arange(window_size, dtype=torch.int32) - (window_size - 1)
    lead_offset = window_size if q_len > 0 else (window_size - 3)

    for s in range(num_seqs):
        start = int(query_start_loc[s].item())
        end = int(query_start_loc[s + 1].item())
        length = end - start
        if length <= 0:
            continue
        base_rec = s * (tokens // max(1, num_seqs) + window_size + 16) + window_size
        seq_pos = torch.arange(length, dtype=torch.int32) + lead_offset
        logical = seq_pos[:, None] + w_offs[None, :]
        phys = base_rec + logical - lead_offset
        valid_col = logical >= 0
        swa_records[start:end] = torch.where(valid_col, phys, torch.full_like(phys, -1))
        swa_slots[start:end] = base_rec + torch.arange(length, dtype=torch.int32)

    assert int((swa_slots[:valid_tokens] >= 0).sum().item()) == valid_tokens
    assert int((swa_slots[valid_tokens:] == -1).sum().item()) == tokens - valid_tokens
    return swa_pool, swa_records, swa_slots


def _build_main_kv_inputs(
    torch,
    rng: Any,
    tokens: int,
    kv_len: int,
    compress_ratio: int,
    page_size: int,
    topk: int,
    query_start_loc: Any,
    unique_per_chunk: int,
) -> tuple[Any, Any, Any, Any]:
    """Build `(main_pool, rope_pool, pages, main_records)` on CPU."""
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import encode_fp8_kv

    num_seqs = int(query_start_loc.numel() - 1)
    valid_tokens = int(query_start_loc[-1].item())
    compressed_kv_len = max(topk, kv_len // compress_ratio)
    pages_per_seq = (compressed_kv_len + page_size - 1) // page_size
    # Physical pool sized to fit at least one full sequence plus per-seq offsets
    # while staying compact (~64 MiB per rank) for fast deterministic init.
    total_pages = max(pages_per_seq + num_seqs * 64, 6144)
    total_rows = total_pages * page_size

    # Encode a 1024-row template via `encode_fp8_kv` and tile across total_rows
    # so every physical row has valid DSV4 FP8 NoPE + UE8M0 scales and BF16 RoPE.
    template_rows = 1024
    latent_f32 = (
        torch.randn((template_rows, 512), generator=rng, dtype=torch.float32) * 0.5
    ).to(torch.bfloat16)
    words_jax, rope_jax = encode_fp8_kv(
        jnp.asarray(latent_f32.float().numpy(), jnp.bfloat16)
    )
    words_t = torch.from_numpy(
        np.asarray(words_jax, dtype=np.int32).copy()
    )  # [1024, 128]
    rope_t = torch.from_numpy(np.asarray(rope_jax, dtype=np.int32).copy())  # [1024, 32]

    repeats = (total_rows + template_rows - 1) // template_rows
    main_pool = words_t.repeat((repeats, 1))[:total_rows].contiguous()
    rope_words = rope_t.repeat((repeats, 1))[:total_rows].contiguous()
    rope_pool = rope_words.reshape(total_rows // 4, 128).contiguous()

    # Build page table `[num_seqs, pages_per_seq]`.
    pages = torch.empty((num_seqs, pages_per_seq), dtype=torch.int32)
    base_pages = torch.arange(pages_per_seq, dtype=torch.int32)
    for s in range(num_seqs):
        pages[s] = (base_pages + s * 61) % total_pages

    # Build `main_records` `[tokens, topk]` with realistic ~8x overlap inside
    # each 128-token chunk, unsorted, plus partial `-1` entries on early tokens
    # and all `-1` on tail padded tokens.
    main_records = torch.full((tokens, topk), -1, dtype=torch.int32)
    chunk_tokens = 128

    for chunk_start in range(0, valid_tokens, chunk_tokens):
        chunk_end = min(valid_tokens, chunk_start + chunk_tokens)
        seqs_in_chunk = max(
            1,
            sum(
                1
                for s in range(num_seqs)
                if max(chunk_start, int(query_start_loc[s].item()))
                < min(chunk_end, int(query_start_loc[s + 1].item()))
            ),
        )
        chunk_pool_width = min(
            compressed_kv_len, max(topk, unique_per_chunk // seqs_in_chunk)
        )
        # Shared candidate pool for this 128-token chunk.
        shared_pool = torch.randperm(
            compressed_kv_len, generator=rng, dtype=torch.int32
        )[:chunk_pool_width]
        for t in range(chunk_start, chunk_end):
            # Pick `topk` unique logical positions from `shared_pool` in random
            # (unsorted) order so `compact_records` must sort them.
            perm = torch.randperm(chunk_pool_width, generator=rng)[:topk]
            row = shared_pool[perm]
            if (t - chunk_start) == 0:
                # First token in each chunk has 16 invalid (`-1`) slots interspersed
                # to test moving `-1` to the tail in `compact_records` & `SparseMLA`.
                row[::32] = -1
            main_records[t] = row

    assert int((main_records[valid_tokens:] == -1).all().item()) == 1
    return main_pool, rope_pool, pages, main_records


def _compact_records_torch_ref(torch, records: Any) -> Any:
    """Exact PyTorch CPU reference for `compact_records`."""
    width = records.shape[1]
    padded = max(128, -(-width // 128) * 128)
    if padded > width:
        records = torch.nn.functional.pad(records, (0, padded - width), value=-1)
    limit = torch.iinfo(torch.int32).max
    filled = torch.where(records >= 0, records, torch.full_like(records, limit))
    sorted_vals, _ = torch.sort(filled, dim=-1)
    return torch.where(
        sorted_vals == limit, torch.full_like(sorted_vals, -1), sorted_vals
    )


def _sc_gather_only_jax(
    main_pool,
    rope_pool,
    compacted_records,
    pages,
    query_start_loc,
    *,
    page_size: int,
    gather_chunk_size: int = 128,
    kernel_batch_size_cap: int = 16,
):
    """Isolate `gather_page_ids` + `csa_gather` across all 128-token chunks."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather, sparse_mla

    num_tokens, topk = compacted_records.shape
    count = main_pool.shape[0] // page_size
    cache_kv_nope = main_pool.reshape(count, page_size, main_pool.shape[1])
    cache_kv_rope = rope_pool.reshape(count, page_size // 4, rope_pool.shape[1])
    page_indices = pages.reshape(-1)
    max_num_seqs = pages.shape[0]

    tokens_per_seq = query_start_loc[1:] - query_start_loc[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs, dtype=jnp.int32),
        tokens_per_seq,
        total_repeat_length=num_tokens,
    )
    seq_page_ids = compacted_records // page_size
    token_offset = compacted_records % page_size
    page_ids = sparse_mla.gather_page_ids(
        page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
    )
    is_padding = compacted_records == -1
    total_num_pages = cache_kv_nope.shape[0]
    flat_element_index = jnp.arange(num_tokens * topk, dtype=jnp.int32).reshape(
        num_tokens, topk
    )
    scattered_page_ids = (flat_element_index * 104729) % total_num_pages
    scattered_token_offset = (flat_element_index * 15485863) % page_size
    page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
    token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)
    gather_indices = (page_ids * page_size + token_offset).reshape(-1)

    num_chunks = sparse_mla.cdiv(num_tokens, gather_chunk_size)
    valid_total = query_start_loc[max_num_seqs]
    nope_samples = []
    rope_samples = []

    for i in range(num_chunks):
        start_pos = i * gather_chunk_size
        end_pos = min(start_pos + gather_chunk_size, num_tokens)
        chunk_size = end_pos - start_pos
        indices = gather_indices[start_pos * topk : end_pos * topk]
        kbs = sparse_mla._largest_divisor(chunk_size, kernel_batch_size_cap)
        batch_end = start_pos + (
            sparse_mla.cdiv(
                jnp.maximum(0, jnp.minimum(valid_total, end_pos) - start_pos),
                kbs,
            )
            * kbs
        )
        num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk
        g_nope, g_rope = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            indices,
            num_valid_indices=num_valid_indices,
            top_k=topk,
        )
        g_nope = g_nope.reshape(chunk_size, topk, -1)
        g_rope = g_rope.reshape(chunk_size, topk // 4, -1)
        # Keep a deterministic 4-word sample per token from every chunk live so
        # XLA must execute every chunk's `sc_csa_gather` kernel and `--verify`
        # can check exact gathered values against the pool.
        nope_samples.append(g_nope[:, 0, :4])
        rope_samples.append(g_rope[:, 0, :4])

    return jnp.concatenate(nope_samples, axis=0), jnp.concatenate(rope_samples, axis=0)


def _sc_gather_union_only_jax(
    main_pool,
    rope_pool,
    compacted_records,
    pages,
    query_start_loc,
    *,
    page_size: int,
    gather_chunk_size: int = 128,
    union_cap: int = 4096,
):
    """Isolate `compute_chunk_union_gather_indices` + union `csa_gather`."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather, sparse_mla
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
        compute_chunk_union_gather_indices,
    )

    num_tokens, topk = compacted_records.shape
    count = main_pool.shape[0] // page_size
    cache_kv_nope = main_pool.reshape(count, page_size, main_pool.shape[1])
    cache_kv_rope = rope_pool.reshape(count, page_size // 4, rope_pool.shape[1])
    page_indices = pages.reshape(-1)
    max_num_seqs = pages.shape[0]

    union_phys, inv_idx, num_unique = compute_chunk_union_gather_indices(
        compacted_records,
        page_indices,
        query_start_loc,
        page_size=page_size,
        total_num_pages=count,
        chunk_size=gather_chunk_size,
        union_cap=union_cap,
    )
    num_chunks = union_phys.shape[0]
    valid_total = query_start_loc[max_num_seqs]
    u_nope_4_chunks = []
    u_rope_chunks = []

    for i in range(num_chunks):
        start_pos = i * gather_chunk_size
        has_valid_tokens = valid_total > start_pos
        num_valid_union = jnp.where(
            has_valid_tokens,
            jnp.minimum(union_cap, sparse_mla.align_to(num_unique[i], topk)),
            0,
        )
        u_nope, u_rope = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            union_phys[i],
            num_valid_indices=num_valid_union,
            top_k=topk,
        )
        u_nope_4_chunks.append(u_nope.reshape(union_cap, -1)[:, :4])
        u_rope_chunks.append(u_rope)

    u_nope_4 = jnp.stack(u_nope_4_chunks, axis=0)
    u_rope_all = jnp.stack(u_rope_chunks, axis=0)
    num_periods = union_cap // topk
    half_rows = topk // 4
    periods = u_rope_all.reshape(num_chunks, num_periods, half_rows, 128)

    def _unpack_half_4(lo, hi):
        even = jnp.bitwise_or(jnp.bitwise_and(lo, 0xFFFF), jnp.left_shift(hi, 16))
        odd = jnp.bitwise_or(
            jnp.right_shift(lo.astype(jnp.uint32), jnp.uint32(16)).astype(jnp.int32),
            jnp.bitwise_and(hi, jnp.int32(-65536)),
        )
        return jnp.stack([even, odd], axis=3).reshape(
            num_chunks, num_periods, half_rows * 2, 4
        )

    u_rope_4 = jnp.concatenate(
        [
            _unpack_half_4(periods[:, :, :, :4], periods[:, :, :, 32:36]),
            _unpack_half_4(periods[:, :, :, 64:68], periods[:, :, :, 96:100]),
        ],
        axis=2,
    ).reshape(num_chunks, union_cap, 4)

    u_nope_t = jnp.swapaxes(u_nope_4, 1, 2)
    u_rope_t = jnp.swapaxes(u_rope_4, 1, 2)
    combined_table = jnp.concatenate([u_nope_t, u_rope_t, u_rope_t], axis=1).reshape(
        num_chunks * 12, union_cap
    )
    idx0_rep = jnp.broadcast_to(
        inv_idx[:, None, :, 0], (num_chunks, 4, gather_chunk_size)
    )
    idx1_rep = jnp.broadcast_to(
        inv_idx[:, None, :, 1], (num_chunks, 4, gather_chunk_size)
    )
    combined_idx = jnp.concatenate([idx0_rep, idx0_rep, idx1_rep], axis=1).reshape(
        num_chunks * 12, gather_chunk_size
    )
    gathered_all = jnp.take_along_axis(combined_table, combined_idx, axis=1).reshape(
        num_chunks, 3, 4, gather_chunk_size
    )
    gathered_all = jnp.swapaxes(gathered_all, 2, 3)
    nope_out = gathered_all[:, 0].reshape(num_chunks * gather_chunk_size, 4)[
        :num_tokens
    ]
    even0 = gathered_all[:, 1].reshape(num_chunks * gather_chunk_size, 4)[:num_tokens]
    odd1 = gathered_all[:, 2].reshape(num_chunks * gather_chunk_size, 4)[:num_tokens]
    rope_out = jnp.bitwise_or(jnp.bitwise_and(even0, 0xFFFF), jnp.left_shift(odd1, 16))
    return nope_out, rope_out


def _pregather_all_chunks_jax(
    main_pool,
    rope_pool,
    compacted_records,
    pages,
    query_start_loc,
    *,
    page_size: int,
    gather_chunk_size: int = 128,
    kernel_batch_size_cap: int = 16,
):
    """Pre-gather `(nope_chunks, rope_chunks, kv_lens)` outside timed `mla_tc_only`."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather, sparse_mla

    num_tokens, topk = compacted_records.shape
    count = main_pool.shape[0] // page_size
    cache_kv_nope = main_pool.reshape(count, page_size, main_pool.shape[1])
    cache_kv_rope = rope_pool.reshape(count, page_size // 4, rope_pool.shape[1])
    page_indices = pages.reshape(-1)
    max_num_seqs = pages.shape[0]

    tokens_per_seq = query_start_loc[1:] - query_start_loc[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs, dtype=jnp.int32),
        tokens_per_seq,
        total_repeat_length=num_tokens,
    )
    kv_lens = jnp.sum(compacted_records != -1, axis=-1)
    seq_page_ids = compacted_records // page_size
    token_offset = compacted_records % page_size
    page_ids = sparse_mla.gather_page_ids(
        page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
    )
    is_padding = compacted_records == -1
    total_num_pages = cache_kv_nope.shape[0]
    flat_element_index = jnp.arange(num_tokens * topk, dtype=jnp.int32).reshape(
        num_tokens, topk
    )
    scattered_page_ids = (flat_element_index * 104729) % total_num_pages
    scattered_token_offset = (flat_element_index * 15485863) % page_size
    page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
    token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)
    gather_indices = (page_ids * page_size + token_offset).reshape(-1)

    num_chunks = sparse_mla.cdiv(num_tokens, gather_chunk_size)
    valid_total = query_start_loc[max_num_seqs]
    nope_chunks = []
    rope_chunks = []
    for i in range(num_chunks):
        start_pos = i * gather_chunk_size
        end_pos = min(start_pos + gather_chunk_size, num_tokens)
        chunk_size = end_pos - start_pos
        indices = gather_indices[start_pos * topk : end_pos * topk]
        kbs = sparse_mla._largest_divisor(chunk_size, kernel_batch_size_cap)
        batch_end = start_pos + (
            sparse_mla.cdiv(
                jnp.maximum(0, jnp.minimum(valid_total, end_pos) - start_pos),
                kbs,
            )
            * kbs
        )
        num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk
        g_nope, g_rope = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            indices,
            num_valid_indices=num_valid_indices,
            top_k=topk,
        )
        nope_chunks.append(g_nope.reshape(chunk_size, topk, -1))
        rope_chunks.append(g_rope.reshape(chunk_size, topk // 4, -1))
    return jnp.stack(nope_chunks, axis=0), jnp.stack(rope_chunks, axis=0), kv_lens


def _mla_tc_only_jax(
    q,
    nope_chunks,
    rope_chunks,
    kv_lens,
    query_start_loc,
    sink,
    swa_num,
    swa_den,
    swa_max,
    *,
    scale: float,
    max_num_seqs: int,
    gather_chunk_size: int = 128,
    kernel_batch_size_cap: int = 16,
    vmem_limit_bytes: int = 100 * 1024 * 1024,
):
    """Run ONLY the TensorCore `_attention_kernel` across pre-gathered chunks."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla

    _, actual_num_q_heads, actual_head_dim = q.shape
    q = sparse_mla.prepare_q_inputs(q)
    head_dim = q.shape[-1]
    attention_sinks = jnp.pad(
        sink.astype(jnp.float32),
        (0, q.shape[1] - actual_num_q_heads),
        constant_values=jnp.finfo(jnp.float32).min,
    )
    swa_accumution, swa_l, swa_m = sparse_mla.prepare_swa_inputs(
        swa_num.astype(q.dtype), swa_den, swa_max
    )
    _, num_q_heads, _ = q.shape
    num_chunks = nope_chunks.shape[0]
    valid_total = query_start_loc[max_num_seqs]

    for i in range(num_chunks):
        start_pos = i * gather_chunk_size
        end_pos = min(start_pos + gather_chunk_size, q.shape[0])
        chunk_size = end_pos - start_pos
        kbs = sparse_mla._largest_divisor(chunk_size, kernel_batch_size_cap)
        batch_end = start_pos + (
            sparse_mla.cdiv(
                jnp.maximum(0, jnp.minimum(valid_total, end_pos) - start_pos),
                kbs,
            )
            * kbs
        )
        cache_kv_nope = nope_chunks[i]
        cache_kv_rope = rope_chunks[i]
        end_seq_idx = jnp.maximum(start_pos, batch_end)
        grid = (sparse_mla.cdiv(end_seq_idx - start_pos, kbs),)
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)
        _, topk_rows, lanes = cache_kv_nope.shape
        _, rope_rows, _ = cache_kv_rope.shape
        num_l_heads = sparse_mla.align_to(num_q_heads, 128)
        scratch_shapes = [
            pltpu.VMEM((2, kbs, topk_rows * 4, lanes), jnp.uint8),
            pltpu.VMEM((2, kbs, rope_rows * 2, lanes), jnp.bfloat16),
            pltpu.VMEM((2, kbs, num_q_heads, head_dim), q.dtype),
            pltpu.VMEM((2, kbs, num_q_heads, head_dim), q.dtype),
            pltpu.VMEM((2, kbs, num_l_heads), jnp.float32),
            pltpu.VMEM((2, kbs, num_l_heads), jnp.float32),
            pltpu.VMEM((2, kbs, num_q_heads, head_dim), q.dtype),
            pltpu.SemaphoreType.DMA((7, 2)),
        ]
        scalar_prefetches = (
            kv_lens,
            jnp.array([start_pos, end_seq_idx, valid_total], jnp.int32),
            jnp.zeros((2,), jnp.int32),
            jnp.full((2,), -1, jnp.int32),
        )
        scope_name = f"SparseMLA-p_{topk_rows}-bz_{kbs}-gcz_{chunk_size}"
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    sparse_mla._attention_kernel,
                    sm_scale=scale,
                    batch_size=kbs,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                input_output_aliases={5: 0},
                name=scope_name,
            )
        )
        q = kernel(
            *scalar_prefetches,
            attention_sinks,
            q,
            cache_kv_nope,
            cache_kv_rope,
            swa_accumution,
            swa_l,
            swa_m,
        )
    return sparse_mla.prepare_outputs(q, actual_num_q_heads, actual_head_dim)


def _sparse_fp8_attention_precompact_jax(
    q,
    swa_pool,
    main_pool,
    rope_pool,
    swa_records,
    compacted_main_records,
    pages,
    query_start_loc,
    sink,
    *,
    scale: float,
    page_size: int,
    max_requests: int | None = None,
):
    """`sparse_fp8_attention_jax` when `main_records` is already compacted."""
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
        sparse_ragged_paged_attention,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
        _attend_window,
        _attention_state,
    )

    state = _attend_window(
        q,
        swa_pool,
        swa_records,
        _attention_state(q),
        scale,
        max_requests=max_requests,
    )
    (maximum, denominator, numerator), main_pool, rope_pool = (
        jax.lax.optimization_barrier((state, main_pool, rope_pool))
    )
    count = main_pool.shape[0] // page_size
    return sparse_ragged_paged_attention(
        q,
        main_pool.reshape(count, page_size, main_pool.shape[1]),
        rope_pool.reshape(count, page_size // 4, rope_pool.shape[1]),
        compacted_main_records,
        pages.reshape(-1),
        query_start_loc,
        jnp.array([0, 0, pages.shape[0]], jnp.int32),
        sink.astype(jnp.float32),
        numerator.astype(q.dtype),
        denominator,
        maximum,
        sm_scale=scale,
    )


def _compute_roofline(
    hw: common.Tpu7xRoofline,
    subop: str,
    mode: str,
    tokens: int,
    tp_size: int,
    heads_total: int,
    head_dim: int,
    window_size: int,
    topk: int,
    measured_ms: float | None,
    union_cap: int = 4096,
    q_len: int = 0,
) -> dict[str, Any]:
    """First-principles hardware roofline for the selected `subop` and `mode`."""
    if mode == "token_parallel":
        t_local = tokens // tp_size
        h_local = heads_total
        # Two All-to-Alls on bf16[T, heads_total // tp_size, head_dim].
        a2a_bytes = tokens * (heads_total // tp_size) * head_dim * 2
        comm_s = 2.0 * hw.all_to_all_s(a2a_bytes, tp_size) if subop == "all" else 0.0
    else:
        t_local = tokens
        h_local = heads_total // tp_size
        comm_s = 0.0

    if q_len > 0:
        num_active_reqs = max(1, (t_local + q_len - 1) // q_len)
        swa_flops = 4.0 * t_local * h_local * window_size * head_dim
        swa_hbm_bytes = (
            num_active_reqs * (q_len - 1 + window_size) * head_dim * 2.0
            + 2.0 * t_local * h_local * head_dim * 2.0
        )
    else:
        num_swa_tiles = max(1, t_local // 64)
        swa_union_w = 64 + window_size  # 192
        swa_flops = 4.0 * t_local * h_local * swa_union_w * head_dim
        swa_hbm_bytes = (
            num_swa_tiles * swa_union_w * head_dim * 2.0
            + 2.0 * t_local * h_local * head_dim * 2.0
        )

    compact_hbm_bytes = 2.0 * t_local * topk * 4.0
    # Logical bytes gathered by SparseCore (512B NoPE + 128B RoPE per topk entry)
    # plus HBM write to gathered_nope/rope buffers.
    if mode == "union_gather":
        num_gather_chunks = max(1, (t_local + 127) // 128)
        sc_gather_hbm_bytes = 2.0 * num_gather_chunks * union_cap * (512.0 + 128.0)
    else:
        sc_gather_hbm_bytes = 2.0 * t_local * topk * (512.0 + 128.0)

    mla_tc_flops = 4.0 * t_local * h_local * topk * head_dim
    mla_tc_hbm_bytes = (
        t_local * topk * (512.0 + 128.0) + 3.0 * t_local * h_local * head_dim * 2.0
    )

    if subop == "swa_only":
        flops, hbm_bytes = swa_flops, swa_hbm_bytes
    elif subop == "compact_records_only":
        flops, hbm_bytes = 0.0, compact_hbm_bytes
    elif subop == "sc_gather_only":
        flops, hbm_bytes = 0.0, sc_gather_hbm_bytes
    elif subop == "mla_tc_only":
        flops, hbm_bytes = mla_tc_flops, mla_tc_hbm_bytes
    else:
        flops = swa_flops + mla_tc_flops
        hbm_bytes = swa_hbm_bytes + sc_gather_hbm_bytes + mla_tc_hbm_bytes
        if mode not in ("precompact", "union_gather", "token_parallel"):
            hbm_bytes += compact_hbm_bytes

    est = common.RooflineEstimate(
        name=f"sparse_mla_swa[{subop},{mode}] T_local={t_local} H_local={h_local}",
        flops=flops,
        hbm_bytes=hbm_bytes,
        comm_s=comm_s,
        dtype="bf16",
    ).report(hw, measured_ms)
    est["t_local"] = t_local
    est["h_local"] = h_local
    est["x38_compressed_layers_lower_bound_ms"] = (
        est["lower_bound_ms"] * COMPRESSED_LAYERS
    )
    if measured_ms is not None:
        est["x38_compressed_layers_measured_ms"] = measured_ms * COMPRESSED_LAYERS
    return est


def worker(ctx: common.WorkerContext, args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import torch
    import torch.distributed as dist
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas
    from vllm.distributed import get_tp_group

    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
        _attend,
        _attend_window,
        _attention_state,
        compact_records,
        initialize_paged_attention,
        sparse_fp8_attention,
    )

    geo = common.Dsv41Geometry()
    tokens = args.chunk_size
    q_len = int(getattr(args, "q_len", 0))
    swa_width_arg = int(getattr(args, "swa_width", 0))
    if q_len > 0:
        if tokens % 4 != 0:
            raise ValueError(
                f"--chunk-size {tokens} must be a multiple of 4 when --q-len > 0"
            )
        if args.mode == "token_parallel" and (
            tokens % ctx.tp_size != 0 or (tokens // ctx.tp_size) % 4 != 0
        ):
            raise ValueError(
                f"--chunk-size {tokens} / tp_size={ctx.tp_size} must be a multiple of 4"
            )
        if args.num_seqs * q_len > tokens:
            raise ValueError(
                f"--num-seqs {args.num_seqs} * --q-len {q_len} exceeds "
                f"--chunk-size {tokens}"
            )
    else:
        if tokens % 128 != 0:
            raise ValueError(f"--chunk-size {tokens} must be a multiple of 128")
        if tokens % ctx.tp_size != 0 or (tokens // ctx.tp_size) % 128 != 0:
            raise ValueError(
                f"--chunk-size {tokens} / tp_size={ctx.tp_size} must be a "
                "multiple of 128"
            )
    heads_total = geo.num_attention_heads
    heads_local = geo.heads_per_rank(ctx.tp_size)
    head_dim = geo.head_dim
    window_size = swa_width_arg if swa_width_arg > 0 else geo.sliding_window
    topk = geo.index_topk
    scale = float(head_dim**-0.5)
    max_requests = args.num_seqs if args.num_seqs <= 4 else None
    union_cap = max(topk, ((args.unique_per_chunk + topk - 1) // topk) * topk)
    tp_rank = get_tp_group().rank_in_group

    # Build deterministic inputs on CPU with fixed seed across all ranks.
    rng = torch.Generator(device="cpu").manual_seed(args.seed + 1009)
    q_full_cpu = (
        torch.randn((tokens, heads_total, head_dim), generator=rng, dtype=torch.float32)
        * 0.25
    ).to(torch.bfloat16)
    sink_full_cpu = (
        torch.randn((heads_total,), generator=rng, dtype=torch.float32) * 0.5
    )
    query_start_loc_cpu = _build_query_start_loc(
        torch, tokens, args.num_seqs, q_len=q_len
    )
    swa_pool_cpu, swa_records_cpu, swa_slots_cpu = _build_swa_inputs(
        torch, rng, tokens, window_size, query_start_loc_cpu, q_len=q_len
    )
    main_pool_cpu, rope_pool_cpu, pages_cpu, main_records_cpu = _build_main_kv_inputs(
        torch,
        rng,
        tokens,
        args.kv_len,
        args.compress_ratio,
        args.page_size,
        topk,
        query_start_loc_cpu,
        args.unique_per_chunk,
    )
    compacted_records_cpu = _compact_records_torch_ref(torch, main_records_cpu)

    # Move rank-local and replicated tensors to TPU.
    q_hp = (
        q_full_cpu[:, tp_rank * heads_local : (tp_rank + 1) * heads_local, :]
        .contiguous()
        .to("tpu")
    )
    sink_hp = (
        sink_full_cpu[tp_rank * heads_local : (tp_rank + 1) * heads_local]
        .contiguous()
        .to("tpu")
    )
    sink_full = sink_full_cpu.contiguous().to("tpu")
    swa_pool = swa_pool_cpu.to("tpu")
    swa_records = swa_records_cpu.to("tpu")
    swa_slots = swa_slots_cpu.to("tpu")
    main_pool = main_pool_cpu.to("tpu")
    rope_pool = rope_pool_cpu.to("tpu")
    pages = pages_cpu.to("tpu")
    query_start_loc = query_start_loc_cpu.to("tpu")
    main_records = main_records_cpu.to("tpu")
    compacted_records = compacted_records_cpu.to("tpu")

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    with common.model_wrapper_context():
        mesh = get_vllm_model_wrapper_context().mesh
        initialize_paged_attention()

        def _swa_fn(
            q_in: jax.Array, pool_in: jax.Array, rec_in: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jax.shard_map(
                lambda q_a, p_a, r_a: _attend_window(
                    q_a,
                    p_a,
                    r_a,
                    _attention_state(q_a),
                    scale,
                    max_requests=max_requests,
                ),
                mesh=mesh,
                in_specs=(P(),) * 3,
                out_specs=P(),
                check_vma=False,
            )(q_in, pool_in, rec_in)

        def _swa_ref_fn(
            q_in: jax.Array, pool_in: jax.Array, rec_in: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jax.shard_map(
                lambda q_a, p_a, r_a: _attend(
                    q_a,
                    p_a,
                    r_a,
                    _attention_state(q_a),
                    scale,
                ),
                mesh=mesh,
                in_specs=(P(),) * 3,
                out_specs=P(),
                check_vma=False,
            )(q_in, pool_in, rec_in)

        def _compact_fn(rec_in: jax.Array) -> jax.Array:
            return jax.shard_map(
                compact_records,
                mesh=mesh,
                in_specs=(P(),),
                out_specs=P(),
                check_vma=False,
            )(rec_in)

        def _sc_gather_fn(
            m_pool: jax.Array,
            r_pool: jax.Array,
            c_rec: jax.Array,
            pgs: jax.Array,
            q_loc: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return jax.shard_map(
                functools.partial(_sc_gather_only_jax, page_size=args.page_size),
                mesh=mesh,
                in_specs=(P(),) * 5,
                out_specs=P(),
                check_vma=False,
            )(m_pool, r_pool, c_rec, pgs, q_loc)

        def _sc_gather_union_fn(
            m_pool: jax.Array,
            r_pool: jax.Array,
            c_rec: jax.Array,
            pgs: jax.Array,
            q_loc: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return jax.shard_map(
                functools.partial(
                    _sc_gather_union_only_jax,
                    page_size=args.page_size,
                    union_cap=union_cap,
                ),
                mesh=mesh,
                in_specs=(P(),) * 5,
                out_specs=P(),
                check_vma=False,
            )(m_pool, r_pool, c_rec, pgs, q_loc)

        def _pregather_fn(
            m_pool: jax.Array,
            r_pool: jax.Array,
            c_rec: jax.Array,
            pgs: jax.Array,
            q_loc: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jax.shard_map(
                functools.partial(_pregather_all_chunks_jax, page_size=args.page_size),
                mesh=mesh,
                in_specs=(P(),) * 5,
                out_specs=P(),
                check_vma=False,
            )(m_pool, r_pool, c_rec, pgs, q_loc)

        def _mla_tc_fn(
            q_in: jax.Array,
            n_chunks: jax.Array,
            r_chunks: jax.Array,
            kv_l: jax.Array,
            q_loc: jax.Array,
            snk: jax.Array,
            s_num: jax.Array,
            s_den: jax.Array,
            s_max: jax.Array,
        ) -> jax.Array:
            return jax.shard_map(
                functools.partial(
                    _mla_tc_only_jax,
                    scale=scale,
                    max_num_seqs=args.num_seqs,
                ),
                mesh=mesh,
                in_specs=(P(),) * 9,
                out_specs=P(),
                check_vma=False,
            )(q_in, n_chunks, r_chunks, kv_l, q_loc, snk, s_num, s_den, s_max)

        def _precompact_all_fn(
            q_in: jax.Array,
            s_pool: jax.Array,
            m_pool: jax.Array,
            r_pool: jax.Array,
            s_rec: jax.Array,
            c_rec: jax.Array,
            pgs: jax.Array,
            q_loc: jax.Array,
            snk: jax.Array,
        ) -> jax.Array:
            return jax.shard_map(
                functools.partial(
                    _sparse_fp8_attention_precompact_jax,
                    scale=scale,
                    page_size=args.page_size,
                    max_requests=max_requests,
                ),
                mesh=mesh,
                in_specs=(P(),) * 9,
                out_specs=P(),
                check_vma=False,
            )(q_in, s_pool, m_pool, r_pool, s_rec, c_rec, pgs, q_loc, snk)

        swa_op = pallas.jax_op(
            "pallas::dsv41_bench_swa_only",
            _swa_fn,
            mesh=mesh,
            input_partition_specs=(P(),) * 3,
        )
        swa_ref_op = pallas.jax_op(
            "pallas::dsv41_bench_swa_dense_ref",
            _swa_ref_fn,
            mesh=mesh,
            input_partition_specs=(P(),) * 3,
        )
        compact_op = pallas.jax_op(
            "pallas::dsv41_bench_compact_only",
            _compact_fn,
            mesh=mesh,
            input_partition_specs=(P(),),
        )
        sc_gather_op = pallas.jax_op(
            "pallas::dsv41_bench_sc_gather_only",
            _sc_gather_fn,
            mesh=mesh,
            input_partition_specs=(P(),) * 5,
        )
        sc_gather_union_op = pallas.jax_op(
            "pallas::dsv41_bench_sc_gather_union_only",
            _sc_gather_union_fn,
            mesh=mesh,
            input_partition_specs=(P(),) * 5,
        )
        pregather_op = pallas.jax_op(
            "pallas::dsv41_bench_pregather_chunks",
            _pregather_fn,
            mesh=mesh,
            input_partition_specs=(P(),) * 5,
        )
        mla_tc_op = pallas.jax_op(
            "pallas::dsv41_bench_mla_tc_only",
            _mla_tc_fn,
            mesh=mesh,
            input_partition_specs=(P(),) * 9,
        )
        precompact_all_op = pallas.jax_op(
            "pallas::dsv41_bench_sparse_fp8_precompact",
            _precompact_all_fn,
            mesh=mesh,
            input_partition_specs=(P(),) * 9,
        )

        class _MockMainCache:
            def __init__(self, raw, rope, page_sz):
                self.raw = raw
                self.rope = rope
                self.shape = (raw.shape[0] // page_sz, page_sz, 512)

        class _MockSwaCache:
            def __init__(self, raw):
                self.raw = raw

        swa_wrapper = _MockSwaCache(swa_pool)
        main_wrapper = _MockMainCache(main_pool, rope_pool, args.page_size)
        pad_mask = (swa_slots < 0)[:, None, None]

        # Helper for Token-Parallel All-to-All (`[T, H/8, D] <-> [T/8, H, D]`).
        tp_group = get_tp_group().device_group
        can_tp = (
            tokens % ctx.tp_size == 0
            and (tokens // ctx.tp_size) >= 4
            and (tokens // ctx.tp_size) % 4 == 0
        )
        t_shard = (tokens // ctx.tp_size) if can_tp else tokens
        t_start = (tp_rank * t_shard) if can_tp else 0
        t_end = ((tp_rank + 1) * t_shard) if can_tp else tokens
        swa_records_tp = swa_records[t_start:t_end].contiguous()
        main_records_tp = main_records[t_start:t_end].contiguous()
        compacted_records_tp = compacted_records[t_start:t_end].contiguous()
        query_start_loc_tp = (
            (query_start_loc - t_start).clamp(min=0, max=t_shard).to(torch.int32)
        )

        def _all_to_all_hp_to_tp(x_hp: Any) -> Any:
            # x_hp: [T, H_local, D] -> split T into tp_size chunks of
            # [T/8, H_local, D], all_to_all, concat heads -> [T/8, H_total, D].
            send = x_hp.reshape(
                ctx.tp_size, t_shard, heads_local, head_dim
            ).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=tp_group)
            return (
                recv.permute(1, 0, 2, 3)
                .reshape(t_shard, heads_total, head_dim)
                .contiguous()
            )

        def _all_to_all_tp_to_hp(x_tp: Any) -> Any:
            # x_tp: [T/8, H_total, D] -> [tp_size, T/8, H_local, D] ->
            # all_to_all -> [T, H_local, D].
            send = (
                x_tp.reshape(t_shard, ctx.tp_size, heads_local, head_dim)
                .permute(1, 0, 2, 3)
                .contiguous()
            )
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=tp_group)
            return recv.reshape(tokens, heads_local, head_dim).contiguous()

        q_tp_static = q_full_cpu[t_start:t_end].contiguous().to("tpu")
        pad_mask_tp = (swa_slots[t_start:t_end] < 0)[:, None, None]
        tp_rank_t = torch.tensor([tp_rank], dtype=torch.int32, device="tpu")

        # Pre-compute state / chunks when needed by `mla_tc_only`.
        nope_chunks = rope_chunks = kv_lens_t = None
        swa_max_t = swa_den_t = swa_num_t = None
        if args.subop == "mla_tc_only":
            if args.mode == "token_parallel":
                swa_max_t, swa_den_t, swa_num_t = swa_op(
                    q_tp_static, swa_pool, swa_records_tp
                )
                nope_chunks, rope_chunks, kv_lens_t = pregather_op(
                    main_pool,
                    rope_pool,
                    compacted_records_tp,
                    pages,
                    query_start_loc_tp,
                )
            else:
                swa_max_t, swa_den_t, swa_num_t = swa_op(q_hp, swa_pool, swa_records)
                nope_chunks, rope_chunks, kv_lens_t = pregather_op(
                    main_pool, rope_pool, compacted_records, pages, query_start_loc
                )
            common.synchronize(
                (swa_max_t, swa_den_t, swa_num_t, nope_chunks, rope_chunks, kv_lens_t)
            )

        # Define the timed `step()` function for the requested `(subop, mode)`.
        if args.subop == "swa_only":
            if args.mode == "token_parallel":

                def step():
                    m, d, n = swa_op(q_tp_static, swa_pool, swa_records_tp)
                    return {"max": m, "den": d, "num": n}
            else:

                def step():
                    m, d, n = swa_op(q_hp, swa_pool, swa_records)
                    return {"max": m, "den": d, "num": n}

        elif args.subop == "compact_records_only":
            if args.mode == "token_parallel":

                def step():
                    return {"compacted": compact_op(main_records_tp)}
            else:

                def step():
                    return {"compacted": compact_op(main_records)}

        elif args.subop == "sc_gather_only":
            if args.mode == "token_parallel":

                def step():
                    n_s, r_s = sc_gather_op(
                        main_pool,
                        rope_pool,
                        compacted_records_tp,
                        pages,
                        query_start_loc_tp,
                    )
                    return {"nope_sample": n_s, "rope_sample": r_s}
            elif args.mode == "union_gather":

                def step():
                    n_s, r_s = sc_gather_union_op(
                        main_pool,
                        rope_pool,
                        compacted_records,
                        pages,
                        query_start_loc,
                    )
                    return {"nope_sample": n_s, "rope_sample": r_s}
            else:

                def step():
                    n_s, r_s = sc_gather_op(
                        main_pool, rope_pool, compacted_records, pages, query_start_loc
                    )
                    return {"nope_sample": n_s, "rope_sample": r_s}

        elif args.subop == "mla_tc_only":
            if args.mode == "token_parallel":

                def step():
                    res = mla_tc_op(
                        q_tp_static,
                        nope_chunks,
                        rope_chunks,
                        kv_lens_t,
                        query_start_loc_tp,
                        sink_full,
                        swa_num_t,
                        swa_den_t,
                        swa_max_t,
                    )
                    return {"out": res.masked_fill(pad_mask_tp, 0)}
            else:

                def step():
                    res = mla_tc_op(
                        q_hp,
                        nope_chunks,
                        rope_chunks,
                        kv_lens_t,
                        query_start_loc,
                        sink_hp,
                        swa_num_t,
                        swa_den_t,
                        swa_max_t,
                    )
                    return {"out": res.masked_fill(pad_mask, 0)}

        else:  # subop == "all"
            if args.mode == "precompact":

                def step():
                    res = sparse_fp8_attention(
                        q_hp,
                        swa_wrapper,
                        main_wrapper,
                        swa_records,
                        compacted_records,
                        sink_hp,
                        scale,
                        pages=pages,
                        starts=query_start_loc,
                        max_requests=max_requests,
                        precompacted=True,
                    )
                    return {"out": res.masked_fill(pad_mask, 0)}
            elif args.mode == "union_gather":

                def step():
                    res = sparse_fp8_attention(
                        q_hp,
                        swa_wrapper,
                        main_wrapper,
                        swa_records,
                        compacted_records,
                        sink_hp,
                        scale,
                        pages=pages,
                        starts=query_start_loc,
                        max_requests=max_requests,
                        precompacted=True,
                        union_gather=True,
                        union_cap=union_cap,
                    )
                    return {"out": res.masked_fill(pad_mask, 0)}
            elif args.mode == "token_parallel":

                def step():
                    res_hp = sparse_fp8_attention(
                        q_hp,
                        swa_wrapper,
                        main_wrapper,
                        swa_records,
                        compacted_records,
                        sink_hp,
                        scale,
                        pages=pages,
                        starts=query_start_loc,
                        max_requests=max_requests,
                        precompacted=True,
                        token_parallel=True,
                        full_sink=sink_full,
                        tp_rank=tp_rank_t,
                    )
                    return {"out": res_hp.masked_fill(pad_mask, 0)}
            else:

                def step():
                    res = sparse_fp8_attention(
                        q_hp,
                        swa_wrapper,
                        main_wrapper,
                        swa_records,
                        main_records,
                        sink_hp,
                        scale,
                        pages=pages,
                        starts=query_start_loc,
                        max_requests=max_requests,
                        precompacted=False,
                    )
                    return {"out": res.masked_fill(pad_mask, 0)}

        common.synchronize(
            (
                q_hp,
                q_tp_static,
                swa_pool,
                main_pool,
                rope_pool,
                swa_records,
                main_records,
            )
        )
        hbm_before = common.hbm_stats()
        samples, out = common.time_steps(
            ctx, step, warmup=args.warmup, iters=args.iters
        )
        profile_dir = common.profile_steps(
            ctx, args, step, label=f"sparse_mla_swa_{args.subop}_{args.mode}"
        )
        hbm_after = common.hbm_stats()

        # ------------------------------------------------------------------
        # Mandatory Correctness Verification (`--verify`)
        # ------------------------------------------------------------------
        verify_report: dict[str, Any] = {}
        if args.verify:
            if args.subop == "swa_only":
                q_chk = q_tp_static if args.mode == "token_parallel" else q_hp
                rec_chk = (
                    swa_records_tp if args.mode == "token_parallel" else swa_records
                )
                slots_chk = (
                    swa_slots[t_start:t_end]
                    if args.mode == "token_parallel"
                    else swa_slots
                )
                ref_m, ref_d, ref_n = swa_ref_op(q_chk, swa_pool, rec_chk)
                common.synchronize((ref_m, ref_d, ref_n))
                valid_q = (slots_chk >= 0).cpu()
                verify_report["vs_dense_attend"] = common.verify_outputs(
                    {
                        "max": out["max"].cpu()[valid_q],
                        "den": out["den"].cpu()[valid_q],
                        "num": out["num"].cpu()[valid_q],
                    },
                    {
                        "max": ref_m.cpu()[valid_q],
                        "den": ref_d.cpu()[valid_q],
                        "num": ref_n.cpu()[valid_q],
                    },
                    common.LOSSLESS,
                    label="swa_only vs _attend",
                )
                criteria = common.LOSSLESS

            elif args.subop == "compact_records_only":
                exp_comp = (
                    compacted_records_cpu[t_start:t_end]
                    if args.mode == "token_parallel"
                    else compacted_records_cpu
                )
                verify_report["vs_torch_sort_ref"] = common.verify_outputs(
                    {"compacted": out["compacted"].cpu()},
                    {"compacted": exp_comp},
                    common.EXACT,
                    label="compact_records_only vs torch_sort_ref",
                )
                criteria = common.EXACT

            elif args.subop == "sc_gather_only":
                # Verify gathered first-column samples against direct pool indexing.
                rec_check = (
                    compacted_records_cpu[t_start:t_end]
                    if args.mode == "token_parallel"
                    else compacted_records_cpu
                )
                qsl_check = (
                    (query_start_loc_cpu - t_start).clamp(min=0, max=t_shard)
                    if args.mode == "token_parallel"
                    else query_start_loc_cpu
                )
                valid_count = int(qsl_check[-1].item())
                tok_per_seq = qsl_check[1:] - qsl_check[:-1]
                seq_seg = torch.repeat_interleave(
                    torch.arange(args.num_seqs, dtype=torch.int64), tok_per_seq
                )
                first_col = rec_check[:valid_count, 0].long()
                valid_first = first_col >= 0
                phys_page = pages_cpu[
                    seq_seg, (first_col.clamp(min=0) // args.page_size)
                ].long()
                phys_row = phys_page * args.page_size + (
                    first_col.clamp(min=0) % args.page_size
                )
                expected_nope = main_pool_cpu[phys_row, :4]
                verify_report["vs_cpu_pool_direct"] = common.verify_outputs(
                    {
                        "nope_sample": out["nope_sample"].cpu()[:valid_count][
                            valid_first
                        ]
                    },
                    {"nope_sample": expected_nope[valid_first]},
                    common.EXACT,
                    label="sc_gather_only vs direct_pool_index",
                )
                if args.mode == "union_gather":
                    base_n_s, base_r_s = sc_gather_op(
                        main_pool, rope_pool, compacted_records, pages, query_start_loc
                    )
                    common.synchronize((base_n_s, base_r_s))
                    valid_first4 = (rec_check[:valid_count, :4] >= 0).all(dim=1)
                    verify_report["union_vs_baseline_sc_gather_exact"] = (
                        common.verify_outputs(
                            {
                                "nope_sample": out["nope_sample"].cpu()[:valid_count][
                                    valid_first
                                ],
                                "rope_sample": out["rope_sample"].cpu()[:valid_count][
                                    valid_first4
                                ],
                            },
                            {
                                "nope_sample": base_n_s.cpu()[:valid_count][
                                    valid_first
                                ],
                                "rope_sample": base_r_s.cpu()[:valid_count][
                                    valid_first4
                                ],
                            },
                            common.EXACT,
                            label="union_gather vs baseline sc_gather_only",
                        )
                    )
                    if args.num_seqs == 1:
                        rng_s4 = torch.Generator(device="cpu").manual_seed(
                            args.seed + 2027
                        )
                        qsl_s4_cpu = _build_query_start_loc(torch, tokens, 4)
                        _, _, pages_s4_cpu, main_rec_s4_cpu = _build_main_kv_inputs(
                            torch,
                            rng_s4,
                            tokens,
                            args.kv_len,
                            args.compress_ratio,
                            args.page_size,
                            topk,
                            qsl_s4_cpu,
                            args.unique_per_chunk,
                        )
                        comp_s4_cpu = _compact_records_torch_ref(torch, main_rec_s4_cpu)
                        comp_s4 = comp_s4_cpu.to("tpu")
                        pages_s4 = pages_s4_cpu.to("tpu")
                        qsl_s4 = qsl_s4_cpu.to("tpu")
                        u_n_s4, u_r_s4 = sc_gather_union_op(
                            main_pool, rope_pool, comp_s4, pages_s4, qsl_s4
                        )
                        common.synchronize((u_n_s4, u_r_s4))
                        b_n_s4, b_r_s4 = sc_gather_op(
                            main_pool, rope_pool, comp_s4, pages_s4, qsl_s4
                        )
                        common.synchronize((b_n_s4, b_r_s4))
                        vc_s4 = int(qsl_s4_cpu[-1].item())
                        vf_s4 = comp_s4_cpu[:vc_s4, 0] >= 0
                        vf4_s4 = (comp_s4_cpu[:vc_s4, :4] >= 0).all(dim=1)
                        verify_report["union_num_seqs_4_vs_baseline_exact"] = (
                            common.verify_outputs(
                                {
                                    "nope_sample": u_n_s4.cpu()[:vc_s4][vf_s4],
                                    "rope_sample": u_r_s4.cpu()[:vc_s4][vf4_s4],
                                },
                                {
                                    "nope_sample": b_n_s4.cpu()[:vc_s4][vf_s4],
                                    "rope_sample": b_r_s4.cpu()[:vc_s4][vf4_s4],
                                },
                                common.EXACT,
                                label="union_gather (num_seqs=4) vs baseline",
                            )
                        )
                criteria = common.EXACT

            elif args.subop == "mla_tc_only":
                base_out = sparse_fp8_attention(
                    q_hp,
                    swa_wrapper,
                    main_wrapper,
                    swa_records,
                    main_records,
                    sink_hp,
                    scale,
                    pages=pages,
                    starts=query_start_loc,
                    max_requests=max_requests,
                ).masked_fill(pad_mask, 0)
                common.synchronize(base_out)
                if args.mode == "token_parallel":
                    act_hp = _all_to_all_tp_to_hp(out["out"]).masked_fill(pad_mask, 0)
                    common.synchronize(act_hp)
                    verify_report["vs_full_sparse_fp8_attention"] = (
                        common.verify_outputs(
                            {"out": act_hp.cpu()},
                            {"out": base_out.cpu()},
                            common.LOSSLESS,
                            label="mla_tc_only[tp] vs sparse_fp8_attention",
                        )
                    )
                    criteria = common.LOSSLESS
                else:
                    verify_report["vs_full_sparse_fp8_attention"] = (
                        common.verify_outputs(
                            {"out": out["out"].cpu()},
                            {"out": base_out.cpu()},
                            common.EXACT,
                            label="mla_tc_only vs full sparse_fp8_attention",
                        )
                    )
                    criteria = common.EXACT

            else:  # subop == "all"
                base_out = sparse_fp8_attention(
                    q_hp,
                    swa_wrapper,
                    main_wrapper,
                    swa_records,
                    main_records,
                    sink_hp,
                    scale,
                    pages=pages,
                    starts=query_start_loc,
                    max_requests=max_requests,
                ).masked_fill(pad_mask, 0)
                common.synchronize(base_out)
                precompact_out = precompact_all_op(
                    q_hp,
                    swa_pool,
                    main_pool,
                    rope_pool,
                    swa_records,
                    compacted_records,
                    pages,
                    query_start_loc,
                    sink_hp,
                ).masked_fill(pad_mask, 0)
                common.synchronize(precompact_out)
                verify_report["precompact_vs_baseline_exact"] = common.verify_outputs(
                    {"out": precompact_out.cpu()},
                    {"out": base_out.cpu()},
                    common.EXACT,
                    label="precompact vs baseline",
                )
                if can_tp:
                    q_tp_v = _all_to_all_hp_to_tp(q_hp)
                    tp_out_local = precompact_all_op(
                        q_tp_v,
                        swa_pool,
                        main_pool,
                        rope_pool,
                        swa_records_tp,
                        compacted_records_tp,
                        pages,
                        query_start_loc_tp,
                        sink_full,
                    )
                    tp_out = _all_to_all_tp_to_hp(tp_out_local).masked_fill(pad_mask, 0)
                    common.synchronize(tp_out)
                    verify_report["token_parallel_vs_baseline"] = common.verify_outputs(
                        {"out": tp_out.cpu()},
                        {"out": base_out.cpu()},
                        common.LOSSLESS,
                        label="token_parallel vs baseline",
                    )
                verify_report["selected_mode_vs_baseline"] = common.verify_outputs(
                    {"out": out["out"].cpu()},
                    {"out": base_out.cpu()},
                    common.EXACT if args.mode != "token_parallel" else common.LOSSLESS,
                    label=f"mode={args.mode} vs baseline",
                )
                if args.mode == "token_parallel" and args.num_seqs == 1:
                    qsl_s4 = _build_query_start_loc(torch, tokens, 4).to("tpu")
                    pages_s4 = pages.repeat(4, 1).contiguous()
                    pad_s4 = (torch.arange(tokens, device="tpu") >= qsl_s4[-1])[
                        :, None, None
                    ]
                    base_s4 = sparse_fp8_attention(
                        q_hp,
                        swa_wrapper,
                        main_wrapper,
                        swa_records,
                        compacted_records,
                        sink_hp,
                        scale,
                        pages=pages_s4,
                        starts=qsl_s4,
                        max_requests=4,
                        precompacted=True,
                        token_parallel=False,
                    ).masked_fill(pad_s4, 0)
                    common.synchronize(base_s4)
                    tp_s4 = sparse_fp8_attention(
                        q_hp,
                        swa_wrapper,
                        main_wrapper,
                        swa_records,
                        compacted_records,
                        sink_hp,
                        scale,
                        pages=pages_s4,
                        starts=qsl_s4,
                        max_requests=4,
                        precompacted=True,
                        token_parallel=True,
                        full_sink=sink_full,
                        tp_rank=tp_rank_t,
                    ).masked_fill(pad_s4, 0)
                    common.synchronize(tp_s4)
                    verify_report["token_parallel_num_seqs_4_vs_baseline"] = (
                        common.verify_outputs(
                            {"out": tp_s4.cpu()},
                            {"out": base_s4.cpu()},
                            common.LOSSLESS,
                            label="token_parallel (num_seqs=4) vs baseline",
                        )
                    )
                if args.mode == "union_gather" and args.num_seqs == 1:
                    rng_s4 = torch.Generator(device="cpu").manual_seed(args.seed + 2027)
                    qsl_s4_cpu = _build_query_start_loc(torch, tokens, 4)
                    _, _, pages_s4_cpu, main_rec_s4_cpu = _build_main_kv_inputs(
                        torch,
                        rng_s4,
                        tokens,
                        args.kv_len,
                        args.compress_ratio,
                        args.page_size,
                        topk,
                        qsl_s4_cpu,
                        args.unique_per_chunk,
                    )
                    comp_s4 = _compact_records_torch_ref(torch, main_rec_s4_cpu).to(
                        "tpu"
                    )
                    pages_s4 = pages_s4_cpu.to("tpu")
                    qsl_s4 = qsl_s4_cpu.to("tpu")
                    pad_s4 = (torch.arange(tokens, device="tpu") >= qsl_s4[-1])[
                        :, None, None
                    ]
                    base_s4 = sparse_fp8_attention(
                        q_hp,
                        swa_wrapper,
                        main_wrapper,
                        swa_records,
                        comp_s4,
                        sink_hp,
                        scale,
                        pages=pages_s4,
                        starts=qsl_s4,
                        max_requests=4,
                        precompacted=True,
                    ).masked_fill(pad_s4, 0)
                    common.synchronize(base_s4)
                    ug_s4 = sparse_fp8_attention(
                        q_hp,
                        swa_wrapper,
                        main_wrapper,
                        swa_records,
                        comp_s4,
                        sink_hp,
                        scale,
                        pages=pages_s4,
                        starts=qsl_s4,
                        max_requests=4,
                        precompacted=True,
                        union_gather=True,
                        union_cap=union_cap,
                    ).masked_fill(pad_s4, 0)
                    common.synchronize(ug_s4)
                    verify_report["union_gather_num_seqs_4_vs_baseline_exact"] = (
                        common.verify_outputs(
                            {"out": ug_s4.cpu()},
                            {"out": base_s4.cpu()},
                            common.EXACT,
                            label="union_gather (num_seqs=4) vs baseline",
                        )
                    )
                if q_len > 0:
                    s_max, s_den, s_num = swa_op(q_hp, swa_pool, swa_records)
                    n_chk, r_chk, k_len = pregather_op(
                        main_pool, rope_pool, compacted_records, pages, query_start_loc
                    )
                    common.synchronize((s_max, s_den, s_num, n_chk, r_chk, k_len))
                    valid_q = (swa_slots >= 0).cpu()
                    suite_ops: list[tuple[str, Any]] = [
                        (
                            "swa_only",
                            lambda: swa_op(q_hp, swa_pool, swa_records),
                        ),
                        (
                            "swa_dense_ref",
                            lambda: swa_ref_op(q_hp, swa_pool, swa_records),
                        ),
                        (
                            "compact_records_only",
                            lambda: compact_op(main_records),
                        ),
                        (
                            "sc_gather_only",
                            lambda: sc_gather_op(
                                main_pool,
                                rope_pool,
                                compacted_records,
                                pages,
                                query_start_loc,
                            ),
                        ),
                        (
                            "mla_tc_only",
                            lambda: mla_tc_op(
                                q_hp,
                                n_chk,
                                r_chk,
                                k_len,
                                query_start_loc,
                                sink_hp,
                                s_num,
                                s_den,
                                s_max,
                            ).masked_fill(pad_mask, 0),
                        ),
                        (
                            "all_precompact",
                            lambda: precompact_all_op(
                                q_hp,
                                swa_pool,
                                main_pool,
                                rope_pool,
                                swa_records,
                                compacted_records,
                                pages,
                                query_start_loc,
                                sink_hp,
                            ).masked_fill(pad_mask, 0),
                        ),
                    ]
                    if can_tp:
                        suite_ops.append(
                            (
                                "all_token_parallel",
                                lambda: _all_to_all_tp_to_hp(
                                    precompact_all_op(
                                        _all_to_all_hp_to_tp(q_hp),
                                        swa_pool,
                                        main_pool,
                                        rope_pool,
                                        swa_records_tp,
                                        compacted_records_tp,
                                        pages,
                                        query_start_loc_tp,
                                        sink_full,
                                    )
                                ).masked_fill(pad_mask, 0),
                            )
                        )
                    subop_suite_res: dict[str, Any] = {}
                    swa_win_out = swa_ref_out = None
                    for s_name, s_fn in suite_ops:
                        s_samples, s_out = common.time_steps(
                            ctx, s_fn, warmup=2, iters=5
                        )
                        if s_name == "swa_only":
                            swa_win_out = s_out
                        elif s_name == "swa_dense_ref":
                            swa_ref_out = s_out
                        subop_suite_res[s_name] = {
                            "wall_median_ms": statistics.median(s_samples),
                            "wall_min_ms": min(s_samples),
                        }
                    assert swa_win_out is not None and swa_ref_out is not None
                    verify_report["swa_window_vs_dense_attend"] = common.verify_outputs(
                        {
                            "max": swa_win_out[0].cpu()[valid_q],
                            "den": swa_win_out[1].cpu()[valid_q],
                            "num": swa_win_out[2].cpu()[valid_q],
                        },
                        {
                            "max": swa_ref_out[0].cpu()[valid_q],
                            "den": swa_ref_out[1].cpu()[valid_q],
                            "num": swa_ref_out[2].cpu()[valid_q],
                        },
                        common.LOSSLESS,
                        label="swa_only vs _attend",
                    )
                    verify_report["decode_subop_suite"] = subop_suite_res
                criteria = (
                    common.EXACT if args.mode != "token_parallel" else common.LOSSLESS
                )
        else:
            criteria = common.LOSSLESS

        case_suffix = (
            f"_Q{q_len}_W{window_size}"
            if (q_len > 0 or window_size != geo.sliding_window)
            else ""
        )
        case_name = (
            f"sparse_mla_swa_{args.subop}_{args.mode}_T{tokens}_S{args.num_seqs}"
            f"_KV{args.kv_len}_CR{args.compress_ratio}_P{args.page_size}{case_suffix}"
        )
        g_path = common.golden_path(args.golden_dir, case_name, ctx.rank)
        if args.update_golden or g_path.exists():
            golden = common.check_against_golden(ctx, args, case_name, out, criteria)
        else:
            golden = {
                "golden": "in_process_reference_verified_only",
                "expected_golden_path": str(g_path),
            }

    hw = common.Tpu7xRoofline()
    roof = _compute_roofline(
        hw,
        args.subop,
        args.mode,
        tokens,
        ctx.tp_size,
        heads_total,
        head_dim,
        window_size,
        topk,
        statistics.median(samples),
        union_cap=union_cap,
        q_len=q_len,
    )
    return {
        "subop": args.subop,
        "mode": args.mode,
        "chunk_size": tokens,
        "num_seqs": args.num_seqs,
        "q_len": q_len,
        "window_size": window_size,
        "kv_len": args.kv_len,
        "compress_ratio": args.compress_ratio,
        "page_size": args.page_size,
        "wall_samples_ms": samples,
        "verify": verify_report,
        "golden": golden,
        "roofline_vs_wall": roof,
        "hbm": {"before": hbm_before, "after": hbm_after},
        "profile_dir": profile_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser, SUBOPS)
    parser.add_argument("--mode", choices=list(MODES), default="baseline")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--compress-ratio", type=int, default=2, choices=[1, 2])
    parser.add_argument("--unique-per-chunk", type=int, default=4096)
    parser.add_argument(
        "--q-len",
        type=int,
        default=0,
        help="Per-request query length (0=prefill, 1=Decode, 6=DSpark Verify).",
    )
    parser.add_argument(
        "--swa-width",
        type=int,
        default=0,
        help="Override sliding window width (e.g. 512 causal, 576 DSpark).",
    )
    common.run_main("sparse_mla_swa", parser, worker)


if __name__ == "__main__":
    main()
