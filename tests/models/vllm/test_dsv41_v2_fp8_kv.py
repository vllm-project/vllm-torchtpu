"""CPU checks for the DeepSeek V4.1 FP8 main-KV record and RoPE companion."""

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import compact_records
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    NOPE_DIM,
    PagedCacheRows,
    encode_fp8_kv,
    scatter_cache_values,
)


def _latent(n, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n, 512)) * rng.uniform(0.01, 30, (n, 1))
    return jnp.asarray(values, jnp.bfloat16)


def _decode(words, rope):
    """Dequantize records back to float32 ``[n, 512]``."""
    record = np.asarray(csa_cache_layout.words_to_slabs(words)).reshape(-1, 512)
    values = record[:, :NOPE_DIM].view(jnp.float8_e4m3fn).astype(np.float32)
    scales = record[:, NOPE_DIM:].view(jnp.float8_e8m0fnu).astype(np.float32)
    nope = values * np.tile(scales, NOPE_DIM // 64)
    bits = np.asarray(rope).view(np.uint32)
    halves = np.concatenate((bits & 0xFFFF, bits >> 16), axis=-1).astype(np.uint16)
    return np.concatenate((nope, halves.view(jnp.bfloat16).astype(np.float32)), -1)


def test_encode_matches_exact_lane_periodic_quantizer():
    """V4's lane-periodic UE8M0 quantizer, with exact power-of-two scales."""
    latent = _latent(37)
    # Exact e4m3 ties (bf16 inputs make them common) and zero lanes.
    latent = latent.at[0, :NOPE_DIM].set(0).at[1, :NOPE_DIM].set(76 * 2.0**-19)
    words, rope = encode_fp8_kv(latent)
    assert words.shape == (37, 128) and words.dtype == jnp.int32
    assert rope.shape == (37, 32) and rope.dtype == jnp.int32
    nope = np.asarray(latent, np.float64)[:, :NOPE_DIM]
    amax = np.maximum(np.abs(nope).reshape(37, -1, 64).max(axis=1), 1e-4)
    exponent = np.ceil(np.log2(amax / 448)).astype(np.int64)
    q = (nope / np.tile(2.0**exponent, NOPE_DIM // 64)).astype(np.float32)
    expected = np.concatenate(
        (
            q.astype(ml_dtypes.float8_e4m3fn).view(np.uint8),
            (exponent + 127).astype(np.uint8),
        ),
        axis=-1,
    )
    got = np.asarray(csa_cache_layout.words_to_slabs(words)).reshape(37, 512)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(
        np.asarray(rope), np.asarray(csa_cache_layout.rope_words(latent[:, NOPE_DIM:]))
    )


def test_encode_round_trip_error():
    latent = _latent(64, seed=1)
    decoded = _decode(*encode_fp8_kv(latent))
    reference = np.asarray(latent, np.float32)
    np.testing.assert_array_equal(decoded[:, NOPE_DIM:], reference[:, NOPE_DIM:])
    # e4m3 keeps 3 mantissa bits; power-of-two scales add at most one more.
    nope = reference[:, :NOPE_DIM]
    lane_max = np.tile(np.abs(nope).reshape(64, -1, 64).max(axis=1), NOPE_DIM // 64)
    assert np.all(np.abs(decoded[:, :NOPE_DIM] - nope) <= lane_max / 8 + 1e-6)


def test_companion_scatter_places_slot_by_row():
    rows = 16
    companion = jnp.zeros((rows * 32 // 128, 128), jnp.int32)
    records = jnp.asarray([5, -1, 0, 15], jnp.int32)
    values = jnp.arange(4 * 32, dtype=jnp.int32).reshape(4, 32) + 1
    got = np.asarray(scatter_cache_values(companion, records, values)).reshape(-1, 32)
    expected = np.zeros((rows, 32), np.int32)
    for record, value in zip(np.asarray(records), np.asarray(values)):
        if record >= 0:
            expected[record] = value
    np.testing.assert_array_equal(got, expected)


def test_paged_cache_rows_fp8_cpu_write():
    pages, block = 3, 4
    raw = torch.zeros((pages * block * 2, 128), dtype=torch.int32)
    rope = torch.zeros((raw.shape[0] // 4, 128), dtype=torch.int32)
    # Second half of each 2-row page stride belongs to this layer.
    layout = ((pages, block, 512), torch.uint8, 2 * block * 512, block * 512)
    cache = PagedCacheRows(raw, layout, rope)
    latent = _latent(3, seed=2)
    slots = torch.tensor([0, -1, 9])
    values = torch.from_numpy(np.asarray(latent, np.float32)).to(torch.bfloat16)
    cache.write(slots, values)
    words, rope_words = encode_fp8_kv(latent)
    for index, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        row = int(cache.row_indices(torch.tensor(slot)))
        np.testing.assert_array_equal(raw[row].numpy(), np.asarray(words[index]))
        np.testing.assert_array_equal(
            rope.view(-1, 32)[row].numpy(), np.asarray(rope_words[index])
        )
    written = {int(cache.row_indices(torch.tensor(s))) for s in (0, 9)}
    untouched = [r for r in range(raw.shape[0]) if r not in written]
    assert not raw[untouched].any()
    assert not rope.view(-1, 32)[untouched].any()


def test_paged_cache_rows_pages_match_row_indices():
    pages, block = 3, 4
    raw = torch.zeros((pages * block * 2, 128), dtype=torch.int32)
    rope = torch.zeros((raw.shape[0] // 4, 128), dtype=torch.int32)
    layout = ((pages, block, 512), torch.uint8, 2 * block * 512, block * 512)
    cache = PagedCacheRows(raw, layout, rope)
    table = torch.tensor([[2, 0], [1, 2]])
    physical = cache.pages(table)
    for request in range(2):
        for logical in range(2 * block):
            block_id = table[request, logical // block]
            expected = cache.row_indices(block_id * block + logical % block)
            got = physical[request, logical // block] * block + logical % block
            assert int(got) == int(expected)
    # A layer offset that is not a whole page cannot be expressed as pages.
    odd = ((pages, block, 512), torch.uint8, 2 * block * 512, 2 * 512)
    assert PagedCacheRows(raw, odd, rope).pages(table) is None


def test_compact_records_moves_invalid_to_tail():
    records = jnp.asarray([[-1, 7, -1, 3, 9], [-1, -1, -1, -1, -1]], jnp.int32)
    got = np.asarray(compact_records(records))
    assert got.shape == (2, 128)
    np.testing.assert_array_equal(got[0, :3], [3, 7, 9])
    assert np.all(got[0, 3:] == -1) and np.all(got[1] == -1)


def test_pad_compacted_records_matches_compact_records():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
        _compact_selected,
        _pad_compacted_records,
    )

    for width in (5, 128, 300, 512):
        rng = np.random.default_rng(width)
        raw = rng.integers(-1, 1024, (8, width), dtype=np.int32)
        raw[:, ::3] = -1
        raw_jnp = jnp.asarray(raw, jnp.int32)
        compacted_sel = _compact_selected(raw_jnp)
        np.testing.assert_array_equal(
            np.asarray(_pad_compacted_records(compacted_sel)),
            np.asarray(compact_records(raw_jnp)),
        )


def test_parallel_engram_embedding_cpu_offload_and_row_slice(monkeypatch):
    from vllm.models.deepseek_v41.common import engram as upstream_engram

    from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import (
        ParallelEngramEmbedding,
    )

    monkeypatch.setattr(
        upstream_engram, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(upstream_engram, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda: 0, raising=False
    )

    head_sizes = (16, 16, 16, 16)
    num_embeddings = sum(head_sizes)
    dim = 64

    with torch.device("meta"):
        monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", "1")
        emb_cpu = ParallelEngramEmbedding(
            num_embeddings, dim, head_sizes, block_size=32
        )
    assert emb_cpu.cpu_offload is True
    assert emb_cpu.weight.device.type == "cpu"
    assert emb_cpu.weight_scale_inv.device.type == "cpu"
    assert getattr(emb_cpu.weight, "dummy_weight_value", None) == 1.0
    assert getattr(emb_cpu.weight_scale_inv, "dummy_weight_value", None) == 127

    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", "0")
    emb_dev = ParallelEngramEmbedding(num_embeddings, dim, head_sizes, block_size=32)
    assert emb_dev.cpu_offload is False
    rng = torch.Generator().manual_seed(42)
    full_w = torch.randn(num_embeddings, dim, generator=rng).to(torch.float8_e4m3fn)
    full_s = torch.randint(
        120, 134, (num_embeddings, dim // 32), generator=rng, dtype=torch.uint8
    )

    # Full-table load via upstream shard loader vs row-sliced load via _load_local_rows
    emb_dev._weight_loader(emb_dev.weight, full_w)
    emb_dev._weight_loader(emb_dev.weight_scale_inv, full_s)
    start = emb_cpu.vocab_start_idx
    rows = emb_cpu.part_num_embeddings
    emb_cpu._weight_loader(emb_cpu.weight, full_w[start : start + rows])
    emb_cpu._weight_loader(emb_cpu.weight_scale_inv, full_s[start : start + rows])

    indices = torch.tensor(
        [[0, 17, 35, 50], [5, 20, 47, 63], [-1, 32, 48, 999]], dtype=torch.int32
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import Engram

    out_dev = torch.empty(
        (indices.shape[0], emb_dev.part_n_hash_cols, dim), dtype=torch.bfloat16
    )
    out_cpu = torch.empty_like(out_dev)
    emb_dev.lookup(indices, out_dev)
    emb_cpu.lookup(indices, out_cpu)
    assert torch.equal(out_dev, out_cpu)

    # Also exercise the host lookup: the host gathers raw FP8/scale records and
    # Engram.stage_rows dequantizes them, bitwise equal to the device lookup.
    from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import EngramHostLookup

    engram_stub = object.__new__(Engram)
    torch.nn.Module.__init__(engram_stub)
    engram_stub.embed_tokens = emb_cpu
    engram_stub.layer_hash_index = 0
    engram_stub.staged_rows = torch.full(
        (5, emb_cpu.part_n_hash_cols, dim), 3.0, dtype=torch.bfloat16
    )
    rows = EngramHostLookup([engram_stub], threads=1)(
        indices[:, None, :], torch.device("cpu")
    )
    engram_stub.stage_rows(rows[:, engram_stub.host_slot])
    assert torch.equal(engram_stub.staged_rows[: indices.shape[0]], out_cpu)
    # Rows past this step keep their previous contents.
    assert torch.all(engram_stub.staged_rows[indices.shape[0] :] == 3.0)


def test_dequant_fp8_rows_matches_exp2_lookup_bitwise():
    from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import dequant_fp8_rows

    # Every FP8 byte against every normal-range E8M0 byte (1..254).
    raw = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    raw = raw.repeat(254, 1)  # [254, 256]
    scale = torch.arange(1, 255, dtype=torch.int32).to(torch.uint8)[:, None]
    scale = scale.expand(254, 2).contiguous()  # block 128 -> 2 blocks
    got = dequant_fp8_rows(raw, scale, 128)
    ref = (
        raw.view(torch.float8_e4m3fn).float()
        * (scale.float() - 127).exp2().repeat_interleave(128, -1)
    ).to(torch.bfloat16)
    assert torch.equal(got.isnan(), ref.isnan())
    keep = ~ref.isnan()
    assert torch.equal(got[keep].view(torch.int16), ref[keep].view(torch.int16))


def test_reserve_rope_companion_caps_at_sparsecore_16gib_limit():
    from types import SimpleNamespace

    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
        SPARSECORE_MAX_POOL_BYTES,
        reserve_rope_companion,
    )

    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
    )
    small_avail = 10 << 30
    assert reserve_rope_companion(cfg, small_avail) == small_avail * 4 // 5
    # When Engram CPU offload frees ~23.6 GiB/rank (available ~ 41.1 GiB),
    # reserve_rope_companion must cap below 16 GiB (1 << 34) for SparseCore.
    offload_avail = 42 << 30
    capped = reserve_rope_companion(cfg, offload_avail)
    assert capped == SPARSECORE_MAX_POOL_BYTES
    assert capped < (16 << 30)


def test_slice_token_parallel_attention_inputs():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
        slice_token_parallel_attention_inputs,
    )

    tokens, tp_size = 2048, 8
    t_local = tokens // tp_size
    rng = np.random.default_rng(42)
    swa_rec = jnp.asarray(rng.integers(-1, 1000, (tokens, 128), dtype=np.int32))
    main_rec = jnp.asarray(rng.integers(-1, 5000, (tokens, 512), dtype=np.int32))
    # 4 ragged requests spanning rank boundaries (including padded trailing tokens).
    q_loc = jnp.asarray([0, 300, 512, 1400, 1984], dtype=jnp.int32)
    for r in range(tp_size):
        start = r * t_local
        end = (r + 1) * t_local
        swa_l, main_l, qloc_l = slice_token_parallel_attention_inputs(
            swa_rec,
            main_rec,
            q_loc,
            jnp.asarray([r], dtype=jnp.int32),
            tp_size,
            tokens,
        )
        np.testing.assert_array_equal(np.asarray(swa_l), np.asarray(swa_rec[start:end]))
        np.testing.assert_array_equal(
            np.asarray(main_l), np.asarray(main_rec[start:end])
        )
        expected_qloc = np.clip(np.asarray(q_loc) - start, 0, t_local)
        np.testing.assert_array_equal(np.asarray(qloc_l), expected_qloc)


def test_chunk_union_gather_indices_and_rope_repack_bitwise_exact():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
        compute_chunk_union_gather_indices,
        remap_union_kv_to_chunk,
        repack_chunk_rope_words,
        unpack_union_rope_words,
    )

    rng = np.random.default_rng(2026)
    num_tokens, topk, chunk_size = 256, 512, 128
    max_num_seqs, pages_per_seq, page_size = 4, 64, 16
    total_num_pages = 512
    union_cap = 1024

    q_loc = jnp.asarray([0, 60, 140, 210, 254], dtype=jnp.int32)
    pages = jnp.asarray(
        (np.arange(max_num_seqs * pages_per_seq, dtype=np.int32) * 13)
        % total_num_pages,
        dtype=jnp.int32,
    )
    records_np = np.full((num_tokens, topk), -1, dtype=np.int32)
    for c in range(num_tokens // chunk_size):
        pool = rng.choice(pages_per_seq * page_size, size=256, replace=False).astype(
            np.int32
        )
        for t in range(c * chunk_size, min(254, (c + 1) * chunk_size)):
            picked = np.sort(rng.choice(pool, size=topk - 8, replace=True))
            records_np[t, : topk - 8] = picked
    compacted_records = jnp.asarray(records_np, dtype=jnp.int32)

    union_phys, inv_idx, num_unique = compute_chunk_union_gather_indices(
        compacted_records,
        pages,
        q_loc,
        page_size=page_size,
        total_num_pages=total_num_pages,
        chunk_size=chunk_size,
        union_cap=union_cap,
    )
    assert union_phys.shape == (2, union_cap)
    assert inv_idx.shape == (2, chunk_size, topk)
    assert int(jnp.max(num_unique)) <= union_cap

    tokens_per_seq = q_loc[1:] - q_loc[:-1]
    seq_seg = jnp.repeat(
        jnp.arange(max_num_seqs, dtype=jnp.int32),
        tokens_per_seq,
        total_repeat_length=num_tokens,
    )
    page_table = pages.reshape(max_num_seqs, pages_per_seq)
    expected_phys = (
        page_table[seq_seg[:, None], jnp.clip(compacted_records // page_size, 0)]
        * page_size
        + (compacted_records % page_size)
    ).reshape(2, chunk_size, topk)
    got_phys = jnp.take_along_axis(union_phys[:, None, :], inv_idx, axis=2)
    valid_mask = compacted_records.reshape(2, chunk_size, topk) >= 0
    np.testing.assert_array_equal(
        np.asarray(got_phys[valid_mask]),
        np.asarray(expected_phys[valid_mask]),
    )

    # Verify unpack_union_rope_words + repack_chunk_rope_words / remap_union_kv_to_chunk
    # round-trip bit-for-bit across multi-period union_cap.
    u_nope = jnp.arange(union_cap * 128, dtype=jnp.int32).reshape(union_cap, 128)
    raw_rope_words = (
        jnp.arange(union_cap * 32, dtype=jnp.int32).reshape(union_cap, 32) + 7
    )
    u_rope_packed = repack_chunk_rope_words(
        raw_rope_words.reshape(union_cap // topk, topk, 32)
    ).reshape(union_cap // 4, 128)
    np.testing.assert_array_equal(
        np.asarray(unpack_union_rope_words(u_rope_packed, topk)),
        np.asarray(raw_rope_words),
    )
    idx0 = inv_idx[0]
    g_nope, g_rope = remap_union_kv_to_chunk(u_nope, u_rope_packed, idx0, topk)
    exp_nope = u_nope[idx0]
    exp_rope = repack_chunk_rope_words(raw_rope_words[idx0])
    np.testing.assert_array_equal(np.asarray(g_nope), np.asarray(exp_nope))
    np.testing.assert_array_equal(np.asarray(g_rope), np.asarray(exp_rope))


def test_chunk_union_gather_overflow_fallback_bitwise_exact():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
        compute_chunk_union_gather_indices,
        gather_sparse_kv_chunk_with_union_fallback,
        repack_chunk_rope_words,
    )

    rng = np.random.default_rng(2027)
    num_tokens, topk, chunk_size = 256, 512, 128
    max_num_seqs, pages_per_seq, page_size = 2, 128, 16
    total_num_pages = 256
    # Use union_cap = 512 so chunk 0 (<= 512 unique rows) takes _union_branch
    # and chunk 1 (> 512 unique rows, e.g. 1200 unique rows) takes _direct_branch!
    union_cap = 512

    q_loc = jnp.asarray([0, 128, 256], dtype=jnp.int32)
    pages = jnp.asarray(
        np.arange(max_num_seqs * pages_per_seq, dtype=np.int32) % total_num_pages,
        dtype=jnp.int32,
    )
    records_np = np.full((num_tokens, topk), -1, dtype=np.int32)
    pool0 = rng.choice(pages_per_seq * page_size, size=300, replace=False).astype(
        np.int32
    )
    for t in range(chunk_size):
        records_np[t, : topk - 4] = np.sort(
            rng.choice(pool0, size=topk - 4, replace=True)
        )
    pool1 = rng.choice(pages_per_seq * page_size, size=1400, replace=False).astype(
        np.int32
    )
    for t in range(chunk_size, num_tokens):
        records_np[t, : topk - 4] = np.sort(
            rng.choice(pool1, size=topk - 4, replace=False)
        )
    compacted_records = jnp.asarray(records_np, dtype=jnp.int32)

    union_phys, inv_idx, num_unique = compute_chunk_union_gather_indices(
        compacted_records,
        pages,
        q_loc,
        page_size=page_size,
        total_num_pages=total_num_pages,
        chunk_size=chunk_size,
        union_cap=union_cap,
    )
    assert int(num_unique[0]) <= union_cap
    assert int(num_unique[1]) > union_cap

    cache_kv_nope = jnp.arange(
        total_num_pages * page_size * 128, dtype=jnp.int32
    ).reshape(total_num_pages, page_size, 128)
    cache_kv_rope = (
        jnp.arange(total_num_pages * (page_size // 4) * 128, dtype=jnp.int32).reshape(
            total_num_pages, page_size // 4, 128
        )
        + 19
    )
    tokens_per_seq = q_loc[1:] - q_loc[:-1]
    seq_seg = jnp.repeat(
        jnp.arange(max_num_seqs, dtype=jnp.int32),
        tokens_per_seq,
        total_repeat_length=num_tokens,
    )
    flat_nope = cache_kv_nope.reshape(total_num_pages * page_size, 128)
    flat_rope_words = cache_kv_rope.reshape(total_num_pages * page_size, 32)
    page_table = pages.reshape(max_num_seqs, pages_per_seq)

    for c in range(2):
        start_pos = c * chunk_size
        end_pos = (c + 1) * chunk_size
        chunk_rec = compacted_records[start_pos:end_pos]
        chunk_seq = seq_seg[start_pos:end_pos]
        g_nope, g_rope = gather_sparse_kv_chunk_with_union_fallback(
            cache_kv_nope,
            cache_kv_rope,
            union_phys[c],
            inv_idx[c],
            num_unique[c],
            chunk_rec,
            chunk_seq,
            pages,
            max_num_seqs=max_num_seqs,
            start_pos=start_pos,
            batch_end=jnp.asarray(end_pos, dtype=jnp.int32),
            topk=topk,
        )
        seq_page_ids = chunk_rec // page_size
        token_offset = chunk_rec % page_size
        page_ids = page_table[chunk_seq[:, None], jnp.clip(seq_page_ids, 0)]
        is_pad = chunk_rec == -1
        flat_idx = start_pos * topk + jnp.arange(
            chunk_size * topk, dtype=jnp.int32
        ).reshape(chunk_size, topk)
        page_ids = jnp.where(is_pad, (flat_idx * 104729) % total_num_pages, page_ids)
        token_offset = jnp.where(
            is_pad, (flat_idx * 15485863) % page_size, token_offset
        )
        direct_idx = (page_ids * page_size + token_offset).reshape(-1)
        exp_nope = flat_nope[direct_idx].reshape(chunk_size, topk, 128)
        exp_rope_words = flat_rope_words[direct_idx].reshape(chunk_size, topk, 32)
        exp_rope = repack_chunk_rope_words(exp_rope_words)
        valid_mask = chunk_rec >= 0
        np.testing.assert_array_equal(
            np.asarray(g_nope[valid_mask]),
            np.asarray(exp_nope[valid_mask]),
        )
        if c == 1:
            # Overflow fallback branch matches direct gather across all entries
            np.testing.assert_array_equal(np.asarray(g_nope), np.asarray(exp_nope))
            np.testing.assert_array_equal(np.asarray(g_rope), np.asarray(exp_rope))
