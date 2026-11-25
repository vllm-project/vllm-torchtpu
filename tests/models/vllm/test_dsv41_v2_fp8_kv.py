"""CPU checks for the DeepSeek V4.1 FP8 main-KV record and RoPE companion."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import compact_records
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    PagedCacheRows,
    scatter_cache_values,
)

pytestmark = pytest.mark.usefixtures("cpu_engram_transfer")


def _latent(n, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((n, 512)) * rng.uniform(0.01, 30, (n, 1))
    return jnp.asarray(values, jnp.bfloat16)


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
        _pad_compacted_records,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select import (
        _compact_selected,
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
    assert emb_cpu.host_offload is True
    assert emb_cpu.weight.device.type == "cpu"
    assert emb_cpu.weight_scale_inv.device.type == "cpu"
    assert getattr(emb_cpu.weight, "dummy_weight_value", None) == 1.0
    assert getattr(emb_cpu.weight_scale_inv, "dummy_weight_value", None) == 127

    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", "0")
    emb_dev = ParallelEngramEmbedding(num_embeddings, dim, head_sizes, block_size=32)
    assert emb_dev.host_offload is False
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

    # Also exercise Engram._init_staging + Engram.prepare_embeddings: the host
    # gathers raw FP8/scale bytes (invalid rows get the sentinel scale;
    # invalid raw bytes may contain FP8 NaN 0x7F) and _ready_rows
    # unpacks the int32 staging and dequantizes it, bitwise equal to the
    # host-dequant lookup.
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.engram import _INVALID_SCALE

    engram_stub = object.__new__(Engram)
    torch.nn.Module.__init__(engram_stub)
    engram_stub.embed_tokens = emb_cpu
    Engram._init_staging(engram_stub, max_tokens=5, head_dim=dim)
    assert engram_stub._staged_raw is not None
    engram_stub._host_packed_buf.fill_(0x7F)
    engram_stub._host_scale.fill_(200)
    packed_bytes = indices.shape[0] * emb_cpu.part_n_hash_cols * (dim + dim // 32)
    untouched_tail = engram_stub._host_packed_buf[packed_bytes:].clone()
    engram_stub.prepare_embeddings(indices)
    got = engram_stub._ready_rows(indices.shape[0])
    assert torch.equal(got.view(torch.int16), out_cpu.view(torch.int16))
    assert engram_stub._active_raw.shape[0] == indices.shape[0]
    assert torch.all(engram_stub._host_scale[2, 1] == _INVALID_SCALE)
    assert engram_stub._host_scale.shape[0] == indices.shape[0]
    assert torch.equal(engram_stub._host_packed_buf[packed_bytes:], untouched_tail)


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


def test_dequant_fp8_rows_invalid_scale_is_positive_zero():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.engram import (
        _INVALID_SCALE,
        dequant_fp8_rows,
    )

    # Every FP8 byte (incl. NaN 0x7F/0xFF and -0) under the sentinel -> +0;
    # the other block keeps its exact value.
    raw = torch.arange(256, dtype=torch.int32).to(torch.uint8).repeat(1, 2)
    scale = torch.tensor([[_INVALID_SCALE, 127]], dtype=torch.uint8)
    got = dequant_fp8_rows(raw, scale, 256).view(torch.int16)
    assert torch.count_nonzero(got[:, :256]) == 0
    ref = raw[:, 256:].view(torch.float8_e4m3fn).to(torch.bfloat16).view(torch.int16)
    keep = ~raw[:, 256:].view(torch.float8_e4m3fn).float().isnan()
    assert torch.equal(got[:, 256:][keep], ref[keep])


def test_unpack_bytes_round_trips_host_int32_view():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.engram import _unpack_bytes

    data = torch.randint(0, 256, (4 * 37,), dtype=torch.int32).to(torch.uint8)
    words = data.view(torch.int32).clone()
    assert torch.equal(_unpack_bytes(words, data.numel()), data)
    assert torch.equal(_unpack_bytes(words[:10], 37), data[:37])
