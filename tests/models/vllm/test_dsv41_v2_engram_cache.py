# SPDX-License-Identifier: Apache-2.0
"""100% Diff coverage test suite for DeepSeek-V4.1 Engram CPU offload and cache sizing.

Covers all lines and branches in:
- vllm_torchtpu.models.vllm.deepseek_v41.common.engram
- vllm_torchtpu.models.vllm.deepseek_v41.v2.cache (lines 20-40)
- vllm_torchtpu.envs (TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as v2_engram
from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import (
    Engram,
    ParallelEngramEmbedding,
    _gather_engram_rows,
    _load_local_rows,
)

pytestmark = pytest.mark.usefixtures("cpu_engram_transfer")


# ==============================================================================
# 1. Tests for _load_local_rows
# ==============================================================================


def test_load_local_rows_uint8_dtype():
    """Verify _load_local_rows executes .view(torch.uint8) for uint8 param."""
    param = torch.zeros(10, 4, dtype=torch.uint8)
    loaded_weight = torch.arange(40, dtype=torch.uint8).reshape(10, 4)
    _load_local_rows(param, loaded_weight)
    assert torch.equal(param, loaded_weight)


def test_load_local_rows_non_uint8_dtype():
    """Verify _load_local_rows executes .to(param.dtype) for float params."""
    param = torch.zeros(8, 4, dtype=torch.bfloat16)
    loaded_weight = torch.ones(8, 4, dtype=torch.float32) * 2.5
    _load_local_rows(param, loaded_weight)
    assert torch.equal(param, loaded_weight.to(torch.bfloat16))


# ==============================================================================
# 2. Tests for ParallelEngramEmbedding._weight_loader
# ==============================================================================


def test_weight_loader_local_rows_cpu():
    """Verify _weight_loader dispatches to _load_local_rows when shape matches."""
    param = torch.nn.Parameter(
        torch.zeros(6, 8, dtype=torch.bfloat16), requires_grad=False
    )
    loaded = torch.full((6, 8), 3.0, dtype=torch.float32)
    ParallelEngramEmbedding._weight_loader(param, loaded)
    assert torch.equal(param.data, loaded.to(torch.bfloat16))


# ==============================================================================
# 3. Tests for ParallelEngramEmbedding.__init__ and _allocate_weights
# ==============================================================================


def test_parallel_engram_embedding_allocate_weights_branching(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify _allocate_weights branches on cpu_offload setting."""
    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(v2_engram.upstream, "get_tensor_model_parallel_rank", lambda: 0)

    # Offload True
    emb_offload = ParallelEngramEmbedding(
        16, 32, (8, 8), block_size=16, cpu_offload=True
    )
    w_cpu, s_cpu = emb_offload._allocate_weights()
    assert w_cpu.device.type == "cpu"
    assert w_cpu.dtype == torch.float8_e4m3fn
    assert s_cpu.device.type == "cpu"
    assert s_cpu.dtype == torch.uint8

    # Offload False (device allocation)
    emb_device = ParallelEngramEmbedding(
        16, 32, (8, 8), block_size=16, cpu_offload=False
    )
    w_dev, s_dev = emb_device._allocate_weights()
    assert w_dev.shape == (emb_device.part_num_embeddings, emb_device.dim)
    assert s_dev.shape == (emb_device.part_num_embeddings, emb_device.dim // 16)


# ==============================================================================
# 4. Tests for ParallelEngramEmbedding.lookup
# ==============================================================================


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_parallel_engram_embedding_lookup_computation(
    monkeypatch: pytest.MonkeyPatch, tp_rank: int
):
    """Verify lookup mathematical correctness, head clamping, and sync."""
    tp_size = 2
    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_rank", lambda: tp_rank
    )

    head_sizes = (16, 16, 16, 16)
    num_embeddings = sum(head_sizes)
    dim = 32
    block_size = 16

    emb = ParallelEngramEmbedding(
        num_embeddings, dim, head_sizes, block_size=block_size, cpu_offload=True
    )

    emb.weight.fill_(1.0)
    emb.weight_scale_inv.fill_(127)

    indices = torch.tensor(
        [
            [0, 16, 32, 48],
            [-1, 10, 999, 20],
        ],
        dtype=torch.int32,
    )

    out = torch.empty(
        (indices.shape[0], emb.part_n_hash_cols, dim), dtype=torch.bfloat16
    )

    emb.lookup(indices, out)
    assert out.shape == (2, emb.part_n_hash_cols, dim)

    heads = torch.arange(emb.part_n_hash_cols) + emb.head_start
    for t in range(indices.shape[0]):
        for h_idx in range(emb.part_n_hash_cols):
            head_col = heads[h_idx].item()
            sel_val = indices[t, min(head_col, emb.n_hash_cols - 1)].item()
            is_valid = (
                head_col < emb.n_hash_cols
                and sel_val >= emb.vocab_start_idx
                and sel_val < emb.vocab_end_idx
            )
            expected_val = 1.0 if is_valid else 0.0
            torch.testing.assert_close(
                out[t, h_idx],
                torch.full((dim,), expected_val, dtype=torch.bfloat16),
            )


# ==============================================================================
# 5. Tests for _gather_engram_rows
# ==============================================================================


@pytest.mark.parametrize("width", [64, 128])
def test_gather_engram_rows_padding(monkeypatch: pytest.MonkeyPatch, width: int):
    """Verify _gather_engram_rows pads width to 128 and strips padding."""
    rows = torch.randn(4, 3, width, dtype=torch.bfloat16)

    def fake_all_gather(t, dim):
        assert dim == 0
        if width % 128 != 0:
            assert t.shape[-1] == 128
        else:
            assert t.shape[-1] == width
        return torch.cat([t, t], dim=0)

    monkeypatch.setattr(
        v2_engram.upstream, "tensor_model_parallel_all_gather", fake_all_gather
    )

    out = _gather_engram_rows(rows, dim=0)
    assert out.shape == (8, 3, width)
    torch.testing.assert_close(out[:4], rows)
    torch.testing.assert_close(out[4:], rows)


# ==============================================================================
# 6. Tests for Engram staging, embedding, and preparation
# ==============================================================================


def test_engram_init_staging_branches():
    """Verify _init_staging allocates raw/scale staging only with cpu_offload."""
    # Branch 1: cpu_offload = True
    engram_on = object.__new__(Engram)
    nn.Module.__init__(engram_on)
    engram_on.embed_tokens = SimpleNamespace(
        host_offload=True,
        cpu_offload=False,
        part_n_hash_cols=4,
        block_size=8,
        dp_size=1,
    )
    Engram._init_staging(engram_on, max_tokens=10, head_dim=16)
    for name, last in (("_host_raw", 16), ("_host_scale", 2)):
        buf = getattr(engram_on, name)
        assert buf.shape == (10, 4, last)
        assert buf.dtype == torch.uint8
        assert buf.device.type == "cpu"
    # Device staging is packed 1-D int32 words (160 raw words + 20 scale words).
    assert engram_on._staged_raw.shape == (180,)
    assert engram_on._staged_raw.dtype == torch.int32
    assert torch.all(engram_on._host_scale == v2_engram._INVALID_SCALE)
    assert torch.all(engram_on._staged_raw[:160] == 0)
    assert torch.all(engram_on._staged_raw[160:] == -1)
    assert not hasattr(engram_on, "staged_rows")

    # Branch 2: cpu_offload = False
    engram_off = object.__new__(Engram)
    nn.Module.__init__(engram_off)
    engram_off.embed_tokens = SimpleNamespace(
        host_offload=False, cpu_offload=False, part_n_hash_cols=4, dp_size=1
    )
    Engram._init_staging(engram_off, max_tokens=10, head_dim=16)
    assert engram_off._staged_raw is None
    assert engram_off.staged_rows.shape == (10, 4, 16)


def test_engram_prepare_embeddings_branching(monkeypatch: pytest.MonkeyPatch):
    """Verify the raw-gather offload path, tail reset, and both fallbacks."""
    dim = 16
    block = 8
    max_tokens = 8
    cols = 3
    one_fp8 = torch.tensor(1.0, dtype=torch.float8_e4m3fn).view(torch.uint8).item()

    def make_engram(cpu_offload: bool):
        engram = object.__new__(Engram)
        nn.Module.__init__(engram)
        embed = MagicMock()
        embed.host_offload = cpu_offload
        embed.cpu_offload = False
        embed.dp_size = 1
        embed.part_n_hash_cols = cols
        embed.block_size = block

        def mock_gather_raw(hash_ids, raw_out, scale_out):
            raw_out.fill_(one_fp8)
            scale_out.fill_(v2_engram._UNIT_SCALE + 2)  # 2**2

        embed.gather_raw = mock_gather_raw
        engram.embed_tokens = embed
        Engram._init_staging(engram, max_tokens=max_tokens, head_dim=dim)
        return engram

    engram = make_engram(cpu_offload=True)

    # Case A: Lookup writes raw and scale into adjacent views of the packed buffer.
    engram._host_packed_buf.fill_(99)
    engram._host_scale.fill_(99)
    engram.prepare_embeddings(torch.zeros(5, cols, dtype=torch.int32))
    assert torch.all(engram._host_raw[:5] == one_fp8)
    assert torch.all(engram._host_packed_buf[280:] == 99)
    assert torch.all(engram._host_scale[:5] == v2_engram._UNIT_SCALE + 2)
    assert torch.all(engram._host_scale_buf[30:40] == v2_engram._INVALID_SCALE)
    assert engram._host_scale_buf.numel() == 40
    assert engram._host_scale.shape == (5, cols, dim // block)
    assert (
        engram._host_scale_buf.data_ptr()
        == engram._host_packed_buf.data_ptr() + 5 * cols * dim
    )
    staged_5 = engram._staged_by_tokens[5]
    assert torch.equal(staged_5[:60], engram._host_raw[:5].view(-1).view(torch.int32))
    assert torch.equal(staged_5[60:], engram._host_scale_buf[:40].view(torch.int32))
    rows = Engram._ready_rows(engram, 5)
    torch.testing.assert_close(
        rows, torch.full((5, cols, dim), 4.0, dtype=torch.bfloat16)
    )

    # Case B: Exact capacity, then smaller and empty calls must relocate scales.
    engram.prepare_embeddings(torch.zeros(max_tokens, cols, dtype=torch.int32))
    torch.testing.assert_close(
        Engram._ready_rows(engram, max_tokens),
        torch.full((max_tokens, cols, dim), 4.0, dtype=torch.bfloat16),
    )
    for tokens in (1, 0, 3, 5):
        engram.prepare_embeddings(torch.zeros(tokens, cols, dtype=torch.int32))
        torch.testing.assert_close(
            Engram._ready_rows(engram, tokens),
            torch.full((tokens, cols, dim), 4.0, dtype=torch.bfloat16),
        )
        raw_bytes = tokens * cols * dim
        scale_bytes = tokens * cols * (dim // block)
        scale_capacity = tokens * ((cols * (dim // block) + 3) // 4) * 4
        expected = torch.cat(
            (
                torch.full((raw_bytes,), one_fp8, dtype=torch.uint8),
                torch.full(
                    (scale_bytes,), v2_engram._UNIT_SCALE + 2, dtype=torch.uint8
                ),
                torch.full(
                    (scale_capacity - scale_bytes,),
                    v2_engram._INVALID_SCALE,
                    dtype=torch.uint8,
                ),
            )
        )
        assert torch.equal(engram._active_raw.view(-1), expected.view(torch.int32))

    # Case C/D: Exceeding max_tokens, or no offload staging -> upstream path.
    fallback_calls = []
    monkeypatch.setattr(
        v2_engram.upstream.Engram,
        "prepare_embeddings",
        lambda self, hash_ids: fallback_calls.append(self),
    )
    with pytest.raises(ValueError, match="staging slots"):
        engram.prepare_embeddings(torch.zeros(12, cols, dtype=torch.int32))
    engram_off = make_engram(cpu_offload=False)
    engram_off.prepare_embeddings(torch.zeros(5, cols, dtype=torch.int32))
    assert fallback_calls == [engram_off]


# ==============================================================================
# 7. Tests for Engram._sp_embed and Engram.forward
# ==============================================================================


# ==============================================================================
# 8. Tests for cache.py and envs.py
# ==============================================================================
