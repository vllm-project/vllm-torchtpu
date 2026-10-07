"""Checks for the DeepSeek V4.1 Engram host offload.

The oracle is the device-table path: ``ParallelEngramEmbedding.lookup``
writing dequantized rows into ``Engram.staged_rows``. The offload path gathers
raw rows from the host table (``EngramHostLookup``) and dequantizes them with
``Engram.stage_rows``; the two must agree exactly.
"""

import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import (
    Engram,
    EngramHostLookup,
    ParallelEngramEmbedding,
    engram_offload_mode,
)

CPU = torch.device("cpu")

DIM = 64
BLOCK = 32
HEAD_SIZES = (11, 13, 17, 19, 23, 29)  # six hash columns


def _embedding(head_start, local_heads, seed):
    """A rank's shard, built without a TP group; host table filled at random."""
    embedding = ParallelEngramEmbedding.__new__(ParallelEngramEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.cpu_offload = True
    embedding.dim, embedding.block_size = DIM, BLOCK
    embedding.n_hash_cols = len(HEAD_SIZES)
    embedding.part_n_hash_cols = local_heads
    embedding.head_start = head_start
    head_end = min(head_start + local_heads, len(HEAD_SIZES))
    embedding.vocab_start_idx = sum(HEAD_SIZES[:head_start])
    embedding.vocab_end_idx = sum(HEAD_SIZES[:head_end])
    embedding.part_num_embeddings = embedding.vocab_end_idx - embedding.vocab_start_idx
    rng = np.random.default_rng(seed)
    table = np.empty((embedding.part_num_embeddings, DIM + DIM // BLOCK), np.uint8)
    values = rng.standard_normal((embedding.part_num_embeddings, DIM)) * 4
    table[:, :DIM] = (
        torch.from_numpy(values).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    )
    table[:, DIM:] = rng.integers(
        120, 135, (embedding.part_num_embeddings, DIM // BLOCK)
    )
    embedding.host_table = torch.from_numpy(table)
    weight = embedding.host_table[:, :DIM].view(torch.float8_e4m3fn)
    embedding.weight = torch.nn.Parameter(weight, requires_grad=False)
    embedding.weight_scale_inv = torch.nn.Parameter(
        embedding.host_table[:, DIM:], requires_grad=False
    )
    return embedding


def _engram(layer_hash_index, embedding, max_tokens):
    engram = Engram.__new__(Engram)
    torch.nn.Module.__init__(engram)
    engram.layer_hash_index = layer_hash_index
    engram.embed_tokens = embedding
    engram.staged_rows = torch.zeros(
        max_tokens, embedding.part_n_hash_cols, DIM, dtype=torch.bfloat16
    )
    return engram


def _hash_ids(tokens, layers, seed):
    """Hashes over the whole table plus out-of-range values on both sides."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(-5, sum(HEAD_SIZES) + 5, (tokens, layers, len(HEAD_SIZES)))
    return torch.from_numpy(ids.astype(np.int32))


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens", [1, 7, 2000])
@pytest.mark.parametrize(
    "head_start,local_heads",
    [(0, 3), (3, 3), (4, 4)],  # first rank, last rank, last rank with a padding head
)
def test_host_rows_match_device_lookup(tokens, head_start, local_heads):
    engrams = [
        _engram(index, _embedding(head_start, local_heads, seed=index), tokens)
        for index in range(2)
    ]
    hash_ids = _hash_ids(tokens, len(engrams), seed=tokens)

    expected = []
    for engram in engrams:
        out = torch.empty(tokens, local_heads, DIM, dtype=torch.bfloat16)
        engram.embed_tokens.lookup(hash_ids[:, engram.layer_hash_index], out)
        expected.append(out)

    rows = EngramHostLookup(engrams, threads=4)(hash_ids, CPU)
    assert rows.shape == (tokens, 2, local_heads, DIM + DIM // BLOCK)
    for engram, want in zip(engrams, expected):
        assert engram.host_slot == engram.layer_hash_index
        engram.stage_rows(rows[:, engram.host_slot])
        torch.testing.assert_close(engram.staged_rows[:tokens], want, rtol=0, atol=0)


@pytest.mark.cpu_test
def test_late_mode_offloads_only_later_layers():
    """Only the second Engram layer is host-resident, as in "late" mode: its
    rows land in slot 0 and come from its own hash column."""
    tokens = 33
    engram = _engram(1, _embedding(0, 3, seed=5), tokens)
    hash_ids = _hash_ids(tokens, 2, seed=6)
    want = torch.empty(tokens, 3, DIM, dtype=torch.bfloat16)
    engram.embed_tokens.lookup(hash_ids[:, 1], want)

    rows = EngramHostLookup([engram], threads=1)(hash_ids, CPU)
    assert engram.host_slot == 0
    assert rows.shape == (tokens, 1, 3, DIM + DIM // BLOCK)
    engram.stage_rows(rows[:, engram.host_slot])
    torch.testing.assert_close(engram.staged_rows[:tokens], want, rtol=0, atol=0)


@pytest.mark.cpu_test
def test_host_lookup_reuses_staging_across_sizes():
    engrams = [_engram(0, _embedding(0, 3, seed=0), 64)]
    lookup = EngramHostLookup(engrams, threads=1)
    big = _hash_ids(64, 1, seed=1)
    small = _hash_ids(5, 1, seed=2)
    first = lookup(big, CPU).clone()
    lookup(small, CPU)
    torch.testing.assert_close(lookup(big, CPU), first, rtol=0, atol=0)


@pytest.mark.cpu_test
def test_stage_rows_overwrites_only_the_step_rows():
    """A short step leaves the staging rows past it untouched."""
    engram = _engram(0, _embedding(0, 3, seed=7), 16)
    engram.staged_rows.fill_(5.0)
    rows = EngramHostLookup([engram], threads=1)(_hash_ids(4, 1, seed=8), CPU)
    engram.stage_rows(rows[:, engram.host_slot])
    assert torch.all(engram.staged_rows[4:] == 5.0)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "offload,alias,impl,sp,expected",
    [
        ("auto", "0", "v2", "0", "late"),
        ("auto", "0", "v2", "1", "off"),
        ("auto", "0", "v1", "0", "off"),
        ("all", "0", "v1", "0", "all"),
        ("off", "0", "v2", "0", "off"),
        ("late", "0", "v1", "0", "late"),
        ("auto", "1", "v2", "0", "all"),
        ("all", "1", "v1", "0", "all"),
    ],
)
def test_engram_offload_mode(monkeypatch, offload, alias, impl, sp, expected):
    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_OFFLOAD", offload)
    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", alias)
    monkeypatch.setenv("TPU_DEEPSEEK_V41_IMPL", impl)
    monkeypatch.setenv("TPU_DEEPSEEK_V41_SP", sp)
    assert engram_offload_mode() == expected


@pytest.mark.cpu_test
@pytest.mark.parametrize("offload", ["off", "late"])
def test_engram_offload_mode_rejects_conflicting_alias(monkeypatch, offload):
    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_OFFLOAD", offload)
    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", "1")
    with pytest.raises(ValueError):
        engram_offload_mode()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "mode,expected",
    [("off", [False, False]), ("late", [False, True]), ("all", [True, True])],
)
def test_create_embedding_places_tables_per_mode(monkeypatch, mode, expected):
    """ "late" keeps the earliest layer's table on the device."""
    monkeypatch.setattr(common_engram, "engram_offload_mode", lambda: mode)
    monkeypatch.setattr(common_engram, "register_row_slice", lambda *args: None)
    created = []

    class FakeEmbedding:
        def __init__(self, *args, cpu_offload=False):
            created.append(cpu_offload)
            self.vocab_start_idx = 0
            self.part_num_embeddings = 1

    monkeypatch.setattr(common_engram, "ParallelEngramEmbedding", FakeEmbedding)
    layout = type(
        "Layout",
        (),
        {
            "num_embeddings": [8, 8],
            "head_dim": 4,
            "primes": [[(2, 2)], [(2, 2)]],
            "layer_ids": [1, 14],
        },
    )()
    engram = Engram.__new__(Engram)
    for index in range(2):
        Engram._create_embedding(engram, layout, index)
    assert created == expected
