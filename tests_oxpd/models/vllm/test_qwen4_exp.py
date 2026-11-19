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
"""Tests for the TPU Qwen4Exp (Qwen3.8 Flash Next) text model."""

from __future__ import annotations

import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import (
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor import parameter as vllm_parameter
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers import vocab_parallel_embedding as vpe
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
)
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpForCausalLM
from vllm_torchtpu.models.vllm.qwen4_exp.ple import (
    PLEInputs,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
    Qwen4ExpPLETable,
    head_vocab_sizes,
    layer_multipliers,
    ngram_ids,
    token_requests,
)

HIDDEN = 16
HC = 4
VOCAB = 32
LAYER_TYPES = ["linear_attention", "full_attention"]


@pytest.fixture(autouse=True)
def _cpu_model_tests(monkeypatch):
    # Same guard as the Kimi model tests: keep Dynamo from querying a TPU
    # device while these CPU-only tests run.
    tpu = getattr(torch, "tpu", None)
    if tpu is not None:
        monkeypatch.setattr(tpu, "is_available", lambda: False)
        monkeypatch.setattr(tpu, "current_device", lambda: 0)
        monkeypatch.setattr(tpu, "device_count", lambda: 0)
        monkeypatch.setattr(tpu, "manual_seed_all", lambda seed: None)
    if hasattr(torch, "accelerator"):
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
        monkeypatch.setattr(torch.accelerator, "device_count", lambda: 0)

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _text_config(**overrides) -> Qwen4ExpTextConfig:
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        hc_count=HC,
        hc_lowrank=8,
        num_hidden_layers=len(LAYER_TYPES),
        layer_types=LAYER_TYPES,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=3,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    kwargs.update(overrides)
    return Qwen4ExpTextConfig(**kwargs)


# ---------------------------------------------------------------------------
# Registration and config checks
# ---------------------------------------------------------------------------


_QSA = dict(
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=8,
    indexer_budget=2048,
    indexer_compress_ratio=4,
)


# ---------------------------------------------------------------------------
# Decoder wiring with stand-in sublayers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


_LM = "model.language_model"


def test_fp8_checkpoint_quantizes_routed_experts_and_ple_table() -> None:
    not_converted = [
        "lm_head",
        f"{_LM}.embed_tokens",
        *(
            f"{_LM}.layers.0.linear_attn.{name}"
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
        ),
        f"{_LM}.layers.0.linear_attn.out_proj",
        *(
            f"{_LM}.layers.1.self_attn.{name}"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ),
        *(
            f"{_LM}.layers.{idx}.mlp.shared_expert.{name}"
            for idx in (0, 1)
            for name in ("gate_proj", "up_proj", "down_proj")
        ),
        *(
            f"{_LM}.layers.1.ple.{name}"
            for name in ("conv1d", "key_proj", "value_proj")
        ),
    ]
    quant_config = VllmFp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": not_converted,
        }
    )
    configure_quant_config(quant_config, Qwen4ExpForCausalLM)

    def skipped(prefix: str) -> bool:
        return is_layer_skipped(
            prefix,
            quant_config.ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
        )

    for prefix in (
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.0.linear_attn.in_proj_ba",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.k_proj",
        "model.layers.1.self_attn.v_proj",
        "model.layers.1.self_attn.o_proj",
        "model.layers.0.mlp.shared_expert.gate_up_proj",
        "model.layers.1.mlp.shared_expert.down_proj",
        "model.layers.1.ple.key_proj",
        "model.layers.1.ple.value_proj",
    ):
        assert skipped(prefix), prefix
    assert not skipped("model.layers.0.mlp.experts")
    table_method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
        quant_config, "model.layers.1.ple.ple_embedding.ngram_embedding"
    )
    assert isinstance(table_method, Qwen4ExpPLEFp8EmbeddingMethod)


# ---------------------------------------------------------------------------
# N-gram PLE
# ---------------------------------------------------------------------------
_PLE = dict(
    ple_layer_ids=[2],
    ngram_size=3,
    heads_per_ngram=2,
    # The 4 heads hash into 11, 13, 17 and 19 rows: 60 rows, padded to 64.
    ngram_vocab_size_base=10,
    make_ngram_vocab_size_divisible_by=16,
    # The checkpoint splits the table into shards of 22, 22 and 20 rows.
    split_ngram_parts=3,
    ple_embed_dim=16,
    ple_conv_kernel_size=4,
    rms_norm_eps=1e-6,
    eos_token_id=3,
)
_TABLE_ROWS = 64
_HEAD_DIM = 4

# Step i runs chunk i of every request: one prefill step, chunks that start
# on and right after an EOS, and token-by-token decode.
_CHUNKS = {
    "one_step": [[12], [7]],
    "chunked": [[5, 1, 6], [2, 4, 1]],
    "decode": [[4] + [1] * 8, [1] * 7],
}


def _set_tp(monkeypatch, rank: int, size: int) -> None:
    for module in (vllm_linear, vllm_parameter, vpe):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: size
        )


def _sequences(vocab_size: int, eos: int) -> list[torch.Tensor]:
    """Two requests: one with back-to-back EOS mid-sequence, one starting
    with EOS."""
    generator = torch.Generator().manual_seed(0)
    first = torch.randint(vocab_size, (12,), generator=generator)
    first[4:6] = eos
    second = torch.randint(vocab_size, (7,), generator=generator)
    second[0] = eos
    return [first, second]


def _steps(chunks: list[list[int]]) -> list[list[tuple[int, int, int]]]:
    """(request, start, end) of chunk i of every request, for each step i."""
    return [
        [
            (request, sum(lens[:step]), sum(lens[: step + 1]))
            for request, lens in enumerate(chunks)
            if step < len(lens)
        ]
        for step in range(max(map(len, chunks)))
    ]


def _batch(
    sequences: list[torch.Tensor], step: list[tuple[int, int, int]], eos: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packs a step as the TPU runner does: padding tokens go to an empty
    trailing request, and each chunk's context is the two tokens before it."""
    chunks = [sequences[request][start:end] for request, start, end in step]
    lens = torch.tensor([len(chunk) for chunk in chunks])
    num_tokens = int(lens.sum())
    padding = torch.zeros(num_tokens // 8 * 8 + 8 - num_tokens, dtype=torch.long)
    query_start_loc = torch.tensor([0, *lens.cumsum(0).tolist(), num_tokens])
    context = []
    for request, start, _ in step:
        before = sequences[request][max(0, start - 2) : start].tolist()
        context.append([eos] * (2 - len(before)) + before)
    context.append([eos, eos])
    return torch.cat([*chunks, padding]), query_start_loc, torch.tensor(context)


# Reference math adapted from HF transformers' Qwen4Exp modeling code
# (Qwen4ExpTextNGramEmbedding and Qwen4ExpTextPLELayer without a cache), which
# the transformers version CI installs does not include.
def _ref_shift_right_ignore_eos(
    tokens: torch.Tensor, shift: int, eos: int
) -> torch.Tensor:
    positions = torch.arange(tokens.numel())
    eos_positions = torch.where(tokens == eos, positions, -1)
    previous_eos = torch.cat(
        [eos_positions.new_full((1,), -1), eos_positions.cummax(0).values[:-1]]
    )
    source = positions - shift
    valid = (positions - previous_eos - 1 >= shift) & (source >= 0)
    return torch.where(valid, tokens[source.clamp(min=0)], eos)


def _ref_ngram_ids(
    tokens: torch.Tensor,
    multipliers: torch.Tensor,
    sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    ngram_size = multipliers.numel()
    history = torch.cat([torch.full((ngram_size - 1,), eos), tokens])
    shifted = [
        _ref_shift_right_ignore_eos(history, shift, eos) for shift in range(ngram_size)
    ]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        heads = slice((ngram - 2) * heads_per_ngram, (ngram - 1) * heads_per_ngram)
        mixed = shifted[0] * multipliers[0]
        for index in range(1, ngram):
            mixed = mixed ^ (shifted[index] * multipliers[index])
        blocks.append(mixed.unsqueeze(-1) % sizes[heads] + offsets[heads])
    return torch.cat(blocks, dim=-1)[ngram_size - 1 :]


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    groups = x.unflatten(-1, (-1, HIDDEN))
    normed = groups * torch.rsqrt(groups.pow(2).mean(-1, keepdim=True) + 1e-6)
    return normed.flatten(-2) * (1.0 + weight)


def _ref_ple(
    ple: Qwen4ExpPLELayer, hidden_states: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    embedding = ple.ple_embedding
    ids = _ref_ngram_ids(
        tokens,
        embedding.layer_multipliers,
        embedding.ngram_heads_vocab_sizes,
        embedding.ngram_heads_offsets,
        _PLE["eos_token_id"],
        _PLE["heads_per_ngram"],
    )
    embeddings = embedding.ngram_embedding.weight[ids].flatten(-2)
    key = _ref_rms_norm(embeddings @ ple.key_proj.weight.T, ple.norm_key.weight)
    query = _ref_rms_norm(hidden_states, ple.norm_query.weight)
    gate = (key * query).unflatten(-1, (HC, HIDDEN)).sum(-1, keepdim=True)
    gate = gate / math.sqrt(HIDDEN)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    value = embeddings @ ple.value_proj.weight.T
    gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
    dilation = _PLE["ngram_size"]
    history = (_PLE["ple_conv_kernel_size"] - 1) * dilation
    conv_input = F.pad(_ref_rms_norm(gated, ple.norm_conv.weight).T, (history, 0))
    conv = F.conv1d(
        conv_input, ple.conv1d.weight, groups=HC * HIDDEN, dilation=dilation
    )
    return gated + F.silu(conv.T)


def _randomize_ple(ple: nn.Module) -> None:
    with torch.no_grad():
        for param in ple.parameters():
            param.normal_(0.0, 0.5)


@pytest.mark.parametrize(
    "ple_layer_index,multipliers,sizes",
    [
        # The buffers stored in the released Qwen3.8-Flash-Next checkpoint.
        (
            0,
            [23703573157769, 20109073645365, 8052911324071],
            [
                *(20000003, 20000023, 20000033, 20000047, 20000059, 20000063),
                *(20000069, 20000077, 20000081, 20000093, 20000107, 20000147),
                *(20000153, 20000159, 20000161, 20000171),
            ],
        ),
        # Upstream's _make_layer_multipliers and _make_vocab_layout for a
        # second PLE layer.
        (
            1,
            [3352040966061, 19752442147475, 10742103100073],
            [
                *(20000213, 20000221, 20000243, 20000269, 20000287, 20000297),
                *(20000303, 20000311, 20000327, 20000329, 20000339, 20000353),
                *(20000359, 20000377, 20000389, 20000429),
            ],
        ),
    ],
    ids=["released_checkpoint", "second_ple_layer"],
)
def test_hash_buffers_match_checkpoint_and_upstream(
    ple_layer_index: int, multipliers: list[int], sizes: list[int]
) -> None:
    """Config-derived multipliers and head sizes at the released checkpoint's
    vocab (248320) and ngram_vocab_size_base (20M)."""
    assert layer_multipliers(248320, 3, ple_layer_index, 1234) == multipliers
    assert head_vocab_sizes(20_000_000, 16, ple_layer_index) == sizes


@pytest.mark.parametrize(
    "query_start_loc,num_tokens,expected_request,expected_chunk_pos",
    [
        # Requests 1 and 3 are empty; the two padding tokens go to request 3.
        ([0, 2, 2, 5, 5], 7, [0, 0, 2, 2, 2, 3, 3], [0, 1, 0, 1, 2, 0, 1]),
        # A request start past the token bucket is dropped.
        ([0, 2, 6, 8], 4, [0, 0, 1, 1], [0, 1, 0, 1]),
    ],
    ids=["empty_requests_and_padding", "start_past_bucket"],
)
def test_token_requests(
    query_start_loc: list[int],
    num_tokens: int,
    expected_request: list[int],
    expected_chunk_pos: list[int],
) -> None:
    request, chunk_pos = token_requests(torch.tensor(query_start_loc), num_tokens)
    assert request.tolist() == expected_request
    assert chunk_pos.tolist() == expected_chunk_pos


@pytest.mark.parametrize("chunks", list(_CHUNKS.values()), ids=list(_CHUNKS))
def test_ngram_ids_match_reference_for_any_chunking(chunks) -> None:
    """Hashing each chunk with the two tokens before it gives the same ids as
    hashing the whole sequence, at the released checkpoint's sizes."""
    vocab_size, eos, heads_per_ngram = 248320, 248044, 8
    multipliers = torch.tensor(layer_multipliers(vocab_size, 3, 0, 1234))
    sizes = torch.tensor(head_vocab_sizes(20_000_000, 2 * heads_per_ngram, 0))
    offsets = sizes.cumsum(0) - sizes
    sequences = _sequences(vocab_size, eos)
    got: list[list[torch.Tensor]] = [[] for _ in sequences]
    for step in _steps(chunks):
        input_ids, query_start_loc, context = _batch(sequences, step, eos)
        request, chunk_pos = token_requests(query_start_loc, input_ids.numel())
        ids = ngram_ids(
            input_ids,
            context,
            request,
            chunk_pos,
            multipliers,
            sizes,
            offsets,
            eos_token_id=eos,
            heads_per_ngram=heads_per_ngram,
        )
        for index, (request, _, _) in enumerate(step):
            start, end = query_start_loc[index], query_start_loc[index + 1]
            got[request].append(ids[start:end])
    for request, tokens in enumerate(sequences):
        expected = _ref_ngram_ids(
            tokens, multipliers, sizes, offsets, eos, heads_per_ngram
        )
        assert torch.equal(torch.cat(got[request]), expected)


@pytest.mark.parametrize("chunks", list(_CHUNKS.values()), ids=list(_CHUNKS))
def test_ple_layer_matches_reference_for_any_chunking(monkeypatch, chunks) -> None:
    """Padded, chunked batches that carry each request's conv window between
    steps match the reference run over each whole sequence."""
    _set_tp(monkeypatch, 0, 1)
    torch.manual_seed(0)
    ple = Qwen4ExpPLELayer(_text_config(**_PLE), 0, None, prefix="model.layers.1.ple")
    _randomize_ple(ple)
    eos = _PLE["eos_token_id"]
    sequences = _sequences(VOCAB, eos)
    hidden_states = [torch.randn(len(tokens), HC * HIDDEN) for tokens in sequences]
    # The caller owns the conv state. As in the TPU runner, slot 0 serves the
    # padding request, and a request without history gets a zero window
    # instead of its stale slot.
    conv_state = torch.randn(len(sequences) + 1, ple.conv_state_len, HC * HIDDEN)
    got: list[list[torch.Tensor]] = [[] for _ in sequences]
    with torch.no_grad():
        for step in _steps(chunks):
            input_ids, query_start_loc, context = _batch(sequences, step, eos)
            hidden = torch.cat([hidden_states[r][start:end] for r, start, end in step])
            hidden = F.pad(hidden, (0, 0, 0, input_ids.numel() - hidden.shape[0]))
            slots = torch.tensor([r + 1 for r, _, _ in step] + [0])
            has_history = torch.tensor([s > 0 for _, s, _ in step] + [False])
            windows = torch.where(has_history.view(-1, 1, 1), conv_state[slots], 0)
            output, new_windows = ple(
                hidden, PLEInputs(input_ids, query_start_loc, context), windows
            )
            conv_state[slots] = new_windows
            for index, (request, _, _) in enumerate(step):
                start, end = query_start_loc[index], query_start_loc[index + 1]
                got[request].append(output[start:end])
        for request, tokens in enumerate(sequences):
            torch.testing.assert_close(
                torch.cat(got[request]),
                _ref_ple(ple, hidden_states[request], tokens),
                rtol=1e-5,
                atol=1e-5,
            )


@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "unquantized"])
def test_ple_table_shards_load_across_tp_ranks(monkeypatch, fp8: bool) -> None:
    """Each of two TP ranks keeps its rows of every checkpoint shard, and the
    ranks' lookups sum to the dequantized table."""
    quant_config = (
        VllmFp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            }
        )
        if fp8
        else None
    )
    table = torch.randn(_TABLE_ROWS, _HEAD_DIM)
    if fp8:
        table = table.to(torch.float8_e4m3fn)
    # The middle shard straddles the two ranks' row ranges.
    weights = {
        f"ngram_embedding.shard_{index}.weight": rows
        for index, rows in enumerate(table.split(22))
    }
    expected_loaded = {"ngram_embedding.weight"}
    if fp8:
        weights["ngram_embedding.weight_scale"] = torch.tensor(
            [0.25], dtype=torch.bfloat16
        )
        expected_loaded.add("ngram_embedding.weight_scale")
    ids = torch.randperm(_TABLE_ROWS).view(-1, 4)
    rows = torch.zeros(*ids.shape, _HEAD_DIM)
    for rank in range(2):
        _set_tp(monkeypatch, rank, 2)
        embedding = Qwen4ExpNGramEmbedding(
            _text_config(**_PLE),
            _PLE["ple_embed_dim"],
            0,
            quant_config,
            prefix="model.layers.1.ple.ple_embedding",
        )
        assert embedding.load_weights(iter(weights.items())) == expected_loaded
        rank_table = embedding.ngram_embedding
        assert rank_table.weight.shape == (_TABLE_ROWS // 2, _HEAD_DIM)
        assert rank_table.weight.dtype == table.dtype
        # Raises if an FP8 table's scale was never loaded.
        rank_table.embedding_method.process_weights_after_loading(rank_table)
        rows += rank_table.local_rows(ids)
    expected = table.float() * 0.25 if fp8 else table
    torch.testing.assert_close(rows, expected[ids])


@pytest.mark.parametrize(
    "weights,match",
    [
        (
            {"ngram_embedding.shard_0.weight": torch.zeros(21, _HEAD_DIM)},
            "shard 0 has shape",
        ),
        (
            {"ngram_embedding.shard_3.weight": torch.zeros(22, _HEAD_DIM)},
            "out of range",
        ),
        (
            {"ngram_embedding.shard_x.weight": torch.zeros(22, _HEAD_DIM)},
            "shard_<index>",
        ),
        ({"ngram_embedding.weight_scale": torch.ones(1)}, "Unexpected PLE"),
        ({"layer_multipliers": torch.tensor([1 << 62, 3, 5])}, "overflows int64"),
    ],
    ids=["shard_shape", "shard_index", "shard_name", "scale_without_fp8", "overflow"],
)
def test_ngram_embedding_rejects_malformed_checkpoint_tensors(
    monkeypatch, weights: dict[str, torch.Tensor], match: str
) -> None:
    _set_tp(monkeypatch, 0, 1)
    embedding = Qwen4ExpNGramEmbedding(
        _text_config(**_PLE),
        _PLE["ple_embed_dim"],
        0,
        None,
        prefix="model.layers.1.ple.ple_embedding",
    )
    with pytest.raises(ValueError, match=match):
        embedding.load_weights(iter(weights.items()))


@pytest.mark.parametrize(
    "num_ids,context_delta,window_delta,match",
    [
        (2, 0, 0, "input ids"),
        (3, -1, 0, "ngram_context"),
        (3, 0, -1, "conv windows"),
    ],
    ids=["input_ids", "ngram_context", "conv_windows"],
)
def test_ple_layer_rejects_inconsistent_inputs(
    monkeypatch, num_ids: int, context_delta: int, window_delta: int, match: str
) -> None:
    _set_tp(monkeypatch, 0, 1)
    ple = Qwen4ExpPLELayer(_text_config(**_PLE), 0, None, prefix="model.layers.1.ple")
    inputs = PLEInputs(
        input_ids=torch.arange(num_ids) + 5,
        query_start_loc=torch.tensor([0, 3]),
        ngram_context=torch.full(
            (1, _PLE["ngram_size"] - 1 + context_delta), _PLE["eos_token_id"]
        ),
    )
    windows = torch.zeros(1, ple.conv_state_len + window_delta, HC * HIDDEN)
    with torch.no_grad(), pytest.raises(ValueError, match=match):
        ple(torch.zeros(3, HC * HIDDEN), inputs, windows)


def test_ple_layer_rejects_kernel_size_one(monkeypatch) -> None:
    _set_tp(monkeypatch, 0, 1)
    config = _text_config(**{**_PLE, "ple_conv_kernel_size": 1})
    with pytest.raises(NotImplementedError, match="ple_conv_kernel_size"):
        Qwen4ExpPLELayer(config, 0, None, prefix="model.layers.1.ple")


def test_ple_storage_factory_allocates_once_and_reuses_shard_loader(monkeypatch):
    _set_tp(monkeypatch, 0, 1)
    allocations = []

    class RecordingTable(Qwen4ExpPLETable):
        def allocate_embedding_weight(self, rows, width, dtype):
            allocations.append((rows, width, dtype))
            return torch.zeros(rows, width, dtype=dtype)

    ple = Qwen4ExpPLELayer(
        _text_config(**_PLE),
        0,
        None,
        prefix="model.layers.1.ple",
        table_factory=RecordingTable,
    )
    table = ple.ple_embedding.ngram_embedding
    assert isinstance(table, RecordingTable)
    assert allocations == [(_TABLE_ROWS, _HEAD_DIM, torch.float32)]
    rows = torch.arange(_TABLE_ROWS * _HEAD_DIM).reshape(_TABLE_ROWS, _HEAD_DIM).float()
    ple.ple_embedding.load_weights(
        (f"ngram_embedding.shard_{i}.weight", shard)
        for i, shard in enumerate(rows.split(22))
    )
    ids = torch.tensor([[0, 21, 22, 63]])
    assert torch.equal(table(ids), rows[ids])


def test_ple_explicit_group_controls_sharding_and_reduction(monkeypatch):
    _set_tp(monkeypatch, 0, 1)

    class Group:
        rank_in_group = 1
        world_size = 2

        def all_reduce(self, rows):
            self.seen = rows.clone()
            return rows + 1

    group = Group()
    table = Qwen4ExpPLETable(
        _TABLE_ROWS,
        _HEAD_DIM,
        embedding_method=Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, "ple"),
        padding_size=16,
        prefix="ple",
        parallel_group=group,
    )
    rows = torch.arange(_TABLE_ROWS * _HEAD_DIM).reshape(_TABLE_ROWS, _HEAD_DIM).float()
    table.weight_loader(table.weight, rows, checkpoint_start=0)
    assert table.weight.shape == (32, _HEAD_DIM)
    ids = torch.tensor([[0, 31, 32, 63]])
    expected = rows[ids].masked_fill((ids < 32).unsqueeze(-1), 0)
    assert torch.equal(table(ids), expected + 1)
    assert torch.equal(group.seen, expected)


@pytest.mark.parametrize("storage", ["resident", "host", "shared"])
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_ple_storage_reuses_common_checkpoint_loader(
    monkeypatch, tmp_path, storage, fp8
):
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_torchtpu.models.vllm.qwen4_exp.ple_storage import (
        HostPLETable,
        ResidentPLETable,
        SharedPLEOptions,
    )

    _set_tp(monkeypatch, 0, 1)
    if storage == "resident":
        factory = ResidentPLETable
    else:
        options = (
            SharedPLEOptions(tmp_path, "fixture", {"checkpoint": "small"}, 1)
            if storage == "shared"
            else None
        )
        factory = partial(HostPLETable, shared=options)
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    config = _text_config(**_PLE, ple_embedding_dtype="float8_e4m3fn" if fp8 else None)
    with set_default_torch_dtype(torch.bfloat16):
        embedding = Qwen4ExpNGramEmbedding(
            config, _PLE["ple_embed_dim"], 0, None, "ple", table_factory=factory
        )
    table = embedding.ngram_embedding
    checkpoint = (
        (torch.arange(_TABLE_ROWS * _HEAD_DIM).float() % 8)
        .reshape(_TABLE_ROWS, _HEAD_DIM)
        .to(dtype)
    )
    try:
        assert table.weight.is_meta == (storage == "shared")
        weights = [
            (f"ngram_embedding.shard_{i}.weight", shard)
            for i, shard in enumerate(checkpoint.split(22))
        ]
        if fp8:
            weights.append(("ngram_embedding.weight_scale", torch.tensor([0.25])))
        embedding.load_weights(weights)
        visited = []
        weight = table.weight

        def visit(tensor):
            visited.append(tensor)
            return tensor

        table._apply(visit)
        assert table.weight is weight
        assert all(tensor is not weight for tensor in visited)
        table.finalize_weights()
        table.finalize_weights()
        if storage == "resident":
            assert table.weight.numel() == 0
            assert torch.equal(
                table.table.reshape(_TABLE_ROWS, _HEAD_DIM).view(dtype).float(),
                checkpoint.float(),
            )
        else:
            ids = torch.tensor([[0, 22, 63, -1], [64, 31, 32, 1]])
            raw = table.lookup_host(ids).reshape(*ids.shape, _HEAD_DIM)
            expected = checkpoint[ids.clamp(0, 63)].float()
            expected[(ids < 0) | (ids >= 64)] = 0
            assert torch.equal(raw.view(dtype).float(), expected)
        with pytest.raises(ValueError, match="finalized"):
            embedding.load_weights(weights[:1])
    finally:
        if isinstance(table, HostPLETable):
            table.close()
    assert not list(tmp_path.iterdir())


def test_ple_storage_rejects_missing_and_duplicate_rows(monkeypatch):
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_torchtpu.models.vllm.qwen4_exp.ple_storage import ResidentPLETable

    _set_tp(monkeypatch, 0, 1)
    with set_default_torch_dtype(torch.bfloat16):
        embedding = Qwen4ExpNGramEmbedding(
            _text_config(**_PLE),
            _PLE["ple_embed_dim"],
            0,
            None,
            "ple",
            table_factory=ResidentPLETable,
        )
    shard = ("ngram_embedding.shard_0.weight", torch.ones(22, 4, dtype=torch.bfloat16))
    embedding.load_weights([shard])
    with pytest.raises(ValueError, match="Duplicate"):
        embedding.load_weights([shard])
    with pytest.raises(ValueError, match="Missing"):
        embedding.ngram_embedding.finalize_weights()


@pytest.mark.parametrize("storage", ["resident", "host", "shared"])
def test_ple_native_adapter_reuses_common_math_and_checkpoint(
    monkeypatch, tmp_path, storage
):
    """CPU adapter contract; only device lookup/conv bridges are dense doubles."""
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_torchtpu.models.vllm.qwen4_exp import ple_offload, ple_storage, pooled_ple
    from vllm_torchtpu.models.vllm.qwen4_exp.parallel import PLEParallel
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import dilated_short_conv

    _set_tp(monkeypatch, 0, 1)

    class Group:
        rank_in_group = 0
        world_size = 1

        def all_reduce(self, rows):
            return rows

    group = Group()
    parallel = PLEParallel(group, group, embedding_across_dp=False)

    def raw_lookup(rows, width, chunk_tokens=8):
        def run(table, ids, vocab_start):
            result = table.reshape(rows, width)[ids.clamp(0, rows - 1)].clone()
            result[(ids < 0) | (ids >= rows)] = 0
            return result.flatten(-2)

        return run

    monkeypatch.setattr(ple_storage, "build_raw_lookup_op", raw_lookup)
    monkeypatch.setattr(
        ple_storage,
        "build_decode_op",
        lambda: lambda raw, scale: raw.view(torch.bfloat16),
    )

    def conv_bridge(dilation):
        def run(x, weight, history, starts, slots, initial):
            request, chunk = token_requests(starts, x.shape[0])
            windows = torch.where(
                initial[:, None, None], history[slots.clamp_min(0)], 0
            )
            output, updated = dilated_short_conv(
                x, weight, dilation, windows, starts, request, chunk
            )
            for i, slot in enumerate(slots.tolist()):
                if slot > 0 and starts[i + 1] > starts[i]:
                    history[slot] = updated[i]
            return F.silu(output)

        return run

    monkeypatch.setattr(pooled_ple, "build_short_conv_op", conv_bridge)
    shared = (
        ple_storage.SharedPLEOptions(tmp_path, "layer", {"checkpoint": "fixture"}, 1)
        if storage == "shared"
        else None
    )
    with set_default_torch_dtype(torch.bfloat16):
        reference = Qwen4ExpPLELayer(_text_config(**_PLE), 0, None, "ple")
        _randomize_ple(reference)
        adapted = pooled_ple.Qwen4ExpPooledPLE(
            _text_config(**_PLE),
            0,
            None,
            "ple",
            parallel=parallel,
            table_group=group,
            cpu_offload=storage != "resident",
            shared=shared,
        )
    weights = dict(reference.state_dict())
    full_table = weights.pop("ple_embedding.ngram_embedding.weight")
    for i, rows in enumerate(full_table.split(22)):
        weights[f"ple_embedding.ngram_embedding.shard_{i}.weight"] = rows
    adapted.load_checkpoint(iter(weights.items()))
    adapted.finalize_weights()
    monkeypatch.setattr(
        ple_offload,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": adapted}),
    )
    history = torch.zeros(
        4, reference.conv_state_len, HC * HIDDEN, dtype=torch.bfloat16
    )
    history[1].fill_(4)  # A reused slot must ignore stale history on first prefill.
    adapted.bind_history(history)
    contexts = torch.full((3, 2), _PLE["eos_token_id"], dtype=torch.int64)
    expected_history = history.clone()
    generator = torch.Generator(device="cpu").manual_seed(314)
    with torch.no_grad():
        for ids, starts, slots, initial in (
            ([3, 4, 5, 6, 7, 0], [0, 3, 5, 5], [1, 2, -1], [False, False, False]),
            ([8, 9, 0], [0, 1, 2, 2], [2, 1, -1], [True, True, False]),
        ):
            ids, starts, slots = map(torch.tensor, (ids, starts, slots))
            initial = torch.tensor(initial)
            inputs = PLEInputs(ids, starts, contexts)
            hidden = torch.randn(len(ids), HC * HIDDEN, generator=generator).to(
                torch.bfloat16
            )
            windows = torch.where(
                initial[:, None, None], expected_history[slots.clamp_min(0)], 0
            )
            expected, next_windows = reference(hidden, inputs, windows)
            valid = starts[-1]
            raw = (
                adapted.lookup_for_model(inputs, valid) if adapted.cpu_offload else None
            )
            actual = adapted.forward_with_history(
                hidden, inputs, slots, initial, valid, raw
            )
            torch.testing.assert_close(actual[:valid], expected[:valid], rtol=0, atol=0)
            assert torch.count_nonzero(actual[valid:]) == 0
            for i, slot in enumerate(slots.tolist()):
                if slot > 0:
                    expected_history[slot] = next_windows[i]
            torch.testing.assert_close(history, expected_history, rtol=0, atol=0)
    if isinstance(adapted.storage, ple_storage.HostPLETable):
        adapted.storage.close()
