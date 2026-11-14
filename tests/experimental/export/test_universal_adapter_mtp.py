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
"""CPU tests of `UniversalVllmExportableLLM`'s MTP draft chain.

They run the adapter's real `mtp_draft`, `mtp_abstract_states` and
`_load_mtp_draft` with a small fake draft module and fake paged attention
layers, so the per-pass attention metadata, the drafted tokens and the KV
write-back are checked against a hand-computed reference.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_torchtpu.experimental.export import api, tpu_lowering, universal_adapter

VOCAB = 16
HIDDEN = 8
BLOCK_SIZE = 4
NUM_BLOCKS = 8
MAX_NUM_SEQS = 4
MAX_POSITION = 16
DRAFT_LAYER = "mtp.layers.0.attn"
TARGET_LAYER = "model.layers.0.attn"


class _FakePagedAttention(nn.Module, AttentionLayerBase):
    """A paged attention layer reading its metadata from the forward context.

    Each call records the metadata it saw, writes its input of each token row
    to that row's KV slot in place (as the real kernel's `copy_` writeback
    does) and returns the input plus a per-position vector, so the output
    depends on the positions in the metadata.
    """

    def __init__(self, name: str, forward_context: list[dict[str, Any]]) -> None:
        nn.Module.__init__(self)
        self.layer_name = name
        self.kv_cache: Any = [torch.empty(0)]
        self.pos_table = torch.randn(MAX_POSITION, HIDDEN)
        self.forward_context = forward_context
        self.calls: list[dict[str, torch.Tensor]] = []

    def get_attn_backend(self) -> Any:
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config: Any) -> Any:
        return None

    def forward(self, h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        md = self.forward_context[-1][self.layer_name]
        self.calls.append(
            {
                "positions_arg": positions.clone(),
                "input_positions": md.input_positions.clone(),
                "seq_lens": md.seq_lens.clone(),
                "query_start_loc": md.query_start_loc.clone(),
                "request_distribution": md.request_distribution.clone(),
                "block_tables": md.block_tables.clone(),
            }
        )
        qsl, seq_lens = md.query_start_loc.tolist(), md.seq_lens.tolist()
        for r in range(int(md.request_distribution[2])):
            end = qsl[r + 1]
            for i in range(qsl[r], end):
                idx = seq_lens[r] - (end - i)
                block = int(md.block_tables[r, idx // BLOCK_SIZE])
                self.kv_cache[block, idx % BLOCK_SIZE] = h[i]
        return h + self.pos_table[md.input_positions.long()]


class _FakeMtpDraft(nn.Module):
    """An MTP draft as vLLM builds one: `forward(input_ids, positions,
    hidden_states)` fuses the token embedding with the previous hidden state
    and runs one attention layer; the embedding and LM head are the target's."""

    def __init__(
        self, embed: nn.Module, lm_head: nn.Module, attn: _FakePagedAttention
    ) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = embed
        self.model.fc = nn.Linear(2 * HIDDEN, HIDDEN, bias=False)
        self.model.attn = attn
        self.lm_head = lm_head

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        emb = self.model.embed_tokens(input_ids)
        h = self.model.fc(torch.cat([emb, hidden], dim=-1))
        return self.model.attn(h, positions)


class _Target(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)


def _mtp_adapter(
    num_speculative_tokens: int,
) -> tuple[Any, _FakePagedAttention, list[dict[str, Any]]]:
    """An adapter around a target with an MTP draft, and the draft's attention."""
    forward_context: list[dict[str, Any]] = []
    target = _Target()
    attn = _FakePagedAttention(DRAFT_LAYER, forward_context)
    draft = _FakeMtpDraft(target.model.embed_tokens, target.lm_head, attn)
    object.__setattr__(target, universal_adapter.MTP_DRAFT_ATTR, draft)
    # vLLM registers every attention layer, the target's and the draft's.
    static_forward_context = {
        TARGET_LAYER: _FakePagedAttention(TARGET_LAYER, forward_context),
        DRAFT_LAYER: attn,
    }
    adapter = object.__new__(universal_adapter.UniversalVllmExportableLLM)
    adapter._export_config = api.TextOnlyLLMExportConfig(
        model_name="tiny-mtp",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_layers=1,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        max_num_seqs=MAX_NUM_SEQS,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
    )
    adapter._vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context=static_forward_context
        )
    )
    adapter._model = target
    adapter._mtp_draft = draft
    return adapter, attn, forward_context


@contextlib.contextmanager
def _recording_forward_context(
    forward_context: list[dict[str, Any]], num_tokens: list[int]
) -> Any:
    """Stands in for vLLM's `set_forward_context` and its one-rank all-gather."""

    @contextlib.contextmanager
    def set_forward_context(attn_metadata, vllm_config, num_tokens=None, **kwargs):
        del vllm_config, kwargs
        forward_context.append(attn_metadata)
        num_tokens_seen.append(num_tokens)
        try:
            yield
        finally:
            forward_context.pop()

    num_tokens_seen = num_tokens
    with (
        mock.patch("vllm.forward_context.set_forward_context", set_forward_context),
        mock.patch(
            "vllm.distributed.tensor_model_parallel_all_gather",
            lambda x, dim=-1: x,
        ),
    ):
        yield


# A step of 8 token rows and 4 request rows, decodes first:
#   request 0: a decode, row 0 at position 7 (KV length 8);
#   request 1: a speculative verify of 3 rows at positions 9-11 (KV length 12)
#     that accepted one draft, so its last row is row 2 and row 3 is rejected;
#   request 2: a prefill of 3 rows at positions 0-2 (KV length 3);
#   token row 7 and request row 3 are padding.
INPUT_IDS = torch.tensor([3, 5, 7, 9, 11, 13, 15, 0], dtype=torch.int32)
POSITIONS = torch.tensor([7, 9, 10, 11, 0, 1, 2, 0], dtype=torch.int32)
KV_LENS = torch.tensor([8, 12, 3, 0], dtype=torch.int32)
QUERY_START_LOC = torch.tensor([0, 1, 4, 7, 7], dtype=torch.int32)
REQ_DIST = torch.tensor([1, 1, 3], dtype=torch.int32)
LAST_TOKEN_INDICES = torch.tensor([0, 2, 6, 0], dtype=torch.int32)
BLOCK_TABLES = torch.tensor(
    [[4, 5, 6, 0], [1, 2, 3, 0], [7, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int32
)
NUM_SEQS = 3


def _run_mtp_draft(num_speculative_tokens: int) -> dict[str, Any]:
    torch.manual_seed(0)
    adapter, attn, forward_context = _mtp_adapter(num_speculative_tokens)
    weights = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN),
        "mtp.fc.weight": torch.randn(HIDDEN, 2 * HIDDEN),
    }
    hidden_states = torch.randn(int(INPUT_IDS.shape[0]), HIDDEN)
    kv = torch.full((NUM_BLOCKS, BLOCK_SIZE, HIDDEN), -100.0)
    num_tokens: list[int] = []
    with torch.no_grad(), _recording_forward_context(forward_context, num_tokens):
        updated, drafts = adapter.mtp_draft(
            weights,
            [kv],
            input_ids=INPUT_IDS,
            positions=POSITIONS,
            kv_lens=KV_LENS,
            block_tables=BLOCK_TABLES,
            query_start_loc=QUERY_START_LOC,
            req_dist=REQ_DIST,
            last_token_indices=LAST_TOKEN_INDICES,
            hidden_states=hidden_states,
        )
    return {
        "adapter": adapter,
        "attn": attn,
        "weights": weights,
        "hidden_states": hidden_states,
        "kv": kv,
        "updated": updated,
        "drafts": drafts,
        "num_tokens": num_tokens,
    }


def _reference_pass(
    weights: dict[str, torch.Tensor],
    pos_table: torch.Tensor,
    ids: torch.Tensor,
    positions: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One draft pass with the exported weights: (KV written, hidden, token)."""
    emb = weights["model.embed_tokens.weight"][ids.long()]
    kv_in = torch.cat([emb, hidden], dim=-1) @ weights["mtp.fc.weight"].T
    out = kv_in + pos_table[positions.long()]
    token = (out @ weights["lm_head.weight"].T).argmax(dim=-1)
    return kv_in, out, token


def test_mtp_draft_chains_two_passes_with_per_step_metadata() -> None:
    run = _run_mtp_draft(num_speculative_tokens=2)
    first, second = run["attn"].calls
    # Pass 1 runs every token row of the step with the step's own metadata.
    assert first["input_positions"].tolist() == POSITIONS.tolist()
    assert first["positions_arg"].tolist() == POSITIONS.tolist()
    assert first["seq_lens"].tolist() == KV_LENS.tolist()
    assert first["query_start_loc"].tolist() == QUERY_START_LOC.tolist()
    assert first["request_distribution"].tolist() == REQ_DIST.tolist()
    # Pass 2 runs one row per request at the position after its last kept row
    # (rows 0, 2 and 6, at positions 7, 10 and 2), all as decodes. Its KV
    # length is that position + 1; the padded request gets none.
    assert second["input_positions"][:NUM_SEQS].tolist() == [8, 11, 3]
    assert second["positions_arg"][:NUM_SEQS].tolist() == [8, 11, 3]
    assert second["seq_lens"].tolist() == [9, 12, 4, 0]
    assert second["query_start_loc"].tolist() == [0, 1, 2, 3, 3]
    assert second["request_distribution"].tolist() == [3, 3, 3]
    for call in (first, second):
        assert call["block_tables"].tolist() == BLOCK_TABLES.tolist()
    assert run["num_tokens"] == [8, MAX_NUM_SEQS]


def test_mtp_draft_tokens_and_kv_match_a_hand_computed_reference() -> None:
    run = _run_mtp_draft(num_speculative_tokens=2)
    weights, attn = run["weights"], run["attn"]
    rows = torch.tensor([0, 2, 6])
    kv_in1, out1, _ = _reference_pass(
        weights, attn.pos_table, INPUT_IDS, POSITIONS, run["hidden_states"]
    )
    # The first draft comes from each request's last kept row.
    token1 = (out1[rows] @ weights["lm_head.weight"].T).argmax(dim=-1)
    # The second feeds that draft and the draft's own hidden state, one
    # position further.
    kv_in2, _, token2 = _reference_pass(
        weights, attn.pos_table, token1, torch.tensor([8, 11, 3]), out1[rows]
    )
    drafts = run["drafts"]
    assert drafts.dtype == torch.int32
    assert tuple(drafts.shape) == (MAX_NUM_SEQS, 2)
    assert drafts[:NUM_SEQS].tolist() == torch.stack([token1, token2], 1).tolist()

    # KV slot of each write: KV index idx of a request lands in its block
    # table entry idx // 4, slot idx % 4.
    expected = torch.full((NUM_BLOCKS, BLOCK_SIZE, HIDDEN), -100.0)
    # Pass 1: row 0 at index 7; rows 1-3 at 9-11; rows 4-6 at 0-2. Row 7 is
    # padding and writes nothing.
    expected[5, 3] = kv_in1[0]
    expected[3, 1], expected[3, 2], expected[3, 3] = kv_in1[1], kv_in1[2], kv_in1[3]
    expected[7, 0], expected[7, 1], expected[7, 2] = kv_in1[4], kv_in1[5], kv_in1[6]
    # Pass 2: indices 8, 11 (over the rejected row 3) and 3; the padded
    # request writes nothing.
    expected[6, 0], expected[3, 3], expected[7, 3] = kv_in2[0], kv_in2[1], kv_in2[2]
    (updated,) = run["updated"]
    torch.testing.assert_close(updated, expected)
    assert updated is run["kv"]


def test_mtp_draft_runs_with_the_bound_weights_not_the_modules() -> None:
    run = _run_mtp_draft(num_speculative_tokens=1)
    draft = run["adapter"]._mtp_draft
    assert not torch.equal(draft.model.fc.weight, run["weights"]["mtp.fc.weight"])
    # With K = 1 there is a single pass and one draft per request.
    (only,) = run["attn"].calls
    assert only["seq_lens"].tolist() == KV_LENS.tolist()
    _, out1, _ = _reference_pass(
        run["weights"],
        run["attn"].pos_table,
        INPUT_IDS,
        POSITIONS,
        run["hidden_states"],
    )
    token1 = (out1[[0, 2, 6]] @ run["weights"]["lm_head.weight"].T).argmax(dim=-1)
    assert run["drafts"][:NUM_SEQS, 0].tolist() == token1.tolist()
    # The module keeps its own weights after the call.
    assert draft.model.fc.weight is draft.model.fc._parameters["weight"]


def _pallas_impl() -> Any:
    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl

    impl = object.__new__(PallasAttentionBackendImpl)
    impl.num_kv_heads, impl.head_size = 2, 128
    impl.kv_cache_quantized_dtype = None
    return impl


def test_mtp_abstract_states_declare_only_the_drafts_paged_kv() -> None:
    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend

    adapter, attn, _ = _mtp_adapter(num_speculative_tokens=2)
    attn.impl = _pallas_impl()
    static = adapter._vllm_config.compilation_config.static_forward_context
    static[TARGET_LAYER].impl = _pallas_impl()
    (spec,) = adapter.mtp_abstract_states()
    assert spec.dtype == "bf16"
    assert spec.shape == PallasAttentionBackend.get_kv_cache_shape(
        NUM_BLOCKS, BLOCK_SIZE, 2, 128, tpu_lowering.torch_dtype("bf16")
    )
    assert [n for n, _ in adapter._mtp_attention_layers()] == [DRAFT_LAYER]
    assert [n for n, _ in adapter._named_attention_layers()] == [TARGET_LAYER]


def test_mtp_abstract_states_reject_a_draft_layer_without_paged_kv() -> None:
    adapter, _, _ = _mtp_adapter(num_speculative_tokens=2)
    with pytest.raises(NotImplementedError, match="keeps no paged KV cache"):
        adapter.mtp_abstract_states()


def test_mtp_abstract_states_need_a_draft() -> None:
    adapter, _, _ = _mtp_adapter(num_speculative_tokens=2)
    adapter._mtp_draft = None
    with pytest.raises(ValueError, match="the model has no MTP draft"):
        adapter.mtp_abstract_states()


class _Embed(VocabParallelEmbedding):
    """A `VocabParallelEmbedding` that needs no tensor-parallel group."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.randn(VOCAB, HIDDEN))


class _Head(ParallelLMHead):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.randn(VOCAB, HIDDEN))


def _vllm_model(own: str) -> nn.Module:
    m = nn.Module()
    m.model = nn.Module()
    m.model.embed_tokens = _Embed()
    m.model.layers = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN)])
    m.lm_head = _Head()
    if own:
        setattr(m.model, own, nn.Linear(2 * HIDDEN, HIDDEN, bias=False))
    return m


def test_load_mtp_draft_loads_the_draft_config_and_shares_embedding_and_head() -> None:
    target = _vllm_model("")
    loaded = _vllm_model("fc")
    draft_model_config = SimpleNamespace(runner_type="generate", model="ckpt")
    load_config = object()
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp", draft_model_config=draft_model_config
        ),
        load_config=load_config,
    )
    current: list[Any] = []

    @contextlib.contextmanager
    def set_current_vllm_config(config):
        current.append(config)
        yield

    loader = mock.MagicMock()

    def load_model(*, vllm_config, model_config):
        # Loaded under the draft's own copy of the config.
        assert current == [vllm_config]
        return loaded

    loader.load_model.side_effect = load_model
    with (
        mock.patch("vllm.config.set_current_vllm_config", set_current_vllm_config),
        mock.patch(
            "vllm.model_executor.model_loader.get_model_loader", return_value=loader
        ) as get_loader,
    ):
        draft = universal_adapter._load_mtp_draft(vllm_config, target)
    assert draft is loaded
    get_loader.assert_called_once_with(load_config)
    kwargs = loader.load_model.call_args.kwargs
    assert kwargs["vllm_config"] is not vllm_config
    assert kwargs["vllm_config"].speculative_config is vllm_config.speculative_config
    assert kwargs["model_config"] is not draft_model_config
    assert kwargs["model_config"].runner_type == "draft"
    assert kwargs["model_config"].model == "ckpt"
    assert draft_model_config.runner_type == "generate"
    # The draft projects with the target's embedding and LM head.
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is target.lm_head
    object.__setattr__(target, universal_adapter.MTP_DRAFT_ATTR, draft)
    names = universal_adapter.mtp_draft_names(target)
    assert names["model.embed_tokens.weight"] == "model.embed_tokens.weight"
    assert names["lm_head.weight"] == "lm_head.weight"
    assert names["model.fc.weight"] == "mtp.fc.weight"
    assert names["model.layers.0.weight"] == "mtp.layers.0.weight"


@pytest.mark.parametrize("spec", [None, SimpleNamespace(method="eagle")])
def test_load_mtp_draft_skips_configs_without_mtp(spec: Any) -> None:
    with mock.patch("vllm.model_executor.model_loader.get_model_loader") as get_loader:
        draft = universal_adapter._load_mtp_draft(
            SimpleNamespace(speculative_config=spec), _vllm_model("")
        )
    assert draft is None
    get_loader.assert_not_called()


def test_share_embedding_and_lm_head_needs_exactly_one_embedding() -> None:
    draft = _vllm_model("fc")
    draft.extra = nn.Module()
    draft.extra.embed_tokens = _Embed()
    with pytest.raises(
        ValueError, match=r"expected one VocabParallelEmbedding named \*embed_tokens"
    ):
        universal_adapter._share_embedding_and_lm_head(draft, _vllm_model(""))
