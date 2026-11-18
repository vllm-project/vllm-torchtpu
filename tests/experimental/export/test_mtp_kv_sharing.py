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
"""An MTP draft that reads the target's KV cache (Gemma 4's assistant): the
export wires it as the TPU runner's MTP proposer does, and drafts with it as
that proposer chains it."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.experimental.export import api, cli, universal_adapter
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

_SLIDING, _FULL = "sliding_attention", "full_attention"


def _config(layer_types: list[str], num_kv_shared_layers: int = 0) -> Any:
    text = SimpleNamespace(
        layer_types=layer_types, num_kv_shared_layers=num_kv_shared_layers
    )
    return SimpleNamespace(get_text_config=lambda: text)


def _draft(layer_types: list[str], shared: bool = True) -> nn.Module:
    """A draft whose attention layers declare `is_kv_shared_layer`."""
    draft = nn.Module()
    draft.config = _config(layer_types)
    draft.model = nn.Module()
    draft.model.layers = nn.ModuleList([nn.Module() for _ in layer_types])
    for i, layer in enumerate(draft.model.layers):
        layer.self_attn = nn.Module()
        layer.self_attn.is_kv_shared_layer = shared
        layer.self_attn.attn = SimpleNamespace(
            layer_name=f"draft.layers.{i}.self_attn.attn",
            kv_sharing_target_layer_name=None,
            impl=SimpleNamespace(kv_sharing_target_layer_name=None),
        )
    return draft


def _wired(draft: nn.Module) -> list[tuple[Any, Any]]:
    return [
        (
            layer.self_attn.attn.kv_sharing_target_layer_name,
            layer.self_attn.attn.impl.kv_sharing_target_layer_name,
        )
        for layer in draft.model.layers
    ]


def _runner_wiring(draft: nn.Module, target: Any, names: set[str]) -> None:
    proposer = SimpleNamespace(
        draft_model=draft,
        _draft_kv_shared=False,
        _validate_kv_sharing_supported=lambda: None,
    )
    Eagle3Proposer._setup_gemma4_kv_sharing(proposer, target, names)  # type: ignore[arg-type]


_TARGET_NAMES = {
    f"language_model.model.layers.{i}.self_attn.attn" for i in range(8)
} | {"vision.encoder.attn"}


@pytest.mark.parametrize(
    "target_types, num_kv_shared, draft_types",
    [
        # Gemma 4 31B: no KV-shared target layers; three sliding draft layers
        # and one full one.
        ([_SLIDING] * 5 + [_FULL, _SLIDING, _FULL], 0, [_SLIDING] * 3 + [_FULL]),
        # A target whose last layers share KV themselves (Gemma 4 E2B style):
        # the draft reads the last layers that keep their own.
        ([_SLIDING, _FULL] * 4, 4, [_SLIDING, _FULL]),
        # A draft layer type past the draft config's list is full attention.
        ([_SLIDING, _FULL] * 4, 0, [_SLIDING]),
    ],
)
def test_kv_sharing_wiring_matches_the_tpu_runner(
    target_types: list[str], num_kv_shared: int, draft_types: list[str]
) -> None:
    target = SimpleNamespace(config=_config(target_types, num_kv_shared))
    runner_draft, export_draft = _draft(draft_types), _draft(draft_types)
    if len(draft_types) == 1:
        for d in (runner_draft, export_draft):
            d.model.layers.append(_draft([_FULL]).model.layers[0])
    _runner_wiring(runner_draft, target, _TARGET_NAMES)
    universal_adapter._share_target_kv_cache(export_draft, target, _TARGET_NAMES)
    assert _wired(export_draft) == _wired(runner_draft)
    assert all(a is not None and a == b for a, b in _wired(export_draft))


def test_kv_sharing_leaves_a_draft_that_keeps_its_own_kv() -> None:
    target = SimpleNamespace(config=_config([_SLIDING, _FULL]))
    draft = _draft([_SLIDING, _FULL], shared=False)
    universal_adapter._share_target_kv_cache(draft, target, _TARGET_NAMES)
    assert _wired(draft) == [(None, None), (None, None)]


def test_kv_sharing_refuses_a_layer_no_target_layer_matches() -> None:
    target = SimpleNamespace(config=_config([_SLIDING] * 4))
    names = {f"model.layers.{i}.self_attn.attn" for i in range(4)}
    with pytest.raises(RuntimeError, match="full_attention") as runner_err:
        _runner_wiring(_draft([_FULL]), target, names)
    with pytest.raises(RuntimeError, match="full_attention") as export_err:
        universal_adapter._share_target_kv_cache(_draft([_FULL]), target, names)
    assert str(export_err.value) == str(runner_err.value)


def test_a_draft_with_its_own_lm_head_keeps_it_and_shares_the_embedding() -> None:
    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = nn.Embedding(64, 16)
    target.lm_head = nn.Linear(16, 64, bias=False)
    draft = nn.Module()
    draft.model = nn.Module()
    draft.model.embed_tokens = nn.Embedding(64, 8)
    draft.lm_head = nn.Linear(8, 64, bias=False)
    draft.lm_head.weight = draft.model.embed_tokens.weight
    own_head = draft.lm_head
    draft.has_own_lm_head = True
    universal_adapter._share_embedding_and_lm_head(draft, target)
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is own_head

    # A draft without its own head projects with the target's.
    del draft.has_own_lm_head
    universal_adapter._share_embedding_and_lm_head(draft, target)
    assert draft.lm_head is target.lm_head


def test_export_speculative_config_takes_an_mtp_draft_model() -> None:
    raw = {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "model": "org/assistant",
        "draft_tensor_parallel_size": 8,
    }
    assert cli.export_speculative_config(raw, tensor_parallel_size=8) == {
        "method": "mtp",
        "num_speculative_tokens": 3,
    }
    with pytest.raises(ValueError, match="draft_tensor_parallel_size"):
        cli.export_speculative_config(
            {**raw, "draft_tensor_parallel_size": 1}, tensor_parallel_size=8
        )


# --- Drafting ------------------------------------------------------------------

_H, _V, _MAX_SEQS = 4, 8, 2


class _QOnlyAttn(nn.Module):
    """A draft attention layer that reads the target's KV: it records the KV
    it is bound to and adds the KV's first value, so the drafts depend on it."""

    def __init__(self, name: str, target: str, calls: list[Any]) -> None:
        super().__init__()
        self.layer_name = name
        self.kv_sharing_target_layer_name = target
        self.kv_cache: Any = None
        self.calls = calls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append((self.layer_name, self.kv_cache))
        return x + self.kv_cache.flatten()[0]


class _AssistantDraft(nn.Module):
    """Gemma 4's assistant in miniature: it reads the token embedding and the
    hidden state it is fed, returns `(draft_hidden, backbone_hidden)`, and
    projects with its own LM head."""

    has_own_lm_head = True

    def __init__(self, calls: list[Any]) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed = nn.Embedding(_V, _H)
        self.model.attn0 = _QOnlyAttn("d0", "model.layers.2.self_attn.attn", calls)
        self.model.attn1 = _QOnlyAttn("d1", "model.layers.0.self_attn.attn", calls)
        self.model.attn2 = _QOnlyAttn("d2", "model.layers.2.self_attn.attn", calls)
        self.post_projection = nn.Linear(_H, _H, bias=False)
        self.lm_head = nn.Linear(_H, _V, bias=False)
        self.calls = calls

    def forward(self, ids, positions, hidden):
        self.calls.append(("forward", positions.clone()))
        x = self.model.embed(ids.to(torch.int64)) + hidden
        for attn in (self.model.attn0, self.model.attn1, self.model.attn2):
            x = attn(x)
        return x, self.post_projection(x)


def _assistant_adapter(k: int) -> tuple[Any, _AssistantDraft, list[Any]]:
    calls: list[Any] = []
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny",
        vocab_size=_V,
        hidden_size=_H,
        max_num_seqs=_MAX_SEQS,
        max_model_len=64,
        speculative_config={"method": "mtp", "num_speculative_tokens": k},
    )
    adapter = object.__new__(universal_adapter.UniversalVllmExportableLLM)
    adapter._export_config = cfg
    adapter._vllm_config = None
    adapter._model = nn.Module()
    adapter._model.lm_head = nn.Linear(_H, _V, bias=False)
    draft = _AssistantDraft(calls)
    adapter._mtp_draft = draft
    object.__setattr__(adapter._model, universal_adapter.MTP_DRAFT_ATTR, draft)
    target_layers = [
        (f"model.layers.{i}.self_attn.attn", SimpleNamespace()) for i in range(3)
    ]
    draft_layers = [
        (a.layer_name, a)
        for a in (draft.model.attn0, draft.model.attn1, draft.model.attn2)
    ]
    adapter._all_attention_layers = lambda: target_layers + draft_layers
    adapter.abstract_states = lambda: [
        api.ShapeDtypeStruct((16, i + 1), "bf16") for i in range(3)
    ]
    return adapter, draft, calls


def test_a_kv_sharing_draft_takes_the_target_kv_as_its_states() -> None:
    adapter, _, _ = _assistant_adapter(k=2)
    # Each target layer the draft reads, in the order its layers first read it.
    assert adapter.mtp_state_indices() == [2, 0]
    assert [s.shape for s in adapter.mtp_abstract_states()] == [(16, 3), (16, 1)]


def test_a_kv_sharing_draft_counts_no_kv_of_its_own() -> None:
    adapter, _, _ = _assistant_adapter(k=2)
    target_only = adapter._state_bytes(16)
    adapter._mtp_draft = None
    assert adapter._state_bytes(16) == target_only


def test_a_kv_sharing_draft_chains_as_the_runner_does() -> None:
    """Each pass reads the target's KV at the first pass's last position and
    KV length (the runner's `advance_draft_positions` is False for a draft
    sharing the target's KV), feeds its backbone-width hidden state on, and
    picks tokens with its own LM head."""
    k = 3
    adapter, draft, calls = _assistant_adapter(k)
    metadata: list[dict[str, Any]] = []

    def record_metadata(**kwargs: Any) -> dict[str, Any]:
        metadata.append(kwargs)
        return kwargs

    heads: list[torch.Tensor] = []

    def greedy(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        heads.append(weight)
        return torch.nn.functional.linear(hidden, weight).argmax(-1).to(torch.int32)

    adapter.greedy_tokens = greedy
    kv = [torch.full((16, 3), 2.0), torch.full((16, 1), 3.0)]
    own_head = torch.randn(_V, _H)
    weights = {
        "mtp.embed.weight": draft.model.embed.weight.detach(),
        "mtp.post_projection.weight": draft.post_projection.weight.detach(),
        "mtp.lm_head.weight": own_head,
        "lm_head.weight": torch.randn(_V, _H),
    }
    # Request 0 verified rows 0-2 (KV length 12, positions 9-11) and accepted
    # through row 1; request 1 decoded row 3 (KV length 5, position 4).
    with mock.patch.object(
        universal_adapter, "make_attention_metadata", side_effect=record_metadata
    ):
        states, drafts = adapter.mtp_draft(
            weights,
            kv,
            input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            positions=torch.tensor([9, 10, 11, 4], dtype=torch.int32),
            kv_lens=torch.tensor([12, 5], dtype=torch.int32),
            block_tables=torch.zeros(8, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3, 4], dtype=torch.int32),
            req_dist=torch.tensor([1, 1, 2], dtype=torch.int32),
            last_token_indices=torch.tensor([1, 3], dtype=torch.int32),
            hidden_states=torch.randn(4, _H),
        )
    assert drafts.shape == (_MAX_SEQS, k)
    # The states are the target's KV, passed through.
    assert states[0] is kv[0] and states[1] is kv[1]
    # Draft layers read the KV of the target layer they share.
    reads = {name: t for name, t in calls if name in ("d0", "d1", "d2")}
    assert reads["d0"] is kv[0] and reads["d2"] is kv[0] and reads["d1"] is kv[1]
    # Loop passes stay at the last accepted row's position and KV length.
    loop_positions = [p for name, p in calls if name == "forward"][1:]
    assert all(p.tolist() == [10, 4] for p in loop_positions)
    assert len(loop_positions) == k - 1
    for md in metadata[1:]:
        assert md["seq_lens"].tolist() == [11, 5]
    # Every token comes from the draft's own LM head.
    assert len(heads) == k and all(h is own_head for h in heads)


def test_a_draft_with_masked_logits_is_refused() -> None:
    adapter, draft, _ = _assistant_adapter(k=1)
    draft.masked_embedding = nn.Identity()
    with pytest.raises(NotImplementedError, match="ordered embeddings"):
        adapter.mtp_draft(
            {},
            [],
            input_ids=torch.zeros(4, dtype=torch.int32),
            positions=torch.zeros(4, dtype=torch.int32),
            kv_lens=torch.zeros(2, dtype=torch.int32),
            block_tables=torch.zeros(8, dtype=torch.int32),
            query_start_loc=torch.zeros(3, dtype=torch.int32),
            req_dist=torch.zeros(3, dtype=torch.int32),
            last_token_indices=torch.zeros(2, dtype=torch.int32),
            hidden_states=torch.zeros(4, _H),
        )
