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
"""Export of a separate draft model (DSpark) as the artifact's drafter."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import torch.nn as nn

from tests.experimental.export.test_export_pipeline import (
    _mock_lower_reading_used_args,
    _TinyTorchExportableLLM,
)
from vllm_torchtpu.experimental.export import (
    api,
    cli,
    draft_model,
    exporter,
    tpu_lowering,
    universal_adapter,
    validator,
)

_AUX = 32  # two target layers of hidden size 16


@pytest.fixture(autouse=True)
def _tpu_accelerator_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-8")


class _TinyDSparkExportableLLM(_TinyTorchExportableLLM):
    """The tiny model with a separate draft model: `forward` also returns the
    residual stream of two layers, which the draft's own `draft.fc` reads; the
    draft shares the target's LM head and keeps one paged KV state."""

    def abstract_params(self) -> Mapping[str, api.ShapeDtypeStruct]:
        cfg = self._cfg
        return {
            **super().abstract_params(),
            "draft.fc.weight": api.ShapeDtypeStruct((_AUX, cfg.hidden_size), cfg.dtype),
        }

    def forward_extra_outputs(self) -> list[tuple[str, int]]:
        return [("aux_hidden_states", _AUX)]

    def forward(self, weights, kv_caches, **controls):  # type: ignore[override]
        kvs, hidden = super().forward(weights, kv_caches, **controls)
        return kvs, hidden, torch.cat([hidden, hidden], dim=-1)

    def dspark_abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        cfg = self._cfg
        return [
            api.ShapeDtypeStruct(
                (cfg.num_blocks, cfg.block_size, 2 * cfg.num_kv_heads, cfg.head_dim),
                cfg.dtype,
            )
        ]

    def dspark_draft(
        self,
        weights: Mapping[str, torch.Tensor],
        kv_caches: Sequence[torch.Tensor],
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_lens: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        req_dist: torch.Tensor,
        last_token_indices: torch.Tensor,
        aux_hidden_states: torch.Tensor,
    ) -> tuple[Sequence[torch.Tensor], torch.Tensor]:
        del input_ids, positions, kv_lens, block_tables, query_start_loc, req_dist
        rows = aux_hidden_states[last_token_indices.to(torch.int64)]
        hidden = torch.matmul(rows, weights["draft.fc.weight"])
        tokens = torch.matmul(hidden, weights["lm_head.weight"]).argmax(dim=-1)
        k = int(self._cfg.speculative_config["num_speculative_tokens"])  # type: ignore[index]
        return list(kv_caches), tokens.to(torch.int32).view(-1, 1).repeat(1, k)


def _dspark_config() -> api.TextOnlyLLMExportConfig:
    return api.TextOnlyLLMExportConfig(
        model_name="tiny-torch",
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8, 16),
        req_buckets=(4, 8, 16),
        emit_byte_tokenizer=True,
        speculative_config={"method": "dspark", "num_speculative_tokens": 2},
    )


def _export_dspark(tmp_path: Path, model_cls: type = _TinyDSparkExportableLLM) -> str:
    out_dir = str(tmp_path / "exported")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_reading_used_args,
    ):
        exporter.export(model_cls(_dspark_config()), out_dir)
    return out_dir


def test_dspark_draft_follows_the_contract(tmp_path) -> None:
    out_dir = _export_dspark(tmp_path)
    meta = json.loads(Path(out_dir, "metadata.json").read_text())
    functions = meta["functions"]
    assert meta["serving_config"]["speculative_config"] == {
        "method": "dspark",
        "num_speculative_tokens": 2,
    }
    assert set(functions["draft"]) == set(functions["forward"]) == {"8", "16"}

    # forward returns the target layers' residual stream after hidden_states.
    forward = functions["forward"]["16"]
    outs = [(o["name"], o["argument_type"], o["shape"]) for o in forward["outputs"]]
    assert outs[-2:] == [
        ("hidden_states", "intermediate", [16, 16]),
        ("aux_hidden_states", "intermediate", [16, _AUX]),
    ]

    variant = functions["draft"]["16"]
    assert variant["file_path"].endswith("draft_bucket16.mlir")
    by_name = {i["name"]: i for i in variant["inputs"]}
    user = [i["name"] for i in variant["inputs"] if i["argument_type"] == "user_io"]
    assert user == [
        "input_ids",
        "positions",
        "kv_lens",
        "block_tables",
        "query_start_loc",
        "req_dist",
        "last_token_indices",
    ]
    # The draft reads aux_hidden_states, not hidden_states.
    intermediates = [
        (i["name"], i["shape"])
        for i in variant["inputs"]
        if i["argument_type"] == "intermediate"
    ]
    assert intermediates == [("aux_hidden_states", [16, _AUX])]

    # The shared LM head binds the target's slot; the draft's own weight its own.
    logits_slots = {
        i["name"]: i["index"]
        for i in functions["compute_logits"]["4"]["inputs"]
        if i["argument_type"] == "weight"
    }
    forward_slots = {
        i["name"]: i["index"]
        for i in forward["inputs"]
        if i["argument_type"] == "weight"
    }
    weights = [i["name"] for i in variant["inputs"] if i["argument_type"] == "weight"]
    assert weights == ["lm_head.weight", "draft.fc.weight"]
    assert by_name["lm_head.weight"]["index"] == logits_slots["lm_head.weight"]
    assert "draft.fc.weight" not in forward_slots
    assert by_name["draft.fc.weight"]["index"] not in forward_slots.values()

    # Its paged KV is a donated state numbered after the target's.
    target_states = [i for i in forward["inputs"] if i["argument_type"] == "state"]
    state = by_name["draft_kv_cache_layer_0"]
    assert state["buffer_donor"] is True
    assert state["index"] == len(target_states)
    assert state["shape"][0] == 16  # num_blocks
    out_names = [o["name"] for o in variant["outputs"]]
    assert out_names == ["draft_kv_cache_out_layer_0", "draft_tokens"]
    assert variant["outputs"][-1]["shape"] == [8, 2]


def test_validator_rejects_a_draft_intermediate_forward_does_not_return(
    tmp_path,
) -> None:
    meta = json.loads(Path(_export_dspark(tmp_path), "metadata.json").read_text())
    validator._check_draft(meta, meta["functions"])
    meta["functions"]["forward"]["8"]["outputs"][-1]["name"] = "other_hidden_states"
    with pytest.raises(ValueError, match="takes intermediate aux_hidden_states"):
        validator._check_draft(meta, meta["functions"])


def test_validator_rejects_a_draft_intermediate_of_another_shape(tmp_path) -> None:
    meta = json.loads(Path(_export_dspark(tmp_path), "metadata.json").read_text())
    meta["functions"]["forward"]["8"]["outputs"][-1]["shape"] = [8, 8]
    with pytest.raises(ValueError, match="takes aux_hidden_states"):
        validator._check_draft(meta, meta["functions"])


class _TinyDSparkWithoutWeights(_TinyDSparkExportableLLM):
    def abstract_params(self) -> Mapping[str, api.ShapeDtypeStruct]:
        return _TinyTorchExportableLLM.abstract_params(self)


def test_dspark_needs_the_draft_models_weights(tmp_path) -> None:
    with pytest.raises(ValueError, match="no 'draft.' parameters"):
        _export_dspark(tmp_path, model_cls=_TinyDSparkWithoutWeights)


def test_dspark_needs_a_model_that_drafts(tmp_path) -> None:
    with pytest.raises(ValueError, match="dspark_abstract_states"):
        _export_dspark(tmp_path, model_cls=_TinyTorchExportableLLM)


def test_export_speculative_config_takes_a_dspark_draft_model() -> None:
    raw = {
        "method": "dspark",
        "num_speculative_tokens": 7,
        "model": "org/dspark-draft",
        "draft_tensor_parallel_size": 4,
    }
    # The artifact records what the runtime reads; vLLM loads the draft model.
    assert cli.export_speculative_config(raw, tensor_parallel_size=4) == {
        "method": "dspark",
        "num_speculative_tokens": 7,
    }


def test_export_speculative_config_declares_a_dflash_draft_as_dspark() -> None:
    # A DFlash draft is driven like a DSpark one (its own paged KV, the target's
    # aux hidden states, one block per step), so the artifact declares the
    # runtime protocol, dspark; vLLM still loads it as dflash.
    raw = {
        "method": "dflash",
        "num_speculative_tokens": 15,
        "model": "org/dflash-draft",
        "draft_tensor_parallel_size": 1,
    }
    assert cli.export_speculative_config(raw) == {
        "method": "dspark",
        "num_speculative_tokens": 15,
    }


@pytest.mark.parametrize(
    "raw, error",
    [
        ({"method": "dspark", "num_speculative_tokens": 7}, "'model'"),
        ({"method": "dflash", "num_speculative_tokens": 15}, "'model'"),
        (
            {
                "method": "dspark",
                "num_speculative_tokens": 7,
                "model": "x",
                "draft_tensor_parallel_size": 1,
            },
            "draft_tensor_parallel_size",
        ),
        (
            {
                "method": "dspark",
                "num_speculative_tokens": 7,
                "model": "x",
                "prompt_lookup_max": 3,
            },
            "fields",
        ),
    ],
)
def test_export_speculative_config_rejects_a_dspark_config_it_cannot_export(
    raw: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        cli.export_speculative_config(raw, tensor_parallel_size=4)


def _block_adapter(max_num_seqs: int = 4) -> Any:
    adapter = object.__new__(universal_adapter.UniversalVllmExportableLLM)
    adapter._export_config = api.TextOnlyLLMExportConfig(
        model_name="tiny", max_num_seqs=max_num_seqs, max_model_len=64
    )
    adapter._vllm_config = None
    adapter._model = nn.Module()
    return adapter


def test_draft_block_starts_at_the_token_after_each_last_accepted_row() -> None:
    # Two requests: a verify window at rows 0-3 (positions 10-13) that accepted
    # through row 1, and a one-token decode at row 4 (position 20). Shifted
    # input_ids carry each request's newly sampled token at that row.
    input_ids = torch.tensor([5, 6, 0, 0, 9, 0, 0, 0], dtype=torch.int32)
    positions = torch.tensor([10, 11, 12, 13, 20, 0, 0, 0], dtype=torch.int32)
    last = torch.tensor([1, 4, 0, 0], dtype=torch.int32)
    ids, pos, md = _block_adapter()._draft_block_inputs(
        input_ids=input_ids,
        positions=positions,
        block_tables=torch.zeros(32, dtype=torch.int32),
        num_seqs=torch.tensor(2, dtype=torch.int32),
        last_token_indices=last,
        block=3,
        mask_token_id=31,
    )
    assert ids.tolist() == [6, 31, 31, 9, 31, 31, 31, 31, 31, 31, 31, 31]
    assert pos.tolist() == [12, 13, 14, 21, 22, 23, 0, 1, 2, 0, 1, 2]
    # Each live block sees its context and itself; dead rows see nothing.
    assert md.seq_lens.tolist() == [15, 24, 0, 0]
    assert md.query_start_loc.tolist() == [0, 3, 6, 6, 6]
    assert md.request_distribution.tolist() == [0, 0, 2]


def test_draft_block_covers_only_the_requests_a_bucket_holds() -> None:
    # A bucket of 2 tokens holds at most 2 requests of the 4 max_num_seqs.
    ids, pos, md = _block_adapter()._draft_block_inputs(
        input_ids=torch.tensor([7, 8], dtype=torch.int32),
        positions=torch.tensor([3, 5], dtype=torch.int32),
        block_tables=torch.zeros(32, dtype=torch.int32),
        num_seqs=torch.tensor(2, dtype=torch.int32),
        last_token_indices=torch.tensor([0, 1], dtype=torch.int32),
        block=2,
        mask_token_id=31,
    )
    assert ids.tolist() == [7, 31, 8, 31]
    assert pos.tolist() == [4, 5, 6, 7]
    assert md.seq_lens.tolist() == [6, 8, 0, 0]
    assert md.query_start_loc.tolist() == [0, 2, 4, 4, 4]


def test_draft_model_tensors_are_named_after_the_draft_prefix() -> None:
    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = nn.Embedding(4, 2)
    draft = nn.Module()
    draft.model = nn.Module()
    draft.model.embed_tokens = target.model.embed_tokens
    draft.model.fc = nn.Linear(4, 2, bias=False)
    object.__setattr__(target, draft_model.DRAFT_MODEL_ATTR, draft)
    assert universal_adapter.draft_model_names(target) == {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.fc.weight": "draft.fc.weight",
    }
    assert set(universal_adapter.draft_model_tensors(target)) == {"draft.fc.weight"}


def test_query_block_follows_the_checkpoint_layout() -> None:
    # Dense DSpark checkpoints draft from the anchor; others keep 1 + K.
    assert draft_model.query_block_size(7, True) == 7
    assert draft_model.query_block_size(7, False) == 8
    dense = nn.Module()
    dense.config = SimpleConfig({})
    assert draft_model.sample_from_anchor(dense) is True
    dense.config = SimpleConfig({"sample_from_anchor": False})
    assert draft_model.sample_from_anchor(dense) is False


@dataclasses.dataclass
class SimpleConfig:
    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)
