# SPDX-License-Identifier: Apache-2.0
"""Preserve embedding values and token metadata across the DP stage boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as model_module


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("offload", [False, True])
def test_dp_stage_receives_materialized_embeddings_and_original_token_metadata(
    monkeypatch, supplied, offload
):
    ids = torch.tensor([3, 1, 3, 7])
    # Exercise the TPU boundary without allocating a device or loading a model.
    device_ids = SimpleNamespace(device=SimpleNamespace(type="tpu"), values=ids)
    positions = torch.arange(len(ids))
    lookback = torch.tensor([[8, 9]])
    table = torch.arange(16 * 256, dtype=torch.float32).reshape(16, 256).bfloat16()
    expected = table[ids]
    events = []

    def lookup(value):
        assert value is device_ids
        events.append("lookup")
        return table[value.values]

    def synchronize(values, *, wait):
        assert len(values) == 1
        assert wait is False
        torch.testing.assert_close(values[0], expected, rtol=0, atol=0)
        events.append("synchronize")

    def prepare(stage_ids, stage_positions, stage_lookback, *, stage):
        assert stage_ids is device_ids
        assert stage_positions is positions
        assert stage_lookback is lookback
        assert stage == "full"
        events.append("prepare")

    def stage(stage_ids, stage_positions, intermediate, embeddings, stage_lookback):
        assert stage_ids is device_ids
        assert stage_positions is positions
        assert stage_lookback is lookback
        assert intermediate is None
        torch.testing.assert_close(embeddings, expected, rtol=0, atol=0)
        torch.testing.assert_close(embeddings[0], embeddings[2], rtol=0, atol=0)
        events.append("stage")
        return embeddings

    owner = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=8),
        embed_input_ids=lookup,
        _has_cpu_offloaded_engram=lambda: offload,
        use_native_mega_moe=False,
        _prepare_engram=prepare,
        _ced_enabled=False,
        full_stage=stage,
    )
    monkeypatch.setattr(model_module, "synchronize_tensors", synchronize)
    result = model_module.DeepseekV41Model.forward(
        owner,
        device_ids,
        positions,
        inputs_embeds=expected if supplied else None,
        lookback_token_ids=lookback,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    expected_events = [] if supplied else ["lookup", "synchronize"]
    if offload:
        expected_events.append("prepare")
        assert owner._engram_prepared_outside_stage is False
    assert events == expected_events + ["stage"]


def test_tp_path_keeps_embedding_inside_the_stage(monkeypatch):
    ids = SimpleNamespace(device=SimpleNamespace(type="tpu"))
    positions = torch.arange(4)
    owner = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        embed_input_ids=Mock(side_effect=AssertionError("unexpected eager lookup")),
        _has_cpu_offloaded_engram=lambda: False,
        _ced_enabled=False,
        full_stage=Mock(return_value="result"),
    )
    synchronize = Mock(side_effect=AssertionError("unexpected synchronization"))
    monkeypatch.setattr(model_module, "synchronize_tensors", synchronize)
    assert model_module.DeepseekV41Model.forward(owner, ids, positions) == "result"
    owner.full_stage.assert_called_once_with(ids, positions, None, None, None)
