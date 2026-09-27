# Copyright 2025 Google LLC
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

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpecDecodeMetadata,
    SpeculativeDecodingManager,
)
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer


def test_speculative_decoding_manager_metadata_indices(device):
    # Mock TPUModelRunner
    mock_runner = MagicMock()
    mock_runner.device = device
    mock_runner.arange_np = np.arange(100, dtype=np.int32)
    mock_runner.num_tokens_paddings = [8, 16, 32, 64]

    # Mock CPU input IDs tensor
    # Total length is logits indices size
    # E.g., 10 scheduled tokens
    input_ids_cpu = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int32)
    mock_runner.input_ids_cpu = input_ids_cpu

    # Instantiate manager
    manager = SpeculativeDecodingManager(mock_runner)

    # Mock input batch:
    # num_draft_tokens = [3, 2] (batch_size = 2)
    # cu_num_scheduled_tokens = [4, 7] (first request has 4 scheduled tokens,
    # second has 3)
    num_draft_tokens = np.array([3, 2], dtype=np.int32)
    cu_num_scheduled_tokens = np.array([4, 7], dtype=np.int32)
    padded_num_reqs = 4

    metadata = manager.get_spec_decode_metadata(
        num_draft_tokens=num_draft_tokens,
        cu_num_scheduled_tokens=cu_num_scheduled_tokens,
        padded_num_reqs=padded_num_reqs,
    )

    # Verify shapes and properties of the returned SpecDecodeMetadata
    assert isinstance(metadata, SpecDecodeMetadata)

    # Expected draft lengths cpu array
    assert np.array_equal(metadata.draft_lengths_cpu, num_draft_tokens)

    # Output tensors are copied to target device
    assert metadata.draft_token_ids.device.type == device.type
    assert metadata.draft_lengths.device.type == device.type
    assert metadata.target_logits_indices.device.type == device.type
    assert metadata.bonus_logits_indices.device.type == device.type
    assert metadata.final_logits_indices.device.type == device.type
    assert metadata.segment_ids.device.type == device.type
    assert metadata.group_indices.device.type == device.type

    # 1. Draft Lengths: num_draft_tokens padded to padded_num_reqs (4) -> [3, 2, 0, 0]
    expected_draft_lengths = torch.tensor(
        [3, 2, 0, 0], dtype=torch.int32, device=device
    )
    assert torch.equal(metadata.draft_lengths, expected_draft_lengths)

    # 2. Bonus Logits Indices: [3, 6] (cu_num_sampled_tokens - 1) padded to
    # padded_num_reqs (4) -> [3, 6, 0, 0]
    expected_bonus_indices = torch.tensor(
        [3, 6, 0, 0], dtype=torch.int32, device=device
    )
    assert torch.equal(metadata.bonus_logits_indices, expected_bonus_indices)

    # 3. Segment IDs: repeat request indices [0, 0, 0, 1, 1] padded to nearest
    # static token bucket (8)
    # Unpadded size: 3 + 2 = 5 tokens. Nearest static padding is 8.
    # Padded with total request count (2) -> [0, 0, 0, 1, 1, 2, 2, 2]
    expected_segment_ids = torch.tensor(
        [0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int64, device=device
    )
    assert torch.equal(metadata.segment_ids, expected_segment_ids)

    # 4. Group Indices: range per segment [0, 1, 2, 0, 1] padded to static bucket (8)
    # with 0 -> [0, 1, 2, 0, 1, 0, 0, 0]
    expected_group_indices = torch.tensor(
        [0, 1, 2, 0, 1, 0, 0, 0], dtype=torch.int32, device=device
    )
    assert torch.equal(metadata.group_indices, expected_group_indices)


def _make_manager_with_eagle3_drafter(num_reqs=2):
    mock_runner = MagicMock()
    mock_runner.input_batch.num_reqs = num_reqs
    mock_runner.speculative_config.method = "eagle3"
    mock_drafter = MagicMock(spec=Eagle3Proposer)
    mock_runner.drafter = mock_drafter
    manager = SpeculativeDecodingManager(mock_runner)
    return manager, mock_runner, mock_drafter


def test_propose_draft_token_ids_sync_truncates_and_caches():
    manager, _, mock_drafter = _make_manager_with_eagle3_drafter(num_reqs=2)
    mock_drafter.propose.return_value = [[1, 2], [3, 4]]
    num_rejected = np.array([0, 1], dtype=np.int32)

    result = manager.propose_draft_token_ids(
        sampled_token_ids=[[10], [20], [30]],
        discard_sampled_tokens_req_indices=[1],
        num_rejected_tokens_np=num_rejected,
        scheduler_output="SO",
    )

    assert result is None
    assert manager._draft_token_ids == [[1, 2], [3, 4]]
    args, kwargs = mock_drafter.propose.call_args
    assert args[0] == [[10], [20]]  # truncated to num_reqs
    assert args[1] == [1]
    assert np.array_equal(args[2], num_rejected)
    assert args[3] == "SO"
    assert kwargs == {
        "return_device": False,
        "next_tokens_per_chunk": None,
        "device_seed": None,
    }


def test_propose_draft_token_ids_sync_device_seeded_caches_host_list():
    manager, _, mock_drafter = _make_manager_with_eagle3_drafter(num_reqs=2)
    mock_drafter.propose.return_value = [[1, 2], [3, 4]]

    result = manager.propose_draft_token_ids(
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output="SO",
        return_device=False,
        next_tokens_per_chunk=["chunk0"],
    )

    assert result is None
    assert manager._draft_token_ids == [[1, 2], [3, 4]]
    args, kwargs = mock_drafter.propose.call_args
    assert args[0] is None
    assert kwargs["return_device"] is False
    assert kwargs["next_tokens_per_chunk"] == ["chunk0"]


def test_propose_draft_token_ids_async_spec_returns_device_tensor():
    manager, _, mock_drafter = _make_manager_with_eagle3_drafter(num_reqs=3)
    device_tensor = torch.zeros((3, 4))
    mock_drafter.propose.return_value = device_tensor

    result = manager.propose_draft_token_ids(
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[0],
        num_rejected_tokens_np=None,
        scheduler_output="SO",
        return_device=True,
        next_tokens_per_chunk=["chunk0"],
    )

    assert result is device_tensor
    assert manager._draft_token_ids is None  # not cached on the async path
    args, kwargs = mock_drafter.propose.call_args
    assert args[0] is None
    assert kwargs["return_device"] is True
    assert kwargs["next_tokens_per_chunk"] == ["chunk0"]
    assert kwargs["device_seed"] is None


def test_propose_draft_token_ids_async_bootstrap_returns_device_tensor():
    manager, _, mock_drafter = _make_manager_with_eagle3_drafter(num_reqs=1)
    device_tensor = torch.ones((1, 2))
    mock_drafter.propose.return_value = device_tensor
    seed = torch.tensor([7])

    result = manager.propose_draft_token_ids(
        sampled_token_ids=[],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output="SO",
        return_device=True,
        device_seed=seed,
    )

    assert result is device_tensor
    assert manager._draft_token_ids is None
    args, kwargs = mock_drafter.propose.call_args
    assert args[0] == []
    assert kwargs["next_tokens_per_chunk"] is None
    assert kwargs["device_seed"] is seed


def test_stage_and_take_draft_token_ids_snapshots_req_ids():
    # Async scheduling: stage launches the D2H copy and snapshots req_ids;
    # take is only called after the NEXT step's execute_model has already
    # mutated input_batch, so it must answer from the snapshot.
    manager, mock_runner, _ = _make_manager_with_eagle3_drafter(num_reqs=2)
    mock_runner.input_batch.req_ids = ["req-a", "req-b", None]
    drafts = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

    manager.stage_draft_token_ids_for_host(drafts)

    # Simulate the next step's batch update (req-a finished, req-c joined).
    mock_runner.input_batch.num_reqs = 2
    mock_runner.input_batch.req_ids = ["req-c", "req-b", None]

    result = manager.take_draft_token_ids()
    assert result is not None
    assert result.req_ids == ["req-a", "req-b"]
    assert result.draft_token_ids == [[1, 2, 3], [4, 5, 6]]
    # Consumed: a second take (nothing staged, no sync cache) returns None.
    assert manager._staged_draft_copy is None
    assert manager.take_draft_token_ids() is None


def test_stage_draft_token_ids_rejects_row_mismatch():
    manager, mock_runner, _ = _make_manager_with_eagle3_drafter(num_reqs=2)
    mock_runner.input_batch.req_ids = ["req-a", "req-b", None]
    with pytest.raises(AssertionError):
        manager.stage_draft_token_ids_for_host(torch.zeros((3, 2), dtype=torch.int32))


def test_propose_draft_token_ids_ngram_dispatches_correctly():
    mock_runner = MagicMock()
    mock_runner.input_batch.num_reqs = 2
    mock_runner.speculative_config.method = "ngram"
    mock_drafter = MagicMock(spec=NgramProposer)
    mock_drafter.propose.return_value = [[5], [6]]
    mock_runner.drafter = mock_drafter
    mock_runner.input_batch.num_tokens_no_spec = "NTS"
    mock_runner.input_batch.token_ids_cpu = "TIDS"
    manager = SpeculativeDecodingManager(mock_runner)
    scheduler_output = MagicMock()
    scheduler_output.num_spec_tokens_to_schedule = 4

    result = manager.propose_draft_token_ids(
        sampled_token_ids=[[1], [2], [3]], scheduler_output=scheduler_output
    )

    assert result is None
    assert manager._draft_token_ids == [[5], [6]]
    mock_drafter.propose.assert_called_once_with(4, [[1], [2]], "NTS", "TIDS")


def _fresh_metadata_manager(device, input_ids):
    mock_runner = MagicMock()
    mock_runner.device = device
    mock_runner.arange_np = np.arange(100, dtype=np.int32)
    mock_runner.num_tokens_paddings = [8, 16, 32, 64]
    mock_runner.input_ids_cpu = input_ids
    return SpeculativeDecodingManager(mock_runner)


def test_spec_decode_metadata_cache_hit_reuses_index_tensors(device):
    # Same (num_draft_tokens, cu_num_scheduled_tokens, padded_num_reqs) ->
    # the 7 index tensors are reused (same device objects, no re-transfer),
    # while draft_token_ids is re-extracted from the CURRENT input_ids_cpu.
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int32)
    manager = _fresh_metadata_manager(device, input_ids)
    nd = np.array([3, 2], dtype=np.int32)
    cu = np.array([4, 7], dtype=np.int32)

    m1 = manager.get_spec_decode_metadata(nd, cu, 4)
    # New step: same layout, different token values.
    manager.runner.input_ids_cpu = input_ids + 100
    m2 = manager.get_spec_decode_metadata(nd.copy(), cu.copy(), 4)

    for f in (
        "draft_lengths",
        "target_logits_indices",
        "bonus_logits_indices",
        "final_logits_indices",
        "segment_ids",
        "group_indices",
    ):
        assert getattr(m2, f) is getattr(m1, f), f"{f} not reused"
    assert m2.draft_token_ids is not m1.draft_token_ids
    # Values must equal an uncached build on the same inputs.
    ref = _fresh_metadata_manager(device, input_ids + 100).get_spec_decode_metadata(
        nd, cu, 4
    )
    assert torch.equal(m2.draft_token_ids.cpu(), ref.draft_token_ids.cpu())


def test_spec_decode_metadata_cache_miss_on_changed_inputs(device):
    # Any change in the key inputs must rebuild, with values matching an
    # uncached build.
    input_ids = torch.arange(1, 21, dtype=torch.int32)
    manager = _fresh_metadata_manager(device, input_ids)
    m1 = manager.get_spec_decode_metadata(
        np.array([3, 2], dtype=np.int32), np.array([4, 7], dtype=np.int32), 4
    )
    nd2 = np.array([2, 2], dtype=np.int32)
    cu2 = np.array([3, 6], dtype=np.int32)
    m2 = manager.get_spec_decode_metadata(nd2, cu2, 4)
    assert m2.final_logits_indices is not m1.final_logits_indices
    ref = _fresh_metadata_manager(device, input_ids).get_spec_decode_metadata(
        nd2, cu2, 4
    )
    for f in (
        "draft_token_ids",
        "draft_lengths",
        "target_logits_indices",
        "bonus_logits_indices",
        "final_logits_indices",
        "segment_ids",
        "group_indices",
    ):
        assert torch.equal(getattr(m2, f).cpu(), getattr(ref, f).cpu()), (
            f"{f} differs from uncached build"
        )
    assert np.array_equal(m2.draft_lengths_cpu, nd2)
