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
import torch

from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState


def test_local_suffix_update():
    mask = TokenPaddingState(local_padding_mask=torch.zeros(8, dtype=torch.bool))
    mask.update(3, 6)
    assert mask.get_local_padding_mask(6).tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    mask.update(2, 2)
    assert mask.get_local_padding_mask(8).sum().item() == 0


def test_dummy_batch_is_all_padding():
    state = TokenPaddingState(local_padding_mask=torch.zeros(8, dtype=torch.bool))
    state.update(0, 6)
    assert state.get_local_padding_mask(6).tolist() == [True] * 6


def test_zero_routing_weights_for_padding_zeroes_weights_keeps_ids(monkeypatch):
    provider = TokenPaddingState(local_padding_mask=torch.zeros(4, dtype=torch.bool))
    token_padding.set_padding_state(provider)

    class _FakeGroup:
        world_size = 1

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dp_group", lambda: _FakeGroup()
    )
    try:
        provider.update(1, 4)
        ids = torch.arange(8, dtype=torch.int32).reshape(4, 2)
        weights = torch.full((4, 2), 0.5)
        ids2, w2 = token_padding.zero_routing_weights_for_padding(ids, weights)
        # ids untouched everywhere; weights zeroed on padded rows only
        assert torch.equal(ids2, ids)
        assert w2[0].tolist() == [0.5, 0.5]
        assert float(w2[1:].abs().sum()) == 0.0
    finally:
        token_padding.set_padding_state(None)
    ids3, w3 = token_padding.zero_routing_weights_for_padding(ids, weights)
    assert ids3 is ids and w3 is weights


def test_zero_routing_weights_for_padding_dp_gather_rank_order(monkeypatch):
    # Rank-local mask [F, T]; fake DP group of 2 whose all_gather stacks
    # rank0 then rank1 masks -> gathered [F, T, T, F].
    provider = TokenPaddingState(local_padding_mask=torch.tensor([False, True]))
    token_padding.set_padding_state(provider)

    class _FakeGroup:
        world_size = 2

        @staticmethod
        def all_gather(t, dim=0):
            other = torch.tensor([1, 0], dtype=t.dtype)
            return torch.cat([t, other], dim=dim)

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dp_group", lambda: _FakeGroup()
    )
    try:
        ids = torch.arange(8, dtype=torch.int32).reshape(4, 2)
        weights = torch.full((4, 2), 0.5)
        ids2, w2 = token_padding.zero_routing_weights_for_padding(ids, weights)
        assert torch.equal(ids2, ids)
        # rows: rank0=[valid, pad], rank1=[pad, valid]
        assert w2[0].tolist() == [0.5, 0.5]
        assert float(w2[1].abs().sum()) == 0.0
        assert float(w2[2].abs().sum()) == 0.0
        assert w2[3].tolist() == [0.5, 0.5]
    finally:
        token_padding.set_padding_state(None)


def test_zero_routing_weights_for_padding_pcp_gather_rank_order(monkeypatch):
    # Rank-local mask [F, T]; fake PCP group of 2 whose all_gather stacks
    # rank0 then rank1 masks -> gathered [F, T, T, F].
    provider = TokenPaddingState(local_padding_mask=torch.tensor([False, True]))
    token_padding.set_padding_state(provider)

    class _FakeDpGroup:
        world_size = 1

    class _FakePcpGroup:
        world_size = 2

        @staticmethod
        def all_gather(t, dim=0):
            other = torch.tensor([1, 0], dtype=t.dtype)
            return torch.cat([t, other], dim=dim)

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dp_group", lambda: _FakeDpGroup()
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pcp_group", lambda: _FakePcpGroup()
    )
    try:
        ids = torch.arange(8, dtype=torch.int32).reshape(4, 2)
        weights = torch.full((4, 2), 0.5)
        ids2, w2 = token_padding.zero_routing_weights_for_padding(ids, weights)
        assert torch.equal(ids2, ids)
        # rows: rank0=[valid, pad], rank1=[pad, valid]
        assert w2[0].tolist() == [0.5, 0.5]
        assert float(w2[1].abs().sum()) == 0.0
        assert float(w2[2].abs().sum()) == 0.0
        assert w2[3].tolist() == [0.5, 0.5]
    finally:
        token_padding.set_padding_state(None)


def test_zero_routing_weights_for_padding_dp_and_pcp_gather_order(monkeypatch):
    provider = TokenPaddingState(local_padding_mask=torch.tensor([False]))
    token_padding.set_padding_state(provider)

    class _FakeDpGroup:
        world_size = 2

        @staticmethod
        def all_gather(t, dim=0):
            other = torch.tensor([1], dtype=t.dtype)
            return torch.cat([t, other], dim=dim)

    class _FakePcpGroup:
        world_size = 2

        @staticmethod
        def all_gather(t, dim=0):
            other = torch.tensor([1, 0], dtype=t.dtype)
            return torch.cat([t, other], dim=dim)

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dp_group", lambda: _FakeDpGroup()
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pcp_group", lambda: _FakePcpGroup()
    )
    try:
        ids = torch.arange(8, dtype=torch.int32).reshape(4, 2)
        weights = torch.full((4, 2), 0.5)
        ids2, w2 = token_padding.zero_routing_weights_for_padding(ids, weights)
        assert torch.equal(ids2, ids)
        assert w2[0].tolist() == [0.5, 0.5]
        assert float(w2[1].abs().sum()) == 0.0
        assert float(w2[2].abs().sum()) == 0.0
        assert w2[3].tolist() == [0.5, 0.5]
    finally:
        token_padding.set_padding_state(None)


def test_zero_routing_weights_for_padding_is_local_tensor(monkeypatch):
    provider = TokenPaddingState(local_padding_mask=torch.tensor([False, True]))
    token_padding.set_padding_state(provider)

    class _FakeGroup:
        world_size = 2

        @staticmethod
        def all_gather(t, dim=0):
            raise AssertionError(
                "all_gather should not be called when is_local_tensor=True"
            )

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dp_group", lambda: _FakeGroup()
    )
    try:
        ids = torch.arange(4, dtype=torch.int32).reshape(2, 2)
        weights = torch.full((2, 2), 0.5)
        ids2, w2 = token_padding.zero_routing_weights_for_padding(
            ids, weights, is_local_tensor=True
        )
        assert torch.equal(ids2, ids)
        assert w2[0].tolist() == [0.5, 0.5]
        assert float(w2[1].abs().sum()) == 0.0
    finally:
        token_padding.set_padding_state(None)


def test_mask_router_logits_for_padding():
    provider = TokenPaddingState(local_padding_mask=torch.zeros(4, dtype=torch.bool))
    token_padding.set_padding_state(provider)
    try:
        provider.update(1, 4)
        logits = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        masked = token_padding.mask_router_logits_for_padding(logits)
        assert torch.equal(masked[0], logits[0])
        assert torch.isnan(masked[1:]).all()
    finally:
        token_padding.set_padding_state(None)
    assert token_padding.mask_router_logits_for_padding(logits) is logits


def test_valid_tokens_allocation_is_gated_by_track_valid_tokens():
    state_off = TokenPaddingState.create(8, "cpu")
    assert state_off.local_valid_tokens is None
    token_padding.set_padding_state(state_off)
    try:
        state_off.update(3, 8)
        assert state_off.local_valid_tokens is None
        assert token_padding.get_valid_tokens_for_padding(is_local_tensor=True) is None
    finally:
        token_padding.set_padding_state(None)

    state_on = TokenPaddingState.create(8, "cpu", track_valid_tokens=True)
    assert state_on.local_valid_tokens is not None
    assert state_on.local_valid_tokens.shape == (1,)
    assert state_on.local_valid_tokens.dtype == torch.int32
    token_padding.set_padding_state(state_on)
    try:
        state_on.update(5, 8)
        count = token_padding.get_valid_tokens_for_padding(is_local_tensor=True)
        assert count is state_on.local_valid_tokens
        assert count is not None and count.tolist() == [5]
        state_on.update(0, 8)
        assert count.tolist() == [0]
    finally:
        token_padding.set_padding_state(None)
    assert token_padding.get_valid_tokens_for_padding(is_local_tensor=True) is None


def test_get_valid_tokens_for_padding_local_and_single_rank(monkeypatch):
    state = TokenPaddingState.create(8, "cpu", track_valid_tokens=True)
    state.update(5, 8)
    token_padding.set_padding_state(state)

    class _FailGroup:
        world_size = 2

        @staticmethod
        def all_gather(t, dim=0):
            raise AssertionError("all_gather must not be called")

    class _SingleRankGroup:
        world_size = 1

        @staticmethod
        def all_gather(t, dim=0):
            raise AssertionError("all_gather must not be called for world_size=1")

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dp_group", lambda: _FailGroup()
    )
    try:
        local = token_padding.get_valid_tokens_for_padding(is_local_tensor=True)
        assert local is state.local_valid_tokens
        assert local is not None and local.tolist() == [5]

        monkeypatch.setattr(
            "vllm.distributed.parallel_state.get_dp_group", lambda: _SingleRankGroup()
        )
        single = token_padding.get_valid_tokens_for_padding(is_local_tensor=False)
        assert single is state.local_valid_tokens
        assert single is not None and single.tolist() == [5]
    finally:
        token_padding.set_padding_state(None)
    assert token_padding.get_valid_tokens_for_padding() is None


def test_get_valid_tokens_for_padding_gathers_across_dp(monkeypatch):
    state = TokenPaddingState.create(8, "cpu", track_valid_tokens=True)
    state.update(3, 8)
    token_padding.set_padding_state(state)

    class _FakeDpGroup:
        world_size = 4

        @staticmethod
        def all_gather(t, dim=0):
            return torch.cat([t, torch.tensor([0, 8, 2], dtype=t.dtype)], dim=dim)

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dp_group", lambda: _FakeDpGroup()
    )
    try:
        gathered = token_padding.get_valid_tokens_for_padding(is_local_tensor=False)
        assert gathered is not None
        assert gathered.dtype == torch.int32
        assert gathered.tolist() == [3, 0, 8, 2]
    finally:
        token_padding.set_padding_state(None)


def test_get_valid_tokens_for_padding_returns_none_when_untracked():
    state = TokenPaddingState.create(8, "cpu", track_valid_tokens=False)
    token_padding.set_padding_state(state)
    try:
        assert token_padding.get_valid_tokens_for_padding() is None
    finally:
        token_padding.set_padding_state(None)
