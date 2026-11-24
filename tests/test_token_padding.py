from types import SimpleNamespace

import pytest
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


@pytest.mark.parametrize(
    "valid,padded", [(3, 6), (2, 2), (0, 6), (0, 8), (5, 8), (3, 6)]
)
def test_on_device_mask_matches_host_write(valid, padded):
    host = TokenPaddingState(local_padding_mask=torch.zeros(8, dtype=torch.bool))
    device = TokenPaddingState(
        local_padding_mask=torch.ones(8, dtype=torch.bool), on_device=True
    )
    host.update(valid, padded)
    device.update(valid, padded)
    assert torch.equal(device.local_padding_mask, host.local_padding_mask)


def test_on_device_mask_is_built_once_per_pair():
    state = TokenPaddingState(
        local_padding_mask=torch.zeros(8, dtype=torch.bool), on_device=True
    )
    state.update(3, 6)
    first = state.local_padding_mask
    state.update(0, 8)
    assert state.get_local_padding_mask(8).tolist() == [True] * 8
    state.update(3, 6)
    assert state.local_padding_mask is first
    assert sorted(state._pair_masks) == [(0, 8), (3, 6)]
    assert first.tolist() == [False] * 3 + [True] * 3 + [False] * 2


def test_on_device_mask_is_never_written_after_use():
    state = TokenPaddingState(
        local_padding_mask=torch.zeros(8, dtype=torch.bool), on_device=True
    )
    state.update(3, 6)
    in_flight = state.get_local_padding_mask(8)
    state.update(0, 8)
    assert in_flight.tolist() == [False] * 3 + [True] * 3 + [False] * 2


def test_on_device_mask_stops_caching_past_its_bound(monkeypatch):
    monkeypatch.setattr(token_padding, "_MAX_PAIR_MASKS", 1)
    state = TokenPaddingState(
        local_padding_mask=torch.zeros(8, dtype=torch.bool), on_device=True
    )
    for valid, padded in [(3, 6), (0, 8), (5, 8)]:
        state.update(valid, padded)
    assert list(state._pair_masks) == [(3, 6)]
    assert state.get_local_padding_mask(8).tolist() == [False] * 5 + [True] * 3


def test_create_reads_the_on_device_switch(monkeypatch):
    monkeypatch.setenv("TPU_MOE_PADDING_MASK_ON_DEVICE", "1")
    assert TokenPaddingState.create(8, torch.device("cpu")).on_device
    monkeypatch.delenv("TPU_MOE_PADDING_MASK_ON_DEVICE")
    assert not TokenPaddingState.create(8, torch.device("cpu")).on_device


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


def test_local_padding_rows_reads_runner_mask_for_matching_rows():
    provider = TokenPaddingState(local_padding_mask=torch.zeros(8, dtype=torch.bool))
    token_padding.set_padding_state(provider)
    try:
        provider.update(3, 6)
        assert token_padding.local_padding_rows(6).tolist() == [
            False,
            False,
            False,
            True,
            True,
            True,
        ]
        assert token_padding.local_padding_rows(9) is None
    finally:
        token_padding.set_padding_state(None)
    assert token_padding.local_padding_rows(6) is None


@pytest.mark.parametrize(
    "stage_mask,expected",
    [
        (torch.tensor([False, True, True, True]), [False, True, True, True]),
        (torch.tensor([False, True]), None),
    ],
)
def test_local_padding_rows_prefers_the_stage_mask(monkeypatch, stage_mask, expected):
    provider = TokenPaddingState(local_padding_mask=torch.zeros(8, dtype=torch.bool))
    token_padding.set_padding_state(provider)
    context = SimpleNamespace(additional_kwargs={"token_padding_mask": stage_mask})
    monkeypatch.setattr(token_padding, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(token_padding, "get_forward_context", lambda: context)
    try:
        provider.update(3, 4)
        rows = token_padding.local_padding_rows(4)
        assert (rows if rows is None else rows.tolist()) == expected
    finally:
        token_padding.set_padding_state(None)
