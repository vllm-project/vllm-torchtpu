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


def test_update_writes_only_changed_counts(monkeypatch):
    state = TokenPaddingState(local_padding_mask=torch.zeros(8, dtype=torch.bool))
    writes = []
    real_copy = state.local_padding_mask.copy_

    def counting_copy(src):
        writes.append(src.clone())
        return real_copy(src)

    monkeypatch.setattr(state.local_padding_mask, "copy_", counting_copy)
    state.update(3, 6)
    state.update(3, 6)
    state.update(1, 4)
    state.update(3, 6)
    assert len(writes) == 3
    # No padded row: the same mask whatever the counts.
    state.update(6, 6)
    state.update(2, 2)
    state.update(4, 3)
    assert len(writes) == 4
    state.update(3, 6)
    assert len(writes) == 5
    assert (
        state.get_local_padding_mask(8).tolist()
        == [False] * 3 + [True] * 3 + [False] * 2
    )


def test_drop_padded_routes_sends_padded_rows_nowhere(monkeypatch):
    provider = TokenPaddingState(local_padding_mask=torch.zeros(4, dtype=torch.bool))
    token_padding.set_padding_state(provider)

    class _FakeGroup:
        world_size = 2

        @staticmethod
        def all_gather(t, dim=0):
            raise AssertionError("local rows need no all_gather")

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dp_group", lambda: _FakeGroup()
    )
    try:
        provider.update(2, 4)
        ids = torch.arange(8, dtype=torch.int32).reshape(4, 2)
        weights = torch.full((4, 2), 0.5, dtype=torch.bfloat16)
        ids2, w2 = token_padding.drop_padded_routes(ids, weights, is_local_tensor=True)
        assert ids2.dtype == ids.dtype and w2.dtype == weights.dtype
        assert ids2[:2].tolist() == ids[:2].tolist()
        assert w2[:2].tolist() == weights[:2].tolist()
        # Padded rows: unrouted, no weight.
        assert ids2[2:].tolist() == [[-1, -1], [-1, -1]]
        assert float(w2[2:].abs().sum()) == 0.0
    finally:
        token_padding.set_padding_state(None)
    ids3, w3 = token_padding.drop_padded_routes(ids, weights, is_local_tensor=True)
    assert ids3 is ids and w3 is weights


def test_drop_padded_routes_dp_gather_rank_order(monkeypatch):
    # Rank-local mask [F, T]; the other DP rank's is [T, F] -> gathered rows
    # [valid, pad, pad, valid].
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
        ids2, w2 = token_padding.drop_padded_routes(ids, weights, is_local_tensor=False)
        assert ids2.tolist() == [[0, 1], [-1, -1], [-1, -1], [6, 7]]
        assert w2.tolist() == [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [0.5, 0.5]]
    finally:
        token_padding.set_padding_state(None)


def test_runner_padding_update_takes_a_draft_forward_count():
    """Target forwards mark rows past the last plan's tokens; a draft forward
    passes its own valid count (capped at its rows)."""
    from types import SimpleNamespace

    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = TPUModelRunner.__new__(TPUModelRunner)
    state = TokenPaddingState(local_padding_mask=torch.zeros(8, dtype=torch.bool))
    runner._token_padding_state = state
    runner._last_sequence_layout_plan = SimpleNamespace(
        kind=SequenceLayoutKind.ALL, local_num_tokens=5
    )
    runner._token_padding_update(8)
    assert state.get_local_padding_mask(8).tolist() == [False] * 5 + [True] * 3
    runner._token_padding_update(6, 2)
    assert state.get_local_padding_mask(8).tolist() == (
        [False] * 2 + [True] * 4 + [False] * 2
    )
    runner._token_padding_update(4, 9)
    assert state.get_local_padding_mask(8).sum().item() == 0
