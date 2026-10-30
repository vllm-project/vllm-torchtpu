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


class _CountingMask:
    """A CPU mask that counts the uploads update() issues."""

    def __init__(self, size: int):
        self.tensor = torch.zeros(size, dtype=torch.bool)
        self.copies = 0

    def install(self, state: TokenPaddingState) -> None:
        tensor = self.tensor

        def copy_(src):
            self.copies += 1
            return torch.Tensor.copy_(tensor, src)

        tensor.copy_ = copy_
        state.local_padding_mask = tensor


def _reference_mask(size: int, valid: int, padded: int) -> list[bool]:
    host = torch.zeros(size, dtype=torch.bool)
    host[valid:padded] = True
    return host.tolist()


def test_skip_unchanged_keeps_every_mask_and_skips_repeats(monkeypatch):
    """Flag on: after every update the mask equals a fresh write, and only an
    update that changes the padding rows uploads."""
    monkeypatch.setenv("TPU_TOKEN_PADDING_SKIP_UNCHANGED", "1")
    size = 16
    seq = [
        (4, 4),
        (4, 4),
        (4, 4),  # steady b1 decode: no padding rows
        (3, 4),
        (3, 4),  # one padding row
        (3, 8),
        (3, 8),
        (2, 8),  # same end, new start
        (0, 8),
        (0, 8),  # warmup: all padding
        (8, 8),
        (16, 16),
        (9, 2),  # empty ranges: all the same mask
        (12, 40),
        (12, 16),  # end past the mask: clamped like a slice
        (-4, 16),
        (12, 16),  # negative start indexes from the end
        (4, 4),
    ]
    counting = _CountingMask(size)
    state = TokenPaddingState(local_padding_mask=torch.zeros(size, dtype=torch.bool))
    counting.install(state)
    expected_copies = 0
    prev = None
    for valid, padded in seq:
        ref = _reference_mask(size, valid, padded)
        if ref != prev:
            expected_copies += 1
        state.update(valid, padded)
        assert state.local_padding_mask.tolist() == ref, (valid, padded)
        prev = ref
    # The first write always uploads; afterwards exactly the mask changes do.
    assert counting.copies == expected_copies
    assert counting.copies < len(seq)


def test_skip_unchanged_off_uploads_every_time(monkeypatch):
    monkeypatch.setenv("TPU_TOKEN_PADDING_SKIP_UNCHANGED", "0")  # on by default
    counting = _CountingMask(8)
    state = TokenPaddingState(local_padding_mask=torch.zeros(8, dtype=torch.bool))
    counting.install(state)
    for _ in range(3):
        state.update(4, 4)
    assert counting.copies == 3


def test_skip_unchanged_first_write_always_uploads(monkeypatch):
    """The device mask's initial contents are not trusted, even if they would
    match: a fresh state must write once."""
    monkeypatch.setenv("TPU_TOKEN_PADDING_SKIP_UNCHANGED", "1")
    counting = _CountingMask(8)
    state = TokenPaddingState(local_padding_mask=torch.zeros(8, dtype=torch.bool))
    counting.install(state)
    counting.tensor.fill_(True)  # stale contents a skip would leave behind
    state.update(8, 8)
    assert counting.copies == 1
    assert state.local_padding_mask.sum().item() == 0


def test_skip_unchanged_catches_a_rows_change_with_the_same_count(monkeypatch):
    """Positive control: (2, 4) and (4, 6) pad the same number of rows; a cache
    keyed on the count instead of the rows would keep the stale mask."""
    monkeypatch.setenv("TPU_TOKEN_PADDING_SKIP_UNCHANGED", "1")
    state = TokenPaddingState(local_padding_mask=torch.zeros(8, dtype=torch.bool))
    state.update(2, 4)
    state.update(4, 6)
    assert state.local_padding_mask.tolist() == _reference_mask(8, 4, 6)
