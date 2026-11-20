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


def test_update_skips_the_copy_only_when_counts_repeat():
    state = TokenPaddingState(local_padding_mask=torch.zeros(8, dtype=torch.bool))
    copies = []
    real_copy = state.local_padding_mask.copy_

    def counting_copy(src):
        copies.append(src.clone())
        return real_copy(src)

    state.local_padding_mask.copy_ = counting_copy
    # decode steps, an MTP rejection that shrinks the valid count, padding
    # bucket growth, then back to an earlier count pair
    steps = [(3, 4), (3, 4), (2, 4), (2, 4), (2, 8), (3, 4)]
    for valid, padded in steps:
        state.update(valid, padded)
        expected = torch.zeros(8, dtype=torch.bool)
        expected[valid:padded] = True
        assert torch.equal(state.local_padding_mask, expected)
    assert len(copies) == 4


def test_update_writes_the_zero_mask_once_across_unpadded_steps():
    """An unpadded step after another unpadded one leaves the mask as it is;
    the padded step after them still writes its own suffix."""
    state = TokenPaddingState.create(8, device="cpu")
    copies = []
    real_copy = state.local_padding_mask.copy_

    def counting_copy(src):
        copies.append(src.clone())
        return real_copy(src)

    state.local_padding_mask.copy_ = counting_copy
    writes = []
    for valid, padded in [(3, 6), (4, 4), (5, 5), (2, 6), (6, 6)]:
        before = len(copies)
        state.update(valid, padded)
        writes.append(len(copies) - before)
        expected = torch.zeros(8, dtype=torch.bool)
        expected[valid:padded] = True
        assert torch.equal(state.local_padding_mask, expected), (valid, padded)
    # (5, 5) follows the unpadded (4, 4), so it writes nothing.
    assert writes == [1, 1, 0, 1, 1]


def test_the_valid_count_is_kept_for_the_step():
    state = TokenPaddingState.create(8, device="cpu")
    assert token_padding.get_local_valid_tokens() is None
    with token_padding.use_padding_state(state):
        state.update(3, 8)
        assert token_padding.get_local_valid_tokens() is state.local_valid_tokens
        assert token_padding.get_local_valid_tokens().tolist() == [3]
        state.update(5, 8)
        assert state.get_local_valid_tokens().tolist() == [5]


def test_use_padding_state_binds_and_restores():
    outer = TokenPaddingState.create(4, device="cpu")
    inner = TokenPaddingState.create(4, device="cpu")
    token_padding.set_padding_state(outer)
    try:
        with token_padding.use_padding_state(inner):
            assert token_padding._padding_state is inner
            with token_padding.use_padding_state(None):
                assert token_padding._padding_state is None
            assert token_padding._padding_state is inner
        assert token_padding._padding_state is outer
        try:
            with token_padding.use_padding_state(inner):
                raise RuntimeError("loop graph failed")
        except RuntimeError:
            pass
        assert token_padding._padding_state is outer
    finally:
        token_padding.set_padding_state(None)


def test_retain_padding_graph_input_reads_the_mask_and_keeps_the_values():
    out = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    assert token_padding.retain_padding_graph_input(out) is out
    state = TokenPaddingState.create(4, device="cpu")
    with token_padding.use_padding_state(state):
        state.update(1, 4)
        kept = token_padding.retain_padding_graph_input(out)
    assert kept is not out
    assert torch.equal(kept, out)


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
