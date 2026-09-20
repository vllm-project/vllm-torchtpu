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

import contextlib
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_torchtpu.layers.core.sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL, SequenceLayoutDescriptor,
    SequenceLayoutKind, SequenceLayoutPlan)
from vllm_torchtpu.spec_decode import utils as spec_decode_utils
from vllm_torchtpu.spec_decode.eagle3 import (DraftChunkInputs, Eagle3Proposer,
                                              _force_draft_tp1,
                                              _maybe_pad_dim0)


def _make_proposer(draft_tp: int | None = 1,
                   target_tp: int = 1,
                   method: str = "eagle3") -> Eagle3Proposer:
    speculative_config = SimpleNamespace(draft_tensor_parallel_size=draft_tp,
                                         method=method,
                                         draft_model_config=None)
    parallel_config = SimpleNamespace(tensor_parallel_size=target_tp)
    vllm_config = SimpleNamespace(speculative_config=speculative_config,
                                  parallel_config=parallel_config)
    return Eagle3Proposer(runner=mock.MagicMock(), vllm_config=vllm_config)


def _make_vocab_models():
    target_embed = torch.nn.Embedding(3, 2)
    target_embed.org_vocab_size = 3
    draft_embed = torch.nn.Embedding(3, 2)
    draft_embed.org_vocab_size = 3
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed),
        lm_head=torch.nn.Linear(2, 3, bias=False),
    )
    draft = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=draft_embed),
        lm_head=torch.nn.Linear(2, 3, bias=False),
        logits_processor=SimpleNamespace(
            _gather_logits=lambda logits: logits.clone()),
        has_own_embed_tokens=False,
        has_own_lm_head=False,
    )
    return draft, target


def _fake_gather_sharded_weight(module):
    weight = module.weight.detach().clone()
    return weight, weight.shape[0], weight.shape[1], weight.device


@pytest.mark.parametrize(
    "data, target_len, expected",
    [
        # 1-D: pad up with zeros.
        ([1, 2, 3], 5, [1, 2, 3, 0, 0]),
        # 2-D: pad rows, leave columns intact.
        ([[1, 2], [3, 4]], 4, [[1, 2], [3, 4], [0, 0], [0, 0]]),
        # noop: target equals current dim0.
        ([1, 2, 3], 3, [1, 2, 3]),
        # noop: target smaller than current dim0.
        ([1, 2, 3], 2, [1, 2, 3]),
    ],
)
def test_maybe_pad_dim0(device, data, target_len, expected):
    out = _maybe_pad_dim0(torch.tensor(data, device=device), target_len)
    assert torch.equal(out, torch.tensor(expected, device=device))


def test_maybe_pad_dim0_rejects_3d(device):
    with pytest.raises(ValueError):
        _maybe_pad_dim0(torch.zeros((2, 2, 2), device=device), 4)


def test_force_draft_tp1_overrides_and_restores():
    fake_tp = SimpleNamespace(world_size=8, rank_in_group=3)
    with mock.patch("vllm.distributed.parallel_state.get_tp_group",
                    return_value=fake_tp):
        with _force_draft_tp1():
            assert fake_tp.world_size == 1
            assert fake_tp.rank_in_group == 0
        # Restored on normal exit.
        assert fake_tp.world_size == 8
        assert fake_tp.rank_in_group == 3


def test_force_draft_tp1_restores_on_exception():
    fake_tp = SimpleNamespace(world_size=4, rank_in_group=2)
    with mock.patch("vllm.distributed.parallel_state.get_tp_group",
                    return_value=fake_tp):
        with pytest.raises(RuntimeError), _force_draft_tp1():
            raise RuntimeError("boom")
        assert fake_tp.world_size == 4
        assert fake_tp.rank_in_group == 2


def test_draft_tp_defaults_to_target_tp():
    # vLLM's _verify_and_get_draft_tp already resolves an unset eagle3 draft
    # tp to target_tp; Eagle3Proposer sets it explicitly too (defensive +
    # self-documenting).
    proposer = _make_proposer(draft_tp=None, target_tp=8)
    assert proposer.speculative_config.draft_tensor_parallel_size == 8
    assert proposer._draft_replicated is False


@pytest.mark.parametrize("draft_tp,target_tp", [(1, 1), (1, 8), (8, 8)])
def test_load_model_vocab_weights_follow_tp_layout(draft_tp, target_tp,
                                                   monkeypatch):
    proposer = _make_proposer(draft_tp=draft_tp,
                              target_tp=target_tp,
                              method="mtp")
    draft, target = _make_vocab_models()
    draft_head = draft.lm_head
    original_gather_logits = draft.logits_processor._gather_logits

    def load_draft_model():
        proposer.draft_model = draft

    proposer._load_draft_model = load_draft_model
    monkeypatch.setattr(spec_decode_utils, "gather_sharded_weight",
                        _fake_gather_sharded_weight)

    with mock.patch(
            "vllm_torchtpu.spec_decode.eagle3.get_layers_from_vllm_config",
            return_value={}):
        proposer.load_model(target)

    if draft_tp == target_tp:
        assert draft.model.embed_tokens is target.model.embed_tokens
        assert draft.lm_head is target.lm_head
    else:
        assert draft.model.embed_tokens is not target.model.embed_tokens
        assert torch.equal(draft.model.embed_tokens.weight,
                           target.model.embed_tokens.weight)
        assert draft.lm_head is draft_head

    if draft_tp == 1:
        logits = torch.randn(2, 3)
        assert draft.logits_processor._gather_logits(logits) is logits
    else:
        assert draft.logits_processor._gather_logits is original_gather_logits


@pytest.mark.parametrize("draft_tp", [2, 4])
def test_draft_tp_invalid_raises(draft_tp):
    # Only replicated (1) or fully sharded (== target tp) are supported.
    with pytest.raises(ValueError):
        _make_proposer(draft_tp=draft_tp, target_tp=8)


class TestDpLockstepSharded:
    """Which drafts must have their collective trace replayed by idle DP ranks.

    A draft emits cross-DP collectives either because it is TP-sharded, or
    because it carries MoE whose experts are expert-parallel across the DP
    group — the latter holds even at draft_tp=1, since forcing the draft to
    tp=1 does not make its MoE rank-local.
    """

    def _proposer(self, *, target_tp, lockstep, has_moe, loaded=True):
        proposer = _make_proposer(draft_tp=None, target_tp=target_tp)
        proposer.runner._dp_lockstep_enabled = lambda: lockstep
        proposer.draft_model = object() if loaded else None
        return proposer, mock.patch(
            "vllm.model_executor.models.interfaces.is_mixture_of_experts",
            return_value=has_moe)

    def test_replicated_moe_draft_under_ep_dp_pairs(self):
        # TP=1 + DP>1 + EP: the MoE draft still needs idle-rank pairing.
        proposer, moe_patch = self._proposer(target_tp=1,
                                             lockstep=True,
                                             has_moe=True)
        with moe_patch:
            assert proposer._dp_lockstep_sharded() is True

    def test_replicated_dense_draft_does_not_pair(self):
        # A tp=1 dense draft is entirely rank-local: nothing to pair with.
        proposer, moe_patch = self._proposer(target_tp=1,
                                             lockstep=True,
                                             has_moe=False)
        with moe_patch:
            assert proposer._dp_lockstep_sharded() is False

    def test_sharded_draft_pairs_regardless_of_moe(self):
        proposer, moe_patch = self._proposer(target_tp=8,
                                             lockstep=True,
                                             has_moe=False)
        with moe_patch:
            assert proposer._dp_lockstep_sharded() is True

    def test_no_lockstep_never_pairs(self):
        # DP=1 or EP off: no cross-DP collective program to match.
        proposer, moe_patch = self._proposer(target_tp=1,
                                             lockstep=False,
                                             has_moe=True)
        with moe_patch:
            assert proposer._dp_lockstep_sharded() is False

    def test_moe_detection_is_cached(self):
        proposer, moe_patch = self._proposer(target_tp=1,
                                             lockstep=True,
                                             has_moe=True)
        with moe_patch as m:
            assert proposer._dp_lockstep_sharded() is True
            assert proposer._dp_lockstep_sharded() is True
            assert m.call_count == 1

    def test_unloaded_draft_reports_no_moe(self):
        # Queried before load_model: must not raise, and must not cache a
        # False that would outlive the load.
        proposer, moe_patch = self._proposer(target_tp=1,
                                             lockstep=True,
                                             has_moe=True,
                                             loaded=False)
        with moe_patch:
            assert proposer._dp_lockstep_sharded() is False
            proposer.draft_model = object()
            assert proposer._dp_lockstep_sharded() is True


def _make_chunk(*,
                input_ids,
                position_ids,
                query_start_loc_np,
                start_index,
                num_reqs,
                hidden=8,
                device,
                padded_tokens=None,
                attn_ctx=None,
                hidden_states=None,
                draft_lengths=None,
                sequence_layout_plan=None):
    """Build a DraftChunkInputs for tests. aux/attn_ctx are only needed by
    paths that consume them; _prepare_draft_inputs does not touch attn_ctx, so
    it defaults to an inert placeholder. Pass a real attn_ctx for the propose
    loop, which reads attn_ctx.use_max_model_len."""
    padded_tokens = padded_tokens or input_ids.shape[0]
    chunk = DraftChunkInputs(
        input_ids=input_ids,
        position_ids=position_ids,
        query_start_loc_np=query_start_loc_np,
        attn_ctx=attn_ctx,
        start_index=start_index,
        num_reqs=num_reqs,
        # Eagle3 always exposes exactly 3 aux hidden states (the draft combine
        # consumes 3 * hidden); the compiled _draft_combine_hidden_states wrapper
        # takes them positionally, so provide 3.
        aux_hidden_states=[
            torch.zeros((padded_tokens, hidden), device=device)
            for _ in range(3)
        ],
        hidden_states=hidden_states,
        draft_lengths=draft_lengths,
    )
    if sequence_layout_plan is not None:
        chunk.sequence_layout_plan = sequence_layout_plan
    return chunk


def _rank0_pcp_plan() -> SequenceLayoutPlan:
    return SequenceLayoutPlan(
        descriptor=SequenceLayoutDescriptor(
            kind=SequenceLayoutKind.PARTIAL,
            protocol=PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
        ),
        token_slice=slice(0, 3),
        global_num_tokens=5,
        global_padded_num_tokens=6,
        local_num_tokens=3,
        local_padded_num_tokens=3,
        _packed_to_request_major_token_indices=np.array([0, 2, 4, 1, 3, -1],
                                                        dtype=np.int64),
        _request_major_to_packed_token_indices=np.array([0, 3, 1, 4, 2],
                                                        dtype=np.int64),
    )


class _CarryAggregationAfterGatherPlan:

    def __init__(self, events):
        self._events = events

    def aggregate_request_aligned_tensor(self, tensor, owner_mask):
        if "gather" not in self._events:
            raise AssertionError("layout aggregation ran before carry gather")
        self._events.append("aggregate")
        assert owner_mask.tolist() == [False, True]
        return tensor


def test_prepare_draft_inputs(device):
    proposer = _make_proposer(draft_tp=1)
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r0", "r1"]),
        device=device,
        requests={},
        num_tokens_paddings=[16, 32, 64, 128],
        _dp_lockstep_enabled=lambda: False,
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={})

    # Single chunk covering the whole batch (start_index=0). Chunk-local
    # padded token buffer is [0..10]; req lengths are 4 and 7.
    chunk = _make_chunk(
        input_ids=torch.arange(11, dtype=torch.int32, device=device),
        position_ids=torch.arange(11, dtype=torch.int32, device=device),
        query_start_loc_np=np.array([0, 4, 11], dtype=np.int32),
        start_index=0,
        num_reqs=2,
        device=device,
    )

    (draft_input_ids, _positions, last_token_indices,
     num_rejected_np) = proposer._prepare_draft_inputs(
         chunk,
         sampled_token_ids=[[101], [202]],
         discard_sampled_tokens_req_indices=[],
         num_rejected_tokens_np=np.array([1, 3], dtype=np.int32),
         scheduler_output=scheduler_output,
     )

    # Rejection counts clamp to per-request lengths-1 [3, 6] -> [1, 3].
    assert np.array_equal(num_rejected_np, np.array([1, 3]))
    # Accepted-prefix ends: qsl[1:]-1-rejected = [3,10]-[1,3] = [2, 7].
    # The returned gather index is bucket-padded (tail repeats the last real
    # entry); check the real [:num_reqs] prefix.
    assert torch.equal(last_token_indices.cpu()[:2],
                       torch.tensor([2, 7], dtype=torch.int64))

    # input_ids [0..10]; left shift -> [1,2,...,10,10]; then patch the last
    # sampled token at the accepted-prefix slots [2, 7] with [101, 202].
    expected_ids = torch.tensor([1, 2, 101, 4, 5, 6, 7, 202, 9, 10, 10],
                                dtype=torch.int32,
                                device=device)
    assert torch.equal(draft_input_ids, expected_ids)


def test_prepare_draft_inputs_chunk_offset(device):
    """A non-zero start_index must slice the batch-level sampled tokens and
    rejection counts by the chunk's offset."""
    proposer = _make_proposer(draft_tp=1)
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r0", "r1", "r2", "r3"]),
        device=device,
        requests={},
        num_tokens_paddings=[16, 32, 64, 128],
        _dp_lockstep_enabled=lambda: False,
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={})

    # Second chunk: batch requests 2 and 3, lengths 3 and 2 (chunk-local
    # padded buffer [0..4]).
    chunk = _make_chunk(
        input_ids=torch.arange(5, dtype=torch.int32, device=device),
        position_ids=torch.arange(5, dtype=torch.int32, device=device),
        query_start_loc_np=np.array([0, 3, 5], dtype=np.int32),
        start_index=2,
        num_reqs=2,
        device=device,
    )

    (draft_input_ids, _positions, last_token_indices,
     num_rejected_np) = proposer._prepare_draft_inputs(
         chunk,
         # Batch-level arrays of length 4; only [2:4] applies to this chunk.
         sampled_token_ids=[[0], [0], [201], [202]],
         discard_sampled_tokens_req_indices=[],
         num_rejected_tokens_np=np.array([9, 9, 1, 0], dtype=np.int32),
         scheduler_output=scheduler_output,
     )

    # Chunk slice of rejections [1, 0], clamped to lengths-1 [2, 1] -> [1, 0].
    assert np.array_equal(num_rejected_np, np.array([1, 0]))
    # qsl[1:]-1-rejected = [2,4]-[1,0] = [1, 4]. Returned gather index is
    # bucket-padded; check the real [:num_reqs] prefix.
    assert torch.equal(last_token_indices.cpu()[:2],
                       torch.tensor([1, 4], dtype=torch.int64))
    # [0..4] left shift -> [1,2,3,4,4]; patch [1,4] with batch tokens [201,202].
    expected_ids = torch.tensor([1, 201, 3, 4, 202],
                                dtype=torch.int32,
                                device=device)
    assert torch.equal(draft_input_ids, expected_ids)


@pytest.mark.parametrize("seed_source",
                         ["host", "device_seed", "next_tokens_device"])
def test_prepare_draft_inputs_pcp_mtp_seeds_request_major_then_localizes(
        device, seed_source):
    """The seed scatter stays in request-major coordinates; only the final
    draft input and gather values cross the chunk-local PCP layout boundary."""
    proposer = _make_proposer(draft_tp=1, method="mtp")
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r0", "r1"]),
        device=device,
        requests={},
        num_tokens_paddings=[16, 32],
        _dp_lockstep_enabled=lambda: False,
    )
    plan = _rank0_pcp_plan()
    request_major = torch.tensor([10, 11, 20, 21, 22, 0],
                                 dtype=torch.int32,
                                 device=device)
    local_positions = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    chunk = _make_chunk(
        input_ids=request_major,
        position_ids=local_positions,
        query_start_loc_np=np.array([0, 2, 5], dtype=np.int32),
        start_index=0,
        num_reqs=2,
        device=device,
        attn_ctx=SimpleNamespace(query_start_loc=torch.tensor(
            [0, 2, 5], dtype=torch.int32, device=device)),
        sequence_layout_plan=plan,
    )

    kwargs = {}
    if seed_source == "device_seed":
        kwargs["device_seed"] = torch.tensor([101, 202],
                                             dtype=torch.int32,
                                             device=device)
    elif seed_source == "next_tokens_device":
        kwargs["next_tokens_device"] = torch.tensor([[101, -1], [202, -1]],
                                                    dtype=torch.int32,
                                                    device=device)
    draft_input_ids, positions, local_gather_indices, rejected = (
        proposer._prepare_draft_inputs(
            chunk,
            sampled_token_ids=([[101], [202]]
                               if seed_source == "host" else None),
            discard_sampled_tokens_req_indices=[],
            num_rejected_tokens_np=None,
            scheduler_output=SimpleNamespace(num_scheduled_tokens={},
                                             scheduled_spec_decode_tokens={}),
            **kwargs,
        ))

    # Full request-major shift/scatter is
    # [10,11,20,21,22,0] -> [11,101,21,22,202,0]. Rank 0 then owns
    # request-major rows [0,2,4]. Shifting the already-local target tensor or
    # scattering with local gather values produces a different literal result.
    torch.testing.assert_close(
        draft_input_ids.cpu(),
        torch.tensor([11, 21, 202], dtype=torch.int32),
    )
    assert positions is local_positions
    assert local_gather_indices.cpu()[:2].tolist() == [-1, 2]
    assert local_gather_indices.cpu()[2:].eq(-1).all()
    if isinstance(rejected, torch.Tensor):
        assert rejected.cpu().tolist() == [0, 0]
    else:
        assert np.array_equal(rejected, np.zeros(2, dtype=np.int32))


def test_prepare_draft_inputs_all_layout_keeps_device_seed_gathers(
        device, monkeypatch):
    import vllm_torchtpu.spec_decode.eagle3 as e3

    proposer = _make_proposer(draft_tp=1, method="mtp")
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r0", "r1"]),
        device=device,
        requests={},
        num_tokens_paddings=[16, 32],
        _dp_lockstep_enabled=lambda: False,
    )
    chunk = _make_chunk(
        input_ids=torch.tensor([10, 11, 20, 21, 22, 0],
                               dtype=torch.int32,
                               device=device),
        position_ids=torch.tensor([0, 1, 0, 1, 2, 0],
                                  dtype=torch.int32,
                                  device=device),
        query_start_loc_np=np.array([0, 2, 5], dtype=np.int32),
        start_index=0,
        num_reqs=2,
        device=device,
    )

    device_seed = torch.tensor([101, 202], dtype=torch.int32, device=device)

    def reject_device_padding(*_args, **_kwargs):
        raise AssertionError("device_seed gather padding must stay on host")

    monkeypatch.setattr(e3, "_maybe_pad_dim0", reject_device_padding)

    draft_input_ids, positions, gather_indices, _ = (
        proposer._prepare_draft_inputs(
            chunk,
            sampled_token_ids=None,
            discard_sampled_tokens_req_indices=[],
            num_rejected_tokens_np=None,
            scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
            device_seed=device_seed,
        ))

    torch.testing.assert_close(
        draft_input_ids.cpu(),
        torch.tensor([11, 101, 21, 22, 202, 0], dtype=torch.int32),
    )
    assert positions is chunk.position_ids
    assert gather_indices.cpu()[:2].tolist() == [1, 4]
    assert gather_indices.cpu()[2:].eq(0).all()


def test_propose_delegates_carry_aggregation_after_gather():
    device = torch.device("cpu")
    proposer = _make_proposer(draft_tp=1, method="mtp")
    proposer.speculative_config.num_speculative_tokens = 1
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=2, req_ids=["r0", "r1"]),
        device=device,
        requests={},
        num_tokens_paddings=[16, 32],
        uses_mrope=False,
        _dp_lockstep_enabled=lambda: False,
    )
    events = []
    chunk = _make_chunk(
        input_ids=torch.zeros(3, dtype=torch.int32, device=device),
        position_ids=torch.zeros(3, dtype=torch.int32, device=device),
        query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
        start_index=0,
        num_reqs=2,
        hidden=1,
        device=device,
        hidden_states=torch.zeros((3, 1), device=device),
        attn_ctx=SimpleNamespace(use_max_model_len=True),
        sequence_layout_plan=_CarryAggregationAfterGatherPlan(events),
    )
    proposer.draft_chunks = [chunk]
    proposer.draft_model = SimpleNamespace(model=SimpleNamespace(
        use_aux_hidden_state=False))
    proposer._prepare_draft_inputs = lambda *_args, **_kwargs: (
        torch.zeros(3, dtype=torch.int32, device=device),
        chunk.position_ids,
        torch.tensor([-1, 2], dtype=torch.int64, device=device),
        np.zeros(2, dtype=np.int32),
    )
    proposer._forward_draft = lambda **_kwargs: (
        torch.tensor([[99.0], [7.0], [20.0]], device=device),
        torch.zeros((3, 1), device=device),
    )

    def record_gather(hidden, positions, last_hidden, gather_indices):
        events.append("gather")
        assert gather_indices.cpu().tolist() == [0, 2]
        return (hidden[:2], positions[:2], last_hidden[gather_indices])

    proposer._draft_gather_carries = record_gather

    def record_token(last_hidden):
        events.append("propose_token")
        return last_hidden[:, 0].to(torch.int32)

    proposer._draft_propose_token = record_token
    proposer.propose(
        sampled_token_ids=[[101], [202]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
    )

    assert events == ["gather", "aggregate", "propose_token"]


def test_prepare_draft_inputs_async_device(device):
    """Async device path derives the seed token, the seed *position*
    (last_token_indices) and the rejected count from the on-device rejection
    output — matching the host path fed the equivalent num_rejected."""
    proposer = _make_proposer(draft_tp=1)
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r0", "r1"]),
        device=device,
        requests={},
        num_tokens_paddings=[16, 32, 64, 128],
        _dp_lockstep_enabled=lambda: False,
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={})

    def make_chunk():
        # K=3 verify spans of 4 tokens each (qsl [0, 4, 8]); input_ids 0..7.
        # The async path reads num_draft on-device from chunk.draft_lengths and
        # qsl_end from chunk.attn_ctx.query_start_loc (mirroring production), so
        # both must be supplied. K=3 drafts per req here.
        return _make_chunk(
            input_ids=torch.arange(8, dtype=torch.int32, device=device),
            position_ids=torch.arange(8, dtype=torch.int32, device=device),
            query_start_loc_np=np.array([0, 4, 8], dtype=np.int32),
            start_index=0,
            num_reqs=2,
            device=device,
            draft_lengths=torch.tensor([3, 3],
                                       dtype=torch.int32,
                                       device=device),
            attn_ctx=SimpleNamespace(query_start_loc=torch.tensor(
                [0, 4, 8], dtype=torch.int32, device=device)),
        )

    # Device rejection output: req0 has 3 valid (bonus 201, 1 draft rejected),
    # req1 has 1 valid (bonus 202, 3 rejected) -> num_rejected = orig - num_valid
    # = [4, 4] - [3, 1] = [1, 3]; seeds (bonus) = [201, 202].
    next_tokens = torch.tensor([[91, 92, 201, -1], [202, -1, -1, -1]],
                               dtype=torch.int32,
                               device=device)
    dev_ids, _, dev_last, dev_rej = proposer._prepare_draft_inputs(
        make_chunk(),
        sampled_token_ids=None,  # async path ignores the host count/list
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=scheduler_output,
        next_tokens_device=next_tokens,
    )

    # Host path fed the equivalent num_rejected=[1,3] and bonus seeds.
    host_ids, _, host_last, host_rej = proposer._prepare_draft_inputs(
        make_chunk(),
        sampled_token_ids=[[201], [202]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=np.array([1, 3], dtype=np.int32),
        scheduler_output=scheduler_output,
    )

    assert torch.equal(dev_ids, host_ids)
    # Gather-index real prefix matches (seed positions [2, 4]); the padded tail
    # differs by pad convention (async zero-pads, sync repeats the last entry)
    # and is sliced off by propose(), so compare only [:num_reqs].
    assert torch.equal(dev_last.cpu()[:2], host_last.cpu()[:2])
    assert dev_rej.cpu().tolist() == host_rej.tolist() == [1, 3]


@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 8])
@pytest.mark.parametrize("chunk_sizes", [[2], [2, 2], [3, 1]])
@pytest.mark.parametrize("return_device", [False, True])
def test_propose(num_speculative_tokens, chunk_sizes, return_device, device):
    # Each variant exercises the @torch.compile draft wrappers at distinct
    # shapes; across the full parametrize sweep they otherwise accumulate past
    # dynamo's recompile_limit (8) in one process (FailOnRecompileLimitHit).
    # Reset dynamo so each variant starts fresh. (Production hits only a bounded
    # bucketed set after warmup, so this is a test-only concern.)
    import torch._dynamo
    torch._dynamo.reset()
    hidden_size = 8
    vocab_size = 128
    num_reqs = sum(chunk_sizes)
    # Distinct base token per request so cross-chunk ordering is checkable.
    base_token_ids = [40 + 10 * i for i in range(num_reqs)]

    proposer = _make_proposer(draft_tp=1)
    proposer.speculative_config.num_speculative_tokens = num_speculative_tokens
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=num_reqs),
        max_num_reqs=16,
        num_reqs_max_model_len=16,
        num_reqs_most_model_len=16,
        num_tokens_paddings=[16, 32, 64, 128],
        device=device,
        uses_mrope=False,
        _dp_lockstep_enabled=lambda: False,
    )

    chunks = []
    start = 0
    for nr in chunk_sizes:
        chunks.append(
            _make_chunk(
                input_ids=torch.zeros(16, dtype=torch.int32, device=device),
                position_ids=torch.zeros(16, dtype=torch.int32, device=device),
                query_start_loc_np=np.arange(nr + 1, dtype=np.int32),
                start_index=start,
                num_reqs=nr,
                hidden=hidden_size,
                device=device,
                attn_ctx=SimpleNamespace(use_max_model_len=True),
            ))
        start += nr
    proposer.draft_chunks = chunks

    # A traceable stub (not a MagicMock): propose() now calls
    # combine_hidden_states / compute_logits inside @torch.compile regions
    # (_draft_combine_hidden_states / _draft_propose_token), and torch.compile
    # with fullgraph=True cannot trace a MagicMock. combine is identity; the
    # argmax of the one-hot logits recovers round(hidden[:, 0]) as the token.
    class _StubDraftModel:

        # Mirrors the real draft's `.model.use_aux_hidden_state`, read by
        # _draft_uses_aux_hidden_state() to pick the combine path.
        model = SimpleNamespace(use_aux_hidden_state=True)

        def combine_hidden_states(self, x):
            return x

        def compute_logits(self, hidden):
            tokens = hidden[:, 0].round().clamp(min=0).to(torch.int64)
            return torch.nn.functional.one_hot(tokens,
                                               vocab_size).to(torch.float32)

    proposer.draft_model = _StubDraftModel()

    # Per-chunk next_tokens sentinels to verify propose threads the right one
    # to each chunk's _prepare_draft_inputs.
    nt_per_chunk = [
        torch.full((nr, num_speculative_tokens + 1),
                   700 + ci,
                   dtype=torch.int32,
                   device=device) for ci, nr in enumerate(chunk_sizes)
    ]
    received_next_tokens = []

    # Per-chunk: accepted-prefix slots are the first `num_reqs` rows.
    def _prepare(chunk, *_args, **_kwargs):
        received_next_tokens.append(_kwargs.get("next_tokens_device"))
        padded = 16
        local_idx = torch.arange(chunk.num_reqs, device=device)
        # num_rejected: the async (is_async, return_device) path consumes this
        # via _maybe_pad_dim0, so it must be a device tensor (mirroring the real
        # async _prepare_draft_inputs, which derives it on-device), not numpy.
        return (
            torch.zeros(padded, dtype=torch.int32, device=device),
            torch.zeros(padded, dtype=torch.int32, device=device),
            local_idx,
            torch.ones(chunk.num_reqs, dtype=torch.int32, device=device),
        )

    proposer._prepare_draft_inputs = _prepare

    def _forward_draft(*, chunk, input_ids, step_idx, **_kwargs):
        n = input_ids.shape[0]
        last_hidden = torch.zeros((n, hidden_size), device=device)
        if step_idx == 0:
            base = torch.tensor(
                base_token_ids[chunk.start_index:chunk.start_index +
                               chunk.num_reqs],
                dtype=torch.float32,
                device=device)
            last_hidden[torch.arange(chunk.num_reqs, device=device), 0] = base
        else:
            # input_ids carries the previous step's draft tokens; +1 each step.
            last_hidden[:, 0] = (input_ids + 1).to(torch.float32)
        return last_hidden, last_hidden

    proposer._forward_draft = _forward_draft

    result = proposer.propose(
        sampled_token_ids=[[0]] * num_reqs,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
        return_device=return_device,
        next_tokens_per_chunk=nt_per_chunk,
    )

    # propose threads each chunk's next_tokens to its _prepare_draft_inputs.
    assert len(received_next_tokens) == len(chunk_sizes)
    for ci in range(len(chunk_sizes)):
        assert received_next_tokens[ci] is nt_per_chunk[ci]

    expected = [[
        base_token_ids[i] + step for step in range(num_speculative_tokens)
    ] for i in range(num_reqs)]
    if return_device:
        # Async path: raw [num_reqs, K] device tensor, same values.
        assert isinstance(result, torch.Tensor)
        assert result.shape == (num_reqs, num_speculative_tokens)
        assert result.device.type == device.type
        assert result.cpu().tolist() == expected
    else:
        assert result == expected


def test_draft_combine_input_size_no_aux_hidden_state():
    proposer = _make_proposer(draft_tp=1)
    proposer.draft_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=8192),
        model=SimpleNamespace(use_aux_hidden_state=False),
    )
    assert proposer._draft_uses_aux_hidden_state() is False
    assert proposer._draft_combine_input_size() == 8192


def test_draft_combine_input_size_with_aux_hidden_state():
    proposer = _make_proposer(draft_tp=1)
    proposer.draft_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=8192),
        model=SimpleNamespace(use_aux_hidden_state=True, fc_input_size=24576),
    )
    assert proposer._draft_uses_aux_hidden_state() is True
    assert proposer._draft_combine_input_size() == 24576


def test_propose_without_aux_hidden_state(device):
    hidden_size = 8
    proposer = _make_proposer(draft_tp=1)
    proposer.speculative_config.num_speculative_tokens = 1
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=1),
        max_num_reqs=16,
        num_reqs_max_model_len=16,
        num_reqs_most_model_len=16,
        num_tokens_paddings=[16],
        device=device,
        _dp_lockstep_enabled=lambda: False,
    )

    aux = torch.full((16, hidden_size), 111.0, device=device)
    plain_hidden = torch.full((16, hidden_size), 222.0, device=device)
    chunk = _make_chunk(
        input_ids=torch.zeros(16, dtype=torch.int32, device=device),
        position_ids=torch.zeros(16, dtype=torch.int32, device=device),
        query_start_loc_np=np.arange(2, dtype=np.int32),
        start_index=0,
        num_reqs=1,
        hidden=hidden_size,
        device=device,
        attn_ctx=SimpleNamespace(use_max_model_len=True),
        hidden_states=plain_hidden,
    )
    chunk.aux_hidden_states = [aux]
    proposer.draft_chunks = [chunk]

    # Plain stub, not a MagicMock: compute_logits runs inside the
    # @torch.compile(fullgraph=True) _draft_propose_token wrapper, which
    # cannot trace mocks (same reason test_propose uses _StubDraftModel).
    combine_calls = []

    class _StubNoAuxDraftModel:

        model = SimpleNamespace(use_aux_hidden_state=False)

        def combine_hidden_states(self, x):
            combine_calls.append(x)
            return x

        def compute_logits(self, hidden):
            return torch.zeros((hidden.shape[0], 4), device=device)

    proposer.draft_model = _StubNoAuxDraftModel()

    def _prepare(chunk, *_args, **_kwargs):
        return (
            torch.zeros(16, dtype=torch.int32, device=device),
            torch.zeros(16, dtype=torch.int32, device=device),
            torch.arange(chunk.num_reqs, device=device),
            np.zeros(chunk.num_reqs, dtype=np.int32),
        )

    proposer._prepare_draft_inputs = _prepare

    def _forward_draft(*, input_ids, **_kwargs):
        n = input_ids.shape[0]
        last_hidden = torch.zeros((n, hidden_size), device=device)
        return last_hidden, last_hidden

    proposer._forward_draft = _forward_draft

    proposer.propose(
        sampled_token_ids=[[0]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
    )

    # combine_hidden_states must see the plain hidden_states tensor, not the
    # (single-element, but distinct-valued) aux_hidden_states concatenation.
    assert len(combine_calls) == 1
    assert torch.equal(combine_calls[0], plain_hidden)


def test_propose_empty_batch():
    proposer = _make_proposer(draft_tp=1)
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=0))
    assert proposer.propose([], [], None,
                            SimpleNamespace(num_scheduled_tokens={})) == []


def test_propose_pads_to_coordinated_chunk_bound(device):
    """Under EP-DP lockstep, a rank with fewer chunks than the coordinated
    bound must pad with run_dp_dummy_draft chunks so draft-forward counts pair
    across DP ranks."""
    proposer = _make_proposer(draft_tp=1)
    proposer.speculative_config.num_speculative_tokens = 1
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=1),
        max_num_reqs=16,
        num_reqs_max_model_len=16,
        num_reqs_most_model_len=16,
        num_tokens_paddings=[16],
        device=device,
        # Lockstep ON with a coordinated bound of 3 chunks; this rank has 1.
        _dp_lockstep_enabled=lambda: True,
        _dp_step_num_chunks=3,
        _dp_target_bucket=16,
    )

    chunk = _make_chunk(
        input_ids=torch.zeros(16, dtype=torch.int32, device=device),
        position_ids=torch.zeros(16, dtype=torch.int32, device=device),
        query_start_loc_np=np.arange(2, dtype=np.int32),
        start_index=0,
        num_reqs=1,
        device=device,
        attn_ctx=SimpleNamespace(use_max_model_len=True),
    )
    proposer.draft_chunks = [chunk]

    class _StubDraft:

        model = SimpleNamespace(use_aux_hidden_state=True)

        def combine_hidden_states(self, x):
            return x

        def compute_logits(self, hidden):
            return torch.zeros((hidden.shape[0], 4), device=device)

    proposer.draft_model = _StubDraft()
    proposer._prepare_draft_inputs = lambda *a, **k: (
        torch.zeros(16, dtype=torch.int32, device=device),
        torch.zeros(16, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
        np.zeros(1, dtype=np.int32),
    )
    proposer._forward_draft = lambda **k: (
        torch.zeros((16, 8), device=device),
        torch.zeros((16, 8), device=device),
    )
    dummy_chunks = []
    proposer.run_dp_dummy_draft = lambda n: dummy_chunks.append(n)

    proposer.propose(
        sampled_token_ids=[[0]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
    )

    # coordinated bound 3 − 1 real chunk = 2 padding chunks.
    assert dummy_chunks == [2]


def test_loop_bucket_constant_under_sharded_lockstep():
    """Sharded EP-DP lockstep pins the loop bucket to pad(max_num_reqs) so
    collective shapes match across DP ranks regardless of per-rank load;
    the replicated draft keeps the per-chunk bucket."""
    proposer = _make_proposer(draft_tp=2, target_tp=2)
    proposer.runner = SimpleNamespace(
        num_tokens_paddings=[16, 32, 64, 128],
        max_num_reqs=50,
        _dp_lockstep_enabled=lambda: True,
    )
    assert proposer._loop_bucket(1) == 64
    assert proposer._loop_bucket(33) == 64

    rep = _make_proposer(draft_tp=1, target_tp=2)
    rep.runner = SimpleNamespace(num_tokens_paddings=[16, 32, 64, 128],
                                 _dp_lockstep_enabled=lambda: False)
    assert rep._loop_bucket(1) == 16
    assert rep._loop_bucket(33) == 64


def test_run_dp_dummy_draft_sharded_replays_propose_trace(monkeypatch):
    """Sharded draft under EP-DP lockstep: each padding chunk must replay the
    real propose's collective trace — combine (1/chunk), first-pass forward
    @bucket, K lm-head gathers @constant loop bucket, K-1 loop forwards @that
    bucket — not just K bare forwards."""
    import vllm_torchtpu.spec_decode.eagle3 as e3
    monkeypatch.setattr(e3, "synchronize_tensors", lambda *a, **k: None)

    proposer = _make_proposer(draft_tp=2, target_tp=2)
    proposer.speculative_config.num_speculative_tokens = 3
    proposer.runner = SimpleNamespace(
        _dp_target_bucket=64,
        num_reqs_max_model_len=16,
        max_num_reqs=16,
        num_tokens_paddings=[16, 32, 64],
        device=torch.device("cpu"),
        _hidden_states_dtype=torch.float32,
        uses_mrope=False,
        _dp_lockstep_enabled=lambda: True,
    )
    proposer.draft_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=8),
        model=SimpleNamespace(fc_input_size=24),
    )
    proposer._draft_uses_aux_hidden_state = lambda: True

    combine, fwd, tok_shapes = [], [], []
    proposer._draft_combine_hidden_states = lambda *a: combine.append(a)

    def fake_forward(**kw):
        fwd.append(kw)
        return (torch.zeros((kw["num_tokens_padded"], 8)), None)

    proposer._forward_draft = fake_forward

    def fake_propose_token(h):
        tok_shapes.append(tuple(h.shape))
        return torch.zeros(h.shape[0], dtype=torch.int32)

    proposer._draft_propose_token = fake_propose_token

    proposer.run_dp_dummy_draft(2)

    # One combine per chunk, at the coordinated bucket width (aux thirds).
    assert len(combine) == 2
    assert all(t.shape == (64, 8) for t in combine[0])
    # K forwards per chunk: first pass @bucket=64, then K-1 loop @p=16
    # (constant loop bucket = pad(max_num_reqs)).
    assert [f["num_tokens_padded"] for f in fwd] == [64, 16, 16] * 2
    assert [f["step_idx"] for f in fwd] == [0, 1, 2] * 2
    # Loop steps carry the loop metadata tensors (shape parity with propose).
    assert all(
        f.get("loop_query_start_loc") is not None for f in fwd
        if f["step_idx"] > 0)
    # K lm-head gathers per chunk, all at the constant loop bucket.
    assert tok_shapes == [(16, 8)] * 6


def test_run_dp_dummy_draft_mtp_replays_no_combine(monkeypatch):
    """An MTP draft has no combine_hidden_states, so the replay must not
    emit one.

    propose() runs combine only for eagle3; MTP feeds the target hidden
    state straight to the forward. Replaying a combine here would both
    raise AttributeError on the draft and add a step the real trace does
    not have.
    """
    import vllm_torchtpu.spec_decode.eagle3 as e3
    monkeypatch.setattr(e3, "synchronize_tensors", lambda *a, **k: None)

    proposer = _make_proposer(draft_tp=1, target_tp=1, method="qwen3_next_mtp")
    proposer.speculative_config.num_speculative_tokens = 3
    proposer.runner = SimpleNamespace(
        _dp_target_bucket=64,
        num_reqs_max_model_len=16,
        max_num_reqs=16,
        num_tokens_paddings=[16, 32, 64],
        device=torch.device("cpu"),
        _hidden_states_dtype=torch.float32,
        uses_mrope=False,
        _dp_lockstep_enabled=lambda: True,
    )
    # A real Qwen3_5MoeMTP has no combine_hidden_states attribute at all;
    # SimpleNamespace reproduces that, so a stray call raises here too.
    proposer.draft_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=8),
        model=SimpleNamespace(use_aux_hidden_state=False),
    )
    proposer._draft_has_moe_cache = True

    fwd, tok_shapes = [], []

    def fake_forward(**kw):
        fwd.append(kw)
        return (torch.zeros((kw["num_tokens_padded"], 8)), None)

    proposer._forward_draft = fake_forward

    def fake_propose_token(h):
        tok_shapes.append(tuple(h.shape))
        return torch.zeros(h.shape[0], dtype=torch.int32)

    proposer._draft_propose_token = fake_propose_token

    proposer.run_dp_dummy_draft(2)

    # Same forward/lm-head trace as eagle3, only without the combine step.
    assert [f["num_tokens_padded"] for f in fwd] == [64, 16, 16] * 2
    assert [f["step_idx"] for f in fwd] == [0, 1, 2] * 2
    assert tok_shapes == [(16, 8)] * 6


@pytest.mark.parametrize("uses_mrope", [False, True])
def test_run_dp_dummy_draft_positions_match_draft_graph_rank(
        monkeypatch, uses_mrope):
    """Dummy positions must carry the rank the draft graph was compiled for.

    The draft's compiled graph marks `positions` dim -1 dynamic, so on an
    mrope model it is specialized on the [3, N] rank the real propose
    passes. A 1-D dummy trips the guard with `IndexError: Dimension out of
    range`. The attn-metadata build is the opposite: it always wants the
    1-D token positions, like runner.position_ids.
    """
    import vllm_torchtpu.spec_decode.eagle3 as e3
    monkeypatch.setattr(e3, "synchronize_tensors", lambda *a, **k: None)

    proposer = _make_proposer(draft_tp=1, target_tp=1, method="qwen3_next_mtp")
    proposer.speculative_config.num_speculative_tokens = 3
    proposer.runner = SimpleNamespace(
        _dp_target_bucket=64,
        num_reqs_max_model_len=16,
        max_num_reqs=16,
        num_tokens_paddings=[16, 32, 64],
        device=torch.device("cpu"),
        _hidden_states_dtype=torch.float32,
        uses_mrope=uses_mrope,
        _dp_lockstep_enabled=lambda: True,
    )
    proposer.draft_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=8),
        model=SimpleNamespace(use_aux_hidden_state=False),
    )
    proposer._draft_has_moe_cache = True

    fwd = []

    def fake_forward(**kw):
        fwd.append(kw)
        return (torch.zeros((kw["num_tokens_padded"], 8)), None)

    proposer._forward_draft = fake_forward
    proposer._draft_propose_token = lambda h: torch.zeros(h.shape[0],
                                                          dtype=torch.int32)

    proposer.run_dp_dummy_draft(1)

    first, loop = fwd[0], fwd[1]
    if uses_mrope:
        # [3, N] for both the first pass (bucket) and the loop carry (p).
        assert first["positions"].shape == (3, 64)
        assert loop["positions"].shape == (3, 16)
    else:
        assert first["positions"].shape == (64, )
        assert loop["positions"].shape == (16, )
    # The attn-metadata override stays 1-D either way.
    assert first["chunk"].attn_ctx.position_ids_override.shape == (64, )


def test_build_draft_attn_metadata_loop_cache(device):
    """Loop iterations >= 2 must reuse the step-1 metadata objects with only
    seq_lens swapped (advanced by +1), value-identical to a full rebuild."""
    import dataclasses as dc

    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

    proposer = _make_proposer(draft_tp=1)
    proposer._draft_attn_layer_names = {"draft.attn.0"}
    runner = SimpleNamespace(
        _has_mamba_state=False,
        _unified_kv_layout=False,
        num_reqs_max_model_len=8,
        num_reqs_most_model_len=8,
        device=device,
        _attn_metadata_builder_ctx=None,
        empty_slot_mappings={},
    )
    proposer.runner = runner

    # chunk_ctx.seq_lens is the full [kernel_num_reqs]-padded tensor in
    # production; tail entries are ignored by the kernel.
    base_seq_lens = torch.tensor([5, 9, 0, 0, 0, 0, 0, 0],
                                 dtype=torch.int32,
                                 device=device)
    chunk = SimpleNamespace(
        num_reqs=2,
        start_index=0,
        query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
        attn_ctx=SimpleNamespace(
            use_max_model_len=True,
            seq_lens=base_seq_lens,
            query_start_loc="QSL",
            request_distribution="RD",
            position_ids_override=None,
            sequence_layout_descriptor=SequenceLayoutDescriptor(),
        ),
    )

    build_calls = []

    def fake_build_attention_metadata(**kw):
        build_calls.append(kw)
        ctx = runner._attn_metadata_builder_ctx
        md = AttentionMetadata(
            input_positions=torch.zeros(1, device=device),
            block_tables=torch.zeros(1, dtype=torch.int32, device=device),
            seq_lens=ctx.seq_lens,
            query_start_loc=ctx.query_start_loc,
            request_distribution=ctx.request_distribution,
        )
        return {"draft.attn.0": md, "target.attn.0": md}, None

    runner._build_attention_metadata = fake_build_attention_metadata
    loop_qsl = torch.arange(9, dtype=torch.int32, device=device)
    loop_rd = torch.tensor([2, 2, 2], dtype=torch.int32, device=device)

    def build(step):
        return proposer._build_draft_attn_metadata(
            chunk,
            step_idx=step,
            seq_lens_delta=step,
            num_rejected_np=np.array([1, 0], dtype=np.int32),
            num_tokens_padded=8,
            loop_query_start_loc=loop_qsl,
            loop_request_distribution=loop_rd,
        )

    md1 = build(1)
    assert len(build_calls) == 1
    md2 = build(2)
    md3 = build(3)
    # No further upstream builder calls after step 1.
    assert len(build_calls) == 1
    # Draft-layer filter held; non-seq_lens fields are the SAME objects.
    assert set(md2) == {"draft.attn.0"}
    assert md2["draft.attn.0"].block_tables is md1["draft.attn.0"].block_tables
    assert md2["draft.attn.0"].query_start_loc is loop_qsl
    # seq_lens advance: step s = base + s - rejected.
    rej = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0],
                       dtype=torch.int32,
                       device=device)
    for s, md in ((1, md1), (2, md2), (3, md3)):
        expect = base_seq_lens + s - rej
        assert torch.equal(md["draft.attn.0"].seq_lens.cpu(), expect.cpu()), s
    # Cached objects are fresh dataclass copies, not mutated step-1 objects.
    assert md2["draft.attn.0"] is not md1["draft.attn.0"]
    assert dc.is_dataclass(md2["draft.attn.0"])


def test_build_draft_attn_metadata_propagates_chunk_layout_descriptor(device):
    """Rebuilding draft metadata must preserve the target chunk's PCP
    descriptor; defaulting it to ALL selects an incompatible attention path."""
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

    proposer = _make_proposer(draft_tp=1, method="mtp")
    proposer._draft_attn_layer_names = {"draft.attn.0"}
    descriptor = _rank0_pcp_plan().descriptor
    runner = SimpleNamespace(
        _has_mamba_state=False,
        _unified_kv_layout=False,
        num_reqs_max_model_len=8,
        num_reqs_most_model_len=8,
        device=device,
        _attn_metadata_builder_ctx=None,
        empty_slot_mappings={},
    )
    proposer.runner = runner
    captured_descriptors = []

    def fake_build_attention_metadata(**_kwargs):
        ctx = runner._attn_metadata_builder_ctx
        captured_descriptors.append(ctx.sequence_layout_descriptor)
        md = AttentionMetadata(
            input_positions=torch.zeros(3, device=device),
            block_tables=torch.zeros(1, dtype=torch.int32, device=device),
            seq_lens=ctx.seq_lens,
            query_start_loc=ctx.query_start_loc,
            request_distribution=ctx.request_distribution,
        )
        return {"draft.attn.0": md}, None

    runner._build_attention_metadata = fake_build_attention_metadata
    chunk = SimpleNamespace(
        num_reqs=2,
        start_index=0,
        query_start_loc_np=np.array([0, 2, 5], dtype=np.int32),
        attn_ctx=SimpleNamespace(
            use_max_model_len=True,
            seq_lens=torch.tensor([2, 3, 0, 0, 0, 0, 0, 0],
                                  dtype=torch.int32,
                                  device=device),
            query_start_loc=torch.tensor([0, 2, 5],
                                         dtype=torch.int32,
                                         device=device),
            request_distribution=torch.tensor([0, 2, 2],
                                              dtype=torch.int32,
                                              device=device),
            position_ids_override=None,
            sequence_layout_descriptor=descriptor,
        ),
    )

    proposer._build_draft_attn_metadata(
        chunk,
        step_idx=0,
        seq_lens_delta=0,
        num_rejected_np=np.zeros(2, dtype=np.int32),
        num_tokens_padded=3,
    )

    assert captured_descriptors == [descriptor]


def test_load_draft_model_saves_and_wraps_with_draft_config(monkeypatch):
    """The exact copied config used to construct the draft must also be the
    TPU wrapper config; recovering it from an optional model attribute is not
    an equivalent contract."""
    import vllm_torchtpu.spec_decode.eagle3 as e3

    proposer = _make_proposer(draft_tp=1, method="mtp")
    proposer.runner = SimpleNamespace(mesh=object())
    proposer.vllm_config.load_config = object()
    proposer.vllm_config.compilation_config = SimpleNamespace(
        inductor_compile_config={})
    proposer.speculative_config.draft_model_config = SimpleNamespace(
        runner_type=None)
    loaded = {}

    class _Loader:

        def load_model(self, *, vllm_config, model_config):
            loaded["vllm_config"] = vllm_config
            loaded["model_config"] = model_config
            return object()

    wrapper_configs = []

    @contextlib.contextmanager
    def wrapper_context(*, mesh, vllm_config=None):
        del mesh
        wrapper_configs.append(vllm_config)
        yield

    monkeypatch.setattr(e3, "get_model_loader", lambda _cfg: _Loader())
    monkeypatch.setattr(e3, "set_model_tag",
                        lambda _tag: contextlib.nullcontext())
    monkeypatch.setattr(e3, "set_current_vllm_config",
                        lambda _cfg: contextlib.nullcontext())
    monkeypatch.setattr(e3, "set_vllm_model_wrapper_context", wrapper_context)
    monkeypatch.setattr(e3, "_force_draft_tp1",
                        lambda: contextlib.nullcontext())

    proposer._load_draft_model()

    assert proposer.draft_vllm_config is loaded["vllm_config"]
    assert wrapper_configs == [proposer.draft_vllm_config]


def test_forward_draft_uses_saved_draft_config_in_both_contexts(
        monkeypatch, device):
    """Both vLLM forward metadata and the TPU wrapper must observe the saved
    draft config so PCP descriptor and interleave configuration agree."""
    import vllm_torchtpu.spec_decode.eagle3 as e3

    proposer = _make_proposer(draft_tp=1, method="mtp")
    draft_config = object()
    proposer.draft_vllm_config = draft_config
    proposer.runner = SimpleNamespace(
        mesh=object(),
        _dp_num_tokens_across_dp=lambda n: n,
    )
    proposer._build_draft_attn_metadata = lambda **_kwargs: {"draft": "md"}

    class _Draft:

        def __call__(self, **kwargs):
            return kwargs["hidden_states"]

    proposer.draft_model = _Draft()
    seen = {}

    @contextlib.contextmanager
    def forward_context(metadata, vllm_config, **kwargs):
        seen["forward"] = (metadata, vllm_config, kwargs)
        yield

    @contextlib.contextmanager
    def wrapper_context(*, mesh, vllm_config=None):
        seen["wrapper"] = (mesh, vllm_config)
        yield

    monkeypatch.setattr(e3, "set_model_tag",
                        lambda _tag: contextlib.nullcontext())
    monkeypatch.setattr(e3, "set_forward_context", forward_context)
    monkeypatch.setattr(e3, "set_vllm_model_wrapper_context", wrapper_context)

    hidden = torch.arange(6, dtype=torch.float32, device=device).reshape(3, 2)
    last_hidden, carry = proposer._forward_draft(
        chunk=SimpleNamespace(),
        input_ids=torch.zeros(3, dtype=torch.int32, device=device),
        positions=torch.arange(3, dtype=torch.int32, device=device),
        target_hidden_states=hidden,
        step_idx=0,
        seq_lens_delta=0,
        num_rejected_np=None,
    )

    assert seen["forward"][1] is draft_config
    assert seen["wrapper"] == (proposer.runner.mesh, draft_config)
    assert torch.equal(last_hidden, hidden)
    assert torch.equal(carry, hidden)


def test_pcp_mtp_draft_cache_specs_require_full_attention():
    """PCP K=1 rejects draft cache specs that need unsupported routing."""
    proposer = _make_proposer(draft_tp=1, method="mtp")
    draft_attn_layer_names = {"draft.attn"}
    full_spec = FullAttentionSpec(block_size=16,
                                  num_kv_heads=1,
                                  head_size=128,
                                  dtype=torch.bfloat16)
    proposer.runner = SimpleNamespace(
        get_kv_cache_spec=lambda: {"draft.attn": full_spec})

    proposer._validate_pcp_draft_model(draft_attn_layer_names)

    mamba_spec = MambaSpec(block_size=1,
                           shapes=((1, 4, 8), ),
                           dtypes=(torch.bfloat16, ))
    proposer.runner = SimpleNamespace(
        get_kv_cache_spec=lambda: {"draft.attn": mamba_spec})
    with pytest.raises(NotImplementedError,
                       match="FullAttentionSpec.*draft.attn"):
        proposer._validate_pcp_draft_model(draft_attn_layer_names)


def test_build_draft_attn_metadata_cache_cleared_per_propose(device):
    """A new propose() run must not reuse a previous step's cached metadata
    for a recycled chunk id — the cache is cleared at propose entry."""
    proposer = _make_proposer(draft_tp=1)
    proposer._draft_md_cache[12345] = ("stale", None)
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=0))
    assert proposer.propose([], [], None, None) == []
    assert proposer._draft_md_cache == {}


# ---------------------------------------------------------------------------
# MTP IndexShare (GLM-5.2 / DeepSeek-V3.2)
# ---------------------------------------------------------------------------


class _FakeMlaAttn:

    def __init__(self, buf):
        self.topk_indices_buffer = buf


class _FakeMtpLayer:

    def __init__(self, buf):
        self.mtp_block = SimpleNamespace(self_attn=SimpleNamespace(
            mla_attn=_FakeMlaAttn(buf)))


class _FakeMtpModel:
    """Stands in for DeepSeekMultiTokenPredictor: one shared buffer, front
    anchored compaction, and a runtime skip_topk toggle."""

    def __init__(self, buf, num_layers: int = 1):
        self.layers = {
            str(78 + i): _FakeMtpLayer(buf)
            for i in range(num_layers)
        }
        self.skip_topk = None

    def set_skip_topk(self, skip: bool):
        self.skip_topk = skip

    def compact_topk_indices(self, slot_ids):
        num_slots = slot_ids.numel()
        for layer in self.layers.values():
            buf = layer.mtp_block.self_attn.mla_attn.topk_indices_buffer
            buf[:num_slots] = buf[slot_ids]


def _mtp_proposer_with_buffer(buf,
                              share: bool = True,
                              num_layers: int = 1,
                              max_num_reqs: int = 2,
                              num_tokens_paddings=(2, 4, 6, 8)):
    proposer = _make_proposer(method="mtp")
    proposer.draft_model = SimpleNamespace(
        model=_FakeMtpModel(buf, num_layers))
    proposer.speculative_config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(index_share_for_mtp_iteration=share))
    # `_init_mtp_index_sharing` checks the buffer against the loop bucket it
    # will be indexed by, which comes off the runner's token paddings.
    proposer.runner.max_num_reqs = max_num_reqs
    proposer.runner.num_tokens_paddings = list(num_tokens_paddings)
    proposer.vllm_config.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=buf.shape[0])
    proposer._init_mtp_index_sharing()
    return proposer


def test_index_sharing_off_when_checkpoint_does_not_ask():
    buf = torch.zeros(8, 4, dtype=torch.int32)
    proposer = _mtp_proposer_with_buffer(buf, share=False)
    assert proposer._share_mtp_indices is False
    assert proposer._mtp_topk_buffers == []


def test_index_sharing_seeds_buffer_with_padding_sentinel():
    buf = torch.zeros(8, 4, dtype=torch.int32)
    proposer = _mtp_proposer_with_buffer(buf)
    assert proposer._share_mtp_indices is True
    # -1 is the sparse-MLA padding sentinel; an unwritten row must not gather.
    assert torch.equal(buf, torch.full((8, 4), -1, dtype=torch.int32))


def test_index_sharing_dedupes_aliased_buffers():
    buf = torch.zeros(8, 4, dtype=torch.int32)
    proposer = _mtp_proposer_with_buffer(buf, num_layers=3)
    assert len(proposer._mtp_topk_buffers) == 1


def test_compact_gathers_last_token_rows_to_front():
    buf = torch.arange(6 * 4, dtype=torch.int32).reshape(6, 4)
    proposer = _mtp_proposer_with_buffer(buf.clone())
    live = proposer._mtp_topk_buffers[0]
    live.copy_(buf)

    # Two requests whose last tokens sit at rows 1 and 4 of the step-0 forward.
    saved = proposer._mtp_compact_and_save(torch.tensor([1, 4]))

    assert torch.equal(live[0], buf[1])
    assert torch.equal(live[1], buf[4])
    assert torch.equal(saved[0], torch.stack([buf[1], buf[4]]))


def test_second_chunk_step0_clobbers_first_chunk_rows_and_restore_fixes_it():
    """The buffer is one tensor indexed by position-in-the-current-forward, so
    chunk 1's step-0 indexer write lands on chunk 0's compacted rows. The
    snapshot taken at compaction time is what makes IndexShare correct for more
    than one chunk."""
    buf = torch.zeros(6, 4, dtype=torch.int32)
    proposer = _mtp_proposer_with_buffer(buf)
    live = proposer._mtp_topk_buffers[0]

    # Chunk 0 step 0 writes its per-token rows, then compacts + snapshots.
    live.copy_(torch.arange(6 * 4, dtype=torch.int32).reshape(6, 4))
    chunk0_rows = torch.stack([live[1].clone(), live[4].clone()])
    saved0 = proposer._mtp_compact_and_save(torch.tensor([1, 4]))
    assert torch.equal(live[:2], chunk0_rows)

    # Chunk 1 step 0 writes over the front of the same buffer.
    live.copy_(torch.full((6, 4), 99, dtype=torch.int32))
    saved1 = proposer._mtp_compact_and_save(torch.tensor([0, 2]))
    assert not torch.equal(live[:2], chunk0_rows)  # chunk 0's rows are gone

    # Restoring before chunk 0's loop-step forward brings them back.
    proposer._mtp_restore_topk(saved0)
    assert torch.equal(live[:2], chunk0_rows)
    # ... and chunk 1's snapshot is still intact for its own turn.
    proposer._mtp_restore_topk(saved1)
    assert torch.equal(live[:2], torch.full((2, 4), 99, dtype=torch.int32))


def test_skip_topk_toggle_is_noop_without_sharing():
    buf = torch.zeros(8, 4, dtype=torch.int32)
    proposer = _mtp_proposer_with_buffer(buf, share=False)
    proposer._mtp_set_skip_topk(True)
    assert proposer.draft_model.model.skip_topk is None


def test_skip_topk_toggle_applies_with_sharing():
    buf = torch.zeros(8, 4, dtype=torch.int32)
    proposer = _mtp_proposer_with_buffer(buf)
    proposer._mtp_set_skip_topk(False)
    assert proposer.draft_model.model.skip_topk is False
    proposer._mtp_set_skip_topk(True)
    assert proposer.draft_model.model.skip_topk is True


def test_propose_index_share_order_across_two_chunks():
    """End-to-end ordering: compact+snapshot each chunk right after its step 0,
    and restore each chunk's rows before its own loop-step forward. Which
    program each step runs, and the skip_topk it runs with, is decided inside
    _forward_draft and tested through it below."""
    device = torch.device("cpu")
    K = 3
    proposer = _make_proposer(draft_tp=1, method="mtp")
    proposer.speculative_config.num_speculative_tokens = K
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=4,
                                    req_ids=["r0", "r1", "r2", "r3"]),
        device=device,
        requests={},
        num_tokens_paddings=[2, 4, 8],
        max_num_reqs=2,
        uses_mrope=False,
        num_reqs_max_model_len=2,
        num_reqs_most_model_len=None,
        _dp_lockstep_enabled=lambda: False,
    )
    proposer.vllm_config.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8)

    buf = torch.zeros(8, 4, dtype=torch.int32)
    proposer.draft_model = SimpleNamespace(model=_FakeMtpModel(buf))
    proposer.speculative_config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(index_share_for_mtp_iteration=True))
    proposer._init_mtp_index_sharing()
    assert proposer._share_mtp_indices is True

    chunks = []
    for ci in range(2):
        chunks.append(
            _make_chunk(
                input_ids=torch.zeros(2, dtype=torch.int32, device=device),
                position_ids=torch.zeros(2, dtype=torch.int32, device=device),
                query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
                start_index=ci * 2,
                num_reqs=2,
                hidden=1,
                device=device,
                hidden_states=torch.zeros((2, 1), device=device),
                attn_ctx=SimpleNamespace(use_max_model_len=True),
            ))
    proposer.draft_chunks = chunks
    proposer._prepare_draft_inputs = lambda *_a, **_k: (
        torch.zeros(2, dtype=torch.int32, device=device),
        torch.zeros(2, dtype=torch.int32, device=device),
        torch.tensor([0, 1], dtype=torch.int64, device=device),
        np.zeros(2, dtype=np.int32),
    )
    proposer._loop_bucket = lambda n: 2

    trace = []
    # Distinct per-chunk step-0 index rows, so a restore mix-up is visible.
    step0_rows = {
        0: torch.tensor([[10, 11, 12, 13], [14, 15, 16, 17]],
                        dtype=torch.int32),
        1: torch.tensor([[20, 21, 22, 23], [24, 25, 26, 27]],
                        dtype=torch.int32),
    }
    live = proposer._mtp_topk_buffers[0]
    # DraftChunkInputs holds tensors, so `==` is ambiguous; key by identity.
    chunk_ids = {id(c): i for i, c in enumerate(chunks)}

    def fake_forward(*, chunk, step_idx, **_k):
        ci = chunk_ids[id(chunk)]
        if step_idx == 0:
            # Simulate the indexer writing this chunk's per-token rows.
            live[:2] = step0_rows[ci]
        else:
            trace.append((step_idx, ci, live[:2].clone()))
        return (torch.zeros((2, 1),
                            device=device), torch.zeros((2, 1), device=device))

    proposer._forward_draft = fake_forward
    proposer._draft_gather_carries = lambda h, p, lh, gi: (h[:2], p[:2], lh[:2]
                                                           )
    proposer._draft_propose_token = lambda lh: torch.zeros(
        2, dtype=torch.int32, device=device)

    proposer.propose(
        sampled_token_ids=[[1], [2], [3], [4]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
    )

    # Every loop-step forward saw its OWN chunk's step-0 rows, at every step.
    assert len(trace) == (K - 1) * 2
    for step_idx, ci, seen in trace:
        assert torch.equal(seen, step0_rows[ci]), (
            f"step {step_idx} chunk {ci} read the wrong chunk's indices")


# --------------------------------------------------------------------------
# DSA (sparse-MLA) draft KV-cache validation
# --------------------------------------------------------------------------


def _dsa_layer_classes():
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

    class _FakeIndexerCache(DeepseekV32IndexerCache):
        # The real __init__ registers itself in static_forward_context and
        # needs a CacheConfig; only isinstance matters here.
        def __init__(self):
            torch.nn.Module.__init__(self)

    class _FakeMlaLayer(MLAAttention):

        def __init__(self):
            torch.nn.Module.__init__(self)

    return _FakeIndexerCache, _FakeMlaLayer


def _dsa_draft_proposer(layers, specs, draft_tp=8, target_tp=8):
    proposer = _make_proposer(draft_tp=draft_tp,
                              target_tp=target_tp,
                              method="mtp")
    proposer.runner.get_kv_cache_spec.return_value = specs
    return proposer, set(layers)


def test_validate_dsa_draft_is_a_noop_for_a_non_sparse_draft():
    """An eagle3 / plain-MTP draft has no lightning indexer, so the check must
    not even ask the runner for specs (it is called on every load)."""
    proposer = _make_proposer(method="eagle3")
    layers = {"draft.attn": torch.nn.Module()}
    proposer._validate_dsa_draft(layers, {"draft.attn"})
    proposer.runner.get_kv_cache_spec.assert_not_called()


def test_validate_dsa_draft_rejects_an_indexer_without_mla():
    indexer_cls, _ = _dsa_layer_classes()
    layers = {"d.indexer.k_cache": indexer_cls()}
    proposer, names = _dsa_draft_proposer(layers, {})
    with pytest.raises(RuntimeError, match="no MLA attention layer"):
        proposer._validate_dsa_draft(layers, names)


def test_validate_dsa_draft_rejects_a_replicated_draft():
    """The sparse-MLA and indexer Pallas ops are mesh-sharded with no LOCAL
    variant to relocate to, unlike the dense RPA path."""
    indexer_cls, mla_cls = _dsa_layer_classes()
    layers = {
        "d.indexer.k_cache": indexer_cls(),
        "d.attn": mla_cls(),
    }
    proposer, names = _dsa_draft_proposer(layers, {}, draft_tp=1, target_tp=8)
    with pytest.raises(NotImplementedError,
                       match="draft_tensor_parallel_size=1"):
        proposer._validate_dsa_draft(layers, names)


def test_validate_dsa_draft_rejects_a_layer_with_no_kv_cache_spec():
    """A layer the runner's spec walk skipped is allocated no KV cache, and the
    indexer then returns an unwritten top-k buffer with no error anywhere."""
    indexer_cls, mla_cls = _dsa_layer_classes()
    layers = {
        "d.indexer.k_cache": indexer_cls(),
        "d.attn": mla_cls(),
    }
    # Only the MLA layer got a spec.
    proposer, names = _dsa_draft_proposer(layers, {"d.attn": object()})
    with pytest.raises(RuntimeError, match="no KVCacheSpec"):
        proposer._validate_dsa_draft(layers, names)


def test_validate_dsa_draft_accepts_fully_registered_layers():
    indexer_cls, mla_cls = _dsa_layer_classes()
    layers = {
        "d.indexer.k_cache": indexer_cls(),
        "d.attn": mla_cls(),
        # A target layer of the same types must not be pulled in: the check
        # only ever looks at the draft's half of the registry.
        "t.attn": mla_cls(),
    }
    specs = {"d.attn": object(), "d.indexer.k_cache": object()}
    proposer, _ = _dsa_draft_proposer(layers, specs)
    proposer._validate_dsa_draft(layers, {"d.attn", "d.indexer.k_cache"})


def test_index_sharing_rejects_a_topk_buffer_smaller_than_the_loop_bucket():
    """The buffer is sized in tokens but indexed by request in the K-loop."""
    buf = torch.zeros(4, 4, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="topk_indices_buffer"):
        _mtp_proposer_with_buffer(buf,
                                  max_num_reqs=6,
                                  num_tokens_paddings=(2, 4, 8))


def test_validate_dsa_draft_allows_a_single_chip_tp1_run():
    """draft_tp=1 with target_tp=1 sets `_draft_replicated`, but nothing is
    sharded there -- only a draft replicated *below* the target's TP is a
    problem."""
    indexer_cls, mla_cls = _dsa_layer_classes()
    layers = {
        "d.indexer.k_cache": indexer_cls(),
        "d.attn": mla_cls(),
    }
    specs = {"d.attn": object(), "d.indexer.k_cache": object()}
    proposer, names = _dsa_draft_proposer(layers,
                                          specs,
                                          draft_tp=1,
                                          target_tp=1)
    proposer._validate_dsa_draft(layers, names)


def _nextn_layers(num_layers: int, with_shared_head: bool):
    """Nextn layers shaped like DeepSeekMTP's (`shared_head.head` per layer) or
    like Qwen3.5 MTP's (no per-layer head; one top-level lm_head instead)."""
    if with_shared_head:
        return [
            SimpleNamespace(shared_head=SimpleNamespace(head=object()))
            for _ in range(num_layers)
        ]
    return [SimpleNamespace() for _ in range(num_layers)]


def test_validate_mtp_layer_count_rejects_multi_layer_per_layer_heads():
    """DeepSeek-style: the proposer reads logits off layer 0's head at every
    step, and only the layer that ran at step 0 has its top-k buffer written."""
    proposer = _make_proposer(method="mtp")
    proposer.draft_model = SimpleNamespace(model=SimpleNamespace(
        num_mtp_layers=2, layers=_nextn_layers(2, with_shared_head=True)))
    with pytest.raises(NotImplementedError,
                       match="num_nextn_predict_layers=2"):
        proposer._validate_mtp_layer_count()


def test_validate_mtp_layer_count_allows_multi_layer_single_head():
    """Qwen3.5-style: several nextn layers under one top-level lm_head and no
    indexer. Neither failure mode applies, so the count alone must not refuse
    it."""
    proposer = _make_proposer(method="mtp")
    proposer.draft_model = SimpleNamespace(model=SimpleNamespace(
        num_mtp_layers=3, layers=_nextn_layers(3, with_shared_head=False)))
    proposer._validate_mtp_layer_count()


def test_validate_mtp_layer_count_accepts_a_single_layer_checkpoint():
    proposer = _make_proposer(method="mtp")
    proposer.draft_model = SimpleNamespace(model=SimpleNamespace(
        num_mtp_layers=1, layers=_nextn_layers(1, with_shared_head=True)))
    proposer._validate_mtp_layer_count()


def test_validate_mtp_layer_count_is_a_noop_for_a_non_mtp_draft():
    """An eagle3 draft has no nextn layers to count."""
    proposer = _make_proposer(method="eagle3")
    proposer.draft_model = SimpleNamespace(model=SimpleNamespace())
    proposer._validate_mtp_layer_count()


# --------------------------------------------------------------------------
# IndexShare's second compiled program (draft steps 1+)
# --------------------------------------------------------------------------


class _RecordingProgram:
    """Stands in for a compiled draft program and records every call."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def __call__(self, **kwargs):
        self.log.append((self.name, kwargs["spec_step_idx"]))
        return kwargs["hidden_states"]


def _forward_draft_harness(monkeypatch):
    """A proposer whose real _forward_draft runs on plain CPU tensors, with
    contexts and attention metadata stubbed out as in the saved-draft-config
    test. Which program a step runs does not depend on the device."""
    import vllm_torchtpu.spec_decode.eagle3 as e3

    proposer = _make_proposer(draft_tp=1, method="mtp")
    proposer.draft_vllm_config = object()
    proposer.runner = SimpleNamespace(mesh=object(),
                                      _dp_num_tokens_across_dp=lambda n: n)
    proposer._build_draft_attn_metadata = lambda **_kwargs: {}

    def null(*_args, **_kwargs):
        return contextlib.nullcontext()

    monkeypatch.setattr(e3, "set_model_tag", null)
    monkeypatch.setattr(e3, "set_forward_context", null)
    monkeypatch.setattr(e3, "set_vllm_model_wrapper_context", null)

    def run(step_idx):
        return proposer._forward_draft(
            chunk=SimpleNamespace(),
            input_ids=torch.zeros(2, dtype=torch.int32),
            positions=torch.zeros(2, dtype=torch.int32),
            target_hidden_states=torch.zeros((2, 2)),
            step_idx=step_idx,
            seq_lens_delta=step_idx,
            num_rejected_np=None,
        )

    return proposer, run


def test_forward_draft_without_index_share_runs_the_draft_at_every_step(
        monkeypatch):
    """Every other drafter keeps exactly one compiled program: no second
    program is consulted and skip_topk is never touched."""
    proposer, run = _forward_draft_harness(monkeypatch)
    log = []
    draft = _RecordingProgram("draft", log)
    draft.model = mock.MagicMock()
    proposer.draft_model = draft

    for step in range(3):
        run(step)

    assert log == [("draft", 0), ("draft", 1), ("draft", 2)]
    draft.model.set_skip_topk.assert_not_called()


def test_forward_draft_routes_index_share_loop_steps_to_their_own_program(
        monkeypatch):
    """Step 0 runs the draft with the indexer on; steps 1+ run the loop-step
    program with it off. skip_topk is set before each call because a program
    bakes it in on its first run. That first run of the loop-step program is
    repeated once (compile, then run), and never again."""
    proposer, run = _forward_draft_harness(monkeypatch)
    log = []
    fake_mtp = _FakeMtpModel(torch.zeros(8, 4, dtype=torch.int32))
    skips = []
    set_skip = fake_mtp.set_skip_topk

    def record_skip(skip):
        skips.append(skip)
        set_skip(skip)

    fake_mtp.set_skip_topk = record_skip
    draft = _RecordingProgram("draft", log)
    draft.model = fake_mtp
    proposer.draft_model = draft
    proposer._share_mtp_indices = True
    proposer._mtp_loop_model = _RecordingProgram("loop", log)

    for _ in range(2):  # two propose rounds with K = 3
        for step in range(3):
            run(step)

    assert log == [("draft", 0), ("loop", 1), ("loop", 1), ("loop", 2),
                   ("draft", 0), ("loop", 1), ("loop", 2)]
    assert skips == [False, True, True, False, True, True]
    assert fake_mtp.skip_topk is True


def test_loop_step_program_runs_the_drafts_own_forward():
    """With compilation off the program is plain Python, which isolates the
    delegation: the draft's own forward, on the draft object, with every
    argument passed through."""
    from vllm.config import CompilationMode

    from vllm_torchtpu.spec_decode.mtp_loop_model import MtpLoopStepModel

    seen = {}

    class _Mtp(torch.nn.Module):

        def forward(self,
                    input_ids,
                    positions,
                    hidden_states,
                    intermediate_tensors=None,
                    inputs_embeds=None,
                    spec_step_idx=0):
            seen.update(self=self,
                        positions=positions,
                        intermediate=intermediate_tensors,
                        embeds=inputs_embeds,
                        step=spec_step_idx)
            return hidden_states + 1

    mtp = _Mtp()
    config = SimpleNamespace(compilation_config=SimpleNamespace(
        mode=CompilationMode.NONE))
    program = MtpLoopStepModel(vllm_config=config,
                               prefix="mtp_index_share_loop",
                               mtp=mtp)
    positions = torch.arange(2, dtype=torch.int32)

    out = program(input_ids=torch.zeros(2, dtype=torch.int32),
                  positions=positions,
                  hidden_states=torch.zeros(2, 3),
                  spec_step_idx=2)

    assert torch.equal(out, torch.ones(2, 3))
    assert seen["self"] is mtp and seen["positions"] is positions
    assert seen["intermediate"] is None and seen["embeds"] is None
    assert seen["step"] == 2
    assert program.prefix == "mtp_index_share_loop"


def test_build_mtp_loop_model_uses_the_draft_config_and_its_own_prefix(
        monkeypatch):
    """Built like the draft itself: under the draft's config (the compile
    wrapper reads the current one), wrapping the draft, in a cache folder of
    its own."""
    import vllm_torchtpu.spec_decode.eagle3 as e3
    import vllm_torchtpu.spec_decode.mtp_loop_model as mlm

    proposer = _mtp_proposer_with_buffer(torch.zeros(8, 4, dtype=torch.int32))
    proposer.draft_model._dynamic_arg_dims = dict(mlm.MTP_DYNAMIC_ARG_DIMS)
    proposer.draft_vllm_config = object()
    current = []
    built = {}

    @contextlib.contextmanager
    def set_current(config):
        current.append(config)
        yield

    class _Program:

        def __init__(self, **kwargs):
            built.update(kwargs, current=list(current))

    def null(*_args, **_kwargs):
        return contextlib.nullcontext()

    monkeypatch.setattr(e3, "set_model_tag", null)
    monkeypatch.setattr(e3, "set_vllm_model_wrapper_context", null)
    monkeypatch.setattr(e3, "set_current_vllm_config", set_current)
    monkeypatch.setattr(mlm, "MtpLoopStepModel", _Program)

    proposer._build_mtp_loop_model()

    assert isinstance(proposer._mtp_loop_model, _Program)
    assert built["vllm_config"] is proposer.draft_vllm_config
    assert built["mtp"] is proposer.draft_model
    assert built["prefix"] == "mtp_index_share_loop"
    assert built["current"] == [proposer.draft_vllm_config]
    assert proposer._share_mtp_indices is True
    assert proposer._mtp_loop_model_warm is False


def test_index_share_turns_off_when_its_loop_step_program_cannot_be_built():
    """IndexShare is either fully on, with its second program, or off -- never
    left in place doing nothing. A draft whose compiled argument dims differ
    from the loop-step program's is one way the build can fail."""
    proposer = _mtp_proposer_with_buffer(torch.zeros(8, 4, dtype=torch.int32))
    assert proposer._share_mtp_indices is True
    proposer.draft_model._dynamic_arg_dims = {"input_ids": 0}

    proposer._build_mtp_loop_model()

    assert proposer._share_mtp_indices is False
    assert proposer._mtp_topk_buffers == []
    assert proposer._mtp_loop_model is None
    # With sharing off, the toggle no longer reaches the model.
    proposer._mtp_set_skip_topk(True)
    assert proposer.draft_model.model.skip_topk is None
