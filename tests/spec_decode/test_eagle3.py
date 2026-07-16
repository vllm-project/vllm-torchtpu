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

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from vllm_torchtpu.spec_decode.eagle3 import (DraftChunkInputs, Eagle3Proposer,
                                              _force_draft_tp1,
                                              _maybe_pad_dim0)


def _make_proposer(draft_tp: int | None = 1,
                   target_tp: int = 1,
                   method: str = "eagle3") -> Eagle3Proposer:
    speculative_config = SimpleNamespace(draft_tensor_parallel_size=draft_tp,
                                         method=method)
    parallel_config = SimpleNamespace(tensor_parallel_size=target_tp)
    vllm_config = SimpleNamespace(speculative_config=speculative_config,
                                  parallel_config=parallel_config)
    return Eagle3Proposer(runner=mock.MagicMock(), vllm_config=vllm_config)


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
        with pytest.raises(RuntimeError):
            with _force_draft_tp1():
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


@pytest.mark.parametrize("draft_tp", [2, 4])
def test_draft_tp_invalid_raises(draft_tp):
    # Only replicated (1) or fully sharded (== target tp) are supported.
    with pytest.raises(ValueError):
        _make_proposer(draft_tp=draft_tp, target_tp=8)


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
                draft_lengths=None):
    """Build a DraftChunkInputs for tests. aux/attn_ctx are only needed by
    paths that consume them; _prepare_draft_inputs does not touch attn_ctx, so
    it defaults to an inert placeholder. Pass a real attn_ctx for the propose
    loop, which reads attn_ctx.use_max_model_len."""
    padded_tokens = padded_tokens or input_ids.shape[0]
    return DraftChunkInputs(
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


def test_prepare_draft_inputs(device):
    proposer = _make_proposer(draft_tp=1)
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r0", "r1"]),
        device=device,
        requests={},
        num_tokens_paddings=[16, 32, 64, 128],
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
    rep.runner = SimpleNamespace(num_tokens_paddings=[16, 32, 64, 128])
    assert rep._loop_bucket(1) == 16
    assert rep._loop_bucket(33) == 64


def test_run_dp_dummy_draft_sharded_replays_propose_trace(monkeypatch):
    """Sharded draft under EP-DP lockstep: each padding chunk must replay the
    real propose's collective trace — combine (1/chunk), first-pass forward
    @bucket, K lm-head gathers @constant loop bucket, K-1 loop forwards @that
    bucket — not just K bare forwards."""
    import vllm_torchtpu.spec_decode.eagle3 as e3
    monkeypatch.setattr(e3.sync, "synchronize", lambda *a, **k: None)

    proposer = _make_proposer(draft_tp=2, target_tp=2)
    proposer.speculative_config.num_speculative_tokens = 3
    proposer.runner = SimpleNamespace(
        _dp_target_bucket=64,
        num_reqs_max_model_len=16,
        max_num_reqs=16,
        num_tokens_paddings=[16, 32, 64],
        device=torch.device("cpu"),
        _hidden_states_dtype=torch.float32,
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
