# SPDX-License-Identifier: Apache-2.0
"""MTP draft step 0 at decode rows (VLLM_TPU_MTP_DRAFT_DECODE_ROWS): the row
count, the compiled wrappers' dims and inputs, and which program a chunk runs."""

import types

import numpy as np
import pytest
import torch

from vllm_torchtpu.spec_decode import eagle3, mtp_fused_loop, mtp_loop_model
from vllm_torchtpu.spec_decode.mtp_loop_model import decode_rows_count


@pytest.mark.parametrize(
    "tokens,rows",
    [(16, 16), (32, 16), (64, 16), (96, 32), (128, 32), (160, 48), (256, 64), (448, 112)],
)
def test_decode_rows_cover_a_pure_decode_step(tokens, rows):
    assert decode_rows_count(tokens, 4) == rows
    assert rows % 16 == 0 and rows * 4 >= tokens and rows <= tokens


class FakeDraft(torch.nn.Module):
    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None,
                inputs_embeds=None, spec_step_idx=0, draft_rows=None):
        return hidden_states[draft_rows.clamp_min(0)], draft_rows


def _fake_compile(seen):
    def fake_support_torch_compile(*, dynamic_arg_dims):
        seen["dims"] = dict(dynamic_arg_dims)

        def wrap(cls):
            seen["forward"] = cls.forward.__qualname__
            return lambda **kwargs: cls(**kwargs)

        return wrap

    return fake_support_torch_compile


DRAFT_DIMS = {"input_ids": 0, "positions": 0, "hidden_states": 0, "intermediate_tensors": 0}


@pytest.mark.parametrize("tokens_per_request", [0, 4])
def test_step0_wrappers(monkeypatch, tokens_per_request):
    seen = {}
    monkeypatch.setattr(mtp_loop_model, "support_torch_compile", _fake_compile(seen))
    model = mtp_loop_model.build_mtp_step0_rows_model(
        vllm_config=None, prefix="p", mtp=FakeDraft(), dynamic_arg_dims=DRAFT_DIMS,
        tokens_per_request=tokens_per_request,
    )
    want = {"input_ids": 0, "positions": 0, "hidden_states": 0}
    tokens = 160
    rows = torch.full((tokens,), -1, dtype=torch.int64)
    rows[:3] = torch.tensor([3, 7, 11])
    hidden = torch.arange(tokens, dtype=torch.float32)[:, None]
    out, used = model.forward(torch.zeros(tokens), torch.zeros(tokens), hidden, rows)
    if tokens_per_request:
        want["draft_rows"] = 0
        assert seen["forward"] == "MtpStep0DecodeRows.forward"
        assert used.shape[0] == 48
    else:
        assert seen["forward"] == "MtpStep0CapacityRows.forward"
        assert used.shape[0] == tokens
    assert seen["dims"] == want
    assert out[:3, 0].tolist() == [3.0, 7.0, 11.0]


def test_check_loop_has_its_own_forward(monkeypatch):
    seen = {}
    monkeypatch.setattr(mtp_fused_loop, "support_torch_compile", _fake_compile(seen))
    for reference, name in ((False, "_MtpFusedLoop.forward"),
                            (True, "_MtpCheckFusedLoop.forward")):
        mtp_fused_loop.build_mtp_fused_loop(
            vllm_config=None, prefix="p", mtp=None, num_steps=2,
            dynamic_arg_dims=DRAFT_DIMS, reference=reference,
        )
        assert seen["forward"] == name


def _proposer(capacity=128, decode=True):
    proposer = eagle3.Eagle3Proposer.__new__(eagle3.Eagle3Proposer)
    proposer.runner = types.SimpleNamespace(
        num_reqs_max_model_len=capacity, num_reqs_most_model_len=None
    )
    proposer.speculative_config = types.SimpleNamespace(num_speculative_tokens=3)
    proposer._draft_rows = True
    proposer._decode_rows_model = object() if decode else None
    proposer._decode_rows_warm = False
    proposer._draft_step0_warm = False
    proposer._check_left = 0
    proposer._check_step0 = object()
    proposer._check_warm = [False, False]
    proposer._check_decode_calls = 0
    proposer._draft_rows_len = lambda chunk: capacity
    calls = []

    def fake_forward_draft(**kw):
        rows = kw["draft_rows"]
        if rows is not None:
            # Rows past the chunk's requests are -1: no row, zero length.
            real = int((rows >= 0).sum())
            assert rows[:real].tolist() == list(range(real))
            assert (rows[real:] == -1).all(), rows
        calls.append((kw["step0_model"], None if rows is None else rows.shape[0],
                      kw["serves"] is not None))
        return "last", "hidden"

    proposer._forward_draft = fake_forward_draft
    return proposer, calls


def _step0(proposer, tokens, num_reqs, program="auto"):
    chunk = types.SimpleNamespace(num_reqs=num_reqs)
    lti = torch.arange(num_reqs, dtype=torch.int64)
    return proposer._forward_draft_step0(
        chunk, torch.zeros(tokens, dtype=torch.int32), torch.zeros(tokens),
        torch.zeros(tokens, 4), np.zeros(num_reqs), lti, program=program,
    )


def test_decode_step_runs_decode_rows_after_compiling_both():
    proposer, calls = _proposer()
    decode = proposer._decode_rows_model
    _step0(proposer, 256, 64)
    # First call: the capacity-row program once, the decode program twice.
    assert calls == [(None, 128, False), (decode, 256, True), (decode, 256, True)]
    calls.clear()
    _step0(proposer, 128, 32)
    assert calls == [(decode, 128, True)]


def test_steps_outside_the_decode_bound_keep_capacity_rows():
    proposer, calls = _proposer()
    _step0(proposer, 128, 33)  # more requests than 128 tokens' decode rows
    _step0(proposer, 512, 100)  # decode rows 128 = capacity: no bucket
    # The first call of the capacity-row program runs twice (compile).
    assert calls == [(None, 128, False)] * 3
    assert not proposer._decode_rows_serves(512)
    assert proposer._decode_rows_serves(448)


def test_references_run_capacity_rows():
    proposer, calls = _proposer()
    _step0(proposer, 64, 16, program="check")
    check = proposer._check_step0
    assert calls == [(check, 128, False), (check, 128, False)]
    calls.clear()
    _step0(proposer, 64, 16, program="capacity")
    _step0(proposer, 64, 16, program="capacity")
    assert calls == [(None, 128, False)] * 3


def test_flag_off_keeps_capacity_rows():
    proposer, calls = _proposer(decode=False)
    proposer._check_step0 = None
    _step0(proposer, 64, 16)
    assert calls == [(None, 128, False)]


def test_check_counts_each_pair():
    proposer, _ = _proposer()
    proposer.runner.device = "cpu"
    proposer._check_mismatch = None
    proposer._check_rows = 0
    proposer._check_calls = 0
    proposer._check_left = 5
    check = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]], dtype=torch.int32)
    capacity = check.clone()
    capacity[1, 2] = 60  # the skip pair differs in request 1, position 2
    new = capacity.clone()
    new[0, 0] = 10  # the rows pair differs in request 0, position 0
    new[3, :] = 99  # padding row (past num_reqs): not counted
    proposer._check_draft_tokens(
        [(check, capacity)], [new], [types.SimpleNamespace(num_reqs=3)]
    )
    skip, rows, total = proposer._check_mismatch.tolist()
    assert skip == [0, 0, 1, 1]
    assert rows == [1, 0, 0, 1]
    assert total == [1, 0, 1, 2]
    assert proposer._check_rows == 3 and proposer._check_left == 4


def test_draft_padding_mask(monkeypatch):
    """VLLM_TPU_MTP_DRAFT_PADDING_MASK: rows past the chunk's requests are
    padding for the draft MoE (step 0 at its evaluated rows, the loop at p)."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import token_padding

    state = token_padding.TokenPaddingState.create(256, "cpu")
    state.update(96, 128)  # the target step: 96 real tokens of a 128 bucket
    monkeypatch.setattr(token_padding, "_padding_state", state)
    monkeypatch.setattr(envs, "VLLM_TPU_MTP_DRAFT_PADDING_MASK", True)
    monkeypatch.setattr(envs, "VLLM_TPU_PROPOSE_DEVICE_CONSTANTS", True)
    proposer, calls = _proposer()
    proposer.runner.device = "cpu"
    seen = []

    def fake_forward_draft(**kw):
        seen.append(state.local_padding_mask.clone())
        return "last", "hidden"

    proposer._forward_draft = fake_forward_draft
    _step0(proposer, 128, 24)  # capacity warm call, then the decode program twice
    for mask, rows in zip(seen, (128, 32, 32)):
        assert not mask[:24].any(), "real rows keep their router weight"
        assert mask[24:rows].all(), "padded draft rows get none"
        assert not mask[rows:].any()
    proposer._mark_draft_padding(24, 32)  # the loop at p = 32
    assert state.local_padding_mask.nonzero().flatten().tolist() == list(range(24, 32))
    monkeypatch.setattr(envs, "VLLM_TPU_MTP_DRAFT_PADDING_MASK", False)
    state.update(96, 128)
    proposer._mark_draft_padding(24, 32)  # off: the target's mask stays
    assert state.local_padding_mask.nonzero().flatten().tolist() == list(range(96, 128))
