# SPDX-License-Identifier: Apache-2.0
"""MTP draft step 0 at decode rows (VLLM_TPU_MTP_DRAFT_DECODE_ROWS) and the
draft padding mask (VLLM_TPU_MTP_DRAFT_PADDING_MASK): the row count, the
compiled wrappers' dims and inputs, which program a chunk runs, and the mask.

Ported from origin/caojx/qwen38-1118-offload-fp4-cache d523c2ce0e (the
VLLM_TPU_MTP_DRAFT_CHECK cases dropped; the bitwise real-row checks added)."""

import types

import numpy as np
import pytest
import torch

from vllm_torchtpu.spec_decode import eagle3, mtp_loop_model
from vllm_torchtpu.spec_decode.mtp_loop_model import decode_rows_count


@pytest.mark.parametrize(
    "tokens,rows",
    [(16, 16), (32, 16), (64, 16), (96, 32), (128, 32), (160, 48), (256, 64), (448, 112)],
)
def test_decode_rows_cover_a_pure_decode_step(tokens, rows):
    assert decode_rows_count(tokens, 4) == rows
    assert rows % 16 == 0 and rows * 4 >= tokens and rows <= tokens


class FakeDraft(torch.nn.Module):
    """Evaluates a per-row function of the selected rows (-1: no row)."""

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None,
                inputs_embeds=None, spec_step_idx=0, draft_rows=None):
        x = hidden_states[draft_rows.clamp_min(0)]
        out = torch.tanh(x * 1.7 + 0.3) * x
        return torch.where((draft_rows >= 0)[:, None], out, 0.0), draft_rows


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
    np.testing.assert_array_equal(out[:3], FakeDraft()(None, None, hidden, draft_rows=rows[:3])[0])


@pytest.mark.parametrize("tokens,num_reqs", [(64, 5), (128, 32), (256, 17)])
def test_decode_rows_real_rows_bitwise_capacity(monkeypatch, tokens, num_reqs):
    """The decode-row program's real-row outputs are bitwise those of the
    capacity-row program (the draft's own forward at the capacity rows)."""
    seen = {}
    monkeypatch.setattr(mtp_loop_model, "support_torch_compile", _fake_compile(seen))
    draft = FakeDraft()
    decode = mtp_loop_model.build_mtp_step0_rows_model(
        vllm_config=None, prefix="p", mtp=draft, dynamic_arg_dims=DRAFT_DIMS,
        tokens_per_request=4,
    )
    proposer = eagle3.Eagle3Proposer.__new__(eagle3.Eagle3Proposer)
    capacity = 128
    g = torch.Generator().manual_seed(tokens)
    hidden = torch.randn(tokens, 8, generator=g)
    lti = torch.sort(torch.randperm(tokens, generator=g)[:num_reqs]).values
    cap_rows = proposer._padded_draft_rows(lti, capacity)
    tok_rows = proposer._padded_draft_rows(lti, tokens)
    ids, pos = torch.zeros(tokens), torch.zeros(tokens)
    ref, _ = draft(ids, pos, hidden, draft_rows=cap_rows)
    out, used = decode.forward(ids, pos, hidden, tok_rows)
    assert used.shape[0] == decode_rows_count(tokens, 4) >= num_reqs
    np.testing.assert_array_equal(out[:num_reqs].numpy(), ref[:num_reqs].numpy())


def _proposer(capacity=128, decode=True):
    proposer = eagle3.Eagle3Proposer.__new__(eagle3.Eagle3Proposer)
    proposer.runner = types.SimpleNamespace(
        num_reqs_max_model_len=capacity, num_reqs_most_model_len=None
    )
    proposer.speculative_config = types.SimpleNamespace(num_speculative_tokens=3)
    proposer._draft_rows = True
    proposer._decode_rows_model = object() if decode else None
    proposer._decode_rows_models = {}
    proposer._decode_rows_build = lambda prefix: object()
    proposer._decode_rows_warm = set()
    proposer._draft_step0_warm = False
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


def _step0(proposer, tokens, num_reqs):
    chunk = types.SimpleNamespace(num_reqs=num_reqs)
    lti = torch.arange(num_reqs, dtype=torch.int64)
    capacity_rows = proposer._padded_draft_rows(lti, proposer._draft_rows_len(chunk))
    return proposer._forward_draft_step0(
        chunk, torch.zeros(tokens, dtype=torch.int32), torch.zeros(tokens),
        torch.zeros(tokens, 4), np.zeros(num_reqs), lti, capacity_rows,
    )


def test_decode_step_runs_decode_rows_after_compiling_both():
    proposer, calls = _proposer()
    decode = proposer._decode_rows_model
    _step0(proposer, 256, 64)
    # First call: the capacity-row program once, the decode program twice.
    assert calls == [(None, 128, False), (decode, 256, True), (decode, 256, True)]
    calls.clear()
    _step0(proposer, 256, 48)
    assert calls == [(decode, 256, True)]
    # Another bucket gets its own compiled module (one compile per vLLM
    # backend), warmed once like the first.
    calls.clear()
    _step0(proposer, 128, 32)
    other = proposer._decode_rows_models[128]
    assert other is not decode
    assert calls == [(other, 128, True), (other, 128, True)]
    calls.clear()
    _step0(proposer, 128, 20)
    assert calls == [(other, 128, True)]


def test_steps_outside_the_decode_bound_keep_capacity_rows():
    proposer, calls = _proposer()
    _step0(proposer, 128, 33)  # more requests than 128 tokens' decode rows
    _step0(proposer, 512, 100)  # decode rows 128 = capacity: no bucket
    # The first call of the capacity-row program runs twice (compile).
    assert calls == [(None, 128, False)] * 3
    assert not proposer._decode_rows_serves(512)
    assert proposer._decode_rows_serves(448)


def test_flag_off_keeps_capacity_rows():
    proposer, calls = _proposer(decode=False)
    _step0(proposer, 64, 16)
    assert calls == [(None, 128, False)]


def test_flag_off_build_is_a_no_op(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "VLLM_TPU_MTP_DRAFT_DECODE_ROWS", False)
    proposer, _ = _proposer(decode=False)
    proposer._build_step0_programs()
    assert proposer._decode_rows_model is None


@pytest.mark.parametrize("preflat", [False, True])
def test_draft_padding_mask(monkeypatch, preflat):
    """VLLM_TPU_MTP_DRAFT_PADDING_MASK: rows past the chunk's requests are
    padding for the draft MoE (step 0 at its evaluated rows, the loop at p)."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import token_padding

    monkeypatch.setattr(envs, "VLLM_TPU_PREFLAT_FWD_ARGS", preflat)
    monkeypatch.setattr(envs, "VLLM_TPU_TOKEN_PADDING_REUSE", True)
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
    # The target's next write is not skipped by TOKEN_PADDING_REUSE.
    state.update(96, 128)
    assert state.local_padding_mask.nonzero().flatten().tolist() == list(range(96, 128))
    monkeypatch.setattr(envs, "VLLM_TPU_MTP_DRAFT_PADDING_MASK", False)
    proposer._mark_draft_padding(24, 32)  # off: the target's mask stays
    assert state.local_padding_mask.nonzero().flatten().tolist() == list(range(96, 128))


def test_draft_padding_mask_zeroes_routing_of_padded_rows_only(monkeypatch):
    """zero_routing_weights_for_padding under the draft's mask: real rows
    bitwise unchanged, padded rows zero weight."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import token_padding

    state = token_padding.TokenPaddingState.create(256, "cpu")
    state.update(96, 128)
    monkeypatch.setattr(token_padding, "_padding_state", state)
    monkeypatch.setattr(envs, "VLLM_TPU_MTP_DRAFT_PADDING_MASK", True)
    proposer, _ = _proposer()
    proposer.runner.device = "cpu"
    ids = torch.arange(32 * 8).reshape(32, 8) % 10
    w = torch.rand(32, 8, generator=torch.Generator().manual_seed(0))
    # Without the draft mask the target's mask (96 real) keeps all 32 rows.
    _, w_off = token_padding.zero_routing_weights_for_padding(ids, w, True)
    assert torch.equal(w_off, w)
    proposer._mark_draft_padding(24, 32)
    ids2, w_on = token_padding.zero_routing_weights_for_padding(ids, w, True)
    assert torch.equal(ids2, ids)
    assert torch.equal(w_on[:24], w[:24])
    assert (w_on[24:] == 0).all()


def test_decode_rows_programs_get_their_own_compile_config(monkeypatch):
    """Each decode-row module compiles under its own compilation config with
    no pre-grad pass: by the time it is built the stock draft compile has
    marked the shared inductor config, which a fresh vLLM backend asserts
    absent (configure_post_pass)."""
    import contextlib

    from vllm_torchtpu import envs
    from vllm_torchtpu.spec_decode import mtp_loop_model

    monkeypatch.setattr(envs, "VLLM_TPU_MTP_DRAFT_DECODE_ROWS", True)
    seen = []

    def fake_build(*, vllm_config, prefix, **kw):
        seen.append((prefix, vllm_config))
        return object()

    monkeypatch.setattr(mtp_loop_model, "build_mtp_step0_rows_model", fake_build)
    for name in ("set_model_tag", "set_current_vllm_config"):
        monkeypatch.setattr(eagle3, name, lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(
        eagle3, "set_vllm_model_wrapper_context", lambda **k: contextlib.nullcontext()
    )
    proposer, _ = _proposer(decode=False)
    proposer._decode_rows_models = {}
    proposer.draft_model = types.SimpleNamespace(_dynamic_arg_dims={"input_ids": 0})
    proposer.runner.mesh = None
    shared_icc = {"pre_grad_custom_pass": object(), "_vllm_model_tag": "eagle_head"}
    shared_cc = types.SimpleNamespace(inductor_compile_config=shared_icc)
    proposer.draft_vllm_config = types.SimpleNamespace(compilation_config=shared_cc)
    proposer._build_step0_programs()
    proposer._decode_rows_model_for(256)
    proposer._decode_rows_model_for(128)
    assert len(seen) == 2
    for _, cfg in seen:
        icc = cfg.compilation_config.inductor_compile_config
        assert "pre_grad_custom_pass" not in icc
        assert icc["_vllm_model_tag"] == "eagle_head"
        assert cfg.compilation_config is not shared_cc
    # The shared config is left as it was.
    assert "pre_grad_custom_pass" in shared_icc
