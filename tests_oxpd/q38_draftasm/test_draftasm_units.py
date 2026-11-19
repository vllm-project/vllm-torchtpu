# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT-ASM: each folded program against the base program(s) it replaces, bit for bit (CPU; the
compiled bodies run eagerly). Rows 4 / 8 / 12 / 16 / 32 / 48 / 64, K = 3, fallback rows on and off.

1. tail_bonus_rejected + fold_source == ``_spec_tail`` (the compiled spec tail, REJ_PAD) on the exact-count
   drafts the base path stacks and concatenates.
2. The folded seed's first five outputs == ``_async_seed_inputs_impl``'s; its step-1 seq_lens == the base
   loop's eager ``seq_lens + 1 - rejected`` (eagle3 ``_loop_step_attn_inputs`` itself); its bonus / rejected
   rows == item 1's.
3. Both folded carries == the base carries, plus ``positions_carry + 1``; the fresh hidden carry is a buffer
   of its own.
4. ``_MtpFusedLoopAsm`` returns ``_MtpFusedLoop``'s tokens (and advances seq_lens the same way), then the
   source of item 1.
"""

import numpy as np
import pytest
import torch

import draftasm_harness as h
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len
from vllm_torchtpu.spec_decode import draft_asm, eagle3
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer
from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop

ROWS = [4, 8, 12, 16, 32, 48, 64]
BOUND = h.MAX_REQS * (1 + h.K)


def _loop_p(R):
    return _get_padded_token_len(h.PADDINGS, R)


@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("R", ROWS)
def test_tail_and_source_equal_the_spec_tail(R, fallback, short):
    step = h.make_step(100 + R, R, fallback=fallback, short=short)
    n_dev = torch.tensor(R, dtype=torch.int64)
    bonus, rej = draft_asm.tail_bonus_rejected(step.nt, step.chunk.draft_lengths, n_dev, h.MAX_REQS)
    g = np.random.default_rng(R + 7)
    p = _loop_p(R)
    token0 = torch.from_numpy(g.integers(1, h.V, p).astype(np.int32))
    tokens = tuple(torch.from_numpy(g.integers(1, h.V, p).astype(np.int32)) for _ in range(h.K - 1))
    src = draft_asm.fold_source(token0, tokens, bonus, n_dev, h.MAX_REQS)
    # the base path: stack, slice to the live rows, concatenate, then the compiled spec tail
    drafts = torch.cat([torch.stack((token0, *tokens), dim=1)[:R]], dim=0)
    want_src, want_rej = tpu_runner._spec_tail(
        step.nt, drafts, step.chunk.draft_lengths, R, R, BOUND, h.MAX_REQS
    )
    h.same(src, want_src, f"source R={R}")
    h.same(rej, want_rej, f"rejected R={R}")
    print(f"\nDRAFTASM_BITWISE tail R={R} fallback={int(fallback)} short={int(short)} "
          f"source {src.numel()} rejected {rej.numel()} equal")


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("R", ROWS)
def test_seed_fold_equals_base_seed_and_loop_seq_lens(R, fallback):
    cap = h.MAX_REQS
    step = h.make_step(200 + R, R, cap=cap, fallback=fallback)
    c = step.chunk
    n_dev = torch.tensor(R, dtype=torch.int64)
    p = _loop_p(R)
    base = eagle3._async_seed_inputs_impl(
        step.nt, c.draft_lengths, c.attn_ctx.query_start_loc, n_dev, c.input_ids, p, cap
    )
    asm = draft_asm._async_seed_inputs_asm_impl(
        step.nt, c.draft_lengths, c.attn_ctx.query_start_loc, n_dev, c.input_ids, c.attn_ctx.seq_lens,
        p, cap, h.MAX_REQS,
    )
    assert len(base) == 5 and len(asm) == 8
    for i in range(5):
        h.same(asm[i], base[i], f"seed output {i} R={R}")
    prop = h.make_proposer(False, cap=cap)
    loop_qsl = torch.minimum(torch.arange(cap + 1, dtype=torch.int32), torch.tensor(R, dtype=torch.int32))
    seq_base, _, _ = Eagle3Proposer._loop_step_attn_inputs(
        prop, c, seq_lens_delta=1, num_rejected_np=None, num_rejected_dev=base[4],
        loop_query_start_loc=loop_qsl, loop_request_distribution=torch.tensor([R] * 3, dtype=torch.int32),
    )
    h.same(asm[5], seq_base, f"step-1 seq_lens R={R}")
    bonus, rej = draft_asm.tail_bonus_rejected(step.nt, c.draft_lengths, n_dev, h.MAX_REQS)
    h.same(asm[6], bonus, f"bonus R={R}")
    h.same(asm[7], rej, f"rejected R={R}")
    print(f"\nDRAFTASM_BITWISE seed R={R} fallback={int(fallback)} base5 + seq_lens_step1 + bonus + rejected equal")


@pytest.mark.parametrize("R", ROWS)
def test_carry_folds_equal_base_carries_plus_one(R):
    cap = h.MAX_REQS
    prop = h.make_proposer(False, cap=cap)
    g = np.random.default_rng(300 + R)
    p = _loop_p(R)
    T = _get_padded_token_len(h.PADDINGS, R * (1 + h.K))
    positions = torch.from_numpy(g.integers(0, 50000, T).astype(np.int32))
    idx = torch.from_numpy(g.integers(-1, T, p).astype(np.int64))
    full = torch.from_numpy(g.standard_normal((cap, h.H)).astype(np.float32))
    full2 = torch.from_numpy(g.standard_normal((cap, h.H)).astype(np.float32))
    base = h.unwrap(Eagle3Proposer._draft_rows_carry_token_fresh)(prop, positions, idx, full, full2)
    asm = draft_asm._carry_token_fresh_asm_impl(prop, positions, idx, full, full2)
    for i in range(3):
        h.same(asm[i], base[i], f"fresh carry output {i} R={R}")
    h.same(asm[3], base[0] + 1, f"fresh step-1 positions R={R}")
    assert asm[2].untyped_storage().data_ptr() != full2.untyped_storage().data_ptr()
    rows = full2[:p]
    base_p = h.unwrap(Eagle3Proposer._draft_rows_carry_and_token)(prop, positions, idx, rows)
    asm_p = draft_asm._carry_and_token_asm_impl(prop, positions, idx, rows)
    h.same(asm_p[0], base_p[0], f"plain carry positions R={R}")
    h.same(asm_p[1], base_p[1], f"plain carry token R={R}")
    h.same(asm_p[2], base_p[0] + 1, f"plain step-1 positions R={R}")
    print(f"\nDRAFTASM_BITWISE carry R={R} fresh 3 + positions_step1, plain 2 + positions_step1 equal")


@pytest.mark.parametrize("R", ROWS)
def test_folded_loop_equals_base_loop_plus_source(R):
    cap = h.MAX_REQS
    mtp = h.FakeMtp()
    base = _MtpFusedLoop(vllm_config=None, prefix="b", mtp=mtp, num_steps=h.K - 1)
    asm = draft_asm._MtpFusedLoopAsm(vllm_config=None, prefix="a", mtp=mtp, num_steps=h.K - 1,
                                     max_reqs=h.MAX_REQS)
    g = np.random.default_rng(400 + R)
    p = _loop_p(R)
    ids = torch.from_numpy(g.integers(1, h.V, p).astype(np.int32))
    pos = torch.from_numpy(g.integers(0, 50000, p).astype(np.int32))
    hid = torch.from_numpy(g.standard_normal((p, h.H)).astype(np.float32))
    seq = torch.from_numpy(g.integers(100, 60000, cap).astype(np.int32))
    seq_a, seq_b = seq.clone(), seq.clone()
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

    n_dev = torch.tensor(R, dtype=torch.int64)
    bonus = torch.from_numpy(g.integers(1, h.V, h.MAX_REQS).astype(np.int32))
    with h.fake_forward_context({"draft.attn.0": AttentionMetadata(input_positions=None, seq_lens=seq_a)}):
        tb = base(ids, pos, hid, seq_a)
    with h.fake_forward_context({"draft.attn.0": AttentionMetadata(input_positions=None, seq_lens=seq_b)}):
        ta = asm(ids, pos, hid, seq_b, bonus, n_dev)
    assert len(tb) == h.K - 1 and len(ta) == h.K
    for i in range(h.K - 1):
        h.same(ta[i], tb[i], f"loop token {i + 1} R={R}")
    h.same(seq_b, seq_a, f"seq_lens after the loop R={R}")
    h.same(ta[-1], draft_asm.fold_source(ids, tb, bonus, n_dev, h.MAX_REQS), f"loop source R={R}")
    assert ta[-1].shape == (BOUND,)
    print(f"\nDRAFTASM_BITWISE loop R={R} tokens {h.K - 1} + seq_lens + source[{BOUND}] equal")
