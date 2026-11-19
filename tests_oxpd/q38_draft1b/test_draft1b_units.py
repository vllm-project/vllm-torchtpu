# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT1-B units (spec_decode/draft_one.py), CPU only.

1. carry_rows == eagle3's fresh carry ([:p].clone()) and plain carry (identity / zero pad), bit for bit,
   including NaN, -0.0 and bf16 rows.
2. The superset seed == DRAFT-ASM's folded seed plus eagle3's position carry and + 1.
3. propose_tokens == eagle3's carry token rule (compute_logits argmax; get_top_tokens under dvp).
4. Stage A's program body == eagle3's carry, then _MtpFusedLoop, then DRAFT-ASM's source.
5. The agreement entry's stage bits: packing and the all-rank rule.
"""

import pytest
import torch

import draft1b_harness as h
from vllm_torchtpu.spec_decode import draft_asm, draft_one, eagle3
from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop


@pytest.mark.parametrize("n,rows", [(64, 16), (64, 48), (64, 64), (40, 48), (16, 16), (48, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_carry_rows_is_the_fresh_and_plain_carry(n, rows, dtype):
    g = torch.Generator().manual_seed(n * 100 + rows)
    x = torch.randn(n, 24, generator=g).to(dtype)
    x[3, 5] = float("nan")
    x[min(n - 1, 7), 1] = -0.0
    got = draft_one.carry_rows(x, rows)
    if rows <= n:
        want = x[:rows].clone()  # eagle3 _draft_rows_carry_token_fresh / plain identity
    else:
        want = eagle3.Eagle3Proposer._fit_draft_rows_output(x, rows)  # eagle3's pad
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
                       want.view(torch.int16 if dtype == torch.bfloat16 else torch.int32))


@pytest.mark.parametrize("R", [3, 12, 40, 64])
@pytest.mark.parametrize("fallback", [False, True])
def test_superset_seed_is_draft_asm_seed_plus_position_carry(R, fallback):
    step = h.make_step(21000 + R, R, fallback=fallback)
    c = step.chunk
    p = 16 if R <= 16 else (32 if R <= 32 else (48 if R <= 48 else 64))
    n_dev = torch.tensor(R, dtype=torch.int64)
    base = (step.nt, c.draft_lengths, c.attn_ctx.query_start_loc, n_dev, c.input_ids, c.attn_ctx.seq_lens)
    want = draft_asm._async_seed_inputs_asm_impl(*base, p, 64, h.MAX_REQS)
    got = draft_one._seed_cl_impl(*base, c.position_ids, p, 64, h.MAX_REQS)
    assert len(got) == len(want) + 2
    for i, (a, b) in enumerate(zip(got, want)):
        h.same(a, b, f"seed output {i}")
    idx = want[1].clamp_min(0)
    h.same(got[draft_one.SEED_POSC], c.position_ids.index_select(0, idx), "position carry")
    h.same(got[draft_one.SEED_POS1], c.position_ids.index_select(0, idx) + 1, "step 1 positions")


class _Lp:
    def get_top_tokens(self, head, hidden):
        return (hidden.float() @ head).argmax(dim=-1)


class _Mtp:
    def __init__(self):
        g = torch.Generator().manual_seed(5)
        self.lm_head = torch.randn(16, 97, generator=g)
        self.logits_processor = _Lp()
        self.draft_id_to_target_id = None

    def compute_logits(self, h):
        return torch.sin(h.float()) @ self.lm_head


@pytest.mark.parametrize("dvp", [False, True])
def test_propose_tokens_is_the_carry_token_rule(dvp, monkeypatch):
    from types import SimpleNamespace

    mtp = _Mtp()
    hidden = torch.randn(48, 16, generator=torch.Generator().manual_seed(9))
    prop = SimpleNamespace(draft_model=mtp, _dvp=dvp)
    if dvp:
        monkeypatch.setattr(eagle3.vps_wiring, "draft_top_tokens",
                            lambda m, hh: m.logits_processor.get_top_tokens(m.lm_head, hh).to(torch.int32))
    _, want = eagle3.Eagle3Proposer._carry_and_token_body(prop, torch.arange(48, dtype=torch.int32),
                                                          torch.arange(48), hidden)
    got = draft_one.propose_tokens(mtp, hidden, dvp)
    h.same(got, want, f"token dvp={dvp}")


@pytest.mark.parametrize("p,cap", [(16, 64), (48, 64), (64, 64)])
def test_carry_loop_body_is_carry_then_loop_then_source(p, cap):
    g = torch.Generator().manual_seed(p + cap)
    mtp = h.FakeMtp2()
    loop = _MtpFusedLoop(vllm_config=None, prefix="t", mtp=mtp, num_steps=h.K - 1)
    last_hidden = torch.randn(cap, h.H, generator=g)
    hidden = torch.randn(cap, h.H, generator=g)
    positions1 = torch.randint(0, 9999, (p,), generator=g, dtype=torch.int32)
    seq = torch.randint(100, 9999, (cap,), generator=g, dtype=torch.int32)
    qsl = torch.arange(cap + 1, dtype=torch.int32).clamp(max=p - 3)
    rd = torch.tensor([p - 3] * 3, dtype=torch.int32)
    bonus = torch.randint(1, 97, (h.MAX_REQS,), generator=g, dtype=torch.int32)
    n = torch.tensor(p - 3, dtype=torch.int64)
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

    def md(sl):
        return {"d": AttentionMetadata(input_positions=positions1, block_tables=torch.arange(4), seq_lens=sl,
                                       query_start_loc=qsl, request_distribution=rd)}

    # reference: eagle3's carry (fresh at p < cap: rows trimmed; plain at p == cap), _MtpFusedLoop, source
    s1 = seq.clone()
    with h.h.fake_forward_context(md(s1)):
        token0 = mtp.compute_logits(last_hidden[:p]).argmax(dim=-1).to(torch.int32)
        toks = _MtpFusedLoop.forward(loop, token0, positions1, hidden[:p].clone(), s1)
    want = (token0, *toks, draft_asm.fold_source(token0, toks, bonus, n, h.MAX_REQS))
    s2 = seq.clone()
    with h.h.fake_forward_context(md(s2)):
        got = draft_one.carry_loop_body(loop, positions1, last_hidden, hidden, s2, bonus, n, h.MAX_REQS)
    assert len(got) == len(want) == h.K + 1
    for i, (a, b) in enumerate(zip(got, want)):
        h.same(a, b, f"output {i}")
    h.same(s2, s1, "seq_lens advanced in place the same")


def _ctl(monkeypatch, cl=True, one=False, cl_ab=None):
    h.install(monkeypatch, fold=True, cl=cl, one=one, cl_ab=cl_ab)
    p = h.make_proposer()
    return p._draft_asm


def test_agreement_bits_pack_and_need_every_rank(monkeypatch):
    ctl = _ctl(monkeypatch)
    vec = ctl.local_vector()
    assert vec[0] == draft_one.BIT_FOLD | draft_one.BIT_CL and len(vec) == 5
    ctl.window_on = False
    assert ctl.local_vector()[0] == 0  # the external clock clears the fold and the stage wishes
    ctl.window_on = True
    fp = ctl._fingerprint()
    both = [draft_one.BIT_FOLD | draft_one.BIT_CL, 0, 0, 0, fp]
    fold_only = [draft_one.BIT_FOLD, 0, 0, 0, fp]
    ctl.apply([both] * 8)
    assert ctl.agreed and ctl.cl_agreed
    ctl.apply([both] * 7 + [fold_only])  # one rank's stage wish off: the fold stays, stage A goes
    assert ctl.agreed and not ctl.cl_agreed
    ctl.apply([both] * 8)
    assert ctl.cl_agreed
    ctl.apply([both] * 7 + [[draft_one.BIT_CL, 0, 0, 0, fp]])  # one rank's fold wish off: neither
    assert not ctl.agreed and not ctl.cl_agreed and not ctl.failed
    ctl.apply([both] * 7 + [[3, 0, 0, 0, fp + 1]])  # unequal warm keys: base for the run
    assert ctl.failed and not ctl.agreed and not ctl.cl_agreed


def test_stage_configuration_and_windows(monkeypatch):
    c = _ctl(monkeypatch, cl=True)
    assert c.cl_conf and not c.one_conf and c.seed_cl and not c.dha_needed
    c = _ctl(monkeypatch, cl=True, cl_ab="0")
    assert not c.cl_conf and not c.seed_cl and c.dha_needed
    c = _ctl(monkeypatch, cl=True, cl_ab="ab:1")
    assert c.cl_conf and c.dha_needed
    assert c.local_vector()[0] == draft_one.BIT_FOLD  # "ab:" with no l6_ab lever: stage off, one ERROR
    c = _ctl(monkeypatch, cl=False, one=True)
    assert c.one_conf and c.seed_cl and c.dha_needed and not c.cl_conf
