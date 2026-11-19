# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v6 composition (q38-l4-v6c2 + FOLDFIX 2fde9f5e8): T1-K1FUSE's take through the REAL draft path, VFY-ONE's
verify (with K1's hook) -> Eagle3Proposer.propose (K1's take) -> SpeculativeDecodingManager.propose_draft_token_ids ->
the runner's sample_tokens consumer -> the next step's input ids. CDK CPU pod only. V6COMP-BUILD (helper of L4-E).

Rig: K1FUSE's harness (tests/q38_k1fuse: the real TPUModelRunner._sample_spec_verify_vps -> _vfy_verify on a stub
runner of the real class, the real propose; DRAFT-CAP's per-group metadata via tests/q38_l4_v5c/test_v6c_k1.py) with
FOLDFIX's runner pieces (tests/q38_l4_foldfix/foldfix_harness.py: the sample_tokens segment read from this tree's
tpu_runner.py, the real manager, the substitution program as its body). Configuration K (VRNG off, K1 on, VFY-ONE 2,
DH-D; DRAFT-CAP on unless stated). Every step is bitwise against the same step through DraftOne (today's VFY + DH-D
chain) on the same path: drafts, the parked source and rejected count, both index dicts, the placeholder advance, the
next step's substituted input ids, the read offsets, stats and draft KV writes.

KR  16 serving steps at 8 / 20 / 33 / 64 rows: cap_nocheck (DRAFT-CAP, K1 CHECK 0: K1 from the first step), cap_check4
    (K1 CHECK 4 with DRAFT-ASM's boundary every 4 engine steps: CHECK through the manager, then K1), nocap_nocheck_f8
    (DH-D at the capacity, 8 fallback rows). On every K1 step the manager returns the K1Result itself (result type
    K1Result, a DraftAsmResult), the consumer parks its source as is (the parked tensor IS K1's source), and no spec-tail
    program runs. With the FOLDFIX change reverted (the manager's old assert) the first K1 step raises.
Run: python -m pytest -q -p no:cacheprovider -p no:logging -s tests/q38_l4_v5c/test_v6c_foldfix_k1.py
"""

import pathlib
import sys
from collections import Counter
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "q38_l4_foldfix"))
import foldfix_harness as fh  # noqa: E402
import test_v6c_k1 as v6k  # noqa: E402
from test_v6c_k1 import hc, kh  # noqa: E402

from vllm_torchtpu.runner import tpu_runner  # noqa: E402
from vllm_torchtpu.runner.speculative_decoding_manager import SpeculativeDecodingManager  # noqa: E402
from vllm_torchtpu.spec_decode import draft_asm, k1fuse  # noqa: E402

RS = (8, 20, 33, 64)
N = 16
COUNTS: Counter = Counter()


def _rig(p):
    """FOLDFIX's runner pieces on K1FUSE's stub runner (fh.make_rig's attributes, the VPS / VFY state kept)."""
    r = p.runner
    r.scheduler_config = NS(async_scheduling=True)
    r.speculative_config.use_eagle = lambda: True
    r._pcp_enabled = False
    r.vllm_config = NS(kv_transfer_config=None)
    r._stage_timing = fh._Timing()
    r._is_dspark_pd_producer = False
    r._mtp_prefill_only = False
    r.is_pooling_model = False
    r.maybe_wait_for_kv_save = lambda: None
    r.get_finished_kv_transfers = lambda scheduler_output: (None, None, None, None, None, None)
    r.prev_waits = []
    r._modify_prev_results = lambda: r.prev_waits.append(1)
    r._pre_async_results = None
    r._tp_rank0_mask = None
    r._last_sequence_layout_plan = None
    r._fast_token_substitution = False
    r.spec_decode_manager = SpeculativeDecodingManager(r)
    return r


def _step(p, vs, F, seed):
    """One serving engine step: the clock tick, the real verify (K1's hook), then the runner segment (propose ->
    manager -> consumer) and the next step's input ids."""
    r = p.runner
    r.warmup_done = True
    r._l4_agree.tick(r)
    n = vs.R
    req_ids = [f"r{i}" for i in range(n)]
    r.input_batch = NS(num_reqs=n, req_ids=list(req_ids), req_id_to_index={x: i for i, x in enumerate(req_ids)},
                       num_tokens_no_spec=np.full(kh.MAX_REQS, 4000, dtype=np.int32))
    vs.step.chunk.start_index = 0
    kh.CEN.reset()
    with kh.CEN:
        kh.CEN.on = True
        try:
            nt, gen = kh.verify(r, vs, F, seed)
            p.draft_chunks = [vs.step.chunk]
            rsl = [(i, NS(req_id=x, output_token_ids=[]), 0, x) for i, x in enumerate(req_ids)]
            so = NS(has_structured_output_requests=False)
            state = NS(num_reqs_list=[n], spec_decode_metadata_list=[vs.md])
            r._pre_async_results = None
            loc = fh.segment()(r, [nt[:, 0].clone()], [n], True, [], so, [nt], req_ids, False, None, state, rsl, [], {})
        finally:
            kh.CEN.on = False
    d = loc["eagle3_drafts"]
    pre = r._pre_async_results
    rec = NS(nt=nt, gen=gen.get_state(), d=d, result_type=type(d).__name__, calls=dict(kh.CEN.calls),
             drafts=d.drafts() if isinstance(d, draft_asm.DraftAsmResult) else d, pre=pre,
             src=pre.next_tokens_tpu, rejected=pre.spec_decode_num_rejected_tokens, idx=dict(pre.req_id_to_index_copy),
             ndpr=dict(pre.num_draft_per_req or {}),
             placeholders=(r.input_batch.num_tokens_no_spec[:n].copy(), [len(s.output_token_ids) for _, s, _, _ in rsl]),
             next_ids=fh.next_step_ids(r, n))
    return rec


def _same(a, b, ra, rb, tag):
    kh.same(a.nt, b.nt, f"{tag} tokens")
    assert torch.equal(a.gen, b.gen), f"{tag} random stream"
    for what in ("drafts", "src", "rejected", "next_ids"):
        kh.same(getattr(a, what), getattr(b, what), f"{tag} {what}")
    assert a.idx == b.idx and a.ndpr == b.ndpr, f"{tag} index dicts"
    assert np.array_equal(a.placeholders[0], b.placeholders[0]) and a.placeholders[1] == b.placeholders[1], tag
    kh.same(ra.mamba_slot_read_offsets, rb.mamba_slot_read_offsets, f"{tag} read offsets")
    if ra._vps_fb_stats is not None or rb._vps_fb_stats is not None:
        kh.same(ra._vps_fb_stats, rb._vps_fb_stats, f"{tag} stats")
    kh.same(ra.drafter.draft_model.kv, rb.drafter.draft_model.kv, f"{tag} draft KV writes")


LEGS = {"cap_nocheck": (True, 0, 0), "cap_check4": (True, 4, 0), "nocap_nocheck_f8": (False, 0, 8)}


@pytest.mark.parametrize("leg", list(LEGS))
def test_k1_real_path_through_the_manager(monkeypatch, leg):
    cap, check, F = LEGS[leg]
    monkeypatch.setattr(tpu_runner, "_substitute_placeholder_token", fh._substitute_cpu)
    if check:
        monkeypatch.setattr(draft_asm, "AGREE_EVERY", 4)
    with kh.quiet(monkeypatch):
        props = {}
        for kind in ("one", "k1"):
            if cap:
                v6k._cap_install(monkeypatch, k1=kind == "k1", check=check)
                p = v6k._cap_proposer(kind, F)
            else:
                kh.install(monkeypatch, k1=kind == "k1", check=check)
                p = kh.make_pair_proposer(kind, F=F)
            _rig(p)
            for j, R in enumerate(RS):
                st = hc.make_step(56000 + j, R, cap=kh.CAP) if cap else kh.h.make_step(56000 + j, R, cap=kh.CAP)
                kh.h.run_step(p, st, serving=False)
            p.draft_model.kv.zero_()  # K1's warm-up calls wrote the synthetic requests' KV too
            props[kind] = p
        bp, kp = props["one"], props["k1"]
        assert isinstance(kp._draft_asm, k1fuse.DraftK1) and kp._draft_asm.one_cap is cap
        k1_steps = []
        for i in range(N):
            R, seed = RS[i % len(RS)], 56100 + i
            mk = v6k._vstep_cap if cap else kh.make_vstep
            ob = _step(bp, mk(seed, R), F, seed)
            served0 = kp._draft_asm.stats["folded_k1"]
            ok = _step(kp, mk(seed, R), F, seed)
            served = kp._draft_asm.stats["folded_k1"] - served0  # K1 served (a CHECK step runs K1 and serves the chain)
            tag = f"KR {leg} step {i + 1} R={R}"
            _same(ok, ob, kp.runner, bp.runner, tag)
            assert ob.result_type == "DraftAsmResult" and ob.calls.get("spec_tail", 0) == 0, (tag, ob.result_type, ob.calls)
            if served:
                k1_steps.append(i + 1)
                assert ok.result_type == "K1Result", (tag, ok.result_type)
                assert ok.src is ok.d.src, (tag, "the parked source is not K1's (re-made or padded)")
                assert ok.calls.get("spec_tail", 0) == 0 and ok.calls.get("one", 0) == 0, (tag, ok.calls)
            else:
                assert ok.result_type == "DraftAsmResult", (tag, ok.result_type)
    ctl = kp._draft_asm
    if check:
        assert ctl.k1_stats["check"] == check and ctl.k1_stats["checked"] == check and ctl.k1_mismatches == 0, (
            dict(ctl.k1_stats), ctl.k1_mismatches)
        assert k1_steps and min(k1_steps) > check and len(k1_steps) == N - min(k1_steps) + 1, k1_steps
    else:
        assert k1_steps == list(range(1, N + 1)), (k1_steps, dict(ctl.k1_reasons))
    COUNTS[leg] = len(k1_steps)
    print(f"\nV6C_FOLDFIX_K1 {leg}: {N} engine steps through verify -> propose -> manager -> consumer; K1 steps "
          f"{len(k1_steps)} (first {min(k1_steps)}), manager -> K1Result on each, parked source = K1's, 0 spec-tail "
          f"programs; CHECK {ctl.k1_stats['check']} checked {ctl.k1_stats['checked']} mismatches {ctl.k1_mismatches}; "
          f"bitwise the DraftOne chain on the same path (drafts, source, rejected, index dicts, placeholders, next-step "
          f"ids, read offsets, stats, draft KV)")


def test_zz_counts():
    print("\nV6C_FOLDFIX_K1_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
