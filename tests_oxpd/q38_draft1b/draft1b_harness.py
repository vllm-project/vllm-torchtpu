# SPDX-License-Identifier: Apache-2.0
"""CPU harness for q38 L4 DRAFT1-B (spec_decode/draft_one.py). Not collected (no test_ prefix).

Extends DRAFT-ASM's harness (tests/q38_draftasm/draftasm_harness.py: the real ``Eagle3Proposer.propose``
and the real async spec substitution, every compiled program run as its own body inside a counted
"compiled region"), with:

  * a stub MTP draft whose every forward reads seq_lens, query_start_loc AND request_distribution from
    the forward-context metadata (a wrong step-1 or step-0 metadata changes its hidden state), and
    whose step 0 takes ``draft_rows`` as the real draft does; the base path's step-0 forward runs it
    through the proposer's own metadata build (``fwd0_md``), not DRAFT-ASM's metadata-blind stub;
  * the stages' programs as regions: the superset seed ("seed_cl"), stage A's carry-loop ("carry_loop")
    and DH-D's one program ("one", per loop bucket);
  * the stage flags and window variables.
"""

import contextlib
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "q38_draftasm"))
import draftasm_harness as h  # noqa: E402

from vllm_torchtpu.spec_decode import draft_asm, draft_one  # noqa: E402
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer  # noqa: E402

K, H, MAX_REQS, PADDINGS, CEN = h.K, h.H, h.MAX_REQS, h.PADDINGS, h.CEN
CLEAR_ENV = (
    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP",
    "VLLM_TPU_DRAFT_ASM_ONE",
    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",
    "VLLM_TPU_DRAFT_ASM_ONE_AB",
)


def _fit(x: torch.Tensor, n: int) -> torch.Tensor:
    return x[:n] if x.shape[0] >= n else torch.nn.functional.pad(x, (0, n - x.shape[0]))


class FakeMtp2(h.FakeMtp):
    """FakeMtp whose forward reads the whole loop-relevant metadata (seq_lens, query lengths from
    query_start_loc, request_distribution) and, at step 0, evaluates `draft_rows` only (as the real
    draft does); last_hidden and the next hidden differ, so a carry that swaps them is caught."""

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None,
                spec_step_idx=0, draft_rows=None):
        md = h.CTX["md"]
        m = next(iter(md.values()))
        seq, qsl, rd = m.seq_lens, m.query_start_loc, m.request_distribution
        if draft_rows is not None:
            rows = draft_rows.clamp(min=0)
            input_ids, positions, hidden_states = input_ids[rows], positions[rows], hidden_states[rows]
        n = hidden_states.shape[0]
        s = _fit(seq.float(), n)
        q = _fit((qsl[1:] - qsl[:-1]).float(), n)
        r = rd.float().sum()
        hh = hidden_states * 0.5 + (
            input_ids.float() * 0.01 + positions.float() * 0.37 + s * 0.53 + q * 0.29 + r * 0.011
            + spec_step_idx
        ).unsqueeze(1)
        return hh, hh * 1.5 + 0.25


def fwd0_md(self, chunk, input_ids, positions, target_hidden_states, step_idx, seq_lens_delta, num_rejected_np,
            num_tokens_padded=None, num_rejected_dev=None, loop_query_start_loc=None,
            loop_request_distribution=None, draft_rows=None):
    """Step 0's draft forward (one compiled program in serving) as eagle3 runs it: the proposer's own
    step-0 metadata (the target's draft-layer entries), then the draft at `draft_rows`."""
    md = self._build_draft_attn_metadata(chunk=chunk, step_idx=0, seq_lens_delta=0, num_rejected_np=None)
    with h.fake_forward_context(md):
        return self.draft_model.forward(input_ids, positions, target_hidden_states, spec_step_idx=0,
                                        draft_rows=draft_rows)


class CarryLoopRegion:
    """Stage A's program as one compiled region, with the defer census of its inputs and the loop-entry
    record DRAFT-ASM's harness compares (step 1's seq_lens / positions, the first token, the hidden
    carry; the hidden carry recomputed here by slicing, independent of the code under test)."""

    def __init__(self, module, name):
        self.module, self.name = module, name

    def __call__(self, positions, last_hidden, hidden_states, seq_lens, tail_bonus, num_reqs):
        md = h.CTX.get("md") or {}
        mds = list({id(m): m for m in md.values()}.values())
        for what, t in [("positions", positions), ("last_hidden", last_hidden), ("hidden_states", hidden_states),
                        ("seq_lens", seq_lens), ("tail_bonus", tail_bonus), ("num_reqs", num_reqs)] + [
                (f"md.{f}", getattr(m, f)) for m in mds for f in ("seq_lens", "query_start_loc", "request_distribution")]:
            if isinstance(t, torch.Tensor) and getattr(t, "_q38_eager", None) == CEN.step_id:
                CEN.deferred.append(what)
        args = (positions, last_hidden, hidden_states, seq_lens, tail_bonus, num_reqs)
        CEN.calls[self.name] += 1
        CEN.keys.append((self.name, draft_asm._key(*args)))
        p = positions.shape[0]
        CEN.depth += 1  # the record's clones are the harness's, not the step's (inside the region)
        try:
            entry = {"positions": positions.clone(), "seq_lens": seq_lens.clone(),
                     "hidden": _fit_rows(hidden_states, p)}
            out = self.module(*args)
            entry["input_ids"] = out[0].clone()
        finally:
            CEN.depth -= 1
        CEN.loop_entry = entry
        return out


class OneRegion:
    """DH-D's program (one loop bucket) as one compiled region: the defer census of its inputs and the
    loop-entry record DRAFT-ASM's harness compares (step 1's positions and seq_lens as handed in, the first
    token as returned; the hidden carry is internal to the program and checked through the drafts)."""

    NAMES = ("input_ids", "positions", "hidden_states", "draft_rows", "positions_step1", "seq_lens0",
             "query_start_loc0", "request_distribution0", "seq_lens1", "query_start_loc1",
             "request_distribution1", "seq_lens_buf", "query_start_loc_buf", "request_distribution_buf",
             "tail_bonus", "num_reqs")

    def __init__(self, module, name):
        self.module, self.name = module, name

    def __call__(self, *args):
        assert len(args) == len(self.NAMES)
        md = h.CTX.get("md") or {}
        mds = list({id(m): m for m in md.values()}.values())
        for what, t in list(zip(self.NAMES, args)) + [
                (f"md.{f}", getattr(m, f)) for m in mds for f in ("seq_lens", "query_start_loc", "request_distribution")]:
            if isinstance(t, torch.Tensor) and getattr(t, "_q38_eager", None) == CEN.step_id:
                CEN.deferred.append(what)
        CEN.calls[self.name] += 1
        CEN.keys.append((self.name, draft_asm._key(*args)))
        a = dict(zip(self.NAMES, args))
        CEN.depth += 1  # the record's clones are the harness's, not the step's (inside the region)
        try:
            entry = {"positions": a["positions_step1"].clone(), "seq_lens": a["seq_lens1"].clone()}
            out = self.module(*args)
            entry["input_ids"] = out[0].clone()
        finally:
            CEN.depth -= 1
        CEN.loop_entry = entry
        return out


class RecMtp(FakeMtp2):
    """FakeMtp2 that records, per forward, the step index and the metadata values it read (the per-pass
    metadata proof: step 0 sees the verify step's, each loop step the loop's, advanced by one per step)."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None,
                spec_step_idx=0, draft_rows=None):
        m = next(iter(h.CTX["md"].values()))
        self.seen.append((spec_step_idx, m.seq_lens.clone(), m.query_start_loc.clone(),
                          m.request_distribution.clone(), None if draft_rows is None else draft_rows.clone()))
        return super().forward(input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds,
                               spec_step_idx, draft_rows)


def _fit_rows(x, n):
    if x.shape[0] >= n:
        return x[:n].clone()
    return torch.nn.functional.pad(x, (0, 0, 0, n - x.shape[0]))


def install(monkeypatch, fold=True, cl=False, one=False, cl_ab=None, one_ab=None, check_steps=0, ab=None):
    """DRAFT-ASM's install (served env, regions, fake contexts), the stub draft's metadata-reading step 0,
    the stages' regions and flags."""
    for name in CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    h.install(monkeypatch, fold=fold, check_steps=check_steps, ab=ab)
    monkeypatch.setattr(Eagle3Proposer, "_forward_draft", h.region("draft_fwd0", fwd0_md))
    if cl:
        monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_CARRY_LOOP", "1")
    if one:
        monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_ONE", "1")
    if cl_ab is not None:
        monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB", cl_ab)
    if one_ab is not None:
        monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_ONE_AB", one_ab)
    monkeypatch.setattr(draft_one, "_seed_cl_compiled", h.region("seed_cl", draft_one._seed_cl_impl))

    def build_cl(*, vllm_config, prefix, mtp, num_steps, dvp=False, max_reqs):
        m = draft_one._MtpFusedLoopCarry(vllm_config=None, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp,
                                        max_reqs=max_reqs)
        return CarryLoopRegion(m, "carry_loop")

    monkeypatch.setattr(draft_one, "build_mtp_fused_loop_carry", build_cl)

    def build_one(*, vllm_config, prefix, mtp, num_steps, dynamic_arg_dims, dvp=False, max_reqs, p):
        cls = draft_one._program_class(p)
        m = cls(vllm_config=None, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp, max_reqs=max_reqs)
        region = OneRegion(m, "one")
        region.p = p
        return region

    monkeypatch.setattr(draft_one, "build_draft_one_program", build_one)


def make_proposer(fold=True, cap=64, mtp_cls=None):
    """DRAFT-ASM's stub proposer with FakeMtp2; with `fold`, eagle3's controller choice under the current
    env (DraftOne when a stage flag is set, else DraftAsm)."""
    from vllm_torchtpu import envs

    p = h.make_proposer(False, cap)
    p.draft_model = (mtp_cls or FakeMtp2)()
    p._mtp_fused_loop.module.mtp = p.draft_model
    if fold:
        if envs.VLLM_TPU_DRAFT_ASM_CARRY_LOOP or envs.VLLM_TPU_DRAFT_ASM_ONE:
            p._draft_asm = draft_one.DraftOne(p)
        else:
            p._draft_asm = draft_asm.DraftAsm(p)
    return p


def run_step(p, step, serving, structured=False):
    return h.run_step(p, step, serving, structured)


make_step = h.make_step
same = h.same
assert_same_outputs = h.assert_same_outputs
region = h.region


@contextlib.contextmanager
def quiet_logs(monkeypatch, rec=None):
    from types import SimpleNamespace

    rec = rec if rec is not None else {"info": [], "error": []}
    log = SimpleNamespace(info=lambda *a: rec["info"].append(a[0] % a[1:]),
                          error=lambda *a: rec["error"].append(a[0] % a[1:]),
                          warning=lambda *a, **k: None)
    monkeypatch.setattr(draft_asm, "logger", log)
    monkeypatch.setattr(draft_one, "logger", log)
    yield rec
