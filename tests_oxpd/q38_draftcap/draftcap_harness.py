# SPDX-License-Identifier: Apache-2.0
"""CPU harness for q38 L4 DRAFT-CAP inside DH-D (spec_decode/draft_cap.py). Not collected (no test_ prefix).

Extends DRAFT1-B's harness (tests/q38_draft1b/draft1b_harness.py: the real ``Eagle3Proposer.propose`` and the
real async spec substitution, every compiled program run as its own body inside a counted "compiled region"):

  * the draft layer's metadata as the real QSA MTP layer sees it: distinct per-group AttentionMetadata objects
    (main, compressed-key and raw-ring groups, plus a second layer name on the main group's object) sharing
    seq_lens / query_start_loc / request_distribution, each with its own request-major flat block table
    [capacity * cols] (live request rows filled, rows past the live requests zero, as the runner's builder
    zeroes them);
  * a stub MTP draft (``FakeMtpCap``) whose every forward reads, per request slot, seq_lens, the query lengths,
    and EVERY group's block-table row (reshaped by len(seq_lens), as pooled QSA reshapes its tables by
    len(context)), plus query_start_loc's last entry (the token total, which the real draft's valid mask reads)
    and request_distribution: a dropped, shifted or mis-sized table row, a wrong query_start_loc tail or a
    wrong seq_lens slot changes a live row's hidden state;
  * the R_d program as the "one" region (``CapRegion``, the DH-D region with the table arguments) and the flag.
"""

import contextlib
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "q38_draft1b"))
sys.path.insert(0, str(HERE.parent / "q38_draftasm"))
import draft1b_harness as h1  # noqa: E402
import draftasm_harness as h0  # noqa: E402

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata  # noqa: E402
from vllm_torchtpu.spec_decode import draft_asm, draft_cap, draft_one  # noqa: E402

K, H, MAX_REQS, PADDINGS, CEN = h0.K, h0.H, h0.MAX_REQS, h0.PADDINGS, h0.CEN
MAIN, COMP, RING, MAIN2 = "draft.attn.0", "draft.attn.0.compressed", "draft.attn.0.raw_ring", "draft.attn.1"
COLS = {MAIN: 5, COMP: 3, RING: 1}
GROUPS = (MAIN, COMP, RING)
NAMES = (MAIN, COMP, RING, MAIN2)
CAP_ENV = "VLLM_TPU_DRAFT_ASM_CAP"
same = h0.same
make_step_base = h0.make_step


def _fit(x: torch.Tensor, n: int) -> torch.Tensor:
    return x[:n] if x.shape[0] >= n else torch.nn.functional.pad(x, (0, n - x.shape[0]))


def group_md(step, seed: int, groups: dict | None = None, share: dict | None = None) -> dict:
    """Replace the step chunk's draft metadata with per-group objects (see the module docstring). `groups`:
    name -> cols (default COLS); `share`: name -> name whose block-table TENSOR it reuses (distinct object)."""
    groups = dict(COLS if groups is None else groups)
    share = share or {}
    c = step.chunk
    cap = c.attn_ctx.seq_lens.shape[0]
    base = c.attn_metadata["target.attn.0"]
    g = np.random.default_rng(seed + 7919)
    md = {"target.attn.0": base}
    tables = {}
    for name, cols in groups.items():
        if name in share:
            tables[name] = tables[share[name]]
        else:
            t = np.zeros((cap, cols), dtype=np.int32)
            t[: step.R] = g.integers(1, 1 << 20, (step.R, cols))
            tables[name] = torch.from_numpy(t.reshape(-1))
        md[name] = AttentionMetadata(
            input_positions=base.input_positions, block_tables=tables[name], seq_lens=c.attn_ctx.seq_lens,
            query_start_loc=c.attn_ctx.query_start_loc, request_distribution=c.attn_ctx.request_distribution,
        )
    if MAIN in md and MAIN2 not in groups:
        md[MAIN2] = md[MAIN]  # a second layer name on the main group's object
    c.attn_metadata = md
    return md


def make_step(seed: int, R: int, cap: int = 64, groups: dict | None = None, share: dict | None = None, **kw):
    step = make_step_base(seed, R, cap=cap, **kw)
    group_md(step, seed, groups, share)
    return step


def _slot_features(md: dict, slots: int) -> torch.Tensor:
    """Per request slot: every group's block-table row (reshaped by `slots`, as pooled QSA reshapes by
    len(context)), weighted per column and per group."""
    out = torch.zeros(slots)
    names = [n for n in md if n.startswith("draft.")]
    for gi, name in enumerate(sorted(set(names))):
        tb = md[name].block_tables.reshape(slots, -1).float()
        w = (torch.arange(tb.shape[1], dtype=torch.float32) + 1.0) * (0.0007 * (gi + 1))
        out = out + (tb * w).sum(1)
    return out


class FakeMtpCap(h1.FakeMtp2):
    """The stub draft of the module docstring. last_hidden and the next hidden differ (a swapped carry is caught)."""

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None,
                spec_step_idx=0, draft_rows=None):
        md = h0.CTX["md"]
        m = md[MAIN]
        seq, qsl, rd = m.seq_lens, m.query_start_loc, m.request_distribution
        slots = seq.shape[0]
        assert qsl.shape[0] == slots + 1, (tuple(seq.shape), tuple(qsl.shape))
        if draft_rows is not None:
            rows = draft_rows.clamp(min=0)
            input_ids, positions, hidden_states = input_ids[rows], positions[rows], hidden_states[rows]
        n = hidden_states.shape[0]
        s = _fit(seq.float(), n)
        q = _fit((qsl[1:] - qsl[:-1]).float(), n)
        t = _fit(_slot_features(md, slots), n)
        total = qsl[-1].float()
        r = rd.float().sum()
        hh = hidden_states * 0.5 + (
            input_ids.float() * 0.01 + positions.float() * 0.37 + s * 0.53 + q * 0.29 + t + total * 0.003
            + r * 0.011 + spec_step_idx
        ).unsqueeze(1)
        return hh, hh * 1.5 + 0.25


class RecMtpCap(FakeMtpCap):
    """FakeMtpCap that records, per forward: (step index, seq_lens, query_start_loc, request_distribution,
    {group: block table}, draft_rows) as read from the forward context."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None,
                spec_step_idx=0, draft_rows=None):
        md = h0.CTX["md"]
        m = md[MAIN]
        self.seen.append((spec_step_idx, m.seq_lens.clone(), m.query_start_loc.clone(),
                          m.request_distribution.clone(),
                          {n: md[n].block_tables.clone() for n in md if n.startswith("draft.")},
                          None if draft_rows is None else draft_rows.clone()))
        return super().forward(input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds,
                               spec_step_idx, draft_rows)


class CapRegion:
    """The R_d program (one loop bucket) as one compiled region: the defer census of its inputs (and of the
    forward-context metadata it reads, block tables included), its shape key, and DH-D's loop-entry record."""

    NAMES = h1.OneRegion.NAMES + tuple(f"table{i}" for i in range(draft_cap.TABLE_SLOTS)) + tuple(
        f"table_buf{i}" for i in range(draft_cap.TABLE_SLOTS))

    def __init__(self, module, name):
        self.module, self.name = module, name

    def __call__(self, *args):
        assert len(args) == len(self.NAMES), (len(args), len(self.NAMES))
        md = h0.CTX.get("md") or {}
        mds = list({id(m): m for m in md.values()}.values())
        for what, t in list(zip(self.NAMES, args)) + [
                (f"md.{f}", getattr(m, f)) for m in mds
                for f in ("seq_lens", "query_start_loc", "request_distribution", "block_tables")]:
            if isinstance(t, torch.Tensor) and getattr(t, "_q38_eager", None) == CEN.step_id:
                CEN.deferred.append(what)
        CEN.calls[self.name] += 1
        CEN.keys.append((self.name, draft_asm._key(*args)))
        a = dict(zip(self.NAMES, args))
        CEN.depth += 1
        try:
            entry = {"positions": a["positions_step1"].clone(), "seq_lens": a["seq_lens1"].clone()}
            out = self.module(*args)
            entry["input_ids"] = out[0].clone()
        finally:
            CEN.depth -= 1
        CEN.loop_entry = entry
        return out


def install(monkeypatch, fold=True, cl=False, one=False, cap=False, **kw):
    """DRAFT1-B's install (served env, regions, stage flags), the DRAFT-CAP flag and its R_d program region."""
    monkeypatch.delenv(CAP_ENV, raising=False)
    h1.install(monkeypatch, fold=fold, cl=cl, one=one, **kw)
    if cap:
        monkeypatch.setenv(CAP_ENV, "1")

    def build_cap(*, vllm_config, prefix, mtp, num_steps, dynamic_arg_dims, dvp=False, max_reqs, p):
        cls = draft_cap._cap_program_class(p)
        m = cls(vllm_config=None, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp, max_reqs=max_reqs)
        region = CapRegion(m, "one")
        region.p = p
        region.prefix = prefix
        return region

    monkeypatch.setattr(draft_cap, "build_draft_cap_program", build_cap)


def make_proposer(fold=True, cap=64, mtp_cls=None, names=NAMES):
    p = h1.make_proposer(fold=fold, cap=cap, mtp_cls=mtp_cls or FakeMtpCap)
    p._draft_attn_layer_names = set(names)
    return p


def variant(monkeypatch, name, cap=64, mtp_cls=None, names=NAMES, **kw):
    """base (fold off), dha (DRAFT-ASM), cl (+ stage A), one (+ DH-D at the capacity), cap (+ DRAFT-CAP)."""
    install(monkeypatch, fold=name != "base", cl=name in ("cl", "one", "cap"), one=name in ("one", "cap"),
            cap=name == "cap", **kw)
    return make_proposer(fold=name != "base", cap=cap, mtp_cls=mtp_cls, names=names)


def run_step(p, step, serving, structured=False):
    return h0.run_step(p, step, serving, structured)


def serve(monkeypatch, name, seeds, R, warm_each=False, cap=64, step_kw=None, mtp_cls=None, rs=None, **kw):
    """One warm-up step (or one before every step), then a serving step per seed. `rs`: per-step request counts
    (default R for every step); every distinct count is warmed first."""
    step_kw = step_kw or {}
    rs = rs or [R] * len(seeds)
    p = variant(monkeypatch, name, cap=cap, mtp_cls=mtp_cls, **kw)
    if not warm_each:
        for j, r in enumerate(sorted(set(rs))):
            run_step(p, make_step(seeds[0] - 1 - j, r, cap=cap, **step_kw), serving=False)
    outs = []
    for s, r in zip(seeds, rs):
        if warm_each:
            run_step(p, make_step(s, r, cap=cap, **step_kw), serving=False)
        outs.append(run_step(p, make_step(s, r, cap=cap, **step_kw), serving=True))
    return outs, p


def same_out(a, b, tag):
    same(a.drafts, b.drafts, f"{tag} drafts")
    same(a.src, b.src, f"{tag} source")
    same(a.rejected, b.rejected, f"{tag} rejected count")
    assert a.idx == b.idx and a.ndpr == b.ndpr, f"{tag} index dicts"
    for k in ("seq_lens", "positions", "input_ids"):
        same(a.loop_entry[k], b.loop_entry[k], f"{tag} loop-entry {k}")


def one_key(out):
    """The serving step's DH-D program shape key (the "one" region's), or None."""
    keys = [k for n, k in out.keys if n == "one"]
    return keys[-1] if keys else None


@contextlib.contextmanager
def quiet_logs(monkeypatch, rec=None):
    from types import SimpleNamespace

    rec = rec if rec is not None else {"info": [], "error": [], "warning": []}
    rec.setdefault("warning", [])
    log = SimpleNamespace(info=lambda *a: rec["info"].append(a[0] % a[1:]),
                          error=lambda *a: rec["error"].append(a[0] % a[1:]),
                          warning=lambda *a, **k: rec["warning"].append(a[0] % a[1:] if len(a) > 1 else a[0]))
    for mod in (draft_asm, draft_one, draft_cap):
        monkeypatch.setattr(mod, "logger", log)
    yield rec


def cap_bufs(ctl):
    """The controller's DRAFT-CAP buffer sets: {R_d: (bufs, table_bufs)}."""
    return {k[1]: v for k, v in ctl.one_bufs.items() if isinstance(k, tuple) and k and k[0] == "cap"}

