# SPDX-License-Identifier: Apache-2.0
"""CPU harness for q38 L4 T1-K1FUSE (spec_decode/k1fuse.py). Not collected (no test_ prefix). CDK CPU pod only.

One serving step is driven through the REAL code on both sides:
  * the runner side: the real ``TPUModelRunner._sample_spec_verify_vps`` -> ``_vfy_verify`` (VFY-ONE value 2, with
    D83's fold: ``_apply_folded_read_offsets``, the stats accumulate, ``_vfy_fb_tick``) on a stub runner object of
    the real class, with a TP1 vocab-shard LM head (hidden @ W^T in bf16) behind the real ``vps_wiring.local_logits``
    and the real ``vp_sampler.vp_verify_fb``;
  * the draft side: DRAFT1-B's harness (tests/q38_draft1b/draft1b_harness.py over DRAFT-ASM's): the real
    ``Eagle3Proposer.propose`` and the real ``_assemble_async_spec_substitution``, every compiled program run as its
    own body inside a counted "compiled region" (verify program "vfy_one", DH-D "one", K1 "k1", ...).
The stub draft (``FakeMtpKV``) reads its ragged metadata from the forward context like DRAFT1-B's and writes a
deterministic "KV cache" (int64, overwritten) at slots derived from the
positions, seq_lens, step index and block tables, so KV writes can be compared bitwise.
"""

import contextlib
import dataclasses
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "q38_draft1b"))
sys.path.insert(0, str(HERE.parent / "q38_draftasm"))
import draft1b_harness as h1  # noqa: E402
import draftasm_harness as h  # noqa: E402

from vllm_torchtpu.runner import tpu_runner, vfy_one, vps_wiring  # noqa: E402
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner  # noqa: E402
from vllm_torchtpu.spec_decode import draft_asm, draft_one, k1fuse  # noqa: E402

K, H, MAX_REQS, PADDINGS, CEN = h.K, h.H, h.MAX_REQS, h.PADDINGS, h.CEN
VT = 5440  # target vocab: 34 whole blocks of vp_sampler.VP_BLOCK (local candidates need >= VP_KC + 1 blocks)
HID = 32
KV_SLOTS = 8192
NSLOT = 4096
CAP = 64
CLEAR_ENV = ("VLLM_TPU_K1FUSE", "VLLM_TPU_K1FUSE_AB", "VLLM_TPU_K1FUSE_CHECK")


def same(a, b, what):
    h.same(a, b, what)


# ---------------------------------------------------------------- the stub draft with KV writes


def _fit(x, n):
    return x[:n] if x.shape[0] >= n else torch.nn.functional.pad(x, (0, n - x.shape[0]))


class FakeMtpKV(h1.FakeMtp2):
    """FakeMtp2 (reads seq_lens, query_start_loc and request_distribution from the forward context; draft_rows at
    step 0) that also writes a KV cache: kv[slot] = round(1000 * h[:, 0]) at slot = f(position, seq_len, step,
    block tables), an overwrite as a real cache write is (a repeated write of the same values is idempotent). The
    hidden state does not read the block tables, so a misdirected KV write changes only the cache."""

    def __init__(self):
        super().__init__()
        self.kv = torch.zeros(KV_SLOTS, dtype=torch.int64)

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None,
                spec_step_idx=0, draft_rows=None):
        hh, nxt = super().forward(input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds,
                                  spec_step_idx, draft_rows)
        m = next(iter(h.CTX["md"].values()))
        pos = positions[draft_rows.clamp(min=0)] if draft_rows is not None else positions
        n = hh.shape[0]
        slot = (_fit(pos.long(), n) * 131 + _fit(m.seq_lens.long(), n) * 7 + spec_step_idx * 17
                + m.block_tables.long().sum() * 3) % KV_SLOTS
        self.kv.index_put_((slot,), (hh[:, 0].float() * 1000).round().to(torch.int64))  # overwrite, as a cache
        return hh, nxt


# ---------------------------------------------------------------- the TP1 VPS head


class Proc:
    """A TP1 vocab shard: the rank's local logits = hidden @ W^T (bf16), as LogitsProcessor._get_logits(skip_gather)."""

    def _get_logits(self, hidden, head, bias, skip_gather=False):
        assert bias is None and skip_gather
        return torch.nn.functional.linear(hidden, head.weight)


_gw = torch.Generator().manual_seed(4242)
HEAD = SimpleNamespace(weight=(torch.randn(VT, HID, generator=_gw) * 0.5).bfloat16())
VH = vps_wiring.VpsHead(Proc(), HEAD, 0, VT, VT, 1)


class AlwaysWarm(set):
    def __contains__(self, k):
        return True


class StubRunner(TPUModelRunner):
    """The real runner class without __init__; the properties the harness sets are plain attributes here."""

    _is_async_drafter = True
    _fence_spec_kv_under_token_copy = False
    _pp_is_last = True

    @property
    def _async_drafter(self):
        return self.drafter


def _padded_sampling_params(self, s, e, n, dtype, device):
    k = torch.where(self.top_k >= VT, 0, self.top_k)
    return (self.temps.to(dtype).view(-1, 1)[:n], k.view(-1, 1)[:n].clone(),
            torch.full((n, 1), 0.95, dtype=torch.float32))


def make_runner(p, F=0, cand=True, seed=0, flagged=None):
    """The stub runner for proposer `p` (replaces p.runner): VPS state as VFY-ONE's tests build it, F fallback rows
    (F > 0: tiegap = F so the served F is F), VFY-ONE value 2 armed with a warm logits dtype."""
    base = p.runner
    r = object.__new__(StubRunner)
    for k_, v in vars(base).items():
        setattr(r, k_, v)
    # one rate per draft slot, plus one for the padding group index K (padded slots are inactive: group K is at or
    # above every draft length; served metadata pads segment ids with the request count, which a CPU gather of the
    # [B] sampling params cannot take when R == B, so the harness pads segments with min(R, B - 1) instead)
    r.rejection_sampler = SimpleNamespace(synthetic_conditional_rates=torch.tensor([1.0, 0.32, 0.0, 0.0]),
                                          synthetic_mode=True)
    r.speculative_config = p.speculative_config
    r.scheduler_config = SimpleNamespace(async_scheduling=True)
    r.model_config = SimpleNamespace(get_hidden_size=lambda: HID)
    r._vps_head = VH
    r._vps_checks_left = r._vps_check_faults = r._vps_checks_done = 0
    r._vps_mismatches = None
    r._vps_fb_path, r._vps_fb_max, r._vps_tiegap, r._vps_cand_draws = True, F, F, cand
    r._vps_fb_stats, r._vps_fb_chunks = None, 0
    # q38 L4 v6 composition (V6COMP-BUILD; tests only): VPSHOST T-1(i)'s state on the composed runner (statsasync
    # off, nothing in flight), as the v5 composition gave VFY-ONE's and SEEDFIX's stub runners
    from vllm_torchtpu.runner import l4_vpshost

    r._vps_fb_async = l4_vpshost.AsyncCounts()
    r._num_reqs_device = {}
    r._fused_read_offsets = (None, 0)
    r._fused_read_offsets_applied = False
    r._vps_ro_fold = True
    r._vps_ro_checks_left = r._vps_ro_check_faults = r._vps_ro_checks_done = 0
    r._vps_ro_mismatches = None
    r._vps_ro_folds = 0
    r._vps_ro_warmed = set()
    g = torch.Generator().manual_seed(7000 + seed)
    buf = torch.randint(0, 9, (NSLOT,), generator=g, dtype=torch.int32)
    buf[0] = 0
    r.mamba_slot_read_offsets = buf
    r.top_k = torch.full((256,), 20, dtype=torch.int32)
    for i in (flagged if flagged is not None else ((1, 5) if F else ())):
        r.top_k[i] = VT  # top-k off: the candidate path cannot cover the row, a fallback row does
    r.temps = torch.full((256,), 0.6)
    r._has_mamba_state = False
    mk = lambda m: SimpleNamespace(metadata_builders=[SimpleNamespace(is_mamba_group=m)])  # noqa: E731
    r.attn_groups = [[mk(True), mk(False)], [mk(True)], [mk(False), mk(True)]]
    r.most_model_len = None
    r._hidden_states_dtype = torch.bfloat16
    r._vfy = vfy_one.VfyOne(2, 0, "")
    r._vfy.logits_dtype = torch.bfloat16
    r._vfy.acc_zero = torch.zeros(3, dtype=torch.int64)
    r._vfy.warm = AlwaysWarm()
    r._vfy.rank = 0
    r.drafter = p
    r._padded_sampling_params = types.MethodType(_padded_sampling_params, r)
    r._upload_unless_unchanged = lambda slot, host: host
    r.compute_selected_local_logits = types.MethodType(h.unwrap(TPUModelRunner.compute_selected_local_logits), r)
    r.compute_local_logits_from_hidden_states = types.MethodType(
        h.unwrap(TPUModelRunner.compute_local_logits_from_hidden_states), r)
    p.runner = r
    return r


# ---------------------------------------------------------------- the K1 program as a counted region


class K1Region:
    """K1's program (one static signature) as one compiled region: counted, keyed (vfy_one.key) and outside the
    eager census; ops inside are recorded (``ops``) for the in-program RNG check."""

    def __init__(self, module, name="k1"):
        self.module, self.name = module, name
        self.ops: list = []

    def __call__(self, *args):
        assert len(args) == len(k1fuse.ARG_NAMES)
        CEN.calls[self.name] += 1
        CEN.keys.append((self.name, vfy_one.key(*args)))
        CEN.depth += 1
        try:
            with _OpRec(self.ops):
                return self.module(*args)
        finally:
            CEN.depth -= 1


class _OpRec(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        self.sink.append(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


BUILT: list = []


def install(monkeypatch, k1=True, check=0, k1_ab=None, one_ab=None, cl_ab=None):
    """DRAFT1-B's install (served env, regions, fake contexts; fold + stage A + DH-D), the verify program as a
    region, the TP1 all-gather, and the K1 flags with K1's program as a region."""
    for name in CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    h1.install(monkeypatch, fold=True, cl=True, one=True, cl_ab=cl_ab, one_ab=one_ab)
    if k1:
        monkeypatch.setenv("VLLM_TPU_K1FUSE", "1")
        monkeypatch.setenv("VLLM_TPU_K1FUSE_CHECK", str(check))
    if k1_ab is not None:
        monkeypatch.setenv("VLLM_TPU_K1FUSE_AB", k1_ab)
    monkeypatch.setattr(tpu_runner, "_vfy_one_compiled", h.region("vfy_one", tpu_runner._vfy_one_impl))
    monkeypatch.setattr(tpu_runner, "_vps_verify_fb_compiled", h.region("vps_verify_fb", tpu_runner._vps_verify_fb_impl))
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "tensor_model_parallel_all_gather", lambda x, dim=-1: x, raising=False)

    def build(*, vllm_config, prefix, mtp, num_steps, dvp=False, vh, statics, index):
        m = k1fuse._program_class(index)(vllm_config=None, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp,
                                         vh=vh, statics=statics)
        region = K1Region(m)
        region.index, region.statics = index, statics
        BUILT.append(region)
        return region

    monkeypatch.setattr(k1fuse, "build_k1fuse_program", build)


def make_pair_proposer(kind, F=0, cand=True, seed=0, mtp_cls=None, flagged=None):
    """kind "k1" (DraftK1, flag on) or "one" (DraftOne: today's VFY + DH-D chain), with its stub runner."""
    BUILT.clear()
    p = h1.make_proposer(fold=False, cap=CAP, mtp_cls=mtp_cls or FakeMtpKV)
    p._draft_asm = k1fuse.DraftK1(p) if kind == "k1" else draft_one.DraftOne(p)
    make_runner(p, F=F, cand=cand, seed=seed, flagged=flagged)
    return p


# ---------------------------------------------------------------- one verify step


@dataclasses.dataclass
class VStep:
    step: h.Step
    hidden: torch.Tensor
    idx: torch.Tensor
    md: SimpleNamespace
    groups: list
    R: int


def make_vstep(seed, R, prev_drafts=None):
    """One decode verify step of R requests: DRAFT-ASM's draft chunk (T = token bucket of R * (K + 1), width B) and
    the verify inputs consistent with it, padded as the served metadata is: hidden [T, HID], logits indices
    arange(T), target slots / draft ids / segment ids / group indices [T] (padding: segment min(R, B - 1), group K,
    i.e. inactive), bonus indices [B]. `prev_drafts` [>= R, K] are the previous step's drafts (the carry); else drafts
    are the verify row's argmax 60 % of the time."""
    st = h.make_step(seed, R, cap=CAP)
    c = st.chunk
    T, B = int(c.input_ids.shape[0]), int(c.draft_lengths.shape[0])
    g = torch.Generator().manual_seed(seed + 5)
    hidden = torch.randn(T, HID, generator=g).bfloat16()
    idx = torch.arange(T, dtype=torch.int32)
    logits = Proc()._get_logits(hidden, HEAD, None, True)
    qsl = c.query_start_loc_np
    d = c.draft_lengths.numpy()
    t_idx = torch.zeros(T, dtype=torch.int32)
    dti = torch.zeros(T, dtype=torch.int32)
    seg = torch.full((T,), min(R, B - 1), dtype=torch.int64)
    grp = torch.full((T,), K, dtype=torch.int32)
    b_idx = torch.zeros(B, dtype=torch.int32)
    pos = 0
    for i in range(R):
        for j in range(int(d[i])):
            t = int(qsl[i]) + j
            t_idx[pos], seg[pos], grp[pos] = t, i, j
            if prev_drafts is not None:
                dti[pos] = int(prev_drafts[i, j])
            elif float(torch.rand((), generator=g)) < 0.6:
                dti[pos] = int(logits[t].float().argmax())
            else:
                dti[pos] = int(torch.randint(0, VT, (), generator=g))
            pos += 1
        b_idx[i] = int(qsl[i]) + int(d[i])
    md = SimpleNamespace(bonus_logits_indices=b_idx, target_logits_indices=t_idx, draft_token_ids=dti,
                         draft_lengths=c.draft_lengths, segment_ids=seg, group_indices=grp,
                         draft_lengths_cpu=st.md.draft_lengths_cpu)
    st.md = md
    gs = torch.Generator().manual_seed(9000 + seed)
    slots = (torch.randperm(NSLOT - 1, generator=gs)[: 3 * CAP] + 1).to(torch.int32).view(3, CAP)
    slots[:, R:] = 0
    return VStep(st, hidden, idx, md, [slots[i].clone() for i in range(3)], R)


@dataclasses.dataclass
class Out:
    nt: torch.Tensor
    gen_state: torch.Tensor
    drafts: torch.Tensor
    src: torch.Tensor
    rejected: torch.Tensor
    idx: dict
    ndpr: dict
    result: object
    calls: dict
    eager: list
    keys: list


def verify(r, vs, F, seed):
    r._fused_read_offsets = (vs.groups, vs.R)
    r._fused_read_offsets_applied = False
    r.drafter.draft_chunks = [vs.step.chunk]
    r._vfy.step_mode = "vfy"
    gen = torch.Generator().manual_seed(1000 + seed)
    nt = r._sample_spec_verify_vps(vfy_one.VfyChunk(vs.hidden, vs.idx), vs.md, 0, vs.R, gen, F)
    return nt, gen


def serve(p, vs, F, seed, between=None):
    """One serving step: the verify (sample_tokens' VFY path), `between(r, nt)` (writes sample_tokens makes before
    the propose), the propose and the async spec substitution. Returns an Out."""
    r = p.runner
    r.warmup_done = True
    agree_clock = r.__dict__.get("_l4_agree")
    if agree_clock is not None:
        # q38 L4 AGREE port (v6 composition, V6COMP-BUILD; tests only): one engine step = one tick of the runner's
        # agreement clock, before the step's verify (execute_model's hook); a propose no longer agrees.
        agree_clock.tick(r)
    r.input_batch = SimpleNamespace(num_reqs=vs.R, req_ids=[f"r{i}" for i in range(vs.R)])
    vs.step.chunk.start_index = 0
    CEN.reset()
    with CEN:
        CEN.on = True
        try:
            nt, gen = verify(r, vs, F, seed)
            if between is not None:
                between(r, nt)
            p.draft_chunks = [vs.step.chunk]
            drafts = p.propose(
                sampled_token_ids=None, discard_sampled_tokens_req_indices=[], num_rejected_tokens_np=None,
                scheduler_output=SimpleNamespace(has_structured_output_requests=False), return_device=True,
                next_tokens_per_chunk=[nt],
            )
            state = SimpleNamespace(num_reqs_list=[vs.R], spec_decode_metadata_list=[vs.md])
            srcs, idx, rej, ndpr = r._assemble_async_spec_substitution(drafts, [nt], state)
        finally:
            CEN.on = False
    d = drafts.drafts() if isinstance(drafts, draft_asm.DraftAsmResult) else drafts
    return Out(nt, gen.get_state(), d, srcs[0], rej, idx, ndpr, drafts, dict(CEN.calls), list(CEN.eager),
               list(CEN.keys))


def warm(p, rs, seed0=500):
    """Warm-up steps (runner.warmup_done False): the real warm-mode propose at each request count."""
    for i, R in enumerate(rs):
        h.run_step(p, h.make_step(seed0 + i, R, cap=CAP), serving=False)


def agree(p):
    """One DRAFT-ASM boundary (as the first serving call makes): the fold, DH-D and K1 agreements. q38 L4 AGREE port
    (v6 composition, V6COMP-BUILD; tests only): the boundary is one gather of the runner's agreement clock
    (runner/l4_agree.py ``Agreement._agree``, what DRAFT-ASM's removed ``_boundary`` did), without a clock step."""
    p.runner.warmup_done = True
    p.runner._l4_agree._agree()


def assert_same_step(a: Out, b: Out, ra, rb, tag):
    same(a.nt, b.nt, f"{tag} tokens")
    assert torch.equal(a.gen_state, b.gen_state), f"{tag} random stream"
    same(a.drafts, b.drafts, f"{tag} drafts")
    same(a.src, b.src, f"{tag} source")
    same(a.rejected, b.rejected, f"{tag} rejected count")
    assert a.idx == b.idx and a.ndpr == b.ndpr, f"{tag} index dicts"
    assert (ra._vps_fb_stats is None) == (rb._vps_fb_stats is None), f"{tag} stats presence"
    if ra._vps_fb_stats is not None:
        same(ra._vps_fb_stats, rb._vps_fb_stats, f"{tag} stats")
    same(ra.mamba_slot_read_offsets, rb.mamba_slot_read_offsets, f"{tag} read offsets")
    assert (ra._vps_fb_chunks, ra._vps_ro_folds) == (rb._vps_fb_chunks, rb._vps_ro_folds), f"{tag} chunk counts"
    same(ra.drafter.draft_model.kv, rb.drafter.draft_model.kv, f"{tag} draft KV writes")


def program_calls(out: Out) -> int:
    return sum(v for k_, v in out.calls.items() if k_ in ("vfy_one", "one", "k1", "seed_cl", "draft_fwd0",
                                                           "carry_loop", "loop_asm", "seed_asm", "seed"))


@contextlib.contextmanager
def quiet(monkeypatch, rec=None):
    rec = rec if rec is not None else {"info": [], "error": [], "warning": []}
    log = SimpleNamespace(info=lambda *a: rec["info"].append(a[0] % a[1:]),
                          error=lambda *a: rec["error"].append(a[0] % a[1:]),
                          warning=lambda *a, **k: rec["warning"].append(a[0] % a[1:]))
    for mod in (draft_asm, draft_one, k1fuse):
        monkeypatch.setattr(mod, "logger", log)
    yield rec


os.environ.setdefault("JAX_PLATFORMS", "cpu")
_ = np  # numpy stays imported for DRAFT-ASM's step builder
