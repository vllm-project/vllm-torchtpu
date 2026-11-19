# SPDX-License-Identifier: Apache-2.0
"""CPU harness for q38 L4 FOLDFIX: the REAL draft path from the manager to the runner consumer. Not collected.

FOLDFIX-BUILD (helper of L4-E), 2026-11-14. CDK CPU pod only (never the Mac or the VM).

Why it exists. DRAFT-ASM's fold returns a ``DraftAsmResult`` from ``Eagle3Proposer.propose``; the manager's
return_device branch asserted ``isinstance(result, torch.Tensor | list)`` and the engine died on the first folded
step (T81r on TPU). Every earlier L4 CPU harness called ``propose`` directly and so never ran the manager. This
harness runs, per engine step, exactly the code the runner runs between the drafter and the next step:

  * the drafter: the real ``Eagle3Proposer.propose`` with DRAFT-ASM / DRAFT1-B DH-D / DRAFT-CAP controllers, through
    the existing CPU harnesses (tests/q38_draftasm, tests/q38_draft1b, tests/q38_draftcap: every compiled program
    replaced by its own body inside a counted "compiled region");
  * the manager: the real ``SpeculativeDecodingManager`` (its ``propose_draft_token_ids``,
    ``stage_draft_token_ids_for_host`` and ``take_draft_token_ids``);
  * the runner: the real ``TPUModelRunner.sample_tokens`` statements from the unified draft propose
    (``is_async = bool(self.scheduler_config.async_scheduling)``) up to ``host_phases.sub("st_async")``, read from
    the tree's own tpu_runner.py at test time and compiled with their original file name and line numbers
    (``segment``), on a ``TPUModelRunner`` instance whose methods are the real ones: the fence, the TP pin, the
    host staging, ``_assemble_async_spec_substitution`` (the folded-source consumer), the prefill bootstrap,
    ``_update_placeholder`` and the ``AsyncPreResults`` it parks;
  * the next step's input ids: the real ``_prepare_async_token_substitution_indices`` and
    ``_apply_async_token_substitution`` (the substitution program run as its body);
  * the engine-step clock: the real ``runner/l4_agree.py`` Agreement with DRAFT-ASM and VFY-ONE registered as the
    runner registers them; VFY-ONE's own ``decide`` picks its step mode, and on a "vfy" step the drafter's seed is
    stashed as ``TPUModelRunner._vfy_verify`` stashes it (``spec.fn(next_tokens, *spec.args)`` through the runner's
    own ``_vfy_seed_spec``; the VFY program's identity to the D83 chain is tests/l4vfy's).

Tree-independent: no src hash or line-number pins. The segment is located by its two statements and must contain the
consumer calls listed in SEG_MUST, so a refactor fails loudly instead of testing less.
"""

import __future__

import ast
import contextlib
import pathlib
import sys
import textwrap
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
for _d in ("q38_draftasm", "q38_draft1b", "q38_draftcap"):
    sys.path.insert(0, str(HERE.parent / _d))
import draft1b_harness as h1  # noqa: E402,F401  (installs DRAFT1-B's regions through draftcap_harness)
import draftasm_harness as h0  # noqa: E402
import draftcap_harness as hc  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.runner import l4_agree, tpu_runner, vfy_one  # noqa: E402
from vllm_torchtpu.runner.speculative_decoding_manager import SpeculativeDecodingManager  # noqa: E402
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner, _get_padded_token_len  # noqa: E402
from vllm_torchtpu.spec_decode import draft_asm, draft_one  # noqa: E402
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer  # noqa: E402

K, MAX_REQS, PADDINGS, CEN = h0.K, h0.MAX_REQS, h0.PADDINGS, h0.CEN
ROWS = (8, 32, 48, 64)
BOUND = MAX_REQS * (1 + K)

# STACK-D v5's served decode env (L1 #5's render scratch/decL1/v5/car/exports/v5c_g0m0l0_idx/smoke_params.txt, relayed by
# L4-E 2026-11-14): every member the drafter / manager / runner-consumer path can read, plus the v5 members that path does
# not read (set anyway, as served). Megakernel, QSA, PLE, L6 host levers and host phases are runner- / model-level and
# are not modelled here, so they are not set. None = absent from the render (the code default applies).
DECODE_TRIM = "16:6.51/32:9.10/48:11.35/64:13.38/80:17.47/96:20.21/112:22.94/128:25.68/al:2.297/above:64"
V5_ENV = {
    "VLLM_TPU_DRAFT_ASM_FOLD": "1", "VLLM_TPU_DRAFT_ASM_ONE": "1", "VLLM_TPU_DRAFT_ASM_CAP": "1",
    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP": None, "VLLM_TPU_DRAFT_ASM_CHECK_STEPS": None,
    "VLLM_TPU_VPS_VFY_ONE": "2", "VLLM_TPU_VPS_VFY_ONE_CHECK": None,
    "VLLM_TPU_L4_SAMP_SNAP": "1", "VLLM_TPU_L4_VPS_STATS_ASYNC": "1", "VLLM_TPU_L4_SAMP_FB_ORDER": "1",
    "VLLM_TPU_DRAFT_EMBED_REPL": "1", "VLLM_TPU_TARGET_EMBED_REPL": "1", "VLLM_TPU_DRAFT_EMBED_H8": "1",
    "VLLM_TPU_VPS_INGRAPH_RNG": "1", "VLLM_TPU_VPS_TOPK_CUTOFF": "1", "VLLM_TPU_VPS_FUSED_READ_OFFSETS": "1",
    "VLLM_TPU_DRAFT_CARRY_FUSED": "1", "VLLM_TPU_DRAFT_CARRY_ROWS_FRESH": "1", "VLLM_TPU_DRAFT_SEED_EXTRAS": "1",
    "VLLM_TPU_DRAFT_SEED_FUSED": "1", "VLLM_TPU_DRAFT_TOP_TOKENS": "1", "VLLM_TPU_DRAFT_TOP_TOKENS_CHECK": "200",
    "VLLM_TPU_MTP_DRAFT_ROWS": "1", "VLLM_TPU_FUSED_DRAFT_LOOP": "1", "VLLM_TPU_PROPOSE_DEVICE_CONSTANTS": "1",
    "VLLM_TPU_SAMPLING_COPY_FULL": "1", "VLLM_TPU_SAMPLING_PARAMS_REUSE": "1",
    "VLLM_TPU_SPEC_TAIL_COMPILED": "1", "VLLM_TPU_SPEC_TAIL_REJ_PAD": "1",
    "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_FULL_DRAFT_MAX_SEQS": "64", "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS": "32",
    "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY": "1",
    "VLLM_TPU_VPS_DRAWS": "full", "VLLM_TPU_VPS_FALLBACK_ROWS": "8", "VLLM_TPU_VPS_STATS_EVERY": "200",
    "VLLM_TPU_VPS_TOPK_KEYS": "1", "VLLM_TPU_VPS_VERIFY": "1", "VLLM_TPU_VPS_WARM_DRAWS": "4",
    "VLLM_TPU_WARMED_SUBSTITUTE_SOURCE": "1", "VLLM_TPU_VERIFY_FUSED": "1", "VLLM_TPU_VERIFY_FUSED_READ_OFFSETS": "1",
    "VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE": "1",
    "VLLM_TPU_DECODE_TRIM": DECODE_TRIM, "VLLM_TPU_DECODE_TRIM_DRAFT_STASH": "1",
}
# STACK-D v5 F1 (L4-E 2026-11-14): the DRAFT-ASM family off, VFY-ONE value 2 on, everything else as v5.
F1_ENV = dict(V5_ENV, **{"VLLM_TPU_DRAFT_ASM_FOLD": "0", "VLLM_TPU_DRAFT_ASM_ONE": "0", "VLLM_TPU_DRAFT_ASM_CAP": "0",
                         "VLLM_TPU_DRAFT_ASM_CARRY_LOOP": "0"})


# ---------------------------------------------------------------- the runner segment


SEG_START = "is_async = bool(self.scheduler_config.async_scheduling)"
SEG_END = "host_phases.sub('st_async')"
SEG_PARAMS = (
    "self", "combined_selected_tokens", "combined_selected_tokens_real_lens", "is_spec_step",
    "discard_sampled_tokens_req_indices", "scheduler_output", "next_tokens_per_chunk", "req_ids", "defer_prev_wait",
    "copy_state", "state", "request_seq_lens", "next_tokens_tpu_chunks", "next_token_indices",
)
SEG_MUST = (
    "self.spec_decode_manager.propose_draft_token_ids(", "self._sync_replicated_drafts_across_tp(",
    "self.spec_decode_manager.stage_draft_token_ids_for_host(", "self._assemble_async_spec_substitution(",
    "self._assemble_async_prefill_bootstrap(", "self._update_placeholder(", "AsyncPreResults(",
    "isinstance(eagle3_drafts, torch.Tensor)",
)
_SEG: list = []


def segment():
    """The runner's statements from the unified draft propose to the async block's end, as a function of the
    sample_tokens locals they read (SEG_PARAMS), returning its locals. Read from this tree's tpu_runner.py."""
    if _SEG:
        return _SEG[0]
    path = tpu_runner.__file__
    src = open(path).read()
    tree = ast.parse(src)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TPUModelRunner"]
    assert len(cls) == 1, "TPUModelRunner not found once in tpu_runner.py"
    fns = [n for n in cls[0].body if isinstance(n, ast.FunctionDef) and n.name == "sample_tokens"]
    assert len(fns) == 1, "sample_tokens not found once"
    body = fns[0].body
    texts = [ast.unparse(s) for s in body]
    assert texts.count(SEG_START) == 1 and texts.count(SEG_END) == 1, "the consumer segment's bounds moved"
    i, j = texts.index(SEG_START), texts.index(SEG_END)
    assert i < j
    first, last = body[i].lineno, body[j].lineno - 1
    lines = src.splitlines(keepends=True)
    seg = "".join(lines[first - 1:last])
    missing = [m for m in SEG_MUST if m not in seg]
    assert not missing, f"the consumer segment lacks {missing}"
    code = ("\n" * (first - 2) + "def _q38_foldfix_segment(" + ", ".join(SEG_PARAMS) + "):\n"
            + textwrap.indent(textwrap.dedent(seg), "    ") + "    return dict(locals())\n")
    compiled = compile(code, path, "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
    ns: dict = {}
    exec(compiled, tpu_runner.__dict__, ns)
    fn = ns["_q38_foldfix_segment"]
    fn.lines = (first, last)
    _SEG.append(fn)
    return fn


# ---------------------------------------------------------------- the runner


class _Timing:
    @contextlib.contextmanager
    def stage(self, name):
        yield

    host_stage = stage


def make_rig(p, fence=False):
    """A TPUModelRunner (real methods, no __init__) as the runner of proposer `p` and of a real manager."""
    R = object.__new__(TPUModelRunner)
    for k, v in vars(p.runner).items():
        if k != "_fence_spec_kv_under_token_copy":  # a property of TPUModelRunner (computed below, really)
            setattr(R, k, v)
    R.scheduler_config = SimpleNamespace(async_scheduling=True)
    R.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=K, draft_tensor_parallel_size=None,
                                           use_eagle=lambda: True)
    R._pp_is_last = True
    R._pcp_enabled = False
    R.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=True) if fence else None)
    R.drafter = p
    R._stage_timing = _Timing()
    R._is_dspark_pd_producer = False
    R._mtp_prefill_only = False
    R.is_pooling_model = False
    R.maybe_wait_for_kv_save = lambda: None
    R.get_finished_kv_transfers = lambda scheduler_output: (None, None, None, None, None, None)
    R.prev_waits = []
    R._modify_prev_results = lambda: R.prev_waits.append(1)
    R._pre_async_results = None
    R._tp_rank0_mask = None
    R._last_sequence_layout_plan = None
    R._fast_token_substitution = False
    R._vfy = None
    # VFY-ONE's static eligibility reads these (runner/vfy_one.py VfyOne.static); the VPS head itself is not modelled.
    R._vps_head = object()
    R._vps_fb_path = True
    R._vps_ro_fold = True
    R.mamba_slot_read_offsets = object()
    R._vps_checks_left = 0
    R.spec_decode_manager = SpeculativeDecodingManager(R)
    p.runner = R
    return R


_SUB_BODY = h0.unwrap(tpu_runner._substitute_placeholder_token)  # taken at import, before any monkeypatch


def _substitute_cpu(input_ids, cur_input_indices, pre_next_tokens_indices, next_tokens):
    """The substitution program (``_substitute_placeholder_token``) run as its body on CPU; its int32 index tensors
    are widened to int64 because CPU ``scatter_`` requires int64 indices (same index values)."""
    return _SUB_BODY(input_ids, cur_input_indices.long(), pre_next_tokens_indices.long(), next_tokens)


def set_env(monkeypatch, env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)


def install(monkeypatch, env=None, fold=False, one=False, cap=False, check=0):
    """DRAFT-CAP's install (over DRAFT1-B's and DRAFT-ASM's: served draft env, compiled programs as counted regions),
    then `env`, then the substitution program as its body (the next step's input ids)."""
    hc.install(monkeypatch, fold=fold, cl=False, one=one, cap=cap, check_steps=check)
    if env:
        set_env(monkeypatch, env)
    monkeypatch.setattr(tpu_runner, "_substitute_placeholder_token", _substitute_cpu)


def build(monkeypatch, env=None, fold=False, one=False, cap=False, check=0, fence=False, proposer=None):
    """The rig under the current env: proposer (DRAFT-CAP's stub draft) with the controller Eagle3Proposer.__init__ builds
    under the env (eagle3.py), runner with VFY-ONE as TPUModelRunner.__init__ builds and registers it."""
    install(monkeypatch, env, fold, one, cap, check)
    p = proposer if proposer is not None else hc.make_proposer(fold=False)
    R = make_rig(p, fence=fence)
    if proposer is None and envs.VLLM_TPU_DRAFT_ASM_FOLD:
        if envs.VLLM_TPU_DRAFT_ASM_CARRY_LOOP or envs.VLLM_TPU_DRAFT_ASM_ONE:
            p._draft_asm = draft_one.DraftOne(p)
        else:
            p._draft_asm = draft_asm.DraftAsm(p)
    R._vfy = vfy_one.from_env(envs)
    if R._vfy is not None:
        l4_agree.register(R, R._vfy)
    return R


class ForcedProposer(Eagle3Proposer):
    """An Eagle3Proposer whose propose returns `ret` (a recorded real result): drives the manager and the runner
    with a DraftAsmResult on steps where DraftAsm itself never folds (structured output, no verify output)."""

    def propose(self, *args, **kwargs):
        self.seen.append(kwargs)
        return self.ret


def forced_proposer(ret):
    fp = object.__new__(ForcedProposer)
    fp.runner = h0.make_runner()
    fp._draft_asm = None
    fp.draft_chunks = []
    fp.seen = []
    fp.ret = ret
    return fp


# ---------------------------------------------------------------- one engine step


def engine_step(R, step, serving, structured=False, sync=False, copy_state=None, spec=True):
    """One engine step's draft half: the clock (VFY begin_step, the agreement tick, VFY decide), VFY's seed stash
    on a "vfy" step, then the runner segment (propose -> manager -> consumer), then the next step's input ids."""
    p = R.drafter
    R.warmup_done = serving
    R.scheduler_config.async_scheduling = not sync
    vfy = R._vfy
    if vfy is not None:
        vfy.begin_step(R)  # execute_model: before the tick
    agree = R.__dict__.get("_l4_agree")
    if agree is not None:
        agree.tick(R)
    vmode = None
    if vfy is not None and serving and spec:
        vfy.decide(R, True)
        vmode = vfy.step_mode
    n = step.R
    req_ids = [f"r{i}" for i in range(n)]
    R.input_batch = SimpleNamespace(
        num_reqs=n, req_ids=list(req_ids), req_id_to_index={r: i for i, r in enumerate(req_ids)},
        num_tokens_no_spec=np.full(MAX_REQS, 4000, dtype=np.int32),
    )
    step.chunk.start_index = 0
    p.draft_chunks = [step.chunk]
    stashed = False
    if vmode == "vfy":
        sp = R._vfy_seed_spec(n)
        if sp is not None:  # value 2 with the seed on: TPUModelRunner._vfy_verify's stash
            outs = tuple(sp.fn(step.nt, *sp.args))
            R._vfy_seed = (sp, step.nt, outs, vfy_one.versions((step.nt, *sp.args, *sp.base_args)))
            stashed = True
        vfy.note("vfy")
    elif vmode == "check":
        vfy.note("check")
    ctl = getattr(p, "_draft_asm", None)
    folded0 = ctl.stats["folded"] if ctl is not None else 0
    taken0 = vfy.stats["seed_taken"] if vfy is not None else 0
    rsl = [(i, SimpleNamespace(req_id=r, output_token_ids=[]), 0, r) for i, r in enumerate(req_ids)]
    so = SimpleNamespace(has_structured_output_requests=structured)
    sel = step.nt[:, 0].clone()  # this step's sampled (bonus) tokens, one per verify row
    state = SimpleNamespace(num_reqs_list=[n], spec_decode_metadata_list=[step.md if spec else None])
    nt_chunks = [] if spec else [sel.clone()]  # a non-spec step parked its stride-1 source
    R._pre_async_results = None
    CEN.reset()
    loc = segment()(R, [sel], [n], spec, [], so, [step.nt] if spec else [], req_ids, False, copy_state, state, rsl,
                    nt_chunks, {})
    d = loc["eagle3_drafts"]
    rec = SimpleNamespace(
        n=n, vmode=vmode, stashed=stashed, drafts_obj=d, result_type=type(d).__name__,
        ctl_folded=(ctl.stats["folded"] - folded0) if ctl is not None else 0,
        seed_taken=(vfy.stats["seed_taken"] - taken0) if vfy is not None else 0,
        stash_left="_vfy_seed" in R.__dict__, calls=Counter(CEN.calls), structured=structured, sync=sync,
    )
    if isinstance(d, draft_asm.DraftAsmResult):
        rec.drafts = d.drafts()
    else:
        rec.drafts = d
    if structured or sync:
        rec.host = R.spec_decode_manager.take_draft_token_ids()
    pre = R._pre_async_results
    rec.pre = pre
    if pre is not None:
        rec.src = pre.next_tokens_tpu
        rec.rejected = pre.spec_decode_num_rejected_tokens
        rec.idx = dict(pre.req_id_to_index_copy)
        rec.ndpr = dict(pre.num_draft_per_req or {})
        rec.placeholders = (R.input_batch.num_tokens_no_spec[:n].copy(), [len(s.output_token_ids) for _, s, _, _ in rsl])
        rec.next_ids = next_step_ids(R, n)
    return rec


def next_step_ids(R, n):
    """The next decode step's input ids (each request scheduled with 1 + K placeholder slots), substituted from the
    parked results by the runner's own index builder and substitution."""
    cur, pre = R._prepare_async_token_substitution_indices(0, n, np.full(n, 1 + K, dtype=np.int32))
    ids = torch.full((_get_padded_token_len(PADDINGS, n * (1 + K)),), -7, dtype=torch.int32)
    return R._apply_async_token_substitution(ids, cur, pre).clone()


def same(a, b, what):
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), (what, type(a), type(b))
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, tuple(a.shape), tuple(b.shape))
    assert torch.equal(a, b), what


def same_rec(a, b, tag):
    """Bitwise: drafts, the parked source and rejected count, both index dicts, the placeholder advance, and the next
    step's substituted input ids."""
    same(a.drafts, b.drafts, f"{tag} drafts")
    same(a.src, b.src, f"{tag} source")
    same(a.rejected, b.rejected, f"{tag} rejected count")
    assert a.idx == b.idx and a.ndpr == b.ndpr, f"{tag} index dicts"
    assert np.array_equal(a.placeholders[0], b.placeholders[0]) and a.placeholders[1] == b.placeholders[1], \
        f"{tag} placeholders"
    same(a.next_ids, b.next_ids, f"{tag} next-step input ids")


def step_kw(i):
    """The serving schedule: rows cycle 8 / 32 / 48 / 64; every 7th step forces the extreme verify patterns."""
    return ROWS[i % len(ROWS)], (i % 7 == 3)


def warm(R, seed0):
    """The spec-decode warm-up: one step per row count (warmup_done False)."""
    for j, r in enumerate(ROWS):
        engine_step(R, hc.make_step(seed0 - 1 - j, r), serving=False)
