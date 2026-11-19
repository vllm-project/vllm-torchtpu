# SPDX-License-Identifier: Apache-2.0
"""CPU harness for q38 L4 DRAFT-ASM (VLLM_TPU_DRAFT_ASM_FOLD). Not collected (no test_ prefix).

Drives the real ``Eagle3Proposer.propose`` and the real ``TPUModelRunner._assemble_async_spec_substitution``
for one async decode verify step with every served draft lever on (fused seed with extras, fresh carry,
fused draft loop, compiled spec tail with the padded rejected count). Every compiled program is replaced by
its own body run eagerly inside a "compiled region": the fused seed, both carries, the spec tail, the folded
seed / carries, the two fused draft-loop modules (the unwrapped ``_MtpFusedLoop`` / ``_MtpFusedLoopAsm``
classes) and a stub step-0 draft forward. Recorded per step:

  * outputs: the drafts, step 1's seq_lens / positions / tokens / hidden at the loop call, the source,
    the rejected count and the two index dicts;
  * compiled calls by program, with each call's shape key (the compile census);
  * every aten op dispatched outside a compiled region, view ops excluded (the eager census);
  * loop-call inputs produced by such an op (the defer census: on TPU those are deferred arguments).

The one served condition a CPU tensor cannot meet is the base fused seed's ``nt.device.type == "tpu"``;
``patched_prepare_draft_inputs`` recompiles ``_prepare_draft_inputs`` from the module's own source with that
test widened to CPU (one literal replaced, asserted unique), so the base path is the served one.
"""

import __future__

import contextlib
import dataclasses
import inspect
import textwrap
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutDescriptor
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _get_padded_num_reqs_with_upper_limit,
    _get_padded_token_len,
)
from vllm_torchtpu.spec_decode import draft_asm, eagle3
from vllm_torchtpu.spec_decode.eagle3 import DraftChunkInputs, Eagle3Proposer
from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop

K = 3
H = 16
V = 97
MAX_REQS = 64
PADDINGS = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
INVALID = eagle3.INVALID_TOKEN_ID

SERVED_ENV = {
    "VLLM_TPU_FUSED_ASYNC_DRAFT_SEED": "1",
    "VLLM_TPU_DRAFT_SEED_FUSED": "1",
    "VLLM_TPU_DRAFT_SEED_EXTRAS": "1",
    "VLLM_TPU_DRAFT_CARRY_FUSED": "1",
    "VLLM_TPU_DRAFT_CARRY_ROWS_FRESH": "1",
    "VLLM_TPU_FUSED_DRAFT_LOOP": "1",
    "VLLM_TPU_MTP_DRAFT_ROWS": "1",
    "VLLM_TPU_PROPOSE_DEVICE_CONSTANTS": "1",
    "VLLM_TPU_SPEC_TAIL_COMPILED": "1",
    "VLLM_TPU_SPEC_TAIL_REJ_PAD": "1",
}
CLEAR_ENV = (
    "VLLM_TPU_DRAFT_ASM_FOLD",
    "VLLM_TPU_DRAFT_ASM_CHECK_STEPS",
    "VLLM_TPU_DRAFT_ASM_AB",
    "VLLM_TPU_VERIFY_FUSED_CHECK",
    "VLLM_TPU_DRAFT_ROWS_FRESH",
)

# aten ops that only make a view or alias (no device program on their own)
VIEW_OPS = {
    "view", "_unsafe_view", "slice", "select", "alias", "detach", "unsqueeze", "squeeze", "expand",
    "t", "transpose", "permute", "as_strided", "_reshape_alias", "unbind", "split", "narrow",
    "split_with_sizes", "view_as", "expand_as",
}


# ---------------------------------------------------------------- census


class Census(TorchDispatchMode):
    """Eager aten ops outside compiled regions; their outputs are tagged ``_q38_eager``."""

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.on = False
        self.eager: list[str] = []
        self.calls: Counter = Counter()
        self.keys: list[tuple] = []
        self.deferred: list[str] = []
        self.loop_entry: dict = {}

    def reset(self):
        self.eager, self.calls, self.keys, self.deferred, self.loop_entry = [], Counter(), [], [], {}
        self.step_id = getattr(self, "step_id", 0) + 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if self.on and self.depth == 0:
            name = func.overloadpacket.__name__
            if name not in VIEW_OPS:
                self.eager.append(name)
                for t in pytree.tree_leaves(out):
                    if isinstance(t, torch.Tensor):
                        with contextlib.suppress(Exception):
                            t._q38_eager = self.step_id
        return out


CEN = Census()


def _shape_key(args):
    return draft_asm._key(*[a for a in args if not isinstance(a, Eagle3Proposer)])


def region(name, fn):
    """`fn` as one compiled program: counted, keyed, and not part of the eager census."""

    def wrapped(*args, **kwargs):
        CEN.calls[name] += 1
        CEN.keys.append((name, _shape_key(args)))
        CEN.depth += 1
        try:
            return fn(*args, **kwargs)
        finally:
            CEN.depth -= 1

    wrapped.__name__ = f"region_{name}"
    return wrapped


def unwrap(fn):
    """The Python body under eagle3's ``_static_shape_args`` and ``torch.compile``."""
    inner = getattr(fn, "__wrapped__", fn)
    return getattr(inner, "_torchdynamo_orig_callable", None) or getattr(inner, "__wrapped__", None) or inner


# ---------------------------------------------------------------- the stub draft model


CTX: dict = {}


@contextlib.contextmanager
def fake_forward_context(attn_metadata, vllm_config=None, **_kw):
    CTX["md"] = attn_metadata
    try:
        yield
    finally:
        CTX.pop("md", None)


class FakeMtp:
    """A deterministic MTP stand-in: a loop step reads seq_lens from the forward context, as the draft
    attention does, so a wrong step-1 seq_lens or positions changes its hidden state."""

    uses_target_aux_hidden_states = False

    def __init__(self):
        g = torch.Generator().manual_seed(11)
        self.W = torch.randn(H, V, generator=g)

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None,
                spec_step_idx=0):
        md = CTX["md"]
        seq = next(iter(md.values())).seq_lens
        p = hidden_states.shape[0]
        s = seq[:p].float() if seq.shape[0] >= p else torch.nn.functional.pad(seq.float(), (0, p - seq.shape[0]))
        h = hidden_states * 0.5 + (
            input_ids.float() * 0.01 + positions.float() * 0.37 + s * 0.53 + spec_step_idx
        ).unsqueeze(1)
        return h, h

    def compute_logits(self, h):
        return torch.sin(h.float()) @ self.W


class LoopRegion:
    """A fused loop module as one compiled program, with the defer census of its inputs."""

    def __init__(self, module, name):
        self.module, self.name = module, name

    def __call__(self, input_ids, positions, hidden_states, seq_lens, *rest):
        md = CTX.get("md") or {}
        mds = list({id(m): m for m in md.values()}.values())
        for what, t in [("input_ids", input_ids), ("positions", positions), ("hidden_states", hidden_states),
                        ("seq_lens", seq_lens), *[(f"tail{i}", r) for i, r in enumerate(rest)]] + [
                (f"md.{f}", getattr(m, f)) for m in mds for f in ("seq_lens", "query_start_loc", "request_distribution")]:
            if isinstance(t, torch.Tensor) and getattr(t, "_q38_eager", None) == CEN.step_id:
                CEN.deferred.append(what)
        CEN.calls[self.name] += 1
        CEN.keys.append((self.name, draft_asm._key(input_ids, positions, hidden_states, seq_lens, *rest)))
        CEN.depth += 1
        try:
            CEN.loop_entry = {
                "positions": positions.clone(), "seq_lens": seq_lens.clone(),
                "input_ids": input_ids.clone(), "hidden": hidden_states.clone(),
            }
            return self.module(input_ids, positions, hidden_states, seq_lens, *rest)
        finally:
            CEN.depth -= 1


def fwd0(self, chunk, input_ids, positions, target_hidden_states, step_idx, seq_lens_delta, num_rejected_np,
         num_tokens_padded=None, num_rejected_dev=None, loop_query_start_loc=None,
         loop_request_distribution=None, draft_rows=None):
    """Step 0's draft forward at `draft_rows` (one compiled program in serving)."""
    rows = draft_rows.clamp(min=0)
    h = target_hidden_states[rows] * 0.25 + (
        input_ids[rows].float() * 0.01 + positions[rows].float() * 0.001
    ).unsqueeze(1)
    return h, h * 2


# ---------------------------------------------------------------- the patched base seed device test


# The originals, taken once at import (install() patches the class attributes; monkeypatch restores them).
_ORIG_CARRY_FRESH = unwrap(Eagle3Proposer._draft_rows_carry_token_fresh)
_ORIG_CARRY = unwrap(Eagle3Proposer._draft_rows_carry_and_token)
_ORIG_PREPARE_SRC = textwrap.dedent(inspect.getsource(Eagle3Proposer._prepare_draft_inputs))
_PATCHED: list = []


def patched_prepare_draft_inputs():
    if _PATCHED:
        return _PATCHED[0]
    src = _ORIG_PREPARE_SRC
    old = 'and nt.device.type == "tpu"'
    assert src.count(old) == 1, "the base fused seed's device test moved"
    src = src.replace(old, 'and nt.device.type in ("tpu", "cpu")')
    code = compile(src, eagle3.__file__, "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
    ns: dict = {}
    exec(code, eagle3.__dict__, ns)
    _PATCHED.append(ns["_prepare_draft_inputs"])
    return _PATCHED[0]


# ---------------------------------------------------------------- setup


def install(monkeypatch, fold: bool, check_steps: int = 0, ab: str | None = None):
    """Served env, compiled programs as regions, the fake contexts. Returns nothing; call make_proposer next."""
    for name in CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    for k, v in SERVED_ENV.items():
        monkeypatch.setenv(k, v)
    if fold:
        monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_FOLD", "1")
    if check_steps:
        monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_CHECK_STEPS", str(check_steps))
    if ab is not None:
        monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_AB", ab)
    monkeypatch.setattr(eagle3, "_async_seed_inputs_compiled", region("seed", eagle3._async_seed_inputs_impl))
    monkeypatch.setattr(Eagle3Proposer, "_draft_rows_carry_token_fresh", region("carry_fresh", _ORIG_CARRY_FRESH))
    monkeypatch.setattr(Eagle3Proposer, "_draft_rows_carry_and_token", region("carry", _ORIG_CARRY))
    monkeypatch.setattr(draft_asm, "_async_seed_inputs_asm_compiled",
                        region("seed_asm", draft_asm._async_seed_inputs_asm_impl))
    monkeypatch.setattr(draft_asm, "_carry_token_fresh_asm_compiled",
                        region("carry_fresh_asm", draft_asm._carry_token_fresh_asm_impl))
    monkeypatch.setattr(draft_asm, "_carry_and_token_asm_compiled",
                        region("carry_asm", draft_asm._carry_and_token_asm_impl))
    monkeypatch.setattr(tpu_runner, "_spec_tail_compiled", region("spec_tail", tpu_runner._spec_tail))
    monkeypatch.setattr(Eagle3Proposer, "_forward_draft", region("draft_fwd0", fwd0))
    monkeypatch.setattr(Eagle3Proposer, "_prepare_draft_inputs", patched_prepare_draft_inputs())
    monkeypatch.setattr(draft_asm, "_is_tpu", lambda t: True)
    for mod in (eagle3, draft_asm):
        monkeypatch.setattr(mod, "set_forward_context", fake_forward_context)
        monkeypatch.setattr(mod, "set_vllm_model_wrapper_context", lambda **_kw: contextlib.nullcontext())
    monkeypatch.setattr(draft_asm, "set_current_vllm_config", lambda *_a, **_kw: contextlib.nullcontext())
    monkeypatch.setattr(draft_asm, "synchronize_tensors", lambda *a, **k: None)
    monkeypatch.setattr(draft_asm, "_L6_AB", [])

    def build_asm(*, vllm_config, prefix, mtp, num_steps, dynamic_arg_dims, dvp=False, max_reqs):
        m = draft_asm._MtpFusedLoopAsm(vllm_config=None, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp,
                                       max_reqs=max_reqs)
        return LoopRegion(m, "loop_asm")

    monkeypatch.setattr(draft_asm, "build_mtp_fused_loop_asm", build_asm)


def make_runner(cap=64):
    return SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(num_reqs=0, req_ids=[]),
        num_reqs_most_model_len=cap,
        num_reqs_max_model_len=cap,
        num_tokens_paddings=list(PADDINGS),
        num_reqs_paddings=[8, 16, 32, 64],
        max_num_reqs=MAX_REQS,
        uses_mrope=False,
        mesh=None,
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda n: None,
        _dp_step_num_chunks=1,
        warmup_done=False,
        _fence_spec_kv_under_token_copy=False,
        _precompile_timed=lambda name: contextlib.nullcontext(),
        _hidden_states_dtype=torch.float32,
        enforce_eager=False,
    )


def make_proposer(fold: bool, cap=64):
    """An Eagle3Proposer (MTP, K=3, draft rows) without __init__'s model plumbing; with `fold` its controller
    is built as __init__ builds it under the flag."""
    p = object.__new__(Eagle3Proposer)
    p.runner = make_runner(cap)
    p.speculative_config = SimpleNamespace(num_speculative_tokens=K, method="mtp", draft_tensor_parallel_size=None)
    p.draft_chunks = []
    p._draft_md_cache = {}
    p._seed_extras = {}
    p._draft_rows = True
    p._share_mtp_indices = False
    p._draft_kv_shared = False
    p._draft_attn_layer_names = {"draft.attn.0"}
    p._seed_checks_done = 0
    p._carry_checks_done = 0
    p.draft_model = FakeMtp()
    p.draft_vllm_config = None
    p._dvp = False
    p._draft_top_checks_left = 0
    p._mtp_fused_loop = LoopRegion(
        _MtpFusedLoop(vllm_config=None, prefix="t", mtp=p.draft_model, num_steps=K - 1), "loop"
    )
    p._mtp_fused_loop_warm = False
    p._draft_hidden_size = lambda: H
    if fold:
        p._draft_asm = draft_asm.DraftAsm(p)
    return p


def make_assembler_runner(p):
    r = object.__new__(TPUModelRunner)
    r.drafter = p
    r.speculative_config = SimpleNamespace(num_speculative_tokens=K)
    r.scheduler_config = SimpleNamespace(async_scheduling=True)
    r.max_num_reqs = MAX_REQS
    r.device = torch.device("cpu")
    return r


# ---------------------------------------------------------------- one verify step


@dataclasses.dataclass
class Step:
    R: int
    chunk: DraftChunkInputs
    nt: torch.Tensor
    md: SimpleNamespace


def make_step(seed: int, R: int, cap=64, fallback=False, short=False, mixed=False):
    """One decode verify chunk of R requests (each 1 + d_i tokens), the verify output and its spec metadata.
    Deterministic in `seed`: two calls give equal values in fresh tensors. `fallback` forces up to 8 rows
    to the extreme verify patterns (all K accepted + bonus, and bonus only) and leaves the padded rows
    INVALID; `short` gives some requests
    fewer than K drafts; `mixed` makes request 0 a 2K-token prefill chunk (not a decode verify)."""
    g = np.random.default_rng(seed)
    width = _get_padded_num_reqs_with_upper_limit(R, MAX_REQS)
    d = np.full(R, K, dtype=np.int32)
    if short:
        d = g.choice([K, K, 1, 2, 0], R).astype(np.int32)
    n_tok = 1 + d
    if mixed:
        n_tok[0] = 2 * K + 1
    qsl_np = np.concatenate([[0], np.cumsum(n_tok)]).astype(np.int32)
    total = int(qsl_np[-1])
    T = _get_padded_token_len(PADDINGS, total)
    input_ids = torch.from_numpy(g.integers(1, V, T).astype(np.int32))
    positions = torch.from_numpy(g.integers(0, 50000, T).astype(np.int32))
    seq = np.zeros(cap, dtype=np.int32)
    seq[:R] = g.integers(100, 60000, R)
    seq_lens = torch.from_numpy(seq)
    qsl_dev = np.full(cap + 1, total, dtype=np.int32)
    qsl_dev[: R + 1] = qsl_np
    qsl_t = torch.from_numpy(qsl_dev)
    rd = torch.tensor([0, R, R], dtype=torch.int32)
    hidden = torch.from_numpy(g.standard_normal((T, H)).astype(np.float32))
    dl = np.zeros(width, dtype=np.int32)
    dl[:R] = d
    nt = np.full((width, K + 1), INVALID, dtype=np.int32)
    acc = np.minimum(g.integers(0, K + 1, R), d)
    if fallback:
        for i in range(min(8, R)):
            acc[i] = d[i] if i % 2 == 0 else 0
    for i in range(R):
        nt[i, : acc[i] + 1] = g.integers(1, V, acc[i] + 1)
    if not fallback:
        nt[R:] = g.integers(1, V, (width - R, K + 1))  # filler tokens in the padded rows
    # (fallback: the padded rows stay INVALID, so a rejected count there would read 1, not 0)
    ctx = SimpleNamespace(
        use_max_model_len=False,
        seq_lens=seq_lens,
        query_start_loc=qsl_t,
        request_distribution=rd,
        position_ids_override=None,
        sequence_layout_descriptor=SequenceLayoutDescriptor(),
    )
    shared = AttentionMetadata(
        input_positions=positions, block_tables=torch.arange(16, dtype=torch.int32),
        seq_lens=seq_lens, query_start_loc=qsl_t, request_distribution=rd,
    )
    chunk = DraftChunkInputs(
        input_ids=input_ids,
        position_ids=positions,
        query_start_loc_np=qsl_np,
        attn_ctx=ctx,
        start_index=0,
        num_reqs=R,
        aux_hidden_states=[],
        hidden_states=hidden,
        draft_lengths=torch.from_numpy(dl),
        attn_metadata={"target.attn.0": shared, "draft.attn.0": shared},
    )
    md = SimpleNamespace(draft_lengths=chunk.draft_lengths, draft_lengths_cpu=dl[:R].copy())
    return Step(R, chunk, torch.from_numpy(nt), md)


@dataclasses.dataclass
class StepOut:
    drafts: torch.Tensor  # [R, K]
    src: torch.Tensor
    rejected: torch.Tensor
    idx: dict
    ndpr: dict
    folded: bool
    eager: list
    calls: Counter
    keys: list
    deferred: list
    loop_entry: dict


def run_step(p, step, serving: bool, structured=False) -> StepOut:
    """propose + the runner's async spec substitution for `step` (a Step, or a list of Steps = one chunk
    each, laid out back to back), under the census."""
    steps = step if isinstance(step, list) else [step]
    start = 0
    for s in steps:
        s.chunk.start_index = start
        start += s.R
    p.runner.warmup_done = serving
    agree = p.runner.__dict__.get("_l4_agree")
    if agree is not None:
        # q38 L4 AGREE: one engine step = one tick of the runner's agreement clock (execute_model's hook)
        agree.tick(p.runner)
    p.runner.input_batch = SimpleNamespace(num_reqs=start, req_ids=[f"r{i}" for i in range(start)])
    p.draft_chunks = [s.chunk for s in steps]
    nts = [s.nt for s in steps]
    CEN.reset()
    with CEN:
        CEN.on = True
        try:
            drafts = p.propose(
                sampled_token_ids=None,
                discard_sampled_tokens_req_indices=[],
                num_rejected_tokens_np=None,
                scheduler_output=SimpleNamespace(has_structured_output_requests=structured),
                return_device=True,
                next_tokens_per_chunk=nts,
            )
            state = SimpleNamespace(
                num_reqs_list=[s.R for s in steps], spec_decode_metadata_list=[s.md for s in steps]
            )
            srcs, idx, rej, ndpr = TPUModelRunner._assemble_async_spec_substitution(
                make_assembler_runner(p), drafts, nts, state
            )
        finally:
            CEN.on = False
    folded = isinstance(drafts, draft_asm.DraftAsmResult)
    d = drafts.drafts() if folded else drafts
    src = srcs[0] if len(srcs) == 1 else torch.cat(srcs)
    return StepOut(d, src, rej, idx, ndpr, folded, list(CEN.eager), Counter(CEN.calls), list(CEN.keys),
                   list(CEN.deferred), dict(CEN.loop_entry))


def same(a, b, what):
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), what
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, tuple(a.shape), tuple(b.shape))
    assert torch.equal(a, b), what


def assert_same_outputs(a: StepOut, b: StepOut, tag: str):
    same(a.drafts, b.drafts, f"{tag} drafts")
    same(a.src, b.src, f"{tag} source")
    same(a.rejected, b.rejected, f"{tag} rejected count")
    assert a.idx == b.idx and a.ndpr == b.ndpr, f"{tag} index dicts"
    for k in ("seq_lens", "positions", "input_ids", "hidden"):
        same(a.loop_entry[k], b.loop_entry[k], f"{tag} loop-call {k}")


def warm(p, seeds_rs, **kw):
    """One warm-up step (runner.warmup_done False) per (seed, R)."""
    for seed, R in seeds_rs:
        run_step(p, make_step(seed, R, **kw), serving=False)
