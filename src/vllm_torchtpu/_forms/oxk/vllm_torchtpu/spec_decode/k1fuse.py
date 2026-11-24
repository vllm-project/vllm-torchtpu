# SPDX-License-Identifier: Apache-2.0
"""q38 L4 T1-K1FUSE: the VFY-ONE verify and the draft as ONE compiled program per decode step (the K-step loop's
R4 contract at K = 1; host_levers_h2 card T1-K1FUSE).

Today's served chain with VLLM_TPU_VPS_VFY_ONE=2, DRAFT-ASM and DH-D runs two compiled calls per steady decode step:
``tpu_runner._vfy_one_compiled`` (the VPS verify with D83's read-offset operands, the stats accumulate and the draft's
superset seed, DH-C) in sample_tokens, then DH-D's ``draft_one._DraftOneProgram`` (draft step 0, the carry and the K-1
loop steps, the spec tail's source) in propose. With VLLM_TPU_K1FUSE=1 one program, ``k1fuse_body``, runs
``tpu_runner._vfy_one_impl``'s body (no seed) and then ``draft_one.draft_body`` (the frozen K-step draft interface:
the superset seed + DH-D's body) on the verify's own tokens, and returns (tokens, stats_acc, read-offset indices,
read-offset values, drafts, source, rejected).

Seam (KLOOP R4, scratch/l4/draft1b/t1_k1fuse_notes.md): (1) nt [W, 1+k] is output 0, so the sampled-token D2H and
D83's read-offset scatter start from the program's outputs; (2) source [max_reqs*(1+k)] / rejected [max_reqs] int32
are draft_body's outputs, untouched; (3) the three uniform draws (and the fallback rows' draw) stay eager in
``TPUModelRunner._vfy_verify``'s prologue, unchanged: only the program call is replaced; (4) every input is a device
tensor or a static baked into the instance; the forward context holds the draft-layer metadata replaced to DH-D's
persistent buffers, exactly as DH-D's ``_one_ready`` builds it.

Programs. The module is ``support_torch_compile``'d (vLLM's TorchCompileWithNoGuardsWrapper: every Dynamo guard is
dropped), so the per-step forward context and metadata objects never retrace it. Because guards are dropped, every
shape other than the token dim T and every static is part of the instance key: one instance per static signature
(the full call key with T abstracted out), each with its own forward code object (as ``draft_one._program_class``)
and serving only the token buckets of its steady decode shapes (``compilation.bucket_filter``). The serving step runs
K1 only at a key recorded at warm-up; every other step runs today's chain (counted).

Warm-up. VFY-ONE is never engaged in warm-up, so K1 warms on the draft side: on every warm-mode propose (the real
spec-decode warm-up's synthetic decode steps) ``DraftK1.forward0`` runs one K1 call per fallback-row variant at the
step's real draft inputs and forward context, with zero verify inputs built as ``_vfy_precompile`` builds them (so the
keys are the serving keys). It runs first, so the base path's KV writes stand (as DH-D's warm-up does); outputs are
discarded.

All-rank agreement (lead safety rule). K1 switches programs that carry TP collectives, so it changes only at
DRAFT-ASM's agreement boundaries through DraftOne's one CPU-group all_gather: entry 0 carries BIT_K1 (the window
wish), BIT_K1CONF (K1 configured on this rank), BIT_K1CLOSE (this rank's K1 CHECK window is closed) and BIT_K1BAD
(this rank saw a K1 CHECK mismatch); the K1 warm keys are in DRAFT-ASM's warm-key fingerprint. K1 is on until the
next boundary iff the fold, DH-D and K1 are agreed and every rank wishes it. A K1-configured rank RAISES (fail-stop)
when the BIT_K1CONF bits or the fingerprints differ across ranks. Per-step choices (VFY's step mode, eligibility,
warm key) depend only on rank-uniform inputs. DRAFT-ASM's boundary is in propose, after the verify, so a decision
reached at propose n applies from verify n+1; the take at propose n is unconditional when a result exists (every
rank holds one). The take checks chunk / token identity and the in-place version counters of the tokens and of every
draft input; a miss after K1 ran raises (its draft KV writes used the old inputs). Any exception in a K1 call or
build raises.

CHECK (VLLM_TPU_K1FUSE_CHECK=N, default 256; L4-E): the first N K1-eligible serving steps run K1 FIRST, then today's
chain for the step's results, and compare tokens, stats, read offsets (at the propose) and drafts, source and rejected
count (at its end). Mismatches count; the next boundary after the window turns K1 off on every rank for the run
(not a fail-stop). K1 serves only after its CHECK window closed clean, and only with DRAFT-ASM's and VFY-ONE's CHECK
windows closed.
"""

import collections
import dataclasses
import types
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import bucket_filter
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import host_phases, tpu_runner, vfy_one, vps_wiring
from vllm_torchtpu.spec_decode import draft_asm, draft_cap, draft_one
from vllm_torchtpu.spec_decode.draft_asm import DraftAsmResult
from vllm_torchtpu.spec_decode.draft_one import DraftOne, _meta_i32
from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop

logger = init_logger(__name__)

# core/l6_ab.py lever of the window (VLLM_TPU_K1FUSE_AB).
LEVER = "k1fuse"
# Agreement entry 0 bits above DraftOne's (BIT_FOLD 1, BIT_CL 2, BIT_ONE 4).
BIT_K1, BIT_K1CONF, BIT_K1CLOSE, BIT_K1BAD = 8, 16, 32, 64
LOG_EVERY = 5000
_ON = ("", "1", "true", "on")
_OFF = ("0", "false", "off")

# The program's tensor arguments, in forward order. T_ARGS carry the token dim T as dim 0 (the one dynamic dim).
ARG_NAMES = (
    "hidden_states", "logits_indices", "bonus_logits_indices", "target_logits_indices", "temperatures", "top_k",
    "top_p", "bonus_u", "accept_u", "recover_u", "draft_token_ids", "draft_lengths", "segment_ids", "group_indices",
    "synthetic_conditional_rates", "num_active", "fb_u", "ro_groups", "ro_num_reqs", "stats_acc", "num_draft",
    "query_start_loc", "num_reqs", "input_ids", "positions", "seq_lens", "request_distribution", "target_hidden",
    "loop_query_start_loc", "loop_request_distribution", "seq_lens_buf", "query_start_loc_buf",
    "request_distribution_buf",
)
T_ARGS = ("hidden_states", "logits_indices", "target_logits_indices", "accept_u", "recover_u", "draft_token_ids",
          "segment_ids", "group_indices", "input_ids", "positions", "target_hidden")
T_POS = tuple(ARG_NAMES.index(n) for n in T_ARGS)
K1_DYNAMIC_ARG_DIMS = {n: 0 for n in T_ARGS}
# q38 L4 v6 composition (V6COMP-BUILD), K1 x DRAFT-CAP: with VLLM_TPU_DRAFT_ASM_CAP on, the program's draft half is
# DRAFT-CAP's R_d body (``_draft_body_cap``), which also takes each distinct per-group block table and its [R_d * cols]
# buffer (unused slots None), as draft_cap._DraftCapProgram does.
CAP_ARG_NAMES = tuple(f"table{i}" for i in range(draft_cap.TABLE_SLOTS)) + tuple(
    f"table_buf{i}" for i in range(draft_cap.TABLE_SLOTS)
)


class K1Statics(NamedTuple):
    """The trace-time statics of one instance (baked; part of the instance key)."""

    k: int
    synthetic_mode: bool
    fallback_rows: int
    cand_draws: bool
    ro_dtype: torch.dtype
    p: int
    num_rows: int
    max_reqs: int


# ---------------------------------------------------------------- the traced body


def k1fuse_body(
    loop: Any,
    vh: Any,
    hidden_states: torch.Tensor,
    logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    bonus_u: torch.Tensor,
    accept_u: torch.Tensor,
    recover_u: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_lengths: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor | None,
    num_active: torch.Tensor,
    fb_u: torch.Tensor | None,
    ro_groups: list,
    ro_num_reqs: torch.Tensor,
    stats_acc: torch.Tensor,
    num_draft: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_reqs: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    request_distribution: torch.Tensor,
    target_hidden: torch.Tensor,
    loop_query_start_loc: torch.Tensor,
    loop_request_distribution: torch.Tensor,
    bufs: tuple,
    s: K1Statics,
    tables: tuple | None = None,
    table_bufs: tuple | None = None,
) -> tuple[torch.Tensor, ...]:
    """VFY-ONE's verify (``_vfy_one_impl`` without the seed: local logits, the VPS verify with the fallback rows,
    D83's read-offset operands, the stats accumulate) and then ``draft_body`` on the verify's tokens. No host read.
    Returns (tokens [W, 1+k], stats_acc [3] int64, read-offset indices, read-offset values, drafts [p, k],
    source [max_reqs*(1+k)], rejected [max_reqs])."""
    tokens, acc, ro_idx, ro_vals = tpu_runner._vfy_one_impl(
        vh, hidden_states, logits_indices, bonus_logits_indices, target_logits_indices, temperatures, top_k, top_p,
        bonus_u, accept_u, recover_u, draft_token_ids, draft_lengths, segment_ids, group_indices,
        synthetic_conditional_rates, s.k, s.synthetic_mode, num_active, s.fallback_rows, s.cand_draws, fb_u,
        list(ro_groups), ro_num_reqs, s.ro_dtype, stats_acc,
    )
    # q38 L4 v6 composition (V6COMP-BUILD): with DRAFT-CAP's tables the draft half runs at R_d request slots.
    body = draft_one.draft_body if tables is None else _draft_body_cap
    extra = {} if tables is None else {"tables": tables, "table_bufs": table_bufs}
    drafts, source, rejected = body(
        loop,
        next_tokens=tokens,
        num_draft=num_draft,
        query_start_loc=query_start_loc,
        num_reqs=num_reqs,
        input_ids=input_ids,
        positions=positions,
        seq_lens=seq_lens,
        request_distribution=request_distribution,
        target_hidden=target_hidden,
        loop_query_start_loc=loop_query_start_loc,
        loop_request_distribution=loop_request_distribution,
        bufs=bufs,
        p=s.p,
        num_rows=s.num_rows,
        max_reqs=s.max_reqs,
        **extra,
    )
    return tokens, acc, ro_idx, ro_vals, drafts, source, rejected


def _draft_body_cap(
    loop: Any,
    *,
    next_tokens: torch.Tensor,
    num_draft: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_reqs: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    request_distribution: torch.Tensor,
    target_hidden: torch.Tensor,
    loop_query_start_loc: torch.Tensor,
    loop_request_distribution: torch.Tensor,
    bufs: tuple,
    tables: tuple,
    table_bufs: tuple,
    p: int,
    num_rows: int,
    max_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q38 L4 v6 composition (V6COMP-BUILD), K1 x DRAFT-CAP: ``draft_one.draft_body`` with DH-D's body at R_d =
    len(bufs[0]) request slots, i.e. the superset seed and then ``draft_cap.dh_cap_body`` with exactly the arguments
    DRAFT-CAP's ``ready`` gives the DH-D R_d program (step 0's draft_rows = the seed's [p] gather indices when
    p == R_d, else its capacity rows; md0 / md1 / tables capacity-sized, read only). Same returns as draft_body."""
    seed = draft_one._seed_cl_impl(
        next_tokens, num_draft, query_start_loc, num_reqs, input_ids, seq_lens, positions, p, num_rows,
        max_reqs,
    )
    r_d = bufs[0].shape[0]
    rows = seed[1] if seed[1].shape[0] == r_d else seed[3]
    out = draft_cap.dh_cap_body(
        loop,
        input_ids=seed[0],
        positions=positions,
        target_hidden=target_hidden,
        draft_rows=rows,
        positions_step1=seed[draft_one.SEED_POS1],
        md0=(seq_lens, query_start_loc, request_distribution),
        md1=(seed[draft_one.SEED_SEQ1], loop_query_start_loc, loop_request_distribution),
        bufs=bufs,
        tables=tables,
        table_bufs=table_bufs,
        tail_bonus=seed[draft_one.SEED_BONUS],
        num_reqs=num_reqs,
        max_reqs=max_reqs,
    )
    return torch.stack(out[:-1], dim=1), out[-1], seed[draft_one.SEED_REJ]


class _K1FuseProgram(_MtpFusedLoop):
    """The verify + draft program at one static signature (``k1fuse_body``). The token dim T of the verify rows and
    of draft step 0's inputs is the one dynamic dim; every other shape and the statics are fixed per instance."""

    def __init__(
        self,
        *,
        vllm_config: Any,
        prefix: str = "",
        mtp: Any,
        num_steps: int,
        dvp: bool = False,
        vh: Any = None,
        statics: K1Statics | None = None,
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp)
        self.vh = vh
        # The target LM head and its processor ride the program as submodules, as DH-D shares the draft.
        for name, sub in (("vfy_processor", getattr(vh, "processor", None)), ("vfy_lm_head", getattr(vh, "head", None))):
            if isinstance(sub, torch.nn.Module):
                setattr(self, name, sub)
        self.k1_statics = statics

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits_indices: torch.Tensor,
        bonus_logits_indices: torch.Tensor,
        target_logits_indices: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        bonus_u: torch.Tensor,
        accept_u: torch.Tensor,
        recover_u: torch.Tensor,
        draft_token_ids: torch.Tensor,
        draft_lengths: torch.Tensor,
        segment_ids: torch.Tensor,
        group_indices: torch.Tensor,
        synthetic_conditional_rates: torch.Tensor | None,
        num_active: torch.Tensor,
        fb_u: torch.Tensor | None,
        ro_groups: list,
        ro_num_reqs: torch.Tensor,
        stats_acc: torch.Tensor,
        num_draft: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_reqs: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        request_distribution: torch.Tensor,
        target_hidden: torch.Tensor,
        loop_query_start_loc: torch.Tensor,
        loop_request_distribution: torch.Tensor,
        seq_lens_buf: torch.Tensor,
        query_start_loc_buf: torch.Tensor,
        request_distribution_buf: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return k1fuse_body(
            self, self.vh, hidden_states, logits_indices, bonus_logits_indices, target_logits_indices, temperatures,
            top_k, top_p, bonus_u, accept_u, recover_u, draft_token_ids, draft_lengths, segment_ids, group_indices,
            synthetic_conditional_rates, num_active, fb_u, ro_groups, ro_num_reqs, stats_acc, num_draft,
            query_start_loc, num_reqs, input_ids, positions, seq_lens, request_distribution, target_hidden,
            loop_query_start_loc, loop_request_distribution,
            (seq_lens_buf, query_start_loc_buf, request_distribution_buf), self.k1_statics,
        )


def _program_class(index: int) -> type:
    """A subclass of `_K1FuseProgram` whose forward has a code object of its own (vLLM's compile wrapper and Dynamo
    key compiled code on the forward's code object; two instances sharing one would share, guards dropped, the
    first one's graph). As ``draft_one._program_class``."""
    base = _K1FuseProgram.forward
    code = base.__code__.replace(co_name=f"forward_k1_{index}")
    fwd = types.FunctionType(code, base.__globals__, f"forward_k1_{index}", base.__defaults__, base.__closure__)
    fwd.__qualname__ = f"K1FuseProgram{index}.forward"
    fwd.__annotations__ = dict(base.__annotations__)
    return type(f"K1FuseProgram{index}", (_K1FuseProgram,), {"forward": fwd})


class _K1FuseCapProgram(_K1FuseProgram):
    """q38 L4 v6 composition (V6COMP-BUILD), K1 x DRAFT-CAP: the verify + draft program whose draft half runs at R_d
    request slots (``_draft_body_cap``): K1's arguments, then each distinct per-group block table and its [R_d * cols]
    buffer (unused slots None), as ``draft_cap._DraftCapProgram`` takes them."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits_indices: torch.Tensor,
        bonus_logits_indices: torch.Tensor,
        target_logits_indices: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        bonus_u: torch.Tensor,
        accept_u: torch.Tensor,
        recover_u: torch.Tensor,
        draft_token_ids: torch.Tensor,
        draft_lengths: torch.Tensor,
        segment_ids: torch.Tensor,
        group_indices: torch.Tensor,
        synthetic_conditional_rates: torch.Tensor | None,
        num_active: torch.Tensor,
        fb_u: torch.Tensor | None,
        ro_groups: list,
        ro_num_reqs: torch.Tensor,
        stats_acc: torch.Tensor,
        num_draft: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_reqs: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        request_distribution: torch.Tensor,
        target_hidden: torch.Tensor,
        loop_query_start_loc: torch.Tensor,
        loop_request_distribution: torch.Tensor,
        seq_lens_buf: torch.Tensor,
        query_start_loc_buf: torch.Tensor,
        request_distribution_buf: torch.Tensor,
        table0: torch.Tensor | None = None,
        table1: torch.Tensor | None = None,
        table2: torch.Tensor | None = None,
        table3: torch.Tensor | None = None,
        table_buf0: torch.Tensor | None = None,
        table_buf1: torch.Tensor | None = None,
        table_buf2: torch.Tensor | None = None,
        table_buf3: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return k1fuse_body(
            self, self.vh, hidden_states, logits_indices, bonus_logits_indices, target_logits_indices, temperatures,
            top_k, top_p, bonus_u, accept_u, recover_u, draft_token_ids, draft_lengths, segment_ids, group_indices,
            synthetic_conditional_rates, num_active, fb_u, ro_groups, ro_num_reqs, stats_acc, num_draft,
            query_start_loc, num_reqs, input_ids, positions, seq_lens, request_distribution, target_hidden,
            loop_query_start_loc, loop_request_distribution,
            (seq_lens_buf, query_start_loc_buf, request_distribution_buf), self.k1_statics,
            tables=(table0, table1, table2, table3),
            table_bufs=(table_buf0, table_buf1, table_buf2, table_buf3),
        )


def _cap_program_class(index: int) -> type:
    """``_program_class`` for the R_d (DRAFT-CAP) program: a forward code object of its own per instance."""
    base = _K1FuseCapProgram.forward
    code = base.__code__.replace(co_name=f"forward_k1cap_{index}")
    fwd = types.FunctionType(code, base.__globals__, f"forward_k1cap_{index}", base.__defaults__, base.__closure__)
    fwd.__qualname__ = f"K1FuseCapProgram{index}.forward"
    fwd.__annotations__ = dict(base.__annotations__)
    return type(f"K1FuseCapProgram{index}", (_K1FuseCapProgram,), {"forward": fwd})


def build_k1fuse_program(
    *, vllm_config: Any, prefix: str, mtp: Any, num_steps: int, dvp: bool = False, vh: Any, statics: K1Statics,
    index: int, cap: bool = False,
) -> torch.nn.Module:
    """``support_torch_compile`` (guard-free) of instance `index`'s program class, with T as the one dynamic dim
    (`cap`: the DRAFT-CAP R_d program class, q38 L4 v6 composition)."""
    cls = support_torch_compile(dynamic_arg_dims=dict(K1_DYNAMIC_ARG_DIMS))(
        (_cap_program_class if cap else _program_class)(index)
    )
    return cls(vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp, vh=vh, statics=statics)


# ---------------------------------------------------------------- results


@dataclasses.dataclass
class K1Result(DraftAsmResult):
    """DraftAsmResult of a K1 step: the spec tail's source / rejected count come from the program; the drafts are
    the program's [p, k] (``tokens`` stays empty)."""

    k1_drafts: torch.Tensor | None = None

    def drafts(self) -> torch.Tensor:
        return self.k1_drafts[: self.num_reqs]


@dataclasses.dataclass
class _Stash:
    chunk: Any
    tokens: torch.Tensor
    drafts: torch.Tensor
    source: torch.Tensor
    rejected: torch.Tensor
    num_reqs: int
    watch: tuple
    vers: tuple


@dataclasses.dataclass
class _CheckRec:
    chunk: Any
    out: tuple
    pre_ro: torch.Tensor
    num_reqs: int
    mism: torch.Tensor | None = None
    what: list = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------- the controller


class DraftK1(DraftOne):
    """DraftOne with T1-K1FUSE (see the module docstring)."""

    def __init__(self, proposer: Any) -> None:
        super().__init__(proposer)
        self.k1_ab = (envs.VLLM_TPU_K1FUSE_AB or "").strip().lower()
        self.k1_conf = bool(envs.VLLM_TPU_K1FUSE) and self.one_conf and self.k1_ab not in _OFF
        self.warm["k1"] = set()
        self.k1_agreed = False
        self.k1_failed = False
        self.k1_local_disabled: list[str] = []
        self.k1_check_steps = int(envs.VLLM_TPU_K1FUSE_CHECK)
        self.k1_check_left = self.k1_check_steps
        self.k1_check_pending = self.k1_check_steps > 0
        self.k1_mismatches = 0
        self.k1_modules: dict[tuple, Any] = {}  # static signature -> program
        self.k1_warm_modules: set = set()
        self.k1_serves_cache: dict[tuple, frozenset] = {}
        self.k1_stash: _Stash | None = None
        self.k1_check_rec: _CheckRec | None = None
        self.k1_stats: collections.Counter = collections.Counter()
        self.k1_reasons: collections.Counter = collections.Counter()
        self.k1_static_reason: str | None = None
        self.k1_fvars: list | None = None
        self.k1_restricted = False
        self.k1_rss = [0, 0]  # bytes, programs (warm-up)
        self.k1_logged: set = set()
        if envs.VLLM_TPU_K1FUSE and not self.k1_conf:
            why = "VLLM_TPU_K1FUSE_AB=0" if self.k1_ab in _OFF else "needs VLLM_TPU_DRAFT_ASM_ONE (DH-D) on"
            logger.warning("q38 K1FUSE stays off: %s", why)
        elif self.k1_conf:
            logger.info("q38 K1FUSE armed: window %r, CHECK %d steps (agreement at DRAFT-ASM's boundaries)",
                        self.k1_ab, self.k1_check_steps)

    # -------------------------------------------------------- eligibility

    def _k1_static(self, runner: Any) -> str:
        """Why this runner can never run K1 ("" when it can); computed once (logged)."""
        if self.k1_static_reason is None:
            vfy = runner.__dict__.get("_vfy")
            reason = ""
            if not self.k1_conf:
                reason = "not_configured"
            elif getattr(runner, "_vrng", None) is not None:
                # q38 L4 v6 composition (V6COMP-BUILD), K1 x VRNG-VPS: K1's verify half is VFY-ONE's eager-uniform
                # program; VRNG steps draw in-graph (no eager draws exist) and run before K1's hook, and K1 serving
                # only VRNG's off steps / windows would confound VRNG's A/B. So K1 stays off for the run (logged).
                reason = "vrng_ingraph_uniforms"
            elif self._static():
                reason = "draft_asm_" + self._static()
            elif vfy is None or vfy.mode != 2:
                reason = "needs_vps_vfy_one_2"
            elif vfy.static(runner):
                reason = "vfy_" + vfy.static(runner)
            elif not vfy.seed_on(runner):
                reason = "vfy_seed_off"
            elif not bool(getattr(getattr(runner, "scheduler_config", None), "async_scheduling", False)):
                reason = "not_async"
            elif runner._read_offset_fold_geometry()[0] == 0:
                reason = "no_read_offset_groups"
            self.k1_static_reason = reason
            if reason and self.k1_conf:
                logger.warning("q38 K1FUSE stays off on this runner: %s", reason)
        return self.k1_static_reason

    def _k1_mode(self) -> str | None:
        """This verify step's K1 mode: "fold" (K1 serves), "check" (K1 first, today's chain serves) or None."""
        if not self.k1_conf or self.k1_failed or not self.k1_agreed:
            return None
        if self.failed or not self.agreed or not self.one_agreed or self.check_pending or self.check_left > 0:
            return None
        if self.k1_check_pending:
            return "check" if self.k1_check_left > 0 else None
        return "fold"

    def _k1_fvariants(self, runner: Any) -> list:
        """The fallback-row counts K1 warms: VFY's warm-up variants (served-only when VFY warms only the served one);
        restricted to the served count when the program total would exceed twice DH-D's (L4-E rule 4)."""
        if self.k1_fvars is None:
            fb_max, tiegap = runner._vps_fb_max, runner._vps_tiegap
            variants = vps_wiring.fallback_variants(fb_max, tiegap)
            served = vps_wiring.fallback_slots(0, tiegap, max(fb_max, tiegap))
            if getattr(envs, "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY", False):
                variants = [v for v in variants if v == served]
            shapes = self.decode_shapes()
            dhd = len({(t, lp, c) for t, _w, lp, c in shapes})
            if len(variants) * len(shapes) > 2 * dhd and served in variants:
                variants = [served]
                self.k1_restricted = True
            self.k1_fvars = list(variants)
        return self.k1_fvars

    def _k1_serves(self, p_rows: int, width: int, cap: int) -> frozenset:
        """Token buckets of the steady decode shapes with this loop bucket, request bucket and capacity."""
        key = (p_rows, width, cap)
        if key not in self.k1_serves_cache:
            self.k1_serves_cache[key] = frozenset(
                t for t, w, lp, c in self.decode_shapes() if (lp, w, c) == key
            )
        return self.k1_serves_cache[key]

    def _k1_ready(self, runner, vh, vargs, n_active, fallback_rows, cand_draws, fb_u, ro, acc, spec_args, chunk,
                  warm: bool):
        """(module, md, args, statics, key, T, serves, watch) for one K1 call, or (None, reason)."""
        p = self.p
        if self._k1_static(runner):
            return None, "static"
        chunks = getattr(p, "draft_chunks", None) or []
        if len(chunks) != 1 or chunks[0] is not chunk:
            return None, "chunks"
        why = self._reason(chunks, [None], None, True)
        if why:
            return None, "draft_" + why
        if (
            p._carry_checks_done < envs.VLLM_TPU_VERIFY_FUSED_CHECK
            or p._seed_checks_done < envs.VLLM_TPU_VERIFY_FUSED_CHECK
            or getattr(p, "_draft_top_checks_left", 0) > 0
        ):
            return None, "debug_checks"  # those checks read the seed / step 0's outputs on the host
        num_draft, qsl, n_dev, input_ids, seq_lens, positions, p_rows, cap, max_reqs = spec_args
        if not (num_draft is chunk.draft_lengths and qsl is chunk.attn_ctx.query_start_loc
                and input_ids is chunk.input_ids and seq_lens is chunk.attn_ctx.seq_lens
                and positions is chunk.position_ids):
            return None, "seed_args"
        if cap != p._draft_rows_len(chunk) or p_rows > cap:
            return None, "capacity"
        target_md = p._target_draft_attn_metadata(chunk)
        if target_md is None:
            return None, "no_metadata"
        distinct = list({id(m): m for m in target_md.values()}.values())
        md0 = (distinct[0].seq_lens, distinct[0].query_start_loc, distinct[0].request_distribution)
        if any(a is not b for m in distinct[1:]
               for a, b in zip((m.seq_lens, m.query_start_loc, m.request_distribution), md0)):
            return None, "metadata"
        if md0[0] is not seq_lens or md0[1] is not qsl:
            return None, "metadata_alias"  # draft_body feeds one seq_lens / query_start_loc to the seed and step 0
        # The loop steps' query_start_loc / request_distribution: the proposer's device constants (the very tensors
        # DraftOne._loop_md and propose hand the loop).
        runner_p = p.runner
        knr = runner_p.num_reqs_max_model_len if chunk.attn_ctx.use_max_model_len else runner_p.num_reqs_most_model_len
        n = chunk.num_reqs
        loop_qsl = p._device_constant(
            ("loop_qsl", knr, n), lambda: torch.from_numpy(np.minimum(np.arange(knr + 1, dtype=np.int32), n))
        )
        loop_rd = p._device_constant(("loop_reqdist", n), lambda: torch.tensor([n] * 3, dtype=torch.int32))
        if any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(md0, (seq_lens, loop_qsl, loop_rd))):
            return None, "shapes"
        tables: list = []
        if self.one_cap:
            # q38 L4 v6 composition (V6COMP-BUILD), K1 x DRAFT-CAP: DRAFT-CAP's R_d buffers and per-group tables, read
            # exactly as draft_cap.ready reads them (R_d = min(p, capacity) = p: p_rows <= cap above).
            n_cap = int(md0[0].shape[0])
            for m in distinct:
                t = m.block_tables
                if not isinstance(t, torch.Tensor) or t.dim() != 1 or t.shape[0] % n_cap:
                    return None, "cap_tables"
                if all(t is not u for u in tables):
                    tables.append(t)
            if not tables or len(tables) > draft_cap.TABLE_SLOTS:
                return None, "cap_tables"
            bufs, tbufs = draft_cap.cap_buffers(self, draft_cap.cap_rows(int(p_rows), n_cap), md0, tables)
        else:
            bufs = self._one_buffers(md0)
        target_hidden = p._mtp_target_hidden_states(chunk)
        groups, ro_n, ro_dtype = ro
        # vargs: the 14 verify tensors (hidden, logits indices, then md / params / draws), rates, k, synthetic mode.
        args = (*vargs[:14], vargs[14], n_active, fb_u, list(groups), ro_n, acc, num_draft, qsl, n_dev, input_ids,
                positions, seq_lens, md0[2], target_hidden, loop_qsl, loop_rd, *bufs)
        assert len(args) == len(ARG_NAMES)
        if tables:
            pad = (None,) * (draft_cap.TABLE_SLOTS - len(tables))
            args = (*args, *tables, *pad, *tbufs, *pad)
            assert len(args) == len(ARG_NAMES) + len(CAP_ARG_NAMES)
        statics = K1Statics(int(vargs[15]), bool(vargs[16]), int(fallback_rows), bool(cand_draws), ro_dtype,
                            int(p_rows), int(cap), int(max_reqs))
        T = int(input_ids.shape[0])
        if any(args[i].shape[0] != T for i in T_POS):
            return None, "token_dims"  # the program's one dynamic dim must be shared by every T input
        width = int(num_draft.shape[0])
        serves = self._k1_serves(int(p_rows), width, int(cap))
        if T not in serves:
            return None, "bucket"
        key = vfy_one.key(*args, *statics)
        if not warm and key not in self.warm["k1"]:
            return None, "cold"
        watch = (num_draft, qsl, n_dev, input_ids, positions, seq_lens, md0[2], target_hidden, loop_qsl, loop_rd,
                 *(getattr(m, f.name, None) for m in distinct for f in dataclasses.fields(m)
                   if isinstance(getattr(m, f.name, None), torch.Tensor)))
        if vfy_one.versions(watch) is None:
            return None, "no_versions"  # inference tensors: the take could not prove its inputs unwritten
        sig = tuple(
            ("T", ("tok", *e[1][1:]), e[2]) if i in T_POS else e for i, e in enumerate(key[: len(args)])
        ) + key[len(args):]
        replaced: dict[int, Any] = {}
        md = {}
        for name, m in target_md.items():
            if id(m) not in replaced:
                if tables:  # DRAFT-CAP: the R_d table buffer of this group's table too (draft_cap.ready)
                    tb = next(b for t, b in zip(tables, tbufs) if t is m.block_tables)
                    replaced[id(m)] = dataclasses.replace(
                        m, seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2], block_tables=tb
                    )
                else:
                    replaced[id(m)] = dataclasses.replace(
                        m, seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2]
                    )
            md[name] = replaced[id(m)]
        module = self._k1_module(sig, vh, statics, serves, cap=bool(tables))
        return (module, md, args, statics, key, T, serves, watch), ""

    def _k1_module(self, sig: tuple, vh: Any, statics: K1Statics, serves: frozenset, cap: bool = False):
        module = self.k1_modules.get(sig)
        if module is None:
            p = self.p
            index = len(self.k1_modules)
            try:
                with (
                    set_model_tag("eagle_head"),
                    draft_asm.set_vllm_model_wrapper_context(mesh=p.runner.mesh, vllm_config=p.draft_vllm_config),
                    draft_asm.set_current_vllm_config(p.draft_vllm_config),
                ):
                    k = p.speculative_config.num_speculative_tokens
                    module = build_k1fuse_program(
                        vllm_config=p.draft_vllm_config,
                        prefix=f"k1fuse_k{k}_{index}",
                        mtp=p.draft_model,
                        num_steps=k - 1,
                        dvp=getattr(p, "_dvp", False),
                        vh=vh,
                        statics=statics,
                        index=index,
                        **({"cap": True} if cap else {}),
                    )
            except Exception as exc:
                # Fail-stop: every rank runs the K1 program on the same steps.
                raise RuntimeError(
                    "q38 K1FUSE: could not build the verify + draft program; set VLLM_TPU_K1FUSE=0 to run without it"
                ) from exc
            bucket_filter.assign(module, lambda n, serves=serves: n in serves)
            self.k1_modules[sig] = module
        return module

    def _k1_call(self, module, md, args, T: int, serves: frozenset) -> tuple:
        p = self.p
        mtp_model = getattr(p.draft_model, "model", None)
        with (
            set_model_tag("eagle_head"),
            bucket_filter.serving(lambda n, serves=serves: n in serves),
            draft_asm.set_forward_context(
                md, p.draft_vllm_config, num_tokens=T, num_tokens_across_dp=p.runner._dp_num_tokens_across_dp(T),
            ),
            draft_asm.set_vllm_model_wrapper_context(mesh=p.runner.mesh, vllm_config=p.draft_vllm_config),
            p._draft_moe_ctx(),
            _meta_i32(mtp_model),
        ):
            if id(module) not in self.k1_warm_modules:
                # As DH-D's first call: trace and compile (every token bucket this program serves), then run
                # compiled. The repeat rewrites the same KV entries; the inputs are never written.
                module(*args)
                self.k1_warm_modules.add(id(module))
            host_phases.sub("k1_rest")
            out = tuple(module(*args))
            host_phases.sub("k1_call")
        return out

    # -------------------------------------------------------- the verify hook (tpu_runner._vfy_verify)

    def k1_verify(self, runner: Any, vchunk: Any, call: tuple, spec: Any, num_active: int):
        """In place of ``_vfy_one_compiled(*call)`` on a VFY step: K1's (tokens, stats_acc, ro_idx, ro_vals) when K1
        serves this step (its drafts are stashed for the propose), else None (today's chain runs; on a CHECK step K1
        ran first and is compared later)."""
        if self.k1_stash is not None:
            raise RuntimeError(
                "q38 K1FUSE: the previous fused step's draft result was never taken by the propose; stopping "
                "(its draft KV writes were made; fail-stop)"
            )
        if self.k1_check_rec is not None:
            self.k1_stats["check_uncompared"] += 1
            self.k1_check_rec = None
        mode = self._k1_mode()
        if mode is None:
            return None
        if spec is None or spec.fn is not draft_one._seed_cl_impl or vchunk.indices is None:
            self.k1_reasons["seed_spec" if vchunk.indices is not None else "layout"] += 1
            return None
        if len(call) != 27 + len(spec.args):
            self.k1_reasons["call_layout"] += 1
            return None
        vh = call[0]
        vargs = call[3:18]
        n_active, fallback_rows, cand, fb_u = call[18:22]
        ro = call[22:25]
        acc = call[25]
        chunks = getattr(self.p, "draft_chunks", None) or []
        chunk = chunks[0] if len(chunks) == 1 else None
        ready, why = self._k1_ready(runner, vh, (vchunk.hidden, vchunk.indices, *vargs), n_active, fallback_rows,
                                    cand, fb_u, ro, acc, spec.args, chunk, warm=False)
        if ready is None:
            self.k1_reasons[why] += 1
            return None
        module, md, args, statics, key, T, serves, watch = ready
        if mode == "check":
            self.k1_check_left -= 1
            pre = runner.mamba_slot_read_offsets.clone()
            out = self._k1_call(module, md, args, T, serves)
            self.k1_check_rec = _CheckRec(chunk, out, pre, int(num_active))
            self.k1_stats["check"] += 1
            return None
        out = self._k1_call(module, md, args, T, serves)
        tokens = out[0]
        watch = (tokens, *watch)
        self.k1_stash = _Stash(chunk, tokens, out[4], out[5], out[6], int(num_active), watch,
                               vfy_one.versions(watch))
        return out[:4]

    # -------------------------------------------------------- the propose hook (eagle3.propose, after begin)

    def k1_result(self, chunks, next_tokens_per_chunk, return_device):
        """Right after ``begin``: this step's K1Result when the verify ran K1 (the take), else None. On a K1 CHECK
        step: compare the verify-side outputs, None (the propose runs today's draft; ``result`` compares the
        draft side)."""
        rec = self.k1_check_rec
        if rec is not None:
            self._k1_check_verify_side(rec, chunks, next_tokens_per_chunk)
            return None
        stash = self.k1_stash
        if stash is None:
            return None
        self.k1_stash = None
        miss = ""
        if len(chunks) != 1 or chunks[0] is not stash.chunk:
            miss = "another draft chunk"
        elif not next_tokens_per_chunk or len(next_tokens_per_chunk) != 1 or next_tokens_per_chunk[0] is not stash.tokens:
            miss = "other verify tokens"
        elif not return_device:
            miss = "a host-list propose"
        elif stash.vers is None or vfy_one.versions(stash.watch) != stash.vers:
            miss = "an input written in place"
        if miss:
            self.k1_stats["take_miss"] += 1
            raise RuntimeError(
                "q38 K1FUSE: the propose after a fused verify + draft step saw %s; the program's draft (and its KV "
                "writes) used the inputs of the verify; stopping (fail-stop)" % miss
            )
        self.p.runner.__dict__.pop("_vfy_seed", None)  # the verify's seed stash (VFY-ONE value 2) is not needed
        self.stats["folded"] += 1
        self.stats["folded_k1"] += 1
        self.k1_stats["k1"] += 1
        if "on" not in self.k1_logged:
            self.k1_logged.add("on")
            logger.info("q38 K1FUSE on: verify + draft as one program (TP rank %d)", self._rank())
        l6 = draft_asm._l6_ab()
        if l6 is not None:
            levers = getattr(l6, "LEVERS", {})
            for lever in (draft_asm.LEVER, LEVER):
                if lever in levers:
                    l6.note(lever, "fold")
        return K1Result(
            src=stash.source,
            num_rejected=stash.rejected,
            next_tokens=stash.tokens,
            draft_lengths=stash.chunk.draft_lengths,
            num_reqs=stash.chunk.num_reqs,
            tokens=[],
            k1_drafts=stash.drafts,
        )

    # -------------------------------------------------------- CHECK

    def _k1_check_verify_side(self, rec: _CheckRec, chunks, next_tokens_per_chunk) -> None:
        runner = self.p.runner
        tokens, acc, ro_idx, ro_vals = rec.out[:4]
        mism = torch.zeros((), dtype=torch.int64)
        nt = next_tokens_per_chunk[0] if next_tokens_per_chunk and len(next_tokens_per_chunk) == 1 else None
        live_acc = runner._vps_fb_stats
        if len(chunks) != 1 or chunks[0] is not rec.chunk or nt is None:
            rec.what.append("coverage")
            mism = mism + 1
        else:
            for what, a, b in (("tokens", tokens, nt), ("stats", acc, live_acc)):
                if b is None or a.shape != b.shape or a.dtype != b.dtype:
                    rec.what.append(what)
                    mism = mism + 1
                else:
                    mism = mism + (a != b).sum().cpu().to(torch.int64)
            pre = rec.pre_ro
            pre.index_put_((ro_idx,), ro_vals)
            mism = mism + (pre != runner.mamba_slot_read_offsets).sum().cpu().to(torch.int64)
        rec.mism = mism

    def result(self, chunks, draft_tokens_per_chunk, positions_carry_per_chunk, return_device):
        out = super().result(chunks, draft_tokens_per_chunk, positions_carry_per_chunk, return_device)
        rec = self.k1_check_rec
        if rec is not None:
            self.k1_check_rec = None
            self._k1_check_draft_side(rec, out, draft_tokens_per_chunk)
        return out

    def _k1_check_draft_side(self, rec: _CheckRec, out, draft_tokens_per_chunk) -> None:
        mism = rec.mism if rec.mism is not None else torch.ones((), dtype=torch.int64)
        drafts, source, rejected = rec.out[4:7]
        base = None
        if draft_tokens_per_chunk and draft_tokens_per_chunk[0]:
            base = torch.stack(list(draft_tokens_per_chunk[0]), dim=1)
        pairs = [("drafts", drafts, base)]
        if isinstance(out, DraftAsmResult):
            pairs += [("source", source, out.src), ("rejected", rejected, out.num_rejected)]
        else:
            self.k1_stats["check_partial"] += 1  # today's chain ran the base spec tail: drafts compared only
        for what, a, b in pairs:
            if b is None or a.shape != b.shape or a.dtype != b.dtype:
                rec.what.append(what)
                mism = mism + 1
            else:
                mism = mism + (a != b).sum().cpu().to(torch.int64)
        n = int(mism.item())
        self.k1_stats["checked"] += 1
        if n:
            self.k1_mismatches += n
            if "mismatch" not in self.k1_logged:
                self.k1_logged.add("mismatch")
                logger.error(
                    "q38 K1FUSE CHECK mismatch: %d entries differ from the VFY + DH-D chain at %d requests (%s); "
                    "K1 goes off on every rank at the next agreement (TP rank %d)",
                    n, rec.num_reqs, ",".join(rec.what) or "values", self._rank(),
                )

    # -------------------------------------------------------- warm-up (draft side, warm-mode propose)

    def forward0(self, chunk, input_ids, positions, target_hidden_states):
        st = self.step
        if (
            self.k1_conf
            and st is not None
            and st.mode == "warm"
            and getattr(st, "seed", None) is not None
            and chunk is st.chunk
        ):
            self._k1_warm(chunk)
        return super().forward0(chunk, input_ids, positions, target_hidden_states)

    def _k1_warm(self, chunk) -> None:
        """One K1 call per fallback-row variant (and read-offset width) at this warm-up step's draft inputs, with
        zero verify inputs built as ``TPUModelRunner._vfy_precompile`` builds them; outputs discarded."""
        p = self.p
        runner = p.runner
        if self._k1_static(runner):
            return
        vfy = runner._vfy
        if vfy.logits_dtype is None or vfy.acc_zero is None:
            self.k1_reasons["warm_no_vfy_precompile"] += 1
            return
        from vllm_torchtpu.layers.adapter.sample import vp_sampler

        vh = runner._vps_head
        k = p.speculative_config.num_speculative_tokens
        dev = runner.device
        T = int(chunk.input_ids.shape[0])
        B = int(chunk.draft_lengths.shape[0])
        n = int(chunk.num_reqs)
        hsize = runner.model_config.get_hidden_size()
        width = vh.tp_size * vp_sampler.VP_KC if runner._vps_cand_draws else vh.full_vocab
        rates = runner.rejection_sampler.synthetic_conditional_rates
        vargs = (
            torch.zeros((T, hsize), dtype=runner._hidden_states_dtype).to(dev),
            torch.zeros(T, dtype=torch.int32).to(dev),
            torch.zeros(B, dtype=torch.int32).to(dev),
            torch.zeros(T, dtype=torch.int32).to(dev),
            torch.ones((B, 1), dtype=vfy.logits_dtype).to(dev),
            torch.zeros((B, 1), dtype=torch.int32).to(dev),
            torch.ones((B, 1), dtype=torch.float32).to(dev),
            torch.zeros((B, width), dtype=torch.float32).to(dev),
            torch.zeros(T, dtype=torch.float32).to(dev),
            torch.zeros((T, width), dtype=torch.float32).to(dev),
            torch.zeros(T, dtype=torch.int32).to(dev),
            torch.zeros(B, dtype=torch.int32).to(dev),
            torch.zeros(T, dtype=torch.int64).to(dev),
            torch.zeros(T, dtype=torch.int32).to(dev),
            rates,
            k,
            runner.rejection_sampler.synthetic_mode,
        )
        n_active = torch.tensor([n], dtype=torch.int32).to(dev)
        num_groups, widths = runner._read_offset_fold_geometry()
        spec_args, _base = self._vfy_seed_args_cl(
            chunk.draft_lengths, chunk.attn_ctx.query_start_loc, n, chunk.input_ids, chunk.attn_ctx.seq_lens,
            chunk.position_ids, p._draft_rows_len(chunk),
        )
        for rows in self._k1_fvariants(runner):
            fb_u = None
            if runner._vps_cand_draws and rows > 0:
                fb_u = torch.zeros((rows, vh.full_vocab), dtype=torch.float32).to(dev)
            for ro_w in widths:
                if n > ro_w:
                    continue
                groups = [torch.zeros(ro_w, dtype=torch.int32).to(dev) for _ in range(num_groups)]
                ro = (groups, runner._num_reqs_tensor(1), runner.mamba_slot_read_offsets.dtype)
                ready, why = self._k1_ready(runner, vh, vargs, n_active, rows, runner._vps_cand_draws, fb_u, ro,
                                            vfy.acc_zero, spec_args, chunk, warm=True)
                if ready is None:
                    self.k1_reasons["warm_skip_" + why] += 1
                    continue
                module, md, args, statics, key, T_, serves, _watch = ready
                if key in self.warm["k1"]:
                    continue
                rss0 = draft_asm._rss_bytes()
                self._k1_call(module, md, args, T_, serves)
                rss1 = draft_asm._rss_bytes()
                if rss0 is not None and rss1 is not None:
                    self.k1_rss[0] += max(0, rss1 - rss0)
                self.k1_rss[1] += 1
                self.warm["k1"].add(key)

    # -------------------------------------------------------- windows and agreement

    def _period(self) -> int:
        if self.period is None:
            super()._period()
            l6 = draft_asm._l6_ab()
            steps = int(getattr(l6, "STEPS", 0) or 0) if l6 is not None else 0
            if self.k1_ab.startswith("ab:") and l6 is not None and LEVER in getattr(l6, "LEVERS", {}) and steps > 0:
                self.period = steps
        return self.period

    def k1_disable_locally(self, why: str) -> None:
        """A rank-local reason to stop K1: this rank's wish goes off, so K1 goes off on every rank at the next
        boundary."""
        if why not in self.k1_local_disabled:
            self.k1_local_disabled.append(why)
            logger.error("q38 K1FUSE: local disable on TP rank %d (%s); every rank leaves K1 at the next agreement",
                         self._rank(), why)

    def local_vector(self) -> list:
        vec = super().local_vector()
        w = int(vec[0])
        if self.k1_conf:
            w |= BIT_K1CONF
            if not self.k1_local_disabled and self._stage_window(LEVER, self.k1_ab):
                w |= BIT_K1
            if self.k1_check_pending and self.k1_check_left == 0:
                w |= BIT_K1CLOSE
            if self.k1_mismatches:
                w |= BIT_K1BAD
        vec[0] = w
        return vec

    def apply(self, rows: list) -> None:
        super().apply(rows)
        words = [int(r[0]) for r in rows]
        confs = [bool(w & BIT_K1CONF) for w in words]
        if len(set(confs)) != 1:
            raise RuntimeError(
                "q38 K1FUSE: VLLM_TPU_K1FUSE is configured on TP ranks %s and not on the others; the ranks would run "
                "different programs; stopping (fail-stop)" % [r for r, c in enumerate(confs) if c]
            )
        if not confs[0]:
            self.k1_agreed = False
            return
        fps = {int(r[4]) for r in rows}
        if len(fps) != 1:
            raise RuntimeError(
                "q38 K1FUSE: the TP ranks' warm-key sets differ (K1 / DH-D / DRAFT-ASM fingerprints %s); stopping "
                "(fail-stop)" % sorted(fps)
            )
        why = []
        closes = [bool(w & BIT_K1CLOSE) for w in words]
        bads = [r for r, w in enumerate(words) if w & BIT_K1BAD]
        if len(set(closes)) != 1:
            why.append("ranks disagree on the K1 CHECK window %s" % closes)
        elif closes[0] and self.k1_check_pending:
            self.k1_check_pending = False
            if not bads:
                logger.info("q38 K1FUSE CHECK: %d steps, 0 mismatches over the TP group; armed (TP rank %d)",
                            self.k1_check_steps, self._rank())
        if bads:
            why.append("CHECK mismatches on TP ranks %s (%d on this rank)" % (bads, self.k1_mismatches))
        if why and not self.k1_failed:
            self.k1_failed = True
            logger.error("q38 K1FUSE agreement at serving call %d: %s; every rank runs the VFY + DH-D chain for the "
                         "rest of the run (TP rank %d)", self.serving_calls, "; ".join(why), self._rank())
        k1 = (self.k1_conf and not self.k1_failed and self.agreed and self.one_agreed
              and all(w & BIT_K1 for w in words))
        if k1 != self.k1_agreed or self.boundaries == 1:
            logger.info("q38 K1FUSE agreement at serving call %d: k1 %s (wishes %d of %d) (TP rank %d)",
                        self.serving_calls, "on" if k1 else "off", sum(1 for w in words if w & BIT_K1), len(words),
                        self._rank())
        self.k1_agreed = k1
        if self.boundaries == 1:
            shapes = self.decode_shapes()
            logger.info(
                "q38 K1FUSE warm-up: %d instances, %d programs (F %s%s; DH-D %d programs over %d loop buckets); RSS "
                "delta %.1f MiB (TP rank %d)",
                len(self.k1_modules), len(self.warm["k1"]), self.k1_fvars,
                " restricted to the served F" if self.k1_restricted else "",
                len({(t, lp, c) for t, _w, lp, c in shapes}), len(self.one_modules), self.k1_rss[0] / 2**20,
                self._rank(),
            )

    # -------------------------------------------------------- markers

    def counts(self, done: int | None = None) -> dict:
        c = super().counts(done)
        c.update(
            folded_k1=self.stats["folded_k1"],
            k1_agreed=self.k1_agreed,
            k1_failed=self.k1_failed,
            k1_check=self.k1_stats["check"],
            k1_checked=self.k1_stats["checked"],
            k1_mismatches=self.k1_mismatches,
            k1_instances=len(self.k1_modules),
            k1_programs=len(self.warm["k1"]),
            k1_reasons=dict(self.k1_reasons),
        )
        return c

    def _maybe_log(self) -> None:
        super()._maybe_log()
        n = self.serving_calls - 1
        if not self.k1_conf or n <= 0 or n % LOG_EVERY:
            return
        logger.info(
            "q38 K1FUSE: k1=%d check=%d checked=%d mismatches=%d take_miss=%d agreed=%s failed=%s reasons=%s "
            "(TP rank %d)",
            self.k1_stats["k1"], self.k1_stats["check"], self.k1_stats["checked"], self.k1_mismatches,
            self.k1_stats["take_miss"], self.k1_agreed, self.k1_failed, dict(self.k1_reasons), self._rank(),
        )
