# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT1-B: the folded async draft path with fewer compiled draft programs (H1 cards DH-B
CARRY-INTO-LOOP and DH-D DRAFT1-B, with X-D14-DRAFT-3's int32 draft metadata).

DRAFT-ASM (spec_decode/draft_asm.py, VLLM_TPU_DRAFT_ASM_FOLD) leaves a folded steady decode verify step
with four compiled draft programs: the folded seed, draft step 0's forward, the folded carry (position
carry, first draft token, hidden carry) and the folded fused loop. This module adds two stages on top of
it, each behind its own hashed flag that needs the fold:

  * stage A (VLLM_TPU_DRAFT_ASM_CARRY_LOOP): the carry becomes the fused loop's prologue
    (``_MtpFusedLoopCarry``), so the step runs three draft programs: seed, step 0, carry + loop. It ports
    the idea of IX PR14 540483b25 / f8b9dd93a (Jordan Dotzel: the fused loop proposes step 0's token
    from step 0's hidden carry; the post-step-0 position / seq_lens prep moves into the input prep).
    The position carry cannot ride the loop program: vLLM compiles one graph per token bucket by setting
    every symbol of the trace to one size (``create_concrete_args``), so a [T] input cannot sit beside
    the loop's [p] dimension. The folded seed (``_seed_cl_impl``) therefore also emits step 0's position
    carry and step 1's positions at the seed's gather indices. The prologue gathers p rows of step 0's
    untrimmed [capacity, .] outputs with one clamped gather and a row mask, which is ``[:p]`` when
    p < capacity (eagle3's fresh carry) and the identity when p == capacity (the plain carry), so both
    carry paths fold; p > capacity keeps DRAFT-ASM's own carry.
  * DH-D (VLLM_TPU_DRAFT_ASM_ONE): step 0's forward, the carry and the loop run as one program per
    (token bucket, loop bucket) (``_DraftOneProgram``; built in the DH-D section below).

draft_asm.py is unchanged: ``DraftOne`` subclasses ``DraftAsm`` and eagle3 builds it only when a stage
flag is set, so with both stage flags off every path is DRAFT-ASM's (or, with the fold off, the base's).

SAB dual-path and windows. Each stage has a window variable with VLLM_TPU_DRAFT_ASM_AB's values
(VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB, l6_ab lever "draftasmcl"; VLLM_TPU_DRAFT_ASM_ONE_AB, "draftasmone");
a stage that is off for a step runs the next lower fold (DH-D -> stage A -> DRAFT-ASM -> base). The
warm-up runs the base path for results and every configured folded variant beside it, first, so the
base path's KV writes stand. A variant whose shape key was never warmed sends the step down the next
lower one (counted as cold), so serving never compiles a folded program. A stage whose window can be
off keeps the lower variant warm; with a stage unconditionally on (window "" or "1") the lower
variant's programs are not warmed.

All-rank agreement (lead safety rule). The stages' programs carry TP collectives, like DRAFT-ASM's, so
they change only at DRAFT-ASM's agreement boundaries through its one CPU-group all_gather of five ints:
entry 0 (DRAFT-ASM's window wish, 0 / 1) carries a bit per stage (``BIT_CL``, ``BIT_ONE``) when a stage
flag is set. A stage is on until the next boundary iff the fold is agreed and every rank sets its bit;
the stages' warm keys enter DRAFT-ASM's warm-key fingerprint, so ranks with different stage flags end on
the base path, never on different programs. Per-step choices left (eligibility, a never-warmed key,
p > capacity) depend only on the scheduler output and agreed-equal warm keys. Any exception in a folded
program or a build failure raises (fail-stop).

CHECK (VLLM_TPU_DRAFT_ASM_CHECK_STEPS) runs every configured variant beside the base path on its steps
(each on its own copy of step 1's seq_lens) and compares the drafts, step 1's seq_lens and positions, the
source and the rejected count of each; mismatches count toward DRAFT-ASM's agreement as before.
"""

import contextlib
import dataclasses
import types
from typing import Any

import numpy as np
import torch
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.sequence import IntermediateTensors

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import bucket_filter
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import host_phases
from vllm_torchtpu.spec_decode import draft_asm, eagle3
from vllm_torchtpu.spec_decode.draft_asm import DraftAsm, DraftAsmResult, _key, fold_source
from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop

logger = init_logger(__name__)

# core/l6_ab.py levers of the two stage windows.
CL_LEVER = "draftasmcl"
ONE_LEVER = "draftasmone"
# Agreement entry 0: DRAFT-ASM's fold wish is bit 0; the stages' wishes ride above it.
BIT_FOLD, BIT_CL, BIT_ONE = 1, 2, 4
_ON = ("", "1", "true", "on")
_OFF = ("0", "false", "off")


# ---------------------------------------------------------------- traced bodies


def _seed_cl_impl(
    next_tokens: torch.Tensor,
    num_draft: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_reqs: torch.Tensor,
    input_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    p: int,
    num_rows: int,
    max_reqs: int,
):
    """DRAFT-ASM's folded seed (``_async_seed_inputs_asm_impl``: the base five outputs, step 1's seq_lens,
    the spec tail's bonus and rejected rows), plus step 0's position carry and step 1's positions: the
    chunk's positions at the seed's gather indices (index 1 of the outputs, [p]), clamped as eagle3's
    carry clamps them, and that carry + 1 (the base loop call's eager add)."""
    out = draft_asm._async_seed_inputs_asm_impl(
        next_tokens, num_draft, query_start_loc, num_reqs, input_ids, seq_lens, p, num_rows, max_reqs
    )
    indices = out[1].clamp_min(0)
    positions_carry = torch.index_select(positions, positions.ndim - 1, indices)
    return (*out, positions_carry, positions_carry + 1)


_seed_cl_compiled = eagle3._static_shape_args(
    torch.compile(backend="tpu", fullgraph=True, dynamic=False)(_seed_cl_impl)
)

# Indices of the folded seed's outputs.
SEED_SEQ1, SEED_BONUS, SEED_REJ, SEED_POSC, SEED_POS1 = 5, 6, 7, 8, 9


def carry_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """``x[:rows]`` when rows <= len(x), else ``x`` zero-padded to `rows` (eagle3's fresh and plain carry
    of draft step 0's [capacity, .] row outputs), as one clamped gather and a row mask: `rows` may be
    the loop program's symbolic dim, so there is no shape branch. The values are copies of x's rows (or
    zeros), bit for bit."""
    n = x.shape[0]
    idx = torch.arange(rows, device=x.device, dtype=torch.int32)
    got = torch.index_select(x, 0, idx.clamp(max=n - 1))
    keep = (idx < n).reshape(-1, *([1] * (x.dim() - 1)))
    return torch.where(keep, got, torch.zeros_like(got))


def propose_tokens(mtp: Any, hidden: torch.Tensor, dvp: bool) -> torch.Tensor:
    """The draft's greedy token per row, int32, by eagle3's carry rule (``_carry_and_token_body`` with no
    draft->target map, which the fused loop requires): get_top_tokens under VLLM_TPU_DRAFT_TOP_TOKENS,
    else compute_logits + argmax."""
    if dvp:
        return mtp.logits_processor.get_top_tokens(mtp.lm_head, hidden).to(torch.int32)
    return mtp.compute_logits(hidden).argmax(dim=-1).to(torch.int32)


def carry_loop_body(
    loop: Any,
    positions: torch.Tensor,
    last_hidden: torch.Tensor,
    hidden_states: torch.Tensor,
    seq_lens: torch.Tensor,
    tail_bonus: torch.Tensor,
    num_reqs: torch.Tensor,
    max_reqs: int,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Stage A's program: the carry prologue (p rows of step 0's outputs, the first draft token), the
    K-1 loop steps of `loop` (an ``_MtpFusedLoop``) and the spec tail's source. `positions` is step 1's
    [p]; `last_hidden` / `hidden_states` are step 0's [capacity, .] outputs. Returns
    (token0, token1 .. token_{K-1}, source)."""
    p = positions.shape[0]
    token0 = propose_tokens(loop.mtp, carry_rows(last_hidden, p), loop.dvp)
    tokens = _MtpFusedLoop.forward(
        loop, token0, positions, carry_rows(hidden_states, p), seq_lens,
        intermediate_tensors, inputs_embeds,
    )
    return (token0, *tokens, fold_source(token0, tokens, tail_bonus, num_reqs, max_reqs))


class _MtpFusedLoopCarry(_MtpFusedLoop):
    """``_MtpFusedLoop`` with the step-0 carry as its prologue and the spec tail's source as its last
    output (see `carry_loop_body`). Only `positions` is dynamic: step 0's row outputs are
    [capacity, .] on every call."""

    def __init__(
        self,
        *,
        vllm_config: Any,
        prefix: str = "",
        mtp: Any,
        num_steps: int,
        dvp: bool = False,
        max_reqs: int = 0,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp
        )
        self.max_reqs = max_reqs

    def forward(
        self,
        positions: torch.Tensor,
        last_hidden: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        tail_bonus: torch.Tensor,
        num_reqs: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return carry_loop_body(
            self, positions, last_hidden, hidden_states, seq_lens, tail_bonus, num_reqs,
            self.max_reqs, intermediate_tensors, inputs_embeds,
        )


CARRY_LOOP_DYNAMIC_ARG_DIMS = {"positions": 0}


def build_mtp_fused_loop_carry(
    *, vllm_config: Any, prefix: str, mtp: Any, num_steps: int, dvp: bool = False, max_reqs: int
) -> torch.nn.Module:
    """``support_torch_compile`` of `_MtpFusedLoopCarry` with step 1's positions as its one dynamic dim
    (the draft's other marked inputs are step 0's static [capacity, .] rows here)."""
    cls = support_torch_compile(dynamic_arg_dims=dict(CARRY_LOOP_DYNAMIC_ARG_DIMS))(
        type("MtpFusedLoopCarry", (_MtpFusedLoopCarry,), {})
    )
    return cls(
        vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp,
        max_reqs=max_reqs,
    )


# ---------------------------------------------------------------- DH-D: the draft as one program


def _unwrap(out) -> tuple[torch.Tensor, torch.Tensor]:
    """(last_hidden, next hidden) of a draft forward (Eagle3Proposer._unwrap_model_out's rule)."""
    if isinstance(out, (list, tuple)) and len(out) == 1:
        out = out[0]
    if isinstance(out, (list, tuple)):
        return out[0], out[1]
    return out, out


def dh_one_body(
    loop: Any,
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    target_hidden: torch.Tensor,
    draft_rows: torch.Tensor,
    positions_step1: torch.Tensor,
    md0: tuple,
    md1: tuple,
    bufs: tuple,
    tail_bonus: torch.Tensor,
    num_reqs: torch.Tensor,
    max_reqs: int,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """DH-D's program, and the in-graph interface for a K-step device loop (KLOOP): draft step 0, the carry
    and the K-1 loop steps of `loop` (an object with ``mtp``, ``num_steps`` and ``dvp``, e.g. an
    ``_MtpFusedLoop``), then the spec tail's source; every input and output a device tensor, no host op.

    The draft attention reads its ragged geometry from the forward context, so the caller installs a
    forward context whose draft-layer metadata holds `bufs` = (seq_lens, query_start_loc,
    request_distribution) buffers; this body switches them in place: to step 0's values `md0` (the verify
    step's own draft-layer metadata) before step 0, to the loop's `md1` (step 1's seq_lens from the seed,
    the one-query-per-request query_start_loc / request_distribution) before step 1, and the loop advances
    seq_lens by one per step (as ``_MtpFusedLoop`` does). `md0` / `md1` are never written.

    `input_ids` / `positions` / `target_hidden` are step 0's [T, .] inputs, `draft_rows` its per-request
    rows [capacity], `positions_step1` step 1's positions [p]. Returns (token0, .., token_{K-1}, source)."""
    slb, qb, rb = bufs
    slb.copy_(md0[0])
    qb.copy_(md0[1])
    rb.copy_(md0[2])
    out = type(loop.mtp).forward(
        loop.mtp, input_ids, positions, target_hidden, intermediate_tensors, inputs_embeds,
        spec_step_idx=0, draft_rows=draft_rows,
    )
    last_hidden, hidden = _unwrap(out)
    p = positions_step1.shape[0]
    token0 = propose_tokens(loop.mtp, carry_rows(last_hidden, p), loop.dvp)
    slb.copy_(md1[0])
    qb.copy_(md1[1])
    rb.copy_(md1[2])
    tokens = _MtpFusedLoop.forward(
        loop, token0, positions_step1, carry_rows(hidden, p), slb, intermediate_tensors, inputs_embeds
    )
    return (token0, *tokens, fold_source(token0, tokens, tail_bonus, num_reqs, max_reqs))


def draft_body(
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
    p: int,
    num_rows: int,
    max_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The frozen K-step interface (reports/kstep_design_v1.md section 5, R4 GLUE): one verify step's
    output in, that step's drafts out, traced inside the caller's compiled region with no host read
    (no .item() / .tolist() / .cpu(), no shape that depends on a value). It is the superset seed
    (`_seed_cl_impl`: the eagle3 input shift and seed scatter, draft rows, step 1's seq_lens / positions,
    the spec tail's bonus and rejected count) followed by `dh_one_body`. Interface note:
    scratch/kstep/DRAFT_BODY_INTERFACE.md.

    Returns (drafts [p, k] int32, source [max_reqs * (1 + k)] int32, rejected [max_reqs] int32): the
    drafts as the async spec substitution stacks them (rows past the live requests are padding), and the
    spec tail's source and rejected count for the next verify (DRAFT-ASM's fold, bitwise the spec tail's)."""
    seed = _seed_cl_impl(
        next_tokens, num_draft, query_start_loc, num_reqs, input_ids, seq_lens, positions, p, num_rows,
        max_reqs,
    )
    out = dh_one_body(
        loop,
        input_ids=seed[0],
        positions=positions,
        target_hidden=target_hidden,
        draft_rows=seed[3],
        positions_step1=seed[SEED_POS1],
        md0=(seq_lens, query_start_loc, request_distribution),
        md1=(seed[SEED_SEQ1], loop_query_start_loc, loop_request_distribution),
        bufs=bufs,
        tail_bonus=seed[SEED_BONUS],
        num_reqs=num_reqs,
        max_reqs=max_reqs,
    )
    return torch.stack(out[:-1], dim=1), out[-1], seed[SEED_REJ]


class _DraftOneProgram(_MtpFusedLoop):
    """DH-D's program at one loop bucket p (`dh_one_body`). The token dimension T of step 0's inputs is the
    one dynamic dim; p, the request capacity and every metadata shape are static, so there is one instance
    per loop bucket, each serving only its token buckets (compilation.bucket_filter)."""

    def __init__(
        self,
        *,
        vllm_config: Any,
        prefix: str = "",
        mtp: Any,
        num_steps: int,
        dvp: bool = False,
        max_reqs: int = 0,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp
        )
        self.max_reqs = max_reqs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        draft_rows: torch.Tensor,
        positions_step1: torch.Tensor,
        seq_lens0: torch.Tensor,
        query_start_loc0: torch.Tensor,
        request_distribution0: torch.Tensor,
        seq_lens1: torch.Tensor,
        query_start_loc1: torch.Tensor,
        request_distribution1: torch.Tensor,
        seq_lens_buf: torch.Tensor,
        query_start_loc_buf: torch.Tensor,
        request_distribution_buf: torch.Tensor,
        tail_bonus: torch.Tensor,
        num_reqs: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return dh_one_body(
            self,
            input_ids=input_ids,
            positions=positions,
            target_hidden=hidden_states,
            draft_rows=draft_rows,
            positions_step1=positions_step1,
            md0=(seq_lens0, query_start_loc0, request_distribution0),
            md1=(seq_lens1, query_start_loc1, request_distribution1),
            bufs=(seq_lens_buf, query_start_loc_buf, request_distribution_buf),
            tail_bonus=tail_bonus,
            num_reqs=num_reqs,
            max_reqs=self.max_reqs,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )


ONE_DYNAMIC_ARG_DIMS = ("input_ids", "positions", "hidden_states")


@contextlib.contextmanager
def _meta_i32(model: Any):
    """X-D14-DRAFT-3 (a): the draft's ragged metadata in int32 (no int64 casts, hence no X64Split pairs) in the
    traces of the DH-D program only. ``Qwen4ExpMultiTokenPredictor.meta_i32`` is read while Dynamo traces and
    baked into that program (the trace-time switch eagle3 uses for IndexShare's skip_topk), so it is set
    around every DH-D call and the draft's own programs keep tracing with it off."""
    if model is None or not hasattr(model, "meta_i32"):
        yield
        return
    old = model.meta_i32
    model.meta_i32 = True
    try:
        yield
    finally:
        model.meta_i32 = old


def _program_class(p: int) -> type:
    """A subclass of `_DraftOneProgram` whose forward has a code object of its own: vLLM's compile wrapper
    and Dynamo key compiled code on the forward's code object, so two loop buckets sharing one would share
    (with guards dropped) the first one's graph."""
    base = _DraftOneProgram.forward
    code = base.__code__.replace(co_name=f"forward_p{p}")
    fwd = types.FunctionType(code, base.__globals__, f"forward_p{p}", base.__defaults__, base.__closure__)
    fwd.__qualname__ = f"DraftOneProgramP{p}.forward"
    fwd.__annotations__ = dict(base.__annotations__)
    return type(f"DraftOneProgramP{p}", (_DraftOneProgram,), {"forward": fwd})


def build_draft_one_program(
    *, vllm_config: Any, prefix: str, mtp: Any, num_steps: int, dynamic_arg_dims: dict, dvp: bool = False,
    max_reqs: int, p: int,
) -> torch.nn.Module:
    """``support_torch_compile`` of the loop bucket `p`'s program class, with the draft's own token dims
    (the inputs of step 0)."""
    dims = {k: v for k, v in dict(dynamic_arg_dims).items() if k in ONE_DYNAMIC_ARG_DIMS}
    cls = support_torch_compile(dynamic_arg_dims=dims)(_program_class(p))
    return cls(
        vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp, max_reqs=max_reqs,
    )


# ---------------------------------------------------------------- per-step record


@dataclasses.dataclass
class _Step1(draft_asm._Step):
    """DRAFT-ASM's per-step record plus the stages' outputs."""

    raw: tuple | None = None  # draft step 0's untrimmed (last_hidden, hidden)
    cl: tuple | None = None  # stage A's (token0, .., token_{K-1}, source)
    one: tuple | None = None  # DH-D's (token0, .., token_{K-1}, source)
    used: str = ""  # the folded variant that produced a fold step's result: "one" | "cl" | ""
    one_skip: str = ""  # why DH-D did not run on this step (not a fold fault)
    checked: list = dataclasses.field(default_factory=list)


def _seq_lens_of(md: dict) -> torch.Tensor | None:
    """The one seq_lens tensor every draft layer's metadata holds, or None."""
    found = {id(m.seq_lens): m.seq_lens for m in md.values()}
    return next(iter(found.values())) if len(found) == 1 else None


# ---------------------------------------------------------------- the controller


class DraftOne(DraftAsm):
    """DRAFT-ASM's controller with stage A and DH-D (see the module docstring)."""

    def __init__(self, proposer: Any) -> None:
        super().__init__(proposer)
        self.cl_ab = (envs.VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB or "").strip().lower()
        self.one_ab = (envs.VLLM_TPU_DRAFT_ASM_ONE_AB or "").strip().lower()
        # A stage is configured when its flag is on and its window is not hard off.
        self.cl_conf = bool(envs.VLLM_TPU_DRAFT_ASM_CARRY_LOOP) and self.cl_ab not in _OFF
        self.one_conf = bool(envs.VLLM_TPU_DRAFT_ASM_ONE) and self.one_ab not in _OFF
        # q38 L4 DRAFT-CAP (spec_decode/draft_cap.py): DH-D's programs over R_d = min(p, capacity) request slots.
        self.one_cap = bool(envs.VLLM_TPU_DRAFT_ASM_CAP) and self.one_conf
        if envs.VLLM_TPU_DRAFT_ASM_CAP and not self.one_cap:
            logger.warning("q38 DRAFT-CAP: VLLM_TPU_DRAFT_ASM_CAP needs DH-D (VLLM_TPU_DRAFT_ASM_ONE, window not "
                           "hard off); ignored")
        # The superset seed (step 1's positions) whenever a stage may run.
        self.seed_cl = self.cl_conf or self.one_conf
        # DRAFT-ASM's own carry + loop stay warm unless stage A is unconditionally on.
        self.dha_needed = not (self.cl_conf and self.cl_ab in _ON)
        self.cl_agreed = False
        self.one_agreed = False
        self.warm["carry_loop"] = set()
        self.warm["one"] = set()
        self.cl_module: Any = None
        self.cl_module_warm = False
        self.cl_rss = [0, 0]  # bytes, programs (warm-up)
        self.stage_logged: set = set()
        self.stage_error_logged: set = set()
        self.one_modules: dict[int, Any] = {}  # loop bucket p -> DH-D program
        self.one_warm_modules: set = set()
        self.one_serves: dict[int, frozenset] = {}  # loop bucket p -> token buckets it compiles
        self.one_bufs: dict[tuple, tuple] = {}  # metadata shapes -> (seq_lens, qsl, rd) buffers
        self.one_rss = [0, 0]  # bytes, programs (warm-up)

    # -------------------------------------------------------- windows and agreement

    def _stage_window(self, lever: str, ab: str) -> bool:
        """A stage's wish: DRAFT-ASM's external clock, then the stage's l6_ab lever (read as
        DraftAsm.window reads lever "draftasm")."""
        if not self.window_on:
            return False
        l6 = draft_asm._l6_ab()
        registered = l6 is not None and lever in getattr(l6, "LEVERS", {})
        if not registered:
            if ab.startswith("ab:"):
                self._stage_error(lever, "window %s needs core/l6_ab.py with the lever %r; stage off"
                                  % (ab, lever))
                return False
            return ab not in _OFF
        if ab in _ON:
            return True
        if ab in _OFF:
            return False
        if getattr(l6, "STEPS", 0) <= 0:
            self._stage_error(lever, "window %s: stage windows need VLLM_TPU_L6_AB_STEPS (step "
                              "windows); stage off" % ab)
            return False
        return bool(l6.on(lever))

    def _stage_error(self, lever: str, msg: str) -> None:
        if lever not in self.stage_error_logged:
            self.stage_error_logged.add(lever)
            logger.error("q38 DRAFT1-B %s: %s", lever, msg)

    def _period(self) -> int:
        """DRAFT-ASM's boundary period; l6_ab's step window when any lever of the fold is windowed."""
        if self.period is None:
            l6 = draft_asm._l6_ab()
            steps = int(getattr(l6, "STEPS", 0) or 0) if l6 is not None else 0
            levers = getattr(l6, "LEVERS", {}) if l6 is not None else {}
            windowed = any(
                ab.startswith("ab:") and lever in levers
                for lever, ab in ((draft_asm.LEVER, self.ab), (CL_LEVER, self.cl_ab), (ONE_LEVER, self.one_ab))
            )
            self.period = steps if (windowed and steps > 0) else draft_asm.AGREE_EVERY
        return self.period

    def local_vector(self) -> list:
        vec = super().local_vector()
        wish = int(vec[0])
        if self.cl_conf and self._stage_window(CL_LEVER, self.cl_ab):
            wish |= BIT_CL
        if self.one_conf and self._stage_window(ONE_LEVER, self.one_ab):
            wish |= BIT_ONE
        vec[0] = wish
        return vec

    def apply(self, rows: list) -> None:
        """DRAFT-ASM's decision on bit 0 of every rank's wish, then each stage on its bit."""
        super().apply([[int(r[0]) & BIT_FOLD, *r[1:]] for r in rows])
        wants = [int(r[0]) for r in rows]
        cl = self.agreed and self.cl_conf and all(w & BIT_CL for w in wants)
        one = self.agreed and self.one_conf and all(w & BIT_ONE for w in wants)
        for name, new, old, bit in (("stage A", cl, self.cl_agreed, BIT_CL), ("DH-D", one, self.one_agreed, BIT_ONE)):
            conf = self.cl_conf if bit == BIT_CL else self.one_conf
            if conf and (new != old or self.boundaries == 1):
                logger.info("q38 DRAFT1-B agreement at serving call %d: %s %s (wishes %d of %d) (TP rank %d)",
                            self.serving_calls, name, "on" if new else "off",
                            sum(1 for w in wants if w & bit), len(wants), self._rank())
        self.cl_agreed, self.one_agreed = cl, one
        if self.boundaries == 1 and self.one_rss[1]:
            logger.info("q38 DRAFT1-B warm-up: %d DH-D program keys over %d loop buckets, RSS delta %.1f MiB "
                        "(%.1f MiB per key) (TP rank %d)", self.one_rss[1], len(self.one_modules),
                        self.one_rss[0] / 2**20, self.one_rss[0] / 2**20 / self.one_rss[1], self._rank())
        if self.boundaries == 1 and self.cl_rss[1]:
            logger.info("q38 DRAFT1-B warm-up: %d stage A carry-loop programs, RSS delta %.1f MiB (%.1f MiB "
                        "per program) (TP rank %d)", self.cl_rss[1], self.cl_rss[0] / 2**20,
                        self.cl_rss[0] / 2**20 / self.cl_rss[1], self._rank())

    def begin(self, chunks, next_tokens_per_chunk, scheduler_output, return_device) -> None:
        super().begin(chunks, next_tokens_per_chunk, scheduler_output, return_device)
        st = self.step
        if st is not None:
            self.step = _Step1(**{f.name: getattr(st, f.name) for f in dataclasses.fields(st)})

    def _stage_on(self, st: _Step1, stage: str) -> bool:
        """Whether this step runs a stage's program: every configured stage in warm-up and CHECK, the
        agreed ones in fold mode."""
        conf = self.cl_conf if stage == "cl" else self.one_conf
        if not conf:
            return False
        if st.mode in ("warm", "check"):
            return True
        return st.mode == "fold" and (self.cl_agreed if stage == "cl" else self.one_agreed)

    # -------------------------------------------------------- the seed

    def seed(self, chunk, nt, num_draft_full, num_reqs, ov_i):
        """DraftAsm.seed with the superset seed (`_seed_cl_impl`) when a stage is configured."""
        if not self.seed_cl:
            return super().seed(chunk, nt, num_draft_full, num_reqs, ov_i)
        st = self.step
        if st is None or ov_i or chunk is not st.chunk or nt is not st.next_tokens:
            return None
        if not draft_asm._is_tpu(nt):
            self.step = None
            return None
        p = self.p
        n_dev = p._device_constant(
            ("num_reqs", num_reqs), lambda: torch.tensor(num_reqs, dtype=torch.int64)
        )
        args = (
            nt,
            num_draft_full,
            chunk.attn_ctx.query_start_loc,
            n_dev,
            chunk.input_ids,
            chunk.attn_ctx.seq_lens,
            chunk.position_ids,
            p._loop_bucket(num_reqs),
            p._draft_rows_len(chunk),
            self.max_reqs,
        )
        key = _key(*args)
        if st.mode == "fold" and key not in self.warm["seed"]:
            self._cold(st, "seed")
            return None
        out = None
        if st.mode == "fold" and hasattr(p, "_vfy_seed_take"):
            # VFY-ONE value 2 (DH-C): the verify program made this superset seed (vfy_seed_spec below).
            out = p._vfy_seed_take(chunk, nt, _seed_cl_impl, args[1:])
        if out is None:
            out = _seed_cl_compiled(*args)
        st.seed, st.num_reqs_dev = out, n_dev
        if st.mode == "warm":
            self.warm["seed"].add(key)
        if st.mode != "fold":
            return None
        draft_input_ids, gather_indices, num_rejected_out = out[:3]
        p._seed_extras[id(chunk)] = (out[3], out[4])
        if p._seed_checks_done < envs.VLLM_TPU_VERIFY_FUSED_CHECK:
            p._check_seed_fused(
                nt, num_draft_full, chunk, num_reqs, draft_input_ids, gather_indices,
                num_rejected_out, p._seed_extras.get(id(chunk)),
            )
        draft_input_ids, gather_indices = (
            chunk.sequence_layout_plan.localize_token_tensor_and_gather_indices(
                draft_input_ids, gather_indices, num_valid_gathers=num_reqs
            )
        )
        return draft_input_ids, chunk.position_ids, gather_indices, num_rejected_out

    def _vfy_seed_args_cl(self, num_draft, qsl, num_reqs, input_ids, seq_lens, positions, capacity):
        base = self.p._vfy_seed_args(num_draft, qsl, num_reqs, input_ids, capacity, True)
        return (*base[:4], seq_lens, positions, base[4], base[5], self.max_reqs), base

    def vfy_seed_spec(self, chunk, num_reqs):
        """DraftAsm.vfy_seed_spec with the superset seed when a stage is configured."""
        if not self.seed_cl:
            return super().vfy_seed_spec(chunk, num_reqs)
        from vllm_torchtpu.runner.vfy_one import VfySeedSpec

        if self._static() or chunk.draft_lengths is None:
            return None
        if chunk.sequence_layout_plan.descriptor != draft_asm.DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR:
            return None
        args, base = self._vfy_seed_args_cl(
            chunk.draft_lengths, chunk.attn_ctx.query_start_loc, num_reqs, chunk.input_ids,
            chunk.attn_ctx.seq_lens, chunk.position_ids, self.p._draft_rows_len(chunk),
        )
        return VfySeedSpec(_seed_cl_impl, args, base)

    def vfy_seed_warm_spec(self, num_tokens, num_reqs_padded, capacity, num_reqs):
        if not self.seed_cl:
            return super().vfy_seed_warm_spec(num_tokens, num_reqs_padded, capacity, num_reqs)
        from vllm_torchtpu.runner.vfy_one import VfySeedSpec

        if self._static():
            return None
        dev = self.p.runner.device
        args, base = self._vfy_seed_args_cl(
            torch.zeros(num_reqs_padded, dtype=torch.int32).to(dev),
            torch.zeros(capacity + 1, dtype=torch.int32).to(dev),
            num_reqs,
            torch.zeros(num_tokens, dtype=torch.int32).to(dev),
            torch.zeros(capacity, dtype=torch.int32).to(dev),
            torch.zeros(num_tokens, dtype=torch.int32).to(dev),
            capacity,
        )
        return VfySeedSpec(_seed_cl_impl, args, base)

    # -------------------------------------------------------- stage A: carry into the loop

    def _carry_loop_module(self):
        if self.cl_module is None:
            p = self.p
            try:
                with (
                    set_model_tag("eagle_head"),
                    draft_asm.set_vllm_model_wrapper_context(
                        mesh=p.runner.mesh, vllm_config=p.draft_vllm_config
                    ),
                    draft_asm.set_current_vllm_config(p.draft_vllm_config),
                ):
                    k = p.speculative_config.num_speculative_tokens
                    self.cl_module = build_mtp_fused_loop_carry(
                        vllm_config=p.draft_vllm_config,
                        prefix=f"mtp_fused_loop_carry_k{k}",
                        mtp=p.draft_model,
                        num_steps=k - 1,
                        dvp=getattr(p, "_dvp", False),
                        max_reqs=self.max_reqs,
                    )
            except Exception as exc:
                # Fail-stop: every rank runs the stage A program on the same steps.
                raise RuntimeError(
                    "q38 DRAFT1-B: could not build the stage A carry-loop program; set "
                    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP=0 to run without it"
                ) from exc
        return self.cl_module

    def _loop_md(self, st: _Step1, seq_lens1: torch.Tensor):
        """The draft layers' loop-step metadata (DraftAsm.loop's): the target's step-0 entries with step 1's
        seq_lens and the hoisted loop query_start_loc / request_distribution (eagle3's device constants,
        the same tensors propose builds for the loop). None when the chunk has no target metadata."""
        p, chunk = self.p, st.chunk
        target_md = p._target_draft_attn_metadata(chunk)
        if target_md is None:
            return None
        runner = p.runner
        knr = runner.num_reqs_max_model_len if chunk.attn_ctx.use_max_model_len else runner.num_reqs_most_model_len
        n = chunk.num_reqs
        qsl = p._device_constant(
            ("loop_qsl", knr, n), lambda: torch.from_numpy(np.minimum(np.arange(knr + 1, dtype=np.int32), n))
        )
        rd = p._device_constant(("loop_reqdist", n), lambda: torch.tensor([n] * 3, dtype=torch.int32))
        replaced: dict[int, Any] = {}
        md = {}
        for name, m in target_md.items():
            if id(m) not in replaced:
                replaced[id(m)] = dataclasses.replace(
                    m, seq_lens=seq_lens1, query_start_loc=qsl, request_distribution=rd
                )
            md[name] = replaced[id(m)]
        return md

    def _cl_args(self, st: _Step1, last_hidden, hidden):
        seed = st.seed
        return (seed[SEED_POS1], last_hidden, hidden, seed[SEED_SEQ1], seed[SEED_BONUS], st.num_reqs_dev)

    def _cl_ready(self, st: _Step1) -> tuple | None:
        """Stage A's inputs for this step, or None when it cannot run here (no raw step-0 outputs, the
        loop bucket above the request capacity, or, in fold mode, a never-warmed key)."""
        if st.seed is None or st.raw is None:
            return None
        last_hidden, hidden = st.raw
        p_rows = st.seed[SEED_POS1].shape[0]
        if p_rows > last_hidden.shape[0] or p_rows > hidden.shape[0]:
            return None
        args = self._cl_args(st, last_hidden, hidden)
        # A vLLM-compiled program keeps the static shapes of its first trace: off warm-up it runs only on
        # keys the warm-up recorded (CHECK included).
        if st.mode != "warm" and _key(*args) not in self.warm["carry_loop"]:
            return None
        return args

    def keep_rows(self, chunk, last_hidden, hidden, rows) -> bool:
        """eagle3's hook before step 0's [:rows] views: records step 0's untrimmed outputs, and returns
        True (the views are skipped) when this fold step will take stage A or has taken DH-D."""
        st = self.step
        if st is None or chunk is not st.chunk or not isinstance(st, _Step1):
            return False
        if st.mode != "fold":
            st.raw = (last_hidden, hidden)
            return False
        if st.used == "one":
            return True
        st.raw = (last_hidden, hidden)
        p = self.p
        return (
            self._stage_on(st, "cl")
            and self._cl_ready(st) is not None
            and p._carry_checks_done >= envs.VLLM_TPU_VERIFY_FUSED_CHECK
            and getattr(p, "_draft_top_checks_left", 0) <= 0
        )

    def _run_carry_loop(self, st: _Step1, args: tuple) -> tuple | None:
        """Stage A's program on `args`; its outputs, or None when the chunk has no target metadata. Off
        fold mode it runs on a copy of step 1's seq_lens (the program advances it in place, and DRAFT-ASM's
        loop and CHECK read the seed's)."""
        seq_lens1 = args[3] if st.mode == "fold" else args[3].clone()
        md = self._loop_md(st, seq_lens1)
        module = self._carry_loop_module()
        if md is None:
            return None
        args = (*args[:3], seq_lens1, *args[4:])
        p = self.p
        num_tokens = args[0].shape[0]
        key = _key(*args)
        new_key = st.mode == "warm" and key not in self.warm["carry_loop"]
        rss0 = draft_asm._rss_bytes() if new_key else None
        with (
            set_model_tag("eagle_head"),
            draft_asm.set_forward_context(
                md,
                p.draft_vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=p.runner._dp_num_tokens_across_dp(num_tokens),
            ),
            draft_asm.set_vllm_model_wrapper_context(mesh=p.runner.mesh, vllm_config=p.draft_vllm_config),
        ):
            if not self.cl_module_warm:
                # As the fused loops' first call: trace and compile, put seq_lens back, run compiled.
                keep = seq_lens1.clone()
                module(*args)
                seq_lens1.copy_(keep)
                self.cl_module_warm = True
            host_phases.sub("dc_rest")
            out = tuple(module(*args))
            host_phases.sub("dl_fold")
        if rss0 is not None:
            rss1 = draft_asm._rss_bytes()
            if rss1 is not None:
                self.cl_rss[0] += max(0, rss1 - rss0)
                self.cl_rss[1] += 1
        if st.mode == "warm":
            self.warm["carry_loop"].add(key)
        return out

    # -------------------------------------------------------- DH-D: the draft as one program

    def _one_serves(self, p_rows: int) -> frozenset:
        """The token buckets loop bucket `p_rows`'s program compiles: those of the steady decode shapes with
        that loop bucket (DraftAsm.decode_shapes)."""
        if p_rows not in self.one_serves:
            self.one_serves[p_rows] = frozenset(t for t, _w, lp, _c in self.decode_shapes() if lp == p_rows)
        return self.one_serves[p_rows]

    def _one_module(self, p_rows: int):
        module = self.one_modules.get(p_rows)
        if module is None:
            p = self.p
            try:
                with (
                    set_model_tag("eagle_head"),
                    draft_asm.set_vllm_model_wrapper_context(mesh=p.runner.mesh, vllm_config=p.draft_vllm_config),
                    draft_asm.set_current_vllm_config(p.draft_vllm_config),
                ):
                    k = p.speculative_config.num_speculative_tokens
                    module = build_draft_one_program(
                        vllm_config=p.draft_vllm_config,
                        prefix=f"draft_one_k{k}_p{p_rows}",
                        mtp=p.draft_model,
                        num_steps=k - 1,
                        dynamic_arg_dims=getattr(p.draft_model, "_dynamic_arg_dims", {}),
                        dvp=getattr(p, "_dvp", False),
                        max_reqs=self.max_reqs,
                        p=p_rows,
                    )
            except Exception as exc:
                # Fail-stop: every rank runs the DH-D program on the same steps.
                raise RuntimeError(
                    "q38 DRAFT1-B: could not build the DH-D draft program; set VLLM_TPU_DRAFT_ASM_ONE=0 to "
                    "run without it"
                ) from exc
            serves = self._one_serves(p_rows)
            bucket_filter.assign(module, lambda n, serves=serves: n in serves)
            self.one_modules[p_rows] = module
        return module

    def _one_buffers(self, md0: tuple) -> tuple:
        """Persistent (seq_lens, query_start_loc, request_distribution) buffers of these shapes, which the DH-D
        program's forward context holds and the program fills in place (allocated once, at warm-up)."""
        key = tuple((tuple(t.shape), t.dtype) for t in md0)
        bufs = self.one_bufs.get(key)
        if bufs is None:
            bufs = tuple(torch.zeros_like(t) for t in md0)
            self.one_bufs[key] = bufs
        return bufs

    def _one_ready(self, st: _Step1, input_ids, positions, target_hidden):
        """DH-D's (module, metadata, args) for this step, or None (a reason counted) when it cannot run: no
        target metadata, metadata the program cannot switch (several seq_lens tensors, step-0 and loop shapes
        that differ), a token bucket its loop bucket's program does not compile, or, off warm-up, a key the
        warm-up never recorded."""
        p, chunk, seed = self.p, st.chunk, st.seed
        target_md = p._target_draft_attn_metadata(chunk)
        if target_md is None:
            return None, "one_no_metadata"
        if st.mode == "fold" and (
            p._carry_checks_done < envs.VLLM_TPU_VERIFY_FUSED_CHECK
            or getattr(p, "_draft_top_checks_left", 0) > 0
        ):
            return None, "one_debug_checks"  # those checks read draft step 0's real outputs
        distinct = list({id(m): m for m in target_md.values()}.values())
        md0 = (distinct[0].seq_lens, distinct[0].query_start_loc, distinct[0].request_distribution)
        if any(a is not b for m in distinct[1:]
               for a, b in zip((m.seq_lens, m.query_start_loc, m.request_distribution), md0)):
            return None, "one_metadata"  # one buffer set cannot stand for several metadata tensors
        loop_md = self._loop_md(st, seed[SEED_SEQ1])
        any_loop = next(iter(loop_md.values()))
        md1 = (seed[SEED_SEQ1], any_loop.query_start_loc, any_loop.request_distribution)
        if any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(md0, md1)):
            return None, "one_shapes"
        p_rows = seed[SEED_POS1].shape[0]
        if input_ids.shape[0] not in self._one_serves(p_rows):
            return None, "one_bucket"
        if self.one_cap:
            # q38 L4 DRAFT-CAP: the program at R_d = min(p, capacity) slots, its prefix buffers and per-group tables.
            from vllm_torchtpu.spec_decode import draft_cap

            return draft_cap.ready(self, st, target_md, distinct, md0, md1, p_rows, input_ids, positions,
                                   target_hidden)
        bufs = self._one_buffers(md0)
        args = (input_ids, positions, target_hidden, seed[3], seed[SEED_POS1], *md0, *md1, *bufs,
                seed[SEED_BONUS], st.num_reqs_dev)
        if st.mode != "warm" and _key(*args) not in self.warm["one"]:
            return None, "one_cold"
        replaced: dict[int, Any] = {}
        md = {}
        for name, m in target_md.items():
            if id(m) not in replaced:
                replaced[id(m)] = dataclasses.replace(
                    m, seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2]
                )
            md[name] = replaced[id(m)]
        return (self._one_module(p_rows), md, args), ""

    def forward0(self, chunk, input_ids, positions, target_hidden_states):
        """eagle3's hook in place of draft step 0's forward. Fold with DH-D agreed and ready: the one program;
        returns placeholder step-0 outputs (the carry and loop hooks then pass DH-D's tokens through).
        Warm-up / CHECK: the program beside the base path, first; returns None. None also otherwise."""
        st = self.step
        if not (isinstance(st, _Step1) and st.seed is not None and chunk is st.chunk):
            return None
        if not self._stage_on(st, "one"):
            return None
        ready, why = self._one_ready(st, input_ids, positions, target_hidden_states)
        if ready is None:
            st.one_skip = why
            self.reasons[("cold_one" if why == "one_cold" else why) if st.mode == "fold"
                         else st.mode + "_skip_" + why] += 1
            return None
        module, md, args = ready
        p = self.p
        key = _key(*args)
        new_key = st.mode == "warm" and key not in self.warm["one"]
        rss0 = draft_asm._rss_bytes() if new_key else None
        serves = self._one_serves(args[4].shape[0])
        mtp_model = getattr(p.draft_model, "model", None)
        num_tokens = args[0].shape[0]
        with (
            set_model_tag("eagle_head"),
            bucket_filter.serving(lambda n, serves=serves: n in serves),
            draft_asm.set_forward_context(
                md, p.draft_vllm_config, num_tokens=num_tokens,
                num_tokens_across_dp=p.runner._dp_num_tokens_across_dp(num_tokens),
            ),
            draft_asm.set_vllm_model_wrapper_context(mesh=p.runner.mesh, vllm_config=p.draft_vllm_config),
            p._draft_moe_ctx(),
            _meta_i32(mtp_model),
        ):
            if id(module) not in self.one_warm_modules:
                # As the fused loops' first call: trace and compile (every token bucket this program serves),
                # then run compiled. The repeat rewrites the same KV entries; md0 / md1 are never written.
                module(*args)
                self.one_warm_modules.add(id(module))
            host_phases.sub("df0_rest")
            out = tuple(module(*args))
            host_phases.sub("df_fold")
        if rss0 is not None:
            rss1 = draft_asm._rss_bytes()
            if rss1 is not None:
                self.one_rss[0] += max(0, rss1 - rss0)
                self.one_rss[1] += 1
        if st.mode == "warm":
            self.warm["one"].add(key)
        st.one = out
        if st.mode != "fold":
            return None
        st.used = "one"
        # Placeholders with >= p rows (T >= p): propose's [:rows] views are skipped (keep_rows) and the loop
        # pads are no-ops; nothing reads their values.
        st.raw = (target_hidden_states, target_hidden_states)
        return target_hidden_states, target_hidden_states

    def carry(self, chunk, positions, last_token_indices, untrimmed, hidden, last_hidden):
        """In place of eagle3's carry-fused branch. Fold with DH-D done: DH-D's first token and pass-through
        carries. Fold with stage A: the carry-loop program, then its first token and pass-through carries
        (the loop hook appends the rest). Otherwise DRAFT-ASM's carry (which also runs beside the base one
        in warm-up and CHECK when the lower variant is kept warm); stage A runs first there."""
        st = self.step
        if st is None or st.seed is None or chunk is not st.chunk or not isinstance(st, _Step1):
            return super().carry(chunk, positions, last_token_indices, untrimmed, hidden, last_hidden)
        if st.mode == "fold" and st.used == "one":
            return self._pass_through(st, st.one)
        if self._stage_on(st, "cl"):
            args = self._cl_ready(st)
            if args is not None:
                out = self._run_carry_loop(st, args)
                if out is not None:
                    st.cl = out
                    if st.mode == "fold":
                        st.used = "cl"
                        p = self.p
                        if p._carry_checks_done < envs.VLLM_TPU_VERIFY_FUSED_CHECK:
                            p._check_carry_fused(
                                positions, last_token_indices, last_hidden, st.seed[SEED_POSC], out[0]
                            )
                        return self._pass_through(st, out)
            elif st.mode == "fold":
                self.reasons["cold_carry_loop"] += 1
            else:
                self.reasons["check_cold_carry_loop" if st.mode == "check" else "warm_skip_carry_loop"] += 1
        if st.mode != "fold" and not self.dha_needed:
            return None
        host_phases.sub("dc_rest")
        out = super().carry(chunk, positions, last_token_indices, untrimmed, hidden, last_hidden)
        host_phases.sub("dc_fold")
        return out

    def _pass_through(self, st: _Step1, out: tuple):
        """(positions_carry, token0, hidden_carry, last_hidden_carry) for propose after a stage folded the
        carry: the seed's position carry and the stage's first token; the hidden carries are step 0's
        untrimmed outputs, which propose only pads (a no-op at capacity >= p) and hands to the loop hook."""
        last_hidden, hidden = st.raw
        if getattr(self.p, "_draft_top_checks_left", 0) > 0:
            # VLLM_TPU_DRAFT_TOP_TOKENS_CHECK (debug) recomputes the token from last_hidden_carry: [p] rows.
            last_hidden = eagle3.Eagle3Proposer._fit_draft_rows_output(last_hidden, out[0].shape[0])
        return st.seed[SEED_POSC], out[0], hidden, last_hidden

    def loop(self, chunks, draft_tokens_per_chunk, loop_hidden, loop_qsl, loop_reqdist) -> bool:
        """Fold with a stage's program done: append its loop tokens; True (no loop runs). Otherwise
        DraftAsm.loop (DRAFT-ASM's folded loop, also run beside the base loop in warm-up / CHECK when the
        lower variant is kept warm)."""
        st = self.step
        if isinstance(st, _Step1) and st.mode == "fold" and st.used:
            out = st.one if st.used == "one" else st.cl
            draft_tokens_per_chunk[0].extend(out[1:-1])
            return True
        if isinstance(st, _Step1) and st.mode != "fold" and not self.dha_needed:
            return False
        return super().loop(chunks, draft_tokens_per_chunk, loop_hidden, loop_qsl, loop_reqdist)

    def result(self, chunks, draft_tokens_per_chunk, positions_carry_per_chunk, return_device):
        st = self.step
        if not (isinstance(st, _Step1) and st.mode == "fold" and st.used):
            return super().result(chunks, draft_tokens_per_chunk, positions_carry_per_chunk, return_device)
        if not return_device:
            self.stats["partial"] += 1
            self.reasons["partial_result"] += 1
            self.step = None
            return None
        out = st.one if st.used == "one" else st.cl
        self.stats["folded"] += 1
        self.stats["folded_" + st.used] += 1
        lever = ONE_LEVER if st.used == "one" else CL_LEVER
        if not self.engaged_logged:
            self.engaged_logged = True
            logger.info("q38 DRAFT-ASM fold: on (TP rank %d)", self._rank())
        if st.used not in self.stage_logged:
            self.stage_logged.add(st.used)
            logger.info("q38 DRAFT1-B %s: on (TP rank %d)", "DH-D one program" if st.used == "one"
                        else "stage A carry-loop", self._rank())
        l6 = draft_asm._l6_ab()
        if l6 is not None:
            levers = getattr(l6, "LEVERS", {})
            if draft_asm.LEVER in levers:
                l6.note(draft_asm.LEVER, "fold")
            if lever in levers:
                l6.note(lever, "fold")
        return DraftAsmResult(
            src=out[-1],
            num_rejected=st.seed[SEED_REJ],
            next_tokens=st.next_tokens,
            draft_lengths=st.chunk.draft_lengths,
            num_reqs=st.num_reqs,
            tokens=list(draft_tokens_per_chunk[0]),
        )

    # -------------------------------------------------------- CHECK

    def _check_draft_side(self, st, draft_tokens_per_chunk, positions_carry_per_chunk) -> None:
        if not isinstance(st, _Step1) or not self.seed_cl:
            return super()._check_draft_side(st, draft_tokens_per_chunk, positions_carry_per_chunk)
        self.stats["checked"] += 1
        if st.seed is None:
            self._mismatch(st, "coverage (the folded seed did not run)")
            return
        base = draft_tokens_per_chunk[0]
        pos_base = positions_carry_per_chunk[0].detach().cpu() + 1
        if not draft_asm._same(st.seed[SEED_POS1], pos_base):
            self._mismatch(st, "positions_step1")
        rejected = self.p._seed_extras.get(id(st.chunk))
        # DRAFT-ASM's folded loop advanced the seed's seq_lens in place (its copy is seq1_host); the stages
        # ran on copies of it.
        seq1 = st.seq1_host if st.seq1_host is not None else st.seed[SEED_SEQ1]
        if rejected is None:
            self._mismatch(st, "seq_lens_step1 (no base rejected count)")
        else:
            want = st.chunk.attn_ctx.seq_lens.detach().cpu() + 1 - rejected[1].detach().cpu()
            if not draft_asm._same(seq1, want):
                self._mismatch(st, "seq_lens_step1")
        variants = []
        if self.dha_needed:
            variants.append(("DRAFT-ASM", None if st.carry is None or st.loop is None
                             else [st.carry[1], *st.loop[:-1]]))
        if self.cl_conf:
            variants.append(("stage A", None if st.cl is None else list(st.cl[:-1])))
        if self.one_conf:
            variants.append(("DH-D", None if st.one is None else list(st.one[:-1])))
        for name, folded in variants:
            if folded is None and name == "stage A" and self._cl_skipped(st):
                continue  # a key the warm-up never recorded: not run, not compared
            if folded is None and name == "DH-D" and st.one_skip:
                continue  # a token bucket its program does not serve, or a never-warmed key: not compared
            if folded is None:
                self._mismatch(st, "coverage (%s did not run)" % name)
            elif len(base) != len(folded) or not all(draft_asm._same(a, b) for a, b in zip(base, folded)):
                self._mismatch(st, "%s drafts" % name)
            st.checked.append(name)

    def _cl_skipped(self, st: _Step1) -> bool:
        """Stage A could not run on this CHECK step for a reason that is not a fold fault: no raw step-0
        outputs (a non-default layout), p above the capacity, or a key the warm-up never recorded."""
        if st.raw is None or st.seed is None:
            return True
        last_hidden, hidden = st.raw
        if st.seed[SEED_POS1].shape[0] > min(last_hidden.shape[0], hidden.shape[0]):
            return True
        return _key(*self._cl_args(st, last_hidden, hidden)) not in self.warm["carry_loop"]

    def check_tail(self, src: torch.Tensor, num_rejected: torch.Tensor) -> None:
        st = self.step
        if not isinstance(st, _Step1) or not self.seed_cl:
            return super().check_tail(src, num_rejected)
        if st.mode != "check" or st.seed is None:
            return
        if not draft_asm._same(st.seed[SEED_REJ], num_rejected):
            self._mismatch(st, "rejected count")
        for name, out in (("DRAFT-ASM", st.loop if self.dha_needed else None), ("stage A", st.cl),
                          ("DH-D", st.one)):
            if out is not None and not draft_asm._same(out[-1], src):
                self._mismatch(st, "%s source" % name)

    # -------------------------------------------------------- precompile

    def precompile(self) -> None:
        """DraftAsm.precompile with the superset seed when a stage is configured; DRAFT-ASM's folded carries
        only while its own carry + loop are kept warm. The stages' loop programs warm in the spec-decode
        warm-up, as DRAFT-ASM's loop does."""
        if not self.seed_cl:
            return super().precompile()
        if self._static():
            logger.info("q38 DRAFT-ASM precompile skipped: %s", self._static())
            return
        p, runner = self.p, self.p.runner
        k = p.speculative_config.num_speculative_tokens
        hidden_size = p._draft_hidden_size()
        dev = runner.device
        n_seed = n_carry = 0
        rss0 = draft_asm._rss_bytes()
        with set_model_tag("eagle_head"), runner._precompile_timed("drafter DRAFT1-B fold"):
            for t, width, loop_p, cap in self.decode_shapes():
                nt = torch.full((width, k + 1), draft_asm.INVALID_TOKEN_ID, dtype=torch.int32).to(dev)
                nd = torch.zeros(width, dtype=torch.int32).to(dev)
                qsl = torch.zeros(cap + 1, dtype=torch.int32).to(dev)
                n_dev = torch.tensor(1, dtype=torch.int64).to(dev)
                ids = torch.zeros(t, dtype=torch.int32).to(dev)
                seq_lens = torch.zeros(cap, dtype=torch.int32).to(dev)
                positions = torch.zeros(t, dtype=torch.int32).to(dev)
                args = (nt, nd, qsl, n_dev, ids, seq_lens, positions, loop_p, cap, self.max_reqs)
                key = _key(*args)
                if key not in self.warm["seed"]:
                    draft_asm.synchronize_tensors(_seed_cl_compiled(*args))
                    self.warm["seed"].add(key)
                    n_seed += 1
                if not self.dha_needed or loop_p > t:
                    continue
                idx = torch.zeros(loop_p, dtype=torch.int64).to(dev)
                if loop_p < cap:
                    full = torch.zeros((cap, hidden_size), dtype=runner._hidden_states_dtype).to(dev)
                    cargs = (positions, idx, full, torch.zeros_like(full))
                    fn = draft_asm._carry_token_fresh_asm_compiled
                else:
                    rows = torch.zeros((loop_p, hidden_size), dtype=runner._hidden_states_dtype).to(dev)
                    cargs = (positions, idx, rows)
                    fn = draft_asm._carry_and_token_asm_compiled
                ckey = _key(*cargs)
                if ckey not in self.warm["carry"]:
                    draft_asm.synchronize_tensors(fn(p, *cargs))
                    self.warm["carry"].add(ckey)
                    n_carry += 1
        rss1 = draft_asm._rss_bytes()
        self.precompile_rss = (rss0, rss1, n_seed + n_carry)
        delta = "n/a" if rss0 is None or rss1 is None else "%.1f MiB (%.1f MiB per program)" % (
            (rss1 - rss0) / 2**20, (rss1 - rss0) / 2**20 / max(1, n_seed + n_carry))
        logger.info(
            "q38 DRAFT1-B precompile: %d superset seed + %d DRAFT-ASM carry programs (steady decode shapes; "
            "stage A %s, DH-D %s); RSS delta %s; the loop programs warm in the spec-decode warm-up",
            n_seed, n_carry, "on" if self.cl_conf else "off", "on" if self.one_conf else "off", delta,
        )

    # -------------------------------------------------------- markers

    def counts(self, done: int | None = None) -> dict:
        c = super().counts(done)
        c.update(
            folded_cl=self.stats["folded_cl"],
            folded_one=self.stats["folded_one"],
            cl_agreed=self.cl_agreed,
            one_agreed=self.one_agreed,
        )
        return c

    def _maybe_log(self) -> None:
        super()._maybe_log()
        n = self.serving_calls - 1
        if n <= 0 or n % draft_asm.LOG_EVERY:
            return
        logger.info(
            "q38 DRAFT1-B: %d folded = %d DH-D + %d stage A + %d DRAFT-ASM; stage A %s, DH-D %s; "
            "cold_carry_loop %d, cold_one %d; TP rank %d",
            self.stats["folded"], self.stats["folded_one"], self.stats["folded_cl"],
            self.stats["folded"] - self.stats["folded_one"] - self.stats["folded_cl"],
            "on" if self.cl_agreed else "off", "on" if self.one_agreed else "off",
            self.reasons["cold_carry_loop"], self.reasons["cold_one"], self._rank(),
        )
