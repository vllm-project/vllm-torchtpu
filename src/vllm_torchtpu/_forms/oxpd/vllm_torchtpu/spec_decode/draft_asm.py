# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT-ASM (VLLM_TPU_DRAFT_ASM_FOLD): the async draft path with no eager op and no
spec-tail program (H1 cards DH-A NODEFER-LOOP and T-4 SPEC-TAIL-FOLD).

On a steady async decode verify step with every served draft lever on (fused seed with extras,
fresh carry, fused draft loop, compiled spec tail with the padded rejected count), the base path
runs three eager groups that torch-tpu's DEFER_AND_FUSE materializes as programs of their own
(T98g census, M: ``tt_jit_sub``, ``tt_jit_add`` and ``tt_jit_cat`` once per step each), blocks
on two deferred arguments at the fused loop call, and calls one more compiled program after the
draft chain (``_spec_tail``):

  * step 1's ``seq_lens = attn seq_lens + 1 - rejected`` (eagle3 ``_loop_step_attn_inputs``);
  * step 1's ``positions = carry positions + 1`` (the fused loop call in ``propose``);
  * ``torch.stack`` of the K draft tokens and ``torch.cat`` of the chunk (end of ``propose``);
  * ``_spec_tail``: the stride-(1+K) ``[bonus, drafts]`` source and the rejected count.

The fold computes the same integers inside programs that already run:

  1. the fused seed program also returns step 1's seq_lens and, from the verify output and the
     draft lengths it already reads, the spec tail's bonus and rejected count at max_num_reqs
     rows (rows past the live requests are zeros, as the spec tail pads them);
  2. the fresh carry program also returns step 1's positions;
  3. a second fused draft-loop program (``_MtpFusedLoopAsm``) takes them, runs the same loop
     and also returns the source, stacked and masked to max_num_reqs * (1 + K) entries.

Every new input and output of the loop program has a constant shape: vLLM's
``support_torch_compile`` skips dynamo guards after its first compile, so nothing in the loop
may depend on a shape other than the dims it marks dynamic.

SAB dual-path. The base programs stay as they are and warm as they do today. The folded seed
and carry are precompiled over the decode-only shapes (``DraftAsm.precompile``); the folded loop
is warmed in the real spec-decode warm-up by a call beside the base loop on every fold-eligible
warm-up step (its draft forward cannot be zeros-precompiled, as the base loop's cannot). Every
folded call's shape key is recorded when it returns; after warm-up a key that was never warmed
sends that step down the base path (counted as cold), so serving never compiles a folded
program. A window clock picks the path per step: ``DraftAsm.window_on`` (an external lockstep
clock may clear it) and, where core/l6_ab.py registers the lever ``draftasm`` to
VLLM_TPU_DRAFT_ASM_AB, that lever's step-window state. Both paths give the same integers.

All-rank agreement (lead safety rule). The two paths' programs carry different TP collectives, so
no rank may change path on its own. The path changes only at agreement boundaries, which are
ENGINE STEPS of the runner's one agreement clock (runner/l4_agree.py, q38 L4 AGREE): the first
serving engine step and every AGREE_EVERY engine steps after (or every VLLM_TPU_L6_AB_STEPS with
l6_ab step windows, plus every l6_ab step-window roll), never a propose call, so the K inner propose
calls of a K-step burst all run on the agreement of the burst's engine step. At a boundary every
rank contributes (window wish, sticky local disable, CHECK closed, CHECK mismatches, warm-key
fingerprint) as its 5 entries of the runner's one all_gather over the TP group's CPU group (every
registered lever's entries ride the same gather); every rank then computes the same decision: fold
until the next boundary iff every rank wishes it, no rank is disabled and every warm-key set is
equal; any disable, mismatch or unequal fingerprint keeps every rank on the base path for the rest
of the run. Between boundaries nothing rank-local (a window read, an l6_ab desync, a CHECK mismatch)
changes the path; the per-step choices left (eligibility, a never-warmed shape) depend only on the
scheduler output and the agreed-equal warm keys. A gather that fails raises (fail-stop), and so does
a folded-loop build failure or any exception in a folded call. Cost: 5 int64 per rank in the one
CPU all_gather per boundary, never per step.

CHECK (VLLM_TPU_DRAFT_ASM_CHECK_STEPS=N): the first N fold-eligible serving steps run the base
path for the step's results and the folded programs beside it, and compare the drafts, step 1's
seq_lens and positions, the source and the rejected count (one D2H each). A mismatch logs one
ERROR at once and only counts; the next boundary after the window sums the counts over the TP
group (the same gather) and, if any rank saw one, every rank keeps the base path for the run.
"""

import collections
import dataclasses
import inspect
from typing import Any

import numpy as np
import torch
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors

from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.runner import l4_agree
from vllm_torchtpu.spec_decode import eagle3
from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop
from vllm_torchtpu.spec_decode.utils import fit_rows
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

INVALID_TOKEN_ID = eagle3.INVALID_TOKEN_ID
# core/l6_ab.py lever name; its env var is VLLM_TPU_DRAFT_ASM_AB.
LEVER = "draftasm"
LOG_EVERY = 5000
# Serving engine steps between all-rank agreements (the window boundary when core/l6_ab.py does not
# set step windows; with them, its VLLM_TPU_L6_AB_STEPS). Counted by the runner's agreement clock
# (runner/l4_agree.py), never by propose calls.
AGREE_EVERY = 200


# ---------------------------------------------------------------- traced bodies


def tail_bonus_rejected(
    next_tokens: torch.Tensor,
    num_draft: torch.Tensor,
    num_reqs: torch.Tensor,
    max_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The spec tail's per-request bonus and rejected count, at `max_reqs` rows.

    `next_tokens` is the verify output ``[width, K+1]``, `num_draft` the chunk's draft lengths,
    `num_reqs` a 0-d tensor. Rows below `num_reqs` carry what ``assemble_spec_next_tokens``
    (bonus = last non-padding column) and ``compute_num_rejected`` give for them; every other
    row is 0, which is what ``_spec_tail`` pads them with (VLLM_TPU_SPEC_TAIL_REJ_PAD)."""
    width = next_tokens.shape[0]
    num_valid = (next_tokens != INVALID_TOKEN_ID).sum(dim=1)
    bonus = (
        next_tokens.gather(1, (num_valid.clamp(min=1) - 1).unsqueeze(1))
        .squeeze(1)
        .to(torch.int32)
    )
    rejected = (fit_rows(num_draft, width) - num_valid + 1).clamp(min=0).to(torch.int32)
    live = torch.arange(width, device=next_tokens.device) < num_reqs
    bonus = torch.where(live, bonus, torch.zeros_like(bonus))
    rejected = torch.where(live, rejected, torch.zeros_like(rejected))
    return fit_rows(bonus, max_reqs), fit_rows(rejected, max_reqs)


def fold_source(
    token0: torch.Tensor,
    tokens: tuple[torch.Tensor, ...],
    tail_bonus: torch.Tensor,
    num_reqs: torch.Tensor,
    max_reqs: int,
) -> torch.Tensor:
    """The spec tail's source ``[max_reqs * (1 + K)]``: per row ``[bonus, token0, tokens...]``
    for rows below `num_reqs`, zeros after (``_spec_tail`` pads the exact-count source with
    zeros to the same bound). `token0` and `tokens` are ``[p]``; p may be symbolic here (the
    loop's marked dim), so the rows are fitted to `max_reqs` without a shape branch."""
    drafts = torch.stack((token0, *tokens), dim=1)
    drafts = torch.cat(
        (drafts, drafts.new_zeros((max_reqs, drafts.shape[1]))), dim=0
    )[:max_reqs]
    src = torch.cat((tail_bonus.unsqueeze(1).to(drafts.dtype), drafts), dim=1)
    live = torch.arange(max_reqs, device=src.device) < num_reqs
    src = torch.where(live.unsqueeze(1), src, torch.zeros_like(src))
    return src.reshape(-1)


def _async_seed_inputs_asm_impl(
    next_tokens: torch.Tensor,
    num_draft: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_reqs: torch.Tensor,
    input_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    p: int,
    num_rows: int,
    max_reqs: int,
):
    """eagle3 ``_async_seed_inputs_impl`` with its extras (``num_rows`` > 0), plus step 1's
    seq_lens (``seq_lens + 1 - rejected``, the base loop's eager pair) and the spec tail's
    bonus and rejected count. Returns the base five outputs, then those three."""
    out = eagle3._async_seed_inputs_impl(
        next_tokens, num_draft, query_start_loc, num_reqs, input_ids, p, num_rows
    )
    seq_lens_step1 = seq_lens + 1 - out[4]
    bonus, tail_rejected = tail_bonus_rejected(next_tokens, num_draft, num_reqs, max_reqs)
    return (*out, seq_lens_step1, bonus, tail_rejected)


_async_seed_inputs_asm_compiled = eagle3._static_shape_args(
    torch.compile(backend="tpu", fullgraph=True, dynamic=False)(
        _async_seed_inputs_asm_impl
    )
)


def _carry_token_fresh_asm_impl(proposer, positions, indices, last_hidden, hidden):
    """eagle3 ``_draft_rows_carry_token_fresh`` (position carry, first draft token, hidden carry
    in a buffer of its own), plus step 1's positions (``carry + 1``, the base loop call's eager
    add)."""
    rows = indices.shape[0]
    positions_carry, token = proposer._carry_and_token_body(
        positions, indices, last_hidden[:rows]
    )
    return positions_carry, token, hidden[:rows].clone(), positions_carry + 1


_carry_token_fresh_asm_compiled = eagle3._static_shape_args(
    torch.compile(backend="tpu", fullgraph=True, dynamic=False)(
        _carry_token_fresh_asm_impl
    )
)


def _carry_and_token_asm_impl(proposer, positions, indices, last_hidden):
    """eagle3 ``_draft_rows_carry_and_token`` (the carry when step 0's rows are not wider than
    the carry rows, p >= capacity), plus step 1's positions."""
    positions_carry, token = proposer._carry_and_token_body(positions, indices, last_hidden)
    return positions_carry, token, positions_carry + 1


_carry_and_token_asm_compiled = eagle3._static_shape_args(
    torch.compile(backend="tpu", fullgraph=True, dynamic=False)(_carry_and_token_asm_impl)
)


class _MtpFusedLoopAsm(_MtpFusedLoop):
    """``_MtpFusedLoop`` that also returns the spec tail's source (see `fold_source`)."""

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
        seq_lens: torch.Tensor,
        tail_bonus: torch.Tensor,
        num_reqs: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """The K-1 loop tokens, then the ``[max_reqs * (1 + K)]`` source."""
        tokens = _MtpFusedLoop.forward(
            self, input_ids, positions, hidden_states, seq_lens,
            intermediate_tensors, inputs_embeds,
        )
        return (*tokens, fold_source(input_ids, tokens, tail_bonus, num_reqs, self.max_reqs))


def build_mtp_fused_loop_asm(
    *,
    vllm_config: Any,
    prefix: str,
    mtp: Any,
    num_steps: int,
    dynamic_arg_dims: dict,
    dvp: bool = False,
    max_reqs: int,
) -> torch.nn.Module:
    """``build_mtp_fused_loop`` for `_MtpFusedLoopAsm`: the draft's dims the forward takes;
    the tail inputs are constant-shaped and stay static."""
    loop_args = inspect.signature(_MtpFusedLoopAsm.forward).parameters
    dims = {k: v for k, v in dict(dynamic_arg_dims).items() if k in loop_args}
    cls = support_torch_compile(dynamic_arg_dims=dims)(
        type("MtpFusedLoopAsm", (_MtpFusedLoopAsm,), {})
    )
    return cls(
        vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp,
        max_reqs=max_reqs,
    )


# ---------------------------------------------------------------- results and helpers


@dataclasses.dataclass
class DraftAsmResult:
    """What a folded ``propose`` returns in place of the eager ``[num_reqs, K]`` drafts:
    the spec tail's outputs, and the inputs they were made from (the runner stores them only
    for the same verify output, draft lengths and request count)."""

    src: torch.Tensor
    num_rejected: torch.Tensor
    next_tokens: torch.Tensor
    draft_lengths: torch.Tensor
    num_reqs: int
    tokens: list

    def drafts(self) -> torch.Tensor:
        """The base path's drafts, eagerly (only if the runner cannot use the fold)."""
        return torch.stack(self.tokens, dim=1)[: self.num_reqs]


@dataclasses.dataclass
class _Step:
    mode: str  # "fold" | "check" | "warm"
    chunk: Any
    next_tokens: torch.Tensor
    num_reqs: int
    num_reqs_dev: torch.Tensor | None = None
    seed: tuple | None = None
    carry: tuple | None = None
    loop: tuple | None = None
    seq1_host: torch.Tensor | None = None
    mismatches: list = dataclasses.field(default_factory=list)


def _key(*args) -> tuple:
    return tuple(
        (tuple(a.shape), a.dtype) if isinstance(a, torch.Tensor) else a for a in args
    )


def _is_tpu(t: torch.Tensor) -> bool:
    """The base fused seed's own device test (``nt.device.type == "tpu"``)."""
    return t.device.type == "tpu"


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - no TP group (tests, single chip)
        return 0


def _rss_bytes() -> int | None:
    """This process's resident set (Linux /proc/self/statm), or None."""
    try:
        import os

        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001 - not Linux
        return None


_L6_AB: list = []


def _l6_ab():
    """core/l6_ab.py, or None on a tree without it (looked up once per process)."""
    if not _L6_AB:
        try:
            from vllm_torchtpu.core import l6_ab
        except ImportError:  # a tree without core/l6_ab.py
            l6_ab = None
        _L6_AB.append(l6_ab)
    return _L6_AB[0]


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.detach().cpu(), b.detach().cpu()
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))


# ---------------------------------------------------------------- the per-proposer controller


class DraftAsm:
    """Per-proposer state of the fold: step mode, warm keys, CHECK, window, counters."""

    # q38 L4 AGREE (runner/l4_agree.py): this lever's name and entries in the runner's one gather.
    AGREE_NAME = LEVER
    AGREE_WIDTH = 5

    def __init__(self, proposer: Any) -> None:
        self.p = proposer
        self.window_on = True
        self.step: _Step | None = None
        self.warm: dict[str, set] = {"seed": set(), "carry": set(), "loop": set()}
        self.loop_module: Any = None
        self.loop_module_failed = False
        self.loop_warm = False
        self.failed = False
        self.agreed = False
        self.local_disabled: list[str] = []
        self.boundaries = 0
        self.period: int | None = None
        self.fp_cache: tuple | None = None
        self.loop_rss = [0, 0]  # bytes, programs (warm-up)
        self.check_steps = int(envs.VLLM_TPU_DRAFT_ASM_CHECK_STEPS)
        self.check_left = self.check_steps
        self.check_pending = self.check_steps > 0
        self.check_mismatches = 0
        self.ab = (envs.VLLM_TPU_DRAFT_ASM_AB or "").strip().lower()
        self.ab_error_logged = False
        self.stats: collections.Counter = collections.Counter()
        self.reasons: collections.Counter = collections.Counter()
        self.serving_calls = 0
        self.engaged_logged = False
        self.mismatch_logged = False
        self.static_reason: str | None = None
        self.rank: int | None = None
        # The runner's agreement clock; set by l4_agree.register (its `step` is the engine step).
        self.agreement: Any = None
        l4_agree.register(proposer.runner, self)

    # -------------------------------------------------------- eligibility and mode

    @property
    def max_reqs(self) -> int:
        return int(self.p.runner.max_num_reqs)

    def _static(self) -> str:
        """Why the served config can never fold ("" when it can); computed once."""
        if self.static_reason is not None:
            return self.static_reason
        p, runner = self.p, self.p.runner
        reason = ""
        if not (
            envs.VLLM_TPU_FUSED_ASYNC_DRAFT_SEED
            and envs.VLLM_TPU_DRAFT_SEED_FUSED
            and envs.VLLM_TPU_DRAFT_SEED_EXTRAS
        ):
            reason = "seed_flags"
        elif not (envs.VLLM_TPU_DRAFT_CARRY_FUSED and envs.VLLM_TPU_DRAFT_CARRY_ROWS_FRESH):
            reason = "carry_flags"
        elif not (envs.VLLM_TPU_SPEC_TAIL_COMPILED and envs.VLLM_TPU_SPEC_TAIL_REJ_PAD):
            reason = "tail_flags"
        elif not getattr(p, "_draft_rows", False):
            reason = "draft_rows"
        elif getattr(p, "_mtp_fused_loop", None) is None or runner._dp_lockstep_enabled():
            reason = "fused_loop"
        elif p.speculative_config.num_speculative_tokens < 2 or not p.advance_draft_positions:
            reason = "fused_loop"
        elif getattr(p, "bucketed_drafts", False) or getattr(runner, "uses_mrope", False):
            reason = "layout"
        elif getattr(runner, "_fence_spec_kv_under_token_copy", False):
            reason = "kv_fence"
        elif self._needs_tp_pin():
            reason = "tp_pin"
        self.static_reason = reason
        return reason

    def _needs_tp_pin(self) -> bool:
        """Whether the runner pins replicated drafts to rank 0 (an all-reduce on the drafts
        the fold does not build)."""
        sc = self.p.speculative_config
        if getattr(sc, "draft_tensor_parallel_size", None) != 1:
            return False
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size

            return get_tensor_model_parallel_world_size() > 1
        except Exception:  # noqa: BLE001 - no TP group
            return False

    def _reason(self, chunks, next_tokens_per_chunk, scheduler_output, return_device) -> str:
        if next_tokens_per_chunk is None or not return_device:
            return "not_async_verify"
        reason = self._static()
        if reason:
            return reason
        if len(chunks) != 1 or len(next_tokens_per_chunk) != 1:
            return "chunks"
        chunk = chunks[0]
        if chunk.draft_lengths is None:
            return "no_spec_metadata"
        if chunk.sequence_layout_plan.descriptor != DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR:
            return "layout"
        if chunk.attn_ctx.position_ids_override is not None or chunk.position_ids.ndim != 1:
            return "layout"
        if getattr(scheduler_output, "has_structured_output_requests", False):
            return "structured_output"
        qsl = chunk.query_start_loc_np[: chunk.num_reqs + 1]
        per_req = qsl[1:] - qsl[:-1]
        k = self.p.speculative_config.num_speculative_tokens
        if per_req.size == 0 or int(per_req.min()) < 1 or int(per_req.max()) > 1 + k:
            return "mixed"
        return ""

    def window(self) -> bool:
        """Whether this step may fold: the external clock's `window_on`, then the l6_ab
        lever where core/l6_ab.py registers it."""
        if not self.window_on:
            return False
        ab = self.ab
        l6 = _l6_ab()
        registered = l6 is not None and LEVER in getattr(l6, "LEVERS", {})
        if not registered:
            if ab.startswith("ab:"):
                self._ab_error(
                    "VLLM_TPU_DRAFT_ASM_AB=%s needs core/l6_ab.py with the lever %r; base path"
                    % (ab, LEVER)
                )
                return False
            return ab not in ("0", "false", "off")
        if ab in ("", "1", "true", "on"):
            return True
        if ab in ("0", "false", "off"):
            return False
        # Windowed: the two paths' programs carry different collectives, so the classes must
        # be l6_ab's lockstep-confirmed step windows, never wall-clock ones.
        if getattr(l6, "STEPS", 0) <= 0:
            self._ab_error(
                "VLLM_TPU_DRAFT_ASM_AB=%s: draftasm windows need VLLM_TPU_L6_AB_STEPS "
                "(step windows); base path" % ab
            )
            return False
        return bool(l6.on(LEVER))

    def _ab_error(self, msg: str) -> None:
        if not self.ab_error_logged:
            self.ab_error_logged = True
            logger.error("q38 DRAFT-ASM: %s", msg)

    def begin(self, chunks, next_tokens_per_chunk, scheduler_output, return_device) -> None:
        """Fix this propose call's mode (None = base path) from the last agreement. A propose call never
        agrees: the runner's engine-step clock does (runner/l4_agree.py), so every propose call of one
        engine step (the K inner steps of a K-step burst) sees the same agreement."""
        self.step = None
        serving = bool(getattr(self.p.runner, "warmup_done", False))
        if serving:
            self.serving_calls += 1
        reason = self._reason(chunks, next_tokens_per_chunk, scheduler_output, return_device)
        if reason:
            if serving:
                self.stats["ineligible"] += 1
                self.reasons[reason] += 1
            self._maybe_log()
            return
        if not serving:
            mode = "warm"
        elif self.failed:
            mode = None
            self.stats["failed"] += 1
        elif self.check_left > 0:
            self.check_left -= 1
            mode = "check"
        elif self.check_pending or not self.agreed:
            mode = None
            self.stats["window_off"] += 1
        else:
            mode = "fold"
        if mode is not None:
            chunk = chunks[0]
            self.step = _Step(mode, chunk, next_tokens_per_chunk[0], chunk.num_reqs)
        self._maybe_log()

    def _period(self) -> int:
        if self.period is None:
            l6 = _l6_ab()
            steps = int(getattr(l6, "STEPS", 0) or 0) if l6 is not None else 0
            windowed = self.ab.startswith("ab:") and l6 is not None and LEVER in getattr(l6, "LEVERS", {})
            self.period = steps if (windowed and steps > 0) else AGREE_EVERY
        return self.period

    def agree_period(self) -> int:
        """Engine steps between this lever's agreement boundaries (runner/l4_agree.py)."""
        return self._period()

    def agree_roll(self) -> bool:
        """Whether the wish follows l6_ab step windows (then every window roll is also a boundary)."""
        l6 = _l6_ab()
        return (self.ab.startswith("ab:") and l6 is not None and LEVER in getattr(l6, "LEVERS", {})
                and int(getattr(l6, "STEPS", 0) or 0) > 0)

    def disable_locally(self, why: str) -> None:
        """A rank-local reason to stop folding; it takes effect at the next boundary, on every rank."""
        if why not in self.local_disabled:
            self.local_disabled.append(why)
            logger.error("q38 DRAFT-ASM: local disable on TP rank %d (%s); every rank goes to the base "
                         "path at the next agreement", self._rank(), why)

    def _fingerprint(self) -> int:
        key = tuple(len(v) for v in self.warm.values())
        if self.fp_cache is None or self.fp_cache[0] != key:
            import hashlib

            raw = repr(sorted((k, sorted(repr(x) for x in v)) for k, v in self.warm.items())).encode()
            self.fp_cache = (key, int(hashlib.sha1(raw).hexdigest()[:15], 16))
        return self.fp_cache[1]

    def local_vector(self) -> list:
        """This rank's contribution to a boundary agreement."""
        close = self.check_pending and self.check_left == 0
        return [
            int(self.window()),
            int(bool(self.local_disabled)),
            int(close),
            self.check_mismatches if close else 0,
            self._fingerprint(),
        ]

    def apply(self, rows: list) -> None:
        """The decision every rank computes from the same gathered rows (this lever's 5 entries of every
        rank, from the runner's one agreement gather)."""
        self.boundaries += 1
        n = self.agreement.step if self.agreement is not None else 0
        wants, disabled, closes, mism, fps = (list(c) for c in zip(*rows))
        why = []
        if len(set(closes)) != 1:
            why.append("ranks disagree on the CHECK window (%s)" % closes)
        elif closes[0]:
            self.check_pending = False
            if sum(mism):
                why.append("CHECK: %d mismatches over the TP group (%d on this rank) in the first %d "
                           "checked steps" % (sum(mism), self.check_mismatches, self.check_steps))
            else:
                logger.info("q38 DRAFT-ASM CHECK: %d steps, 0 mismatches over the TP group; fold armed "
                            "(TP rank %d)", self.check_steps, self._rank())
        if any(disabled):
            why.append("local disable on TP ranks %s" % [r for r, d in enumerate(disabled) if d])
        if len(set(fps)) != 1:
            why.append("unequal warm-key sets across TP ranks")
        if why and not self.failed:
            self.failed = True
            logger.error("q38 DRAFT-ASM agreement at engine step %d: %s; every rank keeps the base path "
                         "for the rest of the run (TP rank %d)", n, "; ".join(why), self._rank())
        if self.boundaries == 1 and self.loop_rss[1]:
            logger.info("q38 DRAFT-ASM warm-up: %d folded loop programs, RSS delta %.1f MiB (%.1f MiB per "
                        "program) (TP rank %d)", self.loop_rss[1], self.loop_rss[0] / 2**20,
                        self.loop_rss[0] / 2**20 / self.loop_rss[1], self._rank())
        agreed = not self.failed and all(wants)
        if agreed != self.agreed or self.boundaries == 1:
            logger.info("q38 DRAFT-ASM agreement at engine step %d: fold %s (wishes %d of %d) "
                        "(TP rank %d)", n, "on" if agreed else "off", sum(wants), len(wants), self._rank())
        self.agreed = agreed

    def _cold(self, st: _Step, what: str) -> None:
        self.stats["cold"] += 1
        self.reasons["cold_" + what] += 1
        self.step = None

    def _rank(self) -> int:
        if self.rank is None:
            self.rank = _tp_rank()
        return self.rank

    # -------------------------------------------------------- the three programs

    def seed(self, chunk, nt, num_draft_full, num_reqs, ov_i):
        """In place of eagle3's fused async seed block. Fold: the folded seed, then the rest
        of that block; returns its result. Warm / check: the folded seed beside the base one
        (which the caller then runs); returns None. None also when the step does not fold."""
        st = self.step
        if st is None or ov_i or chunk is not st.chunk or nt is not st.next_tokens:
            return None
        if not _is_tpu(nt):
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
            # q38 L4 VFY-ONE value 2 (DH-C; one seed fold per step): the verify program already
            # made this superset seed (vfy_seed_spec below); taken only on full argument identity.
            out = p._vfy_seed_take(chunk, nt, _async_seed_inputs_asm_impl, args[1:])
        if out is None:
            out = _async_seed_inputs_asm_compiled(*args)
        st.seed, st.num_reqs_dev = out, n_dev
        if st.mode == "warm":
            self.warm["seed"].add(key)
        if st.mode != "fold":
            return None
        # eagle3's fused-seed block from here, on the folded seed's base outputs.
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

    # -------------------------------------------------------- VFY-ONE value 2 (DH-C) hooks
    # q38 L4 VFY-BUILD composition rule: one seed fold per step. With VLLM_TPU_VPS_VFY_ONE=2 the
    # runner's verify program traces the seed eagle3.Eagle3Proposer.vfy_seed_spec names; with a
    # DraftAsm on the proposer that is this superset (the base five outputs, then step 1's seq_lens
    # and the spec tail's bonus / rejected rows), so the fold-mode seed above takes it and DH-A's
    # carry / loop / tail folds keep their inputs, while eagle3's base block (steps DRAFT-ASM does
    # not fold) takes its first five. None when DRAFT-ASM can never fold here: then the verify does
    # not fold the seed and every seed program runs as without VFY-ONE.

    def _vfy_seed_args(self, num_draft, qsl, num_reqs, input_ids, seq_lens, capacity):
        base = self.p._vfy_seed_args(num_draft, qsl, num_reqs, input_ids, capacity, True)
        return (*base[:4], seq_lens, base[4], base[5], self.max_reqs), base

    def vfy_seed_spec(self, chunk, num_reqs):
        """The superset seed of this chunk for the VFY verify program (runner/vfy_one.VfySeedSpec)."""
        from vllm_torchtpu.runner.vfy_one import VfySeedSpec

        if self._static() or chunk.draft_lengths is None:
            return None
        if chunk.sequence_layout_plan.descriptor != DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR:
            return None
        args, base = self._vfy_seed_args(
            chunk.draft_lengths, chunk.attn_ctx.query_start_loc, num_reqs, chunk.input_ids,
            chunk.attn_ctx.seq_lens, self.p._draft_rows_len(chunk),
        )
        return VfySeedSpec(_async_seed_inputs_asm_impl, args, base)

    def vfy_seed_warm_spec(self, num_tokens, num_reqs_padded, capacity, num_reqs):
        """The same at a steady decode shape, with zero tensors (VFY-ONE warm-up)."""
        from vllm_torchtpu.runner.vfy_one import VfySeedSpec

        if self._static():
            return None
        dev = self.p.runner.device
        args, base = self._vfy_seed_args(
            torch.zeros(num_reqs_padded, dtype=torch.int32).to(dev),
            torch.zeros(capacity + 1, dtype=torch.int32).to(dev),
            num_reqs,
            torch.zeros(num_tokens, dtype=torch.int32).to(dev),
            torch.zeros(capacity, dtype=torch.int32).to(dev),
            capacity,
        )
        return VfySeedSpec(_async_seed_inputs_asm_impl, args, base)

    def carry(self, chunk, positions, last_token_indices, untrimmed, hidden, last_hidden):
        """In place of eagle3's carry-fused branch (fresh carry when step 0's rows are wider
        than the carry rows, else the plain carry on the trimmed rows). Fold: the folded carry;
        returns (positions_carry, token0, hidden_carry, last_hidden_carry). Warm / check: the
        folded carry beside the base one; returns None (the base branch then runs).
        `st.carry` is (positions_carry, token0, hidden_carry, positions_step1)."""
        st = self.step
        if st is None or st.seed is None or chunk is not st.chunk:
            return None
        p = self.p
        fresh = untrimmed is not None and untrimmed[2] == last_token_indices.shape[0]
        if fresh:
            args = (positions, last_token_indices, untrimmed[0], untrimmed[1])
        else:
            args = (positions, last_token_indices, last_hidden)
        key = _key(*args)
        if st.mode == "fold" and key not in self.warm["carry"]:
            self._cold(st, "carry")
            return None
        with set_model_tag("eagle_head"):
            if fresh:
                out = _carry_token_fresh_asm_compiled(p, *args)
            else:
                pc, tok, pos1 = _carry_and_token_asm_compiled(p, *args)
                out = (pc, tok, hidden, pos1)
        st.carry = out
        if st.mode == "warm":
            self.warm["carry"].add(key)
        if st.mode != "fold":
            return None
        positions_carry, token0, hidden_carry, _ = out
        if fresh and not getattr(p, "_carry_rows_fresh_logged", False):
            p._carry_rows_fresh_logged = True
            logger.info(
                "DRAFT_CARRY_ROWS_FRESH: step 0 carry, token and hidden trimmed in one "
                "program (%d of %d rows)", untrimmed[2], untrimmed[0].shape[0],
            )
        if p._carry_checks_done < envs.VLLM_TPU_VERIFY_FUSED_CHECK:
            p._check_carry_fused(
                positions, last_token_indices, last_hidden, positions_carry, token0
            )
        return positions_carry, token0, hidden_carry, last_hidden

    def _loop(self):
        if self.loop_module is None and not self.loop_module_failed:
            p = self.p
            try:
                with (
                    set_model_tag("eagle_head"),
                    set_vllm_model_wrapper_context(
                        mesh=p.runner.mesh, vllm_config=p.draft_vllm_config
                    ),
                    set_current_vllm_config(p.draft_vllm_config),
                ):
                    k = p.speculative_config.num_speculative_tokens
                    self.loop_module = build_mtp_fused_loop_asm(
                        vllm_config=p.draft_vllm_config,
                        prefix=f"mtp_fused_loop_asm_k{k}",
                        mtp=p.draft_model,
                        num_steps=k - 1,
                        dynamic_arg_dims=getattr(p.draft_model, "_dynamic_arg_dims", {}),
                        dvp=getattr(p, "_dvp", False),
                        max_reqs=self.max_reqs,
                    )
            except Exception as exc:
                # Fail-stop: every rank runs the folded loop on the same steps (its collectives
                # pair across ranks), so one rank may not quietly skip it.
                raise RuntimeError(
                    "q38 DRAFT-ASM: could not build the folded draft loop; set "
                    "VLLM_TPU_DRAFT_ASM_FOLD=0 to run without the fold"
                ) from exc
        return self.loop_module

    def loop(self, chunks, draft_tokens_per_chunk, loop_hidden, loop_qsl, loop_reqdist) -> bool:
        """In place of the fused loop call. Fold: the folded loop; appends its K-1 tokens and
        returns True (the base loop is skipped). Warm / check: the folded loop on the folded
        seed and carry outputs, before the base loop (whose KV writes then stand); False."""
        st = self.step
        if st is None or st.seed is None or st.carry is None:
            return False
        if len(chunks) != 1 or chunks[0] is not st.chunk:
            self.step = None
            return False
        p, chunk = self.p, st.chunk
        target_md = p._target_draft_attn_metadata(chunk)
        loop_module = self._loop()
        if target_md is None or loop_module is None:
            if st.mode == "fold":
                self.stats["partial"] += 1
                self.reasons["partial_loop"] += 1
            self.step = None
            return False
        seq_lens1, tail_bonus, n_dev = st.seed[5], st.seed[6], st.num_reqs_dev
        token0, hidden, positions1 = st.carry[1], st.carry[2], st.carry[3]
        if st.mode == "fold":
            # The same tensors the base path would hand the loop.
            token0, hidden = draft_tokens_per_chunk[0][-1], loop_hidden[0]
        key = _key(token0, positions1, hidden, seq_lens1, tail_bonus, n_dev, loop_qsl[0])
        if st.mode == "fold" and key not in self.warm["loop"]:
            self._cold(st, "loop")
            return False
        replaced: dict[int, Any] = {}
        md = {}
        for name, m in target_md.items():
            if id(m) not in replaced:
                replaced[id(m)] = dataclasses.replace(
                    m,
                    seq_lens=seq_lens1,
                    query_start_loc=loop_qsl[0],
                    request_distribution=loop_reqdist[0],
                )
            md[name] = replaced[id(m)]
        if st.mode == "check":
            # A copy taken before the loop advances seq_lens in place (on CPU .cpu() is the same tensor).
            st.seq1_host = seq_lens1.detach().cpu().clone()
        num_tokens = token0.shape[0]
        with (
            set_model_tag("eagle_head"),
            set_forward_context(
                md,
                p.draft_vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=p.runner._dp_num_tokens_across_dp(num_tokens),
            ),
            set_vllm_model_wrapper_context(
                mesh=p.runner.mesh, vllm_config=p.draft_vllm_config
            ),
        ):
            new_key = st.mode == "warm" and key not in self.warm["loop"]
            rss0 = _rss_bytes() if new_key else None
            if not self.loop_warm:
                # As the base loop's first call: trace and compile, put seq_lens back (the
                # program advanced it), then run compiled.
                keep = seq_lens1.clone()
                loop_module(token0, positions1, hidden, seq_lens1, tail_bonus, n_dev)
                seq_lens1.copy_(keep)
                self.loop_warm = True
            out = loop_module(token0, positions1, hidden, seq_lens1, tail_bonus, n_dev)
            if rss0 is not None:
                rss1 = _rss_bytes()
                if rss1 is not None:
                    self.loop_rss[0] += max(0, rss1 - rss0)
                    self.loop_rss[1] += 1
        st.loop = tuple(out)
        if st.mode == "warm":
            self.warm["loop"].add(key)
        if st.mode != "fold":
            return False
        draft_tokens_per_chunk[0].extend(st.loop[:-1])
        return True

    def result(self, chunks, draft_tokens_per_chunk, positions_carry_per_chunk, return_device):
        """At the end of propose. Fold: the DraftAsmResult (in place of the eager stack + cat).
        Check: compare the draft-side values with the base path's; None. Warm: None."""
        st = self.step
        if st is None:
            return None
        if st.mode == "check":
            self._check_draft_side(st, draft_tokens_per_chunk, positions_carry_per_chunk)
            return None
        if st.mode != "fold":
            return None
        if st.loop is None or not return_device:
            self.stats["partial"] += 1
            self.reasons["partial_result"] += 1
            self.step = None
            return None
        self.stats["folded"] += 1
        if not self.engaged_logged:
            self.engaged_logged = True
            logger.info("q38 DRAFT-ASM fold: on (TP rank %d)", self._rank())
        l6 = _l6_ab()
        if l6 is not None and LEVER in getattr(l6, "LEVERS", {}):
            l6.note(LEVER, "fold")
        return DraftAsmResult(
            src=st.loop[-1],
            num_rejected=st.seed[7],
            next_tokens=st.next_tokens,
            draft_lengths=st.chunk.draft_lengths,
            num_reqs=st.num_reqs,
            tokens=list(draft_tokens_per_chunk[0]),
        )

    # -------------------------------------------------------- CHECK

    def _mismatch(self, st: _Step, what: str) -> None:
        st.mismatches.append(what)
        self.check_mismatches += 1
        if not self.mismatch_logged:
            self.mismatch_logged = True
            logger.error(
                "q38 DRAFT-ASM CHECK mismatch: %s differs from the base path at %d requests; "
                "counted for the next agreement (TP rank %d)", what, st.num_reqs, self._rank(),
            )

    def _check_draft_side(self, st, draft_tokens_per_chunk, positions_carry_per_chunk) -> None:
        self.stats["checked"] += 1
        if st.seed is None or st.carry is None or st.loop is None:
            self._mismatch(st, "coverage (a folded program did not run)")
            return
        pos_base = positions_carry_per_chunk[0].detach().cpu() + 1
        if not _same(st.carry[3], pos_base):
            self._mismatch(st, "positions_step1")
        rejected = self.p._seed_extras.get(id(st.chunk))
        if rejected is None or st.seq1_host is None:
            self._mismatch(st, "seq_lens_step1 (no base rejected count)")
        else:
            want = st.chunk.attn_ctx.seq_lens.detach().cpu() + 1 - rejected[1].detach().cpu()
            if not _same(st.seq1_host, want):
                self._mismatch(st, "seq_lens_step1")
        base = draft_tokens_per_chunk[0]
        folded = [st.carry[1], *st.loop[:-1]]
        if len(base) != len(folded) or not all(_same(a, b) for a, b in zip(base, folded)):
            self._mismatch(st, "drafts")

    def check_tail(self, src: torch.Tensor, num_rejected: torch.Tensor) -> None:
        """After the base spec tail of a CHECK step: compare its source and rejected count."""
        st = self.step
        if st is None or st.mode != "check":
            return
        if st.loop is None or st.seed is None:
            return
        if not _same(st.loop[-1], src):
            self._mismatch(st, "source")
        if not _same(st.seed[7], num_rejected):
            self._mismatch(st, "rejected count")

    # -------------------------------------------------------- runner side

    def assemble_tail(self, runner, drafts, next_tokens_per_chunk, state):
        """``_assemble_async_spec_substitution`` for a folded propose: (the four outputs, drafts)
        when the fold's source and rejected count are this chunk's, else (None, eager drafts)."""
        if not isinstance(drafts, DraftAsmResult):
            return None, drafts
        md = state.spec_decode_metadata_list[0] if len(state.spec_decode_metadata_list) == 1 else None
        bucketed = getattr(getattr(runner, "drafter", None), "bucketed_drafts", False)
        if (
            len(next_tokens_per_chunk) == 1
            and md is not None
            and next_tokens_per_chunk[0] is drafts.next_tokens
            and md.draft_lengths is drafts.draft_lengths
            and state.num_reqs_list[0] == drafts.num_reqs
            and not bucketed
            and envs.VLLM_TPU_SPEC_TAIL_COMPILED
            and envs.VLLM_TPU_SPEC_TAIL_REJ_PAD
        ):
            n = drafts.num_reqs
            return (
                [drafts.src],
                {r: r for r in range(n)},
                drafts.num_rejected,
                {r: int(md.draft_lengths_cpu[r]) for r in range(n)},
            ), drafts
        self.stats["tail_miss"] += 1
        return None, drafts.drafts()

    # -------------------------------------------------------- precompile

    def decode_shapes(self) -> list[tuple]:
        """(T, width, p, cap) of every steady decode verify step: 1..max_num_reqs requests with K drafts
        each (token bucket of R * (1 + K), verify request bucket, loop bucket, request capacity). These
        are the served decode buckets (T <= 256 at 64 requests, K = 3); a step whose requests carry
        fewer drafts can land in a lower token bucket, finds its key cold and takes the base path."""
        from vllm_torchtpu.runner.tpu_runner import (
            _get_padded_num_reqs_with_upper_limit,
            _get_padded_token_len,
        )

        p, runner = self.p, self.p.runner
        k = p.speculative_config.num_speculative_tokens
        pads = list(runner.num_tokens_paddings)
        caps = sorted(
            {int(c) for c in (runner.num_reqs_max_model_len, runner.num_reqs_most_model_len) if c}
        )
        out = set()
        for r in range(1, self.max_reqs + 1):
            if r * (1 + k) > pads[-1]:
                continue
            t = _get_padded_token_len(pads, r * (1 + k))
            for cap in caps:
                out.add((t, _get_padded_num_reqs_with_upper_limit(r, self.max_reqs), p._loop_bucket(r), cap))
        return sorted(out)

    def precompile(self) -> None:
        """The folded seed and carry at every decode-only shape (the base ones are warmed by
        eagle3's own precompiles; the folded loop by the spec-decode warm-up)."""
        if self._static():
            logger.info("q38 DRAFT-ASM precompile skipped: %s", self._static())
            return
        p, runner = self.p, self.p.runner
        k = p.speculative_config.num_speculative_tokens
        hidden_size = p._draft_hidden_size()
        dev = runner.device
        n_seed = n_carry = 0
        rss0 = _rss_bytes()
        with set_model_tag("eagle_head"), runner._precompile_timed("drafter DRAFT-ASM fold"):
            for t, width, loop_p, cap in self.decode_shapes():
                nt = torch.full((width, k + 1), INVALID_TOKEN_ID, dtype=torch.int32).to(dev)
                nd = torch.zeros(width, dtype=torch.int32).to(dev)
                qsl = torch.zeros(cap + 1, dtype=torch.int32).to(dev)
                n_dev = torch.tensor(1, dtype=torch.int64).to(dev)
                ids = torch.zeros(t, dtype=torch.int32).to(dev)
                seq_lens = torch.zeros(cap, dtype=torch.int32).to(dev)
                args = (nt, nd, qsl, n_dev, ids, seq_lens, loop_p, cap, self.max_reqs)
                key = _key(*args)
                if key not in self.warm["seed"]:
                    synchronize_tensors(_async_seed_inputs_asm_compiled(*args))
                    self.warm["seed"].add(key)
                    n_seed += 1
                if loop_p > t:
                    continue
                positions = torch.zeros(t, dtype=torch.int32).to(dev)
                idx = torch.zeros(loop_p, dtype=torch.int64).to(dev)
                if loop_p < cap:
                    # fresh carry on step 0's untrimmed [cap, h] outputs (two distinct
                    # tensors, as serving passes them)
                    full = torch.zeros((cap, hidden_size), dtype=runner._hidden_states_dtype).to(dev)
                    cargs = (positions, idx, full, torch.zeros_like(full))
                    fn = _carry_token_fresh_asm_compiled
                else:
                    # plain carry on the [p, h] rows
                    rows = torch.zeros((loop_p, hidden_size), dtype=runner._hidden_states_dtype).to(dev)
                    cargs = (positions, idx, rows)
                    fn = _carry_and_token_asm_compiled
                ckey = _key(*cargs)
                if ckey not in self.warm["carry"]:
                    synchronize_tensors(fn(p, *cargs))
                    self.warm["carry"].add(ckey)
                    n_carry += 1
        rss1 = _rss_bytes()
        self.precompile_rss = (rss0, rss1, n_seed + n_carry)
        delta = "n/a" if rss0 is None or rss1 is None else "%.1f MiB (%.1f MiB per program)" % (
            (rss1 - rss0) / 2**20, (rss1 - rss0) / 2**20 / max(1, n_seed + n_carry))
        logger.info(
            "q38 DRAFT-ASM precompile: %d folded seed + %d folded carry programs (steady decode "
            "shapes); RSS delta %s; the folded loop warms in the spec-decode warm-up", n_seed,
            n_carry, delta,
        )

    # -------------------------------------------------------- markers

    def counts(self, done: int | None = None) -> dict:
        """Counters over the serving propose calls; `done` = calls finished (default: all seen)."""
        done = self.serving_calls if done is None else done
        return {
            "folded": self.stats["folded"],
            "base": done - self.stats["folded"],
            "window_off": self.stats["window_off"],
            "ineligible": self.stats["ineligible"],
            "cold": self.stats["cold"],
            "partial": self.stats["partial"],
            "failed": self.stats["failed"],
            "checked": self.stats["checked"],
            "mismatches": self.check_mismatches,
            "tail_miss": self.stats["tail_miss"],
            "agreed": self.agreed,
            "boundaries": self.boundaries,
            "warm_keys": {k: len(v) for k, v in self.warm.items()},
        }

    def _maybe_log(self) -> None:
        """Every LOG_EVERY finished serving calls (checked as the next one begins)."""
        n = self.serving_calls - 1
        if n <= 0 or n % LOG_EVERY:
            return
        c = self.counts(done=n)
        top = "+".join(f"{k}={v}" for k, v in self.reasons.most_common(4)) or "-"
        logger.info(
            "q38 DRAFT-ASM: %d folded, %d base (window_off %d, ineligible %d [%s], cold %d, "
            "partial %d, failed %d), check %d steps %d mismatches, tail_miss %d; TP rank %d",
            c["folded"], c["base"], c["window_off"], c["ineligible"], top, c["cold"],
            c["partial"], c["failed"], c["checked"], c["mismatches"], c["tail_miss"],
            self._rank(),
        )
