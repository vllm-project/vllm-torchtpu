# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VFY-ONE (VLLM_TPU_VPS_VFY_ONE; H1 cards T-3 VFY-ONE and DH-C SEED-INTO-VERIFY).

On a single-chunk VPS fallback step with D83's read-offset fold on, the D83 chain runs two compiled
programs and three eager ops after the forward: the local LM-head logits
(``compute_selected_local_logits``) in execute_model, then in sample_tokens ``_vps_verify_fb_compiled``
(which already returns D83's read-offset operands), the stats ``.to(int64)`` and ``+`` and D83's one
``index_put_``. VFY-ONE runs ONE program, ``tpu_runner._vfy_one_impl``: the index_select, the local
logits, the very same ``_vps_verify_fb_impl``, D83's operands and the stats accumulate as an
input-to-output pair; D83's ``index_put_`` stays. The three uniform draws (and the fallback rows' draw)
stay eager in front of the program, in the same order and shapes, so tokens, random stream, stats and
read offsets are the D83 chain's. Value 2 (DH-C) also traces the drafter's fused seed body on the
verify's tokens; the drafter takes those outputs instead of calling its seed program
(``Eagle3Proposer._vfy_seed_take``, only on full argument identity).

The compile key sees VLLM_TPU_VPS_VFY_ONE as on / off: both program variants (with and without the
seed) are precompiled whenever it is on, so the seed is switched on by value. The D83 chain stays
precompiled beside them (SAB dual-path). Precompile covers steady decode shapes (n live requests with K
drafts each, n = 1 .. max_num_reqs); a step whose program key was never warmed runs the D83 chain
(logits computed late), so serving never compiles a VFY program.

All-rank agreement (lead safety rule). The VFY program and the D83 chain carry their TP all-gather in
different programs, so no rank may change path on its own. The path changes only at boundaries, which are
ENGINE STEPS of the runner's one agreement clock (runner/l4_agree.py, q38 L4 AGREE; the runner registers
this controller): the first serving step, then every l6_ab step-window roll with
``VLLM_TPU_VPS_VFY_ONE_AB=ab:<c>`` (step windows only: a wall-clock window roll can fall on different steps
on different ranks, so it never makes a boundary), else every AGREE_EVERY serving steps, plus any boundary
another registered lever's schedule adds. At a boundary every TP rank contributes (enable bit = window wish
and no local disable, local-disable bit, CHECK closed, CHECK mismatches, warm-key fingerprint, config hash,
serving step count, window class) as its 8 entries of the runner's one all_gather over the TP group's CPU
group, and every rank computes the same decision: VFY until the next boundary iff the MIN of the enable bits
is 1. Unequal config / fingerprints / step counts / classes, any local disable or any CHECK mismatch keeps
every rank on the D83 chain for the rest of the run. Between boundaries nothing rank-local changes the path;
the per-step choices left (single chunk, VPS fallback chunk, a never-warmed key) depend only on the
scheduler output, the batch's sampling parameters and the agreed-equal warm keys. A gather that fails
raises (fail-stop), and so does any exception in a VFY call.

CHECK (VLLM_TPU_VPS_VFY_ONE_CHECK=N; gated on l6_ab.check_allowed where it exists): the first N
eligible serving steps run the D83 chain for the step's results and the VFY program beside it on the
same draws, and count mismatching tokens, stats, read offsets and seed entries (one D2H each). The next
boundary sums the counts over the TP group; VFY is armed only if the sum is 0.

KEYLITE (VLLM_TPU_VPS_VFY_ONE_KEYLITE=1; H2 card T1-VFY-KEYLITE). The warm-key test walks the whole call tuple
every VFY step (`key`: 30 entries plus one per read-offset group, 37 groups on served Flash, plus the seed's).
With KEYLITE the runner passes `lite_key`, a tuple of the scheduler integers that fix every shape and value of
that full key; the verdict of a lite key's first call (its full key looked up then) is reused. The full key is
recomputed on every CHECK step and every KEYLITE_AUDIT_EVERY-th VFY call; an unequal full key turns KEYLITE off
for the run (that step and every later one use the full key). The verdict is a function of rank-uniform inputs
only, so every rank makes the same verdicts and turns KEYLITE off on the same call; it changes no program.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import os
from typing import Any, Callable

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# core/l6_ab.py lever name; its env var is VLLM_TPU_VPS_VFY_ONE_AB.
LEVER = "vfyone"
# Serving steps between all-rank agreements when the switch is uniform.
AGREE_EVERY = 200
LOG_EVERY = 5000
# KEYLITE: VFY calls between full-key audits of the lite-key verdicts.
KEYLITE_AUDIT_EVERY = 256


@dataclasses.dataclass
class VfyChunk:
    """A VPS chunk whose local logits were deferred to the VFY program: what execute_model would have
    passed to compute_selected_local_logits (``indices`` set) or to
    compute_local_logits_from_hidden_states (``indices`` None, the layout planner's pre-selected rows)."""

    hidden: torch.Tensor
    indices: torch.Tensor | None


@dataclasses.dataclass
class VfySeedSpec:
    """DH-C: the drafter's fused seed for one chunk, as its seed block would call it. ``fn(next_tokens,
    *args)`` is traced inside the verify program; ``base_args`` are the arguments eagle3's base fused
    seed block passes after next_tokens (the same as ``args`` unless a DraftAsm supplies a superset
    body whose first five outputs are the base seed's)."""

    fn: Callable
    args: tuple
    base_args: tuple


def key(*args: Any) -> tuple:
    """A program call's shape key: tensors by (shape, dtype), lists / tuples element-wise (with an
    aliasing flag: dynamo specializes on repeated list entries), functions by qualified name."""
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append(("T", tuple(a.shape), str(a.dtype)))
        elif isinstance(a, (list, tuple)):
            out.append(("L", key(*a), len({id(x) for x in a}) == len(a)))
        elif isinstance(a, torch.dtype):
            out.append(("D", str(a)))
        elif callable(a):
            out.append(("F", getattr(a, "__module__", ""), getattr(a, "__qualname__", repr(a))))
        else:
            out.append(("V", a))
    return tuple(out)


def lite_key(hidden: torch.Tensor, indices: torch.Tensor | None, padded_num_reqs: int, md: Any,
             fallback_rows: int, ro_groups: list, spec: VfySeedSpec | None) -> tuple:
    """KEYLITE: the integers that fix every shape and value of a VFY call's full key, without walking the call.
      * T, the hidden rows (the token bucket), and N, the logits indices (-1: the planner's pre-selected rows);
      * P, the padded request rows (bonus indices, sampling params, bonus draws, draft lengths);
      * D, the drafts (draft ids, accept draws, segment and group ids), and the target rows (recover draws);
      * F, the fallback rows (its draw);
      * the read-offset groups: their count, the first one's width and the aliasing flag (dynamo specializes
        on all three; the groups of one step share one width, which the full-key audit checks);
      * the seed: spec.fn, then per argument a tensor's shape or a non-tensor's value.
    The rest of the call (K, the vocab, the dtypes, the stats accumulator, the synthetic rates, cand_draws, the
    two 1-row operands) is fixed for the run; the audit checks that too."""
    seed = None
    if spec is not None:
        seed = (spec.fn, tuple(tuple(a.shape) if isinstance(a, torch.Tensor) else a for a in spec.args))
    return (
        hidden.shape[0],
        -1 if indices is None else indices.shape[0],
        padded_num_reqs,
        md.draft_token_ids.shape[0],
        md.target_logits_indices.shape[0],
        fallback_rows,
        len(ro_groups),
        ro_groups[0].shape[0],
        len(set(map(id, ro_groups))) == len(ro_groups),
        seed,
    )


def versions(args: tuple) -> tuple | None:
    """The in-place version counter of every tensor in `args` (None for other values); None when a tensor
    does not track one (an inference tensor). DH-C's take compares these at the seed block: any in-place
    write to a seed input (or to the verify's tokens) between the verify and the seed makes it a miss."""
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            try:
                out.append(int(a._version))
            except Exception:  # noqa: BLE001 - inference tensors track no version
                return None
        else:
            out.append(None)
    return tuple(out)


def same_args(a: tuple, b: tuple) -> bool:
    """Every tensor the same object, every other value equal."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            if x is not y:
                return False
        elif x != y:
            return False
    return True


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - no TP group (tests, single chip)
        return 0


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


class VfyOne:
    """Per-runner state of VFY-ONE: agreement, step mode, warm keys, CHECK, counters."""

    # q38 L4 AGREE (runner/l4_agree.py): this lever's name and entries in the runner's one gather.
    AGREE_NAME = LEVER
    AGREE_WIDTH = 8

    def __init__(self, mode: int, check: int, ab: str, keylite: bool | None = None) -> None:
        assert mode in (1, 2), mode
        self.mode = mode
        if keylite is None:  # direct construction: the environment, parsed as envs.py parses it
            keylite = (os.environ.get("VLLM_TPU_VPS_VFY_ONE_KEYLITE") or "").strip().lower() in ("1", "true", "yes", "on")
        self.keylite = bool(keylite)
        self.keylite_off = False
        self.lite: dict = {}  # lite key -> (full key, warm verdict), for one warm set (lite_of)
        self.lite_of: tuple = (None, -1)  # (the warm set, its size) the memo was made against
        self.lite_calls = 0
        self.ab = (ab or "").strip().lower()
        l6 = _l6_ab()
        allowed = getattr(l6, "check_allowed", lambda: True)() if l6 is not None else True
        self.check_steps = int(check) if allowed else 0
        self.check_left = self.check_steps
        self.check_pending = self.check_steps > 0
        self.check_mismatches = 0
        self.failed = False
        self.local_disabled: list[str] = []
        self.engaged = False
        self.boundaries = 0
        self.serving = 0
        self.warm: set = set()
        self.fp_cache: tuple | None = None
        self.static_reason: str | None = None
        self.seed_reason: str | None = None
        self.step_mode: str | None = None
        self.logits_dtype: torch.dtype | None = None
        self.acc_zero: torch.Tensor | None = None
        self.programs = 0
        self.rss = [0, 0]  # bytes, programs (precompile)
        self.stats: collections.Counter = collections.Counter()
        self.ab_error_logged = False
        self.engaged_logged = False
        self.rank: int | None = None
        # The runner's agreement clock; set by l4_agree.register (its `step` is the engine step).
        self.agreement: Any = None
        raw = repr((self.mode, self.ab, self.check_steps) + (("keylite",) if self.keylite else ())).encode()
        self.config_hash = int(hashlib.sha1(raw).hexdigest()[:15], 16)

    # -------------------------------------------------------- static eligibility

    def static(self, runner: Any) -> str:
        """Why this runner can never take VFY-ONE ("" when it can); computed once."""
        if self.static_reason is None:
            if getattr(runner, "_vps_head", None) is None:
                reason = "no_vps_head"
            elif not getattr(runner, "_vps_fb_path", False):
                reason = "no_vps_fallback_path"
            elif not getattr(runner, "_vps_ro_fold", False):
                reason = "no_d83_fold"
            elif getattr(runner, "mamba_slot_read_offsets", None) is None:
                reason = "no_read_offsets"
            else:
                reason = ""
            self.static_reason = reason
            if reason:
                logger.warning("q38 VPS_VFY_ONE=%d stays off on this runner: %s", self.mode, reason)
        return self.static_reason

    def seed_on(self, runner: Any) -> bool:
        """Whether value 2 folds the seed on this runner (computed once; the reason is logged)."""
        if self.seed_reason is None:
            drafter = getattr(runner, "drafter", None)
            asm = getattr(drafter, "_draft_asm", None)
            if not getattr(runner, "_is_async_drafter", False) or not hasattr(drafter, "vfy_seed_spec"):
                reason = "no_async_eagle3_drafter"
            elif asm is not None and not hasattr(asm, "vfy_seed_spec"):
                # DRAFT-ASM folds the seed itself and offers no superset hook: one seed fold per step.
                reason = "draft_asm_without_vfy_seed_hook"
            elif not drafter.vfy_seed_static():
                reason = "seed_flags"
            else:
                reason = ""
            self.seed_reason = reason
            if self.mode == 2 and reason:
                logger.warning("q38 VPS_VFY_ONE=2: the draft seed stays with the drafter (%s); VFY-ONE "
                               "runs as value 1", reason)
        return self.seed_reason == ""

    # -------------------------------------------------------- the window switch and the agreement

    def windowed(self) -> bool:
        return self.ab.startswith("ab:")

    def _ab_error(self, msg: str) -> None:
        if not self.ab_error_logged:
            self.ab_error_logged = True
            logger.error("q38 VPS_VFY_ONE: %s", msg)

    def wish(self) -> bool:
        """This rank's wish for the window that starts at this boundary."""
        ab = self.ab
        if ab in ("", "1", "true", "on"):
            return True
        if ab in ("0", "false", "off"):
            return False
        if not ab.startswith("ab:"):
            self._ab_error("VLLM_TPU_VPS_VFY_ONE_AB=%r: expected '', 0, 1 or ab:<c>[+<c>]; D83 chain" % ab)
            return False
        l6 = _l6_ab()
        if l6 is None or LEVER not in getattr(l6, "LEVERS", {}):
            self._ab_error("VLLM_TPU_VPS_VFY_ONE_AB=%s needs core/l6_ab.py with the lever %r; D83 chain"
                           % (ab, LEVER))
            return False
        if int(getattr(l6, "STEPS", 0) or 0) <= 0:
            self._ab_error("VLLM_TPU_VPS_VFY_ONE_AB=%s needs l6_ab step windows (VLLM_TPU_L6_AB_STEPS); "
                           "D83 chain" % ab)
            return False
        return bool(l6.on(LEVER))

    def _cls(self) -> str:
        l6 = _l6_ab()
        cls = l6.current_class() if l6 is not None else None
        return "uniform" if cls is None else str(cls)

    def _rank(self) -> int:
        if self.rank is None:
            self.rank = _tp_rank()
        return self.rank

    def begin_step(self, runner: Any) -> None:
        """Once per execute_model on every rank, before the runner's agreement tick: count serving steps.
        The boundaries and the gather are the runner's engine-step clock's (runner/l4_agree.py)."""
        self.step_mode = None
        if not getattr(runner, "warmup_done", False):
            return
        self.serving += 1
        if self.serving % LOG_EVERY == 0:
            self.log_counts()

    def agree_period(self) -> int:
        """Engine steps between this lever's agreement boundaries: AGREE_EVERY with the uniform switch;
        windowed, none (its boundaries are l6_ab's step-window rolls, `agree_roll`)."""
        return 0 if self.windowed() else AGREE_EVERY

    def agree_roll(self) -> bool:
        return self.windowed()

    def disable_locally(self, why: str) -> None:
        """A rank-local reason to stop; it takes effect at the next boundary, on every rank."""
        if why not in self.local_disabled:
            self.local_disabled.append(why)
            logger.error("q38 VPS_VFY_ONE: local disable on TP rank %d (%s); every rank goes to the D83 "
                         "chain at the next agreement", self._rank(), why)

    def fingerprint(self) -> int:
        n = len(self.warm)
        if self.fp_cache is None or self.fp_cache[0] != n:
            raw = repr(sorted(repr(k) for k in self.warm)).encode()
            self.fp_cache = (n, int(hashlib.sha1(raw).hexdigest()[:15], 16))
        return self.fp_cache[1]

    def local_vector(self) -> list:
        close = self.check_pending and self.check_left == 0
        cls = self._cls() if self.windowed() else "uniform"
        disabled = bool(self.local_disabled)
        return [
            int(self.wish() and not disabled),
            int(disabled),
            int(close),
            self.check_mismatches if close else 0,
            self.fingerprint(),
            self.config_hash,
            self.serving,
            int(cls) if cls.isdigit() else -1,
        ]

    def apply(self, rows: list) -> None:
        """The decision every rank computes from the same gathered rows."""
        self.boundaries += 1
        en, dis, close, mism, fps, cfgs, ns, clss = (list(c) for c in zip(*rows))
        why = []
        if len(set(cfgs)) != 1:
            why.append("unequal VFY-ONE settings across TP ranks")
        if len(set(fps)) != 1:
            why.append("unequal warm-key sets across TP ranks")
        if len(set(ns)) != 1:
            why.append("serving step counts differ across TP ranks %s" % ns)
        if len(set(clss)) != 1:
            why.append("window classes differ across TP ranks %s" % clss)
        if any(dis):
            why.append("local disable on TP ranks %s" % [r for r, d in enumerate(dis) if d])
        if len(set(close)) != 1:
            why.append("TP ranks disagree on the CHECK window %s" % close)
        elif close[0]:
            self.check_pending = False
            if sum(mism):
                why.append("CHECK: %d mismatches over the TP group (%d on this rank) in the first %d "
                           "checked steps" % (sum(mism), self.check_mismatches, self.check_steps))
            else:
                logger.info("q38 VPS_VFY_ONE CHECK: %d steps, 0 mismatches over the TP group; armed "
                            "(TP rank %d)", self.check_steps, self._rank())
        if why and not self.failed:
            self.failed = True
            logger.error("q38 VPS_VFY_ONE agreement at serving step %d: %s; every rank keeps the D83 chain "
                         "for the rest of the run (TP rank %d)", self.serving, "; ".join(why), self._rank())
        engaged = not self.failed and min(en) == 1
        if engaged != self.engaged or self.boundaries == 1:
            logger.info("q38 VPS_VFY_ONE agreement at serving step %d: vfy %s (enable %d of %d ranks) "
                        "(TP rank %d)", self.serving, "on" if engaged else "off", sum(en), len(en),
                        self._rank())
        self.engaged = engaged

    # -------------------------------------------------------- per-step choice

    def decide(self, runner: Any, single_chunk: bool) -> bool:
        """execute_model, on a VPS chunk with spec metadata: whether this step defers its local logits
        to the VFY path (step mode "vfy", or "check" in the CHECK window). Every input is rank-uniform."""
        self.step_mode = None
        if not self.engaged or self.failed or not single_chunk:
            return False
        if self.static(runner):
            return False
        if getattr(runner, "_vps_checks_left", 0) > 0:
            return False  # the VPS identity check reads the chunk's logits
        if self.check_left > 0:
            self.check_left -= 1
            self.step_mode = "check"
        elif self.check_pending:
            return False  # CHECK done, armed only by the next agreement
        else:
            self.step_mode = "vfy"
        return True

    def warm_check(self, call: tuple, lite: tuple | None, audit: bool) -> bool:
        """Whether the VFY program's key for `call` (the runner's call tuple) was warmed; else the step runs
        the D83 chain (cold). `lite` None: the full key. KEYLITE: the verdict of `lite`'s first call, reused;
        the full key again on CHECK steps (`audit`) and every KEYLITE_AUDIT_EVERY-th call, where an unequal
        full key turns KEYLITE off for the run (see the module docstring)."""
        if lite is None or self.keylite_off:
            return key(*call[1:]) in self.warm
        if self.lite_of[0] is not self.warm or self.lite_of[1] != len(self.warm):
            # the memo holds verdicts of one warm set: a replaced or grown set (warm-up adds keys, serving
            # never does) starts it again
            self.lite.clear()
            self.lite_of = (self.warm, len(self.warm))
        self.lite_calls += 1
        hit = self.lite.get(lite)
        if hit is not None and not audit and self.lite_calls % KEYLITE_AUDIT_EVERY:
            self.stats["keylite_hit"] += 1
            return hit[1]
        full = key(*call[1:])
        if hit is None:
            hit = self.lite[lite] = (full, full in self.warm)
            self.stats["keylite_miss"] += 1
            return hit[1]
        self.stats["keylite_audit"] += 1
        if hit[0] != full:
            self.keylite_off = True
            logger.error("q38 VPS_VFY_ONE KEYLITE: the full key of VFY call %d differs from its lite key's first "
                         "full key; KEYLITE off for the rest of the run (TP rank %d)", self.lite_calls, self._rank())
            return full in self.warm
        return hit[1]

    def note(self, what: str) -> None:
        self.stats[what] += 1
        l6 = _l6_ab()
        if l6 is not None and getattr(l6, "ACTIVE", False):
            l6.note(LEVER, what)
        if what == "vfy" and not self.engaged_logged:
            self.engaged_logged = True
            self.log_counts(first=True)

    def log_counts(self, first: bool = False) -> None:
        s = self.stats
        logger.info(
            "q38 VPS_VFY_ONE%s: mode=%d seed=%s vfy=%d check=%d cold=%d no_ro=%d seed_taken=%d "
            "seed_miss=%d seed_written=%d seed_unused=%d check_mismatches=%d programs=%d agreed=%s failed=%s "
            "boundaries=%d serving=%d keylite=%s hit=%d miss=%d audit=%d off=%s (TP rank %d)",
            " engaged" if first else "", self.mode, self.seed_reason == "" and self.mode == 2,
            s["vfy"], s["check"], s["cold"], s["no_ro"], s["seed_taken"], s["seed_miss"], s["seed_written"],
            s["seed_unused"], self.check_mismatches, self.programs, self.engaged, self.failed,
            self.boundaries, self.serving, self.keylite, s["keylite_hit"], s["keylite_miss"], s["keylite_audit"],
            self.keylite_off, self._rank(),
        )


def from_env(envs: Any) -> VfyOne | None:
    """The runner's constructor: None when VLLM_TPU_VPS_VFY_ONE is 0 / unset."""
    mode = envs.vfy_one_value()
    if not mode:
        return None
    v = VfyOne(mode, int(envs.VLLM_TPU_VPS_VFY_ONE_CHECK), envs.VLLM_TPU_VPS_VFY_ONE_AB,
               bool(envs.VLLM_TPU_VPS_VFY_ONE_KEYLITE))
    logger.info("q38 VPS_VFY_ONE=%d armed: window switch %r, CHECK %d steps, agreement every %s (engine steps "
                "of the runner's agreement clock), KEYLITE %s", mode, v.ab or "", v.check_steps,
                "l6_ab step-window roll" if v.windowed() else "%d serving steps" % AGREE_EVERY,
                "on (full-key audit every %d calls and on CHECK steps)" % KEYLITE_AUDIT_EVERY if v.keylite else "off")
    return v
