# SPDX-License-Identifier: Apache-2.0
"""q38 L4 AGREE: one engine-step clock and one capability gather for the L4 levers whose two paths carry
different TP collectives (AGREE-BUILD, helper of L4-E; reports/kstep_design_v1.md §5 interface ask).

Why. DRAFT-ASM (spec_decode/draft_asm.py) and VFY-ONE (runner/vfy_one.py) each pick between precompiled
programs that carry different TP collectives, so no rank may change path on its own: a path changes only at
an agreement boundary, through one all-rank CPU-group all_gather. Before this module each lever kept its own
clock (DRAFT-ASM counted propose calls, VFY-ONE execute_model calls) and its own gather. Under a K-step burst
propose runs once per inner step, so a per-call clock puts the two levers' boundaries on different engine
steps. Here every lever keys its boundaries on one engine-step clock, and one gather per boundary carries
every lever's vector.

Clock. The runner calls ``Agreement.tick(runner)`` once per ``execute_model`` on every rank, before the
step's forward. The inner steps of a K-step burst do not enter execute_model, so they never tick and never
meet a boundary. Only serving steps count (``runner.warmup_done``). The boundaries are pure functions of the
engine step count and of l6_ab's rank-uniform step window:
  * engine step 1;
  * every step that is a multiple of a lever's ``agree_period()``;
  * when a lever's ``agree_roll()`` is true and core/l6_ab.py runs step windows (VLLM_TPU_L6_AB_STEPS > 0),
    every step at which l6_ab's window class rolls. Wall-clock l6_ab windows never make a boundary: their
    roll can fall on different steps on different ranks.
The periods and the roll flag are read once, at the first serving step.

Gather. At a boundary every rank contributes one vector of CAP int64: a header (MAGIC, the hash of the
registered (name, width) list, the engine step, the boundary index), then every lever's ``local_vector()`` in
name order, zero-padded to CAP. CAP is fixed, so ranks whose lever sets differ still gather tensors of one size
(no size-mismatch hang) and the header tells them apart. Every rank gets the same rows. Unequal headers (a
different lever set or width, or ranks at different engine steps, i.e. a gather paired across steps) raise on
every rank: the agreement could not be made, so the run stops rather than letting the ranks run different
programs. A failed all_gather raises too (fail-stop). Otherwise each lever's ``apply`` gets every rank's slice
of its own entries, in rank order, and decides exactly as it did with a gather of its own (its rules: the MIN
of the wishes; any local disable, unequal fingerprint or CHECK mismatch keeps every rank on the base path for
the rest of the run). Cost: one CPU all_gather of CAP int64 per rank per boundary, never per step.

Joining (the registration hook). A lever is any object with
  * ``AGREE_NAME`` (str, unique per runner) and ``AGREE_WIDTH`` (int, the length of its vector);
  * ``local_vector() -> list[int]``: exactly AGREE_WIDTH int64 values (wish, local disable, warm-key
    fingerprint, CHECK state, ...);
  * ``apply(rows)``: rows[r] is rank r's local_vector; the decision must depend on the rows alone;
  * ``agree_period() -> int``: engine steps between its boundaries (0 = none besides step 1 and rolls);
  * optionally ``agree_roll() -> bool``: also a boundary at every l6_ab step-window roll.
``register(runner, lever)`` adds it to the runner's Agreement (created on first use as ``runner._l4_agree``)
before the first serving step and sets ``lever.agreement``, whose ``step`` is the engine step count. A lever's
rank-local disables, window wishes and CHECK counts reach the other ranks only through its vector at a
boundary; between boundaries a lever may change path only on rank-uniform facts (the scheduler output and
agreed-equal warm keys).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Callable

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Header word 0 of every rank's vector ("Q8AG"): a gather paired with some other CPU-group all_gather of the
# same size reads as a header mismatch.
MAGIC = 0x51384147
# Header words: MAGIC, schema hash, engine step, boundary index.
HEADER = 4
# int64 entries per rank per gather (header + every lever's vector + zero padding), fixed.
CAP = 64


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - no TP group (tests, single chip)
        return 0


def gather(vec: list) -> list:
    """Every TP rank's `vec`, in rank order, over the TP group's CPU group (one blocking all_gather);
    ``[vec]`` when the process has no TP group. A failure raises: the levers' paths carry their collectives in
    different programs, so a rank that cannot agree must stop, not diverge."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        g = get_tp_group()
        world = int(getattr(g, "world_size", 1))
    except Exception:  # noqa: BLE001 - no TP group (tests, single chip)
        return [list(vec)]
    if world <= 1:
        return [list(vec)]
    t = torch.tensor([int(v) for v in vec], dtype=torch.int64)
    out = [torch.zeros_like(t) for _ in range(world)]
    try:
        torch.distributed.all_gather(out, t, group=g.cpu_group)
    except Exception as exc:
        raise RuntimeError(
            "q38 L4 AGREE: the TP ranks could not agree on the lever paths (CPU-group all_gather failed); "
            "stopping rather than letting the ranks run different programs"
        ) from exc
    return [o.tolist() for o in out]


def _import_l6_ab():
    """core/l6_ab.py, or None on a tree without it."""
    try:
        from vllm_torchtpu.core import l6_ab
    except ImportError:  # a tree without core/l6_ab.py
        return None
    return l6_ab


@dataclasses.dataclass
class _Slot:
    name: str
    width: int
    lever: Any


def _schema(slots: list) -> int:
    raw = repr([(s.name, s.width) for s in slots]).encode()
    return int(hashlib.sha1(raw).hexdigest()[:15], 16)


class Agreement:
    """One runner's engine-step clock and its registered levers (see the module docstring)."""

    def __init__(self) -> None:
        self.slots: list[_Slot] = []
        self.schema = _schema([])
        self.step = 0
        self.boundaries = 0
        self.gathers = 0
        self.periods: tuple | None = None
        self.roll = False
        self.last_cls: str | None = None
        # A stand-in all_gather (tests); None = this module's gather, looked up at call time.
        self.transport: Callable | None = None
        self._l6: list = []
        self.rank: int | None = None

    # -------------------------------------------------------- registration

    def add(self, lever: Any) -> None:
        """Register `lever` (the registration hook; see the module docstring)."""
        name, width = str(lever.AGREE_NAME), int(lever.AGREE_WIDTH)
        if self.step:
            raise RuntimeError(
                "q38 L4 AGREE: lever %r registered at engine step %d; levers join before the first serving step"
                % (name, self.step))
        if not name or any(s.name == name for s in self.slots):
            raise ValueError("q38 L4 AGREE: lever name %r is empty or already registered" % name)
        used = sum(s.width for s in self.slots)
        if width < 1 or HEADER + used + width > CAP:
            raise ValueError("q38 L4 AGREE: lever %r needs %d entries; %d of %d are free"
                             % (name, width, CAP - HEADER - used, CAP))
        self.slots.append(_Slot(name, width, lever))
        self.slots.sort(key=lambda s: s.name)
        self.schema = _schema(self.slots)
        lever.agreement = self

    # -------------------------------------------------------- the clock

    def _rank(self) -> int:
        if self.rank is None:
            self.rank = _tp_rank()
        return self.rank

    def l6(self):
        """core/l6_ab.py (looked up once per Agreement)."""
        if not self._l6:
            self._l6.append(_import_l6_ab())
        return self._l6[0]

    def window_class(self) -> str | None:
        """l6_ab's window class when it runs step windows (a function of its step count, so rank-uniform),
        else None (no l6_ab, or wall-clock windows)."""
        l6 = self.l6()
        if l6 is None or int(getattr(l6, "STEPS", 0) or 0) <= 0:
            return None
        fn = getattr(l6, "current_class", None)
        return None if fn is None else fn()

    def _freeze(self) -> None:
        self.periods = tuple(max(0, int(s.lever.agree_period())) for s in self.slots)
        self.roll = any(bool(getattr(s.lever, "agree_roll", lambda: False)()) for s in self.slots)
        every = sorted({p for p in self.periods if p > 0})
        logger.info(
            "q38 L4 AGREE: levers %s on one engine-step clock; boundaries at step 1%s%s; one CPU-group all_gather "
            "of %d int64 per rank per boundary (TP rank %d)",
            "+".join("%s(%d)" % (s.name, s.width) for s in self.slots),
            "".join(", every %d steps" % p for p in every),
            ", every l6_ab step-window roll" if self.roll else "", CAP, self._rank(),
        )

    def tick(self, runner: Any) -> bool:
        """Once per execute_model on every rank: count the serving engine step and, at a boundary, agree.
        Returns whether this step was a boundary."""
        if not getattr(runner, "warmup_done", False):
            return False
        self.step += 1
        if self.periods is None:
            self._freeze()
        n = self.step
        boundary = n == 1 or any(p > 0 and n % p == 0 for p in self.periods)
        if self.roll:
            cls = self.window_class()
            if cls != self.last_cls:
                self.last_cls = cls
                boundary = True
        if boundary:
            self._agree()
        return boundary

    # -------------------------------------------------------- the gather

    def local_vector(self) -> list:
        """This rank's CAP entries: the header, then every lever's vector in name order, zero-padded."""
        out = [MAGIC, self.schema, self.step, self.boundaries]
        for s in self.slots:
            vec = [int(v) for v in s.lever.local_vector()]
            if len(vec) != s.width:
                raise RuntimeError("q38 L4 AGREE: lever %r gave %d entries, registered %d"
                                   % (s.name, len(vec), s.width))
            out.extend(vec)
        return out + [0] * (CAP - len(out))

    def _agree(self) -> None:
        vec = self.local_vector()
        send = self.transport if self.transport is not None else gather
        rows = [list(r) for r in send(vec)]
        self.gathers += 1
        self.boundaries += 1
        if any(len(r) != CAP for r in rows):
            raise RuntimeError("q38 L4 AGREE at engine step %d: gathered rows of %s entries, expected %d"
                               % (self.step, sorted({len(r) for r in rows}), CAP))
        heads = [tuple(r[:HEADER]) for r in rows]
        if len(set(heads)) != 1:
            raise RuntimeError(
                "q38 L4 AGREE at engine step %d: the TP ranks' agreement headers differ %s (lever set, widths or "
                "engine step); stopping rather than letting the ranks run different programs (TP rank %d)"
                % (self.step, heads, self._rank()))
        off = HEADER
        for s in self.slots:
            s.lever.apply([r[off:off + s.width] for r in rows])
            off += s.width

    def counts(self) -> dict:
        return {"step": self.step, "boundaries": self.boundaries, "gathers": self.gathers,
                "levers": [s.name for s in self.slots], "periods": self.periods, "roll": self.roll}


def for_runner(runner: Any) -> Agreement:
    """The runner's Agreement (``runner._l4_agree``), created on first use. Read from the instance dict, as the
    runner's execute_model hook reads it (a class attribute or a mock's auto-attribute is never an Agreement)."""
    agree = runner.__dict__.get("_l4_agree")
    if agree is None:
        agree = Agreement()
        runner._l4_agree = agree
    return agree


def register(runner: Any, lever: Any) -> Agreement:
    """Join `lever` to the runner's engine-step agreement (the registration hook)."""
    agree = for_runner(runner)
    agree.add(lever)
    return agree
