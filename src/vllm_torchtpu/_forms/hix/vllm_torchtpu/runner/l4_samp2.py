# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VRNG-VPS controller (VLLM_TPU_VPS_INGRAPH_RNG; SAMP2-BUILD, helper of L4-E).

The lever (layers/adapter/sample/vps_rng.py) gives the VPS verify a second program, `_vps_verify_rng_compiled`,
that draws its uniforms in-graph from a device int32 counter instead of taking the three eager `torch.rand` arrays.
This module decides, per VPS verify call, which of the two programs runs, and keeps the counter.

SAB dual-path. The base programs stay exactly as they are and warm as they do today; the VRNG program is warmed
beside them over the same buckets (`warm_key` records each warmed shape key). After warm-up a call whose key was never
warmed runs the base path (counted ``cold``), so serving never compiles a VRNG program. A call whose chunk holds a
request with its own SamplingParams.seed (``input_batch.generators``) runs the base path (counted ``seeded``): the
lever never changes what a seeded request draws. NOTE (C): on this tree the base path itself draws every chunk from
the runner's one sampling generator (tpu_runner `_get_sampling_generator`); per-request generators are used only to
rewind partial requests, so a seeded request is not reproducible by its seed on either path.

Window switch, VLLM_TPU_VPS_INGRAPH_RNG_AB (host only, outside the compile key; no ',' in any value):
  * "" or "1": VRNG on every agreed call; "0": base path (VRNG programs warm, never run: a same-compile control);
  * "alt:P": SAB-like windows of P serving ENGINE STEPS (P >= 2); window k = n // P for engine step n of the
    runner's agreement clock (runner/l4_agree.py), VRNG in even windows and the base path in odd ones.
The windows count the runner's serving engine steps (q38 L4 AGREE's one clock; v5 composition, V5COMP-BUILD: before
the port they counted VPS serving calls), which every TP rank makes identically, never l6_ab steps or wall-clock time,
so the clock cannot drift between ranks.

All-rank agreement (lead safety rule; the pattern of spec_decode/draft_asm.py on q38-l4-draftasm, credited). The two
paths draw different uniforms, so a step on which the ranks disagree would sample different tokens on different
ranks. The path changes only at agreement boundaries, the engine steps of the runner's agreement clock
(runner/l4_agree.py, q38 L4 AGREE; registered as lever "vrng", AGREE_WIDTH 4): the first serving engine step, every
window start (agree_period() = P) and every AGREE_EVERY engine steps in the unwindowed modes. At a boundary every rank
contributes (window wish, sticky local disable, warm-key fingerprint, counter value) as its 4 entries of the runner's
one all_gather over the TP group's CPU group (DRAFT-ASM's and VFY-ONE's entries ride the same gather), and every rank
computes the same decision: VRNG until the next boundary iff every rank wishes it, none is disabled and the fingerprints and
counter values are equal. Any disable, unequal fingerprint or unequal counter logs one ERROR and keeps every rank on
the base path for the rest of the run. A gather that fails raises (fail-stop; l4_agree.gather). A VPS verify call
never agrees on its own (begin() only reads the last agreement). Between boundaries only scheduler-output facts
(cold keys, seeded chunks) change the path, identically on every rank.

Counter. It starts at vps_rng.vps_initial_counter(model seed) on every rank and only serving VRNG calls advance it,
by one each (the program returns counter + 1). The host keeps the exact value without reading the device
(counter0 + VRNG calls, mod 2**32); that value is the agreement's counter word. The device word wraps in int32
arithmetic; the counter_uniform hash keys on the 32-bit word, so a run may make at most 2**32 VRNG calls before a
counter value (and so a (counter, element) pair) would repeat: the 2**32-th call and every later one run the base path
(counted ``wrapped``, one line). At 100 VPS calls per second (I) that is about 497 days of serving.

AL canary (in-run). The VRNG program also returns this call's [accepted draft tokens, drafted tokens, spec rows] over
the active requests; the controller accumulates them on device and copies them to the host without blocking
(an event-gated non-blocking copy) every LOG_EVERY_S seconds, with host-side per-path counters (calls, rows, drafted tokens from
the chunk's host draft lengths) for both paths:
  ``q38 VRNG-VPS AL t=<s> vrng calls=<n> rows=<r> drafted=<d> | base calls=<n> rows=<r> drafted=<d> | device(vrng)
  accepted=<a> drafted=<d> rows=<r> al=<1 + a/r> at_call=<n>``
Differencing these against vLLM's spec-decode metrics lines (Accepted / Drafted per interval) splits accepted tokens
by path; the device line gives the VRNG path's acceptance length directly.
"""

from __future__ import annotations

import collections
import hashlib
import time
from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.sample import vps_rng
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

AGREE_EVERY = 200
LOG_EVERY = 5000
LOG_EVERY_S = 10.0
WRAP = 1 << 32
# l6_ab note name: the L6AB window line then counts this process's VRNG calls per window (`vrng.calls=N`).
NOTE = "vrng"


def parse_ab(raw: str) -> tuple[str, int]:
    """('on' | 'off' | 'alt', period) for a VLLM_TPU_VPS_INGRAPH_RNG_AB value; raises on anything else."""
    v = (raw or "").strip().lower()
    if v in ("", "1", "true", "on"):
        return "on", AGREE_EVERY
    if v in ("0", "false", "off"):
        return "off", AGREE_EVERY
    if v.startswith("alt:"):
        p = int(v[4:])
        if p < 2:
            raise ValueError(f"VLLM_TPU_VPS_INGRAPH_RNG_AB={raw!r}: alt:P needs P >= 2")
        return "alt", p
    raise ValueError(f"VLLM_TPU_VPS_INGRAPH_RNG_AB={raw!r}: expected 1, 0 or alt:<P>")


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - no TP group (tests, single chip)
        return 0


def _l6_ab():
    try:
        from vllm_torchtpu.core import l6_ab
    except ImportError:  # a tree without core/l6_ab.py
        return None
    return l6_ab


def _event() -> Any:
    """A device event recorded behind everything enqueued so far (torch_tpu); l4_vpshost's helper, inlined because
    the composed v5 tree has no runner/l4_vpshost.py."""
    ev = torch.tpu.Event()
    ev.record()
    return ev


def seeded_chunk(input_batch: Any, start: int, end: int) -> bool:
    """Whether a request of the chunk [start, end) carries its own SamplingParams.seed (vLLM InputBatch.generators:
    batch index -> torch.Generator, identical on every TP rank)."""
    gens = getattr(input_batch, "generators", None)
    if not gens:
        return False
    return any(start <= int(i) < end for i in gens)


class VrngCtl:
    """Per-runner state of VRNG-VPS: path choice, agreement, warm keys, counter, AL canary."""

    # q38 L4 AGREE (runner/l4_agree.py): this lever's name and entries in the runner's one gather.
    AGREE_NAME = NOTE
    AGREE_WIDTH = 4

    def __init__(self, seed: int, device: Any, ab: str | None = None) -> None:
        self.seed = int(seed or 0)
        self.device = device
        self.counter0 = vps_rng.vps_initial_counter(self.seed)
        self.counter: torch.Tensor | None = None
        self.mode, self.period = parse_ab(envs.VLLM_TPU_VPS_INGRAPH_RNG_AB if ab is None else ab)
        self.warm: set = set()
        self.failed = False
        self.agreed = False
        self.local_disabled: list[str] = []
        self.boundaries = 0
        self.serving_calls = 0
        self.vrng_calls = 0
        self.wrap_logged = False
        self.stats: collections.Counter = collections.Counter()
        self.rank: int | None = None
        self.al_dev: torch.Tensor | None = None
        self.al_pending: collections.deque = collections.deque()
        self.al_last: tuple | None = None
        self.path = {"vrng": [0, 0, 0], "base": [0, 0, 0]}  # calls, rows, drafted tokens (host)
        self.t_log = time.monotonic()
        self.event_fn = None  # device event factory (l4_vpshost._event; tests set a fake)
        # The runner's agreement clock; set by l4_agree.register (its `step` is the engine step).
        self.agreement: Any = None
        logger.info(
            "q38 VRNG-VPS: VLLM_TPU_VPS_INGRAPH_RNG on: VPS verify uniforms drawn in-graph from a device counter "
            "(seed %d, counter0 %d); windows %s period %d; agreement over the TP group",
            self.seed, self.counter0, self.mode, self.period,
        )

    # -------------------------------------------------------- counter

    def counter_word(self) -> int:
        """The counter's exact current value (int32), from the host count of VRNG calls (no device read)."""
        x = (self.counter0 + self.vrng_calls) % WRAP
        return x - WRAP if x >= (1 << 31) else x

    def device_counter(self) -> torch.Tensor:
        if self.counter is None:
            self.counter = torch.tensor(self.counter_word(), dtype=torch.int32).to(self.device)
        return self.counter

    def advanced(self, nxt: torch.Tensor) -> None:
        """Keep the program's returned counter (one VRNG call done)."""
        self.counter = nxt
        self.vrng_calls += 1

    # -------------------------------------------------------- warm keys

    @staticmethod
    def key(local_logits: torch.Tensor, num_bonus: int, num_target: int, accept_shape: tuple, fb_path: bool,
            fallback_rows: int, extra: tuple = ()) -> tuple:
        """The D83-chain VRNG program's shape key; `extra` carries the read-offset operands' shapes (composed
        tree: q38 L4 D83's fold)."""
        return (tuple(local_logits.shape), str(local_logits.dtype), int(num_bonus), int(num_target),
                tuple(accept_shape), bool(fb_path), int(fallback_rows) if fb_path else 0, tuple(extra))

    def warm_key(self, key: tuple) -> None:
        self.warm.add(key)

    def fingerprint(self) -> int:
        raw = repr(sorted(repr(k) for k in self.warm)).encode()
        return int(hashlib.sha1(raw).hexdigest()[:15], 16)

    # -------------------------------------------------------- windows and agreement

    def _rank(self) -> int:
        if self.rank is None:
            self.rank = _tp_rank()
        return self.rank

    def _step(self) -> int:
        """The engine step of the runner's agreement clock (the serving call count when unregistered: tests)."""
        return self.agreement.step if self.agreement is not None else self.serving_calls

    def wish(self) -> bool:
        if self.mode == "on":
            return True
        if self.mode == "off":
            return False
        k = self._step() // self.period
        return k % 2 == 0

    def agree_period(self) -> int:
        """Engine steps between this lever's agreement boundaries (runner/l4_agree.py): the alt:P window length,
        else AGREE_EVERY."""
        return self.period

    def disable_locally(self, why: str) -> None:
        """A rank-local reason to stop; it takes effect at the next boundary, on every rank."""
        if why not in self.local_disabled:
            self.local_disabled.append(why)
            logger.error("q38 VRNG-VPS: local disable on TP rank %d (%s); every rank goes to the base path at the "
                         "next agreement", self._rank(), why)

    def local_vector(self) -> list:
        return [int(self.wish()), int(bool(self.local_disabled)), self.fingerprint(), self.counter_word()]

    def apply(self, rows: list) -> None:
        """The decision every rank computes from the same gathered rows."""
        self.boundaries += 1
        n = self._step()
        wants, disabled, fps, ctrs = (list(c) for c in zip(*rows))
        why = []
        if any(disabled):
            why.append("local disable on TP ranks %s" % [r for r, d in enumerate(disabled) if d])
        if len(set(fps)) != 1:
            why.append("unequal warm-key sets across TP ranks")
        if len(set(ctrs)) != 1:
            why.append("unequal counters across TP ranks %s" % ctrs)
        if why and not self.failed:
            self.failed = True
            logger.error("q38 VRNG-VPS agreement at engine step %d: %s; every rank keeps the base path for the "
                         "rest of the run (TP rank %d)", n, "; ".join(why), self._rank())
        agreed = not self.failed and all(wants)
        if agreed != self.agreed or self.boundaries == 1 or self.mode == "alt":
            logger.info("q38 VRNG-VPS agreement at engine step %d window %d: vrng %s (wishes %d of %d) warm_keys %d "
                        "counter %d (TP rank %d)", n, n // self.period, "on" if agreed else "off", sum(wants),
                        len(wants), len(self.warm), ctrs[0], self._rank())
        self.agreed = agreed

    def begin(self, runner: Any, key: tuple, start: int, end: int) -> bool:
        """Whether this VPS verify call runs the VRNG program (else the base path), from the last agreement of the
        runner's engine-step clock (runner/l4_agree.py; a call never agrees itself). Warm-up and dummy calls (before
        runner.warmup_done) always take the base path."""
        if not getattr(runner, "warmup_done", False):
            return False
        self.serving_calls += 1
        use = False
        if self.failed:
            self.stats["failed"] += 1
        elif not self.agreed:
            self.stats["window_off"] += 1
        elif self.vrng_calls >= WRAP:
            self.stats["wrapped"] += 1
            if not self.wrap_logged:
                self.wrap_logged = True
                logger.info("q38 VRNG-VPS: %d VRNG calls, the counter's 2**32 values are used; base path for the "
                            "rest of the run (TP rank %d)", self.vrng_calls, self._rank())
        elif key not in self.warm:
            self.stats["cold"] += 1
        elif seeded_chunk(getattr(runner, "input_batch", None), start, end):
            self.stats["seeded"] += 1
        else:
            self.stats["vrng"] += 1
            use = True
            l6 = _l6_ab()
            if l6 is not None:
                l6.note(NOTE, "calls")
        if self.serving_calls % LOG_EVERY == 0:
            logger.info("q38 VRNG-VPS counts serving_calls=%d %s counter_calls=%d (TP rank %d)", self.serving_calls,
                        " ".join(f"{k}={v}" for k, v in sorted(self.stats.items())), self.vrng_calls, self._rank())
        return use

    # -------------------------------------------------------- AL canary

    def note_call(self, use: bool, md: Any, num_active: int) -> None:
        """Host per-path counters of one serving call: calls, rows, drafted tokens (host draft lengths)."""
        p = self.path["vrng" if use else "base"]
        p[0] += 1
        p[1] += int(num_active)
        dl = getattr(md, "draft_lengths_cpu", None)
        if dl is not None:
            p[2] += int(sum(int(x) for x in dl[:num_active]))

    def add_al(self, al: torch.Tensor) -> None:
        """Accumulate one VRNG call's device [accepted, drafted, rows] (int32) without reading it."""
        al = al.to(torch.int64)
        self.al_dev = al if self.al_dev is None else self.al_dev + al

    def maybe_log(self, now: float | None = None) -> None:
        """Every LOG_EVERY_S: issue a non-blocking copy of the device AL counts and log the counts whose copy landed
        (the line carries the host per-path counters at the time it is written)."""
        now = time.monotonic() if now is None else now
        while self.al_pending:
            host, ev, at_call = self.al_pending[0]
            if not ev.query():
                break
            self.al_pending.popleft()
            self.al_last = (host.tolist(), at_call)
        if now - self.t_log < LOG_EVERY_S:
            return
        self.t_log = now
        if self.al_dev is not None:
            ev_fn = self.event_fn or _event
            self.al_pending.append((self.al_dev.to("cpu", non_blocking=True), ev_fn(), self.vrng_calls))
        acc, dra, rows = (self.al_last[0] if self.al_last else [0, 0, 0])
        at_call = self.al_last[1] if self.al_last else 0
        v, b = self.path["vrng"], self.path["base"]
        logger.info(
            "q38 VRNG-VPS AL t=%.3f vrng calls=%d rows=%d drafted=%d | base calls=%d rows=%d drafted=%d | "
            "device(vrng) accepted=%d drafted=%d rows=%d al=%.4f at_call=%d",
            now, v[0], v[1], v[2], b[0], b[1], b[2], acc, dra, rows, 1.0 + acc / rows if rows else 0.0, at_call,
        )
