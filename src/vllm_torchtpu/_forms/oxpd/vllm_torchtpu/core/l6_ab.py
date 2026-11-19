# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""q38 L6-HOST: step-time switches and in-process A/B windows for host levers.

Every L6 host lever is read here once per worker step, never at trace time,
so turning a lever on or off changes no compiled program and an arm that
sets it reuses the compile cache of the arm that does not.

Lever flags (one env var per lever, see ``LEVERS``):

* unset, ``""`` or ``"0"``: off;
* ``"1"``: on for every step;
* ``"ab:<c>[,<c>...]"``: on only in wall-clock windows whose class is one of
  the listed classes (needs ``VLLM_TPU_L6_AB_PERIOD_MS`` > 0).

Windows: ``k = time_ns // (VLLM_TPU_L6_AB_PERIOD_MS ms)``, class
``k mod VLLM_TPU_L6_AB_CLASSES`` (default 2). Or, with
``VLLM_TPU_L6_AB_STEPS=S`` (> 0, exclusive with the period), by step count:
``k = j // S`` for the process's j-th step (j counts from 1, like
runner/host_phases.py). With S equal to ``VLLM_TPU_HOST_PHASES`` every
HOSTPHASES window (the line with ``steps=T``) lies in exactly one L6 window,
``k = (T - 1) // S``, and every rank, stepping in lockstep, is in the same
class on the same step. With a period that divides
86,400,000 ms into a multiple of the class count (for example 20000 ms and 2
or 4 classes), the class is fixed by the time of day, so every rank and the
engine core agree on it without communicating. Example 2x2 factorial over two
levers: ``CLASSES=4``, lever A ``ab:1,3``, lever B ``ab:2,3``.

Lockstep (step-count windows with more than one rank). Host-only levers
cannot desynchronise device collectives, but a rank whose step count drifted
(a skipped or extra step) would read the wrong class. So, following L5's SAB
protocol (``megakernel_runtime/served_ab.py``, q38 L5, credited): at step
``k*S - D`` (``D = VLLM_TPU_L6_AB_CONFIRM``, default 16) each rank writes a
fingerprint of that step's SchedulerOutput (request ids in order and their
scheduled token counts, identical on every rank at the same step) to
``<dir>/w<k>_r<rank>`` (``dir = VLLM_TPU_L6_AB_DIR`` or
``/dev/shm/q38_l6ab_<ppid>``; the decode's TP workers share a host and a
parent). At step ``k*S - 1`` each rank reads all ``world`` entries of window
k; window k runs its classes only if every entry exists and all are equal.
Otherwise every ``ab:`` lever is off for the rest of the run (uniform ``1``
levers are unaffected) and the rank logs one ERROR. The decision is made
before the window starts, so a HOSTPHASES window never mixes classes.
Window 0 is never confirmed and runs with ``ab:`` levers off.

Composition with other in-run A/Bs. D1e's ``VLLM_TPU_HOST_AB_PERIOD_MS``
toggles the GIL / GC knobs, which move every host phase, so next to it every
``ab:`` lever stays off (uniform ``"1"`` levers are unaffected) and one ERROR
is logged. The D1-norm windows (``VLLM_TPU_D1N_AB_PERIOD_MS``) and L5's SAB
(``VLLM_TPU_SERVED_AB=whole:P[:D]``) may run in the same decode as extra
factorial dimensions: every one of their windows must hold equal shares of
each L6 class, so with SAB set the L6 class cycle ``S * CLASSES`` must divide
SAB's period P (for example S = 200 and 4 classes against P = 4000), else
the ``ab:`` levers stay off with one ERROR. Use step windows (not the
wall-clock period) next to SAB.

Batch-size gates (q38 L6-GATE, ``GATES``): ``VLLM_TPU_L6_CARRY_MIN_REQS=N`` /
``VLLM_TPU_L6_STEADY_MIN_REQS=N`` (> 0) turn that lever off for every worker
step whose scheduled batch holds fewer than N requests, whatever its window
says, and count the step as ``carry.why_minreqs`` / ``steady.minreqs``. The
window line's ``on=`` still names the window's levers.

S3D load gate (q38 L1-S3DGATE, HUNT-11 §5.1 form b; ``VLLM_TPU_S3_DEFER_GATE_MIN_REQS=T``, > 0): lever
s3defer, set as ``ab:<classes>`` with step windows, runs its classes in window k only if window k-1's mean
scheduled-request count is at least T; every other window runs it off. Inside the eligible windows the class
pattern is unchanged, so its off classes stay an in-run off share that is not confounded with batch size. The
mean is over window k-1's steps before its lockstep write step (``j mod S < S - D``: S - D steps, the counts that
gate_step has recorded when the decision is taken and the fingerprint of window k is written). The decision is folded into that
fingerprint, so window k runs its ``ab:`` levers only if every rank reached the same decision from the same counts
(else the lockstep disables them all, as for any rank out of step). Inputs are rank-identical only: the scheduled
request count (``gate_step``) and the step index; never the clock. Each worker logs one line per window when the
window starts:

``S3DEFER gate rank<r>: window=<k> class=<c> win_eligible=<0|1> win_on=<0|1> mean_reqs_prev=<m|nan> n_prev=<n> T=<T>
windows=<n> eligible=<e> on=<o> share_on=<o/n> share_on_eligible=<o/e|nan>`` (one line; counts since the process
started). ``win_eligible`` = the load test passed and the window runs its ``ab:`` classes (confirmed, not refused).
Unset or 0: nothing here runs and every line, file and state is as before.

Each process logs one line per window when the next window starts (and, with
no windows, one line every 20 s), so a reader can join HOSTPHASES windows to
lever states by timestamp:

``q38 L6AB role=<r> window=<k> class=<c> steps=<n> on=<lever+lever|-> counts=<lever.key=n,...|->``
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Callable

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# lever name -> env var. A lever joins here, in envs.py and in the compile-key
# ignore list (compilation/tpu_compiler.py) at the same time.
LEVERS: dict[str, str] = {
    "envcache": "VLLM_TPU_L6_ENVS_CACHE",
    "fctx": "VLLM_TPU_L6_FAST_FWD_CTX",
    "gcfreeze": "VLLM_TPU_L6_GC_FREEZE",
    "switch": "VLLM_TPU_L6_SWITCH",
    "mainpin": "VLLM_TPU_L6_MAIN_PIN",
    # core/l6_jitter.py: MAINPIN-2 (main thread on its own core, static exclusion) and SYNC-GC
    # (automatic GC off, lockstep collections).
    "mainpin2": "VLLM_TPU_L6_MAIN_PIN2",
    "syncgc": "VLLM_TPU_L6_SYNC_GC",
    "defer": "VLLM_TPU_L6_DEFER_PREV",
    # q38 L4 VPSHOST (L4-E helper; names cleared with L6): T-2 stage 0 FB-ORDER, stage 1 SAMP-SNAP, T-1(i).
    "sampfb": "VLLM_TPU_L4_SAMP_FB_ORDER",
    "sampsnap": "VLLM_TPU_L4_SAMP_SNAP",
    "statsasync": "VLLM_TPU_L4_VPS_STATS_ASYNC",
    # q38 L4 VFY-ONE's window switch. It picks between precompiled programs with TP
    # collectives, so runner/vfy_one.py reads this lever only as each rank's wish at a
    # window boundary and switches by an all-rank agreement (MIN), never per rank.
    "vfyone": "VLLM_TPU_VPS_VFY_ONE_AB",
    # q38 L4 DRAFT-ASM / DRAFT1-B window switches (spec_decode/draft_asm.py, draft_one.py): like
    # vfyone, read only as each rank's wish at DRAFT-ASM's agreement boundary (one CPU-group
    # all_gather; stage bits ride entry 0), never per rank. v5 serves them uniform.
    "draftasm": "VLLM_TPU_DRAFT_ASM_AB",
    "draftasmcl": "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",
    "draftasmone": "VLLM_TPU_DRAFT_ASM_ONE_AB",
    # runner/l6_steady.py: steady-step fast path (attention metadata reuse,
    # cached reorder / chunk count, upstream reorder no-op skip, adapter tables).
    "steady": "VLLM_TPU_L6_STEADY_STEP",
    "carry": "VLLM_TPU_L6_DECODE_CARRY",
    # runner/adapt_carry.py: H0-1 ADAPT-CARRY (a)-(c) + POSTFWD-MERGE (merged programs precompiled at warm-up).
    "adcarry": "VLLM_TPU_L6_ADAPT_CARRY",
    # Engine core (core/l6_engine_alloc.py): read per schedule() there, not per worker step.
    "engalloc": "VLLM_TPU_L6_ENGINE_ALLOC_FAST",
    # runner/l6_ptrims.py (q38 H0-2 PREP-TRIMS): input-prep host trims, read per worker step.
    "ptrims": "VLLM_TPU_L6_PREP_TRIMS",
    # runner/l6_setupfree.py (q38 T-5(b)): ping-pong metadata sets on STEADY's fast path, read per worker step.
    "setupfree": "VLLM_TPU_L6_SETUP_FREE",
    # layers/core/attention_metadata.py: count-change block-table packs (ADMIT-INC) and batched
    # MultiGroupBlockTable row writes.
    "admitinc": "VLLM_TPU_L6_ADMIT_INC",
    # q38 L6-MK (runner/l6_mk.py, core/l6_fuse_rpc.py): host trims of the MK-type compiled calls and
    # of the per-step RPCs. fuserpc is read by the engine core (per schedule()); the rest per worker step.
    "mkctx": "VLLM_TPU_L6_MK_CTX",
    "mkthin": "VLLM_TPU_L6_MK_THIN_TT",
    "mkdirect": "VLLM_TPU_L6_MK_DIRECT",
    "fuserpc": "VLLM_TPU_L6_FUSE_RPC",
    "e2kvonly": "VLLM_TPU_L6_OUT_KVONLY",
    # runner/l6_carry_lean.py (q38 H2 CARRY-LEAN): CARRY's host check / record from STEADY's verdict.
    "clean": "VLLM_TPU_L6_CARRY_LEAN",
    # runner/l6_prep_short.py (q38 H2 PREP-SHORT): the carried steady step's short _prepare_inputs.
    "pshort": "VLLM_TPU_L6_PREP_SHORT",
    # runner/l6_pmeter.py (q38 L6 P-METER, log-only): the HOSTLAUNCH ring + THREADSTAT, and
    # the DELAY-INJECT classes (one-rank / all-rank GIL-held spin before the forward).
    "pmeter": "VLLM_TPU_L6_PMETER",
    "inject": "VLLM_TPU_L6_INJECT",
    "injectall": "VLLM_TPU_L6_INJECT_ALL",
    # runner/l6_nodefer.py (q38 L6 NODEFER census, log-only): deferred VFY-call arguments.
    "nodefer": "VLLM_TPU_L6_NODEFER_CENSUS",
    # q38 L6-H2 (overflow unit l6_h2_cheap): FENCE-SKIP (runner/tpu_runner.py
    # _record_input_staging_fence), TOKEN-LISTS-ONCE (runner/tpu_runner_async_output.py),
    # CF-2 AOTSKIP-AB (compilation/l6_aotskip.py).
    "fskip": "VLLM_TPU_L6_FENCE_SKIP",
    "toklists": "VLLM_TPU_L6_TOKEN_LISTS",
    "aotskip": "VLLM_TPU_L6_AOT_SKIP",
    # core/l6_gcadmit.py: gc.freeze() after each worker step that admitted requests (GC-ADMIT-FREEZE).
    "gcadmit": "VLLM_TPU_L6_GC_ADMIT_FREEZE",
    # q38 L4 T1-K1FUSE's window switch (spec_decode/k1fuse.py): like draftasmone, read only as each
    # rank's wish at DRAFT-ASM's agreement boundary (its bit rides entry 0), never per rank.
    "k1fuse": "VLLM_TPU_K1FUSE_AB",
    # q38 L6-S3ASYNC (distributed/kv_transfer/stage3_async.py): Stage-3 load coordination on a
    # per-rank coordinator thread. Read once per submit pass on the model-runner thread; a pass whose
    # state is off first applies every outstanding job (same pass on every rank), then runs today's path.
    "s3async": "VLLM_TPU_S3_ASYNC_SUBMIT",
    # q38 L6-S3ASYNC S3-DEFER-AR (distributed/kv_transfer/tpu_connector.py s3d_*, Qwen start_load_kv): the
    # Stage-3 submit pass after the forward dispatch, its admission collective issued early. Read once per
    # step in start_load_kv; never windowed together with s3async.
    "s3defer": "VLLM_TPU_S3_DEFER_AR",
}

# q38 L6-GATE batch-size gates: lever -> env var (an int, default 0 = no gate). With N > 0, a worker step whose
# scheduled batch holds fewer than N requests runs that lever off for that step only, exactly as a step of an off
# window does (the runner's lever-off paths run, and STEADY drops its caches), and counts <lever>.<_GATE_KEY>. Read
# once by _resolve(), like the lever flags. The runner calls gate_step() right after begin_step() only when GATE is
# non-zero (TPUModelRunner._l6_gate_step), so with every gate at 0 nothing here runs.
GATES: dict[str, str] = {
    "carry": "VLLM_TPU_L6_CARRY_MIN_REQS",
    "steady": "VLLM_TPU_L6_STEADY_MIN_REQS",
}
_GATE_KEY = {"carry": "why_minreqs", "steady": "minreqs"}

# q38 L1-S3DGATE: the S3D load gate's env var (an int T, default 0 = no gate) and the lever it gates. Not a per-step
# GATES entry: it decides per window (see the module docstring).
S3D_GATE_ENV = "VLLM_TPU_S3_DEFER_GATE_MIN_REQS"
S3D_GATE_LEVER = "s3defer"

_UNIFORM_LOG_MS = 20000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    v = int(raw)
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


def _parse(name: str, raw: str, classes: int) -> tuple[bool, frozenset[int] | None]:
    """(uniform_on, window classes or None) for one lever flag value."""
    raw = (raw or "").strip().lower()
    if raw in ("", "0", "false", "off"):
        return False, None
    if raw in ("1", "true", "on"):
        return True, None
    if raw.startswith("ab:"):
        cls = frozenset(int(x) for x in raw[3:].replace("+", ",").split(",") if x.strip())
        if not cls or any(c < 0 or c >= classes for c in cls):
            raise ValueError(f"{name}={raw!r}: classes must lie in [0, {classes})")
        return False, cls
    raise ValueError(f"{name}={raw!r}: expected 0, 1 or ab:<c>[,<c>...]")


# Module state, (re)built by _resolve(); tests set os.environ and call it.
PERIOD_MS = 0
STEPS = 0
CLASSES = 2
CONFIRM = 16
CHECK_STEPS = 0
WINDOWED = False
REFUSED = False
ACTIVE = False
MIN_REQS: dict[str, int] = {}  # gated lever -> N (> 0 only)
GATE = 0  # the largest N of any gate, per-step MIN_REQS or the S3D load gate's T (0 = no gate set)
S3D_GATE_T = 0  # q38 L1-S3DGATE: T of the S3D load gate (0 = no gate)
_cfg: dict[str, tuple[bool, frozenset[int] | None]] = {}
_state: dict = {}


def _resolve() -> None:
    global PERIOD_MS, STEPS, CLASSES, CONFIRM, CHECK_STEPS, WINDOWED, REFUSED, ACTIVE, GATE, S3D_GATE_T
    PERIOD_MS = _env_int("VLLM_TPU_L6_AB_PERIOD_MS", 0)
    STEPS = _env_int("VLLM_TPU_L6_AB_STEPS", 0)
    if PERIOD_MS and STEPS:
        raise ValueError("set one of VLLM_TPU_L6_AB_PERIOD_MS and VLLM_TPU_L6_AB_STEPS, not both")
    CLASSES = _env_int("VLLM_TPU_L6_AB_CLASSES", 2)
    if CLASSES < 2:
        raise ValueError("VLLM_TPU_L6_AB_CLASSES must be >= 2")
    _cfg.clear()
    for lever, env in LEVERS.items():
        _cfg[lever] = _parse(env, os.environ.get(env, ""), CLASSES)
    windowed = any(c is not None for _, c in _cfg.values())
    WINDOWED = windowed
    CONFIRM = _env_int("VLLM_TPU_L6_AB_CONFIRM", 16)
    CHECK_STEPS = _env_int("VLLM_TPU_L6_CHECK_STEPS", 0)
    if STEPS and not 1 <= CONFIRM < STEPS:
        raise ValueError("VLLM_TPU_L6_AB_CONFIRM must satisfy 1 <= D < VLLM_TPU_L6_AB_STEPS")
    if windowed and PERIOD_MS == 0 and STEPS == 0:
        raise ValueError("an L6 lever uses ab:<classes> but neither VLLM_TPU_L6_AB_PERIOD_MS nor _STEPS is set")
    REFUSED = windowed and (
        _env_int("VLLM_TPU_HOST_AB_PERIOD_MS", 0) > 0 or not _balanced_with_sab()
    )
    ACTIVE = any(u or c is not None for u, c in _cfg.values())
    MIN_REQS.clear()
    for lever, env in GATES.items():
        n = _env_int(env, 0)
        if n:
            MIN_REQS[lever] = n
    GATE = max(MIN_REQS.values(), default=0)
    S3D_GATE_T = _env_int(S3D_GATE_ENV, 0)
    if S3D_GATE_T:
        if STEPS == 0:
            raise ValueError(f"{S3D_GATE_ENV} needs step windows (VLLM_TPU_L6_AB_STEPS > 0): its inputs must be "
                             "rank-identical, so it never follows a wall-clock period")
        if _cfg[S3D_GATE_LEVER][1] is None:
            raise ValueError(f"{S3D_GATE_ENV} needs {LEVERS[S3D_GATE_LEVER]}=ab:<classes>: the gate keeps the "
                             "class pattern's off share inside the eligible windows")
        GATE = max(GATE, S3D_GATE_T)  # so the runner calls gate_step, which records each step's count
    _state.clear()
    _state.update(win=None, cls=None, steps=0, counts={}, on={}, warned=False, role="worker", j=0,
                  confirmed=None, sync_off=False, rank_world=None, store=None, check_closed=False,
                  seq=0, gate=None)
    if S3D_GATE_T:
        # acc_*: window acc_k's counted steps so far; dec: (k, open, sum, n), the decision for window k, made at
        # window k-1's lockstep write step; windows / eligible / on: this process's window counts.
        _state["g3"] = dict(acc_k=None, acc_sum=0, acc_n=0, dec=None, windows=0, eligible=0, on=0)


def check_allowed() -> bool:
    """Whether an L6 lever's CHECK mode may run on this step. VLLM_TPU_L6_CHECK_STEPS=N (> 0)
    bounds every CHECK mode to the process's first N worker steps (j from 1, as the windows), so
    a served car can verify its levers during warm-up and the start of its first level and time
    them afterwards; 0 leaves each lever's CHECK setting unbounded. One INFO line when the
    window closes."""
    if CHECK_STEPS <= 0:
        return True
    if _state["j"] <= CHECK_STEPS:
        return True
    if not _state["check_closed"]:
        _state["check_closed"] = True
        logger.info("q38 L6 CHECK window closed after %d worker steps (VLLM_TPU_L6_CHECK_STEPS)", CHECK_STEPS)
    return False


def _sab_period() -> int | None:
    """L5's SAB window length in calls, or None when SAB is not armed. Parsed by SAB's own
    ``served_ab.parse`` so the two never disagree; a spec SAB rejects leaves SAB off
    (``served_ab.from_env`` logs its ERROR), so it is None here too."""
    try:
        from vllm_torchtpu.megakernel_runtime import served_ab
    except ImportError:  # a tree without L5's SAB (served_ab.py): SAB cannot be armed
        return None
    try:
        pc = served_ab.parse(os.environ.get("VLLM_TPU_SERVED_AB", ""))
    except ValueError:
        return None
    return None if pc is None else pc[0]


def _balanced_with_sab() -> bool:
    """Every SAB window holds equal shares of each L6 class: step windows whose class
    cycle (STEPS * CLASSES) divides SAB's period."""
    p = _sab_period()
    if p is None:
        return True
    return STEPS > 0 and p % (STEPS * CLASSES) == 0


def _rank_world() -> tuple[int, int]:
    """This worker's rank and the number of ranks that step together (1 = no lockstep check)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:  # noqa: BLE001 - no torch / no group: single rank
        pass
    return 0, 1


def sched_fingerprint(scheduler_output: Any) -> str:
    """A step's SchedulerOutput, identical on every TP rank at the same step: the
    scheduled request ids in order with their token counts, and the total."""
    try:
        items = list(scheduler_output.num_scheduled_tokens.items())
        raw = repr((items, int(scheduler_output.total_num_scheduled_tokens))).encode()
    except Exception as exc:  # noqa: BLE001 - a missing output is itself a fingerprint
        raw = repr(("no-output", type(exc).__name__)).encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def _store() -> str:
    if _state["store"] is None:
        _state["store"] = os.environ.get("VLLM_TPU_L6_AB_DIR") or f"/dev/shm/q38_l6ab_{os.getppid()}"
    return _state["store"]


def _sync_disable(why: str) -> None:
    _state["sync_off"] = True
    logger.error("q38 L6AB rank %d DISABLED (every ab: L6 lever off for the rest of the run): %s",
                 _state["rank_world"][0], why)


def _lockstep(j: int, fp_fn: Callable[[], str] | None) -> None:
    """Write this rank's fingerprint for window k at step k*S - D; confirm window k at step k*S - 1."""
    rank, world = _state["rank_world"]
    if (j + CONFIRM) % STEPS == 0:
        k = (j + CONFIRM) // STEPS
        fp = fp_fn() if fp_fn is not None else "no-fingerprint"
        if S3D_GATE_T:
            # q38 L1-S3DGATE: window k's gate decision and its inputs ride the fingerprint, so window k is
            # confirmed only if every rank decided the same from the same counts.
            fp += "|s3dgate=%s" % ":".join(str(int(x)) for x in _state["g3"]["dec"])
        path = os.path.join(_store(), f"w{k}_r{rank}")
        try:
            os.makedirs(_store(), exist_ok=True)
            with open(path + ".tmp", "w") as f:
                f.write(fp)
            os.replace(path + ".tmp", path)
        except OSError as exc:
            _sync_disable(f"cannot write {path}: {exc}")
    if (j + 1) % STEPS == 0 and not _state["sync_off"]:
        k = (j + 1) // STEPS
        seen = {}
        for r in range(world):
            try:
                with open(os.path.join(_store(), f"w{k}_r{r}")) as f:
                    seen[r] = f.read()
            except OSError:
                seen[r] = None
        if any(v is None for v in seen.values()) or len(set(seen.values())) != 1:
            _sync_disable(f"window {k}: rank fingerprints {seen} (missing or unequal: ranks out of step)")
        else:
            _state["confirmed"] = k


_resolve()


def _now_ns() -> int:
    return time.time_ns()


def _window(now_ns: int) -> tuple[int, str]:
    if STEPS > 0:
        k = _state["j"] // STEPS
        return k, str(k % CLASSES)
    if PERIOD_MS > 0:
        k = now_ns // (PERIOD_MS * 1_000_000)
        return k, str(k % CLASSES)
    return now_ns // (_UNIFORM_LOG_MS * 1_000_000), "uniform"


def _lever_on(lever: str, cls: str, k: int) -> bool:
    uniform, classes = _cfg[lever]
    if uniform:
        return True
    if classes is None or REFUSED or cls == "uniform":
        return False
    if STEPS > 0 and _state["rank_world"][1] > 1 and (_state["sync_off"] or _state["confirmed"] != k):
        return False
    if S3D_GATE_T and lever == S3D_GATE_LEVER and not _gate_open(k):
        return False
    return int(cls) in classes


# ---------------------------------------------------------------- q38 L1-S3DGATE: the S3D load gate (S3D_GATE_T > 0)


def _gate_count(num_reqs: int) -> None:
    """Add this step's scheduled request count to its window's sum. Window k-1's decision for window k is taken at
    its lockstep write step (j = k*S - D) before that step's count is added, so it covers exactly the window's first
    S - D steps (window 0: S - D - 1, as j starts at 1); the later steps' counts are dropped when window k starts."""
    g = _state["g3"]
    k = _state["j"] // STEPS
    if g["acc_k"] != k:
        g["acc_k"], g["acc_sum"], g["acc_n"] = k, 0, 0
    g["acc_sum"] += int(num_reqs)
    g["acc_n"] += 1


def _gate_decide(k: int) -> None:
    """At window k-1's lockstep write step (j = k*S - D), before the write: window k is open if window k-1's
    counted steps hold at least T requests on average (integer test: sum >= T * n, n > 0)."""
    g = _state["g3"]
    s, n = (g["acc_sum"], g["acc_n"]) if g["acc_k"] == k - 1 else (0, 0)
    g["dec"] = (k, n > 0 and s >= S3D_GATE_T * n, s, n)


def _gate_open(k: int) -> bool:
    d = _state["g3"]["dec"]
    return d is not None and d[0] == k and bool(d[1])


def _gate_line(k: int, cls: str) -> None:
    """One line per window when it starts (after its lever states are fixed): the window's decision and the counts."""
    g = _state["g3"]
    d = g["dec"] if (g["dec"] is not None and g["dec"][0] == k) else (k, False, 0, 0)
    rank, world = _state["rank_world"]
    armed = not REFUSED and cls != "uniform" and not (
        STEPS > 0 and world > 1 and (_state["sync_off"] or _state["confirmed"] != k))
    elig = bool(d[1]) and armed
    on_now = bool(_state["on"].get(S3D_GATE_LEVER, False))
    g["windows"] += 1
    g["eligible"] += elig
    g["on"] += on_now
    logger.info(
        "S3DEFER gate rank%d: window=%d class=%s win_eligible=%d win_on=%d mean_reqs_prev=%s n_prev=%d T=%d "
        "windows=%d eligible=%d on=%d share_on=%.3f share_on_eligible=%s",
        rank, k, cls, elig, on_now, ("%.3f" % (d[2] / d[3])) if d[3] else "nan", d[3], S3D_GATE_T,
        g["windows"], g["eligible"], g["on"], g["on"] / g["windows"],
        ("%.3f" % (g["on"] / g["eligible"])) if g["eligible"] else "nan",
    )


def _log_window() -> None:
    # on= is the window's lever states: if the window's last step was gated, the states gate_step replaced.
    g = _state["gate"]
    win_on = g[0] if g is not None and _state["on"] is g[1] else _state["on"]
    on = "+".join(sorted(k for k, v in win_on.items() if v)) or "-"
    counts = ",".join(f"{k}={v}" for k, v in sorted(_state["counts"].items())) or "-"
    logger.info(
        "q38 L6AB role=%s window=%d class=%s steps=%d on=%s counts=%s",
        _state["role"], _state["win"], _state["cls"], _state["steps"], on, counts,
    )


def begin_step(role: str = "worker", fp_fn: Callable[[], str] | None = None) -> None:
    """Roll the window and fix every lever's state for this step. Call once per
    step before any lever is read; `fp_fn` returns this step's batch fingerprint
    (called only on the lockstep write steps). No-op (beyond one branch) when no
    L6 lever is set."""
    if not ACTIVE:
        return
    if _state["rank_world"] is None:
        _state["rank_world"] = _rank_world() if (STEPS > 0 and WINDOWED) else (0, 1)
    if REFUSED and not _state["warned"]:
        _state["warned"] = True
        logger.error(
            "q38 L6AB: L6 windows (period %d ms, steps %d, classes %d) next to D1e's "
            "VLLM_TPU_HOST_AB_PERIOD_MS, or not balanced inside SAB's windows (SAB period %s must be a "
            "multiple of steps x classes); every ab: L6 lever stays off.",
            PERIOD_MS, STEPS, CLASSES, _sab_period(),
        )
    _state["j"] += 1
    if S3D_GATE_T and (_state["j"] + CONFIRM) % STEPS == 0:
        _gate_decide((_state["j"] + CONFIRM) // STEPS)  # before the lockstep write, which carries the decision
    if STEPS > 0 and WINDOWED and not REFUSED and _state["rank_world"][1] > 1 and not _state["sync_off"]:
        _lockstep(_state["j"], fp_fn)
    k, cls = _window(_now_ns())
    if _state["win"] is None:
        _state["win"], _state["cls"], _state["role"] = k, cls, role
    elif k != _state["win"]:
        _log_window()
        _state["win"], _state["cls"], _state["steps"], _state["counts"] = k, cls, 0, {}
    if _state["steps"] == 0 or not _state["on"] or _state["sync_off"]:
        _state["on"] = {lever: _lever_on(lever, cls, k) for lever in _cfg}
    if S3D_GATE_T and _state["steps"] == 0 and role == "worker":
        _gate_line(k, cls)
    _state["steps"] += 1
    _state["seq"] += 1


def gate_step(num_reqs: int) -> tuple[str, ...]:
    """q38 L6-GATE: the batch-size gates (``GATES``) for this step; call right after begin_step, only when GATE is
    non-zero. `num_reqs` is this step's scheduled request count. Each gated lever that is on for this step and whose N
    exceeds `num_reqs` is turned off for this step only and counted (``<lever>.<_GATE_KEY>``); returns those levers.
    The previous step's gated states are undone first, so a gate never outlives its step, and the window line keeps
    reporting the window's states (_log_window). With the S3D load gate set (q38 L1-S3DGATE), it also records
    `num_reqs` for the window decision; that changes no state of this step."""
    if not ACTIVE:
        return ()
    if S3D_GATE_T:
        _gate_count(num_reqs)
    on_now = _state["on"]
    g = _state["gate"]
    if g is not None:
        _state["gate"] = None
        if on_now is g[1]:  # begin_step kept last step's gated states (no new window): the window's states again
            on_now = _state["on"] = g[0]
    off = tuple(lever for lever, n in MIN_REQS.items() if num_reqs < n and on_now.get(lever, False))
    if off:
        gated = dict(on_now)
        for lever in off:
            gated[lever] = False
            note(lever, _GATE_KEY[lever])
        _state["on"] = gated
        _state["gate"] = (on_now, gated)
    return off


def configured(lever: str) -> bool:
    """Whether `lever`'s flag is set to 1 or ab:<classes> (resolved at import / _resolve())."""
    uniform, classes = _cfg.get(lever, (False, None))
    return uniform or classes is not None


def on(lever: str) -> bool:
    """Whether `lever` is on for the current step (False before the first begin_step)."""
    return bool(_state["on"].get(lever, False))


def note(lever: str, key: str, n: int = 1) -> None:
    """Count an engagement event for the current window's log line."""
    if not ACTIVE:
        return
    name = f"{lever}.{key}"
    _state["counts"][name] = _state["counts"].get(name, 0) + n


def current_class() -> str | None:
    return _state["cls"]


def step_seq() -> int:
    """Steps begun since the switches were resolved (never reset by a window
    change; 0 before the first begin_step or with no lever set). A lever that
    keeps state across steps uses it to tell the next step from a later one."""
    return _state["seq"]
