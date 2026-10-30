"""Campaign scorer that measures a SERVED MODEL, not a kernel call.

WHY THIS EXISTS. Every campaign this corpus has run scored a microbenchmark, and
on 2026-09-19 one of them was finally measured end to end: 1.6793x on the scorer,
+0.23% on the server, inside a 2.27% MDE. The mechanism is nameable and it is not
a fluke of that kernel (ledger 29dhk224-01). The scorer times ONE PRE-STAGED
KERNEL CALL. The server also pays for producing the layout that call wants, and
the winning candidate had moved that cost inside the kernel — so the scorer
stopped counting it while the server still paid, and XLA put an `as_strided` back
in front of it. A microbenchmark cannot see work it has hoisted out of its own
timed region.

`kernel_evolve/DESIGN.md` §5.2(d) already required the micro and the end-to-end
numbers to be shown moving together, and §5.12 required the end-to-end pair
before anything left the archive. Both were waived on every campaign here. This
file removes the need for the first by making the second the thing being scored.

WHAT IT COSTS, AND WHY THAT IS AFFORDABLE. A boot is minutes where a kernel call
is microseconds, and DESIGN.md says that gap is what lets a search run for days.
That argument is weaker than it looks HERE: this instance's generations run two
to three hours each and almost all of it is agents thinking, not measuring. A
three-arm job at ~15 minutes an arm fits inside one and changes what the search
is climbing.

THE ARM IS THE TREE THE PROCESS IS RUNNING IN. `perfkit scorer run` starts one
process per arm with that arm's root as cwd and first on PYTHONPATH; the server
this file boots inherits both, which is what makes a candidate's server actually
run the candidate. `origin` reports the file the arm imported so the driver can
refuse an arm that resolved to somebody else's tree — an editable install points
at one fixed tree whatever worktree you meant, and that failure is silent.

    perfkit scorer run --bench $PERFKIT_HOME/scorers/serve_e2e.py \
      --arm "golden=$RECORD_TREE:control" --arm "cand=$WORKTREE" \
      --config <config> --candidate cand

WHAT A CANDIDATE CAN CHANGE, AND WHAT IT MUST NOT. Anything in its own tree, which
is a wider space than a kernel scorer allows and is the second reason this file
exists. In particular XLA FLAGS ARE IN THE TREE: `src/vllm_torchtpu/env_override.py`
assembles `LIBTPU_INIT_ARGS` at import, so editing that file is how a candidate
tunes the compiler, and a kernel microbenchmark that imports the kernel directly
cannot score such a change at all.

What it must NOT change is the serve command, and the mechanism already stops it:
the command is derived from the target rather than written by the arm, and twelve
of its flags are CELL-KEYED (`perfkit.cells.SERVE_FLAGS`): moving one makes a
different cell, so the result would not be comparable to the bar it is scored
against. `--prefill-schedule-interval` is on that list, which is why the +12-15%
that x7kyt6en-01 simulates for psi=64-128 on cell 2f3a09e00c2b is not a win
available to a campaign there -- it is a different operating point, trading TTFT
for throughput, not the same workload served faster.

THAT USED TO BE ASSERTED HERE AND IT WAS FALSE. `Target.retarget` moves `repo`,
and `bench_script` and `configs_dir` hang off `repo`, so the command is derived
from the ARM's `scripts/vllm/benchmarking/`. An arm could edit it. `_target_for`
now re-derives each arm's cell key from the command its own tree would launch and
refuses one that is not the frozen cell -- see `_assert_same_cell`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("PERFKIT_HOME", str(Path.home() / "tpu-perf-forge")))

#: The target used by a config that does not name its own. Its `repo` is
#: retargeted per arm — see `_target_for` — so the arms differ in the tree and in
#: nothing else: same install mechanism, same PYTHONPATH, same env. That is the
#: property `vllm-rpa-recap` was built around for the re-capture and the reason
#: its two boots were comparable.
TARGET = os.environ.get("SERVE_E2E_TARGET", "vllm-torchtpu")

#: Serve points this scorer can score: NONE BUILT IN. A config names a target
#: and a cell, which are one installation's facts, so every config lives in the
#: corpus, in `$PERFKIT_CORPUS/scorers/serve_e2e.toml` (`perfkit/scorer_configs.py`),
#: and `_add_corpus_configs` below reads them. A campaign freezes a subset by
#: name, and each name is a config its target declares: the same string `perfkit
#: bench --config` takes, so the scorer and the ledger measure the same sweep.
#:
#: A CONFIG NAMES ITS OWN TARGET, because a config name only means anything
#: relative to one and the pair is what the cell's bar was set with. A bar held
#: by a target that arms env vars the default target does not, scored against
#: the default, measures a different configuration, and the first time that was
#: tried it failed to boot.
CONFIGS = {}

#: CONFIGS MEASURED SOMEWHERE ELSE, and where: a config the corpus declares with
#: `remote = "buildkite"` or `remote = "cdk"`. A model this host cannot serve is
#: measured on its target's Buildkite dev pipeline (`perfkit/buildkite.py`) or a
#: Cloud DevKit slice (`perfkit/cdk_arm.py`): each arm's tree is built and run
#: there, and scored here from what the job wrote, by the same code a local arm
#: is scored by. `perfkit scorer run` reads the corpus's `remote` without
#: importing this file: a job measured elsewhere takes no hold on this host's
#: TPU, and runs its arms side by side.
REMOTE = {}

#: LAYER RUNS: every full config the corpus declares, served from the model cut to the leading
#: layers that hold every kind of layer it has (`perfkit/layers.py`), and scored
#: on the device's op time per step instead of throughput. Same target, same
#: serve command, same cell: `_assert_same_cell` checks the base config's
#: command, and the cut checkpoint is swapped in after it.
#:
#: THE STEP IS WEIGHTED BY WHAT THE WORKLOAD RUNS. Op time is taken from two
#: phase captures, decode-only steps and steps that carry a prefill chunk, and
#: each counts by its share of the steps in the arm's timed sweep
#: (`_profile_phases`). On a 1k/1k cell nearly every step is decode-only. At
#: 8k/1k most of the device's time is in steps that carry a prefill chunk, which
#: a decode-only score cannot see. The step counts come from the scheduler and
#: the workload, not from the model's depth, so the cut model's stand for the
#: full model's.
#:
#: THE SLOWEST PLANE IS THE STEP. Under data parallelism the engines step
#: together, and one that carries a prefill chunk while the rest decode makes
#: them wait for it: inside a collective on some ranks, between programs on
#: others, by amounts that move from capture to capture. So a phase's op time
#: is the slowest plane's, the critical path, and the mean over planes is
#: reported beside it. On the cut 397B's decode-only steps (one engine of eight
#: still prefilling) four copies of golden read 5.92-5.94 ms on the slowest
#: plane, 0.39% apart, against 5.48-5.60 ms, 2.2% apart, for the mean, which
#: also lost a plane whenever a rank's program fell outside the window.
#:
#: WHAT A LAYER RUN CAN SCORE is device time in the step -- every kernel, the
#: glue around it, the collectives -- through the program the server compiles.
#: Work moved out of one kernel still counts, which is what sank the kernel
#: scorers (see the top of this file). WHAT IT CANNOT: host time between programs, because a
#: short model's step is too short for the host to keep up and its idle time says
#: nothing about the full model's; the scheduler; and wins that come from sharing
#: work across many layers, which one period of the pattern does not repeat.
#: Those score on the full config. A record whose output a layer run changed is
#: served on the full config, with the config's eval tasks (ACCURACY_ENV), in
#: the eval the harness runs before the segment closes on it, and the record is
#: served again by the campaign's final check at its end.
LAYER_SUFFIX = "@layers"
#: Decode steps in each profiled capture of a layer run, against the three or
#: four of a job's ordinary capture.
#:
#: OP TIME, NOT BUSY TIME, is what a layer run scores. A cut model's step is too
#: short for the host to keep up, so the ranks launch their programs at
#: different times and wait for each other inside them. On the 35B's four-layer
#: cut more than half of the programs' time was that waiting, and it moved by a
#: factor of two between captures of one boot, while the op time held to a
#: fraction of a percent across captures and boots. The host tracer does not
#: touch op time either. So one capture per arm is the default: every record is
#: already two jobs on two boots.
LAYER_STEPS = 20
#: The phase captures that stand for a step carrying a prefill chunk, in the
#: order a layer run takes them. The phased profiler names a step by its share
#: of prefill tokens. The steady step of an 8k/1k sweep is a chunk filling the
#: token budget beside the running decodes, which is `prefill_heavy` or, with
#: many decodes, `ambiguous`. `prefill_only` comes last because it is mostly
#: the ramp, one prompt with no decode load.
PREFILL_PHASES = ("prefill_heavy", "ambiguous", "balanced", "decode_heavy", "prefill_only")
#: Where a layer run's phase captures begin: two of the phased profiler's own
#: settings, keys of vllm-torchtpu's `additional_config` (`_phase_settings`).
#:
#: The profiler records LAYER_STEPS + 1 consecutive steps from the first step of
#: a phase it has not captured yet, and files all of them under that phase. Left
#: alone, it captured the first prompt's prefill (steps 1-21) and then whatever
#: step 22 was. That was a race. Step 22 is decode-only if the second request
#: has not reached the engine yet, and a chunk of that request if it has. When
#: it was decode-only, the "decode-only" capture held one decode step and twenty
#: chunk steps, and the chunk capture began at step 43. About one Buildkite arm
#: in eight went the other way and was scored on steps 22-42 against a golden
#: scored on 43-63, which carry more requests: GLM-5.2's c5 read 1.047 on one
#: window and 0.995 on the other.
#:
#: These settings capture no prefill-only steps, which come last in
#: PREFILL_PHASES anyway. They also hold the decode-only capture back until the
#: shortest context is a quarter of the output past the prompt, which no step
#: of the ramp reaches. Nothing is captured while the requests arrive. The chunk
#: capture then begins at the second request's first chunk, and every arm
#: schedules the steps after it alike. The decode-only steps come from steady
#: decoding.
PREFILL_ONLY_FROM = "phased_profiler_prefill_only_kv_len_threshold"
DECODE_ONLY_FROM = "phased_profiler_decode_only_kv_len_threshold"


def _cells_settings(meta: dict) -> dict:
    """The serve command's `--additional-config` for a cells arm: the profiler
    captures decode-only steps from the first one after the driver arms it, and
    never a prefill-only step, which no cell serves."""
    found = re.search(r"--additional[-_]config[= ](\S+)", meta.get("serve_cmd") or "")
    try:
        settings = json.loads(found.group(1).strip("'")) if found else {}
    except ValueError:
        raise RuntimeError(f"{meta.get('config')}: its serve command's --additional-config "
                           f"{found.group(1)!r} is not a JSON object the profiler's settings "
                           f"can be added to") from None
    settings.update({PREFILL_ONLY_FROM: 1 << 30, DECODE_ONLY_FROM: 0})
    return settings


def _phase_settings(meta: dict, score_phase=None) -> dict:
    """The serve command's `--additional-config` with the two settings that
    place a layer run's captures (PREFILL_ONLY_FROM) added to it: a second
    `--additional-config` replaces the first rather than adding to it.

    A config that scores the prefill-carrying steps alone captures no
    decode-only steps at all: nothing reads them, and each capture stalls the
    engine while its trace is written."""
    first = (str(meta.get("isl_osl") or "").split() or [""])[0]
    isl, _, osl = first.partition(":")
    if not (isl.isdigit() and osl.isdigit()):
        raise RuntimeError(f"{meta.get('config')}: its sweep names no input and output length "
                           f"({first!r}), so the decode-only capture cannot be placed")
    found = re.search(r"--additional[-_]config[= ](\S+)", meta.get("serve_cmd") or "")
    try:
        settings = json.loads(found.group(1).strip("'")) if found else {}
        settings.update({PREFILL_ONLY_FROM: 1 << 30,
                         DECODE_ONLY_FROM: (1 << 30) if score_phase == "prefill"
                         else int(isl) + int(osl) // 4})
    except (ValueError, AttributeError):
        raise RuntimeError(f"{meta.get('config')}: its serve command's --additional-config "
                           f"{found.group(1)!r} is not a JSON object the profiler's settings "
                           f"can be added to") from None
    return settings


def _scored_phases(score_phase) -> tuple:
    """The phase captures a config's scoring formula reads (`_phase_score`). A
    profiled sweep stops the profiler once every rank has written them, so the
    rest of the sweep runs without the stall each further capture costs."""
    if score_phase == "prefill":
        return ("prefill_heavy",)
    if score_phase == "decode":
        return ("decode_only",)
    return ("decode_only", "prefill_heavy")


def _layer_variant(cfg: dict) -> dict:
    """The layer run of a full config: same target, serve command and cell.
    A `decode_share` the config declares is the full model's step mix, which its
    layer runs weigh their phases by instead of their own (`_score_phases`)."""
    return {"config": cfg["config"], "target": cfg["target"], "cell": cfg["cell"],
            "layers": "auto", "repeats": 1, "warmup": 0,
            **{k: cfg[k] for k in ("ack_degraded_baseline", "score_phase", "weights", "cells",
                                   "cells_source", "decode_share")
               if cfg.get(k)}}


#: THE CUT ONE PERIOD LONGER, which a campaign does not score. Scoping serves
#: golden at both cuts, and what the longer one adds to each op is that op's
#: time per period of layers: `perfkit layers budget` projects the cell's
#: blocks and census to the full model from the two (`perfkit/layer_budget.py`).
LONG_LAYER_SUFFIX = "@layers2"


def _long_layer_variant(cfg: dict) -> dict:
    return {**_layer_variant(cfg), "periods": 2}


#: How long to wait for a server. The 397B takes 21-32 minutes and the 35B about
#: thirteen, so a boot that has not answered in 2400 s has failed rather than
#: been slow, and the arm stops there instead of at the driver's arm timeout.
READY_TIMEOUT_S = 2400
# The output capture is 48 prompts one at a time; a few minutes when the
# server is healthy. A bound, so a capture that hangs costs this and not the
# rest of the driver's arm timeout.
NUMERICS_TIMEOUT_S = 1200
#: The prompt suite the serial capture with logprobs adds, each prompt cut to
#: the most context the server takes (perfkit.numerics.long_items): what reaches
#: a kernel whose work depends on the context length, which the default suite's
#: prompts do not. Its first tokens are a prompt set of their own in the
#: harness's first-token check.
LONG_SUITE = "long-context"

#: The metric every ratio is taken on. Higher is better, so `measure` returns its
#: reciprocal: the harness computes `control / arm` and expects a latency.
METRIC = "output_token_throughput"

#: Sweeps run and thrown away before the timed ones, per arm per config.
#:
#: A FRESH SERVER IS NOT AT ITS STEADY STATE. The first A/A of this scorer
#: (2026-09-19, 0.6B) had both arms rising monotonically across their three
#: sweeps -- 3039.5, 3089.7, 3116.9 and 2944.0, 2985.9, 3005.5 -- so the first
#: sweep is measuring something that is still settling, and including it pulls
#: an arm's median down by whatever fraction of the boot it caught. That A/A
#: came out 3.4% apart on two byte-identical trees, against a within-arm spread
#: of about 1.3%, and a warmup that lands unevenly across arms is one of the two
#: candidate explanations. Discarding it costs one sweep per arm.
#:
#: WHAT IT MEASURED AFTERWARDS: an A/A/A of three identical arms on the same
#: config came out 1.0000 / 1.0066 / 1.0011 -- 0.66% end to end, against 3.4%
#: before. So the spread a serve-scored campaign on this config has to clear is
#: under a percent, and not the three the first run suggested.
#:
#: ONE THING THAT RUN COULD NOT SEE. All three of its arms booted warm, 167 s
#: each, off a shared compile cache. The first A/A's slow arm was the one that
#: booted COLD, at 262 s, and a campaign has that asymmetry in every job: the
#: control's tree is the same one every time and each candidate's is new. If the
#: cold boot is what the first A/A caught, one discarded sweep is the mitigation
#: and a second may be needed. An A/A/A whose arms start from a tree this host
#: has never served is the experiment that says.
WARMUP_SWEEPS = 1
#: Timed sweeps a full config's arm runs on its one server when the config
#: sets no `repeats`, on every backend. A candidate's score is their lower
#: median and golden's their median (`_timed`).
REPEATS = 3
#: Timing samples a record needs (the harness's campaign.MinRecordMeasurements).
#: A full config's arm runs at least this many timed sweeps, whatever its
#: `repeats`, so that one job can give them.
RECORD_SAMPLES = 2


def say(msg: str) -> None:
    # stderr: `scorer_arm` redirects fd 1 for the whole measurement precisely
    # because the serving stack writes to stdout at import, and one stray line
    # there voids the job.
    print(f"[{time.strftime('%H:%M:%S')}] serve_e2e: {msg}", file=sys.stderr, flush=True)


#: CONFIGS THE CORPUS DECLARES (`perfkit/scorer_configs.py`), in
#: `$PERFKIT_CORPUS/scorers/serve_e2e.toml`. A new model's configs go there, beside
#: its targets, rather than in this file: a scoping run that needed one forked
#: this file, and the fork stopped receiving forge's fixes. A name this file
#: defines stays this file's. A full config declared there gets its `@layers` variant
#: like the ones above; one with `remote` is measured where that says, and
#: `perfkit scorer run` reads the same file to know it.
CORPUS_CONFIGS = "serve_e2e"


def _add_corpus_configs() -> None:
    from perfkit import scorer_configs

    declared = scorer_configs.load(CORPUS_CONFIGS)
    for name, cfg in declared.items():
        if name in CONFIGS:
            say(f"{scorer_configs.path(CORPUS_CONFIGS)} declares {name}, which this file "
                f"already defines; this file's stands")
            continue
        cfg = dict(cfg)
        where = cfg.pop("remote", None)
        CONFIGS[name] = cfg
        if where:
            REMOTE[name] = where
        if LAYER_SUFFIX in name:
            continue
        for variant, make in ((name + LAYER_SUFFIX, _layer_variant),
                              (name + LONG_LAYER_SUFFIX, _long_layer_variant)):
            if variant not in CONFIGS and variant not in declared:
                CONFIGS[variant] = make(cfg)
                if where:
                    REMOTE[variant] = where


_add_corpus_configs()


class _Handle:
    def __init__(self, root: Path, tree: str):
        self.root = root
        self.tree = tree
        self.pkg = None
        # An arm measured somewhere else (REMOTE) imports nothing here: the
        # serving stack's import reaches this host's TPU, which another job may
        # be using. `perfkit scorer run` says so in PERFKIT_REMOTE_ARM.
        if os.environ.get("PERFKIT_REMOTE_ARM"):
            return
        # The module the arm actually imported, for `origin`. Imported here
        # rather than at module scope so the failure is attributed to setup.
        import vllm_torchtpu

        self.pkg = vllm_torchtpu


def setup():
    """Record which tree this arm is. The target is per-config; see `_target_for`.

    The tree of the working copy, uncommitted changes included, as the
    harness knows the arm (`perfkit.scorer.git_tree`), not HEAD's. An arm
    whose change is not committed has HEAD's tree in common with the control,
    and `binary_hash` and `dispatched` are made from this: they then named the
    control's code, and the job read as one arm loading the other's."""
    from perfkit import scorer as pk_scorer

    root = Path.cwd().resolve()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
                          capture_output=True, text=True).stdout.strip()
    tree = pk_scorer.git_tree(root) or head
    say(f"arm root {root} tree {tree or '(not a git tree)'}"
        + (f" (HEAD's is {head}; the difference is uncommitted)" if head and tree != head else ""))
    return _Handle(root, tree)


def _arm_dir(arm: str, name: str) -> Path:
    """Where this arm keeps what it measured on config `name`: one directory
    per job, arm and config. Keyed on the job too, because a run's artifact
    directory outlives one job: an agent's second job with the same arm names
    emptied the first one's directory, its profiles with it."""
    base = Path(os.environ.get("AMPLIO_ARTIFACT_DIR", "/tmp")) / "serve_e2e"
    job = re.sub(r"[^A-Za-z0-9._-]+", "-", os.environ.get("PERFKIT_JOB_ID", "")).strip("-.")
    return (base / job if job else base) / f"{arm}-{name}"


def _acks(cfg: dict) -> list:
    """The bench's acknowledgement of a degraded baseline, when the config
    carries one (`ack_degraded_baseline`). Every bench this scorer runs for that
    config passes it."""
    why = cfg.get("ack_degraded_baseline")
    return ["--ack-degraded-baseline", why] if why else []


def _target_for(cfg, root: Path):
    """The config's target, serving THIS arm's tree, checked.

    Per config rather than per scorer: a config name only means something
    relative to a target, and the pair is what the cell's bar was set with.
    """
    from perfkit import targets

    t = targets.load(cfg.get("target") or TARGET)
    # The one thing that differs between arms. Everything else — adapter, conda
    # env, port, the serve command the config declares — comes from the target,
    # so two arms differ in the tree and in nothing else.
    #
    # `retarget`, not `t.repo = root`. A target that declares its own PYTHONPATH
    # puts its original tree back on the path after the arm has put itself
    # there, and then every arm serves the same code while reporting a different
    # tree. See Target.retarget.
    t.retarget(root)
    # EVERY ARM COMPILES FROM SCRATCH. Two reasons, and either alone is enough.
    #
    # 1. THE 35B CANNOT BOOT FROM ITS OWN CACHE ON THIS HOST. Measured
    #    2026-09-19: cell 2f3a09e00c2b's config compiles cold and serves at
    #    5393.8 tok/s, and every later boot dies loading `tt_jit_copy__as_strided`
    #    -- "attempting to reserve 68.00G at the bottom of memory, there are
    #    66.92G free". Reproduced three times on an idle device with nothing else
    #    holding HBM. With the cache off it boots in 467 s against the same cache
    #    on disk. Without this a serve-scored campaign there dies on its second
    #    arm, every job.
    #
    # 2. IT REMOVES A SYSTEMATIC BIAS. The control's tree is the same one every
    #    job and each candidate's is new, so with a cache the control boots warm
    #    and candidates boot cold -- the 0.6B's first A/A had its cold arm 3.4%
    #    slow against a 0.66% A/A/A where all three booted warm. Paying the
    #    compile every time makes the arms symmetric, which is the property the
    #    whole comparison rests on.
    #
    # It costs a boot: 467 s against 170 s on the 35B. An arm is fifteen minutes
    # either way and the alternative is a measurement nobody can trust.
    #
    # A LAYER ARM KEEPS THE CACHE, and neither reason above applies to it. Op
    # time is a property of the program, not of how long it took to compile, so
    # a warm control and a cold candidate are scored on the same terms; and the
    # cut 397B booted from its own cache in 78 s against 313 s cold, with the two
    # arms' op times within their usual spread. SERVE_E2E_LAYER_COMPILE_CACHE=0
    # turns it off.
    #
    # WHAT THE CACHE KEY DOES NOT SEE. The tree's compiler keys an executable on
    # vLLM's traced files and on the source under its own list of custom-op
    # paths (`_RUNTIME_CACHE_KEY_PATHS` in compilation/tpu_compiler.py), the
    # kernels among them. An edit behind a custom op outside that list would
    # load golden's executable: the candidate scores as golden, a null and
    # never a false win. A campaign editing such a file turns the cache off.
    if not cfg.get("layers") or os.environ.get("SERVE_E2E_LAYER_COMPILE_CACHE") == "0":
        t.spec.setdefault("env", {})["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    # And prove it, because that failure is silent and this is the last place
    # that can see it. Nothing downstream can tell a candidate that scored 1.0000
    # from a candidate that served golden.
    on_path = [Path(p).resolve() for p in t.env().get("PYTHONPATH", "").split(os.pathsep) if p]
    if not any(p == root or root in p.parents for p in on_path):
        raise RuntimeError(
            f"this arm's tree {root} is not on the PYTHONPATH the server would get "
            f"({on_path}); it would serve somebody else's code and report this tree")
    _assert_same_cell(t, cfg)
    return t


def _assert_same_cell(t, cfg) -> None:
    """Refuse an arm that moved the operating point instead of serving it faster.

    THE SERVE COMMAND IS REACHABLE FROM THE ARM'S TREE, which the rest of this
    file used to assert it was not. `Target.retarget` moves `repo`, and
    `bench_script` and `configs_dir` are both derived from `repo` --
    `perfkit/targets/vllm_torchtpu.py` asks THIS arm's
    `scripts/vllm/benchmarking/run_benchmarks.sh --dry-run` what to launch. So a
    candidate can edit `--max-num-batched-tokens`, `--max-num-seqs`,
    `--block-size` or `--prefill-schedule-interval` in its own tree, and twelve
    of those flags are what `perfkit.cells` keys a cell on. Widening the prefill
    interval is worth +12-15% on this workload and is not the same workload; it
    would have been measured against the old cell's bar and admitted.

    The check is the project's own definition of "the same configuration" rather
    than a string compare: `cells.resolve` re-derives the arm's cell key from the
    command its tree would actually run, and a difference in anything unkeyed --
    the port, a flag reorder -- is correctly not a difference. What it does not
    police is env vars, and that is deliberate: `env_override.py` and the config
    script's exports are tree code and are the search space.

    Refusal is a raise, which drops the config, which voids the job. That is the
    contract's only channel for "I cannot vouch for this" and it is the right
    one: a number from a different cell is not a worse measurement, it is a
    measurement of something else.
    """
    want = cfg.get("cell")
    if not want:
        return
    from perfkit import cells

    # refresh, because the cache is keyed on the target NAME and every arm of a
    # job shares it. The stamp would catch a different commit, and relying on
    # that instead of asking is the kind of saving that reads as a clean pass.
    record, why = cells.resolve(t, cfg["config"], refresh=True)
    if record is None:
        raise RuntimeError(
            f"cannot derive this arm's cell from {t.repo}: {why}. The frozen cell is "
            f"{want} and an arm that cannot say which one it is has not been measured")
    if record["cell_key"] != want:
        raise RuntimeError(
            f"this arm serves cell {record['cell_key']}, not the frozen {want}: it changed the "
            f"serve command in its own tree rather than the code behind it. Fields: "
            f"{record['fields']}")
    say(f"cell {want} confirmed from {t.repo}")


def origin(handle) -> str:
    # Measured elsewhere, the code that ran is this tree's, committed and
    # built there (`_measure_buildkite`).
    if handle.pkg is None:
        return str(handle.root / "src" / "vllm_torchtpu" / "__init__.py")
    return handle.pkg.__file__


#: Secondary metrics reported on every config, for the harness to bound or just
#: to print. Medians only: the same three identical arms that spread 0.07% on
#: median TTFT spread 8.15% on p99 TTFT (76.98% on the 0.6B), so a tail is worth
#: REPORTING and is not worth guarding -- a bound on it fires on noise.
#: `p99_ttft_ms` is here for the report and should not be guarded.
#:
#: `completed` is the SUCCESSFUL REQUEST COUNT, and it is here for a different
#: reason than the latencies. A served arm has no fp32 reference, so
#: `worst_error` is identically 0 and the campaign's tolerance bounds nothing;
#: where golden is not reproducible at the token level the output fingerprint
#: cannot be compared either, and the loop is then left with no in-loop
#: integrity check at all. The request count is the coarsest one that still
#: works: a candidate whose scheduling change drops or fails requests reports
#: fewer than the control did in the same job, whatever it does to throughput.
#: Guard it as higher-is-better. It is coarse -- it cannot see a wrong token --
#: and it is not a substitute for an out-of-loop accuracy verdict.
SECONDARY_METRICS = ("median_ttft_ms", "median_tpot_ms", "p99_ttft_ms", "completed")

#: Set in every arm by `perfkit scorer run --accuracy` (perfkit/scorer.py's
#: ACCURACY_ENV): after its sweeps, every full config's arm runs the serving
#: config's own eval tasks against its server and reports the scores as
#: metrics (`perfkit/lmeval.py`). The jobs the harness writes for its own
#: checks ask for it.
ACCURACY_ENV = "PERFKIT_ACCURACY"
#: Set by the harness in every command a campaign's evaluator runs
#: (harness/internal/tool/evolve/golden.go's GoldenEnv): per config, golden's
#: output checksums from the segment's jobs, and whether golden's eval scores
#: are known yet, as `{"<config>": {"outputs": ["sha256:..."], "scored": true}}`.
#: Each arm decides from it whether to run the eval tasks (`_eval_rule`).
GOLDEN_ENV = "PERFKIT_GOLDEN"
#: Set by `perfkit scorer run` when its job measures golden alone
#: (perfkit/scorer.py's GOLDEN_ALONE_ENV): scoping's golden job, and the job
#: that profiles the base of a campaign's next segment (`_profile_read`).
GOLDEN_ALONE_ENV = "PERFKIT_GOLDEN_ALONE"
#: What a remote arm running the evals may take beyond its target's timeout.
ACCURACY_MINUTES = 90


def _profile_read() -> bool:
    """Whether a full config's profiled run is read: only in a job that
    measures golden alone, whose capture ranks the cell's blocks.

    A config with `profile` boots every arm's server with the profiler armed,
    in every job, so arms are compared on equal terms. The profiled run is an
    extra sweep, after the timed ones on this host and after the first on a
    remote arm, and it runs only where it is read. In a
    campaign's job a candidate's per-block times fed nothing but structural
    milestones, which a remote full config never reported, so campaigns scored
    on full configs have none."""
    return bool(os.environ.get(GOLDEN_ALONE_ENV))


def _timed(tputs: list) -> tuple:
    """(the throughput an arm reports, how many timing samples it is) from its
    timed sweeps, and the samples are the sweeps at least that fast.

    A record needs two timing samples that both clear the bar. Every sweep at
    or above an arm's throughput clears what it clears, so the sweeps of one
    server are samples: on the full model most of the spread between two jobs
    of one tree is between sweeps, not between server starts (on GLM-5.2 on
    cdk the start added nothing measurable, on Qwen3.5-35B on this host about
    0.5%). A candidate's arm reports the lower median, the middle sweep of
    three and the slower of two, so either is two samples; golden's, the
    denominator, reports the median. The median also leaves out one stray
    sweep, which a single sweep cannot. More rounds of sweeps on the same
    server moved no ratio by more than 0.2% in six candidates' jobs
    (2026-10-18)."""
    reported = (statistics.median(tputs) if os.environ.get("PERFKIT_ARM_CONTROL")
                else statistics.median_low(tputs))
    return reported, sum(1 for t in tputs if t >= reported)


def _eval_rule(name, cfg):
    """When this arm runs the serving config's eval tasks before its server
    stops: `("always", why)`, `("unless", golden's outputs)` to run them unless
    its output checksum is one of golden's, or None for never.

    A CHANGED OUTPUT IS CHECKED IN THE SAME SERVER START. A served candidate
    whose output differs from golden's is judged by its eval tasks against
    golden's, and the server that produced the output is still up: the eval
    costs minutes there, and a separate job costs another start (the harness
    runs one for the record before its segment closes, when no job did). Golden's arm runs them while the segment has no scores of golden's
    to compare with, and the harness compares later arms with those. Asked
    for (--accuracy), every arm runs them. A cut model's tokens are not text
    to score, so a layer config never does. Outside a campaign's job there is
    no GOLDEN_ENV, nothing to compare with, and no eval."""
    if cfg.get("layers"):
        return None
    if os.environ.get(ACCURACY_ENV):
        return ("always", "asked for")
    try:
        known = json.loads(os.environ.get(GOLDEN_ENV) or "null")
    except ValueError:
        return None
    if not isinstance(known, dict):
        return None
    golden = known.get(name) if isinstance(known.get(name), dict) else {}
    if os.environ.get("PERFKIT_ARM_CONTROL"):
        return None if golden.get("scored") else (
            "always", "golden has no eval scores in this segment yet")
    outputs = golden.get("outputs")
    return ("unless", [o for o in outputs if isinstance(o, str)]
            if isinstance(outputs, list) else [])


def _eval_why(rule, checksum: str) -> str:
    """Why an arm under `rule` whose output checksum is `checksum` runs the
    eval tasks, "" when it does not. An arm with no checksum is refused the
    record whatever it scores, so it runs none."""
    if rule is None:
        return ""
    kind, arg = rule
    if kind == "always":
        return arg
    if not checksum or checksum in arg:
        return ""
    return ("its output differs from golden's" if arg
            else "golden's output on this config is not known yet")


def _remote_eval_env(rule) -> dict:
    """`_eval_rule` as a remote arm's step reads it (run_arm.sh):
    FORGE_ACCURACY=1 runs the eval tasks, and FORGE_ACCURACY=changed runs them
    unless the arm's output checksum is one of FORGE_GOLDEN_OUTPUTS."""
    if rule is None:
        return {}
    kind, arg = rule
    if kind == "always":
        return {"FORGE_ACCURACY": "1"}
    return {"FORGE_ACCURACY": "changed", "FORGE_GOLDEN_OUTPUTS": " ".join(arg)}


def _accuracy_local(t, config: str, arm: str, srv, art: Path) -> dict:
    """The serving config's eval tasks against this arm's live server, as
    accuracy metrics. A task whose eval fails is said and left out, and the
    harness reads a task left out as not measured: the arm's throughput still
    stands."""
    from perfkit import lmeval

    env = t.config_env(config) if hasattr(t, "config_env") else {}
    tasks, tolerance = lmeval.eval_settings(env)
    parsed: dict = {}
    for task in tasks:
        try:
            parsed.update(t.accuracy(config, task, server=srv, out_dir=art / "eval" / task))
        except (SystemExit, Exception) as e:   # lm_eval's failure is a SystemExit
            say(f"{arm}/{config}: the {task} eval failed: {str(e)[-400:]}")
    return _said_accuracy(lmeval.accuracy_metrics(parsed, tasks, tolerance), tasks,
                          f"{arm}/{config}", art / "eval")


def _accuracy_remote(out: Path, label: str) -> dict:
    """The accuracy metrics out of what the repo's eval flow wrote under
    `out/eval` in a remote arm (perfkit/buildkite_arm/run_arm.sh)."""
    from perfkit import lmeval

    m = lmeval.accuracy_from_flow(out / "eval")
    try:
        cfg = json.loads((out / "eval" / "config.json").read_text())
    except (OSError, ValueError):
        cfg = {}
    tasks = (cfg.get("lm_eval_tasks") or "").split() or list(lmeval.DEFAULT_TASKS)
    return _said_accuracy(m, tasks, label, out / "eval.log")


def _said_accuracy(m: dict, tasks, label: str, where: Path) -> dict:
    from perfkit import lmeval

    got = [t for t in tasks if lmeval.ACCURACY_PREFIX + t in m]
    if got:
        say(f"{label}: accuracy " + ", ".join(
            f"{t} {m[lmeval.ACCURACY_PREFIX + t]:.4f}" for t in got))
    missing = [t for t in tasks if t not in got]
    if missing:
        say(f"{label}: NO ACCURACY for {', '.join(missing)}: the eval left no score; see {where}")
    return m


def _secondary(bundle: Path, warmup: int = WARMUP_SWEEPS) -> dict:
    """The secondary metrics out of `perfkit bench`'s JSON bundle.

    From the BUNDLE rather than the human lines, unlike the throughput: the rep
    lines carry tput and tpot and nothing else, and TTFT is the metric a
    scheduling campaign most needs. Median across sweeps, matching how the
    primary metric is reduced -- a mean would let one slow sweep set the bound.

    Missing is not zero. A metric this cannot find is left out of the map, and
    the harness reports a guard that matched nothing rather than reading an
    absent number as a pass or a breach.
    """
    try:
        blob = json.loads(bundle.read_text())
    except (OSError, ValueError):
        return {}
    found: dict[str, list[float]] = {}

    def walk(node):
        if isinstance(node, dict):
            m = node.get("metrics")
            if isinstance(m, dict):
                for k in SECONDARY_METRICS:
                    v = m.get(k)
                    if isinstance(v, (int, float)):
                        found.setdefault(k, []).append(float(v))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(blob)
    # Drop the warmup from the front, the same sweep the primary metric drops,
    # or the two numbers describe different sets of sweeps.
    out = {}
    for k, vals in found.items():
        timed = vals[warmup:] if len(vals) > warmup else vals
        if timed:
            out[k] = statistics.median(timed)
    return out


def _bundle_for(text: str) -> Path | None:
    """Where `perfkit bench` said it wrote its JSON, off its own output."""
    # Quote-stripped: the path is printed inside the JSON bundle's own echo, so
    # a bare \S+ capture takes the opening quote with it and the Path never
    # exists. Verified against a real bench.out rather than assumed.
    m = re.findall(r"[\"'\s]([^\"'\s]+/bench\.json)", text)
    return Path(m[-1]) if m else None


def _throughputs(text: str) -> list[float]:
    """Every sweep's metric out of a `perfkit bench` run, ONCE EACH.

    Anchored on the `repN:` line. A bare `tput=` search finds each sweep twice —
    `perfkit bench` prints the reps as they finish and then repeats them inside
    the JSON bundle's `summary` array — and the first run of this scorer
    reported six values for three sweeps. The median survived that (duplicating
    every element does not move it) which is exactly why it would have gone
    unnoticed; the count in the log did not.

    Parsed from the human lines rather than the bundle because the bundle lands
    under a run directory the arm does not own, and a scorer that had to find it
    would be one more thing to get wrong between the boot and the number.
    """
    return [float(m.group(1))
            for m in re.finditer(r"^\s*rep\d+:.*?tput=([0-9.]+)", text, re.M)]


def measure(handle, name, cfg):
    """Boot this arm's tree, sweep it, and report throughput as a latency.

    ONE BOOT PER ARM PER CONFIG, and the sweeps run against it with
    `--against-port` so the boot is paid once. A fresh boot per sweep would
    measure the compile cache as much as the kernel.

    A TWIN IS REFUSED. A config that cuts the experts (`experts`) or pins
    another cell's KV allocation (`kv_cell`) served a deployment on fewer
    chips. Its all-to-all crossed fewer chips and its engines stepped in
    lockstep at another size, so it scored a step the deployment does not
    run, and a candidate it scored well moved the deployment far less.
    """
    twin = [k for k in ("experts", "kv_cell") if cfg.get(k)]
    if twin:
        raise RuntimeError(
            f"{name} declares {' and '.join(twin)}: it is a twin of a deployment on fewer chips, "
            f"and forge measures no twins. Score the deployment's own config as "
            f"`<config>@layers`, on the deployment's machines")
    if REMOTE.get(name) == "buildkite":
        return _measure_buildkite(handle, name, cfg)
    if REMOTE.get(name) == "cdk":
        return _measure_cdk(handle, name, cfg)
    if cfg.get("layers"):
        return _measure_layers(handle, name, cfg)
    config = cfg["config"]
    t = _target_for(cfg, handle.root)
    repeats = max(int(cfg.get("repeats", REPEATS)), RECORD_SAMPLES)
    warmup = int(cfg.get("warmup", WARMUP_SWEEPS))
    asked = repeats + warmup
    # Keyed on the ARM and the config. `name` is the config, so keying on it
    # alone put both arms of the A/A in one directory.
    arm = os.environ.get("PERFKIT_ARM", "arm")
    art = _arm_dir(arm, config)
    art.mkdir(parents=True, exist_ok=True)

    profile = bool(cfg.get("profile"))
    prof_dir = art / "profile"
    env = {}
    if profile:
        shutil.rmtree(prof_dir, ignore_errors=True)
        prof_dir.mkdir(parents=True)
        # The phased profiler writes one trace per phase, and only the profiled
        # sweep below turns it on.
        _add_serve_args(t, f"--profiler-config.profiler torch --profiler-config.torch_profiler_dir "
                           f"{prof_dir} --profiler-config.max_iterations 5")
        env = {"USE_PHASED_PROFILER": "true"}

    t.cleanup()
    t0 = time.time()
    with t.serve(config, env_overrides=env, log_path=art / "server.log") as srv:
        srv.wait_ready(timeout_s=READY_TIMEOUT_S)
        say(f"{arm}/{config}: READY in {time.time() - t0:.0f}s on :{srv.port}")
        _record_kv(cfg, _kv_seen(art / "server.log"), arm)
        bench_out = art / "bench.out"
        with open(bench_out, "w") as fh:
            rc = subprocess.call(
                ["perfkit", "bench", "--target", t.name, "--config", config,
                 "--repeats", str(asked), "--tag", f"serve-e2e-{arm}",
                 "--against-port", str(srv.port), "--env", "CAPTURE_PROFILE=0"] + _acks(cfg),
                stdout=fh, stderr=subprocess.STDOUT)
        text = bench_out.read_text()
        every = _throughputs(text)
        # BEFORE THE CAPTURE. A dead engine fails every sweep at zero throughput
        # in a minute, and the capture then waited on the dead server: segment
        # 3's scoping lost 45 minutes of device time that way. The arm is
        # refused either way; this only stops it paying for the refusal.
        failed = _bench_failed(rc, every, warmup)
        if failed:
            raise RuntimeError(f"{failed}; see {bench_out}")
        # The numerics fingerprint, which is this scorer's correctness half. An
        # end-to-end arm has no fp32 reference to diff a tensor against; what it
        # has is whether the served model still produces the same tokens, and
        # that is the check that caught c34 not being output-preserving at the
        # served config when its micro tolerance passed (29dhk224-01).
        num_out = art / "numerics.out"
        num_ok = _capture(t, config, arm, srv.port, num_out, "", ["--long-suite", LONG_SUITE])
        # THE SAME SUITE WITH NO LOGPROBS ASKED FOR. Some sampling paths only run
        # when no request asks for logprobs -- a vocab-parallel greedy argmax is
        # the one this was written for -- and the capture above, which asks for
        # the top 5, never reaches them. Serial, so it reproduces like the first.
        nolp_out = art / "numerics-nologprobs.out"
        nolp_ok = _capture(t, config, arm, srv.port, nolp_out, "-nologprobs",
                           ["--no-logprobs"])
        # AND ONE BATCHED REQUEST, as a sanity check rather than a fingerprint.
        # At the benchmark's batch size greedy outputs depend on what else is in
        # the batch, so two identical batched requests do not reproduce. What a
        # broken batched kernel does show is its outputs leaving the serial ones
        # almost at once, so the share of each serial output the batch repeats
        # before diverging is reported, and a campaign can guard on it.
        batch_out = art / "numerics-batch.out"
        batch_ok = _capture(t, config, arm, srv.port, batch_out, f"-batch{BATCH_ROWS}",
                            ["--batch", str(BATCH_ROWS), "--no-logprobs"])
        device = _profile_blocks(t, config, arm, srv.port, prof_dir,
                                 cfg.get("cell", ""), _acks(cfg)) if profile and _profile_read() else {}
        checksum = _combined(_fingerprint(num_out) if num_ok else "",
                             _fingerprint(nolp_out) if nolp_ok else "")
        # Last, so nothing measured above ran beside the eval's requests.
        why = _eval_why(_eval_rule(name, cfg), checksum)
        if why:
            say(f"{arm}/{config}: running the eval tasks: {why}")
        accuracy = _accuracy_local(t, config, arm, srv, art) if why else {}

    # Drop the warmup FROM THE FRONT, and only if the arm produced enough sweeps
    # to have one to spare. An arm that lost sweeps to a flake keeps what it has
    # rather than being cut down to nothing and voiding the job.
    tputs = every[warmup:] if len(every) > warmup else every
    median = statistics.median(tputs)
    reported, samples = _timed(tputs)
    if reported <= 0:
        raise RuntimeError(f"its score would be a timed sweep at zero throughput ({tputs}): the "
                           f"server stopped serving during them; see {bench_out}")
    spread = (max(tputs) - min(tputs)) / median if len(tputs) > 1 and median else 0.0
    say(f"{arm}/{config}: warmup {every[:warmup]} timed {tputs} "
        f"median {median:.1f} spread {spread * 100:.2f}% {METRIC}"
        + (f"; scored on {reported:.1f}, the lower median" if reported != median else ""))
    secondary = _secondary(_bundle_for(text) or Path("/nonexistent"), warmup)
    # How many timing samples the score is: the harness's record counts them.
    secondary["timing_rounds"] = samples
    # The arm's own noise: how far apart its timed sweeps read. With A/A arms
    # off it is the only spread a job measures, and the full-model check of a
    # layer-scored record is bounded by golden's (campaign.ConfirmationSpread).
    if len(tputs) > 1:
        secondary["sweep_spread"] = spread
    agreement = _prefix_agreement(batch_out, nolp_out) if batch_ok and nolp_ok else None
    if agreement is not None:
        secondary["batched_prefix_agreement"] = agreement
        say(f"{arm}/{config}: a {BATCH_ROWS}-row batch repeats {agreement:.2f} of each serial "
            f"output before diverging, on average")
    secondary.update(device)
    secondary.update(accuracy)
    if secondary:
        say(f"{arm}/{config}: " + "  ".join(f"{k}={v:.4g}" for k, v in sorted(secondary.items())))
    else:
        # Loud, because a campaign that guards one of these would silently bound
        # nothing and the harness's note is the only other place it shows.
        say(f"{arm}/{config}: NO SECONDARY METRICS FOUND in the bench bundle; "
            f"any guard on {', '.join(SECONDARY_METRICS)} will bound nothing")
    return {
        # Everything the campaign is NOT optimising. Reported unconditionally:
        # bounding them is the campaign's choice, measuring them is not.
        "metrics": secondary,
        # A LATENCY, because the harness ratio is control/arm and a speedup has
        # to come out above 1. Throughput is higher-is-better, so it is inverted
        # here rather than in the harness, where it would silently invert the
        # win rule for every other scorer too.
        "latency_us": 1e6 / reported,
        # Reported directly: there is no reference tensor to diff. The tolerance
        # the campaign applies is over the numerics fingerprint, and a serving
        # arm that produced tokens at all has a defined error of zero against
        # itself — the real check is the comparison the harness cannot do here,
        # so it is recorded and a campaign scoring this way carries a numerics
        # gate beside it.
        "worst_error": 0.0,
        "checksum": checksum,
        "first_tokens": _first_tokens(num_out) if num_ok else {},
        # What ran, from the tree and the serve command rather than a label.
        "dispatched": f"serve/{handle.tree}/{config}",
        "binary_hash": "sha256:" + (handle.tree or "unknown"),
    }


def _measure_layers(handle, name, cfg):
    """Boot this arm's tree on the model cut to its leading layers, take the
    output captures, and report the device's op time per step, by the formula
    the config declares (`_profile_phases`), as the latency. No throughput
    sweeps: a cut model's throughput is the host's."""
    t = _target_for(cfg, handle.root)
    arm = os.environ.get("PERFKIT_ARM", "arm")
    art = _arm_dir(arm, name)
    art.mkdir(parents=True, exist_ok=True)
    prof_dir = art / "profile"
    shutil.rmtree(prof_dir, ignore_errors=True)
    prof_dir.mkdir(parents=True)
    _add_serve_args(t, f"--profiler-config.profiler torch --profiler-config.torch_profiler_dir "
                       f"{prof_dir} --profiler-config.max_iterations {LAYER_STEPS}")
    pin = _kv_pin(cfg)
    if pin:
        _add_serve_args(t, f"--num-gpu-blocks-override {pin['blocks']}")
    return _cut_arm(handle, name, cfg, t, art, prof_dir, pin)


def _speculative(meta: dict) -> str:
    """The serve argument that turns on speculative decoding in a config's
    serve command, as the command spells it, or "". vLLM's `--speculative-config`
    and its older `--speculative-model` and `--num-speculative-tokens`, with
    dashes or underscores, and SGLang's `--speculative-algorithm`."""
    for word in str(meta.get("serve_cmd") or "").split():
        flag = word.split("=", 1)[0]
        if flag.startswith("--") and flag[2:].replace("_", "-").startswith(
                ("speculative-", "num-speculative-")):
            return flag
    return ""


def _refuse_speculative(meta: dict, name: str) -> None:
    """Refuse a layer arm whose serve config runs speculative decoding, before
    anything boots or is submitted.

    A layer run cannot stand in for that step. A cut keeps a few decoder layers
    beside the drafter, whose passes are then a far larger share of the step
    than in the full model. Serving the cut without the drafter changes every
    decode step's shape instead: one token per sequence, not the drafted ones
    plus one. And the drafter's own layer is stored past the last decoder
    layer, which the cut drops. GLM-5.2's AgentX config served MTP, and its cut
    failed to load the drafter an hour into a cluster job."""
    flag = _speculative(meta)
    if flag:
        raise RuntimeError(
            f"{name}: its serve config runs speculative decoding ({flag}), and a layer run "
            f"cannot stand in for that step: a cut keeps a few layers beside the full drafter, "
            f"and serving it without the drafter changes every decode step's shape. Score this "
            f"config on the full model: profile golden with a full-config job and `perfkit "
            f"profile blocks --write`, and pick the block with `amplio campaign next-block "
            f"--cell <cell> --full-model`")


def _add_serve_args(t, flags: str) -> None:
    """Add serve arguments to this arm's server, after any the target passes.

    THROUGH THE TARGET, NOT THE PROCESS ENVIRONMENT. The dry run that builds
    the serve command reads EXTRA_SERVE_ARGS from the target's environment, and
    a target that sets the variable itself replaces whatever the process
    exported. The 397B's bar target does, for `--mamba-ssm-cache-dtype`, so a
    profiler exported here never reached its server. `t` is this arm's own
    target object, so nothing outlives the arm either."""
    env = t.spec.setdefault("env", {})
    base = env.get("EXTRA_SERVE_ARGS", os.environ.get("EXTRA_SERVE_ARGS", ""))
    env["EXTRA_SERVE_ARGS"] = f"{base} {flags}".strip()


def _cut_arm(handle, name, cfg, t, art: Path, prof_dir: Path, pin=None) -> dict:
    """A layer arm once its serve command carries the profiler: cut, boot,
    capture, score. Its weights come from `weights` or the config's MODEL_URI
    when this host holds none."""
    from perfkit import layers

    config = cfg["config"]
    arm = os.environ.get("PERFKIT_ARM", "arm")
    meta = t.config_meta(config)
    _refuse_speculative(meta, name)
    # Where the captures begin (PREFILL_ONLY_FROM). Added after the serve
    # command is read, so the settings merge into the config's own
    # --additional-config; `t.serve` reads the command again with them.
    cells = cfg.get("cells")
    _add_serve_args(t, "--additional-config " + json.dumps(
        _cells_settings(meta) if cells else _phase_settings(meta, cfg.get("score_phase")),
        separators=(",", ":")))
    if cells:
        # The driver primes each prefix once and its decodes reuse it.
        _add_serve_args(t, "--enable-prefix-caching")
    src = layers.resolve_checkpoint(
        meta.get("model") or "", t.env(),
        uri=cfg.get("weights") or (meta.get("config_env") or {}).get("MODEL_URI", ""))
    budget, decodes = _step_budget(meta)
    if budget is None:
        say(f"{arm}/{name}: the serve command sets no --max-num-batched-tokens, so the steps "
            f"that carried a prefill chunk cannot be counted; scoring decode-only steps")
    want = cfg.get("layers")
    full = layers.read_config(src)
    plan = layers.plan(full, periods=int(cfg.get("periods", 1)),
                       layers=None if want == "auto" else int(want))
    load = _light_load(meta, full.get("vocab_size") or (full.get("text_config") or {}).get("vocab_size"))
    if not plan.complete:
        raise RuntimeError(f"{plan.describe()}: this cut leaves out a kind of layer the model has")
    ckpt = layers.build(src, plan.layers)
    say(f"{arm}/{name}: {plan.describe()}, from {ckpt}")

    warmup = int(cfg.get("warmup", 0))
    t.cleanup()
    t0 = time.time()
    with t.serve(config, env_overrides={"USE_PHASED_PROFILER": "true"},
                 log_path=art / "server.log", model_dir=ckpt) as srv:
        srv.wait_ready(timeout_s=READY_TIMEOUT_S)
        say(f"{arm}/{name}: READY in {time.time() - t0:.0f}s on :{srv.port}")
        kv = _kv_check(pin, _kv_seen(art / "server.log"), f"{arm}/{name}")
        if warmup:
            out = art / "warmup.out"
            with open(out, "w") as fh:
                rc = subprocess.call(
                    ["perfkit", "bench", "--target", t.name, "--config", config,
                     "--repeats", str(warmup), "--tag", f"serve-layers-{arm}-warmup",
                     "--against-port", str(srv.port), "--env", "CAPTURE_PROFILE=0"] + _acks(cfg),
                    stdout=fh, stderr=subprocess.STDOUT)
            if rc != 0:
                raise RuntimeError(f"the warm-up sweep failed (rc={rc}); see {out}")
        # THE NUMERICS CHECK, as on the full config. A cut model's tokens are not
        # the full model's, but golden's cut and the candidate's are the same
        # weights through the same layers, so their fingerprints agree exactly
        # when the candidate preserved the output.
        num_out = art / "numerics.out"
        nolp_out = art / "numerics-nologprobs.out"
        batch_out = art / "numerics-batch.out"
        num_ok = _capture(t, config, arm, srv.port, num_out, "", ["--long-suite", LONG_SUITE])
        nolp_ok = _capture(t, config, arm, srv.port, nolp_out, "-nologprobs", ["--no-logprobs"])
        batch_ok = _capture(t, config, arm, srv.port, batch_out, f"-batch{BATCH_ROWS}",
                            ["--batch", str(BATCH_ROWS), "--no-logprobs"])
        captures = []
        if cells:
            captures.append(_run_cells(t, config, arm, srv.port, prof_dir, art, cfg))
        else:
            for k in range(int(cfg.get("repeats", 1))):
                captures.append(_profile_phases(t, config, arm, srv.port, prof_dir, cfg["cell"],
                                                budget, decodes, _acks(cfg),
                                                cfg.get("score_phase"), cfg.get("decode_share"),
                                                load))
                _set_aside(prof_dir, k)

    ops, metrics = _layer_score(captures)
    if ops is None:
        raise RuntimeError(f"no profiled run reported the device's op time; see {art}")
    # NAMED, NOT REFUSED. Which kinds of layer a cut keeps is the model's, not
    # the candidate's, so whether a cut lost one is settled once, by golden's
    # arm at scoping. Refused here it would void every arm once the rows went
    # stale, and every candidate that removes a block outright -- the wins
    # that delete work.
    missing = _missing_blocks(cfg["cell"], metrics,
                              "prefill" if cfg.get("score_phase") == "prefill" else "decode")
    metrics["layer_blocks_missing"] = len(missing)
    if missing:
        say(f"{arm}/{name}: the {plan.layers}-layer capture holds no time for "
            f"{', '.join(missing)}, which the cell's rows have. On golden at scoping that "
            f"means the cut lost a kind of layer: score this block on {config} instead")
    agreement = _prefix_agreement(batch_out, nolp_out) if batch_ok and nolp_ok else None
    if agreement is not None:
        metrics["batched_prefix_agreement"] = agreement
    metrics.update({"layers_kept": plan.layers, "layers_total": plan.total,
                    "layer_mix_matches": 1.0 if plan.mix_matches else 0.0, **kv})
    _keep_metrics(art, metrics)
    share = metrics.get("step_share_decode")
    say(f"{arm}/{name}: device ops {ops:.4f} ms per step, the median of {len(captures)} "
        f"capture(s): decode-only {metrics.get('device_ops_ms', 0):.4f} ms"
        + (f", with a prefill chunk {metrics['device_ops_ms_prefill']:.4f} ms"
           if metrics.get("device_ops_ms_prefill") else "")
        + (f", {100 * share:.0f}% of {metrics.get('steps_counted', 0):.0f} steps decode-only"
           if share is not None else ", no step shares, so scored on decode-only steps"))
    return {
        "metrics": metrics,
        # Lower is better already, so unlike throughput it is not inverted.
        "latency_us": ops * 1000.0,
        "worst_error": 0.0,
        "checksum": _combined(_fingerprint(num_out) if num_ok else "",
                              _fingerprint(nolp_out) if nolp_ok else ""),
        "first_tokens": _first_tokens(num_out) if num_ok else {},
        "dispatched": f"serve-layers/{handle.tree}/{config}/L{plan.layers}",
        "binary_hash": "sha256:" + (handle.tree or "unknown"),
    }


def _keep_metrics(where: Path, metrics: dict) -> None:
    """A layer arm's metrics, beside its blocks: `perfkit layers budget` reads
    the step counts of its profiled sweep from here, which the job's stdout
    carries only for as long as somebody keeps it."""
    (where / "layer-metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True))


def _run_cells(t, config: str, arm: str, port: int, prof_dir: Path, art: Path, cfg) -> dict:
    """A cells arm against its running server: the driver serves each cell and
    files its captures (`perfkit/buildkite_arm/cells.py`), and they are scored
    here (`_score_cells`)."""
    from perfkit import paths as pk_paths

    plan = art / "cells-plan.json"
    plan.write_text(json.dumps(cfg["cells"]))
    out = art / "cells"
    with open(art / "cells.out", "w") as fh:
        rc = subprocess.call(
            [sys.executable, str(Path(pk_paths.FORGE) / "perfkit" / "buildkite_arm" / CELLS_DRIVER),
             "run", "--port", str(port), "--cells", str(plan), "--profile-dir", str(prof_dir),
             "--out", str(out), "--steps", str(LAYER_STEPS),
             "--text", str(pk_paths.EVALS_DIR / "numerics_suites" / "default.json")],
            stdout=fh, stderr=subprocess.STDOUT)
    if rc:
        say(f"{arm}/{config}: the cell driver exited {rc}; see {art / 'cells.out'}")

    def blocks(phase: Path, rules: str, dest: Path):
        return _phase_blocks(phase, cfg["cell"], rules, dest, arm, config, t.name)

    return _score_cells(out, cfg["cells"], blocks, f"{arm}/{config}")


def _set_aside(prof_dir: Path, k: int) -> None:
    """Copy each phase capture just taken beside the profile directory and empty
    the live ones, so the next capture is read on its own. While the server runs
    the live directory is a link to tmpfs (`profile_phases_on_tmpfs`), and the
    link stays."""
    if not prof_dir.is_dir():
        return
    for phase in sorted(p for p in prof_dir.iterdir() if p.is_dir()):
        shutil.copytree(phase, prof_dir.parent / f"{phase.name}-{k}")
        for f in phase.iterdir():
            shutil.rmtree(f) if f.is_dir() and not f.is_symlink() else f.unlink()


def _layer_score(captures: list[dict]):
    """The median scored op time per step (`device_ops_ms_weighted`, by the
    config's formula) over the captures that reported one, and each metric's
    median over the captures that carry it. A capture that could not compute
    its config's formula reports none, and is not scored on another one: two
    arms of a job scored by different formulas make a ratio of nothing. See
    LAYER_STEPS for why op time and not busy time."""
    ops = [c["device_ops_ms_weighted"] for c in captures if c.get("device_ops_ms_weighted")]
    keys = {k for c in captures for k in c}
    metrics = {k: statistics.median(c[k] for c in captures if k in c) for k in keys}
    return (statistics.median(ops) if ops else None), metrics


def _flag(serve_cmd: str, name: str):
    """An integer flag of the serve command, such as `--max-num-seqs`, or None."""
    m = re.search(rf"--{re.escape(name)}[= ](\d+)", serve_cmd)
    return int(m.group(1)) if m else None


def _step_budget(meta: dict):
    """(token budget of one engine step, decodes running in it) for a config,
    from its serve command and sweep. The decodes are the sweep's concurrency
    per engine, capped at `--max-num-seqs`. (None, 0) with no budget flag."""
    cmd = meta.get("serve_cmd") or ""
    budget = _flag(cmd, "max-num-batched-tokens")
    seqs = _flag(cmd, "max-num-seqs")
    try:
        per_engine = -(-int(str(meta.get("concurrency")).split()[0]) // int(meta.get("dp") or 1))
    except (TypeError, ValueError):
        per_engine = None
    decodes = min(x for x in (seqs, per_engine) if x) if (seqs or per_engine) else 0
    return budget, decodes


def _iteration_tokens(port: int):
    """The server's step counters, summed over its engines: the histogram of
    tokens per engine step as the cumulative step count at each bucket edge,
    with `inf` holding every step, and "prompt" for the prompt tokens it has
    processed. None if `/metrics` cannot be read."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
            text = r.read().decode()
    except OSError:
        return None
    return _parse_counters(text)


def _parse_counters(text: str):
    """`_iteration_tokens` from the text of a `/metrics` reading."""
    counts: dict = {}
    for line in text.splitlines():
        m = re.match(r"vllm:iteration_tokens_total_bucket\{(.*)\}\s+(\S+)$", line)
        le = re.search(r'le="([^"]+)"', m.group(1)) if m else None
        if le:
            edge = float(le.group(1))
            counts[edge] = counts.get(edge, 0.0) + float(m.group(2))
        p = re.match(r"vllm:prompt_tokens_total\{.*\}\s+(\S+)$", line)
        if p:
            counts["prompt"] = counts.get("prompt", 0.0) + float(p.group(1))
    return counts or None


def _step_shares(before: dict, after: dict, budget, decodes: int):
    """(share of decode-only steps, steps, steps that carried a prefill chunk)
    between two readings of the server's counters, or None.

    THE HISTOGRAM CANNOT SAY WHICH STEPS CARRIED A CHUNK. vLLM books a
    request's whole prompt in the step where its prefill completes, so the chunk
    steps before it read as decode-only: a GLM-5.2 sweep with a 1024-token
    budget had one step per request above 8192 tokens and none between 65 and
    8192. Its count of steps is right, and the prompt tokens are counted apart.
    While prompts wait, a step fills its budget with the running decodes and a
    chunk, so the steps that carried one number the prompt tokens over what the
    decodes leave: between P/budget and P/(budget - decodes). The midpoint is
    taken. With a budget of hundreds of tokens and tens of decodes the two are
    a few percent apart."""
    steps = after.get(float("inf"), 0.0) - before.get(float("inf"), 0.0)
    prompt = after.get("prompt", 0.0) - before.get("prompt", 0.0)
    if steps <= 0 or not budget or budget <= decodes / 2:
        return None
    with_prefill = min(steps, prompt / (budget - decodes / 2))
    return (steps - with_prefill) / steps, int(steps), int(round(with_prefill))


#: A cell row this small may be absent from a layer run's capture: below it a
#: block is rounding, not a kind of layer the cut lost.
MISSING_BLOCK_MS = 0.005


def _missing_blocks(cell: str, metrics: dict, phase: str = "decode") -> list[str]:
    """The cell's rows for `phase`, the one the config's capture holds, that a
    layer run's capture holds no time for. Its host-time rows are left out:
    idle time is not an op a capture attributes. So are rows no rule of the
    cell's names: no capture holds time for those, whatever the cut kept, and a
    cell with rows and no rules would otherwise report every row as a kind of
    layer the cut lost."""
    from perfkit import headroom

    prefix = "" if phase == "decode" else "prefill_"
    doc = headroom.load()
    idle = headroom.host_blocks_for(doc, cell).get(phase, set())
    ruled = {r.block for r in (doc.get("rules") or {}).get(cell, []) if r.phase == phase}
    rows = [b for b in doc.get("blocks", {}).get(cell, [])
            if b.phase == phase and b.name not in idle and b.ms_per_step >= MISSING_BLOCK_MS]
    if not rows:
        say(f"cell {cell} has no {phase} rows, so nothing checks that the cut kept every block")
    unruled = sorted(b.name for b in rows if b.name not in ruled)
    if unruled:
        say(f"cell {cell} has no rule that divides a capture into {', '.join(unruled)}, "
            f"so nothing checks that the cut kept {'it' if len(unruled) == 1 else 'them'}")
    return sorted(b.name for b in rows
                  if b.name in ruled and not metrics.get(f"{prefix}block:{b.name}"))


#: Rows in the batched sanity capture: the concurrency the served cells run at.
BATCH_ROWS = 64
#: Seconds the profiler is given to flush its trace after the profiled sweep:
#: the fixed wait for a sweep with no phases to watch, and the most a watched
#: one waits for a capture still being written.
PROFILE_FLUSH_S = 30
#: How often a watched sweep reads the server log for written captures.
PROFILE_POLL_S = 2

#: A rank's phased profiler starting or finishing a capture, in the server log:
#: "(Worker_DP3_EP3 pid=1) INFO 01:53:44 [utils.py:281] Starting profiling for
#: prefill_heavy phase", then "... Profiling for prefill_heavy phase finished",
#: which the profiler logs once the trace is written and merged.
_CAPTURE_RE = re.compile(r"\(Worker_DP(\d+)[^)]*\)[^\n]*?(Starting profiling for|Profiling for) "
                         r"(\w+) phase( finished)?")


def _capture_state(text: str) -> dict:
    """{phase: (ranks that started it, ranks that finished it)} from log text."""
    state: dict = {}
    for m in _CAPTURE_RE.finditer(text):
        rank, _, phase, finished = m.groups()
        state.setdefault(phase, (set(), set()))[1 if finished else 0].add(rank)
    return state


def _captured(state: dict, phases) -> bool:
    """Every phase in `phases` written on every rank that started it."""
    return bool(phases) and all(
        phase in state and state[phase][0] and state[phase][0] <= state[phase][1]
        for phase in phases)


def _in_flight(state: dict) -> bool:
    """A capture started on some rank and not written there yet."""
    return any(started - finished for started, finished in state.values())


def _light_load(meta: dict, vocab) -> dict | None:
    """What a layer arm's profiled run sends (`_light_sweep`), from the config's
    dry run: the run_benchmarks.sh profile run's load, one full wave of the
    config's concurrency with its random-dataset lengths. None for a config
    this does not reproduce -- another dataset, extra bench arguments or an
    extra request body, or no vocabulary to draw prompts from -- which keeps
    the repository's own client."""
    env = meta.get("config_env") or {}
    if (env.get("DATASET_NAME") or "random") != "random" or env.get("EXTRA_BENCH_ARGS") \
            or env.get("BENCHMARK_EXTRA_BODY") or not vocab:
        return None
    try:
        isl, osl = (int(x) for x in str(meta.get("isl_osl")).split()[0].split(":"))
        concurrency = int(str(meta.get("concurrency")).split()[0])
        ratio = float(env["RANDOM_RANGE_RATIO"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    # run_benchmarks.sh's two styles: "min" samples in [r*len, len], vllm bench
    # serve's own "symmetric" in [(1-r)*len, (1+r)*len].
    if (env.get("RANGE_RATIO_STYLE") or "symmetric") == "min":
        span = lambda n: (max(1, round(ratio * n)), n)
    else:
        span = lambda n: (max(1, round((1 - ratio) * n)), round((1 + ratio) * n))
    temperature = env.get("BENCHMARK_TEMPERATURE")
    return {"input": span(isl), "output": span(osl), "concurrency": concurrency,
            "prompts": int(env.get("PROFILE_NUM_PROMPTS") or concurrency),
            "temperature": float(temperature) if temperature not in (None, "") else None,
            "vocab": int(vocab)}


def _light_sweep(port: int, load: dict, out, seed: int = 0) -> int:
    """The profiled run, sent by this process instead of run_benchmarks.sh: the
    profiler armed, then `load`'s prompts at its concurrency, then disarmed.
    Its exit code, 0 when every request was served.

    WHY NOT THE REPOSITORY'S CLIENT. With CAPTURE_PROFILE=1, run_benchmarks.sh
    runs the config's measured sweep first -- 640 prompts on the 397B -- and only
    then the profile run, and starts `vllm bench serve` for each. A layer arm
    reads the profile run alone, so a third of its time went to a sweep
    nothing read. Op time is device time, which the client does not change.

    Prompts are token ids, as vllm's random dataset draws them: a run of
    consecutive ids from a random offset, modulo the vocabulary. The seed is
    fixed, so every arm is sent the same prompts."""
    import random
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    base = f"http://127.0.0.1:{port}"
    rng = random.Random(seed)
    requests = []
    for _ in range(load["prompts"]):
        n, m = rng.randint(*load["input"]), rng.randint(*load["output"])
        offset = rng.randrange(load["vocab"])
        body = {"prompt": [(offset + k) % load["vocab"] for k in range(n)], "max_tokens": m,
                "ignore_eos": True, "stream": False}
        if load["temperature"] is not None:
            body["temperature"] = load["temperature"]
        requests.append(body)

    def post(path: str, body=None, timeout: float = 60):
        data = json.dumps(body).encode() if body is not None else b""
        req = urllib.request.Request(base + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as r:
            model = json.loads(r.read())["data"][0]["id"]
        post("/start_profile")
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"light sweep: the server did not answer ({exc})", file=out, flush=True)
        return 1
    failed = []

    def one(body):
        try:
            post("/v1/completions", {**body, "model": model}, timeout=3600)
        except OSError as exc:
            failed.append(str(exc))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=load["concurrency"]) as pool:
        list(pool.map(one, requests))
    try:
        post("/stop_profile")
    except OSError:
        pass
    print(f"light sweep: {len(requests) - len(failed)} of {len(requests)} requests served in "
          f"{time.time() - t0:.0f}s at concurrency {load['concurrency']}"
          + (f"; first failure: {failed[0]}" if failed else ""), file=out, flush=True)
    return 1 if failed else 0


def _stop_profile(port: int) -> bool:
    """Disarm the server's phased profiler, so it captures nothing more."""
    import urllib.request

    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/stop_profile", data=b"",
                                     method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
        return True
    except OSError as exc:
        say(f"stopping the profiler failed ({exc}); the sweep goes on capturing")
        return False


def _capture(t, config: str, arm: str, port: int, out: Path, label: str, extra=()) -> bool:
    """One `perfkit numerics capture` against the running server, its output
    in `out`. False, said loudly, when it failed: the fingerprint is this
    scorer's only correctness signal, and a served campaign refuses the record
    to a candidate whose measurement carries none."""
    with open(out, "w") as fh:
        try:
            rc = subprocess.call(
                ["perfkit", "numerics", "capture", "--target", t.name, "--config", config,
                 "--label", f"serve-e2e-{arm}{label}", "--against-port", str(port), *extra],
                stdout=fh, stderr=subprocess.STDOUT, timeout=NUMERICS_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            rc = -1
    if rc != 0:
        say(f"{arm}/{config}: NUMERICS CAPTURE{label.upper()} FAILED (rc={rc}); see {out}")
    return rc == 0


def _combined(*prints: str) -> str:
    """One fingerprint for an arm's serial captures. Empty when any is empty:
    a capture that did not run is not one that matched."""
    if not prints or not all(prints):
        return ""
    if len(prints) == 1:
        return prints[0]
    return "sha256:" + hashlib.sha256("|".join(prints).encode()).hexdigest()[:16]


def _saved(out: Path) -> dict:
    """The capture a `perfkit numerics capture` run saved, from its output.

    The output holds what the capture said on stderr too (`_capture` keeps
    both), such as why a server with no tokenizer endpoint got no long prompts,
    so the path is read by its field when the whole is not JSON, as
    `_fingerprint` reads the digest.
    """
    try:
        text = out.read_text()
        try:
            saved = json.loads(text)["saved"]
        except ValueError:
            m = re.search(r'"saved"\s*:\s*"([^"]+)"', text)
            if not m:
                return {}
            saved = m.group(1)
        saved = Path(saved)
        return json.loads((saved if saved.is_absolute() else out.parent / saved).read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _first_tokens(num_out: Path) -> dict:
    """Each prompt's first generated token and its logprob, by prompt set
    (`set` on the capture's items, the default suite's when none), from the
    capture `num_out` names. The harness checks a candidate's against golden's
    (scorer.FirstTokenCheck): the first token is the one position where both
    read the same input. Empty when the capture has none."""
    out: dict = {}
    for it in _saved(num_out).get("items", []):
        toks, lps = it.get("token_ids") or [], it.get("logprobs") or []
        if toks and lps and isinstance(lps[0], (int, float)):
            out.setdefault(it.get("set") or "default", {})[it["id"]] = [toks[0], lps[0]]
    return out


def _prefix_agreement(batch_out: Path, serial_out: Path):
    """The mean share of each prompt's serial output that the batched capture
    repeats before diverging, over the batch's rows. None if either capture
    cannot be read."""
    serial = {it["id"]: it.get("text", "") for it in _saved(serial_out).get("items", [])}
    rows = _saved(batch_out).get("items", [])
    if not serial or not rows:
        return None
    shares = []
    for it in rows:
        ref, got = serial.get(it["id"].split("#")[0], ""), it.get("text", "")
        n = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), min(len(ref), len(got)))
        shares.append(n / len(ref) if ref else 0.0)
    return sum(shares) / len(shares)


def _profile_blocks(t, config: str, arm: str, port: int, prof_dir: Path, cell: str,
                    extra=()) -> dict:
    """Device time per block for this arm, from one profiled sweep on its own
    server after the timed ones, divided by the cell's rules
    (`perfkit profile blocks`), and reported as metrics. Only a job of golden
    alone takes it (`_profile_read`), whose blocks scoping and the next segment
    rank. Empty, said loudly, if any step fails: the throughput is the score and
    does not depend on this."""
    rc = _profiled_sweep(t, config, arm, port, prof_dir, extra)
    phase = prof_dir / "decode_only"
    if rc != 0 or not phase.is_dir():
        say(f"{arm}/{config}: PROFILED SWEEP FAILED (rc={rc}, {phase} "
            f"{'exists' if phase.is_dir() else 'missing'}); no device time per block")
        return {}
    d = _phase_blocks(phase, cell, "decode", prof_dir.parent / "blocks.json", arm, config, t.name)
    if d is None:
        return {}
    metrics = _blocks_metrics(d)
    say(f"{arm}/{config}: device {d['attributed_ms']:.3f} ms of ops per step in "
        f"{len(d['blocks'])} block(s), {sum(r['ms'] for r in d['unclaimed']):.4f} ms unclaimed"
        + (" (host tracer on: step and program totals not reported)" if d.get("host_traced") else ""))
    return metrics


def _profiled_sweep(t, config: str, arm: str, port: int, prof_dir: Path, extra=(),
                    phases=(), log: Path | None = None, load: dict | None = None) -> int:
    """One sweep against the running server with the phased profiler armed.
    Its exit code. `extra` is more bench arguments, such as the config's
    acknowledgement (`_acks`).

    With `phases` and the server's `log`, the profiler is stopped as soon as
    every rank has written those captures (`_scored_phases`), so the rest of the
    sweep runs without the stall each further capture costs, and the sweep
    returns once no capture is still being written rather than after a fixed
    wait. Without them it waits PROFILE_FLUSH_S.

    With `load` (`_light_load`), this process sends the profiled run itself
    (`_light_sweep`); without it, `perfkit bench` runs the repository's client."""
    out = prof_dir.parent / "profile-bench.out"
    start = log.stat().st_size if log is not None and log.exists() else 0

    def read() -> dict:
        try:
            with open(log, errors="replace") as fh:
                fh.seek(start)
                return _capture_state(fh.read())
        except (OSError, TypeError):
            return {}

    done = threading.Event()

    def watch():
        while not done.wait(PROFILE_POLL_S):
            if _captured(read(), phases):
                if _stop_profile(port):
                    say(f"{arm}/{config}: {', '.join(phases)} written on every rank; "
                        f"the profiler is stopped for the rest of the sweep")
                return

    watcher = threading.Thread(target=watch, daemon=True) if phases and log else None
    if watcher is not None:
        watcher.start()
    with open(out, "w") as fh:
        if load is not None:
            rc = _light_sweep(port, load, fh)
        else:
            rc = subprocess.call(
                ["perfkit", "bench", "--target", t.name, "--config", config, "--repeats", "1",
                 "--tag", f"serve-e2e-{arm}-profile", "--against-port", str(port),
                 "--env", "CAPTURE_PROFILE=1", *extra], stdout=fh, stderr=subprocess.STDOUT)
    done.set()
    if watcher is None:
        time.sleep(PROFILE_FLUSH_S)
        return rc
    watcher.join()
    deadline = time.time() + PROFILE_FLUSH_S
    while _in_flight(read()) and time.time() < deadline:
        time.sleep(PROFILE_POLL_S)
    return rc


def _phase_blocks(phase: Path, cell: str, rules: str, out: Path, arm: str, config: str,
                 target: str = ""):
    """`perfkit profile blocks` on one phase capture, under the cell's rules for
    `rules` (decode or prefill), written to `out`. None, said loudly, if it
    cannot be read.

    By target and config when the caller has them: they name the frozen cell
    too (`_assert_same_cell` checked), and they also name a cell no result has
    named yet, which `--cell` refuses as unknown."""
    which = ["--target", target, "--config", config] if target else ["--cell", cell]
    err = out.with_suffix(".err")
    with open(out, "w") as fh, open(err, "w") as eh:
        # --own-windows: each plane over its own main program. A window taken
        # from the first rank can leave out the slowest plane, which is the
        # one a layer arm scores.
        rc = subprocess.call(["perfkit", "profile", "blocks", str(phase), *which,
                              "--phase", rules, "--own-windows", "--json"],
                             stdout=fh, stderr=eh)
    try:
        return json.loads(out.read_text())
    except (OSError, ValueError):
        say(f"{arm}/{config}: PROFILE BLOCKS FAILED on {phase.name} (rc={rc}); see {err}")
        return None


def _blocks_metrics(d: dict, prefix: str = "", label: str = "") -> dict:
    """A phase capture's op time, blocks and planes as metrics.

    Op time, which the host tracer does not touch, on the slowest plane: the
    step's critical path (see LAYER_SUFFIX). The mean over planes is
    `device_ops_ms_pooled`, and the blocks are means over planes too. The
    step's period and the programs' total do read high under the host tracer,
    by the launch delays it adds, so they are reported only from a capture that
    recorded no host activity."""
    planes = [p for p in d.get("per_plane") or [] if p.get("ops_ms")]
    metrics = {f"{prefix}device_ops_ms": max((p["ops_ms"] for p in planes),
                                              default=d["attributed_ms"]),
               f"{prefix}device_ops_ms_pooled": d["attributed_ms"]}
    if not d.get("host_traced"):
        metrics.update({f"{prefix}device_step_ms": d["period_ms"],
                        f"{prefix}device_busy_ms": d["busy_ms"]})
    for r in d["blocks"] + d["unclaimed"]:
        metrics[f"{prefix}block:{r['name']}"] = r["ms"]
    metrics.update(_rank_metrics(planes, prefix, label))
    return metrics


#: Ops named in the log, by how far apart the planes read on them.
RANK_OPS_SHOWN = 3


def _rank_metrics(planes: list, prefix: str = "", label: str = "") -> dict:
    """How far apart the planes read, as metrics, with the ops that differ
    most named in the log.

    AN OP LONG ON ONE RANK IS A CAUSE; A COLLECTIVE LONG ON THE OTHERS IS ITS
    COST. A rank that finished early waits for the late one at the next
    collective, so read pooled, the step shows a long collective and nothing
    that explains it. Per plane, the op that is long on exactly one rank is
    what made the others wait. Reported, never scored."""
    if len(planes) < 2:
        return {}
    totals = [p["ops_ms"] for p in planes]
    ops = set().union(*(p.get("by_op") or {} for p in planes))
    spread = []
    for op in ops:
        vals = [(p.get("by_op") or {}).get(op, 0.0) for p in planes]
        hi, lo = max(vals), min(vals)
        spread.append((hi - lo, op, planes[vals.index(hi)].get("rank"), hi,
                       planes[vals.index(lo)].get("rank"), lo))
    spread.sort(key=lambda x: -x[0])
    if spread and label:
        each = ", ".join(f"rank {p.get('rank')} {p['ops_ms']:.3f}" for p in planes)
        apart = "; ".join(f"{op} {hi:.3f} ms on rank {rh} vs {lo:.3f} on rank {rl}"
                          for _, op, rh, hi, rl, lo in spread[:RANK_OPS_SHOWN])
        say(f"{label}: op time per plane {each} ms; furthest apart: {apart}")
    return {f"{prefix}rank_planes": float(len(planes)),
            f"{prefix}rank_ops_ms_min": min(totals),
            f"{prefix}rank_op_spread_ms": spread[0][0] if spread else 0.0}


def _profile_phases(t, config: str, arm: str, port: int, prof_dir: Path, cell: str,
                    budget, decodes: int, extra=(), score_phase=None, decode_share=None,
                    load: dict | None = None) -> dict:
    """A layer arm's profiled sweep, read by phase.

    Op time per decode-only step and per step that carries a prefill chunk,
    each from its own phase capture, and the share of decode-only steps in the
    sweep, read from the server's step counters before and after it
    (`_step_shares`). The weighted op time is what the arm scores. With no prefill
    capture or no shares it is the decode-only op time, as on a 1k/1k cell.
    A config whose `score_phase` is "prefill" scores the prefill-carrying steps
    alone. Empty, said loudly, if the sweep failed."""
    before = _iteration_tokens(port)
    rc = _profiled_sweep(t, config, arm, port, prof_dir, extra, _scored_phases(score_phase),
                         prof_dir.parent / "server.log", load)
    after = _iteration_tokens(port)
    if rc != 0:
        say(f"{arm}/{config}: PROFILED SWEEP FAILED (rc={rc}); no device time")
        return {}
    shares = _step_shares(before, after, budget, decodes) if before and after else None

    def blocks(phase: Path, rules: str, out: Path):
        return _phase_blocks(phase, cell, rules, out, arm, config, t.name)

    return _score_phases(prof_dir, blocks, shares, budget, score_phase, f"{arm}/{config}",
                         decode_share=decode_share)


def _score_phases(prof_dir: Path, blocks, shares, budget, score_phase, label: str,
                  decode_share=None) -> dict:
    """A layer arm's metrics and score from its phase captures under
    `prof_dir`: op time per decode-only step and per step that carries a
    prefill chunk, weighed by `shares` as the config's `score_phase` says
    (`_phase_score`). `blocks(phase, rules, out)` divides one phase capture,
    or returns None, said loudly, when it cannot.

    A config's frozen `decode_share` replaces the arm's own share. Under data
    parallelism the number of decode-only steps a sweep takes moves from boot
    to boot: two copies of GLM-5.2's golden read op times 0.2% apart on each
    phase and 1.9% apart weighed by their own shares, because one sweep took
    3.9% more steps. The arm's own share is still reported."""
    metrics = {}
    decode = prof_dir / "decode_only"
    d = blocks(decode, "decode", prof_dir.parent / "blocks.json") if decode.is_dir() else None
    if d is not None:
        metrics.update(_blocks_metrics(d, label=f"{label} decode-only"))
    prefill = next((prof_dir / p for p in PREFILL_PHASES if (prof_dir / p).is_dir()), None)
    p = (blocks(prefill, "prefill", prof_dir.parent / f"blocks-{prefill.name}.json")
         if prefill is not None else None)
    if p is not None:
        metrics.update(_blocks_metrics(p, prefix="prefill_", label=f"{label} {prefill.name}"))
        metrics["device_ops_ms_prefill"] = metrics.pop("prefill_device_ops_ms")
    if shares is not None:
        metrics["step_share_decode"], metrics["steps_counted"], metrics["steps_with_prefill"] = shares
    weigh = shares
    if decode_share is not None:
        weigh = (float(decode_share),) + tuple(shares[1:] if shares else (0, 0))
    score, why = _phase_score(score_phase, budget, weigh, metrics.get("device_ops_ms"),
                              metrics.get("device_ops_ms_prefill"))
    if score is not None:
        metrics["device_ops_ms_weighted"] = score
    else:
        say(f"{label}: NO SCORE: {why}")
    return metrics


#: The cell driver a cells arm runs beside its server (`perfkit/buildkite_arm`).
CELLS_DRIVER = "cells.py"


def _cell_shape(phase: Path, cell: dict) -> tuple:
    """(whether the captured steps are the cell's shape, what they were), from
    the batch composition the phased profiler wrote beside each step. A decode
    cell's steps carry no prefill and its batch on each engine; a mixed cell's
    carry about its chunk and one more request. None when nothing was written."""
    stats = []
    for f in sorted(phase.rglob("batch_composition_stats_*.json")):
        try:
            stats.append(json.loads(f.read_text().splitlines()[0]))
        except (OSError, ValueError, IndexError):
            continue
    if not stats:
        return None, "no batch composition was written beside the steps"
    reqs = statistics.median(x.get("num_reqs", 0) for x in stats)
    pre = statistics.median(x.get("num_prefill_tokens", 0) for x in stats)
    b = int(cell["batch"])
    if cell.get("kind") == "mixed":
        want = int(cell.get("chunk") or 0)
        ok = reqs == b + 1 and (not want or abs(pre - want) <= 0.25 * want + b)
    else:
        ok = reqs == b and pre == 0
    return ok, f"{reqs:g} requests and {pre:g} prefill tokens a step over {len(stats)} steps"


def _score_cells(cells_dir: Path, cells: list, blocks, label: str) -> dict:
    """A cells arm's metrics and score. Each cell's captured steps are divided
    by the cell's rules (`blocks(phase, rules, out)`, None when it cannot) and
    timed on the slowest plane, and the cells are weighed by their steps in the
    workload (`perfkit/mix.py`).

    EVERY CELL OR NO SCORE. Two arms scored over different sets of cells make a
    ratio of nothing, so a cell with no capture leaves the arm unscored, said.
    Decode cells and mixed cells are also reported as the two classes a layer
    budget prices (`device_ops_ms`, `device_ops_ms_prefill`), with their blocks
    weighed the same way."""
    metrics, missing = {}, []
    classes = {"decode": {"w": 0.0, "ops": 0.0, "blocks": {}},
               "prefill": {"w": 0.0, "ops": 0.0, "blocks": {}}}
    for c in cells:
        here = cells_dir / c["name"]
        cls = "prefill" if c.get("kind") == "mixed" else "decode"
        phases = PREFILL_PHASES if cls == "prefill" else ("decode_only",)
        phase = next((here / ph for ph in phases if (here / ph).is_dir()), None)
        d = blocks(phase, cls, here / "blocks.json") if phase is not None else None
        if d is None:
            missing.append(c["name"])
            continue
        m = _blocks_metrics(d, label=f"{label} {c['name']}")
        ops, w = m["device_ops_ms"], float(c.get("steps") or 0)
        metrics[f"cell:{c['name']}:device_ops_ms"] = ops
        ok, what = _cell_shape(phase, c)
        if ok is not None:
            metrics[f"cell:{c['name']}:shape_ok"] = 1.0 if ok else 0.0
            if not ok:
                say(f"{label}: CELL {c['name']} RAN ANOTHER SHAPE: {what}, for "
                    f"{c['batch']} decodes" + (f" and a {c.get('chunk')}-token chunk"
                                               if cls == "prefill" else ""))
        k = classes[cls]
        k["w"] += w
        k["ops"] += w * ops
        for key, v in m.items():
            if key.startswith("block:"):
                k["blocks"][key] = k["blocks"].get(key, 0.0) + w * v
    if missing:
        say(f"{label}: NO SCORE: no capture of cell(s) {', '.join(missing)}; see "
            f"{cells_dir / 'cells.json'}")
        return metrics
    total = sum(k["w"] for k in classes.values())
    if not total:
        say(f"{label}: NO SCORE: the cells weigh no steps")
        return metrics
    metrics["device_ops_ms_weighted"] = sum(k["ops"] for k in classes.values()) / total
    for cls, prefix, ops_key in (("decode", "", "device_ops_ms"),
                                 ("prefill", "prefill_", "device_ops_ms_prefill")):
        k = classes[cls]
        if k["w"]:
            metrics[ops_key] = k["ops"] / k["w"]
            metrics.update({f"{prefix}{key}": v / k["w"] for key, v in k["blocks"].items()})
    metrics.update({"steps_counted": total, "steps_with_prefill": classes["prefill"]["w"],
                    "step_share_decode": classes["decode"]["w"] / total,
                    "cells_scored": float(len(cells))})
    return metrics


def _phase_score(score_phase, budget, shares, ops_d, ops_p):
    """(the op time per step a config's formula gives, why there is none).

    THE FORMULA IS THE CONFIG'S, never the capture's. "prefill" scores the
    steps that carry a prefill chunk, "decode" the decode-only steps, and the
    default weighs the two by their shares of the sweep. A config whose serve
    command sets no step budget cannot count the shares and scores decode-only
    steps, on every arm alike. Anything the formula needs and this capture
    lacks leaves the arm unscored rather than scored by another formula.

    The default's shares are the arm's own, so a tree whose scheduler takes
    more steps for the same sweep moves it with no change in any step's time:
    GLM-5.2's cut ran 74,161 steps on one nightly's tree and 49,440 on one a
    fortnight older. A search of one phase's kernel scores that phase."""
    if score_phase == "prefill":
        return (ops_p, "") if ops_p else (None, "this config scores the steps that carry a "
                                                "prefill chunk, and the profiler captured none")
    if score_phase == "decode" or budget is None:
        return (ops_d, "") if ops_d else (None, "the profiler captured no decode-only steps")
    if shares is None:
        return None, "the server's step counters could not be read, so the steps cannot be weighed"
    if not ops_p:
        return None, "the profiler captured no step that carried a prefill chunk"
    if not ops_d and shares[0] > 0:
        return None, "the profiler captured no decode-only steps"
    return shares[0] * (ops_d or 0.0) + (1 - shares[0]) * ops_p, ""


# --- the full model's KV cache ---------------------------------------------

def _kv_pin(cfg) -> dict | None:
    """The full model's KV allocation that a layer config pins its cut to: the
    cell's record (`perfkit layers kv`), or None, said, when there is none.

    THE CUT'S KV CACHE IS THE FULL MODEL'S. A cut model leaves its server most
    of the memory the full model's layers take, and vLLM gives it to the KV
    cache: the cut 397B allocated 38,441 blocks per engine against the full
    model's 987, and GLM-5.2's 16-chip cut 14,702 against its nightly's 443.
    Work that scales with the whole cache scales with that. And a model whose
    full cache caps its concurrency would run the cut at a concurrency the
    full model never reaches: GLM-5.2's nightly fits 49 of the 64 requests an
    engine is given. So the cut is given the full model's block count."""
    from perfkit import layers

    cell = cfg.get("cell", "")
    rec = layers.kv_blocks_for(cell)
    if not rec:
        # No full-model run is started for this: a campaign serves the full
        # config at a segment's close on a changed output and in its final
        # check, and golden's arm there records the cache (`_record_kv`).
        say(f"cell {cell} has no record of the full model's KV cache, so this cut keeps the "
            f"larger cache its dropped layers leave it, golden's arm and every candidate's alike. "
            f"The first full-model job on this cell records one"
            + ("; a cells config fixes each cell's batch and context, so the larger cache "
               "changes no shape it serves" if cfg.get("cells") else ""))
    return rec


def _cache_config_info(counters: Path):
    """(KV blocks, KV tokens) per engine from a step's counters that carry
    `vllm:cache_config_info`, the pair most of their readings give; None
    without one. Every vLLM exports it, on every topology, where a multi-host
    server's log prints neither.

    THE BLOCKS ARE A SUM. `num_gpu_blocks` is the sum over the engines the
    server fronts, and a reading has one line for each of them: GLM-5.2's
    sixteen engines pinned to 443 blocks each report 7088, and
    DeepSeek-V4-Pro's sixteen at 1777 each report 28432. Read as one engine's,
    that sum pinned DeepSeek-V4-Pro's cut to sixteen times its full model's
    cache. `kv_cache_size_tokens` is each engine's own. The block size is not
    always the cache's: DeepSeek-V4-Pro's says 16 where its cache's blocks
    hold 1024 tokens. So the tokens over the block size count the blocks only
    in a reading that has no sum."""
    try:
        text = counters.read_text(errors="replace")
    except OSError:
        return None

    def most(values):
        values = [v for v in values if v]
        return max(set(values), key=values.count) if values else None

    def number(row, key):
        try:
            return int(row.get(key, ""))
        except ValueError:
            return None

    pairs = []
    for reading in re.split(r"(?m)^# ", text):
        rows = [dict(re.findall(r'(\w+)="([^"]*)"', line)) for line in reading.splitlines()
                if line.startswith("vllm:cache_config_info{")]
        if not rows:
            continue
        engines = len({row.get("engine", "") for row in rows})
        total = most(number(row, "num_gpu_blocks") for row in rows)
        tokens = most(number(row, "kv_cache_size_tokens") for row in rows)
        size = most(number(row, "block_size") for row in rows)
        if total:
            blocks = total // engines
        elif tokens and size and tokens % size == 0:
            blocks = tokens // size
        else:
            continue
        pairs.append((blocks, tokens or (blocks * size if size else None)))
    return max(set(pairs), key=pairs.count) if pairs else None


def _kv_seen(log: Path, counters: Path | None = None) -> tuple:
    """(KV blocks, KV tokens) per engine the server allocated, or Nones: from
    the KV cache's own line in its log, which names one engine's blocks, else
    from its metrics when a reading of them has the cache config, else the
    capacity its log reports. A multi-host server's log has no such line, and
    its metrics are what it has."""
    from perfkit import layers

    try:
        text = log.read_text(errors="replace")
    except OSError:
        text = ""
    blocks, tokens = layers.kv_blocks_in_log(text), layers.kv_tokens_in_log(text)
    if blocks:
        return blocks, tokens
    info = _cache_config_info(counters) if counters is not None else None
    return info or (blocks, tokens)


def _kv_check(pin, seen: tuple, label: str) -> dict:
    """The KV cache the cut got, as metrics. Said loudly when a pin did not
    give it the full model's: by blocks when the server named them, else by
    tokens.

    THE SAME BLOCKS HOLD THE SAME TOKENS. A cut keeps every kind of layer its
    model has, so at the full model's block count it holds the full model's
    tokens: Qwen3.5-397B's cut at 987 blocks holds 1,139,491, as its full
    model does. When it holds other than its record says, the record's blocks
    are not the cut's, as when a sum over sixteen engines was recorded as one
    engine's."""
    blocks, tokens = seen
    out = {"kv_blocks_pinned": float(pin["blocks"]) if pin else 0.0}
    if blocks:
        out["kv_blocks"] = float(blocks)
    if tokens:
        out["kv_cache_tokens"] = float(tokens)
    if pin and blocks and blocks != pin["blocks"]:
        say(f"{label}: KV PIN MISSED: the cut allocated {blocks} KV cache blocks per engine and "
            f"the full model {pin['blocks']}, so its scheduler does not run the full model's batches")
    elif pin and blocks and pin.get("tokens") and tokens and \
            abs(tokens - pin["tokens"]) > 0.01 * pin["tokens"]:
        say(f"{label}: KV PIN MISSED: at the record's {blocks} KV cache blocks per engine the cut "
            f"holds {tokens} tokens of KV cache and the full model {pin['tokens']}, so the "
            f"record's blocks are not this cut's. Golden's next arm on the full config says what "
            f"the full model allocates")
    elif pin and not blocks and pin.get("tokens") and tokens and tokens != pin["tokens"]:
        say(f"{label}: KV PIN MISSED: the cut holds {tokens} tokens of KV cache and the full "
            f"model {pin['tokens']}, so its scheduler does not run the full model's batches")
    return out


def _record_kv(cfg, seen: tuple, arm: str, source: str = "") -> None:
    """Record the full model's KV allocation from golden's server, for the
    config's layer runs to pin their cut to. Golden's alone: a candidate that
    frees memory allocates more, and the pin is the baseline's. A record
    already there stands (`perfkit.layers.kv_record`)."""
    blocks, tokens = seen
    if not os.environ.get("PERFKIT_ARM_CONTROL") or not cfg.get("cell") or not blocks:
        return
    from perfkit import layers

    rec = layers.kv_record(cfg["cell"], blocks, tokens=tokens,
                           source=source or f"serve_e2e job {os.environ.get('PERFKIT_JOB_ID', '')}"
                                            f", arm {arm}", replace=False)
    say(f"{arm}/{cfg['config']}: the full model allocates {blocks} KV cache blocks per engine; "
        f"cell {cfg['cell']}'s layer runs pin their cut to {rec['blocks']}")


# --- arms measured on a Buildkite dev pipeline -------------------------------

#: How long past the step's own timeout an arm waits once its step has started:
#: the artifact upload after the command. The queue before it has its own
#: allowance (`perfkit.buildkite.QUEUE_S`).
BUILDKITE_UPLOAD_S = 1800
#: The files of the step that serves an arm, and where its commit carries them.
BUILDKITE_ARM_DIR = "scripts/forge"


def _remote_files(t, name, cfg, arm: str):
    """What a remote arm serves, as the files it writes over its tree: the
    benchmark config, the target's own copy, so a tree cannot change what it is
    served with; a layer config's cut of it; the step's script and fingerprint
    client (`perfkit/buildkite_arm`) and the numerics suite. Also the patches
    the tree needs, and the step's environment. Buildkite and Cloud DevKit run
    the same files, so an arm reads the same wherever it is measured.

    A layer config serves the cut through `--hf-overrides` and the loader's
    layer filter, which streams only the layers kept (`perfkit.layer_filter`,
    appended to the loader file): neither place has disk for a cut checkpoint.
    Its KV cache is pinned to the full model's when the cell has a record.

    A full config that sets `profile` takes the phased capture, as a local arm
    of it does. Otherwise the step profiles only if the serving repo's own
    config script turns the profiler on: DeepSeek-V4's nightly ones do, and a
    config that does not would leave a run that needed the capture without it.
    Either way the capture is read only from a job of golden alone
    (`_profile_read`), and any other arm's step ends its profiled sweep as soon
    as it starts."""
    from types import SimpleNamespace

    from perfkit import layer_filter, layers
    from perfkit import paths as pk_paths

    config = cfg["config"]
    configs = "scripts/vllm/benchmarking/configs"
    base = Path(t.configs_dir) / f"{config}.sh"
    files = {f"{configs}/{config}.sh": (base.read_bytes(), False)}
    served, edits, plan, pin = config, {}, None, None
    if cfg.get("layers"):
        meta = t.config_meta(config)
        _refuse_speculative(meta, name)
        full = layers.read_config(cfg.get("weights") or layers.resolve_checkpoint(
            "", t.env(), uri=(t.config_env(config) or {}).get("MODEL_URI", "")))
        want = cfg.get("layers")
        plan = layers.plan(full, periods=int(cfg.get("periods", 1)),
                           layers=None if want == "auto" else int(want))
        if not plan.complete:
            raise RuntimeError(f"{plan.describe()}: this cut leaves out a kind of layer the "
                               f"model has")
        pin = _kv_pin(cfg)
        served = f"{config}-forge-layers"
        files[f"{configs}/{served}.sh"] = (_buildkite_layer_config(
            config, layers.hf_overrides(full, plan.layers), pin, cfg.get("cut_env"),
            _cells_settings(meta) if cfg.get("cells") else
            _phase_settings(meta, cfg.get("score_phase")),
            extra="--enable-prefix-caching" if cfg.get("cells") else ""),
            False)
        edits = {layer_filter.LOADER: layer_filter.append}
        say(f"{arm}/{name}: {plan.describe()}"
            + (f"; KV cache pinned to {pin['blocks']} blocks per engine" if pin else ""))
    here = Path(pk_paths.FORGE) / "perfkit" / "buildkite_arm"
    files[f"{BUILDKITE_ARM_DIR}/run_arm.sh"] = ((here / "run_arm.sh").read_bytes(), True)
    files[f"{BUILDKITE_ARM_DIR}/capture.py"] = ((here / "capture.py").read_bytes(), False)
    files[f"{BUILDKITE_ARM_DIR}/suite.json"] = (
        (pk_paths.EVALS_DIR / "numerics_suites" / "default.json").read_bytes(), False)
    files[f"{BUILDKITE_ARM_DIR}/suite-long.json"] = (
        (pk_paths.EVALS_DIR / "numerics_suites" / f"{LONG_SUITE}.json").read_bytes(), False)
    if plan is not None and cfg.get("cells"):
        files[f"{BUILDKITE_ARM_DIR}/{CELLS_DRIVER}"] = ((here / CELLS_DRIVER).read_bytes(), False)
        files[f"{BUILDKITE_ARM_DIR}/cells.json"] = (json.dumps(cfg["cells"]).encode(), False)
    repeats, warmup = int(cfg.get("repeats", REPEATS)), int(cfg.get("warmup", WARMUP_SWEEPS))
    if plan is not None:
        warmup = int(cfg.get("warmup", 0))
    env = {"BENCHMARK_WARMUP_RUNS": str(warmup), "SKIP_DB_UPLOAD": "1",
           "FORGE_TARGET": t.name, "FORGE_CONFIG": config,
           **({"FORGE_SCORED_PHASES": " ".join(_scored_phases(cfg.get("score_phase")))}
              if plan is not None else {}),
           **({"FORGE_CELLS": "1"} if plan is not None and cfg.get("cells") else {}),
           **_remote_eval_env(_eval_rule(name, cfg)),
           **({"CAPTURE_PROFILE": "1", "USE_PHASED_PROFILER": "true"}
              if plan is None and cfg.get("profile") else {}),
           # The server is armed as golden's is, by forge or by the serving
           # repo's own config, and the profiled run that follows the first
           # sweep is ended as soon as it starts (run_arm.sh).
           **({"FORGE_SCORED_PHASES": "none"} if plan is None and not _profile_read() else {})}
    # A layer run scores its capture, not its sweeps; another sweep would
    # capture again into the directory the score is read from.
    extra = 0 if plan is not None else max(repeats, RECORD_SAMPLES) - 1
    return SimpleNamespace(config=config, files=files, edits=edits, served=served,
                           plan=plan, pin=pin, repeats=repeats, extra=extra, env=env)


def _measure_buildkite(handle, name, cfg):
    """This arm on its target's Buildkite dev pipeline (`perfkit/buildkite.py`),
    scored here from what the build wrote, by the code a local arm is scored by.

    The arm's tree is committed with forge's files over it (`_remote_files`)
    and the pipeline that builds the image and runs the step."""
    from perfkit import buildkite, targets
    from perfkit import scorer as pk_scorer

    t = targets.load(cfg.get("target") or TARGET)
    s = buildkite.settings(t.spec, t.name)
    if _eval_rule(name, cfg) is not None:
        s = dataclasses.replace(s, timeout_minutes=s.timeout_minutes + ACCURACY_MINUTES)
    arm = os.environ.get("PERFKIT_ARM", "arm")
    job = os.environ.get("PERFKIT_JOB_ID") or f"job-{int(time.time())}"
    art = _arm_dir(arm, name)
    shutil.rmtree(art, ignore_errors=True)
    art.mkdir(parents=True)
    r = _remote_files(t, name, cfg, arm)
    results = f"{s.results_root.rstrip('/')}/{r.served}"
    command = " ".join(x for x in (s.runner, f"bash {BUILDKITE_ARM_DIR}/run_arm.sh", r.served,
                                   results, str(r.extra)) if x)
    r.files[".buildkite/pipeline_dev.yml"] = (buildkite.pipeline_yaml(
        s, label=f"forge {job}: {arm} on {name}", command=command, env=r.env).encode(), False)

    tree = pk_scorer.git_tree(handle.root) or handle.tree
    sha = buildkite.commit(handle.root, r.files, edits=r.edits,
                           message=f"[dev] forge {job}: arm {arm} on {name}")
    branch = buildkite.branch(s, job, arm)
    buildkite.push(handle.root, s, sha, branch)
    number = None
    try:
        b = buildkite.find_build(s, sha, branch)
        if b is not None:
            say(f"{arm}/{name}: build {b['number']} of commit {sha[:12]} is already "
                f"{b.get('state')}; waiting on it rather than building again")
        else:
            b = buildkite.trigger(s, sha, branch, f"forge {job}: {arm} on {name}")
        number = b["number"]
        say(f"{arm}/{name}: commit {sha[:12]} on {branch}, build {number} {b.get('web_url', '')}")
        try:
            b = buildkite.wait(s, number, timeout_s=s.timeout_minutes * 60 + BUILDKITE_UPLOAD_S,
                               say=lambda m: say(f"{arm}/{name}: {m}"))
        except (RuntimeError, SystemExit, KeyboardInterrupt):
            # Given up on, or stopped: a build nobody will read holds a shared
            # queue for nothing. A stopped waiter's build was once left to run
            # for the same job run again to wait on (`find_build`), when a
            # harness restart killed every job. A restart leaves jobs running
            # now, so a waiter is stopped only on purpose, by its run's cancel
            # or its campaign's stop, and a stopped GLM-5.2 job left its golden
            # arm holding the one v7x-16 agent CI shares. `find_build` still
            # waits on a build that outlived a waiter killed outright.
            say(f"{arm}/{name}: cancelling build {number}: "
                + ("cancelled" if buildkite.cancel(s, number) else "the API refused"))
            raise
        j = buildkite.job(b)
        if j is not None and j.get("started_at"):
            buildkite.fetch(s, number, j["id"], art)
        if j is None or j.get("state") != "passed":
            bad = next((x for x in b.get("jobs") or [] if x.get("state") in ("failed", "timed_out")),
                       j)
            tail = buildkite.log_tail(s, number, bad["id"]) if bad and bad.get("id") else ""
            logs = sorted(art.glob("perf_eval_results/*/server.log"))
            cause = buildkite.first_error(logs[0].read_text(errors="replace")) if logs else ""
            raise RuntimeError(
                f"build {number} {b.get('state')}, its arm step "
                f"{j.get('state') if j else 'never created'}; {b.get('web_url', '')}\n"
                + (f"first error in the server log: {cause}\n" if cause else "")
                + tail[-2000:])
    finally:
        buildkite.delete_branch(handle.root, s, branch)
    return _score_remote(art, r, cfg, arm=arm, name=name, label=f"build {number}",
                         where=f"buildkite/{s.pipeline}#{number}/{sha[:12]}",
                         source=f"buildkite {s.pipeline} build {number}", tree=tree,
                         root=handle.root)


def _measure_cdk(handle, name, cfg):
    """This arm on the shared TPU cluster through Cloud DevKit (`perfkit/cdk_arm.py`),
    scored here from what its job wrote, by the code a Buildkite arm is scored by.

    The arm's tree is committed with forge's files over it as a Buildkite
    arm's is (`_remote_files`), with the step that copies its results into the
    job's outputs, and mapped into every pod of a JobSet on the target's slice.
    A slice of several hosts serves the config across them, through Ray, the
    way the repository's own nightly does."""
    from perfkit import buildkite, cdk_arm, targets
    from perfkit import scorer as pk_scorer

    t = targets.load(cfg.get("target") or TARGET)
    s = cdk_arm.settings(t.spec, t.name)
    if _eval_rule(name, cfg) is not None:
        s = dataclasses.replace(s, timeout_minutes=s.timeout_minutes + ACCURACY_MINUTES)
    arm = os.environ.get("PERFKIT_ARM", "arm")
    job = os.environ.get("PERFKIT_JOB_ID") or f"job-{int(time.time())}"
    art = _arm_dir(arm, name)
    shutil.rmtree(art, ignore_errors=True)
    art.mkdir(parents=True)
    r = _remote_files(t, name, cfg, arm)
    r.files[cdk_arm.STEP_PATH] = (cdk_arm.STEP_SCRIPT.encode(), True)
    tree = pk_scorer.git_tree(handle.root) or handle.tree
    sha = buildkite.commit(handle.root, r.files, edits=r.edits,
                           message=f"[dev] forge {job}: arm {arm} on {name}")
    job_id = cdk_arm.run(s, root=handle.root, sha=sha, job=job, arm=arm, served=r.served,
                         extra=r.extra, env=r.env, art=art,
                         say=lambda m: say(f"{arm}/{name}: {m}"))
    return _score_remote(art, r, cfg, arm=arm, name=name, label=f"cdk job {job_id}",
                         where=f"cdk/{job_id}/{sha[:12]}", source=f"cdk job {job_id}",
                         tree=tree, root=handle.root)


def _score_remote(art: Path, r, cfg, *, arm: str, name: str, label: str, where: str,
                  source: str, tree: str, root) -> dict:
    """A remote arm's result from what its job wrote under
    `art/perf_eval_results/<served>`, by the code every remote arm is scored
    by. `label` names the job in what is said (`build 457`, `cdk job j-...`),
    `where` begins the dispatched string, and `source` is what a full-model
    KV record says it came from."""
    out = art / "perf_eval_results" / r.served
    if not out.is_dir():
        raise RuntimeError(f"{label} passed and left nothing under perf_eval_results/"
                           f"{r.served}; see {art}")
    plan, pin = r.plan, r.pin
    num_out, nolp_out = out / "numerics.out", out / "numerics-nologprobs.out"
    batch_out = out / "numerics-batch.out"
    if plan is not None:
        metrics = _buildkite_layer_metrics(out, cfg, f"{arm}/{name}", root)
        metrics.update(_kv_check(pin, _kv_seen(out / "server.log", out / "iteration_tokens.log"),
                                 f"{arm}/{name}"))
        ops = metrics.get("device_ops_ms_weighted")
        if not ops:
            raise RuntimeError(f"{label}'s capture gave no op time to score; see {out}")
        missing = _missing_blocks(cfg["cell"], metrics,
                                  "prefill" if cfg.get("score_phase") == "prefill" else "decode")
        metrics["layer_blocks_missing"] = len(missing)
        metrics.update({"layers_kept": plan.layers, "layers_total": plan.total,
                        "layer_mix_matches": 1.0 if plan.mix_matches else 0.0})
        _keep_metrics(out, metrics)
        latency_us = ops * 1000.0
        dispatched = f"{where}/{r.config}/L{plan.layers}"
        say(f"{arm}/{name}: device ops {ops:.4f} ms per step: decode-only "
            f"{metrics.get('device_ops_ms', 0):.4f} ms, with a prefill chunk "
            f"{metrics.get('device_ops_ms_prefill', 0):.4f} ms")
    else:
        tputs, metrics = _buildkite_sweeps(out)
        if not tputs or max(tputs) <= 0:
            raise RuntimeError(f"{label} reported no throughput; see {out}")
        median = statistics.median(tputs)
        reported, metrics["timing_rounds"] = _timed(tputs)
        if reported <= 0:
            raise RuntimeError(f"{label}'s score would be a timed sweep at zero throughput "
                               f"({tputs}): the server stopped serving during them; see {out}")
        if len(tputs) > 1:
            metrics["sweep_spread"] = (max(tputs) - min(tputs)) / median
        _record_kv(cfg, _kv_seen(out / "server.log", out / "iteration_tokens.log"), arm,
                   source=source)
        latency_us = 1e6 / reported
        dispatched = f"{where}/{r.config}"
        say(f"{arm}/{name}: sweeps {tputs} median {median:.1f} {METRIC}"
            + (f"; scored on {reported:.1f}, the lower median" if reported != median else ""))
        # The step writes eval.log only when it ran the eval tasks
        # (`_remote_eval_env`), so an arm that ran none is not reported as
        # having lost its scores.
        if (out / "eval.log").exists():
            metrics.update(_accuracy_remote(out, f"{arm}/{name}"))
    agreement = _prefix_agreement(batch_out, nolp_out)
    if agreement is not None:
        metrics["batched_prefix_agreement"] = agreement
    checksum = _combined(_fingerprint(num_out), _fingerprint(nolp_out))
    if not checksum:
        say(f"{arm}/{name}: NO OUTPUT FINGERPRINT in {label}; see {out}")
    return {"metrics": metrics, "latency_us": latency_us, "worst_error": 0.0,
            "checksum": checksum, "first_tokens": _first_tokens(num_out),
            "dispatched": dispatched, "binary_hash": "sha256:" + tree}


def _buildkite_layer_config(config: str, overrides: dict, pin, cut_env=None,
                            settings=None, extra: str = "") -> bytes:
    """The benchmark config a layer arm's step serves: the frozen config, cut
    to its leading layers, with the phased profiler on.

    The overrides are compact JSON with every quote escaped, inside a double-
    quoted assignment: run_benchmarks.sh splits EXTRA_SERVE_ARGS on spaces, so
    the JSON must hold none and reach the server as one argument. So is
    `settings`, the `--additional-config` that places the captures
    (`_phase_settings`).

    `cut_env` is the config's own variables for the cut, set after the frozen
    config is sourced, for a model whose cut cannot be served with its full
    model's settings, such as a KV pool its cut cannot allocate beside the
    arrays the full model's layers would have held."""
    def word(d: dict) -> str:
        return json.dumps(d, separators=(",", ":")).replace('"', '\\"')

    args = f"--hf-overrides {word(overrides)} --profiler-config.max_iterations {LAYER_STEPS}"
    # run_benchmarks.sh runs the config's measured sweep before its profile run,
    # and a layer arm reads only the profile run: one prompt is enough for the
    # sweep nobody reads. The profile run keeps its full wave
    # (PROFILE_NUM_PROMPTS, the concurrency).
    sets_after = "NUM_PROMPTS=1\n"
    if settings:
        args += f" --additional-config {word(settings)}"
    if pin:
        args += f" --num-gpu-blocks-override {int(pin['blocks'])}"
    if extra:
        args += f" {extra}"
    sets = "".join(f"{k}={json.dumps(str(v))}\n" for k, v in sorted((cut_env or {}).items()))
    return (
        "#!/bin/bash\n"
        "# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh\n"
        f"# Written by forge (scorers/serve_e2e.py): {config} on its leading layers,\n"
        "# with the phased profiler on, for one scorer arm.\n"
        f"# shellcheck source={config}.sh\n"
        f'source "$(dirname "${{BASH_SOURCE[0]}}")/{config}.sh"\n'
        f"{sets}"
        f"{sets_after}"
        f'EXTRA_SERVE_ARGS="${{EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }}{args}"\n'
        "export USE_PHASED_PROFILER=true\n"
        "CAPTURE_PROFILE=1\n").encode()


def _last_counters(log: Path):
    """The last complete reading in a step's iteration_tokens.log, in
    `_iteration_tokens`'s shape, or None."""
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return None
    readings = [r for r in text.split("# ") if r.strip()]
    for r in reversed(readings):
        counts = _parse_counters(r)
        if counts and float("inf") in counts:
            return counts
    return None


def _buildkite_layer_metrics(out: Path, cfg, label: str, root: Path) -> dict:
    """A layer arm's metrics from its build's results: each phase capture
    divided by the cell's rules here, and the step shares from the server's
    counters as the sweep ended, the step budget from the config the build
    served (`config.json`)."""
    from perfkit import blockrules

    try:
        conf = json.loads((out / "config.json").read_text())
    except (OSError, ValueError):
        conf = {}
    budget = conf.get("max_num_batched_tokens")
    seqs = conf.get("max_num_seqs")
    dp = int(conf.get("data_parallelism") or 1)
    try:
        per_engine = -(-int(str(conf.get("concurrency_options")).split()[0]) // dp)
    except (TypeError, ValueError, IndexError):
        per_engine = None
    decodes = min(x for x in (seqs, per_engine) if x) if (seqs or per_engine) else 0
    if budget is None:
        say(f"{label}: the build's config.json names no max_num_batched_tokens, so the steps "
            f"that carried a prefill chunk cannot be counted")
    after = _last_counters(out / "iteration_tokens.log")
    # The server was started for this sweep, so its counters began at zero.
    shares = _step_shares({}, after, budget, decodes) if after else None

    def blocks(phase: Path, rules: str, dest: Path):
        try:
            d = blockrules.derive(phase, cfg["cell"], phase=rules, repo=root, own_windows=True)
        except Exception as exc:  # noqa: BLE001 -- said, and the phase is left out
            say(f"{label}: PROFILE BLOCKS FAILED on {phase.name}: {exc}")
            return None
        dest.write_text(json.dumps(d, indent=1, default=str))
        return d

    if cfg.get("cells"):
        return _score_cells(out / "cells", cfg["cells"], blocks, label)
    return _score_phases(out / "profile", blocks, shares, budget, cfg.get("score_phase"), label,
                         decode_share=cfg.get("decode_share"))


def _buildkite_sweeps(out: Path):
    """(each timed sweep's throughput, the secondary metrics' medians) from a
    full-config arm's build: its first sweep and the extra ones run against the
    same server (`rep<N>/`). Not its warm-up sweeps: run_benchmarks.sh saves
    each beside the timed one as `<point>.warmup<N>.json`, which a bare glob of
    the point's name also matches."""
    docs = []
    timed = re.compile(r"isl\d+_osl\d+_c\d+\.json")
    # rep2 before rep10, in the order they ran.
    found = sorted(out.glob("isl*_osl*_c*.json")) + sorted(
        out.glob("rep*/isl*_osl*_c*.json"), key=lambda f: (len(f.parent.name), f.parent.name))
    for f in (f for f in found if timed.fullmatch(f.name)):
        try:
            docs.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    tputs = [float(d.get("output_throughput") or 0.0) for d in docs]
    metrics = {}
    for k in SECONDARY_METRICS:
        vals = [float(d[k]) for d in docs if isinstance(d.get(k), (int, float))]
        if vals:
            metrics[k] = statistics.median(vals)
    return tputs, metrics



def _bench_failed(rc: int, every: list, warmup: int = WARMUP_SWEEPS) -> str:
    """Why an arm's bench produced nothing to score, or "" if it did.

    Omitted rather than zeroed: `scorer_arm` drops a config that raises, the
    driver refuses an output missing a frozen config, and the job is VOID. That
    says "nothing was learned here", which is true. A zero would say the arm is
    infinitely slow, which nobody measured -- and sweeps that all came back at
    zero are what a dead engine looks like from the client.
    """
    tputs = every[warmup:] if len(every) > warmup else every
    if rc != 0 or not tputs:
        return f"bench rc={rc} with {len(every)} sweep(s)"
    if max(tputs) <= 0:
        return f"bench returned {len(every)} sweep(s) at zero throughput; the server died"
    return ""


def _fingerprint(num_out: Path) -> str:
    """The served model's output fingerprint, if the capture wrote one.

    Two arms reporting the same fingerprint produced the same tokens. That is
    the clone check, and on a served campaign it is also the correctness check:
    a candidate whose fingerprint differs from golden's is judged by its tree's
    eval tasks.

    THE `digest` FIELD AND NOTHING ELSE. This used to take the first 16-64 hex
    token on any line of the capture's output, which a full commit sha, a hash
    in a path or a hex id in an error message would all have satisfied -- and a
    constant string there makes every arm look like it produced the same output.
    Empty when there is no digest, which the harness treats as "the check did
    not run" and not as a pass.
    """
    try:
        text = num_out.read_text()
    except OSError:
        return ""
    digest = None
    try:
        doc = json.loads(text)
        if isinstance(doc, dict):
            digest = doc.get("digest")
    except ValueError:
        m = re.search(r'"digest"\s*:\s*"([0-9a-f]{8,64})"', text)
        digest = m.group(1) if m else None
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{8,64}", digest):
        return ""
    return "sha256:" + digest


def self_check() -> int:
    """What this file can check without a device: that it agrees with the target.

    Every config it offers has to be one the target declares, or a campaign
    freezes a config name no `perfkit bench` will accept and finds out at its
    first measurement.
    """
    from perfkit import targets

    ok = True
    for name, cfg in CONFIGS.items():
        tname = cfg.get("target") or TARGET
        try:
            declared = set(targets.load(tname).configs())
        except SystemExit as exc:
            print(f"  {name:34s} -> target {tname}: {exc}")
            ok = False
            continue
        good = cfg["config"] in declared
        print(f"  {name:34s} -> {tname}/{cfg['config']:26s} {'ok' if good else 'NOT DECLARED'}")
        ok = ok and good
    print("self-check PASSED" if ok else "self-check FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_check())
