#!/usr/bin/env python3
"""rescore.py - Re-score lm_eval samples_*.jsonl WITHOUT re-running the model.

Reads the raw generations (`resps`) stored in lm_eval sample logs, re-applies
the answer-extraction filters, recomputes exact_match + stderr for every
subtask and group, and writes a new sweep.json in the same schema as
run_sweep_orchestrator.

Nothing in the orchestrator or runners is touched; this is a standalone tool.
No TPU, vLLM or lm_eval is required (lm_eval is not imported).

Usage
-----
  python3 rescore.py <results_dir> [--config CONFIG.yaml]
                     [--out-root DIR_OR_GS] [--name NAME] [--dry-run]

  <results_dir>  local dir or gs:// prefix containing samples_*.jsonl and
                 sweep.json (searched recursively, so a VM results/<ts>/ dir
                 also works). Every sub-folder holding samples_*.jsonl
                 (e.g. eval_logs/<model>/) is rescored. Each lm_eval run
                 (sub-folder + file timestamp) is matched to the sweep.json
                 trial that produced it.

Where the filters come from (--config wins when given)
------------------------------------------------------
  1. --config <yaml>     evaluation.task_overrides.<group>.filter_list
  2. input sweep.json    metadata.evaluation.task_overrides.<group>.filter_list
  3. neither             scores are carried over unchanged (reported)

An override key K applies to task T when T == K or T starts with K + "_";
the longest matching key wins (so "mmlu_llama" covers
mmlu_llama_formal_logic, etc.).

exact_match options (ignore_case, ignore_punctuation, regexes_to_ignore) are
resolved per task, later wins: defaults (True, True, []) < the task's config
in that run's results_*.json < --ignore-case/--ignore-punctuation <
task_overrides metric_list. So tasks with different metrics (e.g. mmlu_pro
and gsm8k) can be rescored together.

Validation built in
-------------------
Before rescoring, the stored `filtered_resps` are re-compared against `target`
with the chosen exact_match options. If that doesn't reproduce the stored
per-sample exact_match, the metric options are wrong and the script aborts.

Output
------
  <out-root>/<name>/sweep.json
      (default name: <UTC now>_refer_to_<original id>)
  plus a local rescore_report.json with before/after per task and changed
  samples.

Example
-------
    cd /mnt/pd/xujie/vllm-torchtpu/experiments
    python3 rescore.py gs://cloud-tpu-inference-eval-staging/incoming/xujiezhang/20260922_225312

By default it:
  • takes the filters from the task_overrides stored in the input sweep.json,
  • names the output folder <UTC date>_<time>_refer_to_<original id>,
  • puts that folder in the parent of the input folder.
"""
import argparse
import copy
import datetime as dt
import glob
import json
import math
import os
import re
import string
import subprocess
import sys
import tempfile
from collections import OrderedDict, defaultdict

import yaml

SAMPLE_RE = re.compile(
    r"^samples_(?P<task>.+?)_(?P<ts>\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
EM_OPTION_KEYS = ("ignore_case", "ignore_punctuation", "regexes_to_ignore")


# -------------------------------------------------------------------------- io
def is_gs(p):
    """True if `p` is a gs:// URI (handled with gcloud, not local file IO)."""
    return p.startswith("gs://")


def run(cmd):
    """Echoes and runs `cmd` (argv list); raises on a non-zero exit."""
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch_input(results_dir, workdir):
    """Returns a local directory containing the run's eval outputs.

    If `results_dir` is a local path, it is returned as an absolute path and
    nothing is copied. If it is a gs:// prefix, the whole tree (eval_logs/,
    sweep.json, ...) is mirrored into `<workdir>/input` via
    `gcloud storage rsync -r`, and that local directory is returned.
    """
    if not is_gs(results_dir):
        return os.path.abspath(results_dir)
    dst = os.path.join(workdir, "input")
    os.makedirs(dst, exist_ok=True)
    run(["gcloud", "storage", "rsync", "-r", results_dir.rstrip("/"), dst])
    return dst


def find_inputs(root):
    """Groups sample logs into lm_eval runs.

    Returns (runs sorted by timestamp, path of the sweep.json or None).
    Every sub-folder holding samples_*.jsonl is scanned. One lm_eval
    invocation writes all of its samples_<task>_<ts>.jsonl plus one
    results_<ts>.json with the same <ts> into one folder, so (folder, ts)
    identifies a run.
    """
    runs = {}
    pattern = os.path.join(root, "**", "samples_*.jsonl")
    for path in sorted(glob.glob(pattern, recursive=True)):
        m = SAMPLE_RE.match(os.path.basename(path))
        if not m:
            print(f"  WARNING: skipping unrecognised file name {path}")
            continue
        folder, ts = os.path.dirname(path), m.group("ts")
        entry = runs.setdefault(
            (folder, ts), {
                "label": f"{os.path.relpath(folder, root)}@{ts}",
                "ts": ts,
                "samples": OrderedDict(),
                "results_json": None,
            })
        entry["samples"][m.group("task")] = path
        results_json = os.path.join(folder, f"results_{ts}.json")
        if os.path.isfile(results_json):
            entry["results_json"] = results_json
    sweeps = sorted(
        glob.glob(os.path.join(root, "**", "sweep.json"), recursive=True))
    runs = sorted(runs.values(), key=lambda x: x["ts"])
    return runs, (sweeps[-1] if sweeps else None)


def assign_runs(experiments, runs):
    """Maps each sweep.json trial to the lm_eval run that produced it.

    Returns a list parallel to `experiments` (None where no run matched).
    A run is a candidate for a trial when every task in the run appears in
    the trial's metrics as <task>_sample_len. Trials are matched newest-first
    to the newest unused candidate, so leftover older runs (e.g. a failed
    attempt before --resume) are skipped.
    """
    assigned, used = [None] * len(experiments), set()
    for i in reversed(range(len(experiments))):
        metrics = experiments[i].get("metrics") or {}
        has_len = any(k.endswith("_sample_len") for k in metrics)
        if not has_len and len(experiments) > 1:
            continue  # e.g. a profile trial in mode: both
        for j in reversed(range(len(runs))):
            if j in used:
                continue
            if has_len and not all(f"{t}_sample_len" in metrics
                                   for t in runs[j]["samples"]):
                continue
            assigned[i] = runs[j]
            used.add(j)
            break
    for j, run_info in enumerate(runs):
        if j not in used:
            print("  WARNING: sample logs not matched to any sweep.json "
                  f"trial, ignored: {run_info['label']}")
    return assigned


# --------------------------------------------------------------------- filters
def f_regex(values, step):
    """Applies one "regex" filter step to the candidate answers of a sample.

    Mirrors lm_eval.filters.extraction.RegexFilter so rescoring reproduces
    lm_eval output.

    Input:
      values: list[str] - candidate answers for one doc. For the first step
        of a chain this is the raw model generations (resps[0] in
        samples_*.jsonl); for later steps it is the output of the previous
        step.
      step: dict - one entry of a task filter chain (filter_list[*].filter
        in the task YAML / override config), e.g.
        {"function": "regex", "regex_pattern": "answer is ([A-J])",
         "group_select": 0, "fallback": "[invalid]"}
        regex_pattern (required) is the pattern searched with re.findall;
        group_select (default 0) picks which match to keep when several are
        found (0 = first, -1 = last); fallback (default "[invalid]") is
        returned when nothing matches.

    Output:
      list[str] with the same length and order as `values`: for each value,
      the selected match with surrounding whitespace stripped, or `fallback`
      if the pattern does not match. If the pattern has several capture
      groups, findall yields tuples and the first non-empty group is kept
      (fallback if all are empty).
    """
    rx = re.compile(step["regex_pattern"])
    group_select = step.get("group_select", 0)
    fallback = step.get("fallback", "[invalid]")
    out = []
    for v in values:
        match = rx.findall(v)
        if match:
            match = match[group_select]
            # same semantics as lm_eval RegexFilter
            if isinstance(match, tuple):
                match = [m for m in match if m]
                match = match[0] if match else fallback
            out.append(match.strip())
        else:
            out.append(fallback)
    return out


FILTERS = {
    "regex": f_regex,
    "take_first": lambda vs, _s: vs[:1],
    "remove_whitespace": lambda vs, _s: [v.removeprefix(" ") for v in vs],
    "lowercase": lambda vs, _s: [v.lower() for v in vs],
    "uppercase": lambda vs, _s: [v.upper() for v in vs],
}


def apply_filter_chain(resps, steps):
    """Runs one sample's generations through a filter chain.

    resps: list[str] (resps[0] of a samples_*.jsonl row). steps: the chain's
    list of {"function": ..., ...} dicts, applied in order via FILTERS.
    Returns the first value left after the last step ("" if none), i.e. what
    lm_eval stores as filtered_resps[0].
    """
    values = list(resps)
    for step in steps:
        fn = step["function"]
        if fn not in FILTERS:
            raise ValueError(f"unsupported filter function '{fn}' "
                             f"(supported: {sorted(FILTERS)})")
        values = FILTERS[fn](values, step)
    return values[0] if values else ""


# ---------------------------------------------------------------------- metric
def exact_match(pred, ref, opts):
    """Scores one prediction against the target: 1.0 if equal, else 0.0.

    Same normalisation (and order) as lm_eval's exact_match: strip
    `regexes_to_ignore`, then lowercase if `ignore_case`, then drop
    punctuation if `ignore_punctuation`. `opts` is the dict produced by
    metric_opts().
    """
    for rx in opts["regexes_to_ignore"]:
        pred, ref = re.sub(rx, "", pred), re.sub(rx, "", ref)
    if opts["ignore_case"]:
        pred, ref = pred.lower(), ref.lower()
    if opts["ignore_punctuation"]:
        pred, ref = pred.translate(PUNCT_TABLE), ref.translate(PUNCT_TABLE)
    return 1.0 if pred == ref else 0.0


def mean_stderr(xs):
    """Standard error of the mean (sample std / sqrt(n)); 0.0 if n < 2.

    Same as lm_eval.api.metrics.mean_stderr, i.e. what lm_eval reports as
    *_stderr.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mu = sum(xs) / n
    return math.sqrt(sum((x - mu)**2 for x in xs) / (n - 1) / n)


def pooled_stderr(stderrs, sizes):
    """lm_eval.api.metrics.pooled_sample_stderr (for group aggregation)."""
    total, k = sum(sizes), len(sizes)
    if total - k <= 0:
        return 0.0
    pooled = sum((n - 1) * s * s * n for n, s in zip(sizes, stderrs))
    return math.sqrt(pooled / (total - k) / total)


# ---------------------------------------------------------------------- config
def load_overrides(config_path, sweep):
    """Picks the task_overrides to re-apply; returns (overrides, source).

    --config wins outright: when given, its evaluation.task_overrides is used
    as-is, even if empty (no fallback to sweep.json). The chosen overrides are
    also recorded in the output sweep.json metadata so a rescored run
    documents what was applied.

    Input:
      config_path: path of the --config sweep YAML, or None.
      sweep: the parsed input sweep.json; without --config its
        metadata.evaluation.task_overrides (what the original run used) is
        taken.

    Output:
      overrides: dict {task_or_group_key: {"filter_list": [...],
        "metric_list": [...]}}, same shape as evaluation.task_overrides in a
        sweep config. Keys are matched to tasks by override_for (longest
        prefix wins). {} means every task is carried over unchanged.
      source: str describing where the overrides came from (for the log
        line).
    """
    if config_path:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        ov = (cfg.get("evaluation") or {}).get("task_overrides") or {}
        return ov, f"--config {config_path}"
    ev = (sweep.get("metadata") or {}).get("evaluation") or {}
    ov = ev.get("task_overrides") or {}
    return ov, "input sweep.json metadata.evaluation.task_overrides"


def override_for(task, overrides):
    """Returns the task_overrides entry that applies to `task`, or None.

    A key K matches when task == K or task starts with K + "_"; the longest
    match wins (so "mmlu_llama" covers mmlu_llama_formal_logic and beats a
    shorter "mmlu" key).
    """
    keys = [k for k in overrides if task == k or task.startswith(k + "_")]
    return overrides[max(keys, key=len)] if keys else None


def _apply_metric_list(metric_list, opts):
    """Copies exact_match options from a lm_eval metric_list into opts."""
    for m in metric_list or []:
        if m.get("metric") == "exact_match":
            for k in EM_OPTION_KEYS:
                if k in m:
                    opts[k] = m[k]


def metric_opts(override, args, task_cfg=None):
    """Resolves the exact_match options for one task.

    Keys: ignore_case, ignore_punctuation, regexes_to_ignore. Layered, later
    wins: defaults (True, True, []) < the task's metric_list as recorded in
    that run's results_*.json (`task_cfg`) < --ignore-case /
    --ignore-punctuation flags < metric_list of the task_overrides entry
    (`override`). Returns the dict exact_match() uses.
    """
    opts = {
        "ignore_case": True,
        "ignore_punctuation": True,
        "regexes_to_ignore": [],
    }
    # as recorded by lm_eval for this run
    _apply_metric_list((task_cfg or {}).get("metric_list"), opts)
    if args.ignore_case is not None:
        opts["ignore_case"] = args.ignore_case
    if args.ignore_punctuation is not None:
        opts["ignore_punctuation"] = args.ignore_punctuation
    _apply_metric_list((override or {}).get("metric_list"), opts)
    return opts


# --------------------------------------------------------------------- scoring
def _stored_pred(row):
    """The answer lm_eval extracted for this sample (filtered_resps[0])."""
    return str((row.get("filtered_resps") or [""])[0])


def _stored_em(row):
    """The exact_match lm_eval stored for this sample (0.0 if missing)."""
    return float(row.get("exact_match") or 0)


def score_task(task, path, overrides, args, report, task_cfg=None):
    """Returns {filter_name: [per-sample scores]}; fills report[task]."""
    ov = override_for(task, overrides)
    filter_list = (ov or {}).get("filter_list") or []
    chains = OrderedDict((f["name"], f["filter"]) for f in filter_list)
    opts = metric_opts(ov, args, task_cfg)

    rows = [json.loads(line) for line in open(path) if line.strip()]
    by_filter = defaultdict(list)
    for r in rows:
        by_filter[r.get("filter", "none")].append(r)

    out, info = OrderedDict(), OrderedDict()
    for fname, recs in by_filter.items():
        # --- validation: stored filtered_resps + our metric must reproduce
        # the stored exact_match
        mismatch = []
        for r in recs:
            em = exact_match(_stored_pred(r), str(r["target"]), opts)
            if em != _stored_em(r):
                mismatch.append(r["doc_id"])
        if mismatch:
            sys.exit(f"ABORT: exact_match options {opts} do not reproduce "
                     f"stored scores for {task}/{fname} (doc_ids "
                     f"{mismatch[:10]}). Adjust --ignore-case/"
                     "--ignore-punctuation.")

        old = [_stored_em(r) for r in recs]
        if fname in chains:
            new, changed = [], []
            for r in recs:
                pred = apply_filter_chain(r["resps"][0], chains[fname])
                s = exact_match(pred, str(r["target"]), opts)
                new.append(s)
                prev = _stored_pred(r)
                if pred != prev or s != _stored_em(r):
                    changed.append({
                        "doc_id": r["doc_id"],
                        "target": r["target"],
                        "old_pred": prev,
                        "new_pred": pred,
                        "old_em": _stored_em(r),
                        "new_em": s,
                    })
            mode = "rescored"
        else:
            new, changed = old, []
            mode = "carried (no filter_list for this task/filter)"
        out[fname] = new
        info[fname] = {
            "mode": mode,
            "n": len(new),
            "old_exact_match": round(sum(old) / len(old), 6) if old else None,
            "new_exact_match": round(sum(new) / len(new), 6) if new else None,
            "changed_samples": changed,
        }

    # filters declared in the override but absent from the logs: build them
    # from resps
    for fname, chain in chains.items():
        if fname in out:
            continue
        base = next(iter(by_filter.values()))
        new = [
            exact_match(apply_filter_chain(r["resps"][0], chain),
                        str(r["target"]), opts) for r in base
        ]
        out[fname] = new
        info[fname] = {
            "mode": "new filter",
            "n": len(new),
            "old_exact_match": None,
            "new_exact_match": round(sum(new) / len(new), 6),
            "changed_samples": [],
        }
    report[task] = info
    return out


def group_members(sweep_exp, leaves, results_json):
    """Map group -> direct children. Prefers lm_eval's own group_subtasks."""
    if results_json:
        gs = json.load(open(results_json)).get("group_subtasks") or {}
        if gs:
            return {g: list(ch) for g, ch in gs.items() if ch}
    # fallback: the trial's requested task aggregates every leaf in the trial
    top = str((sweep_exp.get("parameters") or {}).get("tasks", ""))
    return {top: sorted(leaves)} if top and top not in leaves else {}


def aggregate(name, members, leaf_stats, cache):
    """Returns {"n", "acc", "stderr"} for a task or group (None if empty).

    Leaves come straight from `leaf_stats`; a group recurses over
    members[name] and combines the children: n = total samples, acc =
    sample-size-weighted mean (lm_eval weight_by_size), stderr =
    pooled_stderr. `cache` memoises results and marks which names were
    visited. None means the name has no scored leaves.
    """
    if name in cache:
        return cache[name]
    if name in leaf_stats:
        cache[name] = leaf_stats[name]
        return cache[name]
    kids = [
        aggregate(c, members, leaf_stats, cache)
        for c in members.get(name, [])
    ]
    kids = [k for k in kids if k and k["n"]]
    if not kids:
        cache[name] = None
        return None
    sizes = [k["n"] for k in kids]
    n = sum(sizes)
    acc = sum(k["acc"] * k["n"] for k in kids) / n
    stderr = pooled_stderr([k["stderr"] for k in kids], sizes)
    cache[name] = {"n": n, "acc": acc, "stderr": stderr}
    return cache[name]


# ------------------------------------------------------------------------ main
def main():
    """CLI entry point: rescores one results dir and writes a new sweep.json.

    Steps: fetch the input (local or gs://) -> group sample logs into lm_eval
    runs -> pick the task_overrides -> rescore every leaf task of every run
    -> rebuild group metrics -> write <workdir>/<name>/sweep.json +
    rescore_report.json, print a per-run old/new table, and upload/copy
    sweep.json to <out-root>/<name>/ unless --dry-run.
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--config",
                    help="sweep config YAML whose evaluation.task_overrides "
                    "to apply")
    ap.add_argument("--out-root",
                    help="where to create <name>/ "
                    "(default: parent of results_dir)")
    ap.add_argument("--name",
                    help="output folder / sweep_id "
                    "(default: <UTC ts>_refer_to_<orig id>)")
    ap.add_argument("--local-dir",
                    help="local staging dir (default: a temp dir)")
    ap.add_argument("--ignore-case",
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    help="force exact_match ignore_case "
                    "(default: from results_*.json, else True)")
    ap.add_argument("--ignore-punctuation",
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    help="force exact_match ignore_punctuation "
                    "(default: from results_*.json, else True)")
    ap.add_argument("--dry-run",
                    action="store_true",
                    help="write locally, do not upload")
    args = ap.parse_args()

    workdir = args.local_dir or tempfile.mkdtemp(prefix="rescore_")
    os.makedirs(workdir, exist_ok=True)
    print(f">>> [RESCORE] input  : {args.results_dir}")
    root = fetch_input(args.results_dir, workdir)
    runs, sweep_path = find_inputs(root)
    if not runs:
        sys.exit("ABORT: no samples_*.jsonl found")
    if not sweep_path:
        sys.exit("ABORT: no sweep.json found next to the samples")
    sweep = json.load(open(sweep_path))
    src = args.results_dir.rstrip("/")
    orig_id = sweep.get("sweep_id") or os.path.basename(src)

    now = dt.datetime.now(dt.timezone.utc)
    name = args.name or f"{now:%Y%m%d_%H%M%S}_refer_to_{orig_id}"
    out_root = args.out_root or src.rsplit("/", 1)[0]
    # filters/metrics to re-apply: --config YAML if given, else the input
    # sweep.json
    overrides, ov_src = load_overrides(args.config, sweep)
    n_logs = sum(len(x["samples"]) for x in runs)
    print(f">>> [RESCORE] runs   : {len(runs)} lm_eval run(s), "
          f"{n_logs} task logs")
    for x in runs:
        print(f"      - {x['label']}: {len(x['samples'])} task logs | "
              f"results_json: {bool(x['results_json'])}")
    print(f">>> [RESCORE] filters: {ov_src} -> "
          f"groups {list(overrides) or 'NONE'}")

    # score every leaf task of every run
    report, run_scores = OrderedDict(), {}
    for x in runs:
        run_report = report.setdefault(x["label"], OrderedDict())
        task_cfgs = {}
        if x["results_json"]:
            results = json.load(open(x["results_json"]))
            task_cfgs = results.get("configs") or {}
        scores = OrderedDict()
        for task, path in x["samples"].items():
            scores[task] = score_task(task, path, overrides, args, run_report,
                                      task_cfgs.get(task))
        run_scores[x["label"]] = scores

    new = copy.deepcopy(sweep)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    cmd = "python3 rescore.py " + " ".join(sys.argv[1:])
    summary = []
    experiments = new.get("experiments", [])
    for exp, run_info in zip(experiments, assign_runs(experiments, runs)):
        metrics = exp.setdefault("metrics", {})
        if run_info is None:
            print("  WARNING: no sample logs for trial "
                  f"{exp.get('experiment_id')}; metrics left unchanged")
            filters, leaf_scores, exp_leaves, members = [], {}, [], {}
        else:
            leaf_scores = run_scores[run_info["label"]]
            exp_leaves = list(leaf_scores)
            filters = sorted({f for t in exp_leaves for f in leaf_scores[t]})
            members = group_members(exp, set(exp_leaves),
                                    run_info["results_json"])
        for fname in filters:
            leaf_stats = {}
            for t in exp_leaves:
                xs = leaf_scores[t].get(fname)
                if xs is None:
                    continue
                leaf_stats[t] = {
                    "n": len(xs),
                    "acc": sum(xs) / len(xs),
                    "stderr": mean_stderr(xs),
                }
            cache = {}
            names = list(leaf_stats)
            names += [g for g in members if g not in leaf_stats]
            for t in names:
                st = aggregate(t, members, leaf_stats, cache)
                if not st:
                    continue
                k_acc = f"{t}_exact_match,{fname}"
                k_se = f"{t}_exact_match_stderr,{fname}"
                old = metrics.get(k_acc)
                metrics[k_acc] = f"{st['acc']:.4f}"
                metrics[k_se] = f"{st['stderr']:.4f}"
                metrics[f"{t}_sample_len"] = f"{st['n']:.4f}"
                is_leaf = t in leaf_stats
                summary.append((run_info["label"], t, fname, old,
                                metrics[k_acc], is_leaf))
            suffix = "_sample_len"
            unresolved = [
                k[:-len(suffix)] for k in metrics
                if k.endswith(suffix) and k[:-len(suffix)] not in cache
            ]
            if unresolved:
                print("  WARNING: groups left UNCHANGED (membership unknown "
                      f"without results_*.json): {unresolved}")
        exp_id = str(exp.get("experiment_id", ""))
        exp["experiment_id"] = exp_id.replace(orig_id, name)
        exp["timestamp"] = stamp
        exp["reproduction_command"] = cmd

    new["sweep_id"] = name
    new["created_at"] = new["completed_at"] = stamp
    md = new.setdefault("metadata", {})
    md["description"] = (
        f"RESCORE of {orig_id}: filters re-applied to stored lm_eval "
        "samples; model NOT re-run. Eval_* timing metrics are from the "
        "original run.")
    md.setdefault("evaluation", {})["task_overrides"] = overrides
    md["rescore"] = {
        "source_sweep_id": str(orig_id),
        "source_path": args.results_dir,
        "filter_source": ov_src,
        "rescored_at": stamp,
        "command": cmd,
    }

    # write + upload
    out_dir = os.path.join(workdir, name)
    os.makedirs(out_dir, exist_ok=True)
    local_sweep = os.path.join(out_dir, "sweep.json")
    local_report = os.path.join(workdir, "rescore_report.json")
    with open(local_sweep, "w") as fh:
        json.dump(new, fh, indent=2)
    with open(local_report, "w") as fh:
        json.dump(report, fh, indent=2)

    current = None
    summary.sort(key=lambda x: (x[0], not x[5], x[1]))
    for label, t, f, old, nw, is_leaf in summary:
        if label != current:
            current = label
            print(f"\n[run] {label}")
            print("%-48s %-13s %8s -> %-8s" % ("task", "filter", "old", "new"))
        flag = "" if old == nw else "   <-- changed"
        print("%-48s %-13s %8s -> %-8s%s" %
              (("" if is_leaf else "[group] ") + t, f, old, nw, flag))
    n_changed = sum(
        len(v["changed_samples"]) for run_report in report.values()
        for t in run_report.values() for v in t.values())
    print(f"\n>>> [RESCORE] samples whose extracted answer or score changed: "
          f"{n_changed}")
    print(f">>> [RESCORE] local sweep.json : {local_sweep}")
    print(f">>> [RESCORE] local report     : {local_report}")

    dest = f"{out_root.rstrip('/')}/{name}/sweep.json"
    if args.dry_run:
        print(f">>> [RESCORE] dry run; would upload to {dest}")
    elif is_gs(out_root):
        run(["gcloud", "storage", "cp", local_sweep, dest])
        print(f">>> [RESCORE] uploaded: {dest}")
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as fh:
            json.dump(new, fh, indent=2)
        print(f">>> [RESCORE] written: {dest}")


if __name__ == "__main__":
    main()
