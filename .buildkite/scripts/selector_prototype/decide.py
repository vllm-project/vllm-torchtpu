"""Decide which PR TPU steps a diff needs, from recordings first and rules second.

PROTOTYPE. The decision follows vllm-project/ci-infra's ci-selector
(Apache-2.0, buildkite/ci_selector/ci_selector/decide.py at 04dd673a92): a step
runs if its recording ran a changed function; it is skipped only on a healthy
row that shows it did not; anything a recording cannot answer goes to the
rules (here, PR 2a's select_pr_tests.py), and a file no rule covers runs
everything.

    python3 decide.py --table table.json --base origin/main [--head HEAD]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import select_pr_tests as rules  # noqa: E402
from changed_funcs import FileQuery, build, names_of, read_at  # noqa: E402

REPO_ROOT = HERE.parents[2]

# v7x chips each selectable step holds, read off pipeline_tests.yml and
# pipeline_integration.yml on Nov 22 (parallelism x chips per job). Sums to
# the 68 chips a full PR build uses.
CHIPS = {
    "unit_tests_general_pr": 3,
    "unit_tests_multichip_pr": 4,
    "unit_tests_spec_decode_pr": 3,
    "unit_tests_spec_decode_multichip_pr": 4,
    "unit_tests_omni_pr": 1,
    "integration_qwen35_pcp4_dp4_disagg_correctness": 4,
    "integration_qwen35_pcp4_2tp2dp_disagg_correctness": 4,
    "integration_qwen35_pcp4tp1_tp4dcp2_disagg_correctness": 4,
    "presubmit_disagg_single_host": 4,
    "integration_qwen35_apc_correctness": 4,
    "integration_qwen35_gpc_global_dram_pool": 4,
    "integration_compilation_cache_e2e": 1,
    "integration_k3_pruned75_apc_offload_correctness": 4,
    "vllm_torchtpu_multihost_presubmit": 8,
    "integration_qwen35_pcp8_2tp4dp_disagg_correctness": 8,
    "integration_qwen35_gpc_correctness": 8,
}


class Table:
    """The recorded table, plus the source at the commit it was recorded at."""

    def __init__(self, data: dict, repo: pathlib.Path):
        self.commit: str = data["commit"]
        self.rows: dict = data["rows"]
        self.repo = repo
        self._names: dict[str, frozenset[str] | None] = {}

    @classmethod
    def load(cls, path: pathlib.Path, repo: pathlib.Path) -> Table:
        return cls(json.loads(path.read_text()), repo)

    def usable_row(self, step: str) -> tuple[dict | None, str]:
        """(row, why not) - the row only when it may authorize a skip."""
        row = self.rows.get(step)
        if row is None:
            return None, "no recording for this step"
        if row["health"]["thin"]:
            return None, "recording too weak: " + "; ".join(row["health"]["thin"])
        return row, ""

    def names_at_record(self, path: str) -> frozenset[str] | None:
        """Every name the file had when the recording ran; None if absent."""
        if path not in self._names:
            source = read_at(self.repo, self.commit, path)
            try:
                self._names[path] = None if source is None else names_of(source, path)
            except Exception:
                self._names[path] = None
        return self._names[path]


@dataclass
class StepDecision:
    step: str
    run: bool = False
    reasons: list[str] = field(default_factory=list)
    used_recording: bool = False


def _rule_verdict(query: FileQuery, step: str, config) -> tuple[bool, str]:
    """(run, why) from PR 2a's rules. A rename is judged on both paths."""
    needed: frozenset[str] | None = frozenset()
    names = []
    for path in filter(None, (query.path, query.old_path)):
        steps, rule, _phase = rules.classify(path, config, REPO_ROOT)
        names.append(rule)
        if steps is None:
            needed = None
        elif needed is not None:
            needed |= steps
    rule = " / ".join(sorted(set(names)))
    if needed is None:
        return True, f"rule '{rule}': no rule covers it, so every step runs"
    if step in needed:
        return True, f"rule '{rule}' needs this step"
    return False, f"rule '{rule}' does not need this step"


def _recording_verdict(query: FileQuery, row: dict, table: Table) -> tuple[bool | None, str]:
    """(run, why) from the step's recording; run=None when it cannot answer."""
    recorded = set(row["functions"].get(query.path, ()))
    if query.old_path:
        recorded |= set(row["functions"].get(query.old_path, ()))
    if not recorded:
        if table.names_at_record(query.old_path or query.path) is None:
            return None, "file did not exist when recorded"
        return False, "step never ran any code in this file"
    hits = sorted(query.function_names & recorded)
    if hits:
        shown = ", ".join(hits[:3]) + (" ..." if len(hits) > 3 else "")
        return True, f"recording ran changed function(s): {shown}"
    if query.import_time:
        return True, "changes code that runs on import, and the step imports this file"
    known = table.names_at_record(query.path) or frozenset()
    if new := sorted(query.names - known):
        return True, f"adds new code ({', '.join(new[:2])}) to a file the step imports"
    return False, "recording never ran the changed functions"


def decide(queries: list[FileQuery], table: Table, config) -> dict[str, StepDecision]:
    decisions: dict[str, StepDecision] = {}
    for step in sorted(config.selectable):
        d = StepDecision(step)
        decisions[step] = d
        if not queries:
            d.run, d.reasons = True, ["empty diff: every step runs"]
            continue
        row, why_not = table.usable_row(step)
        for query in queries:
            verdict: bool | None = None
            if query.inert and query.answerable:
                # Only comments, docstrings or annotations changed.
                continue
            if row is not None and query.answerable:
                verdict, why = _recording_verdict(query, row, table)
                if verdict is not None:
                    d.used_recording = True
            if verdict is None:
                verdict, why = _rule_verdict(query, step, config)
                if row is None and query.answerable:
                    why += f" ({why_not})"
                elif query.python and query.in_scope and query.failed:
                    why += f" (names unreadable: {query.failed})"
            if verdict:
                d.run = True
                d.reasons.append(f"{query.path}: {why}")
        if not d.run:
            d.reasons.append("no changed file needs it")
    return decisions


def rules_only(queries: list[FileQuery], config) -> frozenset[str] | None:
    """What PR 2a's rules alone would run (None = every step), for comparison."""
    files = [p for q in queries for p in (q.path, q.old_path) if p]
    return rules.select(files, config, mode="phase2", repo_root=REPO_ROOT).needed


def summarize(decisions: dict[str, StepDecision]) -> dict:
    skipped = [s for s, d in decisions.items() if not d.run]
    return {
        "steps_total": len(decisions),
        "steps_skipped": len(skipped),
        "chips_total": sum(CHIPS.get(s, 0) for s in decisions),
        "chips_skipped": sum(CHIPS.get(s, 0) for s in skipped),
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", type=pathlib.Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--repo", type=pathlib.Path, default=REPO_ROOT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    config = rules.load_config(REPO_ROOT)
    table = Table.load(args.table, args.repo)
    queries = build(args.repo, args.base, args.head)
    decisions = decide(queries, table, config)
    summary = summarize(decisions)
    if args.json:
        print(json.dumps({
            "summary": summary,
            "steps": {s: {"run": d.run, "reasons": d.reasons} for s, d in decisions.items()},
        }, indent=2))
        return 0
    for step, d in decisions.items():
        print(f"{'RUN ' if d.run else 'SKIP'} {step}")
        for reason in d.reasons[:3]:
            print(f"       {reason}")
    print(
        f"Skips {summary['steps_skipped']} of {summary['steps_total']} steps, "
        f"{summary['chips_skipped']} of {summary['chips_total']} chips."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
