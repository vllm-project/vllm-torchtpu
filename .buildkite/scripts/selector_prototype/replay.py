"""Replay merged commits through the prototype selector and report the result.

PROTOTYPE. For each squash-merged PR on main, diff it against its parent and
compare what PR 2a's rules alone would skip with what rules plus recordings
would skip.

    python3 replay.py --table table.json --last 30 --out replay.md
    python3 replay.py --table table.json --commits SHA SHA ...

Caveat printed in the report: the recordings were taken after these PRs
merged, so a function a PR added is already in the table. In production the
recording always predates the PR, and new code falls back to the rules.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from changed_funcs import build  # noqa: E402
from decide import CHIPS, REPO_ROOT, Table, decide, rules, rules_only, summarize  # noqa: E402

DOCS_ONLY = ("docs/",)


def docs_only(paths: list[str]) -> bool:
    """CI already skips these builds (bootstrap's docs-only check)."""
    return bool(paths) and all(
        p.startswith(DOCS_ONLY) or p.endswith(".md") or p == "LICENSE" for p in paths
    )


def merged_commits(repo: pathlib.Path, rev: str, last: int) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--first-parent", "--format=%H%x09%s",
         f"-n{last * 3}", rev],
        capture_output=True, text=True, check=True,
    ).stdout
    shas = [line.split("\t", 1)[0] for line in out.splitlines()
            if "[skip ci]" not in line]
    return shas[:last]


def subject(repo: pathlib.Path, sha: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%s", sha],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def replay_one(repo: pathlib.Path, sha: str, table: Table, config) -> dict:
    queries = build(repo, f"{sha}^", sha)
    paths = [q.path for q in queries]
    result = {"sha": sha[:10], "subject": subject(repo, sha), "files": len(paths)}
    if docs_only(paths):
        result["docs_only"] = True
        return result
    decisions = decide(queries, table, config)
    summary = summarize(decisions)
    needed = rules_only(queries, config)
    rule_skips = [] if needed is None else sorted(set(CHIPS) - needed)
    result.update(
        docs_only=False,
        rules_steps_skipped=len(rule_skips),
        rules_chips_skipped=sum(CHIPS[s] for s in rule_skips),
        steps_skipped=summary["steps_skipped"],
        chips_skipped=summary["chips_skipped"],
        skipped=summary["skipped"],
        by_recording=[s for s, d in decisions.items() if d.used_recording],
        reasons={s: d.reasons for s, d in decisions.items()},
    )
    return result


def report(results: list[dict], table: Table) -> str:
    counted = [r for r in results if not r["docs_only"]]
    total = sum(CHIPS.values())
    lines = [
        "# Selector prototype: replay of merged PRs",
        "",
        f"Recording commit `{table.commit[:10]}`; recorded steps: "
        + ", ".join(
            f"`{s}` ({'healthy' if not r['health']['thin'] else 'thin'})"
            for s, r in table.rows.items()
        ),
        "",
        f"{len(counted)} PRs replayed ({len(results) - len(counted)} docs-only PRs "
        "left out: CI already skips them). Full PR build: "
        f"{len(CHIPS)} TPU steps, {total} chips.",
        "",
    ]
    if counted:
        rules_chips = sum(r["rules_chips_skipped"] for r in counted)
        rec_chips = sum(r["chips_skipped"] for r in counted)
        lines += [
            "| | Rules only (PR 2a) | Rules + recordings |",
            "|---|---|---|",
            f"| PRs that skip at least one step | "
            f"{sum(1 for r in counted if r['rules_steps_skipped'])} | "
            f"{sum(1 for r in counted if r['steps_skipped'])} |",
            f"| Chips skipped (sum over PRs) | {rules_chips} | {rec_chips} |",
            f"| Share of PR TPU chips skipped | {rules_chips / (total * len(counted)):.0%} "
            f"| {rec_chips / (total * len(counted)):.0%} |",
            "",
        ]
    lines += [
        "| PR | Files | Rules only: chips skipped | + recordings: chips skipped | Skipped by recordings |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        title = r["subject"][:70].replace("|", "/")
        if r["docs_only"]:
            lines.append(f"| `{r['sha']}` {title} | {r['files']} | docs-only | docs-only | |")
            continue
        extra = sorted(set(r["skipped"]) & set(r["by_recording"]))
        lines.append(
            f"| `{r['sha']}` {title} | {r['files']} | {r['rules_chips_skipped']} "
            f"| {r['chips_skipped']} | {', '.join(f'`{s}`' for s in extra)} |"
        )
    lines += [
        "",
        "Caveat: the recordings were taken after these PRs merged, so code a PR "
        "added is already in the table. In production the recording predates the "
        "PR and new code falls back to the rules.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", type=pathlib.Path, required=True)
    ap.add_argument("--repo", type=pathlib.Path, default=REPO_ROOT)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--last", type=int, help="the last N PRs merged before the recording")
    group.add_argument("--commits", nargs="+")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("replay.md"))
    ap.add_argument("--json-out", type=pathlib.Path)
    args = ap.parse_args(argv)

    config = rules.load_config(REPO_ROOT)
    table = Table.load(args.table, args.repo)
    shas = args.commits or merged_commits(args.repo, table.commit, args.last)
    results = []
    for sha in shas:
        r = replay_one(args.repo, sha, table, config)
        results.append(r)
        state = "docs-only" if r["docs_only"] else (
            f"rules skip {r['rules_chips_skipped']} chips, "
            f"+recordings skip {r['chips_skipped']}"
        )
        print(f"{r['sha']} {r['subject'][:60]}: {state}")
    args.out.write_text(report(results, table))
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
