#!/usr/bin/env python3
"""Find the CODEOWNERS reviewers for a set of changed files.

Typical use, from anywhere inside the repo:

    # Who should review my current branch?
    scripts/find_owners.py

    # Who should review PR #123?
    scripts/find_owners.py --pr 123

    # Who owns these paths?
    scripts/find_owners.py src/vllm_torchtpu/kernels/ docs/index.md

    # Just the handles, e.g. to pipe into `gh pr edit --add-reviewer`
    scripts/find_owners.py --handles

    # Request their review (on the current branch's PR, or --pr 123)
    scripts/find_owners.py --request
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import NoReturn

CODEOWNERS_CANDIDATES = (
    ".github/CODEOWNERS",
    "CODEOWNERS",
    "docs/CODEOWNERS",
)


# --------------------------------------------------------------------------
# shell helpers
# --------------------------------------------------------------------------
def run(cmd: list[str], cwd: str | None = None) -> str | None:
    """Run a command, returning stdout, or None if it failed."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def run_checked(cmd: list[str]) -> tuple[bool, str]:
    """Run a command, returning (succeeded, output-or-error-message)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return False, f"{cmd[0]} not found"
    if proc.returncode != 0:
        return False, (proc.stderr.strip() or proc.stdout.strip())
    return True, proc.stdout.strip()


def repo_root() -> str:
    root = run(["git", "rev-parse", "--show-toplevel"])
    if not root:
        die("not inside a git repository (and no paths were given)")
    return root


def die(msg: str) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------
# CODEOWNERS parsing and matching
# --------------------------------------------------------------------------
class Rule:
    """One CODEOWNERS line: a path pattern plus the owners of that path."""

    def __init__(self, pattern: str, owners: list[str], lineno: int):
        self.pattern = pattern
        self.owners = owners
        self.lineno = lineno
        self.regex = compile_pattern(pattern)

    def matches(self, path: str) -> bool:
        return self.regex.match(path) is not None


def compile_pattern(pattern: str) -> re.Pattern:
    """Translate a gitignore-style CODEOWNERS pattern into a regex.

    Follows the rules GitHub documents: `/` anchors to the repo root, a
    pattern containing a slash is anchored, `*` stops at a path separator,
    `**` spans separators, and matching a directory matches everything
    beneath it -- except for a rule ending in `*`; see below.
    """
    p = pattern
    directory_rule = p.endswith("/")
    if directory_rule:
        p = p[:-1]
    anchored = p.startswith("/") or "/" in p
    p = p.lstrip("/")

    out: list[str] = []
    i, n = 0, len(p)
    while i < n:
        c = p[i]
        if c == "*":
            if p[i : i + 3] == "**/":
                out.append("(?:.*/)?")
                i += 3
            elif p[i : i + 2] == "**":
                out.append(".*")
                i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1

    prefix = "" if anchored else "(?:.*/)?"
    # A directory rule owns everything under it; a rule whose last segment
    # ends in `*` does not. This is the one place CODEOWNERS departs from
    # gitignore: `docs/*` matches `docs/getting-started.md` but not
    # `docs/build-app/troubleshooting.md`. Bare `*` and `*.py` still reach any
    # depth, via the unanchored prefix.
    globbed_leaf = not directory_rule and p.rsplit("/", 1)[-1].endswith("*")
    subtree = "" if globbed_leaf else "(?:/.*)?"
    return re.compile("^" + prefix + "".join(out) + subtree + "$")


def load_rules(root: str, override: str | None = None) -> tuple[list[Rule], str]:
    if override:
        with open(override, encoding="utf-8") as f:
            return parse_codeowners(f.read()), override
    for rel in CODEOWNERS_CANDIDATES:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return parse_codeowners(f.read()), rel
    die(f"no CODEOWNERS file found (looked in {', '.join(CODEOWNERS_CANDIDATES)})")


def parse_codeowners(content: str) -> list[Rule]:
    rules = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        owners = [o for o in parts[1:] if o.startswith("@") or "@" in o]
        # A pattern with no owners is kept, not skipped: CODEOWNERS uses it to
        # un-own a subdirectory that an earlier rule claimed. Since the last
        # match wins, such a rule leaves the path with no owner at all.
        rules.append(Rule(parts[0], owners, lineno))
    return rules


def owners_for(path: str, rules: list[Rule]) -> Rule | None:
    """Return the last matching rule, which is the one GitHub applies."""
    match = None
    for rule in rules:
        if rule.matches(path):
            match = rule
    return match


# --------------------------------------------------------------------------
# figuring out which files changed
# --------------------------------------------------------------------------
def resolve_base(root: str, base: str | None) -> str:
    if base:
        return base
    for candidate in ("origin/main", "upstream/main", "main"):
        if run(["git", "rev-parse", "--verify", "--quiet", candidate], cwd=root):
            return candidate
    die("could not find a base branch; pass --base <ref>")


def changed_files(root: str, base: str) -> tuple[list[str], str]:
    """Files changed on this branch: committed, staged, unstaged, untracked."""
    merge_base = run(["git", "merge-base", "HEAD", base], cwd=root) or base
    files = set()

    # Deleted files are kept: GitHub still applies ownership to a path you
    # delete, and `gh pr diff --name-only` lists them, so dropping them here
    # would make local runs disagree with --pr runs.
    diff = run(["git", "diff", "--name-only", merge_base], cwd=root)
    if diff:
        files.update(diff.splitlines())

    untracked = run(["git", "ls-files", "--others", "--exclude-standard"], cwd=root)
    if untracked:
        files.update(untracked.splitlines())

    return sorted(files), f"vs {base}"


def pr_files(pr: str, repo: str | None) -> tuple[list[str], str]:
    cmd = ["gh", "pr", "diff", str(pr), "--name-only"]
    if repo:
        cmd += ["--repo", repo]
    out = run(cmd)
    if out is None:
        die(f"could not read PR #{pr}; is `gh` installed and authenticated?")
    return sorted(set(out.splitlines())), f"PR #{pr}"


def expand_paths(root: str, paths: list[str]) -> list[str]:
    """Turn user-supplied paths into repo-relative file paths."""
    files = set()
    for raw in paths:
        abs_path = os.path.abspath(raw)
        rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
        if os.path.isdir(abs_path):
            tracked = run(["git", "ls-files", "--", rel], cwd=root)
            if tracked:
                files.update(tracked.splitlines())
            else:
                # Not a tracked directory; still resolve ownership for it.
                files.add(rel.rstrip("/") + "/")
        else:
            files.add(rel)
    return sorted(files)


# --------------------------------------------------------------------------
# requesting reviewers on a PR
# --------------------------------------------------------------------------
def resolve_author(pr: str | None, repo: str | None) -> str | None:
    """Who will own the PR: its author, or you for not-yet-pushed changes.

    Returns None when `gh` cannot answer, in which case the author simply is
    not excluded from the suggestions.
    """
    if pr:
        cmd = ["gh", "pr", "view", str(pr), "--json", "author", "--jq", ".author.login"]
        if repo:
            cmd += ["--repo", repo]
    else:
        # `gh api` takes no --repo; the login is account-wide anyway.
        cmd = ["gh", "api", "user", "--jq", ".login"]
    return run(cmd) or None


def resolve_pr(pr: str | None, repo: str | None) -> str:
    """The PR to act on: the one given, else the current branch's PR."""
    if pr:
        return str(pr)
    cmd = ["gh", "pr", "view", "--json", "number", "--jq", ".number"]
    if repo:
        cmd += ["--repo", repo]
    ok, out = run_checked(cmd)
    if not ok or not out:
        die(
            "no PR found for the current branch; open one first or pass "
            "--pr <PR_NUMBER>"
        )
    return out


def request_reviewers(
    owners: list[str],
    pr: str | None,
    repo: str | None,
    style: Style,
) -> None:
    """Request review from the given owners on a PR."""
    handles = [o.lstrip("@") for o in owners]
    if not handles:
        print("No owners to request; nothing to do.")
        return

    pr = resolve_pr(pr, repo)
    cmd = ["gh", "pr", "edit", pr, "--add-reviewer", ",".join(handles)]
    if repo:
        cmd += ["--repo", repo]

    ok, out = run_checked(cmd)
    if not ok:
        die(f"could not request review on PR #{pr}: {out}")

    joined = ", ".join(f"@{h}" for h in handles)
    print(f"\nRequested review on PR #{pr} from {style.green(joined)}")


# --------------------------------------------------------------------------
# grouping and reporting
# --------------------------------------------------------------------------
def group_by_rule(files: list[str], rules: list[Rule]) -> tuple[dict, list[str]]:
    groups: dict[str, dict] = {}
    unowned: list[str] = []
    for path in files:
        rule = owners_for(path, rules)
        if rule is None or not rule.owners:
            unowned.append(path)
            continue
        group = groups.setdefault(rule.pattern, {"owners": rule.owners, "files": []})
        group["files"].append(path)
    return groups, unowned


def eligible_owners(owners: list[str], author: str | None) -> list[str]:
    """Owners who can actually review, i.e. everyone but the author."""
    if not author:
        return list(owners)
    return [o for o in owners if o.lstrip("@").lower() != author.lower()]


def self_owned_areas(groups: dict, author: str | None) -> list[str]:
    """Patterns whose only owner is the author, so nobody else can be asked."""
    return sorted(
        pat
        for pat, g in groups.items()
        if g["owners"] and not eligible_owners(g["owners"], author)
    )


def minimal_cover(groups: dict, author: str | None = None) -> list[str]:
    """Smallest set of owners with at least one owner per matched rule.

    Every module needs one approval from its owners, so this is the shortest
    reviewer list that can unblock the whole PR. Greedy set cover: repeatedly
    take the owner who covers the most still-uncovered rules.

    The author is never chosen. GitHub refuses to register a self-review
    request, and asking for one drops the rest of the request along with it.
    """
    eligible = {pat: eligible_owners(g["owners"], author) for pat, g in groups.items()}
    # Areas the author owns outright have nobody left to ask; they are
    # reported separately rather than silently covered.
    remaining = {pat for pat, owners in eligible.items() if owners}
    chosen: list[str] = []
    while remaining:
        best, best_cover = None, set()
        for owner in sorted({o for owners in eligible.values() for o in owners}):
            covered = {pat for pat in remaining if owner in eligible[pat]}
            if len(covered) > len(best_cover):
                best, best_cover = owner, covered
        if not best:
            break
        chosen.append(best)
        remaining -= best_cover
    return chosen


class Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t):
        return self(t, "1")

    def dim(self, t):
        return self(t, "2")

    def green(self, t):
        return self(t, "32")

    def yellow(self, t):
        return self(t, "33")


def report(
    source: str,
    files: list[str],
    groups: dict,
    unowned: list[str],
    codeowners_path: str,
    show_files: bool,
    style: Style,
    author: str | None = None,
) -> None:
    if not files:
        print(f"No files found ({source}). Nothing to review.")
        return

    print(f"{style.bold(str(len(files)))} file(s) — {source}")
    print(style.dim(f"ownership from {codeowners_path}\n"))

    if not groups:
        print("No CODEOWNERS rule matches these files.")
    else:
        print(style.bold("Owners by area"))
        width = max(len(p) for p in groups)
        for pattern in sorted(groups):
            group = groups[pattern]
            owners = " ".join(group["owners"])
            count = len(group["files"])
            plural = "" if count == 1 else "s"
            print(
                f"  {pattern.ljust(width)}  {style.green(owners)}"
                f"  {style.dim(f'({count} file{plural})')}"
            )
            if show_files:
                for path in group["files"]:
                    print(style.dim(f"      {path}"))

        cover = minimal_cover(groups, author)
        mine = self_owned_areas(groups, author)
        print()
        if cover:
            print(style.bold("Ping these reviewers"))
            print(f"  {style.green(' '.join(cover))}")
            print(
                style.dim(
                    "  (smallest set covering every area above; "
                    "one approval per area is required)"
                )
            )
            handles = ",".join(o.lstrip("@") for o in cover)
            print()
            print(style.dim("  gh pr edit --add-reviewer ") + handles)
            print(style.dim("  (or re-run with --request to do that for you)"))
        else:
            print(style.bold("No one to ping"))
            print(style.dim("  you own every area this change touches"))

        if mine:
            print()
            print(
                style.yellow(
                    f"You are the only owner of {len(mine)} area(s); "
                    "ask anyone for a second pair of eyes:"
                )
            )
            for pattern in mine:
                print(f"  {pattern}")

    if unowned:
        print()
        print(style.yellow(f"No owner matched ({len(unowned)} file(s)):"))
        for path in unowned:
            print(f"  {path}")


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the CODEOWNERS reviewers for your changes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Typical use, from anywhere inside the repo:")[-1],
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="files or directories to look up (default: files changed on this branch)",
    )
    parser.add_argument("--pr", help="look up the files of an existing PR number")
    parser.add_argument(
        "--repo", help="owner/name to use with --pr (default: current repo)"
    )
    parser.add_argument(
        "--base", help="base ref to diff against (default: origin/main)"
    )
    parser.add_argument(
        "--codeowners",
        help="read ownership from this file instead of the checked-out one",
    )
    parser.add_argument(
        "--files",
        action="store_true",
        help="list the matched files under each owner group",
    )
    parser.add_argument(
        "--handles",
        action="store_true",
        help="print only a comma-separated reviewer list, nothing else",
    )
    parser.add_argument(
        "--request",
        action="store_true",
        help="request review from the owners on the PR "
        "(--pr, else the current branch's PR)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON output")
    parser.add_argument("--no-color", action="store_true", help="disable color")
    args = parser.parse_args()

    root = repo_root()
    rules, codeowners_path = load_rules(root, args.codeowners)

    if args.pr:
        files, source = pr_files(args.pr, args.repo)
    elif args.paths:
        files, source = expand_paths(root, args.paths), "explicit paths"
    else:
        base = resolve_base(root, args.base)
        files, source = changed_files(root, base)

    groups, unowned = group_by_rule(files, rules)
    author = resolve_author(args.pr, args.repo)

    if args.handles:
        print(",".join(o.lstrip("@") for o in minimal_cover(groups, author)))
        return

    if args.json:
        print(
            json.dumps(
                {
                    "source": source,
                    "codeowners": codeowners_path,
                    "files": files,
                    "groups": [
                        {
                            "pattern": pattern,
                            "owners": groups[pattern]["owners"],
                            "files": groups[pattern]["files"],
                        }
                        for pattern in sorted(groups)
                    ],
                    "reviewers": minimal_cover(groups, author),
                    "author": author,
                    "self_owned": self_owned_areas(groups, author),
                    "unowned": unowned,
                },
                indent=2,
            )
        )
        return

    color = sys.stdout.isatty() and not args.no_color and not os.environ.get("NO_COLOR")
    style = Style(color)
    report(
        source,
        files,
        groups,
        unowned,
        codeowners_path,
        args.files,
        style,
        author,
    )

    if args.request:
        request_reviewers(minimal_cover(groups, author), args.pr, args.repo, style)


if __name__ == "__main__":
    main()
