#!/usr/bin/env python3
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
"""Check tracked files or GitHub PR event metadata for internal references.

This script is the single source of truth for both the ``no-internal-refs``
pre-commit hook and ``.github/workflows/pr-text.yml`` (R-7.12 in
``docs/developers_guide/coding_guidance.md``). Output names only where a hit
occurred, never the matched text, so logs do not repeat internal references.

Usage:

    tools/check_internal_refs.py <file> [<file> ...]
    tools/check_internal_refs.py --pr-event "$GITHUB_EVENT_PATH"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Matches Git's FIRST_FEW_BYTES (8000) NUL-byte prefix check in buffer_is_binary.
_BINARY_SNIFF_BYTES = 8000

# Shared across tracked text files and GitHub PR metadata.
_COMMON_PATTERNS: tuple[str, ...] = (
    # Lowercase `go/<letter>` plus any case-variant targeting a lowercase slug
    # (e.g., `Go/<lowercase>` or `GO/<lowercase>`), so prose like `Go/Rust`,
    # `GO/RUST`, and paths like `/usr/local/go/bin` pass while internal
    # shortlinks fail.
    r"""(?:^|[\s(\[{"'`<,]|https?://)(?:go/[A-Za-z]|(?i:go)/[a-z])""",
    # Tracker/experiment/Q&A IDs: `b/` and `cl/` use >= 6 digits; `xid/` and
    # `yaqs/` use >= 5 digits.
    r"(?i:(?<![A-Za-z0-9])(?:b/[0-9]{6,}|cl/[0-9]{6,}|xid/[0-9]{5,}|yaqs/[0-9]{5,}))",
    # Internal monorepo root and corp/prod/workstation hostnames.
    r"(?i:\bgoogle3\b|\b(?:corp|prod|goto)\.google\.com\b"
    r"|\bgoogleplex\b|\bc\.googlers\.com\b)",
    # Internal job IDs use a `j-` prefix followed by an 8-hex-character hash.
    r"(?i:(?<![A-Za-z0-9])j-[0-9a-f]{8}(?![A-Za-z0-9]))",
)

# Extra patterns checked in PR titles, descriptions, and branch names:
# spaced or unspaced >= 6-digit CL IDs and tracker tool name.
_PR_TEXT_EXTRA_PATTERNS: tuple[str, ...] = (
    r"(?i:\bcl\s+[0-9]{6,}|(?<![A-Za-z0-9])cl[0-9]{6,}(?![A-Za-z0-9])|\bbuganizer\b)",
)

# Unspaced >= 8-digit `b<digits>` is scoped to PR branch names only so 9+ char
# hex commit SHAs or UUID trailing segments in PR descriptions pass.
_PR_BRANCH_EXTRA_PATTERNS: tuple[str, ...] = (
    r"(?i:(?<![A-Za-z0-9])b[0-9]{8,}(?![A-Za-z0-9]))",
)

FILE_PATTERN = re.compile("|".join(_COMMON_PATTERNS))
PR_TEXT_PATTERN = re.compile("|".join((*_COMMON_PATTERNS, *_PR_TEXT_EXTRA_PATTERNS)))
PR_BRANCH_PATTERN = re.compile(
    "|".join((*_COMMON_PATTERNS, *_PR_TEXT_EXTRA_PATTERNS, *_PR_BRANCH_EXTRA_PATTERNS))
)


class EventPayloadError(TypeError):
    """Raised when the GitHub event JSON lacks the expected pull_request shape."""


def check_text_lines(text: str, pattern: re.Pattern[str] = FILE_PATTERN) -> list[int]:
    """Return 1-based line numbers in ``text`` matching ``pattern``."""
    return [
        line_no
        for line_no, raw_line in enumerate(text.split("\n"), 1)
        if pattern.search(raw_line.removesuffix("\r"))
    ]


def check_files(paths: Sequence[pathlib.Path]) -> list[tuple[pathlib.Path, int]]:
    """Scan text files and return ``(path, line_no)`` hits without echoing lines."""
    hits: list[tuple[pathlib.Path, int]] = []
    for path in paths:
        raw = path.read_bytes()
        if b"\x00" in raw[:_BINARY_SNIFF_BYTES]:
            continue
        text = raw.decode("utf-8", errors="replace")
        hits.extend((path, line_no) for line_no in check_text_lines(text, FILE_PATTERN))
    return hits


def check_pr_payload(payload: dict[str, object]) -> list[str]:
    """Return human-readable locations of internal references in a PR event."""
    pr = payload.get("pull_request")
    if not isinstance(pr, dict):
        msg = "event JSON is missing a pull_request object"
        raise EventPayloadError(msg)

    head = pr.get("head")
    if not isinstance(head, dict):
        msg = "pull_request is missing a head object"
        raise EventPayloadError(msg)

    branch = head.get("ref")
    if not isinstance(branch, str):
        msg = "pull_request.head is missing a string ref"
        raise EventPayloadError(msg)

    title = pr.get("title")
    if not isinstance(title, str):
        msg = "pull_request is missing a string title"
        raise EventPayloadError(msg)

    raw_body = pr.get("body")
    if raw_body is not None and not isinstance(raw_body, str):
        msg = "pull_request.body must be a string or null"
        raise EventPayloadError(msg)
    body = "" if raw_body is None else raw_body

    places: list[str] = []
    if PR_BRANCH_PATTERN.search(branch):
        places.append("the branch name")
    if PR_TEXT_PATTERN.search(title):
        places.append("the title")
    places.extend(
        f"line {line_no} of the description"
        for line_no in check_text_lines(body, PR_TEXT_PATTERN)
    )
    return places


def _run_pr_event(event_path: pathlib.Path) -> int:
    """Validate a GitHub pull_request event JSON file and print non-echoing errors."""
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "event JSON root must be an object"
        raise EventPayloadError(msg)
    places = check_pr_payload(payload)
    if not places:
        print("No internal links in the PR branch name, title, or description.")
        return 0
    for place in places:
        print(f"::error::Internal link or ID in {place}.")
    print(
        "This repo is public. Remove internal links, tracker IDs, internal "
        "job IDs, internal code paths, and internal hosts from the branch name, "
        "PR title, and description (R-7.12 in docs/developers_guide/"
        "coding_guidance.md). Editing the PR re-runs this check for title and "
        "description fixes; branch-name matches require opening a new PR from "
        "a renamed branch."
    )
    return 1


def _run_files(filenames: Sequence[str]) -> int:
    """Scan positional file paths and report matching path:line locations."""
    hits = check_files([pathlib.Path(name) for name in filenames])
    if not hits:
        return 0
    for path, line_no in hits:
        print(
            f"{path}:{line_no}: internal link, tracker ID, internal job ID, "
            "internal code path, or internal host (R-7.12)"
        )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for file and GitHub PR event modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr-event",
        type=pathlib.Path,
        default=None,
        help="Path to the GitHub Actions pull_request event JSON payload.",
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Text files to scan (passed by the pre-commit hook).",
    )
    args = parser.parse_args(argv)
    if args.pr_event is not None:
        return _run_pr_event(args.pr_event)
    return _run_files(args.filenames)


if __name__ == "__main__":
    sys.exit(main())
