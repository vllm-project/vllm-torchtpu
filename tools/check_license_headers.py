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
"""Fail when a source file lacks the Apache 2.0 license header.

The open-source release scanner rejects a source file without the Apache 2.0
license text, and an ``SPDX-License-Identifier`` tag makes it ignore that
text even when present: it reports the tag instead. So every source file
carries the full header text and no tag.

A file passes when the header text appears in its first lines, in any comment
style, and no line is an SPDX license tag. ``--fix`` inserts the standard
header below a shebang or encoding cookie, completes a truncated header in
place, and drops an ``Apache-2.0`` tag. A file that names a copyright holder
(an ``SPDX-FileCopyrightText`` or ``Copyright`` line, as upstream-derived
files do) keeps that notice and gains only the license text beneath it.

    tools/check_license_headers.py <files>          # check
    tools/check_license_headers.py --fix <files>    # repair
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import subprocess
import sys

LICENSE_TEXT = """\
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Source suffixes, keyed by the comment leader `--fix` writes. Markdown, JSON,
# TOML, plain text and binary files are not checked; the release scanner does
# not check them either.
LINE_COMMENT = {
    "#": {".py", ".pyi", ".sh", ".bash", ".yml", ".yaml", ".rb", ".cmake"},
    "//": {
        ".proto",
        ".js",
        ".mjs",
        ".ts",
        ".tsx",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hh",
        ".hpp",
        ".cu",
        ".cuh",
        ".go",
        ".java",
        ".kt",
        ".rs",
        ".swift",
        ".dart",
    },
}
COMMENT_BY_SUFFIX = {
    suffix: comment for comment, suffixes in LINE_COMMENT.items() for suffix in suffixes
}

# Lines scanned for the header: room for a shebang, a provenance note and an
# attribution notice above it.
HEADER_LINES = 40

_COMMENT_LEADER = re.compile(r"^\s*(?:#|//|/\*+|\*+/?|<!--|-->|--|;)?\s*")
_ENCODING_COOKIE = re.compile(r"^[ \t\f]*#.*?coding[:=]")  # PEP 263
# The scanner picks the tag up anywhere in the file, in any comment style.
_SPDX_TAG = re.compile(r"^[ \t]*(?:#|//)[ \t]*SPDX-License-Identifier[ \t]*:", re.M)


def _normalize(lines: list[str]) -> str:
    """Comment leaders and whitespace stripped, so any comment style matches."""
    words: list[str] = []
    for line in lines:
        words.extend(_COMMENT_LEADER.sub("", line).split())
    return " ".join(words).replace("https://", "http://")


_EXPECTED = _normalize(LICENSE_TEXT.splitlines())


def has_header(text: str) -> bool:
    return _EXPECTED in _normalize(text.splitlines()[:HEADER_LINES])


def problem(text: str) -> str | None:
    if _SPDX_TAG.search(text):
        return "SPDX-License-Identifier tag"
    if not has_header(text):
        return "missing header"
    return None


def _copyright_year(name: str) -> int:
    """Year the file entered the repository, or this year for a new file."""
    log = subprocess.run(
        ["git", "log", "--follow", "--format=%as", "--", name],
        capture_output=True,
        text=True,
        check=False,
    )
    dates = log.stdout.split()
    if log.returncode != 0 or not dates:
        return datetime.date.today().year
    return int(dates[-1][:4])


def without_apache_tag(text: str, comment: str) -> str:
    """Only the Apache-2.0 tag is dropped; any other license tag is a finding
    for a human, since the file is not under this repository's license."""
    tag_line = re.compile(
        rf"^[ \t]*{re.escape(comment)}[ \t]*"
        r"SPDX-License-Identifier:[ \t]*Apache-2\.0[ \t]*\n?",
        re.M,
    )
    return tag_line.sub("", text)


def with_header(text: str, comment: str, year: int) -> str:
    lines = text.splitlines(keepends=True)
    start = 0
    if lines and lines[0].startswith("#!"):
        start = 1
    if start < len(lines) and _ENCODING_COOKIE.match(lines[start]):
        start += 1

    notice_line = re.compile(
        rf"^{re.escape(comment)}\s*(SPDX-(?!License-Identifier)|Copyright\b)"
    )
    end = start
    while end < len(lines) and notice_line.match(lines[end]):
        end += 1
    notice = lines[start:end]
    if not any(re.search(r"SPDX-FileCopyrightText|Copyright\b", n) for n in notice):
        notice = [f"{comment} Copyright {year} Google LLC\n"]

    # A truncated header is a prefix of the full text: replace it rather than
    # leave it dangling below the new one.
    keep_from = end
    probe = end
    while probe < len(lines) and lines[probe].startswith(comment):
        partial = _normalize(lines[end : probe + 1])
        if not _EXPECTED.startswith(partial):
            break
        probe += 1
        if partial:
            keep_from = probe

    header = notice + [f"{comment}\n"]
    header += [
        f"{comment} {line}".rstrip() + "\n" for line in LICENSE_TEXT.splitlines()
    ]
    if keep_from < len(lines) and lines[keep_from].startswith(comment):
        header.append("\n")
    return "".join(lines[:start] + header + lines[keep_from:])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="insert a missing header and drop an Apache-2.0 SPDX tag",
    )
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    failures: list[tuple[str, str]] = []
    for name in args.filenames:
        path = pathlib.Path(name)
        comment = COMMENT_BY_SUFFIX.get(path.suffix)
        if comment is None or not path.is_file():
            continue
        raw = path.read_bytes()
        if not raw:  # a bare package marker
            continue
        text = raw.decode("utf-8", errors="surrogateescape")
        if args.fix:
            fixed = without_apache_tag(text, comment)
            if not has_header(fixed) and not _SPDX_TAG.search(fixed):
                fixed = with_header(fixed, comment, _copyright_year(name))
            if fixed != text:
                path.write_bytes(fixed.encode("utf-8", errors="surrogateescape"))
                print(f"fixed license header: {name}")
            text = fixed
        if reason := problem(text):
            failures.append((name, reason))

    if failures:
        print("Apache 2.0 license header problems:\n")
        for name, reason in failures:
            print(f"  {name}: {reason}")
        print(
            "\nCopy the header from a sibling file, or run\n"
            "  tools/check_license_headers.py --fix <files>\n"
            "which inserts it, keeps an existing copyright notice, and drops "
            "the Apache-2.0 SPDX tag (the release scanner reads the tag "
            "instead of the header text)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
