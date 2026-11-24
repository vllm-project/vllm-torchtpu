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
"""Tests for tools/check_internal_refs.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import types

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "tools" / "check_internal_refs.py"
)


@pytest.fixture(scope="module")
def checker() -> types.ModuleType:
    """Load tools/check_internal_refs.py dynamically for test execution."""
    spec = importlib.util.spec_from_file_location("check_internal_refs", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Synthetic test tokens are built from parts so this test file stays clean
# under `no-internal-refs --all-files`.
@pytest.mark.parametrize(
    "sample",
    [
        "See " + "go" + "/my-doc for details",
        "https://" + "go" + "/my-doc",
        "Go" + "/my-link starts a sentence",
        "GO" + "/foo-doc in uppercase heading",
        "gO" + "/foo mixed case",
        "Tracked in " + "b" + "/123456",
        "Fixed in " + "cl" + "/123456",
        "Experiment " + "xid" + "/12345",
        "Question " + "yaqs" + "/12345",
        "Path in " + "google" + "3 repo",
        "Host foo." + "corp." + "google.com",
        "Host foo." + "prod." + "google.com",
        "Link " + "goto." + "google.com/abc",
        "Paste " + "google" + "plex.com/123",
        "Workstation host." + "c." + "googlers.com",
        "Run job " + "j-" + "0123abcd failed",
    ],
)
def test_file_pattern_flags_internal_refs(
    checker: types.ModuleType,
    sample: str,
) -> None:
    """File pattern catches internal links, trackers, hosts, and job IDs."""
    assert checker.check_text_lines(sample) == [1]


@pytest.mark.parametrize(
    "sample",
    [
        "Compare Go/Rust, Go/Python, and GO/RUST toolchains",
        "Installed at /usr/local/go/bin/go",
        "Cargo path cargo/bin/tool",
        "Commit SHA " + "b668874506" + " fixed it",
        "Unspaced token " + "cl123456" + " only checked in PR metadata",
        "Word " + "bugan" + "izer only checked in PR metadata",
        "Public link https://docs.google.com/document/d/123456",
        "Public repo https://github.com/vllm-project/vllm-torchtpu",
    ],
)
def test_file_pattern_allows_clean_lines(
    checker: types.ModuleType,
    sample: str,
) -> None:
    """File pattern allows clean prose, paths, SHAs, and public URLs."""
    assert checker.check_text_lines(sample) == []


@pytest.mark.parametrize(
    ("branch", "title", "body", "want_places"),
    [
        # Clean PR including a 10-char hex SHA and UUID tail starting with `b`
        (
            "feature/add-kernel",
            "[Core] Add clean feature",
            "Rebased on commit b668874506, build 12345678-1234-1234-1234-b12345678901, "
            "and compares Go/Rust.\n",
            [],
        ),
        # Null body from GitHub webhook payload
        ("feature/empty-body", "[Doc] Clean title", None, []),
        # Capitalized shortlink in title
        (
            "feature/clean",
            "Go" + "/my-link in title",
            "Clean body",
            ["the title"],
        ),
        # All-caps shortlink prefix targeting lowercase slug in description
        (
            "feature/clean",
            "[Doc] Clean title",
            "See " + "GO" + "/foo-doc\n",
            ["line 1 of the description"],
        ),
        # Unspaced bug ID is caught in branch name only
        (
            "user/fix-" + "b" + "123456789",
            "[Bugfix] Clean title",
            "Commit b668874506 in body is allowed.\n",
            ["the branch name"],
        ),
        # Unspaced CL, spaced CL, buganizer, and job ID in description lines
        (
            "user/fix-" + "cl" + "123456",
            "[Bugfix] Follow up " + "CL " + "123456",
            f"Line 1 clean\r\nLine 2 mentions {'bugan'}izer\r\n"
            f"Line 3 job {'j-'}0a1b2c3d\r\n",
            [
                "the branch name",
                "the title",
                "line 2 of the description",
                "line 3 of the description",
            ],
        ),
        # Form feed does not shift line numbering
        (
            "feature/clean",
            "[Core] Clean",
            "Line 1\x0cstill line 1\nLine 2 " + "go" + "/secret-doc\n",
            ["line 2 of the description"],
        ),
    ],
)
def test_pr_payload_samples(
    checker: types.ModuleType,
    branch: str,
    title: str,
    body: str | None,
    want_places: list[str],
) -> None:
    """PR payload check scopes rules properly across branch, title, and body."""
    payload = {
        "pull_request": {
            "title": title,
            "body": body,
            "head": {"ref": branch},
        }
    }
    assert checker.check_pr_payload(payload) == want_places


def test_pr_event_cli_does_not_echo_and_gives_new_pr_guidance(
    checker: types.ModuleType,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PR event CLI omits matched text and instructs opening a new PR."""
    secret_branch = "user/fix-" + "b" + "987654321"
    secret_link = "go" + "/super-secret-design"
    event_file = tmp_path / "event.json"
    event_file.write_text(
        json.dumps(
            {
                "pull_request": {
                    "title": "[Core] Clean title",
                    "body": f"Clean line\nSee {secret_link}\n",
                    "head": {"ref": secret_branch},
                }
            }
        ),
        encoding="utf-8",
    )

    rc = checker.main(["--pr-event", str(event_file)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "::error::Internal link or ID in the branch name." in out
    assert "::error::Internal link or ID in line 2 of the description." in out
    assert "opening a new PR from a renamed branch" in out
    assert secret_branch not in out
    assert secret_link not in out


def test_file_cli_does_not_echo_and_skips_binary(
    checker: types.ModuleType,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """File CLI reports only path:line without echoing secrets and skips binary."""
    secret_token = "go" + "/private-note"
    dirty = tmp_path / "dirty.py"
    dirty.write_text(f"# ok\n# see {secret_token}\n", encoding="utf-8")
    clean = tmp_path / "clean.py"
    clean.write_text("# Go/Rust and /usr/local/go/bin\n", encoding="utf-8")
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\x00\x01" + secret_token.encode("utf-8"))

    assert checker.main([str(clean), str(binary)]) == 0
    assert capsys.readouterr().out == ""

    assert checker.main([str(dirty)]) == 1
    out = capsys.readouterr().out
    assert f"{dirty}:2:" in out
    assert secret_token not in out


def test_file_cli_names_all_five_reference_kinds(
    checker: types.ModuleType,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """File CLI output names all five matched kinds without echoing tokens."""
    job_token = "j-" + "0123abcd"
    path_token = "google" + "3"
    dirty = tmp_path / "hits.py"
    dirty.write_text(f"# job {job_token}\n# path {path_token}\n", encoding="utf-8")

    assert checker.main([str(dirty)]) == 1
    out = capsys.readouterr().out
    assert f"{dirty}:1:" in out
    assert f"{dirty}:2:" in out
    for kind in (
        "internal link",
        "tracker ID",
        "internal job ID",
        "internal code path",
        "internal host",
    ):
        assert kind in out
    assert job_token not in out
    assert path_token not in out


def test_pr_event_cli_allows_clean_event_and_rejects_malformed(
    checker: types.ModuleType,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PR event CLI passes clean payloads and raises on malformed event JSON."""
    clean_event = tmp_path / "clean.json"
    clean_event.write_text(
        json.dumps(
            {
                "pull_request": {
                    "title": "[Core] Clean title",
                    "body": "Commit b668874506 compares Go/Rust.\n",
                    "head": {"ref": "feature/clean"},
                }
            }
        ),
        encoding="utf-8",
    )
    assert checker.main(["--pr-event", str(clean_event)]) == 0
    assert "No internal links" in capsys.readouterr().out

    malformed_payloads = [
        "[]",
        "{}",
        json.dumps({"pull_request": {}}),
        json.dumps({"pull_request": {"title": "Clean", "head": {}}}),
        json.dumps({"pull_request": {"head": {"ref": "feature/clean"}}}),
        json.dumps(
            {"pull_request": {"title": "Clean", "body": 123, "head": {"ref": "ok"}}}
        ),
    ]
    for index, raw_payload in enumerate(malformed_payloads):
        bad_file = tmp_path / f"bad_{index}.json"
        bad_file.write_text(raw_payload, encoding="utf-8")
        with pytest.raises(checker.EventPayloadError):
            checker.main(["--pr-event", str(bad_file)])
