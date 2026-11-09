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
"""Fail unless vllm-torchtpu-ci passed a full nightly-style run of a commit.

    ci_gate.py <commit sha>

Reads the commit's GitHub statuses, which Buildkite posts for every
vllm-torchtpu-ci build of it, so no Buildkite token is needed on the release
agent. Passing means the newest buildkite/vllm-torchtpu-ci status is success
and at least one nightly-only step passed in that same build: a PR-lane build
alone doesn't run the nightly suites, so it can't vouch for a release.

A step counts as nightly-only when its status context contains "nightly",
i.e. its label in the CI pipeline says "(Nightly)". Renaming those steps
changes what this gate accepts.
"""

import json
import sys
import urllib.error
import urllib.request

REPO = "vllm-project/vllm-torchtpu"
CONTEXT = "buildkite/vllm-torchtpu-ci"


def statuses(sha):
    """Every status of the commit, newest first, across all pages."""
    out = []
    for page in range(1, 11):
        url = f"https://api.github.com/repos/{REPO}/commits/{sha}/statuses?per_page=100&page={page}"
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            batch = json.load(resp)
        out += batch
        if len(batch) < 100:
            break
    return out


def check(sha, all_statuses):
    """(build URL, None) if the statuses (newest first) vouch for a release,
    else (None, the reason they don't)."""
    build = next((s for s in all_statuses if s["context"] == CONTEXT), None)
    if build is None:
        return None, (
            f"no {CONTEXT} build of {sha}. Run a full build first: "
            f"vllm-torchtpu-ci on this commit with NIGHTLY=1 "
            f"(or the run-full-ci label on a PR)."
        )
    print(f"{CONTEXT}: {build['state']} ({build['target_url']})")
    if build["state"] != "success":
        return None, (
            f"the newest {CONTEXT} build of {sha} is {build['state']}, not success."
        )
    # Step statuses link to <build URL>#<job id>. Only the newest build's
    # steps count: a nightly step that passed in another build of the commit
    # (say a failed NIGHTLY=1 one) says nothing about this build.
    latest = {}
    for s in all_statuses:
        if s["context"].startswith(CONTEXT + "/") and (
            s.get("target_url") or ""
        ).startswith(build["target_url"] + "#"):
            latest.setdefault(s["context"], s)  # newest first, so the first one wins
    nightly = [
        c for c, s in latest.items() if "nightly" in c and s["state"] == "success"
    ]
    print(f"nightly-only steps passed: {len(nightly)}")
    if not nightly:
        return None, (
            f"{CONTEXT} passed on {sha}, but no nightly step ran in that build. "
            f"Run it again with NIGHTLY=1 (or the run-full-ci label on a PR) "
            f"so the release is tested by the full suite."
        )
    return build["target_url"], None


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: ci_gate.py <commit sha>")
    sha = sys.argv[1]
    try:
        all_statuses = statuses(sha)
    except urllib.error.HTTPError as e:
        sys.exit(f"ERROR: GitHub answered {e.code} for the statuses of {sha}.")
    url, reason = check(sha, all_statuses)
    if reason:
        sys.exit(f"ERROR: {reason}")
    print(url)


if __name__ == "__main__":
    main()
