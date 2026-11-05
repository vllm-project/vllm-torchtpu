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
"""Fetch vLLM's Rust frontend, vllm-rs, from the vLLM wheel published for a ref.

Usage: fetch_vllm_rs.py VLLM_COMMIT_HASH OUT_DIR

A release tag (v0.32.0) takes the wheel from PyPI; a full commit hash takes it
from wheels.vllm.ai, where vLLM's CI publishes a wheel per commit some time
after the merge. The binary is written to OUT_DIR/vllm-rs.

A ref with no published wheel, such as an upstream commit not built yet, or a
wheel without vllm-rs, leaves OUT_DIR empty and says so. Any other failure, a
network error included, exits non-zero.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from urllib.parse import urljoin

MEMBER = "vllm/vllm-rs"
RELEASE_TAG = re.compile(r"v(\d+\.\d+\.\d+\S*)")
COMMIT = re.compile(r"[0-9a-f]{40}")


class NotPublished(Exception):
    """No vLLM wheel with vllm-rs is published for the ref."""


def _get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise NotPublished(f"{url} answered 404") from e
        raise


def wheel_url(ref: str, arch: str) -> str:
    """The URL of the vLLM wheel published for `ref` on `arch`."""
    if match := RELEASE_TAG.fullmatch(ref):
        version = match.group(1)
        release = _get_json(f"https://pypi.org/pypi/vllm/{version}/json")
        for file in release["urls"]:
            if file["filename"].endswith(f"_{arch}.whl"):
                return file["url"]
        raise NotPublished(f"PyPI has no {arch} wheel for vllm {version}")
    if COMMIT.fullmatch(ref):
        index = f"https://wheels.vllm.ai/{ref}/vllm/"
        for wheel in _get_json(index + "metadata.json"):
            if wheel.get("package_name") == "vllm" and arch in wheel.get(
                "platform_tag", ""
            ):
                return urljoin(index, wheel["path"])
        raise NotPublished(f"{index} has no {arch} wheel")
    raise NotPublished(f"{ref!r} is neither a release tag nor a full commit hash")


def fetch(ref: str, out_dir: str, arch: str) -> bool:
    """Write vllm-rs from `ref`'s published wheel to `out_dir`. False when no
    wheel with vllm-rs is published for it."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        url = wheel_url(ref, arch)
        print(f"vllm-rs: extracting {MEMBER} from {url}")
        with tempfile.TemporaryFile() as wheel:
            with urllib.request.urlopen(url, timeout=600) as response:
                shutil.copyfileobj(response, wheel)
            with zipfile.ZipFile(wheel) as archive:
                if MEMBER not in archive.namelist():
                    raise NotPublished(f"the wheel at {url} has no {MEMBER}")
                data = archive.read(MEMBER)
    except NotPublished as e:
        print(f"vllm-rs: no published binary for {ref} ({e}); the image has none.")
        return False
    target = os.path.join(out_dir, "vllm-rs")
    with open(target, "wb") as f:
        f.write(data)
    os.chmod(target, 0o755)
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 3 or not argv[1]:
        print(
            "usage: fetch_vllm_rs.py VLLM_COMMIT_HASH OUT_DIR; docker/build_image.sh "
            "passes the hash (the pyproject.toml pin unless overridden)",
            file=sys.stderr,
        )
        return 2
    fetch(argv[1], argv[2], platform.machine())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
