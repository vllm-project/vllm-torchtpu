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
"""Unit tests for docker/fetch_vllm_rs.py."""

import importlib.util
import io
import json
import os
import pathlib
import sys
import urllib.error
import zipfile

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "docker" / "fetch_vllm_rs.py"
_spec = importlib.util.spec_from_file_location("fetch_vllm_rs", _SCRIPT)
fetch_vllm_rs = importlib.util.module_from_spec(_spec)
sys.modules["fetch_vllm_rs"] = fetch_vllm_rs
_spec.loader.exec_module(fetch_vllm_rs)

COMMIT = "0123456789abcdef0123456789abcdef01234567"
BINARY = b"\x7fELF vllm-rs"


def _wheel(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def serve(monkeypatch):
    """Answer urlopen from a {url: bytes | int} map; an int is an HTTP error."""
    routes: dict[str, bytes | int] = {}
    requested: list[str] = []

    def urlopen(url, timeout=None):
        requested.append(url)
        answer = routes.get(url, 404)
        if isinstance(answer, int):
            raise urllib.error.HTTPError(url, answer, "error", {}, None)
        return io.BytesIO(answer)

    monkeypatch.setattr(fetch_vllm_rs.urllib.request, "urlopen", urlopen)
    return routes, requested


def test_release_tag_reads_the_pypi_wheel(serve, tmp_path):
    routes, requested = serve
    wheel = "https://files.example/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl"
    routes["https://pypi.org/pypi/vllm/0.32.0/json"] = json.dumps(
        {
            "urls": [
                {
                    "filename": "vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl",
                    "url": "x",
                },
                {"filename": wheel.rsplit("/", 1)[1], "url": wheel},
                {"filename": "vllm-0.32.0.tar.gz", "url": "y"},
            ]
        }
    ).encode()
    routes[wheel] = _wheel({"vllm/vllm-rs": BINARY, "vllm/__init__.py": b""})

    assert fetch_vllm_rs.fetch("v0.32.0", str(tmp_path), "x86_64")

    target = tmp_path / "vllm-rs"
    assert target.read_bytes() == BINARY
    assert os.access(target, os.X_OK)
    assert requested[-1] == wheel


def test_commit_reads_the_wheels_vllm_ai_wheel(serve, tmp_path):
    routes, _ = serve
    index = f"https://wheels.vllm.ai/{COMMIT}/vllm/"
    routes[index + "metadata.json"] = json.dumps(
        [
            {
                "package_name": "vllm",
                "platform_tag": "manylinux_2_28_aarch64",
                "path": "../a.whl",
            },
            {
                "package_name": "vllm",
                "platform_tag": "manylinux_2_28_x86_64",
                "path": "../x.whl",
            },
        ]
    ).encode()
    routes[f"https://wheels.vllm.ai/{COMMIT}/x.whl"] = _wheel({"vllm/vllm-rs": BINARY})

    assert fetch_vllm_rs.fetch(COMMIT, str(tmp_path), "x86_64")
    assert (tmp_path / "vllm-rs").read_bytes() == BINARY


@pytest.mark.parametrize(
    "ref",
    [
        COMMIT,  # not published yet: metadata.json answers 404
        "v99.0.0",  # not on PyPI
        "main",  # neither a release tag nor a full commit
        COMMIT[:12],
    ],
)
def test_unpublished_ref_leaves_the_image_without_vllm_rs(serve, tmp_path, capsys, ref):
    assert not fetch_vllm_rs.fetch(ref, str(tmp_path), "x86_64")
    assert list(tmp_path.iterdir()) == []
    assert "the image has none" in capsys.readouterr().out


def test_wheel_without_vllm_rs_leaves_the_image_without_it(serve, tmp_path):
    routes, _ = serve
    wheel = "https://files.example/vllm-0.10.0-cp38-abi3-manylinux1_x86_64.whl"
    routes["https://pypi.org/pypi/vllm/0.10.0/json"] = json.dumps(
        {"urls": [{"filename": wheel.rsplit("/", 1)[1], "url": wheel}]}
    ).encode()
    routes[wheel] = _wheel({"vllm/__init__.py": b""})

    assert not fetch_vllm_rs.fetch("v0.10.0", str(tmp_path), "x86_64")
    assert list(tmp_path.iterdir()) == []


def test_other_http_errors_fail_the_build(serve, tmp_path):
    routes, _ = serve
    routes["https://pypi.org/pypi/vllm/0.32.0/json"] = 503

    with pytest.raises(urllib.error.HTTPError):
        fetch_vllm_rs.fetch("v0.32.0", str(tmp_path), "x86_64")


def test_missing_ref_is_a_usage_error(capsys):
    assert fetch_vllm_rs.main(["fetch_vllm_rs.py", "", "/out"]) == 2
    assert "build_image.sh" in capsys.readouterr().err
