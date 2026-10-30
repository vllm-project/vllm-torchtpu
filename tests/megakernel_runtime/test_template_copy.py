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
"""A model folder copied under a fresh package name works: the copy's own CPU
tests pass, and the original and the copy import side by side in one process
(neither registers anything at import time). The template is named by string
only, so the runtime's tests import no model folder."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

_TEMPLATE = "vllm_torchtpu.models.qwen3_5.megakernel"
_COPY = "mk_template_copy"
_SRC = Path(importlib.util.find_spec("vllm_torchtpu").origin).resolve().parents[1]


def _copy(tmp_path: Path) -> Path:
    spec = importlib.util.find_spec(_TEMPLATE)
    assert spec is not None and spec.submodule_search_locations
    src = Path(next(iter(spec.submodule_search_locations)))
    dst = tmp_path / _COPY
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def _env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(_SRC)])
    env.pop("VLLM_TPU_MEGAKERNEL", None)
    return env


_BOTH = f"""
import importlib
import vllm_torchtpu.megakernel as mk
mods = {{}}
for pkg in ({_TEMPLATE!r}, {_COPY!r}):
    for sub in ("", ".adapter", ".model", ".profiles", ".tests.tiny"):
        mods[pkg + sub] = importlib.import_module(pkg + sub)
orig, copy = mods[{_TEMPLATE!r} + ".model"], mods[{_COPY!r} + ".model"]
assert orig.Qwen35 is not copy.Qwen35
assert copy.Qwen35.__module__ == {_COPY!r} + ".model"
assert mods[{_COPY!r}].SPEC.create == ".adapter:create"
assert mk.list_models() == () and mk.list_profiles() == ("default",)
print("OK")
"""


def test_copy_runs_its_tests_and_imports_beside_the_original(tmp_path: Path) -> None:
    dst = _copy(tmp_path)
    assert (dst / "README.md").is_file() and (dst / "serving" / "serve.sh").is_file()
    env = _env(tmp_path)
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=dst,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = tests.stdout + tests.stderr
    assert tests.returncode == 0, out[-4000:]
    assert " passed" in out and "failed" not in out, out[-2000:]
    both = subprocess.run(
        [sys.executable, "-c", _BOTH],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert both.returncode == 0 and both.stdout.strip() == "OK", both.stderr[-4000:]
