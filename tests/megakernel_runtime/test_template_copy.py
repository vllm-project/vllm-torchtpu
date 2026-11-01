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
"""A model folder scaffolded from the template under a fresh package name
works: `python -m vllm_torchtpu.megakernel_runtime.new` copies it with fresh
registry names and provenance headers, the copy's own CPU tests pass, and the
original and the copy import side by side in one process (neither registers
anything at import time). The template is named by string only, so the
runtime's tests import no model folder."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from vllm_torchtpu.megakernel_runtime import provenance

_TEMPLATE = "vllm_torchtpu.models.qwen3_5.megakernel"
_COPY = "mk_template_copy"
_SRC = Path(importlib.util.find_spec("vllm_torchtpu").origin).resolve().parents[1]


def _template_dir() -> Path:
    spec = importlib.util.find_spec(_TEMPLATE)
    assert spec is not None and spec.submodule_search_locations
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(_SRC)])
    env.pop("VLLM_TPU_MEGAKERNEL", None)
    return env


def _scaffold(tmp_path: Path) -> tuple[Path, str]:
    dst = tmp_path / _COPY
    cmd = [sys.executable, "-m", "vllm_torchtpu.megakernel_runtime.new"]
    cmd += ["--from", _TEMPLATE, "--dir", str(dst), "--name", "mkcopy"]
    cmd += ["--model-name", "mkcopy_model"]
    out = subprocess.run(
        cmd, env=_env(tmp_path), capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-4000:]
    assert f"(package {_COPY})" in out.stdout and "Next steps:" in out.stdout
    commit = out.stdout.split(" @ ", 1)[1].split(".\n", 1)[0]
    return dst, commit


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
assert copy.Qwen35.name == "mkcopy_model" != orig.Qwen35.name
spec, orig_spec = mods[{_COPY!r}].SPEC, mods[{_TEMPLATE!r}].SPEC
assert spec.create == ".adapter:create"
assert spec.name == "mkcopy" != orig_spec.name
o_profs = mods[{_TEMPLATE!r} + ".profiles"].PROFILES
c_profs = mods[{_COPY!r} + ".profiles"].PROFILES
assert len(c_profs) == len(o_profs)
assert all(n.startswith("mkcopy-") for n in c_profs), sorted(c_profs)
assert all(p.model == "mkcopy_model" for p in c_profs.values())
assert not set(c_profs) & set(o_profs)
assert mk.list_models() == () and mk.list_profiles() == ("default",)
print("OK")
"""


def test_scaffolded_copy_runs_its_tests_and_imports_beside_the_original(
    tmp_path: Path,
) -> None:
    dst, commit = _scaffold(tmp_path)
    assert (dst / "README.md").is_file() and (dst / "serving" / "serve.sh").is_file()
    src = _template_dir()
    copies = sorted(dst.rglob("*.py"))
    assert len(copies) == len(list(src.rglob("*.py")))
    for path in copies:
        rel = provenance.repo_relative(src / path.relative_to(dst))
        assert provenance.read_provenance(path.read_text("utf-8")) == (rel, commit)
    assert provenance.list_copies(src / "adapter.py", [tmp_path]) == [
        (dst / "adapter.py", commit)
    ]
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
