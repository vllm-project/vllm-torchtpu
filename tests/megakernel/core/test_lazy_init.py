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
"""The lazy (PEP 562) `megakernel.core` and `megakernel.core.runtime` namespaces."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import vllm_torchtpu.megakernel.core as mk
import vllm_torchtpu.megakernel.core.dma as mk_dma
import vllm_torchtpu.megakernel.core.runtime as mk_rt

_SRC = Path(mk.__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "pkg", [mk, mk_dma, mk_rt], ids=["megakernel", "dma", "runtime"]
)
def test_every_public_name_is_in_the_lazy_tables_and_resolves(pkg) -> None:
    lazy, subs = set(pkg._LAZY), set(pkg._SUBMODULES)
    assert not lazy & subs
    assert set(pkg.__all__) <= lazy | subs, sorted(set(pkg.__all__) - lazy - subs)
    assert lazy <= set(pkg.__all__), sorted(lazy - set(pkg.__all__))
    for name in sorted(set(pkg.__all__) | subs):
        assert getattr(pkg, name) is not None, name
        assert name in dir(pkg)
    for name in subs:
        assert getattr(pkg, name).__name__ == f"{pkg.__name__}.{name}"
    with pytest.raises(AttributeError):
        _ = pkg.no_such_name


def _run(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_SRC), "PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu"},
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


_HEAVY = (
    "vllm_torchtpu.megakernel.core.exec_spec",
    "vllm_torchtpu.megakernel.core.ops",
    "vllm_torchtpu.megakernel.core.planner",
    "vllm_torchtpu.megakernel.core.pool",
    "vllm_torchtpu.megakernel.core.runtime.eligibility",
    "vllm_torchtpu.megakernel.core.runtime.policy",
)

_BLOCKED = """
import sys
for name in ("jax", "jaxlib", "torch", "vllm"):
    sys.modules[name] = None
import vllm_torchtpu.megakernel.core as mk
import vllm_torchtpu.megakernel.core.dma as dma
import vllm_torchtpu.megakernel.core.runtime as rt
assert mk.MegakernelLayoutError and mk.MegakernelConfigError
assert dma.__all__ and rt.__all__
assert sys.modules["jax"] is None
heavy = [m for m in {heavy!r} if m in sys.modules]
assert not heavy, heavy
print("OK")
"""


def test_package_imports_without_jax_torch_or_vllm() -> None:
    """With jax / torch / vllm unimportable, the packages and the stdlib
    modules still import: nothing pulls them in at package import."""
    assert _run(_BLOCKED.format(heavy=_HEAVY)) == "OK"


_LAZY_ONLY = """
import pathlib, sys, types
pkg = types.ModuleType("vllm_torchtpu")
pkg.__path__ = [str(pathlib.Path({src!r}) / "vllm_torchtpu")]
sys.modules["vllm_torchtpu"] = pkg
import vllm_torchtpu.megakernel.core as mk
import vllm_torchtpu.megakernel.core.dma
import vllm_torchtpu.megakernel.core.runtime
assert mk.MegakernelLayoutError and mk.MegakernelConfigError
assert "jax" not in sys.modules
heavy = [m for m in {heavy!r} if m in sys.modules]
assert not heavy, heavy
print("OK")
"""


def test_package_import_loads_no_megakernel_submodule() -> None:
    """With jax importable, importing the packages imports neither jax nor any
    heavy megakernel submodule."""
    assert _run(_LAZY_ONLY.format(src=str(_SRC), heavy=_HEAVY)) == "OK"
