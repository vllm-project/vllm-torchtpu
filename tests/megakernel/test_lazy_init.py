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
"""The lazy (PEP 562) `megakernel` and `megakernel.dma` package namespaces."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import vllm_torchtpu.megakernel as mk
import vllm_torchtpu.megakernel.dma as mk_dma

_SRC = Path(mk.__file__).resolve().parents[2]


@pytest.mark.parametrize("pkg", [mk, mk_dma], ids=["megakernel", "dma"])
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
    "vllm_torchtpu.megakernel.ops",
    "vllm_torchtpu.megakernel.planner",
    "vllm_torchtpu.megakernel.pool",
    "vllm_torchtpu.megakernel.profile",
)

_BLOCKED = """
import sys
for name in ("jax", "jaxlib", "torch", "vllm"):
    sys.modules[name] = None
import vllm_torchtpu.megakernel as mk
import vllm_torchtpu.megakernel._builtins as _builtins
import vllm_torchtpu.megakernel.errors as errors
assert errors.MegakernelLayoutError and errors.MegakernelConfigError
_builtins.ensure_builtins()
heavy = [m for m in {heavy!r} if m in sys.modules]
assert not heavy, heavy
print("OK")
"""


def test_package_imports_without_jax_torch_or_vllm() -> None:
    """With jax / torch / vllm unimportable, the packages and the stdlib
    modules still import: nothing pulls them in at package import."""
    assert _run(_BLOCKED.format(heavy=_HEAVY)) == "OK"


_LAZY_ONLY = """
import sys
import vllm_torchtpu
root_jax = "jax" in sys.modules
import vllm_torchtpu.megakernel.errors
heavy = [m for m in {heavy!r} if m in sys.modules]
assert not heavy, heavy
print("root_imports_jax" if root_jax else "no_jax")
"""


def test_package_import_loads_no_megakernel_submodule() -> None:
    """With jax importable, importing the packages imports no heavy submodule."""
    assert _run(_LAZY_ONLY.format(heavy=_HEAVY)) in ("root_imports_jax", "no_jax")


_FIRST_LOOKUP = """
from vllm_torchtpu import megakernel as mk
assert mk.get_profile("default").numerics == "exact"
print("OK")
"""


def test_first_lookup_is_a_profile_lookup() -> None:
    assert _run(_FIRST_LOOKUP) == "OK"
