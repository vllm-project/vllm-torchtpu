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
    "vllm_torchtpu.megakernel.core.kernel",
    "vllm_torchtpu.megakernel.core.cache",
    "vllm_torchtpu.megakernel.core.modules",
    "vllm_torchtpu.megakernel.core.planner",
    "vllm_torchtpu.megakernel.core.runtime.checks",
    "vllm_torchtpu.megakernel.core.runtime.exec_policy",
    "vllm_torchtpu.megakernel.core.runtime.policy",
)

_BLOCKED = """
import sys
for name in ("jax", "jaxlib", "torch", "vllm"):
    sys.modules[name] = None
import vllm_torchtpu.megakernel.core as mk
import vllm_torchtpu.megakernel.core.dma as dma
import vllm_torchtpu.megakernel.core.runtime as rt
import vllm_torchtpu.megakernel.core.runtime.contract as contract
assert mk.MegakernelLayoutError and mk.MegakernelConfigError
assert mk.EligibilityResult is contract.EligibilityResult is rt.EligibilityResult
assert contract.TpuMonoSpec("toy", "1", create="m:create").parallelism == {{"tp"}}
assert dma.__all__
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
import vllm_torchtpu.megakernel.core.runtime.contract
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


_REGISTRIES = """
import vllm_torchtpu.megakernel.core as mk
# Registry lookups before anything touches mk.modules.
assert mk.Tiles().module("gqa") is not None
assert mk.Tiles().with_modules({"gqa": {"kv_sub": 2}}).module("gqa").kv_sub == 2
from vllm_torchtpu.megakernel.core import module_options
assert {"gqa", "gdn", "moe"} <= set(module_options.registered_tunables())
# Models and their profiles live in model folders, which register explicitly.
assert mk.list_models() == () and mk.list_profiles() == ("default",)
# The fused pool I/O implementation registers on the first lookup.
from vllm_torchtpu.megakernel.core.fused_io_hook import fused_io, fused_io_available
assert fused_io_available() and fused_io() is not None
print("OK")
"""


def test_registry_lookups_load_the_builtins_first() -> None:
    assert _run(_REGISTRIES) == "OK"


_FIRST_LOOKUP = """
import importlib
from vllm_torchtpu.megakernel import core as mk
from vllm_torchtpu.megakernel.core import _builtins

calls = []
real_import = importlib.import_module

def fake_import(name, package=None):
    if name == "vllm_torchtpu.megakernel.core._reentrant_builtin":
        calls.append(name)
        assert len(calls) == 1, f"re-entered builtin import: {calls}"
        # Re-entrant lookup while `profile` is mid-import (building
        # DEFAULT_PROFILE) and `_reentrant_builtin` is mid-import: fails with
        # RecursionError unless `_LOADED.add(name)` runs before import.
        p = mk.Profile(
            name="reentrant-exact",
            numerics="exact",
            tiles=lambda tc: mk.default_tiles(tc),
        )
        assert p.numerics == "exact"
        return None
    return real_import(name, package)

importlib.import_module = fake_import
_builtins.TUNABLES = ("_reentrant_builtin", *_builtins.TUNABLES)
_builtins._LOADED.discard("_reentrant_builtin")

assert mk.get_profile("default").numerics == "exact"
mk.register_profile(
    mk.Profile(name="test-approx-profile", numerics="approx", fused_pool_io=True)
)
assert mk.get_profile("test-approx-profile").numerics == "approx"
assert calls == ["vllm_torchtpu.megakernel.core._reentrant_builtin"], calls
print("OK")
"""


def test_first_lookup_is_a_profile_lookup() -> None:
    assert _run(_FIRST_LOOKUP) == "OK"


_FUNCTION_NAMED_SUBMODULE = """
import sys
import types
from vllm_torchtpu.megakernel import core as mk
import vllm_torchtpu.megakernel.core.verify_module
assert "vllm_torchtpu.megakernel.core.verify_module" in sys.modules
assert isinstance(mk.verify_module, types.FunctionType), mk.verify_module
from vllm_torchtpu.megakernel.core import verify_module
assert verify_module is mk.verify_module
print("OK")
"""


def test_verify_module_stays_the_function_after_its_submodule_loads() -> None:
    assert _run(_FUNCTION_NAMED_SUBMODULE) == "OK"
