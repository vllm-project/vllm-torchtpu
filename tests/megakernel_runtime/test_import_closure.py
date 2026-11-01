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
"""The model-agnostic runtime imports without the megakernel library.

With every `vllm_torchtpu.megakernel` module blocked except the package, its
errors and the framework-free `runtime.contract` / `runtime.stats`, the
runtime modules that land first (config, fingerprint, salt, plugin, hooks,
testing, agreement, guards, driver, the generic jax bridge and weight source)
import fine. The library-bound pieces (`exported_bridge`, `weight_collector`,
`step_meta_ops`, `pool_facts`) are not covered.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import vllm_torchtpu.megakernel as mk

_MK = Path(mk.__file__).resolve().parent
_SRC = _MK.parents[1]
_ALLOWED = frozenset(
    {
        "vllm_torchtpu.megakernel",
        "vllm_torchtpu.megakernel.errors",
        "vllm_torchtpu.megakernel.runtime",
        "vllm_torchtpu.megakernel.runtime.contract",
        "vllm_torchtpu.megakernel.runtime.stats",
    }
)
# Must stay importable without the library (the bridge slices B1-B5, B8).
_RUNTIME = (
    "vllm_torchtpu.megakernel.runtime.contract",
    "vllm_torchtpu.megakernel.runtime.stats",
    "vllm_torchtpu.megakernel_runtime.config",
    "vllm_torchtpu.megakernel_runtime.fingerprint",
    "vllm_torchtpu.megakernel_runtime.salt",
    "vllm_torchtpu.megakernel_runtime.plugin",
    "vllm_torchtpu.megakernel_runtime.hooks",
    "vllm_torchtpu.megakernel_runtime.testing",
    "vllm_torchtpu.megakernel_runtime.agreement",
    "vllm_torchtpu.megakernel_runtime.guards",
    "vllm_torchtpu.megakernel_runtime.driver",
    "vllm_torchtpu.megakernel_runtime.jax_bridge",
    "vllm_torchtpu.megakernel_runtime.weight_source",
)


def _library_modules() -> list[str]:
    names = []
    for path in sorted(_MK.rglob("*.py")):
        parts = list(path.relative_to(_SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names.append(".".join(parts))
    return names


_SCRIPT = """
import importlib, json, sys
blocked, wanted = json.loads(sys.argv[1]), json.loads(sys.argv[2])
for name in blocked:
    sys.modules[name] = None
for name in wanted:
    importlib.import_module(name)
loaded = sorted(n for n in sys.modules if n.startswith("vllm_torchtpu.megakernel."))
print(json.dumps([n for n in loaded if sys.modules[n] is not None]))
"""


def test_runtime_imports_without_the_megakernel_library() -> None:
    library = _library_modules()
    blocked = [n for n in library if n not in _ALLOWED]
    for must_block in (
        "vllm_torchtpu.megakernel.cache",
        "vllm_torchtpu.megakernel.runtime.policy",
        "vllm_torchtpu.megakernel.runtime.checks",
        "vllm_torchtpu.megakernel.runtime.step_meta",
        "vllm_torchtpu.megakernel.weights",
    ):
        assert must_block in blocked, must_block
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT, json.dumps(blocked), json.dumps(_RUNTIME)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_SRC), "PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu"},
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert loaded <= _ALLOWED, sorted(loaded - _ALLOWED)
