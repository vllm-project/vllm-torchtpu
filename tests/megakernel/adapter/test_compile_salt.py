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
"""The megakernel compile-cache salt: the digest of a plugin's cache-layout
request (`None` without one), the same plugin resolution as the driver, and
the megakernel settings in the stock compile key."""

from __future__ import annotations

import ast
import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_torchtpu
import vllm_torchtpu.megakernel.adapter.agreement as agreement_mod
import vllm_torchtpu.megakernel.adapter.driver as driver_mod
from tests.megakernel.adapter import toy_plugin
from vllm_torchtpu.megakernel.adapter import config as cfg_mod
from vllm_torchtpu.megakernel.adapter import plugin as plugin_mod
from vllm_torchtpu.megakernel.adapter import salt as salt_mod
from vllm_torchtpu.megakernel.adapter import testing
from vllm_torchtpu.megakernel.adapter.salt import compile_cache_salt

_SRC = Path(vllm_torchtpu.__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)


_LAYOUT_PLUGIN = """
from vllm_torchtpu.megakernel.core.runtime.contract import TpuMonoSpec

def layout_request(vllm_config, options):
    return {"kv": "stacked", "n": options.get("n", 1)}

SPEC = TpuMonoSpec("lay", "1", create="lay_mk_plugin:create",
                          layout_request=layout_request)
"""


# A built-in shorthand and a checkpoint arch it claims: conftest's
# `mk_fake_template`, entered in the table for every test here.
_SHORTHAND = "mkfake"
_ENTRY = plugin_mod.Shorthand("mk_fake_template", ("MkFakeForCausalLM",), ("mk_fake",))


@pytest.fixture(autouse=True)
def _fake_template(mk_fake_template: str) -> None:
    assert plugin_mod.SHORTHANDS[mk_fake_template] == _ENTRY


_HASH_DEFS = (
    "_TPU_COMPILE_ENV_IGNORED",
    "_NATIVE_TPU_COMPILE_ENV_VARS",
    "_tpu_compile_env_factors",
    "compute_tpu_compilation_hash",
)


def _compute_hash_fn() -> Any:
    """`compute_tpu_compilation_hash`, compiled from source (with the real env
    factors) and CPU stand-ins for the package versions and the rest."""
    path = _SRC / "vllm_torchtpu" / "compilation" / "tpu_compiler.py"
    src = path.read_text(encoding="utf-8")
    defs = {
        name: n
        for n in ast.parse(src).body
        for name in (
            [n.name]
            if isinstance(n, ast.FunctionDef)
            else [t.id for t in n.targets if isinstance(t, ast.Name)]
            if isinstance(n, ast.Assign)
            else []
        )
        if name in _HASH_DEFS
    }
    assert set(defs) == set(_HASH_DEFS), sorted(defs)
    import hashlib

    import vllm_torchtpu.envs as envs

    ns: dict[str, Any] = {
        "Any": Any,
        "envs": envs,
        "torch": torch,
        "importlib_metadata": SimpleNamespace(version=lambda _n: "0"),
        "os": os,
        "utils": SimpleNamespace(get_dp_size=lambda _p: 1),
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "_iter_runtime_cache_key_files": lambda _root: [],
        "logger": logging.getLogger("t"),
        "compile_cache_salt": compile_cache_salt,
        "__file__": str(path),
        "VllmConfig": Any,
        "prefill_checkpoint_enabled": lambda _cfg: False,
    }
    for name in _HASH_DEFS:
        code = ast.get_source_segment(src, defs[name]) or ""
        exec(compile(code, str(path), "exec"), ns)
    return ns["compute_tpu_compilation_hash"]


def _vllm_config(megakernel: Any) -> Any:
    return SimpleNamespace(
        cache_config=SimpleNamespace(gpu_memory_utilization=0.9, num_gpu_blocks=10),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=4),
        speculative_config=None,
        parallel_config=None,
        model_config=SimpleNamespace(
            max_model_len=1024,
            hf_config=SimpleNamespace(architectures=[_ENTRY.architectures[0]]),
        ),
        additional_config={} if megakernel is None else {"megakernel": megakernel},
    )


def test_stock_compile_hash_keys_the_megakernel_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The megakernel settings key the stock graphs like any other setting (a
    plugin may change the stock model); a layout request adds its digest."""
    compute = _compute_hash_fn()
    base = compute(_vllm_config(None))
    # VLLM_TPU_MEGAKERNEL is one of the hashed env factors.
    monkeypatch.setenv("VLLM_TPU_MEGAKERNEL", "1")
    on = compute(_vllm_config(None))
    assert on != base
    monkeypatch.setenv("VLLM_TPU_MEGAKERNEL", "mode=tp")
    assert compute(_vllm_config(None)) not in (base, on)
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL")
    # additional_config["megakernel"] reaches the key through
    # VllmConfig.compute_hash (next test); this hash adds only the salt.
    shorthand_cfg = {"plugin": _SHORTHAND, "options": {"mode": "tp"}}
    for mk in (True, _SHORTHAND, shorthand_cfg):
        assert compute(_vllm_config(mk)) == base
    # A plugin with a cache-layout request keys its graphs by the request only.
    (tmp_path / "lay_mk_plugin.py").write_text(_LAYOUT_PLUGIN, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        lay = compute(_vllm_config({"plugin": "lay_mk_plugin"}))
        assert lay != base
        mk2 = {"plugin": "lay_mk_plugin", "options": {"n": 2}}
        assert compute(_vllm_config(mk2)) not in (base, lay)
    finally:
        sys.modules.pop("lay_mk_plugin", None)


def test_megakernel_additional_config_is_not_hash_ignored() -> None:
    """`VllmConfig.compute_hash` (the AOT / piecewise cache key) keeps
    `additional_config["megakernel"]`."""
    utils = pytest.importorskip("vllm_torchtpu.runner.utils", reason="needs vLLM")
    assert "megakernel" not in utils.HASH_IGNORED_ADDITIONAL_CONFIG_KEYS


def _salt_config(megakernel: Any) -> Any:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[_ENTRY.architectures[0]])
        ),
        additional_config={} if megakernel is None else {"megakernel": megakernel},
    )


def test_compile_cache_salt_digests_the_layout_request_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shorthand_cfg = {"plugin": _SHORTHAND, "options": {"mode": "tp"}}
    for mk in (None, False, True, _SHORTHAND, shorthand_cfg):
        assert compile_cache_salt(_salt_config(mk)) is None
    # Without the shorthand's model folder installed, too.
    gone = _ENTRY._replace(module="no_such_mk_folder.megakernel")
    with monkeypatch.context() as m:
        m.setitem(plugin_mod.SHORTHANDS, _SHORTHAND, gone)
        assert compile_cache_salt(_salt_config(True)) is None
    path = tmp_path / "lay_mk_plugin.py"
    path.write_text(_LAYOUT_PLUGIN, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        salt1 = compile_cache_salt(_salt_config({"plugin": "lay_mk_plugin"}))
        assert salt1 is not None and len(salt1) == 16
        mk2 = {"plugin": "lay_mk_plugin", "options": {"n": 2}}
        assert compile_cache_salt(_salt_config(mk2)) not in (None, salt1)
        # Only the request counts: editing the plugin's source doesn't re-key.
        path.write_text(_LAYOUT_PLUGIN + "\n# edited\n", encoding="utf-8")
        sys.modules.pop("lay_mk_plugin", None)
        assert compile_cache_salt(_salt_config({"plugin": "lay_mk_plugin"})) == salt1
    finally:
        sys.modules.pop("lay_mk_plugin", None)


_FRESH_IMPORT = """
import sys
for name in ("torch", "jax", "jaxlib", "vllm", "numpy"):
    sys.modules[name] = None
import vllm_torchtpu.megakernel.adapter.salt as s
assert s.compile_cache_salt
print("OK")
"""


def test_salt_imports_without_torch_jax_or_vllm() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _FRESH_IMPORT],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_SRC), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert out.returncode == 0 and out.stdout.strip() == "OK", out.stderr


mk_toy_plugin = toy_plugin.mk_toy_plugin


def test_driver_and_salt_resolve_the_same_plugin(
    mk_toy_plugin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: None)
    seen: dict[str, list[Any]] = {"driver": [], "salt": []}

    def spy(who: str) -> Any:
        def imp(ref: Any) -> Any:
            seen[who].append((ref.module, ref.attr))
            return cfg_mod.import_plugin(ref)

        return imp

    monkeypatch.setattr(driver_mod, "import_plugin", spy("driver"))
    monkeypatch.setattr(salt_mod, "import_plugin", spy("salt"))
    by_arch = SimpleNamespace(architectures=[_ENTRY.architectures[0]])
    by_type = SimpleNamespace(architectures=None, model_type=_ENTRY.model_types[0])
    other = SimpleNamespace(architectures=["LlamaForCausalLM"])
    cases = [
        (by_arch, True),
        (by_type, _SHORTHAND),
        (None, {"plugin": _SHORTHAND}),
        (other, True),
        (None, True),
        (other, {"plugin": "toy_mk_plugin"}),
    ]
    for hf, mk in cases:
        seen["driver"].clear()
        seen["salt"].clear()
        model = type(_ENTRY.architectures[0], (), {})()  # never used to resolve
        runner = testing.fake_runner(model=model, hf_config=hf, megakernel=mk)
        driver_mod.ProgramDriver.maybe_create(runner)
        salt_mod.compile_cache_salt(runner.vllm_config)
        assert seen["driver"] == seen["salt"], (hf, mk, seen)
    assert seen["driver"] == [("toy_mk_plugin", "SPEC")]
