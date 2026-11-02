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
request (`None` without one), and stock compile keys unchanged."""

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
from vllm_torchtpu.megakernel_runtime import plugin as plugin_mod
from vllm_torchtpu.megakernel_runtime.salt import compile_cache_salt

_SRC = Path(vllm_torchtpu.__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)


_LAYOUT_PLUGIN = """
from vllm_torchtpu.megakernel_runtime.plugin import TpuMonoSpec

def layout_request(vllm_config, options):
    return {"kv": "stacked", "n": options.get("n", 1)}

SPEC = TpuMonoSpec("lay", "1", create="lay_mk_plugin:create",
                          layout_request=layout_request)
"""


def _compute_hash_fn() -> Any:
    """`compute_tpu_compilation_hash`, compiled from source with CPU stand-ins."""
    path = _SRC / "vllm_torchtpu" / "compilation" / "tpu_compiler.py"
    src = path.read_text(encoding="utf-8")
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "compute_tpu_compilation_hash"
    )
    import hashlib

    ns: dict[str, Any] = {
        "torch": torch,
        "importlib_metadata": SimpleNamespace(version=lambda _n: "0"),
        "_tpu_compile_env_factors": dict,
        "_NATIVE_TPU_COMPILE_ENV_VARS": (),
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
    }
    exec(compile(ast.get_source_segment(src, fn) or "", str(path), "exec"), ns)
    return ns["compute_tpu_compilation_hash"]


def _vllm_config(megakernel: Any) -> Any:
    return SimpleNamespace(
        cache_config=SimpleNamespace(gpu_memory_utilization=0.9, num_gpu_blocks=10),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=4),
        speculative_config=None,
        parallel_config=None,
        model_config=SimpleNamespace(
            max_model_len=1024,
            hf_config=SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"]),
        ),
        additional_config={} if megakernel is None else {"megakernel": megakernel},
    )


def test_stock_compile_hash_unchanged_with_megakernel_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compute = _compute_hash_fn()
    base = compute(_vllm_config(None))
    for mk in (True, "qwen35", {"plugin": "qwen35", "options": {"mode": "tp"}}):
        assert compute(_vllm_config(mk)) == base
    monkeypatch.setenv("VLLM_TPU_MEGAKERNEL", "1")
    assert compute(_vllm_config(None)) == base
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL")
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


# A built-in shorthand and a checkpoint arch it claims.
_SHORTHAND, _ENTRY = next(iter(plugin_mod.SHORTHANDS.items()))


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
import vllm_torchtpu.megakernel_runtime.salt as s
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
