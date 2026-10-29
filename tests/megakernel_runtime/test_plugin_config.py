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
"""Tests for the plugin interface, runtime config parsing and source hashes."""

from __future__ import annotations

import importlib
import json
import py_compile
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_torchtpu.megakernel_runtime import config as cfg_mod
from vllm_torchtpu.megakernel_runtime import fingerprint as fp
from vllm_torchtpu.megakernel_runtime import plugin as plugin_mod
from vllm_torchtpu.megakernel_runtime.errors import (
    MegakernelConfigError,
    MegakernelPluginError,
)

_SRC = Path(plugin_mod.__file__).resolve().parents[2]
_QWEN_HF = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])


# --- Source hash ----------------------------------------------------------------


@pytest.fixture
def mod_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh `sys.path` entry; modules imported from it are dropped after."""
    root = tmp_path / "site"
    root.mkdir()
    monkeypatch.syspath_prepend(str(root))
    before = set(sys.modules)
    yield root
    for name in set(sys.modules) - before:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    fp.clear_caches()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def _import(name: str) -> Any:
    importlib.invalidate_caches()
    return importlib.import_module(name)


def test_source_hash_package_covers_py_files_only(mod_dir: Path) -> None:
    _write(mod_dir / "pkg_a" / "__init__.py", "X = 1\n")
    _write(mod_dir / "pkg_a" / "sub" / "m.py", "Y = 2\n")
    _write(mod_dir / "pkg_a" / ".hidden" / "h.py", "Z = 3\n")
    _write(mod_dir / "pkg_a" / "data.txt", "not hashed\n")
    mod = _import("pkg_a")
    h1 = fp.source_hash(mod)
    # Hidden dirs, non-.py files and mtimes don't count.
    (mod_dir / "pkg_a" / ".hidden" / "h.py").write_text("Z = 4\n", encoding="utf-8")
    (mod_dir / "pkg_a" / "data.txt").write_text("changed\n", encoding="utf-8")
    fp.clear_caches()
    assert fp.source_hash(mod) == h1
    # Any .py byte does, also in subpackages.
    (mod_dir / "pkg_a" / "sub" / "m.py").write_text("Y = 3\n", encoding="utf-8")
    assert fp.source_hash(mod) == h1, "computed once per process"
    fp.clear_caches()
    assert fp.source_hash(mod) != h1


def test_source_hash_single_file_module_is_that_file(mod_dir: Path) -> None:
    _write(mod_dir / "single_mod.py", "A = 1\n")
    _write(mod_dir / "sibling.py", "B = 1\n")
    mod = _import("single_mod")
    h1 = fp.source_hash(mod)
    (mod_dir / "sibling.py").write_text("B = 2\n", encoding="utf-8")
    fp.clear_caches()
    assert fp.source_hash(mod) == h1


def test_source_hash_fails_closed_for_namespace_packages(mod_dir: Path) -> None:
    _write(mod_dir / "ns_pkg" / "m.py", "A = 1\n")
    mod = _import("ns_pkg")
    with pytest.raises(
        MegakernelPluginError, match="plugin_source_unhashable:namespace"
    ):
        fp.source_hash(mod)
    # An explicit source_root makes it hashable.
    assert fp.source_hash(mod, source_root=str(mod_dir / "ns_pkg"))


def test_source_hash_fails_closed_for_sourceless_installs(mod_dir: Path) -> None:
    src = _write(mod_dir / "build" / "nosrc_mod.py", "A = 1\n")
    py_compile.compile(str(src), cfile=str(mod_dir / "nosrc_mod.pyc"))
    mod = _import("nosrc_mod")
    assert mod.__file__.endswith(".pyc")
    with pytest.raises(MegakernelPluginError, match="unhashable:sourceless"):
        fp.source_hash(mod)
    # A package carrying a stray .pyc next to its sources is sourceless too.
    _write(mod_dir / "pkg_pyc" / "__init__.py", "")
    py_compile.compile(str(src), cfile=str(mod_dir / "pkg_pyc" / "x.pyc"))
    with pytest.raises(MegakernelPluginError, match="unhashable:sourceless"):
        fp.source_hash(_import("pkg_pyc"))


def test_source_hash_fails_closed_for_zip_imports(
    mod_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zpath = mod_dir.parent / "plugins.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("zipped_mod.py", "A = 1\n")
    monkeypatch.syspath_prepend(str(zpath))
    mod = _import("zipped_mod")
    with pytest.raises(MegakernelPluginError, match="unhashable:zip"):
        fp.source_hash(mod)
    sys.modules.pop("zipped_mod", None)


def test_source_hash_is_bounded(mod_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for i in range(4):
        _write(mod_dir / "big_pkg" / f"m{i}.py", "A = 1\n")
    _write(mod_dir / "big_pkg" / "__init__.py", "")
    monkeypatch.setattr(fp, "MAX_SOURCE_FILES", 3)
    with pytest.raises(MegakernelPluginError, match="plugin_source_too_large"):
        fp.source_hash(_import("big_pkg"))


def test_runtime_src_hash_excludes_templates() -> None:
    files = []
    h = fp.runtime_src_hash()
    assert len(h) == 64
    pkg = _SRC / "vllm_torchtpu"
    for sub in ("megakernel", "megakernel_runtime"):
        files += [
            (f"{sub}/{rel}", p)
            for rel, p in fp._walk_py(pkg / sub, sub)
            if not (sub == "megakernel_runtime" and rel.startswith("models/"))
        ]
    assert fp.hash_files(files) == h


def test_fingerprint_and_tokens() -> None:
    plug = plugin_mod.MegakernelPlugin("toy", "1", create=lambda v, o: None)
    kw = dict(module="m", attr="PLUGIN", plugin=plug, source_hash="s", raw_env="")
    f1 = fp.plugin_fingerprint(config={"options": {"a": 1}}, **kw)
    assert f1 == fp.plugin_fingerprint(config={"options": {"a": 1}}, **kw)
    assert f1 != fp.plugin_fingerprint(config={"options": {"a": 2}}, **kw)
    assert f1 != fp.plugin_fingerprint(
        config={"options": {"a": 1}}, **{**kw, "source_hash": "t"}
    )
    tok = fp.agreement_token(f1, "r")
    assert 0 < tok < 2**60 and -tok >= -(2**63)
    assert tok != fp.agreement_token(f1, "r2")
    assert len(fp.src8("a", "b")) == 8


# --- Plugin interface --------------------------------------------------------------


def test_plugin_validates_capabilities_and_normalizes() -> None:
    plug = plugin_mod.MegakernelPlugin(
        "toy", "1", create="x:y", capabilities={"spec_decode", "mrope"}
    )
    assert plug.capabilities == frozenset({"spec_decode", "mrope"})
    with pytest.raises(MegakernelConfigError, match="unknown capabilities"):
        plugin_mod.MegakernelPlugin("toy", "1", create="x:y", capabilities={"warp"})
    with pytest.raises(MegakernelConfigError):
        plugin_mod.MegakernelPlugin("", "1", create="x:y")


def test_lazy_library_exports() -> None:
    from vllm_torchtpu.megakernel.errors import EligibilityResult, MegakernelLayoutError
    from vllm_torchtpu.megakernel.runtime import StepDecision

    assert plugin_mod.EligibilityResult is EligibilityResult
    assert plugin_mod.StepDecision is StepDecision
    assert issubclass(plugin_mod.LayoutDecline, MegakernelLayoutError)
    dec = plugin_mod.Decline("too_small", "msg")
    assert dec.result == EligibilityResult(False, "too_small", "msg")
    with pytest.raises(AttributeError):
        _ = plugin_mod.NoSuchName


def test_shorthands_do_not_collide_with_config_tokens() -> None:
    reserved = {"ep", "tp", "auto", *cfg_mod._TRUE_STRS, *cfg_mod._FALSE_STRS}
    assert not set(plugin_mod.SHORTHANDS) & reserved
    for name in plugin_mod.SHORTHANDS:
        assert name == name.lower() and name.isidentifier()
        assert not plugin_mod.SHORTHANDS[name].legacy_keys & cfg_mod.RUNTIME_KEYS


_FRESH_IMPORT = """
import sys
for name in ("torch", "jax", "jaxlib", "vllm", "numpy"):
    sys.modules[name] = None
import vllm_torchtpu.megakernel_runtime.plugin as p
import vllm_torchtpu.megakernel_runtime.config as c
import vllm_torchtpu.megakernel_runtime.fingerprint as f
import vllm_torchtpu.megakernel_runtime.salt as s
assert p.MegakernelPlugin and c.parse_config and f.source_hash
assert s.compile_cache_salt
try:
    p.LayoutDecline
except ImportError:
    pass
else:
    raise AssertionError("LayoutDecline needs the library (jax)")
print("OK")
"""


def test_plugin_and_config_import_without_torch_jax_or_vllm() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _FRESH_IMPORT],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_SRC), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert out.returncode == 0 and out.stdout.strip() == "OK", out.stderr


# --- VLLM_TPU_MEGAKERNEL / additional_config matrix ---------------------------------

_TEMPLATE_KEYS = (
    "mode",
    "profile",
    "buckets",
    "verify_buckets",
    "buckets_explicit",
    "verify_buckets_explicit",
    "fused_pool_io",
    "fused_pool_io_max_b",
    "fused_pool_io_min_b_hi",
    "pool_gather",
    "compile_hlo_check",
    "explicit",
)

_LEGACY_ENV = (
    None,
    "",
    "0",
    "false",
    "no",
    "off",
    "none",
    "1",
    "true",
    "YES",
    "on",
    "ep",
    "tp",
    '{"profile": "qwen35-baseline"}',
    '{"enabled": false, "mode": "tp"}',
    '{"mode": "auto", "buckets": "8,16", "verify_buckets": [4, 8]}',
    "profile=qwen35-candidate30,buckets=8:16",
    "verify_buckets=4:8,fused_pool_io=auto,pool_gather=0",
    "ab=1,stats_period_s=5",
    "fused_pool_io_max_b=16,fused_pool_io_min_b_hi=48,compile_hlo_check=0",
)
_LEGACY_ADD = (
    None,
    True,
    False,
    "ep",
    "0",
    {"profile": "qwen35-candidate30-fused"},
    {"buckets": [8, 16], "ab": True},
    {"enabled": "0"},
    {"mode": "tp", "stats_period_s": 2},
)


@pytest.mark.parametrize("env", _LEGACY_ENV)
@pytest.mark.parametrize("add", _LEGACY_ADD)
def test_legacy_inputs_match_parse_runtime_config(env: Any, add: Any) -> None:
    """Legacy syntaxes resolve to what `parse_runtime_config` gave before: the
    runtime keys directly, the template settings through `options`."""
    from vllm_torchtpu.megakernel.runtime import parse_runtime_config

    old = parse_runtime_config(env, add)
    new = cfg_mod.parse_config(env, add, raw_env=env)
    assert new.enabled == old["enabled"]
    assert new.ab == old["ab"] and new.stats_period_s == old["stats_period_s"]
    resolved = cfg_mod.resolve_plugin_ref(new, _QWEN_HF)
    assert resolved is not None
    ref, options = resolved
    assert ref.shorthand == "qwen35"
    tmpl = parse_runtime_config(options)
    assert {k: tmpl[k] for k in _TEMPLATE_KEYS} == {k: old[k] for k in _TEMPLATE_KEYS}


def test_additional_config_overrides_env_per_key() -> None:
    cfg = cfg_mod.collect_config(
        '{"options": {"profile": "a", "mode": "ep"}, "ab": true}',
        {"megakernel": {"options": {"mode": "tp"}, "stats_period_s": 3}},
    )
    assert cfg.enabled and cfg.ab and cfg.stats_period_s == 3.0
    assert dict(cfg.options) == {"profile": "a", "mode": "tp"}
    off = cfg_mod.collect_config("1", {"megakernel": False})
    assert not off.enabled
    assert not cfg_mod.collect_config("1", {"megakernel": "0"}).enabled
    assert not cfg_mod.collect_config("", {"other": 1}).enabled


def test_bare_shorthand_is_new_syntax() -> None:
    for src in ("qwen35", " QWEN35 "):
        cfg = cfg_mod.parse_config(src)
        assert cfg.enabled and cfg.plugin == "qwen35"
    cfg = cfg_mod.parse_config({"plugin": "qwen35"})
    ref, options = cfg_mod.resolve_plugin_ref(cfg, hf_config=None)
    assert ref.module == plugin_mod.SHORTHANDS["qwen35"].module and options == {}
    with pytest.raises(MegakernelConfigError, match="Invalid megakernel config entry"):
        cfg_mod.parse_config("glm5")


def test_plugin_key_and_arch_resolution() -> None:
    cfg = cfg_mod.parse_config("plugin=my.pkg:THING")
    ref, _ = cfg_mod.resolve_plugin_ref(cfg)
    assert (ref.module, ref.attr, ref.shorthand) == ("my.pkg", "THING", None)
    ref, _ = cfg_mod.resolve_plugin_ref(cfg_mod.parse_config("plugin=a.b"))
    assert (ref.module, ref.attr) == ("a.b", "PLUGIN")
    on = cfg_mod.parse_config("1")
    assert cfg_mod.resolve_plugin_ref(on, SimpleNamespace(architectures=["X"])) is None
    by_type = SimpleNamespace(architectures=None, model_type="qwen3_5_moe")
    assert cfg_mod.resolve_plugin_ref(on, by_type)[0].shorthand == "qwen35"
    by_cls = type("Qwen3_5MoeForCausalLM", (), {})()
    assert cfg_mod.resolve_plugin_ref(on, None, by_cls)[0].shorthand == "qwen35"
    with pytest.raises(MegakernelConfigError, match="invalid plugin"):
        cfg_mod.parse_config({"plugin": "a.b:not an attr"})


@pytest.mark.parametrize(
    "src, match",
    [
        ('{"profiel": "x"}', "unknown key 'profiel'"),
        ({"buckets": "8", "options": {"buckets": "16"}}, "both at top level"),
        ('{"options": "nope"}', "options must be a JSON object"),
        ("enabled=maybe", "invalid boolean"),
        ("{not json", "invalid JSON"),
        ("[1, 2]", "Invalid megakernel config entry"),
    ],
)
def test_config_errors_name_the_problem(src: Any, match: str) -> None:
    with pytest.raises(MegakernelConfigError, match=match):
        cfg_mod.parse_config(src)


def test_legacy_keys_only_forward_to_shorthand_templates() -> None:
    cfg = cfg_mod.parse_config(json.dumps({"plugin": "my.pkg", "profile": "x"}))
    with pytest.raises(MegakernelConfigError, match="unknown key 'profile'"):
        cfg_mod.resolve_plugin_ref(cfg)
    cfg = cfg_mod.parse_config(json.dumps({"plugin": "qwen35", "mode": "tp"}))
    _, options = cfg_mod.resolve_plugin_ref(cfg)
    assert options == {"mode": "tp"}


def test_import_plugin_errors(mod_dir: Path) -> None:
    ref = cfg_mod.PluginRef("no_such_plugin_mod")
    with pytest.raises(MegakernelConfigError, match="no_such_plugin_mod"):
        cfg_mod.import_plugin(ref)
    _write(mod_dir / "not_a_plugin.py", "PLUGIN = 3\n")
    with pytest.raises(MegakernelConfigError, match="not a MegakernelPlugin"):
        cfg_mod.import_plugin(cfg_mod.PluginRef("not_a_plugin"))
    _write(
        mod_dir / "future_plugin.py",
        """
        from vllm_torchtpu.megakernel_runtime.plugin import MegakernelPlugin
        PLUGIN = MegakernelPlugin("f", "1", create=".x:y", api_version=99)
        """,
    )
    with pytest.raises(MegakernelConfigError, match="api_version 99"):
        cfg_mod.import_plugin(cfg_mod.PluginRef("future_plugin"))
    _write(mod_dir / "rel_pkg" / "__init__.py", "")
    _write(mod_dir / "rel_pkg" / "impl.py", "def create(v, o):\n    return 7\n")
    pkg = _import("rel_pkg")
    fn = cfg_mod.resolve_callable(".impl:create", pkg, what="create")
    assert fn(None, None) == 7
    with pytest.raises(MegakernelConfigError, match="can't resolve create"):
        cfg_mod.resolve_callable(".impl:missing", pkg, what="create")
    with pytest.raises(MegakernelConfigError, match="'module:attr'"):
        cfg_mod.resolve_callable("noattr", pkg, what="create")
