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
"""Runtime config parsing (`VLLM_TPU_MEGAKERNEL` / `additional_config`),
plugin resolution and import, and plugin source hashes and fingerprints."""

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

from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.runtime.contract import TpuMonoSpec
from vllm_torchtpu.megakernel_runtime import config as cfg_mod
from vllm_torchtpu.megakernel_runtime import fingerprint as fp
from vllm_torchtpu.megakernel_runtime import plugin as plugin_mod
from vllm_torchtpu.megakernel_runtime.errors import MegakernelPluginError

_SRC = Path(plugin_mod.__file__).resolve().parents[2]
# A built-in shorthand and a checkpoint arch it claims.
_SHORTHAND, _ENTRY = next(iter(plugin_mod.SHORTHANDS.items()))
_ARCH_HF = SimpleNamespace(architectures=[_ENTRY.architectures[0]])


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


def test_source_hash_accepts_pyc_beside_its_source(mod_dir: Path) -> None:
    """`compileall -b` puts `m.pyc` next to `m.py`: hashed as the sources only."""
    src = _write(mod_dir / "pkg_b" / "m.py", "A = 1\n")
    _write(mod_dir / "pkg_b" / "__init__.py", "")
    pkg = _import("pkg_b")
    h = fp.source_hash(pkg)
    py_compile.compile(str(src), cfile=str(src.with_suffix(".pyc")))
    fp.clear_caches()
    assert fp.source_hash(pkg) == h
    src.unlink()  # now the .pyc has no source: sourceless
    fp.clear_caches()
    with pytest.raises(MegakernelPluginError, match="unhashable:sourceless"):
        fp.source_hash(pkg)


def test_source_hash_is_bounded(mod_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for i in range(4):
        _write(mod_dir / "big_pkg" / f"m{i}.py", "A = 1\n")
    _write(mod_dir / "big_pkg" / "__init__.py", "")
    monkeypatch.setattr(fp, "MAX_SOURCE_FILES", 3)
    with pytest.raises(MegakernelPluginError, match="plugin_source_too_large"):
        fp.source_hash(_import("big_pkg"))


def test_runtime_src_hash_covers_the_runtime_sources() -> None:
    files = []
    h = fp.runtime_src_hash()
    assert len(h) == 64
    pkg = _SRC / "vllm_torchtpu"
    for sub in ("megakernel", "megakernel_runtime"):
        files += [(f"{sub}/{rel}", p) for rel, p in fp._walk_py(pkg / sub, sub)]
    assert fp.hash_files(files) == h


def test_fingerprint_and_tokens() -> None:
    plug = TpuMonoSpec("toy", "1", create=lambda v, o: None)
    kw = dict(module="m", attr="SPEC", plugin=plug, source_hash="s", raw_env="")
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


def test_shorthands_do_not_collide_with_config_tokens() -> None:
    reserved = {"ep", "tp", "auto", *cfg_mod._TRUE_STRS, *cfg_mod._FALSE_STRS}
    assert not set(plugin_mod.SHORTHANDS) & reserved
    for entry in plugin_mod.SHORTHANDS.values():
        assert not entry.legacy_keys & cfg_mod.RUNTIME_KEYS


_FRESH_IMPORT = """
import sys
for name in ("torch", "jax", "jaxlib", "vllm", "numpy"):
    sys.modules[name] = None
import vllm_torchtpu.megakernel.runtime.contract as contract
import vllm_torchtpu.megakernel.errors as errors
import vllm_torchtpu.megakernel_runtime.plugin as p
import vllm_torchtpu.megakernel_runtime.config as c
import vllm_torchtpu.megakernel_runtime.fingerprint as f
assert p.RunnerView and c.parse_config and f.source_hash
assert issubclass(contract.LayoutDecline, errors.MegakernelLayoutError)
assert errors.EligibilityResult is contract.EligibilityResult
print("OK")
"""


def test_plugin_config_and_fingerprint_import_without_torch_jax_or_vllm() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _FRESH_IMPORT],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_SRC), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert out.returncode == 0 and out.stdout.strip() == "OK", out.stderr


# --- VLLM_TPU_MEGAKERNEL / additional_config -----------------------------------


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


def test_enabled_follows_the_last_explicit_source() -> None:
    # An explicit off holds whatever settings a later source adds.
    off = cfg_mod.parse_config("0", {"options": {"mode": "tp"}})
    assert not off.enabled and dict(off.options) == {"mode": "tp"}
    assert not cfg_mod.parse_config("ab=1", {"enabled": False}).enabled
    assert not cfg_mod.parse_config({"enabled": False}, {"options": {"a": 1}}).enabled
    assert not cfg_mod.parse_config("1", {"enabled": "0", "ab": True}).enabled
    # Without an explicit on / off, any settings turn it on; nothing is off.
    ab = cfg_mod.parse_config("ab=1")
    assert ab.enabled and ab.ab
    assert cfg_mod.parse_config({"options": {"mode": "tp"}}).enabled
    assert cfg_mod.parse_config("", {"options": {"mode": "tp"}}).enabled
    assert not cfg_mod.parse_config("", None).enabled
    assert cfg_mod.parse_config(None, {"stats_period_s": 3}).enabled
    assert not cfg_mod.parse_config().enabled
    assert not cfg_mod.parse_config(None, {}).enabled
    assert not cfg_mod.parse_config({"options": None}).enabled
    # A plugin key without `enabled` counts as on; an explicit `enabled` wins.
    assert cfg_mod.parse_config("0", {"plugin": "my.pkg"}).enabled
    assert not cfg_mod.parse_config({"plugin": "my.pkg", "enabled": False}).enabled
    assert not cfg_mod.parse_config({"plugin": "my.pkg"}, "0").enabled


@pytest.mark.parametrize(
    "src",
    [
        {"stats_period_s": True},
        '{"stats_period_s": false}',
        "stats_period_s=-1",
        {"stats_period_s": -0.5},
        "stats_period_s=nan",
        {"stats_period_s": float("inf")},
        "stats_period_s=-inf",
        "stats_period_s=soon",
        {"stats_period_s": [3]},
    ],
)
def test_invalid_stats_period_s_raises(src: Any) -> None:
    with pytest.raises(MegakernelConfigError, match="stats_period_s"):
        cfg_mod.parse_config(src)


def test_stats_period_s_accepts_finite_non_negative_numbers() -> None:
    assert cfg_mod.parse_config("stats_period_s=0").stats_period_s == 0.0
    assert cfg_mod.parse_config({"stats_period_s": 2}).stats_period_s == 2.0
    assert cfg_mod.parse_config('{"stats_period_s": "1.5"}').stats_period_s == 1.5


def test_mode_overrides_across_sources() -> None:
    cfg = cfg_mod.parse_config("ep", {"options": {"mode": "tp"}})
    assert cfg.enabled and dict(cfg.options) == {"mode": "tp"} and not cfg.legacy
    cfg = cfg_mod.parse_config({"options": {"mode": "tp"}}, "ep")
    assert dict(cfg.options) == {"mode": "ep"}
    # A top-level template key and `options` override each other too.
    cfg = cfg_mod.parse_config("ep", {"mode": "tp"})
    assert not cfg.options and dict(cfg.legacy) == {"mode": "tp"}
    cfg = cfg_mod.parse_config('{"mode": "tp"}', {"options": {"mode": "ep"}})
    assert dict(cfg.options) == {"mode": "ep"} and not cfg.legacy
    _, options = cfg_mod.resolve_plugin_ref(
        cfg_mod.parse_config(_SHORTHAND, "ep", {"mode": "tp"})
    )
    assert options == {"mode": "tp"}
    # Only one source setting a key both ways is an error.
    with pytest.raises(MegakernelConfigError, match="both at top level"):
        cfg_mod.parse_config({"mode": "tp", "options": {"mode": "ep"}})


def test_ep_with_an_explicit_plugin() -> None:
    cfg = cfg_mod.parse_config("ep", {"plugin": "my.pkg"})
    ref, options = cfg_mod.resolve_plugin_ref(cfg)
    assert (ref.module, ref.shorthand) == ("my.pkg", None)
    assert options == {"mode": "ep"}


def test_bare_shorthand_selects_its_template() -> None:
    for src in (_SHORTHAND, f" {_SHORTHAND.upper()} "):
        cfg = cfg_mod.parse_config(src)
        assert cfg.enabled and cfg.plugin == _SHORTHAND
    cfg = cfg_mod.parse_config({"plugin": _SHORTHAND})
    ref, options = cfg_mod.resolve_plugin_ref(cfg, hf_config=None)
    assert ref.module == _ENTRY.module and options == {}
    with pytest.raises(MegakernelConfigError, match="Invalid megakernel config entry"):
        cfg_mod.parse_config("glm5")


def test_plugin_key_and_arch_resolution() -> None:
    cfg = cfg_mod.parse_config("plugin=my.pkg:THING")
    ref, _ = cfg_mod.resolve_plugin_ref(cfg)
    assert (ref.module, ref.attr, ref.shorthand) == ("my.pkg", "THING", None)
    ref, _ = cfg_mod.resolve_plugin_ref(cfg_mod.parse_config("plugin=a.b"))
    assert (ref.module, ref.attr) == ("a.b", "SPEC")
    on = cfg_mod.parse_config("1")
    assert cfg_mod.resolve_plugin_ref(on, SimpleNamespace(architectures=["X"])) is None
    by_type = SimpleNamespace(architectures=None, model_type=_ENTRY.model_types[0])
    assert cfg_mod.resolve_plugin_ref(on, by_type)[0].shorthand == _SHORTHAND
    # Only the HF config resolves the arch.
    assert cfg_mod.resolve_plugin_ref(on, None) is None
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


def test_top_level_template_keys_forward_to_shorthands_only() -> None:
    cfg = cfg_mod.parse_config(json.dumps({"plugin": "my.pkg", "profile": "x"}))
    with pytest.raises(MegakernelConfigError, match="unknown key 'profile'"):
        cfg_mod.resolve_plugin_ref(cfg)
    legacy = sorted(_ENTRY.legacy_keys)[0]
    cfg = cfg_mod.parse_config(json.dumps({"plugin": _SHORTHAND, legacy: "x"}))
    _, options = cfg_mod.resolve_plugin_ref(cfg)
    assert options == {legacy: "x"}


def test_import_plugin_errors(mod_dir: Path) -> None:
    ref = cfg_mod.PluginRef("no_such_plugin_mod")
    with pytest.raises(MegakernelConfigError, match="no_such_plugin_mod"):
        cfg_mod.import_plugin(ref)
    _write(mod_dir / "not_a_plugin.py", "SPEC = 3\n")
    with pytest.raises(MegakernelConfigError, match="not a TpuMonoSpec"):
        cfg_mod.import_plugin(cfg_mod.PluginRef("not_a_plugin"))
    _write(
        mod_dir / "future_plugin.py",
        """
        from vllm_torchtpu.megakernel.runtime.contract import TpuMonoSpec
        SPEC = TpuMonoSpec("f", "1", create=".x:y", api_version=99)
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


def test_missing_shorthand_template_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shorthands are imported by string: without the model folder installed,
    the config names it; an explicit plugin reference that doesn't import is
    an error."""
    gone = _ENTRY._replace(module="no_such_mk_folder.megakernel")
    monkeypatch.setitem(plugin_mod.SHORTHANDS, _SHORTHAND, gone)
    ref, _ = cfg_mod.resolve_plugin_ref(cfg_mod.parse_config("1"), _ARCH_HF)
    assert cfg_mod.missing_template(ref, _ARCH_HF) == (
        f"Megakernel: no megakernel template for arch {_ENTRY.architectures[0]} "
        "(no_such_mk_folder.megakernel is not installed); skipping."
    )
    explicit = cfg_mod.PluginRef("no_such_mk_folder.megakernel")
    assert cfg_mod.missing_template(explicit, _ARCH_HF) is None
    with pytest.raises(MegakernelConfigError, match="failed to import"):
        cfg_mod.import_plugin(explicit)
    assert cfg_mod.missing_template(cfg_mod.PluginRef("json", shorthand="x")) is None


def test_missing_template_reraises_a_broken_import_in_a_parent(
    mod_dir: Path,
) -> None:
    """`find_spec` imports the parent packages: a missing module imported by
    a parent is a real error, not "template not installed"."""
    _write(mod_dir / "mk_broken_parent" / "__init__.py", "import mk_no_such_dep\n")
    _write(mod_dir / "mk_broken_parent" / "megakernel" / "__init__.py", "")
    ref = cfg_mod.PluginRef("mk_broken_parent.megakernel", shorthand="x")
    importlib.invalidate_caches()
    with pytest.raises(ModuleNotFoundError) as exc:
        cfg_mod.missing_template(ref, _ARCH_HF)
    assert exc.value.name == "mk_no_such_dep"
    # A missing parent package (or the package itself) is "not installed".
    for module in ("mk_no_such_parent.megakernel", "mk_broken_parent.missing"):
        sys.modules.pop("mk_broken_parent", None)
        _write(mod_dir / "mk_broken_parent" / "__init__.py", "")
        importlib.invalidate_caches()
        msg = cfg_mod.missing_template(
            cfg_mod.PluginRef(module, shorthand="x"), _ARCH_HF
        )
        assert msg is not None and f"({module} is not installed)" in msg
