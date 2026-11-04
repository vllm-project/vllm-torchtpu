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
"""Model folders found by the checkpoints their `SPEC` declares: no shorthand
table entry needed, the folder's `create` target never imported, broken
folders skipped, and the explicit table winning."""

from __future__ import annotations

import importlib
import logging
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel_runtime import config as cfg_mod
from vllm_torchtpu.megakernel_runtime import plugin as plugin_mod
from vllm_torchtpu.megakernel_runtime.plugin import (
    MODELS_PACKAGE,
    SHORTHANDS,
    Shorthand,
    TpuMonoSpec,
    discovered_shorthands,
    shorthands,
)

_SPEC = """
from vllm_torchtpu.megakernel_runtime.plugin import TpuMonoSpec

SPEC = TpuMonoSpec("{name}", "1", create=".adapter:create", {fields})
"""


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An extra `MODELS_PACKAGE` directory; discovery starts afresh."""
    models = importlib.import_module(MODELS_PACKAGE)
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(models, "__path__", [*models.__path__, str(root)])
    discovered_shorthands.cache_clear()
    before = set(sys.modules)
    yield root
    for name in set(sys.modules) - before:
        if name.startswith(f"{MODELS_PACKAGE}."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    discovered_shorthands.cache_clear()


def _folder(root: Path, pkg: str, name: str, fields: str, init: str = "") -> str:
    """A model folder `pkg` whose `create` target raises if imported."""
    mk = root / pkg / "megakernel"
    mk.mkdir(parents=True)
    (root / pkg / "__init__.py").write_text(init, encoding="utf-8")
    spec = _SPEC.format(name=name, fields=fields)
    (mk / "__init__.py").write_text(textwrap.dedent(spec), encoding="utf-8")
    (mk / "adapter.py").write_text("raise RuntimeError('adapter imported')\n")
    importlib.invalidate_caches()
    return f"{MODELS_PACKAGE}.{pkg}.megakernel"


def test_the_built_in_folders_declare_what_the_table_lists() -> None:
    discovered_shorthands.cache_clear()
    found = discovered_shorthands()
    for name, entry in SHORTHANDS.items():
        if name not in found:  # the folder isn't installed
            continue
        assert found[name].module == entry.module
        assert found[name].architectures == entry.architectures
        assert found[name].model_types == entry.model_types
        assert found[name].legacy_keys == frozenset()
        # The table entry (with its top-level template keys) wins.
        assert shorthands()[name] is entry
    assert discovered_shorthands() is found  # once per process


def test_a_new_folder_is_found_by_architecture_and_model_type(
    models_dir: Path,
) -> None:
    module = _folder(
        models_dir,
        "zz_fake_mk",
        "fakemk",
        'architectures=("FakeForCausalLM",), model_types=("fake_mt",)',
    )
    (models_dir / "zz_plain").mkdir()
    (models_dir / "zz_plain" / "__init__.py").write_text(
        "raise RuntimeError('a package without a megakernel folder')\n"
    )
    by_arch = SimpleNamespace(architectures=["FakeForCausalLM"])
    by_type = SimpleNamespace(
        architectures=None, text_config=SimpleNamespace(model_type="fake_mt")
    )
    assert cfg_mod.shorthand_for_arch(by_arch) == "fakemk"
    assert cfg_mod.shorthand_for_arch(by_type) == "fakemk"
    ref, options = cfg_mod.resolve_plugin_ref(cfg_mod.parse_config("1"), by_arch)
    assert (ref.module, ref.attr, ref.shorthand) == (module, "SPEC", "fakemk")
    ref, _ = cfg_mod.resolve_plugin_ref(cfg_mod.parse_config({"plugin": "FakeMK"}))
    assert (ref.module, ref.shorthand) == (module, "fakemk")
    assert cfg_mod.missing_template(ref, by_arch) is None
    # Settings go through `options`: no top-level template keys.
    cfg = cfg_mod.parse_config({"plugin": "fakemk", "profile": "x"})
    with pytest.raises(MegakernelConfigError, match="unknown key 'profile'"):
        cfg_mod.resolve_plugin_ref(cfg)
    # Only the folders' stdlib-only `__init__` ran.
    assert module in sys.modules and f"{module}.adapter" not in sys.modules
    assert f"{MODELS_PACKAGE}.zz_plain" not in sys.modules


def test_a_broken_folder_is_skipped(
    models_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _folder(models_dir, "zz_broken_mk", "brokenmk", "", init="import no_such_pkg\n")
    _folder(models_dir, "zz_fake_mk", "fakemk", 'model_types=("fake_mt",)')
    with caplog.at_level(logging.DEBUG, logger=plugin_mod.__name__):
        found = discovered_shorthands()
    assert "fakemk" in found and "brokenmk" not in found
    assert "skipping vllm_torchtpu.models.zz_broken_mk.megakernel" in caplog.text


def test_the_explicit_table_wins(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folder(models_dir, "zz_fake_mk", "fakemk", 'architectures=("FakeForCausalLM",)')
    other = Shorthand("other.mk", ("FakeForCausalLM",), (), frozenset({"mode"}))
    monkeypatch.setitem(SHORTHANDS, "zz_claim", other)
    hf = SimpleNamespace(architectures=["FakeForCausalLM"])
    assert cfg_mod.shorthand_for_arch(hf) == "zz_claim"
    monkeypatch.setitem(SHORTHANDS, "fakemk", other)
    ref, _ = cfg_mod.resolve_plugin_ref(cfg_mod.parse_config({"plugin": "fakemk"}))
    assert ref.module == "other.mk"


def test_hf_fields_narrow_a_match_and_are_preferred(models_dir: Path) -> None:
    base = _folder(models_dir, "zz_base_mk", "basemk", 'architectures=("ArchA",)')
    big = _folder(
        models_dir,
        "zz_big_mk",
        "bigmk",
        'architectures=("ArchA",), '
        'hf_fields={"hidden_size": 8192, "num_hidden_layers": 92}',
    )
    entries = shorthands()
    assert entries["bigmk"].hf_fields == (
        ("hidden_size", 8192),
        ("num_hidden_layers", 92),
    )
    nested = SimpleNamespace(
        architectures=["ArchA"],
        text_config=SimpleNamespace(hidden_size=8192, num_hidden_layers=92),
    )
    flat = {"architectures": ["ArchA"], "hidden_size": 8192, "num_hidden_layers": 92}
    other = SimpleNamespace(architectures=["ArchA"], hidden_size=4096)
    assert entries[cfg_mod.shorthand_for_arch(nested)].module == big
    assert entries[cfg_mod.shorthand_for_arch(flat)].module == big
    assert entries[cfg_mod.shorthand_for_arch(other)].module == base


def test_spec_normalizes_the_checkpoint_fields() -> None:
    spec = TpuMonoSpec(
        "t",
        "1",
        create="m:c",
        architectures=["A"],  # type: ignore[arg-type]
        model_types=("m",),
        hf_fields={"hidden_size": 8},  # type: ignore[arg-type]
    )
    assert spec.architectures == ("A",) and spec.model_types == ("m",)
    assert spec.hf_fields == (("hidden_size", 8),)
    assert TpuMonoSpec("t", "1", create="m:c").architectures == ()


_BLOCKED = """
import sys
for name in ("torch", "jax", "jaxlib", "vllm", "numpy"):
    sys.modules[name] = None
from vllm_torchtpu.megakernel_runtime import plugin
found = plugin.discovered_shorthands()
assert not any(m.endswith(".adapter") for m in sys.modules), sorted(sys.modules)
print(",".join(sorted(found)))
"""


def test_discovery_imports_no_framework() -> None:
    """Discovery runs with torch, jax, numpy and vLLM blocked."""
    import subprocess

    src = Path(plugin_mod.__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-c", _BLOCKED],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    discovered_shorthands.cache_clear()
    assert out.stdout.strip().split(",") == sorted(discovered_shorthands()) or (
        out.stdout.strip() == "" and not discovered_shorthands()
    )
