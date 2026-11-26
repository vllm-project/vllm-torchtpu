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
"""Every package under vllm_torchtpu.models registers itself."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import vllm_torchtpu.layers as layers
import vllm_torchtpu.models as models
from vllm_torchtpu.models import iter_model_directories, register_model_directories

# Blocked before `import vllm_torchtpu`, which itself tries JAX and tolerates
# its absence; a model directory that imports any of them fails the check.
_FRAMEWORKS = ("torch", "torch_tpu", "jax", "jaxlib", "vllm", "numpy")

_CHECK = """
import importlib
import sys
for name in {frameworks!r}:
    sys.modules[name] = None
import vllm_torchtpu.models as models
if {path!r}:
    models.__path__ = [{path!r}]
package = importlib.import_module(f"{{models.__name__}}.{name}")
assert callable(package.__dict__.get("register")), "no callable register()"
"""


def _check_import_free(name: str, path: str = "") -> subprocess.CompletedProcess:
    src = Path(models.__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _CHECK.format(frameworks=_FRAMEWORKS, path=path, name=name),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
        timeout=120,
    )


@pytest.fixture
def model_dirs(tmp_path, monkeypatch):
    """`models.__path__` points at `tmp_path` only; `add` creates a package."""
    monkeypatch.setattr(models, "__path__", [str(tmp_path)])
    created: list[str] = []

    def add(name: str, init: str) -> None:
        pkg = tmp_path / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text(init)
        created.append(name)

    yield add
    for name in created:
        sys.modules.pop(f"{models.__name__}.{name}", None)
        # import_module also binds the package as an attribute of the parent.
        models.__dict__.pop(name, None)


def test_iter_model_directories_skips_legacy_and_non_packages(
    tmp_path, model_dirs
) -> None:
    model_dirs("_test_b", "")
    model_dirs("_test_a", "")
    (tmp_path / "vllm").mkdir()  # the legacy package is not a model directory
    (tmp_path / "vllm" / "__init__.py").write_text("")
    (tmp_path / "_test_readme_only").mkdir()  # no __init__: not a package

    assert list(iter_model_directories()) == ["_test_a", "_test_b"]


def test_model_directories_register_themselves(model_dirs) -> None:
    model_dirs(
        "_test_dir", "CALLS = []\ndef register():\n    CALLS.append('registered')\n"
    )

    register_model_directories()

    assert sys.modules[f"{models.__name__}._test_dir"].CALLS == ["registered"]


def test_model_directories_run_in_name_order_and_later_wins(model_dirs) -> None:
    # A shared fake registry: each directory registers the same architecture.
    model_dirs("_test_log", "CALLS = []\nREGISTRY = {}\ndef register():\n    pass\n")
    body = (
        "def register():\n"
        "    from vllm_torchtpu.models import _test_log as log\n"
        "    log.CALLS.append(__name__.rsplit('.', 1)[1])\n"
        "    log.REGISTRY['Arch'] = __name__.rsplit('.', 1)[1]\n"
    )
    model_dirs("_test_z", body)  # created first, sorts last
    model_dirs("_test_m", body)

    register_model_directories()

    log = sys.modules[f"{models.__name__}._test_log"]
    assert log.CALLS == ["_test_m", "_test_z"]
    assert log.REGISTRY == {"Arch": "_test_z"}


def test_register_layers_runs_model_directories_after_register_models(
    model_dirs, monkeypatch
) -> None:
    # register_layers() is the plugin entry; directories must run after the
    # central register_models() so a directory's registration wins.
    calls: list[str] = []
    monkeypatch.setattr(layers, "register_models", lambda: calls.append("central"))
    monkeypatch.setattr(layers, "register_tokenizers", lambda: None)
    model_dirs(
        "_test_entry",
        "import sys\n"
        "def register():\n"
        "    sys.modules['vllm_torchtpu.layers']._order.append('directory')\n",
    )
    monkeypatch.setattr(layers, "_order", calls, raising=False)

    layers.register_layers()

    assert calls == ["central", "directory"]


def test_model_directory_without_register_is_an_error(model_dirs) -> None:
    model_dirs("_test_no_register", "X = 1\n")

    with pytest.raises(TypeError, match="_test_no_register"):
        register_model_directories()


def test_import_free_check_passes_for_stdlib_directory(tmp_path) -> None:
    (tmp_path / "_test_ok").mkdir()
    (tmp_path / "_test_ok" / "__init__.py").write_text(
        "import os\ndef register():\n    pass\n"
    )

    res = _check_import_free("_test_ok", str(tmp_path))
    assert res.returncode == 0, res.stderr[-3000:]


def test_import_free_check_fails_for_directory_importing_jax(tmp_path) -> None:
    (tmp_path / "_test_jax").mkdir()
    (tmp_path / "_test_jax" / "__init__.py").write_text(
        "import jax\ndef register():\n    pass\n"
    )

    res = _check_import_free("_test_jax", str(tmp_path))
    assert res.returncode != 0 and "jax" in res.stderr, res.stderr[-3000:]


def test_import_free_check_fails_without_register(tmp_path) -> None:
    (tmp_path / "_test_bare").mkdir()
    (tmp_path / "_test_bare" / "__init__.py").write_text("X = 1\n")

    res = _check_import_free("_test_bare", str(tmp_path))
    assert res.returncode != 0 and "register()" in res.stderr, res.stderr[-3000:]


@pytest.mark.parametrize("name", list(iter_model_directories()))
def test_model_directory_defines_register_import_free(name: str) -> None:
    # Every real model directory: a new folder under models/ without a
    # callable register(), or whose __init__ imports a framework, fails here
    # instead of at plugin load.
    res = _check_import_free(name)
    assert res.returncode == 0, res.stderr[-3000:]
