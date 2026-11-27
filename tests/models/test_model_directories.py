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
"""Model directories listed in MODEL_DIRECTORIES register themselves."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import vllm_torchtpu.layers as layers
import vllm_torchtpu.models as models
from vllm_torchtpu.models import MODEL_DIRECTORIES, register_model_directories

# Blocked before `import vllm_torchtpu`, which itself tries JAX and tolerates
# its absence. They stay blocked while register() runs: a model directory
# whose __init__ or register() imports any of them fails the check.
_FRAMEWORKS = ("torch", "torch_tpu", "jax", "jaxlib", "vllm", "numpy")

# register() may import vLLM's registries (and nothing else from vLLM). The real
# ones import vLLM's platform and so torch_tpu, so the check runs register()
# against stand-ins that record what it registers.
_CHECK = """
import importlib
import sys
import types
for name in {frameworks!r}:
    sys.modules[name] = None
import vllm_torchtpu.models as models
if {path!r}:
    models.__path__ = [{path!r}]
package = importlib.import_module(f"{{models.__name__}}.{name}")
register = package.__dict__.get("register")
assert callable(register), "no callable register()"

registered = {{}}
class ModelRegistry:
    @staticmethod
    def register_model(arch, path):
        assert isinstance(path, str), "register lazy 'module:Class' paths"
        registered[arch] = path
class VerifyAndUpdateConfig:
    pass
stubs = {{
    "vllm": {{}},
    "vllm.model_executor": {{}},
    "vllm.model_executor.models": {{"ModelRegistry": ModelRegistry}},
    "vllm.model_executor.models.config": {{
        "MODELS_CONFIG_MAP": {{}},
        "VerifyAndUpdateConfig": VerifyAndUpdateConfig,
    }},
}}
for mod_name, attrs in stubs.items():
    mod = types.ModuleType(mod_name)
    mod.__path__ = []
    mod.__dict__.update(attrs)
    sys.modules[mod_name] = mod
register()
print(sorted(registered))
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
    """`models.__path__` points at `tmp_path` only and `MODEL_DIRECTORIES` is
    empty; `add` creates a package and, if `listed`, appends it to the list."""
    monkeypatch.setattr(models, "__path__", [str(tmp_path)])
    monkeypatch.setattr(models, "MODEL_DIRECTORIES", ())
    created: list[str] = []

    def add(name: str, init: str, listed: bool = True) -> None:
        pkg = tmp_path / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text(init)
        created.append(name)
        if listed:
            monkeypatch.setattr(
                models, "MODEL_DIRECTORIES", (*models.MODEL_DIRECTORIES, name)
            )

    yield add
    for name in created:
        sys.modules.pop(f"{models.__name__}.{name}", None)
        # import_module also binds the package as an attribute of the parent.
        models.__dict__.pop(name, None)


def test_listed_model_directory_registers_itself(model_dirs) -> None:
    model_dirs(
        "_test_dir", "CALLS = []\ndef register():\n    CALLS.append('registered')\n"
    )

    register_model_directories()

    assert sys.modules[f"{models.__name__}._test_dir"].CALLS == ["registered"]


def test_unlisted_package_is_not_imported(model_dirs) -> None:
    model_dirs("_test_unlisted", "raise RuntimeError('imported')\n", listed=False)

    register_model_directories()

    assert f"{models.__name__}._test_unlisted" not in sys.modules


def test_model_directories_run_in_list_order_and_later_wins(model_dirs) -> None:
    # A shared fake registry: each directory registers the same architecture.
    model_dirs("_test_log", "CALLS = []\nREGISTRY = {}\ndef register():\n    pass\n")
    body = (
        "def register():\n"
        "    from vllm_torchtpu.models import _test_log as log\n"
        "    log.CALLS.append(__name__.rsplit('.', 1)[1])\n"
        "    log.REGISTRY['Arch'] = __name__.rsplit('.', 1)[1]\n"
    )
    model_dirs("_test_z", body)  # listed first although it sorts last
    model_dirs("_test_m", body)

    register_model_directories()

    log = sys.modules[f"{models.__name__}._test_log"]
    assert log.CALLS == ["_test_z", "_test_m"]
    assert log.REGISTRY == {"Arch": "_test_m"}


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


def test_listed_directory_without_register_is_an_error(model_dirs) -> None:
    model_dirs("_test_no_register", "X = 1\n")

    with pytest.raises(TypeError, match="_test_no_register"):
        register_model_directories()


def test_import_free_check_passes_for_stdlib_directory(tmp_path) -> None:
    (tmp_path / "_test_ok").mkdir()
    (tmp_path / "_test_ok" / "__init__.py").write_text(
        "import os\ndef register():\n    import json\n"
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


@pytest.mark.parametrize("framework", ("jax", "torch_tpu", "torch"))
def test_import_free_check_fails_when_register_imports_framework(
    tmp_path, framework
) -> None:
    (tmp_path / "_test_lazy").mkdir()
    (tmp_path / "_test_lazy" / "__init__.py").write_text(
        f"def register():\n    import {framework}\n"
    )

    res = _check_import_free("_test_lazy", str(tmp_path))
    assert res.returncode != 0 and framework in res.stderr, res.stderr[-3000:]


def test_import_free_check_runs_register_against_vllm_registries(tmp_path) -> None:
    (tmp_path / "_test_reg").mkdir()
    (tmp_path / "_test_reg" / "__init__.py").write_text(
        "def register():\n"
        "    from vllm.model_executor.models import ModelRegistry\n"
        "    from vllm.model_executor.models.config import MODELS_CONFIG_MAP\n"
        "    ModelRegistry.register_model('XForCausalLM', 'pkg.model:X')\n"
        "    MODELS_CONFIG_MAP['XForCausalLM'] = object\n"
    )

    res = _check_import_free("_test_reg", str(tmp_path))
    assert res.returncode == 0, res.stderr[-3000:]
    assert "XForCausalLM" in res.stdout


def test_import_free_check_fails_without_register(tmp_path) -> None:
    (tmp_path / "_test_bare").mkdir()
    (tmp_path / "_test_bare" / "__init__.py").write_text("X = 1\n")

    res = _check_import_free("_test_bare", str(tmp_path))
    assert res.returncode != 0 and "register()" in res.stderr, res.stderr[-3000:]


@pytest.mark.parametrize("name", MODEL_DIRECTORIES)
def test_model_directory_defines_register_import_free(name: str) -> None:
    # Every listed model directory: its __init__ and register() import no
    # framework, and register() runs against vLLM's registries.
    res = _check_import_free(name)
    assert res.returncode == 0, res.stderr[-3000:]
