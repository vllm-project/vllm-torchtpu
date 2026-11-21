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

import subprocess
import sys

import pytest

import vllm_torchtpu.models as models
from vllm_torchtpu.models.vllm import register_model_directories


def _model_dir(tmp_path, monkeypatch, name: str, init: str) -> None:
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(init)
    monkeypatch.setattr(models, "__path__", [*models.__path__, str(tmp_path)])
    monkeypatch.delitem(sys.modules, f"{models.__name__}.{name}", raising=False)


def test_model_directories_register_themselves(tmp_path, monkeypatch) -> None:
    _model_dir(
        tmp_path,
        monkeypatch,
        "_test_model_dir",
        "CALLS = []\ndef register():\n    CALLS.append('registered')\n",
    )
    (tmp_path / "_test_readme_only").mkdir()  # no __init__: not a package

    register_model_directories()

    registered = sys.modules[f"{models.__name__}._test_model_dir"]
    assert registered.CALLS == ["registered"]


def test_model_directory_without_register_is_an_error(tmp_path, monkeypatch) -> None:
    _model_dir(tmp_path, monkeypatch, "_test_no_register", "X = 1\n")

    with pytest.raises(TypeError, match="_test_no_register"):
        register_model_directories()


def _model_directory_packages() -> list[str]:
    import pkgutil

    return [
        m.name
        for m in pkgutil.iter_modules(models.__path__)
        if m.ispkg and m.name != "vllm"
    ]


@pytest.mark.parametrize("name", _model_directory_packages())
def test_model_directory_init_is_import_free(name: str) -> None:
    # register() runs at plugin load in every process; importing a model
    # directory must not pull in the accelerator stack.
    code = (
        "import sys\n"
        f"import {models.__name__}.{name}\n"
        "heavy = [m for m in ('jax', 'torch_tpu') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
