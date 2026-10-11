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
"""--target: built-in names, target files, and SCENARIO_TARGET."""

import pathlib
import re
import textwrap

import pytest
from scenario.targets.kubectl import KubectlTarget
from scenario.targets.loader import (
    TargetLoadError,
    load_module,
    load_target,
    target_spec,
)
from scenario.targets.local import LocalTarget

HEADER = "from scenario.targets.kubernetes import KubernetesTarget\n"


def write(directory: pathlib.Path, name: str, body: str) -> pathlib.Path:
    path = directory / name
    path.write_text(HEADER + textwrap.dedent(body))
    return path


def test_target_spec() -> None:
    assert target_spec("kubectl", {"SCENARIO_TARGET": "x.py"}) == "kubectl"
    assert target_spec(None, {"SCENARIO_TARGET": "x.py"}) == "x.py"
    assert target_spec(None, {"SCENARIO_TARGET": ""}) is None
    assert target_spec(None, {}) is None


def test_builtins(tmp_path: pathlib.Path) -> None:
    assert isinstance(load_target("kubectl", tmp_path), KubectlTarget)
    local = load_target("local", tmp_path)
    assert isinstance(local, LocalTarget)
    assert local.options.repo_root == tmp_path


def test_unknown_name(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TargetLoadError, match="name local or kubectl, or a .py file"):
        load_target("fleet", tmp_path)


def test_a_file_with_one_target_class(tmp_path: pathlib.Path) -> None:
    write(
        tmp_path,
        "fleet.py",
        """
        class Helper:
            pass

        class Fleet(KubernetesTarget):
            def container_name(self):
                return "workload"
        """,
    )
    target = load_target("fleet.py", tmp_path)
    assert type(target).__name__ == "Fleet"
    assert target.container_name() == "workload"
    (tmp_path / "sub").mkdir()
    again = load_target("../fleet.py", tmp_path / "sub")
    assert type(again) is type(target)


def test_make_target_wins(tmp_path: pathlib.Path) -> None:
    write(
        tmp_path,
        "two.py",
        """
        class A(KubernetesTarget):
            pass

        class B(KubernetesTarget):
            pass

        def make_target():
            return B()
        """,
    )
    assert type(load_target("./two.py", tmp_path)).__name__ == "B"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "class A(KubernetesTarget): pass\nclass B(KubernetesTarget): pass\n",
            "2 target",
        ),
        ("X = 1\n", "defines 0 target classes (none)"),
        ("def make_target():\n    return 3\n", "3 is not a target"),
        ("def make_target():\n    raise OSError('no')\n", "building the target failed"),
        ("import no_such_module\n", "importing"),
        (
            "class A(KubernetesTarget):\n    def render(self, plan): return ''\n",
            "overrides render()",
        ),
    ],
)
def test_bad_files(tmp_path: pathlib.Path, body: str, message: str) -> None:
    path = tmp_path / f"bad_{abs(hash(body))}.py"
    path.write_text(HEADER + body)
    with pytest.raises(TargetLoadError, match=re.escape(message)):
        load_target(str(path), tmp_path)


def test_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TargetLoadError, match="no target file"):
        load_target("nothere.py", tmp_path)


def test_a_file_imports_once(tmp_path: pathlib.Path) -> None:
    path = write(tmp_path, "once.py", "class T(KubernetesTarget):\n    pass\n")
    assert load_module(path) is load_module(path)
