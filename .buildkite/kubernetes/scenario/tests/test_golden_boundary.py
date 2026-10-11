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
"""Nothing outside golden/ imports it.

golden/ is temporary: M4 deletes it with today's kube scripts, so no other
code or test may depend on it. Its conftest puts golden/ on sys.path, so its
modules are imported by their bare names.
"""

import ast
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden"
REPO = HERE.parents[3]
SEARCHED = (REPO / ".buildkite", REPO / "tools" / "scenario")


def golden_modules() -> set[str]:
    """The names golden/ can be imported by; conftest is pytest's own."""
    return {path.stem for path in GOLDEN.glob("*.py")} - {"conftest"}


def golden_imports(source: str, modules: set[str]) -> list[str]:
    """`line N: <module>` for every import of one of `modules`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module]
        else:
            continue
        found += [
            f"line {node.lineno}: {name}"
            for name in names
            if name.split(".")[0] in modules
        ]
    return found


def test_nothing_outside_golden_imports_it() -> None:
    modules = golden_modules()
    assert "golden_capture" in modules
    found = {
        str(path.relative_to(REPO)): hits
        for root in SEARCHED
        for path in sorted(root.rglob("*.py"))
        if not path.is_relative_to(GOLDEN)
        and (hits := golden_imports(path.read_text(), modules))
    }
    assert found == {}


@pytest.mark.parametrize(
    "source",
    [
        "import golden_capture",
        "from golden_cases import all_cases",
        "import stubs as recorder",
    ],
)
def test_seeded_imports_are_caught(source: str) -> None:
    modules = {"golden_capture", "golden_cases", "stubs"}
    assert golden_imports(f"import json\n{source}\n", modules) == [
        f"line 2: {source.split()[1]}"
    ]


def test_other_imports_pass() -> None:
    source = "import json\nfrom scenario.plan import codec\nimport golden_captured\n"
    assert golden_imports(source, {"golden_capture"}) == []
