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
"""Import boundaries between the tool's modules (R-TEST-6).

Scenario code never imports a target, targets never import the scenario
model, and pods never import target code. Every module of the package
needs a row here, so a new one cannot slip past the rule.
"""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "scenario"

# Every module may import the package itself (version constants) and the
# reserved env names.
BASE = ("scenario.reserved",)
LEAVES = ("scenario.paths", "scenario.errors")
SCENARIO_CODE = (
    *BASE,
    *LEAVES,
    "scenario.model",
    "scenario.load",
    "scenario.keycheck",
    "scenario.merge",
    "scenario.overrides",
    "scenario.resolve",
    "scenario.plan",
)
ANYTHING = ("*",)
KUBERNETES = (*BASE, "scenario.target", "scenario.plan.types", "scenario.plan.codec")

# (module or package, the scenario modules it may import); first match wins.
RULES: list[tuple[str, tuple[str, ...]]] = [
    ("scenario.reserved", ()),
    ("scenario.paths", ()),
    ("scenario.errors", ("scenario.paths",)),
    ("scenario.model", SCENARIO_CODE),
    ("scenario.load", SCENARIO_CODE),
    ("scenario.keycheck", SCENARIO_CODE),
    ("scenario.merge", SCENARIO_CODE),
    ("scenario.overrides", SCENARIO_CODE),
    ("scenario.resolve", SCENARIO_CODE),
    ("scenario.plan", SCENARIO_CODE),
    ("scenario.target", (*BASE, "scenario.plan.types")),
    ("scenario.targets.kubernetes", KUBERNETES),
    ("scenario.targets.kubectl", (*KUBERNETES, "scenario.targets.kubernetes")),
    (
        "scenario.targets.local",
        (*BASE, "scenario.target", "scenario.plan", "scenario.runtime"),
    ),
    ("scenario.targets.loader", (*BASE, "scenario.target", "scenario.targets")),
    (
        "scenario.runtime",
        (
            *BASE,
            "scenario.target",
            "scenario.plan.types",
            "scenario.plan.codec",
            "scenario.runtime",
        ),
    ),
    ("scenario.show", SCENARIO_CODE),
    ("scenario.gen", (*SCENARIO_CODE, "scenario.show")),
    ("scenario.cli", ANYTHING),
    ("scenario.__main__", ("scenario.cli",)),
]

# Imported only by runtime/warm.py and by `python3 -c` subprocesses.
HEAVY = {"vllm", "ray", "torch", "transformers", "jax"}
WARM = "scenario.runtime.warm"


def _within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def rule_for(module: str) -> tuple[str, ...] | None:
    if module == "scenario":
        return ()
    return next((allowed for prefix, allowed in RULES if _within(module, prefix)), None)


def module_name(path: pathlib.Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_module(name: str) -> bool:
    relative = pathlib.Path(*name.split("."))
    root = PACKAGE.parent
    return (root / relative).with_suffix(".py").is_file() or (
        root / relative / "__init__.py"
    ).is_file()


def imports(source: str) -> list[tuple[int, str]]:
    """(line, module) for every import; `from a import b` names a.b when b is one."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                found.append((node.lineno, full if _is_module(full) else node.module))
    return found


def violations(module: str, source: str) -> list[str]:
    allowed = rule_for(module)
    assert allowed is not None, f"no import rule for {module}; add a row to RULES"
    problems = []
    for line, name in imports(source):
        if name.split(".")[0] in HEAVY and module != WARM:
            problems.append(f"line {line}: imports {name}")
        if not _within(name, "scenario") or name in ("scenario", module):
            continue
        if allowed != ANYTHING and not any(_within(name, p) for p in allowed):
            problems.append(f"line {line}: {module} imports {name}")
    return problems


def package_modules() -> list[pathlib.Path]:
    return sorted(PACKAGE.rglob("*.py"))


@pytest.mark.parametrize(
    "path", package_modules(), ids=lambda path: str(path.relative_to(PACKAGE))
)
def test_module_imports(path: pathlib.Path) -> None:
    assert violations(module_name(path), path.read_text()) == []


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("scenario.plan.build", "from scenario.target import Target"),
        ("scenario.resolve", "import scenario.targets.kubectl"),
        ("scenario.model.server", "from scenario.show import render_plan"),
        ("scenario.target", "from scenario.model.scenario import Scenario"),
        ("scenario.target", "from scenario.plan.build import plan"),
        ("scenario.target", "from scenario import resolve"),
        ("scenario.targets.kubernetes", "from scenario.resolve import Resolved"),
        ("scenario.runtime.driver", "from scenario.targets import local"),
        ("scenario.runtime.driver", "import vllm"),
        ("scenario.reserved", "from scenario.paths import KeyPath"),
        ("scenario.show", "from scenario.cli import main"),
    ],
)
def test_seeded_violations_are_caught(module: str, source: str) -> None:
    assert violations(module, source)


def test_allowed_imports_pass() -> None:
    assert violations("scenario.target", "from scenario.plan.types import Plan") == []
    assert violations("scenario.plan.build", "from scenario import __version__") == []
    assert violations("scenario.cli", "from scenario.target import Target") == []
    assert violations("scenario.runtime.warm", "import vllm") == []


def test_every_module_has_a_rule() -> None:
    missing = [m for m in map(module_name, package_modules()) if rule_for(m) is None]
    assert missing == []
