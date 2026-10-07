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
"""The committed schemas and reference match the code (R-SCH-1)."""

import json
import pathlib
import shutil
from typing import Any

import pytest
from click.testing import CliRunner
from scenario import gen
from scenario.cli import main
from scenario.model.rules import rule_table
from scenario.model.scenario import Defaults, Scenario
from scenario.plan.plumbing import ITEMS
from scenario.reserved import AGENTX_CLIENT_ENV, TOOL_ENV


def test_generated_files_are_current() -> None:
    stale = gen.stale(gen.tool_dir())
    assert stale == [], f"stale: {', '.join(stale)}; run: {gen.REGEN}"


@pytest.fixture
def stale_copy(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "scenario"
    for name in gen.GENERATORS:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(gen.tool_dir() / name, root / name)
    (root / gen.REFERENCE).write_text("# Old reference\n")
    (root / gen.DEFAULTS_SCHEMA).unlink()
    return root


def test_stale_files_are_found_and_rewritten(stale_copy: pathlib.Path) -> None:
    assert gen.stale(stale_copy) == [gen.DEFAULTS_SCHEMA, gen.REFERENCE]
    assert gen.write(stale_copy) == [gen.DEFAULTS_SCHEMA, gen.REFERENCE]
    assert gen.stale(stale_copy) == []
    assert gen.write(stale_copy) == []


def test_dev_regen_check(
    stale_copy: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = CliRunner().invoke(main, ["dev", "regen", "--check"])
    assert current.exit_code == 0, current.output
    monkeypatch.setattr(gen, "tool_dir", lambda: stale_copy)
    stale = CliRunner().invoke(main, ["dev", "regen", "--check"])
    assert stale.exit_code == 1
    assert stale.stderr == (
        "stale: tools/scenario/schema/defaults.schema.json; run: uv run --project "
        "tools/scenario scenario dev regen\n"
        "stale: tools/scenario/REFERENCE.md; run: uv run --project tools/scenario "
        "scenario dev regen\n"
    )
    assert not (stale_copy / gen.DEFAULTS_SCHEMA).exists()
    wrote = CliRunner().invoke(main, ["dev", "regen"])
    assert wrote.exit_code == 0
    assert wrote.stdout == (
        "wrote tools/scenario/schema/defaults.schema.json\n"
        "wrote tools/scenario/REFERENCE.md\n"
    )


def _walk(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, list):
        return [found for item in node for found in _walk(item)]
    if not isinstance(node, dict):
        return []
    return [node, *(found for value in node.values() for found in _walk(value))]


@pytest.mark.parametrize("name", [gen.SCENARIO_SCHEMA, gen.DEFAULTS_SCHEMA])
def test_layer_schemas(name: str) -> None:
    schema = json.loads((gen.tool_dir() / name).read_text())
    assert schema["$schema"] == gen.SCHEMA_DIALECT
    for node in _walk(schema):
        assert "required" not in node
        for prop in node.get("properties", {}).values():
            assert {"type": "null"} in prop.get("anyOf", []), prop
        if node.get("type") == "object" and "properties" in node:
            assert "additionalProperties" in node, node.get("title")


def test_reference_lists_everything() -> None:
    text = (gen.tool_dir() / gen.REFERENCE).read_text()
    for root in (Scenario, Defaults):
        for model in gen.models_of(root):
            assert f"### {model.__name__}\n" in text
            for name, field in model.model_fields.items():
                assert f"| `{field.alias or name}` |" in text, (model, name)
    assert all(f"| {rule.id} |" in text for rule in rule_table())
    assert all(f"| {item} |" in text for item in ITEMS)
    for name in [*TOOL_ENV, *AGENTX_CLIENT_ENV]:
        assert f"| `{name}` |" in text


def test_reference_is_stable_markdown() -> None:
    text = (gen.tool_dir() / gen.REFERENCE).read_text()
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert "\n\n\n" not in text
    assert all(line == line.rstrip() for line in text.splitlines())
    assert text.count("\n# ") == 0 and text.startswith("# ")
