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
"""Tests for the schema validator and task config parser of eval_json_schema.py.

The requests the client sends, the results file it writes and its gate are
tested through ``run_eval_flow.sh`` in ``test_run_eval_flow_cli.py``.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "benchmarking"
    / "eval_json_schema.py"
)
_spec = importlib.util.spec_from_file_location("eval_json_schema", _SCRIPT)
assert _spec is not None and _spec.loader is not None
eval_json_schema = importlib.util.module_from_spec(_spec)
# dataclasses looks its module up in sys.modules while the class is built.
sys.modules[_spec.name] = eval_json_schema
_spec.loader.exec_module(eval_json_schema)

# Nests without a depth bound, like structured_schemas/complex_ast.json.
_TREE_SCHEMA = {
    "$defs": {
        "Node": {"anyOf": [{"type": "number"}, {"$ref": "#/$defs/Tree"}]},
        "Tree": {
            "type": "object",
            "properties": {"left": {"$ref": "#/$defs/Node"}},
            "required": ["left"],
        },
    },
    "$ref": "#/$defs/Tree",
}


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        pytest.param({"type": "integer"}, 7, id="integer"),
        pytest.param({"type": "number"}, 7.5, id="number"),
        pytest.param({"type": "boolean"}, True, id="boolean"),
        pytest.param({"type": "null"}, None, id="null"),
        pytest.param({"type": "string", "pattern": "^[a-z-]+$"}, "a-b", id="pattern"),
        pytest.param({"enum": ["a", "b"]}, "b", id="enum"),
        pytest.param({"minimum": 1, "maximum": 3}, 3, id="range"),
        pytest.param(
            {"type": "array", "minItems": 2, "items": {"type": "string"}},
            ["a", "b"],
            id="array",
        ),
        pytest.param(
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"a": "x", "extra": 1},
            id="additional_property_allowed_by_default",
        ),
        pytest.param(_TREE_SCHEMA, {"left": {"left": 1}}, id="recursive_ref"),
    ],
)
def test_validator_accepts_a_value_that_satisfies_the_schema(schema, value):
    validator = eval_json_schema.SchemaValidator(schema)

    assert validator.check(json.dumps(value), "stop") is None


@pytest.mark.parametrize(
    ("schema", "value", "error"),
    [
        pytest.param(
            {"type": "integer"},
            True,
            "Expected JSON type integer, got bool",
            id="bool_is_not_an_integer",
        ),
        pytest.param(
            {"type": "integer"},
            7.5,
            "Expected JSON type integer, got float",
            id="float_is_not_an_integer",
        ),
        pytest.param(
            {"type": "string", "pattern": "^[a-z-]+$"},
            "A b",
            "String 'A b' failed pattern '^[a-z-]+$'",
            id="pattern",
        ),
        pytest.param(
            {"enum": ["a", "b"]}, "c", "Value 'c' not in enum ['a', 'b']", id="enum"
        ),
        pytest.param({"minimum": 1}, 0, "0 < minimum 1", id="minimum"),
        pytest.param({"maximum": 3}, 4, "4 > maximum 3", id="maximum"),
        pytest.param(
            {"type": "array", "minItems": 2},
            ["a"],
            "Array length 1 < minItems 2",
            id="min_items",
        ),
        pytest.param(
            {"items": {"type": "string"}},
            ["a", 1],
            "Item [1]: Expected JSON type string, got int",
            id="items",
        ),
        pytest.param(
            {"required": ["a", "b"]},
            {"a": 1},
            "Missing required property 'b'",
            id="required",
        ),
        pytest.param(
            {"properties": {"a": {"type": "string"}}},
            {"a": 1},
            "Property 'a': Expected JSON type string, got int",
            id="properties",
        ),
        pytest.param(
            {"properties": {"a": {}}, "additionalProperties": False},
            {"a": 1, "extra": 2},
            "Disallowed additional property 'extra'",
            id="additional_properties",
        ),
        pytest.param(
            _TREE_SCHEMA,
            {"left": {"left": "x"}},
            "Property 'left': Instance {'left': 'x'} matched none of anyOf",
            id="any_of_through_a_recursive_ref",
        ),
    ],
)
def test_validator_reports_the_first_violation(schema, value, error):
    validator = eval_json_schema.SchemaValidator(schema)

    assert validator.check(json.dumps(value), "stop") == error


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        pytest.param(
            {"type": "string", "format": "email", "maxLength": 9},
            "Schema keywords ['format', 'maxLength'] at # are not checked by "
            "eval_json_schema.py",
            id="unchecked_keywords",
        ),
        pytest.param(
            {"properties": {"tags": {"type": "array", "uniqueItems": True}}},
            "Schema keywords ['uniqueItems'] at #/properties/tags are not checked",
            id="unchecked_keyword_in_a_property",
        ),
        pytest.param(
            {"anyOf": [{"type": "string"}, {"const": 1}]},
            "Schema keywords ['const'] at #/anyOf/1 are not checked",
            id="unchecked_keyword_in_any_of",
        ),
        pytest.param(
            {"$defs": {"A": {"multipleOf": 5}}},
            "Schema keywords ['multipleOf'] at #/$defs/A are not checked",
            id="unchecked_keyword_in_defs",
        ),
        pytest.param(
            {"items": [{"type": "integer"}]},
            "Schema at #/items is not an object",
            id="items_as_a_list",
        ),
        pytest.param(
            {"$ref": "#/$defs/Missing"},
            "$ref '#/$defs/Missing' at # does not name a '#/$defs/<name>' of this "
            "schema",
            id="unresolved_ref",
        ),
        pytest.param(
            {"$ref": "other.json#/$defs/A", "$defs": {"A": {}}},
            "$ref 'other.json#/$defs/A' at # does not name a '#/$defs/<name>' of "
            "this schema",
            id="non_local_ref",
        ),
        pytest.param(
            {"$ref": "#/$defs/A", "type": "object", "$defs": {"A": {}}},
            "Keywords beside $ref at # are not checked",
            id="keywords_beside_a_ref",
        ),
        pytest.param(
            {"type": ["string", "null"]},
            "type ['string', 'null'] at # is not one JSON type name",
            id="type_as_a_list",
        ),
        pytest.param(
            {"enum": [1, "a"]},
            "enum at # has members that are not strings",
            id="non_string_enum_member",
        ),
        pytest.param(
            {"additionalProperties": {"type": "string"}},
            "additionalProperties at # is not true or false",
            id="additional_properties_as_a_schema",
        ),
    ],
)
def test_validator_refuses_a_schema_it_cannot_check(schema, message):
    with pytest.raises(ValueError) as excinfo:
        eval_json_schema.SchemaValidator(schema)

    assert message in str(excinfo.value)


def _write_task_config(tmp_path: pathlib.Path, config: Any) -> pathlib.Path:
    path = tmp_path / "task.yaml"
    path.write_text(json.dumps(config))
    return path


def test_load_cases_reads_inline_and_file_schemas_prompts_and_datasets(tmp_path):
    schema = {"type": "object", "required": ["name"]}
    (tmp_path / "schema.json").write_text(
        json.dumps({"structured_outputs": {"json": schema}})
    )
    records = [
        {
            "id": f"record-{i}",
            "conversations": [
                {"from": "gpt", "value": "ignored"},
                {"from": "human", "value": f"prompt {i}"},
            ],
        }
        for i in range(3)
    ]
    (tmp_path / "dataset.json").write_text(json.dumps(records))
    path = _write_task_config(
        tmp_path,
        {
            "cases": [
                {
                    "name": "from_files",
                    "max_tokens": 512,
                    "schema_path": "schema.json",
                    "dataset_path": "dataset.json",
                    "limit": 2,
                },
                {
                    "name": "inline",
                    "max_tokens": 64,
                    "schema": {"type": "string"},
                    "prompts": ["first", "second"],
                },
            ]
        },
    )

    from_files, inline = eval_json_schema.load_cases(path)

    sample = eval_json_schema.Sample
    assert (from_files.name, from_files.max_tokens) == ("from_files", 512)
    assert from_files.validator.schema == schema
    assert from_files.samples == (
        sample("record-0", "prompt 0"),
        sample("record-1", "prompt 1"),
    )
    assert (inline.name, inline.max_tokens) == ("inline", 64)
    assert inline.validator.schema == {"type": "string"}
    assert inline.samples == (sample("0", "first"), sample("1", "second"))


_CASE = {
    "name": "case",
    "max_tokens": 64,
    "schema": {"type": "string"},
    "prompts": ["prompt"],
}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        pytest.param(
            [_CASE], "must be a mapping with one key, 'cases'", id="root_is_a_list"
        ),
        pytest.param(
            {"cases": [_CASE], "concurrency": 4},
            "must be a mapping with one key, 'cases'",
            id="second_top_level_key",
        ),
        pytest.param(
            {"cases": []}, "must be a mapping with one key, 'cases'", id="no_cases"
        ),
        pytest.param(
            {"cases": [{k: v for k, v in _CASE.items() if k != "max_tokens"}]},
            "Case 'case' needs 'max_tokens'",
            id="no_max_tokens",
        ),
        pytest.param(
            {"cases": [{k: v for k, v in _CASE.items() if k != "prompts"}]},
            "Case 'case' needs 'prompts'",
            id="no_prompts_key",
        ),
        pytest.param(
            {"cases": [{**_CASE, "prompts": []}]},
            "Case 'case' has no prompts",
            id="empty_prompts",
        ),
        pytest.param(
            {"cases": [{**_CASE, "limit": 0}]},
            "Case 'case' has no prompts",
            id="limit_leaves_no_prompts",
        ),
        pytest.param(
            {"cases": [{**_CASE, "schema_path": "schema.json"}]},
            "Case 'case' has unexpected keys ['schema']",
            id="schema_and_schema_path",
        ),
        pytest.param(
            {"cases": [{**_CASE, "temperature": 0.5}]},
            "Case 'case' has unexpected keys ['temperature']",
            id="unknown_key",
        ),
        pytest.param(
            {"cases": [_CASE, _CASE]},
            "Case names must be unique, got ['case', 'case']",
            id="duplicate_names",
        ),
        pytest.param(
            {"cases": [{**_CASE, "schema": {"format": "email"}}]},
            "Case 'case': Schema keywords ['format'] at # are not checked",
            id="uncheckable_schema",
        ),
    ],
)
def test_load_cases_rejects_a_malformed_task_config(tmp_path, config, message):
    (tmp_path / "schema.json").write_text(
        json.dumps({"structured_outputs": {"json": {"type": "string"}}})
    )
    path = _write_task_config(tmp_path, config)

    with pytest.raises(ValueError) as excinfo:
        eval_json_schema.load_cases(path)

    assert message in str(excinfo.value)
