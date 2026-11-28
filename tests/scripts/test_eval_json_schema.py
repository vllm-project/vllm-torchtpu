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
tested through ``run_eval_flow.sh`` in ``test_run_eval_flow_cli.py``, against
an endpoint that answers. The last test here covers one that does not.
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import pathlib
import socket
import subprocess
import sys
import threading
from typing import Any

import pytest
import yaml

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
        pytest.param(
            {"pattern": "[unclosed"},
            "Invalid pattern '[unclosed' at #",
            id="invalid_regex_pattern",
        ),
        pytest.param(
            {"pattern": 123},
            "pattern at # must be a string",
            id="non_string_pattern",
        ),
        pytest.param(
            {"$ref": 123},
            "$ref at # must be a string",
            id="non_string_ref",
        ),
        pytest.param(
            {"enum": "ab"},
            "enum at # must be a list",
            id="enum_not_a_list",
        ),
        pytest.param(
            {"required": "a"},
            "required at # must be a list of strings",
            id="required_not_a_list",
        ),
        pytest.param(
            {"required": [1]},
            "required at # must be a list of strings",
            id="required_non_string_member",
        ),
        pytest.param(
            {"anyOf": {"type": "string"}},
            "anyOf at # must be a list",
            id="any_of_not_a_list",
        ),
        pytest.param(
            {"minimum": True},
            "minimum at # must be a finite number",
            id="minimum_is_bool",
        ),
        pytest.param(
            {"maximum": float("inf")},
            "maximum at # must be a finite number",
            id="maximum_is_infinite",
        ),
        pytest.param(
            {"minimum": float("nan")},
            "minimum at # must be a finite number",
            id="minimum_is_nan",
        ),
        pytest.param(
            {"minItems": -1},
            "minItems at # must be a non-negative integer",
            id="min_items_negative",
        ),
        pytest.param(
            {"minItems": True},
            "minItems at # must be a non-negative integer",
            id="min_items_is_bool",
        ),
        pytest.param(
            {"properties": []},
            "properties at # must be an object",
            id="properties_not_an_object",
        ),
        pytest.param(
            {"$defs": []},
            "$defs at # must be an object",
            id="defs_not_an_object",
        ),
    ],
)
def test_validator_refuses_a_schema_it_cannot_check(schema, message):
    with pytest.raises(ValueError) as excinfo:
        eval_json_schema.SchemaValidator(schema)

    assert message in str(excinfo.value)


@pytest.mark.parametrize(
    ("schema", "raw_json", "token"),
    [
        pytest.param(
            {"type": "number", "minimum": 1, "maximum": 10},
            "NaN",
            "NaN",
            id="top_level_nan",
        ),
        pytest.param(
            {"type": "number", "minimum": 1, "maximum": 10},
            "Infinity",
            "Infinity",
            id="top_level_infinity",
        ),
        pytest.param(
            {"type": "number", "minimum": 1, "maximum": 10},
            "-Infinity",
            "-Infinity",
            id="top_level_neg_infinity",
        ),
        pytest.param(
            {
                "type": "object",
                "properties": {"limit": {"type": "number", "minimum": 1}},
            },
            '{"limit": NaN}',
            "NaN",
            id="nested_nan",
        ),
    ],
)
def test_validator_rejects_non_rfc_8259_constants(schema, raw_json, token):
    validator = eval_json_schema.SchemaValidator(schema)

    assert validator.check(raw_json, "stop") == f"Non-standard JSON constant {token!r}"


@pytest.mark.parametrize(
    ("pattern", "valid_value", "trailing_newline_value"),
    [
        pytest.param(
            "^[a-z0-9-]+$",
            "gateway-router",
            "gateway-router\n",
            id="service_slug",
        ),
        pytest.param(
            r"^[\w\d\s]{1,16}$",
            "1234567890123456",
            "1234567890123456\n",
            id="character_field_max_16",
        ),
        pytest.param(
            r"^\$$",
            "$",
            "$\n",
            id="escaped_dollar_then_anchor",
        ),
        pytest.param(
            r"^[$]+$",
            "$$",
            "$$\n",
            id="dollar_in_char_class_then_anchor",
        ),
    ],
)
def test_validator_enforces_ecma_end_anchor_without_trailing_newline(
    pattern, valid_value, trailing_newline_value
):
    validator = eval_json_schema.SchemaValidator({"type": "string", "pattern": pattern})

    assert validator.check(json.dumps(valid_value), "stop") is None
    error = validator.check(json.dumps(trailing_newline_value), "stop")
    assert error == f"String {trailing_newline_value!r} failed pattern {pattern!r}"


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
        pytest.param(
            {"cases": [{**_CASE, "limit": -1}]},
            "Case 'case': limit must be a non-negative integer",
            id="negative_limit",
        ),
        pytest.param(
            {"cases": [{**_CASE, "limit": True}]},
            "Case 'case': limit must be a non-negative integer",
            id="bool_limit",
        ),
        pytest.param(
            {"cases": [{**_CASE, "max_tokens": 0}]},
            "Case 'case': max_tokens must be a positive integer",
            id="zero_max_tokens",
        ),
        pytest.param(
            {"cases": [{**_CASE, "max_tokens": True}]},
            "Case 'case': max_tokens must be a positive integer",
            id="bool_max_tokens",
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


def test_load_cases_wraps_yaml_os_json_and_missing_human_errors(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("cases:\n  - [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to read task config"):
        eval_json_schema.load_cases(bad_yaml)

    with pytest.raises(ValueError, match="Failed to read task config"):
        eval_json_schema.load_cases(tmp_path / "missing.yaml")

    bad_schema_cfg = _write_task_config(
        tmp_path,
        {
            "cases": [
                {
                    "name": "bad_schema",
                    "max_tokens": 32,
                    "schema_path": "missing_schema.json",
                    "prompts": ["hi"],
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="failed to read schema_path"):
        eval_json_schema.load_cases(bad_schema_cfg)

    bad_dataset = tmp_path / "bad_dataset.json"
    bad_dataset.write_text("{not json", encoding="utf-8")
    bad_dataset_cfg = _write_task_config(
        tmp_path,
        {
            "cases": [
                {
                    "name": "bad_dataset",
                    "max_tokens": 32,
                    "schema": {"type": "string"},
                    "dataset_path": "bad_dataset.json",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="Failed to read dataset"):
        eval_json_schema.load_cases(bad_dataset_cfg)

    no_human = tmp_path / "no_human.json"
    no_human.write_text(
        json.dumps(
            [{"id": "rec-0", "conversations": [{"from": "gpt", "value": "only"}]}]
        ),
        encoding="utf-8",
    )
    no_human_cfg = _write_task_config(
        tmp_path,
        {
            "cases": [
                {
                    "name": "no_human",
                    "max_tokens": 32,
                    "schema": {"type": "string"},
                    "dataset_path": "no_human.json",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="has no 'human' turn"):
        eval_json_schema.load_cases(no_human_cfg)


def test_flat_task_config_matches_default_cases_verbatim():
    task_configs = _SCRIPT.parent / "task_configs"
    default_cfg = yaml.safe_load(
        (task_configs / "json_schema_default.yaml").read_text(encoding="utf-8")
    )
    flat_cfg = yaml.safe_load(
        (task_configs / "json_schema_flat.yaml").read_text(encoding="utf-8")
    )
    default_by_name = {case["name"]: case for case in default_cfg["cases"]}

    assert flat_cfg["cases"]
    for case in flat_cfg["cases"]:
        assert case["name"] in default_by_name
        assert case == default_by_name[case["name"]]


def _start_stub_server(handler_cls):
    class Server(http.server.ThreadingHTTPServer):
        request_queue_size = 32
        daemon_threads = True

    server = Server(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    return server


@pytest.mark.parametrize("status_code", [401, 403, 404, 429, 500])
def test_evaluate_raises_on_fatal_http_status(status_code):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            body = json.dumps({"error": f"status-{status_code}"}).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = _start_stub_server(Handler)
    try:
        case = eval_json_schema.SchemaCase(
            "case",
            eval_json_schema.SchemaValidator({"type": "string"}),
            32,
            (eval_json_schema.Sample("0", "hello"),),
        )
        with pytest.raises(RuntimeError, match=f"HTTP {status_code}"):
            eval_json_schema.evaluate(
                f"http://127.0.0.1:{server.server_address[1]}/",
                "any-model",
                [case],
                api_key=None,
            )
    finally:
        server.shutdown()
        server.server_close()


def test_evaluate_accepts_400_and_422_as_invalid_trials_and_strips_trailing_slash():
    seen_paths: list[str] = []
    codes = iter([400, 422, 200])

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:
            seen_paths.append(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            code = next(codes)
            if code == 200:
                payload = {
                    "choices": [
                        {"message": {"content": '"ok"'}, "finish_reason": "stop"}
                    ]
                }
            else:
                payload = {"error": f"bad-{code}"}
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = _start_stub_server(Handler)
    try:
        case = eval_json_schema.SchemaCase(
            "case",
            eval_json_schema.SchemaValidator({"type": "string"}),
            32,
            (
                eval_json_schema.Sample("0", "p0"),
                eval_json_schema.Sample("1", "p1"),
                eval_json_schema.Sample("2", "p2"),
            ),
        )
        # Single worker guarantees deterministic code order across samples.
        orig_concurrency = eval_json_schema._CONCURRENCY
        eval_json_schema._CONCURRENCY = 1
        try:
            trials = eval_json_schema.evaluate(
                f"http://127.0.0.1:{server.server_address[1]}/",
                "any-model",
                [case],
                api_key=None,
            )
        finally:
            eval_json_schema._CONCURRENCY = orig_concurrency
    finally:
        server.shutdown()
        server.server_close()

    assert seen_paths == ["/v1/chat/completions"] * 3
    assert [t.error for t in trials] == [
        'HTTP 400: {"error": "bad-400"}',
        'HTTP 422: {"error": "bad-422"}',
        None,
    ]


def test_evaluate_rejects_non_string_message_content():
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": ["not", "str"]},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = _start_stub_server(Handler)
    try:
        case = eval_json_schema.SchemaCase(
            "case",
            eval_json_schema.SchemaValidator({"type": "string"}),
            32,
            (eval_json_schema.Sample("0", "hello"),),
        )
        with pytest.raises(RuntimeError, match="Expected string or null content"):
            eval_json_schema.evaluate(
                f"http://127.0.0.1:{server.server_address[1]}",
                "any-model",
                [case],
                api_key=None,
            )
    finally:
        server.shutdown()
        server.server_close()


def test_evaluate_aborts_queued_trials_immediately_on_first_fatal_error(monkeypatch):
    monkeypatch.setattr(eval_json_schema, "_CONCURRENCY", 2)
    release_slow = threading.Event()
    seen_prompts: list[str] = []
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:
            payload = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            prompt = payload["messages"][0]["content"]
            with lock:
                seen_prompts.append(prompt)
            if prompt == "slow":
                release_slow.wait(timeout=10.0)
                body = json.dumps(
                    {
                        "choices": [
                            {"message": {"content": '"ok"'}, "finish_reason": "stop"}
                        ]
                    }
                ).encode("utf-8")
                self.send_response(200)
            else:
                body = b'{"error": "unauthorized"}'
                self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = _start_stub_server(Handler)
    try:
        samples = (
            eval_json_schema.Sample("0", "slow"),
            eval_json_schema.Sample("1", "fail"),
            *(eval_json_schema.Sample(str(i), f"queued-{i}") for i in range(2, 12)),
        )
        case = eval_json_schema.SchemaCase(
            "case",
            eval_json_schema.SchemaValidator({"type": "string"}),
            32,
            samples,
        )
        with pytest.raises(RuntimeError, match="HTTP 401"):
            eval_json_schema.evaluate(
                f"http://127.0.0.1:{server.server_address[1]}",
                "any-model",
                [case],
                api_key=None,
            )
        # Returning before release_slow is set proves we did not wait for the
        # in-flight slow request, and queued prompts were never dispatched.
        assert not release_slow.is_set()
        with lock:
            assert len(seen_prompts) < len(samples)
    finally:
        release_slow.set()
        server.shutdown()
        server.server_close()


def test_cli_exits_1_without_a_results_file_when_nothing_is_listening(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    output = tmp_path / "results_json_schema.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--model",
            "any-model",
            "--task-config",
            str(_write_task_config(tmp_path, {"cases": [_CASE]})),
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 1
    (error_line,) = proc.stderr.splitlines()
    assert error_line.startswith("ERROR: Request for case 'case' sample '0' failed: ")
    assert "Connection refused" in error_line
    assert not output.exists()
