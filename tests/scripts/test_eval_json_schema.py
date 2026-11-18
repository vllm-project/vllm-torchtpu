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
"""Unit tests for eval_json_schema.py and run_eval_flow.sh suite CLI."""

from __future__ import annotations

import http.server
import importlib.util
import json
import math
import pathlib
import subprocess
import sys
import threading
from typing import Any

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BENCH_DIR = _REPO_ROOT / "scripts" / "vllm" / "benchmarking"
_EVAL_SCRIPT = _BENCH_DIR / "eval_json_schema.py"
_CHECK_REG_SCRIPT = _BENCH_DIR / "check_regression.py"


def _load_module(name: str, path: pathlib.Path):
    assert path.exists(), f"Missing script: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_eval_json_schema():
    return _load_module("eval_json_schema", _EVAL_SCRIPT)


def _load_check_regression():
    return _load_module("check_regression_for_schema_test", _CHECK_REG_SCRIPT)


class _MockOpenAIHandler(http.server.BaseHTTPRequestHandler):
    models_payload: dict[str, Any] = {"data": [{"id": "Qwen/Qwen3-0.6B"}]}
    recorded_requests: list[dict[str, Any]] = []
    responder = None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            body = json.dumps(self.models_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.__class__.recorded_requests.append(payload)
            fn = self.__class__.responder
            status, resp_obj = fn(payload)
            body = json.dumps(resp_obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


@pytest.fixture
def mock_openai_server():
    class Handler(_MockOpenAIHandler):
        models_payload = {"data": [{"id": "Qwen/Qwen3-0.6B"}]}
        recorded_requests = []
        responder = staticmethod(
            lambda req: (
                200,
                {
                    "choices": [
                        {
                            "message": {"content": '{"ok": true}'},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        )

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", Handler
    finally:
        server.shutdown()
        server.server_close()


def test_validator_checks_shipped_schemas():
    mod = _load_eval_json_schema()

    flat_schema = json.loads(
        (_BENCH_DIR / "structured_schemas" / "flat_json.json").read_text()
    )["structured_outputs"]["json"]
    flat_validator = mod.SchemaValidator(flat_schema, use_external=False)

    valid_users = {
        "users": [
            {
                "name": f"user_{i}",
                "age": 20 + i,
                "email": f"u{i}@example.com",
                "is_active": True,
                "role": "admin",
            }
            for i in range(10)
        ]
    }
    assert flat_validator.validate(json.dumps(valid_users), "stop").is_valid
    # Too few items (< minItems 10)
    short_users = {"users": valid_users["users"][:3]}
    res_short = flat_validator.validate(json.dumps(short_users), "stop")
    assert not res_short.is_valid
    assert res_short.json_parse_ok and not res_short.schema_ok
    # Bool passed where integer is required must fail even though isinstance(True, int)
    bad_int = {"users": [dict(valid_users["users"][0], age=True)] * 10}
    assert not flat_validator.validate(json.dumps(bad_int), "stop").is_valid

    ast_schema = json.loads(
        (_BENCH_DIR / "structured_schemas" / "complex_ast.json").read_text()
    )["structured_outputs"]["json"]
    ast_validator = mod.SchemaValidator(ast_schema, use_external=False)
    valid_ast = {
        "expressions": [
            {
                "ast": {
                    "op": "+",
                    "left": 1,
                    "right": {"op": "*", "left": "x", "right": 2},
                },
                "description": f"expr {i}",
            }
            for i in range(10)
        ]
    }
    assert ast_validator.validate(json.dumps(valid_ast), "stop").is_valid
    bad_ast = {
        "expressions": [
            {"ast": {"op": "^", "left": 1, "right": 2}, "description": "bad op"}
            for _ in range(10)
        ]
    }
    assert not ast_validator.validate(json.dumps(bad_ast), "stop").is_valid

    char_schema = json.loads(
        (_BENCH_DIR / "structured_schemas" / "character_schema.json").read_text()
    )["structured_outputs"]["json"]
    char_validator = mod.SchemaValidator(char_schema, use_external=False)
    valid_char = {
        "name": "Harry Potter",
        "house": "Gryffindor",
        "blood status": "Half-blood",
        "occupation": "auror",
        "wand": {"wood": "Holly", "core": "Phoenix feather", "length": 11},
        "alive": "Alive",
        "patronus": "Stag",
        "bogart": "Dementor",
    }
    assert char_validator.validate(json.dumps(valid_char), "stop").is_valid
    extra_prop_char = dict(valid_char, extra_key="nope")
    assert not char_validator.validate(json.dumps(extra_prop_char), "stop").is_valid
    truncated = char_validator.validate(json.dumps(valid_char), "length")
    assert not truncated.is_valid and not truncated.non_truncated


def test_cli_end_to_end_and_check_regression_integration(tmp_path, mock_openai_server):
    mod = _load_eval_json_schema()
    check_reg = _load_check_regression()
    base_url, handler = mock_openai_server

    call_idx = {"n": 0}

    def responder(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        assert payload["model"] == "Qwen/Qwen3-0.6B"
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["chat_template_kwargs"] == {
            "enable_thinking": False,
            "thinking": False,
        }
        call_idx["n"] += 1
        # 3 valid, 1 invalid JSON across 4 prompts -> acc = 0.75
        content = '{"answer": 42}' if call_idx["n"] <= 3 else '{"answer": "wrong"}'
        return (
            200,
            {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]},
        )

    handler.responder = staticmethod(responder)

    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        json.dumps(
            {
                "task": "json_schema",
                "concurrency": 1,
                "temperature": 0.0,
                "max_tokens": 128,
                "thinking": False,
                "cases": [
                    {
                        "name": "simple_answer",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                        "prompts": ["q1", "q2", "q3", "q4"],
                    }
                ],
            }
        )
    )

    out_json = tmp_path / "results" / "results_json_schema.json"
    rc = mod.main(
        [
            "--base-url",
            f"{base_url}/v1",
            "--task-config",
            str(task_yaml),
            "--output-json",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.exists()

    loaded = check_reg.load_eval_results(out_json.parent)
    assert "json_schema" in loaded
    assert loaded["json_schema"]["acc,none"] == 0.75
    assert loaded["json_schema"]["acc_stderr,none"] == 0.0

    # Verify check_regression.py passes against 0.75 baseline and fails against 1.0
    # (deterministic=True emits acc_stderr=0.0 so a single broken schema response
    # triggers a strict gate failure rather than hiding inside binomial noise).
    pass_check = check_reg.check_eval(
        loaded,
        {"json_schema": {"acc,none": 0.75, "acc_stderr,none": 0.0}},
        tolerance=0.0,
    )
    assert pass_check.failed_count == 0
    fail_check = check_reg.check_eval(
        loaded,
        {"json_schema": {"acc,none": 1.0, "acc_stderr,none": 0.0}},
        tolerance=0.0,
    )
    assert fail_check.failed_count == 1

    # Also verify non-deterministic binomial stderr calculation when deterministic=False
    _, binom_stderr, _, _ = mod.compute_accuracy_and_stderr(
        [
            mod.SampleTrialResult(
                case_name="c",
                sample_id=str(i),
                run_idx=0,
                validation=mod.ValidationResult(
                    is_valid=(i < 3),
                    json_parse_ok=True,
                    schema_ok=(i < 3),
                    non_truncated=True,
                ),
                finish_reason="stop",
                latency_ms=1.0,
            )
            for i in range(4)
        ],
        1,
        deterministic=False,
    )
    assert math.isclose(binom_stderr, math.sqrt(0.75 * 0.25 / 4), rel_tol=1e-6)


def test_auto_detect_model_rejects_zero_or_multiple_models(mock_openai_server):
    mod = _load_eval_json_schema()
    base_url, handler = mock_openai_server

    handler.models_payload = {"data": []}
    with pytest.raises(RuntimeError, match="No models"):
        mod.auto_detect_model(base_url, None)

    handler.models_payload = {"data": [{"id": "m1"}, {"id": "m2"}]}
    with pytest.raises(RuntimeError, match="Multiple models"):
        mod.auto_detect_model(base_url, None)

    assert mod.auto_detect_model(base_url, "m2") == "m2"


def test_task_config_rejects_duplicate_case_names():
    mod = _load_eval_json_schema()
    case = {"name": "dup", "schema": {"type": "object"}, "prompts": ["Return {}."]}
    with pytest.raises(ValueError, match="unique"):
        mod.TaskConfig.from_dict({"cases": [case, dict(case)]})


def test_default_task_config_loads_all_shipped_cases():
    mod = _load_eval_json_schema()
    default_cfg_path = _BENCH_DIR / "task_configs" / "json_schema_default.yaml"
    assert default_cfg_path.exists()
    cfg = mod.TaskConfig.from_yaml_file(default_cfg_path)
    assert cfg.task_name == "json_schema"
    case_names = [c.name for c in cfg.cases]
    assert case_names == [
        "flat_json",
        "complex_ast",
        "character_schema",
        "tool_call_arguments",
    ]
    assert sum(len(c.samples) for c in cfg.cases) >= 20

    flat_cfg_path = _BENCH_DIR / "task_configs" / "json_schema_flat.yaml"
    assert flat_cfg_path.exists()
    flat_cfg = mod.TaskConfig.from_yaml_file(flat_cfg_path)
    assert [c.name for c in flat_cfg.cases] == [
        "flat_json",
        "character_schema",
        "tool_call_arguments",
    ]
    assert sum(len(c.samples) for c in flat_cfg.cases) == 24


def _chat_reply(obj: Any) -> tuple[int, dict[str, Any]]:
    message = {"content": json.dumps(obj)}
    return 200, {"choices": [{"message": message, "finish_reason": "stop"}]}


def _valid_reply(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """A schema-valid answer to any case of json_schema_default.yaml."""
    schema = payload["response_format"]["json_schema"]["schema"]
    props = schema.get("properties", {})
    if "users" in props:
        obj = {
            "users": [
                {
                    "name": "a",
                    "age": 1,
                    "email": "a@b.c",
                    "is_active": True,
                    "role": "user",
                }
                for _ in range(10)
            ]
        }
    elif "expressions" in props:
        obj = {
            "expressions": [
                {"ast": {"op": "+", "left": 1, "right": 2}, "description": "d"}
                for _ in range(10)
            ]
        }
    elif "house" in props:
        obj = {
            "name": "Harry Potter",
            "house": "Gryffindor",
            "blood status": "Half-blood",
            "occupation": "auror",
            "wand": {"wood": "Holly", "core": "Phoenix", "length": 11},
            "alive": "Alive",
            "patronus": "Stag",
            "bogart": "Dementor",
        }
    else:
        obj = {
            "tool_name": "query_logs",
            "arguments": {"service": "gateway", "limit": 10, "severity": "ERROR"},
        }
    return _chat_reply(obj)


def _run_flow_json_schema(
    tmp_path: pathlib.Path,
    base_url: str,
    config: str,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess, pathlib.Path]:
    """Runs only the json_schema suite of `config` against `base_url`."""
    results_dir = tmp_path / "flow_results"
    # Stub vllm pkg on PYTHONPATH so run_benchmarks.sh --dry-run runs on CPU.
    fake_site = tmp_path / "fake_site"
    (fake_site / "vllm").mkdir(parents=True)
    (fake_site / "vllm" / "__init__.py").write_text("")
    proc = subprocess.run(
        [
            "bash",
            str(_BENCH_DIR / "run_eval_flow.sh"),
            "--config",
            config,
            "--base-url",
            base_url,
            "--perf-suites",
            "none",
            "--eval-suites",
            "json_schema",
            "--results-dir",
            str(results_dir),
            "--skip-db-upload",
        ],
        cwd=str(_REPO_ROOT),
        env={"PYTHONPATH": str(fake_site), "PATH": "/usr/bin:/bin", **(env or {})},
        capture_output=True,
        text=True,
    )
    return proc, results_dir


def test_run_eval_flow_json_schema_passes_when_every_response_is_valid(
    tmp_path, mock_openai_server
):
    base_url, handler = mock_openai_server
    handler.responder = staticmethod(_valid_reply)

    # The config's own EVAL_SUITES names json_schema too; the flag is explicit.
    proc, results_dir = _run_flow_json_schema(
        tmp_path, base_url, "qwen3-0.6b-structured-flat-json"
    )

    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    data = json.loads((results_dir / "results_json_schema.json").read_text())
    assert data["results"]["json_schema"]["acc,none"] == 1.0
    assert data["meta"] == {
        "model": "Qwen/Qwen3-0.6B",
        "total_trials": 24,
        "successful_trials": 24,
        "failed_trials": [],
    }
    assert (results_dir / "eval_check_json_schema.md").exists()
    assert "PASS: all evaluated tasks within tolerance." in proc.stdout
    # The client is given the model name the endpoint serves.
    assert {request["model"] for request in handler.recorded_requests} == {
        "Qwen/Qwen3-0.6B"
    }


def test_run_eval_flow_json_schema_fails_on_one_invalid_response(
    tmp_path, mock_openai_server
):
    base_url, handler = mock_openai_server
    replies = {"count": 0}

    def one_invalid(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        replies["count"] += 1
        if replies["count"] == 5:
            return _chat_reply({"not": "what the schema asked for"})
        return _valid_reply(payload)

    handler.responder = staticmethod(one_invalid)

    # qwen3-0.6b-tp1 ships no json_schema baseline: the rule needs none, and
    # the suite runs check_regression with --tolerance 0, so the config's
    # EVAL_TOLERANCE does not apply.
    proc, results_dir = _run_flow_json_schema(tmp_path, base_url, "qwen3-0.6b-tp1")

    assert proc.returncode == 1, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "| `json_schema` | acc,none | 0.9643 | 1.0000 |" in proc.stdout
    assert "FAIL" in (results_dir / "eval_check_json_schema.md").read_text()
    assert "json_schema: 27 of 28 responses are schema-valid" in proc.stderr
    assert "::error::Eval regression check failed for json_schema!" in proc.stdout


def test_run_eval_flow_json_schema_rule_ignores_reported_stderr(
    tmp_path, mock_openai_server
):
    # deterministic: false reports a binomial stderr, which check_regression.py
    # turns into an allowed drop. One invalid response in 50 must still fail.
    base_url, handler = mock_openai_server
    replies = {"count": 0}

    def one_invalid(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        replies["count"] += 1
        return _chat_reply({"answer": "wrong" if replies["count"] == 7 else 42})

    handler.responder = staticmethod(one_invalid)
    task_config = tmp_path / "task.json"
    task_config.write_text(
        json.dumps(
            {
                "task": "json_schema",
                "deterministic": False,
                "cases": [
                    {
                        "name": "simple_answer",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                        },
                        "prompts": [f"q{i}" for i in range(50)],
                    }
                ],
            }
        )
    )

    proc, _ = _run_flow_json_schema(
        tmp_path,
        base_url,
        "qwen3-0.6b-tp1",
        env={"JSON_SCHEMA_TASK_CONFIG": str(task_config)},
    )

    assert proc.returncode == 1, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "PASS: all evaluated tasks within tolerance." in proc.stdout
    assert "json_schema: 49 of 50 responses are schema-valid" in proc.stderr


def _write_single_case_task(tmp_path: pathlib.Path, prompts: list[str]) -> pathlib.Path:
    task_config = tmp_path / "task.json"
    task_config.write_text(
        json.dumps(
            {
                "task": "json_schema",
                "concurrency": 1,
                "cases": [
                    {
                        "name": "simple_answer",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                        },
                        "prompts": prompts,
                    }
                ],
            }
        )
    )
    return task_config


def test_http_error_response_counts_as_an_invalid_trial(tmp_path, mock_openai_server):
    mod = _load_eval_json_schema()
    base_url, handler = mock_openai_server
    replies = {"count": 0}

    def reject_second(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        replies["count"] += 1
        if replies["count"] == 2:
            return 400, {"error": {"message": "schema not supported"}}
        return _chat_reply({"answer": 42})

    handler.responder = staticmethod(reject_second)
    out_json = tmp_path / "results_json_schema.json"

    rc = mod.main(
        [
            "--base-url",
            base_url,
            "--task-config",
            str(_write_single_case_task(tmp_path, ["q1", "q2", "q3", "q4"])),
            "--output-json",
            str(out_json),
        ]
    )

    assert rc == 0
    data = json.loads(out_json.read_text())
    assert data["results"]["json_schema"]["acc,none"] == 0.75
    assert data["meta"]["total_trials"] == 4
    assert data["meta"]["successful_trials"] == 3
    assert data["meta"]["failed_trials"] == [
        {
            "case_name": "simple_answer",
            "sample_id": "1",
            "run_idx": 0,
            "finish_reason": None,
            "error": 'HTTP 400: {"error": {"message": "schema not supported"}}',
        }
    ]


def test_unreachable_endpoint_exits_with_one_line_and_no_results(tmp_path, capsys):
    mod = _load_eval_json_schema()
    out_json = tmp_path / "results_json_schema.json"

    rc = mod.main(
        [
            "--base-url",
            "http://127.0.0.1:9",
            "--model",
            "any",
            "--task-config",
            str(_write_single_case_task(tmp_path, ["q1"])),
            "--output-json",
            str(out_json),
        ]
    )

    assert rc == 1
    assert not out_json.exists()
    stderr_lines = capsys.readouterr().err.splitlines()
    assert len(stderr_lines) == 1
    assert stderr_lines[0].startswith(
        "ERROR: Request to http://127.0.0.1:9/v1/chat/completions failed on case "
        "'simple_answer' sample "
    )


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (
            {"type": "integer", "exclusiveMinimum": 0},
            "Schema keywords ['exclusiveMinimum'] at # need the jsonschema package",
        ),
        (
            {
                "type": "object",
                "properties": {"tags": {"type": "array", "uniqueItems": True}},
            },
            "Schema keywords ['uniqueItems'] at #/properties/tags need the "
            "jsonschema package",
        ),
        (
            {"anyOf": [{"type": "string", "format": "email"}]},
            "Schema keywords ['format'] at #/anyOf/0 need the jsonschema package",
        ),
        (
            {"type": "array", "items": {"type": "number", "multipleOf": 5}},
            "Schema keywords ['multipleOf'] at #/items need the jsonschema package",
        ),
        (
            {"type": "array", "items": [{"type": "integer"}]},
            "Schema at #/items is not an object",
        ),
    ],
)
def test_builtin_validator_refuses_schema_keywords_it_does_not_check(schema, message):
    mod = _load_eval_json_schema()

    with pytest.raises(ValueError) as excinfo:
        mod.SchemaValidator(schema, use_external=False)

    assert message in str(excinfo.value)


def test_builtin_validator_accepts_every_shipped_schema():
    mod = _load_eval_json_schema()
    config = mod.TaskConfig.from_yaml_file(
        _BENCH_DIR / "task_configs" / "json_schema_default.yaml"
    )

    for case in config.cases:
        mod.SchemaValidator(case.schema, use_external=False)
