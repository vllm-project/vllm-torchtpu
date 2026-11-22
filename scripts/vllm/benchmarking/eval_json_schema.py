#!/usr/bin/env python3
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
"""Check that schema-constrained chat completions are valid for their schema.

``run_eval_flow.sh`` runs this as its ``json_schema`` suite. Every prompt of a
task config goes to ``<base-url>/v1/chat/completions`` with its case's schema
as ``response_format``. A response is valid when it is not truncated
(``finish_reason`` other than ``length``), parses as JSON and satisfies the
schema. The share of valid responses is written as ``acc,none`` in the
``results_*.json`` layout that ``check_regression.py --mode eval`` reads.

Usage:
    eval_json_schema.py --base-url <api-root> --model <served-name> \\
        --task-config <yaml> --output-json <path>

``--base-url`` is the API root without ``/v1`` and ``--model`` the name the
endpoint serves, both as ``run_eval_flow.sh`` resolved them. ``OPENAI_API_KEY``,
when set, is sent as a bearer token.

A task config is a YAML mapping with one key, ``cases``, a list. Each case has:

    name          sent as the ``json_schema`` name, and names the case in the
                  report
    max_tokens    completion budget of each request
    schema        the JSON Schema, inline; or
    schema_path   a ``structured_schemas/*.json`` request body, whose
                  ``structured_outputs.json`` member is the schema
    prompts       the user prompts, a list of strings; or
    dataset_path  a JSON list of ``{"id", "conversations"}`` records, whose
                  first ``human`` turn is the prompt
    limit         optional: send only the first N prompts

Relative paths are resolved against the task config's directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

TASK = "json_schema"

# Requests in flight at once.
_CONCURRENCY = 16
_REQUEST_TIMEOUT_S = 120.0

# The Python types json.loads gives each JSON Schema type. bool is handled
# apart in _has_type: True is an int in Python and is not one in JSON.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "boolean": (bool,),
    "integer": (int,),
    "number": (int, float),
    "null": (type(None),),
}


def _has_type(value: Any, type_name: str) -> bool:
    if isinstance(value, bool):
        return type_name == "boolean"
    return isinstance(value, _JSON_TYPES[type_name])


class SchemaValidator:
    """Check completions against one JSON Schema, for a fixed set of keywords.

    Construction raises ``ValueError`` for a schema that uses any other
    keyword, or a form of a keyword that is not checked, so a constraint this
    class would skip never reads as satisfied.
    """

    _KEYWORDS = frozenset(
        {
            "$defs",
            "$ref",
            "additionalProperties",
            "anyOf",
            "enum",
            "items",
            "maximum",
            "minItems",
            "minimum",
            "pattern",
            "properties",
            "required",
            "type",
        }
    )

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self._require_checkable(schema, "#")

    def _require_checkable(self, schema: Any, where: str) -> None:
        if not isinstance(schema, dict):
            raise ValueError(f"Schema at {where} is not an object")
        unchecked = sorted(set(schema) - self._KEYWORDS)
        if unchecked:
            raise ValueError(
                f"Schema keywords {unchecked} at {where} are not checked by "
                "eval_json_schema.py"
            )
        if "$ref" in schema:
            ref = schema["$ref"]
            name = ref.removeprefix("#/$defs/")
            if name == ref or name not in self.schema.get("$defs", {}):
                raise ValueError(
                    f"$ref '{ref}' at {where} does not name a '#/$defs/<name>' "
                    "of this schema"
                )
            if set(schema) - {"$ref", "$defs"}:
                raise ValueError(f"Keywords beside $ref at {where} are not checked")
        if "type" in schema and schema["type"] not in tuple(_JSON_TYPES):
            raise ValueError(
                f"type {schema['type']!r} at {where} is not one JSON type name"
            )
        if not all(isinstance(member, str) for member in schema.get("enum", ())):
            raise ValueError(f"enum at {where} has members that are not strings")
        if not isinstance(schema.get("additionalProperties", True), bool):
            raise ValueError(f"additionalProperties at {where} is not true or false")
        for keyword in ("properties", "$defs"):
            for name, sub in schema.get(keyword, {}).items():
                self._require_checkable(sub, f"{where}/{keyword}/{name}")
        for index, sub in enumerate(schema.get("anyOf", ())):
            self._require_checkable(sub, f"{where}/anyOf/{index}")
        if "items" in schema:
            self._require_checkable(schema["items"], f"{where}/items")

    def check(self, content: str, finish_reason: str | None) -> str | None:
        """Return why a completion is invalid, or None when it is valid."""
        if finish_reason == "length":
            return "Completion hit max_tokens (finish_reason=length)"
        try:
            value = json.loads(content)
        except json.JSONDecodeError as err:
            return f"JSONDecodeError: {err}"
        return self._mismatch(value, self.schema)

    def _mismatch(self, value: Any, schema: dict[str, Any]) -> str | None:
        """Return the first way ``value`` violates ``schema``, or None."""
        if "$ref" in schema:
            target = self.schema["$defs"][schema["$ref"].removeprefix("#/$defs/")]
            return self._mismatch(value, target)
        if "anyOf" in schema and all(
            self._mismatch(value, sub) for sub in schema["anyOf"]
        ):
            return f"Instance {value!r} matched none of anyOf"
        if "enum" in schema and value not in schema["enum"]:
            return f"Value {value!r} not in enum {schema['enum']}"
        if "type" in schema and not _has_type(value, schema["type"]):
            return f"Expected JSON type {schema['type']}, got {type(value).__name__}"
        if (
            isinstance(value, str)
            and "pattern" in schema
            and re.search(schema["pattern"], value) is None
        ):
            return f"String {value!r} failed pattern {schema['pattern']!r}"
        if _has_type(value, "number"):
            if "minimum" in schema and value < schema["minimum"]:
                return f"{value} < minimum {schema['minimum']}"
            if "maximum" in schema and value > schema["maximum"]:
                return f"{value} > maximum {schema['maximum']}"
        if isinstance(value, list):
            if len(value) < schema.get("minItems", 0):
                return f"Array length {len(value)} < minItems {schema['minItems']}"
            if "items" in schema:
                for index, item in enumerate(value):
                    error = self._mismatch(item, schema["items"])
                    if error:
                        return f"Item [{index}]: {error}"
        if isinstance(value, dict):
            for key in schema.get("required", ()):
                if key not in value:
                    return f"Missing required property '{key}'"
            properties = schema.get("properties", {})
            for key, member in value.items():
                if key in properties:
                    error = self._mismatch(member, properties[key])
                    if error:
                        return f"Property '{key}': {error}"
                elif not schema.get("additionalProperties", True):
                    return f"Disallowed additional property '{key}'"
        return None


@dataclasses.dataclass(frozen=True)
class Sample:
    """One prompt of a case."""

    sample_id: str
    prompt: str


@dataclasses.dataclass(frozen=True)
class SchemaCase:
    """A schema, the prompts sent under it, and their completion budget."""

    name: str
    validator: SchemaValidator
    max_tokens: int
    samples: tuple[Sample, ...]


@dataclasses.dataclass(frozen=True)
class Trial:
    """The verdict on one response; ``error`` is None when it is valid."""

    case_name: str
    sample_id: str
    finish_reason: str | None
    error: str | None


def _dataset_samples(path: Path) -> list[Sample]:
    return [
        Sample(
            str(record["id"]),
            next(
                turn["value"]
                for turn in record["conversations"]
                if turn["from"] == "human"
            ),
        )
        for record in json.loads(path.read_text(encoding="utf-8"))
    ]


def _parse_case(entry: dict[str, Any], base_dir: Path) -> SchemaCase:
    # Each key is taken out as it is read, so a key left over is one this
    # function does not know, or the second of two that exclude each other.
    fields = dict(entry)
    try:
        name = fields.pop("name")
        max_tokens = fields.pop("max_tokens")
        if "schema_path" in fields:
            body = (base_dir / fields.pop("schema_path")).read_text(encoding="utf-8")
            schema = json.loads(body)["structured_outputs"]["json"]
        else:
            schema = fields.pop("schema")
        if "dataset_path" in fields:
            samples = _dataset_samples(base_dir / fields.pop("dataset_path"))
        else:
            prompts = fields.pop("prompts")
            samples = [Sample(str(i), prompt) for i, prompt in enumerate(prompts)]
    except KeyError as err:
        raise ValueError(f"Case {entry.get('name')!r} needs {err}") from err
    samples = samples[: fields.pop("limit", None)]
    if fields:
        raise ValueError(f"Case '{name}' has unexpected keys {sorted(fields)}")
    if not samples:
        raise ValueError(f"Case '{name}' has no prompts")
    try:
        validator = SchemaValidator(schema)
    except ValueError as err:
        raise ValueError(f"Case '{name}': {err}") from err
    return SchemaCase(name, validator, max_tokens, tuple(samples))


def load_cases(path: Path) -> tuple[SchemaCase, ...]:
    """Parse the task config at ``path``; the module docstring has the format.

    Raises:
        ValueError: The config does not have that format, or one of its
            schemas uses a keyword ``SchemaValidator`` does not check.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or set(config) != {"cases"}
        or not isinstance(config["cases"], list)
        or not config["cases"]
    ):
        raise ValueError(
            f"{path} must be a mapping with one key, 'cases', a non-empty list"
        )
    cases = tuple(_parse_case(entry, path.parent) for entry in config["cases"])
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        # A failed response is reported by its case name.
        raise ValueError(f"Case names must be unique, got {names}")
    return cases


def _request_body(model: str, case: SchemaCase, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": case.max_tokens,
        # Reasoning would spend the budget before the JSON starts. Chat
        # templates differ in the switch they read: most read enable_thinking,
        # Kimi's reads thinking.
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": case.name,
                "strict": True,
                "schema": case.validator.schema,
            },
        },
    }


def _run_trial(
    chat_url: str,
    headers: dict[str, str],
    model: str,
    case: SchemaCase,
    sample: Sample,
) -> Trial:
    request = urllib.request.Request(
        chat_url,
        data=json.dumps(_request_body(model, case, sample.prompt)).encode("utf-8"),
        headers=headers,
    )
    where = f"case '{case.name}' sample '{sample.sample_id}'"
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as resp:
            choice = json.load(resp)["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace").strip()[:500]
        if 400 <= err.code < 500:
            # The server refusing a request, for example a schema it cannot
            # compile, is an outcome of the suite and not a broken run.
            return Trial(
                case.name, sample.sample_id, None, f"HTTP {err.code}: {detail}"
            )
        raise RuntimeError(f"HTTP {err.code} for {where}: {detail}") from err
    # URLError covers a refused connection, OSError timeouts and resets,
    # HTTPException a broken HTTP reply, ValueError a body that is not JSON,
    # and LookupError and TypeError a body that is not a chat completion.
    except (
        urllib.error.URLError,
        OSError,
        http.client.HTTPException,
        ValueError,
        LookupError,
        TypeError,
    ) as err:
        raise RuntimeError(f"Request for {where} failed: {err!r}") from err
    # A reply without text, for example one that is only a tool call, has
    # null content and is invalid like any other non-JSON reply.
    error = case.validator.check(content or "", finish_reason)
    return Trial(case.name, sample.sample_id, finish_reason, error)


def evaluate(
    base_url: str,
    model: str,
    cases: Sequence[SchemaCase],
    *,
    api_key: str | None,
) -> list[Trial]:
    """Send every prompt of ``cases`` and return one verdict per response.

    ``api_key``, when set, is sent as a bearer token.

    Raises:
        RuntimeError: The endpoint is unreachable, answers with a 5xx, or does
            not answer with a chat completion.
    """
    chat_url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        futures = [
            pool.submit(_run_trial, chat_url, headers, model, case, sample)
            for case in cases
            for sample in case.samples
        ]
        return [future.result() for future in futures]


def results_json(model: str, trials: Sequence[Trial]) -> dict[str, Any]:
    """Return the ``results_json_schema.json`` content for ``trials``."""
    failed = [dataclasses.asdict(trial) for trial in trials if trial.error is not None]
    valid = len(trials) - len(failed)
    return {
        "results": {
            TASK: {
                "acc,none": valid / len(trials),
                # The share is a count over this run's responses and not an
                # estimate, so check_regression.py must not widen its
                # tolerance by a standard error.
                "acc_stderr,none": 0.0,
            }
        },
        "meta": {
            "model": model,
            "total_trials": len(trials),
            "successful_trials": valid,
            "failed_trials": failed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that schema-constrained chat completions are valid."
    )
    parser.add_argument(
        "--base-url", required=True, help="API root of the endpoint, without /v1."
    )
    parser.add_argument(
        "--model", required=True, help="Model name the endpoint serves."
    )
    parser.add_argument(
        "--task-config", required=True, type=Path, help="YAML task config."
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Where to write the results_json_schema.json.",
    )
    args = parser.parse_args()
    try:
        cases = load_cases(args.task_config)
        api_key = os.environ.get("OPENAI_API_KEY") or None
        trials = evaluate(args.base_url, args.model, cases, api_key=api_key)
    except (RuntimeError, ValueError) as err:
        # No results file is written: the caller fails the suite on the exit
        # status.
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    results = results_json(args.model, trials)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    meta = results["meta"]
    print(
        f"[{TASK}] model={args.model} "
        f"valid={meta['successful_trials']}/{meta['total_trials']} "
        f"-> {args.output_json}"
    )
    for failed in meta["failed_trials"][:10]:
        print(
            f"  INVALID [{failed['case_name']}#{failed['sample_id']}]: "
            f"{failed['error']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
