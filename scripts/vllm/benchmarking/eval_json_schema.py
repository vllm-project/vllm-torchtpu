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
import math
import os
import re
import sys
import threading
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


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"Non-standard JSON constant {token!r}")


def _translate_ecma_end_anchor(pattern: str) -> str:
    """Replace unescaped ``$`` outside character classes with ``\\Z``.

    Python's ``$`` matches before a trailing newline as well as at end of
    string; ECMA-262 ``$`` (without multiline mode) matches only at end of
    string.
    """
    out: list[str] = []
    in_char_class = False
    char_class_start = -1
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "\\" and i + 1 < n:
            out.append(pattern[i : i + 2])
            i += 2
            continue
        if in_char_class:
            if ch == "]" and i > char_class_start:
                in_char_class = False
            out.append(ch)
        elif ch == "[":
            in_char_class = True
            char_class_start = i + 1
            if i + 1 < n and pattern[i + 1] == "^":
                char_class_start = i + 2
            out.append(ch)
        elif ch == "$":
            out.append(r"\Z")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


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
        self._patterns: dict[str, re.Pattern[str]] = {}
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
            if not isinstance(ref, str):
                raise ValueError(f"$ref at {where} must be a string, got {ref!r}")
            name = ref.removeprefix("#/$defs/")
            defs = self.schema.get("$defs", {})
            if name == ref or not isinstance(defs, dict) or name not in defs:
                raise ValueError(
                    f"$ref '{ref}' at {where} does not name a '#/$defs/<name>' "
                    "of this schema"
                )
            if set(schema) - {"$ref", "$defs"}:
                raise ValueError(f"Keywords beside $ref at {where} are not checked")
        if "type" in schema and (
            not isinstance(schema["type"], str) or schema["type"] not in _JSON_TYPES
        ):
            raise ValueError(
                f"type {schema['type']!r} at {where} is not one JSON type name"
            )
        if "enum" in schema:
            enum_values = schema["enum"]
            if not isinstance(enum_values, list):
                raise ValueError(f"enum at {where} must be a list")
            if not all(isinstance(member, str) for member in enum_values):
                raise ValueError(f"enum at {where} has members that are not strings")
        if "required" in schema:
            required = schema["required"]
            if not isinstance(required, list) or not all(
                isinstance(key, str) for key in required
            ):
                raise ValueError(f"required at {where} must be a list of strings")
        if "pattern" in schema:
            pattern = schema["pattern"]
            if not isinstance(pattern, str):
                raise ValueError(
                    f"pattern at {where} must be a string, got {pattern!r}"
                )
            try:
                self._patterns[pattern] = re.compile(
                    _translate_ecma_end_anchor(pattern)
                )
            except re.error as err:
                raise ValueError(
                    f"Invalid pattern {pattern!r} at {where}: {err}"
                ) from err
        for bound_key in ("minimum", "maximum"):
            if bound_key in schema:
                bound = schema[bound_key]
                if (
                    isinstance(bound, bool)
                    or not isinstance(bound, (int, float))
                    or not math.isfinite(bound)
                ):
                    raise ValueError(
                        f"{bound_key} at {where} must be a finite number, got {bound!r}"
                    )
        if "minItems" in schema:
            min_items = schema["minItems"]
            if (
                isinstance(min_items, bool)
                or not isinstance(min_items, int)
                or min_items < 0
            ):
                raise ValueError(
                    f"minItems at {where} must be a non-negative integer, "
                    f"got {min_items!r}"
                )
        if not isinstance(schema.get("additionalProperties", True), bool):
            raise ValueError(f"additionalProperties at {where} is not true or false")
        for keyword in ("properties", "$defs"):
            if keyword in schema:
                mapping = schema[keyword]
                if not isinstance(mapping, dict):
                    raise ValueError(f"{keyword} at {where} must be an object")
                for name, sub in mapping.items():
                    self._require_checkable(sub, f"{where}/{keyword}/{name}")
        if "anyOf" in schema:
            any_of = schema["anyOf"]
            if not isinstance(any_of, list):
                raise ValueError(f"anyOf at {where} must be a list")
            for index, sub in enumerate(any_of):
                self._require_checkable(sub, f"{where}/anyOf/{index}")
        if "items" in schema:
            self._require_checkable(schema["items"], f"{where}/items")

    def check(self, content: str, finish_reason: str | None) -> str | None:
        """Return why a completion is invalid, or None when it is valid."""
        if finish_reason == "length":
            return "Completion hit max_tokens (finish_reason=length)"
        try:
            value = json.loads(content, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as err:
            return f"JSONDecodeError: {err}"
        except ValueError as err:
            return str(err)
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
            and self._patterns[schema["pattern"]].search(value) is None
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
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"Failed to read dataset {path}: {err}") from err
    if not isinstance(records, list):
        raise ValueError(f"Dataset {path} must be a JSON list")
    samples: list[Sample] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset {path} record [{index}] must be an object")
        try:
            record_id = str(record["id"])
            conversations = record["conversations"]
            if not isinstance(conversations, list):
                raise ValueError(
                    f"Dataset {path} record {record_id!r} conversations must be a list"
                )
            prompt = next(
                turn["value"]
                for turn in conversations
                if isinstance(turn, dict) and turn.get("from") == "human"
            )
        except KeyError as err:
            raise ValueError(f"Dataset {path} record [{index}] needs {err}") from err
        except StopIteration as err:
            raise ValueError(
                f"Dataset {path} record {record.get('id')!r} has no 'human' turn"
            ) from err
        if not isinstance(prompt, str):
            raise ValueError(
                f"Dataset {path} record {record_id!r} 'human' turn value "
                "must be a string"
            )
        samples.append(Sample(record_id, prompt))
    return samples


def _parse_case(entry: Any, base_dir: Path) -> SchemaCase:
    if not isinstance(entry, dict):
        raise ValueError("Each case entry must be a mapping")
    # Each key is taken out as it is read, so a key left over is one this
    # function does not know, or the second of two that exclude each other.
    fields = dict(entry)
    try:
        name = fields.pop("name")
        max_tokens = fields.pop("max_tokens")
        if "schema_path" in fields:
            schema_rel = fields.pop("schema_path")
            if not isinstance(schema_rel, str):
                raise ValueError(f"Case {name!r}: schema_path must be a string")
            schema_file = base_dir / schema_rel
            try:
                body = schema_file.read_text(encoding="utf-8")
                parsed_body = json.loads(body)
                schema = parsed_body["structured_outputs"]["json"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as err:
                raise ValueError(
                    f"Case {name!r}: failed to read schema_path {schema_file}: {err}"
                ) from err
        else:
            schema = fields.pop("schema")
        if "dataset_path" in fields:
            dataset_rel = fields.pop("dataset_path")
            if not isinstance(dataset_rel, str):
                raise ValueError(f"Case {name!r}: dataset_path must be a string")
            samples = _dataset_samples(base_dir / dataset_rel)
        else:
            prompts = fields.pop("prompts")
            if not isinstance(prompts, list) or not all(
                isinstance(prompt, str) for prompt in prompts
            ):
                raise ValueError(f"Case {name!r}: prompts must be a list of strings")
            samples = [Sample(str(i), prompt) for i, prompt in enumerate(prompts)]
    except KeyError as err:
        raise ValueError(f"Case {entry.get('name')!r} needs {err}") from err
    if not isinstance(name, str) or not name:
        raise ValueError(f"Case name must be a non-empty string, got {name!r}")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError(
            f"Case '{name}': max_tokens must be a positive integer, got {max_tokens!r}"
        )
    limit = fields.pop("limit", None)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError(
                f"Case '{name}': limit must be a non-negative integer, got {limit!r}"
            )
        samples = samples[:limit]
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
    try:
        raw_text = path.read_text(encoding="utf-8")
        config = yaml.safe_load(raw_text)
    except (OSError, yaml.YAMLError) as err:
        raise ValueError(f"Failed to read task config {path}: {err}") from err
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
    abort: threading.Event,
) -> Trial | None:
    if abort.is_set():
        return None
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
            if not isinstance(content, (str, type(None))):
                raise TypeError(
                    f"Expected string or null content, got {type(content).__name__}"
                )
            if not isinstance(finish_reason, (str, type(None))):
                raise TypeError(
                    "Expected string or null finish_reason, "
                    f"got {type(finish_reason).__name__}"
                )
    except urllib.error.HTTPError as err:
        try:
            detail = err.read().decode("utf-8", errors="replace").strip()[:500]
        except (OSError, http.client.HTTPException) as read_err:
            detail = f"<failed to read error body: {read_err!r}>"
        finally:
            err.close()
        if err.code in (400, 422):
            # The server refusing a request, for example a schema it cannot
            # compile, is an outcome of the suite and not a broken run.
            return Trial(
                case.name, sample.sample_id, None, f"HTTP {err.code}: {detail}"
            )
        abort.set()
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
        abort.set()
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
        RuntimeError: The endpoint is unreachable, answers with an HTTP error
            other than 400 or 422, or does not answer with a chat completion.
    """
    chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    abort = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=_CONCURRENCY)
    failed = False
    try:
        futures = [
            pool.submit(_run_trial, chat_url, headers, model, case, sample, abort)
            for case in cases
            for sample in case.samples
        ]
        trials: list[Trial | None] = [None] * len(futures)
        future_to_index = {future: idx for idx, future in enumerate(futures)}
        for future in concurrent.futures.as_completed(futures):
            trials[future_to_index[future]] = future.result()
    except BaseException:
        failed = True
        abort.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not failed:
            pool.shutdown(wait=True)
    return [trial for trial in trials if trial is not None]


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
