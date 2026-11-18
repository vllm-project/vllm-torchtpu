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
"""Evaluate OpenAI-compatible structured outputs against JSON Schema suites.

Usage:
    eval_json_schema.py --base-url <url> [--model <name>] \\
        --task-config <yaml> --output-json <path>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import enum
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class RequestMode(enum.StrEnum):
    """How structured-output constraints are passed in chat completion requests."""

    RESPONSE_FORMAT = "response_format"
    STRUCTURED_OUTPUTS = "structured_outputs"

    @classmethod
    def from_str(cls, value: str) -> RequestMode:
        norm = value.strip().lower()
        if norm in ("response_format", "json_schema", "openai"):
            return cls.RESPONSE_FORMAT
        if norm in ("structured_outputs", "json", "vllm"):
            return cls.STRUCTURED_OUTPUTS
        raise ValueError(
            f"Unknown request_mode '{value}'. Expected 'response_format' or "
            "'structured_outputs'."
        )


@dataclasses.dataclass(frozen=True)
class PromptSample:
    """One prompt sent to the model under a schema constraint."""

    sample_id: str
    prompt: str
    system_prompt: str | None = None


@dataclasses.dataclass(frozen=True)
class SchemaCase:
    """One schema and its associated prompt set."""

    name: str
    schema: dict[str, Any]
    samples: tuple[PromptSample, ...]
    request_mode: RequestMode | None = None
    max_tokens: int | None = None


@dataclasses.dataclass(frozen=True)
class TaskConfig:
    """Parsed configuration for a JSON Schema evaluation run."""

    task_name: str
    cases: tuple[SchemaCase, ...]
    request_mode: RequestMode = RequestMode.RESPONSE_FORMAT
    deterministic: bool = True
    num_runs: int = 1
    concurrency: int = 16
    temperature: float = 0.0
    max_tokens: int = 512
    enable_thinking: bool = False
    allow_truncated: bool = False
    timeout_seconds: float = 120.0

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], base_dir: Path | None = None
    ) -> TaskConfig:
        if not isinstance(data, Mapping):
            raise ValueError("TaskConfig root must be a mapping.")
        task_name = str(data.get("task", data.get("task_name", "json_schema")))
        request_mode = RequestMode.from_str(
            str(data.get("request_mode", "response_format"))
        )
        deterministic = bool(data.get("deterministic", True))
        global_limit = data.get("limit")
        limit = int(global_limit) if global_limit is not None else None
        num_runs = max(1, int(data.get("num_runs", 1)))
        concurrency = max(1, int(data.get("concurrency", 16)))
        temperature = float(data.get("temperature", 0.0))
        max_tokens = int(data.get("max_tokens", 512))
        enable_thinking = bool(data.get("thinking", data.get("enable_thinking", False)))
        allow_truncated = bool(data.get("allow_truncated", False))
        timeout_seconds = float(data.get("timeout_seconds", 120.0))

        raw_cases = data.get("cases")
        if raw_cases is not None:
            if not isinstance(raw_cases, Sequence) or isinstance(
                raw_cases, (str, bytes)
            ):
                raise ValueError("'cases' must be a non-empty list of case mappings.")
            cases = tuple(
                _parse_case(c, idx, base_dir=base_dir, default_limit=limit)
                for idx, c in enumerate(raw_cases)
            )
        else:
            cases = (_parse_case(data, 0, base_dir=base_dir, default_limit=limit),)

        if not cases or all(not c.samples for c in cases):
            raise ValueError("TaskConfig resolved zero prompt samples.")
        names = [c.name for c in cases]
        if len(set(names)) != len(names):
            # Validators and per-case scores are keyed by name; a repeat would
            # check one case's samples against another case's schema.
            raise ValueError(f"TaskConfig case names must be unique: {names}")

        return cls(
            task_name=task_name,
            cases=cases,
            request_mode=request_mode,
            deterministic=deterministic,
            num_runs=num_runs,
            concurrency=concurrency,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            allow_truncated=allow_truncated,
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def from_yaml_file(cls, path: Path | str) -> TaskConfig:
        resolved = Path(path).resolve()
        content = resolved.read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore[import-untyped]

            parsed = yaml.safe_load(content)
        except ImportError:
            parsed = json.loads(content)
        except Exception as err:
            raise ValueError(f"Failed to parse task config {resolved}: {err}") from err
        if not isinstance(parsed, Mapping):
            raise ValueError(f"Task config {resolved} must parse to a mapping.")
        return cls.from_dict(parsed, base_dir=resolved.parent)


def _resolve_path(raw: str | Path, base_dir: Path | None) -> Path:
    path = Path(raw)
    if not path.is_absolute() and base_dir is not None:
        return (base_dir / path).resolve()
    return path


def _unwrap_schema(raw_schema: Mapping[str, Any]) -> dict[str, Any]:
    schema = dict(raw_schema)
    so = schema.get("structured_outputs")
    if isinstance(so, Mapping) and isinstance(so.get("json"), Mapping):
        return dict(so["json"])
    return schema


def _parse_case(
    entry: Mapping[str, Any],
    index: int,
    *,
    base_dir: Path | None,
    default_limit: int | None,
) -> SchemaCase:
    if not isinstance(entry, Mapping):
        raise ValueError(f"Case [{index}] must be a mapping.")
    name = str(entry.get("name", entry.get("task", f"case_{index}")))

    if "schema" in entry:
        if not isinstance(entry["schema"], Mapping):
            raise ValueError(f"Case '{name}' inline schema must be a mapping.")
        schema = _unwrap_schema(entry["schema"])
    elif "schema_path" in entry:
        schema_file = _resolve_path(entry["schema_path"], base_dir)
        schema = _unwrap_schema(json.loads(schema_file.read_text(encoding="utf-8")))
    else:
        raise ValueError(f"Case '{name}' requires 'schema' or 'schema_path'.")

    samples: list[PromptSample] = []
    if "dataset_path" in entry:
        ds_file = _resolve_path(entry["dataset_path"], base_dir)
        raw_ds = json.loads(ds_file.read_text(encoding="utf-8"))
        if not isinstance(raw_ds, list):
            raise ValueError(f"Dataset {ds_file} must be a JSON list.")
        for idx, item in enumerate(raw_ds):
            sid = str(item.get("id", idx))
            conv = item.get("conversations", [])
            prompt = next(
                (m["value"] for m in conv if m.get("from") == "human" and "value" in m),
                "",
            )
            if prompt:
                samples.append(PromptSample(sample_id=sid, prompt=prompt))
    elif "prompts" in entry:
        for idx, item in enumerate(entry["prompts"]):
            if isinstance(item, str):
                samples.append(PromptSample(sample_id=str(idx), prompt=item))
            elif isinstance(item, Mapping) and "prompt" in item:
                sys_prompt = item.get("system_prompt")
                samples.append(
                    PromptSample(
                        sample_id=str(item.get("id", idx)),
                        prompt=str(item["prompt"]),
                        system_prompt=str(sys_prompt) if sys_prompt else None,
                    )
                )
            else:
                raise ValueError(f"Invalid prompt entry at index {idx} in '{name}'.")
    else:
        samples.append(
            PromptSample(
                sample_id="0",
                prompt=(
                    "Return a JSON object that conforms to the requested JSON schema."
                ),
            )
        )

    case_limit_raw = entry.get("limit", default_limit)
    if case_limit_raw is not None:
        case_limit = int(case_limit_raw)
        if case_limit > 0:
            samples = samples[:case_limit]

    req_mode = (
        RequestMode.from_str(str(entry["request_mode"]))
        if "request_mode" in entry
        else None
    )
    case_max_tokens = int(entry["max_tokens"]) if "max_tokens" in entry else None
    return SchemaCase(
        name=name,
        schema=schema,
        samples=tuple(samples),
        request_mode=req_mode,
        max_tokens=case_max_tokens,
    )


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    """Validation verdict for a single completion."""

    is_valid: bool
    json_parse_ok: bool
    schema_ok: bool
    non_truncated: bool
    error_message: str | None = None


@dataclasses.dataclass(frozen=True)
class SampleTrialResult:
    """Outcome of a single prompt evaluation."""

    case_name: str
    sample_id: str
    run_idx: int
    validation: ValidationResult
    finish_reason: str | None
    latency_ms: float


@dataclasses.dataclass(frozen=True)
class SuiteResult:
    """Aggregated JSON Schema evaluation metrics."""

    task_name: str
    model_name: str
    total_trials: int
    successful_trials: int
    accuracy: float
    accuracy_stderr: float
    json_parse_rate: float
    non_truncated_rate: float
    per_case_accuracy: dict[str, float]
    trials: tuple[SampleTrialResult, ...]

    def to_regression_dict(self) -> dict[str, Any]:
        task_entry: dict[str, Any] = {
            "acc,none": self.accuracy,
            "acc_stderr,none": self.accuracy_stderr,
            "json_parse_rate,none": self.json_parse_rate,
            "non_truncated_rate,none": self.non_truncated_rate,
        }
        for case_name, score in sorted(self.per_case_accuracy.items()):
            task_entry[f"case_{case_name}_acc,none"] = score
        failed_trials = [
            {
                "case_name": t.case_name,
                "sample_id": t.sample_id,
                "run_idx": t.run_idx,
                "finish_reason": t.finish_reason,
                "error": t.validation.error_message,
            }
            for t in self.trials
            if not t.validation.is_valid
        ]
        return {
            "results": {self.task_name: task_entry},
            "meta": {
                "model": self.model_name,
                "total_trials": self.total_trials,
                "successful_trials": self.successful_trials,
                "failed_trials": failed_trials,
            },
        }

    def write_regression_json(self, destination: Path | str) -> None:
        path = Path(destination).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self.to_regression_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)


class SchemaValidator:
    """Self-contained Draft-07 / 2020-12 JSON Schema validator.

    Uses `jsonschema` when installed and `use_external=True`, falling back to a
    recursive standard-library validator so CPU test runners need no extra deps.
    The fallback implements a subset of the keywords and refuses a schema that
    uses any other, so an unchecked constraint never reads as a valid response.
    """

    # Checked by _validate_node.
    _CHECKED_KEYWORDS = frozenset(
        {
            "$ref",
            "additionalProperties",
            "allOf",
            "anyOf",
            "const",
            "enum",
            "items",
            "maxItems",
            "maxLength",
            "maximum",
            "minItems",
            "minLength",
            "minimum",
            "oneOf",
            "pattern",
            "properties",
            "required",
            "type",
        }
    )
    # Constrain nothing, so skipping them cannot pass an invalid instance.
    _ANNOTATION_KEYWORDS = frozenset(
        {
            "$comment",
            "$defs",
            "$id",
            "$schema",
            "default",
            "definitions",
            "description",
            "examples",
            "title",
        }
    )

    def __init__(self, root_schema: dict[str, Any], *, use_external: bool = True):
        self._root_schema = root_schema
        self._external_validator: Any = None
        if use_external:
            try:
                import jsonschema  # type: ignore[import-untyped]

                self._external_validator = jsonschema.Draft202012Validator(root_schema)
            except Exception:
                self._external_validator = None
        if self._external_validator is None:
            self._require_checked_keywords(root_schema, "#")

    def _require_checked_keywords(self, schema: Any, where: str) -> None:
        """Raise ValueError if the fallback would skip a constraint in `schema`."""
        if isinstance(schema, bool):
            raise ValueError(f"Boolean schema at {where} needs the jsonschema package")
        if not isinstance(schema, Mapping):
            raise ValueError(f"Schema at {where} is not an object")
        unchecked = sorted(
            set(schema) - self._CHECKED_KEYWORDS - self._ANNOTATION_KEYWORDS
        )
        if unchecked:
            raise ValueError(
                f"Schema keywords {unchecked} at {where} need the jsonschema "
                "package; the built-in validator does not check them"
            )
        for keyword in ("properties", "$defs", "definitions"):
            for name, sub in schema.get(keyword, {}).items():
                self._require_checked_keywords(sub, f"{where}/{keyword}/{name}")
        for keyword in ("anyOf", "oneOf", "allOf"):
            for idx, sub in enumerate(schema.get(keyword, ())):
                self._require_checked_keywords(sub, f"{where}/{keyword}/{idx}")
        if "items" in schema:
            self._require_checked_keywords(schema["items"], f"{where}/items")
        if isinstance(schema.get("additionalProperties"), Mapping):
            self._require_checked_keywords(
                schema["additionalProperties"], f"{where}/additionalProperties"
            )

    def validate(
        self,
        completion_text: str,
        finish_reason: str | None,
        *,
        allow_truncated: bool = False,
    ) -> ValidationResult:
        non_truncated = (finish_reason != "length") or allow_truncated
        if not non_truncated:
            return ValidationResult(
                is_valid=False,
                json_parse_ok=False,
                schema_ok=False,
                non_truncated=False,
                error_message="Completion hit max_tokens (finish_reason=length)",
            )

        try:
            parsed = json.loads(completion_text)
        except json.JSONDecodeError as err:
            return ValidationResult(
                is_valid=False,
                json_parse_ok=False,
                schema_ok=False,
                non_truncated=True,
                error_message=f"JSONDecodeError: {err}",
            )

        if self._external_validator is not None:
            try:
                self._external_validator.validate(parsed)
                return ValidationResult(
                    is_valid=True,
                    json_parse_ok=True,
                    schema_ok=True,
                    non_truncated=True,
                )
            except Exception as err:
                return ValidationResult(
                    is_valid=False,
                    json_parse_ok=True,
                    schema_ok=False,
                    non_truncated=True,
                    error_message=str(err),
                )

        try:
            self._validate_node(parsed, self._root_schema)
            return ValidationResult(
                is_valid=True,
                json_parse_ok=True,
                schema_ok=True,
                non_truncated=True,
            )
        except ValueError as err:
            return ValidationResult(
                is_valid=False,
                json_parse_ok=True,
                schema_ok=False,
                non_truncated=True,
                error_message=str(err),
            )

    def _resolve_ref(self, ref: str) -> dict[str, Any]:
        if not ref.startswith("#/"):
            raise ValueError(f"Unsupported non-local $ref '{ref}'")
        curr: Any = self._root_schema
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(curr, Mapping) or part not in curr:
                raise ValueError(f"Unresolvable $ref '{ref}' at '{part}'")
            curr = curr[part]
        if not isinstance(curr, Mapping):
            raise ValueError(f"$ref '{ref}' did not resolve to a schema object")
        return dict(curr)

    def _validate_node(self, instance: Any, schema: Mapping[str, Any]) -> None:
        if "$ref" in schema:
            self._validate_node(instance, self._resolve_ref(str(schema["$ref"])))
            return

        if "anyOf" in schema and not any(
            self._matches(instance, sub) for sub in schema["anyOf"]
        ):
            raise ValueError(f"Instance {instance!r} matched none of anyOf")

        if "oneOf" in schema:
            matches = sum(1 for sub in schema["oneOf"] if self._matches(instance, sub))
            if matches != 1:
                raise ValueError(
                    f"Instance {instance!r} matched {matches} oneOf branches"
                )

        if "allOf" in schema:
            for sub in schema["allOf"]:
                self._validate_node(instance, sub)

        if "enum" in schema and instance not in schema["enum"]:
            raise ValueError(f"Value {instance!r} not in enum {schema['enum']}")

        if "const" in schema and instance != schema["const"]:
            raise ValueError(f"Value {instance!r} != const {schema['const']!r}")

        if "type" in schema:
            raw_type = schema["type"]
            allowed_types = [raw_type] if isinstance(raw_type, str) else list(raw_type)
            if not any(_is_json_type(instance, t) for t in allowed_types):
                actual_type = type(instance).__name__
                raise ValueError(
                    f"Expected JSON type {allowed_types}, got {actual_type}"
                )

        if isinstance(instance, str):
            if "minLength" in schema and len(instance) < int(schema["minLength"]):
                raise ValueError("String shorter than minLength")
            if "maxLength" in schema and len(instance) > int(schema["maxLength"]):
                raise ValueError("String longer than maxLength")
            if (
                "pattern" in schema
                and re.search(str(schema["pattern"]), instance) is None
            ):
                raise ValueError(
                    f"String {instance!r} failed pattern {schema['pattern']!r}"
                )

        if isinstance(instance, (int, float)) and not isinstance(instance, bool):
            if "minimum" in schema and instance < schema["minimum"]:
                raise ValueError(f"{instance} < minimum {schema['minimum']}")
            if "maximum" in schema and instance > schema["maximum"]:
                raise ValueError(f"{instance} > maximum {schema['maximum']}")

        if isinstance(instance, list):
            if "minItems" in schema and len(instance) < int(schema["minItems"]):
                raise ValueError(
                    f"Array length {len(instance)} < minItems {schema['minItems']}"
                )
            if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
                raise ValueError(
                    f"Array length {len(instance)} > maxItems {schema['maxItems']}"
                )
            items_schema = schema.get("items")
            if isinstance(items_schema, Mapping):
                for idx, elem in enumerate(instance):
                    try:
                        self._validate_node(elem, items_schema)
                    except ValueError as err:
                        raise ValueError(f"Item [{idx}]: {err}") from err

        if isinstance(instance, dict):
            for req_key in schema.get("required", ()):
                if req_key not in instance:
                    raise ValueError(f"Missing required property '{req_key}'")
            props = schema.get("properties", {})
            add_props = schema.get("additionalProperties", True)
            for key, val in instance.items():
                if key in props:
                    try:
                        self._validate_node(val, props[key])
                    except ValueError as err:
                        raise ValueError(f"Property '{key}': {err}") from err
                elif add_props is False:
                    raise ValueError(f"Disallowed additional property '{key}'")
                elif isinstance(add_props, Mapping):
                    self._validate_node(val, add_props)

    def _matches(self, instance: Any, sub_schema: Mapping[str, Any]) -> bool:
        try:
            self._validate_node(instance, sub_schema)
            return True
        except ValueError:
            return False


def _is_json_type(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "null":
        return instance is None
    return False


def normalize_api_root(base_url: str) -> str:
    """Return the URL with a default `http://` scheme and any trailing `/v1` stripped.

    A path prefix before `/v1` is kept, for endpoints served behind a gateway.
    """
    cleaned = base_url.strip()
    if "://" not in cleaned:
        cleaned = f"http://{cleaned}"
    cleaned = cleaned.rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3]
    return cleaned


def auto_detect_model(base_url: str, specified_model: str | None = None) -> str:
    """Resolve model name from `--model` or `GET <url>/v1/models`."""
    if specified_model and specified_model.strip():
        return specified_model.strip()

    models_url = f"{normalize_api_root(base_url)}/v1/models"
    req = urllib.request.Request(models_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as err:
        raise RuntimeError(
            f"Failed to query {models_url} to resolve served model: {err}. "
            "Pass --model explicitly if the endpoint disables /v1/models."
        ) from err

    entries = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(
            f"No models returned at {models_url}; pass --model explicitly."
        )
    if len(entries) > 1:
        ids = [m.get("id") for m in entries if isinstance(m, Mapping)]
        raise RuntimeError(
            f"Multiple models served at {models_url} ({ids}); pass --model explicitly."
        )
    model_id = entries[0].get("id") if isinstance(entries[0], Mapping) else None
    if not model_id:
        raise RuntimeError(f"Malformed /v1/models entry at {models_url}: {entries[0]}")
    return str(model_id)


def build_request_payload(
    *,
    model: str,
    case: SchemaCase,
    sample: PromptSample,
    config: TaskConfig,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if sample.system_prompt:
        messages.append({"role": "system", "content": sample.system_prompt})
    messages.append({"role": "user", "content": sample.prompt})

    mode = case.request_mode or config.request_mode
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": case.max_tokens or config.max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": config.enable_thinking,
            "thinking": config.enable_thinking,
        },
    }
    if mode == RequestMode.RESPONSE_FORMAT:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": re.sub(r"[^a-zA-Z0-9_-]", "_", case.name),
                "strict": True,
                "schema": case.schema,
            },
        }
    else:
        payload["structured_outputs"] = {"json": case.schema}
    return payload


def _run_one_trial(
    *,
    chat_url: str,
    model: str,
    case: SchemaCase,
    validator: SchemaValidator,
    sample: PromptSample,
    run_idx: int,
    config: TaskConfig,
) -> SampleTrialResult:
    payload = build_request_payload(
        model=model, case=case, sample=sample, config=config
    )
    req = urllib.request.Request(
        chat_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_seconds) as resp:
            raw_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace").strip()
        if 400 <= err.code < 500:
            # An HTTP 4xx response (for example the server rejecting a schema)
            # is a server correctness outcome: record it as a failed trial with
            # the status and body snippet so the suite finishes and reports it.
            return SampleTrialResult(
                case_name=case.name,
                sample_id=sample.sample_id,
                run_idx=run_idx,
                validation=ValidationResult(
                    is_valid=False,
                    json_parse_ok=False,
                    schema_ok=False,
                    non_truncated=True,
                    error_message=f"HTTP {err.code}: {detail[:500]}",
                ),
                finish_reason=None,
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )
        raise RuntimeError(
            f"HTTP {err.code} from {chat_url} on case '{case.name}' "
            f"sample '{sample.sample_id}': {detail[:500]}"
        ) from err
    except Exception as err:
        raise RuntimeError(
            f"Request to {chat_url} failed on case '{case.name}' "
            f"sample '{sample.sample_id}': {err}"
        ) from err

    latency_ms = (time.monotonic() - t0) * 1000.0
    body = json.loads(raw_body)
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"Empty choices in response from {chat_url}: {body}")
    choice0 = choices[0]
    content = (choice0.get("message") or {}).get("content") or ""
    finish_reason = choice0.get("finish_reason")
    verdict = validator.validate(
        content, finish_reason, allow_truncated=config.allow_truncated
    )
    return SampleTrialResult(
        case_name=case.name,
        sample_id=sample.sample_id,
        run_idx=run_idx,
        validation=verdict,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
    )


def compute_accuracy_and_stderr(
    trials: Sequence[SampleTrialResult],
    num_runs: int,
    *,
    deterministic: bool = True,
) -> tuple[float, float, float, float]:
    total = len(trials)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    valid = sum(1 for t in trials if t.validation.is_valid)
    parsed = sum(1 for t in trials if t.validation.json_parse_ok)
    non_trunc = sum(1 for t in trials if t.validation.non_truncated)

    acc = valid / total
    parse_rate = parsed / total
    non_trunc_rate = non_trunc / total

    # Grammar-constrained schema validity is a deterministic invariant rather
    # than a noisy multiple-choice sample: report stderr=0.0 so a single broken
    # schema completion fails a strict tolerance gate instead of inflating a
    # binomial confidence band.
    if deterministic:
        return acc, 0.0, parse_rate, non_trunc_rate

    if num_runs > 1:
        run_means = []
        for r_idx in range(num_runs):
            r_trials = [t for t in trials if t.run_idx == r_idx]
            if r_trials:
                run_means.append(
                    sum(1 for t in r_trials if t.validation.is_valid) / len(r_trials)
                )
        if len(run_means) > 1:
            mean = sum(run_means) / len(run_means)
            var = sum((x - mean) ** 2 for x in run_means) / (len(run_means) - 1)
            return acc, math.sqrt(var / len(run_means)), parse_rate, non_trunc_rate

    stderr = math.sqrt(acc * (1.0 - acc) / total) if 0.0 < acc < 1.0 else 0.0
    return acc, stderr, parse_rate, non_trunc_rate


def evaluate_json_schema(
    base_url: str,
    task_config: TaskConfig,
    model: str | None = None,
) -> SuiteResult:
    """Run all configured JSON Schema cases against `base_url` and aggregate."""
    resolved_model = auto_detect_model(base_url, model)
    chat_url = f"{normalize_api_root(base_url)}/v1/chat/completions"
    validators = {c.name: SchemaValidator(c.schema) for c in task_config.cases}

    work_items: list[tuple[SchemaCase, PromptSample, int]] = [
        (case, sample, run_idx)
        for run_idx in range(task_config.num_runs)
        for case in task_config.cases
        for sample in case.samples
    ]

    results: list[SampleTrialResult] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=task_config.concurrency
    ) as pool:
        future_map = [
            pool.submit(
                _run_one_trial,
                chat_url=chat_url,
                model=resolved_model,
                case=case,
                validator=validators[case.name],
                sample=sample,
                run_idx=run_idx,
                config=task_config,
            )
            for case, sample, run_idx in work_items
        ]
        for fut in future_map:
            results.append(fut.result())

    acc, stderr, parse_rate, non_trunc_rate = compute_accuracy_and_stderr(
        results,
        task_config.num_runs,
        deterministic=task_config.deterministic,
    )
    per_case: dict[str, float] = {}
    for case in task_config.cases:
        c_trials = [t for t in results if t.case_name == case.name]
        if c_trials:
            per_case[case.name] = sum(
                1 for t in c_trials if t.validation.is_valid
            ) / len(c_trials)

    return SuiteResult(
        task_name=task_config.task_name,
        model_name=resolved_model,
        total_trials=len(results),
        successful_trials=sum(1 for t in results if t.validation.is_valid),
        accuracy=acc,
        accuracy_stderr=stderr,
        json_parse_rate=parse_rate,
        non_truncated_rate=non_trunc_rate,
        per_case_accuracy=per_case,
        trials=tuple(results),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OpenAI-compatible structured outputs against JSON Schemas."
        )
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible server base URL (e.g. http://localhost:8000).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Served model name. Defaults to the single model at /v1/models.",
    )
    parser.add_argument(
        "--task-config",
        required=True,
        type=Path,
        help="Path to YAML or JSON task configuration file.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Output path for check_regression.py compatible results_<task>.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = TaskConfig.from_yaml_file(args.task_config)
    try:
        summary = evaluate_json_schema(
            base_url=args.base_url,
            task_config=config,
            model=args.model,
        )
    except (RuntimeError, ValueError) as err:
        # No results file: the endpoint could not be reached, or a schema
        # cannot be checked here. The caller fails the suite on the exit code.
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    summary.write_regression_json(args.output_json)
    print(
        f"[{summary.task_name}] model={summary.model_name} "
        f"acc={summary.accuracy:.4f} +/- {summary.accuracy_stderr:.4f} "
        f"({summary.successful_trials}/{summary.total_trials}) -> {args.output_json}"
    )
    for idx, t in enumerate(summary.trials):
        if not t.validation.is_valid and idx < 10:
            err_msg = t.validation.error_message
            print(
                f"  INVALID [{t.case_name}#{t.sample_id}]: {err_msg}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
