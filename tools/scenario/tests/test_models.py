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
"""The models: what each accepts, what it rejects, and what it derives."""

import importlib
import inspect
import pkgutil
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from scenario.model import baseline, server, tasks
from scenario.model.common import Model

MINIMAL: dict[type[Model], dict[str, Any]] = {
    server.ModelSpec: {"name": "m"},
    server.Kv: {"connector": "TPURaidenConnector", "role": "producer"},
    server.Role: {
        "accelerator": "tpu7x",
        "topology": "2x2x1",
        "hosts": 1,
        "chips_per_host": 4,
    },
    server.Router: {"type": "toy-proxy", "command": ["python3", "proxy.py"]},
    server.Server: {"image": "ci", "model": {"name": "m"}, "startup_timeout": 60},
    tasks.PerConcurrency: {"per_concurrency": 2},
    tasks.Dataset: {"name": "random"},
    tasks.PerfCellGroup: {"isl": 1, "osl": 1, "concurrency": 1},
    tasks.PerfTask: {
        "dataset": {"name": "random"},
        "num_prompts": 1,
        "warmup_runs": 0,
        "percentile_metrics": ["ttft"],
        "tolerance": 0.05,
    },
    tasks.AgentxLabels: {
        "model_prefix": "m",
        "precision": "fp8",
        "framework": "vllm",
        "spec_decoding": "none",
    },
    tasks.AgentxTask: {
        "concurrency": 1,
        "duration": "1h",
        "timeout": "2h",
        "labels": {
            "model_prefix": "m",
            "precision": "fp8",
            "framework": "vllm",
            "spec_decoding": "none",
        },
        "tolerance": 0.05,
    },
    tasks.LmEvalTask: {"name": "gsm8k"},
    tasks.LmEval: {
        "tasks": ["gsm8k"],
        "model": "local-chat-completions",
        "seed": "0",
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
    },
    tasks.EvalTask: {
        "lm_eval": {
            "tasks": ["gsm8k"],
            "model": "local-chat-completions",
            "seed": "0",
            "apply_chat_template": True,
            "fewshot_as_multiturn": True,
        },
        "tolerance": 0.02,
    },
    tasks.CustomTask: {"name": "c", "command": ["true"], "timeout": 60},
    tasks.Tasks: {},
    baseline.PerfCellBaseline: {"completed": 1},
    baseline.PerfBaseline: {},
    baseline.EvalBaseline: {},
    baseline.AgentxBaseline: {},
    baseline.BaselineRef: {"from": "a/b"},
    baseline.PointBaselines: {},
}

# Keyed baselines take any key as an entry, so a stray key is a bad entry.
KEYED = {baseline.PerfBaseline, baseline.EvalBaseline, baseline.AgentxBaseline}


def all_models() -> list[type[BaseModel]]:
    """Every model class defined in scenario.model, but not the base class."""
    package = importlib.import_module("scenario.model")
    found = set()
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"scenario.model.{info.name}")
        for _, value in inspect.getmembers(module, inspect.isclass):
            if issubclass(value, Model) and value.__module__ == module.__name__:
                found.add(value)
    found.discard(Model)
    return sorted(found, key=lambda model: model.__name__)


def test_every_model_has_a_minimal_case() -> None:
    public = {model for model in all_models() if not model.__name__.startswith("_")}
    assert public == set(MINIMAL)


@pytest.mark.parametrize("model", list(MINIMAL), ids=lambda model: model.__name__)
def test_minimal_valid(model: type[Model]) -> None:
    model.model_validate(MINIMAL[model])


@pytest.mark.parametrize(
    "model", [m for m in MINIMAL if m not in KEYED], ids=lambda model: model.__name__
)
def test_unknown_key_fails(model: type[Model]) -> None:
    with pytest.raises(ValidationError) as info:
        model.model_validate({**MINIMAL[model], "bogus": 1})
    assert [error["type"] for error in info.value.errors()] == ["extra_forbidden"]


@pytest.mark.parametrize("model", all_models(), ids=lambda model: model.__name__)
def test_every_field_has_a_description(model: type[BaseModel]) -> None:
    for name, field in model.model_fields.items():
        assert field.description, f"{model.__name__}.{name} has no description"


def test_models_are_frozen() -> None:
    spec = server.ModelSpec.model_validate({"name": "m"})
    with pytest.raises(ValidationError):
        spec.name = "x"  # pyrefly: ignore


def test_strict_values() -> None:
    perf = {**MINIMAL[tasks.PerfTask], "num_prompts": "8"}
    with pytest.raises(ValidationError):
        tasks.PerfTask.model_validate(perf)


def test_lm_eval_items_and_run_keys() -> None:
    lm_eval = tasks.LmEval.model_validate(
        {
            **MINIMAL[tasks.LmEval],
            "concurrency": [1, 4],
            "tasks": [
                {"name": "gsm8k", "limit": 128, "limit_per_concurrency": 2},
                "mmlu_pro",
            ],
        }
    )
    assert [item.name for item in lm_eval.task_items()] == ["gsm8k", "mmlu_pro"]
    assert lm_eval.run_keys() == (
        "gsm8k_c1",
        "gsm8k_c4",
        "mmlu_pro_c1",
        "mmlu_pro_c4",
    )


def test_lm_eval_limit_all_and_model_args() -> None:
    assert tasks.LmEval.model_validate({**MINIMAL[tasks.LmEval], "limit": "all"})
    with pytest.raises(ValidationError, match="cannot hold"):
        tasks.LmEval.model_validate(
            {**MINIMAL[tasks.LmEval], "model_args": {"x": "a,b"}}
        )
    with pytest.raises(ValidationError):
        tasks.LmEval.model_validate({**MINIMAL[tasks.LmEval], "limit": "some"})


def test_baselines() -> None:
    perf = tasks.PerfTask.model_validate(
        {
            **MINIMAL[tasks.PerfTask],
            "baseline": {
                "_source": {"note": "from nightly"},
                "isl1_osl1_c1": {"completed": 4, "median_ttft_ms": 1.5},
            },
        }
    )
    assert isinstance(perf.baseline, baseline.PerfBaseline)
    assert perf.baseline.model_extra == {
        "isl1_osl1_c1": baseline.PerfCellBaseline(completed=4, median_ttft_ms=1.5)
    }
    ref = tasks.PerfTask.model_validate(
        {**MINIMAL[tasks.PerfTask], "baseline": {"from": "glm-5.2/fp8-dp16-ep"}}
    )
    assert isinstance(ref.baseline, baseline.BaselineRef)


@pytest.mark.parametrize(
    "bad",
    [
        {"isl1_osl1_c1": {"median_tpot": 1}},
        {"isl1_osl1_c1": {}},
        {"_note": {"completed": 1}},
        {"from": "a/b", "isl1_osl1_c1": {"completed": 1}},
    ],
)
def test_bad_perf_baselines(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        tasks.PerfTask.model_validate({**MINIMAL[tasks.PerfTask], "baseline": bad})


def test_eval_baseline_metric_keys() -> None:
    good = {"mmlu_pro": {"exact_match,custom-extract": 0.8}}
    assert baseline.EvalBaseline.model_validate(good).model_extra == good
    with pytest.raises(ValidationError, match="<metric>,<filter>"):
        baseline.EvalBaseline.model_validate({"mmlu_pro": {"exact_match": 0.8}})


def test_agentx_concurrency_is_one_value() -> None:
    data = {**MINIMAL[tasks.AgentxTask], "concurrency": [1, 2]}
    with pytest.raises(ValidationError) as info:
        tasks.AgentxTask.model_validate(data)
    (error,) = info.value.errors()
    assert error["type"] == "V-31"
    assert "one concurrency per fresh server" in error["msg"]
