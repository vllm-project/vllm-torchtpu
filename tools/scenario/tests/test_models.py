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
from scenario.model import baseline, server, sweep, tasks
from scenario.model import scenario as top
from scenario.model.common import Model
from scenario.model.rules import check_scenario

MINIMAL: dict[type[Model], dict[str, Any]] = {
    server.ModelSpec: {"name": "m"},
    server.Kv: {"connector": "TPURaidenConnector", "role": "producer"},
    server.Role: {"slice": "tpu7x/2x2x1"},
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
    sweep.SweepPoint: {"name": "p"},
    sweep.Sweep: {},
    top.Scenario: {
        "schema": 1,
        "name": "n",
        "server": {
            "image": "ci",
            "model": {"name": "m"},
            "slice": "tpu7x/2x2x1",
            "startup_timeout": 60,
        },
    },
    top.DefaultsServer: {},
    top.DefaultsPerf: {},
    top.DefaultsAgentxLabels: {},
    top.DefaultsAgentx: {},
    top.DefaultsLmEval: {},
    top.DefaultsEval: {},
    top.DefaultsTasks: {},
    top.Defaults: {"schema": 1},
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


def test_json_schemas_generate() -> None:
    for model in (top.Scenario, top.Defaults):
        schema = model.model_json_schema(by_alias=True)
        assert "schema" in schema["properties"]


def test_only_yaml_spellings() -> None:
    data = {**MINIMAL[top.Scenario]}
    data["schema_version"] = data.pop("schema")
    with pytest.raises(ValidationError):
        top.Scenario.model_validate(data)
    ref = baseline.BaselineRef.model_validate({"from": "a/b"})
    assert ref.model_dump() == {"from": "a/b"}


def test_strict_values() -> None:
    perf = {**MINIMAL[tasks.PerfTask], "num_prompts": "8"}
    with pytest.raises(ValidationError):
        tasks.PerfTask.model_validate(perf)


@pytest.mark.parametrize("fixture", ["single", "pd", "agentx"])
def test_fixtures_are_valid(fixture: str, request: pytest.FixtureRequest) -> None:
    model = top.Scenario.model_validate(request.getfixturevalue(fixture))
    assert check_scenario(model) == []


def test_durations_and_sizes_are_stored_as_numbers(single: dict[str, Any]) -> None:
    single["server"]["scratch"] = "64Gi"
    model = top.Scenario.model_validate(single)
    assert model.server.startup_timeout == 7200
    assert model.server.scratch == 64 * 2**30


def test_role_views_combine_server_and_role(pd: dict[str, Any]) -> None:
    pd["server"]["args"]["compilation-config"] = {"a": 1, "b": 2}
    pd["server"]["roles"]["decode"]["args"]["compilation-config"] = {"b": None}
    pd["server"]["roles"]["decode"]["args"]["enable-expert-parallel"] = None
    pd["server"]["roles"]["decode"]["env"] = {"TPU_KV_SHM_POOL_GB": None}
    prefill, decode = top.Scenario.model_validate(pd).server.role_views()
    assert prefill.args["compilation-config"] == {"a": 1, "b": 2}
    assert decode.args["compilation-config"] == {"a": 1}
    assert "enable-expert-parallel" not in decode.args
    assert decode.args["data-parallel-size"] == 8
    assert decode.env == {"JAX_PLATFORMS": "tpu,cpu"}
    assert prefill.env["TPU_KV_SHM_POOL_GB"] == "8"
    assert decode.arg_path("data-parallel-size") == (
        "server",
        "roles",
        "decode",
        "args",
        "data-parallel-size",
    )
    assert decode.arg_path("max-model-len") == ("server", "args", "max-model-len")
    assert (decode.shape.hosts, decode.shape.chips) == (1, 4)


def test_single_role_view(single: dict[str, Any]) -> None:
    (role,) = top.Scenario.model_validate(single).server.role_views()
    assert role.name == "server"
    assert role.path == ("server",)
    assert role.shape.hosts == 2


def test_pd_cells_expand_in_order(pd: dict[str, Any]) -> None:
    perf = top.Scenario.model_validate(pd).tasks.perf
    assert perf is not None
    keys = [cell.key for cell in perf.cell_list()]
    assert len(keys) == 17
    assert keys[:3] == [
        "pd_disagg_decode_smoke_c8_i8192_o32",
        "pd_disagg_prefill_c8_i8192_o1",
        "pd_disagg_prefill_c16_i8192_o1",
    ]
    assert keys[-3:] == [
        f"pd_disagg_decode_daily_round{r}_c256_i65536_o1024" for r in (1, 2, 3)
    ]
    first = perf.cell_list()[0]
    assert (first.isl, first.osl, first.concurrency, first.num_prompts) == (
        8192,
        32,
        8,
        8,
    )


def test_product_cells(single: dict[str, Any]) -> None:
    single["tasks"]["perf"]["isl_osl"] = [[1024, 128], [2048, 256]]
    single["tasks"]["perf"]["concurrency"] = [1, 8]
    single["tasks"]["perf"]["num_prompts"] = {"per_concurrency": 10}
    perf = top.Scenario.model_validate(single).tasks.perf
    assert perf is not None
    cells = perf.cell_list()
    assert [cell.key for cell in cells] == [
        "isl1024_osl128_c1",
        "isl1024_osl128_c8",
        "isl2048_osl256_c1",
        "isl2048_osl256_c8",
    ]
    assert tasks.count_for(cells[1].num_prompts, cells[1].concurrency) == 80
    assert cells[2].path == ("tasks", "perf", "isl_osl", 1)


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


def test_schema_version() -> None:
    for version in (2, "1", True):
        data = {**MINIMAL[top.Scenario], "schema": version}
        with pytest.raises(ValidationError) as info:
            top.Scenario.model_validate(data)
        (error,) = info.value.errors()
        assert error["type"] == "V-01"
        assert "this tool reads schema 1" in error["msg"]


def test_defaults_file_shape() -> None:
    defaults = top.Defaults.model_validate(
        {
            "schema": 1,
            "server": {
                "image": "ci",
                "startup_timeout": "2h",
                "args": {"max-model-len": 16384, "api-server-count": 1},
            },
            "tasks": {
                "perf": {
                    "dataset": {"name": "random"},
                    "num_prompts": 320,
                    "range_ratio_style": "symmetric",
                    "warmup_runs": 0,
                    "percentile_metrics": ["ttft", "tpot", "itl", "e2el"],
                    "tolerance": 0.05,
                    "client_args": {"backend": "vllm", "seed": 42},
                },
                "agentx": {"labels": {"framework": "vllm", "spec_decoding": "none"}},
                "eval": {
                    "lm_eval": {
                        "model": "local-chat-completions",
                        "model_args": {"num_concurrent": 128},
                        "seed": "0,1234,None,1234",
                        "limit": 100,
                    }
                },
            },
        }
    )
    assert defaults.server is not None
    assert defaults.server.startup_timeout == 7200
    for misplaced in (
        {"server": {"model": {"name": "m"}}},
        {"server": {"slice": "tpu7x/2x2x1"}},
        {"tasks": {"perf": {"baseline": {}}}},
        {"tasks": {"perf": {"isl_osl": [[1, 1]]}}},
        {"name": "x"},
        {"sweep": {}},
    ):
        with pytest.raises(ValidationError):
            top.Defaults.model_validate({"schema": 1, **misplaced})


def test_defaults_fields_share_descriptions() -> None:
    assert (
        top.DefaultsServer.model_fields["startup_timeout"].description
        == server.Server.model_fields["startup_timeout"].description
    )


def test_sweep_expansion(agentx: dict[str, Any]) -> None:
    model = sweep.Sweep.model_validate(agentx["sweep"])
    points = sweep.expand(model)
    assert [point.name for point in points] == ["c1", "c2", "c4", "c8"]
    assert points[3].assignments == (
        (("tasks", "agentx", "concurrency"), 8),
        (("server", "args", "max-num-seqs"), 32),
    )
    product = sweep.Sweep.model_validate(
        {
            "name": "tp{server.args.tensor-parallel-size}-c{tasks.perf.concurrency[0]}",
            "product": {
                "server.args.tensor-parallel-size": [4, 8],
                "tasks.perf.concurrency[0]": [1, 2],
            },
        }
    )
    assert [point.name for point in sweep.expand(product)] == [
        "tp4-c1",
        "tp4-c2",
        "tp8-c1",
        "tp8-c2",
    ]
    named = sweep.Sweep.model_validate({"points": [{"name": "a"}, {"name": "b"}]})
    assert [(point.name, point.index) for point in sweep.expand(named)] == [
        ("a", 0),
        ("b", 1),
    ]


def test_point_name_errors() -> None:
    with pytest.raises(ValueError, match="is not one of the axes"):
        sweep.point_name("c{tasks.x}", {("tasks", "y"): 1})
    with pytest.raises(ValueError, match="cannot go into a point name"):
        sweep.point_name("c{tasks.y}", {("tasks", "y"): [1]})
    assert sweep.point_name("x{tasks.y}", {("tasks", "y"): True}) == "xtrue"
