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
"""One case per validation rule message, on scenarios that are otherwise valid."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from scenario.errors import Issue
from scenario.model.rules import RULES, check_scenario, check_sweep, rule_table
from scenario.model.scenario import Scenario
from scenario.model.sweep import Sweep
from scenario.paths import parse_path

Data = dict[str, Any]
DROP = object()


def put(data: Data, text: str, value: object) -> None:
    *parents, last = parse_path(text)
    node: Any = data
    for key in parents:
        if isinstance(key, str) and key not in node:
            node[key] = {}
        node = node[key]
    if value is DROP:
        del node[last]
    else:
        node[last] = value


@dataclass(frozen=True)
class Case:
    rule: str
    fixture: str
    edits: dict[str, object]
    path: str
    message: str
    hint: str | None = None
    severity: str = "error"
    extra: Callable[[Data], None] | None = field(default=None, compare=False)


def apply(case: Case, data: Data) -> Data:
    for text, value in case.edits.items():
        put(data, text, value)
    if case.extra is not None:
        case.extra(data)
    return data


def _second_custom(data: Data) -> None:
    data["tasks"]["custom"].append(dict(data["tasks"]["custom"][0]))


def _agentx_with_custom(data: Data) -> None:
    data["tasks"]["custom"] = [{"name": "probe", "command": ["true"], "timeout": 60}]


AGENTX_ZIP = "sweep.zip"
LM = "tasks.eval.lm_eval"
CELLS = "tasks.perf.cells"
CUSTOM_ENV = "tasks.custom[0].env"

CASES = [
    Case(
        "V-02",
        "single",
        {"server.slice": DROP},
        "server",
        "set server.slice for one role",
    ),
    Case(
        "V-02",
        "pd",
        {"server.slice": "tpu7x/2x2x1"},
        "server.roles",
        "server.slice is set too",
        hint='add "slice: null" under server to drop an inherited slice',
    ),
    Case(
        "V-03",
        "pd",
        {"server.roles.decode": DROP},
        "server.roles",
        "has one role; use server.slice for a single role",
    ),
    Case(
        "V-04",
        "single",
        {"server.multihost": DROP},
        "server.slice",
        "tpu7x/2x2x2 has 2 hosts; set server.multihost: mp or ray",
    ),
    Case(
        "V-04",
        "pd",
        {"server.multihost": "mp"},
        "server.multihost",
        "is set but every slice is single-host (tpu7x/2x2x1); remove it",
    ),
    Case(
        "V-05",
        "single",
        {"server.router": {"type": "toy-proxy", "command": ["p"]}},
        "server.router",
        "needs server.roles",
    ),
    Case(
        "V-06",
        "pd",
        {"server.roles.decode.kv.role": "producer"},
        "server.router.type",
        "toy-proxy needs one producer and one consumer role; found producers "
        "[prefill, decode] and consumers []",
    ),
    Case(
        "V-08",
        "pd",
        {"tasks.driver": DROP},
        "tasks",
        "tasks.driver is required with server.roles (one of: prefill, decode, client)",
    ),
    Case(
        "V-08", "pd", {"tasks.driver": "router"}, "tasks.driver", "router is not a role"
    ),
    Case(
        "V-08",
        "single",
        {"tasks.driver": "client"},
        "tasks.driver",
        "a scenario without roles runs tasks on its first host",
    ),
    Case(
        "V-09",
        "single",
        {"server.args.load-format": DROP},
        "server.model.path",
        "is object storage; set load-format: runai_streamer (no other loader",
    ),
    Case(
        "V-10",
        "single",
        {"server.args.distributed-executor-backend": "mp"},
        "server.args.distributed-executor-backend",
        "is mp but server.multihost is ray",
    ),
    Case(
        "V-11",
        "single",
        {"server.env.TPU_TOPOLOGY": "2x2x2"},
        "server.env.TPU_TOPOLOGY",
        "TPU_TOPOLOGY is set by the tool (multi-host and backend wiring); remove it",
    ),
    Case(
        "V-11",
        "pd",
        {"server.roles.decode.env.SCENARIO_HOST_INDEX": "1"},
        "server.roles.decode.env.SCENARIO_HOST_INDEX",
        "SCENARIO_HOST_INDEX is set by the tool (SCENARIO_* names are reserved)",
    ),
    Case(
        "V-11",
        "pd",
        {f"{CUSTOM_ENV}.VLLM_PLUGINS": ""},
        f"{CUSTOM_ENV}.VLLM_PLUGINS",
        "VLLM_PLUGINS is set by the tool on a client driver",
    ),
    Case(
        "V-11",
        "agentx",
        {"tasks.agentx.env.CONC": "4"},
        "tasks.agentx.env.CONC",
        "CONC is set by the tool from tasks.agentx.concurrency; remove it",
    ),
    Case(
        "V-12",
        "single",
        {"server.env.VLLM_ENGINE_READY_TIMEOUT_S": "1800"},
        "server.env.VLLM_ENGINE_READY_TIMEOUT_S",
        "VLLM_ENGINE_READY_TIMEOUT_S is set from server.startup_timeout; remove it",
        hint="set server.startup_timeout instead (now 2h)",
    ),
    Case(
        "V-13",
        "single",
        {"server.env.JAX_PLATFORMS": "tpu"},
        "server.env.JAX_PLATFORMS",
        "JAX_PLATFORMS is set by the ray backend; remove it",
    ),
    Case(
        "V-14",
        "single",
        {"server.args.data-parallel-size-local": 8},
        "server.args.data-parallel-size-local",
        "is set by the tool (data-parallel-size over hosts); remove it",
    ),
    Case(
        "V-14",
        "pd",
        {"server.router.args.port": 9000},
        "server.router.args.port",
        "is set by the tool (router wiring); remove it",
    ),
    Case(
        "V-14",
        "single",
        {"tasks.perf.client_args.num-prompts": 8},
        "tasks.perf.client_args.num-prompts",
        "is set by tasks.perf.num_prompts; remove it",
    ),
    Case(
        "V-16",
        "single",
        {"server.args.data-parallel-size": 24},
        "server.args",
        "TP 1 x DP 24 = 24 devices, but tpu7x/2x2x2 has 16 (2 hosts x 8)",
    ),
    Case(
        "V-16",
        "pd",
        {"server.roles.prefill.args.tensor-parallel-size": 2},
        "server.roles.prefill.args",
        "TP 2 x PCP 8 x DP 1 = 16 devices, but tpu7x/2x2x1 has 8 (1 hosts x 8)",
    ),
    Case(
        "V-16",
        "single",
        {"server.args.tensor-parallel-size": "8"},
        "server.args.tensor-parallel-size",
        "must be a positive integer; got '8'",
    ),
    Case(
        "V-17",
        "single",
        {"server.multihost": "mp", "server.args.data-parallel-size": 15},
        "server.args.data-parallel-size",
        "multihost mp needs data-parallel-size divisible by 2 hosts; got 15",
    ),
    Case(
        "V-18",
        "single",
        {CELLS: [{"isl": 1, "osl": 1, "concurrency": 1}]},
        CELLS,
        "tasks.perf sets cells and isl_osl/concurrency; use one",
    ),
    Case(
        "V-18",
        "single",
        {"tasks.perf.concurrency": DROP},
        "tasks.perf",
        "needs isl_osl and concurrency together, or cells",
    ),
    Case(
        "V-19",
        "single",
        {"tasks.perf.isl_osl": [[8192, 2048]]},
        "tasks.perf.isl_osl[0]",
        "cell isl8192_osl2048_c1024 needs 10240 tokens; max-model-len is 9216",
    ),
    Case(
        "V-19",
        "pd",
        {"server.roles.decode.args.max-model-len": 66000},
        f"{CELLS}[4]",
        "cell pd_disagg_decode_daily_round1_c256_i65536_o1024 needs 66560 tokens; "
        "max-model-len is 66000 for role decode",
    ),
    Case(
        "V-20",
        "pd",
        {f"{CELLS}[1].label": "x_c{conc}"},
        f"{CELLS}[1].label",
        "uses {conc}; it may use {isl}, {osl}, {concurrency}, {repeat}",
    ),
    Case(
        "V-20",
        "pd",
        {f"{CELLS}[3].label": "fixed"},
        f"{CELLS}[3].label",
        "cells repeat 3 times but label has no {repeat}",
    ),
    Case(
        "V-20",
        "pd",
        {f"{CELLS}[0].label": "pd_disagg_prefill_c8_i8192_o1"},
        f"{CELLS}[1]",
        "cell pd_disagg_prefill_c8_i8192_o1 appears more than once",
    ),
    Case(
        "V-20",
        "pd",
        {f"{CELLS}[0].label": "a/b"},
        f"{CELLS}[0]",
        "cell key 'a/b' is not filename-safe",
    ),
    Case(
        "V-21",
        "single",
        {"tasks.perf.concurrency": [512]},
        "tasks.perf.baseline",
        "has no entry for isl8192_osl1024_c512 and an extra entry "
        "isl8192_osl1024_c1024",
        hint="its keys are the cell keys: isl8192_osl1024_c512",
    ),
    Case(
        "V-21",
        "single",
        {f"{LM}.tasks": ["mmlu_pro", "gsm8k"]},
        "tasks.eval.baseline",
        "has no entry for gsm8k",
    ),
    Case(
        "V-22",
        "pd",
        {"tasks.custom[0].name": "eval"},
        "tasks.custom[0].name",
        "eval names a built-in task",
    ),
    Case(
        "V-22",
        "pd",
        {},
        "tasks.custom[1].name",
        "another custom task is named correctness",
        extra=_second_custom,
    ),
    Case(
        "V-22",
        "pd",
        {f"{CUSTOM_ENV}.LOG_DIR": "${RESULTS}/x"},
        f"{CUSTOM_ENV}.LOG_DIR",
        "uses ${RESULTS}; known: DECODE_URL, PREFILL_URL, PROXY_URL, "
        "SCENARIO_API_PORT, SCENARIO_HOST_DECODE, SCENARIO_HOST_PREFILL, "
        "SCENARIO_RESULTS_DIR, SCENARIO_ROUTER_PORT, SCENARIO_SERVED_MODEL",
    ),
    Case(
        "V-22",
        "pd",
        {f"{CUSTOM_ENV}.COST": "$5"},
        f"{CUSTOM_ENV}.COST",
        "has a lone $; write $$ for a literal $, or ${NAME}",
    ),
    Case(
        "V-24",
        "agentx",
        {"tasks.agentx.baseline": {"tput": 1.0}},
        "tasks.agentx.baseline",
        "is not allowed with a sweep; put it under sweep.baselines.<point>.<task>",
    ),
    Case(
        "V-24",
        "agentx",
        {f"{AGENTX_ZIP}[tasks.agentx.baseline.x]": [1, 2, 3, 4]},
        "sweep.zip[tasks.agentx.baseline.x]",
        "point baselines go under sweep.baselines",
    ),
    Case(
        "V-25",
        "agentx",
        {f"{AGENTX_ZIP}[server.slice]": ["tpu7x/2x2x4"] * 4},
        "sweep.zip[server.slice]",
        "a sweep point keeps the scenario's image, slices, roles",
    ),
    Case(
        "V-25",
        "agentx",
        {"sweep": {"points": [{"name": "a", "server": {"multihost": "mp"}}]}},
        "sweep.points[0].server.multihost",
        "a sweep point keeps the scenario's image",
    ),
    Case(
        "V-25",
        "agentx",
        {
            "sweep": {
                "points": [
                    {"name": "a", "server": {"roles": {"p": {"slice": "tpu7x/1x1x1"}}}}
                ]
            }
        },
        "sweep.points[0].server.roles.p.slice",
        "a sweep point keeps the scenario's image",
    ),
    Case(
        "V-25",
        "agentx",
        {f"{AGENTX_ZIP}.name": ["a", "b", "c", "d"]},
        "sweep.zip.name",
        "axes override server.* or tasks.* only",
    ),
    Case(
        "V-27",
        "single",
        {f"{LM}.model_args.base_url": "http://x"},
        f"{LM}.model_args.base_url",
        "is set by the tool (the server or router URL); remove it",
    ),
    Case(
        "V-28",
        "single",
        {"tasks.perf.random_range_ratio": DROP},
        "tasks.perf.random_range_ratio",
        "is required with dataset random",
    ),
    Case(
        "V-28",
        "single",
        {
            "tasks.perf.dataset": {"name": "sharegpt"},
            "tasks.perf.range_ratio_style": "min",
        },
        "tasks.perf.range_ratio_style",
        "min applies to the random dataset only",
    ),
    Case(
        "V-32",
        "agentx",
        {"tasks.agentx.concurrency": 32, "sweep": DROP},
        "server.args.max-num-seqs",
        "max-num-seqs 16 x DP 1 = 16, under tasks.agentx.concurrency 32; the "
        "server would cap the measured concurrency",
    ),
    Case(
        "V-32",
        "agentx",
        {"tasks.agentx.concurrency": 12, "sweep": DROP},
        "server.args.max-num-seqs",
        "max-num-seqs 16 x DP 1 = 16, under twice tasks.agentx.concurrency (24)",
        severity="warning",
    ),
    Case(
        "V-33",
        "agentx",
        {f"{AGENTX_ZIP}[server.args.max-num-seqs]": [16, 16, 32]},
        AGENTX_ZIP,
        "axes need equal lengths; tasks.agentx.concurrency has 4, "
        "server.args.max-num-seqs has 3",
    ),
    Case(
        "V-34",
        "agentx",
        {"sweep": DROP},
        "tasks.agentx",
        "must be the only task, but tasks.custom is set too: AgentX needs a fresh "
        "server, and an earlier task would warm it",
        extra=_agentx_with_custom,
    ),
    Case(
        "V-35",
        "single",
        {"tasks.perf.client": "inferencex"},
        "tasks.perf.range_ratio_style",
        "applies to vllm-bench only; inferencex reads random_range_ratio as written",
    ),
    Case(
        "V-35",
        "single",
        {"tasks.perf.range_ratio_style": DROP},
        "tasks.perf.range_ratio_style",
        "is required with client vllm-bench and dataset random: symmetric or min",
    ),
    Case(
        "V-36",
        "single",
        {f"{LM}.tasks": ["mmlu_pro", {"name": "mmlu_pro"}]},
        f"{LM}.tasks[1]",
        "mmlu_pro is listed twice",
    ),
    Case(
        "V-36",
        "single",
        {f"{LM}.concurrency": [1, 2]},
        f"{LM}.model_args.num_concurrent",
        "is set per run from tasks.eval.lm_eval.concurrency; remove it",
    ),
    Case(
        "V-36",
        "single",
        {f"{LM}.concurrency": [1, 1], f"{LM}.model_args.num_concurrent": DROP},
        f"{LM}.concurrency",
        "has repeated values",
    ),
    Case(
        "V-36",
        "single",
        {f"{LM}.tasks": [{"name": "gsm8k", "limit_per_concurrency": 2}]},
        f"{LM}.tasks[0].limit_per_concurrency",
        "needs tasks.eval.lm_eval.concurrency",
    ),
    Case(
        "V-36",
        "single",
        {
            f"{LM}.concurrency": [1, 2],
            f"{LM}.model_args.num_concurrent": DROP,
            f"{LM}.tasks": [
                {"name": "gsm8k", "limit": "all", "limit_per_concurrency": 2}
            ],
        },
        f"{LM}.tasks[0].limit_per_concurrency",
        "raises a limit, but this task runs the full set; set limit",
    ),
    Case(
        "V-36",
        "single",
        {f"{LM}.tasks": [{"name": "mmlu_pro", "subtasks": 14}]},
        f"{LM}.tasks[0].subtasks",
        "only divides limit_per_concurrency",
    ),
    Case(
        "V-37",
        "agentx",
        {"tasks.agentx.timeout": "1h"},
        "tasks.agentx.timeout",
        "must be longer than duration (1h), to cover warmup and drain",
    ),
    Case(
        "V-38",
        "agentx",
        {"sweep": {"name": "x"}},
        "sweep",
        "set exactly one of points, zip and product",
    ),
    Case(
        "V-38",
        "agentx",
        {"sweep.product": {"server.args.max-num-seqs": [16]}},
        "sweep",
        "set exactly one of points, zip and product; found zip, product",
    ),
    Case(
        "V-38",
        "agentx",
        {"sweep.name": DROP, "sweep.baselines": DROP},
        "sweep",
        "name is required with zip, e.g. name: x{tasks.agentx.concurrency}",
    ),
    Case(
        "V-38",
        "agentx",
        {"sweep": {"name": "x", "points": [{"name": "a"}]}},
        "sweep.name",
        "is for zip and product; named points carry their own names",
    ),
    Case(
        "V-38",
        "agentx",
        {"sweep": {"points": [{"name": "a"}], "baselines": {"a": {}}}},
        "sweep.baselines",
        "is for zip and product",
    ),
    Case(
        "V-38",
        "agentx",
        {"sweep": {"points": [{"name": "a"}, {"name": "a"}]}},
        "sweep.points[1].name",
        "another point is named a",
    ),
    Case(
        "V-38",
        "agentx",
        {"sweep.name": "C{tasks.agentx.concurrency}", "sweep.baselines": DROP},
        "sweep.name",
        "gives the point name 'C1'; point names are lowercase letters",
    ),
    Case(
        "V-38",
        "agentx",
        {"sweep.name": "mns{server.args.max-num-seqs}", "sweep.baselines": DROP},
        "sweep.name",
        "gives mns16 to 3 points; use every axis that varies",
    ),
    Case(
        "V-38",
        "agentx",
        {"sweep.baselines.c3": {}},
        "sweep.baselines.c3",
        "no point is named c3; points: c1, c2, c4, c8",
    ),
    Case(
        "V-38",
        "agentx",
        {"sweep.name": "c{tasks.agentx.conc}"},
        "sweep.name",
        "{tasks.agentx.conc} is not one of the axes",
    ),
]


def issues_for(data: Data) -> list[Issue]:
    return check_scenario(Scenario.model_validate(data))


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[f"{case.rule}-{index}" for index, case in enumerate(CASES)],
)
def test_rule_message(case: Case, request: pytest.FixtureRequest) -> None:
    data = apply(case, request.getfixturevalue(case.fixture))
    found = [issue for issue in issues_for(data) if issue.rule == case.rule]
    assert len(found) == 1, found
    (issue,) = found
    assert issue.path == parse_path(case.path)
    assert issue.message.startswith(case.message), issue.message
    assert issue.severity == case.severity
    if case.hint is not None:
        assert issue.hint == case.hint


def test_every_scenario_rule_has_a_case() -> None:
    covered = {case.rule for case in CASES}
    scenario_rules = {rule.id for rule in RULES.values() if rule.stage == "scenario"}
    assert scenario_rules <= covered


def test_rule_table_is_complete_and_ordered() -> None:
    ids = [rule.id for rule in rule_table()]
    assert ids == [f"V-{n:02d}" for n in range(1, 41)]
    assert all(rule.summary for rule in rule_table())


def test_all_problems_reported_together(single: Data) -> None:
    put(single, "server.multihost", "mp")
    put(single, "server.args.data-parallel-size", 15)
    put(single, "server.env.TPU_TOPOLOGY", "x")
    put(single, "tasks.perf.concurrency", [512])
    rules = [issue.rule for issue in issues_for(single)]
    assert rules == ["V-11", "V-17", "V-21"]


def test_provided_names_pass(pd: Data) -> None:
    put(pd, f"{CUSTOM_ENV}.URLS", "${PREFILL_URL},${DECODE_URL},${PROXY_URL}")
    put(pd, f"{CUSTOM_ENV}.HOSTS", "${SCENARIO_HOST_PREFILL}")
    assert issues_for(pd) == []


def test_role_driver_gets_server_log(pd: Data) -> None:
    put(pd, "tasks.driver", "prefill")
    put(pd, f"{CUSTOM_ENV}.LOG", "${SCENARIO_SERVER_LOG}")
    put(pd, f"{CUSTOM_ENV}.VLLM_PLUGINS", "")
    assert issues_for(pd) == []


def test_agentx_capacity_skips_the_producer(pd: Data) -> None:
    pd["tasks"] = {
        "driver": "client",
        "agentx": {
            "concurrency": 64,
            "duration": "1h",
            "timeout": "2h",
            "labels": {
                "model_prefix": "q",
                "precision": "fp8",
                "framework": "vllm",
                "spec_decoding": "none",
            },
            "tolerance": 0.05,
        },
    }
    assert issues_for(pd) == []


def test_check_sweep_on_its_own() -> None:
    sweep = Sweep.model_validate(
        {
            "name": "c{tasks.perf.concurrency[0]}",
            "product": {"tasks.perf.concurrency[0]": [1, 2]},
        }
    )
    assert check_sweep(sweep) == []
    bad = Sweep.model_validate({"zip": {"tasks.x": [1], "tasks.y": [1, 2]}})
    assert [issue.rule for issue in check_sweep(bad)] == ["V-33", "V-38"]
