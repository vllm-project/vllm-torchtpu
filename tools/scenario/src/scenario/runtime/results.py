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
"""The run directory: where it is, and the files the runtime writes there.

Root: the --results-dir flag, else the target's results root, else
./scenario-results/<name>-<UTC stamp>. The run directory is <root>/<name>,
plus /<point> for a sweep point. It holds today's result file names and a
config.json of today's shape, so the existing uploaders read it unchanged,
plus scenario_run.json (what ran) and summary.json (how it went). Hosts
that follow the driver write their summary.json and server.log under
<run dir>/<role>-<index>.
"""

import json
import os
import pathlib
import time
from collections.abc import Mapping
from typing import Any

from scenario import __version__
from scenario.plan import codec
from scenario.plan.types import Plan
from scenario.target import TargetInfo

DEFAULT_ROOT = "scenario-results"
CONFIG = "config.json"
SCENARIO_RUN = "scenario_run.json"
SUMMARY = "summary.json"
SERVER_LOG = "server.log"
ROUTER_LOG = "router.log"


def results_root(
    flag: pathlib.Path | None, target_root: str | None, name: str, stamp: str
) -> pathlib.Path:
    if flag is not None:
        return flag
    if target_root:
        return pathlib.Path(target_root)
    return pathlib.Path(DEFAULT_ROOT) / f"{name}-{stamp}"


def run_dir(root: pathlib.Path, plan: Plan) -> pathlib.Path:
    found = root / plan.scenario_name
    return found / plan.point if plan.point is not None else found


def part_dir(run: pathlib.Path, role: str, index: int) -> pathlib.Path:
    """Where a host that follows the driver keeps its summary and server log.

    The run directory is the driver's; a follower's files go in a directory
    of their own, so hosts that share a filesystem never write the same file.
    """
    return run / f"{role}-{index}"


def write_json(path: pathlib.Path, data: Any) -> None:
    """Write `data` so a reader sees the old file or the new one, never half."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _words(values: list[Any]) -> str:
    return " ".join(str(value) for value in dict.fromkeys(values))


def _listed(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _perf_cells(perf: Mapping[str, Any]) -> tuple[str, str]:
    """isl_osl_configs ("I:O I:O") and concurrency_options ("C C"), in order."""
    if perf.get("cells"):
        pairs, levels = [], []
        for group in perf["cells"]:
            for isl in _listed(group.get("isl")):
                for osl in _listed(group.get("osl")):
                    pairs.append(f"{isl}:{osl}")
            levels += _listed(group.get("concurrency"))
        return _words(pairs), _words(levels)
    pairs = [f"{isl}:{osl}" for isl, osl in perf.get("isl_osl") or []]
    return _words(pairs), _words(perf.get("concurrency") or [])


def _model_args(pairs: Mapping[str, Any]) -> str:
    def text(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    return ",".join(f"{key}={text(value)}" for key, value in pairs.items())


def _gen_kwargs(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _role_args(server: Mapping[str, Any], role: str | None) -> dict[str, Any]:
    """The server args one role runs with: server.args under the role's own."""
    args = dict(server.get("args") or {})
    roles = server.get("roles") or {}
    if roles:
        name = role if role in roles else next(iter(roles))
        for key, value in (roles[name].get("args") or {}).items():
            if value is None:
                args.pop(key, None)
            else:
                args[key] = value
    return args


def config_json(
    plan: Plan, scenario: Mapping[str, Any] | None, timestamp: str
) -> dict[str, Any]:
    """config.json with run_benchmarks.sh's keys, in its order, from the scenario.

    Values the scenario does not set are null; the uploaders read the
    engine values and the model, which every scenario states.
    """
    scenario = scenario or {}
    server = scenario.get("server") or {}
    tasks = scenario.get("tasks") or {}
    args = _role_args(server, plan.driver.role)
    perf = tasks.get("perf") or {}
    evals = tasks.get("eval") or {}
    lm_eval = evals.get("lm_eval") or {}
    isl_osl, concurrency = _perf_cells(perf) if perf else (None, None)
    items = [
        item if isinstance(item, dict) else {"name": item}
        for item in lm_eval.get("tasks") or []
    ]
    multiturn = lm_eval.get("fewshot_as_multiturn")
    mmlu_pro_single_turn = any(
        item["name"] == "mmlu_pro"
        and item.get("fewshot_as_multiturn", multiturn) is False
        for item in items
    )
    async_scheduling = args.get("async-scheduling")
    return {
        "model": plan.model_name,
        "model_uri": plan.model_path if plan.model_path != plan.model_name else None,
        "tensor_parallelism": args.get("tensor-parallel-size", 1),
        "data_parallelism": args.get("data-parallel-size", 1),
        "enable_ep": args.get("enable-expert-parallel") is True,
        "quantization": args.get("quantization") or "none",
        "isl_osl_configs": isl_osl,
        "dataset_name": (perf.get("dataset") or {}).get("name") if perf else None,
        "concurrency_options": concurrency,
        "num_prompts": perf.get("num_prompts") if perf else None,
        "random_range_ratio": perf.get("random_range_ratio"),
        "range_ratio_style": perf.get("range_ratio_style") if perf else None,
        "benchmark_warmup_runs": perf.get("warmup_runs", 0) if perf else None,
        "benchmark_temperature": perf.get("temperature"),
        "benchmark_extra_body": perf.get("extra_body"),
        "max_model_len": args.get("max-model-len"),
        "max_num_batched_tokens": args.get("max-num-batched-tokens"),
        "max_num_seqs": args.get("max-num-seqs"),
        "async_scheduling": async_scheduling
        if isinstance(async_scheduling, bool)
        else None,
        "kv_cache_dtype": args.get("kv-cache-dtype"),
        "capture_profile": 0,
        "profile_dir": None,
        "profile_gcs_dir": None,
        "timestamp": timestamp,
        "mmlu_pro_disable_multiturn_args": mmlu_pro_single_turn,
        "lm_eval_tasks": " ".join(item["name"] for item in items),
        "extra_lm_eval_model_args": _model_args(lm_eval.get("model_args") or {}),
        "lm_eval_gen_kwargs": _gen_kwargs(lm_eval.get("gen_kwargs")),
        "perf_tolerance": perf.get("tolerance") if perf else None,
        "eval_tolerance": evals.get("tolerance"),
    }


def scenario_run(
    plan: Plan,
    *,
    resolved: Mapping[str, Any] | None,
    target: TargetInfo,
    placement: Mapping[str, Any],
    image: str,
) -> dict[str, Any]:
    """What ran: the resolved scenario, the target, the Plan, apart (R-RUN-14)."""
    return {
        "tool_version": __version__,
        "scenario": {
            "ref": plan.scenario_ref,
            "name": plan.scenario_name,
            "point": plan.point,
        },
        "experiment": bool(plan.experiment),
        "overrides": list(plan.experiment),
        "image": image,
        "target": {"name": target.name, "commit": target.commit},
        "placement": dict(placement),
        "resolved": dict(resolved) if resolved is not None else None,
        "plan_sha": codec.digest(plan),
        "plan": codec.to_data(plan),
    }


class Summary:
    """summary.json: state, exit code, commands_ok, phases, tasks, after-tasks.

    Saved after every change worth keeping, so a pod killed mid-run leaves
    a consistent record of what finished.
    """

    def __init__(
        self, path: pathlib.Path, plan: Plan, started_at: str, clock: Any = None
    ) -> None:
        self.path = path
        self._clock = clock or time.monotonic
        self._start = self._clock()
        self.data: dict[str, Any] = {
            "scenario": plan.scenario_name,
            "point": plan.point,
            "plan_sha": codec.digest(plan),
            "started_at": started_at,
            "state": "starting",
            "exit": None,
            "commands_ok": True,
            "phases": {},
            "tasks": [],
            "after_tasks": [],
        }

    def elapsed(self) -> float:
        return round(self._clock() - self._start, 3)

    def phase(self, name: str) -> None:
        self.data["phases"][name] = self.elapsed()
        self.save()

    def set(self, **values: Any) -> None:
        self.data.update(values)
        self.save()

    def command_failed(self) -> None:
        self.data["commands_ok"] = False

    @property
    def commands_ok(self) -> bool:
        return bool(self.data["commands_ok"])

    def task(self, kind: str, name: str) -> dict[str, Any]:
        entry: dict[str, Any] = {"kind": kind, "name": name, "status": "pending"}
        self.data["tasks"].append(entry)
        return entry

    def save(self) -> None:
        if self.path.parent.is_dir():
            write_json(self.path, self.data)
