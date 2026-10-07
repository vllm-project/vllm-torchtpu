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
"""Task commands for the driver: perf cells, AgentX, lm_eval runs, custom tasks.

The argv shapes follow today's scripts: run_benchmarks.sh for vllm bench
serve, InferenceX's qwen3.5/bench.sh for its benchmark_serving client,
run_eval_flow.sh and eval_conc_sweep.sh for lm_eval.
"""

import math
from collections.abc import Mapping, Sequence

from scenario.model.server import ModelSpec
from scenario.model.tasks import (
    AgentxTask,
    CustomTask,
    EvalTask,
    PerfCell,
    PerfTask,
    count_for,
)
from scenario.plan.argv import compact_json, render_args, render_model_args
from scenario.plan.env import substitute
from scenario.plan.plumbing import (
    AGENTX_PYTHON,
    agentx_env,
    client_tokenizer,
    inferencex_perf_env,
    result_path,
)
from scenario.plan.types import (
    AgentxTaskPlan,
    Command,
    CustomTaskPlan,
    EvalRunPlan,
    EvalTaskPlan,
    GatePlan,
    PerfCellPlan,
    PerfTaskPlan,
    Ref,
    RolePlan,
    Value,
    concat,
    env_items,
)

VLLM_BENCH = ("vllm", "bench", "serve")
INFERENCEX_BENCH = ("python3", "-m", "infx.bench_serving.benchmark_serving")
AGENTX = (AGENTX_PYTHON, "-m", "infx.bench", "agentic")


def _random_lengths(perf: PerfTask, cell: PerfCell) -> tuple[str, str, str]:
    """ISL, OSL and range ratio as the client reads them.

    A `min` ratio samples in [r x len, len]; vllm bench serve samples in
    [(1-h) x mid, (1+h) x mid], so it gets mid = len x (1+r)/2 and
    h = (1-r)/(1+r), rounded as run_benchmarks.sh does.
    """
    ratio = perf.random_range_ratio or 0.0
    if perf.client == "vllm-bench" and perf.range_ratio_style == "min":
        isl, osl = (f"{n * (1 + ratio) / 2:.0f}" for n in (cell.isl, cell.osl))
        return isl, osl, f"{(1 - ratio) / (1 + ratio):.6f}"
    return str(cell.isl), str(cell.osl), str(ratio)


def _dataset_args(perf: PerfTask, cell: PerfCell) -> list[str]:
    if perf.dataset.name == "random":
        isl, osl, ratio = _random_lengths(perf, cell)
        return [
            "--dataset-name",
            "random",
            "--random-input-len",
            isl,
            "--random-output-len",
            osl,
            "--random-range-ratio",
            ratio,
        ]
    args = ["--dataset-name", perf.dataset.name]
    if perf.dataset.path is not None:
        args += ["--dataset-path", perf.dataset.path]
    if perf.dataset.name == "sharegpt":
        args += ["--sharegpt-output-len", str(cell.osl)]
    return args


def _cell_argv(
    perf: PerfTask,
    cell: PerfCell,
    model: ModelSpec,
    host: Value,
    port: int,
    file: str,
) -> tuple[Value, ...]:
    labeled = perf.cells is not None
    metrics = perf.percentile_metrics
    if perf.cells is not None:
        group = perf.cells[int(cell.path[-1])]
        metrics = group.percentile_metrics or metrics
    load: list[str] = [
        "--num-prompts",
        str(count_for(cell.num_prompts, cell.concurrency)),
        "--max-concurrency",
        str(cell.concurrency),
    ]
    if perf.num_warmups is not None:
        load += ["--num-warmups", str(count_for(perf.num_warmups, cell.concurrency))]
    load += ["--percentile-metrics", ",".join(metrics)]
    if perf.temperature is not None:
        load += ["--temperature", str(perf.temperature)]
    if perf.extra_body is not None:
        load += ["--extra-body", compact_json(perf.extra_body)]
    if labeled:
        load += ["--label", cell.key]
    tokenizer = client_tokenizer(perf.tokenizer, model)
    dataset = _dataset_args(perf, cell)
    client_args = render_args(perf.client_args)
    if perf.client == "inferencex":
        return (
            *INFERENCEX_BENCH,
            "--model",
            model.name,
            *tokenizer,
            *client_args,
            "--base-url",
            concat("http://", host, f":{port}"),
            *dataset,
            *load,
            "--save-result",
            "--result-dir",
            Ref("results_dir"),
            "--result-filename",
            file,
        )
    return (
        *VLLM_BENCH,
        *client_args,
        "--model",
        model.name,
        *tokenizer,
        "--host",
        host,
        "--port",
        str(port),
        *dataset,
        *load,
        "--save-result",
        "--result-filename",
        result_path(file),
    )


def perf_task(
    perf: PerfTask, model: ModelSpec, host: Value, port: int, gate: GatePlan | None
) -> PerfTaskPlan:
    env: dict[str, Value] = dict(perf.env)
    if perf.client == "inferencex":
        env |= inferencex_perf_env()
    cells = []
    for cell in perf.cell_list():
        warmups = tuple(
            Command(
                _cell_argv(perf, cell, model, host, port, f"{cell.key}.warmup{n}.json"),
                env_items(env),
            )
            for n in range(1, perf.warmup_runs + 1)
        )
        file = f"{cell.key}.json"
        measured = Command(
            _cell_argv(perf, cell, model, host, port, file), env_items(env)
        )
        cells.append(PerfCellPlan(cell.key, warmups, measured, result_path(file)))
    return PerfTaskPlan(
        client=perf.client,
        mode="cells" if perf.cells is not None else "product",
        cells=tuple(cells),
        gate=gate,
    )


def agentx_task(
    task: AgentxTask,
    model: ModelSpec,
    roles: Sequence[RolePlan],
    host: Value,
    port: int,
    gate: GatePlan | None,
) -> AgentxTaskPlan:
    env = agentx_env(task, model, roles, concat("http://", host, f":{port}"))
    env |= task.env
    command = Command(AGENTX, env_items(env), timeout_s=task.timeout)
    return AgentxTaskPlan(command=command, timeout_s=task.timeout, gate=gate)


def eval_task(
    evals: EvalTask,
    model: ModelSpec,
    host: Value,
    port: int,
    gates: Mapping[str, GatePlan],
) -> EvalTaskPlan:
    """One lm_eval run per task, per concurrency (outer) in a sweep."""
    lm_eval = evals.lm_eval
    base_url = concat("http://", host, f":{port}/v1/chat/completions")
    runs = []
    for concurrency in lm_eval.concurrency or [None]:
        for item in lm_eval.task_items():
            key = item.name if concurrency is None else f"{item.name}_c{concurrency}"
            limit = item.limit if item.limit is not None else lm_eval.limit
            limit = None if limit == "all" else limit
            if (
                concurrency is not None
                and item.limit_per_concurrency is not None
                and isinstance(limit, int)
            ):
                needed = item.limit_per_concurrency * concurrency / (item.subtasks or 1)
                limit = max(limit, math.ceil(needed))
            model_args: list[Value] = ["model=", model.name, ",base_url=", base_url]
            if concurrency is not None:
                model_args.append(f",num_concurrent={concurrency}")
            if lm_eval.model_args:
                model_args.append("," + render_model_args(lm_eval.model_args))
            output = Ref("results_dir") if concurrency is None else result_path(key)
            argv: list[Value] = [
                "lm_eval",
                "--tasks",
                item.name,
                "--seed",
                lm_eval.seed,
                "--output_path",
                output,
            ]
            if item.include_path is not None:
                argv += ["--include_path", item.include_path]
            if limit is not None:
                argv += ["--limit", str(limit)]
            argv += ["--model", lm_eval.model, "--model_args", concat(*model_args)]
            if lm_eval.apply_chat_template:
                argv.append("--apply_chat_template")
            multiturn = item.fewshot_as_multiturn
            if multiturn is None:
                multiturn = lm_eval.fewshot_as_multiturn
            if multiturn:
                argv += ["--fewshot_as_multiturn", "true"]
            gen_kwargs = item.gen_kwargs
            if gen_kwargs is None:
                gen_kwargs = lm_eval.gen_kwargs
            if gen_kwargs is not None:
                text = (
                    gen_kwargs
                    if isinstance(gen_kwargs, str)
                    else compact_json(gen_kwargs)
                )
                argv += ["--gen_kwargs", text]
            if item.confirm_run_unsafe_code:
                argv.append("--confirm_run_unsafe_code")
            command = Command(tuple(argv), env_items(dict(item.env)))
            runs.append(EvalRunPlan(key, item.name, "lm_eval", command, gates.get(key)))
    return EvalTaskPlan(tuple(runs))


def custom_task(task: CustomTask, provided: Mapping[str, Value]) -> CustomTaskPlan:
    env = dict(provided)
    env |= {name: substitute(value, provided) for name, value in task.env.items()}
    command = Command(tuple(task.command), env_items(env), timeout_s=task.timeout)
    return CustomTaskPlan(task.name, command, task.timeout, task.skip_rest_if_exit_ge)
