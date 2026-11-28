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
"""Tasks: what runs against the servers once they are ready.

Tasks run in a fixed order, perf, then custom in list order, then eval. An
agentx task runs alone, on a fresh server, and so does an eval that runs
InferenceX's eval client.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BeforeValidator, Discriminator, Field, JsonValue, Tag

from scenario.model.baseline import EvalBaselineField, PerfBaselineField
from scenario.model.common import (
    ArgMap,
    Duration,
    EnvMap,
    Label,
    Matches,
    Model,
    NonNegInt,
    PosInt,
    TaskName,
    Tolerance,
    UnitInterval,
    fail,
)
from scenario.paths import KeyPath

TASK_ORDER = ("perf", "custom", "eval")
BUILTIN_TASKS = ("perf", "agentx", "eval")
# The tasks gated against a baseline; AgentX and InferenceX evals are checked
# by InferenceX itself.
GATED_TASKS = ("perf", "eval")


def _shape(value: object) -> str:
    if isinstance(value, dict):
        return "mapping"
    if isinstance(value, list):
        return "list"
    return "scalar"


class PerConcurrency(Model):
    """A count that scales with each cell's concurrency."""

    per_concurrency: PosInt = Field(
        description="Multiplied by the cell's concurrency: N x concurrency."
    )


PromptCount = Annotated[
    Annotated[PosInt, Tag("scalar")] | Annotated[PerConcurrency, Tag("mapping")],
    Discriminator(_shape),
]
WarmupCount = Annotated[
    Annotated[NonNegInt, Tag("scalar")] | Annotated[PerConcurrency, Tag("mapping")],
    Discriminator(_shape),
]


def count_for(value: int | PerConcurrency, concurrency: int) -> int:
    if isinstance(value, PerConcurrency):
        return value.per_concurrency * concurrency
    return value


OneOrMore = Annotated[
    Annotated[PosInt, Tag("scalar")]
    | Annotated[list[PosInt], Field(min_length=1), Tag("list")],
    Discriminator(_shape),
]
IslOsl = Annotated[list[PosInt], Field(min_length=2, max_length=2)]
PercentileMetric = Literal["ttft", "tpot", "itl", "e2el"]
PercentileMetrics = Annotated[list[PercentileMetric], Field(min_length=1)]
RangeRatioStyle = Literal["symmetric", "min"]


class Dataset(Model):
    """Where perf prompts come from."""

    name: Literal["random", "sharegpt"] = Field(
        description="--dataset-name. Others are added with a lane that uses them."
    )
    path: str | None = Field(
        default=None, description="--dataset-path, repo-relative or absolute."
    )


LABEL_FIELDS = ("isl", "osl", "concurrency", "repeat")
LABEL_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


def format_label(template: str, values: dict[str, int]) -> str:
    """Fill a cell label; unknown placeholders stay as written."""

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        return str(values[name]) if name in values else match.group(0)

    return LABEL_PLACEHOLDER.sub(fill, template)


class PerfCellGroup(Model):
    """Explicit perf cells, as an alternative to the isl_osl x concurrency product."""

    label: str | None = Field(
        default=None,
        description="Cell key and --label. May use {isl}, {osl}, {concurrency} "
        "and {repeat}; without it the key is isl{I}_osl{O}_c{C}.",
    )
    isl: OneOrMore = Field(description="Input length, or several.")
    osl: OneOrMore = Field(description="Output length, or several.")
    concurrency: OneOrMore = Field(description="Client concurrency, or several.")
    num_prompts: PromptCount | None = Field(
        default=None, description="Prompts per cell; defaults to the task's."
    )
    percentile_metrics: PercentileMetrics | None = Field(
        default=None, description="--percentile-metrics; defaults to the task's."
    )
    repeat: PosInt = Field(
        default=1,
        description="Runs of each cell; more than one needs {repeat} in label.",
    )


@dataclass(frozen=True)
class PerfCell:
    key: str
    isl: int
    osl: int
    concurrency: int
    repeat: int
    num_prompts: int | PerConcurrency
    path: KeyPath


def _listed(value: int | list[int]) -> list[int]:
    return value if isinstance(value, list) else [value]


class PerfTask(Model):
    """Fixed-length benchmark cells against the server or router."""

    client: Literal["vllm-bench", "inferencex"] = Field(
        default="vllm-bench",
        description="vllm-bench runs vllm bench serve; inferencex runs "
        "InferenceX's benchmark_serving at the version baked into the image.",
    )
    dataset: Dataset = Field(description="Prompt source.")
    isl_osl: list[IslOsl] | None = Field(
        default=None,
        description="[input length, output length] pairs; one cell per pair and "
        "concurrency.",
    )
    concurrency: Annotated[list[PosInt], Field(min_length=1)] | None = Field(
        default=None, description="Client concurrency levels."
    )
    num_prompts: PromptCount = Field(
        description="Prompts per measured run: a number or {per_concurrency: N}."
    )
    num_warmups: WarmupCount | None = Field(
        default=None,
        description="Warmup requests before each measured run (--num-warmups): a "
        "number or {per_concurrency: N}.",
    )
    cells: Annotated[list[PerfCellGroup], Field(min_length=1)] | None = Field(
        default=None,
        description="Explicit cells instead of isl_osl x concurrency.",
    )
    random_range_ratio: UnitInterval | None = Field(
        default=None,
        description="Length jitter of the random dataset; required with it.",
    )
    range_ratio_style: RangeRatioStyle | None = Field(
        default=None,
        description="How random_range_ratio reads for vllm-bench: symmetric is "
        "vllm bench serve's own form; min is translated to it as "
        "run_benchmarks.sh does. Not used by inferencex.",
    )
    warmup_runs: NonNegInt = Field(
        description="Whole warmup passes per cell, discarded, before the measured run."
    )
    temperature: Annotated[float, Field(ge=0)] | None = Field(
        default=None,
        description="Sampling temperature; unset sends none (server default).",
    )
    extra_body: dict[str, JsonValue] | None = Field(
        default=None, description="JSON merged into every request (--extra-body)."
    )
    tokenizer: str | None = Field(
        default=None,
        description="Client tokenizer. Unset: the server's pulled weights for an "
        "object-storage model, else model.name.",
    )
    percentile_metrics: PercentileMetrics = Field(description="--percentile-metrics.")
    client_args: ArgMap = Field(
        default_factory=dict,
        description="Extra client flags, rendered like server.args.",
    )
    env: EnvMap = Field(
        default_factory=dict,
        description="Env for the client process, passed through untouched.",
    )
    tolerance: Tolerance = Field(description="Relative tolerance of the perf gate.")
    baseline: PerfBaselineField | None = Field(
        default=None,
        description="Inline baseline keyed by cell, or {from: <ref>}.",
    )

    def cell_list(self) -> tuple[PerfCell, ...]:
        """Every cell in run order; duplicate keys are left for validation."""
        cells: list[PerfCell] = []
        if self.cells is not None:
            for index, group in enumerate(self.cells):
                for isl in _listed(group.isl):
                    for osl in _listed(group.osl):
                        for concurrency in _listed(group.concurrency):
                            for repeat in range(1, group.repeat + 1):
                                values = {
                                    "isl": isl,
                                    "osl": osl,
                                    "concurrency": concurrency,
                                    "repeat": repeat,
                                }
                                key = (
                                    format_label(group.label, values)
                                    if group.label is not None
                                    else f"isl{isl}_osl{osl}_c{concurrency}"
                                )
                                cells.append(
                                    PerfCell(
                                        key=key,
                                        isl=isl,
                                        osl=osl,
                                        concurrency=concurrency,
                                        repeat=repeat,
                                        num_prompts=group.num_prompts
                                        or self.num_prompts,
                                        path=("tasks", "perf", "cells", index),
                                    )
                                )
        elif self.isl_osl is not None and self.concurrency is not None:
            for index, (isl, osl) in enumerate(self.isl_osl):
                for concurrency in self.concurrency:
                    cells.append(
                        PerfCell(
                            key=f"isl{isl}_osl{osl}_c{concurrency}",
                            isl=isl,
                            osl=osl,
                            concurrency=concurrency,
                            repeat=1,
                            num_prompts=self.num_prompts,
                            path=("tasks", "perf", "isl_osl", index),
                        )
                    )
        return tuple(cells)


# defaults.yaml perf entries that only mean something for vllm bench serve;
# resolution leaves them out of an inferencex perf task.
VLLM_BENCH_ONLY_DEFAULTS = ("client_args", "range_ratio_style")


SpecDecoding = Literal["mtp", "draft_model", "none"]
KvOffloading = Literal["none", "dram"]


class KvOffloadBackend(Model):
    """The KV offload backend InferenceX records with a dram row."""

    name: Label = Field(description="Backend name, e.g. mooncake.")
    version: str | None = Field(default=None, description="Backend version.")


class Router(Model):
    """The router InferenceX records with a row."""

    name: Label = Field(description="Router name, e.g. dynamo-router.")
    version: str = Field(description="Router version.")


class InferencexIdentity(Model):
    """What InferenceX files a deployment's results under.

    The AgentX rows of a deployment and its InferenceX evals carry the same
    identity, which is how InferenceX joins an eval to its rows.
    """

    model_prefix: Label = Field(
        description="InferenceX model prefix (MODEL_PREFIX); also selects the "
        "AgentX trace corpus and the eval score floors."
    )
    precision: Label = Field(description="Precision (PRECISION), e.g. fp4.")
    framework: Label = Field(
        description="Serving framework (FRAMEWORK); selects InferenceX's metrics "
        "parser."
    )
    spec_decoding: SpecDecoding = Field(
        description="Speculative decoding method (SPEC_DECODING)."
    )
    kv_offloading: KvOffloading = Field(
        description="KV cache offloading (KV_OFFLOADING); part of the row's key."
    )
    kv_offload_backend: KvOffloadBackend | None = Field(
        default=None,
        description="KV_OFFLOAD_BACKEND and its metadata; required with "
        "kv_offloading dram, not allowed with none.",
    )
    total_cpu_dram_gb: PosInt | None = Field(
        default=None,
        description="CPU DRAM given to KV offload, in GB (TOTAL_CPU_DRAM_GB); "
        "required with kv_offloading dram.",
    )
    router: Router | None = Field(
        default=None, description="The router in front of the roles (ROUTER_METADATA)."
    )
    kv_p2p_transfer: Label | None = Field(
        default=None,
        description="KV transfer between P/D roles (KV_P2P_TRANSFER), e.g. nixl.",
    )


# InferenceX's fast mode replaces DURATION with this window (infx/bench/agentic/
# replay.py), so a fast run states it rather than a duration it would ignore.
FAST_WINDOW_S = 1200


def _one_concurrency(value: object) -> object:
    if isinstance(value, list):
        raise fail(
            "V-31",
            "AgentX measures one concurrency per fresh server; write one value, "
            "and sweep it for a ladder: sweep: {name: c{tasks.agentx.concurrency}, "
            "zip: {tasks.agentx.concurrency: [1, 2, 4, 8]}}",
        )
    return value


class AgentxTask(Model):
    """InferenceX AgentX trace replay at one concurrency, by infx.bench agentic.

    The fields are what InferenceX's matrix and workflow give a point; every
    other client setting, including the corpus and the context length, is
    pass-through env, as an InferenceX recipe's benchmark.env is. The row's
    identity is tasks.inferencex.
    """

    concurrency: Annotated[PosInt, BeforeValidator(_one_concurrency)] = Field(
        description="Concurrent sessions; one value per server."
    )
    duration: Duration = Field(
        description="The profile window (DURATION); InferenceX publishes 1h "
        "windows, and aiperf marks one under 15m invalid."
    )
    fast: bool = Field(
        default=False,
        description="InferenceX's bring-up mode (AIPERF_EXPERIMENTAL_FAST): a "
        "fixed 20m window and one warmup request per lane; duration must then be "
        "20m. InferenceX never publishes it.",
    )
    timeout: Duration = Field(
        description="The task is killed and fails after this; longer than "
        "duration, to cover warmup and drain."
    )
    env: EnvMap = Field(
        default_factory=dict,
        description="Client settings passed through untouched, as a recipe's "
        "benchmark.env, e.g. AGENTIC_WARMUP_GRACE_PERIOD or WEKA_LOADER_OVERRIDE.",
    )


LmEvalTaskName = Annotated[
    str,
    Matches("lm_eval_task", r"[^\s,]+", "lm_eval task names have no spaces or commas"),
]
Limit = Annotated[
    Annotated[PosInt, Tag("scalar")] | Annotated[Literal["all"], Tag("all")],
    Discriminator(lambda value: "all" if isinstance(value, str) else "scalar"),
]
GenKwargs = str | dict[str, JsonValue]
ModelArgKey = Annotated[
    str,
    Matches(
        "model_arg",
        r"[A-Za-z_][A-Za-z0-9_]*",
        "lm_eval model_args keys are letters, digits and '_'",
    ),
]


def _model_arg_value(value: object) -> object:
    if isinstance(value, str) and ("," in value or "=" in value):
        raise fail(
            "model_arg",
            "lm_eval reads model_args as k=v,k=v; a value cannot hold ',' or '='",
        )
    return value


ModelArgValue = Annotated[str | int | float | bool, BeforeValidator(_model_arg_value)]


class LmEvalTask(Model):
    """One lm_eval task with options of its own."""

    name: LmEvalTaskName = Field(description="lm_eval task name.")
    limit: Limit | None = Field(
        default=None,
        description="Questions (per subtask); overrides lm_eval.limit; all runs "
        "the full set.",
    )
    limit_per_concurrency: PosInt | None = Field(
        default=None,
        description="Raises the limit at each lm_eval.concurrency value to "
        "ceil(N x concurrency / subtasks), so that concurrency is reached.",
    )
    subtasks: PosInt | None = Field(
        default=None,
        description="Subtasks lm_eval applies the limit to one by one, e.g. 14 "
        "for mmlu_pro; divides limit_per_concurrency.",
    )
    gen_kwargs: GenKwargs | None = Field(
        default=None, description="Replaces lm_eval.gen_kwargs for this task."
    )
    fewshot_as_multiturn: bool | None = Field(
        default=None, description="Overrides lm_eval.fewshot_as_multiturn."
    )
    include_path: str | None = Field(
        default=None, description="--include_path, for tasks defined in this repo."
    )
    confirm_run_unsafe_code: bool = Field(
        default=False, description="--confirm_run_unsafe_code, for code tasks."
    )
    env: EnvMap = Field(
        default_factory=dict,
        description="Env of this task's lm_eval run, e.g. HF_ALLOW_CODE_EVAL.",
    )


def _task_shape(value: object) -> str:
    return "mapping" if isinstance(value, dict) else "scalar"


LmEvalTaskItem = Annotated[
    Annotated[LmEvalTaskName, Tag("scalar")] | Annotated[LmEvalTask, Tag("mapping")],
    Discriminator(_task_shape),
]


class LmEval(Model):
    """lm-eval against the endpoint's /v1/chat/completions, one run per task."""

    tasks: Annotated[list[LmEvalTaskItem], Field(min_length=1)] = Field(
        description="Task names, or mappings with per-task options."
    )
    model: str = Field(description="--model, e.g. local-chat-completions.")
    model_args: dict[ModelArgKey, ModelArgValue] = Field(
        default_factory=dict,
        description="--model_args entries; the tool adds model and base_url.",
    )
    gen_kwargs: GenKwargs | None = Field(
        default=None,
        description="--gen_kwargs: a string verbatim, or a mapping as JSON.",
    )
    limit: Limit | None = Field(
        default=None,
        description="Questions per task (per subtask); unset or all runs the full set.",
    )
    seed: str = Field(description="--seed.")
    apply_chat_template: bool = Field(description="--apply_chat_template.")
    fewshot_as_multiturn: bool = Field(description="--fewshot_as_multiturn true.")
    concurrency: Annotated[list[PosInt], Field(min_length=1)] | None = Field(
        default=None,
        description="Client-side sweep: every task runs once per value, with "
        "num_concurrent set to it, against one server.",
    )

    def task_items(self) -> tuple[LmEvalTask, ...]:
        return tuple(
            item if isinstance(item, LmEvalTask) else LmEvalTask(name=item)
            for item in self.tasks
        )

    def run_keys(self) -> tuple[str, ...]:
        """Result and baseline keys: the task, with _c<N> per concurrency."""
        names = [task.name for task in self.task_items()]
        if self.concurrency is None:
            return tuple(names)
        return tuple(f"{name}_c{c}" for name in names for c in self.concurrency)


EvalClient = Literal["lm_eval", "inferencex"]
InferencexFramework = Literal["lm-eval", "kimi-vendor", "minimax-vendor", "bfcl"]
# InferenceX's vendor frameworks, each with a full suite as an example (bfcl's
# depends on the model); without EVAL_SUITE a vendor framework runs its smoke
# suite.
VENDOR_SUITES: Mapping[str, str] = {
    "kimi-vendor": "kimi_tool_call_schema_full",
    "minimax-vendor": "minimax_m3_full",
    "bfcl": "bfcl_vllm_kimi",
}
EvalSuite = Annotated[
    str,
    Matches(
        "eval_suite",
        r"[A-Za-z0-9_.-]+",
        "InferenceX suite names are letters, digits, '.', '_' and '-'",
    ),
]

# The tasks.eval fields each client reads; V-46 rejects the other client's.
LM_EVAL_CLIENT_FIELDS = ("lm_eval", "tolerance", "baseline")
INFERENCEX_CLIENT_FIELDS = (
    "framework",
    "suite",
    "concurrency",
    "timeout",
    "env",
)


def _one_eval_concurrency(value: object) -> object:
    if isinstance(value, list):
        raise fail(
            "V-31",
            "InferenceX evaluates a deployment at one concurrency; write one "
            "value, and add a sweep point for another",
        )
    return value


class EvalTask(Model):
    """Accuracy evals: lm_eval runs, or InferenceX's eval client.

    lm_eval runs are gated against a baseline. InferenceX's eval client runs
    alone on a fresh deployment and is gated by InferenceX's own score floor.
    """

    client: EvalClient = Field(
        default="lm_eval",
        description="lm_eval runs lm_eval once per task, gated against a "
        "baseline; inferencex runs InferenceX's eval client (infx.bench eval) at "
        "the version baked into the image, alone on a fresh deployment, gated by "
        "InferenceX's score floor.",
    )
    lm_eval: LmEval | None = Field(
        default=None, description="lm-eval runs; client lm_eval, which needs it."
    )
    tolerance: Tolerance | None = Field(
        default=None,
        description="Smallest score drop the eval gate allows, in absolute "
        "terms; the gate widens it by the measured stderr. Client lm_eval, which "
        "needs it.",
    )
    baseline: EvalBaselineField | None = Field(
        default=None,
        description="Inline baseline by run key, or {from: <ref>}; client lm_eval.",
    )
    framework: InferencexFramework | None = Field(
        default=None,
        description="InferenceX's eval framework (--framework); unset runs "
        "lm-eval, which is GSM8K. Client inferencex.",
    )
    suite: EvalSuite | None = Field(
        default=None,
        description="EVAL_SUITE: required with kimi-vendor and minimax-vendor, "
        "which run their smoke suite without it; not allowed with lm-eval.",
    )
    concurrency: Annotated[PosInt, BeforeValidator(_one_eval_concurrency)] | None = (
        Field(
            default=None,
            description="Concurrent requests, one value: the highest AgentX "
            "concurrency of the deployment, which InferenceX evaluates. Client "
            "inferencex, which needs it.",
        )
    )
    timeout: Duration | None = Field(
        default=None,
        description="The eval is killed and fails after this. Client "
        "inferencex, which needs it.",
    )
    env: EnvMap = Field(
        default_factory=dict,
        description="Settings of InferenceX's eval client passed through "
        "untouched, e.g. EVAL_LIMIT. Client inferencex.",
    )


class CustomTask(Model):
    """A command run against the servers, e.g. a correctness script."""

    name: TaskName = Field(
        description="Unique name; selects the task in --only and names its log."
    )
    command: Annotated[list[str], Field(min_length=1)] = Field(
        description="Argv, run from the repo root."
    )
    env: EnvMap = Field(
        default_factory=dict,
        description="Extra env. Values may use ${NAME} for names the tool "
        "provides (e.g. ${SCENARIO_RESULTS_DIR}); $$ is a literal $.",
    )
    timeout: Duration = Field(description="The task is killed and fails after this.")
    skip_rest_if_exit_ge: PosInt | None = Field(
        default=None,
        description="Skip the remaining tasks when this one exits with this "
        "code or higher.",
    )


class Tasks(Model):
    """What runs against the servers. Empty means serve only."""

    driver: (
        Annotated[
            str,
            Matches(
                "driver",
                r"[a-z][a-z0-9-]{0,14}",
                "tasks.driver is a role name, or client for a chipless host",
            ),
        ]
        | None
    ) = Field(
        default=None,
        description="Where tasks and the router run: a role (its first host) or "
        "client (a chipless host). Required with server.roles.",
    )
    inferencex: InferencexIdentity | None = Field(
        default=None,
        description="The deployment's InferenceX identity, shared by the AgentX "
        "rows and the InferenceX evals; both need it.",
    )
    perf: PerfTask | None = Field(default=None, description="Benchmark cells.")
    agentx: AgentxTask | None = Field(
        default=None,
        description="AgentX trace replay; must be the only task.",
    )
    eval: EvalTask | None = Field(
        default=None,
        description="Accuracy evals; with client inferencex, the only task.",
    )
    custom: list[CustomTask] = Field(
        default_factory=list, description="Commands, run after perf and before eval."
    )
