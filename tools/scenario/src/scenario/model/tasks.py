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
agentx task runs alone, on a fresh server.
"""

import re
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BeforeValidator, Discriminator, Field, JsonValue, Tag

from scenario.model.baseline import (
    AgentxBaselineField,
    EvalBaselineField,
    PerfBaselineField,
)
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


class AgentxLabels(Model):
    """What InferenceX files the AgentX result row under."""

    model_prefix: Label = Field(
        description="InferenceX model prefix; also selects the trace corpus."
    )
    precision: Label = Field(description="Precision label, e.g. fp4.")
    framework: Label = Field(
        description="Serving framework label; selects InferenceX's metrics parser."
    )
    spec_decoding: Label = Field(
        description="Speculative decoding method recorded with the row, or none."
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
    """InferenceX AgentX trace replay at one concurrency.

    Only what the tool provides is modeled; every other client setting,
    including the context length, is pass-through env.
    """

    concurrency: Annotated[PosInt, BeforeValidator(_one_concurrency)] = Field(
        description="Concurrent sessions; one value per server."
    )
    duration: Duration = Field(description="How long the replay runs.")
    fast: bool = Field(
        default=False,
        description="InferenceX's bring-up mode (AIPERF_EXPERIMENTAL_FAST): a "
        "fixed 20m window and one warmup per lane; duration must then be 20m. "
        "Its numbers are bring-up evidence, not a canonical result.",
    )
    timeout: Duration = Field(
        description="The task is killed and fails after this; longer than "
        "duration, to cover warmup and drain."
    )
    labels: AgentxLabels = Field(description="Result row labels.")
    env: EnvMap = Field(
        default_factory=dict,
        description="Client settings passed through untouched, e.g. "
        "AIPERF_MAX_CONTEXT_LENGTH.",
    )
    tolerance: Tolerance = Field(description="Relative tolerance of the AgentX gate.")
    baseline: AgentxBaselineField | None = Field(
        default=None, description="Inline baseline by metric, or {from: <ref>}."
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


class EvalTask(Model):
    """Accuracy evals, gated against a baseline."""

    lm_eval: LmEval = Field(description="lm-eval runs.")
    tolerance: Tolerance = Field(
        description="Smallest score drop the eval gate allows, in absolute "
        "terms; the gate widens it by the measured stderr."
    )
    baseline: EvalBaselineField | None = Field(
        default=None, description="Inline baseline by run key, or {from: <ref>}."
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
    perf: PerfTask | None = Field(default=None, description="Benchmark cells.")
    agentx: AgentxTask | None = Field(
        default=None,
        description="AgentX trace replay; must be the only task.",
    )
    eval: EvalTask | None = Field(default=None, description="Accuracy evals.")
    custom: list[CustomTask] = Field(
        default_factory=list, description="Commands, run after perf and before eval."
    )
