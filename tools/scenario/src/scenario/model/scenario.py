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
"""The top level of a scenario file, and of scenarios/defaults.yaml."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field

from scenario.model.common import (
    ArgMap,
    Duration,
    EnvMap,
    ImageRef,
    Label,
    Model,
    Name,
    NonNegInt,
    RefOrPath,
    Tolerance,
    fail,
)
from scenario.model.server import Server
from scenario.model.sweep import Sweep
from scenario.model.tasks import (
    AgentxLabels,
    AgentxTask,
    Dataset,
    GenKwargs,
    Limit,
    LmEval,
    ModelArgKey,
    ModelArgValue,
    PercentileMetrics,
    PerfTask,
    PromptCount,
    RangeRatioStyle,
    Tasks,
    WarmupCount,
)

SCHEMA_VERSION = 1


def _schema_version(value: object) -> object:
    if value != SCHEMA_VERSION or isinstance(value, bool):
        raise fail(
            "V-01",
            f"schema {value!r} is not supported; this tool reads schema "
            f"{SCHEMA_VERSION}",
        )
    return value


SchemaVersion = Annotated[Literal[1], BeforeValidator(_schema_version)]


class Scenario(Model):
    """One workload: the servers to start and the tasks to run against them."""

    schema_version: SchemaVersion = Field(
        alias="schema", description="Schema version; always 1."
    )
    name: Name = Field(
        description="Unique across scenarios/; names the results directory. "
        "Not inherited through extends."
    )
    extends: RefOrPath | None = Field(
        default=None,
        description="Parent scenario, merged under this file: mappings merge, "
        "lists and scalars replace, null deletes. Baselines are not inherited.",
    )
    server: Server = Field(description="What runs.")
    tasks: Tasks = Field(
        default_factory=Tasks, description="What runs against the servers."
    )
    sweep: Sweep | None = Field(
        default=None,
        description="Server-side sweep: each point is a fresh server with its "
        "overrides.",
    )


def _like(model: type[BaseModel], field: str) -> Any:
    """An optional defaults.yaml field described like `model.field`."""
    return Field(default=None, description=model.model_fields[field].description)


class DefaultsServer(Model):
    """Server values applied under every scenario."""

    image: ImageRef | None = _like(Server, "image")
    startup_timeout: Duration | None = _like(Server, "startup_timeout")
    args: ArgMap | None = _like(Server, "args")
    env: EnvMap | None = _like(Server, "env")


class DefaultsPerf(Model):
    """tasks.perf values, applied only to scenarios that declare tasks.perf."""

    dataset: Dataset | None = _like(PerfTask, "dataset")
    num_prompts: PromptCount | None = _like(PerfTask, "num_prompts")
    num_warmups: WarmupCount | None = _like(PerfTask, "num_warmups")
    range_ratio_style: RangeRatioStyle | None = _like(PerfTask, "range_ratio_style")
    warmup_runs: NonNegInt | None = _like(PerfTask, "warmup_runs")
    percentile_metrics: PercentileMetrics | None = _like(PerfTask, "percentile_metrics")
    client_args: ArgMap | None = _like(PerfTask, "client_args")
    tolerance: Tolerance | None = _like(PerfTask, "tolerance")


class DefaultsAgentxLabels(Model):
    """tasks.agentx.labels values."""

    model_prefix: Label | None = _like(AgentxLabels, "model_prefix")
    precision: Label | None = _like(AgentxLabels, "precision")
    framework: Label | None = _like(AgentxLabels, "framework")
    spec_decoding: Label | None = _like(AgentxLabels, "spec_decoding")


class DefaultsAgentx(Model):
    """tasks.agentx values, applied only to scenarios that declare tasks.agentx."""

    duration: Duration | None = _like(AgentxTask, "duration")
    timeout: Duration | None = _like(AgentxTask, "timeout")
    labels: DefaultsAgentxLabels | None = _like(AgentxTask, "labels")
    tolerance: Tolerance | None = _like(AgentxTask, "tolerance")


class DefaultsLmEval(Model):
    """tasks.eval.lm_eval values, applied only where tasks.eval.lm_eval is declared."""

    model: str | None = _like(LmEval, "model")
    model_args: dict[ModelArgKey, ModelArgValue] | None = _like(LmEval, "model_args")
    gen_kwargs: GenKwargs | None = _like(LmEval, "gen_kwargs")
    limit: Limit | None = _like(LmEval, "limit")
    seed: str | None = _like(LmEval, "seed")
    apply_chat_template: bool | None = _like(LmEval, "apply_chat_template")
    fewshot_as_multiturn: bool | None = _like(LmEval, "fewshot_as_multiturn")


class DefaultsEval(Model):
    """tasks.eval values."""

    lm_eval: DefaultsLmEval | None = Field(
        default=None, description="Defaults for tasks.eval.lm_eval."
    )


class DefaultsTasks(Model):
    """Task values, each applied only to scenarios that declare that task."""

    perf: DefaultsPerf | None = Field(
        default=None, description="Defaults for tasks.perf."
    )
    agentx: DefaultsAgentx | None = Field(
        default=None, description="Defaults for tasks.agentx."
    )
    eval: DefaultsEval | None = Field(
        default=None, description="Defaults for tasks.eval."
    )


class Defaults(Model):
    """scenarios/defaults.yaml: the choices every scenario starts from."""

    schema_version: SchemaVersion = Field(
        alias="schema", description="Schema version; always 1."
    )
    server: DefaultsServer | None = Field(
        default=None, description="Applied under every scenario."
    )
    tasks: DefaultsTasks | None = Field(
        default=None, description="Applied only to the tasks a scenario declares."
    )
