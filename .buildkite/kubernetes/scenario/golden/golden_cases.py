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
"""The golden cases: a scenario, and how today's side of it starts.

Steps are read from the pipeline files, found by the command they run, so a
change to a step's command or env reaches the comparison. A scenario no kube
step runs gets a step written here, shaped like the multi-host steps of
pipeline_perf_kube.yml, with the env its own lane sets.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from golden_capture import REPO, DirectPod, Step, find_step, pipeline_env

PERF_KUBE = REPO / ".buildkite" / "pipeline_perf_kube.yml"
K3_SWEEP = REPO / ".buildkite" / "pipeline_k3_sweep.yml"
QWEN_SWEEP = REPO / "scripts/vllm/benchmark/inferencex/qwen3.5/sweep.sh"
QWEN_SERVER = "scripts/vllm/benchmark/inferencex/qwen3.5/server.sh"
RUN_MULTIHOST = ".buildkite/kubernetes/run_multihost.sh"

EVERY_TASK = frozenset({"router", "perf", "agentx", "eval", "custom", "gate", "other"})
# Server-side cases still compare the perf cell today's server boot runs.
SERVER_TASKS = frozenset({"perf", "gate", "other"})


@dataclass(frozen=True)
class Case:
    id: str
    scenario: str
    launch: Callable[[], Step | DirectPod]
    # Today's pod group -> plan role; `client` is the chipless driver.
    roles: Mapping[str, str]
    # The pod whose exit ends the run; None when every pod only serves.
    driver: str | None
    # Today's results directories, compared as <results>.
    results: tuple[str, ...]
    points: tuple[str | None, ...] = (None,)
    # Points of the scenario this case leaves out, each with why.
    skipped: Mapping[str, str] = field(default_factory=dict)
    compared: frozenset[str] = EVERY_TASK
    # Env names this case's today side gets as script inputs, by reason.
    inputs: Mapping[str, str] = field(default_factory=dict)


def perf_kube_step(command: str, env: Mapping[str, str | None] | None = None) -> Step:
    return find_step(PERF_KUBE, command, env)


def written_step(command: str, env: Mapping[str, str]) -> Step:
    """A step for a scenario no kube lane runs, in pipeline_perf_kube.yml."""
    return Step(command=command, env=dict(env), pipeline_env=pipeline_env(PERF_KUBE))


def k3_sweep_env(command: str) -> Mapping[str, str]:
    return find_step(K3_SWEEP, command).env


def results(config: str) -> tuple[str, ...]:
    return (f"perf_eval_results/{config}", "perf_eval_results")


SLICE = {"slice": "server"}

SHARDING_SLUGS = {"TP8_EP": "tp8", "DP4TP2_EP": "dp4tp2", "DP8_EP": "dp8"}
_TABLE_ROW = re.compile(r"\[(\d+),(\d+),(\d+)\]=(\w+)")


def sharding_table() -> list[tuple[int, int, int, str]]:
    """qwen3.5/sweep.sh's SHARDING_TABLE: (ISL, OSL, CONC, sharding) per point."""
    text = QWEN_SWEEP.read_text()
    block = text[text.index("declare -A SHARDING_TABLE=(") :]
    block = block[: block.index("\n)")]
    return [
        (int(isl), int(osl), int(conc), sharding)
        for isl, osl, conc, sharding in _TABLE_ROW.findall(block)
    ]


def qwen_point(conc: int, sharding: str) -> str:
    return f"{SHARDING_SLUGS[sharding]}-c{conc}"


def qwen_cases() -> list[Case]:
    cases = []
    for isl, osl, conc, sharding in sharding_table():
        env = {
            "SHARDING": sharding,
            "ISL": str(isl),
            "OSL": str(osl),
            "CONC": str(conc),
        }
        point = qwen_point(conc, sharding)
        cases.append(
            Case(
                id=f"qwen3.5-inferencex:{point}",
                scenario="qwen3.5-397b/inferencex-sweep",
                launch=lambda env=env: DirectPod(("bash", QWEN_SERVER), env),
                roles={"pod": "server"},
                driver=None,
                results=(),
                points=(point,),
                compared=frozenset(),
                inputs=dict.fromkeys(
                    env,
                    "sweep.sh's inputs to server.sh; the point writes out what "
                    "server.sh derives from them",
                ),
            )
        )
    return cases


def all_cases() -> list[Case]:
    return [
        Case(
            id="P2",
            scenario="qwen3-0.6b/tp1",
            launch=lambda: perf_kube_step("run_eval_flow.sh --config qwen3-0.6b-tp1"),
            roles={"pod": "server"},
            driver="pod-0",
            results=results("qwen3-0.6b-tp1"),
        ),
        Case(
            id="P11",
            scenario="glm-5.2/fp8-dp16-ep",
            launch=lambda: perf_kube_step(
                "run_eval_flow.sh --config glm-5.2-fp8-dp16-ep"
            ),
            roles=SLICE,
            driver="slice-0",
            results=results("glm-5.2-fp8-dp16-ep"),
        ),
        Case(
            id="P13",
            scenario="glm-5.2/fp8-dp16-ep-mp",
            launch=lambda: perf_kube_step(
                "run_multihost.sh --topology 2x2x2 --backend mp glm-5.2-fp8-dp16-ep"
            ),
            roles=SLICE,
            driver="slice-0",
            results=results("glm-5.2-fp8-dp16-ep"),
        ),
        Case(
            id="P14",
            scenario="qwen3.5-397b/fp8-pcp8-dp8-1p1d",
            launch=lambda: perf_kube_step(
                "run_multihost.sh --manifest qwen3.5-397b-1p1d.yaml"
            ),
            roles={"prefill": "prefill", "decode": "decode", "benchmark": "client"},
            driver="benchmark-0",
            results=("/tmp/benchmark_results",),
        ),
        Case(
            id="P15",
            scenario="kimi-k3/tp8-dp4-ep-1p1d",
            launch=lambda: perf_kube_step(
                "run_multihost.sh --manifest kimi-k3-1p1d-4dp8tp.yaml"
            ),
            roles={"prefill": "prefill", "decode": "decode"},
            driver="prefill-0",
            results=results("kimi-k3-tp8-dp4-ep-1p1d"),
        ),
        Case(
            id="kimi-k3-agentx-1p1d",
            scenario="kimi-k3/agentx-1p1d",
            # The replay step: an eval-only step may run the same manifest with
            # K3_AGENTX_EVAL_ONLY=1.
            launch=lambda: perf_kube_step(
                "run_multihost.sh --manifest kimi-k3-agentx-1p1d.yaml",
                env={"K3_AGENTX_EVAL_ONLY": None},
            ),
            roles={"prefill": "prefill", "decode": "decode", "client": "client"},
            driver="client-0",
            results=results("k3-agentx-1p1d"),
            points=("c1", "c4"),
            skipped={
                "gsm8k": "no kube step runs InferenceX's eval-only GSM8K, so "
                "there is nothing to compare; .buildkite/kubernetes/scenario/"
                "tests/test_scenarios.py snapshots its plan instead",
            },
            inputs=dict.fromkeys(
                ("K3_PD_CONFIG", "K3_PD_EXTERNAL_CLIENT", "AGENTX_CHIPS"),
                "the manifest's inputs to k3_1p1d.sh and k3_agentx_client_pod.sh",
            ),
        ),
        # Parent-only consistency check: no pipeline runs kimi-k3-tp32-ep; the
        # 1P1D config sources it and gates its eval on its baseline.
        Case(
            id="P16-parent-only",
            scenario="kimi-k3/tp32-ep",
            launch=lambda: written_step(
                f"{RUN_MULTIHOST} --topology 2x2x4 bash "
                "./scripts/vllm/benchmarking/run_eval_flow.sh --config "
                "kimi-k3-tp32-ep --run-lm-eval",
                {"TPU_ACCELERATOR_TYPE": "tpu7x"},
            ),
            roles=SLICE,
            driver="slice-0",
            results=results("kimi-k3-tp32-ep"),
        ),
        # agentx_sweep.sh boots its server with run_benchmarks.sh --keep-alive.
        Case(
            id="kimi-k3-agentx",
            scenario="kimi-k3/agentx",
            launch=lambda: written_step(
                f"{RUN_MULTIHOST} --topology 2x2x4 bash "
                "./scripts/vllm/benchmarking/run_benchmarks.sh --config "
                "kimi-k3-tp32-ep-agentx --results-dir "
                "perf_eval_results/kimi-k3-tp32-ep-agentx --keep-alive --port 8000",
                k3_sweep_env("agentx_sweep.sh"),
            ),
            roles=SLICE,
            driver="slice-0",
            results=("perf_eval_results/kimi-k3-tp32-ep-agentx",),
            points=("c1", "c2", "c4", "c8"),
            compared=SERVER_TASKS,
        ),
        # eval_conc_sweep.sh boots its server the same way.
        Case(
            id="kimi-k3-eval-conc-sweep",
            scenario="kimi-k3/eval-conc-sweep",
            launch=lambda: written_step(
                f"{RUN_MULTIHOST} --topology 2x2x4 bash "
                "./scripts/vllm/benchmarking/run_benchmarks.sh --config "
                "kimi-k3-tp32-ep-concsweep --results-dir "
                "perf_eval_results/k3-eval-conc-sweep --keep-alive --port 8000",
                k3_sweep_env("eval_conc_sweep.sh"),
            ),
            roles=SLICE,
            driver="slice-0",
            results=("perf_eval_results/k3-eval-conc-sweep",),
            compared=SERVER_TASKS,
        ),
        *qwen_cases(),
    ]
