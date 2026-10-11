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
"""Resolution: layers, extends, defaults, sweeps, --set, baselines, provenance."""

import pathlib
import textwrap

import pytest
from scenario.errors import DiagnosticError
from scenario.merge import TOOL, Origin
from scenario.resolve import Resolved, find_root, resolve

DEFAULTS = """\
schema: 1
server:
  image: us-docker.pkg.dev/test/images/vllm:test
  startup_timeout: 2h
  args:
    tensor-parallel-size: 1
    data-parallel-size: 1
    max-model-len: 16384
    max-num-seqs: 512
    api-server-count: 1
tasks:
  perf:
    dataset: {name: random}
    num_prompts: 320
    range_ratio_style: symmetric
    warmup_runs: 0
    percentile_metrics: [ttft, tpot, itl, e2el]
    tolerance: 0.05
    client_args: {backend: vllm, request-rate: inf}
  inferencex: {framework: vllm, spec_decoding: none, kv_offloading: none}
  agentx:
    duration: 1h
    timeout: 3h
    fast: false
  eval:
    lm_eval:
      model: local-chat-completions
      model_args: {num_concurrent: 128}
      seed: "0,1234,None,1234"
      limit: 100
      apply_chat_template: true
      fewshot_as_multiturn: true
"""

BASE = """\
schema: 1
name: glm-base
server:
  model: {name: zai-org/GLM, path: gs://bucket/glm}
  accelerator: tpu7x
  topology: 2x2x2
  hosts: 2
  chips_per_host: 4
  multihost: ray
  args:
    load-format: runai_streamer
    data-parallel-size: 16
    max-num-seqs: 64
    compilation-config: {backend: x, compile_sizes: [1, 8]}
tasks:
  perf:
    isl_osl: [[8192, 1024]]
    concurrency: [1024]
    random_range_ratio: 0.0
    baseline:
      isl8192_osl1024_c1024: {completed: 2048}
  eval:
    lm_eval: {tasks: [mmlu_pro]}
    tolerance: 0.025
    baseline:
      mmlu_pro: {"exact_match,custom-extract": 0.76}
"""

AGENTX = """\
schema: 1
name: k3-agentx
server:
  model: {name: moonshotai/Kimi-K3}
  accelerator: tpu7x
  topology: 2x2x4
  hosts: 4
  chips_per_host: 4
  multihost: ray
  args:
    tensor-parallel-size: 32
    max-model-len: null
    max-num-seqs: 16
tasks:
  inferencex: {model_prefix: kimik3, precision: fp4}
  agentx:
    concurrency: 1
    env: {AGENTIC_WARMUP_GRACE_PERIOD: "3600"}
sweep:
  name: c{tasks.agentx.concurrency}
  zip:
    tasks.agentx.concurrency: [1, 2, 4, 8]
    server.args.max-num-seqs: [16, 16, 16, 32]
"""


PD = """\
schema: 1
name: pd
server:
  model: {name: m}
  args: {max-model-len: 4096}
  roles:
    prefill:
      accelerator: tpu7x
      topology: 2x2x1
      hosts: 1
      chips_per_host: 4
      kv: {connector: TPURaidenConnector, role: producer}
    decode:
      accelerator: tpu7x
      topology: 2x2x1
      hosts: 1
      chips_per_host: 4
      kv: {connector: TPURaidenConnector, role: consumer}
    spare:
      accelerator: tpu7x
      topology: 2x2x1
      hosts: 1
      chips_per_host: 4
  router: {type: toy-proxy, command: [python3, proxy.py]}
tasks:
  driver: client
  custom:
    - {name: probe, command: ["true"], timeout: 1m}
sweep:
  points:
    - name: base
    - name: slim
      server:
        roles:
          spare: null
"""


class Repo:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.write("defaults", DEFAULTS)
        self.write("glm/base", BASE)

    def write(self, ref: str, text: str) -> pathlib.Path:
        path = self.root / "scenarios" / f"{ref}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
        return path

    def resolve(
        self, ref: str, *sets: str, point: str | None = None
    ) -> tuple[Resolved, ...]:
        return resolve(self.root / "scenarios" / f"{ref}.yaml", sets, point=point)

    def one(self, ref: str, *sets: str) -> Resolved:
        (resolved,) = self.resolve(ref, *sets)
        return resolved

    def errors(self, ref: str, *sets: str, point: str | None = None) -> list[str]:
        with pytest.raises(DiagnosticError) as info:
            self.resolve(ref, *sets, point=point)
        return [d.render() for d in info.value.diagnostics]


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> Repo:
    return Repo(tmp_path)


def test_base_resolves_with_defaults(repo: Repo) -> None:
    resolved = repo.one("glm/base")
    assert resolved.ref == "glm/base"
    assert resolved.point is None
    assert not resolved.experiment
    server = resolved.model.server
    assert server.image == "us-docker.pkg.dev/test/images/vllm:test"
    assert server.startup_timeout == 7200
    assert server.args["max-num-seqs"] == 64
    assert server.args["api-server-count"] == 1
    perf = resolved.model.tasks.perf
    assert perf is not None and perf.num_prompts == 320
    assert perf.client_args == {"backend": "vllm", "request-rate": "inf"}
    prov = resolved.provenance
    assert prov.origin(("server", "image")) == Origin(
        "defaults", "scenarios/defaults.yaml", 3
    )
    assert prov.origin(("server", "args", "max-num-seqs")) == Origin(
        "scenario", "scenarios/glm/base.yaml", 13
    )
    assert prov.superseded[("server", "args", "max-num-seqs")] == [
        Origin("defaults", "scenarios/defaults.yaml", 9)
    ]
    assert prov.origin(("tasks", "perf", "client")) == TOOL


def test_one_key_at_every_layer(repo: Repo) -> None:
    repo.write(
        "chain/gp",
        """\
        schema: 1
        name: gp
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
          args: {max-num-seqs: 100}
        """,
    )
    repo.write(
        "chain/p",
        "schema: 1\nname: p\nextends: chain/gp\nserver: {args: {max-num-seqs: 200}}\n",
    )
    repo.write(
        "chain/s",
        """\
        schema: 1
        name: s
        extends: ./p.yaml
        server: {args: {max-num-seqs: 300}}
        sweep:
          points:
            - name: a
              server: {args: {max-num-seqs: 400}}
        """,
    )
    (resolved,) = repo.resolve("chain/s", "server.args.max-num-seqs=500")
    assert resolved.point == "a"
    assert resolved.model.server.args["max-num-seqs"] == 500
    path = ("server", "args", "max-num-seqs")
    assert resolved.provenance.origin(path) == Origin(
        "set:0", "--set[0] server.args.max-num-seqs=500"
    )
    assert resolved.provenance.superseded[path] == [
        Origin("defaults", "scenarios/defaults.yaml", 9),
        Origin("extends:chain/gp", "scenarios/chain/gp.yaml", 9),
        Origin("extends:chain/p", "scenarios/chain/p.yaml", 4),
        Origin("scenario", "scenarios/chain/s.yaml", 4),
        Origin("sweep:a", "scenarios/chain/s.yaml", 8),
    ]
    assert resolved.experiment
    assert resolved.overrides == ("server.args.max-num-seqs=500",)
    assert resolved.data["extends"] == "chain/p"
    assert resolved.model.name == "s"
    assert "sweep" not in resolved.data


def test_defaults_only_for_declared_tasks(repo: Repo) -> None:
    repo.write(
        "plain",
        """\
        schema: 1
        name: plain
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          custom:
            - {name: probe, command: ["true"], timeout: 1m}
        """,
    )
    tasks = repo.one("plain").data["tasks"]
    assert set(tasks) == {"custom"}
    repo.write(
        "child",
        """\
        schema: 1
        name: child
        extends: glm/base
        tasks:
          perf: null
        """,
    )
    resolved = repo.one("child")
    assert resolved.model.tasks.perf is None
    assert resolved.model.tasks.eval is not None
    assert (
        resolved.provenance.deleted[("tasks", "perf")].source == "scenarios/child.yaml"
    )


def test_inferencex_eval_gets_no_lm_eval_defaults(repo: Repo) -> None:
    """lm_eval defaults need tasks.eval.lm_eval, which client inferencex lacks."""
    repo.write(
        "ix-eval",
        """\
        schema: 1
        name: ix-eval
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          inferencex: {model_prefix: kimik3, precision: fp4}
          eval:
            client: inferencex
            concurrency: 4
            timeout: 8h
        """,
    )
    tasks = repo.one("ix-eval").model.tasks
    evals = tasks.eval
    assert (evals.client, evals.lm_eval, evals.tolerance) == ("inferencex", None, None)
    assert tasks.inferencex.framework == "vllm"
    assert repo.one("glm/base").model.tasks.eval.lm_eval.model == (
        "local-chat-completions"
    )


def test_lm_eval_defaults_need_lm_eval(repo: Repo) -> None:
    repo.write(
        "evals",
        """\
        schema: 1
        name: evals
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          eval:
            tolerance: 0.02
            lm_eval:
              tasks: [gsm8k]
              concurrency: [1, 2]
        """,
    )
    lm_eval = repo.one("evals").model.tasks.eval.lm_eval
    assert lm_eval.model == "local-chat-completions"
    assert lm_eval.model_args == {}
    repo.write(
        "evals2",
        """\
        schema: 1
        name: evals2
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          eval:
            tolerance: 0.02
            lm_eval: {tasks: [gsm8k], concurrency: [1], model_args: {num_concurrent: 4}}
        """,
    )
    assert repo.errors("evals2") == [
        "scenarios/evals2.yaml:12: tasks.eval.lm_eval.model_args.num_concurrent: "
        "is set per run from tasks.eval.lm_eval.concurrency; remove it"
    ]


def test_gen_kwargs_replaces_the_lower_layer(repo: Repo) -> None:
    repo.write(
        "evals",
        """\
        schema: 1
        name: evals
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          eval:
            tolerance: 0.02
            lm_eval:
              tasks: [gsm8k]
              gen_kwargs: {chat_template_kwargs: {thinking: false}, max_gen_toks: 64}
        """,
    )
    repo.write(
        "evals-child",
        """\
        schema: 1
        name: evals-child
        extends: evals
        tasks:
          eval:
            lm_eval:
              gen_kwargs: {chat_template_kwargs: {enable_thinking: false}}
        """,
    )
    resolved = repo.one("evals-child", "tasks.eval.lm_eval.gen_kwargs.until=x")
    assert resolved.model.tasks.eval.lm_eval.gen_kwargs == {
        "chat_template_kwargs": {"enable_thinking": False},
        "until": "x",
    }
    path = ("tasks", "eval", "lm_eval", "gen_kwargs")
    assert resolved.provenance.origin(path).layer == "set:0"
    assert [o.layer for o in resolved.provenance.superseded[path]] == [
        "extends:evals",
        "scenario",
    ]


def test_inferencex_perf_gets_no_vllm_bench_defaults(repo: Repo) -> None:
    repo.write(
        "ix",
        """\
        schema: 1
        name: ix
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          perf:
            client: inferencex
            isl_osl: [[8192, 1024]]
            concurrency: [4]
            random_range_ratio: 0.8
            num_prompts: {per_concurrency: 10}
            num_warmups: {per_concurrency: 2}
        """,
    )
    perf = repo.one("ix").model.tasks.perf
    assert perf.client_args == {}
    assert perf.range_ratio_style is None
    assert perf.percentile_metrics == ["ttft", "tpot", "itl", "e2el"]
    assert perf.num_warmups.per_concurrency == 2


def test_agentx_and_identity_defaults_merge(repo: Repo) -> None:
    repo.write("k3-agentx", AGENTX)
    points = repo.resolve("k3-agentx")
    assert [p.point for p in points] == ["c1", "c2", "c4", "c8"]
    tasks = points[0].model.tasks
    assert tasks.inferencex is not None
    assert tasks.inferencex.model_dump(exclude_none=True) == {
        "model_prefix": "kimik3",
        "precision": "fp4",
        "framework": "vllm",
        "spec_decoding": "none",
        "kv_offloading": "none",
    }
    agentx = tasks.agentx
    assert (agentx.duration, agentx.timeout, agentx.fast) == (3600, 10800, False)


def test_identity_defaults_need_tasks_inferencex(repo: Repo) -> None:
    """tasks.inferencex defaults apply only where a layer declares it."""
    assert repo.one("glm/base").model.tasks.inferencex is None


def test_extends_merges_and_strips(repo: Repo) -> None:
    repo.write(
        "glm/child",
        """\
        schema: 1
        name: glm-child
        extends: ./base.yaml
        server:
          args:
            compilation-config: {compile_sizes: [1, 8, 1296]}
        """,
    )
    resolved = repo.one("glm/child")
    assert resolved.model.server.args["compilation-config"] == {
        "backend": "x",
        "compile_sizes": [1, 8, 1296],
    }
    assert resolved.model.name == "glm-child"
    assert resolved.data["extends"] == "glm/base"
    assert resolved.model.tasks.perf.baseline is None
    assert resolved.model.tasks.eval.baseline is None


SWEPT = """\
schema: 1
name: glm-sweep
extends: glm/base
sweep:
  name: s{server.args.max-num-seqs}
  zip:
    server.args.max-num-seqs: [64, 128]
  baselines:
    s64:
      perf:
        isl8192_osl1024_c1024: {completed: 2048}
"""


def test_point_baselines_are_not_inherited(repo: Repo) -> None:
    repo.write("glm/sweep", SWEPT)
    repo.write("glm/child", "schema: 1\nname: glm-child\nextends: glm/sweep\n")
    points = repo.resolve("glm/child")
    assert all(p.model.tasks.perf.baseline is None for p in points)
    assert repo.resolve("glm/sweep")[0].model.tasks.perf.baseline is not None


def test_extends_problems(repo: Repo) -> None:
    repo.write("loop/a", "schema: 1\nname: a\nextends: loop/b\n")
    repo.write("loop/b", "schema: 1\nname: b\nextends: loop/a\n")
    assert repo.errors("loop/a") == [
        "scenarios/loop/b.yaml:3: extends: extends cycle: loop/a -> loop/b -> loop/a"
    ]
    repo.write("typo", "schema: 1\nname: t\nextends: glm/bse\n")
    assert repo.errors("typo") == [
        "scenarios/typo.yaml:3: extends: no scenario glm/bse (did you mean glm/base?)"
    ]
    repo.write("out", "schema: 1\nname: o\nextends: ../../elsewhere.yaml\n")
    assert repo.errors("out") == [
        "scenarios/out.yaml:3: extends: ../../elsewhere.yaml is outside scenarios/"
    ]
    for depth in range(7):
        parent = f"extends: deep/d{depth + 1}\n" if depth < 6 else ""
        repo.write(f"deep/d{depth}", f"schema: 1\nname: d{depth}\n{parent}")
    (message,) = repo.errors("deep/d0")
    assert message.startswith(
        "scenarios/deep/d5.yaml:3: extends: extends chains are at most 5 deep"
    )


def test_typos_are_reported_where_they_are(repo: Repo) -> None:
    repo.write(
        "p",
        """\
        schema: 1
        name: p
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          perf:
            isl_osll: [[1, 1]]
        """,
    )
    repo.write("c", "schema: 1\nname: c\nextends: p\ntasks: {perf: null}\n")
    assert repo.errors("c") == [
        "scenarios/p.yaml:11: tasks.perf.isl_osll: unknown key; did you mean "
        "tasks.perf.isl_osl?"
    ]


def test_every_problem_in_one_run(repo: Repo) -> None:
    repo.write(
        "bad",
        """\
        schema: 1
        name: bad
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 2
          multihostt: mp
          env: {TPU_TOPOLOGY: 2x2x1}
        """,
    )
    assert repo.errors("bad", "server.argz.x=1", "name=x") == [
        "--set[0] server.argz.x=1: server.argz: unknown key; did you mean server.args?",
        "--set[1] name=x: --set changes server.* or tasks.* only",
        "scenarios/bad.yaml:6: server.topology: topology 2x2x1 has 4 chips, but "
        "hosts 1 x chips_per_host 2 = 2",
        "scenarios/bad.yaml:9: server.multihostt: unknown key; did you mean "
        "server.multihost?",
        "scenarios/bad.yaml:10: server.env.TPU_TOPOLOGY: TPU_TOPOLOGY is set by the "
        "tool (multi-host and backend wiring); remove it",
    ]


def test_missing_field_points_at_its_parent(repo: Repo) -> None:
    repo.write("nomodel", "schema: 1\nname: n\nserver:\n  hosts: 1\n")
    assert repo.errors("nomodel") == [
        "scenarios/nomodel.yaml:3: server.model: required"
    ]


def test_every_file_states_its_schema(repo: Repo) -> None:
    repo.write("noschema", "name: n\nextends: glm/base\n")
    assert repo.errors("noschema") == [
        "scenarios/noschema.yaml:1: schema: required: every scenario file starts "
        "with schema: 1"
    ]


def test_set_values_and_lists(repo: Repo) -> None:
    repo.write(
        "custom",
        """\
        schema: 1
        name: custom
        server:
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          custom:
            - {name: probe, command: ["true"], timeout: 1m}
        """,
    )
    resolved = repo.one(
        "custom",
        "tasks.custom[0].timeout=2h",
        "server.args.enable-expert-parallel=true",
        "server.env.X=1",
        "server.args.api-server-count=null",
    )
    assert resolved.model.tasks.custom[0].timeout == 7200
    assert resolved.model.server.args["enable-expert-parallel"] is True
    assert resolved.model.server.env == {"X": "1"}
    assert "api-server-count" not in resolved.model.server.args
    assert resolved.provenance.origin(("tasks", "custom")).layer == "set:0"
    assert repo.errors("custom", "tasks.custom[3].timeout=1h") == [
        "--set[0] tasks.custom[3].timeout=1h: tasks.custom[3].timeout: tasks.custom "
        "has no item [3]"
    ]


def test_sweep_points_resolve_one_by_one(repo: Repo) -> None:
    repo.write("k3-agentx", AGENTX)
    points = repo.resolve("k3-agentx")
    assert [p.model.tasks.agentx.concurrency for p in points] == [1, 2, 4, 8]
    assert [p.model.server.args["max-num-seqs"] for p in points] == [16, 16, 16, 32]
    assert all("sweep" not in p.data for p in points)
    prov = points[3].provenance
    assert prov.origin(("tasks", "agentx", "concurrency")) == Origin(
        "sweep:c8", "scenarios/k3-agentx.yaml", 22
    )
    repo.write("glm/sweep", SWEPT)
    swept = repo.resolve("glm/sweep")
    assert swept[0].provenance.origin(
        ("tasks", "perf", "baseline", "isl8192_osl1024_c1024", "completed")
    ) == Origin("sweep:s64", "scenarios/glm/sweep.yaml", 11)
    assert (
        swept[1].provenance.origin(
            ("tasks", "perf", "baseline", "isl8192_osl1024_c1024", "completed")
        )
        == TOOL
    )
    (only,) = repo.resolve("k3-agentx", point="c4")
    assert only.point == "c4"


def test_each_point_is_validated(repo: Repo) -> None:
    repo.write(
        "k3-agentx",
        AGENTX.replace("[16, 16, 16, 32]", "[16, 16, 16, 4]"),
    )
    assert repo.errors("k3-agentx") == [
        "scenarios/k3-agentx.yaml:23: server.args.max-num-seqs: max-num-seqs 4 x DP "
        "1 = 4, under tasks.agentx.concurrency 8; the server would cap the measured "
        "concurrency",
    ]


def test_named_points(repo: Repo) -> None:
    repo.write(
        "named",
        """\
        schema: 1
        name: named
        extends: glm/base
        tasks:
          eval: null
          perf:
            concurrency: [64]
        sweep:
          points:
            - name: tp8
              server: {args: {tensor-parallel-size: 8, data-parallel-size: 2}}
              tasks:
                perf:
                  baseline:
                    isl8192_osl1024_c64: {completed: 640}
            - name: dp16
        """,
    )
    tp8, dp16 = repo.resolve("named")
    assert tp8.model.server.args["tensor-parallel-size"] == 8
    assert tp8.provenance.origin(("server", "args", "tensor-parallel-size")) == Origin(
        "sweep:tp8", "scenarios/named.yaml", 11
    )
    assert tp8.model.tasks.perf.baseline is not None
    assert dp16.model.server.args["tensor-parallel-size"] == 1
    assert dp16.model.tasks.perf.baseline is None


def test_sweep_problems(repo: Repo) -> None:
    repo.write("k3-agentx", AGENTX)
    assert repo.errors("k3-agentx", "tasks.agentx.concurrency=16") == [
        "--set[0] tasks.agentx.concurrency=16: tasks.agentx.concurrency is the sweep "
        "axis tasks.agentx.concurrency; change the axis in the scenario, or run one "
        "point with --point"
    ]
    assert repo.errors("k3-agentx", point="c3") == [
        "scenarios/k3-agentx.yaml: no sweep point c3; points: c1, c2, c4, c8"
    ]
    assert repo.errors("glm/base", point="c1") == [
        "scenarios/glm/base.yaml: has no sweep; drop --point c1"
    ]
    repo.write(
        "axes",
        AGENTX.replace(
            "server.args.max-num-seqs:", "server.args.max-num-batched-tokens:"
        ),
    )
    assert repo.errors("axes") == [
        "scenarios/axes.yaml:23: sweep.zip[server.args.max-num-batched-tokens]: "
        "server.args.max-num-batched-tokens is not set in the scenario; an axis "
        "replaces a value the scenario has"
    ]
    repo.write(
        "axes",
        AGENTX + "  baselines:\n    c1:\n      perf: {isl1_osl1_c1: {completed: 1}}\n",
    )
    assert repo.errors("axes") == [
        "scenarios/axes.yaml:26: sweep.baselines.c1.perf: point c1 has no perf task"
    ]
    repo.write(
        "top",
        SWEPT.replace("name: glm-sweep", "name: top").replace(
            "sweep:\n",
            "tasks:\n  perf:\n    baseline: {isl8192_osl1024_c1024: {completed: 1}}\n"
            "sweep:\n",
        ),
    )
    assert repo.errors("top")[0] == (
        "scenarios/top.yaml:6: tasks.perf.baseline: is not allowed with a sweep; "
        "put it under sweep.baselines.<point>.<task>"
    )


EVAL_POINT = """\
schema: 1
name: k3-pair
server:
  model: {name: moonshotai/Kimi-K3}
  accelerator: tpu7x
  topology: 2x2x4
  hosts: 4
  chips_per_host: 4
  multihost: ray
  args:
    tensor-parallel-size: 32
    max-model-len: null
    max-num-seqs: 16
tasks:
  inferencex: {model_prefix: kimik3, precision: fp4}
  agentx: {timeout: 3h}
sweep:
  points:
    - name: c1
      tasks: {agentx: {concurrency: 1}}
    - name: c4
      tasks: {agentx: {concurrency: 4}}
    - name: gsm8k
      tasks:
        agentx: null
        eval:
          client: inferencex
          concurrency: CONC
          timeout: 8h
"""


def test_inferencex_eval_runs_at_the_top_agentx_concurrency(repo: Repo) -> None:
    """V-45 compares a point with every other point, --point or not."""
    repo.write("k3-pair", EVAL_POINT.replace("CONC", "4"))
    points = repo.resolve("k3-pair")
    assert [p.point for p in points] == ["c1", "c4", "gsm8k"]
    assert points[2].model.tasks.agentx is None
    assert all(p.warnings == () for p in points)
    repo.write("k3-pair", EVAL_POINT.replace("CONC", "2"))
    for found in (repo.resolve("k3-pair"), repo.resolve("k3-pair", point="gsm8k")):
        (warning,) = found[-1].warnings
        assert (warning.rule, warning.severity) == ("V-45", "warning")
        assert warning.render().startswith(
            "scenarios/k3-pair.yaml:28: warning: tasks.eval.concurrency: is 2, but "
            "the highest agentx concurrency among this scenario's points is 4"
        )


def test_points_may_change_the_pods(repo: Repo) -> None:
    repo.write("pd", PD)
    base, slim = repo.resolve("pd")
    assert list(base.model.server.roles) == ["prefill", "decode", "spare"]
    assert list(slim.model.server.roles) == ["prefill", "decode"]


def test_from_baselines(repo: Repo) -> None:
    repo.write(
        "glm/mp",
        """\
        schema: 1
        name: glm-mp
        extends: glm/base
        server: {multihost: mp}
        tasks:
          perf:
            baseline: {from: glm/base}
          eval:
            baseline: {from: glm/base}
        """,
    )
    resolved = repo.one("glm/mp")
    assert resolved.ref_baselines["perf"].source == "glm/base"
    assert dict(resolved.ref_baselines["perf"].baseline) == {
        "isl8192_osl1024_c1024": {"completed": 2048}
    }
    assert dict(resolved.ref_baselines["eval"].baseline) == {
        "mmlu_pro": {"exact_match,custom-extract": 0.76}
    }
    repo.write(
        "glm/mp2",
        """\
        schema: 1
        name: glm-mp2
        extends: glm/base
        tasks:
          perf:
            concurrency: [512]
            baseline: {from: glm/base}
          eval:
            baseline: {from: glm/mp}
        """,
    )
    assert repo.errors("glm/mp2") == [
        "scenarios/glm/mp2.yaml:7: tasks.perf.baseline.from: glm/base has perf "
        "baseline keys isl8192_osl1024_c1024, but this task's keys are "
        "isl8192_osl1024_c512",
        "scenarios/glm/mp2.yaml:9: tasks.eval.baseline.from: glm/mp takes its eval "
        "baseline from glm/base; refer to that scenario directly",
    ]
    repo.write(
        "glm/mp3",
        "schema: 1\nname: m3\nextends: glm/base\n"
        "tasks: {eval: {baseline: {from: glm/bas}}}\n",
    )
    assert repo.errors("glm/mp3") == [
        "scenarios/glm/mp3.yaml:4: tasks.eval.baseline.from: no scenario glm/bas "
        "(did you mean glm/base, glm/mp, glm/mp3?)"
    ]


def test_from_baseline_of_an_eval_sweep(repo: Repo) -> None:
    sweep = """\
        schema: 1
        name: {name}
        server:
          model: {{name: m}}
          accelerator: tpu7x
          topology: 2x2x1
          hosts: 1
          chips_per_host: 4
        tasks:
          eval:
            tolerance: 0.02
            lm_eval: {{tasks: [gsm8k], concurrency: [1, 2]}}
            baseline: {baseline}
        """
    repo.write(
        "ev/a", sweep.format(name="a", baseline='{gsm8k_c2: {"exact_match,none": 0.9}}')
    )
    repo.write("ev/b", sweep.format(name="b", baseline="{from: ev/a}"))
    assert set(repo.one("ev/b").ref_baselines["eval"].baseline) == {"gsm8k_c2"}


def test_warnings_do_not_fail(repo: Repo) -> None:
    repo.write("k3-agentx", AGENTX)
    points = repo.resolve("k3-agentx", point="c8")
    assert points[0].warnings == ()
    repo.write(
        "k3-warn",
        AGENTX.replace("[16, 16, 16, 32]", "[16, 16, 16, 12]").replace(
            "name: k3-agentx", "name: k3-warn"
        ),
    )
    (warned,) = repo.resolve("k3-warn", point="c8")
    (warning,) = warned.warnings
    assert warning.severity == "warning"
    assert warning.rule == "V-32"


def test_root_discovery(repo: Repo, tmp_path_factory: pytest.TempPathFactory) -> None:
    nested = repo.root / "scenarios" / "glm" / "base.yaml"
    assert find_root(nested) == repo.root
    scratch_dir = tmp_path_factory.mktemp("scratch")
    scratch = scratch_dir / "exp.yaml"
    scratch.write_text("schema: 1\nname: exp\nextends: glm/base\n")
    with pytest.raises(DiagnosticError, match="pass --root"):
        find_root(scratch)
    assert find_root(scratch, cwd=repo.root) == repo.root
    (resolved,) = resolve(scratch, root=repo.root)
    assert resolved.ref == f"@{scratch.resolve()}"
    assert resolved.model.name == "exp"


K3_TP32 = """\
schema: 1
name: kimi-k3-tp32-ep
server:
  model:
    name: moonshotai/Kimi-K3
    path: gs://tpu-commons-ci/moonshootai/kimi/k3
  accelerator: tpu7x
  topology: 2x2x4
  hosts: 4
  chips_per_host: 4
  multihost: ray
  startup_timeout: 4h
  args:
    load-format: runai_streamer
    tensor-parallel-size: 32
    enable-expert-parallel: true
    max-model-len: 10240
    max-num-seqs: 8
    trust-remote-code: true
    limit-mm-per-prompt: {image: 0, video: 0}
    block-size: 256
    compilation-config: {compile_sizes: [1, 8, 8192]}
tasks:
  perf:
    isl_osl: [[8192, 1024]]
    concurrency: [1]
    num_prompts: 64
    random_range_ratio: 0.0
    temperature: 0
    baseline:
      _source:
        description: Values from a nightly build.
        previous_values: {median_tpot_ms: 18.95}
      isl8192_osl1024_c1:
        completed: 64
        num_prompts: 64
        median_tpot_ms: 19.2
  eval:
    lm_eval:
      tasks: [mmlu_pro]
      model_args: {num_concurrent: 16}
      fewshot_as_multiturn: false
      gen_kwargs: {chat_template_kwargs: {thinking: false}}
    tolerance: 0.025
    baseline:
      mmlu_pro: {"exact_match,custom-extract": 0.834}
"""

K3_1P1D = """\
schema: 1
name: kimi-k3-tp8-dp4-ep-1p1d
extends: kimi-k3/tp32-ep
server:
  accelerator: null
  topology: null
  hosts: null
  chips_per_host: null
  multihost: mp
  scratch: 64Gi
  args:
    tensor-parallel-size: 8
    data-parallel-size: 4
    max-num-batched-tokens: 1296
    compilation-config: {compile_sizes: [1, 8, 1296]}
    distributed-executor-backend: mp
  env:
    TPU_RAIDEN_DST_SHARDS: "8"
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: "10800"
  roles:
    prefill:
      accelerator: tpu7x
      topology: 2x2x4
      hosts: 4
      chips_per_host: 4
      kv: {connector: TPURaidenConnector, role: producer}
    decode:
      accelerator: tpu7x
      topology: 2x2x4
      hosts: 4
      chips_per_host: 4
      kv: {connector: TPURaidenConnector, role: consumer}
  router:
    type: toy-proxy
    command: [python3, examples/disagg/toy_proxy_server.py]
tasks:
  driver: prefill
  perf: null
  custom:
    - name: correctness
      command: [bash, scripts/vllm/integration/run_k3_1p1d_disagg_correctness.sh]
      env:
        PREFILL_CHUNK_TOKENS: "1296"
        LOG_DIR: "${SCENARIO_RESULTS_DIR}/k3_1p1d_correctness"
      timeout: 4h
      skip_rest_if_exit_ge: 2
  eval:
    baseline: {from: kimi-k3/tp32-ep}
"""


def test_drafts_from_the_plan(repo: Repo) -> None:
    repo.write("kimi-k3/tp32-ep", K3_TP32)
    repo.write("kimi-k3/tp8-dp4-ep-1p1d", K3_1P1D)
    tp32 = repo.one("kimi-k3/tp32-ep")
    assert tp32.model.tasks.perf.baseline.source["description"].startswith("Values")
    pd = repo.one("kimi-k3/tp8-dp4-ep-1p1d")
    server = pd.model.server
    assert (server.accelerator, server.topology, server.hosts) == (None,) * 3
    assert server.chips_per_host is None
    assert list(server.roles) == ["prefill", "decode"]
    assert server.args["compilation-config"] == {"compile_sizes": [1, 8, 1296]}
    assert server.startup_timeout == 4 * 3600
    assert pd.model.tasks.perf is None
    assert pd.model.tasks.eval.lm_eval.model_args == {"num_concurrent": 16}
    assert pd.ref_baselines["eval"].source == "kimi-k3/tp32-ep"
    prefill, decode = server.role_views()
    assert (prefill.hardware.hosts, prefill.args["data-parallel-size"]) == (4, 4)
    assert pd.provenance.deleted[("server", "hosts")] == Origin(
        "scenario", "scenarios/kimi-k3/tp8-dp4-ep-1p1d.yaml", 7
    )
