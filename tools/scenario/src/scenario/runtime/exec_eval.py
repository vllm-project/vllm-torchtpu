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
"""The eval task: each eval run of the Plan, gated right after it runs.

lm_eval runs one task per run (and one concurrency, in a sweep) with the
Plan's argv. evalscope's report is translated into lm_eval's results shape,
as run_eval_flow.sh does, so the same gate reads it. InferenceX's eval
client runs, then its score validation, whose exit code is the gate. A
failed run does not stop the next; a dead server stops the task.

lm_eval's request cache follows run_eval_flow.sh. lm-eval builds every
request before sending the first, about 40 s per task with the server
idle, and the requests depend on the arguments, the lm-eval version and
the task definitions, not on the model. They are cached under a key of
exactly those, since lm-eval's own key leaves out --limit, --gen_kwargs and
the task files. The cache is a runner optimization, so its root comes from
the target or a flag; without one, lm_eval runs uncached.
"""

import hashlib
import json
import pathlib
import shutil
import subprocess
from collections.abc import Mapping
from typing import Any

from scenario.plan.types import EvalRunPlan, EvalTaskPlan
from scenario.runtime import gate
from scenario.runtime.context import ServerDied, TaskContext, TaskOutcome

LM_EVAL_VERSION = "import importlib.metadata as m; print(m.version('lm_eval'))"
# model_args entries that change no request: where requests go, and how many
# at once.
UNCACHED_MODEL_ARGS = ("model", "base_url", "num_concurrent")


def flag(argv: list[str], name: str) -> str | None:
    """The value after the last `name` in argv."""
    value = None
    for index, arg in enumerate(argv[:-1]):
        if arg == name:
            value = argv[index + 1]
    return value


def request_spec(argv: list[str]) -> list[str]:
    """What decides lm_eval's requests: its arguments without where they go.

    --output_path is dropped, and --model_args is replaced by its entries
    other than the endpoint, the served name and the concurrency; the
    served name counts again when a tokenizer setting makes requests
    depend on the model.
    """
    spec: list[str] = []
    model_args = ""
    rest = iter(argv[1:])
    for arg in rest:
        if arg in ("--output_path", "--model_args"):
            value = next(rest, "")
            if arg == "--model_args":
                model_args = value
            continue
        spec.append(arg)
    pairs = [pair for pair in model_args.split(",") if pair]
    extras = [
        pair for pair in pairs if pair.split("=", 1)[0] not in UNCACHED_MODEL_ARGS
    ]
    spec.append("extra_model_args=" + ",".join(extras))
    if "tokeniz" in ",".join(extras):
        spec += [pair for pair in pairs if pair.startswith("model=")]
    return spec


def task_files(root: pathlib.Path) -> list[str]:
    """sha256sum lines of every file under `root`, in name order."""
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  ./{path.relative_to(root).as_posix()}")
    return lines


def lm_eval_version(env: Mapping[str, str], cwd: pathlib.Path) -> str | None:
    try:
        done = subprocess.run(
            ["python3", "-c", LM_EVAL_VERSION],
            env=dict(env),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    version = done.stdout.strip()
    return version if done.returncode == 0 and version else None


def request_key(argv: list[str], version: str, cwd: pathlib.Path) -> str:
    """The cache key of one lm_eval run: version, request spec, task files."""
    lines = [version, *request_spec(argv)]
    include = flag(argv, "--include_path")
    if include is not None:
        lines += task_files(cwd / include)
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()[:16]


def request_cache(
    ctx: TaskContext, argv: list[str], env: Mapping[str, str]
) -> pathlib.Path | None:
    """The cache directory of this run, or None to run uncached.

    Best effort, as today: if an input cannot be read, the run is uncached.
    """
    if not ctx.lm_eval_cache:
        return None
    version = lm_eval_version(env, ctx.cwd)
    if version is None:
        ctx.log.say("WARNING: cannot read lm_eval's version; running uncached")
        return None
    try:
        cache = pathlib.Path(ctx.lm_eval_cache) / request_key(argv, version, ctx.cwd)
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        ctx.log.say(f"WARNING: no lm_eval request cache: {error}")
        return None
    return cache


class EvalRun:
    def __init__(self, ctx: TaskContext, run: EvalRunPlan, out: dict[str, Any]) -> None:
        self.ctx = ctx
        self.run = run
        self.out = out

    def lm_eval(self) -> bool:
        ctx, run = self.ctx, self.run
        argv = ctx.refs.argv(run.command.argv)
        cache = request_cache(ctx, argv, {**ctx.env, **ctx.refs.env(run.command.env)})
        extra_argv, extra_env = [], {}
        if cache is not None:
            extra_argv = ["--cache_requests", "true"]
            extra_env = {"LM_HARNESS_CACHE_PATH": str(cache)}
        ctx.log.say(
            f"Running lm_eval for task: {run.task} (request cache: {cache or 'off'})..."
        )
        done = ctx.run(
            f"eval {run.key}",
            run.command,
            log_path=ctx.run_dir / f"lm_eval_{run.key}.log",
            extra_argv=extra_argv,
            extra_env=extra_env,
        )
        if not self._ran(done.code):
            return False
        return self._gate()

    def evalscope(self) -> bool:
        ctx, run = self.ctx, self.run
        argv = ctx.refs.argv(run.command.argv)
        work = flag(argv, "--work-dir")
        work_dir = pathlib.Path(work) if work else ctx.run_dir
        if work:
            shutil.rmtree(work_dir, ignore_errors=True)
        done = ctx.run(
            f"eval {run.key}",
            run.command,
            log_path=ctx.run_dir / f"evalscope_{run.key}.log",
        )
        if not self._ran(done.code):
            return False
        reports = sorted(work_dir.glob(f"**/reports/*/{run.task}.json"))
        if not reports:
            problem = (
                f"evalscope did not produce a {run.task}.json report under {work_dir}"
            )
            ctx.log.error(problem)
            self.out.update(status="failed", reason="no evalscope report")
            return False
        report = json.loads(reports[0].read_text())
        translated = {"results": {run.task: {"acc,none": report["score"]}}}
        target = ctx.run_dir / f"results_{run.task}.json"
        target.write_text(json.dumps(translated, indent=2))
        ctx.log.say(f"{run.task}: score={report['score']} (num={report.get('num')})")
        return self._gate()

    def inferencex(self) -> bool:
        ctx, run = self.ctx, self.run
        done = ctx.run(
            f"eval {run.key}",
            run.command,
            log_path=ctx.run_dir / f"inferencex_{run.key}.log",
        )
        if not self._ran(done.code):
            return False
        if run.floor is None:
            verdict = gate.Verdict("skipped", "no score floor")
        elif ctx.experiment:
            verdict = gate.Verdict("skipped", "experiment: gates are off")
        else:
            floor = ctx.run(
                f"eval {run.key} floor",
                run.floor,
                log_path=ctx.run_dir / f"inferencex_{run.key}.floor.log",
            )
            if floor.ok:
                verdict = gate.Verdict("pass")
            else:
                problem = (
                    f"InferenceX's score floor failed for {run.key} (exit {floor.code})"
                )
                ctx.log.error(problem)
                verdict = gate.Verdict("fail", f"score floor exited {floor.code}")
        return self._judged(verdict)

    def _ran(self, code: int) -> bool:
        self.out["exit"] = code
        if code == 0:
            return True
        self.ctx.command_failed()
        self.out.update(
            status="failed",
            reason=f"{self.run.tool} exited {code}",
            gate=gate.Verdict("skipped", "the eval failed").to_data(),
        )
        return False

    def _gate(self) -> bool:
        verdict = gate.eval_gate(
            repo_root=self.ctx.cwd,
            run_dir=self.ctx.run_dir,
            run=self.run,
            experiment=self.ctx.experiment,
            log=self.ctx.log,
        )
        return self._judged(verdict)

    def _judged(self, verdict: gate.Verdict) -> bool:
        self.out.update(
            status="failed" if verdict.failed else "ok", gate=verdict.to_data()
        )
        return not verdict.failed


def run_eval(
    ctx: TaskContext, task: EvalTaskPlan, entry: dict[str, Any]
) -> TaskOutcome:
    runs: list[dict[str, Any]] = []
    entry.update(client=task.client, runs=runs)
    failed = False
    for run in task.runs:
        out: dict[str, Any] = {"key": run.key, "tool": run.tool, "status": "running"}
        runs.append(out)
        ctx.summary.save()
        executor = EvalRun(ctx, run, out)
        tools = {
            "lm_eval": executor.lm_eval,
            "evalscope": executor.evalscope,
            "inferencex": executor.inferencex,
        }
        try:
            ctx.alive()
            ok = tools[run.tool]()
            ctx.alive()
        except ServerDied:
            out.update(status="failed", reason="the server died")
            raise
        finally:
            ctx.summary.save()
        failed |= not ok
    return TaskOutcome("failed" if failed else "passed")
