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
"""The perf task: per cell, its warmup passes, then the measured run.

The Plan holds every client argv, prompt and warmup count included, so the
executor fills Refs and runs them. File names are run_benchmarks.sh's:
<cell>.json and <cell>.log for the measured run, <cell>.warmup<i>.json and
.log for each warmup pass; in cells mode the cell key is its label. A
failed warmup skips its cell's measured run. A failed cell does not stop
the next while the server lives; a dead server stops the task (exit 5).
The perf gate runs once the cells are done. A target's `when: result`
after-tasks run after each measured run that wrote its result.
"""

import datetime
import json
import pathlib
import time
from typing import Any

from scenario.plan.types import PerfCellPlan, PerfTaskPlan
from scenario.runtime import gate, results
from scenario.runtime.context import ServerDied, TaskContext, TaskOutcome

BENCHMARK_LOG = "benchmark.log"
RULE = "=" * 40


def _stamp() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _start_block(ctx: TaskContext) -> str:
    """run_benchmarks.sh's benchmark.log header, from config.json."""
    try:
        config = json.loads((ctx.run_dir / results.CONFIG).read_text())
    except (OSError, ValueError):
        config = {}
    ep = "true" if config.get("enable_ep") else "false"
    return "\n".join(
        [
            RULE,
            f"START TIME: {_stamp()}",
            f"MODEL: {ctx.plan.model_name}",
            f"CONFIG: {ctx.plan.scenario_name}",
            f"TP: {config.get('tensor_parallelism')}  "
            f"DP: {config.get('data_parallelism')}  EP: {ep}",
            RULE,
            "",
        ]
    )


def _end_block(code: int) -> str:
    return "\n".join(
        ["", RULE, f"END TIME: {_stamp()}", f"EXIT CODE: {code}", RULE, ""]
    )


def last_result(path: pathlib.Path) -> dict[str, Any] | None:
    """The result in a client's result file: its last JSON line, or the whole file."""
    try:
        text = path.read_text()
    except OSError:
        return None
    lines = [line for line in text.splitlines() if line.strip()]
    for candidate in (lines[-1] if lines else "", text):
        try:
            found = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(found, dict):
            return found
    return None


def _display(ctx: TaskContext, path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(ctx.run_dir.resolve()))
    except ValueError:
        return str(path)


class PerfRun:
    def __init__(self, ctx: TaskContext, task: PerfTaskPlan, entry: dict[str, Any]):
        self.ctx = ctx
        self.task = task
        self.entry = entry
        self.cells: list[dict[str, Any]] = []
        entry.update(client=task.client, mode=task.mode, cells=self.cells)

    def run(self) -> TaskOutcome:
        ctx = self.ctx
        log_path = ctx.run_dir / BENCHMARK_LOG
        log_path.write_text(_start_block(ctx))
        failed = False
        try:
            for index, cell in enumerate(self.task.cells):
                if index and self.task.pause_s:
                    time.sleep(self.task.pause_s)
                outcome = {"key": cell.key, "status": "running"}
                self.cells.append(outcome)
                try:
                    stop = self._cell(cell, outcome)
                except ServerDied:
                    outcome.update(status="failed", reason="the server died")
                    raise
                finally:
                    ctx.summary.save()
                failed |= outcome["status"] != "ok"
                if stop:
                    for rest in self.task.cells[index + 1 :]:
                        self.cells.append(
                            {"key": rest.key, "status": "skipped", "reason": stop}
                        )
                    break
        finally:
            with log_path.open("a") as log:
                log.write(_end_block(1 if failed else 0))
            ctx.summary.save()
        verdict = gate.perf_gate(
            repo_root=ctx.cwd,
            run_dir=ctx.run_dir,
            gate=self.task.gate,
            cell_results=self._labeled_results() if self.task.mode == "cells" else None,
            experiment=ctx.experiment,
            log=ctx.log,
        )
        self.entry["gate"] = verdict.to_data()
        return TaskOutcome("failed" if failed or verdict.failed else "passed")

    def _labeled_results(self) -> dict[str, dict[str, Any]]:
        """Cells mode's results by label, for the gate."""
        found = {}
        for cell in self.task.cells:
            result = last_result(pathlib.Path(self.ctx.refs.value(cell.result_file)))
            if result is not None:
                found[cell.key] = result
        return found

    def _cell(self, cell: PerfCellPlan, outcome: dict[str, Any]) -> str | None:
        """Run one cell into `outcome`; why the task stops after it, if it does."""
        ctx = self.ctx
        ctx.log.say(f"  {cell.key}")
        count = len(cell.warmups)
        for index, warmup in enumerate(cell.warmups, 1):
            ctx.log.say(f"    Warmup run {index}/{count}")
            done = ctx.run(
                f"perf {cell.key} warmup{index}",
                warmup,
                log_path=ctx.run_dir / f"{cell.key}.warmup{index}.log",
            )
            if not done.ok:
                ctx.log.say(f"    WARMUP FAILED (exit {done.code})")
                ctx.command_failed()
                outcome.update(
                    status="failed", reason=f"warmup {index} exited {done.code}"
                )
                ctx.alive()
                return None
            ctx.alive()
        done = ctx.run(
            f"perf {cell.key}", cell.measured, log_path=ctx.run_dir / f"{cell.key}.log"
        )
        result = pathlib.Path(ctx.refs.value(cell.result_file))
        outcome.update(exit=done.code, result=_display(ctx, result))
        if not done.ok:
            ctx.log.say(f"    FAILED (exit {done.code})")
            ctx.command_failed()
            outcome.update(status="failed", reason=f"the client exited {done.code}")
            ctx.alive()
            return None
        ctx.alive()
        found = last_result(result)
        if found is None:
            outcome.update(status="failed", reason=f"no result in {outcome['result']}")
            return None
        ctx.log.say(f"    OK -> {outcome['result']}")
        outcome["status"] = "ok"
        publisher = ctx.publisher
        if publisher is not None:
            publisher.result(
                mode=self.task.mode, file=result, label=cell.key, require="always"
            )
        limit = self.task.max_failed_requests
        if limit is not None and int(found.get("failed") or 0) > limit:
            failed = found.get("failed")
            ctx.log.say(
                f"ERROR: {cell.key} recorded failed requests: completed "
                f"{found.get('completed')}, failed {failed}"
            )
            ctx.command_failed()
            reason = f"{failed} failed requests, over the {limit} allowed"
            outcome.update(status="failed", reason=reason)
            return f"{cell.key} had {reason}"
        if publisher is not None:
            publisher.result(
                mode=self.task.mode, file=result, label=cell.key, require="result_ok"
            )
        return None


def run_perf(
    ctx: TaskContext, task: PerfTaskPlan, entry: dict[str, Any]
) -> TaskOutcome:
    return PerfRun(ctx, task, entry).run()
