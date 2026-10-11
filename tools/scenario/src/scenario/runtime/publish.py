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
"""After-tasks: the target's work on the results, run on the driver host.

A target passes them as data (SCENARIO_TARGET_DATA), so pods never run
target code. `when: result` tasks run after each measured perf cell whose
mode matches: the `require: always` ones first (artifacts), then the
failed-request check, then the `require: result_ok` ones if the cell is
ok. Publishing as results come in keeps the finished cells when a deadline
kills the run. `when: end` tasks run once after teardown, on failure too,
since artifacts matter most then; `require: commands_ok` ones only when
every command exited 0, which is today's upload rule.

Placeholders in argv and env values: {run_dir}, {file}, {label},
{unpublished}, {image}, {run_stamp}; other braces are left alone.
{unpublished} is the run directory's files that no `when: result` task has
already handed on, joined by `;`, so nothing is uploaded twice. A failed
fatal task makes a run that would exit 0 exit 6; a failed non-fatal task is
a warning.
"""

import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from scenario.runtime.log import Log
from scenario.runtime.proc import run as run_command
from scenario.runtime.results import Summary
from scenario.target import AfterTask

PLACEHOLDER = re.compile(r"\{(run_dir|file|label|unpublished|image|run_stamp)\}")


def fill(text: str, values: Mapping[str, str]) -> str:
    """`text` with the placeholders `values` has filled; the rest as written."""
    return PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), text)


class Publisher:
    """Runs a target's after-tasks for one run directory.

    `run_dir` is the run directory as composed from the results root, so
    paths handed to the target keep that root's form (a relative root gives
    artifact names under it).
    """

    def __init__(
        self,
        tasks: Sequence[AfterTask],
        *,
        run_dir: pathlib.Path,
        image: str,
        run_stamp: str,
        experiment: bool,
        env: Mapping[str, str],
        cwd: pathlib.Path,
        log: Log,
        summary: Summary | None,
    ) -> None:
        self.tasks = tuple(tasks)
        self.run_dir = run_dir
        self.image = image
        self.run_stamp = run_stamp
        self.experiment = experiment
        self.env = dict(env)
        self.cwd = cwd
        self.log = log
        self.summary = summary
        self.published: set[pathlib.Path] = set()
        self.fatal_failed = False

    def handle(self, path: pathlib.Path) -> str:
        """`path` in the run directory's form: under it as composed."""
        try:
            return str(
                self.run_dir / path.resolve().relative_to(self.run_dir.resolve())
            )
        except ValueError:
            return str(path)

    def result(
        self,
        *,
        mode: str,
        file: pathlib.Path,
        label: str,
        require: str,
    ) -> None:
        """Run the `when: result` tasks of one require stage for a cell.

        `require` is "always", or "result_ok" once the cell is known to be ok.
        """
        for task in self.tasks:
            if task.when != "result" or task.results not in ("any", mode):
                continue
            stage = "always" if task.require == "always" else "result_ok"
            if stage != require:
                continue
            values = {"file": self.handle(file), "label": label}
            if task.require == "commands_ok" and not self._commands_ok():
                self._record(task, "skipped", label=label, reason="a command failed")
                continue
            if self._run(task, values, label=label) and "{file}" in " ".join(
                (*task.argv, *(value for _, value in task.env))
            ):
                self.published.add(file.resolve())

    def end(self) -> None:
        for task in self.tasks:
            if task.when != "end":
                continue
            if task.require != "always" and not self._commands_ok():
                self._record(task, "skipped", reason="a command failed")
                continue
            self._run(task, {"unpublished": self.unpublished()})

    def unpublished(self) -> str:
        files = sorted(
            path
            for path in self.run_dir.rglob("*")
            if path.is_file() and path.resolve() not in self.published
        )
        return ";".join(self.handle(path) for path in files)

    def _commands_ok(self) -> bool:
        return self.summary is None or self.summary.commands_ok

    def _run(self, task: AfterTask, values: Mapping[str, str], label: str = "") -> bool:
        if task.skip_on_experiment and self.experiment:
            self._record(task, "skipped", label=label, reason="experiment")
            return False
        values = {
            "run_dir": str(self.run_dir),
            "image": self.image,
            "run_stamp": self.run_stamp,
            **values,
        }
        argv = [fill(word, values) for word in task.argv]
        env = {name: fill(value, values) for name, value in task.env}
        self.log.spawn(f"after:{task.name}", argv, {"env": env})
        done = run_command(
            argv,
            env={**self.env, **env},
            cwd=self.cwd,
            prefix=f"[after:{task.name}] ",
            timeout_s=task.timeout_s,
        )
        if done.ok:
            self._record(task, "ok", label=label, exit=0)
            return True
        status = "timeout" if done.timed_out else "failed"
        self._record(task, status, label=label, exit=done.code)
        what = "timed out" if done.timed_out else f"exited {done.code}"
        if task.fatal:
            self.fatal_failed = True
            problem = f"after-task {task.name} {what}"
            self.log.error(problem)
        else:
            problem = f"after-task {task.name} {what}; not failing the run"
            self.log.warning(problem)
        return False

    def _record(self, task: AfterTask, status: str, **details: Any) -> None:
        if self.summary is None:
            return
        entry = {"name": task.name, "when": task.when, "status": status}
        entry.update({key: value for key, value in details.items() if value != ""})
        self.summary.data["after_tasks"].append(entry)
        self.summary.save()
