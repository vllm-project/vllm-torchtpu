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
"""Custom tasks and the AgentX replay: one command each, under a timeout.

A custom task runs from the repository root with the env contract the Plan
gives it (the server URLs, SCENARIO_RESULTS_DIR and the rest of PLAN §3.8),
and logs to custom/<name>.log. It fails on a non-zero exit or on its
timeout, which counts as exit 124 as timeout(1) reports it; an exit of
skip_rest_if_exit_ge or more skips the remaining tasks.

AgentX runs InferenceX's agentic replay with the env the Plan derived for
this point. Its rows land in the run directory under RESULT_FILENAME. It has
no gate of the tool's: InferenceX fails the replay itself on its error rate.
aiperf's own verdict, submission_valid, is recorded and warned about, not
gated on, as InferenceX does.
"""

import json
import pathlib
from typing import Any

from scenario.plan.types import AgentxTaskPlan, CustomTaskPlan
from scenario.runtime.context import TaskContext, TaskOutcome
from scenario.runtime.proc import Finished

# Where the Plan's AgentX command env (P-21) tells the replay to write.
RESULT_DIR = "RESULT_DIR"
RESULT_FILENAME = "RESULT_FILENAME"
# aiperf's export, whose metadata says whether the run kept the AgentX rules.
AIPERF_EXPORT = "profile_export_aiperf.json"


def _finished(
    ctx: TaskContext, what: str, done: Finished, entry: dict[str, Any]
) -> bool:
    entry["exit"] = done.code
    if done.ok:
        return True
    ctx.command_failed()
    if done.timed_out:
        entry["reason"] = "timed out"
        ctx.log.say(f"ERROR: {what} timed out")
    else:
        entry["reason"] = f"exited {done.code}"
    return False


def run_custom(
    ctx: TaskContext, task: CustomTaskPlan, entry: dict[str, Any]
) -> TaskOutcome:
    done = ctx.run(
        f"custom {task.name}",
        task.command,
        log_path=ctx.run_dir / "custom" / f"{task.name}.log",
        timeout_s=task.timeout_s,
    )
    ok = _finished(ctx, task.name, done, entry)
    ctx.alive()
    skip = None
    limit = task.skip_rest_if_exit_ge
    if limit is not None and done.code >= limit:
        skip = (
            f"{task.name} exited {done.code}, at least its skip_rest_if_exit_ge {limit}"
        )
    return TaskOutcome("passed" if ok else "failed", skip_rest=skip)


def run_agentx(
    ctx: TaskContext, task: AgentxTaskPlan, entry: dict[str, Any]
) -> TaskOutcome:
    done = ctx.run(
        "agentx",
        task.command,
        log_path=ctx.run_dir / "agentx.log",
        timeout_s=task.timeout_s,
    )
    ok = _finished(ctx, "agentx", done, entry)
    ctx.alive()
    planned = ctx.refs.env(task.command.env)
    name = planned.get(RESULT_FILENAME, "")
    directory = pathlib.Path(planned.get(RESULT_DIR) or ctx.run_dir)
    rows = sorted(path.name for path in directory.glob(f"{name}*") if path.is_file())
    entry["results"] = rows
    if ok and not rows:
        entry["reason"] = f"no {name}* result in {directory}"
        ok = False
    valid = submission_valid(directory)
    entry["submission_valid"] = valid
    if valid is False:
        warning = (
            f"aiperf marked the run invalid (submission_valid false in "
            f"{AIPERF_EXPORT}); the numbers are not a valid AgentX result"
        )
        ctx.log.warning(warning)
    return TaskOutcome("passed" if ok else "failed")


def submission_valid(directory: pathlib.Path) -> bool | None:
    """aiperf's submission_valid, False if any export says so; None if none does."""
    found: list[bool] = []
    for path in sorted(directory.rglob(AIPERF_EXPORT)):
        try:
            metadata = json.loads(path.read_text()).get("metadata")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(metadata, dict) and isinstance(
            metadata.get("submission_valid"), bool
        ):
            found.append(metadata["submission_valid"])
    return all(found) if found else None
