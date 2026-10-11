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
"""Gates: check_regression.py's verdict on this run's results.

The repository's check_regression.py is loaded by path and called with the
Plan's baseline, the way its own command line calls it, so a verdict here is
the script's verdict on the same files; there is one copy of the rules. The
report goes to regression_check.md and eval_check_<key>.md, as today.

A soft gate's failure is a warning and leaves the exit code alone. An
experiment (a run with --set on server or task values) has no baselines in
its Plan, so its gates are skipped.
"""

import importlib.util
import pathlib
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

from scenario.plan.types import EvalRunPlan, GatePlan
from scenario.runtime.log import Log
from scenario.runtime.proc import capture_output

CHECK_REGRESSION = pathlib.Path("scripts/vllm/benchmarking/check_regression.py")
PERF_REPORT = "regression_check.md"
_MODULE = "scenario_check_regression"
_loaded: dict[pathlib.Path, ModuleType] = {}


class GateError(Exception):
    """check_regression.py cannot be loaded."""


def load_check_regression(repo_root: pathlib.Path) -> ModuleType:
    """The repository's check_regression.py as a module, loaded once per path.

    Raises:
        GateError: there is no such file under `repo_root`.
    """
    path = (repo_root / CHECK_REGRESSION).resolve()
    if path in _loaded:
        return _loaded[path]
    if not path.is_file():
        raise GateError(
            f"the gate needs {CHECK_REGRESSION} under {repo_root.resolve()}; run "
            "from the repository root"
        )
    spec = importlib.util.spec_from_file_location(_MODULE, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The script's dataclasses resolve their annotations through sys.modules.
    sys.modules[_MODULE] = module
    spec.loader.exec_module(module)
    _loaded[path] = module
    return module


Status = Literal["pass", "fail", "soft_fail", "skipped"]


@dataclass(frozen=True)
class Verdict:
    status: Status
    reason: str = ""

    @property
    def failed(self) -> bool:
        """Whether the verdict fails the run: a hard gate that did not pass."""
        return self.status == "fail"

    def to_data(self) -> dict[str, str]:
        return {
            "status": self.status,
            **({"reason": self.reason} if self.reason else {}),
        }


def _skip(
    gate: GatePlan | None, experiment: bool, what: str, log: Log
) -> Verdict | None:
    if gate is not None:
        return None
    if experiment:
        return Verdict("skipped", "experiment: gates are off")
    log.say(f"WARNING: no baseline for {what}; skipping the regression check.")
    return Verdict("skipped", "no baseline")


def _report(
    log: Log,
    report: pathlib.Path,
    text: str,
    failed: bool,
    soft: bool,
    messages: tuple[str, str],
) -> Verdict:
    """Write and print the report; `messages` are the soft and hard failure lines."""
    report.write_text(text)
    log.say(text.rstrip("\n"))
    if not failed:
        return Verdict("pass")
    if soft:
        log.warning(messages[0])
        return Verdict("soft_fail", "regressed")
    log.error(messages[1])
    return Verdict("fail", "regressed")


def perf_gate(
    *,
    repo_root: pathlib.Path,
    run_dir: pathlib.Path,
    gate: GatePlan | None,
    cell_results: dict[str, dict[str, Any]] | None,
    experiment: bool,
    log: Log,
) -> Verdict:
    """The perf gate over the run directory's results.

    `cell_results` are cells mode's results by label; None reads product
    cells as check_regression.py does, from isl*_osl*_c*.json.
    """
    skipped = _skip(gate, experiment, "perf", log)
    if skipped is not None:
        return skipped
    assert gate is not None
    log.say("=== Checking Perf Regression ===")
    try:
        check = load_check_regression(repo_root)
    except GateError as error:
        log.error(str(error))
        return Verdict("fail", str(error))
    with capture_output() as out:
        results = (
            cell_results
            if cell_results is not None
            else check.load_perf_results(run_dir)
        )
        if not results:
            print(
                f"ERROR: no isl*_osl*_c*.json files found in {run_dir}", file=sys.stderr
            )
            failed = True
        else:
            tolerance = gate.tolerance
            count = check.check_perf(results, gate.as_dict(), tolerance)
            failed = bool(count)
            if failed:
                beyond = f"{tolerance * 100:.2f}%"
                print(
                    f"FAILED: {count} metric(s) regressed beyond {beyond}",
                    file=sys.stderr,
                )
            else:
                print("PASS: all gated metrics within tolerance.")
    messages = (
        "Perf regression (soft gate, not failing the run). See logs above.",
        "Perf regression check failed! See logs above for details.",
    )
    return _report(
        log, run_dir / PERF_REPORT, out.getvalue(), failed, gate.soft, messages
    )


def eval_results(check: ModuleType, run_dir: pathlib.Path, run: EvalRunPlan) -> dict:
    """The run's results by its key.

    Without a concurrency sweep lm_eval writes under the run directory and
    the key is the task. In a sweep each run has its own directory, named
    after its key, and lm_eval names the result after the task.
    """
    if run.key == run.task:
        return check.load_eval_results(run_dir)
    found = check.load_eval_results(run_dir / run.key)
    return {run.key: found[run.task]} if run.task in found else {}


def eval_gate(
    *,
    repo_root: pathlib.Path,
    run_dir: pathlib.Path,
    run: EvalRunPlan,
    experiment: bool,
    log: Log,
) -> Verdict:
    skipped = _skip(run.gate, experiment, run.key, log)
    if skipped is not None:
        return skipped
    gate = run.gate
    assert gate is not None
    log.say(f"=== Checking Eval Regression for {run.key} ===")
    try:
        check = load_check_regression(repo_root)
    except GateError as error:
        log.error(str(error))
        return Verdict("fail", str(error))
    with capture_output() as out:
        results = eval_results(check, run_dir, run)
        if not results:
            print(f"ERROR: no results_*.json found under {run_dir}", file=sys.stderr)
            failed = True
        else:
            result = check.check_eval(results, gate.as_dict(), gate.tolerance)
            failed = bool(result)
            if failed:
                drops = [row.allowed_drop for row in result.failed_rows]
                print(
                    f"FAILED: {result.failed_count} task(s) regressed beyond "
                    f"{check.format_drop_range(drops)}",
                    file=sys.stderr,
                )
            else:
                print("PASS: all evaluated tasks within tolerance.")
    messages = (
        f"Eval regression for {run.key} (soft gate, not failing the run). See logs "
        "above.",
        f"Eval regression check failed for {run.key}! See logs above for details.",
    )
    report = run_dir / f"eval_check_{run.key}.md"
    return _report(log, report, out.getvalue(), failed, gate.soft, messages)
