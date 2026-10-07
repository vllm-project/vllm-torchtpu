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
"""From a scenario file to resolved, validated points.

Layers, later wins:

    tool (model defaults) < scenarios/defaults.yaml < extends chain, root
    parent first < the scenario < sweep point < --set, in order

Defaults for a task apply only when a later layer declares that task, and
perf defaults that only mean something for vllm bench serve stay out of an
inferencex perf task. `name`, `extends` and baselines are not inherited. A
sweep becomes one resolved scenario per point. Resolution reads only the
files and its arguments: no environment, clock, network or subprocess.
"""

import copy
import dataclasses
import difflib
import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from scenario.errors import (
    Diagnostic,
    DiagnosticError,
    Issue,
    issues_from_validation_error,
    sort_diagnostics,
)
from scenario.keycheck import check_keys
from scenario.load import LoadedFile, load_file
from scenario.merge import (
    AssignError,
    Origin,
    OriginOf,
    Provenance,
    assign,
    merge_into,
)
from scenario.model.baseline import BaselineRef
from scenario.model.rules import (
    baseline_key_problem,
    baseline_keys,
    check_scenario,
    check_sweep,
    sweep_baseline_message,
)
from scenario.model.scenario import SCHEMA_VERSION, Defaults, Scenario
from scenario.model.sweep import ExpandedPoint, Sweep, expand
from scenario.model.tasks import BUILTIN_TASKS, VLLM_BENCH_ONLY_DEFAULTS
from scenario.overrides import Override, OverrideError, parse_override, sparse_layer
from scenario.paths import KeyPath, format_path, is_prefix, parse_path

SCENARIOS_DIR = "scenarios"
DEFAULTS_FILE = "defaults.yaml"
MAX_EXTENDS_DEPTH = 5


@dataclass(frozen=True)
class RefBaseline:
    """An inline baseline copied from another scenario by `from:`."""

    source: str
    baseline: Mapping[str, Any]


@dataclass(frozen=True)
class Resolved:
    """One point of a scenario (the whole scenario when it has no sweep)."""

    ref: str
    file: pathlib.Path
    point: str | None
    data: Mapping[str, Any]
    model: Scenario
    provenance: Provenance
    overrides: tuple[str, ...]
    ref_baselines: Mapping[str, RefBaseline]
    warnings: tuple[Diagnostic, ...] = ()
    # Layer names, lowest first: tool, defaults, extends:<ref>..., scenario,
    # sweep:<point>, set:<i>...
    layers: tuple[str, ...] = ()

    @property
    def experiment(self) -> bool:
        """A run with --set overrides skips its gates and is tagged."""
        return bool(self.overrides)


def find_root(
    file: pathlib.Path,
    root: pathlib.Path | None = None,
    cwd: pathlib.Path | None = None,
) -> pathlib.Path:
    """The directory holding scenarios/defaults.yaml.

    It is `root` when given, else the nearest ancestor of `file`, else of
    `cwd` (which the CLI passes, for scratch files outside the repo).
    """
    if root is not None:
        candidates = [root]
    else:
        candidates = list(file.resolve().parents)
        if cwd is not None:
            candidates += [cwd, *cwd.parents]
    for candidate in candidates:
        if (candidate / SCENARIOS_DIR / DEFAULTS_FILE).is_file():
            return candidate
    where = f" in {root}" if root is not None else " above this file; pass --root"
    message = f"no {SCENARIOS_DIR}/{DEFAULTS_FILE}{where}"
    raise DiagnosticError([Diagnostic(str(file), None, "", message)])


def resolve(
    file: pathlib.Path,
    sets: Sequence[str] = (),
    *,
    root: pathlib.Path | None = None,
    cwd: pathlib.Path | None = None,
    point: str | None = None,
) -> tuple[Resolved, ...]:
    """Resolve and validate every point of a scenario file (or one, `point`).

    Raises:
        DiagnosticError: every problem found, in every file and point.
    """
    resolver = Resolver(find_root(file, root, cwd))
    return resolver.resolve(file.resolve(), tuple(sets), point)


@dataclass(frozen=True)
class _FileLayer:
    path: pathlib.Path
    ref: str
    layer: str
    loaded: LoadedFile
    data: dict[str, Any]

    def origin_of(self, layer: str | None = None, base: KeyPath = ()) -> OriginOf:
        """Origins of this file's data, or of the overlay at `base` in it."""
        name = layer or self.layer
        loaded = self.loaded
        return lambda path: Origin(name, loaded.source, loaded.line((*base, *path)))

    def origin(self, path: KeyPath) -> Origin:
        return self.origin_of()(path)


@dataclass(frozen=True)
class _Merge:
    """A step that merges a layer over the data built so far."""

    data: dict[str, Any]
    origin_of: OriginOf


@dataclass(frozen=True)
class _Assign:
    """A step that sets one path (a --set or a sweep axis)."""

    path: KeyPath
    value: Any
    origin: Origin


_Step = _Merge | _Assign


@dataclass
class _Report:
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def at(self, origin: Origin, issue: Issue) -> None:
        self.diagnostics.append(
            Diagnostic(
                origin.source,
                origin.line,
                format_path(issue.path),
                issue.message,
                hint=issue.hint,
                rule=issue.rule,
                severity=issue.severity,
            )
        )

    def in_file(self, layer: _FileLayer, issues: Sequence[Issue]) -> None:
        for issue in issues:
            self.at(layer.origin(issue.path), issue)

    def located(self, prov: Provenance, issues: Sequence[Issue]) -> None:
        for issue in issues:
            self.at(prov.locate(issue.path), issue)

    def plain(self, source: str, message: str, rule: str | None = None) -> None:
        self.diagnostics.append(Diagnostic(source, None, "", message, rule=rule))

    @property
    def failed(self) -> bool:
        return self.failed_since(0)

    def failed_since(self, mark: int) -> bool:
        return any(d.severity == "error" for d in self.diagnostics[mark:])


def _get(data: Any, path: KeyPath) -> Any:
    node = data
    for key in path:
        in_dict = isinstance(node, dict) and key in node
        in_list = isinstance(node, list) and isinstance(key, int) and key < len(node)
        if not (in_dict or in_list):
            return None
        node = node[key]
    return node


def _has(data: Any, path: KeyPath) -> bool:
    *parents, last = path
    node = _get(data, tuple(parents))
    if isinstance(node, dict):
        return last in node
    return isinstance(node, list) and isinstance(last, int) and last < len(node)


def _without(data: Any, paths: Sequence[KeyPath]) -> Any:
    data = copy.deepcopy(data)
    for path in paths:
        del _get(data, path[:-1])[path[-1]]
    return data


def _drop_baseline(tasks: Any) -> Any:
    if not isinstance(tasks, dict):
        return tasks
    return {
        name: (
            {k: v for k, v in task.items() if k != "baseline"}
            if name in BUILTIN_TASKS and isinstance(task, dict)
            else task
        )
        for name, task in tasks.items()
    }


def _strip_inherited(data: dict[str, Any]) -> dict[str, Any]:
    """A parent's layer without what a child never inherits."""
    data = {k: v for k, v in data.items() if k not in ("name", "extends")}
    if "tasks" in data:
        data["tasks"] = _drop_baseline(data["tasks"])
    sweep = data.get("sweep")
    if isinstance(sweep, dict):
        sweep = {k: v for k, v in sweep.items() if k != "baselines"}
        if isinstance(sweep.get("points"), list):
            sweep["points"] = [
                {**point, "tasks": _drop_baseline(point["tasks"])}
                if isinstance(point, dict) and "tasks" in point
                else point
                for point in sweep["points"]
            ]
        data["sweep"] = sweep
    return data


def _scoped_defaults(defaults: dict[str, Any], upper: dict[str, Any]) -> dict[str, Any]:
    """The defaults layer for a scenario whose own layers merge to `upper`.

    Task defaults apply only to declared tasks; tasks.perf, tasks.agentx,
    tasks.eval and tasks.eval.lm_eval count separately. Defaults that a
    declared setting replaces wholesale stay out: vllm-bench-only perf
    entries for another client, and num_concurrent when lm_eval sweeps
    concurrency.
    """
    scoped = copy.deepcopy(defaults)
    tasks = scoped.get("tasks")
    if not isinstance(tasks, dict):
        return scoped
    for name in BUILTIN_TASKS:
        if _get(upper, ("tasks", name)) is None:
            tasks.pop(name, None)
    perf = tasks.get("perf")
    client = _get(upper, ("tasks", "perf", "client")) or "vllm-bench"
    if isinstance(perf, dict) and client != "vllm-bench":
        for key in VLLM_BENCH_ONLY_DEFAULTS:
            perf.pop(key, None)
    evals = tasks.get("eval")
    if isinstance(evals, dict):
        if _get(upper, ("tasks", "eval", "lm_eval")) is None:
            evals.pop("lm_eval", None)
        model_args = _get(evals, ("lm_eval", "model_args"))
        sweeps = _get(upper, ("tasks", "eval", "lm_eval", "concurrency")) is not None
        if isinstance(model_args, dict) and sweeps:
            model_args.pop("num_concurrent", None)
    return scoped


def _schema_issues(data: dict[str, Any]) -> list[Issue]:
    """Every scenario file states its schema version (V-01)."""
    if "schema" not in data:
        message = f"required: every scenario file starts with schema: {SCHEMA_VERSION}"
        return [Issue(("schema",), message, rule="V-01")]
    version = data["schema"]
    if version != SCHEMA_VERSION or isinstance(version, bool):
        message = (
            f"schema {version!r} is not supported; this tool reads schema "
            f"{SCHEMA_VERSION}"
        )
        return [Issue(("schema",), message, rule="V-01")]
    return []


@dataclass(frozen=True)
class _Point:
    """One point's layer: steps that may declare tasks, then its baselines."""

    name: str | None
    steps: list[_Step]
    baselines: dict[str, tuple[_Step, KeyPath]]


class Resolver:
    """Resolves scenarios under one root, sharing loaded files."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.scenarios = root / SCENARIOS_DIR
        self._files: dict[pathlib.Path, LoadedFile | DiagnosticError] = {}
        self._refs: dict[str, tuple[Resolved, ...] | DiagnosticError] = {}

    # Files and reference ids ---------------------------------------------

    def scenario_files(self) -> list[pathlib.Path]:
        """Every scenario under scenarios/, by reference id."""
        return [self.file_of(ref) for ref in self.all_refs()]

    def declared_name(self, file: pathlib.Path) -> tuple[str, int] | None:
        """The file's own `name` and its line, or None when it has none."""
        try:
            loaded = self._load(file)
        except DiagnosticError:
            return None
        name = loaded.data.get("name") if isinstance(loaded.data, dict) else None
        return (name, loaded.line(("name",))) if isinstance(name, str) else None

    def ref_of(self, file: pathlib.Path) -> str:
        """The path under scenarios/ without .yaml; `@<path>` outside it."""
        if file.is_relative_to(self.scenarios):
            return file.relative_to(self.scenarios).with_suffix("").as_posix()
        return f"@{file}"

    def file_of(self, ref: str) -> pathlib.Path:
        return self.scenarios / f"{ref}.yaml"

    def source_of(self, file: pathlib.Path) -> str:
        if file.is_relative_to(self.root):
            return file.relative_to(self.root).as_posix()
        return str(file)

    def all_refs(self) -> list[str]:
        return sorted(
            self.ref_of(path)
            for path in self.scenarios.rglob("*.yaml")
            if path.name != DEFAULTS_FILE or path.parent != self.scenarios
        )

    def _near(self, ref: str) -> str:
        matches = difflib.get_close_matches(ref, self.all_refs(), n=3, cutoff=0.5)
        return f" (did you mean {', '.join(matches)}?)" if matches else ""

    def _load(self, file: pathlib.Path) -> LoadedFile:
        if file not in self._files:
            try:
                self._files[file] = load_file(file, self.source_of(file))
            except DiagnosticError as error:
                self._files[file] = error
        loaded = self._files[file]
        if isinstance(loaded, DiagnosticError):
            raise loaded
        return loaded

    def _layer(
        self,
        file: pathlib.Path,
        layer: str,
        model: type[BaseModel],
        report: _Report,
    ) -> _FileLayer | None:
        """A file as a layer: loaded, key-checked, unknown keys left out."""
        try:
            loaded = self._load(file)
        except DiagnosticError as error:
            report.diagnostics.extend(error.diagnostics)
            return None
        unknown = check_keys(loaded.data, model)
        data = _without(loaded.data, [issue.path for issue in unknown])
        file_layer = _FileLayer(file, self.ref_of(file), layer, loaded, data)
        report.in_file(file_layer, [*unknown, *_schema_issues(loaded.data)])
        return file_layer

    # The extends chain ----------------------------------------------------

    def _parent(self, layer: _FileLayer, report: _Report) -> str | None:
        text = layer.data["extends"]
        problem: str | None = None
        ref = text if isinstance(text, str) else ""
        if not isinstance(text, str):
            problem = "expected a scenario reference"
        elif text.startswith(("./", "../")):
            target = (layer.path.parent / text).resolve()
            if target.is_relative_to(self.scenarios):
                ref = self.ref_of(target)
            else:
                problem = f"{text} is outside {SCENARIOS_DIR}/"
        if problem is None and not self.file_of(ref).is_file():
            problem = f"no scenario {ref}{self._near(ref)}"
        if problem is not None:
            report.in_file(layer, [Issue(("extends",), problem, rule="V-29")])
            return None
        return ref

    def _chain(self, file: pathlib.Path, report: _Report) -> list[_FileLayer] | None:
        """The scenario's layer, then its parents', nearest first (V-29)."""
        layer = self._layer(file, "scenario", Scenario, report)
        if layer is None:
            return None
        chain = [layer]
        while layer.data.get("extends") is not None:
            ref = self._parent(layer, report)
            if ref is None:
                return None
            refs = [link.ref for link in chain]
            if ref in refs or len(chain) > MAX_EXTENDS_DEPTH:
                walk = " -> ".join([*refs, ref])
                message = (
                    f"extends cycle: {walk}"
                    if ref in refs
                    else f"extends chains are at most {MAX_EXTENDS_DEPTH} deep: {walk}"
                )
                report.in_file(layer, [Issue(("extends",), message, rule="V-29")])
                return None
            parent = self._layer(self.file_of(ref), f"extends:{ref}", Scenario, report)
            if parent is None:
                return None
            chain.append(parent)
            layer = parent
        return chain

    # Resolution -----------------------------------------------------------

    def resolve(
        self,
        file: pathlib.Path,
        sets: tuple[str, ...] = (),
        point: str | None = None,
        follow_refs: bool = True,
    ) -> tuple[Resolved, ...]:
        """Resolve every point of `file`; see `resolve()`."""
        report = _Report()
        results = self._resolve(file, sets, point, follow_refs, report)
        diagnostics = sort_diagnostics(dict.fromkeys(report.diagnostics))
        if report.failed:
            raise DiagnosticError(diagnostics)
        warnings = tuple(d for d in diagnostics if d.severity == "warning")
        return tuple(dataclasses.replace(r, warnings=warnings) for r in results)

    def _resolve(
        self,
        file: pathlib.Path,
        sets: tuple[str, ...],
        point: str | None,
        follow_refs: bool,
        report: _Report,
    ) -> list[Resolved]:
        overrides = self._overrides(sets, report)
        defaults = self._layer(
            self.scenarios / DEFAULTS_FILE, "defaults", Defaults, report
        )
        chain = self._chain(file, report)
        if defaults is None or chain is None:
            return []
        try:
            Defaults.model_validate(defaults.data)
        except ValidationError as error:
            report.in_file(defaults, issues_from_validation_error(error, defaults.data))

        scenario, parents = chain[0], chain[1:]
        own = copy.deepcopy(scenario.data)
        if parents:
            own["extends"] = parents[0].ref
        else:
            own.pop("extends", None)
        file_steps: list[_Step] = [
            _Merge(_strip_inherited(layer.data), layer.origin_of())
            for layer in reversed(parents)
        ]
        file_steps.append(_Merge(own, scenario.origin_of()))
        upper, upper_prov = self._apply(file_steps, report)
        plain_steps: list[_Step] = [
            _Merge({k: v for k, v in step.data.items() if k != "sweep"}, step.origin_of)
            for step in file_steps
            if isinstance(step, _Merge)
        ]

        if "sweep" not in upper:
            if point is not None:
                report.plain(
                    scenario.loaded.source, f"has no sweep; drop --point {point}"
                )
                return []
            points = [_Point(None, [], {})]
        else:
            found = self._points(
                upper,
                upper_prov,
                defaults,
                plain_steps,
                chain,
                overrides,
                point,
                report,
            )
            if found is None:
                return []
            points = found

        set_steps: list[_Step] = [
            _Assign(o.path, o.value, Origin(f"set:{o.index}", o.source))
            for o in overrides
        ]
        results = []
        for current in points:
            declared, _ = self._apply(
                [*plain_steps, *current.steps, *set_steps], _Report()
            )
            baseline_steps = []
            for task, (step, where) in current.baselines.items():
                if _get(declared, ("tasks", task)) is None:
                    message = f"point {current.name} has no {task} task"
                    report.located(upper_prov, [Issue(where, message, rule="V-38")])
                else:
                    baseline_steps.append(step)
            scoped = _scoped_defaults(defaults.data, declared)
            steps = [
                _Merge(scoped, defaults.origin_of()),
                *plain_steps,
                *current.steps,
                *baseline_steps,
                *set_steps,
            ]
            data, prov = self._apply(steps, report)
            resolved = self._validate(
                scenario, current.name, data, prov, overrides, follow_refs, report
            )
            if resolved is not None:
                layers = (
                    "tool",
                    "defaults",
                    *(f"extends:{layer.ref}" for layer in reversed(parents)),
                    "scenario",
                    *([f"sweep:{current.name}"] if current.name is not None else []),
                    *(f"set:{o.index}" for o in overrides),
                )
                results.append(dataclasses.replace(resolved, layers=layers))
        return results

    def _overrides(self, sets: tuple[str, ...], report: _Report) -> list[Override]:
        """Parse and key-check every --set (V-30)."""
        overrides = []
        for index, text in enumerate(sets):
            source = f"--set[{index}] {text}"
            try:
                override = parse_override(text, index)
            except OverrideError as error:
                report.plain(source, str(error), rule="V-30")
                continue
            unknown = check_keys(sparse_layer(override.path, override.value), Scenario)
            for issue in unknown:
                report.at(Origin(f"set:{index}", source), issue)
            if not unknown:
                overrides.append(override)
        return overrides

    def _apply(
        self, steps: Sequence[_Step], report: _Report
    ) -> tuple[dict, Provenance]:
        data: dict[str, Any] = {}
        prov = Provenance()
        for step in steps:
            if isinstance(step, _Merge):
                merge_into(data, prov, step.data, step.origin_of)
                continue
            try:
                assign(data, prov, step.path, step.value, step.origin)
            except AssignError as error:
                report.at(step.origin, Issue(step.path, str(error)))
        return data, prov

    def _points(
        self,
        upper: dict[str, Any],
        prov: Provenance,
        defaults: _FileLayer,
        plain_steps: list[_Step],
        chain: list[_FileLayer],
        overrides: list[Override],
        point: str | None,
        report: _Report,
    ) -> list[_Point] | None:
        """Validate the sweep and turn each point into a layer."""
        mark = len(report.diagnostics)
        sweep_data = upper["sweep"]
        try:
            sweep = Sweep.model_validate(sweep_data)
        except ValidationError as error:
            issues = issues_from_validation_error(error, sweep_data, ("sweep",))
            report.located(prov, issues)
            return None
        issues = check_sweep(sweep)
        for task in BUILTIN_TASKS:
            if _get(upper, ("tasks", task, "baseline")) is not None:
                message = sweep_baseline_message(sweep)
                issues.append(Issue(("tasks", task, "baseline"), message, rule="V-24"))
        if sweep.points is None:
            issues += self._axis_issues(sweep, defaults, plain_steps)
        report.located(prov, issues)
        for override in overrides:
            for text in sweep.axes:
                axis = parse_path(text)
                if is_prefix(axis, override.path) or is_prefix(override.path, axis):
                    report.plain(
                        override.source,
                        f"{format_path(override.path)} is the sweep axis {text}; "
                        "change the axis in the scenario, or run one point with "
                        "--point",
                        rule="V-40",
                    )
        if report.failed_since(mark):
            return None
        expanded = list(expand(sweep))
        if point is not None:
            names = [p.name for p in expanded]
            if point not in names:
                source = chain[0].loaded.source
                report.plain(
                    source, f"no sweep point {point}; points: {', '.join(names)}"
                )
                return None
            expanded = [p for p in expanded if p.name == point]
        holders = {layer.loaded.source: layer for layer in chain}
        return [self._point(sweep, p, upper, prov, holders) for p in expanded]

    def _axis_issues(
        self, sweep: Sweep, defaults: _FileLayer, plain_steps: list[_Step]
    ) -> list[Issue]:
        """Every axis replaces a value the scenario has (V-39)."""
        base, _ = self._apply(plain_steps, _Report())
        scoped = _scoped_defaults(defaults.data, base)
        full, _ = self._apply(
            [_Merge(scoped, defaults.origin_of()), *plain_steps], _Report()
        )
        form = "zip" if sweep.zip is not None else "product"
        return [
            Issue(
                ("sweep", form, text),
                f"{text} is not set in the scenario; an axis replaces a value "
                "the scenario has",
                rule="V-39",
            )
            for text in sweep.axes
            if not _has(full, parse_path(text))
        ]

    def _point(
        self,
        sweep: Sweep,
        point: ExpandedPoint,
        upper: dict[str, Any],
        prov: Provenance,
        holders: dict[str | None, _FileLayer],
    ) -> _Point:
        layer = f"sweep:{point.name}"
        if sweep.points is not None:
            overlay = upper["sweep"]["points"][point.index]
            holder = holders[prov.origin(("sweep", "points")).source]
            origin_of = holder.origin_of(layer, ("sweep", "points", point.index))
            data = {k: overlay[k] for k in ("server", "tasks") if overlay.get(k)}
            return _Point(point.name, [_Merge(copy.deepcopy(data), origin_of)], {})
        form = "zip" if sweep.zip is not None else "product"
        steps: list[_Step] = []
        for (path, value), text in zip(point.assignments, sweep.axes, strict=True):
            axis = prov.locate(("sweep", form, text))
            steps.append(_Assign(path, value, Origin(layer, axis.source, axis.line)))
        baselines: dict[str, tuple[_Step, KeyPath]] = {}
        blocks = (upper["sweep"].get("baselines") or {}).get(point.name) or {}
        for task, block in blocks.items():
            if block is None:
                continue
            where = ("sweep", "baselines", point.name, task)
            origin_of = _relocated(prov, layer, where, ("tasks", task, "baseline"))
            step = _Merge(
                {"tasks": {task: {"baseline": copy.deepcopy(block)}}}, origin_of
            )
            baselines[task] = (step, where)
        return _Point(point.name, steps, baselines)

    def _validate(
        self,
        scenario: _FileLayer,
        point: str | None,
        data: dict[str, Any],
        prov: Provenance,
        overrides: list[Override],
        follow_refs: bool,
        report: _Report,
    ) -> Resolved | None:
        try:
            model = Scenario.model_validate(data)
        except ValidationError as error:
            report.located(prov, issues_from_validation_error(error, data))
            return None
        issues = check_scenario(model)
        ref_baselines: dict[str, RefBaseline] = {}
        if follow_refs:
            issues += self._ref_baselines(model, point, ref_baselines)
        report.located(prov, issues)
        if any(issue.severity == "error" for issue in issues):
            return None
        return Resolved(
            ref=scenario.ref,
            file=scenario.path,
            point=point,
            data=data,
            model=model,
            provenance=prov,
            overrides=tuple(o.text for o in overrides),
            ref_baselines=ref_baselines,
        )

    # Baselines by reference -----------------------------------------------

    def _resolve_ref(self, ref: str) -> tuple[Resolved, ...] | DiagnosticError:
        if ref not in self._refs:
            try:
                self._refs[ref] = self.resolve(self.file_of(ref), follow_refs=False)
            except DiagnosticError as error:
                self._refs[ref] = error
        return self._refs[ref]

    def _ref_baselines(
        self, model: Scenario, point: str | None, found: dict[str, RefBaseline]
    ) -> list[Issue]:
        """Copy `from:` baselines in, after checking them (V-23)."""
        issues = []
        for task in BUILTIN_TASKS:
            block = getattr(model.tasks, task)
            if block is None or not isinstance(block.baseline, BaselineRef):
                continue
            ref = block.baseline.from_
            copied, problem, hint = self._ref_baseline(model, task, ref, point)
            if copied is not None:
                found[task] = copied
            else:
                path = ("tasks", task, "baseline", "from")
                issues.append(Issue(path, problem, hint=hint, rule="V-23"))
        return issues

    def _ref_baseline(
        self, model: Scenario, task: str, ref: str, point: str | None
    ) -> tuple[RefBaseline | None, str, str | None]:
        if not self.file_of(ref).is_file():
            return None, f"no scenario {ref}{self._near(ref)}", None
        targets = self._resolve_ref(ref)
        if isinstance(targets, DiagnosticError):
            return None, f"{ref} does not resolve", targets.diagnostics[0].render()
        target = next((t for t in targets if t.point == point), None)
        if target is None:
            where = f"a point named {point}" if point else "a single point"
            return None, f"{ref} is a sweep without {where}", None
        other = getattr(target.model.tasks, task)
        if other is None or other.baseline is None:
            return None, f"{ref} has no inline tasks.{task} baseline", None
        if isinstance(other.baseline, BaselineRef):
            return (
                None,
                f"{ref} takes its {task} baseline from {other.baseline.from_}; refer "
                "to that scenario directly",
                None,
            )
        rule = baseline_keys(model, task)
        have = tuple(other.baseline.model_extra or {})
        if rule is not None and baseline_key_problem(set(have), *rule) is not None:
            want, subset = rule
            among = "are among" if subset else "are"
            return (
                None,
                f"{ref} has {task} baseline keys {', '.join(have)}, but this task's "
                f"keys {among} {', '.join(want)}",
                None,
            )
        dumped = other.baseline.model_dump(by_alias=True, exclude_none=True)
        return RefBaseline(source=ref, baseline=dumped), "", None


def _relocated(prov: Provenance, layer: str, where: KeyPath, at: KeyPath) -> OriginOf:
    """Origins for data moved from `where` in the file to `at` in the scenario."""

    def origin_of(path: KeyPath) -> Origin:
        found = prov.locate((*where, *path[len(at) :]))
        return Origin(layer, found.source, found.line)

    return origin_of
