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
"""The `scenario` command line.

Exit codes: 0 when everything is fine (warnings allowed), 1 when a scenario
has problems, 2 for usage errors, including a Plan that cannot be selected
or decoded.
"""

import pathlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NoReturn

import click

from scenario import PLAN_FORMAT, __version__
from scenario.errors import Diagnostic, DiagnosticError
from scenario.keycheck import check_keys
from scenario.model.scenario import Scenario
from scenario.overrides import sparse_layer
from scenario.paths import PathSyntaxError, parse_path
from scenario.plan import codec
from scenario.plan.build import PlanError, plan, resolved_payload, select_point
from scenario.plan.types import CLIENT_ROLE, Plan
from scenario.resolve import DEFAULTS_FILE, Resolved, Resolver, find_root
from scenario.show import describe, render_plan, render_resolved, render_why

EXIT_PROBLEMS = 1
EXIT_USAGE = 2

FileArg = click.Path(path_type=pathlib.Path, dir_okay=False)
RootOption = click.option(
    "--root",
    type=click.Path(path_type=pathlib.Path, file_okay=False, exists=True),
    help="The directory holding scenarios/defaults.yaml; found from the file "
    "or the current directory when not given.",
)
SetOption = click.option(
    "--set",
    "sets",
    multiple=True,
    metavar="PATH=VALUE",
    help="Override one value, e.g. server.args.max-model-len=4096; repeatable. "
    "The run becomes an experiment and skips its gates.",
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(
    __version__,
    "--version",
    prog_name="scenario",
    message=f"%(prog)s %(version)s (plan format {PLAN_FORMAT})",
)
def main() -> None:
    """Validate, show, render and run vLLM server scenarios."""


def _fail(message: str, code: int = EXIT_USAGE) -> NoReturn:
    click.echo(f"error: {message}", err=True)
    sys.exit(code)


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _resolve(
    file: pathlib.Path,
    sets: Sequence[str],
    root: pathlib.Path | None,
    point: str | None = None,
) -> tuple[Resolved, ...]:
    """Resolve FILE, or print every problem and exit 1."""
    try:
        resolver = Resolver(find_root(file, root, cwd=pathlib.Path.cwd()))
        resolved = resolver.resolve(file.resolve(), tuple(sets), point)
    except DiagnosticError as error:
        click.echo("\n".join(d.render() for d in error.diagnostics), err=True)
        sys.exit(EXIT_PROBLEMS)
    for warning in resolved[0].warnings if resolved else ():
        click.echo(warning.render(), err=True)
    return resolved


def _plan(points: Sequence[Resolved], point: str | None, only: str | None) -> Plan:
    try:
        tasks = [name.strip() for name in only.split(",")] if only else None
        return plan(select_point(points, point), only=tasks)
    except PlanError as error:
        _fail(str(error))


# validate ---------------------------------------------------------------


@dataclass
class _Outcome:
    file: pathlib.Path
    ref: str
    name: str | None = None
    summary: str = ""
    problems: list[Diagnostic] = field(default_factory=list)
    warnings: list[Diagnostic] = field(default_factory=list)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.problems if d.severity == "error"]


def _check(resolver: Resolver, file: pathlib.Path, sets: tuple[str, ...]) -> _Outcome:
    outcome = _Outcome(file, resolver.ref_of(file))
    try:
        points = resolver.resolve(file, sets)
    except DiagnosticError as error:
        outcome.problems += error.diagnostics
        return outcome
    outcome.name = points[0].model.name
    outcome.warnings += points[0].warnings
    summaries = []
    source = resolver.source_of(file)
    for resolved in points:
        where = f" of point {resolved.point}" if resolved.point is not None else ""
        try:
            planned = plan(resolved)
        except PlanError as error:
            outcome.problems.append(Diagnostic(source, None, "", str(error)))
            continue
        summaries.append(describe(planned))
        for what, size in (
            ("SCENARIO_PLAN", len(codec.encode(planned))),
            ("SCENARIO_RESOLVED", len(codec.encode_data(resolved_payload(resolved)))),
        ):
            if size > codec.MAX_ENCODED_BYTES:
                outcome.problems.append(
                    Diagnostic(
                        source,
                        None,
                        "",
                        f"the encoded {what}{where} is {size} bytes, over the "
                        f"{codec.MAX_ENCODED_BYTES}-byte limit",
                    )
                )
    if not summaries:
        return outcome
    outcome.summary = summaries[0]
    if len(points) > 1:
        names = ", ".join(str(r.point) for r in points)
        outcome.summary += f"; {len(points)} points: {names}"
    return outcome


def _duplicate_names(resolver: Resolver, outcomes: Sequence[_Outcome]) -> None:
    """Names are unique across scenarios/ (V-26)."""
    owners: dict[str, list[tuple[pathlib.Path, int]]] = {}
    for file in resolver.scenario_files():
        declared = resolver.declared_name(file)
        if declared is not None:
            owners.setdefault(declared[0], []).append((file, declared[1]))
    for outcome in outcomes:
        if not outcome.file.is_relative_to(resolver.scenarios):
            continue
        declared = resolver.declared_name(outcome.file)
        if declared is None:
            continue
        name, line = declared
        others = [
            f"{resolver.source_of(other)}:{other_line}"
            for other, other_line in owners.get(name, [])
            if other != outcome.file
        ]
        if others:
            outcome.problems.append(
                Diagnostic(
                    resolver.source_of(outcome.file),
                    line,
                    "name",
                    f"{name} is also the name of {', '.join(others)}",
                    rule="V-26",
                )
            )


@main.command()
@click.argument("files", nargs=-1, type=FileArg)
@SetOption
@RootOption
def validate(
    files: tuple[pathlib.Path, ...], sets: tuple[str, ...], root: pathlib.Path | None
) -> None:
    """Check scenario files; with none, every scenario under scenarios/.

    Each one is resolved and planned (every sweep point), names must be
    unique across scenarios/, and the encoded Plan must fit its env value.
    """
    cwd = pathlib.Path.cwd()
    groups: dict[pathlib.Path, list[pathlib.Path]] = {}
    try:
        if files:
            for file in files:
                if not file.is_file():
                    _fail(f"no such file: {file}")
                found = find_root(file, root, cwd=cwd)
                groups.setdefault(found, []).append(file.resolve())
        else:
            found = find_root(cwd / DEFAULTS_FILE, root, cwd=cwd)
            groups[found] = Resolver(found).scenario_files()
    except DiagnosticError as error:
        _fail(error.diagnostics[0].message)
    outcomes: list[_Outcome] = []
    for found, members in groups.items():
        resolver = Resolver(found)
        checked = [_check(resolver, file, sets) for file in members]
        _duplicate_names(resolver, checked)
        outcomes += checked
    errors = warnings = 0
    width = max((len(o.ref) for o in outcomes), default=0)
    name_width = max((len(o.name or "") for o in outcomes), default=0)
    # A problem in a shared parent is reported by every child; print it once.
    shown: set[str] = set()
    for outcome in outcomes:
        if outcome.errors:
            click.echo(f"error  {outcome.ref}")
        else:
            ref, name = f"{outcome.ref:<{width}}", f"{outcome.name or '':<{name_width}}"
            click.echo(f"ok     {ref}  {name}  {outcome.summary}")
        repeated = 0
        for diagnostic in [*outcome.problems, *outcome.warnings]:
            text = diagnostic.render()
            if text in shown:
                repeated += 1
                continue
            shown.add(text)
            errors += diagnostic.severity == "error"
            warnings += diagnostic.severity == "warning"
            click.echo(_indent(text))
        if repeated:
            click.echo(
                f"  and {repeated} problem{'s' if repeated > 1 else ''} shown above"
            )
    count = len(outcomes)
    tally = f"{count} scenario{'s' if count != 1 else ''}, {errors} error"
    tally += "s" if errors != 1 else ""
    if warnings:
        tally += f", {warnings} warning{'s' if warnings != 1 else ''}"
    click.echo(tally)
    sys.exit(EXIT_PROBLEMS if errors else 0)


# show -------------------------------------------------------------------


def _key_path(key: str) -> tuple[str | int, ...]:
    try:
        path = parse_path(key)
    except PathSyntaxError as error:
        _fail(str(error))
    unknown = check_keys(sparse_layer(path, 0), Scenario)
    if unknown:
        _fail(f"{key}: {unknown[0].message}", EXIT_PROBLEMS)
    return path


def _read_encoded(value: str) -> Plan:
    text = sys.stdin.read() if value == "-" else value
    try:
        return codec.decode(text)
    except codec.PlanFormatError as error:
        _fail(str(error))


@main.command()
@click.argument("file", required=False, type=FileArg)
@SetOption
@click.option("--point", help="The sweep point to show.")
@click.option("--plan", "show_plan", is_flag=True, help="Show the Plan of a point.")
@click.option("--why", metavar="KEY", help="Show every layer that set KEY.")
@click.option(
    "--encoded",
    metavar="VALUE",
    help="Show the Plan in a pod's SCENARIO_PLAN value ('-' reads stdin).",
)
@RootOption
def show(
    file: pathlib.Path | None,
    sets: tuple[str, ...],
    point: str | None,
    show_plan: bool,
    why: str | None,
    encoded: str | None,
    root: pathlib.Path | None,
) -> None:
    """Show a scenario's resolved values and where each came from.

    --plan shows what every host runs; --why KEY the layers that set KEY;
    --encoded the Plan a pod received.
    """
    if encoded is not None:
        if file is not None or sets or show_plan or why or point:
            _fail("--encoded takes no scenario file and no other option")
        click.echo(render_plan(_read_encoded(encoded)), nl=False)
        return
    if file is None:
        _fail("show needs a scenario FILE, or --encoded VALUE")
    if show_plan and why:
        _fail("pass --plan or --why, not both")
    path = _key_path(why) if why else None
    points = _resolve(file, sets, root, point)
    if show_plan:
        click.echo(render_plan(_plan(points, point, None)), nl=False)
        return
    texts = []
    for resolved in points:
        if path is None:
            texts.append(render_resolved(resolved))
        elif len(points) > 1:
            texts.append(f"point {resolved.point}:\n" + render_why(resolved, path))
        else:
            texts.append(render_why(resolved, path))
    click.echo("\n".join(texts), nl=False)


# run --------------------------------------------------------------------


@main.command()
@click.argument("file", type=FileArg)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print every host's commands and env instead of running them.",
)
@SetOption
@click.option("--point", help="The sweep point to run.")
@click.option("--role", help="Only this role's hosts (client: the chipless driver).")
@click.option("--host-index", type=int, help="Only this host of --role.")
@click.option("--only", metavar="TASKS", help="Only these tasks, comma-separated.")
@RootOption
def run(
    file: pathlib.Path,
    dry_run: bool,
    sets: tuple[str, ...],
    point: str | None,
    role: str | None,
    host_index: int | None,
    only: str | None,
    root: pathlib.Path | None,
) -> None:
    """Run a scenario on this machine; this version prints it (--dry-run)."""
    if not dry_run:
        _fail(
            "this version only prints a run: pass --dry-run. Running scenarios "
            "comes with the runtime."
        )
    planned = _plan(_resolve(file, sets, root, point), point, only)
    roles = [r.name for r in planned.roles]
    if planned.driver.chipless:
        roles.append(CLIENT_ROLE)
    if host_index is not None and role is None:
        if len(planned.roles) > 1:
            _fail(f"--host-index needs --role; roles: {', '.join(roles)}")
        role = planned.roles[0].name
    if role is not None and role not in roles:
        _fail(f"no role {role}; roles: {', '.join(roles)}")
    if host_index is not None:
        hosts = 1 if role == CLIENT_ROLE else planned.role(str(role)).hosts
        if not 0 <= host_index < hosts:
            _fail(
                f"role {role} has hosts 0..{hosts - 1}; got --host-index {host_index}"
            )
    click.echo(render_plan(planned, role, host_index), nl=False)
