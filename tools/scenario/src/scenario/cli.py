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

validate, show and calibrate exit 0 when everything is fine (warnings
allowed), 1 when a scenario has problems, 2 for usage errors, including a
Plan that cannot be selected or decoded. run, up, perf and eval exit with
the runtime's codes (runtime/exits.py), where 1 is a failed task, so a
scenario with problems is a usage error (2) there.
"""

import os
import pathlib
import sys
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, NoReturn

import click

from scenario import PLAN_FORMAT, __version__, gen
from scenario import calibrate as calibration
from scenario.errors import Diagnostic, DiagnosticError
from scenario.keycheck import check_keys
from scenario.model.scenario import Scenario
from scenario.overrides import sparse_layer
from scenario.paths import PathSyntaxError, parse_path
from scenario.plan import codec
from scenario.plan.build import PlanError, plan, resolved_payload, select_point
from scenario.plan.types import CLIENT_ROLE, Plan
from scenario.resolve import DEFAULTS_FILE, Resolved, Resolver, find_root
from scenario.runtime import driver, exits, init
from scenario.runtime.exits import RunError
from scenario.runtime.gate import GateError
from scenario.runtime.placement import ENV_PLAN, PlacementFlags, decode_plan
from scenario.show import describe, render_plan, render_resolved, render_why
from scenario.targets.local import LocalTarget

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
    code: int = EXIT_PROBLEMS,
) -> tuple[Resolved, ...]:
    """Resolve FILE, or print every problem and exit with `code`."""
    try:
        resolver = Resolver(find_root(file, root, cwd=pathlib.Path.cwd()))
        resolved = resolver.resolve(file.resolve(), tuple(sets), point)
    except DiagnosticError as error:
        click.echo("\n".join(d.render() for d in error.diagnostics), err=True)
        sys.exit(code)
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


# run, up, perf, eval ---------------------------------------------------

EXIT_CODES = "\n".join(
    f"  {code:>3}  {meaning}" for code, meaning in sorted(exits.MEANINGS.items())
)


def _placement(command: Callable[..., Any]) -> Callable[..., Any]:
    """The placement flags; each wins over its SCENARIO_* env var."""
    options = [
        click.option(
            "--role-hosts",
            metavar="HOSTS",
            help="Every role's hosts: role=host,host;role=host, or JSON.",
        ),
        click.option("--advertise-ip", metavar="IP", help="This host's own address."),
        click.option(
            "--head-address",
            metavar="HOST",
            help="The first host of this host's role (name or IP).",
        ),
        click.option("--host-index", type=int, help="This host's index in its role."),
        click.option("--role", help="This host's role (client: the chipless driver)."),
    ]
    for option in options:
        command = option(command)
    return command


ResultsOption = click.option(
    "--results-dir",
    type=click.Path(path_type=pathlib.Path, file_okay=False),
    help="Results root; the run directory is <root>/<name>[/<point>]. Default: "
    "the target's root, else ./scenario-results/<name>-<UTC stamp>.",
)
CacheOption = click.option(
    "--lm-eval-cache",
    metavar="DIR",
    help="Where lm_eval keeps its request cache; without it, lm_eval runs uncached.",
)


def _options(
    mode: driver.Mode,
    results_dir: pathlib.Path | None,
    *,
    flags: PlacementFlags | None = None,
    resolved: Resolved | None = None,
    url: str | None = None,
    lm_eval_cache: str | None = None,
) -> driver.RunOptions:
    return driver.RunOptions(
        env=dict(os.environ),
        flags=flags or PlacementFlags(),
        results_root=results_dir,
        repo_root=pathlib.Path.cwd(),
        resolved=resolved_payload(resolved) if resolved is not None else None,
        mode=mode,
        url=url,
        lm_eval_cache=lm_eval_cache,
    )


def _supervise() -> None:
    """As a container's first process, fork and stay behind to reap."""
    code = init.maybe_supervise()
    if code is not None:
        sys.exit(code)


def _select(points: Sequence[Resolved], point: str | None) -> Resolved:
    try:
        return select_point(points, point)
    except PlanError as error:
        _fail(str(error))


def _dry_run(planned: Plan, role: str | None, host_index: int | None) -> None:
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


@main.command(epilog=f"\b\nExit codes:\n{EXIT_CODES}")
@click.argument("file", required=False, type=FileArg)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the selected hosts' commands and env instead of running them.",
)
@SetOption
@click.option("--point", help="The sweep point to run.")
@_placement
@ResultsOption
@click.option("--only", metavar="TASKS", help="Only these tasks, comma-separated.")
@CacheOption
@RootOption
def run(
    file: pathlib.Path | None,
    dry_run: bool,
    sets: tuple[str, ...],
    point: str | None,
    role: str | None,
    host_index: int | None,
    head_address: str | None,
    advertise_ip: str | None,
    role_hosts: str | None,
    results_dir: pathlib.Path | None,
    only: str | None,
    lm_eval_cache: str | None,
    root: pathlib.Path | None,
) -> None:
    """Run this host's part of a scenario: serve, run the tasks, keep results.

    With FILE, the local target runs it on this machine. Without FILE, the
    Plan comes from SCENARIO_PLAN and placement from the SCENARIO_* env, as
    in a pod. A scenario without tasks serves until SIGINT or SIGTERM.
    """
    resolved = None
    if file is None:
        text = os.environ.get(ENV_PLAN)
        if not text:
            _fail(f"run needs a scenario FILE, or {ENV_PLAN} in the environment")
        if sets or point or only:
            _fail(
                "--set, --point and --only need a scenario FILE; a pod's Plan is fixed"
            )
        try:
            planned = decode_plan(text)
        except RunError as error:
            _fail(error.message)
    else:
        points = _resolve(file, sets, root, point, EXIT_USAGE)
        resolved = _select(points, point)
        planned = _plan(points, point, only)
    if dry_run:
        _dry_run(planned, role, host_index)
        return
    _supervise()
    flags = PlacementFlags(role, host_index, head_address, advertise_ip, role_hosts)
    options = _options(
        "run", results_dir, flags=flags, resolved=resolved, lm_eval_cache=lm_eval_cache
    )
    if file is None:
        sys.exit(driver.run(planned, options))
    sys.exit(LocalTarget(options).submit(planned))


@main.command(epilog=f"\b\nExit codes:\n{EXIT_CODES}")
@click.argument("file", type=FileArg)
@SetOption
@click.option("--point", help="The sweep point to serve.")
@_placement
@ResultsOption
@RootOption
def up(
    file: pathlib.Path,
    sets: tuple[str, ...],
    point: str | None,
    role: str | None,
    host_index: int | None,
    head_address: str | None,
    advertise_ip: str | None,
    role_hosts: str | None,
    results_dir: pathlib.Path | None,
    root: pathlib.Path | None,
) -> None:
    """Serve a scenario on this host until SIGINT or SIGTERM, running no task.

    Stopping it that way is its normal end (exit 0); a server that exits
    on its own is exit 5.
    """
    points = _resolve(file, sets, root, point, EXIT_USAGE)
    resolved = _select(points, point)
    planned = _plan(points, point, None)
    _supervise()
    flags = PlacementFlags(role, host_index, head_address, advertise_ip, role_hosts)
    options = _options("up", results_dir, flags=flags, resolved=resolved)
    sys.exit(LocalTarget(options).submit(planned))


def _endpoint(url: str) -> tuple[str, int]:
    parts = urllib.parse.urlsplit(url)
    if (
        parts.scheme != "http"
        or not parts.hostname
        or parts.path not in ("", "/")
        or parts.query
    ):
        _fail(f"--url is http://HOST:PORT, not {url}")
    return parts.hostname, parts.port or 80


def _against(
    kind: str,
    file: pathlib.Path,
    url: str,
    sets: Sequence[str],
    point: str | None,
    results_dir: pathlib.Path | None,
    root: pathlib.Path | None,
    lm_eval_cache: str | None = None,
) -> NoReturn:
    points = _resolve(file, sets, root, point, EXIT_USAGE)
    resolved = _select(points, point)
    try:
        planned = plan(resolved, only=[kind], endpoint=_endpoint(url))
    except PlanError as error:
        _fail(str(error))
    options = _options(
        "url", results_dir, resolved=resolved, url=url, lm_eval_cache=lm_eval_cache
    )
    sys.exit(LocalTarget(options).submit(planned))


UrlOption = click.option(
    "--url", required=True, metavar="URL", help="The running server: http://HOST:PORT."
)


@main.command(epilog=f"\b\nExit codes:\n{EXIT_CODES}")
@click.argument("file", type=FileArg)
@UrlOption
@SetOption
@click.option("--point", help="The sweep point whose perf task to run.")
@ResultsOption
@RootOption
def perf(
    file: pathlib.Path,
    url: str,
    sets: tuple[str, ...],
    point: str | None,
    results_dir: pathlib.Path | None,
    root: pathlib.Path | None,
) -> None:
    """Run a scenario's perf task against a server that is already up.

    The client tokenizes with the served name unless the scenario names a
    tokenizer; results and the gate are as in `run`.
    """
    _against("perf", file, url, sets, point, results_dir, root)


@main.command(name="eval", epilog=f"\b\nExit codes:\n{EXIT_CODES}")
@click.argument("file", type=FileArg)
@UrlOption
@SetOption
@click.option("--point", help="The sweep point whose eval task to run.")
@ResultsOption
@CacheOption
@RootOption
def eval_(
    file: pathlib.Path,
    url: str,
    sets: tuple[str, ...],
    point: str | None,
    results_dir: pathlib.Path | None,
    lm_eval_cache: str | None,
    root: pathlib.Path | None,
) -> None:
    """Run a scenario's eval task against a server that is already up."""
    _against("eval", file, url, sets, point, results_dir, root, lm_eval_cache)


# calibrate ----------------------------------------------------------------


@main.command()
@click.argument("file", type=FileArg)
@click.argument(
    "results",
    nargs=-1,
    required=True,
    type=click.Path(path_type=pathlib.Path, file_okay=False, exists=True),
)
@click.option("--point", help="The sweep point to calibrate.")
@click.option("--write", is_flag=True, help="Write the new baselines into FILE.")
@RootOption
def calibrate(
    file: pathlib.Path,
    results: tuple[pathlib.Path, ...],
    point: str | None,
    write: bool,
    root: pathlib.Path | None,
) -> None:
    """New baselines from run directories: print the change, or --write it.

    RESULTS are run directories of FILE's scenario, e.g. downloaded
    artifacts of several runs. The values are check_regression.py's
    calibration, the medians over the runs. --write rewrites only the
    baseline entries it has values for; every other byte of FILE stays,
    comments included. A baseline that is `from:` another scenario is
    refused: calibrate that one.
    """
    points = _resolve(file, (), root, point)
    resolved = _select(points, point)
    try:
        proposals = calibration.propose(resolved, results, pathlib.Path.cwd())
    except (calibration.CalibrateError, GateError) as error:
        _fail(str(error), EXIT_PROBLEMS)
    click.echo(calibration.render(proposals), nl=False)
    if not write:
        return
    original = file.read_text()
    try:
        changed = calibration.write(file, resolved.point, proposals)
        _resolve(file, (), root, point)
    except calibration.CalibrateError as error:
        _fail(str(error), EXIT_PROBLEMS)
    except SystemExit:
        file.write_text(original)
        _fail(
            f"the rewritten {file} does not resolve; it is left as it was",
            EXIT_PROBLEMS,
        )
    entries = f"{changed} baseline entr{'y' if changed == 1 else 'ies'}"
    click.echo(f"wrote {entries} to {file}" if changed else "baselines unchanged")


# dev --------------------------------------------------------------------


@main.group()
def dev() -> None:
    """Commands for working on the tool itself."""


@dev.command()
@click.option(
    "--check", is_flag=True, help="Write nothing; fail when a generated file is stale."
)
def regen(check: bool) -> None:
    """Regenerate the JSON schemas and REFERENCE.md from the models."""
    try:
        root = gen.tool_dir()
    except FileNotFoundError as error:
        _fail(str(error))
    if check:
        names = gen.stale(root)
        for name in names:
            click.echo(f"stale: tools/scenario/{name}; run: {gen.REGEN}", err=True)
        sys.exit(EXIT_PROBLEMS if names else 0)
    for name in gen.write(root):
        click.echo(f"wrote tools/scenario/{name}")
