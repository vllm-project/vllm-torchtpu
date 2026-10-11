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
"""Diagnostics: rendering, placing in files, and pydantic error rewriting."""

import textwrap
from typing import Any

import pytest
from pydantic import ValidationError
from scenario.errors import (
    Diagnostic,
    DiagnosticError,
    Issue,
    issues_from_validation_error,
    place,
    sort_diagnostics,
    suggest,
)
from scenario.keycheck import check_keys
from scenario.load import load_yaml
from scenario.model.rules import check_scenario
from scenario.model.scenario import Scenario
from scenario.paths import format_path


def test_render() -> None:
    assert (
        Diagnostic("a.yaml", 3, "server.argz", "unknown key").render()
        == "a.yaml:3: server.argz: unknown key"
    )
    assert Diagnostic("a.yaml", 3, "x", "m", hint="do this").render() == (
        "a.yaml:3: x: m\n  hint: do this"
    )
    assert (
        Diagnostic("a.yaml", 9, "x", "m", severity="warning").render()
        == "a.yaml:9: warning: x: m"
    )
    assert Diagnostic("--set[0] a=1", None, "", "bad").render() == "--set[0] a=1: bad"


def test_error_aggregates_and_sorts() -> None:
    diagnostics = sort_diagnostics(
        [
            Diagnostic("b.yaml", 1, "", "two"),
            Diagnostic("a.yaml", 9, "", "one"),
            Diagnostic("a.yaml", 2, "", "zero"),
        ]
    )
    assert str(DiagnosticError(diagnostics)) == (
        "a.yaml:2: zero\na.yaml:9: one\nb.yaml:1: two"
    )


def test_suggest() -> None:
    assert suggest("argz", ["args", "env"]) == "args"
    assert suggest("zzz", ["args", "env"]) is None


def rewrite(data: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    with pytest.raises(ValidationError) as info:
        Scenario.model_validate(data)
    return [
        (format_path(i.path), i.message, i.rule)
        for i in issues_from_validation_error(info.value, data)
    ]


def test_missing_field_points_at_its_name(single: dict[str, Any]) -> None:
    del single["tasks"]["perf"]["tolerance"]
    assert rewrite(single) == [("tasks.perf.tolerance", "required", None)]


def test_union_branches_become_one_message(single: dict[str, Any]) -> None:
    single["tasks"]["eval"]["lm_eval"]["gen_kwargs"] = 5
    assert rewrite(single) == [
        ("tasks.eval.lm_eval.gen_kwargs", "expected a string or a mapping; got 5", None)
    ]


def test_union_tags_are_not_in_the_path(single: dict[str, Any]) -> None:
    single["tasks"]["perf"]["num_prompts"] = "8"
    single["tasks"]["perf"]["baseline"]["isl8192_osl1024_c1024"]["completed"] = 1.5
    assert rewrite(single) == [
        ("tasks.perf.num_prompts", "expected an integer; got '8'", None),
        (
            "tasks.perf.baseline.isl8192_osl1024_c1024.completed",
            "expected an integer; got 1.5",
            None,
        ),
    ]


def test_key_errors_point_at_the_key(single: dict[str, Any]) -> None:
    single["server"]["args"]["max_num_seqs"] = 8
    assert rewrite(single) == [
        ("server.args.max_num_seqs", "use max-num-seqs", "V-15"),
    ]


def test_literal_bound_and_custom_messages(single: dict[str, Any]) -> None:
    single["server"]["multihost"] = "slurm"
    single["tasks"]["perf"]["tolerance"] = 2
    single["server"]["startup_timeout"] = "2 hours"
    assert rewrite(single) == [
        (
            "server.multihost",
            "expected 'mp' or 'ray'; got 'slurm'",
            None,
        ),
        (
            "server.startup_timeout",
            "expected a duration such as 2h, 90m, 4h10m or 30s, or whole seconds; "
            "got '2 hours'",
            None,
        ),
        ("tasks.perf.tolerance", "must be less than 1.0; got 2", None),
    ]


def test_extra_key_path(single: dict[str, Any]) -> None:
    single["server"]["model"]["revision"] = "main"
    assert rewrite(single) == [("server.model.revision", "unknown key", None)]


def test_one_file_reports_every_problem() -> None:
    text = textwrap.dedent(
        """\
        schema: 1
        name: demo
        server:
          image: us-docker.pkg.dev/test/images/vllm:test
          model: {name: m}
          accelerator: tpu7x
          topology: 2x2x2
          hosts: 2
          chips_per_host: 4
          startup_timout: 2h
          args:
            tensor-parallel-size: "8"
        tasks:
          custom:
            - name: probe
              command: [true]
              timeout: 10m
        """
    )
    loaded = load_yaml(text, "demo.yaml")
    issues = check_keys(loaded.data, Scenario)
    trimmed = {
        **loaded.data,
        "server": {
            k: v for k, v in loaded.data["server"].items() if k != "startup_timout"
        },
    }
    try:
        model = Scenario.model_validate(trimmed)
    except ValidationError as error:
        issues += issues_from_validation_error(error, trimmed)
    else:
        issues += check_scenario(model)

    def locate(path: tuple[str | int, ...]) -> tuple[str, int]:
        return loaded.source, loaded.line(path)

    rendered = [d.render() for d in sort_diagnostics(place(issues, locate))]
    assert rendered == [
        "demo.yaml:3: server.startup_timeout: required",
        "demo.yaml:10: server.startup_timout: unknown key; did you mean "
        "server.startup_timeout?",
        "demo.yaml:16: tasks.custom[0].command[0]: expected a string; got True",
    ]


def test_place_keeps_rule_and_severity() -> None:
    (diagnostic,) = place(
        [Issue(("a",), "m", hint="h", rule="V-32", severity="warning")],
        lambda path: ("f.yaml", 4),
    )
    assert diagnostic == Diagnostic(
        "f.yaml", 4, "a", "m", hint="h", rule="V-32", severity="warning"
    )
