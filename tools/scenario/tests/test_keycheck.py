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
"""Unknown keys, per layer, with the nearest valid key."""

from typing import Any

from scenario.keycheck import check_keys
from scenario.model.scenario import Defaults, Scenario
from scenario.paths import format_path


def found(data: Any, model: type = Scenario) -> list[tuple[str, str]]:
    return [(format_path(i.path), i.message) for i in check_keys(data, model)]


def test_suggests_the_nearest_key() -> None:
    assert found({"server": {"argz": {"x": 1}}}) == [
        ("server.argz", "unknown key; did you mean server.args?")
    ]


def test_lists_keys_without_a_near_match() -> None:
    ((path, message),) = found({"zzz": 1})
    assert path == "zzz"
    assert message == (
        "unknown key; known keys here: schema, name, extends, server, tasks, sweep"
    )


def test_walks_nested_models_lists_and_unions() -> None:
    data = {
        "server": {
            "roles": {"prefill": {"slice": "tpu7x/2x2x1", "kvv": {}}},
            "router": {"typ": "toy-proxy"},
        },
        "tasks": {
            "perf": {
                "num_prompts": {"per_concurency": 2},
                "cells": [{"isl": 1, "osll": 1}],
                "baseline": {"isl1_osl1_c1": {"median_tpot": 1.0}},
            },
            "eval": {
                "lm_eval": {"tasks": ["gsm8k", {"name": "x", "limt": 3}]},
                "baseline": {"from": "a/b", "form": 1},
            },
            "custom": [{"name": "c", "comand": ["x"]}],
        },
        "sweep": {
            "points": [{"name": "p", "server": {"multihostt": "mp"}}],
            "baselines": {"c1": {"agentxx": {}}},
        },
    }
    assert found(data) == [
        (
            "server.roles.prefill.kvv",
            "unknown key; did you mean server.roles.prefill.kv?",
        ),
        ("server.router.typ", "unknown key; did you mean server.router.type?"),
        (
            "tasks.perf.num_prompts.per_concurency",
            "unknown key; did you mean tasks.perf.num_prompts.per_concurrency?",
        ),
        (
            "tasks.perf.cells[0].osll",
            "unknown key; did you mean tasks.perf.cells[0].osl?",
        ),
        (
            "tasks.perf.baseline.isl1_osl1_c1.median_tpot",
            "unknown key; did you mean "
            "tasks.perf.baseline.isl1_osl1_c1.median_tpot_ms?",
        ),
        (
            "tasks.eval.lm_eval.tasks[1].limt",
            "unknown key; did you mean tasks.eval.lm_eval.tasks[1].limit?",
        ),
        (
            "tasks.eval.baseline.form",
            "unknown key; did you mean tasks.eval.baseline.from?",
        ),
        (
            "tasks.custom[0].comand",
            "unknown key; did you mean tasks.custom[0].command?",
        ),
        (
            "sweep.points[0].server.multihostt",
            "unknown key; did you mean sweep.points[0].server.multihost?",
        ),
        (
            "sweep.baselines.c1.agentxx",
            "unknown key; did you mean sweep.baselines.c1.agentx?",
        ),
    ]


def test_free_form_maps_are_not_checked() -> None:
    data = {
        "server": {
            "args": {"anything-goes": {"nested_key": 1}},
            "env": {"WHATEVER": "1"},
            "roles": {"decode": {"args": {"x": None}}},
        },
        "tasks": {
            "perf": {"extra_body": {"k": 1}, "client_args": {"y": 1}},
            "eval": {"lm_eval": {"model_args": {"z": 1}}},
        },
        "sweep": {"zip": {"server.args.max-num-seqs": [1, 2]}},
    }
    assert found(data) == []


def test_null_deletes_without_checking() -> None:
    assert found({"server": {"slice": None}, "tasks": {"perf": None}}) == []


def test_defaults_tree() -> None:
    assert found({"server": {"model": {"name": "m"}}}, Defaults) == [
        (
            "server.model",
            "unknown key; known keys here: image, startup_timeout, args, env",
        )
    ]
    assert found({"tasks": {"eval": {"lm_eval": {"limits": 1}}}}, Defaults) == [
        (
            "tasks.eval.lm_eval.limits",
            "unknown key; did you mean tasks.eval.lm_eval.limit?",
        )
    ]


def test_base_path_prefixes_issues() -> None:
    (issue,) = check_keys(
        {"argz": 1}, Scenario.model_fields["server"].annotation, ("server",)
    )
    assert issue.path == ("server", "argz")
