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
"""Merge rules and provenance."""

from typing import Any

import pytest
from scenario.merge import (
    TOOL,
    AssignError,
    Origin,
    Provenance,
    assign,
    keeps_null,
    merge_into,
)

A = Origin("defaults", "defaults.yaml", 1)
B = Origin("scenario", "s.yaml", 2)


def merged(*layers: tuple[dict[str, Any], Origin]) -> tuple[dict, Provenance]:
    data: dict[str, Any] = {}
    prov = Provenance()
    for layer, origin in layers:
        merge_into(data, prov, layer, lambda _, origin=origin: origin)
    return data, prov


def test_mappings_merge_lists_and_scalars_replace() -> None:
    data, _ = merged(
        ({"args": {"a": 1, "cc": {"backend": "x", "sizes": [1, 8]}}, "l": [1, 2]}, A),
        ({"args": {"b": 2, "cc": {"sizes": [1, 8, 1296]}}, "l": [3]}, B),
    )
    assert data == {
        "args": {"a": 1, "cc": {"backend": "x", "sizes": [1, 8, 1296]}, "b": 2},
        "l": [3],
    }
    assert list(data["args"]) == ["a", "cc", "b"]


def test_null_deletes_and_missing_stays_missing() -> None:
    data, prov = merged(
        ({"server": {"slice": "tpu7x/2x2x1", "args": {"x": 1}}}, A),
        ({"server": {"slice": None, "args": {"y": None}}}, B),
    )
    assert data == {"server": {"args": {"x": 1}}}
    assert prov.deleted == {("server", "slice"): B}
    assert ("server", "slice") not in prov.leaves


def test_type_changes_replace() -> None:
    data, prov = merged(({"g": "a=b"}, A), ({"g": {"k": 1}}, B))
    assert data == {"g": {"k": 1}}
    assert prov.superseded == {("g",): [A]}
    data, _ = merged(({"g": {"k": 1}}, A), ({"g": "a=b"}, B))
    assert data == {"g": "a=b"}


def test_role_args_and_env_keep_null() -> None:
    assert keeps_null(("server", "roles", "decode", "args", "api-server-count"))
    assert keeps_null(("server", "roles", "decode", "env", "X"))
    assert not keeps_null(("server", "args", "api-server-count"))
    data, prov = merged(
        ({"server": {"roles": {"decode": {"args": {"x": 1}}}}}, A),
        ({"server": {"roles": {"decode": {"args": {"x": None, "y": None}}}}}, B),
    )
    assert data["server"]["roles"]["decode"]["args"] == {"x": None, "y": None}
    assert prov.leaves[("server", "roles", "decode", "args", "y")] == B


def test_provenance_of_leaves_and_nodes() -> None:
    data, prov = merged(
        ({"server": {"args": {"a": 1, "b": 2}}, "c": {}}, A),
        ({"server": {"args": {"a": 3}}, "tasks": {"custom": [{"name": "x"}]}}, B),
    )
    assert prov.leaves == {
        ("server", "args", "a"): B,
        ("server", "args", "b"): A,
        ("c",): A,
        ("tasks", "custom"): B,
    }
    assert prov.superseded == {("server", "args", "a"): [A]}
    assert prov.nodes[("server",)] == B
    assert prov.nodes[("tasks", "custom", 0, "name")] == B
    assert prov.origin(("server", "args", "a")) == B
    assert prov.origin(("tasks", "driver")) == TOOL
    assert prov.locate(("server", "args", "zzz")) == B
    assert prov.locate(("nothing",)) == TOOL


def test_empty_mapping_is_a_leaf_until_filled() -> None:
    _, prov = merged(({"env": {}}, A), ({"env": {"X": "1"}}, B))
    assert ("env",) not in prov.leaves
    assert prov.leaves[("env", "X")] == B


def test_assign_creates_mappings() -> None:
    data, prov = merged(({"server": {"args": {"a": 1}}}, A))
    assign(data, prov, ("server", "args", "b"), 2, B)
    assign(data, prov, ("tasks", "perf", "concurrency"), [1, 2], B)
    assert data == {
        "server": {"args": {"a": 1, "b": 2}},
        "tasks": {"perf": {"concurrency": [1, 2]}},
    }
    assert prov.leaves[("tasks", "perf", "concurrency")] == B
    assign(data, prov, ("server", "args", "a"), None, B)
    assert data["server"]["args"] == {"b": 2}
    assert prov.deleted[("server", "args", "a")] == B


def test_assign_mapping_merges() -> None:
    data, prov = merged(({"args": {"cc": {"a": 1}}}, A))
    assign(data, prov, ("args", "cc"), {"b": 2}, B)
    assert data == {"args": {"cc": {"a": 1, "b": 2}}}


def test_assign_into_a_list_item() -> None:
    data, prov = merged(({"tasks": {"custom": [{"name": "x", "timeout": 60}]}}, A))
    assign(data, prov, ("tasks", "custom", 0, "timeout"), "2h", B)
    assert data["tasks"]["custom"] == [{"name": "x", "timeout": "2h"}]
    assert prov.leaves[("tasks", "custom")] == B
    assert prov.superseded[("tasks", "custom")] == [A]
    assert prov.nodes[("tasks", "custom", 0, "timeout")] == B


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("tasks", "custom", 3, "timeout"), "tasks.custom has no item [3]"),
        (("tasks", "custom", "x", "y"), "tasks.custom is not a mapping"),
        (("tasks", "custom", 0), "null cannot delete a list item"),
    ],
)
def test_assign_errors(path: tuple[str | int, ...], message: str) -> None:
    data, prov = merged(({"tasks": {"custom": [{"name": "x"}]}}, A))
    value = None if message.startswith("null") else 1
    with pytest.raises(AssignError, match=message.replace("[", r"\[")):
        assign(data, prov, path, value, B)


def test_gen_kwargs_mapping_replaces_whole() -> None:
    path = ("tasks", "eval", "lm_eval", "gen_kwargs")
    lower = {"tasks": {"eval": {"lm_eval": {"gen_kwargs": {"a": 1, "b": {"c": 2}}}}}}
    upper = {"tasks": {"eval": {"lm_eval": {"gen_kwargs": {"b": {"d": 3}}}}}}
    data, prov = merged((lower, A), (upper, B))
    assert data["tasks"]["eval"]["lm_eval"]["gen_kwargs"] == {"b": {"d": 3}}
    assert prov.origin(path) == B
    assert prov.superseded[path] == [A]
    assert (*path, "b", "d") not in prov.leaves


def test_assign_inside_gen_kwargs_owns_the_whole_value() -> None:
    path = ("tasks", "eval", "lm_eval", "gen_kwargs")
    data, prov = merged(({"tasks": {"eval": {"lm_eval": {"gen_kwargs": {"a": 1}}}}}, A))
    assign(data, prov, (*path, "b"), 2, B)
    assert data["tasks"]["eval"]["lm_eval"]["gen_kwargs"] == {"a": 1, "b": 2}
    assert prov.origin(path) == B
    assert prov.superseded[path] == [A]
