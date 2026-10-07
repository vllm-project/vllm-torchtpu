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
"""Loading YAML into plain data with line numbers."""

import pathlib
import textwrap

import pytest
from scenario.errors import DiagnosticError
from scenario.load import load_file, load_yaml


def load(text: str):
    return load_yaml(textwrap.dedent(text), "x.yaml")


def problems(text: str) -> list[tuple[int | None, str]]:
    with pytest.raises(DiagnosticError) as info:
        load(text)
    return [(d.line, d.message) for d in info.value.diagnostics]


def test_lines_of_keys_and_items() -> None:
    loaded = load(
        """\
        schema: 1
        server:
          args:
            max-model-len: 9216
        tasks:
          custom:
            - name: a
              command: [x, y]
            - name: b
        """
    )
    assert loaded.lines[()] == 1
    assert loaded.lines[("server", "args", "max-model-len")] == 4
    assert loaded.lines[("tasks", "custom", 1)] == 9
    assert loaded.lines[("tasks", "custom", 1, "name")] == 9
    assert loaded.lines[("tasks", "custom", 0, "command", 1)] == 8
    assert loaded.line(("server", "args", "max-model-len", "x")) == 4
    assert loaded.line(("nowhere",)) == 1


def test_plain_types() -> None:
    data = load(
        """\
        hex: 0x10
        under: 1_000
        float: 1.5
        exp: 1e3
        yes: yes
        on: true
        none: ~
        quoted: "8"
        block: |
          text
        """
    ).data
    assert data == {
        "hex": 16,
        "under": 1000,
        "float": 1.5,
        "exp": 1000.0,
        "yes": "yes",
        "on": True,
        "none": None,
        "quoted": "8",
        "block": "text\n",
    }
    assert type(data["hex"]) is int
    assert type(data["exp"]) is float
    assert type(data["block"]) is str


def test_anchors_expand() -> None:
    data = load(
        """\
        a: &shared {x: [1, 2]}
        b: *shared
        """
    ).data
    assert data["a"] == data["b"] == {"x": [1, 2]}
    assert data["a"] is not data["b"]


def test_every_duplicate_key_is_reported() -> None:
    assert problems(
        """\
        a: 1
        b:
          c: 1
          c: 2
        a: 3
        """
    ) == [
        (4, "duplicate key c (first at line 3)"),
        (5, "duplicate key a (first at line 1)"),
    ]


def test_tags_dates_merge_keys_and_key_types() -> None:
    assert problems(
        """\
        a: !secret x
        b: 2026-10-06
        c:
          <<: {d: 1}
        1: x
        """
    ) == [
        (1, "YAML tags are not allowed (!secret)"),
        (2, "dates are not supported; quote the value"),
        (4, "YAML merge keys (<<) are not supported; share settings with extends"),
        (5, "keys are strings; quote '1'"),
    ]


def test_syntax_error_has_a_line() -> None:
    ((line, message),) = problems("a: [1,\nb: 2\n")
    assert line == 3
    assert message.startswith("YAML syntax: while parsing a flow sequence")


def test_one_document_only() -> None:
    ((_, message),) = problems("a: 1\n---\nb: 2\n")
    assert "expected a single document" in message


@pytest.mark.parametrize(
    ("text", "message"),
    [("", "the file is empty"), ("- a\n", "the top level must be a mapping")],
)
def test_shape(text: str, message: str) -> None:
    assert problems(text) == [(1, message)]


def test_load_file_names_its_source(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("a: 1\na: 2\n")
    with pytest.raises(DiagnosticError) as info:
        load_file(path, "scenarios/s.yaml")
    assert str(info.value) == ("scenarios/s.yaml:2: duplicate key a (first at line 1)")
