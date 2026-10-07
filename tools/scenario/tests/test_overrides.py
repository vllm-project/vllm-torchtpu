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
"""The --set grammar."""

import pytest
from scenario.overrides import OverrideError, parse_override, parse_value, sparse_layer


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("4096", 4096),
        ("0.9", 0.9),
        ("true", True),
        ("null", None),
        ("~", None),
        ("[1, 2]", [1, 2]),
        ("{a: 1, b: [x]}", {"a": 1, "b": ["x"]}),
        ("'\"8\"'", '"8"'),
        ('"8"', "8"),
        ("glm45", "glm45"),
        ("until=['<|im_end|>']", "until=['<|im_end|>']"),
        ("a: 1", "a: 1"),
        ("- x", "- x"),
        ("2026-10-06", "2026-10-06"),
        (" 7 ", 7),
    ],
)
def test_values(text: str, value: object) -> None:
    assert parse_value(text) == value


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "the value is empty"),
        ("[1, 2", "not a YAML value"),
    ],
)
def test_bad_values(text: str, message: str) -> None:
    with pytest.raises(OverrideError, match=message):
        parse_value(text)


def test_paths() -> None:
    override = parse_override("server.args.max-model-len=4096", 0)
    assert (override.path, override.value) == (
        ("server", "args", "max-model-len"),
        4096,
    )
    assert parse_override("tasks.custom[0].timeout=2h", 1).path == (
        "tasks",
        "custom",
        0,
        "timeout",
    )
    assert parse_override("server.args[profiler-config.x]=1", 2).path == (
        "server",
        "args",
        "profiler-config.x",
    )
    assert parse_override("server.env.A=b=c", 3).value == "b=c"
    assert parse_override("server.args.x=1", 4).source == "--set[4] server.args.x=1"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("server.args.x", "expected path=value"),
        ("name=x", r"--set changes server\.\* or tasks\.\* only"),
        ("sweep.zip.x=[1]", r"--set changes server\.\* or tasks\.\* only"),
        ("server=1", r"--set changes server\.\* or tasks\.\* only"),
        ("tasks.perf.baseline.x=1", "baselines change only through"),
        ("server..args=1", "empty segment"),
    ],
)
def test_bad_overrides(text: str, message: str) -> None:
    with pytest.raises(OverrideError, match=message):
        parse_override(text, 0)


def test_sparse_layer() -> None:
    assert sparse_layer(("server", "args", "x"), 1) == {"server": {"args": {"x": 1}}}
    assert sparse_layer(("tasks", "custom", 1, "timeout"), "2h") == {
        "tasks": {"custom": [None, {"timeout": "2h"}]}
    }
