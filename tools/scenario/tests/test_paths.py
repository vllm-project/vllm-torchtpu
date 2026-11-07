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
"""Dotted key paths."""

import pytest
from scenario.paths import PathSyntaxError, format_path, is_prefix, parse_path


@pytest.mark.parametrize(
    ("text", "path"),
    [
        ("server.args.max-model-len", ("server", "args", "max-model-len")),
        ("tasks.custom[0].timeout", ("tasks", "custom", 0, "timeout")),
        (
            "server.args[profiler-config.profiler]",
            ("server", "args", "profiler-config.profiler"),
        ),
        ("a.b[1]", ("a", "b", 1)),
        ("server.env.TPU_X", ("server", "env", "TPU_X")),
    ],
)
def test_round_trip(text: str, path: tuple[str | int, ...]) -> None:
    assert parse_path(text) == path
    assert format_path(path) == text


@pytest.mark.parametrize(
    "text", ["", ".a", "a.", "a..b", "a[", "a[]", "a b", "a.[b", "a]b", "[0]x"]
)
def test_rejects(text: str) -> None:
    with pytest.raises(PathSyntaxError):
        parse_path(text)


def test_brackets_around_plain_keys() -> None:
    assert parse_path("a[b][1]") == ("a", "b", 1)


def test_format_brackets_odd_keys() -> None:
    assert format_path(("server", "args", "a b")) == "server.args[a b]"
    assert format_path(("tasks", "eval", "baseline", "1")) == "tasks.eval.baseline[1]"


def test_is_prefix() -> None:
    assert is_prefix(("server",), ("server", "args"))
    assert is_prefix(("server", "args"), ("server", "args"))
    assert not is_prefix(("server", "args", "x"), ("server", "args"))
