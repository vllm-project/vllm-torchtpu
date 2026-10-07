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
"""Arg rendering (R-SCH-12)."""

import pytest
from scenario.plan.argv import render_args, render_model_args


@pytest.mark.parametrize(
    ("args", "argv"),
    [
        ({"enable-expert-parallel": True}, ["--enable-expert-parallel"]),
        ({"enable-prefix-caching": False}, ["--no-enable-prefix-caching"]),
        (
            {"profiler-config.ignore_frontend": True},
            ["--profiler-config.ignore_frontend", "true"],
        ),
        ({"profiler-config.x": False}, ["--profiler-config.x", "false"]),
        ({"max-model-len": 9216}, ["--max-model-len", "9216"]),
        ({"gpu-memory-utilization": 0.9}, ["--gpu-memory-utilization", "0.9"]),
        ({"tool-call-parser": "glm47"}, ["--tool-call-parser", "glm47"]),
        (
            {"compilation-config": {"compile_sizes": [1, 8], "backend": "x"}},
            ["--compilation-config", '{"compile_sizes":[1,8],"backend":"x"}'],
        ),
        ({"limit-mm-per-prompt": ["a", 1]}, ["--limit-mm-per-prompt", '["a",1]']),
        ({"skipped": None}, []),
    ],
)
def test_render_args(args: dict, argv: list[str]) -> None:
    assert render_args(args) == argv


def test_render_args_keeps_order() -> None:
    assert render_args({"b": 1, "a": True, "c": "x"}) == ["--b", "1", "--a", "--c", "x"]


def test_model_args() -> None:
    assert render_model_args({"num_concurrent": 32, "timeout": 7200, "x": True}) == (
        "num_concurrent=32,timeout=7200,x=true"
    )
