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
"""Arg mappings to argv (R-SCH-12).

    true             --k            (dotted key: --a.b true)
    false            --no-k         (dotted key: --a.b false)
    int, float, str  --k <value>
    list, mapping    --k <compact JSON, key order kept>

A null never reaches here: merging removes the key. Every arg map renders
this way: server and role args, router args and perf client args.
"""

import json
from collections.abc import Mapping
from typing import Any

RULES = (
    ("true", "--k; for a dotted key, --a.b true"),
    ("false", "--no-k; for a dotted key, --a.b false"),
    ("integer, number, string", "--k <value>"),
    ("list, mapping", "--k <compact JSON>, key order kept"),
    ("null", "removes the key when layers merge; never rendered"),
)


def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def scalar(value: bool | int | float | str) -> str:
    """The argv text of a scalar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_arg(key: str, value: Any) -> list[str]:
    flag = f"--{key}"
    dotted = "." in key
    if isinstance(value, bool):
        if dotted:
            return [flag, scalar(value)]
        return [flag] if value else [f"--no-{key}"]
    if isinstance(value, dict | list):
        return [flag, compact_json(value)]
    return [flag, scalar(value)]


def render_args(args: Mapping[str, Any]) -> list[str]:
    """Every key of `args` in order; a None value is skipped."""
    argv: list[str] = []
    for key, value in args.items():
        if value is not None:
            argv += render_arg(key, value)
    return argv


def render_model_args(pairs: Mapping[str, Any]) -> str:
    """lm_eval's --model_args: k=v joined by commas, booleans as true/false."""
    return ",".join(f"{key}={scalar(value)}" for key, value in pairs.items())
