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
"""Tests for the count comparison in tools/check_pyrefly_baseline.py."""

import importlib.util
import pathlib

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "tools" / "check_pyrefly_baseline.py"
)
_spec = importlib.util.spec_from_file_location("check_pyrefly_baseline", _SCRIPT)
check_pyrefly_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_pyrefly_baseline)

compare = check_pyrefly_baseline.compare

_PATH = "src/vllm_torchtpu/a.py"
_MESSAGE = "Object of class `NoneType` has no attribute `at`"


def _error(path=_PATH, message=_MESSAGE, line=1):
    return {
        "path": path,
        "name": "missing-attribute",
        "concise_description": message,
        "line": line,
    }


def _entry(path=_PATH, message=_MESSAGE):
    return {"path": path, "name": "missing-attribute", "concise_description": message}


def test_one_entry_per_error_passes():
    extra, stale = compare([_error(line=1), _error(line=2)], [_entry(), _entry()])
    assert extra == {} and stale == {}


def test_repeat_of_listed_message_is_reported():
    extra, stale = compare([_error(line=1), _error(line=9)], [_entry()])
    assert extra == {(_PATH, "missing-attribute", _MESSAGE): (1, 2)}
    assert stale == {}


def test_entry_for_fixed_error_is_reported():
    extra, stale = compare([_error()], [_entry(), _entry()])
    assert extra == {}
    assert stale == {(_PATH, "missing-attribute", _MESSAGE): (2, 1)}


def test_new_message_is_reported():
    extra, _ = compare([_error(message="Cannot set item in `None`")], [])
    assert extra == {(_PATH, "missing-attribute", "Cannot set item in `None`"): (0, 1)}


def test_absolute_path_matches_relative_entry():
    absolute = str(check_pyrefly_baseline.REPO_ROOT / _PATH)
    extra, stale = compare([_error(path=absolute)], [_entry()])
    assert extra == {} and stale == {}
