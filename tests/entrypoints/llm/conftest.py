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
"""Test configuration for the in-process engine suites."""

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Record each phase's result on the item so fixtures can read it.

    A teardown fixture cannot otherwise tell a passing test from a failing one.
    pytest collects hooks from conftest files, not from test modules, and
    applies them only to tests under the directory holding the file.
    """
    outcome = yield
    setattr(item, f"report_{call.when}", outcome.get_result())
