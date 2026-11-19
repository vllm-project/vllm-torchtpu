# SPDX-License-Identifier: Apache-2.0
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
