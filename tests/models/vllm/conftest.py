# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures for model CPU tests."""

import pytest
from engram_test_utils import install_cpu_engram_transfer


@pytest.fixture
def cpu_engram_transfer(monkeypatch):
    install_cpu_engram_transfer(monkeypatch)
