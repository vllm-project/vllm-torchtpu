"""Shared pytest configuration and fixtures for the experiments test suite."""

import os

import pytest
from core import json_recorder

# Hugging Face Hub must be unreachable from unit tests. huggingface_hub
# reads HF_HUB_OFFLINE at import time, so set it here (conftest is loaded
# before any test module) in addition to the per-test fixture below.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@pytest.fixture(autouse=True)
def hf_hub_offline(monkeypatch):
    """Keeps every test hermetic w.r.t. the Hugging Face Hub."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    yield


@pytest.fixture(autouse=True)
def _never_upload_to_gcs(monkeypatch):
    """Guarantees that no test can shell out to gsutil or touch GCS."""
    monkeypatch.setenv("EXPERIMENTS_NO_UPLOAD_GCS", "1")

    def _forbidden(self, *args, **kwargs):
        raise AssertionError("GCS upload attempted during tests")

    for name in ("_upload_traces_unlocked", "_upload_to_gcs_unlocked"):
        monkeypatch.setattr(json_recorder.JsonResultRecorder, name, _forbidden)
