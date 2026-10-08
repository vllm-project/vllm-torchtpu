"""Shared pytest fixtures for the experiments test suite."""

import pytest
from core import json_recorder


@pytest.fixture(autouse=True)
def _never_upload_to_gcs(monkeypatch):
    """Guarantees that no test can shell out to gsutil or touch GCS."""
    monkeypatch.setenv("EXPERIMENTS_NO_UPLOAD_GCS", "1")

    def _forbidden(self, *args, **kwargs):
        raise AssertionError("GCS upload attempted during tests")

    for name in ("_upload_traces_unlocked", "_upload_to_gcs_unlocked"):
        monkeypatch.setattr(json_recorder.JsonResultRecorder, name, _forbidden)
