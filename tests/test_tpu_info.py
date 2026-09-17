"""Tests for TPU info detection and metadata handling."""
from unittest.mock import patch

import requests

from vllm_torchtpu import tpu_info


class MockResponse:

    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text


def test_get_tpu_metadata_success():
    with patch("requests.get", return_value=MockResponse(200, "tpu7x-8")):
        result = tpu_info.get_tpu_metadata("accelerator-type")
        assert result == "tpu7x-8"


def test_get_tpu_metadata_404_handled_gracefully(caplog):
    with patch("requests.get", return_value=MockResponse(404, "Not Found")):
        with caplog.at_level("ERROR"):
            result = tpu_info.get_tpu_metadata("agent-worker-number")
            assert result is None
            # No ERROR log should be emitted for 404
            assert "Unable to poll TPU GCE Metadata" not in caplog.text


def test_get_tpu_metadata_error_status_logged(caplog):
    with patch(
            "requests.get",
            return_value=MockResponse(500, "Internal Server Error")):
        with caplog.at_level("ERROR"):
            result = tpu_info.get_tpu_metadata("accelerator-type")
            assert result is None
            assert "Unable to poll TPU GCE Metadata for 'accelerator-type'" in caplog.text


def test_get_tpu_metadata_request_exception(caplog):
    with patch(
            "requests.get",
            side_effect=requests.RequestException("connection timed out")):
        with caplog.at_level("ERROR"):
            result = tpu_info.get_tpu_metadata("accelerator-type")
            assert result is None
            assert "Unable to poll the TPU GCE Metadata: connection timed out" in caplog.text


def test_get_node_worker_id_from_env(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_WORKER_ID", "3")
    assert tpu_info.get_node_worker_id() == 3


def test_get_node_worker_id_from_metadata_404(monkeypatch, caplog):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_WORKER_ID", None)
    with patch("requests.get", return_value=MockResponse(404, "Not Found")):
        with caplog.at_level("ERROR"):
            worker_id = tpu_info.get_node_worker_id()
            assert worker_id == 0
            assert "Unable to poll TPU GCE Metadata" not in caplog.text


def test_get_node_worker_id_from_metadata_success(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_WORKER_ID", None)
    with patch("requests.get", return_value=MockResponse(200, "2")):
        assert tpu_info.get_node_worker_id() == 2


def test_get_node_name_from_env(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_NAME", "tpu-test-node")
    assert tpu_info.get_node_name() == "tpu-test-node"


def test_get_node_name_from_metadata_404(monkeypatch, caplog):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_NAME", None)
    with patch("requests.get", return_value=MockResponse(404, "Not Found")):
        with caplog.at_level("ERROR"):
            assert tpu_info.get_node_name() is None
            assert "Unable to poll TPU GCE Metadata" not in caplog.text


def test_get_tpu_type_from_metadata_404(monkeypatch, caplog):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_ACCELERATOR_TYPE", None)
    with patch("requests.get", return_value=MockResponse(404, "Not Found")):
        with caplog.at_level("ERROR"):
            assert tpu_info.get_tpu_type() is None
            assert "Unable to poll TPU GCE Metadata" not in caplog.text


def test_get_num_cores_per_chip(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.tpu_info.get_tpu_type", lambda: "v6e-8")
    assert tpu_info.get_num_cores_per_chip() == 1

    monkeypatch.setattr("vllm_torchtpu.tpu_info.get_tpu_type",
                        lambda: "v5litepod-8")
    assert tpu_info.get_num_cores_per_chip() == 1

    monkeypatch.setattr("vllm_torchtpu.tpu_info.get_tpu_type", lambda: "tpu7x-8")
    assert tpu_info.get_num_cores_per_chip() == 2

    monkeypatch.setattr("vllm_torchtpu.tpu_info.get_tpu_type", lambda: None)
    assert tpu_info.get_num_cores_per_chip() == 2
