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


def test_get_tpu_metadata_whitespace_stripped():
    with patch("requests.get", return_value=MockResponse(200, "  tpu7x-8 \n")):
        result = tpu_info.get_tpu_metadata("accelerator-type")
        assert result == "tpu7x-8"


def test_get_tpu_metadata_200_empty_body(caplog):
    with patch("requests.get", return_value=MockResponse(200, "  \n  ")):
        with caplog.at_level("ERROR"):
            result = tpu_info.get_tpu_metadata("accelerator-type")
            assert result is None
            assert "Unable to poll TPU GCE Metadata" not in caplog.text


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


def test_get_tpu_metadata_error_with_percent_in_body(caplog):
    # Tests that response text with '%' does not crash Python logging formatting
    with patch(
            "requests.get",
            return_value=MockResponse(500, "Error 100% failed at %s")):
        with caplog.at_level("ERROR"):
            result = tpu_info.get_tpu_metadata("accelerator-type")
            assert result is None
            assert "Unable to poll TPU GCE Metadata for 'accelerator-type'" in caplog.text
            assert "100% failed at %s" in caplog.text


def test_get_tpu_metadata_leading_slash():
    with patch("requests.get", return_value=MockResponse(200, "2")) as mock_get:
        result = tpu_info.get_tpu_metadata("/agent-worker-number")
        assert result == "2"
        called_url = mock_get.call_args[0][0]
        assert called_url.endswith("attributes/agent-worker-number")
        assert not called_url.startswith("/agent-worker-number")


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


def test_get_node_worker_id_from_env_whitespace(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_WORKER_ID", "  4 \n")
    assert tpu_info.get_node_worker_id() == 4


def test_get_node_worker_id_from_env_invalid(monkeypatch, caplog):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_WORKER_ID", "invalid_id")
    with caplog.at_level("WARNING"):
        assert tpu_info.get_node_worker_id() == 0
        assert "Invalid TPU worker ID 'invalid_id'" in caplog.text


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


def test_get_node_worker_id_from_metadata_whitespace(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_WORKER_ID", None)
    with patch("requests.get", return_value=MockResponse(200, " 5 \n")):
        assert tpu_info.get_node_worker_id() == 5


def test_get_node_name_from_env(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_NAME", "tpu-test-node")
    assert tpu_info.get_node_name() == "tpu-test-node"


def test_get_node_name_from_metadata_404(monkeypatch, caplog):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_NAME", None)
    with patch("requests.get", return_value=MockResponse(404, "Not Found")):
        with caplog.at_level("ERROR"):
            assert tpu_info.get_node_name() is None
            assert "Unable to poll TPU GCE Metadata" not in caplog.text


def test_get_node_name_from_metadata_whitespace(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_NAME", None)
    with patch("requests.get", return_value=MockResponse(200, "  node-123 \n")):
        assert tpu_info.get_node_name() == "node-123"


def test_get_tpu_type_from_metadata_404(monkeypatch, caplog):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_ACCELERATOR_TYPE", None)
    with patch("requests.get", return_value=MockResponse(404, "Not Found")):
        with caplog.at_level("ERROR"):
            assert tpu_info.get_tpu_type() is None
            assert "Unable to poll TPU GCE Metadata" not in caplog.text


def test_get_tpu_type_from_metadata_whitespace(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_ACCELERATOR_TYPE", None)
    with patch("requests.get", return_value=MockResponse(200, "  v6e-8 \n")):
        assert tpu_info.get_tpu_type() == "v6e-8"


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


def test_get_num_chips(monkeypatch):
    with patch("glob.glob", return_value=["/dev/accel0", "/dev/accel1"]):
        assert tpu_info.get_num_chips() == 2

    with patch("glob.glob", return_value=[]), patch("os.listdir",
                                                    return_value=["0", "1", "2"]):
        assert tpu_info.get_num_chips() == 3

    with patch("glob.glob", return_value=[]), patch(
            "os.listdir", side_effect=PermissionError("access denied")):
        assert tpu_info.get_num_chips() == 0


def test_get_tpu_metadata_none_or_empty():
    with patch("requests.get") as mock_get:
        assert tpu_info.get_tpu_metadata(None) is None
        assert tpu_info.get_tpu_metadata("") is None
        assert tpu_info.get_tpu_metadata("   ") is None
        mock_get.assert_not_called()


def test_get_tpu_metadata_none_text_response():
    with patch("requests.get", return_value=MockResponse(200, None)):
        assert tpu_info.get_tpu_metadata("accelerator-type") is None


def test_get_chip_version_tpu_prefix():
    from jax.experimental.pallas import tpu as pltpu

    assert tpu_info.get_chip_version("tpu-v6e-8") == pltpu.ChipVersion.TPU_V6E
    assert tpu_info.get_chip_version("tpu7x-8") == pltpu.ChipVersion.TPU_7X
    assert tpu_info.get_chip_version("v5e-8") == pltpu.ChipVersion.TPU_V5E
    assert tpu_info.get_chip_version("tpu-v5e-8") == pltpu.ChipVersion.TPU_V5E
    assert tpu_info.get_chip_version("v6e-8") == pltpu.ChipVersion.TPU_V6E


def test_get_node_worker_id_from_env_empty_string(monkeypatch, caplog):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_WORKER_ID", "")
    with patch("requests.get", return_value=MockResponse(404, "Not Found")):
        with caplog.at_level("WARNING"):
            assert tpu_info.get_node_worker_id() == 0
            # Empty string should fall back cleanly to metadata without invalid ID warning
            assert "Invalid TPU worker ID" not in caplog.text


def test_get_node_worker_id_negative(monkeypatch, caplog):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_WORKER_ID", "-1")
    with caplog.at_level("WARNING"):
        assert tpu_info.get_node_worker_id() == 0
        assert "Invalid negative TPU worker ID '-1'" in caplog.text


def test_get_tpu_type_from_env_empty_string(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_ACCELERATOR_TYPE", "")
    with patch("requests.get", return_value=MockResponse(200, "v6e-8")):
        assert tpu_info.get_tpu_type() == "v6e-8"


def test_get_node_name_from_env_empty_string(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.envs.TPU_NAME", "")
    with patch("requests.get", return_value=MockResponse(200, "node-custom")):
        assert tpu_info.get_node_name() == "node-custom"


def test_get_num_cores_per_chip_variants(monkeypatch):
    for variant in ["V6E-8", "v5e-8", "tpu-v6e-8", "tpu-v5e-8", "V5LITEPOD-8"]:
        monkeypatch.setattr("vllm_torchtpu.tpu_info.get_tpu_type",
                            lambda v=variant: v)
        assert tpu_info.get_num_cores_per_chip() == 1, f"Failed for {variant}"

    for variant in ["TPU7X-8", "tpu7x-8", "v4-8", None]:
        monkeypatch.setattr("vllm_torchtpu.tpu_info.get_tpu_type",
                            lambda v=variant: v)
        assert tpu_info.get_num_cores_per_chip() == 2, f"Failed for {variant}"

