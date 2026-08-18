from unittest.mock import MagicMock, patch
from vllm_torchtpu import envs
from vllm_torchtpu.distributed import utils


def test_kv_transfer_names_default(monkeypatch):
    monkeypatch.delenv("TPU_KV_TRANSFER_NAMESPACE", raising=False)

    assert utils.get_ipc_socket_path(0) == "ipc:///tmp/tpu_conn_node0_0.sock"
    assert utils.get_shm_name(0) == "tpu_conn_kv_node0_0"


def test_kv_transfer_names_are_namespaced(monkeypatch):
    monkeypatch.setenv("TPU_KV_TRANSFER_NAMESPACE", "prefill")

    assert utils.get_ipc_socket_path(
        0) == "ipc:///tmp/tpu_conn_prefill_node0_0.sock"
    assert utils.get_shm_name(0) == "tpu_conn_kv_prefill_node0_0"


def test_kv_transfer_namespace_is_sanitized(monkeypatch):
    monkeypatch.setenv("TPU_KV_TRANSFER_NAMESPACE", "decode:9400")

    assert utils.get_ipc_socket_path(
        2) == "ipc:///tmp/tpu_conn_decode_9400_node2_0.sock"
    assert utils.get_shm_name(2) == "tpu_conn_kv_decode_9400_node2_0"


def test_get_raiden_telemetry_module_success(monkeypatch):
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_MODULE", None)
    mock_impl = MagicMock()
    mock_kcm = MagicMock()
    mock_kcm._torch_impl.return_value = mock_impl
    mock_torch = MagicMock()
    mock_torch.kv_cache_manager = mock_kcm
    mock_api = MagicMock()
    mock_api.torch = mock_torch
    mock_tpu_sync = MagicMock()
    mock_tpu_sync.api = mock_api

    with patch.dict(
        "sys.modules",
        {
            "tpu_sync": mock_tpu_sync,
            "tpu_sync.api": mock_api,
            "tpu_sync.api.torch": mock_torch,
            "tpu_sync.api.torch.kv_cache_manager": mock_kcm,
        },
    ):
        result = utils.get_raiden_telemetry_module()
        assert result is mock_impl
        mock_kcm._torch_impl.assert_called_once()

        # Check caching behavior
        result_cached = utils.get_raiden_telemetry_module()
        assert result_cached is mock_impl
        mock_kcm._torch_impl.assert_called_once()


def test_get_raiden_telemetry_module_import_failure(monkeypatch):
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_MODULE", None)
    with patch.dict("sys.modules", {"tpu_sync": None, "tpu_sync.api.torch.kv_cache_manager": None}):
        result = utils.get_raiden_telemetry_module()
        assert result is None


def test_get_raiden_telemetry_module_torch_impl_exception(monkeypatch):
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_MODULE", None)
    mock_kcm = MagicMock()
    mock_kcm._torch_impl.side_effect = RuntimeError("Extension load error")
    mock_torch = MagicMock()
    mock_torch.kv_cache_manager = mock_kcm
    mock_api = MagicMock()
    mock_api.torch = mock_torch
    mock_tpu_sync = MagicMock()
    mock_tpu_sync.api = mock_api

    with patch.dict(
        "sys.modules",
        {
            "tpu_sync": mock_tpu_sync,
            "tpu_sync.api": mock_api,
            "tpu_sync.api.torch": mock_torch,
            "tpu_sync.api.torch.kv_cache_manager": mock_kcm,
        },
    ):
        result = utils.get_raiden_telemetry_module()
        assert result is None


def test_configure_raiden_telemetry_default(monkeypatch):
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
    mock_telemetry = MagicMock()
    monkeypatch.setattr(utils, "get_raiden_telemetry_module", lambda: mock_telemetry)

    utils.configure_raiden_telemetry()

    mock_telemetry.configure_telemetry.assert_called_once_with({"buffered"})
    assert utils._RAIDEN_TELEMETRY_CONFIGURED is True


def test_configure_raiden_telemetry_explicit_backends(monkeypatch):
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
    mock_telemetry = MagicMock()
    monkeypatch.setattr(utils, "get_raiden_telemetry_module", lambda: mock_telemetry)

    # Test list of backends
    utils.configure_raiden_telemetry(backends=["buffered", "file"])
    mock_telemetry.configure_telemetry.assert_called_with({"buffered", "file"})

    # Test set of backends
    utils.configure_raiden_telemetry(backends={"custom"})
    mock_telemetry.configure_telemetry.assert_called_with({"custom"})


def test_configure_raiden_telemetry_env_override(monkeypatch):
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
    mock_telemetry = MagicMock()
    monkeypatch.setattr(utils, "get_raiden_telemetry_module", lambda: mock_telemetry)
    monkeypatch.setattr(envs, "TPU_RAIDEN_TELEMETRY_BACKENDS", "buffered, otel, custom", raising=False)

    utils.configure_raiden_telemetry()
    mock_telemetry.configure_telemetry.assert_called_once_with({"buffered", "otel", "custom"})


def test_configure_raiden_telemetry_no_module(monkeypatch):
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
    monkeypatch.setattr(utils, "get_raiden_telemetry_module", lambda: None)

    # Should execute cleanly without errors
    utils.configure_raiden_telemetry()
    assert utils._RAIDEN_TELEMETRY_CONFIGURED is False


def test_configure_raiden_telemetry_missing_configure_attribute(monkeypatch):
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
    mock_telemetry = object()  # Has no configure_telemetry
    monkeypatch.setattr(utils, "get_raiden_telemetry_module", lambda: mock_telemetry)

    utils.configure_raiden_telemetry()
    assert utils._RAIDEN_TELEMETRY_CONFIGURED is False


def test_configure_raiden_telemetry_exception_handled(monkeypatch):
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
    mock_telemetry = MagicMock()
    mock_telemetry.configure_telemetry.side_effect = RuntimeError("Config failed")
    monkeypatch.setattr(utils, "get_raiden_telemetry_module", lambda: mock_telemetry)

    # Should catch exception and not re-raise
    utils.configure_raiden_telemetry()
    assert utils._RAIDEN_TELEMETRY_CONFIGURED is False

