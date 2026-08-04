# ruff: noqa
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

_tpu_info_logged = False


def _log_tpu_info() -> None:
    global _tpu_info_logged
    if _tpu_info_logged:
        return
    _tpu_info_logged = True

    from vllm_torchtpu import tpu_info as ti
    from vllm_torchtpu.logger import init_logger

    logger = init_logger(__name__)
    try:
        logger.info(
            "TPU info: node_name=%s | tpu_type=%s | worker_id=%s | "
            "num_chips=%s | num_cores_per_chip=%s", ti.get_node_name(),
            ti.get_tpu_type(), ti.get_node_worker_id(), ti.get_num_chips(),
            ti.get_num_cores_per_chip())
    except Exception as error:
        logger.error(
            "Error occurred while logging TPU info: %s. "
            "Are you running on CPU?", error)


def _register_tpu_kv_connectors() -> None:
    """Register the TPU KV connectors in the factory's name registry.

    Recipes load these via kv_connector_module_path, which resolves the
    class without registering it, and the API server process never
    instantiates connectors at all — but its metrics path rebuilds
    TPUMultiConnector child stats through get_connector_class_by_name,
    which is registry-only. Without this, the first stats batch naming a
    child (e.g. TPURaidenOffloadingConnector) kills the AsyncLLM output
    handler with "Connector ... is not registered" and every subsequent
    request 500s (j-39fe6677). The plugin entry point runs in every vllm
    process, so registering here covers the API server too. Lazy loaders:
    nothing is imported until a name is actually looked up.
    """
    from vllm.distributed.kv_transfer.kv_connector.factory import \
        KVConnectorFactory

    for name, module_path in (
        ("TPURaidenConnector",
         "vllm_torchtpu.distributed.kv_transfer.tpu_connector"),
        ("TPUMultiConnector",
         "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector"),
        ("TPURaidenOffloadingConnector",
         "vllm_torchtpu.offload.raiden_connector"),
    ):
        if name not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(name, module_path, name)


def register_tpu_platform() -> str:
    """vLLM out-of-tree platform plugin entry point."""
    _log_tpu_info()
    _register_tpu_kv_connectors()
    return f"{TpuPlatform.__module__}.{TpuPlatform.__qualname__}"
