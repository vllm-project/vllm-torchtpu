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


def register_tpu_platform() -> str:
    """vLLM out-of-tree platform plugin entry point."""
    _log_tpu_info()
    # Config-schema patches must land here rather than in apply_tpu_patches():
    # this entry point runs while plugins are loaded, whereas
    # apply_tpu_patches() runs from check_and_update_config(), by which point
    # ProfilerConfig has already been constructed and would have rejected
    # profiler='torch_phased'.
    from vllm_torchtpu import _patch_vllm_profiler_kind_torch_phased
    _patch_vllm_profiler_kind_torch_phased()
    return f"{TpuPlatform.__module__}.{TpuPlatform.__qualname__}"
