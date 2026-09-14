# ruff: noqa
"""The TPU platform plugin.

vLLM resolves its current platform while ``vllm`` itself is being imported,
by calling every platform plugin's entry point; ours is
``register_tpu_platform`` below. The entry point and ``vllm`` are therefore
set up before this package imports its platform module, so a process whose
first vLLM-related import is a module of this package (a test file imported
on its own, for one) does not call the entry point on a half-initialized
package.
"""

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
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
    _log_tpu_info()
    return f"{TpuPlatform.__module__}.{TpuPlatform.__qualname__}"


import vllm  # noqa: E402,F401  # isort: skip

from vllm_torchtpu.platforms.tpu_platform import TpuPlatform  # noqa: E402
