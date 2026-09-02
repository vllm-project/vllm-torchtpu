# SPDX-License-Identifier: Apache-2.0
"""TPU platform plugin for vllm-omni."""

import os

import torch
from torch_tpu._internal.utils import hardware
from vllm.logger import init_logger
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

from vllm_torchtpu.omni.patches import apply_omni_tpu_patches
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

logger = init_logger(__name__)


class OmniTpuPlatform(OmniPlatform, TpuPlatform):
    """TPU implementation of OmniPlatform.

    Inherits all TPU implementations from vLLM's TpuPlatform and implements
    Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.OOT

    @property
    def dist_backend(self) -> str:
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        return TpuPlatform.get_worker_distributed_backend(
            world_size=world_size)

    def __init__(self) -> None:
        super().__init__()
        apply_omni_tpu_patches()

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = False,
    ) -> str:
        logger.info_once(
            "Using TpuSDPABackend for diffusion attention (requested: %s).",
            selected_backend)
        return "vllm_torchtpu.omni.attention.TpuSDPABackend"

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def supports_float64(cls) -> bool:
        return False

    @classmethod
    def supports_cpu_offload(cls) -> bool:
        return False

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        cls._prepare_singlehost_tpu_env(world_size)
        torch.accelerator.set_device_index(0)
        _ = torch.empty(1, device=torch.device("tpu"))
        torch.tpu.synchronize()

    @classmethod
    def get_torch_device(cls, _local_rank: int | None = None) -> torch.device:
        return torch.device("tpu")

    @classmethod
    def get_device_count(cls) -> int:
        return cls.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    def synchronize(cls) -> None:
        torch.tpu.synchronize()

    @classmethod
    def empty_cache(cls) -> None:
        pass

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        hbm = hardware.get_hbm_bytes_per_device()
        if hbm is not None:
            return hbm
        raise ValueError(
            f"Could not determine HBM memory for attached TPU device: {hardware.get_tpu_device_name()}"
        )

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.accelerator.get_memory_info(device)
        return free

    @classmethod
    def get_device_memory(
        cls,
        device: torch.device | None = None,
    ) -> tuple[int, int]:
        return torch.accelerator.get_memory_info(device)

    @classmethod
    def max_memory_reserved(cls, device: torch.device | None = None) -> int:
        return torch.accelerator.max_memory_reserved(device)

    @classmethod
    def max_memory_allocated(cls, device: torch.device | None = None) -> int:
        return torch.accelerator.max_memory_allocated(device)

    @classmethod
    def reset_peak_memory_stats(cls,
                                device: torch.device | None = None) -> None:
        torch.accelerator.reset_peak_memory_stats(device)

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_torchtpu.worker.tpu_worker.TPUWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_torchtpu.worker.tpu_worker.TPUWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/deploy"
