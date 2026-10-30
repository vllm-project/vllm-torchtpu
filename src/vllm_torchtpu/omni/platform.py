# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU platform plugin for vllm-omni."""

import os
from typing import TYPE_CHECKING

import torch
from torch_tpu._internal.utils import hardware
from vllm.logger import init_logger
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

from vllm_torchtpu.env_override import set_libtpu_flag_default
from vllm_torchtpu.omni.patches import apply_omni_tpu_patches
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_OMNI_CONFIG_CHECK_INSTALLED = False
_BASE_TPU_RUNTIME_METRICS_PORT = 8431


def _parse_visible_chip_ids(devices: str | int | None) -> list[int]:
    """Parse comma-separated non-negative TPU chip IDs for stage pinning."""
    if isinstance(devices, bool):
        raise ValueError(
            f"Invalid TPU visible devices specification: {devices!r}. "
            "Expected comma-separated non-negative integer chip IDs."
        )
    if devices is None:
        return []
    raw = str(devices).strip()
    if not raw:
        return []
    chip_ids: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            raise ValueError(
                f"Invalid TPU visible devices specification: {devices!r}. "
                "Empty chip ID entry is not allowed."
            )
        try:
            chip_id = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid TPU visible devices specification: {devices!r}. "
                f"Chip ID {token!r} is not an integer."
            ) from exc
        if chip_id < 0:
            raise ValueError(
                f"Invalid TPU visible devices specification: {devices!r}. "
                f"Chip ID {chip_id} must be non-negative."
            )
        if chip_id in seen:
            raise ValueError(
                f"Invalid TPU visible devices specification: {devices!r}. "
                f"Duplicate chip ID {chip_id} is not allowed."
            )
        seen.add(chip_id)
        chip_ids.append(chip_id)
    return chip_ids


def _install_tpu_platform_omni_config_check() -> None:
    """Validate OmniModelConfig when vLLM invokes TpuPlatform config checks."""
    global _OMNI_CONFIG_CHECK_INSTALLED
    if _OMNI_CONFIG_CHECK_INSTALLED:
        return
    orig_check_func = TpuPlatform.__dict__["check_and_update_config"].__func__

    def _checked_check_and_update_config(
        cls: type[TpuPlatform], vllm_config: "VllmConfig"
    ) -> None:
        OmniTpuPlatform._validate_omni_model_runner_config(vllm_config)
        orig_check_func(cls, vllm_config)

    TpuPlatform.check_and_update_config = classmethod(  # type: ignore[method-assign]
        _checked_check_and_update_config
    )
    _OMNI_CONFIG_CHECK_INSTALLED = True


class OmniTpuPlatform(OmniPlatform, TpuPlatform):
    """TPU implementation of OmniPlatform.

    Inherits all TPU implementations from vLLM's TpuPlatform and implements
    Omni-specific interfaces from OmniPlatform.
    """

    _omni_enum = OmniPlatformEnum.OOT

    # Diffusion worker world size recorded by set_device (see dist_backend).
    _worker_world_size: int = 1
    _saved_spawn_env: dict[str, str | None] | None = None

    @property
    def dist_backend(self) -> str:
        return TpuPlatform.get_worker_distributed_backend(
            world_size=self._worker_world_size
        )

    def __init__(self) -> None:
        super().__init__()
        apply_omni_tpu_patches()
        _install_tpu_platform_omni_config_check()

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        allow_trtllm_default: bool = False,
    ) -> str:
        from vllm_torchtpu.omni.patches import apply_omni_model_specific_patches

        apply_omni_model_specific_patches()
        if selected_backend is not None and selected_backend.upper() in (
            "SDPA",
            "TPU_SDPA",
            "TORCH_SDPA",
        ):
            backend = "vllm_torchtpu.omni.attention.TpuSDPABackend"
        else:
            backend = "vllm_torchtpu.omni.attention.TpuFlashAttentionBackend"
        logger.info_once(
            "Using %s for diffusion attention (requested: %s).",
            backend,
            selected_backend,
        )
        return backend

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def supports_diffusion_dense_flash_attention(cls) -> bool:
        return False

    @classmethod
    def get_profiler_cls(cls) -> str:
        return "vllm_torchtpu.omni.profiler.TpuOmniTorchProfilerWrapper"

    @classmethod
    def supports_float64(cls) -> bool:
        return False

    @classmethod
    def supports_cpu_offload(cls) -> bool:
        return False

    @classmethod
    def supports_talker_mtp_graph_capture(cls) -> bool:
        return False

    @staticmethod
    def _validate_omni_model_runner_config(vllm_config: "VllmConfig") -> None:
        from vllm_omni.config.model import OmniModelConfig

        model_config = vllm_config.model_config
        if (
            isinstance(model_config, OmniModelConfig)
            and model_config.use_v2_model_runner
        ):
            raise ValueError(
                "vLLM-Omni model_runner='v2' is not supported on TPU because "
                "TPUModelRunner implements the vLLM V1 runner contract. Set "
                "model_runner='v1' in the stage config (for example under "
                "platforms.tpu.model_runner or via --deploy-config)."
            )

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        cls._validate_omni_model_runner_config(vllm_config)
        super().check_and_update_config(vllm_config)

    @classmethod
    def set_device_control_env_var(cls, devices: str | int | None) -> None:
        """Configure TPU device-isolation env vars before spawning a stage."""
        chip_ids = _parse_visible_chip_ids(devices)
        if cls._saved_spawn_env is None:
            cls._saved_spawn_env = {
                "ALLOW_MULTIPLE_LIBTPU_LOAD": os.environ.get(
                    "ALLOW_MULTIPLE_LIBTPU_LOAD"
                ),
                "DEBUG_TPU_LOCAL_RANK_OFFSET": os.environ.get(
                    "DEBUG_TPU_LOCAL_RANK_OFFSET"
                ),
                "MASTER_PORT": os.environ.get("MASTER_PORT"),
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": os.environ.get(
                    "TORCH_TPU_SLICEBUILDER_ADDRESSES"
                ),
                "TORCH_TPU_TOPOLOGY": os.environ.get("TORCH_TPU_TOPOLOGY"),
                "TPU_RUNTIME_METRICS_PORTS": os.environ.get(
                    "TPU_RUNTIME_METRICS_PORTS"
                ),
                "TPU_VISIBLE_DEVICES": os.environ.get("TPU_VISIBLE_DEVICES"),
                "VLLM_TPU_DEVICE_ORDER": os.environ.get("VLLM_TPU_DEVICE_ORDER"),
                "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            }
        if not chip_ids:
            os.environ[cls.device_control_env_var] = ""
            os.environ["TPU_VISIBLE_DEVICES"] = ""
            os.environ["TPU_RUNTIME_METRICS_PORTS"] = ""
            os.environ.pop("DEBUG_TPU_LOCAL_RANK_OFFSET", None)
            os.environ.pop("VLLM_TPU_DEVICE_ORDER", None)
            return

        normalized = ",".join(str(chip_id) for chip_id in chip_ids)
        os.environ[cls.device_control_env_var] = normalized
        os.environ["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"
        # Isolate each stage's libtpu Runtime Metric Service port(s) by physical
        # chip ID; when unset, GCE TPU VM metadata supplies 8431,8432,... and
        # every stage's local rank 0 collides on port 8431.
        os.environ["TPU_RUNTIME_METRICS_PORTS"] = ",".join(
            str(_BASE_TPU_RUNTIME_METRICS_PORT + chip_id) for chip_id in chip_ids
        )
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("MASTER_PORT", None)
        os.environ.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
        os.environ.pop("TORCH_TPU_TOPOLOGY", None)

        if len(chip_ids) == 1:
            os.environ["TPU_VISIBLE_DEVICES"] = str(chip_ids[0])
            os.environ.pop("DEBUG_TPU_LOCAL_RANK_OFFSET", None)
            os.environ.pop("VLLM_TPU_DEVICE_ORDER", None)
            return

        # Never set TPU_VISIBLE_DEVICES to a multi-chip list: torch_tpu falls
        # back to LOCAL_RANK when TPU_VISIBLE_DEVICES is not a single integer.
        os.environ.pop("TPU_VISIBLE_DEVICES", None)
        is_contiguous = chip_ids == list(
            range(chip_ids[0], chip_ids[0] + len(chip_ids))
        )
        if is_contiguous:
            os.environ["DEBUG_TPU_LOCAL_RANK_OFFSET"] = str(chip_ids[0])
            os.environ.pop("VLLM_TPU_DEVICE_ORDER", None)
        else:
            os.environ["VLLM_TPU_DEVICE_ORDER"] = normalized
            os.environ.pop("DEBUG_TPU_LOCAL_RANK_OFFSET", None)

        if cls.device_count() > 0:
            cls._prepare_singlehost_tpu_env(len(chip_ids))

    @classmethod
    def unset_device_control_env_var(cls) -> None:
        """Restore parent TPU env vars after spawning a stage subprocess."""
        os.environ.pop(cls.device_control_env_var, None)
        saved = cls._saved_spawn_env
        cls._saved_spawn_env = None
        if saved is None:
            return
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    @classmethod
    def set_device(cls, device: torch.device | int) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        if world_size > 1:
            cls._worker_world_size = world_size
            chips = os.environ.get("TPU_VISIBLE_CHIPS")
            if chips and [int(c) for c in chips.split(",")] != list(range(world_size)):
                logger.warning(
                    "Diffusion stage is pinned to TPU_VISIBLE_CHIPS=%s, but its "
                    "%d ranks bind chips 0-%d by local rank. Multi-rank diffusion "
                    "stages on other chips (e.g. in multi-stage pipelines) are "
                    "not supported on TPU yet.",
                    chips,
                    world_size,
                    world_size - 1,
                )
            cls._set_shared_slicebuilder_addresses(world_size)
            cls._offload_dense_all_to_all()
        cls._prepare_singlehost_tpu_env(world_size)
        torch.accelerator.set_device_index(0)
        _ = torch.empty(1, device=torch.device("tpu"))
        torch.accelerator.synchronize()

    @staticmethod
    def _offload_dense_all_to_all() -> None:
        """Run dense all-to-alls on SparseCore in multi-rank diffusion workers.

        Ulysses sequence parallelism moves q, k, v and the attention output
        through dense all-to-alls in every transformer block. By default XLA
        runs them on the TensorCore, which waits for them.
        """
        # TODO: Remove once libtpu enables the SparseCore dense all-to-all offload by
        # default. This sets os.environ after torch_tpu is imported, but it still
        # takes effect because libtpu reads LIBTPU_INIT_ARGS when the PjRt client
        # is created on the first device allocation, the same pre-PjRt window
        # that _set_shared_slicebuilder_addresses relies on.
        set_libtpu_flag_default(
            "--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all", "true"
        )

    @staticmethod
    def _set_shared_slicebuilder_addresses(world_size: int) -> None:
        """Give every diffusion worker the same PjRt slicebuilder addresses.

        Root cause: vLLM-Omni spawns diffusion workers with no platform hook
        in the parent, so each worker reaches ``_prepare_singlehost_tpu_env``
        without TORCH_TPU_SLICEBUILDER_ADDRESSES and picks its own random
        ports. torch_tpu needs one identical list on every rank, otherwise
        the slice bootstrap never completes. The workers already share
        MASTER_PORT (DiffusionWorker exports od_config.master_port), so the
        list is derived from it.

        An inherited list is replaced even when its size matches: in
        multi-stage pipelines it is an LLM stage's list (built in the
        orchestrator process), and sharing it hangs that stage's workers.
        """
        # TODO: Remove once vLLM-Omni calls a platform hook in the parent before
        # spawning diffusion workers, so every rank inherits one address list, or
        # torch_tpu allows more than one addressable device per process.
        base_port = int(os.environ["MASTER_PORT"])
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
            f"localhost:{base_port + 1 + i}" for i in range(world_size)
        )

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
    def synchronize(cls, device: torch.device | int | None = None) -> None:
        torch.accelerator.synchronize(device)

    @classmethod
    def empty_cache(cls) -> None:
        pass

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        hbm = hardware.get_hbm_bytes_per_device()
        if hbm is not None:
            return hbm
        raise ValueError(
            "Could not determine HBM memory for attached TPU device: "
            f"{hardware.get_tpu_device_name()}"
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
    def reset_peak_memory_stats(cls, device: torch.device | None = None) -> None:
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
