# SPDX-License-Identifier: Apache-2.0
"""Resolve the opt-in split-SparseCore MoE backend before tracing."""

import shlex
from functools import cache

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.fused_moe.v2.preprocessing import (
    PreprocessTopology,
    topology_for_coordinates,
)


@cache
def prepare_moe_preprocessing(mesh) -> PreprocessTopology | None:
    """Retain native preprocessing on unsupported hardware or rank order."""
    if envs.MOE_FUSED_EP_PREPROCESS_BACKEND == "native":
        return None
    devices = list(mesh.devices.flat)
    if len(devices) != 8 or any(d.device_kind != "TPU7x" for d in devices):
        return None
    # TorchTPU bridge devices do not expose coordinates; resolve their IDs
    # against the runtime's global physical-device attributes at the adapter.
    if not hasattr(devices[0], "coords"):
        from torch_tpu._internal.distributed import tpu_distributed

        physical = {d.id: d for d in tpu_distributed.global_device_attributes()}
        devices = [physical[d.id] for d in devices]
    coordinates = tuple((*tuple(d.coords), int(d.core_on_chip)) for d in devices)
    topology = topology_for_coordinates(coordinates)
    if topology is None:
        return None
    flags = dict(
        token.split("=", 1)
        for token in shlex.split(envs.LIBTPU_INIT_ARGS)
        if token.startswith("--") and "=" in token
    )
    required = {
        "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes": "0",
        "--xla_mosaic_unsafe_allow_multicore_remote_dma": "true",
    }
    if any(flags.get(key) != value for key, value in required.items()):
        raise ValueError(
            "MOE_FUSED_EP_PREPROCESS_BACKEND=topology requires startup "
            "LIBTPU_INIT_ARGS: " + " ".join(f"{k}={v}" for k, v in required.items())
        )
    from vllm_torchtpu.compilation.moe_schedule import install_moe_schedule

    install_moe_schedule()
    return topology
