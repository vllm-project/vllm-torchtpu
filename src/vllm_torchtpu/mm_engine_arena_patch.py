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
"""Large multimodal tensors from the API server to EngineCore through shared memory.

The API server sends each request to EngineCore over a ZMQ ipc socket, the
pixels as zero-copy frames. A unix socket still copies them twice in the
kernel, once into it and once out: 6-7 ms for a 3-image request's 36 MB of
pixels, at ~6 GB/s. Here they take the shared-memory tensor arena EngineCore
already uses for its workers: the API server writes each tensor of at least
`MIN_BYTES` into a slot of its own arena (1.3 ms for 32 MB on the copy pool,
see `mm_host_copy_patch`), the message carries the slot, and EngineCore reads
a zero-copy view of it. The slot frees once EngineCore drops the tensor. A
tensor the arena cannot take, as when all its slots are in use, takes the
socket as before.

The two ends ride on msgspec's out-of-band tensor seam (vLLM's
`OOBTensorConsumer` and tensor provider), which vLLM's own `torch_shm`
transport uses; they are installed only where that transport is off.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import torch
from vllm.distributed.device_communicators.shm_tensor_arena import (
    _ARENA_MIN_BYTES,
    ShmTensorArena,
)
from vllm.v1.engine import core, core_client
from vllm.v1.serial_utils import OOBTensorConsumer

# The upstream code this module wraps; the tests pin their sources.
UPSTREAM_CLIENT_INIT = core_client.MPClient.__init__
UPSTREAM_PROCESS_INPUT_SOCKETS = core.EngineCoreProc.process_input_sockets

# The size from which EngineCore's own arena takes tensors.
MIN_BYTES = _ARENA_MIN_BYTES


class ArenaTensorSender(OOBTensorConsumer):
    """API server end: writes large CPU tensors into its arena (one reader,
    EngineCore), made on first use."""

    def __init__(self) -> None:
        self._arena: ShmTensorArena | None = None

    def new_message(self) -> None:
        """Slots carry no per-message state."""

    def __call__(self, tensor: torch.Tensor) -> dict | None:
        if not (
            type(tensor) is torch.Tensor
            and tensor.is_cpu
            and tensor.layout == torch.strided
            and tensor.is_contiguous()
            and not tensor.requires_grad
            and tensor.nbytes >= MIN_BYTES
        ):
            return None
        if self._arena is None:
            self._arena = ShmTensorArena(n_reader=1)
        slot = self._arena.write_tensor(tensor)
        if slot is None:
            return None
        return {"arena": self._arena.handle(), "slot": slot}


class ArenaTensorReceiver:
    """EngineCore end: a zero-copy view of the slot; the slot frees at the
    next call after the view's storage dies."""

    def __init__(self) -> None:
        self._arenas: dict[str, ShmTensorArena] = {}

    def __call__(
        self, dtype: str, shape: tuple[int, ...], meta: dict[str, Any]
    ) -> torch.Tensor:
        n_reader, slot_bytes, n_slots, name = meta["arena"]
        arena = self._arenas.get(name)
        if arena is None:
            arena = self._arenas[name] = ShmTensorArena(
                n_reader, slot_bytes, n_slots, name=name, reader_rank=0
            )
        arena.flush_releases()
        torch_dtype = getattr(torch, dtype)
        nbytes = math.prod(shape) * torch_dtype.itemsize
        return arena.get_tensor(meta["slot"], nbytes, torch_dtype, tuple(shape))


@functools.wraps(UPSTREAM_CLIENT_INIT)
def _client_init(self: core_client.MPClient, *args: Any, **kwargs: Any) -> None:
    UPSTREAM_CLIENT_INIT(self, *args, **kwargs)
    encoder = getattr(self, "encoder", None)
    if encoder is not None and encoder.oob_tensor_consumer is None:
        encoder.oob_tensor_consumer = ArenaTensorSender()


@functools.wraps(UPSTREAM_PROCESS_INPUT_SOCKETS)
def _process_input_sockets(self: core.EngineCoreProc, *args: Any, **kwargs: Any) -> None:
    if self.tensor_ipc_receiver is None:
        self.tensor_ipc_receiver = ArenaTensorReceiver()
    return UPSTREAM_PROCESS_INPUT_SOCKETS(self, *args, **kwargs)


def patch_mm_engine_arena() -> None:
    """Send the API server's large tensors to EngineCore through an arena."""
    core_client.MPClient.__init__ = _client_init
    core.EngineCoreProc.process_input_sockets = _process_input_sockets
