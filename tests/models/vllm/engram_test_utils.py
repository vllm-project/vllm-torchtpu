# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU transport for Engram packing tests; DMA lifetimes have separate tests."""

from types import SimpleNamespace

import torch


def install_cpu_engram_transfer(monkeypatch):
    from vllm_torchtpu.models.vllm.deepseek_v41.common import host_transfer
    from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import Engram

    monkeypatch.setattr(Engram, "layer_hash_index", 0, raising=False)

    class HostBuffer:
        def __init__(self, size_bytes):
            self.size_bytes = size_bytes
            self.words = torch.zeros(size_bytes // 4, dtype=torch.int32)

    class Transfer:
        def __init__(self, device, buffer, unsafe_skip_buffer_lock):
            self.device = device
            self.host = buffer.words

        def d2h(self):
            self.host.copy_(self.device)

        def h2d_async(self):
            return SimpleNamespace(wait=lambda: self.device.copy_(self.host))

    monkeypatch.setattr(host_transfer, "_physical_words", lambda tensor: tensor.numel())
    monkeypatch.setattr(
        host_transfer, "_host_words", lambda buffer, count: buffer.words[:count]
    )
    monkeypatch.setattr(
        host_transfer,
        "raw_transfer",
        SimpleNamespace(RawHostBuffer=HostBuffer, PreparedTorchRawTransfer=Transfer),
    )
    monkeypatch.setattr(
        host_transfer, "sync", SimpleNamespace(synchronize=lambda **kwargs: None)
    )
    activate = host_transfer.PreparedEngramUpload.activate

    def activate_cpu(upload):
        activate(upload)
        # CPU operations finish eagerly; these tests do not enqueue device graphs.
        upload.record_consumers([upload.device])

    monkeypatch.setattr(host_transfer.PreparedEngramUpload, "activate", activate_cpu)
