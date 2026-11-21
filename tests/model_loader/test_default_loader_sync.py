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
"""Unit tests for the DefaultModelLoader per-tensor device sync.

Under TorchTPU's deferred eager mode every shard write into a fused parameter
keeps a whole-parameter copy alive until the device queue flushes, so the
loader must sync after the model consumes each checkpoint tensor. No TPU, no
network.
"""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_torchtpu.model_loader_patches import (
    patch_default_loader_sync_weight_writes,
)

_CHECKPOINT = {
    "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 1.0),
    "model.layers.0.self_attn.k_proj.weight": torch.full((2, 2), 2.0),
    "model.layers.0.self_attn.v_proj.weight": torch.full((2, 2), 3.0),
}


@pytest.fixture
def patched_loader(tmp_path, monkeypatch):
    """A DefaultModelLoader over a one-shard checkpoint, with syncs recorded."""
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import default_loader

    loader_cls = default_loader.DefaultModelLoader
    # setattr registers teardown that restores the unpatched class.
    monkeypatch.setattr(loader_cls, "get_all_weights", loader_cls.get_all_weights)
    monkeypatch.setattr(loader_cls, "_tpu_load_sync_patch", False, raising=False)
    events: list[str] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))

    save_file(_CHECKPOINT, str(tmp_path / "model.safetensors"))
    loader = object.__new__(loader_cls)
    loader.load_config = LoadConfig(load_format="safetensors")
    loader.local_expert_ids = None
    loader._encoder_only_lm_prefixes = None
    model_config = SimpleNamespace(
        model=str(tmp_path), revision=None, quantization=None
    )

    def load(model: SimpleNamespace) -> None:
        for name, _ in loader.get_all_weights(model_config, model):
            events.append(name)

    return load, events


def test_device_synced_after_each_consumed_tensor(patched_loader):
    load, events = patched_loader
    patch_default_loader_sync_weight_writes()

    load(SimpleNamespace())

    loaded = [event for event in events if event != "sync"]
    assert sorted(loaded) == sorted(_CHECKPOINT)
    assert events == [e for name in loaded for e in (name, "sync")]


def test_repeated_patch_syncs_once_per_tensor(patched_loader):
    load, events = patched_loader
    patch_default_loader_sync_weight_writes()
    patch_default_loader_sync_weight_writes()

    load(SimpleNamespace())

    assert events.count("sync") == len(_CHECKPOINT)
