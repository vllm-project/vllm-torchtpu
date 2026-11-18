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
"""TPU test for Diffusion Session State Manager."""

from __future__ import annotations

import pytest
import torch
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.experimental.world_models.session_state import (
    LatentBuffer,
    SessionStateManager,
    resolve_session_state_config,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

_FRAME_SHAPE = (1, 4, 8)
_MAXLEN = 2
_MAX_SESSIONS = 2


def test_diffusion_session_state_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diffusion Session State Manager

    - TPU path: resolve ``enable_session_state_manager`` and ``max_sessions`` from
      ``OmniDiffusionConfig`` and env vars, then store, slice-alias, accumulate, and
      evict ``"tpu"`` bf16 ``LatentBuffer`` frames and ``SessionState.attrs``.
    - Setup check: ``enabled is True``, ``manager.max_sessions == 2``,
      ``LatentBuffer.resident is True``, and stored frame tensors reside on ``"tpu"``.
    - Outcome check: cross-call hit, count-based LRU eviction, storage-deduplicated
      ``"tpu"`` byte accounting, zeroed bytes after ``drop_session``, and exact
      parity of retained ``"tpu"`` frames against a CPU fp32 reference ring buffer.
    """
    od_cfg = OmniDiffusionConfig(
        model=tiny_diffusion_dir(),
        enable_session_state_manager=True,
    )
    monkeypatch.setenv("OMNI_DIFFUSION_SESSION_STATE_MANAGER", "1")
    monkeypatch.setenv(
        "OMNI_DIFFUSION_SESSION_STATE_MANAGER_MAX_SESSIONS", str(_MAX_SESSIONS)
    )
    enabled, max_sessions = resolve_session_state_config(
        enable=od_cfg.enable_session_state_manager
    )
    assert enabled is True, "SessionStateManager was not enabled"
    assert max_sessions == _MAX_SESSIONS, (
        f"max_sessions {max_sessions} != {_MAX_SESSIONS}"
    )

    manager = SessionStateManager(max_sessions=max_sessions)
    assert manager.max_sessions == _MAX_SESSIONS, (
        f"manager.max_sessions {manager.max_sessions} != {_MAX_SESSIONS}"
    )

    torch.manual_seed(0)
    frames_cpu = [torch.randn(_FRAME_SHAPE, dtype=torch.float32) for _ in range(3)]
    tpu_dev = torch.device("tpu")

    s1 = manager.get_or_create_session("s1")
    buf: LatentBuffer[torch.Tensor] = LatentBuffer()
    buf.allocate(maxlen=_MAXLEN)
    for frame in frames_cpu:
        buf.append(frame.to(dtype=torch.bfloat16, device=tpu_dev))
    s1.put("latents", buf)
    assert buf.resident is True, "LatentBuffer is not resident"
    items = buf.view()
    assert items, "LatentBuffer.view() returned no frames after append"
    assert items[-1].device.type == "tpu", (
        f"Stored LatentBuffer frame device {items[-1].device.type!r} != 'tpu'"
    )

    # Alias a slice of the latest frame in attrs to verify storage deduplication.
    s1.attrs["alias"] = items[-1][:, :2]
    expected_bytes = sum(t.untyped_storage().nbytes() for t in items)
    by_device = manager.nbytes_by_device()
    tpu_key = str(items[-1].device)
    assert (
        len({(str(t.device), t.untyped_storage().data_ptr()) for t in items}) == _MAXLEN
    ), "Expected 2 distinct storages in ring buffer"
    assert by_device.get(tpu_key) == expected_bytes, (
        f"nbytes_by_device[{tpu_key!r}]={by_device.get(tpu_key)} != {expected_bytes}"
    )

    # Cross-call hit promotes s1, then s2 + s3 evict s2 (LRU).
    assert manager.get_or_create_session("s1") is s1, (
        "Expected cache hit to return existing session 's1'"
    )
    assert manager.hits == 1, f"Expected manager.hits == 1, got {manager.hits}"
    manager.get_or_create_session("s2")
    assert manager.get_or_create_session("s1") is s1, (
        "Expected promoted session 's1' to remain the same object"
    )
    manager.get_or_create_session("s3")
    assert manager.evictions == 1, (
        f"Expected manager.evictions == 1, got {manager.evictions}"
    )
    assert "s2" not in manager, "LRU eviction failed to evict 's2'"
    assert "s1" in manager, "LRU eviction unexpectedly evicted promoted 's1'"
    assert "s3" in manager, "LRU eviction failed to retain newly created 's3'"

    # Retained frames in ring buffer match the last 2 CPU reference frames.
    retained = buf.view()
    assert len(retained) == _MAXLEN, (
        f"Expected {_MAXLEN} retained frames, got {len(retained)}"
    )
    for idx, (tpu_frame, cpu_frame) in enumerate(zip(retained, frames_cpu[-_MAXLEN:])):
        bf16_ref = cpu_frame.to(dtype=torch.bfloat16).float()
        assert_parity(
            tpu_frame.float(),
            bf16_ref,
            name=f"session_state_frame_{idx}",
            max_rel_l2=0.0,
            max_abs=0.0,
        )

    assert manager.drop_session("s1") is True, (
        "Expected drop_session('s1') to return True"
    )
    assert manager.nbytes_by_device() == {}, (
        f"Expected empty nbytes_by_device after drop, got {manager.nbytes_by_device()}"
    )
