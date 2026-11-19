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
"""TPU tests for WorldKV pose-indexed KV memory (LingBot World)."""

from __future__ import annotations

import pytest
import torch
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

CHUNK_FRAMES = 2
TOKENS_PER_FRAME = 4
NUM_HEADS = 2
HEAD_DIM = 16


def _rigid_pose(tx: float, ty: float, tz: float) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float64)
    pose[0, 3] = tx
    pose[1, 3] = ty
    pose[2, 3] = tz
    return pose


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_worldkv_pose_indexed_kv_memory() -> None:
    """WorldKV Pose-Indexed KV Memory (LingBot World).

    - TPU path: import ``WorldKVConfig`` and ``WorldKV``, commit contiguous
      bf16 KV chunks on ``torch.device("tpu")`` across distinct camera poses so
      post-sink chunks bank to CPU memory, then revisit the first pose.
    - Setup check: ``current_omni_platform.device_type == "tpu"``, parsed
      ``WorldKVConfig`` validates window/sink/retrieval geometry, and evicted
      post-sink chunks are recorded in ``world_kv.bank``.
    - Outcome check: revisiting pose 1 retrieves its banked KV chunk back onto
      ``"tpu"`` with exact parity against the original committed tensors.
    """
    from vllm_omni.experimental.ar_diffusion.worldkv import (
        WorldKV,
        WorldKVConfig,
    )

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    cfg = WorldKVConfig.parse(
        {
            "window_frames": 8,
            "sink_frames": 2,
            "retrieval_frames": 2,
            "bank_bytes": 4 * 1024 * 1024,
            "world_keep_ratio": 1.0,
            "world_score": "centroid",
        },
        window_frames=8,
        sink_frames=2,
        chunk_frames=CHUNK_FRAMES,
    )
    assert cfg is not None, "WorldKVConfig.parse returned None"

    torch.manual_seed(0)
    tpu_device = torch.device("tpu")
    tokens = CHUNK_FRAMES * TOKENS_PER_FRAME
    shape = (1, tokens, NUM_HEADS, HEAD_DIM)

    world_kv = WorldKV(cfg, num_layers=1, tokens_per_frame=TOKENS_PER_FRAME)
    try:
        poses = [
            _rigid_pose(0.0, 0.0, 0.0),
            _rigid_pose(1.0, 0.0, 0.0),
            _rigid_pose(5.0, 0.0, 0.0),
            _rigid_pose(10.0, 0.0, 0.0),
        ]
        committed_k: list[torch.Tensor] = []
        committed_v: list[torch.Tensor] = []

        for idx, pose in enumerate(poses):
            q_cpu = torch.randn(shape, dtype=torch.bfloat16)
            k_cpu = torch.randn(shape, dtype=torch.bfloat16)
            v_cpu = torch.randn(shape, dtype=torch.bfloat16)
            committed_k.append(k_cpu)
            committed_v.append(v_cpu)
            world_kv.set_pose(pose)
            with world_kv.forward(idx * CHUNK_FRAMES, CHUNK_FRAMES, commit=True):
                world_kv.assemble(
                    0,
                    q_cpu.to(tpu_device),
                    k_cpu.to(tpu_device),
                    v_cpu.to(tpu_device),
                )

        assert world_kv.next_frame == len(poses) * CHUNK_FRAMES, (
            f"Expected next_frame={len(poses) * CHUNK_FRAMES}, "
            f"got {world_kv.next_frame}"
        )
        assert CHUNK_FRAMES in world_kv.bank, (
            f"Expected chunk starting at frame {CHUNK_FRAMES} in CPU bank, "
            f"got keys {sorted(world_kv.bank)}"
        )

        # Revisit pose 1 (chunk 1, now evicted from recent tail) and verify retrieval.
        world_kv.set_pose(poses[1])
        q_rev = torch.randn(shape, dtype=torch.bfloat16, device=tpu_device)
        k_rev = torch.randn(shape, dtype=torch.bfloat16, device=tpu_device)
        v_rev = torch.randn(shape, dtype=torch.bfloat16, device=tpu_device)
        with world_kv.forward(world_kv.next_frame, CHUNK_FRAMES, commit=False):
            assembled_k, assembled_v = world_kv.assemble(0, q_rev, k_rev, v_rev)

        assert assembled_k.device.type == "tpu", (
            f"Expected assembled_k on tpu, got {assembled_k.device}"
        )
        assert assembled_v.device.type == "tpu", (
            f"Expected assembled_v on tpu, got {assembled_v.device}"
        )
        sink_tokens = cfg.sink_frames * TOKENS_PER_FRAME
        retrieved_k = assembled_k[:, sink_tokens : sink_tokens + tokens]
        retrieved_v = assembled_v[:, sink_tokens : sink_tokens + tokens]
        assert_parity(
            retrieved_k,
            committed_k[1],
            name="worldkv_retrieved_k",
            max_rel_l2=0.0,
            max_abs=0.0,
        )
        assert_parity(
            retrieved_v,
            committed_v[1],
            name="worldkv_retrieved_v",
            max_rel_l2=0.0,
            max_abs=0.0,
        )
    finally:
        world_kv.close()
