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
"""The pooled host copies give the arena and the batch upstream's bytes."""

import hashlib
import inspect

import pytest
import torch
import vllm
from vllm.distributed.device_communicators.shm_tensor_arena import ShmTensorArena
from vllm.multimodal.inputs import MultiModalFlatField

from vllm_torchtpu import mm_host_copy_patch as patch_mod

PINNED_VLLM_VERSION = "0.33.0"
PINNED_SOURCE_HASHES = {
    "__init__": ("f550f1794a43766b415fbf3adf0728becc071a585e00d45d0b63cc43ff26b1e7"),
    "write_tensor": (
        "b68dfda05bea53ee9522e65519f14519d6a0c2cff2f393fc3717bd2b2e23afee"
    ),
    "_reduce_data": (
        "962c5b6c7dcfb6c39c8c269d8f3e0ed1bbc8592485861b929b3ea100fd8a39b6"
    ),
}


@pytest.mark.skipif(
    vllm.__version__ != PINNED_VLLM_VERSION,
    reason=f"write_tensor reproduces vLLM {PINNED_VLLM_VERSION}",
)
@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("__init__", patch_mod.UPSTREAM_ARENA_INIT),
        ("write_tensor", patch_mod.UPSTREAM_WRITE_TENSOR),
        ("_reduce_data", patch_mod.UPSTREAM_FLAT_REDUCE_DATA),
    ],
)
def test_upstream_sources_unchanged(name, fn):
    """Fail when upstream changes what the pooled copies reproduce."""
    digest = hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()
    assert digest == PINNED_SOURCE_HASHES[name], (
        f"Upstream {name} has drifted; re-sync src/vllm_torchtpu/mm_host_copy_patch.py."
    )


# One batch under the pool threshold; one over it, whose items end mid-chunk.
@pytest.mark.parametrize("rows", [(5, 7), (3001, 11, 9000)])
def test_pooled_copies_match_upstream(rows):
    gen = torch.Generator().manual_seed(0)
    batch = [
        torch.randint(0, 256, (n, 1536), dtype=torch.uint8, generator=gen) for n in rows
    ]
    field = MultiModalFlatField(slices=[], dim=0)
    want = patch_mod.UPSTREAM_FLAT_REDUCE_DATA(field, batch, pin_memory=False)
    got = patch_mod._flat_reduce_data(field, batch, pin_memory=False)
    assert torch.equal(got, want)

    arena = ShmTensorArena(n_reader=1, slot_bytes=want.numel(), n_slots=1)
    idx = patch_mod.write_tensor(arena, want)
    assert idx == 0
    slot = arena._slot(idx, want.numel())
    try:
        view = torch.frombuffer(slot, dtype=torch.uint8).view(want.shape)
        assert torch.equal(view, want)
        del view
    finally:
        slot.release()


def test_touched_arena_takes_items_in_turn():
    """The creator's touch leaves an empty arena that takes items in turn."""
    arena = ShmTensorArena.__new__(ShmTensorArena)
    patch_mod._init_arena(arena, n_reader=1, slot_bytes=1 << 20, n_slots=2)
    t = torch.ones(1 << 20, dtype=torch.uint8)
    assert [patch_mod.write_tensor(arena, t) for _ in range(3)] == [0, 1, None]
