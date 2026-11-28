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
"""The multimodal processor gets its own threads, with unchanged pixels."""

import hashlib
import inspect

import numpy as np
import pytest
import torch
import vllm
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)
from vllm.renderers import base

from vllm_torchtpu import mm_processor_threads_patch as patch_mod

PINNED_VLLM_VERSION = "0.33.0"
# The sha256 of each pinned upstream source.
PINNED_SOURCES = {
    "bbeb6d6bbba030f9633550ce20fa0055645aa1bc80a9ac603d331a499d889d4c": (
        patch_mod.UPSTREAM_SET_DEFAULT_TORCH_NUM_THREADS
    ),
    "0073f61ba0fc1665b2fa0d7b664b509f58f24daff6302efaa888738b2b87f70a": (
        patch_mod.UPSTREAM_PROCESS_MULTIMODAL
    ),
}


@pytest.mark.skipif(
    vllm.__version__ != PINNED_VLLM_VERSION,
    reason=f"the patch relies on vLLM {PINNED_VLLM_VERSION}'s renderer",
)
@pytest.mark.parametrize(("digest", "fn"), PINNED_SOURCES.items())
def test_upstream_sources_unchanged(digest, fn):
    """Fail when upstream changes how the renderer sets the processor's threads."""
    source = inspect.getsource(fn)
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == digest, (
        f"Upstream {fn.__qualname__} has drifted; re-check "
        "src/vllm_torchtpu/mm_processor_threads_patch.py."
    )


def test_patch_sets_the_renderer_default(monkeypatch):
    """The default call gets the processor's count; an explicit one passes."""
    name = "set_default_torch_num_threads"
    monkeypatch.setattr(base, name, getattr(base, name))
    monkeypatch.setattr(patch_mod, "mm_processor_threads", lambda: 3)
    patch_mod.patch_mm_processor_threads()
    before = torch.get_num_threads()
    with base.set_default_torch_num_threads():
        assert torch.get_num_threads() == 3
    with base.set_default_torch_num_threads(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == before


_PROCESSOR_KWARGS = dict(
    patch_size=16,
    merge_size=2,
    temporal_patch_size=2,
    image_mean=[0.5] * 3,
    image_std=[0.5] * 3,
)


@pytest.mark.parametrize("do_normalize", [True, False])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_pixels_do_not_depend_on_threads(modality, do_normalize):
    """The image and video processors give the same pixels at 1 thread and at
    the counts the patch can pick."""
    rng = np.random.default_rng(0)
    if modality == "image":
        proc = Qwen2VLImageProcessor(
            size={"shortest_edge": 65536, "longest_edge": 16777216},
            **_PROCESSOR_KWARGS,
        )
        shapes = [(1754, 3034, 3), (341, 512, 3)]  # the first resized up
        inputs = dict(images=[rng.integers(0, 256, s, dtype=np.uint8) for s in shapes])
        key = "pixel_values"
    else:
        proc = Qwen3VLVideoProcessor(
            size={"shortest_edge": 4096, "longest_edge": 25165824},
            **_PROCESSOR_KWARGS,
        )
        video = rng.integers(0, 256, (8, 240, 416, 3), dtype=np.uint8)
        inputs = dict(videos=[video], do_sample_frames=False)
        key = "pixel_values_videos"
    kwargs = dict(do_normalize=do_normalize, do_rescale=do_normalize)

    outs = []
    for threads in (1, 4, 8, 16):
        with patch_mod.UPSTREAM_SET_DEFAULT_TORCH_NUM_THREADS(threads):
            outs.append(proc(**inputs, return_tensors="pt", **kwargs)[key])
    for out in outs[1:]:
        assert torch.equal(out, outs[0])
