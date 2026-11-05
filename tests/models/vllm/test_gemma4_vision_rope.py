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
"""Parity of single-pass Gemma-4 vision RoPE against the HF reference.

Runs on CPU by default. Set GEMMA4_ROPE_TEST_DEVICE=tpu to run on TPU, and run
this file directly (``python3 <file>``) to also print a micro-benchmark.
"""

import os
import time

import pytest
import torch

modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

from vllm_torchtpu.models.vllm import gemma4_mm_patch  # noqa: E402

# Gemma-4 vision tower: batch of 8 images, 2520 patches, 16 heads, head_dim 72.
_SHAPE = (8, 2520, 16, 72)


def _reference():
    fn = modeling_gemma4.apply_multidimensional_rope
    if getattr(fn, "_tpu_rope_patched", False):
        fn = gemma4_mm_patch._orig_apply_multidimensional_rope
    assert fn is not None
    return fn


def _device() -> torch.device:
    return torch.device(os.environ.get("GEMMA4_ROPE_TEST_DEVICE", "cpu"))


def _inputs(shape, device, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    batch, length, _, head_dim = shape
    x = torch.randn(shape, generator=gen).to(dtype)
    cos = torch.randn((batch, length, head_dim), generator=gen).to(dtype)
    sin = torch.randn((batch, length, head_dim), generator=gen).to(dtype)
    position_ids = torch.zeros((batch, length, 2), dtype=torch.long)
    return [t.to(device) for t in (x, cos, sin, position_ids)]


def test_rotate_half_matrix_matches_rotate_half():
    x = torch.randn(3, 72)
    rot = gemma4_mm_patch._rotate_half_matrix(72, 36, x.device, x.dtype)
    expected = torch.cat(
        [modeling_gemma4.rotate_half(p) for p in torch.split(x, 36, -1)], -1
    )
    assert torch.equal(x @ rot, expected)


@pytest.mark.parametrize("shape", [(2, 16, 4, 72), _SHAPE])
def test_single_pass_rope_matches_hf(shape):
    device = _device()
    x, cos, sin, position_ids = _inputs(shape, device)
    expected = _reference()(x, cos, sin, position_ids)
    actual = gemma4_mm_patch.tpu_apply_multidimensional_rope(x, cos, sin, position_ids)
    max_diff = (actual.float() - expected.float()).abs().max().item()
    print(f"[{device}] shape={shape} max_abs_diff={max_diff:.3e}")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_patch_installs_and_falls_back():
    gemma4_mm_patch.patch_gemma4_vision_rope()
    assert getattr(
        modeling_gemma4.apply_multidimensional_rope, "_tpu_rope_patched", False
    )
    # 70 channels over 2 dims does not split evenly: must use the HF path.
    x, cos, sin, position_ids = _inputs((1, 4, 2, 70), torch.device("cpu"))
    expected = _reference()(x, cos, sin, position_ids)
    actual = gemma4_mm_patch.tpu_apply_multidimensional_rope(x, cos, sin, position_ids)
    assert torch.equal(actual, expected)


def _bench(fn, args, iters: int = 50) -> float:
    for _ in range(3):
        fn(*args)
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


if __name__ == "__main__":
    device = _device()
    args = _inputs(_SHAPE, device)
    ref = _reference()
    fast = gemma4_mm_patch.tpu_apply_multidimensional_rope

    eager_diff = (fast(*args).float() - ref(*args).float()).abs().max().item()
    print(f"[{device}] eager max_abs_diff={eager_diff:.3e}")

    if device.type == "tpu":
        ref_c = torch.compile(ref, backend="tpu", fullgraph=True, dynamic=False)
        fast_c = torch.compile(fast, backend="tpu", fullgraph=True, dynamic=False)
        out_ref = ref_c(*args)
        out_fast = fast_c(*args)
        torch.accelerator.synchronize()
        diff = (out_fast.float() - out_ref.float()).abs().max().item()
        print(f"[tpu] compiled max_abs_diff={diff:.3e}")
        ref_us = _bench(ref_c, args)
        fast_us = _bench(fast_c, args)
        print(
            f"[tpu] compiled per-call: hf={ref_us:.1f} us, "
            f"single_pass={fast_us:.1f} us, speedup={ref_us / fast_us:.2f}x"
        )
