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
"""Multimodal processors' uint8 image resize on several host threads.

The HF torchvision-backend processors resize each image with `tvF.resize`. On
uint8 CPU images that is torch's Pillow-SIMD port, which runs on one thread
whatever the torch thread count: about 9 ms of a 3-image request's 18 ms HF
step, for its 3034x1754 image. The kernel is separable with a uint8
intermediate: a horizontal pass computes each output row from its input row,
then a vertical pass each output column from its input column, with weights
that depend only on that dimension's sizes. `resize` runs the passes as row
and column strips, each a `tvF.resize` of its own on the host copy pool, so
every output byte comes from the same kernel, weights and inputs as the
whole-image call. In the API server a 3-image request's resizes take 7.1 ms
instead of 10.4.

`resize` is a drop-in for `transformers.image_processing_backends.
TorchvisionBackend.resize`, whose last call it replaces.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

import numpy as np
import torch
import transformers
from torchvision.transforms.v2 import functional as tvF
from transformers import image_processing_backends as backends
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    pil_torch_interpolation_mapping,
)

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.mm_host_copy_patch import host_pool, host_threads

logger = init_logger(__name__)

# The upstream method `resize` reproduces, and its source's sha256 in
# transformers 5.20.2: any other source keeps upstream's method.
UPSTREAM_RESIZE = backends.TorchvisionBackend.resize
UPSTREAM_SHA256 = "3f84c32c17b2f8d659e41a2e0bf6f2bb6cae383d2c52a92b9cbf9d98ade07041"

# The kernel writes ~1.7 ns per output pixel (9 ms for 3040x1760), so a strip of
# this many pixels runs ~0.2 ms, ten times what handing it to a thread costs.
MIN_STRIP_PIXELS = 1 << 17


def _bounds(n: int, parts: int) -> list[tuple[int, int]]:
    """`parts` contiguous ranges covering range(n), none empty."""
    parts = min(parts, n)
    return [(n * i // parts, n * (i + 1) // parts) for i in range(parts)]


def _native_uint8(image: torch.Tensor, interpolation: Any) -> bool:
    """Whether `tvF.resize` resizes `image` with torch's uint8 kernel in two
    uint8 passes (a float path would round once, at the end), in its 3 or
    4-channel channels-last layout, which a PIL image converts to."""
    if image.dtype != torch.uint8 or image.device.type != "cpu" or image.ndim < 3:
        return False
    if interpolation == tvF.InterpolationMode.BILINEAR:
        if torch.backends.cpu.get_cpu_capability() not in ("AVX2", "AVX512"):
            return False
    elif interpolation != tvF.InterpolationMode.BICUBIC:
        return False
    *_, c, h, w = image.shape
    return c in (3, 4) and image.reshape(-1, c, h, w).is_contiguous(
        memory_format=torch.channels_last
    )


def resize_in_strips(
    image: torch.Tensor,
    size: list[int] | tuple[int, int],
    interpolation: Any,
    antialias: bool,
) -> torch.Tensor:
    """`tvF.resize(image, size, interpolation=..., antialias=...)`, with large
    uint8 resizes split into strips on the host pool."""
    h2, w2 = size
    *lead, c, h, w = image.shape
    strips = min(host_threads(), h2 * w2 // MIN_STRIP_PIXELS)
    if strips < 2 or (h, w) == (h2, w2) or not _native_uint8(image, interpolation):
        return tvF.resize(
            image, list(size), interpolation=interpolation, antialias=antialias
        )

    def one(src: np.ndarray, out_size: list[int]) -> np.ndarray:
        t = torch.from_numpy(src).permute(2, 0, 1)
        out = tvF.resize(t, out_size, interpolation=interpolation, antialias=antialias)
        return out.permute(1, 2, 0).numpy()

    src = image.reshape(-1, c, h, w).permute(0, 2, 3, 1).numpy()  # (N, H, W, C) view
    n = src.shape[0]
    out = np.empty((n, h2, w2, c), np.uint8)
    mid = src
    if w2 != w:
        mid = np.empty((n, h, w2, c), np.uint8) if h2 != h else out

        def horizontal(task: tuple[int, int, int]) -> None:
            i, a, b = task
            np.copyto(mid[i, a:b], one(src[i, a:b], [b - a, w2]))

        tasks = [(i, a, b) for i in range(n) for a, b in _bounds(h, strips)]
        list(host_pool().map(horizontal, tasks))
    if h2 != h:

        def vertical(task: tuple[int, int, int]) -> None:
            i, a, b = task
            col = np.ascontiguousarray(mid[i, :, a:b])
            np.copyto(out[i, :, a:b], one(col, [h2, b - a]))

        tasks = [(i, a, b) for i in range(n) for a, b in _bounds(w2, strips)]
        list(host_pool().map(vertical, tasks))
    return torch.from_numpy(out).permute(0, 3, 1, 2).reshape(*lead, c, h2, w2)


def resize(
    self: backends.TorchvisionBackend,
    image: torch.Tensor,
    size: backends.SizeDict,
    resample: PILImageResampling | tvF.InterpolationMode | int | None = None,
    antialias: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Resize an image using Torchvision."""
    # Convert PIL resample to torchvision interpolation if needed
    if resample is not None:
        if isinstance(resample, (PILImageResampling, int)):
            interpolation = pil_torch_interpolation_mapping[
                PILImageResampling(resample)
            ]
        else:
            interpolation = resample
    else:
        interpolation = tvF.InterpolationMode.BILINEAR
    if interpolation == tvF.InterpolationMode.LANCZOS and (
        not backends.is_torchvision_greater_or_equal("0.27")
        or image.device.type != "cpu"
    ):
        backends.logger.warning_once(
            "LANCZOS resample requires torchvision >= 0.27 and processing on CPU; "
            "it is not supported on CUDA or for torchvision < 0.27. Falling back "
            "to BICUBIC which approximates LANCZOS. To match the original model "
            "exactly, upgrade torchvision and move the tensors to CPU before "
            "resizing, or use a PIL backend image processor instead."
        )
        interpolation = tvF.InterpolationMode.BICUBIC

    if size.shortest_edge and size.longest_edge:
        new_size = backends.get_size_with_aspect_ratio(
            image.size()[-2:],
            size.shortest_edge,
            size.longest_edge,
        )
    elif size.shortest_edge:
        new_size = backends.get_resize_output_image_size(
            # Annotated as an ndarray, but reads only the tensor's shape.
            # pyrefly: ignore[bad-argument-type]
            image,
            size=size.shortest_edge,
            default_to_square=False,
            input_data_format=ChannelDimension.FIRST,
        )
    elif size.max_height and size.max_width:
        new_size = backends.get_image_size_for_max_height_width(
            (image.shape[-2], image.shape[-1]), size.max_height, size.max_width
        )
    elif size.height and size.width:
        new_size = (size.height, size.width)
    else:
        raise ValueError(
            "Size must contain 'height' and 'width' keys, or 'max_height' and "
            f"'max_width', or 'shortest_edge' key. Got {size}."
        )

    # Workaround for torch.compile issue with uint8 on AMD GPUs
    if backends.is_torchdynamo_compiling() and backends.is_rocm_platform():
        return self._compile_friendly_resize(image, new_size, interpolation, antialias)
    return resize_in_strips(image, new_size, interpolation, antialias)


def patch_mm_resize_strips() -> None:
    """Resize the HF processors' uint8 images in strips on the host pool."""
    source = inspect.getsource(UPSTREAM_RESIZE)
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != UPSTREAM_SHA256:
        logger.warning(
            "Skipped TPU patch: transformers %s's TorchvisionBackend.resize differs "
            "from the one mm_resize_patch reproduces; images resize on one thread.",
            transformers.__version__,
        )
        return
    backends.TorchvisionBackend.resize = resize
