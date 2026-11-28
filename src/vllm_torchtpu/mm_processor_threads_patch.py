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
"""Torch threads for the multimodal processor in vLLM's API server.

vLLM's renderer runs the HF multimodal processor (image resize and patchify,
video sampling) under `set_default_torch_num_threads()`, which takes
OMP_NUM_THREADS or 1. Raising OMP_NUM_THREADS would also raise the engine's
and workers' threads, which vLLM keeps at 1 against spin-wait contention, so
the renderer's default call gets its own count here. Those ops compute every
output element on its own, so the pixels do not depend on the thread count.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

from vllm.renderers import base
from vllm.utils import torch_utils

# The renderer code this patch relies on; the tests pin their sources.
UPSTREAM_SET_DEFAULT_TORCH_NUM_THREADS = torch_utils.set_default_torch_num_threads
UPSTREAM_PROCESS_MULTIMODAL = base.BaseRenderer._process_multimodal

# Knee of a 1/4/8/16 sweep: a 3-image request's HF image processor takes 41.3 ms
# at 1 thread, 24.0 at 8 and no less at 16.
MM_PROCESSOR_THREADS = 8


def mm_processor_threads() -> int:
    """The processor's threads: the knee, within this process's CPUs."""
    return min(MM_PROCESSOR_THREADS, len(os.sched_getaffinity(0)))


@contextlib.contextmanager
def _set_default_torch_num_threads(num_threads: int | None = None) -> Iterator[None]:
    if num_threads is None:
        num_threads = mm_processor_threads()
    with UPSTREAM_SET_DEFAULT_TORCH_NUM_THREADS(num_threads):
        yield


def patch_mm_processor_threads() -> None:
    """Give the renderer's default thread count to the multimodal processor.

    Only the renderer's binding changes; explicit counts (its 1-thread
    warmups) pass through.
    """
    base.set_default_torch_num_threads = _set_default_torch_num_threads
