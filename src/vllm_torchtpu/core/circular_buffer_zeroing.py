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
"""Record newly claimed CircularBufferSpec block IDs for KV cache zeroing."""

import inspect
from functools import wraps

from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def patch_circular_buffer_block_zeroing() -> None:
    """Enable new_block_ids tracking on CircularBufferManager when zeroing is on."""
    original_init = CircularBufferManager.__init__
    if getattr(original_init, "_tpu_circular_zeroing_patch", False):
        return

    # TODO(https://github.com/vllm-project/vllm/pull/53896): Upstream
    # SingleTypeKVCacheManager.__init__ excludes CircularBufferSpec from
    # _record_new_block_ids even when needs_kv_cache_zeroing=True, so newly
    # claimed ring blocks in mixed-dtype KV pools are never emitted in
    # SchedulerOutput.new_block_ids_to_zero for worker-side zeroing. Remove
    # this patch once upstream allows CircularBufferSpec to record
    # new_block_ids when zeroing is enabled (either by default or via spec
    # opt-in), or once TPU ring kernels never read unwritten ring slots.
    orig_sig = inspect.signature(original_init)

    @wraps(original_init)
    def patched_init(
        self: CircularBufferManager,
        *args,
        **kwargs,
    ) -> None:
        original_init(self, *args, **kwargs)
        bound = orig_sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["needs_kv_cache_zeroing"]:
            self._record_new_block_ids = True

    patched_init.__dict__["_tpu_circular_zeroing_patch"] = True
    CircularBufferManager.__init__ = patched_init
    logger.info(
        "Applied TPU patch: CircularBufferManager records newly claimed blocks "
        "for KV zeroing."
    )
