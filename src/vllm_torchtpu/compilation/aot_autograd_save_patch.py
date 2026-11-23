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
"""Filter unread synthetic `torchtpu_` entries from `AOTAutogradCache.save`.

With `TORCHINDUCTOR_AUTOGRAD_CACHE=0` (set in `sitecustomize.py`),
`torch_tpu==0.2.0` injects a synthetic `torchtpu_<hash>` `cache_info` so
`TpuBackend` can capture the bundled entry for `.serialize()`. PyTorch then
runs `AOTAutogradCache.save` for a key nothing ever reads back: an extra
`SerializeExecutable()` pass, a pickled copy retained in `CacheArtifactManager`
until warmup ends, an unread `aotautograd/` file on disk, and an
`entry.pre_save()` shallow copy of non-split `TorchTpuCompiledExecutable`
wrappers.
"""

from typing import Any


def patch_torchtpu_skip_unread_aot_autograd_save() -> None:
    """Skip `AOTAutogradCache.save` for synthetic `torchtpu_` cache keys."""
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

    orig_save = AOTAutogradCache.save
    marker = "_tpu_skip_torchtpu_save_patch"
    if orig_save.__dict__.get(marker, False):
        return

    def patched_save(key: str, *args: Any, **kwargs: Any) -> None:
        if key.startswith("torchtpu_"):
            return
        orig_save(key, *args, **kwargs)

    patched_save.__dict__[marker] = True
    AOTAutogradCache.save = staticmethod(patched_save)
