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
"""Applies `_patch_vllm_same_step_prefix_hits` for one test."""

import inspect

import pytest
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputSchedulerConnector,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager

from vllm_torchtpu import _patch_vllm_same_step_prefix_hits

_PATCHED_METHODS = (
    (BlockPool, "_insert_block_hash"),
    (BlockPool, "get_cached_block"),
    (KVCacheManager, "new_step_starts"),
    (AuxOutputSchedulerConnector, "_gpu_cached"),
)


def apply_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Puts back upstream's methods and patches them once, and `monkeypatch`
    restores the methods the process had after the test.

    The plugin may already have patched them in this process, and the patch
    does nothing a second time.
    """
    for owner, name in _PATCHED_METHODS:
        monkeypatch.setattr(owner, name, inspect.unwrap(getattr(owner, name)))
    _patch_vllm_same_step_prefix_hits()
