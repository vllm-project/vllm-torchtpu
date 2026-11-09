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
"""Unit tests for the cache execution extension (`pool.config`)."""

from __future__ import annotations

import pytest

from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.pool import (
    CacheExec,
    PoolBackend,
    PoolLayout,
    cache_exec,
    resolved_max_pages_per_seq,
    with_cache,
)


def test_cache_exec_pool_gather_default_is_split():
    assert ExecSpec().cache is None
    # A `CacheExec` equal to its defaults is the same as no cache extension.
    assert ExecSpec(cache=CacheExec()) == ExecSpec()
    assert hash(ExecSpec(cache=CacheExec())) == hash(ExecSpec())
    assert with_cache(ExecSpec(), paged_kv=False).cache is None
    assert ExecSpec(cache=CacheExec(paged_kv=True)) != ExecSpec()
    assert cache_exec(ExecSpec()).pool_gather is False
    assert not CacheExec(paged_kv=True).GPR
    assert not CacheExec(paged_kv=True).FUSED
    assert CacheExec(paged_kv=True, fused_pool_io=True).FUSED
    for val in pool.POOL_GATHER_VALUES:
        assert CacheExec(pool_gather=val).pool_gather == val
    for bad_gather in ("sometimes", 0, 1, 1.0):
        with pytest.raises(ValueError, match="pool_gather must be one of"):
            CacheExec(pool_gather=bad_gather)  # type: ignore[arg-type]
    for mode in pool.ACT_MODE_VALUES:
        assert CacheExec(act_mode=mode).act_mode == mode
    with pytest.raises(ValueError, match="act_mode must be one of"):
        CacheExec(act_mode="al")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CacheExec"):
        cache_exec(ExecSpec(cache="invalid"))
    ex = with_cache(ExecSpec(B=24, T=4), paged_kv=True, pool_gather=True)
    assert cache_exec(ex).GPR and not cache_exec(ex).FUSED


def test_resolved_max_pages_per_seq_and_pool_backend_properties():
    ex = ExecSpec(ctx=512)
    assert resolved_max_pages_per_seq(ex) == 4
    assert resolved_max_pages_per_seq(ExecSpec(ctx=513)) == 5
    assert resolved_max_pages_per_seq(with_cache(ex, max_pages_per_seq=16)) == 16

    for slots in (3, 1, 0):
        lay = PoolLayout(
            num_pools=2,
            slots_per_pool=slots,
            kh=2 if slots else 0,
            vh=4 if slots else 0,
        )
        backend = PoolBackend(
            layout=lay,
            kernel_names=("gather_gdn_states", "scatter_pool_states"),
        )
        assert backend.layout is lay and backend.num_pools == 2
        assert backend.has_recurrent_pool == (slots > 0)
        assert backend.recurrent_slots_per_pool == max(1, slots)
        assert backend.ckpt_pages_per_slot == 5
