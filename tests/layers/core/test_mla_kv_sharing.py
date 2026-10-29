# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""CPU checks at the cache-update/gather boundary of sparse MLA."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.kv_cache_utils import KVCacheLayout
from vllm_torchtpu.layers.core import attention_interface as attention


@pytest.mark.parametrize("skip_update", [False, True])
@pytest.mark.parametrize("route", ["sparse", "masked_dense", "dcp"])
def test_sparse_mla_shared_cache_is_read_only(monkeypatch, skip_update, route):
    # Keep the wrapper's real routing and array operations; replace device
    # sharding and Pallas kernels so the cache-update contract runs on CPU.
    monkeypatch.setattr(jax, "jit", lambda fn, **kwargs: fn)

    def local_shard(fn, **kwargs):
        def run(*args):
            if route == "dcp":
                args = (*args[:6], args[6][:2], *args[7:])
            return fn(*args)

        return run

    monkeypatch.setattr(attention.shard_map, "shard_map", local_shard)
    monkeypatch.setattr(attention.streamindex_topk, "cp_rank_as_data", lambda *a: 0)
    monkeypatch.setattr(attention.envs, "TPU_MLA_SPARSE_SPLIT_Q", False)
    monkeypatch.setattr(attention.envs, "TPU_DCP_MLA_TOPK_CAP", 0)
    monkeypatch.setattr(attention.mla_dispatch, "applies", lambda *a: True)
    q = jnp.ones((2, 1, 512), jnp.bfloat16)
    q_pe = jnp.ones((2, 1, 64), jnp.bfloat16)
    old_nope = jnp.full((2, 512), 3, jnp.bfloat16)
    old_rope = jnp.full((2, 64), 5, jnp.bfloat16)
    new_nope = jnp.full_like(old_nope, 9)
    new_rope = jnp.full_like(old_rope, 11)
    update = Mock(return_value=(new_nope, new_rope))
    monkeypatch.setattr(attention, "update_sparse_mla_kv_cache", update)
    monkeypatch.setattr(attention, "update_sparse_mla_kv_cache_dcp", update)
    expected_nope = old_nope if skip_update else new_nope
    expected_rope = old_rope if skip_update else new_rope
    output = jnp.full((2, 1, 576), 7, jnp.bfloat16)

    def gather(q, nope, rope, *args, **kwargs):
        np.testing.assert_array_equal(nope, expected_nope)
        np.testing.assert_array_equal(rope, expected_rope)
        return (output, jnp.zeros((2, 1))) if route == "dcp" else output

    def masked_gather(ql, qr, nope, rope, *args, **kwargs):
        return gather(ql, nope, rope)[..., :512]

    monkeypatch.setattr(attention, "sparse_mla_ragged_paged_attention", gather)
    monkeypatch.setattr(
        attention.sparse_mla_kernel, "sparse_ragged_paged_attention", gather
    )
    monkeypatch.setattr(attention.mla_dispatch, "attention_from_parts", masked_gather)
    mesh = SimpleNamespace(axis_names=("dcp",), shape={"dcp": 2})
    spec = SimpleNamespace(layout=KVCacheLayout.TENSORCORE, num_pages=1)
    fn = (
        attention.sparse_mla_attention_dcp
        if route == "dcp"
        else attention.sparse_mla_attention
    )
    kwargs = (
        dict(dcp_size=2, interleave_size=1, dcp_axis_name="dcp")
        if route == "dcp"
        else dict(allow_masked_dense=route == "masked_dense")
    )
    result = fn(
        q,
        q_pe,
        new_nope,
        new_rope,
        old_nope,
        old_rope,
        jnp.tile(
            jnp.array([[0, -1], [0, 1]], jnp.int32), (2 if route == "dcp" else 1, 1)
        ),
        jnp.array([2], jnp.int32),
        jnp.array([0], jnp.int32),
        jnp.array([0, 2], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
        mesh,
        spec,
        spec,
        skip_kv_update=skip_update,
        **kwargs,
    )
    assert update.call_count == (0 if skip_update else 1)
    np.testing.assert_array_equal(result[0], expected_nope)
    np.testing.assert_array_equal(result[1], expected_rope)
    np.testing.assert_array_equal(result[2], output[..., :512])
