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
"""Chunk size of the token-parallel split's sparse MLA gather + attention."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op
from vllm_torchtpu.layers.core import attention_interface

ENV = "TPU_SPARSE_MLA_CHUNK_TOKENS"
CHUNK_KWARG = "gather_and_attention_chunk_size"


def _make_attend():
    return dcp_token_parallel_op.make_attend(
        sm_scale=0.1,
        k_scale=1.0,
        cache_layout="native_sc",
        packed_rope=False,
        lkv_dim=512,
    )


def _call_attend_capturing_kwargs(monkeypatch) -> dict:
    captured = {}

    def fake_kernel(q, *args, **kwargs):
        captured.update(kwargs)
        return jnp.zeros((q.shape[0], q.shape[1], 512), jnp.bfloat16)

    monkeypatch.setattr(sparse_mla_kernel, "sparse_ragged_paged_attention", fake_kernel)
    q = jnp.zeros((1024, 64, 576), jnp.bfloat16)
    _make_attend()(q, None, None, None, jnp.int32(1024))
    return captured


def test_should_use_256_row_chunks_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)

    kwargs = dcp_token_parallel_op.attend_chunk_kwargs(
        attention_interface.sparse_mla_chunk_kwargs()
    )

    assert kwargs == {CHUNK_KWARG: 256}


@pytest.mark.parametrize("value", [16, 64, 128, 256])
def test_should_keep_env_chunk_when_env_set(monkeypatch, value):
    monkeypatch.setenv(ENV, str(value))

    kwargs = dcp_token_parallel_op.attend_chunk_kwargs(
        attention_interface.sparse_mla_chunk_kwargs()
    )

    assert kwargs == {CHUNK_KWARG: value}


def test_should_be_a_whole_number_of_kernel_batches():
    batch = sparse_mla_kernel.DEFAULT_ATTENTION_KERNEL_BATCH_SIZE

    assert dcp_token_parallel_op.ATTEND_CHUNK_TOKENS % batch == 0


def test_should_differ_from_kernel_default():
    assert (
        dcp_token_parallel_op.ATTEND_CHUNK_TOKENS
        != sparse_mla_kernel.DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE
    )


def test_should_pass_256_row_chunks_to_kernel_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)

    kwargs = _call_attend_capturing_kwargs(monkeypatch)

    assert kwargs[CHUNK_KWARG] == 256


def test_should_pass_env_chunk_to_kernel_when_env_set(monkeypatch):
    monkeypatch.setenv(ENV, "64")

    kwargs = _call_attend_capturing_kwargs(monkeypatch)

    assert kwargs[CHUNK_KWARG] == 64


def test_should_filter_mixed_live_widths_and_forward_batch_override(monkeypatch):
    captured: dict = {}

    def fake_kernel(q, *_args, **kwargs):
        captured.update(kwargs)
        return (
            jnp.zeros((q.shape[0], q.shape[1], 512), jnp.bfloat16),
            jnp.zeros((q.shape[0], q.shape[1]), jnp.float32),
        )

    monkeypatch.setattr(sparse_mla_kernel, "sparse_ragged_paged_attention", fake_kernel)
    monkeypatch.setenv("TPU_DCP_MLA_TOPK_CAP", "256")
    devs = jax.devices()
    if len(devs) < 2:
        pytest.skip(f"needs >=2 JAX devices, got {len(devs)}")
    dcp = 2
    mesh = Mesh(np.array(devs[:dcp]), ("dcp",))
    shd = NamedSharding(mesh, P("dcp"))
    nope_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.NOPE,
        kv_cache_utils.KVCacheLayout.SPARSECORE,
        4,
        16,
        512,
        4,
    )
    rope_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.ROPE,
        kv_cache_utils.KVCacheLayout.SPARSECORE,
        4,
        16,
        64,
        4,
        packed_rope=True,
    )

    attention_interface.sparse_mla_attention_dcp(
        jnp.zeros((64, 4, 512), jnp.bfloat16),
        jnp.zeros((64, 4, 64), jnp.bfloat16),
        jnp.zeros((64, 512), jnp.float8_e4m3fn),
        jnp.zeros((64, 64), jnp.float8_e4m3fn),
        jax.device_put(
            jnp.zeros(
                (dcp * nope_spec.shape[0], *nope_spec.shape[1:]), nope_spec.jax_dtype
            ),
            shd,
        ),
        jax.device_put(
            jnp.zeros(
                (dcp * rope_spec.shape[0], *rope_spec.shape[1:]), rope_spec.jax_dtype
            ),
            shd,
        ),
        jax.device_put(jnp.full((dcp * 64, 512), -1, jnp.int32), shd),
        jnp.array([64], jnp.int32),
        jnp.zeros((1, 4), jnp.int32),
        jnp.array([0, 64], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
        mesh,
        nope_spec,
        rope_spec,
        dcp_size=dcp,
        interleave_size=4,
        dcp_axis_name="dcp",
        skip_kv_write=True,
        merge_pieces=False,
        mixed_live_widths_override=(128, 256, 384),
        attention_kernel_batch_size=8,
    )

    assert captured["mixed_live_widths"] == (128,)
    assert captured["attention_kernel_batch_size"] == 8
