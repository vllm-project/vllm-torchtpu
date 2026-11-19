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

import jax.numpy as jnp
import pytest

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


def test_should_use_128_row_chunks_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)

    kwargs = dcp_token_parallel_op.attend_chunk_kwargs(
        attention_interface.sparse_mla_chunk_kwargs()
    )

    assert kwargs == {CHUNK_KWARG: 128}


@pytest.mark.parametrize("value", [16, 64, 256])
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


def test_should_pass_128_row_chunks_to_kernel_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)

    kwargs = _call_attend_capturing_kwargs(monkeypatch)

    assert kwargs[CHUNK_KWARG] == 128


def test_should_pass_env_chunk_to_kernel_when_env_set(monkeypatch):
    monkeypatch.setenv(ENV, "64")

    kwargs = _call_attend_capturing_kwargs(monkeypatch)

    assert kwargs[CHUNK_KWARG] == 64
