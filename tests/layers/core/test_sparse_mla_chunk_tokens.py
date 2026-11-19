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
"""TPU_SPARSE_MLA_CHUNK_TOKENS: the sparse MLA gather + attention chunk size."""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.layers.core import attention_interface

ENV = "TPU_SPARSE_MLA_CHUNK_TOKENS"


@pytest.mark.parametrize("value", [None, ""])
def test_unset_keeps_the_kernel_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)

    # No kwarg: the kernel keeps its default chunk, exactly as without the flag.
    assert attention_interface.sparse_mla_chunk_kwargs() == {}


@pytest.mark.parametrize("value", [16, 64, 256, 1024])
def test_set_overrides_the_default(monkeypatch, value):
    monkeypatch.setenv(ENV, str(value))

    assert attention_interface.sparse_mla_chunk_kwargs() == {
        "gather_and_attention_chunk_size": value
    }


@pytest.mark.parametrize("value", ["0", "-64"])
def test_non_positive_is_rejected(monkeypatch, value):
    monkeypatch.setenv(ENV, value)

    with pytest.raises(ValueError, match="must be > 0"):
        attention_interface.sparse_mla_chunk_kwargs()


@pytest.mark.parametrize("value", ["8", "100"])
def test_partial_kernel_batch_is_rejected(monkeypatch, value):
    monkeypatch.setenv(ENV, value)

    with pytest.raises(ValueError, match="multiple of"):
        attention_interface.sparse_mla_chunk_kwargs()


@pytest.mark.parametrize(("value", "want"), [(None, 128), ("64", 64)])
def test_a_caller_default_applies_only_when_unset(monkeypatch, value, want):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)

    assert attention_interface.sparse_mla_chunk_kwargs(128) == {
        "gather_and_attention_chunk_size": want
    }


@pytest.mark.parametrize(("value", "want"), [(None, 128), ("256", 256)])
def test_the_token_parallel_split_attends_in_its_own_chunk(monkeypatch, value, want):
    from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as op

    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    seen = {}

    def kernel(q, *args, **kwargs):
        seen.update(kwargs)
        return q

    monkeypatch.setattr(op.sparse_mla_kernel, "sparse_ragged_paged_attention", kernel)
    attend = op.make_attend(
        sm_scale=1.0, k_scale=None, cache_layout="", packed_rope=False, lkv_dim=4
    )
    attend(jnp.zeros((2, 1, 4)), None, None, None, jnp.int32(2))
    assert seen["gather_and_attention_chunk_size"] == want
    assert op.CHUNK_TOKENS == 128
