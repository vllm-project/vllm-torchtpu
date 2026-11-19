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
