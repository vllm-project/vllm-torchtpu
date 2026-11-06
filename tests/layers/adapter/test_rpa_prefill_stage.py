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
"""Which RPA kernels get a fixed-length PREFILL stage (chunk_prefill_size)."""

import pytest

from vllm_torchtpu.layers.adapter import attention


def _impl(cls, *, head_size=128, chunk_prefill_size=16, kernel_entry=None):
    impl = cls.__new__(cls)
    impl.head_size = head_size
    impl.chunk_prefill_size = chunk_prefill_size
    if kernel_entry is not None:
        impl._kernel_entry = kernel_entry
    return impl


@pytest.mark.parametrize(
    "kernel_entry",
    [None, attention._pallas_rpa_kernel_local],
    ids=["v3_sharded", "v3_local"],
)
def test_rpa_v3_entries_get_the_prefill_stage(kernel_entry):
    impl = _impl(attention.PallasAttentionBackendImpl, kernel_entry=kernel_entry)
    assert impl._kernel_chunk_prefill_size(use_pcp_streaming=False) == 16


@pytest.mark.parametrize(
    "impl",
    [
        # Batched RPA has no PREFILL stage (it folds PREFILL into MIXED).
        _impl(attention.PallasBatchedRPAAttentionBackendImpl),
        # hd64 reroutes to a kernel without the stage.
        _impl(attention.PallasAttentionBackendImpl, head_size=64),
        # Not requested.
        _impl(attention.PallasAttentionBackendImpl, chunk_prefill_size=None),
    ],
    ids=["batched", "hd64", "unset"],
)
def test_other_kernels_get_no_prefill_stage(impl):
    assert impl._kernel_chunk_prefill_size(use_pcp_streaming=False) is None


def test_pcp_streaming_gets_no_prefill_stage():
    impl = _impl(attention.PallasAttentionBackendImpl)
    assert impl._kernel_chunk_prefill_size(use_pcp_streaming=True) is None


def test_replicated_batched_draft_on_local_v3_gets_the_stage():
    # The runner rebinds a replicated batched-RPA draft to the local RPA v3
    # kernel, which does have the stage.
    impl = _impl(
        attention.PallasBatchedRPAAttentionBackendImpl,
        kernel_entry=attention._pallas_rpa_kernel_local,
    )
    assert impl._kernel_chunk_prefill_size(use_pcp_streaming=False) == 16
