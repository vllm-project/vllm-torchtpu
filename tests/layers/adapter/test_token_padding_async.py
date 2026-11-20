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
"""TokenPaddingState.update with and without VLLM_TPU_ASYNC_HOST_OVERLAP."""

import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import token_padding


def _expected(max_len, valid, padded):
    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[valid:padded] = True
    return mask


@pytest.fixture
def overlap(monkeypatch):
    monkeypatch.setattr(envs, "USE_MOE_TP_DECODE_KERNEL", False, raising=False)

    def set_flag(on):
        monkeypatch.setattr(envs, "VLLM_TPU_ASYNC_HOST_OVERLAP", on, raising=False)

    return set_flag


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("valid, padded", [(0, 0), (3, 8), (5, 16), (16, 16)])
def test_update_writes_the_padding_mask(overlap, on, valid, padded):
    overlap(on)
    state = token_padding.TokenPaddingState.create(16, torch.device("cpu"))
    state.update(valid, padded)
    torch.testing.assert_close(state.get_local_padding_mask(16), _expected(16, valid, padded))
    torch.testing.assert_close(state.get_local_padding_mask(8), _expected(16, valid, padded)[:8])


def test_flag_off_updates_in_place(overlap):
    overlap(False)
    state = token_padding.TokenPaddingState.create(16, torch.device("cpu"))
    buf = state.local_padding_mask
    state.update(3, 8)
    state.update(5, 16)
    assert state.local_padding_mask is buf
    torch.testing.assert_close(buf, _expected(16, 5, 16))
    assert not state._async_mask_cache


def test_flag_on_reuses_one_upload_per_bucket(overlap):
    overlap(True)
    state = token_padding.TokenPaddingState.create(16, torch.device("cpu"))
    state.update(3, 8)
    first = state.local_padding_mask
    state.update(5, 16)
    second = state.local_padding_mask
    assert second is not first
    torch.testing.assert_close(first, _expected(16, 3, 8))
    torch.testing.assert_close(second, _expected(16, 5, 16))
    # The same bucket again returns the cached device tensor; the CPU source
    # stays referenced so an in-flight non_blocking copy remains valid.
    state.update(3, 8)
    assert state.local_padding_mask is first
    assert set(state._async_mask_cache) == {(16, 3, 8), (16, 5, 16)}
    for dev, host in state._async_mask_cache.values():
        torch.testing.assert_close(dev, host)


def test_tp_decode_kernel_keeps_the_synchronous_path(overlap, monkeypatch):
    overlap(True)
    monkeypatch.setattr(envs, "USE_MOE_TP_DECODE_KERNEL", True, raising=False)
    state = token_padding.TokenPaddingState.create(16, torch.device("cpu"))
    buf = state.local_padding_mask
    state.update(3, 8)
    assert state.local_padding_mask is buf
    assert not state._async_mask_cache
