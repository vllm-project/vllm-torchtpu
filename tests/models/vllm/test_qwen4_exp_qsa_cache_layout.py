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
"""The planner must include native head-packing padding in its allocation."""

import pytest
import torch
from qwen4_exp_test_utils import q4exp_text_config, qsa_test_pools

from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA


@pytest.fixture(autouse=True)
def cpu_vllm_config_context():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


@pytest.mark.parametrize("dtype", ["bfloat16", "fp8_e4m3"])
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_planner_covers_native_owner_and_binding(dtype, tp, monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: tp
    )
    model = Qwen4ExpQSA(
        config=q4exp_text_config(hidden_size=16),
        tp_rank=0,
        tp_size=tp,
        max_model_len=128,
        device="cpu",
        cache_dtype=dtype,
    )
    owners, layout = qsa_test_pools(model, 3)
    model.bind_pool(*owners)
    expected_dtype = torch.float8_e4m3fn if dtype == "fp8_e4m3" else torch.bfloat16
    assert owners[0].dtype == expected_dtype
    assert owners[1].dtype == torch.bfloat16
    assert owners[0].numel() * owners[0].element_size() == 3 * layout.main_page_bytes
    assert owners[1].numel() * owners[1].element_size() == 3 * layout.index_page_bytes
    assert owners[0].untyped_storage()._cdata != owners[1].untyped_storage()._cdata
    assert len(model.cache_owners) == 2
    assert all(
        actual is expected for actual, expected in zip(model.cache_owners, owners)
    )
    backend = model.get_attn_backend()
    assert backend.get_supported_kernel_block_sizes()[0].base == 128
    with pytest.raises(TypeError):
        model.bind_pool(owners[0])
