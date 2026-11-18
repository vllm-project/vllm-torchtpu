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
import copy

import pytest
import torch
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image_21.decode_graph import (
    QwenImage21DecodeGraphManager,
)
from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21Transformer2DModel,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MAX_REL_L2 = 3e-2
MAX_ABS = 8e-2


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "_check_static_eligibility requires param.device.type == 'cuda', "
        "so manager.eligible() is False on TPU"
    ),
)
def test_diffusion_cuda_graph_decode() -> None:
    """Diffusion CUDA Graph Decode

    - TPU path: build QwenImage21DecodeGraphManager on TPU bf16 and run 2 steps.
    - Setup check: assert eligible() is True, len(entries) == 1, captures == 1.
    - Outcome check: compare TPU bf16 decode outputs against CPU fp32 eager decode.
    """
    od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(),
        enable_cuda_graph_decode=True,
        enforce_eager=False,
    )
    torch.manual_seed(0)
    with set_current_diffusion_config(od_config):
        cpu_model = (
            QwenImage21Transformer2DModel(
                od_config=od_config,
                in_channels=16,
                out_channels=16,
                num_layers=1,
                attention_head_dim=128,
                num_attention_heads=1,
                context_in_dim=16,
            )
            .float()
            .eval()
        )
    tpu_model = copy.deepcopy(cpu_model).to(device="tpu", dtype=torch.bfloat16).eval()

    manager = QwenImage21DecodeGraphManager(tpu_model)
    assert manager.eligible() is True, (
        "expected QwenImage21DecodeGraphManager.eligible() is True on TPU"
    )

    batch, prefix_len, target_len = 1, 4, 4
    head_dim = 128
    img_shapes = [(1, 2, 2)]
    hidden_cpu = torch.randn((batch, target_len, 16), dtype=torch.bfloat16)
    timestep_cpu = torch.tensor([500.0], dtype=torch.bfloat16)
    img_mask_cpu = torch.ones((batch, prefix_len + target_len), dtype=torch.bool)
    img_mask_cpu[:, :prefix_len] = False
    target_freqs_cpu = torch.ones((target_len, head_dim // 2), dtype=torch.complex64)
    prefix_k_cpu = torch.randn((batch, prefix_len, 1, head_dim), dtype=torch.bfloat16)
    prefix_v_cpu = torch.randn((batch, prefix_len, 1, head_dim), dtype=torch.bfloat16)

    cpu_kv_cache = [
        {"cond": {"key": prefix_k_cpu.float(), "value": prefix_v_cpu.float()}}
    ]
    tpu = torch.device("tpu")
    tpu_kv_cache = [
        {"cond": {"key": prefix_k_cpu.to(tpu), "value": prefix_v_cpu.to(tpu)}}
    ]

    with torch.no_grad():
        ref = cpu_model._forward_decode(
            hidden_states=hidden_cpu.float(),
            timestep=timestep_cpu.float(),
            kv_cache=cpu_kv_cache,
            cache_branch="cond",
            img_shapes=img_shapes,
            img_mask=img_mask_cpu,
            encoder_hidden_states_mask=None,
        )

    manager.register_prefill(
        kv_cache=tpu_kv_cache,
        cache_branch="cond",
        prefix_len=prefix_len,
        img_shapes=img_shapes,
        target_freqs=target_freqs_cpu.to(tpu),
        joint_key_valid=None,
        dtype=torch.bfloat16,
    )
    assert len(manager.entries) == 1, (
        f"expected 1 graph entry, got {len(manager.entries)}"
    )

    with torch.no_grad():
        out1 = manager.try_decode(
            hidden_states=hidden_cpu.to(tpu),
            timestep=timestep_cpu.to(tpu),
            kv_cache=tpu_kv_cache,
            cache_branch="cond",
            img_shapes=img_shapes,
            img_mask=img_mask_cpu.to(tpu),
            encoder_hidden_states_mask=None,
        )
        assert out1 is not None, (
            "expected first try_decode to capture and return output"
        )
        assert manager.captures == 1, (
            f"expected 1 capture after first step, got {manager.captures}"
        )

        out2 = manager.try_decode(
            hidden_states=hidden_cpu.to(tpu),
            timestep=timestep_cpu.to(tpu),
            kv_cache=tpu_kv_cache,
            cache_branch="cond",
            img_shapes=img_shapes,
            img_mask=img_mask_cpu.to(tpu),
            encoder_hidden_states_mask=None,
        )
        assert out2 is not None, (
            "expected second try_decode to replay and return output"
        )
        assert manager.captures == 1, (
            f"expected captures to stay 1 on replay, got {manager.captures}"
        )

    assert_parity(
        out1,
        ref,
        name="diffusion_cuda_graph_decode_capture",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
    assert_parity(
        out2,
        ref,
        name="diffusion_cuda_graph_decode_replay",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
