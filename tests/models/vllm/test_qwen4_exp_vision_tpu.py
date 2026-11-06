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
"""One actual checkpoint vision block through the TPU tower and merger."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel
from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer


@pytest.mark.tpu_test
@torch.inference_mode()
def test_checkpoint_one_block_tower_matches_hf_fp32(dist_ctx, record_property):
    checkpoint = Path("/tmpfs/Qwen3.8-Flash-Next-FP8")
    if not checkpoint.is_dir():
        pytest.skip("Qwen4 vision checkpoint is not mounted")
    model_config = ModelConfig(
        model=str(checkpoint),
        dtype="bfloat16",
        max_model_len=8192,
        hf_overrides={"text_config": {"decoder_sparse_step": 1, "mlp_only_layers": []}},
    )
    config = VllmConfig(
        model_config=model_config, device_config=DeviceConfig(device="tpu")
    )
    vision_config = copy.deepcopy(AutoConfig.from_pretrained(checkpoint).vision_config)
    vision_config.depth = 1
    vision_config._attn_implementation = "eager"
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with set_current_vllm_config(config), torch.device("cpu"):
            native = Qwen3_VisionTransformer(
                vision_config, norm_eps=1e-6, prefix="visual"
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    reference = Qwen4ExpVisionModel(vision_config).float().eval()
    with safe_open(
        checkpoint / "model-00001-of-00131.safetensors", framework="pt", device="cpu"
    ) as f:
        weights = {}
        names = f.keys()
        for name in names:
            if not name.startswith("model.visual."):
                continue
            local = name.removeprefix("model.visual.")
            if local.startswith("blocks.") and not local.startswith("blocks.0."):
                continue
            weights[local] = f.get_tensor(name)
    reference.load_state_dict(weights)
    native.load_weights(iter(weights.items()))
    native = native.to("tpu").eval()
    rng = np.random.default_rng(732)
    for grids in [[[1, 16, 16]], [[1, 24, 32]], [[1, 16, 16], [1, 24, 32]]]:
        count = sum(t * h * w for t, h, w in grids)
        pixels = torch.tensor(rng.normal(0, 0.3, (count, 1536)), dtype=torch.bfloat16)
        grid = torch.tensor(grids, dtype=torch.int64)
        expected = reference(pixels.float(), grid_thw=grid).pooler_output
        with set_current_vllm_config(config):
            metadata = native.prepare_encoder_metadata(
                grids, device=torch.device("tpu")
            )
            compiled = torch.compile(
                lambda x, grids=grids, metadata=metadata: native(
                    x, grids, encoder_metadata=metadata
                ),
                backend="tpu",
                fullgraph=True,
                dynamic=False,
            )
            actual = compiled(pixels.to("tpu")).cpu().float()
        error = (
            torch.linalg.vector_norm(actual - expected)
            / torch.linalg.vector_norm(expected)
        ).item()
        assert torch.isfinite(actual).all() and error < 0.01, (grids, error)
        assert list(actual.shape) == [count // 4, 2560]
        record_property(
            "tower_case",
            {
                "grids": grids,
                "relative_L2": error,
                "threshold": 0.01,
                "output_shape": list(actual.shape),
                "checkpoint_tensors": len(weights),
            },
        )
