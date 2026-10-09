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

"""Tests for tiling.choose_tiling; runs on CPU against TpuInfo presets."""

from __future__ import annotations

from unittest import mock

import jax
from absl.testing import absltest, parameterized
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as rpa_configs
from vllm_torchtpu.kernels.experimental.fused_qkv import models
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import rpa, tiling

CHIPS = (pltpu.ChipVersion.TPU_7X,)
KV_LAYOUTS = (
    rpa_configs.KVLayout.SEQ_ALONG_LANE,
    rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
)


def choose(model, num_tokens, chip=pltpu.ChipVersion.TPU_7X, **kwargs):
    """choose_tiling on chip, with abstract weights laid out by prepare_weights."""
    tpu_info = pltpu.get_tpu_info_for_chip(chip, 1)
    with mock.patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info):
        s = model.config
        weights = jax.eval_shape(
            lambda: rpa.prepare_weights(s, models.make_inputs(model, 128)[2])
        )
        return (
            tpu_info,
            weights,
            rpa.choose_tiling(s, num_tokens, weights, **kwargs),
        )


class ChooseTilingTest(parameterized.TestCase):
    @parameterized.product(
        name=[m.name for m in models.MODELS],
        num_tokens=(128, 1024, 16384),
        kv_layout=KV_LAYOUTS,
        chip=CHIPS,
    )
    def test_valid(self, name, num_tokens, kv_layout, chip):
        model = models.BY_NAME[name]
        tpu_info, weights, t = choose(model, num_tokens, chip, kv_layout=kv_layout)
        self.assertEqual(num_tokens % t.tile_size, 0)
        self.assertEqual(t.tile_size % tpu_info.num_lanes, 0)
        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info
        ):
            bytes_per_row = rpa.bytes_per_row(
                model.config, kv_layout, weights.w.shape[0]
            )
            need = sum(tiling.vmem_bytes(bytes_per_row, t.tile_size).values())
        self.assertLessEqual(need + tiling.VMEM_MODEL_MARGIN_BYTES, t.vmem_limit_bytes)
        self.assertEqual(t.tile_size % t.matmul_rows, 0)
        self.assertEqual(t.matmul_rows % t.prep_rows, 0)
        self.assertEqual(t.norm_cols % tpu_info.num_lanes, 0)

    @parameterized.parameters(
        (pltpu.ChipVersion.TPU_7X, 32, 512),
    )
    def test_hardware_tiles(self, chip, prep_rows, norm_cols):
        _, _, t = choose(models.QWEN3_5_TP8, 1024, chip)
        self.assertEqual(t.prep_rows, prep_rows)
        self.assertEqual(t.norm_cols, norm_cols)

    # The "picked" column of the sweep table in tiling.py.
    @parameterized.parameters(
        ("qwen3_coder_tp8", 1024, 256),
        ("qwen3_coder_tp8", 16384, 256),
        ("qwen3_5_tp8", 1024, 256),
        ("qwen3_5_tp8", 16384, 512),
        ("qwen3_5_tp2", 1024, 128),
        ("qwen3_5_tp2", 16384, 128),
        ("gemma4_sliding_tp4", 1024, 256),
        ("gemma4_sliding_tp4", 16384, 512),
        ("gemma4_full_tp4", 1024, 256),
        ("gemma4_full_tp4", 16384, 512),
        ("gpt_oss_tp4", 1024, 256),
        ("gpt_oss_tp4", 16384, 512),
    )
    def test_matches_sweep(self, name, num_tokens, picked):
        _, _, t = choose(models.BY_NAME[name], num_tokens)
        self.assertEqual(t.tile_size, picked)

    def test_tile_size_override(self):
        _, _, t = choose(models.GPT_OSS_TP4, 16384, tile_size=128)
        self.assertEqual(t.tile_size, 128)
        self.assertEqual(t.matmul_rows, 128)


if __name__ == "__main__":
    absltest.main()
