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
"""Axial rotary selection and compressed-group position ownership."""

import copy

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

from vllm_torchtpu.kernels.qwen4_exp.qsa import _indexer_updates, main_rope
from vllm_torchtpu.kernels.qwen4_exp.rope import select_rope_rows
from vllm_torchtpu.models.vllm.qwen4_exp.qsa import make_rope_cache
from vllm_torchtpu.models.vllm.qwen4_exp.qsa_geometry import rope_axis_assignment


@pytest.fixture(autouse=True)
def vllm_context():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


def geometry(sections=(11, 11, 10)):
    cfg = Qwen4ExpTextConfig(
        head_dim=256,
        partial_rotary_factor=0.25,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": list(sections),
            "mrope_interleaved": True,
        },
    )
    cfg = copy.deepcopy(cfg)
    cfg.rope_parameters["mrope_section"] = list(sections)
    axes = rope_axis_assignment(cfg.rope_parameters, 64)
    trig = make_rope_cache(64, rotary_dim=64, theta=cfg.rope_parameters["rope_theta"])
    return cfg, axes, jnp.asarray(trig.float().numpy(), jnp.bfloat16)


@pytest.mark.parametrize("sections", [(11, 11, 10), (2, 29, 1), (31, 1, 0), (0, 0, 32)])
def test_axial_rows_match_independent_transformers_rotary(sections):
    cfg, axes, trig = geometry(sections)
    positions = np.array([[2, 7, 19, 23], [3, 17, 4, 11], [9, 5, 13, 29]], np.int32)
    hf = Qwen4ExpTextRotaryEmbedding(cfg)
    cos, sin = hf(
        torch.zeros((1, 4, 256), dtype=torch.bfloat16), torch.tensor(positions[:, None])
    )
    expected = torch.cat((cos[0, :, :32], sin[0, :, :32]), -1).float().numpy()
    actual = select_rope_rows(jnp.asarray(positions), trig, 64, axes)
    np.testing.assert_array_equal(np.asarray(actual[:, :64], np.float32), expected)


def test_equal_axes_retain_original_scalar_rope():
    _, axes, trig = geometry()
    rng = np.random.default_rng(41)
    x = jnp.asarray(rng.normal(size=(9, 2, 128)), jnp.bfloat16)
    positions = jnp.arange(9, dtype=jnp.int32)
    scalar = main_rope(x, positions, trig, 64)
    axial = main_rope(x, jnp.broadcast_to(positions, (3, 9)), trig, 64, axes)
    np.testing.assert_array_equal(scalar, axial)


def test_compressed_first_token_three_axes_survive_chunk_boundaries():
    _, axes, trig = geometry()
    rng = np.random.default_rng(61)
    raw = jnp.asarray(rng.normal(0, 0.3, (12, 128)), jnp.bfloat16)
    norm = jnp.asarray(rng.normal(0, 0.03, (128,)), jnp.bfloat16)
    t = np.arange(12)
    positions = jnp.asarray(
        np.stack([3 + t // 2, 1 + 7 * t % 23, 2 + 11 * t % 29]), jnp.int32
    )
    ring = jnp.zeros((1, 8, 128), jnp.bfloat16)
    tags = jnp.zeros((1, 8, 4), jnp.int32)

    def update(a, p, ring, tags, context):
        return _indexer_updates(
            a,
            p,
            ring,
            tags,
            norm,
            trig,
            jnp.array([0, len(a)], jnp.int32),
            jnp.array([context], jnp.int32),
            jnp.array([0], jnp.int32),
            4,
            1e-6,
            64,
            axes,
        )

    full = update(raw, positions, ring, tags, 0)
    completed = []
    offset = 0
    for size in [3, 2, 7]:
        result = update(
            raw[offset : offset + size],
            positions[:, offset : offset + size],
            ring,
            tags,
            offset,
        )
        values, complete, _, logical, ring, tags = result
        completed.extend(np.asarray(values)[np.asarray(complete)])
        np.testing.assert_array_equal(logical, np.arange(offset, offset + size))
        offset += size
    np.testing.assert_array_equal(completed, np.asarray(full[0])[np.asarray(full[1])])
    np.testing.assert_array_equal(ring, full[-2])
    np.testing.assert_array_equal(tags, full[-1])
    np.testing.assert_array_equal(tags[0, :4, :3], positions[:, 8:12].T)
    np.testing.assert_array_equal(tags[0, :4, 3], 0)
    with pytest.raises(ValueError, match="state ABI"):
        update(raw, positions, ring, jnp.zeros((1, 8, 2), jnp.int32), 0)
