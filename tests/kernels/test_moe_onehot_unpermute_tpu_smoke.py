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
"""Single-device target-shape smoke for blockwise output one-hot combine."""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import gmm_v2, moe_onehot_unpermute


def _require_tpu() -> jax.Device:
    devices = jax.devices("tpu")
    if not devices:
        pytest.fail("Expected at least one TPU device.")
    return devices[0]


def _target_group_sizes() -> np.ndarray:
    sizes = np.asarray(
        [1, 14, 2, 16, 17, 3] + [3] * 24 + [4, 3] + [4] * 32,
        dtype=np.int32,
    )
    assert sizes.shape == (64, )
    assert int(sizes.sum()) == 260
    return sizes


@pytest.mark.nightly
def test_qwen35_target_shape_single_device_smoke() -> None:
    device = _require_tpu()
    route_capacity = 2560
    valid_count_value = 260
    num_tokens = 256
    size_k = 1024
    size_n = 4096
    num_experts = 64
    tile_info = gmm_v2.TileSizes(
        tile_m=512,
        tile_k=1024,
        tile_n=4096,
        bucket_base=128,
    )

    with jax.default_device(device):
        row_factor = (
            (jnp.arange(route_capacity, dtype=jnp.float32) % 17) - 8) / 32
        k_factor = ((jnp.arange(size_k, dtype=jnp.float32) % 13) - 6) / 16
        lhs = (row_factor[:, None] * k_factor[None, :]).astype(jnp.bfloat16)

        expert_factor = (
            (jnp.arange(num_experts, dtype=jnp.float32) % 7) + 1) / 32
        n_factor = ((jnp.arange(size_n, dtype=jnp.float32) % 11) - 5) / 16
        rhs = jnp.broadcast_to(
            expert_factor[:, None, None] * n_factor[None, None, :],
            (num_experts, size_k, size_n),
        ).astype(jnp.float8_e4m3fn)
        rhs_scale = jnp.ones(
            (num_experts, 1, 1, size_n),
            dtype=jnp.float32,
        )

        group_sizes = jnp.asarray(_target_group_sizes(), dtype=jnp.int32)
        token_ids = np.full((route_capacity, ), -1, dtype=np.int32)
        token_ids[:valid_count_value] = (
            np.arange(valid_count_value, dtype=np.int32) % num_tokens)
        token_ids = jnp.asarray(token_ids)
        route_weights = np.full(
            (route_capacity, ),
            np.nan,
            dtype=np.float32,
        )
        route_weights[:valid_count_value] = (
            (np.arange(valid_count_value, dtype=np.float32) % 7) + 1) / 8
        route_weights = jnp.asarray(route_weights, dtype=jnp.bfloat16)
        valid_count = jnp.asarray([valid_count_value], dtype=jnp.int32)

        baseline_routes = gmm_v2.gmm_v2(
            lhs,
            rhs,
            group_sizes,
            rhs_scale=rhs_scale,
            tile_info=tile_info,
            zero_initialize=False,
        )
        candidate = moe_onehot_unpermute.blockwise_onehot_unpermute(
            baseline_routes,
            token_ids,
            route_weights,
            valid_count,
            num_tokens=num_tokens,
        )
        zero_candidate = moe_onehot_unpermute.blockwise_onehot_unpermute(
            jnp.full_like(baseline_routes, jnp.nan),
            token_ids,
            route_weights,
            jnp.asarray([0], dtype=jnp.int32),
            num_tokens=num_tokens,
        )
        expected = jnp.zeros(
            (num_tokens, size_n),
            dtype=jnp.float32,
        ).at[token_ids[:valid_count_value]].add(
            baseline_routes[:valid_count_value].astype(jnp.float32) *
            route_weights[:valid_count_value, None].astype(
                jnp.float32)).astype(jnp.bfloat16)
        candidate.block_until_ready()
        zero_candidate.block_until_ready()
        expected.block_until_ready()

    candidate_np = np.asarray(candidate, dtype=np.float32)
    zero_np = np.asarray(zero_candidate, dtype=np.float32)
    expected_np = np.asarray(expected, dtype=np.float32)
    diff = candidate_np - expected_np
    result = {
        "git_commit":
        os.environ.get("ONEHOT_UNPERMUTE_SMOKE_GIT_COMMIT"),
        "resolved_vllm_torchtpu_source":
        os.environ.get("ONEHOT_UNPERMUTE_SMOKE_SOURCE"),
        "device":
        str(device),
        "route_capacity":
        route_capacity,
        "valid_count":
        valid_count_value,
        "num_tokens":
        num_tokens,
        "num_experts":
        num_experts,
        "size_k":
        size_k,
        "size_n":
        size_n,
        "tail_route_weights_are_nan":
        True,
        "rmse":
        float(np.sqrt(np.mean(np.square(diff)))),
        "max_abs_diff":
        float(np.max(np.abs(diff))),
        "zero_route_max_abs":
        float(np.max(np.abs(zero_np))),
    }
    print("MOE_BLOCKWISE_ONEHOT_UNPERMUTE_SMOKE " +
          json.dumps(result, sort_keys=True))
    configured_result = os.environ.get("ONEHOT_UNPERMUTE_SMOKE_RESULT")
    if configured_result:
        result_path = Path(configured_result)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    assert candidate.devices() == {device}
    assert zero_candidate.devices() == {device}
    assert np.isfinite(candidate_np).all()
    assert np.isfinite(zero_np).all()
    assert result["rmse"] == 0.0
    assert result["max_abs_diff"] == 0.0
    assert result["zero_route_max_abs"] == 0.0
