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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.experimental.deepseek_v41.mhc import (
    mhc_post,
    mhc_post_2d,
    select_token_block,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.o_projection import (
    FP8_E4M3_MAX,
    fused_reverse_rope_wo_a_projection,
    wo_a_projection,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.rope import (
    LANE,
    largest_divisor,
    rope,
)


@pytest.fixture(autouse=True)
def _tpu_or_cpu_interpret_env(request: pytest.FixtureRequest):
    if jax.default_backend() == "tpu":
        yield
        return
    if request.node.get_closest_marker("tpu_test") is not None:
        pytest.skip("Requires TPU hardware backend")
    mesh = AbstractMesh((), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh), pltpu.force_tpu_interpret_mode():
        yield


def _ref_mhc_post(
    x: np.ndarray,
    residual: np.ndarray,
    post_mix: np.ndarray,
    comb_mix: np.ndarray,
) -> np.ndarray:
    mixed = np.einsum("...ij,...ih->...jh", comb_mix, residual.astype(np.float32))
    out = mixed + post_mix * x[..., None, :].astype(np.float32)
    return np.asarray(jnp.asarray(out, dtype=jnp.bfloat16), dtype=np.float32)


def _make_cos_sin_cache(
    max_position: int, rotary_dim: int, rng: np.random.Generator
) -> np.ndarray:
    freqs = rng.uniform(0.0, 2.0 * np.pi, size=(max_position, rotary_dim // 2)).astype(
        np.float32
    )
    return np.concatenate([np.cos(freqs), np.sin(freqs)], axis=-1).astype(np.float32)


def _ref_rope(
    x: np.ndarray,
    positions: np.ndarray,
    cos_sin_cache: np.ndarray,
    *,
    inverse: bool = False,
) -> np.ndarray:
    rotary_dim = cos_sin_cache.shape[1]
    xf = x.astype(np.float32)
    cos, sin = np.split(cos_sin_cache[positions].astype(np.float32), 2, axis=-1)
    cos = np.repeat(cos, 2, axis=-1)
    sin = np.repeat(sin, 2, axis=-1)
    if inverse:
        sin = -sin
    if xf.ndim == 3:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    x_pass, x_rot = xf[..., :-rotary_dim], xf[..., -rotary_dim:]
    rotated = np.stack([-x_rot[..., 1::2], x_rot[..., 0::2]], axis=-1).reshape(
        x_rot.shape
    )
    out_rot = x_rot * cos + rotated * sin
    return np.concatenate([x_pass, out_rot], axis=-1)


def _ref_o_projection(
    x: np.ndarray,
    positions: np.ndarray,
    cos_sin_cache: np.ndarray,
    wo_a: np.ndarray,
    wo_a_scale: np.ndarray,
    *,
    num_groups: int,
    inverse: bool,
    quantize_activations: bool,
) -> np.ndarray:
    roped_bf16 = jnp.asarray(
        _ref_rope(x, positions, cos_sin_cache, inverse=inverse),
        dtype=jnp.bfloat16,
    )
    num_tokens = roped_bf16.shape[0]
    flat_bf16 = roped_bf16.reshape(num_tokens, num_groups, -1)
    w_f32 = wo_a.reshape(flat_bf16.shape[-1], num_groups, -1)
    s_f32 = wo_a_scale.reshape(num_groups, -1)
    if quantize_activations:
        amax = jnp.max(jnp.abs(flat_bf16), axis=-1, keepdims=True)
        inv_bf16 = (FP8_E4M3_MAX / jnp.maximum(amax, jnp.bfloat16(1e-30))).astype(
            jnp.bfloat16
        )
        lhs_fp8 = np.asarray(
            (flat_bf16 * inv_bf16).astype(jnp.float8_e4m3fn), dtype=np.float32
        )
        acc = np.einsum("tgd,dgr->tgr", lhs_fp8, w_f32) * (
            1.0 / np.asarray(inv_bf16, dtype=np.float32)
        )
    else:
        acc = np.einsum("tgd,dgr->tgr", np.asarray(flat_bf16, dtype=np.float32), w_f32)
    out = (acc * s_f32[None, ...]).reshape(num_tokens, -1)
    return np.asarray(jnp.asarray(out, dtype=jnp.bfloat16), dtype=np.float32)


def test_should_recombine_mhc_post_across_unpadded_and_vmem_halved_blocks() -> None:
    rng = np.random.default_rng(101)
    x = jnp.asarray(rng.standard_normal((2, 3, 128)), dtype=jnp.bfloat16)
    res = jnp.asarray(rng.standard_normal((2, 3, 2, 128)), dtype=jnp.bfloat16)
    post = jnp.asarray(rng.uniform(0.2, 1.5, size=(2, 3, 2, 1)), dtype=jnp.float32)
    comb = jnp.asarray(rng.uniform(0.1, 0.9, size=(2, 3, 2, 2)), dtype=jnp.float32)

    actual_3d = mhc_post(x, res, post, comb, token_block_size=16, interpret=True)
    expected_3d = _ref_mhc_post(
        np.asarray(x, dtype=np.float32),
        np.asarray(res, dtype=np.float32),
        np.asarray(post, dtype=np.float32),
        np.asarray(comb, dtype=np.float32),
    )
    assert actual_3d.shape == (2, 3, 2, 128) and actual_3d.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(actual_3d, dtype=np.float32), expected_3d, rtol=2e-2, atol=2e-2
    )

    x2d = jnp.asarray(rng.standard_normal((32, 128)), dtype=jnp.bfloat16)
    res2d = jnp.asarray(rng.standard_normal((32, 256)), dtype=jnp.bfloat16)
    post2d = jnp.asarray(rng.uniform(0.2, 1.5, size=(32, 2)), dtype=jnp.float32)
    comb2d = jnp.asarray(rng.uniform(0.1, 0.9, size=(32, 2, 2)), dtype=jnp.float32)
    actual_2d = mhc_post_2d(
        x2d,
        res2d,
        post2d,
        comb2d,
        token_block_size=32,
        vmem_limit_bytes=1,
        interpret=True,
    )
    expected_2d = _ref_mhc_post(
        np.asarray(x2d, dtype=np.float32),
        np.asarray(res2d, dtype=np.float32).reshape(32, 2, 128),
        np.asarray(post2d, dtype=np.float32)[..., None],
        np.asarray(comb2d, dtype=np.float32),
    ).reshape(32, 256)
    assert actual_2d.shape == (32, 256) and actual_2d.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(actual_2d, dtype=np.float32), expected_2d, rtol=2e-2, atol=2e-2
    )
    assert select_token_block(
        32, 32, vmem_need=lambda tb: tb * 100, vmem_limit_bytes=2000
    ) == (16, 32)


def test_should_scale_default_vmem_limit_across_tpu_generations() -> None:
    def _vmem_need(tb: int) -> int:
        return (60 * 1024 * 1024 * tb) // 64

    tpu7x_mesh = AbstractMesh((), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x_mesh):
        assert select_token_block(64, 64, vmem_need=_vmem_need) == (32, 64)

    v6e_mesh = AbstractMesh((), (), abstract_device=AbstractDevice("TPU v6e", 1, "tpu"))
    with use_abstract_mesh(v6e_mesh):
        assert select_token_block(64, 64, vmem_need=_vmem_need) == (64, 64)


def test_should_reject_invalid_mhc_post_dtypes() -> None:
    x = jnp.zeros((2, 128), dtype=jnp.bfloat16)
    res = jnp.zeros((2, 2, 128), dtype=jnp.float32)
    post = jnp.zeros((2, 2, 1), dtype=jnp.float32)
    comb = jnp.zeros((2, 2, 2), dtype=jnp.float32)
    with pytest.raises(AssertionError):
        mhc_post(x, res, post, comb, interpret=True)
    with pytest.raises(AssertionError):
        mhc_post_2d(
            x,
            jnp.zeros((2, 256), dtype=jnp.bfloat16),
            post.astype(jnp.bfloat16),
            comb,
            interpret=True,
        )
    with pytest.raises(AssertionError):
        mhc_post_2d(
            x,
            jnp.zeros((2, 256), dtype=jnp.bfloat16),
            post,
            comb.astype(jnp.bfloat16),
            interpret=True,
        )


def test_should_apply_forward_and_inverse_rope_on_2d_and_3d_inputs() -> None:
    rng = np.random.default_rng(202)
    max_pos, rotary_dim = 16, 64
    cos_sin_cache = _make_cos_sin_cache(max_pos, rotary_dim, rng)
    positions = rng.integers(0, max_pos, size=(2,), dtype=np.int32)

    x2d = rng.standard_normal((2, 256)).astype(np.float32)
    actual_2d = np.asarray(
        rope(
            jnp.asarray(x2d),
            jnp.asarray(positions),
            jnp.asarray(cos_sin_cache),
            inverse=False,
            interpret=True,
        )
    )
    expected_2d = _ref_rope(x2d, positions, cos_sin_cache, inverse=False)
    np.testing.assert_array_equal(actual_2d[..., :-rotary_dim], x2d[..., :-rotary_dim])
    np.testing.assert_allclose(actual_2d, expected_2d, rtol=1e-5, atol=1e-5)

    x3d = rng.standard_normal((2, 2, LANE)).astype(np.float32)
    actual_3d = np.asarray(
        rope(
            jnp.asarray(x3d),
            jnp.asarray(positions),
            jnp.asarray(cos_sin_cache),
            inverse=True,
            interpret=True,
        )
    )
    expected_3d = _ref_rope(x3d, positions, cos_sin_cache, inverse=True)
    np.testing.assert_allclose(actual_3d, expected_3d, rtol=1e-5, atol=1e-5)
    assert largest_divisor(192, 128) == 96
    assert largest_divisor(128, 128) == 128


def test_should_reject_invalid_rope_ranks_and_shapes() -> None:
    positions = jnp.zeros((2,), dtype=jnp.int32)
    cos_sin = jnp.zeros((8, 64), dtype=jnp.float32)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        rope(jnp.zeros((2,), dtype=jnp.float32), positions, cos_sin, interpret=True)
    with pytest.raises(AssertionError):
        rope(jnp.zeros((2, 64), dtype=jnp.float32), positions, cos_sin, interpret=True)


def test_should_compute_fused_reverse_rope_wo_a_projection_across_quant_modes() -> None:
    rng = np.random.default_rng(303)
    num_tokens, num_groups, heads_per_group, head_dim, lora_rank = 4, 1, 8, 256, 128
    rotary_dim, max_pos = 64, 16
    x = jnp.asarray(
        0.1 * rng.standard_normal((num_tokens, num_groups * heads_per_group, head_dim)),
        dtype=jnp.bfloat16,
    )
    wo_a = jnp.asarray(
        rng.standard_normal((heads_per_group * head_dim, num_groups * lora_rank)),
        dtype=jnp.float8_e4m3fn,
    )
    wo_a_scale = jnp.asarray(
        rng.uniform(0.5, 1.5, size=(num_groups * lora_rank,)), dtype=jnp.float32
    )
    positions = jnp.asarray(
        rng.integers(0, max_pos, size=(num_tokens,)), dtype=jnp.int32
    )
    cos_sin_cache = jnp.asarray(
        _make_cos_sin_cache(max_pos, rotary_dim, rng), dtype=jnp.float32
    )

    out_q = fused_reverse_rope_wo_a_projection(
        x,
        positions,
        cos_sin_cache,
        wo_a,
        wo_a_scale,
        head_dim=head_dim,
        inverse=True,
        quantize_activations=True,
        interpret=True,
    )
    exp_q = _ref_o_projection(
        np.asarray(x, dtype=np.float32),
        np.asarray(positions),
        np.asarray(cos_sin_cache),
        np.asarray(wo_a, dtype=np.float32),
        np.asarray(wo_a_scale, dtype=np.float32),
        num_groups=num_groups,
        inverse=True,
        quantize_activations=True,
    )
    assert out_q.shape == (num_tokens, num_groups * lora_rank)
    assert out_q.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(out_q, dtype=np.float32), exp_q, rtol=5e-2, atol=1e-1
    )

    out_bf16 = fused_reverse_rope_wo_a_projection(
        x,
        positions,
        cos_sin_cache,
        wo_a,
        wo_a_scale,
        head_dim=head_dim,
        inverse=False,
        tile_t=num_tokens,
        tile_r=lora_rank,
        sub_t=2,
        quantize_activations=False,
        interpret=True,
    )
    exp_bf16 = _ref_o_projection(
        np.asarray(x, dtype=np.float32),
        np.asarray(positions),
        np.asarray(cos_sin_cache),
        np.asarray(wo_a, dtype=np.float32),
        np.asarray(wo_a_scale, dtype=np.float32),
        num_groups=num_groups,
        inverse=False,
        quantize_activations=False,
    )
    np.testing.assert_allclose(
        np.asarray(out_bf16, dtype=np.float32), exp_bf16, rtol=2e-2, atol=2e-2
    )


def test_should_reject_invalid_o_projection_inputs() -> None:
    cos_sin = jnp.zeros((2, 2 * LANE), dtype=jnp.float32)
    with pytest.raises(AssertionError):
        wo_a_projection(
            jnp.zeros((2, 8, LANE), dtype=jnp.bfloat16),
            jnp.zeros((8 * LANE, 128), dtype=jnp.float8_e4m3fn),
            jnp.ones((128,), dtype=jnp.float32),
            cos_sin,
            interpret=True,
        )


@pytest.mark.tpu_test
def test_tpu_should_execute_mhc_rope_and_o_projection_on_tpu7x() -> None:
    rng = np.random.default_rng(404)
    num_tokens = 48

    x_mhc = jnp.asarray(rng.standard_normal((num_tokens, 2560)), dtype=jnp.bfloat16)
    res_mhc = jnp.asarray(
        rng.standard_normal((num_tokens, 4, 2560)), dtype=jnp.bfloat16
    )
    post_mix = jnp.asarray(
        rng.uniform(0.1, 1.5, size=(num_tokens, 4, 1)), dtype=jnp.float32
    )
    comb_mix = jnp.asarray(
        rng.uniform(0.05, 0.8, size=(num_tokens, 4, 4)), dtype=jnp.float32
    )
    out_mhc = mhc_post(x_mhc, res_mhc, post_mix, comb_mix)
    exp_mhc = _ref_mhc_post(
        np.asarray(x_mhc, dtype=np.float32),
        np.asarray(res_mhc, dtype=np.float32),
        np.asarray(post_mix, dtype=np.float32),
        np.asarray(comb_mix, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(out_mhc, dtype=np.float32), exp_mhc, rtol=2e-2, atol=2e-2
    )

    cos_sin_cache = jnp.asarray(_make_cos_sin_cache(4096, 64, rng), dtype=jnp.float32)
    positions = jnp.asarray(rng.integers(0, 4096, size=(num_tokens,)), dtype=jnp.int32)
    kv = jnp.asarray(rng.standard_normal((num_tokens, 512)), dtype=jnp.bfloat16)
    exp_rope = _ref_rope(
        np.asarray(kv, dtype=np.float32),
        np.asarray(positions),
        np.asarray(cos_sin_cache),
    )
    out_rope = rope(kv, positions, cos_sin_cache)
    np.testing.assert_allclose(
        np.asarray(out_rope, dtype=jnp.float32), exp_rope, rtol=2e-2, atol=2e-2
    )

    num_groups, heads_per_group, head_dim, lora_rank = 8, 8, 512, 512
    o_in = jnp.asarray(
        0.1 * rng.standard_normal((num_tokens, num_groups * heads_per_group, head_dim)),
        dtype=jnp.bfloat16,
    )
    wo_a = jnp.asarray(
        rng.standard_normal((heads_per_group * head_dim, num_groups * lora_rank)),
        dtype=jnp.float8_e4m3fn,
    )
    wo_a_scale = jnp.asarray(
        rng.uniform(0.5, 1.5, size=(num_groups * lora_rank,)), dtype=jnp.float32
    )
    out_o = fused_reverse_rope_wo_a_projection(
        o_in,
        positions,
        cos_sin_cache,
        wo_a,
        wo_a_scale,
        head_dim=head_dim,
        inverse=True,
        quantize_activations=True,
    )
    exp_o = _ref_o_projection(
        np.asarray(o_in, dtype=np.float32),
        np.asarray(positions),
        np.asarray(cos_sin_cache),
        np.asarray(wo_a, dtype=np.float32),
        np.asarray(wo_a_scale, dtype=np.float32),
        num_groups=num_groups,
        inverse=True,
        quantize_activations=True,
    )
    assert out_o.shape == (num_tokens, num_groups * lora_rank)
    assert out_o.dtype == jnp.bfloat16
    assert bool(jnp.all(jnp.isfinite(out_o)))
    out_o_f32 = np.asarray(out_o, dtype=np.float32)
    np.testing.assert_allclose(out_o_f32, exp_o, rtol=5e-2, atol=1.0)
    assert np.linalg.norm(out_o_f32 - exp_o) / np.linalg.norm(exp_o) < 2e-2

    out_o_bf16 = fused_reverse_rope_wo_a_projection(
        o_in,
        positions,
        cos_sin_cache,
        wo_a,
        wo_a_scale,
        head_dim=head_dim,
        inverse=True,
        quantize_activations=False,
    )
    exp_o_bf16 = _ref_o_projection(
        np.asarray(o_in, dtype=np.float32),
        np.asarray(positions),
        np.asarray(cos_sin_cache),
        np.asarray(wo_a, dtype=np.float32),
        np.asarray(wo_a_scale, dtype=np.float32),
        num_groups=num_groups,
        inverse=True,
        quantize_activations=False,
    )
    np.testing.assert_allclose(
        np.asarray(out_o_bf16, dtype=np.float32), exp_o_bf16, rtol=2e-2, atol=2e-2
    )
