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
from jax.sharding import Mesh

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import prefill as k3_prefill
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import fused


def test_should_keep_prefill_moe_dense_fp8_off_by_default(monkeypatch):
    monkeypatch.delenv("K3_PREFILL_MOE_DENSE_FP8", raising=False)
    cfg = KernelConfig()

    assert cfg.prefill_moe_dense_fp8 is False
    assert cfg.key() != cfg.replace(prefill_moe_dense_fp8=True).key()


def test_should_enable_prefill_moe_dense_fp8_when_env_set(monkeypatch):
    monkeypatch.setenv("K3_PREFILL_MOE_DENSE_FP8", "1")

    assert KernelConfig().prefill_moe_dense_fp8 is True


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"prefill_fused_moe": False}, "prefill_fused_moe"),
        ({"prefill_expert_parallel": 32}, "prefill_expert_parallel"),
    ],
)
def test_should_reject_incompatible_config_when_moe_dense_fp8_enabled(overrides, match):
    with pytest.raises(ValueError, match=match):
        KernelConfig(prefill_moe_dense_fp8=True, **overrides)


def test_should_keep_router_unquantized_and_align_family_scale_order():
    assert fused.MOE_DENSE_FP8_FAMILIES == (
        "latent_down",
        "shared_gu",
        "shared_down",
        "latent_up",
    )
    assert (
        tuple(f"{name}_scale" for name in fused.MOE_DENSE_FP8_FAMILIES)
        == fused.MOE_DENSE_FP8_SCALES
    )
    assert "router" not in fused.MOE_DENSE_FP8_FAMILIES
    assert "router_bias" not in fused.MOE_DENSE_FP8_FAMILIES


@pytest.mark.parametrize("leading_shape", [(2,), (1, 2)])
def test_should_quantize_per_column_with_dead_column_isolation_and_zero_safety(
    leading_shape,
):
    rng = np.random.default_rng(7)
    cols_live, cols_total = 32, 40
    raw = rng.normal(0.0, 0.35, (*leading_shape, 64, cols_total)).astype(np.float32)
    raw[..., 0] *= 13.0
    raw[..., 1] = 0.0
    raw[..., cols_live:] = 9999.0
    weight_bf16 = jnp.asarray(raw, dtype=jnp.bfloat16)

    w_fp8, w_scale = fused.quantize_moe_dense_weight_fp8(
        weight_bf16, slice_cols=cols_live
    )

    assert w_fp8.dtype == jnp.float8_e4m3fn
    assert w_scale.dtype == jnp.float32
    assert w_fp8.shape == (*leading_shape, 64, cols_live)
    assert w_scale.shape == (*leading_shape, 1, cols_live)
    assert bool(jnp.all(jnp.isfinite(w_scale)))
    assert bool(jnp.all(w_scale > 0))

    reconstructed = w_fp8.astype(jnp.float32) * w_scale
    expected = weight_bf16[..., :cols_live].astype(jnp.float32)
    rel_l2 = float(
        jnp.linalg.norm(reconstructed - expected) / jnp.linalg.norm(expected)
    )
    assert rel_l2 < 0.035
    assert np.all(np.asarray(reconstructed[..., 1]) == 0.0)
    pos_ctrl = float(
        jnp.linalg.norm(w_fp8.astype(jnp.float32) - expected)
        / jnp.linalg.norm(expected)
    )
    assert pos_ctrl > 0.8

    w_fp8_clean, w_scale_clean = fused.quantize_moe_dense_weight_fp8(
        weight_bf16[..., :cols_live]
    )
    assert np.array_equal(np.asarray(w_scale), np.asarray(w_scale_clean))
    assert np.array_equal(
        np.asarray(w_fp8.astype(jnp.float32)),
        np.asarray(w_fp8_clean.astype(jnp.float32)),
    )


@pytest.mark.parametrize("dense_fp8", [False, True])
def test_should_quantize_and_materialize_moe_dense_families_in_prefill_load_weights(
    monkeypatch, tmp_path, dense_fp8
):
    monkeypatch.setattr(
        k3_config,
        "_ACTIVE",
        KernelConfig(prefill_moe_dense_fp8=dense_fp8),
    )
    one_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    monkeypatch.setattr(k3_prefill, "kernel_mesh", lambda axis, size: (one_mesh, 1))

    prog = k3_prefill.PrefillProgram(windows=(512,), capacity=512, slots=1)
    for scale_name in fused.MOE_DENSE_FP8_SCALES:
        assert (scale_name in prog.names["moe"]) is dense_fp8

    prog.names = {
        "moe": (
            (
                "router",
                "expert_ds",
                *fused.MOE_DENSE_FP8_FAMILIES,
                *fused.MOE_DENSE_FP8_SCALES,
            )
            if dense_fp8
            else ("router", "expert_ds", *fused.MOE_DENSE_FP8_FAMILIES)
        )
    }
    rng = np.random.default_rng(19)
    raw_weights = {
        "router": jnp.asarray(rng.normal(0.0, 0.2, (1, 1, 16, 8)), dtype=jnp.bfloat16),
        "latent_down": jnp.asarray(
            rng.normal(0.0, 0.2, (1, 1, 16, 4096)), dtype=jnp.bfloat16
        ),
        "latent_up": jnp.asarray(
            rng.normal(0.0, 0.2, (1, 1, 16, 32)), dtype=jnp.bfloat16
        ),
        "shared_gu": jnp.asarray(
            rng.normal(0.0, 0.2, (1, 1, 16, 64)), dtype=jnp.bfloat16
        ),
        "shared_down": jnp.asarray(
            rng.normal(0.0, 0.2, (1, 1, 16, 32)), dtype=jnp.bfloat16
        ),
    }

    def _fake_stream(iterator, store):
        del iterator
        for name, val in raw_weights.items():
            store("moe", name, val)
        return {"gib": 0.1, "wall": 0.1}

    registered_ops: list[str] = []
    materialized_shapes: list[tuple[tuple[int, ...], str]] = []

    def _fake_register(name, mesh, fn, in_specs, out_specs):
        del mesh, in_specs, out_specs
        registered_ops.append(name)
        return lambda x: fn(x)

    monkeypatch.setattr(k3_prefill, "stream_to_device", _fake_stream)
    monkeypatch.setattr(k3_prefill, "register_op", _fake_register)
    monkeypatch.setattr(
        k3_prefill,
        "materialize",
        lambda x: materialized_shapes.append((tuple(x.shape), str(x.dtype))),
    )
    monkeypatch.setattr(k3_prefill, "report_load_stats", lambda *a, **kw: None)

    prog.load_weights(tmp_path, shared={"expert_ds": jnp.zeros((1,), jnp.uint8)})

    assert prog.weights["moe"]["router"].dtype == jnp.bfloat16
    if dense_fp8:
        assert registered_ops == [
            "pallas::k3_prefill_moe_dense_fp8_latent_down",
            "pallas::k3_prefill_moe_dense_fp8_latent_up",
            "pallas::k3_prefill_moe_dense_fp8_shared_gu",
            "pallas::k3_prefill_moe_dense_fp8_shared_down",
        ]
        assert prog.weights["moe"]["latent_down"].shape == (1, 1, 16, 3584)
        assert prog.weights["moe"]["latent_down_scale"].shape == (1, 1, 1, 3584)
        for name in fused.MOE_DENSE_FP8_FAMILIES:
            assert prog.weights["moe"][name].dtype == jnp.float8_e4m3fn
            assert prog.weights["moe"][f"{name}_scale"].dtype == jnp.float32
        assert len(materialized_shapes) == 8 + 1
    else:
        assert registered_ops == []
        assert prog.weights["moe"]["latent_down"].dtype == jnp.bfloat16
        assert prog.weights["moe"]["latent_down"].shape == (1, 1, 16, 4096)
        for scale_name in fused.MOE_DENSE_FP8_SCALES:
            assert scale_name not in prog.weights["moe"]
