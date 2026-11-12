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
"""CPU unit tests for MoE FP8->FP4 W4A8 requantization and routing_moe bridge."""

from __future__ import annotations

import functools
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.routing_moe.common import (
    out_format as out_format_lib,
)
from vllm_torchtpu.kernels.experimental.routing_moe.common import spec as spec_lib
from vllm_torchtpu.kernels.experimental.routing_moe.gmm import plan as gmm_plan_lib
from vllm_torchtpu.kernels.experimental.routing_moe.gmm import spec as gmm_spec_lib
from vllm_torchtpu.layers.adapter import fused_moe
from vllm_torchtpu.layers.adapter import routing_moe as routing_moe_mod
from vllm_torchtpu.layers.adapter.fused_moe_ep import register_fused_scale_buffers
from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    _get_moe_quant_config,
    _is_mtp_moe_layer,
    _process_fp8_moe_weights,
    _quantize_and_format_single_moe_weight,
)
from vllm_torchtpu.layers.adapter.routing_moe import (
    _MAX_PACKED_GROUPS,
    _get_routing_moe_op,
    _select_gmm_plan,
    _sort_routed_indices,
    routing_moe_jax_fn,
)


def _torch_to_jax_cpu(t: Any) -> Any:
    if not isinstance(t, torch.Tensor):
        return t
    if hasattr(torch, "float4_e2m1fn_x2") and t.dtype == torch.float4_e2m1fn_x2:
        u8 = t.view(torch.uint8).detach().cpu().numpy()
        fp4_pairs = jax.lax.bitcast_convert_type(jnp.asarray(u8), jnp.float4_e2m1fn)
        return fp4_pairs.reshape(*u8.shape[:-1], u8.shape[-1] * 2)
    if t.dtype == torch.bfloat16:
        u16 = t.view(torch.uint16).detach().cpu().numpy()
        return jax.lax.bitcast_convert_type(jnp.asarray(u16), jnp.bfloat16)
    if t.dtype == torch.float8_e4m3fn:
        u8 = t.view(torch.uint8).detach().cpu().numpy()
        return jax.lax.bitcast_convert_type(jnp.asarray(u8), jnp.float8_e4m3fn)
    return jnp.asarray(t.detach().cpu().numpy())


def _jax_to_torch_cpu(a: Any) -> Any:
    if not isinstance(a, jax.Array):
        return a
    if a.dtype == jnp.float4_e2m1fn:
        paired = a.reshape(*a.shape[:-1], a.shape[-1] // 2, 2)
        packed = np.array(jax.lax.bitcast_convert_type(paired, jnp.uint8), copy=True)
        return torch.from_numpy(packed).view(torch.float4_e2m1fn_x2)
    if a.dtype == jnp.bfloat16:
        u16 = np.array(jax.lax.bitcast_convert_type(a, jnp.uint16), copy=True)
        return torch.from_numpy(u16).view(torch.bfloat16)
    if a.dtype == jnp.float8_e4m3fn:
        u8 = np.array(jax.lax.bitcast_convert_type(a, jnp.uint8), copy=True)
        return torch.from_numpy(u8).view(torch.float8_e4m3fn)
    return torch.from_numpy(np.array(a, copy=True))


def _cpu_jax_op(_name: str, fn):
    class _Op:
        def __call__(self, *args):
            j_args = [_torch_to_jax_cpu(x) for x in args]
            out = fn(*j_args)
            if isinstance(out, tuple):
                return tuple(_jax_to_torch_cpu(v) for v in out)
            return _jax_to_torch_cpu(out)

        def register_fake(self, _fake_fn):
            return _fake_fn

    return _Op()


def test_mtp_moe_prefix_detection() -> None:
    assert _is_mtp_moe_layer("mtp.layers.48.mlp.experts")
    assert _is_mtp_moe_layer("model.mtp.layers.0.mlp.experts")
    assert not _is_mtp_moe_layer("model.layers.0.mlp.experts")
    assert not _is_mtp_moe_layer("model.layers.47.mlp.experts")


def test_moe_quant_config_fp4_and_force_fp8(monkeypatch) -> None:
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 unavailable")
    monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "128")
    import vllm_torchtpu.envs as envs

    assert envs.environment_variables["MOE_REQUANTIZE_WEIGHT_DTYPE"]() == "fp4"
    name, dt, blk = _get_moe_quant_config("test", force_fp8=False)
    assert name == "fp4"
    assert dt == torch.float4_e2m1fn_x2
    assert blk == 128

    f_name, f_dt, f_blk = _get_moe_quant_config("test", force_fp8=True)
    assert f_name == "float8_e4m3fn"
    assert f_dt == torch.float8_e4m3fn
    assert f_blk is None


def test_process_fp8_moe_weights_to_fp4_kmajor_and_mtp_exclusion(monkeypatch) -> None:
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("torch.float4_e2m1fn_x2 unavailable")
    monkeypatch.setattr(fused_moe.pallas, "jax_op", _cpu_jax_op)
    monkeypatch.setattr(fused_moe, "_quantize_native_fp4_kmajor_ops", {})
    monkeypatch.setattr(fp8_mod, "_concat_fp4_ops", {})
    monkeypatch.setattr(fp8_mod, "fresh_buffer", lambda x: x)
    monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "fp4")
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "128")

    # Build synthetic 2-expert [E=2, N_w13=256, K=256] and [E=2, N_w2=256, K=128]
    # block-[128, 128] FP8 checkpoint layer.
    torch.manual_seed(7)
    w13_ref = torch.randn(2, 256, 256, dtype=torch.float32) * 0.5
    w2_ref = torch.randn(2, 256, 128, dtype=torch.float32) * 0.5

    def _to_block_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e, n, k = w.shape
        wb = w.view(e, n // 128, 128, k // 128, 128)
        amax = wb.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-6)
        scale = (amax / 448.0).squeeze(4).squeeze(2).contiguous()
        q = (wb / amax * 448.0).reshape(e, n, k).to(torch.float8_e4m3fn)
        return q, scale

    moe_ns = types.SimpleNamespace(
        intermediate_size_per_partition=128,
        intermediate_size_per_partition_unpadded=128,
    )
    layer = torch.nn.Module()
    layer.moe_config = moe_ns  # type: ignore[assignment]
    w13_q, w13_s = _to_block_fp8(w13_ref)
    w2_q, w2_s = _to_block_fp8(w2_ref)
    layer.w13_weight = torch.nn.Parameter(w13_q, requires_grad=False)
    layer.w13_weight_scale_inv = torch.nn.Parameter(w13_s, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_q, requires_grad=False)
    layer.w2_weight_scale_inv = torch.nn.Parameter(w2_s, requires_grad=False)

    # 1. Routed backbone (force_fp8=False): converts to K-major FP4 + block-128 scale
    w13_fp4, w13_fp4_s, w2_fp4, w2_fp4_s, dt_name, blk = _process_fp8_moe_weights(
        layer,
        weight_block_size=(128, 128),
        activation="silu",
        force_fp8=False,  # type: ignore[arg-type]
    )
    assert dt_name == "fp4" and blk == 128
    assert w13_fp4.dtype == torch.float4_e2m1fn_x2
    assert w2_fp4.dtype == torch.float4_e2m1fn_x2
    # K-major packed x2: [E, K, N // 2] and scales [E, K // 128, 1, N]
    assert tuple(w13_fp4.shape) == (2, 256, 128)
    assert tuple(w13_fp4_s.shape) == (2, 2, 1, 256)
    assert tuple(w2_fp4.shape) == (2, 128, 128)
    assert tuple(w2_fp4_s.shape) == (2, 1, 1, 256)

    # Reconstruct dequantized K-major weight and verify closeness to w2_ref^T
    w2_e2m1 = np.asarray(_torch_to_jax_cpu(w2_fp4).astype(jnp.float32))
    w2_sc_np = w2_fp4_s.float().numpy()  # [2, 1, 1, 256]
    w2_recon = w2_e2m1 * np.repeat(w2_sc_np[:, :, 0, :], 128, axis=1)
    w2_want = w2_ref.transpose(1, 2).numpy()
    rel_err = np.linalg.norm(w2_recon - w2_want) / np.linalg.norm(w2_want)
    assert rel_err < 0.22

    # 2. MTP draft layer exclusion (force_fp8=True): stays in per-channel FP8
    layer_mtp = torch.nn.Module()
    layer_mtp.moe_config = moe_ns  # type: ignore[assignment]
    layer_mtp.w13_weight = torch.nn.Parameter(w13_q.clone(), requires_grad=False)
    layer_mtp.w13_weight_scale_inv = torch.nn.Parameter(
        w13_s.clone(), requires_grad=False
    )
    layer_mtp.w2_weight = torch.nn.Parameter(w2_q.clone(), requires_grad=False)
    layer_mtp.w2_weight_scale_inv = torch.nn.Parameter(
        w2_s.clone(), requires_grad=False
    )
    w13_m, w13_ms, w2_m, w2_ms, m_dt, m_blk = _process_fp8_moe_weights(
        layer_mtp,
        weight_block_size=(128, 128),
        activation="silu",
        force_fp8=True,  # type: ignore[arg-type]
    )
    assert m_dt == "float8_e4m3fn" and m_blk is None
    assert w13_m.dtype == torch.float8_e4m3fn and w2_m.dtype == torch.float8_e4m3fn
    assert tuple(w13_ms.shape) == (2, 1, 1, 256)

    # 3. Unaligned contracting dimension raises ValueError
    with pytest.raises(ValueError, match="not divisible by block_size=128"):
        _quantize_and_format_single_moe_weight(
            torch.randn(2, 256, 192, dtype=torch.float32),
            name="w13",
            quant_dtype=torch.float4_e2m1fn_x2,
            block_size=128,
        )


def test_register_fused_scale_buffers_supports_scale_inv() -> None:
    layer = torch.nn.Module()
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.ones((4, 20, 1, 1280), dtype=torch.float32), requires_grad=False
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.ones((4, 5, 1, 2560), dtype=torch.float32) * 2.0, requires_grad=False
    )
    register_fused_scale_buffers(layer)
    assert hasattr(layer, "_tpu_fused_w13_scale")
    assert hasattr(layer, "_tpu_fused_w2_scale")
    assert tuple(layer._tpu_fused_w13_scale.shape) == (4, 20, 1280)
    assert tuple(layer._tpu_fused_w2_scale.shape) == (4, 5, 2560)
    assert tuple(layer.w13_weight_scale_inv.shape) == (4, 20, 1, 1280)
    assert tuple(layer.w2_weight_scale_inv.shape) == (4, 5, 1, 2560)


def test_packed_single_key_sort_and_overflow_fallback() -> None:
    key = jax.random.key(123)
    flat_ids = jax.random.randint(key, (320,), 0, 513, dtype=jnp.int32)
    row_group, argsort_idx = _sort_routed_indices(flat_ids, 513)
    ref_group, ref_argsort = jax.lax.sort_key_val(
        flat_ids, jnp.arange(320, dtype=jnp.int32), is_stable=True
    )
    np.testing.assert_array_equal(np.asarray(row_group), np.asarray(ref_group))
    np.testing.assert_array_equal(np.asarray(argsort_idx), np.asarray(ref_argsort))

    # Fallback path when num_groups >= _MAX_PACKED_GROUPS
    fb_group, fb_argsort = _sort_routed_indices(flat_ids, _MAX_PACKED_GROUPS)
    np.testing.assert_array_equal(np.asarray(fb_group), np.asarray(ref_group))
    np.testing.assert_array_equal(np.asarray(fb_argsort), np.asarray(ref_argsort))


def test_select_gmm_plan_decode_and_prefill_tiers() -> None:
    p16 = _select_gmm_plan(16, acc_dtype_name="float32", x_quant_block_size=128)
    assert p16.bucket_base == 8
    assert p16.tile_m == 16
    assert p16.num_w_buffers == 3
    assert p16.acc_dtype == jnp.float32
    assert p16.x_quant_block_size == 128

    p128 = _select_gmm_plan(128, acc_dtype_name="float32", x_quant_block_size=None)
    assert p128.bucket_base == 16
    assert p128.tile_m == 32
    assert p128.num_w_buffers == 3

    p4096 = _select_gmm_plan(4096, acc_dtype_name="bfloat16", x_quant_block_size=None)
    assert p4096.bucket_base is None
    assert p4096.tile_m is None
    assert p4096.acc_dtype == jnp.bfloat16


@pytest.mark.parametrize("chip", [pltpu.ChipVersion.TPU_7X, pltpu.ChipVersion.TPU_8I])
def test_resolved_gmm_plan_admits_640_partial_chunk_width(chip) -> None:
    tpu_info = pltpu.get_tpu_info_for_chip(chip, 1)
    x_spec = spec_lib.ActivationSpec(
        row_size=32, col_size=2560, dtype=jnp.bfloat16, quant=None
    )
    w1_spec = spec_lib.WeightSpec(
        group_size=16,
        in_size=2560,
        out_size=1280,
        dtype=jnp.float4_e2m1fn,
        quant=spec_lib.SubchannelQuantSpec(block_size=128),
        has_bias=False,
        fused_gate_up=True,
    )
    w2_spec = spec_lib.WeightSpec(
        group_size=16,
        in_size=640,
        out_size=2560,
        dtype=jnp.float4_e2m1fn,
        quant=spec_lib.SubchannelQuantSpec(block_size=128),
        has_bias=False,
        fused_gate_up=False,
    )
    r_spec = spec_lib.RoutingSpec(num_indices=320, remote_axis=None)
    gmm_spec = gmm_spec_lib.RoutingGmmSpec(
        x=x_spec,
        w=w1_spec,
        w2=w2_spec,
        gather=r_spec,
        scatter=r_spec,
        num_global_groups=64,
        fuse_act="silu",
        size_ep=1,
        size_row_group=320,
    )
    out_fmt = out_format_lib.ResolvedActivationFormat.create(
        None,
        default_dtype=jnp.bfloat16,
        default_quant=out_format_lib.ResolvedLaneStrided(block_col_size=128),
    )
    resolved = gmm_plan_lib.ResolvedGmmPlan.create(
        _select_gmm_plan(32, acc_dtype_name="float32", x_quant_block_size=128),
        spec=gmm_spec,
        out_format=out_fmt,
        tpu_info=tpu_info,
    )
    assert resolved.fuse_up_and_down
    assert resolved.tile_k2 == 640


def test_routing_moe_jax_fn_ep_and_zero_padded_row_skipping(monkeypatch) -> None:
    seen_group_sizes: list[np.ndarray] = []

    def _ref_routing_gmm(
        *,
        x,
        w,
        w2,
        group_sizes,
        gather,
        scatter,
        fuse_act,
        group_offset,
        row_group,
        custom_plan=None,
    ):
        del custom_plan, fuse_act
        seen_group_sizes.append(np.asarray(group_sizes))
        x_2d = (
            x.data[:, :, 0, :]
            .reshape(x.spec.row_size, x.spec.col_size)
            .astype(jnp.float32)
        )
        g_idx = gather.indices
        s_idx = scatter.indices
        start = int(group_offset[0])
        local_e = w.weight.shape[0]
        num_routed = g_idx.shape[0]
        out_rows = np.zeros((num_routed, x.spec.col_size), dtype=np.float32)
        for pos in range(num_routed):
            exp_id = int(row_group[pos])
            if start <= exp_id < start + local_e:
                tok = int(g_idx[pos])
                dst = int(s_idx[pos])
                le = exp_id - start
                h = np.asarray(x_2d[tok])
                gate_up = h @ np.asarray(w.weight[le], dtype=np.float32)
                inter = gate_up.shape[-1] // 2
                gate, up = gate_up[:inter], gate_up[inter:]
                act = (gate / (1.0 + np.exp(-gate))) * up
                out_rows[dst] = act @ np.asarray(w2.weight[le], dtype=np.float32)
        return jnp.asarray(out_rows, dtype=jnp.float32)

    def _ref_output_reduce_unpack(
        gmm_res,
        effective_weights,
        effective_ids=None,
        group_offset=None,
        local_num_experts=None,
        *,
        out_dtype=jnp.bfloat16,
    ):
        num_tokens, topk = effective_weights.shape
        reshaped = gmm_res.reshape(num_tokens, topk, -1)
        w = effective_weights.astype(jnp.float32)
        if effective_ids is not None and group_offset is not None:
            start = group_offset[0]
            in_shard = (effective_ids >= start) & (
                effective_ids < start + local_num_experts
            )
            w = jnp.where(in_shard, w, 0.0)
        return jnp.sum(reshaped * w[:, :, None], axis=1).astype(out_dtype)

    monkeypatch.setattr(routing_moe_mod.gmm, "routing_gmm", _ref_routing_gmm)
    monkeypatch.setattr(
        routing_moe_mod.output_reduce, "output_reduce_unpack", _ref_output_reduce_unpack
    )

    rng = np.random.default_rng(42)
    x_np = rng.normal(0.0, 0.5, size=(6, 128)).astype(np.float32)
    # Rows 4 and 5 are zero-padded runner rows, even though topk_weights are non-zero.
    x_np[4:] = 0.0
    x = jnp.asarray(x_np, dtype=jnp.bfloat16)
    w1_all = jnp.asarray(rng.normal(0.0, 0.2, size=(4, 128, 256)), dtype=jnp.bfloat16)
    w2_all = jnp.asarray(rng.normal(0.0, 0.2, size=(4, 128, 128)), dtype=jnp.bfloat16)
    topk_weights = jnp.asarray(
        [[0.7, 0.3], [0.5, 0.5], [0.8, 0.2], [0.6, 0.4], [0.9, 0.1], [0.4, 0.6]],
        dtype=jnp.float32,
    )
    topk_ids = jnp.asarray(
        [[0, 2], [1, 3], [2, 3], [0, 1], [0, 2], [1, 3]], dtype=jnp.int32
    )

    # Single-shard execution (all 4 experts local)
    out_full = routing_moe_jax_fn(
        x,
        w1_all,
        w2_all,
        None,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        jnp.array([0], dtype=jnp.int32),
        topk=2,
        global_num_experts=4,
        activation="silu",
        use_ep=False,
        acc_dtype_name="float32",
        x_quant_block_size=None,
    )
    # Zeroed padding rows 4 & 5 must skip expert routing and output 0
    assert int(seen_group_sizes[-1].sum()) == 4 * 2
    np.testing.assert_array_equal(np.asarray(out_full[4:], dtype=np.float32), 0.0)

    # Sum of two EP=2 shards (experts [0, 1] + [2, 3]) matches single-shard output
    out_ep0 = routing_moe_jax_fn(
        x,
        w1_all[:2],
        w2_all[:2],
        None,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        jnp.array([0], dtype=jnp.int32),
        topk=2,
        global_num_experts=4,
        activation="silu",
        use_ep=True,
        acc_dtype_name="float32",
        x_quant_block_size=None,
    )
    out_ep1 = routing_moe_jax_fn(
        x,
        w1_all[2:],
        w2_all[2:],
        None,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        jnp.array([2], dtype=jnp.int32),
        topk=2,
        global_num_experts=4,
        activation="silu",
        use_ep=True,
        acc_dtype_name="float32",
        x_quant_block_size=None,
    )
    ep_sum = out_ep0.astype(jnp.float32) + out_ep1.astype(jnp.float32)
    np.testing.assert_allclose(
        np.asarray(ep_sum), np.asarray(out_full, dtype=np.float32), rtol=2e-2, atol=2e-2
    )
    np.testing.assert_array_equal(np.asarray(ep_sum[4:]), 0.0)


def test_routing_moe_tpu_lowering_and_error_contracts(monkeypatch) -> None:
    # Error contracts
    x = jnp.zeros((8, 128), dtype=jnp.bfloat16)
    w1 = jnp.zeros((2, 128, 256), dtype=jnp.bfloat16)
    w2 = jnp.zeros((2, 128, 128), dtype=jnp.bfloat16)
    bias = jnp.zeros((2, 1, 128), dtype=jnp.bfloat16)
    tw = jnp.ones((8, 2), dtype=jnp.float32)
    ti = jnp.zeros((8, 2), dtype=jnp.int32)
    es = jnp.zeros((1,), dtype=jnp.int32)
    with pytest.raises(NotImplementedError, match="does not support expert bias"):
        routing_moe_jax_fn(
            x,
            w1,
            w2,
            None,
            None,
            None,
            bias,
            tw,
            ti,
            es,
            topk=2,
            global_num_experts=2,
            activation="silu",
            use_ep=False,
            acc_dtype_name="float32",
            x_quant_block_size=None,
        )
    with pytest.raises(ValueError, match="VLLM_TPU_ROUTING_MOE_ACC_DTYPE"):
        _get_routing_moe_op(
            topk=2,
            global_num_experts=2,
            activation="silu",
            use_ep=False,
            acc_dtype_name="int8",
        )

    # Abstract TPU7x HLO lowering check with W4A8 + 640 partial-chunk width
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    mesh = jax.sharding.AbstractMesh(
        (1,), ("x",), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    )
    fn = jax.jit(
        functools.partial(
            routing_moe_jax_fn,
            topk=2,
            global_num_experts=16,
            activation="silu",
            use_ep=True,
            acc_dtype_name="float32",
            x_quant_block_size=128,
        )
    )
    shape = jax.ShapeDtypeStruct
    with jax.sharding.use_abstract_mesh(mesh):
        hlo = (
            fn.trace(
                shape((16, 2560), jnp.bfloat16),
                shape((4, 2560, 1280), jnp.float4_e2m1fn),
                shape((4, 640, 2560), jnp.float4_e2m1fn),
                shape((4, 20, 1, 1280), jnp.bfloat16),
                shape((4, 5, 1, 2560), jnp.bfloat16),
                None,
                None,
                shape((16, 2), jnp.float32),
                shape((16, 2), jnp.int32),
                shape((1,), jnp.int32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
    assert "routing_gmm" in hlo
