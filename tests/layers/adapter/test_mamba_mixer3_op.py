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
"""Unit tests for VllmMambaMixer3 custom op and JAX/Pallas Mamba-3 core."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from vllm.config import (
    CompilationMode,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)

from vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op import (
    Mamba3RMSNormGated,
    MambaMixer3,
    VllmMambaMixer3,
    mamba_mixer3_core_tpu,
)
from vllm_torchtpu.layers.core.mamba_mixer3 import run_jax_mamba_mixer3
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


def _mesh():
    return SimpleNamespace(shape={"model": 1}, axis_names=("model",))


def _vllm_config(
    *,
    dp_size: int = 1,
    pcp_size: int = 1,
    speculative_config: object | None = None,
):
    return SimpleNamespace(
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )


def _mamba3_layer(
    prefix: str = "model.layers.0.mixer",
    *,
    tp_size: int = 1,
    n_groups: int = 1,
    is_mimo: bool = False,
    mimo_rank: int = 4,
    use_rms_norm: bool = True,
):
    layer = VllmMambaMixer3.__new__(VllmMambaMixer3)
    torch.nn.Module.__init__(layer)
    layer.prefix = prefix
    layer.tp_size = tp_size
    layer.num_heads = 8
    layer.head_dim = 16
    layer.ssm_state_size = 16
    layer.n_groups = n_groups
    layer.is_mimo = is_mimo
    layer.mimo_rank = mimo_rank if is_mimo else 1
    layer.rope_fraction = 0.5 if is_mimo else 1.0
    layer.num_rope_angles = int((layer.ssm_state_size // 2) * layer.rope_fraction)
    layer.A_floor = 1e-4
    layer.chunk_size = 8
    layer.rms_norm_eps = 1e-5
    layer.use_rms_norm = use_rms_norm
    layer.model_config = None
    layer.cache_config = None
    layer._apply_py_norm = False

    layer.intermediate_size = layer.num_heads * layer.head_dim
    layer.tped_intermediate_size = layer.intermediate_size // layer.tp_size
    layer.tped_dt_size = layer.num_heads // layer.tp_size
    layer.tp_n_groups = (
        layer.n_groups // layer.tp_size
        if layer.n_groups % layer.tp_size == 0
        else layer.n_groups
    )
    layer.groups_ssm_state_size = (
        layer.mimo_rank * layer.n_groups * layer.ssm_state_size
    )
    layer.tped_bc_size = layer.mimo_rank * layer.tp_n_groups * layer.ssm_state_size
    layer.aux_dim = (
        layer.mimo_rank * (layer.head_dim + layer.ssm_state_size)
        + layer.num_rope_angles
    )

    layer.B_norm_weight = torch.nn.Parameter(torch.ones(layer.ssm_state_size))
    layer.C_norm_weight = torch.nn.Parameter(torch.ones(layer.ssm_state_size))
    if is_mimo:
        layer.B_bias = torch.nn.Parameter(
            torch.ones(layer.tped_dt_size, layer.mimo_rank, layer.ssm_state_size)
        )
        layer.C_bias = torch.nn.Parameter(
            torch.ones(layer.tped_dt_size, layer.mimo_rank, layer.ssm_state_size)
        )
        layer.Psi = torch.nn.Parameter(
            torch.ones(layer.tped_dt_size, layer.mimo_rank, layer.head_dim)
            / layer.mimo_rank
        )
        layer.Phi = torch.nn.Parameter(
            torch.ones(layer.tped_dt_size, layer.mimo_rank, layer.head_dim)
            / layer.mimo_rank
        )
        layer.Zeta = torch.nn.Parameter(
            torch.ones(layer.tped_dt_size, layer.mimo_rank, layer.head_dim)
        )
    else:
        layer.B_bias = torch.nn.Parameter(
            torch.ones(layer.tped_dt_size, layer.ssm_state_size)
        )
        layer.C_bias = torch.nn.Parameter(
            torch.ones(layer.tped_dt_size, layer.ssm_state_size)
        )
        layer.Psi = None
        layer.Phi = None
        layer.Zeta = None

    layer.D = torch.nn.Parameter(torch.ones(layer.tped_dt_size))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(layer.tped_dt_size))
    layer.norm = SimpleNamespace(
        weight=(torch.ones(layer.tped_intermediate_size) if use_rms_norm else None),
        variance_epsilon=1e-5,
        n_groups=n_groups,
    )
    return layer


def _np_silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _np_rms_norm_last(
    x: np.ndarray, weight: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    var = np.mean(np.square(x), axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + eps)) * weight


def _mamba3_numpy_sequential_ref(
    proj: np.ndarray,
    aux_init: np.ndarray,
    rec_init: np.ndarray,
    B_norm_w: np.ndarray,
    C_norm_w: np.ndarray,
    B_bias: np.ndarray,
    C_bias: np.ndarray,
    D: np.ndarray,
    dt_bias: np.ndarray,
    norm_w: np.ndarray | None,
    psi: np.ndarray | None,
    phi: np.ndarray | None,
    zeta: np.ndarray | None,
    state_indices: np.ndarray,
    query_start_loc: np.ndarray,
    distribution: np.ndarray,
    seq_lens: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    num_rope_angles: int,
    is_mimo: bool,
    mimo_rank: int,
    A_floor: float = 1e-4,
    rms_norm_eps: float = 1e-5,
    apply_rms_norm: bool = True,
    flip_trap: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent per-token sequential NumPy Mamba-3 oracle (SISO & MIMO).

    Evaluates ``h_t = exp(A_t dt_t) * (h_{t-1} + beta_t x_{t-1} k_{t-1}^T)
    + gamma_t x_t k_t^T`` and ``y_t = q_t @ h_t + D * x_t`` step by step per
    request with explicit cumulative 2D RoPE, without calling any JAX helper or
    separating diagonal vs strictly-causal terms.
    """
    r = mimo_rank if is_mimo else 1
    num_tokens = proj.shape[0]
    num_slots = rec_init.shape[0]
    max_reqs = seq_lens.shape[0]
    num_valid = int(np.clip(distribution[2], 0, max_reqs))

    d_inner = num_heads * head_dim
    d_bc = r * n_groups * d_state
    group_rep = num_heads // n_groups
    dim_x_aux = r * head_dim
    dim_k_aux = r * d_state

    aux_out = np.array(aux_init, dtype=np.float32, copy=True)
    rec_out = np.array(rec_init, dtype=np.float32, copy=True)
    y_out = np.zeros((num_tokens, d_inner), dtype=np.float32)

    s1 = d_inner
    s2 = s1 + d_inner
    s3 = s2 + d_bc
    s4 = s3 + d_bc
    s5 = s4 + num_heads
    s6 = s5 + num_heads
    s7 = s6 + num_heads

    for seq_idx in range(num_valid):
        start = int(query_start_loc[seq_idx])
        end = int(query_start_loc[seq_idx + 1])
        q_len = max(end - start, 0)
        if q_len == 0:
            continue
        slot = int(state_indices[seq_idx])
        has_slot = 0 <= slot < num_slots
        has_init = has_slot and (int(seq_lens[seq_idx]) - q_len > 0)

        if has_init:
            h = rec_out[slot].copy()
            aux_s = aux_out[slot]
            prev_x = aux_s[:, :dim_x_aux].reshape(num_heads, r, head_dim).copy()
            prev_k = (
                aux_s[:, dim_x_aux : dim_x_aux + dim_k_aux]
                .reshape(num_heads, r, d_state)
                .copy()
            )
            theta = aux_s[:, dim_x_aux + dim_k_aux :].copy()
        else:
            h = np.zeros((num_heads, head_dim, d_state), dtype=np.float32)
            prev_x = np.zeros((num_heads, r, head_dim), dtype=np.float32)
            prev_k = np.zeros((num_heads, r, d_state), dtype=np.float32)
            theta = np.zeros((num_heads, num_rope_angles), dtype=np.float32)

        for local_t, tok in enumerate(range(start, end)):
            row = proj[tok].astype(np.float32)
            z_raw = row[:s1]
            v_t = row[s1:s2].reshape(num_heads, head_dim)
            b_raw = row[s2:s3].reshape(n_groups, r, d_state)
            c_raw = row[s3:s4].reshape(n_groups, r, d_state)
            dd_dt = row[s4:s5]
            dd_A = row[s5:s6]
            trap_raw = row[s6:s7]
            angle_raw = row[s7:]

            b_norm = _np_rms_norm_last(b_raw, B_norm_w, rms_norm_eps)
            c_norm = _np_rms_norm_last(c_raw, C_norm_w, rms_norm_eps)
            b_exp = np.repeat(b_norm, group_rep, axis=0)
            c_exp = np.repeat(c_norm, group_rep, axis=0)
            if is_mimo:
                k_pre = b_exp + B_bias
                q_pre = c_exp + C_bias
            else:
                k_pre = b_exp + B_bias[:, None, :]
                q_pre = c_exp + C_bias[:, None, :]

            A_t = np.minimum(-np.logaddexp(0.0, dd_A), -A_floor)
            dt_t = np.logaddexp(0.0, dd_dt + dt_bias)
            trap_t = 1.0 / (1.0 + np.exp(-trap_raw))
            if flip_trap:
                trap_t = 1.0 - trap_t
            gamma_t = dt_t * trap_t
            beta_t = dt_t * (1.0 - trap_t)

            theta = np.mod(
                theta + np.tanh(angle_raw)[None, :] * np.pi * dt_t[:, None],
                2.0 * np.pi,
            )
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            if is_mimo:
                assert psi is not None and phi is not None and zeta is not None
                half = d_state // 2
                k_rot = k_pre.copy()
                q_rot = q_pre.copy()
                for arr in (k_rot, q_rot):
                    a0 = arr[:, :, :num_rope_angles].copy()
                    a1 = arr[:, :, half : half + num_rope_angles].copy()
                    arr[:, :, :num_rope_angles] = (
                        a0 * cos_t[:, None, :] - a1 * sin_t[:, None, :]
                    )
                    arr[:, :, half : half + num_rope_angles] = (
                        a0 * sin_t[:, None, :] + a1 * cos_t[:, None, :]
                    )
                x_t = v_t[:, None, :] * psi
            else:
                rot_dim = 2 * num_rope_angles
                k_rot = k_pre.copy()
                q_rot = q_pre.copy()
                for arr in (k_rot, q_rot):
                    pairs = arr[:, 0, :rot_dim].reshape(num_heads, num_rope_angles, 2)
                    p0 = pairs[:, :, 0].copy()
                    p1 = pairs[:, :, 1].copy()
                    out_pairs = np.stack(
                        [p0 * cos_t - p1 * sin_t, p0 * sin_t + p1 * cos_t], axis=-1
                    ).reshape(num_heads, rot_dim)
                    arr[:, 0, :rot_dim] = out_pairs
                x_t = v_t[:, None, :]

            use_prev = (local_t > 0) or has_init
            prev_outer = (
                np.einsum("hrp,hrn->hpn", prev_x, prev_k)
                if use_prev
                else np.zeros_like(h)
            )
            cur_outer = np.einsum("hrp,hrn->hpn", x_t, k_rot)
            h = (
                np.exp(A_t * dt_t)[:, None, None]
                * (h + beta_t[:, None, None] * prev_outer)
                + gamma_t[:, None, None] * cur_outer
            )

            y_ssm = np.einsum("hrn,hpn->hrp", q_rot, h) + D[:, None, None] * x_t
            if is_mimo:
                assert phi is not None and zeta is not None
                z_r = z_raw.reshape(num_heads, 1, head_dim)
                y_tok = np.sum(y_ssm * _np_silu(z_r * zeta) * phi, axis=1).reshape(
                    d_inner
                )
                gate_tok = None
            else:
                y_tok = y_ssm[:, 0, :].reshape(d_inner)
                gate_tok = z_raw

            if apply_rms_norm:
                val = y_tok * _np_silu(gate_tok) if gate_tok is not None else y_tok
                if norm_w is not None:
                    grouped = val.reshape(n_groups, d_inner // n_groups)
                    var = np.mean(np.square(grouped), axis=-1, keepdims=True)
                    val = (grouped / np.sqrt(var + rms_norm_eps)).reshape(
                        d_inner
                    ) * norm_w
                y_out[tok] = val
            else:
                y_out[tok] = y_tok

            prev_x = x_t.copy()
            prev_k = k_rot.copy()

        if has_slot:
            rec_out[slot] = h
            aux_out[slot] = np.concatenate(
                [
                    prev_x.reshape(num_heads, dim_x_aux),
                    prev_k.reshape(num_heads, dim_k_aux),
                    theta,
                ],
                axis=-1,
            )

    return aux_out, rec_out, y_out


class TestVllmMambaMixer3:
    def test_init_builds_mamba3_op_and_inherits_state_dtype(self):
        assert "get_state_dtype" not in VllmMambaMixer3.__dict__
        mock_op = MagicMock()

        def _fake_base_init(self, *args, **kwargs):
            self.cache_config = None

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch.object(MambaMixer3, "__init__", _fake_base_init),
            patch.object(
                VllmMambaMixer3, "_build_mamba_op", return_value=mock_op
            ) as mock_build,
        ):
            layer = VllmMambaMixer3()

        mock_build.assert_called_once_with()
        assert layer.mamba_op is mock_op

    @pytest.mark.parametrize(
        "cache_kwargs",
        [
            {"enable_prefix_caching": True, "mamba_cache_mode": "none"},
            {"enable_prefix_caching": False, "mamba_cache_mode": "all"},
            {"enable_prefix_caching": False, "mamba_cache_mode": "align"},
        ],
    )
    def test_init_rejects_prefix_caching(self, cache_kwargs):
        def _fake_base_init(self, *args, **kwargs):
            self.cache_config = SimpleNamespace(**cache_kwargs)

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch.object(MambaMixer3, "__init__", _fake_base_init),
            pytest.raises(
                NotImplementedError,
                match="Prefix caching is not yet supported for MambaMixer3 on TPU",
            ),
        ):
            VllmMambaMixer3()

    def test_init_rejects_speculative_decoding_and_pcp(self):
        def _fake_base_init(self, *args, **kwargs):
            self.cache_config = SimpleNamespace(
                enable_prefix_caching=False, mamba_cache_mode="none"
            )

        spec_vllm = _vllm_config(
            speculative_config=SimpleNamespace(num_speculative_tokens=2)
        )
        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=spec_vllm),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_current_vllm_config_or_none",
                return_value=spec_vllm,
            ),
            patch.object(MambaMixer3, "__init__", _fake_base_init),
            pytest.raises(
                NotImplementedError,
                match="Speculative decoding is not yet supported for MambaMixer3",
            ),
        ):
            VllmMambaMixer3()

        pcp_vllm = _vllm_config(pcp_size=2)
        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=pcp_vllm),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_current_vllm_config_or_none",
                return_value=pcp_vllm,
            ),
            patch.object(MambaMixer3, "__init__", _fake_base_init),
            pytest.raises(
                NotImplementedError,
                match="Prefill context parallelism \\(PCP > 1\\) is not yet supported",
            ),
        ):
            VllmMambaMixer3()

    def test_mamba_mixer3_tp_validation_and_weight_loaders(self):
        vllm_cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
        vllm_cfg.compilation_config.mode = CompilationMode.NONE

        def _tp_patches(tp_size: int, tp_rank: int):
            targets_ws = (
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_tensor_model_parallel_world_size",
                "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
                "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_world_size",
                "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            )
            targets_rk = (
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_tensor_model_parallel_rank",
                "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
                "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_rank",
                "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            )
            return [patch(t, return_value=tp_size) for t in targets_ws] + [
                patch(t, return_value=tp_rank) for t in targets_rk
            ]

        from contextlib import ExitStack

        # 1. Invalid num_heads % tp_size != 0 -> ValueError
        with ExitStack() as stack:
            stack.enter_context(set_current_vllm_config(vllm_cfg))
            for p in _tp_patches(3, 0):
                stack.enter_context(p)
            with pytest.raises(ValueError, match="num_heads \\(8\\) to be divisible"):
                MambaMixer3(
                    hidden_size=32,
                    ssm_state_size=8,
                    intermediate_size=64,
                    num_heads=8,
                    head_dim=8,
                    n_groups=1,
                )

        # 2. Invalid n_groups % tp_size != 0 and n_groups != 1 -> ValueError
        with ExitStack() as stack:
            stack.enter_context(set_current_vllm_config(vllm_cfg))
            for p in _tp_patches(2, 0):
                stack.enter_context(p)
            with pytest.raises(ValueError, match="n_groups \\(3\\) to be divisible"):
                MambaMixer3(
                    hidden_size=32,
                    ssm_state_size=8,
                    intermediate_size=64,
                    num_heads=8,
                    head_dim=8,
                    n_groups=3,
                )

        # 3. TP=2, rank=1: n_groups=1 (replicated B/C) vs n_groups=2 (sharded)
        with ExitStack() as stack:
            stack.enter_context(set_current_vllm_config(vllm_cfg))
            stack.enter_context(
                set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config())
            )
            stack.enter_context(
                patch(
                    "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                    return_value=MagicMock(),
                )
            )
            for p in _tp_patches(2, 1):
                stack.enter_context(p)
            siso_tp = MambaMixer3(
                hidden_size=16,
                ssm_state_size=8,
                intermediate_size=32,
                use_bias=True,
                n_groups=1,
                num_heads=4,
                head_dim=8,
                rope_fraction=0.5,
            )
            mimo_tp = MambaMixer3(
                hidden_size=16,
                ssm_state_size=8,
                intermediate_size=32,
                n_groups=2,
                num_heads=4,
                head_dim=8,
                is_mimo=True,
                mimo_rank=4,
                rope_fraction=0.5,
            )

        # SISO TP=2 rank=1: global=32+32+8+8+4+4+4+2=94, local=16+16+8+8+2+2+2+2=56
        assert siso_tp.total_proj_size == 94
        assert siso_tp.local_proj_size == 56
        w_global = torch.arange(94 * 16, dtype=torch.float32).reshape(94, 16)
        siso_tp.in_proj.weight.weight_loader(siso_tp.in_proj.weight, w_global)
        expected_siso_r1 = torch.cat(
            [
                w_global[16:32],  # z shard 1
                w_global[48:64],  # v shard 1
                w_global[64:72],  # B replicated (n_groups=1)
                w_global[72:80],  # C replicated (n_groups=1)
                w_global[82:84],  # dd_dt shard 1
                w_global[86:88],  # dd_A shard 1
                w_global[90:92],  # trap shard 1
                w_global[92:94],  # rope angles replicated
            ],
            dim=0,
        )
        torch.testing.assert_close(siso_tp.in_proj.weight, expected_siso_r1)

        # 3D SISO B_bias (H, 1, N) squeezed & sharded to rank 1
        b_bias_3d = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 1, 8)
        siso_tp.B_bias.weight_loader(siso_tp.B_bias, b_bias_3d)
        torch.testing.assert_close(siso_tp.B_bias, b_bias_3d.squeeze(1)[2:4])

        # MIMO TP=2 rank=1 with n_groups=2: B/C (64 each) are sharded -> 32 each!
        assert mimo_tp.total_proj_size == 32 + 32 + 64 + 64 + 4 + 4 + 4 + 2
        assert mimo_tp.local_proj_size == 16 + 16 + 32 + 32 + 2 + 2 + 2 + 2
        w_mimo_global = torch.arange(
            mimo_tp.total_proj_size * 16, dtype=torch.float32
        ).reshape(mimo_tp.total_proj_size, 16)
        mimo_tp.in_proj.weight.weight_loader(mimo_tp.in_proj.weight, w_mimo_global)
        torch.testing.assert_close(
            mimo_tp.in_proj.weight[32:64], w_mimo_global[64 + 32 : 64 + 64]
        )

        psi_global = torch.arange(4 * 4 * 8, dtype=torch.float32).reshape(4, 4, 8)
        assert mimo_tp.Psi is not None
        mimo_tp.Psi.weight_loader(mimo_tp.Psi, psi_global)
        torch.testing.assert_close(mimo_tp.Psi, psi_global[2:4])

        with pytest.raises(ValueError, match="Expected in_proj loaded_weight dim 0"):
            siso_tp.in_proj.weight.weight_loader(
                siso_tp.in_proj.weight, torch.zeros(10, 16)
            )
        with pytest.raises(ValueError, match="Expected head parameter loaded_weight"):
            siso_tp.D.weight_loader(siso_tp.D, torch.zeros(7))

    def test_build_mamba_op_caches_shared_jax_op_and_registers_fake(self):
        layer0 = _mamba3_layer("model.layers.0.mixer")
        layer1 = _mamba3_layer("model.layers.1.mixer")
        layer2 = _mamba3_layer("model.layers.2.mixer", is_mimo=True)
        fake_jax_op = MagicMock()
        shared_mesh = _mesh()

        with (
            set_vllm_model_wrapper_context(
                mesh=shared_mesh, vllm_config=_vllm_config()
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                return_value=fake_jax_op,
            ) as mock_jax_op,
        ):
            layer0._build_mamba_op()
            layer1._build_mamba_op()
            assert mock_jax_op.call_count == 1
            assert fake_jax_op.register_fake.call_count == 1

            layer2._build_mamba_op()
            assert mock_jax_op.call_count == 2
            assert fake_jax_op.register_fake.call_count == 2

        assert mock_jax_op.call_args_list[0].args[0].startswith("pallas::mamba_mixer3_")
        assert mock_jax_op.call_args_list[1].args[0].startswith("pallas::mamba_mixer3_")
        assert (
            mock_jax_op.call_args_list[0].args[0]
            != mock_jax_op.call_args_list[1].args[0]
        )
        assert mock_jax_op.call_args_list[0].kwargs["donate_argnums"] == (1, 2)
        assert mock_jax_op.call_args_list[0].args[1].keywords["interpret"] is False

        fake_fn = fake_jax_op.register_fake.call_args_list[0].args[0]
        total_proj_dim = (
            2 * layer0.tped_intermediate_size
            + 2 * layer0.tped_bc_size
            + 3 * layer0.tped_dt_size
            + layer0.num_rope_angles
        )
        proj = torch.zeros((16, total_proj_dim))
        aux_shape, ssm_shape = layer0.get_state_shape()
        aux_state = torch.zeros((8, *aux_shape))
        rec_state = torch.zeros((8, *ssm_shape))
        fake_aux, fake_rec, fake_out = fake_fn(proj, aux_state, rec_state)
        assert fake_aux.shape == aux_state.shape
        assert fake_rec.shape == rec_state.shape
        assert fake_out.shape == (16, layer0.tped_intermediate_size)
        assert layer0.get_state_dtype() == (torch.float32, torch.float32)

    def test_use_fused_jax_rms_norm_multiprocess_tp_single_group(self):
        layer_tp2_g1 = _mamba3_layer(tp_size=2, n_groups=1, use_rms_norm=True)
        assert layer_tp2_g1._use_fused_jax_rms_norm() is False

        layer_tp2_g1_no_norm = _mamba3_layer(tp_size=2, n_groups=1, use_rms_norm=False)
        assert layer_tp2_g1_no_norm._use_fused_jax_rms_norm() is True

        layer_tp2_g2 = _mamba3_layer(tp_size=2, n_groups=2, use_rms_norm=True)
        assert layer_tp2_g2._use_fused_jax_rms_norm() is True

        layer_tp1_g1 = _mamba3_layer(tp_size=1, n_groups=1, use_rms_norm=True)
        assert layer_tp1_g1._use_fused_jax_rms_norm() is True

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                return_value=MagicMock(),
            ) as mock_jax_op,
        ):
            layer_tp2_g1._build_mamba_op()

        assert layer_tp2_g1._apply_py_norm is True
        wrapped_fn = mock_jax_op.call_args.args[1]
        assert wrapped_fn.keywords["apply_rms_norm"] is False

    def test_siso_apply_py_norm_avoids_double_silu_gating(self):
        """Verifies _apply_py_norm=True matches fused JAX norm without double SiLU."""
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        def _passthrough_jax_op(op_name, fn, donate_argnums=()):
            def _op(*torch_args):
                jax_args = [
                    jnp.asarray(t.detach().cpu().numpy())
                    if isinstance(t, torch.Tensor)
                    else None
                    for t in torch_args
                ]
                new_aux, new_rec, out = fn(*jax_args)
                return (
                    torch.from_numpy(np.array(new_aux, copy=True)),
                    torch.from_numpy(np.array(new_rec, copy=True)),
                    torch.from_numpy(np.array(out, copy=True)),
                )

            _op.register_fake = lambda _: None
            return _op

        torch.manual_seed(77)
        layer_fused = _mamba3_layer("model.layers.0.mixer", tp_size=1, n_groups=1)
        layer_py = _mamba3_layer("model.layers.1.mixer", tp_size=1, n_groups=1)

        vllm_cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
        vllm_cfg.compilation_config.mode = CompilationMode.NONE
        with (
            set_current_vllm_config(vllm_cfg),
            patch(
                "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_rank",
                return_value=0,
            ),
        ):
            norm_weight = torch.linspace(0.5, 1.5, layer_fused.tped_intermediate_size)
            real_norm = Mamba3RMSNormGated(
                layer_fused.intermediate_size, 1, True, eps=1e-5
            )
            real_norm.weight.data.copy_(norm_weight)
            layer_fused.norm = real_norm
            layer_py.norm = real_norm

        total_proj_dim = (
            2 * layer_fused.tped_intermediate_size
            + 2 * layer_fused.tped_bc_size
            + 3 * layer_fused.tped_dt_size
            + layer_fused.num_rope_angles
        )
        proj_tensor = torch.randn(6, total_proj_dim)
        layer_fused.in_proj = MagicMock(return_value=(proj_tensor.clone(), None))
        layer_py.in_proj = MagicMock(return_value=(proj_tensor.clone(), None))
        layer_fused.out_proj = MagicMock(side_effect=lambda x: (x, None))
        layer_py.out_proj = MagicMock(side_effect=lambda x: (x, None))

        with (
            set_vllm_model_wrapper_context(mesh=mesh, vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                side_effect=_passthrough_jax_op,
            ),
        ):
            layer_fused.mamba_op = layer_fused._build_mamba_op(interpret=True)
            with patch.object(layer_py, "_use_fused_jax_rms_norm", return_value=False):
                layer_py.mamba_op = layer_py._build_mamba_op(interpret=True)

        assert layer_fused._apply_py_norm is False
        assert layer_py._apply_py_norm is True

        aux_shape, ssm_shape = layer_fused.get_state_shape()
        layer_fused.kv_cache = (
            torch.zeros((2, *aux_shape)),
            torch.zeros((2, *ssm_shape)),
        )
        layer_py.kv_cache = (
            torch.zeros((2, *aux_shape)),
            torch.zeros((2, *ssm_shape)),
        )
        meta = SimpleNamespace(
            mamba_state_indices=torch.tensor([0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
            request_distribution=torch.tensor([0, 1, 1], dtype=torch.int32),
            seq_lens=torch.tensor([6], dtype=torch.int32),
            block_tables=None,
        )
        fc = SimpleNamespace(
            attn_metadata={
                layer_fused.prefix: meta,
                layer_py.prefix: meta,
            }
        )
        with patch(
            "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.get_forward_context",
            return_value=fc,
        ):
            out_fused = layer_fused.forward(torch.zeros(6, 64))
            out_py = layer_py.forward(torch.zeros(6, 64))

        torch.testing.assert_close(out_py, out_fused, rtol=1e-4, atol=1e-4)

    def test_mimo_apply_py_norm_runs_ungated_tp_rmsnorm(self):
        """Verifies MIMO with tp_size=2, n_groups=1 runs ungated TP all-reduce norm."""
        layer = _mamba3_layer(
            "model.layers.0.mixer",
            tp_size=2,
            n_groups=1,
            is_mimo=True,
            use_rms_norm=True,
        )
        vllm_cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
        vllm_cfg.compilation_config.mode = CompilationMode.NONE
        with (
            set_current_vllm_config(vllm_cfg),
            patch(
                "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_world_size",
                return_value=2,
            ),
            patch(
                "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_rank",
                return_value=0,
            ),
        ):
            norm = Mamba3RMSNormGated(layer.intermediate_size, 1, True, eps=1e-5)
            norm.weight.data.copy_(
                torch.linspace(0.75, 1.25, layer.tped_intermediate_size)
            )
            layer.norm = norm

        ssm_rank0 = torch.randn(4, layer.tped_intermediate_size)
        ssm_rank1 = torch.randn(4, layer.tped_intermediate_size)
        other_rank_sq_sum = ssm_rank1.pow(2).sum(dim=-1, keepdim=True)

        aux_shape, ssm_shape = layer.get_state_shape()
        aux_state = torch.zeros((2, *aux_shape))
        rec_state = torch.zeros((2, *ssm_shape))
        layer.kv_cache = (aux_state, rec_state)

        fake_jax_op = MagicMock(return_value=(aux_state, rec_state, ssm_rank0))
        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                return_value=fake_jax_op,
            ),
        ):
            layer.mamba_op = layer._build_mamba_op()

        assert layer._apply_py_norm is True

        total_proj_dim = (
            2 * layer.tped_intermediate_size
            + 2 * layer.tped_bc_size
            + 3 * layer.tped_dt_size
            + layer.num_rope_angles
        )
        layer.in_proj = MagicMock(return_value=(torch.randn(4, total_proj_dim), None))
        layer.out_proj = MagicMock(side_effect=lambda x: (x, None))

        meta = SimpleNamespace(
            mamba_state_indices=torch.tensor([0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            request_distribution=torch.tensor([0, 1, 1], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            block_tables=None,
        )
        fc = SimpleNamespace(attn_metadata={layer.prefix: meta})

        with (
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.get_forward_context",
                return_value=fc,
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.tensor_model_parallel_all_reduce",
                side_effect=lambda local_sum: local_sum + other_rank_sq_sum,
            ) as mock_all_reduce,
        ):
            out_rank0 = layer.forward(torch.zeros(4, 64))

        mock_all_reduce.assert_called_once()
        full_x = torch.cat([ssm_rank0, ssm_rank1], dim=-1)
        global_var = full_x.pow(2).mean(dim=-1, keepdim=True)
        expected_rank0 = norm.weight * (ssm_rank0 * torch.rsqrt(global_var + 1e-5))
        torch.testing.assert_close(out_rank0, expected_rank0, rtol=1e-5, atol=1e-5)

    def test_forward_warmup_and_rejects_unified_single_tensor_pool(self):
        layer = _mamba3_layer()
        layer.kv_cache = (torch.tensor([]), torch.tensor([]))
        layer.mamba_op = MagicMock()

        total_proj_dim = (
            2 * layer.tped_intermediate_size
            + 2 * layer.tped_bc_size
            + 3 * layer.tped_dt_size
            + layer.num_rope_angles
        )
        layer.in_proj = MagicMock(
            side_effect=lambda x: (torch.ones(x.size(0), total_proj_dim), None)
        )
        layer.norm = MagicMock(side_effect=lambda x, gate: x)
        layer.out_proj = MagicMock(
            side_effect=lambda x: (torch.full((x.size(0), 256), 2.0), None)
        )

        hidden_states = torch.randn(8, 256)
        out = layer.forward(hidden_states)

        layer.mamba_op.assert_not_called()
        layer.norm.assert_called_once()
        assert out.shape == (8, 256)

        layer.kv_cache = torch.zeros((4, 8, 16))
        with pytest.raises(
            NotImplementedError, match="Unified single-tensor Mamba block pool"
        ):
            layer.forward(hidden_states)

    @patch("vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.get_forward_context")
    def test_forward_executes_mamba_op_and_updates_caches_in_place(
        self, mock_get_forward_context
    ):
        layer = _mamba3_layer()
        num_tokens = 12
        max_seqs = 4
        aux_shape, ssm_shape = layer.get_state_shape()
        aux_state = torch.zeros((max_seqs, *aux_shape))
        rec_state = torch.zeros((max_seqs, *ssm_shape))
        layer.kv_cache = (aux_state, rec_state)

        new_aux = torch.ones_like(aux_state) * 3.0
        new_rec = torch.ones_like(rec_state) * 5.0
        expected_ssm_out = torch.ones((num_tokens, layer.tped_intermediate_size)) * 7.0

        fake_jax_op = MagicMock(return_value=(new_aux, new_rec, expected_ssm_out))
        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                return_value=fake_jax_op,
            ),
        ):
            layer.mamba_op = layer._build_mamba_op()

            total_proj_dim = (
                2 * layer.tped_intermediate_size
                + 2 * layer.tped_bc_size
                + 3 * layer.tped_dt_size
                + layer.num_rope_angles
            )
            layer.in_proj = MagicMock(
                side_effect=lambda x: (
                    torch.ones(x.size(0), total_proj_dim),
                    None,
                )
            )
            layer.out_proj = MagicMock(side_effect=lambda x: (x[:, :64], None))

            attn_metadata = SimpleNamespace(
                mamba_state_indices=torch.tensor([0, 1], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 6, 12], dtype=torch.int32),
                request_distribution=torch.tensor([0, 2, 2], dtype=torch.int32),
                seq_lens=torch.tensor([6, 6], dtype=torch.int32),
                block_tables=None,
            )
            mock_fc = MagicMock()
            mock_fc.attn_metadata = {layer.prefix: attn_metadata}
            mock_get_forward_context.return_value = mock_fc

            out = layer.forward(torch.randn(num_tokens, 64))

        assert torch.allclose(aux_state, new_aux)
        assert torch.allclose(rec_state, new_rec)
        assert torch.allclose(out, torch.ones(num_tokens, 64) * 7.0)

    @pytest.mark.parametrize("is_mimo", [False, True])
    def test_ragged_mamba3_matches_independent_numpy_oracle_and_perturbation_fails(
        self, is_mimo: bool
    ):
        """Compares ragged SISO and MIMO execution against an independent per-token
        NumPy sequential oracle and verifies must-fail perturbation controls."""
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        num_heads = 4
        head_dim = 8
        n_groups = 2
        d_state = 8
        r = 4 if is_mimo else 1
        num_rope_angles = d_state // 4 if is_mimo else d_state // 2
        chunk_size = 4
        d_inner = num_heads * head_dim
        aux_dim = r * (head_dim + d_state) + num_rope_angles
        total_proj_dim = (
            2 * d_inner + 2 * r * n_groups * d_state + 3 * num_heads + num_rope_angles
        )

        # Ragged mixed batch: seq 0 prefill (5, fresh), seq 1 decode (1, carried in
        # slot 0), seq 2 prefill continuation (7, carried in slot 4), seq 3 inactive
        # (-1 sentinel slot — must not clobber slot 0!).
        query_start_loc = np.array([0, 5, 6, 13, 13], dtype=np.int32)
        seq_lens = np.array([5, 9, 12, 0], dtype=np.int32)
        state_indices = np.array([2, 0, 4, -1], dtype=np.int32)
        distribution = np.array([1, 3, 3], dtype=np.int32)
        num_tokens = 16  # 3 padded tail tokens beyond active 13
        num_slots = 6

        rng = jax.random.key(5150 + int(is_mimo))
        keys = jax.random.split(rng, 13)
        proj = jax.random.normal(keys[0], (num_tokens, total_proj_dim)) * 0.5
        B_norm_w = jax.random.uniform(keys[1], (d_state,), minval=0.6, maxval=1.4)
        C_norm_w = jax.random.uniform(keys[2], (d_state,), minval=0.6, maxval=1.4)
        bc_shape = (num_heads, r, d_state) if is_mimo else (num_heads, d_state)
        B_bias = jax.random.normal(keys[3], bc_shape) * 0.1
        C_bias = jax.random.normal(keys[4], bc_shape) * 0.1
        D = jax.random.normal(keys[5], (num_heads,)) * 0.5
        dt_bias = jax.random.normal(keys[6], (num_heads,)) * 0.1
        norm_w = jax.random.uniform(keys[7], (d_inner,), minval=0.7, maxval=1.3)
        psi = (
            jax.random.normal(keys[8], (num_heads, r, head_dim)) * 0.3
            if is_mimo
            else None
        )
        phi = (
            jax.random.normal(keys[9], (num_heads, r, head_dim)) * 0.3
            if is_mimo
            else None
        )
        zeta = (
            jax.random.normal(keys[10], (num_heads, r, head_dim)) * 0.5
            if is_mimo
            else None
        )
        aux_init = jax.random.normal(keys[11], (num_slots, num_heads, aux_dim)) * 0.2
        rec_init = (
            jax.random.normal(keys[12], (num_slots, num_heads, head_dim, d_state)) * 0.2
        )

        aux_jax, rec_jax, y_jax = run_jax_mamba_mixer3(
            proj,
            aux_init,
            rec_init,
            B_norm_w,
            C_norm_w,
            B_bias,
            C_bias,
            D,
            dt_bias,
            norm_w,
            psi,
            phi,
            zeta,
            jnp.asarray(state_indices),
            jnp.asarray(query_start_loc),
            jnp.asarray(distribution),
            jnp.asarray(seq_lens),
            mesh=mesh,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
            num_rope_angles=num_rope_angles,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
            mimo_rank=r,
            apply_rms_norm=True,
            interpret=True,
        )

        aux_ref, rec_ref, y_ref = _mamba3_numpy_sequential_ref(
            np.asarray(proj),
            np.asarray(aux_init),
            np.asarray(rec_init),
            np.asarray(B_norm_w),
            np.asarray(C_norm_w),
            np.asarray(B_bias),
            np.asarray(C_bias),
            np.asarray(D),
            np.asarray(dt_bias),
            np.asarray(norm_w),
            None if psi is None else np.asarray(psi),
            None if phi is None else np.asarray(phi),
            None if zeta is None else np.asarray(zeta),
            state_indices,
            query_start_loc,
            distribution,
            seq_lens,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
            num_rope_angles=num_rope_angles,
            is_mimo=is_mimo,
            mimo_rank=r,
            apply_rms_norm=True,
        )

        np.testing.assert_allclose(
            np.asarray(y_jax[:13]), y_ref[:13], rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(np.asarray(rec_jax), rec_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(aux_jax), aux_ref, rtol=1e-4, atol=1e-4)

        # Must-fail control: flipping trapezoidal weight (trap -> 1 - trap) or D
        # skip must diverge measurably from the JAX/Pallas output.
        _, _, y_bad_trap = _mamba3_numpy_sequential_ref(
            np.asarray(proj),
            np.asarray(aux_init),
            np.asarray(rec_init),
            np.asarray(B_norm_w),
            np.asarray(C_norm_w),
            np.asarray(B_bias),
            np.asarray(C_bias),
            np.asarray(D),
            np.asarray(dt_bias),
            np.asarray(norm_w),
            None if psi is None else np.asarray(psi),
            None if phi is None else np.asarray(phi),
            None if zeta is None else np.asarray(zeta),
            state_indices,
            query_start_loc,
            distribution,
            seq_lens,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
            num_rope_angles=num_rope_angles,
            is_mimo=is_mimo,
            mimo_rank=r,
            apply_rms_norm=True,
            flip_trap=True,
        )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                np.asarray(y_jax[:13]), y_bad_trap[:13], rtol=1e-3, atol=1e-3
            )

    def test_siso_prefill_then_decode_matches_full_prefill_exactly(self):
        """Verifies that splitting a sequence into prefill(0..k) + decode(k..L)
        produces the exact same outputs and final states as a single contiguous
        prefill(0..L), proving RoPE phase and trapezoidal boundary continuity."""
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        total_len = 10
        prefill_len = 6
        max_seqs = 2
        num_heads = 4
        head_dim = 8
        n_groups = 1
        d_state = 8
        num_rope_angles = d_state // 2
        chunk_size = 4
        d_inner = num_heads * head_dim
        aux_dim = head_dim + d_state + num_rope_angles

        total_proj_dim = (
            2 * d_inner + 2 * n_groups * d_state + 3 * num_heads + num_rope_angles
        )

        rng = jax.random.key(314)
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 7)

        proj_all = jax.random.normal(k1, (total_len, total_proj_dim), dtype=jnp.float32)
        B_norm_w = jax.random.uniform(k2, (d_state,), minval=0.5, maxval=1.5)
        C_norm_w = jax.random.uniform(k3, (d_state,), minval=0.5, maxval=1.5)
        B_bias = jax.random.normal(k4, (num_heads, d_state)) * 0.1
        C_bias = jax.random.normal(k5, (num_heads, d_state)) * 0.1
        D = jax.random.normal(k6, (num_heads,))
        dt_bias = jax.random.normal(k7, (num_heads,)) * 0.1
        norm_w = jnp.ones((d_inner,), dtype=jnp.float32)

        aux_contig = jnp.zeros((max_seqs, num_heads, aux_dim), dtype=jnp.float32)
        rec_contig = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        aux_contig_out, rec_contig_out, y_contig = run_jax_mamba_mixer3(
            proj_all,
            aux_contig,
            rec_contig,
            B_norm_w,
            C_norm_w,
            B_bias,
            C_bias,
            D,
            dt_bias,
            norm_w,
            None,
            None,
            None,
            jnp.array([0, 0], dtype=jnp.int32),
            jnp.array([0, total_len, total_len], dtype=jnp.int32),
            jnp.array([0, 1, 1], dtype=jnp.int32),
            jnp.array([total_len, 0], dtype=jnp.int32),
            mesh=mesh,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
            num_rope_angles=num_rope_angles,
            chunk_size=chunk_size,
            is_mimo=False,
            mimo_rank=1,
            apply_rms_norm=True,
            interpret=True,
        )

        aux_step = jnp.zeros((max_seqs, num_heads, aux_dim), dtype=jnp.float32)
        rec_step = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )

        aux_step, rec_step, y_prefill = run_jax_mamba_mixer3(
            proj_all[:prefill_len],
            aux_step,
            rec_step,
            B_norm_w,
            C_norm_w,
            B_bias,
            C_bias,
            D,
            dt_bias,
            norm_w,
            None,
            None,
            None,
            jnp.array([0, 0], dtype=jnp.int32),
            jnp.array([0, prefill_len, prefill_len], dtype=jnp.int32),
            jnp.array([0, 1, 1], dtype=jnp.int32),
            jnp.array([prefill_len, 0], dtype=jnp.int32),
            mesh=mesh,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
            num_rope_angles=num_rope_angles,
            chunk_size=chunk_size,
            is_mimo=False,
            mimo_rank=1,
            apply_rms_norm=True,
            interpret=True,
        )

        y_decodes = []
        for step_idx in range(prefill_len, total_len):
            cur_seq_len = step_idx + 1
            aux_step, rec_step, y_dec = run_jax_mamba_mixer3(
                proj_all[step_idx : step_idx + 1],
                aux_step,
                rec_step,
                B_norm_w,
                C_norm_w,
                B_bias,
                C_bias,
                D,
                dt_bias,
                norm_w,
                None,
                None,
                None,
                jnp.array([0, 0], dtype=jnp.int32),
                jnp.array([0, 1, 1], dtype=jnp.int32),
                jnp.array([1, 1, 1], dtype=jnp.int32),
                jnp.array([cur_seq_len, 0], dtype=jnp.int32),
                mesh=mesh,
                num_heads=num_heads,
                head_dim=head_dim,
                n_groups=n_groups,
                d_state=d_state,
                num_rope_angles=num_rope_angles,
                chunk_size=chunk_size,
                is_mimo=False,
                mimo_rank=1,
                apply_rms_norm=True,
                interpret=True,
            )
            y_decodes.append(y_dec)

        y_split = jnp.concatenate([y_prefill, *y_decodes], axis=0)
        np.testing.assert_allclose(y_split, y_contig, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(rec_step[0], rec_contig_out[0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(aux_step[0], aux_contig_out[0], rtol=1e-4, atol=1e-4)

    def test_mimo_prefill_then_decode_matches_full_prefill_exactly(self):
        """Verifies MIMO (is_mimo=True, mimo_rank=4) prefill + decode continuity."""
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        total_len = 6
        prefill_len = 4
        max_seqs = 2
        num_heads = 2
        head_dim = 8
        n_groups = 1
        d_state = 8
        mimo_rank = 4
        num_rope_angles = d_state // 4
        chunk_size = 4
        d_inner = num_heads * head_dim
        aux_dim = mimo_rank * (head_dim + d_state) + num_rope_angles

        total_proj_dim = (
            2 * d_inner
            + 2 * mimo_rank * n_groups * d_state
            + 3 * num_heads
            + num_rope_angles
        )

        rng = jax.random.key(2718)
        keys = jax.random.split(rng, 10)

        proj_all = jax.random.normal(keys[0], (total_len, total_proj_dim))
        B_norm_w = jax.random.uniform(keys[1], (d_state,), minval=0.5, maxval=1.5)
        C_norm_w = jax.random.uniform(keys[2], (d_state,), minval=0.5, maxval=1.5)
        B_bias = jax.random.normal(keys[3], (num_heads, mimo_rank, d_state)) * 0.1
        C_bias = jax.random.normal(keys[4], (num_heads, mimo_rank, d_state)) * 0.1
        D = jax.random.normal(keys[5], (num_heads,))
        dt_bias = jax.random.normal(keys[6], (num_heads,)) * 0.1
        psi = jax.random.normal(keys[7], (num_heads, mimo_rank, head_dim)) * 0.25
        phi = jax.random.normal(keys[8], (num_heads, mimo_rank, head_dim)) * 0.25
        zeta = jax.random.normal(keys[9], (num_heads, mimo_rank, head_dim))

        aux_contig = jnp.zeros((max_seqs, num_heads, aux_dim), dtype=jnp.float32)
        rec_contig = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        aux_contig_out, rec_contig_out, y_contig = run_jax_mamba_mixer3(
            proj_all,
            aux_contig,
            rec_contig,
            B_norm_w,
            C_norm_w,
            B_bias,
            C_bias,
            D,
            dt_bias,
            None,
            psi,
            phi,
            zeta,
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0, total_len], dtype=jnp.int32),
            jnp.array([0, 1, 1], dtype=jnp.int32),
            jnp.array([total_len], dtype=jnp.int32),
            mesh=mesh,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
            num_rope_angles=num_rope_angles,
            chunk_size=chunk_size,
            is_mimo=True,
            mimo_rank=mimo_rank,
            apply_rms_norm=False,
            interpret=True,
        )

        aux_step = jnp.zeros((max_seqs, num_heads, aux_dim), dtype=jnp.float32)
        rec_step = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        aux_step, rec_step, y_prefill = mamba_mixer3_core_tpu(
            proj_all[:prefill_len],
            aux_step,
            rec_step,
            B_norm_w,
            C_norm_w,
            B_bias,
            C_bias,
            D,
            dt_bias,
            None,
            psi,
            phi,
            zeta,
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0, prefill_len], dtype=jnp.int32),
            jnp.array([0, 1, 1], dtype=jnp.int32),
            jnp.array([prefill_len], dtype=jnp.int32),
            mesh=mesh,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
            num_rope_angles=num_rope_angles,
            chunk_size=chunk_size,
            is_mimo=True,
            mimo_rank=mimo_rank,
            A_floor=1e-4,
            rms_norm_eps=1e-5,
            apply_rms_norm=False,
            interpret=True,
        )

        y_decodes = []
        for step_idx in range(prefill_len, total_len):
            aux_step, rec_step, y_dec = mamba_mixer3_core_tpu(
                proj_all[step_idx : step_idx + 1],
                aux_step,
                rec_step,
                B_norm_w,
                C_norm_w,
                B_bias,
                C_bias,
                D,
                dt_bias,
                None,
                psi,
                phi,
                zeta,
                jnp.array([0], dtype=jnp.int32),
                jnp.array([0, 1], dtype=jnp.int32),
                jnp.array([1, 1, 1], dtype=jnp.int32),
                jnp.array([step_idx + 1], dtype=jnp.int32),
                mesh=mesh,
                num_heads=num_heads,
                head_dim=head_dim,
                n_groups=n_groups,
                d_state=d_state,
                num_rope_angles=num_rope_angles,
                chunk_size=chunk_size,
                is_mimo=True,
                mimo_rank=mimo_rank,
                A_floor=1e-4,
                rms_norm_eps=1e-5,
                apply_rms_norm=False,
                interpret=True,
            )
            y_decodes.append(y_dec)

        y_split = jnp.concatenate([y_prefill, *y_decodes], axis=0)
        np.testing.assert_allclose(y_split, y_contig, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(rec_step[0], rec_contig_out[0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(aux_step[0], aux_contig_out[0], rtol=1e-4, atol=1e-4)

    def test_bfloat16_output_dtype_preservation(self):
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        num_tokens = 8
        max_seqs = 2
        num_heads = 2
        head_dim = 8
        n_groups = 1
        d_state = 8
        num_rope_angles = d_state // 2
        chunk_size = 4
        d_inner = num_heads * head_dim
        aux_dim = head_dim + d_state + num_rope_angles

        total_proj_dim = (
            2 * d_inner + 2 * n_groups * d_state + 3 * num_heads + num_rope_angles
        )

        j_projected_states = jnp.ones((num_tokens, total_proj_dim), dtype=jnp.bfloat16)
        aux_state = jnp.zeros((max_seqs, num_heads, aux_dim + 4), dtype=jnp.float32)
        recurrent_state = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        B_norm_w = jnp.ones((d_state,), dtype=jnp.bfloat16)
        C_norm_w = jnp.ones((d_state,), dtype=jnp.bfloat16)
        B_bias = jnp.zeros((num_heads, d_state), dtype=jnp.bfloat16)
        C_bias = jnp.zeros((num_heads, d_state), dtype=jnp.bfloat16)
        D = jnp.ones((num_heads,), dtype=jnp.float32)
        dt_bias = jnp.zeros((num_heads,), dtype=jnp.float32)
        norm_w = jnp.ones((d_inner,), dtype=jnp.bfloat16)

        for apply_rms_norm in (True, False):
            new_aux, new_rec, y_out = mamba_mixer3_core_tpu(
                j_projected_states,
                aux_state,
                recurrent_state,
                B_norm_w,
                C_norm_w,
                B_bias,
                C_bias,
                D,
                dt_bias,
                norm_w,
                None,
                None,
                None,
                jnp.array([0], dtype=jnp.int32),
                jnp.array([0, num_tokens], dtype=jnp.int32),
                jnp.array([0, 1, 1], dtype=jnp.int32),
                jnp.array([num_tokens], dtype=jnp.int32),
                mesh=mesh,
                num_heads=num_heads,
                head_dim=head_dim,
                n_groups=n_groups,
                d_state=d_state,
                num_rope_angles=num_rope_angles,
                chunk_size=chunk_size,
                is_mimo=False,
                mimo_rank=1,
                A_floor=1e-4,
                rms_norm_eps=1e-5,
                apply_rms_norm=apply_rms_norm,
                interpret=True,
            )
            assert y_out.dtype == jnp.bfloat16
            assert new_aux.shape == aux_state.shape
            assert new_rec.shape == recurrent_state.shape

    def test_multi_request_decode_batch_with_padded_tokens_siso_and_mimo(self):
        """Verifies pure decode fast-path across multiple requests with padding."""
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        num_active = 4
        num_tokens = 12
        num_heads = 2
        head_dim = 8
        n_groups = 1
        d_state = 8
        chunk_size = 4

        rng = jax.random.key(9001)
        keys = jax.random.split(rng, 12)

        for is_mimo in (False, True):
            r = 4 if is_mimo else 1
            num_rope_angles = d_state // 4 if is_mimo else d_state // 2
            d_inner = num_heads * head_dim
            aux_dim = r * (head_dim + d_state) + num_rope_angles
            total_proj_dim = (
                2 * d_inner
                + 2 * r * n_groups * d_state
                + 3 * num_heads
                + num_rope_angles
            )

            proj = jax.random.normal(keys[0], (num_tokens, total_proj_dim))
            B_norm_w = jax.random.uniform(keys[1], (d_state,), minval=0.5, maxval=1.5)
            C_norm_w = jax.random.uniform(keys[2], (d_state,), minval=0.5, maxval=1.5)
            B_shape = (num_heads, r, d_state) if is_mimo else (num_heads, d_state)
            B_bias = jax.random.normal(keys[3], B_shape) * 0.1
            C_bias = jax.random.normal(keys[4], B_shape) * 0.1
            D = jax.random.normal(keys[5], (num_heads,))
            dt_bias = jax.random.normal(keys[6], (num_heads,)) * 0.1
            norm_w = None if is_mimo else jnp.ones((d_inner,), dtype=jnp.float32)
            psi = (
                jax.random.normal(keys[7], (num_heads, r, head_dim)) * 0.25
                if is_mimo
                else None
            )
            phi = (
                jax.random.normal(keys[8], (num_heads, r, head_dim)) * 0.25
                if is_mimo
                else None
            )
            zeta = (
                jax.random.normal(keys[9], (num_heads, r, head_dim))
                if is_mimo
                else None
            )

            aux_init = jax.random.normal(keys[10], (8, num_heads, aux_dim)) * 0.1
            rec_init = (
                jax.random.normal(keys[11], (8, num_heads, head_dim, d_state)) * 0.1
            )

            state_indices = jnp.array([2, 0, 5, 3, -1, 0], dtype=jnp.int32)
            query_start_loc = jnp.array([0, 1, 2, 3, 4, 4, 4], dtype=jnp.int32)
            distribution = jnp.array(
                [num_active, num_active, num_active], dtype=jnp.int32
            )
            seq_lens = jnp.array([10, 1, 25, 4, 0, 0], dtype=jnp.int32)

            aux_batch, rec_batch, y_batch = run_jax_mamba_mixer3(
                proj,
                aux_init,
                rec_init,
                B_norm_w,
                C_norm_w,
                B_bias,
                C_bias,
                D,
                dt_bias,
                norm_w,
                psi,
                phi,
                zeta,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                mesh=mesh,
                num_heads=num_heads,
                head_dim=head_dim,
                n_groups=n_groups,
                d_state=d_state,
                num_rope_angles=num_rope_angles,
                chunk_size=chunk_size,
                is_mimo=is_mimo,
                mimo_rank=r,
                apply_rms_norm=not is_mimo,
                interpret=True,
            )

            for i in range(num_active):
                slot = int(state_indices[i])
                aux_i, rec_i, y_i = run_jax_mamba_mixer3(
                    proj[i : i + 1],
                    aux_init,
                    rec_init,
                    B_norm_w,
                    C_norm_w,
                    B_bias,
                    C_bias,
                    D,
                    dt_bias,
                    norm_w,
                    psi,
                    phi,
                    zeta,
                    jnp.array([slot], dtype=jnp.int32),
                    jnp.array([0, 1], dtype=jnp.int32),
                    jnp.array([1, 1, 1], dtype=jnp.int32),
                    jnp.array([int(seq_lens[i])], dtype=jnp.int32),
                    mesh=mesh,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    n_groups=n_groups,
                    d_state=d_state,
                    num_rope_angles=num_rope_angles,
                    chunk_size=chunk_size,
                    is_mimo=is_mimo,
                    mimo_rank=r,
                    apply_rms_norm=not is_mimo,
                    interpret=True,
                )
                np.testing.assert_allclose(y_batch[i], y_i[0], rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(
                    rec_batch[slot], rec_i[slot], rtol=1e-4, atol=1e-4
                )
                np.testing.assert_allclose(
                    aux_batch[slot], aux_i[slot], rtol=1e-4, atol=1e-4
                )

    def test_mixed_prefill_and_decode_batch_matches_individual_sequences(self):
        """Verifies mixed prefill (q_len > 1) + decode (q_len == 1) batches."""
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        q_lens = [1, 7, 1, 10, 1, 0]
        num_active = 5
        total_tokens = sum(q_lens) + 4
        num_heads = 2
        head_dim = 8
        n_groups = 1
        d_state = 8
        chunk_size = 4

        rng = jax.random.key(4242)
        keys = jax.random.split(rng, 12)

        for is_mimo in (False, True):
            r = 4 if is_mimo else 1
            num_rope_angles = d_state // 4 if is_mimo else d_state // 2
            d_inner = num_heads * head_dim
            aux_dim = r * (head_dim + d_state) + num_rope_angles
            total_proj_dim = (
                2 * d_inner
                + 2 * r * n_groups * d_state
                + 3 * num_heads
                + num_rope_angles
            )

            proj = jax.random.normal(keys[0], (total_tokens, total_proj_dim))
            B_norm_w = jax.random.uniform(keys[1], (d_state,), minval=0.5, maxval=1.5)
            C_norm_w = jax.random.uniform(keys[2], (d_state,), minval=0.5, maxval=1.5)
            B_shape = (num_heads, r, d_state) if is_mimo else (num_heads, d_state)
            B_bias = jax.random.normal(keys[3], B_shape) * 0.1
            C_bias = jax.random.normal(keys[4], B_shape) * 0.1
            D = jax.random.normal(keys[5], (num_heads,))
            dt_bias = jax.random.normal(keys[6], (num_heads,)) * 0.1
            norm_w = None if is_mimo else jnp.ones((d_inner,), dtype=jnp.float32)
            psi = (
                jax.random.normal(keys[7], (num_heads, r, head_dim)) * 0.25
                if is_mimo
                else None
            )
            phi = (
                jax.random.normal(keys[8], (num_heads, r, head_dim)) * 0.25
                if is_mimo
                else None
            )
            zeta = (
                jax.random.normal(keys[9], (num_heads, r, head_dim))
                if is_mimo
                else None
            )

            aux_init = jax.random.normal(keys[10], (8, num_heads, aux_dim)) * 0.1
            rec_init = (
                jax.random.normal(keys[11], (8, num_heads, head_dim, d_state)) * 0.1
            )

            state_indices = jnp.array([1, 2, 4, 5, 6, 0], dtype=jnp.int32)
            qsl = [0]
            for ql in q_lens:
                qsl.append(qsl[-1] + ql)
            query_start_loc = jnp.array(qsl, dtype=jnp.int32)
            distribution = jnp.array([3, num_active, num_active], dtype=jnp.int32)
            seq_lens = jnp.array([15, 7, 8, 22, 1, 0], dtype=jnp.int32)

            aux_batch, rec_batch, y_batch = run_jax_mamba_mixer3(
                proj,
                aux_init,
                rec_init,
                B_norm_w,
                C_norm_w,
                B_bias,
                C_bias,
                D,
                dt_bias,
                norm_w,
                psi,
                phi,
                zeta,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                mesh=mesh,
                num_heads=num_heads,
                head_dim=head_dim,
                n_groups=n_groups,
                d_state=d_state,
                num_rope_angles=num_rope_angles,
                chunk_size=chunk_size,
                is_mimo=is_mimo,
                mimo_rank=r,
                apply_rms_norm=not is_mimo,
                interpret=True,
            )

            for i in range(num_active):
                slot = int(state_indices[i])
                start_t = qsl[i]
                end_t = qsl[i + 1]
                ql_i = q_lens[i]
                aux_i, rec_i, y_i = run_jax_mamba_mixer3(
                    proj[start_t:end_t],
                    aux_init,
                    rec_init,
                    B_norm_w,
                    C_norm_w,
                    B_bias,
                    C_bias,
                    D,
                    dt_bias,
                    norm_w,
                    psi,
                    phi,
                    zeta,
                    jnp.array([slot], dtype=jnp.int32),
                    jnp.array([0, ql_i], dtype=jnp.int32),
                    jnp.array([1 if ql_i == 1 else 0, 1, 1], dtype=jnp.int32),
                    jnp.array([int(seq_lens[i])], dtype=jnp.int32),
                    mesh=mesh,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    n_groups=n_groups,
                    d_state=d_state,
                    num_rope_angles=num_rope_angles,
                    chunk_size=chunk_size,
                    is_mimo=is_mimo,
                    mimo_rank=r,
                    apply_rms_norm=not is_mimo,
                    interpret=True,
                )
                np.testing.assert_allclose(
                    y_batch[start_t:end_t], y_i, rtol=1e-4, atol=1e-4
                )
                np.testing.assert_allclose(
                    rec_batch[slot], rec_i[slot], rtol=1e-4, atol=1e-4
                )
                np.testing.assert_allclose(
                    aux_batch[slot], aux_i[slot], rtol=1e-4, atol=1e-4
                )
