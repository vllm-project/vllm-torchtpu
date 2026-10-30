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
"""Unit tests for VllmMambaMixer3 custom op and JAX Mamba-3 core."""

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


def _mesh(tp_size: int = 1):
    return SimpleNamespace(
        shape={"attn_dp": 1, "expert": 1, "model": tp_size},
        axis_names=("attn_dp", "expert", "model"),
    )


def _vllm_config(*, dp_size: int = 1):
    return SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=dp_size))


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


class TestVllmMambaMixer3:
    def test_init_builds_mamba3_op(self):
        mock_op = MagicMock()
        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch.object(MambaMixer3, "__init__", return_value=None),
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
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.pallas.jax_op",
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
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.pallas.jax_op",
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
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.pallas.jax_op",
                side_effect=_passthrough_jax_op,
            ),
        ):
            layer_fused.mamba_op = layer_fused._build_mamba_op()
            with patch.object(layer_py, "_use_fused_jax_rms_norm", return_value=False):
                layer_py.mamba_op = layer_py._build_mamba_op()

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
            "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_forward_context",
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
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.pallas.jax_op",
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
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_forward_context",
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

    def test_forward_warmup_with_unallocated_kv_cache(self):
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

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.get_forward_context"
    )
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
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op.pallas.jax_op",
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

        # 1. Single contiguous prefill over all `total_len` tokens
        aux_contig = jnp.zeros((max_seqs, num_heads, aux_dim), dtype=jnp.float32)
        rec_contig = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        (aux_contig_out, rec_contig_out), y_contig = run_jax_mamba_mixer3(
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
        )

        # 2. Split execution: prefill first `prefill_len` tokens, then decode
        #    remaining tokens one by one (with inactive seq 1 aliasing slot 0)
        aux_step = jnp.zeros((max_seqs, num_heads, aux_dim), dtype=jnp.float32)
        rec_step = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )

        (aux_step, rec_step), y_prefill = run_jax_mamba_mixer3(
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
        )

        y_decodes = []
        for step_idx in range(prefill_len, total_len):
            cur_seq_len = step_idx + 1
            (aux_step, rec_step), y_dec = run_jax_mamba_mixer3(
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
        num_rope_angles = d_state // 4  # rope_fraction = 0.5
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
        (aux_contig_out, rec_contig_out), y_contig = run_jax_mamba_mixer3(
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
            )
            assert y_out.dtype == jnp.bfloat16
            assert new_aux.shape == aux_state.shape
            assert new_rec.shape == recurrent_state.shape

    def test_multi_request_decode_batch_with_padded_tokens_siso_and_mimo(self):
        """Verifies pure decode fast-path across multiple requests with padding."""
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        num_active = 4
        num_tokens = 12  # padded token bucket > num_active
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

            state_indices = jnp.array([2, 0, 5, 3, 0, 0], dtype=jnp.int32)
            query_start_loc = jnp.array([0, 1, 2, 3, 4, 4, 4], dtype=jnp.int32)
            distribution = jnp.array(
                [num_active, num_active, num_active], dtype=jnp.int32
            )
            seq_lens = jnp.array([10, 1, 25, 4, 0, 0], dtype=jnp.int32)

            (aux_batch, rec_batch), y_batch = run_jax_mamba_mixer3(
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
            )

            for i in range(num_active):
                slot = int(state_indices[i])
                (aux_i, rec_i), y_i = run_jax_mamba_mixer3(
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
        total_tokens = sum(q_lens) + 4  # include 4 trailing padded tokens
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

            (aux_batch, rec_batch), y_batch = run_jax_mamba_mixer3(
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
            )

            for i in range(num_active):
                slot = int(state_indices[i])
                start_t = qsl[i]
                end_t = qsl[i + 1]
                ql_i = q_lens[i]
                (aux_i, rec_i), y_i = run_jax_mamba_mixer3(
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
