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
"""Unit tests for VllmMambaMixer2 custom op and JAX Mamba-2 core."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op import (
    VllmMambaMixer2,
    mamba_mixer2_core_tpu,
)
from vllm_torchtpu.layers.core.mamba_mixer2 import (
    run_jax_mamba_mixer,
)
from vllm_torchtpu.layers.core.ssd_scan import _segsum_jax, ssd_minimal_discrete_hybrid
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


def _granite_mamba_layer(
    prefix: str = "model.layers.0.mamba",
    *,
    tp_size: int = 1,
    n_groups: int = 4,
    bias: bool = True,
):
    layer = VllmMambaMixer2.__new__(VllmMambaMixer2)
    torch.nn.Module.__init__(layer)
    layer.prefix = prefix
    layer.tp_size = tp_size
    layer.num_heads = 32
    layer.head_dim = 32
    layer.ssm_state_size = 64
    layer.conv_kernel_size = 4
    layer.n_groups = n_groups
    layer.intermediate_size = layer.num_heads * layer.head_dim
    layer.groups_ssm_state_size = layer.n_groups * layer.ssm_state_size
    layer.conv_dim = layer.intermediate_size + 2 * layer.groups_ssm_state_size
    layer.tped_intermediate_size = layer.intermediate_size // layer.tp_size
    layer.tped_conv_size = layer.conv_dim // layer.tp_size
    layer.tped_dt_size = layer.num_heads // layer.tp_size

    layer.conv1d = SimpleNamespace(
        weight=torch.randn(layer.tped_conv_size, 1, layer.conv_kernel_size),
        bias=torch.randn(layer.tped_conv_size) if bias else None,
    )
    layer.A = torch.nn.Parameter(-torch.ones(layer.tped_dt_size))
    layer.D = torch.nn.Parameter(torch.ones(layer.tped_dt_size))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(layer.tped_dt_size))
    layer.norm = SimpleNamespace(
        weight=torch.ones(layer.tped_intermediate_size),
        variance_epsilon=1e-5,
        n_groups=n_groups,
    )
    return layer


class TestVllmMambaMixer2:
    def test_init_builds_mamba_op(self):
        mock_op = MagicMock()
        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "MambaMixer2.__init__",
                return_value=None,
            ),
            patch.object(
                VllmMambaMixer2, "_build_mamba_op", return_value=mock_op
            ) as mock_build,
        ):
            layer = VllmMambaMixer2()

        mock_build.assert_called_once_with()
        assert layer.mamba_op is mock_op

    @pytest.mark.parametrize(
        ("enable_prefix_caching", "mamba_cache_mode"),
        [
            (True, "none"),
            (False, "align"),
            (False, "all"),
        ],
    )
    def test_init_rejects_prefix_caching(
        self, enable_prefix_caching: bool, mamba_cache_mode: str
    ):
        def _fake_super_init(self_obj, *args, **kwargs):
            self_obj.cache_config = SimpleNamespace(
                enable_prefix_caching=enable_prefix_caching,
                mamba_cache_mode=mamba_cache_mode,
            )

        with (
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "MambaMixer2.__init__",
                side_effect=_fake_super_init,
                autospec=True,
            ),
            pytest.raises(
                NotImplementedError,
                match="Prefix caching is not yet supported for MambaMixer2",
            ),
        ):
            VllmMambaMixer2(
                hidden_size=512,
                ssm_state_size=64,
                conv_kernel_size=4,
                intermediate_size=1024,
                use_conv_bias=True,
                use_bias=False,
            )

    def test_build_mamba_op_registers_pallas_jax_op_and_fake(self):
        layer0 = _granite_mamba_layer("model.layers.0.mamba")
        layer1 = _granite_mamba_layer("model.layers.1.mamba")
        fake_jax_op = MagicMock()

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op.pallas.jax_op",
                return_value=fake_jax_op,
            ) as mock_jax_op,
        ):
            layer0._build_mamba_op()
            layer1._build_mamba_op()

        assert mock_jax_op.call_count == 2
        assert (
            mock_jax_op.call_args_list[0].args[0]
            == "pallas::mamba_mixer2_model_layers_0_mamba"
        )
        assert (
            mock_jax_op.call_args_list[1].args[0]
            == "pallas::mamba_mixer2_model_layers_1_mamba"
        )
        assert mock_jax_op.call_args_list[0].kwargs["donate_argnums"] == (1, 2)
        assert fake_jax_op.register_fake.call_count == 2

        fake_fn = fake_jax_op.register_fake.call_args_list[0].args[0]
        proj = torch.zeros(
            (
                16,
                layer0.tped_intermediate_size
                + layer0.tped_conv_size
                + layer0.tped_dt_size,
            )
        )
        conv_state = torch.zeros(
            (8, layer0.conv_kernel_size - 1, layer0.tped_conv_size)
        )
        rec_state = torch.zeros(
            (8, layer0.tped_dt_size, layer0.head_dim, layer0.ssm_state_size)
        )
        fake_conv, fake_rec, fake_out = fake_fn(proj, conv_state, rec_state)
        assert fake_conv.shape == conv_state.shape
        assert fake_rec.shape == rec_state.shape
        assert fake_out.shape == (16, layer0.tped_intermediate_size)

    def test_forward_warmup_with_unallocated_kv_cache(self):
        layer = _granite_mamba_layer()
        layer.kv_cache = (torch.tensor([]), torch.tensor([]))
        layer.mamba_op = MagicMock()

        total_proj_dim = (
            layer.tped_intermediate_size + layer.tped_conv_size + layer.tped_dt_size
        )
        layer.in_proj = MagicMock(
            side_effect=lambda x: (torch.ones(x.size(0), total_proj_dim), None)
        )
        layer.norm = MagicMock(side_effect=lambda x, gate: x)
        layer.out_proj = MagicMock(
            side_effect=lambda x: (torch.full((x.size(0), 512), 2.0), None)
        )

        hidden_states = torch.randn(8, 512)
        out = layer.forward(hidden_states)

        layer.mamba_op.assert_not_called()
        layer.norm.assert_called_once()
        assert out.shape == (8, 512)

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op.get_forward_context"
    )
    def test_forward_executes_mamba_op_and_updates_caches_in_place(
        self, mock_get_forward_context
    ):
        layer = _granite_mamba_layer()
        num_tokens = 12
        max_seqs = 4
        conv_state = torch.zeros(
            (max_seqs, layer.conv_kernel_size - 1, layer.tped_conv_size)
        )
        rec_state = torch.zeros(
            (max_seqs, layer.tped_dt_size, layer.head_dim, layer.ssm_state_size)
        )
        layer.kv_cache = (conv_state, rec_state)

        new_conv = torch.ones_like(conv_state) * 3.0
        new_rec = torch.ones_like(rec_state) * 5.0
        expected_ssm_out = torch.ones((num_tokens, layer.tped_intermediate_size)) * 7.0

        fake_jax_op = MagicMock(return_value=(new_conv, new_rec, expected_ssm_out))
        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op.pallas.jax_op",
                return_value=fake_jax_op,
            ),
        ):
            layer.mamba_op = layer._build_mamba_op()

            total_proj_dim = (
                layer.tped_intermediate_size + layer.tped_conv_size + layer.tped_dt_size
            )
            layer.in_proj = MagicMock(
                side_effect=lambda x: (
                    torch.ones(x.size(0), total_proj_dim),
                    None,
                )
            )
            layer.out_proj = MagicMock(side_effect=lambda x: (x[:, :512], None))

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

            out = layer.forward(torch.randn(num_tokens, 512))

        assert torch.allclose(conv_state, new_conv)
        assert torch.allclose(rec_state, new_rec)
        assert torch.allclose(out, torch.ones(num_tokens, 512) * 7.0)

    def test_run_jax_mamba_mixer_e2e_pipeline(self):
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        num_tokens = 16
        max_seqs = 4
        num_heads = 4
        head_dim = 8
        n_groups = 2
        d_state = 8
        kernel_size = 4
        chunk_size = 8
        tped_intermediate_size = num_heads * head_dim
        tped_conv_size = tped_intermediate_size + 2 * n_groups * d_state
        tped_dt_size = num_heads

        j_projected_states = jnp.ones(
            (
                num_tokens,
                tped_intermediate_size + tped_conv_size + tped_dt_size,
            ),
            dtype=jnp.float32,
        )
        conv_state = jnp.zeros(
            (max_seqs, kernel_size - 1, tped_conv_size), dtype=jnp.float32
        )
        recurrent_state = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        j_conv_weight = jnp.ones((tped_conv_size, 1, kernel_size), dtype=jnp.float32)
        j_conv_bias = jnp.zeros((tped_conv_size,), dtype=jnp.float32)
        j_A = -jnp.ones((num_heads,), dtype=jnp.float32)
        j_D = jnp.ones((num_heads,), dtype=jnp.float32)
        j_dt_bias = jnp.zeros((num_heads,), dtype=jnp.float32)
        j_norm_weight = jnp.ones((tped_intermediate_size,), dtype=jnp.float32)
        state_indices = jnp.array([0, 1], dtype=jnp.int32)
        query_start_loc = jnp.array([0, 8, 16], dtype=jnp.int32)
        distribution = jnp.array([0, 2, 2], dtype=jnp.int32)
        seq_lens = jnp.array([8, 8], dtype=jnp.int32)

        def _fake_conv1d(x, c_state, weight, bias, *args, **kwargs):
            return x, c_state + 1.0

        with patch(
            "vllm_torchtpu.layers.core.mamba_mixer2.causal_conv1d.ragged_causal_conv1d",
            side_effect=_fake_conv1d,
        ):
            (new_conv, new_rec), y_out = run_jax_mamba_mixer(
                j_projected_states,
                conv_state,
                recurrent_state,
                j_conv_weight,
                j_conv_bias,
                j_A,
                j_D,
                j_dt_bias,
                j_norm_weight,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                mesh=mesh,
                tped_intermediate_size=tped_intermediate_size,
                tped_conv_size=tped_conv_size,
                tped_dt_size=tped_dt_size,
                num_heads=num_heads,
                head_dim=head_dim,
                n_groups=n_groups,
                d_state=d_state,
                kernel_size=kernel_size,
                chunk_size=chunk_size,
                rms_norm_eps=1e-5,
                apply_rms_norm=True,
            )

        assert new_conv.shape == conv_state.shape
        assert new_rec.shape == recurrent_state.shape
        assert y_out.shape == (num_tokens, tped_intermediate_size)
        assert not jnp.any(jnp.isnan(y_out))
        assert jnp.any(new_rec[0] != 0.0)
        assert jnp.any(new_rec[1] != 0.0)
        assert jnp.all(new_rec[2] == 0.0)

    def test_strided_ldst_load_and_store_roundtrip(self):
        from vllm_torchtpu.kernels.causal_conv1d import strided_ldst

        assert hasattr(strided_ldst, "store_compact_to_large")

        class _FakeVmemRef:
            def __init__(self, arr: np.ndarray):
                self.arr = arr

            @property
            def ndim(self) -> int:
                return self.arr.ndim

            @property
            def shape(self) -> tuple[int, ...]:
                return self.arr.shape

            @property
            def dtype(self):
                return jnp.dtype(self.arr.dtype)

            def __getitem__(self, idx):
                return jnp.asarray(self.arr[idx])

            def __setitem__(self, idx, val):
                self.arr[idx] = np.asarray(val)

        src = np.arange(32, dtype=np.float32).reshape(4, 8)
        src_ref = _FakeVmemRef(src.copy())
        compact = strided_ldst.load_large_to_compact(src_ref, jnp.float32)
        assert compact.shape == (4, 1, 8)

        dst_ref = _FakeVmemRef(np.zeros_like(src))
        strided_ldst.store_compact_to_large(dst_ref, compact * 2.0)
        np.testing.assert_allclose(dst_ref.arr, src * 2.0)

    def test_post_conv1d_silu_activation_applied(self):
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        num_tokens = 8
        max_seqs = 2
        num_heads = 2
        head_dim = 8
        n_groups = 1
        d_state = 8
        kernel_size = 4
        chunk_size = 8
        tped_intermediate_size = num_heads * head_dim
        tped_conv_size = tped_intermediate_size + 2 * n_groups * d_state
        tped_dt_size = num_heads

        j_projected_states = jnp.full(
            (num_tokens, tped_intermediate_size + tped_conv_size + tped_dt_size),
            -1.0,
            dtype=jnp.float32,
        )
        conv_state = jnp.zeros(
            (max_seqs, kernel_size - 1, tped_conv_size), dtype=jnp.float32
        )
        recurrent_state = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        j_conv_weight = jnp.ones((tped_conv_size, 1, kernel_size), dtype=jnp.float32)
        j_A = -jnp.ones((num_heads,), dtype=jnp.float32)
        j_D = jnp.ones((num_heads,), dtype=jnp.float32)
        j_dt_bias = jnp.zeros((num_heads,), dtype=jnp.float32)
        state_indices = jnp.array([0], dtype=jnp.int32)
        query_start_loc = jnp.array([0, 8], dtype=jnp.int32)
        distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
        seq_lens = jnp.array([8], dtype=jnp.int32)

        def _linear_conv1d(x, c_state, weight, bias, *args, **kwargs):
            return x, c_state

        with patch(
            "vllm_torchtpu.layers.core.mamba_mixer2.causal_conv1d.ragged_causal_conv1d",
            side_effect=_linear_conv1d,
        ):
            _, y_with_silu = run_jax_mamba_mixer(
                j_projected_states,
                conv_state,
                recurrent_state,
                j_conv_weight,
                None,
                j_A,
                j_D,
                j_dt_bias,
                None,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                mesh=mesh,
                tped_intermediate_size=tped_intermediate_size,
                tped_conv_size=tped_conv_size,
                tped_dt_size=tped_dt_size,
                num_heads=num_heads,
                head_dim=head_dim,
                n_groups=n_groups,
                d_state=d_state,
                kernel_size=kernel_size,
                chunk_size=chunk_size,
                apply_rms_norm=False,
                has_silu_activation=True,
            )
            _, y_without_silu = run_jax_mamba_mixer(
                j_projected_states,
                conv_state,
                recurrent_state,
                j_conv_weight,
                None,
                j_A,
                j_D,
                j_dt_bias,
                None,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                mesh=mesh,
                tped_intermediate_size=tped_intermediate_size,
                tped_conv_size=tped_conv_size,
                tped_dt_size=tped_dt_size,
                num_heads=num_heads,
                head_dim=head_dim,
                n_groups=n_groups,
                d_state=d_state,
                kernel_size=kernel_size,
                chunk_size=chunk_size,
                apply_rms_norm=False,
                has_silu_activation=False,
            )

        assert not jnp.allclose(y_with_silu, y_without_silu)

    def test_multi_seq_decode_initial_states_and_padded_state_indices(self):
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        num_heads = 2
        head_dim = 8
        n_groups = 1
        d_state = 8
        kernel_size = 4
        tped_intermediate_size = num_heads * head_dim
        tped_conv_size = tped_intermediate_size + 2 * n_groups * d_state
        tped_dt_size = num_heads

        # Case 1: 2 active decode sequences in a 4-seq/4-token batch with chunk_size=8,
        # where inactive seqs 2 and 3 have state_indices == 0 aliasing active slot 0.
        # Case 2: 8 active decode sequences in a 16-token batch with chunk_size=64
        # (num_tokens < chunk_size).
        for num_tokens, max_seqs, active_seqs, chunk_size in [
            (4, 4, 2, 8),
            (16, 8, 8, 64),
        ]:
            rng = jax.random.key(7 + chunk_size)
            k1, k2 = jax.random.split(rng)
            j_projected_states = jax.random.normal(
                k1,
                (num_tokens, tped_intermediate_size + tped_conv_size + tped_dt_size),
                dtype=jnp.float32,
            )
            conv_state = jnp.zeros(
                (max_seqs, kernel_size - 1, tped_conv_size), dtype=jnp.float32
            )
            recurrent_state = jax.random.normal(
                k2, (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
            )
            j_conv_weight = jnp.ones(
                (tped_conv_size, 1, kernel_size), dtype=jnp.float32
            )
            j_A = -jnp.ones((num_heads,), dtype=jnp.float32)
            j_D = jnp.ones((num_heads,), dtype=jnp.float32)
            j_dt_bias = jnp.array([0.5, -0.3], dtype=jnp.float32)

            state_indices = jnp.array(
                [s if s < active_seqs else 0 for s in range(max_seqs)],
                dtype=jnp.int32,
            )
            query_start_loc = jnp.array(
                [min(s, active_seqs) for s in range(max_seqs + 1)],
                dtype=jnp.int32,
            )
            distribution = jnp.array(
                [active_seqs, active_seqs, active_seqs], dtype=jnp.int32
            )
            seq_lens = jnp.array(
                [5 + s if s < active_seqs else 0 for s in range(max_seqs)],
                dtype=jnp.int32,
            )

            def _linear_conv1d(x, c_state, weight, bias, *args, **kwargs):
                return x, c_state

            with patch(
                "vllm_torchtpu.layers.core.mamba_mixer2.causal_conv1d.ragged_causal_conv1d",
                side_effect=_linear_conv1d,
            ):
                (_, new_rec), y_out = run_jax_mamba_mixer(
                    j_projected_states,
                    conv_state,
                    recurrent_state,
                    j_conv_weight,
                    None,
                    j_A,
                    j_D,
                    j_dt_bias,
                    None,
                    state_indices,
                    query_start_loc,
                    distribution,
                    seq_lens,
                    mesh=mesh,
                    tped_intermediate_size=tped_intermediate_size,
                    tped_conv_size=tped_conv_size,
                    tped_dt_size=tped_dt_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    n_groups=n_groups,
                    d_state=d_state,
                    kernel_size=kernel_size,
                    chunk_size=chunk_size,
                    apply_rms_norm=False,
                    has_silu_activation=True,
                )

            assert y_out.shape == (num_tokens, tped_intermediate_size)

            # Exact single-step recurrence reference for all active decode sequences:
            for s in range(active_seqs):
                row = j_projected_states[s]
                xbc = jax.nn.silu(
                    row[
                        tped_intermediate_size : tped_intermediate_size + tped_conv_size
                    ]
                )
                dt_val = jax.nn.softplus(
                    row[tped_intermediate_size + tped_conv_size :] + j_dt_bias
                )
                decay_val = jnp.exp(dt_val * j_A)
                x_s = xbc[:tped_intermediate_size].reshape(num_heads, head_dim)
                b_s = jnp.repeat(
                    xbc[
                        tped_intermediate_size : tped_intermediate_size + d_state
                    ].reshape(1, d_state),
                    num_heads,
                    axis=0,
                )
                c_s = jnp.repeat(
                    xbc[tped_intermediate_size + d_state :].reshape(1, d_state),
                    num_heads,
                    axis=0,
                )
                init_h = recurrent_state[s]
                expected_h = (
                    init_h * decay_val[:, None, None]
                    + (x_s * dt_val[:, None])[:, :, None] * b_s[:, None, :]
                )
                expected_y = (
                    jnp.einsum("hdn,hn->hd", expected_h, c_s) + x_s * j_D[:, None]
                ).reshape(-1)
                np.testing.assert_allclose(new_rec[s], expected_h, rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(y_out[s], expected_y, rtol=1e-4, atol=1e-4)

    def test_prefill_multi_step_dt_discretization_and_unscaled_d_skip(self):
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        num_tokens = 8
        max_seqs = 1
        num_heads = 2
        head_dim = 8
        n_groups = 1
        d_state = 8
        kernel_size = 4
        chunk_size = 4
        tped_intermediate_size = num_heads * head_dim
        tped_conv_size = tped_intermediate_size + 2 * n_groups * d_state
        tped_dt_size = num_heads

        rng = jax.random.key(123)
        j_projected_states = jax.random.normal(
            rng,
            (num_tokens, tped_intermediate_size + tped_conv_size + tped_dt_size),
            dtype=jnp.float32,
        )
        conv_state = jnp.zeros(
            (max_seqs, kernel_size - 1, tped_conv_size), dtype=jnp.float32
        )
        recurrent_state = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        j_conv_weight = jnp.ones((tped_conv_size, 1, kernel_size), dtype=jnp.float32)
        j_A = jnp.array([-0.7, -1.3], dtype=jnp.float32)
        j_D = jnp.array([2.5, -1.5], dtype=jnp.float32)
        j_dt_bias = jnp.array([0.8, -0.4], dtype=jnp.float32)

        state_indices = jnp.array([0], dtype=jnp.int32)
        query_start_loc = jnp.array([0, num_tokens], dtype=jnp.int32)
        distribution = jnp.array([0, 0, 1], dtype=jnp.int32)
        seq_lens = jnp.array([num_tokens], dtype=jnp.int32)

        def _linear_conv1d(x, c_state, weight, bias, *args, **kwargs):
            return x, c_state

        with patch(
            "vllm_torchtpu.layers.core.mamba_mixer2.causal_conv1d.ragged_causal_conv1d",
            side_effect=_linear_conv1d,
        ):
            (_, new_rec), y_out = run_jax_mamba_mixer(
                j_projected_states,
                conv_state,
                recurrent_state,
                j_conv_weight,
                None,
                j_A,
                j_D,
                j_dt_bias,
                None,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                mesh=mesh,
                tped_intermediate_size=tped_intermediate_size,
                tped_conv_size=tped_conv_size,
                tped_dt_size=tped_dt_size,
                num_heads=num_heads,
                head_dim=head_dim,
                n_groups=n_groups,
                d_state=d_state,
                kernel_size=kernel_size,
                chunk_size=chunk_size,
                apply_rms_norm=False,
                has_silu_activation=True,
            )

        # Sequential step-by-step discrete recurrence reference:
        # h_t = exp(dt_t * A) * h_{t-1} + (dt_t * x_t) @ B_t^T
        # y_t = h_t @ C_t + D * x_t
        h = jnp.zeros((num_heads, head_dim, d_state), dtype=jnp.float32)
        expected_ys = []
        for t in range(num_tokens):
            row = j_projected_states[t]
            xbc = jax.nn.silu(
                row[tped_intermediate_size : tped_intermediate_size + tped_conv_size]
            )
            dt_t = jax.nn.softplus(
                row[tped_intermediate_size + tped_conv_size :] + j_dt_bias
            )
            dA_t = jnp.exp(dt_t * j_A)
            x_t = xbc[:tped_intermediate_size].reshape(num_heads, head_dim)
            b_t = jnp.repeat(
                xbc[tped_intermediate_size : tped_intermediate_size + d_state].reshape(
                    1, d_state
                ),
                num_heads,
                axis=0,
            )
            c_t = jnp.repeat(
                xbc[tped_intermediate_size + d_state :].reshape(1, d_state),
                num_heads,
                axis=0,
            )
            h = (
                h * dA_t[:, None, None]
                + (x_t * dt_t[:, None])[:, :, None] * b_t[:, None, :]
            )
            y_t = (jnp.einsum("hdn,hn->hd", h, c_t) + x_t * j_D[:, None]).reshape(-1)
            expected_ys.append(y_t)

        expected_y_all = jnp.stack(expected_ys, axis=0)
        np.testing.assert_allclose(new_rec[0], h, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(y_out, expected_y_all, rtol=1e-4, atol=1e-4)

    def test_bfloat16_output_dtype_and_2d_conv_weight(self):
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        num_tokens = 8
        max_seqs = 2
        num_heads = 2
        head_dim = 8
        n_groups = 2
        d_state = 8
        kernel_size = 4
        chunk_size = 8
        tped_intermediate_size = num_heads * head_dim
        tped_conv_size = tped_intermediate_size + 2 * n_groups * d_state
        tped_dt_size = num_heads

        j_projected_states = jnp.ones(
            (num_tokens, tped_intermediate_size + tped_conv_size + tped_dt_size),
            dtype=jnp.bfloat16,
        )
        conv_state = jnp.zeros(
            (max_seqs, kernel_size - 1, tped_conv_size), dtype=jnp.float32
        )
        recurrent_state = jnp.zeros(
            (max_seqs, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        # Pass 2D conv_weight (dim, kernel_size) to verify rank normalization.
        j_conv_weight_2d = jnp.ones((tped_conv_size, kernel_size), dtype=jnp.bfloat16)
        j_A = -jnp.ones((num_heads,), dtype=jnp.float32)
        j_D = jnp.ones((num_heads,), dtype=jnp.float32)
        j_dt_bias = jnp.zeros((num_heads,), dtype=jnp.float32)
        j_norm_weight = jnp.ones((tped_intermediate_size,), dtype=jnp.bfloat16)
        state_indices = jnp.array([0], dtype=jnp.int32)
        query_start_loc = jnp.array([0, 8], dtype=jnp.int32)
        distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
        seq_lens = jnp.array([8], dtype=jnp.int32)

        def _linear_conv1d(x, c_state, weight, bias, *args, **kwargs):
            assert weight.ndim == 3
            return x, c_state

        with patch(
            "vllm_torchtpu.layers.core.mamba_mixer2.causal_conv1d.ragged_causal_conv1d",
            side_effect=_linear_conv1d,
        ):
            for apply_rms_norm in (True, False):
                (_, _), y_out = run_jax_mamba_mixer(
                    j_projected_states,
                    conv_state,
                    recurrent_state,
                    j_conv_weight_2d,
                    None,
                    j_A,
                    j_D,
                    j_dt_bias,
                    j_norm_weight,
                    state_indices,
                    query_start_loc,
                    distribution,
                    seq_lens,
                    mesh=mesh,
                    tped_intermediate_size=tped_intermediate_size,
                    tped_conv_size=tped_conv_size,
                    tped_dt_size=tped_dt_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    n_groups=n_groups,
                    d_state=d_state,
                    kernel_size=kernel_size,
                    chunk_size=chunk_size,
                    apply_rms_norm=apply_rms_norm,
                )
                assert y_out.dtype == jnp.bfloat16

    def test_jax_gated_rms_norm_per_group_and_weight_none(self):
        from vllm_torchtpu.layers.core.mamba_mixer2 import jax_gated_rms_norm

        # Group 0 has scale 1.0, Group 1 has scale 10.0.
        x = jnp.concatenate(
            [
                jnp.ones((2, 4), dtype=jnp.float32),
                jnp.full((2, 4), 10.0, dtype=jnp.float32),
            ],
            axis=-1,
        )
        gate = jnp.ones((2, 8), dtype=jnp.float32)
        weight = jnp.ones((8,), dtype=jnp.float32)
        eps = 1e-5

        out_grouped = jax_gated_rms_norm(
            x=x,
            gate=gate,
            weight=weight,
            eps=eps,
            n_groups=2,
        )
        # Within each group of constant values, RMS normalization scales every
        # element to ~1.0 regardless of the group's magnitude.
        np.testing.assert_allclose(
            out_grouped[:, :4], out_grouped[:, 4:], rtol=1e-4, atol=1e-4
        )

        # When weight is None (use_rms_norm=False), only SiLU gating is applied.
        out_no_norm = jax_gated_rms_norm(
            x=x.astype(jnp.bfloat16),
            gate=gate.astype(jnp.bfloat16),
            weight=None,
            eps=eps,
            n_groups=2,
        )
        expected_no_norm = (x * jax.nn.silu(gate)).astype(jnp.bfloat16)
        assert out_no_norm.dtype == jnp.bfloat16
        np.testing.assert_allclose(
            np.asarray(out_no_norm, dtype=np.float32),
            np.asarray(expected_no_norm, dtype=np.float32),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_vllm_mamba_mixer2_state_dtype_and_process_weights(self):
        layer = _granite_mamba_layer()
        layer.conv1d.weight = torch.nn.Parameter(
            torch.randn(layer.tped_conv_size, layer.conv_kernel_size)[:, :]
        )
        layer.process_weights_after_loading(torch.bfloat16)
        assert layer.conv1d.weight.shape == (
            layer.tped_conv_size,
            1,
            layer.conv_kernel_size,
        )
        assert layer.conv1d.weight.is_contiguous()

        with patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2.MambaMixer2.get_state_dtype",
            return_value=(torch.bfloat16, torch.float32),
        ):
            assert layer.get_state_dtype() == (torch.float32, torch.float32)

    def test_use_fused_jax_rms_norm_multiprocess_tp(self):
        # TP=1, n_groups=1 -> fused in JAX.
        layer_tp1 = _granite_mamba_layer(tp_size=1, n_groups=1)
        assert layer_tp1._use_fused_jax_rms_norm() is True

        # TP=2, n_groups=4 -> groups divide TP, fused in JAX.
        layer_tp2_sharded = _granite_mamba_layer(tp_size=2, n_groups=4)
        assert layer_tp2_sharded._use_fused_jax_rms_norm() is True

        # TP=2, original n_groups=1 (replicated to self.n_groups=2, norm.n_groups=1)
        # -> requires cross-rank PyTorch all-reduce, not fused in JAX.
        layer_tp2_replicated = _granite_mamba_layer(tp_size=2, n_groups=2)
        layer_tp2_replicated.norm.n_groups = 1
        assert layer_tp2_replicated._use_fused_jax_rms_norm() is False

    def test_mamba_mixer2_core_tpu_updates_only_widened_conv_state_prefix(self):
        kernel_size = 4
        conv_state = jnp.arange(2 * kernel_size * 3, dtype=jnp.float32).reshape(
            2, kernel_size, 3
        )
        rec_state = jnp.zeros((2, 1, 2, 2), dtype=jnp.float32)
        out = jnp.zeros((4, 2), dtype=jnp.float32)

        def _fake_run_jax_mamba_mixer(_proj, c_state_in, r_state, *args, **kwargs):
            return (c_state_in + 10.0, r_state), out

        with patch(
            "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op.run_jax_mamba_mixer",
            side_effect=_fake_run_jax_mamba_mixer,
        ) as mock_run:
            new_conv, _, _ = mamba_mixer2_core_tpu(
                jnp.zeros((4, 6), dtype=jnp.float32),
                conv_state,
                rec_state,
                jnp.zeros((3, 1, kernel_size), dtype=jnp.float32),
                None,
                -jnp.ones((1,), dtype=jnp.float32),
                jnp.ones((1,), dtype=jnp.float32),
                jnp.zeros((1,), dtype=jnp.float32),
                None,
                jnp.array([0, 1], dtype=jnp.int32),
                jnp.array([0, 2, 4], dtype=jnp.int32),
                jnp.array([0, 2, 2], dtype=jnp.int32),
                jnp.array([2, 2], dtype=jnp.int32),
                mesh=_mesh(),
                tped_intermediate_size=2,
                tped_conv_size=3,
                tped_dt_size=1,
                num_heads=1,
                head_dim=2,
                n_groups=1,
                d_state=2,
                kernel_size=kernel_size,
                chunk_size=4,
                rms_norm_eps=1e-5,
                apply_rms_norm=False,
            )

        kernel_conv_state = mock_run.call_args.args[1]
        assert kernel_conv_state.shape == (2, kernel_size - 1, 3)
        np.testing.assert_array_equal(
            np.asarray(kernel_conv_state), np.asarray(conv_state[:, :-1, :])
        )
        np.testing.assert_array_equal(
            np.asarray(new_conv[:, :-1, :]), np.asarray(conv_state[:, :-1, :] + 10.0)
        )
        np.testing.assert_array_equal(
            np.asarray(new_conv[:, -1:, :]), np.asarray(conv_state[:, -1:, :])
        )

    def test_ssd_minimal_discrete_hybrid_matches_jax_reference(self):
        batch, seqlen, block_len, n_heads, d_head, d_state = 2, 16, 4, 4, 8, 8
        n_chunks_per_seq = seqlen // block_len
        n_chunks = batch * n_chunks_per_seq

        rng = jax.random.key(42)
        k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

        X_c = jax.random.normal(k1, (n_chunks, n_heads, block_len, d_head))
        A_c = -jnp.exp(jax.random.normal(k2, (n_chunks, n_heads, block_len)))
        B_c = jax.random.normal(k3, (n_chunks, n_heads, block_len, d_state))
        C_c = jax.random.normal(k4, (n_chunks, n_heads, block_len, d_state))
        init_states_per_batch = jax.random.normal(k5, (batch, n_heads, d_head, d_state))

        initial_states = jnp.zeros((n_chunks, n_heads, d_head, d_state))
        reset_mask = np.zeros((n_chunks,), dtype=bool)
        for b in range(batch):
            initial_states = initial_states.at[b * n_chunks_per_seq].set(
                init_states_per_batch[b]
            )
            reset_mask[b * n_chunks_per_seq] = True
        reset_mask = jnp.array(reset_mask)
        last_chunk_indices = jnp.array(
            [b * n_chunks_per_seq + n_chunks_per_seq - 1 for b in range(batch)],
            dtype=jnp.int32,
        )

        # Reference math
        A_cumsum = jnp.cumsum(A_c, axis=-1)
        L = jnp.exp(_segsum_jax(A_c))
        Y_diag = jnp.einsum(
            "chln,chsn,chls,chsp->chlp",
            C_c,
            B_c,
            L,
            X_c,
            precision=jax.lax.Precision.HIGHEST,
        )
        decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
        states_c_ref = jnp.einsum(
            "chln,chlp,chl->chpn",
            B_c,
            X_c,
            decay_states,
            precision=jax.lax.Precision.HIGHEST,
        )
        decay_chunk = jnp.exp(A_cumsum[..., -1])

        def scan_body(carry, args):
            s_c, d_c, init_s, r_c = args
            inc = jnp.where(r_c, init_s, carry)
            nxt = inc * d_c[:, None, None] + s_c
            return nxt, (inc, nxt)

        _, (states_inc_ref, states_out_ref) = jax.lax.scan(
            scan_body,
            jnp.zeros((n_heads, d_head, d_state), dtype=X_c.dtype),
            (states_c_ref, decay_chunk, initial_states, reset_mask),
        )
        Y_off = jnp.einsum(
            "chln,chpn,chl->chlp",
            C_c,
            states_inc_ref,
            jnp.exp(A_cumsum),
            precision=jax.lax.Precision.HIGHEST,
        )
        y_ref = Y_diag + Y_off
        final_state_ref = states_out_ref[last_chunk_indices]

        y_hybrid, final_state_hybrid = ssd_minimal_discrete_hybrid(
            X_c,
            A_c,
            B_c,
            C_c,
            last_chunk_indices,
            initial_states,
            reset_mask,
        )

        np.testing.assert_allclose(y_hybrid, y_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            final_state_hybrid, final_state_ref, rtol=1e-4, atol=1e-4
        )
