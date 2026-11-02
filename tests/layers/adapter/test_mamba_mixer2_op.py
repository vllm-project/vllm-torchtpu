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
    _DEFAULT_CHUNK_SIZE,
    VllmMambaMixer2,
)
from vllm_torchtpu.layers.core.mamba_mixer2 import run_jax_mamba_mixer
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


def _mesh(tp_size: int = 1):
    return SimpleNamespace(
        shape={"model": tp_size},
        axis_names=("model",),
    )


def _vllm_config():
    return SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))


def _fake_mamba_super_init(
    self_obj,
    *,
    cache_config=None,
    num_spec: int = 0,
    use_replayssm: bool = False,
    activation: str | None = "silu",
):
    self_obj.cache_config = cache_config
    self_obj.num_spec = num_spec
    self_obj.use_replayssm = use_replayssm
    self_obj.activation = activation


def _granite_mamba_layer(
    prefix: str = "model.layers.0.mamba",
    *,
    tp_size: int = 1,
    n_groups: int = 4,
    bias: bool = True,
    activation: str | None = "silu",
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
    layer.cache_config = None
    layer.num_spec = 0
    layer.use_replayssm = False
    layer.activation = activation
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


def _sequential_mamba2_reference(
    j_projected_states: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_A: jnp.ndarray,
    j_D: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    has_silu_activation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact step-by-step discrete Mamba-2 SSM reference (post-identity conv1d)."""
    tped_intermediate_size = num_heads * head_dim
    local_groups_ssm = n_groups * d_state
    tped_conv_size = tped_intermediate_size + 2 * local_groups_ssm
    group_repeat = num_heads // n_groups

    num_tokens = j_projected_states.shape[0]
    expected_y = np.zeros((num_tokens, tped_intermediate_size), dtype=np.float32)
    expected_rec = np.array(
        recurrent_state.astype(jnp.float32), dtype=np.float32, copy=True
    )

    num_valid_seqs = int(np.asarray(distribution)[2])
    qsl = np.asarray(query_start_loc)
    slots = np.asarray(state_indices)
    s_lens = np.asarray(seq_lens)

    for s in range(num_valid_seqs):
        q_start = int(qsl[s])
        q_end = int(qsl[s + 1])
        q_len = q_end - q_start
        if q_len <= 0:
            continue
        slot = int(slots[s])
        has_init = (int(s_lens[s]) - q_len) > 0
        h = (
            jnp.asarray(expected_rec[slot], dtype=jnp.float32)
            if has_init
            else jnp.zeros((num_heads, head_dim, d_state), dtype=jnp.float32)
        )
        for t in range(q_start, q_end):
            row = j_projected_states[t]
            xbc = row[tped_intermediate_size : tped_intermediate_size + tped_conv_size]
            if has_silu_activation:
                xbc = jax.nn.silu(xbc)
            dt_t = jax.nn.softplus(
                row[tped_intermediate_size + tped_conv_size :].astype(jnp.float32)
                + j_dt_bias.astype(jnp.float32)
            )
            dA_t = jnp.exp(dt_t * j_A.astype(jnp.float32))
            x_t = xbc[:tped_intermediate_size].reshape(num_heads, head_dim)
            b_t = jnp.repeat(
                xbc[
                    tped_intermediate_size : tped_intermediate_size + local_groups_ssm
                ].reshape(n_groups, d_state),
                group_repeat,
                axis=0,
            )
            c_t = jnp.repeat(
                xbc[tped_intermediate_size + local_groups_ssm :].reshape(
                    n_groups, d_state
                ),
                group_repeat,
                axis=0,
            )
            x_f32 = x_t.astype(jnp.float32)
            h = (
                h * dA_t[:, None, None]
                + (x_f32 * dt_t[:, None])[:, :, None]
                * b_t.astype(jnp.float32)[:, None, :]
            )
            y_t = (
                jnp.einsum(
                    "hdn,hn->hd",
                    h,
                    c_t.astype(jnp.float32),
                    precision=jax.lax.Precision.HIGHEST,
                )
                + x_f32 * j_D.astype(jnp.float32)[:, None]
            ).reshape(-1)
            expected_y[t] = np.asarray(y_t, dtype=np.float32)
        expected_rec[slot] = np.asarray(h, dtype=np.float32)

    return expected_y, expected_rec


_MAMBA_INIT_KWARGS = {
    "hidden_size": 512,
    "ssm_state_size": 64,
    "conv_kernel_size": 4,
    "intermediate_size": 1024,
    "use_conv_bias": True,
    "use_bias": False,
}


class TestVllmMambaMixer2:
    def test_init_builds_mamba_op(self):
        mock_op = MagicMock()

        def _super_init(self_obj, *args, **kwargs):
            _fake_mamba_super_init(self_obj)

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "MambaMixer2.__init__",
                side_effect=_super_init,
                autospec=True,
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "is_conv_state_dim_first",
                return_value=False,
            ),
            patch.object(
                VllmMambaMixer2, "_build_mamba_op", return_value=mock_op
            ) as mock_build,
        ):
            layer = VllmMambaMixer2(**_MAMBA_INIT_KWARGS)

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
        def _super_init(self_obj, *args, **kwargs):
            _fake_mamba_super_init(
                self_obj,
                cache_config=SimpleNamespace(
                    enable_prefix_caching=enable_prefix_caching,
                    mamba_cache_mode=mamba_cache_mode,
                ),
            )

        with (
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "MambaMixer2.__init__",
                side_effect=_super_init,
                autospec=True,
            ),
            pytest.raises(
                NotImplementedError,
                match="Prefix caching is not yet supported for MambaMixer2",
            ),
        ):
            VllmMambaMixer2(**_MAMBA_INIT_KWARGS)

    def test_init_rejects_speculative_decoding(self):
        def _super_init(self_obj, *args, **kwargs):
            _fake_mamba_super_init(self_obj, num_spec=3)

        with (
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "MambaMixer2.__init__",
                side_effect=_super_init,
                autospec=True,
            ),
            pytest.raises(
                NotImplementedError,
                match="Speculative decoding is not yet supported for MambaMixer2",
            ),
        ):
            VllmMambaMixer2(**_MAMBA_INIT_KWARGS)

    def test_init_rejects_replayssm(self):
        def _super_init(self_obj, *args, **kwargs):
            _fake_mamba_super_init(self_obj, use_replayssm=True)

        with (
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "MambaMixer2.__init__",
                side_effect=_super_init,
                autospec=True,
            ),
            pytest.raises(
                NotImplementedError,
                match="--use-replayssm is not supported for MambaMixer2",
            ),
        ):
            VllmMambaMixer2(**_MAMBA_INIT_KWARGS)

    def test_init_rejects_dim_first_conv_state_layout(self):
        def _super_init(self_obj, *args, **kwargs):
            _fake_mamba_super_init(self_obj)

        with (
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "MambaMixer2.__init__",
                side_effect=_super_init,
                autospec=True,
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "is_conv_state_dim_first",
                return_value=True,
            ),
            pytest.raises(
                NotImplementedError,
                match="VLLM_SSM_CONV_STATE_LAYOUT=DS is not supported",
            ),
        ):
            VllmMambaMixer2(**_MAMBA_INIT_KWARGS)

    def test_init_rejects_unsupported_activation(self):
        def _super_init(self_obj, *args, **kwargs):
            _fake_mamba_super_init(self_obj, activation="gelu")

        with (
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "MambaMixer2.__init__",
                side_effect=_super_init,
                autospec=True,
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "is_conv_state_dim_first",
                return_value=False,
            ),
            pytest.raises(
                NotImplementedError,
                match="Unsupported MambaMixer2 activation 'gelu'",
            ),
        ):
            VllmMambaMixer2(**_MAMBA_INIT_KWARGS)

    @pytest.mark.parametrize(
        ("activation", "expected_has_silu"),
        [
            (None, False),
            ("silu", True),
            ("swish", True),
        ],
    )
    def test_build_mamba_op_wires_activation_chunk_size_and_interpret(
        self, activation: str | None, expected_has_silu: bool
    ):
        layer = _granite_mamba_layer("model.layers.0.mamba", activation=activation)
        fake_jax_op = MagicMock()

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op.pallas.jax_op",
                return_value=fake_jax_op,
            ) as mock_jax_op,
        ):
            layer._build_mamba_op(interpret=True)

        wrapped_fn = mock_jax_op.call_args.args[1]
        assert wrapped_fn.keywords["has_silu_activation"] is expected_has_silu
        assert wrapped_fn.keywords["chunk_size"] == _DEFAULT_CHUNK_SIZE
        assert wrapped_fn.keywords["interpret"] is True

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
        assert mock_jax_op.call_args_list[0].args[1].keywords["interpret"] is False
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
            layer.mamba_op = layer._build_mamba_op(interpret=True)

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
                interpret=True,
            )

        assert new_conv.shape == conv_state.shape
        assert new_rec.shape == recurrent_state.shape
        assert y_out.shape == (num_tokens, tped_intermediate_size)
        assert not jnp.any(jnp.isnan(y_out))
        assert jnp.any(new_rec[0] != 0.0)
        assert jnp.any(new_rec[1] != 0.0)
        assert jnp.all(new_rec[2] == 0.0)

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
                interpret=True,
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
                interpret=True,
            )

        assert not jnp.allclose(y_with_silu, y_without_silu)

    @pytest.mark.parametrize("proj_dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize(
        ("num_tokens", "max_seqs", "active_seqs", "chunk_size"),
        [
            (4, 4, 2, 8),  # active_seqs < max_seqs, inactive slots alias slot 0
            (16, 8, 8, 64),  # num_tokens > max_seqs (compile-bucket padding)
            (4, 8, 3, 64),  # num_tokens < max_seqs
            (4, 4, 0, 8),  # R == 0 idle DP dummy batch
        ],
    )
    def test_multi_seq_decode_initial_states_and_padded_state_indices(
        self,
        proj_dtype,
        num_tokens: int,
        max_seqs: int,
        active_seqs: int,
        chunk_size: int,
    ):
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        num_heads = 2
        head_dim = 8
        n_groups = 1
        d_state = 8
        kernel_size = 4
        num_slots = max_seqs + 2
        tped_intermediate_size = num_heads * head_dim
        tped_conv_size = tped_intermediate_size + 2 * n_groups * d_state
        tped_dt_size = num_heads

        rng = jax.random.key(7 + chunk_size + active_seqs + num_tokens)
        k1, k2 = jax.random.split(rng)
        j_projected_states = jax.random.normal(
            k1,
            (num_tokens, tped_intermediate_size + tped_conv_size + tped_dt_size),
            dtype=proj_dtype,
        )
        conv_state = jnp.zeros(
            (num_slots, kernel_size - 1, tped_conv_size), dtype=jnp.float32
        )
        recurrent_state = jax.random.normal(
            k2, (num_slots, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        recurrent_orig = np.array(recurrent_state, copy=True)
        j_conv_weight = jnp.ones((tped_conv_size, 1, kernel_size), dtype=jnp.float32)
        j_A = -jnp.ones((num_heads,), dtype=jnp.float32)
        j_D = jnp.ones((num_heads,), dtype=jnp.float32)
        j_dt_bias = jnp.array([0.5, -0.3], dtype=jnp.float32)

        # Inactive sequences alias slot 0 or carry -1 sentinels.
        state_indices = jnp.array(
            [
                s + 1 if s < active_seqs else (0 if s % 2 == 0 else -1)
                for s in range(max_seqs)
            ],
            dtype=jnp.int32,
        )
        query_start_loc = jnp.array(
            [min(s, active_seqs) for s in range(max_seqs + 1)],
            dtype=jnp.int32,
        )
        distribution = jnp.array(
            [active_seqs, active_seqs, active_seqs], dtype=jnp.int32
        )
        # Mix of continuation decode (seq_len > 1) and 1-token fresh prompt
        # (seq_len == 1).
        seq_lens = jnp.array(
            [
                (1 if s == 1 else 5 + s) if s < active_seqs else 0
                for s in range(max_seqs)
            ],
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
                interpret=True,
            )
            if active_seqs > 0:
                prefill_distribution = jnp.array([0, 0, active_seqs], dtype=jnp.int32)
                (_, new_rec_prefill), y_prefill = run_jax_mamba_mixer(
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
                    prefill_distribution,
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
                    interpret=True,
                )

        assert y_out.shape == (num_tokens, tped_intermediate_size)
        assert y_out.dtype == proj_dtype

        expected_y, expected_rec = _sequential_mamba2_reference(
            j_projected_states,
            recurrent_state,
            j_A,
            j_D,
            j_dt_bias,
            state_indices,
            query_start_loc,
            distribution,
            seq_lens,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
        )
        tol = 2e-2 if proj_dtype == jnp.bfloat16 else 1e-4
        if active_seqs > 0:
            np.testing.assert_allclose(
                np.asarray(y_out[:active_seqs].astype(jnp.float32)),
                expected_y[:active_seqs],
                rtol=tol,
                atol=tol,
            )
            np.testing.assert_allclose(
                np.asarray(y_out[:active_seqs].astype(jnp.float32)),
                np.asarray(y_prefill[:active_seqs].astype(jnp.float32)),
                rtol=tol,
                atol=tol,
            )
            np.testing.assert_allclose(
                np.asarray(new_rec),
                np.asarray(new_rec_prefill),
                rtol=tol,
                atol=tol,
            )
        # Inactive / padded decode rows must be zero-masked.
        np.testing.assert_array_equal(
            np.asarray(y_out[active_seqs:].astype(jnp.float32)),
            np.zeros((num_tokens - active_seqs, tped_intermediate_size)),
        )
        active_slots = {s + 1 for s in range(active_seqs)}
        for slot in range(num_slots):
            if slot in active_slots:
                np.testing.assert_allclose(
                    np.asarray(new_rec[slot]),
                    expected_rec[slot],
                    rtol=1e-4,
                    atol=1e-4,
                )
            else:
                np.testing.assert_array_equal(
                    np.asarray(new_rec[slot]), recurrent_orig[slot]
                )

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
                interpret=True,
            )

        expected_y, expected_rec = _sequential_mamba2_reference(
            j_projected_states,
            recurrent_state,
            j_A,
            j_D,
            j_dt_bias,
            state_indices,
            query_start_loc,
            distribution,
            seq_lens,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
        )
        np.testing.assert_allclose(new_rec[0], expected_rec[0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(y_out, expected_y, rtol=1e-4, atol=1e-4)

    def test_ragged_mixed_batch_matches_sequential_reference(self):
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

        # 2 decodes (len 1 w/ init, len 1 w/o init) + 2 non-multiple prefills
        # (len 11 continuation w/ init across 2 chunks, len 5 fresh w/o init)
        # = 18 valid tokens inside a 24-token padded batch, chunk_size = 8.
        num_tokens = 24
        num_slots = 8
        num_heads = 4
        head_dim = 8
        n_groups = 2
        d_state = 8
        kernel_size = 4
        chunk_size = 8
        tped_intermediate_size = num_heads * head_dim
        tped_conv_size = tped_intermediate_size + 2 * n_groups * d_state
        tped_dt_size = num_heads

        rng = jax.random.key(987)
        k1, k2 = jax.random.split(rng)
        j_projected_states = jax.random.normal(
            k1,
            (num_tokens, tped_intermediate_size + tped_conv_size + tped_dt_size),
            dtype=jnp.float32,
        )
        conv_state = jnp.zeros(
            (num_slots, kernel_size - 1, tped_conv_size), dtype=jnp.float32
        )
        recurrent_state = jax.random.normal(
            k2, (num_slots, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        recurrent_orig = np.array(recurrent_state, copy=True)

        j_conv_weight = jnp.ones((tped_conv_size, 1, kernel_size), dtype=jnp.float32)
        j_A = jnp.array([-0.4, -0.9, -1.2, -0.6], dtype=jnp.float32)
        j_D = jnp.array([1.5, -0.8, 0.3, 2.0], dtype=jnp.float32)
        j_dt_bias = jnp.array([0.4, -0.2, 0.7, -0.5], dtype=jnp.float32)

        # Active slots {3, 1, 5, 2}; inactive entries alias slot 3 and sentinel -1.
        state_indices = jnp.array([3, 1, 5, 2, 3, -1], dtype=jnp.int32)
        query_start_loc = jnp.array([0, 1, 2, 13, 18, 18, 18], dtype=jnp.int32)
        distribution = jnp.array([2, 2, 4], dtype=jnp.int32)
        seq_lens = jnp.array([9, 1, 19, 5, 0, 0], dtype=jnp.int32)

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
                interpret=True,
            )

        expected_y, expected_rec = _sequential_mamba2_reference(
            j_projected_states,
            recurrent_state,
            j_A,
            j_D,
            j_dt_bias,
            state_indices,
            query_start_loc,
            distribution,
            seq_lens,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            d_state=d_state,
        )
        np.testing.assert_allclose(
            np.asarray(y_out[:18]), expected_y[:18], rtol=1e-4, atol=1e-4
        )
        for slot in (3, 1, 5, 2):
            np.testing.assert_allclose(
                np.asarray(new_rec[slot]),
                expected_rec[slot],
                rtol=1e-4,
                atol=1e-4,
            )
        for untouched in (0, 4, 6, 7):
            np.testing.assert_array_equal(
                np.asarray(new_rec[untouched]), recurrent_orig[untouched]
            )

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
                    interpret=True,
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
