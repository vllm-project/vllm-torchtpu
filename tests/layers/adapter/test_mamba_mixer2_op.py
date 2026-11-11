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
"""Unit tests for VllmMambaMixer2 custom op, mamba_op_utils, and JAX Mamba-2 core."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from vllm.model_executor.parameter import BasevLLMParameter

from vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op import (
    _DEFAULT_CHUNK_SIZE,
    VllmMambaMixer2,
)
from vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils import (
    set_param_weight_loader,
)
from vllm_torchtpu.layers.core.mamba_mixer2 import run_jax_mamba_mixer
from vllm_torchtpu.layers.core.mamba_utils import jax_gated_rms_norm
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


def _mesh(tp_size: int = 1):
    return SimpleNamespace(
        shape={"model": tp_size},
        axis_names=("model",),
    )


def _vllm_config(*, speculative_config=None, pcp_size: int = 1):
    return SimpleNamespace(
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            prefill_context_parallel_size=pcp_size,
        ),
    )


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
    layer.kv_cache = (torch.tensor([]), torch.tensor([]))
    layer._apply_py_norm = False

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


def _identity_conv_weight(dim: int, kernel_size: int, dtype=jnp.float32) -> jax.Array:
    """Depthwise weight [dim, 1, K] with 1.0 on the current-token tap (K - 1)."""
    w = jnp.zeros((dim, 1, kernel_size), dtype=dtype)
    return w.at[:, 0, kernel_size - 1].set(1.0)


def _sequential_mamba2_reference(
    j_projected_states: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    j_conv_weight: jax.Array,
    j_conv_bias: jax.Array | None,
    j_A: jax.Array,
    j_D: jax.Array,
    j_dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    has_silu_activation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent per-token NumPy reference for Mamba-2 (conv1d + SSD scan)."""
    tped_intermediate_size = num_heads * head_dim
    local_groups_ssm = n_groups * d_state
    tped_conv_size = tped_intermediate_size + 2 * local_groups_ssm
    group_repeat = num_heads // n_groups

    num_tokens = j_projected_states.shape[0]
    history = conv_state.shape[1]
    num_slots = recurrent_state.shape[0]

    expected_y = np.zeros((num_tokens, tped_intermediate_size), dtype=np.float32)
    expected_conv = np.array(
        conv_state.astype(jnp.float32), dtype=np.float32, copy=True
    )
    expected_rec = np.array(
        recurrent_state.astype(jnp.float32), dtype=np.float32, copy=True
    )

    w_np = np.asarray(j_conv_weight.astype(jnp.float32), dtype=np.float32)
    if w_np.ndim == 3:
        w_kd = np.swapaxes(w_np[:, 0, :], 0, 1)
    else:
        w_kd = np.swapaxes(w_np, 0, 1)
    bias_np = (
        np.asarray(j_conv_bias.astype(jnp.float32), dtype=np.float32)
        if j_conv_bias is not None
        else np.zeros((tped_conv_size,), dtype=np.float32)
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
        valid_slot = 0 <= slot < num_slots
        has_init = (int(s_lens[s]) - q_len) > 0 and valid_slot
        c_win = (
            expected_conv[slot].copy()
            if has_init
            else np.zeros((history, tped_conv_size), dtype=np.float32)
        )
        h = (
            jnp.asarray(expected_rec[slot], dtype=jnp.float32)
            if has_init
            else jnp.zeros((num_heads, head_dim, d_state), dtype=jnp.float32)
        )
        for t in range(q_start, q_end):
            row = j_projected_states[t]
            xbc_in = np.asarray(
                row[
                    tped_intermediate_size : tped_intermediate_size + tped_conv_size
                ].astype(jnp.float32),
                dtype=np.float32,
            )
            taps = np.concatenate([c_win, xbc_in[None, :]], axis=0)
            conv_acc = jnp.asarray(
                np.sum(taps * w_kd, axis=0) + bias_np, dtype=jnp.float32
            )
            if has_silu_activation:
                conv_acc = jax.nn.silu(conv_acc)
            xbc = conv_acc.astype(j_projected_states.dtype)
            c_win = np.concatenate([c_win[1:], xbc_in[None, :]], axis=0)

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
        if valid_slot:
            expected_conv[slot] = c_win
            expected_rec[slot] = np.asarray(h, dtype=np.float32)

    return expected_y, expected_conv, expected_rec


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

    def test_init_rejects_speculative_decoding_and_pcp(self):
        def _super_init_spec(self_obj, *args, **kwargs):
            _fake_mamba_super_init(self_obj, num_spec=3)

        with (
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer2_op."
                "MambaMixer2.__init__",
                side_effect=_super_init_spec,
                autospec=True,
            ),
            pytest.raises(
                NotImplementedError,
                match="Speculative decoding is not yet supported for MambaMixer2",
            ),
        ):
            VllmMambaMixer2(**_MAMBA_INIT_KWARGS)

        layer = _granite_mamba_layer()
        with (
            set_vllm_model_wrapper_context(
                mesh=_mesh(), vllm_config=_vllm_config(speculative_config=object())
            ),
            pytest.raises(
                NotImplementedError,
                match="Speculative decoding is not yet supported for MambaMixer2",
            ),
        ):
            layer._build_mamba_op()

        with (
            set_vllm_model_wrapper_context(
                mesh=_mesh(), vllm_config=_vllm_config(pcp_size=2)
            ),
            pytest.raises(
                NotImplementedError,
                match="Prefill context parallelism",
            ),
        ):
            layer._build_mamba_op()

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
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                return_value=fake_jax_op,
            ) as mock_jax_op,
        ):
            layer._build_mamba_op(interpret=True)

        wrapped_fn = mock_jax_op.call_args.args[1]
        assert wrapped_fn.keywords["has_silu_activation"] is expected_has_silu
        assert wrapped_fn.keywords["chunk_size"] == _DEFAULT_CHUNK_SIZE
        assert wrapped_fn.keywords["interpret"] is True

    def test_build_mamba_op_shares_cached_jax_op_across_identical_layers(self):
        layer0 = _granite_mamba_layer("model.layers.0.mamba")
        layer1 = _granite_mamba_layer("model.layers.1.mamba")
        layer2 = _granite_mamba_layer("model.layers.2.mamba", activation=None)
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

        assert mock_jax_op.call_args_list[0].args[0].startswith("pallas::mamba_mixer2_")
        assert mock_jax_op.call_args_list[0].kwargs["donate_argnums"] == (1, 2)
        assert mock_jax_op.call_args_list[0].args[1].keywords["interpret"] is False

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

    def test_forward_warmup_and_rejects_unified_single_tensor_pool(self):
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

        layer.kv_cache = (torch.zeros((4, 16)),)
        with pytest.raises(NotImplementedError, match="Unified single-tensor"):
            layer.forward(hidden_states)

    @patch("vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.get_forward_context")
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
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
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

        new_conv, new_rec, y_out = run_jax_mamba_mixer(
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
        assert jnp.any(new_conv[0] != 0.0)
        assert jnp.any(new_conv[1] != 0.0)
        assert jnp.all(new_conv[2] == 0.0)
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
        j_conv_weight = _identity_conv_weight(tped_conv_size, kernel_size)
        j_A = -jnp.ones((num_heads,), dtype=jnp.float32)
        j_D = jnp.ones((num_heads,), dtype=jnp.float32)
        j_dt_bias = jnp.zeros((num_heads,), dtype=jnp.float32)
        state_indices = jnp.array([0], dtype=jnp.int32)
        query_start_loc = jnp.array([0, 8], dtype=jnp.int32)
        distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
        seq_lens = jnp.array([8], dtype=jnp.int32)

        _, _, y_with_silu = run_jax_mamba_mixer(
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
        _, _, y_without_silu = run_jax_mamba_mixer(
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
            (4, 4, 2, 8),
            (16, 8, 8, 64),
            (4, 8, 3, 64),
            (4, 4, 0, 8),
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
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        j_projected_states = jax.random.normal(
            k1,
            (num_tokens, tped_intermediate_size + tped_conv_size + tped_dt_size),
            dtype=proj_dtype,
        )
        conv_state = jax.random.normal(
            k2, (num_slots, kernel_size - 1, tped_conv_size), dtype=jnp.float32
        )
        conv_orig = np.array(conv_state, copy=True)
        recurrent_state = jax.random.normal(
            k3, (num_slots, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        recurrent_orig = np.array(recurrent_state, copy=True)
        j_conv_weight = jax.random.normal(
            k4, (tped_conv_size, 1, kernel_size), dtype=proj_dtype
        )
        j_A = -jnp.ones((num_heads,), dtype=jnp.float32)
        j_D = jnp.ones((num_heads,), dtype=jnp.float32)
        j_dt_bias = jnp.array([0.5, -0.3], dtype=jnp.float32)

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
        seq_lens = jnp.array(
            [
                (1 if s == 1 else 5 + s) if s < active_seqs else 0
                for s in range(max_seqs)
            ],
            dtype=jnp.int32,
        )

        new_conv, new_rec, y_out = run_jax_mamba_mixer(
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
            new_conv_prefill, new_rec_prefill, y_prefill = run_jax_mamba_mixer(
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

        expected_y, expected_conv, expected_rec = _sequential_mamba2_reference(
            j_projected_states,
            conv_state,
            recurrent_state,
            j_conv_weight,
            None,
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
                np.asarray(new_conv), np.asarray(new_conv_prefill), rtol=tol, atol=tol
            )
            np.testing.assert_allclose(
                np.asarray(new_rec), np.asarray(new_rec_prefill), rtol=tol, atol=tol
            )

        np.testing.assert_array_equal(
            np.asarray(y_out[active_seqs:].astype(jnp.float32)),
            np.zeros((num_tokens - active_seqs, tped_intermediate_size)),
        )
        active_slots = {s + 1 for s in range(active_seqs)}
        for slot in range(num_slots):
            if slot in active_slots:
                np.testing.assert_allclose(
                    np.asarray(new_conv[slot]), expected_conv[slot], rtol=tol, atol=tol
                )
                np.testing.assert_allclose(
                    np.asarray(new_rec[slot]), expected_rec[slot], rtol=tol, atol=tol
                )
            else:
                np.testing.assert_array_equal(
                    np.asarray(new_conv[slot]), conv_orig[slot]
                )
                np.testing.assert_array_equal(
                    np.asarray(new_rec[slot]), recurrent_orig[slot]
                )

    def test_ragged_mixed_batch_matches_sequential_reference(self):
        devices = np.asarray(jax.devices()[:1]).reshape((1,))
        mesh = Mesh(devices, axis_names=("model",))

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
        k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
        j_projected_states = jax.random.normal(
            k1,
            (num_tokens, tped_intermediate_size + tped_conv_size + tped_dt_size),
            dtype=jnp.float32,
        )
        conv_state = jax.random.normal(
            k2, (num_slots, kernel_size - 1, tped_conv_size), dtype=jnp.float32
        )
        conv_orig = np.array(conv_state, copy=True)
        recurrent_state = jax.random.normal(
            k3, (num_slots, num_heads, head_dim, d_state), dtype=jnp.float32
        )
        recurrent_orig = np.array(recurrent_state, copy=True)

        j_conv_weight = jax.random.normal(
            k4, (tped_conv_size, 1, kernel_size), dtype=jnp.float32
        )
        j_conv_bias = jax.random.normal(k5, (tped_conv_size,), dtype=jnp.float32) * 0.2
        j_A = jnp.array([-0.4, -0.9, -1.2, -0.6], dtype=jnp.float32)
        j_D = jnp.array([1.5, -0.8, 0.3, 2.0], dtype=jnp.float32)
        j_dt_bias = jnp.array([0.4, -0.2, 0.7, -0.5], dtype=jnp.float32)

        # Include slot 0 as an active request while an inactive entry aliases slot 0
        # and another carries -1.
        state_indices = jnp.array([0, 1, 5, 2, 0, -1], dtype=jnp.int32)
        query_start_loc = jnp.array([0, 1, 2, 13, 18, 18, 18], dtype=jnp.int32)
        distribution = jnp.array([2, 2, 4], dtype=jnp.int32)
        seq_lens = jnp.array([9, 1, 19, 5, 0, 0], dtype=jnp.int32)

        new_conv, new_rec, y_out = run_jax_mamba_mixer(
            j_projected_states,
            conv_state,
            recurrent_state,
            j_conv_weight,
            j_conv_bias,
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

        expected_y, expected_conv, expected_rec = _sequential_mamba2_reference(
            j_projected_states,
            conv_state,
            recurrent_state,
            j_conv_weight,
            j_conv_bias,
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
        for slot in (0, 1, 5, 2):
            np.testing.assert_allclose(
                np.asarray(new_conv[slot]), expected_conv[slot], rtol=1e-4, atol=1e-4
            )
            np.testing.assert_allclose(
                np.asarray(new_rec[slot]), expected_rec[slot], rtol=1e-4, atol=1e-4
            )
        for untouched in (3, 4, 6, 7):
            np.testing.assert_array_equal(
                np.asarray(new_conv[untouched]), conv_orig[untouched]
            )
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
        j_conv_weight_2d = jnp.ones((tped_conv_size, kernel_size), dtype=jnp.bfloat16)
        j_A = -jnp.ones((num_heads,), dtype=jnp.float32)
        j_D = jnp.ones((num_heads,), dtype=jnp.float32)
        j_dt_bias = jnp.zeros((num_heads,), dtype=jnp.float32)
        j_norm_weight = jnp.ones((tped_intermediate_size,), dtype=jnp.bfloat16)
        state_indices = jnp.array([0], dtype=jnp.int32)
        query_start_loc = jnp.array([0, 8], dtype=jnp.int32)
        distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
        seq_lens = jnp.array([8], dtype=jnp.int32)

        for apply_rms_norm in (True, False):
            _, _, y_out = run_jax_mamba_mixer(
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
        np.testing.assert_allclose(
            out_grouped[:, :4], out_grouped[:, 4:], rtol=1e-4, atol=1e-4
        )

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

    def test_vllm_mamba_mixer2_state_dtype_process_weights_and_loader_helper(self):
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

        plain_param = torch.nn.Parameter(torch.zeros(4))

        def loader_a(p: torch.Tensor, w: torch.Tensor) -> None:
            p.data.copy_(w)

        def loader_b(p: torch.Tensor, w: torch.Tensor) -> None:
            p.data.copy_(w * 2)

        set_param_weight_loader(plain_param, loader_a)
        set_param_weight_loader(plain_param, loader_b)
        assert plain_param.weight_loader is loader_b

        with (
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
        ):
            vllm_param = BasevLLMParameter(data=torch.zeros(4), weight_loader=loader_a)
        set_param_weight_loader(vllm_param, loader_b)
        assert vllm_param.weight_loader is loader_b

    def test_use_fused_jax_rms_norm_multiprocess_tp(self):
        layer_tp1 = _granite_mamba_layer(tp_size=1, n_groups=1)
        assert layer_tp1._use_fused_jax_rms_norm() is True

        layer_tp2_sharded = _granite_mamba_layer(tp_size=2, n_groups=4)
        assert layer_tp2_sharded._use_fused_jax_rms_norm() is True

        layer_tp2_replicated = _granite_mamba_layer(tp_size=2, n_groups=2)
        layer_tp2_replicated.norm.n_groups = 1
        assert layer_tp2_replicated._use_fused_jax_rms_norm() is False
