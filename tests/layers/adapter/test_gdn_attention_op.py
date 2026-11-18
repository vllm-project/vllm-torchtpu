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

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import vllm.envs as vllm_envs
from jax.sharding import PartitionSpec
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.block_major_pool import BlockMajorPoolLayout, flat_manager_ids
from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op
from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
    VllmGatedDeltaNetAttention,
    gdn_attention_core_tpu_pcp_prefill,
)
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


@pytest.fixture
def cpu_vllm_config_context():
    # These adapter tests use CPU tensors and mocked kernels. They must not
    # require registration of torch's TPU device type just to resolve layout.
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _mesh():
    return SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})


def _vllm_config(
    *,
    pcp_size: int = 1,
    tp_size: int = 1,
    interleave_size: int = 16,
    block_size: int = 16,
    mamba_page_size_padded: int | None = None,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            tensor_parallel_size=tp_size,
            cp_kv_cache_interleave_size=interleave_size,
            data_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            mamba_block_size=4096,
            mamba_page_size_padded=mamba_page_size_padded,
            mamba_cache_mode=None,
            use_kda_recoverssm=False,
            use_replayssm=False,
        ),
        speculative_config=None,
        num_speculative_tokens=0,
    )


def _qwen35_397b_gdn_attn(prefix: str, *, bias: bool = False, tp_size: int = 1):
    attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
    attn.prefix = prefix
    attn.num_k_heads = 64
    attn.gqa_interleaved_layout = False
    attn.num_v_heads = 64
    attn.tp_size = tp_size
    attn.head_k_dim = 128
    attn.head_v_dim = 128
    attn.conv_kernel_size = 4
    attn.num_spec = 0
    attn.conv1d = SimpleNamespace(
        bias=torch.randn(64) if bias else None,
    )
    attn.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        architecture="Qwen3_5MoeForCausalLM",
    )
    attn.cache_config = SimpleNamespace(
        mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="float32"
    )
    return attn


def _qwen38_gdn_attn(prefix: str, *, tp_size: int = 1):
    attn = _qwen35_397b_gdn_attn(prefix, tp_size=tp_size)
    attn.num_k_heads = 4
    attn.num_v_heads = 32
    return attn


class TestVllmGatedDeltaNetAttention:
    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
        "run_jax_gdn_attention_pcp_tp_prefill"
    )
    def test_pcp_core_updates_only_widened_conv_state_prefix(self, mock_run):
        kernel_size = 4
        conv_state = jnp.arange(2 * kernel_size * 3, dtype=jnp.float32).reshape(
            2, kernel_size, 3
        )
        recurrent_state = jnp.zeros((2, 1, 1, 1), dtype=jnp.float32)
        new_prefix = jnp.full((2, kernel_size - 1, 3), 77, dtype=jnp.float32)
        new_recurrent = jnp.ones_like(recurrent_state)
        output = jnp.zeros((5, 1, 1), dtype=jnp.float32)
        mock_run.return_value = (new_prefix, new_recurrent), output

        new_conv, returned_recurrent, returned_output = (
            gdn_attention_core_tpu_pcp_prefill(
                *(jnp.zeros((1,), dtype=jnp.float32) for _ in range(3)),
                conv_state,
                recurrent_state,
                *(jnp.zeros((1,), dtype=jnp.float32) for _ in range(8)),
                mesh=MagicMock(),
                n_kq=1,
                n_v=1,
                d_k=1,
                d_v=1,
                kernel_size=kernel_size,
                pcp_size=2,
                interleave_size=1,
            )
        )

        kernel_conv_state = mock_run.call_args.args[3]
        assert kernel_conv_state.shape == (2, kernel_size - 1, 3)
        np.testing.assert_array_equal(
            np.asarray(kernel_conv_state), np.asarray(conv_state[:, :-1, :])
        )
        np.testing.assert_array_equal(np.asarray(new_conv[:, :-1, :]), 77)
        np.testing.assert_array_equal(
            np.asarray(new_conv[:, -1:, :]), np.asarray(conv_state[:, -1:, :])
        )
        np.testing.assert_array_equal(
            np.asarray(returned_recurrent), np.asarray(new_recurrent)
        )
        np.testing.assert_array_equal(np.asarray(returned_output), np.asarray(output))

    @pytest.mark.parametrize("interleaved", [False, True])
    @pytest.mark.parametrize(
        ("ssm_cache_dtype", "expected_dtype"),
        [
            ("float32", jnp.float32),
            ("bfloat16", jnp.bfloat16),
        ],
    )
    @pytest.mark.parametrize("projection_dtype", [torch.bfloat16, torch.float8_e4m3fn])
    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_pcp_world_size",
        return_value=8,
    )
    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_or_create_pcp_mesh"
    )
    def test_pooled_pcp_impl_uses_donation_and_copy_writeback(
        self,
        mock_get_pcp_mesh,
        _mock_get_pcp_world_size,
        ssm_cache_dtype,
        expected_dtype,
        projection_dtype,
        interleaved,
        cpu_vllm_config_context,
    ):
        attn = _qwen35_397b_gdn_attn("copy_test_layer")
        attn.gqa_interleaved_layout = interleaved
        attn.cache_config.mamba_ssm_cache_dtype = ssm_cache_dtype
        mock_get_pcp_mesh.return_value = SimpleNamespace(shape={"pcp": 8, "tp": 1})
        pool = torch.zeros((4, 8), dtype=torch.float32)
        pool_alias = pool.view_as(pool)
        original_storage = pool.untyped_storage().data_ptr()

        def fake_op(*_args, **_kwargs):
            output = torch.ones((2, 64, 128), dtype=torch.float32)
            return torch.full_like(pool, 7), output, output + 1

        fake_jax_op = MagicMock(side_effect=fake_op)
        with (
            set_vllm_model_wrapper_context(
                mesh=_mesh(), vllm_config=_vllm_config(pcp_size=8)
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pallas.jax_op",
                return_value=fake_jax_op,
            ) as mock_pcp_jax_op,
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "run_jax_gdn_attention_pooled_pcp_prefill_projection",
            ) as mock_run,
        ):
            mock_run.return_value = (MagicMock(), MagicMock(), MagicMock())
            pooled_pcp_impl = attn._build_pooled_pcp_gdn_op()
            mock_pcp_jax_op.call_args.args[1](*([MagicMock()] * 14))

        weight_scale = (
            torch.ones(192) if projection_dtype == torch.float8_e4m3fn else None
        )
        output, z = pooled_pcp_impl(
            torch.zeros((2, 64)),
            torch.zeros((192, 64), dtype=projection_dtype),
            weight_scale,
            torch.zeros((2, 64)),
            torch.zeros((2, 64)),
            pool,
            torch.zeros((64, 1, 4)),
            None,
            torch.zeros(64),
            torch.zeros(64),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(3, dtype=torch.int32),
            torch.zeros(3, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )

        assert mock_pcp_jax_op.call_args.kwargs["donate_argnums"] == (4,)
        assert mock_run.call_args.kwargs["recurrent_state_dtype"] == jnp.dtype(
            expected_dtype
        )
        op_args = fake_jax_op.call_args.args
        assert op_args[4] is pool
        assert op_args[13] is weight_scale
        # No prefill checkpoint operands were given.
        assert op_args[14:] == (None, None)
        # Optional bias/scale must not shift the runtime pool donation index.
        assert [arg for arg in op_args if arg is not None][4] is pool
        fake_pool, fake_output, fake_z = fake_jax_op.register_fake.call_args.args[0](
            *op_args
        )
        assert fake_pool.shape == pool.shape
        assert fake_output.shape == fake_z.shape == output.shape
        assert "gqa_interleaved_layout" not in mock_run.call_args.kwargs
        assert pool.untyped_storage().data_ptr() == original_storage
        assert torch.all(pool_alias == 7)
        assert torch.all(output == 1)
        assert torch.all(z == 2)

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_pcp_world_size",
        return_value=8,
    )
    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_or_create_pcp_mesh"
    )
    def test_pooled_pcp_seq_on_lane_uses_native_pool_layout(
        self, mock_get_pcp_mesh, _mock_get_pcp_world_size, monkeypatch
    ):
        vllm_envs.disable_envs_cache()
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
        attn = _qwen35_397b_gdn_attn("seq_on_lane_test_layer")
        mock_get_pcp_mesh.return_value = SimpleNamespace(shape={"pcp": 8, "tp": 1})
        vllm_config = _vllm_config(pcp_size=8, block_size=4096)
        captured = {}

        def fake_pcp_jax_op(_name, wrapped_fn, **_kwargs):
            captured["wrapped_fn"] = wrapped_fn
            return MagicMock()

        raw_pool = jnp.zeros((8, 2, 32, 4, 256), dtype=jnp.float8_e4m3fn)

        def fake_run(*args, **kwargs):
            captured["pooled_state"] = args[5]
            captured["pool_block_tokens"] = kwargs["pool_block_tokens"]
            return args[5], MagicMock(), MagicMock()

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=vllm_config),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pallas.jax_op",
                side_effect=fake_pcp_jax_op,
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "run_jax_gdn_attention_pooled_pcp_prefill_projection",
                side_effect=fake_run,
            ),
        ):
            attn._build_pooled_pcp_gdn_op()
            args = [MagicMock()] * 14
            args[4] = raw_pool  # Pool is the donated argument; scale is last.
            new_pool, _, _ = captured["wrapped_fn"](*args)

        assert captured["pooled_state"].shape == raw_pool.shape
        assert captured["pool_block_tokens"] == 4096
        assert new_pool.shape == raw_pool.shape

    def test_get_kv_cache_spec_localizes_gdn_state_for_pcp(self):
        attn = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
        attn.get_state_dtype = lambda: (torch.bfloat16, torch.float32)

        full_spec = attn.get_kv_cache_spec(_vllm_config(pcp_size=1))
        local_spec = attn.get_kv_cache_spec(_vllm_config(pcp_size=8))

        assert isinstance(full_spec, MambaSpec)
        assert isinstance(local_spec, MambaSpec)
        assert local_spec.shapes[0][:-1] == full_spec.shapes[0][:-1]
        assert local_spec.shapes[0][-1] == full_spec.shapes[0][-1] // 8
        assert local_spec.shapes[1][0] == full_spec.shapes[1][0] // 8
        assert local_spec.shapes[1][1:] == full_spec.shapes[1][1:]
        assert local_spec.page_size_bytes == full_spec.page_size_bytes // 8

    def test_get_kv_cache_spec_accounts_for_replicated_kq_heads(self):
        attn = _qwen38_gdn_attn("language_model.model.layers.0.linear_attn")
        attn.get_state_dtype = lambda: (torch.bfloat16, torch.float32)

        full_spec = attn.get_kv_cache_spec(_vllm_config(pcp_size=1))
        local_spec = attn.get_kv_cache_spec(_vllm_config(pcp_size=8))

        assert isinstance(full_spec, MambaSpec)
        assert isinstance(local_spec, MambaSpec)
        assert full_spec.shapes[0][-1] == 5120
        assert local_spec.shapes[0][-1] == 768
        assert local_spec.shapes[0][:-1] == full_spec.shapes[0][:-1]
        assert full_spec.shapes[1][0] == 32
        assert local_spec.shapes[1] == (4, 128, 128)
        assert local_spec.page_size_bytes == (3 * 768 * 2 + 4 * 128 * 128 * 4)

    def test_get_kv_cache_spec_drops_full_unpadded_padding_after_localizing(self):
        attn = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
        attn.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
        full_spec = attn.get_kv_cache_spec(_vllm_config(pcp_size=1))
        assert isinstance(full_spec, MambaSpec)

        local_spec = attn.get_kv_cache_spec(
            _vllm_config(pcp_size=8, mamba_page_size_padded=full_spec.page_size_bytes)
        )

        assert isinstance(local_spec, MambaSpec)
        assert local_spec.page_size_padded is None
        assert local_spec.page_size_bytes == full_spec.page_size_bytes // 8

    def test_get_kv_cache_spec_rejects_dim_first_state_for_pcp(self):
        attn = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
        attn.get_state_dtype = lambda: (torch.bfloat16, torch.float32)

        with (
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "is_conv_state_dim_first",
                return_value=True,
            ),
            pytest.raises(NotImplementedError, match="VLLM_SSM_CONV_STATE_LAYOUT=SD"),
        ):
            attn.get_kv_cache_spec(_vllm_config(pcp_size=8))

    def test_init_builds_only_regular_gdn_op_without_pcp(self):
        regular_op = MagicMock()
        pooled_op = MagicMock()

        with (
            patch.object(VllmGatedDeltaNetAttention, "_create_conv1d"),
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "QwenGatedDeltaNetAttention.__init__",
                return_value=None,
            ),
            patch.object(
                VllmGatedDeltaNetAttention, "_build_gdn_op", return_value=regular_op
            ) as mock_build,
            patch.object(
                VllmGatedDeltaNetAttention,
                "_build_pooled_gdn_op",
                return_value=pooled_op,
            ) as mock_build_pooled,
        ):
            attn = VllmGatedDeltaNetAttention()

        mock_build.assert_called_once_with()
        mock_build_pooled.assert_called_once_with(block_major=False)
        assert attn.gdn_op is regular_op
        assert attn.gdn_pooled_op is pooled_op
        assert not attn._pcp_streaming_configured
        assert attn.gdn_pcp_op is None
        assert attn.gdn_pooled_pcp_op is None

    def test_init_builds_pcp_gdn_op_when_pcp_enabled(self):
        regular_op = MagicMock()
        pooled_op = MagicMock()
        pcp_op = MagicMock()
        pooled_pcp_op = MagicMock()

        with (
            patch.object(VllmGatedDeltaNetAttention, "_create_conv1d"),
            set_vllm_model_wrapper_context(
                mesh=_mesh(), vllm_config=_vllm_config(pcp_size=8)
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "QwenGatedDeltaNetAttention.__init__",
                return_value=None,
            ),
            patch.object(
                VllmGatedDeltaNetAttention,
                "_build_gdn_op",
                side_effect=[regular_op, pcp_op],
            ) as mock_build,
            patch.object(
                VllmGatedDeltaNetAttention,
                "_build_pooled_gdn_op",
                return_value=pooled_op,
            ) as mock_build_pooled,
            patch.object(
                VllmGatedDeltaNetAttention,
                "_build_pooled_pcp_gdn_op",
                return_value=pooled_pcp_op,
            ) as mock_build_pooled_pcp,
        ):
            attn = VllmGatedDeltaNetAttention()

        assert mock_build.call_args_list == [call(), call(pcp_streaming=True)]
        mock_build_pooled.assert_called_once_with(block_major=False)
        mock_build_pooled_pcp.assert_called_once_with(block_major=False)
        assert attn.gdn_op is regular_op
        assert attn.gdn_pooled_op is pooled_op
        assert attn._pcp_streaming_configured
        assert attn.gdn_pcp_op is pcp_op
        assert attn.gdn_pooled_pcp_op is pooled_pcp_op

    @pytest.mark.parametrize(
        ("ssm_cache_dtype", "expected_dtype"),
        [
            ("float32", jnp.float32),
            ("bfloat16", jnp.bfloat16),
        ],
    )
    def test_pooled_op_reads_block_size_at_call_time(
        self, ssm_cache_dtype, expected_dtype, cpu_vllm_config_context
    ):
        """The pooled op must not freeze cache_config.block_size at build.

        Hybrid models construct GDN layers during load_model(), but the
        executor only calls update_block_size_for_backend() afterwards, so
        the value visible at build time is the pre-adjustment one. Binding
        it then makes the kernel run at the input block size while the pool
        was built at the derived one, tripping
        `pool_block_tokens % recurrent_state.shape[1]` inside the kernel.
        Only reproduces when --block-size is not passed explicitly, which
        is why every earlier unified-pool run missed it.
        """
        attn = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
        attn.cache_config.mamba_ssm_cache_dtype = ssm_cache_dtype
        # Pre-adjustment value, i.e. what a GDN layer sees during load.
        vllm_config = _vllm_config(block_size=16)
        captured = {}

        def fake_jax_op(_name, wrapped_fn, **_kwargs):
            captured["wrapped_fn"] = wrapped_fn
            return MagicMock()

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=vllm_config),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pallas.jax_op",
                side_effect=fake_jax_op,
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "gdn_attention_pooled_core_tpu",
            ) as mock_core,
        ):
            attn._build_pooled_gdn_op()

            # The executor finalizes the manager block size only after
            # load_model() has already built every GDN layer.
            vllm_config.cache_config.block_size = 256
            captured["wrapped_fn"](*([MagicMock()] * 12))

        assert mock_core.call_args.kwargs["pool_block_tokens"] == 256
        assert mock_core.call_args.kwargs["recurrent_state_dtype"] == jnp.dtype(
            expected_dtype
        )

    def test_pooled_impl_forwards_every_operand_the_forward_passes(
        self, cpu_vllm_config_context
    ):
        """`gdn_impl` must accept exactly what `forward` calls it with.

        The pooled operand list is threaded through four layers (forward ->
        gdn_impl -> jax_op/wrapped_fn -> the pooled runner), and an arity
        mismatch between them is invisible until torch.compile traces
        `gdn_impl` on device — a very expensive way to learn about a missing
        parameter. Calling it with the full positional list here catches it
        on CPU instead.
        """
        attn = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
        pool = torch.zeros((4, 8), dtype=torch.float32)
        captured = {}

        def fake_op(*args, **kwargs):
            captured["args"] = args
            return torch.zeros_like(pool), torch.zeros((2, 64, 128))

        fake_jax_op = MagicMock(side_effect=fake_op)
        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pallas.jax_op",
                return_value=fake_jax_op,
            ),
        ):
            gdn_impl = attn._build_pooled_gdn_op()

        # Mirrors the operand order at the forward's call site, ending with
        # slot_read_offsets, the per-checkpoint block indices and the prefill
        # checkpoint offsets and block ids.
        operands = (
            torch.zeros((2, 64)),  # mixed_qkv
            torch.zeros((2, 64)),  # b
            torch.zeros((2, 64)),  # a
            pool,  # recurrent_state
            torch.zeros((64, 1, 4)),  # conv_weight
            None,  # conv_bias
            torch.zeros(64),  # A_log
            torch.zeros(64),  # dt_bias
            torch.zeros(2, dtype=torch.int32),  # state_indices
            torch.zeros(3, dtype=torch.int32),  # query_start_loc
            torch.zeros(3, dtype=torch.int32),  # request_distribution
            torch.zeros(2, dtype=torch.int32),  # seq_lens
            torch.zeros(2, dtype=torch.int32),  # slot_read_offsets
            torch.zeros((2, 5), dtype=torch.int32),  # ckpt_indices
            torch.zeros(2, dtype=torch.int32),  # prefill_ckpt_offsets
            torch.full((2,), -1, dtype=torch.int32),  # prefill_ckpt_indices
        )
        gdn_impl(*operands)

        # Nothing silently dropped on the way to the kernel.
        assert len(captured["args"]) == len(operands)
        assert captured["args"][-1] is operands[-1]

    def test_pooled_op_is_shared_by_layers_with_one_kernel_config(
        self, cpu_vllm_config_context
    ):
        """Every op exports its kernel on its own, so layers must share one.

        One op per layer re-exported the identical kernel for each GDN layer
        at every compile bucket, which dominated cold-start FX interpretation.
        """
        attn0 = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
        attn1 = _qwen35_397b_gdn_attn("language_model.model.layers.1.linear_attn")
        spec_attn = _qwen35_397b_gdn_attn("language_model.model.layers.2.linear_attn")
        spec_attn.num_spec = 3

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pallas.jax_op",
                side_effect=lambda *args, **kwargs: MagicMock(),
            ) as mock_jax_op,
        ):
            for attn in (attn0, attn1, spec_attn):
                attn._build_pooled_gdn_op()

        names = [c.args[0] for c in mock_jax_op.call_args_list]
        assert len(names) == 2
        assert len(set(names)) == 2
        assert all(re.fullmatch(r"pallas::gdn_attention_pooled_\d+", n) for n in names)

    def test_build_gdn_op_keeps_regular_jax_op_per_layer(self):
        attn0 = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
        attn1 = _qwen35_397b_gdn_attn("language_model.model.layers.30.linear_attn")
        fake_jax_op = MagicMock()

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pallas.jax_op",
                return_value=fake_jax_op,
            ) as mock_jax_op,
        ):
            attn0._build_gdn_op()
            attn1._build_gdn_op()

        assert mock_jax_op.call_count == 2
        assert mock_jax_op.call_args_list[0].args[0] == (
            "pallas::gdn_attention_language_model_model_layers_0_linear_attn"
        )
        assert mock_jax_op.call_args_list[1].args[0] == (
            "pallas::gdn_attention_language_model_model_layers_30_linear_attn"
        )
        assert fake_jax_op.register_fake.call_count == 2

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_pcp_world_size",
        return_value=8,
    )
    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_or_create_pcp_mesh"
    )
    def test_build_gdn_op_keeps_pcp_jax_op_per_layer(
        self, mock_get_pcp_mesh, _mock_get_pcp_world_size
    ):
        attn0 = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
        attn1 = _qwen35_397b_gdn_attn("language_model.model.layers.30.linear_attn")
        fake_jax_op = MagicMock()
        pcp_mesh = SimpleNamespace(shape={"pcp": 8, "tp": 1})
        mock_get_pcp_mesh.return_value = pcp_mesh

        with (
            set_vllm_model_wrapper_context(
                mesh=_mesh(), vllm_config=_vllm_config(pcp_size=8)
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pallas.jax_op",
                return_value=fake_jax_op,
            ) as mock_pcp_jax_op,
        ):
            attn0._build_gdn_op(pcp_streaming=True)
            attn1._build_gdn_op(pcp_streaming=True)

        assert mock_pcp_jax_op.call_count == 2
        assert mock_pcp_jax_op.call_args_list[0].args[0] == (
            "pallas::gdn_attention_pcp_language_model_model_layers_0_linear_attn"
        )
        assert mock_pcp_jax_op.call_args_list[1].args[0] == (
            "pallas::gdn_attention_pcp_language_model_model_layers_30_linear_attn"
        )
        assert mock_pcp_jax_op.call_args_list[0].kwargs["mesh"] is pcp_mesh
        assert mock_pcp_jax_op.call_args_list[1].kwargs["mesh"] is pcp_mesh
        assert mock_pcp_jax_op.call_args_list[0].kwargs["input_partition_specs"][
            3
        ] == PartitionSpec(None, None, ("tp", "pcp"))
        assert mock_pcp_jax_op.call_args_list[0].kwargs["input_partition_specs"][
            4
        ] == PartitionSpec(None, ("tp", "pcp"), None, None)
        assert mock_pcp_jax_op.call_args_list[0].kwargs["output_partition_specs"][
            :2
        ] == (
            PartitionSpec(None, None, ("tp", "pcp")),
            PartitionSpec(None, ("tp", "pcp"), None, None),
        )
        mock_get_pcp_mesh.assert_has_calls(
            [
                call(axis_name="pcp", tp_axis_name="tp"),
                call(axis_name="pcp", tp_axis_name="tp"),
            ]
        )
        assert fake_jax_op.register_fake.call_count == 2

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_pcp_world_size",
        return_value=8,
    )
    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_or_create_pcp_mesh"
    )
    def test_pcp_compact_and_unified_ops_construct_with_mtp_k1(
        self, mock_get_pcp_mesh, _mock_get_pcp_world_size, cpu_vllm_config_context
    ):
        attn = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
        attn.num_spec = 1
        mock_get_pcp_mesh.return_value = SimpleNamespace(shape={"pcp": 8, "tp": 1})
        fake_jax_op = MagicMock()

        with (
            set_vllm_model_wrapper_context(
                mesh=_mesh(), vllm_config=_vllm_config(pcp_size=8)
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pallas.jax_op",
                return_value=fake_jax_op,
            ),
        ):
            compact_op = attn._build_gdn_op(pcp_streaming=True)
            unified_op = attn._build_pooled_pcp_gdn_op()

        assert callable(compact_op)
        assert callable(unified_op)

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_pcp_world_size",
        return_value=8,
    )
    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_or_create_pcp_mesh"
    )
    def test_pcp_ops_construct_with_replicated_kq_heads(
        self, mock_get_pcp_mesh, _mock_get_pcp_world_size, cpu_vllm_config_context
    ):
        attn = _qwen38_gdn_attn("language_model.model.layers.0.linear_attn")
        mock_get_pcp_mesh.return_value = SimpleNamespace(shape={"pcp": 8, "tp": 1})
        fake_jax_op = MagicMock()

        with (
            set_vllm_model_wrapper_context(
                mesh=_mesh(), vllm_config=_vllm_config(pcp_size=8)
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pallas.jax_op",
                return_value=fake_jax_op,
            ) as mock_pcp_jax_op,
        ):
            compact_op = attn._build_gdn_op(pcp_streaming=True)
            unified_op = attn._build_pooled_pcp_gdn_op()

        assert callable(compact_op)
        assert callable(unified_op)
        assert mock_pcp_jax_op.call_count == 2
        compact_wrapped_fn = mock_pcp_jax_op.call_args_list[0].args[1]
        assert compact_wrapped_fn.keywords["n_kq"] == 4
        assert compact_wrapped_fn.keywords["n_v"] == 32
        assert compact_wrapped_fn.keywords["pcp_size"] == 8

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_pcp_world_size",
        return_value=4,
    )
    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_or_create_pcp_mesh"
    )
    def test_pcp4_tp2_ops_preserve_replicated_kq_geometry(
        self, mock_get_pcp_mesh, _mock_get_pcp_world_size, cpu_vllm_config_context
    ):
        attn = _qwen38_gdn_attn("language_model.model.layers.0.linear_attn", tp_size=2)
        pcp_mesh = SimpleNamespace(shape={"pcp": 4, "tp": 2})
        mock_get_pcp_mesh.return_value = pcp_mesh
        fake_jax_op = MagicMock()

        with (
            set_vllm_model_wrapper_context(
                mesh=_mesh(), vllm_config=_vllm_config(pcp_size=4, tp_size=2)
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pallas.jax_op",
                return_value=fake_jax_op,
            ) as mock_pcp_jax_op,
        ):
            compact_op = attn._build_gdn_op(pcp_streaming=True)
            unified_op = attn._build_pooled_pcp_gdn_op()

        assert callable(compact_op)
        assert callable(unified_op)
        assert mock_pcp_jax_op.call_count == 2
        compact_wrapped_fn = mock_pcp_jax_op.call_args_list[0].args[1]
        assert compact_wrapped_fn.keywords["n_kq"] == 2
        assert compact_wrapped_fn.keywords["n_v"] == 16
        assert compact_wrapped_fn.keywords["pcp_size"] == 4
        pooled_specs = mock_pcp_jax_op.call_args_list[1].kwargs
        assert pooled_specs["mesh"] is pcp_mesh
        assert pooled_specs["input_partition_specs"] == (
            PartitionSpec("pcp", None),
            PartitionSpec("tp"),
            PartitionSpec("pcp", "tp"),
            PartitionSpec("pcp", "tp"),
            PartitionSpec(("tp", "pcp")),
            PartitionSpec("tp"),
            None,
            PartitionSpec("tp"),
            PartitionSpec("tp"),
            PartitionSpec(),
            PartitionSpec(),
            PartitionSpec(),
            PartitionSpec(),
            PartitionSpec("tp"),
            PartitionSpec(),
            PartitionSpec(),
        )
        assert pooled_specs["output_partition_specs"] == (
            PartitionSpec(("tp", "pcp")),
            PartitionSpec("pcp", "tp", None),
            PartitionSpec("pcp", "tp", None),
        )

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_cuda_lora(self, mock_get_forward_context):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.num_spec = 0
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"

        attn.conv1d = MagicMock()
        attn.conv1d.weight = torch.randn(1)
        attn.conv1d.bias = torch.randn(1)
        attn.A_log = torch.randn(1)
        attn.dt_bias = torch.randn(1)
        attn.kv_cache = (torch.ones(1), torch.ones(1))
        attn.gdn_op = MagicMock()

        attn.in_proj_qkv = MagicMock()
        attn.in_proj_z = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)

        attn.in_proj_qkv.return_value = (torch.randn(num_tokens, 96), None)
        attn.in_proj_z.return_value = (torch.randn(num_tokens, 64), None)
        attn.in_proj_ba.return_value = (torch.randn(num_tokens, 32), None)

        norm_out = torch.randn(num_tokens, 4, 16)
        attn.norm.return_value = norm_out
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)
        attn.gdn_op.return_value = torch.randn(num_tokens, 4, 16)

        mock_fc = MagicMock()
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.seq_lens = torch.ones(2)
        mock_attn_metadata.block_tables = torch.tensor([5, 6, 7, 8])
        # None exercises the legacy block_tables[:, 0] fallback used by
        # narrow unit tests; runtime Mamba metadata usually provides explicit
        # block-table-derived state indices.
        mock_attn_metadata.mamba_state_indices = None
        mock_attn_metadata.query_start_loc = torch.zeros(2)
        mock_attn_metadata.request_distribution = torch.zeros(3)
        # Non-spec path: the runner leaves these unset (None) unless
        # speculative decoding is active.
        mock_attn_metadata.mamba_request_distribution = None
        mock_attn_metadata.mamba_slot_read_offsets = None
        mock_attn_metadata.mamba_prefill_ckpt_offsets = None
        mock_fc.attn_metadata = {"test_layer": mock_attn_metadata}
        mock_get_forward_context.return_value = mock_fc

        output = attn.forward(hidden_states)

        attn.in_proj_qkv.assert_called_once_with(hidden_states)
        attn.in_proj_z.assert_called_once_with(hidden_states)
        attn.in_proj_ba.assert_called_once_with(hidden_states)

        assert attn.gdn_op.call_count == 1
        core_args = attn.gdn_op.call_args[0]

        assert core_args[0].shape == (num_tokens, 96)  # mixed_qkv
        assert core_args[1].shape == (num_tokens, 16)  # b
        assert core_args[2].shape == (num_tokens, 16)  # a
        assert core_args[3] is attn.kv_cache[0]
        assert core_args[4] is attn.kv_cache[1]
        assert core_args[5] is attn.conv1d.weight
        assert core_args[6] is attn.conv1d.bias
        assert core_args[7] is attn.A_log
        assert core_args[8] is attn.dt_bias
        assert torch.all(core_args[9] == torch.tensor([5, 7], dtype=torch.int32))
        assert core_args[10] is mock_attn_metadata.query_start_loc
        assert core_args[11] is mock_attn_metadata.request_distribution

        attn.norm.assert_called_once()
        # Verify z was correctly reshaped: [num_tokens, -1, head_v_dim]
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        # Verify reshaped output from norm went to out_proj
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        assert torch.all(output == 5)

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_uses_canonical_projection_for_interleaved_checkpoint(
        self, mock_get_forward_context
    ):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.num_spec = 0
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.num_k_heads = 4
        attn.head_k_dim = 8
        attn.value_dim = 64
        attn.tp_size = 1
        attn.prefix = "test_layer"
        attn.gqa_interleaved_layout = True

        # Minimal additions for TorchTPU
        attn.conv1d = MagicMock()
        attn.conv1d.weight = torch.randn(1)
        attn.conv1d.bias = torch.randn(1)
        attn.A_log = torch.randn(1)
        attn.dt_bias = torch.randn(1)
        attn.kv_cache = (torch.ones(1), torch.ones(1))
        attn.gdn_op = MagicMock()

        # Mocks for non-LoRA GQA path
        attn.in_proj_qkvz = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)

        mixed_qkv = torch.randn(num_tokens, 128)
        z = torch.randn(num_tokens, 4, 16)
        b = torch.randn(num_tokens, 4)
        a = torch.randn(num_tokens, 4)
        attn.in_proj_qkvz.return_value = (
            torch.cat((mixed_qkv, z.flatten(1)), dim=-1),
            None,
        )
        attn.in_proj_ba.return_value = (torch.cat((b, a), dim=-1), None)

        norm_out = torch.randn(num_tokens, 4, 16)
        attn.norm.return_value = norm_out
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)
        attn.gdn_op.return_value = torch.randn(num_tokens, 4, 16)

        mock_fc = MagicMock()
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.seq_lens = torch.ones(2)
        mock_attn_metadata.block_tables = torch.tensor([5, 6, 7, 8])
        # None exercises the legacy block_tables[:, 0] fallback used by
        # narrow unit tests; runtime Mamba metadata usually provides explicit
        # block-table-derived state indices.
        mock_attn_metadata.mamba_state_indices = None
        mock_attn_metadata.query_start_loc = torch.zeros(2)
        mock_attn_metadata.request_distribution = torch.zeros(3)
        # Non-spec path: the runner leaves these unset (None) unless
        # speculative decoding is active.
        mock_attn_metadata.mamba_request_distribution = None
        mock_attn_metadata.mamba_slot_read_offsets = None
        mock_attn_metadata.mamba_prefill_ckpt_offsets = None
        mock_fc.attn_metadata = {"test_layer": mock_attn_metadata}
        mock_get_forward_context.return_value = mock_fc

        output = attn.forward(hidden_states)

        attn.in_proj_qkvz.assert_called_once_with(hidden_states)
        attn.in_proj_ba.assert_called_once_with(hidden_states)

        assert attn.gdn_op.call_count == 1
        core_args = attn.gdn_op.call_args[0]

        torch.testing.assert_close(core_args[0], mixed_qkv)
        torch.testing.assert_close(core_args[1], b)
        torch.testing.assert_close(core_args[2], a)
        assert core_args[3] is attn.kv_cache[0]
        assert core_args[4] is attn.kv_cache[1]
        assert core_args[5] is attn.conv1d.weight
        assert core_args[6] is attn.conv1d.bias
        assert core_args[7] is attn.A_log
        assert core_args[8] is attn.dt_bias
        assert torch.all(core_args[9] == torch.tensor([5, 7], dtype=torch.int32))
        assert core_args[10] is mock_attn_metadata.query_start_loc
        assert core_args[11] is mock_attn_metadata.request_distribution

        attn.norm.assert_called_once()
        # Verify unpacked z is natively used
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        assert torch.all(output == 5)

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_uses_compact_mamba_state_indices(self, mock_get_forward_context):
        """Compact mamba: when attn_metadata.mamba_state_indices is set, the op
        passes it through verbatim and ignores block_tables[:, 0]."""
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.num_spec = 0
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"
        attn.gqa_interleaved_layout = True

        attn.conv1d = MagicMock()
        attn.conv1d.weight = torch.randn(1)
        attn.conv1d.bias = torch.randn(1)
        attn.A_log = torch.randn(1)
        attn.dt_bias = torch.randn(1)
        attn.kv_cache = (torch.ones(1), torch.ones(1))
        attn.gdn_op = MagicMock()

        attn.in_proj_qkv = MagicMock()
        attn.in_proj_z = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)

        attn.in_proj_qkv.return_value = (torch.randn(num_tokens, 96), None)
        attn.in_proj_z.return_value = (torch.randn(num_tokens, 64), None)
        attn.in_proj_ba.return_value = (torch.randn(num_tokens, 32), None)
        attn.norm.return_value = torch.randn(num_tokens, 4, 16)
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)
        attn.gdn_op.return_value = torch.randn(num_tokens, 4, 16)

        mock_fc = MagicMock()
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.seq_lens = torch.ones(2)
        # block_tables would yield [5, 7] via the fallback; the compact slot
        # ids [3, 1] must win instead.
        mock_attn_metadata.block_tables = torch.tensor([5, 6, 7, 8])
        mock_attn_metadata.mamba_state_indices = torch.tensor([3, 1], dtype=torch.int32)
        mock_attn_metadata.query_start_loc = torch.zeros(2)
        mock_attn_metadata.request_distribution = torch.zeros(3)
        # Non-spec path: the runner leaves these unset (None) unless
        # speculative decoding is active.
        mock_attn_metadata.mamba_request_distribution = None
        mock_attn_metadata.mamba_slot_read_offsets = None
        mock_attn_metadata.mamba_prefill_ckpt_offsets = None
        mock_fc.attn_metadata = {"test_layer": mock_attn_metadata}
        mock_get_forward_context.return_value = mock_fc

        attn.forward(hidden_states)

        assert attn.gdn_op.call_count == 1
        core_args = attn.gdn_op.call_args[0]
        assert torch.all(core_args[9] == torch.tensor([3, 1], dtype=torch.int32))

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_forward_context"
    )
    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_pcp_rank",
        return_value=1,
    )
    def test_forward_uses_native_gdn_pcp_op_for_pcp_prefill(
        self, _mock_rank, mock_get_forward_context
    ):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.num_spec = 1
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"
        attn.gqa_interleaved_layout = True

        attn.conv1d = MagicMock()
        attn.conv1d.weight = torch.randn(1)
        attn.conv1d.bias = torch.randn(1)
        attn.A_log = torch.randn(1)
        attn.dt_bias = torch.randn(1)
        attn.kv_cache = (torch.ones(1), torch.ones(1))
        attn.gdn_op = MagicMock()
        attn.gdn_pcp_op = MagicMock()

        attn.in_proj_qkv = MagicMock()
        attn.in_proj_z = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)
        output = torch.zeros(5, 64)

        mixed_local = torch.full((num_tokens, 96), -1.0)
        b_local = torch.full((num_tokens, 16), -2.0)
        a_local = torch.full((num_tokens, 16), -3.0)
        attn.in_proj_qkv.return_value = (mixed_local, None)
        attn.in_proj_z.return_value = (torch.randn(num_tokens, 64), None)
        attn.in_proj_ba.return_value = (torch.cat((b_local, a_local), dim=-1), None)
        attn.norm.side_effect = lambda core, _z: core
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)
        native_output = torch.stack(
            [
                torch.full((64,), 10.0),
                torch.full((64,), 11.0),
                torch.full((64,), 12.0),
                torch.full((64,), 13.0),
            ]
        )
        attn.gdn_pcp_op.return_value = native_output

        mock_fc = MagicMock()
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.seq_lens = torch.tensor([4], dtype=torch.int32)
        mock_attn_metadata.block_tables = torch.tensor([5, 6, 7, 8])
        mock_attn_metadata.mamba_state_indices = torch.tensor([3, 1], dtype=torch.int32)
        mock_attn_metadata.sequence_layout_kind = SequenceLayoutKind.PARTIAL.value
        mock_attn_metadata.sequence_layout_protocol = "pcp_streaming"
        mock_attn_metadata.query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
        mock_attn_metadata.request_distribution = torch.zeros(3)
        # PCP prefill does not consume non-PCP verify rollback offsets.
        mock_attn_metadata.mamba_request_distribution = None
        mock_attn_metadata.mamba_slot_read_offsets = None
        mock_attn_metadata.mamba_prefill_ckpt_offsets = None
        mock_fc.attn_metadata = {"test_layer": mock_attn_metadata}
        mock_get_forward_context.return_value = mock_fc

        with set_vllm_model_wrapper_context(
            mesh=_mesh(), vllm_config=_vllm_config(pcp_size=2, interleave_size=1)
        ):
            output = attn.forward(hidden_states)

        attn.gdn_op.assert_not_called()
        attn.gdn_pcp_op.assert_called_once()
        core_args = attn.gdn_pcp_op.call_args[0]
        assert core_args[0] is mixed_local
        assert torch.equal(core_args[1], b_local)
        assert torch.equal(core_args[2], a_local)
        assert core_args[3] is attn.kv_cache[0]
        assert core_args[4] is attn.kv_cache[1]
        assert core_args[9] is mock_attn_metadata.mamba_state_indices
        assert core_args[10] is mock_attn_metadata.query_start_loc
        assert core_args[11] is mock_attn_metadata.request_distribution
        assert core_args[12] is mock_attn_metadata.seq_lens
        assert len(core_args) == 13
        expected_local = torch.stack(
            [torch.full((4, 16), 12.0), torch.full((4, 16), 13.0)]
        )
        assert torch.equal(attn.norm.call_args[0][0], expected_local)
        assert torch.all(output == 5)

    @pytest.mark.parametrize("shape", [(), (1,), (192,), (6, 2), (1, 2, 1, 192)])
    def test_pcp_projection_accepts_fp8_scale_layouts(self, shape):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.gqa_interleaved_layout = False
        weight = torch.ones((192, 256), dtype=torch.float8_e4m3fn)
        scale = torch.ones(shape, dtype=torch.float32)
        attn.in_proj_qkvz = SimpleNamespace(
            weight=weight, weight_scale=scale, bias=None
        )
        result = attn._require_pcp_projection_parameters()
        assert result[0] is weight and result[1] is scale

    @pytest.mark.parametrize(
        "dtype,scale_size,bias,error",
        [
            (torch.bfloat16, 192, False, "must not have a scale"),
            (torch.float8_e4m3fn, None, False, "FP32 scale"),
            (torch.float8_e4m3fn, 3, False, "scale must"),
            (torch.float32, None, False, "BF16 or"),
            (torch.bfloat16, None, True, "bias-free"),
        ],
    )
    def test_pcp_projection_rejects_invalid_parameters(
        self, dtype, scale_size, bias, error
    ):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.gqa_interleaved_layout = False
        attn.in_proj_qkvz = SimpleNamespace(
            weight=torch.ones((192, 64), dtype=dtype),
            weight_scale=None if scale_size is None else torch.ones(scale_size),
            bias=torch.zeros(192) if bias else None,
        )
        with pytest.raises(RuntimeError, match=error):
            attn._require_pcp_projection_parameters()

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_non_pcp_spec_verify_still_requires_slot_offsets(
        self, mock_get_forward_context
    ):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.num_spec = 1
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"
        attn.gqa_interleaved_layout = True
        attn.conv1d = SimpleNamespace(weight=torch.randn(1), bias=torch.randn(1))
        attn.A_log = torch.randn(1)
        attn.dt_bias = torch.randn(1)
        attn.kv_cache = (torch.ones(1), torch.ones(1))
        attn.gdn_op = MagicMock()
        attn.gdn_pcp_op = MagicMock()
        attn.in_proj_qkv = MagicMock(return_value=(torch.randn(2, 96), None))
        attn.in_proj_z = MagicMock(return_value=(torch.randn(2, 64), None))
        attn.in_proj_ba = MagicMock(return_value=(torch.randn(2, 32), None))

        metadata = SimpleNamespace(
            seq_lens=torch.ones(2),
            block_tables=torch.tensor([5, 6, 7, 8]),
            mamba_state_indices=torch.tensor([3, 1], dtype=torch.int32),
            sequence_layout_kind=SequenceLayoutKind.ALL.value,
            sequence_layout_protocol="default",
            query_start_loc=torch.zeros(2),
            request_distribution=torch.zeros(3),
            mamba_request_distribution=None,
            mamba_slot_read_offsets=None,
            mamba_prefill_ckpt_offsets=None,
        )
        mock_get_forward_context.return_value = SimpleNamespace(
            attn_metadata={"test_layer": metadata}
        )

        with pytest.raises(AssertionError, match="requires mamba_slot_read_offsets"):
            attn.forward(torch.randn(2, 64))

        attn.gdn_op.assert_not_called()
        attn.gdn_pcp_op.assert_not_called()

    @pytest.mark.parametrize("projection_dtype", [torch.bfloat16, torch.float8_e4m3fn])
    @pytest.mark.parametrize("projection_bias", [False, True])
    @pytest.mark.parametrize("interleaved", [False, True])
    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_uses_fused_projection_for_pooled_pcp_prefill(
        self, mock_get_forward_context, projection_dtype, interleaved, projection_bias
    ):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.num_spec = 1
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.num_k_heads = 2
        attn.tp_size = 1
        attn.prefix = "test_layer"
        attn.gqa_interleaved_layout = interleaved
        attn._pcp_streaming_configured = True

        attn.conv1d = MagicMock()
        attn.conv1d.weight = torch.randn(1)
        attn.conv1d.bias = torch.randn(1)
        attn.A_log = torch.randn(1)
        attn.dt_bias = torch.randn(1)
        attn.kv_cache = (torch.ones(1),)
        attn.gdn_op = MagicMock()
        attn.gdn_pooled_op = MagicMock()
        attn.gdn_pooled_pcp_op = MagicMock()

        qkvz_weight = torch.ones((192, 64), dtype=projection_dtype)
        qkvz_weight_scale = (
            torch.ones(192, dtype=torch.float32)
            if projection_dtype == torch.float8_e4m3fn
            else None
        )
        attn.in_proj_qkvz = SimpleNamespace(
            weight=qkvz_weight,
            weight_scale=qkvz_weight_scale,
            bias=torch.zeros(192) if projection_bias else None,
        )
        attn.in_proj_ba = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)
        b_local = torch.arange(num_tokens * 4).reshape(num_tokens, 4).float()
        a_local = b_local + 100
        # Both checkpoint formats have already been normalized by loading.
        ba = torch.cat((b_local, a_local), dim=-1)
        attn.in_proj_ba.return_value = (ba.reshape(num_tokens, -1), None)

        fused_output = torch.arange(
            num_tokens * 4 * 16,
            dtype=torch.float32,
        ).reshape(num_tokens, 4, 16)
        fused_z = fused_output + 1000
        attn.gdn_pooled_pcp_op.return_value = fused_output, fused_z
        attn.norm.side_effect = lambda core, _z: core
        attn.out_proj.return_value = (torch.full((num_tokens, 64), 5.0), None)

        metadata = MagicMock()
        metadata.seq_lens = torch.tensor([4], dtype=torch.int32)
        metadata.mamba_state_indices = torch.tensor([3, 1], dtype=torch.int32)
        metadata.sequence_layout_kind = SequenceLayoutKind.PARTIAL.value
        metadata.sequence_layout_protocol = "pcp_streaming"
        metadata.query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
        metadata.request_distribution = torch.zeros(3, dtype=torch.int32)
        metadata.mamba_prefill_ckpt_offsets = None
        metadata.mamba_prefill_ckpt_indices = None
        mock_get_forward_context.return_value = SimpleNamespace(
            attn_metadata={"test_layer": metadata}
        )

        with set_vllm_model_wrapper_context(
            mesh=_mesh(), vllm_config=_vllm_config(pcp_size=2, interleave_size=1)
        ):
            if projection_bias:
                with pytest.raises(RuntimeError, match="bias-free"):
                    attn.forward(hidden_states)
                attn.in_proj_ba.assert_not_called()
                attn.gdn_op.assert_not_called()
                attn.gdn_pooled_op.assert_not_called()
                attn.gdn_pooled_pcp_op.assert_not_called()
                return
            output = attn.forward(hidden_states)

        attn.gdn_op.assert_not_called()
        attn.gdn_pooled_op.assert_not_called()
        attn.gdn_pooled_pcp_op.assert_called_once()
        core_args = attn.gdn_pooled_pcp_op.call_args.args
        assert core_args[0] is hidden_states
        assert core_args[1] is qkvz_weight
        assert core_args[2] is qkvz_weight_scale
        assert torch.equal(core_args[3], b_local)
        assert torch.equal(core_args[4], a_local)
        assert core_args[5] is attn.kv_cache[0]
        assert core_args[10] is metadata.mamba_state_indices
        assert core_args[11] is metadata.query_start_loc
        assert core_args[12] is metadata.request_distribution
        assert core_args[13] is metadata.seq_lens
        assert core_args[14:] == (None, None)
        assert len(core_args) == 16
        assert torch.equal(attn.norm.call_args.args[0], fused_output)
        assert attn.norm.call_args.args[1] is fused_z
        assert torch.all(output == 5)

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_pcp_prefill_requires_initialized_pcp_op(
        self, mock_get_forward_context
    ):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.num_spec = 0
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"
        attn.gqa_interleaved_layout = True

        attn.conv1d = MagicMock()
        attn.conv1d.weight = torch.randn(1)
        attn.conv1d.bias = torch.randn(1)
        attn.A_log = torch.randn(1)
        attn.dt_bias = torch.randn(1)
        attn.kv_cache = (torch.ones(1), torch.ones(1))
        attn.gdn_op = MagicMock()
        attn.gdn_pcp_op = None

        attn.in_proj_qkv = MagicMock()
        attn.in_proj_z = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)

        attn.in_proj_qkv.return_value = (torch.randn(num_tokens, 96), None)
        attn.in_proj_z.return_value = (torch.randn(num_tokens, 64), None)
        attn.in_proj_ba.return_value = (torch.randn(num_tokens, 32), None)
        attn.gdn_op.return_value = torch.randn(num_tokens, 4, 16)
        attn.norm.return_value = torch.randn(num_tokens, 4, 16)
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)

        mock_fc = MagicMock()
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.seq_lens = torch.ones(2)
        mock_attn_metadata.block_tables = torch.tensor([5, 6, 7, 8])
        mock_attn_metadata.mamba_state_indices = torch.tensor([3, 1], dtype=torch.int32)
        mock_attn_metadata.sequence_layout_kind = SequenceLayoutKind.PARTIAL.value
        mock_attn_metadata.sequence_layout_protocol = "pcp_streaming"
        mock_attn_metadata.query_start_loc = torch.zeros(2)
        mock_attn_metadata.request_distribution = torch.zeros(3)
        # Non-spec path: the runner leaves these unset (None) unless
        # speculative decoding is active.
        mock_attn_metadata.mamba_request_distribution = None
        mock_attn_metadata.mamba_slot_read_offsets = None
        mock_attn_metadata.mamba_prefill_ckpt_offsets = None
        mock_fc.attn_metadata = {"test_layer": mock_attn_metadata}
        mock_get_forward_context.return_value = mock_fc

        with (
            set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
            pytest.raises(RuntimeError, match="GDN PCP prefill op was not initialized"),
        ):
            attn.forward(hidden_states)

        attn.gdn_op.assert_not_called()

    @patch(
        "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_cuda_non_lora_no_gqa(self, mock_get_forward_context):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.num_spec = 0
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"
        attn.gqa_interleaved_layout = False
        attn.key_dim = 32
        attn.num_k_heads = 2
        attn.head_k_dim = 16
        attn.value_dim = 64

        attn.conv1d = MagicMock()
        attn.conv1d.weight = torch.randn(1)
        attn.conv1d.bias = torch.randn(1)
        attn.A_log = torch.randn(1)
        attn.dt_bias = torch.randn(1)
        attn.kv_cache = (torch.ones(1), torch.ones(1))
        attn.gdn_op = MagicMock()

        attn.in_proj_qkvz = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)
        output = torch.zeros(5, 64)

        qkv_size = (attn.key_dim * 2 + attn.value_dim) // attn.tp_size  # 128
        z_size = attn.value_dim // attn.tp_size  # 64
        mixed_qkvz = torch.randn(num_tokens, qkv_size + z_size)

        attn.in_proj_qkvz.return_value = (mixed_qkvz, None)
        attn.in_proj_ba.return_value = (torch.randn(num_tokens, 32), None)

        norm_out = torch.randn(num_tokens, 4, 16)
        attn.norm.return_value = norm_out
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)
        attn.gdn_op.return_value = torch.randn(num_tokens, 4, 16)

        mock_fc = MagicMock()
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.seq_lens = torch.ones(2)
        mock_attn_metadata.block_tables = torch.tensor([5, 6, 7, 8])
        # None exercises the legacy block_tables[:, 0] fallback used by
        # narrow unit tests; runtime Mamba metadata usually provides explicit
        # block-table-derived state indices.
        mock_attn_metadata.mamba_state_indices = None
        mock_attn_metadata.query_start_loc = torch.zeros(2)
        mock_attn_metadata.request_distribution = torch.zeros(3)
        # Non-spec path: the runner leaves these unset (None) unless
        # speculative decoding is active.
        mock_attn_metadata.mamba_request_distribution = None
        mock_attn_metadata.mamba_slot_read_offsets = None
        mock_attn_metadata.mamba_prefill_ckpt_offsets = None
        mock_fc.attn_metadata = {"test_layer": mock_attn_metadata}
        mock_get_forward_context.return_value = mock_fc

        output = attn.forward(hidden_states)

        attn.in_proj_qkvz.assert_called_once_with(hidden_states)
        attn.in_proj_ba.assert_called_once_with(hidden_states)

        assert attn.gdn_op.call_count == 1
        core_args = attn.gdn_op.call_args[0]

        # mixed_qkv should be separated accurately
        assert core_args[0].shape == (num_tokens, 128)
        assert core_args[1].shape == (num_tokens, 16)
        assert core_args[2].shape == (num_tokens, 16)
        assert core_args[3] is attn.kv_cache[0]
        assert core_args[4] is attn.kv_cache[1]
        assert core_args[5] is attn.conv1d.weight
        assert core_args[6] is attn.conv1d.bias
        assert core_args[7] is attn.A_log
        assert core_args[8] is attn.dt_bias
        assert torch.all(core_args[9] == torch.tensor([5, 7], dtype=torch.int32))
        assert core_args[10] is mock_attn_metadata.query_start_loc
        assert core_args[11] is mock_attn_metadata.request_distribution

        attn.norm.assert_called_once()
        # Verify z was split and reshaped correctly
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        assert torch.all(output == 5)


class TestConvWeightReHome:
    """process_weights_after_loading backs the 3-D conv weight, a view over
    the 2-D ColumnParallelLinear buffer, with a buffer of its own shape."""

    def test_weight_is_copied_into_its_own_buffer(self):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        base = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        weight = torch.nn.Parameter(base.unsqueeze(1), requires_grad=False)
        attn.conv1d = SimpleNamespace(weight=weight)
        aliased_ptr = weight.data_ptr()
        VllmGatedDeltaNetAttention.process_weights_after_loading(attn, torch.bfloat16)
        w = attn.conv1d.weight
        assert w is weight
        assert tuple(w.shape) == (6, 1, 4)
        assert w.dtype == torch.float32
        assert w.data_ptr() != aliased_ptr
        assert torch.equal(w.data.squeeze(1), base)


@pytest.mark.parametrize("pcp_size,expected", [(1, False), (8, True)])
def test_pcp_streaming_follows_the_declared_pcp_width(pcp_size, expected):
    with set_vllm_model_wrapper_context(
        mesh=_mesh(), vllm_config=_vllm_config(pcp_size=pcp_size)
    ):
        assert VllmGatedDeltaNetAttention._pcp_streaming_enabled() is expected


def test_pcp_streaming_is_off_without_a_vllm_config():
    with set_vllm_model_wrapper_context(mesh=_mesh()):
        assert VllmGatedDeltaNetAttention._pcp_streaming_enabled() is False


# --- block-major unified pool ---------------------------------------------------

_BM_LAYOUT = BlockMajorPoolLayout(
    num_blocks=4, num_pools=3, split=4, pool_index_by_layer={}
)
# Pool index 1 maps -1 to -2, so the -1 checkpoint ids below depend on the
# re-mask after the remap.
_BM_POOL_INDEX = 1


@pytest.mark.parametrize("block_major", [True, False])
def test_fused_pooled_pcp_forward_remaps_state_ids_for_the_block_major_pool(
    monkeypatch, block_major
):
    attn = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
    attn.conv1d = SimpleNamespace(weight=torch.zeros(64, 1, 4), bias=None)
    attn.A_log = torch.zeros(64)
    attn.dt_bias = torch.zeros(64)
    attn.in_proj_ba = lambda hidden: (torch.zeros(hidden.shape[0], 4), None)
    attn._require_pcp_projection_parameters = lambda: (torch.zeros(1, 1), None)
    attn._apply_output_projection = lambda out, _z, _num_tokens: out
    captured = {}

    def fake_pooled_pcp_op(
        hidden, _w, _s, _b, _a, _pool, _cw, _cb, _a_log, _dt, state_indices, *rest
    ):
        captured["state_indices"] = state_indices
        captured["prefill_ckpt"] = rest[3:]
        out = torch.zeros(hidden.shape[0], 64, 128)
        return out, out

    attn.gdn_pooled_pcp_op = fake_pooled_pcp_op
    if block_major:
        attn._bm_kernel = True
        attn.set_block_major_pool(_BM_POOL_INDEX, _BM_LAYOUT)
    monkeypatch.setattr(
        gdn_attention_op, "is_pcp_streaming_attention_metadata", lambda _m: True
    )
    state_indices = torch.tensor([0, 1, 3], dtype=torch.int32)
    ckpt_offsets = torch.tensor([0, 3072, 0], dtype=torch.int32)
    ckpt_ids = torch.tensor([-1, 2, -1], dtype=torch.int32)
    metadata = SimpleNamespace(
        mamba_state_indices=state_indices,
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        request_distribution=torch.tensor([3, 0, 0], dtype=torch.int32),
        seq_lens=torch.ones(3, dtype=torch.int32),
        mamba_prefill_ckpt_offsets=ckpt_offsets,
        mamba_prefill_ckpt_indices=ckpt_ids,
    )

    attn._forward_fused_pooled_pcp(torch.zeros(3, 8), torch.zeros(4, 12, 2), metadata)

    passed = captured["state_indices"]
    passed_offsets, passed_ids = captured["prefill_ckpt"]
    assert passed_offsets is ckpt_offsets
    if block_major:
        expected = flat_manager_ids(state_indices, _BM_POOL_INDEX, 3)
        assert passed.tolist() == expected.tolist() == [1, 4, 10]
        # Checkpoint ids take the same remap; "none" stays -1.
        assert passed_ids.tolist() == [-1, 7, -1]
    else:
        assert passed is state_indices
        assert passed_ids is ckpt_ids


@pytest.mark.parametrize(
    "block_major, suffix, kernel_pool_shape",
    [(True, "_bm", (4 * 12, 2, 3)), (False, "", (4, 12, 2, 3))],
)
def test_pooled_op_block_major_folds_the_pool_and_restores_its_shape(
    block_major, suffix, kernel_pool_shape
):
    attn = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
    captured = {}

    def fake_jax_op(name, wrapped_fn, **_kwargs):
        captured["name"] = name
        captured["wrapped_fn"] = wrapped_fn
        return MagicMock()

    def fake_core(mixed_qkv, _b, _a, pool, *_args, **_kwargs):
        captured["kernel_pool_shape"] = tuple(pool.shape)
        return pool, mixed_qkv

    pool = torch.zeros(4, 12, 2, 3)
    with (
        set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
        patch(
            "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.pallas.jax_op",
            side_effect=fake_jax_op,
        ),
        patch(
            "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
            "gdn_attention_pooled_core_tpu",
            side_effect=fake_core,
        ),
    ):
        attn._build_pooled_gdn_op(block_major=block_major)
        new_pool, _ = captured["wrapped_fn"](
            *([MagicMock()] * 3), pool, *([MagicMock()] * 8)
        )

    assert re.fullmatch(rf"pallas::gdn_attention_pooled{suffix}_\d+", captured["name"])
    assert captured["kernel_pool_shape"] == kernel_pool_shape
    assert tuple(new_pool.shape) == tuple(pool.shape)


# --- internal prefill checkpoints ---------------------------------------------


def _ckpt_vllm_config(
    *,
    pcp_size: int = 8,
    mamba_cache_mode: str | None = "align",
    dcp_size: int = 1,
    architecture: str = "Qwen3_5MoeForCausalLM",
    prefix_match_unit: int | None = None,
):
    config = _vllm_config(pcp_size=pcp_size)
    config.cache_config.mamba_cache_mode = mamba_cache_mode
    config.cache_config.prefix_match_unit = prefix_match_unit
    config.cache_config.mamba_block_size = 384
    config.parallel_config.decode_context_parallel_size = dcp_size
    config.model_config = SimpleNamespace(architecture=architecture)
    return config


def _spec_attn():
    attn = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
    attn.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
    return attn


@pytest.fixture
def ckpt_env(monkeypatch):
    monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", raising=False)
    return monkeypatch


@pytest.mark.parametrize("pcp_size", [1, 8])
def test_spec_declares_one_prefill_checkpoint_block_in_align_mode(ckpt_env, pcp_size):
    spec = _spec_attn().get_kv_cache_spec(_ckpt_vllm_config(pcp_size=pcp_size))
    plain = _spec_attn().get_kv_cache_spec(
        _ckpt_vllm_config(pcp_size=pcp_size, mamba_cache_mode="none")
    )

    assert isinstance(spec, MambaSpec)
    assert spec.num_prefill_checkpoint_blocks == 1
    assert spec.prefill_checkpoint_alignment == 1
    # Only the two checkpoint fields change; the PCP-local state shapes stay.
    assert spec.shapes == plain.shapes
    assert spec.page_size_bytes == plain.page_size_bytes


@pytest.mark.parametrize(
    "env, config_kwargs",
    [
        ({}, {"mamba_cache_mode": "none"}),
        ({}, {"mamba_cache_mode": "all"}),
        ({}, {"dcp_size": 2}),
        ({"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "0"}, {}),
        ({}, {"architecture": "LlamaForCausalLM"}),
        # A hash block finer than the 3072-token logical Mamba block.
        ({}, {"prefix_match_unit": 1024}),
    ],
)
def test_spec_declares_no_prefill_checkpoint_block_when_gated_off(
    ckpt_env, env, config_kwargs
):
    for name, value in env.items():
        ckpt_env.setenv(name, value)
    spec = _spec_attn().get_kv_cache_spec(_ckpt_vllm_config(**config_kwargs))

    assert isinstance(spec, MambaSpec)
    assert spec.num_prefill_checkpoint_blocks == 0
    assert spec.prefill_checkpoint_alignment is None


def _pooled_forward_attn(kv_cache):
    attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
    attn.num_spec = 0
    attn.head_v_dim = 16
    attn.num_v_heads = 4
    attn.tp_size = 1
    attn.prefix = "test_layer"
    attn._pcp_streaming_configured = False
    attn.conv1d = SimpleNamespace(weight=torch.randn(1), bias=None)
    attn.A_log = torch.randn(1)
    attn.dt_bias = torch.randn(1)
    attn.kv_cache = kv_cache
    attn.gdn_op = MagicMock(return_value=torch.zeros(2, 4, 16))
    attn.gdn_pooled_op = MagicMock(return_value=torch.zeros(2, 4, 16))
    attn.in_proj_qkv = MagicMock(return_value=(torch.randn(2, 96), None))
    attn.in_proj_z = MagicMock(return_value=(torch.randn(2, 64), None))
    attn.in_proj_ba = MagicMock(return_value=(torch.randn(2, 32), None))
    attn._apply_output_projection = lambda out, _z, _num_tokens: out
    return attn


def _pooled_metadata(offsets, ids):
    return SimpleNamespace(
        seq_lens=torch.tensor([8191, 1], dtype=torch.int32),
        block_tables=torch.zeros(8, dtype=torch.int32),
        mamba_state_indices=torch.tensor([3, 1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
        mamba_request_distribution=None,
        mamba_slot_read_offsets=None,
        mamba_ckpt_indices=None,
        mamba_prefill_ckpt_offsets=offsets,
        mamba_prefill_ckpt_indices=ids,
        sequence_layout_kind=SequenceLayoutKind.ALL.value,
        sequence_layout_protocol="default",
    )


@pytest.mark.parametrize("block_major", [False, True])
@patch("vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_forward_context")
def test_pooled_forward_passes_prefill_checkpoint_operands(
    mock_get_forward_context, block_major
):
    attn = _pooled_forward_attn((torch.zeros(4, 12, 2),))
    if block_major:
        attn._bm_kernel = True
        attn.set_block_major_pool(_BM_POOL_INDEX, _BM_LAYOUT)
    offsets = torch.tensor([3072, 0], dtype=torch.int32)
    ids = torch.tensor([2, -1], dtype=torch.int32)
    metadata = _pooled_metadata(offsets, ids)
    mock_get_forward_context.return_value = SimpleNamespace(
        attn_metadata={"test_layer": metadata}
    )

    attn.forward(torch.randn(2, 64))

    attn.gdn_op.assert_not_called()
    args = attn.gdn_pooled_op.call_args.args
    assert len(args) == 16
    assert args[14] is offsets
    if block_major:
        # The same remap as the state ids (block * 3 + 1); -1 stays -1.
        assert args[8].tolist() == [10, 4]
        assert args[15].tolist() == [7, -1]
    else:
        assert args[15] is ids


@patch("vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_forward_context")
def test_pooled_forward_without_checkpoints_passes_none(mock_get_forward_context):
    attn = _pooled_forward_attn((torch.zeros(4, 12, 2),))
    mock_get_forward_context.return_value = SimpleNamespace(
        attn_metadata={"test_layer": _pooled_metadata(None, None)}
    )

    attn.forward(torch.randn(2, 64))

    args = attn.gdn_pooled_op.call_args.args
    assert args[14:] == (None, None)


@patch("vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.get_forward_context")
def test_split_pool_forward_rejects_prefill_checkpoints(mock_get_forward_context):
    attn = _pooled_forward_attn((torch.zeros(4, 3, 8), torch.zeros(4, 2, 2, 2)))
    metadata = _pooled_metadata(
        torch.tensor([3072, 0], dtype=torch.int32),
        torch.tensor([2, -1], dtype=torch.int32),
    )
    mock_get_forward_context.return_value = SimpleNamespace(
        attn_metadata={"test_layer": metadata}
    )

    with pytest.raises(AssertionError, match="unified KV pool"):
        attn.forward(torch.randn(2, 64))
    attn.gdn_op.assert_not_called()


def test_pooled_op_threads_prefill_checkpoint_operands_to_the_core(
    cpu_vllm_config_context,
):
    attn = _qwen35_397b_gdn_attn("language_model.model.layers.0.linear_attn")
    captured = {}

    def fake_jax_op(name, wrapped_fn, **_kwargs):
        captured["wrapped_fn"] = wrapped_fn
        return MagicMock()

    def fake_core(mixed_qkv, _b, _a, pool, *args, **_kwargs):
        captured["core_args"] = args
        return pool, mixed_qkv

    with (
        set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()),
        patch(
            "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op.pallas.jax_op",
            side_effect=fake_jax_op,
        ),
        patch(
            "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
            "gdn_attention_pooled_core_tpu",
            side_effect=fake_core,
        ),
    ):
        attn._build_pooled_gdn_op()
        offsets, ids = MagicMock(), MagicMock()
        captured["wrapped_fn"](
            *([MagicMock()] * 3),
            torch.zeros(4, 12, 2, 3),
            *([MagicMock()] * 10),
            offsets,
            ids,
        )

    # conv_weight .. ckpt_indices, then the two checkpoint operands.
    assert captured["core_args"][-2] is offsets
    assert captured["core_args"][-1] is ids


def _set_qkvz_kernel_layout(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("TPU_GDN_QKVZ_KERNEL_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("TPU_GDN_QKVZ_KERNEL_LAYOUT", flag)


@pytest.mark.parametrize(
    "flag, pcp_enabled, keeps_layout",
    [
        (None, False, True),
        (None, True, True),
        ("1", False, False),
        ("1", True, True),
    ],
)
def test_qkvz_layout_follows_the_flag_except_under_pcp(
    make_gdn_attention, monkeypatch, flag, pcp_enabled, keeps_layout
):
    """in_proj_qkvz keeps vLLM's [n_out, n_in] layout by default. With
    TPU_GDN_QKVZ_KERNEL_LAYOUT it flips to (k, n) like every other linear,
    except under PCP, whose pooled op reads the weight raw."""
    from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR

    _set_qkvz_kernel_layout(monkeypatch, flag)
    monkeypatch.setattr(
        VllmGatedDeltaNetAttention,
        "_pcp_streaming_enabled",
        staticmethod(lambda: pcp_enabled),
    )
    warning_once = MagicMock()
    monkeypatch.setattr(gdn_attention_op.logger, "warning_once", warning_once)

    attention, _ = make_gdn_attention(1, 0, "fp8_block")

    assert attention._pcp_streaming_configured is pcp_enabled
    keeps = getattr(attention.in_proj_qkvz, KEEP_VLLM_LAYOUT_ATTR, False)
    assert keeps is keeps_layout
    warned = any(
        "TPU_GDN_QKVZ_KERNEL_LAYOUT" in c.args[0] for c in warning_once.call_args_list
    )
    assert warned is (flag is not None and pcp_enabled)


def test_flipped_qkvz_projection_matches_the_kept_one(make_gdn_attention, monkeypatch):
    """TPU_GDN_QKVZ_KERNEL_LAYOUT changes only how in_proj_qkvz is stored: the
    loaded projection gives the same output in either layout."""
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmUnquantizedLinearMethod,
    )

    generator = torch.Generator().manual_seed(20261103)
    weight = x = None
    results = {}
    for flag in (None, "1"):
        _set_qkvz_kernel_layout(monkeypatch, flag)
        attention, _ = make_gdn_attention(1, 0)
        projection = attention.in_proj_qkvz
        # Production BF16 models get this method from VllmUnquantizedConfig;
        # the fixture's bf16 layer has vLLM's stock one, which never flips.
        projection.quant_method = VllmUnquantizedLinearMethod()
        if weight is None:
            # Multiples of 1/16 keep every product and sum exact.
            n_out, n_in = projection.weight.shape
            weight = (torch.randint(-8, 9, (n_out, n_in), generator=generator) / 16).to(
                torch.bfloat16
            )
            x = (torch.randint(-8, 9, (3, n_in), generator=generator) / 16).to(
                torch.bfloat16
            )
        projection.weight.data.copy_(weight)
        projection.quant_method.process_weights_after_loading(projection)
        output, _ = projection(x)
        results[flag] = (tuple(projection.weight.shape), output)

    assert results[None][0] == tuple(weight.shape)
    assert results["1"][0] == tuple(weight.T.shape)
    expected = (x.float() @ weight.float().T).to(torch.bfloat16)
    for _, output in results.values():
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_flipped_qkvz_fp8_block_weight_is_the_kept_one_transposed(
    make_gdn_attention, monkeypatch
):
    """On the per-block FP8 path the flag targets, post-load requantization
    stores the same weight and scales, with the weight in (k, n)."""
    generator = torch.Generator().manual_seed(20261103)
    weight = None
    loaded = {}
    for flag in (None, "1"):
        _set_qkvz_kernel_layout(monkeypatch, flag)
        attention, _ = make_gdn_attention(1, 0, "fp8_block")
        projection = attention.in_proj_qkvz
        if weight is None:
            weight = (
                torch.randint(
                    -16, 17, tuple(projection.weight.shape), generator=generator
                )
                / 16
            ).to(torch.float8_e4m3fn)
        projection.weight.data.copy_(weight)
        projection.weight_scale_inv.data.fill_(1.0)
        projection.quant_method.process_weights_after_loading(projection)
        loaded[flag] = (projection.weight.float(), projection.weight_scale.float())

    kept_weight, kept_scale = loaded[None]
    flipped_weight, flipped_scale = loaded["1"]
    assert kept_weight.shape == weight.shape
    torch.testing.assert_close(flipped_weight, kept_weight.T, rtol=0, atol=0)
    torch.testing.assert_close(flipped_scale, kept_scale, rtol=0, atol=0)


def test_pack_qkvz_op_registration_sharing_and_mesh_none(
    make_gdn_attention, monkeypatch
):
    """TPU_GDN_PACK_QKVZ builds and shares one op per geometry, stays None when
    off, and tolerates mesh=None during CPU/checkpoint export."""
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    gdn_attention_op.get_chip_version.cache_clear()

    # Flag off: no pack op registered.
    monkeypatch.delenv("TPU_GDN_PACK_QKVZ", raising=False)
    attn_off, vllm_config = make_gdn_attention(1, 0)
    assert attn_off.gdn_pack_qkvz_op is None

    # Flag on: registered once and reused across layers with matching geometry,
    # and fake registrations return float32 3D qkv + bf16 3D z / core_attn_out.
    registered_fakes = {}

    def capture_jax_op(name, fn, **_kwargs):
        op = MagicMock(name=name, side_effect=fn)

        def register_fake(fake_fn):
            registered_fakes[name] = fake_fn

        op.register_fake.side_effect = register_fake
        return op

    monkeypatch.setenv("TPU_GDN_PACK_QKVZ", "1")
    monkeypatch.setattr(gdn_attention_op.pallas, "jax_op", capture_jax_op)
    gdn_attention_op._shared_gdn_ops.clear()

    attn_1, _ = make_gdn_attention(1, 0)
    attn_2, _ = make_gdn_attention(1, 0)
    assert attn_1.gdn_pack_qkvz_op is not None
    assert attn_1.gdn_pack_qkvz_op is attn_2.gdn_pack_qkvz_op

    pack_fakes = [
        fn for name, fn in registered_fakes.items() if "gdn_split_pack_qkvz" in name
    ]
    assert len(pack_fakes) == 1
    qkv_size = attn_1.tp_head_geometry.local_conv_dim(
        attn_1.head_k_dim, attn_1.head_v_dim
    )
    z_size = attn_1.value_dim // attn_1.tp_size
    fake_qkv, fake_z = pack_fakes[0](
        torch.empty((16, qkv_size + z_size), dtype=torch.bfloat16)
    )
    assert fake_qkv.shape == (16, 1, qkv_size)
    assert fake_qkv.dtype == torch.float32
    assert fake_z.shape == (16, attn_1.num_v_heads, attn_1.head_v_dim)
    assert fake_z.dtype == torch.bfloat16

    pooled_fakes = [fn for name, fn in registered_fakes.items() if "pooled" in name]
    assert len(pooled_fakes) >= 1
    _, fake_pooled_out = pooled_fakes[0](
        fake_qkv,
        torch.empty((16, attn_1.num_v_heads), dtype=torch.bfloat16),
        torch.empty((16, attn_1.num_v_heads), dtype=torch.bfloat16),
        torch.empty((4, 2, 128, 128), dtype=torch.float32),
    )
    assert fake_pooled_out.dtype == torch.bfloat16

    regular_fakes = [
        fn
        for name, fn in registered_fakes.items()
        if "pooled" not in name
        and "pcp" not in name
        and "gdn_split_pack_qkvz" not in name
    ]
    assert len(regular_fakes) >= 1
    _, _, fake_reg_out = regular_fakes[0](
        fake_qkv,
        torch.empty((16, attn_1.num_v_heads), dtype=torch.bfloat16),
        torch.empty((16, attn_1.num_v_heads), dtype=torch.bfloat16),
        torch.empty((4, 3, qkv_size), dtype=torch.float32),
        torch.empty((4, attn_1.num_v_heads, 128, 128), dtype=torch.float32),
    )
    assert fake_reg_out.dtype == torch.bfloat16

    # CPU/checkpoint export passes mesh=None; construction must succeed.
    with (
        torch.device("cpu"),
        set_current_vllm_config(vllm_config),
        set_vllm_model_wrapper_context(mesh=None, vllm_config=vllm_config),
    ):
        attn_export = VllmGatedDeltaNetAttention(
            config=vllm_config.model_config.hf_text_config,
            vllm_config=vllm_config,
            prefix="model.layers.1.linear_attn",
            gqa_interleaved_layout=False,
        )
    assert attn_export.gdn_pack_qkvz_op is attn_1.gdn_pack_qkvz_op


def test_pack_qkvz_forward_profile_pooled_split_and_mtp_guard(
    make_gdn_attention, monkeypatch
):
    """Forward routes aligned steps through gdn_pack_qkvz_op during profile_run,
    pooled kv_cache, and 2-tensor kv_cache, and falls back on unaligned rows,
    PCP streaming, or compiled modules outside shape_variants warmup."""
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    monkeypatch.setenv("TPU_GDN_PACK_QKVZ", "1")
    gdn_attention_op.get_chip_version.cache_clear()

    attn, _ = make_gdn_attention(1, 0)
    for sub in (attn.in_proj_qkvz, attn.in_proj_ba, attn.out_proj):
        sub.quant_method.process_weights_after_loading(sub)
    qkv_size = attn.tp_head_geometry.local_conv_dim(attn.head_k_dim, attn.head_v_dim)
    local_n_v = attn.num_v_heads // attn.tp_size
    pack_calls = []
    seen_qkv = []

    def real_split_pack(mixed_qkvz: torch.Tensor):
        pack_calls.append(tuple(mixed_qkvz.shape))
        qkv = mixed_qkvz[:, :qkv_size].to(torch.float32).unsqueeze(1).contiguous()
        z = (
            mixed_qkvz[:, qkv_size:]
            .clone()
            .reshape(mixed_qkvz.size(0), local_n_v, attn.head_v_dim)
        )
        return qkv, z

    def stub_core(mixed_qkv: torch.Tensor, b: torch.Tensor, *_args):
        seen_qkv.append((tuple(mixed_qkv.shape), mixed_qkv.dtype))
        return torch.zeros(
            (mixed_qkv.size(0), local_n_v, attn.head_v_dim),
            dtype=b.dtype,
            device=b.device,
        )

    attn.gdn_pack_qkvz_op = real_split_pack
    attn.gdn_pooled_op = stub_core
    attn.gdn_op = stub_core

    hidden_16 = torch.randn((16, 128), dtype=torch.bfloat16)

    # 1. profile_run (kv_cache=None): runs pack_op for HBM accounting and returns bf16.
    attn.kv_cache = None
    out_prof = attn.forward(hidden_16)
    assert out_prof.dtype == torch.bfloat16
    assert len(pack_calls) == 1
    assert seen_qkv == []

    # 2. Unified pool (len(kv_cache) == 1) on aligned 16 rows.
    metadata = SimpleNamespace(
        mamba_state_indices=torch.tensor([1, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8, 16], dtype=torch.int32),
        request_distribution=torch.tensor([0, 2, 2], dtype=torch.int32),
        seq_lens=torch.tensor([8, 8], dtype=torch.int32),
        mamba_prefill_ckpt_offsets=None,
        mamba_prefill_ckpt_indices=None,
    )
    fwd_ctx = SimpleNamespace(attn_metadata={attn.prefix: metadata})
    monkeypatch.setattr(gdn_attention_op, "get_forward_context", lambda: fwd_ctx)
    attn.kv_cache = (torch.ones((4, 2, 128, 128), dtype=torch.float32),)
    pack_calls.clear()
    seen_qkv.clear()
    out_pooled = attn.forward(hidden_16)
    assert out_pooled.shape == (16, 128)
    assert len(pack_calls) == 1
    assert seen_qkv == [((16, 1, qkv_size), torch.float32)]

    # 3. Split state (len(kv_cache) == 2) on aligned 16 rows vs unaligned 4 rows.
    attn.kv_cache = (
        torch.ones((4, 3, qkv_size), dtype=torch.float32),
        torch.ones((4, local_n_v, 128, 128), dtype=torch.float32),
    )
    pack_calls.clear()
    seen_qkv.clear()
    attn.forward(hidden_16)
    assert len(pack_calls) == 1
    assert seen_qkv == [((16, 1, qkv_size), torch.float32)]

    pack_calls.clear()
    seen_qkv.clear()
    attn.forward(torch.randn((4, 128), dtype=torch.bfloat16))
    assert pack_calls == []
    assert seen_qkv == [((4, qkv_size), torch.bfloat16)]

    # 4. Compiled outside shape_variants warmup (e.g. MTP): skips pack_op.
    @torch.library.custom_op("test_pack_qkvz::pack", mutates_args=())
    def custom_pack(mixed_qkvz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pack_calls.append(tuple(mixed_qkvz.shape))
        qkv = mixed_qkvz[:, :qkv_size].to(torch.float32).unsqueeze(1).contiguous()
        z = (
            mixed_qkvz[:, qkv_size:]
            .clone()
            .reshape(mixed_qkvz.size(0), local_n_v, attn.head_v_dim)
        )
        return qkv, z

    @custom_pack.register_fake
    def _(mixed_qkvz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(
                (mixed_qkvz.size(0), 1, qkv_size),
                dtype=torch.float32,
                device=mixed_qkvz.device,
            ),
            torch.empty(
                (mixed_qkvz.size(0), local_n_v, attn.head_v_dim),
                dtype=mixed_qkvz.dtype,
                device=mixed_qkvz.device,
            ),
        )

    @torch.library.custom_op("test_pack_qkvz::core", mutates_args=())
    def custom_core(mixed_qkv: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        seen_qkv.append((tuple(mixed_qkv.shape), mixed_qkv.dtype))
        return torch.zeros(
            (mixed_qkv.size(0), local_n_v, attn.head_v_dim),
            dtype=b.dtype,
            device=b.device,
        )

    @custom_core.register_fake
    def _(mixed_qkv: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (mixed_qkv.size(0), local_n_v, attn.head_v_dim),
            dtype=b.dtype,
            device=b.device,
        )

    attn.gdn_pack_qkvz_op = custom_pack
    attn.gdn_op = lambda mixed_qkv, b, *_args: custom_core(mixed_qkv, b)

    pack_calls.clear()
    seen_qkv.clear()
    compiled_outside_warmup = torch.compile(
        attn.forward, backend="eager", fullgraph=True
    )
    out_compiled = compiled_outside_warmup(hidden_16)
    assert out_compiled.shape == (16, 128)
    assert pack_calls == []
    assert seen_qkv == [((16, qkv_size), torch.bfloat16)]

    # 5. Inside shape_variants warmup: traces fullgraph with pack_op.
    pack_calls.clear()
    seen_qkv.clear()
    monkeypatch.setattr(gdn_attention_op.shape_variants, "aot_cache_tag", lambda: "b16")
    torch._dynamo.reset()
    compiled_in_warmup = torch.compile(attn.forward, backend="eager", fullgraph=True)
    out_warmup = compiled_in_warmup(hidden_16)
    assert out_warmup.shape == (16, 128)
    assert len(pack_calls) == 1
    assert seen_qkv == [((16, 1, qkv_size), torch.float32)]
