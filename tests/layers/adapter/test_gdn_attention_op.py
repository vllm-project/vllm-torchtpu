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

from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
    VllmGatedDeltaNetAttention,
    gdn_attention_core_tpu_pcp_prefill,
)
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

pytestmark = pytest.mark.cpu_test


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
    interleave_size: int = 16,
    block_size: int = 16,
    mamba_page_size_padded: int | None = None,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            cp_kv_cache_interleave_size=interleave_size,
            data_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            mamba_block_size=4096,
            mamba_page_size_padded=mamba_page_size_padded,
            mamba_cache_mode=None,
            use_kda_recoverssm=False,
        ),
        speculative_config=None,
        num_speculative_tokens=0,
    )


def _qwen35_397b_gdn_attn(prefix: str, *, bias: bool = False):
    attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
    attn.prefix = prefix
    attn.num_k_heads = 64
    attn.gqa_interleaved_layout = False
    attn.num_v_heads = 64
    attn.tp_size = 1
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


def _qwen38_gdn_attn(prefix: str):
    attn = _qwen35_397b_gdn_attn(prefix)
    attn.num_k_heads = 4
    attn.num_v_heads = 32
    return attn


class TestVllmGatedDeltaNetAttention:
    @pytest.mark.parametrize("page_size", [128, 256, 512])
    @pytest.mark.parametrize(
        "num_kv_heads,kv_packing,head_dim,dtype",
        [
            (1, 4, 256, jnp.float8_e4m3fn),
            (1, 2, 256, jnp.bfloat16),
            (2, 2, 256, jnp.bfloat16),
            (4, 4, 256, jnp.float8_e4m3fn),
        ],
    )
    def test_seq_on_lane_native_pool_plan(
        self, monkeypatch, page_size, num_kv_heads, kv_packing, head_dim, dtype
    ):
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
        from vllm_torchtpu.layers.core.gdn_attention import _build_v3_pool_state_plan

        manager_block_size = 4096 if page_size == 512 else 4352
        num_kv_heads_x2 = num_kv_heads * 2
        packed_head_dim = head_dim // kv_packing
        shape = (10, num_kv_heads_x2, packed_head_dim, kv_packing, page_size)
        pool = jnp.zeros(shape, dtype=dtype)

        plan = _build_v3_pool_state_plan(
            pool,
            conv_dim=1536,
            n_v=8,
            d_k=128,
            d_v=128,
            kernel_size=4,
            pool_block_tokens=manager_block_size,
            qk_pair_layout=False,
            recurrent_state_dtype=jnp.float32,
        )
        assert plan.stride == manager_block_size // page_size
        assert plan.recurrent.view_dtype == jnp.float32
        assert plan.conv.view_dtype == jnp.bfloat16
        assert plan.whole_block_dma is True

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
        mock_get_pcp_mesh.return_value = SimpleNamespace(shape={"pcp": 8})
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
                "pcp_streaming_jax_op",
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
        assert op_args[-1] is weight_scale
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
        mock_get_pcp_mesh.return_value = SimpleNamespace(shape={"pcp": 8})
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
                "pcp_streaming_jax_op",
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
        mock_build_pooled.assert_called_once_with()
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
        mock_build_pooled.assert_called_once_with()
        mock_build_pooled_pcp.assert_called_once_with()
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
        # slot_read_offsets and the per-checkpoint block indices.
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
        )
        gdn_impl(*operands)

        # Nothing silently dropped on the way to the kernel.
        assert len(captured["args"]) == len(operands)
        assert captured["args"][-1] is operands[-1]

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
        pcp_mesh = SimpleNamespace(shape={"pcp": 8})
        mock_get_pcp_mesh.return_value = pcp_mesh

        with (
            set_vllm_model_wrapper_context(
                mesh=_mesh(), vllm_config=_vllm_config(pcp_size=8)
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pcp_streaming_jax_op",
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
        ] == PartitionSpec(None, None, "pcp")
        assert mock_pcp_jax_op.call_args_list[0].kwargs["input_partition_specs"][
            4
        ] == PartitionSpec(None, "pcp", None, None)
        assert mock_pcp_jax_op.call_args_list[0].kwargs["output_partition_specs"][
            :2
        ] == (
            PartitionSpec(None, None, "pcp"),
            PartitionSpec(None, "pcp", None, None),
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
        mock_get_pcp_mesh.return_value = SimpleNamespace(shape={"pcp": 8})
        fake_jax_op = MagicMock()

        with (
            set_vllm_model_wrapper_context(
                mesh=_mesh(), vllm_config=_vllm_config(pcp_size=8)
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pcp_streaming_jax_op",
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
        self, mock_get_pcp_mesh, _mock_get_pcp_world_size
    ):
        attn = _qwen38_gdn_attn("language_model.model.layers.0.linear_attn")
        mock_get_pcp_mesh.return_value = SimpleNamespace(shape={"pcp": 8})
        fake_jax_op = MagicMock()

        with (
            set_vllm_model_wrapper_context(
                mesh=_mesh(), vllm_config=_vllm_config(pcp_size=8)
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op."
                "pcp_streaming_jax_op",
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
        assert len(core_args) == 14
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
        mock_fc.attn_metadata = {"test_layer": mock_attn_metadata}
        mock_get_forward_context.return_value = mock_fc

        with set_vllm_model_wrapper_context(mesh=_mesh(), vllm_config=_vllm_config()):
            with pytest.raises(
                RuntimeError, match="GDN PCP prefill op was not initialized"
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
