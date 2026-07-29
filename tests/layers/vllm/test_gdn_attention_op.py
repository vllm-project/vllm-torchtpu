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

import pytest
import torch
from jax.sharding import PartitionSpec
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.layers.common.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op import \
    VllmGatedDeltaNetAttention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context


def _mesh():
    return SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})


def _vllm_config(*,
                 pcp_size: int = 1,
                 interleave_size: int = 16,
                 block_size: int = 16,
                 mamba_page_size_padded: int | None = None):
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
        ),
        speculative_config=None,
    )


def _qwen35_397b_gdn_attn(prefix: str, *, bias: bool = False):
    attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
    attn.prefix = prefix
    attn.num_k_heads = 64
    attn.num_v_heads = 64
    attn.tp_size = 1
    attn.head_k_dim = 128
    attn.head_v_dim = 128
    attn.conv_kernel_size = 4
    attn.num_spec = 0
    attn.conv1d = SimpleNamespace(bias=torch.randn(64) if bias else None, )
    return attn


class TestVllmGatedDeltaNetAttention:

    def test_get_kv_cache_spec_localizes_gdn_state_for_pcp(self):
        attn = _qwen35_397b_gdn_attn(
            "language_model.model.layers.0.linear_attn")
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

    def test_get_kv_cache_spec_drops_full_unpadded_padding_after_localizing(
            self):
        attn = _qwen35_397b_gdn_attn(
            "language_model.model.layers.0.linear_attn")
        attn.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
        full_spec = attn.get_kv_cache_spec(_vllm_config(pcp_size=1))
        assert isinstance(full_spec, MambaSpec)

        local_spec = attn.get_kv_cache_spec(
            _vllm_config(pcp_size=8,
                         mamba_page_size_padded=full_spec.page_size_bytes))

        assert isinstance(local_spec, MambaSpec)
        assert local_spec.page_size_padded is None
        assert local_spec.page_size_bytes == full_spec.page_size_bytes // 8

    def test_get_kv_cache_spec_rejects_dim_first_state_for_pcp(self):
        attn = _qwen35_397b_gdn_attn(
            "language_model.model.layers.0.linear_attn")
        attn.get_state_dtype = lambda: (torch.bfloat16, torch.float32)

        with patch(
                "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op."
                "is_conv_state_dim_first",
                return_value=True,
        ), pytest.raises(NotImplementedError,
                         match="VLLM_SSM_CONV_STATE_LAYOUT=SD"):
            attn.get_kv_cache_spec(_vllm_config(pcp_size=8))

    def test_init_builds_only_regular_gdn_op_without_pcp(self):
        regular_op = MagicMock()
        pooled_op = MagicMock()

        with set_vllm_model_wrapper_context(mesh=_mesh(),
                                            vllm_config=_vllm_config()), \
             patch(
                 "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op."
                 "QwenGatedDeltaNetAttention.__init__",
                 return_value=None,
             ), \
             patch.object(VllmGatedDeltaNetAttention,
                          "_build_gdn_op",
                          return_value=regular_op) as mock_build, \
             patch.object(VllmGatedDeltaNetAttention,
                          "_build_pooled_gdn_op",
                          return_value=pooled_op) as mock_build_pooled:
            attn = VllmGatedDeltaNetAttention()

        mock_build.assert_called_once_with()
        mock_build_pooled.assert_called_once_with()
        assert attn.gdn_op is regular_op
        assert attn.gdn_pooled_op is pooled_op
        assert attn.gdn_pcp_op is None
        assert attn.gdn_pooled_pcp_op is None

    def test_init_builds_pcp_gdn_op_when_pcp_enabled(self):
        regular_op = MagicMock()
        pooled_op = MagicMock()
        pcp_op = MagicMock()
        pooled_pcp_op = MagicMock()

        with set_vllm_model_wrapper_context(mesh=_mesh(),
                                            vllm_config=_vllm_config(
                                                pcp_size=8)), \
             patch(
                 "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op."
                 "QwenGatedDeltaNetAttention.__init__",
                 return_value=None,
             ), \
             patch.object(VllmGatedDeltaNetAttention,
                          "_build_gdn_op",
                          side_effect=[regular_op, pcp_op]) as mock_build, \
             patch.object(VllmGatedDeltaNetAttention,
                          "_build_pooled_gdn_op",
                          return_value=pooled_op) as mock_build_pooled, \
             patch.object(
                 VllmGatedDeltaNetAttention,
                 "_build_pooled_pcp_gdn_op",
                 return_value=pooled_pcp_op) as mock_build_pooled_pcp:
            attn = VllmGatedDeltaNetAttention()

        assert mock_build.call_args_list == [call(), call(pcp_streaming=True)]
        mock_build_pooled.assert_called_once_with()
        mock_build_pooled_pcp.assert_called_once_with()
        assert attn.gdn_op is regular_op
        assert attn.gdn_pooled_op is pooled_op
        assert attn.gdn_pcp_op is pcp_op
        assert attn.gdn_pooled_pcp_op is pooled_pcp_op

    def test_pooled_op_reads_block_size_at_call_time(self):
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
        attn = _qwen35_397b_gdn_attn(
            "language_model.model.layers.0.linear_attn")
        # Pre-adjustment value, i.e. what a GDN layer sees during load.
        vllm_config = _vllm_config(block_size=16)
        captured = {}

        def fake_jax_op(_name, wrapped_fn, **_kwargs):
            captured["wrapped_fn"] = wrapped_fn
            return MagicMock()

        with set_vllm_model_wrapper_context(mesh=_mesh(),
                                            vllm_config=vllm_config), \
             patch(
                 "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op."
                 "pallas.jax_op",
                 side_effect=fake_jax_op,
             ), \
             patch(
                 "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op."
                 "gdn_attention_pooled_core_tpu",
             ) as mock_core:
            attn._build_pooled_gdn_op()

            # The executor finalizes the manager block size only after
            # load_model() has already built every GDN layer.
            vllm_config.cache_config.block_size = 256
            captured["wrapped_fn"](*([MagicMock()] * 12))

        assert mock_core.call_args.kwargs["pool_block_tokens"] == 256

    def test_build_gdn_op_keeps_regular_jax_op_per_layer(self):
        attn0 = _qwen35_397b_gdn_attn(
            "language_model.model.layers.0.linear_attn")
        attn1 = _qwen35_397b_gdn_attn(
            "language_model.model.layers.30.linear_attn")
        fake_jax_op = MagicMock()

        with set_vllm_model_wrapper_context(mesh=_mesh(),
                                            vllm_config=_vllm_config()), \
             patch(
                 "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op."
                 "pallas.jax_op",
                 return_value=fake_jax_op,
             ) as mock_jax_op:
            attn0._build_gdn_op()
            attn1._build_gdn_op()

        assert mock_jax_op.call_count == 2
        assert mock_jax_op.call_args_list[0].args[0] == (
            "pallas::gdn_attention_"
            "language_model_model_layers_0_linear_attn")
        assert mock_jax_op.call_args_list[1].args[0] == (
            "pallas::gdn_attention_"
            "language_model_model_layers_30_linear_attn")
        assert fake_jax_op.register_fake.call_count == 2

    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_pcp_world_size",
        return_value=8)
    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_or_create_pcp_mesh"
    )
    def test_build_gdn_op_keeps_pcp_jax_op_per_layer(self, mock_get_pcp_mesh,
                                                     _mock_get_pcp_world_size):
        attn0 = _qwen35_397b_gdn_attn(
            "language_model.model.layers.0.linear_attn")
        attn1 = _qwen35_397b_gdn_attn(
            "language_model.model.layers.30.linear_attn")
        fake_jax_op = MagicMock()
        pcp_mesh = SimpleNamespace(shape={"pcp": 8})
        mock_get_pcp_mesh.return_value = pcp_mesh

        with set_vllm_model_wrapper_context(mesh=_mesh(),
                                            vllm_config=_vllm_config(
                                                pcp_size=8)), \
             patch(
                 "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op."
                 "pcp_streaming_jax_op",
                 return_value=fake_jax_op,
             ) as mock_pcp_jax_op:
            attn0._build_gdn_op(pcp_streaming=True)
            attn1._build_gdn_op(pcp_streaming=True)

        assert mock_pcp_jax_op.call_count == 2
        assert mock_pcp_jax_op.call_args_list[0].args[0] == (
            "pallas::gdn_attention_pcp_"
            "language_model_model_layers_0_linear_attn")
        assert mock_pcp_jax_op.call_args_list[1].args[0] == (
            "pallas::gdn_attention_pcp_"
            "language_model_model_layers_30_linear_attn")
        assert mock_pcp_jax_op.call_args_list[0].kwargs["mesh"] is pcp_mesh
        assert mock_pcp_jax_op.call_args_list[1].kwargs["mesh"] is pcp_mesh
        assert mock_pcp_jax_op.call_args_list[0].kwargs[
            "input_partition_specs"][3] == PartitionSpec(None, None, "pcp")
        assert mock_pcp_jax_op.call_args_list[0].kwargs[
            "input_partition_specs"][4] == PartitionSpec(
                None, "pcp", None, None)
        assert mock_pcp_jax_op.call_args_list[0].kwargs[
            "output_partition_specs"][:2] == (
                PartitionSpec(None, None, "pcp"),
                PartitionSpec(None, "pcp", None, None),
            )
        assert fake_jax_op.register_fake.call_count == 2

    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
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
        output = torch.zeros(5, 64)

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

        attn.forward(hidden_states, output)

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
        assert torch.all(
            core_args[9] == torch.tensor([5, 7], dtype=torch.int32))
        assert core_args[10] is mock_attn_metadata.query_start_loc
        assert core_args[11] is mock_attn_metadata.request_distribution

        attn.norm.assert_called_once()
        # Verify z was correctly reshaped: [num_tokens, -1, head_v_dim]
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        # Verify reshaped output from norm went to out_proj
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        # Check that output buffer was updated only up to num_tokens
        assert torch.all(output[:num_tokens] == 5)
        assert torch.all(output[num_tokens:] == 0)

    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_cuda_non_lora_gqa(self, mock_get_forward_context):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.num_spec = 0
        attn.head_v_dim = 16
        attn.num_v_heads = 4
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
        attn.fix_query_key_value_ordering = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)
        output = torch.zeros(5, 64)

        attn.in_proj_qkvz.return_value = (torch.randn(num_tokens, 192), None)
        attn.in_proj_ba.return_value = (torch.randn(num_tokens, 32), None)

        query = torch.randn(num_tokens, 4, 8)
        key = torch.randn(num_tokens, 4, 8)
        value = torch.randn(num_tokens, 4, 8)
        z = torch.randn(num_tokens, 4, 16)
        b = torch.randn(num_tokens, 16)
        a = torch.randn(num_tokens, 16)

        attn.fix_query_key_value_ordering.return_value = (query, key, value, z,
                                                          b, a)

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

        attn.forward(hidden_states, output)

        attn.in_proj_qkvz.assert_called_once_with(hidden_states)
        attn.in_proj_ba.assert_called_once_with(hidden_states)
        attn.fix_query_key_value_ordering.assert_called_once()

        assert attn.gdn_op.call_count == 1
        core_args = attn.gdn_op.call_args[0]

        # mixed_qkv should be cat of rearranged query, key, value
        # rearranged from "l p d -> l (p d)", e.g. 2x(4*8) = 2x32 -> cat into 2x96
        assert core_args[0].shape == (num_tokens, 96)
        assert core_args[1].shape == (num_tokens, 16)
        assert core_args[2].shape == (num_tokens, 16)
        assert core_args[3] is attn.kv_cache[0]
        assert core_args[4] is attn.kv_cache[1]
        assert core_args[5] is attn.conv1d.weight
        assert core_args[6] is attn.conv1d.bias
        assert core_args[7] is attn.A_log
        assert core_args[8] is attn.dt_bias
        assert torch.all(
            core_args[9] == torch.tensor([5, 7], dtype=torch.int32))
        assert core_args[10] is mock_attn_metadata.query_start_loc
        assert core_args[11] is mock_attn_metadata.request_distribution

        attn.norm.assert_called_once()
        # Verify unpacked z is natively used
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        assert torch.all(output[:num_tokens] == 5)
        assert torch.all(output[num_tokens:] == 0)

    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_uses_compact_mamba_state_indices(
            self, mock_get_forward_context):
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
        output = torch.zeros(5, 64)

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
        mock_attn_metadata.mamba_state_indices = torch.tensor(
            [3, 1], dtype=torch.int32)
        mock_attn_metadata.query_start_loc = torch.zeros(2)
        mock_attn_metadata.request_distribution = torch.zeros(3)
        # Non-spec path: the runner leaves these unset (None) unless
        # speculative decoding is active.
        mock_attn_metadata.mamba_request_distribution = None
        mock_attn_metadata.mamba_slot_read_offsets = None
        mock_fc.attn_metadata = {"test_layer": mock_attn_metadata}
        mock_get_forward_context.return_value = mock_fc

        attn.forward(hidden_states, output)

        assert attn.gdn_op.call_count == 1
        core_args = attn.gdn_op.call_args[0]
        assert torch.all(
            core_args[9] == torch.tensor([3, 1], dtype=torch.int32))

    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
    )
    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_pcp_rank",
        return_value=1)
    def test_forward_uses_native_gdn_pcp_op_for_pcp_prefill(
            self, _mock_rank, mock_get_forward_context):
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
        attn.in_proj_ba.return_value = (torch.cat((b_local, a_local),
                                                  dim=-1), None)
        attn.norm.side_effect = lambda core, _z: core
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)
        native_output = torch.stack([
            torch.full((64, ), 10.0),
            torch.full((64, ), 11.0),
            torch.full((64, ), 12.0),
            torch.full((64, ), 13.0),
        ])
        attn.gdn_pcp_op.return_value = native_output

        mock_fc = MagicMock()
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.seq_lens = torch.tensor([4], dtype=torch.int32)
        mock_attn_metadata.block_tables = torch.tensor([5, 6, 7, 8])
        mock_attn_metadata.mamba_state_indices = torch.tensor(
            [3, 1], dtype=torch.int32)
        mock_attn_metadata.sequence_layout_kind = (
            SequenceLayoutKind.PARTIAL.value)
        mock_attn_metadata.sequence_layout_protocol = "pcp_streaming"
        mock_attn_metadata.query_start_loc = torch.tensor([0, 4],
                                                          dtype=torch.int32)
        mock_attn_metadata.request_distribution = torch.zeros(3)
        # Non-spec path: the runner leaves these unset (None) unless
        # speculative decoding is active.
        mock_attn_metadata.mamba_request_distribution = None
        mock_attn_metadata.mamba_slot_read_offsets = None
        mock_fc.attn_metadata = {"test_layer": mock_attn_metadata}
        mock_get_forward_context.return_value = mock_fc

        with set_vllm_model_wrapper_context(
                mesh=_mesh(),
                vllm_config=_vllm_config(pcp_size=2, interleave_size=1)):
            attn.forward(hidden_states, output)

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
            [torch.full((4, 16), 12.0),
             torch.full((4, 16), 13.0)])
        assert torch.equal(attn.norm.call_args[0][0], expected_local)
        assert torch.all(output[:num_tokens] == 5)
        assert torch.all(output[num_tokens:] == 0)

    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_pcp_prefill_requires_initialized_pcp_op(
            self, mock_get_forward_context):
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
        output = torch.zeros(5, 64)

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
        mock_attn_metadata.mamba_state_indices = torch.tensor(
            [3, 1], dtype=torch.int32)
        mock_attn_metadata.sequence_layout_kind = (
            SequenceLayoutKind.PARTIAL.value)
        mock_attn_metadata.sequence_layout_protocol = "pcp_streaming"
        mock_attn_metadata.query_start_loc = torch.zeros(2)
        mock_attn_metadata.request_distribution = torch.zeros(3)
        # Non-spec path: the runner leaves these unset (None) unless
        # speculative decoding is active.
        mock_attn_metadata.mamba_request_distribution = None
        mock_attn_metadata.mamba_slot_read_offsets = None
        mock_fc.attn_metadata = {"test_layer": mock_attn_metadata}
        mock_get_forward_context.return_value = mock_fc

        with set_vllm_model_wrapper_context(mesh=_mesh(),
                                            vllm_config=_vllm_config()):
            with pytest.raises(RuntimeError,
                               match="GDN PCP prefill op was not initialized"):
                attn.forward(hidden_states, output)

        attn.gdn_op.assert_not_called()

    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
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

        attn.forward(hidden_states, output)

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
        assert torch.all(
            core_args[9] == torch.tensor([5, 7], dtype=torch.int32))
        assert core_args[10] is mock_attn_metadata.query_start_loc
        assert core_args[11] is mock_attn_metadata.request_distribution

        attn.norm.assert_called_once()
        # Verify z was split and reshaped correctly
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        assert torch.all(output[:num_tokens] == 5)
        assert torch.all(output[num_tokens:] == 0)
