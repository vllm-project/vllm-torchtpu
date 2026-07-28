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
from unittest.mock import MagicMock, patch

import torch

from vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op import \
    VllmGatedDeltaNetAttention


def _mesh():
    return SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})


def _vllm_config(*,
                 pcp_size: int = 1,
                 interleave_size: int = 16,
                 mamba_page_size_padded: int | None = None):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            cp_kv_cache_interleave_size=interleave_size,
            data_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
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

    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_pcp_world_size",
        return_value=8)
    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_or_create_pcp_mesh"
    )
    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_cuda_lora(self, mock_get_forward_context):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
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
    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
    )
    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_pcp_rank",
        return_value=1)
    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
    )
    @patch(
        "vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_forward_cuda_non_lora_no_gqa(self, mock_get_forward_context):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
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
