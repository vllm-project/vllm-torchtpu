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
from unittest import mock

import numpy as np
import pytest
import torch
import torch._dynamo

from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.eagle3 import DraftChunkInputs


def _make_proposer(draft_tp: int | None = 1) -> DFlashProposer:
    hf_config_mock = mock.MagicMock()
    hf_config_mock.to_dict.return_value = {
        "dflash_config": {
            "mask_token_id": 0
        }
    }

    speculative_config = SimpleNamespace(
        num_speculative_tokens=4,
        draft_tensor_parallel_size=draft_tp,
        draft_model_config=SimpleNamespace(hf_config=hf_config_mock))
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        compilation_config=SimpleNamespace(
            static_forward_context={
                "model.layers.0.self_attn": mock.MagicMock(),
                "model.layers.1.self_attn": mock.MagicMock(),
            },
            fast_moe_cold_start=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
            is_moe_model=False,
            use_sequence_parallel_moe=False,
        ),
    )
    runner = mock.MagicMock()
    runner.device = torch.device("cpu")
    return DFlashProposer(runner=runner, vllm_config=vllm_config)


def test_propose_empty_batch():
    proposer = _make_proposer(draft_tp=1)
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=0))
    assert proposer.propose([], [], None, None) == []


def test_tpu_precompute_context_kv(device):
    # Setup dimension sizes
    L = 2  # layers
    nkv = 2  # KV heads
    hd = 8  # head dim
    target_hidden_size = 32
    num_ctx = 3  # token count

    proposer = _make_proposer(draft_tp=1)

    # Create random hidden states and positions
    hidden_states = torch.randn((num_ctx, target_hidden_size),
                                dtype=torch.float32,
                                device=device)
    positions = torch.tensor([10, 11, 12], dtype=torch.int32, device=device)

    # Create draft model mock
    draft_model = mock.MagicMock()
    del draft_model.combine_hidden_states
    del draft_model.fc
    self_model = mock.MagicMock()

    # Initialize fused weight matrix
    fused_weight = torch.randn((L * 2 * nkv * hd, target_hidden_size),
                               dtype=torch.float32,
                               device=device)
    self_model._fused_kv_weight = fused_weight
    del self_model.fc
    del self_model.hidden_norm
    self_model._fused_kv_bias = None

    # Split the fused weight to mock individual layers eagerly
    # all_kv shape: [num_ctx, L, 2, nkv, hd]
    # Weight shape: [L, 2, nkv, hd, target_hidden_size]
    weight_structured = fused_weight.view(L, 2, nkv, hd, target_hidden_size)

    # Define layers and their proj weights
    layers = []
    for i in range(L):
        layer = mock.MagicMock()
        layer.self_attn.head_dim = hd
        layer.self_attn.num_kv_heads = nkv
        del layer.self_attn.k_norm

        # Extract individual weights for assertion comparison
        k_w = weight_structured[i, 0].reshape(nkv * hd, target_hidden_size)
        v_w = weight_structured[i, 1].reshape(nkv * hd, target_hidden_size)

        layer.self_attn.k_proj = torch.nn.Linear(target_hidden_size,
                                                 nkv * hd,
                                                 bias=False).to(device)
        layer.self_attn.k_proj.weight.data.copy_(k_w)

        layer.self_attn.v_proj = torch.nn.Linear(target_hidden_size,
                                                 nkv * hd,
                                                 bias=False).to(device)
        layer.self_attn.v_proj.weight.data.copy_(v_w)

        layers.append(layer)

    self_model.layers = layers

    # Mock rotary_emb: adds positions to keys to verify they map 1-to-1
    def mock_rotary_emb(positions_rep, dummy_q, keys_flat):
        # positions_rep has shape [L * num_ctx]
        # keys_flat has shape [L * num_ctx, nkv, hd]
        roped_k = keys_flat + positions_rep.view(-1, 1, 1).to(keys_flat.dtype)
        return dummy_q, roped_k

    self_model.layers[0].self_attn.rotary_emb = mock_rotary_emb

    draft_model.model = self_model
    proposer.draft_model = draft_model

    # Initialize mock forward implementations on the layers to record outputs
    for layer in layers:
        layer.self_attn.attn.impl = mock.MagicMock()
        layer.self_attn.attn.impl.forward = mock.MagicMock(
            side_effect=lambda **kwargs: kwargs["key"])

    # 1. Run compiled/batched math eagerly inside test suite
    fn_eager = torch._dynamo.disable(
        DFlashProposer._tpu_precompute_and_update_kv_cache)
    dummy_md = mock.MagicMock()
    fn_eager(proposer, hidden_states, positions, tuple([dummy_md] * L))

    # Extract roped keys and values from the mocked layer forward calls
    roped_k_list = []
    all_v_list = []
    for layer in layers:
        call_kwargs = layer.self_attn.attn.impl.forward.call_args[1]
        roped_k_list.append(call_kwargs["key"])
        all_v_list.append(call_kwargs["value"])
    roped_k_all = torch.stack(roped_k_list, dim=0)
    all_v = torch.stack(all_v_list, dim=0)

    # 2. Compute expected values layer-by-layer eagerly
    expected_ks = []
    expected_vs = []

    for i in range(L):
        layer = self_model.layers[i]
        # Project target hidden to layer keys and values
        k_proj_out = layer.self_attn.k_proj(hidden_states).view(
            num_ctx, nkv, hd)
        v_proj_out = layer.self_attn.v_proj(hidden_states).view(
            num_ctx, nkv, hd)

        # Apply RoPE layer-by-layer
        dummy_q_i = torch.zeros_like(k_proj_out)
        _, k_roped_i = mock_rotary_emb(positions, dummy_q_i, k_proj_out)

        expected_ks.append(k_roped_i)
        expected_vs.append(v_proj_out)

    expected_k_all = torch.stack(expected_ks, dim=0)
    expected_v_all = torch.stack(expected_vs, dim=0)

    # 3. Assert exact equality
    assert torch.allclose(roped_k_all, expected_k_all, atol=1e-5)
    assert torch.allclose(all_v, expected_v_all, atol=1e-5)


def _make_chunk(num_reqs,
                start_index,
                device,
                query_start_loc_np,
                position_ids,
                aux_hidden_states=None,
                seq_lens=None):
    query_start_loc = torch.from_numpy(query_start_loc_np).to(
        device) if query_start_loc_np is not None else None

    return DraftChunkInputs(
        input_ids=None,
        position_ids=position_ids,
        query_start_loc_np=query_start_loc_np,
        attn_ctx=SimpleNamespace(seq_lens=seq_lens,
                                 query_start_loc=query_start_loc,
                                 use_max_model_len=True,
                                 position_ids_override=None),
        start_index=start_index,
        num_reqs=num_reqs,
        aux_hidden_states=aux_hidden_states,
        attn_metadata={
            "dummy_layer":
            SimpleNamespace(block_tables=torch.ones(10,
                                                    dtype=torch.int32,
                                                    device=device),
                            seq_lens=seq_lens)
        },
    )


def test_prepare_dflash_inputs(device):
    proposer = _make_proposer(draft_tp=1)
    K = 3
    proposer.speculative_config.num_speculative_tokens = K
    proposer.runner = SimpleNamespace(
        device=device,
        num_tokens_paddings=[8, 16, 32],
    )
    chunk = _make_chunk(num_reqs=2,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 3, 5], dtype=np.int32),
                        position_ids=torch.tensor(
                            [10, 11, 12, 40, 41, 0, 0, 0],
                            dtype=torch.int32,
                            device=device))

    next_tokens_device = torch.tensor(
        [[101, 102, 103, -1], [201, 202, -1, -1]],
        dtype=torch.int32,
        device=device)

    input_ids, position_ids, seq_lens = proposer._prepare_dflash_inputs(
        chunk, next_tokens_device=next_tokens_device)

    assert input_ids.shape == (8, )
    assert position_ids.shape == (8, )
    assert torch.equal(
        input_ids,
        torch.tensor([103, 0, 0, 0, 202, 0, 0, 0],
                     dtype=torch.int32,
                     device=device))
    assert torch.equal(
        position_ids,
        torch.tensor([13, 14, 15, 16, 42, 43, 44, 45],
                     dtype=torch.int32,
                     device=device))


def test_update_draft_kv_cache_from_target_unit(device):
    proposer = _make_proposer(draft_tp=1)
    proposer._tpu_precompute_and_update_kv_cache = mock.MagicMock(
        return_value=(None, []))

    slot_mapping = torch.zeros(8, dtype=torch.int32, device=device)
    proposer._compute_slot_mapping = mock.MagicMock(return_value=slot_mapping)
    proposer._draft_attn_layer_names = {"dummy_layer"}
    proposer.runner = SimpleNamespace(
        device=device,
        kv_caches=[
            torch.zeros((1, 8, 2, 2, 16), dtype=torch.float32, device=device)
        ],
        block_size=8,
        num_tokens_paddings=[8, 16, 32],
        mesh=None,
    )

    draft_model = mock.MagicMock()
    layer_mock1 = mock.MagicMock()
    layer_mock1.self_attn.attn.impl = mock.MagicMock()
    layer_mock2 = mock.MagicMock()
    layer_mock2.self_attn.attn.impl = mock.MagicMock()
    # Draft layers resolve their metadata by name; an unnamed layer is an
    # error, not a cue to borrow another group's block table.
    layer_mock1.self_attn.attn.layer_name = "dummy_layer"
    layer_mock2.self_attn.attn.layer_name = "dummy_layer"
    draft_model.model.layers = [layer_mock1, layer_mock2]
    proposer.draft_model = draft_model
    proposer.num_target_layers = 2

    chunk = _make_chunk(num_reqs=1,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 3]),
                        position_ids=torch.zeros(3),
                        aux_hidden_states=torch.zeros((3, 32),
                                                      dtype=torch.float32,
                                                      device=device))

    proposer._update_draft_kv_cache_from_target(chunk)

    proposer._tpu_precompute_and_update_kv_cache.assert_called_once()


def test_propose_unit(device):
    proposer = _make_proposer(draft_tp=1)
    K = 3
    proposer.speculative_config.num_speculative_tokens = K
    proposer._draft_attn_layer_names = {"dummy_layer"}
    proposer.runner = SimpleNamespace(
        device=device,
        input_batch=SimpleNamespace(num_reqs=2),
        _attn_metadata_builder_ctx=None,
        max_num_blocks_per_req=2,
        _attn_layer_names={"dummy_layer"},
        empty_slot_mappings=torch.zeros(128, dtype=torch.int32, device=device),
        _build_attention_metadata=mock.MagicMock(return_value=({}, None)),
        mesh=None,
        _is_async_drafter=True,
    )

    chunk = _make_chunk(num_reqs=2,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 2, 4]),
                        position_ids=torch.zeros(4))
    proposer.draft_chunks = [chunk]

    proposer._prepare_dflash_inputs = mock.MagicMock(
        return_value=(torch.zeros(8, dtype=torch.int32, device=device),
                      torch.zeros(8, dtype=torch.int32, device=device),
                      torch.zeros(2, dtype=torch.int32, device=device)))
    proposer._update_draft_kv_cache_from_target = mock.MagicMock()

    logits = torch.zeros((8, 1000), dtype=torch.float32, device=device)
    logits[1, 500] = 100.0
    logits[2, 501] = 100.0
    logits[3, 502] = 100.0
    logits[5, 600] = 100.0
    logits[6, 601] = 100.0
    logits[7, 602] = 100.0

    expected_tokens = torch.tensor([[500, 501, 502], [600, 601, 602]],
                                   device=device)
    proposer._dflash_forward_and_sample = mock.MagicMock(
        return_value=(expected_tokens, torch.zeros(8, 32, device=device)))

    draft_tokens = proposer.propose(
        sampled_token_ids=[[10], [20]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=np.array([0, 0]),
        scheduler_output=None,
        next_tokens_per_chunk=[
            torch.zeros((2, 4), dtype=torch.int32, device=device)
        ],
    )

    assert draft_tokens == [[500, 501, 502], [600, 601, 602]]
    proposer._prepare_dflash_inputs.assert_called_once()
    proposer._update_draft_kv_cache_from_target.assert_called_once()
    proposer._dflash_forward_and_sample.assert_called_once()


def test_build_draft_layer_metadata_rejects_an_unresolved_layer():
    """An unnamed draft layer must raise, not borrow another group's metadata.

    `chunk.attn_metadata` is the target's full per-layer dict, so falling back
    to an arbitrary entry can hand a draft layer a mamba group's block table —
    a different block size indexing different blocks. That corrupts the draft
    KV cache silently instead of failing.
    """
    proposer = _make_proposer(draft_tp=1)
    named = mock.MagicMock()
    named.self_attn.attn.layer_name = "dummy_layer"
    unnamed = mock.MagicMock()
    unnamed.self_attn.attn.layer_name = "not_in_the_metadata"
    draft_model = mock.MagicMock()
    draft_model.model.layers = [named, unnamed]
    proposer.draft_model = draft_model

    md = {"dummy_layer": SimpleNamespace(block_tables=None)}

    with pytest.raises(RuntimeError, match="not_in_the_metadata"):
        proposer._build_draft_layer_metadata(md)

    # The resolvable layer on its own comes back in layer order.
    draft_model.model.layers = [named]
    assert proposer._build_draft_layer_metadata(md) == (md["dummy_layer"], )
