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

from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID
from vllm_torchtpu.spec_decode import dflash as dflash_module
from vllm_torchtpu.spec_decode import utils as spec_decode_utils
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.eagle3 import DraftChunkInputs


def _make_proposer(draft_tp: int | None = 1, target_tp: int = 1) -> DFlashProposer:
    hf_config_mock = mock.MagicMock()
    hf_config_mock.to_dict.return_value = {
        "dflash_config": {
            "mask_token_id": 0,
            "target_layer_ids": [1, 2],
        }
    }

    speculative_config = SimpleNamespace(
        num_speculative_tokens=4,
        draft_tensor_parallel_size=draft_tp,
        draft_model_config=SimpleNamespace(hf_config=hf_config_mock),
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        model_config=SimpleNamespace(max_model_len=2048),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "model.layers.0.self_attn": mock.MagicMock(),
                "model.layers.1.self_attn": mock.MagicMock(),
            },
            fast_moe_cold_start=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=target_tp,
            is_moe_model=False,
            use_sequence_parallel_moe=False,
        ),
    )
    runner = mock.MagicMock()
    runner.device = torch.device("cpu")
    return DFlashProposer(runner=runner, vllm_config=vllm_config)


def _make_vocab_models():
    target_embed = torch.nn.Embedding(3, 2)
    target_embed.org_vocab_size = 3
    draft_embed = torch.nn.Embedding(3, 2)
    draft_embed.org_vocab_size = 3
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed),
        lm_head=torch.nn.Linear(2, 3, bias=False),
        set_aux_hidden_state_layers=lambda _: None,
    )
    draft = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=draft_embed, layers=[]),
        lm_head=torch.nn.Linear(2, 3, bias=False),
        get_draft_attn_causal=lambda: [],
        logits_processor=SimpleNamespace(_gather_logits=lambda logits: logits.clone()),
        has_own_embed_tokens=False,
        has_own_lm_head=False,
    )
    return draft, target


def _fake_gather_sharded_weight(module):
    weight = module.weight.detach().clone()
    return weight, weight.shape[0], weight.shape[1], weight.device


@pytest.mark.parametrize("draft_tp,target_tp", [(1, 1), (1, 8), (8, 8)])
def test_load_model_vocab_weights_follow_tp_layout(draft_tp, target_tp, monkeypatch):
    proposer = _make_proposer(draft_tp=draft_tp, target_tp=target_tp)
    draft, target = _make_vocab_models()

    def load_draft_model():
        proposer.draft_model = draft

    proposer._load_draft_model = load_draft_model
    monkeypatch.setattr(
        spec_decode_utils, "gather_sharded_weight", _fake_gather_sharded_weight
    )

    with mock.patch(
        "vllm_torchtpu.spec_decode.dflash.get_layers_from_vllm_config", return_value={}
    ):
        proposer.load_model(target)

    if draft_tp == target_tp:
        assert draft.model.embed_tokens is target.model.embed_tokens
        assert draft.lm_head is target.lm_head
    else:
        assert draft.model.embed_tokens is not target.model.embed_tokens
        assert torch.equal(
            draft.model.embed_tokens.weight, target.model.embed_tokens.weight
        )
        assert draft.lm_head is not target.lm_head
        assert torch.equal(draft.lm_head.weight, target.lm_head.weight)


def test_propose_empty_batch():
    proposer = _make_proposer(draft_tp=1)
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=0))
    assert proposer.propose([], [], None, None) == []


def test_precompile_skips_sub_block_draft_forward():
    proposer = _make_proposer(draft_tp=8, target_tp=8)
    proposer.speculative_config.num_speculative_tokens = 7
    proposer.draft_model = SimpleNamespace(
        model=SimpleNamespace(
            context_proj=SimpleNamespace(weight=torch.empty((1, 32))),
            embed_tokens=SimpleNamespace(weight=torch.empty((1, 1))),
        )
    )
    proposer.runner = SimpleNamespace(
        num_reqs_max_model_len=8,
        num_reqs_most_model_len=None,
        most_model_len=None,
        max_num_blocks_per_req=4,
        num_tokens_paddings=[1, 8, 16, 512],
        _precompile_timed=mock.MagicMock(
            return_value=mock.MagicMock(
                __enter__=mock.MagicMock(return_value=None),
                __exit__=mock.MagicMock(return_value=False),
            )
        ),
    )
    proposer._dummy_precompute_and_update_kv_cache = mock.MagicMock()
    proposer._dummy_draft_forward = mock.MagicMock()

    proposer.precompile()

    assert [
        call.kwargs["num_tokens"]
        for call in proposer._dummy_precompute_and_update_kv_cache.call_args_list
    ] == [1, 8, 16, 512]
    # 1 is skipped (sub-block); 512 stays: _get_padded_len rounds the
    # 8-req * 8-token upper bound (64) up to the next runner bucket.
    assert [
        call.kwargs["num_tokens"]
        for call in proposer._dummy_draft_forward.call_args_list
    ] == [8, 16, 512]


@pytest.mark.tpu_test
@pytest.mark.parametrize("has_upstream_hook", [False, True])
def test_tpu_precompute_context_kv(device, has_upstream_hook):
    # Setup dimension sizes
    L = 2  # layers
    nkv = 2  # KV heads
    hd = 8  # head dim
    target_hidden_size = 32
    num_ctx = 3  # token count

    proposer = _make_proposer(draft_tp=1)

    # Create random hidden states and positions
    hidden_states = torch.randn(
        (num_ctx, target_hidden_size), dtype=torch.float32, device=device
    )
    positions = torch.tensor([10, 11, 12], dtype=torch.int32, device=device)

    # Create draft model mock
    self_model = mock.MagicMock()

    # Initialize fused weight matrix
    fused_weight = torch.randn(
        (L * 2 * nkv * hd, target_hidden_size), dtype=torch.float32, device=device
    )
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
        layer.self_attn.num_heads = nkv
        del layer.self_attn.k_norm

        # Extract individual weights for assertion comparison
        k_w = weight_structured[i, 0].reshape(nkv * hd, target_hidden_size)
        v_w = weight_structured[i, 1].reshape(nkv * hd, target_hidden_size)

        layer.self_attn.k_proj = torch.nn.Linear(
            target_hidden_size, nkv * hd, bias=False
        ).to(device)
        layer.self_attn.k_proj.weight.data.copy_(k_w)

        layer.self_attn.v_proj = torch.nn.Linear(
            target_hidden_size, nkv * hd, bias=False
        ).to(device)
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

    proposer.draft_model = SimpleNamespace(model=self_model)
    upstream_hook = mock.Mock(
        side_effect=AssertionError("The upstream context-KV hook must not run on TPU")
    )
    if has_upstream_hook:
        proposer.draft_model.precompute_and_store_context_kv = upstream_hook

    # Initialize mock forward implementations on the layers to record outputs
    for layer in layers:
        layer.self_attn.attn.impl = mock.MagicMock()
        layer.self_attn.attn.impl.forward = mock.MagicMock(
            side_effect=lambda **kwargs: kwargs["key"]
        )

    # 1. Run compiled/batched math eagerly inside test suite
    fn_eager = torch._dynamo.disable(DFlashProposer._tpu_precompute_and_update_kv_cache)
    dummy_md = mock.MagicMock()
    fn_eager(proposer, hidden_states, positions, tuple([dummy_md] * L))
    upstream_hook.assert_not_called()

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
        k_proj_out = layer.self_attn.k_proj(hidden_states).view(num_ctx, nkv, hd)
        v_proj_out = layer.self_attn.v_proj(hidden_states).view(num_ctx, nkv, hd)

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


@pytest.mark.tpu_test
def test_tpu_precompute_context_kv_uses_explicit_tpu_hook(device):
    proposer = _make_proposer(draft_tp=1)
    hidden_states = torch.randn(3, 32, device=device)
    positions = torch.arange(3, device=device)
    metadata = (SimpleNamespace(),)
    expected = torch.randn(3, 16, device=device)
    tpu_hook = mock.Mock(return_value=expected)
    upstream_hook = mock.Mock(
        side_effect=AssertionError("The upstream context-KV hook must not run on TPU")
    )
    proposer.draft_model = SimpleNamespace(
        model=SimpleNamespace(),
        tpu_precompute_and_store_context_kv=tpu_hook,
        precompute_and_store_context_kv=upstream_hook,
    )

    fn_eager = torch._dynamo.disable(DFlashProposer._tpu_precompute_and_update_kv_cache)
    result = fn_eager(proposer, hidden_states, positions, metadata)

    assert result is expected
    tpu_hook.assert_called_once_with(hidden_states, positions, metadata)
    upstream_hook.assert_not_called()


def _make_chunk(
    num_reqs,
    start_index,
    device,
    query_start_loc_np,
    position_ids,
    aux_hidden_states=None,
    seq_lens=None,
):
    query_start_loc = (
        torch.from_numpy(query_start_loc_np).to(device)
        if query_start_loc_np is not None
        else None
    )

    return DraftChunkInputs(
        input_ids=None,
        position_ids=position_ids,
        query_start_loc_np=query_start_loc_np,
        attn_ctx=SimpleNamespace(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            use_max_model_len=True,
            position_ids_override=None,
            # [decode_end, prefill_end, num_reqs]: the compiled input builders
            # read the live request count from its last entry on device.
            request_distribution=torch.tensor(
                [0, 0, num_reqs], dtype=torch.int32, device=device
            ),
        ),
        start_index=start_index,
        num_reqs=num_reqs,
        aux_hidden_states=aux_hidden_states,
        attn_metadata={
            "dummy_layer": SimpleNamespace(
                block_tables=torch.ones(10, dtype=torch.int32, device=device),
                seq_lens=seq_lens,
            )
        },
    )


@pytest.mark.tpu_test
def test_prepare_dflash_inputs(device):
    proposer = _make_proposer(draft_tp=1)
    K = 3
    proposer.speculative_config.num_speculative_tokens = K
    proposer.runner = SimpleNamespace(
        device=device,
        num_tokens_paddings=[8, 16, 32],
        max_num_reqs=8,
        _dp_lockstep_enabled=lambda: False,
    )
    chunk = _make_chunk(
        num_reqs=2,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 3, 5], dtype=np.int32),
        position_ids=torch.tensor(
            [10, 11, 12, 40, 41, 0, 0, 0], dtype=torch.int32, device=device
        ),
    )

    next_tokens_device = torch.tensor(
        [[101, 102, 103, -1], [201, 202, -1, -1]], dtype=torch.int32, device=device
    )

    input_ids, position_ids, seq_lens = proposer._prepare_dflash_inputs(
        chunk, next_tokens_device=next_tokens_device
    )

    assert input_ids.shape == (8,)
    assert position_ids.shape == (8,)
    assert torch.equal(
        input_ids,
        torch.tensor([103, 0, 0, 0, 202, 0, 0, 0], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        position_ids,
        torch.tensor(
            [13, 14, 15, 16, 42, 43, 44, 45], dtype=torch.int32, device=device
        ),
    )


@pytest.mark.tpu_test
def test_update_draft_kv_cache_from_target_unit(device):
    proposer = _make_proposer(draft_tp=1)
    proposer._tpu_precompute_and_update_kv_cache = mock.MagicMock(
        return_value=(None, [])
    )

    slot_mapping = torch.zeros(8, dtype=torch.int32, device=device)
    proposer._compute_slot_mapping = mock.MagicMock(return_value=slot_mapping)
    proposer._draft_attn_layer_names = {"dummy_layer"}
    proposer.runner = SimpleNamespace(
        device=device,
        kv_caches=[torch.zeros((1, 8, 2, 2, 16), dtype=torch.float32, device=device)],
        block_size=8,
        num_tokens_paddings=[8, 16, 32],
        mesh=None,
        _dp_lockstep_enabled=lambda: False,
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

    chunk = _make_chunk(
        num_reqs=1,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 3]),
        position_ids=torch.zeros(3),
        aux_hidden_states=torch.zeros((3, 32), dtype=torch.float32, device=device),
    )

    proposer._update_draft_kv_cache_from_target(chunk)

    proposer._tpu_precompute_and_update_kv_cache.assert_called_once()


@pytest.mark.tpu_test
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
        build_attention_metadata_for_layers=mock.MagicMock(return_value={}),
        mesh=None,
        _is_async_drafter=True,
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda num_tokens: None,
        _dp_step_num_chunks=0,
    )

    chunk = _make_chunk(
        num_reqs=2,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 2, 4]),
        position_ids=torch.zeros(4),
    )
    proposer.draft_chunks = [chunk]

    proposer._prepare_dflash_inputs = mock.MagicMock(
        return_value=(
            torch.zeros(8, dtype=torch.int32, device=device),
            torch.zeros(8, dtype=torch.int32, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
        )
    )
    proposer._update_draft_kv_cache_from_target = mock.MagicMock()

    logits = torch.zeros((8, 1000), dtype=torch.float32, device=device)
    logits[1, 500] = 100.0
    logits[2, 501] = 100.0
    logits[3, 502] = 100.0
    logits[5, 600] = 100.0
    logits[6, 601] = 100.0
    logits[7, 602] = 100.0

    expected_tokens = torch.tensor([[500, 501, 502], [600, 601, 602]], device=device)
    proposer._dflash_forward_and_sample = mock.MagicMock(
        return_value=(expected_tokens, torch.zeros(8, 32, device=device))
    )

    draft_tokens = proposer.propose(
        sampled_token_ids=[[10], [20]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=np.array([0, 0]),
        scheduler_output=None,
        next_tokens_per_chunk=[torch.zeros((2, 4), dtype=torch.int32, device=device)],
    )

    assert draft_tokens == [[500, 501, 502], [600, 601, 602]]
    proposer._prepare_dflash_inputs.assert_called_once()
    proposer._update_draft_kv_cache_from_target.assert_called_once()
    proposer._dflash_forward_and_sample.assert_called_once()


def _padding_mask_at_draft_forward(proposer, run, num_rows: int) -> list[bool]:
    """Run `run()` with the padding mask left in a target step's state and
    return the mask the draft forward saw."""
    state = TokenPaddingState.create(32, torch.device("cpu"))
    # A target step with 3 real rows of 16: stale for any draft forward.
    state.update(3, 16)
    seen = []

    def _forward(input_ids, positions, block_size):
        seen.append(state.get_local_padding_mask(num_rows).tolist())
        num_reqs = input_ids.shape[0] // block_size
        return torch.zeros((num_reqs, block_size - 1), dtype=torch.int64), None

    proposer._dflash_forward_and_sample = _forward
    token_padding.set_padding_state(state)
    try:
        run()
    finally:
        token_padding.set_padding_state(None)
    assert len(seen) == 1
    return seen[0]


def test_propose_marks_padding_for_draft_rows():
    proposer = _make_proposer(draft_tp=1)
    proposer.speculative_config.num_speculative_tokens = 3
    proposer.runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(num_reqs=2),
        mesh=None,
        max_num_reqs=8,
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda num_tokens: None,
    )
    proposer.draft_chunks = [
        SimpleNamespace(
            num_reqs=2,
            start_index=0,
            # The live count reaches the builders as a device scalar.
            attn_ctx=SimpleNamespace(
                request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32)
            ),
        )
    ]
    proposer._update_draft_kv_cache_from_target = mock.MagicMock()
    # 2 requests x block 4 = 8 real draft rows, padded to a 16-row bucket.
    proposer._prepare_dflash_inputs = mock.MagicMock(
        return_value=(
            torch.zeros(16, dtype=torch.int32),
            torch.zeros(16, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )
    )
    proposer._build_draft_attn_metadata = mock.MagicMock(return_value={})

    mask = _padding_mask_at_draft_forward(
        proposer,
        lambda: proposer.propose(
            [], [], None, None, return_device=True, device_seed=[torch.zeros(2)]
        ),
        num_rows=16,
    )

    assert mask == [False] * 8 + [True] * 8


def test_dummy_draft_forward_marks_all_rows_padding(monkeypatch):
    proposer = _make_proposer(draft_tp=1)
    proposer.speculative_config.num_speculative_tokens = 3
    proposer.runner = SimpleNamespace(
        device=torch.device("cpu"),
        mesh=None,
        _dp_num_tokens_across_dp=lambda num_tokens: None,
    )
    proposer._build_attn_metadata_for_draft = mock.MagicMock(return_value={})
    monkeypatch.setattr(dflash_module, "synchronize_tensors", lambda *a, **k: None)

    mask = _padding_mask_at_draft_forward(
        proposer,
        lambda: proposer._dummy_draft_forward(
            num_tokens=8, use_max_model_len=True, sync=False
        ),
        num_rows=8,
    )

    assert mask == [True] * 8


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
    draft_model = SimpleNamespace(model=SimpleNamespace(layers=[named, unnamed]))
    proposer.draft_model = draft_model

    md = {"dummy_layer": SimpleNamespace(block_tables=None)}

    with pytest.raises(RuntimeError, match="not_in_the_metadata"):
        proposer._build_draft_layer_metadata(md)

    # The resolvable layer on its own comes back in layer order.
    draft_model.model.layers = [named]
    assert proposer._build_draft_layer_metadata(md) == (md["dummy_layer"],)


def test_dp_lockstep_run_dummy_draft():
    proposer = _make_proposer(draft_tp=8, target_tp=8)
    proposer.draft_model = mock.MagicMock()
    proposer.runner = SimpleNamespace(
        _dp_lockstep_enabled=lambda: True,
        _dp_target_bucket=32,
        _dp_step_max_reqs=4,
        num_reqs_max_model_len=8,
    )
    proposer._dp_dummy_kv_update = mock.MagicMock()
    proposer._dp_dummy_forward = mock.MagicMock()

    proposer.run_dp_dummy_draft(num_chunks=2)

    assert proposer._dp_dummy_kv_update.call_count == 2
    assert proposer._dp_dummy_forward.call_count == 2


@pytest.mark.tpu_test
def test_prepare_dflash_inputs_dp_lockstep(device):
    proposer = _make_proposer(draft_tp=8, target_tp=8)
    K = 3
    proposer.speculative_config.num_speculative_tokens = K
    proposer.runner = SimpleNamespace(
        device=device,
        num_tokens_paddings=[8, 16, 32],
        _dp_lockstep_enabled=lambda: True,
        _dp_step_max_reqs=4,
        max_num_reqs=8,
    )
    chunk = _make_chunk(
        num_reqs=1,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 3], dtype=np.int32),
        position_ids=torch.tensor([10, 11, 12, 0], dtype=torch.int32, device=device),
    )
    next_tokens_device = torch.tensor(
        [[101, 102, 103, -1]], dtype=torch.int32, device=device
    )

    input_ids, position_ids, _ = proposer._prepare_dflash_inputs(
        chunk, next_tokens_device=next_tokens_device
    )

    # Coordinated reqs = 4 -> 4 * 4 = 16 tokens padded.
    assert input_ids.shape == (16,)
    assert position_ids.shape == (16,)


class _FakeMoERunner(torch.nn.Module):
    pass


def _lockstep_proposer(draft_model):
    """A replicated (tp=1) draft on an EP-DP lockstep runner: the only
    configuration where the gate depends on detecting MoE in the draft."""
    proposer = _make_proposer(draft_tp=1, target_tp=1)
    proposer.draft_model = draft_model
    proposer.runner = SimpleNamespace(_dp_lockstep_enabled=lambda: True)
    return proposer


def test_dp_lockstep_sharded_detects_moe_draft_without_protocol(monkeypatch):
    """The TPU DSV4/K3 drafts hold fused-MoE runners but do not implement
    vLLM's MixtureOfExperts protocol; a replicated such draft must still
    join the lockstep, or its EP all-to-alls hang against idle ranks."""
    import vllm.model_executor.layers.fused_moe as fused_moe

    monkeypatch.setattr(fused_moe, "MoERunner", _FakeMoERunner)
    moe_draft = torch.nn.Module()
    moe_draft.layer = torch.nn.Module()
    moe_draft.layer.experts = _FakeMoERunner()
    assert _lockstep_proposer(moe_draft)._dp_lockstep_sharded() is True

    dense_draft = torch.nn.Module()
    dense_draft.layer = torch.nn.Linear(2, 2)
    assert _lockstep_proposer(dense_draft)._dp_lockstep_sharded() is False


def test_dp_lockstep_sharded_off_without_lockstep():
    proposer = _make_proposer(draft_tp=1, target_tp=1)
    proposer.draft_model = mock.MagicMock()
    proposer.runner = SimpleNamespace(_dp_lockstep_enabled=lambda: False)
    assert proposer._dp_lockstep_sharded() is False


@pytest.mark.tpu_test
@pytest.mark.parametrize("builder", ["next_tokens", "seed"])
def test_compiled_input_builders_compile_no_program_per_request_count(device, builder):
    """The builders take the live request count as a device scalar, so one
    shape bucket is one program however many requests are live."""
    if device.type != "tpu":
        pytest.skip("needs the TPU compile cache")
    K, block, rows, padded_len = 5, 5, 32, 128
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer._max_model_len = 9216
    proposer.mask_token_id = 7777
    # rows verify windows of 1 + K tokens; query_start_loc and positions.
    qsl = torch.arange(0, (rows + 1) * (1 + K), 1 + K, dtype=torch.int32).to(device)
    pos = torch.arange(rows * (1 + K), dtype=torch.int32).to(device)

    def run(num_reqs: int):
        nr = torch.tensor(num_reqs, dtype=torch.int32).to(device)
        if builder == "seed":
            seed = torch.ones(rows, dtype=torch.int32).to(device)
            return proposer._tpu_build_dflash_inputs_seed(
                seed, qsl, pos, nr, block, padded_len, 7777
            )
        next_tokens = torch.ones(rows, K + 1, dtype=torch.int32).to(device)
        draft_lengths = torch.full((rows,), K, dtype=torch.int32).to(device)
        return proposer._tpu_build_dflash_inputs_next_tokens(
            next_tokens,
            qsl,
            pos,
            draft_lengths,
            nr,
            block,
            padded_len,
            7777,
            INVALID_TOKEN_ID,
        )

    def programs() -> int:
        return len(torch.tpu._get_cache_stats().per_entry_stats)

    run(1)  # the first use of the shape may compile
    before = programs()
    for num_reqs in (2, 17, 32):
        run(num_reqs)
    assert programs() == before, (
        f"{builder} builder compiled a program per request count"
    )
