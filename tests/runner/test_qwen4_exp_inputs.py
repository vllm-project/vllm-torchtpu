# SPDX-License-Identifier: Apache-2.0
"""Input contracts: history rows, persistent slots and native pages differ."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_torchtpu.models.vllm.qwen4_exp.inputs import (
    Qwen4ExpAdapter, dummy_query_lengths, ngram_context_from_store)
from vllm_torchtpu.runner.model_adapters import (ModelInputSnapshot,
                                                 select_model_adapter)


def test_reordered_history_slots_and_group_tables_are_independent():
    store = torch.tensor(
        [[90, 91, 92, 93], [20, 21, 22, 23], [30, 31, 32, 33]],
        dtype=torch.int32)
    ctx = NS(query_start_loc=torch.tensor([0, 1, 4, 4], dtype=torch.int32),
             seq_lens=torch.tensor([4, 4, 0], dtype=torch.int32),
             mamba_state_indices=torch.tensor([7, 2, 0], dtype=torch.int32))
    runner = NS(device='cpu', _attn_metadata_builder_ctx=ctx)
    adapter = Qwen4ExpAdapter(runner)
    metadata = {
        'model.layers.3.self_attn':
        NS(block_tables=torch.tensor([11, 12, 21, 22, 0, 0])),
        'model.layers.3.self_attn.compressed_k':
        NS(block_tables=torch.tensor([31, 32, 41, 42, 0, 0]))
    }
    actual = adapter.prepare_inputs(
        ModelInputSnapshot(store, 0, 2, 8, metadata))
    assert actual['ngram_context'].tolist() == [[91, 92], [248044, 20],
                                                [248044, 248044]]
    assert actual['state_slots'].tolist() == [7, 2, -1]
    assert actual['main_block_table'].tolist()[0] == [11, 12]
    assert actual['compressed_block_table'].tolist()[0] == [31, 32]
    assert actual['valid_tokens'].tolist() == [True] * 4 + [False] * 4
    assert actual['query_start_loc'] is ctx.query_start_loc


@pytest.mark.parametrize('tokens,rows,limit', [(128, 8, 1), (128, 8, 8),
                                               (1, 8, 8), (8, 8, 8),
                                               (128, 64, 64)])
def test_dummy_geometry_exercises_every_token(tokens, rows, limit):
    lengths = dummy_query_lengths(tokens, rows, limit)
    assert len(lengths) == rows
    assert sum(lengths) == tokens
    assert sum(x > 0 for x in lengths) <= limit
    assert lengths == sorted(lengths[:min(tokens, rows, limit)]
                             ) + lengths[min(tokens, rows, limit):]


def test_reset_history_does_not_read_recycled_row():
    store = torch.tensor([[199, 198, 197], [42, 248044, 43]],
                         dtype=torch.int32)
    history = ngram_context_from_store(store, torch.tensor([1, 0]),
                                       torch.tensor([3, 0]))
    assert history.tolist() == [[248044, 43], [248044, 248044]]


def test_old_model_has_no_adapter():
    assert select_model_adapter(
        NS(model_config=NS(architecture='Qwen3ForCausalLM'))) is None


def test_profile_binds_state_and_uses_serving_dtypes_and_full_work(
        monkeypatch):
    from contextlib import nullcontext

    from vllm_torchtpu.models.vllm.qwen4_exp import inputs

    captured = {}

    def context(metadata, *args, **kwargs):
        captured['metadata'] = metadata
        return nullcontext()

    def forward(**kwargs):
        captured.update(kwargs)
        return torch.zeros((128, 2560)), None

    monkeypatch.setattr(inputs, 'set_forward_context', context)
    monkeypatch.setattr(inputs, 'set_vllm_model_wrapper_context',
                        lambda **kw: nullcontext())
    monkeypatch.setattr(inputs, 'synchronize_tensors', lambda tensor: None)
    runner = NS(device='cpu',
                scheduler_config=NS(max_num_seqs=1),
                input_batch=NS(num_reqs=0,
                               token_ids_cpu_tensor=torch.zeros(
                                   (8, 4096), dtype=torch.int32)),
                max_model_len=4096,
                vllm_config=NS(),
                mesh=None,
                forward_model=forward)
    adapter = Qwen4ExpAdapter(runner)
    allocations = []
    adapter.cache = NS(allocate=lambda: allocations.append(True),
                       pages=44,
                       specs={
                           'gdn': None,
                           'qsa': None,
                           'ple': None
                       })
    adapter.dummy_run(128, 8)
    kwargs = captured['model_kwargs']
    assert allocations == [True]
    assert captured['positions'].dtype == torch.int32
    assert captured['input_ids'].dtype == torch.int32
    assert kwargs['valid_tokens'].sum().item() == 128
    assert kwargs['query_start_loc'].tolist() == [0] + [128] * 8
    assert captured['metadata']['gdn'].query_start_loc is kwargs[
        'query_start_loc']
    assert captured['metadata']['ple'].seq_lens.tolist() == [128] + [0] * 7
    assert kwargs['state_slots'].tolist() == [1] + [-1] * 7
    assert kwargs['main_block_table'].max().item() == 1
    assert adapter.token_store.shape == (8, 4096)
    assert kwargs['ngram_context'].tolist() == [[248044, 248044]] * 8
    runner.input_batch.num_reqs = 1
    with pytest.raises(AssertionError, match='no live requests'):
        adapter.dummy_run(128, 8)
