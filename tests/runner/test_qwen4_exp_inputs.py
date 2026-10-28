# SPDX-License-Identifier: Apache-2.0
"""Input contracts: history rows, persistent slots and native pages differ."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from vllm_torchtpu.models.vllm.qwen4_exp.inputs import (
    Qwen4ExpAdapter,
    dummy_query_lengths,
    ngram_context_from_store,
)
from vllm_torchtpu.runner.model_adapters import ModelInputSnapshot, select_model_adapter


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("qsa_index", [0, 3])
@pytest.mark.parametrize("pooled", [False, True])
def test_reordered_history_slots_and_group_tables_are_independent(
    monkeypatch, offload, qsa_index, pooled
):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", str(int(pooled)))
    store = torch.tensor(
        [[90, 91, 92, 93], [20, 21, 22, 23], [30, 31, 32, 33]], dtype=torch.int32
    )
    ctx = NS(
        query_start_loc=torch.tensor([0, 1, 4, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4, 4, 0], dtype=torch.int32),
        mamba_state_indices=torch.tensor([7, 2, 0], dtype=torch.int32),
    )
    runner = NS(device="cpu", _attn_metadata_builder_ctx=ctx)
    runner.model_config = NS(hf_text_config=q4exp_text_config())
    runner.cache_config = NS(block_size=128)
    runner.model_config.hf_text_config.layer_types[qsa_index] = "full_attention"
    adapter = Qwen4ExpAdapter(runner)
    device_ids = torch.tensor([999, 21, 22, 23, 0, 0, 0, 0], dtype=torch.int32)
    raw = torch.zeros((8, 2560), dtype=torch.int16)
    metadata = {
        f"model.layers.{qsa_index}.self_attn": NS(
            block_tables=torch.tensor([11, 12, 21, 22, 0, 0])
        ),
        f"model.layers.{qsa_index}.self_attn.compressed_k": NS(
            block_tables=torch.tensor([31, 32, 41, 42, 0, 0])
        ),
    }
    actual = adapter.prepare_inputs(
        ModelInputSnapshot(
            store,
            0,
            2,
            8,
            metadata,
            device_ids,
        )
    )
    assert actual["ngram_context"].tolist() == [
        [91, 92],
        [248044, 20],
        [248044, 248044],
    ]
    assert actual["state_slots"].tolist() == ([0, 1, -1] if pooled else [7, 2, -1])
    assert actual["main_block_table"].tolist()[0] == [11, 12]
    assert actual["compressed_block_table"].tolist()[0] == [31, 32]
    assert actual["valid_tokens"].tolist() == [True] * 4 + [False] * 4
    assert actual["query_start_loc"] is ctx.query_start_loc
    assert "ple_raw_embeddings" not in actual
    # The model owns lookup. Async sampling can leave the current CPU slot
    # stale, so the boundary must consume substituted device IDs and the
    # committed history, not reconstruct the current token from the store.
    from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpBackbone

    hidden = torch.zeros((8, 2))
    lookup = Mock(return_value=raw)
    layer = Mock(return_value=hidden, ple=NS(lookup_for_model=lookup))
    backbone = NS(
        host_ple_index=0 if offload else None,
        layers=[layer],
        hc_count=1,
        embed_tokens=Mock(return_value=hidden),
        hyper_connection_mixer=NS(mix=lambda hidden: (hidden, None)),
        emit_mtp_hidden=False,
    )
    Qwen4ExpBackbone.forward(
        backbone, device_ids, torch.zeros_like(device_ids), **actual
    )
    if offload:
        args = lookup.call_args.args
        assert args[0].input_ids is device_ids
        assert args[0].query_start_loc is ctx.query_start_loc
        assert args[0].ngram_context is actual["ngram_context"]
        assert args[1].tolist() == [4]
        assert layer.call_args.args[-1] is raw
    else:
        lookup.assert_not_called()
        assert layer.call_args.args[-1] is None


@pytest.mark.parametrize(
    "tokens,rows,limit",
    [
        (128, 8, 1),
        (128, 8, 8),
        (1, 8, 8),
        (8, 8, 8),
        (128, 64, 64),
        (128, 128, 128),
        (128, 256, 256),
    ],
)
def test_dummy_geometry_exercises_every_token(tokens, rows, limit):
    lengths = dummy_query_lengths(tokens, rows, limit)
    assert len(lengths) == rows
    assert sum(lengths) == tokens
    assert sum(x > 0 for x in lengths) <= limit
    assert (
        lengths
        == sorted(lengths[: min(tokens, rows, limit)])
        + lengths[min(tokens, rows, limit) :]
    )


def test_reset_history_does_not_read_recycled_row():
    store = torch.tensor([[199, 198, 197], [42, 248044, 43]], dtype=torch.int32)
    history = ngram_context_from_store(
        store,
        torch.tensor([1, 0]),
        torch.tensor([3, 0]),
        eos_token_id=248044,
        ngram_size=3,
    )
    assert history.tolist() == [[248044, 43], [248044, 248044]]


def test_old_model_has_no_adapter():
    assert (
        select_model_adapter(NS(model_config=NS(architecture="Qwen3ForCausalLM")))
        is None
    )


@pytest.mark.parametrize("tp", [1, 2])
def test_profile_binds_state_and_uses_serving_dtypes_and_full_work(monkeypatch, tp):
    from contextlib import nullcontext

    from vllm_torchtpu.models.vllm.qwen4_exp import inputs

    # This fixture supplies native owners; pooled dummy ownership is covered
    # by test_qwen4_exp_pool with the real pooled-cache adapter.
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    captured = {}
    logits = Mock(return_value=torch.zeros((8, 1)))

    def context(metadata, *args, **kwargs):
        captured["metadata"] = metadata
        captured["context_kwargs"] = kwargs
        return nullcontext()

    def forward(**kwargs):
        captured.update(kwargs)
        return torch.zeros((128, 2560)), None

    monkeypatch.setattr(inputs, "set_forward_context", context)
    monkeypatch.setattr(
        inputs, "set_vllm_model_wrapper_context", lambda **kw: nullcontext()
    )
    monkeypatch.setattr(inputs, "synchronize_tensors", lambda tensor: None)
    runner = NS(
        device="cpu",
        parallel_config=NS(tensor_parallel_size=tp),
        scheduler_config=NS(max_num_seqs=1),
        input_batch=NS(
            num_reqs=0, token_ids_cpu_tensor=torch.zeros((8, 4096), dtype=torch.int32)
        ),
        max_model_len=4096,
        vllm_config=NS(),
        mesh=None,
        _dp_num_tokens_across_dp=lambda n: torch.full((8,), n, dtype=torch.int32),
        forward_model=forward,
        compute_selected_logits=logits,
    )
    runner.model_config = NS(hf_text_config=q4exp_text_config())
    runner.cache_config = NS(block_size=128)
    adapter = Qwen4ExpAdapter(runner)
    allocations = []
    adapter.cache = NS(
        allocate=lambda: allocations.append(True),
        pages=44,
        specs={"gdn": None, "qsa": None, "ple": None},
    )
    adapter.dummy_run(128, 8)
    kwargs = captured["model_kwargs"]
    assert allocations == [True]
    assert captured["positions"].dtype == torch.int32
    assert captured["input_ids"].dtype == torch.int32
    assert kwargs["valid_tokens"].sum().item() == 128
    assert kwargs["query_start_loc"].tolist() == [0] + [128] * 8
    assert captured["metadata"]["gdn"].query_start_loc is kwargs["query_start_loc"]
    assert captured["metadata"]["ple"].seq_lens.tolist() == [128] + [0] * 7
    assert kwargs["state_slots"].tolist() == [1] + [-1] * 7
    assert kwargs["main_block_table"].max().item() == 1
    assert adapter.token_store.shape == (8, 4096)
    assert kwargs["ngram_context"].tolist() == [[248044, 248044]] * 8
    assert "ple_raw_embeddings" not in kwargs
    logits.assert_not_called()
    runner.input_batch.num_reqs = 1
    with pytest.raises(AssertionError, match="no live requests"):
        adapter.dummy_run(128, 8)
    # An idle DP may still own active requests and serve remote embedding/
    # expert work. Its synthetic step must neither rebind owners nor replace
    # the history snapshot; explicit physical counts avoid another CPU sync.
    before = adapter.token_store
    before.fill_(123)
    adapter.cache.allocate = lambda: pytest.fail("idle must not allocate/rebind cache")
    adapter.dummy_run(128, 8, dp_lockstep=True)
    # Idle TP groups must also issue the LM-head collective before returning
    # to DP coordination, or busy ranks cannot materialize their sample.
    if tp > 1:
        logits.assert_called_once()
        hidden, indices = logits.call_args.args
        assert hidden.shape == (128, 2560)
        assert indices.shape == (8,) and indices.dtype == torch.int32
        assert not indices.any()
    else:
        logits.assert_not_called()  # TP1 has no LM-head collective to pair.
    assert adapter.token_store is before
    assert (adapter.token_store == 123).all()
    kwargs = captured["model_kwargs"]
    md = captured["metadata"]["gdn"]
    assert not kwargs["valid_tokens"].any()
    assert kwargs["state_slots"].tolist() == [-1] * 8
    assert kwargs["ngram_context"].tolist() == [[248044, 248044]] * 8
    assert "ple_raw_embeddings" not in kwargs
    for value in (
        kwargs["query_start_loc"],
        kwargs["context_lens"],
        kwargs["main_block_table"],
        kwargs["compressed_block_table"],
        md.seq_lens,
        md.request_distribution,
        md.mamba_state_indices,
    ):
        assert not value.any()
    assert captured["context_kwargs"]["num_tokens_across_dp"].tolist() == [128] * 8


def test_async_history_reads_committed_prefix_after_reorder_and_row_reuse():
    # The runner substitutes the pending token at context on device. PLE's
    # history must stop BEFORE that slot, whose CPU value is still -1.
    host = torch.tensor(
        [[11, 12, 13, -1, 99], [21, 22, -1, 98, 97], [91, 92, 93, 94, 95]],
        dtype=torch.int32,
    )
    order = torch.tensor([1, 2, 0])
    runner = NS(device="cpu", input_batch=NS(token_ids_cpu_tensor=host[order]))
    runner.model_config = NS(hf_text_config=q4exp_text_config())
    runner.cache_config = NS(block_size=128)
    adapter = Qwen4ExpAdapter(runner)
    snapshot = adapter.begin_step().clone()
    context = torch.tensor([2, 0, 3])
    history = ngram_context_from_store(
        snapshot, torch.arange(3), context, eos_token_id=248044, ngram_size=3
    )
    assert history.tolist() == [[21, 22], [248044, 248044], [12, 13]]
    # End-of-step host commit followed by the next optimistic placeholder.
    # The prior sample now belongs to history, irrespective of row order.
    runner.input_batch.token_ids_cpu_tensor[0, 2:4] = torch.tensor([23, -1])
    runner.input_batch.token_ids_cpu_tensor[2, 3:5] = torch.tensor([14, -1])
    next_snapshot = adapter.begin_step().clone()
    next_history = ngram_context_from_store(
        next_snapshot,
        torch.arange(3),
        torch.tensor([3, 0, 4]),
        eos_token_id=248044,
        ngram_size=3,
    )
    assert next_history.tolist() == [[22, 23], [248044, 248044], [13, 14]]
    assert (
        ngram_context_from_store(
            snapshot, torch.arange(3), context, eos_token_id=248044, ngram_size=3
        ).tolist()
        == history.tolist()
    )


def test_ngram_history_uses_configured_width_and_eos():
    store = torch.tensor([[11, 12, 13, 14]], dtype=torch.int32)
    history = ngram_context_from_store(
        store, torch.tensor([0, 0]), torch.tensor([1, 4]), eos_token_id=7, ngram_size=4
    )
    assert history.tolist() == [[7, 7, 11], [12, 13, 14]]


@pytest.mark.parametrize("spec", [0, 3])
def test_ngram_window_matches_full_store(spec):
    """VLLM_TPU_QWEN4_NGRAM_WINDOW: the per-row host window (width spec + ngram - 1 before the host
    context) gives the full-store n-gram context for every device context the async rejection
    correction can produce (host context minus 0..spec), incl. contexts shorter than the n-gram;
    a window one token too narrow is caught."""
    import numpy as np

    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import (
        ngram_context_from_window,
        ngram_window_from_host,
    )

    ngram, eos, rows, n, start = 4, 7, 6, 4, 1
    rng = np.random.default_rng(spec)
    store = rng.integers(100, 999, (8, 64)).astype(np.int32)
    host_ctx = np.array([0, 0, 1, 2, 30, 63, 5, 9], dtype=np.int64)
    width = spec + ngram - 1
    window = torch.from_numpy(ngram_window_from_host(store, host_ctx, start, n, rows, width, eos))
    row_ids = torch.arange(rows, dtype=torch.int64) + start
    for r in range(spec + 1):
        dev_ctx = torch.zeros(rows, dtype=torch.int64)
        dev_ctx[:n] = torch.from_numpy(np.maximum(host_ctx[start : start + n] - r, 0))
        ref = ngram_context_from_store(torch.from_numpy(store), row_ids, dev_ctx, eos, ngram)
        got = ngram_context_from_window(window, dev_ctx, eos, ngram)
        assert torch.equal(got[:n], ref[:n]), (r, got[:n], ref[:n])
    if spec:
        narrow = torch.from_numpy(ngram_window_from_host(store, host_ctx, start, n, rows, width - 1, eos))
        dev_ctx = torch.zeros(rows, dtype=torch.int64)
        dev_ctx[:n] = torch.from_numpy(np.maximum(host_ctx[start : start + n] - spec, 0))
        ref = ngram_context_from_store(torch.from_numpy(store), row_ids, dev_ctx, eos, ngram)
        assert not torch.equal(ngram_context_from_window(narrow, dev_ctx, eos, ngram)[:n], ref[:n])
