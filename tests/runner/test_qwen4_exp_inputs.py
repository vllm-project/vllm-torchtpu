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


@pytest.mark.parametrize(
    "architecture", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_qwen_adapter_constructor_does_not_access_device(monkeypatch, architecture):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")

    class Runner:
        model_config = NS(architecture=architecture, hf_text_config=q4exp_text_config())
        cache_config = NS(block_size=128)

        @property
        def device(self):
            raise AssertionError("adapter constructor accessed runner.device")

    runner = Runner()
    adapter = select_model_adapter(runner)
    assert type(adapter) is Qwen4ExpAdapter
    assert adapter.runner is runner
    assert adapter.config is runner.model_config.hf_text_config
    assert adapter.block_size == 128
    assert adapter.cache is None and adapter.ngram_context is None


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("qsa_index", [0, 3])
def test_reordered_history_slots_and_group_tables_are_independent(
    monkeypatch, offload, qsa_index
):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    store = torch.tensor(
        [[90, 91, 92, 93], [20, 21, 22, 23], [30, 31, 32, 33]], dtype=torch.int32
    )
    ctx = NS(
        query_start_loc=torch.tensor([0, 1, 4, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4, 4, 0], dtype=torch.int32),
        mamba_state_indices=torch.tensor([7, 2, 0], dtype=torch.int32),
    )
    runner = NS(
        device="cpu",
        _attn_metadata_builder_ctx=ctx,
        input_batch=NS(
            num_reqs=2, token_ids_cpu_tensor=store, num_computed_tokens_cpu=[3, 1]
        ),
    )
    runner.model_config = NS(hf_text_config=q4exp_text_config())
    runner.cache_config = NS(block_size=128)
    runner.model_config.hf_text_config.layer_types[qsa_index] = "indexed_attention"
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
            adapter.begin_step(),
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
    assert actual["state_slots"].tolist() == [0, 1, -1]
    assert actual["main_block_table"].tolist()[0] == [11, 12]
    assert actual["compressed_block_table"].tolist()[0] == [11, 12]
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
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
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
        dummy_metadata=lambda metadata, active: {
            name: metadata for name in ("gdn", "qsa", "ple")
        },
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
    assert adapter.ngram_context.shape == (8, 2)
    assert kwargs["ngram_context"].tolist() == [[248044, 248044]] * 8
    assert "ple_raw_embeddings" not in kwargs
    logits.assert_not_called()
    runner.input_batch.num_reqs = 1
    with pytest.raises(AssertionError, match="no live requests"):
        adapter.dummy_run(128, 8)
    # An idle DP may still own active requests and serve remote embedding/
    # expert work. Its synthetic step must neither rebind owners nor replace
    # the history snapshot; explicit physical counts avoid another CPU sync.
    before = adapter.ngram_context
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
    assert adapter.ngram_context is before
    assert (adapter.ngram_context == 123).all()
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
    runner = NS(
        device="cpu",
        input_batch=NS(
            num_reqs=3,
            token_ids_cpu_tensor=host[order],
            num_computed_tokens_cpu=[2, 0, 3],
        ),
    )
    runner.model_config = NS(hf_text_config=q4exp_text_config())
    runner.cache_config = NS(block_size=128)
    adapter = Qwen4ExpAdapter(runner)
    snapshot = adapter.begin_step()
    assert snapshot.tolist() == [[21, 22], [248044, 248044], [12, 13]]
    runner.input_batch.token_ids_cpu_tensor[0, 2:4] = torch.tensor([23, -1])
    runner.input_batch.token_ids_cpu_tensor[2, 3:5] = torch.tensor([14, -1])
    runner.input_batch.num_computed_tokens_cpu = [3, 0, 4]
    next_snapshot = adapter.begin_step()
    assert next_snapshot.tolist() == [[22, 23], [248044, 248044], [13, 14]]
    assert snapshot.tolist() == [[21, 22], [248044, 248044], [12, 13]]


@pytest.mark.parametrize("ngram_size", [2, 5])
@pytest.mark.parametrize("num_tokens", [0, 8])
def test_empty_request_snapshot_preserves_padded_input_contract(
    monkeypatch, ngram_size, num_tokens
):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    runner = NS(
        device="cpu",
        cache_config=NS(block_size=128),
        model_config=NS(hf_text_config=q4exp_text_config(ngram_size=ngram_size)),
        input_batch=NS(
            num_reqs=0,
            token_ids_cpu_tensor=torch.full((4, 4096), -999, dtype=torch.int32),
            num_computed_tokens_cpu=[],
        ),
        _attn_metadata_builder_ctx=NS(
            query_start_loc=torch.zeros(4, dtype=torch.int32),
            seq_lens=torch.zeros(3, dtype=torch.int32),
            mamba_state_indices=torch.tensor([9, 7, 8]),
        ),
    )
    adapter = Qwen4ExpAdapter(runner)
    snapshot = adapter.begin_step()
    assert snapshot.shape == (0, ngram_size - 1)
    main = adapter.attention_prefix + ".self_attn"
    metadata = {main: NS(block_tables=torch.zeros(3, 2, dtype=torch.int32))}
    kwargs = adapter.prepare_inputs(
        ModelInputSnapshot(
            snapshot,
            0,
            0,
            num_tokens,
            metadata,
            torch.zeros(num_tokens, dtype=torch.int32),
        )
    )
    assert (
        kwargs["ngram_context"].tolist()
        == [[adapter.config.eos_token_id] * (ngram_size - 1)] * 3
    )
    assert kwargs["state_slots"].tolist() == [-1, -1, -1]
    assert not kwargs["valid_tokens"].any()
    assert kwargs["main_block_table"].shape == (3, 2)
    assert torch.equal(kwargs["main_block_table"], kwargs["compressed_block_table"])


def test_ngram_history_uses_configured_width_and_eos():
    store = torch.tensor([[11, 12, 13, 14]], dtype=torch.int32)
    history = ngram_context_from_store(
        store, torch.tensor([0, 0]), torch.tensor([1, 4]), eos_token_id=7, ngram_size=4
    )
    assert history.tolist() == [[7, 7, 11], [12, 13, 14]]


@pytest.mark.parametrize("kv", ["bfloat16", "fp8_e4m3"])
def test_native_pool_metadata_consumes_bounded_history(kv):
    from tests.runner.qwen4_pool_test_utils import prepare_native_pool_inputs
    from tests.runner.test_qwen4_exp_compact_pool import (
        compact_config,
        native_allocation,
    )

    cfg = compact_config(kv=kv, length=128)
    _, _, plan, _ = native_allocation(cfg)
    rows = [
        [
            [row + 1] * group.kv_cache_spec.max_num_blocks_per_req(cfg, 128)
            for group in plan.kv_cache_groups
        ]
        for row in range(2)
    ]
    contexts = [0, 3]
    runner, kwargs, _ = prepare_native_pool_inputs(cfg, plan, rows, contexts, [3, 1], 8)
    config = cfg.model_config.hf_text_config
    width = config.ngram_size - 1
    expected = [
        [
            int(runner.input_batch.token_ids_cpu_tensor[row, position])
            if position >= 0
            else config.eos_token_id
            for position in range(context - width, context)
        ]
        for row, context in enumerate(contexts)
    ]
    expected += [[config.eos_token_id] * width] * (len(kwargs["state_slots"]) - 2)
    assert kwargs["ngram_context"].tolist() == expected


@pytest.mark.parametrize("ngram_size", [2, 3, 5])
@pytest.mark.parametrize("has_ple", [False, True])
def test_step_snapshot_is_bounded_and_shared_across_chunks(ngram_size, has_ple):
    eos = 7
    store = torch.arange(6 * 128, dtype=torch.int32).reshape(6, 128)
    context = [0, 1, 9, 4, 17]
    runner = NS(
        device="cpu",
        cache_config=NS(block_size=128),
        model_config=NS(
            hf_text_config=q4exp_text_config(
                ngram_size=ngram_size,
                eos_token_id=eos,
                ple_layer_ids=[1] if has_ple else [],
            )
        ),
        input_batch=NS(
            num_reqs=5, token_ids_cpu_tensor=store, num_computed_tokens_cpu=context
        ),
    )
    adapter = Qwen4ExpAdapter(runner)
    snapshot = adapter.begin_step()
    assert snapshot.shape == (5 if has_ple else 0, ngram_size - 1)
    expected = [
        [int(store[r, p]) if p >= 0 else eos for p in range(c - ngram_size + 1, c)]
        for r, c in enumerate(context)
    ]
    store.fill_(-999)  # Later host commits cannot mutate any chunk's snapshot.
    for start, count in [(0, 3), (3, 2)]:
        starts = torch.tensor([0, 2, 3, 4, 4][: count + 1] + [4] * (3 - count))
        starts[count:] = starts[count]
        lengths = starts[1:] - starts[:-1]
        runner._attn_metadata_builder_ctx = NS(
            query_start_loc=starts,
            seq_lens=lengths
            + torch.tensor(context[start : start + count] + [0] * (3 - count)),
            mamba_state_indices=torch.arange(3),
        )
        metadata = {
            adapter.attention_prefix + "." + role: NS(block_tables=torch.zeros(3, 2))
            for role in ("self_attn", "self_attn.compressed_k")
        }
        kwargs = adapter.prepare_inputs(
            ModelInputSnapshot(
                snapshot, start, count, 8, metadata, torch.zeros(8, dtype=torch.int32)
            )
        )
        want = (
            expected[start : start + count]
            if has_ple
            else [[eos] * (ngram_size - 1)] * count
        )
        want += [[eos] * (ngram_size - 1)] * (3 - count)
        assert kwargs["ngram_context"].tolist() == want


@pytest.mark.parametrize("ngram_size", [2, 3, 5])
def test_real_runner_chunks_and_async_input_hash(monkeypatch, ngram_size):
    import numpy as np
    from runner_input_test_utils import _make_runner, _scheduler_output
    from vllm.utils.torch_utils import set_default_torch_dtype

    from tests.models.vllm.test_qwen4_exp import _PLE, _set_tp, _text_config
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import (
        PLEInputs,
        Qwen4ExpNGramEmbedding,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.pooled_ple import Qwen4ExpPooledPLE
    from vllm_torchtpu.runner import tpu_runner

    _set_tp(monkeypatch, 0, 1)
    # Only the TPU compilation wrapper is removed in this CPU fixture.
    monkeypatch.setattr(
        tpu_runner,
        "_substitute_placeholder_token",
        tpu_runner._substitute_placeholder_token.__wrapped__,
    )
    scheduled, computed = [3, 1, 1, 2, 1], [0, 6, 4, 2, 9]
    runner = _make_runner(
        num_computed_tokens=computed,
        prompt_tokens=[16] * 5,
        scheduled_tokens=scheduled,
        token_paddings=[16],
        pcp_size=1,
    )
    runner.input_batch.token_ids_cpu_tensor.remainder_(32)
    runner.num_reqs_max_model_len = 3
    runner.cache_config.block_size = 16
    runner.model_config = NS(
        hf_text_config=q4exp_text_config(ngram_size=ngram_size, eos_token_id=3)
    )
    runner._pre_async_results = NS(
        req_id_to_index_copy={"req4": 0, "req1": 1, "req2": 2},
        spec_decode_num_rejected_tokens=None,
        next_tokens_tpu=torch.tensor([17, 19, 23], dtype=torch.int32),
    )
    runner._prepare_async_token_substitution_indices = (
        tpu_runner.TPUModelRunner._prepare_async_token_substitution_indices.__get__(
            runner
        )
    )
    for row in (1, 2, 4):
        runner.input_batch.token_ids_cpu_tensor[row, computed[row]] = -1
    batch = runner.input_batch

    def swap(i, j):
        batch.req_ids[i], batch.req_ids[j] = batch.req_ids[j], batch.req_ids[i]
        for values in (
            batch.num_computed_tokens_cpu,
            batch.num_prompt_tokens,
            batch.token_ids_cpu,
        ):
            values[[i, j]] = values[[j, i]]
        batch.req_id_to_index = {req: r for r, req in enumerate(batch.req_ids)}

    batch.swap_states = swap
    scheduler = _scheduler_output(scheduled)
    tpu_runner.TPUModelRunner._reorder_batch_for_rpa(runner, scheduler)
    computed = batch.num_computed_tokens_cpu.tolist()
    scheduled = [scheduler.num_scheduled_tokens[req] for req in batch.req_ids]
    source = runner.input_batch.token_ids_cpu_tensor.clone()
    adapter = Qwen4ExpAdapter(runner)
    snapshot = adapter.begin_step()
    config = _text_config(
        **dict(_PLE, ngram_size=ngram_size, ple_embed_dim=(ngram_size - 1) * 2 * 4)
    )
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(cfg, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(cfg), set_default_torch_dtype(torch.bfloat16):
        embedding = Qwen4ExpNGramEmbedding(config, config.ple_embed_dim, 0, None, "ple")
    model = NS(ple_embedding=embedding, parallel=NS(gather_ids=lambda ids: ids))
    start = 0
    while start < 5:
        prepared = tpu_runner.TPUModelRunner._prepare_inputs(
            runner, scheduler, start, 0
        )
        end = prepared[4]
        ctx = runner._attn_metadata_builder_ctx
        ctx.mamba_state_indices = torch.arange(len(ctx.seq_lens))
        current, previous = runner._prepare_async_token_substitution_indices(
            start, end - start, np.asarray(scheduled[start:end], dtype=np.int32)
        )
        ids = tpu_runner.TPUModelRunner._apply_async_token_substitution(
            runner, runner.input_ids, current, previous
        )
        rows = len(ctx.seq_lens)
        metadata = {
            adapter.attention_prefix + "." + role: NS(
                block_tables=torch.zeros(rows, 2, dtype=torch.int32)
            )
            for role in ("self_attn", "self_attn.compressed_k")
        }
        kwargs = adapter.prepare_inputs(
            ModelInputSnapshot(snapshot, start, end - start, len(ids), metadata, ids)
        )
        expected_history = [
            [
                int(source[r, p]) if p >= 0 else config.eos_token_id
                for p in range(computed[r] - ngram_size + 1, computed[r])
            ]
            for r in range(start, end)
        ]
        assert kwargs["ngram_context"][: end - start].tolist() == expected_history
        valid = sum(scheduled[start:end])
        expected_tokens = []
        for row in range(start, end):
            values = source[
                row, computed[row] : computed[row] + scheduled[row]
            ].tolist()
            pending = {"req1": 19, "req2": 23, "req4": 17}
            if batch.req_ids[row] in pending:
                values[0] = pending[batch.req_ids[row]]
            expected_tokens.extend(values)
        assert ids[:valid].tolist() == expected_tokens
        actual = Qwen4ExpPooledPLE._lookup_ids(
            model,
            PLEInputs(ids, kwargs["query_start_loc"], kwargs["ngram_context"]),
            torch.tensor(valid),
        )
        want = []
        offset = 0
        multipliers = embedding.layer_multipliers.tolist()
        sizes = embedding.ngram_heads_vocab_sizes.tolist()
        offsets = embedding.ngram_heads_offsets.tolist()
        for local, row in enumerate(range(start, end)):
            tokens = expected_tokens[offset : offset + scheduled[row]]
            for pos, token in enumerate(tokens):
                mixed, crossed, heads = token * multipliers[0], False, []
                past = expected_history[local] + tokens[:pos]
                for shift in range(1, ngram_size):
                    predecessor = config.eos_token_id if crossed else past[-shift]
                    crossed |= predecessor == config.eos_token_id
                    mixed ^= predecessor * multipliers[shift]
                    for h in range((shift - 1) * 2, shift * 2):
                        heads.append(mixed % sizes[h] + offsets[h])
                want.append(heads)
            offset += scheduled[row]
        assert actual[:valid].tolist() == want
        assert (actual[valid:] == -1).all()
        start = end
    for row, req in enumerate(batch.req_ids):
        if req in ("req1", "req2", "req4"):
            assert source[row, computed[row]] == -1
