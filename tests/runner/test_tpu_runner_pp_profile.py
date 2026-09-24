# SPDX-License-Identifier: Apache-2.0
"""Timed prefill-shaped dummy runs behind the pipeline chunk scheduler."""

import contextlib
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config import CacheConfig
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as rpa_batched
from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

# The CPU runner harness of the PCP integration tests builds a runner whose
# metadata path is real down to the attention builder.
_harness = importlib.import_module("test_pcp_runner_integration")


def _profile_run(monkeypatch, runner, num_tokens, prefix, first_stage=True):
    seen = {}

    def capture_forward_context(attn_metadata, *_args, **_kwargs):
        seen["metadata"] = attn_metadata
        return contextlib.nullcontext()

    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_forward_context", capture_forward_context
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.synchronize_tensors",
        lambda tensors=None, **_kwargs: seen.setdefault("synced", []).append(tensors),
    )
    runner.maybe_select_dummy_loras = lambda *_args, **_kwargs: contextlib.nullcontext()
    runner._pp_is_first = first_stage
    runner.vocab_size = 1000
    runner._profile_inputs = {}
    runner._profile_input = TPUModelRunner._profile_input.__get__(runner)
    runner._pp_intermediate_template = {
        "hidden_states": ((4,), torch.float32),
        "residual": ((4,), torch.float32),
    }
    runner._pp_intermediate_tensors = TPUModelRunner._pp_intermediate_tensors.__get__(
        runner
    )

    def forward(input_ids, positions, inputs_embeds, intermediate_tensors):
        seen["input_ids"] = input_ids
        seen["intermediate"] = intermediate_tensors
        rows = positions.shape[-1]
        if first_stage:
            return torch.zeros((rows, 4)), None
        return IntermediateTensors({"hidden_states": torch.zeros((rows, 4))}), None

    runner.forward_model = MagicMock(side_effect=forward)
    elapsed = TPUModelRunner._dummy_run(
        runner,
        num_tokens,
        runner.num_reqs_max_model_len,
        runner.input_batch.block_table[0].max_num_blocks_per_req,
        use_max_model_len=True,
        profile_prefix=prefix,
    )
    return elapsed, seen["metadata"]["layer.0"], seen


def _runner(max_model_len=4096):
    runner = _harness._make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[32],
        scheduled_tokens=[32],
        token_paddings=[256],
    )
    runner.max_model_len = max_model_len
    # A dense model: no shared sparse index table rides the hand-off.
    runner._pp_topk_buffer = None
    runner._pp_take_topk_indices = TPUModelRunner._pp_take_topk_indices.__get__(runner)
    return runner


def test_profile_run_is_one_prefill_request_behind_the_prefix(monkeypatch):
    elapsed, md, seen = _profile_run(monkeypatch, _runner(), 256, 512)

    assert isinstance(elapsed, float) and elapsed >= 0.0
    assert md.query_start_loc.tolist() == [0] + [256] * 8
    assert md.seq_lens.tolist() == [768] + [1] * 7
    assert md.request_distribution.tolist() == [0, 0, 1]
    # Token ids vary so rows route on their own.
    assert len(set(seen["input_ids"].tolist())) > 1


def test_profile_run_marks_every_row_valid(monkeypatch):
    runner = _runner()
    runner._token_padding_state = SimpleNamespace(
        update=MagicMock(), local_padding_mask=torch.zeros(256)
    )
    _profile_run(monkeypatch, runner, 256, 0)
    runner._token_padding_state.update.assert_called_once_with(256, 256)
    _profile_run(monkeypatch, runner, 256, None)
    runner._token_padding_state.update.assert_called_with(0, 256)


def test_profile_run_splits_a_bucket_longer_than_the_model(monkeypatch):
    elapsed, md, _ = _profile_run(monkeypatch, _runner(max_model_len=100), 256, 0)
    assert isinstance(elapsed, float)
    assert md.query_start_loc.tolist() == [0, 100, 200, 256] + [256] * 5
    assert md.seq_lens.tolist() == [100, 100, 56] + [1] * 5
    assert md.request_distribution.tolist() == [0, 0, 3]


def test_profile_run_on_a_later_stage_times_random_hidden_states(monkeypatch):
    elapsed, _, seen = _profile_run(monkeypatch, _runner(), 256, 512, first_stage=False)
    assert isinstance(elapsed, float) and elapsed >= 0.0
    hidden = seen["intermediate"]["hidden_states"]
    assert hidden.shape == (256, 4) and hidden.abs().sum() > 0
    # Inputs are materialized before the clock starts; the hand-off output
    # is synchronized after the forward.
    assert [len(t) for t in seen["synced"]] == [3, 1]


def test_plain_dummy_run_keeps_the_decode_shape_and_returns_nothing(monkeypatch):
    elapsed, md, seen = _profile_run(monkeypatch, _runner(), 256, None)

    assert elapsed is None
    assert md.query_start_loc.tolist() == list(range(9))
    assert md.seq_lens.tolist() == [1] * 8
    assert md.request_distribution.tolist() == [8, 8, 8]
    assert seen["input_ids"].abs().sum() == 0


def _bucket_runner(paddings, block_size, mamba_state, mode, pp=8, cls=None):
    from vllm_torchtpu.core.pp_chunk_scheduler import SCHEDULER_CLS

    runner = MagicMock(spec=TPUModelRunner)
    runner.num_tokens_paddings = list(paddings)
    runner.max_num_tokens = paddings[-1]
    runner.max_model_len = 65536
    runner.num_reqs_max_model_len = 8
    runner.max_num_blocks_per_req = 16
    runner._pp_chunk_granularity = None
    runner._has_mamba_state = mamba_state
    runner.cache_config = SimpleNamespace(block_size=block_size, mamba_cache_mode=mode)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        scheduler_config=SimpleNamespace(
            scheduler_cls=SCHEDULER_CLS if cls is None else cls
        ),
        compilation_config=SimpleNamespace(compile_sizes=list(paddings)),
    )
    runner._add_pipeline_chunk_buckets = (
        TPUModelRunner._add_pipeline_chunk_buckets.__get__(runner)
    )
    return runner


def test_chunk_buckets_follow_the_final_block_when_aligned():
    runner = _bucket_runner([16, 4096, 8192, 16384], 4352, True, "align")
    runner._add_pipeline_chunk_buckets()
    assert runner._pp_chunk_granularity == 4352
    assert runner.num_tokens_paddings == [16, 4096, 4352, 8192, 8704, 13056, 16384]
    assert runner.vllm_config.compilation_config.compile_sizes == (
        runner.num_tokens_paddings
    )
    # Idempotent.
    runner._add_pipeline_chunk_buckets()
    assert runner.num_tokens_paddings.count(4352) == 1


def test_chunk_buckets_use_an_eighth_step_without_alignment():
    runner = _bucket_runner([16, 4096, 8192, 16384], 4352, True, "none")
    runner._add_pipeline_chunk_buckets()
    assert runner._pp_chunk_granularity == 2048
    assert runner.num_tokens_paddings == [
        16,
        2048,
        4096,
        6144,
        8192,
        10240,
        12288,
        14336,
        16384,
    ]


def test_chunk_buckets_are_skipped_without_dynamic_chunks():
    runner = _bucket_runner([16, 16384], 4096, False, "none", cls="x.Y")
    runner._add_pipeline_chunk_buckets()
    assert runner._pp_chunk_granularity is None
    assert runner.num_tokens_paddings == [16, 16384]


def test_profile_pipeline_chunks_times_every_point_twice_and_keeps_faster():
    runner = MagicMock(spec=TPUModelRunner)
    runner.num_tokens_paddings = [16, 4096, 8192, 16384]
    runner.max_num_tokens = 16384
    runner.max_model_len = 65536
    runner.num_reqs_max_model_len = 8
    runner.max_num_blocks_per_req = 16
    runner._pp_chunk_granularity = 2048
    calls = []

    def dummy_run(tokens, num_reqs, num_blocks, use_max_model_len, profile_prefix):
        calls.append((tokens, profile_prefix))
        return 100.0 - (len(calls) % 2 == 0)

    runner._dummy_run = dummy_run
    runner._attention_schedule_capacity = lambda: None
    report = TPUModelRunner.profile_pipeline_chunks(runner)

    points = [(t, p) for t, p, _ in report["samples"]]
    assert points == [
        (16, 0),
        (4096, 0),
        (4096, 16384),
        (4096, 32768),
        (8192, 0),
        (8192, 16384),
        (8192, 32768),
        (16384, 0),
        (16384, 16384),
        (16384, 32768),
    ]
    assert calls == [p for p in points for _ in range(2)]
    assert report["samples"][0][:2] == (16, 0)
    # The faster of the two runs is kept.
    assert report["samples"][0][2] == 99.0
    assert report["granularity"] == 2048
    assert report["schedule"] is None


def _profile_runner(paddings, max_tokens, capacity):
    runner = MagicMock(spec=TPUModelRunner)
    runner.num_tokens_paddings = paddings
    runner.max_num_tokens = max_tokens
    runner.max_model_len = 262144
    runner.num_reqs_max_model_len = 64
    runner.max_num_blocks_per_req = 64
    runner._pp_chunk_granularity = max_tokens // 8
    runner._dummy_run = lambda *args, **kwargs: 1.0
    runner._attention_schedule_capacity = lambda: capacity
    return runner


def test_profile_skips_points_past_the_schedule_capacity():
    capacity = {"decode": (3912, 1, 1792), "mixed": (13436, 256, 256)}
    runner = _profile_runner([16, 8192, 16384, 32768], 32768, capacity)
    report = TPUModelRunner.profile_pipeline_chunks(runner)
    points = [(t, p) for t, p, _ in report["samples"]]
    # 16K behind 32K needs 10,272 pairs of 256x256 tiles; behind 64K it
    # needs 18,464, and any 32K chunk behind a prefix needs more.
    assert points == [
        (16, 0),
        (8192, 0),
        (8192, 32768),
        (8192, 65536),
        (16384, 0),
        (16384, 32768),
        (32768, 0),
    ]
    assert report["schedule"] == (13436, 256, 256)


def test_profile_refuses_a_step_one_request_cannot_run():
    capacity = {"decode": (3912, 1, 1792), "mixed": (13436, 256, 256)}
    runner = _profile_runner([16, 32768, 65536], 65536, capacity)
    with pytest.raises(RuntimeError, match="32896 .* holds 13436"):
        TPUModelRunner.profile_pipeline_chunks(runner)


def _capacity_runner():
    """A runner shaped like Qwen3.5-397B-A17B-FP8 served with DP8 (TP1) on
    tpu7x at 262K context with the HND KV layout: 32 query heads, 2 KV
    heads, head_dim 256, fp8 KV, 4224-token blocks paged at 128 tokens.
    The kernel's sizing needs a TPU, so the tests stub it."""
    runner = MagicMock(spec=TPUModelRunner)
    runner._attention_capacity = {}
    runner._attention_runs_batched_kernel = None
    runner.parallel_config = SimpleNamespace()
    runner.vllm_config = SimpleNamespace(parallel_config=runner.parallel_config)
    runner.head_size = 256
    runner.num_kv_heads = 2
    # The kernel is called with the bucket's request cap and the block
    # table's width, not with the engine-wide maxima.
    runner.max_num_reqs = 64
    runner.max_model_len = 262144
    runner.num_reqs_max_model_len = 8
    runner.most_model_len = None
    runner.num_reqs_most_model_len = None
    runner._attention_kernel_block_size = 128
    runner._attention_kv_cache_group_id = 0
    runner.input_batch = SimpleNamespace(
        block_table=[SimpleNamespace(max_num_blocks_per_req=2079)]
    )
    runner.cache_config = CacheConfig(block_size=4224)
    runner.cache_config.kv_cache_layout = "LBHNC"
    runner.model_config = SimpleNamespace(
        dtype=torch.bfloat16, get_num_attention_heads=lambda _config: 32
    )
    runner.kv_cache_dtype = torch.float8_e4m3fn
    runner._attention_kernel_shapes = TPUModelRunner._attention_kernel_shapes.__get__(
        runner
    )
    return runner


# The kernel's own sizing for the runner above, as logged on tpu7x:
# (pairs, query tile, KV tile) per mode.
_MEASURED_CAPACITY = {"decode": (2672, 1, 1536), "mixed": (8444, 256, 256)}


def _attention_layers(monkeypatch, batched, capacity=None):
    """Stub the model's attention layers; ``batched`` says per layer whether
    its kernel keeps the batched SMEM schedule. Returns the schedule sizings
    the runner asked the kernel for. ``capacity`` replaces the stub's
    per-mode sizing when given."""
    layers = {}
    for i, runs_batched in enumerate(batched):
        impl = MagicMock(spec=PallasAttentionBackendImpl)
        impl.runs_batched_rpa_schedule.return_value = runs_batched
        layers[f"layer.{i}"] = SimpleNamespace(impl=impl)
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config",
        lambda _config, _layer_type: layers,
    )
    sized = []

    def schedule_capacity(*, mode, **kwargs):
        sized.append(
            (
                mode.name,
                kwargs["num_seqs"],
                kwargs["pages_per_seq"],
                kwargs["page_size"],
            )
        )
        if capacity is not None:
            return capacity[mode.name.lower()]
        return (100 + kwargs["num_seqs"], 256, 256)

    monkeypatch.setattr(rpa_batched, "schedule_capacity", schedule_capacity)
    return sized


def test_schedule_capacity_comes_from_the_batched_kernel(monkeypatch):
    # One layer on the batched kernel is enough to bound the step.
    sized = _attention_layers(monkeypatch, [False, True])
    runner = _capacity_runner()
    capacity = TPUModelRunner._attention_schedule_capacity(runner)
    assert capacity == {"decode": (108, 256, 256), "mixed": (108, 256, 256)}
    # Sized by the max-model-len bucket's 8 sequences and the block table's
    # 2079 kernel pages, not by 64 requests x 262144 / 128.
    assert sized == [("DECODE", 8, 2079, 128), ("MIXED", 8, 2079, 128)]
    assert runner._attention_runs_batched_kernel
    # Cached per bucket shape.
    assert TPUModelRunner._attention_schedule_capacity(runner) is capacity
    assert len(sized) == 2


def test_schedule_capacity_follows_the_most_model_len_bucket(monkeypatch):
    sized = _attention_layers(monkeypatch, [True])
    runner = _capacity_runner()
    runner.most_model_len = 65536
    runner.num_reqs_most_model_len = 32
    capacity = TPUModelRunner._attention_schedule_capacity(
        runner, use_max_model_len=False
    )
    assert capacity == {"decode": (132, 256, 256), "mixed": (132, 256, 256)}
    assert sized == [("DECODE", 32, 512, 128), ("MIXED", 32, 512, 128)]
    # The max bucket keeps its own entry.
    TPUModelRunner._attention_schedule_capacity(runner, use_max_model_len=True)
    assert sized[2:] == [("DECODE", 8, 2079, 128), ("MIXED", 8, 2079, 128)]


def test_schedule_capacity_is_unbounded_when_no_layer_runs_the_batched_kernel(
    monkeypatch,
):
    sized = _attention_layers(monkeypatch, [False, False])
    runner = _capacity_runner()
    assert TPUModelRunner._attention_schedule_capacity(runner) is None
    assert sized == []
    assert runner._attention_runs_batched_kernel is False


def test_check_refuses_a_long_context_step_on_the_batched_kernel(monkeypatch):
    # DP8 prefill of one request in 4096-token steps at the measured
    # capacity: the step behind 131,072 tokens needs 8,328 pairs and runs;
    # the next one, behind 135,168 tokens, needs 8,584 and is refused
    # before it could truncate the kernel's schedule.
    _attention_layers(monkeypatch, [True], capacity=_MEASURED_CAPACITY)
    runner = _capacity_runner()
    runner._attention_schedule_capacity = (
        TPUModelRunner._attention_schedule_capacity.__get__(runner)
    )
    step = np.array([4096], dtype=np.int32)
    runner.seq_lens_np = np.array([131072 + 4096], dtype=np.int32)
    TPUModelRunner._check_attention_schedule(runner, 1, step, 0)
    runner.seq_lens_np = np.array([135168 + 4096], dtype=np.int32)
    with pytest.raises(RuntimeError, match="needs 8584 .* holds 8444"):
        TPUModelRunner._check_attention_schedule(runner, 1, step, 0)


def test_check_passes_a_long_context_step_without_the_batched_kernel(monkeypatch):
    _attention_layers(monkeypatch, [False])
    runner = _capacity_runner()
    runner._attention_schedule_capacity = (
        TPUModelRunner._attention_schedule_capacity.__get__(runner)
    )
    # A 64K-token request behind a 192K prefix: far past the batched
    # kernel's SMEM schedule.
    runner.seq_lens_np = np.array([262144], dtype=np.int32)
    TPUModelRunner._check_attention_schedule(
        runner, 1, np.array([65536], dtype=np.int32), 0
    )
