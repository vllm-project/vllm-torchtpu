# SPDX-License-Identifier: Apache-2.0
"""Timed prefill-shaped dummy runs behind the pipeline chunk scheduler."""

import contextlib
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

# The CPU runner harness of the PCP integration tests builds a runner whose
# metadata path is real down to the attention builder.
_harness = importlib.import_module("test_pcp_runner_integration")


def _profile_run(monkeypatch, runner, num_tokens, prefix, first_stage=True):
    seen = {}

    def capture_forward_context(attn_metadata, *_args, **_kwargs):
        seen["metadata"] = attn_metadata
        return contextlib.nullcontext()

    monkeypatch.setattr("vllm_torchtpu.runner.tpu_runner.set_forward_context",
                        capture_forward_context)
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
        lambda *_args, **_kwargs: contextlib.nullcontext())
    monkeypatch.setattr("vllm_torchtpu.runner.tpu_runner.synchronize_tensors",
                        lambda tensors=None, **_kwargs: seen.setdefault(
                            "synced", []).append(tensors))
    runner.maybe_select_dummy_loras = (
        lambda *_args, **_kwargs: contextlib.nullcontext())
    runner._pp_is_first = first_stage
    runner.vocab_size = 1000
    runner._profile_inputs = {}
    runner._profile_input = TPUModelRunner._profile_input.__get__(runner)
    runner._pp_intermediate_template = {
        "hidden_states": ((4, ), torch.float32),
        "residual": ((4, ), torch.float32),
    }
    runner._pp_intermediate_tensors = (
        TPUModelRunner._pp_intermediate_tensors.__get__(runner))

    def forward(input_ids, positions, inputs_embeds, intermediate_tensors):
        seen["input_ids"] = input_ids
        seen["intermediate"] = intermediate_tensors
        rows = positions.shape[-1]
        if first_stage:
            return torch.zeros((rows, 4)), None
        return IntermediateTensors({"hidden_states": torch.zeros(
            (rows, 4))}), None

    runner.forward_model = MagicMock(side_effect=forward)
    elapsed = TPUModelRunner._dummy_run(
        runner,
        num_tokens,
        runner.num_reqs_max_model_len,
        runner.input_batch.block_table[0].max_num_blocks_per_req,
        use_max_model_len=True,
        profile_prefix=prefix)
    return elapsed, seen["metadata"]["layer.0"], seen


def _runner(max_model_len=4096):
    runner = _harness._make_runner(num_computed_tokens=[0],
                                   prompt_tokens=[32],
                                   scheduled_tokens=[32],
                                   token_paddings=[256])
    runner.max_model_len = max_model_len
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
        update=MagicMock(), local_padding_mask=torch.zeros(256))
    _profile_run(monkeypatch, runner, 256, 0)
    runner._token_padding_state.update.assert_called_once_with(256, 256)
    _profile_run(monkeypatch, runner, 256, None)
    runner._token_padding_state.update.assert_called_with(0, 256)


def test_profile_run_splits_a_bucket_longer_than_the_model(monkeypatch):
    elapsed, md, _ = _profile_run(monkeypatch, _runner(max_model_len=100), 256,
                                  0)
    assert isinstance(elapsed, float)
    assert md.query_start_loc.tolist() == [0, 100, 200, 256] + [256] * 5
    assert md.seq_lens.tolist() == [100, 100, 56] + [1] * 5
    assert md.request_distribution.tolist() == [0, 0, 3]


def test_profile_run_on_a_later_stage_times_random_hidden_states(monkeypatch):
    elapsed, _, seen = _profile_run(monkeypatch,
                                    _runner(),
                                    256,
                                    512,
                                    first_stage=False)
    assert isinstance(elapsed, float) and elapsed >= 0.0
    hidden = seen["intermediate"]["hidden_states"]
    assert hidden.shape == (256, 4) and hidden.abs().sum() > 0
    # Inputs are materialized before the clock starts; the hand-off output
    # is synchronized after the forward.
    assert [len(t) for t in seen["synced"]] == [3, 1]


def test_plain_dummy_run_keeps_the_decode_shape_and_returns_nothing(
        monkeypatch):
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
    runner.cache_config = SimpleNamespace(block_size=block_size,
                                          mamba_cache_mode=mode)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        scheduler_config=SimpleNamespace(
            scheduler_cls=SCHEDULER_CLS if cls is None else cls),
        compilation_config=SimpleNamespace(compile_sizes=list(paddings)))
    runner._add_pipeline_chunk_buckets = (
        TPUModelRunner._add_pipeline_chunk_buckets.__get__(runner))
    return runner


def test_chunk_buckets_follow_the_final_block_when_aligned():
    runner = _bucket_runner([16, 4096, 8192, 16384], 4352, True, "align")
    runner._add_pipeline_chunk_buckets()
    assert runner._pp_chunk_granularity == 4352
    assert runner.num_tokens_paddings == [
        16, 4096, 4352, 8192, 8704, 13056, 16384
    ]
    assert runner.vllm_config.compilation_config.compile_sizes == (
        runner.num_tokens_paddings)
    # Idempotent.
    runner._add_pipeline_chunk_buckets()
    assert runner.num_tokens_paddings.count(4352) == 1


def test_chunk_buckets_use_an_eighth_step_without_alignment():
    runner = _bucket_runner([16, 4096, 8192, 16384], 4352, True, "none")
    runner._add_pipeline_chunk_buckets()
    assert runner._pp_chunk_granularity == 2048
    assert runner.num_tokens_paddings == [
        16, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384
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

    def dummy_run(tokens, num_reqs, num_blocks, use_max_model_len,
                  profile_prefix):
        calls.append((tokens, profile_prefix))
        return 100.0 - (len(calls) % 2 == 0)

    runner._dummy_run = dummy_run
    runner._attention_schedule_capacity = lambda: None
    report = TPUModelRunner.profile_pipeline_chunks(runner)

    points = [(t, p) for t, p, _ in report["samples"]]
    assert points == [(16, 0), (4096, 0), (4096, 16384), (4096, 32768),
                      (8192, 0), (8192, 16384), (8192, 32768), (16384, 0),
                      (16384, 16384), (16384, 32768)]
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
    assert points == [(16, 0), (8192, 0), (8192, 32768), (8192, 65536),
                      (16384, 0), (16384, 32768), (32768, 0)]
    assert report["schedule"] == (13436, 256, 256)


def test_profile_refuses_a_step_one_request_cannot_run():
    capacity = {"decode": (3912, 1, 1792), "mixed": (13436, 256, 256)}
    runner = _profile_runner([16, 32768, 65536], 65536, capacity)
    with pytest.raises(RuntimeError, match="32896 .* holds 13436"):
        TPUModelRunner.profile_pipeline_chunks(runner)
