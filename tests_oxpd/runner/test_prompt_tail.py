# SPDX-License-Identifier: Apache-2.0
"""q38 ADM-P worker half: PromptTail, the worker's trim-condition check, the
tail-aware InputBatch.add_request and the re-add / resume guard."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.runner.prompt_tail import (
    MIN_TOKENS_BELOW_COMPUTED,
    PromptTail,
    PromptTailError,
    check_admissible,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

PROMPT = [(7 * i + 3) % 509 for i in range(600)]


def _tail(start=500):
    return PromptTail(len(PROMPT), start, PROMPT[start:])


def test_prompt_tail_reads_tail_and_refuses_early_positions():
    t = _tail()
    assert len(t) == 600 and bool(t)
    assert t[500] == PROMPT[500] and t[599] == PROMPT[599] and t[-1] == PROMPT[-1]
    assert t[550:560] == PROMPT[550:560] and t[590:] == PROMPT[590:] and t[600:] == []
    for bad in (lambda: t[499], lambda: t[0], lambda: t[-101], lambda: t[:10],
                lambda: t[499:510], lambda: list(t), lambda: np.asarray(t)):
        with pytest.raises(PromptTailError):
            bad()
    with pytest.raises(IndexError):
        t[600]
    z = t.zeros()
    assert z.shape == (600,) and z.dtype == np.int32 and not z.any()


def test_pickle_is_compact_and_roundtrips():
    big = list(range(300_000))
    t = PromptTail(len(big), len(big) - 64, big[-64:])
    blob = pickle.dumps(t, protocol=pickle.HIGHEST_PROTOCOL)
    back = pickle.loads(blob)
    assert isinstance(back, PromptTail)
    assert (back.length, back.start, back.tail) == (t.length, t.start, t.tail)
    assert len(blob) < len(pickle.dumps(big, protocol=pickle.HIGHEST_PROTOCOL)) / 100


def _new_req(**over):
    base = dict(
        req_id="r",
        prompt_token_ids=_tail(),
        prompt_embeds=None,
        prompt_is_token_ids=None,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0, max_tokens=8),
        num_computed_tokens=599,
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "over,spec",
    [
        (dict(num_computed_tokens=500 + MIN_TOKENS_BELOW_COMPUTED - 1), None),
        (dict(sampling_params=SamplingParams(frequency_penalty=0.5)), None),
        (dict(sampling_params=SamplingParams(presence_penalty=0.5)), None),
        (dict(sampling_params=SamplingParams(repetition_penalty=1.1)), None),
        (dict(sampling_params=SamplingParams(prompt_logprobs=1)), None),
        (dict(prompt_embeds=torch.zeros(1)), None),
        (dict(prompt_is_token_ids=[True] * 600), None),
        (dict(mm_features=["image"]), None),
        ({}, "ngram"),
    ],
)
def test_check_admissible_refuses_early_readers(over, spec):
    req = _new_req(**over)
    with pytest.raises(PromptTailError):
        check_admissible(req, req.num_computed_tokens, spec)


def test_check_admissible_accepts_plain_decode_request():
    req = _new_req()
    check_admissible(req, req.num_computed_tokens, "mtp")


def _state(req_id, prompt, num_computed=599):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0, max_tokens=8),
        generator=None,
        block_ids=([0, 1, 2],),
        num_computed_tokens=num_computed,
        output_token_ids=[],
    )


def _runner():
    runner = object.__new__(TPUModelRunner)
    runner.input_batch = InputBatch(
        max_num_reqs=4,
        max_model_len=1024,
        max_num_batched_tokens=32,
        device=torch.device("cpu"),
        vocab_size=512,
        block_sizes=[256],
        kernel_block_sizes=[256],
        max_num_blocks_per_req=[4],
    )
    runner.requests = {}
    runner.speculative_config = None
    return runner


def test_tail_add_request_matches_full_prompt_on_real_input_batch():
    runner = _runner()
    batch = runner.input_batch
    runner._install_prompt_tail_add_request()
    runner._install_prompt_tail_add_request()  # idempotent
    full_row = batch.add_request(_state("full", list(PROMPT)))
    tail = _tail()
    tail_row = batch.add_request(_state("tail", tail))
    ids = batch.token_ids_cpu
    assert np.array_equal(ids[tail_row, 500:600], ids[full_row, 500:600])
    assert not ids[tail_row, :500].any()
    assert batch.num_prompt_tokens[tail_row] == batch.num_prompt_tokens[full_row] == 600
    assert batch.num_tokens_no_spec[tail_row] == batch.num_tokens_no_spec[full_row] == 600
    assert batch.is_token_ids[tail_row, :600].all()
    # The request keeps its PromptTail, and so does the logits-processor record.
    assert runner.input_batch.req_output_token_ids[tail_row] == []
    added = batch.batch_update_builder.added
    assert added[-1][0] == tail_row and added[-1][2] is tail


def _so(new=(), req_ids=(), resumed=(), num_computed=(), all_token_ids=None):
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(req_ids),
            resumed_req_ids=set(resumed),
            num_computed_tokens=list(num_computed),
            all_token_ids=all_token_ids or {},
        ),
    )


def test_tail_refused_when_worker_flag_off(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_PD_PROMPT_TAIL", raising=False)
    runner = _runner()
    with pytest.raises(PromptTailError):
        runner._prepare_prompt_tails(_so(new=[_new_req()]))
    runner._prepare_prompt_tails(_so(new=[_new_req(prompt_token_ids=list(PROMPT))]))


def test_new_tail_checked_and_wrapper_installed(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_PD_PROMPT_TAIL", "64")
    from vllm_torchtpu.runner import prompt_tail, tpu_runner

    monkeypatch.setattr(prompt_tail.WorkerTailStats, "accepted", 0)
    monkeypatch.setattr(prompt_tail.WorkerTailStats, "tokens_not_received", 0)
    lines = []
    monkeypatch.setattr(tpu_runner.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    runner = _runner()
    runner._prepare_prompt_tails(_so(new=[_new_req()]))
    assert runner.input_batch._q38_prompt_tail_wrapped
    # Worker engagement line at the first accepted tail (then every 100th).
    assert lines == [
        "PD prompt tail (worker): accepted 1 tails, 500 prompt tokens not received "
        "(latest PromptTail(length=600, start=500, tail_len=100))"
    ]
    runner._prepare_prompt_tails(_so(new=[_new_req()]))
    assert len(lines) == 1 and prompt_tail.WorkerTailStats.accepted == 2
    with pytest.raises(PromptTailError):
        runner._prepare_prompt_tails(
            _so(new=[_new_req(sampling_params=SamplingParams(repetition_penalty=1.2))])
        )


def test_resumed_request_below_tail_gets_full_prompt_back(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_PD_PROMPT_TAIL", "64")
    runner = _runner()
    runner.requests["r"] = _state("r", _tail(), num_computed=599)
    runner._prepare_prompt_tails(
        _so(req_ids=["r"], resumed=["r"], num_computed=[100],
            all_token_ids={"r": list(PROMPT) + [11, 12]})
    )
    assert runner.requests["r"].prompt_token_ids == list(PROMPT)


def test_resumed_request_below_tail_without_token_ids_fails(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_PD_PROMPT_TAIL", "64")
    runner = _runner()
    runner.requests["r"] = _state("r", _tail(), num_computed=599)
    with pytest.raises(PromptTailError):
        runner._prepare_prompt_tails(_so(req_ids=["r"], resumed=["r"], num_computed=[100]))


def test_readded_request_above_tail_keeps_tail(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_PD_PROMPT_TAIL", "64")
    runner = _runner()
    tail = _tail()
    runner.requests["r"] = _state("r", tail, num_computed=599)
    runner._prepare_prompt_tails(
        _so(req_ids=["r"], resumed=["r"], num_computed=[610],
            all_token_ids={"r": list(PROMPT) + [1] * 10})
    )
    assert runner.requests["r"].prompt_token_ids is tail


def test_flag_is_compile_neutral():
    assert "VLLM_TPU_PD_PROMPT_TAIL" in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_slow_dump_context_marks_tails(monkeypatch):
    from vllm_torchtpu.runner import host_phases

    monkeypatch.setattr(host_phases, "_SLOW_NS", 1)
    monkeypatch.setattr(host_phases, "_ctx", [])
    runner = _runner()
    so = _so(new=[SimpleNamespace(prompt_token_ids=_tail(), num_computed_tokens=599,
                                  block_ids=([1, 2],))])
    so.total_num_scheduled_tokens = 4
    TPUModelRunner._note_admissions(runner, so)
    assert host_phases._ctx[1] == "admit:prompt=600:computed=599:blocks=2:tail=500"
