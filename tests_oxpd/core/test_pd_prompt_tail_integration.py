# SPDX-License-Identifier: Apache-2.0
"""q38 ADM-P integration (identity check I1 of the CP0 ADM-P addendum).

The engine-core half (core/pd_prompt_tail.py) trims a real vLLM
NewRequestData to a PromptTail; the object goes through what serving does to
it (dataclasses.replace, as qwen4_exp/stage_pages.py:160 does, and the
broadcast's pickle round trip); then the worker half
(TPUModelRunner._prepare_prompt_tails + the tail-aware InputBatch.add_request)
adds it to a real vLLM InputBatch. The row must equal a full-prompt add at
every position the decode reads (>= start), with the same prompt and token
counts, and a read below start must raise. Built from DEC-HOST-B's scratch
check scratch/dechost_b/q38_dhb_admp_integ.py.
"""

import dataclasses
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_torchtpu.core.pd_prompt_tail import tail_new_request_data
from vllm_torchtpu.runner.prompt_tail import PromptTail, PromptTailError
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

W = 64
N = 700
COMPUTED = 699


def _nrd(prompt):
    fields = {f.name for f in dataclasses.fields(NewRequestData)}
    kw = dict(
        req_id="r1", prompt_token_ids=list(prompt), mm_features=[],
        sampling_params=SamplingParams(temperature=0, max_tokens=8), pooling_params=None,
        block_ids=([0, 1, 2],), num_computed_tokens=COMPUTED, lora_request=None,
    )
    for f in dataclasses.fields(NewRequestData):
        if f.name not in kw and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            kw[f.name] = None
    return NewRequestData(**{k: v for k, v in kw.items() if k in fields})


def _batch():
    return InputBatch(
        max_num_reqs=4, max_model_len=1024, max_num_batched_tokens=32, device=torch.device("cpu"),
        vocab_size=512, block_sizes=[256], kernel_block_sizes=[256], max_num_blocks_per_req=[4],
    )


def _add(nrd, monkeypatch, flag):
    if flag:
        monkeypatch.setenv("VLLM_TPU_PD_PROMPT_TAIL", str(W))
    else:
        monkeypatch.delenv("VLLM_TPU_PD_PROMPT_TAIL", raising=False)
    runner = object.__new__(TPUModelRunner)
    runner.input_batch = _batch()
    runner.requests = {}
    runner.speculative_config = None
    so = SimpleNamespace(
        scheduled_new_reqs=[nrd],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], resumed_req_ids=set(), num_computed_tokens=[], all_token_ids={}),
    )
    runner._prepare_prompt_tails(so)
    st = CachedRequestState(
        req_id="r1", prompt_token_ids=nrd.prompt_token_ids, mm_features=[], sampling_params=nrd.sampling_params,
        generator=None, block_ids=nrd.block_ids, num_computed_tokens=nrd.num_computed_tokens, output_token_ids=[],
    )
    row = runner.input_batch.add_request(st)
    return runner.input_batch, row


@pytest.fixture
def prompt():
    return [(5 * i + 1) % 499 for i in range(N)]


def test_tail_row_equals_full_row_at_every_read_position(monkeypatch, prompt):
    full_b, full_row = _add(_nrd(prompt), monkeypatch, flag=False)

    nrd = tail_new_request_data(_nrd(prompt), W)
    assert isinstance(nrd.prompt_token_ids, PromptTail)
    start = COMPUTED - W
    assert nrd.prompt_token_ids.start == start and nrd.prompt_len == N
    full_bytes = len(pickle.dumps(_nrd(prompt), protocol=pickle.HIGHEST_PROTOCOL))
    nrd = pickle.loads(pickle.dumps(dataclasses.replace(nrd), protocol=pickle.HIGHEST_PROTOCOL))
    assert isinstance(nrd.prompt_token_ids, PromptTail) and len(nrd.prompt_token_ids) == N
    assert len(pickle.dumps(nrd, protocol=pickle.HIGHEST_PROTOCOL)) < full_bytes

    tail_b, tail_row = _add(nrd, monkeypatch, flag=True)
    np.testing.assert_array_equal(
        tail_b.token_ids_cpu[tail_row, start:N], full_b.token_ids_cpu[full_row, start:N]
    )
    assert list(tail_b.token_ids_cpu[tail_row, start:N]) == prompt[start:]
    assert tail_b.num_prompt_tokens[tail_row] == full_b.num_prompt_tokens[full_row] == N
    assert tail_b.num_tokens_no_spec[tail_row] == full_b.num_tokens_no_spec[full_row] == N
    assert tail_b.num_computed_tokens_cpu[tail_row] == full_b.num_computed_tokens_cpu[full_row] == COMPUTED


def test_a_read_below_start_raises(prompt):
    nrd = tail_new_request_data(_nrd(prompt), W)
    t = nrd.prompt_token_ids
    assert t[COMPUTED - W] == prompt[COMPUTED - W] and t[-1] == prompt[-1]
    with pytest.raises(PromptTailError):
        t[COMPUTED - W - 1]
    with pytest.raises(PromptTailError):
        t[0:10]
    with pytest.raises(PromptTailError):
        list(t)


def test_worker_with_flag_off_refuses_a_tail(monkeypatch, prompt):
    nrd = tail_new_request_data(_nrd(prompt), W)
    with pytest.raises(PromptTailError):
        _add(nrd, monkeypatch, flag=False)
