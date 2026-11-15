# SPDX-License-Identifier: Apache-2.0
"""A new request's prompt reaches the worker as an int32 array with the same
contents and the same worker state.

See `vllm_torchtpu.core.prompt_token_array`. ``NewRequestData`` built with the
patch equals the one built without it in every field; its prompt is an int32
array holding the request's tokens, it survives the message queue's pickling,
and a trimmed prompt list goes out as the list. On the worker, adding the
request to an ``InputBatch`` from the array leaves the same token ids,
lengths and token lookups as adding it from the list.
"""

import dataclasses
import pickle
import random

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_torchtpu.core import prompt_token_array as pta


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(
        NewRequestData, "from_request", NewRequestData.__dict__["from_request"]
    )
    monkeypatch.setattr(
        EngineCore, "preprocess_add_request", EngineCore.preprocess_add_request
    )
    monkeypatch.setenv("VLLM_TPU_PROMPT_TOKEN_ARRAY", "1")
    pta.patch_prompt_token_array()
    assert getattr(EngineCore.preprocess_add_request, pta._PATCH_FLAG, False)


def _prompt(n: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(0, 151_000) for _ in range(n)]


def _request(prompt: list[int], req_id: str = "r0") -> Request:
    return Request(req_id, prompt, SamplingParams(temperature=0.0, max_tokens=8), None)


def _blocks() -> tuple[list[int], ...]:
    return ([1, 2, 3], [4], [5, 6])


def _mq_roundtrip(obj):
    """Pickles like vLLM's shared-memory MessageQueue (protocol 5, buffers of
    1 MiB and more out of band)."""
    bufs = []

    def cb(buf):
        if len(buf.raw()) < 1024 * 1024:
            return True
        bufs.append(buf.raw())
        return False

    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=cb)
    return pickle.loads(data, buffers=bufs)


@pytest.mark.parametrize("n", [1, 5, 96_661, 300_000])
def test_new_request_data_carries_the_prompt_as_int32(patched, n):
    prompt = _prompt(n)
    req = _request(prompt)
    pta.attach(req)  # what the patched preprocess_add_request adds
    plain = NewRequestData.from_request.__func__.__wrapped__(
        NewRequestData, req, _blocks()
    )
    data = NewRequestData.from_request(req, _blocks())
    arr = data.prompt_token_ids
    assert isinstance(arr, np.ndarray) and arr.dtype == np.int32
    assert arr.tolist() == prompt
    for f in dataclasses.fields(NewRequestData):
        if f.name != "prompt_token_ids":
            assert getattr(data, f.name) == getattr(plain, f.name), f.name
    assert plain.prompt_token_ids is req.prompt_token_ids
    # The engine core's own request keeps its list.
    assert req.prompt_token_ids is prompt
    back = _mq_roundtrip(data)
    assert np.array_equal(back.prompt_token_ids, arr)
    assert back.prompt_len == len(prompt) == data.prompt_len


def test_trimmed_prompt_goes_out_as_the_list(patched):
    prompt = _prompt(4096)
    req = _request(prompt)
    pta.attach(req)
    while len(req.prompt_token_ids) > 2048:
        req.prompt_token_ids.pop()
    data = NewRequestData.from_request(req, _blocks())
    assert data.prompt_token_ids is req.prompt_token_ids


def test_request_without_a_copy_is_unchanged(patched):
    req = _request(_prompt(100))
    data = NewRequestData.from_request(req, _blocks())
    assert data.prompt_token_ids is req.prompt_token_ids


def test_out_of_range_ids_keep_the_list():
    assert pta.prompt_array([1, 2**31]) is None
    req = _request([1, 2, 3])
    req.prompt_token_ids[1] = 2**31
    pta.attach(req)
    assert pta.ATTR not in req.__dict__


def test_preprocess_add_request_attaches_the_copy(monkeypatch):
    prompt = _prompt(5000)
    req = _request(prompt)
    calls = []

    def original(self, request):
        calls.append(request)
        return req, 3

    monkeypatch.setattr(EngineCore, "preprocess_add_request", original)
    monkeypatch.setattr(
        NewRequestData, "from_request", NewRequestData.__dict__["from_request"]
    )
    monkeypatch.setenv("VLLM_TPU_PROMPT_TOKEN_ARRAY", "1")
    pta.patch_prompt_token_array()
    out = EngineCore.preprocess_add_request(object(), "engine-core-request")
    assert out == (req, 3) and calls == ["engine-core-request"]
    assert req.__dict__[pta.ATTR].tolist() == prompt


def test_flag_off_installs_nothing(monkeypatch):
    before = EngineCore.preprocess_add_request
    monkeypatch.delenv("VLLM_TPU_PROMPT_TOKEN_ARRAY", raising=False)
    pta.patch_prompt_token_array()
    assert EngineCore.preprocess_add_request is before


def _batch() -> InputBatch:
    return InputBatch(
        max_num_reqs=4,
        max_model_len=320_000,
        max_num_batched_tokens=512,
        device=torch.device("cpu"),
        vocab_size=151_936,
        block_sizes=[2048, 512, 512],
        kernel_block_sizes=[2048, 512, 512],
        max_num_blocks_per_req=[160, 640, 640],
        num_spec_tokens=3,
    )


def _state(req_id: str, prompt) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
        generator=None,
        block_ids=_blocks(),
        num_computed_tokens=len(prompt) - 1,
        output_token_ids=[],
    )


@pytest.mark.parametrize("n", [7, 96_661])
def test_worker_input_batch_is_the_same(patched, n):
    prompt = _prompt(n, seed=n)
    req = _request(prompt)
    pta.attach(req)
    arr = _mq_roundtrip(NewRequestData.from_request(req, _blocks())).prompt_token_ids
    assert isinstance(arr, np.ndarray)
    from_list, from_array = _batch(), _batch()
    s_list, s_arr = _state("a", list(prompt)), _state("a", arr)
    # A second request first, so the new one lands in row 1 of both batches.
    for b in (from_list, from_array):
        b.add_request(_state("other", _prompt(33, seed=1)))
    i_list = from_list.add_request(s_list)
    i_arr = from_array.add_request(s_arr)
    assert i_list == i_arr == 1
    assert s_list.num_prompt_tokens == s_arr.num_prompt_tokens == n
    assert s_list.num_tokens == s_arr.num_tokens
    for name in (
        "token_ids_cpu",
        "is_token_ids",
        "num_prompt_tokens",
        "num_tokens_no_spec",
        "num_computed_tokens_cpu",
    ):
        assert np.array_equal(getattr(from_list, name), getattr(from_array, name)), name
    s_arr.output_token_ids.extend([11, 12])
    s_list.output_token_ids.extend([11, 12])
    for idx in (0, n // 2, n - 1, n, n + 1):
        assert int(s_arr.get_token_id(idx)) == s_list.get_token_id(idx)
