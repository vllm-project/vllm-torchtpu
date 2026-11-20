# SPDX-License-Identifier: Apache-2.0
"""A new request's prompt reaches the worker as an int32 array with the same
contents and the same worker state (VLLM_TPU_PROMPT_TOKEN_ARRAY).

See `vllm_torchtpu.core.prompt_token_array`. ``NewRequestData`` built with the
patch equals the one built without it in every field; its prompt is an int32
array holding the request's tokens, it survives the message queue's pickling,
and a trimmed or extended prompt list goes out as the list. Requests whose
prompt reaches a reader that needs a list keep it. On the worker, adding the
request to an ``InputBatch`` from the array leaves the same token ids,
lengths and token lookups as adding it from the list, and an array that views
a message-queue slot is copied before the slot can be reused.
"""

import dataclasses
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu.core import prompt_token_array as pta
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

# vLLM's MessageQueue pickles buffers of this size and more out of band.
_OOB_BYTES = 1024 * 1024


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


def _request(prompt: list[int], req_id: str = "r0", **sampling) -> Request:
    params = SamplingParams(temperature=0.0, max_tokens=8, **sampling)
    return Request(req_id, prompt, params, None)


def _blocks() -> tuple[list[int], ...]:
    return ([1, 2, 3], [4], [5, 6])


def _pickle_like_mq(obj) -> tuple[bytes, list]:
    """Pickles like vLLM's MessageQueue.enqueue (protocol 5, buffers of 1 MiB
    and more out of band)."""
    bufs = []

    def cb(buf):
        if len(buf.raw()) < _OOB_BYTES:
            return True
        bufs.append(buf.raw())
        return False

    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=cb), bufs


def _mq_roundtrip(obj):
    data, bufs = _pickle_like_mq(obj)
    return pickle.loads(data, buffers=bufs)


def _engine(logits_processors=None):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(logits_processors=logits_processors)
        )
    )


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


def test_scheduler_output_round_trip_keeps_the_ids(patched):
    """A whole SchedulerOutput with two new requests crosses the queue's
    pickling with the same prompts."""
    prompts = [_prompt(300_000, seed=1), _prompt(17, seed=2)]
    reqs = [_request(p, f"r{i}") for i, p in enumerate(prompts)]
    for req in reqs:
        pta.attach(req)
    so = SchedulerOutput.make_empty()
    so.scheduled_new_reqs = [NewRequestData.from_request(r, _blocks()) for r in reqs]
    back = _mq_roundtrip(so)
    for data, prompt in zip(back.scheduled_new_reqs, prompts):
        assert data.prompt_token_ids.tolist() == prompt


def test_trimmed_prompt_goes_out_as_the_list(patched):
    prompt = _prompt(4096)
    req = _request(prompt)
    pta.attach(req)
    while len(req.prompt_token_ids) > 2048:
        req.prompt_token_ids.pop()
    data = NewRequestData.from_request(req, _blocks())
    assert data.prompt_token_ids is req.prompt_token_ids


def test_streaming_extension_goes_out_as_the_list(patched):
    """vLLM's streaming update extends the session's prompt list
    (scheduler.py _update_request_as_session)."""
    req = _request(_prompt(64))
    pta.attach(req)
    req.prompt_token_ids.extend([7, 8, 9])
    data = NewRequestData.from_request(req, _blocks())
    assert data.prompt_token_ids is req.prompt_token_ids


def test_request_without_a_copy_is_unchanged(patched):
    req = _request(_prompt(100))
    data = NewRequestData.from_request(req, _blocks())
    assert data.prompt_token_ids is req.prompt_token_ids


@pytest.mark.parametrize(
    "sampling",
    [{"prompt_logprobs": 1}, {"thinking_token_budget": 64}],
    ids=["prompt_logprobs", "thinking_token_budget"],
)
def test_requests_whose_readers_need_a_list_keep_it(patched, sampling):
    """Prompt logprobs and the thinking-budget state read the worker's prompt
    as a list (`x or []` raises on an array)."""
    req = _request(_prompt(100), **sampling)
    pta.attach(req)
    assert pta.ATTR not in req.__dict__
    assert NewRequestData.from_request(req, _blocks()).prompt_token_ids is (
        req.prompt_token_ids
    )


def test_pooling_requests_keep_the_list(patched):
    req = Request("p0", _prompt(10), None, PoolingParams(task="embed"))
    pta.attach(req)
    assert pta.ATTR not in req.__dict__


@pytest.mark.parametrize("flag", ["uses_mrope", "uses_xdrope"])
def test_rotary_position_models_keep_the_list(patched, flag):
    """M-RoPE and XD-RoPE models build positions from the worker's prompt."""
    req = _request(_prompt(100))
    pta.attach(req)
    data = NewRequestData.from_request(req, _blocks(), **{flag: True})
    assert data.prompt_token_ids is req.prompt_token_ids


def test_out_of_range_ids_keep_the_list():
    assert pta.prompt_array([1, 2**31]) is None
    req = _request([1, 2, 3])
    req.prompt_token_ids[1] = 2**31
    pta.attach(req)
    assert pta.ATTR not in req.__dict__


def _install_with_original(monkeypatch, req):
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
    return calls


def test_preprocess_add_request_attaches_the_copy(monkeypatch):
    prompt = _prompt(5000)
    req = _request(prompt)
    calls = _install_with_original(monkeypatch, req)
    monkeypatch.setattr(pta, "entry_points", lambda group: [])
    out = EngineCore.preprocess_add_request(_engine(), "engine-core-request")
    assert out == (req, 3) and calls == ["engine-core-request"]
    assert req.__dict__[pta.ATTR].tolist() == prompt


@pytest.mark.parametrize("source", ["configured", "installed"])
def test_custom_logits_processors_keep_the_list(monkeypatch, source):
    """A custom logits processor receives the prompt in its batch update."""
    req = _request(_prompt(50))
    _install_with_original(monkeypatch, req)
    installed = [object()] if source == "installed" else []
    monkeypatch.setattr(pta, "entry_points", lambda group: installed)
    engine = _engine(["my.LogitsProc"] if source == "configured" else None)
    EngineCore.preprocess_add_request(engine, "engine-core-request")
    assert pta.ATTR not in req.__dict__


def test_flag_off_installs_nothing(monkeypatch):
    before = EngineCore.preprocess_add_request
    monkeypatch.delenv("VLLM_TPU_PROMPT_TOKEN_ARRAY", raising=False)
    pta.patch_prompt_token_array()
    assert EngineCore.preprocess_add_request is before


def test_array_viewing_a_reused_queue_slot_is_copied_first():
    """A shared-memory reader unpickles an out-of-band array as a view of the
    ring slot (vLLM shm_broadcast.py MessageQueue.dequeue). The worker owns a
    copy before the writer reuses the slot."""
    prompt = _prompt(300_000, seed=3)
    data = NewRequestData(
        req_id="r",
        prompt_token_ids=pta.prompt_array(prompt),
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_ids=_blocks(),
        num_computed_tokens=0,
        lora_request=None,
    )
    stream, bufs = _pickle_like_mq(data)
    assert len(bufs) == 1
    slot = bytearray(bufs[0])
    received = pickle.loads(stream, buffers=[memoryview(slot)])
    assert not received.prompt_token_ids.flags.owndata
    viewed = received.prompt_token_ids
    pta.own_prompt_arrays([received])
    assert received.prompt_token_ids.flags.owndata
    slot[:] = bytes(len(slot))  # the writer reuses the slot
    assert int(viewed[5]) == 0  # the view changed with it
    assert received.prompt_token_ids.tolist() == prompt


def test_lists_and_owned_arrays_are_left_alone():
    lst = [1, 2, 3]
    owned = np.arange(4, dtype=np.int32)
    datas = [SimpleNamespace(prompt_token_ids=x) for x in (lst, owned, None)]
    pta.own_prompt_arrays(datas)
    assert [d.prompt_token_ids for d in datas[::2]] == [lst, None]
    assert datas[1].prompt_token_ids is owned


def test_runner_owns_prompt_arrays_before_the_base_adds_requests(monkeypatch):
    seen = []

    def base_update(_runner, scheduler_output):
        for data in scheduler_output.scheduled_new_reqs:
            seen.append(data.prompt_token_ids.flags.owndata)

    monkeypatch.setattr(GPUModelRunner, "_update_states", base_update)
    runner = object.__new__(TPUModelRunner)
    runner.requests = {}
    runner._mamba_state_pos = {}
    runner._mamba_offset_seeded = set()
    runner._pre_async_results = None
    runner._install_spec_token_room_guard = lambda: None
    so = SchedulerOutput.make_empty()
    base = np.arange(2 * 300_000, dtype=np.int32)
    so.scheduled_new_reqs = [SimpleNamespace(prompt_token_ids=base[::2])]
    TPUModelRunner._update_states(runner, so)
    assert seen == [True]


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


@pytest.mark.parametrize("n", [7, 96_661, 300_000])
def test_worker_input_batch_is_the_same(patched, n):
    prompt = _prompt(n, seed=n)
    req = _request(prompt)
    pta.attach(req)
    received = _mq_roundtrip(NewRequestData.from_request(req, _blocks()))
    pta.own_prompt_arrays([received])
    arr = received.prompt_token_ids
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
    # Penalties read the prompt from token_ids_cpu, now identical.
    assert torch.equal(
        from_list._make_prompt_token_ids_cpu_tensor(),
        from_array._make_prompt_token_ids_cpu_tensor(),
    )
    s_arr.output_token_ids.extend([11, 12])
    s_list.output_token_ids.extend([11, 12])
    idxs = (0, n // 2, n - 1, n, n + 1)
    for idx in idxs:
        assert int(s_arr.get_token_id(idx)) == s_list.get_token_id(idx)
    # The drafters' partial-prefill seed (vLLM llm_base_proposer.py and
    # vllm_torchtpu eagle3.py) builds int32 tensors from get_token_id.
    as_tensor = torch.tensor([s_arr.get_token_id(i) for i in idxs], dtype=torch.int32)
    assert as_tensor.tolist() == [s_list.get_token_id(i) for i in idxs]
    seed = np.zeros(len(idxs), dtype=np.int32)
    for k, i in enumerate(idxs):
        seed[k] = s_arr.get_token_id(i)
    assert seed.tolist() == as_tensor.tolist()
    # Re-adding the request (left out of a step, then scheduled again)
    # copies the same prompt again.
    from_array.remove_request("a")
    from_array.condense()
    from_array.add_request(s_arr)
    row = from_array.req_id_to_index["a"]
    assert from_array.token_ids_cpu[row, :n].tolist() == prompt
