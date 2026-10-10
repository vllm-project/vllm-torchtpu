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
"""TPU_SCHED_PACK_PROMPT_IDS: packed int32 prompt ids must reach the worker's
InputBatch exactly as the list would, through the same pickling as
vLLM's shm MessageQueue."""

import copyreg
import io
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_torchtpu.core.packed_ids import (
    PackedPromptIds,
    pack_new_request_prompt_ids,
    pack_prompt_ids,
)
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler


def _mq_roundtrip(obj):
    """Pickle exactly like vLLM v0.29 MessageQueue.enqueue/dequeue."""
    oob = []

    def oob_callback(buf):
        raw = buf.raw()
        if len(raw) < 1024 * 1024:
            return True
        oob.append(raw)
        return False

    dispatch_table = dict(copyreg.dispatch_table)
    with io.BytesIO() as bio:
        p = pickle.Pickler(bio, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=oob_callback)
        p.dispatch_table = dispatch_table
        p.dump(obj)
        data = bio.getvalue()
    return pickle.loads(data, buffers=oob), data, oob


def _new_req(ids, **sp_kwargs):
    return NewRequestData(
        req_id="r",
        prompt_token_ids=ids,
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=8, **sp_kwargs),
        pooling_params=None,
        block_ids=([1, 2],),
        num_computed_tokens=0,
        lora_request=None,
    )


@pytest.mark.parametrize("n", [1024, 94_000, 262_143, 300_000])
def test_roundtrip_is_exact_inline_and_plain(n):
    rng = random.Random(n)
    ids = [rng.randrange(248_077) for _ in range(n)]
    so = SimpleNamespace(scheduled_new_reqs=[_new_req(list(ids))])
    assert pack_new_request_prompt_ids(so) == n
    assert isinstance(so.scheduled_new_reqs[0].prompt_token_ids, PackedPromptIds)
    out, data, oob = _mq_roundtrip(so.scheduled_new_reqs[0])
    assert not oob  # bytes are always inline: never aliases the shm ring
    got = out.prompt_token_ids
    assert type(got) is np.ndarray and got.dtype == np.int32
    assert not got.flags.writeable
    assert got.tolist() == ids
    assert len(data) < 4 * n + 4096
    assert out.prompt_len == n


def test_list_pickle_baseline_is_larger():
    ids = list(range(100_000, 194_000))
    _, d_list, _ = _mq_roundtrip(_new_req(list(ids)))
    _, d_packed, _ = _mq_roundtrip(_new_req(pack_prompt_ids(ids)))
    assert len(d_packed) < len(d_list)


def test_worker_input_batch_identical(monkeypatch):
    import vllm.v1.worker.gpu_input_batch as gib
    import vllm.v1.worker.block_table as bt

    # CPU-only test host: no pinned-memory allocator.
    monkeypatch.setattr(gib, "PIN_MEMORY", False, raising=False)
    monkeypatch.setattr(bt, "PIN_MEMORY", False, raising=False)
    rng = random.Random(1)
    n = 94_000
    ids = [rng.randrange(248_077) for _ in range(n)]
    packed, _, _ = _mq_roundtrip(pack_prompt_ids(ids))
    rows = []
    for prompt in (list(ids), packed):
        ib = InputBatch(
            max_num_reqs=2,
            max_model_len=131_072,
            max_num_batched_tokens=8192,
            device=torch.device("cpu"),
            vocab_size=248_077,
            block_sizes=[256],
            kernel_block_sizes=[256],
            max_num_blocks_per_req=[512],
        )
        st = CachedRequestState(
            req_id="r",
            prompt_token_ids=prompt,
            mm_features=[],
            sampling_params=SamplingParams(max_tokens=8),
            generator=None,
            block_ids=([1, 2, 3],),
            num_computed_tokens=0,
            output_token_ids=[7, 8],
        )
        assert st.num_prompt_tokens == n and st.num_tokens == n + 2
        assert [int(st.get_token_id(i)) for i in (0, 1, n - 1, n, n + 1, n + 2)] == [
            ids[0], ids[1], ids[-1], 7, 8, -1
        ]
        idx = ib.add_request(st)
        rows.append(
            (
                ib.token_ids_cpu[idx, : n + 2].copy(),
                int(ib.num_prompt_tokens[idx]),
                int(ib.num_tokens_no_spec[idx]),
                ib.is_token_ids[idx, : n + 2].copy(),
            )
        )
    assert np.array_equal(rows[0][0], rows[1][0])
    assert rows[0][1:3] == rows[1][1:3]
    assert np.array_equal(rows[0][3], rows[1][3])


def test_exclusions_and_small_prompts():
    ids = list(range(5000))
    cases = [
        _new_req(list(ids), thinking_token_budget=16),
        _new_req(list(ids), prompt_logprobs=1),
        _new_req(list(ids), extra_args={"x": 1}),
        _new_req(list(range(100))),
    ]
    mm = _new_req(list(ids))
    mm.mm_features = [object()]
    cases.append(mm)
    so = SimpleNamespace(scheduled_new_reqs=cases)
    assert pack_new_request_prompt_ids(so) == 0
    assert all(type(r.prompt_token_ids) is list for r in cases)
    assert pack_prompt_ids([1, 2**40]) is None


def test_scheduler_hook_and_kill_switch(monkeypatch):
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    so = SimpleNamespace(scheduled_new_reqs=[_new_req(list(range(4096)))])
    s._maybe_pack_prompt_ids(so)
    assert isinstance(so.scheduled_new_reqs[0].prompt_token_ids, PackedPromptIds)

    monkeypatch.setenv("TPU_SCHED_PACK_PROMPT_IDS", "0")
    s2 = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    so2 = SimpleNamespace(scheduled_new_reqs=[_new_req(list(range(4096)))])
    s2._maybe_pack_prompt_ids(so2)
    assert type(so2.scheduled_new_reqs[0].prompt_token_ids) is list
