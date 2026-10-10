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
"""TPU_SCHED_TRIM_RESEND_TOKEN_IDS: the trimmed CachedRequestData.all_token_ids
must give the worker exactly the tokens upstream GPUModelRunner._update_states
reads (``all_token_ids[req_id][-num_output_tokens:]`` when
``num_output_tokens > 0``)."""

import pickle
import random
import time

from vllm.v1.core.sched.output import CachedRequestData

from vllm_torchtpu.core.prefix_aware_scheduler import (
    PrefixAwareScheduler,
    trim_resend_token_ids,
)


def _worker_view(data: CachedRequestData, req_id: str):
    """What upstream v0.29.0 gpu_model_runner.py:1469-1473 consumes."""
    i = data.req_ids.index(req_id)
    n = data.num_output_tokens[i]
    if n > 0 and req_id in data.all_token_ids:
        return data.all_token_ids[req_id][-n:]
    return None


def _make(rng, n_reqs):
    req_ids, all_ids, n_out, resumed = [], {}, [], set()
    for k in range(n_reqs):
        rid = f"r{k}"
        req_ids.append(rid)
        plen = rng.choice([0, 1, 5, 300, 94_000])
        olen = rng.choice([0, 0, 1, 3, 17, 4000])
        n = olen + rng.choice([0, 0, 1, 3])  # + async/spec placeholders
        n_out.append(n)
        if rng.random() < 0.7:
            all_ids[rid] = [rng.randrange(250_000) for _ in range(plen + olen)]
        if rng.random() < 0.1:
            resumed.add(rid)
    return CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids=resumed,
        new_token_ids=[],
        all_token_ids=all_ids,
        new_block_ids=[None] * n_reqs,
        num_computed_tokens=[0] * n_reqs,
        num_output_tokens=n_out,
    )


def test_trim_is_exact_for_worker_reads():
    rng = random.Random(0)
    for _ in range(300):
        data = _make(rng, rng.randint(1, 8))
        before = {rid: _worker_view(data, rid) for rid in data.req_ids}
        full = {rid: list(v) for rid, v in data.all_token_ids.items()}
        trim_resend_token_ids(data)
        after = {rid: _worker_view(data, rid) for rid in data.req_ids}
        assert before == after
        assert set(full) == set(data.all_token_ids)
        for rid in data.resumed_req_ids:
            if rid in full:
                assert data.all_token_ids[rid] == full[rid]
        for i, rid in enumerate(data.req_ids):
            if rid in full and rid not in data.resumed_req_ids:
                assert len(data.all_token_ids[rid]) <= max(data.num_output_tokens[i], 0)


def test_trim_shrinks_payload_at_94k():
    ids = list(range(1, 94_001))
    data = CachedRequestData(
        req_ids=["a"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={"a": ids + [7, 8, 9]},
        new_block_ids=[None],
        num_computed_tokens=[94_000],
        num_output_tokens=[3],
    )
    big = len(pickle.dumps(data))
    t0 = time.perf_counter()
    saved = trim_resend_token_ids(data)
    dt = time.perf_counter() - t0
    assert saved == 94_000
    assert data.all_token_ids == {"a": [7, 8, 9]}
    assert len(pickle.dumps(data)) < big // 100
    assert dt < 0.01


def test_scheduler_override_and_kill_switch(monkeypatch):
    calls = []

    def fake_super(self, *a, **k):
        calls.append(1)
        return CachedRequestData(
            req_ids=["a"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={"a": [1, 2, 3, 4, 5]},
            new_block_ids=[None],
            num_computed_tokens=[3],
            num_output_tokens=[2],
        )

    base = PrefixAwareScheduler.__mro__[1]
    monkeypatch.setattr(base, "_make_cached_request_data", fake_super)

    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    out = s._make_cached_request_data([], [], {}, {}, {})
    assert out.all_token_ids == {"a": [4, 5]}

    monkeypatch.setenv("TPU_SCHED_TRIM_RESEND_TOKEN_IDS", "0")
    s2 = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    out2 = s2._make_cached_request_data([], [], {}, {}, {})
    assert out2.all_token_ids == {"a": [1, 2, 3, 4, 5]}
    assert len(calls) == 2
