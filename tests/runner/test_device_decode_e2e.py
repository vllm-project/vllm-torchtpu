# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Device decode end to end, through the runner's execute_model and
sample_tokens: a step samples the same tokens and logprobs on the device path
as on the stock path, and leaves the same host state, whatever ran before."""

import numpy as np
import pytest
from device_decode_test_utils import SimEngine


def assert_same_host_state(a: SimEngine, b: SimEngine) -> None:
    """The runners' request states and persistent batches agree."""
    assert a.runner.requests.keys() == b.runner.requests.keys()
    for req_id, state in a.runner.requests.items():
        other = b.runner.requests[req_id]
        assert state.output_token_ids == other.output_token_ids, req_id
        assert state.num_computed_tokens == other.num_computed_tokens, req_id
    batch, other = a.runner.input_batch, b.runner.input_batch
    assert batch.req_ids == other.req_ids
    num_reqs = batch.num_reqs
    ends = batch.num_tokens_no_spec[:num_reqs]
    assert np.array_equal(ends, other.num_tokens_no_spec[:num_reqs])
    for row, end in enumerate(ends):
        assert np.array_equal(
            batch.token_ids_cpu[row, :end], other.token_ids_cpu[row, :end]
        ), batch.req_ids[row]


class Lockstep:
    """Engines given the same requests and stepped together, their host
    states compared after every step."""

    def __init__(self, *engines: SimEngine):
        self.engines = engines

    def add(self, req_id: str, prompt_len: int, **params) -> None:
        for engine in self.engines:
            engine.add(req_id, prompt_len, **params)

    def abort(self, req_id: str) -> None:
        for engine in self.engines:
            engine.sched.abort(req_id)

    def preempt(self, req_id: str) -> None:
        for engine in self.engines:
            engine.sched.preempt(req_id)

    def run(self, steps: int) -> None:
        for _ in range(steps):
            for engine in self.engines:
                engine.step()
            first, *rest = self.engines
            for engine in rest:
                assert_same_host_state(first, engine)

    def run_until_done(self, max_steps: int = 60) -> None:
        for _ in range(max_steps):
            if not any(engine.sched.reqs for engine in self.engines):
                return
            self.run(1)
        raise AssertionError("requests left after max_steps")


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_device_steps_sample_what_the_stock_path_samples(monkeypatch, async_scheduling):
    device = SimEngine(monkeypatch, async_scheduling=async_scheduling, device=True)
    stock = SimEngine(monkeypatch, async_scheduling=async_scheduling, device=True)
    # The reference takes the stock path every step, with the keyed noise.
    monkeypatch.setattr(stock.controller, "try_execute", lambda *_args: False)
    engines = Lockstep(device, stock)
    # Prefill, decode; a request joins (a mixed step); a request is aborted
    # and resubmitted under its id; one is preempted and resumes.
    engines.add("a", 5, temperature=0.8, max_tokens=14)
    engines.add("b", 9, temperature=0.0, max_tokens=14)
    engines.add("c", 20, temperature=0.8, top_k=5, top_p=0.9, logprobs=0)
    engines.run(4)
    engines.add("d", 30, temperature=1.0, logprobs=3, max_tokens=10)
    engines.run(3)
    engines.abort("a")
    engines.add("a", 7, temperature=0.7, max_tokens=8)
    engines.run(3)
    engines.preempt("c")
    engines.run_until_done()

    assert device.tokens == stock.tokens
    assert device.logprobs == stock.logprobs
    assert [len(tokens) for tokens in device.tokens["b"]] == [14]
    assert [len(tokens) for tokens in device.tokens["c"]] == [12]
    assert [len(rows) for rows in device.logprobs["d"]] == [10]
    # Both paths ran, and stock steps followed device steps and vice versa.
    counters = device.controller.counters()
    assert counters["device_steps"] >= 8, counters
    assert counters["not_pure_decode"] >= 3, counters
    assert device.controller.enabled
    slots = device.controller.slots
    null = slots.config.null_slot
    assert int(slots.num_computed[null]) == 0 and int(slots.last_token[null]) == 0


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_greedy_tokens_equal_the_default_path(monkeypatch, async_scheduling):
    """Greedy sampling uses no noise: flag on and flag off agree."""
    on = SimEngine(monkeypatch, async_scheduling=async_scheduling, device=True)
    off = SimEngine(monkeypatch, async_scheduling=async_scheduling, device=False)
    engines = Lockstep(on, off)
    engines.add("a", 5, temperature=0.0, logprobs=2)
    engines.add("b", 13, temperature=0.0)
    engines.run(3)
    engines.add("c", 21, temperature=0.0, max_tokens=6)
    engines.run_until_done()
    assert on.tokens == off.tokens
    assert on.logprobs == off.logprobs
    assert on.controller.counters()["device_steps"] > 0
