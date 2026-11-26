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

"""K3_PREFILL_SJF waiting queue: shortest uncached prompt first, aging so old
long prompts overtake new short ones, front requeues, peek/pop agreement."""

from vllm_torchtpu.core.sjf_queue import SJFRequestQueue


class _Request:
    def __init__(self, name, tokens, arrival):
        self.name, self.num_tokens, self.arrival_time = name, tokens, arrival
        self.block_hashes = []

    def __repr__(self):
        return self.name


def _queue(now, aging=1000.0):
    return SJFRequestQueue(lambda r: r.num_tokens, aging, clock=lambda: now[0])


def test_shortest_first_then_aging():
    now = [0.0]
    queue = _queue(now)
    long_ = _Request("long", 30000, 0.0)
    queue.add_request(long_)
    now[0] = 1.0
    short = _Request("short", 2000, 1.0)
    queue.add_request(short)
    now[0] = 2.0
    mid = _Request("mid", 8000, 2.0)
    queue.add_request(mid)
    assert queue.peek_request() is short
    assert queue.pop_request() is short
    assert queue.peek_request() is mid
    # After 40 s the old long prompt is ahead of a newly arrived short one.
    now[0] = 40.0
    queue.pop_request()
    newer = _Request("newer", 500, 40.0)
    queue.add_request(newer)
    order = list(queue)
    assert order.index(long_) < order.index(newer)


def test_prepend_goes_first_in_order():
    now = [0.0]
    queue = _queue(now)
    a, b, c = (_Request(n, 1, 0.0) for n in "abc")
    queue.add_request(c)
    queue.prepend_requests([a, b])
    assert list(queue)[:2] == [a, b]
    queue.prepend_request(c)  # latest prepend first
    assert queue.peek_request() is c


def test_pop_returns_peeked_even_if_keys_change():
    now = [0.0]
    queue = _queue(now)
    first = _Request("first", 1000, 0.0)
    second = _Request("second", 5000, 0.0)
    queue.add_request(first)
    queue.add_request(second)
    peeked = queue.peek_request()
    now[0] = 1e6
    assert queue.pop_request() is peeked


def test_remove_and_estimate_fallback():
    now = [0.0]

    def broken(_):
        raise RuntimeError("no cache lookup")

    queue = SJFRequestQueue(broken, 1000.0, clock=lambda: now[0])
    big, small = _Request("big", 9000, 0.0), _Request("small", 10, 0.0)
    queue.add_request(big)
    queue.add_request(small)  # falls back to num_tokens
    assert queue.peek_request() is small
    queue.remove_requests([small])
    assert list(queue) == [big]
    queue.remove_request(big)
    assert not queue


def test_prepend_and_remove_invalidate_stale_peeked():
    now = [0.0]
    queue = _queue(now)
    first = _Request("first", 1000, 0.0)
    urgent = _Request("urgent", 8000, 1.0)
    queue.add_request(first)
    assert queue.peek_request() is first
    queue.prepend_request(urgent)
    assert queue.pop_request() is urgent
    assert queue.peek_request() is first
    shorter = _Request("shorter", 100, 0.5)
    queue.add_request(shorter)
    assert queue._peeked is None
    assert queue.pop_request() is shorter
    assert queue.peek_request() is first
    queue.remove_request(first)
    assert queue._peeked is None


def test_aging_allows_waiting_long_request_to_overtake_waiting_short_request():
    now = [0.0]
    queue = _queue(now, aging=1000.0)
    long_ = _Request("long", 5000, 0.0)
    queue.add_request(long_)
    now[0] = 1.0
    short = _Request("short", 2000, 1.0)
    queue.add_request(short)
    assert list(queue) == [short, long_]
    now[0] = 6.0
    assert list(queue) == [long_, short]


def test_patch_vllm_prefill_sjf_queue_scopes_to_k3_megakernel_roles(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.core.sched import scheduler as sched_mod

    import vllm_torchtpu
    from vllm_torchtpu.core import sjf_queue
    from vllm_torchtpu.models.kimi_k3.megakernel import backend

    class _DummyScheduler:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.waiting = []
            self.kv_cache_manager = SimpleNamespace(
                get_computed_blocks=lambda req: ((), 0, False)
            )
            self.connector = None

    monkeypatch.setattr(sched_mod, "Scheduler", _DummyScheduler)
    monkeypatch.setattr(sjf_queue, "enabled", lambda: True)
    monkeypatch.setattr(sjf_queue, "aging", lambda: 1000.0)
    vllm_torchtpu._patch_vllm_prefill_sjf_queue()

    # Non-megakernel kv_producer (e.g. Gemma 4 / GLM-5.3 / operator K3) stays FCFS.
    monkeypatch.setattr(backend, "role", lambda _cfg: None)
    non_mk_cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
    )
    assert not isinstance(_DummyScheduler(non_mk_cfg).waiting, SJFRequestQueue)

    # K3 megakernel decode role does not get SJF waiting queue.
    monkeypatch.setattr(backend, "role", lambda _cfg: "decode")
    assert not isinstance(_DummyScheduler(SimpleNamespace()).waiting, SJFRequestQueue)

    # K3 megakernel prefill and aggregate roles enable SJFRequestQueue.
    for active_role in ("prefill", "aggregate"):
        monkeypatch.setattr(backend, "role", lambda _cfg, r=active_role: r)
        assert isinstance(_DummyScheduler(SimpleNamespace()).waiting, SJFRequestQueue)
