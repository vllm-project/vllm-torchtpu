# SPDX-License-Identifier: Apache-2.0
"""K3_PREFILL_SJF waiting queue: shortest uncached prompt first, aging so old
long prompts overtake new short ones, front requeues, peek/pop agreement."""

import pytest

from vllm_torchtpu.core.sjf_queue import SJFRequestQueue

pytestmark = pytest.mark.cpu_test


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
