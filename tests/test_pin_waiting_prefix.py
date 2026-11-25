# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PIN_WAITING_PREFIX (core/pin_waiting_prefix.py) on a real BlockPool."""

from types import SimpleNamespace as NS

from vllm_torchtpu.core import pin_waiting_prefix as pw

BS = 16


def _request(req_id, n_tokens, seed=0):
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    return Request(
        req_id, prompt_token_ids=[seed * 100_000 + i for i in range(n_tokens)],
        sampling_params=SamplingParams(max_tokens=1), pooling_params=None,
        block_hasher=get_request_block_hasher(BS, sha256),
    )


class _Coordinator:
    """Single full-attention group: the contiguous cached run of block hashes."""

    def __init__(self, pool):
        self.pool = pool

    def find_longest_cache_hit(self, hashes, max_len):
        out = []
        for h in hashes[: max_len // BS]:
            got = self.pool.get_cached_block(h, [0])
            if got is None:
                break
            out.append(got[0])
        return (out,), len(out) * BS, 0


def _setup(n_blocks=10):
    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(num_gpu_blocks=n_blocks, enable_caching=True, hash_block_size=BS)
    manager = NS(block_pool=pool, enable_caching=True, coordinator=_Coordinator(pool))
    return pool, manager


def _cache_and_free(pool, req, n):
    blocks = pool.get_new_blocks(n)
    pool.cache_full_blocks(req, blocks, 0, n, BS, 0)
    pool.free_blocks(blocks)
    return blocks


def test_a_waiting_request_keeps_its_prefix_until_scheduled():
    from vllm.v1.request import RequestStatus

    pool, manager = _setup()
    old = _request("old", 3 * BS + 1, seed=1)
    _cache_and_free(pool, old, 3)  # the session's previous turn, now free
    pins = pw._Pins(0.5)
    old.status = RequestStatus.WAITING
    pw.pin_new(pins, [old], manager)
    assert pins.by_req["old"] and pins.blocks == 3
    # Fill the pool with other work: the pinned prefix cannot be evicted.
    pool.get_new_blocks(pool.get_num_free_blocks())
    assert all(pool.get_cached_block(h, [0]) is not None for h in old.block_hashes[:3])
    # Scheduled: the pin goes back at the next step.
    old.status = RequestStatus.RUNNING
    pw.release_done(pins, {"old": old}, pool)
    assert pins.blocks == 0 and not pins.by_req
    assert pins.counts["released_reqs"] == 1


def test_without_a_pin_the_queued_prefix_is_evicted():
    pool, _ = _setup()
    old = _request("old", 3 * BS + 1, seed=1)
    _cache_and_free(pool, old, 3)
    pool.get_new_blocks(pool.get_num_free_blocks())
    assert pool.get_cached_block(old.block_hashes[0], [0]) is None


def test_budget_and_try_once():
    from vllm.v1.request import RequestStatus

    pool, manager = _setup(n_blocks=10)  # 9 usable; budget 0.3 -> 3 blocks
    a, b = _request("a", 3 * BS + 1, seed=1), _request("b", 3 * BS + 1, seed=2)
    _cache_and_free(pool, a, 3)
    _cache_and_free(pool, b, 3)
    pins = pw._Pins(0.3)
    for r in (a, b):
        r.status = RequestStatus.WAITING
    pw.pin_new(pins, [a, b], manager)
    assert list(pins.by_req) == ["a"] and pins.counts["over_budget"] == 1
    # a leaves; b is pinned next time; a is not tried again.
    a.status = RequestStatus.FINISHED_ABORTED
    pw.release_done(pins, {"a": a, "b": b}, pool)
    pw.pin_new(pins, [a, b], manager)
    assert list(pins.by_req) == ["b"]
    cold = _request("cold", 2 * BS + 1, seed=3)
    cold.status = RequestStatus.WAITING
    fresh = pw._Pins(0.5)
    pw.pin_new(fresh, [cold], manager)
    assert "cold" not in fresh.by_req and fresh.counts["zero_hit"] == 1


def test_a_request_gone_from_the_scheduler_releases():
    from vllm.v1.request import RequestStatus

    pool, manager = _setup()
    r = _request("r", 2 * BS + 1, seed=4)
    _cache_and_free(pool, r, 2)
    r.status = RequestStatus.WAITING
    pins = pw._Pins(0.5)
    pw.pin_new(pins, [r], manager)
    free_before = pool.get_num_free_blocks()
    pw.release_done(pins, {}, pool)
    assert pool.get_num_free_blocks() == free_before + 2


def test_reserve_releases_latest_queued_pins_first():
    """Pins never squeeze scheduling: below the free reserve, the last-queued pins go first."""
    from vllm.v1.request import RequestStatus

    pool, manager = _setup(n_blocks=13)  # 12 usable
    a, b = _request("a", 3 * BS + 1, seed=1), _request("b", 3 * BS + 1, seed=2)
    _cache_and_free(pool, a, 3)
    _cache_and_free(pool, b, 3)
    for r in (a, b):
        r.status = RequestStatus.WAITING
    pins = pw._Pins(1.0)
    pw.pin_new(pins, [a, b], manager)
    assert set(pins.by_req) == {"a", "b"} and pool.get_num_free_blocks() == 6
    taken = pool.get_new_blocks(4)  # running work leaves 2 free
    pw.keep_reserve(pins, [a, b], pool, reserve_frac=4 / 13)  # need 4 free
    assert list(pins.by_req) == ["a"] and pool.get_num_free_blocks() == 5
    pool.free_blocks(taken)
    # With the reserve not met, nothing new is pinned.
    c = _request("c", 2 * BS + 1, seed=5)
    _cache_and_free(pool, c, 2)
    c.status = RequestStatus.WAITING
    pool.get_new_blocks(pool.get_num_free_blocks() - 1)
    pw.pin_new(pins, [c], manager, reserve_frac=0.5)
    assert "c" not in pins.by_req
