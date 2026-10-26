# SPDX-License-Identifier: Apache-2.0
"""TPU_FAST_BLOCK_HASH: the fast prefix-cache block hasher must induce exactly
the same prefix-equality relation as vLLM's stock hasher, stay chained across
incremental calls, and fall back for requests that need extra keys."""

import random
import time

import pytest
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request

from vllm_torchtpu.core import fast_block_hasher as fbh

HBS = 16


def _req(ids, hasher, rid="r", cache_salt=None):
    ecr = EngineCoreRequest(
        request_id=rid, prompt_token_ids=list(ids), mm_features=None,
        sampling_params=SamplingParams(max_tokens=8), pooling_params=None,
        arrival_time=0.0, lora_request=None, cache_salt=cache_salt,
        data_parallel_rank=None)
    return Request.from_engine_core_request(ecr, hasher)


def _hashers(algo):
    fn = get_hash_fn_by_name(algo)
    init_none_hash(fn)
    stock = get_request_block_hasher(HBS, fn)
    return stock, fbh.make_fast_request_block_hasher(HBS, fn, stock)


@pytest.mark.parametrize("algo", ["sha256", "sha256_cbor", "xxhash"])
def test_same_prefix_relation_as_stock(algo):
    stock, fast = _hashers(algo)
    rng = random.Random(0)
    base = [rng.randrange(0, 200_000) for _ in range(16 * 40 + 7)]
    variants = [base, base[:16 * 23], base[:16 * 23 + 5],
                base[:16 * 17] + [base[16 * 17] + 1] + base[16 * 17 + 1:],
                [rng.randrange(0, 200_000) for _ in range(len(base))],
                base + [5] * 33]
    hs = [(_req(v, stock).block_hashes, _req(v, fast).block_hashes)
          for v in variants]
    for v, (s, f) in zip(variants, hs):
        assert len(s) == len(f) == len(v) // HBS
        assert len(set(f)) == len(f)
    for a in range(len(hs)):
        for b in range(len(hs)):
            sa, fa = hs[a]
            sb, fb = hs[b]
            for i in range(min(len(sa), len(sb))):
                assert (sa[i] == sb[i]) == (fa[i] == fb[i]), (a, b, i)


def test_chained_across_incremental_calls():
    stock, fast = _hashers("sha256")
    rng = random.Random(1)
    ids = [rng.randrange(0, 150_000) for _ in range(16 * 10 + 3)]
    one_shot = _req(ids + [7] * 45, fast).block_hashes
    r = _req(ids, fast)
    assert len(r.block_hashes) == 10
    for t in [7] * 45:  # decode appends; hashes extend block by block
        r.append_output_token_ids(t)
    assert r.block_hashes == one_shot


def test_cache_salt_falls_back_to_stock():
    stock, fast = _hashers("sha256")
    ids = list(range(100, 100 + 16 * 4))
    assert (_req(ids, fast, cache_salt="s").block_hashes
            == _req(ids, stock, cache_salt="s").block_hashes)
    assert _req(ids, fast).block_hashes != _req(ids, stock).block_hashes


def test_install_is_env_gated(monkeypatch):
    from vllm.v1.engine import core as engine_core
    orig = engine_core.get_request_block_hasher
    monkeypatch.setattr(engine_core, "_tpu_fast_block_hash", False,
                        raising=False)
    monkeypatch.delenv("TPU_FAST_BLOCK_HASH", raising=False)
    fbh.install()
    assert engine_core.get_request_block_hasher is orig
    monkeypatch.setenv("TPU_FAST_BLOCK_HASH", "1")
    fbh.install()
    try:
        assert engine_core.get_request_block_hasher is not orig
        fn = get_hash_fn_by_name("sha256")
        init_none_hash(fn)
        h = engine_core.get_request_block_hasher(HBS, fn)
        assert len(_req(range(16 * 3), h).block_hashes) == 3
    finally:
        engine_core.get_request_block_hasher = orig


def test_faster_than_stock_on_long_prompt():
    stock, fast = _hashers("sha256")
    rng = random.Random(2)
    ids = [rng.randrange(0, 150_000) for _ in range(100_000)]

    def best(h):
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            _req(ids, h)
            ts.append(time.perf_counter() - t)
        return min(ts)

    t_stock, t_fast = best(stock), best(fast)
    print(f"100k tokens: stock {t_stock*1e3:.1f} ms, fast {t_fast*1e3:.1f} ms")
    assert t_fast < t_stock / 2


def _reuse_hashers(algo="sha256", mb=64):
    fn = get_hash_fn_by_name(algo)
    init_none_hash(fn)
    stock = get_request_block_hasher(HBS, fn)
    plain = fbh.make_fast_request_block_hasher(HBS, fn, stock)
    reuse = fbh.make_fast_request_block_hasher(HBS, fn, stock, reuse_cache_mb=mb)
    return plain, reuse


def test_reuse_cache_is_exact_over_a_conversation():
    plain, reuse = _reuse_hashers()
    rng = random.Random(3)
    system = [rng.randrange(0, 150_000) for _ in range(16 * 30 + 5)]
    convs = [list(system) for _ in range(3)]
    for turn in range(6):
        for c, conv in enumerate(convs):
            conv += [rng.randrange(0, 150_000) for _ in range(rng.randrange(1, 300))]
            if turn == 3 and c == 1:  # history rewritten mid-prompt
                conv[16 * 40 + 3] += 1
            if turn == 4 and c == 2:  # truncated history
                del conv[16 * 31:16 * 35]
            assert (_req(conv, reuse).block_hashes
                    == _req(conv, plain).block_hashes), (turn, c)
    assert reuse.cache.hits > 0
    assert reuse.cache.reused_blocks > reuse.cache.total_blocks // 2


def test_reuse_cache_evicts_to_budget():
    _, reuse = _reuse_hashers(mb=1)
    rng = random.Random(4)
    for i in range(40):
        ids = [i] * 16 * 20 + [rng.randrange(0, 150_000) for _ in range(16 * 400)]
        _req(ids, reuse)
    c = reuse.cache
    assert 0 < c.nbytes <= 1 << 20
    assert sum(len(b) for b in c.buckets.values()) == len(c.order)


def test_reuse_faster_on_extended_prompt():
    plain, reuse = _reuse_hashers()
    rng = random.Random(5)
    ids = [rng.randrange(0, 150_000) for _ in range(150_000)]
    _req(ids, reuse)
    ext = ids + [rng.randrange(0, 150_000) for _ in range(500)]
    t = time.perf_counter()
    r1 = _req(ext, reuse).block_hashes
    t_reuse = time.perf_counter() - t
    t = time.perf_counter()
    r2 = _req(ext, plain).block_hashes
    t_plain = time.perf_counter() - t
    assert r1 == r2
    print(f"150k ext: plain {t_plain*1e3:.1f} ms, reuse {t_reuse*1e3:.1f} ms")
    assert t_reuse < t_plain / 2
