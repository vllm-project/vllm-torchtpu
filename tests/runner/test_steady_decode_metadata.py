# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_STEADY_DECODE_METADATA: the incremental metadata path must hand
the model bitwise the same tensors as the full rebuild, every step."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.runner.steady_decode_fakes import Workload, env, make_runner, run_prepare
from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.runner import steady_decode_metadata as steady_md


def _assert_same(a, b, where):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), where
        assert a.dtype == b.dtype and a.shape == b.shape, where
        assert torch.equal(a, b), where
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and np.array_equal(a, b), where
    else:
        assert a == b, where


def _assert_metadata_equal(full, inc, step):
    for key in ("input_ids", "position_ids", "qsl_np", "seq_lens_np"):
        _assert_same(full[key], inc[key], f"step {step}: {key}")
    assert full["metadata"].keys() == inc["metadata"].keys()
    for layer, md_full in full["metadata"].items():
        md_inc = inc["metadata"][layer]
        assert type(md_full) is type(md_inc), (step, layer)
        for name in AttentionMetadata.__dataclass_fields__:
            if name in ("query_start_loc_cpu", "seq_lens_cpu"):
                continue
            _assert_same(
                getattr(md_full, name),
                getattr(md_inc, name),
                f"step {step}: {layer}.{name}",
            )
    for name in ("seq_lens", "query_start_loc", "request_distribution"):
        _assert_same(
            getattr(full["ctx"], name),
            getattr(inc["ctx"], name),
            f"step {step}: {name}",
        )
    a = full["ctx"].unified_mamba_state_indices
    b = inc["ctx"].unified_mamba_state_indices
    assert len(a) == len(b), step
    for x, y in zip(a, b):
        _assert_same(x, y, f"step {step}: unified_mamba_state_indices")


# The c1/c4 forward-path host flags H1 ships next to (H2 et al.).
HOST_FLAGS = {
    "VLLM_TPU_PREFLAT_FWD_ARGS": "1",
    "VLLM_TPU_TOKEN_PADDING_REUSE": "1",
    "VLLM_TPU_SUBST_STATIC_ARGS": "1",
    "VLLM_TPU_FAST_TOKEN_SUBSTITUTION": "1",
    "VLLM_TPU_SUBST_FRESH_IDS": "1",
}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("num_reqs", [1, 4])
def test_incremental_metadata_matches_full_rebuild(seed, num_reqs):
    """Random acceptance (0..K), block-boundary crossings, request
    add/finish/reorder and prefill pieces: the flag-on runner's metadata
    equals the flag-off runner's at every step, and the incremental paths
    (count-only reuse and the skipped mamba walk) actually ran."""
    _run_equivalence(seed, num_reqs, {}, fast_token_substitution=False)


@pytest.mark.parametrize("num_reqs", [1, 4])
def test_incremental_metadata_composes_with_host_flags(num_reqs):
    """Same check with the other c1/c4 host-path flags on for both runners
    (fast token substitution caches the combined request distribution)."""
    _run_equivalence(3, num_reqs, HOST_FLAGS, fast_token_substitution=True)


def _run_equivalence(seed, num_reqs, extra, fast_token_substitution):
    wl = Workload(seed, num_reqs, max_model_len=8192, churn=0.06, prefill=0.04)
    full = make_runner(max_model_len=8192)
    inc = make_runner(max_model_len=8192)
    full._fast_token_substitution = inc._fast_token_substitution = (
        fast_token_substitution
    )
    for step in range(300):
        if step:
            wl.step()
        num_decode, num_windowed = wl.segments()
        wl.apply(full)
        wl.apply(inc)
        so = wl.scheduler_output()
        with env(steady=False, extra=extra):
            out_full = run_prepare(full, so, num_decode, num_windowed)
        with env(steady=True, extra=extra):
            out_inc = run_prepare(inc, so, num_decode, num_windowed)
        _assert_metadata_equal(out_full, out_inc, step)
    cache = inc._steady_decode_cache
    assert cache.hits > 100, (cache.hits, cache.misses)
    assert cache.misses > 5
    assert 0 < cache.walk_skips < cache.walks
    assert full._steady_decode_cache.hits == full._steady_decode_cache.misses == 0


def test_flag_off_is_a_no_op():
    """Off (the default) the runner never consults or seeds the cache, the
    walk carries no steady-decode bounds and nothing caches walk products."""
    assert envs.VLLM_TPU_STEADY_DECODE_METADATA is False
    wl = Workload(7, 2, max_model_len=8192)
    runner = make_runner(max_model_len=8192)
    for step in range(20):
        if step:
            wl.step()
        num_decode, num_windowed = wl.segments()
        wl.apply(runner)
        with env(steady=False):
            out = run_prepare(runner, wl.scheduler_output(), num_decode, num_windowed)
        assert out["ctx"].steady_walk_bounds is None
        assert out["ctx"].steady_hit is False
    cache = runner._steady_decode_cache
    assert cache.hits == cache.misses == cache.walks == 0
    assert cache.entry is None
    assert "_steady_walk_products" not in runner.__dict__


def test_key_rejects_non_steady_chunks():
    ids = ("a", "b")
    four = np.array([4, 4], dtype=np.int32)
    drafts = np.array([3, 3], dtype=np.int32)
    kw = dict(
        start_index=0,
        batch_num_reqs=2,
        use_max_model_len=True,
        target_num_reqs=8,
        padded_num_reqs=8,
        rpa_num_decode_reqs=0,
        num_windowed_reqs=2,
    )
    base = steady_md.steady_decode_key(ids, four, drafts, **kw)
    assert base is not None
    assert base == steady_md.steady_decode_key(ids, four.copy(), drafts.copy(), **kw)
    # Prefill piece: more tokens than drafts + 1.
    assert (
        steady_md.steady_decode_key(ids, np.array([4, 9], np.int32), drafts, **kw)
        is None
    )
    # Second chunk / partial chunk.
    assert (
        steady_md.steady_decode_key(ids, four, drafts, **{**kw, "start_index": 1})
        is None
    )
    assert (
        steady_md.steady_decode_key(ids, four, drafts, **{**kw, "batch_num_reqs": 3})
        is None
    )
    # Reorder, different counts and different buckets change the key.
    assert steady_md.steady_decode_key(("b", "a"), four, drafts, **kw) != base
    fewer = np.array([2, 2], dtype=np.int32)
    assert steady_md.steady_decode_key(ids, fewer + 1, fewer, **kw) != base
    assert (
        steady_md.steady_decode_key(ids, four, drafts, **{**kw, "padded_num_reqs": 16})
        != base
    )
    # Plain decode without speculative decoding.
    ones = np.ones(2, dtype=np.int32)
    assert steady_md.steady_decode_key(ids, ones, None, **kw) is not None
    assert steady_md.steady_decode_key(ids, ones + 1, None, **kw) is None


def test_scheduler_output_is_quiet():
    quiet = SimpleNamespace(
        scheduled_new_reqs=[],
        finished_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )
    assert steady_md.scheduler_output_is_quiet(quiet)
    for field, value in (
        ("scheduled_new_reqs", [object()]),
        ("finished_req_ids", {"x"}),
        ("preempted_req_ids", {"x"}),
        ("scheduled_cached_reqs", SimpleNamespace(resumed_req_ids={"x"})),
        (
            "scheduled_cached_reqs",
            SimpleNamespace(resumed_from_preemption=[False, True]),
        ),
    ):
        noisy = SimpleNamespace(**{**vars(quiet), field: value})
        assert not steady_md.scheduler_output_is_quiet(noisy), field


def test_mamba_offsets_stable_matches_brute_force():
    """The endpoint check equals checking every device seq_lens value the two
    bounds allow."""
    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randrange(1, 4)
        margin = rng.randrange(0, 5)
        geom = tuple(
            (0, rng.choice([4, 8, 64]), rng.choice([4, 16, 64]))
            for _ in range(rng.randrange(1, 3))
        )
        tail = bytes(rng.randrange(2))
        prev_hi = np.array([rng.randrange(0, 200) for _ in range(n)], dtype=np.int64)
        hi = prev_hi + np.array([rng.randrange(0, 6) for _ in range(n)])
        prev = (prev_hi - margin, prev_hi, tail)
        cur = (hi - margin, hi, tail if rng.random() < 0.95 else b"z")
        expect = cur[2] == tail
        for i in range(n):
            values = range(
                int(min(prev[0][i], cur[0][i])), int(max(prev_hi[i], hi[i])) + 1
            )
            for _, t, bs in geom:
                seen = {(v > 0, min(max((v - 1) // bs, 0), t - 1)) for v in values}
                expect = expect and len(seen) == 1 and min(values) > 0
        assert steady_md.mamba_offsets_stable(prev, cur, geom) == expect


def test_walk_seq_bounds():
    seq = np.array([10, 20, 0, 0, 7], dtype=np.int32)
    lo, hi, tail = steady_md.walk_seq_bounds(seq, 2, 4, margin=4)
    assert lo.tolist() == [6, 16] and hi.tolist() == [10, 20]
    assert tail == np.array([0, 0], dtype=np.int32).tobytes()
