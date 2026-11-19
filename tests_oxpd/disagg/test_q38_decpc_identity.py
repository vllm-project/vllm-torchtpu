# SPDX-License-Identifier: Apache-2.0
"""The decode prefix-cache identity driver's plan, parsing and verdict."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "examples/disagg"))

import q38_decpc_identity as ident  # noqa: E402


def test_plan_is_deterministic_and_each_turn_extends_the_last():
    a = ident.plan(7, [100, 300], [10, 20])
    assert a == ident.plan(7, [100, 300], [10, 20])
    assert a != ident.plan(8, [100, 300], [10, 20])
    for turns in a:
        assert [len(t) for t in turns] == [len(turns[0]), len(turns[0]) + 10, len(turns[0]) + 30]
        for prev, nxt in zip(turns, turns[1:]):
            assert nxt[: len(prev)] == prev
    # Conversations share no prefix.
    assert a[0][0][:50] != a[1][0][:50]


def test_parse_metrics_sums_labelled_samples_with_or_without_total():
    text = "\n".join(
        [
            "# HELP vllm:prefix_cache_hits_total x",
            'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 100.0',
            'vllm:prefix_cache_hits_total{engine="1",model_name="m"} 28.0',
            "vllm:spec_decode_num_drafts 5",
            "vllm:prefix_cache_hits_created 1.7e9",
        ]
    )
    got = ident.parse_metrics(text, ("vllm:prefix_cache_hits", "vllm:spec_decode_num_drafts"))
    assert got == {"vllm:prefix_cache_hits": 128.0, "vllm:spec_decode_num_drafts": 5.0}


def _record(turn, mode, tokens=(1, 2, 3), spec=(4, 12, 9), hit=0):
    return {
        "conv": 0,
        "turn": turn,
        "mode": mode,
        "token_ids": list(tokens),
        "spec": list(spec),
        "local_hit_tokens": hit,
    }


def _good():
    return [
        _record(0, "ref"),
        _record(0, "ref2"),
        _record(0, "hit", hit=38912),
        _record(1, "natural", hit=38912),
        _record(1, "ref"),
        _record(1, "ref2"),
        _record(1, "hit", hit=40960),
    ]


def test_compare_passes_identical_runs_with_hits():
    assert ident.compare(_good()) == []


def test_compare_fails_any_difference_or_a_missing_hit():
    for index, change, reason in [
        (2, {"token_ids": [1, 2, 4]}, "token ids differ"),
        (3, {"spec": [4, 12, 8]}, "speculative counters differ"),
        (6, {"local_hit_tokens": 0}, "no decode prefix hit"),
        (4, {"local_hit_tokens": 2048}, "the reset did not clear the decode"),
        (5, {"local_hit_tokens": 2048}, "the reset did not clear the decode"),
        (5, {"error": "ReadTimeout"}, "missing or failed"),
        (2, {"error": "ReadTimeout"}, "ReadTimeout"),
    ]:
        records = _good()
        records[index] = {**records[index], **change}
        failures = ident.compare(records)
        assert [f[3] for f in failures] == [reason], (index, failures)
    records = [r for r in _good() if r["mode"] != "natural"]
    assert ident.compare(records) == [(0, 1, "natural", "missing")]


def _swap(same):
    return [
        {"conv": "swap", "turn": 1, "mode": "swap_ref", "token_ids": [5, 6, 7], "spec": [0, 0, 0], "local_hit_tokens": 0},
        {"conv": "swap", "turn": 1, "mode": "swap", "token_ids": [5, 6, 7] if same else [5, 6, 8]},
    ]


def test_the_wrong_landing_mutant_must_bite_and_is_kept_out_of_the_comparison():
    assert ident.swap_bites(_good() + _swap(same=False))
    assert not ident.swap_bites(_good() + _swap(same=True))
    assert not ident.swap_bites(_good())
    errored = _swap(same=False)
    errored[1] = {**errored[1], "error": "HTTPError"}
    assert not ident.swap_bites(errored)
    assert ident.compare(_good() + _swap(same=False)) == []


def test_verdict_needs_identity_and_a_biting_mutant():
    assert ident.verdict(_good() + _swap(same=False))["pass"]
    assert not ident.verdict(_good() + _swap(same=True))["pass"]
    bad = _good()
    bad[2] = {**bad[2], "token_ids": [9]}
    v = ident.verdict(bad + _swap(same=False))
    assert not v["pass"] and v["mutant_bites"] and v["turns"] == 2
    assert v["status"] == "FAIL"
    assert ident.verdict(_good() + _swap(same=True))["status"] == "VOID"
    assert ident.verdict(_good() + _swap(same=False))["status"] == "PASS"


def test_fail_fast_only_on_a_failed_full_landing():
    assert not ident.fail_fast(_good())
    hit_failed = _good()
    hit_failed[2] = {**hit_failed[2], "error": "RemoteProtocolError"}
    assert not ident.fail_fast(hit_failed)  # a hit failure is identity data
    ref_failed = _good()
    ref_failed[0] = {**ref_failed[0], "error": "RemoteProtocolError"}
    assert ident.fail_fast(ref_failed)
    assert not ident.verdict(ref_failed)["pass"]


def test_a_difference_is_noise_limited_only_where_ref2_also_differs():
    # hit differs from ref, and ref2 (same prompt, same configuration, cache
    # empty) differs from ref too: the engine is not repeatable here.
    noisy = _good()
    noisy[1] = {**noisy[1], "token_ids": [1, 2, 9]}  # ref2 of turn 0
    noisy[2] = {**noisy[2], "token_ids": [1, 2, 4]}  # hit of turn 0
    v = ident.verdict(noisy + _swap(same=False))
    assert v["failures"] == [] and not v["pass"]
    assert v["status"] == "NOISE-LIMITED"
    assert v["noise_limited"] == [(0, 0, "hit", "token ids differ (ref2 differs from ref too)")]
    # The same hit difference with ref2 == ref is a real FAIL.
    real = _good()
    real[2] = {**real[2], "token_ids": [1, 2, 4]}
    v = ident.verdict(real + _swap(same=False))
    assert v["status"] == "FAIL" and v["noise_limited"] == []
    assert v["failures"] == [(0, 0, "hit", "token ids differ")]
    # A ref2 difference alone (hit equals ref) is neither.
    floor_only = _good()
    floor_only[1] = {**floor_only[1], "spec": [4, 12, 8]}
    v = ident.verdict(floor_only + _swap(same=False))
    assert v["status"] == "PASS" and v["aa_floor_differs"] == [(0, 0)]
    assert ident.verdict(_good() + _swap(same=False))["aa_floor_differs"] == []
