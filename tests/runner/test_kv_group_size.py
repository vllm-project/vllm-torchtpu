# SPDX-License-Identifier: Apache-2.0
"""KV cache group size rule (vllm_torchtpu.kv_group_size)."""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/kv_group_size.py"
_spec = importlib.util.spec_from_file_location("kv_group_size", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
size = _mod.kv_cache_group_size


def test_qwen38_pipeline_prefill_groups_by_twelve():
    # 36 GDN, 1 PLE, 12 main KV, 12 compressed keys, 12 raw rings.
    assert size([36, 1, 12, 12, 12]) == 12
    assert size([36, 1, 12, 12, 12], "min") == 1


def test_qwen38_mtp_packed_decode_stays_one_group_per_layer():
    # 36 GDN, 1 PLE, 12 main KV, 2 packed compressed owners, 13 rings, 1 draft main KV.
    assert size([36, 1, 12, 2, 13, 1]) == 1


def test_qwen35_and_uniform_cases_match_upstream():
    assert size([45, 15]) == 15 == size([45, 15], "min")
    assert size([12, 13], "min") == 13
    assert size([7]) == 7
    assert size([1, 1]) == 1


def test_rule_validation():
    with pytest.raises(ValueError):
        size([], "gcd")
    with pytest.raises(ValueError):
        size([4], "max")


buckets = _mod.kv_cache_layer_buckets


class _Spec:
    """A stand-in KV spec: equal fields compare and hash equal within a class; merge accepts
    any list whose fields agree, subclasses included (as vLLM's merge did for the F2 draft)."""

    def __init__(self, kind):
        self.kind = kind

    def __eq__(self, other):
        return type(other) is type(self) and other.kind == self.kind

    def __hash__(self):
        return hash(self.kind)

    @classmethod
    def merge(cls, specs):
        if len({s.kind for s in specs}) != 1:
            raise AssertionError("different kinds")
        return specs[0]


class _DraftSpec(_Spec):
    q38_kv_bucket = "q38-f2-producer-draft"


def _producer_plan(draft_cls):
    plan = {f"model.layers.{i}.linear_attn": _Spec("gdn") for i in range(36)}
    for i in range(12):
        plan[f"model.layers.{i}.self_attn"] = _Spec("main")
        plan[f"model.layers.{i}.self_attn.raw_ring"] = _Spec("ring")
    plan["model.layers.0.ple"] = _Spec("ple")
    plan["mtp.layers.48.self_attn"] = draft_cls("main")
    plan["mtp.layers.48.self_attn.raw_ring"] = draft_cls("ring")
    return plan


def test_the_producer_draft_never_joins_the_target_buckets():
    """q38 F2 smoke attempt 3 (j-58a40b86): the draft's specs (a subclass with equal fields) merged
    into the target's buckets, 13 against 36, and the GCD rule fell from 12 to 1. Marked with
    q38_kv_bucket they keep buckets of their own and the target keeps groups of 12."""
    got = buckets(_producer_plan(_DraftSpec))
    sizes = sorted(len(b) for b in got)
    assert sizes == [1, 1, 1, 12, 12, 36]
    assert all(len(b) == 1 for b in got if any(n.startswith("mtp.") for n in b))
    assert size([len(b) for b in got]) == 12


def test_an_unmarked_subclass_still_merges_as_vllm_does():
    """Without the marker the subclass merges (the failure mode): 13-layer buckets, groups of 1."""
    got = buckets(_producer_plan(type("Plain", (_Spec,), {})))
    assert sorted(len(b) for b in got) == [1, 13, 13, 36]
    assert size([len(b) for b in got]) == 1
