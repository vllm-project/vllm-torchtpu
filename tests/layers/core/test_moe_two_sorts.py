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
"""The two-sort routing metadata of the non-EP grouped-matmul MoE: the same
metadata as the three sorts with one sort fewer, only on the non-EP path above
the single-sort cap, and selected when the op is built."""

import functools
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.adapter import fused_moe
from vllm_torchtpu.layers.core import fused_moe_gmm

pytestmark = pytest.mark.cpu_test

# A whole-expert (non-EP) layer of the Qwen3.5 prefill legs: 512 experts,
# top-10. 256 tokens route 2,560 rows, above the 2,048-row single-sort cap.
EXPERTS, TOPK, HIDDEN = 512, 10, 16


def _inputs(num_tokens, seed, padded_tokens=0):
    rng = np.random.default_rng(seed)
    ids = np.stack(
        [rng.choice(EXPERTS, TOPK, replace=False) for _ in range(num_tokens)]
    ).astype(np.int32)
    weights = rng.random((num_tokens, TOPK)).astype(np.float32)
    if padded_tokens:
        # Padded tokens keep their ids with zero weights; with
        # skip_padded_tokens they become invalid rows on the sentinel key.
        weights[-padded_tokens:] = 0.0
    hidden = rng.standard_normal((num_tokens, HIDDEN)).astype(np.float32)
    return jnp.asarray(hidden), jnp.asarray(ids), jnp.asarray(weights)


def _prepare(permute, *args, use_ep=False, skip_padded_tokens=False):
    return fused_moe_gmm.prepare_routed_gmm_inputs(
        *args,
        local_num_experts=EXPERTS,
        topk=TOPK,
        use_ep=use_ep,
        use_sparse_core=False,
        skip_padded_tokens=skip_padded_tokens,
        nonep_permute=permute,
    )


def _sorts(permute, *args, **kwargs):
    """The number of sort ops prepare_routed_gmm_inputs traces."""
    jaxpr = jax.make_jaxpr(functools.partial(_prepare, permute, **kwargs))(*args)

    def subjaxprs(eqn):
        for value in eqn.params.values():
            for v in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(v, "jaxpr", v)
                if hasattr(inner, "eqns"):
                    yield inner

    def count(jpr):
        n = 0
        for eqn in jpr.eqns:
            n += eqn.primitive.name == "sort"
            for sub in subjaxprs(eqn):
                n += count(sub)
        return n

    return count(jaxpr.jaxpr)


@pytest.mark.parametrize(
    "num_tokens,padded,skip",
    [(256, 0, False), (256, 37, True), (2048, 0, False)],
    ids=["2560_rows", "2560_rows_padded", "20480_rows"],
)
def test_two_sorts_give_the_three_sorts_metadata(num_tokens, padded, skip):
    args = _inputs(num_tokens, seed=num_tokens + padded, padded_tokens=padded)
    three = _prepare("sorts", *args, skip_padded_tokens=skip)
    two = _prepare("two_sorts", *args, skip_padded_tokens=skip)
    assert len(three) == len(two) == 7
    for a, b in zip(three, two, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_two_sorts_drop_the_packed_sort():
    args = _inputs(256, seed=1)
    assert _sorts("sorts", *args) == 3
    assert _sorts("two_sorts", *args) == 2


@pytest.mark.parametrize(
    "num_tokens,use_ep",
    [(256, True), (128, False)],
    ids=["expert_parallel", "within_the_single_sort_cap"],
)
def test_other_paths_are_unchanged(num_tokens, use_ep):
    """Expert parallelism, and the rows main's counting sort or single sort
    already serve (at most 2,048), trace the same program either way."""
    args = _inputs(num_tokens, seed=2)
    assert _sorts("two_sorts", *args, use_ep=use_ep) == _sorts(
        "sorts", *args, use_ep=use_ep
    )
    three = _prepare("sorts", *args, use_ep=use_ep)
    two = _prepare("two_sorts", *args, use_ep=use_ep)
    for a, b in zip(three, two, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_unknown_permute_is_refused():
    with pytest.raises(ValueError, match="non-EP permute"):
        _prepare("counting", *_inputs(4, seed=3))


@pytest.fixture
def mains_default(monkeypatch):
    monkeypatch.setattr(fused_moe_gmm, "NONEP_PERMUTE", "sorts")


def test_the_default_keeps_three_sorts():
    assert fused_moe_gmm.NONEP_PERMUTE == "sorts"


def test_set_nonep_permute(mains_default):
    fused_moe_gmm.set_nonep_permute("two_sorts")
    assert fused_moe_gmm.NONEP_PERMUTE == "two_sorts"
    fused_moe_gmm.set_nonep_permute("sorts")
    assert fused_moe_gmm.NONEP_PERMUTE == "sorts"
    with pytest.raises(ValueError, match="non-EP permute"):
        fused_moe_gmm.set_nonep_permute("counting")
    assert fused_moe_gmm.NONEP_PERMUTE == "sorts"


@pytest.fixture
def built_ops(monkeypatch, mains_default):
    """The grouped-matmul op builder with pallas.jax_op recorded, not built."""
    monkeypatch.setattr(fused_moe, "_fused_moe_kernel_cache", {})
    jax_op = Mock(name="jax_op", side_effect=lambda name, fn: Mock(name=name, fn=fn))
    monkeypatch.setattr(fused_moe.pallas, "jax_op", jax_op)
    return jax_op


def _get_op():
    return fused_moe._get_fused_moe_custom_op(
        topk=TOPK, activation="silu", use_ep=False, skip_padded_tokens=False
    )


@pytest.mark.parametrize("permute", ["sorts", "two_sorts"])
def test_the_op_is_built_with_the_selected_permute(built_ops, permute):
    fused_moe_gmm.set_nonep_permute(permute)
    op = _get_op()
    assert op.fn.func is fused_moe_gmm.fused_moe_func
    assert op.fn.keywords["nonep_permute"] == permute


def test_a_permute_change_builds_a_new_op(built_ops):
    first = _get_op()
    assert _get_op() is first
    fused_moe_gmm.set_nonep_permute("two_sorts")
    second = _get_op()
    assert second is not first
    assert second.fn.keywords["nonep_permute"] == "two_sorts"
