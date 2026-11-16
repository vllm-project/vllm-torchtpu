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
"""Helper-subgraph precompilation: how the shape sweeps are ordered across a
host's local ranks, that the sampling pass is not repeated, and that warmed
graphs carry the guards serving checks."""

from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _rotate_precompile_items,
)

ITEMS = list(range(10))


def _runner(tp_size=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size)
    )


def _orders(monkeypatch, *, local_world, tp_size=1, collective_free=False):
    """The order each local rank of one host walks ITEMS in."""
    monkeypatch.setenv("TPU_PARALLEL_PRECOMPILE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local_world))
    orders = []
    for rank in range(local_world):
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        orders.append(
            _rotate_precompile_items(
                _runner(tp_size), ITEMS, collective_free=collective_free
            )
        )
    return orders


def test_rotation_off_without_parallel_precompile(monkeypatch):
    monkeypatch.delenv("TPU_PARALLEL_PRECOMPILE", raising=False)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "5")
    assert _rotate_precompile_items(_runner(), ITEMS) == ITEMS


def test_rotation_off_when_host_size_unknown(monkeypatch):
    monkeypatch.setenv("TPU_PARALLEL_PRECOMPILE", "1")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert _rotate_precompile_items(_runner(), ITEMS) == ITEMS


@pytest.mark.parametrize("collective_free", [False, True])
def test_single_rank_engines_each_start_on_their_share(monkeypatch, collective_free):
    """DP8 x TP1: eight groups of one rank start at len * rank // 8."""
    orders = _orders(monkeypatch, local_world=8, collective_free=collective_free)
    assert [order[0] for order in orders] == [0, 1, 2, 3, 5, 6, 7, 8]
    for order in orders:
        assert sorted(order) == ITEMS


def test_tp_group_ranks_keep_one_order(monkeypatch):
    """DP2 x TP4: the ranks of a TP group walk the same order, so a graph with
    a collective is reached by all of them together; the groups differ."""
    orders = _orders(monkeypatch, local_world=8, tp_size=4)
    assert orders[:4] == [ITEMS] * 4
    assert orders[4:] == [ITEMS[5:] + ITEMS[:5]] * 4


def test_one_tp_group_keeps_the_sweep_order(monkeypatch):
    """TP8 on 8 local ranks: one group, so a sweep that may hold a collective
    is not reordered."""
    assert _orders(monkeypatch, local_world=8, tp_size=8) == [ITEMS] * 8


def test_collective_free_sweeps_rotate_inside_a_tp_group(monkeypatch):
    """TP8: a collective-free sweep still starts every rank on its own share."""
    orders = _orders(monkeypatch, local_world=8, tp_size=8, collective_free=True)
    assert [order[0] for order in orders] == [0, 1, 2, 3, 5, 6, 7, 8]
    for order in orders:
        assert sorted(order) == ITEMS


def test_substitute_placeholder_token_has_no_collective():
    """The precompile sweep of _substitute_placeholder_token rotates per local
    rank (collective_free=True), so ranks reach a shape at different times.
    That is only safe while the graph holds no collective: a collective added
    later must fail here rather than hang at startup."""
    fn = tpu_runner._substitute_placeholder_token
    fn = getattr(fn, "_torchdynamo_orig_callable", fn)
    n = 16
    graph = make_fx(fn)(
        torch.zeros(n, dtype=torch.int32),
        torch.arange(n, dtype=torch.int32),
        torch.full((n,), -1, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int64),
    )
    for node in graph.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        namespace = getattr(target, "namespace", None)
        assert namespace in ("aten", "prims"), f"unexpected op {target}"
        assert "c10d" not in str(target) and "all_" not in str(target), target


def _sampling_runner():
    calls = []
    runner = object.__new__(TPUModelRunner)
    runner.is_pooling_model = False
    runner._pp_is_last = True
    runner.num_tokens_paddings = [16, 32]
    for name in (
        "_precompile_compute_selected_logits",
        "_precompile_compute_logits_from_hidden_states",
        "_precompile_compute_top_tokens",
        "_precompile_structured_decoding",
        "_precompile_min_tokens_mask",
        "_precompile_sample_from_logits",
        "_precompile_gather_logprobs",
    ):
        setattr(runner, name, lambda name=name: calls.append(name))
    runner.get_pcp_planner = lambda: SimpleNamespace(
        uses_selected_logits_hidden_states=False
    )
    return runner, calls


def test_sampling_precompile_runs_once_per_token_buckets():
    """profile_run warms the sampling subgraphs and capture_model asks again:
    the second pass over the same token buckets is skipped."""
    runner, calls = _sampling_runner()
    TPUModelRunner._precompile_sampling_subgraphs(runner)
    first = list(calls)
    assert "_precompile_sample_from_logits" in first
    TPUModelRunner._precompile_sampling_subgraphs(runner)
    assert calls == first


def test_sampling_precompile_reruns_for_new_token_buckets():
    runner, calls = _sampling_runner()
    TPUModelRunner._precompile_sampling_subgraphs(runner)
    first = len(calls)
    runner.num_tokens_paddings = [16, 32, 64]
    TPUModelRunner._precompile_sampling_subgraphs(runner)
    assert len(calls) == 2 * first


def test_greedy_warmup_matches_greedy_sample(monkeypatch):
    """The all_greedy graph is warmed with the inputs _greedy_sample passes at
    serving time ((1, 1) placeholders of the same dtypes) and under no_grad,
    so dynamo's guards hit on the first greedy batch of any size."""
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **k: None)
    seen = []

    def sample_from_logits(logits, temperatures, u, top_k, top_p, **kwargs):
        seen.append(
            (
                kwargs.get("all_greedy", False),
                [(tuple(t.shape), t.dtype) for t in (temperatures, u, top_k, top_p)],
                torch.is_grad_enabled(),
            )
        )
        return torch.zeros((logits.shape[0], 1), dtype=torch.int64)

    runner = object.__new__(TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.num_reqs_paddings = [8, 16]
    runner.vocab_size = 32
    runner._hidden_states_dtype = torch.bfloat16
    runner.speculative_config = None
    runner._topk_candidate_blocks = lambda: 0
    runner._update_num_xla_graphs = lambda name: None
    runner.sample_from_logits_func = sample_from_logits

    TPUModelRunner._precompile_sample_from_logits(runner)
    warmed = [args for greedy, args, _ in seen if greedy]
    assert warmed and all(not grad for _, _, grad in seen)

    seen.clear()
    TPUModelRunner._greedy_sample(runner, runner._dummy_logits(16))
    ((_, serving, _),) = seen
    assert all(args == serving for args in warmed)
