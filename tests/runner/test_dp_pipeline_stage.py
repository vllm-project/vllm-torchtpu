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
"""A pipelined prefill's stages on CPU: which steps are stage steps, the
hand-off from one engine's runner to the next, and the K/V the owner gets."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dp_pipeline import transport
from vllm_torchtpu.distributed.dp_pipeline.kv_plan import KvGroup
from vllm_torchtpu.distributed.dp_pipeline.plan import (
    PLAN_KEY,
    PipelinePlan,
    make_pipeline_id,
)
from vllm_torchtpu.runner.dp_pipeline_stage import DpPipelineStage, HandOff, KvAccess

_servers = itertools.count(51000)
CHUNKS = [(0, 8), (8, 8), (16, 5)]  # (first position, tokens) of a 21-token prompt


ROWS = 8  # the full chunk's bucket, which every pipelined chunk pads to


def _stage(dp_rank, tp_rank=0, tp_size=1, per_rank=False):
    rows = ROWS // tp_size if per_rank else ROWS
    return DpPipelineStage(
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        tp_size=tp_size,
        max_rows=ROWS,
        hand_off=HandOff(per_rank, rows, (4,), torch.float32),
        device=torch.device("cpu"),
        materialize=lambda t, wait=True: None,
    )


def _sent(call, out):
    """``call`` after a forward that left ``out`` to hand on."""
    call.output.append(out)
    return call


def _batch(requests):
    """An input batch holding ``requests``: {req_id: (computed, prompt_len)}."""
    ids = list(requests)
    return SimpleNamespace(
        req_ids=ids,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        num_computed_tokens_cpu=np.array([c for c, _ in requests.values()]),
        num_prompt_tokens=np.array([p for _, p in requests.values()]),
    )


def _states(plans):
    return {
        r: SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args=None if p is None else {PLAN_KEY: p.to_json()}
            )
        )
        for r, p in plans.items()
    }


def _unlink(p):
    transport.unlink_rings(p.server_id, max(p.ranks) + 1, tp_size=2)


@pytest.fixture
def plan():
    p = PipelinePlan(make_pipeline_id(next(_servers), 1), (2, 0), (0, 3, 6), 21)
    yield p
    _unlink(p)


@pytest.fixture
def kv_plan_(monkeypatch):
    """A K/V plan over a 6-layer model."""
    monkeypatch.setattr(envs, "TPU_DP_PIPELINE_STAGES", 2)
    monkeypatch.setattr(envs, "TPU_DP_PIPELINE_SPLITS", "2:3")
    p = PipelinePlan(
        make_pipeline_id(next(_servers), 1), (2, 0), (0, 3, 6), 21, kv=True
    )
    yield p
    _unlink(p)


def test_only_prefill_chunks_of_a_plan_are_stage_steps(plan):
    helper, owner = _stage(2), _stage(0)
    states = _states({"a": plan, "b": None})
    step = helper.step(["a"], _batch({"a": (8, 21)}), states, {"a": 8})
    assert (step.stage, step.first_position, step.is_first, step.is_last) == (
        0,
        8,
        True,
        False,
    )
    step = owner.step(["a"], _batch({"a": (16, 21)}), states, {"a": 5})
    assert (step.stage, step.is_last, step.is_final_chunk) == (1, True, True)
    # The owner's decode steps and ordinary requests run the whole model.
    assert owner.step(["a"], _batch({"a": (21, 21)}), states, {"a": 1}) is None
    assert owner.step(["b"], _batch({"b": (0, 40)}), states, {"b": 8}) is None
    with pytest.raises(RuntimeError, match="alone in its step"):
        helper.step(
            ["a", "b"], _batch({"a": (0, 21), "b": (0, 40)}), states, {"a": 8, "b": 8}
        )
    # A prompt the owner has run is recomputed alone after a preemption.
    assert owner.step(["a"], _batch({"a": (0, 21)}), states, {"a": 8}) is None
    # Every pipelined chunk pads to the segments' bucket.
    step = helper.step(["a"], _batch({"a": (16, 21)}), states, {"a": 5})
    with pytest.raises(RuntimeError, match="pads to 5 rows"):
        helper.call(step, 5)


def _run_helper(helper, plan, states, outputs=None, chunks=CHUNKS):
    prompt = chunks[-1][0] + chunks[-1][1]  # a helper holds the pipelined tokens
    for first, tokens in chunks:
        step = helper.step(["a"], _batch({"a": (first, prompt)}), states, {"a": tokens})
        call = helper.call(step, ROWS)
        assert (call.start, call.end, call.hidden_states) == (0, 3, None)
        out = torch.full((ROWS, 4), float(first))
        helper.send(step, _sent(call, out))
        if outputs is not None:
            outputs.append(out)


def test_hand_off_between_engines(plan):
    helper, owner = _stage(2), _stage(0)
    states = _states({"a": plan})
    outs = []
    _run_helper(helper, plan, states, outs)
    for (first, tokens), out in zip(CHUNKS, outs):
        step = owner.step(["a"], _batch({"a": (first, 21)}), states, {"a": tokens})
        call = owner.call(step, ROWS)
        assert (call.start, call.end) == (3, 6)
        assert torch.equal(call.hidden_states, out)
        owner.send(step, call)  # the last stage keeps its output
    # The input of every chunk after the first was copied in ahead of it.
    assert owner._prefetched == {}
    helper.close()
    owner.close()


def test_a_failed_hand_off_surfaces_on_the_next_step(plan, monkeypatch):
    helper = _stage(2)
    monkeypatch.setattr(helper, "timeout_s", 0.01)
    states = _states({"a": plan})
    for first in (0, 8, 16):  # two slots fill; nobody takes them
        step = helper.step(["a"], _batch({"a": (first, 21)}), states, {"a": 5})
        helper.send(step, _sent(helper.call(step, ROWS), torch.zeros(8, 4)))
    helper._sent[-1].exception(timeout=5.0)
    with pytest.raises(TimeoutError, match="free slot"):
        helper.step(["a"], _batch({"a": (0, 21)}), states, {"a": 8})
    helper.close()


# Layers 1 and 4 are global, the others slide over 8 tokens; pages of 4 tokens.
GROUPS = (
    KvGroup(0, ((1, "l1"), (4, "l4")), None),
    KvGroup(1, ((0, "l0"), (2, "l2"), (3, "l3"), (5, "l5")), 8),
)


class _Engine:
    """One engine's KV caches and block table for request "a"."""

    def __init__(self, block_of, fill):
        self.caches = {
            name: (
                torch.arange(16 * 4 * 2, dtype=torch.float32).reshape(16, 4, 2)
                + 1000 * index
                if fill
                else torch.zeros(16, 4, 2)
            )
            for g in GROUPS
            for index, name in g.layers
        }
        self.block_of = block_of
        self.scatters = 0

    def access(self):
        def block_ids(group_id, req_id, blocks):
            assert req_id == "a"
            return [self.block_of(group_id, b) for b in blocks]

        def replace(old, new):
            for name, cache in self.caches.items():
                if cache is old:
                    self.caches[name] = new

        def gather(caches, ids, pages):
            out = []
            for cache in caches:
                rows = torch.zeros(pages, *cache.shape[1:])
                rows[: len(ids)] = cache[ids]
                out.append(rows)
            return out

        def scatter(sources, caches, ids, pages):
            self.scatters += 1
            out = []
            for src, cache in zip(sources, caches):
                assert src.shape[0] == pages
                new = cache.clone()
                new[ids] = src[: len(ids)]
                out.append(new)
            return out

        return KvAccess(
            groups=GROUPS,
            block_size=4,
            page_shape=(4, 2),
            dtype=torch.float32,
            cache=lambda name: self.caches[name],
            replace=replace,
            block_ids=block_ids,
            gather=gather,
            scatter=scatter,
        )


def test_the_owner_gets_the_helpers_kv_before_it_decodes(kv_plan_):
    states = _states({"a": kv_plan_})
    helper_kv = _Engine(lambda g, b: 1 + b + 7 * g, fill=True)
    owner_kv = _Engine(lambda g, b: 15 - b - 8 * g, fill=False)
    helper, owner = _stage(2), _stage(0)
    helper.bind_kv(helper_kv.access(), num_layers=6, max_chunk_tokens=8)
    owner.bind_kv(owner_kv.access(), num_layers=6, max_chunk_tokens=8)
    _run_helper(helper, kv_plan_, states)
    for first, tokens in CHUNKS:
        batch = _batch({"a": (first, 21)})
        owner.before_forward(["a"], batch, states)
        step = owner.step(["a"], batch, states, {"a": tokens})
        owner.send(step, owner.call(step, ROWS))
    # The last pipelined step waited for every hand-off, so its sampled token
    # does: the first decode step finds nothing left to wait for.
    assert owner._receipts == {}
    owner.before_forward(["a"], _batch({"a": (21, 21)}), states)
    prompt_blocks = range(6)  # 21 tokens in pages of 4
    window_blocks = range((21 - 8 + 1) // 4, 6)
    for name, group, blocks in (
        ("l1", 0, prompt_blocks),
        ("l0", 1, window_blocks),
        ("l2", 1, window_blocks),
    ):
        for b in blocks:
            got = owner_kv.caches[name][owner_kv.block_of(group, b)]
            want = helper_kv.caches[name][helper_kv.block_of(group, b)]
            assert torch.equal(got, want), (name, b)
    # Pages outside the hand-off and the owner's own layers stay as they were.
    assert not owner_kv.caches["l0"][owner_kv.block_of(1, 0)].any()
    for name in ("l3", "l4", "l5"):
        assert not owner_kv.caches[name].any()
    helper.close()
    owner.close()


@pytest.mark.parametrize("finished", [True, False])
def test_a_request_that_left_the_batch_drops_its_kv(kv_plan_, finished):
    states = _states({"a": kv_plan_})
    helper, owner = _stage(2), _stage(0)
    helper_kv = _Engine(lambda g, b: 1 + b, fill=True)
    owner_kv = _Engine(lambda g, b: 15 - b, fill=False)
    helper.bind_kv(helper_kv.access(), num_layers=6, max_chunk_tokens=8)
    owner.bind_kv(owner_kv.access(), num_layers=6, max_chunk_tokens=8)
    # bind_kv compiled one empty scatter per layer count: l1 (pages 3), l0+l2.
    assert owner_kv.scatters == 2
    owner_kv.scatters = 0
    _run_helper(helper, kv_plan_, states)
    for first, tokens in CHUNKS:
        step = owner.step(["a"], _batch({"a": (first, 21)}), states, {"a": tokens})
        owner.call(step, ROWS)
    # Finished, or preempted after its prompt (its blocks are freed).
    owner.before_forward([], _batch({}), {} if finished else states)
    assert owner._receipts == {}
    owner._receiver.shutdown(wait=True)
    assert owner_kv.scatters == 0
    # Every held hand-off was released.
    assert owner._held._value == 4
    helper.close()
    owner.close()


def test_the_owner_computes_the_rest_of_the_prompt_alone(monkeypatch):
    """A pipeline over the full chunks only: the owner runs the 5-token rest as
    an ordinary step, once every helper page is in its cache."""
    monkeypatch.setattr(envs, "TPU_DP_PIPELINE_STAGES", 2)
    monkeypatch.setattr(envs, "TPU_DP_PIPELINE_SPLITS", "2:3")
    plan = PipelinePlan(
        make_pipeline_id(next(_servers), 1), (2, 0), (0, 3, 6), 16, kv=True
    )
    states = _states({"a": plan})
    helper_kv = _Engine(lambda g, b: 1 + b + 7 * g, fill=True)
    owner_kv = _Engine(lambda g, b: 15 - b - 8 * g, fill=False)
    helper, owner = _stage(2), _stage(0)
    try:
        helper.bind_kv(helper_kv.access(), num_layers=6, max_chunk_tokens=8)
        owner.bind_kv(owner_kv.access(), num_layers=6, max_chunk_tokens=8)
        _run_helper(helper, plan, states, chunks=CHUNKS[:2])
        for first, tokens in CHUNKS[:2]:
            batch = _batch({"a": (first, 21)})
            owner.before_forward(["a"], batch, states)
            owner.call(owner.step(["a"], batch, states, {"a": tokens}), ROWS)
        rest = _batch({"a": (16, 21)})
        owner.before_forward(["a"], rest, states)
        assert owner.step(["a"], rest, states, {"a": 5}) is None
        assert owner._receipts == {}
        for name, group, blocks in (
            ("l1", 0, range(4)),  # positions [0, 16)
            ("l0", 1, range((16 - 8 + 1) // 4, 4)),  # the window before 16
        ):
            for b in blocks:
                got = owner_kv.caches[name][owner_kv.block_of(group, b)]
                want = helper_kv.caches[name][helper_kv.block_of(group, b)]
                assert torch.equal(got, want), (name, b)
    finally:
        helper.close()
        owner.close()
        _unlink(plan)


def test_each_rank_hands_its_own_rows_to_the_same_rank(plan):
    """Sequence-parallel hand-offs: TP rank r of a stage sends its rows to TP
    rank r of the next one."""
    states = _states({"a": plan})
    helpers = [_stage(2, r, 2, per_rank=True) for r in range(2)]
    owners = [_stage(0, r, 2, per_rank=True) for r in range(2)]
    try:
        for first, tokens in CHUNKS:
            batch = _batch({"a": (first, 21)})
            for r, helper in enumerate(helpers):
                step = helper.step(["a"], batch, states, {"a": tokens})
                rows = torch.full((ROWS // 2, 4), float(10 * first + r))
                helper.send(step, _sent(helper.call(step, ROWS), rows))
            for r, owner in enumerate(owners):
                step = owner.step(["a"], batch, states, {"a": tokens})
                got = owner.call(step, ROWS).hidden_states
                assert torch.equal(
                    got, torch.full((ROWS // 2, 4), float(10 * first + r))
                )
    finally:
        for stage in helpers + owners:
            stage.close()
