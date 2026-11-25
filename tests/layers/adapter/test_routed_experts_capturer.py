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
"""TPU routed-experts capture: every forward returns a table of its own as an
output of the compiled forward, and a step's rows reach the host from the
copies of its forwards' tables."""

from functools import partial

import numpy as np
import pytest
import torch
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    bind_routed_experts_capturer,
)
from vllm.platforms import current_platform

from tests.distributed.tpu_test_utils import run_in_isolated_process
from tests.layers.adapter.routed_experts_test_utils import (
    NUM_LAYERS,
    PADDING,
    TOKEN_PADDINGS,
    TOPK,
    HostEvent,
    expert_ids,
    make_capturer,
    make_moe_model,
    make_vllm_config,
    model_forward,
    run_hooks,
    use_host_events,
)

TPU = "tpu"


@pytest.fixture
def on_host(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    use_host_events(monkeypatch)


def _bound_model(capturer, layer_ids=None):
    """MoE layers of the given ids whose hooks call `capturer`."""
    model = make_moe_model(layer_ids)
    bind_routed_experts_capturer(model, capturer)
    return model


class _Backbone(torch.nn.Module):
    """Runs the hooks as the backbone's MoE layers do, and returns a result of
    its own, as the backbone returns hidden states."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        run_hooks(self.model, ids)
        return ids.sum()


def _compile_like_vllm(model, graphs):
    """Compiles as vLLM compiles the backbone: one trace for every bucket,
    with every guard dropped. Appends each traced graph's example inputs."""

    def backend(gm, example_inputs):
        graphs.append(example_inputs)
        return gm.forward

    torch._dynamo.reset()
    return torch.compile(
        _Backbone(model),
        backend=backend,
        fullgraph=True,
        dynamic=True,
        options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    )


def _run(forward, ids, context_num_tokens=None):
    if context_num_tokens is None:
        context_num_tokens = ids.shape[0]
    with set_forward_context(None, make_vllm_config(), num_tokens=context_num_tokens):
        forward(ids)


@pytest.mark.parametrize("is_warmup", [False, True])
def test_one_trace_returns_a_table_for_each_forward(on_host, is_warmup):
    """The table is created inside the compiled forward and leaves it as an
    output, so one trace serves every bucket and no table is an input of the
    forward. A warmup forward's context holds 0 tokens."""
    capturer = make_capturer()
    graphs = []
    forward = _compile_like_vllm(_bound_model(capturer), graphs)
    # A table left on the capturer must not enter the trace.
    stale = torch.full((16, NUM_LAYERS, TOPK), 7, dtype=capturer.output_dtype)
    capturer.device_buffer = stale

    tables = []
    for bucket in (16, 32):
        ids = expert_ids(bucket)
        _run(forward, ids, context_num_tokens=0 if is_warmup else bucket)
        table = capturer.check_forward(bucket)
        assert torch.equal(table.to(torch.int32), ids)
        tables.append(table)

    assert len(graphs) == 1
    tensor_inputs = [x for x in graphs[0] if isinstance(x, torch.Tensor)]
    assert len(tensor_inputs) == 1, "only the expert ids enter the forward"
    assert (stale == 7).all()
    assert tables[0].data_ptr() != tables[1].data_ptr()


def test_layers_without_experts_return_zeros(on_host):
    """Nothing writes the column of a layer without experts, and vLLM returns
    0 for it."""
    capturer = make_capturer(moe_layer_ids=(0, 2))
    forward = _compile_like_vllm(_bound_model(capturer, (0, 2)), [])
    ids = expert_ids(16)

    _run(forward, ids)

    table = capturer.check_forward(16).to(torch.int32)
    assert not table[:, 1].any()
    assert torch.equal(table[:, [0, 2]], ids[:, [0, 2]])


def test_a_model_without_moe_layers_is_rejected(on_host):
    with pytest.raises(ValueError, match="needs a model with MoE layers"):
        make_capturer(moe_layer_ids=[])


def test_capture_rejects_a_layer_past_the_table(on_host):
    capturer = make_capturer()
    with (
        set_forward_context(None, make_vllm_config(), num_tokens=16),
        pytest.raises(IndexError, match="exceeds capture buffer"),
    ):
        capturer.capture(NUM_LAYERS, expert_ids(16)[:, 0])


def test_a_table_of_another_bucket_fails_the_check(on_host):
    capturer = make_capturer()
    model_forward(_bound_model(capturer), expert_ids(16))

    with pytest.raises(AssertionError, match=r"table of \(16, 3, 4\)"):
        capturer.check_forward(32)


def _take(capturer, model, bucket, num_tokens, seed):
    """Runs one forward of `bucket` rows, takes it, and returns its real
    rows."""
    ids = expert_ids(bucket, num_experts=PADDING, seed=seed)
    ids[num_tokens:] = PADDING
    model_forward(model, ids)
    capturer.take_forward(bucket, num_tokens)
    return ids[:num_tokens]


def test_a_one_forward_step_reads_its_copy(on_host):
    """The step's rows are the first rows of its forward's copy: nothing is
    copied again on the host."""
    capturer = make_capturer()
    model = _bound_model(capturer)
    rows = _take(capturer, model, 16, 10, seed=0)

    snapshot_rows = capturer.snapshot_routing_data(10)
    snapshot = capturer.take_snapshot()
    snapshot.wait()

    assert snapshot_rows.shape == (10, NUM_LAYERS, TOPK)
    assert torch.equal(snapshot_rows.to(torch.int32), rows)
    # On the host the forward's copy is its table itself, so the rows must be
    # a view of that table, not a second copy.
    assert snapshot_rows.data_ptr() == capturer.device_buffer.data_ptr()
    (event,) = HostEvent.events
    assert event.synchronized


def test_a_step_joins_the_real_rows_of_its_forwards(on_host):
    """Each forward writes its table from row 0; the step's rows are each
    forward's real rows, in the order the forwards ran."""
    capturer = make_capturer()
    model = _bound_model(capturer)
    expected = [
        _take(capturer, model, bucket, num_tokens, seed)
        for seed, (bucket, num_tokens) in enumerate([(32, 20), (16, 9), (16, 5)])
    ]

    rows = capturer.snapshot_routing_data(34)
    snapshot = capturer.take_snapshot()
    # The next step's forward runs before this step's copies are waited on.
    _take(capturer, model, 16, 16, seed=9)
    snapshot.wait()

    assert rows.shape[0] == 34
    assert torch.equal(rows.to(torch.int32), torch.cat(expected))


def test_a_snapshot_of_other_rows_than_taken_fails(on_host):
    capturer = make_capturer()
    _take(capturer, _bound_model(capturer), 16, 10, seed=0)

    with pytest.raises(AssertionError, match="took 10 rows for a step of 12"):
        capturer.snapshot_routing_data(12)


def test_a_forward_taken_with_more_tokens_than_its_bucket_fails(on_host):
    capturer = make_capturer()
    model_forward(_bound_model(capturer), expert_ids(16))

    with pytest.raises(AssertionError, match="16-token forward was taken with 20"):
        capturer.take_forward(16, 20)


def test_a_rank_that_does_not_copy_has_no_rows_to_snapshot(on_host):
    """A rank other than the first TP rank runs the same forward but starts
    no copy of its table."""
    capturer = make_capturer(copies_to_host=False)
    _take(capturer, _bound_model(capturer), 16, 10, seed=0)

    with pytest.raises(AssertionError, match="took 0 rows"):
        capturer.snapshot_routing_data(10)


# The tests below run on the chip, each in a spawned child, and
# spawns_tpu_workers runs them before tests that hold the chip in-process.


def _tpu_backbone(num_experts=512):
    """A capturer, and a forward compiled for the chip whose hooks call it."""
    from torch_tpu._internal import execution_mode

    # The engine runs eager ops in this mode.
    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE
    capturer = make_capturer(num_experts)
    model = _bound_model(capturer)
    torch._dynamo.reset()
    forward = torch.compile(
        _Backbone(model), backend="tpu", fullgraph=True, dynamic=False
    )
    return capturer, forward


def _tpu_step(capturer, forward, forwards, num_experts, seed):
    """Runs a step of (bucket, real tokens) forwards on the chip and takes its
    snapshot. Returns the snapshot, its rows and the rows it must hold."""
    expected = []
    for i, (bucket, num_tokens) in enumerate(forwards):
        ids = expert_ids(bucket, num_experts, seed=seed + i)
        _run(forward, ids.to(TPU))
        capturer.take_forward(bucket, num_tokens)
        expected.append(ids[:num_tokens])
    rows = capturer.snapshot_routing_data(sum(n for _, n in forwards))
    return capturer.take_snapshot(), rows, torch.cat(expected)


def _forward_tables_reach_the_host_on_tpu(num_experts):
    capturer, forward = _tpu_backbone(num_experts)
    snapshot, rows, expected = _tpu_step(
        capturer, forward, [(32, 20), (16, 9), (16, 5)], num_experts, seed=0
    )
    # The next step's forward runs before this step's copies are waited on.
    next_snapshot, next_rows, next_expected = _tpu_step(
        capturer, forward, [(16, 16)], num_experts, seed=9
    )
    snapshot.wait()
    next_snapshot.wait()

    assert rows.dtype == capturer.output_dtype
    assert rows.shape == (34, NUM_LAYERS, TOPK)
    assert np.array_equal(rows.numpy().astype(np.int32), expected.numpy())
    assert np.array_equal(next_rows.numpy().astype(np.int32), next_expected.numpy())


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
@pytest.mark.parametrize("num_experts", [256, 512])
def test_forward_tables_reach_the_host_on_tpu(num_experts):
    """Each forward's table leaves the compiled forward in the connector's
    dtype (uint8 up to 256 experts, else uint16), and a step's rows are the
    real rows of its forwards, while the next step already runs."""
    run_in_isolated_process(partial(_forward_tables_reach_the_host_on_tpu, num_experts))


def _serving_compiles_nothing_after_warmup_on_tpu():
    capturer, forward = _tpu_backbone()

    def serve(steps, seed):
        for i, forwards in enumerate(steps):
            snapshot, rows, expected = _tpu_step(
                capturer, forward, forwards, 512, seed=seed + 10 * i
            )
            snapshot.wait()
            assert np.array_equal(rows.numpy().astype(np.int32), expected.numpy())

    # Warmup compiles the forward of each bucket.
    serve([[(bucket, bucket)] for bucket in TOKEN_PADDINGS], seed=0)
    compiles = torch.tpu._get_cache_stats().num_compilations
    # Other row counts and other sequences of buckets.
    serve(
        [[(16, 5)], [(32, 17), (16, 3)], [(64, 33), (16, 16), (16, 1)], [(64, 64)]],
        seed=100,
    )

    assert torch.tpu._get_cache_stats().num_compilations == compiles


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_serving_compiles_nothing_after_warmup_on_tpu():
    """Copying a whole table to the host and joining rows there need no
    device program, so once each bucket's forward is compiled, no row count
    or sequence of forwards compiles anything."""
    run_in_isolated_process(_serving_compiles_nothing_after_warmup_on_tpu)
