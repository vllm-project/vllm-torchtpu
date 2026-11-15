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
"""TPU routed-experts capture: every forward of a step writes the table from
row 0, and its rows reach the step's snapshot at the step's row offset."""

from functools import partial
from types import SimpleNamespace
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch._dynamo.utils
from vllm.config import VllmConfig
from vllm.platforms import current_platform

from tests.distributed.tpu_test_utils import run_in_isolated_process
from vllm_torchtpu import utils
from vllm_torchtpu.layers.adapter import routed_experts_capturer

NUM_LAYERS, TOPK = 3, 4
TOKEN_PADDINGS = [16, 32, 64]
MAX_NUM_BATCHED_TOKENS = 64
STAGING_ROWS = MAX_NUM_BATCHED_TOKENS + TOKEN_PADDINGS[-1]
TPU = "tpu"


def _capturer(num_experts=512, with_staging=True):
    model_config = SimpleNamespace(
        get_total_num_hidden_layers=lambda: NUM_LAYERS,
        get_num_experts=lambda: num_experts,
        get_num_experts_per_tok=lambda: TOPK,
        hf_text_config=SimpleNamespace(model_type="qwen3_moe"),
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS),
        parallel_config=SimpleNamespace(data_parallel_rank=0, tensor_parallel_size=1),
    )
    return routed_experts_capturer.TPURoutedExpertsCapturer(
        cast(VllmConfig, vllm_config), TOKEN_PADDINGS, with_staging=with_staging
    )


def _ids(num_tokens, num_experts=512, seed=0):
    """[num_tokens, layers, k] int32 expert ids, from the host generator only:
    seeding torch would also open the chip in this process."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(num_experts, size=(num_tokens, NUM_LAYERS, TOPK))
    return torch.from_numpy(ids.astype(np.int32))


@pytest.fixture
def cpu_capturer(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    return _capturer()


def test_capture_traces_once_for_every_bucket(cpu_capturer):
    """The hook reads nothing but its input, so one trace with a dynamic
    token dim, as vLLM compiles the backbone, writes every bucket's rows."""
    torch._dynamo.reset()
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    compiled = torch.compile(
        lambda ids: cpu_capturer.capture(1, ids),
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    for num_tokens in (16, 32):
        ids = _ids(num_tokens)[:, 1]
        compiled(ids)
        assert torch.equal(cpu_capturer.device_buffer[:num_tokens, 1], ids)

    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs + 1
    assert not cpu_capturer.device_buffer[:, [0, 2]].any()


@pytest.mark.parametrize("row_offset", [0, 7, STAGING_ROWS - 16])
def test_staging_writes_rows_at_the_offset(row_offset):
    staging = np.zeros((STAGING_ROWS, NUM_LAYERS, TOPK), dtype=np.uint16)
    rows = _ids(16).numpy()

    staged = routed_experts_capturer._stage_rows_jax(
        jnp.asarray(staging), jnp.asarray(rows), jnp.asarray([row_offset], jnp.int32)
    )

    expected = staging.copy()
    expected[row_offset : row_offset + 16] = rows
    assert np.array_equal(np.asarray(staged), expected)


def test_staging_past_the_table_raises(cpu_capturer):
    """dynamic_update_slice would clamp the start and overwrite other rows."""
    with pytest.raises(AssertionError, match="overruns"):
        cpu_capturer.stage_rows(16, STAGING_ROWS - 15)


def test_without_staging_only_the_table_is_written(monkeypatch):
    """Every rank can call stage_rows; only a staging rank has a snapshot."""
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    capturer = _capturer(with_staging=False)

    capturer.stage_rows(16, 0)

    with pytest.raises(AssertionError, match="with_staging"):
        capturer.snapshot_routing_data(16)


def test_snapshot_is_a_bucket_sized_copy(cpu_capturer):
    """Padded to the next bucket, and its own copy: the next step's staging
    writes must not reach a snapshot still being copied to the host."""
    first = cpu_capturer.snapshot_routing_data(20)
    second = cpu_capturer.snapshot_routing_data(20)
    first.fill_(1)

    assert first.shape[0] == 32 and first.dtype == torch.uint16
    assert not second.any()


# The tests below run on the chip, each in a spawned child, and
# spawns_tpu_workers runs them before tests that hold the chip in-process.


def _tpu_capturer(num_experts=512):
    from torch_tpu._internal import execution_mode

    # The engine runs eager ops in this mode.
    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE
    capturer = _capturer(num_experts)
    # Otherwise the deferred zero fills join the first write's program.
    utils.synchronize_tensors()
    return capturer


def _forward(capturer, ids):
    for layer in range(NUM_LAYERS):
        capturer.capture(layer, ids[:, layer].to(TPU))


def _step_rows_reach_the_snapshot_on_tpu(num_experts):
    capturer = _tpu_capturer(num_experts)
    # (real rows, bucket) of three forwards in one step.
    forwards = [(20, 32), (9, 16), (5, 16)]
    expected, row_offset = [], 0
    for seed, (num_tokens, bucket) in enumerate(forwards):
        ids = _ids(bucket, num_experts, seed)
        _forward(capturer, ids)
        capturer.stage_rows(bucket, row_offset)
        expected.append(ids[:num_tokens])
        row_offset += num_tokens
    host = capturer.snapshot_routing_data(row_offset).to("cpu", non_blocking=True)
    copied = torch.tpu.Event()
    copied.record()
    # The next step rewrites the staging table before that copy is waited on.
    _forward(capturer, _ids(16, num_experts, seed=9))
    capturer.stage_rows(16, 0)
    capturer.snapshot_routing_data(16).cpu()
    copied.synchronize()

    assert host.dtype == capturer.output_dtype
    assert host.shape[0] == 64
    assert np.array_equal(host[:row_offset].numpy(), torch.cat(expected).numpy())


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
@pytest.mark.parametrize("num_experts", [256, 512])
def test_step_rows_reach_the_snapshot_on_tpu(num_experts):
    """Each forward's real rows land after the previous forward's, in the
    connector's dtype (uint8 up to 256 experts, else uint16), and a snapshot
    keeps them while the next step rewrites the staging table."""
    run_in_isolated_process(partial(_step_rows_reach_the_snapshot_on_tpu, num_experts))


def _staging_compiles_once_per_bucket_on_tpu():
    capturer = _tpu_capturer()

    def stage(bucket, row_offset):
        capturer.stage_rows(bucket, row_offset)
        capturer.snapshot_routing_data(bucket).cpu()

    for bucket in TOKEN_PADDINGS:
        stage(bucket, 0)
    compiles = torch.tpu._get_cache_stats().num_compilations
    for bucket in TOKEN_PADDINGS:
        for row_offset in (5, STAGING_ROWS - bucket):
            stage(bucket, row_offset)

    assert torch.tpu._get_cache_stats().num_compilations == compiles


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_staging_compiles_once_per_bucket_on_tpu():
    """The offset is program data, so a new offset compiles nothing.

    Each write is waited on, as the runner's forwards separate them: eager
    writes deferred back to back fuse into programs of their own.
    """
    run_in_isolated_process(_staging_compiles_once_per_bucket_on_tpu)
