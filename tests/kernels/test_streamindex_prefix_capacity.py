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
"""Prefix capacity planning and complete-pipeline branch contracts (CPU)."""

import dataclasses
import functools
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tests.kernels.test_streamindex_topk_backend import DenseKeyReader, PrefixSchedule
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.components import (
    StreamIndexComponents,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
)


@dataclasses.dataclass(frozen=True)
class LegacySchedule(PrefixSchedule):
    offset: int = 0

    def validate(self, compression_ratio):
        if not 0 <= self.offset < compression_ratio:
            raise ValueError("offset must fit the compression group")

    def positions(self, positions):
        return positions - self.offset

    def visible_lengths(self, lengths):
        return jnp.maximum(lengths, 0)


@dataclasses.dataclass(frozen=True)
class PreservingSchedule(LegacySchedule):
    admitted: bool = True

    @property
    def preserves_kv_extent(self):
        return self.admitted


def preserving_components(offset=0, admitted=True):
    return StreamIndexComponents(
        DenseKeyReader(scale_gain=1.0),
        LegacySchedule(offset)
        if admitted is None
        else PreservingSchedule(offset, admitted),
    )


module = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)


@pytest.mark.parametrize(
    "tokens,ratio,k,page,blocks,expected",
    [
        (8192, 4, 512, 32, (8, 8, 8), 2048),
        (8193, 4, 512, 32, (8, 8, 8), 2304),
        (513, 4, 512, 32, (8, 16, 8), 512),
        (8192, 4, 512, 32, (12, 8, 8), 2304),
    ],
)
def test_complete_dma_tiles_fit_physical_score_pitch(
    tokens, ratio, k, page, blocks, expected
):
    sublanes = module._score_sublanes(
        (tokens + ratio - 1) // ratio, k, page, blocks, 1024
    )
    assert sublanes * 128 == expected
    # Independent last-tile endpoint, including partial compressed groups.
    for pages in blocks:
        last = (max(k, (tokens + ratio - 1) // ratio) - 1) // (page * pages)
        assert (last + 1) * page * pages <= sublanes * 128


@pytest.mark.parametrize(
    "lengths,starts,distribution,expected",
    [
        ([8192, 2147483647], [0, 8192, 8192], [0, 0, 1], True),
        ([8193, 0], [0, 8192, 8192], [0, 0, 1], False),
        ([32768, 0], [0, 8192, 8192], [0, 0, 1], False),
        ([8191, 32769], [0, 8191, 8192], [0, 0, 2], False),
        ([8191, 0], [0, 8191, 8191], [0, 0, 1], True),
        ([1, 0], [0, 1, 1], [0, 0, 1], True),
        ([0, 0], [0, 0, 0], [0, 0, 0], True),
        ([1, 0], [0, 8192, 8192], [0, 0, 1], False),
        ([8192, 0], [0, 8192, 8192], [0, 0, 3], False),
        ([2147483647, 0], [0, 8192, 8192], [0, 0, 1], False),
        ([1, 0], [0, 1, 1], [-1, 0, 1], False),
        ([1, 0], [0, 1, 1], [1, 0, 1], False),
        ([1, 0], [0, 1, 1], [0, 2, 1], False),
        ([1, 0], [0, 1, 1], [3, 3, 3], False),
    ],
)
def test_active_context_extent_guards_not_visible_floor(
    lengths, starts, distribution, expected
):
    decision = module._query_prefix_fits(
        jnp.array(lengths, jnp.int32),
        jnp.array(starts, jnp.int32),
        jnp.array(distribution, jnp.int32),
        4,
        2048,
    )
    assert bool(decision) == expected


@dataclasses.dataclass(frozen=True)
class Alternate(SparseCoreTopK):
    pass


@pytest.mark.parametrize(
    "backend,bounded,opt_in",
    [
        (SparseCoreTopK(), True, True),
        (Alternate(), True, True),
        (SparseCoreTopK(), False, False),
        (SparseCoreTopK(), False, None),
    ],
)
def test_real_pipeline_selects_same_ids_and_keeps_branch_groups_disjoint(
    monkeypatch, backend, bounded, opt_in
):
    # Only hardware kernels are replaced. Real preparation, schedules, pipeline
    # passes, output stride, selector arguments and branch structure execute.
    outputs = []
    groups = []
    seen = []

    def pallas_call(kernel, **kwargs):
        shape = kwargs["out_shape"].shape
        outputs.append(shape)

        def call(*args):
            bits = jnp.arange(shape[1] * 128, dtype=jnp.int32) + 0x3F000000
            return jnp.broadcast_to(bits.reshape(1, shape[1], 128), shape)

        return call

    def select(self, scores, k, *, row_lengths, scheduling_group_id, return_scores):
        groups.append(scheduling_group_id)
        seen.append(scores.shape)
        slots = jnp.arange(k, dtype=jnp.int32)[None, :]
        ids = jnp.maximum(row_lengths - k, 0)[:, None] + slots
        return jnp.where(slots < jnp.minimum(row_lengths, k)[:, None], ids, -1)

    monkeypatch.setattr(module.pl, "pallas_call", pallas_call)
    monkeypatch.setattr(SparseCoreTopK, "select", select)
    T = 1024
    arguments = (
        jnp.ones((T, 2, 128), jnp.bfloat16),
        jnp.ones((T, 2), jnp.float32),
        jnp.zeros((129, 32, 1, 256), jnp.bfloat16),
        jnp.array([T, 0], jnp.int32),
        jnp.tile(jnp.arange(128, dtype=jnp.int32), 2),
        jnp.array([0, T, T], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
    )
    result = module.streamindex_topk.__wrapped__(
        *arguments,
        k=128,
        compression_ratio=4,
        num_kv_pages_per_block=8,
        num_queries_per_block=(1, 128, 128),
        decode_req_batch_size=8,
        chunk_tokens=512,
        components=preserving_components(3, opt_in),
        topk_backend=backend,
        scheduling_group_namespace=5,
        prefix_capacity=256 if opt_in is True else None,
    )
    visible = np.arange(1, T + 1) // 4
    expected = np.where(
        np.arange(128)[None, :] < np.minimum(visible, 128)[:, None],
        np.maximum(visible - 128, 0)[:, None] + np.arange(128)[None, :],
        -1,
    )
    np.testing.assert_array_equal(result, expected)
    assert {s[1] for s in seen} == ({256, 4096} if bounded else {4096})
    actual_groups = {g for g in groups if g is not None}
    assert actual_groups == {
        config.scoped_scheduling_group_id(5, i) for i in range(2 if bounded else 1)
    }
    assert all(s[0] == 512 and s[2] == 128 for s in outputs)
    assert preserving_components().schedule.preserves_kv_extent


def test_standalone_and_native_branch_group_domains_do_not_alias():
    config.reset_scheduling_group_ids()
    base = config.reserve_scheduling_group_ids(30)
    standalone = {2 * (base + i) - 1 for i in range(30)}
    native = {
        config.scoped_scheduling_group_id(ns, i) for ns in (1, 2, 3) for i in range(30)
    }
    assert len(native) == 90 and not standalone & native


@functools.cache
def capacity_probe(preserves):
    from torch_tpu._internal import pallas

    components = preserving_components(admitted=preserves)

    def core(
        query: jax.Array,
        cache: jax.Array,
        lengths: jax.Array,
        table: jax.Array,
        requests: jax.Array,
        starts: jax.Array,
        context: jax.Array,
        scheduling_group_namespace: int = 0,
    ) -> jax.Array:
        return module.streamindex_topk(
            query,
            jnp.ones(query.shape[:2], jnp.float32),
            cache,
            context + jnp.diff(starts),
            table.reshape(-1),
            starts,
            jnp.array([0, 0, 1], jnp.int32),
            k=512,
            compression_ratio=1,
            num_kv_pages_per_block=2,
            num_queries_per_block=(1, 128, 128),
            chunk_tokens=512,
            components=components,
            prefix_capacity=1024 if preserves else None,
            scheduling_group_namespace=scheduling_group_namespace,
        )

    op = pallas.jax_op(f"streamindex_capacity_test::select_{preserves}", core)

    @op.register_fake
    def fake(
        query,
        cache,
        lengths,
        table,
        requests,
        starts,
        context,
        scheduling_group_namespace=0,
    ):
        return query.new_empty((len(query), 512), dtype=torch.int32)

    return op


def test_bounded_native_schema_without_device():
    with torch.device("meta"):
        q = torch.empty((1024, 4, 128), dtype=torch.bfloat16)
        cache = torch.empty((161, 128, 1, 256), dtype=torch.bfloat16)
        lengths = torch.empty((1024,), dtype=torch.int32)
        table = torch.empty((8, 80), dtype=torch.int32)
        requests = torch.empty((1024,), dtype=torch.int32)
        starts = torch.empty((9,), dtype=torch.int32)
        context = torch.empty((8,), dtype=torch.int32)
        for preserves in (False, True):
            ids = capacity_probe(preserves)(
                q, cache, lengths, table, requests, starts, context
            )
            assert ids.shape == (1024, 512) and ids.dtype == torch.int32


@pytest.mark.tpu_test
def test_native_prefix_capacity_and_full_fallback_share_one_bucket():
    from vllm.config.utils import Range

    from tests.kernels.test_streamindex_scheduling import fixture_inputs
    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

    full, bounded = capacity_probe(False), capacity_probe(True)
    compiler = TpuCompilerAdaptor()
    graphs = []

    def backend(graph, args):
        graphs.append(graph.code)
        compiled, _ = compiler.compile(graph, args, {}, Range(1, 1))
        assert compiled is not None
        return compiled

    def compare(*args):
        return full(*args), bounded(*args)

    compiled = torch.compile(compare, backend=backend, fullgraph=True, dynamic=False)
    for history in (0, 4096, None, 0):
        cpu = list(fixture_inputs(1024))
        cpu[2] = (
            torch.zeros(1024, dtype=torch.int32)
            if history is None
            else history + torch.arange(1, 1025, dtype=torch.int32)
        )
        cpu[4] = torch.zeros(1024, dtype=torch.int32)
        cpu[5] = torch.tensor(
            [0] + [0 if history is None else 1024] * 8, dtype=torch.int32
        )
        cpu[6] = torch.tensor([history or 0] + [0] * 7, dtype=torch.int32)
        args = tuple(value.to("tpu") for value in cpu)
        actual_full, actual_bounded = (value.cpu().numpy() for value in compiled(*args))
        visible = cpu[2].numpy()[:, None]
        slots = np.arange(512)[None, :]
        expected = np.where(
            slots < 512 - np.minimum(visible, 512), -1, visible - 512 + slots
        )
        np.testing.assert_array_equal(np.sort(actual_full, axis=1), expected)
        np.testing.assert_array_equal(actual_bounded, actual_full)
        torch.testing.assert_close(
            args[1].cpu().view(torch.uint16), cpu[1].view(torch.uint16), rtol=0, atol=0
        )
    assert len(graphs) == 1


def dma_endpoints(lengths, starts, tile, *, batch=1, ratio=1, tokens=1, end=None):
    meta = module.metadata.compute_metadata(
        seq_lens=jnp.array(lengths, jnp.int32),
        cu_q_lens=jnp.array(starts, jnp.int32),
        start_seq_idx=0,
        end_seq_idx=len(lengths) if end is None else end,
        bq_sz=1,
        bkv_sz=tile,
        pages_per_seq=128,
        page_size=32,
        max_num_tokens=tokens,
        compression_ratio=ratio,
        static_q_len=1 if batch > 1 else None,
        seq_batch_size=batch,
        chunk_token_start=0,
        chunk_tokens=tokens,
    )
    return (np.asarray(meta.bkv_idx[: int(np.sum(meta.num_steps))]) + 1) * tile


def record_score_widths(monkeypatch):
    executed = []

    def pallas_call(kernel, **kwargs):
        output = kwargs["out_shape"]
        return lambda *args: jnp.zeros(output.shape, output.dtype)

    def select(self, scores, k, **kwargs):
        jax.debug.callback(lambda: executed.append(scores.shape[1]), ordered=True)
        return jnp.zeros((scores.shape[0], k), jnp.int32)

    # Only hardware bodies are replaced; JIT, metadata and runtime guards are real.
    monkeypatch.setattr(module.pl, "pallas_call", pallas_call)
    monkeypatch.setattr(SparseCoreTopK, "select", select)
    module.streamindex_topk.clear_cache()
    return executed


def run_prefix_case(lengths, starts, distribution, *, ratio, capacity, blocks):
    tokens = starts[-1]
    result = module.streamindex_topk(
        jnp.ones((tokens, 2, 128), jnp.bfloat16),
        jnp.ones((tokens, 2), jnp.float32),
        jnp.zeros((129, 8, 4, 256), jnp.uint8),
        jnp.array(lengths, jnp.int32),
        jnp.tile(jnp.arange(128, dtype=jnp.int32), len(lengths)),
        jnp.array(starts, jnp.int32),
        jnp.array(distribution, jnp.int32),
        k=128,
        compression_ratio=ratio,
        num_kv_pages_per_block=blocks,
        num_queries_per_block=(1, 128, 128),
        decode_req_batch_size=4,
        chunk_tokens=512,
        prefix_capacity=capacity,
        scheduling_group_namespace=5,
    )
    result.block_until_ready()


def test_complete_mixed_dma_requires_full_pipeline(monkeypatch):
    executed = record_score_widths(monkeypatch)
    for context, expected_width in ((2048, 2304), (2200, 4224)):
        for tile in (384, 256, 512):
            np.testing.assert_array_equal(
                dma_endpoints([context], [0, 1], tile),
                [start + tile for start in range(0, context, tile)],
            )
        executed.clear()
        run_prefix_case(
            [context, 0],
            [0, 2048, 2048],
            [0, 0, 1],
            ratio=1,
            capacity=2048,
            blocks=(12, 8, 16),
        )
        assert executed == [expected_width] * 4
    assert dma_endpoints([2200], [0, 1], 512)[-1] == 2560


def test_idle_decode_neighbor_widens_real_grouped_metadata(monkeypatch):
    starts = [0, 1, 1, 1, 1, 1024]
    lengths = [1, 4096, 0, 0, 1023]
    np.testing.assert_array_equal(
        dma_endpoints(lengths, starts, 256, batch=4, ratio=4, tokens=512, end=4),
        [256, 512, 768, 1024],
    )
    executed = record_score_widths(monkeypatch)
    for idle_context, expected_width in ((1024, 256), (4096, 4096)):
        executed.clear()
        run_prefix_case(
            [1, idle_context, 0, 0, 1023],
            starts,
            [4, 4, 5],
            ratio=4,
            capacity=256,
            blocks=(8, 8, 8),
        )
        assert executed == [expected_width] * 2
