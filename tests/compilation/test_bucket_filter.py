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
"""Two compiled modules splitting the token buckets between them.

A model can send sequence-parallel-eligible buckets to a second
compiled backbone. Each module must compile, and reload from the cache, only
its own buckets, so every bucket is built exactly once and served by the graph
that was traced for it. Runs vLLM's real compile pipeline against the fake
torch_tpu backend of test_shape_variants.
"""

import os
from functools import partial
from types import SimpleNamespace

import pytest
import test_shape_variants as sv
import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.piecewise_backend import PiecewiseBackend
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.utils import Range
from vllm.forward_context import set_forward_context

import vllm_torchtpu
from vllm_torchtpu.compilation import bucket_filter, shape_variants
from vllm_torchtpu.patch_registry import PATCHES

TP = 8
# 960: under the floor. 1040: over it, but its shard is not row-aligned.
BUCKETS = [16, 960, 1040, 2048, 4096]
SP_BUCKETS = {2048, 4096}


@support_torch_compile(dynamic_arg_dims={"x": 0})
class Replicated(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # Its own compile prefix and cache dir (_patch_vllm_compile_prefix_isolation).
        self.prefix = "replicated"
        self.weight = nn.Parameter(torch.ones(sv.HIDDEN))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sv.WAVE(x * self.weight)


@support_torch_compile(dynamic_arg_dims={"x": 0})
class Sharded(nn.Module):
    """One op per rank's shard, so its executable is told apart by count.

    Holds the replicated module's parameters without registering them, as a
    sequence-parallel backbone wrapper would.
    """

    def __init__(self, *, backbone: nn.Module) -> None:
        super().__init__()
        self.prefix = "sharded"
        object.__setattr__(self, "_backbone", backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self._backbone.weight
        tokens = x.shape[0]
        torch._check(tokens % TP == 0)
        shards = x.reshape(TP, tokens // TP, x.shape[1]).unbind(0)
        return torch.cat([sv.WAVE(shard) for shard in shards])


def sharded_eligible(tokens: int) -> bool:
    """Large buckets whose per-rank shard is row-aligned go to the sharded graph."""
    return tokens >= 1024 and (tokens // TP) % 128 == 0


class Router(nn.Module):
    """A model routing large prefill buckets to a second compiled module."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__()
        with set_current_vllm_config(config):
            self.replicated, self.sharded = self.build(config)
        bucket_filter.assign(self.route, (self.replicated, self.sharded))

    def route(self, tokens: int) -> nn.Module:
        return self.sharded if sharded_eligible(tokens) else self.replicated

    def build(self, config: VllmConfig) -> tuple[nn.Module, nn.Module]:
        replicated = Replicated(vllm_config=config)
        return replicated, Sharded(vllm_config=config, backbone=replicated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module = self.route(x.shape[0])
        with bucket_filter.serving(module):
            return module(x)


def _prepare(cache_root, router=Router, buckets=BUCKETS, trace_first=False):
    sv.BUCKETS[:] = buckets
    if trace_first:
        os.environ.update(
            TPU_PARALLEL_PRECOMPILE="1",
            TPU_PRECOMPILE_BIGGEST_FIRST="1",
            TPU_PRECOMPILE_PEER_REPLAY="0",
            LOCAL_RANK="0",
        )
    config, backend = sv.prepare(cache_root)
    # The registry order: the filter wraps the compile_all_ranges that
    # TPU_PARALLEL_PRECOMPILE installs.
    vllm_torchtpu._patch_vllm_compile_all_ranges()
    bucket_filter.patch_bucket_filter()
    vllm_torchtpu._patch_vllm_compile_prefix_isolation()
    return config, backend, router(config)


def _serve_all(router, config, buckets=BUCKETS):
    served = {}
    with set_current_vllm_config(config):
        for bucket in buckets:
            sv.CALLS.clear()
            with set_forward_context({}, config):
                router(torch.zeros(bucket, sv.HIDDEN))
            served[bucket] = sv.CALLS[-1]
    return served


def _expected():
    return {b: (b, TP if b in SP_BUCKETS else 1) for b in BUCKETS}


def _each_bucket_compiled_once_by_its_graph(cache_root):
    config, backend, router = _prepare(cache_root)
    served = _serve_all(router, config)
    assert sorted(backend.compiled) == sorted(_expected().values()), backend.compiled
    assert served == _expected(), served


def test_each_bucket_is_compiled_once_by_the_graph_that_serves_it(tmp_path):
    sv.in_child(partial(_each_bucket_compiled_once_by_its_graph, str(tmp_path)))


def _warm_start_loads_only_own_buckets(cache_root):
    _each_bucket_compiled_once_by_its_graph(cache_root)
    torch._dynamo.reset()
    config, backend, router = _prepare(cache_root)
    served = _serve_all(router, config)
    assert backend.compiled == [], backend.compiled
    assert served == _expected(), served


def test_warm_start_loads_each_graph_only_its_own_buckets(tmp_path):
    sv.in_child(partial(_warm_start_loads_only_own_buckets, str(tmp_path)))


def _one_copy_of_the_weights(cache_root):
    _, _, router = _prepare(cache_root)
    assert [n for n, _ in router.named_parameters()] == ["replicated.weight"]


def test_the_sharded_graph_registers_no_second_copy_of_the_weights(tmp_path):
    sv.in_child(partial(_one_copy_of_the_weights, str(tmp_path)))


@support_torch_compile(dynamic_arg_dims={"x": 0})
class TieredReplicated(nn.Module):
    """Branches on the token count, so the ladder retraces it at 960."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waves = 2 if x.shape[0] >= 512 else 1
        return torch.cat([sv.WAVE(chunk) for chunk in torch.chunk(x, waves)])


@support_torch_compile(dynamic_arg_dims={"x": 0})
class TieredSharded(nn.Module):
    """Branches on the shard length, so the ladder retraces it at 4096 -- the
    last bucket, after the replicated module's last one."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[0]
        torch._check(tokens % TP == 0)
        shards = x.reshape(TP, tokens // TP, x.shape[1]).unbind(0)
        per_shard = 2 if tokens // TP >= 512 else 1
        return torch.cat([sv.WAVE(s) for s in shards for _ in range(per_shard)])


class TieredRouter(Router):
    def build(self, config: VllmConfig) -> tuple[nn.Module, nn.Module]:
        replicated = TieredReplicated(vllm_config=config, prefix="replicated")
        sharded = TieredSharded(vllm_config=config, prefix="sharded")
        return replicated, sharded


# Waves each bucket's executable runs, per bucket list, for TieredRouter.
TIERED = {16: 1, 960: 2, 1040: 2, 2048: TP, 4096: 2 * TP}
# 3000 stays replicated (its shard is not row-aligned) between two sharded
# buckets, so both modules hold a refusal at the retrace for 3000.
INTERLEAVED = [16, 2048, 3000, 4096]
TIERED_INTERLEAVED = {16: 1, 2048: TP, 3000: 2, 4096: 2 * TP}


def _check_tiered(router, config, backend, buckets, expected):
    compiled = list(backend.compiled)
    # Served after the ladder (spec-decode warmup, decode): no new trace.
    served = _serve_all(router, config, buckets)
    assert backend.compiled == compiled, backend.compiled[len(compiled) :]
    assert served == {b: (b, w) for b, w in expected.items()}, served
    return compiled


def _a_retrace_keeps_the_other_graph(cache_root, buckets, expected):
    config, backend, router = _prepare(cache_root, TieredRouter, buckets)
    sv.warm(router, config, buckets)
    compiled = _check_tiered(router, config, backend, buckets, expected)
    assert sorted(compiled) == sorted((b, w) for b, w in expected.items()), compiled


@pytest.mark.parametrize(
    "buckets, expected",
    [(BUCKETS, TIERED), (INTERLEAVED, TIERED_INTERLEAVED)],
    ids=["sharded-last", "interleaved"],
)
def test_a_retrace_for_one_graph_keeps_the_other_graph_live(
    tmp_path, buckets, expected
):
    sv.in_child(
        partial(_a_retrace_keeps_the_other_graph, str(tmp_path), buckets, expected)
    )


@support_torch_compile(dynamic_arg_dims={"x": 0})
class Unrouted(nn.Module):
    """A compiled module the router never assigns (say a vision encoder's)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.prefix = "unrouted"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class TieredRouterWithUnrouted(TieredRouter):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        with set_current_vllm_config(config):
            self.unrouted = Unrouted(vllm_config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.unrouted(x))


def _an_unassigned_module_keeps_the_other_refusal(cache_root):
    """The unassigned module serves every bucket, so it must not make a
    partial retrace drop the other module's pending refusal (4096)."""
    config, backend, router = _prepare(
        cache_root, TieredRouterWithUnrouted, INTERLEAVED
    )
    sv.warm(router, config, INTERLEAVED)
    _check_tiered(router, config, backend, INTERLEAVED, TIERED_INTERLEAVED)


def test_an_unassigned_compiled_module_does_not_drop_refusals(tmp_path):
    sv.in_child(partial(_an_unassigned_module_keeps_the_other_refusal, str(tmp_path)))


def _trace_first_keeps_each_modules_last_trace(cache_root):
    """TPU_PRECOMPILE_TRACE_FIRST: defer() runs inside serving(), so each
    deferred trace knows its module, and compile_deferred keeps the replicated
    module's trace live although the sharded module traced after it."""
    config, backend, router = _prepare(cache_root, TieredRouter, trace_first=True)
    with set_current_vllm_config(config), shape_variants.warmup() as tracked:
        with shape_variants.trace_only() as deferred:
            for bucket in BUCKETS:
                if tracked.needs_retrace(bucket):
                    shape_variants.retrace(router, config, bucket)
                with set_forward_context({}, config):
                    router(torch.zeros(bucket, sv.HIDDEN))
            assert not tracked.refused
        assert backend.compiled == []
        assert {b._tpu_serves for b in deferred.backends} == {
            router.replicated,
            router.sharded,
        }
        deferred.compile()
    compiled = _check_tiered(router, config, backend, BUCKETS, TIERED)
    assert sorted(compiled) == sorted((b, w) for b, w in TIERED.items()), compiled


def test_trace_first_keeps_each_modules_last_trace_live(tmp_path):
    sv.in_child(partial(_trace_first_keeps_each_modules_last_trace, str(tmp_path)))


class _Compiled(nn.Module, TorchCompileWithNoGuardsWrapper):
    """A stand-in compiled module (never traced)."""

    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _routed(large_from):
    """Two stand-in modules split at ``large_from`` tokens by one route."""
    small, large = _Compiled(), _Compiled()
    bucket_filter.assign(lambda n: large if n >= large_from else small, (small, large))
    return small, large


def test_unfiltered_outside_a_routed_call():
    small, large = _routed(1000)
    assert bucket_filter.serves(16) and bucket_filter.current() is None
    with bucket_filter.serving(large):
        assert bucket_filter.current() is large
        assert not bucket_filter.serves(16)
        assert bucket_filter.serves(2048)
    with bucket_filter.serving(small):
        assert bucket_filter.serves(16)
        assert not bucket_filter.serves(2048)
    assert bucket_filter.serves(16)


def test_one_route_gives_each_bucket_exactly_one_module():
    small, large = _routed(1024)
    for n in (16, 1023, 1024, 4096):
        assert [bucket_filter.module_serves(m, n) for m in (small, large)].count(
            True
        ) == 1
    # A module that was never assigned serves every bucket.
    assert bucket_filter.module_serves(_Compiled(), 16)


def test_assign_needs_compiled_modules_and_serving_an_assigned_one():
    with pytest.raises(TypeError, match="compiled modules"):
        bucket_filter.assign(lambda n: None, (_Compiled(), nn.Linear(1, 1)))
    with pytest.raises(ValueError, match="assign"), bucket_filter.serving(_Compiled()):
        pass


def test_retrace_rejects_a_route_that_returns_no_assigned_module():
    model = nn.Module()
    model.small, model.large = _Compiled(), _Compiled()
    stranger = _Compiled()
    bucket_filter.assign(
        lambda n: stranger if n == 1024 else model.small, (model.small, model.large)
    )
    with pytest.raises(AssertionError, match="bucket 1024 is served by 0"):
        shape_variants.retrace(model, None, 1024)


def test_bucket_filter_wraps_the_parallel_compile_all_ranges():
    """TPU_PARALLEL_PRECOMPILE replaces compile_all_ranges outright, so the
    bucket filter must be applied after it to wrap the method actually used."""
    order = [p.target for p in PATCHES]
    assert order.index(
        "vllm_torchtpu.compilation.bucket_filter:patch_bucket_filter"
    ) > order.index("vllm_torchtpu:_patch_vllm_compile_all_ranges")


def test_filtered_entries_are_put_back_in_order(monkeypatch):
    """Each module builds only its own buckets; afterwards range_entries holds
    every entry again, in its original order."""
    seen = []
    monkeypatch.setattr(
        PiecewiseBackend,
        "compile_all_ranges",
        lambda self: seen.append([r.start for r in self.range_entries]),
    )
    bucket_filter.patch_bucket_filter()
    ranges = [Range(s, s) for s in BUCKETS]
    backend = SimpleNamespace(range_entries={r: r.start for r in ranges})
    _, large = _routed(2048)
    with bucket_filter.serving(large):
        PiecewiseBackend.compile_all_ranges(backend)
    assert seen == [sorted(SP_BUCKETS)]
    assert list(backend.range_entries) == ranges
