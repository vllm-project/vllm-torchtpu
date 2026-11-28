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

from functools import partial
from types import SimpleNamespace

import pytest
import test_shape_variants as sv
import torch
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.piecewise_backend import PiecewiseBackend
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.utils import Range
from vllm.forward_context import set_forward_context

import vllm_torchtpu
from vllm_torchtpu.compilation import bucket_filter, shape_variants

TP = 8
# 960: under the floor. 1040: over it, but its shard is not row-aligned.
BUCKETS = [16, 960, 1040, 2048, 4096]
SP_BUCKETS = {2048, 4096}


@support_torch_compile(dynamic_arg_dims={"x": 0})
class Replicated(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
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
        eligible = sharded_eligible
        bucket_filter.assign(self.sharded, eligible)
        bucket_filter.assign(self.replicated, lambda n: not eligible(n))

    def build(self, config: VllmConfig) -> tuple[nn.Module, nn.Module]:
        replicated = Replicated(vllm_config=config)
        return replicated, Sharded(vllm_config=config, backbone=replicated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if bucket_filter.module_serves(self.sharded, x.shape[0]):
            with bucket_filter.serving(self.sharded), set_model_tag("sharded"):
                return self.sharded(x)
        with bucket_filter.serving(self.replicated):
            return self.replicated(x)


def _prepare(cache_root, router=Router, buckets=BUCKETS):
    sv.BUCKETS[:] = buckets
    import vllm_torchtpu

    vllm_torchtpu._patch_vllm_bucket_filter()
    config, backend = sv.prepare(cache_root)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waves = 2 if x.shape[0] >= 512 else 1
        return torch.cat([sv.WAVE(chunk) for chunk in torch.chunk(x, waves)])


@support_torch_compile(dynamic_arg_dims={"x": 0})
class TieredSharded(nn.Module):
    """Branches on the shard length, so the ladder retraces it at 4096 -- the
    last bucket, after the replicated module's last one."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[0]
        torch._check(tokens % TP == 0)
        shards = x.reshape(TP, tokens // TP, x.shape[1]).unbind(0)
        per_shard = 2 if tokens // TP >= 512 else 1
        return torch.cat([sv.WAVE(s) for s in shards for _ in range(per_shard)])


class TieredRouter(Router):
    def build(self, config: VllmConfig) -> tuple[nn.Module, nn.Module]:
        return TieredReplicated(vllm_config=config), TieredSharded(vllm_config=config)


def _ladder(router, config, buckets=BUCKETS):
    """TPUModelRunner._precompile_backbone over the router."""
    with set_current_vllm_config(config), shape_variants.warmup() as tracked:
        for bucket in buckets:
            if tracked.needs_retrace(bucket):
                shape_variants.retrace(router, config, bucket)
            with set_forward_context({}, config):
                router(torch.zeros(bucket, sv.HIDDEN))
        assert not tracked.refused


def _a_retrace_keeps_the_other_graph(cache_root):
    config, backend, router = _prepare(cache_root, TieredRouter)
    _ladder(router, config)
    compiled = list(backend.compiled)
    # Served after the ladder (spec-decode warmup, decode): no new trace.
    served = _serve_all(router, config)
    assert backend.compiled == compiled, backend.compiled[len(compiled) :]
    expected = {16: 1, 960: 2, 1040: 2, 2048: TP, 4096: 2 * TP}
    assert served == {b: (b, w) for b, w in expected.items()}, served
    assert sorted(compiled) == sorted(served.values()), compiled


def test_a_retrace_for_one_graph_keeps_the_other_graph_live(tmp_path):
    sv.in_child(partial(_a_retrace_keeps_the_other_graph, str(tmp_path)))


class _Compiled(nn.Module, TorchCompileWithNoGuardsWrapper):
    """A stand-in compiled module (never traced)."""

    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _assigned(predicate):
    module = _Compiled()
    bucket_filter.assign(module, predicate)
    return module


def test_unfiltered_outside_a_routed_call():
    large = _assigned(lambda n: n > 1000)
    assert bucket_filter.serves(16) and bucket_filter.current() is None
    with bucket_filter.serving(large):
        assert bucket_filter.current() is large
        assert not bucket_filter.serves(16)
        assert bucket_filter.serves(2048)
    assert bucket_filter.serves(16)


def test_assign_needs_a_compiled_module_and_serving_an_assigned_one():
    with pytest.raises(TypeError, match="compiled module"):
        bucket_filter.assign(nn.Linear(1, 1), lambda n: True)
    with pytest.raises(ValueError, match="assign"), bucket_filter.serving(_Compiled()):
        pass


@pytest.mark.parametrize(
    "small, large",
    [
        (lambda n: n < 1024, lambda n: n > 1024),  # gap at 1024
        (lambda n: n <= 1024, lambda n: n >= 1024),  # overlap at 1024
    ],
)
def test_retrace_rejects_predicates_that_do_not_partition_the_bucket(small, large):
    model = nn.Module()
    model.small, model.large = _assigned(small), _assigned(large)
    with pytest.raises(AssertionError, match="bucket 1024 is served by"):
        shape_variants.retrace(model, None, 1024)


def test_bucket_filter_wraps_the_parallel_compile_all_ranges():
    """TPU_PARALLEL_PRECOMPILE replaces compile_all_ranges outright, so the
    bucket filter must be applied after it to wrap the method actually used."""
    from vllm_torchtpu.patch_registry import PATCHES

    order = [p.target for p in PATCHES]
    assert order.index("vllm_torchtpu:_patch_vllm_bucket_filter") > order.index(
        "vllm_torchtpu:_patch_vllm_compile_all_ranges"
    )


def test_filtered_entries_are_put_back_in_order(monkeypatch):
    """Each module builds only its own buckets; afterwards range_entries holds
    every entry again, in its original order."""
    seen = []
    monkeypatch.setattr(
        PiecewiseBackend,
        "compile_all_ranges",
        lambda self: seen.append([r.start for r in self.range_entries]),
    )
    monkeypatch.setattr(PiecewiseBackend, "load_all_ranges", lambda self: None)
    vllm_torchtpu._patch_vllm_bucket_filter()
    ranges = [Range(s, s) for s in BUCKETS]
    backend = SimpleNamespace(range_entries={r: r.start for r in ranges})
    with bucket_filter.serving(_assigned(lambda n: n in SP_BUCKETS)):
        PiecewiseBackend.compile_all_ranges(backend)
    assert seen == [sorted(SP_BUCKETS)]
    assert list(backend.range_entries) == ranges
