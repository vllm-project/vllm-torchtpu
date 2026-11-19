# SPDX-License-Identifier: Apache-2.0
"""Two compiled modules splitting the token buckets between them.

The qwen4_exp router sends sequence-parallel-eligible buckets to a second
compiled backbone. Each module must compile, and reload from the cache, only
its own buckets, so every bucket is built exactly once and served by the graph
that was traced for it. Runs vLLM's real compile pipeline against the fake
torch_tpu backend of test_shape_variants.
"""

import os
from functools import partial

import test_shape_variants as sv
import torch
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context

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

    Holds the replicated module's parameters without registering them, as
    Qwen4ExpBackboneSP holds the backbone.
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


class Router(nn.Module):
    """Qwen4ExpForCausalLM's routing, with the real eligibility rule."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__()
        from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp

        with set_current_vllm_config(config):
            self.replicated, self.sharded = self.build(config)
        eligible = partial(sp.sequence_parallel_eligible, tp_size=TP)
        bucket_filter.assign(self.sharded, eligible)
        bucket_filter.assign(self.replicated, lambda n: not eligible(n))

    def build(self, config: VllmConfig) -> tuple[nn.Module, nn.Module]:
        replicated = Replicated(vllm_config=config)
        return replicated, Sharded(vllm_config=config, backbone=replicated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        serves_sharded = bucket_filter.assigned(self.sharded)
        if serves_sharded(x.shape[0]):
            with bucket_filter.serving(serves_sharded), set_model_tag("sharded"):
                return self.sharded(x)
        with bucket_filter.serving(bucket_filter.assigned(self.replicated)):
            return self.replicated(x)


def _prepare(cache_root, router=Router, buckets=BUCKETS):
    os.environ["QWEN4_EXP_PREFILL_SP"] = "1"
    os.environ["QWEN4_EXP_PREFILL_SP_MIN_TOKENS"] = "1024"
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


PADDED_BUCKETS = [16, 960, 1024, 2048, 3072, 4096]
HC_TILE = 256


def hc_kernel(x: torch.Tensor) -> torch.Tensor:
    """GatedResidual's gate: the Pallas kernel only on rows dividing its tile."""
    if x.shape[0] >= HC_TILE and x.shape[0] % HC_TILE == 0:
        return sv.WAVE(x)
    return sv.WAVE(sv.WAVE(x))  # the XLA fallback: another structure


@support_torch_compile(dynamic_arg_dims={"x": 0})
class PaddedSharded(nn.Module):
    """QWEN4_EXP_PREFILL_SP_PAD_HC: the shard is padded to the HC tile, so
    1024 and 3072 (128 and 384 rows at TP8) take the kernel like 2048/4096."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp

        tokens = x.shape[0]
        torch._check(tokens % TP == 0)
        rows = tokens // TP
        shards = x.reshape(TP, rows, x.shape[1]).unbind(0)
        return torch.cat(
            [sp.unpad(hc_kernel(sp.pad_shard(s, HC_TILE)), rows) for s in shards]
        )


class PaddedRouter(Router):
    def build(self, config: VllmConfig) -> tuple[nn.Module, nn.Module]:
        return Replicated(vllm_config=config), PaddedSharded(vllm_config=config)


def _padded_shards_trace_once(cache_root):
    config, backend, router = _prepare(cache_root, PaddedRouter, PADDED_BUCKETS)
    _ladder(router, config, PADDED_BUCKETS)
    compiled = list(backend.compiled)
    served = _serve_all(router, config, PADDED_BUCKETS)
    assert backend.compiled == compiled, backend.compiled[len(compiled) :]
    sp_buckets = [1024, 2048, 3072, 4096]
    # One structure for every SP bucket: the kernel path on each shard.
    expected = {b: (b, TP if b in sp_buckets else 1) for b in PADDED_BUCKETS}
    assert served == expected, served
    assert sorted(compiled) == sorted(expected.values()), compiled


def test_padded_shards_put_1024_and_3072_on_one_sp_trace(tmp_path):
    sv.in_child(partial(_padded_shards_trace_once, str(tmp_path)))


def test_unfiltered_outside_a_routed_call():
    assert bucket_filter.serves(16)
    with bucket_filter.serving(lambda n: n > 1000):
        assert not bucket_filter.serves(16)
        assert bucket_filter.serves(2048)
    assert bucket_filter.serves(16)
