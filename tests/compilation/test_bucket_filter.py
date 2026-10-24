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

from vllm_torchtpu.compilation import bucket_filter

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
    """Qwen4ExpForCausalLM.forward's routing, with the real eligibility rule."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__()
        from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp

        self.sp = sp
        with set_current_vllm_config(config):
            self.replicated = Replicated(vllm_config=config)
            self.sharded = Sharded(vllm_config=config, backbone=self.replicated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eligible = partial(self.sp.sequence_parallel_eligible, tp_size=TP)
        if eligible(x.shape[0]):
            with bucket_filter.serving(eligible), set_model_tag("sharded"):
                return self.sharded(x)
        with bucket_filter.serving(lambda n: not eligible(n)):
            return self.replicated(x)


def _prepare(cache_root):
    os.environ["QWEN4_EXP_PREFILL_SP"] = "1"
    os.environ["QWEN4_EXP_PREFILL_SP_MIN_TOKENS"] = "1024"
    sv.BUCKETS[:] = BUCKETS
    import vllm_torchtpu

    vllm_torchtpu._patch_vllm_bucket_filter()
    config, backend = sv.prepare(cache_root)
    return config, backend, Router(config)


def _serve_all(router, config):
    served = {}
    with set_current_vllm_config(config):
        for bucket in BUCKETS:
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


def test_unfiltered_outside_a_routed_call():
    assert bucket_filter.serves(16)
    with bucket_filter.serving(lambda n: n > 1000):
        assert not bucket_filter.serves(16)
        assert bucket_filter.serves(2048)
    assert bucket_filter.serves(16)
