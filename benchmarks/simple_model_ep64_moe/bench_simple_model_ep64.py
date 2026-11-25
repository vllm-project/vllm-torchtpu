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
"""Synthetic single-layer MoE regression and device profiling for EP64."""

import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from types import ModuleType

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_tpu  # noqa: F401


def load_kernel_packages():
    # The synthetic benchmark never initializes vLLM's serving platform.
    # Resolve the source modules in this checkout without registering its
    # serving layers or plugins in the benchmark process.
    source = Path(__file__).resolve().parents[2] / "src"
    for name in ("vllm_torchtpu", "vllm_torchtpu.layers"):
        package = ModuleType(name)
        package.__path__ = [str(source.joinpath(*name.split(".")))]
        sys.modules[name] = package


DP, TP, EP = 16, 4, 64
TOKENS, HIDDEN, INTERMEDIATE, SHARED = 48, 8192, 2048, 2048
EXPERTS, TOPK, SCALE_BLOCK = 768, 10, 256
CALLS, WARMUP, ITERATIONS, REPEATS = 10, 15, 50, 5


def fp4(shape):
    lo = torch.randint(1, 3, shape, dtype=torch.uint8)
    hi = torch.randint(1, 3, shape, dtype=torch.uint8)
    lo += torch.randint(0, 2, shape, dtype=torch.uint8) * 8
    hi += torch.randint(0, 2, shape, dtype=torch.uint8) * 8
    return (lo | (hi << 4)).view(torch.float4_e2m1fn_x2).to("tpu")


def scale(experts, contraction, channels, value, singleton_axis):
    shape = (experts, contraction // SCALE_BLOCK)
    if singleton_axis:
        shape += (1,)
    return torch.full((*shape, channels), value, device="tpu")


def main():
    load_kernel_packages()
    from routed_moe import MODE, SCALE_SINGLETON_AXIS, build_routed

    torch.empty(1, device="tpu").cpu()
    dist.init_process_group("tpu_dist", timeout=timedelta(minutes=15))
    rank = dist.get_rank()
    if dist.get_world_size() != EP:
        raise ValueError("Expected exactly 64 TPU ranks")
    out = Path(os.environ["UT_ROOT"]) / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=True)
    cpu = dist.new_group(backend="gloo")
    for start in range(0, EP, TP):
        group = dist.new_group(list(range(start, start + TP)), backend="tpu")
        if start <= rank < start + TP:
            tp_group = group

    local_experts, shared_shard = EXPERTS // EP, SHARED // TP
    torch.manual_seed(271)
    router = (torch.randn(HIDDEN, EXPERTS) / 90).to(torch.bfloat16).to("tpu")
    gate = (torch.randn(HIDDEN, 1) / 90).to(torch.bfloat16).to("tpu")
    torch.manual_seed(731 + rank % TP)
    sw1 = (torch.randn(HIDDEN, 2 * shared_shard) / 90).to(torch.bfloat16).to("tpu")
    sw2 = (torch.randn(shared_shard, HIDDEN) / 23).to(torch.bfloat16).to("tpu")
    torch.manual_seed(991 + rank)
    w1 = fp4((local_experts, HIDDEN, INTERMEDIATE))
    w2 = fp4((local_experts, INTERMEDIATE, HIDDEN // 2))
    s1 = scale(local_experts, HIDDEN, INTERMEDIATE * 2, 1 / 64, SCALE_SINGLETON_AXIS)
    s2 = scale(local_experts, INTERMEDIATE, HIDDEN, 1 / 32, SCALE_SINGLETON_AXIS)
    torch.manual_seed(1000 + rank // TP)
    x = (torch.randn(TOKENS, HIDDEN) * 0.25).to(torch.bfloat16).to("tpu")
    mask = torch.zeros((TOKENS, 1), dtype=torch.int32, device="tpu")
    r = torch.tensor([rank], dtype=torch.int32, device="tpu")
    args = (r, router, w1, w2, s1, s2, mask, gate, sw1, sw2)
    route = build_routed(DP, TP, TOPK, SCALE_BLOCK, (s1.shape, s2.shape))

    def layer(x, r, router, w1, w2, s1, s2, mask, gate, sw1, sw2):
        gu = x @ sw1
        shared = (F.silu(gu[:, :shared_shard]) * gu[:, shared_shard:]) @ sw2
        shared = shared * torch.sigmoid(x @ gate)
        routed = route(r, x, router, w1, w2, s1, s2, mask)
        dist.all_reduce(shared, group=tp_group)
        return routed + shared

    def chain(x, *operands):
        outputs = []
        for _ in range(CALLS):
            x = x + layer(x, *operands)
            outputs.append(x)
        return tuple(outputs)

    print("COMPILE", MODE, flush=True)
    single = torch.compile(layer, backend="tpu", fullgraph=True, dynamic=False)
    compiled = torch.compile(chain, backend="tpu", fullgraph=True, dynamic=False)
    chain_outputs = compiled(x, *args)
    reference = x
    max_abs = []
    for observed in chain_outputs:
        reference = reference + single(reference, *args)
        actual, expected = observed.cpu(), reference.cpu()
        assert actual.shape == (TOKENS, HIDDEN) and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.001)
        max_abs.append(float((actual.float() - expected.float()).abs().max()))
    alternate = x.clone()
    alternate[:2] = x[:2].flip(0)
    torch.testing.assert_close(
        compiled(alternate, *args)[-1][:2].cpu(),
        actual[:2].flip(0),
        rtol=0.01,
        atol=0.001,
    )
    replicas = [torch.empty_like(actual) for _ in range(EP)]
    dist.all_gather(replicas, actual, group=cpu)
    for lane in range(rank // TP * TP, rank // TP * TP + TP):
        torch.testing.assert_close(actual, replicas[lane], rtol=0, atol=0)
    print("VALIDATED_WITHIN_MODE", MODE, flush=True)

    for _ in range(WARMUP):
        compiled(x, *args)
    torch.accelerator.synchronize()
    samples = []
    for _ in range(REPEATS):
        dist.barrier(group=cpu)
        started = time.perf_counter()
        for _ in range(ITERATIONS):
            compiled(x, *args)
        torch.accelerator.synchronize()
        samples.append((time.perf_counter() - started) * 1e6 / (ITERATIONS * CALLS))
    peers = [None] * EP
    dist.all_gather_object(peers, samples, group=cpu)
    result = dict(
        mode=MODE,
        ep=EP,
        dp=DP,
        tp=TP,
        tokens_per_dp=TOKENS,
        calls=CALLS,
        slowest_rank_host_us=mean(map(max, zip(*peers, strict=True))),
        chain_max_abs=max(max_abs),
        versions={p: version(p) for p in ("torch", "torch-tpu", "jax", "jaxlib")},
    )
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", MODE, result, flush=True)

    from torch_tpu._internal.profiler import TpuProfilerConfig

    profile = out / "profile" / MODE
    config = TpuProfilerConfig(
        run_dir=str(profile),
        host_tracer_level=1,
        device_tracer_level=1,
        python_tracer_level=0,
    )
    dist.barrier(group=cpu)
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(profile), use_gzip=True
            ),
            experimental_config=config,
        )
        if rank < 8
        else nullcontext()
    )
    with profiler:
        time.sleep(0.2)
        dist.barrier(group=cpu)
        with torch.profiler.record_function("SIMPLE_MODEL_MOE_" + MODE):
            for _ in range(ITERATIONS):
                compiled(x, *args)
            torch.accelerator.synchronize()
        dist.barrier(group=cpu)
        time.sleep(0.2)
    dist.barrier(group=cpu)
    print("PROFILE_DONE", MODE, flush=True)
    (out / "done").write_text("ok\n")


if __name__ == "__main__":
    main()
