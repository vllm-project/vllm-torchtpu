# SPDX-License-Identifier: Apache-2.0
"""Four-host PR 2821 overlay reproduction; see the adjacent README."""

import datetime
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401

ROOT = Path(os.environ["UT_ROOT"])
RANK = int(os.environ["RANK"])
OUT = ROOT / f"rank{RANK}"
OUT.mkdir(exist_ok=True)


def main() -> None:
    torch._dynamo.config.recompile_limit = 64
    dist.init_process_group("tpu_dist", timeout=datetime.timedelta(seconds=600))
    cpu = dist.new_group(backend="gloo")
    torch.empty(1, device="tpu").cpu()
    from torch_tpu._internal.pallas import get_pallas_mesh

    rank_tensor = torch.tensor([RANK], dtype=torch.int32, device="tpu")
    torch.manual_seed(271)
    router = (torch.randn(8192, 512) / 90).to(torch.bfloat16).to("tpu")
    gate = (torch.randn(8192, 1) / 90).to(torch.bfloat16).to("tpu")
    # Shared-expert TP shards repeat across DP, as in the service.
    torch.manual_seed(731 + RANK % 4)
    sw1 = torch.randn(8192, 1024).to(torch.float8_e4m3fn).to("tpu")
    sw2 = torch.randn(512, 8192).to(torch.float8_e4m3fn).to("tpu")
    ss1 = torch.full((1, 1024), 1 / 90, dtype=torch.float32, device="tpu")
    ss2 = torch.full((1, 8192), 1 / 23, dtype=torch.float32, device="tpu")
    torch.manual_seed(991 + RANK)

    def fp4(shape):
        low = torch.randint(1, 4, shape, dtype=torch.uint8)
        high = torch.randint(1, 4, shape, dtype=torch.uint8)
        signs = torch.randint(0, 2, shape, dtype=torch.uint8) * 8
        return (
            (low + signs + ((high + signs) << 4)).view(torch.float4_e2m1fn_x2).to("tpu")
        )

    w1 = fp4((16, 8192, 2048))
    w2 = fp4((16, 2048, 4096))
    s1 = torch.full((16, 32, 1, 4096), 1 / 90, device="tpu")
    s2 = torch.full((16, 8, 1, 8192), 1 / 45, device="tpu")

    weights = (router, gate, sw1, ss1, sw2, ss2, w1, w2, s1, s2)

    def input_cpu(dp, capacity, requests):
        gen = torch.Generator().manual_seed(1000 + dp)
        x = (torch.randn(64, 8192, generator=gen) * 0.25).to(torch.bfloat16)
        valid = 4 * (requests // 8 + int(dp < requests % 8))
        if valid > capacity:
            raise ValueError("Actual requests exceed per-DP compile capacity")
        # Keep nonzero padding; the runtime mask must determine communication.
        return x[:capacity].clone()

    attributes = dict(torch.tpu._get_local_device_attributes())
    topology = [None] * 32
    dist.all_gather_object(topology, attributes, group=cpu)
    (OUT / "topology.json").write_text(json.dumps(topology, default=str))
    x = input_cpu(RANK // 4, 64, 96).to("tpu")
    mask = (torch.arange(64) >= 48).to(torch.int32).reshape(-1, 1).to("tpu")
    inputs = (rank_tensor, x, *weights, mask)
    from adapter import build_pr_op

    prmesh = get_pallas_mesh(("dp", "tp"), (8, 4))
    results = {}

    def pr_fn(mode, iterations=0, diagnostics=False):
        op = build_pr_op(prmesh, mode, iterations, diagnostics)

        @op.register_fake
        def fake(rank, x, *rest):
            return x.new_empty(x.shape), x.new_empty(
                (1, 1, 1 if iterations else 8),
                dtype=torch.float32 if iterations else torch.int32,
            )

        def run(rank, x, *rest):
            a, b = op(*(t.reshape(1, 1, *t.shape) for t in (rank, x, *rest)))
            return a.reshape(x.shape), b.reshape(-1)

        return run

    diagnostic = torch.compile(
        pr_fn("capacity", diagnostics=True), backend="tpu", fullgraph=True
    )
    capacity = diagnostic(*inputs)[1].cpu().tolist()
    assert capacity[2] == 0, f"Capacity overflow: {capacity}"
    (OUT / "capacity.json").write_text(json.dumps(capacity))
    full = torch.compile(pr_fn("full"), backend="tpu", fullgraph=True)
    reference = full(*inputs)[0].cpu()
    modes = {"pr_capacity": pr_fn("capacity")}
    from torch_tpu._internal.profiler import TpuProfilerConfig

    profiled = []

    def profile_one(name, f):
        path = OUT / "profile" / (name + "_host100")
        cfg = TpuProfilerConfig(
            run_dir=str(path),
            host_tracer_level=1,
            device_tracer_level=1,
            python_tracer_level=0,
            experimental_options={"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"},
        )
        dist.barrier(group=cpu)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(path), use_gzip=True
            ),
            experimental_config=cfg,
        ):
            time.sleep(0.2)
            dist.barrier(group=cpu)
            for _ in range(5):
                f(*inputs)
            torch.tpu.synchronize()
            dist.barrier(group=cpu)
            time.sleep(0.2)
        profiled.append(name + "_host100")
        (OUT / "profile-cases.json").write_text(json.dumps(profiled))
        print("PROFILE_DONE", name, flush=True)

    for name, fn in modes.items():
        print("COMPILE", name, flush=True)
        f = torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)
        y = f(*inputs)[0]
        torch.tpu.synchronize()
        actual = y.cpu()
        assert torch.isfinite(actual).all(), name
        diff = actual[:48].float() - reference[:48].float()
        comparison = {
            "max_abs_error": diff.abs().max().item(),
            "rms_error": diff.square().mean().sqrt().item(),
            "reference_rms": reference[:48].float().square().mean().sqrt().item(),
        }
        print("VALIDATION", name, comparison, flush=True)
        torch.testing.assert_close(actual[:48], reference[:48], rtol=0.03, atol=0.001)
        for _ in range(20):
            f(*inputs)
        torch.tpu.synchronize()
        if os.environ.get("REPRO_PROFILE", "1") == "1":
            profile_one(name, f)
        else:
            timings = []
            for _ in range(10):
                dist.barrier(group=cpu)
                start = time.perf_counter()
                for _ in range(100):
                    f(*inputs)
                torch.tpu.synchronize()
                elapsed = (time.perf_counter() - start) * 1e6 / 100
                all_times = [None] * 32
                dist.all_gather_object(all_times, elapsed, group=cpu)
                timings.append(max(all_times))
            (OUT / "timing.json").write_text(
                json.dumps(
                    {"max_rank_us": timings, "mean_us": statistics.mean(timings)}
                )
            )
        results[name + "_host100"] = {
            "comparison": comparison,
            "profile_calls": 5 if os.environ.get("REPRO_PROFILE", "1") == "1" else 0,
        }
        (OUT / "results.json").write_text(json.dumps(results, indent=2))
        print("CASE", name, "profile complete", flush=True)
    (OUT / "done").write_text("ok")
    dist.barrier(group=cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
