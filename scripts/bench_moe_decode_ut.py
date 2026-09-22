#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Reproducible single-layer Qwen3.5 decode MoE baseline on eight TPU chiplets.

By default this uses the serving revision's original JAX/Pallas kernels.
--moe-impl dense_expert selects the experimental fused implementation. DP4/TP2: 256 independent tokens, 64/DP, TP replicas, EP8.
Synthetic deterministic weights/inputs and explicitly controlled router pools
are intentional: the original capture did not save activations or expert IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_environment():
    defaults = {
        "TPU_SKIP_MDS_QUERY": "true",
        "JAX_PLATFORMS": "tpu,cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "USE_MOE_SPARSE_CORE": "1",
        "TPU_MOE_OWNER_OUTPUT_MODE": "on",
        "ONEHOT_MOE_PERMUTE_THRESHOLD": "32768",
        "LIBTPU_INIT_ARGS": " --xla_tpu_use_dynamic_smem_negotiation=true --xla_tpu_scoped_vmem_limit_kib=65536",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


class Baseline:
    def __init__(self, seed=20260921):
        configure_environment()
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2
        from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
        from vllm_torchtpu.kernels.router_topk import router_topk
        from vllm_torchtpu.layers.core.fused_moe_gmm import fused_moe_func

        jax.config.update(
            "jax_compilation_cache_dir", str(ROOT / "benchmark_runs/.moe-ut-jax-cache")
        )
        self.jax, self.jnp, self.np = jax, jnp, np
        devices = jax.devices()
        assert len(devices) == 8 and all(d.platform == "tpu" for d in devices), devices
        self.mesh = Mesh(np.array(devices).reshape(4, 2), ("dp", "tp"))
        self.seed = seed
        self.specs = (
            P("dp"),
            P(),
            P(("dp", "tp")),
            P(("dp", "tp")),
            P(("dp", "tp")),
            P(("dp", "tp")),
            P(None, "tp"),
            P("tp"),
            P("tp"),
            P("tp"),
            P(),
            P(),
        )

        def init():
            dp, tp = jax.lax.axis_index("dp"), jax.lax.axis_index("tp")
            rank = dp * 2 + tp

            def key(salt, identity):
                return jax.random.fold_in(jax.random.key(seed + salt), identity)

            def weight(salt, identity, shape):
                return jax.random.randint(key(salt, identity), shape, -15, 16).astype(
                    jnp.float8_e4m3fn
                )

            x = jax.random.normal(key(0, dp), (64, 4096), dtype=jnp.bfloat16)
            router = (jax.random.normal(key(1, 0), (4096, 512)) / 64).astype(
                jnp.bfloat16
            )
            w1 = weight(2, rank, (64, 4096, 2048))
            w2 = weight(3, rank, (64, 1024, 4096))
            s1 = jnp.full((64, 1, 1, 2048), 1 / (9 * 64), jnp.float32)
            s2 = jnp.full((64, 1, 1, 4096), 1 / (9 * 32), jnp.float32)
            sw1 = weight(4, tp, (4096, 1024))
            sw2 = weight(5, tp, (512, 4096))
            ss1 = jnp.full((1024,), 1 / (9 * 64), jnp.float32)
            ss2 = jnp.full((1, 4096), 1 / (9 * 32), jnp.float32)
            sg = (jax.random.normal(key(6, 0), (4096, 1)) / 64).astype(jnp.bfloat16)
            return (
                x,
                router,
                w1,
                w2,
                s1,
                s2,
                sw1,
                sw2,
                ss1,
                ss2,
                sg,
                jnp.zeros((512,), jnp.float32),
            )

        print(
            "Initializing deterministic full-size FP8 weights on 8 chiplets", flush=True
        )
        self.inputs = jax.jit(
            jax.shard_map(
                init, mesh=self.mesh, in_specs=(), out_specs=self.specs, check_vma=False
            )
        )()
        jax.block_until_ready(self.inputs)

        def routing(x, router, bias):
            logits = jnp.matmul(x, router, preferred_element_type=jnp.float32).astype(
                jnp.bfloat16
            )
            logits = jax.lax.all_gather(logits, "dp", axis=0, tiled=True)
            scores = jax.nn.softmax(logits.astype(jnp.float32) + bias[None], axis=-1)
            tw, ti = router_topk(scores, 10)
            tw = tw / jnp.maximum(tw.sum(axis=-1, keepdims=True), 1e-20)
            gx = jax.lax.all_gather(x, "dp", axis=0, tiled=True)
            return gx, tw.astype(jnp.bfloat16), ti

        def shared(x, sw1, sw2, ss1, ss2, sg):
            z = xla_quantized_matmul(x, sw1, ss1)
            z = (jax.nn.silu(z[:, :512]) * z[:, 512:]).astype(jnp.bfloat16)
            z = xla_quantized_matmul(z, sw2, ss2[0])
            gate = jax.nn.sigmoid(
                jnp.matmul(x, sg, preferred_element_type=jnp.float32).astype(
                    jnp.bfloat16
                )
            )
            return (z * gate).astype(jnp.bfloat16)

        def reference_mm(x, w, scale):
            # Independent dense contraction, reproducing the serving GMM's
            # 512-wide dynamic FP8 activation quantization and BF16 accumulation.
            acc = jnp.zeros((x.shape[0], w.shape[1]), jnp.bfloat16)
            for start in range(0, x.shape[1], 512):
                block = x[:, start : start + 512]
                bs = jnp.max(jnp.abs(block), axis=1, keepdims=True) / 448.0
                inv = jnp.where(bs == 0, 0, 1 / bs)
                q = (block * inv).astype(jnp.float8_e4m3fn)
                part = jnp.matmul(
                    q, w[start : start + 512], preferred_element_type=jnp.float32
                ).astype(jnp.bfloat16)
                part = (part * bs.astype(jnp.bfloat16)).astype(jnp.bfloat16)
                part = (part * scale.astype(jnp.bfloat16)).astype(jnp.bfloat16)
                acc = (acc + part).astype(jnp.bfloat16)
            return acc

        def run_local(*args, reference=False, routing_reference=False):
            x, router, w1, w2, s1, s2, sw1, sw2, ss1, ss2, sg, bias = args
            gx, tw, ti = routing(x, router, bias)
            rank = jax.lax.axis_index("dp") * 2 + jax.lax.axis_index("tp")
            if reference or routing_reference:

                def body(e, out):
                    coeff = jnp.sum(jnp.where(ti == rank * 64 + e, tw, 0), axis=-1)

                    def compute(out):
                        if routing_reference:
                            # Same arithmetic kernel, but no production sort,
                            # permutation or unpermutation: all tokens per expert.
                            sizes = jnp.array([256], jnp.int32)
                            a = gmm_v2(
                                gx,
                                w1[e][None],
                                sizes,
                                s1[e][None],
                                fuse_act="silu",
                                zero_initialize=False,
                            )
                            z = gmm_v2(
                                a,
                                w2[e][None],
                                sizes,
                                s2[e][None],
                                zero_initialize=False,
                            )
                        else:
                            a = reference_mm(gx, w1[e], s1[e, 0, 0])
                            a = (jax.nn.silu(a[:, :1024]) * a[:, 1024:]).astype(
                                jnp.bfloat16
                            )
                            z = reference_mm(a, w2[e], s2[e, 0, 0])
                        return out + z.astype(jnp.float32) * coeff[:, None].astype(
                            jnp.float32
                        )

                    return jax.lax.cond(
                        jnp.any(coeff != 0), compute, lambda out: out, out
                    )

                routed = jax.lax.fori_loop(
                    0, 64, body, jnp.zeros((256, 4096), jnp.float32)
                ).astype(jnp.bfloat16)
            else:
                routed = fused_moe_func(
                    gx,
                    w1,
                    w2,
                    s1,
                    s2,
                    None,
                    None,
                    tw,
                    ti,
                    experts_start=rank * 64,
                    topk=10,
                    activation="silu",
                    use_ep=True,
                    use_sparse_core=True,
                    onehot_moe_permute_threshold=32768,
                )
            # Serving's DP reduce-scatter followed by TP all-reduce: each TP
            # member owns a different half of the experts and shared MLP.
            routed = jax.lax.psum_scatter(routed, "dp", scatter_dimension=0, tiled=True)
            sh = shared(x, sw1, sw2, ss1, ss2, sg)
            return jax.lax.psum((routed + sh).astype(jnp.bfloat16), "tp")

        def diagnostics(x, router, bias):
            _, tw, ti = routing(x, router, bias)
            rank = jax.lax.axis_index("dp") * 2 + jax.lax.axis_index("tp")
            counts = jnp.sum(
                jax.nn.one_hot(ti.reshape(-1), 512, dtype=jnp.int32), axis=0
            )
            local = jax.lax.dynamic_slice_in_dim(counts, rank * 64, 64)
            return local[None], tw, ti

        self.forward = jax.jit(
            jax.shard_map(
                run_local,
                mesh=self.mesh,
                in_specs=self.specs,
                out_specs=P("dp"),
                check_vma=False,
            )
        )
        self.reference = jax.jit(
            jax.shard_map(
                lambda *args: run_local(*args, reference=True),
                mesh=self.mesh,
                in_specs=self.specs,
                out_specs=P("dp"),
                check_vma=False,
            )
        )
        self.routing_reference = jax.jit(
            jax.shard_map(
                lambda *args: run_local(*args, routing_reference=True),
                mesh=self.mesh,
                in_specs=self.specs,
                out_specs=P("dp"),
                check_vma=False,
            )
        )
        self.diagnostics = jax.jit(
            jax.shard_map(
                diagnostics,
                mesh=self.mesh,
                in_specs=(P("dp"), P(), P()),
                out_specs=(P(("dp", "tp")), P(), P()),
                check_vma=False,
            )
        )

    def set_pool(self, active_pool):
        assert 1 <= active_pool <= 64
        bias = self.np.where(
            self.np.arange(512) % 64 < active_pool, 0.0, -1000.0
        ).astype("float32")
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        self.inputs = (
            *self.inputs[:-1],
            self.jax.device_put(bias, NamedSharding(self.mesh, P())),
        )

    def check(self, directory):
        jax, np = self.jax, self.np
        print(
            "Compiling/running baseline and independent dense routing reference",
            flush=True,
        )
        got = jax.block_until_ready(self.forward(*self.inputs))
        want = jax.block_until_ready(self.reference(*self.inputs))
        a, b = np.asarray(got).astype("float32"), np.asarray(want).astype("float32")
        assert a.shape == (256, 4096) and np.isfinite(a).all() and np.isfinite(b).all()
        rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
        worst = float(
            np.max(
                np.linalg.norm(a - b, axis=1)
                / np.maximum(np.linalg.norm(b, axis=1), 1e-20)
            )
        )
        # Dense XLA and Pallas differ in BF16 intermediate rounding before
        # dynamic FP8 quantization. Validate routing separately with identical
        # GMM arithmetic and a tighter tolerance instead of hiding that error.
        route_want = np.asarray(
            jax.block_until_ready(self.routing_reference(*self.inputs))
        ).astype("float32")
        route_rel = float(np.linalg.norm(a - route_want) / np.linalg.norm(route_want))
        route_worst = float(
            np.max(
                np.linalg.norm(a - route_want, axis=1)
                / np.maximum(np.linalg.norm(route_want, axis=1), 1e-20)
            )
        )
        print("Oracle errors:", rel, worst, route_rel, route_worst, flush=True)
        assert rel < 0.03 and worst < 0.06, ("dense", rel, worst)
        assert route_rel < 0.008 and route_worst < 0.015, (
            "routing",
            route_rel,
            route_worst,
        )
        shards = {s.device.id: np.asarray(s.data) for s in got.addressable_shards}
        for pair in self.mesh.devices:
            np.testing.assert_array_equal(shards[pair[0].id], shards[pair[1].id])
        counts, tw, ti = jax.block_until_ready(
            self.diagnostics(self.inputs[0], self.inputs[1], self.inputs[-1])
        )
        counts, tw, ti = (
            np.asarray(counts),
            np.asarray(tw).astype("float32"),
            np.asarray(ti),
        )
        assert counts.shape == (8, 64) and counts.sum() == 2560
        assert all(len(set(row)) == 10 for row in ti)
        np.testing.assert_allclose(tw.sum(axis=-1), 1.0, atol=0.01)
        assert np.count_nonzero(counts.sum(axis=1)) == 8
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "fixture.npz",
            hidden=np.asarray(self.inputs[0]).astype("float32"),
            router_weight=np.asarray(self.inputs[1]).astype("float32"),
            router_bias=np.asarray(self.inputs[-1]),
            topk_ids=ti,
            topk_weights=tw,
            expert_counts=counts,
            baseline_output=a,
            reference_output=b,
            routing_reference_output=route_want,
        )
        result = {
            "relative_l2": rel,
            "worst_token_relative_l2": worst,
            "routing_relative_l2": route_rel,
            "routing_worst_token_relative_l2": route_worst,
            "tp_replicas_bitwise_equal": True,
            "route_count": int(counts.sum()),
            "active_experts_per_rank": np.count_nonzero(counts, axis=1).tolist(),
            "routed_rows_per_rank": counts.sum(axis=1).tolist(),
            "output_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        }
        print("Correctness/routing:", json.dumps(result), flush=True)
        return result

    def bench(self, iterations=200, warmup=20):
        jax = self.jax
        for _ in range(warmup):
            jax.block_until_ready(self.forward(*self.inputs))
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            jax.block_until_ready(self.forward(*self.inputs))
            times.append((time.perf_counter_ns() - start) / 1000)
        return {
            "scope": "synchronized host dispatch plus one complete device MoE",
            "iterations": iterations,
            "warmup": warmup,
            "median_us": statistics.median(times),
            "min_us": min(times),
            "p90_us": float(self.np.percentile(times, 90)),
            "samples_us": times,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pools", type=int, nargs="+", choices=range(1, 65), default=[16, 32, 64]
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--moe-impl",
        choices=["standard", "dense_expert"],
        default="standard",
        help="Set the dense-expert token threshold to 0 (standard) or 256.",
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.profile_steps < 0:
        parser.error("iterations must be positive and profile-steps nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be a new or empty directory")
    os.environ["TPU_MOE_DENSE_EXPERT_THRESHOLD"] = (
        "256" if args.moe_impl == "dense_expert" else "0"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.profile_steps:
        configure_environment()
        import torch
        import torch_tpu  # noqa: F401 - registers the matching libtpu profiler plugin.

        # A real allocation initializes the PJRT client; lazy_init alone does
        # not register device tracing and produces an empty capture.
        profiler_anchor = torch.ones((8,), device="tpu")  # noqa: F841 - keep the allocation alive
        torch.tpu.synchronize()
    b = Baseline(args.seed)
    result = {
        "configuration": {
            "global_tokens": 256,
            "dp": 4,
            "tp": 2,
            "ep": 8,
            "hidden": 4096,
            "intermediate": 1024,
            "experts": 512,
            "topk": 10,
            "weight": "FP8",
            "activation": "BF16",
            "seed": args.seed,
            "moe_impl": args.moe_impl,
            "shared_expert": True,
            "input_source": "deterministic synthetic, not captured model activations",
        },
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "profile_steps": args.profile_steps,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ["jax", "jaxlib", "libtpu", "torch", "torch-tpu"]
        },
        "environment": {
            key: os.environ.get(key)
            for key in [
                "TPU_MOE_DENSE_EXPERT_THRESHOLD",
                "USE_MOE_SPARSE_CORE",
                "TPU_MOE_OWNER_OUTPUT_MODE",
                "ONEHOT_MOE_PERMUTE_THRESHOLD",
                "LIBTPU_INIT_ARGS",
            ]
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "src/vllm_torchtpu/layers/core/fused_moe_gmm.py",
                ROOT / "src/vllm_torchtpu/kernels/megablox/moe_dense_expert.py",
            ]
        },
        "jax_version": b.jax.__version__,
        "devices": [str(d) for d in b.jax.devices()],
        "cases": {},
    }
    for pool in args.pools:
        name = f"pool{pool}"
        directory = args.output / name
        print("CASE", name, flush=True)
        b.set_pool(pool)
        checks = b.check(directory)
        timing = b.bench(args.iterations)
        print(
            "TIMING",
            name,
            {k: v for k, v in timing.items() if k != "samples_us"},
            flush=True,
        )
        # Save pre-optimization StableHLO; this libtpu build cannot export
        # optimized HLO with async-update collectives (invalid operand arity).
        (directory / "baseline.stablehlo.txt").write_text(
            b.forward.lower(*b.inputs).as_text()
        )
        result["cases"][name] = {"checks": checks, "host_timing": timing}
        (args.output / "summary.json").write_text(json.dumps(result, indent=2))
        if args.profile_steps:
            print("Capturing device trace", name, flush=True)
            # TorchTPU's bundled profiler matches the installed libtpu ABI;
            # jax.profiler's plugin API is older and segfaults with this build.
            from torch_tpu._internal.profiler import _impl as profiler

            profiler.start_trace(
                str(directory / "trace"),
                profiler.ProfileOptions(host_tracer_level=2, device_tracer_level=1),
            )
            try:
                for step in range(args.profile_steps):
                    with b.jax.profiler.StepTraceAnnotation(
                        "moe_baseline", step_num=step
                    ):
                        b.jax.block_until_ready(b.forward(*b.inputs))
            finally:
                profiler.stop_trace()
    print("DONE", args.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
