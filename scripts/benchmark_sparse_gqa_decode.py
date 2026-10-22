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
"""Matched-input QSA decode A/B including metadata, padding and output gating.

Both arms own and donate identical native caches. Compilation, allocation,
index selection and cache updates are outside the measured attention boundary.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import qsa_attention, rpa_attention


def provenance(devices):
    """Record loaded source and device identity outside the timing boundary."""
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.PIPE
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    sources = {}
    for name, module in sorted(sys.modules.items()):
        if name.startswith("vllm_torchtpu.kernels.") and getattr(
            module, "__file__", None
        ):
            path = Path(module.__file__).resolve()
            sources[name] = dict(
                path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
    return dict(
        argv=sys.argv,
        git_head=git("rev-parse", "HEAD"),
        git_status=git("status", "--porcelain"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        loaded_sources=sources,
        devices=[
            dict(
                id=d.id,
                kind=d.device_kind,
                coords=getattr(d, "coords", None),
                core_on_chip=getattr(d, "core_on_chip", None),
                process_index=d.process_index,
            )
            for d in devices
        ],
        jax_config={
            name: str(jax.config.values[name])
            for name in (
                "jax_enable_x64",
                "jax_platforms",
                "jax_default_matmul_precision",
            )
        },
        warmups_per_arm=3,
        timing_boundary="host perf_counter_ns before call through block_until_ready",
    )


def make_case(
    batch: int,
    context: int,
    page: int,
    fp8: bool,
    heads: int,
    kv_heads: int,
    dim: int,
    ratio: int,
    groups: int,
):
    rng = np.random.default_rng(721 + batch + context)
    columns = (context + page - 1) // page
    table = rng.permutation(batch * columns).reshape(batch, columns).astype(np.int32)
    lengths = context - np.arange(batch, dtype=np.int32) % 4
    blocks = np.full((batch, groups), -1, np.int32)
    for row, length in enumerate(lengths):
        n = min(length // ratio, groups)
        blocks[row, :n] = rng.choice(length // ratio, n, replace=False)
    q = jnp.asarray(rng.normal(0, 0.5, (batch, heads, dim)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 3, q.shape), jnp.bfloat16)
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    packing = 4 if fp8 else 2
    # Generate on device to avoid a many-GB temporary host FP64 allocation.
    cache = jax.random.uniform(
        jax.random.key(721),
        (batch * columns, page, math.ceil(2 * kv_heads / packing), packing, dim),
        minval=-0.5,
        maxval=0.5,
    ).astype(dtype)
    return (
        q,
        gate,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table),
        cache,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192])
    parser.add_argument("--page-size", type=int, default=1664)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--groups", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    report = dict(
        schema_version=2,
        device=str(devices[0]),
        provenance=provenance(devices),
        versions={
            k: importlib.metadata.version(k) for k in ("jax", "jaxlib", "libtpu")
        },
        cases=[],
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    for batch in args.batches:
        for context in args.contexts:
            name = f"b{batch}-l{context}"
            print("CASE", name, flush=True)
            directory = args.output / name
            directory.mkdir()
            inputs = make_case(
                batch,
                context,
                args.page_size,
                args.fp8,
                args.heads,
                args.kv_heads,
                args.dim,
                args.ratio,
                args.groups,
            )
            opts = dict(
                kv_heads=args.kv_heads, ratio=args.ratio, k_scale=1.0, v_scale=1.0
            )
            starts = jnp.arange(batch + 1, dtype=jnp.int32)
            funcs = {
                "rpa": functools.partial(rpa_attention, **opts),
                "fused": functools.partial(qsa_attention, starts=starts, **opts),
            }
            compiled, owned, outputs, memory = {}, {}, {}, {}
            for label, fn in funcs.items():
                exe = jax.jit(fn, donate_argnums=(5,)).lower(*inputs).compile()
                hlo = exe.as_text()
                (directory / f"{label}.hlo.txt").write_text(hlo)
                if label == "fused":
                    assert "fused_qsa_decode" in hlo
                stats = exe.memory_analysis()
                memory[label] = dict(
                    temp_bytes=stats.temp_size_in_bytes,
                    argument_bytes=stats.argument_size_in_bytes,
                    output_bytes=stats.output_size_in_bytes,
                    alias_bytes=stats.alias_size_in_bytes,
                )
                compiled[label] = exe
                owned[label] = [*inputs[:-1], inputs[-1].copy()]
                for _ in range(3):
                    outputs[label], cache = jax.block_until_ready(exe(*owned[label]))
                    owned[label][-1] = cache
            compiled_provenance = provenance(devices)
            expected = np.asarray(outputs["rpa"], np.float32)
            actual = np.asarray(outputs["fused"], np.float32)
            assert np.isfinite(actual).all() and np.isfinite(expected).all()
            rel = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
            assert rel < 0.01, rel
            samples = {k: [] for k in funcs}
            for i in range(args.iterations):
                for label in list(funcs) if i % 2 == 0 else list(reversed(funcs)):
                    start = time.perf_counter_ns()
                    _, cache = jax.block_until_ready(compiled[label](*owned[label]))
                    samples[label].append((time.perf_counter_ns() - start) / 1000)
                    owned[label][-1] = cache
            if args.trace:
                with jax.profiler.trace(str(directory / "trace")):
                    for label, exe in compiled.items():
                        with jax.profiler.TraceAnnotation(label):
                            for _ in range(3):
                                _, cache = jax.block_until_ready(exe(*owned[label]))
                                owned[label][-1] = cache
            medians = {k: statistics.median(v) for k, v in samples.items()}
            row = dict(
                name=name,
                batch=batch,
                context=context,
                page_size=args.page_size,
                cache_dtype=str(inputs[-1].dtype),
                heads=args.heads,
                kv_heads=args.kv_heads,
                dim=args.dim,
                ratio=args.ratio,
                groups=args.groups,
                median_us=medians,
                samples_us=samples,
                speedup=medians["rpa"] / medians["fused"],
                relative_l2=rel,
                max_abs=float(np.max(np.abs(actual - expected))),
                memory=memory,
                compiled_provenance=compiled_provenance,
            )
            report["cases"].append(row)
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(row), flush=True)
            del inputs, owned, outputs, compiled, exe, cache
            jax.clear_caches()


if __name__ == "__main__":
    main()
