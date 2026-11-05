# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Decode MoE paths at 64-128 rows on real TPU cores (every local device as one
TP mesh), at GLM-5.2's TP16 per-rank shapes, through the adapter's op body:

  kernel   the fused TP decode kernel (moe_tp_decode), at row counts it admits;
  prefill  the fused MoE prefill kernel (FUSED_MOE_PREFILL_KERNEL, stage A), as
           FUSED_MOE_DECODE_KERNEL_MAX_TOKENS below the row count selects it;
  psum     the usual TP path (fused_moe_func, the shared expert) and a psum.

Each path's output is checked against the f32 reference of
test_moe_tp_decode_kernel, and its device time is measured over a chain of
LAYERS calls in one program. Measurement only: prints one RESULT line.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# The decode leg's MoE settings in the GLM-5.2 P/D recipe.
_ENV = {
    "FUSED_MOE_PREFILL_KERNEL": "1",
    "FUSED_MOE_PREFILL_KERNEL_V2": "1",
    "TPU_MOE_SKIP_PADDED_TOKENS": "1",
    "USE_MOE_SPARSE_CORE": "1",
    "ONEHOT_MOE_PERMUTE_THRESHOLD": "2048",
    "TPU_MOE_ONEHOT_SORTFREE": "1",
    "TPU_MOE_ROUTER_TOPK_BIAS": "1",
    "TPU_MOE_TP_DECODE_FAST": "1",
}
ROWS = (64, 80, 96, 128)
LAYERS = 32
REPS = 10
TOLERANCE = dict(rel_l2=0.06, cos=0.998)


def _bench():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    here = Path(__file__).resolve()
    # The mapped tree's source, whatever the image installed.
    sys.path.insert(0, str(here.parents[3] / "src"))
    sys.path.insert(0, str(here.parent))
    import test_moe_tp_decode_kernel as base

    import vllm_torchtpu
    from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as kimpl
    from vllm_torchtpu.layers.adapter import moe_tp_decode as adapter

    devices = jax.devices()
    n = len(devices)
    c = dict(base.GLM52_TP16)
    mesh = Mesh(np.asarray(devices), ("d",))
    keys = jax.random.split(jax.random.key(1), n + 3)
    ranks = [base._rank_weights(keys[r], c) for r in range(n)]
    stacked = [jnp.concatenate(ws, axis=0) for ws in zip(*ranks, strict=True)]
    gate = jax.random.normal(keys[-1], (c["e"], c["h"])) / np.sqrt(c["h"])
    bias = jax.random.uniform(keys[-2], (c["e"],), jnp.float32, -0.05, 0.05)
    rank_ids = jnp.arange(n, dtype=jnp.int32).reshape(n, 1)
    e0 = jnp.zeros((n, 1), jnp.int32)
    specs = (P("d"), P("d"), P(), P()) + (P("d"),) * 10
    weights = [jax.device_put(w, NamedSharding(mesh, P("d"))) for w in stacked]
    base_buckets = tuple(kimpl.ROW_BUCKETS)
    result = dict(
        devices=n,
        device_kind=devices[0].device_kind,
        jax=jax.__version__,
        vllm_torchtpu=vllm_torchtpu.__file__,
        row_buckets=list(base_buckets),
        vmem_capacity_bytes=int(kimpl.pltpu.get_tpu_info().vmem_capacity_bytes),
        layers=LAYERS,
        reps=REPS,
        cases={},
    )

    def statics(prefill_above):
        return adapter._OpStatics(
            topk=c["topk"],
            routed_scale=base.ROUTED_SCALE,
            activation="silu",
            use_sparse_core=True,
            onehot_threshold=2048,
            use_ep=False,
            router_bias=True,
            prefill_above=prefill_above,
            fast=True,
            hbm_operands=False,
        )

    def operands(rows, n_valid):
        x = jax.random.normal(keys[-3], (rows, c["h"])).astype(jnp.bfloat16)
        scores = jax.nn.sigmoid(x.astype(jnp.float32) @ gate.T)
        pad = jnp.arange(max(ROWS)) >= n_valid
        put = [
            jax.device_put(a, NamedSharding(mesh, s))
            for a, s in zip(
                (jnp.tile(x, (n, 1)), jnp.tile(scores, (n, 1)), bias, pad),
                specs[:4],
                strict=True,
            )
        ]
        tail = [
            jax.device_put(a, NamedSharding(mesh, P("d"))) for a in (rank_ids, e0)
        ]
        return x, scores, (*put, *weights, *tail)

    def chained(body):
        def run(x, scores, b, pad, *rest):
            def step(carry, _):
                y = body(carry, scores, b, pad, *rest).astype(jnp.float32)
                y = y * lax.rsqrt(jnp.mean(y * y) + 1e-6)
                return y.astype(jnp.bfloat16), None

            out, _ = lax.scan(step, x, None, length=LAYERS)
            return out

        return run

    def sharded(fn):
        return jax.jit(
            jax.shard_map(
                fn, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
            )
        )

    def identity(x, scores, b, pad, *rest):
        return x

    def admitted(rows):
        sds = jax.ShapeDtypeStruct
        local = [sds((w.shape[0] // n, *w.shape[1:]), w.dtype) for w in stacked]
        return kimpl.can_use_moe_tp_decode(
            sds((rows, c["h"]), jnp.bfloat16),
            sds((rows, c["e"]), jnp.float32),
            sds((c["e"],), jnp.float32),
            *local,
            topk=c["topk"],
            n_ranks=n,
        )

    def time_chain(fn, ops):
        start = time.monotonic()
        out = fn(*ops)
        out.block_until_ready()
        compile_s = time.monotonic() - start
        fn(*ops).block_until_ready()
        start = time.perf_counter()
        for _ in range(REPS):
            out = fn(*ops)
        out.block_until_ready()
        per_layer_us = (time.perf_counter() - start) / REPS / LAYERS * 1e6
        return round(per_layer_us, 2), round(compile_s, 1), bool(
            np.isfinite(np.asarray(out, np.float32)).all()
        )

    def check(out, x, scores, rows, n_valid, path):
        out = np.asarray(out, np.float32).reshape(n, rows, c["h"])
        bits = out.view(np.uint32)
        with jax.default_matmul_precision("highest"):
            want = base._reference(x, scores, bias, n_valid, ranks, c).astype(np.float64)
        got = out[0, :n_valid].astype(np.float64)
        rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        cos = float(np.vdot(got, want) / (np.linalg.norm(got) * np.linalg.norm(want)))
        case = dict(
            rel_l2=round(rel, 5),
            cos=round(cos, 6),
            ranks_identical=bool((bits == bits[0]).all()),
            nan=int(np.isnan(out).sum()),
        )
        if path == "kernel":
            pad = out[:, n_valid:]
            case["padding_rows_positive_zero"] = bool(
                np.all((pad == 0) & ~np.signbit(pad))
            )
        return case

    # Overhead of the chain itself (scan, normalization), to subtract.
    _, _, ops = operands(64, 64)
    result["chain_overhead_us"] = time_chain(sharded(chained(identity)), ops)[0]

    for rows in ROWS:
        others = tuple(b for b in base_buckets if b != rows)
        paths = {
            # ROW_BUCKETS for the trace, prefill_above.
            "kernel": (tuple(sorted({*base_buckets, rows})), None),
            "prefill": (others, rows - 1),
            "psum": (others, None),
        }
        for path, (buckets, prefill_above) in paths.items():
            key = f"{rows}:{path}"
            kimpl.ROW_BUCKETS = buckets
            try:
                if path == "kernel" and not admitted(rows):
                    result["cases"][key] = dict(admitted=False)
                    continue
                body = adapter._op_body(statics(prefill_above), n, "d")
                x, scores, ops = operands(rows, rows - 3)
                case = check(sharded(body)(*ops), x, scores, rows, rows - 3, path)
                _, _, ops = operands(rows, rows)
                us, compile_s, finite = time_chain(sharded(chained(body)), ops)
                case.update(per_layer_us=us, compile_s=compile_s, finite=finite)
                result["cases"][key] = case
            except Exception as e:  # report and go on with the other paths
                result["cases"][key] = dict(error=f"{type(e).__name__}: {e}"[:600])
            finally:
                kimpl.ROW_BUCKETS = base_buckets
            print(key, json.dumps(result["cases"][key]), flush=True)
    print("RESULT " + json.dumps(result), flush=True)


def test_decode_moe_paths_by_row_count():
    proc = subprocess.run(
        [sys.executable, __file__, "bench"],
        env=dict(os.environ, **_ENV),
        capture_output=True,
        text=True,
        timeout=2400,
    )
    print(proc.stdout[-20000:])
    print(proc.stderr[-6000:])
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    if proc.returncode != 0 and "needs a TPU host" in proc.stdout:
        pytest.skip("needs a TPU host")
    assert proc.returncode == 0 and lines, "benchmark process failed"
    result = json.loads(lines[-1][len("RESULT ") :])
    for key, case in sorted(result["cases"].items()):
        if "rel_l2" in case:
            assert case["rel_l2"] < TOLERANCE["rel_l2"], (key, case)
            assert case["cos"] > TOLERANCE["cos"], (key, case)
            assert case["ranks_identical"] and not case["nan"], (key, case)
        if case.get("padding_rows_positive_zero") is False:
            raise AssertionError((key, case))


if __name__ == "__main__" and sys.argv[1:] == ["bench"]:
    import jax

    if jax.default_backend() != "tpu" or len(jax.devices()) not in (8, 16):
        print("needs a TPU host (8 or 16 cores)")
        sys.exit(3)
    _bench()
