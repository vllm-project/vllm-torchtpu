# SPDX-License-Identifier: Apache-2.0
"""On-TPU parity and per-op device time of VLLM_TPU_MK_IO_LEAN=pleconv (`short_conv`).

    python tests/models/vllm/tpu_bench_ple_conv.py --buckets 4 8 16 32 --reqs 16

The PLE conv as the served op runs it (request-owned history, the history-fast path):
Qwen3.8-Flash geometry, 4 HC streams x 2560 channels, dilation 3, kernel 4. Per bucket,
lever off vs on: bit-identical outputs, and the device op time per call from a jax
profile (`OPSUM`).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np

C, DIL, K = 4 * 2560, 3, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", type=int, nargs="+", default=[4, 8, 16, 24, 32, 48, 64])
    ap.add_argument("--reqs", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=os.environ.get("CDK_OUTPUT_DIR", "/tmp"))
    a = ap.parse_args()
    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv

    envs.VLLM_TPU_PLE_CONV_HISTORY_FAST = True
    state_len = (K - 1) * DIL
    gen = np.random.default_rng(0)
    summary = {}
    for B in a.buckets:
        R = a.reqs
        active = min(R, B)
        cuts = np.sort(gen.choice(np.arange(1, B), active - 1, replace=False)) if active > 1 else []
        qsl = np.full(R + 1, B, np.int32)
        qsl[0] = 0
        qsl[1:active] = cuts
        x = jnp.asarray(gen.standard_normal((B, C)), jnp.bfloat16)
        w = jnp.asarray(gen.standard_normal((C, K)), jnp.bfloat16)
        hist = jnp.asarray(gen.standard_normal((R, state_len, C)), jnp.bfloat16)
        slots = np.where(np.arange(R) < active, np.arange(R), -1).astype(np.int32)
        init = jnp.asarray(gen.integers(0, 2, R).astype(bool))
        args = (x, w, hist, jnp.asarray(qsl), jnp.asarray(slots), init)
        res, outs = {}, {}
        for name, lean in (("off", ""), ("pleconv", "pleconv")):
            envs.VLLM_TPU_MK_IO_LEAN = lean
            f = jax.jit(
                lambda *z: short_conv(*z, dilation=DIL, request_owned_history=True)
            )
            outs[name] = jax.block_until_ready(f(*args))
            d = f"/tmp/prof_conv_{name}_{B}"
            shutil.rmtree(d, ignore_errors=True)
            with jax.profiler.trace(d):
                for _ in range(a.iters):
                    o = f(*args)
                jax.block_until_ready(o)
            from jax.profiler import ProfileData

            ops = collections.Counter()
            fpb = sorted(glob.glob(d + "/**/*.xplane.pb", recursive=True))[-1]
            for plane in ProfileData.from_file(fpb).planes:
                if plane.name != "/device:TPU:0":
                    continue
                for line in plane.lines:
                    if line.name == "XLA Ops":
                        for ev in line.events:
                            ops[ev.name.split(" ")[0]] += ev.duration_ns / a.iters / 1e3
            res[f"{name}_ops_us"] = round(sum(ops.values()), 2)
            res[f"{name}_n_ops"] = len(ops)
            with open(f"{a.out}/convops_{name}_b{B}.txt", "w") as fo:
                for k, v in ops.most_common():
                    fo.write(f"OPS XLA_Ops {v:.3f} {k}\n")
        res["bitwise"] = all(
            np.array_equal(np.asarray(p).view(np.uint16), np.asarray(q).view(np.uint16))
            for p, q in zip(outs["off"], outs["pleconv"], strict=True)
        )
        summary[B] = res
        print("RESULT", B, json.dumps(res), flush=True)
    print("SUMMARY", json.dumps(summary), flush=True)
    assert all(r["bitwise"] for r in summary.values()), summary


if __name__ == "__main__":
    main()
