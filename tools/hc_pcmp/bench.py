# SPDX-License-Identifier: Apache-2.0
"""Device time per call of the prefill HC kernels, served against cmp
(TPU_QWEN4_HC_PREFILL_COMPARE), at prefill chunk rows.

For each row count and kernel it runs a chain of calls inside one jit, each
call's outputs feeding the next as in the decoder (combined streams -> next
residual, mixed -> next block, injection -> next gate), captures an xprof
trace of a few chains and reads every kernel op's duration from the device's
"XLA Ops" line. Kernels:

* ``served_mix``: hc_prefill_fused without prologue (qwen4exp_hc_prefill_inj_m256)
* ``served_combine``: hc_prefill_fused with the combine prologue
  (qwen4exp_hc_prefill_combine_inj_m256)
* ``gated_residual``: the separate combine (gated_residual_b4_h2560_m256)
* ``cmp_mix`` / ``cmp_combine``: hc_prefill_cmp without / with the prologue

A pipeline stage of L layers runs, served: L mixes, L combine+mixes and L
separate combines; with the flag: one mix, 2L - 1 combine+mixes (each layer's
mlp combine folded into the next attn mix) and one separate combine. The
projected stage gain at L = 6 (s10623's stage 6) is printed against the
work order's line (375 us). Loop carries and the weight preparation around
the served kernels (the [down; inject; 0] concatenation and the up weight's
relayout copy, which the cmp kernel does not need: 132.6 us a stage in the
s10623 capture) are not timed; the pass line adds that capture figure to the
timed kernel gain, as the pod's HC rows count it.

usage: python tools/hc_pcmp/bench.py [--tokens 4096,8192] [--layers 6]
       [--kernels cmp_mix,cmp_combine --swap 0 --unroll 0]
Prints one ``P5HC_TIME`` line per case and a ``P5HC_STAGE`` line per row
count; writes JSON to $P5HC_OUT/bench.json when that is set. Exits 3 when the
projected gain at the first row count is under --line.
"""

import argparse
import glob
import json
import os
import shutil
import statistics
import sys
import tempfile

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xspace  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_cmp import hc_prefill_mix_cmp  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_fused import hc_prefill_mix  # noqa: E402

BRANCHES, HIDDEN, LOWRANK, EPS, TILE = 4, 2560, 320, 1e-6, 256
WIDTH = BRANCHES * HIDDEN
KERNELS = ("served_mix", "served_combine", "gated_residual", "cmp_mix", "cmp_combine")
# Op-name markers on the "XLA Ops" line.
MARKS = {
    "served_mix": "qwen4exp_hc_prefill_inj_m",
    "served_combine": "qwen4exp_hc_prefill_combine_inj_m",
    "gated_residual": "gated_residual_b",
    "cmp_mix": "qwen4exp_hc_prefill_cmp_inj_m",
    "cmp_combine": "qwen4exp_hc_prefill_cmp_combine_inj_m",
}
# Weight preparation around the served kernels in the s10623 stage-6 capture
# (cmp-served47/data/xplane/s10623-r6-step.json, group hc): 12 relayout copies
# of the up weight (76.8 us) and the [down; inject; 0] concatenation (55.8).
SERVED_PREP_US = 132.6


def _chain(kernel, tokens, iters, swap, unroll, interpret=False):
    partial = bool(tokens % TILE)
    static = dict(
        branches=BRANCHES, hidden=HIDDEN, eps=EPS, token_tile=TILE, partial=partial, interpret=interpret
    )
    combine = gated_residual_fn(BRANCHES, HIDDEN, TILE, partial, interpret)

    def call(weights, residual, block, injection):
        if kernel == "gated_residual":
            return combine(residual, block, injection), block, injection
        if kernel.startswith("served"):
            fn = lambda *a, **k: hc_prefill_mix(*a, norm_impl="bf16ep64", **k)  # noqa: E731
        else:
            fn = lambda *a, **k: hc_prefill_mix_cmp(*a, swap=swap, unroll=unroll, **k)  # noqa: E731
        if kernel.endswith("_mix"):
            mixed, inj = fn(residual, *weights, **static)
            return residual, mixed, inj
        mixed, inj, combined = fn(
            residual, *weights, block=block, prev_injection=injection, **static
        )
        return combined, mixed, inj

    def chain(weights, *carry):
        return jax.lax.fori_loop(0, iters, lambda _, c: call(weights, *c), carry)

    return jax.jit(chain)


def _durations(trace_dir):
    """Durations (us) of the HC kernel ops on the first TPU's "XLA Ops" line."""
    out = {}
    for path in glob.glob(os.path.join(trace_dir, "plugins", "profile", "*", "*.xplane.pb")):
        space = xspace.load(path)
        for plane in space.planes:
            if plane.name != "/device:TPU:0":
                continue
            ops = plane.line("XLA Ops")
            if ops is None:
                continue
            md = plane.event_metadata
            for _start, dur, mid in ops.timed():
                e = md.get(mid)
                name = (e.name + " " + e.display_name) if e is not None else ""
                # Longest marker first: "_cmp_combine_inj" also holds "_inj_m".
                for key in sorted(MARKS, key=lambda k: -len(MARKS[k])):
                    if MARKS[key] in name:
                        out.setdefault(key, []).append(dur / 1e6)
                        break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="4096,8192")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--chains", type=int, default=3)
    # The work order's line: half the expected 750 us a stage.
    parser.add_argument("--line", type=float, default=375.0)
    parser.add_argument("--swap", type=int, default=1)
    parser.add_argument("--unroll", type=int, default=1)
    # A subset of KERNELS (timings only; the stage projection needs all).
    parser.add_argument("--kernels", default=",".join(KERNELS))
    # CPU dry run of the script (no device times).
    parser.add_argument("--interpret", action="store_true")
    opts = parser.parse_args()
    keys = jax.random.split(jax.random.key(0), 8)
    bf16 = jnp.bfloat16
    weights = (
        (0.3 * jax.random.normal(keys[3], (WIDTH,))).astype(bf16),
        (0.02 * jax.random.normal(keys[4], (LOWRANK, WIDTH))).astype(bf16),
        (0.05 * jax.random.normal(keys[5], (WIDTH, LOWRANK))).astype(bf16),
        (0.02 * jax.random.normal(keys[6], (BRANCHES, WIDTH))).astype(bf16),
    )
    rows, stages = [], []
    print(f"P5HC device {jax.devices()[0].device_kind}, {opts.iters} chained calls x {opts.chains}", flush=True)
    for tokens in [int(t) for t in opts.tokens.split(",")]:
        residual = (4.0 * jax.random.normal(keys[0], (tokens, WIDTH))).astype(bf16)
        block = jax.random.normal(keys[1], (tokens, HIDDEN)).astype(bf16)
        gate = (2.0 * jax.nn.sigmoid(jax.random.normal(keys[2], (tokens, BRANCHES)))).astype(jnp.float32)
        med = {}
        for kernel in opts.kernels.split(","):
            fn = _chain(kernel, tokens, opts.iters, bool(opts.swap), bool(opts.unroll), opts.interpret)
            jax.block_until_ready(fn(weights, residual, block, gate))
            trace_dir = tempfile.mkdtemp(prefix="p5hc-")
            jax.profiler.start_trace(trace_dir)
            for _ in range(opts.chains):
                jax.block_until_ready(fn(weights, residual, block, gate))
            jax.profiler.stop_trace()
            durs = _durations(trace_dir).get(kernel, [])
            shutil.rmtree(trace_dir, ignore_errors=True)
            row = dict(tokens=tokens, kernel=kernel, n=len(durs))
            if durs:
                row.update(median=statistics.median(durs), min=min(durs), max=max(durs))
                med[kernel] = row["median"]
                print(
                    f"P5HC_TIME tokens {tokens} {kernel:15s} {row['median']:8.2f} us "
                    f"(n {len(durs)}, {row['min']:.2f}-{row['max']:.2f})",
                    flush=True,
                )
            else:
                print(f"P5HC_TIME tokens {tokens} {kernel:15s} no ops found", flush=True)
            rows.append(row)
        if all(k in med for k in KERNELS):
            n = opts.layers
            served = n * (med["served_mix"] + med["served_combine"] + med["gated_residual"])
            cmp = med["cmp_mix"] + (2 * n - 1) * med["cmp_combine"] + med["gated_residual"]
            kernel_only = n * (med["cmp_mix"] + med["cmp_combine"] + med["gated_residual"])
            stage = dict(
                tokens=tokens, layers=n, served_us=served, cmp_us=cmp, gain_us=served - cmp,
                gain_kernel_only_us=served - kernel_only, gain_with_prep_us=served - cmp + SERVED_PREP_US,
            )
            stages.append(stage)
            print(
                f"P5HC_STAGE tokens {tokens} layers {n}: served {served:.1f} us "
                f"(mix+combine+gated_residual x {n}), cmp {cmp:.1f} us (1 mix + {2 * n - 1} "
                f"combine + 1 gated_residual): gain {served - cmp:.1f} us, kernel only (no fold) "
                f"{served - kernel_only:.1f}, with the served weight prep {served - cmp + SERVED_PREP_US:.1f}"
                f" (line {opts.line:.0f})",
                flush=True,
            )
    out = os.environ.get("P5HC_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "bench.json"), "w") as f:
            json.dump(dict(calls=rows, stages=stages, line=opts.line), f, indent=1)
    if not stages:
        if opts.kernels != ",".join(KERNELS):
            return 0
        print("P5HC_GAIN FAIL: no stage projection", flush=True)
        return 2
    first = stages[0]
    # The pod's HC rows include the served weight prep, which the cmp path
    # does not run, so the projection that predicts the pod gate counts it.
    ok = first["gain_with_prep_us"] >= opts.line
    print(
        f"P5HC_GAIN {'PASS' if ok else 'FAIL'} tokens {first['tokens']}: gain with the served "
        f"weight prep {first['gain_with_prep_us']:.1f} us a stage against the line "
        f"{opts.line:.0f} (kernels alone {first['gain_us']:.1f})",
        flush=True,
    )
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
