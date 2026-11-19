# SPDX-License-Identifier: Apache-2.0
"""Device time per call of the fused decode HC kernels, served against cmp.

For each row count, kernel (``served``: hc_decode.py at TILE_ROWS 512, ``cmp``:
hc_decode_cmp.py) and variant it runs a chain of calls inside one jit, each
call's outputs feeding the next as in the decoder (combined streams -> next
residual, mixed -> next block, injection -> next gate), captures an xprof
trace of a few chains, and reads every kernel op's duration from the device's
"XLA Ops" line. Loop carries and copies around the kernel are not counted.

Variants: ``combine`` (combine + mix with the injection gate: 95 calls a
served step), ``mix`` (mix with the injection gate, no combine) and
``final`` (combine + mix without the injection gate: the final mixer).
For ``combine`` it also times the cmp kernel's copies alone (``cmp-dma``) and
its compute alone (``cmp-compute``).

usage: python tools/hc_cmp/bench.py [--tokens 16,32,...] [--variants combine,mix,final]
Prints one ``HCCMP_TIME`` line per case and writes JSON to
$CDK_OUTPUT_DIR/hc-cmp/bench.json when that is set.
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

from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_cmp import (  # noqa: E402
    cmp_applies,
    hc_decode_mix_cmp,
)

BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN
# cmp kernel variants by name: --opts "name:lookahead,overlap,up_whole;..."
OPTS = {"cmp": None}


def _parse_opts(text):
    for item in [x for x in text.split(";") if x]:
        name, vals = item.split(":")
        la, ov, uw = vals.split(",")
        OPTS[name] = (("lookahead", int(la)), ("overlap", ov == "1"), ("up_whole", uw == "1"))


def _chain(kernel, variant, iters, interpret=False):
    static = dict(branches=BRANCHES, hidden=HIDDEN, eps=EPS, interpret=interpret)
    if kernel == "served":
        fn = lambda *a, **k: hc_decode_mix(*a, max_tile=512, **k)  # noqa: E731
    else:
        name, _, mode = kernel.partition("-")
        opts = OPTS[name]
        fn = lambda *a, **k: hc_decode_mix_cmp(  # noqa: E731
            *a, _mode=mode or "full", _opts=opts, **k
        )

    def chain(weights, *carry):
        w = weights if variant != "final" else weights[:3]

        def step(_, carry):
            residual, block, injection = carry
            if variant == "mix":
                mixed, inj = fn(residual, *w, **static)
                return residual, mixed, inj
            mixed, inj, combined = fn(
                residual, *w, block=block, prev_injection=injection, **static
            )
            return combined, mixed, injection if inj is None else inj

        return jax.lax.fori_loop(0, iters, step, carry)

    return jax.jit(chain)


def _kernel_durations(trace_dir):
    """Durations (us) of the hc kernel ops on the first TPU's "XLA Ops" line."""
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
                if "qwen4exp_hc_decode" not in name:
                    continue
                key = "cmp" if "_cmp" in name else "served"
                out.setdefault(key, []).append(dur / 1e6)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", default="16,32,64,96,112,128,160,192,208,224,240,256"
    )
    parser.add_argument("--variants", default="combine,mix,final")
    # Row counts of the variants other than combine.
    parser.add_argument("--side-tokens", default="16,64,128,256")
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--opts", default="")
    # Variant names timed with mode=compute as well (combine only).
    parser.add_argument("--compute-of", default="cmp")
    parser.add_argument("--dma-of", default="cmp")
    # CPU dry run of the script (no device times).
    parser.add_argument("--interpret", action="store_true")
    opts = parser.parse_args()
    _parse_opts(opts.opts)
    keys = jax.random.split(jax.random.key(0), 8)
    bf16 = jnp.bfloat16
    weights = [
        (0.3 * jax.random.normal(keys[3], (WIDTH,))).astype(bf16),
        (0.02 * jax.random.normal(keys[4], (LOWRANK, WIDTH))).astype(bf16),
        (0.05 * jax.random.normal(keys[5], (WIDTH, LOWRANK))).astype(bf16),
        (0.02 * jax.random.normal(keys[6], (BRANCHES, WIDTH))).astype(bf16),
    ]
    rows = []
    print(f"HCCMP device {jax.devices()[0].device_kind}, {opts.iters} chained calls x {opts.chains}")
    for tokens in [int(t) for t in opts.tokens.split(",")]:
        residual = jax.random.normal(keys[0], (tokens, WIDTH)).astype(bf16)
        block = jax.random.normal(keys[1], (tokens, HIDDEN)).astype(bf16)
        gate = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[2], (tokens, BRANCHES)))
        args = (residual, block, gate.astype(jnp.float32))
        for variant in opts.variants.split(","):
            if variant != "combine" and str(tokens) not in opts.side_tokens.split(","):
                continue
            result = dict(tokens=tokens, variant=variant)
            kernels = ["served"] + list(OPTS)
            if variant == "combine":
                kernels += [f"{n}-compute" for n in opts.compute_of.split(",") if n]
                kernels += [f"{n}-dma" for n in opts.dma_of.split(",") if n]
            for kernel in kernels:
                if kernel != "served" and not cmp_applies(tokens):
                    continue
                fn = _chain(kernel, variant, opts.iters, opts.interpret)
                jax.block_until_ready(fn(weights, *args))
                trace_dir = tempfile.mkdtemp(prefix="hccmp-")
                jax.profiler.start_trace(trace_dir)
                for _ in range(opts.chains):
                    jax.block_until_ready(fn(weights, *args))
                jax.profiler.stop_trace()
                key = "served" if kernel == "served" else "cmp"
                durs = _kernel_durations(trace_dir).get(key, [])
                shutil.rmtree(trace_dir, ignore_errors=True)
                if durs:
                    result[kernel] = dict(
                        n=len(durs),
                        median=statistics.median(durs),
                        min=min(durs),
                        max=max(durs),
                    )
            rows.append(result)
            parts = [
                f"{k} {result[k]['median']:7.2f} us (n {result[k]['n']}, "
                f"{result[k]['min']:.2f}-{result[k]['max']:.2f})"
                for k in kernels
                if k in result
            ]
            ratio = ""
            if "served" in result:
                ratio = " " + " ".join(
                    f"{k}/served {result[k]['median'] / result['served']['median']:.3f}"
                    for k in OPTS
                    if k in result
                )
            print(f"HCCMP_TIME tokens {tokens:3d} {variant:7s} " + " | ".join(parts) + ratio, flush=True)
    out = os.environ.get("CDK_OUTPUT_DIR")
    if out:
        os.makedirs(os.path.join(out, "hc-cmp"), exist_ok=True)
        with open(os.path.join(out, "hc-cmp", "bench.json"), "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
