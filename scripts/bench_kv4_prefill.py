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
"""Microbench: where one DCP rank's MSA prefill time goes, fp8 vs 4-bit caches.

Per layer and prefill chunk (one request, 100-150K tokens of context), on
one TensorCore: "write" = `msa_write_dcp` alone (rotate, quantize, cache
writes), "score" = `msa_write_and_candidates_dcp` (writes + indexer + local
top-k), "attend" = `msa_attend_dcp` (merge + attention). L chained layers per
program, per-layer cost ``(t_L - t_1) / (L - 1)`` (`bench_kv4_msa`). With
`--xprof` each part's L-layer program is traced once and its device ops are
summed by name (per layer).

The served MSA env is a preset (`--preset`), applied before any import, plus
`--env A=1,B=2`; one process per preset (the kernels read env at import).

    PYTHONPATH=.:src python scripts/bench_kv4_prefill.py --preset c16 --kv fp8,int4
"""

import argparse
import glob
import os
import sys
import tempfile

PRESETS = {
    # c16 reference (cmd_c16.txt) + launch_1120 _PFPORT + _GAP's MSA levers.
    "c16": {
        "TPU_TIGHT_VMEM": "1",
        "TPU_MINIMAX_M3_MSA_DECODE_WORK_LIST_KERNEL": "1",
        "TPU_MINIMAX_M3_INDEXER_LARGE_PAGES_PER_STEP": "64",
        "TPU_MINIMAX_M3_INDEXER_DECODE_OPT": "noclamp+defer",
        "TPU_MINIMAX_M3_INDEX_FOLD_OPT": "gather+compose",
        "TPU_MINIMAX_M3_MSA_PREFILL_BLOCK_Q": "32",
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGES_PER_STEP": "32",
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE": "512",
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_TILE_PAGES": "64",
        "TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT": "packed",
        "TPU_MINIMAX_M3_MSA_PREFILL_WIDTHS": "2/4/6/8/12/16/20/24",
        "TPU_MINIMAX_M3_MSA_MAX_FREE": "1",
        "TPU_MINIMAX_M3_MSA_MXU_MASK": "1",
    },
    # c64 reference (cmd_c64.txt) without its lossy qk8 (bench_kv4_msa's env).
    "c64": {
        "TPU_TIGHT_VMEM": "1",
        "TPU_MINIMAX_M3_MSA_DECODE_WORK_LIST_KERNEL": "1",
        "TPU_MINIMAX_M3_MSA_DECODE_ONE_CALL": "1",
        "TPU_MINIMAX_M3_INDEXER_LARGE_PAGES_PER_STEP": "64",
        "TPU_MINIMAX_M3_INDEXER_DECODE_OPT": "noclamp+defer",
        "TPU_MINIMAX_M3_INDEX_FOLD_OPT": "gather+compose",
        "TPU_MINIMAX_M3_MSA_PREFILL_BLOCK_Q": "32",
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGES_PER_STEP": "16",
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE": "512",
        "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_TILE_PAGES": "192",
        "TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT": "packed",
        "TPU_MINIMAX_M3_MSA_PREFILL_WIDTHS": "2/4/6/8/12",
    },
}
# The c16 int4 config of launch_1120 --kv4 (c48's 4-bit prefill tiles).
PRESETS["c16kv4"] = dict(
    PRESETS["c16"],
    TPU_MINIMAX_M3_MSA_PREFILL_BLOCK_Q="16",
    TPU_MINIMAX_M3_MSA_PREFILL_ADAPTIVE="30:8:64:48:2.4.6.8.12.16.20.24.32.40",
)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="c16", choices=sorted(PRESETS))
    p.add_argument("--env", default="", help="extra A=1,B=2 (after the preset)")
    p.add_argument("--label", default="")
    p.add_argument("--tokens", default="4096,8192")
    p.add_argument("--dcp", default="2")
    p.add_argument("--kv", default="fp8,int4")
    p.add_argument("--parts", default="write,score,attend")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--ctx", default="100000:150000")
    p.add_argument("--large-bq", type=int, default=0, help="indexer large-class block_q")
    p.add_argument("--xprof", default="", help="trace dir (per-op sums)")
    p.add_argument(
        "--fix-env",
        default="TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q=1024",
        help="env of --kv int4fix (int4's caches, set while tracing)",
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


ARGS = _parse()
for _k, _v in PRESETS[ARGS.preset].items():
    os.environ[_k] = _v
for _item in filter(None, ARGS.env.split(",")):
    _k, _v = _item.split("=", 1)
    os.environ[_k] = _v

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_kv4_msa as base  # noqa: E402

from vllm_torchtpu.models.minimax_m3.kernels import (  # noqa: E402
    msa_attention_lse,
    msa_dcp,
    msa_fp4,
    msa_indexer,
)


def op_sums(trace_dir: str, layers: int) -> list[tuple[str, float]]:
    """Device op time (us per layer) by op name, from the xplane trace."""
    from vllm_torchtpu.tools import xplane_pb2

    files = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True)
    sums: dict[str, float] = {}
    for path in files:
        space = xplane_pb2.XSpace()
        with open(path, "rb") as f:
            space.ParseFromString(f.read())
        for plane in space.planes:
            if not plane.name.startswith("/device:TPU:0"):
                continue
            for line in plane.lines:
                if line.name != "XLA Ops":
                    continue
                for ev in line.events:
                    name = plane.event_metadata[ev.metadata_id].name
                    # "%op.17.x = shape ..." -> "op" (one entry per layer's op)
                    name = name.split(" = ")[0].lstrip("%").split(".")[0]
                    sums[name] = sums.get(name, 0.0) + ev.duration_ps / 1e6 / layers
    return sorted(sums.items(), key=lambda kv: -kv[1])


def write_program(kv, dcp, g, num_layers, interpret):
    fp4 = kv != "fp8"

    def write(k, v, ik, kv_c, idx, sc, seq_lens, tables, cu):
        link = None
        for _ in range(num_layers):
            if link is not None:
                k = k + link
            out = msa_dcp.msa_write_dcp(
                k, v, ik, kv_c, idx, seq_lens, tables, cu, rank=0, dcp=dcp,
                scale_cache=sc if fp4 else None, interpret=interpret,
            )  # fmt: skip
            kv_c, idx = out[0], out[1]
            if fp4:
                sc = out[2]
            link = (idx[0, 0, 0, 0] * 0).astype(k.dtype)
        return kv_c, idx, sc

    return jax.jit(write, donate_argnums=(3, 4, 5))


def main():
    args = ARGS
    if args.smoke:
        args.layers, args.iters, args.ctx, args.tokens = 2, 1, "3000:5000", "1024"
    interpret = args.smoke
    if args.large_bq:
        msa_indexer.DEFAULT_LARGE_BLOCK_Q = args.large_bq
    if os.environ.get("TPU_MINIMAX_M3_MSA_MAX_FREE") == "1":
        # Random q / k: every score is far below the bound.
        msa_attention_lse.configure_max_free(
            10.0, mxu_mask=os.environ.get("TPU_MINIMAX_M3_MSA_MXU_MASK") == "1"
        )
    label = args.label or args.preset
    ctx = tuple(int(x) for x in args.ctx.split(":"))
    parts = args.parts.split(",")
    print(f"devices {jax.devices()[:1]}; preset {args.preset} env {args.env!r} "
          f"large_bq {msa_indexer.DEFAULT_LARGE_BLOCK_Q}", flush=True)  # fmt: skip
    for dcp in (int(x) for x in args.dcp.split(",")):
        g = max(1, dcp // 2)
        for T in (int(x) for x in args.tokens.split(",")):
            rng = np.random.default_rng(1000 * dcp + T)
            meta, dist, num_pages, cands = base.build_step(T, dcp, g, ctx, rng)

            def bf16(*shape):
                return jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)

            k, v, iq = bf16(T, g, base.D), bf16(T, g, base.D), bf16(T, g, base.D)
            ik = bf16(T, base.D)
            q = bf16(dcp * base.HG, T, base.D)
            outs = {}
            for kv_name in args.kv.split(","):
                fix = kv_name == "int4fix"
                kv = "int4" if fix else kv_name
                for item in filter(None, args.fix_env.split(",")):
                    key, val = item.split("=", 1)
                    if fix:
                        os.environ[key] = val
                    else:
                        os.environ.pop(key, None)
                caches = base.build_caches(kv, num_pages, g, np.random.default_rng(7 + T))
                qq = msa_fp4.rotate_queries(q) if kv != "fp8" else q
                res = {}
                for L in (1, args.layers):
                    score, attend = base.programs(kv, dcp, g, L, interpret)
                    write = write_program(kv, dcp, g, L, interpret)

                    def cache_copy(caches=caches):
                        return tuple(None if c is None else jnp.copy(c) for c in caches)

                    def score_args():
                        kv_c, idx, sc = cache_copy()
                        return (k, v, iq, ik, kv_c, idx, sc, *meta, dist)

                    def write_args():
                        kv_c, idx, sc = cache_copy()
                        return (k, v, ik, kv_c, idx, sc, *meta)

                    def attend_args():
                        return (qq, caches[0], caches[2], cands, *meta)

                    progs = {
                        "write": (write, write_args),
                        "score": (score, score_args),
                        "attend": (attend, attend_args),
                    }
                    for part in parts:
                        fn, mk = progs[part]
                        res[(part, L)] = base.timed(fn, mk, args.iters)
                        if L == args.layers and args.xprof:
                            d = tempfile.mkdtemp(dir=args.xprof)
                            a = mk()
                            jax.block_until_ready(a)
                            with jax.profiler.trace(d):
                                jax.block_until_ready(fn(*a))
                            top = op_sums(d, args.layers)
                            total = sum(t for _, t in top)
                            print(f"OPS {label} dcp{dcp} T{T} {kv_name} {part}: "
                                  f"device {total:.1f} us/layer", flush=True)  # fmt: skip
                            for name, t in top[:14]:
                                print(f"OPS   {t:8.1f}  {name[:110]}", flush=True)
                    if L == args.layers and "attend" in parts:
                        out = np.asarray(attend(*attend_args())[0].astype(jnp.float32))
                        outs[(kv_name, "attend")] = out
                        print(f"CHK {label} dcp{dcp} T{T} {kv_name} attend sum "
                              f"{float(np.nansum(out)):.6e} finite {bool(np.isfinite(out).all())}",
                              flush=True)  # fmt: skip
                    if L == args.layers and "score" in parts:
                        sout = score(*score_args())
                        outs[(kv_name, "score")] = [np.asarray(x) for x in sout]
                n = max(args.layers - 1, 1)
                line = " ".join(
                    f"{part} {(res[(part, args.layers)] - res[(part, 1)]) / n * 1e3:8.1f}"
                    for part in parts
                )
                sa = sum(
                    (res[(p, args.layers)] - res[(p, 1)]) / n * 1e3
                    for p in ("score", "attend")
                    if p in parts
                )
                print(f"RES {label:14s} dcp{dcp} T{T:5d} {kv_name:7s} {line}  score+attend {sa:8.1f} us",
                      flush=True)  # fmt: skip
                del caches
            for part in ("score", "attend"):
                a, b = outs.get(("int4", part)), outs.get(("int4fix", part))
                if a is not None and b is not None:
                    a = a if isinstance(a, list) else [a]
                    b = b if isinstance(b, list) else [b]
                    same = all(x.tobytes() == y.tobytes() for x, y in zip(a, b))
                    print(f"BITWISE {label} dcp{dcp} T{T} int4 vs int4fix {part}: {same}",
                          flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
