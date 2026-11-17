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
"""Microbenchmark: per-layer MSA time of a DCP step with cross-layer index
reuse (key minimax_msa_index_reuse) at stride 1 / 2 / 4, on all 8
TensorCores at the served DCP degree (notes handover/launch_v5.sh).

Layout as served (TP 8, 4 KV heads, 8 query heads and one index head per
rank): a DCP group of ``dcp`` ranks spans ``G = dcp / 2`` KV heads (1 at
DCP 2); the 8 TCs form ``8 / dcp`` groups (mesh axes ``rep`` x ``dcp``).
Per layer and rank, as `ops.msa.get_msa_dcp_op` runs them:

  Full    all-gather k / v / index_q (G > 1), write + score + candidates
          (`msa_dcp.msa_write_and_candidates_dcp`), query gather, candidate
          all-gather, merge + attend (`msa_attend_dcp(return_selected)`), LSE
          merge + reduce-scatter
  Shared  all-gather k / v (G > 1), write only (`msa_dcp.msa_write_dcp`),
          query gather, attend over the Full layer's selection
          (`msa_attend_selected_dcp`), LSE merge + reduce-scatter

Stride s runs groups of one Full and s - 1 Shared layers in a `fori_loop`
over n groups (each layer its own inputs; caches carried through the
writes). n is a runtime argument, so one compile serves n = 1 and n = N:
per-layer time = ``(t(N) - t(1)) / ((N - 1) * s)``. ``x60`` projects the
60 MSA layers of a step.

Cells ``<config>:<T>@<ctx>``: T = step tokens; ``dec<T>`` = T / 8 decode
requests of 8 rows (1 + 7 MTP) at ctx, else (optional `--decode-reqs`
decode requests at `--decode-ctx` plus) one prefill chunk with ctx tokens
before it.

Usage (TPU host, 8 TCs):
  PYTHONPATH=.:src python scripts/bench_msa_index_reuse.py
  PYTHONPATH=.:src python scripts/bench_msa_index_reuse.py \\
      --cells c48:2048@131072 --strides 1,2,4,8 --json /tmp/reuse.jsonl
CPU smoke (8 host devices, interpret mode, tiny shapes):
  PYTHONPATH=.:src python scripts/bench_msa_index_reuse.py --smoke
"""

from __future__ import annotations

import os
import sys

if "--smoke" in sys.argv:  # before jax is imported
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    )
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import functools
import json
import math
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.minimax_m3 import kernel_toggles
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4, msa_indexer

TP = 8
KV_HEADS = 4
Q_HEADS = 8  # per rank
REPLICAS = TP // KV_HEADS
D = 128
PAGE = 128
TOPK, INIT_BLOCKS, LOCAL_BLOCKS = 16, 0, 1
SM_SCALE = 1.0 / math.sqrt(D)
DECODE_ROWS = 8
NUM_LAYERS = 60

# The MSA keys of the served v5 additional_config (launch_v5.sh c4 / AC48 /
# AC64); c4 runs bf16 caches, c48 / c64 4-bit (int4_per_token_head).
_PREFILL_COMMON = {
    "minimax_msa_tile_select": True,
    "msa_index_int4_fp8": True,
    "minimax_msa_unstable_sort": True,
    "minimax_indexer_dma_ring3": True,
    "minimax_msa_lse_r6": True,
    "minimax_indexer_large_pages_per_step": 64,
    "minimax_indexer_hoist_metadata": True,
    "minimax_indexer_row_gather": True,
    "minimax_msa_prefill_page_cache": 512,
}
CONFIGS = {
    "c4": dict(
        dcp=2,
        max_seqs=8,
        fp4=False,
        additional={
            "minimax_msa_tile_select": True,
            "minimax_dcp_fine_scan_widths": True,
            "minimax_indexer_fp8_transposed": True,
            "minimax_indexer_large_pages_per_step": 64,
            "minimax_indexer_hoist_metadata": True,
            "minimax_indexer_row_gather": True,
            "tight_vmem": True,
        },
    ),
    "c48": dict(
        dcp=4,
        max_seqs=48,
        fp4=True,
        additional={
            **_PREFILL_COMMON,
            "minimax_msa_live_select": True,
            "minimax_msa_prefill_page_cache_tile_pages": 64,
            "minimax_msa_prefill_widths": "4/8/12",
        },
    ),
    "c64": dict(
        dcp=8,
        max_seqs=64,
        fp4=True,
        additional={
            **_PREFILL_COMMON,
            "minimax_msa_decode_work_list_kernel": True,
            "minimax_msa_decode_one_call": True,
        },
    ),
}
DEFAULT_CELLS = (
    "c4:dec32@60000,c48:2048@32768,c48:2048@131072,c64:2048@32768,c64:2048@131072"
)

INTERPRET = False


def apply_config(additional: dict) -> None:
    """Module flags as the attention layer / runner set them (read at trace)."""
    msa_dcp.TILE_SELECT_ENABLED = bool(additional.get("minimax_msa_tile_select"))
    msa_dcp.LIVE_SELECT_ENABLED = bool(additional.get("minimax_msa_live_select"))
    msa_indexer.set_int4_fp8_keys(bool(additional.get("msa_index_int4_fp8")))
    kernel_toggles.apply(additional)
    jax.clear_caches()


# ----------------------------------------------------------------- the step
class Step:
    """Host-side ragged metadata of one cell."""

    def __init__(self, args, cfg, cell_tokens: str, ctx: int):
        self.dcp = dcp = cfg["dcp"]
        self.G = max(1, dcp // REPLICAS)
        self.fp4 = cfg["fp4"]
        self.max_seqs = cfg["max_seqs"]
        if cell_tokens.startswith("dec"):
            self.T = int(cell_tokens[3:])
            nd = self.T // DECODE_ROWS
            q_lens, ctxs = [DECODE_ROWS] * nd, [ctx] * nd
            self.kind = "decode"
        else:
            self.T = int(cell_tokens)
            nd = args.decode_reqs
            q_lens = [DECODE_ROWS] * nd + [self.T - DECODE_ROWS * nd]
            ctxs = [args.decode_ctx] * nd + [ctx]
            self.kind = "prefill"
        assert len(q_lens) <= self.max_seqs, (len(q_lens), self.max_seqs)
        self.q_lens = q_lens
        B = len(q_lens)
        seq_lens = np.zeros(self.max_seqs, np.int32)
        seq_lens[:B] = np.add(ctxs, q_lens)
        cu = np.full(self.max_seqs + 1, sum(q_lens), np.int32)
        cu[: B + 1] = np.concatenate([[0], np.cumsum(q_lens)])
        self.seq_lens, self.cu = seq_lens, cu
        pages_per_seq = -(-args.max_model_len // (PAGE * dcp))
        need = [-(-int(s) // (PAGE * dcp)) for s in seq_lens[:B]]
        self.num_pages = int(sum(need)) + 8
        perm = np.random.default_rng(1234).permutation(np.arange(1, self.num_pages))
        tables = np.zeros((self.max_seqs, pages_per_seq), np.int32)
        at = 0
        for s in range(B):
            tables[s, : need[s]] = perm[at : at + need[s]]
            at += need[s]
        self.tables = tables
        self.dist = np.asarray([nd, nd, B], np.int32)

    def meta(self):
        return (
            jnp.asarray(self.seq_lens),
            jnp.asarray(self.tables.reshape(-1)),
            jnp.asarray(self.cu),
            jnp.asarray(self.dist),
        )


def coherent_queries(rng, step, layers, args):
    """Index queries ``[layers, T, 1, D]``: a request vector, a drift vector
    per `drift_tokens` of the request, and per-token noise (as
    bench_msa_prefill_sweep.py)."""
    out = np.zeros((layers, step.T, 1, D), np.float32)
    for start, q_len in zip(step.cu[:-1], step.q_lens):
        base = rng.standard_normal((layers, 1, 1, D))
        segs = -(-q_len // args.drift_tokens)
        drift = np.repeat(
            rng.standard_normal((layers, segs, 1, D)), args.drift_tokens, axis=1
        )[:, :q_len]
        noise = rng.standard_normal((layers, q_len, 1, D))
        out[:, start : start + q_len] = (
            base + args.q_drift * drift + args.q_noise * noise
        )
    return out


# ------------------------------------------------------------- the layers
def _all_gather(x, dim):
    return jax.lax.all_gather(x, "dcp", axis=dim, tiled=True)


def _keep(*xs):
    acc = jnp.zeros((), jnp.float32)
    for x in xs:
        acc = acc + x.reshape(-1)[:1].astype(jnp.float32).sum()
    return acc


class Layers:
    """Per-rank Full / Shared layer bodies of one cell, as the op runs them."""

    def __init__(self, step):
        self.s = step

    def _gather(self, kvq, n):
        """`ops.msa.gather_group_kv` (n = 3) / `gather_group_kv_only` (n = 2):
        [T, 3, D] per rank -> n tensors [T, G, D]."""
        if self.s.G == 1:
            return tuple(kvq[:, i : i + 1] for i in range(n))
        T = kvq.shape[0]
        per_rank = _all_gather(kvq[:, :n], 1)  # [T, dcp * n, D]
        per_head = per_rank.reshape(T, self.s.dcp, n, D)[:, ::REPLICAS]
        return tuple(per_head[:, :, i] for i in range(n))

    def _q_all(self, q):
        return _all_gather(msa_fp4.rotate_queries(q) if self.s.fp4 else q, 1)

    def _lse_merge(self, out, lse):
        """`ops.msa.merge_head_partials` (TPU_DCP_RS_OUT_PRECISION on)."""
        lse = lse.astype(jnp.float32).T
        lses = _all_gather(lse[None], 0)
        weight = jnp.exp(lse - jax.scipy.special.logsumexp(lses, axis=0)).T[..., None]
        weighted = out.astype(jnp.float32) * weight
        weighted = jnp.where(weight > 0, weighted, 0.0).astype(out.dtype)
        merged = jax.lax.psum_scatter(weighted, "dcp", scatter_dimension=0, tiled=True)
        if self.s.fp4:
            merged = msa_fp4.unrotate_output(merged.astype(out.dtype))
        return merged.transpose(1, 0, 2)

    def full(self, x, caches, M, rank):
        kv, ic, sc = caches
        seq_lens, tables, cu, dist = M
        k, v, iq = self._gather(x["kvq"], 3)
        res = msa_dcp.msa_write_and_candidates_dcp(
            k,
            v,
            iq,
            x["ik"],
            kv,
            ic,
            seq_lens,
            tables,
            cu,
            dist,
            topk=TOPK,
            init_blocks=INIT_BLOCKS,
            local_blocks=LOCAL_BLOCKS,
            rank=rank,
            dcp=self.s.dcp,
            scale_cache=sc,
            interpret=INTERPRET,
        )
        if self.s.fp4:
            kv, ic, sc, cand = res
        else:
            kv, ic, cand = res
        q_all = self._q_all(x["q"])
        cand_all = _all_gather(cand[None], 0)
        out, lse, selected = msa_dcp.msa_attend_dcp(
            q_all,
            kv,
            cand_all,
            seq_lens,
            tables,
            cu,
            sm_scale=SM_SCALE,
            topk=TOPK,
            rank=rank,
            dcp=self.s.dcp,
            scale_cache=sc,
            interpret=INTERPRET,
            return_selected=True,
        )
        return self._lse_merge(out, lse), (kv, ic, sc), selected

    def shared(self, x, caches, M, rank, selected, write_index):
        kv, ic, sc = caches
        seq_lens, tables, cu, _ = M
        k, v = self._gather(x["kvq"], 2)
        res = msa_dcp.msa_write_dcp(
            k,
            v,
            x["ik"] if write_index else None,
            kv,
            ic if write_index else None,
            seq_lens,
            tables,
            cu,
            rank=rank,
            dcp=self.s.dcp,
            scale_cache=sc,
            interpret=INTERPRET,
        )
        kv, rest = res[0], list(res[1:])
        if write_index:
            ic = rest.pop(0)
        if self.s.fp4:
            sc = rest.pop(0)
        q_all = self._q_all(x["q"])
        out, lse = msa_dcp.msa_attend_selected_dcp(
            q_all,
            kv,
            selected,
            seq_lens,
            tables,
            cu,
            sm_scale=SM_SCALE,
            rank=rank,
            dcp=self.s.dcp,
            scale_cache=sc,
            interpret=INTERPRET,
        )
        return self._lse_merge(out, lse), (kv, ic, sc)


def program(mesh, step, stride, in_layers, write_index):
    """jit(shard_map) of ``fori_loop(0, n, group)``: one Full + stride - 1
    Shared layers per group."""
    dev = P(("rep", "dcp"))
    layers = Layers(step)

    def per_rank(n, X, M, caches):
        X = {k: v[0] for k, v in X.items()}
        caches = tuple(c[0] if c is not None else None for c in caches)
        rank = jax.lax.axis_index("dcp").astype(jnp.int32).reshape(1)

        def group(i, carry):
            acc, kv, ic, sc = carry
            state = (kv, ic, sc)

            def x(j):
                layer = (i * stride + j) % in_layers
                return {k: v[layer] for k, v in X.items()}

            out, state, selected = layers.full(x(0), state, M, rank)
            acc = acc + _keep(out)
            for j in range(1, stride):
                out, state = layers.shared(x(j), state, M, rank, selected, write_index)
                acc = acc + _keep(out)
            return (acc, *state)

        acc = jnp.zeros((), jnp.float32)
        out = jax.lax.fori_loop(0, n, group, (acc, *caches))
        return tuple(o[None] if o is not None else None for o in out)

    return jax.jit(
        jax.shard_map(
            per_rank,
            mesh=mesh,
            in_specs=(P(), dev, P(), dev),
            out_specs=dev,
            check_vma=False,
        )
    )


def make_inputs(mesh, step, args, in_layers):
    rng = np.random.default_rng(7)
    sharding = NamedSharding(mesh, P(("rep", "dcp")))
    T, G = step.T, step.G

    def shard(x):
        return jax.device_put(x, sharding)

    q = rng.standard_normal((TP, in_layers, T, Q_HEADS, D)).astype(np.float32)
    kvq = rng.standard_normal((TP, in_layers, T, 3, D)).astype(np.float32)
    kvq[:, :, :, 2:] = np.stack(
        [coherent_queries(rng, step, in_layers, args) for _ in range(TP)]
    )
    ik = rng.standard_normal((TP, in_layers, T, D)).astype(np.float32)
    X = dict(
        q=shard(jnp.asarray(q, jnp.bfloat16)),
        kvq=shard(jnp.asarray(kvq, jnp.bfloat16)),
        ik=shard(jnp.asarray(ik, jnp.bfloat16)),
    )
    pages = step.num_pages

    if step.fp4:

        @functools.partial(jax.jit, out_shardings=(sharding,) * 3)
        def caches(key):
            k1, k2, k3, k4 = jax.random.split(key, 4)
            kv = jax.random.bits(k1, (TP, pages, PAGE // 8, 4, G * 2 * D), jnp.uint8)
            nib = jax.random.randint(k2, (TP * pages, PAGE, D), 0, 16, jnp.int32)
            isc = jax.random.uniform(k3, (TP * pages, D), jnp.float32, 0.01, 0.03)
            ic = msa_fp4.pack_rows(nib.astype(jnp.uint8), isc)
            ic = ic.reshape((TP, pages) + ic.shape[1:])
            sc = jax.random.uniform(
                k4, (TP, pages, msa_fp4.scale_rows(G), PAGE), jnp.float32, 0.01, 0.03
            )
            return kv, ic, sc

        return X, caches(jax.random.PRNGKey(0))

    @functools.partial(jax.jit, out_shardings=(sharding,) * 2)
    def caches16(key):
        k1, k2 = jax.random.split(key)
        kv = jax.random.normal(k1, (TP, pages, PAGE // 2, 2, G * 2 * D), jnp.bfloat16)
        ic = jax.random.normal(k2, (TP, pages, PAGE // 2, 2, D), jnp.bfloat16)
        return kv, ic

    return X, (*caches16(jax.random.PRNGKey(0)), None)


def time_program(fn, n, X, M, caches, iters):
    nn = jnp.int32(n)
    jax.block_until_ready(fn(nn, X, M, caches))
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(nn, X, M, caches))
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e6


def main():
    global INTERPRET
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cells", default=DEFAULT_CELLS)
    p.add_argument("--strides", default="1,2,4")
    p.add_argument("--groups", type=int, default=4, help="N groups for the slope")
    p.add_argument("--in-layers", type=int, default=4, help="distinct layer inputs")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--skip-index-write",
        action="store_true",
        help="Shared layers skip the index-key write",
    )
    p.add_argument(
        "--decode-reqs",
        type=int,
        default=0,
        help="decode requests (8 rows each) alongside a prefill chunk",
    )
    p.add_argument("--decode-ctx", type=int, default=60000)
    p.add_argument("--max-model-len", type=int, default=196608)
    p.add_argument("--q-noise", type=float, default=0.2)
    p.add_argument("--q-drift", type=float, default=0.7)
    p.add_argument("--drift-tokens", type=int, default=64)
    p.add_argument("--json", default="", help="append one JSON line per row")
    p.add_argument("--smoke", action="store_true", help="CPU, interpret, tiny shapes")
    args = p.parse_args()
    if args.smoke:
        INTERPRET = True
        if args.cells == DEFAULT_CELLS:
            args.cells = "c4:dec16@2048,c48:256@1024"
        args.groups, args.iters, args.in_layers = 2, 1, 2
        args.max_model_len = 8192
        msa_dcp.TILE_SELECT_MIN_TOKENS = 0
        msa_dcp._MIN_CANDIDATE_TOKENS = 0
        for cfg in CONFIGS.values():
            cfg["max_seqs"] = 4
    devices = jax.devices()
    print(f"devices: {len(devices)} x {devices[0].device_kind}")
    assert len(devices) >= TP, f"need {TP} devices, got {len(devices)}"
    strides = [int(s) for s in args.strides.split(",")]
    rows = []
    for cell in args.cells.split(","):
        cfg_name, rest = cell.split(":")
        tokens, ctx = rest.split("@")
        cfg = CONFIGS[cfg_name]
        apply_config(cfg["additional"])
        dcp = cfg["dcp"]
        mesh = Mesh(np.asarray(devices[:TP]).reshape(TP // dcp, dcp), ("rep", "dcp"))
        step = Step(args, cfg, tokens, int(ctx))
        X, caches = make_inputs(mesh, step, args, args.in_layers)
        M = step.meta()
        base = None
        for stride in strides:
            fn = program(mesh, step, stride, args.in_layers, not args.skip_index_write)
            t1 = time_program(fn, 1, X, M, caches, args.iters)
            tn = time_program(fn, args.groups, X, M, caches, args.iters)
            per_layer = (tn - t1) / ((args.groups - 1) * stride)
            base = base or per_layer
            row = dict(
                cell=cell,
                kind=step.kind,
                dcp=dcp,
                G=step.G,
                fp4=step.fp4,
                T=step.T,
                ctx=int(ctx),
                stride=stride,
                skip_index_write=args.skip_index_write,
                per_layer_us=round(per_layer, 2),
                x60_ms=round(per_layer * NUM_LAYERS / 1e3, 3),
                vs_stride1=round(base / per_layer, 3),
            )
            rows.append(row)
            print(
                f"{cell:22s} {step.kind:7s} dcp{dcp} G{step.G} stride {stride}: "
                f"{per_layer:9.1f} us/layer  x60 {row['x60_ms']:8.3f} ms  "
                f"speedup {row['vs_stride1']:.3f}x",
                flush=True,
            )
    if args.json:
        with open(args.json, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
