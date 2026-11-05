# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Decode kernel parity and latency at 64, 80 and 96 rows on TPU cores:

  fused_norm_qkva         MLA decode prologue vs reference add+RMSNorm+fp8 dot+norms;
  lm_head_hbm             HBM-streamed LM head vs XLA dot_general and f64 product;
  fp8_oproj_all_reduce    fused W_UV + o_proj + two-shot all-reduce vs separate calls;
  mla_kv_write_in_gather  SparseCore gather with 4-6 round kv_write vs writer + gather.

Each benchmark runs in an isolated subprocess, checks numerical / bitwise parity
against the fallback path and reference, times repeated executions, and prints a
single RESULT JSON line per kernel.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROWS = (64, 80, 96)
REPS = 20


def _time_fn(fn, args, reps=REPS, layers=1):
    import numpy as np

    t0 = time.monotonic()
    out = fn(*args)
    if isinstance(out, tuple):
        out[0].block_until_ready()
        arr = np.asarray(out[0], np.float32)
    else:
        out.block_until_ready()
        arr = np.asarray(out, np.float32)
    compile_s = time.monotonic() - t0
    warmup = fn(*args)
    (warmup[0] if isinstance(warmup, tuple) else warmup).block_until_ready()
    t1 = time.perf_counter()
    for _ in range(reps):
        res = fn(*args)
    (res[0] if isinstance(res, tuple) else res).block_until_ready()
    us = (time.perf_counter() - t1) / (reps * layers) * 1e6
    finite = bool(np.isfinite(arr).all()) if arr.dtype.kind == "f" else True
    return round(us, 2), round(compile_s, 2), finite


def _rel_cos(got, want):
    import numpy as np

    g = np.asarray(got, np.float64).ravel()
    w = np.asarray(want, np.float64).ravel()
    wn = float(np.linalg.norm(w))
    gn = float(np.linalg.norm(g))
    if wn == 0.0 or gn == 0.0:
        same = bool(np.array_equal(g, w))
        return (0.0 if same else 1.0), (1.0 if same else 0.0)
    rel = float(np.linalg.norm(g - w) / wn)
    cos = float(np.vdot(g, w) / (gn * wn))
    return round(rel, 6), round(cos, 6)


def _setup_src_path():
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[3] / "src"))
    sys.path.insert(0, str(here.parents[1]))


def _bench_fused_norm_qkva():
    import jax
    import jax.numpy as jnp
    import numpy as np

    _setup_src_path()
    from vllm_torchtpu.kernels import fused_norm_qkva as k

    h, q, kv, rope = 6144, 2048, 512, 64
    n_out = q + kv + rope
    eps, q_eps, kv_eps = 1e-5, 1e-6, 1e-6
    qmax = float(jnp.finfo(jnp.float8_e4m3fn).max)
    n_layers = 8  # 8 * 16.1 MB = 129 MB > 64 MB VMEM -> cold HBM weights

    def rms_f64(v, w, e):
        v64 = np.asarray(v, np.float64)
        w64 = np.asarray(w, np.float64)
        var = np.mean(v64 * v64, axis=-1, keepdims=True)
        return (v64 / np.sqrt(var + e)) * w64

    def max_ulps_floored(got_bf16, ref_f64):
        g64 = np.asarray(got_bf16, np.float64)
        r64 = np.asarray(ref_f64, np.float64)
        ulp = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(r64), 2.0**-6))) - 7)
        return float(np.max(np.abs(g64 - r64) / ulp))

    def kern_step(x, r, ln_w, w_q_t, w_s, qa_w, kva_w):
        return k.fused_norm_qkva(
            x,
            r,
            ln_w,
            w_q_t.T,
            w_s,
            qa_w,
            kva_w,
            eps=eps,
            q_eps=q_eps,
            kv_eps=kv_eps,
            q_rank=q,
            kv_rank=kv,
            emit_q8=True,
            hbm_pin=True,
        )

    def fb_step(x, r, ln_w, w_q_t, w_s, qa_w, kva_w):
        return k.reference(
            x,
            r,
            ln_w,
            w_q_t.T,
            w_s,
            qa_w,
            kva_w,
            eps=eps,
            q_eps=q_eps,
            kv_eps=kv_eps,
            q_rank=q,
            kv_rank=kv,
        )

    def make_chain(step_fn):
        @jax.jit
        def chain(x, r, layer_weights):
            for ln_i, wqt_i, ws_i, qa_i, kva_i in layer_weights:
                outs = step_fn(x, r, ln_i, wqt_i, ws_i, qa_i, kva_i)
                tag = (
                    outs[2][:, :1]
                    + outs[3][:, :1]
                    + outs[4][:, :1]
                    + outs[5][:, :1].astype(jnp.bfloat16)
                    + outs[6].astype(jnp.bfloat16)
                ) * jnp.bfloat16(1e-4)
                x = outs[0] + tag
                r = outs[1]
            return x, r

        return chain

    kern_fn = jax.jit(kern_step)
    fb_fn = jax.jit(fb_step)
    kern_chain = make_chain(kern_step)
    fb_chain = make_chain(fb_step)
    quant_rows_fn = jax.jit(
        lambda qc: k._quant_rows(qc.astype(jnp.float32), qmax, jnp.float8_e4m3fn)
    )
    result = dict(
        kernel="fused_norm_qkva",
        devices=len(jax.devices()),
        device_kind=jax.devices()[0].device_kind,
        max_rows=int(k.MAX_ROWS),
        chain_overhead_us=0.0,
        cases={},
    )

    rng_layers = np.random.default_rng(7)
    layer_weights = tuple(
        (
            jnp.asarray(
                (1 + 0.1 * rng_layers.standard_normal(h, np.float32)).astype(
                    jnp.bfloat16
                )
            ),
            jnp.asarray(
                (rng_layers.standard_normal((n_out, h), np.float32) * 0.3).astype(
                    jnp.float8_e4m3fn
                )
            ),
            jnp.asarray(rng_layers.uniform(0.002, 0.02, n_out).astype(np.float32)),
            jnp.asarray(
                (1 + 0.1 * rng_layers.standard_normal(q, np.float32)).astype(
                    jnp.bfloat16
                )
            ),
            jnp.asarray(
                (1 + 0.1 * rng_layers.standard_normal(kv, np.float32)).astype(
                    jnp.bfloat16
                )
            ),
        )
        for _ in range(n_layers)
    )

    for rows in ROWS:
        rng = np.random.default_rng(100 + rows)
        x = (rng.standard_normal((rows, h), np.float32) * 0.5).astype(jnp.bfloat16)
        r = (rng.standard_normal((rows, h), np.float32) * 2.0).astype(jnp.bfloat16)
        x[-1] = 0
        r[-1] = 0
        ln_w, w_q_t, w_s, qa_w, kva_w = layer_weights[0]
        x_j = jnp.asarray(x)
        r_j = jnp.asarray(r)
        args = (x_j, r_j, ln_w, w_q_t, w_s, qa_w, kva_w)

        admitted = k.supports(rows, h, n_out, q, kv, rope)
        if not admitted:
            result["cases"][f"{rows}:kernel"] = dict(admitted=False)
            continue

        got = kern_fn(*args)
        want = fb_fn(*args)
        q8_ref, qs_ref = quant_rows_fn(got[2])

        # Exact f64 oracle for bf16 x fp8 f32-accumulated matmul + latent norms.
        xq_y, xs_y = quant_rows_fn(got[0])
        acc_f64 = (
            np.asarray(xq_y.astype(jnp.bfloat16), np.float64)
            @ np.asarray(w_q_t, np.float64).T
        )
        deq_bf16 = (
            (acc_f64 * np.asarray(xs_y, np.float64) * np.asarray(w_s, np.float64)[None, :])
            .astype(np.float32)
            .astype(jnp.bfloat16)
        )
        qc_f64 = rms_f64(deq_bf16[:, :q], qa_w, q_eps)
        kvc_f64 = rms_f64(deq_bf16[:, q : q + kv], kva_w, kv_eps)
        kpe_f64 = np.asarray(deq_bf16[:, q + kv :], np.float64)
        ulps = max(
            max_ulps_floored(got[2], qc_f64),
            max_ulps_floored(got[3], kvc_f64),
            max_ulps_floored(got[4], kpe_f64),
        )
        ulps_yr = max(
            max_ulps_floored(got[0], want[0]),
            max_ulps_floored(got[1], want[1]),
        )
        bitwise_ok = bool(
            np.asarray(got[1]).tobytes() == np.asarray(want[1]).tobytes()
            and np.asarray(got[5]).tobytes() == np.asarray(q8_ref).tobytes()
            and np.asarray(got[6]).tobytes() == np.asarray(qs_ref).tobytes()
        )
        rel_l2, cos = _rel_cos(got[2], qc_f64)
        chain_args = (x_j, r_j, layer_weights)
        k_us, k_comp, k_fin = _time_fn(kern_chain, chain_args, layers=n_layers)
        f_us, f_comp, f_fin = _time_fn(fb_chain, chain_args, layers=n_layers)
        result["cases"][f"{rows}:kernel"] = dict(
            admitted=True,
            per_layer_us=k_us,
            compile_s=k_comp,
            finite=k_fin,
            rel_l2=rel_l2,
            cos=cos,
            max_ulps=round(ulps, 3),
            max_ulps_yr=round(ulps_yr, 3),
            bitwise_exact_fields=bitwise_ok,
            ranks_identical=bool(ulps_yr <= 1.0 and ulps <= 4.0),
            nan=int(sum(np.isnan(np.asarray(t, np.float32)).sum() for t in got[:5])),
        )
        result["cases"][f"{rows}:fallback"] = dict(
            per_layer_us=f_us,
            compile_s=f_comp,
            finite=f_fin,
        )
    print("RESULT " + json.dumps(result), flush=True)


def _bench_lm_head_hbm():
    import jax
    import jax.numpy as jnp
    import numpy as np

    _setup_src_path()
    from vllm_torchtpu.kernels import lm_head_hbm as lh

    kk, nn = 6144, 9680
    rng_w = np.random.default_rng(1)
    w = jnp.asarray(rng_w.standard_normal((nn, kk), np.float32) * 0.02, jnp.bfloat16)
    kern_fn = jax.jit(lh.lm_head_hbm)
    fb_fn = jax.jit(lambda a, b: jax.lax.dot_general(a, b, (((1,), (1,)), ((), ()))))

    result = dict(
        kernel="lm_head_hbm",
        devices=len(jax.devices()),
        device_kind=jax.devices()[0].device_kind,
        max_m=int(lh.MAX_M),
        chain_overhead_us=0.0,
        cases={},
    )
    w_f64 = np.asarray(w, np.float64).T

    for rows in ROWS:
        rng = np.random.default_rng(200 + rows)
        x_np = rng.standard_normal((rows, kk), np.float32)
        x_np[:, rng.choice(kk, 8, replace=False)] *= 8.0
        x = jnp.asarray(x_np, jnp.bfloat16)
        if not lh.supports(rows, x, w):
            result["cases"][f"{rows}:kernel"] = dict(admitted=False)
            continue

        got = kern_fn(x, w)
        ref = fb_fn(x, w)
        f64 = np.asarray(x, np.float64) @ w_f64
        g64 = np.asarray(got, np.float64)
        r64 = np.asarray(ref, np.float64)
        ulp = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(r64), 2.0**-6))) - 7)
        d_rel = float((np.abs(g64 - r64) / ulp).max())
        err = float(np.abs(g64 - f64).max())
        ref_err = float(np.abs(r64 - f64).max())
        argmax_ok = bool(
            (
                np.argmax(np.asarray(got, np.float32), 1)
                == np.argmax(np.asarray(ref, np.float32), 1)
            ).all()
        )
        judge_ok = bool(
            d_rel <= 8.0
            and err <= 1.25 * ref_err + float(np.abs(f64).max()) * 2.0**-8
            and argmax_ok
        )
        rel_l2, cos = _rel_cos(g64, f64)
        k_us, k_comp, k_fin = _time_fn(kern_fn, (x, w))
        f_us, f_comp, f_fin = _time_fn(fb_fn, (x, w))
        result["cases"][f"{rows}:kernel"] = dict(
            admitted=True,
            per_layer_us=k_us,
            compile_s=k_comp,
            finite=k_fin,
            rel_l2=rel_l2,
            cos=cos,
            max_ulp_floored=round(d_rel, 2),
            argmax_ok=argmax_ok,
            ranks_identical=judge_ok,
            nan=int(np.isnan(g64).sum()),
        )
        result["cases"][f"{rows}:fallback"] = dict(
            per_layer_us=f_us,
            compile_s=f_comp,
            finite=f_fin,
        )
    print("RESULT " + json.dumps(result), flush=True)


def _bench_fp8_oproj_all_reduce():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    _setup_src_path()
    from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce
    from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce as op
    from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

    devices = jax.devices()
    n = len(devices)
    mesh = Mesh(np.asarray(devices), ("d",))
    spec = P("d")
    sharding = NamedSharding(mesh, spec)
    heads, latent, v_dim = 8, 512, 128
    kk, n_out = heads * v_dim, 6144
    n_layers = 10  # 10 * 7.37 MB/core = 73.7 MB > 64 MB VMEM -> cold HBM weights

    rng = np.random.default_rng(42)
    layer_weights = tuple(
        (
            jax.device_put(
                jnp.asarray(
                    (
                        rng.standard_normal((n, heads, latent, v_dim), np.float32)
                        * 0.05
                    ).astype(jnp.bfloat16)
                ).reshape(n * heads, latent, v_dim),
                sharding,
            ),
            jax.device_put(
                jnp.asarray(
                    (rng.standard_normal((n, kk, n_out), np.float32) * 0.2).astype(
                        jnp.float8_e4m3fn
                    )
                ).reshape(n * kk, n_out),
                sharding,
            ),
            jax.device_put(
                jnp.asarray(rng.uniform(0.002, 0.02, (n, n_out)).astype(np.float32)).reshape(
                    n * n_out
                ),
                sharding,
            ),
        )
        for _ in range(n_layers)
    )
    ranks_np = np.arange(n, dtype=np.int32)
    rank_ids = jax.device_put(jnp.asarray(ranks_np), sharding)

    def kern_body(x_lat, wuv, wq, ws, r):
        return op.fp8_uv_matmul_all_reduce(
            x_lat,
            wuv,
            wq,
            ws,
            r,
            num_cores=n,
            axis="d",
            out_dtype=jnp.bfloat16,
            waves=True,
            max_m=op.LIMIT_M,
        )

    def two_call_body(x_lat, wuv, wq, ws, r):
        dn = (((1,), (0,)), ((), ()))
        parts = []
        for h in range(heads):
            yh = jax.lax.dot_general(
                x_lat[:, h * latent : (h + 1) * latent],
                wuv[h].astype(jnp.bfloat16),
                dn,
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            parts.append(yh)
        xv = jnp.concatenate(parts, axis=1)
        qmax = float(jnp.finfo(jnp.float8_e4m3fn).max)
        sc = jnp.max(jnp.abs(xv), axis=1, keepdims=True).astype(
            jnp.float32
        ) * jnp.float32(1.0 / qmax)
        sc = jnp.where(sc == 0, jnp.float32(1.0), sc)
        xq = jnp.clip(
            xv.astype(jnp.float32) * (jnp.float32(1.0) / sc), -qmax, qmax
        ).astype(jnp.float8_e4m3fn)
        mm = jax.lax.dot_general(
            xq.astype(jnp.bfloat16), wq, dn, preferred_element_type=jnp.float32
        )
        partial = (mm * sc * ws[None, :]).astype(jnp.bfloat16)
        return all_reduce.small_all_reduce(
            partial, num_cores=n, axis="d", rank=r, pair_rows=0
        )

    def fb_body(x_lat, wuv, wq, ws, r):
        # Production fallback when uv_oproj_fits is False (attention.py:3203-3218
        # + mla_attention_op.py:2579 -> small_all_reduce_op.py:MAX_ROWS=96).
        outputs_t = jnp.swapaxes(x_lat.reshape(-1, heads, latent), 0, 1)
        out_proj = jax.lax.dot_general(
            outputs_t,
            wuv,
            (((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        xv = jnp.swapaxes(out_proj, 0, 1).astype(jnp.bfloat16).reshape(-1, kk)
        partial = xla_quantized_matmul(xv, wq, ws)
        return all_reduce.small_all_reduce(
            partial, num_cores=n, axis="d", rank=r, pair_rows=0
        )

    def make_chain(step_fn):
        def chain_body(x_lat, layer_ws, r):
            out = x_lat
            for wuv_i, wq_i, ws_i in layer_ws:
                out = step_fn(x_lat, wuv_i, wq_i, ws_i, r)
                x_lat = out[:, : heads * latent]
            return out

        return jax.jit(
            jax.shard_map(
                chain_body,
                mesh=mesh,
                in_specs=(spec, ((spec, spec, spec),) * n_layers, spec),
                out_specs=spec,
                check_vma=False,
            )
        )

    kern_fn = jax.jit(
        jax.shard_map(
            kern_body, mesh=mesh, in_specs=(spec,) * 5, out_specs=spec, check_vma=False
        )
    )
    two_call_fn = jax.jit(
        jax.shard_map(
            two_call_body,
            mesh=mesh,
            in_specs=(spec,) * 5,
            out_specs=spec,
            check_vma=False,
        )
    )
    kern_chain = make_chain(kern_body)
    fb_chain = make_chain(fb_body)

    result = dict(
        kernel="fp8_oproj_all_reduce",
        devices=n,
        device_kind=devices[0].device_kind,
        limit_m=int(op.LIMIT_M),
        chain_overhead_us=0.0,
        cases={},
    )

    for rows in ROWS:
        admitted = op.supports_uv(rows, heads, latent, kk, n_out, n, op.LIMIT_M)
        if not admitted:
            result["cases"][f"{rows}:kernel"] = dict(admitted=False)
            continue

        rng_x = np.random.default_rng(300 + rows)
        x_np = (rng_x.standard_normal((n, rows, heads * latent), np.float32) * 0.5).astype(
            jnp.bfloat16
        )
        x_lat = jax.device_put(
            jnp.asarray(x_np).reshape(n * rows, heads * latent),
            sharding,
        )
        w_uv_flat, w_q_flat, w_s_flat = layer_weights[0]
        args = (x_lat, w_uv_flat, w_q_flat, w_s_flat, rank_ids)

        got = np.asarray(kern_fn(*args), np.float32).reshape(n, rows, n_out)
        want = np.asarray(two_call_fn(*args), np.float32).reshape(n, rows, n_out)
        bits = got.view(np.uint32)
        ranks_same = bool((bits == bits[0]).all())
        bitwise_two_call = bool(np.array_equal(bits, want.view(np.uint32)))
        mag = np.maximum(np.maximum(np.abs(got[0]), np.abs(want[0])), 1e-30)
        ulps = float(
            np.max(np.abs(got[0] - want[0]) / np.exp2(np.floor(np.log2(mag)) - 7))
        )
        rel_l2, cos = _rel_cos(got[0], want[0])
        chain_args = (x_lat, layer_weights, rank_ids)
        k_us, k_comp, k_fin = _time_fn(kern_chain, chain_args, layers=n_layers)
        f_us, f_comp, f_fin = _time_fn(fb_chain, chain_args, layers=n_layers)
        result["cases"][f"{rows}:kernel"] = dict(
            admitted=True,
            per_layer_us=k_us,
            compile_s=k_comp,
            finite=k_fin,
            rel_l2=rel_l2,
            cos=cos,
            max_ulps=round(ulps, 3),
            bitwise_two_call=bitwise_two_call,
            ranks_identical=bool(ranks_same and ulps <= 2.0),
            nan=int(np.isnan(got).sum()),
        )
        result["cases"][f"{rows}:fallback"] = dict(
            per_layer_us=f_us,
            compile_s=f_comp,
            finite=f_fin,
        )
    print("RESULT " + json.dumps(result), flush=True)


def _bench_mla_kv_write_in_gather():
    import jax
    import jax.numpy as jnp
    import numpy as np

    _setup_src_path()
    from vllm_torchtpu.kernels.mla import kv_cache_utils
    from vllm_torchtpu.kernels.mla.kv_cache_utils import (
        KVCacheLayout,
        KVCacheType,
        SparseMLAKVCacheSpec,
    )
    from vllm_torchtpu.kernels.mla.sparse import native_sc_gather
    from vllm_torchtpu.layers.core import attention_interface

    num_pages, page_size = 128, 256
    lkv_dim, rope_dim, kv_packing = 512, 64, 4
    dcp_size, interleave = 16, 4
    topk_cap = 512
    nope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.NOPE, KVCacheLayout.SPARSECORE, num_pages, page_size, lkv_dim, kv_packing
    )
    rope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.ROPE, KVCacheLayout.SPARSECORE, num_pages, page_size, rope_dim, kv_packing
    )

    result = dict(
        kernel="mla_kv_write_in_gather",
        devices=len(jax.devices()),
        device_kind=jax.devices()[0].device_kind,
        max_write_tokens=int(native_sc_gather.MAX_WRITE_TOKENS),
        chain_overhead_us=0.0,
        cases={},
    )

    for rows in ROWS:
        if rows > native_sc_gather.MAX_WRITE_TOKENS:
            result["cases"][f"{rows}:kernel"] = dict(admitted=False)
            continue

        n_seqs = rows // 4
        rng = np.random.default_rng(400 + rows)
        seqs = [
            ((r % dcp_size) * interleave + c * dcp_size * interleave + 1, 4)
            for c, r in enumerate(rng.integers(0, dcp_size, n_seqs))
        ]
        q_lens = jnp.asarray([n for _, n in seqs], jnp.int32)
        seq_lens = jnp.asarray([p + n for p, n in seqs], jnp.int32)
        q_start = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(q_lens)])
        block_tables = jnp.asarray(
            rng.permutation(num_pages)[: n_seqs * 3], jnp.int32
        )
        finite_u8 = np.asarray(
            [b for b in range(256) if b not in (0x7F, 0xFF)], dtype=np.uint8
        )
        kv_c = jax.lax.bitcast_convert_type(
            jnp.asarray(rng.choice(finite_u8, (rows, lkv_dim))),
            jnp.float8_e4m3fn,
        )
        k_pe = jax.lax.bitcast_convert_type(
            jnp.asarray(rng.choice(finite_u8, (rows, rope_dim))),
            jnp.float8_e4m3fn,
        )
        caches = tuple(
            jnp.asarray(rng.integers(0, 2**32, s.shape, np.uint32), s.jax_dtype)
            for s in (nope_spec, rope_spec)
        )
        rank = jnp.int32(0)
        page, slot = kv_cache_utils._get_dcp_page_and_slot(
            rows,
            seq_lens,
            block_tables,
            q_start,
            rank,
            nope_spec=nope_spec,
            dcp_size=dcp_size,
            interleave_size=interleave,
        )
        dst = kv_cache_utils._dst_rows(page, slot, page_size)
        out_size = rows * topk_cap
        num_rows = caches[0].shape[0] * caches[0].shape[1]
        flat = rng.integers(0, num_rows, out_size, dtype=np.int32)
        owned = np.asarray(dst)[np.asarray(dst) >= 0]
        if owned.size:
            per_owner = out_size // 32
            flat[::per_owner] = np.resize(owned, 32)
        idx = jnp.asarray(flat.reshape(rows, topk_cap))

        def kern_step(c0, c1, kv_in, pe_in, dst_in, idx_in, osz=out_size):
            np_pack = attention_interface._pack_lane_dense(kv_in, nope_spec.token_bytes)
            rp_pack = attention_interface._pack_lane_dense(pe_in, rope_spec.token_bytes)
            w_rows = native_sc_gather.pack_write_rows(dst_in, np_pack, rp_pack)
            return native_sc_gather.dsa_gather_native_sc(
                c0,
                c1,
                idx_in,
                out_size=osz,
                atoms_per_batch=1,
                num_live_indices=jnp.int32(osz),
                live_multiple=2048,
                kv_write=w_rows,
            )

        def fb_step(
            c0, c1, kv_in, pe_in, sl_in, bt_in, qs_in, rk_in, idx_in, osz=out_size
        ):
            wc0, wc1 = kv_cache_utils.update_sparse_mla_kv_cache_dcp(
                c0,
                c1,
                kv_in,
                pe_in,
                sl_in,
                bt_in,
                qs_in,
                rk_in,
                nope_spec=nope_spec,
                rope_spec=rope_spec,
                dcp_size=dcp_size,
                interleave_size=interleave,
            )
            g0, g1 = native_sc_gather.dsa_gather_native_sc(
                wc0,
                wc1,
                idx_in,
                out_size=osz,
                atoms_per_batch=1,
                num_live_indices=jnp.int32(osz),
                live_multiple=2048,
            )
            return g0, g1, wc0, wc1

        kern_fn = jax.jit(kern_step)
        fb_fn = jax.jit(fb_step)
        k_args = (caches[0], caches[1], kv_c, k_pe, dst, idx)
        f_args = (caches[0], caches[1], kv_c, k_pe, seq_lens, block_tables, q_start, rank, idx)

        oracle_c0, oracle_c1 = kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax(
            caches[0],
            caches[1],
            kv_c,
            k_pe,
            seq_lens,
            block_tables,
            q_start,
            rank,
            nope_spec=nope_spec,
            rope_spec=rope_spec,
            dcp_size=dcp_size,
            interleave_size=interleave,
        )
        got = kern_fn(*k_args)
        want = fb_fn(*f_args)
        exact = [
            bool(np.array_equal(np.asarray(g), np.asarray(w)))
            for g, w in zip(got, want)
        ] + [
            bool(np.array_equal(np.asarray(got[2]), np.asarray(oracle_c0))),
            bool(np.array_equal(np.asarray(got[3]), np.asarray(oracle_c1))),
        ]
        all_exact = all(exact)
        k_us, k_comp, k_fin = _time_fn(kern_fn, k_args)
        f_us, f_comp, f_fin = _time_fn(fb_fn, f_args)
        result["cases"][f"{rows}:kernel"] = dict(
            admitted=True,
            per_layer_us=k_us,
            compile_s=k_comp,
            finite=k_fin,
            rel_l2=0.0 if all_exact else 1.0,
            cos=1.0 if all_exact else 0.0,
            exact_parts=exact,
            ranks_identical=all_exact,
            nan=0,
        )
        result["cases"][f"{rows}:fallback"] = dict(
            per_layer_us=f_us,
            compile_s=f_comp,
            finite=f_fin,
        )
    print("RESULT " + json.dumps(result), flush=True)


_BENCHES = {
    "fused_norm_qkva": _bench_fused_norm_qkva,
    "lm_head_hbm": _bench_lm_head_hbm,
    "fp8_oproj_all_reduce": _bench_fp8_oproj_all_reduce,
    "mla_kv_write_in_gather": _bench_mla_kv_write_in_gather,
}


@pytest.mark.parametrize("kernel_name", list(_BENCHES))
def test_cap20_decode_kernels_by_row_count(kernel_name):
    env = dict(os.environ)
    src_dir = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = (
        f"{src_dir}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_dir
    )
    proc = subprocess.run(
        [sys.executable, __file__, "bench", kernel_name],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    print(proc.stdout[-20000:])
    print(proc.stderr[-6000:])
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    if proc.returncode != 0 and "needs a TPU host" in proc.stdout:
        pytest.skip("needs a TPU host")
    assert proc.returncode == 0 and lines, f"{kernel_name} benchmark failed"
    result = json.loads(lines[-1][len("RESULT ") :])
    for key, case in sorted(result["cases"].items()):
        if key.endswith(":kernel"):
            assert case.get("admitted") is True, (kernel_name, key, case)
            assert case["ranks_identical"] and not case["nan"], (kernel_name, key, case)
            assert case["rel_l2"] < 0.06 and case["cos"] > 0.998, (
                kernel_name,
                key,
                case,
            )


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "bench":
    import jax

    if jax.default_backend() != "tpu" or len(jax.devices()) < 2:
        print("needs a TPU host (>= 2 cores)")
        sys.exit(3)
    _BENCHES[sys.argv[2]]()
