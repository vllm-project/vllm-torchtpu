# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""One GLM-5.2 MoE layer, TP experts vs EP x TP experts, on real TPU ranks.

Runs the per-rank bodies production runs (layers/adapter/moe_tp_decode._op_body)
under shard_map on the 8 devices of one host:

  TP arm (candidate):  TP experts, USE_MOE_TP_DECODE_KERNEL decode kernel at the
                       decode buckets (4 and 8 padded onto 16 rows), the fused MoE
                       prefill kernel (#1823 / #1876) from 256 tokens.
  EP arm (baseline):   EP x TP experts (EP 4 x TP 2 here), the expert-range decode
                       kernel, and the EP grouped-matmul path with the SparseCore
                       dispatch / combine (TPU_MOE_SC_COMBINE_* from the env) + psum
                       for other token counts.

Shapes per rank are GLM-5.2's served ones: with intermediate 128 x 8 = 1024 the TP8
arm holds 256 experts x 128 columns (= TP16 at intermediate 2048) and the EP4 x TP2
arm 64 experts x 512 columns (= EP4 x TP4 at 2048); the shared expert is 128 columns
per rank in both. Only the all-reduce runs over 8 ranks instead of 16.

Weights are built in their quantized form (FP8 values and per-channel scales), so
every sharding holds the same dequantized layer exactly. Each arm is checked
against an f32 reference (rows subset for 4096 tokens) and against the other arm;
the gates follow the PRs' stated accuracy (#1823: worst row 0.055 relative L2 vs
f32; #1763: relative L2 0.045). Two fault controls run the TP arm broken (routed
scale 1.0; one rank holding the wrong TP slice) and must fail the same gate.

Run on TPU: pytest -s tests/kernels/experimental/test_tp_vs_eptp_moe_equivalence.py
A RESULT json line carries every number. (No timings: a host-timed loop of an
8-device program is dominated by dispatch jitter; the mirror microbenchmark
measures the layers.)
"""

import dataclasses
import json
import math
import os
import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

N = 8
E = 256
H = 6144
BLK = 128
I_FULL = BLK * N
FS_FULL = BLK * N
TOPK = 8
SCALE = 2.5
EP = 4
TP_IN_EP = N // EP
NB = I_FULL // BLK
F8 = jnp.float8_e4m3fn
DECODE = [(4, 4), (8, 7), (16, 13), (32, 27), (64, 57), (128, 117)]
# Row counts at which each arm's kernel is also checked against the psum path
# it replaced (_tp_chain + lax.psum, the same op body's fallback).
VS_CHAIN = (64, 128)
PREFILL = [(4096, 4000)]
REF_ROWS = 256
TOL = {"rel_l2": 0.06, "worst_row": 0.08, "tp_vs_ep": 0.06}


def _on_tpu():
    try:
        return jax.devices()[0].platform == "tpu" and len(jax.devices()) >= N
    except Exception:  # noqa: BLE001
        return False


def _q(key, shape):
    return jnp.clip(jax.random.normal(key, shape, jnp.float32) * 48, -448, 448).astype(
        F8
    )


class _Layer:
    """The global quantized layer, generated block by block (128 intermediate
    columns) on device 0, and its per-rank shards for either sharding."""

    def __init__(self, seed=1234):
        root = jax.random.key(seed)
        fk = lambda i: jax.random.fold_in(root, i)  # noqa: E731
        d0 = jax.devices()[0]
        with jax.default_device(d0):
            self.gate = [_q(fk(1000 + b), (E, H, BLK)) for b in range(NB)]
            self.up = [_q(fk(2000 + b), (E, H, BLK)) for b in range(NB)]
            self.w2 = [_q(fk(3000 + b), (E, BLK, H)) for b in range(NB)]
            self.sgate = [_q(fk(4000 + b), (H, BLK)) for b in range(N)]
            self.sup = [_q(fk(5000 + b), (H, BLK)) for b in range(N)]
            self.sw2 = [_q(fk(6000 + b), (BLK, H)) for b in range(N)]
            u = lambda i, s: jax.random.uniform(fk(i), s, jnp.float32, 0.5, 1.5)  # noqa: E731
            self.s13 = u(7, (E, 2, I_FULL)) / (48 * math.sqrt(H))
            self.s2 = u(8, (E, H)) / (48 * math.sqrt(I_FULL))
            self.ss13 = u(9, (2, FS_FULL)) / (48 * math.sqrt(H))
            self.ss2 = u(10, (H,)) / (48 * math.sqrt(FS_FULL))
            self.router = jax.random.normal(fk(11), (E, H), jnp.float32) / math.sqrt(H)
            self.bias = jax.random.uniform(fk(12), (E,), jnp.float32, -0.05, 0.05)
            self.xkey = fk(13)

    def shared(self, r):
        cols = slice(r * BLK, (r + 1) * BLK)
        return (
            jnp.concatenate([self.sgate[r], self.sup[r]], -1),
            jnp.concatenate([self.ss13[0, cols], self.ss13[1, cols]]),
            self.sw2[r],
            self.ss2,
        )

    def tp_shard(self, r, bad_block=None):
        b = r if bad_block is None else bad_block
        cols = slice(b * BLK, (b + 1) * BLK)
        w13 = jnp.concatenate([self.gate[b], self.up[b]], -1)
        s13 = jnp.concatenate([self.s13[:, 0, cols], self.s13[:, 1, cols]], -1)
        return (
            w13,
            self.w2[b],
            s13.reshape(E, 1, 1, 2 * BLK),
            self.s2.reshape(E, 1, 1, H),
            *self.shared(r),
        ), 0

    def ep_shard(self, r):
        g, t = divmod(r, TP_IN_EP)
        le = E // EP
        ex = slice(g * le, (g + 1) * le)
        nbl = NB // TP_IN_EP
        blocks = range(t * nbl, (t + 1) * nbl)
        cols = slice(t * nbl * BLK, (t + 1) * nbl * BLK)
        w13 = jnp.concatenate(
            [self.gate[b][ex] for b in blocks] + [self.up[b][ex] for b in blocks], -1
        )
        w2 = jnp.concatenate([self.w2[b][ex] for b in blocks], axis=1)
        s13 = jnp.concatenate([self.s13[ex, 0, cols], self.s13[ex, 1, cols]], -1)
        i = nbl * BLK
        return (
            w13,
            w2,
            s13.reshape(le, 1, 1, 2 * i),
            self.s2[ex].reshape(le, 1, 1, H),
            *self.shared(r),
        ), g * le

    def inputs(self, rows):
        x = jax.random.normal(jax.random.fold_in(self.xkey, rows), (rows, H))
        x = x.astype(jnp.bfloat16)
        scores = jax.nn.sigmoid(x.astype(jnp.float32) @ self.router.T)
        return x, scores

    def reference(self, x, scores):
        """f32 shared + 2.5 x routed of the whole (unsharded) layer."""
        f32 = jnp.float32
        hp = jax.lax.Precision.HIGHEST
        xf = x.astype(f32)
        _, ids = lax.top_k(scores + self.bias, TOPK)
        w = jnp.take_along_axis(scores, ids, axis=-1)
        w = w / jnp.sum(w, axis=-1, keepdims=True)
        coef = jnp.sum(jax.nn.one_hot(ids, E, dtype=f32) * w[..., None], axis=1)
        routed = jnp.zeros(xf.shape, f32)
        for b in range(NB):
            cols = slice(b * BLK, (b + 1) * BLK)
            g = jnp.einsum(
                "nh,ehk->nek",
                xf,
                self.gate[b].astype(f32) * self.s13[:, None, 0, cols],
                precision=hp,
            )
            u = jnp.einsum(
                "nh,ehk->nek",
                xf,
                self.up[b].astype(f32) * self.s13[:, None, 1, cols],
                precision=hp,
            )
            act = jax.nn.silu(g) * u * coef[..., None]
            routed = routed + jnp.einsum(
                "nek,ekh->nh",
                act,
                self.w2[b].astype(f32) * self.s2[:, None, :],
                precision=hp,
            )
        shared = jnp.zeros(xf.shape, f32)
        for r in range(N):
            sw13, ss13, sw2, ss2 = self.shared(r)
            gu = jnp.dot(xf, sw13.astype(f32) * ss13, precision=hp)
            a = jax.nn.silu(gu[:, :BLK]) * gu[:, BLK:]
            shared = shared + jnp.dot(a, sw2.astype(f32) * ss2, precision=hp)
        return np.asarray(shared + SCALE * routed, np.float64)


def _stack(mesh, shards):
    devs = list(mesh.devices.flat)
    out = []
    for parts in zip(*shards, strict=True):
        local = [jax.device_put(p, d) for p, d in zip(parts, devs, strict=True)]
        shape = (sum(p.shape[0] for p in parts),) + parts[0].shape[1:]
        out.append(
            jax.make_array_from_single_device_arrays(
                shape, NamedSharding(mesh, P("d")), local
            )
        )
    return out


def _metrics(got, want):
    d = got - want
    row = np.linalg.norm(d, axis=-1) / np.maximum(np.linalg.norm(want, axis=-1), 1e-30)
    return {
        "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(want)),
        "worst_row": float(row.max()),
        "median_row": float(np.median(row)),
        "max_abs": float(np.abs(d).max()),
    }


def _passes(m):
    return m["rel_l2"] <= TOL["rel_l2"] and m["worst_row"] <= TOL["worst_row"]


def _run_all():
    from vllm_torchtpu.layers.adapter import moe_tp_decode as adapter
    from vllm_torchtpu.layers.core.fused_moe_gmm import resolve_onehot_permute_threshold

    mesh = Mesh(np.asarray(jax.devices()[:N]), ("d",))
    layer = _Layer()
    common = dict(
        topk=TOPK,
        activation="silu",
        use_sparse_core=os.environ.get("USE_MOE_SPARSE_CORE", "1") == "1",
        onehot_threshold=resolve_onehot_permute_threshold(),
        router_bias=os.environ.get("TPU_MOE_ROUTER_TOPK_BIAS", "0") == "1",
    )
    st_tp = adapter._OpStatics(
        routed_scale=SCALE, use_ep=False, prefill_min_tokens=256, **common
    )
    st_ep = adapter._OpStatics(
        routed_scale=SCALE, use_ep=True, prefill_min_tokens=None, **common
    )
    st_tp_bad_scale = adapter._OpStatics(
        routed_scale=1.0, use_ep=False, prefill_min_tokens=256, **common
    )
    specs = (P("d"), P("d"), P(), P()) + (P("d"),) * 10

    def weights(kind):
        shards, e0 = [], []
        for r in range(N):
            if kind == "tp":
                w, s = layer.tp_shard(r)
            elif kind == "tp_bad_slice":
                # Rank 3 holds rank 4's TP slice of every routed expert.
                w, s = layer.tp_shard(r, bad_block=4 if r == 3 else None)
            else:
                w, s = layer.ep_shard(r)
            shards.append(w)
            e0.append(s)
        stacked = _stack(mesh, shards)
        rank = jax.device_put(
            jnp.arange(N, dtype=jnp.int32).reshape(N, 1), NamedSharding(mesh, P("d"))
        )
        e0 = jax.device_put(
            jnp.asarray(e0, jnp.int32).reshape(N, 1), NamedSharding(mesh, P("d"))
        )
        return stacked, rank, e0

    def program(statics):
        body = adapter._op_body(statics, N, "d")
        return jax.jit(
            jax.shard_map(
                body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
            )
        )

    def chain_program(statics):
        # The op body's fallback at any token count: this rank's partial from
        # the usual path, psummed (what an unadmitted bucket runs).
        def body(x, scores, b, pad_mask, *rest):
            *w, _rank, e0 = rest
            is_pad = pad_mask[: x.shape[0]]
            partial = adapter._tp_chain(
                x, scores, b, is_pad, *w, statics, e0.reshape(())
            )
            return lax.psum(partial, "d")

        return jax.jit(
            jax.shard_map(
                body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
            )
        )

    def operands(rows, n_valid, w):
        stacked, rank, e0 = w
        x, scores = layer.inputs(rows)
        rep = NamedSharding(mesh, P())
        per = NamedSharding(mesh, P("d"))
        pad_mask = jnp.arange(rows) >= n_valid
        ops = (
            jax.device_put(jnp.tile(x, (N, 1)), per),
            jax.device_put(jnp.tile(scores, (N, 1)), per),
            jax.device_put(layer.bias, rep),
            jax.device_put(pad_mask, rep),
            *stacked,
            rank,
            e0,
        )
        return ops, x, scores

    def run(prog, ops, rows):
        out = np.asarray(jax.block_until_ready(prog(*ops))).astype(np.float32)
        out = out.reshape(N, rows, H)
        bits = out.view(np.uint32)
        return out, bool((bits == bits[0]).all())

    def kernels_in(prog, ops):
        txt = prog.lower(*ops).as_text()
        return {
            "moe_tp_decode": txt.count("moe_tp_decode-e_"),
            "fused_moe_prefill": txt.count("fused_moe_prefill-t_"),
            "ragged_or_gmm": txt.count("gmm") + txt.count("ragged"),
        }

    w_tp, w_ep, w_bad = weights("tp"), weights("ep"), weights("tp_bad_slice")
    res = {"tol": TOL, "cases": {}, "controls": {}}
    prog_tp, prog_ep, prog_bs = program(st_tp), program(st_ep), program(st_tp_bad_scale)
    chain_tp, chain_ep = chain_program(st_tp), chain_program(st_ep)
    # TPU_MOE_TP_DECODE_FAST (served on): the same output bits.
    fast_tp = program(dataclasses.replace(st_tp, fast=True))
    fast_ep = program(dataclasses.replace(st_ep, fast=True))
    for rows, n_valid in DECODE + PREFILL:
        ops_tp, x, scores = operands(rows, n_valid, w_tp)
        ops_ep, _, _ = operands(rows, n_valid, w_ep)
        t0 = time.monotonic()
        out_tp, same_tp = run(prog_tp, ops_tp, rows)
        out_ep, same_ep = run(prog_ep, ops_ep, rows)
        idx = np.arange(n_valid)
        if n_valid > REF_ROWS:
            rng = np.random.default_rng(0)
            idx = np.sort(
                np.concatenate(
                    [
                        np.arange(64),
                        rng.choice(np.arange(64, n_valid), REF_ROWS - 64, False),
                    ]
                )
            )
        want = layer.reference(x[idx], scores[idx])
        a = out_tp[0, idx].astype(np.float64)
        b = out_ep[0, idx].astype(np.float64)
        pad_tp = out_tp[:, n_valid:]
        case = {
            "tp_vs_ref": _metrics(a, want),
            "ep_vs_ref": _metrics(b, want),
            "tp_vs_ep_all_rows": _metrics(
                out_tp[0, :n_valid].astype(np.float64),
                out_ep[0, :n_valid].astype(np.float64),
            ),
            "ranks_identical": {"tp": same_tp, "ep": same_ep},
            # The decode kernel writes +0 past n_valid; the prefill kernel and
            # the psum path still run the shared expert on padded rows.
            "tp_padding_rows_zero": bool(rows >= 256 or np.all(pad_tp == 0)),
            "nan": int(np.isnan(out_tp).sum() + np.isnan(out_ep).sum()),
            "kernels_tp": kernels_in(prog_tp, ops_tp),
            "kernels_ep": kernels_in(prog_ep, ops_ep),
            "seconds": round(time.monotonic() - t0, 1),
        }
        if rows in VS_CHAIN:
            # Fused kernel vs the unfused path on the same rows (valid rows;
            # the psum path runs the shared expert on padded rows).
            for arm, prog, ops, out in (
                ("tp", chain_tp, ops_tp, out_tp),
                ("ep", chain_ep, ops_ep, out_ep),
            ):
                fast_prog = fast_tp if arm == "tp" else fast_ep
                fast_out, _ = run(fast_prog, ops, rows)
                case[f"{arm}_fast_bitwise"] = bool(
                    np.array_equal(fast_out.view(np.uint32), out.view(np.uint32))
                )
                ref, same = run(prog, ops, rows)
                got = out[0, :n_valid]
                want_c = ref[0, :n_valid]
                case[f"{arm}_kernel_vs_chain"] = dict(
                    _metrics(got.astype(np.float64), want_c.astype(np.float64)),
                    bitwise=bool(
                        np.array_equal(got.view(np.uint32), want_c.view(np.uint32))
                    ),
                    chain_ranks_identical=same,
                    chain_vs_ref=_metrics(ref[0, idx].astype(np.float64), want),
                    chain_kernels=kernels_in(prog, ops),
                )
        res["cases"][str(rows)] = case
        print("CASE", rows, json.dumps(case), flush=True)
        if rows in (16, 4096):
            bad, _ = run(prog_bs, ops_tp, rows)
            res["controls"][f"routed_scale_1.0@{rows}"] = _metrics(
                bad[0, idx].astype(np.float64), want
            )
            ops_bad, _, _ = operands(rows, n_valid, w_bad)
            bad, _ = run(prog_tp, ops_bad, rows)
            res["controls"][f"rank3_wrong_tp_slice@{rows}"] = _metrics(
                bad[0, idx].astype(np.float64), want
            )
    res["selftest"] = _selftest_all_buckets(mesh)
    print("RESULT " + json.dumps(res), flush=True)
    return res


def _selftest_all_buckets(mesh):
    """Arming's all-reduce self-test (all_reduce.standalone over every row path
    of selftest_row_counts) at every ROW_BUCKETS entry, on the real ranks:
    {"rows": [failing n_valid, ...]}."""
    from vllm_torchtpu.kernels.experimental.moe_tp_decode import (
        ROW_BUCKETS,
        all_reduce,
    )

    per = NamedSharding(mesh, P("d"))
    rank = jax.device_put(jnp.arange(N, dtype=jnp.int32).reshape(N, 1), per)
    out = {}
    for m in ROW_BUCKETS:
        spec = all_reduce.make_spec(N, m, H)
        call = all_reduce.standalone(spec)

        def body(x, r, nv, call=call):
            meta = jnp.stack([nv.reshape(()), r.reshape(())]).astype(jnp.int32)
            return call(meta, x)

        prog = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("d"), P("d"), P()),
                out_specs=P("d"),
                check_vma=False,
            )
        )
        x = all_reduce.selftest_inputs(spec)
        xs = jax.device_put(jnp.asarray(x.reshape(N * m, -1), jnp.bfloat16), per)
        bad = []
        for nv in all_reduce.selftest_row_counts(spec):
            got = np.asarray(
                jax.block_until_ready(prog(xs, rank, jnp.asarray([nv], jnp.int32)))
            ).astype(np.float32)
            got = got.reshape(N, m, -1)
            if not all(all_reduce.selftest_ok(spec, x, got[r], nv) for r in range(N)):
                bad.append(nv)
        out[str(m)] = {"row_counts": all_reduce.selftest_row_counts(spec), "bad": bad}
    return out


@pytest.fixture(scope="module")
def result():
    # Not at import: a JAX backend initialized during collection claims the TPU.
    if not _on_tpu():
        pytest.skip(f"needs {N} TPU devices")
    return _run_all()


@pytest.mark.parametrize("rows", [r for r, _ in DECODE + PREFILL])
def test_tp_and_ep_match_the_f32_reference(result, rows):
    c = result["cases"][str(rows)]
    assert _passes(c["tp_vs_ref"]), c["tp_vs_ref"]
    assert _passes(c["ep_vs_ref"]), c["ep_vs_ref"]
    assert c["tp_vs_ep_all_rows"]["rel_l2"] <= TOL["tp_vs_ep"], c["tp_vs_ep_all_rows"]
    assert c["ranks_identical"]["tp"] and c["ranks_identical"]["ep"]
    assert c["tp_padding_rows_zero"] and c["nan"] == 0


@pytest.mark.parametrize("rows", [r for r, _ in DECODE + PREFILL])
def test_the_tp_arm_runs_its_kernels(result, rows):
    k = result["cases"][str(rows)]["kernels_tp"]
    if rows >= 256:
        assert k["fused_moe_prefill"] > 0 and k["moe_tp_decode"] == 0, k
    else:
        assert k["moe_tp_decode"] > 0 and k["fused_moe_prefill"] == 0, k


@pytest.mark.parametrize("rows", [r for r, _ in DECODE])
def test_the_ep_arm_runs_its_kernel_at_decode_buckets(result, rows):
    # Every decode bucket up to the kernel's MAX_ROWS (128) runs the
    # expert-range kernel, none falls back to the psum path.
    k = result["cases"][str(rows)]["kernels_ep"]
    assert k["moe_tp_decode"] > 0, k


@pytest.mark.parametrize("rows", VS_CHAIN)
def test_kernel_matches_the_psum_path_it_replaces(result, rows):
    c = result["cases"][str(rows)]
    for arm in ("tp", "ep"):
        v = c[f"{arm}_kernel_vs_chain"]
        assert c[f"{arm}_fast_bitwise"], arm
        assert v["chain_kernels"]["moe_tp_decode"] == 0, v
        assert v["chain_ranks_identical"], v
        assert _passes(v["chain_vs_ref"]), v
        assert v["rel_l2"] <= TOL["tp_vs_ep"], v


def test_all_reduce_selftest_at_every_bucket(result):
    st = result["selftest"]
    assert "128" in st, st
    assert all(not v["bad"] for v in st.values()), st


def test_fault_controls_fail_the_gate(result):
    assert len(result["controls"]) == 4
    for name, m in result["controls"].items():
        assert not _passes(m), (name, m)
