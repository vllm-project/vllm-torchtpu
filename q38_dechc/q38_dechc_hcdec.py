#!/usr/bin/env python3
"""q38_dechc_hcdec.py (DEC-HC, plan v4; ported from DS-O1S's q38_dso1s_hcdec.py sha256 7e12570d): one-device
exactness gate and microbench of the decode hyper-connection (HC) mix at the served decode row counts, TPU v7x: the
served unfused path against the fused decode kernel (TPU_QWEN4_HC_DECODE_FUSED=1: kernels/qwen4_exp/hc_decode.py, one
Pallas call per mix, the attention combine fused ahead of the MLP mix) and the fused kernel reading the up weight as up.T
(+ TPU_QWEN4_HC_DECODE_UP_T=1, DEC-HC).

The up weight's layout is the served one: the served programs hold the bf16 [10240, 320] up weight in the {0,1} tiled
layout (M: j-b91368df O1 B64 HLO, copy.weight_ref = bf16[10240,320]{1,0} copy(bf16[10240,320]{0,1} ...)), i.e. as a
row-major [320, 10240] array. So every arm here holds up_t = a row-major [320, 10240] array and uses up = up_t.T: the
'fused' arm then pays the served relayout copy before each call and 'fusedT' should not (relayout_ops per arm).

Shapes (Qwen3.8 Flash, TP-replicated HC, so per-rank = full): 4 streams x hidden 2560 (W = 10240), low rank 320.
A step is LAYERS chained decoder layers: attn mix -> stand-in block -> attn combine -> mlp mix -> stand-in block ->
mlp combine. The stand-in block is one bf16 [2560, 2560] matmul, the same in every arm; a 'standin' arm times it alone.
Arms per row count (16, 64, 112, 128, 192, 256 = 4 verify rows x B4 / B16 / B28 / B32 / B48 / B64, the served buckets
with TPU_TOKEN_BUCKET_EXTRA=48:80:96:112:160:192:224):
  base    the served unfused math (GatedResidual.mix / combine, models/vllm/qwen4_exp/hyperconnection.py): per-branch
          norm bf16(f32(x) * rsqrt(mean(x^2) + eps) * (1 + f32(w))), gate bf16(silu(f32(bf16(norm @ down^T) / hc))),
          mixed bf16(sum_b(sigmoid(f32(bf16(gate @ up_b^T))) * f32(norm_b)) / hc), injection 2 sigmoid(f32(bf16(norm @
          inject^T)) / hc), combine bf16(f32(residual) + f32(block) * injection_b)
  fused   hc_decode_mix for the attn mix (with the injection), hc_decode_mix with the combine prologue for the mlp mix,
          XLA combine after the mlp block (the served wiring: GatedResidual.mix / combine_mix)
  fusedT  fused with up_t=True (TPU_QWEN4_HC_DECODE_UP_T)
  mutant  fusedT with the attn injection forced to 1.0: must differ in hash and FAIL the float64 rule (else VOID)
Per arm: median wall ms per step over --iters, output hash, engagement (fused: qwen4exp_hc_decode without _upt;
fusedT/mutant: qwen4exp_hc_decode_upt; base: neither, in the compiled program), relayout_ops (copy/transpose/fusion ops whose
result is bf16[10240,320] or bf16[320,10240] in the compiled program), and the float64 2x rule on every row (rows are
independent): ||arm - f64|| <= 2 ||base - f64|| for RMS (relative) and max. Writes rows.jsonl and result.json in --out
(the bench recipe's BENCH=script contract: ok and cases). ok = every arm ran, fused and fusedT engaged and passing the
rule at every row count, base not engaged, every mutant biting. mech_ok (reported, not in ok) = fusedT has 0
relayout_ops and fused has >= 1 at every row count.
Q38_DECHC_HCDEC_DRY=1: CPU, Pallas interpret mode, the same real shapes, few iterations (the pre-launch CPU pass).
  PYTHONPATH=src python3 q38_dechc/q38_dechc_hcdec.py [--rows ...] [--iters 50] [--layers 12]
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import traceback

DRY = os.environ.get("Q38_DECHC_HCDEC_DRY") == "1"
# A materialized up weight: a copy, transpose or fusion whose result is the [10240, 320] / [320, 10240] matrix
# (a bitcast is free and does not count).
RELAYOUT_RE = re.compile(r"=\s*bf16\[(?:10240,320|320,10240)\]\{[^}]*\}\s+(?:copy|transpose|fusion|copy-start)\(")
BR, HID, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BR * HID


def _hash(x):
    import numpy as np

    return hashlib.sha1(np.ascontiguousarray(np.asarray(x)).view(np.uint8).tobytes()).hexdigest()[:16]


def weights(layers, seed=2026):
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)

    def n(shape, s):
        return jnp.asarray(rng.normal(size=shape) * s, jnp.bfloat16)

    # up_t: the up weight as a row-major [lowrank, width] array, i.e. the served bytes; every arm uses up_t.T.
    hcs = [[dict(w=n((WIDTH,), 0.3), down=n((LOWRANK, WIDTH), 0.02), up_t=n((LOWRANK, WIDTH), 0.05),
                 inject=n((BR, WIDTH), 0.02)) for _ in range(2)] for _ in range(layers)]
    return hcs, n((HID, HID), HID ** -0.5), rng


def make_step(arm, interpret):
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix

    ip = dict(interpret=interpret) if interpret is not None else {}
    kw = dict(branches=BR, hidden=HID, eps=EPS, **ip)

    def linear(a, w):  # torch F.linear(bf16, bf16): f32 accumulation, bf16 result
        return jnp.dot(a, w.T, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

    def combine(residual, block, inj):  # GatedResidual.combine (flat layout)
        r, b = residual.astype(jnp.float32), block.astype(jnp.float32)
        return jnp.concatenate([r[:, i * HID:(i + 1) * HID] + b * inj[:, i:i + 1] for i in range(BR)],
                               axis=-1).astype(jnp.bfloat16)

    def base_mix(x, p):
        v = x.astype(jnp.float32).reshape(-1, BR, HID)
        n = v * jax.lax.rsqrt(jnp.mean(v * v, -1, keepdims=True) + EPS)
        norm = (n.reshape(-1, WIDTH) * (1.0 + p["w"].astype(jnp.float32))).astype(jnp.bfloat16)
        gate = (linear(norm, p["down"]).astype(jnp.float32) / BR).astype(jnp.bfloat16)
        gate = jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16)
        wt = jax.nn.sigmoid(linear(gate, p["up_t"].T).astype(jnp.float32)) * norm.astype(jnp.float32)
        mixed = wt[:, :HID]
        for b in range(1, BR):
            mixed = mixed + wt[:, b * HID:(b + 1) * HID]
        mixed = (mixed / BR).astype(jnp.bfloat16)
        inj = 2.0 * jax.nn.sigmoid(linear(norm, p["inject"]).astype(jnp.float32) / BR)
        return mixed, inj

    def step(x, layers, wb):
        for attn, mlp in layers:
            if arm == "standin":
                x = x.at[:, :HID].set(linear(linear(x[:, :HID], wb), wb))
            elif arm == "base":
                mixed, inj = base_mix(x, attn)
                h = combine(x, linear(mixed, wb), inj)
                mixed2, inj2 = base_mix(h, mlp)
                x = combine(h, linear(mixed2, wb), inj2)
            else:  # fused / fusedT / mutant (= fusedT with the attn injection forced to 1.0)
                kt = dict(kw, up_t=arm in ("fusedT", "mutant"))
                mixed, inj = hc_decode_mix(x, attn["w"], attn["down"], attn["up_t"].T, attn["inject"], **kt)
                if arm == "mutant":
                    inj = jnp.ones_like(inj)
                mixed2, inj2, h = hc_decode_mix(x, mlp["w"], mlp["down"], mlp["up_t"].T, mlp["inject"],
                                                block=linear(mixed, wb), prev_injection=inj, **kt)
                x = combine(h, linear(mixed2, wb), inj2)
        return x

    return jax.jit(step)


def reference_f64(x, layers, wb):
    """The exact HC math in float64 (no intermediate rounding)."""
    import numpy as np

    def f(a):
        return np.asarray(a, np.float64)

    x, wb, rows = f(x), f(wb), x.shape[0]
    for attn, mlp in layers:
        for p in (attn, mlp):
            v = x.reshape(rows, BR, HID)
            n = v / np.sqrt(np.mean(v * v, -1, keepdims=True) + EPS)
            norm = (n.reshape(rows, WIDTH) * (1.0 + f(p["w"])))
            g = norm @ f(p["down"]).T / BR
            g = g / (1.0 + np.exp(-g))
            lg = g @ f(p["up_t"])
            mixed = ((1.0 / (1.0 + np.exp(-lg))) * norm).reshape(rows, BR, HID).sum(1) / BR
            inj = 2.0 / (1.0 + np.exp(-(norm @ f(p["inject"]).T) / BR))
            block = mixed @ wb.T
            x = (x.reshape(rows, BR, HID) + block[:, None, :] * inj[:, :, None]).reshape(rows, WIDTH)
    return x


def run(a):
    import jax
    import jax.numpy as jnp
    import numpy as np

    os.makedirs(a.out, exist_ok=True)
    rowf = open(os.path.join(a.out, "rows.jsonl"), "w")

    def emit(row):
        rowf.write(json.dumps(row) + "\n")
        rowf.flush()
        print(json.dumps(row), flush=True)

    interpret = None
    if DRY:
        from jax.experimental.pallas import tpu as pltpu

        interpret = pltpu.InterpretParams()
    else:
        assert jax.default_backend() == "tpu", jax.default_backend()
    emit(dict(event="start", device=str(jax.devices()[0]), jax=jax.__version__, dry=DRY, rows=a.rows,
              layers=a.layers, iters=a.iters))
    layers, wb, rng = weights(a.layers)
    f32 = jnp.float32
    lw = [[{k: np.asarray(v.astype(f32)) for k, v in p.items()} for p in l] for l in layers]
    wbn = np.asarray(wb.astype(f32))
    rows_out, judges = [], []
    for T in a.rows:
        x = jnp.asarray(rng.normal(size=(T, WIDTH)), jnp.bfloat16)
        ref = reference_f64(np.asarray(x.astype(f32)), lw, wbn)
        res = {}
        for arm in ("standin", "base", "fused", "fusedT", "mutant"):
            row = dict(rows=T, arm=arm, layers=a.layers)
            try:
                fn = make_step(arm, interpret)
                args = (x, layers, wb)
                t0 = time.time()
                txt = ""
                if not DRY:
                    lowered = fn.lower(*args)
                    compiled = lowered.compile()
                    txt = lowered.as_text() + "\n" + compiled.as_text()
                    fn = compiled
                out = jax.block_until_ready(fn(*args))
                row["compile_s"] = round(time.time() - t0, 1)
                samples = []
                for _ in range(a.iters):
                    t0 = time.perf_counter()
                    out = jax.block_until_ready(fn(*args))
                    samples.append((time.perf_counter() - t0) * 1e3)
                row["ms"] = round(statistics.median(samples), 4)
                row["ms_p10"] = round(sorted(samples)[int(0.1 * (len(samples) - 1))], 4)
                if not DRY:
                    names = set(re.findall(r"qwen4exp_hc_decode[a-z_]*", txt))
                    upt = {n for n in names if n.endswith("_upt")}
                    if arm == "fused":
                        row["engaged"] = bool(names) and not upt
                    elif arm in ("fusedT", "mutant"):
                        row["engaged"] = bool(upt) and upt == names
                    else:
                        row["engaged"] = not names
                    row["kernels"] = sorted(names)
                    row["relayout_ops"] = len(RELAYOUT_RE.findall(compiled.as_text()))
                o = np.asarray(out.astype(f32), np.float64)
                row["finite"] = bool(np.isfinite(o).all())
                if arm != "standin":
                    row["f64_err"] = dict(rms=float(np.linalg.norm(o - ref) / np.linalg.norm(ref)),
                                          max=float(np.abs(o - ref).max()))
                row["out_hash"] = _hash(out)
                res[arm] = row
            except Exception as e:  # noqa: BLE001
                row.update(error=repr(e)[:400], trace=traceback.format_exc()[-1500:])
            emit(row)
            rows_out.append(row)
        b = res.get("base", {}).get("f64_err")
        sb = res.get("standin", {}).get("ms")
        for arm in ("base", "fused", "fusedT", "mutant"):
            r = res.get(arm)
            if not r:
                continue
            j = dict(judge=True, rows=T, arm=arm)
            if sb is not None and r.get("ms") is not None:
                j["hc_ms_per_step"] = round(r["ms"] - sb, 4)
                j["hc_us_per_mix"] = round((r["ms"] - sb) * 1e3 / (2 * a.layers), 2)
            if b and r.get("f64_err") and arm != "base":
                j["f64_2x_rule"] = bool(r["f64_err"]["rms"] <= 2 * b["rms"] and r["f64_err"]["max"] <= 2 * b["max"])
                j["err_ratio_rms"] = round(r["f64_err"]["rms"] / b["rms"], 3) if b["rms"] else None
                j["bitwise_vs_base"] = r["out_hash"] == res["base"]["out_hash"]
            if arm == "mutant":
                j["mutant_bites"] = bool(j.get("f64_2x_rule") is False and r["out_hash"] != res.get("fusedT", {}).get("out_hash"))
            if "relayout_ops" in r:
                j["relayout_ops"] = r["relayout_ops"]
            emit(j)
            judges.append(j)
    rowf.close()
    errors = [dict(rows=r["rows"], arm=r["arm"], error=r["error"][:200]) for r in rows_out if r.get("error")]
    not_engaged = [dict(rows=r["rows"], arm=r["arm"]) for r in rows_out if r.get("engaged") is False]
    fused_fail = [j for j in judges if j["arm"] in ("fused", "fusedT") and j.get("f64_2x_rule") is not True]
    rl = {(r["rows"], r["arm"]): r.get("relayout_ops") for r in rows_out if "relayout_ops" in r}
    mech_ok = None if DRY else bool(rl) and all(
        rl.get((T, "fusedT")) == 0 and (rl.get((T, "fused")) or 0) >= 1 for T in a.rows)
    weak = [j for j in judges if j["arm"] == "mutant" and not j.get("mutant_bites")]
    nonfinite = [dict(rows=r["rows"], arm=r["arm"]) for r in rows_out if r.get("finite") is False and r["arm"] != "mutant"]
    ok = not errors and not not_engaged and not fused_fail and not weak and not nonfinite and bool(judges)
    summary = dict(ok=ok, mech_ok=mech_ok, bench="q38-dechc-hcdec", dry=DRY, errors=errors, not_engaged=not_engaged,
                   fused_rule_failures=fused_fail, weak_mutants=weak, nonfinite=nonfinite, judges=judges,
                   cases=[dict(ctx=0, new=r["rows"], reqs=r["rows"] // 4, choice=r["arm"], ms=r.get("ms"))
                          for r in rows_out if r.get("ms") is not None])
    json.dump(summary, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print("==> Q38_DECHC_HCDEC_DONE ok=%s mech_ok=%s errors=%d not_engaged=%d fused_rule_failures=%d weak_mutants=%d"
          % (ok, mech_ok, len(errors), len(not_engaged), len(fused_fail), len(weak)), flush=True)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"),
                                                 "q38f-bench", "script", "run01_A"))
    p.add_argument("--rows", type=int, nargs="+", default=[16, 64, 112, 128, 192, 256])
    p.add_argument("--iters", type=int, default=3 if DRY else 50)
    p.add_argument("--layers", type=int, default=2 if DRY else 12)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
