"""K-HCA accuracy instrument: float64 oracle + stock (native) reference + fault-injection control.

  python test_accuracy.py [--cells a,b] [--candidate module:fn] [--out results.jsonl]

Per cell:
  * stock (incumbent: kv_rope -> HCA compressor -> _attention_hca, fin env) vs the f64 oracle
    -> err_max_rel (max|o-ref| / max|ref|), err_mean, per output; cache checks (new SWA rows, closed HCA rows,
    saved states vs the oracle; every other cache byte unchanged).
  * tolerance for a candidate = TOL(stock error) (see `gate`), anchored on f64 and the incumbent's own error.
  * fault control: each FAULTS entry applied to the oracle; it is CAUGHT if it fails the same gate a candidate must pass
    (attention output vs the f64 oracle, or cache rows vs stock bytes). Catch rate reported per cell class.
  * --candidate: module:fn with the stock_region signature -> also gated (outputs vs f64 at TOL, caches bit-exact vs stock).
"""
import argparse
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hca_common as C  # noqa: E402

import numpy as np  # noqa: E402

BF16_REL = 2.0 ** -8


def err(a, ref):
    a = np.asarray(a, np.float64)
    ref = np.asarray(ref, np.float64)
    d = np.abs(a - ref)
    return float(d.max() / max(np.abs(ref).max(), 1e-30)), float(d.mean())


def gate(stock_err):
    """Candidate tolerance from the incumbent's own error vs f64: 2x its max-rel error, at least +1 bf16 ulp."""
    mx, mean = stock_err
    return max(2 * mx, mx + BF16_REL), max(2 * mean, 1e-6)


def decode_caches(inp, swa, hca, st):
    from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout as CL
    import jax.numpy as jnp
    swa = np.asarray(swa)
    st = np.asarray(st)
    hca_w = np.asarray(CL.slabs_to_words(jnp.asarray(hca)))
    return swa, hca_w, st


def cache_rows(inp, swa_w, hca_w, st_w, keys_swa, keys_hca, keys_st):
    c = inp.cell
    rows = {}
    for (r, p) in keys_swa:
        pg = inp.swa_bt[r, p // C.SWA_PAGE]
        s = p % C.SWA_PAGE
        rows[("swa", r, p)] = C.words_to_tokens(swa_w[pg, 2 * s:2 * s + 2])
    for (r, k) in keys_hca:
        pg = inp.hca_bt[r, k // C.HCA_PAGE]
        s = k % C.HCA_PAGE
        rows[("hca", r, k)] = C.words_to_tokens(hca_w[pg, 2 * s:2 * s + 2])
    for (r, p) in keys_st:
        pg = inp.st_bt[r, p // inp.sbs]
        s = p % inp.sbs
        rows[("st", r, p)] = st_w[pg, 8 * s:8 * s + 8].reshape(-1).view(np.float32)
    return rows


def untouched_equal(inp, swa_w, hca_w, st_w, keys_swa, keys_hca, keys_st):
    """Every cache word outside the rows this step may write is unchanged."""
    N = inp.cell.pool
    s0 = C.scatter_pages(N, 2 * C.SWA_PAGE, inp.swa_words)
    h0 = C.scatter_pages(N, 2 * C.HCA_PAGE, inp.hca_words)
    t0 = C.scatter_pages(N, 8 * inp.sbs, inp.st_words)
    ms, mh, mt = (np.ones(a.shape[:2], bool) for a in (s0, h0, t0))
    for (r, p) in keys_swa:
        s = p % C.SWA_PAGE
        ms[inp.swa_bt[r, p // C.SWA_PAGE], 2 * s:2 * s + 2] = False
    for (r, k) in keys_hca:
        s = k % C.HCA_PAGE
        mh[inp.hca_bt[r, k // C.HCA_PAGE], 2 * s:2 * s + 2] = False
    for (r, p) in keys_st:
        s = p % inp.sbs
        mt[inp.st_bt[r, p // inp.sbs], 8 * s:8 * s + 8] = False
    return dict(swa=bool((s0 == swa_w)[ms].all()), hca=bool((h0 == hca_w)[mh].all()), st=bool((t0 == st_w)[mt].all()))


def run_cell(name, f_stock, cs_np, cs_dev, cand=None, seed=0):
    import jax
    inp = C.build_inputs(C.CELLS[name], seed=seed)
    d = C.device_inputs(inp)
    t0 = time.time()
    out, swa, hca, st = jax.block_until_ready(jax.jit(f_stock)(d, cs_dev))
    t_stock = time.time() - t0
    n_real = C.VQ * inp.cell.R
    o_stock = np.asarray(out.astype(np.float32))[:n_real]
    res = dict(cell=name, cls="boundary" if inp.cell.boundary else "plain", T=inp.cell.T, H=inp.cell.heads,
               R=inp.cell.R, ctx=list(inp.cell.ctx), stock_s=round(t_stock, 1))
    best = None
    for mode in ("floor", "pos"):
        o_ref, n_swa, n_hca, n_st = C.oracle(inp, cs_np, rope_pos_mode=mode)
        swa_w, hca_w, st_w = decode_caches(inp, swa, hca, st)
        rows = cache_rows(inp, swa_w, hca_w, st_w, n_swa, n_hca, n_st)
        hca_err = max([err(rows[("hca", r, k)], v)[0] for (r, k), v in n_hca.items()] or [0.0])
        if best is None or hca_err < best[0]:
            best = (hca_err, mode, o_ref, n_swa, n_hca, n_st, rows, (swa_w, hca_w, st_w))
    hca_err, mode, o_ref, n_swa, n_hca, n_st, rows, (swa_w, hca_w, st_w) = best
    o_ref = o_ref[:n_real]
    e_stock = err(o_stock, o_ref)
    tol = gate(e_stock)
    swa_ulp = max([float(np.max(np.abs(rows[("swa", r, p)] - C.bf16_round(v)) / np.maximum(np.abs(C.bf16_round(v)), 1e-3)))
                   for (r, p), v in n_swa.items()] or [0.0])
    st_err = max([err(rows[("st", r, p)], v)[0] for (r, p), v in n_st.items()] or [0.0])
    res.update(rope_pos_mode=mode, stock_err_max_rel=e_stock[0], stock_err_mean=e_stock[1], tol_max_rel=tol[0],
               tol_mean=tol[1], new_hca_rows=len(n_hca), hca_row_err_max_rel=hca_err, swa_row_relerr_max=swa_ulp,
               state_err_max_rel=st_err,
               untouched=untouched_equal(inp, swa_w, hca_w, st_w, n_swa, n_hca, n_st))
    # fault control: each fault applied to the oracle, judged by the candidate gate
    caught = {}
    for fname, (cls, _) in C.FAULTS.items():
        fo, f_swa, f_hca, f_st = C.oracle(inp, cs_np, rope_pos_mode=mode, faults=frozenset([fname]))
        e = err(fo[:n_real], o_ref)
        out_fail = e[0] > tol[0] or e[1] > tol[1]
        cache_fail = False
        if set(f_hca) != set(n_hca):
            cache_fail = True
        else:
            for k, v in f_hca.items():
                if err(v, rows[("hca",) + k])[0] > max(2 * hca_err, BF16_REL):
                    cache_fail = True
        caught[fname] = dict(caught=bool(out_fail or cache_fail), out_err_max_rel=e[0], out_err_mean=e[1],
                             cache_fail=cache_fail)
    res["faults"] = caught
    if cand is not None:
        co, cswa, chca, cst = jax.block_until_ready(jax.jit(cand)(d, cs_dev))
        ec = err(np.asarray(co.astype(np.float32))[:n_real], o_ref)
        res["candidate"] = dict(err_max_rel=ec[0], err_mean=ec[1], out_pass=ec[0] <= tol[0] and ec[1] <= tol[1],
                                swa_bitexact=bool(np.array_equal(np.asarray(cswa), np.asarray(swa))),
                                hca_bitexact=bool(np.array_equal(np.asarray(chca), np.asarray(hca))),
                                st_bitexact=bool(np.array_equal(np.asarray(cst), np.asarray(st))))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(C.CELLS))
    ap.add_argument("--candidate", default="")
    ap.add_argument("--out", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "."), "accuracy.jsonl"))
    a = ap.parse_args()
    C.stock_env()
    import jax
    import jax.numpy as jnp
    print(jax.devices()[0], flush=True)
    f_stock, sbs = C.stock_region()
    assert sbs == 32, sbs
    cand = None
    if a.candidate:
        m, fn = a.candidate.split(":")
        cand = getattr(importlib.import_module(m), fn)
    cs_np = C.cos_sin_np()
    cs_dev = jnp.asarray(cs_np)
    summary = {}
    with open(a.out, "a") as fo:
        for name in a.cells.split(","):
            try:
                r = run_cell(name, f_stock, cs_np, cs_dev, cand)
            except Exception as ex:  # report and continue: one bad cell must not void the rest
                import traceback
                traceback.print_exc()
                r = dict(cell=name, error=repr(ex)[:500])
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print("ACC " + json.dumps({k: v for k, v in r.items() if k != "faults"}), flush=True)
            if "faults" in r:
                print("FAULTS " + name + " " + " ".join(f"{k}={'C' if v['caught'] else 'MISS'}" for k, v in r["faults"].items()),
                      flush=True)
                for k, v in r["faults"].items():
                    summary.setdefault((k, r["cls"]), []).append(v["caught"])
    for (k, cls), v in sorted(summary.items()):
        print(f"CATCH {k:12s} class={cls:8s} {sum(v)}/{len(v)}", flush=True)


if __name__ == "__main__":
    main()
