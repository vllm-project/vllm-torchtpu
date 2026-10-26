"""K1v2 accuracy gate: float64 oracle, native reference (incumbent), fault control per cell class, poison control.

  python test_accuracy.py [--cells a,b] [--arms inc,udx,udx1,mod:fn] [--faults all|none]

Per cell: native err = incumbent vs f64 (max-rel over valid rows). Every arm is judged by gate(): max-rel err <=
max(2 x native, native + 2^-8). Faults (k1v2_common.FAULTS) are outputs a buggy kernel of that class would produce,
judged by the same gate -> catch rate per cell class (a fault the gate passes on a production-class cell = BLIND).
Poison: cache rows that no valid top-k entry references are 0xFF bytes; every arm must stay finite and within gate.
"""
import argparse, importlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import k1v2_common as C

ACC_CELLS = ["c4T8-mid", "c8T16-mid", "c8T32-first", "c16T16-mid", "c16T32-u90", "c16T8-last", "c8T32-ragged",
             "c16T32-ragged", "c16m512-dec", "c16m512-pf32k", "edge-short", "edge-tiny", "edge-pageb", "edge-maxb", "edge-sink",
             "edge-maxb8", "pfw-c8T16-short", "pfw-c16T32-short", "pfw-c8T32-mid"]


def cls(name):
    if name.startswith("pfw"):
        return "prefill-window"
    if name.startswith("edge"):
        return "edge"
    if "m512" in name:
        return "mixed"
    if "ragged" in name:
        return "ragged"
    return "production"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(ACC_CELLS))
    ap.add_argument("--arms", default="inc,udx,udx1")
    ap.add_argument("--faults", default="all")
    ap.add_argument("--no-poison", action="store_true")
    a = ap.parse_args()
    import jax, jax.numpy as jnp
    print(jax.devices()[0], jax.__version__, flush=True)
    catch = {}
    summary = []
    for name in a.cells.split(","):
        cell = C.CELLS[name]
        t0 = time.time()
        inp = C.build_inputs(cell)
        print(json.dumps(dict(cell=name, drivers=C.drivers(inp), ctx=inp["ctx"], U_target=inp["U_target"])), flush=True)
        arms = {}
        for arm in a.arms.split(","):
            if arm in C.ARMS:
                arms[arm] = C.ARMS[arm](cell.conc)
            else:
                m, fn = arm.split(":")
                arms[arm] = getattr(importlib.import_module(m), fn)(cell.conc)
        for poison in ([False] if a.no_poison else [False, True]):
            d, deq = C.device_inputs(inp, cache="quant", poison=poison)
            want, rows = C.oracle(inp, d, deq)
            ref_err = None
            for arm, f in arms.items():
                try:
                    out = np.asarray(jax.jit(f)(d, d["q"]).astype(jnp.float32))
                except Exception as e:  # noqa
                    rec = dict(cell=name, cls=cls(name), arm=arm, poison=poison, error=repr(e)[:800])
                    print("ACC " + json.dumps(rec), flush=True); summary.append(rec); continue
                o = out[rows]
                finite = bool(np.isfinite(o).all())
                err, ok, tol = C.gate(o, want, ref_err if arm != "inc" else None)
                if arm == "inc" and not poison:
                    ref_err = err
                elif arm == "inc":
                    ref_err = err
                pad_ok = True
                pad_rows = np.setdiff1d(np.arange(inp["T"]), rows)
                if pad_rows.size:  # rows outside every request must return q unchanged or at least be finite
                    pad_ok = bool(np.isfinite(out[pad_rows]).all())
                rec = dict(cell=name, cls=cls(name), arm=arm, poison=poison, err_max_rel=round(err, 6), tol=round(tol, 6),
                           ok=bool(ok and finite), finite=finite, pad_finite=pad_ok,
                           err_mean_rel=round(float(np.abs(o - want).mean() / np.abs(want).max()), 8))
                print("ACC " + json.dumps(rec), flush=True)
                summary.append(rec)
            if not poison and a.faults != "none" and ref_err is not None:
                for fl in C.FAULTS:
                    fo, frows = C.oracle(inp, d, deq, fault=fl)
                    if fo.shape != want.shape:
                        continue
                    err, ok, tol = C.gate(fo, want, ref_err)
                    k = (fl, cls(name))
                    catch.setdefault(k, [0, 0])
                    catch[k][1] += 1
                    catch[k][0] += int(not ok)
                    print("FAULT " + json.dumps(dict(cell=name, cls=cls(name), fault=fl, err_max_rel=round(err, 6),
                                                     tol=round(tol, 6), caught=not ok)), flush=True)
        print(json.dumps(dict(cell=name, seconds=round(time.time() - t0, 1))), flush=True)
    for (fl, c), (n, m) in sorted(catch.items()):
        print("CATCH " + json.dumps(dict(fault=fl, cls=c, caught=n, of=m, blind=(c in ("production", "ragged", "mixed", "prefill-window") and n < m))), flush=True)
    bad = [s for s in summary if not s.get("ok")]
    print("ACC_SUMMARY " + json.dumps(dict(n=len(summary), bad=[(b["cell"], b["arm"], b.get("poison"), b.get("err_max_rel"), b.get("error", "")[:200]) for b in bad])), flush=True)


if __name__ == "__main__":
    main()
