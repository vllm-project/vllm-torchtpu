"""K-MLAM accuracy instrument: f64 oracle + stock (native) reference + fault-injection control, mixed-step cells.

  python acc_mixed.py [--cells a,b] [--candidate module:fn[,module:fn2]] [--out acc.jsonl]

Per cell: stock _attention_hca (served env) vs the f64 oracle -> err (max-rel, mean) over the real tokens;
candidate tolerance = test_accuracy.gate(stock error) (2x stock max-rel, >= +1 bf16 ulp). Fault control: every
mlam_common.FAULTS entry applied to the oracle is CAUGHT if it fails that same gate. SWA cache: candidate must
equal stock bit for bit (whole array). Pad output rows (past the step's last token): candidate must be all zero.
"""
import argparse
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlam_common as MC  # noqa: E402
from test_accuracy import err, gate  # noqa: E402

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(MC.CELLS))
    ap.add_argument("--candidate", default="")
    ap.add_argument("--no-faults", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "."), "acc.jsonl"))
    a = ap.parse_args()
    MC.stock_env()
    import jax
    import jax.numpy as jnp
    cands = {}
    for c in [x for x in a.candidate.split(",") if x]:
        m, fn = c.split(":")
        cands[c] = getattr(importlib.import_module(m), fn)
    f_stock = jax.jit(MC.stock_attn())
    fo = open(a.out, "a")
    for name in a.cells.split(","):
        t0 = time.time()
        try:
            inp = MC.build(MC.CELLS[name])
            d = MC.device(inp)
            so, sswa = jax.block_until_ready(f_stock(d))
            n = int(inp.cu[len(inp.cell.reqs)])
            ref = MC.oracle(inp)
            so_np = np.asarray(so.astype(jnp.float32))
            es = err(so_np[:n], ref[:n])
            tol = gate(es)
            rec = dict(cell=name, T=inp.cell.T, H=inp.cell.heads, reqs=[list(x) for x in inp.cell.reqs],
                       dist=inp.dist.tolist(), stock_err_max_rel=es[0], stock_err_mean=es[1], tol=list(tol),
                       stock_pad_finite=bool(np.all(np.isfinite(so_np[n:]))))
            if not a.no_faults:
                caught = {}
                for fl in MC.FAULTS:
                    ef = err(MC.oracle(inp, fault=fl)[:n], ref[:n])
                    caught[fl] = bool(ef[0] > tol[0] or ef[1] > tol[1])
                rec["faults_caught"] = caught
            sswa_np = np.asarray(sswa)
            for cn, cf in cands.items():
                co, cswa = jax.block_until_ready(jax.jit(cf)(d))
                co_np = np.asarray(co.astype(jnp.float32))
                ec = err(co_np[:n], ref[:n])
                cswa_np = np.asarray(cswa)
                diff = np.argwhere(cswa_np != sswa_np)
                rec[cn] = dict(err_max_rel=ec[0], err_mean=ec[1], out_pass=bool(ec[0] <= tol[0] and ec[1] <= tol[1]),
                               swa_bitexact=bool(diff.size == 0), swa_ndiff=int(diff.shape[0]),
                               swa_first_diff=diff[:3].tolist(), pad_rows_zero=bool(np.all(co_np[n:] == 0)),
                               max_tok_err=float(np.abs(co_np[:n] - ref[:n]).max(axis=(1, 2)).argmax()))
            rec["secs"] = round(time.time() - t0, 1)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            rec = dict(cell=name, error=repr(ex)[:800])
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
        print("ACC " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
