"""Stage-B rung B24 correctness: the fused kernel with q RMSNorm+RoPE and kv RoPE in its prologue must equal, BIT FOR BIT, the
stage-A kernel fed by the stock rope ops (rope.qnorm_rope_flat + rope.rope), outputs and SWA cache; and the SWA/HCA/state caches
must equal the stock path's. Cells: all phase cells + edges (hca_common) and the mixed cells (mixed.py).
  python b24.py [--cells ...] [--mixed]"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import hca_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(C.CELLS))
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--knobs", default="{}")
    ap.add_argument("--out", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "."), "b24.jsonl"))
    a = ap.parse_args()
    C.stock_env()
    import jax
    import jax.numpy as jnp
    import khca_arms as K
    knobs = json.loads(a.knobs)
    f_stock = jax.jit(K._b24("stock")(("rope", "comp", "attn")))
    f_a3 = jax.jit(K._b24("a3", knobs)(("rope", "comp", "attn")))
    f_b24 = jax.jit(K._b24("b24", knobs)(("rope", "comp", "attn")))
    cs = jnp.asarray(C.cos_sin_np())
    items = []
    if a.mixed:
        import mixed as M
        for n in M.MIXED:
            inp = M.build(n)
            d = {k: v for k, v in M.device(inp).items() if v is not None}
            d["hidden"] = jnp.zeros((inp.T, C.HIDDEN), jnp.bfloat16)
            items.append((n, d, int(inp.cu[len(inp.reqs)]), True))
    for n in (a.cells.split(",") if a.cells else []):
        inp = C.build_inputs(C.CELLS[n])
        items.append((n, C.device_inputs(inp), C.VQ * inp.cell.R, False))
    with open(a.out, "a") as fo:
        for n, d, nreal, mixed in items:
            try:
                if mixed:  # attention only (no compressor state arrays in mixed cells)
                    fs = jax.jit(K._b24("stock")(("attn",)))
                    fa = jax.jit(K._b24("a3", knobs)(("attn",)))
                    fb = jax.jit(K._b24("b24", knobs)(("attn",)))
                else:
                    fs, fa, fb = f_stock, f_a3, f_b24
                so, sswa, shca, sst = jax.block_until_ready(fs(d, cs))
                ao, aswa, ahca, ast = jax.block_until_ready(fa(d, cs))
                bo, bswa, bhca, bst = jax.block_until_ready(fb(d, cs))
                eq = lambda x, y: bool(np.array_equal(np.asarray(x), np.asarray(y)))
                ref = np.asarray(so.astype(jnp.float32))[:nreal].astype(np.float64)
                rel = lambda o: float(np.abs(np.asarray(o.astype(jnp.float32))[:nreal] - ref).max() / np.abs(ref).max())
                r = dict(cell=n, b24_eq_a3_out=eq(bo[:nreal], ao[:nreal]), b24_eq_a3_swa=eq(bswa, aswa),
                         swa_eq_stock=eq(bswa, sswa), hca_eq_stock=eq(bhca, shca), st_eq_stock=eq(bst, sst),
                         b24_vs_stock_maxrel=rel(bo), a3_vs_stock_maxrel=rel(ao),
                         pad_zero=bool(np.all(np.asarray(bo.astype(jnp.float32))[nreal:] == 0)))
            except Exception as ex:
                import traceback
                traceback.print_exc()
                r = dict(cell=n, error=repr(ex)[:400])
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print("B24 " + json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
