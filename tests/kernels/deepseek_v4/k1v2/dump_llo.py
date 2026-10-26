"""Dump-only LLO / Mosaic capture of one arm at one cell (for /xprof-llo static reading).
  LIBTPU dump flags set before jax import.  python dump_llo.py --cell c16T16-mid --arm inc"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--cell"); ap.add_argument("--arm", default="inc"); ap.add_argument("--keep", default="SparseMLA,CsaUnionDense,csa_gather,union,gather_page_ids")
a = ap.parse_args()
out = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "llo", f"{a.cell}.{a.arm}")
os.makedirs(out + "/jf", exist_ok=True); os.makedirs(out + "/mosaic", exist_ok=True)
os.environ["LIBTPU_INIT_ARGS"] = (os.environ.get("LIBTPU_INIT_ARGS", "") +
    f" --xla_jf_dump_to={out}/jf --xla_jf_dump_llo_text=true --xla_mosaic_dump_to={out}/mosaic")
import jax
import k1v2_common as C
cell = C.CELLS[a.cell]; inp = C.build_inputs(cell); d, _ = C.device_inputs(inp)
f = C.ARMS[a.arm](cell.conc)
jax.block_until_ready(jax.jit(lambda d: f(d, d["q"]))(d))
keep = a.keep.split(",")
for sub in ("jf", "mosaic"):
    for root, _, files in os.walk(os.path.join(out, sub)):
        for fn in files:
            if not any(k.lower() in fn.lower() for k in keep):
                os.remove(os.path.join(root, fn))
print("kept", sum(len(f) for _, _, f in os.walk(out)), out, flush=True)
