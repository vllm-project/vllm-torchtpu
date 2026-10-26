"""Dump-only LLO/Mosaic capture of the incumbent region (one call per cell); no timing is read from this process.
  python dump_llo.py --cells c4T8-mid,c16T32-mid
"""
import argparse
import os
import sys

out = os.environ.get("CDK_OUTPUT_DIR", ".")
os.environ["LIBTPU_INIT_ARGS"] = (os.environ.get("LIBTPU_INIT_ARGS", "") +
                                  f" --xla_jf_dump_to={out}/llo/jf --xla_jf_dump_llo_text=true"
                                  f" --xla_mosaic_dump_to={out}/llo/mosaic")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hca_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="c4T8-mid")
    a = ap.parse_args()
    C.stock_env()
    import jax
    import jax.numpy as jnp
    f, _ = C.stock_region()
    cs = jnp.asarray(C.cos_sin_np())
    for name in a.cells.split(","):
        d = C.device_inputs(C.build_inputs(C.CELLS[name]))
        jax.block_until_ready(jax.jit(f)(d, cs))
        print("DUMPED", name, flush=True)


if __name__ == "__main__":
    main()
