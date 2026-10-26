"""bench.py on cells whose HCA pages are physically CONTIGUOUS per request (the L2 coalescing upside cell; production
page contiguity is unknown). Same clock, same arms flags as bench.py; cells get the suffix '-contig' in the output.
  python bench_contig.py --cells c8T32-mid,... --arms ... [--profile]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import hca_common as C  # noqa: E402

_orig = C.build_inputs


def build_contig(cell, seed=0, sbs=32):
    """Same as hca_common.build_inputs, then physical page ids are relabelled (a permutation applied to every table and
    cache) so each request's HCA pages are consecutive ids 1.., request after request."""
    inp = _orig(cell, seed=seed, sbs=sbs)
    N = cell.pool
    f = np.arange(N)  # old id -> new id
    g = np.arange(N)  # new id -> old id
    t = 1
    for r in range(cell.R):
        for j in range(inp.hca_bt.shape[1]):
            o = int(inp.hca_bt[r, j])
            if o == 0:
                continue
            a, b = o, int(g[t])  # move old page o to new id t; the page now at t takes o's new id
            fa, fb = int(f[a]), int(f[b])
            f[a], f[b] = fb, fa
            g[fb], g[fa] = a, b
            t += 1
    for bt in (inp.swa_bt, inp.hca_bt, inp.st_bt):
        nz = bt != 0
        bt[nz] = f[bt[nz]]
    inp.swa_words = {int(f[k]): v for k, v in inp.swa_words.items()}
    inp.hca_words = {int(f[k]): v for k, v in inp.hca_words.items()}
    inp.st_words = {int(f[k]): v for k, v in inp.st_words.items()}
    return inp


C.build_inputs = build_contig
C.CELLS = {k + "-contig": v for k, v in C.CELLS.items()}
C.CELLS = {k: type(v)(**{**v.__dict__, "name": k}) for k, v in C.CELLS.items()}

if __name__ == "__main__":
    import bench
    bench.C = C
    bench.main()
