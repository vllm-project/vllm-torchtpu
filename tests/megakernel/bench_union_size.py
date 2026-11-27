"""Union size per request from a `tpu_bench_wholestep_call.py --export` selection dump.

usage: bench_union_size.py <sel_b<B>_ctx<C>.npy> <budget>
The bench saves the dump to $CDK_OUTPUT_DIR (the last QSA layer's rows, stock `selected`
layout). For each request (4 verify rows) it prints the number of distinct selected groups
(the union the union-attention path streams) and the rows' groups over the union.
"""
import sys

import numpy as np

sel = np.load(sys.argv[1])
K = int(sys.argv[2])
unions, share = [], []
for q in range(sel.shape[0] // 4):
    groups = [r[: 4 * K : 4] for r in sel[4 * q : 4 * q + 4]]
    groups = [g[g >= 0] // 4 for g in groups]
    u = np.unique(np.concatenate(groups))
    unions.append(len(u))
    share.append(sum(len(g) for g in groups) / max(len(u), 1))
print(f"UNION nreq={len(unions)} union_mean={np.mean(unions):.1f} per_req={unions} "
      f"rows_over_union={np.mean(share):.2f}")
