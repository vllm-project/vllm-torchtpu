import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "src")]
from tests.models.minimax_m3 import test_msa_index_reuse as t  # noqa: E402
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads, msa_fp4  # noqa: E402

dcp, g = int(sys.argv[1]), int(sys.argv[2])
variants = sys.argv[3].split(",")  # e.g. off:1,split:1,off:2,split:2
if len(sys.argv) > 4:
    os.environ["TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG"] = sys.argv[4]
t.Q_LENS[:] = [1, 40, 2, 45]
t.CTX_LENS[:] = [1100, 1000, 0, 900]
rng, num_real_pad, meta = t._setup(7, dcp)
num_tokens = 96
seq_lens, tables, cu = meta
layer = t._layer(rng, num_tokens, dcp, g, "fp4")
caches, cands = t._full(layer, meta, dcp)
q = msa_fp4.rotate_queries(layer["q"])  # [T, g * HEADS, D]
reps = dcp // g
hl = q.shape[1] // dcp
q_ranks = jnp.concatenate([q[:, r * hl : (r + 1) * hl] for r in range(dcp)])
dist = jnp.asarray([0, 0, len(t.Q_LENS)], jnp.int32)
mesh = Mesh(np.array(jax.devices()[:dcp]), ("dcp",))
sh, rep = P("dcp"), P()
args = (
    jnp.arange(dcp, dtype=jnp.int32),
    q_ranks,
    jnp.concatenate([c[0] for c in caches]),
    jnp.concatenate([c[2] for c in caches]),
    jnp.concatenate(list(cands)),
    seq_lens,
    tables,
    cu,
    dist,
)


def run():
    f = jax.jit(
        jax.shard_map(
            lambda *a: msa_dcp_heads.attend_and_merge(
                *a,
                axis="dcp",
                dcp=dcp,
                kv_head_replicas=reps,
                sm_scale=t.SCALE,
                topk=t.TOPK,
                decode_rows=2,
                rs_out_precision=False,
                head_parallel_min_tokens=0,
                interpret=True,
            ),
            mesh=mesh,
            in_specs=(sh,) * 5 + (rep,) * 4,
            out_specs=sh,
            check_vma=False,
        )
    )
    return np.asarray(f(*args).astype(jnp.float32)).reshape(dcp, num_tokens, hl, -1)


outs = {}
for v in variants:
    mode, cap = v.split(":")
    os.environ["TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"] = "1" if mode == "split" else ""
    os.environ["TPU_MINIMAX_M3_DCP_HP_ROUND_CAPACITY"] = cap
    t0 = time.time()
    outs[v] = run()
    print(
        f"{v}: {time.time() - t0:.1f}s finite={np.isfinite(outs[v]).all()}", flush=True
    )
real = sum(t.Q_LENS)
cu_np = np.asarray(cu)
dec = np.zeros(num_tokens, bool)
for i, ql in enumerate(t.Q_LENS):
    if 0 < ql <= 2:
        dec[cu_np[i] : cu_np[i] + ql] = True
hp = np.zeros(num_tokens, bool)
hp[:real] = ~dec[:real]
ref = outs[variants[0]]
assert np.abs(ref[:, hp]).max() > 0
for v in variants[1:]:
    o = outs[v]
    d_hp = np.abs(o[:, hp] - ref[:, hp]).max() / np.abs(ref[:, hp]).max()
    print(
        f"{v} vs {variants[0]}: "
        f"decode bitwise={np.array_equal(o[:, dec], ref[:, dec])} "
        f"hp bitwise={np.array_equal(o[:, hp], ref[:, hp])} hp rel={d_hp:.2e} "
        f"pad zero={np.all(o[:, real:] == 0)}"
    )
for a, b in [("off:2", "split:2"), ("off:1", "split:1"), ("off:4", "split:4")]:
    if a in outs and b in outs:
        print(f"PAIR {a} {b} bitwise={np.array_equal(outs[a], outs[b])}")
print("OK")
