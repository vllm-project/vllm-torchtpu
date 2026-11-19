# SPDX-License-Identifier: Apache-2.0
"""q38 HC-FIX-2: TPU-only in-pod bitwise check of the MOEPORT-R umbrella TPU_MOE_NONEP_RFORM_PORT (default off; this script changes
nothing served). Pattern: HC-FIX's tests/q38_hcfix/tpu_trim_pairs_hash.py (credited).

Runs one prefill MoE layer through fused_moe_gmm.fused_moe_func -- the jitted function the served torch op wraps
(layers/adapter/fused_moe.py) -- with the pod's env statics (USE_MOE_SPARSE_CORE, TPU_MOE_SKIP_PADDED_TOKENS, TPU_MOE_NONEP_*: the
RK2144 prefill env), at served dims (512 experts, top 10, hidden 2560, expert inter 640, fp8 per-channel weights), with the umbrella
OFF and ON. The four umbrella values come from the adapter's own nonep_dispatch_values(rform_port), so the arms are exactly what the
served op builds: OFF = the env's TPU_MOE_NONEP_PERMUTE / _GATHER / _DISPATCH_CHUNKS / TPU_MOE_COMBINE_PRESORTED (RK2144:
sorts / sc_padded / 1 / 0), ON = packed_onehot / sc_fp8 / 2 / 1 whatever the env says. This is the only check of DISPATCH_CHUNKS=2
and COMBINE_PRESORTED (SparseCore-only) on this tree. One hash row per case:
  MOEPORTHASH T <rows> <routing>: off <sha16> on <sha16> equal <True|False>
then 'MOEPORTHASH VERDICT PASS' iff every row is equal (and 'MOEPORTHASH ENGAGED ...' shows the ON program differs from OFF).
Routings: AgentX layers 36 / 42 (recorded routing of real conversations; agentx_routing.npz next to this file, MOEHARN's file) and
the adversarial set (alternating empty experts, one hot expert, single odd-index experts, empty first and last). Activations are
random bf16 with stress values (zeros, -0.0, bf16 extremes); weights random fp8 with +-448 entries. Not a timing.
Run in the A/B pod: python3 tests/q38_hcfix/tpu_moeport_hash.py [--rows 2048,6400] [--routings ...]"""
import argparse
import hashlib
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

TOPK, EXPERTS, HIDDEN, INTER = 10, 512, 2560, 640
HERE = os.path.dirname(os.path.abspath(__file__))
ROUTINGS = "agentx_L36,agentx_L42,alternating,one_hot,odd_single,first_last_empty"


def routing(kind, T, r):
    if kind.startswith("agentx"):
        z = np.load(os.path.join(HERE, "agentx_routing.npz"))
        pool = np.asarray(z["experts"][:, :, int(kind.split("_L")[1]), :], np.int32).reshape(-1, TOPK)
        return pool[r.integers(0, pool.shape[0], T)]
    pick = lambda c: np.stack([r.choice(c, TOPK, replace=False) for _ in range(T)]).astype(np.int32)  # noqa: E731
    if kind == "alternating":
        return pick(np.arange(0, EXPERTS, 2))
    if kind == "one_hot":
        ids = pick(np.arange(8, 72))
        ids[:, 0] = 7
        return ids
    if kind == "odd_single":
        return np.tile(np.arange(1, 2 * TOPK, 2, dtype=np.int32), (T, 1))
    if kind == "first_last_empty":
        return pick(np.arange(1, EXPERTS - 1))
    raise ValueError(kind)


def fp8(r, shape):
    w = r.standard_normal(shape, dtype=np.float32) / 16
    scale = np.abs(w).max(axis=1, keepdims=True) / 448.0
    q = np.clip(w / scale, -448.0, 448.0)
    flat = q.reshape(-1)
    idx = r.integers(0, flat.size, flat.size // 211)
    flat[idx] = np.where(r.random(idx.size) < 0.5, -448.0, 448.0)
    return jnp.asarray(q, jnp.float8_e4m3fn), jnp.asarray(scale[:, :, None, :], jnp.float32)


def statics_for(rform_port):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter.fused_moe import nonep_dispatch_values
    from vllm_torchtpu.layers.core import fused_moe_gmm as F
    permute, gather, chunks, presorted = nonep_dispatch_values(rform_port)
    return dict(
        topk=TOPK, activation="silu", use_ep=False, use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_moe_permute_threshold=F.resolve_onehot_permute_threshold(), skip_padded_tokens=envs.TPU_MOE_SKIP_PADDED_TOKENS,
        nonep_permute=permute, nonep_gather=gather,
        nonep_gather_cols=envs.TPU_MOE_NONEP_GATHER_COLS, nonep_gather_chunk=envs.TPU_MOE_NONEP_GATHER_CHUNK,
        nonep_combine_split=int(envs.TPU_MOE_NONEP_COMBINE_SPLIT), nonep_gmm_tile_m=int(envs.TPU_MOE_NONEP_GMM_TILE_M),
        nonep_gmm_hoist_quant=envs.TPU_MOE_NONEP_GMM_HOIST_QUANT, nonep_gmm2_rhs_split=int(envs.TPU_MOE_NONEP_GMM2_RHS_SPLIT),
        nonep_gmm_experts_per_visit=int(envs.TPU_MOE_NONEP_GMM_EXPERTS_PER_VISIT),
        nonep_combine_packed=envs.TPU_MOE_NONEP_COMBINE_PACKED, nonep_gmm_bucket=int(envs.TPU_MOE_NONEP_GMM_BUCKET),
        nonep_gmm2_experts_per_visit=int(envs.TPU_MOE_NONEP_GMM2_EXPERTS_PER_VISIT), nonep_fused_mlp=envs.TPU_MOE_NONEP_FUSED_MLP,
        # passed only when set, as the adapter does, so the OFF arm is the served flag-off call
        **({"nonep_dispatch_chunks": int(chunks)} if chunks != "1" else {}),
        **({"combine_presorted": True} if presorted else {}),
    )


def engaged_check(F, st, x, w1, w2, s1, s2, T):
    sp = lambda kw: str(jax.make_jaxpr(lambda *z: F.fused_moe_func(*z, **kw))(  # noqa: E731
        x, w1, w2, s1, s2, None, None, jnp.zeros((T, TOPK), jnp.bfloat16), jnp.zeros((T, TOPK), jnp.int32), None))
    jo, jn = sp(st["off"]), sp(st["on"])
    print(f"MOEPORTHASH ENGAGED T {T}: on program differs {jo != jn}; pack_fp8_rows on {'pack_fp8_rows' in jn} off "
          f"{'pack_fp8_rows' in jo}")
    return jo != jn and "pack_fp8_rows" in jn and "pack_fp8_rows" not in jo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="2048,6400")
    ap.add_argument("--routings", default=ROUTINGS)
    a = ap.parse_args()
    if jax.default_backend() != "tpu":
        print("MOEPORTHASH SKIP: not on TPU")
        return 0
    from vllm_torchtpu.layers.core import fused_moe_gmm as F

    st = {"off": statics_for(False), "on": statics_for(True)}
    for arm in ("off", "on"):
        print(f"MOEPORTHASH statics {arm}", {k: v for k, v in st[arm].items() if k.startswith(("nonep", "use_", "skip", "combine"))})
    r = np.random.default_rng(1119)
    w1, s1 = fp8(r, (EXPERTS, HIDDEN, 2 * INTER))
    w2, s2 = fp8(r, (EXPERTS, INTER, HIDDEN))
    bad, engaged = 0, None
    for T in [int(t) for t in a.rows.split(",")]:
        x = (r.standard_normal((T, HIDDEN), dtype=np.float32) / 4)
        x[r.integers(0, T, T // 100)] = 0.0
        x[r.integers(0, T, T // 100)] = -0.0
        x[1, 7], x[2, :] = 3.0e38, 1e-30
        x = jnp.asarray(x, jnp.bfloat16)
        if engaged is None:   # the ON program must be a different program (the umbrella engaged), else equal rows mean nothing
            engaged = engaged_check(F, st, x, w1, w2, s1, s2, T)
        for kind in a.routings.split(","):
            ids = jnp.asarray(routing(kind, T, r))
            wts = jnp.asarray(r.random((T, TOPK), dtype=np.float32), jnp.bfloat16)
            outs = {}
            for arm in ("off", "on"):
                o = F.fused_moe_func(x, w1, w2, s1, s2, None, None, wts, ids, None, **st[arm])
                outs[arm] = np.asarray(jax.block_until_ready(o))
            sha = {k: hashlib.sha1(v.tobytes()).hexdigest()[:16] for k, v in outs.items()}
            eq = outs["off"].tobytes() == outs["on"].tobytes()
            bad += not eq
            print(f"MOEPORTHASH T {T} {kind}: off {sha['off']} on {sha['on']} equal {eq}", flush=True)
    ok = not bad and bool(engaged)
    print(f"MOEPORTHASH VERDICT {'PASS' if ok else 'FAIL'}{'' if engaged else ' (umbrella not engaged)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
