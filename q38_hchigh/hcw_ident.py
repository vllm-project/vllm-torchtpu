# SPDX-License-Identifier: Apache-2.0
"""q38 HC-HIGHROWS flag-off identity probe. Run from a tree root with PYTHONPATH=src (base a64b43606c or head).

Captures the cores the adapter hands to pallas.jax_op (build_decode_mix_op / build_decode_combine_mix_op, inject on and off;
jax_op is replaced by a recorder, nothing is registered) under the current env, lowers each for one deviceless TPU7x core at
every bucket, and prints the sha256 of the lowered StableHLO text (no debug info) per (op, bucket), plus the op names and the
sorted compile env-factor keys. The proof body compares base vs head (flag unset / 0 must equal base everywhere; WIDE=1 / 2
must equal base below 160 rows and differ at 160-496: the probe sees the flag).

usage: hcw_ident.py --out JSON
"""
import argparse
import hashlib
import json
import os
import sys

BR, H, LR, EPS = 4, 2560, 320, 1e-6
W = BR * H
BUCKETS = (16, 64, 112, 128, 144, 160, 192, 240, 288, 384, 448, 496)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, "src")
    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies

    from vllm_torchtpu.compilation import tpu_compiler
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as va

    class _Fake:
        def register_fake(self, fn):
            return fn

    cores = {}
    real = va.pallas.jax_op
    va.pallas.jax_op = lambda name, core: cores.__setitem__(name, core) or _Fake()
    try:
        order = []
        for kind, build in (("mix", va.build_decode_mix_op), ("combine", va.build_decode_combine_mix_op)):
            for inject in (True, False):
                before = set(cores)
                build.__wrapped__(BR, H, EPS, inject)
                (name,) = set(cores) - before
                order.append((f"{kind}_{'inj' if inject else 'noinj'}", name))
    finally:
        va.pallas.jax_op = real
    dev = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1").devices[0]
    sh = jax.sharding.SingleDeviceSharding(dev)
    s = lambda shape, dt=jnp.bfloat16: jax.ShapeDtypeStruct(shape, dt, sharding=sh)  # noqa: E731
    out = {"env": {k: os.environ.get(k) for k in ("TPU_QWEN4_HC_DECODE_WIDE", "TPU_QWEN4_HC_DECODE_COMPARE",
                                                  "TPU_QWEN4_HC_DECODE_TILE_ROWS")},
           "names": dict(order), "hash": {}, "errors": [],
           "factor_keys": sorted(tpu_compiler._tpu_compile_env_factors())}
    for key, name in order:
        core = cores[name]
        for t in BUCKETS:
            if key.startswith("mix"):
                args = [s((t, W)), s((W,)), s((LR, W)), s((W, LR))] + ([s((BR, W))] if key.endswith("_inj") else [])
            else:
                args = [s((t, W)), s((t, H)), s((t, BR), jnp.float32), s((W,)), s((LR, W)), s((W, LR))]
                args += [s((BR, W))] if key.endswith("_inj") else []
            try:
                txt = jax.jit(core).lower(*args).as_text()
                out["hash"][f"{key}@{t}"] = hashlib.sha256(txt.encode()).hexdigest()[:16]
                out["hash"][f"{key}@{t}#wide"] = "_w1" in txt or "_w2" in txt
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"{key}@{t}: {type(e).__name__}: {str(e)[:300]}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("HCWID", json.dumps({"env": out["env"], "n_hash": len(out["hash"]) // 2, "errors": out["errors"][:3]}), flush=True)
    return 1 if out["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
