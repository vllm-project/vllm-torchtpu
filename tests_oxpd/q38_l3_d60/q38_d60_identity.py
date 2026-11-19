# SPDX-License-Identifier: Apache-2.0
"""q38 L3 D60 flag-off identity: without compact_pipe, the tip's fused MoE decode programs equal the base's.

Usage: python3 -B q38_d60_identity.py <tip src> <ctl src>
Runs itself twice in subprocesses (PYTHONPATH=<tip src>, then <ctl src>), each dumping location-free jaxpr texts
(source paths and line numbers replaced by <loc>) of fused_moe_decode at Qwen3.8-Flash EP8's served per-rank shapes
(64 local experts, hidden 2560, routed intermediate 640, BF16 shared expert padded to 128, top-10, 8 ranks):
  D  dense, folded and unfolded, at 16 / 64 / 128 / 160 / 192 / 208 / 224 / 256 tokens (and 250, padded);
  C  lever 5b compact_rows=32, folded, at 160 / 192 / 208 / 224 / 256;
then compares case by case. Sensitivity controls on the tip (must DIFFER after normalisation): compact_rows=32 with
compact_pipe vs without at 224 and 256 tokens, and dense vs compact_rows=32 at 256. The jaxpr embeds the Pallas body,
so this is the kernel program itself. Last line: Q38_D60_IDENTITY IDENTITY=PASS|FAIL|PARSE-FAIL ...
(PARSE-FAIL: a dump is missing, unreadable or incomplete). Exit 0 PASS, 1 FAIL, 2 PARSE-FAIL.
Needs JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

N, E, H, INTER, ISP, TOPK = 8, 64, 2560, 640, 128, 10
DENSE_ROWS = (16, 64, 128, 160, 192, 208, 224, 250, 256)
COMPACT_ROWS = (160, 192, 208, 224, 256)
_LOC = re.compile(r"[\w./<>-]*\.py:\d+(?::\d+)?")


def norm(text: str) -> str:
    return _LOC.sub("<loc>", text)


def dump(path: str, with_controls: bool) -> None:
    import importlib.util
    import types

    if importlib.util.find_spec("vllm") is None:
        import logging

        v = types.ModuleType("vllm")
        vl = types.ModuleType("vllm.logger")
        vl.init_logger = logging.getLogger
        v.logger = vl
        sys.modules.setdefault("vllm", v)
        sys.modules.setdefault("vllm.logger", vl)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.experimental.fused_moe_decode import fused_moe_decode_kernel as kmod

    kmod._vmem_capacity = lambda: 64 << 20
    F8, BF, F32 = jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32
    mesh = Mesh(np.asarray(jax.devices()[:N]), ("d",))
    S = jax.ShapeDtypeStruct

    def program(tokens, fold, compact=None, pipe=False):
        def body(rank, w1, w2, w1_s, w2_s, sw1, sw2, x, gate_row, tw, ids):
            kw = dict(n_ranks=N, expert_start="rank" if fold else rank[0, 1], fold=fold)
            if compact is not None:
                kw["compact_rows"] = compact
            if pipe:
                kw["compact_pipe"] = True
            if fold:
                return kmod.fused_moe_decode(
                    rank, x, w1[0], w2[0], w1_s[0][:, :, None, :], w2_s[0][:, :, None, :], sw1[0],
                    jnp.ones((2 * ISP,), F32), sw2[0], jnp.ones((H,), F32), gate_row, tw, ids, **kw)[None]
            return kmod.fused_moe_decode(
                rank[:, :1], x, w1[0], w2[0], w1_s[0][:, :, None, :], w2_s[0][:, :, None, :], sw1[0],
                jnp.ones((1, 2 * ISP), F32), sw2[0], jnp.ones((1, H), F32), gate_row.reshape(-1, 1), tw,
                ids, **kw)[None]

        args = [S((N, 2), jnp.int32), S((N, E, H, 2 * INTER), F8), S((N, E, INTER, H), F8),
                S((N, E, 1, 2 * INTER), F32), S((N, E, 1, H), F32), S((N, H, 2 * ISP), BF), S((N, ISP, H), BF),
                S((tokens, H), BF), S((1, H), BF), S((tokens, TOPK), BF), S((tokens, TOPK), jnp.int32)]
        f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("d"),) * 7 + (P(),) * 4, out_specs=P("d"),
                                  check_vma=False))
        return norm(str(jax.make_jaxpr(f)(*args)))

    out = {"src": os.path.dirname(kmod.__file__)}
    for tokens in DENSE_ROWS:
        for fold in (True, False):
            out[f"D {'fold' if fold else 'unfolded'} {tokens}"] = program(tokens, fold)
    for tokens in COMPACT_ROWS:
        out[f"C cr32 fold {tokens}"] = program(tokens, True, 32)
    if with_controls:
        for tokens in (224, 256):
            out[f"X cr32p fold {tokens}"] = program(tokens, True, 32, pipe=True)
    out["complete"] = True
    with open(path, "w") as f:
        json.dump(out, f)


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--dump":
        dump(sys.argv[2], sys.argv[3] == "1")
        return 0
    tip, ctl = (os.path.abspath(p) for p in sys.argv[1:3])
    tmp = tempfile.mkdtemp(prefix="q38_d60_id_")
    dumps = {}
    for tag, src, ctrl in (("tip", tip, "1"), ("ctl", ctl, "0")):
        path = os.path.join(tmp, f"{tag}.json")
        env = dict(os.environ, PYTHONPATH=src)
        r = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), "--dump", path, ctrl], env=env,
                           capture_output=True, text=True, timeout=3000)
        print(f"Q38_D60_IDENTITY dump {tag} rc={r.returncode} src={src}")
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-6000:])
        try:
            dumps[tag] = json.load(open(path))
        except Exception as exc:  # noqa: BLE001 - a missing / unreadable dump is PARSE-FAIL
            print(f"Q38_D60_IDENTITY dump {tag} unreadable: {type(exc).__name__}: {exc}")
            print("Q38_D60_IDENTITY IDENTITY=PARSE-FAIL")
            return 2
    t, c = dumps["tip"], dumps["ctl"]
    want_src = {"tip": tip, "ctl": ctl}
    for tag in ("tip", "ctl"):
        if not dumps[tag].get("src", "").startswith(want_src[tag] + os.sep) or not dumps[tag].get("complete"):
            print(f"Q38_D60_IDENTITY {tag} imported from {dumps[tag].get('src')} (want under {want_src[tag]}) "
                  f"or incomplete")
            print("Q38_D60_IDENTITY IDENTITY=PARSE-FAIL")
            return 2
    keys = sorted(k for k in c if k[:2] in ("D ", "C "))
    want = len(DENSE_ROWS) * 2 + len(COMPACT_ROWS)
    if len(keys) != want or set(keys) - set(t):
        print(f"Q38_D60_IDENTITY case set mismatch: ctl {len(keys)} (want {want}), missing on tip "
              f"{sorted(set(keys) - set(t))}")
        print("Q38_D60_IDENTITY IDENTITY=PARSE-FAIL")
        return 2
    bad = 0
    for k in keys:
        same = t[k] == c[k]
        bad += not same
        print(f"Q38_D60_IDENTITY {k}: {'IDENTICAL' if same else 'DIFFERS'} (len {len(t[k])} / {len(c[k])})")
    ctl_ok = 0
    for tokens in (224, 256):
        d = t[f"X cr32p fold {tokens}"] != t[f"C cr32 fold {tokens}"]
        ctl_ok += d
        print(f"Q38_D60_IDENTITY control cr32p vs cr32 at {tokens}: {'DIFFERS (ok)' if d else 'SAME (control FAILED)'}")
    d = t["C cr32 fold 256"] != t["D fold 256"]
    ctl_ok += d
    print(f"Q38_D60_IDENTITY control cr32 vs dense at 256: {'DIFFERS (ok)' if d else 'SAME (control FAILED)'}")
    ok = bad == 0 and ctl_ok == 3
    print(f"Q38_D60_IDENTITY IDENTITY={'PASS' if ok else 'FAIL'} cases {len(keys)} differ {bad} controls {ctl_ok}/3")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
