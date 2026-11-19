# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL MOE flag-off identity: with both keys unset, the head's fused MoE decode op program and PLE lookup program
equal the base's (v8 6f230100a); with the keys set, the D14 lookup buckets still equal the base's.

Usage: python3 -B q38_portbl_identity.py <head src> <base src>
Runs itself twice in subprocesses (PYTHONPATH=<head src>, then <base src>; the same interpreter and plugins), each dumping
location-free jaxpr texts (source paths and line numbers replaced by <loc>; the jaxpr embeds the Pallas kernel body, so
this is the kernel program itself):
  MOE  the op function layers/adapter/fused_moe_decode._build_op builds for the target model (torch_tpu's jax_op replaced
       by a recorder that keeps the function), v8's served MoE keys, Qwen3.8-Flash EP8 per-rank shapes (64 local experts,
       hidden 2560, routed intermediate 640, BF16 shared expert padded to 128, top-10, 8 ranks), at 16 / 64 / 80 / 96 /
       112 / 128 / 160 / 192 / 224 / 256 tokens;
  PLE  the function layers/adapter/custom_ops/ple_op.build_raw_lookup_op(4096, 160, 8) builds, ids [T, 16], at 8 / 16 /
       32 / 48 / 64 / 80 / 96 / 112 / 128 / 160 / 192 / 224 / 256 / 512 tokens.
Head-only sensitivity / D14 controls: MOE with exact width at 96 and 192 must DIFFER from base; PLE built as the car builds
it (512, min_rows 80) must EQUAL base at 8..64 (the D14 buckets) and DIFFER at 80..512.
Last line: Q38_PORTBL_IDENTITY IDENTITY=PASS|FAIL|PARSE-FAIL ...; exit 0 PASS, 1 FAIL, 2 PARSE-FAIL.
Needs JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

MOE_ROWS = (16, 64, 80, 96, 112, 128, 160, 192, 224, 256)
PLE_ROWS = (8, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512)
D14_ROWS = (8, 16, 32, 48, 64)
N, E, H, INTER, ISP, TOPK = 8, 64, 2560, 640, 128, 10
V8_MOE_ENV = {
    "FUSED_MOE_DECODE_KERNEL": "1",
    "FUSED_MOE_DECODE_KERNEL_MAX_TOKENS": "256",
    "FUSED_MOE_DECODE_KERNEL_FOLD": "xla",
    "FUSED_MOE_DECODE_COMPACT_PIPE": "160",
    "FUSED_MOE_DECODE_COMPACT_PIPE_ROWS": "32",
    "FUSED_MOE_DECODE_COMPACT_ROWS": "32",
}
_LOC = re.compile(r"[\w./<>-]*\.py:\d+(?::\d+)?")


def _sha(text: str) -> str:
    return hashlib.sha256(_LOC.sub("<loc>", text).encode()).hexdigest()[:16]


def dump(out_path: str, controls: bool) -> None:
    for name in list(os.environ):
        if name.startswith(("FUSED_MOE_DECODE_", "VLLM_TPU_PLE_LOOKUP_")):
            del os.environ[name]
    os.environ.update(V8_MOE_ENV)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh
    from torch_tpu._internal import pallas as tt_pallas

    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
    from vllm_torchtpu.kernels.experimental.fused_moe_decode import fused_moe_decode_kernel as kmod
    from vllm_torchtpu.layers.adapter import fused_moe_decode as bridge
    from vllm_torchtpu.layers.adapter.custom_ops import ple_op

    class _Op:
        def __init__(self, name, fn, **kwargs):
            self.name, self.fn = name, fn

        def register_fake(self, fake):
            return fake

    class _Quiet:
        def info(self, *a, **k):
            pass

        debug = warning = info

    kmod._vmem_capacity = lambda: 64 << 20
    bridge.jax_op = _Op
    bridge.logger = _Quiet()
    tt_pallas.jax_op = _Op
    if hasattr(ple_op, "logger"):
        ple_op.logger = _Quiet()
    F8, BF, F32, I32 = jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32, jnp.int32
    S = jax.ShapeDtypeStruct
    mesh = Mesh(np.asarray(jax.devices()[:N]), (EP_AXIS_NAME,))
    res = {"src": os.path.dirname(os.path.dirname(kmod.__file__))}

    def moe_args(t):
        return (S((N * t, H), BF), S((N * E, H, 2 * INTER), F8), S((N * E, INTER, H), F8),
                S((N * E, 1, 1, 2 * INTER), F32), S((N * E, 1, 1, H), F32), S((N * H, 2 * ISP), BF),
                S((N * 2 * ISP,), F32), S((N * ISP, H), BF), S((N * H,), F32), S((H, 1), BF),
                S((N * t, TOPK), F32), S((N * t, TOPK), I32), S((N, 2), I32))

    def moe(tag, **kw):
        bridge._OPS.clear()
        op = bridge._build_op(mesh, N, TOPK, True, 256, True, 1.0, **kw)
        rows = MOE_ROWS if not kw else (96, 192)
        for t in rows:
            res[f"{tag} {t}"] = _sha(str(jax.make_jaxpr(op.fn)(*moe_args(t))))

    def ple(tag, *build):
        ple_op.build_raw_lookup_op.cache_clear()
        op = ple_op.build_raw_lookup_op(4096, 160, *build)
        table = S((4096 * 160 // 128, 128), jnp.uint8)
        for t in PLE_ROWS:
            res[f"{tag} {t}"] = _sha(str(jax.make_jaxpr(op.fn)(table, S((t, 16), I32), S((1,), I32))))

    moe("MOE")
    ple("PLE", 8)
    if controls:
        # Each control only where its mechanism is in the tree (the two commits compose independently).
        import inspect

        if "exact_width" in inspect.signature(bridge._build_op).parameters:
            moe("CTL-MOE-XW", exact_width=True)
        if "min_rows" in inspect.signature(ple_op.build_raw_lookup_op.__wrapped__).parameters:
            ple("CTL-PLE-CAR", 512, 80)
    with open(out_path, "w") as f:
        json.dump(res, f)


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--dump":
        dump(sys.argv[2], sys.argv[3] == "1")
        return 0
    head, base = sys.argv[1:3]
    got = {}
    for label, src, controls in (("head", head, True), ("base", base, False)):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        env = dict(os.environ, PYTHONPATH=src)
        rc = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), "--dump", path, "1" if controls else "0"],
                            env=env, cwd=os.path.dirname(src) or ".").returncode
        print(f"Q38_PORTBL_IDENTITY dump {label} rc={rc} src={src}")
        try:
            got[label] = json.load(open(path))
        except Exception as exc:  # noqa: BLE001
            print(f"Q38_PORTBL_IDENTITY IDENTITY=PARSE-FAIL {label} dump unreadable: {exc}")
            return 2
    h, b = got["head"], got["base"]
    print(f"Q38_PORTBL_IDENTITY origins head {h.get('src')} base {b.get('src')}")
    cases = [f"MOE {t}" for t in MOE_ROWS] + [f"PLE {t}" for t in PLE_ROWS]
    if any(c not in h or c not in b for c in cases) or h.get("src") == b.get("src"):
        print("Q38_PORTBL_IDENTITY IDENTITY=PARSE-FAIL incomplete dumps or both dumps from one tree")
        return 2
    differ = [c for c in cases if h[c] != b[c]]
    for c in cases:
        print(f"Q38_PORTBL_IDENTITY case {c}: head {h[c]} base {b[c]} {'SAME' if h[c] == b[c] else 'DIFFERS'}")
    ctl_bad = []
    if "CTL-MOE-XW 96" not in h and "CTL-PLE-CAR 8" not in h:
        ctl_bad.append("no control ran")
    for t in (96, 192) if "CTL-MOE-XW 96" in h else ():
        ok = h[f"CTL-MOE-XW {t}"] != b[f"MOE {t}"]
        print(f"Q38_PORTBL_IDENTITY control MOE exact width at {t}: {'DIFFERS (ok)' if ok else 'SAME (BAD)'}")
        ctl_bad += [] if ok else [f"moe-xw {t}"]
    for t in PLE_ROWS if "CTL-PLE-CAR 8" in h else ():
        same = h[f"CTL-PLE-CAR {t}"] == b[f"PLE {t}"]
        want_same = t in D14_ROWS
        ok = same == want_same
        print(f"Q38_PORTBL_IDENTITY control PLE car (512, min_rows 80) at {t}: {'SAME' if same else 'DIFFERS'} "
              f"(want {'SAME: D14 bucket' if want_same else 'DIFFERS: stock bucket'}) {'ok' if ok else 'BAD'}")
        ctl_bad += [] if ok else [f"ple-car {t}"]
    verdict = "PASS" if not differ and not ctl_bad else "FAIL"
    print(f"Q38_PORTBL_IDENTITY IDENTITY={verdict} cases {len(cases)} differ {len(differ)} {differ} controls "
          f"moe-xw {'ran' if 'CTL-MOE-XW 96' in h else 'n/a'} ple-car {'ran' if 'CTL-PLE-CAR 8' in h else 'n/a'} "
          f"bad {ctl_bad}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
