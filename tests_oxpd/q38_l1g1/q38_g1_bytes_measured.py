# SPDX-License-Identifier: Apache-2.0
"""q38 L1-G1: MEASURED (CPU interpret) DMA bytes of today's GDN checkpoint scatter vs the G1 commit.

  python3 tests/q38_l1g1/q38_g1_bytes_measured.py <tree root>

Runs the real kernels (`qwen38_scatter.scatter_gdn_ckpts_fast`, `qwen38_gdn_commit.commit_gdn_ckpts_fast`) in Pallas
interpret mode with `pltpu.make_async_copy` wrapped so that every DMA `start()` that executes adds its static byte count
to a host counter (`jax.debug.callback`): loads (bf16 / f32 sources) and pool writes (fp8 sources) are counted
separately, so pad rows, null ids and the commit's skipped checkpoints count only if their DMAs really run. The XLA prep
around each kernel (today: the f32 conv-window build `_conv_windows`; G1: the accept gather and the src / dst ids) is
measured with XLA:CPU's cost analysis ('bytes accessed'). Real per-request geometry (6 x 128 x 128 bf16 state, QKV 1280,
8 history rows, 400 pool rows of 1024 B per checkpoint); NL = 2 layers (bytes scale linearly in NL: the 33-layer column
multiplies by 33 / 2); rows 16 / 32 / 48 / 64 with every request live, and a half-pad case per rows count. Each case also
checks the measured kernel bytes against the closed form (`q38_g1_bytes.py`).
Last line: Q38G1BM SUMMARY PASS|FAIL.
"""

from __future__ import annotations

import math
import os
import sys

root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
sys.path[:0] = [os.path.join(root, "src"), root]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

from vllm_torchtpu.megakernel_runtime import qwen38_gdn_commit as gc  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_gdn_state as gs  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_scatter as sc  # noqa: E402

assert gc.__file__.startswith(os.path.join(root, "src")), gc.__file__
T, NL = 4, 2
COUNT = {"load": 0, "store": 0, "n_load": 0, "n_store": 0}
_real = pltpu.make_async_copy


def _add(kind: str, nb: int) -> None:
    COUNT[kind] += nb
    COUNT["n_" + kind] += 1


class _Counted:
    def __init__(self, d, kind: str, nb: int):
        self.d, self.kind, self.nb = d, kind, nb

    def start(self):
        jax.debug.callback(lambda: _add(self.kind, self.nb))
        self.d.start()

    def wait(self):
        self.d.wait()


def counting(src, dst, sem):
    d = _real(src, dst, sem)
    nb = math.prod(src.shape) * jnp.dtype(src.dtype).itemsize
    kind = "store" if jnp.dtype(src.dtype).itemsize == 1 else "load"
    return _Counted(d, kind, nb)


def _xla_bytes(f, *args) -> int:
    ca = jax.jit(f).lower(*args).compile().cost_analysis()
    ca = ca[0] if isinstance(ca, (list, tuple)) else ca
    return int(ca.get("bytes accessed", 0))


def measure(rows: int, pads: int, seed: int) -> dict:
    NS = rows // T
    rng = np.random.default_rng(seed)
    nmb = 1 + NL * NS * T
    shape = (nmb * 4, 128, 1, 4, 256)
    layout = gs.Qwen38GdnPoolLayout(pool_shape=shape, pool_dtype="float8_e4m3fn")
    pool = jax.lax.bitcast_convert_type(jnp.asarray(rng.integers(0, 256, shape, dtype=np.uint8)), jnp.float8_e4m3fn)
    conv = jnp.asarray(rng.standard_normal((NL, NS, 8, 1280)), jnp.float32)
    rec = jnp.asarray(rng.standard_normal((NL, NS * T * 6, 128, 128)), jnp.float32).astype(jnp.bfloat16)
    wb = (1 + np.arange(NL * NS * T, dtype=np.int32)).reshape(NL, NS, T)
    if pads:
        wb[:, NS - pads:] = 0
    acc = jnp.asarray(rng.integers(0, T, (NL, NS)).astype(np.int32))
    wbj = jnp.asarray(wb)
    out = {}
    pltpu.make_async_copy = counting
    try:
        for name, run in (
            ("today", lambda: sc.scatter_gdn_ckpts_fast(pool, conv, rec, wbj, layout=layout, interpret=True)),
            ("g1", lambda: gc.commit_gdn_ckpts_fast(pool, conv, rec, wbj, acc, layout=layout, interpret=True)),
        ):
            for k in COUNT:
                COUNT[k] = 0
            jax.block_until_ready(run())
            jax.effects_barrier()
            out[name] = dict(COUNT)
    finally:
        pltpu.make_async_copy = _real
    out["today"]["prep"] = _xla_bytes(lambda c: gs._conv_windows(c.astype(jnp.float32), T).reshape(NL * NS * T, 3, 1280),
                                      conv)
    out["g1"]["prep"] = _xla_bytes(lambda w, a: gc.commit_plan(w, a), wbj, acc)
    live = NS - pads
    want = {
        "today": dict(load=NL * NS * T * (6 * 128 * 128 * 2 + 3 * 1280 * 4),
                      store=NL * live * T * (3 * 128 + 16) * 1024),
        "g1": dict(load=NL * live * (6 * 128 * 128 * 2 + 8 * 1280 * 4), store=NL * live * (3 * 128 + 16) * 1024),
    }
    out["closed_form_ok"] = all(out[w][k] == want[w][k] for w in want for k in ("load", "store"))
    out["want"] = want
    return out


def main() -> None:
    mib = 2 ** 20
    ok = True
    print("Q38G1BM rows live/NS | today: load store prep total MiB (NL 2) | G1: load store prep total MiB | G1/today | "
          "removed | 33-layer today -> G1 MiB | DMAs today / G1 | closed form")
    for rows in (16, 32, 48, 64):
        for pads in (0, rows // T // 2):
            m = measure(rows, pads, seed=rows + pads)
            a, b = m["today"], m["g1"]
            ta, tb = a["load"] + a["store"] + a["prep"], b["load"] + b["store"] + b["prep"]
            NS = rows // T
            print(f"Q38G1BM rows {rows:2d} live {NS - pads:2d}/{NS:2d} | {a['load'] / mib:6.2f} {a['store'] / mib:6.2f} "
                  f"{a['prep'] / mib:5.2f} {ta / mib:6.2f} | {b['load'] / mib:6.2f} {b['store'] / mib:6.2f} "
                  f"{b['prep'] / mib:5.3f} {tb / mib:6.2f} | {tb / ta:.4f} | {1 - tb / ta:.4f} | "
                  f"{ta * 33 / 2 / mib:7.1f} -> {tb * 33 / 2 / mib:6.1f} | {a['n_load'] + a['n_store']} / "
                  f"{b['n_load'] + b['n_store']} | {'OK' if m['closed_form_ok'] else 'MISMATCH ' + repr(m['want'])}",
                  flush=True)
            ok &= m["closed_form_ok"] and a["load"] > 0 and b["store"] > 0
    print(f"Q38G1BM SUMMARY {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
