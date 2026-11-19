# SPDX-License-Identifier: Apache-2.0
"""q38 L1-G1 flag-off identity probe (CPU; run once per tree, the job diffs the lines).

  python3 tests/q38_l1g1/q38_g1_identity.py <tree root>

Builds the real `build_wholestep_call` closures of the tree (`call` and D07-b1's `call_shared`; write True / False) with
the D07-b1 test's stubs for the kernel, QSA inputs and writers (tests/megakernel_runtime/test_l1d07b1_gdn_meta_shared.py,
present in both trees), and prints one line per case: sha256 of the jaxpr text, of the lowered StableHLO text, of the
output specs and of the output bytes. VLLM_TPU_MK_GDN_COMMIT1 is taken from the environment (the job runs the head with it
unset and set, the base unset).
"""

from __future__ import annotations

import hashlib
import os
import sys

root = os.path.abspath(sys.argv[1])
sys.path[:0] = [os.path.join(root, "src"), root]
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from _pytest.monkeypatch import MonkeyPatch  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from tests.megakernel_runtime import test_l1d07b1_gdn_meta_shared as d7  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc  # noqa: E402

assert wsc.__file__.startswith(os.path.join(root, "src")), (wsc.__file__, root)


def sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else str(b).encode()).hexdigest()[:16]


def main() -> None:
    from types import SimpleNamespace

    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import QSALayerCache

    mesh = Mesh(np.array(jax.devices()[: d7.R]), ("tp",))
    md, layers, gdn_ids = d7._metadata_objects()
    per_layer = [np.asarray(t.numpy()) for t in d7._per_layer_operands(md, layers, gdn_ids)]
    shared = [np.asarray(t.numpy()) for t in d7._module(layers, gdn_ids)._gdn_meta_shared(md)]
    rng = np.random.default_rng(11)
    hidden = rng.standard_normal((d7.R * d7.B, d7.H)).astype(np.float32)
    w0 = rng.standard_normal((d7.R * 4, d7.H)).astype(np.float32)
    pool = np.zeros((d7.R * d7.POOL,), np.int32)
    tabs = [rng.integers(0, 50, (48 * n,)).astype(np.int32) for _ in range(d7.N_ATT) for n in (8, 32, 32)]
    tail = [*tabs, rng.integers(0, 300, (d7.B,)).astype(np.int32), np.arange(d7.B + 1, dtype=np.int32),
            rng.integers(0, 300, (d7.B,)).astype(np.int32), rng.standard_normal((16, 128)).astype(np.float32),
            *[rng.standard_normal((n,)).astype(np.float32) for _ in range(d7.N_ATT) for n in (256, 128)]]
    lq = [QSALayerCache(compressed_slot=o % 2, main_slot=(o + 1) % 2) for o in range(d7.N_ATT)]
    flag = os.environ.get("VLLM_TPU_MK_GDN_COMMIT1", "")
    for dedup in (False, True):
        for write in (True, False):
            mp = MonkeyPatch()
            try:
                d7._stub(mp)
                if dedup:
                    mp.setenv("VLLM_TPU_MK_GDN_META_DEDUP", "1")
                else:
                    mp.delenv("VLLM_TPU_MK_GDN_META_DEDUP", raising=False)
                call, in_specs, out_specs, don = wsc.build_wholestep_call(
                    mesh, SimpleNamespace(att_every=4), d7.B, None, lq, L=d7.L, budget=64, max_groups=1024,
                    norm_eps=1e-6, rotary_dim=64, write=write, interpret=True)
                args = (hidden, w0, pool, *(shared if dedup else per_layer), *tail)
                jx = str(jax.make_jaxpr(call)(*args))
                hlo = jax.jit(call).lower(*args).as_text()
                outs = jax.jit(call)(*args)
                ob = b"".join(np.ascontiguousarray(np.asarray(o)).tobytes() for o in outs)
                print(f"Q38G1ID {call.__name__} write={int(write)} jaxpr {sha(jx)} hlo {sha(hlo)} specs "
                      f"{sha(repr((in_specs, out_specs, don)))} outs {len(outs)} {sha(ob)} (flag '{flag}')", flush=True)
            finally:
                mp.undo()
    head_cases(mesh, flag)


def head_cases(mesh, flag: str) -> None:
    """The head group call (`build_segment_call(head=True)`, L = 3, B = 16) and a body segment (L = 4), with the kernel and
    the scatter stubbed (the stubs fold everything they receive into their outputs)."""
    import jax.numpy as jnp
    from vllm_torchtpu.megakernel_runtime import qwen38_segment_call as sc

    R, B, H, NBK = 2, 16, 8, 64
    NS = B // 4

    def fn(pos, x, *rest):
        xs, w = rest[:3], rest[3:]
        pool, rb, hi = xs
        bump = jnp.sum(rb).astype(jnp.float32) + jnp.sum(hi).astype(jnp.float32) + jnp.sum(w[0])
        return (x + bump, jnp.broadcast_to(rb[None], (R,) + rb.shape).astype(jnp.float32),
                jnp.broadcast_to(hi[None], (R,) + hi.shape).astype(jnp.float32))

    fn.extra_names = ("gdn_pool", "gdn_rb", "gdn_hi")

    def scatter(pool, c, r, wb, layout=None, interpret=False):
        flat = jnp.concatenate([c.reshape(-1), r.reshape(-1), wb.reshape(-1).astype(jnp.float32)]).astype(pool.dtype)
        return pool.at[: flat.shape[0]].set(flat)

    rng = np.random.default_rng(5)
    for L, head in ((3, True), (4, False)):
        NL = L if head else L - 1
        mp = MonkeyPatch()
        try:
            mp.setattr(sc.qwen38, "segment_spec", lambda *a, **k: None)
            mp.setattr(sc, "make_flat_segment_step", lambda mesh, spec, ex, extra_outs=(): (fn, None, None, 1))
            mp.setattr(sc, "scatter_gdn_ckpts_fast", scatter)
            call, in_specs, out_specs, don = sc.build_segment_call(mesh, None, L, B, None, interpret=True, head=head)
            hidden = rng.standard_normal((R * B, H)).astype(np.float32)
            attn = rng.standard_normal((R * B, H)).astype(np.float32)
            w0 = rng.standard_normal((R * 4, H)).astype(np.float32)
            pool = np.zeros((R * 4096,), np.float32)
            meta = []
            for ln in range(NL):
                ck = rng.integers(1, NBK, (NS, 4)).astype(np.int32)
                meta += [ck[:, 0].copy(), rng.integers(0, 4, NBK).astype(np.int32), ck,
                         (np.arange(NS + 1) * 4).astype(np.int32), np.full(NS, 50, np.int32)]
            args = ((hidden,) if head else (hidden, attn)) + (w0, pool, *meta)
            jx = str(jax.make_jaxpr(call)(*args))
            hlo = jax.jit(call).lower(*args).as_text()
            outs = jax.jit(call)(*args)
            ob = b"".join(np.ascontiguousarray(np.asarray(o)).tobytes() for o in outs)
            print(f"Q38G1ID seg_{'head' if head else 'body'}_l{L} {'write=1' if head else 'keep=1'} jaxpr {sha(jx)} hlo {sha(hlo)} specs "
                  f"{sha(repr((in_specs, out_specs, don)))} outs {len(outs)} {sha(ob)} (flag '{flag}')", flush=True)
        finally:
            mp.undo()


if __name__ == "__main__":
    main()
