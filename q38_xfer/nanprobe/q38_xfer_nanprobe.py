"""Test-only TPU probe: which step of the staged PD repack path canonicalises FP8 NaN bytes (0xFF -> 0x7F)?

The rowdump riders and the IDENTICAL re-run (j-823b51fc) showed every 0xFF byte of compressed keys and GDN state
arriving in the decode pool as 0x7F, with or without the DMA row gather. This runs a tensor holding every byte value
(0..255 in every row) through each step on its own, on one TPU device, and compares the bytes read back:

  a  jax.default_backend() / devices (the DMA gather's default depends on it)
  b  host -> device -> host round trip of an FP8 torch tensor (write_carrier's path)
  c  torch copy_ FP8 -> FP8 on device
  d  pallas.jax_op identity (FP8 in, FP8 out): the torch <-> JAX op boundary
  e  jax_op XLA gather of the FP8 array (the old repack gather)
  f  jax_op DMA row gather (pd_staging.dma_row_gather), with a trace-time marker
  g1 jax_op bitcast FP8 -> uint8 (uint8 out);  g2 jax_op bitcast uint8 -> FP8 (uint8 tensor in)
  h  the production repack op (StagedConsumer._build_op over make_repack, compiled, donated, copy_ writeback):
     default gather, and h2 with the XLA gather forced
  i  jax_op identity with uint8 in and out (the candidate fix's boundary)

Prints one ``Q38_NANPROBE {json}`` per step: equal or not, differing bytes, top (want, got) pairs. Writes all to
$CDK_OUTPUT_DIR/q38_xfer_nanprobe.json when set. Never raises out of a step (an error is reported for that step)."""
import collections
import json
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = []


def emit(step, **kw):
    rec = dict(step=step, **kw)
    OUT.append(rec)
    print("Q38_NANPROBE " + json.dumps(rec, default=str), flush=True)


def compare(step, got, want, **extra):
    got = np.asarray(got, dtype=np.uint8).reshape(-1)
    want = np.asarray(want, dtype=np.uint8).reshape(-1)
    if got.shape != want.shape:
        emit(step, equal=False, error=f"shape {got.shape} != {want.shape}", **extra)
        return
    diff = got != want
    pairs = collections.Counter(zip(want[diff].tolist(), got[diff].tolist()))
    emit(step, equal=not diff.any(), diff_bytes=int(diff.sum()), bytes=int(want.size),
         top_pairs_want_got=[[a, b, c] for (a, b), c in pairs.most_common(8)], **extra)


def guarded(step, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - a probe step must not stop the others
        emit(step, error=f"{type(e).__name__}: {e}"[:400], tb=traceback.format_exc()[-1200:])


def main():
    import torch
    import jax
    import jax.numpy as jnp
    from torch_tpu._internal import pallas, sync

    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging as st

    dev = "tpu"
    R = 128
    host = np.tile(np.arange(256, dtype=np.uint8), R * 4).reshape(R, 1, 4, 256)
    F8 = torch.float8_e4m3fn

    def to_dev(arr, dtype):
        t = torch.from_numpy(np.ascontiguousarray(arr)).view(dtype).to(dev)
        sync.synchronize([t], wait=True)
        return t

    def back(t):
        sync.synchronize([t], wait=True)
        return t.cpu().contiguous().view(torch.uint8).numpy()

    emit("a", backend=jax.default_backend(), devices=[str(d) for d in jax.devices()][:4],
         torch_tpu_has_jax_op=hasattr(pallas, "jax_op"))
    X = torch.zeros((R, 1, 4, 256), dtype=F8, device=dev)
    X.copy_(torch.from_numpy(host).view(F8).to(dev))
    sync.synchronize([X], wait=True)
    guarded("b", lambda: compare("b", back(X), host))

    def step_c():
        Y = torch.empty_like(X)
        Y.copy_(X)
        compare("c", back(Y), host)
    guarded("c", step_c)

    def mk(name, fn, fake):
        op = pallas.jax_op("pallas::q38_nanprobe_" + name, fn)
        op.register_fake(fake)
        return op

    # torch_tpu's pallas.jax_op reads the JAX function's argument annotations (j-bd653083: "Missing argument type
    # annotation"), so every probe function is a def with jax.Array annotations, as pd_staging's repack_fn is.
    def f_ident(a: jax.Array) -> jax.Array:
        return a

    def f_xla_gather(a: jax.Array, i: jax.Array) -> jax.Array:
        return a.reshape(-1, 4, 256)[i]

    def f_bc_f8_u8(a: jax.Array) -> jax.Array:
        return jax.lax.bitcast_convert_type(a, jnp.uint8)

    def f_bc_u8_f8(b: jax.Array) -> jax.Array:
        return jax.lax.bitcast_convert_type(b, jnp.float8_e4m3fn)

    def f_ident_u8(b: jax.Array) -> jax.Array:
        return b

    guarded("d", lambda: compare("d", back(mk("ident", f_ident, lambda a: torch.empty_like(a))(X)), host))
    idx_np = np.arange(R, dtype=np.int32)[::-1].copy()
    idx = torch.from_numpy(idx_np).to(dev)
    want_g = host.reshape(R, 4, 256)[idx_np]
    guarded("e", lambda: compare("e", back(mk(
        "xla_gather", f_xla_gather, lambda a, i: torch.empty((R, 4, 256), dtype=a.dtype, device=a.device))(X, idx)), want_g))
    TRACED = {}

    def dma_fn(a: jax.Array, i: jax.Array) -> jax.Array:
        TRACED["dma"] = True
        print("Q38_NANPROBE_MARK dma_row_gather traced backend=%s" % jax.default_backend(), flush=True)
        return st.dma_row_gather(a.reshape(-1, 4, 256), i)
    guarded("f", lambda: compare("f", back(mk(
        "dma_gather", dma_fn, lambda a, i: torch.empty((R, 4, 256), dtype=a.dtype, device=a.device))(X, idx)), want_g,
        traced=bool(TRACED.get("dma"))))
    guarded("g1", lambda: compare("g1", back(mk(
        "bc_f8_u8", f_bc_f8_u8,
        lambda a: torch.empty(a.shape, dtype=torch.uint8, device=a.device))(X)), host))
    U = to_dev(host, torch.uint8)
    guarded("g2", lambda: compare("g2", back(mk(
        "bc_u8_f8", f_bc_u8_f8,
        lambda b: torch.empty(b.shape, dtype=F8, device=b.device))(U)), host))
    guarded("i", lambda: compare("i", back(mk(
        "ident_u8", f_ident_u8, lambda b: torch.empty_like(b))(U)), host))

    def repack_step(step, gather):
        fn = st.make_repack(src_block=64, dst_block=64, src_gdn=None, dst_gdn=None, kv_head=0, v_head_start=0,
                            channels=[0], gather=gather)

        class Holder:
            pass
        h = Holder()
        h._fn = fn
        run = st.StagedConsumer._build_op(h)
        staging = X.reshape(R // 64, 64, 1, 4, 256)
        pool = torch.zeros((4, 64, 1, 4, 256), dtype=F8, device=dev)
        sync.synchronize([pool], wait=True)
        # Production-shaped inputs (repack_chunks pads every kind to its fixed program shape; an empty kind
        # rewrites the null row). Main KV: staging rows 0..63 -> pool page 3; copies: staging rows 0..127 ->
        # pool rows 64..191 (pages 1-2), compared byte for byte.
        main_s = np.arange(64, dtype=np.int64)
        main_d = np.arange(64, dtype=np.int64) + 192
        copy_s = np.arange(R, dtype=np.int64)
        copy_d = np.arange(R, dtype=np.int64) + 64
        none = np.zeros((0,), dtype=np.int64)
        for chunk in st.repack_chunks(main_s, main_d, copy_s, copy_d, none, none):
            args = [torch.from_numpy(np.ascontiguousarray(v, dtype=np.int32)).to(dev) for v in chunk]
            run(pool, staging, *args)
        got = back(pool).reshape(4 * 64, 1024)[64:64 + R]
        compare(step, got, host.reshape(R, 1024), gather=getattr(gather, "__name__", str(gather)))
    guarded("h", lambda: repack_step("h", None))
    guarded("h2", lambda: repack_step("h2", lambda a, f: a[f]))
    if os.environ.get("CDK_OUTPUT_DIR"):
        with open(os.path.join(os.environ["CDK_OUTPUT_DIR"], "q38_xfer_nanprobe.json"), "w") as fh:
            json.dump(OUT, fh, indent=1, default=str)
    # The bench recipe scores a script run OK only if <run_dir>/run01_A/result.json has ok and cases (as
    # q38_xfer_gate_result.py writes); ok = every step produced a result without an error (a diagnostic, not a gate).
    run_dir = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"), "q38f-bench", "script", "run01_A")
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "result.json"), "w") as fh:
            json.dump({"ok": all("error" not in r for r in OUT),
                       "cases": [{"step": r["step"], "equal": r.get("equal"), "error": bool(r.get("error"))} for r in OUT]}, fh)
    except OSError as e:
        print("Q38_NANPROBE result.json not written: %s" % e, flush=True)
    print("Q38_NANPROBE_DONE steps=%d" % len(OUT), flush=True)


if __name__ == "__main__":
    main()
