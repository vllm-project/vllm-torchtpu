"""shard_map wrapper for the TP x DCP megakernel: every operand carries a leading rank axis [R, ...]
(per-rank arrays differ along it; replicated ones are repeated), and one step = the megakernel (caches
read-only) followed by the barrier-free cache writer (caches aliased + donated). Used by the tests and
the probes; the vLLM adapter builds its own global arrays the same way."""

import jax
from jax.sharding import PartitionSpec as P

from . import kernel as K


def make(cfg, mesh, axis="d", donate=True, **build_kw):
    """Returns (fn, info). fn(*operands), every operand [R, ...]: fixed (8), weights (info['w_names']),
    caches (info['c_names']) -> (out [R, M, H], *caches [R, ...], *debug [R, ...])."""
    call, info = K.build(cfg, axis_name=axis, **build_kw)
    wkw = {k: build_kw[k] for k in ("nb", "max_pages", "bmax", "interpret", "stack_c") if k in build_kw}
    writer, _ = K.build_writer(cfg, axis_name=axis, **wkw)
    n_w, n_c = len(info["w_names"]), len(info["c_names"])
    debug = build_kw.get("debug", False)

    def body(*ops):
        ops = [o[0] for o in ops]
        scal, rmeta, qmeta, bt = ops[:4]
        caches = ops[8 + n_w:]
        outs = call(*ops)
        out, nrow, rrow, irec = outs[:4]
        new = writer(scal, rmeta, bt, nrow, rrow, irec, *caches)
        return tuple(o[None] for o in (out, *new, *outs[4:]))

    n_in = 8 + n_w + n_c
    n_out = 1 + n_c + (4 if debug else 0)
    fn = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(axis),) * n_in,
                               out_specs=(P(axis),) * n_out, check_vma=False),
                 donate_argnums=tuple(range(8 + n_w, n_in)) if donate else ())
    return fn, info


def make_draft(cfg, mesh, axis="d", donate=True, **build_kw):
    """The MTP drafter (cfg.V > 0, L = 3 passes). fn(*operands), every operand [R, ...]: fixed (12:
    scal, rmeta, qmeta [3, ...], bt, rvec [3, ...], h0, rope_tab [3, ...], smalls, dmeta, embed shard,
    lm_head shard, eh_proj^T shard), weights (info['w_names'], the MTP layer 3x), the draft layer's
    caches ONCE (info['c_unique']) -> (tokens [R, M, 128], *caches, *debug). The kernel reads the
    caches as one view per pass; the writer applies pass 0's rows (the only live KV writes)."""
    import dataclasses
    # 9-bank weight ring: the drafter streams one layer per pass, and 11 banks overflow VMEM (68.2 of
    # 63.9 MiB with the drafter's scratch; FULL AOT, 2026-10-09)
    build_kw.setdefault("tiles", K.Tiles(nbank=9))
    call, info = K.build(cfg, axis_name=axis, draft=True, **build_kw)
    cfg1 = dataclasses.replace(cfg, L=1, V=0)
    wkw = {k: build_kw[k] for k in ("nb", "max_pages", "bmax", "interpret", "stack_c") if k in build_kw}
    writer, winfo = K.build_writer(cfg1, axis_name=axis, **wkw)
    uniq = list(winfo["c_names"])                    # (kind, 0)
    pick = [uniq.index((k, 0)) for k, _ in info["c_names"]]
    info["c_unique"] = uniq
    n_w, n_u = len(info["w_names"]), len(uniq)
    debug = build_kw.get("debug", False)

    def body(*ops):
        ops = [o[0] for o in ops]
        fixed, w, cu = ops[:12], ops[12:12 + n_w], ops[12 + n_w:]
        flat = list(fixed)                       # per-pass metadata flattened (rank <= 2 SMEM refs)
        for i_ in (1, 2, 4):
            flat[i_] = fixed[i_].reshape((-1,) + fixed[i_].shape[2:])
        flat[6] = fixed[6].reshape((-1,) + fixed[6].shape[2:])
        outs = call(*flat, *w, *[cu[i] for i in pick])
        tok, nrow, rrow, irec = outs[:4]
        new = writer(fixed[0][0], fixed[1][0], fixed[3], nrow[0:1], rrow[0:1], irec[0:1], *cu)
        return tuple(o[None] for o in (tok, *new, *outs[4:]))

    n_in = 12 + n_w + n_u
    n_out = 1 + n_u + (4 if debug else 0)
    fn = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(axis),) * n_in,
                               out_specs=(P(axis),) * n_out, check_vma=False),
                 donate_argnums=tuple(range(12 + n_w, n_in)) if donate else ())
    return fn, info
