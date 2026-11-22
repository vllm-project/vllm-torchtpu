# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Multi-controller helpers shared by bench.py and accuracy_tpu.py: distributed init, the 1-D mesh, global inputs
(identical x / scores on every rank, as the layer sees them), per-rank random fp8 weights generated on device."""

import os

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

try:
    from . import refs, spec
except ImportError:
    import refs  # noqa: E402
    import spec  # noqa: E402

PER, REP = P("d"), P()


def init():
    """jax.distributed.initialize from the 2-node recipe's COORD / NPROC / TPU_WORKER_ID when present."""
    nproc = int(os.environ.get("NPROC", "1"))
    if nproc > 1 and not jax.distributed.is_initialized():
        jax.distributed.initialize(os.environ["COORD"], nproc, int(os.environ.get("TPU_WORKER_ID", "0")))
    return jax.process_index()


def mesh():
    devs = sorted(jax.devices(), key=lambda d: d.id)
    return Mesh(np.asarray(devs), ("d",))


def global_from_host(msh, host, spec_=PER):
    """Global array from a host array that every process holds identically."""
    sh = NamedSharding(msh, spec_)
    return jax.make_array_from_callback(host.shape, sh, lambda idx: host[idx])


def cell_inputs(msh, dep, m, kind, n_valid):
    """(x, scores, bias, n_valid, rank, e0) as global arrays for the kernel's shard_map specs, plus the picks."""
    n = len(msh.devices.ravel())
    scores, bias, p = spec.scores_bias(m, kind, spec.seed_of(m, dep.name, kind))
    rng = np.random.default_rng(spec.seed_of(m, dep.name, kind, 1))
    x = rng.standard_normal((m, dep.h)).astype(ml_dtypes.bfloat16)
    xs = global_from_host(msh, np.tile(x, (n, 1)))
    ss = global_from_host(msh, np.tile(scores, (n, 1)))
    bb = global_from_host(msh, bias, REP)
    nv = global_from_host(msh, np.full((n,), n_valid, np.int32))
    rk = global_from_host(msh, np.arange(n, dtype=np.int32))
    e0 = global_from_host(msh, (np.arange(n, dtype=np.int32) * dep.e_l) if dep.ep else np.zeros((n,), np.int32))
    return (xs, ss, bb, nv, rk, e0), (x, scores, bias, p)


def rank_weights_global(msh, dep, seed=0):
    """Per-rank weights (refs.rank_weights with key fold_in(seed, rank)), concatenated along axis 0 (PER specs)."""
    n = len(msh.devices.ravel())
    rk = global_from_host(msh, np.arange(n, dtype=np.int32))

    def gen(r):
        return refs.rank_weights(jax.random.fold_in(jax.random.key(seed), r[0]), dep)

    f = jax.jit(jax.shard_map(gen, mesh=msh, in_specs=(PER,), out_specs=(PER,) * 8, check_vma=False))
    return f(rk)


KERNEL_SPECS = (PER, PER, REP, PER, PER) + (PER,) * 8 + (PER,)


def kernel_args(kargs, w):
    """cell_inputs args (x, scores, bias, nv, rank, e0) + weights -> the kernel's positional order."""
    return (*kargs[:5], *w, kargs[5])


def kernel_call(fn, dep, n_ranks, **extra):
    """body(x, scores, bias, nv, rank, w13, w2, s13, s2, sw13, ss13, sw2, ss2, e0) -> fn(...) with production flags."""

    def body(x, scores, bias, nv, rank, w13, w2, s13, s2, sw13, ss13, sw2, ss2, e0):
        return fn(x, scores, bias, nv.reshape(()), rank.reshape(()), w13, w2, s13, s2, sw13, ss13, sw2, ss2,
                  e0.reshape(()), routed_scale=spec.ROUTED_SCALE, topk=spec.TOPK, n_ranks=n_ranks, fast=True,
                  hbm_operands=False, **extra)

    return body


def local_shard(arr, index=0):
    """The addressable shard holding global shard `index` along axis 0 (None if not on this process)."""
    for s in arr.addressable_shards:
        if s.index[0].start in (None, 0) and index == 0:
            return np.asarray(s.data)
        start = s.index[0].start or 0
        size = s.data.shape[0]
        if start == index * size:
            return np.asarray(s.data)
    return None


def as_f32(a):
    return np.asarray(jnp.asarray(a).astype(jnp.float32))
