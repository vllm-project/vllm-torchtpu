# SPDX-License-Identifier: Apache-2.0
"""Per-shape MoE tile sweep (Qwen3.8-Flash, one v7x chip, per-rank shapes).

Served MoE paths (ixza6 recipe; EP over 8 chips, TP attention):

* decode, <= FUSED_MOE_DECODE_KERNEL_MAX_TOKENS (256) tokens: the fused MoE
  decode kernel (kernels/experimental/fused_moe_decode), one Pallas call per
  rank: shared expert + this rank's 64 routed experts + in-kernel all-reduce.
  Every rank sees all T tokens; routes to other ranks' experts are skipped.
  Bench: the same call on a 1-device mesh, n_ranks=1, 64 local experts,
  global top-10 ids over 512 experts (so the local route load matches a rank's).
  The all-reduce degenerates to a self copy (hp 2560 instead of 3072).
* prefill (>= QWEN4_EXP_PREFILL_SP_MIN_TOKENS = 1024): sequence parallel, the
  adaptive fused EP kernel (kernels/experimental/adaptive_fused_moe) on each
  rank's T/8 rows. Bench: ep=1 mesh, 64 experts, top-10 over 64 (T/8 * 10 local
  pairs, the per-rank pair load under uniform routing).

Knobs (config dict keys):
  decode : fold (FUSED_MOE_DECODE_KERNEL_FOLD off|xla|kernel), i_block (kwarg;
           None = whole experts), nbuf (weight ring depth cap, module constant
           _NUM_BUFFERS), gmm (1 = the GMM path the op takes above MAX_TOKENS).
  prefill: capacity (token tile tile_m, vllm_adapter._TILE_M), prefill_rungs
           (MOE_ADAPTIVE_PREFILL_RUNGS), height_rungs (kernel.HEIGHT_RUNGS),
           nbuf (host.NBUF: 3 -> 3+2 W1/W2 rings, 2 -> 2+1), qchunk (kernel.QCHUNK).

ST_PHASES=decode,prefill restricts phases; ST_TINY=1 builds tiny inputs only
(CPU construction check, no kernel run).
"""

import contextlib
import os

# Served MoE env the shared SERVED_ENV does not carry (non TPU_/VLLM_ names in
# the ixza6 recipe). MOE_ADAPTIVE_PREFILL_RUNGS is read at import time.
_MOE_ENV = {
    "FUSED_MOE_DECODE_KERNEL": "1",
    "FUSED_MOE_DECODE_KERNEL_FOLD": "xla",
    "FUSED_MOE_DECODE_KERNEL_MAX_TOKENS": "256",
    "MOE_FUSED_EP_KERNEL_IMPL": "adaptive",
    "MOE_FUSED_EP_KERNEL_MIN_TOKENS": "1024",
    "MOE_ADAPTIVE_2D_CHANNEL_SCALES": "1",
    "MOE_ADAPTIVE_AG_LEAD_AXIS": "1",
    "MOE_ADAPTIVE_PREFILL_RUNGS": "1",
    "USE_MOE_FUSED_EP_KERNEL": "1",
    "USE_MOE_SPARSE_CORE": "1",
    "ONEHOT_MOE_PERMUTE_THRESHOLD": "2048",
    "QWEN4_EXP_PREFILL_SP": "1",
    "QWEN4_EXP_PREFILL_SP_MOE": "1",
}
for _k, _v in _MOE_ENV.items():
    os.environ.setdefault(_k, _v)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from common import bench, digest, env, try_config  # noqa: E402

TINY = os.environ.get("ST_TINY") == "1"
PHASES = os.environ.get("ST_PHASES", "decode,prefill").split(",")

# Qwen3.8-Flash MoE (megakernel/models/qwen38.py Qwen38Geometry): H 2560,
# 512 experts, top-10, expert intermediate 640, shared expert 640 (BF16 in the
# checkpoint, TP8 slice 80 padded to 128), sigmoid-gated shared expert.
EP = 8
if TINY:
    H, E_TOTAL, TOPK, INTER, I_S = 512, 64, 4, 256, 128
else:
    H, E_TOTAL, TOPK, INTER, I_S = 2560, 512, 10, 640, 128
E_LOCAL = E_TOTAL // EP
F8 = jnp.float8_e4m3fn

DECODE_TOKENS = (4, 8, 16, 32, 64, 96, 128)
PREFILL_TOKENS = (1024, 2048, 4096, 8192)


@contextlib.contextmanager
def patched(obj, **values):
    old = {k: getattr(obj, k) for k in values}
    try:
        for k, v in values.items():
            setattr(obj, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def _flag(outs):
    """0.0 at run time, but data dependent on the previous call's output."""
    leaf = jax.tree_util.tree_leaves(outs)[0]
    return (leaf.reshape(-1)[0].astype(jnp.float32) > 1e30).astype(jnp.float32)


def measure(fwd, ops):
    """fwd(ops, flag) -> outs; returns (us, digest). ``ops`` are copied first:
    the carry is donated, and the caller's arrays serve the next config."""
    ops = jax.tree_util.tree_map(jnp.copy, ops)
    outs0 = jax.jit(lambda o: fwd(o, jnp.float32(0)))(ops)
    dg = digest(outs0)

    def body(carry):
        o, prev = carry
        return o, fwd(o, _flag(prev))

    us, _ = bench(body, (ops, outs0))
    return us, dg


def _quant(w, axis):
    amax = jnp.max(jnp.abs(w), axis=axis, keepdims=True)
    scale = jnp.where(amax == 0, 1.0, amax / float(jnp.finfo(F8).max))
    return (w / scale).astype(F8), scale.astype(jnp.float32)


@jax.jit
def _expert_weights():
    """This rank's 64 experts, FP8 per output channel (K-major)."""
    k1, k2 = jax.random.split(jax.random.key(0))
    w1, s1 = _quant(jax.random.normal(k1, (E_LOCAL, H, 2 * INTER)) * 0.02, 1)
    w2, s2 = _quant(jax.random.normal(k2, (E_LOCAL, INTER, H)) * 0.02, 1)
    return w1, w2, s1.reshape(E_LOCAL, 2 * INTER), s2.reshape(E_LOCAL, H)


def _shared_weights():
    rng = np.random.default_rng(1)

    def bf16(shape, scale):
        return jnp.asarray(rng.normal(size=shape) * scale, jnp.bfloat16)

    # gate | 0-pad | up | 0-pad as pad_shared_intermediate lays out 80 -> 128.
    sw1 = bf16((H, 2 * I_S), 0.02)
    sw2 = bf16((I_S, H), 0.02)
    return (
        sw1,
        jnp.ones((1, 2 * I_S), jnp.float32),
        sw2,
        jnp.ones((1, H), jnp.float32),
        bf16((H, 1), 0.02),  # shared_expert_gate
    )


def _routing(tokens, num_experts, seed):
    """Uniform top-k without replacement over `num_experts` (global ids)."""
    rng = np.random.default_rng(seed)
    ids = np.argsort(rng.random((tokens, num_experts)), axis=1)[:, :TOPK]
    w = rng.random((tokens, TOPK)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)
    return jnp.asarray(w), jnp.asarray(ids.astype(np.int32))


def _hidden(tokens, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(tokens, H)) * 0.5, jnp.bfloat16)


# --------------------------------------------------------------------------- #
# decode: fused MoE decode kernel


def _decode_fwd(mesh, cfg):
    from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
        fused_moe_decode,
    )
    from vllm_torchtpu.layers.adapter import fused_moe_decode as adapter

    fold = cfg.get("fold", "xla")
    i_block = cfg.get("i_block")
    gmm = cfg.get("gmm", 0)

    def local(rank, x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, sg, tw, ti):
        s1 = s1.reshape(E_LOCAL, 1, 1, 2 * INTER)
        s2 = s2.reshape(E_LOCAL, 1, 1, H)
        if gmm:
            return adapter._gmm_layer(
                x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, sg, tw, ti, rank[0, 1],
                topk=TOPK, shared_gated=True, routed_scale=1.0,
            )
        if fold != "off":
            return fused_moe_decode(
                rank, x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, sg, tw, ti,
                n_ranks=1, expert_start="rank", fold=True,
                gate_in_kernel=fold == "kernel", i_block=i_block,
            )
        return fused_moe_decode(
            rank[0, :1], x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, sg, tw, ti,
            n_ranks=1, expert_start=rank[0, 1], i_block=i_block,
        )

    sm = jax.shard_map(
        local, mesh=mesh, in_specs=(P(),) * 13, out_specs=P(), check_vma=False
    )

    def fwd(ops, flag):
        rank, x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, sg, tw, ti = ops
        # The flag keeps the routing prologue (XLA on fold=off) inside the loop.
        return sm(rank, x, w1, w2, s1, s2, sw1, ss1, sw2, ss2, sg, tw + flag,
                  ti + flag.astype(jnp.int32))

    return fwd


DECODE_CONFIGS = (
    {"fold": "xla"},  # served default: FOLD=xla, whole experts, ring 6
    {"fold": "off"},
    {"fold": "kernel"},
    {"fold": "xla", "i_block": 128},
    {"fold": "xla", "nbuf": 4},
    {"gmm": 1},
)


def _run_decode(emit, mesh, weights, shared):
    from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
        fused_moe_decode_kernel as fmd,
    )

    rank = jnp.zeros((1, 2), jnp.int32)  # [[mesh index, first global expert]]
    for tokens in DECODE_TOKENS:
        bucket = {"phase": "decode", "rows": tokens}
        tw, ti = _routing(tokens, E_TOTAL, seed=tokens)
        x = _hidden(tokens, seed=1000 + tokens)
        for i, cfg in enumerate(DECODE_CONFIGS):

            def fn(cfg=cfg):
                fwd = _decode_fwd(mesh, cfg)
                ops = (rank, x, *weights, *shared, tw, ti)
                with patched(fmd, _NUM_BUFFERS=cfg.get("nbuf", fmd._NUM_BUFFERS)):
                    return measure(fwd, ops)

            try_config(emit, bucket, cfg, fn, default=i == 0)


# --------------------------------------------------------------------------- #
# prefill: adaptive fused EP kernel (sequence parallel, T/8 rows a rank)


def _prefill_fwd(mesh, cfg):
    from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
        adaptive_fused_moe,
    )

    capacity = cfg.get("capacity", 128)

    def fwd(ops, flag):
        x, w1, w2, s1, s2, gating = ops
        return adaptive_fused_moe(
            x + flag.astype(x.dtype), w1, w2, s1, s2, gating + flag,
            rank=jnp.zeros((1, 1), jnp.int32), topk=TOPK, renormalize=True,
            mesh=mesh, capacity=capacity, block=None, ragged_stride=None,
            act_fn="silu",
        )

    return fwd


PREFILL_CONFIGS = (
    {"capacity": 128},  # served default: _TILE_M 128, PREFILL_RUNGS=1, 3+2 rings
    {"capacity": 256},
    {"capacity": 64},
    {"capacity": 128, "prefill_rungs": 0},
    {"capacity": 128, "height_rungs": 2},
    {"capacity": 128, "nbuf": 2},
    {"capacity": 128, "qchunk": 640},
)


def _run_prefill(emit, mesh, weights):
    from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
        host,
        kernel,
        layer,
    )

    for tokens in PREFILL_TOKENS:
        rows = tokens // EP
        bucket = {"phase": "prefill", "tokens": tokens, "rows": rows}
        x = _hidden(rows, seed=2000 + tokens)
        gating = jnp.asarray(
            np.random.default_rng(tokens).normal(size=(rows, E_LOCAL)), jnp.float32
        )
        for i, cfg in enumerate(PREFILL_CONFIGS):

            def fn(cfg=cfg):
                kernel._BUILD_CACHE.clear()
                layer._LAYER_SM_CACHE.clear()
                fwd = _prefill_fwd(mesh, cfg)
                ops = (x, *weights, gating)
                consts = dict(
                    PREFILL_RUNGS=bool(cfg.get("prefill_rungs", 1)),
                    HEIGHT_RUNGS=cfg.get("height_rungs", 4),
                    QCHUNK=cfg.get("qchunk", 512),
                )
                try:
                    with patched(kernel, **consts), patched(
                        host, NBUF=cfg.get("nbuf", host.NBUF)
                    ):
                        return measure(fwd, ops)
                finally:
                    kernel._BUILD_CACHE.clear()
                    layer._LAYER_SM_CACHE.clear()

            try_config(emit, bucket, cfg, fn, default=i == 0)


def build_inputs():
    """Weights for both phases (shared); cheap to call on CPU with ST_TINY=1."""
    return _expert_weights(), _shared_weights()


def main(emit):
    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME

    mesh = Mesh(np.asarray(jax.devices()[:1]), (EP_AXIS_NAME,))
    (w1, w2, s1, s2), shared = build_inputs()
    if TINY:
        print("bench_moe tiny inputs:", w1.shape, w2.shape, s1.shape, s2.shape,
              [a.shape for a in shared])
        return
    if "decode" in PHASES:
        with env(FUSED_MOE_DECODE_KERNEL_FOLD="xla"):
            _run_decode(emit, mesh, (w1, w2, s1, s2), shared)
    if "prefill" in PHASES:
        _run_prefill(emit, mesh, (w1, w2, s1, s2))
