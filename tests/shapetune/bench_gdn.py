# SPDX-License-Identifier: Apache-2.0
"""Per-shape GDN tile sweep (Qwen3.8-Flash, one v7x chip, per-rank TP8 heads).

The served stock path: `run_jax_gdn_attention_pooled` (layers/core/
gdn_attention.py) -> `fused_conv1d_gdn` (kernels/gdn/v3/wrapper.py) over the
unified pool, 36 GDN layers a step. Per rank (TP8): 2 kq heads, 6 v heads,
head dim 128, conv width 1280, kernel 4; SSM state BF16 and conv state FP32 in
the pool; MTP K=3 (verify windows of 4 tokens, 4 checkpoints a request).

Buckets:
  decode : rows R = 4 x requests (full verify windows), hoisted step metadata
           (VLLM_TPU_GDN_HOIST_METADATA), raw qkv input (prepare_gdn_input
           keeps the 2D projection at <= slots x 4 rows).
  prefill: one request chunk of T tokens continuing 8192 cached tokens, with a
           tail-fold checkpoint (VLLM_TPU_MAMBA_TAIL_FOLD, unit 256: the served
           non-hoisted `_ckpt` op variant), compact Conv1DInput.

Knobs (config dict keys):
  decode_tile : BATCHED/verify sequences per tile (dynamic tiling picks
                decode_tile_target(min(rows, slots), n_v) capped by VMEM).
                Rows of different sequences are independent: bitwise expected.
  chunk       : PER_SEQ chunk size C (dynamic: largest of 128..1 fitting
                VMEM). Changes the chunked-scan association: NOT bitwise.
  pair        : TPU_GDN_CHUNK_PAIR (tile = 2C, bitwise by test_chunk_pair).
  fold_unit   : VLLM_TPU_MAMBA_TAIL_FOLD_UNIT, only to admit tile 512 (the
                fold unit must be a multiple of the largest prefill tile).

Overrides go through tiling.get_tile_sizes' decode_tile_size /
mixed_tile_size arguments, which the wrapper does not pass today (the
VLLM_TPU_SHAPE_TUNED wiring point, wrapper.py fused_conv1d_gdn).

ST_PHASES=decode,prefill restricts phases; ST_TINY=1 builds inputs only.
ST_GDN_SLOTS (default 24 = served --max-num-seqs) sizes the slot table.
"""

import contextlib
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from common import bench, digest, env, try_config

TINY = os.environ.get("ST_TINY") == "1"
PHASES = os.environ.get("ST_PHASES", "decode,prefill").split(",")
SLOTS = int(os.environ.get("ST_GDN_SLOTS", "24"))

N_KQ, N_V, D_K, D_V, KS = 2, 6, 128, 128, 4
DIM = 2 * N_KQ * D_K + N_V * D_V  # 1280
NUM_SPEC = 3
WINDOW = NUM_SPEC + 1
BLOCK = 512  # manager block (pool page); fold unit 256 divides it
KBS, PAYLOAD, LANES = 512, (1, 4), 256  # 1 KiB a pool token
NUM_BLOCKS = 40 if TINY else 136
CONTEXT = 8192

DECODE_ROWS = (4, 8, 16, 32, 64, 96, 128)
PREFILL_TOKENS = (1024, 2048, 4096, 8192)
DECODE_TILES = (1, 2, 4, 8, 16)
# (chunk, pair[, fold_unit]); the default (dynamic, pair on) runs first.
PREFILL_CONFIGS = (
    (64, 1, None),
    (32, 1, None),
    (128, 0, None),
    (64, 0, None),
    (256, 0, None),
    (256, 1, 512),
)


@contextlib.contextmanager
def tile_override(decode_tile=None, chunk=None, record=None):
    """Force get_tile_sizes' decode/mixed tiles; record what it returned."""
    from vllm_torchtpu.kernels.gdn.v3 import tiling

    real = tiling.get_tile_sizes

    def forced(*args, **kwargs):
        if decode_tile is not None:
            kwargs["decode_tile_size"] = decode_tile
        if chunk is not None:
            kwargs["mixed_tile_size"] = chunk
        out = real(*args, **kwargs)
        if record is not None:
            record.append(out)
        return out

    tiling.get_tile_sizes = forced
    try:
        yield
    finally:
        tiling.get_tile_sizes = real


def _flag(out):
    return (out.reshape(-1)[0].astype(jnp.float32) > 1e30).astype(jnp.float32)


def measure(fwd, ops):
    """fwd(ops, flag) -> (ops', out), the pool threaded through the carry."""
    ops = jax.tree_util.tree_map(jnp.copy, ops)
    _, out0 = jax.jit(lambda o: fwd(o, jnp.float32(0)))(ops)
    dg = digest(out0)

    def body(carry):
        o, prev = carry
        return fwd(o, _flag(prev))

    us, _ = bench(body, (ops, out0))
    return us, dg


def _weights():
    rng = np.random.default_rng(5)
    return (
        # VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR: float32 [kernel, 1, dim].
        jnp.asarray(rng.normal(size=(KS, 1, DIM)) * 0.2, jnp.float32),
        jnp.asarray(rng.normal(size=(N_V,)) * 0.5, jnp.float32),  # A_log
        jnp.asarray(rng.normal(size=(N_V,)) * 0.5 - 1, jnp.float32),  # dt_bias
    )


def _activations(tokens, seed):
    rng = np.random.default_rng(seed)

    def bf16(shape):
        return jnp.asarray(rng.normal(size=shape) * 0.5, jnp.bfloat16)

    return bf16((tokens, DIM)), bf16((tokens, N_V)), bf16((tokens, N_V))


def _pool():
    return jnp.zeros((NUM_BLOCKS * (BLOCK // KBS), KBS) + PAYLOAD + (LANES,), jnp.int8)


def _metadata(qlens, seq_lens, seed):
    """Slot table of len(qlens) entries (zero qlen = idle padded slot)."""
    rng = np.random.default_rng(seed)
    slots = len(qlens)
    live = qlens > 0
    qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    # Blocks for live slots only: a state block plus WINDOW - 1 more
    # checkpoint blocks each, then one spare (the tail-fold checkpoint).
    n_live = int(live.sum())
    blocks = rng.permutation(np.arange(1, NUM_BLOCKS))
    assert n_live * WINDOW + 1 <= blocks.size, (n_live, NUM_BLOCKS)
    state = np.zeros(slots, np.int32)
    ckpt = np.zeros((slots, WINDOW), np.int32)
    state[live] = blocks[:n_live]
    ckpt[live, 0] = blocks[:n_live]
    ckpt[live, 1:] = blocks[n_live : n_live * WINDOW].reshape(n_live, WINDOW - 1)
    slot_read_offsets = rng.integers(0, WINDOW, NUM_BLOCKS).astype(np.int32)
    return dict(
        state=jnp.asarray(state),
        qsl=jnp.asarray(qsl),
        seq_lens=jnp.asarray(np.where(live, seq_lens, 0).astype(np.int32)),
        ckpt=jnp.asarray(ckpt),
        slot_read_offsets=jnp.asarray(slot_read_offsets),
        spare_block=int(blocks[n_live * WINDOW]),
    )


def _decode_case(rows):
    n_req = rows // WINDOW
    slots = max(SLOTS, n_req)
    qlens = np.zeros(slots, np.int32)
    qlens[:n_req] = WINDOW
    md = _metadata(qlens, qlens + CONTEXT, seed=rows)
    md["distribution"] = jnp.asarray([n_req] * 3, jnp.int32)
    return slots, md


def _prefill_case(tokens):
    qlens = np.zeros(SLOTS, np.int32)
    qlens[0] = tokens
    md = _metadata(qlens, qlens + CONTEXT, seed=tokens)
    md["distribution"] = jnp.asarray([0, 1, 1], jnp.int32)
    return SLOTS, md


def _fwd(mesh, slots, md, phase):
    from vllm_torchtpu.core.mamba_tail_fold import fold_unit
    from vllm_torchtpu.layers.core import gdn_attention
    from vllm_torchtpu.models.vllm.qwen4_exp.gdn import prepare_gdn_input

    kw = dict(
        n_kq=N_KQ, n_v=N_V, d_k=D_K, d_v=D_V, kernel_size=KS,
        pool_block_tokens=BLOCK, mesh=mesh, recurrent_state_dtype=jnp.bfloat16,
        conv_state_dtype=jnp.float32, num_spec_tokens=NUM_SPEC,
        conv_weight_kmajor=True,
    )

    def fwd(ops, flag):
        (pool, qkv, b, a, conv_w, a_log, dt_bias, state, qsl, dist, seq_lens,
         *meta) = ops
        b = b + flag.astype(b.dtype)
        prepared = prepare_gdn_input(qkv, slots * WINDOW, mesh)
        args = (prepared, b, a, pool, conv_w, None, a_log, dt_bias, state, qsl,
                dist, seq_lens)
        if phase == "decode":
            new_pool, out = gdn_attention.run_jax_gdn_attention_pooled(
                *args, None, None, step_metadata=tuple(meta), **kw
            )
        else:
            slot_ro, ckpt = meta
            unit = fold_unit(BLOCK)
            tokens = qkv.shape[0]
            offsets = jnp.zeros((slots,), jnp.int32).at[0].set(tokens - unit)
            blocks = jnp.zeros((slots,), jnp.int32).at[0].set(md["spare_block"])
            new_pool, out = gdn_attention.run_jax_gdn_attention_pooled(
                *args, slot_ro, ckpt, prefill_ckpt=(offsets, blocks), **kw
            )
        return (new_pool, *ops[1:]), out

    return fwd


def _ops(phase, tokens, md, weights):
    qkv, b, a = _activations(tokens, seed=3000 + tokens)
    base = (_pool(), qkv, b, a, *weights, md["state"], md["qsl"],
            md["distribution"], md["seq_lens"])
    if phase == "decode":
        from vllm_torchtpu.layers.core import gdn_attention

        step = jax.jit(gdn_attention.gdn_step_metadata)(
            md["slot_read_offsets"], md["state"], md["qsl"], md["seq_lens"],
            md["distribution"], md["ckpt"],
        )
        return base + tuple(step)
    return base + (md["slot_read_offsets"], md["ckpt"])


def _run(emit, mesh, weights, phase, size):
    if phase == "decode":
        slots, md = _decode_case(size)
        bucket = {"phase": "decode", "rows": size, "requests": size // WINDOW}
    else:
        slots, md = _prefill_case(size)
        bucket = {"phase": "prefill", "tokens": size}
    ops = _ops(phase, size, md, weights)
    fwd = _fwd(mesh, slots, md, phase)

    def run(cfg, overrides, default=False):
        def fn():
            jax.clear_caches()
            return measure(fwd, ops)

        rec = []
        with env(**overrides), tile_override(record=rec, **cfg):
            try_config(emit, bucket, {**cfg, **overrides}, fn, default=default)
        return rec[0] if rec else None

    # Served default first: dynamic tiling, chunk pair on.
    rec = run({}, {}, default=True)
    auto_t, auto_c = rec if rec else (None, None)
    if phase == "decode":
        for t in DECODE_TILES:
            if t != auto_t and t <= max(2, size // WINDOW):
                run({"decode_tile": t}, {})
    else:
        for chunk, pair, unit in PREFILL_CONFIGS:
            if chunk == auto_c and pair == 1 and unit is None:
                continue
            overrides = {"TPU_GDN_CHUNK_PAIR": pair}
            if unit:
                overrides["VLLM_TPU_MAMBA_TAIL_FOLD_UNIT"] = unit
            run({"chunk": chunk}, overrides)


def main(emit):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("model",))
    weights = _weights()
    if TINY:
        for phase, size in (("decode", 32), ("prefill", 1024)):
            slots, md = (_decode_case if phase == "decode" else _prefill_case)(size)
            ops = _ops(phase, size, md, weights)
            print("bench_gdn tiny", phase, size, slots,
                  [getattr(o, "shape", o) for o in ops])
        return
    if "decode" in PHASES:
        for rows in DECODE_ROWS:
            _run(emit, mesh, weights, "decode", rows)
    if "prefill" in PHASES:
        for tokens in PREFILL_TOKENS:
            _run(emit, mesh, weights, "prefill", tokens)
