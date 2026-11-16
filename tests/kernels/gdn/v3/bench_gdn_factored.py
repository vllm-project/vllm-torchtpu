# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Times the factored LOG-mode window body against SELECT and LOG.

Device time per BATCHED call (wrapper included) of the pooled GDN v3 decode
kernel at the served Flash geometry, verify windows of 4 tokens, 0 to 48
running, as the mean of N calls back to back inside one jitted loop. Variants:
SELECT, LOG, the factored body with its update pass as a bf16-split MXU
product (default), a plain MXU product and on the VPU, and the factored body
with parts taken out (the product with the base, the vector work, its decay
products, its key products, the update pass, the base's HBM write).

usage: python tests/kernels/gdn/v3/bench_gdn_factored.py [iters] [variant ...]
"""

import contextlib
import dataclasses
import functools
import os
import re
import sys
import time

os.environ.setdefault("TPU_ENABLE_GDN_DYNAMIC_TILING", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, memory_ref, tiling, vmem_ldst, wrapper  # noqa: E402
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput  # noqa: E402

N_KQ, N_V, D, DIM, KS = 2, 6, 128, 1280, 4
WINDOW = 4
F32 = jnp.dtype(jnp.float32)
SELECT_PLAN = config.StateSourcePlan(
    stride=1,
    conv=config.StateRegion(0, 1, 384, 16, F32, 1, 15),
    recurrent=config.StateRegion(0, 1, 0, 384, F32, 2, 768),
)
LOG_PLAN = dataclasses.replace(
    SELECT_PLAN,
    log=config.StateRegion(0, 1, 400, 14, F32, 2, 27),
    log_conv=config.StateRegion(0, 1, 414, 15, F32, 1, 15),
)
MAX_SEQS = 64
NUM_BLOCKS = 1 + MAX_SEQS * WINDOW
TOKENS = MAX_SEQS * WINDOW  # the 256-row decode bucket
RUNNING = (0, 8, 16, 24, 32, 48)


def _no_product(rows, base):
    del base
    return rows


def _no_vectors(proj, rows, keys, v, beta_row, decay_row, log_updates, masks, cfg):
    """Keeps the operands live, drops the coefficients, the key products, the
    solve and the outputs' sums."""
    del rows, decay_row, log_updates, masks
    window, n_v = cfg.chunk_size, cfg.num_v_heads
    beta = beta_row[0, :, :1][jnp.newaxis]  # [1, W, 1]
    v_new = beta * (v - proj[:, :window])
    return proj[:, window:], v_new, jnp.ones((n_v, 1, keys.shape[1]), jnp.float32)


def _no_coef(decay_col, masks):
    return masks[2] * jnp.ones((decay_col.shape[0], 1, 1), jnp.float32)


def _no_dots(rows, keys):
    return jnp.ones((rows.shape[0], rows.shape[1], keys.shape[1]), jnp.float32)


def _decay_products_select(decay_col, masks):
    """`factored_decay_products` with its masks built per call at the full
    operand shape and applied by select (same values)."""
    valid = masks[2]
    n_v, n_upd = decay_col.shape[0], decay_col.shape[1]
    num_rows, num_cols = valid.shape[1], valid.shape[2]
    window = num_rows // 2
    num_pos = n_upd - window
    shape = (n_v, num_rows, num_cols)
    r = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    j = jax.lax.broadcasted_iota(jnp.int32, shape, 2)
    p = num_pos + jnp.where(r < window, r, r - window)
    t = jnp.where(
        j < window,
        num_pos + j,
        jnp.where(j < window + num_pos, j - window, jnp.where(j == window + num_pos, -1, 1 << 20)),
    )
    coef = jnp.ones(shape, jnp.float32)
    for m in range(n_upd):
        coef = coef * jnp.where((t < m) & (m <= p), decay_col[:, m : m + 1], 1.0)
    return jnp.where(t < p + jnp.where(r < window, 0, 1), coef, 0.0)


def _update_fullrows(base, coef_row, keys, v_new, log_updates, cfg):
    """The split update over all N key rows (zero weights off the update
    columns) instead of the gathered update rows."""
    window, num_pos, n_v = cfg.chunk_size, cfg.log_positions, cfg.num_v_heads
    num_cols = keys.shape[1]
    coef_col = compute_gdn.fused_transpose_broadcast(coef_row, src_dim=2, dst_dim=1)
    pad = num_cols - window - num_pos
    updates = jnp.concat(
        [v_new, log_updates, jnp.zeros((n_v, pad, cfg.v_head_dim), jnp.float32)], axis=1
    )
    k_hi, k_lo = compute_gdn._bf16_split(keys)
    w_hi, w_lo = compute_gdn._bf16_split(coef_col * updates)
    lhs = jnp.repeat(jnp.concat([k_hi, k_hi, k_lo, k_lo], axis=1), cfg.v_per_kq_head, axis=0)
    rhs = jnp.concat([w_hi, w_lo, w_hi, w_lo], axis=1)
    base_col = window + num_pos
    return base * coef_row[:, :, base_col : base_col + 1] + jax.lax.dot(
        lhs,
        rhs,
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )


def _no_update(base, coef_row, keys, v_new, log_updates, cfg):
    del coef_row, keys, v_new, log_updates, cfg
    return base


@contextlib.contextmanager
def _patched(variant):
    """Swaps in the variant's body parts and store hooks."""
    saved_cg = {
        name: getattr(compute_gdn, name)
        for name in (
            "factored_gdn",
            "factored_base_product",
            "factored_window_vectors",
            "factored_update_pass",
            "factored_decay_products",
            "factored_key_dots",
        )
    }
    saved_store = vmem_ldst.store_state_region
    saved_tiling = tiling.get_tile_sizes
    out_cls = memory_ref.ExternalStateOutBufferedRef
    saved_copy = (out_cls.copy_out, out_cls.wait_out)
    parts = set(variant.split("-")[1:])
    if "seqorder" in parts:
        compute_gdn.factored_gdn = functools.partial(saved_cg["factored_gdn"], order="seq")
    if "phased" in parts:
        compute_gdn.factored_gdn = functools.partial(saved_cg["factored_gdn"], order="phased")
    if "pipelined" in parts:
        compute_gdn.factored_gdn = functools.partial(saved_cg["factored_gdn"], order="pipelined")
    tile = next((int(x[4:]) for x in parts if x.startswith("tile")), None)
    if tile:
        def tile_sizes(*a, **k):
            return (tile, saved_tiling(*a, **k)[1])

        tiling.get_tile_sizes = tile_sizes
    if "selcoef" in parts:
        compute_gdn.factored_decay_products = _decay_products_select
    if "fullrows" in parts:
        compute_gdn.factored_update_pass = _update_fullrows
    if "vpu" in parts:
        compute_gdn.factored_update_pass = compute_gdn.factored_update_pass_vpu
    if "bf16" in parts:
        compute_gdn.factored_update_pass = compute_gdn.factored_update_pass_bf16
    if "nocoef" in parts:
        compute_gdn.factored_decay_products = _no_coef
    if "nodots" in parts:
        compute_gdn.factored_key_dots = _no_dots
    if "noproduct" in parts:
        compute_gdn.factored_base_product = _no_product
    if "novector" in parts:
        compute_gdn.factored_window_vectors = _no_vectors
    if "noupdate" in parts:
        compute_gdn.factored_update_pass = _no_update
    if "novmemstore" in parts:
        # Only the recurrent state's VMEM store; conv and log stores stay.
        def store(slot_ref, region, values):
            if region == SELECT_PLAN.recurrent:
                return None
            return saved_store(slot_ref, region, values)

        vmem_ldst.store_state_region = store
    if "nohbmwrite" in parts:
        def copy_out(self, *a, **k):
            if self.region == SELECT_PLAN.recurrent:
                return None
            return saved_copy[0](self, *a, **k)

        def wait_out(self, *a, **k):
            if self.region == SELECT_PLAN.recurrent:
                return None
            return saved_copy[1](self, *a, **k)

        out_cls.copy_out, out_cls.wait_out = copy_out, wait_out
    jax.clear_caches()
    try:
        yield
    finally:
        for name, fn in saved_cg.items():
            setattr(compute_gdn, name, fn)
        vmem_ldst.store_state_region = saved_store
        tiling.get_tile_sizes = saved_tiling
        out_cls.copy_out, out_cls.wait_out = saved_copy
        jax.clear_caches()


def _gdn_loop(variant, iters):
    plan = SELECT_PLAN if variant == "select" else LOG_PLAN
    factored = variant.startswith("fact")

    def call(pool, inputs):
        qkv, b, a, conv_w, a_log, dt_bias, qsl, si, dist, sl, ro, ck = inputs
        return wrapper.fused_conv1d_gdn(
            Conv1DInput(qkv, jnp.bfloat16), b, a, None, None, conv_w, None,
            a_log, dt_bias, qsl, si, dist, sl, ro, ck,
            n_kq=N_KQ, n_v=N_V, d_k=D, d_v=D, kernel_size=KS,
            state_source=pool, state_plan=plan, num_spec_tokens=WINDOW - 1,
            conv_weight_kmajor=True, batched_only=True, zero_initialize_out=False,
            factored_window=factored,
        )

    def step(_, carry):
        pool, inputs = carry
        pool, out = call(pool, inputs)
        qkv = inputs[0].at[0, 0, 0].add(out[0, 0].astype(jnp.float32) * 0)
        return pool, (qkv,) + inputs[1:]

    @functools.partial(jax.jit, donate_argnums=0)
    def run(pool, inputs):
        return jax.lax.fori_loop(0, iters, step, (pool, inputs))[0]

    return run, jax.jit(call)


def _inputs(rng, num_seqs):
    qlens = np.zeros(MAX_SEQS, np.int32)
    qlens[:num_seqs] = WINDOW
    qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    groups = np.arange(1, 1 + MAX_SEQS * WINDOW, dtype=np.int32).reshape(MAX_SEQS, WINDOW)
    return (
        jnp.asarray(rng.normal(0, 1, (TOKENS, 1, DIM)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, (TOKENS, N_V)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, (TOKENS, N_V)), jnp.float32),
        jnp.asarray(rng.normal(0, 0.5, (KS, 1, DIM)), jnp.float32),
        jnp.asarray(np.log(rng.uniform(1, 16, N_V)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, N_V), jnp.bfloat16),
        jnp.asarray(qsl),
        jnp.asarray(groups[:, 0].copy()),
        jnp.asarray(np.array([num_seqs] * 3, np.int32)),
        # Every running sequence has an initial state.
        jnp.asarray(np.where(qlens > 0, 1000 + qlens, 0).astype(np.int32)),
        jnp.asarray(rng.integers(0, WINDOW, MAX_SEQS).astype(np.int32)),
        jnp.asarray(groups.copy()),
    )


def _time(run, pool, args, iters):
    pool = run(pool, *args)
    pool.block_until_ready()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        pool = run(pool, *args)
        pool.block_until_ready()
        best = min(best, time.perf_counter() - t0)
    return pool, best / iters * 1e6


def _tile(call, pool, inputs):
    text = call.lower(pool, inputs).as_text()
    m = re.search(r"fused_conv1d_gdn_batched_b(\d+)_c(\d+)", text)
    return int(m.group(1)) if m else -1


VARIANTS = (
    "fact-phased",
    "select",
    "fact-pipelined",
    "fact-phased-tile16",
    "fact-pipelined-tile16",
    "fact-phased-tile4",
    "fact",
)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    variants = sys.argv[2:] or VARIANTS
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(NUM_BLOCKS, 512, 1, 4, 256), dtype=np.uint8)
    # A finite pool: random f8 bytes, NaN patterns (0x7f / 0xff) cleared, so
    # the state and log every variant reads are finite.
    raw[(raw & 0x7F) == 0x7F] = 0
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    del raw
    for variant in variants:
        try:
            with _patched(variant):
                run, call = _gdn_loop(variant, iters)
                tile = None
                pts = []
                for num_seqs in RUNNING:
                    inputs = _inputs(rng, num_seqs)
                    if tile is None and num_seqs:
                        tile = _tile(call, pool, inputs)
                    pool, us = _time(run, pool, (inputs,), iters)
                    pts.append((num_seqs, us))
                    print(f"GDN_FACT variant={variant} seqs={num_seqs} us_per_call={us:.2f}", flush=True)
        except Exception as e:  # report and go on with the next variant
            print(f"GDN_FACT_ERROR variant={variant} {type(e).__name__}: {str(e)[:2000]}", flush=True)
            continue
        x = np.array([n for n, _ in pts if n])
        y = np.array([u for n, u in pts if n])
        slope, icpt = np.polyfit(x, y, 1)
        print(f"GDN_FACT_FIT variant={variant} tile={tile} empty_us={pts[0][1]:.2f} "
              f"intercept_us={icpt:.2f} us_per_request={slope:.3f} us_per_tile={slope * tile:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
