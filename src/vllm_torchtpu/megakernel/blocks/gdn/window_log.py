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
"""GDN window log (VLLM_TPU_GDN_WINDOW_LOG) in the megakernel's verify step.

With the window log a GDN checkpoint block 0 holds the state after window
position 0 plus the operands of draft positions 1..K, and the read offset
counts the logged positions to fold in (`kernels/gdn/v3/vmem_ldst.py`
`fold_window_log`). The verify recurrence records those operands per request
in the `logb` scratch ([NS, LOG_ROWS, d_v] f32) in the stock typed-row layout
(`kernels/gdn/v3/config.py` `log_layout_rows`):

* row `j` (draft position j + 1): the gate, lane h = exp(decay) of value head h;
* rows `K + j * kh + h`: the l2-normalized key of key head h;
* rows `K + K * kh + j * vh + h`: the update `beta * (v - k . (S * g))` of
  value head h (the stock `v_new`).

The block emits `logb` as the per-layer output `log_ckpt`; the whole-step call
writes checkpoint 0 + log + conv inputs into block 0
(`megakernel_runtime/qwen38_scatter.scatter_gdn_log_fast`) and folds a block's
log into its checkpoint before the kernel reads it
(`megakernel_runtime/qwen38_window_log.fold_gdn_window_log`).
"""

from __future__ import annotations

import os
from typing import Any

import jax.numpy as jnp

# 27 typed rows for K = 3, kh = 2, vh = 6 (3 + 6 + 18), padded to a sublane tile.
LOG_ROWS = 32


def window_log_on() -> bool:
    """VLLM_TPU_GDN_WINDOW_LOG (same spellings as `envs.env_bool`)."""
    v = os.environ.get("VLLM_TPU_GDN_WINDOW_LOG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def log_rows_used(K: int, kh: int, vh: int) -> tuple[int, int, int]:
    """`(keys, updates, rows_used)` typed-row offsets (stock `log_layout_rows`)."""
    keys = K
    updates = keys + K * kh
    return keys, updates, updates + K * vh


def write_log_row(
    ctx: Any, gp: Any, s: Any, t: int, decay: Any, k: Any, delta: Any
) -> None:
    """Record window position `t` (> 0) of request `s` in `logb` (no-op without it)."""
    if t == 0 or "logb" not in ctx.scr:
        return
    logb = ctx.scr["logb"]
    K, kh, vh = ctx.ex.T - 1, gp.kh, gp.vh
    keys, updates, used = log_rows_used(K, kh, vh)
    assert used <= LOG_ROWS, (used, LOG_ROWS)
    j = t - 1
    # Whole gate row: lanes past the value heads hold 1 (the stock kernel's
    # masked gate), not stale VMEM; the stock factored window body reads the
    # log rows whole. The first position also clears the pad rows.
    logb[s, j : j + 1, :] = jnp.ones((1, logb.shape[-1]), jnp.float32)
    logb[s, j : j + 1, 0:vh] = jnp.exp(decay).reshape(1, vh)
    if j == 0 and used < LOG_ROWS:
        logb[s, used:LOG_ROWS, :] = jnp.zeros((LOG_ROWS - used, logb.shape[-1]), jnp.float32)
    logb[s, keys + j * kh : keys + (j + 1) * kh, :] = k
    if delta is not None:  # the head-major recurrence writes them per head
        logb[s, updates + j * vh : updates + (j + 1) * vh, :] = delta


def write_log_update(ctx: Any, gp: Any, s: Any, t: int, h: int, delta_h: Any) -> None:
    """Value head `h`'s update row `[1, VD]` of window position `t` (> 0)."""
    if t == 0 or "logb" not in ctx.scr:
        return
    K = ctx.ex.T - 1
    _, updates, _ = log_rows_used(K, gp.kh, gp.vh)
    r = updates + (t - 1) * gp.vh + h
    ctx.scr["logb"][s, r : r + 1, :] = delta_h
