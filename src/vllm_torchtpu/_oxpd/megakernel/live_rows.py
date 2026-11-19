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
"""q38 X-K-LIVE: the whole step's live-row mask (`VLLM_TPU_MK_LIVE_ROWS`, default off).

A served bucket pads its rows: c16 runs a median of 2 requests (8 rows at MTP k3) in the 16-row
bucket. With the flag, `build_wholestep_call` sets `ExecSpec.live_rows`; the planner then adds one
kernel input, `live_rows` (SMEM int32 `[B]`, replicated: 1 = live row, 0 = pad row), and blocks may
skip pad rows' work. Pad rows' outputs are discarded and their GDN state writes go to the null block
(`qwen38_segment_call.gdn_blocks`), so live rows' outputs stay bit-identical.

Interface (shared by the GDN half, L1, and the MoE half, L3):
  - gate on `ctx.ex.live_rows` (never on the env: segment calls never supply the operand);
  - read `ctx.extra[LIVE_ROWS][b]` (`row_live`); do not declare the input in a block (the planner
    declares it once, `planner.extra_inputs`);
  - the mask is per row and constant over each request's `T` rows:
    `live_rows[b] = (qsl[b // T + 1] - qsl[b // T]) > 0`, the same `live` that `gdn_blocks` uses for
    the GDN read / write / has-initial ids (`live_rows_from_qsl`). Consumers must use the mask, not a
    count (pads are a suffix in practice; do not rely on it).
"""

from __future__ import annotations

from typing import Any

LIVE_ROWS = "live_rows"
ENV = "VLLM_TPU_MK_LIVE_ROWS"


def live_rows_on() -> bool:
    """`VLLM_TPU_MK_LIVE_ROWS` (registered in envs, so it keys the TPU compile cache)."""
    from vllm_torchtpu import envs

    return bool(getattr(envs, ENV))


def live_rows_input(ex: Any) -> Any:
    """The kernel input: SMEM int32 `[B]`, replicated."""
    from .ops import I32
    from .spec import ExtraInput

    return ExtraInput(LIVE_ROWS, "smem", (ex.B,), I32)


def live_rows_from_qsl(qsl: Any, NS: int, T: int) -> Any:
    """int32 `[NS * T]`: 1 for the rows of requests with a query (`q_len > 0`), else 0.

    `qsl` is the runner's query start locations (any length; fitted to `NS + 1` like `gdn_blocks`,
    whose padded entries repeat the last valid location, so pad requests have `q_len == 0`)."""
    import jax.numpy as jnp

    q = qsl[: NS + 1]  # `qwen38_segment_call._fit`: first NS + 1 entries, zero-padded
    if q.shape[0] < NS + 1:
        q = jnp.pad(q, (0, NS + 1 - q.shape[0]))
    live = (q[1:] - q[:-1]) > 0
    return jnp.repeat(live.astype(jnp.int32), T)


def row_live(ctx: Any, b: Any) -> Any:
    """Traced bool: row `b` (static or traced) is live."""
    return ctx.extra[LIVE_ROWS][b] != 0


def req_live(ctx: Any, s: Any) -> Any:
    """Traced bool: request `s` (rows `s T .. s T + T - 1`) is live."""
    return ctx.extra[LIVE_ROWS][s * ctx.ex.T] != 0


# q38 X-K-LIVE GDN sub-mode (L1 GDN half; appended).
def live_rows_mode() -> int:
    """0 = off; 1 = VLLM_TPU_MK_LIVE_ROWS (the mask; GDN skips pad requests' pool-gather DMAs only); 2 = also
    VLLM_TPU_MK_LIVE_GDN_COMPUTE (GDN skips pad requests' recurrence and state copies too). Both envs are hashed."""
    from vllm_torchtpu import envs

    if not live_rows_on():
        return 0
    return 2 if bool(getattr(envs, "VLLM_TPU_MK_LIVE_GDN_COMPUTE")) else 1


def gdn_compute_skip(ex: Any) -> bool:
    """Mode 2: GDN pad requests skip the recurrence and their state copies (`ExecSpec.live_rows >= 2`)."""
    return int(getattr(ex, "live_rows", 0) or 0) >= 2
