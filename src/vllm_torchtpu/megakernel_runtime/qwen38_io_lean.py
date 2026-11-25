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
"""VLLM_TPU_MK_IO_LEAN: which operand-prep levers of the whole-step call are on.

Every lever computes the same values as the code it replaces (integer indexing and data
movement only); it only changes how many XLA ops the call spends around the kernel.

`ple` and `pleconv` trim the per-layer-embedding (PLE) ops the megakernel programs run
before the head kernels: `ple` (megakernel programs only) feeds the n-gram hash and the
embedding mask int32 operands and trace-time hash constants instead of int64 operands,
and finds each token's request with one comparison; `pleconv` reads each conv tap's
rows of the step as a static shift instead of a gather, and the taps' history rows in one
gather (every program that runs the PLE conv op, up to
VLLM_TPU_MK_IO_LEAN_PLECONV_MAX_ROWS rows).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs

LEVERS = ("ring", "cktab", "rope", "conv", "ple", "pleconv")


def io_lean_levers(raw: str | None = None) -> frozenset[str]:
    """The levers named by VLLM_TPU_MK_IO_LEAN (`raw` overrides the env, for tests)."""
    raw = (envs.VLLM_TPU_MK_IO_LEAN if raw is None else raw) or ""
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    if parts & {"1", "all", "true"}:
        return frozenset(LEVERS)
    parts -= {"0", "false", "off"}
    unknown = parts - set(LEVERS)
    assert not unknown, f"VLLM_TPU_MK_IO_LEAN: unknown levers {sorted(unknown)}; known {LEVERS}"
    return frozenset(parts)


# Levers that stay off above a row count, and the env var holding it. cktab / conv:
# VLLM_TPU_MK_IO_LEAN_MAX_ROWS (default 48): at c32 main-KV DCP-2 b64 either one turns
# `qsa_select` +12..17 us slower on every rank (an XLA scheduling / placement effect;
# outputs identical, ~/q38f/scag/mk_iolean.md). pleconv: VLLM_TPU_MK_IO_LEAN_PLECONV_MAX_ROWS
# (default 32): the lean PLE conv program is faster at every step of <= 32 rows but up
# to 20 us slower at 48..96 rows of 32 requests (stock decode buckets; envs.py).
_ROW_CAP_ENV = {
    "cktab": "VLLM_TPU_MK_IO_LEAN_MAX_ROWS",
    "conv": "VLLM_TPU_MK_IO_LEAN_MAX_ROWS",
    "pleconv": "VLLM_TPU_MK_IO_LEAN_PLECONV_MAX_ROWS",
}
ROW_CAPPED = frozenset(_ROW_CAP_ENV)


def io_lean(lever: str, rows: int | None = None) -> bool:
    """Whether `lever` is on for a call of `rows` rows (`None`: not bucket-dependent)."""
    assert lever in LEVERS, lever
    if lever not in io_lean_levers():
        return False
    if lever in ROW_CAPPED and rows is not None:
        return rows <= getattr(envs, _ROW_CAP_ENV[lever])
    return True


_SELF_CHECK_LOGGED = False


def io_lean_line() -> str | None:
    """The startup self-check line, or None when no lever and no PLE_TRIM is on."""
    levers = io_lean_levers()
    trim = bool(envs.VLLM_TPU_MK_PLE_TRIM)
    if not levers and not trim:
        return None
    names = ",".join(lv for lv in LEVERS if lv in levers) or "none"
    return (
        f"MK_IO_LEAN levers={names} max_rows={envs.VLLM_TPU_MK_IO_LEAN_MAX_ROWS} "
        f"pleconv_max_rows={envs.VLLM_TPU_MK_IO_LEAN_PLECONV_MAX_ROWS} ple_trim={int(trim)}"
    )


def _rank0() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except Exception:  # no TP group (CPU tests, tools)
        return True


def log_io_lean_once() -> None:
    """Rank 0, once per process: which MK_IO_LEAN levers / PLE_TRIM the server runs with
    (so a served log proves the flags took). Silent when all are off. Logging only."""
    global _SELF_CHECK_LOGGED
    if _SELF_CHECK_LOGGED:
        return
    _SELF_CHECK_LOGGED = True
    line = io_lean_line()
    if line is None or not _rank0():
        return
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info(line)


def ring_columns(tables: Any, n_rows: int, per_layer: int = 3) -> list:
    """VLLM_TPU_MK_IO_LEAN=ring: every QSA layer's ring-id table (flat `[n_rows * W]`, the
    runner's `[n_rows, W]` block table) replaced by its column 0, the only one read
    (`r.reshape(n, -1)[:, 0]` in the inputs and the cache writer), as rows of one array.

    One strided read of every layer's table instead of a `[n_rows, W]` relayout or column
    slice per layer and reader (xprof j-3d2506a9: 12 x ~0.7 us in each of the two readers).
    The barrier keeps XLA from pushing the readers' slices back into the 12 parameters.
    The other two tables of each layer pass through unchanged.
    """
    tables = list(tables)
    rings = [tables[o] for o in range(0, len(tables), per_layer)]
    cols = jnp.stack([r.reshape(-1)[:: r.size // n_rows] for r in rings])
    cols = jax.lax.optimization_barrier(cols)
    for k, o in enumerate(range(0, len(tables), per_layer)):
        tables[o] = cols[k]
    return tables
