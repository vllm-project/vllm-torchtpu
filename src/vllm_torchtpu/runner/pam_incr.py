# SPDX-License-Identifier: Apache-2.0
"""Row-incremental block-table pack for the target walk (VLLM_TPU_PAM_INCR; q38 SV-PAM).

On a stage-key miss, ``stage_block_table_uploads`` packs every KV cache group's rows into one flat host tensor
(``_pack_and_upload``, layers/core/attention_metadata.py): it zero-fills the whole tensor, copies each group's first
``num_reqs`` rows (59 groups for Qwen4Exp), compares the result with the previous pack, clones it and uploads it. At B1
decode rows that pack covers 2,635,008 int32 (10.5 MB) per walk and misses the key on 55 % of walks, because any block
crossing, admission or finish bumps the global BlockTable version (q38 served probe pair b1 j-4e4367f2 / b2 j-c39952b1:
``pam_pack`` + ``pam_stage`` 3.722 ms a step on rank 0, the group builds 0.209 ms).

With the key on, a fenced, non-dummy walk packs into one of two dedicated host buffers instead (never the shared
``_block_table_stage_cpu`` scratch) and writes only the rows that changed since that buffer was last packed:

- Every BlockTable mutation marks its rows dirty on the table: ``append_row`` (non-empty blocks) / ``add_row`` /
  ``clear_row`` mark the row, ``move_row`` and ``swap_row`` mark both rows (vLLM 0.31's ``move_row`` also zeroes the
  vacated source row), ``clear`` marks every row.
- Each pack drains the walk's tables' dirty rows into both buffers' pending sets, then brings the buffer it writes up to
  date from the BlockTable source: the pending rows below ``num_reqs``, the rows a grown ``num_reqs`` adds, and zeros for
  the rows a shrunk ``num_reqs`` drops. The buffers alternate, so the buffer an upload may still be reading is never the
  one being written.
- A full pack (the stock zero fill and copies) runs on the first use of a buffer, on any geometry change (tables,
  their host tensors, target_num_blocks, num_reqs_padded, start_index), after ``clear()``, and on the first pack after
  an A/B OFF window. Dummy runs and walks issued while the input staging fence is pending (the drafter's) take the stock
  pack unchanged.
- The whole flat buffer is uploaded, so the H2D bytes are those of a stock miss. The full zero fill, ``torch.equal``
  and ``clone`` are skipped; a key miss always uploads (the stock equal-content reuse measured 0.0000 of walks).

The staged bytes are byte-equal to the stock pack at every walk (tests/runner/test_q38_svpam.py). Host-only: no
program, shape or input changes, and layers/core is not edited (the source part of the compile key is unchanged).

``VLLM_TPU_PAM_INCR=1`` turns it on (default off: nothing is patched). ``VLLM_TPU_PAM_INCR_AB_MS=<P>`` (0 = off):
in-process A/B in wall-clock windows, the incremental pack in even ``time_ns // P ms`` windows and the stock pack in odd
ones; P must divide 86,400,000 with an even quotient. Both keys are host-only and excluded from the compile key.
Installed from ``TPUModelRunner.__init__``; refused loudly (RuntimeError) if the stock pack or vLLM's BlockTable
mutators differ from the sources this copy was written against.

Runtime stats line, from the thread that packs, every LOG_EVERY packs through the patched function:
``PAM_INCR incr=.. full=.. fallback=.. ab_off=.. dirty_rows_mean=.. pack_ms_mean=.. full_ms_mean=.. off_ms_mean=.. rank=..``
(incr / full: incremental / full packs into the dedicated buffers; fallback: dummy or unfenced walks, stock pack;
ab_off: stock packs in A/B OFF windows; dirty_rows_mean: rows copied per incremental pack; pack_ms_mean /
full_ms_mean / off_ms_mean: host ms per incremental / full / OFF-window pack including the H2D enqueue).
"""

import functools
import hashlib
import inspect
import time

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DAY_MS = 86_400_000
# About 18 packs a second at B1 decode (55 % of the walks miss the key), so about one line a minute per rank.
LOG_EVERY = 1000
_PATCH_FLAG = "_tpu_pam_incr_patch"
_DIRTY = "_tpu_pam_dirty"
_ALL = "_tpu_pam_all"
_MUTATORS = ("append_row", "add_row", "clear_row", "move_row", "swap_row", "clear")
# sha256 of the normalized sources (blank lines dropped, lines right-stripped) this copy was written against:
# K3's stock attention_metadata._pack_and_upload (271cf69f0e; H-b.3 e2f2fe8490 attention_metadata.py:722, the same
# source as at q38-svhost 3417bb806c), and vLLM 0.31's BlockTable mutators + get_cpu_tensor.
STOCK_PACK_SHA256 = "8dd49806dc7d3ec6064fa0dcbd7d37a7e312945e289841be8333176408a60508"
BLOCK_TABLE_SHA256 = "c97bd1895ea3a81a3790170522c711c17b895501c7c33edad601fa09a3cf04ce"

ENABLED = False
AB_MS = 0
_state = {"ab_window": None}


def _load() -> None:
    global ENABLED, AB_MS
    ENABLED = envs.VLLM_TPU_PAM_INCR
    AB_MS = envs.VLLM_TPU_PAM_INCR_AB_MS
    if AB_MS and (DAY_MS % AB_MS or (DAY_MS // AB_MS) % 2):
        raise ValueError(
            f"Invalid value '{AB_MS}' for VLLM_TPU_PAM_INCR_AB_MS: must be 0 or divide {DAY_MS} with an even quotient."
        )


_load()


def enabled() -> bool:
    return bool(ENABLED)


def _now_ns() -> int:
    return time.time_ns()


def ab_on(st=None) -> bool:
    """Whether the incremental pack is on in the current A/B window (always, without AB)."""
    if AB_MS <= 0:
        return True
    k = _now_ns() // (AB_MS * 1_000_000)
    if k != _state["ab_window"]:
        _state["ab_window"] = k
        s = st["stats"] if st is not None else _new_stats()
        logger.info(
            "PAM_INCR AB window=%d on=%d incr=%d full=%d fallback=%d ab_off=%d",
            k, int(k % 2 == 0), s["incr"], s["full"], s["fallback"], s["ab_off"],
        )
    return k % 2 == 0


def source_sha256(*fns) -> str:
    h = hashlib.sha256()
    for fn in fns:
        src = inspect.getsource(inspect.unwrap(fn))
        h.update("\n".join(line.rstrip() for line in src.splitlines() if line.strip()).encode())
        h.update(b"\0")
    return h.hexdigest()


def _h2d(flat: torch.Tensor, device) -> torch.Tensor:
    return flat.to(device, non_blocking=True)


def _alloc(n: int) -> torch.Tensor:
    from vllm.utils.torch_utils import PIN_MEMORY

    return torch.zeros((n,), dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY)


def _new_stats() -> dict:
    return {"calls": 0, "incr": 0, "full": 0, "fallback": 0, "ab_off": 0, "rows": 0, "incr_ns": 0, "full_ns": 0,
            "off_ns": 0}


def _new_buf() -> dict:
    # geom: (tables, their host tensors, target_num_blocks, num_reqs_padded, start_index, total) the buffer holds, or
    # None; n: the num_reqs it was packed with; pending: source rows changed since, or None (= every row).
    return {"t": None, "old": None, "flat": None, "geom": None, "n": 0, "pending": None, "views": ()}


def _runner_state(runner) -> dict:
    st = runner.__dict__.get("_pam_incr")
    if st is None:
        st = {"bufs": (_new_buf(), _new_buf()), "next": 0, "after_off": False, "stats": _new_stats()}
        runner.__dict__["_pam_incr"] = st
    return st


def _mark(bt, rows) -> None:
    d = bt.__dict__.get(_DIRTY)
    if d is None:
        d = bt.__dict__[_DIRTY] = set()
    d.update(int(r) for r in rows)


def _install_dirty_rows(BlockTable) -> None:
    """Marks the rows every BlockTable mutation can change (on the table instance)."""
    if getattr(BlockTable, _PATCH_FLAG, False):
        return
    orig = {name: getattr(BlockTable, name) for name in _MUTATORS}

    @functools.wraps(orig["append_row"])
    def append_row(self, block_ids, row_idx):
        if len(block_ids):
            _mark(self, (row_idx,))
        return orig["append_row"](self, block_ids, row_idx)

    @functools.wraps(orig["add_row"])
    def add_row(self, block_ids, row_idx):
        _mark(self, (row_idx,))
        return orig["add_row"](self, block_ids, row_idx)

    @functools.wraps(orig["clear_row"])
    def clear_row(self, row_idx):
        _mark(self, (row_idx,))
        return orig["clear_row"](self, row_idx)

    @functools.wraps(orig["move_row"])
    def move_row(self, src, tgt):
        _mark(self, (src, tgt))  # vLLM 0.31 also zeroes the vacated source row
        return orig["move_row"](self, src, tgt)

    @functools.wraps(orig["swap_row"])
    def swap_row(self, src, tgt):
        _mark(self, (src, tgt))
        return orig["swap_row"](self, src, tgt)

    @functools.wraps(orig["clear"])
    def clear(self):
        self.__dict__[_ALL] = True
        return orig["clear"](self)

    for name, fn in zip(_MUTATORS, (append_row, add_row, clear_row, move_row, swap_row, clear)):
        setattr(BlockTable, name, fn)
    setattr(BlockTable, _PATCH_FLAG, True)


def _same(a, b) -> bool:
    return len(a) == len(b) and all(x is y for x, y in zip(a, b))


def _full_pack(b, entries, srcs, n, start, npad, total, geom) -> torch.Tensor:
    """The stock pack into buffer b (zero fill, then each group's first n rows); records its row views."""
    t = b["t"]
    if t is None or t.numel() < total:
        b["old"] = t  # kept one more turn: an upload may still read it
        t = b["t"] = _alloc(total)
    flat = t[:total]
    flat.zero_()
    flat_np = flat.numpy()
    views = []
    offset = 0
    for (_, _, tnb), src in zip(entries, srcs):
        rows = flat[offset : offset + npad * tnb].view(npad, tnb)
        rows[:n] = src[start : start + n, :tnb]
        views.append((flat_np[offset : offset + npad * tnb].reshape(npad, tnb), src.numpy()[start:, :tnb]))
        offset += npad * tnb
    b.update(flat=flat, geom=geom, n=n, pending=set(), views=tuple(views))
    return flat


def _pack(st, ctx, entries, npad):
    """(flat, full, rows copied) for this walk, packed into the next dedicated buffer."""
    n, start = ctx.num_reqs, ctx.start_index
    bts = tuple(bt for _, bt, _ in entries)
    srcs = tuple(bt.get_cpu_tensor() for bt in bts)
    tnbs = tuple(tnb for _, _, tnb in entries)
    total = npad * sum(tnbs)
    drained: set[int] = set()
    every = False
    for bt in bts:
        d = bt.__dict__.get(_DIRTY)
        if d:
            drained |= d
            d.clear()
        if bt.__dict__.get(_ALL):
            every = True
            bt.__dict__[_ALL] = False
    bufs = st["bufs"]
    for buf in bufs:
        if every:
            buf["pending"] = None
        elif drained and buf["pending"] is not None:
            buf["pending"] |= drained
    b = bufs[st["next"]]
    st["next"] ^= 1
    g = b["geom"]
    if (
        g is None
        or b["pending"] is None
        or g[2] != tnbs
        or g[3] != npad
        or g[4] != start
        or g[5] != total
        or not _same(g[0], bts)
        or not _same(g[1], srcs)
    ):
        return _full_pack(b, entries, srcs, n, start, npad, total, (bts, srcs, tnbs, npad, start, total)), True, 0
    n_b = b["n"]
    lo = min(n, n_b)
    rows = sorted(p for p in (r - start for r in b["pending"]) if 0 <= p < lo)
    b["pending"] = set()
    idx = np.asarray(rows, dtype=np.intp) if len(rows) > 1 else None
    for dst, src in b["views"]:
        if idx is not None:
            dst[idx] = src[idx]
        elif rows:
            dst[rows[0]] = src[rows[0]]
        if n > n_b:
            dst[n_b:n] = src[n_b:n]
        elif n < n_b:
            dst[n:n_b] = 0
    b["n"] = n
    return b["flat"], False, len(rows) + max(0, n - n_b)


def _tick(runner, st) -> None:
    s = st["stats"]
    s["calls"] += 1
    if s["calls"] % LOG_EVERY == 0:
        # The runtime stats line (the model thread, which packs): see the module docstring.
        logger.info(
            "PAM_INCR incr=%d full=%d fallback=%d ab_off=%d dirty_rows_mean=%.2f pack_ms_mean=%.3f "
            "full_ms_mean=%.3f off_ms_mean=%.3f rank=%s",
            s["incr"], s["full"], s["fallback"], s["ab_off"], s["rows"] / max(1, s["incr"]),
            s["incr_ns"] / 1e6 / max(1, s["incr"]), s["full_ns"] / 1e6 / max(1, s["full"]),
            s["off_ns"] / 1e6 / max(1, s["ab_off"]), getattr(runner, "_profiler_rank", None),
        )


def make_pack_and_upload(stock):
    """``attention_metadata._pack_and_upload`` with the incremental pack on fenced, non-dummy walks."""

    def _pack_and_upload(runner, ctx, entries, num_reqs_padded, table_geom, stage_key):
        st = _runner_state(runner)
        s = st["stats"]
        if ctx.position_ids_override is not None or runner.__dict__.get("_input_staging_fence") is not None:
            # Dummy runs (zero-filled layout) and walks packing while this step's uploads may still read the host
            # staging tensors (the drafter's): the stock pack.
            s["fallback"] += 1
            out = stock(runner, ctx, entries, num_reqs_padded, table_geom, stage_key)
            _tick(runner, st)
            return out
        if not ab_on(st):
            st["after_off"] = True
            t0 = time.perf_counter_ns()
            out = stock(runner, ctx, entries, num_reqs_padded, table_geom, stage_key)
            s["off_ns"] += time.perf_counter_ns() - t0
            s["ab_off"] += 1
            _tick(runner, st)
            return out
        if st["after_off"]:
            st["after_off"] = False
            for buf in st["bufs"]:
                buf["pending"] = None
        t0 = time.perf_counter_ns()
        flat, full, nrows = _pack(st, ctx, entries, num_reqs_padded)
        staged_dev = _h2d(flat, runner.device)
        dt = time.perf_counter_ns() - t0
        if full:
            s["full"] += 1
            s["full_ns"] += dt
        else:
            s["incr"] += 1
            s["incr_ns"] += dt
            s["rows"] += nrows
        # The stock bookkeeping, without the clone: the buffer keeps these bytes until the pack after next, and every
        # pack through here replaces _block_table_stage_prev before that.
        if envs.VLLM_TPU_BLOCK_TABLE_STAGE_CHECK:
            runner._block_table_stage_check_last = (flat.clone(), staged_dev)
        if envs.VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE:
            runner._block_table_stage_prev = (table_geom, flat, staged_dev, stage_key)
        _tick(runner, st)
        return staged_dev

    setattr(_pack_and_upload, _PATCH_FLAG, True)
    _pack_and_upload.__wrapped__ = stock
    return _pack_and_upload


def patch_pam_incr() -> bool:
    """TPUModelRunner.__init__; a no-op (nothing imported, nothing patched) unless VLLM_TPU_PAM_INCR is set."""
    if not enabled():
        return False
    from vllm.v1.worker.block_table import BlockTable

    from vllm_torchtpu.layers.core import attention_metadata as am

    cur = am._pack_and_upload
    if getattr(cur, _PATCH_FLAG, False):
        return True
    sha = source_sha256(cur)
    bsha = source_sha256(*(getattr(BlockTable, name) for name in _MUTATORS), BlockTable.get_cpu_tensor)
    if sha != STOCK_PACK_SHA256 or bsha != BLOCK_TABLE_SHA256:
        raise RuntimeError(
            "VLLM_TPU_PAM_INCR refused: the stock pack (%s, want %s) or vLLM's BlockTable mutators (%s, want %s) differ "
            "from the sources the incremental pack was written against" % (sha[:12], STOCK_PACK_SHA256[:12], bsha[:12],
                                                                          BLOCK_TABLE_SHA256[:12])
        )
    _install_dirty_rows(BlockTable)
    am._pack_and_upload = make_pack_and_upload(cur)
    logger.info(
        "VLLM_TPU_PAM_INCR=1: row-incremental target-walk block-table pack in two dedicated host buffers "
        "(AB %d ms; a PAM_INCR stats line every %d packs)",
        AB_MS, LOG_EVERY,
    )
    return True
