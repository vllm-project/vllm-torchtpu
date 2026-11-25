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
"""q38 L6-STEADY: the steady-step fast path of the worker's input prep.

Lever ``steady`` (``VLLM_TPU_L6_STEADY_STEP``), read once per worker step
through ``core/l6_ab.py`` (0 / 1 / ab:<classes>; never part of the compile
key; no new compiled program). With it on, a step whose batch did not change
since the last full step reuses that step's work instead of redoing it:

* **Attention metadata** (``prep_attention_metadata``). The last full walk's
  staged block tables, unified-pool mamba state / checkpoint indices and its
  per-group ``AttentionMetadata`` are reused when (a) no ``BlockTable`` was
  mutated since (``attention_metadata._BLOCK_TABLE_VERSION``, bumped by every
  row write), (b) the chunk geometry, layout descriptor and request order are
  the same, and (c) every running request's mamba state column
  ``(seq_len - 1) // block_size`` is provably the one the cached products were
  derived from. The device ``seq_lens`` of a request is its host length minus
  the previous step's rejected drafts, so it lies in ``[hi - K, hi]`` with
  ``hi = num_computed + num_scheduled`` and K the speculative token count;
  (c) holds when the column is the same at both ends of that interval and
  equal to the column recorded with the cache (the products cache key of
  Jordan Dotzel's 3fecdd951 on the Qwen3.5 runner host path, ported). The
  reuse skips the block-table staging, the walk program (one compiled
  dispatch with a tensor output per mamba group: 74 at Qwen3.8 TP8) and the
  per-group metadata construction; fresh ``AttentionMetadata`` objects are
  cloned from per-group templates with this step's per-step tensors
  (positions, seq_lens, query_start_loc, request distributions, read
  offsets), so no object is shared across steps.
* **Reorder + chunk count** (``reorder_count``): ``_reorder_batch_for_rpa``
  and ``_count_input_chunks`` results are reused when no request joined, left,
  resumed or was preempted, the batch order equals the order the last full
  reorder left, and the per-request scheduled token counts and the set of
  requests with drafts are unchanged (a pure decode / verify batch only). The
  DP step coordination still runs (it may be a collective).
* **Upstream reorder** (inside ``update_states``): vLLM's
  ``reorder_batch_to_split_decodes_and_prefills`` is a no-op when every
  request is in its decode region (scheduled <= threshold, has context, done
  prefilling); then it is skipped (IX a46d91d20 drops it outright; here only
  when provably a no-op).
* **Adapter block tables** (``adapter_prepare_inputs``): the Qwen3.8 adapter's
  two ``reshape(rows, -1)`` views are reused while their source tensors are
  the same objects (models/vllm/qwen4_exp/inputs.py).

``VLLM_TPU_L6_STEADY_STEP_CHECK=N``: every N fast steps (per part) the full
path runs as well and the results are compared on the host (block-table
bytes, mamba products, every metadata field, the reorder results, the
upstream order). A mismatch logs one ERROR, counts ``steady.mismatch`` and
turns the fast path off for the process; the full result is the one used.

Engagement: counts on the L6AB window line (``core/l6_ab.py``), as
``steady.<key>=n``: ``fast`` / ``full`` / ``check`` (attention metadata, one
per target chunk), ``ro_fast`` / ``ro_full`` (reorder + count), ``up_skip`` /
``up_run`` (upstream reorder), ``tbl_reuse`` (adapter views), ``mismatch``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.layers.core import attention_metadata as _am
from vllm_torchtpu.runner import l6_setupfree as _l6_setupfree
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LEVER = "steady"

# Per-step fields of an FMB-built AttentionMetadata (runner/tpu_runner.py
# _build_attention_metadata_staged): every group carries the first four, a
# mamba group the last two as well. Everything else is fixed while the batch,
# its block tables and its mamba state columns are.
_STEP_ALL = ("input_positions", "seq_lens", "query_start_loc", "request_distribution")
_STEP_MAMBA = _STEP_ALL + ("mamba_slot_read_offsets", "mamba_request_distribution")
_FIELDS = (
    "input_positions",
    "block_tables",
    "seq_lens",
    "query_start_loc",
    "request_distribution",
    "mamba_state_indices",
    "mamba_slot_read_offsets",
    "mamba_ckpt_indices",
    "mamba_request_distribution",
    "sequence_layout_kind",
    "sequence_layout_protocol",
    "sequence_layout_version",
)

# VLLM_TPU_L6_STEADY_STEP_CHECK, resolved on first use (tests reset it).
_CHECK: list[int | None] = [None]


def check_every() -> int:
    v = _CHECK[0]
    if v is None:
        raw = os.environ.get("VLLM_TPU_L6_STEADY_STEP_CHECK", "").strip()
        v = int(raw) if raw else 0
        if v < 0:
            raise ValueError(f"VLLM_TPU_L6_STEADY_STEP_CHECK must be >= 0, got {v}")
        _CHECK[0] = v
    # VLLM_TPU_L6_CHECK_STEPS bounds every L6 CHECK mode to the first N worker steps.
    return v if _l6_ab.check_allowed() else 0


def on(runner: Any) -> bool:
    """The lever's state for this step (False once a CHECK found a mismatch)."""
    return _l6_ab.on(LEVER) and not runner.__dict__.get("_l6_steady_off", False)


def drop(runner: Any) -> None:
    """Forget every cached step (the lever is off this step)."""
    d = runner.__dict__
    d.pop("_l6_steady_am", None)
    d.pop("_l6_steady_ro", None)
    d.pop("_l6_sf", None)  # q38 L6 lever setupfree's kept sets (runner/l6_setupfree.py)


def _disable(runner: Any, what: str, detail: str) -> None:
    runner._l6_steady_off = True
    drop(runner)
    _l6_ab.note(LEVER, "mismatch")
    logger.error(
        "q38 L6 STEADY: CHECK mismatch in %s (%s); the steady fast path is off "
        "for the rest of this process",
        what,
        detail,
    )


def _check_due(cache: Any) -> bool:
    n = check_every()
    if n <= 0:
        return False
    cache.since_check += 1
    if cache.since_check >= n:
        cache.since_check = 0
        return True
    return False


# --------------------------------------------------------------------------
# Attention metadata


class _AmCache:
    __slots__ = (
        "version",
        "attn_groups",
        "input_batch",
        "plan",
        "key",
        "qcls",
        "q",
        "groups",
        "name_idx",
        "tables",
        "products",
        "states",
        "since_check",
    )


def seq_lens_bound(
    runner: Any, start_index: int, num_reqs: int, num_scheduled_tokens_per_req: Any
) -> tuple[np.ndarray, int]:
    """(hi, K): this chunk's host sequence lengths ``num_computed +
    num_scheduled`` (the device seq_lens before the async rejected-draft
    subtraction) and the largest subtraction any request can see (the
    speculative token count; 0 without speculative decoding)."""
    k = runner.__dict__.get("_l6_steady_k")
    if k is None:
        spec = getattr(runner, "speculative_config", None)
        k = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0
        runner._l6_steady_k = k
    nc = runner.input_batch.num_computed_tokens_cpu[start_index : start_index + num_reqs]
    hi = nc.astype(np.int64) + np.asarray(num_scheduled_tokens_per_req, dtype=np.int64)
    return hi, k


def _key(runner: Any, ctx: Any, num_reqs_padded: int) -> tuple:
    d = ctx.sequence_layout_descriptor
    s, n = ctx.start_index, ctx.num_reqs
    return (
        s,
        n,
        num_reqs_padded,
        ctx.use_max_model_len,
        ctx.mamba_ckpt_window,
        runner._unified_kv_layout,
        ctx.unified_mamba_state_indices is None,
        d.kind.value,
        d.protocol,
        d.version,
        tuple(runner.input_batch.req_ids[s : s + n]),
    )


def _q_classes(runner: Any, ctx: Any) -> tuple:
    """The distinct (target_num_blocks, block_size) of the unified-pool mamba
    groups whose state indices the walk derives (empty without them)."""
    if ctx.mamba_state_indices is not None or not runner._unified_kv_layout:
        return ()
    gkey = (runner.attn_groups, ctx.use_max_model_len, ctx.mamba_ckpt_window)
    cached = runner.__dict__.get("_l6_steady_qcls")
    if cached is not None and cached[0] == gkey[1:] and cached[1] is gkey[0]:
        return cached[2]
    seen: set[int] = set()
    classes: set[tuple[int, int]] = set()
    for group_list in runner.attn_groups:
        for group in group_list:
            b = group.metadata_builders[0]
            if b.kv_cache_group_id in seen:
                continue
            seen.add(b.kv_cache_group_id)
            if b.is_mamba_group:
                _, t = b._block_table_geometry(ctx)
                classes.add((int(t), int(b.target_block_size)))
    out = tuple(sorted(classes))
    runner._l6_steady_qcls = (gkey[1:], gkey[0], out)
    return out


def _q(ctx: Any, qcls: tuple) -> tuple | None:
    """Each class's per-request state column, or None when some request's
    column is not fixed over its possible device lengths [hi - K, hi]."""
    if not qcls:
        return ()
    hi1 = ctx.steady_seq_lens_hi - 1
    lo = hi1 - ctx.steady_max_rejected
    if lo.size and int(lo.min()) < 0:
        return None
    out = []
    for t, bs in qcls:
        a = np.minimum(lo // bs, t - 1)
        b = np.minimum(hi1 // bs, t - 1)
        if not np.array_equal(a, b):
            return None
        out.append(b.tobytes())
    return tuple(out)


def _clone(cache: _AmCache, runner: Any, ctx: Any) -> dict:
    step_all = {
        "input_positions": runner.position_ids,
        "seq_lens": ctx.seq_lens,
        "query_start_loc": ctx.query_start_loc,
        "request_distribution": ctx.request_distribution,
    }
    step_m = dict(
        step_all,
        mamba_slot_read_offsets=ctx.mamba_slot_read_offsets,
        mamba_request_distribution=ctx.mamba_request_distribution,
    )
    cls = _am.AttentionMetadata
    new = object.__new__
    objs = []
    for tmpl, is_mamba in cache.groups:
        o = new(cls)
        d = o.__dict__
        d.update(tmpl)
        d.update(step_m if is_mamba else step_all)
        objs.append(o)
    return {name: objs[i] for name, i in cache.name_idx}


def am_try(runner: Any, ctx: Any, num_reqs_padded: int):
    """``("fast", per_layer)`` with ``ctx`` populated as the full walk would
    (staged tables, mamba products, unified state list); ``("check",
    snapshot)`` when this fast step is a CHECK step (``ctx`` untouched; the
    caller runs the full path and hands the snapshot to ``am_record``); None
    when the step needs the full path."""
    c = runner.__dict__.get("_l6_steady_am")
    if (
        c is None
        or c.q is None
        or _am._BLOCK_TABLE_VERSION[0] != c.version
        or ctx.position_ids_override is not None
        or ctx.staged_block_tables is not None
        or ctx.mamba_state_indices is not None
        or runner.attn_groups is not c.attn_groups
        or runner.input_batch is not c.input_batch
        or runner.__dict__.get("_fast_metadata_plan") is not c.plan
        or _key(runner, ctx, num_reqs_padded) != c.key
        or _q(ctx, c.qcls) != c.q
    ):
        return None
    if _l6_setupfree.on(runner):
        # q38 L6 lever setupfree (runner/l6_setupfree.py): the same values in one
        # of two kept object sets, rewritten in place when nothing else holds it.
        per_layer = _l6_setupfree.clone(c, runner, ctx, _clone)
    else:
        per_layer = _clone(c, runner, ctx)
    if _check_due(c):
        return ("check", (per_layer, c))
    ctx.staged_block_tables = dict(c.tables)
    ctx.staged_mamba_products = None if c.products is None else dict(c.products)
    if c.states and ctx.unified_mamba_state_indices is not None:
        ctx.unified_mamba_state_indices.extend(c.states)
    _l6_ab.note(LEVER, "fast")
    return ("fast", per_layer)


def _pairs_differ(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> str | None:
    """Compare tensor pairs on the host with one copy per dtype and side."""
    by_dtype: dict[Any, tuple[list, list]] = {}
    for x, y in pairs:
        if x is y:
            continue
        if x.dtype != y.dtype or tuple(x.shape) != tuple(y.shape):
            return f"dtype/shape {x.dtype}{tuple(x.shape)} vs {y.dtype}{tuple(y.shape)}"
        a, b = by_dtype.setdefault(x.dtype, ([], []))
        a.append(x.reshape(-1))
        b.append(y.reshape(-1))
    for dtype, (a, b) in by_dtype.items():
        if not torch.equal(torch.cat(a).cpu(), torch.cat(b).cpu()):
            return f"values ({dtype})"
    return None


def _compare(snapshot: Any, full: dict, ctx: Any) -> str | None:
    fast, c = snapshot
    if fast.keys() != full.keys():
        return "layer names"
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    seen: set[tuple[int, int]] = set()
    for name, b in full.items():
        a = fast[name]
        for f in _FIELDS:
            x, y = getattr(a, f, None), getattr(b, f, None)
            if (x is None) != (y is None):
                return f"{name}.{f}: None on one side"
            if x is None:
                continue
            if isinstance(x, torch.Tensor):
                k = (id(x), id(y))
                if k not in seen:
                    seen.add(k)
                    pairs.append((x, y))
            elif x != y:
                return f"{name}.{f}: {x!r} != {y!r}"
    staged = ctx.staged_block_tables or {}
    if staged.keys() != c.tables.keys():
        return "staged table groups"
    pairs += [(c.tables[g], staged[g]) for g in staged]
    prods = ctx.staged_mamba_products
    if (prods is None) != (c.products is None):
        return "mamba products: None on one side"
    if prods is not None:
        if prods.keys() != c.products.keys():
            return "mamba product groups"
        for g, (st, ck) in prods.items():
            cst, cck = c.products[g]
            if (ck is None) != (cck is None):
                return f"group {g} ckpt: None on one side"
            pairs.append((cst, st))
            if ck is not None:
                pairs.append((cck, ck))
    states = ctx.unified_mamba_state_indices
    if (states is None) != (c.states is None) or (
        states is not None and len(states) != len(c.states)
    ):
        return "unified state list"
    if states is not None:
        pairs += list(zip(c.states, states))
    return _pairs_differ(pairs)


def am_record(
    runner: Any, ctx: Any, num_reqs_padded: int, per_layer: dict | None, attempt: Any
) -> None:
    """After a full walk of a steady-eligible chunk: compare a CHECK step's
    snapshot, then cache this walk (FMB-built metadata only) for the next steps."""
    if attempt is not None and attempt[0] == "check":
        _l6_ab.note(LEVER, "check")
        why = (
            "the full walk built no FMB metadata"
            if per_layer is None
            else _compare(attempt[1], per_layer, ctx)
        )
        if why is not None:
            _disable(runner, "attention metadata", why)
            return
    else:
        _l6_ab.note(LEVER, "full")
    plan = runner.__dict__.get("_fast_metadata_plan")
    if (
        per_layer is None
        or ctx.position_ids_override is not None
        or ctx.mamba_state_indices is not None
        or ctx.staged_block_tables is None
        or plan is None
        or plan[0] is not runner.attn_groups
    ):
        runner.__dict__.pop("_l6_steady_am", None)
        return
    groups = []
    name_idx = []
    for gid, is_mamba, layer_names in plan[1]:
        if not layer_names:
            continue
        tmpl = dict(per_layer[layer_names[0]].__dict__)
        for f in _STEP_MAMBA if is_mamba else _STEP_ALL:
            tmpl.pop(f, None)
        i = len(groups)
        groups.append((tmpl, is_mamba))
        name_idx.extend((name, i) for name in layer_names)
    prev = runner.__dict__.get("_l6_steady_am")
    c = _AmCache()
    c.version = _am._BLOCK_TABLE_VERSION[0]
    c.attn_groups = runner.attn_groups
    c.input_batch = runner.input_batch
    c.plan = plan
    c.key = _key(runner, ctx, num_reqs_padded)
    c.qcls = _q_classes(runner, ctx)
    c.q = _q(ctx, c.qcls)
    c.groups = groups
    c.name_idx = name_idx
    c.tables = dict(ctx.staged_block_tables)
    products = ctx.staged_mamba_products
    c.products = None if products is None else dict(products)
    states = ctx.unified_mamba_state_indices
    c.states = None if states is None else list(states)
    c.since_check = prev.since_check if prev is not None else 0
    runner._l6_steady_am = c


# --------------------------------------------------------------------------
# Reorder + chunk count


class _RoCache:
    # fast_seq: the l6_ab step whose reorder_try returned this cache's result (its
    # same-step ro_fast verdict, read by levers clean / pshort; runner/l6_carry_lean.py).
    __slots__ = ("batch", "req_ids", "sched", "spec", "cfg", "result", "since_check", "fast_seq")


def _ro_cfg(runner: Any) -> tuple:
    return (
        runner.most_model_len,
        runner.num_reqs_most_model_len,
        runner.num_reqs_max_model_len,
    )


def reorder_try(runner: Any, so: Any):
    """The cached ``(num_decode, num_windowed, num_chunks, max_chunk_reqs)``,
    ``("check", cached)`` on a CHECK step, or None."""
    c = runner.__dict__.get("_l6_steady_ro")
    if c is None:
        return None
    cached = so.scheduled_cached_reqs
    if (
        so.scheduled_new_reqs
        or so.finished_req_ids
        or cached.resumed_req_ids
        or so.preempted_req_ids
        or runner.input_batch is not c.batch
        or runner.input_batch.req_ids != c.req_ids
        or so.num_scheduled_tokens != c.sched
        or so.scheduled_spec_decode_tokens.keys() != c.spec
        or _ro_cfg(runner) != c.cfg
    ):
        return None
    if _check_due(c):
        return ("check", c.result)
    _l6_ab.note(LEVER, "ro_fast")
    c.fast_seq = _l6_ab.step_seq()
    return c.result


def reorder_and_count(runner: Any, so: Any) -> tuple[int, int, int, int]:
    """``runner._reorder_batch_for_rpa`` then ``runner._count_input_chunks`` as
    (num_decode, num_windowed, num_chunks, max_chunk_reqs), or the last full
    call's results on a step whose batch, order and schedule it left (the
    reorder is then a no-op). Called by execute_model with the lever on."""
    attempt = reorder_try(runner, so)
    if attempt is not None and attempt[0] != "check":
        return attempt
    num_decode, num_windowed = runner._reorder_batch_for_rpa(so)
    num_chunks, max_chunk_reqs = runner._count_input_chunks(so)
    result = (num_decode, num_windowed, num_chunks, max_chunk_reqs)
    reorder_record(runner, so, result, attempt)
    return result


def reorder_record(runner: Any, so: Any, result: tuple, attempt: Any = None) -> None:
    if isinstance(attempt, tuple) and attempt and attempt[0] == "check":
        _l6_ab.note(LEVER, "check")
        if tuple(attempt[1]) != tuple(result):
            _disable(runner, "reorder / chunk count", f"{attempt[1]} != {result}")
            return
    else:
        _l6_ab.note(LEVER, "ro_full")
    batch = runner.input_batch
    if result[1] != batch.num_reqs:
        # A prefill / mixed segment: its order may depend on more than the key.
        runner.__dict__.pop("_l6_steady_ro", None)
        return
    prev = runner.__dict__.get("_l6_steady_ro")
    c = _RoCache()
    c.batch = batch
    c.req_ids = list(batch.req_ids)
    c.sched = dict(so.num_scheduled_tokens)
    c.spec = frozenset(so.scheduled_spec_decode_tokens)
    c.cfg = _ro_cfg(runner)
    c.result = tuple(result)
    c.since_check = prev.since_check if prev is not None else 0
    c.fast_seq = -1
    runner._l6_steady_ro = c


# --------------------------------------------------------------------------
# Upstream (vLLM) batch reorder inside update_states


def upstream_reorder_noop(runner: Any, so: Any) -> bool:
    """True when vLLM's ``reorder_batch_to_split_decodes_and_prefills`` would
    swap nothing: every request is in its decode region (scheduled tokens <=
    the threshold, num_computed > 0, num_computed >= num_prompt)."""
    threshold = getattr(runner, "reorder_batch_threshold", None)
    if threshold is None:
        return False
    batch = runner.input_batch
    req_ids = batch.req_ids
    n = len(req_ids)
    if n == 0:
        return False
    sched = so.num_scheduled_tokens
    if max(sched[r] for r in req_ids) > threshold:
        return False
    nc = batch.num_computed_tokens_cpu[:n]
    if not (nc > 0).all():
        return False
    return bool((nc >= batch.num_prompt_tokens[:n]).all())


def upstream_verdict(runner: Any, so: Any) -> str | None:
    """"skip" | "check" (run it and require no swap) | None (run it)."""
    if not upstream_reorder_noop(runner, so):
        _l6_ab.note(LEVER, "up_run")
        return None
    c = runner.__dict__.get("_l6_steady_up")
    if c is None:
        c = runner._l6_steady_up = _UpCounter()
    if _check_due(c):
        return "check"
    _l6_ab.note(LEVER, "up_skip")
    return "skip"


class _UpCounter:
    __slots__ = ("since_check",)

    def __init__(self) -> None:
        self.since_check = 0


def upstream_checked(runner: Any, before: list) -> None:
    _l6_ab.note(LEVER, "check")
    after = list(runner.input_batch.req_ids)
    if after != before:
        _disable(runner, "upstream reorder", "it swapped a batch the predicate called a no-op")
