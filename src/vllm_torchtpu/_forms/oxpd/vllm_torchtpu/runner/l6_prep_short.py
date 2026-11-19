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
"""q38 L6 lever ``pshort`` (H2 PREP-SHORT): a carried steady step's short ``_prepare_inputs``.

Lever ``pshort`` (``VLLM_TPU_L6_PREP_SHORT``), read once per worker step through
``core/l6_ab.py`` (0 / 1 / ab:<classes>; never part of the compile key; no
compiled program changes). ``_prepare_inputs`` calls ``run`` right after its
staging fence; None runs the full path unchanged.

The short path engages on a single-chunk step whose STEADY reorder verdict was
``ro_fast`` against the reorder cache in effect when the CARRY state was
recorded the step before (the epoch, ``CarryState.ro``; runner/l6_carry_lean.py),
and whose CARRY check (lean with lever ``clean``, else CARRY's stock check)
accepts the recorded state with the same position map. The batch is then the
recorded one (same requests, order and scheduled counts), so the short path:

* reuses the recorded ``nsched`` / total / target, builds the ALL layout plan
  (a pure function of the total and the DP bucket) and requires the recorded
  padded total (a bucket change runs the full path);
* writes the host mirrors the full path writes, with a few vector ops:
  ``positions_np[:total]`` (``num_computed[req] + arange`` from cached index
  arrays), ``query_start_loc_np`` (the cached cumulative sum), ``seq_lens_np``
  (``fill_step_seq_lens``) and the mamba-slot and combined-distribution staging
  tensors (the megakernel driver's route check reads the latter and
  ``query_start_loc_np``); it runs the attention-schedule check;
* skips the ``input_ids_cpu`` gather (the carry program supplies every real
  input id; ``StepCtx.materialize`` writes it, and the padding zeros, before any
  upload reads it: CARRY's CHECK and its cold-shape fallback);
* takes the spec-decode metadata from the manager's cache when this step's key
  (draft counts, cumulative counts, padded requests) is the cached key, as the
  cache-hit path does, with a fresh object (``draft_token_ids`` None) whose
  ``deferred_draft_token_ids_cpu`` is None: its only reader runs for a chunk with
  nothing substituted (C tpu_runner.py execute_model), never a carried chunk;
* reuses the previous mamba-slot and combined-distribution device tensors when
  their host bytes, computed here, equal the bytes of the stored upload (so the
  full path would return the same tensor objects), the mamba slot map would
  neither release nor allocate, and reuses the recorded ``query_start_loc``
  device tensor (as the full path does on a carried spec step);
* keeps exactly one host action, CARRY's ``fused_prepare`` (the carry or the
  merged program), then the full path's tail (metadata context, attention
  metadata, substitution indices, seed copies, ``decode_carry.end``), whose
  text tests/q38_l6/test_l6_pshort.py pins against tpu_runner.py.

``VLLM_TPU_L6_PREP_SHORT_CHECK=N``: every N short-eligible steps the full path
runs instead (re-entering ``_prepare_inputs`` with the lever held off) and its
results are compared with the short path's predictions (plan, request counts,
spec metadata, mamba / distribution / query-start device tensors, host
mirrors); a mismatch logs one ERROR, counts ``pshort.mismatch`` and turns the
short path off for the process; the full result is the one used.
``VLLM_TPU_L6_CHECK_STEPS`` bounds it, as every L6 CHECK mode.

Engagement on the L6AB window line: ``pshort.short``, ``pshort.why_<reason>``
(chunk, state, stale, verdict, batch, config, bucket, spec, mamba, comb, carry),
``pshort.materialize``, ``pshort.check``, ``pshort.mismatch``.
"""

from __future__ import annotations

import itertools
import os
from typing import Any

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.layers.core.sequence_layout import AllSequenceLayoutPlanner
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import decode_carry as _dc
from vllm_torchtpu.runner import l6_carry_lean as _clean
from vllm_torchtpu.runner import l6_fskip as _l6_fskip
from vllm_torchtpu.runner import l6_steady as _l6_steady
from vllm_torchtpu.runner.speculative_decoding_manager import SpecDecodeMetadata

logger = init_logger(__name__)

LEVER = "pshort"
_MAMBA_SLOT = "l6_mamba_state_indices"
_COMB_SLOT = "l6_comb_request_distribution"

# VLLM_TPU_L6_PREP_SHORT_CHECK, resolved on first use (tests reset it).
_CHECK: list[int | None] = [None]
# runner/tpu_runner.py, imported on first use (it imports this module).
_TR: list = []


def _tr():
    if not _TR:
        from vllm_torchtpu.runner import tpu_runner

        _TR.append(tpu_runner)
    return _TR[0]


def check_every() -> int:
    v = _CHECK[0]
    if v == 0:
        return 0
    if v is None:
        raw = os.environ.get("VLLM_TPU_L6_PREP_SHORT_CHECK", "").strip()
        v = int(raw) if raw else 0
        if v < 0:
            raise ValueError(f"VLLM_TPU_L6_PREP_SHORT_CHECK must be >= 0, got {v}")
        _CHECK[0] = v
    # VLLM_TPU_L6_CHECK_STEPS bounds every L6 CHECK mode to the first N worker steps.
    return v if _l6_ab.check_allowed() else 0


def on(runner: Any) -> bool:
    """The lever's state for this step (False once a CHECK found a mismatch)."""
    return _l6_ab.on(LEVER) and not runner.__dict__.get("_l6_pshort_off", False)


def _no(why: str) -> None:
    _l6_ab.note(LEVER, "why_" + why)
    return None


class _Plan:
    """What the short path reuses, verified for one STEADY epoch (one reorder cache
    object, so one batch) and DP bucket; kept as ``runner._l6_pshort_c`` and
    re-verified, cheaply, on every later step of the same epoch."""

    __slots__ = (
        "ro", "bucket", "n", "total", "nsched", "max_ns", "use_max", "target", "padded_num_reqs",
        "layout_plan", "padded_total", "req_idx", "ar", "nd", "nw", "ndt_b", "ndt_l", "md_cache", "md_key",
        "md_tmpl", "mamba_ent", "mamba_dev", "slots", "fast_comb", "comb_ent", "comb_dev", "rd", "mrd",
        "ckey", "kobj", "kshape", "st",
    )


def _index(runner: Any, nsched: np.ndarray):
    """(request index per token, batched arange, cumulative counts) for `nsched`."""
    n = int(nsched.shape[0])
    req_idx = np.repeat(runner.arange_np[:n], nsched)
    ar = np.concatenate([runner.arange_np[:k] for k in nsched])
    return req_idx, ar, np.cumsum(nsched)


def _draft_counts(so: Any, req_ids: list, n: int) -> np.ndarray:
    """This step's per-request draft counts, as ``_prepare_inputs`` builds them."""
    spec = so.scheduled_spec_decode_tokens
    return np.array([len(spec.get(r, ())) for r in req_ids[:n]], dtype=np.int32)


def _plan(runner: Any, so: Any, start_index: int, rpa_num_decode_reqs: int,
          num_windowed_reqs: int) -> _Plan | None:
    """Every reuse decision of the short path, and the host mirrors it writes
    (the full path writes the same values, so a refusal after them is harmless).
    None (with a why_ count) when the full path must run."""
    if start_index != 0:
        return _no("chunk")
    d = runner.__dict__
    st = d.get("_l6_carry_state")
    if st is None or getattr(st, "ro", None) is None:
        return _no("state")
    step = _l6_ab.step_seq()
    if st.step != step - 1:
        return _no("stale")
    c = d.get("_l6_steady_ro")
    if c is not st.ro or c.fast_seq != step:
        return _no("verdict")
    n = len(st.req_ids)
    total = st.total
    if runner.input_batch.num_reqs != n or so.total_num_scheduled_tokens != total:
        return _no("batch")
    if (
        not _dc.on()
        or runner.lora_config is not None
        or runner.uses_mrope
        or runner._pcp_enabled
        or runner.mamba_slot_read_offsets is None
        or runner.speculative_config is None
        or not runner.scheduler_config.async_scheduling
        or not envs.VLLM_TPU_POSTFWD_FUSED
    ):
        return _no("config")
    nd = max(0, min(rpa_num_decode_reqs, n))
    nw = max(0, min(num_windowed_reqs, n))
    p = d.get("_l6_pshort_c")
    if (
        p is not None
        and p.ro is c
        and p.n == n
        and p.total == total
        and p.bucket == runner._dp_target_bucket
        and p.nd == nd
        and p.nw == nw
        and p.target == st.target
        and p.padded_total == st.padded_total
        and _cached(runner, so, p)
    ):
        p.st = st
        return p
    return _full(runner, so, st, c, n, total, nd, nw)


def _cached(runner: Any, so: Any, p: _Plan) -> bool:
    """A later step of the plan's epoch: the batch (so nsched, the plan, the counts,
    the cumulative counts in query_start_loc_np) is the plan's. Writes the two
    mirrors that move (positions, sequence lengths) and re-verifies by identity
    what another step may have replaced (the spec cache, the two stored uploads)."""
    ib = runner.input_batch
    n, total = p.n, p.total
    tr = _tr()
    nc = ib.num_computed_tokens_cpu[:n]
    np.add(nc[p.req_idx], p.ar, out=runner.positions_np[:total])
    tr.host_phases.sub("pi_tokens")
    tr.fill_step_seq_lens(runner.seq_lens_np, nc, p.nsched)
    runner._check_attention_schedule(n, p.nsched, p.nd, p.use_max)
    mgr = runner.spec_decode_manager
    if mgr._meta_cache is not p.md_cache or mgr._meta_cache_key is not p.md_key:
        return False
    spec = so.scheduled_spec_decode_tokens
    if list(map(len, map(spec.get, ib.req_ids[:n], itertools.repeat((), n)))) != p.ndt_l:
        return False
    if p.mamba_ent is not None:
        slot_map = runner._mamba_slot_by_req_id
        if (
            runner._h2d_reuse.get(_MAMBA_SLOT) is not p.mamba_ent
            or not slot_map.keys() <= runner.requests.keys()
        ):
            return False
        try:
            if list(map(slot_map.__getitem__, ib.req_ids[:n])) != p.slots:
                return False
        except KeyError:
            return False
    if p.fast_comb:
        if getattr(runner, "_cached_comb_key", None) != (p.nd, p.nw, n) or getattr(
            runner, "_cached_comb_dev", None
        ) is not p.comb_dev:
            return False
    elif runner._h2d_reuse.get(_COMB_SLOT) is not p.comb_ent:
        return False
    return True


def _full(runner: Any, so: Any, st: Any, c: Any, n: int, total: int, nd: int, nw: int):
    """The first step of an epoch (or after a cached check failed): build and verify
    everything by value, as the full path would compute it."""
    tr = _tr()
    planner = tr._get_sequence_layout_planner_for_runner(runner)
    if type(planner) is not AllSequenceLayoutPlanner:
        return _no("config")
    ib = runner.input_batch
    p = _Plan()
    p.st, p.ro, p.n, p.total, p.nd, p.nw = st, c, n, total, nd, nw
    p.bucket = runner._dp_target_bucket
    nsched = p.nsched = st.nsched
    max_ns = p.max_ns = int(nsched.max())
    most = runner.most_model_len
    p.use_max = most is None or max_ns > most
    target = p.target = runner.num_reqs_max_model_len if p.use_max else runner.num_reqs_most_model_len
    if target != st.target or n > target:
        return _no("batch")
    p.padded_num_reqs = tr._get_padded_num_reqs_with_upper_limit(n, runner.max_num_reqs)
    layout_plan = p.layout_plan = planner.prepare_real(
        runner=runner,
        scheduler_output=so,
        start_index=0,
        num_reqs=n,
        num_scheduled_tokens_per_req=nsched,
        total_num_scheduled_tokens=total,
        use_max_model_len=p.use_max,
        target_num_reqs=target,
        padded_num_reqs=p.padded_num_reqs,
    )
    p.padded_total = layout_plan.global_padded_num_tokens
    if p.padded_total != st.padded_total:
        return _no("bucket")
    # The host mirrors the full path writes before its CARRY check.
    req_idx, ar, cum = _index(runner, nsched)
    p.req_idx, p.ar = req_idx, ar
    nc = ib.num_computed_tokens_cpu[:n]
    np.add(nc[req_idx], ar, out=runner.positions_np[:total])
    tr.host_phases.sub("pi_tokens")
    qsl = runner.query_start_loc_np
    qsl[0] = 0
    qsl[1 : n + 1] = cum
    qsl[n + 1 :] = qsl[n]
    tr.fill_step_seq_lens(runner.seq_lens_np, nc, nsched)
    runner._check_attention_schedule(n, nsched, nd, p.use_max)
    # Spec-decode metadata: this step's key must be the manager's cached key.
    req_ids = ib.req_ids
    ndt = _draft_counts(so, req_ids, n)
    if not ndt.any():
        return _no("spec")
    mgr = runner.spec_decode_manager
    p.ndt_b = ndt.tobytes()
    p.ndt_l = ndt.tolist()
    if mgr._meta_cache is None or mgr._meta_cache_key != (
        p.ndt_b, qsl[1 : n + 1].tobytes(), p.padded_num_reqs
    ):
        return _no("spec")
    m = p.md_cache = mgr._meta_cache
    p.md_key = mgr._meta_cache_key
    # The cache-hit path's object, with the deferred host drafts left out (None).
    p.md_tmpl = dict(
        draft_token_ids=None,
        draft_lengths=m.draft_lengths,
        draft_lengths_cpu=m.draft_lengths_cpu,
        target_logits_indices=m.target_logits_indices,
        bonus_logits_indices=m.bonus_logits_indices,
        final_logits_indices=m.final_logits_indices,
        segment_ids=m.segment_ids,
        group_indices=m.group_indices,
        deferred_draft_token_ids_cpu=None,
    )
    # Mamba slots: no release, no allocation, and the stored upload's bytes.
    p.mamba_ent = p.mamba_dev = p.slots = None
    if runner._has_mamba_state:
        slot_map = runner._mamba_slot_by_req_id
        if not slot_map.keys() <= runner.requests.keys():
            return _no("mamba")
        try:
            slots = list(map(slot_map.__getitem__, req_ids[:n]))
        except KeyError:
            return _no("mamba")
        # The staging tensor as _build_mamba_state_indices leaves it, then its bytes.
        host = runner.mamba_state_indices_cpu
        buf = host.numpy()[:target]
        buf[:] = 0
        buf[:n] = slots
        ent = runner._h2d_reuse.get(_MAMBA_SLOT)
        if ent is None or ent[0] != (host.dtype, (target,), buf.tobytes()):
            return _no("mamba")
        p.mamba_ent, p.mamba_dev, p.slots = ent, ent[1], slots
    # Combined RPA / GDN request distribution: the staging tensor as the full path writes it
    # (the megakernel driver's route check reads it, megakernel_runtime/driver.py), then the
    # same six values as the stored upload (or the fast-substitution cache key).
    stg = runner._combined_request_distribution_cpu
    b = stg.numpy()
    b[:6] = (nd, nd, n, nw, nw, n)
    ent = runner._h2d_reuse.get(_COMB_SLOT)
    p.fast_comb = bool(getattr(runner, "_fast_token_substitution", False))
    p.comb_ent = None
    if p.fast_comb:
        if getattr(runner, "_cached_comb_key", None) != (nd, nw, n) or not hasattr(
            runner, "_cached_comb_dev"
        ):
            return _no("comb")
        p.comb_dev = runner._cached_comb_dev
    else:
        if ent is None or ent[0] != (stg.dtype, tuple(stg.shape), b.tobytes()):
            return _no("comb")
        p.comb_ent, p.comb_dev = ent, ent[1]
    p.rd = p.comb_dev[0:3]
    p.mrd = p.comb_dev[3:6]
    p.ckey = p.kobj = p.kshape = None
    runner._l6_pshort_c = p
    return p


def _carry_check(runner: Any, p: _Plan):
    """The carry state when CARRY's check (lean with lever clean, else the stock
    check) accepts it with the same map, else None. Notes nothing on a refusal:
    the full path then checks and counts."""
    st = p.st
    if _clean.on(runner):
        # The short path established every gate of the lean test but its per-step half.
        if _clean.fast_tail(runner, st, p.n, _l6_ab.step_seq()) is None:
            return None
        _l6_ab.note(_clean.LEVER, "fast")
        if _clean._CHECK[0] != 0 and _clean.check_every() > 0 and _clean.cadence(
            runner, st, (runner, 0, p.n, p.nsched, p.total, p.padded_total, p.target, p.layout_plan)
        ) is not None:
            return None
    elif _dc._check(runner, 0, p.n, p.nsched, p.total, p.padded_total, p.target, p.layout_plan)[0] is not st:
        return None
    if st.map_same is not True or st.qsl_dev is None or st.subst is None or st.subst[2] is None:
        return None
    return st


def _materialize(runner: Any, p: _Plan) -> None:
    """The full path's input-id gather and padding zeros (ALL layout), from the
    positions this step wrote."""
    ib = runner.input_batch
    tok = runner.positions_np[: p.total] + p.req_idx * ib.token_ids_cpu.shape[1]
    torch.index_select(
        ib.token_ids_cpu_tensor.flatten(), 0, torch.from_numpy(tok),
        out=runner.input_ids_cpu[: p.total],
    )
    runner.input_ids_cpu[p.total : p.padded_total] = 0
    _l6_ab.note(LEVER, "materialize")


def _md(p: _Plan) -> SpecDecodeMetadata:
    """A fresh metadata object (execute_model sets its draft_token_ids) with the
    cache-hit path's fields."""
    md = object.__new__(SpecDecodeMetadata)
    md.__dict__.update(p.md_tmpl)
    return md


def run(runner: Any, so: Any, start_index: int, num_decode_reqs: int, num_windowed_reqs: int,
        rpa_num_decode_reqs: int):
    """``_prepare_inputs``'s return tuple from the short path, or None (the full
    path runs)."""
    d = runner.__dict__
    if d.get("_l6_pshort_in_check") or not on(runner):
        return None
    p = _plan(runner, so, start_index, rpa_num_decode_reqs, num_windowed_reqs)
    if p is None:
        return None
    if _CHECK[0] != 0 and check_every() > 0:
        k = d.get("_l6_pshort_since", 0) + 1
        if k >= _CHECK[0]:
            runner._l6_pshort_since = 0
            return _checked(runner, so, start_index, num_decode_reqs, num_windowed_reqs, p)
        runner._l6_pshort_since = k
    st = _carry_check(runner, p)
    if st is None:
        return _no("carry")
    return _short(runner, so, p, st)


def _short(runner: Any, so: Any, p: _Plan, st: Any):
    tr = _tr()
    hp = tr.host_phases
    n, target, layout_plan = p.n, p.target, p.layout_plan
    ctx = _dc.StepCtx(st, layout_plan.token_slice, target, p.nsched)
    ctx.materialize = lambda: _materialize(runner, p)
    runner._last_sequence_layout_plan = layout_plan
    runner.input_ids = st.input_ids
    runner.position_ids = st.positions
    # q38 L6-V6 compose: lever fskip's carried mark, as the full path sets it on a carried chunk
    # (tpu_runner.py _prepare_inputs). On a short-eligible step the full path's region the short path
    # replaces issues no upload (every device tensor is reused), so it adds no upload mark either; a
    # later fallback upload (decode_carry.upload) counts its own.
    _l6_fskip.S.carried = True
    hp.sub("pi_layout")
    pre = runner._pre_async_results
    num_rejected, next_tokens = pre.spec_decode_num_rejected_tokens, pre.next_tokens_tpu
    pending = (num_rejected, st.seq_idx_dev, st.pos_idx_dev)
    md = _md(p)
    query_start_loc = st.qsl_dev
    ctx.qsl_dev = query_start_loc
    hp.sub("pi_spec_qsl")
    if p.mamba_dev is not None:
        _dc._note("mamba_reused")
    if not p.fast_comb:
        _dc._note("comb_reused")
    # The carry program's shape key: the previous carried call's when every argument is the
    # same object as then (seq_idx / pos_idx / input_ids / substitution indices / logits
    # indices) or has the same shape and dtype (seq_lens / positions, the carry program's
    # previous outputs; the previous results' rejected counts and token source).
    f_idx, t_idx = md.final_logits_indices, md.target_logits_indices
    sub = st.subst
    kobj = (st.seq_idx_dev, st.pos_idx_dev, st.input_ids, sub[2], sub[3], f_idx, t_idx)
    kshape = (st.seq_lens.shape, st.seq_lens.dtype, st.positions.shape, st.positions.dtype,
              num_rejected.shape, num_rejected.dtype, next_tokens.shape, next_tokens.dtype)
    ko = p.kobj
    if ko is not None and p.kshape == kshape and all(a is b for a, b in zip(ko, kobj)):
        ctx.carry_key = p.ckey
    # q38 L6-V6 compose: lever adcarry's merge context, built as the full path builds it
    # (tpu_runner.py _prepare_inputs, right before _postfwd_fused_prepare), so a carried short
    # step runs the merged carry program exactly when the full path would.
    adc = (
        tr._adapt_carry.begin(
            runner,
            start_index=0,
            num_reqs=n,
            nsched=p.nsched,
            query_start_loc=query_start_loc,
            state_indices=p.mamba_dev,
        )
        if tr._adapt_carry.on()
        else None
    )
    seq_lens = runner._postfwd_fused_prepare(
        st.seq_lens, pending, 0, n, p.nsched, md, l6=ctx, adc=adc
    )
    if ctx.carry_key is not None:
        p.ckey, p.kobj, p.kshape = ctx.carry_key, kobj, kshape
    hp.mark("prep_host_and_uploads")
    steady_hi, steady_k = None, 0
    if _l6_steady.on(runner):
        steady_hi, steady_k = _l6_steady.seq_lens_bound(runner, 0, n, p.nsched)
    runner._attn_metadata_builder_ctx = tr.AttentionMetadataBuilderContext(
        num_reqs=n,
        start_index=0,
        use_max_model_len=p.use_max,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=p.rd,
        mamba_state_indices=p.mamba_dev,
        mamba_slot_read_offsets=runner.mamba_slot_read_offsets,
        mamba_ckpt_window=runner._mamba_ckpt_window,
        mamba_request_distribution=p.mrd,
        unified_mamba_state_indices=(
            [] if runner._unified_kv_layout and runner.mamba_slot_read_offsets is not None else None
        ),
        sequence_layout_descriptor=layout_plan.descriptor,
        steady_seq_lens_hi=steady_hi,
        steady_max_rejected=steady_k,
    )
    per_layer_attn_metadata, _unused = runner._build_attention_metadata(
        num_tokens=layout_plan.local_num_tokens,
        num_reqs=target,
        max_query_len=p.max_ns,
        num_tokens_padded=layout_plan.local_padded_num_tokens,
        num_reqs_padded=target,
        slot_mappings=runner.empty_slot_mappings,
    )
    hp.mark("prep_attention_metadata")
    if ctx.subst_out is not None:
        cur_input_indices, pre_next_tokens_indices = ctx.subst_out
    else:
        cur_input_indices, pre_next_tokens_indices = (
            runner._prepare_async_token_substitution_indices(0, n, p.nsched)
        )
    hp.sub("pi_subst2")  # q38 L6-V6 compose: L6-INSTR's tail sub-marks, as the full path's tail
    runner._collect_mamba_state_seed_copies(so, 0, n)
    hp.sub("pi_cross")
    _dc.end(runner, ctx)
    _l6_ab.note(LEVER, "short")
    return (
        per_layer_attn_metadata,
        f_idx,
        p.padded_num_reqs,
        n,
        n,
        cur_input_indices,
        pre_next_tokens_indices,
        md,
        None,
        layout_plan,
    )


_MD_FIELDS = ("draft_lengths", "draft_lengths_cpu", "target_logits_indices", "bonus_logits_indices",
              "final_logits_indices", "segment_ids", "group_indices")


def _checked(runner: Any, so: Any, start_index: int, num_decode_reqs: int, num_windowed_reqs: int,
             p: _Plan):
    """CHECK: the full path runs (this lever held off for the call) and must
    match the short path's predictions on a step it carries."""
    _l6_ab.note(LEVER, "check")
    st = p.st
    pred_ids = st.input_ids
    pred_qsl = st.qsl_dev
    pos = runner.positions_np[: p.total].copy()
    seq = runner.seq_lens_np.copy()
    qsl = runner.query_start_loc_np.copy()
    runner._l6_pshort_in_check = True
    try:
        # The method itself (not an instance-level wrapper of it): the full path of this same call.
        out = type(runner)._prepare_inputs(runner, so, start_index, num_decode_reqs, num_windowed_reqs)
    finally:
        runner._l6_pshort_in_check = False
    if runner.input_ids is not pred_ids:
        _l6_ab.note(LEVER, "check_nocarry")  # the full path did not carry: no short step to compare
        return out
    ctx = runner._attn_metadata_builder_ctx
    md = out[7]
    bad = []
    if (out[2], out[3], out[4], out[8]) != (p.padded_num_reqs, p.n, p.n, None):
        bad.append("counts")
    if out[9] != p.layout_plan:
        bad.append("layout_plan")
    if md is None or md.draft_token_ids is not None:
        bad.append("spec_metadata")
    else:
        for f in _MD_FIELDS:
            a, b = getattr(md, f), getattr(p.md_cache, f)
            same = a is b or (
                torch.equal(a, b) if isinstance(a, torch.Tensor) else np.array_equal(a, b)
            )
            if not same:
                bad.append("spec_metadata." + f)
    if out[1] is not (md.final_logits_indices if md is not None else None):
        bad.append("logits_indices")
    if ctx.mamba_state_indices is not p.mamba_dev:
        bad.append("mamba_state_indices")
    if not torch.equal(ctx.request_distribution, p.comb_dev[0:3]) or not torch.equal(
        ctx.mamba_request_distribution, p.comb_dev[3:6]
    ):
        bad.append("request_distribution")
    if ctx.query_start_loc is not pred_qsl:
        bad.append("query_start_loc")
    if not np.array_equal(runner.positions_np[: p.total], pos):
        bad.append("positions_np")
    if not np.array_equal(runner.seq_lens_np, seq):
        bad.append("seq_lens_np")
    if not np.array_equal(runner.query_start_loc_np, qsl):
        bad.append("query_start_loc_np")
    if bad:
        runner._l6_pshort_off = True
        _l6_ab.note(LEVER, "mismatch")
        logger.error(
            "q38 L6 PREP-SHORT: CHECK mismatch (%s); the short path is off for the rest of "
            "this process", ",".join(bad),
        )
    return out
