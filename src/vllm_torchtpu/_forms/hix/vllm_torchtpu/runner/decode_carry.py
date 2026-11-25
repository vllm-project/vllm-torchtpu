# SPDX-License-Identifier: Apache-2.0
"""q38 L6-HOST lever ``carry`` (VLLM_TPU_L6_DECODE_CARRY): steady-state decode
carry for async speculative decode.

Ported from the Qwen3.5 PD lane (IX) branch PR14-mtp-draft-path: dc58b5f65
"Steady-state decode carry for async speculative decode", 89026c0c9 "cached
numpy view" and 4a35009a4 "Decode carry by rule" (Shrihari Murali). The idea
and the host check are IX's; the Qwen3.8 runner needs a different hook point
(the fused subtract / substitute / extract program of VLLM_TPU_POSTFWD_FUSED)
and more device state, so this is a port by hand.

Every async-spec decode step uploads the host's optimistic ``input_ids``,
``positions`` and ``seq_lens`` only for the fused program to subtract the
previous step's rejected-draft count and to overwrite every real input id from
the device-resident draft source. In steady state (the same requests in the
same order, the same scheduled tokens each, the fused program last step too)
the host's counts advance by exactly ``1 + K`` minus the rejected count the
device subtracted last step. The runner checks that per request on the host,
from values it already holds (its own host arrays from the previous step, kept
here, and the rejected counts ``_modify_prev_results`` reads back). When it
holds, ``previous corrected device tensor + (1 + K)`` on every real slot equals
what the upload would bring; padding slots keep their previous values, which
the host also left unchanged (checked). The step then feeds the previous
step's corrected device tensors to ``carry_subtract_substitute_extract`` (the
``+ (1 + K)``, then the fused program's function) and skips the three
uploads. It also reuses the previous step's rejected-index uploads, its
substitution indices and its ``query_start_loc`` upload, whose host inputs are
equal by the same check. Anything else takes the upload path. Bit-exact by
construction; ``VLLM_TPU_L6_DECODE_CARRY_CHECK=N`` runs the upload path as a
reference every N carried steps and compares on the host (it synchronizes).

While the lever is on, the runner also reuses the previous upload of the mamba
state indices and of the combined RPA / GDN request distribution whenever
their host bytes are unchanged (``_upload_unless_unchanged``), on every step.

Scope: one chunk per step, the default (ALL) sequence layout, no prefill
context parallelism, no M-RoPE, and VLLM_TPU_POSTFWD_FUSED on (the served
decode configuration). Engagement: the L6AB window line counts
``carry.carried``, ``carry.uploaded`` (with ``carry.why_<reason>``),
``carry.checked`` / ``carry.mismatch`` and ``carry.{mamba,comb}_{reused,uploaded}``.
"""

from __future__ import annotations

import itertools
import os
from typing import Any

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.runner import l6_fskip as _l6_fskip
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import host_phases
from vllm_torchtpu.runner.owned_args import static_shape_args_owned
from vllm_torchtpu.runner.tpu_runner_async_output import subtract_substitute_extract

logger = init_logger(__name__)

LEVER = "carry"
_SEQ = itertools.count(1)


def carry_subtract_substitute_extract(
    step_delta: int,
    seq_lens_prev: torch.Tensor,
    positions_prev: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
    input_ids_prev: torch.Tensor,
    cur_input_indices: torch.Tensor,
    pre_next_tokens_indices: torch.Tensor,
    next_tokens: torch.Tensor,
    logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The carry form of ``subtract_substitute_extract`` (after IX's
    ``carry_subtract_and_substitute_draft_token_ids``).

    Inputs are the previous step's corrected device tensors. On every slot the
    subtraction touches (index >= 0) the host's next optimistic value is
    ``prev + step_delta`` (the runner checked it); the other slots (padding)
    keep their values, which the host also left unchanged. Every real input
    slot is substituted from ``next_tokens``, so ``input_ids_prev`` only
    supplies the zero padding. The result is bit-identical to uploading the host
    values and calling ``subtract_substitute_extract``.
    """
    zero_s = seq_lens_prev.new_zeros(())
    seq_lens = seq_lens_prev + torch.where(
        seq_lens_subtract_indices >= 0, zero_s + step_delta, zero_s
    )
    zero_p = positions_prev.new_zeros(())
    positions = positions_prev + torch.where(
        positions_subtract_indices >= 0, zero_p + step_delta, zero_p
    )
    return subtract_substitute_extract(
        seq_lens,
        positions,
        num_rejected,
        seq_lens_subtract_indices,
        positions_subtract_indices,
        input_ids_prev,
        cur_input_indices,
        pre_next_tokens_indices,
        next_tokens,
        logits_indices,
        target_logits_indices,
    )


def _carry_body(step_delta: int, *args: torch.Tensor):
    return carry_subtract_substitute_extract(step_delta, *args)


_PROGRAM: list = []


def _carry_compiled():
    """The carry program, compiled (``backend="tpu"``, fullgraph, static
    shapes) on first use, i.e. at the warmup precompile."""
    if not _PROGRAM:
        _PROGRAM.append(torch.compile(_carry_body, backend="tpu", fullgraph=True, dynamic=False))
    return _PROGRAM[0]


def carry_fused(step_delta: int, *args: torch.Tensor):
    """The carry program with dynamo's mark_dynamic stamps stripped from its
    arguments, as ``tpu_runner._postfwd_subst_fused`` calls its program (a
    carried positions tensor was the previous forward's input, which vLLM
    stamps)."""
    return static_shape_args_owned(_carry_compiled())(step_delta, *args)


def flag_set() -> bool:
    """Whether VLLM_TPU_L6_DECODE_CARRY is non-zero (1 or ab:<classes>), so the
    warmup compiles the carry programs even if the lever is off right now."""
    cfg = _l6_ab._cfg.get(LEVER)
    return cfg is not None and (cfg[0] or cfg[1] is not None)


def on() -> bool:
    return _l6_ab.ACTIVE and _l6_ab.on(LEVER)


def _note(key: str) -> None:
    _l6_ab.note(LEVER, key)


def tag(results: Any) -> int:
    """A process-unique id for an AsyncPreResults (assigned on first use)."""
    t = results.__dict__.get("_l6_carry_tag")
    if t is None:
        t = next(_SEQ)
        results._l6_carry_tag = t
    return t


def record_rejected(runner: Any, results: Any, rejected: dict[str, int]) -> None:
    """``_modify_prev_results``: the per-request rejected count it read back
    for `results`, stamped with this step."""
    runner._l6_carry_rej = (_l6_ab.step_seq(), tag(results), rejected)


def clear(runner: Any) -> None:
    runner.__dict__["_l6_carry_state"] = None


class CarryState:
    """What one recorded step leaves for the next."""

    __slots__ = (
        "step",
        "src_tag",
        "req_ids",
        "nsched_b",
        "nsched",
        "target",
        "total",
        "padded_total",
        "S",
        "seq_pad",
        "pos",
        "pos_pad",
        "seq_lens",
        "positions",
        "input_ids",
        "step_delta",
        "map",
        "seq_idx_dev",
        "pos_idx_dev",
        "qsl_dev",
        "subst",
        # Set by check() for the consuming step:
        "map_same",
        # q38 L6 levers clean / pshort (runner/l6_carry_lean.py): the STEADY reorder cache in
        # effect at record time (the epoch), the previous results' position map, and the step a
        # lean check accepted this state.
        "ro",
        "map_d",
        "lean_step",
    )


class StepCtx:
    """This chunk's carry context, created by ``begin`` when the lever is on."""

    __slots__ = (
        "carry",
        "token_slice",
        "target",
        "nsched",
        "qsl_dev",
        "subst_out",
        "recorded",
        # q38 L6 lever pshort (runner/l6_prep_short.py): writes the host input ids the short
        # path skipped, before any upload reads them; the carry program's shape key when the
        # short path proved it equal to the previous carried call's (else _run_carry builds it).
        "materialize",
        "carry_key",
    )

    def __init__(self, carry, token_slice, target, nsched):
        self.carry = carry
        self.token_slice = token_slice
        self.target = target
        self.nsched = nsched
        self.qsl_dev = None
        self.subst_out = None
        self.recorded = False
        self.materialize = None
        self.carry_key = None


def _layout_ok(runner: Any, layout_plan: Any, padded_total: int) -> bool:
    if getattr(runner, "_pcp_enabled", False) or getattr(runner, "uses_mrope", False):
        return False
    if getattr(layout_plan, "kind", None) is not SequenceLayoutKind.ALL:
        return False
    sl = layout_plan.token_slice
    return (sl.start or 0) == 0 and sl.stop == padded_total and sl.step in (None, 1)


def _pre_ok(runner: Any) -> bool:
    pre = runner._pre_async_results
    return (
        bool(runner.scheduler_config.async_scheduling)
        and runner.speculative_config is not None
        and pre is not None
        and pre.spec_decode_num_rejected_tokens is not None
        and bool(pre.req_id_to_index_copy)
        and pre.next_tokens_tpu is not None
        and pre.next_tokens_tpu.dim() == 1
    )


def begin(
    runner: Any,
    *,
    start_index: int,
    num_reqs: int,
    nsched: np.ndarray,
    total: int,
    padded_total: int,
    target: int,
    layout_plan: Any,
) -> StepCtx:
    """Decide whether this chunk may skip the uploads; counts the decision."""
    # q38 L6 lever clean (runner/l6_carry_lean.py): the lean check, else this one.
    st, why = (_clean.check if _clean.on(runner) else _check)(
        runner, start_index, num_reqs, nsched, total, padded_total, target, layout_plan
    )
    if st is None:
        _note("uploaded")
        _note("why_" + why)
    return StepCtx(st, layout_plan.token_slice, target, nsched)


def _check(runner, start_index, num_reqs, nsched, total, padded_total, target, layout_plan):
    """(state, "") if the carry holds for this chunk, else (None, reason)."""
    st = runner.__dict__.get("_l6_carry_state")
    if st is None:
        return None, "nostate"
    step = _l6_ab.step_seq()
    if st.step != step - 1:
        return None, "stale"
    if start_index != 0 or num_reqs != runner.input_batch.num_reqs:
        return None, "chunk"
    if not envs.VLLM_TPU_POSTFWD_FUSED or not _layout_ok(runner, layout_plan, padded_total):
        return None, "config"
    if not _pre_ok(runner):
        return None, "pre"
    pre = runner._pre_async_results
    if tag(pre) == st.src_tag:
        return None, "pre"
    if (
        total != st.total
        or padded_total != st.padded_total
        or target != st.target
        or nsched.tobytes() != st.nsched_b
    ):
        return None, "batch"
    req_ids = tuple(runner.input_batch.req_ids[:num_reqs])
    if req_ids != st.req_ids:
        return None, "batch"
    mapped = pre.req_id_to_index_copy
    try:
        map_now = tuple(mapped[r] for r in req_ids)
    except KeyError:
        return None, "map"
    rec = runner.__dict__.get("_l6_carry_rej")
    if rec is None or rec[0] != st.step or rec[1] != st.src_tag:
        return None, "rej"
    rej = rec[2]
    try:
        r_prev = np.fromiter((rej[r] for r in req_ids), dtype=np.int64, count=num_reqs)
    except KeyError:
        return None, "rej"
    d_s = runner.seq_lens_np[:num_reqs] - st.S
    if not np.all(d_s + r_prev == st.step_delta):
        return None, "seq"
    if not np.array_equal(runner.seq_lens_np[num_reqs:target], st.seq_pad):
        return None, "pad"
    if not np.array_equal(runner.positions_np[:total], st.pos + np.repeat(d_s, st.nsched)):
        return None, "pos"
    if not np.array_equal(runner.positions_np[total:padded_total], st.pos_pad):
        return None, "pad"
    if int(st.input_ids.shape[0]) != padded_total or int(st.positions.shape[0]) != padded_total:
        return None, "shape"
    st.map_same = map_now == st.map
    return st, ""


def subst_indices(runner: Any, start_index: int, num_reqs: int, nsched: np.ndarray, input_len: int):
    """(cur_np, pre_np, cur_dev, pre_dev): this chunk's substitution indices,
    as ``_postfwd_fused_prepare`` builds them."""
    cur_np, pre_np = runner._prepare_async_token_substitution_indices(start_index, num_reqs, nsched)
    if len(cur_np) == 0:
        return cur_np, pre_np, None, None
    cur_dev, pre_dev = runner._postfwd_subst_indices(input_len, cur_np, pre_np)
    return cur_np, pre_np, cur_dev, pre_dev


def upload(runner: Any, ctx: StepCtx) -> torch.Tensor:
    """The three uploads ``_prepare_inputs`` skipped (non-M-RoPE form);
    sets ``input_ids`` / ``position_ids`` and returns ``seq_lens``."""
    if ctx.materialize is not None:
        ctx.materialize()  # q38 L6 lever pshort: the host input ids it skipped
    _l6_fskip.S.n += 1  # q38 L6 fskip (runner/l6_fskip.py): three uploads from the staging tensors
    runner.input_ids = runner.input_ids_cpu[ctx.token_slice].to(runner.device, non_blocking=True)
    runner.position_ids = runner.positions_cpu[ctx.token_slice].to(runner.device, non_blocking=True)
    return runner.seq_lens_cpu[: ctx.target].to(runner.device, non_blocking=True)


def rejected_indices(runner: Any, start_index: int, num_reqs: int, nsched: np.ndarray, target: int,
                     padded_total: int) -> tuple[np.ndarray, np.ndarray]:
    """The host seq / position subtraction indices ``_prepare_inputs`` builds."""
    seq_idx_np = np.full(target, -1, dtype=np.int32)
    pos_idx_np = np.full(padded_total, -1, dtype=np.int32)
    mapped = runner._pre_async_results.req_id_to_index_copy
    acc = 0
    for i in range(num_reqs):
        n = int(nsched[i])
        pos = mapped.get(runner.input_batch.req_ids[start_index + i])
        if pos is not None:
            seq_idx_np[i] = pos
            pos_idx_np[acc : acc + n] = pos
        acc += n
    return seq_idx_np, pos_idx_np


def reuse_upload(runner: Any, slot: str, host: torch.Tensor, what: str) -> torch.Tensor:
    """``_upload_unless_unchanged`` with an engagement count."""
    prev = runner._h2d_reuse.get(slot)
    dev = runner._upload_unless_unchanged(slot, host)
    _note(what + ("_reused" if prev is not None and dev is prev[1] else "_uploaded"))
    return dev


def fused_prepare(
    runner: Any,
    seq_lens: torch.Tensor,
    pending: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    start_index: int,
    num_reqs: int,
    nsched: np.ndarray,
    md: Any,
    ctx: StepCtx,
    adc: Any = None,
) -> torch.Tensor:
    """``_postfwd_fused_prepare`` with the lever on: the carry program on a
    carried chunk, else the three uploads (if they were skipped) and the
    original path; records the state for the next step. ``adc`` (lever
    adcarry, runner/adapt_carry.py): the merge context, passed on."""
    st = ctx.carry
    if st is not None:
        out = _run_carry(runner, pending, start_index, num_reqs, nsched, md, ctx, st, adc)
        if out is not None:
            return out
        _note("uploaded")
        seq_lens = upload(runner, ctx)
        # q38 L6-INSTR PR-0 pi_post sub-marks (HOSTSUBS / HOSTSLOW; no-ops when off).
        host_phases.sub("pp_upload")
    input_ids_in = runner.input_ids
    seq_lens = runner._postfwd_fused_prepare(
        seq_lens, pending, start_index, num_reqs, nsched, md, adc=adc
    )
    fused = runner.__dict__.get("_postfwd_fused_out")
    if fused is not None and fused[0] == start_index and _recordable(runner, start_index, num_reqs, md):
        subst = subst_indices(runner, start_index, num_reqs, nsched, int(input_ids_in.shape[0]))
        ctx.subst_out = (subst[0], subst[1])
        _record(runner, ctx, start_index, num_reqs, nsched, seq_lens, runner.position_ids,
                input_ids_in, pending, subst)
        host_phases.sub("pp_record")
    return seq_lens


def _recordable(runner: Any, start_index: int, num_reqs: int, md: Any) -> bool:
    if md is None or start_index != 0 or num_reqs != runner.input_batch.num_reqs:
        return False
    if not _pre_ok(runner):
        return False
    mapped = runner._pre_async_results.req_id_to_index_copy
    return all(r in mapped for r in runner.input_batch.req_ids[:num_reqs])


def _run_carry(runner, pending, start_index, num_reqs, nsched, md, ctx, st, adc=None):
    """The carried chunk's fused program; None (nothing changed) to fall back."""
    pre = runner._pre_async_results
    if md is None:
        _note("why_nospec")
        return None
    num_rejected, seq_idx, pos_idx = pending
    if st.map_same:
        subst = st.subst
    else:
        subst = subst_indices(runner, start_index, num_reqs, nsched, st.padded_total)
    # q38 L6-INSTR PR-0 pi_post sub-marks of the carried chunk (no-ops when off).
    host_phases.sub("pp_subst_idx")
    if subst[2] is None:
        _note("why_nosubst")
        return None
    args = (
        st.seq_lens,
        st.positions,
        num_rejected,
        seq_idx,
        pos_idx,
        st.input_ids,
        subst[2],
        subst[3],
        pre.next_tokens_tpu,
        md.final_logits_indices,
        md.target_logits_indices,
    )
    key = ctx.carry_key  # q38 L6 lever pshort: proven equal, else built here
    if key is None:
        key = ctx.carry_key = (st.step_delta,) + tuple((tuple(a.shape), a.dtype) for a in args)
    warm = runner.__dict__.setdefault("_l6_carry_warm_keys", set())
    if key not in warm and getattr(runner, "warmup_done", False):
        _note("why_cold")  # serving never compiles a new carry shape
        return None
    host_phases.sub("pp_key")
    out = None
    if adc is not None:
        # Lever adcarry: the carry program with the model adapter's input program in it.
        from vllm_torchtpu.runner import adapt_carry

        out = adapt_carry.run_merged(runner, adc, "carry", args, key, st.step_delta)
    if out is None:
        out = carry_fused(st.step_delta, *args)
    host_phases.sub("pp_fused")
    warm.add(key)
    n = runner.__dict__.get("_l6_carry_n", 0) + 1
    runner._l6_carry_n = n
    every = _check_every()
    if every and n % every == 0:
        ok, out = _self_check(runner, out, pending, start_index, num_reqs, nsched, md, ctx, st, subst)
        if not ok:
            return out[0]
    seq_lens, positions, input_ids, drafts = out
    runner.position_ids = positions
    runner.input_ids = st.input_ids
    runner._postfwd_fused_out = (start_index, input_ids, drafts)
    stats = runner.__dict__.setdefault("_postfwd_stats", [0, 0])
    stats[0] += 1
    ctx.subst_out = (subst[0], subst[1])
    _record(runner, ctx, start_index, num_reqs, nsched, seq_lens, positions, st.input_ids,
            pending, subst)
    host_phases.sub("pp_record")
    _note("carried")
    return seq_lens


def _check_every() -> int:
    raw = os.environ.get("VLLM_TPU_L6_DECODE_CARRY_CHECK", "").strip()
    if not raw:
        return 0
    from vllm_torchtpu.core import l6_ab

    # VLLM_TPU_L6_CHECK_STEPS bounds every L6 CHECK mode to the first N worker steps.
    return int(raw) if l6_ab.check_allowed() else 0


def _self_check(runner, out, pending, start_index, num_reqs, nsched, md, ctx, st, subst):
    """VLLM_TPU_L6_DECODE_CARRY_CHECK: rebuild this chunk the upload way
    (fresh uploads, fresh indices, the fused program) and compare on the host.
    Returns (True, carry outputs) or, on a mismatch, (False, (seq_lens,)) with
    the reference applied and the state dropped."""
    from vllm_torchtpu.runner import tpu_runner

    pre = runner._pre_async_results
    ref_seq = upload(runner, ctx)
    seq_idx_np, pos_idx_np = rejected_indices(runner, start_index, num_reqs, nsched, ctx.target,
                                              st.padded_total)
    cur_np, pre_np = runner._prepare_async_token_substitution_indices(start_index, num_reqs, nsched)
    qsl_np = runner.query_start_loc_cpu[: ctx.target + 1].numpy()
    dev = runner.device
    ref = tpu_runner._postfwd_subst_fused(
        ref_seq,
        runner.position_ids,
        pending[0],
        torch.from_numpy(seq_idx_np).to(dev),
        torch.from_numpy(pos_idx_np).to(dev),
        runner.input_ids,
        *runner._postfwd_subst_indices(int(runner.input_ids.shape[0]), cur_np, pre_np),
        pre.next_tokens_tpu,
        md.final_logits_indices,
        md.target_logits_indices,
    )
    names = ("seq_lens", "positions", "input_ids", "draft_token_ids")
    bad = [n for n, a, b in zip(names, out, ref) if not torch.equal(a.cpu(), b.cpu())]
    if not np.array_equal(pending[1].cpu().numpy(), seq_idx_np):
        bad.append("seq_idx")
    if not np.array_equal(pending[2].cpu().numpy(), pos_idx_np):
        bad.append("pos_idx")
    if not (np.array_equal(subst[0], cur_np) and np.array_equal(subst[1], pre_np)):
        bad.append("subst")
    if ctx.qsl_dev is not None and not np.array_equal(ctx.qsl_dev.cpu().numpy(), qsl_np):
        bad.append("query_start_loc")
    _note("checked")
    if not bad:
        runner.input_ids = st.input_ids
        return True, out
    _note("mismatch")
    logger.error(
        "q38 L6 DECODE_CARRY CHECK MISMATCH step=%d fields=%s; this step uses the upload "
        "path's values and the carry restarts", _l6_ab.step_seq(), ",".join(bad),
    )
    from vllm_torchtpu.runner import adapt_carry

    adapt_carry.discard(runner)  # lever adcarry: its parked outputs came from the carry's values
    seq_lens, positions, input_ids, drafts = ref
    runner.position_ids = positions
    runner._postfwd_fused_out = (start_index, input_ids, drafts)
    ctx.subst_out = (cur_np, pre_np)
    clear(runner)
    return False, (seq_lens,)


def _record(runner, ctx, start_index, num_reqs, nsched, seq_lens, positions, input_ids, pending, subst):
    """Keep this chunk's corrected device tensors and the host arrays they
    came from for the next step."""
    if _clean.on(runner) and _clean.record(
        runner, ctx, num_reqs, nsched, seq_lens, positions, input_ids, pending, subst
    ):
        return  # q38 L6 lever clean: same batch, updated in place
    pre = runner._pre_async_results
    total = int(nsched.sum())
    padded_total = int(input_ids.shape[0])
    st = CarryState()
    st.step = _l6_ab.step_seq()
    st.src_tag = tag(pre)
    st.req_ids = tuple(runner.input_batch.req_ids[:num_reqs])
    st.nsched_b = nsched.tobytes()
    st.nsched = nsched.copy()
    st.target = int(ctx.target)
    st.total = total
    st.padded_total = padded_total
    st.S = np.array(runner.seq_lens_np[:num_reqs], dtype=np.int64)
    st.seq_pad = np.array(runner.seq_lens_np[num_reqs : ctx.target])
    st.pos = np.array(runner.positions_np[:total], dtype=np.int64)
    st.pos_pad = np.array(runner.positions_np[total:padded_total])
    st.seq_lens = seq_lens
    st.positions = positions
    st.input_ids = input_ids
    st.step_delta = 1 + int(runner.speculative_config.num_speculative_tokens)
    mapped = pre.req_id_to_index_copy
    st.map = tuple(mapped[r] for r in st.req_ids)
    st.seq_idx_dev = pending[1]
    st.pos_idx_dev = pending[2]
    st.qsl_dev = ctx.qsl_dev
    st.subst = subst
    st.map_same = False
    if _clean.stamping():
        _clean.stamp(runner, st)  # q38 L6 levers clean / pshort: the lean fields
    runner.__dict__["_l6_carry_state"] = st
    ctx.recorded = True


def late_upload(runner: Any, ctx: StepCtx) -> torch.Tensor:
    """A carried chunk that reached no deferred subtraction (``begin`` checks
    the same conditions, so this does not happen): upload after all."""
    _note("uploaded")
    _note("why_late")
    return upload(runner, ctx)


def end(runner: Any, ctx: StepCtx) -> None:
    """End of the chunk: a chunk that recorded nothing leaves no state behind."""
    if not ctx.recorded:
        clear(runner)


def precompile(runner: Any) -> int:
    """Compile the carry program for every shape the fused program warmed, so
    the lever can turn on in an A/B window without compiling while serving.
    Returns the number of shapes."""
    keys = sorted(runner.__dict__.get("_postfwd_warm_keys", ()), key=repr)
    if not keys or runner.speculative_config is None:
        return 0
    step_delta = 1 + int(runner.speculative_config.num_speculative_tokens)
    warm = runner.__dict__.setdefault("_l6_carry_warm_keys", set())
    n = 0
    for key in keys:
        args = tuple(torch.zeros(shape, dtype=dtype).to(runner.device) for shape, dtype in key)
        try:
            carry_fused(step_delta, *args)
        except Exception:
            # The shape stays cold: serving takes the upload path for it.
            logger.warning("q38 L6 DECODE_CARRY: carry program compile failed for %s", key,
                           exc_info=True)
            continue
        warm.add((step_delta,) + tuple(key))
        n += 1
    logger.info(
        "q38 L6 DECODE_CARRY: precompiled the carry program for %d of %d fused shapes",
        n, len(keys),
    )
    return n


# q38 L6 lever clean (CARRY-LEAN): imported last, it imports this module.
from vllm_torchtpu.runner import l6_carry_lean as _clean  # noqa: E402
