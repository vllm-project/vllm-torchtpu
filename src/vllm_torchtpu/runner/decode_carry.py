# SPDX-License-Identifier: Apache-2.0
"""Steady-state device carry for async speculative decode.

Every async-spec decode step uploads three small host arrays (input_ids, positions,
seq_lens; ~0.3 ms of worker main-thread time per step at batch 1-2, measured), only for
the fused subtract-and-substitute program to correct positions / seq_lens by the
previous step's rejected-draft count and to overwrite every real input id from the
device-resident draft source.

In steady state (same requests in the same order, the same 1+K scheduled tokens each,
a fused draft-token step last time too) the host's optimistic counts advance by
exactly `step_delta` minus the rejected count the device subtracted last step. The
runner checks that on the host, per request, from values it already holds (its own
host arrays from the previous step and the rejected counts `_modify_prev_results`
just read back). When it holds, `prev_corrected + step_delta` on every slot the
subtraction touches equals what the upload would bring, so the step feeds the
previous step's corrected device tensors to
`carry_subtract_and_substitute_draft_token_ids` and skips the uploads. Any mismatch
takes the normal upload path. Padding slots keep their previous values, which the
host also left unchanged (checked).

Default off on this branch (the unified gates ran without it); VLLM_TPU_DECODE_CARRY=1
enables it, as the hostov stack (band H j-81e043bf) ran it. VLLM_TPU_DECODE_CARRY_CHECK=1 also runs the
upload path on every carried step and compares both results on the host (debug only:
it synchronizes).
"""

from __future__ import annotations

import itertools
import os
import time
from typing import Any

import numpy as np

ENABLED = os.environ.get("VLLM_TPU_DECODE_CARRY", "0") == "1"
CHECK = os.environ.get("VLLM_TPU_DECODE_CARRY_CHECK", "0") == "1"

_SEQ = itertools.count(1)
STATS = {"carried": 0, "uploaded": 0, "check_ok": 0, "check_bad": 0}
_LAST_LOG = [0.0]


class CarryState:
    __slots__ = (
        "req_ids",
        "nsched",
        "target",
        "padded_total",
        "total",
        "src_seq",
        "S",
        "seq_pad",
        "pos",
        "seq_lens",
        "positions",
        "input_ids",
        "step_delta",
    )


def tag_results(results: Any) -> None:
    """Number each AsyncPreResults so a carry can tell which step's rejections it saw."""
    if ENABLED and results is not None:
        results._decode_carry_seq = next(_SEQ)


def record_rejected(runner: Any, results: Any, rejected: dict[str, int]) -> None:
    """Called by `_modify_prev_results` with the per-request rejected count it read back."""
    runner._decode_carry_rej = (getattr(results, "_decode_carry_seq", None), rejected)


def clear(runner: Any) -> None:
    runner.__dict__["_decode_carry_state"] = None


def _host_positions(runner: Any, total: int) -> np.ndarray:
    if runner.uses_mrope:
        return np.asarray(runner.mrope_positions.cpu[:, :total].numpy(), dtype=np.int64)
    return np.asarray(runner.positions_np[:total], dtype=np.int64)


def _layout_is_all(layout_plan: Any) -> bool:
    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind

    return getattr(layout_plan, "kind", None) is SequenceLayoutKind.ALL


def _maybe_log() -> None:
    now = time.monotonic()
    if now - _LAST_LOG[0] >= 60.0:
        _LAST_LOG[0] = now
        print(f"[decode_carry] {STATS}", flush=True)


def check(
    runner: Any,
    *,
    start_index: int,
    num_reqs: int,
    num_scheduled_tokens_per_req_list: list[int],
    total_num_scheduled_tokens: int,
    padded_total_num_scheduled_tokens: int,
    target_num_reqs: int,
    layout_plan: Any,
) -> CarryState | None:
    """The carried state if this step may skip the uploads, else None."""
    if not ENABLED:
        return None
    st = runner.__dict__.get("_decode_carry_state")
    res = _check(
        runner, st, start_index, num_reqs, num_scheduled_tokens_per_req_list,
        total_num_scheduled_tokens, padded_total_num_scheduled_tokens, target_num_reqs,
        layout_plan,
    )
    STATS["carried" if res is not None else "uploaded"] += 1
    _maybe_log()
    return res


def _check(runner, st, start_index, num_reqs, nsched_list, total, padded_total, target,
           layout_plan):
    if st is None:
        return None
    pre = runner._pre_async_results
    if (
        pre is None
        or pre.spec_decode_num_rejected_tokens is None
        or not pre.req_id_to_index_copy
        or pre.next_tokens_tpu is None
    ):
        return None
    if start_index != 0 or num_reqs != runner.input_batch.num_reqs:
        return None
    if getattr(runner, "_pcp_enabled", False) or not _layout_is_all(layout_plan):
        return None
    if (
        total != st.total
        or padded_total != st.padded_total
        or target != st.target
        or tuple(nsched_list) != st.nsched
    ):
        return None
    req_ids = tuple(runner.input_batch.req_ids[:num_reqs])
    if req_ids != st.req_ids:
        return None
    mapped = pre.req_id_to_index_copy
    if any(r not in mapped for r in req_ids):
        return None
    rec = runner.__dict__.get("_decode_carry_rej")
    if rec is None or rec[0] is None or rec[0] != st.src_seq:
        return None
    rej = rec[1]
    try:
        r_prev = np.fromiter((rej[r] for r in req_ids), dtype=np.int64, count=num_reqs)
    except KeyError:
        return None
    s_now = np.asarray(runner.seq_lens_np[:num_reqs], dtype=np.int64)
    d_s = s_now - st.S
    if not np.all(d_s + r_prev == st.step_delta):
        return None
    if not np.array_equal(np.asarray(runner.seq_lens_np[num_reqs:target]), st.seq_pad):
        return None
    pos_now = _host_positions(runner, total)
    if not np.array_equal(pos_now, st.pos + np.repeat(d_s, st.nsched)):
        return None
    if int(st.input_ids.shape[0]) != padded_total:
        return None
    return st


def will_take_fused_draft_path(runner, pending_rejected_sub, seq_lens, cur_input_indices,
                               spec_decode_metadata) -> bool:
    """Mirror of `_prepare_inputs`' conditions for the fused draft-token subtract path."""
    return (
        pending_rejected_sub is not None
        and seq_lens.device.type == "tpu"
        and getattr(runner, "max_num_reqs", None) is not None
        and len(cur_input_indices) > 0
        and runner._pre_async_results.next_tokens_tpu is not None
        and "_apply_async_token_substitution" not in getattr(runner, "__dict__", {})
        and spec_decode_metadata is not None
        and spec_decode_metadata.final_logits_indices.shape[0] == runner.input_ids.shape[0]
    )


def record(
    runner: Any,
    *,
    start_index: int,
    num_reqs: int,
    num_scheduled_tokens_per_req_list: list[int],
    total_num_scheduled_tokens: int,
    padded_total_num_scheduled_tokens: int,
    target_num_reqs: int,
    sub_pairs: Any,
    seq_lens: Any,
) -> bool:
    """After a fused draft-token subtract: keep its corrected device tensors for the next step."""
    if not ENABLED:
        return False
    pre = runner._pre_async_results
    src_seq = getattr(pre, "_decode_carry_seq", None) if pre is not None else None
    if (
        src_seq is None
        or start_index != 0
        or num_reqs != runner.input_batch.num_reqs
        or sub_pairs is None
        or len(sub_pairs) != num_reqs
        or any(p is None for p, _ in sub_pairs)
        or getattr(runner, "_pcp_enabled", False)
        or not _layout_is_all(runner._last_sequence_layout_plan)
    ):
        return False
    st = CarryState()
    st.req_ids = tuple(runner.input_batch.req_ids[:num_reqs])
    st.nsched = tuple(int(x) for x in num_scheduled_tokens_per_req_list)
    st.target = int(target_num_reqs)
    st.padded_total = int(padded_total_num_scheduled_tokens)
    st.total = int(total_num_scheduled_tokens)
    st.src_seq = src_seq
    st.S = np.array(runner.seq_lens_np[:num_reqs], dtype=np.int64)
    st.seq_pad = np.array(runner.seq_lens_np[num_reqs:target_num_reqs])
    st.pos = _host_positions(runner, st.total).copy()
    st.seq_lens = seq_lens
    st.positions = runner.position_ids
    st.input_ids = runner.input_ids
    st.step_delta = 1 + int(runner.speculative_config.num_speculative_tokens)
    runner.__dict__["_decode_carry_state"] = st
    return True


def compare(carry_out, ref_out) -> None:
    """Debug: both paths' (seq_lens, positions, input_ids, draft_token_ids) must match."""
    import torch

    ok = all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(carry_out, ref_out))
    STATS["check_ok" if ok else "check_bad"] += 1
    if not ok:
        print(
            "[decode_carry] CHECK MISMATCH "
            + " | ".join(f"{a.cpu().tolist()} vs {b.cpu().tolist()}" for a, b in zip(carry_out, ref_out)),
            flush=True,
        )
