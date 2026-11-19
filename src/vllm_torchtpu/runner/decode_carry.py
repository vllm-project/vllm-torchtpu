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
"""Steady-state device carry for async speculative decode.

Every async-spec decode step uploads three small host arrays (input_ids, positions,
seq_lens) from the worker's main thread, only for the fused subtract-and-substitute
program to correct positions / seq_lens by the previous step's rejected-draft count
and to overwrite every real input id from the device-resident draft source.

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

It runs on batch-1 asynchronous decode (max_num_seqs == 1) of a configuration a
model-provided draft-path rule covers (spec_decode/draft_path_rules.py), the
configuration it was measured on; every other engine keeps the uploads.
"""

from __future__ import annotations

import bisect
import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from vllm.config import SchedulerConfig, VllmConfig

    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutPlan
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    from vllm_torchtpu.runner.tpu_runner_async_output import AsyncPreResults

_SEQ = itertools.count(1)


def rule_holds(scheduler_config: SchedulerConfig, vllm_config: VllmConfig) -> bool:
    """The carry's rule: asynchronous scheduling, max_num_seqs == 1, and a
    configuration a model-provided draft-path rule covers
    (spec_decode/draft_path_rules.py). The runner decides it once, in
    __init__ (TPUModelRunner._decode_carry_enabled)."""
    from vllm_torchtpu.spec_decode import draft_path_rules

    return bool(
        scheduler_config.async_scheduling
        and scheduler_config.max_num_seqs == 1
        and draft_path_rules.holds(vllm_config)
    )


def enabled(runner: TPUModelRunner) -> bool:
    return runner._decode_carry_enabled


def token_buckets(runner: TPUModelRunner) -> list[int]:
    """The token buckets of the steps the carry runs: one request
    (max_num_seqs == 1) scheduling its bonus token and 1 to K drafts, padded to
    the first token bucket at or above that count. Under EP-DP lockstep a step
    is raised to the DP-wide bucket, which can be any token bucket."""
    paddings = runner.num_tokens_paddings
    if runner._dp_lockstep_enabled():
        return list(paddings)
    assert runner.speculative_config is not None
    k = runner.speculative_config.num_speculative_tokens
    return sorted({paddings[bisect.bisect_left(paddings, n)] for n in range(2, k + 2)})


class CarryState:
    req_ids: tuple[str, ...]
    nsched: tuple[int, ...]
    target: int
    padded_total: int
    total: int
    src_seq: int
    S: np.ndarray
    seq_pad: np.ndarray
    pos: np.ndarray
    seq_lens: torch.Tensor
    positions: torch.Tensor
    input_ids: torch.Tensor
    step_delta: int

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


def tag_results(runner: TPUModelRunner, results: AsyncPreResults) -> None:
    """Number each AsyncPreResults, so a carry can tell which step's rejections
    it saw."""
    if enabled(runner):
        results.decode_carry_seq = next(_SEQ)


def record_rejected(
    runner: TPUModelRunner, results: AsyncPreResults, rejected: dict[str, int]
) -> None:
    """Called by `_modify_prev_results` with the per-request rejected count it
    read back."""
    runner._decode_carry_rej = (results.decode_carry_seq, rejected)


def clear(runner: TPUModelRunner) -> None:
    runner._decode_carry_state = None


# The mrope position buffer is read through a cached numpy view of the whole
# CPU tensor instead of a torch slice per call. Same values, same dtype.


def _host_positions(runner: TPUModelRunner, total: int) -> np.ndarray:
    if runner.uses_mrope:
        cpu = runner.mrope_positions.cpu
        cached = runner._mrope_np_view
        key = (id(cpu), cpu.data_ptr(), tuple(cpu.shape), cpu.dtype)
        if cached is None or cached[0] != key:
            cached = (key, cpu.numpy())
            runner._mrope_np_view = cached
        return np.asarray(cached[1][:, :total], dtype=np.int64)
    return np.asarray(runner.positions_np[:total], dtype=np.int64)


def _layout_is_all(layout_plan: SequenceLayoutPlan | None) -> bool:
    """Whether the step's tokens use the ALL layout (every device holds the
    whole chunk). None is the PCP plan's place, where the carry never runs."""
    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind

    return layout_plan is not None and layout_plan.kind is SequenceLayoutKind.ALL


def check(
    runner: TPUModelRunner,
    *,
    start_index: int,
    num_reqs: int,
    num_scheduled_tokens_per_req_list: list[int],
    total_num_scheduled_tokens: int,
    padded_total_num_scheduled_tokens: int,
    target_num_reqs: int,
    layout_plan: SequenceLayoutPlan | None,
) -> CarryState | None:
    """The carried state if this step may skip the uploads, else None."""
    if not enabled(runner):
        return None
    return _check(
        runner,
        runner._decode_carry_state,
        start_index=start_index,
        num_reqs=num_reqs,
        num_scheduled_tokens_per_req_list=num_scheduled_tokens_per_req_list,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        padded_total_num_scheduled_tokens=padded_total_num_scheduled_tokens,
        target_num_reqs=target_num_reqs,
        layout_plan=layout_plan,
    )


def _check(
    runner: TPUModelRunner,
    st: CarryState | None,
    *,
    start_index: int,
    num_reqs: int,
    num_scheduled_tokens_per_req_list: list[int],
    total_num_scheduled_tokens: int,
    padded_total_num_scheduled_tokens: int,
    target_num_reqs: int,
    layout_plan: SequenceLayoutPlan | None,
) -> CarryState | None:
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
    if runner._pcp_enabled or not _layout_is_all(layout_plan):
        return None
    if (
        total_num_scheduled_tokens != st.total
        or padded_total_num_scheduled_tokens != st.padded_total
        or target_num_reqs != st.target
        or tuple(num_scheduled_tokens_per_req_list) != st.nsched
    ):
        return None
    req_ids = tuple(runner.input_batch.req_ids[:num_reqs])
    if req_ids != st.req_ids:
        return None
    mapped = pre.req_id_to_index_copy
    if any(r not in mapped for r in req_ids):
        return None
    rec = runner._decode_carry_rej
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
    if not np.array_equal(
        np.asarray(runner.seq_lens_np[num_reqs:target_num_reqs]), st.seq_pad
    ):
        return None
    pos_now = _host_positions(runner, total_num_scheduled_tokens)
    if not np.array_equal(pos_now, st.pos + np.repeat(d_s, st.nsched)):
        return None
    if int(st.input_ids.shape[0]) != padded_total_num_scheduled_tokens:
        return None
    return st


def record(
    runner: TPUModelRunner,
    *,
    start_index: int,
    num_reqs: int,
    num_scheduled_tokens_per_req_list: list[int],
    total_num_scheduled_tokens: int,
    padded_total_num_scheduled_tokens: int,
    target_num_reqs: int,
    sub_pairs: tuple[tuple[int | None, int], ...] | None,
    seq_lens: torch.Tensor,
) -> bool:
    """After a fused draft-token subtract: keep its corrected device tensors
    for the next step."""
    if not enabled(runner):
        return False
    pre = runner._pre_async_results
    src_seq = pre.decode_carry_seq if pre is not None else None
    if (
        src_seq is None
        or start_index != 0
        or num_reqs != runner.input_batch.num_reqs
        or sub_pairs is None
        or len(sub_pairs) != num_reqs
        or any(p is None for p, _ in sub_pairs)
        or runner._pcp_enabled
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
    assert runner.speculative_config is not None
    st.step_delta = 1 + int(runner.speculative_config.num_speculative_tokens)
    runner._decode_carry_state = st
    return True
