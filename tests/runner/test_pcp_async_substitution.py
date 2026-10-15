# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
    SequenceLayoutPlan,
)
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _padded_substitution_indices,
)


def _runner_with_async_state(
    req_ids, previous_req_positions=None, sequence_layout_plan=None
):
    if previous_req_positions is None:
        pre_async_results = None
    else:
        pre_async_results = SimpleNamespace(
            req_id_to_index_copy=dict(previous_req_positions),
            spec_decode_num_rejected_tokens=None,
        )
    return SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=list(req_ids)),
        _pre_async_results=pre_async_results,
        speculative_config=None,
        _last_sequence_layout_plan=sequence_layout_plan,
    )


def _current_async_substitution_indices(
    *,
    req_ids,
    previous_req_positions,
    scheduled_tokens,
    sequence_layout_plan=None,
):
    runner = _runner_with_async_state(
        req_ids, previous_req_positions, sequence_layout_plan
    )
    return TPUModelRunner._prepare_async_token_substitution_indices(
        runner,
        start_index=0,
        num_reqs=len(req_ids),
        num_scheduled_tokens_per_req=np.asarray(scheduled_tokens, dtype=np.int32),
    )


def _expected_pcp_local_async_substitution_indices(
    *,
    req_ids,
    previous_req_positions,
    scheduled_tokens,
    token_start_offsets,
    pcp_size,
    interleave_size,
    local_padded_tokens,
    pcp_rank,
):
    """Expected non-spec async substitution indices after PCP token packing.

    Source indices stay request-position based because next_tokens_tpu is a
    replicated sampled-token vector. Destination indices must be local indices
    into this PCP rank's rank-major token slice.
    """
    _, inverse_order = build_pcp_rank_major_token_order(
        np.asarray(scheduled_tokens, dtype=np.int32),
        pcp_size,
        interleave_size,
        int(local_padded_tokens) * int(pcp_size),
        token_start_offsets_per_req=np.asarray(token_start_offsets, dtype=np.int64),
    )

    local_start = int(pcp_rank) * int(local_padded_tokens)
    local_end = local_start + int(local_padded_tokens)
    cur_indices: list[int] = []
    src_indices: list[int] = []
    req_major_start = 0
    for req_id, n_sched in zip(req_ids, scheduled_tokens):
        n_sched = int(n_sched)
        previous_position = previous_req_positions.get(req_id)
        if previous_position is not None:
            for offset in range(n_sched):
                packed_index = int(inverse_order[req_major_start + offset])
                if local_start <= packed_index < local_end:
                    cur_indices.append(packed_index - local_start)
                    src_indices.append(int(previous_position) + offset)
        req_major_start += n_sched

    cur_indices_array = np.asarray(cur_indices, dtype=np.int32)
    src_indices_array = np.asarray(src_indices, dtype=np.int32)
    return cur_indices_array, src_indices_array


def _pcp_sequence_layout_plan(
    *,
    scheduled_tokens,
    token_start_offsets,
    pcp_size,
    interleave_size,
    local_padded_tokens,
    pcp_rank,
):
    _, inverse_order = build_pcp_rank_major_token_order(
        np.asarray(scheduled_tokens, dtype=np.int32),
        pcp_size,
        interleave_size,
        int(local_padded_tokens) * int(pcp_size),
        token_start_offsets_per_req=np.asarray(token_start_offsets, dtype=np.int64),
    )
    local_start = int(pcp_rank) * int(local_padded_tokens)
    local_end = local_start + int(local_padded_tokens)
    return SequenceLayoutPlan(
        descriptor=SequenceLayoutDescriptor(
            kind=SequenceLayoutKind.PARTIAL,
            protocol=PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
        ),
        token_slice=slice(local_start, local_start + int(local_padded_tokens)),
        global_num_tokens=int(np.sum(scheduled_tokens)),
        global_padded_num_tokens=int(local_padded_tokens) * int(pcp_size),
        local_num_tokens=int(
            np.count_nonzero(
                (local_start <= inverse_order) & (inverse_order < local_end)
            )
        ),
        local_padded_num_tokens=int(local_padded_tokens),
        _request_major_to_packed_token_indices=inverse_order,
    )


def test_prefill_only_without_prior_async_results_has_no_substitution():
    cur_indices, src_indices = _current_async_substitution_indices(
        req_ids=["long_prefill"],
        previous_req_positions=None,
        scheduled_tokens=[64],
    )

    assert cur_indices.tolist() == []
    assert src_indices.tolist() == []


def test_prefill_only_new_request_ignores_disjoint_prior_async_results():
    cur_indices, src_indices = _current_async_substitution_indices(
        req_ids=["new_long_prefill"],
        previous_req_positions={"finished_short": 0},
        scheduled_tokens=[64],
    )

    assert cur_indices.tolist() == []
    assert src_indices.tolist() == []


def test_decode_only_rank0_indices_match_pcp_local_layout_when_aligned():
    req_ids = ["decode0", "decode1", "decode2"]
    scheduled_tokens = [1, 1, 1]
    token_start_offsets = [16, 16, 16]
    previous_req_positions = {
        "decode0": 0,
        "decode1": 1,
        "decode2": 2,
    }
    sequence_layout_plan = _pcp_sequence_layout_plan(
        scheduled_tokens=scheduled_tokens,
        token_start_offsets=token_start_offsets,
        pcp_size=4,
        interleave_size=4,
        local_padded_tokens=32,
        pcp_rank=0,
    )

    current = _current_async_substitution_indices(
        req_ids=req_ids,
        previous_req_positions=previous_req_positions,
        scheduled_tokens=scheduled_tokens,
        sequence_layout_plan=sequence_layout_plan,
    )
    expected = _expected_pcp_local_async_substitution_indices(
        req_ids=req_ids,
        previous_req_positions=previous_req_positions,
        scheduled_tokens=scheduled_tokens,
        token_start_offsets=token_start_offsets,
        pcp_size=4,
        interleave_size=4,
        local_padded_tokens=32,
        pcp_rank=0,
    )

    np.testing.assert_array_equal(current[0], expected[0])
    np.testing.assert_array_equal(current[1], expected[1])


def test_mixed_prefill_decode_requires_pcp_local_destination_indices():
    """Reproduce the mixed prefill/decode corruption seen in service E2E.

    req0 is a new long prefill chunk and is not present in the previous async
    result. req1-req3 are decode tokens whose placeholder slots must be
    replaced with sampled tokens from the previous async step.

    Request-major indices are [16, 17, 18]. After PCP rank-major packing with
    pcp_size=4/interleave=4, those three decode tokens live at local indices
    [4, 5, 6] on rank 0. Using [16, 17, 18] writes the sampled tokens into
    unrelated local token slots, which is the unit-level form of the mixed
    request pollution observed end-to-end.
    """
    req_ids = ["new_long_prefill", "decode0", "decode1", "decode2"]
    scheduled_tokens = [16, 1, 1, 1]
    token_start_offsets = [0, 16, 16, 16]
    previous_req_positions = {
        "decode0": 0,
        "decode1": 1,
        "decode2": 2,
    }
    sequence_layout_plan = _pcp_sequence_layout_plan(
        scheduled_tokens=scheduled_tokens,
        token_start_offsets=token_start_offsets,
        pcp_size=4,
        interleave_size=4,
        local_padded_tokens=32,
        pcp_rank=0,
    )

    current = _current_async_substitution_indices(
        req_ids=req_ids,
        previous_req_positions=previous_req_positions,
        scheduled_tokens=scheduled_tokens,
        sequence_layout_plan=sequence_layout_plan,
    )
    expected = _expected_pcp_local_async_substitution_indices(
        req_ids=req_ids,
        previous_req_positions=previous_req_positions,
        scheduled_tokens=scheduled_tokens,
        token_start_offsets=token_start_offsets,
        pcp_size=4,
        interleave_size=4,
        local_padded_tokens=32,
        pcp_rank=0,
    )

    assert expected[0].tolist() == [4, 5, 6]
    np.testing.assert_array_equal(current[0], expected[0])
    np.testing.assert_array_equal(current[1], expected[1])


def _separate_padded_rows(num_tokens, cur_indices, src_indices):
    """The two index rows as they were built before the [2, N] upload."""
    missing_mask = np.ones(num_tokens, dtype=bool)
    missing_mask[cur_indices] = False
    missing_values = np.flatnonzero(missing_mask).astype(np.int32)
    padded_cur = np.concatenate((cur_indices, missing_values))
    padded_src = np.pad(
        src_indices,
        (0, num_tokens - len(cur_indices)),
        mode="constant",
        constant_values=-1,
    )
    return padded_cur, padded_src


def _separate_substitute(input_ids, padded_cur, padded_src, next_tokens):
    """`_substitute_placeholder_token`'s body over the two separate rows."""
    mask = padded_src > -1
    new_values = next_tokens[torch.clamp_min(padded_src, 0)].to(input_ids.dtype)
    update = torch.where(mask, new_values, input_ids[padded_cur])
    return input_ids.scatter(0, padded_cur, update)


def _mtp_substitution_indices(num_spec, scheduled_tokens, previous_req_positions):
    req_ids = list(scheduled_tokens)
    runner = _runner_with_async_state(req_ids, previous_req_positions)
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=num_spec)
    runner._pre_async_results.spec_decode_num_rejected_tokens = np.zeros(
        len(req_ids), dtype=np.int32
    )
    return TPUModelRunner._prepare_async_token_substitution_indices(
        runner,
        start_index=0,
        num_reqs=len(req_ids),
        num_scheduled_tokens_per_req=np.asarray(
            list(scheduled_tokens.values()), dtype=np.int32
        ),
    )


_SUBSTITUTION_CASES = {
    "decode": (
        8,
        _current_async_substitution_indices(
            req_ids=["a", "b", "c"],
            previous_req_positions={"a": 2, "b": 0, "c": 1},
            scheduled_tokens=[1, 1, 1],
        ),
    ),
    "mixed_prefill_decode": (
        32,
        _current_async_substitution_indices(
            req_ids=["prefill", "a", "b"],
            previous_req_positions={"a": 0, "b": 1},
            scheduled_tokens=[16, 1, 1],
        ),
    ),
    # MTP k=3: each request's source span starts at position * 4; a request
    # whose drafts were rejected schedules fewer tokens this step.
    "mtp_partial_acceptance": (
        16,
        _mtp_substitution_indices(
            3, {"a": 4, "b": 2, "c": 1}, {"a": 1, "b": 0, "c": 2}
        ),
    ),
    "all_padding": (8, (np.array([], np.int32), np.array([], np.int32))),
    "unpadded": (4, (np.arange(4, dtype=np.int32), np.arange(4, dtype=np.int32))),
}


@pytest.mark.parametrize("case", sorted(_SUBSTITUTION_CASES))
def test_one_upload_index_rows_equal_the_two_separate_rows(case):
    num_tokens, (cur_indices, src_indices) = _SUBSTITUTION_CASES[case]

    stacked = _padded_substitution_indices(num_tokens, cur_indices, src_indices)

    padded_cur, padded_src = _separate_padded_rows(num_tokens, cur_indices, src_indices)
    assert stacked.dtype == np.int32
    assert stacked.shape == (2, num_tokens)
    np.testing.assert_array_equal(stacked[0], padded_cur)
    np.testing.assert_array_equal(stacked[1], padded_src)

    input_ids = torch.arange(1000, 1000 + num_tokens, dtype=torch.int32)
    next_tokens = torch.arange(1, 65, dtype=torch.int64)
    substitute = tpu_runner._substitute_placeholder_token._torchdynamo_orig_callable
    out = substitute(input_ids.clone(), torch.from_numpy(stacked), next_tokens)
    expected = _separate_substitute(
        input_ids,
        torch.from_numpy(padded_cur),
        torch.from_numpy(padded_src),
        next_tokens,
    )
    assert torch.equal(out, expected)


@pytest.mark.parametrize("fast", [False, True])
def test_async_substitution_uploads_one_index_tensor(fast):
    num_tokens, (cur_indices, src_indices) = _SUBSTITUTION_CASES["mixed_prefill_decode"]
    runner = SimpleNamespace(
        _fast_token_substitution=fast,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=torch.arange(4, dtype=torch.int64)
        ),
    )
    input_ids = torch.zeros(num_tokens, dtype=torch.int32)
    with patch.object(tpu_runner, "_substitute_placeholder_token") as kernel:
        for _ in range(2):
            TPUModelRunner._apply_async_token_substitution(
                runner, input_ids, cur_indices, src_indices
            )
    assert kernel.call_count == 2
    first, second = (call.args[1] for call in kernel.call_args_list)
    expected = np.stack(_separate_padded_rows(num_tokens, cur_indices, src_indices))
    np.testing.assert_array_equal(first.numpy(), expected)
    np.testing.assert_array_equal(second.numpy(), expected)
    # The fast path's cache hands the same device tensor to the second step.
    assert (first is second) == fast
