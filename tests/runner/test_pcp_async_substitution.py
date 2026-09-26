# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import numpy as np

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
    SequenceLayoutPlan,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


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
