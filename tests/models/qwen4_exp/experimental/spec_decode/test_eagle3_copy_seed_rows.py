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
"""Fused async draft seed extras (draft rows, padded rejected counts) and the
loop-width bootstrap scatter on the model-owned EAGLE/MTP proposer."""

from types import SimpleNamespace

import pytest
import torch

from tests.models.qwen4_exp.experimental.spec_decode.eagle3_copy_utils import (
    make_proposer,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
)
from vllm_torchtpu.models.qwen4_exp.experimental.spec_decode.eagle3 import (
    _async_draft_seed_impl,
    _async_seed_inputs_impl,
    _maybe_pad_dim0,
)


@pytest.mark.parametrize("p,num_rows", [(4, 6), (8, 4)])
def test_async_seed_inputs_impl_extras_and_bootstrap_pad_parity(p: int, num_rows: int):
    next_tokens = torch.tensor(
        [[11, 12, -1], [21, 22, 23], [0, -1, -1], [0, -1, -1]], dtype=torch.int32
    )
    num_draft = torch.tensor([2, 2, 0, 0], dtype=torch.int32)
    qsl = torch.tensor([0, 3, 6, 6, 6], dtype=torch.int32)
    num_reqs = torch.tensor(2, dtype=torch.int64)
    input_ids = torch.arange(100, 112, dtype=torch.int32)

    # Negative control: num_rows=0 omits extras and returns a 3-tuple.
    out_no_extras = _async_seed_inputs_impl(
        next_tokens, num_draft, qsl, num_reqs, input_ids, p, 0
    )
    assert len(out_no_extras) == 3

    out_extras = _async_seed_inputs_impl(
        next_tokens, num_draft, qsl, num_reqs, input_ids, p, num_rows
    )
    assert len(out_extras) == 5
    ids_e, gather_e, rej_e, rows_e, rej_pad_e = out_extras
    assert torch.equal(ids_e, out_no_extras[0])
    assert torch.equal(gather_e, out_no_extras[1])
    assert torch.equal(rej_e, out_no_extras[2])
    assert rows_e.shape == (num_rows,)
    assert rej_pad_e.shape == (num_rows,)
    if p < num_rows:
        assert torch.equal(rows_e[:p], gather_e)
        assert (rows_e[p:] == -1).all()
    else:
        assert torch.equal(rows_e, gather_e[:num_rows])
    expected_rej = torch.zeros(num_rows, dtype=rej_e.dtype)
    copy_n = min(num_rows, rej_e.shape[0])
    expected_rej[:copy_n] = rej_e[:copy_n]
    assert torch.equal(rej_pad_e, expected_rej)

    # Same values as the eager path: the separate seed, the proposer's seed
    # scatter, and the eager draft-rows / rejected-count pads.
    proposer = make_proposer(draft_tp=1, method="eagle3")
    lti, nti, rej = _async_draft_seed_impl(next_tokens, num_draft, qsl, num_reqs)
    assert torch.equal(ids_e, proposer._draft_seed_input_ids(input_ids, lti, nti))
    assert torch.equal(gather_e, _maybe_pad_dim0(lti[:p], p))
    assert torch.equal(rows_e, proposer._padded_draft_rows(gather_e, num_rows))
    assert torch.equal(rej_pad_e, _maybe_pad_dim0(rej[:num_rows], num_rows))

    # Bootstrap pad parity: repeating the last real (index, token) pair up to p
    # produces bit-identical draft_input_ids to the unpadded [num_reqs] scatter.
    lti_real = torch.tensor([2, 5], dtype=torch.int64)
    nti_real = torch.tensor([777, 888], dtype=torch.int32)
    lti_pad = torch.tensor([2, 5, 5, 5, 5, 5][:p], dtype=torch.int64)
    nti_pad = torch.tensor([777, 888, 888, 888, 888, 888][:p], dtype=torch.int32)
    if lti_pad.shape[0] < p:
        lti_pad = torch.nn.functional.pad(lti_pad, (0, p - lti_pad.shape[0]), value=5)
        nti_pad = torch.nn.functional.pad(nti_pad, (0, p - nti_pad.shape[0]), value=888)
    ref_seeded = proposer._draft_seed_input_ids(input_ids, lti_real, nti_real)
    pad_seeded = proposer._draft_seed_input_ids(input_ids, lti_pad, nti_pad)
    assert torch.equal(pad_seeded, ref_seeded)


@pytest.mark.parametrize(
    "draft_rows,default_layout,expected",
    [(True, True, 6), (True, False, 0), (False, True, 0)],
)
def test_seed_extras_rows_follows_draft_rows_and_layout(
    draft_rows: bool, default_layout: bool, expected: int
):
    proposer = make_proposer(draft_tp=1, method="eagle3")
    proposer._draft_rows = draft_rows
    proposer.runner.num_reqs_max_model_len = 6
    descriptor = DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR if default_layout else object()
    chunk = SimpleNamespace(
        attn_ctx=SimpleNamespace(use_max_model_len=True),
        sequence_layout_plan=SimpleNamespace(descriptor=descriptor),
    )
    assert proposer._seed_extras_rows(chunk) == expected
