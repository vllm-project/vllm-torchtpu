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
"""spec_glue_fusion: the substitution glue program must give the eager
sequence's values bit for bit, its outputs never views."""

import pytest
import torch

from vllm_torchtpu.runner.tpu_runner_async_output import (
    GLUE_FUSION_KEY,
    INVALID_TOKEN_ID,
    assemble_spec_next_tokens,
    compute_num_rejected,
    glue_fusion_enabled,
    spec_substitution_glue_impl,
    subtract_num_rejected_tokens,
)
from vllm_torchtpu.spec_decode.utils import fit_rows


# The runner raises this to 1024 (one dynamic=False graph per bucket); these
# tests also compile one graph per parametrized shape.
torch._dynamo.config.recompile_limit = 64


@pytest.fixture
def device():
    """TPU when torch_tpu has one, else CPU (these tests also mean something
    on CPU: the uncompiled bodies against the eager reference)."""
    try:
        import torch_tpu  # noqa: F401

        return torch.device("tpu")
    except Exception:
        return torch.device("cpu")


def _on_tpu(device) -> bool:
    return device.type == "tpu"


def _random_rejection_output(gen, rows, k, live):
    """[rows, 1 + k] int32 rejection output: per live row an accepted prefix
    plus the bonus, then INVALID_TOKEN_ID; padded rows all invalid."""
    out = torch.full((rows, 1 + k), INVALID_TOKEN_ID, dtype=torch.int32)
    for r in range(live):
        n = int(torch.randint(1, k + 2, (1,), generator=gen))
        out[r, :n] = torch.randint(0, 150000, (n,), generator=gen, dtype=torch.int32)
    return out


def _eager_glue(next_tokens, drafts, num_draft, rows, num_reqs, src_len, rej_len):
    src = assemble_spec_next_tokens(next_tokens, fit_rows(drafts, rows), num_reqs)
    if src.shape[0] < src_len:
        src = torch.nn.functional.pad(src, (0, src_len - src.shape[0]))
    rej = torch.cat([compute_num_rejected(next_tokens, fit_rows(num_draft, rows))])
    return src, rej


@pytest.mark.parametrize("rows,live,drafts_rows,bucketed", [
    (8, 3, 8, True),
    (16, 16, 16, True),
    (64, 37, 64, True),
    (32, 5, 16, True),
    (16, 9, 9, False),
])
@pytest.mark.parametrize("draft_dtype", [torch.int32, torch.int64])
def test_substitution_glue_bitwise(device, rows, live, drafts_rows, bucketed, draft_dtype):
    gen = torch.Generator().manual_seed(rows * 1000 + live)
    k, max_num_reqs = 7, 64
    nt = _random_rejection_output(gen, rows, k, live)
    drafts = torch.randint(0, 150000, (drafts_rows, k), generator=gen, dtype=draft_dtype)
    num_draft = torch.randint(0, k + 1, (rows,), generator=gen, dtype=torch.int32)
    num_reqs = None if bucketed else live
    rows_eff = rows if bucketed else live
    src_len, rej_len = max_num_reqs * (1 + k), max_num_reqs
    nt, drafts, num_draft = nt.to(device), drafts.to(device), num_draft.to(device)

    ref_src, ref_rej = _eager_glue(nt, drafts, num_draft, rows_eff, num_reqs, src_len, 0)
    fns = [spec_substitution_glue_impl]
    if _on_tpu(device):
        from vllm_torchtpu.runner.tpu_runner_async_output import spec_substitution_glue

        fns.append(spec_substitution_glue)
    for fn in fns:
        src, rej = fn(nt, drafts, num_draft, rows_eff, num_reqs, src_len, rej_len)
        assert src.dtype == ref_src.dtype and rej.dtype == ref_rej.dtype
        assert torch.equal(src.cpu(), ref_src.cpu())
        n = ref_rej.shape[0]
        assert rej.shape[0] == max(n, rej_len)
        assert torch.equal(rej[:n].cpu(), ref_rej.cpu())
        assert not rej[n:].cpu().any()
        assert src._base is None and rej._base is None


@pytest.mark.parametrize("num_reqs,num_tokens,src_rows", [
    (8, 64, 8), (16, 128, 16), (64, 512, 64), (64, 8192, 64), (8, 2048, 4)])
def test_subtract_with_padded_rejected_bitwise(device, num_reqs, num_tokens, src_rows):
    """The fixed-length (zero-padded) rejected count, and the compiled
    subtract on TPU, give the eager subtract's values."""
    gen = torch.Generator().manual_seed(num_tokens + num_reqs)
    rej_len = 64
    seq_lens = torch.randint(100, 9000, (num_reqs,), generator=gen, dtype=torch.int32)
    positions = torch.randint(0, 9000, (num_tokens,), generator=gen, dtype=torch.int32)
    num_rejected = torch.randint(0, 8, (src_rows,), generator=gen, dtype=torch.int32)
    seq_idx = torch.randint(-1, src_rows, (num_reqs,), generator=gen, dtype=torch.int32)
    pos_idx = torch.randint(-1, src_rows, (num_tokens,), generator=gen, dtype=torch.int32)
    padded = torch.nn.functional.pad(num_rejected, (0, rej_len - src_rows))
    args = [t.to(device) for t in (seq_lens, positions, num_rejected, seq_idx, pos_idx)]
    ref = subtract_num_rejected_tokens(*args)
    args_padded = list(args)
    args_padded[2] = padded.to(device)
    outs = [subtract_num_rejected_tokens(*args_padded)]
    if _on_tpu(device):
        from vllm_torchtpu.runner.tpu_runner import (
            _subtract_num_rejected_tokens_compiled as subtract_num_rejected_tokens_compiled,
        )

        outs.append(subtract_num_rejected_tokens_compiled(*args_padded))
        outs.append(subtract_num_rejected_tokens_compiled(*args))
    for out in outs:
        for a, b in zip(out, ref):
            assert a.dtype == b.dtype
            assert torch.equal(a.cpu(), b.cpu())


def test_glue_fusion_enabled():
    """additional_config gate reproduces the old TPU_SPEC_GLUE_FUSION env
    behaviour: off by default, on only for an explicit bool True."""
    assert not glue_fusion_enabled({})
    assert not glue_fusion_enabled(None)
    assert not glue_fusion_enabled({GLUE_FUSION_KEY: False})
    assert glue_fusion_enabled({GLUE_FUSION_KEY: True})
    with pytest.raises(ValueError):
        glue_fusion_enabled({GLUE_FUSION_KEY: 1})
