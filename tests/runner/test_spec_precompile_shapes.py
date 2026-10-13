# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_torchtpu.runner.spec_precompile_shapes import (
    num_reqs_sweep_pairs,
    spec_selected_logits_pairs,
    spec_verify_logits_lengths,
)

# GLM-5.2-FP8 AgentX serving (2x2x2, MTP K=3, max_num_seqs=8, mnbt=4096).
GLM_TOKENS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
GLM_REQS = [8]


def _pad(value, paddings):
    return next((p for p in sorted(paddings) if p >= value), None)


def test_glm_agentx_covers_every_shape_compiled_while_serving():
    # The (hidden rows, gathered rows) of the 11 compute_selected_logits
    # executables j-4b5c5184 compiled during serving, read off the PJRT log's
    # argument / output sizes (vocab shard 9680 x hidden 6144).
    served = {
        (2048, 16), (1024, 16), (4096, 16), (256, 16), (512, 16), (64, 16),
        (2048, 32), (4096, 32), (512, 32), (1024, 32), (256, 32),
    }  # fmt: skip
    pairs = spec_selected_logits_pairs(GLM_TOKENS, GLM_REQS, 8, 3)
    assert served <= set(pairs)
    assert spec_verify_logits_lengths(GLM_TOKENS, 8, 3) == [16, 32]
    # 9 token buckets at length 16 plus 8 at length 32.
    assert len(pairs) == 17


@pytest.mark.parametrize("spec_k", [1, 3, 7])
@pytest.mark.parametrize(
    "tokens,reqs,max_reqs",
    [
        (GLM_TOKENS, GLM_REQS, 8),
        ([32, 64, 128, 256, 384, 512, 768], [8, 16, 32, 64, 128, 192], 192),
    ],
)
def test_every_reachable_spec_step_is_precompiled(spec_k, tokens, reqs, max_reqs):
    # n requests sample n..n*(K+1) positions out of T >= positions scheduled
    # tokens; the padded (T, positions) pair must be compiled by one sweep.
    compiled = set(spec_selected_logits_pairs(tokens, reqs, max_reqs, spec_k))
    compiled |= num_reqs_sweep_pairs(tokens, reqs, max_reqs)
    for n in range(1, max_reqs + 1):
        for positions in range(n, n * (spec_k + 1) + 1):
            length = _pad(positions, tokens)
            if length is None:
                break
            for num_tokens in tokens:
                if num_tokens >= length:
                    assert (num_tokens, length) in compiled, (n, positions)


def test_pairs_are_new_and_never_gather_more_rows_than_tokens():
    tokens = [32, 64, 128, 256, 384, 512, 768]
    reqs = [8, 16, 32, 64, 128, 192]
    pairs = spec_selected_logits_pairs(tokens, reqs, 192, 3)
    assert all(length <= num_tokens for num_tokens, length in pairs)
    assert not set(pairs) & num_reqs_sweep_pairs(tokens, reqs, 192)
    assert len(pairs) == len(set(pairs))


def test_lengths_cap_at_largest_bucket():
    assert spec_verify_logits_lengths([16, 32, 64], 64, 3) == [16, 32, 64]
    assert spec_verify_logits_lengths([], 8, 3) == []
