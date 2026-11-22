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

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _substitute_and_extract_drafts,
    _substitute_placeholder_token,
)


def _runner(**attrs):
    runner = object.__new__(TPUModelRunner)
    runner.__dict__.update(attrs)
    return runner


def test_env_flag_parsing(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", raising=False)
    assert envs.VLLM_TPU_FAST_TOKEN_SUBSTITUTION is False

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "1")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is True

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "true")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is True

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "0")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is False

    monkeypatch.setenv("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "false")
    assert envs.environment_variables["VLLM_TPU_FAST_TOKEN_SUBSTITUTION"]() is False


def test_substitution_disabled_by_default():
    """When _fast_token_substitution is False, calls _substitute_placeholder_token."""
    runner = _runner(
        _fast_token_substitution=False,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=torch.tensor([101, 102, 103, 104], dtype=torch.int64)
        ),
    )
    input_ids = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
    cur_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    pre_indices = np.array([0, 1, 2, 3], dtype=np.int32)

    with patch(
        "vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token"
    ) as mock_kernel:
        mock_kernel.return_value = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
        out = TPUModelRunner._apply_async_token_substitution(
            runner, input_ids, cur_indices, pre_indices
        )
        assert mock_kernel.called
        assert torch.equal(out, torch.tensor([101, 102, 103, 104], dtype=torch.int32))


def test_substitution_enabled_fast_path():
    """When _fast_token_substitution is True, direct slicing bypasses kernel."""
    runner = _runner(
        _fast_token_substitution=True,
        device=torch.device("cpu"),
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=torch.tensor([201, 202, 203, 204], dtype=torch.int64)
        ),
    )
    input_ids = torch.tensor([-1, -1, -1, -1], dtype=torch.int32)
    cur_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    pre_indices = np.array([0, 1, 2, 3], dtype=np.int32)

    with patch(
        "vllm_torchtpu.runner.tpu_runner._substitute_placeholder_token"
    ) as mock_kernel:
        out = TPUModelRunner._apply_async_token_substitution(
            runner, input_ids, cur_indices, pre_indices
        )
        # Direct slice returns next_tokens_tpu[:4] without calling kernel
        assert not mock_kernel.called
        assert torch.equal(out, torch.tensor([201, 202, 203, 204], dtype=torch.int32))


def test_substitute_placeholder_token_rejects_mismatched_lengths():
    """The length check names all three lengths."""
    with (
        torch.compiler.set_stance("force_eager"),
        pytest.raises(
            AssertionError,
            match=r"placeholder substitution inputs must share one length: "
            r"input_ids\.shape\[0\]=4, "
            r"token_in_tpu_cur_input_indices\.shape\[0\]=3, "
            r"token_in_tpu_pre_next_tokens_indices\.shape\[0\]=4",
        ),
    ):
        _substitute_placeholder_token(
            torch.zeros(4, dtype=torch.int32),
            torch.zeros(3, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("in_order", [True, False])
def test_substitute_and_extract_drafts_matches_two_steps(
    monkeypatch, flag, fast, in_order
):
    """`_substitute_async_tokens_and_drafts` at a warmed shape (one program,
    `TPU_ASYNC_FUSED_SUBSTITUTE_DRAFTS` on) returns the substitution and the
    extracted drafts of the unwarmed path, which builds its substitution inputs
    (uploads, cached slots) once."""
    from vllm_torchtpu.runner.tpu_runner_async_output import extract_draft_token_ids

    monkeypatch.setattr(envs, "TPU_ASYNC_FUSED_SUBSTITUTE_DRAFTS", flag)
    num_tokens, k = 8, 3
    next_tokens = torch.arange(100, 100 + 4 * (1 + k), dtype=torch.int64)
    cur = np.arange(num_tokens if in_order else 6, dtype=np.int32)
    pre = cur if in_order else np.array([0, 1, 2, 4, 5, 6], dtype=np.int32)
    md = SimpleNamespace(
        final_logits_indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2, 4, 5, 6, 0, 0], dtype=torch.int32),
    )

    def runner():
        r = _runner(
            _fast_token_substitution=fast,
            device=torch.device("cpu"),
            _pre_async_results=SimpleNamespace(next_tokens_tpu=next_tokens),
            _substitute_drafts_warm=set(),
            _warming_substitute_drafts=False,
        )
        r.input_builds = 0

        def build(*args):
            r.input_builds += 1
            return TPUModelRunner._async_substitution_inputs(r, *args)

        r._async_substitution_inputs = build
        return r

    def inputs():
        return torch.arange(-num_tokens, 0, dtype=torch.int32)

    with torch.compiler.set_stance("force_eager"):
        base = TPUModelRunner._apply_async_token_substitution(
            runner(), inputs(), cur, pre
        )
        expected = (
            base,
            extract_draft_token_ids(
                base, md.final_logits_indices, md.target_logits_indices
            ),
        )
        cold = runner()
        unwarmed = cold._substitute_async_tokens_and_drafts(inputs(), cur, pre, md)
        assert cold.input_builds == 1
        warm = runner()
        warm._substitute_drafts_warm.add("warmed")
        with (
            patch(
                "vllm_torchtpu.runner.tpu_runner.program_key",
                lambda _args: "warmed",
            ),
            patch(
                "vllm_torchtpu.runner.tpu_runner._substitute_and_extract_drafts",
                wraps=_substitute_and_extract_drafts,
            ) as fused,
        ):
            warmed = warm._substitute_async_tokens_and_drafts(inputs(), cur, pre, md)
            assert fused.called == flag
        assert warm.input_builds == 1
    for got in (unwarmed, warmed):
        assert torch.equal(got[0], expected[0])
        assert torch.equal(got[1], expected[1])


@pytest.mark.parametrize("source_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_reqs", [1, 2, 3, 4])
def test_served_decode_step_meets_the_warmup_keys(monkeypatch, source_dtype, num_reqs):
    """The served step's `_substitute_and_extract_drafts` arguments, built
    as execute_model builds them (substitution indices, spec metadata, the
    drafter's substitution source), against the warmup's: the real
    spec-decode warmup's source sets the grid's template, so every decode-only
    request count runs the one program, in order (a full bucket) or indexed."""
    from contextlib import nullcontext

    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.runner.speculative_decoding_manager import (
        SpeculativeDecodingManager,
    )
    from vllm_torchtpu.runner.tpu_runner_async_output import (
        assemble_spec_next_tokens,
        extract_draft_token_ids,
    )

    k, max_reqs, paddings = 3, 4, [8, 16, 32]
    monkeypatch.setattr(envs, "TPU_ASYNC_FUSED_SUBSTITUTE_DRAFTS", True)
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    req_ids = [f"r{i}" for i in range(num_reqs)]
    rows = torch.arange(max_reqs * (k + 1), dtype=source_dtype).view(max_reqs, k + 1)
    drafts = 100 + torch.arange(max_reqs * k, dtype=source_dtype).view(max_reqs, k)
    # sample_tokens' source: [bonus, drafts] per padded request, bound length.
    source = assemble_spec_next_tokens(rows, drafts)
    runner = _runner(
        _fast_token_substitution=True,
        device=torch.device("cpu"),
        speculative_config=SimpleNamespace(num_speculative_tokens=k),
        num_tokens_paddings=paddings,
        max_num_reqs=max_reqs,
        input_batch=SimpleNamespace(req_ids=req_ids),
        _last_sequence_layout_plan=None,
        _pre_async_results=SimpleNamespace(
            next_tokens_tpu=source,
            req_id_to_index_copy={r: i for i, r in enumerate(req_ids)},
            spec_decode_num_rejected_tokens=torch.zeros(max_reqs, dtype=torch.int32),
        ),
        _substitute_drafts_warm=set(),
        _substitute_source_seen=None,
        _warming_substitute_drafts=False,
        _precompile_timed=lambda name: nullcontext(),
    )
    scheduled = np.full(num_reqs, k + 1, dtype=np.int32)
    bucket = tpu_runner._get_padded_token_len(paddings, int(scheduled.sum()))
    manager = SimpleNamespace(
        _meta_cache_key=None,
        runner=SimpleNamespace(
            input_ids_cpu=torch.zeros(64, dtype=torch.int32),
            arange_np=np.arange(64),
            num_tokens_paddings=paddings,
            device=torch.device("cpu"),
        ),
    )

    def step():
        cur, pre = runner._prepare_async_token_substitution_indices(
            0, num_reqs, scheduled
        )
        md = SpeculativeDecodingManager.get_spec_decode_metadata(
            manager,  # type: ignore[arg-type]
            np.full(num_reqs, k, dtype=np.int32),
            np.cumsum(scheduled).astype(np.int32),
            padded_num_reqs=max_reqs,
            defer_draft_token_ids=True,
        )
        input_ids = torch.zeros(bucket, dtype=torch.int32)
        return runner._substitute_async_tokens_and_drafts(input_ids, cur, pre, md), md

    with (
        torch.compiler.set_stance("force_eager"),
        patch(
            "vllm_torchtpu.runner.tpu_runner._substitute_and_extract_drafts",
            wraps=_substitute_and_extract_drafts,
        ) as fused,
    ):
        # The real warmup's step (another request count) shows the source.
        with runner._warming_substitute_drafts_ctx():
            step()
        assert runner._substitute_source_seen is source
        runner._substitute_drafts_warm.clear()
        TPUModelRunner._precompile_substitute_and_extract_drafts(runner)
        fused.reset_mock()
        (input_ids, got_drafts), md = step()
        assert fused.call_count == 1
        expected = runner._substitute_async_tokens(
            torch.zeros(bucket, dtype=torch.int32),
            *runner._async_substitution_inputs(
                torch.zeros(bucket, dtype=torch.int32),
                *runner._prepare_async_token_substitution_indices(
                    0, num_reqs, scheduled
                ),
            ),
        )
    assert torch.equal(input_ids, expected)
    assert torch.equal(
        got_drafts,
        extract_draft_token_ids(
            expected, md.final_logits_indices, md.target_logits_indices
        ),
    )
