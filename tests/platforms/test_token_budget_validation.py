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
"""Tests for rejecting a scheduler token budget above user compile_sizes."""

import re
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def check_budget():
    """Stub optional dependencies and yield the scheduler budget check."""
    stubs = {
        mod: MagicMock()
        for mod in (
            "torch",
            "torch_tpu",
            "torch_tpu._internal",
            "torch_tpu._internal.compile",
            "torch_tpu._internal.utils",
            "portpicker",
            "vllm",
            "vllm.envs",
            "vllm.logger",
            "vllm.platforms",
            "vllm.platforms.interface",
        )
        if mod not in sys.modules
    }
    with patch.dict(sys.modules, stubs):
        from vllm_torchtpu.platforms.tpu_platform import (
            _validate_token_budget_fits_compile_sizes,
        )

        yield _validate_token_budget_fits_compile_sizes


def _budget_config(sizes, batched, pcp=1, chunked=True, scheduled=None):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(compile_sizes=sizes),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=batched,
            max_num_scheduled_tokens=scheduled,
            enable_chunked_prefill=chunked,
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
    )


@pytest.mark.parametrize(
    "sizes,batched,pcp,scheduled",
    [
        pytest.param([16, 256], 256, 1, None, id="budget-equals-largest"),
        pytest.param([16, 256, 16384], 16384, 1, None, id="largest-covers-budget"),
        # The scheduler stops at the smaller of the two budgets.
        pytest.param([16, 256], 16384, 1, 256, id="scheduled-below-batched"),
        pytest.param([16, 8192], 8192, 1, 16384, id="batched-below-scheduled"),
        # Speculative decoding: vLLM fills scheduled from batched.
        pytest.param([16, 256], 256, 1, 256, id="scheduled-filled-from-batched"),
        # A compile size counts rows per rank under prefill context parallel.
        pytest.param([4096], 16384, 4, None, id="pcp4"),
        pytest.param([4096], 32768, 8, None, id="pcp8"),
    ],
)
def test_budget_within_compile_sizes_is_accepted(
    check_budget, sizes, batched, pcp, scheduled
):
    cfg = _budget_config(sizes, batched, pcp=pcp, scheduled=scheduled)
    check_budget(cfg)
    # The check never rewrites the user's config.
    assert cfg.scheduler_config.max_num_scheduled_tokens == scheduled
    assert cfg.scheduler_config.max_num_batched_tokens == batched


@pytest.mark.parametrize(
    "sizes,batched,pcp,chunked,scheduled,flag,limit,need",
    [
        pytest.param(
            [16, 256],
            16384,
            1,
            True,
            None,
            "--max-num-batched-tokens (16384)",
            256,
            16384,
            id="decode-shared-budget",
        ),
        # Speculative decoding fills scheduled == batched; the message names
        # the flag the user actually sets.
        pytest.param(
            [16, 256],
            16384,
            1,
            True,
            16384,
            "--max-num-batched-tokens (16384)",
            256,
            16384,
            id="spec-decode-filled-scheduled",
        ),
        pytest.param(
            [16, 256],
            16384,
            1,
            True,
            1024,
            "--max-num-scheduled-tokens (1024)",
            256,
            1024,
            id="explicit-scheduled",
        ),
        # No split can help without chunked prefill either.
        pytest.param(
            [16, 256],
            16384,
            1,
            False,
            None,
            "--max-num-batched-tokens (16384)",
            256,
            16384,
            id="chunked-prefill-off",
        ),
        # 32768 tokens over 4 ranks needs 8192 rows per rank.
        pytest.param(
            [4096],
            32768,
            4,
            True,
            None,
            "--max-num-batched-tokens (32768)",
            16384,
            8192,
            id="pcp4",
        ),
        # A budget that pcp does not divide: the hint rounds up, since
        # 4096 rows x 4 = 16384 tokens would still be short of 16385.
        pytest.param(
            [4096],
            16385,
            4,
            True,
            None,
            "--max-num-batched-tokens (16385)",
            16384,
            4097,
            id="pcp4-budget-not-divisible",
        ),
    ],
)
def test_budget_above_compile_sizes_raises(
    check_budget, sizes, batched, pcp, chunked, scheduled, flag, limit, need
):
    cfg = _budget_config(sizes, batched, pcp=pcp, chunked=chunked, scheduled=scheduled)
    with pytest.raises(ValueError) as excinfo:
        check_budget(cfg)
    msg = str(excinfo.value)
    assert msg.startswith(f"{flag} exceeds")
    budget_flag = flag.split(" ")[0]
    assert re.search(
        re.escape(
            f"Set {budget_flag} to at most {limit}, or add a compile size of "
            f"at least {need}."
        ),
        msg,
    )
    # Nothing is rewritten before raising.
    assert cfg.scheduler_config.max_num_scheduled_tokens == scheduled
    assert cfg.scheduler_config.max_num_batched_tokens == batched


@pytest.mark.parametrize(
    "sizes",
    [[], None, ["cudagraph_capture_sizes"], [16, "cudagraph_capture_sizes"]],
)
def test_no_numeric_compile_sizes_is_skipped(check_budget, sizes):
    check_budget(_budget_config(sizes, 16384))
