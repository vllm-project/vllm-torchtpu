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

import importlib.util
import pathlib
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


def _load_pcp_local_token_counts():
    """Load the runner's PCP layout helper by file path. pcp_layout.py needs
    only numpy; its package __init__ imports jax and the Pallas kernels."""
    import vllm_torchtpu

    path = (
        pathlib.Path(vllm_torchtpu.__file__).parent
        / "kernels"
        / "experimental"
        / "pcp_streaming_rpa"
        / "pcp_layout.py"
    )
    spec = importlib.util.spec_from_file_location("_pcp_layout_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pcp_local_token_counts


def _budget_config(sizes, batched, pcp=1, interleave=1, scheduled=None):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(compile_sizes=sizes),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=batched,
            max_num_scheduled_tokens=scheduled,
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp,
            cp_kv_cache_interleave_size=interleave,
        ),
    )


@pytest.mark.parametrize(
    "sizes,batched,pcp,interleave,scheduled",
    [
        pytest.param([16, 256], 256, 1, 1, None, id="budget-equals-largest"),
        pytest.param([16, 256, 16384], 16384, 1, 1, None, id="largest-covers-budget"),
        # The scheduler stops at the smaller of the two budgets.
        pytest.param([16, 256], 16384, 1, 1, 256, id="scheduled-below-batched"),
        pytest.param([16, 8192], 8192, 1, 1, 16384, id="batched-below-scheduled"),
        # Speculative decoding: vLLM fills scheduled from batched.
        pytest.param([16, 256], 256, 1, 1, 256, id="scheduled-filled-from-batched"),
        # A compile size counts rows on one rank under prefill context
        # parallelism; the launchers' sizes are multiples of the interleave.
        pytest.param([4096], 16384, 4, 1, None, id="pcp4"),
        pytest.param([4096], 16384, 4, 256, None, id="pcp4-interleave256"),
        pytest.param([4096], 32768, 8, 256, None, id="pcp8-interleave256"),
        pytest.param(
            [1024, 2048, 4096, 5120], 32768, 8, 256, None, id="pcp8-largest-5120"
        ),
        pytest.param([4096], 16384, 8, 128, None, id="pcp8-interleave128"),
        # The interleave is ignored without prefill context parallelism.
        pytest.param([256], 256, 1, 256, None, id="pcp1-ignores-interleave"),
    ],
)
def test_budget_within_compile_sizes_is_accepted(
    check_budget, sizes, batched, pcp, interleave, scheduled
):
    cfg = _budget_config(
        sizes, batched, pcp=pcp, interleave=interleave, scheduled=scheduled
    )
    check_budget(cfg)
    # The check never rewrites the user's config.
    assert cfg.scheduler_config.max_num_scheduled_tokens == scheduled
    assert cfg.scheduler_config.max_num_batched_tokens == batched


@pytest.mark.parametrize(
    "sizes,batched,pcp,interleave,scheduled,flag,limit,need",
    [
        pytest.param(
            [16, 256],
            16384,
            1,
            1,
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
            1,
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
            1,
            1024,
            "--max-num-scheduled-tokens (1024)",
            256,
            1024,
            id="explicit-scheduled",
        ),
        # 32768 tokens over 4 ranks needs 8192 rows per rank.
        pytest.param(
            [4096],
            32768,
            4,
            1,
            None,
            "--max-num-batched-tokens (32768)",
            16384,
            8192,
            id="pcp4",
        ),
        # Interleave 1: the hint rounds up, since 4096 rows x 4 = 16384
        # tokens would still be short of 16385.
        pytest.param(
            [4096],
            16385,
            4,
            1,
            None,
            "--max-num-batched-tokens (16385)",
            16384,
            4097,
            id="pcp4-budget-not-divisible",
        ),
        # Chunks of 256 tokens from rank 0: a 10000-token step puts
        # 9 x 256 + 256 = 2560 rows on rank 0, not 10000 / 4 = 2500. A largest
        # size of 2500 (9 full chunks + 196) fits a 9 x 1024 + 196 = 9412-token
        # step.
        pytest.param(
            [2500],
            10000,
            4,
            256,
            None,
            "--max-num-batched-tokens (10000)",
            9412,
            2560,
            id="pcp4-interleave256-uneven",
        ),
    ],
)
def test_budget_above_compile_sizes_raises(
    check_budget, sizes, batched, pcp, interleave, scheduled, flag, limit, need
):
    cfg = _budget_config(
        sizes, batched, pcp=pcp, interleave=interleave, scheduled=scheduled
    )
    with pytest.raises(ValueError) as excinfo:
        check_budget(cfg)
    msg = str(excinfo.value)
    assert msg.startswith(f"{flag} exceeds")
    assert f": {limit} tokens)" in msg
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
    "pcp,interleave,budget",
    [
        (1, 1, 16385),
        (4, 1, 16385),
        (4, 256, 10000),
        (4, 256, 16384),
        (8, 128, 16383),
        (8, 256, 32768 + 300),
    ],
)
def test_hinted_compile_size_is_the_smallest_that_fits(
    check_budget, pcp, interleave, budget
):
    """The hint is exactly the smallest largest-compile-size that passes."""
    cfg = _budget_config([1], budget, pcp=pcp, interleave=interleave)
    with pytest.raises(ValueError) as excinfo:
        check_budget(cfg)
    need = int(re.search(r"at least (\d+)\.", str(excinfo.value)).group(1))
    check_budget(_budget_config([need], budget, pcp=pcp, interleave=interleave))
    with pytest.raises(ValueError):
        check_budget(_budget_config([need - 1], budget, pcp=pcp, interleave=interleave))


@pytest.mark.parametrize(
    "pcp,interleave,budget",
    [(4, 256, 10000), (4, 256, 16384), (8, 128, 16383), (8, 256, 33068)],
)
def test_hint_matches_runner_rank_local_rows(check_budget, pcp, interleave, budget):
    """The hint equals the most rows the runner's PCP layout puts on one rank
    for a full-budget step, whether it is one request or several."""
    pcp_local_token_counts = _load_pcp_local_token_counts()
    with pytest.raises(ValueError) as excinfo:
        check_budget(_budget_config([1], budget, pcp=pcp, interleave=interleave))
    need = int(re.search(r"at least (\d+)\.", str(excinfo.value)).group(1))
    # The runner assigns a step's requests as one flat token stream, so the
    # split into requests does not change the per-rank counts.
    for q_lens in ([budget], [budget // 3, budget // 3, budget - 2 * (budget // 3)]):
        starts = [sum(q_lens[:i]) for i in range(len(q_lens))]
        counts = pcp_local_token_counts(
            q_lens, pcp, interleave, token_owner_start_offsets_per_req=starts
        )
        assert int(counts.max()) == need


@pytest.mark.parametrize("sizes", [[], None])
def test_no_compile_sizes_is_skipped(check_budget, sizes):
    check_budget(_budget_config(sizes, 16384))
