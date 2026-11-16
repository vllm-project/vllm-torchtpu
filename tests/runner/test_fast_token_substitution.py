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
from vllm_torchtpu.runner.owned_args import (
    is_owned,
    is_stamped,
    static_shape_args_owned,
)
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _substitute_placeholder_token,
)


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
    runner = SimpleNamespace(
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
    runner = SimpleNamespace(
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


def test_substitute_placeholder_token_strips_dynamic_dim_marks():
    x = torch.tensor([-1, 10, -1, 20], dtype=torch.int32)
    torch._dynamo.mark_dynamic(x, 0)
    assert hasattr(x, "_dynamo_dynamic_indices")

    cur_idx = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
    pre_idx = torch.tensor([0, 1, -1, -1], dtype=torch.int64)
    next_tok = torch.tensor([101, 202], dtype=torch.int64)
    with torch.compiler.set_stance("force_eager"):
        out = _substitute_placeholder_token(x, cur_idx, pre_idx, next_tok)
    # The caller gets its own tensor back, updated in place.
    assert out is x
    assert out.tolist() == [101, 10, 202, 20]
    assert _substitute_placeholder_token.__code__.co_filename.endswith("owned_args.py")


@pytest.mark.tpu_test
def test_substitute_placeholder_token_compiled_writes_stamped_buffer(device):
    """Compiled path: a mark_dynamic buffer is updated in place and returned,
    and a second call of the same shape does not compile again."""
    x_np = np.array([-1, 10, -1, 20, -1, 30], dtype=np.int32)
    cur_np = np.array([0, 2, 1, 3, 4, 5], dtype=np.int64)
    pre_np = np.array([1, 0, -1, -1, -1, -1], dtype=np.int64)
    next_np = np.array([101, 202], dtype=np.int32)
    ref = x_np.copy()
    real = pre_np >= 0
    ref[cur_np[real]] = next_np[pre_np[real]]

    def call():
        x = torch.from_numpy(x_np.copy()).to(device)
        torch._dynamo.mark_dynamic(x, 0)
        out = _substitute_placeholder_token(
            x,
            torch.from_numpy(cur_np).to(device),
            torch.from_numpy(pre_np).to(device),
            torch.from_numpy(next_np).to(device),
        )
        assert out is x
        np.testing.assert_array_equal(x.cpu().numpy(), ref)

    call()
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    call()
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs


def test_static_shape_args_owned_passes_stamp_free_owned_args():
    seen = []

    def probe(a, b):
        seen.append(a)
        a.add_(1)
        return a, b

    wrapped = static_shape_args_owned(probe)
    x = torch.zeros(4, dtype=torch.int32)
    torch._dynamo.mark_dynamic(x, 0)
    big = torch.zeros(8, dtype=torch.int32)
    view = big[2:6]
    torch._dynamo.mark_dynamic(view, 0)

    out_a, out_b = wrapped(x, view)
    alias = seen[0]
    assert alias is not x and not is_stamped(alias)
    assert alias.data_ptr() == x.data_ptr()
    assert out_a is x and x.tolist() == [1, 1, 1, 1]
    # A stamped view becomes an owned, stamp-free copy.
    assert out_b is not view and is_owned(out_b) and not is_stamped(out_b)
    assert torch.equal(out_b, view)
