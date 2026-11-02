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
"""The plugin-facing surface of the runtime: `RunnerView` on the CPU fake
runner, `ForwardView`, the shorthand table, `StepContext`, the suppression
scope and the runtime's errors."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_torchtpu import megakernel_runtime as mkr
from vllm_torchtpu.megakernel import errors as mk_errors
from vllm_torchtpu.megakernel.runtime import contract
from vllm_torchtpu.megakernel_runtime import errors, hooks, testing
from vllm_torchtpu.megakernel_runtime import plugin as plugin_mod
from vllm_torchtpu.megakernel_runtime.plugin import ForwardView, RunnerView


def test_fake_runner_implements_every_runner_view_property() -> None:
    props = [n for n, v in vars(RunnerView).items() if isinstance(v, property)]
    assert len(props) == 17
    model = object()
    runner = testing.fake_runner(
        model=model,
        hf_config={"architectures": ["X"]},
        tp_size=2,
        speculative_tokens=3,
        req_block_pairs=((8, 4), (2, 16)),
        max_pages_per_seq=32,
        mamba_slot_read_offsets=torch.zeros(2),
        uses_mrope=True,
        enforce_eager=True,
    )
    view = RunnerView(runner, tp_rank=1)
    got = {n: getattr(view, n) for n in props}
    assert got["model"] is model and got["vllm_config"] is runner.vllm_config
    assert got["device"] == torch.device("cpu")
    assert got["hf_config"] == {"architectures": ["X"]}
    assert (got["tp_size"], got["tp_rank"]) == (2, 1)
    assert got["speculative_tokens"] == 3 and got["ckpt_window"] == 4
    assert got["cache_block_size"] == 256
    assert got["token_paddings"] == (16, 32, 64)
    assert got["req_block_pairs"] == ((8, 32), (2, 16))
    assert got["max_pages_per_seq"] == 32
    assert got["mamba_slot_read_offsets"] is runner.mamba_slot_read_offsets
    assert got["uses_mrope"] and got["enforce_eager"] and got["unified_kv_layout"]
    assert got["device_kind"] == "cpu"
    assert RunnerView(testing.fake_runner(ckpt_window=6)).ckpt_window == 6


def test_forward_view_batch_shape_calls_the_runtime_inspection() -> None:
    calls = []

    def inspect(attn_md: object, m0: object) -> str:
        calls.append((attn_md, m0))
        return "shape"

    fwd = ForwardView({"a": 1}, None, {}, 3, inspect)
    assert fwd.batch_shape("md", "m0") == "shape" and fwd.batch_shape() == "shape"
    assert calls == [("md", "m0"), (None, None)]
    assert (fwd.attn_metadata, fwd.num_tokens) == ({"a": 1}, 3)


def test_shorthand_table_entries() -> None:
    claimed: dict[str, str] = {}
    for name, entry in plugin_mod.SHORTHANDS.items():
        assert name == name.lower() and name.isidentifier()
        assert isinstance(entry.module, str) and all(
            part.isidentifier() for part in entry.module.split(".")
        )
        assert entry.architectures and entry.model_types
        for key in (*entry.architectures, *entry.model_types):
            assert isinstance(key, str) and claimed.setdefault(key, name) == name


def test_plugin_reexports_the_contract() -> None:
    assert plugin_mod.TpuMonoSpec is contract.TpuMonoSpec
    assert plugin_mod.EligibilityResult is mk_errors.EligibilityResult
    assert plugin_mod.StepDecision is contract.StepDecision
    assert issubclass(plugin_mod.LayoutDecline, mk_errors.MegakernelLayoutError)
    dec = plugin_mod.Decline("too_small", "msg")
    assert dec.result == mk_errors.EligibilityResult(False, "too_small", "msg")
    with pytest.raises(AttributeError):
        _ = plugin_mod.NoSuchName


def test_step_context_and_package_exports() -> None:
    ctx = hooks.StepContext("ids", "pos")
    assert (ctx.inputs_embeds, ctx.intermediate_tensors) == (None, None)
    assert (ctx.model_kwargs, ctx.token_ids) == (None, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.input_ids = "other"  # type: ignore[misc]
    assert mkr.StepContext is hooks.StepContext
    assert mkr.suppressed is hooks.suppressed
    assert mkr.is_suppressed is hooks.is_suppressed
    assert mkr.MegakernelUnsupported is errors.MegakernelUnsupported
    assert issubclass(errors.MegakernelUnsupported, RuntimeError)
    assert issubclass(errors.MegakernelPluginError, RuntimeError)


def test_suppressed_nests_and_restores_on_error() -> None:
    runner = SimpleNamespace()
    assert not mkr.is_suppressed(runner) and not mkr.is_suppressed(None)
    with mkr.suppressed(runner):
        assert mkr.is_suppressed(runner)
        with mkr.suppressed(runner):
            assert mkr.is_suppressed(runner)
        assert mkr.is_suppressed(runner)
    assert not mkr.is_suppressed(runner)
    with pytest.raises(RuntimeError, match="boom"), mkr.suppressed(runner):
        raise RuntimeError("boom")
    assert not mkr.is_suppressed(runner)
    # Scoped to one runner; attribute-synthesizing mocks are never suppressed.
    other = SimpleNamespace()
    with mkr.suppressed(runner):
        assert not mkr.is_suppressed(other)
    assert not mkr.is_suppressed(mock.MagicMock())
