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
"""The plugin-facing surface of the adapter: `RunnerView` on the CPU fake
runner, `ForwardView`, the shorthand table, `StepContext`, the suppression
scope and the adapter's errors."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_torchtpu.megakernel import adapter as mkr
from vllm_torchtpu.megakernel.adapter import errors, hooks, testing
from vllm_torchtpu.megakernel.adapter import plugin as plugin_mod
from vllm_torchtpu.megakernel.adapter.plugin import ForwardView, RunnerView
from vllm_torchtpu.megakernel.core import errors as mk_errors
from vllm_torchtpu.megakernel.core.runtime import contract


def test_fake_runner_implements_every_runner_view_property() -> None:
    props = [n for n, v in vars(RunnerView).items() if isinstance(v, property)]
    assert len(props) == 17
    model = object()
    runner = testing.fake_runner(
        model=model,
        hf_config={"architectures": ["X"]},
        tp_size=2,
        speculative_tokens=3,
        block_size=128,
        req_block_pairs=((8, 32), (2, 16)),
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
    assert got["cache_block_size"] == 128
    assert got["token_paddings"] == (16, 32, 64)
    assert got["req_block_pairs"] == ((8, 32), (2, 16))
    assert got["max_pages_per_seq"] == 32
    assert got["mamba_slot_read_offsets"] is runner.mamba_slot_read_offsets
    assert got["uses_mrope"] and got["enforce_eager"] and got["unified_kv_layout"]
    assert got["device_kind"] == "cpu"
    assert RunnerView(testing.fake_runner(ckpt_window=6)).ckpt_window == 6
    # The fake's attention-kernel shapes are (sequences, pages per sequence,
    # page size), as TPUModelRunner's.
    assert runner._attention_kernel_shapes(True) == (8, 32, 128)
    assert runner._attention_kernel_shapes(False) == (2, 16, 128)


def test_fake_runner_sets_only_attributes_tpu_model_runner_has() -> None:
    """Every attribute `fake_runner` sets is one TPUModelRunner (or vLLM's
    GPUModelRunner, its base) defines, checked against the runner's source."""
    import ast
    from pathlib import Path

    import vllm_torchtpu

    src = Path(vllm_torchtpu.__file__).parent / "runner" / "tpu_runner.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    defined = {
        t.attr
        for n in ast.walk(tree)
        for t in (
            n.targets
            if isinstance(n, ast.Assign)
            else [n.target]
            if isinstance(n, ast.AnnAssign)
            else []
        )
        for t in (t.elts if isinstance(t, ast.Tuple) else [t])
        if isinstance(t, ast.Attribute)
        and isinstance(t.value, ast.Name)
        and t.value.id == "self"
    }
    defined |= {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    # Set by vLLM's GPUModelRunner.__init__ (TPUModelRunner's base class).
    base = {
        "vllm_config",
        "model_config",
        "parallel_config",
        "cache_config",
        "lora_config",
        "device",
        "supports_mm_inputs",
        "uses_mrope",
    }
    attrs = set(vars(testing.fake_runner()))
    assert sorted(attrs - defined - base) == []


def test_kv_cache_properties_need_the_kv_cache() -> None:
    """Before `initialize_kv_cache` (what `create` sees), reading a property
    that describes the KV cache raises, naming it; the rest still read."""
    view = RunnerView(testing.fake_runner(kv_cache_initialized=False), tp_rank=0)
    for name in (
        "cache_block_size",
        "req_block_pairs",
        "max_pages_per_seq",
        "mamba_slot_read_offsets",
    ):
        with pytest.raises(RuntimeError, match=rf"RunnerView\.{name} describes the KV"):
            getattr(view, name)
    assert view.tp_size == 1 and view.ckpt_window == 1 and view.token_paddings


def test_max_pages_per_seq_is_the_first_pair_s_pages() -> None:
    runner = testing.fake_runner(req_block_pairs=((4, 7), (2, 3)))
    view = RunnerView(runner)
    assert view.max_pages_per_seq == view.req_block_pairs[0][1] == 7


def test_req_block_pairs_keeps_one_entry_per_distinct_pair() -> None:
    same = testing.fake_runner(req_block_pairs=((4, 2), (4, 2)))
    assert RunnerView(same).req_block_pairs == ((4, 2),)
    # Without a most-model-len class the second shape isn't read.
    single = testing.fake_runner(req_block_pairs=((4, 2), (1, 8)))
    single.most_model_len = None
    assert RunnerView(single).req_block_pairs == ((4, 2),)


@pytest.mark.parametrize("bad", [0, -2, True, None, 2.0, "8"])
def test_tp_size_must_be_a_positive_int(bad: object) -> None:
    runner = testing.fake_runner()
    runner.parallel_config.tensor_parallel_size = bad
    with pytest.raises(RuntimeError, match="expected a positive int"):
        _ = RunnerView(runner).tp_size


def test_tp_rank_without_an_explicit_rank_asks_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = SimpleNamespace(get_tensor_model_parallel_rank=lambda: 3)
    monkeypatch.setitem(sys.modules, "vllm.distributed", fake)
    assert RunnerView(testing.fake_runner()).tp_rank == 3
    assert RunnerView(testing.fake_runner(), tp_rank=1).tp_rank == 1


def test_device_kind_on_a_tpu_runner_uses_the_utils_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_utils = SimpleNamespace(local_device_kind=lambda: "TPU7x")
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.utils", fake_utils)
    tpu = testing.fake_runner(device=SimpleNamespace(type="tpu"))
    assert RunnerView(tpu).device_kind == "TPU7x"
    assert RunnerView(testing.fake_runner()).device_kind == "cpu"


def test_local_device_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("vllm")
    import jax

    from vllm_torchtpu import utils

    monkeypatch.setattr(
        jax, "local_devices", lambda: [SimpleNamespace(device_kind="TPU7x")]
    )
    assert utils.local_device_kind() == "TPU7x"
    monkeypatch.setattr(jax, "local_devices", lambda: [])
    with pytest.raises(RuntimeError, match="No local JAX devices"):
        utils.local_device_kind()


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
        assert importlib.util.find_spec(entry.module) is not None, entry.module
        assert entry.architectures and entry.model_types
        for key in (*entry.architectures, *entry.model_types):
            assert isinstance(key, str) and claimed.setdefault(key, name) == name


def test_contract_types_have_one_import_path() -> None:
    """The contract types are imported from `megakernel.core.runtime.contract`
    only; `plugin` neither exports nor re-binds them."""
    for name in contract.__all__:
        assert not hasattr(plugin_mod, name), name
    assert not set(plugin_mod.__all__) & set(contract.__all__)
    assert contract.EligibilityResult is mk_errors.EligibilityResult
    assert issubclass(contract.LayoutDecline, mk_errors.MegakernelLayoutError)
    dec = contract.Decline("too_small", "msg")
    assert dec.result == mk_errors.EligibilityResult(False, "too_small", "msg")


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
    assert mkr.MegakernelPluginError is errors.MegakernelPluginError
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


_NO_FRAMEWORKS = """
import sys
for name in ("torch", "jax", "jaxlib", "vllm", "numpy"):
    sys.modules[name] = None
import vllm_torchtpu.megakernel.adapter.errors
import vllm_torchtpu.megakernel.adapter.hooks
import vllm_torchtpu.megakernel.adapter.plugin
"""


def test_hooks_and_plugin_import_no_framework() -> None:
    """`errors`, `hooks` and `plugin` import with torch, jax, vLLM and numpy
    blocked (a model folder's stdlib-only `__init__` imports `plugin`); an
    import of any of them would fail the subprocess."""
    src = Path(plugin_mod.__file__).resolve().parents[3]
    res = subprocess.run(
        [sys.executable, "-c", _NO_FRAMEWORKS],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
        timeout=120,
    )
    assert res.returncode == 0, res.stderr[-3000:]
