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
"""Unit tests for multi-model megakernel hooks, Kimi-K3 shorthand, and verify."""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace
from typing import TYPE_CHECKING

import torch

import vllm_torchtpu
from vllm_torchtpu import megakernel_runtime as mkr
from vllm_torchtpu.megakernel.runtime.contract import EligibilityResult
from vllm_torchtpu.megakernel_runtime import (
    agreement as agreement_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    config as config_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    driver as driver_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    hooks as hooks_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    plugin as plugin_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    testkit,
)
from vllm_torchtpu.megakernel_runtime.testing import fake_runner
from vllm_torchtpu.models.kimi_k3.megakernel import adapter as k3_adapter

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pytest


_EXPECTED_CAPTURE_BIND_CALLS = 2


def test_step_context_kda_and_verify_fields_and_driver_forwarding() -> None:
    """Verify StepContext KDA/verify fields and driver on_verify_accepted forwarding."""
    ctx_default = hooks_mod.StepContext(input_ids="ids", positions="pos")
    assert ctx_default.req_ids is None
    assert ctx_default.finished_req_ids is None
    assert ctx_default.kda_block_tables is None
    assert ctx_default.accepted_counts is None

    kda_tbl = {"r0": torch.ones((69,), dtype=torch.int32)}
    acc = torch.tensor([3], dtype=torch.int32)
    ctx_populated = hooks_mod.StepContext(
        input_ids="ids",
        positions="pos",
        req_ids=["r0"],
        finished_req_ids=["r-old"],
        kda_block_tables=kda_tbl,
        accepted_counts=acc,
    )
    assert ctx_populated.req_ids == ["r0"]
    assert ctx_populated.finished_req_ids == ["r-old"]
    assert ctx_populated.kda_block_tables is kda_tbl
    assert ctx_populated.accepted_counts is acc

    recorded: list[tuple[list[str] | None, list[int]]] = []

    class _VerifyProgram:
        """Stub program recording on_verify_accepted calls."""

        display_name = "verify-stub"

        def on_verify_accepted(
            self,
            req_ids: Sequence[str] | None,
            num_valid: torch.Tensor,
        ) -> None:
            """Record callback arguments."""
            recorded.append(
                (
                    list(req_ids) if req_ids is not None else None,
                    [int(x) for x in num_valid.reshape(-1).tolist()],
                )
            )

    runner = fake_runner(tp_size=1)
    view = plugin_mod.RunnerView(runner, tp_rank=0)
    drv = driver_mod.ProgramDriver(_VerifyProgram(), view=view, tp_rank=0)
    drv.on_verify_accepted(["req-a", "req-b"], torch.tensor([4, 7], dtype=torch.int32))
    assert recorded == [(["req-a", "req-b"], [4, 7])]

    drv_no_hook = driver_mod.ProgramDriver(SimpleNamespace(), view=view, tp_rank=0)
    drv_no_hook.on_verify_accepted(["req-a"], torch.tensor([1], dtype=torch.int32))

    declined = agreement_mod.DeclinedDriver(
        EligibilityResult(
            eligible=False,
            reason="test_decline",
            message="declined for test",
        )
    )
    declined.on_verify_accepted(["req-a"], torch.tensor([2], dtype=torch.int32))


def test_tpu_runner_forward_model_and_mamba_offsets_wire_megakernel_hooks() -> None:
    """Verify TPUModelRunner passes req_ids and calls on_verify_accepted."""
    runner_path = Path(vllm_torchtpu.__file__).parent / "runner" / "tpu_runner.py"
    src = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    fwd_fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "forward_model"
    )
    fwd_src = ast.unparse(fwd_fn)
    assert "self._build_megakernel_step_context(" in fwd_src
    assert "megakernel_runtime.StepContext(" in fwd_src

    ctx_fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_build_megakernel_step_context"
    )
    ctx_src = ast.unparse(ctx_fn)
    assert "req_ids=step_req_ids" in ctx_src
    assert "finished_req_ids=" in ctx_src
    assert "kda_block_tables=kda_tables" in ctx_src

    kda_fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_resolve_megakernel_kda_tables"
    )
    kda_src = ast.unparse(kda_fn)
    assert "unified_mamba_state_indices" in kda_src
    assert "torch.stack(" in kda_src

    prep_fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_prepare_inputs"
    )
    prep_src = ast.unparse(prep_fn)
    assert "self._megakernel is not None" in prep_src

    mamba_fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_update_mamba_slot_read_offsets"
    )
    mamba_src = ast.unparse(mamba_fn)
    assert "on_verify_accepted" in mamba_src
    assert "on_verify_fn(active_req_ids, num_valid)" in mamba_src

    capture_fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "capture_model"
    )
    capture_src = ast.unparse(capture_fn)
    assert (
        capture_src.count("self._megakernel.bind_and_warmup(self)")
        == _EXPECTED_CAPTURE_BIND_CALLS
    )
    assert capture_src.rfind("self._megakernel.bind_and_warmup(self)") > (
        capture_src.find("self._precompile_backbone()")
    )

    resolve_fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_resolve_megakernel_kda_tables"
    )
    build_ctx_fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_build_megakernel_step_context"
    )
    runner_ns: dict[str, object] = {
        "__builtins__": __builtins__,
        "Sequence": Sequence,
        "Mapping": Mapping,
        "dataclasses": dataclasses,
        "torch": torch,
        "megakernel_runtime": SimpleNamespace(StepContext=hooks_mod.StepContext),
    }
    resolve_mod = compile(
        ast.Module(body=[resolve_fn], type_ignores=[]),
        filename=str(runner_path),
        mode="exec",
    )
    build_mod = compile(
        ast.Module(body=[build_ctx_fn], type_ignores=[]),
        filename=str(runner_path),
        mode="exec",
    )
    resolve_kda_tables = FunctionType(
        next(c for c in resolve_mod.co_consts if isinstance(c, CodeType)),
        runner_ns,
    )
    build_step_ctx = FunctionType(
        next(c for c in build_mod.co_consts if isinstance(c, CodeType)),
        runner_ns,
    )

    grp0 = SimpleNamespace(
        metadata_builders=[SimpleNamespace(is_mamba_group=True)],
        layer_names=[f"model.layers.{i}.self_attn" for i in (0, 1, 2)],
    )
    grp1 = SimpleNamespace(
        metadata_builders=[SimpleNamespace(is_mamba_group=True)],
        layer_names=[f"model.layers.{i}.self_attn" for i in (4, 5, 6)],
    )
    grp2 = SimpleNamespace(
        metadata_builders=[SimpleNamespace(is_mamba_group=True)],
        layer_names=[f"model.layers.{i}.self_attn" for i in (8, 9, 10)],
    )
    mla_grp = SimpleNamespace(
        metadata_builders=[SimpleNamespace(is_mamba_group=False)],
        layer_names=[f"model.layers.{i}.self_attn" for i in (3, 7, 11)],
    )
    dummy_runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r-active", "r-pad"], num_reqs=1),
        requests={"r-active": object()},
        _megakernel_seen_reqs={"r-finished"},
        _megakernel_finished_req_ids=None,
        _megakernel_kda_block_tables=None,
        attn_groups=[[mla_grp, grp0, grp1, grp2]],
        _attn_metadata_builder_ctx=SimpleNamespace(
            unified_mamba_state_indices=[
                torch.tensor([11, 0], dtype=torch.int32),
                torch.tensor([22, 0], dtype=torch.int32),
                torch.tensor([33, 0], dtype=torch.int32),
            ]
        ),
    )
    dummy_runner.__dict__["_resolve_megakernel_kda_tables"] = (
        lambda n: resolve_kda_tables(dummy_runner, n)
    )
    built_ctx = build_step_ctx(
        dummy_runner,
        hooks_mod.StepContext(
            input_ids=torch.tensor([1], dtype=torch.int32),
            positions=torch.tensor([0], dtype=torch.int32),
        ),
    )
    assert built_ctx.req_ids == ["r-active"]
    assert built_ctx.finished_req_ids == ("r-finished",)
    assert isinstance(built_ctx.kda_block_tables, torch.Tensor)
    assert built_ctx.kda_block_tables.shape == (1, 9)
    assert built_ctx.kda_block_tables[0].tolist() == [
        11,
        11,
        11,
        22,
        22,
        22,
        33,
        33,
        33,
    ]


def test_kimi_k3_shorthand_registration_and_plugin_resolution() -> None:
    """Verify kimi_k3 shorthand entry in plugin.SHORTHANDS and check_plugin_spec."""
    assert "kimi_k3" in plugin_mod.SHORTHANDS
    entry = plugin_mod.SHORTHANDS["kimi_k3"]
    assert entry.module == "vllm_torchtpu.models.kimi_k3.megakernel"
    assert "KimiK3ForConditionalGeneration" in entry.architectures
    assert "kimi_k3" in entry.model_types

    cfg_arch, resolved_arch = config_mod.select_plugin(
        "1",
        None,
        SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]),
    )
    assert cfg_arch.enabled is True
    assert resolved_arch is not None
    ref_arch, opts_arch = resolved_arch
    assert ref_arch.module == "vllm_torchtpu.models.kimi_k3.megakernel"
    assert ref_arch.attr == "SPEC"
    assert opts_arch == {}

    cfg_name, resolved_name = config_mod.select_plugin("kimi_k3", None, None)
    assert cfg_name.enabled is True
    assert resolved_name is not None
    assert resolved_name[0].module == "vllm_torchtpu.models.kimi_k3.megakernel"

    spec = testkit.check_plugin_spec("vllm_torchtpu.models.kimi_k3.megakernel:SPEC")
    assert spec.name == "kimi_k3"
    assert spec.load_tensor == ".adapter:load_tensor"


def test_observe_model_loader_streams_into_k3_load_tensor_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify observe_model_loader streams weights into K3 CPU RankWeightStreamer."""
    k3_adapter.clear_streamed_weights()
    k3_adapter.clear_observed_weights()
    monkeypatch.setattr(
        k3_adapter.jax,
        "devices",
        lambda: [SimpleNamespace(id=i) for i in range(32)],
    )
    monkeypatch.setattr(
        k3_adapter.jax,
        "local_devices",
        lambda: [SimpleNamespace(id=0)],
    )
    vllm_cfg = SimpleNamespace(
        additional_config={"megakernel": {"plugin": "kimi_k3", "options": {}}},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])
        ),
    )
    pairs = [
        ("vision_tower.patch_embed.weight", torch.ones((4, 4), dtype=torch.bfloat16)),
        (
            "language_model.model.norm.weight",
            torch.ones((7168,), dtype=torch.bfloat16),
        ),
    ]

    class _Loader:
        """Minimal loader stub yielding checkpoint pairs."""

        def get_all_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
            """Yield raw checkpoint pairs."""
            yield from pairs

    loader = _Loader()
    try:
        with mkr.observe_model_loader(vllm_cfg, loader):
            yielded = list(loader.get_all_weights())
        assert yielded == pairs
        streamer = k3_adapter.STREAM_STORE.get_streamer(0)
        assert streamer.device == "cpu"
        assert "final_norm" in streamer.device_tensors
        assert streamer.device_tensors["final_norm"].device.type == "cpu"
    finally:
        k3_adapter.clear_streamed_weights()
        k3_adapter.clear_observed_weights()
