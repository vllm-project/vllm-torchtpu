# SPDX-License-Identifier: Apache-2.0
"""Unit tests for StableHLO export and save-load round-trip API."""

from __future__ import annotations

import contextlib
import copy
import ctypes
import json
import struct
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest  # pyrefly: ignore[missing-import]
import torch
import torch.fx as fx
from vllm.config import VllmConfig
from vllm.config.utils import Range

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
from vllm_torchtpu.export import stablehlo_export


@pytest.fixture(autouse=True)
def _forbid_default_tpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock default TPU backend on CPU so normal compile continuation succeeds."""
    mock_runnable = MagicMock()
    mock_runnable.return_value = torch.ones((16, 64), dtype=torch.bfloat16)
    mock_backend = MagicMock(return_value=mock_runnable)
    monkeypatch.setattr(tpu_compiler, "_tpu_backend", mock_backend)


def _make_sample_graph_and_inputs(
    batch_size: int = 16,
    hidden_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[fx.GraphModule, list[torch.Tensor]]:
    class SimpleModule(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states + hidden_states

    graph = fx.symbolic_trace(SimpleModule())
    example_input = torch.empty((batch_size, hidden_size), dtype=dtype)
    for node in graph.graph.nodes:
        if node.op == "placeholder" or node.op == "output":
            node.meta["example_value"] = example_input
            node.meta["sharding"] = "PartitionSpec(None, 'model')"
    return graph, [example_input]


def _write_test_safetensors(tensors: dict[str, torch.Tensor], out_path: Path) -> None:
    header: dict[str, Any] = {}
    payloads: list[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        cpu_t = tensor.detach().cpu().contiguous()
        nbytes = cpu_t.numel() * cpu_t.element_size()
        raw = ctypes.string_at(cpu_t.data_ptr(), nbytes) if nbytes > 0 else b""
        end = offset + len(raw)
        header[name] = {
            "dtype": "BF16",
            "shape": [int(d) for d in cpu_t.shape],
            "data_offsets": [offset, end],
        }
        payloads.append(raw)
        offset = end
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    if pad:
        hdr_bytes += b" " * pad
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for chunk in payloads:
            f.write(chunk)


class _LmHeadModel(torch.nn.Module):
    """A model whose logits are `lm_head(hidden_states)`."""

    def __init__(self, hidden_size: int = 16, vocab_size: int = 32) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(
            hidden_size, vocab_size, bias=False, dtype=torch.bfloat16
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


def _mlir_tensor_type(t: torch.Tensor) -> str:
    dims = "x".join(str(d) for d in t.shape)
    elem = stablehlo_export._TORCH_NAME_TO_MLIR_DTYPE[
        str(t.dtype).removeprefix("torch.")
    ]
    return f"tensor<{dims}x{elem}>" if dims else f"tensor<{elem}>"


def _render_main(inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> str:
    args = ", ".join(f"%arg{i}: {_mlir_tensor_type(t)}" for i, t in enumerate(inputs))
    ret = _mlir_tensor_type(output)
    return (
        "module @lowered {\n"
        f"  func.func public @main({args}) -> ({ret}) {{\n"
        f"    %0 = stablehlo.constant dense<0> : {ret}\n"
        f"    return %0 : {ret}\n"
        "  }\n"
        "}\n"
    )


class _FakeTpuLowering:
    """CPU stand-in for `stablehlo_export._lower_with_tpu_backend`.

    Runs the function eagerly and reports the module the TPU backend lowers for
    it: `@main(hidden_states, lm_head.weight, *extra_inputs) -> result`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        extra_inputs: tuple[torch.Tensor, ...] = (),
        mlir_text: str | None = None,
    ) -> None:
        self.lm_head_weight = model.lm_head.weight
        self.extra_inputs = extra_inputs
        self.mlir_text = mlir_text
        self.calls: list[tuple[Any, torch.Tensor, torch.Tensor]] = []

    def __call__(
        self, fn: Any, example: torch.Tensor
    ) -> stablehlo_export.LoweredModule:
        with torch.no_grad():
            out = fn(example)
        self.calls.append((fn, example, out))
        inputs = (example, self.lm_head_weight, *self.extra_inputs)
        return stablehlo_export.LoweredModule(
            mlir_text=self.mlir_text or _render_main(inputs, out),
            inputs=inputs,
            outputs=(out,),
        )


def _patch_tpu_lowering(lowering: _FakeTpuLowering) -> Any:
    return patch.object(stablehlo_export, "_lower_with_tpu_backend", lowering)


def test_env_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", "/tmp/mock_export_hlo")
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", "/tmp/mock_hlo_export")

    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compiler = TpuCompilerAdaptor()
    with pytest.raises(ValueError, match="mutually exclusive"):
        stablehlo_export.maybe_export_or_load_stablehlo(
            compiler, graph, inputs, Range(start=16, end=16), "subgraph_0"
        )


def test_deterministic_key_generation() -> None:
    assert (
        stablehlo_export._compute_fn_key(Range(start=16, end=16), 0)
        == "forward_bucket16"
    )
    assert (
        stablehlo_export._compute_fn_key(Range(start=32, end=32), 0)
        == "forward_bucket32"
    )
    assert (
        stablehlo_export._compute_fn_key(Range(start=16, end=16), 1)
        == "forward_piece1_bucket16"
    )


def test_profile_run_preserves_behavior_and_skips_export_and_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path / "hlo"))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    profile_compiler = TpuCompilerAdaptor()
    profile_cache_dir = tmp_path / "cache" / "profile_cache" / "abcd1234"
    profile_compiler.initialize_cache(cache_dir=str(profile_cache_dir))

    mock_mlir_res = MagicMock()
    mock_mlir_res.serialize_text.return_value = "module {}"
    mock_mlir_res.mlir_result_tensors = [inputs[0]]
    mock_mlir_res.is_noop = False

    with (
        patch(
            "torch_tpu._internal.export.fx_to_mlir", return_value=mock_mlir_res
        ) as mock_fx_to_mlir,
        patch.object(
            profile_compiler,
            "_run_backend",
            return_value=(lambda *a, **kw: inputs[0], None),
        ) as mock_run_backend,
    ):
        compiled_fn, _ = profile_compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert compiled_fn is not None
        mock_fx_to_mlir.assert_not_called()
        mock_run_backend.assert_called_once()
    assert not (tmp_path / "hlo").exists()

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path / "hlo"))
    with patch.object(
        profile_compiler,
        "_run_backend",
        return_value=(lambda *a, **kw: inputs[0], None),
    ) as mock_run_backend_load:
        compiled_fn_load, _ = profile_compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert compiled_fn_load is not None
        mock_run_backend_load.assert_called_once()


def test_export_stablehlo_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    dummy_mlir = (
        "module @exported_stablehlo {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )

    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = dummy_mlir
    mock_exported.mlir_result_tensors = [inputs[0]]
    mock_exported.is_noop = False

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with patch(
        "torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported
    ) as mock_fx_to_mlir:
        compiled_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )

    assert mock_fx_to_mlir.call_count == 1
    assert compiled_fn is not None
    assert handle is None

    expected_fn_key = "forward_bucket16"
    mlir_path = tmp_path / f"{expected_fn_key}.mlir"
    metadata_path = tmp_path / "metadata.json"
    assert mlir_path.is_file()
    assert mlir_path.read_text() == dummy_mlir
    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text())
    assert "functions" in metadata
    entry = metadata["functions"]["forward"]["16"]
    assert entry["file_path"] == f"{expected_fn_key}.mlir"
    assert entry["subgraph_index"] == 0
    assert entry["compile_size"] == 16
    assert len(entry["inputs"]) == 1
    assert entry["inputs"][0] == {
        "arg_index": 0,
        "index": 0,
        "name": "hidden_states",
        "shape": [16, 64],
        "dtype": "bfloat16",
        "sharding": "PartitionSpec(None, 'model')",
        "argument_type": "user_io",
        "buffer_donor": False,
        "is_weight": False,
        "weight_name": "",
    }
    assert len(entry["outputs"]) == 1
    assert entry["outputs"][0]["shape"] == [16, 64]
    assert entry["outputs"][0]["dtype"] == "bfloat16"


def test_load_stablehlo_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    fn_key = "forward_bucket16"
    saved_mlir = (
        "module @custom_stablehlo {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    (tmp_path / f"{fn_key}.mlir").write_text(saved_mlir)
    metadata = {
        "functions": {
            "forward": {
                "16": {
                    "file_path": f"{fn_key}.mlir",
                    "subgraph_index": 0,
                    "compile_size": 16,
                    "inputs": [
                        {
                            "arg_index": 0,
                            "index": 0,
                            "name": "hidden_states",
                            "shape": [16, 64],
                            "dtype": "bfloat16",
                            "sharding": "PartitionSpec(None, 'model')",
                        }
                    ],
                    "outputs": [
                        {
                            "arg_index": 0,
                            "index": 0,
                            "name": "output_0",
                            "shape": [16, 64],
                            "dtype": "bfloat16",
                            "sharding": "PartitionSpec(None, 'model')",
                        }
                    ],
                }
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    mock_mlir_module = MagicMock(name="ContextedModule")
    mock_pjrt_exe = MagicMock(name="PjRtLoadedExecutable")
    expected_output = torch.ones((16, 64), dtype=torch.bfloat16)

    with (
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=mock_mlir_module,
        ) as mock_parse,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=mock_pjrt_exe,
        ) as mock_compile_mlir,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.execute",
            return_value=[expected_output],
        ) as mock_execute,
    ):
        compiled_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert compiled_fn is not None
        assert handle is None
        mock_parse.assert_called_once_with(saved_mlir)
        mock_compile_mlir.assert_called_once_with(mock_mlir_module)

        result = compiled_fn(inputs[0])
        assert mock_execute.call_count == 1
        assert torch.equal(result, expected_output)


def test_load_stablehlo_signature_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    fn_key = "forward_bucket16"
    (tmp_path / f"{fn_key}.mlir").write_text("module {}")
    metadata = {
        "functions": {
            "forward": {
                "16": {
                    "file_path": f"{fn_key}.mlir",
                    "subgraph_index": 0,
                    "compile_size": 16,
                    "inputs": [
                        {
                            "arg_index": 0,
                            "index": 0,
                            "name": "hidden_states",
                            "shape": [32, 64],
                            "dtype": "bfloat16",
                            "sharding": "PartitionSpec(None, 'model')",
                        }
                    ],
                    "outputs": [],
                }
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with pytest.raises(ValueError, match="index 0"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)


def test_load_stablehlo_missing_key_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    (tmp_path / "metadata.json").write_text(json.dumps({}))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError, match="missing from metadata.json"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)


def test_export_and_load_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    exported_mlir_text = (
        "module @initial_export {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = exported_mlir_text
    mock_exported.mlir_result_tensors = [inputs[0]]
    mock_exported.is_noop = False

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    fn_key = "forward_bucket16"
    mlir_file = tmp_path / f"{fn_key}.mlir"
    assert mlir_file.is_file()

    customer_modified_mlir = (
        "module @customer_optimized {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mlir_file.write_text(customer_modified_mlir)

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    mock_mlir_module = MagicMock(name="ContextedModule")
    mock_pjrt_exe = MagicMock(name="PjRtLoadedExecutable")
    expected_output = torch.full((16, 64), 2.0, dtype=torch.bfloat16)

    with (
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=mock_mlir_module,
        ) as mock_parse,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=mock_pjrt_exe,
        ) as mock_compile_mlir,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.execute",
            return_value=[expected_output],
        ),
    ):
        reloaded_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert handle is None
        assert reloaded_fn is not None
        mock_parse.assert_called_once_with(customer_modified_mlir)
        mock_compile_mlir.assert_called_once_with(mock_mlir_module)
        out = reloaded_fn(inputs[0])
        assert torch.equal(out, expected_output)


def test_export_stablehlo_explicit_dir_and_aot_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _WithWeight(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.ones((16, 64), dtype=torch.bfloat16), requires_grad=False
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.weight

    mod = _WithWeight()
    graph = fx.symbolic_trace(mod)
    inputs = [torch.ones((16, 64), dtype=torch.bfloat16)]
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            node.meta["example_value"] = inputs[0]
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_2"

    explicit_export_dir = tmp_path / "custom_export_dir"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(explicit_export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path / "unused_cache_dir"))

    exported_mlir_text = (
        "module @aot_exported {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.is_noop = False
    mock_exported.module = MagicMock(name="MlirModule")
    mock_exported.serialize_text.return_value = exported_mlir_text
    mock_exported.mlir_result_tensors = [inputs[0]]

    def _fake_run_backend(
        g: fx.GraphModule, ex_inputs: list[Any], *, serializable: bool = True
    ) -> tuple[Any, None]:
        from torch_tpu._internal import export as torch_tpu_export

        torch_tpu_export.fx_to_mlir(g, ex_inputs, build_mlir_module=False)
        return (lambda *a: (a[0] * 2,)), None

    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported),
        patch.object(compiler, "_run_backend", side_effect=_fake_run_backend),
    ):
        compiled_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert handle is None
        assert compiled_fn is not None
        assert torch.equal(compiled_fn(inputs[0]), inputs[0] * 2)

    metadata_file = explicit_export_dir / "metadata.json"
    assert metadata_file.is_file()
    metadata = json.loads(metadata_file.read_text())
    assert "forward_piece200" in metadata["functions"]
    entry = metadata["functions"]["forward_piece200"]["16"]
    mlir_file = explicit_export_dir / entry["file_path"]
    assert mlir_file.read_text() == exported_mlir_text

    modified_mlir_text = (
        "module @aot_modified {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mlir_file.write_text(modified_mlir_text)

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(explicit_export_dir))

    mock_parsed_mod = MagicMock(name="ParsedModule")
    mock_pjrt_exe = MagicMock(name="CompiledExe")

    def _fake_orig_fx_to_mlir(
        g: fx.GraphModule, ex_inputs: list[Any], **kwargs: Any
    ) -> Any:
        from torch_tpu._internal.compile import tpu_torch_compile

        res = tpu_torch_compile.traverse_and_compile(argument_layouts=[[1, 0]])
        assert res.module is mock_parsed_mod
        assert res.executable is mock_pjrt_exe
        return mock_exported

    with (
        patch(
            "torch_tpu._internal.export.fx_to_mlir",
            side_effect=_fake_orig_fx_to_mlir,
        ),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=mock_parsed_mod,
        ) as mock_parse,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=mock_pjrt_exe,
        ) as mock_compile_mlir,
        patch.object(compiler, "_run_backend", side_effect=_fake_run_backend),
    ):
        reloaded_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert handle is None
        assert reloaded_fn is not None
        mock_parse.assert_called_once_with(modified_mlir_text)
        mock_compile_mlir.assert_called_once_with(mock_parsed_mod, False, [[1, 0]])
        assert torch.equal(reloaded_fn(inputs[0]), inputs[0] * 2)


def test_load_stablehlo_exact_key_lookup_and_missing_key_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(16, 64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"
    expected_key = "forward_bucket16"

    metadata_other = {
        "functions": {
            "forward": {
                "32": {
                    "file_path": "forward_bucket32.mlir",
                    "subgraph_index": 0,
                    "compile_size": 32,
                    "inputs": [],
                    "outputs": [],
                }
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata_other, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError, match=expected_key):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    metadata_exact = {
        "functions": {
            "forward": {
                "16": {
                    "file_path": f"{expected_key}.mlir",
                    "subgraph_index": 0,
                    "compile_size": 16,
                    "inputs": [
                        {
                            "arg_index": 0,
                            "index": 0,
                            "name": "hidden_states",
                            "shape": [32, 64],
                            "dtype": "bfloat16",
                            "sharding": None,
                        }
                    ],
                    "outputs": [],
                }
            }
        }
    }
    (tmp_path / f"{expected_key}.mlir").write_text("module {}")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata_exact, indent=2))

    with pytest.raises(ValueError, match="Input signature mismatch at index 0"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)


def test_export_and_load_integrate_with_shape_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vllm_torchtpu.compilation import shape_variants

    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    dummy_mlir = (
        "module @shape_variant_test {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = dummy_mlir
    mock_exported.mlir_result_tensors = [inputs[0]]
    mock_exported.is_noop = False

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    with (
        shape_variants.warmup(),
        patch(
            "torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported
        ) as mock_fx_to_mlir,
    ):
        fn_trace1, _ = compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
        fn_trace2, _ = compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
        fn_load_trace3 = compiler.load(
            (cache_key, str(tmp_path / cache_key), True),
            graph,
            inputs,
            0,
            compile_range,
        )
        assert mock_fx_to_mlir.call_count == 1
        assert fn_trace2 is fn_trace1
        assert fn_load_trace3 is fn_trace1

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    mock_mlir_module = MagicMock(name="ContextedModule")
    mock_pjrt_exe = MagicMock(name="PjRtLoadedExecutable")

    with (
        shape_variants.warmup(),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=mock_mlir_module,
        ) as mock_parse,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=mock_pjrt_exe,
        ) as mock_compile_mlir,
    ):
        loaded_trace1, _ = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        loaded_trace2, _ = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        loaded_trace3 = compiler.load(
            (cache_key, str(tmp_path / cache_key), True),
            graph,
            inputs,
            0,
            compile_range,
        )
        assert mock_parse.call_count == 1
        assert mock_compile_mlir.call_count == 1
        assert loaded_trace2 is loaded_trace1
        assert loaded_trace3 is loaded_trace1


def test_load_stablehlo_mlir_main_signature_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    fn_key = "forward_bucket16"
    mlir_path = tmp_path / f"{fn_key}.mlir"
    metadata = {
        "functions": {
            "forward": {
                "16": {
                    "file_path": f"{fn_key}.mlir",
                    "subgraph_index": 0,
                    "compile_size": 16,
                    "inputs": [
                        {
                            "arg_index": 0,
                            "index": 0,
                            "name": "hidden_states",
                            "shape": [16, 64],
                            "dtype": "bfloat16",
                            "sharding": "PartitionSpec(None, 'model')",
                        }
                    ],
                    "outputs": [
                        {
                            "arg_index": 0,
                            "index": 0,
                            "name": "output_0",
                            "shape": [16, 64],
                            "dtype": "bfloat16",
                            "sharding": "PartitionSpec(None, 'model')",
                        }
                    ],
                }
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    mlir_path.write_text(
        "module @m {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x128xbf16> {\n"
        "    return %arg0 : tensor<16x128xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main output signature mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(
        "module @m {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xf32> {\n"
        "    return %arg0 : tensor<16x64xf32>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main output signature mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(
        "module @m {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>)"
        " -> (tensor<16x64xbf16>, tensor<16x64xbf16>) {\n"
        "    return %arg0, %arg0 : tensor<16x64xbf16>, tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main output count mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(
        "module @m {\n"
        "  func.func @main(%arg0: tensor<32x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main input signature mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)


def test_export_and_load_isolate_compilation_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            gpu_memory_utilization=0.9,
            num_gpu_blocks=1024,
            num_gpu_blocks_override=None,
        ),
        model_config=SimpleNamespace(max_model_len=4096),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2048,
            max_num_seqs=16,
        ),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        speculative_config=None,
    )

    mlir_file = tmp_path / "subgraph_idx0_bucket16_abcdef123456.mlir"
    mlir_file.write_text("module @v1 {\n  func.func @main() { return }\n}\n")

    with patch(
        "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
        return_value=[],
    ):
        monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
        monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)
        base_hash = tpu_compiler.compute_tpu_compilation_hash(cast(VllmConfig, cfg))

        monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
        export_hash = tpu_compiler.compute_tpu_compilation_hash(cast(VllmConfig, cfg))
        assert export_hash != base_hash

        monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
        monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))
        load_hash_v1 = tpu_compiler.compute_tpu_compilation_hash(cast(VllmConfig, cfg))
        assert load_hash_v1 != base_hash
        assert load_hash_v1 != export_hash

        mlir_file.write_text("module @v2 {\n  func.func @main() { return }\n}\n")
        load_hash_v2 = tpu_compiler.compute_tpu_compilation_hash(cast(VllmConfig, cfg))
        assert load_hash_v2 != load_hash_v1


def test_load_stablehlo_disables_async_compile_and_preserves_tuple_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from torch_tpu._internal.compile.compiler import StaticCompiler

    class _TupleOutputModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x + x, x * 2

    tuple_graph = fx.symbolic_trace(_TupleOutputModule())
    inp = torch.ones((16, 64), dtype=torch.bfloat16)
    for node in tuple_graph.graph.nodes:
        if node.op == "placeholder":
            node.meta["example_value"] = inp
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    fn_key = "forward_bucket16"
    (tmp_path / f"{fn_key}.mlir").write_text(
        "module @tuple_out {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>)"
        " -> (tensor<16x64xbf16>, tensor<16x64xbf16>) {\n"
        "    return %arg0, %arg0 : tensor<16x64xbf16>, tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    metadata = {
        "functions": {
            "forward": {
                "16": {
                    "file_path": f"{fn_key}.mlir",
                    "subgraph_index": 0,
                    "compile_size": 16,
                    "inputs": [
                        {
                            "arg_index": 0,
                            "index": 0,
                            "name": "x",
                            "shape": [16, 64],
                            "dtype": "bfloat16",
                            "sharding": None,
                        }
                    ],
                    "outputs": [
                        {
                            "arg_index": 0,
                            "index": 0,
                            "name": "output_0",
                            "shape": [16, 64],
                            "dtype": "bfloat16",
                            "sharding": None,
                        },
                        {
                            "arg_index": 1,
                            "index": 1,
                            "name": "output_1",
                            "shape": [16, 64],
                            "dtype": "bfloat16",
                            "sharding": None,
                        },
                    ],
                }
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    out0 = torch.full((16, 64), 2.0, dtype=torch.bfloat16)
    out1 = torch.full((16, 64), 3.0, dtype=torch.bfloat16)
    with (
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=MagicMock(),
        ),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=MagicMock(),
        ),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.execute",
            return_value=[out0, out1],
        ),
    ):
        loaded_fn, _ = compiler.compile(
            tuple_graph, [inp], {}, compile_range, key=cache_key
        )
        assert loaded_fn is not None
        res = loaded_fn(inp)
        assert isinstance(res, tuple)
        assert len(res) == 2

    class _WithAttr(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(inp, requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.w

    aot_graph = fx.symbolic_trace(_WithAttr())
    monkeypatch.setattr(StaticCompiler, "_async_compile", True, raising=False)
    observed_async_flags: list[bool] = []

    def _inspect_async_backend(
        g: fx.GraphModule, ex_inputs: list[Any], *args: Any, **kwargs: Any
    ) -> tuple[Any, None]:
        observed_async_flags.append(
            bool(getattr(StaticCompiler, "_async_compile", False))
        )
        return (lambda *a: (a[0],)), None

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path / "aot_out"))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)
    with patch.object(compiler, "_run_backend", side_effect=_inspect_async_backend):
        compiler.compile(aot_graph, [inp], {}, compile_range, key=cache_key)

    assert observed_async_flags == [False]
    assert getattr(StaticCompiler, "_async_compile", None) is True


def test_contract_safeguards_buffer_donor_hardware_and_custom_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    donor_mlir = (
        "module @kv_donor {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16> {jax.buffer_donor = true})"
        " -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = donor_mlir
    mock_exported.mlir_result_tensors = [inputs[0]]
    mock_exported.is_noop = False

    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5p-8")
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    fn_key = "forward_bucket16"
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    entry = metadata["functions"]["forward"]["16"]
    assert entry["inputs"][0]["buffer_donor"] is True
    assert entry["inputs"][0]["argument_type"] == "state"
    assert entry["target_hardware"]["chip_type"] == "v5p-8"
    assert "libtpu" in entry["target_hardware"]

    mlir_path = tmp_path / f"{fn_key}.mlir"
    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    mlir_path.write_text(
        "module @kv_donor {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main buffer_donor mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(donor_mlir)

    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-8")
    with pytest.raises(ValueError, match="Target hardware mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5p-8")

    drifted_metadata = copy.deepcopy(metadata)
    drifted_entry = drifted_metadata["functions"]["forward"]["16"]
    drifted_entry["target_hardware"]["torch_tpu"] = "0.0.0-different"
    drifted_entry["target_hardware"]["jaxlib"] = "0.0.0-different"
    drifted_entry["target_hardware"]["libtpu"] = "0.0.0-different"
    metadata_path.write_text(json.dumps(drifted_metadata, indent=2))
    with (
        patch.object(stablehlo_export.logger, "warning") as mock_warn,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=MagicMock(),
        ),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=MagicMock(),
        ),
    ):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
        assert mock_warn.call_count >= 1
    metadata_path.write_text(json.dumps(metadata, indent=2))

    mlir_path.write_text(
        "module @kv_donor {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16> {jax.buffer_donor = true})"
        " -> tensor<16x64xbf16> {\n"
        "    %0 = stablehlo.custom_call @__xla_custom_call_unvetted(%arg0)"
        " : (tensor<16x64xbf16>) -> tensor<16x64xbf16>\n"
        "    return %0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="Disallowed stablehlo.custom_call target"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)


def test_strict_main_signature_and_single_read_invariants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    valid_mlir = (
        "module @valid_hlo {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16, #stablehlo.bounds<?, ?>>)"
        " -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = valid_mlir
    mock_exported.mlir_result_tensors = [inputs[0]]
    mock_exported.is_noop = False

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{corrupt_json")
    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported),
        pytest.raises(ValueError, match="Corrupt metadata.json"),
    ):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
    metadata_path.unlink()

    with patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    fn_key = "forward_bucket16"
    mlir_path = tmp_path / f"{fn_key}.mlir"

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    mlir_path.write_text("module @empty_main {\n  func.func @main() { return }\n}\n")
    with pytest.raises(ValueError, match="MLIR @main input count mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(
        "module @missing_main {\n"
        "  func.func @other(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main function signature missing"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    mlir_path.write_text(valid_mlir)
    orig_read_text = Path.read_text
    mlir_reads: list[Path] = []

    def _spy_read_text(self_path: Path, *a: Any, **kw: Any) -> str:
        if self_path == mlir_path:
            mlir_reads.append(self_path)
        return orig_read_text(self_path, *a, **kw)

    with (
        patch.object(Path, "read_text", autospec=True, side_effect=_spy_read_text),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=MagicMock(),
        ),
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=MagicMock(),
        ),
    ):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    assert len(mlir_reads) == 1


def test_export_early_exit_skips_backend_compile_and_exits_after_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vllm_torchtpu import envs

    assert "VLLM_TORCHTPU_EXPORT_EARLY_EXIT" not in envs.environment_variables
    assert "VLLM_TORCHTPU_EXPORT_HLO_DIR" in envs.environment_variables

    graph16, inputs16 = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    dummy_mlir_16 = (
        "module @early_exit_16 {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exp = MagicMock()
    mock_exp.serialize_text.return_value = dummy_mlir_16
    mock_exp.mlir_result_tensors = [inputs16[0]]
    mock_exp.is_noop = False
    mock_exp.module = MagicMock()

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exp),
        patch.object(compiler, "_run_backend") as mock_run_backend,
    ):
        fn16, handle16 = compiler.compile(
            graph16,
            inputs16,
            {},
            Range(start=16, end=16),
            key="artifact_compile_range_16_16_subgraph_0",
        )
        assert handle16 is None
        assert fn16 is not None
        out16 = fn16(inputs16[0])
        assert out16.shape == inputs16[0].shape
        assert out16.dtype == inputs16[0].dtype
        assert mock_run_backend.call_count == 0

    assert (tmp_path / "forward_bucket16.mlir").is_file()


def test_extract_input_specs_filters_non_tensor_symint_placeholders() -> None:
    class DynamicModule(torch.nn.Module):
        def forward(self, batch_size: int, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states + hidden_states

    graph = fx.symbolic_trace(DynamicModule())
    tensor_input = torch.empty((16, 64), dtype=torch.bfloat16)

    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    assert len(placeholders) == 2
    placeholders[0].meta["val"] = 16
    placeholders[1].meta["val"] = tensor_input
    placeholders[1].meta["sharding"] = "PartitionSpec(None, 'model')"

    specs = stablehlo_export._extract_input_specs(graph, [16, tensor_input])
    assert len(specs) == 1
    assert specs[0]["arg_index"] == 0
    assert specs[0]["index"] == 0
    assert specs[0]["name"] == "hidden_states"
    assert specs[0]["shape"] == [16, 64]
    assert specs[0]["dtype"] == "bfloat16"


def test_extract_input_specs_annotates_is_weight_and_weight_name() -> None:
    class ModuleWithLiftedWeights(torch.nn.Module):
        def forward(
            self, hidden_states: torch.Tensor, p_q_proj_weight: torch.Tensor
        ) -> torch.Tensor:
            return torch.matmul(hidden_states, p_q_proj_weight)

    model = ModuleWithLiftedWeights()
    graph = fx.symbolic_trace(model)
    tensor_input = torch.empty((16, 64), dtype=torch.bfloat16)
    weight_tensor = torch.nn.Parameter(torch.empty((64, 64), dtype=torch.bfloat16))

    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    placeholders[0].meta["val"] = tensor_input
    placeholders[1].meta["val"] = weight_tensor
    placeholders[1].meta["param_name"] = "model.layers.0.self_attn.q_proj.weight"

    specs = stablehlo_export._extract_input_specs(graph, [tensor_input, weight_tensor])
    assert len(specs) == 2
    assert specs[0]["is_weight"] is False
    assert specs[1]["is_weight"] is True
    assert specs[1]["weight_name"] == "model.layers.0.self_attn.q_proj.weight"


def test_extract_input_specs_annotates_is_weight_with_aot_desc() -> None:
    class ModuleWithAot(torch.nn.Module):
        def forward(self, arg0: torch.Tensor, arg1: torch.Tensor) -> torch.Tensor:
            return arg0 + arg1

    model = ModuleWithAot()
    graph = fx.symbolic_trace(model)
    tensor_input = torch.empty((16, 64), dtype=torch.bfloat16)
    weight_tensor = torch.empty((64, 64), dtype=torch.bfloat16)

    class DummyDesc:
        def __init__(self, target: str, is_param: bool):
            self.target = target
            self.is_param = is_param
            self.is_buffer = False

    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    placeholders[0].meta["val"] = tensor_input
    placeholders[0].meta["desc"] = DummyDesc("", False)
    placeholders[1].meta["val"] = weight_tensor
    placeholders[1].meta["desc"] = DummyDesc("model.embed_tokens.weight", True)

    specs = stablehlo_export._extract_input_specs(graph, [tensor_input, weight_tensor])
    assert len(specs) == 2
    assert specs[0]["is_weight"] is False
    assert specs[1]["is_weight"] is True
    assert specs[1]["weight_name"] == "model.embed_tokens.weight"


def test_export_and_load_serving_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _WeightedGraph(torch.nn.Module):
        def forward(
            self, hidden_states: torch.Tensor, p_embed_weight: torch.Tensor
        ) -> torch.Tensor:
            return hidden_states + p_embed_weight

    graph = fx.symbolic_trace(_WeightedGraph())
    inp_x = torch.empty((16, 64), dtype=torch.bfloat16)
    inp_w = torch.nn.Parameter(torch.empty((16, 64), dtype=torch.bfloat16))
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    placeholders[0].meta["example_value"] = inp_x
    placeholders[0].meta["sharding"] = "PartitionSpec(None, 'model')"
    placeholders[1].meta["example_value"] = inp_w
    placeholders[1].meta["param_name"] = "model.embed_tokens.weight"
    for node in graph.graph.nodes:
        if node.op == "output":
            node.meta["example_value"] = inp_x

    inputs = [inp_x, inp_w]
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    exported_mlir = (
        "module @mlir_test {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16> {jax.buffer_donor = true},"
        " %arg1: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    mock_exported = MagicMock()
    mock_exported.serialize_text.return_value = exported_mlir
    mock_exported.mlir_result_tensors = [inp_x]
    mock_exported.is_noop = False

    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5p-8")
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    assert (tmp_path / "forward_bucket16.mlir").is_file()
    assert (tmp_path / "abstract_params.mlir").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert not (tmp_path / "weight.json").exists()
    serving_ckpt_dir = tmp_path / "serving_checkpoint"
    assert (serving_ckpt_dir / "weight.json").is_file()
    wj_data = json.loads((serving_ckpt_dir / "weight.json").read_text())
    assert "model.embed_tokens.weight" in wj_data
    assert "sources" in wj_data["model.embed_tokens.weight"]
    meta_data = json.loads((tmp_path / "metadata.json").read_text())
    assert "default" in meta_data["functions"]["abstract_params"]

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    mock_mlir_module = MagicMock(name="ContextedModule")
    mock_pjrt_exe = MagicMock(name="PjRtLoadedExecutable")
    expected_out = torch.full((16, 64), 3.0, dtype=torch.bfloat16)

    with (
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.parse_mlir_text",
            return_value=mock_mlir_module,
        ) as mock_parse,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.compile_mlir",
            return_value=mock_pjrt_exe,
        ) as mock_compile_mlir,
        patch(
            "torch_tpu._internal.compile.tpu_torch_compile.execute",
            return_value=[expected_out],
        ),
    ):
        reloaded_fn, handle = compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert handle is None
        assert reloaded_fn is not None
        mock_parse.assert_called_once_with(exported_mlir)
        mock_compile_mlir.assert_called_once_with(mock_mlir_module)
        res = reloaded_fn(inp_x, inp_w)
        assert torch.equal(res, expected_out)


def test_export_sampling_subgraphs_e2e(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    export_dir = tmp_path / "export_subgraphs"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    class MockRunner:
        def __init__(self) -> None:
            self.model = _LmHeadModel()
            self.num_reqs_paddings = [1, 2]
            self.num_tokens_paddings = [2]
            self.model_config = MagicMock()
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    runner = MockRunner()
    with (
        _patch_tpu_lowering(_FakeTpuLowering(runner.model)),
        pytest.raises(SystemExit) as exc_info,
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(runner)
    assert exc_info.value.code == 0

    assert (export_dir / "compute_logits_req1.mlir").exists()
    assert (export_dir / "compute_logits_req2.mlir").exists()
    assert (export_dir / "sample_top_tokens_req1.mlir").exists()
    assert (export_dir / "sample_top_tokens_req2.mlir").exists()
    assert (export_dir / "abstract_params.mlir").exists()
    assert (export_dir / "metadata.json").exists()


def test_export_sampling_subgraphs_bundles_checkpoint_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ckpt_dir = tmp_path / "hf_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "tokenizer.json").write_text('{"version": "1.0"}')
    (ckpt_dir / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrained"}')
    (ckpt_dir / "config.json").write_text(
        '{"model_type": "qwen3", "rope_theta": 1000000.0}'
    )
    _write_test_safetensors(
        {"model.embed_tokens.weight": torch.ones((32, 16), dtype=torch.bfloat16)},
        ckpt_dir / "model.safetensors",
    )

    export_dir = tmp_path / "export_bundled"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    (export_dir / "serving_checkpoint").mkdir(parents=True, exist_ok=True)
    seed_weights = {
        "model.embed_tokens.weight": {
            "index": 1,
            "shape": [32, 16],
            "dtype": "bfloat16",
            "sharding": None,
            "sources": [
                {
                    "path": "",
                    "source_tensor": "model.embed_tokens.weight",
                    "source_shape": [32, 16],
                }
            ],
        },
    }
    (export_dir / "serving_checkpoint" / "weight.json").write_text(
        json.dumps(seed_weights)
    )

    class MockRunner:
        def __init__(self) -> None:
            self.model = _LmHeadModel()
            self.num_reqs_paddings = [1]
            self.num_tokens_paddings = [1]
            self.model_config = MagicMock()
            self.model_config.model = str(ckpt_dir)
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    runner = MockRunner()
    with (
        _patch_tpu_lowering(_FakeTpuLowering(runner.model)),
        pytest.raises(SystemExit) as exc_info,
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(runner)
    assert exc_info.value.code == 0

    assert not (export_dir / "serving_checkpoint" / "constants.safetensors").exists()
    assert not (export_dir / "weight.json").exists()
    wj = json.loads((export_dir / "serving_checkpoint" / "weight.json").read_text())
    assert "rotary_emb_inv_freq" not in wj
    assert wj["model.embed_tokens.weight"]["sources"][0]["path"] == str(
        (ckpt_dir / "model.safetensors").resolve()
    )
    assert (
        wj["lm_head.weight"]["sources"][0]["source_tensor"]
        == "model.embed_tokens.weight"
    )
    assert (export_dir / "abstract_params.mlir").is_file()
    meta = json.loads((export_dir / "metadata.json").read_text())
    abs_spec = meta["functions"]["abstract_params"]["default"]
    assert abs_spec["file_path"] == "abstract_params.mlir"
    assert [inp["name"] for inp in abs_spec["inputs"]] == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    # The bundled tokenizer is referenced relative to the artifact directory.
    assert (export_dir / "tokenizer.json").is_file()
    assert meta["serving_config"]["tokenizer_config"] == {
        "source": {"TokenizerPath": {"path": "tokenizer.json"}}
    }


def test_export_sampling_subgraphs_gives_lm_head_its_own_weight_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `lm_head.weight` is arg 1 of compute_logits and `model.embed_tokens.weight`
    # is arg 1 of forward_bucket. weight.json `index` is the global HBM weight
    # slot, so the two tensors must not share it.
    export_dir = tmp_path / "export_lm_head_index"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)
    (export_dir / "serving_checkpoint").mkdir(parents=True, exist_ok=True)
    seed_weights = {
        name: {
            "index": idx,
            "shape": [32, 16],
            "dtype": "bfloat16",
            "sharding": None,
            "sources": [{"path": "", "source_tensor": name, "source_shape": [32, 16]}],
        }
        for name, idx in (("model.embed_tokens.weight", 1), ("model.norm.weight", 9))
    }
    (export_dir / "serving_checkpoint" / "weight.json").write_text(
        json.dumps(seed_weights)
    )

    class MockRunner:
        def __init__(self) -> None:
            self.model = _LmHeadModel()
            self.num_reqs_paddings = [1]
            self.num_tokens_paddings = [1]
            self.model_config = MagicMock()
            self.model_config.model = str(tmp_path / "no_such_ckpt")
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    runner = MockRunner()
    with (
        _patch_tpu_lowering(_FakeTpuLowering(runner.model)),
        pytest.raises(SystemExit),
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(runner)

    wj = json.loads((export_dir / "serving_checkpoint" / "weight.json").read_text())
    lm_head_index = wj["lm_head.weight"]["index"]
    assert lm_head_index not in {1, 9}
    metadata = json.loads((export_dir / "metadata.json").read_text())
    for fn_name in ("compute_logits", "sample_top_tokens"):
        entry_in = metadata["functions"][fn_name]["1"]["inputs"][1]
        assert entry_in["arg_index"] == 1
        assert entry_in["index"] == lm_head_index
        assert entry_in["argument_type"] == "weight"
    # No checkpoint snapshot, so no tokenizer.json is bundled or referenced.
    assert not (export_dir / "tokenizer.json").exists()
    assert "tokenizer_config" not in metadata.get("serving_config", {})


@pytest.mark.parametrize(
    ("tp", "dp", "expected"),
    [
        # torch_tpu compiles each TP rank's module with num_partitions=TP,
        # replica_count=DP and use_spmd_partitioning=true.
        (1, 1, {"num_replicas": 1, "num_partitions": 1, "use_spmd_partitioning": True}),
        (4, 1, {"num_replicas": 1, "num_partitions": 4, "use_spmd_partitioning": True}),
        (1, 8, {"num_replicas": 8, "num_partitions": 1, "use_spmd_partitioning": True}),
        (2, 4, {"num_replicas": 4, "num_partitions": 2, "use_spmd_partitioning": True}),
    ],
)
def test_export_sampling_subgraphs_writes_compile_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tp: int, dp: int, expected: dict
) -> None:
    export_dir = tmp_path / f"export_compile_options_tp{tp}_dp{dp}"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    class MockRunner:
        def __init__(self) -> None:
            self.model = _LmHeadModel()
            self.num_reqs_paddings = [1]
            self.num_tokens_paddings = [1]
            self.model_config = MagicMock()
            self.model_config.model = str(tmp_path / "no_such_ckpt")
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self.parallel_config = MagicMock()
            self.parallel_config.tensor_parallel_size = tp
            self.parallel_config.data_parallel_size = dp
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.world_size = tp * dp
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    runner = MockRunner()
    cm = pytest.raises(SystemExit) if (tp * dp <= 1) else contextlib.nullcontext()
    with _patch_tpu_lowering(_FakeTpuLowering(runner.model)), cm:
        stablehlo_export.maybe_export_sampling_subgraphs(runner)

    metadata = json.loads((export_dir / "metadata.json").read_text())
    assert metadata["compile_options"] == expected
    assert metadata["config"]["tensor_parallel_size"] == tp
    assert metadata["config"]["data_parallel_size"] == dp


def test_export_sampling_subgraphs_fails_when_metadata_cannot_be_finalized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An invalid tensor_parallel_size must fail the export instead of leaving a
    # metadata.json without config/compile_options behind.
    export_dir = tmp_path / "export_bad_parallel_config"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    class MockRunner:
        def __init__(self) -> None:
            self.model = _LmHeadModel()
            self.num_reqs_paddings = [1]
            self.num_tokens_paddings = [1]
            self.model_config = MagicMock()
            self.model_config.model = str(tmp_path / "no_such_ckpt")
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self.parallel_config = MagicMock()
            self.parallel_config.tensor_parallel_size = "four"
            self.parallel_config.pipeline_parallel_size = 1
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    runner = MockRunner()
    with (
        _patch_tpu_lowering(_FakeTpuLowering(runner.model)),
        pytest.raises(ValueError),
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(runner)


def test_export_early_exit_cleanly_handles_engine_startup_and_multiproc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    export_dir = tmp_path / "export_mp"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "metadata.json").write_text("{}")

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    class MockTPRunner:
        def __init__(self) -> None:
            self.model = _LmHeadModel()
            self.num_reqs_paddings = [1]
            self.num_tokens_paddings = [1]
            self.model_config = MagicMock()
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self.parallel_config = MagicMock()
            self.parallel_config.world_size = 2
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    tp_runner = MockTPRunner()
    with _patch_tpu_lowering(_FakeTpuLowering(tp_runner.model)):
        stablehlo_export.maybe_export_sampling_subgraphs(tp_runner)

    import vllm.v1.engine.utils as engine_utils

    stablehlo_export._install_warmup_early_exit_hook()

    def _fake_startup(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "Engine core initialization failed. See root cause above. "
            "Failed core proc(s): {'EngineCore_DP0': 0}"
        )

    with (
        patch.object(
            engine_utils,
            "_orig_wait_for_engine_startup",
            _fake_startup,
            create=True,
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        engine_utils.wait_for_engine_startup(None, [], None, False, None, None)
    assert exc_info.value.code == 0

    # DP>1 with TP=1 (world_size=1, data_parallel_size=8) must NOT exit inside a
    # single worker before peers finish, and normal startup completion must
    # close launch resources and exit 0.
    class MockDPRunner(MockTPRunner):
        def __init__(self) -> None:
            super().__init__()
            self.parallel_config.world_size = 1
            self.parallel_config.data_parallel_size = 8

    dp_runner = MockDPRunner()
    with _patch_tpu_lowering(_FakeTpuLowering(dp_runner.model)):
        stablehlo_export.maybe_export_sampling_subgraphs(dp_runner)

    mock_launch = MagicMock()
    with (
        patch.object(
            engine_utils,
            "_orig_wait_for_engine_startup",
            lambda *_a, **_kw: None,
            create=True,
        ),
        pytest.raises(SystemExit) as clean_exc_info,
    ):
        engine_utils.wait_for_engine_startup(None, [], None, False, None, mock_launch)
    assert clean_exc_info.value.code == 0
    mock_launch.engine_manager.close.assert_called_once()
    mock_launch.coordinator.close.assert_called_once()


def test_load_hlo_dir_pins_num_gpu_blocks_override_from_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    load_dir = tmp_path / "loaded_hlo"
    load_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "functions": {
            "forward": {
                "16": {
                    "inputs": [
                        {
                            "arg_index": 8,
                            "index": 8,
                            "name": "kv_cache_layer_0",
                            "shape": [50336, 16, 8, 2, 128],
                            "dtype": "bfloat16",
                            "buffer_donor": True,
                        }
                    ]
                }
            }
        }
    }
    (load_dir / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(load_dir))

    vllm_cfg = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None)
    )
    stablehlo_export._install_warmup_early_exit_hook(vllm_cfg)
    assert vllm_cfg.cache_config.num_gpu_blocks_override == 50336


def test_generic_dynamo_placeholder_and_packed_module_reflection() -> None:
    class _FakeFusedProj(torch.nn.Module):
        def __init__(self, in_dim: int, out_sizes: list[int]) -> None:
            super().__init__()
            self.input_size = in_dim
            self.output_sizes = out_sizes
            self.tp_size = 1
            self.weight = torch.nn.Parameter(
                torch.empty((in_dim, sum(out_sizes)), dtype=torch.bfloat16)
            )

    class _FakeLinear(torch.nn.Module):
        def __init__(self, out_dim: int, in_dim: int) -> None:
            super().__init__()
            self.input_size = in_dim
            self.output_size = out_dim
            self.weight = torch.nn.Parameter(
                torch.empty((in_dim, out_dim), dtype=torch.bfloat16)
            )

    class _FakeModel(torch.nn.Module):
        packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }

        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(
                151936, 1024, dtype=torch.bfloat16
            )
            self.model.embed_tokens.output_size_per_partition = 1024
            layer0 = torch.nn.Module()
            layer0.self_attn = torch.nn.Module()
            layer0.self_attn.qkv_proj = _FakeFusedProj(1024, [2048, 1024, 1024])
            layer0.self_attn.o_proj = _FakeLinear(1024, 2048)
            layer0.self_attn.rotary_emb = torch.nn.Module()
            layer0.self_attn.rotary_emb.register_buffer(
                "cos_sin_cache", torch.empty((40960, 128), dtype=torch.bfloat16)
            )
            layer0.mlp = torch.nn.Module()
            layer0.mlp.gate_up_proj = _FakeFusedProj(1024, [3072, 3072])
            self.model.layers = torch.nn.ModuleList([layer0])

    fake_model = _FakeModel()
    stablehlo_export.register_active_model(fake_model)

    outer_names_and_tensors: list[tuple[str, Any]] = [
        ("l_input_ids_", torch.empty((16,), dtype=torch.int32)),
        ("s72", 16),
        (
            "l_self_modules_embed_tokens_parameters_weight_",
            fake_model.model.embed_tokens.weight,
        ),
        (
            "l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_",
            fake_model.model.layers[0].self_attn.qkv_proj.weight,
        ),
        ("l_positions_", torch.empty((16,), dtype=torch.int32)),
        (
            "l_self_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_",
            fake_model.model.layers[0].self_attn.rotary_emb.cos_sin_cache,
        ),
        (
            "l_self_modules_layers_modules_0_modules_self_attn_modules_attn_kv_cache",
            torch.empty((512, 128, 8, 2, 128), dtype=torch.bfloat16),
        ),
        (
            "g_import_vllm_dot_forward_context_forward_context_attn_metadata_model_layers_0_self_attn_attn_seq_lens",
            torch.empty((16,), dtype=torch.int32),
        ),
        (
            "l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_",
            fake_model.model.layers[0].self_attn.o_proj.weight,
        ),
        (
            "l_self_modules_layers_modules_0_modules_mlp_modules_gate_up_proj_parameters_weight_",
            fake_model.model.layers[0].mlp.gate_up_proj.weight,
        ),
    ]

    outer_g = fx.Graph()
    inner_g = fx.Graph()
    inputs: list[Any] = []
    for idx, (ph_name, val) in enumerate(outer_names_and_tensors):
        outer_g.placeholder(ph_name)
        inner_g.placeholder(f"arg{idx}_1")
        inputs.append(val)
    outer_g.output(())
    inner_g.output(())
    outer_gm = fx.GraphModule(torch.nn.Module(), outer_g)
    inner_gm = fx.GraphModule(torch.nn.Module(), inner_g)

    specs = stablehlo_export._extract_input_specs(
        inner_gm, inputs, outer_graph=outer_gm, outer_inputs=inputs
    )
    assert len(specs) == 9
    assert specs[0]["name"] == "input_ids"
    assert specs[0]["argument_type"] == "user_io"

    assert specs[1]["weight_name"] == "model.embed_tokens.weight"
    assert specs[1]["sources"] == [
        {
            "path": "",
            "source_tensor": "model.embed_tokens.weight",
            "source_shape": [151936, 1024],
        }
    ]

    assert specs[2]["weight_name"] == "model.layers.0.self_attn.qkv_proj.weight"
    assert len(specs[2]["sources"]) == 3
    assert (
        specs[2]["sources"][0]["source_tensor"]
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert specs[2]["sources"][0]["transpose"] == [1, 0]
    assert specs[2]["sources"][0]["dst_slices"] == [
        {"dim": 1, "start": 0, "length": 2048}
    ]

    assert specs[4]["name"] == "rotary_emb_inv_freq"
    assert specs[4]["is_weight"] is False
    assert specs[4]["argument_type"] == "constant"
    assert specs[4].get("is_internal_constant") is True

    assert specs[5]["name"] == "kv_cache_layer_0"
    assert specs[5]["argument_type"] == "state"

    assert specs[6]["name"] == "kv_lens"
    assert specs[6]["argument_type"] == "user_io"

    assert specs[7]["weight_name"] == "model.layers.0.self_attn.o_proj.weight"
    assert specs[7]["sources"][0]["transpose"] == [1, 0]
    assert specs[7]["sources"][0]["source_shape"] == [1024, 2048]

    assert specs[8]["weight_name"] == "model.layers.0.mlp.gate_up_proj.weight"
    assert len(specs[8]["sources"]) == 2
    stablehlo_export.register_active_model(None)


def test_multi_rank_writer_guard_and_mtp_draft_collision_free_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    export_dir = tmp_path / "export_mtp_and_buckets"
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    # 1. _split_fn_key and _compute_fn_key for mtp_draft
    assert stablehlo_export._split_fn_key("mtp_draft_bucket16") == ("mtp_draft", "16")
    draft_adaptor = TpuCompilerAdaptor()
    draft_adaptor.cache_dir = "/tmp/cache/eagle_head"
    assert (
        stablehlo_export._compute_fn_key(Range(16, 16), 0, draft_adaptor)
        == "mtp_draft_bucket16"
    )

    # 2. Collision-free global weight indices across forward and mtp_draft
    fwd_mlir = (
        "module @fwd { func.func public @main("
        "%arg0: tensor<16xi32>, %arg1: tensor<16x8xbf16>, %arg2: tensor<8x8xbf16>"
        ") -> (tensor<16x8xbf16>) { return %arg1 : tensor<16x8xbf16> } }"
    )
    stablehlo_export._write_exported_module(
        export_dir=export_dir,
        fn_key="forward_bucket16",
        subgraph_idx=0,
        compile_size=16,
        mlir_text=fwd_mlir,
        input_specs=[
            {
                "arg_index": 0,
                "index": 0,
                "name": "input_ids",
                "shape": [16],
                "dtype": "int32",
                "argument_type": "user_io",
                "is_weight": False,
            },
            {
                "arg_index": 1,
                "index": 1,
                "name": "model.embed_tokens.weight",
                "shape": [16, 8],
                "dtype": "bfloat16",
                "argument_type": "weight",
                "is_weight": True,
                "weight_name": "model.embed_tokens.weight",
            },
            {
                "arg_index": 2,
                "index": 2,
                "name": "model.layers.0.self_attn.o_proj.weight",
                "shape": [8, 8],
                "dtype": "bfloat16",
                "argument_type": "weight",
                "is_weight": True,
                "weight_name": "model.layers.0.self_attn.o_proj.weight",
            },
        ],
        output_specs=[
            {
                "arg_index": 0,
                "index": 0,
                "name": "hidden_states",
                "shape": [16, 8],
                "dtype": "bfloat16",
                "argument_type": "user_io",
            }
        ],
    )

    # Now export mtp_draft_bucket16 where arg_index=1 is shared (embed_tokens)
    # and arg_index=2 is a new MTP weight (mtp.fc.weight) whose arg_index=2
    # collides with forward_bucket16's o_proj.weight index=2!
    draft_mlir = (
        "module @draft { func.func public @main("
        "%arg0: tensor<16xi32>, %arg1: tensor<16x8xbf16>, %arg2: tensor<16x8xbf16>"
        ") -> (tensor<16x1xi32>) { "
        "%c = stablehlo.constant dense<0> : tensor<16x1xi32> "
        "return %c : tensor<16x1xi32> } }"
    )
    stablehlo_export._write_exported_module(
        export_dir=export_dir,
        fn_key="mtp_draft_bucket16",
        subgraph_idx=0,
        compile_size=16,
        mlir_text=draft_mlir,
        input_specs=[
            {
                "arg_index": 0,
                "index": 0,
                "name": "input_ids",
                "shape": [16],
                "dtype": "int32",
                "argument_type": "user_io",
                "is_weight": False,
            },
            {
                "arg_index": 1,
                "index": 1,
                "name": "model.embed_tokens.weight",
                "shape": [16, 8],
                "dtype": "bfloat16",
                "argument_type": "weight",
                "is_weight": True,
                "weight_name": "model.embed_tokens.weight",
            },
            {
                "arg_index": 2,
                "index": 2,
                "name": "model.mtp.fc.weight",
                "shape": [16, 8],
                "dtype": "bfloat16",
                "argument_type": "weight",
                "is_weight": True,
                "weight_name": "model.mtp.fc.weight",
            },
        ],
        output_specs=[
            {
                "arg_index": 0,
                "index": 0,
                "name": "draft_tokens",
                "shape": [16, 1],
                "dtype": "int32",
                "argument_type": "user_io",
            }
        ],
    )

    wj = json.loads((export_dir / "serving_checkpoint" / "weight.json").read_text())
    assert wj["model.embed_tokens.weight"]["index"] == 1
    assert wj["model.layers.0.self_attn.o_proj.weight"]["index"] == 2
    assert wj["model.mtp.fc.weight"]["index"] == 3

    meta = json.loads((export_dir / "metadata.json").read_text())
    draft_inputs = meta["functions"]["mtp_draft"]["16"]["inputs"]
    assert draft_inputs[1]["arg_index"] == 1
    assert draft_inputs[1]["index"] == 1
    assert draft_inputs[2]["arg_index"] == 2
    assert draft_inputs[2]["index"] == 3

    # 3. Non-writer rank (RANK=1) lowers the same modules (tensor-parallel
    # collectives need every rank) but must not write or exit.
    class MockRunner:
        def __init__(self) -> None:
            self.model = _LmHeadModel(hidden_size=8, vocab_size=32)
            # num_reqs_paddings only has [1, 8], while forward_bucket16 has bucket 16
            self.num_reqs_paddings = [1, 8]
            self.num_tokens_paddings = [16]
            self.model_config = MagicMock()
            self.model_config.get_hidden_size.return_value = 8
            self.model_config.vocab_size = 32
            self.model_config.max_model_len = 2048
            self.model_config.eos_token_id = 2
            self.parallel_config = MagicMock()
            self.parallel_config.world_size = 2
            self.parallel_config.tensor_parallel_size = 2
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.data_parallel_size = 4
            self.cache_config = MagicMock()
            self.cache_config.block_size = 16
            self.cache_config.num_gpu_blocks = 256
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    monkeypatch.setenv("RANK", "1")
    rank1_runner = MockRunner()
    rank1_lowering = _FakeTpuLowering(rank1_runner.model)
    with _patch_tpu_lowering(rank1_lowering):
        stablehlo_export.maybe_export_sampling_subgraphs(rank1_runner)
    assert [int(example.shape[0]) for _, example, _ in rank1_lowering.calls] == [
        1,
        1,
        8,
        8,
        16,
        16,
    ]
    meta_after_rank1 = json.loads((export_dir / "metadata.json").read_text())
    assert "sample_top_tokens" not in meta_after_rank1["functions"]

    # 4. Writer rank (RANK=0) exports sampling subgraphs for union of num_reqs_paddings
    # and num_tokens_paddings ([1, 8, 16]) and records data_parallel_size=4
    monkeypatch.setenv("RANK", "0")
    rank0_runner = MockRunner()
    with _patch_tpu_lowering(_FakeTpuLowering(rank0_runner.model)):
        stablehlo_export.maybe_export_sampling_subgraphs(rank0_runner)

    meta_final = json.loads((export_dir / "metadata.json").read_text())
    assert set(meta_final["functions"]["sample_top_tokens"].keys()) == {"1", "8", "16"}
    assert set(meta_final["functions"]["compute_logits"].keys()) == {"1", "8", "16"}
    assert meta_final["config"]["data_parallel_size"] == 4


def test_aot_reordered_arg_mapping_uppercase_dynamo_names_and_dummy_snapshot_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify AOT `arg{K}_1` outer mapping, uppercase Dynamo names, and snapshots."""
    export_dir = tmp_path / "export_qwen_aot"
    monkeypatch.setattr(
        stablehlo_export.envs, "VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir)
    )
    monkeypatch.setattr(stablehlo_export.envs, "VLLM_TORCHTPU_LOAD_HLO_DIR", "")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    class _FakeSymInt:
        pass

    _FakeSymInt.__name__ = "SymInt"

    # Outer Dynamo FX graph with SymInt at index 1 and uppercase L_/G_ names
    outer_g = fx.Graph()
    outer_names = [
        "L_input_ids_",
        "s72",
        "L_self_modules_embed_tokens_parameters_weight_",
        "L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_",
        "L_positions_",
        "L_self_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_",
        "L_self_modules_layers_modules_0_modules_self_attn_modules_attn_kv_cache",
        "G_import_vllm_dot_forward_context_forward_context_attn_metadata_model_layers_0_self_attn_attn_seq_lens",
        "G_import_vllm_dot_forward_context_forward_context_attn_metadata_model_layers_0_self_attn_attn_block_tables",
        "G_import_vllm_dot_forward_context_forward_context_attn_metadata_model_layers_0_self_attn_attn_query_start_loc",
        "G_import_vllm_dot_forward_context_forward_context_attn_metadata_model_layers_0_self_attn_attn_request_distribution",
    ]
    outer_phs = [outer_g.placeholder(n) for n in outer_names]
    outer_g.output((outer_phs[6], outer_phs[0]))
    outer_gm = fx.GraphModule(torch.nn.Module(), outer_g)
    for ph, n in zip(outer_gm.graph.nodes, outer_names):
        if ph.op == "placeholder":
            ph.name = n
            ph.target = n

    outer_inputs: list[Any] = [
        torch.zeros((16,), dtype=torch.int32),
        _FakeSymInt(),
        torch.zeros((32, 8), dtype=torch.bfloat16),
        torch.zeros((8,), dtype=torch.bfloat16),
        torch.zeros((16,), dtype=torch.int32),
        torch.zeros((64, 4), dtype=torch.bfloat16),
        torch.zeros((128, 16, 2, 2, 4), dtype=torch.bfloat16),
        torch.zeros((8,), dtype=torch.int32),
        torch.zeros((128,), dtype=torch.int32),
        torch.zeros((9,), dtype=torch.int32),
        torch.zeros((3,), dtype=torch.int32),
    ]

    # Inner AOT graph skips arg1_1 and reorders arg5_1 before arg4_1
    inner_g = fx.Graph()
    inner_order = [0, 2, 3, 5, 4, 6, 7, 8, 9, 10]
    inner_phs = [inner_g.placeholder(f"arg{k}_1") for k in inner_order]
    inner_g.output((inner_phs[5], inner_phs[0]))
    inner_gm = fx.GraphModule(torch.nn.Module(), inner_g)
    inner_args = [outer_inputs[k] for k in inner_order]

    specs = stablehlo_export._extract_input_specs(
        inner_gm, inner_args, outer_graph=outer_gm, outer_inputs=outer_inputs
    )
    assert len(specs) == 10
    assert specs[0]["name"] == "input_ids"
    assert specs[0]["argument_type"] == "user_io"
    assert specs[1]["name"] == "model.embed_tokens.weight"
    assert specs[1]["argument_type"] == "weight"
    assert specs[1]["is_weight"] is True
    assert specs[2]["name"] == "model.layers.0.input_layernorm.weight"
    assert specs[2]["argument_type"] == "weight"
    assert specs[3]["name"] == "rotary_emb_inv_freq"
    assert specs[3]["argument_type"] == "constant"
    assert specs[3]["is_weight"] is False
    assert specs[3]["is_internal_constant"] is True
    assert specs[4]["name"] == "positions"
    assert specs[4]["argument_type"] == "user_io"
    assert specs[5]["name"] == "kv_cache_layer_0"
    assert specs[5]["argument_type"] == "state"
    assert specs[6]["name"] == "kv_lens"
    assert specs[7]["name"] == "block_tables"
    assert specs[8]["name"] == "query_start_loc"
    assert specs[9]["name"] == "req_dist"

    # Verify _write_exported_module labels donor state and final hidden_states outputs
    fwd_mlir = (
        "module @fwd {\n"
        "  func.func public @main(\n"
        "    %arg0: tensor<16xi32>,\n"
        "    %arg1: tensor<32x8xbf16>,\n"
        "    %arg2: tensor<8xbf16>,\n"
        "    %arg3: tensor<64x4xbf16>,\n"
        "    %arg4: tensor<16xi32>,\n"
        "    %arg5: tensor<128x16x2x2x4xbf16> "
        '{tf.aliasing_output = 0 : i32, mhlo.sharding = "{replicated}", '
        "jax.buffer_donor = true},\n"
        "    %arg6: tensor<8xi32>,\n"
        "    %arg7: tensor<128xi32>,\n"
        "    %arg8: tensor<9xi32>,\n"
        "    %arg9: tensor<3xi32>\n"
        "  ) -> (tensor<128x16x2x2x4xbf16>, tensor<16x8xbf16>) {\n"
        "    %0 = stablehlo.constant dense<0.0> : tensor<16x8xbf16>\n"
        "    return %arg5, %0 : tensor<128x16x2x2x4xbf16>, tensor<16x8xbf16>\n"
        "  }\n"
        "}\n"
    )
    stablehlo_export._write_exported_module(
        export_dir=export_dir,
        fn_key="forward_bucket16",
        subgraph_idx=0,
        compile_size=16,
        mlir_text=fwd_mlir,
        input_specs=specs,
        output_specs=[
            {
                "arg_index": 0,
                "index": 0,
                "name": "output_0",
                "shape": [128, 16, 2, 2, 4],
                "dtype": "bfloat16",
                "sharding": None,
                "argument_type": "user_io",
                "buffer_donor": False,
            },
            {
                "arg_index": 1,
                "index": 1,
                "name": "output_1",
                "shape": [16, 8],
                "dtype": "bfloat16",
                "sharding": None,
                "argument_type": "user_io",
                "buffer_donor": False,
            },
        ],
    )
    meta = json.loads((export_dir / "metadata.json").read_text())
    fwd_entry = meta["functions"]["forward"]["16"]
    assert fwd_entry["inputs"][5]["buffer_donor"] is True
    assert fwd_entry["inputs"][5]["argument_type"] == "state"
    assert fwd_entry["inputs"][5]["index"] == 0
    assert fwd_entry["outputs"][0]["name"] == "kv_cache_out_layer_0"
    assert fwd_entry["outputs"][0]["argument_type"] == "state"
    assert fwd_entry["outputs"][1]["name"] == "hidden_states"
    assert fwd_entry["outputs"][1]["argument_type"] == "intermediate"

    # Verify dummy-load HF cache snapshot (without *.safetensors) is copied
    fake_hf_home = tmp_path / "hf_cache"
    snap = (
        fake_hf_home / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / "dummy_hash"
    )
    snap.mkdir(parents=True)
    (snap / "config.json").write_text('{"vocab_size": 32}')
    (snap / "tokenizer.json").write_text('{"version": "1.0"}')
    monkeypatch.setenv("HF_HOME", str(fake_hf_home))

    class DummyRunner:
        def __init__(self) -> None:
            self.model = _LmHeadModel(hidden_size=8, vocab_size=32)
            self.num_reqs_paddings = [8]
            self.num_tokens_paddings = [16]
            self.model_config = MagicMock()
            self.model_config.model = "Qwen/Qwen3-0.6B"
            self.model_config.get_hidden_size.return_value = 8
            self.model_config.vocab_size = 32
            self.model_config.max_model_len = 2048
            self.model_config.eos_token_id = 2
            self.parallel_config = MagicMock()
            self.parallel_config.world_size = 2
            self.parallel_config.tensor_parallel_size = 1
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.data_parallel_size = 1
            self.cache_config = MagicMock()
            self.cache_config.block_size = 16
            self.cache_config.num_gpu_blocks = 128
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    dummy_runner = DummyRunner()
    with _patch_tpu_lowering(_FakeTpuLowering(dummy_runner.model)):
        stablehlo_export.maybe_export_sampling_subgraphs(dummy_runner)

    assert (export_dir / "config.json").is_file()
    assert (export_dir / "tokenizer.json").is_file()
    assert not (export_dir / "serving_checkpoint" / "constants.safetensors").exists()
    wj_final = json.loads(
        (export_dir / "serving_checkpoint" / "weight.json").read_text()
    )
    assert "model.embed_tokens.weight" in wj_final
    assert "rotary_emb_inv_freq" not in wj_final
    assert "lm_head.weight" in wj_final
    assert wj_final["model.embed_tokens.weight"]["sources"][0]["path"] == str(
        (snap / "model.safetensors").absolute()
    )


def test_scalar_buffer_constants_and_extra_snapshot_assets(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Verify 0-D scalar buffers and extra HF snapshot assets."""
    export_dir = tmp_path / "export_scalar_buf"
    monkeypatch.setattr(
        stablehlo_export.envs, "VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir)
    )
    stablehlo_export._CAPTURED_CONSTANTS.clear()

    class VocabParallelDummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Module()
            self.model.embed_tokens.register_buffer(
                "org_vocab_start_index", torch.tensor(0, dtype=torch.int64)
            )
            self.model.embed_tokens.register_buffer(
                "org_vocab_end_index", torch.tensor(16, dtype=torch.int64)
            )
            self.model.embed_tokens.weight = torch.nn.Parameter(
                torch.zeros((16, 8), dtype=torch.bfloat16)
            )
            self.lm_head = torch.nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.lm_head(hidden_states)

    active_model = VocabParallelDummyModel()
    stablehlo_export.register_active_model(active_model)

    outer_g = fx.Graph()
    outer_g.placeholder("L_input_ids_")
    outer_g.placeholder(
        "L_self_modules_model_modules_embed_tokens_buffers_org_vocab_start_index_"
    )
    outer_g.placeholder(
        "L_self_modules_model_modules_embed_tokens_buffers_org_vocab_end_index_"
    )
    outer_g.placeholder("L_self_modules_model_modules_embed_tokens_parameters_weight_")
    outer_ph = [n for n in outer_g.nodes if n.op == "placeholder"]
    outer_g.output((outer_ph[0],))
    outer_gm = fx.GraphModule(torch.nn.Module(), outer_g)

    inner_g = fx.Graph()
    for k in range(4):
        inner_g.placeholder(f"arg{k}_1")
    inner_ph = [n for n in inner_g.nodes if n.op == "placeholder"]
    inner_g.output((inner_ph[0],))
    inner_gm = fx.GraphModule(torch.nn.Module(), inner_g)

    # Simulate FakeTensor inner args so _CAPTURED_CONSTANTS is not populated in _save_fx
    class FakeScalarTensor:
        def __init__(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
            self.shape = torch.Size(shape)
            self.dtype = dtype

    FakeScalarTensor.__name__ = "FakeTensor"
    fake_inner_args = [
        FakeScalarTensor((16,), torch.int32),
        FakeScalarTensor((), torch.int64),
        FakeScalarTensor((), torch.int64),
        FakeScalarTensor((16, 8), torch.bfloat16),
    ]
    outer_inputs = [
        torch.zeros((16,), dtype=torch.int32),
        active_model.model.embed_tokens.org_vocab_start_index,
        active_model.model.embed_tokens.org_vocab_end_index,
        active_model.model.embed_tokens.weight,
    ]

    mlir_text = """module @forward_bucket16 {
  func.func public @main(
    %arg0: tensor<16xi32>,
    %arg1: tensor<i64>,
    %arg2: tensor<i64>,
    %arg3: tensor<16x8xbf16>
  ) -> (tensor<16x8xbf16>) {
    %0 = stablehlo.constant dense<0.0> : tensor<16x8xbf16>
    return %0 : tensor<16x8xbf16>
  }
}
"""
    specs = stablehlo_export._extract_input_specs(
        inner_gm, fake_inner_args, outer_graph=outer_gm, outer_inputs=outer_inputs
    )
    stablehlo_export._write_exported_module(
        export_dir=export_dir,
        fn_key="forward_bucket16",
        subgraph_idx=0,
        compile_size=16,
        mlir_text=mlir_text,
        input_specs=specs,
        output_specs=[
            {
                "arg_index": 0,
                "index": 0,
                "name": "hidden_states",
                "shape": [16, 8],
                "dtype": "bfloat16",
                "argument_type": "intermediate",
                "buffer_donor": False,
            }
        ],
    )

    fake_hf_home = tmp_path / "hf_cache_2"
    snap = (
        fake_hf_home
        / "hub"
        / "models--moonshotai--Kimi-Linear-48B-A3B-Instruct"
        / "snapshots"
        / "main"
    )
    snap.mkdir(parents=True)
    (snap / "config.json").write_text('{"vocab_size": 16}')
    (snap / "tiktoken.model").write_text("dummy_tiktoken")
    (snap / "configuration_kimi.py").write_text("# custom config")
    monkeypatch.setenv("HF_HOME", str(fake_hf_home))

    class DummyRunner2:
        def __init__(self) -> None:
            self.model = active_model
            self.num_reqs_paddings = [8]
            self.num_tokens_paddings = [16]
            self.model_config = MagicMock()
            self.model_config.model = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
            self.model_config.get_hidden_size.return_value = 8
            self.model_config.vocab_size = 16
            self.model_config.max_model_len = 2048
            self.model_config.eos_token_id = 2
            self.parallel_config = MagicMock()
            self.parallel_config.world_size = 2
            self.parallel_config.tensor_parallel_size = 2
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.data_parallel_size = 1
            self.cache_config = MagicMock()
            self.cache_config.block_size = 16
            self.cache_config.num_gpu_blocks = 64
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    with _patch_tpu_lowering(_FakeTpuLowering(active_model)):
        stablehlo_export.maybe_export_sampling_subgraphs(DummyRunner2())

    assert (export_dir / "tiktoken.model").is_file()
    assert (export_dir / "configuration_kimi.py").is_file()
    const_st = export_dir / "serving_checkpoint" / "constants.safetensors"
    assert const_st.is_file()
    wj = json.loads((export_dir / "serving_checkpoint" / "weight.json").read_text())
    assert wj["model.embed_tokens.org_vocab_start_index"]["shape"] == []
    assert wj["model.embed_tokens.org_vocab_start_index"]["sources"][0]["path"] == str(
        const_st.absolute()
    )
    assert wj["model.embed_tokens.org_vocab_end_index"]["sources"][0]["path"] == str(
        const_st.absolute()
    )


def test_nvfp4_weights_and_mlir_signature_shape_reconciliation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Verify NVFP4 weights are extracted and @main shapes reconciled."""
    export_dir = tmp_path / "export_nvfp4"
    monkeypatch.setattr(
        stablehlo_export.envs, "VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir)
    )
    stablehlo_export.register_active_model(None)

    class FakeFp4Tensor:
        def __init__(self, shape: tuple[int, ...], dtype_str: str) -> None:
            self.shape = torch.Size(shape)
            self.dtype = dtype_str

    FakeFp4Tensor.__name__ = "FakeTensor"

    outer_g = fx.Graph()
    outer_g.placeholder("L_input_ids_")
    outer_g.placeholder(
        "L_self_modules_layers_modules_0_modules_mlp_modules_experts_parameters_w13_weight_"
    )
    outer_g.placeholder(
        "G_import_vllm_dot_forward_context_forward_context_attn_metadata_model_layers_0_self_attn_attn_request_distribution"
    )
    outer_ph = [n for n in outer_g.nodes if n.op == "placeholder"]
    outer_g.output((outer_ph[0],))
    outer_gm = fx.GraphModule(torch.nn.Module(), outer_g)

    inner_g = fx.Graph()
    for k in range(3):
        inner_g.placeholder(f"arg{k}_1")
    inner_ph = [n for n in inner_g.nodes if n.op == "placeholder"]
    inner_g.output((inner_ph[0],))
    inner_gm = fx.GraphModule(torch.nn.Module(), inner_g)

    inner_args = [
        FakeFp4Tensor((16,), "torch.int32"),
        FakeFp4Tensor((64, 6144, 2048), "torch.float4_e2m1fn_x2"),
        FakeFp4Tensor((3,), "torch.int32"),
    ]
    specs = stablehlo_export._extract_input_specs(
        inner_gm, inner_args, outer_graph=outer_gm, outer_inputs=inner_args
    )
    assert len(specs) == 3
    assert specs[1]["is_weight"] is True
    assert specs[1]["weight_name"] == "model.layers.0.mlp.experts.w13_weight"

    # In MLIR @main, NVFP4 weight is unpacked and req_dist is sharded to 6xi32
    mlir_text = """module @forward_bucket16 {
  func.func public @main(
    %arg0: tensor<16xi32>,
    %arg1: tensor<64x6144x4096xf4E2M1FN>,
    %arg2: tensor<6xi32>
  ) -> (tensor<16x8xbf16>) {
    %0 = stablehlo.custom_call @AllocateBuffer() : () -> tensor<16x8xbf16>
    return %0 : tensor<16x8xbf16>
  }
}
"""
    stablehlo_export._write_exported_module(
        export_dir=export_dir,
        fn_key="forward_bucket16",
        subgraph_idx=0,
        compile_size=16,
        mlir_text=mlir_text,
        input_specs=specs,
        output_specs=[
            {
                "arg_index": 0,
                "index": 0,
                "name": "hidden_states",
                "shape": [16, 8],
                "dtype": "bfloat16",
                "argument_type": "intermediate",
                "buffer_donor": False,
            }
        ],
    )
    meta = json.loads((export_dir / "metadata.json").read_text())
    fwd_in = meta["functions"]["forward"]["16"]["inputs"]
    assert len(fwd_in) == 3
    assert fwd_in[1]["shape"] == [64, 6144, 4096]
    assert fwd_in[1]["dtype"] == "float4_e2m1fn"
    assert fwd_in[2]["shape"] == [6]
    abs_mlir = (export_dir / "abstract_params.mlir").read_text()
    assert "tensor<64x6144x4096xf4E2M1FN>" in abs_mlir


def test_duplicate_weight_name_within_first_exported_module_shares_global_index(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Verify repeated weight inputs in the first module share one global index."""
    export_dir = tmp_path / "export_dup_weight"
    monkeypatch.setattr(
        stablehlo_export.envs, "VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir)
    )
    stablehlo_export.register_active_model(None)

    specs = [
        {
            "arg_index": 0,
            "index": 0,
            "name": "input_ids",
            "shape": [16],
            "dtype": "int32",
            "argument_type": "user_io",
            "is_weight": False,
            "weight_name": "",
            "buffer_donor": False,
        },
        {
            "arg_index": 1,
            "index": 1,
            "name": "rotary_emb_inv_freq",
            "shape": [1024, 64],
            "dtype": "float32",
            "argument_type": "weight",
            "is_weight": True,
            "weight_name": "rotary_emb_inv_freq",
            "buffer_donor": False,
        },
        {
            "arg_index": 2,
            "index": 2,
            "name": "rotary_emb_inv_freq",
            "shape": [1024, 64],
            "dtype": "float32",
            "argument_type": "weight",
            "is_weight": True,
            "weight_name": "rotary_emb_inv_freq",
            "buffer_donor": False,
        },
    ]
    mlir_text = """module @forward_bucket16 {
  func.func public @main(
    %arg0: tensor<16xi32>,
    %arg1: tensor<1024x64xf32>,
    %arg2: tensor<1024x64xf32>
  ) -> (tensor<16x8xbf16>) {
    %0 = stablehlo.constant dense<0.0> : tensor<16x8xbf16>
    return %0 : tensor<16x8xbf16>
  }
}
"""
    stablehlo_export._write_exported_module(
        export_dir=export_dir,
        fn_key="forward_bucket16",
        subgraph_idx=0,
        compile_size=16,
        mlir_text=mlir_text,
        input_specs=specs,
        output_specs=[
            {
                "arg_index": 0,
                "index": 0,
                "name": "hidden_states",
                "shape": [16, 8],
                "dtype": "bfloat16",
                "argument_type": "intermediate",
                "buffer_donor": False,
            }
        ],
    )
    meta = json.loads((export_dir / "metadata.json").read_text())
    fwd_in = meta["functions"]["forward"]["16"]["inputs"]
    abs_in = meta["functions"]["abstract_params"]["default"]["inputs"]
    assert len(abs_in) == 1
    global_idx = abs_in[0]["index"]
    assert fwd_in[1]["index"] == global_idx
    assert fwd_in[2]["index"] == global_idx


def test_maybe_export_sampling_subgraphs_handles_list_eos_token_id(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Verify list-typed hf_config.eos_token_id (e.g. GLM-5) writes compile_options."""
    export_dir = tmp_path / "export_list_eos"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "metadata.json").write_text(
        json.dumps({"format_version": 2, "functions": {}})
    )
    monkeypatch.setattr(
        stablehlo_export.envs, "VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir)
    )

    class MockGlmRunner:
        def __init__(self) -> None:
            self.model = _LmHeadModel()
            self.num_reqs_paddings = [8]
            self.num_tokens_paddings = [8]
            self.model_config = MagicMock()
            self.model_config.model = "zai-org/GLM-5.2-FP8"
            self.model_config.get_hidden_size.return_value = 16
            self.model_config.vocab_size = 32
            self.model_config.max_model_len = 2048
            self.model_config.eos_token_id = None
            self.model_config.hf_config.eos_token_id = [154820, 154827, 154829]
            self.parallel_config = MagicMock()
            self.parallel_config.world_size = 4
            self.parallel_config.tensor_parallel_size = 4
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.data_parallel_size = 1
            self.cache_config = MagicMock()
            self.cache_config.block_size = 16
            self.cache_config.num_gpu_blocks = 64
            self.speculative_config = None
            self._hidden_states_dtype = torch.bfloat16
            self.device = torch.device("cpu")

    runner = MockGlmRunner()
    with _patch_tpu_lowering(_FakeTpuLowering(runner.model)):
        stablehlo_export.maybe_export_sampling_subgraphs(runner)

    meta = json.loads((export_dir / "metadata.json").read_text())
    assert meta["config"]["eos_token_id"] == 154820
    assert meta["compile_options"] == {
        "num_replicas": 1,
        "num_partitions": 4,
        "use_spmd_partitioning": True,
    }


class _SamplingRunner:
    """The runner attributes the sampling export reads."""

    def __init__(
        self, model: torch.nn.Module, num_reqs_paddings: list[int], hidden_size: int
    ) -> None:
        self.model = model
        self.num_reqs_paddings = num_reqs_paddings
        self.num_tokens_paddings = num_reqs_paddings
        self.model_config = MagicMock()
        self.model_config.model = "Qwen/Qwen3-0.6B"
        self.model_config.get_hidden_size.return_value = hidden_size
        self.model_config.vocab_size = 32
        self.model_config.max_model_len = 2048
        self.model_config.eos_token_id = 1
        self.parallel_config = MagicMock()
        self.parallel_config.world_size = 2
        self.parallel_config.tensor_parallel_size = 2
        self.parallel_config.pipeline_parallel_size = 1
        self.parallel_config.data_parallel_size = 1
        self.cache_config = MagicMock()
        self.cache_config.block_size = 16
        self.cache_config.num_gpu_blocks = 64
        self.speculative_config = None
        self._hidden_states_dtype = torch.bfloat16
        self.device = torch.device("cpu")


def test_sample_top_tokens_is_argmax_of_compute_logits(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """`sample_top_tokens` lowers the argmax of the model's own `compute_logits`."""
    export_dir = tmp_path / "export_argmax"
    monkeypatch.setattr(
        stablehlo_export.envs, "VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir)
    )
    model = _LmHeadModel(hidden_size=8, vocab_size=32)
    torch.nn.init.normal_(model.lm_head.weight)
    runner = _SamplingRunner(model, [4], hidden_size=8)
    lowering = _FakeTpuLowering(model)

    with _patch_tpu_lowering(lowering):
        stablehlo_export.maybe_export_sampling_subgraphs(runner)

    (logits_fn, _, _), (tokens_fn, _, tokens_out) = lowering.calls
    hidden = torch.randn((4, 8), dtype=torch.bfloat16)
    with torch.no_grad():
        logits = model.compute_logits(hidden)
        assert torch.equal(logits_fn(hidden), logits)
        assert torch.equal(
            tokens_fn(hidden), torch.argmax(logits, dim=-1, keepdim=True)
        )
    assert tokens_out.dtype == torch.int64

    meta = json.loads((export_dir / "metadata.json").read_text())
    tokens_entry = meta["functions"]["sample_top_tokens"]["4"]
    assert [o["name"] for o in tokens_entry["outputs"]] == ["tokens"]
    assert tokens_entry["outputs"][0]["shape"] == [4, 1]
    assert tokens_entry["outputs"][0]["dtype"] == "int64"
    logits_entry = meta["functions"]["compute_logits"]["4"]
    assert [o["name"] for o in logits_entry["outputs"]] == ["logits"]
    assert logits_entry["outputs"][0]["shape"] == [4, 32]


def _with_all_gather(mlir_text: str) -> str:
    return mlir_text.replace(
        "    %0 =",
        '    %ag = "stablehlo.all_gather"(%arg1) {all_gather_dim = 1 : i64}\n    %0 =',
    )


def test_sampling_export_writes_backend_module_verbatim(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The exported `.mlir` is the backend's module byte for byte, collectives
    included, and input roles come from the lowered arguments, not positions."""
    export_dir = tmp_path / "export_verbatim"
    monkeypatch.setattr(
        stablehlo_export.envs, "VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir)
    )
    model = _LmHeadModel(hidden_size=8, vocab_size=16)
    runner = _SamplingRunner(model, [4], hidden_size=8)

    class _Tp2Lowering(_FakeTpuLowering):
        """Weight first, logits gathered across two vocabulary shards."""

        def __call__(
            self, fn: Any, example: torch.Tensor
        ) -> stablehlo_export.LoweredModule:
            with torch.no_grad():
                out = fn(example)
            if out.dtype != torch.int64:
                out = torch.cat([out, out], dim=-1)
            inputs = (self.lm_head_weight, example)
            return stablehlo_export.LoweredModule(
                mlir_text=_with_all_gather(_render_main(inputs, out)),
                inputs=inputs,
                outputs=(out,),
            )

    with _patch_tpu_lowering(_Tp2Lowering(model)):
        stablehlo_export.maybe_export_sampling_subgraphs(runner)

    written = (export_dir / "compute_logits_req4.mlir").read_text()
    assert written == _with_all_gather(
        _render_main(
            (model.lm_head.weight, torch.zeros((4, 8), dtype=torch.bfloat16)),
            torch.zeros((4, 32), dtype=torch.bfloat16),
        )
    )

    meta = json.loads((export_dir / "metadata.json").read_text())
    inputs = meta["functions"]["compute_logits"]["4"]["inputs"]
    assert [(i["arg_index"], i["name"], i["argument_type"]) for i in inputs] == [
        (0, "lm_head.weight", "weight"),
        (1, "hidden_states", "intermediate"),
    ]
    assert meta["functions"]["compute_logits"]["4"]["outputs"][0]["shape"] == [4, 32]


def test_sampling_export_rejects_inputs_outside_the_contract(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A lowered sampling module with an extra input fails the export."""
    export_dir = tmp_path / "export_extra_input"
    monkeypatch.setattr(
        stablehlo_export.envs, "VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir)
    )
    model = _LmHeadModel(hidden_size=8, vocab_size=16)
    runner = _SamplingRunner(model, [4], hidden_size=8)
    bias = torch.zeros((16,), dtype=torch.bfloat16)

    with (
        _patch_tpu_lowering(_FakeTpuLowering(model, extra_inputs=(bias,))),
        pytest.raises(ValueError, match="compute_logits_req4"),
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(runner)
    assert not (export_dir / "compute_logits_req4.mlir").exists()


def test_sampling_export_rejects_mlir_signature_mismatch(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The `@main` signature must match the lowered argument tensors."""
    export_dir = tmp_path / "export_bad_signature"
    monkeypatch.setattr(
        stablehlo_export.envs, "VLLM_TORCHTPU_EXPORT_HLO_DIR", str(export_dir)
    )
    model = _LmHeadModel(hidden_size=8, vocab_size=16)
    runner = _SamplingRunner(model, [4], hidden_size=8)
    stale_text = _render_main(
        (torch.zeros((2, 8), dtype=torch.bfloat16), model.lm_head.weight),
        torch.zeros((2, 16), dtype=torch.bfloat16),
    )

    with (
        _patch_tpu_lowering(_FakeTpuLowering(model, mlir_text=stale_text)),
        pytest.raises(ValueError, match="@main"),
    ):
        stablehlo_export.maybe_export_sampling_subgraphs(runner)


def test_find_lm_head_weight_handles_tied_embeddings() -> None:
    """A tied `lm_head` shares the embedding parameter and is still found."""

    class _Tied(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(16, 8)
            self.lm_head = self.model.embed_tokens

    tied = _Tied()
    assert stablehlo_export._find_lm_head_weight(tied) is tied.model.embed_tokens.weight

    with pytest.raises(ValueError, match="lm_head"):
        stablehlo_export._find_lm_head_weight(torch.nn.Linear(2, 2))


def test_lower_with_tpu_backend_captures_the_backend_module(
    monkeypatch: Any,
) -> None:
    """The compiled function's lowering is captured, then `fx_to_mlir` restored."""
    from torch_tpu._internal import export as torch_tpu_export

    hidden = torch.zeros((4, 8), dtype=torch.bfloat16)
    weight = torch.zeros((16, 8), dtype=torch.bfloat16)
    logits = torch.zeros((4, 16), dtype=torch.bfloat16)
    lowered_text = _render_main((hidden, weight), logits)

    def _fake_fx_to_mlir(module: Any, args: list[Any], **_kw: Any) -> Any:
        res = MagicMock()
        res.is_noop = False
        res.updates_default_generator_state = False
        res.mlir_result_tensors = [logits]
        res.serialize_text.return_value = lowered_text
        return res

    def _fake_compile(fn: Any, **kw: Any) -> Any:
        assert kw == {"backend": "tpu", "fullgraph": True, "dynamic": False}

        def _run(example: torch.Tensor) -> torch.Tensor:
            torch_tpu_export.fx_to_mlir(MagicMock(), [example, weight, 3])
            return fn(example)

        return _run

    monkeypatch.setattr(torch_tpu_export, "fx_to_mlir", _fake_fx_to_mlir)
    monkeypatch.setattr(stablehlo_export.torch, "compile", _fake_compile)

    lowered = stablehlo_export._lower_with_tpu_backend(lambda h: h, hidden)
    assert lowered.mlir_text == lowered_text
    assert lowered.inputs == (hidden, weight)
    assert lowered.outputs == (logits,)
    assert torch_tpu_export.fx_to_mlir is _fake_fx_to_mlir


def test_lower_with_tpu_backend_fails_without_a_lowering(monkeypatch: Any) -> None:
    """A call that lowers nothing (e.g. a compile cache hit) fails the export."""
    from torch_tpu._internal import export as torch_tpu_export

    orig = torch_tpu_export.fx_to_mlir
    monkeypatch.setattr(stablehlo_export.torch, "compile", lambda fn, **_kw: fn)
    with pytest.raises(RuntimeError, match="lowered 0"):
        stablehlo_export._lower_with_tpu_backend(
            lambda h: h, torch.zeros((1, 1), dtype=torch.bfloat16)
        )
    assert torch_tpu_export.fx_to_mlir is orig


def test_internal_constant_takes_the_dtype_the_module_declares() -> None:
    """A constant is stored in the dtype of its `@main` argument.

    vLLM keeps `cos_sin_cache` in float32 while the forward graph takes it in the
    model dtype; storing the float32 buffer would hand the runtime twice the bytes
    the argument holds.
    """
    cache = torch.randn((8, 4), dtype=torch.float32)
    spec = {"shape": [8, 4], "dtype": "bfloat16"}
    stored = stablehlo_export._constant_for_spec("rotary_emb_inv_freq", cache, spec)
    assert stored.dtype == torch.bfloat16
    assert torch.equal(stored, cache.to(torch.bfloat16))

    with pytest.raises(ValueError, match="rotary_emb_inv_freq"):
        stablehlo_export._constant_for_spec(
            "rotary_emb_inv_freq", cache, {"shape": [4, 8], "dtype": "bfloat16"}
        )


def test_infer_weight_sharding_and_packed_1d_sources() -> None:
    """Per-tensor mesh sharding and 1D packed weight_scale sources are generated."""

    class _FakeQKV(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tp_size = 4
            self.output_sizes = [4096, 1024, 1024]
            self.input_size = 5120
            self.weight = torch.nn.Parameter(
                torch.zeros((5120, 1536), dtype=torch.bfloat16)
            )
            self.weight_scale = torch.nn.Parameter(
                torch.zeros((1536,), dtype=torch.float32)
            )

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.packed_modules_mapping = {
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            }
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].self_attn = torch.nn.Module()
            self.model.layers[0].self_attn.qkv_proj = _FakeQKV()

    fake_model = _FakeModel()
    assert stablehlo_export._infer_weight_sharding(
        "model.embed_tokens.weight", [37984, 5120], tp_size_hint=4
    ) == ["tp", None]
    assert (
        stablehlo_export._infer_weight_sharding(
            "model.embed_tokens.org_vocab_start_index", [], tp_size_hint=4
        )
        == "tp"
    )
    assert stablehlo_export._infer_weight_sharding(
        "model.layers.0.self_attn.qkv_proj.weight",
        [5120, 1536],
        model=fake_model,
    ) == [None, "tp"]
    assert stablehlo_export._infer_weight_sharding(
        "model.layers.0.self_attn.qkv_proj.weight_scale",
        [1536],
        model=fake_model,
    ) == ["tp"]
    assert stablehlo_export._infer_weight_sharding(
        "model.layers.0.self_attn.o_proj.weight",
        [2048, 5120],
        tp_size_hint=4,
    ) == ["tp", None]
    assert stablehlo_export._infer_weight_sharding(
        "model.layers.0.mlp.experts.routed_experts.w13_weight",
        [64, 2048, 1024],
        tp_size_hint=4,
        has_ep_hint=True,
    ) == ["ep", None, None]

    sources_1d = stablehlo_export._build_generic_weight_sources(
        "model.layers.0.self_attn.qkv_proj.weight_scale",
        [1536],
        fake_model,
    )
    assert sources_1d == [
        {
            "path": "",
            "source_tensor": "model.layers.0.self_attn.q_proj.weight_scale",
            "source_shape": [1024],
            "dst_slices": [{"dim": 0, "start": 0, "length": 1024}],
        },
        {
            "path": "",
            "source_tensor": "model.layers.0.self_attn.k_proj.weight_scale",
            "source_shape": [256],
            "dst_slices": [{"dim": 0, "start": 1024, "length": 256}],
        },
        {
            "path": "",
            "source_tensor": "model.layers.0.self_attn.v_proj.weight_scale",
            "source_shape": [256],
            "dst_slices": [{"dim": 0, "start": 1280, "length": 256}],
        },
    ]
