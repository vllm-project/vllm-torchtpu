# SPDX-License-Identifier: Apache-2.0
"""Unit tests for StableHLO export and save-load round-trip API."""

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.fx as fx
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


def test_env_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", "/tmp/mock_export_hlo")
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", "/tmp/mock_hlo_export")

    with pytest.raises(ValueError, match="mutually exclusive"):
        stablehlo_export._validate_hlo_export_load_exclusivity()

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)


def test_deterministic_key_generation() -> None:
    key_bucket16 = stablehlo_export.compute_function_key(
        compile_range=Range(start=16, end=16),
        subgraph_index=0,
    )
    assert key_bucket16 == "forward_bucket16"

    key_bucket32 = stablehlo_export.compute_function_key(
        compile_range=Range(start=32, end=32),
        subgraph_index=0,
    )
    assert key_bucket32 == "forward_bucket32"

    key_piece1 = stablehlo_export.compute_function_key(
        compile_range=Range(start=16, end=16),
        subgraph_index=1,
    )
    assert key_piece1 == "forward_piece1_bucket16"


def test_profile_run_preserves_behavior_and_skips_export_and_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph, inputs = _make_sample_graph_and_inputs(batch_size=16, hidden_size=64)
    compile_range = Range(start=16, end=16)
    cache_key = "artifact_compile_range_16_16_subgraph_0"

    # 1. Export mode: cache_dir contains "profile_cache" -> skips export (returns None)
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path / "hlo"))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    profile_compiler = TpuCompilerAdaptor()
    profile_cache_dir = tmp_path / "cache" / "profile_cache" / "abcd1234"
    profile_compiler.initialize_cache(cache_dir=str(profile_cache_dir))

    mock_mlir_res = MagicMock()
    mock_mlir_res.serialize_text.return_value = "module {}"
    mock_mlir_res.mlir_result_tensors = [inputs[0]]
    mock_mlir_res.reconstruct_fx_outputs_fn = lambda a, o: o
    mock_mlir_res.updates_default_generator_state = False
    mock_mlir_res.is_noop = False

    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_mlir_res) as mock_fx_to_mlir,
        patch.object(profile_compiler, "_run_backend", return_value=(lambda *a, **kw: inputs[0], None)) as mock_run_backend,
    ):
        compiled_fn, _ = profile_compiler.compile(
            graph, inputs, {}, compile_range, key=cache_key
        )
        assert compiled_fn is not None
        mock_fx_to_mlir.assert_not_called()
        mock_run_backend.assert_called_once()
    assert not (tmp_path / "hlo").exists()

    # 2. Load mode: cache_dir contains "profile_cache" -> skips load (falls through to normal compile)
    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path / "hlo"))
    with (
        patch.object(profile_compiler, "_run_backend", return_value=(lambda *a, **kw: inputs[0], None)) as mock_run_backend_load,
    ):
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
    mock_exported.reconstruct_fx_outputs_fn = lambda args, outs: outs
    mock_exported.updates_default_generator_state = False
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

    expected_fn_key = stablehlo_export.compute_function_key(
        graph, inputs, compile_range=compile_range, key=cache_key
    )
    mlir_path = tmp_path / f"{expected_fn_key}.mlir"
    metadata_path = tmp_path / "metadata.json"
    assert mlir_path.is_file()
    assert mlir_path.read_text() == dummy_mlir
    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text())
    assert expected_fn_key in metadata
    entry = metadata[expected_fn_key]
    assert entry["file_path"] == f"{expected_fn_key}.mlir"
    assert entry["subgraph_index"] == 0
    assert entry["compile_size"] == 16
    assert len(entry["inputs"]) == 1
    assert entry["inputs"][0] == {
        "index": 0,
        "name": "hidden_states",
        "shape": [16, 64],
        "dtype": "bfloat16",
        "sharding": "PartitionSpec(None, 'model')",
        "buffer_donor": False,
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

    fn_key = stablehlo_export.compute_function_key(
        graph, inputs, compile_range=compile_range, key=cache_key
    )
    saved_mlir = (
        "module @custom_stablehlo {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    (tmp_path / f"{fn_key}.mlir").write_text(saved_mlir)
    metadata = {
        fn_key: {
            "file_path": f"{fn_key}.mlir",
            "subgraph_index": 0,
            "compile_size": 16,
            "inputs": [
                {
                    "index": 0,
                    "name": "hidden_states",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": "PartitionSpec(None, 'model')",
                }
            ],
            "outputs": [
                {
                    "index": 0,
                    "name": "output_0",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": "PartitionSpec(None, 'model')",
                }
            ],
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

    fn_key = stablehlo_export.compute_function_key(
        graph, inputs, compile_range=compile_range, key=cache_key
    )
    (tmp_path / f"{fn_key}.mlir").write_text("module {}")
    metadata = {
        fn_key: {
            "file_path": f"{fn_key}.mlir",
            "subgraph_index": 0,
            "compile_size": 16,
            "inputs": [
                {
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
    mock_exported.reconstruct_fx_outputs_fn = lambda args, outs: outs
    mock_exported.updates_default_generator_state = False
    mock_exported.is_noop = False

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    fn_key = stablehlo_export.compute_function_key(
        graph, inputs, compile_range=compile_range, key=cache_key
    )
    mlir_file = tmp_path / f"{fn_key}.mlir"
    assert mlir_file.is_file()

    # Simulate customer replacing the exported MLIR with custom optimized StableHLO
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
        assert torch.equal(compiled_fn(inputs[0]), inputs[0] * 2)

    metadata_file = explicit_export_dir / "metadata.json"
    assert metadata_file.is_file()
    metadata = json.loads(metadata_file.read_text())
    assert len(metadata) == 1
    fn_key = next(iter(metadata))
    mlir_file = explicit_export_dir / metadata[fn_key]["file_path"]
    assert mlir_file.read_text() == exported_mlir_text

    # Modify MLIR externally and reload via AOT backend hook
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

    # Case A: metadata.json does NOT contain expected_key -> raises FileNotFoundError
    metadata_other = {
        "forward_bucket32": {
            "file_path": "forward_bucket32.mlir",
            "subgraph_index": 0,
            "compile_size": 32,
            "inputs": [],
            "outputs": [],
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata_other, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError, match=expected_key):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    # Case B: metadata.json has expected_key, but input shape differs ->
    # raises ValueError
    metadata_exact = {
        expected_key: {
            "file_path": f"{expected_key}.mlir",
            "subgraph_index": 0,
            "compile_size": 16,
            "inputs": [
                {
                    "index": 0,
                    "name": "hidden_states",
                    "shape": [32, 64],  # expected shape 32x64, runtime has 16x64
                    "dtype": "bfloat16",
                    "sharding": None,
                }
            ],
            "outputs": [],
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
    mock_exported.reconstruct_fx_outputs_fn = lambda args, outs: outs
    mock_exported.updates_default_generator_state = False
    mock_exported.is_noop = False

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    # 1. Export mode inside shape_variants.warmup() should remember built buckets
    # and hand them over to subsequent traces without re-exporting.
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

    # 2. Load mode inside shape_variants.warmup() should remember loaded buckets
    # and hand them over to subsequent traces without compiling a duplicate PJRT exe.
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

    fn_key = stablehlo_export.compute_function_key(
        graph, inputs, compile_range=compile_range, key=cache_key
    )
    mlir_path = tmp_path / f"{fn_key}.mlir"
    metadata = {
        fn_key: {
            "file_path": f"{fn_key}.mlir",
            "subgraph_index": 0,
            "compile_size": 16,
            "inputs": [
                {
                    "index": 0,
                    "name": "hidden_states",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": "PartitionSpec(None, 'model')",
                }
            ],
            "outputs": [
                {
                    "index": 0,
                    "name": "output_0",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": "PartitionSpec(None, 'model')",
                }
            ],
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    # 1. Output shape altered in edited .mlir
    mlir_path.write_text(
        "module @m {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x128xbf16> {\n"
        "    return %arg0 : tensor<16x128xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main output signature mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    # 2. Output dtype altered in edited .mlir
    mlir_path.write_text(
        "module @m {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xf32> {\n"
        "    return %arg0 : tensor<16x64xf32>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main output signature mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    # 3. Output tensor count altered in edited .mlir
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

    # 4. Input tensor shape altered in edited .mlir (even when metadata.json matches)
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
        base_hash = tpu_compiler.compute_tpu_compilation_hash(cfg)

        monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
        export_hash = tpu_compiler.compute_tpu_compilation_hash(cfg)
        assert export_hash != base_hash

        monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
        monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))
        load_hash_v1 = tpu_compiler.compute_tpu_compilation_hash(cfg)
        assert load_hash_v1 != base_hash
        assert load_hash_v1 != export_hash

        # In-place edit to .mlir inside VLLM_TORCHTPU_LOAD_HLO_DIR must also
        # bust the compilation hash so outer AOT caches do not serve stale code.
        mlir_file.write_text("module @v2 {\n  func.func @main() { return }\n}\n")
        load_hash_v2 = tpu_compiler.compute_tpu_compilation_hash(cfg)
        assert load_hash_v2 != load_hash_v1


def test_load_stablehlo_disables_async_compile_and_preserves_tuple_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from torch_tpu._internal.compile.compiler import StaticCompiler

    # Part 1: Direct path on a multi-output (tuple-returning) FX graph must
    # return a tuple (not a list) when was_wrapped is False.
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

    fn_key = stablehlo_export.compute_function_key(
        tuple_graph, [inp], compile_range=compile_range, key=cache_key
    )
    (tmp_path / f"{fn_key}.mlir").write_text(
        "module @tuple_out {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>)"
        " -> (tensor<16x64xbf16>, tensor<16x64xbf16>) {\n"
        "    return %arg0, %arg0 : tensor<16x64xbf16>, tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    metadata = {
        fn_key: {
            "file_path": f"{fn_key}.mlir",
            "subgraph_index": 0,
            "compile_size": 16,
            "inputs": [
                {
                    "index": 0,
                    "name": "x",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": None,
                }
            ],
            "outputs": [
                {
                    "index": 0,
                    "name": "output_0",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": None,
                },
                {
                    "index": 1,
                    "name": "output_1",
                    "shape": [16, 64],
                    "dtype": "bfloat16",
                    "sharding": None,
                },
            ],
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
        res = loaded_fn(inp)
        assert isinstance(res, tuple)
        assert len(res) == 2

    # Part 2: AOT path must force StaticCompiler._async_compile = False while
    # the hook is active and restore it afterward.
    class _WithAttr(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(inp, requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.w

    aot_graph = fx.symbolic_trace(_WithAttr())
    monkeypatch.setattr(StaticCompiler, "_async_compile", True)
    observed_async_flags: list[bool] = []

    def _inspect_async_backend(
        g: fx.GraphModule, ex_inputs: list[Any], *args: Any, **kwargs: Any
    ) -> tuple[Any, None]:
        observed_async_flags.append(StaticCompiler._async_compile)
        return (lambda *a: (a[0],)), None

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path / "aot_out"))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)
    with patch.object(compiler, "_run_backend", side_effect=_inspect_async_backend):
        compiler.compile(aot_graph, [inp], {}, compile_range, key=cache_key)

    assert observed_async_flags == [False, True]
    assert StaticCompiler._async_compile is True


def test_rfc_contract_safeguards_buffer_donor_hardware_and_custom_call(
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
    mock_exported.reconstruct_fx_outputs_fn = lambda args, outs: outs
    mock_exported.updates_default_generator_state = False
    mock_exported.is_noop = False

    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5p-8")
    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    with patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    fn_key = stablehlo_export.compute_function_key(
        graph, inputs, compile_range=compile_range, key=cache_key
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    entry = metadata[fn_key]
    assert entry["inputs"][0]["buffer_donor"] is True
    assert entry["target_hardware"]["chip_type"] == "v5p-8"
    assert "libtpu" in entry["target_hardware"]

    mlir_path = tmp_path / f"{fn_key}.mlir"
    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    # 1. Trap 1: Dropping {jax.buffer_donor = true} in edited .mlir must fail fast.
    mlir_path.write_text(
        "module @kv_donor {\n"
        "  func.func @main(%arg0: tensor<16x64xbf16>) -> tensor<16x64xbf16> {\n"
        "    return %arg0 : tensor<16x64xbf16>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="MLIR @main buffer_donor mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    # Restore valid donor MLIR for hardware / custom_call checks.
    mlir_path.write_text(donor_mlir)

    # 2. Trap 3: Loading a v5p-8 artifact on v6e-8 hardware must fail fast.
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-8")
    with pytest.raises(ValueError, match="Target hardware mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5p-8")

    # 3. Trap 2: Compiler ABI version drift (e.g. libtpu version mismatch)
    # must fail fast.
    drifted_metadata = copy.deepcopy(metadata)
    drifted_metadata[fn_key]["target_hardware"]["libtpu"] = "0.0.0-incompatible"
    metadata_path.write_text(json.dumps(drifted_metadata, indent=2))
    with pytest.raises(ValueError, match="Compiler ABI version mismatch"):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # 4. Trap 5: Unvetted C++ custom_call target must be rejected, while
    # @tpu_custom_call (Pallas Mosaic) is permitted.
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
    mock_exported.reconstruct_fx_outputs_fn = lambda args, outs: outs
    mock_exported.updates_default_generator_state = False
    mock_exported.is_noop = False

    monkeypatch.setenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_TORCHTPU_LOAD_HLO_DIR", raising=False)

    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=str(tmp_path))

    # 1. Corrupt existing metadata.json must fail fast instead of being silently
    # wiped by json.JSONDecodeError swallowing.
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{corrupt_json")
    with (
        patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported),
        pytest.raises(ValueError, match="Corrupt metadata.json"),
    ):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)
    metadata_path.unlink()

    # Valid export with tensor encoding attribute (#stablehlo.bounds<?, ?>)
    with patch("torch_tpu._internal.export.fx_to_mlir", return_value=mock_exported):
        compiler.compile(graph, inputs, {}, compile_range, key=cache_key)

    fn_key = stablehlo_export.compute_function_key(
        graph, inputs, compile_range=compile_range, key=cache_key
    )
    mlir_path = tmp_path / f"{fn_key}.mlir"

    monkeypatch.delenv("VLLM_TORCHTPU_EXPORT_HLO_DIR", raising=False)
    monkeypatch.setenv("VLLM_TORCHTPU_LOAD_HLO_DIR", str(tmp_path))

    # 2. P0: Emptying @main() or renaming @main to @other_fn must NOT bypass
    # signature validation when metadata expects tensor inputs/outputs.
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

    # 3. P1: Zero-waste I/O — loading a valid .mlir must read <fn_key>.mlir
    # from disk only once (not once for validation and a second time for parse).
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
