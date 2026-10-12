# SPDX-License-Identifier: Apache-2.0
"""StableHLO (.mlir) model export and save-load round-trip pipeline.

When ``VLLM_TORCHTPU_EXPORT_HLO_DIR`` is set, traced FX graphs are lowered to
StableHLO ``.mlir`` files alongside ``metadata.json`` (recording input/output
tensor signatures, ``buffer_donor`` KV-cache aliasing, and ``target_hardware``
fingerprints).

When ``VLLM_TORCHTPU_LOAD_HLO_DIR`` is set, ``TpuCompilerAdaptor`` validates
the runtime FX graph, target hardware/ABI versions, ``@main`` signature,
``buffer_donor`` attributes, and ``stablehlo.custom_call`` targets against the
exported contract before compiling the ``.mlir`` artifact via
``tpu_torch_compile.parse_mlir_text`` and ``tpu_torch_compile.compile_mlir``.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import os
import re
from collections.abc import Callable
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx
from vllm.config.utils import Range

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

logger = init_logger(__name__)

_MLIR_DTYPE_TO_TORCH_NAME = {
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i1": "bool",
    "i8": "int8",
    "ui8": "uint8",
    "i16": "int16",
    "ui16": "uint16",
    "i32": "int32",
    "ui32": "uint32",
    "i64": "int64",
    "ui64": "uint64",
    "f8E4M3FN": "float8_e4m3fn",
    "f8E5M2": "float8_e5m2",
}
_TORCH_TENSOR_DTYPES = set(_MLIR_DTYPE_TO_TORCH_NAME.values())

_ALLOWED_CUSTOM_CALL_TARGETS = frozenset({
    "tpu_custom_call",
    "xla_tpu_custom_call",
    "Sharding",
    "SPMDFullToShardShape",
    "SPMDShardToFullShape",
    "pin_to_device",
    "mark_tensor",
})


@dataclasses.dataclass(frozen=True)
class _LoadedCompileResult:
    """Result object returned by `_load_traverse_and_compile` hook."""

    module: Any
    executable: Any


def _validate_hlo_export_load_exclusivity() -> tuple[str, str]:
    export_dir = str(envs.VLLM_TORCHTPU_EXPORT_HLO_DIR or "").strip()
    load_dir = str(envs.VLLM_TORCHTPU_LOAD_HLO_DIR or "").strip()
    if export_dir and load_dir:
        raise ValueError(
            "VLLM_TORCHTPU_EXPORT_HLO_DIR and VLLM_TORCHTPU_LOAD_HLO_DIR "
            "are mutually exclusive.")
    return export_dir, load_dir


def update_hash_with_loaded_hlo(hash_obj: Any) -> None:
    """Fold `VLLM_TORCHTPU_LOAD_HLO_DIR` `.mlir` and metadata bytes into hash."""
    load_hlo_dir = str(envs.VLLM_TORCHTPU_LOAD_HLO_DIR or "").strip()
    if not load_hlo_dir:
        return
    load_dir = Path(load_hlo_dir)
    if not load_dir.is_dir():
        return
    for hlo_path in sorted(
        [*load_dir.glob("*.mlir"), load_dir / "metadata.json"]):
        if hlo_path.is_file():
            hash_obj.update(hlo_path.name.encode())
            with contextlib.suppress(OSError):
                hash_obj.update(hlo_path.read_bytes())


def _is_single_tensor_output(graph: fx.GraphModule) -> bool:
    """Return True if `graph`'s output node returns a single value."""
    output_node = next(
        (node for node in graph.graph.nodes if node.op == "output"), None)
    if output_node is None:
        return False
    return not isinstance(output_node.args[0], (tuple, list))


def _ensure_tuple_output(graph: fx.GraphModule) -> tuple[fx.GraphModule, bool]:
    """Wrap a graph module to return a tuple if it returns a single tensor."""
    if not _is_single_tensor_output(graph):
        return graph, False

    output_node = next(node for node in graph.graph.nodes
                       if node.op == "output")
    output_args = output_node.args[0]
    with graph.graph.inserting_before(output_node):
        output_node.args = ((output_args, ), )
    graph.graph.lint()
    graph.recompile()
    return graph, True


def _parse_subgraph_index(key: str | None) -> int:
    """Extract piecewise subgraph index from a vLLM compile artifact key."""
    if not key:
        return 0
    match = re.search(r"(?:subgraph|submod)_(\d+)", key)
    if match is not None:
        return int(match.group(1))
    trailing = re.search(r"_(\d+)$", key)
    if trailing is not None:
        return int(trailing.group(1))
    return 0


def _dtype_name(val: Any) -> str:
    """Normalize a torch.dtype or Python value to a short dtype string."""
    dtype = getattr(val, "dtype", None)
    if dtype is not None:
        return str(dtype).removeprefix("torch.")
    if isinstance(val, torch.dtype):
        return str(val).removeprefix("torch.")
    return type(val).__name__


def _extract_sharding_spec(node: fx.Node | None, val: Any) -> str | None:
    """Extract sharding annotation from an FX node or tensor value if present."""
    if node is not None:
        for meta_key in ("sharding", "sharding_spec", "partition_spec"):
            if meta_key in node.meta and node.meta[meta_key] is not None:
                return str(node.meta[meta_key])
    sharding = getattr(val, "sharding", None)
    if sharding is not None:
        return str(sharding)
    return None


def _extract_input_specs(graph: fx.GraphModule,
                         example_inputs: list[Any]) -> list[dict[str, Any]]:
    """Extract ordered input tensor metadata for `graph` and `example_inputs`."""
    placeholders = [
        node for node in graph.graph.nodes if node.op == "placeholder"
    ]
    specs: list[dict[str, Any]] = []
    count = max(len(placeholders), len(example_inputs))
    for idx in range(count):
        node = placeholders[idx] if idx < len(placeholders) else None
        inp = example_inputs[idx] if idx < len(example_inputs) else None
        val = inp
        if val is None and node is not None:
            val = node.meta.get("example_value", node.meta.get("val"))
        shape_raw = getattr(val, "shape", None)
        shape: list[int | str] = []
        if shape_raw is not None:
            for dim in shape_raw:
                try:
                    shape.append(int(dim))
                except Exception:
                    shape.append(str(dim))
        specs.append({
            "index": idx,
            "name": node.name if node is not None else f"arg_{idx}",
            "shape": shape,
            "dtype": _dtype_name(val),
            "sharding": _extract_sharding_spec(node, val),
        })
    return specs


def _extract_output_specs(
        graph: fx.GraphModule,
        exported_mlir: Any | None = None) -> list[dict[str, Any]]:
    """Extract output tensor metadata from exported MLIR results or FX output node."""
    specs: list[dict[str, Any]] = []
    result_tensors = getattr(exported_mlir, "mlir_result_tensors", None)
    if result_tensors:
        for idx, val in enumerate(result_tensors):
            shape_raw = getattr(val, "shape", None)
            shape: list[int | str] = []
            if shape_raw is not None:
                for dim in shape_raw:
                    try:
                        shape.append(int(dim))
                    except Exception:
                        shape.append(str(dim))
            specs.append({
                "index": idx,
                "name": f"output_{idx}",
                "shape": shape,
                "dtype": _dtype_name(val),
                "sharding": _extract_sharding_spec(None, val),
            })
        return specs

    output_nodes = [node for node in graph.graph.nodes if node.op == "output"]
    if not output_nodes:
        return specs
    out_node = output_nodes[0]
    raw_args = out_node.args[0] if out_node.args else ()
    items = raw_args if isinstance(raw_args, (tuple, list)) else (raw_args, )
    for idx, item in enumerate(items):
        node = item if isinstance(item, fx.Node) else None
        val = None
        if node is not None:
            val = node.meta.get("example_value", node.meta.get("val"))
        if val is None:
            val = out_node.meta.get("example_value", out_node.meta.get("val"))
            if isinstance(val, (tuple, list)) and idx < len(val):
                val = val[idx]
        shape_raw = getattr(val, "shape", None)
        shape = []
        if shape_raw is not None:
            for dim in shape_raw:
                try:
                    shape.append(int(dim))
                except Exception:
                    shape.append(str(dim))
        specs.append({
            "index": idx,
            "name": node.name if node is not None else f"output_{idx}",
            "shape": shape,
            "dtype": _dtype_name(val),
            "sharding": _extract_sharding_spec(node or out_node, val),
        })
    return specs


def _get_target_hardware_fingerprint() -> dict[str, str | None]:
    """Collect target TPU hardware and compiler ABI versions for metadata validation."""
    fingerprint: dict[str, str | None] = {
        "chip_type": envs.TPU_ACCELERATOR_TYPE,
    }
    for pkg_key, dist_name in (
        ("torch_tpu", "torch-tpu"),
        ("jaxlib", "jaxlib"),
        ("libtpu", "libtpu"),
    ):
        try:
            fingerprint[pkg_key] = importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            fingerprint[pkg_key] = None
    return fingerprint


def _parse_mlir_tensor_type(tensor_body: str) -> tuple[list[int | str], str]:
    """Parse `<dims>x<elem>` inside `tensor<...>` into `(shape, torch_dtype)`."""
    core_body = tensor_body.split(",", 1)[0].strip()
    parts = core_body.split("x")
    elem = parts[-1]
    shape = [
        "?" if dim == "?" else int(dim) for dim in parts[:-1]
        if dim == "?" or dim.isdigit()
    ]
    return shape, _MLIR_DTYPE_TO_TORCH_NAME.get(elem, elem)


def _parse_mlir_main_args(
    args_str: str, ) -> list[tuple[list[int | str], str, bool]]:
    """Parse `@main` tensor arguments into `(shape, torch_dtype, buffer_donor)`."""
    parsed: list[tuple[list[int | str], str, bool]] = []
    for tensor_body, attr_str in re.findall(r"tensor<([^>]+)>(\s*\{[^}]*\})?",
                                            args_str):
        shape, dtype = _parse_mlir_tensor_type(tensor_body)
        is_donor = bool(attr_str
                        and re.search(r"buffer_donor\s*=\s*true", attr_str))
        parsed.append((shape, dtype, is_donor))
    return parsed


def _annotate_input_specs_from_mlir(
        mlir_text: str, input_specs: list[dict[str,
                                               Any]]) -> list[dict[str, Any]]:
    """Annotate tensor input specs with `buffer_donor` parsed from `@main`."""
    match = re.search(
        r"func\.func\s+(?:public\s+|private\s+)?@main\(([^)]*)\)",
        mlir_text,
    )
    if match is None:
        return input_specs
    mlir_inputs = _parse_mlir_main_args(match.group(1).strip())
    tensor_specs = [
        inp for inp in input_specs if inp.get("dtype") in _TORCH_TENSOR_DTYPES
    ]
    if len(mlir_inputs) == len(tensor_specs):
        for spec, (_, _, is_donor) in zip(tensor_specs, mlir_inputs):
            spec["buffer_donor"] = is_donor
    return input_specs


def _validate_mlir_custom_calls(mlir_text: str, resolved_key: str) -> None:
    """Reject `stablehlo.custom_call` targets outside `_ALLOWED_CUSTOM_CALL_TARGETS`."""
    for target in re.findall(r"stablehlo\.custom_call\s+@([a-zA-Z0-9_.:-]+)",
                             mlir_text):
        if target not in _ALLOWED_CUSTOM_CALL_TARGETS:
            raise ValueError(
                f"Disallowed stablehlo.custom_call target '@{target}' in "
                f"'{resolved_key}': allowed targets are "
                f"{sorted(_ALLOWED_CUSTOM_CALL_TARGETS)}")


def _validate_mlir_main_signature(mlir_text: str, entry: dict[str, Any],
                                  resolved_key: str) -> None:
    """Validate `@main` signature, `buffer_donor`, and custom calls in `.mlir`."""
    _validate_mlir_custom_calls(mlir_text, resolved_key)

    saved_inputs = [
        inp for inp in entry.get("inputs", [])
        if inp.get("dtype") in _TORCH_TENSOR_DTYPES
    ]
    saved_outputs = [
        out for out in entry.get("outputs", [])
        if out.get("dtype") in _TORCH_TENSOR_DTYPES
    ]

    match = re.search(
        r"func\.func\s+(?:public\s+|private\s+)?@main\(([^)]*)\)"
        r"(?:\s*->\s*([^{]+))?\s*\{",
        mlir_text,
    )
    if match is None:
        if saved_inputs or saved_outputs:
            raise ValueError(
                f"MLIR @main function signature missing in '{resolved_key}'")
        return

    args_str = match.group(1).strip()
    rets_str = (match.group(2) or "").strip()
    mlir_inputs = _parse_mlir_main_args(args_str)
    mlir_outputs = [
        _parse_mlir_tensor_type(t)
        for t in re.findall(r"tensor<([^>]+)>", rets_str)
    ]

    if len(mlir_inputs) != len(saved_inputs):
        raise ValueError(
            f"MLIR @main input count mismatch for '{resolved_key}': "
            f"expected {len(saved_inputs)}, got {len(mlir_inputs)}")
    for idx, ((actual_shape, actual_dtype, actual_donor),
              expected) in enumerate(zip(mlir_inputs, saved_inputs)):
        if actual_shape != expected.get(
                "shape") or actual_dtype != expected.get("dtype"):
            raise ValueError(
                f"MLIR @main input signature mismatch at index {idx} "
                f"('{expected.get('name')}') for '{resolved_key}': "
                f"expected shape={expected.get('shape')}, "
                f"dtype={expected.get('dtype')}; "
                f"got shape={actual_shape}, dtype={actual_dtype}")
        expected_donor = expected.get("buffer_donor")
        if expected_donor is not None and actual_donor != expected_donor:
            raise ValueError(
                f"MLIR @main buffer_donor mismatch at index {idx} "
                f"('{expected.get('name')}') for '{resolved_key}': "
                f"expected buffer_donor={expected_donor}, got {actual_donor}")

    if len(mlir_outputs) != len(saved_outputs):
        raise ValueError(
            f"MLIR @main output count mismatch for '{resolved_key}': "
            f"expected {len(saved_outputs)}, got {len(mlir_outputs)}")
    for idx, ((actual_shape, actual_dtype),
              expected) in enumerate(zip(mlir_outputs, saved_outputs)):
        if actual_shape != expected.get(
                "shape") or actual_dtype != expected.get("dtype"):
            raise ValueError(
                f"MLIR @main output signature mismatch at index {idx} "
                f"('{expected.get('name')}') for '{resolved_key}': "
                f"expected shape={expected.get('shape')}, "
                f"dtype={expected.get('dtype')}; "
                f"got shape={actual_shape}, dtype={actual_dtype}")


def _resolve_metadata_key(
    metadata: dict[str, Any],
    fn_key: str,
    load_dir: Path,
    *args: Any,
    **kwargs: Any,
) -> str:
    """Resolve `fn_key` directly in `metadata`."""
    if fn_key in metadata:
        return fn_key
    raise FileNotFoundError(
        f"Function key '{fn_key}' missing from metadata.json in {load_dir}")


def _validate_target_hardware(entry: dict[str, Any],
                              resolved_key: str) -> None:
    """Verify saved target TPU chip and compiler ABI versions match runtime."""
    saved_hw = entry.get("target_hardware")
    if not isinstance(saved_hw, dict):
        return
    current_hw = _get_target_hardware_fingerprint()
    saved_chip = saved_hw.get("chip_type")
    current_chip = current_hw.get("chip_type")
    if saved_chip and current_chip and saved_chip != current_chip:
        raise ValueError(
            f"Target hardware mismatch for '{resolved_key}': "
            f"artifact was exported for chip_type='{saved_chip}', "
            f"but current runtime has chip_type='{current_chip}'")
    for abi_key in ("torch_tpu", "jaxlib", "libtpu"):
        saved_ver = saved_hw.get(abi_key)
        current_ver = current_hw.get(abi_key)
        if saved_ver and current_ver and saved_ver != current_ver:
            raise ValueError(
                f"Compiler ABI version mismatch for '{resolved_key}' "
                f"({abi_key}): artifact has '{saved_ver}', "
                f"runtime has '{current_ver}'")


def _validate_runtime_input_specs(
    runtime_inputs: list[dict[str, Any]],
    saved_inputs: list[dict[str, Any]],
    resolved_key: str,
) -> None:
    """Verify runtime FX input count, shapes, dtypes, and shardings against metadata."""
    if len(runtime_inputs) != len(saved_inputs):
        raise ValueError(
            f"Input count mismatch for '{resolved_key}': "
            f"expected {len(saved_inputs)}, got {len(runtime_inputs)}")
    for idx, (actual, expected) in enumerate(zip(runtime_inputs,
                                                 saved_inputs)):
        shape_mismatch = actual["shape"] != expected.get("shape")
        dtype_mismatch = actual["dtype"] != expected.get("dtype")
        expected_sharding = expected.get("sharding")
        actual_sharding = actual.get("sharding")
        sharding_mismatch = (expected_sharding is not None
                             and actual_sharding is not None
                             and actual_sharding != expected_sharding)
        if shape_mismatch or dtype_mismatch or sharding_mismatch:
            raise ValueError(
                f"Input signature mismatch at index {idx} ('{actual['name']}') "
                f"for '{resolved_key}': expected shape={expected.get('shape')}, "
                f"dtype={expected.get('dtype')}, sharding={expected_sharding}; "
                f"got shape={actual['shape']}, dtype={actual['dtype']}, "
                f"sharding={actual_sharding}")


def compute_function_key(
    *args: Any,
    compile_range: Range | None = None,
    subgraph_index: int | None = None,
    key: str | None = None,
    **kwargs: Any,
) -> str:
    """Compute a deterministic key for StableHLO export and reload matching.

    Format:
    ``forward_bucket{start}`` (for whole graph / piece 0)
    ``forward_piece{piecewise_idx}_bucket{start}`` (for piece > 0)
    """
    resolved_range = compile_range
    if resolved_range is None:
        for arg in args:
            if isinstance(arg, Range):
                resolved_range = arg
                break
    if resolved_range is None:
        raise ValueError(
            "compile_range must be provided to compute_function_key")

    piecewise_idx = (subgraph_index if subgraph_index is not None else
                     _parse_subgraph_index(key))
    bucket_size = resolved_range.start
    if piecewise_idx == 0:
        return f"forward_bucket{bucket_size}"
    return f"forward_piece{piecewise_idx}_bucket{bucket_size}"


def _resolve_export_dir(cache_dir: str | None) -> Path:
    """Determine target directory for StableHLO export."""
    export_env = str(envs.VLLM_TORCHTPU_EXPORT_HLO_DIR or "").strip()
    if export_env:
        return Path(export_env)
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(os.getcwd()) / "vllm_torchtpu_hlo_export"


def _needs_aot_backend(graph: fx.GraphModule,
                       example_inputs: list[Any]) -> bool:
    """Whether `graph` requires AOTAutograd decomposition before MLIR lowering."""
    from torch._subclasses.fake_tensor import FakeTensor

    if any(isinstance(t, FakeTensor) for t in example_inputs):
        return True
    return any(node.op in ("get_attr", "call_module")
               for node in graph.graph.nodes)


def _write_exported_module(
    export_dir: Path,
    fn_key: str,
    subgraph_idx: int,
    compile_size: int,
    mlir_text: str,
    input_specs: list[dict[str, Any]],
    output_specs: list[dict[str, Any]],
) -> Path:
    """Persist `<fn_key>.mlir` and update `metadata.json` in `export_dir`."""
    export_dir.mkdir(parents=True, exist_ok=True)
    mlir_filename = f"{fn_key}.mlir"
    mlir_path = export_dir / mlir_filename
    mlir_path.write_text(mlir_text)

    metadata_path = export_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Corrupt metadata.json in {export_dir}: {exc}") from exc

    annotated_inputs = _annotate_input_specs_from_mlir(
        mlir_text, [dict(inp) for inp in input_specs])
    metadata[fn_key] = {
        "file_path": mlir_filename,
        "subgraph_index": subgraph_idx,
        "compile_size": compile_size,
        "target_hardware": _get_target_hardware_fingerprint(),
        "inputs": annotated_inputs,
        "outputs": output_specs,
    }
    tmp_metadata_path = export_dir / f"metadata.json.{os.getpid()}.tmp"
    tmp_metadata_path.write_text(json.dumps(metadata, indent=2))
    os.replace(tmp_metadata_path, metadata_path)

    logger.info(
        "[TpuCompilerAdaptor] Exported StableHLO module '%s' to %s",
        fn_key,
        mlir_path,
    )
    return mlir_path


def _validate_and_resolve_metadata(
    metadata: dict[str, Any],
    load_dir: Path,
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compile_range: Range,
    key: str | None,
    subgraph_idx: int,
    fallback_name: str | None = None,
) -> tuple[str, Path, str]:
    """Validate metadata and `.mlir` contract; return `(key, path, text)`."""
    runtime_inputs = _extract_input_specs(graph, example_inputs)
    fn_key = compute_function_key(
        compile_range=compile_range,
        subgraph_index=subgraph_idx,
        key=key,
    )
    resolved_key = _resolve_metadata_key(metadata, fn_key, load_dir)
    entry = metadata[resolved_key]
    _validate_target_hardware(entry, resolved_key)
    _validate_runtime_input_specs(runtime_inputs, entry.get("inputs", []),
                                  resolved_key)

    mlir_rel_path = entry.get("file_path", f"{resolved_key}.mlir")
    mlir_path = load_dir / mlir_rel_path
    if not mlir_path.is_file():
        raise FileNotFoundError(
            f"StableHLO file '{mlir_path}' missing for '{resolved_key}'")
    mlir_text = mlir_path.read_text()
    _validate_mlir_main_signature(mlir_text, entry, resolved_key)
    return resolved_key, mlir_path, mlir_text


def _run_aot_with_fx_to_mlir_hook(
    adaptor: TpuCompilerAdaptor,
    graph: fx.GraphModule,
    example_inputs: list[Any],
    hook: Callable[..., Any],
) -> tuple[Callable[..., Any], None]:
    """Run `adaptor._run_backend` synchronously with a custom `fx_to_mlir` hook."""
    from torch_tpu._internal import export as torch_tpu_export
    from torch_tpu._internal.compile.compiler import StaticCompiler

    export_graph, was_wrapped = _ensure_tuple_output(copy.deepcopy(graph))
    orig_fx_to_mlir = torch_tpu_export.fx_to_mlir
    orig_async = getattr(StaticCompiler, "_async_compile", False)
    torch_tpu_export.fx_to_mlir = hook
    StaticCompiler._async_compile = False
    try:
        compiled_fn, _ = adaptor._run_backend(export_graph, example_inputs)
    finally:
        torch_tpu_export.fx_to_mlir = orig_fx_to_mlir
        StaticCompiler._async_compile = orig_async

    if was_wrapped:
        compiled_fn = adaptor._unwrap_compiled_fn(compiled_fn)
    return compiled_fn, None


def _export_fx_module_to_disk(
    export_dir: Path,
    module: fx.GraphModule,
    args: list[Any],
    exported_mlir: Any,
    compile_range: Range,
    key: str | None,
    subgraph_idx: int,
    fallback_name: str | None = None,
) -> Path:
    """Compute key, extract I/O specs, and persist a single lowered MLIR module."""
    input_specs = _extract_input_specs(module, args)
    fn_key = compute_function_key(
        compile_range=compile_range,
        subgraph_index=subgraph_idx,
        key=key,
    )
    return _write_exported_module(
        export_dir=export_dir,
        fn_key=fn_key,
        subgraph_idx=subgraph_idx,
        compile_size=compile_range.start,
        mlir_text=exported_mlir.serialize_text(),
        input_specs=input_specs,
        output_specs=_extract_output_specs(module, exported_mlir),
    )


def export_stablehlo(
    adaptor: TpuCompilerAdaptor,
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compile_range: Range,
    key: str | None,
) -> tuple[Callable[..., Any], None]:
    """Export traced FX subgraph to StableHLO (.mlir) and metadata.json."""
    from torch_tpu._internal import export as torch_tpu_export

    export_dir = _resolve_export_dir(adaptor.cache_dir)
    subgraph_idx = _parse_subgraph_index(key)
    outer_name = getattr(graph, "_get_name", lambda: None)()

    if _needs_aot_backend(graph, example_inputs):
        orig_fx_to_mlir = torch_tpu_export.fx_to_mlir
        part_counter = 0

        def _export_fx_to_mlir_hook(
            module: fx.GraphModule,
            args: list[Any],
            build_mlir_module: bool = True,
            **kwargs: Any,
        ) -> Any:
            nonlocal part_counter
            exported_mlir = orig_fx_to_mlir(module,
                                            args,
                                            build_mlir_module=True,
                                            **kwargs)
            if not exported_mlir.is_noop and exported_mlir.module is not None:
                part_idx = subgraph_idx * 100 + part_counter
                part_counter += 1
                _export_fx_module_to_disk(
                    export_dir=export_dir,
                    module=module,
                    args=list(args),
                    exported_mlir=exported_mlir,
                    compile_range=compile_range,
                    key=key,
                    subgraph_idx=part_idx,
                    fallback_name=outer_name,
                )
            return exported_mlir

        _run_aot_with_fx_to_mlir_hook(adaptor, graph, example_inputs,
                                      _export_fx_to_mlir_hook)
        return

    export_graph, _ = _ensure_tuple_output(copy.deepcopy(graph))
    exported_mlir = torch_tpu_export.fx_to_mlir(
        export_graph,
        example_inputs,
        build_mlir_module=True,
    )
    _export_fx_module_to_disk(
        export_dir=export_dir,
        module=graph,
        args=example_inputs,
        exported_mlir=exported_mlir,
        compile_range=compile_range,
        key=key,
        subgraph_idx=subgraph_idx,
    )


def _wrap_direct_executable(adaptor: TpuCompilerAdaptor, graph: fx.GraphModule,
                            pjrt_executable: Any) -> Callable[..., Any]:
    """Wrap a directly compiled PJRT executable to match `graph` outputs."""
    from torch_tpu._internal.compile.torch_tpu_compiled_executable import \
        TorchTpuCompiledExecutable

    exec_fn = TorchTpuCompiledExecutable(
        executable=pjrt_executable,
        reconstruct_fx_outputs_fn=lambda args, outputs: outputs,
        updates_default_generator_state=False,
    )
    if _is_single_tensor_output(graph):
        return adaptor._unwrap_compiled_fn(exec_fn)

    def compiled_fn(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return tuple(exec_fn(*args, **kwargs))

    return compiled_fn


def load_and_compile_stablehlo(
    adaptor: TpuCompilerAdaptor,
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compile_range: Range,
    key: str | None,
) -> Callable[..., Any]:
    """Validate metadata and load/compile StableHLO (.mlir) via TorchTPU APIs."""
    from torch_tpu._internal import export as torch_tpu_export
    from torch_tpu._internal.compile import tpu_torch_compile

    load_dir = Path(envs.VLLM_TORCHTPU_LOAD_HLO_DIR)
    metadata_path = load_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"StableHLO metadata file missing from metadata.json: {metadata_path}"
        )

    metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    subgraph_idx = _parse_subgraph_index(key)
    outer_name = getattr(graph, "_get_name", lambda: None)()

    if _needs_aot_backend(graph, example_inputs):
        orig_fx_to_mlir = torch_tpu_export.fx_to_mlir
        orig_traverse = getattr(tpu_torch_compile, "traverse_and_compile",
                                None)
        part_counter = 0

        def _load_fx_to_mlir_hook(
            module: fx.GraphModule,
            args: list[Any],
            build_mlir_module: bool = False,
            **kwargs: Any,
        ) -> Any:
            nonlocal part_counter

            def _load_traverse_and_compile(**tc_kwargs: Any,
                                           ) -> _LoadedCompileResult:
                nonlocal part_counter
                part_idx = subgraph_idx * 100 + part_counter
                part_counter += 1
                resolved_key, mlir_path, mlir_text = _validate_and_resolve_metadata(
                    metadata=metadata,
                    load_dir=load_dir,
                    graph=module,
                    example_inputs=list(args),
                    compile_range=compile_range,
                    key=key,
                    subgraph_idx=part_idx,
                    fallback_name=outer_name,
                )
                mlir_module = tpu_torch_compile.parse_mlir_text(mlir_text)
                pjrt_executable = tpu_torch_compile.compile_mlir(
                    mlir_module,
                    False,
                    tc_kwargs.get("argument_layouts") or [],
                )
                logger.info(
                    "[TpuCompilerAdaptor] Loaded StableHLO module '%s' from %s",
                    resolved_key,
                    mlir_path,
                )
                return _LoadedCompileResult(module=mlir_module,
                                            executable=pjrt_executable)

            tpu_torch_compile.traverse_and_compile = _load_traverse_and_compile
            try:
                return orig_fx_to_mlir(module,
                                       args,
                                       build_mlir_module=False,
                                       **kwargs)
            finally:
                if orig_traverse is not None:
                    tpu_torch_compile.traverse_and_compile = orig_traverse
                elif hasattr(tpu_torch_compile, "traverse_and_compile"):
                    delattr(tpu_torch_compile, "traverse_and_compile")

        compiled_fn, _ = _run_aot_with_fx_to_mlir_hook(adaptor, graph,
                                                       example_inputs,
                                                       _load_fx_to_mlir_hook)
        if key is not None:
            try:
                from vllm_torchtpu.compilation import shape_variants
                shape_variants.remember(
                    getattr(adaptor, "cache_dir", "") or "", key, graph,
                    compiled_fn)
            except Exception:
                pass
        return compiled_fn

    resolved_key, mlir_path, mlir_text = _validate_and_resolve_metadata(
        metadata=metadata,
        load_dir=load_dir,
        graph=graph,
        example_inputs=example_inputs,
        compile_range=compile_range,
        key=key,
        subgraph_idx=subgraph_idx,
    )
    mlir_module = tpu_torch_compile.parse_mlir_text(mlir_text)
    pjrt_executable = tpu_torch_compile.compile_mlir(mlir_module)
    compiled_fn = _wrap_direct_executable(adaptor, graph, pjrt_executable)
    logger.info(
        "[TpuCompilerAdaptor] Loaded StableHLO module '%s' from %s",
        resolved_key,
        mlir_path,
    )
    if key is not None:
        try:
            from vllm_torchtpu.compilation import shape_variants
            shape_variants.remember(
                getattr(adaptor, "cache_dir", "") or "", key, graph,
                compiled_fn)
        except Exception:
            pass
    return compiled_fn
