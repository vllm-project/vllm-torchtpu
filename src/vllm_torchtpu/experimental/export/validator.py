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
"""Contract validator for exported model artifact directories."""

from __future__ import annotations

import json
import os
import re
from typing import Any

ALLOWED_CUSTOM_CALLS = frozenset(
    {
        "tpu_custom_call",
        "xla_tpu_custom_call",
        "AllocateBuffer",
        "rvllm.ComputeMasks",
        "annotate_device_placement",
        "Sharding",
        "SPMDFullToShardShape",
        "SPMDShardToFullShape",
        "pin_to_device",
        "mark_tensor",
    }
)

_DTYPE_ALIASES = {
    "f32": "f32",
    "float32": "f32",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "f16": "f16",
    "float16": "f16",
    "f64": "f64",
    "float64": "f64",
    "i32": "i32",
    "int32": "i32",
    "ui32": "ui32",
    "uint32": "ui32",
    "i64": "i64",
    "int64": "i64",
    "ui64": "ui64",
    "uint64": "ui64",
    "i16": "i16",
    "int16": "i16",
    "ui16": "ui16",
    "uint16": "ui16",
    "i8": "i8",
    "int8": "i8",
    "ui8": "ui8",
    "uint8": "ui8",
    "i4": "i4",
    "int4": "i4",
    "ui4": "ui4",
    "uint4": "ui4",
    "i1": "i1",
    "bool": "i1",
    "f8e4m3fn": "f8e4m3fn",
    "float8_e4m3fn": "f8e4m3fn",
    "f8e4m3fnuz": "f8e4m3fnuz",
    "float8_e4m3fnuz": "f8e4m3fnuz",
    "f8e5m2": "f8e5m2",
    "float8_e5m2": "f8e5m2",
    "f8e5m2fnuz": "f8e5m2fnuz",
    "float8_e5m2fnuz": "f8e5m2fnuz",
    "f8e4m3b11fnuz": "f8e4m3b11fnuz",
    "float8_e4m3b11fnuz": "f8e4m3b11fnuz",
    "f4e2m1fn": "f4e2m1fn",
    "float4_e2m1fn": "f4e2m1fn",
    "float4_e2m1fn_x2": "f4e2m1fn",
}


def canonical_dtype(dtype_str: str) -> str:
    """Normalizes a dtype spelling to its canonical short name."""
    key = dtype_str.strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported tensor dtype '{dtype_str}'")
    return _DTYPE_ALIASES[key]


def _scan_attr_dict(text: str, start: int) -> int:
    """Returns the index just past the `{...}` attribute dict at `text[start]`."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 1
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError(f"Unterminated MLIR attribute dict: {text[start:]!r}")


def parse_mlir_main_signature(
    mlir_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parses @main input and output tensor specs from a StableHLO MLIR string."""
    m = re.search(
        r"func\.func(?:\s+public)?\s+@main\((.*?)\)\s*->\s*(.*?)\s*\{\s*\n",
        mlir_text,
        re.DOTALL,
    )
    if not m:
        raise ValueError(
            "MLIR module is missing 'func.func [public] @main(...) -> ... {'"
        )

    args_block = m.group(1)
    ret_block = m.group(2)

    inputs: list[dict[str, Any]] = []
    arg_pattern = re.compile(r"%arg(\d+)\s*:\s*tensor<([^>]+)>\s*")
    for match in arg_pattern.finditer(args_block):
        arg_idx = int(match.group(1))
        tensor_spec = match.group(2)
        attrs = ""
        if args_block.startswith("{", match.end()):
            attrs = args_block[match.end() : _scan_attr_dict(args_block, match.end())]
        parts = tensor_spec.split("x")
        dtype = canonical_dtype(parts[-1])
        shape = [int(d) for d in parts[:-1]] if len(parts) > 1 else []
        donor = (
            "jax.buffer_donor = true" in attrs
            or "tf.aliasing_output" in attrs
            or "buffer_donor = true" in attrs
        )
        inputs.append(
            {
                "arg_index": arg_idx,
                "shape": shape,
                "dtype": dtype,
                "buffer_donor": donor,
            }
        )

    outputs: list[dict[str, Any]] = []
    for match in re.finditer(r"tensor<([^>]+)>", ret_block):
        tensor_spec = match.group(1)
        parts = tensor_spec.split("x")
        dtype = canonical_dtype(parts[-1])
        shape = [int(d) for d in parts[:-1]] if len(parts) > 1 else []
        outputs.append(
            {
                "index": len(outputs),
                "shape": shape,
                "dtype": dtype,
            }
        )

    return inputs, outputs


def validate_mlir_custom_calls(mlir_text: str, file_name: str = "") -> None:
    """Verifies that all stablehlo.custom_call targets are in ALLOWED_CUSTOM_CALLS."""
    for match in re.finditer(
        r"stablehlo\.custom_call\s+@([A-Za-z0-9_.:-]+)", mlir_text
    ):
        target = match.group(1)
        if target not in ALLOWED_CUSTOM_CALLS:
            raise ValueError(
                f"Disallowed stablehlo.custom_call target '@{target}' in {file_name}"
            )


def _require_key(entry: dict[str, Any], key: str, where: str) -> Any:
    """Returns `entry[key]`, failing loudly when metadata omits it."""
    if key not in entry:
        raise ValueError(f"{where}: metadata.json entry is missing '{key}'")
    return entry[key]


def validate_exported_model_dir(output_dir: str) -> None:
    """Validates an exported model directory against the artifact contract."""
    if not os.path.isdir(output_dir):
        raise ValueError(f"Exported model directory does not exist: {output_dir}")

    meta_path = os.path.join(output_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise ValueError(f"Missing metadata.json in {output_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("format_version") != 2:
        raise ValueError(
            "Expected format_version == 2 in metadata.json, got "
            f"{meta.get('format_version')}"
        )

    functions = meta.get("functions")
    if not isinstance(functions, dict) or not functions:
        raise ValueError(
            "metadata.json must contain a non-empty 'functions' dictionary"
        )

    if (
        "abstract_params" not in functions
        or "default" not in functions["abstract_params"]
    ):
        raise ValueError(
            "metadata.json 'functions' must include 'abstract_params.default'"
        )

    abs_rel = functions["abstract_params"]["default"].get(
        "file_path", "abstract_params.mlir"
    )
    abs_mlir_path = os.path.join(output_dir, abs_rel)
    if not os.path.isfile(abs_mlir_path):
        raise ValueError(f"Missing abstract_params MLIR at {abs_mlir_path}")

    for fn_name, variants in functions.items():
        if not isinstance(variants, dict) or not variants:
            raise ValueError(f"Function '{fn_name}' in metadata.json has no variants")
        for variant_key, spec in variants.items():
            mlir_rel = spec.get("file_path")
            if not mlir_rel:
                raise ValueError(
                    f"Function '{fn_name}[{variant_key}]' is missing 'file_path'"
                )
            mlir_path = os.path.join(output_dir, mlir_rel)
            if not os.path.isfile(mlir_path):
                raise ValueError(
                    f"Referenced MLIR file '{mlir_rel}' for "
                    f"'{fn_name}[{variant_key}]' does not exist"
                )
            with open(mlir_path) as f:
                mlir_text = f.read()

            validate_mlir_custom_calls(mlir_text, mlir_rel)
            mlir_inputs, mlir_outputs = parse_mlir_main_signature(mlir_text)

            meta_inputs = spec.get("inputs", [])
            if len(meta_inputs) != len(mlir_inputs):
                raise ValueError(
                    f"Input count mismatch in {mlir_rel}: metadata.json has "
                    f"{len(meta_inputs)}, @main has {len(mlir_inputs)}"
                )

            mlir_in_by_arg = {inp["arg_index"]: inp for inp in mlir_inputs}
            for m_in in meta_inputs:
                arg_idx = m_in["arg_index"]
                if arg_idx not in mlir_in_by_arg:
                    raise ValueError(
                        f"arg_index {arg_idx} in metadata.json not found in "
                        f"{mlir_rel} @main"
                    )
                actual = mlir_in_by_arg[arg_idx]
                where = f"Input %arg{arg_idx} ({m_in.get('name')}) in {mlir_rel}"
                if list(_require_key(m_in, "shape", where)) != actual["shape"]:
                    raise ValueError(
                        f"Shape mismatch for %arg{arg_idx} ({m_in.get('name')}) in "
                        f"{mlir_rel}: metadata has {m_in.get('shape')}, MLIR has "
                        f"{actual['shape']}"
                    )
                in_dtype = _require_key(m_in, "dtype", where)
                if canonical_dtype(in_dtype) != actual["dtype"]:
                    raise ValueError(
                        f"Dtype mismatch for %arg{arg_idx} ({m_in.get('name')}) in "
                        f"{mlir_rel}: metadata has {m_in.get('dtype')}, MLIR has "
                        f"{actual['dtype']}"
                    )
                if bool(m_in.get("buffer_donor", False)) != actual["buffer_donor"]:
                    raise ValueError(
                        f"buffer_donor mismatch for %arg{arg_idx} "
                        f"({m_in.get('name')}) in {mlir_rel}: metadata has "
                        f"{m_in.get('buffer_donor')}, MLIR has "
                        f"{actual['buffer_donor']}"
                    )

            meta_outputs = spec.get("outputs", [])
            if len(meta_outputs) != len(mlir_outputs):
                raise ValueError(
                    f"Output count mismatch in {mlir_rel}: metadata.json has "
                    f"{len(meta_outputs)}, @main has {len(mlir_outputs)}"
                )
            for idx, (m_out, actual_out) in enumerate(zip(meta_outputs, mlir_outputs)):
                where = f"Output {idx} ({m_out.get('name')}) in {mlir_rel}"
                if list(_require_key(m_out, "shape", where)) != actual_out["shape"]:
                    raise ValueError(
                        f"Output {idx} ({m_out.get('name')}) shape mismatch in "
                        f"{mlir_rel}: metadata has {m_out.get('shape')}, MLIR has "
                        f"{actual_out['shape']}"
                    )
                out_dtype = _require_key(m_out, "dtype", where)
                if canonical_dtype(out_dtype) != actual_out["dtype"]:
                    raise ValueError(
                        f"Output {idx} ({m_out.get('name')}) dtype mismatch in "
                        f"{mlir_rel}: metadata has {m_out.get('dtype')}, MLIR has "
                        f"{actual_out['dtype']}"
                    )
