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
"""TorchTPU StableHLO MLIR lowering utilities."""

from __future__ import annotations

import copy
import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.fx as fx

from vllm_torchtpu.experimental.export import api


@dataclasses.dataclass(frozen=True)
class _DtypeSpec:
    canonical: str
    torch_dtype: torch.dtype | None
    weight_json_name: str


_DTYPE_SPECS: tuple[_DtypeSpec, ...] = (
    _DtypeSpec("bf16", torch.bfloat16, "bfloat16"),
    _DtypeSpec("f16", torch.float16, "float16"),
    _DtypeSpec("f32", torch.float32, "float32"),
    _DtypeSpec("f64", torch.float64, "float64"),
    _DtypeSpec("i8", torch.int8, "int8"),
    _DtypeSpec("u8", torch.uint8, "uint8"),
    _DtypeSpec("i16", torch.int16, "int16"),
    _DtypeSpec("i32", torch.int32, "int32"),
    _DtypeSpec("i64", torch.int64, "int64"),
    _DtypeSpec("bool", torch.bool, "bool"),
    _DtypeSpec(
        "f8e4m3fn",
        getattr(torch, "float8_e4m3fn", None),
        "float8_e4m3fn",
    ),
    _DtypeSpec(
        "f8e4m3fnuz",
        getattr(torch, "float8_e4m3fnuz", None),
        "float8_e4m3fnuz",
    ),
    _DtypeSpec(
        "f8e5m2",
        getattr(torch, "float8_e5m2", None),
        "float8_e5m2",
    ),
    _DtypeSpec(
        "f8e5m2fnuz",
        getattr(torch, "float8_e5m2fnuz", None),
        "float8_e5m2fnuz",
    ),
)

_BY_NAME: dict[str, _DtypeSpec] = {}
for _spec in _DTYPE_SPECS:
    _BY_NAME[_spec.canonical] = _spec
    _BY_NAME[_spec.weight_json_name] = _spec
    _BY_NAME[f"torch.{_spec.weight_json_name}"] = _spec
_BY_NAME["ui8"] = _BY_NAME["u8"]
_BY_NAME["i1"] = _BY_NAME["bool"]
_BY_NAME["f8E4M3FN"] = _BY_NAME["f8e4m3fn"]
_BY_NAME["f8E4M3FNUZ"] = _BY_NAME["f8e4m3fnuz"]
_BY_NAME["f8E5M2"] = _BY_NAME["f8e5m2"]
_BY_NAME["f8E5M2FNUZ"] = _BY_NAME["f8e5m2fnuz"]


def dtype_spec(dt: Any) -> _DtypeSpec:
    """Resolves a string or `torch.dtype` to its `_DtypeSpec`."""
    if isinstance(dt, str):
        key = dt.strip()
        if key in _BY_NAME:
            return _BY_NAME[key]
        low = key.lower()
        if low in _BY_NAME:
            return _BY_NAME[low]
        raise ValueError(f"Unsupported export dtype: {dt!r}")
    if isinstance(dt, torch.dtype):
        for spec in _DTYPE_SPECS:
            if spec.torch_dtype == dt:
                return spec
    name = getattr(dt, "name", None) or str(dt)
    clean = name.removeprefix("torch.")
    if clean in _BY_NAME:
        return _BY_NAME[clean]
    if clean.lower() in _BY_NAME:
        return _BY_NAME[clean.lower()]
    raise ValueError(f"Unsupported export dtype: {dt!r}")


def dtype_name(dt: Any) -> str:
    """Returns canonical short dtype string (`bf16`, `f32`, `i32`, `f8e4m3fn`, ...)."""
    return dtype_spec(dt).canonical


def torch_dtype(dt: Any) -> torch.dtype:
    """Converts a canonical dtype string or `torch.dtype` to `torch.dtype`."""
    spec = dtype_spec(dt)
    if spec.torch_dtype is None:
        raise ValueError(f"No torch.dtype available for {dt!r}")
    return spec.torch_dtype


def normalize_mlir_text(text: str) -> str:
    """Normalizes StableHLO MLIR text for portable consumption."""
    text = re.sub(
        r"\bfunc\.func\s+@main\s*\(",
        "func.func public @main(",
        text,
    )
    text = re.sub(
        r"(#sdy\.mesh<[^>]*>)",
        lambda m: re.sub(r"\s+", " ", m.group(1)),
        text,
    )
    lines = text.splitlines(keepends=True)
    cleaned = [line for line in lines if not re.match(r"^\s*sdy\.mesh\s+@", line)]
    return "".join(cleaned)


def ensure_tuple_output(graph: fx.GraphModule) -> tuple[fx.GraphModule, bool]:
    """Ensures `graph` returns a tuple of tensors."""
    out = next((n for n in graph.graph.nodes if n.op == "output"), None)
    if out is None or isinstance(out.args[0], (tuple, list)):
        return graph, False
    with graph.graph.inserting_before(out):
        out.args = ((out.args[0],),)
    graph.graph.lint()
    graph.recompile()
    return graph, True


def fx_graph_to_mlir_text(
    graph: fx.GraphModule,
    example_inputs: Sequence[torch.Tensor],
    *,
    donated_inputs: Sequence[int] | None = None,
    module_name: str | None = None,
) -> str:
    """Lowers a `torch.fx.GraphModule` to StableHLO MLIR text via TorchTPU."""
    from torch_tpu._internal import export as torch_tpu_export

    export_graph, _ = ensure_tuple_output(copy.deepcopy(graph))
    exported_mlir = torch_tpu_export.fx_to_mlir(
        export_graph,
        list(example_inputs),
        build_mlir_module=True,
        donated_inputs=donated_inputs,
        module_name=module_name,
    )
    if exported_mlir.module is None:
        raise RuntimeError(
            f"fx_to_mlir produced an empty MLIR module for {module_name}"
        )
    return normalize_mlir_text(exported_mlir.serialize_text())


def lower_torch_callable_to_mlir(
    fn: Callable[..., Any],
    arg_specs: Sequence[api.ShapeDtypeStruct],
    *,
    donate_argnums: Sequence[int] = (),
    module_name: str | None = None,
) -> str:
    """Lowers a PyTorch callable over abstract tensor specs to StableHLO MLIR text."""
    from unittest import mock

    from torch_tpu._internal import execution_mode
    from torch_tpu._internal.compile import tpu_torch_compile

    # Newer torch_tpu releases dropped the leading underscore from this member.
    eager_mode = (
        getattr(
            execution_mode.EagerMode,
            "INTERNAL_COMPILE_FX_GRAPH",
            None,
        )
        or execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
    )
    with (
        execution_mode.set_eager_mode(eager_mode),
        mock.patch.object(torch.compiler, "is_compiling", return_value=True),
        torch.no_grad(),
    ):
        arg_tensors = [
            tpu_torch_compile.placeholder(
                [int(d) for d in s.shape],
                torch_dtype(s.dtype),
                False,
            )
            for s in arg_specs
        ]
        # Clone placeholder args before invoking `fn` so in-place ops (e.g.
        # `kv_cache.copy_(new_kv_cache)`) do not attach deferred ops to the
        # placeholder arguments passed to `traverse_and_compile`.
        mapped_args = [t.clone() for t in arg_tensors]
        raw_out = fn(*mapped_args)
        if isinstance(raw_out, torch.Tensor):
            result_tensors = [raw_out]
        elif isinstance(raw_out, (tuple, list)):
            result_tensors = [t for t in raw_out if isinstance(t, torch.Tensor)]
        else:
            raise TypeError(
                f"Expected torch.Tensor or sequence of torch.Tensor from {fn}, "
                f"got {type(raw_out).__name__}"
            )
        compile_result = tpu_torch_compile.traverse_and_compile(
            result_tensors=result_tensors,
            argument_tensors=arg_tensors,
            build_mlir_module=True,
            donated_inputs=[int(i) for i in donate_argnums],
            module_name=module_name,
        )
    if compile_result.module is None:
        raise RuntimeError(
            f"traverse_and_compile produced an empty MLIR module for {module_name}"
        )
    raw_mlir = tpu_torch_compile.serialize_mlir_text(compile_result.module)
    return normalize_mlir_text(raw_mlir)


def lower_abstract_params_mlir(
    param_items: Sequence[tuple[str, api.ShapeDtypeStruct]],
) -> str:
    """Lowers the `abstract_params.mlir` identity `f(x) -> x` with `donate_argnums`."""
    specs = [spec for _, spec in param_items]
    donate_argnums = tuple(range(len(specs)))
    return lower_torch_callable_to_mlir(
        lambda *args: args,
        specs,
        donate_argnums=donate_argnums,
        module_name="abstract_params",
    )


def lower_torch_forward_to_mlir(
    model: api.ExportableTextOnlyLLM,
    *,
    user_inputs: Sequence[tuple[str, tuple[int, ...] | int]],
    forward_names: Sequence[str],
    params_map: Mapping[str, api.ShapeDtypeStruct],
    state_specs: Sequence[api.ShapeDtypeStruct],
    bucket_size: int,
) -> str:
    """Lowers `model.forward` for `bucket_size` to StableHLO MLIR text via TorchTPU."""
    num_user = len(user_inputs)
    num_weights = len(forward_names)
    state_start = num_user + num_weights
    donate_argnums = tuple(range(state_start, state_start + len(state_specs)))

    user_specs = [
        api.ShapeDtypeStruct(
            (shape_or_len,) if isinstance(shape_or_len, int) else tuple(shape_or_len),
            "i32",
        )
        for _, shape_or_len in user_inputs
    ]
    weight_specs = [
        api.ShapeDtypeStruct(
            tuple(int(d) for d in params_map[n].shape),
            dtype_name(params_map[n].dtype),
        )
        for n in forward_names
    ]
    all_specs = [*user_specs, *weight_specs, *state_specs]

    def _torch_forward_entry(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        user_args = {name: args[i] for i, (name, _) in enumerate(user_inputs)}
        w_dict = {forward_names[i]: args[num_user + i] for i in range(num_weights)}
        kv_caches = list(args[state_start:])
        updated_kvs, hidden_states = model.forward(
            w_dict,
            kv_caches,
            **user_args,
        )
        return (*updated_kvs, hidden_states)

    return lower_torch_callable_to_mlir(
        _torch_forward_entry,
        all_specs,
        donate_argnums=donate_argnums,
        module_name=f"forward_bucket{bucket_size}",
    )
