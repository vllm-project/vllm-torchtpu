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
"""AOT exporter lowering ExportableModel / ExportableTextOnlyLLM to StableHLO."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from typing import Any

from vllm_torchtpu import env_override
from vllm_torchtpu.experimental.export import (
    api,
    sampling,
    tpu_lowering,
    validator,
    weight_mapper,
)
from vllm_torchtpu.experimental.export.trace import trace

# User inputs of forward_bucket{B}.mlir, in @main order; also the keyword names
# of `ExportableTextOnlyLLM.forward`.
_USER_INPUT_NAMES = (
    "input_ids",
    "positions",
    "kv_lens",
    "block_tables",
    "query_start_loc",
    "req_dist",
)
_NUM_USER_INPUTS = len(_USER_INPUT_NAMES)

# Mesh axes of the artifact. Expert parallelism runs on the tensor-parallel
# devices, so expert dimensions are split over `model` too.
_AXIS_NAMES = ("data", "model")
_MESH_AXIS_BY_ALIAS = {"dp": "data", "tp": "model", "ep": "model"}

# Module types are shard-local: each module is the program one device runs,
# with collectives written explicitly in the graph.
_MLIR_SHAPES = "shard_local"

Sharding = list[str | list[str] | None]


def _xla_option_overrides() -> dict[str, bool | int | float | str]:
    """The XLA flags libtpu runs with, for `env_option_overrides`.

    A runtime that does not import vLLM compiles with them from
    `metadata.json` instead. Non-XLA libtpu flags are skipped.
    """
    overrides: dict[str, bool | int | float | str] = {}
    for token in env_override.libtpu_init_args():
        if not token.startswith("--xla_"):
            continue
        name, sep, value = token.removeprefix("--").partition("=")
        overrides[name] = _option_value(value) if sep else True
    return overrides


def _option_value(text: str) -> bool | int | float | str:
    if text in ("true", "false"):
        return text == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        return text
    # JSON has no infinity or NaN.
    return number if math.isfinite(number) else text


def _axis_sizes(cfg: api.TextOnlyLLMExportConfig) -> dict[str, int]:
    """Size of each mesh axis, keyed by `_AXIS_NAMES`."""
    dp_size = int(getattr(cfg, "data_parallel_size", 1) or 1)
    return {"data": dp_size, "model": cfg.tensor_parallel_size}


def _replicated(rank: int) -> Sharding:
    return [None] * rank


def _mesh_sharding(raw: Any, rank: int) -> Sharding:
    """Converts `weight_mapper.infer_weight_sharding` output to a mesh sharding.

    The result has one entry per dimension: `None` or a mesh axis name.
    """
    if raw is None:
        return _replicated(rank)
    if isinstance(raw, str):
        # Rank-0 shardings name the axis a per-device value varies along; such
        # values are constants, which `_tensor_layouts` lays out from their values.
        if rank == 0 or raw.strip() in ("", "None", "{replicated}", "replicated"):
            return _replicated(rank)
        raise ValueError(f"unsupported sharding {raw!r} for a rank-{rank} tensor")
    entries = list(raw)
    if len(entries) != rank:
        raise ValueError(f"sharding {entries} has {len(entries)} entries, rank {rank}")
    return [
        None if e is None else _MESH_AXIS_BY_ALIAS.get(str(e), str(e)) for e in entries
    ]


def _axes_size(entry: str | list[str] | None, axis_sizes: Mapping[str, int]) -> int:
    if entry is None:
        return 1
    axes = [entry] if isinstance(entry, str) else list(entry)
    return math.prod(axis_sizes[a] for a in axes)


def _global_shape(
    local_shape: Sequence[int], sharding: Sharding, axis_sizes: Mapping[str, int]
) -> list[int]:
    """The global shape whose shards, cut along `sharding`, are `local_shape`."""
    return [
        int(d) * _axes_size(entry, axis_sizes)
        for d, entry in zip(local_shape, sharding, strict=True)
    ]


def _state_sharding(shape: Sequence[int], tp_size: int) -> Sharding:
    """KV cache shards hold this device's KV heads, the third dimension."""
    sharding = _replicated(len(shape))
    if tp_size > 1 and len(shape) >= 3:
        sharding[2] = "model"
    return sharding


@dataclasses.dataclass(frozen=True)
class _TensorLayout:
    """How a weight or constant appears globally and in the lowered modules."""

    shape: list[int]  # Global shape, as declared in metadata.json.
    sharding: Sharding
    lowered_shape: tuple[int, ...]  # Shard-local `@main` argument shape.
    # Shape `model.forward` reads, when it differs from `lowered_shape`.
    reshape: tuple[int, ...] | None = None


def _tensor_layouts(
    model: api.ExportableTextOnlyLLM,
    params_map: Mapping[str, Any],
    constants: frozenset[str],
    axis_sizes: Mapping[str, int],
) -> dict[str, _TensorLayout]:
    """Lays out every weight and constant as a global, sharded tensor.

    A constant whose value differs between devices becomes a global array with a
    leading dimension split over `model`: each device holds a `[1, ...]` shard,
    which `model.forward` reads back in its own shape. Gathering constant values
    is collective, so every rank runs this with the same `params_map`.
    """
    tp_size = axis_sizes["model"]
    layouts: dict[str, _TensorLayout] = {}
    for name, spec in params_map.items():
        local = tuple(int(d) for d in spec.shape)
        if name in constants:
            values = model.constant_axis_values(name)  # type: ignore[attr-defined]
            if any(v != values[0] for v in values):
                layouts[name] = _TensorLayout(
                    shape=[len(values), *local],
                    sharding=["model", *_replicated(len(local))],
                    lowered_shape=(1, *local),
                    reshape=local,
                )
            else:
                layouts[name] = _TensorLayout(
                    shape=list(local),
                    sharding=_replicated(len(local)),
                    lowered_shape=local,
                )
            continue
        sharding = _mesh_sharding(
            weight_mapper.infer_weight_sharding(
                name,
                local,
                explicit_sharding=getattr(spec, "sharding", None),
                tp_size_hint=tp_size,
            ),
            len(local),
        )
        layouts[name] = _TensorLayout(
            shape=_global_shape(local, sharding, axis_sizes),
            sharding=sharding,
            lowered_shape=local,
        )
    return layouts


def _lowered_params(
    params_map: Mapping[str, Any], layouts: Mapping[str, _TensorLayout]
) -> dict[str, api.ShapeDtypeStruct]:
    """Param specs with the shard-local shapes the modules are lowered with."""
    return {
        n: api.ShapeDtypeStruct(
            shape=layouts[n].lowered_shape,
            dtype=tpu_lowering.dtype_name(s.dtype),
        )
        for n, s in params_map.items()
    }


def _weight_reshapes(
    layouts: Mapping[str, _TensorLayout],
) -> dict[str, tuple[int, ...]]:
    return {n: lay.reshape for n, lay in layouts.items() if lay.reshape is not None}


def _io_entry(
    *,
    arg_index: int,
    index: int,
    name: str,
    shape: Sequence[int],
    dtype: Any,
    argument_type: str,
    sharding: Sharding | None = None,
    buffer_donor: bool = False,
    is_input: bool = True,
) -> dict[str, Any]:
    """A `metadata.json` entry for a non-weight input or output."""
    entry: dict[str, Any] = {
        "arg_index": arg_index,
        "index": index,
        "name": name,
        "shape": [int(d) for d in shape],
        "dtype": tpu_lowering.dtype_name(dtype),
        "sharding": _replicated(len(shape)) if sharding is None else sharding,
        "argument_type": argument_type,
    }
    if is_input:
        entry["is_weight"] = False
        entry["weight_name"] = ""
    entry["buffer_donor"] = buffer_donor
    return entry


def _user_inputs(
    cfg: api.TextOnlyLLMExportConfig, bucket_size: int
) -> list[tuple[str, tuple[int, ...]]]:
    """Returns `(name, shape)` of the int32 user inputs of `forward_bucket{B}`."""
    pos_shape = (
        (3, bucket_size) if getattr(cfg, "uses_mrope", False) else (bucket_size,)
    )
    shapes = (
        (bucket_size,),
        pos_shape,
        (cfg.max_num_seqs,),
        (cfg.max_num_pages,),
        (cfg.max_cu_seqs,),
        (3,),
    )
    return list(zip(_USER_INPUT_NAMES, shapes, strict=True))


def _weight_slots(
    params_map: Mapping[str, Any],
) -> dict[str, int]:
    """Assigns each param its serving slot, after the user inputs."""
    return {name: _NUM_USER_INPUTS + i for i, name in enumerate(params_map)}


def _weight_input_entry(
    name: str,
    spec: Any,
    layout: _TensorLayout,
    *,
    arg_index: int,
    slot: int,
    buffer_donor: bool,
    constant: bool = False,
) -> dict[str, Any]:
    """Returns the `metadata.json` input entry for a weight or constant argument."""
    return {
        "arg_index": arg_index,
        "index": slot,
        "name": name,
        "shape": list(layout.shape),
        "dtype": tpu_lowering.dtype_name(spec.dtype),
        "sharding": list(layout.sharding),
        "argument_type": "constant" if constant else "weight",
        "is_weight": True,
        "weight_name": name,
        "buffer_donor": buffer_donor,
    }


def _export_abstract_params(
    params_map: Mapping[str, Any],
    layouts: Mapping[str, _TensorLayout],
    output_dir: str,
    *,
    constants: frozenset[str] = frozenset(),
    subdir: str = "subgraphs",
) -> tuple[dict[str, Any], dict[str, int]]:
    """Lowers the `abstract_params.mlir` identity, input `i` aliased to output `i`."""
    mlir_text = tpu_lowering.lower_abstract_params_mlir(
        list(_lowered_params(params_map, layouts).items())
    )

    target_dir = os.path.join(output_dir, subdir) if subdir else output_dir
    os.makedirs(target_dir, exist_ok=True)
    file_name = "abstract_params.mlir"
    rel_path = f"{subdir}/{file_name}" if subdir else file_name
    with open(os.path.join(target_dir, file_name), "w") as f:
        f.write(mlir_text)

    inputs_meta: list[dict[str, Any]] = []
    outputs_meta: list[dict[str, Any]] = []
    weight_slots = _weight_slots(params_map)

    for i, (name, spec) in enumerate(params_map.items()):
        inp_entry = _weight_input_entry(
            name,
            spec,
            layouts[name],
            arg_index=i,
            slot=weight_slots[name],
            buffer_donor=True,
            constant=name in constants,
        )
        inputs_meta.append(inp_entry)
        outputs_meta.append(
            {
                "arg_index": i,
                "index": inp_entry["index"],
                "name": name,
                "shape": inp_entry["shape"],
                "dtype": inp_entry["dtype"],
                "sharding": inp_entry["sharding"],
                "argument_type": inp_entry["argument_type"],
                "buffer_donor": False,
            }
        )

    spec_dict = {
        "file_path": rel_path,
        "inputs": inputs_meta,
        "outputs": outputs_meta,
    }
    return spec_dict, weight_slots


def _lm_head_name(params_map: Mapping[str, Any]) -> str:
    """The weight `compute_logits` projects with (tied embeddings fall back).

    Multimodal models nest the language model, e.g. `language_model.lm_head.weight`.
    """
    for cand in ("lm_head.weight", "model.embed_tokens.weight"):
        if cand in params_map:
            return cand
        nested = [n for n in params_map if n.endswith(f".{cand}")]
        if len(nested) == 1:
            return nested[0]
    return next(iter(params_map))


def _norm_params(params_map: Mapping[str, Any]) -> dict[str, api.ShapeDtypeStruct]:
    return {
        n: api.ShapeDtypeStruct(
            shape=tuple(int(d) for d in s.shape),
            dtype=tpu_lowering.dtype_name(s.dtype),
            sharding=getattr(s, "sharding", None),
        )
        for n, s in params_map.items()
    }


def _state_specs(model: api.ExportableTextOnlyLLM) -> list[api.ShapeDtypeStruct]:
    """Per-layer KV cache specs, from the model when it reports them."""
    cfg = model.export_config
    abstract_states_fn = getattr(model, "abstract_states", None)
    if not callable(abstract_states_fn):
        default_kv_shape = (
            cfg.num_blocks,
            cfg.block_size,
            cfg.num_kv_heads * 2,
            cfg.head_dim,
        )
        act_dtype_str = tpu_lowering.dtype_name(cfg.dtype)
        return [
            api.ShapeDtypeStruct(default_kv_shape, act_dtype_str)
            for _ in range(cfg.num_layers)
        ]
    raw_states = list(abstract_states_fn())
    if not raw_states:
        raise ValueError(f"{type(model).__name__}.abstract_states() returned no state")
    return [
        api.ShapeDtypeStruct(
            shape=tuple(int(d) for d in s.shape),
            dtype=tpu_lowering.dtype_name(s.dtype),
            sharding=s.sharding,
        )
        for s in raw_states
    ]


def _extra_user_inputs(
    model: api.ExportableTextOnlyLLM,
) -> list[tuple[str, tuple[int, ...]]]:
    """`(name, shape)` of the int32 control inputs the model takes after its state.

    Recurrent models take `mamba_state_indices`; they follow the state arguments
    so that the weight and state arguments keep their indices.
    """
    extra_fn = getattr(model, "extra_user_inputs", None)
    return list(extra_fn()) if callable(extra_fn) else []


def _recurrent_serving_config(model: api.ExportableTextOnlyLLM) -> dict[str, int]:
    """`num_state_slots` for a model with recurrent state, else nothing.

    The runtime tells a recurrent state pool (`[num_state_slots, ...]`) from a
    paged KV pool (`[num_blocks, ...]`) by its leading dimension, so the two must
    differ.
    """
    if not any(n == "mamba_state_indices" for n, _ in _extra_user_inputs(model)):
        return {}
    cfg = model.export_config
    if cfg.num_state_slots == cfg.num_blocks:
        raise ValueError(
            f"num_blocks ({cfg.num_blocks}) equals num_state_slots (max_num_seqs + 1), "
            "so the runtime could not tell KV pools from recurrent state pools; pass "
            "another --num-blocks"
        )
    return {"num_state_slots": cfg.num_state_slots}


def _main_args_read(mlir_text: str) -> set[int]:
    """Indices of the `@main` arguments that its body reads."""
    header = re.search(r"func\.func public @main\((.*)\)[^\n]*\{\n", mlir_text)
    if header is None:
        raise ValueError("MLIR module has no public @main")
    num_args = len(re.findall(r"%arg\d+:", header.group(1)))
    depth, end = 1, header.end()
    while depth and end < len(mlir_text):
        depth += {"{": 1, "}": -1}.get(mlir_text[end], 0)
        end += 1
    body = mlir_text[header.end() : end]
    return {k for k in range(num_args) if re.search(rf"%arg{k}\b", body)}


@trace("export.params_read_by_forward")
def _params_read_by_forward(
    model: api.ExportableTextOnlyLLM, params_map: Mapping[str, Any]
) -> dict[str, Any]:
    """Drops the forward weights that the forward graph never reads.

    Candidates include buffers that only some configurations use (vocabulary
    shard bounds are read only under tensor parallelism), so the smallest forward
    bucket is lowered once with every candidate and unread ones are removed.
    """
    cfg = model.export_config
    lm_head_name = _lm_head_name(params_map)
    names = [n for n in params_map if n != lm_head_name]
    bucket = min(cfg.token_buckets)
    mlir_text = tpu_lowering.lower_torch_forward_to_mlir(
        model,
        user_inputs=_user_inputs(cfg, bucket),
        forward_names=names,
        params_map=_norm_params(params_map),
        state_specs=_state_specs(model),
        bucket_size=bucket,
        trailing_user_inputs=_extra_user_inputs(model),
    )
    read = _main_args_read(mlir_text)
    kept = {n for i, n in enumerate(names) if _NUM_USER_INPUTS + i in read}
    return {n: s for n, s in params_map.items() if n == lm_head_name or n in kept}


def _export_text_only_llm_functions(
    model: api.ExportableTextOnlyLLM,
    params_map: Mapping[str, Any],
    layouts: Mapping[str, _TensorLayout],
    weight_slots: Mapping[str, int],
    output_dir: str,
    *,
    constants: frozenset[str] = frozenset(),
    subdir: str = "subgraphs",
) -> dict[str, dict[str, Any]]:
    """Lowers `forward`, `compute_logits`, and the samplers across all buckets."""
    cfg: api.TextOnlyLLMExportConfig = model.export_config
    act_dtype_str = tpu_lowering.dtype_name(cfg.dtype)
    tp_size = cfg.tensor_parallel_size
    axis_sizes = _axis_sizes(cfg)

    target_dir = os.path.join(output_dir, subdir) if subdir else output_dir
    os.makedirs(target_dir, exist_ok=True)

    lm_head_name = _lm_head_name(params_map)
    forward_names = [n for n in params_map if n != lm_head_name] or list(params_map)
    lm_head_spec = params_map[lm_head_name]
    lm_head_entry = _weight_input_entry(
        lm_head_name,
        lm_head_spec,
        layouts[lm_head_name],
        arg_index=1,
        slot=weight_slots[lm_head_name],
        buffer_donor=False,
    )
    state_specs = _state_specs(model)
    state_shardings = [
        _state_sharding(s.shape, tp_size) if s.sharding is None else list(s.sharding)
        for s in state_specs
    ]
    state_entries = [
        (_global_shape(s.shape, sharding, axis_sizes), sharding, s.dtype)
        for s, sharding in zip(state_specs, state_shardings, strict=True)
    ]
    extra_inputs = _extra_user_inputs(model)

    num_fwd_weights = len(forward_names)
    state_start = _NUM_USER_INPUTS + num_fwd_weights

    lowered = _lowered_params(params_map, layouts)
    reshapes = _weight_reshapes(layouts)

    forward_variants: dict[str, Any] = {}
    for b_idx, bucket_size in enumerate(cfg.token_buckets):
        user_inputs = _user_inputs(cfg, bucket_size)
        fwd_mlir = tpu_lowering.lower_torch_forward_to_mlir(
            model,
            user_inputs=user_inputs,
            forward_names=forward_names,
            params_map=lowered,
            state_specs=state_specs,
            bucket_size=bucket_size,
            weight_reshapes=reshapes,
            trailing_user_inputs=extra_inputs,
        )
        fwd_file = f"forward_bucket{bucket_size}.mlir"
        fwd_rel = f"{subdir}/{fwd_file}" if subdir else fwd_file
        with open(os.path.join(target_dir, fwd_file), "w") as f:
            f.write(fwd_mlir)

        fwd_inputs: list[dict[str, Any]] = [
            _io_entry(
                arg_index=k,
                index=k,
                name=u_name,
                shape=u_shape,
                dtype="i32",
                argument_type="user_io",
            )
            for k, (u_name, u_shape) in enumerate(user_inputs)
        ]
        for w_i, w_name in enumerate(forward_names):
            fwd_inputs.append(
                _weight_input_entry(
                    w_name,
                    params_map[w_name],
                    layouts[w_name],
                    arg_index=_NUM_USER_INPUTS + w_i,
                    slot=weight_slots[w_name],
                    buffer_donor=False,
                    constant=w_name in constants,
                )
            )
        for s_i, (s_shape, s_sharding, s_dtype) in enumerate(state_entries):
            fwd_inputs.append(
                _io_entry(
                    arg_index=state_start + s_i,
                    index=s_i,
                    name=f"kv_cache_layer_{s_i}",
                    shape=s_shape,
                    dtype=s_dtype,
                    argument_type="state",
                    sharding=s_sharding,
                    buffer_donor=True,
                )
            )
        extra_start = state_start + len(state_entries)
        for e_i, (e_name, e_shape) in enumerate(extra_inputs):
            fwd_inputs.append(
                _io_entry(
                    arg_index=extra_start + e_i,
                    index=extra_start + e_i,
                    name=e_name,
                    shape=e_shape,
                    dtype="i32",
                    argument_type="user_io",
                )
            )

        fwd_outputs: list[dict[str, Any]] = [
            _io_entry(
                arg_index=s_i,
                index=s_i,
                name=f"kv_cache_out_layer_{s_i}",
                shape=s_shape,
                dtype=s_dtype,
                argument_type="state",
                sharding=s_sharding,
                is_input=False,
            )
            for s_i, (s_shape, s_sharding, s_dtype) in enumerate(state_entries)
        ]
        fwd_outputs.append(
            _io_entry(
                arg_index=len(state_entries),
                index=0,
                name="hidden_states",
                shape=[bucket_size, cfg.hidden_size],
                dtype=act_dtype_str,
                argument_type="intermediate",
                is_input=False,
            )
        )

        forward_variants[str(bucket_size)] = {
            "file_path": fwd_rel,
            "subgraph_index": b_idx,
            "compile_size": bucket_size,
            "inputs": fwd_inputs,
            "outputs": fwd_outputs,
        }

    logits_variants: dict[str, Any] = {}
    # Sampler function -> request bucket -> variant, as export_samplers names them.
    sampler_variants: dict[str, dict[str, Any]] = {}

    lm_head_struct = lowered[lm_head_name]

    for r_idx, req_size in enumerate(cfg.req_buckets):
        # 1. compute_logits_req{R}.mlir
        def _logits_entry(hidden_states: Any, lm_head_w: Any) -> Any:
            return model.compute_logits(hidden_states, lm_head_w)

        hidden_spec = api.ShapeDtypeStruct((req_size, cfg.hidden_size), act_dtype_str)
        logits_mlir = tpu_lowering.lower_torch_callable_to_mlir(
            _logits_entry,
            [hidden_spec, lm_head_struct],
            module_name=f"compute_logits_req{req_size}",
        )
        _, mlir_outs = validator.parse_mlir_main_signature(logits_mlir)
        if mlir_outs:
            logits_shape = tuple(int(d) for d in mlir_outs[0]["shape"])
            logits_dtype_str = tpu_lowering.dtype_name(mlir_outs[0]["dtype"])
        else:
            logits_shape = (req_size, cfg.vocab_size)
            logits_dtype_str = act_dtype_str

        logits_file = f"compute_logits_req{req_size}.mlir"
        logits_rel = f"{subdir}/{logits_file}" if subdir else logits_file
        with open(os.path.join(target_dir, logits_file), "w") as f:
            f.write(logits_mlir)

        # `compute_logits` gathers the vocabulary shards, so logits are replicated.
        logits_variants[str(req_size)] = {
            "file_path": logits_rel,
            "subgraph_index": r_idx,
            "compile_size": req_size,
            "inputs": [
                _io_entry(
                    arg_index=0,
                    index=0,
                    name="hidden_states",
                    shape=[req_size, cfg.hidden_size],
                    dtype=act_dtype_str,
                    argument_type="intermediate",
                ),
                lm_head_entry,
            ],
            "outputs": [
                _io_entry(
                    arg_index=0,
                    index=0,
                    name="logits",
                    shape=logits_shape,
                    dtype=logits_dtype_str,
                    # Stays on device: the samplers take it as an intermediate input.
                    argument_type="intermediate",
                    is_input=False,
                )
            ],
        }

        # 2. Samplers over the device-resident logits
        samplers = sampling.export_samplers(
            output_dir,
            r_idx,
            logits_shape,
            tpu_lowering.torch_dtype(logits_dtype_str),
            logits_dtype_str,
            cfg.max_model_len,
            subdir=subdir,
        )
        for function, variant in samplers.items():
            sampler_variants.setdefault(function, {})[str(req_size)] = variant

    return {
        "forward": forward_variants,
        "compute_logits": logits_variants,
        **sampler_variants,
    }


def _emit_model_companion_files(
    cfg: api.TextOnlyLLMExportConfig, output_dir: str
) -> dict[str, Any]:
    """Copies companion files into the artifact; returns their `serving_config` entries.

    `tokenizer.json` (required unless a byte tokenizer is emitted),
    `generation_config.json`, and the chat template (as `chat_template.jinja`), the
    latter two when the checkpoint has them, are written and declared in
    `serving_config` by path; the runtime reads no other file.
    """
    serving_config: dict[str, Any] = {}

    tokenizer_written = False
    if cfg.hf_checkpoint_dir and os.path.isdir(cfg.hf_checkpoint_dir):
        src_tok = os.path.join(cfg.hf_checkpoint_dir, "tokenizer.json")
        if os.path.isfile(src_tok):
            shutil.copyfile(src_tok, os.path.join(output_dir, "tokenizer.json"))
            tokenizer_written = True

    if not tokenizer_written and cfg.emit_byte_tokenizer:
        vocab = {
            chr(i) if 33 <= i <= 126 else f"<0x{i:02x}>": i
            for i in range(cfg.vocab_size)
        }
        tokenizer_doc = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "<0x00>",
            },
        }
        with open(os.path.join(output_dir, "tokenizer.json"), "w") as f:
            json.dump(tokenizer_doc, f, indent=2)
        tokenizer_written = True

    if tokenizer_written:
        serving_config["tokenizer_config"] = {
            "source": {"TokenizerPath": {"path": "tokenizer.json"}}
        }
    else:
        # The serving runtime reads the tokenizer only from this record.
        raise FileNotFoundError(
            f"no tokenizer.json in the checkpoint directory "
            f"{cfg.hf_checkpoint_dir!r}; the serving artifact requires one"
        )

    if cfg.hf_checkpoint_dir and os.path.isdir(cfg.hf_checkpoint_dir):
        src_gen = os.path.join(cfg.hf_checkpoint_dir, "generation_config.json")
        if os.path.isfile(src_gen):
            dst_gen = os.path.join(output_dir, "generation_config.json")
            shutil.copyfile(src_gen, dst_gen)
            serving_config["generation_config"] = {"path": "generation_config.json"}

        chat_template = _hf_chat_template(cfg.hf_checkpoint_dir)
        if chat_template is not None:
            with open(os.path.join(output_dir, "chat_template.jinja"), "w") as f:
                f.write(chat_template)
            serving_config["chat_template"] = {"path": "chat_template.jinja"}

    return serving_config


def _hf_chat_template(hf_checkpoint_dir: str) -> str | None:
    """The checkpoint's chat template, as HuggingFace finds it.

    `chat_template.jinja` takes precedence over `tokenizer_config.json`'s
    `chat_template`, which is a template or a list of named templates (`default` is
    the one chat requests use).
    """
    jinja_path = os.path.join(hf_checkpoint_dir, "chat_template.jinja")
    if os.path.isfile(jinja_path):
        with open(jinja_path) as f:
            return f.read()
    config_path = os.path.join(hf_checkpoint_dir, "tokenizer_config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        template = json.load(f).get("chat_template")
    if isinstance(template, list):
        named = {t.get("name"): t.get("template") for t in template}
        template = named.get("default")
    return template if isinstance(template, str) else None


@trace("export.export")
def export(
    model: api.ExportableTextOnlyLLM,
    output_dir: str,
    *,
    subdir: str = "subgraphs",
) -> None:
    """Exports an `ExportableTextOnlyLLM` to a serving artifact directory.

    Args:
        model: The model to export.
        output_dir: Artifact directory; created if missing.
        subdir: Subdirectory of `output_dir` for `.mlir` files (default `'subgraphs'`).
    """
    if not isinstance(model, api.ExportableTextOnlyLLM):
        raise TypeError(
            "model must implement the ExportableTextOnlyLLM protocol; "
            f"got {type(model).__name__}"
        )

    cfg = model.export_config
    if not isinstance(cfg, api.TextOnlyLLMExportConfig):
        raise TypeError(
            "model.export_config must be a TextOnlyLLMExportConfig; "
            f"got {type(cfg).__name__}"
        )

    tp_size = cfg.tensor_parallel_size
    dp_size = int(getattr(cfg, "data_parallel_size", 1) or 1)
    pp_size = int(getattr(cfg, "pipeline_parallel_size", 1) or 1)
    params_map = _params_read_by_forward(model, model.abstract_params())
    constant_names_fn = getattr(model, "constant_names", None)
    constants = frozenset(
        constant_names_fn() if callable(constant_names_fn) else ()
    ) & frozenset(params_map)
    layouts = _tensor_layouts(model, params_map, constants, _axis_sizes(cfg))

    os.makedirs(output_dir, exist_ok=True)

    abs_spec, weight_slots = _export_abstract_params(
        params_map, layouts, output_dir, constants=constants, subdir=subdir
    )

    functions_dict: dict[str, dict[str, Any]] = {
        "abstract_params": {
            "default": abs_spec,
        }
    }

    llm_funcs = _export_text_only_llm_functions(
        model,
        params_map,
        layouts,
        weight_slots,
        output_dir,
        constants=constants,
        subdir=subdir,
    )
    functions_dict.update(llm_funcs)
    # `serving_config` holds only what the runtime reads: the capacities the
    # executables were exported with, then the companion files. Token buckets are
    # the `functions.forward` variants and are not repeated here.
    serving_config = {
        "vocab_size": cfg.vocab_size,
        "eos_token_id": cfg.eos_token_id,
        "max_num_seqs": cfg.max_num_seqs,
        "max_num_pages": cfg.max_num_pages,
        "max_cu_seqs": cfg.max_cu_seqs,
        "block_size": cfg.block_size,
        "num_blocks": cfg.num_blocks,
        **_recurrent_serving_config(model),
        "max_model_len": cfg.max_model_len,
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": pp_size,
        "data_parallel_size": dp_size,
        **_emit_model_companion_files(cfg, output_dir),
    }
    if cfg.speculative_config is not None:
        serving_config["speculative_config"] = cfg.speculative_config

    metadata_doc: dict[str, Any] = {
        "format_version": 2,
        "model_name": cfg.model_name,
        "target_tpu": cfg.target_tpu,
        "mlir_shapes": _MLIR_SHAPES,
        "mesh": {
            "mesh_shape": [dp_size, tp_size],
            "axis_names": list(_AXIS_NAMES),
        },
        "serving_config": serving_config,
        "compile_options": {
            "num_replicas": cfg.num_replicas,
            "num_partitions": cfg.num_partitions,
            "use_spmd_partitioning": cfg.use_spmd_partitioning,
        },
        "functions": functions_dict,
    }
    if xla_options := _xla_option_overrides():
        metadata_doc["compile_options"]["env_option_overrides"] = xla_options

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata_doc, f, indent=2)

    validator.validate_exported_model_dir(output_dir)
