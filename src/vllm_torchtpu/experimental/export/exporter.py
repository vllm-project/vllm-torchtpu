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

import json
import os
import re
import shutil
from collections.abc import Mapping
from typing import Any

from vllm_torchtpu.experimental.export import (
    api,
    sampling,
    tpu_lowering,
    validator,
    weight_mapper,
)

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
    *,
    arg_index: int,
    slot: int,
    tp_size: int,
    buffer_donor: bool,
    constant: bool = False,
) -> dict[str, Any]:
    """Returns the `metadata.json` input entry for a weight or constant argument."""
    shape = [int(d) for d in spec.shape]
    return {
        "arg_index": arg_index,
        "index": slot,
        "name": name,
        "shape": shape,
        "dtype": tpu_lowering.dtype_name(spec.dtype),
        "sharding": None
        if constant
        else weight_mapper.infer_weight_sharding(
            name,
            shape,
            explicit_sharding=getattr(spec, "sharding", None),
            tp_size_hint=tp_size,
        ),
        "argument_type": "constant" if constant else "weight",
        "is_weight": True,
        "weight_name": name,
        "buffer_donor": buffer_donor,
    }


def _export_abstract_params(
    params_map: Mapping[str, Any],
    output_dir: str,
    tp_size: int,
    *,
    constants: frozenset[str] = frozenset(),
    subdir: str = "subgraphs",
) -> tuple[dict[str, Any], dict[str, int]]:
    """Lowers the `abstract_params.mlir` identity `f(x) -> x` with `donate_argnums`."""
    param_items = [
        (
            name,
            api.ShapeDtypeStruct(
                shape=tuple(int(d) for d in spec.shape),
                dtype=tpu_lowering.dtype_name(spec.dtype),
                sharding=getattr(spec, "sharding", None),
            ),
        )
        for name, spec in params_map.items()
    ]
    mlir_text = tpu_lowering.lower_abstract_params_mlir(param_items)

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
            arg_index=i,
            slot=weight_slots[name],
            tp_size=tp_size,
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
    """The weight `compute_logits` projects with (tied embeddings fall back)."""
    for cand in ("lm_head.weight", "model.embed_tokens.weight"):
        if cand in params_map:
            return cand
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
    if len(raw_states) != cfg.num_layers:
        raise ValueError(
            f"{type(model).__name__}.abstract_states() returned "
            f"{len(raw_states)} state specs, expected cfg.num_layers == "
            f"{cfg.num_layers}"
        )
    return [
        api.ShapeDtypeStruct(
            shape=tuple(int(d) for d in s.shape),
            dtype=tpu_lowering.dtype_name(s.dtype),
        )
        for s in raw_states
    ]


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
    )
    read = _main_args_read(mlir_text)
    kept = {n for i, n in enumerate(names) if _NUM_USER_INPUTS + i in read}
    return {n: s for n, s in params_map.items() if n == lm_head_name or n in kept}


def _export_text_only_llm_functions(
    model: api.ExportableTextOnlyLLM,
    params_map: Mapping[str, Any],
    weight_slots: Mapping[str, int],
    output_dir: str,
    *,
    constants: frozenset[str] = frozenset(),
    subdir: str = "subgraphs",
) -> dict[str, dict[str, Any]]:
    """Lowers `forward`, `compute_logits`, and the samplers across all buckets."""
    cfg: api.TextOnlyLLMExportConfig = model.export_config
    act_dtype_str = tpu_lowering.dtype_name(cfg.dtype)

    target_dir = os.path.join(output_dir, subdir) if subdir else output_dir
    os.makedirs(target_dir, exist_ok=True)

    lm_head_name = _lm_head_name(params_map)
    forward_names = [n for n in params_map if n != lm_head_name] or list(params_map)
    lm_head_spec = params_map[lm_head_name]
    lm_head_entry = _weight_input_entry(
        lm_head_name,
        lm_head_spec,
        arg_index=1,
        slot=weight_slots[lm_head_name],
        tp_size=cfg.tensor_parallel_size,
        buffer_donor=False,
    )
    state_specs = _state_specs(model)

    num_fwd_weights = len(forward_names)
    state_start = _NUM_USER_INPUTS + num_fwd_weights

    norm_params = _norm_params(params_map)

    forward_variants: dict[str, Any] = {}
    for b_idx, bucket_size in enumerate(cfg.token_buckets):
        user_inputs = _user_inputs(cfg, bucket_size)
        fwd_mlir = tpu_lowering.lower_torch_forward_to_mlir(
            model,
            user_inputs=user_inputs,
            forward_names=forward_names,
            params_map=norm_params,
            state_specs=state_specs,
            bucket_size=bucket_size,
        )
        fwd_file = f"forward_bucket{bucket_size}.mlir"
        fwd_rel = f"{subdir}/{fwd_file}" if subdir else fwd_file
        with open(os.path.join(target_dir, fwd_file), "w") as f:
            f.write(fwd_mlir)

        fwd_inputs: list[dict[str, Any]] = []
        for k, (u_name, u_shape) in enumerate(user_inputs):
            fwd_inputs.append(
                {
                    "arg_index": k,
                    "index": k,
                    "name": u_name,
                    "shape": list(u_shape),
                    "dtype": "i32",
                    "argument_type": "user_io",
                    "is_weight": False,
                    "weight_name": "",
                    "buffer_donor": False,
                }
            )
        for w_i, w_name in enumerate(forward_names):
            fwd_inputs.append(
                _weight_input_entry(
                    w_name,
                    params_map[w_name],
                    arg_index=_NUM_USER_INPUTS + w_i,
                    slot=weight_slots[w_name],
                    tp_size=cfg.tensor_parallel_size,
                    buffer_donor=False,
                    constant=w_name in constants,
                )
            )
        for s_i, s_spec in enumerate(state_specs):
            fwd_inputs.append(
                {
                    "arg_index": state_start + s_i,
                    "index": s_i,
                    "name": f"kv_cache_layer_{s_i}",
                    "shape": [int(d) for d in s_spec.shape],
                    "dtype": tpu_lowering.dtype_name(s_spec.dtype),
                    "argument_type": "state",
                    "is_weight": False,
                    "weight_name": "",
                    "buffer_donor": True,
                }
            )

        fwd_outputs: list[dict[str, Any]] = [
            {
                "arg_index": s_i,
                "index": s_i,
                "name": f"kv_cache_out_layer_{s_i}",
                "shape": [int(d) for d in s_spec.shape],
                "dtype": tpu_lowering.dtype_name(s_spec.dtype),
                "argument_type": "state",
                "buffer_donor": False,
            }
            for s_i, s_spec in enumerate(state_specs)
        ]
        fwd_outputs.append(
            {
                "arg_index": cfg.num_layers,
                "index": 0,
                "name": "hidden_states",
                "shape": [bucket_size, cfg.hidden_size],
                "dtype": act_dtype_str,
                "argument_type": "intermediate",
                "buffer_donor": False,
            }
        )

        forward_variants[str(bucket_size)] = {
            "file_path": fwd_rel,
            "subgraph_index": b_idx,
            "compile_size": bucket_size,
            "inputs": fwd_inputs,
            "outputs": fwd_outputs,
        }

    logits_variants: dict[str, Any] = {}
    sample_tokens_variants: dict[str, Any] = {}
    stochastic_variants: dict[str, Any] = {}
    penalized_variants: dict[str, Any] = {}

    lm_head_struct = api.ShapeDtypeStruct(
        tuple(int(d) for d in lm_head_spec.shape),
        tpu_lowering.dtype_name(lm_head_spec.dtype),
    )

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

        logits_variants[str(req_size)] = {
            "file_path": logits_rel,
            "subgraph_index": r_idx,
            "compile_size": req_size,
            "inputs": [
                {
                    "arg_index": 0,
                    "index": 0,
                    "name": "hidden_states",
                    "shape": [req_size, cfg.hidden_size],
                    "dtype": act_dtype_str,
                    "argument_type": "intermediate",
                    "is_weight": False,
                    "weight_name": "",
                    "buffer_donor": False,
                },
                lm_head_entry,
            ],
            "outputs": [
                {
                    "arg_index": 0,
                    "index": 0,
                    "name": "logits",
                    "shape": list(logits_shape),
                    "dtype": logits_dtype_str,
                    "argument_type": "user_io",
                    "buffer_donor": False,
                }
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
        sample_tokens_variants[str(req_size)] = samplers["sample_tokens"]
        stochastic_variants[str(req_size)] = samplers["sample_tokens_stochastic"]
        penalized_variants[str(req_size)] = samplers["sample_tokens_penalized"]

    return {
        "forward": forward_variants,
        "compute_logits": logits_variants,
        "sample_tokens": sample_tokens_variants,
        "sample_tokens_stochastic": stochastic_variants,
        "sample_tokens_penalized": penalized_variants,
    }


def _emit_declarative_weight_json(
    model: api.ExportableTextOnlyLLM,
    params_map: Mapping[str, Any],
    weight_slots: Mapping[str, int],
    output_dir: str,
    dp_size: int,
    tp_size: int,
    *,
    constants: frozenset[str] = frozenset(),
) -> None:
    """Emits `serving_checkpoint/weight.json` from HF metadata; loads no tensors.

    Weights point at checkpoint tensors (`sources`, with the checkpoint's full
    `source_shape`); constants carry their per-rank values (`axis_values`).
    Gathering those values is collective, so every rank runs this.
    """
    cfg = model.export_config
    ckpt_dir = os.path.join(output_dir, "serving_checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    hf_meta: Mapping[str, Any] = {}
    default_st = "model.safetensors"
    if cfg.hf_checkpoint_dir:
        hf_meta, default_st = weight_mapper.read_hf_safetensors_metadata(
            cfg.hf_checkpoint_dir
        )
    checkpoint_shapes = weight_mapper.read_hf_safetensors_shapes(
        cfg.hf_checkpoint_dir, hf_meta
    )
    torch_model = getattr(model, "model", None)

    weight_map: dict[str, Any] = {
        "mesh": {
            "mesh_shape": [dp_size, tp_size],
            "axis_names": ["dp", "tp"],
        }
    }
    # Source paths are relative to the checkpoint; record where it lives so a
    # runtime can find the safetensors files without being told.
    if cfg.hf_checkpoint_dir and os.path.isdir(cfg.hf_checkpoint_dir):
        weight_map["hf_checkpoint_path"] = os.path.abspath(cfg.hf_checkpoint_dir)
    for name, spec in params_map.items():
        slot_idx = weight_slots[name]
        shape_list = [int(d) for d in spec.shape]
        dt_str = tpu_lowering.dtype_name(spec.dtype)
        if name in constants:
            weight_map[name] = {
                "index": slot_idx,
                "shape": shape_list,
                "dtype": tpu_lowering.dtype_spec(dt_str).weight_json_name,
                "axis_values": {
                    "axis": "tp",
                    "values": model.constant_axis_values(name),  # type: ignore[attr-defined]
                },
            }
            continue
        sharding_val = weight_mapper.infer_weight_sharding(
            name,
            shape_list,
            model=torch_model,
            explicit_sharding=getattr(spec, "sharding", None),
            tp_size_hint=tp_size,
        )
        sources = weight_mapper.build_generic_weight_sources(
            name,
            shape_list,
            model=torch_model,
        )
        resolved_sources = weight_mapper.expand_or_resolve_weight_sources(
            name,
            {"shape": shape_list, "sources": sources},
            dict(hf_meta),
            default_st,
        )
        for src in resolved_sources:
            full_shape = checkpoint_shapes.get(src.get("source_tensor", ""))
            if full_shape is not None:
                src["source_shape"] = full_shape
        weight_map[name] = {
            "index": slot_idx,
            "shape": shape_list,
            "dtype": tpu_lowering.dtype_spec(dt_str).weight_json_name,
            "sharding": sharding_val,
            "sources": resolved_sources,
        }

    with open(os.path.join(ckpt_dir, "weight.json"), "w") as f:
        json.dump(weight_map, f, indent=2)


def _emit_model_companion_files(
    cfg: api.TextOnlyLLMExportConfig, output_dir: str
) -> dict[str, Any]:
    """Copies companion files into the artifact and returns the `serving_config` block.

    `tokenizer.json` and `generation_config.json` are copied when the checkpoint has
    them and declared in `serving_config` by path; the runtime reads no other file.
    """
    serving_config: dict[str, Any] = {
        "bucket_sizes": [int(b) for b in cfg.token_buckets],
        "paged_attention": {
            "block_size": cfg.block_size,
            "max_num_seqs": cfg.max_num_seqs,
            "max_num_pages": cfg.max_num_pages,
        },
    }

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

    if cfg.hf_checkpoint_dir and os.path.isdir(cfg.hf_checkpoint_dir):
        src_gen = os.path.join(cfg.hf_checkpoint_dir, "generation_config.json")
        if os.path.isfile(src_gen):
            dst_gen = os.path.join(output_dir, "generation_config.json")
            shutil.copyfile(src_gen, dst_gen)
            serving_config["generation_config"] = {"path": "generation_config.json"}

    return serving_config


def export(
    model: api.ExportableTextOnlyLLM,
    output_dir: str,
    *,
    subdir: str = "subgraphs",
    emit_weight_json: bool = True,
) -> None:
    """Exports an `ExportableTextOnlyLLM` to a serving artifact directory.

    Args:
        model: The model to export.
        output_dir: Artifact directory; created if missing.
        subdir: Subdirectory of `output_dir` for `.mlir` files (default `'subgraphs'`).
        emit_weight_json: When True, emits declarative `serving_checkpoint/weight.json`
            from HF metadata without loading tensor data.
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

    os.makedirs(output_dir, exist_ok=True)

    abs_spec, weight_slots = _export_abstract_params(
        params_map, output_dir, tp_size=tp_size, constants=constants, subdir=subdir
    )

    functions_dict: dict[str, dict[str, Any]] = {
        "abstract_params": {
            "default": abs_spec,
        }
    }

    llm_funcs = _export_text_only_llm_functions(
        model, params_map, weight_slots, output_dir, constants=constants, subdir=subdir
    )
    functions_dict.update(llm_funcs)
    serving_config = _emit_model_companion_files(cfg, output_dir)
    config_block = {
        "vocab_size": cfg.vocab_size,
        "eos_token_id": cfg.eos_token_id,
        "max_num_seqs": cfg.max_num_seqs,
        "max_num_pages": cfg.max_num_pages,
        "max_cu_seqs": cfg.max_cu_seqs,
        "block_size": cfg.block_size,
        "num_blocks": cfg.num_blocks,
        "max_model_len": cfg.max_model_len,
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": pp_size,
        "data_parallel_size": dp_size,
    }

    metadata_doc: dict[str, Any] = {
        "format_version": 2,
        "model_name": cfg.model_name,
        "target_tpu": cfg.target_tpu,
        "mesh": {
            "mesh_shape": [
                config_block["data_parallel_size"],
                config_block["tensor_parallel_size"],
            ],
            "axis_names": ["dp", "tp"],
        },
        "config": config_block,
        "compile_options": {
            "num_replicas": cfg.num_replicas,
            "num_partitions": cfg.num_partitions,
            "use_spmd_partitioning": cfg.use_spmd_partitioning,
        },
        "functions": functions_dict,
        "serving_config": serving_config,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata_doc, f, indent=2)

    validator.validate_exported_model_dir(output_dir)

    if emit_weight_json:
        _emit_declarative_weight_json(
            model,
            params_map,
            weight_slots,
            output_dir,
            dp_size,
            tp_size,
            constants=constants,
        )
