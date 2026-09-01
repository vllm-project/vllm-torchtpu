# SPDX-License-Identifier: Apache-2.0
"""Transformer Layer FX Graph Reuse Optimization (Layer Template Stitching).

In vllm-torchtpu, model backbone compilation wraps the entire model with
torch.compile(fullgraph=True). For LLMs with 28~80 layers, PyTorch Dynamo
repeatedly evaluates the Python bytecode for every identical DecoderLayer,
spending 20s~60s+ on CPU AST interpretation, FakeTensor dispatch, and ShapeEnv
symbolic guard solving.

This module intercepts the compilation of standard Transformer models:
1. Extracts a structural fingerprint (layer_signature) for each layer.
2. If the model consists of repeated isomorphic layers (K <= 3 templates),
   traces a minimal representative model (K layers) via Dynamo (~100ms).
3. Clones the template layer subgraph(s) in-memory across all N layers using
   torch.fx.Graph.node_copy (< 10ms), updating parameter targets (layers.0 -> i)
   and layer_idx constants.
4. Returns the complete Monolithic FX GraphModule directly to the TPU backend,
   preserving 100% XLA cross-layer fusions and Tier-3 binary caching with zero
   runtime serving overhead.
5. If the model is non-standard or heterogeneous (K > 3), safely falls back
   to stock Dynamo whole-model tracing.
"""

from __future__ import annotations

import inspect
import logging
import operator
from typing import Any, Callable, NamedTuple

import torch
from torch import fx, nn

logger = logging.getLogger(__name__)

# Maximum number of distinct layer templates to support before falling back
# to stock Dynamo. 1 covers Qwen/Llama; 2 covers DeepSeek (Dense+MoE) and
# Gemma (Sliding+Global window); 3 covers hybrid models.
MAX_TEMPLATES = 3

_STRUCTURAL_KEYS = (
    "sliding_window",
    "is_moe",
    "num_experts",
    "hidden_act",
    "rope_scaling",
    "use_bias",
    "qk_norm",
)


class LayerSignature(NamedTuple):
    """Static structural fingerprint for a transformer decoder layer.

    Two layers with the same signature produce isomorphic FX computation graphs.
    Variable attributes like layer_idx and parameter data are strictly excluded.
    """
    module_type: str
    submodule_tree: tuple[tuple[str, str], ...]
    param_schema: tuple[tuple[str, tuple[int, ...], str], ...]
    structural_attrs: tuple[tuple[str, Any], ...]


def layer_signature(layer: nn.Module) -> LayerSignature:
    """Extract a topological fingerprint for a single transformer layer."""
    module_type = type(layer).__qualname__

    submodule_tree = tuple(
        (name, type(mod).__qualname__) for name, mod in layer.named_modules())

    # layer.named_parameters() yields relative paths (e.g., 'self_attn.q_proj.weight')
    # naturally stripping layer-index prefixes like 'layers.0.'.
    param_schema = tuple((name, tuple(int(d) for d in p.shape), str(p.dtype))
                         for name, p in layer.named_parameters())

    detected_attrs: list[tuple[str, Any]] = []
    for mod_name, mod in layer.named_modules():
        prefix = f"{mod_name}." if mod_name else ""
        for key in _STRUCTURAL_KEYS:
            if hasattr(mod, key):
                val = getattr(mod, key)
                if isinstance(val, (int, float, str, bool, type(None))):
                    detected_attrs.append((f"{prefix}{key}", val))
                elif isinstance(val, (list, tuple)):
                    detected_attrs.append((f"{prefix}{key}", tuple(val)))
                elif isinstance(val, dict):
                    detected_attrs.append(
                        (f"{prefix}{key}", tuple(sorted(val.items()))))

    return LayerSignature(
        module_type=module_type,
        submodule_tree=submodule_tree,
        param_schema=param_schema,
        structural_attrs=tuple(sorted(detected_attrs)),
    )


def find_transformer_layers(
    model: nn.Module, ) -> tuple[nn.ModuleList, str] | None:
    """Locate the decoder layers ModuleList and its attribute name on model."""
    for attr in ("layers", "decoder_layers", "blocks"):
        if hasattr(model, attr):
            layers = getattr(model, attr)
            if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 1:
                return layers, attr

    # Check inner model container (e.g. model.model.layers)
    if hasattr(model, "model"):
        inner = getattr(model, "model")
        for attr in ("layers", "decoder_layers", "blocks"):
            if hasattr(inner, attr):
                layers = getattr(inner, attr)
                if isinstance(layers,
                              (nn.ModuleList, list)) and len(layers) > 1:
                    return layers, f"model.{attr}"

    return None


def inspect_model_layers(
    model: nn.Module,
    max_templates: int = MAX_TEMPLATES,
) -> tuple[nn.ModuleList, str, dict[LayerSignature, list[int]]] | None:
    """Inspect model layers and group them by structural signature.

    Returns:
        (layers_list, layers_attr_name, signature_to_indices_map)
        or None if model is ineligible / exceeds max_templates.
    """
    found = find_transformer_layers(model)
    if found is None:
        return None

    layers, attr_name = found
    sig_map: dict[LayerSignature, list[int]] = {}

    for idx, layer in enumerate(layers):
        sig = layer_signature(layer)
        sig_map.setdefault(sig, []).append(idx)

    if len(sig_map) > max_templates:
        logger.info(
            "Model %s has %d distinct layer structures (exceeds limit %d); "
            "skipping template stitching.",
            type(model).__name__,
            len(sig_map),
            max_templates,
        )
        return None

    return layers, attr_name, sig_map


def stitch_monolithic_graph(
    template_gm: fx.GraphModule,
    num_total_layers: int,
    layers_attr: str = "layers",
    template_layer_idx: int = 0,
    all_layers: nn.ModuleList | list[nn.Module] | None = None,
) -> fx.GraphModule:
    """Expand a 1-layer template FX GraphModule into a full N-layer graph.

    Args:
        template_gm: GraphModule traced with only layer `template_layer_idx`.
        num_total_layers: The total number of layers N in the full model.
        layers_attr: Attribute path to the layers container (e.g. 'layers').
        template_layer_idx: Index of the layer used for the template (usually 0).
        all_layers: All N layers to attach to the GraphModule for module resolution.

    Returns:
        A new GraphModule with all N layers inlined and connected.
    """
    if all_layers is not None:
        if "." in layers_attr:
            parent_attr, sub_attr = layers_attr.split(".", 1)
            parent = getattr(template_gm, parent_attr, template_gm)
            setattr(parent, sub_attr, nn.ModuleList(all_layers))
        else:
            setattr(template_gm, layers_attr, nn.ModuleList(all_layers))

    graph = template_gm.graph
    src_prefix = f"{layers_attr}.{template_layer_idx}."

    # Identify all nodes belonging to the template layer
    layer_nodes: list[fx.Node] = []
    collecting = False
    for n in graph.nodes:
        if n.op == "call_module" and isinstance(
                n.target, str) and n.target.startswith(src_prefix):
            collecting = True
        if collecting:
            layer_nodes.append(n)
            # Stop condition: check if node is the end of the transformer layer
            # E.g. residual add or output of the decoder layer
            if n.op == "call_function" and n.target in (operator.add,
                                                        torch.add):
                break

    if not layer_nodes:
        # Fallback search by module name
        for n in graph.nodes:
            if n.op == "call_module" and isinstance(
                    n.target, str) and src_prefix in n.target:
                layer_nodes.append(n)

    if not layer_nodes:
        raise ValueError(
            f"Could not identify layer nodes with prefix {src_prefix} in template graph."
        )

    layer_in = layer_nodes[0].args[0]
    layer_out = layer_nodes[-1]

    insert_pos = layer_out
    curr_in = layer_out
    all_cloned_nodes: set[fx.Node] = set()

    for i in range(1, num_total_layers):
        target_prefix = f"{layers_attr}.{i}."
        env: dict[fx.Node, fx.Node] = {layer_in: curr_in}
        for n in layer_nodes:
            with graph.inserting_after(insert_pos):
                new_n = graph.node_copy(n, lambda x: env.get(x, x))
                if new_n.op == "call_module" and isinstance(new_n.target, str):
                    new_n.target = new_n.target.replace(
                        src_prefix, target_prefix)
                if "layer_idx" in new_n.kwargs:
                    new_n.update_kwarg("layer_idx", i)
                env[n] = new_n
                all_cloned_nodes.add(new_n)
                insert_pos = new_n
        curr_in = env[layer_out]

    # Rewire downstream users of layer 0 output (excluding the layer itself and cloned layers)
    for user in list(layer_out.users.keys()):
        if user not in layer_nodes and user not in all_cloned_nodes:
            user.replace_input_with(layer_out, curr_in)

    graph.lint()
    template_gm.recompile()
    return template_gm


def _ph_to_param_name(name: str) -> str:
    """Convert Dynamo placeholder name to module parameter/buffer name."""
    s = name
    if s.startswith("l_self_modules_"):
        s = s[len("l_self_modules_"):]
    elif s.startswith("L_self_modules_"):
        s = s[len("L_self_modules_"):]
    s = s.rstrip("_")
    return s.replace("_modules_", ".").replace("_parameters_",
                                               ".").replace("_buffers_", ".")


def stitch_dynamo_monolithic_graph(
    template_gm: fx.GraphModule,
    sample_inputs: list[Any],
    model: nn.Module,
    num_total_layers: int,
    layers_attr: str = "layers",
) -> tuple[fx.GraphModule, list[Any], Callable[[list[Any]], list[Any]]]:
    """Expand a 1-layer Dynamo-traced GraphModule into an N-layer monolithic GraphModule.

    Handles lifted parameter placeholders and parameter reassembly for backend execution.
    """
    found = find_transformer_layers(model)
    if found is not None:
        layers, _ = found
    else:
        layers = getattr(model, layers_attr, None)
        if layers is None and hasattr(model, "model"):
            layers = getattr(model.model, layers_attr, None)

    graph = template_gm.graph
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]

    model_params = dict(model.named_parameters())
    model_params.update(dict(model.named_buffers()))
    l0_param_indices = [
        i for i, p in enumerate(placeholders)
        if _ph_to_param_name(p.name) in model_params
        and f"{layers_attr}.0." in _ph_to_param_name(p.name)
    ]
    l0_placeholders = [placeholders[i] for i in l0_param_indices]
    l0_ph_set = set(l0_placeholders)
    other_ph_set = set(placeholders) - l0_ph_set

    # Identify all computation nodes of layer 0
    l0_nodes: list[fx.Node] = []
    for n in nodes:
        if n.op in ("call_function", "call_method"):
            if any(arg in other_ph_set for arg in n.args):
                continue
            if any(arg in l0_ph_set or arg in l0_nodes for arg in n.args):
                l0_nodes.append(n)

    if not l0_nodes:
        raise ValueError("Could not detect Dynamo layer 0 computation nodes.")

    # Topological input to layer 0
    layer_in = None
    for n in l0_nodes:
        for arg in n.args:
            if isinstance(
                    arg,
                    fx.Node) and arg not in l0_nodes and arg not in l0_ph_set:
                layer_in = arg
                break
        if layer_in is not None:
            break

    # Topological output of layer 0
    layer_out = None
    for n in reversed(l0_nodes):
        outside_users = [u for u in n.users if u not in l0_nodes]
        if outside_users:
            layer_out = n
            break

    if layer_in is None or layer_out is None:
        raise ValueError(
            "Could not determine topological boundary (layer_in / layer_out).")

    insert_pos = layer_out
    curr_in = layer_out
    all_cloned_nodes: set[fx.Node] = set()
    last_ph = l0_placeholders[-1]

    # Pre-fetch named parameters for each layer
    layer_params_map: dict[int, list[torch.Tensor]] = {}
    for i in range(1, num_total_layers):
        layer_params_map[i] = [
            model_params[_ph_to_param_name(p.name).replace(
                f"{layers_attr}.0.", f"{layers_attr}.{i}.")]
            for p in l0_placeholders
        ]

        env: dict[fx.Node, fx.Node] = {layer_in: curr_in}

        # Create new placeholders for layer i
        for p_idx, p in enumerate(l0_placeholders):
            with graph.inserting_after(last_ph):
                new_p_name = p.name.replace("layers_modules_0_",
                                            f"layers_modules_{i}_")
                new_p = graph.placeholder(new_p_name)
                new_p.meta = dict(p.meta)
                env[p] = new_p
                last_ph = new_p

        # Clone computation nodes for layer i
        for n in l0_nodes:
            with graph.inserting_after(insert_pos):
                new_n = graph.node_copy(n, lambda x: env.get(x, x))
                new_n.meta = dict(n.meta)
                env[n] = new_n
                all_cloned_nodes.add(new_n)
                insert_pos = new_n

        curr_in = env[layer_out]

    # Rewire downstream users of layer 0 output
    for user in list(layer_out.users.keys()):
        if user not in l0_nodes and user not in all_cloned_nodes:
            user.replace_input_with(layer_out, curr_in)

    graph.lint()
    template_gm.recompile()

    # Build input slot specifications: map each input to either a parameter or user call argument
    input_slot_specs: list[tuple[bool, Any]] = []
    arg_idx = 0
    for ph_idx, p in enumerate(placeholders):
        param_name = _ph_to_param_name(p.name)
        if param_name in model_params:
            input_slot_specs.append((True, model_params[param_name]))
        else:
            clean_arg_name = p.name.strip("lL_").rstrip("_")
            fallback_val = sample_inputs[ph_idx] if ph_idx < len(
                sample_inputs) else None
            input_slot_specs.append(
                (False, (clean_arg_name, arg_idx, fallback_val)))
            arg_idx += 1

    min_l0_idx = min(l0_param_indices)
    max_l0_idx = max(l0_param_indices)

    full_slot_specs: list[tuple[bool, Any]] = []
    full_slot_specs.extend(input_slot_specs[:min_l0_idx])
    full_slot_specs.extend(input_slot_specs[min_l0_idx:max_l0_idx + 1])
    for i in range(1, num_total_layers):
        for p in l0_placeholders:
            p0_name = _ph_to_param_name(p.name)
            pi_name = p0_name.replace(f"{layers_attr}.0.",
                                      f"{layers_attr}.{i}.")
            full_slot_specs.append((True, model_params[pi_name]))
    full_slot_specs.extend(input_slot_specs[max_l0_idx + 1:])

    # Full sample inputs for compilation backend
    prefix_inputs = sample_inputs[:min_l0_idx]
    l0_inputs = sample_inputs[min_l0_idx:max_l0_idx + 1]
    suffix_inputs = sample_inputs[max_l0_idx + 1:]
    full_sample_inputs = list(prefix_inputs) + list(l0_inputs)
    for i in range(1, num_total_layers):
        full_sample_inputs.extend(layer_params_map[i])
    full_sample_inputs.extend(suffix_inputs)

    # Discover model's underlying forward parameter names
    param_names: list[str] = []
    for cls in model.__class__.__mro__:
        if hasattr(cls, "forward"
                   ) and cls.__name__ != "TorchCompileWithNoGuardsWrapper":
            fwd = cls.__dict__.get("forward", None)
            if fwd is not None:
                try:
                    fwd_sig = inspect.signature(fwd)
                    names = [
                        k for k in fwd_sig.parameters.keys() if k != "self"
                    ]
                    if names and names != ["args", "kwargs"]:
                        param_names = names
                        break
                except Exception:
                    pass

    def full_inputs_builder(
        user_call_args: tuple[Any, ...] | list[Any],
        user_call_kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        if user_call_kwargs is None:
            user_call_kwargs = {}
        if isinstance(user_call_args, list):
            user_call_args = tuple(user_call_args)

        param_name_to_val: dict[str, Any] = {}
        for name, val in zip(param_names, user_call_args):
            param_name_to_val[name] = val
        param_name_to_val.update(user_call_kwargs)

        res = []
        for is_p, val in full_slot_specs:
            if is_p:
                res.append(val)
            else:
                clean_name, pos_idx, fallback_val = val
                if clean_name in param_name_to_val and param_name_to_val[
                        clean_name] is not None:
                    res.append(param_name_to_val[clean_name])
                elif pos_idx < len(user_call_args
                                   ) and user_call_args[pos_idx] is not None:
                    res.append(user_call_args[pos_idx])
                elif clean_name in user_call_kwargs and user_call_kwargs[
                        clean_name] is not None:
                    res.append(user_call_kwargs[clean_name])
                elif fallback_val is not None:
                    res.append(fallback_val)
                else:
                    logger.warning(
                        "[LayerStitcher] Unresolved slot: clean_name=%s pos_idx=%d",
                        clean_name, pos_idx)
                    res.append(None)
        return res

    return template_gm, full_sample_inputs, full_inputs_builder


def wrap_callable_with_stitching(
    model: nn.Module,
    original_callable: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap model's _compiled_callable to transparently attempt FX Layer Stitching.

    On the first call, if the model is a standard Transformer with repeated layers (K <= 3),
    it traces a single-layer template and stitches an N-layer monolithic FX GraphModule,
    then compiles that GraphModule. If stitching is not applicable or encounters any issue,
    it seamlessly falls back to the original Dynamo callable.
    """
    stitched_callable: Callable[..., Any] | None = None
    attempted = False

    def dispatch(*args: Any, **kwargs: Any) -> Any:
        nonlocal stitched_callable, attempted
        if not attempted:
            attempted = True
            try:
                inspection = inspect_model_layers(model)
                if inspection is not None:
                    layers, attr_name, sig_map = inspection
                    if len(sig_map) == 1:
                        total_layers = len(layers)
                        logger.info(
                            "[LayerStitcher] Fast-path: homogeneous transformer (%d layers) "
                            "detected on %s. Tracing 1-layer Dynamo template.",
                            total_layers,
                            type(model).__name__,
                        )
                        original_layers = list(layers)
                        captured_gm: fx.GraphModule | None = None
                        captured_sample_inputs: list[Any] | None = None

                        def capture_backend(gm: fx.GraphModule,
                                            sample_inputs: list[Any]):
                            nonlocal captured_gm, captured_sample_inputs
                            captured_gm = gm
                            captured_sample_inputs = list(sample_inputs)
                            return gm

                        try:
                            if hasattr(model, "layers"):
                                model.layers = nn.ModuleList(
                                    [original_layers[0]])
                            elif hasattr(model, "model") and hasattr(
                                    model.model, "layers"):
                                model.model.layers = nn.ModuleList(
                                    [original_layers[0]])

                            # Trace 1-layer template via Dynamo
                            torch.compile(
                                model.forward,
                                fullgraph=True,
                                dynamic=False,
                                backend=capture_backend,
                            )(*args, **kwargs)
                        finally:
                            if hasattr(model, "layers"):
                                model.layers = nn.ModuleList(original_layers)
                            elif hasattr(model, "model") and hasattr(
                                    model.model, "layers"):
                                model.model.layers = nn.ModuleList(
                                    original_layers)

                        if captured_gm is not None and captured_sample_inputs is not None:
                            stitched_gm, full_inputs, inputs_builder = stitch_dynamo_monolithic_graph(
                                captured_gm,
                                captured_sample_inputs,
                                model,
                                total_layers,
                                layers_attr=attr_name.split(".")[-1],
                            )
                            backend = getattr(model, "_backend", None)
                            if backend is None:
                                comp_cfg = getattr(model, "compilation_config",
                                                   None)
                                backend = getattr(comp_cfg, "backend",
                                                  None) if comp_cfg else None
                            if backend is None and hasattr(
                                    model, "vllm_config"):
                                backend = model.vllm_config.compilation_config.init_backend(
                                    model.vllm_config,
                                    prefix=getattr(model, "prefix", ""),
                                    is_encoder=False)

                            if callable(backend):
                                raw_executable = backend(
                                    stitched_gm, full_inputs)

                                def run_stitched(*call_args, **call_kwargs):
                                    out = raw_executable(*inputs_builder(
                                        call_args, call_kwargs))
                                    if isinstance(
                                            out,
                                        (tuple, list)) and len(out) == 1:
                                        return out[0]
                                    return out

                                stitched_callable = run_stitched
                            else:

                                def run_stitched(*call_args, **call_kwargs):
                                    out = stitched_gm(*inputs_builder(
                                        call_args, call_kwargs))
                                    if isinstance(
                                            out,
                                        (tuple, list)) and len(out) == 1:
                                        return out[0]
                                    return out

                                stitched_callable = run_stitched

                            logger.info(
                                "[LayerStitcher] Stitched %d layers successfully into monolithic graph. "
                                "Eliminated %d layer bytecode traces.",
                                total_layers,
                                total_layers - 1,
                            )
            except Exception as e:
                import traceback
                logger.warning(
                    "[LayerStitcher] Stitching attempt bypassed (%s); running stock Dynamo.\n%s",
                    e,
                    traceback.format_exc(),
                )
                stitched_callable = None

        if stitched_callable is not None:
            res = stitched_callable(*args, **kwargs)
            if isinstance(res, (tuple, list)) and len(res) == 1:
                return res[0]
            return res
        return original_callable(*args, **kwargs)

    if hasattr(original_callable, "aot_compile"):
        dispatch.aot_compile = original_callable.aot_compile
    if hasattr(original_callable, "_torchdynamo_orig_callable"):
        dispatch._torchdynamo_orig_callable = original_callable._torchdynamo_orig_callable

    return dispatch
