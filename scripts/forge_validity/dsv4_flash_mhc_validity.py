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
r"""Validity adapter for the DeepSeek-V4 mHC kernels at Flash DP8/EP8.

Run it through perfkit, never directly:

    perfkit validity run --adapter <this file> --arm <checkout> \
        --python "$(perfkit env ensure --jax 0.11.2 --libtpu 0.0.50)"

`perfkit/validity_arm.py` defines the hooks and the two checks. Nothing here
uses a TPU: compiles are ahead of time for the tpu7x topology on the CPU, and
trip counts are read while tracing on the CPU.

What is compiled
----------------
The manifold hyper-connection (mHC) kernels in `kernels/deepseek_v4/mhc/`,
through the arm's own op file,
`layers/adapter/custom_ops/deepseek_v4/deepseek_v4_mhc_op.py`.
`DeepseekV4DecoderLayer.mhc_ops` in `models/vllm/deepseek_v4/model.py`
builds the ops once with `get_mhc_ops(...)`, and every step calls three
kinds of them:

* `pre`, once: the chain start, layer 0's attention pre
  (`pre_kernel.mhc_pre`: the `_mixes_collapse_kernel` Pallas call, then
  the gates and Sinkhorn in XLA).
* `fused`, at every other sublayer seam: one sublayer's post with the next
  sublayer's pre (`fused_post_pre_kernel.mhc_fused_post_pre`:
  `_fused_kernel`, then the same XLA gates). Its Pallas call is the one
  with outputs `bf16[T, 16384]`, `f32[T, 24]`, `f32[T, 1]` and
  `bf16[T, 4096]`.
* `post`, once: the post the last layer deferred
  (`DeepseekV4Model.mhc_post_op`, `post_kernel.mhc_post`: `_post_kernel`).

`setup` runs the arm's `get_mhc_ops` with the arguments
`DeepseekV4DecoderLayer.mhc_ops` passes, each read from `model.py` and
evaluated against the checkpoint's config (`rms_norm_eps` 1e-6, `hc_eps`
1e-6, `hc_sinkhorn_iters` 20, no `hc_post_alpha`, so the layer's 2.0), with
the op registration (`pallas.jax_op`) replaced by a recorder. What it
records for each op is the JAX callable the server traces, with the gate
constants bound as the server binds them, its input partition specs and
its donated arguments. So every static argument the kernels see is the
arm's: `token_block_size`, `gemm_precision` and `vmem_limit_bytes` are the
kernels' defaults unless the op passes them (golden: blocks of 64 tokens
for `pre` and `post`, 32 for `fused`, the 3-pass `"highest"` GEMM, a 100
MiB scoped-VMEM limit), and an arm that retunes any of them, in the
kernels or in the op, is compiled as its server calls it. The gate
constants reach the Mosaic kernels as two scalars (`rms_eps`,
`hc_pre_eps`); the rest only change the XLA epilogue. The torch modules
the op file imports are stubs while it loads, and an op is refused if
its partition specs name a mesh axis: the program compiled here is one
engine's, on one core.

Shapes
------
The serving config is `scripts/vllm/benchmarking/configs/
deepseek-v4-flash-dp8-ep.sh`: TP1 x DP8 with EP, `--max-num-batched-tokens
1024`, `--max-num-seqs 256`. Every mHC tensor is token-major and the op
partitions none of them, so each engine runs the kernels on its own batch.

* Token buckets. An engine pads a step to one of 16, 32, ..., 512, 1024
  tokens (`_get_token_paddings` in `platforms/tpu_platform.py`, doubling
  from 16 to the token budget; the config sets no extra buckets), and DP
  lockstep pads every engine to the same bucket. Prefill and decode steps
  run the same program at a bucket.
* Kinds. `pre`, `fused` and `post` at every bucket.
* Static model geometry: `hc_mult` 4, hidden 4096, so the residual is
  `bf16[T, 4, 4096]` (`bf16[T, 16384]` in the kernels) and the mix
  projection `fn` is `f32[24, 16384]` (24 = 4 x (4 + 2)), with `hc_scale`
  `f32[3]` and `hc_base` `f32[24]`. The sublayer output `x` is
  `bf16[T, 4096]`, `post_mix` `f32[T, 4, 1]`, `res_mix` `f32[T, 4, 4]`.

The token count is the only thing that sizes anything: each kernel's grid
is `T / block` steps of a block of `min(block, T)` tokens, fixed at trace
time (golden: 1 step up to 64 tokens, 16 at 1024 for `pre` and `post`; 1
step up to 32, 32 at 1024 for `fused`). The kernels take no runtime
scalars, so no runtime value changes a grid, a branch or a block: the
rows a step leaves as padding only change values.

That is 21 shapes, 7 buckets x 3 kinds, which covers every program the
server compiles for these kernels.

Checks
------
`compile` traces each kind's recorded callable at each shape and compiles
it for tpu7x, donating what the op donates. `limit` refuses a program that
launches no Mosaic kernel, since such a compile checked nothing.

`trip_counts` traces each shape again and reads every `emit_pipeline` call
made while tracing: its grid, and the buffer count and lookahead of each
input spec (2 buffers and no lookahead for a spec without a pipeline
mode). Golden has none: each kernel is a single `pallas_call` whose grid
moves its blocks, so `pipeline_trip_count` passes with no records. A
`pallas_call` input spec with a deeper pipeline mode is not modelled here.
The check guards a candidate that brings in an `emit_pipeline`: one whose
grid is fixed at trace time gives one record per input pipe, and one of 4
or more buffers whose grid is sized at run time, or split across cores,
makes `trip_counts` raise rather than guess (3 or fewer give no record).

Calibration
-----------
With this adapter on the CPU (jax 0.11.2, libtpu 0.0.50), against golden
5508724a3 and three copies of it, each broken in a way a fixed-shape
benchmark at the common buckets can miss:

  * golden passes both checks: 21 shapes compile (the largest temp
    allocation 72412992 bytes, at fused-t1024), and no pipe records.
  * The fused kernel's token block grown with the bucket (`max(32, T //
    2)`) fails `aot_all_shapes` at fused-t1024 only (VMEM out of memory,
    116.72M of 63.94M; its 256-token block at 512 still fits), and passes
    `pipeline_trip_count`.
  * The post kernel asserting whole 32-token blocks, true at every bucket
    but 16, fails both checks at post-t16 only.
  * The post kernel staging each block through a 4-buffer `emit_pipeline`
    of 16-row sub-blocks compiles at all 21 shapes and fails
    `pipeline_trip_count` at post-t32 only (B=4, N=2); its grid is 1 step
    at 16 tokens and 4 from 64 up.

A run takes about 20 seconds (21 compiles of under 2 seconds each, then
21 traces).
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import functools
import importlib.util
import math
import operator
import os
import sys
import types
from pathlib import Path

# The package root: the tree a verdict names must cover the op and model
# files the static arguments are read from as well as the kernels.
KERNEL_DIR = "src/vllm_torchtpu"
MHC_DIR = f"{KERNEL_DIR}/kernels/deepseek_v4/mhc"
OP_FILE = f"{KERNEL_DIR}/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_mhc_op.py"
OP_MODULE = "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op"
MODEL_FILE = f"{KERNEL_DIR}/models/vllm/deepseek_v4/model.py"
LAYER_CLASS = "DeepseekV4DecoderLayer"
OPS_BUILDER = "get_mhc_ops"
TOPOLOGY = "tpu7x:2x2x1"

# The checkpoint's config.json fields the layer reads for its gates.
HF_CONFIG = {
    "hidden_size": 4096,
    "hc_mult": 4,
    "rms_norm_eps": 1e-6,
    "hc_eps": 1e-6,
    "hc_sinkhorn_iters": 20,
}
HC_MULT = HF_CONFIG["hc_mult"]
HIDDEN = HF_CONFIG["hidden_size"]
HC_MULT3 = HC_MULT * (HC_MULT + 2)
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
KINDS = ("pre", "fused", "post")
SHAPES = {
    f"{kind}-t{tokens}": {"kind": kind, "tokens": tokens}
    for kind in KINDS
    for tokens in TOKEN_PADDINGS
}
# What an expression read from the model may call.
RULE_BUILTINS = {f.__name__: f for f in (getattr, float, int, min, max)}

# A pipe with at most this many buffers starts no copy that no step waits
# for at any trip count (the window 2 <= N <= B - 2 is empty).
SAFE_BUFFERS = 3


def _package(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)] if path is not None else []
    sys.modules[name] = mod
    return mod


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_kernels(root: Path):
    """Stub the arm's packages so its kernels import without torch.

    The package's own `__init__` imports torch and the serving stack, which
    a compile-only environment does not have. `env_override` is loaded
    because it sets the server's LIBTPU_INIT_ARGS and applies its
    import-time Pallas patches, so the program compiled here is the one the
    server compiles. The kernels resolve through the stub packages, whose
    paths are the arm's.
    """
    pkg = root / KERNEL_DIR
    top = _package("vllm_torchtpu", pkg)
    for name in ("envs", "logger", "patch_registry"):
        setattr(top, name, _module(f"vllm_torchtpu.{name}", pkg / f"{name}.py"))
    _module("vllm_torchtpu.env_override", pkg / "env_override.py")
    _package("vllm_torchtpu.kernels", pkg / "kernels")
    _package("vllm_torchtpu.kernels.deepseek_v4", pkg / "kernels" / "deepseek_v4")
    _package("vllm_torchtpu.kernels.deepseek_v4.mhc", root / MHC_DIR)


@dataclasses.dataclass
class _Op:
    """One `pallas.jax_op` registration, as the op file made it."""

    name: str
    fn: object
    donate_argnums: tuple
    input_partition_specs: tuple | None

    def register_fake(self, fake):
        return fake


def _jax_op(
    name,
    fn=None,
    /,
    *,
    donate_argnums=None,
    mesh=None,
    input_partition_specs=None,
    output_partition_specs=None,
):
    if fn is None:
        raise RuntimeError(f"{name}: the decorator form of jax_op is not recorded")
    return _Op(name, fn, tuple(donate_argnums or ()), input_partition_specs)


def _torch_stub(name):
    """A module whose every attribute is an inert placeholder type."""
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda attr: type(attr, (), {})
    return mod


def _load_op(root: Path):
    """Import the arm's op file with its torch-side imports stubbed.

    Only `pallas.jax_op` and `get_pallas_op_mesh` are called, and only when
    the ops are built. The torch stubs leave `sys.modules` once the file
    has loaded, so nothing traced later can see them.
    """
    pallas = types.ModuleType("torch_tpu._internal.pallas")
    pallas.jax_op = _jax_op
    internal = types.ModuleType("torch_tpu._internal")
    internal.pallas = pallas
    context = types.ModuleType("vllm_torchtpu.models.vllm.vllm_model_wrapper_context")
    context.get_pallas_op_mesh = lambda: None
    stubs = {
        "torch": _torch_stub("torch"),
        "torch_tpu": types.ModuleType("torch_tpu"),
        "torch_tpu._internal": internal,
        "torch_tpu._internal.pallas": pallas,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    # Empty paths: nothing real is imported through these.
    for name in (
        "vllm_torchtpu.models",
        "vllm_torchtpu.models.vllm",
        "vllm_torchtpu.layers",
        "vllm_torchtpu.layers.adapter",
        "vllm_torchtpu.layers.adapter.custom_ops",
        "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4",
    ):
        _package(name, None)
    sys.modules[context.__name__] = context
    try:
        return _module(OP_MODULE, root / OP_FILE)
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _only(nodes, what, where):
    if len(nodes) != 1:
        raise RuntimeError(f"{where} has {len(nodes)} {what}, expected one")
    return nodes[0]


def _self_attrs(expr):
    return {
        n.attr
        for n in ast.walk(expr)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "self"
    }


def _gate_args(model_file: Path):
    """The arguments `DeepseekV4DecoderLayer.mhc_ops` passes `get_mhc_ops`.

    Each is an expression over `self`; every `self.<attr>` in it is the
    value of the one `self.<attr> = ...` assignment in the layer's
    `__init__`, evaluated against the checkpoint's config.
    """
    where = str(model_file)
    tree = ast.parse(model_file.read_text(), where)
    cls = _only(
        [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == LAYER_CLASS],
        f"class {LAYER_CLASS}",
        where,
    )
    call = _only(
        [
            n
            for n in ast.walk(cls)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == OPS_BUILDER
        ],
        f"calls of {OPS_BUILDER} in {LAYER_CLASS}",
        where,
    )
    if any(isinstance(a, ast.Starred) for a in call.args) or any(
        kw.arg is None for kw in call.keywords
    ):
        raise RuntimeError(f"{LAYER_CLASS} calls {OPS_BUILDER} with * or ** args")
    init = _only(
        [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        ],
        f"{LAYER_CLASS}.__init__",
        where,
    )
    scope = {"__builtins__": RULE_BUILTINS}
    config = types.SimpleNamespace(**HF_CONFIG)

    def attr_value(attr):
        value = _only(
            [
                s.value
                for s in ast.walk(init)
                if isinstance(s, ast.Assign) and attr in _self_attrs_of(s.targets)
            ],
            f"assignments of self.{attr} in {LAYER_CLASS}.__init__",
            where,
        )
        code = builtins.compile(ast.Expression(value), where, "eval")
        return eval(code, scope, {"config": config})

    def evaluate(expr):
        me = types.SimpleNamespace(**{a: attr_value(a) for a in _self_attrs(expr)})
        code = builtins.compile(ast.Expression(expr), where, "eval")
        return eval(code, scope, {"self": me})

    args = [evaluate(a) for a in call.args]
    kwargs = {kw.arg: evaluate(kw.value) for kw in call.keywords}
    return args, kwargs


def _self_attrs_of(targets):
    """The `self.<attr>` names an assignment's targets set directly."""
    return {
        t.attr
        for t in targets
        if isinstance(t, ast.Attribute)
        and isinstance(t.value, ast.Name)
        and t.value.id == "self"
    }


def _check_unsharded(op: _Op):
    for spec in op.input_partition_specs or ():
        if any(axis is not None for axis in spec):
            raise RuntimeError(
                f"{op.name} partitions an input over the mesh ({spec}); this "
                "adapter compiles one engine's unsharded program"
            )


def setup(root):
    # CPU only: jax must never open the host's TPU, and the compile-only
    # topology does not need the libtpu lock.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    handle = types.SimpleNamespace(pipelines=None, device=None)
    # Installed before the kernels are imported, so a kernel that binds
    # `emit_pipeline` by name at import time gets the recorder too.
    _install_pipeline_recorder(handle)
    _load_kernels(root)
    op_module = _load_op(root)
    args, kwargs = _gate_args(root / MODEL_FILE)
    ops = getattr(op_module, OPS_BUILDER)(*args, **kwargs)
    handle.ops = {kind: getattr(ops, kind) for kind in KINDS}
    for op in handle.ops.values():
        _check_unsharded(op)
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, "__file__", None)
        if (
            name.startswith("vllm_torchtpu.")
            and path
            and not Path(path).resolve().is_relative_to(root)
        ):
            raise RuntimeError(f"{name} was imported from {path}, outside {root}")
    handle.kernel = sys.modules["vllm_torchtpu.kernels.deepseek_v4.mhc.pre_kernel"]

    import jax
    import jax.numpy as jnp

    handle.jax, handle.jnp = jax, jnp
    return handle


def origin(handle) -> str:
    return handle.kernel.__file__


def _arg_specs(handle, params):
    """The arrays the model hands each op, in its call order."""
    jnp, t = handle.jnp, params["tokens"]
    x = ((t, HIDDEN), jnp.bfloat16)
    residual = ((t, HC_MULT, HIDDEN), jnp.bfloat16)
    post_mix = ((t, HC_MULT, 1), jnp.float32)
    res_mix = ((t, HC_MULT, HC_MULT), jnp.float32)
    gates = [
        ((HC_MULT3, HC_MULT * HIDDEN), jnp.float32),  # fn
        ((3,), jnp.float32),  # hc_scale
        ((HC_MULT3,), jnp.float32),  # hc_base
    ]
    return {
        "pre": [residual, *gates],
        "fused": [x, residual, post_mix, res_mix, *gates],
        "post": [x, residual, post_mix, res_mix],
    }[params["kind"]]


def _tpu7x_mesh():
    """Trace as for one tpu7x core, so the kernels' chip queries answer for it."""
    from jax._src import mesh as mesh_lib

    return mesh_lib.use_abstract_mesh(
        mesh_lib.AbstractMesh(
            (),
            (),
            abstract_device=mesh_lib.AbstractDevice(
                device_kind="TPU7x", num_cores=1, platform="tpu"
            ),
        )
    )


def _trace(handle, params):
    jax = handle.jax
    if handle.device is None:
        from jax.experimental import topologies

        topology = topologies.get_topology_desc(TOPOLOGY, platform="tpu")
        handle.device = topology.devices[0]
    op = handle.ops[params["kind"]]
    specs = _arg_specs(handle, params)
    declared = op.input_partition_specs
    if declared is not None and len(declared) != len(specs):
        raise RuntimeError(
            f"{op.name} declares {len(declared)} inputs; the model passes {len(specs)}"
        )
    sharding = jax.sharding.SingleDeviceSharding(handle.device)
    args = [
        jax.ShapeDtypeStruct(shape, dtype, sharding=sharding) for shape, dtype in specs
    ]
    fn = jax.jit(op.fn, donate_argnums=op.donate_argnums)
    with _tpu7x_mesh():
        return fn.trace(*args)


def _pallas_calls(jaxpr):
    """`name grid=... block=...` of every pallas_call in `jaxpr`, nested too.

    The block is the first input's, the token block in every mHC kernel.
    """
    from jax._src import core

    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            gm = eqn.params["grid_mapping"]
            info = getattr(eqn.params["jaxpr"], "debug_info", None)
            name = getattr(info, "func_name", None) or "?"
            block = tuple(
                getattr(d, "block_size", d) for d in gm.block_mappings[0].block_shape
            )
            found.append(f"{name} grid={tuple(gm.grid)} block={block}")
        for value in eqn.params.values():
            for sub in value if isinstance(value, (list, tuple)) else (value,):
                if isinstance(sub, core.ClosedJaxpr):
                    found += _pallas_calls(sub.jaxpr)
                elif isinstance(sub, core.Jaxpr):
                    found += _pallas_calls(sub)
    return found


def compile(handle, name, params):
    traced = _trace(handle, params)
    calls = _pallas_calls(traced.jaxpr.jaxpr)
    lowered = traced.lower(lowering_platforms=("tpu",))
    mem = lowered.compile().memory_analysis()
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "mosaic_kernels": lowered.as_text().count("tpu_custom_call"),
        "kernels": "; ".join(calls),
    }


def limit(name, result):
    if not result["mosaic_kernels"]:
        return (
            "the compiled program launches no Mosaic kernel, so the compile "
            "checked nothing"
        )
    return None


def _input_pipes(specs, label="in"):
    """`(label, buffer_count, lookahead)` for every input block spec.

    A spec without a pipeline mode is double-buffered. The walk follows
    lists, tuples, dicts and dataclasses, since a kernel may wrap its specs.
    """
    from jax.experimental import pallas as pl

    if isinstance(specs, pl.BlockSpec):
        mode = specs.pipeline_mode
        count = getattr(mode, "buffer_count", None) or 2
        return [(label, int(count), bool(getattr(mode, "use_lookahead", False)))]
    found = []
    if isinstance(specs, (list, tuple)):
        for i, s in enumerate(specs):
            found += _input_pipes(s, f"{label}[{i}]")
    elif isinstance(specs, dict):
        for k, s in specs.items():
            found += _input_pipes(s, f"{label}[{k!r}]")
    elif dataclasses.is_dataclass(specs) and not isinstance(specs, type):
        for f in dataclasses.fields(specs):
            found += _input_pipes(getattr(specs, f.name), f"{label}.{f.name}")
    return found


def _qualname(body):
    """The body's own name, through any `functools.partial` around it."""
    while isinstance(body, functools.partial):
        body = body.func
    return getattr(body, "__qualname__", type(body).__name__)


def _install_pipeline_recorder(handle):
    """Wrap `emit_pipeline` to note each call while `handle.pipelines` is a list."""
    from jax._src.pallas.mosaic import pipeline as jax_pipeline
    from jax.experimental.pallas import tpu as pltpu

    real = jax_pipeline.emit_pipeline

    @functools.wraps(real)
    def emit_pipeline(body, *, grid, in_specs=(), **kwargs):
        if handle.pipelines is not None:
            split = (
                kwargs.get("core_axis") is not None
                or kwargs.get("core_axis_name") is not None
            )
            handle.pipelines.append(
                (
                    _qualname(body),
                    tuple(grid),
                    split,
                    _input_pipes(in_specs),
                )
            )
        return real(body, grid=grid, in_specs=in_specs, **kwargs)

    jax_pipeline.emit_pipeline = pltpu.emit_pipeline = emit_pipeline


def _static_dim(d):
    """The grid dimension as an int, or None if it is only known at run time."""
    try:
        return operator.index(d)
    except TypeError:
        return None


def trip_counts(handle, name, params):
    # A fresh trace: the nested jits are cached from `compile`, and a cached
    # trace calls no `emit_pipeline`.
    handle.jax.clear_caches()
    handle.pipelines = []
    try:
        _trace(handle, params)
        calls = handle.pipelines
    finally:
        handle.pipelines = None
    records = []
    for i, (body, grid, split, pipes) in enumerate(calls):
        dims = [_static_dim(d) for d in grid]
        static = not split and None not in dims
        for pipe, b, lookahead in pipes:
            if not static:
                if b > SAFE_BUFFERS:
                    why = "split across cores" if split else "sized at run time"
                    raise RuntimeError(
                        f"pipeline {i} ({body}) {pipe} has {b} buffers and a grid "
                        f"{why}; this adapter reads trip counts at trace time only"
                    )
                continue
            records.append(
                {
                    "pipe": f"pipeline{i} (body {body}) {pipe}",
                    "buffer_count": b,
                    "trip_count": math.prod(dims),
                    "lookahead": lookahead,
                    "input": (
                        f"any runtime input (grid {tuple(dims)} fixed at trace time)"
                    ),
                }
            )
    return records
