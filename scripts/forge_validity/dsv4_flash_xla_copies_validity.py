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

r"""Validity adapter for the DeepSeek-V4 mHC ops and the expert-parallel MoE
path, the two places whose XLA copies and relayouts sit between kernels, at
DeepSeek-V4-Flash-0731 DP8/EP8 serving shapes.

Run it through perfkit, never directly:

    perfkit validity run \
        --adapter scripts/forge_validity/dsv4_flash_xla_copies_validity.py \
        --arm <checkout>

`perfkit/validity_arm.py` defines the hooks and the two checks (`aot_all_shapes`,
`pipeline_trip_count`). Nothing here uses a TPU: compiles are ahead of time for
the tpu7x topology on the CPU, and trip counts are read while tracing.

What it checks
--------------
Under `deepseek-v4-flash-dp8-ep` the runner pads each engine's batch to one of
7 token counts (16 to 1024, `--max-num-batched-tokens=1024`). `SHAPES` is:

  mhc_pre_t*    `_mhc_pre_jax`, the chain-opening pre of layer 0's attention.
  mhc_fused_t*  `_mhc_fused_post_pre_jax`, the seam op that runs every other
                pre together with the post before it.
  mhc_post_t*   `_mhc_post_jax`, the post the last layer defers.
  moe_pad*      `fused_moe_gmm.fused_moe_func` at the 8 engines' gathered
                tokens, 128 to 8192.

The three mHC entries are the functions `pallas.jax_op` registers in
`layers/adapter/custom_ops/deepseek_v4/deepseek_v4_mhc_op.py`, imported from the
arm and called with the keyword constants `get_mhc_ops` binds, so a candidate
that changes what the op calls is compiled as the server compiles it. That
module imports torch, torch_tpu and the model wrapper context at the top but
uses them only to register the ops; `setup` imports it with stand-ins for those
three, removed again once it is loaded, and never calls a registration. Every
mHC operand is at one engine's T tokens: residual bf16 `[T, 4, 4096]`, the
sublayer output bf16 `[T, 4096]`, post mix f32 `[T, 4, 1]`, comb mix f32
`[T, 4, 4]`, and the layer's f32 `fn [24, 16384]`, `scale [3]`, `base [24]`.
The constants are those `DeepseekV4DecoderLayer.mhc_ops` passes:
rms_norm_eps 1e-6 and hc_eps 1e-6 (as both pre and Sinkhorn eps), hc_mult 4,
hc_sinkhorn_iters 20, from the checkpoint's `config.json` (the values
`tests/kernels/deepseek_v4/mhc_test.py` quotes), and hc_post_alpha 2.0, the
layer's default, since the config sets none.

The MoE entry is called as the DeepSeek-V4 MXFP4 MoE method calls it: hidden
4096, top-6 over the rank's 32 of 256 experts, `use_ep`, the clamped SiLU, fp4
weights packed two to a byte with block-512 f32 scales
(`MOE_REQUANTIZE_WEIGHT_DTYPE=fp4`, `MOE_REQUANTIZE_BLOCK_SIZE=512`), no bias,
with `USE_MOE_SPARSE_CORE`, `TPU_MOE_SKIP_PADDED_TOKENS` and
`resolve_onehot_permute_threshold()` read from the arm at their defaults (the
serve config sets none of them and `setup` clears them from the environment).
This is the entry `dsv4_flash_moe_sc_validity.py` checks; it is repeated here
because a campaign freezes one adapter file by its hash.

`aot_all_shapes` compiles each entry whole for tpu7x, the XLA gates, Sinkhorn,
pads, reshapes and routing layout around the kernels included, so a candidate
that moves work between XLA and a kernel is compiled as the server would
compile it. `limit` refuses an mHC program that launches no Mosaic kernel from
`kernels/deepseek_v4/mhc` (read from the lowered module's source locations)
and a MoE program with no SparseCore kernel: either is a program this adapter
did not check, not a candidate it passed.

`trip_counts` traces the same entries, without compiling, with
`pltpu.emit_pipeline` and `pl.pallas_call` replaced by recorders that call the
real ones. Each pipeline's inputs are reported with their buffer count (the
kernel's own `BufferedRef` allocations when it passes them, else `Buffered(n)`
in the spec's `pipeline_mode`, else 2) and lookahead, named by the file and
function that emit them. Pipelines emitted from `megablox` (gmm_v2, which has
its own adapter) are left out. A static grid gives one record at its trip
count, with the block each step reads when the index maps are functions of the
grid indices alone: golden's mHC kernels are `pallas_call`s over
`T / token block` steps (1 to 32). A grid with a runtime dim (golden's
SparseCore dispatch, `cdiv(routed rows, block)`, and combine row pipelines)
gives records N = 1 .. max(B, 8): the routed row count takes every value from 0
to the shape's maximum, and the adapter does not work out which N a shape can
reach. That is conservative. A pipeline partitioned over cores (`core_axis`)
is enumerated the same way.

What it cannot check
--------------------
- Nothing runs: no output is compared, and a wrong gate, Sinkhorn step,
  stream layout, gather or routing layout passes wherever it still compiles.
  The campaign's numerics tier has to catch those.
- The torch side of the step is not compiled: the EP all-gather of hidden
  states and router logits (`distributed/tpu_communicator.py`) and the
  relayout after it, the router and its top-k (`layers/adapter/moe_routing.py`),
  the quantized linears (`layers/adapter/linear_common.py`), the attention
  module and the model's own reshapes all run through torch, which this
  python does not have. A candidate that changes them is checked only where
  it changes the jax entries above, and the shapes and layouts the MoE entry
  is given are golden's, not whatever such a candidate would hand it.
- The `pallas.jax_op` wrapper (its shard_map over the op mesh, donation) is not
  applied; the mHC entries are compiled for one engine's device.
- Only pipeline schedules are read. DMAs a kernel starts itself are not
  counted, so an unwaited manual copy passes; output-only pipes are not
  recorded.
- Runtime grids are not bounded per shape (above).
- Shapes and flags are the serve config's; another config, the fused EP kernel
  (`USE_MOE_FUSED_EP_KERNEL`) or the pipelined collective path is not
  compiled, and neither is the auxiliary-hidden-state post the draft models
  ask for.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import itertools
import math
import os
import re
import sys
import types
from pathlib import Path

#: The whole package: candidates change call sites in layers/ as well as kernels.
KERNEL_DIR = "src/vllm_torchtpu"
MHC_DIR = "src/vllm_torchtpu/kernels/deepseek_v4/mhc"
TOPOLOGY = "tpu7x:2x2x1"

DP = 8
HIDDEN = 4096
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
#: Runtime grids are reported at N = 1 .. max(B, this).
RUNTIME_TRIPS = 8
#: gmm_v2's pipelines are left to its own adapter, which reads their block schedule.
GMM_DIR = "megablox"

#: mHC, as `DeepseekV4DecoderLayer.mhc_ops` binds it.
HC_MULT = 4
HC_MIX = (2 + HC_MULT) * HC_MULT
MHC_CONSTANTS = {
    "rms_eps": 1e-6,
    "hc_pre_eps": 1e-6,
    "hc_sinkhorn_eps": 1e-6,
    "hc_post_mult_value": 2.0,
    "sinkhorn_repeat": 20,
}
MHC_OPS = ("pre", "fused", "post")

#: MoE, as the MXFP4 MoE method calls `fused_moe_func`.
INTERMEDIATE = 2048
LOCAL_EXPERTS = 32  # 256 experts over EP=8
TOPK = 6
#: fp4 weights, two to a byte along K, one f32 scale per 512 rows of K.
PACKING = 2
SCALE_BLOCK = 512
ACTIVATION = "silu_and_mul_with_clamp"
#: Environment the serve config sets; every other MoE selector is cleared so the
#: arm's defaults apply.
SERVE_ENV = {"MOE_REQUANTIZE_WEIGHT_DTYPE": "fp4", "MOE_REQUANTIZE_BLOCK_SIZE": "512"}
CLEARED_PREFIXES = ("MOE_", "TPU_MOE_", "USE_MOE_", "ONEHOT_MOE_", "RAGGED_GATHER_")

#: What a runtime grid dim stands for, by the function that emits the pipeline.
RUNTIME_MEANING = {
    "main_kernel_v2": "dispatch row blocks, cdiv(routed rows, block)",
    "_window_kernel": "combine row blocks in a window, from the partition's rows",
}

SHAPES = {
    **{
        f"mhc_{op}_t{t}": {"part": "mhc", "op": op, "tokens": t}
        for op in MHC_OPS
        for t in TOKEN_PADDINGS
    },
    **{
        f"moe_pad{p}_tokens{DP * p}": {"part": "moe", "padding": p, "tokens": DP * p}
        for p in TOKEN_PADDINGS
    },
}

#: What the mHC op module imports only to register its ops.
OP_IMPORT_STANDINS = (
    "torch",
    "torch_tpu",
    "torch_tpu._internal",
    "torch_tpu._internal.pallas",
    "vllm_torchtpu.models",
    "vllm_torchtpu.models.vllm",
    "vllm_torchtpu.models.vllm.vllm_model_wrapper_context",
)


class _StandIn(types.ModuleType):
    """A module whose every attribute is a placeholder class: enough for the
    annotations and imports the op module evaluates, nothing that can run."""

    __path__: list[str] = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


def _package(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
    return mod


def _load(root: Path):
    """Import `fused_moe_gmm` and the mHC op module from the arm root without
    the `__init__`s of `vllm_torchtpu`, `vllm_torchtpu.layers` and
    `layers.adapter.custom_ops`, which import torch and the serving stack.
    `env_override` is loaded for the server's LIBTPU_INIT_ARGS and import-time
    Pallas patches; everything below imports normally."""
    pkg = root / "src" / "vllm_torchtpu"
    top = _package("vllm_torchtpu", pkg)
    _package("vllm_torchtpu.layers", pkg / "layers")
    _package(
        "vllm_torchtpu.layers.adapter.custom_ops", pkg / "layers/adapter/custom_ops"
    )
    for name in ("envs", "logger", "patch_registry"):
        setattr(top, name, importlib.import_module(f"vllm_torchtpu.{name}"))
    importlib.import_module("vllm_torchtpu.env_override")
    moe = importlib.import_module("vllm_torchtpu.layers.core.fused_moe_gmm")

    clashes = [n for n in OP_IMPORT_STANDINS if n in sys.modules]
    if clashes:
        raise RuntimeError(f"{clashes} already imported; cannot stand in for them")
    sys.modules.update({n: _StandIn(n) for n in OP_IMPORT_STANDINS})
    try:
        mhc = importlib.import_module(
            "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op"
        )
    finally:
        for n in OP_IMPORT_STANDINS:
            del sys.modules[n]
    return moe, mhc


def setup(root):
    # CPU only: jax must never open the host's TPU, and the compile-only
    # topology does not need the libtpu lock.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-4")
    for name in [n for n in os.environ if n.startswith(CLEARED_PREFIXES)]:
        del os.environ[name]
    os.environ.update(SERVE_ENV)
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    moe, mhc = _load(root)

    import jax
    import jax.numpy as jnp

    return types.SimpleNamespace(
        moe=moe,
        mhc=mhc,
        root=root,
        mhc_dir=root / MHC_DIR,
        jax=jax,
        jnp=jnp,
        device=None,
    )


def origin(handle) -> str:
    return handle.mhc.__file__


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


def _entry(handle, params):
    """The jax function the server runs for one shape, with its static
    arguments bound. Call it under `_tpu7x_mesh`: the MoE one-hot threshold
    reads the chip."""
    if params["part"] == "mhc":
        fn = {
            "pre": handle.mhc._mhc_pre_jax,
            "fused": handle.mhc._mhc_fused_post_pre_jax,
            "post": handle.mhc._mhc_post_jax,
        }[params["op"]]
        if params["op"] == "post":
            return fn
        return functools.partial(fn, **MHC_CONSTANTS)
    moe, envs = handle.moe, sys.modules["vllm_torchtpu.envs"]
    return functools.partial(
        moe.fused_moe_func,
        topk=TOPK,
        activation=ACTIVATION,
        use_ep=True,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_moe_permute_threshold=moe.resolve_onehot_permute_threshold(),
        rhs_quant_dtype=handle.jnp.float4_e2m1fn,
        skip_padded_tokens=envs.TPU_MOE_SKIP_PADDED_TOKENS,
    )


def _specs(handle, params):
    """`(shape, dtype)` per positional operand, None for an absent one."""
    jnp, t = handle.jnp, params["tokens"]
    if params["part"] == "mhc":
        residual = ((t, HC_MULT, HIDDEN), jnp.bfloat16)
        post_inputs = [
            ((t, HIDDEN), jnp.bfloat16),  # x, the sublayer output
            residual,
            ((t, HC_MULT, 1), jnp.float32),  # post_layer_mix
            ((t, HC_MULT, HC_MULT), jnp.float32),  # comb_res_mix
        ]
        gates = [
            ((HC_MIX, HC_MULT * HIDDEN), jnp.float32),  # fn
            ((3,), jnp.float32),  # hc_scale
            ((HC_MIX,), jnp.float32),  # hc_base
        ]
        return {
            "pre": [residual, *gates],
            "fused": [*post_inputs, *gates],
            "post": post_inputs,
        }[params["op"]]
    return [
        ((t, HIDDEN), jnp.bfloat16),  # hidden_states
        ((LOCAL_EXPERTS, HIDDEN // PACKING, 2 * INTERMEDIATE), jnp.uint8),  # w1
        ((LOCAL_EXPERTS, INTERMEDIATE // PACKING, HIDDEN), jnp.uint8),  # w2
        ((LOCAL_EXPERTS, HIDDEN // SCALE_BLOCK, 1, 2 * INTERMEDIATE), jnp.float32),
        ((LOCAL_EXPERTS, INTERMEDIATE // SCALE_BLOCK, 1, HIDDEN), jnp.float32),
        None,  # w1_bias
        None,  # w2_bias
        ((t, TOPK), jnp.float32),  # topk_weights
        ((t, TOPK), jnp.int32),  # topk_ids
        ((), jnp.int32),  # experts_start
    ]


def _args(handle, params):
    jax = handle.jax
    if handle.device is None:
        from jax.experimental import topologies

        handle.device = topologies.get_topology_desc(TOPOLOGY, platform="tpu").devices[
            0
        ]
    sharding = jax.sharding.SingleDeviceSharding(handle.device)
    return [
        None if s is None else jax.ShapeDtypeStruct(*s, sharding=sharding)
        for s in _specs(handle, params)
    ]


_LOC_DEF = re.compile(r"^(#loc\d+) = (.*)$", re.M)
_LOC_REF = re.compile(r"#loc\d+")
_KERNEL_NAME = re.compile(r'kernel_name = "([^"]*)"')


def _kernels_from(text, kernel_dir: Path):
    """Names of the Mosaic kernels in a lowered module (with debug info) whose
    source location is a file under `kernel_dir`."""
    defs = dict(_LOC_DEF.findall(text))

    def files(ref):
        found, todo, seen = set(), [ref], set()
        while todo:
            r = todo.pop()
            if r in seen:
                continue
            seen.add(r)
            body = defs.get(r, "")
            found.update(re.findall(r'"([^"]+\.py)"', body))
            todo += _LOC_REF.findall(body)
        return found

    names = []
    for line in text.splitlines():
        refs = _LOC_REF.findall(line)
        if "tpu_custom_call" not in line or not refs:
            continue
        if any(Path(f).resolve().is_relative_to(kernel_dir) for f in files(refs[-1])):
            name = _KERNEL_NAME.search(line)
            names.append(name.group(1) if name else "?")
    return names


def _sparse_core_kernels(text):
    """Names of the Mosaic kernels in a lowered module that run on SparseCore."""
    names = []
    for line in text.splitlines():
        if "tpu_custom_call" in line and "DEVICE_TYPE_SPARSECORE" in line:
            name = _KERNEL_NAME.search(line)
            names.append(name.group(1) if name else "?")
    return names


def compile(handle, name, params):
    jax = handle.jax
    args = _args(handle, params)
    with _tpu7x_mesh():
        lowered = (
            jax.jit(_entry(handle, params))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    mem = lowered.compile().memory_analysis()
    text = lowered.as_text(debug_info=params["part"] == "mhc")
    if params["part"] == "mhc":
        checked = {"mhc_kernels": _kernels_from(text, handle.mhc_dir)}
    else:
        checked = {"sparse_core": _sparse_core_kernels(text)}
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "kernels": text.count("tpu_custom_call"),
        **{k: ",".join(sorted(set(v))) for k, v in checked.items()},
    }


def limit(name, result):
    if result.get("mhc_kernels") == "":
        return (
            f"the compiled program launches no Mosaic kernel from {MHC_DIR}: "
            "nothing was checked"
        )
    if result.get("sparse_core") == "":
        return "the compiled program launches no SparseCore kernel: nothing was checked"
    return None


def _leaves(tree, kind):
    """`(label, leaf)` for every leaf of type `kind` in a pytree."""
    from jax import tree_util

    leaves, _ = tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda x: isinstance(x, kind)
    )
    return [
        ("in" + tree_util.keystr(path), leaf)
        for path, leaf in leaves
        if isinstance(leaf, kind)
    ]


def _blocks(index_map, grid):
    """The block each step of a static grid reads, in step order, or None when
    the index map is not a function of the grid indices alone."""
    if index_map is None:
        return None
    out = []
    for idx in itertools.product(*(range(d) for d in grid)):
        # An index map that reads refs or program ids cannot run here; without
        # blocks the check falls back to its stricter trip-count window.
        try:
            block = index_map(*idx)
        except Exception:
            return None
        block = block if isinstance(block, tuple) else (block,)
        if not all(isinstance(b, int) for b in block):
            return None
        out.append(block)
    return out


def _pipes_from_specs(in_specs):
    """`(label, index_map, B, lookahead)` per input spec, as `emit_pipeline`
    allocates when the kernel passes no allocations."""
    from jax.experimental import pallas as pl

    pipes = []
    for label, spec in _leaves(in_specs, pl.BlockSpec):
        mode = spec.pipeline_mode
        pipes.append(
            (
                label,
                spec.index_map,
                int(getattr(mode, "buffer_count", 2)),
                bool(getattr(mode, "use_lookahead", False)),
            )
        )
    return pipes


def _pipes_from_allocations(allocations):
    """`(label, index_map, B, lookahead)` per buffered input allocation."""
    from jax.experimental.pallas import tpu as pltpu

    return [
        (
            f"{label} {type(bref).__name__}",
            bref.spec.index_map,
            int(bref.buffer_count),
            bool(bref.use_lookahead),
        )
        for label, bref in _leaves(allocations, pltpu.BufferedRef)
        if bref.is_buffered and bref.is_input
    ]


@contextlib.contextmanager
def _recording(handle):
    """Replace `emit_pipeline` and `pallas_call` in `pltpu`/`pl` and in every
    loaded module of the arm with recorders that call the real ones, and yield
    the list of pipelines they saw."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    jax = handle.jax
    src = handle.root / KERNEL_DIR
    seen = []
    real = {"emit_pipeline": pltpu.emit_pipeline, "pallas_call": pl.pallas_call}

    def ours(caller):
        path = Path(caller.co_filename).resolve()
        return path.is_relative_to(src) and GMM_DIR not in path.parts

    def record(caller, grid, pipes, core_axis=None):
        dims = tuple(d if isinstance(d, jax.core.Tracer) else int(d) for d in grid)
        static = core_axis is None and all(isinstance(d, int) for d in dims)
        seen.append(
            {
                "site": f"{Path(caller.co_filename).name}:{caller.co_name}",
                "function": caller.co_name,
                "grid": dims,
                "trips": math.prod(dims) if static else None,
                "pipes": pipes,
            }
        )

    def emit_pipeline(body, *, grid, in_specs=(), **kwargs):
        caller = sys._getframe(1).f_code
        core_axis = kwargs.get("core_axis", kwargs.get("core_axis_name"))
        pipeline = real["emit_pipeline"](body, grid=grid, in_specs=in_specs, **kwargs)
        if not ours(caller):
            return pipeline

        def run(*refs, allocations=None, **kw):
            pipes = (
                _pipes_from_specs(in_specs)
                if allocations is None
                else _pipes_from_allocations(allocations)
            )
            record(caller, grid, pipes, core_axis)
            return pipeline(*refs, allocations=allocations, **kw)

        return run

    def pallas_call(kernel, *args, **kwargs):
        caller = sys._getframe(1).f_code
        spec = kwargs.get("grid_spec")
        grid = getattr(spec, "grid", None) or kwargs.get("grid") or ()
        grid = grid if isinstance(grid, tuple) else (grid,)
        if grid and ours(caller):
            in_specs = getattr(spec, "in_specs", None) or kwargs.get("in_specs", ())
            record(caller, grid, _pipes_from_specs(in_specs))
        return real["pallas_call"](kernel, *args, **kwargs)

    fake = {"emit_pipeline": emit_pipeline, "pallas_call": pallas_call}
    patched = [
        (mod, attr, value)
        for name, mod in list(sys.modules.items())
        if name.startswith("vllm_torchtpu") and mod is not None
        for attr, value in vars(mod).items()
        if any(value is r for r in real.values())
    ]
    homes = {"emit_pipeline": pltpu, "pallas_call": pl}
    for attr, home in homes.items():
        setattr(home, attr, fake[attr])
    for mod, attr, value in patched:
        setattr(mod, attr, fake[_name_of(real, value)])
    # Traces cached by the compiles would skip the recorder.
    jax.clear_caches()
    try:
        yield seen
    finally:
        for attr, home in homes.items():
            setattr(home, attr, real[attr])
        for mod, attr, value in patched:
            setattr(mod, attr, value)
        jax.clear_caches()


def _name_of(real, value):
    return next(n for n, r in real.items() if r is value)


def trip_counts(handle, name, params):
    jax = handle.jax
    args = _args(handle, params)
    with _recording(handle) as seen, _tpu7x_mesh():
        jax.jit(_entry(handle, params)).trace(*args)
    if not any(p["pipes"] for p in seen):
        raise RuntimeError(
            "the trace emitted no pipelined input this adapter can see, so no trip "
            "count was checked"
        )
    records, done = [], set()
    for p in seen:
        for label, index_map, b, lookahead in p["pipes"]:
            key = (p["site"], p["grid"] if p["trips"] else None, label, b, lookahead)
            if key in done:
                continue
            done.add(key)
            pipe = f"{p['site']} {label}"
            if p["trips"] is not None:
                record = {
                    "pipe": pipe,
                    "buffer_count": b,
                    "trip_count": p["trips"],
                    "lookahead": lookahead,
                    "input": f"static grid {p['grid']}",
                }
                blocks = _blocks(index_map, p["grid"])
                if blocks is not None:
                    record["blocks"] = blocks
                records.append(record)
                continue
            meaning = RUNTIME_MEANING.get(p["function"], f"runtime grid of {p['site']}")
            records += [
                {
                    "pipe": pipe,
                    "buffer_count": b,
                    "trip_count": n,
                    "lookahead": lookahead,
                    "input": f"{meaning} = {n}",
                }
                for n in range(1, max(b, RUNTIME_TRIPS) + 1)
            ]
    return records
