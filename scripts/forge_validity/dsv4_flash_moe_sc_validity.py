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

r"""Validity adapter for the expert-parallel MoE dispatch and combine on
SparseCore at DeepSeek-V4-Flash-0731 DP8/EP8 serving shapes.

Run it through perfkit, never directly:

    perfkit validity run \
        --adapter scripts/forge_validity/dsv4_flash_moe_sc_validity.py \
        --arm <checkout>

`perfkit/validity_arm.py` defines the hooks and the two checks (`aot_all_shapes`,
`pipeline_trip_count`). Nothing here uses a TPU: compiles are ahead of time for
the tpu7x topology on the CPU, and trip counts are read while tracing.

What it checks
--------------
The entry is `fused_moe_gmm.fused_moe_func`, called as the DeepSeek-V4 MXFP4 MoE
method calls it under `deepseek-v4-flash-dp8-ep`: the runner pads each engine's
batch to one of 7 token counts (16 to 1024, `--max-num-batched-tokens=1024`) and
the experts see the 8 engines' tokens gathered, so `SHAPES` is 128 to 8192
tokens of hidden 4096, top-6 over the rank's 32 of 256 experts, `use_ep`, the
clamped SiLU, fp4 weights packed two to a byte with block-512 f32 scales
(`MOE_REQUANTIZE_WEIGHT_DTYPE=fp4`, `MOE_REQUANTIZE_BLOCK_SIZE=512`), no bias.
The flags the torch bridge (`layers/adapter/fused_moe.py`, which imports torch)
would add are read from the arm's own `envs` and `fused_moe_gmm`:
`USE_MOE_SPARSE_CORE`, `TPU_MOE_SKIP_PADDED_TOKENS` and
`resolve_onehot_permute_threshold()`, all at their defaults, since the serve
config sets none of them and `setup` clears them from the environment.

`aot_all_shapes` compiles that whole function for tpu7x at each shape: routing
layout, the call-site code that picks the dispatch and combine kernels, both
gmm_v2 matmuls and the kernels it picks, so a candidate that changes which
kernel runs or how it is called is compiled as the server would compile it. On
golden every shape takes the SparseCore dispatch; the combine's own fallback
keeps the two smallest shapes (768 and 1536 routed rows) on a TensorCore gather.
`limit` refuses a program with no SparseCore kernel in it: that is an adapter
that did not see a SparseCore chip, not a candidate that was checked.

SparseCore kernels do not run in Pallas interpret mode (jax 0.11.2: the
dispatch kernel fails to trace with `interpret=True` and with `InterpretParams`),
so `trip_counts` traces the same entry, without compiling, with
`pltpu.emit_pipeline` replaced by a recorder that calls the real one, and
reports every pipelined input of every pipeline the trace emits (dispatch
outer/inner, combine row/column), with its buffer count (`pipeline_mode`
`Buffered(n)`, else 2) and lookahead, named by the file and function that emit
it and the input's place in `in_specs`. A grid whose dims are all static gives one
record at its static trip count. A grid with a runtime dim (golden's dispatch
outer pipeline, `cdiv(valid_count, block)`, and the combine's row pipeline,
`blocks_in_window`) gives records N = 1 .. max(B, 8): the routed row count takes
every value from 0 to the shape's maximum (warmup's dummy batches reach the
small end), and the adapter does not work out which N the shape can reach. That
is conservative: a deep-buffered runtime pipe fails even at a shape whose
largest N is below the unsafe window. A pipeline partitioned over cores
(`core_axis`) is enumerated the same way.

What it cannot check
--------------------
- Nothing runs: no output is compared, and a wrong gather, reduce or routing
  layout passes wherever it still compiles (a reduce group of 7 instead of 6
  fails only the two shapes on the TensorCore fallback). The campaign's
  numerics tier has to catch those.
- Only `emit_pipeline` schedules are read. DMAs a kernel starts itself
  (`make_async_copy`, the scatter in the combine's column body) are not
  counted, so an unwaited manual copy passes; output pipes are not recorded.
- No `blocks` are reported (the index maps read routed indices at run time),
  so a static pipe in the unsafe window fails even if its steps repeat a block.
- Runtime grids are not bounded per shape (above), and the gmm_v2 matmuls'
  pipelines are not recorded here; gmm_v2 has its own adapter.
- Shapes and flags are the serve config's; another config, the fused EP kernel
  (`USE_MOE_FUSED_EP_KERNEL`) or the pipelined collective path is not compiled.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import math
import os
import sys
import types
from pathlib import Path

#: The whole package: candidates may change the call site as well as the kernels.
KERNEL_DIR = "src/vllm_torchtpu"
TOPOLOGY = "tpu7x:2x2x1"

DP = 8
HIDDEN = 4096
INTERMEDIATE = 2048
LOCAL_EXPERTS = 32  # 256 experts over EP=8
TOPK = 6
#: fp4 weights, two to a byte along K, one f32 scale per 512 rows of K.
PACKING = 2
SCALE_BLOCK = 512
ACTIVATION = "silu_and_mul_with_clamp"
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
#: gmm_v2's pipelines are left to its own adapter, which reads their block schedule.
GMM_DIR = "megablox"
#: Runtime grids are reported at N = 1 .. max(B, this).
RUNTIME_TRIPS = 8
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
    f"pad{p}_tokens{DP * p}": {"padding": p, "tokens": DP * p} for p in TOKEN_PADDINGS
}


def _load(root: Path):
    """Import `fused_moe_gmm` from the arm root without the `__init__`s of
    `vllm_torchtpu` and `vllm_torchtpu.layers`, which import torch and the
    serving stack. `env_override` is loaded for the server's LIBTPU_INIT_ARGS
    and import-time Pallas patches; everything below imports normally."""
    pkg = root / "src" / "vllm_torchtpu"

    def package(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        return mod

    top = package("vllm_torchtpu", pkg)
    package("vllm_torchtpu.layers", pkg / "layers")
    for name in ("envs", "logger", "patch_registry"):
        setattr(top, name, importlib.import_module(f"vllm_torchtpu.{name}"))
    importlib.import_module("vllm_torchtpu.env_override")
    return importlib.import_module("vllm_torchtpu.layers.core.fused_moe_gmm")


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
    moe = _load(Path(root).resolve())

    import jax
    import jax.numpy as jnp

    return types.SimpleNamespace(moe=moe, jax=jax, jnp=jnp, device=None)


def origin(handle) -> str:
    return handle.moe.__file__


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


def _entry(handle):
    """`fused_moe_func` with the static arguments the torch bridge binds.
    Call it under `_tpu7x_mesh`: the one-hot threshold reads the chip."""
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


def _args(handle, tokens):
    jax, jnp = handle.jax, handle.jnp
    if handle.device is None:
        from jax.experimental import topologies

        handle.device = topologies.get_topology_desc(TOPOLOGY, platform="tpu").devices[
            0
        ]
    sharding = jax.sharding.SingleDeviceSharding(handle.device)
    specs = [
        ((tokens, HIDDEN), jnp.bfloat16),  # hidden_states
        ((LOCAL_EXPERTS, HIDDEN // PACKING, 2 * INTERMEDIATE), jnp.uint8),  # w1
        ((LOCAL_EXPERTS, INTERMEDIATE // PACKING, HIDDEN), jnp.uint8),  # w2
        ((LOCAL_EXPERTS, HIDDEN // SCALE_BLOCK, 1, 2 * INTERMEDIATE), jnp.float32),
        ((LOCAL_EXPERTS, INTERMEDIATE // SCALE_BLOCK, 1, HIDDEN), jnp.float32),
        None,  # w1_bias
        None,  # w2_bias
        ((tokens, TOPK), jnp.float32),  # topk_weights
        ((tokens, TOPK), jnp.int32),  # topk_ids
        ((), jnp.int32),  # experts_start
    ]
    return [
        None if s is None else jax.ShapeDtypeStruct(*s, sharding=sharding)
        for s in specs
    ]


def _sparse_core_kernels(text):
    """Names of the Mosaic kernels in a lowered module that run on SparseCore."""
    names = []
    for line in text.splitlines():
        if "tpu_custom_call" in line and "DEVICE_TYPE_SPARSECORE" in line:
            start = line.find('kernel_name = "')
            names.append(line[start + 15 :].split('"')[0] if start >= 0 else "?")
    return names


def compile(handle, name, params):
    jax = handle.jax
    args = _args(handle, params["tokens"])
    with _tpu7x_mesh():
        lowered = (
            jax.jit(_entry(handle)).trace(*args).lower(lowering_platforms=("tpu",))
        )
    mem = lowered.compile().memory_analysis()
    text = lowered.as_text()
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "kernels": text.count("tpu_custom_call"),
        "sparse_core": ",".join(sorted(set(_sparse_core_kernels(text)))),
    }


def limit(name, result):
    if not result["sparse_core"]:
        return "the compiled program launches no SparseCore kernel: nothing was checked"
    return None


def _input_specs(specs):
    """`(label, BlockSpec)` for every input spec in a spec pytree."""
    from jax import tree_util
    from jax.experimental import pallas as pl

    leaves, _ = tree_util.tree_flatten_with_path(
        specs, is_leaf=lambda x: isinstance(x, pl.BlockSpec)
    )
    return [
        ("in" + tree_util.keystr(path), spec)
        for path, spec in leaves
        if isinstance(spec, pl.BlockSpec)
    ]


@contextlib.contextmanager
def _recording(handle):
    """Replace `emit_pipeline` (and `emit_pipeline_with_allocations`) in
    `pltpu` and in every loaded module of the arm with a recorder that calls
    the real one, and yield the list of pipelines it saw."""
    from jax.experimental.pallas import tpu as pltpu

    jax = handle.jax
    seen = []
    real = {
        n: getattr(pltpu, n)
        for n in ("emit_pipeline", "emit_pipeline_with_allocations")
    }

    def record(grid, in_specs, core_axis):
        caller = sys._getframe(2).f_code
        if GMM_DIR in Path(caller.co_filename).parts:
            return
        dims = tuple(d if isinstance(d, jax.core.Tracer) else int(d) for d in grid)
        static = core_axis is None and all(isinstance(d, int) for d in dims)
        seen.append(
            {
                "site": f"{Path(caller.co_filename).name}:{caller.co_name}",
                "function": caller.co_name,
                "grid": dims,
                "trips": math.prod(dims) if static else None,
                "pipes": [
                    (
                        label,
                        int(getattr(spec.pipeline_mode, "buffer_count", 2)),
                        bool(getattr(spec.pipeline_mode, "use_lookahead", False)),
                    )
                    for label, spec in _input_specs(in_specs)
                ],
            }
        )

    def emit_pipeline(body, *, grid, in_specs=(), **kwargs):
        core_axis = kwargs.get("core_axis", kwargs.get("core_axis_name"))
        record(grid, in_specs, core_axis)
        return real["emit_pipeline"](body, grid=grid, in_specs=in_specs, **kwargs)

    def emit_pipeline_with_allocations(body, *, grid, in_specs=(), **kwargs):
        record(grid, in_specs, None)
        return real["emit_pipeline_with_allocations"](
            body, grid=grid, in_specs=in_specs, **kwargs
        )

    fake = {
        "emit_pipeline": emit_pipeline,
        "emit_pipeline_with_allocations": emit_pipeline_with_allocations,
    }
    patched = [
        (mod, attr, value)
        for name, mod in list(sys.modules.items())
        if name.startswith("vllm_torchtpu") and mod is not None
        for attr, value in vars(mod).items()
        if any(value is r for r in real.values())
    ]
    for attr in fake:
        setattr(pltpu, attr, fake[attr])
    for mod, attr, value in patched:
        setattr(mod, attr, fake[_name_of(real, value)])
    # Traces cached by the compiles would skip the recorder.
    jax.clear_caches()
    try:
        yield seen
    finally:
        for attr, value in real.items():
            setattr(pltpu, attr, value)
        for mod, attr, value in patched:
            setattr(mod, attr, value)
        jax.clear_caches()


def _name_of(real, value):
    return next(n for n, r in real.items() if r is value)


def trip_counts(handle, name, params):
    jax = handle.jax
    args = _args(handle, params["tokens"])
    with _recording(handle) as seen, _tpu7x_mesh():
        jax.jit(_entry(handle)).trace(*args)
    if not seen:
        raise RuntimeError(
            "the trace emitted no emit_pipeline this adapter can see, so no trip "
            "count was checked"
        )
    records, done = [], set()
    for p in seen:
        for label, b, lookahead in p["pipes"]:
            key = (p["site"], p["grid"] if p["trips"] else None, label, b, lookahead)
            if key in done:
                continue
            done.add(key)
            pipe = f"{p['site']} {label}"
            if p["trips"] is not None:
                records.append(
                    {
                        "pipe": pipe,
                        "buffer_count": b,
                        "trip_count": p["trips"],
                        "lookahead": lookahead,
                        "input": f"static grid {p['grid']}",
                    }
                )
                continue
            meaning = RUNTIME_MEANING.get(p["function"], f"runtime grid of {p['site']}")
            for n in range(1, max(b, RUNTIME_TRIPS) + 1):
                records.append(
                    {
                        "pipe": pipe,
                        "buffer_count": b,
                        "trip_count": n,
                        "lookahead": lookahead,
                        "input": f"{meaning} = {n}",
                    }
                )
    return records
