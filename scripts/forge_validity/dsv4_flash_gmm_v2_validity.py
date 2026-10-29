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
r"""Validity adapter for megablox `gmm_v2` at DeepSeek-V4-Flash DP8/EP8 shapes.

Run it through perfkit, never directly:

    perfkit validity run --adapter <this file> --arm <checkout> \
        --python "$(perfkit env ensure --jax 0.11.2 --libtpu 0.0.50)"

`perfkit/validity_arm.py` defines the hooks and the two checks. Nothing here
uses a TPU: compiles are ahead of time for the tpu7x topology on the CPU, and
trip counts come from Pallas interpret mode on the CPU.

Shapes
------
The serving config is `scripts/vllm/benchmarking/configs/
deepseek-v4-flash-dp8-ep.sh`: TP1 x DP8 with EP, `--max-num-batched-tokens
1024`, `MOE_REQUANTIZE_WEIGHT_DTYPE=fp4`, `MOE_REQUANTIZE_BLOCK_SIZE=512`.

* Token buckets. Each engine pads a step to one of 16, 32, ..., 512, 1024
  tokens (`_get_token_paddings` in `platforms/tpu_platform.py`, doubling from
  16 to the token budget), and DP lockstep pads every engine to the same
  bucket. The MoE layer sees the batch gathered from all 8 engines.
* Routing. 256 experts over EP8 is 32 on a rank (`size_group`), and top-6
  makes `size_m = 8 * 6 * bucket`: 768 to 49152. The hash-routed MoE layers
  take their expert ids from a table instead of the router, then go through
  the same `fused_moe_func` with the same weights, so they send the same
  shapes.
* Legs (`moe_gmm` in `layers/core/fused_moe_gmm.py`). gate/up has K = 4096
  (hidden) and N = 4096 (gate and up, 2048 each) with the fused
  `silu_and_mul_with_clamp`; down has K = 2048 and N = 4096 with a bf16
  output.
* Dtypes (`VllmDeepseekV4Mxfp4MoEMethod` in `quantization/mxfp4.py`). The
  weights are requantized to fp4 e2m1 with block 512 and packed two per uint8
  along K, so rhs is `uint8[32, K/2, N]` with `rhs_quant_dtype=float4_e2m1fn`
  and an f32 scale `[32, K/512, 1, N]`. `gmm_wrapper` quantizes the bf16 lhs
  (it reads the uint8 carrier as not-fp4), so this is the W4A8 path.

That is 7 buckets x 2 legs = 14 shapes. Profiles of golden on the serving
config (prefill steps at the 1024 bucket, decode steps at 128) ran gmm_v2
only at g=32 under these names, and the compile here gives
golden the same names at `size_m` 49152 and 6144: gate/up `tm_512-tk_4096-
tn_2048`, down `tm_512-tk_2048-tn_4096`. The single-group dense-linear path
through gmm_v2 does not run in this model.

Checks
------
`compile` builds each shape through the arm's own `gmm_v2` entry point, so
the tiling is the kernel's own choice for that shape, and compiles it for
tpu7x. `limit` refuses a program that launches no Mosaic kernel, since such a
compile checked nothing.

`trip_counts` runs the arm's kernel body in Pallas interpret mode for each
runtime input, with its DMAs and its pipeline body replaced by no-ops, and
reads every `emit_pipeline` call: the grid it was given at run time, the
buffer count and lookahead of each input spec, and the block index each
spec's own index map gives at each step. The rule it feeds
(`pipeline_trip_count`) refuses a pipe that starts a copy no step waits for,
which halts the chip at the kernel's exit. This assumes the grid and the index
maps are computed from the scalar-prefetch inputs (group sizes and group
offset), which is how `gmm_v2` is written. The runtime inputs, per shape:

  * 0 to 8, 16 and 32 active experts sharing a rank's expected rows
    (`size_m / 8`);
  * one expert spanning 2 to 8 gm tiles, at the kernel's own `tile_m`;
  * every token routed to one expert (`size_m / 6` rows);
  * six experts each holding every token (all `size_m` rows on one rank),
    the most a rank can receive, which hash routing approaches on a batch
    of one repeated token id.

Warmup's dummy batches and low-concurrency tails reach the small active
counts: a rank receives on average 0.75 of each token's 6 picks.

Pipeline bodies are not run, so a wait inside one is never seen, and a DMA
the kernel starts itself is counted but not checked. When a runtime input
starts DMAs of the kernel's own and pipelines no input with more than two
buffers, nothing at that input was checked, and `trip_counts` raises rather
than report no pipes.

Calibration
-----------
With this adapter on the CPU (jax 0.11.2, libtpu 0.0.50), against golden
c4ec11a7e and two copies of it whose weight pipe is deeper than golden's 3
buffers, a change that once passed fixed-shape benchmarks of gmm_v2 and
halted the chip during server warmup:

  * golden passes both checks: 14 shapes compile, and 254 runtime inputs give
    381 pipe records, all at B=3.
  * 4 buffers on gate/up and 6 on down compiles at all 14 shapes and fails
    `pipeline_trip_count` at 19 inputs: 2 active experts on gate/up at
    `size_m` 768 to 6144 (B=4, N=2), and 2, 3 and 4 active experts on down at
    768 to 12288 (B=6, N=2 to 4). No input fails at 24576 or at 49152, the
    1024-token bucket that prefill runs at. One expert spanning several tiles
    reads one block at every step and passes.
  * 6 buffers on both legs fails `aot_all_shapes` at all 7 gate/up shapes
    (scoped VMEM out of memory) and `pipeline_trip_count` at 30 inputs, the
    2 to 4 active experts at 768 to 12288 on both legs.

A run takes about three minutes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib.util
import math
import os
import sys
import types
from pathlib import Path

KERNEL_DIR = "src/vllm_torchtpu/kernels/megablox"
TOPOLOGY = "tpu7x:2x2x1"

NUM_GROUPS = 32  # 256 experts over EP8
EP = 8
DP = 8
TOPK = 6
QUANT_BLOCK = 512
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
LEGS = {
    "gate_up": {
        "size_k": 4096,
        "size_n": 4096,
        "fuse_act": "silu_and_mul_with_clamp",
        "out_dtype": None,
    },
    "down": {"size_k": 2048, "size_n": 4096, "fuse_act": None, "out_dtype": "bfloat16"},
}
SHAPES = {
    f"{leg}_m{DP * TOPK * tokens}": {
        "leg": leg,
        "size_m": DP * TOPK * tokens,
        **LEGS[leg],
    }
    for leg in LEGS
    for tokens in TOKEN_PADDINGS
}

ACTIVE = (*range(9), 16, NUM_GROUPS)
MAX_SPAN = 8


def _load_kernel(root: Path):
    """Import gmm_v2 from the arm root without the package's own `__init__`.

    That `__init__` imports torch and the serving stack, which a compile-only
    environment does not have. `env_override` is loaded because it sets the
    server's LIBTPU_INIT_ARGS and applies its import-time Pallas patches, so
    the program compiled here is the one the server compiles.
    """
    pkg = root / "src" / "vllm_torchtpu"

    def package(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        return mod

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    top = package("vllm_torchtpu", pkg)
    for name in ("envs", "logger", "patch_registry"):
        setattr(top, name, module(f"vllm_torchtpu.{name}", pkg / f"{name}.py"))
    module("vllm_torchtpu.env_override", pkg / "env_override.py")
    package("vllm_torchtpu.kernels", pkg / "kernels")
    package("vllm_torchtpu.kernels.megablox", pkg / "kernels" / "megablox")
    return module(
        "vllm_torchtpu.kernels.megablox.gmm_v2",
        pkg / "kernels" / "megablox" / "gmm_v2.py",
    )


def setup(root):
    # CPU only: jax must never open the host's TPU, and the compile-only
    # topology does not need the libtpu lock.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    sys.dont_write_bytecode = True
    kernel = _load_kernel(Path(root).resolve())

    import jax
    import jax.numpy as jnp
    import numpy as np

    return types.SimpleNamespace(kernel=kernel, jax=jax, jnp=jnp, np=np, device=None)


def origin(handle) -> str:
    return handle.kernel.__file__


def _entry(handle, params):
    """The kernel called as `gmm_wrapper` calls it for this model, at one shape."""
    jnp = handle.jnp
    gmm = handle.kernel.gmm_v2
    gmm = getattr(gmm, "__wrapped__", gmm)
    out_dtype = getattr(jnp, params["out_dtype"]) if params["out_dtype"] else None

    def call(lhs, rhs, group_sizes, rhs_scale, group_offset):
        return gmm(
            lhs,
            rhs,
            group_sizes,
            rhs_scale,
            None,
            group_offset[0],
            zero_initialize=False,
            fuse_act=params["fuse_act"],
            preferred_element_type=out_dtype,
            maybe_quantize_lhs=True,
            rhs_quant_dtype=jnp.float4_e2m1fn,
        )

    return call


def _arg_specs(handle, params):
    jnp = handle.jnp
    m, k, n = params["size_m"], params["size_k"], params["size_n"]
    return [
        ((m, k), jnp.bfloat16),
        ((NUM_GROUPS, k // 2, n), jnp.uint8),
        ((NUM_GROUPS,), jnp.int32),
        ((NUM_GROUPS, k // QUANT_BLOCK, 1, n), jnp.float32),
        ((1,), jnp.int32),
    ]


def _tpu7x_mesh():
    """Trace as for one tpu7x core, so the kernel's chip queries answer for it."""
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


def compile(handle, name, params):
    jax = handle.jax
    if handle.device is None:
        from jax.experimental import topologies

        topology = topologies.get_topology_desc(TOPOLOGY, platform="tpu")
        handle.device = topology.devices[0]
    sharding = jax.sharding.SingleDeviceSharding(handle.device)
    args = [
        jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
        for shape, dtype in _arg_specs(handle, params)
    ]
    with _tpu7x_mesh():
        lowered = (
            jax.jit(_entry(handle, params))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    mem = lowered.compile().memory_analysis()
    text = lowered.as_text()
    start = text.find("gmm_v2-g_")
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "mosaic": "tpu_custom_call" in text,
        "kernel": text[start : start + 200].split('"')[0] if start >= 0 else "",
    }


def limit(name, result):
    if not result["mosaic"]:
        return (
            "the compiled program launches no Mosaic kernel, so the compile "
            "checked nothing"
        )
    return None


class _NoCopy:
    """A DMA that copies nothing and reports each run-time start to `on_start`."""

    def __init__(self, jax, on_start):
        self._jax, self._on_start = jax, on_start

    def start(self, *args, **kwargs):
        self._jax.debug.callback(self._on_start)

    def wait(self, *args, **kwargs):
        pass


class _Override:
    """A module with some attributes replaced."""

    def __init__(self, real, **replaced):
        self._real = real
        self.__dict__.update(replaced)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _input_pipes(specs, label="in"):
    """`(label, spec)` for every input spec with a pipeline mode.

    A spec without one is double-buffered, and two buffers have no unsafe trip
    count. The walk follows lists, tuples and dataclasses, because the kernel
    wraps its specs in its own weight-ref dataclasses.
    """
    found = []
    if getattr(getattr(specs, "pipeline_mode", None), "buffer_count", None):
        found.append((label, specs))
    if isinstance(specs, (list, tuple)):
        for i, s in enumerate(specs):
            found += _input_pipes(s, f"{label}[{i}]")
    elif dataclasses.is_dataclass(specs) and not isinstance(specs, type):
        for f in dataclasses.fields(specs):
            found += _input_pipes(getattr(specs, f.name), f"{label}.{f.name}")
    return found


@contextlib.contextmanager
def _interpret(handle):
    """Run the kernel in interpret mode, recording its pipelines, not running them.

    Yields a namespace that collects, per `emit_pipeline` call, its input pipes
    at trace time, and at run time its grid and the block index each pipe reads
    at each step, plus each `GmmConfigs` built. `copies` counts the DMAs the
    kernel starts itself at run time, outside any pipeline body, since those
    bodies are not run.
    """
    from jax import lax
    from jax._src.pallas import core as pallas_core
    from jax._src.pallas.mosaic import pipeline as jax_pipeline
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    jax, jnp, k = handle.jax, handle.jnp, handle.kernel
    seen = types.SimpleNamespace(pipes=[], grids=[], blocks={}, cfgs=[], copies=0)
    chip = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)

    def emit_pipeline(body, *, grid, in_specs=(), out_specs=(), **kwargs):
        index = len(seen.pipes)
        pipes = _input_pipes(in_specs)
        seen.pipes.append(
            [
                (
                    label,
                    int(spec.pipeline_mode.buffer_count),
                    bool(spec.pipeline_mode.use_lookahead),
                )
                for label, spec in pipes
            ]
        )

        def on_grid(*dims):
            seen.grids.append((index, tuple(int(d) for d in dims)))

        def on_step(sizes, step, *flat):
            flat, blocks = [int(x) for x in flat], []
            for size in sizes:
                blocks.append(tuple(flat[:size]))
                flat = flat[size:]
            seen.blocks.setdefault(index, {})[int(step)] = blocks

        def step(i, indices):
            # The pipeline's own step order and index arithmetic
            # (`_emit_pipeline`'s loop body), and its block comparison reads
            # the same leaves (`_tuples_differ`).
            indices = jax_pipeline._filter_indices(indices, grid)
            with pallas_core.grid_env(list(map(pallas_core.GridAxis, indices, grid))):
                blocks = [
                    [
                        jnp.asarray(x, jnp.int32)
                        for x in jax.tree.leaves(spec.index_map(*indices))
                    ]
                    for _, spec in pipes
                ]
            jax.debug.callback(
                functools.partial(on_step, [len(b) for b in blocks]),
                i,
                *[x for b in blocks for x in b],
            )
            return jax_pipeline._next_index(indices, grid)

        def pipeline(*refs, **kwargs):
            dims = [jnp.asarray(g, jnp.int32) for g in grid]
            jax.debug.callback(on_grid, *dims)
            lax.fori_loop(0, math.prod(dims), step, (0,) * len(grid))

        return pipeline

    def on_copy():
        seen.copies += 1

    def make_async_copy(*args, **kwargs):
        return _NoCopy(jax, on_copy)

    def async_copy(*args, **kwargs):
        copy = _NoCopy(jax, on_copy)
        copy.start()
        return copy

    def make_gmm_configs(*args, **kwargs):
        cfgs = real_configs(*args, **kwargs)
        seen.cfgs.append(cfgs)
        return cfgs

    real_pl, real_pltpu, real_configs = k.pl, k.pltpu, k.make_gmm_configs
    real_info = (tpu_info.get_tpu_info, pltpu.get_tpu_info)
    k.pl = _Override(pl, pallas_call=functools.partial(pl.pallas_call, interpret=True))
    k.pltpu = _Override(
        pltpu,
        emit_pipeline=emit_pipeline,
        get_tpu_info=lambda: chip,
        make_async_copy=make_async_copy,
        async_copy=async_copy,
    )
    k.make_gmm_configs = make_gmm_configs
    tpu_info.get_tpu_info = pltpu.get_tpu_info = lambda: chip
    try:
        yield seen
    finally:
        k.pl, k.pltpu, k.make_gmm_configs = real_pl, real_pltpu, real_configs
        tpu_info.get_tpu_info, pltpu.get_tpu_info = real_info


def _runtime_inputs(size_m, tile_m):
    """`(description, nonzero group sizes)` for every runtime input checked."""
    rows, tokens = size_m // EP, size_m // TOPK
    for active in ACTIVE:
        base, extra = divmod(rows, active) if active else (0, 0)
        yield (
            f"{active} active experts sharing {rows} rows",
            [base + 1] * extra + [base] * (active - extra),
        )
    for span in range(2, MAX_SPAN + 1):
        size = (span - 1) * tile_m + 1
        if size > size_m:
            break
        yield f"one expert of {size} rows spanning {span} tiles", [size]
    yield f"all {tokens} tokens routed to one expert", [tokens]
    yield f"{TOPK} experts each holding all {tokens} tokens", [tokens] * TOPK


def pipelines_at(handle, params, inputs):
    """`[(description, [(pipe, buffer_count, lookahead, trip_count, blocks), ...])]`.

    `inputs` is a callable from the kernel's `tile_m` to `(description, group
    sizes)` pairs, so callers can check any group-size vectors at a shape.
    `blocks` is the block index the pipe reads at each step, in step order.
    """
    jax, jnp, np = handle.jax, handle.jnp, handle.np
    out = []
    with _interpret(handle) as seen, _tpu7x_mesh():
        run = jax.jit(_entry(handle, params))
        lhs, rhs, sizes, scale, offset = (
            jnp.zeros(s, d) for s, d in _arg_specs(handle, params)
        )
        scale = jnp.ones_like(scale)

        def trip_counts_for(group_sizes):
            vec = np.zeros(NUM_GROUPS, np.int32)
            vec[: len(group_sizes)] = group_sizes
            seen.grids.clear()
            seen.blocks.clear()
            seen.copies = 0
            jax.block_until_ready(run(lhs, rhs, jnp.asarray(vec), scale, offset))
            jax.effects_barrier()
            if not seen.grids:
                raise RuntimeError(
                    "the kernel ran no emit_pipeline this adapter can see, so "
                    "its trip counts are unknown"
                )
            if seen.copies and not any(seen.pipes[i] for i, _ in seen.grids):
                plural = "s" if seen.copies > 1 else ""
                raise RuntimeError(
                    f"the kernel started {seen.copies} DMA{plural} of its own and "
                    "pipelines no input with more than two buffers, so nothing here "
                    "was checked. This adapter reads only emit_pipeline schedules, "
                    "and cannot tell whether each of the kernel's own copies is "
                    "waited for"
                )
            named = len(seen.pipes) > 1
            found = []
            for i, grid in seen.grids:
                n = math.prod(grid)
                steps = seen.blocks.get(i, {})
                if sorted(steps) != list(range(n)):
                    raise RuntimeError(
                        f"pipeline {i} reported block indices for {len(steps)} "
                        f"of its {n} steps"
                    )
                for j, (pipe, b, lookahead) in enumerate(seen.pipes[i]):
                    found.append(
                        (
                            f"pipeline{i}:{pipe}" if named else pipe,
                            b,
                            lookahead,
                            n,
                            [steps[s][j] for s in range(n)],
                        )
                    )
            return found

        trip_counts_for([])
        tile_m = seen.cfgs[-1].tiles.tile_m
        for description, group_sizes in inputs(tile_m):
            out.append((description, trip_counts_for(group_sizes)))
    return out


def trip_counts(handle, name, params):
    found = pipelines_at(
        handle, params, functools.partial(_runtime_inputs, params["size_m"])
    )
    return [
        {
            "pipe": pipe,
            "buffer_count": b,
            "trip_count": n,
            "lookahead": lookahead,
            "blocks": blocks,
            "input": description,
        }
        for description, pipes in found
        for pipe, b, lookahead, n, blocks in pipes
    ]
