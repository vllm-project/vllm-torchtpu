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
r"""Validity adapter for DeepSeek-V4 KV compress and store at Flash DP8/EP8 shapes.

Run it through perfkit, never directly:

    perfkit validity run --adapter <this file> --arm <checkout> \
        --python "$(perfkit env ensure --jax 0.11.2 --libtpu 0.0.50)"

`perfkit/validity_arm.py` defines the hooks and the two checks. Nothing here
uses a TPU: compiles are ahead of time for the tpu7x topology on the CPU, and
trip counts are read while tracing on the CPU.

What is compiled
----------------
The entry point is `compressor_forward` in
`kernels/deepseek_v4/compress_and_store/compressor_v1.py`, which the op
(`_run_compressor` in `layers/adapter/custom_ops/deepseek_v4/
deepseek_v4_compressor.py`) calls for every compressor. One call is the
whole block: `proj_and_save_state` (`proj_and_save_state.py`) projects the
hidden states and writes the f32 state, then `compress_norm_rope_store`
(`kernel.py`) compresses, normalizes, applies RoPE and stores each boundary
record. `compile` passes the static arguments `_build_compressor_op` binds:
head_dim, compress_ratio, overlap (`compress_ratio == 4`), quant_block (64
at head_dim 512, 128 at 128), and `state_block_size` from the arm's own
`config.state_block_size` for the 1024-token KV block, as
`VllmCompressorStateCache._derive_block_size` asks it. Both kernels pick
their tiles inside, so an arm that retunes them is compiled as its server
would call it. The op donates its caches, and so does `compile`.

Shapes
------
The serving config is `scripts/vllm/benchmarking/configs/
deepseek-v4-flash-dp8-ep.sh`: TP1 x DP8 with EP, `--max-num-batched-tokens
1024`, `--max-num-seqs 256`, `--max-model-len 9216`, `fp8_e4m3` KV cache.
Attention is data parallel, so each engine calls the op on its own batch.

* Token buckets: 16, 32, ..., 512, 1024 (`_get_token_paddings` in
  `platforms/tpu_platform.py`, doubling from 16 to the token budget; DP
  lockstep pads every engine to the same bucket). Decode and prefill run
  the same program at a bucket.
* Compressors: CSA layers (head_dim 512, 4x, state on the NoPE array,
  RoPE in `int32[12628, 64, 128]`, NoPE `int32[12628, 256, 128]`,
  state_block_size 16), HCA layers (512, 128x, records `u8[12628, 16, 4,
  128]`, state on a CSA NoPE array `int32[12628, 256, 128]`,
  state_block_size 32), and the CSA layers' indexer, which uses the same op
  (`deepseek_v4_indexer.py`; head_dim 128, 4x, one `u8[12628, 64, 4, 256]`
  array for records and state, state_block_size 16).
* Other operands: hidden `bf16[T, 4096]`, `wkv_wgate` `bf16[4096, 2 *
  width]` (stored transposed), `ape` `f32[ratio, width]` with width =
  head_dim x (1 + overlap), norm weight `bf16[head_dim]`, RoPE table
  `f32[1048576, 64]`, positions `int32[T]`, state block tables
  `int32[256 x cdiv(9216, state_block_size)]`, KV block tables
  `int32[2304]` (256 x 9), `query_start_loc` `int32[257]`. 12628 pages, the
  block count the layer-run scorer fixes.

Golden's server profile at the 1024 bucket shows these arrays: the
`proj_and_save_state` and `compress_norm_rope_store` calls of each kind,
with boundary batches of 3076 (CSA, indexer) and 2084 (HCA) slots and state
tables of 147456 and 73728 entries. That is 21 shapes, 7 buckets x 3 kinds.

Checks
------
`compile` builds each shape through `compressor_forward` and compiles it for
tpu7x. Operands the op does not donate take the layout the program prefers,
as the server's resident buffers do. `limit` refuses a program that
launches no Mosaic kernel, since such a compile checked nothing.

`trip_counts` traces each shape again and records, for every
`emit_pipeline`, the buffered refs its prologue sets up
(`Scheduler.initialize_step` at step 0): each input's buffer count and
lookahead as the schedule runs them, whether they came from the specs or
from allocations the kernel passed (as `compress_norm_rope_store` does).
A pipe whose buffer count exceeds the pipeline's stages, so that the
prologue leaves a buffer unfilled, is refused rather than modelled.

* `proj_and_save_state` pipelines its weight, hidden-state and position
  tiles over a static grid `(num_n, num_m, num_k)`: (1, 2, 2) up to 128
  tokens, then (2, 2, 2), (4, 2, 2), (8, 2, 2). One record per input, with
  the block each step reads.
* `compress_norm_rope_store` sizes its grid at run time, from the last
  valid slot of the boundary batch (`grid_size`, scalar-prefetched). The
  adapter finds which kernel operand the grid is read from, runs the
  arm's entry point eagerly on all-zero operands of the traced shapes
  (every slot valid, the longest batch the kernel can be handed) with
  `pallas_call` stood in for, and enumerates 0 to that maximum: 0..265 to
  0..769 for CSA and the indexer, 0..261 to 0..521 for HCA, at tile 4. One
  record per input pipe per value: RoPE table rows, state pages, and the
  RoPE row (CSA) or record rows (indexer) read back for merging. Golden's
  grid, `cdiv(last valid slot + 1, 4)`, takes every value in that range as
  the last valid slot moves, and a decode step with few boundary tokens
  reaches the low end.

A grid sized at run time in more than one dimension, split across cores,
or outside the two entry points, is not enumerated: a pipe of at most 3
buffers there gives no record (it has no unsafe trip count), and a deeper
one makes `trip_counts` raise rather than guess.

Calibration
-----------
With this adapter on the CPU (jax 0.11.2, libtpu 0.0.50), against golden
44ce262ee and three copies of it, each broken in a way a fixed-shape
benchmark at the common buckets can miss:

  * golden passes both checks: 21 shapes compile (the largest temp
    allocation 10118080 bytes, at csa-t64), and 21641 pipe records over
    7994 runtime inputs, all at B=2: 51 for `proj_and_save_state`, the rest
    for `compress_norm_rope_store` (3 input pipes for CSA and the indexer, 2
    for HCA, at every runtime grid size).
  * `proj_and_save_state` with token tiles of up to 1024 and the whole
    hidden dimension as its K tile fails `aot_all_shapes` at csa-t1024
    only (scoped VMEM out of memory, 32.02M of 32.01M) and passes
    `pipeline_trip_count`.
  * The state-page pipe of `compress_norm_rope_store` at 4 buffers (spec
    and allocation) compiles at all 21 shapes and fails
    `pipeline_trip_count` at all 21: B=4 at a runtime grid of 2 steps.
  * The weight pipe of `proj_and_save_state` at 6 buffers compiles at all
    21 shapes and fails `pipeline_trip_count` at 16 to 128 tokens for each
    kind (12 shapes, B=6, N=4); from 256 tokens its grid has 8 or more
    steps and passes.

A run takes about 135 seconds, 240 when a shape runs out of VMEM.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import operator
import os
import sys
import types
from pathlib import Path

# The package root: the verdict's tree covers the kernels, the entry point
# that calls them and the paging rule in `config.py`.
KERNEL_DIR = "src/vllm_torchtpu"
PACKAGE = "vllm_torchtpu.kernels.deepseek_v4.compress_and_store"
PACKAGE_DIR = f"{KERNEL_DIR}/kernels/deepseek_v4/compress_and_store"
TOPOLOGY = "tpu7x:2x2x1"

TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
HIDDEN = 4096
ROPE_DIM = 64
MAX_POSITIONS = 1_048_576
KV_BLOCK_TOKENS = 1024
NUM_PAGES = 12628
MAX_NUM_SEQS = 256
MAX_MODEL_LEN = 9216
RMS_EPS = 1e-6

# The model's three compressors: the static arguments the op passes and the
# arrays the server allocates for them, each `(shape past the page count,
# dtype)`. `second` is CSA's RoPE array or HCA's state array.
KINDS = {
    "csa": {
        "head_dim": 512,
        "compress_ratio": 4,
        "quant_block": 64,
        "cache": ((256, 128), "int32"),
        "second": ("rope_cache", (64, 128), "int32"),
    },
    "hca": {
        "head_dim": 512,
        "compress_ratio": 128,
        "quant_block": 64,
        "cache": ((16, 4, 128), "uint8"),
        "second": ("state_cache", (256, 128), "int32"),
    },
    "indexer": {
        "head_dim": 128,
        "compress_ratio": 4,
        "quant_block": 128,
        "cache": ((64, 4, 256), "uint8"),
        "second": None,
    },
}
SHAPES = {
    f"{kind}-t{tokens}": {"kind": kind, "tokens": tokens}
    for kind in KINDS
    for tokens in TOKEN_PADDINGS
}

# The kernel entry points `compressor_forward` calls, as (module, function).
KERNEL_ENTRIES = (
    ("kernel", "compress_norm_rope_store"),
    ("proj_and_save_state", "proj_and_save_state"),
)

# A pipe with at most this many buffers starts no copy that no step waits
# for at any trip count (the window 2 <= N <= B - 2 is empty).
SAFE_BUFFERS = 3


def _load(root: Path):
    """Import `compressor_v1` from the arm root without the package `__init__`s.

    Those import torch and the serving stack, which a compile-only
    environment does not have, so the packages are stubs whose paths are the
    arm's and the kernels' own imports resolve through them. `env_override`
    is loaded because it sets the server's LIBTPU_INIT_ARGS and applies its
    import-time Pallas patches, so the program compiled here is the one the
    server compiles.
    """
    pkg = root / KERNEL_DIR

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
    package("vllm_torchtpu.kernels.deepseek_v4", pkg / "kernels" / "deepseek_v4")
    package(PACKAGE, root / PACKAGE_DIR)
    entry = module(f"{PACKAGE}.compressor_v1", root / PACKAGE_DIR / "compressor_v1.py")
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, "__file__", None)
        inside = path is None or Path(path).resolve().is_relative_to(root)
        if name.startswith("vllm_torchtpu.") and not inside:
            raise RuntimeError(f"{name} was imported from {path}, outside {root}")
    return entry


def setup(root):
    # CPU only: jax must never open the host's TPU, and the compile-only
    # topology does not need the libtpu lock.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    handle = types.SimpleNamespace(pipes=None, pipe=None, call=None, device=None)
    # Installed before the kernels are imported, so that a kernel binding a
    # pipeline function by name at import time is recorded too.
    _install_pipeline_recorder(handle)
    handle.entry = _load(root)
    handle.config = sys.modules[f"{PACKAGE}.config"]
    handle.kernel = sys.modules[f"{PACKAGE}.kernel"]
    _install_call_recorder(handle)

    import jax
    import jax.numpy as jnp

    handle.jax, handle.jnp = jax, jnp
    return handle


def origin(handle) -> str:
    return handle.kernel.__file__


def _statics(handle, kind):
    """The static arguments `_build_compressor_op` binds for one compressor.

    `state_block_size` is what `VllmCompressorStateCache._derive_block_size`
    asks the arm's `config` for, from the 1024-token KV block.
    """
    k = KINDS[kind]
    overlap = k["compress_ratio"] == 4
    cfg = handle.config
    mode = cfg.select_mode(k["head_dim"], overlap)
    return {
        "state_block_size": cfg.state_block_size(
            mode,
            KV_BLOCK_TOKENS,
            compress_ratio=k["compress_ratio"],
            head_dim=k["head_dim"],
        ),
        "head_dim": k["head_dim"],
        "compress_ratio": k["compress_ratio"],
        "overlap": overlap,
        "rms_eps": RMS_EPS,
        "quant_block": k["quant_block"],
    }


def _arg_specs(handle, params):
    """The op's operands, in its order: the ten inputs, then the caches."""
    jnp, t = handle.jnp, params["tokens"]
    k = KINDS[params["kind"]]
    statics = _statics(handle, params["kind"])
    width = (1 + statics["overlap"]) * k["head_dim"]
    state_blocks = -(-MAX_MODEL_LEN // statics["state_block_size"])
    kv_blocks = -(-MAX_MODEL_LEN // KV_BLOCK_TOKENS)
    specs = [
        ((t, HIDDEN), jnp.bfloat16),  # hidden_states
        ((HIDDEN, 2 * width), jnp.bfloat16),  # wkv_wgate, stored transposed
        ((k["compress_ratio"], width), jnp.float32),  # ape
        ((k["head_dim"],), jnp.bfloat16),  # norm_weight
        ((MAX_POSITIONS, ROPE_DIM), jnp.float32),  # cos_sin_cache
        ((t,), jnp.int32),  # positions
        ((MAX_NUM_SEQS * state_blocks,), jnp.int32),  # state_block_tables
        ((MAX_NUM_SEQS + 1,), jnp.int32),  # query_start_loc
        ((MAX_NUM_SEQS * kv_blocks,), jnp.int32),  # k_block_tables
        ((3,), jnp.int32),  # request_distribution
    ]
    shape, dtype = k["cache"]
    specs.append(((NUM_PAGES, *shape), jnp.dtype(dtype)))
    if k["second"] is not None:
        _, shape, dtype = k["second"]
        specs.append(((NUM_PAGES, *shape), jnp.dtype(dtype)))
    return specs


def _entry(handle, params):
    """`compressor_forward` called as the op's `_compressor_jax_<kind>` calls it."""
    kind = params["kind"]
    statics = _statics(handle, kind)
    second = KINDS[kind]["second"]
    names = (
        "hidden_states",
        "wkv_wgate",
        "ape",
        "norm_weight",
        "cos_sin_cache",
        "positions",
        "block_table",
        "query_start_loc",
        "kv_block_table",
        "distribution",
        "cache",
    )

    def call(*arrays):
        kwargs = dict(zip(names, arrays))
        kwargs["rope_cache"] = kwargs["state_cache"] = None
        if second is not None:
            kwargs[second[0]] = arrays[len(names)]
        out_cache, rope_cache, state_cache = handle.entry.compressor_forward(
            **kwargs, **statics
        )
        if second is None:
            return out_cache
        return out_cache, {"rope_cache": rope_cache, "state_cache": state_cache}[
            second[0]
        ]

    return call


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
    from jax.experimental.layout import Format, Layout

    sharding = jax.sharding.SingleDeviceSharding(handle.device)
    specs = _arg_specs(handle, params)
    args = [jax.ShapeDtypeStruct(s, d, sharding=sharding) for s, d in specs]
    # The op donates its caches, the operands past the first ten. The other
    # operands take the layout the program prefers, as the server's resident
    # buffers do; a default layout would add a relayout copy of the RoPE
    # table (512 MiB of temp) that the server's program does not have.
    n_inputs = 10
    in_shardings = [Format(Layout.AUTO, sharding)] * n_inputs
    in_shardings += [sharding] * (len(args) - n_inputs)
    fn = jax.jit(
        _entry(handle, params),
        in_shardings=tuple(in_shardings),
        donate_argnums=tuple(range(n_inputs, len(args))),
    )
    with _tpu7x_mesh():
        return fn.trace(*args)


def compile(handle, name, params):
    lowered = _trace(handle, params).lower(lowering_platforms=("tpu",))
    mem = lowered.compile().memory_analysis()
    text = lowered.as_text()
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "mosaic_kernels": text.count("tpu_custom_call"),
        **_statics(handle, params["kind"]),
    }


def limit(name, result):
    if not result["mosaic_kernels"]:
        return (
            "the compiled program launches no Mosaic kernel, so the compile "
            "checked nothing"
        )
    return None


# --- pipeline recording --------------------------------------------------


def _install_pipeline_recorder(handle):
    """Note each pipeline the kernels trace, and each buffer its prologue sets up.

    Every `emit_pipeline` builds its schedule in `_emit_pipeline`, and the
    prologue calls `Scheduler.initialize_step` once per buffered ref at step
    0. That ref is the one the loop runs, whether the kernel passed its own
    allocations (as `compress_norm_rope_store` does) or let the pipeline
    allocate from the specs, so its buffer count is the one that schedules
    the copies.
    """
    from jax._src.pallas.mosaic import pipeline as jax_pipeline

    real_emit = jax_pipeline._emit_pipeline
    real_init = jax_pipeline.Scheduler.initialize_step

    @functools.wraps(real_emit)
    def _emit_pipeline(body, **kwargs):
        pipeline = real_emit(body, **kwargs)
        if handle.pipes is None:
            return pipeline
        fn = body.func if isinstance(body, functools.partial) else body
        pipe = {
            "body": getattr(fn, "__qualname__", type(fn).__name__),
            "split": kwargs.get("num_cores") not in (None, 1),
            "call": handle.call,
            "prologues": [],
        }
        handle.pipes.append(pipe)

        @functools.wraps(pipeline)
        def run(*args, **kw):
            outer, handle.pipe = handle.pipe, pipe
            try:
                return pipeline(*args, **kw)
            finally:
                handle.pipe = outer

        return run

    @functools.wraps(real_init)
    def initialize_step(self, buffered_ref, src_ref, step=0, init_limit=None):
        pipe = handle.pipe
        if pipe is not None and step == 0:
            if not pipe["prologues"] or pipe["prologues"][-1]["scheduler"] is not self:
                pipe["prologues"].append(
                    {
                        "scheduler": self,
                        "num_stages": self.num_stages,
                        "dims": tuple(_grid_dim(d) for d in self.grid),
                        "brefs": [],
                    }
                )
            pipe["prologues"][-1]["brefs"].append(buffered_ref)
        return real_init(self, buffered_ref, src_ref, step=step, init_limit=init_limit)

    jax_pipeline._emit_pipeline = _emit_pipeline
    jax_pipeline.Scheduler.initialize_step = initialize_step


def _install_call_recorder(handle):
    """Note the arguments of each kernel entry point while it is traced.

    A grid sized at run time is evaluated by calling the entry point again on
    concrete operands of the same shapes (`_runtime_max`).
    """
    for mod_name, fn_name in KERNEL_ENTRIES:
        mod = sys.modules[f"{PACKAGE}.{mod_name}"]
        real = getattr(mod, fn_name)

        def wrapped(*args, _real=real, **kwargs):
            if handle.pipes is None:
                return _real(*args, **kwargs)
            outer, handle.call = (
                handle.call,
                (_real, _abstract(args), _abstract(kwargs)),
            )
            try:
                return _real(*args, **kwargs)
            finally:
                handle.call = outer

        setattr(mod, fn_name, functools.wraps(real)(wrapped))


def _abstract(tree):
    import jax

    return jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if hasattr(x, "aval") else x,
        tree,
    )


def _grid_dim(d):
    """A grid dimension: an int, or how to compute it from the kernel's operands.

    A dimension known only at run time is a value the kernel body derives
    from its operands. The body's trace so far says which operands and which
    equations; `evaluate(operands)` replays those equations on concrete
    operands. Only whole-ref reads are replayed, since an indexed read inside
    the body is not a function of the operands alone. A dimension that cannot
    be replayed raises when evaluated, so that a pipe that never needs it
    (one with few enough buffers) is not refused.
    """
    from jax._src import core

    if isinstance(d, int):
        return d
    frame = d._trace.frame
    active, needed = {d.val}, []
    for eqn in reversed(frame.get_eqns()):
        if active & set(eqn.outvars):
            needed.append(eqn)
            active -= set(eqn.outvars)
            active |= {v for v in eqn.invars if not isinstance(v, core.Literal)}
    needed.reverse()
    positions = {v: i for i, v in enumerate(frame.invars)}
    problem = None
    if not active <= set(positions):
        problem = "depends on values other than the kernel operands"
    elif any(e.primitive.name == "get" and len(e.invars) != 1 for e in needed):
        problem = "depends on an indexed read in the kernel body"

    def evaluate(operands):
        if problem:
            raise RuntimeError(f"a grid dimension sized at run time {problem}")
        env = {v: operands[positions[v]] for v in active}
        for eqn in needed:
            vals = [
                v.val if isinstance(v, core.Literal) else env[v] for v in eqn.invars
            ]
            if eqn.primitive.name == "get":
                outs = [vals[0]]
            else:
                outs = eqn.primitive.bind(*vals, **eqn.params)
                if not eqn.primitive.multiple_results:
                    outs = [outs]
            env.update(zip(eqn.outvars, outs))
        return int(env[d.val])

    return evaluate


class _Operands(Exception):
    """Raised by the stand-in `pallas_call` to hand back its operands."""


def _runtime_max(handle, call, evaluate):
    """The largest value the grid dimension takes for the entry point `call`.

    The entry point runs eagerly on the CPU on operands of the traced shapes,
    all zeros, with `pallas_call` stood in for to capture the kernel's
    operands. For `compress_norm_rope_store` zeros make every slot of the
    boundary batch valid, the longest batch the kernel can be handed.
    """
    from jax.experimental import pallas as pl

    jax, jnp = handle.jax, handle.jnp
    real, args, kwargs = call

    def concrete(tree):
        return jax.tree.map(
            lambda x: jnp.zeros(x.shape, x.dtype)
            if isinstance(x, jax.ShapeDtypeStruct)
            else x,
            tree,
        )

    def stand_in(*_, **__):
        def launch(*operands):
            raise _Operands([o for o in operands if o is not None])

        return launch

    real_pallas_call, pl.pallas_call = pl.pallas_call, stand_in
    try:
        with jax.disable_jit():
            real(*concrete(args), **concrete(kwargs))
    except _Operands as captured:
        return evaluate(captured.args[0])
    finally:
        pl.pallas_call = real_pallas_call
    raise RuntimeError(
        f"{real.__name__} launched nothing through jax.experimental.pallas.pallas_call"
    )


def _blocks(spec, grid):
    """The block index each step reads, in step order, for a static grid."""
    out = []
    for idx in _grid_indices(grid):
        block = spec.index_map(*idx)
        out.append(tuple(operator.index(b) for b in block))
    return out


def _grid_indices(grid):
    """Grid indices in pipeline order (the last dimension moves fastest)."""
    if not grid:
        return [()]
    return [(i, *rest) for i in range(grid[0]) for rest in _grid_indices(grid[1:])]


def _records(handle, i, pipe):
    for prologue in pipe["prologues"]:
        dims, stages = prologue["dims"], prologue["num_stages"]
        dynamic = [j for j, d in enumerate(dims) if callable(d)]
        top = None
        for bref in prologue["brefs"]:
            if not (bref.is_input and bref.is_buffered) or bref.is_trivial_windowing:
                continue
            b = bref.buffer_count
            label = (
                f"pipeline{i} (body {pipe['body']}) {type(bref).__name__} "
                f"block {tuple(bref.spec.block_shape)}"
            )
            if b > stages:
                # The prologue issues num_stages - 1 fetches, so the pipe
                # would wait on a buffer nothing filled.
                raise RuntimeError(
                    f"{label} has {b} buffers but the pipeline prologue fills "
                    f"{stages - 1}; this adapter cannot model that schedule"
                )
            record = {"pipe": label, "buffer_count": b, "lookahead": bref.use_lookahead}
            if not dynamic and not pipe["split"]:
                yield {
                    **record,
                    "trip_count": math.prod(dims),
                    "input": f"any runtime input (grid {dims} fixed at trace time)",
                    "blocks": _blocks(bref.spec, dims),
                }
                continue
            if b <= SAFE_BUFFERS and (pipe["split"] or len(dynamic) > 1):
                continue
            if pipe["split"] or len(dynamic) > 1 or pipe["call"] is None:
                why = (
                    "split across cores"
                    if pipe["split"]
                    else "sized at run time outside a known entry point"
                    if pipe["call"] is None
                    else "sized at run time in more than one dimension"
                )
                raise RuntimeError(
                    f"{label} has {b} buffers and a grid {why}; this adapter "
                    "cannot enumerate its trip counts"
                )
            (j,) = dynamic
            if top is None:
                top = _runtime_max(handle, pipe["call"], dims[j])
            if top < 2:
                raise RuntimeError(
                    f"{label}: the grid's runtime dimension is at most {top} "
                    "for a full batch, which says nothing about its range"
                )
            rest = math.prod(d for k, d in enumerate(dims) if k != j)
            for n in range(top + 1):
                yield {
                    **record,
                    "trip_count": rest * n,
                    "input": f"grid dimension {j} = {n} of 0..{top} (runtime)",
                }


def trip_counts(handle, name, params):
    from jax._src import config as jax_config

    if jax_config.use_emit_pipeline_primitive.value:
        raise RuntimeError("emit_pipeline schedules at lowering under this config")
    # A fresh trace: the nested jits are cached from `compile`, and a cached
    # trace runs no pipeline prologue.
    handle.jax.clear_caches()
    handle.pipes = []
    try:
        _trace(handle, params)
        pipes = handle.pipes
    finally:
        handle.pipes = None
    records = []
    for i, pipe in enumerate(pipes):
        records += _records(handle, i, pipe)
    return records
