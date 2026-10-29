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

r"""Validity adapter for the DeepSeek-V4 KV compress-and-store kernels
(`proj_and_save_state`, `compress_norm_rope_store`) at DeepSeek-V4-Flash-0731
DP8/EP8 serving shapes.

Run it through perfkit, never directly:

    perfkit validity run \
        --adapter scripts/forge_validity/dsv4_flash_compress_store_validity.py \
        --arm <checkout>

`perfkit/validity_arm.py` defines the hooks and the two checks (`aot_all_shapes`,
`pipeline_trip_count`). Nothing here uses a TPU: compiles are ahead of time for
the tpu7x topology on the CPU, and trip counts are read while tracing.

What it checks
--------------
The entry is `compressor_v1.compressor_forward`, the pure-jax body of the three
serving ops in `layers/adapter/custom_ops/deepseek_v4/deepseek_v4_compressor.py`
(which imports torch). It derives the slot mappings, runs `proj_and_save_state`,
packs the boundary tokens and runs `compress_norm_rope_store`. It is called as
`_compressor_jax_{csa,hca,indexer}` call it, returning what each returns, with
the caches donated as `pallas.jax_op` donates them, under
`deepseek-v4-flash-dp8-ep`. `SHAPES` is each layer kind that runs a compressor
times each token count the runner pads an engine's batch to:

  csa      the 21 compress-ratio-4 attention layers: head_dim 512, overlap,
           fp8 records with quant block 64 in an int32 NoPE array that also
           hosts the states, bf16 RoPE in a second int32 array.
  indexer  the indexer compressor of those 21 layers: head_dim 128, ratio 4,
           overlap, quant block 128, one SEQ_ALONG_LANE uint8 array for records
           and states (`seq_along_lane=True`).
  hca      the 20 compress-ratio-128 layers: head_dim 512, no overlap, bf16
           records in a uint8 array, states on an int32 CSA NoPE array.

Where each value comes from:

- Model dims, from the checkpoint's `config.json`: hidden 4096, head_dim 512,
  qk_rope_head_dim 64, index_head_dim 128, `compress_ratios` (4 and 128),
  rms_norm_eps 1e-6, and a cos/sin cache of 65536 x 16 = 1048576 rows of 64
  f32 (`rope_scaling`; vLLM's DeepSeek-V4 rotary embedding keeps it in f32).
- Operand dtypes and widths, from vLLM's `DeepseekCompressor`: bf16 hidden
  states and `fused_wkv_wgate` (stored transposed, `[4096, 2 coff head_dim]`),
  f32 `ape` `[ratio, coff head_dim]`, bf16 norm weight, quant block 64 at
  head_dim 512 and 128 at 128.
- Token paddings 16 to 1024, from the server's `compile_sizes`
  (`--max-num-batched-tokens=1024`). 256 requests per forward
  (`--max-num-seqs=256`; the runner logs "requests per forward: 256"), so
  `query_start_loc` has 257 entries and each block table 256 rows.
- Cache arrays, as `runner/kv_cache_dsv4.py` allocates them at KV block size
  1024 (the server's) and 14579 blocks per engine: CSA NoPE int32
  `[14579, 256, 128]` and RoPE `[14579, 64, 128]`, indexer uint8
  `[14579, 33, 4, 256]`, HCA uint8 `[14579, 16, 4, 128]`.
- State block sizes, from the arm's own `config.state_block_size` as
  `VllmCompressorStateCache` derives them (16 for csa and indexer, 32 for hca
  on golden), and block tables `cdiv(9216, block size)` wide: state tables 576
  or 288, KV tables 9.

`aot_all_shapes` compiles the whole op for tpu7x at each shape, so a candidate
that changes how a kernel is called is compiled as the server compiles it.
`limit` refuses a program in which no Mosaic kernel was launched from
`KERNEL_DIR` (read from the lowered module's source locations): that is a
program this adapter did not check, not a candidate it passed.

`trip_counts` traces the same entry, without compiling, with
`pltpu.emit_pipeline` replaced by a recorder that calls the real one and reads,
when the pipeline is run, the buffer count and lookahead of every input it
copies in: from the `BufferedRef` allocations when the kernel passes its own
(`compress_norm_rope_store` does, through `buffered_ref.py`), else from each
spec's `pipeline_mode` (`Buffered(n)`, else 2). A `pallas_call` from
`KERNEL_DIR` with a grid is recorded the same way, from its BlockSpecs (golden's
two have none). A static grid gives one record at its trip count, with the block
each step reads when the index maps are functions of the grid indices alone.

The runtime values that size a grid or pick a branch, and how they are covered:

- `compress_norm_rope_store`'s grid, `cdiv(last boundary slot + 1, 4)`: from
  the positions, `query_start_loc`, `distribution` and the KV block table, it is
  0 to `cdiv(T + 4 (T // ratio + 1 + 256), 4)` steps. Golden's page-buffer,
  cos/sin and RoPE-row pipes are reported at every N = 1 .. max(B, 8): warmup's
  dummy batches and steps that end few compressed blocks reach the small end.
  The adapter does not work out which N a shape can reach, which is
  conservative.
- `proj_and_save_state`'s grid is static: `(cdiv(T, tile_n), 2, 2)` steps with
  `tile_n = min(128, T rounded up to 8)` and a 2048-wide K tile, 4 to 32 steps.
- Data-dependent branches inside the kernels (slot -1 for padding tokens, the
  first-in-tile row masks, which state pages a window spans) change which DMAs
  run, not any grid; they are not enumerated.

What it cannot check
--------------------
- Nothing runs: no output is compared, and a wrong record, scale, slot or RoPE
  layout passes wherever it still compiles. The campaign's numerics tier has to
  catch those.
- Only pipeline schedules are read. The DMAs a kernel starts itself are not
  counted: `proj_and_save_state`'s state writes and their semaphore waits, the
  page-buffer and record copies inside `buffered_ref.py`'s `copy_in`/`copy_out`
  (the pipeline waits for each slot as a whole), and the SEQ_ALONG_LANE store
  after the pipeline (`kv_cache_append._write_runs`). An unwaited manual copy
  passes. Output-only pipes are not recorded (jax refuses more than 2 buffers
  for them).
- Runtime grids are not bounded per shape (above).
- Shapes are the serve config's, at golden's cache layout: another block size,
  KV dtype, or the indexer's head-along-sublane layout is not compiled.
"""

from __future__ import annotations

import contextlib
import importlib
import itertools
import math
import os
import re
import sys
import types
from pathlib import Path

KERNEL_DIR = "src/vllm_torchtpu/kernels/deepseek_v4/compress_and_store"
TOPOLOGY = "tpu7x:2x2x1"

HIDDEN = 4096
ROPE_HEAD_DIM = 64
COS_SIN_ROWS = 65536 * 16  # original_max_position_embeddings x yarn factor
RMS_EPS = 1e-6
MAX_MODEL_LEN = 9216
KV_BLOCK = 1024  # tokens per KV cache block
NUM_BLOCKS = 14579
NUM_REQS = 256
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
#: Runtime grids are reported at N = 1 .. max(B, this).
RUNTIME_TRIPS = 8

#: Per layer kind: head_dim, compress ratio, quant block, and which second
#: array the op takes (`rope` for csa, `state` for hca).
MODES = {
    "csa": {"head_dim": 512, "ratio": 4, "quant_block": 64, "second": "rope"},
    "indexer": {"head_dim": 128, "ratio": 4, "quant_block": 128, "second": None},
    "hca": {"head_dim": 512, "ratio": 128, "quant_block": 64, "second": "state"},
}
#: What a runtime grid dim stands for, by the function that emits the pipeline.
RUNTIME_MEANING = {
    "kernel_fn": "compress grid steps, cdiv(last boundary slot + 1, tile_n)",
}

SHAPES = {
    f"{mode}_t{t}": {"mode": mode, "tokens": t}
    for mode in MODES
    for t in TOKEN_PADDINGS
}


def _load(root: Path):
    """Import `compressor_v1` from the arm root without the `__init__`s of
    `vllm_torchtpu` and `vllm_torchtpu.kernels`, which import torch and the
    serving stack. `env_override` is loaded for the server's LIBTPU_INIT_ARGS
    and import-time Pallas patches; everything below imports normally."""
    pkg = root / "src" / "vllm_torchtpu"

    def package(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        return mod

    top = package("vllm_torchtpu", pkg)
    package("vllm_torchtpu.kernels", pkg / "kernels")
    for name in ("envs", "logger", "patch_registry"):
        setattr(top, name, importlib.import_module(f"vllm_torchtpu.{name}"))
    importlib.import_module("vllm_torchtpu.env_override")
    base = "vllm_torchtpu.kernels.deepseek_v4.compress_and_store"
    return (
        importlib.import_module(f"{base}.compressor_v1"),
        importlib.import_module(f"{base}.config"),
    )


def setup(root):
    # CPU only: jax must never open the host's TPU, and the compile-only
    # topology does not need the libtpu lock.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-4")
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    forward, config = _load(root)

    import jax
    import jax.numpy as jnp

    return types.SimpleNamespace(
        forward=forward,
        config=config,
        kernel_dir=root / KERNEL_DIR,
        jax=jax,
        jnp=jnp,
        device=None,
    )


def origin(handle) -> str:
    return handle.forward.__file__


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


def _statics(handle, params):
    """The static arguments `_build_compressor_op` binds for one layer kind."""
    mode = MODES[params["mode"]]
    ratio, head_dim = mode["ratio"], mode["head_dim"]
    cfg = handle.config
    block_size = cfg.state_block_size(
        cfg.select_mode(head_dim, ratio == 4),
        KV_BLOCK,
        compress_ratio=ratio,
        head_dim=head_dim,
        indexer_seq_along_lane=True,
    )
    return {
        "state_block_size": block_size,
        "head_dim": head_dim,
        "compress_ratio": ratio,
        "overlap": ratio == 4,
        "rms_eps": RMS_EPS,
        "quant_block": mode["quant_block"],
    }


def _entry(handle, params):
    """`compressor_forward` as `_compressor_jax_<mode>` calls it, returning the
    caches that op returns."""
    statics = _statics(handle, params)
    second = MODES[params["mode"]]["second"]

    def op(*operands):
        *inputs, cache = operands[:11]
        extra = operands[11] if second else None
        new_cache, rope, state = handle.forward.compressor_forward(
            *inputs[:9],
            cache=cache,
            rope_cache=extra if second == "rope" else None,
            distribution=inputs[9],
            state_cache=extra if second == "state" else None,
            seq_along_lane=second is None,
            **statics,
        )
        if second == "rope":
            return new_cache, rope
        if second == "state":
            return new_cache, state
        return new_cache

    return op


def _args(handle, params):
    """Operands in `compressor_forward`'s positional order, then the caches."""
    jax, jnp, cfg = handle.jax, handle.jnp, handle.config
    if handle.device is None:
        from jax.experimental import topologies

        handle.device = topologies.get_topology_desc(TOPOLOGY, platform="tpu").devices[
            0
        ]
    sharding = jax.sharding.SingleDeviceSharding(handle.device)
    mode = MODES[params["mode"]]
    tokens, head_dim, ratio = params["tokens"], mode["head_dim"], mode["ratio"]
    state_width = (1 + int(ratio == 4)) * head_dim
    state_table = math.ceil(
        MAX_MODEL_LEN / _statics(handle, params)["state_block_size"]
    )
    kv_table = math.ceil(MAX_MODEL_LEN / KV_BLOCK)
    page = KV_BLOCK // ratio  # compressed tokens per KV block
    csa_nope = ((NUM_BLOCKS, KV_BLOCK // 4, 128), jnp.int32)
    caches = {
        "csa": [csa_nope, ((NUM_BLOCKS, KV_BLOCK // 4 // 4, 128), jnp.int32)],
        "indexer": [
            ((NUM_BLOCKS, cfg.SAL_SUBLANE_GROUPS, 4, page), jnp.uint8),
        ],
        "hca": [((NUM_BLOCKS, page * 2, 4, 128), jnp.uint8), csa_nope],
    }[params["mode"]]
    specs = [
        ((tokens, HIDDEN), jnp.bfloat16),  # hidden_states
        ((HIDDEN, 2 * state_width), jnp.bfloat16),  # wkv_wgate, transposed
        ((ratio, state_width), jnp.float32),  # ape
        ((head_dim,), jnp.bfloat16),  # norm_weight
        ((COS_SIN_ROWS, ROPE_HEAD_DIM), jnp.float32),  # cos_sin_cache
        ((tokens,), jnp.int32),  # positions
        ((NUM_REQS * state_table,), jnp.int32),  # state block table
        ((NUM_REQS + 1,), jnp.int32),  # query_start_loc
        ((NUM_REQS * kv_table,), jnp.int32),  # KV block table
        ((3,), jnp.int32),  # request distribution
        *caches,
    ]
    return [jax.ShapeDtypeStruct(*s, sharding=sharding) for s in specs]


def _lower(handle, params):
    jax = handle.jax
    args = _args(handle, params)
    donated = tuple(range(10, len(args)))
    with _tpu7x_mesh():
        return (
            jax.jit(_entry(handle, params), donate_argnums=donated)
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )


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


def compile(handle, name, params):
    lowered = _lower(handle, params)
    mem = lowered.compile().memory_analysis()
    text = lowered.as_text(debug_info=True)
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "kernels": text.count("tpu_custom_call"),
        "kernels_here": ",".join(sorted(set(_kernels_from(text, handle.kernel_dir)))),
    }


def limit(name, result):
    if not result["kernels_here"]:
        return (
            f"the compiled program launches no Mosaic kernel from {KERNEL_DIR}: "
            "nothing was checked"
        )
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
    """The block each step of a grid reads, in step order, or None when the
    index map is not a function of the grid indices alone."""
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
                int(mode.buffer_count) if mode is not None else 2,
                bool(mode.use_lookahead) if mode is not None else False,
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
    seen = []
    real = {"emit_pipeline": pltpu.emit_pipeline, "pallas_call": pl.pallas_call}

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
        if grid and Path(caller.co_filename).resolve().is_relative_to(
            handle.kernel_dir
        ):
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


def _block_list(index_map, grid, trips):
    """`_blocks` for a static grid, or for `trips` steps of a 1-D runtime grid."""
    if trips is None:
        return None
    return _blocks(
        index_map, grid if all(isinstance(d, int) for d in grid) else (trips,)
    )


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
                trips = [(p["trips"], f"static grid {p['grid']}")]
            else:
                meaning = RUNTIME_MEANING.get(
                    p["function"], f"runtime grid of {p['site']}"
                )
                trips = [
                    (n, f"{meaning} = {n}") for n in range(1, max(b, RUNTIME_TRIPS) + 1)
                ]
            one_dim = len(p["grid"]) == 1 or p["trips"] is not None
            for n, described in trips:
                record = {
                    "pipe": pipe,
                    "buffer_count": b,
                    "trip_count": n,
                    "lookahead": lookahead,
                    "input": described,
                }
                blocks = _block_list(index_map, p["grid"], n) if one_dim else None
                if blocks is not None:
                    record["blocks"] = blocks
                records.append(record)
    return records
