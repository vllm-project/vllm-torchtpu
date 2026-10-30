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
r"""Validity adapter for DeepSeek-V4 CSA sparse attention at Flash DP8/EP8 shapes.

Run it through perfkit, never directly:

    perfkit validity run --adapter <this file> --arm <checkout> \
        --python "$(perfkit env ensure --jax 0.11.2 --libtpu 0.0.50)"

`perfkit/validity_arm.py` defines the hooks and the two checks. Nothing here
uses a TPU: compiles are ahead of time for the tpu7x topology on the CPU, and
trip counts are read while tracing on the CPU.

What is compiled
----------------
The entry point is `sparse_ragged_paged_attention` in
`kernels/deepseek_v4/core_attention/sparse_mla.py`, the function the CSA
layers' attention op (`_attention_csa` in
`layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py`) calls.
One call is the whole block: `gather_page_ids` (TensorCore), then per token
chunk the SparseCore gather `sc_csa_gather` (`csa_gather.py`) and the
`SparseMLA-p_<topk>-bz_<batch>-gcz_<chunk>` attention kernel. The sibling
`kernels/mla/sparse/kernel.py` is the DeepSeek-V3.2 path
(`layers/core/attention_interface.py`) and is not reached by this model.

The op picks `gather_and_attention_chunk_size` and
`attention_kernel_batch_size` from the token count. `compile` reads that rule
from the arm's own op file (the static keyword arguments of the
`sparse_ragged_paged_attention` call in `_attention_csa` and the plain
assignments that set them), so an arm that retunes it is compiled as its
server would call it.

Shapes
------
The serving config is `scripts/vllm/benchmarking/configs/
deepseek-v4-flash-dp8-ep.sh`: TP1 x DP8 with EP, `--max-num-batched-tokens
1024`, `--max-num-seqs 256`, `--max-model-len 9216`, `fp8_e4m3` KV cache.
Attention is data parallel, so each engine calls the kernel on its own batch.

* Token buckets. An engine pads a step to one of 16, 32, ..., 512, 1024
  tokens (`_get_token_paddings` in `platforms/tpu_platform.py`, doubling from
  16 to the token budget), and DP lockstep pads every engine to the same
  bucket. Decode and prefill run the same program at a bucket: the kernel
  treats each query token alike, and what separates them is runtime data.
  The op uses chunks of 32 tokens with batches of 8 up to 128 tokens, and
  chunks of 128 with batches of 16 above (golden's rule).
* Static model and cache geometry. 64 query heads of 512 (`q` and the SWA
  partials `bf16[T, 64, 512]`, `swa_l`/`swa_m` `f32[T, 64]`), top-k 512.
  The KV block is 1024 tokens, which CSA compresses 4x into 256 rows a page;
  the CSA arrays are `int32[pages, 256, 128]` (NoPE) and
  `int32[pages, 64, 128]` (RoPE) (`csa_cache_layout.py`). 12628 pages, the
  block count the layer-run scorer fixes; the page count changes only the
  argument bytes, not the kernel's tiling. `cu_q_lens` is `int32[257]`
  (256 requests a forward), and the page table `int32[2304]` is 256 x 9
  pages (`cdiv(9216, 1024)`).

Golden's server profiles show the same arrays and kernel names: decode steps
at the 128 bucket run `SparseMLA-p_512-bz_8-gcz_32`, prefill steps at 1024
run `SparseMLA-p_512-bz_16-gcz_128`, with that cache and that page table.

Runtime values, none of which changes the compiled program:

* `cu_q_lens[distribution[2]]`, the valid tokens of the step (1 to the
  bucket): it sizes each chunk's attention grid,
  `cdiv(min(valid, chunk_end) - chunk_start, batch)` steps, 0 to
  `chunk / batch`, and the SparseCore gather skips each top-k period at or
  past `valid * 512`.
* Per-token valid top-k entries (`topk_indices != -1`, 0 to 512): a prompt
  shorter than 2048 tokens has fewer than 512 compressed rows, at most
  `9216 / 4 = 2304`. They mask scores, not the grid.
* The page table and request distribution (1 to 256 requests, up to 9
  pages each): they only steer which rows are gathered.

That is 7 shapes, one per bucket, which covers every program the server
compiles for this kernel.

Checks
------
`compile` builds each shape through the arm's own entry point, so the
tiling is the kernel's own choice for that shape, and compiles it for
tpu7x. `limit` refuses a program that launches no Mosaic kernel, since such a
compile checked nothing.

`trip_counts` traces each shape again and reads every `emit_pipeline` call
made while tracing: its grid, and the buffer count and lookahead of each
input spec (2 buffers and no lookahead for a spec without a pipeline mode).
Golden's only pipelines are the two in the SparseCore gather, an outer one
over index blocks (1 step at 16 to 128 tokens, 4 at 256 to 1024) and an
inner one of 8 steps per block, all double-buffered. Their grids are fixed
at trace time, so one record per pipe holds for every runtime input. Block
indices are not reported, since the gather's index maps read runtime data,
and `pipeline_trip_count` then applies its trip-count window alone.

The attention kernel itself pipelines with its own DMAs into two-slot
scratch, and `gather_page_ids` uses the plain `pallas_call` grid; neither is
an `emit_pipeline`, so neither is checked here, and a `pallas_call` input
spec with a deeper pipeline mode is not either. A pipe of 2 or 3 buffers has
no unsafe trip count, so one whose grid is sized at run time gives no record.
One of 4 or more buffers whose grid is sized at run time, or split across
cores, cannot be read at trace time, and `trip_counts` raises for it rather
than guess.

Calibration
-----------
With this adapter on the CPU (jax 0.11.2, libtpu 0.0.50), against golden
e33dea2d5 and three copies of it, each broken in a way a fixed-shape
benchmark at the common buckets can miss:

  * golden passes both checks: 7 shapes compile (the largest temp
    allocation 341688576 bytes, at 1024 tokens), and 70 pipe records, all at
    B=2.
  * Batch 64 instead of 16 above 128 tokens, in the op's tiling rule, fails
    `aot_all_shapes` at 256, 512 and 1024 tokens (scoped VMEM out of memory,
    66.23M of 63.94M) and passes `pipeline_trip_count`.
  * The SparseCore gather asserting whole index blocks instead of padding
    them, true at every bucket but 16, fails both checks at 16 tokens only.
  * The gather's outer index pipe at 6 buffers compiles at all 7 shapes and
    fails `pipeline_trip_count` at 256, 512 and 1024 tokens (B=6, N=4); at
    16 to 128 tokens that pipe has one step and passes.

A run takes 25 to 60 seconds (a VMEM failure compiles slowest).
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

# The package root: the tree a verdict names must cover the op file the
# tiling rule is read from as well as the kernels.
KERNEL_DIR = "src/vllm_torchtpu"
KERNEL_FILE = f"{KERNEL_DIR}/kernels/deepseek_v4/core_attention/sparse_mla.py"
OP_FILE = (
    f"{KERNEL_DIR}/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py"
)
OP_FUNCTION = "_attention_csa"
ENTRY = "sparse_ragged_paged_attention"
TOPOLOGY = "tpu7x:2x2x1"

NUM_HEADS = 64
HEAD_DIM = 512
TOPK = 512
CSA_RATIO = 4
BLOCK_TOKENS = 1024
PAGE_ROWS = BLOCK_TOKENS // CSA_RATIO
ROW_WORDS = 128
ROPE_ROWS = PAGE_ROWS // 4
NUM_PAGES = 12628
MAX_NUM_SEQS = 256
PAGES_PER_SEQ = -(-9216 // BLOCK_TOKENS)
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
SHAPES = {f"t{tokens}": {"tokens": tokens} for tokens in TOKEN_PADDINGS}

# The entry point's array parameters, in order; every other keyword of the
# op's call is a static tiling choice.
ARRAY_ARGS = (
    "q",
    "cache_kv_nope",
    "cache_kv_rope",
    "topk_indices",
    "page_indices",
    "cu_q_lens",
    "distribution",
    "attention_sinks",
    "swa_accumution",
    "swa_l",
    "swa_m",
)
SM_SCALE = HEAD_DIM**-0.5

# A pipe with at most this many buffers starts no copy that no step waits
# for at any trip count (the window 2 <= N <= B - 2 is empty).
SAFE_BUFFERS = 3


def _load_kernel(root: Path):
    """Import sparse_mla from the arm root without the package's own `__init__`.

    That `__init__` imports torch and the serving stack, which a compile-only
    environment does not have. `env_override` is loaded because it sets the
    server's LIBTPU_INIT_ARGS and applies its import-time Pallas patches, so
    the program compiled here is the one the server compiles. The kernel's
    own imports (`csa_gather`, `csa_cache_layout`) resolve through the stub
    packages, whose paths are the arm's.
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
    package(
        "vllm_torchtpu.kernels.deepseek_v4.core_attention",
        pkg / "kernels" / "deepseek_v4" / "core_attention",
    )
    kernel = module(
        "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla",
        root / KERNEL_FILE,
    )
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, "__file__", None)
        if name.startswith("vllm_torchtpu.") and path:
            if not Path(path).resolve().is_relative_to(root):
                raise RuntimeError(f"{name} was imported from {path}, outside {root}")
    return kernel


def _assigns_only(stmt, names) -> bool:
    """Whether `stmt` only assigns `names`, directly or under an `if`."""
    if isinstance(stmt, ast.Assign):
        return all(isinstance(t, ast.Name) and t.id in names for t in stmt.targets)
    if isinstance(stmt, ast.If):
        return all(_assigns_only(s, names) for s in stmt.body + stmt.orelse)
    return False


def _tiling_rule(op_file: Path):
    """tokens -> the static keyword arguments the op passes the entry point.

    Reads the `sparse_ragged_paged_attention` call in `_attention_csa`. A
    keyword bound to a name is evaluated by running only the statements of
    that function that assign such names, with `q` standing for the query
    of the given token count.
    """
    tree = ast.parse(op_file.read_text(), str(op_file))
    fns = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == OP_FUNCTION
    ]
    if len(fns) != 1:
        raise RuntimeError(f"{op_file} defines {OP_FUNCTION} {len(fns)} times")
    calls = [
        n
        for n in ast.walk(fns[0])
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == ENTRY
    ]
    if len(calls) != 1:
        raise RuntimeError(f"{OP_FUNCTION} calls {ENTRY} {len(calls)} times")
    call = calls[0]
    if call.args or any(kw.arg is None for kw in call.keywords):
        raise RuntimeError(f"{OP_FUNCTION} calls {ENTRY} with positional or ** args")
    static = [kw for kw in call.keywords if kw.arg not in (*ARRAY_ARGS, "sm_scale")]
    names = {kw.value.id for kw in static if isinstance(kw.value, ast.Name)}
    body = [s for s in fns[0].body if _assigns_only(s, names)]
    # `compile` is this adapter's own hook.
    code = builtins.compile(
        ast.Module(body=body, type_ignores=[]), str(op_file), "exec"
    )

    def rule(tokens):
        env = {"q": types.SimpleNamespace(shape=(tokens, NUM_HEADS, HEAD_DIM))}
        exec(code, {"__builtins__": {}}, env)
        out = {}
        for kw in static:
            if isinstance(kw.value, ast.Name):
                if kw.value.id not in env:
                    raise RuntimeError(
                        f"{OP_FUNCTION} passes {kw.arg}={kw.value.id}, which "
                        "no plain assignment in it sets"
                    )
                out[kw.arg] = env[kw.value.id]
            else:
                out[kw.arg] = ast.literal_eval(kw.value)
        return out

    return rule


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
    # Installed before the kernel is imported, so a kernel that binds
    # `emit_pipeline` by name at import time gets the recorder too.
    _install_pipeline_recorder(handle)
    handle.kernel = _load_kernel(root)
    handle.tiling = _tiling_rule(root / OP_FILE)

    import jax
    import jax.numpy as jnp

    handle.jax, handle.jnp = jax, jnp
    return handle


def origin(handle) -> str:
    return handle.kernel.__file__


def _entry(handle, params):
    """The entry point called as `_attention_csa` calls it, at one shape."""
    tiling = handle.tiling(params["tokens"])
    fn = getattr(handle.kernel, ENTRY)

    def call(*arrays):
        return fn(**dict(zip(ARRAY_ARGS, arrays)), sm_scale=SM_SCALE, **tiling)

    return call


def _arg_specs(handle, params):
    jnp, t = handle.jnp, params["tokens"]
    return [
        ((t, NUM_HEADS, HEAD_DIM), jnp.bfloat16),
        ((NUM_PAGES, PAGE_ROWS, ROW_WORDS), jnp.int32),
        ((NUM_PAGES, ROPE_ROWS, ROW_WORDS), jnp.int32),
        ((t, TOPK), jnp.int32),
        ((MAX_NUM_SEQS * PAGES_PER_SEQ,), jnp.int32),
        ((MAX_NUM_SEQS + 1,), jnp.int32),
        ((3,), jnp.int32),
        ((NUM_HEADS,), jnp.float32),
        ((t, NUM_HEADS, HEAD_DIM), jnp.bfloat16),
        ((t, NUM_HEADS), jnp.float32),
        ((t, NUM_HEADS), jnp.float32),
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


def _trace(handle, params):
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
        return jax.jit(_entry(handle, params)).trace(*args)


def compile(handle, name, params):
    lowered = _trace(handle, params).lower(lowering_platforms=("tpu",))
    mem = lowered.compile().memory_analysis()
    text = lowered.as_text()
    start = text.find("SparseMLA-p_")
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "mosaic": "tpu_custom_call" in text,
        "kernel": text[start : start + 200].split('"')[0] if start >= 0 else "",
        **handle.tiling(params["tokens"]),
    }


def limit(name, result):
    if not result["mosaic"]:
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
                    getattr(body, "__qualname__", type(body).__name__),
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
