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
r"""Validity adapter for the DeepSeek-V4 indexer top-k at Flash DP8/EP8.

Run it through perfkit, never directly:

    perfkit validity run --adapter <this file> --arm <checkout> \
        --python "$(perfkit env ensure --jax 0.11.2 --libtpu 0.0.50)"

`perfkit/validity_arm.py` defines the hooks and the two checks. Nothing here
uses a TPU: compiles are ahead of time for the tpu7x topology on the CPU, and
trip counts are read while tracing on the CPU.

What is compiled
----------------
The entry point is `streamindex_topk` in
`kernels/deepseek_v4/streamindex_topk/streamindex_topk.py`, the lightning
indexer's scoring and top-k that `_indexer_jax` in
`layers/adapter/custom_ops/deepseek_v4/deepseek_v4_indexer.py` runs for
every 4x-compressed (CSA) layer, after `rope_quant` has rotated and
quantized the queries. It is compiled alone, with the operands that
`rope_quant` and the weight folding hand it: fp8 (e4m3) queries and f32
indexer weights. One call issues one TensorCore `pallas_call` per pass,
each over a slice of the request distribution: the decode kernel on whole
batches of `decode_req_batch_size` decode sequences, the decode kernel
again, one sequence per tile, on the decode sequences left over, and the
mixed kernel on the rest, which carries every prefill token (golden:
`StreamIdxTC-d-bq_1-bkvp_32` twice and `StreamIdxTC-m-bq_128-bkvp_16`).
The score buffer then goes to one SparseCore top-k (`sc_topk_direct_*`).
With `chunk_tokens` set, the token axis is cut into chunks and each chunk
runs all of this on its own (the kernel names gain `-c<chunks>`).

`compile` reads the static arguments from the arm's own op file: every
keyword of the entry-point call in `_indexer_jax` that is not an array,
each evaluated with `_indexer_jax`'s parameters bound as the indexer of a
CSA layer binds them (`k` 512, `compression_ratio` 4, `n_head` 64,
`softmax_scale` 128**-0.5) and with `envs` the arm's own
`vllm_torchtpu.envs`, after the plain assignments in `_indexer_jax` and in
the function around the call. Every `TPU_STREAMIDX_*` variable is removed
from the environment before the arm is imported, because neither the serve
config (`scripts/vllm/benchmarking/configs/deepseek-v4-flash-dp8-ep.sh`)
nor the scorer's target sets one, so `chunk_tokens` is what the arm's
`envs.py` (golden: unset, so one unchunked pass) or its `env_override.py`
makes of no variable. Any keyword the op does not pass takes the kernel's
own default (golden: `buffer_count` (4, 3, 3), a 100 MiB scoped-VMEM
limit, the `HEAD_ALONG_SUBLANE` cache layout). So an arm that retunes the
block sizes, the decode batch, the buffer counts or the chunking, in the
op, in `envs.py` or in the kernel, is compiled as its server would call it
(golden's op: `num_kv_pages_per_block` (32, 32, 16),
`num_queries_per_block` (1, 128, 128), `decode_req_batch_size` 4).

Shapes
------
The serving config is TP1 x DP8 with EP, `--max-num-batched-tokens 1024`,
`--max-num-seqs 256`, `--max-model-len 9216`, and the layer-run scorer
pins 12628 KV blocks. Attention, and the indexer with it, is data
parallel, so each engine calls the kernel on its own batch.

* Token buckets. An engine pads a step to one of 16, 32, ..., 512, 1024
  tokens (`_get_token_paddings` in `platforms/tpu_platform.py`, doubling
  from 16 to the token budget), and DP lockstep pads every engine to the
  same bucket. Decode and prefill steps run the same program at a bucket;
  the request distribution decides which pass does the work.
* Static model and cache geometry. 64 indexer heads of 128: `q`
  `f8e4m3fn[T, 64, 128]` and the weights `f32[T, 64]`, which need no
  padding. The KV block is 1024 tokens, which CSA compresses 4x into 256
  records a page, each 128 fp8 bytes and one scale byte padded to a
  256-byte row, four rows to a packed sublane group: `cache_kv` is
  `uint8[12628, 64, 4, 256]` (`runner/kv_cache_dsv4.py`, `(N, T/4, 4,
  256)`, read by the op as the kernel's default `HEAD_ALONG_SUBLANE`
  layout). The page count changes only the argument bytes, not the
  kernels' tiling. `seq_lens` `int32[256]`, `cu_q_lens` `int32[257]`,
  `distribution` `int32[3]`, and the page table `int32[2304]`, 256 x
  `cdiv(9216, 1024)` pages, flattened by the op. Each token keeps the top
  512 of at most 9216 / 4 = 2304 compressed positions; the score rows are
  8192 wide (32 pages of 256, the largest kv block rounded up).

Golden's layer-run server profiles show the same arrays and kernel names:
`StreamIdxTC-d-bq_1-bkvp_32` and `StreamIdxTC-m-bq_128-bkvp_16` on
`f8e4m3fn[1024,64,128]`, `f32[1024,64]`, `u8[12628,64,4,256]` and an
`s32[2304]` page table, writing `s32[1024,64,128]` scores at the 1024
bucket (`s32[128,64,128]` at 128), then `sc_topk_direct_b1024_n8192_k512_p1`.

Runtime values, none of which changes the compiled program:

* `distribution` = `[d, d, n]`: d decode requests, 0 to min(256, bucket),
  and n requests in all, 1 to 256. Each pass builds its own step schedule
  from it (`metadata.compute_metadata`), and the length of that schedule
  is its pipeline's grid: the batched decode pass `d // 4` tiles, the
  leftover pass `d % 4` sequences (0 to 3), the mixed pass one step per
  128-token query block of each of the other n - d sequences, each times
  the kv blocks a tile covers.
* `seq_lens`, 0 to 9216 tokens, so 0 to 2304 compressed records: the kv
  blocks a tile covers, `cdiv(len, bkv)` with bkv 8192 (decode) or 4096
  (mixed) records, so 1 for every length the server allows; and which
  rows of the scores are read.
* `cu_q_lens`, each request's query length, 1 to the bucket: the mixed
  pass's query blocks and the rows each query DMA copies.
* The page table's contents: only which pages are read.

The compile covers every distribution and length at a bucket, since each
only sizes a grid or a loop at run time. What it cannot cover is what
those values make the kernels read: a DMA past the end of an array or a
semaphore left waiting shows only on a device, with real data.

That is 7 shapes, one per bucket, which covers every program the server
compiles for this kernel.

Checks
------
`compile` builds each shape through the arm's own entry point with the
op's static arguments and compiles it for tpu7x. `limit` refuses a program
that launches no Mosaic kernel, since such a compile checked nothing.

`trip_counts` traces each shape again and reads every `emit_pipeline` made
while tracing: its grid, and for each input the buffer count and lookahead
it runs with. Those come from the `BufferedRef` allocations when the
pipeline is called with them, as the TensorCore kernels are, since the
allocations, not the specs, size the buffers; otherwise from the input
specs (2 buffers and no lookahead for a spec without a pipeline mode).

* A grid fixed at trace time gives one record per input pipe at its trip
  count. Golden's SparseCore top-k pipeline is of this kind: its grid is
  `cdiv(T, 32 // p)` waves, with rows split p = 2 ways at 16 tokens and
  not at all from 32 up (1 wave at 16 and 32 tokens, 32 at 1024), and 1
  buffer at one wave and 2 from two up.
* A grid sized at run time, or split across cores, with 3 or fewer
  buffers gives no record: at most 3 buffers start no copy that no step
  waits for at any trip count.
* Such a grid with 4 or more buffers and lookahead gives one record per
  trip count from 2 to B - 2, the counts where a pipe without lookahead
  would start a copy that nothing waits for, marked `lookahead`. The
  check passes them: with lookahead the prologue fetches through
  `fetch_with_lookahead`, which stops at the end of the grid (jax
  0.11.2, `jax/_src/pallas/mosaic/pipeline.py`). Golden's decode passes
  are of this kind: their grid is the step schedule's length, 4 buffers
  and lookahead on all three inputs (q, weights, kv), so each shape gives
  6 records at N = 2, all passing on the lookahead rule. That rule is the
  only thing that passes them; nothing here bounds the trip counts.
* Such a grid with 4 or more buffers and no lookahead makes `trip_counts`
  raise rather than guess: its trip count is known only at run time, and
  2 <= N <= B - 2 is in reach of any small batch.

Not modelled: copies a kernel starts outside `emit_pipeline` (a manual
DMA ring), and a `BufferedRef` subclass that changes the schedule itself
rather than what one copy moves. Golden's subclasses override only
`copy_in` / `wait_in` / `copy_out` / `wait_out`, each one copy of the
window jax's scheduler asks for.

Calibration
-----------
With this adapter on the CPU (jax 0.11.2, libtpu 0.0.50), against golden
8df58749d and copies of it, each broken in a way a fixed-shape benchmark
at the common buckets can miss:

  * golden passes both checks: 7 shapes compile, each to the two decode
    kernels, the mixed kernel and the SparseCore top-k above (p2 at 16
    tokens, p1 above; the largest temp allocation 39745280 bytes, at
    t1024), and 8 pipe records a shape: the 6 decode-pass records at
    B=4, N=2 with lookahead, and the 2 inputs of the top-k at 1 or 2
    buffers.
  * The op growing the mixed kernel's query block with the bucket
    (`max(128, q.shape[0])`) fails `aot_all_shapes` at t1024 only (VMEM
    out of memory, 110.54M of 63.94M; its 512-token block at 512 still
    fits), and passes `pipeline_trip_count`.
  * The entry point refusing a token count that is not a multiple of 32,
    true at every bucket but 16, fails both checks at t16 only.
  * Each TensorCore kernel staging the first indexer-weight rows (up to
    64) through a 4-buffer `emit_pipeline` of 16-row blocks compiles at
    all 7 shapes and fails `pipeline_trip_count` at t32 only, once per
    kernel (B=4, N=2); its grid is 1 step at 16 tokens and 4 from 64 up.
  * The kernel turning lookahead off on its pipes (decode keeps 4
    buffers) compiles at all 7 shapes and fails `pipeline_trip_count` at
    all 7: `trip_counts` raises on the decode pass's runtime-sized grid.

A run takes about 2.5 minutes (7 compiles of about 17 seconds, then 7
traces).
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
import re
import sys
import types
from pathlib import Path

# The package root: the tree a verdict names must cover the op file the
# static arguments are read from, `envs.py`, and the kernel.
KERNEL_DIR = "src/vllm_torchtpu"
KERNEL_PKG = "kernels/deepseek_v4/streamindex_topk"
KERNEL_FILE = f"{KERNEL_DIR}/{KERNEL_PKG}/streamindex_topk.py"
OP_FILE = f"{KERNEL_DIR}/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_indexer.py"
OP_FUNCTION = "_indexer_jax"
ENTRY = "streamindex_topk"
TOPOLOGY = "tpu7x:2x2x1"
# The serve config and the scorer's target set none of these.
UNSET_ENV_PREFIX = "TPU_STREAMIDX_"

NUM_HEADS = 64
HEAD_DIM = 128
TOPK = 512
COMPRESS_RATIO = 4
KV_BLOCK_TOKENS = 1024
PAGE_RECORDS = KV_BLOCK_TOKENS // COMPRESS_RATIO
KV_PACKING = 4
# 128 fp8 bytes and a scale byte per record, padded to the lane width.
RECORD_BYTES = 256
NUM_PAGES = 12628
MAX_NUM_SEQS = 256
MAX_MODEL_LEN = 9216
PAGES_PER_SEQ = -(-MAX_MODEL_LEN // KV_BLOCK_TOKENS)
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
SHAPES = {f"t{tokens}": {"tokens": tokens} for tokens in TOKEN_PADDINGS}

# The entry point's array parameters; every other keyword of the op's call
# is static.
ARRAY_ARGS = (
    "q",
    "indexer_weights",
    "cache_kv",
    "seq_lens",
    "page_indices",
    "cu_q_lens",
    "distribution",
)
# `_indexer_jax`'s static parameters, as `VllmDeepseekV4Indexer` binds them.
OP_PARAMS = {
    "k": TOPK,
    "compression_ratio": COMPRESS_RATIO,
    "softmax_scale": HEAD_DIM**-0.5,
    "n_head": NUM_HEADS,
}
# What a static keyword expression in the op may call.
RULE_BUILTINS = {f.__name__: f for f in (min, max, len, int, bool, tuple)}

# A pipe with at most this many buffers starts no copy that no step waits
# for at any trip count (the window 2 <= N <= B - 2 is empty).
SAFE_BUFFERS = 3
KERNEL_NAMES = (
    r"StreamIdxTC-[a-z]-bq_\d+-bkvp_\d+(?:-sal)?(?:-c\d+)?",
    r"sc_topk_direct_b\d+_n\d+_k\d+_p\d+",
)


def _load_kernel(root: Path):
    """Import the kernel from the arm root without the package's `__init__`.

    That `__init__` imports torch and the serving stack, which a compile-only
    environment does not have. `env_override` is loaded because it sets the
    server's LIBTPU_INIT_ARGS and applies its import-time Pallas patches, so
    the program compiled here is the one the server compiles. Returns the
    kernel module and the arm's `envs`.
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
    parts = ["vllm_torchtpu"]
    for part in KERNEL_PKG.split("/"):
        parts.append(part)
        package(".".join(parts), pkg.joinpath(*parts[1:]))
    kernel = module(".".join([*parts, ENTRY]), root / KERNEL_FILE)
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, "__file__", None)
        if (
            name.startswith("vllm_torchtpu.")
            and path
            and not Path(path).resolve().is_relative_to(root)
        ):
            raise RuntimeError(f"{name} was imported from {path}, outside {root}")
    return kernel, top.envs


def _only(nodes, what):
    if len(nodes) != 1:
        raise RuntimeError(f"{OP_FILE} has {len(nodes)} {what}, expected one")
    return nodes[0]


def _assigns_only(stmt, names) -> bool:
    """Whether `stmt` only assigns `names`, directly or under an `if`."""
    if isinstance(stmt, ast.Assign):
        return all(isinstance(t, ast.Name) and t.id in names for t in stmt.targets)
    if isinstance(stmt, ast.If):
        return all(_assigns_only(s, names) for s in stmt.body + stmt.orelse)
    return False


def _enclosing_functions(fn, call):
    """`fn` and every function nested in it that contains `call`, outermost first."""
    chain = [fn]
    while True:
        inner = [
            n
            for n in chain[-1].body
            if isinstance(n, ast.FunctionDef) and call in ast.walk(n)
        ]
        if not inner:
            return chain
        chain.append(inner[0])


def _static_rule(op_file: Path, envs):
    """tokens -> the static keyword arguments the op passes the entry point.

    Reads the `streamindex_topk` call in `_indexer_jax`. Each static keyword
    is evaluated with `_indexer_jax`'s own parameters bound as a CSA layer's
    indexer binds them and `envs` the arm's own module, after running only
    the statements of `_indexer_jax`, and of the function around the call,
    that assign plain names, with `q` standing for a query of the given
    token count.
    """
    tree = ast.parse(op_file.read_text(), str(op_file))
    fn = _only(
        [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == OP_FUNCTION
        ],
        f"function {OP_FUNCTION}",
    )
    call = _only(
        [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == ENTRY
        ],
        f"calls of {ENTRY} in {OP_FUNCTION}",
    )
    if call.args or any(kw.arg is None for kw in call.keywords):
        raise RuntimeError(f"{OP_FUNCTION} calls {ENTRY} with positional or ** args")
    static = [kw for kw in call.keywords if kw.arg not in ARRAY_ARGS]
    names = {
        n.id for kw in static for n in ast.walk(kw.value) if isinstance(n, ast.Name)
    }
    body = [
        s
        for f in _enclosing_functions(fn, call)
        for s in f.body
        if _assigns_only(s, names)
    ]
    # `compile` is this adapter's own hook.
    code = builtins.compile(
        ast.Module(body=body, type_ignores=[]), str(op_file), "exec"
    )
    exprs = {
        kw.arg: builtins.compile(ast.Expression(kw.value), str(op_file), "eval")
        for kw in static
    }

    def rule(tokens):
        env = {
            "q": types.SimpleNamespace(shape=(tokens, NUM_HEADS, HEAD_DIM)),
            "envs": envs,
            **OP_PARAMS,
        }
        scope = {"__builtins__": RULE_BUILTINS}
        exec(code, scope, env)
        return {k: eval(e, scope, env) for k, e in exprs.items()}

    return rule


def setup(root):
    # CPU only: jax must never open the host's TPU, and the compile-only
    # topology does not need the libtpu lock.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    for name in [n for n in os.environ if n.startswith(UNSET_ENV_PREFIX)]:
        del os.environ[name]
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    handle = types.SimpleNamespace(pipelines=None, device=None)
    # Installed before the kernel is imported, so a kernel that binds
    # `emit_pipeline` by name at import time gets the recorder too.
    _install_pipeline_recorder(handle)
    handle.kernel, envs = _load_kernel(root)
    handle.statics = _static_rule(root / OP_FILE, envs)

    import jax
    import jax.numpy as jnp

    handle.jax, handle.jnp = jax, jnp
    return handle


def origin(handle) -> str:
    return handle.kernel.__file__


def _entry(handle, params):
    """The entry point called as `_indexer_jax` calls it, at one shape."""
    statics = handle.statics(params["tokens"])
    fn = getattr(handle.kernel, ENTRY)

    def call(*arrays):
        return fn(**dict(zip(ARRAY_ARGS, arrays)), **statics)

    return call


def _arg_specs(handle, params):
    jnp, t = handle.jnp, params["tokens"]
    return [
        ((t, NUM_HEADS, HEAD_DIM), jnp.float8_e4m3fn),
        ((t, NUM_HEADS), jnp.float32),
        ((NUM_PAGES, PAGE_RECORDS // KV_PACKING, KV_PACKING, RECORD_BYTES), jnp.uint8),
        ((MAX_NUM_SEQS,), jnp.int32),
        ((MAX_NUM_SEQS * PAGES_PER_SEQ,), jnp.int32),
        ((MAX_NUM_SEQS + 1,), jnp.int32),
        ((3,), jnp.int32),
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
    # Nothing is donated: the op donates none of these operands.
    fn = jax.jit(_entry(handle, params))
    with _tpu7x_mesh():
        return fn.trace(*args)


def compile(handle, name, params):
    lowered = _trace(handle, params).lower(lowering_platforms=("tpu",))
    mem = lowered.compile().memory_analysis()
    text = lowered.as_text()
    statics = handle.statics(params["tokens"])
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "mosaic_kernels": text.count("tpu_custom_call"),
        "kernels": ",".join(
            sorted({m for p in KERNEL_NAMES for m in re.findall(p, text)})
        ),
        **{k: v for k, v in statics.items() if k not in OP_PARAMS},
    }


def limit(name, result):
    if not result["mosaic_kernels"]:
        return (
            "the compiled program launches no Mosaic kernel, so the compile "
            "checked nothing"
        )
    return None


def _spec_pipes(specs, label="in"):
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
            found += _spec_pipes(s, f"{label}[{i}]")
    elif isinstance(specs, dict):
        for k, s in specs.items():
            found += _spec_pipes(s, f"{label}[{k!r}]")
    elif dataclasses.is_dataclass(specs) and not isinstance(specs, type):
        for f in dataclasses.fields(specs):
            found += _spec_pipes(getattr(specs, f.name), f"{label}.{f.name}")
    return found


def _allocation_pipes(allocations):
    """`(label, buffer_count, lookahead)` for every buffered input allocation."""
    import jax
    from jax._src.pallas.mosaic import pipeline as jax_pipeline

    brefs = jax.tree.leaves(
        allocations, is_leaf=lambda b: isinstance(b, jax_pipeline.BufferedRefBase)
    )
    found = []
    for i, b in enumerate(brefs):
        if not isinstance(b, jax_pipeline.BufferedRef) or not b.is_input:
            continue
        if b.is_buffered:
            label = f"in[{i}] {type(b).__name__}"
            found.append((label, int(b.buffer_count), bool(b.use_lookahead)))
    return found


def _qualname(body):
    """The body's own name, through any `functools.partial` around it."""
    while isinstance(body, functools.partial):
        body = body.func
    return getattr(body, "__qualname__", type(body).__name__)


def _install_pipeline_recorder(handle):
    """Wrap `emit_pipeline` to note each call while `handle.pipelines` is a list.

    The pipes are read from the input specs when the pipeline is built, and
    again from the allocations if it is run with them.
    """
    from jax._src.pallas.mosaic import pipeline as jax_pipeline
    from jax.experimental.pallas import tpu as pltpu

    real = jax_pipeline.emit_pipeline

    @functools.wraps(real)
    def emit_pipeline(body, *, grid, in_specs=(), **kwargs):
        run = real(body, grid=grid, in_specs=in_specs, **kwargs)
        if handle.pipelines is None:
            return run
        split = (
            kwargs.get("core_axis") is not None
            or kwargs.get("core_axis_name") is not None
        )
        record = [_qualname(body), tuple(grid), split, _spec_pipes(in_specs)]
        handle.pipelines.append(record)

        @functools.wraps(run)
        def recorded(*args, allocations=None, **kw):
            if allocations is not None:
                record[3] = _allocation_pipes(allocations)
            return run(*args, allocations=allocations, **kw)

        return recorded

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
            label = f"pipeline{i} (body {body}) {pipe}"
            if static:
                records.append(
                    {
                        "pipe": label,
                        "buffer_count": b,
                        "trip_count": math.prod(dims),
                        "lookahead": lookahead,
                        "input": (
                            f"any runtime input (grid {tuple(dims)} fixed at "
                            "trace time)"
                        ),
                    }
                )
                continue
            if b <= SAFE_BUFFERS:
                continue
            why = "split across cores" if split else "sized at run time"
            if not lookahead:
                raise RuntimeError(
                    f"pipeline {i} ({body}) {pipe} has {b} buffers, no lookahead "
                    f"and a grid {why}; this adapter reads trip counts at trace "
                    "time only"
                )
            records += [
                {
                    "pipe": label,
                    "buffer_count": b,
                    "trip_count": n,
                    "lookahead": True,
                    "input": f"a batch that gives the grid ({why}) {n} steps",
                }
                for n in range(2, b - 1)
            ]
    return records
