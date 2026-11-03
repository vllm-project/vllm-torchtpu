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
r"""Validity adapter for DeepSeek-V4 HCA compressed-KV attention at Flash DP8/EP8.

Run it through perfkit, never directly:

    perfkit validity run --adapter <this file> --arm <checkout> \
        --python "$(perfkit env ensure --jax 0.11.2 --libtpu 0.0.50)"

`perfkit/validity_arm.py` defines the hooks and the two checks. Nothing here
uses a TPU: compiles are ahead of time for the tpu7x topology on the CPU, and
trip counts are read while tracing on the CPU.

What is compiled
----------------
The entry point is `mla_ragged_paged_attention` in
`kernels/deepseek_v4/core_attention/mla.py`, the dense compressed-KV pass
that `_attention_hca` in
`layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py` runs
for every HCA (128x compressed) layer, after the sliding-window pass whose
unnormalized output and `swa_l` / `swa_m` it folds into its softmax. One
call issues one `pallas_call` per case, each over a slice of the request
distribution: the batched decode kernel (`MLA-bd-bq_1-bkvp_128-p_8-bsz_8`)
on whole batches of `decode_batch_size` decode sequences, the decode kernel
(`MLA-d-bq_1-bkvp_128-p_8-bsz_1`) on the decode sequences left over, and
the mixed kernel (`MLA-m-bq_32-bkvp_64-p_8-bsz_1`) on the rest, which
carries every prefill token. The prefill kernel is compiled only when the
op passes a `chunk_prefill_size`; golden's op passes none.

`compile` reads the static arguments from the arm's own op file: every
keyword of the entry-point call in `_attention_hca` that is not an array,
each evaluated with `_attention_hca`'s parameters bound as the HCA layer
binds them (`sm_scale` 512**-0.5, `sliding_window` 128, `swa_only` False,
`two_caches_same_buffer` False, as an HCA layer's array is its own,
`non_causal_block` False) after the plain assignments in `_attention_hca`.
`logical_page_size` is the arm's own
`VllmDeepseekV4SWACache._swa_block_size` for the 1024-token KV block and
the 128-token window (golden: 128). Any keyword the op does not pass takes
the kernel's own default (golden: `decode_batch_size` 8, a 100 MiB
scoped-VMEM limit). So an arm that retunes the block sizes, the decode
batch or the prefill chunk, in the op or in the kernel, is compiled as its
server would call it.

Shapes
------
The serving config is `scripts/vllm/benchmarking/configs/
deepseek-v4-flash-dp8-ep.sh`: TP1 x DP8 with EP, `--max-num-batched-tokens
1024`, `--max-num-seqs 256`, `--max-model-len 9216`. Attention is data
parallel, so each engine calls the kernel on its own batch.

* Token buckets. An engine pads a step to one of 16, 32, ..., 512, 1024
  tokens (`_get_token_paddings` in `platforms/tpu_platform.py`, doubling from
  16 to the token budget), and DP lockstep pads every engine to the same
  bucket. Decode and prefill steps run the same program at a bucket; the
  request distribution decides which kernel does the work.
* Static model and cache geometry. 64 query heads of 512: `q`, and the SWA
  partials `swa_accumution` `bf16[T, 64, 512]`, `swa_l` / `swa_m`
  `f32[T, 64]`, sinks `f32[64]` (`prepare_swa_inputs` pads the l/m heads
  to 128 inside the call; 64 heads of 512 need no other padding). The KV block is
  1024 tokens, which HCA compresses 128x into 8 records a page, each a raw
  bf16 latent of 1024 bytes in two rows: `cache_kv` is
  `uint8[12628, 16, 4, 128]` (`runner/kv_cache_dsv4.py`, `(N, T * 2,
  4, 128)`). 12628 pages, the block count the layer-run scorer fixes; the
  page count changes only the argument bytes, not the kernels' tiling.
  `kv_lens` `int32[256]`, `kv_lens_to_attend` `int32[T]`, `cu_q_lens`
  `int32[257]`, `distribution` `int32[3]`, and the page table `int32[2304]`,
  256 x `cdiv(9216, 1024)` pages.

Golden's layer-run server profiles show the same arrays and kernel names:
the three kernels above on `u8[12628,16,4,128]`, with `s32[2304]` page
tables and `s32[128]` `kv_lens_to_attend` at the 128 bucket.

Runtime values, none of which changes the compiled program:

* `distribution` = `[d, d, n]` (`_prepare_inputs` in `runner/tpu_runner.py`):
  d decode requests, 0 to min(256, bucket), and n requests, 1 to 256. With
  a decode batch of b (golden 8), the batched decode kernel's grid is
  `d // b` batches (0 to 32), the decode kernel's `d % b` sequences (0 to
  7), and the mixed kernel's `n - d` sequences (0 to 256). A profiling step
  sends `[0, 0, n]`.
* `kv_lens`, the compressed length of each sequence (`seq_lens // 128`, 0 to
  72): the number of kv blocks a grid step loops over, `max(1, cdiv(len,
  bkv))` with bkv 1024 (decode) or 512 (mixed) records, so 1 for every
  length the server allows; and the bytes each page's DMA copies.
* `cu_q_lens`, each request's query length, 1 to the bucket: the number of
  32-token query blocks the mixed kernel loops over for a sequence, and the
  rows each query DMA copies.
* `kv_lens_to_attend`, `(position + 1) // 128` per token, 0 to 72: it masks
  scores, not the grid.
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

`trip_counts` traces each shape again and reads every `emit_pipeline` call
made while tracing: its grid, and the buffer count and lookahead of each
input spec (2 buffers and no lookahead for a spec without a pipeline mode).
Golden has none: each kernel moves its blocks with its own DMAs into
two-slot scratch, over a `pallas_call` grid sized at run time, so
`pipeline_trip_count` passes with no records. A two-slot manual pipeline
has no unsafe trip count, and neither the manual DMAs nor a `pallas_call`
input spec with a deeper pipeline mode are modelled here. The check guards
a candidate that brings in an `emit_pipeline`: one whose grid is fixed at
trace time gives one record per input pipe, and one of 4 or more buffers
whose grid is sized at run time, or split across cores, makes
`trip_counts` raise rather than guess (3 or fewer give no record).

Calibration
-----------
With this adapter on the CPU (jax 0.11.2, libtpu 0.0.50), against golden
218755777 and three copies of it, each broken in a way a fixed-shape
benchmark at the common buckets can miss:

  * golden passes both checks: 7 shapes compile, each to the three kernels
    above (the largest temp allocation 1405792 bytes, at t1024), and no
    pipe records.
  * The op growing the mixed kernel's query block with the bucket (`max(32,
    T // 8)`) fails `aot_all_shapes` at t1024 only (VMEM out of memory,
    105.93M of 63.94M; its 64-token block at 512 still fits), and passes
    `pipeline_trip_count`.
  * The entry point asserting whole 32-token query blocks, true at every
    bucket but 16, fails both checks at t16 only.
  * Each kernel staging the first query rows (up to 64) through a 4-buffer
    `emit_pipeline` of 16-row blocks compiles at all 7 shapes and fails
    `pipeline_trip_count` at t32 only, once per kernel (B=4, N=2); its grid
    is 1 step at 16 tokens and 4 from 64 up.

The op passing `chunk_prefill_size=32` adds the prefill kernel
(`MLA-p-bq_32-bkvp_64-p_8-bsz_1`) at every bucket, and passes.

A run takes about 3 minutes (7 compiles of about 20 seconds, then 7
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
# static arguments are read from as well as the kernel.
KERNEL_DIR = "src/vllm_torchtpu"
KERNEL_FILE = f"{KERNEL_DIR}/kernels/deepseek_v4/core_attention/mla.py"
OP_FILE = (
    f"{KERNEL_DIR}/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py"
)
OP_FUNCTION = "_attention_hca"
ENTRY = "mla_ragged_paged_attention"
SWA_CACHE_CLASS = "VllmDeepseekV4SWACache"
SWA_BLOCK_SIZE = "_swa_block_size"
TOPOLOGY = "tpu7x:2x2x1"

NUM_HEADS = 64
HEAD_DIM = 512
SLIDING_WINDOW = 128
KV_BLOCK_TOKENS = 1024
HCA_RATIO = 128
# An HCA page holds 1024 / 128 records of one bf16 latent each, two rows of
# 4 x 128 bytes per record.
PAGE_RECORDS = KV_BLOCK_TOKENS // HCA_RATIO
CACHE_ROWS = 2 * PAGE_RECORDS
KV_PACKING = 4
ROW_BYTES = 128
NUM_PAGES = 12628
MAX_NUM_SEQS = 256
MAX_MODEL_LEN = 9216
PAGES_PER_SEQ = -(-MAX_MODEL_LEN // KV_BLOCK_TOKENS)
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
SHAPES = {f"t{tokens}": {"tokens": tokens} for tokens in TOKEN_PADDINGS}

# The entry point's array parameters, in order; every other keyword of the
# op's call is static.
ARRAY_ARGS = (
    "q",
    "cache_kv",
    "kv_lens",
    "kv_lens_to_attend",
    "page_indices",
    "cu_q_lens",
    "distribution",
    "attention_sinks",
    "swa_accumution",
    "swa_l",
    "swa_m",
)
SM_SCALE = HEAD_DIM**-0.5
# What a static keyword expression in the op may call.
RULE_BUILTINS = {f.__name__: f for f in (min, max, len, int, bool, tuple)}

# A pipe with at most this many buffers starts no copy that no step waits
# for at any trip count (the window 2 <= N <= B - 2 is empty).
SAFE_BUFFERS = 3


def _load_kernel(root: Path):
    """Import mla from the arm root without the package's own `__init__`.

    That `__init__` imports torch and the serving stack, which a compile-only
    environment does not have. `env_override` is loaded because it sets the
    server's LIBTPU_INIT_ARGS and applies its import-time Pallas patches, so
    the program compiled here is the one the server compiles.
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
        "vllm_torchtpu.kernels.deepseek_v4.core_attention.mla",
        root / KERNEL_FILE,
    )
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, "__file__", None)
        if (
            name.startswith("vllm_torchtpu.")
            and path
            and not Path(path).resolve().is_relative_to(root)
        ):
            raise RuntimeError(f"{name} was imported from {path}, outside {root}")
    return kernel


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


def _swa_page_size(tree, op_file: Path) -> int:
    """The SWA logical page size the arm's op derives for the served cache.

    Runs `VllmDeepseekV4SWACache._swa_block_size` from the arm's op file on
    the 1024-token KV block and the 128-token window, as the attention
    layer does when it builds the op.
    """
    cls = _only(
        [
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == SWA_CACHE_CLASS
        ],
        f"class {SWA_CACHE_CLASS}",
    )
    fn = _only(
        [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == SWA_BLOCK_SIZE
        ],
        f"{SWA_CACHE_CLASS}.{SWA_BLOCK_SIZE}",
    )
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    env = {}
    code = builtins.compile(
        ast.Module(body=[fn], type_ignores=[]), str(op_file), "exec"
    )
    exec(code, {"__builtins__": RULE_BUILTINS}, env)
    return int(env[SWA_BLOCK_SIZE](KV_BLOCK_TOKENS, SLIDING_WINDOW))


def _static_rule(op_file: Path):
    """tokens -> the static keyword arguments the op passes the entry point.

    Reads the `mla_ragged_paged_attention` call in `_attention_hca`. Each
    static keyword is evaluated with `_attention_hca`'s own parameters bound
    as an HCA layer binds them, after running only the statements of the
    function that assign plain names, with `q` standing for the query of
    the given token count.
    """
    tree = ast.parse(op_file.read_text(), str(op_file))
    page_size = _swa_page_size(tree, op_file)
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
    body = [s for s in fn.body if _assigns_only(s, names)]
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
            "sm_scale": SM_SCALE,
            "sliding_window": SLIDING_WINDOW,
            "logical_page_size": page_size,
            "swa_only": False,
            "two_caches_same_buffer": False,
            "non_causal_block": False,
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
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    handle = types.SimpleNamespace(pipelines=None, device=None)
    # Installed before the kernel is imported, so a kernel that binds
    # `emit_pipeline` by name at import time gets the recorder too.
    _install_pipeline_recorder(handle)
    handle.kernel = _load_kernel(root)
    handle.statics = _static_rule(root / OP_FILE)

    import jax
    import jax.numpy as jnp

    handle.jax, handle.jnp = jax, jnp
    return handle


def origin(handle) -> str:
    return handle.kernel.__file__


def _entry(handle, params):
    """The entry point called as `_attention_hca` calls it, at one shape."""
    statics = handle.statics(params["tokens"])
    fn = getattr(handle.kernel, ENTRY)

    def call(*arrays):
        return fn(**dict(zip(ARRAY_ARGS, arrays)), **statics)

    return call


def _arg_specs(handle, params):
    jnp, t = handle.jnp, params["tokens"]
    return [
        ((t, NUM_HEADS, HEAD_DIM), jnp.bfloat16),
        ((NUM_PAGES, CACHE_ROWS, KV_PACKING, ROW_BYTES), jnp.uint8),
        ((MAX_NUM_SEQS,), jnp.int32),
        ((t,), jnp.int32),
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
    # Nothing is donated: the op donates only the SWA cache, which this
    # kernel does not take.
    fn = jax.jit(_entry(handle, params))
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
        "kernels": ",".join(
            sorted(set(re.findall(r"MLA-[a-z]+-bq_\d+-bkvp_\d+-p_\d+-bsz_\d+", text)))
        ),
        **handle.statics(params["tokens"]),
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
