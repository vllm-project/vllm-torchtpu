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
r"""Validity adapter for DeepSeek-V4 sliding-window attention at Flash DP8/EP8.

Run it through perfkit, never directly:

    perfkit validity run --adapter <this file> --arm <checkout> \
        --python "$(perfkit env ensure --jax 0.11.2 --libtpu 0.0.50)"

`perfkit/validity_arm.py` defines the hooks and the two checks. Nothing here
uses a TPU: compiles are ahead of time for the tpu7x topology on the CPU, and
trip counts are read while tracing on the CPU.

What is compiled
----------------
The entry point is `mla_sliding_window_ragged_paged_attention` in
`kernels/deepseek_v4/core_attention/mla_swa.py`, which `_run_swa` in
`layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py` calls
for every layer: SWA-only layers through `_attention_hca` with
`swa_only=True`, CSA and HCA layers through `_attention_csa` and
`_attention_hca` with `swa_only=False`, ahead of their compressed-KV pass.
One call is the whole block: it narrows the page table to each sequence's
window (`narrow_page_indices`), then runs the decode kernel on the decode
sequences and the mixed kernel (`SWA-m-bq_32-bkvp_2`) on the rest, which
carries every prefill token. The decode kernel is the batched one,
`run_mla_batched_decode_kernel` (`SWA-d-bq_1-bkvp_2`), because the window
(128) fits in one decode kv block (2 pages of 128); the per-sequence
decode variant of the mixed kernel is not compiled. No dedicated prefill
kernel is compiled either: the op passes no `chunk_prefill_size`.

`compile` reads the static arguments from the arm's own op file: every
keyword of the entry-point call in `_run_swa` that is not an array, each
evaluated with `_run_swa`'s parameters bound as the layer binds them
(`sm_scale` 512**-0.5, `sliding_window` 128, `non_causal_block` False,
`swa_only` per kind) after the plain assignments in `_run_swa`. The
logical page size is the arm's own `VllmDeepseekV4SWACache._swa_block_size`
for the 1024-token KV block and the 128-token window (golden: 128). So an
arm that retunes the block sizes, `q_compute_block_size` or the decode
batch in the op is compiled as its server would call it.

Shapes
------
The serving config is `scripts/vllm/benchmarking/configs/
deepseek-v4-flash-dp8-ep.sh`: TP1 x DP8 with EP, `--max-num-batched-tokens
1024`, `--max-num-seqs 256`, `--max-model-len 9216`, `fp8_e4m3` KV cache.
Attention is data parallel, so each engine calls the kernel on its own batch.

* Token buckets. An engine pads a step to one of 16, 32, ..., 512, 1024
  tokens (`_get_token_paddings` in `platforms/tpu_platform.py`, doubling from
  16 to the token budget), and DP lockstep pads every engine to the same
  bucket. Decode and prefill steps run the same program at a bucket; the
  request distribution decides which kernel does the work.
* Kinds. `unnormalized_output` is `not swa_only`, and it changes the
  program (both kernels' epilogues, and a zeroed vs. uninitialized output
  buffer), so each bucket is compiled twice: `swa` for SWA-only
  layers and `csa_hca` for CSA and HCA layers, whose calls are identical.
* Static model and cache geometry. 64 query heads of 512, `q`
  `bf16[T, 64, 512]`, `new_kv` `bf16[T, 512]`, sinks `f32[64]`. Every SWA
  cache overlays a CSA NoPE array (`runner/kv_cache_dsv4.py`), so `cache_kv`
  is `int32[12628, 256, 128]` for all three layer kinds, 128 bf16 tokens a
  page; it is donated, as the entry point declares. 12628 pages, the block
  count the layer-run scorer fixes; the page count changes only the
  argument bytes, not the kernels' tiling. `kv_lens` `int32[256]`,
  `cu_q_lens` `int32[257]`, `distribution` `int32[3]`, and the page table
  `int32[18432]`, 256 x `cdiv(9216, 128)` pages, which `narrow_page_indices`
  cuts to 256 x 6 entries at 16 to 128 tokens, x 7 at 256, x 9 at 512 and
  x 13 at 1024.

Golden's layer-run server profiles show the same arrays and kernel names:
`SWA-d-bq_1-bkvp_2` and `SWA-m-bq_32-bkvp_2` on `s32[12628,256,128]` with
page tables of `s32[1536]` at the 128 bucket and `s32[3328]` at 1024, the
sizes this adapter's programs have.

Runtime values, none of which changes the compiled program:

* `distribution` = `[d, d, n]` (`_prepare_inputs` in `runner/tpu_runner.py`):
  d decode requests, 0 to min(256, bucket), and n requests, 1 to 256. The
  decode kernel's grid is `(1 if d > 0 else 0,)` and it loops over
  `cdiv(d, 32)` batches of 32 sequences (0 to 8). The mixed kernel's grid is
  `n - d` sequences, 0 to 256, and the empty `[d, d)` prefill range runs
  nothing. A profiling step sends `[0, 0, n]`.
* `cu_q_lens`, each request's query length, 1 to the bucket: the number of
  32-token query blocks of a sequence in the mixed kernel, looped at run
  time.
* `kv_lens`, 1 to 9216: where each sequence's window starts, so where the
  kv block a query block reads begins (one block, as `cdiv(32 + 127, 256)`
  is 1) and the page `narrow_page_indices` starts at.
* The page table's contents: only which pages are read and written.

That is 14 shapes, 7 buckets x 2 kinds, which covers every program the
server compiles for this kernel.

Checks
------
`compile` builds each shape through the arm's own entry point with the
op's static arguments and compiles it for tpu7x. `limit` refuses a program
that launches no Mosaic kernel, since such a compile checked nothing.

`trip_counts` traces each shape again and reads every `emit_pipeline` call
made while tracing: its grid, and the buffer count and lookahead of each
input spec (2 buffers and no lookahead for a spec without a pipeline mode).
Golden has none: both kernels move their blocks with their own DMAs into
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
2524c6f35 and three copies of it, each broken in a way a fixed-shape
benchmark at the common buckets can miss:

  * golden passes both checks: 14 shapes compile (the largest temp
    allocation 2962112 bytes, at swa-t1024), and no pipe records.
  * The mixed kernel's query block grown with the bucket (`max(32, T //
    4)`) fails `aot_all_shapes` at 1024 tokens only, both kinds (VMEM out
    of memory, 131.14M and 115.20M of 63.94M), and passes
    `pipeline_trip_count`.
  * The op asserting whole 32-token query blocks, true at every bucket but
    16, fails both checks at 16 tokens only, both kinds.
  * The decode kernel staging its queries through a 4-buffer
    `emit_pipeline` in blocks of up to 128 rows compiles at all 14 shapes
    and fails `pipeline_trip_count` at 256 tokens only, both kinds (B=4,
    N=2); its grid is 1 step up to 128 tokens and 4 or 8 above 256.

A run takes about 2 minutes (14 compiles of 7 to 8 seconds, then 14
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
KERNEL_FILE = f"{KERNEL_DIR}/kernels/deepseek_v4/core_attention/mla_swa.py"
OP_FILE = (
    f"{KERNEL_DIR}/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py"
)
OP_FUNCTION = "_run_swa"
ENTRY = "mla_sliding_window_ragged_paged_attention"
SWA_CACHE_CLASS = "VllmDeepseekV4SWACache"
SWA_BLOCK_SIZE = "_swa_block_size"
TOPOLOGY = "tpu7x:2x2x1"

NUM_HEADS = 64
HEAD_DIM = 512
SLIDING_WINDOW = 128
KV_BLOCK_TOKENS = 1024
# The SWA cache overlays a CSA NoPE array: int32[pages, 1024 / 4, 128].
CACHE_ROWS = KV_BLOCK_TOKENS // 4
ROW_WORDS = 128
NUM_PAGES = 12628
MAX_NUM_SEQS = 256
MAX_MODEL_LEN = 9216
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
# SWA-only layers normalize their output; CSA and HCA layers leave it
# unnormalized for the compressed-KV pass that follows.
KINDS = {"swa": True, "csa_hca": False}
SHAPES = {
    f"{kind}-t{tokens}": {"kind": kind, "tokens": tokens}
    for kind in KINDS
    for tokens in TOKEN_PADDINGS
}

# The entry point's array parameters, in order; every other keyword of the
# op's call is static.
ARRAY_ARGS = (
    "q",
    "new_kv",
    "cache_kv",
    "kv_lens",
    "page_indices",
    "cu_q_lens",
    "distribution",
    "attention_sinks",
)
DONATED = (ARRAY_ARGS.index("cache_kv"),)
SM_SCALE = HEAD_DIM**-0.5
# What a static keyword expression in the op may call.
RULE_BUILTINS = {f.__name__: f for f in (min, max, len, int, bool, tuple)}

# A pipe with at most this many buffers starts no copy that no step waits
# for at any trip count (the window 2 <= N <= B - 2 is empty).
SAFE_BUFFERS = 3


def _load_kernel(root: Path):
    """Import mla_swa from the arm root without the package's own `__init__`.

    That `__init__` imports torch and the serving stack, which a compile-only
    environment does not have. `env_override` is loaded because it sets the
    server's LIBTPU_INIT_ARGS and applies its import-time Pallas patches, so
    the program compiled here is the one the server compiles. The kernel's
    own import (`csa_cache_layout`) resolves through the stub packages,
    whose paths are the arm's.
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
        "vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa",
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
    """(tokens, swa_only) -> the static keyword arguments the op passes.

    Reads the `mla_sliding_window_ragged_paged_attention` call in `_run_swa`.
    Each static keyword is evaluated with `_run_swa`'s own parameters bound
    as the layer binds them, after running only the statements of the
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

    def rule(tokens, swa_only):
        env = {
            "q": types.SimpleNamespace(shape=(tokens, NUM_HEADS, HEAD_DIM)),
            "sm_scale": SM_SCALE,
            "sliding_window": SLIDING_WINDOW,
            "logical_page_size": page_size,
            "swa_only": swa_only,
            "non_causal_block": False,
        }
        scope = {"__builtins__": RULE_BUILTINS}
        exec(code, scope, env)
        return {k: eval(e, scope, env) for k, e in exprs.items()}

    return rule, page_size


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
    handle.statics, handle.page_size = _static_rule(root / OP_FILE)

    import jax
    import jax.numpy as jnp

    handle.jax, handle.jnp = jax, jnp
    return handle


def origin(handle) -> str:
    return handle.kernel.__file__


def _statics(handle, params):
    return handle.statics(params["tokens"], KINDS[params["kind"]])


def _entry(handle, params):
    """The entry point called as `_run_swa` calls it, at one shape."""
    statics = _statics(handle, params)
    fn = getattr(handle.kernel, ENTRY)

    def call(*arrays):
        return fn(**dict(zip(ARRAY_ARGS, arrays)), **statics)

    return call


def _arg_specs(handle, params):
    jnp, t = handle.jnp, params["tokens"]
    pages_per_seq = -(-MAX_MODEL_LEN // handle.page_size)
    return [
        ((t, NUM_HEADS, HEAD_DIM), jnp.bfloat16),
        ((t, HEAD_DIM), jnp.bfloat16),
        ((NUM_PAGES, CACHE_ROWS, ROW_WORDS), jnp.int32),
        ((MAX_NUM_SEQS,), jnp.int32),
        ((MAX_NUM_SEQS * pages_per_seq,), jnp.int32),
        ((MAX_NUM_SEQS + 1,), jnp.int32),
        ((3,), jnp.int32),
        ((NUM_HEADS,), jnp.float32),
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
    # The entry point donates the cache, which the kernels update in place.
    fn = jax.jit(_entry(handle, params), donate_argnums=DONATED)
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
            sorted(set(re.findall(r"SWA-[a-z]-bq_\d+-bkvp_\d+", text)))
        ),
        **_statics(handle, params),
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
