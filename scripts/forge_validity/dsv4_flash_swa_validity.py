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

r"""Validity adapter for DeepSeek-V4 sliding-window attention
(`mla_sliding_window_ragged_paged_attention`) at DeepSeek-V4-Flash-0731 DP8/EP8
serving shapes.

Run it through perfkit, never directly:

    perfkit validity run \
        --adapter scripts/forge_validity/dsv4_flash_swa_validity.py \
        --arm <checkout>

`perfkit/validity_arm.py` defines the hooks and the two checks (`aot_all_shapes`,
`pipeline_trip_count`). Nothing here uses a TPU: compiles are ahead of time for
the tpu7x topology on the CPU, and the run-time check is Pallas interpret mode on
the CPU.

What it checks
--------------
Every layer of the model runs the sliding-window pass, through `_run_swa` in
`layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py`. `SHAPES` is
every token count the server sends it under `deepseek-v4-flash-dp8-ep` (the runner
pads each engine's batch to one of 7 token counts, 16 to 1024,
`--max-num-batched-tokens=1024`), once with the normalized output the SWA-only
layers take and once with the unnormalized output the compressed layers fold into
their own softmax. Each is called with `_run_swa`'s block sizes, the model's 64
heads of 512, its 128-token window and 128-token logical pages, and the cache the
server allocates: the CSA host array, int32 [14579, 256, 128], 256 sequences of
up to 72 pages (`--max-model-len=9216`).

`aot_all_shapes` compiles the whole op -- page-table narrowing, the batched
decode kernel and the mixed-length kernel -- for tpu7x at each shape.

Both kernels issue their own DMAs and size their work at run time: the decode
kernel from the number of decode sequences, the mixed kernel's grid from the
number of mixed sequences and its inner loops from each one's query and key
lengths. Warmup's dummy batches, the decode tail and a chunk that starts partway
into a page reach inputs the layer-run score never does. `trip_counts` runs the
op in Pallas interpret mode over runtime inputs that reach those edges (decode
counts around the decode kernel's 32-sequence batch, prefill lengths around the
32-token query block and the 128-token page, key lengths below, at and across
the window and the page, the served shape of a few decodes beside one long
chunk), with each DMA replaced by one that moves nothing and logs the bytes of its
start and its wait per kernel launch and semaphore. An input fails when a launch
starts a copy it never waits for (Mosaic's exit check halts the core on it),
waits for more than it started (the core hangs), or raises. The kernels wait
for several copies on one semaphore with one descriptor of their total size, so
the bytes are compared, not the number of copies.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import os
import sys
import types
from pathlib import Path

KERNEL_DIR = "src/vllm_torchtpu/kernels/deepseek_v4/core_attention"
TOPOLOGY = "tpu7x:2x2x1"

HEADS = 64
HEAD_DIM = 512
SLIDING_WINDOW = 128
LOGICAL_PAGE_SIZE = 128
TOTAL_PAGES = 14579
CACHE_ROWS = 256  # int32 [pages, 256, 128] host: 128 bf16 tokens of 512 per page
ROW_WORDS = 128
MAX_SEQS = 256
MAX_MODEL_LEN = 9216
PAGES_PER_SEQ = MAX_MODEL_LEN // LOGICAL_PAGE_SIZE
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)
#: `_run_swa`'s block sizes, per (decode, prefill, mixed) pass.
NUM_KV_PAGES_PER_BLOCK = (2, 2, 2)
NUM_QUERIES_PER_BLOCK = (1, 32, 32)
Q_COMPUTE_BLOCK_SIZE = 4
DECODE_SEQ_BATCH = 32  # the op's default `decode_seq_batch_size`
#: The interpret runs use a small cache; its size sizes no grid.
INTERPRET_PAGES = 16

SHAPES = {
    f"swa_{kind}_t{t}": {"tokens": t, "unnormalized": kind == "csa"}
    for kind in ("only", "csa")
    for t in TOKEN_PADDINGS
}


def _load_kernel(root: Path):
    """Import mla_swa from the arm root without the package's own `__init__`,
    which imports torch and the serving stack. `env_override` is loaded for the
    server's LIBTPU_INIT_ARGS and import-time Pallas patches."""
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
    dsv4 = package("vllm_torchtpu.kernels.deepseek_v4", pkg / "kernels" / "deepseek_v4")
    dsv4.csa_cache_layout = module(
        "vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout",
        pkg / "kernels" / "deepseek_v4" / "csa_cache_layout.py",
    )
    package("vllm_torchtpu.kernels.deepseek_v4.core_attention", root / KERNEL_DIR)
    return module(
        "vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa",
        root / KERNEL_DIR / "mla_swa.py",
    )


def setup(root):
    # CPU only: jax must never open the host's TPU, and the compile-only
    # topology does not need the libtpu lock.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-4")
    sys.dont_write_bytecode = True
    kernel = _load_kernel(Path(root).resolve())

    import jax
    import jax.numpy as jnp
    import numpy as np

    return types.SimpleNamespace(kernel=kernel, jax=jax, jnp=jnp, np=np, device=None)


def origin(handle) -> str:
    return handle.kernel.__file__


def _entry(handle, params):
    """The op called as `_run_swa` calls it, at one shape."""
    return functools.partial(
        handle.kernel.mla_sliding_window_ragged_paged_attention,
        sm_scale=HEAD_DIM**-0.5,
        sliding_window=SLIDING_WINDOW,
        logical_page_size=LOGICAL_PAGE_SIZE,
        num_kv_pages_per_block=NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=NUM_QUERIES_PER_BLOCK,
        q_compute_block_size=Q_COMPUTE_BLOCK_SIZE,
        unnormalized_output=params["unnormalized"],
    )


def _arg_specs(handle, tokens, pages=TOTAL_PAGES):
    jnp = handle.jnp
    return [
        ((tokens, HEADS, HEAD_DIM), jnp.bfloat16),  # q
        ((tokens, HEAD_DIM), jnp.bfloat16),  # new_kv
        ((pages, CACHE_ROWS, ROW_WORDS), jnp.int32),  # cache_kv
        ((MAX_SEQS,), jnp.int32),  # kv_lens
        ((MAX_SEQS * PAGES_PER_SEQ,), jnp.int32),  # page_indices
        ((MAX_SEQS + 1,), jnp.int32),  # cu_q_lens
        ((3,), jnp.int32),  # distribution
        ((HEADS,), jnp.float32),  # attention_sinks
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

        handle.device = topologies.get_topology_desc(TOPOLOGY, platform="tpu").devices[
            0
        ]
    sharding = jax.sharding.SingleDeviceSharding(handle.device)
    args = [
        jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
        for shape, dtype in _arg_specs(handle, params["tokens"])
    ]
    with _tpu7x_mesh():
        lowered = (
            jax.jit(_entry(handle, params))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
        )
    mem = lowered.compile().memory_analysis()
    text = lowered.as_text()
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "kernels": text.count("tpu_custom_call"),
        "attention": "SWA-" in text,
    }


def limit(name, result):
    if not result["attention"]:
        return "the compiled program launches no SWA kernel: nothing was checked"
    return None


class _Override:
    """A module with some attributes replaced."""

    def __init__(self, real, **replaced):
        self._real = real
        self.__dict__.update(replaced)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _Ledger:
    """The bytes every DMA start and wait moves, per kernel launch and semaphore.

    The kernels start several copies on one semaphore (a window's pages and its
    new tokens, a cache update's pages) and wait for them with one descriptor
    of their total size, so the ledger sums bytes rather than counting copies.
    Interpret mode cannot order callbacks inside a kernel, so it sums rather
    than replays: per launch and semaphore, more bytes waited than started
    would hang the core, and more started than waited is a copy no step waited
    for, which Mosaic's exit check halts on. A launch is numbered when it is
    traced."""

    def __init__(self):
        self.tracing = 0
        self.counts = {}

    def event(self, kind, key, nbytes, *index):
        key = (*key, *(int(i) for i in index))
        started, waited = self.counts.get(key, (0, 0))
        nbytes = int(nbytes)
        self.counts[key] = (
            started + nbytes * (kind == "start"),
            waited + nbytes * (kind == "wait"),
        )

    def faults(self):
        out = []
        for (launch, sem, *index), (started, waited) in sorted(self.counts.items()):
            if started != waited:
                out.append(
                    f"launch {launch} semaphore {sem}{index}: {started} bytes started, "
                    f"{waited} waited"
                )
        return out


def _dma_bytes(jax, jnp, ref):
    """The bytes a copy into or out of `ref` moves, traced when a slice size is."""
    base = getattr(ref, "ref", ref)
    aval = jax.typeof(base)
    aval = getattr(aval, "inner_aval", aval)
    for t in getattr(ref, "transforms", ()):
        aval = t.transform_type(aval)
    n = jnp.int32(1)
    for d in aval.shape:
        n = n * jnp.asarray(d, jnp.int32)
    return n * aval.dtype.itemsize


class _CountedCopy:
    """A DMA that moves nothing and logs the bytes of its start and wait on its
    semaphore.

    The semaphore is named by the scratch array it belongs to and its run-time
    index into it."""

    def __init__(self, jax, jnp, ledger, names, dst, sem):
        self._jax, self._ledger = jax, ledger
        self._key = (
            ledger.tracing,
            names.setdefault(id(getattr(sem, "ref", sem)), len(names)),
        )
        self._bytes = _dma_bytes(jax, jnp, dst)
        self._index = [
            jnp.asarray(getattr(i, "start", i), jnp.int32)
            for t in getattr(sem, "transforms", ())
            for i in getattr(t, "indices", ())
        ]

    def _log(self, kind):
        self._jax.debug.callback(
            functools.partial(self._ledger.event, kind, self._key),
            self._bytes,
            *self._index,
        )

    def start(self, *args, **kwargs):
        self._log("start")

    def wait(self, *args, **kwargs):
        self._log("wait")


@contextlib.contextmanager
def _interpret(handle):
    """Run the op's Pallas kernels in interpret mode with every DMA replaced by a
    `_CountedCopy`, and yield the `_Ledger` they log to.

    The copies move nothing because interpret mode cannot read the bitcast HBM
    refs the kernels copy from, so the arithmetic runs on zeros and only the DMA
    schedule is checked."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    jax, jnp, k = handle.jax, handle.jnp, handle.kernel
    ledger, names = _Ledger(), {}
    params = pltpu.InterpretParams(
        uninitialized_memory="zero", out_of_bounds_reads="raise"
    )

    def pallas_call(*args, **kwargs):
        launch = pl.pallas_call(*args, interpret=params, **kwargs)

        def numbered(*operands):
            ledger.tracing += 1
            return launch(*operands)

        return numbered

    def make_async_copy(src, dst, sem, *args, **kwargs):
        return _CountedCopy(jax, jnp, ledger, names, dst, sem)

    real = (k.pl, k.pltpu)
    k.pl = _Override(pl, pallas_call=pallas_call)
    k.pltpu = _Override(pltpu, make_async_copy=make_async_copy)
    try:
        yield ledger
    finally:
        k.pl, k.pltpu = real


#: Key lengths, cycled over the decode sequences: below, at and across the
#: window and the page, and the longest the server holds.
DECODE_KV_LENS = (1, 2, 127, 128, 129, 255, 256, 257, 8448, MAX_MODEL_LEN)


def _runtime_inputs(tokens):
    """`(description, [(q_len, kv_len), ...] per sequence, decode count)`.

    Sequences `[0, decodes)` are decode-only and the rest go through the
    mixed pass, as the runner's distribution orders them."""
    bq = NUM_QUERIES_PER_BLOCK[2]
    for n in sorted(
        {
            1,
            DECODE_SEQ_BATCH - 1,
            DECODE_SEQ_BATCH,
            DECODE_SEQ_BATCH + 1,
            min(tokens, MAX_SEQS),
        }
    ):
        if n <= min(tokens, MAX_SEQS):
            seqs = [(1, DECODE_KV_LENS[i % len(DECODE_KV_LENS)]) for i in range(n)]
            yield f"{n} decode sequences", seqs, n
    lengths = {
        1,
        2,
        bq - 1,
        bq,
        bq + 1,
        LOGICAL_PAGE_SIZE - 1,
        LOGICAL_PAGE_SIZE,
        LOGICAL_PAGE_SIZE + 1,
        tokens - 1,
        tokens,
    }
    for n in sorted(q for q in lengths if 1 <= q <= tokens):
        for ctx in (0, 1, 1023, MAX_MODEL_LEN - n):
            yield f"one chunk of {n} tokens after {ctx}", [(n, n + ctx)], 0
    for d in (1, 3):
        n = tokens - d
        if n >= 1:
            seqs = [(1, 8448 + i) for i in range(d)] + [(n, n + 4092)]
            yield f"{d} decodes and a chunk of {n} tokens after 4092", seqs, d
    if tokens >= 4:
        parts = [tokens // 2, tokens // 4, tokens - tokens // 2 - tokens // 4]
        seqs = [(q, q + c) for q, c in zip(parts, (0, 130, 7000))]
        yield f"three chunks of {parts}", seqs, 0


def _inputs_for(handle, tokens, seqs, decodes):
    """Arguments for one runtime input, with a small cache."""
    jnp, np = handle.jnp, handle.np
    rng = np.random.default_rng(len(seqs) * 7919 + tokens)
    q = jnp.asarray(rng.standard_normal((tokens, HEADS, HEAD_DIM)), jnp.bfloat16)
    new_kv = jnp.asarray(rng.standard_normal((tokens, HEAD_DIM)), jnp.bfloat16)
    # The copies move nothing (`_interpret`), so the cache contents are never read.
    cache = jnp.zeros((INTERPRET_PAGES, CACHE_ROWS, ROW_WORDS), jnp.int32)
    kv_lens = np.zeros(MAX_SEQS, np.int32)
    kv_lens[: len(seqs)] = [kv for _, kv in seqs]
    page_indices = (np.arange(MAX_SEQS * PAGES_PER_SEQ) % INTERPRET_PAGES).astype(
        np.int32
    )
    cu = np.zeros(MAX_SEQS + 1, np.int32)
    cu[1 : len(seqs) + 1] = np.cumsum([q for q, _ in seqs])
    cu[len(seqs) + 1 :] = cu[len(seqs)]
    dist = np.array([decodes, decodes, len(seqs)], np.int32)
    sinks = jnp.zeros((HEADS,), jnp.float32)
    return (
        q,
        new_kv,
        cache,
        jnp.asarray(kv_lens),
        jnp.asarray(page_indices),
        jnp.asarray(cu),
        jnp.asarray(dist),
        sinks,
    )


def trip_counts(handle, name, params):
    jax = handle.jax
    tokens = params["tokens"]
    records = []
    with _interpret(handle) as ledger:
        run = jax.jit(_entry(handle, params))
        for description, seqs, decodes in _runtime_inputs(tokens):
            ledger.counts.clear()
            jax.block_until_ready(run(*_inputs_for(handle, tokens, seqs, decodes)))
            jax.effects_barrier()
            if not ledger.counts:
                raise RuntimeError(
                    f"{description}: the kernels started no DMA this adapter "
                    f"can see, so nothing was checked"
                )
            if ledger.faults():
                raise RuntimeError(f"{description}: {'; '.join(ledger.faults()[:2])}")
            records.append(
                {
                    "pipe": "SWA kernels double buffer",
                    "buffer_count": 2,
                    "trip_count": len(seqs) - decodes + (1 if decodes else 0),
                    "lookahead": False,
                    "input": description,
                }
            )
    return records
