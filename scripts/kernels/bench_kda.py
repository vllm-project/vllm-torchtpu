#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Latency benchmark for the KDA Pallas TPU forward kernel at Kimi-K3 shapes.

Standalone: needs only the plugin's venv and a TPU. No vLLM engine, no server.

    # prefill sweep at TP=8 (12 heads/device), the real K3 flag combination
    python scripts/kernels/bench_kda.py --tp 8 --seq-lens 512,1024,2048,4096

    # decode with the recurrent kernel (the default; what K3 actually runs)
    python scripts/kernels/bench_kda.py --tp 8 --mode decode --num-seqs 1,8,32,128

    # the same shapes through the chunked kernel, to see the padding penalty
    python scripts/kernels/bench_kda.py --tp 8 --mode decode --num-seqs 1,8,32,128 \
        --decode-impl chunked

    # machine-readable
    python scripts/kernels/bench_kda.py --tp 8 --json results.json

    # where the device time actually goes (per-HLO-op breakdown via XProf)
    python scripts/kernels/bench_kda.py --tp 8 --seq-lens 1024 --profile /tmp/kda_prof

    # compile-time analysis only: FLOPs, HBM traffic, VMEM pressure (no profiler)
    python scripts/kernels/bench_kda.py --tp 8 --seq-lens 1024 --analyze

Profiling notes
---------------
``--profile`` captures a TPU trace with ``jax.profiler.trace`` and, if ``xprof``
is installed, post-processes it into a per-HLO-op table (device self time, DMA
stall, what each op is bound by). It runs as a **separate pass after** timing, so
profiler overhead never contaminates the reported latencies.

This is worth doing because the Pallas kernel is not the whole story: at
1024 tokens the two KDA custom-calls account for only ~20% of device self time,
with the rest in copies, gathers, a sort and scatter-offload that XLA emits
around them. A latency number alone cannot show that.

``--profile`` requires ``jaxlib`` and ``libtpu`` to agree on the PJRT profiler
ABI. They do not always: ``jaxlib`` 0.10.2 with ``libtpu`` 0.0.44.x aborts the
process with ``Unexpected PLUGIN_Profiler_Api size: expected 80, got 104``
followed by SIGSEGV inside ``PluginTracer::Start()``. The script pre-flights that
pairing and refuses with instructions rather than letting it segfault mid-run;
``--force-profile`` overrides. ``--analyze`` needs no profiler and always works.

Note that the pairing ``pyproject.toml`` declares -- jax/jaxlib 0.10.2 with
``libtpu==0.0.43`` -- profiles correctly. A venv where ``--profile`` refuses has
drifted off that pin (``torch-tpu`` only requires ``libtpu>=0.0.41``, so libtpu
can float upward on its own).

``xprof`` is an optional extra, needed only for the per-op table (the capture
itself is pure ``jax.profiler``):

    uv pip install --python .venv/bin/python xprof

Defaults come from the reference Kimi-K3 config
(``Inferact/Kimi-K3-dummy-9L``): ``linear_attn_config.num_heads=96``,
``head_dim=128``, ``gate_lower_bound=-5.0``, 6 KDA layers out of 8, bfloat16.
Heads are tensor-parallel sharded, so per-device heads = ``num_heads / tp``.

By default the benchmark uses the exact flag combination a K3 prefill issues:
varlen ``segment_ids``, raw gate activated in-kernel from ``A_log``/``dt_bias``,
``lower_bound``, in-kernel q/k L2 norm, per-segment ``initial_state``, and
``output_final_state``. Benchmarking a cheaper combination would understate the
cost of the path that actually runs.

Two caveats on interpreting the numbers:

* ``--mode decode`` uses ``decode_kda``, the single-token recurrent kernel a
  real K3 decode step runs -- no chunk padding, and the per-sequence state
  gather happens inside the kernel as a paged DMA. Pass
  ``--decode-impl chunked`` to instead drive ``chunk_kda`` with 1-token
  segments, which pads each to ``chunk_size=64`` and does ~64x the useful work;
  that path is kept only to quantify the penalty, and is ~14x slower at 128
  sequences.
* The FLOP figure is a **lower bound on the dominant matmuls only** (see
  ``_approx_flops``). Treat it as a relative signal between configurations, not
  as hardware utilisation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.kimi_k3 import chunk_kda, decode_kda

# Reference Kimi-K3 linear_attn_config values.
K3_NUM_HEADS = 96
K3_HEAD_DIM = 128
K3_GATE_LOWER_BOUND = -5.0
K3_NUM_KDA_LAYERS = 6  # 6 of 8 layers are KDA (kda_layers=[1,2,3,5,6,7])
CHUNK_SIZE = 64  # the only size the kernel supports

DTYPES = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}


@dataclass
class Result:
    mode: str
    impl: str
    seq_len: int
    num_seqs: int
    total_tokens: int
    padded_tokens: int
    pad_waste: float
    heads_per_device: int
    head_dim: int
    dtype: str
    compile_s: float
    median_ms: float
    min_ms: float
    p90_ms: float
    tokens_per_s: float
    per_model_forward_ms: float
    approx_tflops: float


def _approx_flops(total_tokens: int, heads: int, k: int, v: int,
                  chunk: int) -> float:
    """Approximate FLOPs for the dominant matmuls. LOWER BOUND.

    Counted terms, per head, per chunk of length ``C``:
      intra-chunk relation matrices Aqk, Akk : 2 * 2*C*C*K
      state read/apply  q @ S                : 2*C*K*V
      state update      kᵀ @ v_new           : 2*C*K*V
    Not counted: the triangular solve, gate cumsum, L2 norm, the w/u
    construction, or any elementwise work. Real FLOPs are higher.
    """
    num_chunks = max(1, total_tokens // chunk)
    per_chunk = 2 * (2 * chunk * chunk * k) + 2 * (2 * chunk * k * v)
    return num_chunks * heads * per_chunk


def _roundtrip_floor(iters: int = 20) -> float:
    """Median ms for a trivial jitted op plus a full sync.

    This is the cost of the measurement apparatus itself -- Python dispatch, PJRT
    enqueue, and one host<->device round trip. Any row whose median is close to
    this is reporting the harness rather than the kernel, which matters most for
    decode: the recurrent step's device time can be well under the floor.
    """

    @jax.jit
    def trivial(x):
        return x + 1.0

    x = jnp.ones((1, ), dtype=jnp.float32)
    jax.block_until_ready(trivial(x))
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(trivial(x))
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _approx_flops_recurrent(total_tokens: int, heads: int, k: int,
                            v: int) -> float:
    """Approximate FLOPs for the single-token recurrent step. LOWER BOUND.

    Per token, per head, three ``[K, V]``-sized contractions:
      ``kv = k @ S``, the rank-1 state update, and ``out = q @ S``.
    Not counted: the per-channel decay multiply, gate activation or L2 norm.
    """
    return total_tokens * heads * 6 * k * v


def _make_segment_ids(num_seqs: int, seq_len: int) -> jnp.ndarray:
    """1-indexed ``[1, T]`` segment IDs: ``num_seqs`` segments of ``seq_len``."""
    total = num_seqs * seq_len
    ids = (jnp.arange(total, dtype=jnp.int32) // seq_len) + 1
    return ids.reshape(1, total)


def _build_inputs(num_seqs: int, seq_len: int, heads: int, head_dim: int,
                  dtype, seed: int = 0):
    """Random inputs in the kernel's head-major ``[H, B, T, D]`` layout.

    Distributions follow the tests (uniform q/k/v, ``logsigmoid`` raw gate,
    sigmoid beta); exact values do not affect latency, but keeping them in range
    avoids denormals or infinities perturbing the timing.
    """
    key = jax.random.key(seed)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    total = num_seqs * seq_len
    B, K, V = 1, head_dim, head_dim

    q = jax.random.uniform(k1, (heads, B, total, K), dtype=dtype)
    k = jax.random.uniform(k2, (heads, B, total, K), dtype=dtype)
    v = jax.random.uniform(k3, (heads, B, total, V), dtype=dtype)
    # Raw gate: activated in-kernel, so pre-activation scale is what matters.
    raw_g = jax.random.normal(k4, (heads, B, total, K), dtype=jnp.float32)
    raw_g = jnp.asarray(jax.nn.log_sigmoid(raw_g), dtype=dtype)
    beta = jax.nn.sigmoid(jax.random.normal(k5, (heads, B, total),
                                           dtype=jnp.float32)).astype(dtype)

    A_log = (jax.random.normal(k6, (heads, ), dtype=jnp.float32) * 0.5) - 1.0
    dt_bias = jnp.zeros((heads * K, ), dtype=jnp.float32)
    # One recurrent state per segment: [B, N, H, K, V], as prefill carries in.
    init = jnp.zeros((B, num_seqs, heads, K, V), dtype=jnp.float32)
    seg = _make_segment_ids(num_seqs, seq_len)
    return q, k, v, raw_g, beta, A_log, dt_bias, init, seg


VMEM_MIB_PER_CORE = 64  # TPU7x per-TensorCore VMEM; see env_override's chip table


def _version_tuple(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in v.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _profiler_abi_status() -> tuple[bool, str]:
    """Check whether jaxlib and libtpu agree on the PJRT profiler ABI.

    There is no API to query the plugin's ``PLUGIN_Profiler_Api`` struct size, and
    a mismatch does not raise -- it SIGSEGVs inside ``PluginTracer::Start()``,
    taking the benchmark with it. So this is a version-pair heuristic derived from
    the observed failure: jaxlib 0.10.2 + libtpu 0.0.44.1 aborts with "expected
    80, got 104", while jaxlib 0.11.0 + libtpu 0.0.44.1 traces correctly on the
    same machine.

    Returns ``(ok, message)``; ``message`` is non-empty only when not ok.
    """
    import importlib.metadata as md

    try:
        jaxlib_v = md.version("jaxlib")
        libtpu_v = md.version("libtpu")
    except md.PackageNotFoundError as e:
        return True, f"could not determine versions ({e}); attempting anyway"

    if (_version_tuple(jaxlib_v) < (0, 11, 0)
            and _version_tuple(libtpu_v) >= (0, 0, 44)):
        return False, (
            f"jaxlib {jaxlib_v} + libtpu {libtpu_v} have incompatible PJRT "
            f"profiler ABIs.\n"
            f"  Tracing would abort with 'Unexpected PLUGIN_Profiler_Api size: "
            f"expected 80, got 104' and SIGSEGV in PluginTracer::Start().\n"
            f"  This is environment drift, not a code problem: pyproject.toml "
            f"pins libtpu==0.0.43, which pairs correctly with jaxlib "
            f"{jaxlib_v}.\n"
            f"  libtpu is able to float because torch-tpu only requires "
            f"libtpu>=0.0.41.\n"
            f"  Preferred fix -- restore the declared pin:\n"
            f"      uv pip install --python .venv/bin/python 'libtpu==0.0.43'\n"
            f"  (verified: jax/jaxlib 0.10.2 + libtpu 0.0.43 traces correctly on "
            f"TPU7x and\n"
            f"   reports the same op breakdown. Check serving still works before "
            f"keeping it.)\n"
            f"  Alternative, if libtpu 0.0.44.x is required: move jax forward "
            f"instead, with\n"
            f"      uv pip install --python .venv/bin/python 'jax[tpu]==0.11.0'\n"
            f"  though that contradicts the jax==0.10.2 pin in pyproject.toml.\n"
            f"  Use --force-profile to try anyway, or --analyze for "
            f"profiler-free analysis.")
    return True, ""


def _static_analysis(compiled) -> dict:
    """Compile-time cost and memory facts. Needs no profiler.

    ``cost_analysis`` reports FLOPs and total bytes accessed, which together give
    arithmetic intensity -- the number that says whether a shape is compute- or
    memory-bound. ``memory_analysis`` reports peak and temp buffer sizes, which is
    how to see a working set outgrowing the 64 MiB of VMEM per TensorCore.
    """
    out: dict = {}
    try:
        ca = compiled.cost_analysis()
        if isinstance(ca, (list, tuple)):  # older jax returned a list of dicts
            ca = ca[0] if ca else {}
        flops = ca.get("flops")
        bytes_accessed = ca.get("bytes accessed")
        out["flops"] = flops
        out["bytes_accessed"] = bytes_accessed
        out["transcendentals"] = ca.get("transcendentals")
        if flops and bytes_accessed:
            out["arithmetic_intensity"] = round(flops / bytes_accessed, 2)
    except Exception as e:
        out["cost_analysis_error"] = f"{type(e).__name__}: {e}"

    try:
        ma = compiled.memory_analysis()
        for attr in ("peak_memory_in_bytes", "temp_size_in_bytes",
                     "argument_size_in_bytes", "output_size_in_bytes",
                     "generated_code_size_in_bytes"):
            out[attr] = getattr(ma, attr, None)
        peak = out.get("peak_memory_in_bytes")
        if peak:
            out["peak_vs_vmem"] = round(
                peak / (VMEM_MIB_PER_CORE * 1024 * 1024), 3)
    except Exception as e:
        out["memory_analysis_error"] = f"{type(e).__name__}: {e}"
    return out


def _print_static_analysis(label: str, a: dict) -> None:
    print(f"\n--- XLA compile-time analysis: {label} ---")
    for err in ("cost_analysis_error", "memory_analysis_error"):
        if err in a:
            print(f"  {err}: {a[err]}")

    def gb(n):
        return f"{n / 1e9:.3f} GB" if n else "n/a"

    if a.get("flops"):
        print(f"  FLOPs                 : {a['flops']:.3e}")
    if a.get("bytes_accessed"):
        print(f"  bytes accessed        : {gb(a['bytes_accessed'])}")
    if a.get("arithmetic_intensity") is not None:
        print(f"  arithmetic intensity  : {a['arithmetic_intensity']} FLOP/byte")
    if a.get("transcendentals"):
        print(f"  transcendentals       : {a['transcendentals']:.3e}")
    peak = a.get("peak_memory_in_bytes")
    if peak:
        print(f"  peak memory           : {peak / 2**20:.1f} MiB "
              f"({a['peak_vs_vmem']:.0%} of {VMEM_MIB_PER_CORE} MiB VMEM/core)")
    for attr, name in (("temp_size_in_bytes", "temp buffers"),
                       ("argument_size_in_bytes", "arguments"),
                       ("output_size_in_bytes", "outputs")):
        if a.get(attr):
            print(f"  {name:<21} : {a[attr] / 2**20:.1f} MiB")
    print("  (peak/temp above the VMEM figure means the working set streams "
          "from HBM)")


def _format_hlo_stats(payload, top: int) -> str:
    """Render xprof's ``hlo_stats`` Google-DataTable JSON as a table.

    Percentages are recomputed locally as a share of **summed op self time**, not
    taken from XProf's ``total_self_time_percent``. XProf's figure is a share of
    the whole capture window, and roughly half of that window is IDLE (the host
    gaps between the captured calls), so its percentages understate every op by
    about 2x and do not sum to 100 over the ops. The absolute microsecond column
    is XProf's own and is unmodified.
    """
    data = json.loads(payload if isinstance(payload, str) else payload.decode())
    if isinstance(data, list):  # some tools wrap the table in a list
        data = data[0]
    cols = [c["id"] for c in data["cols"]]
    rows = data.get("rows", [])
    if not rows:
        return "  (hlo_stats returned no rows)"

    def cell(row, cid, default=None):
        return row["c"][cols.index(cid)].get("v") if cid in cols else default

    total_us = sum(float(cell(r, "total_self_time") or 0.0) for r in rows)
    if total_us <= 0:
        return "  (hlo_stats reported zero device self time)"

    head = (f"{'op':<40} {'category':<16} {'occ':>4} {'self_us':>9} "
            f"{'share':>7} {'dma%':>6} {'bound_by':<13}")
    lines = [head, "-" * len(head)]
    for row in rows[:top]:
        self_us = float(cell(row, "total_self_time") or 0.0)
        lines.append(
            f"{str(cell(row, 'hlo_op_name'))[:40]:<40} "
            f"{str(cell(row, 'category'))[:16]:<16} "
            f"{int(cell(row, 'occurrences') or 0):>4} "
            f"{self_us:>9.2f} {self_us / total_us:>6.1%} "
            f"{float(cell(row, 'dma_stall_percent') or 0.0):>5.1f}% "
            f"{str(cell(row, 'bound_by'))[:13]:<13}")
    if len(rows) > top:
        shown_us = sum(float(cell(r, "total_self_time") or 0.0)
                       for r in rows[:top])
        lines.append(f"... {len(rows) - top} more ops = "
                     f"{1 - shown_us / total_us:.1%} of self time "
                     f"(raise with --top)")

    # Category rollup, and the split that actually matters: how much of the
    # device time is the Pallas kernel versus XLA glue around it.
    by_cat: dict[str, float] = {}
    for row in rows:
        cat = str(cell(row, "category") or "?")
        by_cat[cat] = by_cat.get(cat, 0.0) + float(
            cell(row, "total_self_time") or 0.0)
    lines.append("")
    lines.append(f"  by category (of {total_us:.1f} us total op self time):")
    for cat, us in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {cat:<20} {us:>9.2f} us  {us / total_us:>6.1%}")

    pallas_us = sum(
        float(cell(r, "total_self_time") or 0.0) for r in rows
        if str(cell(r, "category") or "").lower() == "custom-call"
        or "pallas" in str(cell(r, "hlo_op_name") or "").lower()
        or "kda" in str(cell(r, "hlo_op_name") or "").lower())
    lines.append("")
    lines.append(f"  Pallas/custom-call ops  : {pallas_us:>9.2f} us  "
                 f"{pallas_us / total_us:>6.1%}")
    lines.append(f"  XLA glue around them    : {total_us - pallas_us:>9.2f} us  "
                 f"{1 - pallas_us / total_us:>6.1%}")
    return "\n".join(lines)


def _profile_case(case: Case, outdir: str, iters: int, top: int,
                  analyse: bool) -> None:
    """Capture a trace of ``run`` and print a per-HLO-op breakdown.

    Deliberately a separate pass from timing: the device tracer adds overhead, so
    mixing it into the timed loop would corrupt the latency numbers.
    """
    import glob
    import os

    os.makedirs(outdir, exist_ok=True)
    jax.block_until_ready(_advance(case))  # ensure compiled before the capture
    with jax.profiler.trace(outdir):
        for _ in range(iters):
            jax.block_until_ready(_advance(case))

    pbs = sorted(glob.glob(os.path.join(outdir, "**", "*.xplane.pb"),
                           recursive=True))
    if not pbs:
        print(f"  WARNING: no .xplane.pb produced under {outdir}")
        return
    print(f"  trace: {pbs[-1]} ({os.path.getsize(pbs[-1]) / 1e6:.1f} MB)")

    if not analyse:
        print("  (skipping post-processing; view with: xprof "
              f"{outdir})")
        return
    try:
        from xprof.convert import raw_to_tool_data as r
    except ImportError:
        print("  xprof not installed, so no op breakdown. Either:\n"
              "      uv pip install --python .venv/bin/python xprof\n"
              f"  or view the trace directly:  xprof {outdir}")
        return
    try:
        payload, _ctype = r.xspace_to_tool_data([pbs[-1]], "hlo_stats", {})
        print(_format_hlo_stats(payload, top))
        print(f"  shares are of summed op self time over {iters} captured "
              f"calls (idle excluded); 'bound_by' is XProf's attribution.")
    except Exception as e:
        print(f"  hlo_stats post-processing failed: {type(e).__name__}: {e}")
        print(f"  the raw trace is still usable:  xprof {outdir}")


@dataclass
class Case:
    """One benchmark configuration: a jitted callable plus its live inputs.

    ``args`` is mutated as the case runs, because ``decode_kda`` **donates** its
    state pool: the pool that comes back must be fed in as the next call's input.
    Reusing the original would fail with "Donation requested for invalid buffer"
    after the first call -- and that failure surfaces only as a
    MaterializationWorker error on stderr while the caller sees stale results, so
    it is worth being explicit about.

    ``thread`` is ``(index into the returned tuple, index into args)`` for the
    donated buffer, or ``None`` for stateless cases (the chunked kernel takes
    ``initial_state`` as a plain input and does not donate it).
    """
    run: object
    args: tuple
    impl: str
    thread: tuple[int, int] | None = None


def _advance(case: Case):
    """Run one call, threading any donated state back into ``case.args``."""
    out = case.run(*case.args)
    if case.thread is not None:
        ret_idx, arg_idx = case.thread
        args = list(case.args)
        args[arg_idx] = out[ret_idx]
        case.args = tuple(args)
    return out


def _build_decode_case(num_seqs: int, heads: int, head_dim: int,
                       dtype_name: str, seed: int = 0) -> Case:
    """The recurrent decode kernel: one token per sequence, paged state pool.

    This is what a K3 decode step actually runs. Layout differs from the chunked
    kernel: activations are ``[num_seqs, n_v, D]`` rather than head-major
    ``[H, B, T, D]``, ``dt_bias`` is ``[n_v, d_k]`` rather than flat, and the
    state lives in a pool addressed by ``state_indices`` with slot 0 reserved.
    """
    dtype = DTYPES[dtype_name]
    ks = jax.random.split(jax.random.key(seed), 6)
    K = head_dim

    q = jax.random.uniform(ks[0], (num_seqs, heads, K), dtype=dtype)
    k = jax.random.uniform(ks[1], (num_seqs, heads, K), dtype=dtype)
    v = jax.random.uniform(ks[2], (num_seqs, heads, K), dtype=dtype)
    raw_g = jnp.asarray(
        jax.nn.log_sigmoid(
            jax.random.normal(ks[3], (num_seqs, heads, K), dtype=jnp.float32)),
        dtype=dtype)
    beta = jax.nn.sigmoid(
        jax.random.normal(ks[4], (num_seqs, heads),
                          dtype=jnp.float32)).astype(dtype)
    A_log = (jax.random.normal(ks[5], (heads, ), dtype=jnp.float32) * 0.5) - 1.0
    dt_bias = jnp.zeros((heads, K), dtype=jnp.float32)
    # Slot 0 is the reserved null block; sequence s lives in slot s + 1.
    pool = jnp.zeros((num_seqs + 1, heads, K, K), dtype=jnp.float32)
    state_indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
    # Steady-state decode: every sequence carries state in from a prior step.
    has_initial_state = jnp.ones((num_seqs, ), dtype=jnp.int32)

    @jax.jit
    def run(q, k, v, raw_g, beta, pool, A_log, dt_bias, state_indices,
            has_initial_state):
        return decode_kda(q, k, v, raw_g, beta, pool, A_log, dt_bias,
                          state_indices, has_initial_state,
                          lower_bound=K3_GATE_LOWER_BOUND)

    args = (q, k, v, raw_g, beta, pool, A_log, dt_bias, state_indices,
            has_initial_state)
    # decode_kda returns (out, pool); pool is args[5].
    return Case(run=run, args=args, impl="recurrent", thread=(1, 5))


def _build_case(num_seqs: int, seq_len: int, heads: int, head_dim: int,
                dtype_name: str) -> Case:
    """Build a ``Case`` driving the chunked kernel.

    Split out of ``_time_one`` so the same jitted callable and inputs can be
    timed, profiled and analysed without rebuilding or recompiling.
    """
    dtype = DTYPES[dtype_name]
    q, k, v, raw_g, beta, A_log, dt_bias, init, seg = _build_inputs(
        num_seqs, seq_len, heads, head_dim, dtype)
    n_max = num_seqs

    # Flags are Python constants captured here so the jitted function takes only
    # arrays; n_max must be static.
    @jax.jit
    def run(q, k, v, raw_g, beta, A_log, dt_bias, init, seg):
        return chunk_kda(
            q,
            k,
            v,
            raw_g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=K3_GATE_LOWER_BOUND,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            safe_gate=True,
            segment_ids=seg,
            N_max=n_max,
            initial_state=init,
            output_final_state=True,
            chunk_size=CHUNK_SIZE,
        )

    args = (q, k, v, raw_g, beta, A_log, dt_bias, init, seg)
    return Case(run=run, args=args, impl="chunked")


def _time_one(case: Case, num_seqs: int, seq_len: int, heads: int,
              head_dim: int, dtype_name: str, iters: int, warmup: int,
              mode: str) -> Result:
    # First call includes tracing + Pallas compilation; measured separately so it
    # never contaminates the steady-state numbers.
    t0 = time.perf_counter()
    jax.block_until_ready(_advance(case))
    compile_s = time.perf_counter() - t0

    for _ in range(warmup):
        jax.block_until_ready(_advance(case))

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(_advance(case))
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()

    total_tokens = num_seqs * seq_len
    if case.impl == "recurrent":
        # No chunk padding: the recurrent step does exactly one token per
        # sequence, which is the whole point of using it for decode.
        padded_tokens = total_tokens
        flops = _approx_flops_recurrent(total_tokens, heads, head_dim, head_dim)
    else:
        # Each segment is padded up to a chunk_size multiple internally, so this
        # is the token count the kernel actually does work for.
        padded_per_seq = -(-seq_len // CHUNK_SIZE) * CHUNK_SIZE
        padded_tokens = num_seqs * padded_per_seq
        flops = _approx_flops(total_tokens, heads, head_dim, head_dim,
                              CHUNK_SIZE)
    median_ms = statistics.median(samples)

    return Result(
        mode=mode,
        impl=case.impl,
        seq_len=seq_len,
        num_seqs=num_seqs,
        total_tokens=total_tokens,
        padded_tokens=padded_tokens,
        pad_waste=round(padded_tokens / total_tokens, 1),
        heads_per_device=heads,
        head_dim=head_dim,
        dtype=dtype_name,
        compile_s=round(compile_s, 3),
        median_ms=round(median_ms, 4),
        min_ms=round(samples[0], 4),
        p90_ms=round(samples[min(len(samples) - 1,
                                 int(0.9 * len(samples)))], 4),
        tokens_per_s=round(total_tokens / (median_ms * 1e-3)),
        per_model_forward_ms=round(median_ms * K3_NUM_KDA_LAYERS, 4),
        approx_tflops=round(flops / (median_ms * 1e-3) / 1e12, 2),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("prefill", "decode"), default="prefill",
                   help="prefill: one segment of --seq-len per sequence. "
                        "decode: one token per sequence through the chunked "
                        "kernel (padded to 64 internally; see README).")
    p.add_argument("--seq-lens", default="512,1024,2048,4096",
                   help="comma-separated context lengths (prefill mode)")
    p.add_argument("--num-seqs", default="1",
                   help="comma-separated sequence counts (batch size)")
    p.add_argument("--tp", type=int, default=1,
                   help=f"tensor-parallel degree; heads/device = "
                        f"{K3_NUM_HEADS}/tp")
    p.add_argument("--num-heads", type=int, default=None,
                   help="override heads per device (bypasses --tp)")
    p.add_argument("--head-dim", type=int, default=K3_HEAD_DIM)
    p.add_argument("--decode-impl", choices=("recurrent", "chunked"),
                   default="recurrent",
                   help="decode kernel: 'recurrent' is decode_kda, the "
                        "single-token step a real K3 decode runs. 'chunked' "
                        "drives chunk_kda with 1-token segments instead, "
                        "which pads each to chunk_size and is kept only to "
                        "quantify that penalty.")
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--json", metavar="PATH", default=None,
                   help="also write results as JSON")
    p.add_argument("--profile", metavar="DIR", default=None,
                   help="capture an XProf/TPU trace per case into DIR/<case>/ "
                        "and print a per-HLO-op breakdown. Runs as a separate "
                        "pass after timing.")
    p.add_argument("--profile-iters", type=int, default=5,
                   help="iterations inside the profiler capture (default 5)")
    p.add_argument("--no-profile-analysis", action="store_true",
                   help="capture the trace but skip xprof post-processing")
    p.add_argument("--top", type=int, default=15,
                   help="rows of the per-op table to show (default 15)")
    p.add_argument("--force-profile", action="store_true",
                   help="profile even if the jaxlib/libtpu profiler ABI check "
                        "says it will crash")
    p.add_argument("--analyze", action="store_true",
                   help="print XLA compile-time cost/memory analysis (FLOPs, "
                        "HBM traffic, VMEM pressure). Needs no profiler.")
    p.add_argument("--dump-hlo", metavar="DIR", default=None,
                   help="write each case's optimized HLO text to DIR")
    args = p.parse_args()

    profile_dir = args.profile
    if profile_dir:
        ok, msg = _profiler_abi_status()
        if not ok and not args.force_profile:
            print(f"FATAL: --profile is unsafe here.\n{msg}", file=sys.stderr)
            return 2
        if not ok:
            print(f"WARNING: proceeding despite the ABI check (--force-profile)"
                  f"\n{msg}\n", file=sys.stderr)

    if args.num_heads is not None:
        heads = args.num_heads
    else:
        if K3_NUM_HEADS % args.tp:
            p.error(f"--tp {args.tp} does not divide {K3_NUM_HEADS} heads")
        heads = K3_NUM_HEADS // args.tp

    devices = jax.devices()
    if not any(d.platform == "tpu" for d in devices):
        print(f"FATAL: no TPU device found (got {[d.platform for d in devices]})",
              file=sys.stderr)
        return 2

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x]
    num_seqs_list = [int(x) for x in args.num_seqs.split(",") if x]
    if args.mode == "decode":
        # Decode is one token per sequence; --seq-lens does not apply.
        cases = [(n, 1) for n in num_seqs_list]
    else:
        cases = [(n, t) for n in num_seqs_list for t in seq_lens]

    print(f"device       : {devices[0].device_kind} x{len(devices)}")
    decode_impl = args.decode_impl if args.mode == "decode" else "-"
    print(f"mode         : {args.mode}"
          + (f"   decode impl: {decode_impl} "
             f"({'decode_kda' if decode_impl == 'recurrent' else 'chunk_kda'})"
             if args.mode == "decode" else ""))
    print(f"heads/device : {heads}  (K3 num_heads={K3_NUM_HEADS}, tp={args.tp})")
    print(f"head_dim     : {args.head_dim}   dtype: {args.dtype}   "
          f"chunk: {CHUNK_SIZE}")
    print(f"K3 flags     : varlen + raw gate (A_log/dt_bias) + "
          f"lower_bound={K3_GATE_LOWER_BOUND} + l2norm + initial_state")
    floor_ms = _roundtrip_floor()
    print(f"iters        : {args.iters} (warmup {args.warmup})")
    print(f"timing       : per call (dispatch + enqueue + sync); round-trip "
          f"floor {floor_ms:.4f} ms")
    print()
    hdr = (f"{'seqs':>5} {'seqlen':>7} {'tokens':>8} {'padded':>8} "
           f"{'waste':>6} {'median_ms':>10} {'min_ms':>8} {'tok/s':>10} "
           f"{'x6layers_ms':>12} {'~TFLOP/s':>9} {'compile_s':>10}")
    print(hdr)
    print("-" * len(hdr))

    results = []
    built = {}
    for num_seqs, seq_len in cases:
        try:
            if args.mode == "decode" and args.decode_impl == "recurrent":
                case = _build_decode_case(num_seqs, heads, args.head_dim,
                                          args.dtype)
            else:
                case = _build_case(num_seqs, seq_len, heads, args.head_dim,
                                   args.dtype)
            r = _time_one(case, num_seqs, seq_len, heads, args.head_dim,
                          args.dtype, args.iters, args.warmup, args.mode)
        except Exception as e:  # keep sweeping; report the failure inline
            print(f"{num_seqs:>5} {seq_len:>7} {'':>8} "
                  f"FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        built[(num_seqs, seq_len)] = case
        results.append(asdict(r))
        print(f"{r.num_seqs:>5} {r.seq_len:>7} {r.total_tokens:>8} "
              f"{r.padded_tokens:>8} {r.pad_waste:>5.1f}x "
              f"{r.median_ms:>10.4f} {r.min_ms:>8.4f} "
              f"{r.tokens_per_s:>10} {r.per_model_forward_ms:>12.4f} "
              f"{r.approx_tflops:>9.2f} {r.compile_s:>10.3f}")

    print()
    print("x6layers_ms = median_ms x 6 KDA layers (one K3 model forward's worth "
          "of KDA, excluding the 2 MLA layers and everything else).")
    print("~TFLOP/s is a LOWER BOUND over dominant matmuls only -- use it to "
          "compare configurations, not as utilisation.")
    near_floor = [r for r in results if r["median_ms"] < 2 * floor_ms]
    if near_floor:
        print(f"\nWARNING: {len(near_floor)} of {len(results)} row(s) are within "
              f"2x of the {floor_ms:.4f} ms round-trip floor, so they are "
              f"dominated by\nper-call dispatch and sync rather than kernel "
              f"time -- read them as an upper bound. This is expected for "
              f"decode,\nwhose recurrent step is genuinely faster than one host "
              f"round trip. For device time, use --profile (the per-op table\n"
              f"excludes idle) or chain several layers into one program, as a "
              f"real forward does.")
    if args.mode == "decode" and args.decode_impl == "recurrent":
        print("waste = 1.0x: the recurrent step does exactly one token per "
              "sequence, with no chunk padding.")
    else:
        print("waste = padded/useful tokens: each segment is padded to "
              f"chunk_size={CHUNK_SIZE} internally, so a 1-token decode segment "
              f"costs {CHUNK_SIZE}x its useful work. Use "
              "`--mode decode --decode-impl recurrent` for the kernel a real "
              "decode step uses.")

    # --- compile-time analysis and profiling: separate passes, after timing ---
    analyses: dict[str, dict] = {}
    if args.analyze or args.dump_hlo:
        for (num_seqs, seq_len), case in built.items():
            label = f"{args.mode} n{num_seqs} t{seq_len} [{case.impl}]"
            try:
                compiled = case.run.lower(*case.args).compile()
            except Exception as e:
                print(f"\n--- analysis {label}: FAILED "
                      f"{type(e).__name__}: {str(e)[:80]}")
                continue
            if args.analyze:
                a = _static_analysis(compiled)
                analyses[label] = a
                _print_static_analysis(label, a)
            if args.dump_hlo:
                import os
                os.makedirs(args.dump_hlo, exist_ok=True)
                path = os.path.join(
                    args.dump_hlo,
                    f"{args.mode}_n{num_seqs}_t{seq_len}.optimized.hlo.txt")
                with open(path, "w") as f:
                    f.write(compiled.as_text())
                print(f"  wrote optimized HLO: {path}")

    if profile_dir:
        import os
        for (num_seqs, seq_len), case in built.items():
            name = f"{args.mode}_n{num_seqs}_t{seq_len}_{case.impl}"
            print(f"\n--- profile: {name} "
                  f"({args.profile_iters} captured calls) ---")
            try:
                _profile_case(case, os.path.join(profile_dir, name),
                              args.profile_iters, args.top,
                              analyse=not args.no_profile_analysis)
            except Exception as e:
                print(f"  profiling FAILED: {type(e).__name__}: {str(e)[:120]}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"config": vars(args) | {"heads_per_device": heads},
                       "results": results,
                       "analysis": analyses}, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
