#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TPU counterpart to vLLM's ``benchmarks/kernels/bench_kda_module.py``.

Runs ``KimiDeltaAttention`` from
``vllm_torchtpu.models.vllm.kimi_k3.attention`` on the *same* weights, inputs and initial
KV-cache state that the GPU harness dumps, and diffs the output and final state.
Also times the module, and can produce a dump of its own so the GPU harness's
``--validate`` can check the reverse direction.

    # on the GPU box
    python benchmarks/kernels/bench_kda_module.py --mode prefill \
        --seq-lens 2048 --num-heads 12 --hidden-size 7168 \
        --dump /shared/k3_prefill_gpu.pt

    # here
    python scripts/kernels/bench_kda_module_tpu.py \
        --validate /shared/k3_prefill_gpu.pt

    # reverse direction: produce a TPU dump for the GPU harness to check
    python scripts/kernels/bench_kda_module_tpu.py --mode decode --num-seqs 32 \
        --num-heads 12 --hidden-size 7168 --dump /shared/k3_decode_tpu.pt

    # where the module's device time goes (per-HLO-op table), with or without
    # a dump to validate against
    python scripts/kernels/bench_kda_module_tpu.py --validate /shared/k3_decode_gpu.pt \
        --profile /tmp/kda_mod_prof
    python scripts/kernels/bench_kda_module_tpu.py --mode prefill --seq-lens 2048 \
        --num-heads 12 --hidden-size 7168 --profile /tmp/kda_mod_prof

Add ``--compile`` to either form to wrap the complete module forward in one
static ``torch.compile(backend="tpu", fullgraph=True)`` region. Compile time is
paid during the pre-capture warmup and is excluded from timing and profiles.

`--profile` captures with **torch_tpu's** profiler rather than `jax.profiler`,
because the module runs on torch_tpu's PJRT client; see `_profile_module`. This
is the module-level counterpart to `bench_kda.py --profile`, and it is where
non-KDA cost shows up: the ragged Pallas conv1d, the projections, and state
writeback.

Dump format is the GPU harness's, unchanged: ``config`` / ``state_dict`` /
``inputs`` / ``outputs``. Most ``state_dict`` keys line up because the prototype
mirrors upstream's attribute names (``f_b_proj``, ``conv1d``, ``dt_bias``,
``A_log``, ``o_norm``, ``o_proj``); upstream's ``decode_*`` buffers are
``persistent=False`` and so are absent from the dump, which is what the prototype
wants anyway.

The one key that does *not* line up is the input projection. Upstream fuses it
into ``in_proj_qkvgfab``; the prototype splits it into ``in_proj_qkv`` /
``g_proj`` / ``f_a_proj`` / ``b_proj`` so that ``f_a`` can be replicated rather
than sharded at TP > 1. ``_split_in_proj`` and ``_fuse_in_proj`` convert between
the two, so both directions still work unchanged: dumps are split on load and
re-fused on save. They are exact inverses, and a dump already in either layout is
accepted.

Two consumer-side subtleties the GPU harness now records for us:

* ``conv_state_dim_first`` -- conv-state memory layout, set by
  ``VLLM_SSM_CONV_STATE_LAYOUT`` rather than by the platform. If the producer and
  consumer disagree, the conv state is silently transposed and every number
  downstream looks like a kernel bug. Handled in ``_orient_conv_state``.
* ``o_norm_eps`` -- upstream's ``o_norm`` runs at ``FusedRMSNormGated``'s default
  1e-5, NOT at the ``rms_norm_eps`` (1e-6) also present in the dump.

How the module is constructed
-----------------------------
The prototype's real ``__init__`` builds vLLM parallel linears, which needs an
initialized TP group, which needs ``get_current_vllm_config()``, which needs a
``ModelConfig`` -- i.e. a model on disk. Rather than depend on that, this harness
builds the module directly and substitutes plain ``nn.Linear`` for the
projections. At TP=1 that is numerically identical (``nn.Linear.weight`` is
``[out, in]``, same as ``ColumnParallelLinear``'s and ``ReplicatedLinear``'s), and
the weights come from the dump either way, so what is being compared is the KDA
path -- conv, kernels, gate, norm -- and not vLLM's linear layers. It also means
this harness cannot exercise the TP > 1 sharding the split projections exist for;
that needs a real multi-rank run.

The guard against that hand-build drifting from the real ``__init__`` is
``_assert_state_dict_matches``: the constructed module's ``state_dict`` keys and
shapes must equal the dump's exactly, after the input-projection conversion
above.

Unsupported (the prototype raises rather than guessing): ``--mode mixed`` and any
spec-decode batch.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
import types
from collections.abc import Callable
from contextlib import contextmanager, nullcontext

import torch
from torch import nn

import vllm_torchtpu  # noqa: F401  (registers the TPU platform)
import vllm_torchtpu.layers.vllm.custom_ops.kda_attention_op as kda_ops
from vllm_torchtpu.models.vllm.kimi_k3.attention import KimiDeltaAttention

# Defaults mirroring the GPU harness.
K3_NUM_HEADS = 96
K3_HEAD_DIM = 128
K3_HIDDEN_SIZE = 7168  # the checkpoint's real hidden size
K3_CONV_KERNEL = 4
K3_GATE_LOWER_BOUND = -5.0
K3_RMS_EPS = 1e-6
K3_O_NORM_EPS = 1e-5  # FusedRMSNormGated's default, which upstream relies on
K3_NUM_KDA_LAYERS = 7  # the 9-layer checkpoint has KDA at [0,1,2,4,5,6,8]

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


# ---------------------------------------------------------------------------
# Module construction
# ---------------------------------------------------------------------------
def build_module(num_heads: int, head_dim: int, hidden_size: int,
                 conv_kernel: int, gate_lower_bound: float, o_norm_eps: float,
                 dtype, device, naive: bool = False) -> KimiDeltaAttention:
    """Build the layer with nn.Linear stand-ins. See module docstring.

    ``naive`` selects the plain-XLA reference recurrence for every batch, the
    same thing ``VLLM_TPU_USE_NAIVE_KDA=1`` does in production -- except set
    directly on the instance, so an A/B in this script cannot be confused by the
    compile cache the way an engine-level A/B can.
    """
    m = object.__new__(KimiDeltaAttention)
    nn.Module.__init__(m)

    proj = num_heads * head_dim
    m.prefix = "model.layers.1.linear_attn"
    m.head_dim = head_dim
    m.total_heads = num_heads
    m.num_heads = num_heads
    m.tp_size = 1
    m.projection_size = proj
    m.local_projection_size = proj
    m.conv_size = conv_kernel
    m.gate_lower_bound = gate_lower_bound
    m.use_full_rank_gate = True
    m.kv_cache = None
    m.use_naive_kda = naive
    from vllm.model_executor.layers.mamba.mamba_utils import \
        is_conv_state_dim_first
    dim_first = is_conv_state_dim_first()

    # q/k/v are separate projections in the layer; the GPU side keeps them fused
    # as `in_proj_qkvgfab`. `_split_in_proj` / `_fuse_in_proj` convert.
    m.q_proj = _WrappedLinear(hidden_size, proj, dtype, device)
    m.k_proj = _WrappedLinear(hidden_size, proj, dtype, device)
    m.v_proj = _WrappedLinear(hidden_size, proj, dtype, device)
    m.g_proj = _WrappedLinear(hidden_size, proj, dtype, device)
    m.f_a_proj = _WrappedLinear(hidden_size, head_dim, dtype, device)
    m.b_proj = _WrappedLinear(hidden_size, num_heads, dtype, device)
    m.f_b_proj = _WrappedLinear(head_dim, proj, dtype, device)
    m.o_proj = _WrappedLinear(proj, hidden_size, dtype, device)

    # One [proj, 1, conv_kernel] fp32 depthwise weight per projection.
    m.q_conv1d = _ConvHolder(proj, conv_kernel, device)
    m.k_conv1d = _ConvHolder(proj, conv_kernel, device)
    m.v_conv1d = _ConvHolder(proj, conv_kernel, device)
    m.dt_bias = nn.Parameter(
        torch.zeros(proj, dtype=torch.float32, device=device))
    m.A_log = nn.Parameter(
        torch.zeros(num_heads, dtype=torch.float32, device=device))
    m.o_norm = _NormWeight(head_dim, device)

    # The real ops, built exactly as __init__ builds them.
    m.sconv_op = kda_ops.build_kimi_sconv_op(m.prefix,
                                             kernel_size=conv_kernel,
                                             state_dim_first=dim_first)
    m.kda_op = kda_ops.build_kimi_kda_op(m.prefix,
                                         lower_bound=gate_lower_bound,
                                         eps=o_norm_eps)
    m.chunk_kda_op = kda_ops.build_kimi_chunk_kda_op(
        m.prefix, lower_bound=gate_lower_bound, eps=o_norm_eps)
    m.decode_kda_op = kda_ops.build_kimi_decode_kda_op(
        m.prefix,
        lower_bound=gate_lower_bound,
        eps=o_norm_eps,
        state_dim_first=dim_first)
    return m


class _WrappedLinear(nn.Module):
    """nn.Linear whose forward returns ``(out, bias)`` like vLLM's linears."""

    def __init__(self, in_features: int, out_features: int, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=dtype, device=device))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class _ConvHolder(nn.Module):
    """Holds ``weight`` of shape [proj, 1, conv_size], like CausalDepthwiseConv1d."""

    def __init__(self, dim: int, conv_size: int, device):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(dim, 1, conv_size, dtype=torch.float32, device=device))


class _NormWeight(nn.Module):
    """Stand-in for RMSNorm: only ``weight`` is read, by the kernels."""

    def __init__(self, dim: int, device):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(dim, dtype=torch.float32, device=device))


class _Metadata:
    """The subset of AttentionMetadata the KDA layer reads."""

    def __init__(self, *, query_start_loc, mamba_state_indices, seq_lens,
                 is_decode_only):
        self.query_start_loc = query_start_loc
        self.mamba_state_indices = mamba_state_indices
        self.seq_lens = seq_lens
        self.is_decode_only = is_decode_only


# ---------------------------------------------------------------------------
# Fused <-> split input projection
# ---------------------------------------------------------------------------
# Upstream (and so every GPU dump) carries one fused `in_proj_qkvgfab.weight`
# whose rows are [q | k | v | out_gate | f_a | beta | pad]. The TPU layer splits
# that into `in_proj_qkv` / `g_proj` / `f_a_proj` / `b_proj` so f_a can be
# replicated at TP > 1 (see the layer's __init__). These two helpers convert
# between the layouts and are exact inverses, so the harness keeps working in
# both directions: split on --validate, fuse on --dump.
_FUSED_IN_PROJ_KEY = "in_proj_qkvgfab.weight"


def _in_proj_row_sizes(num_heads: int, head_dim: int) -> list[int]:
    """Row counts of the fused projection's shards, at tp_size == 1."""
    proj = num_heads * head_dim
    sizes = [3 * proj, proj, head_dim, num_heads]
    # Upstream 16-aligns the fused row count; the split layer has no such pad,
    # but the dumped fused weight still carries it.
    pad = -sum(sizes) % 16
    if pad:
        sizes.append(pad)
    return sizes


def _split_in_proj(sd: dict, num_heads: int, head_dim: int) -> dict:
    """Fused `in_proj_qkvgfab` -> the four separate projections.

    Returns the state_dict unchanged if it is already split, so a TPU-produced
    dump round-trips without special-casing.
    """
    if _FUSED_IN_PROJ_KEY not in sd:
        return sd

    fused = sd[_FUSED_IN_PROJ_KEY]
    sizes = _in_proj_row_sizes(num_heads, head_dim)
    if fused.shape[0] != sum(sizes):
        raise ValueError(
            f"{_FUSED_IN_PROJ_KEY} has {fused.shape[0]} rows; expected "
            f"{sum(sizes)} for num_heads={num_heads} head_dim={head_dim} "
            f"(shards {sizes}). Does the dump's geometry match its config?")

    shards = list(fused.split(sizes, dim=0))
    out = {k: v for k, v in sd.items() if k != _FUSED_IN_PROJ_KEY}
    out["in_proj_qkv.weight"] = shards[0]
    out["g_proj.weight"] = shards[1]
    out["f_a_proj.weight"] = shards[2]
    out["b_proj.weight"] = shards[3]
    if len(shards) > 4 and shards[4].abs().max().item() != 0:
        # Upstream zeroes the pad rows right after construction, and the layer
        # relies on that: they contribute nothing to any of the four outputs.
        # Non-zero pad means the dump came from something that reused those rows,
        # and dropping them here would silently change the result.
        raise ValueError("the fused projection's alignment padding is non-zero; "
                         "it cannot be dropped safely")
    return out


def _fuse_in_proj(sd: dict, num_heads: int, head_dim: int) -> dict:
    """The four separate projections -> fused `in_proj_qkvgfab`, pad included.

    Inverse of :func:`_split_in_proj`, so a dump written here can be checked by
    the GPU harness, which builds the real upstream module and expects the fused
    weight at its 16-aligned size.
    """
    keys = ("in_proj_qkv.weight", "g_proj.weight", "f_a_proj.weight",
            "b_proj.weight")
    if not all(k in sd for k in keys):
        return sd

    shards = [sd[k] for k in keys]
    sizes = _in_proj_row_sizes(num_heads, head_dim)
    if len(sizes) > 4:
        ref = shards[0]
        shards.append(
            torch.zeros(sizes[4], ref.shape[1], dtype=ref.dtype,
                        device=ref.device))
    out = {k: v for k, v in sd.items() if k not in keys}
    out[_FUSED_IN_PROJ_KEY] = torch.cat(shards, dim=0)
    return out


def _assert_state_dict_matches(module, dump_sd: dict) -> None:
    """Guard against the hand-build drifting from the real __init__."""
    mine = {k: tuple(v.shape) for k, v in module.state_dict().items()}
    theirs = {k: tuple(v.shape) for k, v in dump_sd.items()}
    missing = sorted(set(theirs) - set(mine))
    extra = sorted(set(mine) - set(theirs))
    mismatched = sorted(k for k in set(mine) & set(theirs)
                        if mine[k] != theirs[k])
    if missing or extra or mismatched:
        lines = ["state_dict mismatch between the TPU module and the dump:"]
        for k in missing:
            lines.append(f"  missing on TPU : {k} {theirs[k]}")
        for k in extra:
            lines.append(f"  extra on TPU   : {k} {mine[k]}")
        for k in mismatched:
            lines.append(f"  shape differs  : {k} tpu={mine[k]} dump={theirs[k]}")
        raise AssertionError("\n".join(lines))


# ---------------------------------------------------------------------------
# Dump plumbing
# ---------------------------------------------------------------------------
def _orient_conv_state(conv_state: torch.Tensor,
                       dump_dim_first: bool | None) -> torch.Tensor:
    """Convert a conv state between the dump's layout and this platform's.

    Transposes iff the two layouts differ, which makes this symmetric: the same
    call converts dump -> local (before the forward) and local -> dump (before
    comparing against the dump's ``final_conv_state``). Pass the dump's flag in
    both directions; negating it forces a spurious transpose.

    The prototype's ``_forward`` normalizes using this platform's
    ``is_conv_state_dim_first()``, so what it receives must be in the layout this
    platform *stores*.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import \
        is_conv_state_dim_first

    here = is_conv_state_dim_first()
    if dump_dim_first is None:
        # Older dumps predate the field; "SD" is the default on both sides.
        dump_dim_first = False
    if bool(dump_dim_first) != bool(here):
        return conv_state.transpose(-1, -2).contiguous()
    return conv_state


def _make_segment_ids(query_start_loc: torch.Tensor, num_tokens: int,
                      device) -> torch.Tensor:
    """Build chunk-kernel segment IDs on CPU, outside the module graph."""
    starts = query_start_loc.cpu().tolist()
    segment_ids = torch.zeros((1, num_tokens), dtype=torch.int32)
    for index, (start, end) in enumerate(zip(starts, starts[1:]), start=1):
        segment_ids[0, start:min(end, num_tokens)] = index
    return segment_ids.to(device)


def metadata_from_dump(cfg: dict, inputs: dict, num_tokens: int,
                       device) -> "_Metadata":
    """Rebuild the layer's metadata from the dumped tensors."""
    is_prefilling = list(cfg["is_prefilling"])
    num_prefills = sum(1 for x in is_prefilling if x)
    num_decodes = len(is_prefilling) - num_prefills

    seq_lens = list(cfg["seq_lens"])
    query_lens = list(cfg["query_lens"])

    state_indices = inputs["state_indices"]
    if state_indices is None:
        # The only field that cannot be reconstructed: slot assignment is the
        # allocator's choice, not a function of the batch shape.
        raise ValueError(
            "dump has no state_indices; it came from a spec-decode batch "
            "(spec_state_indices_tensor instead) or an older harness")

    qsl = inputs["non_spec_query_start_loc"]
    if qsl is None:
        qsl = torch.tensor([0] + list(torch.tensor(query_lens).cumsum(0)),
                           dtype=torch.int32)

    has_init = inputs["has_initial_state"]
    if has_init is None:
        # A pure-decode dump legitimately omits this: on the GPU side
        # has_initial_state is only consumed by the prefill `causal_conv1d_fn`,
        # so the metadata builder leaves it unset when there are no prefills.
        # Derive it the same way vLLM does -- a sequence carries state in iff it
        # has previously computed context, i.e. seq_len > query_len.
        has_init = torch.tensor([s > q for s, q in zip(seq_lens, query_lens)],
                                dtype=torch.int32)

    qsl = qsl.to(torch.int32)
    # The ops read carry-in as seq_len > query_len, so rebuild seq_lens to carry
    # the dumped flag rather than passing has_initial_state separately.
    carry = has_init.to(torch.int32)
    seq_lens_t = torch.tensor(query_lens, dtype=torch.int32) + carry
    return _Metadata(
        query_start_loc=qsl.to(device),
        mamba_state_indices=state_indices.to(torch.int32).to(device),
        seq_lens=seq_lens_t.to(device),
        is_decode_only=num_prefills == 0 and num_decodes > 0,
    )


def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    max_abs = (a - b).abs().max().item()
    rmse = (a - b).square().mean().sqrt().item()
    base = b.square().mean().sqrt().item()
    return max_abs, rmse / (base + 1e-8)


# ---------------------------------------------------------------------------
# Profiling the module forward
# ---------------------------------------------------------------------------
def _synchronize_tpu() -> None:
    """Drain queued TPU work without depending on torch_tpu internals."""
    torch.tpu.synchronize()


@contextmanager
def _module_profile_ranges(module: nn.Module):
    """Add readable host ranges without instrumenting the production module."""
    names = {
        "q_proj": "kda/projection/q",
        "k_proj": "kda/projection/k",
        "v_proj": "kda/projection/v",
        "g_proj": "kda/projection/output_gate",
        "f_a_proj": "kda/projection/gate_a",
        "b_proj": "kda/projection/beta",
        "f_b_proj": "kda/projection/gate_b",
        "o_proj": "kda/output_projection",
    }
    handles = []
    ranges: dict[str, list] = {name: [] for name in names}

    for child_name, label in names.items():
        child = getattr(module, child_name)

        def enter(_module, _args, *, _name=child_name, _label=label):
            marker = torch.profiler.record_function(_label)
            marker.__enter__()
            ranges[_name].append(marker)

        def exit_range(_module, _args, _output, *, _name=child_name):
            ranges[_name].pop().__exit__(None, None, None)

        handles.append(child.register_forward_pre_hook(enter))
        handles.append(child.register_forward_hook(exit_range))

    kda_module = sys.modules[KimiDeltaAttention.__module__]
    patched_functions = {}
    for function_name, label in {}.items():
        original = getattr(kda_module, function_name)
        patched_functions[function_name] = original

        def wrapped(*args, _fn=original, _label=label, **kwargs):
            with torch.profiler.record_function(_label):
                return _fn(*args, **kwargs)

        setattr(kda_module, function_name, wrapped)

    patched_methods = {}
    for method_name, label in {
            "_segment_ids": "kda/metadata/segment_ids",
            "_get_conv_op": "kda/conv1d/prefill",
            "_get_chunk_op": "kda/kernel/chunk",
            "_get_decode_op": "kda/kernel/decode",
            "_get_naive_op": "kda/kernel/naive",
    }.items():
        original = getattr(module, method_name)
        patched_methods[method_name] = original
        if method_name.startswith("_get_"):

            def wrapped_get(*args, _fn=original, _label=label, **kwargs):
                op = _fn(*args, **kwargs)

                def profiled_op(*op_args, **op_kwargs):
                    with torch.profiler.record_function(_label):
                        return op(*op_args, **op_kwargs)

                return profiled_op

            setattr(module, method_name, wrapped_get)
        else:

            def wrapped_method(*args, _fn=original, _label=label, **kwargs):
                with torch.profiler.record_function(_label):
                    return _fn(*args, **kwargs)

            setattr(module, method_name, wrapped_method)

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for function_name, original in patched_functions.items():
            setattr(kda_module, function_name, original)
        for method_name in patched_methods:
            delattr(module, method_name)


def _decode_tool_payload(payload):
    raw = payload.decode() if isinstance(payload, bytes) else payload
    return json.loads(raw)


def _data_table_rows(payload) -> list[dict]:
    """Convert an XProf Google DataTable response into ordinary dictionaries."""
    data = _decode_tool_payload(payload)
    if isinstance(data, list):
        data = next((item for item in data
                     if isinstance(item, dict) and "cols" in item), None)
    if not isinstance(data, dict) or "cols" not in data:
        return []
    columns = [column.get("id") or column.get("label")
               for column in data["cols"]]
    return [{column: (cell or {}).get("v")
             for column, cell in zip(columns, row.get("c", []))}
            for row in data.get("rows", [])]


def _write_csv(path: str, rows: list[dict],
               fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_payload(path: str, payload) -> None:
    mode = "wb" if isinstance(payload, bytes) else "w"
    with open(path, mode) as f:
        f.write(payload)


def _write_gap_report(payload, path: str, minimum_us: float = 100.0) -> int:
    """Write idle-looking gaps between complete events on each trace lane."""
    data = _decode_tool_payload(payload)
    events = data.get("traceEvents", []) if isinstance(data, dict) else []
    lane_names: dict[tuple[object, object], str] = {}
    lanes: dict[tuple[object, object], list[dict]] = {}
    for event in events:
        key = (event.get("pid"), event.get("tid"))
        if event.get("ph") == "M" and event.get("name") == "thread_name":
            lane_names[key] = event.get("args", {}).get("name", "")
        elif event.get("ph") == "X" and event.get("dur") is not None:
            lanes.setdefault(key, []).append(event)

    gaps = []
    for key, lane in lanes.items():
        cursor = None
        previous = None
        for event in sorted(lane, key=lambda item: float(item.get("ts", 0))):
            start = float(event.get("ts", 0))
            end = start + float(event.get("dur", 0))
            if cursor is not None and start - cursor >= minimum_us:
                gaps.append({
                    "gap_us": start - cursor,
                    "pid": key[0],
                    "tid": key[1],
                    "lane": lane_names.get(key, ""),
                    "previous_event": previous or "",
                    "next_event": event.get("name", ""),
                })
            if cursor is None or end > cursor:
                cursor = end
                previous = event.get("name", "")
    gaps.sort(key=lambda row: row["gap_us"], reverse=True)
    _write_csv(path, gaps, ["gap_us", "pid", "tid", "lane",
                            "previous_event", "next_event"])
    return len(gaps)


def _profile_module(make_runs: Callable[[int], list[Callable[[], None]]],
                    module: nn.Module, outdir: str, iters: int, top: int,
                    analyse: bool, perf_counters: bool,
                    annotate_python: bool = True) -> None:
    """Capture an annotated CPU+TPU trace and generate XProf report artifacts."""
    from torch_tpu._internal import profiler as tprof

    os.makedirs(outdir, exist_ok=True)
    runs = make_runs(iters + 1)
    _synchronize_tpu()
    runs[0]()  # compile before capture
    _synchronize_tpu()

    activities = [tprof.ProfilerActivity.CPU, tprof.ProfilerActivity.TPU]
    ranges = (_module_profile_ranges(module)
              if annotate_python else nullcontext())
    with ranges:
        with tprof.profile(activities=activities,
                           on_trace_ready=tprof.xprof_trace_handler(outdir)):
            for index, run in enumerate(runs[1:]):
                with torch.profiler.record_function(
                        f"kda/profile_iteration_{index}"):
                    with torch.profiler.record_function("kda/module_forward"):
                        run()
    _synchronize_tpu()

    pbs = sorted(glob.glob(os.path.join(outdir, "**", "*.xplane.pb"),
                           recursive=True))
    if not pbs:
        print(f"  WARNING: no .xplane.pb produced under {outdir}")
        return
    trace_path = max(pbs, key=os.path.getmtime)
    print(f"  trace: {trace_path} "
          f"({os.path.getsize(trace_path) / 1e6:.1f} MB)")
    if not analyse:
        print(f"  (skipping post-processing; view with: xprof {outdir})")
        return

    try:
        from xprof.convert import raw_to_tool_data as r
    except ImportError:
        print("  xprof not installed, so no op breakdown. Either:\n"
              "      uv pip install --python .venv/bin/python xprof\n"
              f"  or view the trace directly:  xprof {outdir}")
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from bench_kda import _format_hlo_stats
    except ImportError as e:
        print(f"  could not import the table formatter from bench_kda.py: {e}")
        return
    analysis_dir = os.path.join(outdir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    tools = ["overview_page", "framework_op_stats", "hlo_stats", "op_profile",
             "roofline_model", "trace_viewer@"]
    if perf_counters:
        tools.append("perf_counters")

    payloads = {}
    for tool in tools:
        try:
            payload, _ctype = r.xspace_to_tool_data([trace_path], tool, {})
            payloads[tool] = payload
            filename = tool.rstrip("@").replace("@", "_") + ".json"
            _write_payload(os.path.join(analysis_dir, filename), payload)
            rows = _data_table_rows(payload)
            _write_csv(os.path.join(analysis_dir, filename[:-5] + ".csv"), rows)
        except Exception as e:
            print(f"  {tool} analysis unavailable: {type(e).__name__}: {e}")

    hlo_payload = payloads.get("hlo_stats")
    if hlo_payload is not None:
        table = _format_hlo_stats(hlo_payload, top)
        print(table)
        with open(os.path.join(analysis_dir, "summary.md"), "w") as f:
            f.write("# KDA module profile\n\n```text\n")
            f.write(table)
            f.write("\n```\n")
        print(f"  shares are of summed op self time over {iters} captured "
              "forwards (idle excluded).")

    trace_payload = payloads.get("trace_viewer@")
    if trace_payload is not None:
        gap_count = _write_gap_report(
            trace_payload, os.path.join(analysis_dir, "largest_gaps.csv"))
        print(f"  gap report: {gap_count} lane gaps >= 100 us")
    print(f"  analysis: {analysis_dir}")
    print("  op_profile and roofline_model provide the most useful custom-call "
          "breakdown; perf counters require --profile-perf-counters.")


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_validate(args, device) -> int:
    dump = torch.load(args.validate, weights_only=False, map_location="cpu")
    cfg, sd, inp, ref = (dump["config"], dump["state_dict"], dump["inputs"],
                         dump["outputs"])
    dtype = DTYPES[cfg["dtype"]]

    if cfg["mode"] == "mixed":
        print("FATAL: the TPU prototype does not support mixed batches "
              "(it raises rather than guessing the token split).",
              file=sys.stderr)
        return 2

    module = build_module(cfg["num_heads"], cfg["head_dim"], cfg["hidden_size"],
                          cfg.get("conv_kernel", K3_CONV_KERNEL),
                          cfg.get("gate_lower_bound", K3_GATE_LOWER_BOUND),
                          cfg.get("o_norm_eps", K3_O_NORM_EPS), dtype, device)
    # GPU dumps carry the fused in_proj; the TPU layer wants it split.
    sd = _split_in_proj(sd, cfg["num_heads"], cfg["head_dim"])
    _assert_state_dict_matches(module, sd)
    module.load_state_dict({k: v.to(device) for k, v in sd.items()},
                           strict=True)

    conv_state = _orient_conv_state(inp["init_conv_state"],
                                    cfg.get("conv_state_dim_first"))
    live_conv_state = conv_state.to(device)
    live_recurrent_state = inp["init_recurrent_state"].to(device)
    module.kv_cache = (live_conv_state, live_recurrent_state)

    hidden_states = inp["hidden_states"].to(device)
    positions = inp["positions"].to(device)
    meta = metadata_from_dump(cfg, inp, hidden_states.shape[0], device)
    forward = _make_forward(module, meta, args.compile)

    out = forward(hidden_states, positions, metadata=meta)
    conv_after, rec_after = module.kv_cache

    # Back to the dump's orientation for comparison (symmetric, same flag).
    conv_cmp = _orient_conv_state(conv_after.cpu(),
                                  cfg.get("conv_state_dim_first"))

    # State pools are mostly untouched slots that are identical by construction,
    # so a whole-pool rmse divides any real error by ~sqrt(num_slots) and can
    # turn a bad slot into a PASS. Report the touched slots separately and gate
    # pass/fail on those.
    slots = inp["state_indices"].to(torch.long).flatten()
    slots = torch.unique(slots[slots > 0])
    checks = [
        ("output", out, ref["output"], True),
        ("final_conv_state[all]", conv_cmp, ref["final_conv_state"], False),
        ("final_conv_state[used]", conv_cmp[slots],
         ref["final_conv_state"][slots], True),
        ("final_recurrent_state[all]", rec_after,
         ref["final_recurrent_state"], False),
        ("final_recurrent_state[used]", rec_after.cpu()[slots],
         ref["final_recurrent_state"][slots], True),
    ]

    print(f"validate      : {args.validate}")
    print(f"dump source   : mode={cfg['mode']} "
          f"prefill_backend={cfg.get('prefill_backend')} "
          f"fused_decode={cfg.get('fused_decode')}")
    print(f"tpu module    : KimiDeltaAttention "
          f"(chunked prefill / recurrent decode)")
    print(f"execution     : {_execution_label(args.compile)}")
    print(f"geometry      : heads/device={cfg['num_heads']} "
          f"head_dim={cfg['head_dim']} hidden={cfg['hidden_size']} "
          f"dtype={cfg['dtype']}")
    print(f"conv layout   : dump dim_first={cfg.get('conv_state_dim_first')} "
          f"o_norm_eps={cfg.get('o_norm_eps', K3_O_NORM_EPS)}")
    print(f"tolerance     : rmse_rel < {args.rtol}")
    print(f"state slots   : pool has {rec_after.shape[0]}, "
          f"touched {slots.tolist()}")
    print()
    print(f"{'tensor':>30} {'max_abs':>12} {'rmse_rel':>12}  result")
    print("-" * 66)
    all_pass = True
    for name, actual, expected, gating in checks:
        max_abs, rmse_rel = _diff(actual, expected)
        ok = rmse_rel < args.rtol
        if gating:
            all_pass &= ok
        verdict = ("PASS" if ok else "FAIL") if gating else "(info)"
        print(f"{name:>30} {max_abs:>12.6f} {rmse_rel:>12.6f}  {verdict}")
    print()
    print("[all] rows are informational: untouched pool slots are identical by "
          "construction and dilute the metric. Pass/fail uses [used].")
    print("VALIDATION PASSED" if all_pass else "VALIDATION FAILED")

    if args.profile:
        print(f"\n--- profile: module forward "
              f"({args.profile_iters} captured forwards) ---")

        initial_conv = conv_state.to(device)
        initial_recurrent = inp["init_recurrent_state"].to(device)
        live_conv_state.copy_(initial_conv)
        live_recurrent_state.copy_(initial_recurrent)

        def make_profile_runs(count):
            def run():
                forward(hidden_states, positions, metadata=meta)

            return [run] * count

        try:
            _profile_module(make_profile_runs, module, args.profile,
                            args.profile_iters, args.top,
                            analyse=not args.no_profile_analysis,
                            perf_counters=args.profile_perf_counters,
                            annotate_python=not args.compile)
        except Exception as e:
            print(f"  profiling FAILED: {type(e).__name__}: {str(e)[:160]}")

    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# Run / dump / time
# ---------------------------------------------------------------------------
def make_batch(mode: str, num_seqs: int, seq_len: int, decode_context: int):
    if mode == "prefill":
        seq_lens = [seq_len] * num_seqs
        query_lens = [seq_len] * num_seqs
        is_prefilling = [True] * num_seqs
    elif mode == "decode":
        seq_lens = [decode_context + 1] * num_seqs
        query_lens = [1] * num_seqs
        is_prefilling = [False] * num_seqs
    else:
        raise ValueError(f"unsupported mode for the TPU prototype: {mode}")
    return seq_lens, query_lens, is_prefilling


def init_weights(module, seed: int, dtype) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def fill(t, scale):
        src = torch.randn(t.shape, generator=gen, dtype=torch.float32) * scale
        t.data.copy_(src.to(t.dtype))

    fill(module.q_proj.weight, 0.02)
    fill(module.k_proj.weight, 0.02)
    fill(module.v_proj.weight, 0.02)
    fill(module.g_proj.weight, 0.02)
    fill(module.f_a_proj.weight, 0.02)
    fill(module.b_proj.weight, 0.02)
    fill(module.f_b_proj.weight, 0.02)
    fill(module.o_proj.weight, 0.02)
    fill(module.q_conv1d.weight, 0.1)
    fill(module.k_conv1d.weight, 0.1)
    fill(module.v_conv1d.weight, 0.1)
    fill(module.dt_bias, 0.1)
    fill(module.A_log, 0.5)
    fill(module.o_norm.weight, 0.2)


def _execution_label(compiled: bool) -> str:
    return ("torch.compile (tpu, fullgraph, static)" if compiled else
            "torch_tpu defer-and-fuse")


def _make_forward(module: KimiDeltaAttention, metadata: "_Metadata",
                  compile_module: bool) -> Callable[..., torch.Tensor]:
    """Return a ``(hidden_states, positions, metadata=...)`` callable.

    The layer takes ``(positions, hidden_states)`` and reads its metadata from
    the forward context; this harness has no engine, so the metadata is bound to
    the instance and the argument order adapted here rather than in the layer.
    """
    module._metadata = lambda: metadata
    inner = (torch.compile(module, backend="tpu", fullgraph=True,
                           dynamic=False) if compile_module else module)

    def forward(hidden_states, positions, metadata=None):
        if metadata is not None:
            module._metadata = lambda: metadata
        return inner(positions, hidden_states)

    return forward


@torch.inference_mode()
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    p.add_argument("--seq-lens", default="2048")
    p.add_argument("--num-seqs", default="1")
    p.add_argument("--num-heads", type=int, default=K3_NUM_HEADS)
    p.add_argument("--head-dim", type=int, default=K3_HEAD_DIM)
    p.add_argument("--hidden-size", type=int, default=K3_HIDDEN_SIZE)
    p.add_argument("--decode-context", type=int, default=0)
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true",
                   help="wrap the complete module forward in torch.compile "
                        "(backend=tpu, fullgraph=True, dynamic=False)")
    p.add_argument("--dump", metavar="PATH", default=None,
                   help="write a snapshot in the GPU harness's format (first "
                        "case only), for its --validate to check")
    p.add_argument("--validate", metavar="PATH", default=None,
                   help="load a GPU (or TPU) dump, rerun on identical weights/"
                        "inputs/state, and diff output + final KV state")
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--profile", metavar="DIR", default=None,
                   help="capture a TPU trace of the module forward into DIR and "
                        "print a per-HLO-op breakdown. Works with --validate "
                        "too (profiles the same forward that was checked).")
    p.add_argument("--profile-iters", type=int, default=1,
                   help="forwards to capture (default 1 for a readable timeline)")
    p.add_argument("--top", type=int, default=15,
                   help="rows of the per-op table to show (default 15)")
    p.add_argument("--no-profile-analysis", action="store_true",
                   help="capture the trace but skip xprof post-processing")
    p.add_argument("--profile-perf-counters", action="store_true",
                   help="also export XProf TPU performance counters (large)")
    p.add_argument("--naive", action="store_true",
                   help="use the plain-XLA reference recurrence for every "
                        "batch, as VLLM_TPU_USE_NAIVE_KDA=1 does. Set on the "
                        "instance, so an A/B here cannot be confounded by the "
                        "engine's compile cache.")
    args = p.parse_args()
    if args.profile_iters < 1:
        p.error("--profile-iters must be at least 1")

    device = torch.device("tpu")
    if args.validate:
        return run_validate(args, device)

    dtype = DTYPES[args.dtype]
    module = build_module(args.num_heads, args.head_dim, args.hidden_size,
                          K3_CONV_KERNEL, K3_GATE_LOWER_BOUND, K3_O_NORM_EPS,
                          dtype, device, naive=args.naive)
    init_weights(module, args.seed, dtype)

    seq_lens_arg = [int(x) for x in args.seq_lens.split(",") if x]
    num_seqs_list = [int(x) for x in args.num_seqs.split(",") if x]
    cases = ([(n, 1) for n in num_seqs_list] if args.mode == "decode" else
             [(n, t) for n in num_seqs_list for t in seq_lens_arg])

    from vllm.model_executor.layers.mamba.mamba_utils import \
        is_conv_state_dim_first
    dim_first = is_conv_state_dim_first()

    print(f"device        : {torch.device('tpu')} (torch_tpu)")
    print(f"module        : KimiDeltaAttention (hidden={args.hidden_size})")
    print(f"heads/device  : {args.num_heads}   head_dim: {args.head_dim}   "
          f"dtype: {args.dtype}")
    print(f"mode          : {args.mode}")
    if args.naive:
        kernel = "naive_kda + ragged_causal_conv1d (reference scan)"
    elif args.mode == "prefill":
        kernel = "chunk_kda + ragged_causal_conv1d"
    else:
        kernel = "fused_conv_decode_kda"

    print(f"kernels       : {kernel}")
    print(f"execution     : {_execution_label(args.compile)}")
    print(f"conv layout   : dim_first={dim_first}   o_norm_eps={K3_O_NORM_EPS}")
    print(f"iters         : {args.iters} (warmup {args.warmup})")
    print()
    hdr = (f"{'seqs':>5} {'seqlen':>7} {'tokens':>8} {'median_ms':>10} "
           f"{'min_ms':>8} {'p90_ms':>8} {'xNlayers_ms':>12}")
    print(hdr)
    print("-" * len(hdr))

    for i, (num_seqs, seq_len) in enumerate(cases):
        seq_lens, query_lens, is_prefilling = make_batch(
            args.mode, num_seqs, seq_len, args.decode_context)
        num_tokens = sum(query_lens)

        conv_dim = 3 * args.num_heads * args.head_dim
        state_len = K3_CONV_KERNEL - 1
        conv_shape = ((conv_dim, state_len) if dim_first else
                      (state_len, conv_dim))
        num_slots = num_seqs + 2

        gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        conv_state = (torch.randn(num_slots, *conv_shape, generator=gen,
                                  dtype=torch.float32) * 0.1).to(device)
        rec_state = (torch.randn(num_slots, args.num_heads, args.head_dim,
                                 args.head_dim, generator=gen,
                                 dtype=torch.float32) * 0.01).to(device)
        module.kv_cache = (conv_state, rec_state)

        qsl = torch.tensor([0] + list(torch.tensor(query_lens).cumsum(0)),
                           dtype=torch.int32)
        # Slots start at 1: slot 0 is the reserved null block.
        carry = torch.tensor([0 if x else 1 for x in is_prefilling],
                             dtype=torch.int32)
        meta = _Metadata(
            query_start_loc=qsl.to(device),
            mamba_state_indices=torch.arange(1, num_seqs + 1,
                                             dtype=torch.int32).to(device),
            seq_lens=(torch.tensor(query_lens, dtype=torch.int32) +
                      carry).to(device),
            is_decode_only=args.mode == "decode",
        )

        g2 = torch.Generator(device="cpu").manual_seed(args.seed + 100 + i)
        hidden_states = (torch.randn(num_tokens, args.hidden_size,
                                     generator=g2, dtype=torch.float32) *
                         0.1).to(dtype).to(device)
        positions = torch.arange(num_tokens, dtype=torch.long, device=device)
        forward = _make_forward(module, meta, args.compile)

        if args.dump and i == 0:
            init_conv_snap = conv_state.clone()
            init_rec_snap = rec_state.clone()
            out = forward(hidden_states, positions, metadata=meta)
            conv_after, rec_after = module.kv_cache
            torch.save(
                {
                    "config": {
                        "num_heads": args.num_heads,
                        "head_dim": args.head_dim,
                        "hidden_size": args.hidden_size,
                        "conv_kernel": K3_CONV_KERNEL,
                        "gate_lower_bound": K3_GATE_LOWER_BOUND,
                        "rms_norm_eps": K3_RMS_EPS,
                        "conv_state_dim_first": dim_first,
                        "o_norm_eps": K3_O_NORM_EPS,
                        "dtype": args.dtype,
                        "mode": args.mode,
                        "prefill_backend": "tpu_chunked_pallas",
                        "fused_decode": args.mode == "decode",
                        "block_size": 64,
                        "seq_lens": seq_lens,
                        "query_lens": query_lens,
                        "is_prefilling": is_prefilling,
                    },
                    # Fuse the input projections back, so the GPU harness --
                    # which builds the real upstream module -- can load this.
                    "state_dict":
                    _fuse_in_proj(
                        {
                            k: v.detach().cpu()
                            for k, v in module.state_dict().items()
                        }, args.num_heads, args.head_dim),
                    "inputs": {
                        "hidden_states": hidden_states.detach().cpu(),
                        "positions": positions.detach().cpu(),
                        "state_indices": meta.state_indices.cpu(),
                        "non_spec_query_start_loc": meta.query_start_loc.cpu(),
                        "has_initial_state": meta.has_initial_state.cpu(),
                        "init_conv_state": init_conv_snap.cpu(),
                        "init_recurrent_state": init_rec_snap.cpu(),
                    },
                    "outputs": {
                        "output": out.detach().cpu(),
                        "final_conv_state": conv_after.detach().cpu(),
                        "final_recurrent_state": rec_after.detach().cpu(),
                    },
                }, args.dump)
            print(f"# dumped TPU snapshot -> {args.dump}", file=sys.stderr)
            conv_state.copy_(init_conv_snap)
            rec_state.copy_(init_rec_snap)

        def call():
            forward(hidden_states, positions, metadata=meta)

        if args.profile and i == 0:
            print(f"\n--- profile: {args.mode} n{num_seqs} t{seq_len} "
                  f"({args.profile_iters} captured forwards) ---")
            snap = ((conv_state.clone(), rec_state.clone())
                    if not args.compile else None)

            def make_profile_runs(count):
                return [call] * count

            try:
                _profile_module(make_profile_runs, module, args.profile,
                                args.profile_iters, args.top,
                                analyse=not args.no_profile_analysis,
                                perf_counters=args.profile_perf_counters,
                                annotate_python=not args.compile)
            except Exception as e:
                print(f"  profiling FAILED: {type(e).__name__}: {str(e)[:160]}")
            # A compiled graph threads mutated cache buffers through its graph
            # outputs. The pre-compile Python handles may therefore be donated;
            # restoring through them can enqueue a write to an invalid buffer.
            if not args.compile:
                assert snap is not None
                conv_state.copy_(snap[0])
                rec_state.copy_(snap[1])
            print()

        call()
        for _ in range(args.warmup):
            call()
        _synchronize_tpu()
        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            call()
            _synchronize_tpu()
            samples.append((time.perf_counter() - t0) * 1e3)
        samples.sort()
        median = statistics.median(samples)
        p90 = samples[min(len(samples) - 1, int(0.9 * len(samples)))]
        print(f"{num_seqs:>5} {seq_len:>7} {num_tokens:>8} {median:>10.4f} "
              f"{samples[0]:>8.4f} {p90:>8.4f} "
              f"{median * K3_NUM_KDA_LAYERS:>12.4f}")

    print()
    print(f"xNlayers_ms = median x {K3_NUM_KDA_LAYERS} KDA layers. Timing "
          "includes the KDA kernel(s) and all surrounding module ops.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
