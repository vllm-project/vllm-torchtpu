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
"""Per-layer, per-expert routing counts of a serving run, counted in-graph.

Opt-in profiling tool (see ``sitecustomize.py`` and the README one directory
up); nothing in vllm_torchtpu imports it.

- Counts live in one int32 device tensor shared by every MoE layer. The
  wrapped forward adds one layer's top-k ids into it with ``index_add_``.
  vLLM compiles the model as one full graph and torch_tpu's backend runs
  AOTAutograd with ``keep_inference_input_mutations=False``, so the update
  becomes an extra program output that the runtime wrapper copies back after
  each forward: the counter lives inside the compiled program, with no host
  callback and no graph break.
- Routing outputs are unchanged: the wrapper calls the original op with the
  same arguments and returns its result. Next to it, it recomputes the op's
  top-k ids from the same scores and bias (``moe_tp_decode`` routes inside
  its op, so its ids are not visible outside) and reads the op's inputs only.
- Each row is classified from the forward's attention metadata: request r is
  decoding iff its query length is at most 1 + num_speculative_tokens (a
  1-token decode or a speculative verify window). A forward whose requests
  all decode is a decode step, any other is a prefill step (its decode rows
  are kept apart). Padded rows come from token_padding's mask, the rows the
  op skips. Step counters count forwards with at least one valid row.
- Each process writes one cumulative dump, ``ep_routing_counts_rank<r>_pid<p>
  .npz``, between steps at most every ``EP_ROUTING_COUNTER_INTERVAL_S``
  seconds, once after an idle interval, and at exit; write-then-rename, from
  one device-to-host copy (consistent across layers).
- Refuses data or prefill-context parallelism: routed rows would not match
  the metadata.

Layers placed by ``TPU_EXPERT_PLACEMENT`` route over reordered router rows;
their counts are mapped back to global expert ids when dumped.
"""

from __future__ import annotations

import atexit
import dataclasses
import importlib
import os
import re
import threading
import time

import numpy as np
import torch

ENV_DIR = "EP_ROUTING_COUNTER_DIR"
ENV_INTERVAL = "EP_ROUTING_COUNTER_INTERVAL_S"
FORMAT_VERSION = 1

# Row classes of one layer's block.
CLS_DECODE = 0  # decode step, valid row
CLS_PREFILL = 1  # prefill step, row of a prefill request, valid
CLS_PREFILL_DECODE = 2  # prefill step, row of a decoding request, valid
CLS_DECODE_PAD = 3  # decode step, padded row
CLS_PREFILL_PAD = 4  # prefill step, padded row
CLS_UNKNOWN = 5  # no attention metadata at trace time, valid row
CLS_UNKNOWN_PAD = 6  # no attention metadata at trace time, padded row
NUM_CLS = 7
CLASS_NAMES = (
    "decode",
    "prefill",
    "prefill_decode_rows",
    "decode_padded",
    "prefill_padded",
    "unknown",
    "unknown_padded",
)
# Step counters after the class block, added once per layer per forward.
STEP_DECODE = 0
STEP_PREFILL = 1
STEP_UNKNOWN = 2
# Rows whose padding-mask bit disagrees with the metadata's valid row count
# (query_start_loc[num_reqs]). Warm-up forwards add some by construction;
# during serving it must not grow.
STEP_PAD_MISMATCH = 3
NUM_STEP = 4
STEP_NAMES = ("decode_steps", "prefill_steps", "unknown_steps", "pad_mismatch_rows")

_MOE_TP_DECODE = "vllm_torchtpu.layers.adapter.moe_tp_decode"
_MOE_ROUTING = "vllm_torchtpu.layers.adapter.moe_routing"
_ROUTER_TOPK = "vllm_torchtpu.layers.adapter.router_topk"
_TOKEN_PADDING = "vllm_torchtpu.layers.adapter.token_padding"
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _say(msg: str) -> None:
    print(f"[ep_routing_counter] {msg} (pid {os.getpid()})", flush=True)


@dataclasses.dataclass
class CountsState:
    """The counts buffer and what each slot (one MoE layer) holds."""

    counts: torch.Tensor  # int32 [num_slots * row_len]
    num_experts: int
    topk: int
    # slot -> (model layer index, module name, "target" | "draft")
    slots: list[tuple[int, str, str]]
    # A request decodes iff its query length is at most this.
    max_decode_query_len: int = 1
    # slot -> global expert of each router row (placed layers only).
    slot_order: dict[int, list[int]] = dataclasses.field(default_factory=dict)

    @property
    def row_len(self) -> int:
        return row_len_for(self.num_experts)

    @property
    def num_slots(self) -> int:
        return len(self.slots)


def row_len_for(num_experts: int) -> int:
    return NUM_CLS * num_experts + NUM_STEP


_STATE: CountsState | None = None


# Resolved outside any trace, so the traced helpers only read globals.
_TP = None  # token_padding
_FC = None  # vllm.forward_context
_MR = None  # moe_routing
_RT = None  # router_topk (torch adapters)


def resolve_runtime_modules() -> None:
    global _TP, _FC, _MR, _RT
    _TP = _TP or importlib.import_module(_TOKEN_PADDING)
    _FC = _FC or importlib.import_module("vllm.forward_context")
    _MR = _MR or importlib.import_module(_MOE_ROUTING)
    _RT = _RT or importlib.import_module(_ROUTER_TOPK)


# --------------------------------------------------------------------------
# Instrumentation (outside any trace, at the end of load_model).
# --------------------------------------------------------------------------


def _expert_modules(model):
    """The RoutedExperts modules of ``model``: what apply_monolithic gets."""
    for module in model.modules():
        inner = getattr(module, "routed_experts", None)
        if isinstance(inner, torch.nn.Module) and hasattr(inner, "global_num_experts"):
            yield inner


def _op_statics(layer):
    """moe_tp_decode's statics for an armed layer's op (topk, router_bias)."""
    tpd = importlib.import_module(_MOE_TP_DECODE)
    op = getattr(getattr(layer, "quant_method", None), "_moe_tp_decode_op", None)
    if op is None:
        return None
    for (_, statics, _), v in getattr(tpd, "_OPS", {}).items():
        if v is op:
            return statics
    return None


def _placement_order(name: str) -> list[int] | None:
    """The global expert behind each router row of a layer this process
    placed with TPU_EXPERT_PLACEMENT, or None for a linear layer."""
    ep = importlib.import_module("vllm_torchtpu.expert_placement")
    if ep.local_expert_ids(name) is None:
        return None
    return [int(e) for e in ep.virtual_order(ep.placement(), ep.layer_index(name))]


def instrument_models(models, max_decode_query_len: int = 1) -> int:
    """Allocates the counts buffer and gives each MoE layer a slot.

    Args:
      models: [(module or None, "target" | "draft")]; the draft is the
        speculative (MTP) model, whose layers get their own slots.
      max_decode_query_len: 1 + num_speculative_tokens.

    Returns:
      The number of instrumented layers.
    """
    global _STATE
    if _STATE is not None:
        return _STATE.num_slots
    resolve_runtime_modules()
    slots: list[tuple[int, str, str]] = []
    order: dict[int, list[int]] = {}
    num_experts, topk, device = 0, 0, None
    for model, tag in models:
        if model is None:
            continue
        for module in _expert_modules(model):
            name = str(getattr(module, "layer_name", "") or "")
            match = _LAYER_RE.search(name + ".")
            idx = int(match.group(1)) if match else len(slots)
            slot = len(slots)
            module._ep_counter_slot = slot
            statics = _op_statics(module)
            k = getattr(statics, "topk", None) or getattr(
                getattr(module, "moe_config", None), "experts_per_token", 0
            )
            module._ep_counter_topk = int(k)
            module._ep_counter_router_bias = (
                bool(statics.router_bias)
                if statics is not None
                else bool(getattr(_MR, "_ROUTER_TOPK_BIAS", False))
            )
            if tag == "target":
                placed = _placement_order(name)
                if placed is not None:
                    order[slot] = placed
            slots.append((idx, name, tag))
            num_experts = max(num_experts, int(module.global_num_experts))
            topk = max(topk, int(k))
            if device is None:
                device = next(
                    (t.device for t in module.parameters()), torch.device("cpu")
                )
    if not slots:
        _say("found no MoE layers to count; nothing is recorded")
        return 0
    counts = torch.zeros(
        len(slots) * row_len_for(num_experts), dtype=torch.int32, device=device
    )
    _STATE = CountsState(
        counts=counts,
        num_experts=num_experts,
        topk=topk,
        slots=slots,
        max_decode_query_len=int(max_decode_query_len),
        slot_order=order,
    )
    layers = sorted({s[0] for s in slots})
    n_draft = sum(1 for s in slots if s[2] == "draft")
    _say(
        f"counting {len(slots)} MoE layers ({len(slots) - n_draft} target, "
        f"{n_draft} draft; layers {layers[0]}-{layers[-1]}; {len(order)} placed) "
        f"in one int32[{counts.numel()}] buffer on {counts.device}; decode "
        f"request = query length <= {int(max_decode_query_len)}"
    )
    return len(slots)


# --------------------------------------------------------------------------
# Traced helpers: they run inside Dynamo's trace of the forward. Python
# branches are trace-time constants; tensor work goes into the program.
# --------------------------------------------------------------------------


def _padding_mask(num_tokens: int):
    """token_padding's mask rows [0, num_tokens) (True = padded), or None."""
    state = getattr(_TP, "_padding_state", None) if _TP is not None else None
    if state is None:
        return None
    # No comparison on num_tokens: under a dynamic token dimension it would
    # add a shape guard. The mask holds max_num_tokens rows.
    return state.local_padding_mask[:num_tokens]


def _attn_metadata():
    """(request_distribution, query_start_loc) of this forward, or None."""
    fc = _FC
    if fc is None or not fc.is_forward_context_available():
        return None
    md = fc.get_forward_context().attn_metadata
    if isinstance(md, list):
        md = md[0] if md else None
    if not isinstance(md, dict):
        return None
    for value in md.values():
        rd = getattr(value, "request_distribution", None)
        qsl = getattr(value, "query_start_loc", None)
        if (
            isinstance(rd, torch.Tensor)
            and isinstance(qsl, torch.Tensor)
            and rd.dim() == 1
            and rd.shape[0] >= 3
            and qsl.dim() == 1
        ):
            return rd, qsl
    return None


def classify_rows(num_tokens: int, device, meta, pad, max_decode_q: int = 1):
    """Per-row class, the step counter slot and its increment, as tensors.

    Args:
      num_tokens: rows of this forward (static).
      device: device of the index tensors.
      meta: (request_distribution, query_start_loc) or None.
      pad: bool [num_tokens] padding mask (True = padded) or None.
      max_decode_q: a request decodes iff its query length is <= this.

    Returns:
      (cls int64 [num_tokens], step int64 [1], step_add int64 [1],
      mismatch int64 [1]): each row's CLS_*, the STEP_* counter to bump,
      1 if the forward has a valid row (else 0), and the rows where the
      padding mask and the metadata disagree.
    """
    tok = torch.arange(num_tokens, device=device, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64, device=device)
    if meta is None:
        if pad is None:
            cls = torch.full_like(tok, CLS_UNKNOWN)
            has_valid = zero + 1
        else:
            cls = torch.where(pad, CLS_UNKNOWN_PAD, CLS_UNKNOWN).to(torch.int64)
            has_valid = (~pad).any().reshape(1).to(torch.int64)
        return cls, zero + STEP_UNKNOWN, has_valid, zero
    rd, qsl = meta
    qsl = qsl.to(torch.int64)
    num_slots = qsl.shape[0] - 1
    n_req = rd.to(torch.int64)[2:3].clamp(0, num_slots)
    starts, ends = qsl[:-1], qsl[1:]
    qlen = ends - starts
    live = torch.arange(num_slots, device=device, dtype=torch.int64) < n_req
    real = live & (qlen > 0)
    dec_req = real & (qlen <= max_decode_q)
    is_pf = (real & ~dec_req).any().reshape(1).to(torch.int64)
    # Class and step ids below add is_pf, so CLS_DECODE must be 0 and ids consecutive.
    member = (tok[:, None] >= starts[None, :]) & (tok[:, None] < ends[None, :])
    dec_row = (member & dec_req[None, :]).any(dim=1)
    in_req = (member & real[None, :]).any(dim=1)
    # Decode step: every valid row is CLS_DECODE (0). Prefill step: rows of
    # decoding requests are CLS_PREFILL_DECODE, the rest CLS_PREFILL. A row
    # the mask calls valid but no request holds is CLS_UNKNOWN.
    cls_valid = torch.where(
        dec_row, CLS_PREFILL_DECODE * is_pf, is_pf * 0 + CLS_PREFILL
    )
    cls_valid = torch.where(in_req, cls_valid, CLS_UNKNOWN)
    meta_pad = tok >= qsl.gather(0, n_req)
    if pad is None:
        pad, mismatch = meta_pad, zero
    else:
        mismatch = (pad != meta_pad).to(torch.int64).sum().reshape(1)
    cls = torch.where(pad, CLS_DECODE_PAD + is_pf, cls_valid)
    has_valid = (~pad).any().reshape(1).to(torch.int64)
    return cls, STEP_DECODE + is_pf, has_valid, mismatch


def accumulate(layer, topk_ids: torch.Tensor) -> None:
    """Adds one forward's routing of ``layer`` into the counts buffer."""
    st = _STATE
    slot = getattr(layer, "_ep_counter_slot", None)
    if st is None or slot is None:
        return
    num_tokens = topk_ids.shape[0]
    device = topk_ids.device
    cls, step, has_valid, mismatch = classify_rows(
        num_tokens,
        device,
        _attn_metadata(),
        _padding_mask(num_tokens),
        st.max_decode_query_len,
    )
    e = st.num_experts
    base = slot * st.row_len
    ids = topk_ids.to(torch.int64)
    # An id outside [0, E) adds 0 instead of landing in a neighbour's row.
    in_range = ((ids >= 0) & (ids < e)).reshape(-1)
    flat = (base + cls[:, None] * e + ids.clamp(0, e - 1)).reshape(-1)
    stats = base + NUM_CLS * e
    idx = torch.cat([flat, stats + step, stats + STEP_PAD_MISMATCH + 0 * step])
    src = torch.cat(
        [
            in_range.to(torch.int32),
            has_valid.to(torch.int32),
            mismatch.to(torch.int32),
        ]
    )
    st.counts.index_add_(0, idx, src)


def armed_topk_ids(layer, router_logits: torch.Tensor) -> torch.Tensor:
    """The ids moe_tp_decode's op picks from these router logits: the same
    scoring function, f32 bias and selection rule as its routing."""
    scores = _MR._apply_scoring_fn(layer.scoring_func, router_logits)
    bias = getattr(layer, "_moe_tp_decode_bias", None)
    if bias is None:
        bias = layer.e_score_correction_bias.detach().to(torch.float32)
    topk = layer._ep_counter_topk
    if layer._ep_counter_router_bias:
        _, ids = _RT.rowmax_select_bias(scores, bias, topk)
    else:
        _, ids = _RT.rowmax_select(scores + bias.reshape(1, -1), topk)
    return ids


def make_counted_tp_decode_forward(orig):
    """Wraps moe_tp_decode.moe_tp_decode_forward (armed layers)."""

    def counted_moe_tp_decode_forward(quant_method, layer, x, router_logits):
        out = orig(quant_method, layer, x, router_logits)
        if _STATE is not None and getattr(layer, "_ep_counter_slot", None) is not None:
            accumulate(layer, armed_topk_ids(layer, router_logits))
        return out

    counted_moe_tp_decode_forward._ep_counter_orig = orig
    return counted_moe_tp_decode_forward


def make_counted_route(orig):
    """Wraps moe_routing.route (layers moe_tp_decode did not arm)."""

    def counted_route(layer, hidden_states, router_logits, *args, **kwargs):
        topk_weights, topk_ids = orig(
            layer, hidden_states, router_logits, *args, **kwargs
        )
        if _STATE is not None and getattr(layer, "_ep_counter_slot", None) is not None:
            accumulate(layer, topk_ids)
        return topk_weights, topk_ids

    counted_route._ep_counter_orig = orig
    return counted_route


def snapshot_arrays(counts: np.ndarray, state: CountsState) -> dict[str, np.ndarray]:
    """by_class [S, NUM_CLS, E] (global expert ids) and steps [S, NUM_STEP]."""
    e = state.num_experts
    arr = np.asarray(counts, dtype=np.int64).reshape(state.num_slots, state.row_len)
    by_class = arr[:, : NUM_CLS * e].reshape(state.num_slots, NUM_CLS, e).copy()
    for slot, order in state.slot_order.items():
        # Column v of a placed layer counts router row v: global expert order[v].
        placed = by_class[slot].copy()
        by_class[slot][:, np.asarray(order)] = placed
    return {"by_class": by_class, "steps": arr[:, NUM_CLS * e :].copy()}


def _rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except (RuntimeError, ValueError):
        pass
    return int(os.getenv("RANK", "0") or 0)


class Dumper:
    """Copies the buffer to the host and writes this process's dump.

    Dumps happen on the thread that runs the model, between steps (``step``
    wraps ``execute_model``), at most once per interval, so no device copy
    overlaps a step. A background thread adds one dump once the runner has
    been idle for an interval, so the end of a run is recorded too.
    """

    def __init__(self, output_dir: str | None = None, interval_s: float | None = None):
        self.output_dir = output_dir or os.getenv(ENV_DIR, "/tmp/ep_routing_counts")
        self.interval_s = float(
            interval_s if interval_s is not None else os.getenv(ENV_INTERVAL, "60")
        )
        # Reentrant: step and the idle thread call dump while holding it.
        self._lock = threading.RLock()
        self._last_dump = time.monotonic()
        self._last_step = 0.0
        self._dirty = False

    def step(self, fn, *args, **kwargs):
        with self._lock:
            out = fn(*args, **kwargs)
            self._last_step = time.monotonic()
            self._dirty = True
            if self._last_step - self._last_dump >= self.interval_s:
                self.dump()
            return out

    @torch.compiler.disable
    def dump(self) -> str:
        """Writes the cumulative counts; returns the path ("" if none yet)."""
        with self._lock:
            self._last_dump = time.monotonic()
            self._dirty = False
            st = _STATE
            if st is None:
                return ""
            counts = st.counts.detach().cpu().numpy()
            snap = snapshot_arrays(counts, st)
            hits = int(snap["by_class"].sum())
            if hits == 0:
                return ""
            rank = _rank()
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(
                self.output_dir, f"ep_routing_counts_rank{rank}_pid{os.getpid()}.npz"
            )
            payload = {
                "counts_by_class": snap["by_class"],
                "step_counts": snap["steps"],
                "class_names": np.array(CLASS_NAMES),
                "step_names": np.array(STEP_NAMES),
                "slot_layer": np.array([s[0] for s in st.slots], dtype=np.int64),
                "slot_is_draft": np.array([s[2] == "draft" for s in st.slots]),
                "slot_placed": np.array(
                    [i in st.slot_order for i in range(st.num_slots)]
                ),
                "num_experts": np.int64(st.num_experts),
                "topk": np.int64(st.topk),
                "rank": np.int64(rank),
                "pid": np.int64(os.getpid()),
                "unix_time": np.float64(time.time()),
                "format_version": np.int64(FORMAT_VERSION),
            }
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **payload)
            os.replace(tmp, path)
            by = snap["by_class"].sum(axis=(0, 2))
            _say(
                f"rank {rank}: {hits} hits to {path} (decode {by[CLS_DECODE]}, "
                f"prefill {by[CLS_PREFILL] + by[CLS_PREFILL_DECODE]}, padded "
                f"{by[CLS_DECODE_PAD] + by[CLS_PREFILL_PAD] + by[CLS_UNKNOWN_PAD]}"
                f", unknown {by[CLS_UNKNOWN] + by[CLS_UNKNOWN_PAD]})"
            )
            return path

    def _idle_loop(self) -> None:
        while True:
            time.sleep(self.interval_s)
            try:
                with self._lock:
                    idle = time.monotonic() - self._last_step >= self.interval_s
                    if self._dirty and idle:
                        self.dump()
            except Exception as e:  # keep the serving process alive
                _say(f"idle dump failed: {e!r}")

    def start(self) -> None:
        """The idle dump thread, plus one dump at exit."""
        threading.Thread(target=self._idle_loop, daemon=True).start()
        atexit.register(self.dump)


_DUMPER: Dumper | None = None


def _start_dumps() -> Dumper:
    global _DUMPER
    if _DUMPER is None:
        _DUMPER = Dumper()
        _DUMPER.start()
    return _DUMPER


# --------------------------------------------------------------------------
# Installation (sitecustomize wraps load_model, which installs the rest).
# --------------------------------------------------------------------------


def install_tp_decode_patch(mod=None) -> bool:
    """Wraps moe_tp_decode_forward. apply_monolithic imports it at call time,
    so patching the module attribute before the first trace is enough."""
    mod = mod or importlib.import_module(_MOE_TP_DECODE)
    fwd = getattr(mod, "moe_tp_decode_forward", None)
    if fwd is None:
        _say("moe_tp_decode has no moe_tp_decode_forward; armed layers not counted")
        return False
    if not hasattr(fwd, "_ep_counter_orig"):
        mod.moe_tp_decode_forward = make_counted_tp_decode_forward(fwd)
        _say("wrapped moe_tp_decode.moe_tp_decode_forward")
    return True


def install_route_patch(mod=None) -> bool:
    """Wraps moe_routing.route (unarmed layers)."""
    mod = mod or importlib.import_module(_MOE_ROUTING)
    if not hasattr(mod, "route"):
        _say("moe_routing has no route; unarmed layers not counted")
        return False
    if not hasattr(mod.route, "_ep_counter_orig"):
        mod.route = make_counted_route(mod.route)
        _say("wrapped moe_routing.route")
    return True


def _draft_model(runner):
    if getattr(runner, "_is_async_drafter", False):
        drafter = getattr(runner, "_async_drafter", None)
        return getattr(drafter, "draft_model", None)
    return None


def _warn_untracked_paths() -> None:
    for env in ("USE_MOE_FUSED_EP_KERNEL", "MOE_FUSED_EP_KERNEL_IMPL"):
        if os.getenv(env, "").strip().lower() not in ("", "0", "false", "no"):
            _say(
                f"WARNING {env}={os.getenv(env)!r}: the fused EP kernel routes "
                "inside its own program; its layers are not counted"
            )


def install_runner_patch(mod) -> bool:
    """Wraps TPUModelRunner.load_model: after the model (and moe_tp_decode's
    arming) is loaded, allocate the buffer and wrap the routing entry points,
    before anything is traced."""
    cls = getattr(mod, "TPUModelRunner", None)
    orig = getattr(cls, "load_model", None)
    if orig is None:
        _say("TPUModelRunner.load_model not found; nothing is counted")
        return False
    if hasattr(orig, "_ep_counter_orig"):
        return True

    def load_model(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        pc = getattr(self, "parallel_config", None)
        dp = int(getattr(pc, "data_parallel_size", 1) or 1)
        pcp = int(getattr(pc, "prefill_context_parallel_size", 1) or 1)
        if dp > 1 or pcp > 1:
            # The rows a layer routes would no longer be the rows the
            # attention metadata and the padding mask describe.
            _say(
                f"NOT counting: data_parallel_size={dp}, "
                f"prefill_context_parallel_size={pcp}; the counter needs both 1"
            )
            return result
        install_tp_decode_patch()
        install_route_patch()
        spec = getattr(self, "speculative_config", None)
        k = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0
        instrument_models(
            [(getattr(self, "model", None), "target"), (_draft_model(self), "draft")],
            max_decode_query_len=1 + k,
        )
        _warn_untracked_paths()
        _wrap_execute_model(cls, _start_dumps())
        return result

    load_model._ep_counter_orig = orig
    cls.load_model = load_model
    _say("wrapped TPUModelRunner.load_model")
    return True


def _wrap_execute_model(cls, dumper: Dumper) -> None:
    """Dumps between steps: every ``execute_model`` call goes through
    ``Dumper.step``."""
    orig = getattr(cls, "execute_model", None)
    if orig is None or hasattr(orig, "_ep_counter_orig"):
        return

    def execute_model(self, *args, **kwargs):
        return dumper.step(orig, self, *args, **kwargs)

    execute_model._ep_counter_orig = orig
    cls.execute_model = execute_model
