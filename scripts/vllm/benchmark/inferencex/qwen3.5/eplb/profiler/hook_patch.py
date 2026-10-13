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
"""Compiled-mode MoE routing profiler hook.

Records, per MoE layer, routing histograms for every forward and, for
forwards of at most 128 rows (decode), per-step co-activation statistics and a
ring of sampled per-step hit sets (see _accumulate_decode_step). The decode
statistics are what solve_placement.py optimises against.

Design constraints:

  1. vLLM's TPU backend compiles the model forward as a single full graph.
     Any TorchDynamo graph break inside the forward path (a Python lock, a
     @torch.compiler.disable helper, a module mutation) aborts engine startup.
     The forward hook therefore performs pure ATen tensor work and nothing else.

  2. --enforce-eager is not a usable escape hatch for DeepSeek-V4-Pro. In eager
     mode the token-id MoE hash table lookup (s32[129280, 6]) is emitted as a
     standalone gather, the compiler offloads it to SparseCore, and it trips a
     layout RET_CHECK (E(32) packing, xid/289448467) that kills every worker.

  3. Consequently the accumulator buffers must already exist before compilation
     starts. They are attached at model load time by wrapping
     TPUModelRunner.load_model, so the forward hook only ever reads an
     attribute that is guaranteed to be present. Wrapping vLLM's generic
     get_model does NOT work here: the torchtpu runner calls
     get_model_loader(...).load_model(...) directly and never touches it.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import logging
import os
import re
import signal
import sys

import torch

logger = logging.getLogger(__name__)

_DEFAULT_NUM_EXPERTS = 512


_MOE_CLASS_RE = re.compile(r"(?i)(routedexperts|fusedmoe|moeblock|sparsemoe|\bmoe\b)")


def _resolve_num_experts(module) -> int | None:
    """Returns the logical routed expert count for a module, or None."""
    for attr in (
        "global_num_experts",
        "num_experts",
        "num_routed_experts",
        "n_routed_experts",
        "num_total_experts",
    ):
        value = getattr(module, attr, None)
        if isinstance(value, int) and value > 1:
            return value
    for cfg_attr in ("moe_config", "config", "moe_parallel_config"):
        cfg = getattr(module, cfg_attr, None)
        if cfg is None:
            continue
        for attr in ("num_total_experts", "num_experts", "n_routed_experts"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 1:
                return value
    # Last resort for MoE-looking modules that expose no scalar count: the
    # routed-expert dimension is the leading dim of the fused expert weights.
    if _MOE_CLASS_RE.search(type(module).__name__):
        for attr in ("w13_weight", "w2_weight", "w1_weight", "gate_up_proj"):
            tensor = getattr(module, attr, None)
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.dim() >= 2
                and tensor.shape[0] > 1
            ):
                return int(tensor.shape[0])
        return _DEFAULT_NUM_EXPERTS
    return None


def _resolve_device(module) -> torch.device:
    """Returns the device a module's tensors live on."""
    for tensor in module.parameters(recurse=True):
        return tensor.device
    for tensor in module.buffers(recurse=True):
        return tensor.device
    return torch.device("tpu")


def _layer_index_from_name(name: str, fallback: int) -> int:
    match = re.search(r"layers?\.(\d+)", name)
    return int(match.group(1)) if match else fallback


def instrument_model(model, profiler) -> int:
    """Attaches on-device accumulator buffers to every MoE routing module.

    Runs once, after weights are loaded and before TorchDynamo traces anything,
    so the forward hook never has to allocate or register.

    Args:
      model: The fully constructed model.
      profiler: The MoeProfiler singleton that will read the buffers back.

    Returns:
      Number of instrumented MoE modules.
    """
    instrumented = 0
    for name, module in model.named_modules():
        if "mtp" in name.split("."):
            continue
        if hasattr(module, "_eplb_hist_total"):
            continue
        num_experts = _resolve_num_experts(module)
        if num_experts is None:
            continue

        device = _resolve_device(module)
        for buffer_name in (
            "_eplb_hist_total",
            "_eplb_hist_prefill",
            "_eplb_hist_decode",
            "_eplb_hist_valid",
            "_eplb_hist_padded",
        ):
            module.register_buffer(
                buffer_name,
                torch.zeros(num_experts, dtype=torch.int32, device=device),
                persistent=False,
            )

        _register_decode_buffers(module, num_experts, device)

        layer_idx = getattr(module, "layer_idx", None)
        if not isinstance(layer_idx, int):
            layer_idx = _layer_index_from_name(name, instrumented)
        profiler.register_layer(layer_idx, module, num_experts)
        instrumented += 1

    print(
        f"[EPLB Hook] Instrumented {instrumented} MoE modules with on-device"
        f" accumulators in PID {os.getpid()}",
        flush=True,
    )

    if instrumented == 0:
        # Nothing matched. Emit a census of plausible MoE modules so the correct
        # attribute or class name can be picked without another multi-hour run.
        census = {}
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if _MOE_CLASS_RE.search(cls_name) or _MOE_CLASS_RE.search(name):
                census.setdefault(cls_name, name)
        print(
            "[EPLB Hook] DIAGNOSTIC: zero modules matched. MoE-like candidates"
            f" (class -> first module path): {census}",
            flush=True,
        )

    return instrumented


# Decode-step sampling: every DECODE_RING_STRIDE-th decode step's hit set is
# kept in a ring of DECODE_RING_SLOTS rows per layer, so placements can be
# replayed against real per-step routing (with its correlations).
DECODE_RING_SLOTS = int(os.getenv("VLLM_TPU_MOE_PROFILE_RING_SLOTS", "4096"))
DECODE_RING_STRIDE = int(os.getenv("VLLM_TPU_MOE_PROFILE_RING_STRIDE", "4"))
DECODE_MAX_ROWS = 128
DECODE_BUFFERS = (
    "_eplb_dec_steps",  # [1] decode steps with at least one valid row
    "_eplb_dec_step_hit",  # [E] steps in which the expert was hit
    "_eplb_dec_cohit",  # [E, E] steps in which both experts were hit
    "_eplb_dec_tok_coact",  # [E, E] valid rows that picked both experts
    "_eplb_dec_padhit",  # [E] steps in which only padded rows hit the expert
    "_eplb_dec_rows_hist",  # [DECODE_MAX_ROWS + 1] valid rows per step
    "_eplb_dec_pad_rows",  # [1] padded rows over all decode steps
    "_eplb_dec_ring",  # [SLOTS, E] uint8 sampled step hit sets
    "_eplb_dec_ring_rows",  # [SLOTS] valid rows of each sampled step
)


def _register_decode_buffers(module, num_experts: int, device) -> None:
    e = num_experts
    shapes = {
        "_eplb_dec_steps": ((1,), torch.int32),
        "_eplb_dec_step_hit": ((e,), torch.int32),
        "_eplb_dec_cohit": ((e, e), torch.int32),
        "_eplb_dec_tok_coact": ((e, e), torch.int32),
        "_eplb_dec_padhit": ((e,), torch.int32),
        "_eplb_dec_rows_hist": ((DECODE_MAX_ROWS + 1,), torch.int32),
        "_eplb_dec_pad_rows": ((1,), torch.int32),
        "_eplb_dec_ring": ((DECODE_RING_SLOTS, e), torch.uint8),
        "_eplb_dec_ring_rows": ((DECODE_RING_SLOTS,), torch.int32),
    }
    for name, (shape, dtype) in shapes.items():
        module.register_buffer(
            name, torch.zeros(shape, dtype=dtype, device=device), persistent=False
        )


def _touch_decode_buffers(layer) -> None:
    for name in DECODE_BUFFERS:
        buf = getattr(layer, name, None)
        if buf is not None:
            buf.add_(0)


def _accumulate_decode_step(layer, topk_ids: torch.Tensor, valid: torch.Tensor) -> None:
    """Per-step co-activation statistics for a few-row (decode) forward.

    Pure ATen, no host sync. `valid` is a [rows] 0/1 int32 mask.
    """
    ring = getattr(layer, "_eplb_dec_ring", None)
    if ring is None:
        return
    rows, _ = topk_ids.shape
    n_exp = layer._eplb_dec_step_hit.shape[0]
    ids = topk_ids.long()
    multi = torch.zeros(rows, n_exp, dtype=torch.float32, device=ids.device)
    multi = multi.scatter(1, ids, 1.0)  # [rows, E]
    vmask = valid.to(torch.float32).reshape(rows, 1)
    m_valid = multi * vmask
    hit = (m_valid.sum(0) > 0).to(torch.float32)  # [E]
    hit_all = (multi.sum(0) > 0).to(torch.float32)
    n_valid = valid.to(torch.int32).sum().reshape(1).to(torch.int32)
    any_valid = (n_valid > 0).to(torch.int32)

    layer._eplb_dec_steps.add_(any_valid)
    layer._eplb_dec_step_hit.add_(hit.to(torch.int32))
    layer._eplb_dec_cohit.add_(torch.outer(hit, hit).to(torch.int32))
    layer._eplb_dec_tok_coact.add_((m_valid.t() @ m_valid).to(torch.int32))
    layer._eplb_dec_padhit.add_(((hit_all - hit) > 0).to(torch.int32))
    layer._eplb_dec_rows_hist.index_add_(
        0, n_valid.long(), torch.ones(1, dtype=torch.int32, device=ids.device)
    )
    layer._eplb_dec_pad_rows.add_((rows - n_valid).to(torch.int32))

    # Ring sample: the step counter after this step picks the slot.
    count = layer._eplb_dec_steps
    take = (any_valid > 0) & (torch.remainder(count, DECODE_RING_STRIDE) == 0)
    slot = torch.remainder(
        torch.div(count, DECODE_RING_STRIDE, rounding_mode="floor"),
        DECODE_RING_SLOTS,
    ).long()
    old = ring.index_select(0, slot)  # [1, E]
    new = torch.where(take.reshape(1, 1), hit.to(torch.uint8).reshape(1, -1), old)
    ring.index_copy_(0, slot, new)
    old_rows = layer._eplb_dec_ring_rows.index_select(0, slot)
    layer._eplb_dec_ring_rows.index_copy_(0, slot, torch.where(take, n_valid, old_rows))


# Padding-mask helpers, resolved once after load_model (see
# _resolve_padding_helpers). The forward hook reads these at trace time.
_PADDING = {"token_padding": None, "is_local_tensor": None}


def _resolve_padding_helpers(profiler) -> bool:
    """Hooks into vllm-torchtpu's token padding mask, if the runner created one.

    With TPU_MOE_SKIP_PADDED_TOKENS=1, TPUModelRunner.load_model creates a
    TokenPaddingState and marks padded rows every step (every row of a warmup
    step is padding). The forward hook then reuses
    token_padding.zero_routing_weights_for_padding, the same helper the MoE
    uses, so valid and padded rows are split with exactly the runner's mask.

    Also wraps TokenPaddingState.update so the profiler can count rows on the
    host and snapshot the histograms when the first real step starts.

    Args:
      profiler: The MoeProfiler singleton.

    Returns:
      True if the padding mask is active.
    """
    try:
        token_padding = importlib.import_module(
            "vllm_torchtpu.layers.adapter.token_padding"
        )
        pipelined = importlib.import_module(
            "vllm_torchtpu.layers.adapter.pipelined_fused_moe"
        )
        is_local_tensor = pipelined.enable_pipelined_collective_and_compute
    except (ImportError, AttributeError) as e:
        print(f"[EPLB Hook] padding mask unavailable: {e}", flush=True)
        return False

    if getattr(token_padding, "_padding_state", None) is None:
        print(
            "[EPLB Hook] WARNING: no token padding mask (set"
            " TPU_MOE_SKIP_PADDED_TOKENS=1). Valid/padded histograms stay zero;"
            " only total/prefill/decode are recorded.",
            flush=True,
        )
        return False

    state_cls = token_padding.TokenPaddingState
    orig_update = state_cls.update
    if not getattr(orig_update, "_is_eplb_counted", False):

        def counted_update(self, local_valid_tokens, local_padded_tokens):
            profiler.note_padding_update(
                int(local_valid_tokens), int(local_padded_tokens)
            )
            return orig_update(self, local_valid_tokens, local_padded_tokens)

        counted_update._is_eplb_counted = True
        state_cls.update = counted_update

    _PADDING["token_padding"] = token_padding
    _PADDING["is_local_tensor"] = is_local_tensor
    profiler.padding_mask_active = True
    print(
        "[EPLB Hook] Token padding mask active: valid and padded rows are"
        f" counted separately in PID {os.getpid()}",
        flush=True,
    )
    return True


def _install_model_load_hook(profiler) -> bool:
    """Wraps TPUModelRunner.load_model so buffers exist before compilation."""
    try:
        runner_mod = importlib.import_module("vllm_torchtpu.runner.tpu_runner")
    except ImportError as e:
        print(
            f"[EPLB Hook] FATAL: cannot import vllm_torchtpu.runner.tpu_runner: {e}",
            flush=True,
        )
        return False

    runner_cls = getattr(runner_mod, "TPUModelRunner", None)
    if runner_cls is None:
        print("[EPLB Hook] FATAL: tpu_runner has no TPUModelRunner", flush=True)
        return False

    orig_load_model = getattr(runner_cls, "load_model", None)
    if orig_load_model is None:
        print("[EPLB Hook] FATAL: TPUModelRunner has no load_model", flush=True)
        return False
    if getattr(orig_load_model, "_is_eplb_instrumented", False):
        return True

    def instrumented_load_model(self, *args, **kwargs):
        result = orig_load_model(self, *args, **kwargs)
        try:
            model = getattr(self, "model", None)
            if model is None:
                print(
                    "[EPLB Hook] FATAL: runner.model is None after load_model",
                    flush=True,
                )
            else:
                instrument_model(model, profiler)
                _resolve_padding_helpers(profiler)
                _patch_lowtok_apply()
        except Exception as ex:  # pylint: disable=broad-exception-caught
            print(f"[EPLB Hook] instrument_model failed: {ex}", flush=True)
        return result

    instrumented_load_model._is_eplb_instrumented = True
    runner_cls.load_model = instrumented_load_model
    print(
        "[EPLB Hook] Wrapped TPUModelRunner.load_model for load-time"
        f" instrumentation in PID {os.getpid()}",
        flush=True,
    )
    return True


def _accumulate_topk_ids(layer, topk_ids: torch.Tensor, num_rows: int) -> None:
    """Pure ATen in-graph accumulation of topk_ids into layer's _eplb_hist_* buffers."""
    hist_total = getattr(layer, "_eplb_hist_total", None)
    if hist_total is None:
        return
    flat_ids = topk_ids.reshape(-1).long()
    ones = torch.ones_like(flat_ids, dtype=torch.int32)
    hist_total.index_add_(0, flat_ids, ones)
    # The runner traces the backbone once per token-count side and hands buckets
    # between traces only if both lift the same buffers, so each side also
    # touches the other side's buffers with a no-op add.
    if num_rows > 128:
        layer._eplb_hist_prefill.index_add_(0, flat_ids, ones)
        layer._eplb_hist_decode.add_(0)
        _touch_decode_buffers(layer)
    else:
        layer._eplb_hist_decode.index_add_(0, flat_ids, ones)
        layer._eplb_hist_prefill.add_(0)
        _accumulate_decode_step(layer, topk_ids, _row_valid(topk_ids))

    token_padding = _PADDING["token_padding"]
    if token_padding is not None:
        _, row_weight = token_padding.zero_routing_weights_for_padding(
            topk_ids,
            torch.ones_like(topk_ids, dtype=torch.float32),
            is_local_tensor=_PADDING["is_local_tensor"](),
        )
        valid = (row_weight > 0).reshape(-1).to(torch.int32)
        layer._eplb_hist_valid.index_add_(0, flat_ids, valid)
        layer._eplb_hist_padded.index_add_(0, flat_ids, ones - valid)


def _row_valid(topk_ids: torch.Tensor) -> torch.Tensor:
    """[rows] int32 1 for real rows, 0 for padding (all ones without a mask)."""
    token_padding = _PADDING["token_padding"]
    if token_padding is None:
        return torch.ones(topk_ids.shape[0], dtype=torch.int32, device=topk_ids.device)
    _, row_weight = token_padding.zero_routing_weights_for_padding(
        topk_ids,
        torch.ones_like(topk_ids, dtype=torch.float32),
        is_local_tensor=_PADDING["is_local_tensor"](),
    )
    return (row_weight.reshape(topk_ids.shape[0], -1)[:, 0] > 0).to(torch.int32)


def _patch_lowtok_apply() -> bool:
    try:
        moe_lowtok_scores = importlib.import_module(
            "vllm_torchtpu.layers.adapter.quantization.moe_lowtok_scores"
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[EPLB Hook] moe_lowtok_scores not importable yet: {e!r}", flush=True)
        return False
    orig = getattr(moe_lowtok_scores, "apply", None)
    if orig is None:
        print("[EPLB Hook] moe_lowtok_scores has no apply", flush=True)
        return False
    if getattr(orig, "_is_eplb_profiled", False):
        return True

    def profiled_lowtok_apply(op, layer, x, router_logits):
        if getattr(layer, "_eplb_hist_total", None) is not None:
            topk = int(layer.moe_config.experts_per_token)
            topk_ids = torch.topk(router_logits, k=topk, dim=-1).indices.to(torch.int32)
            _accumulate_topk_ids(layer, topk_ids, int(router_logits.shape[0]))
        return orig(op, layer, x, router_logits)

    profiled_lowtok_apply._is_eplb_profiled = True
    moe_lowtok_scores.apply = profiled_lowtok_apply
    print(
        "[EPLB Hook] Patched moe_lowtok_scores.apply with in-graph accumulator"
        f" in PID {os.getpid()}",
        flush=True,
    )
    return True


def apply_runtime_hook() -> bool:
    """Installs the compiled-mode MoE routing profiler."""
    moe_routing = None
    import_errors = []
    for module_path in (
        "vllm_torchtpu.layers.adapter.moe_routing",
        "vllm_torchtpu.layers.vllm.moe_routing",
    ):
        try:
            moe_routing = importlib.import_module(module_path)
            print(f"[EPLB Hook] Using {module_path} in PID {os.getpid()}", flush=True)
            break
        except ImportError as e:
            import_errors.append(f"{module_path}: {e}")

    if moe_routing is None:
        print(
            "[EPLB Hook] FATAL: could not import moe_routing in PID"
            f" {os.getpid()}. Tried: {import_errors}",
            flush=True,
        )
        return False

    if not hasattr(moe_routing, "route"):
        print(
            f"[EPLB Hook] FATAL: moe_routing has no 'route' in PID {os.getpid()}",
            flush=True,
        )
        return False

    try:
        from moe_profiler import get_moe_profiler, is_profiling_enabled
    except ImportError:
        try:
            from vllm_torchtpu.layers.vllm.moe_profiler import (
                get_moe_profiler,
                is_profiling_enabled,
            )
        except ImportError as e:
            print(
                "[EPLB Hook] FATAL: could not import moe_profiler in PID"
                f" {os.getpid()}: {e}",
                flush=True,
            )
            return False

    if not is_profiling_enabled():
        logger.info("MoE Profiler disabled (VLLM_TPU_ENABLE_MOE_PROFILER unset).")
        return False

    orig_route = moe_routing.route
    if getattr(orig_route, "_is_eplb_profiled", False):
        logger.info("moe_routing.route is already hooked.")
        return True

    profiler = get_moe_profiler()

    if not _install_model_load_hook(profiler):
        return False

    def profiled_route(layer, hidden_states, router_logits, input_ids=None):
        topk_weights, topk_ids = orig_route(
            layer, hidden_states, router_logits, input_ids=input_ids
        )
        _accumulate_topk_ids(layer, topk_ids, int(router_logits.shape[0]))
        return topk_weights, topk_ids

    profiled_route._is_eplb_profiled = True
    moe_routing.route = profiled_route
    print(
        "[EPLB Hook] Patched moe_routing.route with in-graph accumulator in PID"
        f" {os.getpid()}",
        flush=True,
    )

    # moe_lowtok_scores.apply serves every forward of at most 128 rows under
    # TPU_MOE_LOWTOK=1 and bypasses moe_routing.route. Importing it here (at
    # interpreter start) fails silently, so it is patched after load_model
    # (_patch_lowtok_apply), when vllm_torchtpu is fully imported.
    _patch_lowtok_apply()

    atexit.register(lambda: profiler.dump(step_suffix="shutdown"))

    def _sigterm_handler(signum, frame):
        del signum, frame
        logger.info("Caught termination signal; dumping MoE profiler trace...")
        profiler.dump(step_suffix="sigterm")
        sys.exit(0)

    with contextlib.suppress(ValueError, AttributeError):
        signal.signal(signal.SIGTERM, _sigterm_handler)

    return True
