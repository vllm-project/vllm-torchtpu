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
"""Opt-in per-step MoE routing trace (``TPU_MOE_ROUTE_TRACE=<steps>``).

Off (the default) ``register`` and ``record`` return at once and the compiled
programs are the ones built without this module.

On, every target-model MoE layer writes the per-expert routed-row counts of
the current forward step (all rows, padded ones included: the grouped matmul
computes them too) into slot ``step % N`` of one device ring buffer
``[layers, N, E]``. The first registered layer advances a step counter and
stores the step's token count. TP rank 0 copies the ring to the host every
``TPU_MOE_ROUTE_TRACE_EVERY`` model-runner steps and logs the steps written
since the previous copy, one ``[moe-route-trace]`` line per step:

    [moe-route-trace] s=<step> T=<tokens> r=<e.c,e.c,...>|<e.c,...>|...

with one ``|``-separated field per MoE layer listing ``expert.count`` for each
expert hit. These lines let an offline simulator price expert placements
(``python -m vllm_torchtpu.layers.adapter.moe_route_trace LOG``).

Routing is computed redundantly on every TP rank from the same hidden states,
so rank 0's ids are every rank's ids.
"""

from __future__ import annotations

import sys
from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

RING: int = int(envs.TPU_MOE_ROUTE_TRACE)
ENABLED: bool = RING > 0
EVERY: int = max(1, int(envs.TPU_MOE_ROUTE_TRACE_EVERY))
MAX_ROWS = 128

_ring: torch.Tensor | None = None
_ctr: torch.Tensor | None = None
_tok: torch.Tensor | None = None
_num_experts: int | None = None
_names: list[str] = []
_ep_size = 1
_calls = 0
_last_step = 0
_hooked = False


def _is_target_layer(name: str) -> bool:
    return name.startswith(("model.layers.", "language_model.model.layers.")) or (
        ".layers." in name and "draft" not in name and "dspark" not in name
    )


def register(layer: Any) -> None:
    """Give ``layer`` a ring row. Call from ``process_weights_after_loading``."""
    global _ring, _ctr, _tok, _num_experts, _ep_size
    if not ENABLED or getattr(layer, "_route_trace_row", None) is not None:
        return
    name = str(getattr(layer, "layer_name", "") or "")
    if not _is_target_layer(name):
        return
    num_experts = int(layer.global_num_experts)
    if _num_experts is None:
        _num_experts = num_experts
        device = layer.w13_weight.device
        _ring = torch.zeros(
            (MAX_ROWS, RING, num_experts), dtype=torch.int32, device=device
        )
        _ctr = torch.zeros((1,), dtype=torch.int32, device=device)
        _tok = torch.zeros((RING,), dtype=torch.int32, device=device)
        moe_config = getattr(layer, "moe_config", None)
        if moe_config is not None and moe_config.moe_parallel_config.use_ep:
            _ep_size = int(moe_config.ep_size)
        _install_runner_hook()
    if num_experts != _num_experts or len(_names) >= MAX_ROWS:
        logger.warning_once("[moe-route-trace] skipping layer %s", name)
        return
    layer._route_trace_row = len(_names)
    _names.append(name)
    if len(_names) == 1:
        logger.info(
            "[moe-route-trace] TPU_MOE_ROUTE_TRACE=%d on: E=%d ep=%d every=%d",
            RING,
            num_experts,
            _ep_size,
            EVERY,
        )


def record(layer: Any, topk_ids: torch.Tensor) -> None:
    """Store this step's per-expert row counts in the ring (traced code)."""
    row = getattr(layer, "_route_trace_row", None)
    if row is None or _ring is None:
        return
    assert _ctr is not None and _tok is not None and _num_experts is not None
    ids = topk_ids.flatten().to(torch.int32)
    counts = torch.zeros(_num_experts, dtype=torch.int32, device=ids.device)
    counts = counts.index_add(0, ids.to(torch.int64), torch.ones_like(ids))
    if row == 0:
        _ctr.add_(1)
    slot = torch.remainder(_ctr - 1, RING).to(torch.int64)
    if row == 0:
        tokens = torch.full(
            (1,), topk_ids.shape[0], dtype=torch.int32, device=ids.device
        )
        _tok.index_copy_(0, slot, tokens)
    _ring[row].index_copy_(0, slot, counts.view(1, _num_experts))


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def dump() -> None:
    """Log the ring's steps written since the last dump (TP rank 0 only)."""
    global _last_step
    if _ring is None or _ctr is None or _tok is None or not _names:
        return
    step = int(_ctr.cpu().item())
    new = min(step - _last_step, RING)
    if new <= 0 or _tp_rank() != 0:
        _last_step = step
        return
    ring = _ring.cpu().numpy()
    tok = _tok.cpu().numpy()
    n = len(_names)
    for s in range(step - new, step):
        slot = s % RING
        fields = []
        for r in range(n):
            row = ring[r, slot]
            hit = row.nonzero()[0]
            fields.append(",".join(f"{e}.{row[e]}" for e in hit))
        logger.info(
            "[moe-route-trace] s=%d T=%d r=%s", s, int(tok[slot]), "|".join(fields)
        )
    _last_step = step


def _install_runner_hook() -> None:
    global _hooked
    if _hooked:
        return
    from vllm_torchtpu.runner import tpu_runner

    cls = tpu_runner.TPUModelRunner
    orig = cls.execute_model

    def execute_model(self, *args, **kwargs):
        global _calls
        out = orig(self, *args, **kwargs)
        _calls += 1
        if _calls % EVERY == 0:
            try:
                dump()
            except Exception as exc:  # diagnostics must never kill serving
                logger.warning_once("[moe-route-trace] dump failed: %s", exc)
        return out

    cls.execute_model = execute_model
    _hooked = True


# ---------------------------------------------------------------------------
# Offline parser
# ---------------------------------------------------------------------------


def parse_line(line: str) -> tuple[int, int, list[dict[int, int]]] | None:
    """``(step, tokens, [ {expert: rows} per layer ])`` of one trace line."""
    i = line.find("[moe-route-trace] s=")
    if i < 0:
        return None
    parts = line[i:].split()
    try:
        step = int(parts[1][2:])
        tokens = int(parts[2][2:])
        body = parts[3][2:]
    except (IndexError, ValueError):
        return None
    layers = []
    for field in body.split("|"):
        d: dict[int, int] = {}
        for item in field.split(","):
            if item:
                e, _, c = item.partition(".")
                d[int(e)] = int(c)
        layers.append(d)
    return step, tokens, layers


if __name__ == "__main__":  # pragma: no cover
    seen = 0
    for path in sys.argv[1:]:
        with open(path, errors="replace") as f:
            for line in f:
                if parse_line(line) is not None:
                    seen += 1
    print(f"{seen} trace steps")
