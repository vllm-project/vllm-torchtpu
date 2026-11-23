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
"""TP-sliced decode experts beside the EP layout (NVFP4 MoE layers).

``TPU_MOE_TP_DENSE_FP4_MAX_TOKENS=<n>`` (default 0 = off). Under EP a decode
step's routed experts cost the busiest rank's distinct experts, with whole
experts as the balance unit. With this on, every rank also holds a 1/ep slice
of the intermediate dim of EVERY expert (gate slice | up slice of w13, the
matching rows of w2), and steps of up to ``n`` tokens run all their routed
experts on every rank through ``kernels/experimental/moe_tp_dense_fp4``: each
rank streams 1/ep of every hit expert, so the weight reads are balanced by
construction. Each rank's output is a partial sum over its slice, the same
contract as the EP path's partial sum over its experts, so the layer's existing
all-reduce stays. Prefill (more than ``n`` tokens) keeps the EP path.

The slices come straight from the checkpoint: the EP weight filter is widened
(as redundant experts do), each expert parameter's loader also copies this
rank's rows of every expert into raw buffers, and after loading the raw slices
go through the same scale fusion and block requantization as the EP weights.
Slices align with the 128-row requant blocks, so the TP copy holds exactly the
EP weights' values.

The decision between the two paths is static (token count of the traced
bucket) and made inside one custom op that takes both weight sets, so every
bucket's graph has the same inputs.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MAX_TOKENS: int = int(envs.TPU_MOE_TP_DENSE_FP4_MAX_TOKENS)
ENABLED: bool = MAX_TOKENS > 0
QB = 128
ROW_ALIGN = 8

_RAW = ("w13_weight", "w13_weight_scale", "w13_weight_scale_2",
        "w2_weight", "w2_weight_scale", "w2_weight_scale_2")
_STATS = {"layers": 0, "bytes": 0}


def prepare(layer: Any, hidden_size: int, intermediate_size: int) -> None:
    """From create_weights: decide whether this layer gets a TP copy."""
    if not ENABLED:
        return
    from vllm_torchtpu.layers.adapter import moe_redundant

    use_ep, ep, rank = moe_redundant._layer_ep(layer)
    reason = None
    if not use_ep or ep < 2:
        reason = "expert parallelism is off"
    elif intermediate_size % (ep * QB):
        reason = f"intermediate {intermediate_size} does not split into {ep} x {QB}-row blocks"
    elif hidden_size % QB:
        reason = f"hidden {hidden_size} is not a multiple of {QB}"
    elif moe_redundant._model_tag() != "backbone":
        reason = f"graph {moe_redundant._model_tag()!r} is traced once for every bucket"
    if reason is not None:
        logger.warning_once("[moe-tp-dense] TPU_MOE_TP_DENSE_FP4_MAX_TOKENS ignored: %s.", reason)
        return
    moe_redundant._widen_ep_weight_filter()
    layer._tpd = {
        "ep": ep,
        "rank": rank,
        "E": int(layer.global_num_experts),
        "H": int(hidden_size),
        "I": int(intermediate_size),
        "raw": {},
        "seen": {n: set() for n in _RAW},
    }


def _raw_buffer(st: dict, name: str, shape: tuple, dtype, device) -> torch.Tensor:
    buf = st["raw"].get(name)
    if buf is None:
        buf = torch.zeros(shape, dtype=dtype, device=device)
        st["raw"][name] = buf
    return buf


def _capture(st: dict, pname: str, param: torch.Tensor, lw: torch.Tensor,
             shard_id: str, eid: int) -> None:
    """Copy this rank's TP rows of checkpoint tensor `lw` (one expert, one shard)."""
    ep, c, E, I = st["ep"], st["rank"], st["E"], st["I"]
    if not 0 <= eid < E:
        return
    i_s = I // ep
    dev = torch.device("cpu")
    if pname in ("w13_weight", "w13_weight_scale"):
        if shard_id not in ("w1", "w3"):
            return
        # lw [I, H / pack] (packed weight) or [I, H / group] (block scale).
        rows = lw[c * i_s:(c + 1) * i_s]
        buf = _raw_buffer(st, pname, (E, 2 * i_s, lw.shape[-1]), lw.dtype, dev)
        off = 0 if shard_id == "w1" else i_s
        buf[eid, off:off + i_s].copy_(rows.to(dev))
        st["seen"][pname].add((eid, shard_id))
    elif pname in ("w2_weight", "w2_weight_scale"):
        if shard_id != "w2":
            return
        # lw [H, I / pack] or [H, I / group]: contracting dim last.
        per = lw.shape[-1] // ep
        cols = lw[:, c * per:(c + 1) * per]
        buf = _raw_buffer(st, pname, (E, lw.shape[0], per), lw.dtype, dev)
        buf[eid].copy_(cols.to(dev))
        st["seen"][pname].add((eid, shard_id))
    elif pname == "w13_weight_scale_2":
        if shard_id not in ("w1", "w3"):
            return
        buf = _raw_buffer(st, pname, (E, 2), torch.float32, dev)
        buf[eid, 0 if shard_id == "w1" else 1].copy_(lw.reshape(-1)[0].to(torch.float32).to(dev))
        st["seen"][pname].add((eid, shard_id))
    elif pname == "w2_weight_scale_2":
        if shard_id != "w2":
            return
        buf = _raw_buffer(st, pname, (E,), torch.float32, dev)
        buf[eid].copy_(lw.reshape(-1)[0].to(torch.float32).to(dev))
        st["seen"][pname].add((eid, shard_id))
    want = 2 * E if pname.startswith("w13") else E
    if len(st["seen"][pname]) == want and pname in st["raw"] and st["raw"][pname].device.type == "cpu":
        # One host-to-device copy per tensor once every expert's slice is in.
        st["raw"][pname] = st["raw"][pname].to(param.device)


def wrap_weight_loaders(layer: Any) -> None:
    """From create_weights, after any other wrapper: also capture TP slices."""
    st = getattr(layer, "_tpd", None)
    if st is None:
        return
    for pname, param in list(layer.named_parameters(recurse=False)):
        if pname not in _RAW:
            continue
        orig = getattr(param, "weight_loader", None)
        if orig is None or getattr(orig, "_tpd_wrapped", False):
            continue

        def loader(param, loaded_weight, weight_name, shard_id, expert_id,
                   return_success=False, _orig=orig, _pname=pname):
            _capture(st, _pname, param, loaded_weight, shard_id, int(expert_id))
            return _orig(param, loaded_weight, weight_name, shard_id, expert_id,
                         return_success=return_success)

        loader._tpd_wrapped = True
        param.weight_loader = loader


def _dummy_weights() -> bool:
    from vllm_torchtpu.layers.adapter import moe_redundant

    return moe_redundant._dummy_weights()


def finalize(layer: Any, ep_w13_scale: torch.Tensor) -> None:
    """From process_weights_after_loading, after the EP weights are prepared:
    fuse + requantize the raw TP slices exactly like the EP weights, register
    them as buffers and free the raw slices."""
    st = getattr(layer, "_tpd", None)
    if st is None:
        return
    from vllm_torchtpu.layers.adapter.fused_moe import requant_load_kmajor_fp4

    ep, E, H, I = st["ep"], st["E"], st["H"], st["I"]
    i_s = I // ep
    block = H // int(ep_w13_scale.shape[1])
    if block != QB:
        logger.warning_once("[moe-tp-dense] off: EP weight block %d != %d", block, QB)
        layer._tpd = None
        return
    raw = st["raw"]
    want = {"w13_weight": 2 * E, "w13_weight_scale": 2 * E, "w13_weight_scale_2": 2 * E,
            "w2_weight": E, "w2_weight_scale": E, "w2_weight_scale_2": E}
    missing = {n: want[n] - len(st["seen"][n]) for n in _RAW if len(st["seen"][n]) != want[n]}
    if missing and not _dummy_weights():
        raise RuntimeError(f"[moe-tp-dense] {layer.layer_name}: slices never loaded {missing} "
                           "(is the EP weight filter still on?)")
    dev = layer.w13_weight.device
    if missing:  # dummy weights: shapes only.
        g = 16
        raw.setdefault("w13_weight", torch.zeros((E, 2 * i_s, H // 2), dtype=torch.uint8, device=dev))
        raw.setdefault("w13_weight_scale", torch.ones((E, 2 * i_s, H // g), dtype=torch.float32, device=dev))
        raw.setdefault("w13_weight_scale_2", torch.ones((E, 2), dtype=torch.float32, device=dev))
        raw.setdefault("w2_weight", torch.zeros((E, H, i_s // 2), dtype=torch.uint8, device=dev))
        raw.setdefault("w2_weight_scale", torch.ones((E, H, i_s // g), dtype=torch.float32, device=dev))
        raw.setdefault("w2_weight_scale_2", torch.ones((E,), dtype=torch.float32, device=dev))
    s13 = raw.pop("w13_weight_scale").to(torch.float32)
    g13 = raw.pop("w13_weight_scale_2")
    s13 = torch.cat([s13[:, :i_s] * g13[:, 0].view(-1, 1, 1),
                     s13[:, i_s:] * g13[:, 1].view(-1, 1, 1)], dim=1)
    s2 = raw.pop("w2_weight_scale").to(torch.float32) * raw.pop("w2_weight_scale_2").view(-1, 1, 1)
    w13, s13k = requant_load_kmajor_fp4(raw.pop("w13_weight"), s13, QB)
    del s13
    w2, s2k = requant_load_kmajor_fp4(raw.pop("w2_weight"), s2, QB)
    del s2
    w, s = _pack_op()(w13, s13k, w2, s2k)
    shapes = [list(w13.shape), list(s13k.shape), list(w2.shape), list(s2k.shape)]
    del w13, s13k, w2, s2k
    layer.register_buffer("_tpd_w", w, persistent=False)
    layer.register_buffer("_tpd_s", s, persistent=False)
    st["raw"] = {}
    st["seen"] = None
    nbytes = sum(int(t.numel()) * t.element_size() for t in (w, s))
    _STATS["layers"] += 1
    _STATS["bytes"] += nbytes
    if dev.type == "tpu":
        get_op(topk=layer.moe_config.experts_per_token, fs=i_s)
    if _STATS["layers"] == 1:
        logger.info(
            "[moe-tp-dense] TPU_MOE_TP_DENSE_FP4_MAX_TOKENS=%d on: E=%d ep=%d rank=%d "
            "slice I/ep=%d w13/s13/w2/s2=%s packed w=%s s=%s bytes/layer=%d row_align=%d first_layer=%s",
            MAX_TOKENS, E, ep, st["rank"], i_s, shapes, list(w.shape), list(s.shape), nbytes,
            ROW_ALIGN, getattr(layer, "layer_name", "?"))


_PACK_OP = None


def _pack_jax(w13: jax.Array, s13: jax.Array, w2: jax.Array, s2: jax.Array):
    from vllm_torchtpu.kernels.experimental.moe_tp_dense_fp4 import pack

    return pack(w13, s13.reshape(s13.shape[0], s13.shape[1], -1), w2,
                s2.reshape(s2.shape[0], s2.shape[1], -1))


def _pack_op():
    global _PACK_OP
    if _PACK_OP is None:
        from torch_tpu._internal import pallas

        _PACK_OP = pallas.jax_op("pallas::moe_tp_dense_pack", _pack_jax)
    return _PACK_OP


def active(layer: Any) -> bool:
    return getattr(layer, "_tpd", None) is not None and hasattr(layer, "_tpd_w")


# --------------------------------------------------------------------------- traced


def tp_dense_routed(x, topk_weights, topk_ids, w, s, *, fs):
    """Routed experts of one rank's TP slice for all T tokens (jax). x [T, H];
    topk_* [T, k] with LOGICAL ids; w, s the packed per-expert records
    (moe_tp_dense_fp4.pack). Returns this rank's partial [T, H]."""
    from vllm_torchtpu.kernels.experimental.moe_tp_dense_fp4 import tp_dense_moe_packed

    t, _ = x.shape
    e = w.shape[0]
    mp = -(-t // ROW_ALIGN) * ROW_ALIGN
    ids = topk_ids.astype(jnp.int32)
    tw = jnp.where(jnp.isfinite(topk_weights), topk_weights, 0).astype(jnp.float32)
    oh = (ids[:, :, None] == jnp.arange(e, dtype=jnp.int32)[None, None, :]) & (tw != 0)[:, :, None]
    gw = jnp.sum(jnp.where(oh, tw[:, :, None], 0.0), axis=1)  # [T, E]
    hitmask = jnp.any(oh, axis=(0, 1))  # [E]
    hit = jnp.argsort(jnp.logical_not(hitmask), stable=True).astype(jnp.int32)
    nhit = jnp.sum(hitmask.astype(jnp.int32)).reshape(1)
    gw = jnp.pad(gw, ((0, mp - t), (0, 0))).T  # [E, Mp]
    gw = jnp.broadcast_to(gw[:, :, None], (e, mp, 128))
    xp = jnp.pad(x, ((0, mp - t), (0, 0)))
    out = tp_dense_moe_packed(xp, gw, w, s, hit, nhit, fs=fs)
    return out[:t]


def _op_body(hidden_states: jax.Array, w1: jax.Array, w2: jax.Array, w1_scale: jax.Array,
             w2_scale: jax.Array, topk_weights: jax.Array, topk_ids: jax.Array,
             experts_start: jax.Array, logical_ids: jax.Array, tw: jax.Array,
             ts: jax.Array, *, topk: int, activation: str, fs: int) -> jax.Array:
    from vllm_torchtpu.layers.core.fused_moe_gmm import (
        fused_moe_func,
        resolve_onehot_permute_threshold,
    )

    if 0 < hidden_states.shape[0] <= MAX_TOKENS and activation == "silu":
        logger.info_once("[moe-tp-dense] TP-dense fp4 decode MoE selected (T<=%d)", MAX_TOKENS)
        return tp_dense_routed(hidden_states, topk_weights, logical_ids, tw, ts, fs=fs)
    return fused_moe_func(
        hidden_states, w1, w2, w1_scale, w2_scale, None, None, topk_weights, topk_ids,
        experts_start, topk=topk, activation=activation, use_ep=True,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(),
        rhs_quant_dtype=None, skip_padded_tokens=envs.TPU_MOE_SKIP_PADDED_TOKENS)


_OPS: dict = {}


def get_op(*, topk: int, fs: int, activation: str = "silu"):
    key = (topk, activation, fs)
    op = _OPS.get(key)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    fn = functools.partial(_op_body, topk=topk, activation=activation, fs=fs)
    op = pallas.jax_op(f"pallas::fused_moe_tp_dense_{len(_OPS)}", fn)

    def _fake(hidden_states, *args, **kwargs):
        return torch.empty_like(hidden_states)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


def fused_moe(layer: Any, kwargs: dict, logical_ids: torch.Tensor) -> torch.Tensor:
    """Torch side: the EP fused-MoE call of nvfp4 apply, plus the TP copy."""
    op = get_op(topk=kwargs["topk"], fs=layer._tpd["I"] // layer._tpd["ep"],
                activation=kwargs["activation"])
    return op(kwargs["hidden_states"], kwargs["w1"], kwargs["w2"], kwargs["w1_scale"],
              kwargs["w2_scale"], kwargs["topk_weights"], kwargs["topk_ids"],
              kwargs["experts_start"], logical_ids.to(torch.int32), layer._tpd_w,
              layer._tpd_s)
