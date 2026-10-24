# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Serving glue for the TP x DCP GLM-5.2 decode megakernel (kernels/glm5_mk_dcp).

Enabled by GLM5_MKD=1 on a TP = DCP = world, DP1, EP deployment with MTP (verify rows). The runner
takes the kernel on a step where every scheduled request is in decode/verify (no prefill chunk), at
most QMAX tokens each and TMAX in total, in one chunk; every TP rank sees the same scheduler output, so
the decision is rank-uniform without a collective. Everything else runs stock.

Operands:
- weights: load-time stacked copies [n_layers_of_kind, ...] of every kind except the routed experts
  (~2.8 GB/rank; per-layer operands would add an L-leaf DMA cascade per stream issue site and overflow
  the TensorCore instruction memory: 108.6k bundles vs 49.2k, STRATEGY P5b); the experts alias the
  served tensors per layer;
- caches: the runner's stacked caches (GLM5_MKD_STACK_C=1, runner.glm5_kv_stack), read-only in the
  megakernel and updated in place by the barrier-free writer (donated);
- smalls: one f32 row per layer per rank (norm weights, per-channel scales, router bias; the o_proj,
  shared-down and dense-down scales are per rank: main requantizes them over the local K).
The kernel's rank is Mosaic's device id on a one-axis mesh of all R devices in torch-rank order (the
DCP mesh's order); experts [ER*r, ER*r+ER) live on rank r (linear EP placement, EP rank = TP rank).
"""

import dataclasses
import math
import os
import time
from typing import Any

import jax
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from . import kernel as K
from .config import FULL
from .layout import rup, smalls_layout
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

AXIS = "mkd"
EXPERT_ALIAS = ("ew13", "ew2")       # per layer, aliased (never copied)


class Unsupported(Exception):
    pass


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise Unsupported(why)


def enabled() -> bool:
    return os.environ.get("GLM5_MKD", "0") == "1"


def _unwrap(x):
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x


def _kn(weight: torch.Tensor, k: int, n: int, what: str) -> torch.Tensor:
    if tuple(weight.shape) == (k, n):
        return weight
    _need(tuple(weight.shape) == (n, k), f"{what}: shape {tuple(weight.shape)}, want ({k}, {n})")
    return weight.t().contiguous()


def _pad_n(w: torch.Tensor) -> torch.Tensor:
    n = w.shape[-1]
    return torch.nn.functional.pad(w, (0, rup(n) - n)) if n % 128 else w


def _scale(layer, n: int, what: str) -> torch.Tensor:
    for name in ("weight_scale", "weight_scale_inv"):
        s = getattr(layer, name, None)
        if s is not None:
            _need(s.numel() == n, f"{what}.{name}: {tuple(s.shape)} is not per-channel [{n}]")
            return s.reshape(-1).float()
    raise Unsupported(f"{what}: no weight_scale")


def _find_backbone(model):
    for m in model.modules():
        if hasattr(m, "layers") and hasattr(m, "embed_tokens") and hasattr(m, "norm"):
            return m
    raise Unsupported("no decoder backbone (layers/embed_tokens/norm) in the model")


def _attn_module(self_attn):
    for m in self_attn.modules():
        if hasattr(m, "W_UK_T") and hasattr(m, "W_UV") and hasattr(m, "kv_cache"):
            return m
    raise Unsupported("no MLA attention module with W_UK_T / W_UV / kv_cache")


def _own(t: torch.Tensor) -> torch.Tensor:
    """A tensor with storage of exactly its own shape (not a view): a view operand of the compiled
    step is materialized by torch_tpu's execute_as_strided on EVERY call (2.2 ms/step of copies of
    the stacked weights and smalls, j-725a55a4)."""
    out = torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device)
    out.copy_(t)
    if out._base is not None:
        logger.warning("GLM-5.2 TPxDCP megakernel: owned operand %s %s is still a view",
                       tuple(out.shape), out.dtype)
    return out


def _routed(mlp):
    exp = mlp.experts
    return getattr(exp, "routed_experts", exp)


def _expert_scale(routed, name: str, shape: tuple) -> torch.Tensor:
    for suffix in ("_scale_inv", "_scale"):
        s = getattr(routed, name + suffix, None)
        if s is not None:
            _need(s.numel() == math.prod(shape), f"{name}{suffix}: {tuple(s.shape)}, want {shape}")
            return s.reshape(shape).float()
    raise Unsupported(f"routed experts: no {name} scale")


class Glm5MKD:
    """Built at load_model (weights), bound to the stacked caches at capture_model."""

    @classmethod
    def maybe_create(cls, runner) -> "Glm5MKD | None":
        if not enabled():
            return None
        try:
            return cls(runner)
        except Unsupported as e:
            logger.warning("GLM-5.2 TPxDCP megakernel disabled: %s", e)
            return None

    def __init__(self, runner):
        import torch.distributed as dist

        hf = runner.model_config.hf_config
        pc = runner.parallel_config
        _need(type(hf).__name__.startswith("GlmMoeDsa")
              or getattr(hf, "model_type", "") == "glm_moe_dsa", f"model_type {hf.model_type}")
        _need(pc.pipeline_parallel_size == 1, "pipeline parallel")
        _need(getattr(pc, "data_parallel_size", 1) == 1, "data parallel")
        R = int(pc.tensor_parallel_size)
        _need(int(getattr(pc, "decode_context_parallel_size", 1)) == R, "DCP != TP")
        _need(dist.is_initialized() and dist.get_world_size() == R, "world != TP")
        _need(int(getattr(pc, "cp_kv_cache_interleave_size", 1)) == FULL.ILV, "interleave != 4")
        _need(bool(getattr(pc, "enable_expert_parallel", False)), "EP is off")
        _need(os.environ.get("VLLM_DCP_Q_REPLICATE", "0") == "1", "q replicate is off")
        spec = runner.speculative_config
        _need(spec is not None and getattr(spec, "method", "") == "mtp", "MTP is off")
        qmax = 1 + int(spec.num_speculative_tokens)
        _need(qmax <= 4, f"{qmax} rows per request")
        _need(getattr(hf, "scoring_func", None) == "sigmoid", "router scoring")
        _need(getattr(hf, "n_group", 1) == 1 and getattr(hf, "topk_group", 1) == 1, "grouped routing")
        _need(bool(getattr(hf, "norm_topk_prob", True)), "unnormalized top-k")
        _need(bool(getattr(hf, "rope_interleave", False))
              and bool(getattr(hf, "indexer_rope_interleave", False)), "non-interleaved rope")
        rp = getattr(hf, "rope_parameters", None) or {}
        _need(rp.get("rope_type", "default") == "default", f"rope {rp}")
        _need(runner.block_size == FULL.page, f"block_size {runner.block_size}")
        cfg = dataclasses.replace(
            FULL, R=R, L=hf.num_hidden_layers, first_dense=hf.first_k_dense_replace,
            H=hf.hidden_size, NH=hf.num_attention_heads, QL=hf.q_lora_rank,
            idx_freq=hf.index_topk_freq, idx_skip=hf.index_skip_topk_offset,
            rope_theta=float(rp.get("rope_theta", FULL.rope_theta)), eps=hf.rms_norm_eps,
            routed_scale=float(hf.routed_scaling_factor), QMAX=qmax)
        for got, want, what in (
                (hf.kv_lora_rank, cfg.KL, "kv_lora_rank"), (hf.qk_nope_head_dim, cfg.DN, "qk_nope"),
                (hf.qk_rope_head_dim, cfg.DR, "qk_rope"), (hf.v_head_dim, cfg.DV, "v_head_dim"),
                (hf.n_routed_experts, cfg.E, "experts"), (hf.num_experts_per_tok, cfg.TOPK, "top-k"),
                (hf.moe_intermediate_size, cfg.I, "moe_intermediate"),
                (hf.moe_intermediate_size * hf.n_shared_experts, cfg.IS, "shared intermediate"),
                (hf.intermediate_size, cfg.ID, "dense intermediate"),
                (hf.index_n_heads, cfg.IH, "index heads"), (hf.index_head_dim, cfg.IDIM, "index dim"),
                (hf.index_topk, cfg.KT, "index_topk")):
            _need(got == want, f"{what} {got} != kernel {want}")
        _need(cfg.NH % R == 0 and cfg.E % R == 0 and cfg.IS % R == 0 and cfg.ID % R == 0,
              f"{R} ranks do not divide heads / experts / intermediates")
        self.cfg = cfg
        self.rank = int(dist.get_rank())
        from torch_tpu._internal.pallas import get_pallas_mesh
        self.mesh = get_pallas_mesh(axis_names=(AXIS,), mesh_shape=(R,))
        bb = _find_backbone(runner.model)
        self.backbone = bb
        _need(len(bb.layers) == cfg.L, f"{len(bb.layers)} layers")
        # local pages a request can span on one rank: a block-table entry is a DCP virtual page of
        # page * R positions (runner.max_num_blocks_per_req is the generic width, R times larger,
        # and sized the key scratch 16x too big: 79.6 MiB VMEM, j-f2171e56)
        mml = int(runner.model_config.max_model_len)
        self.max_pages = -(-mml // (FULL.page * R))
        self.rotary = bb.layers[0].self_attn.rotary_emb
        _need(self.rotary.cos_sin_cache.shape[-1] == cfg.DR, "rope cache width")
        # own row-major copy: gathering 16 rows of the served cache made XLA transpose the whole
        # [max_pos, 64] param every step (246 us, j-725a55a4)
        with torch.no_grad():
            self.cos_sin = _own(self.rotary.cos_sin_cache.detach())
        self.wk = K.weight_kinds(cfg)
        self.stack_w = frozenset(set(self.wk) - set(EXPERT_ALIAS))
        with torch.no_grad():
            self.weights, self.w_names, self.w_specs = self._collect_weights()
            self.smalls = self._build_smalls().detach()
        from vllm_torchtpu.utils import synchronize_tensors
        synchronize_tensors(self.weights + [self.smalls])
        copied = sum(t.numel() * t.element_size() for (k, _), t in zip(self.w_names, self.weights)
                     if k not in EXPERT_ALIAS)
        logger.info("GLM-5.2 TPxDCP megakernel: %d weight operands (%.2f GiB load-time copies), rank %d/%d",
                    len(self.weights), copied / 2 ** 30, self.rank, R)
        self.caches = None
        self._step = None
        self.mla_layer_name = _attn_module(bb.layers[0].self_attn).layer_name

    # ---------------------------------------------------------------- weights

    def _layer(self, l: int) -> dict[str, torch.Tensor]:
        cfg, r = self.cfg, self.rank
        H = cfg.H
        L = self.backbone.layers[l]
        sa = L.self_attn
        at = _attn_module(sa)
        NQB = cfg.HR * cfg.DQ
        d = {
            "qkv_a": _pad_n(_kn(sa.fused_qkv_a_proj.weight, H, cfg.QKVA, "fused_qkv_a_proj")),
            "q_b": _kn(sa.q_b_proj.weight, cfg.QL, cfg.NH * cfg.DQ, "q_b_proj")[:, r * NQB:(r + 1) * NQB],
            "w_uk": at.W_UK_T,
            "w_uv": at.W_UV,
            "o_proj": _kn(sa.o_proj.weight, cfg.KR, H, "o_proj"),
        }
        idx = getattr(sa, "indexer", None)
        _need((idx is not None) == cfg.full_indexer(l), f"layer {l}: indexer pattern")
        if idx is not None:
            d["idx_wq"] = _kn(idx.wq_b.weight, cfg.QL, cfg.IH * cfg.IDIM, "indexer.wq_b")
            d["idx_wk"] = _pad_n(_kn(idx.wk_weights_proj.weight, H, cfg.IDIM + cfg.IH,
                                     "indexer.wk_weights_proj"))
        mlp = L.mlp
        if cfg.is_moe(l):
            d["gate"] = _kn(mlp.gate.weight, H, cfg.E, "gate")
            rt = _routed(mlp)
            d["ew13"] = rt.w13_weight
            d["ew2"] = rt.w2_weight
            d["es13"] = _expert_scale(rt, "w13_weight", (cfg.ER, 2 * cfg.I))
            d["es2"] = _expert_scale(rt, "w2_weight", (cfg.ER, H))
            se = mlp.shared_experts
            d["sw13"] = _kn(se.gate_up_proj.weight, H, 2 * cfg.ISR, "shared.gate_up_proj")
            d["sw2"] = _kn(se.down_proj.weight, cfg.ISR, H, "shared.down_proj")
        else:
            d["w13"] = _kn(mlp.gate_up_proj.weight, H, 2 * cfg.IDR, "dense.gate_up_proj")
            d["w2"] = _kn(mlp.down_proj.weight, cfg.IDR, H, "dense.down_proj")
        return d

    def _collect_weights(self):
        """Operands in kernel.build's w_names order: one stacked copy per kind in stack_w, else the
        served per-layer tensors."""
        per_layer = {}
        out, names, specs = [], [], []
        for kind, (layers, shape, dtype) in self.wk.items():
            if not layers:                       # a kind no layer has (kernel.build drops it too)
                continue
            want = {"float8_e4m3fn": torch.float8_e4m3fn, "bfloat16": torch.bfloat16,
                    "float32": torch.float32}[jax.numpy.dtype(dtype).name]
            ts = []
            for l in layers:
                if l not in per_layer:
                    per_layer[l] = self._layer(l)
                t = per_layer[l][kind]
                _need(tuple(t.shape) == tuple(shape), f"{kind}[{l}]: {tuple(t.shape)} != {shape}")
                _need(t.dtype == want, f"{kind}[{l}]: {t.dtype} != {want}")
                ts.append(t.detach())
            if kind in self.stack_w:
                out.append(_own(torch.stack(ts)))
                names.append((kind, "stk"))
                specs.append(P(None, AXIS))
            else:
                for l, t in zip(layers, ts):
                    out.append(t)
                    names.append((kind, l))
                    specs.append(P(AXIS))
        return out, names, specs

    def _build_smalls(self) -> torch.Tensor:
        cfg, r = self.cfg, self.rank
        lay, S = smalls_layout(cfg)
        rows = []
        for l in range(cfg.L + 1):
            f = self._small_fields(l) if l < cfg.L else {"in_ln": self.backbone.norm.weight}
            dev = next(iter(f.values())).device
            parts = []
            for name, (_, n) in lay.items():
                v = f.get(name)
                v = torch.zeros(n, dtype=torch.float32, device=dev) if v is None else v.reshape(-1).float()
                _need(v.numel() == n, f"smalls {name}[{l}]: {v.numel()} != {n}")
                parts.append(torch.nn.functional.pad(v, (0, rup(n) - n)))
            rows.append(torch.cat(parts))
        return _own(torch.stack(rows).reshape(cfg.L + 1, 1, S))

    def _small_fields(self, l: int) -> dict[str, torch.Tensor]:
        cfg, r = self.cfg, self.rank
        L = self.backbone.layers[l]
        sa = L.self_attn
        NQB = cfg.HR * cfg.DQ
        qkv_s = _scale(sa.fused_qkv_a_proj, cfg.QKVA, "fused_qkv_a_proj")
        f = {
            "in_ln": L.input_layernorm.weight,
            "post_ln": L.post_attention_layernorm.weight,
            "q_a_ln": sa.q_a_layernorm.weight,
            "kv_a_ln": sa.kv_a_layernorm.weight,
            "qkv_a_s": torch.nn.functional.pad(qkv_s, (0, rup(cfg.QKVA) - cfg.QKVA)),
            "q_b_s": _scale(sa.q_b_proj, cfg.NH * cfg.DQ, "q_b_proj")[r * NQB:(r + 1) * NQB],
            "o_proj_s": _scale(sa.o_proj, cfg.H, "o_proj"),
        }
        for norm in (L.input_layernorm, L.post_attention_layernorm, sa.q_a_layernorm, sa.kv_a_layernorm):
            _need(abs(float(getattr(norm, "variance_epsilon", cfg.eps)) - cfg.eps) < 1e-12,
                  f"layer {l}: rmsnorm eps")
        idx = getattr(sa, "indexer", None)
        if idx is not None:
            f["idx_wq_s"] = _scale(idx.wq_b, cfg.IH * cfg.IDIM, "indexer.wq_b")
            f["idx_kn_w"] = idx.k_norm.weight
            f["idx_kn_b"] = idx.k_norm.bias
        mlp = L.mlp
        if cfg.is_moe(l):
            f["gate_bias"] = mlp.gate.e_score_correction_bias.float()
            f["sw13_s"] = _scale(mlp.shared_experts.gate_up_proj, 2 * cfg.ISR, "shared.gate_up_proj")
            f["sw2_s"] = _scale(mlp.shared_experts.down_proj, cfg.H, "shared.down_proj")
        else:
            f["w13_s"] = _scale(mlp.gate_up_proj, 2 * cfg.IDR, "dense.gate_up_proj")
            f["w2_s"] = _scale(mlp.down_proj, cfg.H, "dense.down_proj")
        return f

    # ----------------------------------------------------------------- caches

    def bind_caches(self, runner) -> None:
        """Stacked caches (GLM5_MKD_STACK_C=1: runner.glm5_kv_stack) when the runner allocated them,
        else the served per-layer caches (one operand per (kind, layer): a larger program, ~81k
        bundles vs ~49k, but the stock path is untouched)."""
        cfg = self.cfg
        grp = getattr(runner, "glm5_kv_stack_groups", None) or {}
        if not grp:                               # older runner contract: one tensor per kind
            stk = getattr(runner, "glm5_kv_stack", None) or {}
            grp = {k: [v] for k, v in stk.items()}
        caches = []
        if set(grp) == {"nope", "rope", "idx"}:
            nb = int(grp["nope"][0][0].shape[1])
            ck = K.cache_kinds(cfg, nb)
            G = {}
            for kind, (layers, shape, dtype) in ck.items():
                groups = grp[kind]
                gs = len(layers) // len(groups)
                _need(gs * len(groups) == len(layers), f"{kind}: {len(groups)} groups of {len(layers)} layers")
                for gi, (t, names) in enumerate(groups):
                    _need(tuple(t.shape) == (gs,) + tuple(shape),
                          f"stacked cache {kind}[{gi}]: {tuple(t.shape)} != {(gs,) + tuple(shape)}")
                    caches.append(t)
                G[kind] = len(groups)
            _need(G["nope"] == G["rope"], "nope / rope grouped differently")
            self.stack_c = G
            self.c_specs = [P(None, AXIS)] * len(caches)
        else:
            _need(not grp, f"partially stacked caches {sorted(grp)}")
            per = {}
            for l in range(cfg.L):
                sa = self.backbone.layers[l].self_attn
                kv = _unwrap(_attn_module(sa).kv_cache)
                _need(isinstance(kv, (tuple, list)) and len(kv) == 2,
                      f"layer {l}: MLA cache is not the (nope, rope) SparseCore pair")
                per[("nope", l)], per[("rope", l)] = kv[0], kv[1]
                idx = getattr(sa, "indexer", None)
                if idx is not None:
                    per[("idx", l)] = _unwrap(idx.k_cache.kv_cache)
            nb = int(per[("nope", 0)].shape[0])
            ck = K.cache_kinds(cfg, nb)
            for kind, (layers, shape, dtype) in ck.items():
                for l in layers:
                    t = per[(kind, l)]
                    _need(tuple(t.shape) == tuple(shape), f"cache {kind}[{l}]: {tuple(t.shape)} != {shape}")
                    caches.append(t)
            self.stack_c = frozenset()
            self.c_specs = [P(AXIS)] * len(caches)
        logger.info("GLM-5.2 TPxDCP megakernel: %s caches, %d operands, %d blocks",
                    f"stacked {self.stack_c}" if self.stack_c else "per-layer", len(caches), nb)
        self.caches = caches
        self.nb = nb
        self._step = self._build_op()

    def _build_op(self):
        cfg = self.cfg
        build_kw = dict(nb=self.nb, max_pages=self.max_pages, bmax=cfg.TMAX,
                        stack_w=self.stack_w, stack_c=self.stack_c)
        # GLM5_MKD_DEBUG=1: also return the per-layer owner rows (dbg_h) and owned winner lists
        # (dbg_own) per rank -> self.last_dbg (parity diagnostics only)
        self.debug = os.environ.get("GLM5_MKD_DEBUG", "0") == "1"
        call, info = K.build(cfg, axis_name=AXIS, debug=self.debug, dbg_part=False, **build_kw)
        writer, winfo = K.build_writer(cfg, axis_name=AXIS, nb=self.nb, max_pages=self.max_pages,
                                       bmax=cfg.TMAX, stack_c=build_kw["stack_c"])
        assert list(info["w_names"]) == self.w_names, "weight operand order"
        assert len(info["c_names"]) == len(self.caches) == len(winfo["c_names"])
        self.info = info
        n_w, n_c = len(self.weights), len(self.caches)
        rep = P()
        in_specs = (rep,) * 7 + (P(AXIS),) + tuple(self.w_specs) + tuple(self.c_specs)
        n_dbg = 2 if self.debug else 0
        out_specs = (P(AXIS),) + tuple(self.c_specs) + (P(AXIS),) * n_dbg

        def body(*a):
            scal, rmeta, qmeta, bt, rvec, h0, tab, smalls = a[:8]
            caches = a[8 + n_w:]
            outs = call(*a)
            out, nrow, rrow, irec = outs[:4]
            new = writer(scal, rmeta, bt, nrow, rrow, irec, *caches)
            return (out, *new, *outs[4:4 + n_dbg])

        smap = jax.shard_map(body, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs,
                             check_vma=False)
        n_in = 8 + n_w + n_c
        args = [f"x{i}" for i in range(n_in)]
        ret = "tuple[" + ", ".join(["jax.Array"] * len(out_specs)) + "]"
        src = (f"def glm5_mkd({', '.join(a + ': jax.Array' for a in args)}) -> {ret}:\n"
               f"    return _smap({', '.join(args)})\n")
        ns: dict[str, Any] = {"jax": jax, "_smap": smap}
        exec(compile(src, "<glm5_mkd>", "exec"), ns)  # noqa: S102 - arity is data
        name = "glm5_mkd::step"
        # unregistered callable: ~175 operands exceed a torch op schema's 64 (j-a67fe755); the op
        # below takes the weights and caches as two Tensor[] arguments
        flat = sharded_jax_op(name + "_flat", ns["glm5_mkd"], mesh=self.mesh,
                              input_partition_specs=in_specs, output_partition_specs=out_specs,
                              donate_argnums=tuple(range(n_in - n_c, n_in)), register=False)

        @torch.library.custom_op(name, mutates_args=())
        def op(scal: torch.Tensor, rmeta: torch.Tensor, qmeta: torch.Tensor, bt: torch.Tensor,
               rvec: torch.Tensor, h0: torch.Tensor, tab: torch.Tensor, smalls: torch.Tensor,
               weights: list[torch.Tensor], caches: list[torch.Tensor]) -> list[torch.Tensor]:
            return list(flat(scal, rmeta, qmeta, bt, rvec, h0, tab, smalls, *weights, *caches))

        @op.register_fake
        def _(scal, rmeta, qmeta, bt, rvec, h0, tab, smalls, weights, caches):
            M = info["M"]
            dbg = ([torch.empty((cfg.L, 2, cfg.RPR, cfg.H), dtype=torch.float32, device=h0.device),
                    torch.empty((cfg.L, cfg.TMAX, cfg.KT // 128, 128), dtype=torch.int32, device=h0.device)]
                   if n_dbg else [])
            return ([torch.empty((M, cfg.H), dtype=torch.bfloat16, device=h0.device)]
                    + [torch.empty_like(c) for c in caches] + dbg)

        TMAX, M, DR = cfg.TMAX, info["M"], cfg.DR
        nb = int(self.nb)
        BMAX = cfg.TMAX
        max_pages = self.max_pages

        def step(ids, pos, qsl, seq_lens, bt_flat, cos_sin, smalls, weights, caches):
            dev = ids.device
            i32 = torch.int32
            nq_tab = qsl.shape[0] - 1
            bt2 = bt_flat.reshape(nq_tab, -1).to(i32)
            # pad / truncate the request axis to BMAX and the page axis to max_pages
            def fit(x, n, axis=0):
                k = x.shape[axis]
                if k >= n:
                    return x.narrow(axis, 0, n)
                pad = [0, 0] * (x.dim() - 1 - axis) + [0, n - k]
                return torch.nn.functional.pad(x, pad)
            qsl_ = qsl.to(i32)                        # padded entries repeat the total (runner)
            last = qsl_[-1:]
            qsl_b = torch.cat([qsl_, last.expand(BMAX + 1)])[: BMAX + 1]
            T = qsl_[-1]
            seq_b = fit(seq_lens.to(i32), BMAX)
            # DCP block tables hold VIRTUAL page ordinals; every main reader / writer resolves them on
            # this rank as `virtual % num_pages` (kernels/mla/kv_cache_utils.py:583, :713). Unresolved,
            # ids >= nb made the megakernel read (and its writer write) out of bounds -- the
            # real-weight parity failure (bounds checks are off in the kernel)
            bt_b = torch.remainder(fit(fit(bt2, BMAX, 0), max_pages, 1), nb)
            t = torch.arange(TMAX, device=dev, dtype=i32)
            valid = (t < T).to(i32)
            req = (qsl_b[1:BMAX + 1][None, :] <= t[:, None]).sum(1).to(i32)
            req = torch.clamp(req, max=BMAX - 1)
            p = fit(pos.to(i32), TMAX) * valid
            seq_t = seq_b[req] * valid
            z = torch.zeros(TMAX, dtype=i32, device=dev)
            rmeta = torch.stack([valid, p, req * valid, seq_t, z, z, z, z], 1)
            t0 = qsl_b[:BMAX]
            nq = qsl_b[1:BMAX + 1] - t0
            qmeta = torch.stack([t0, nq, seq_b, (nq > 0).to(i32)], 1)
            B = (nq > 0).sum().to(i32)
            scal = torch.cat([torch.stack([T, B]), torch.zeros(6, dtype=i32, device=dev)])
            rvec = torch.zeros((M, 128), dtype=i32, device=dev)
            rvec[:TMAX, 0] = valid
            rvec[:TMAX, 1] = p
            # rope tables: GPT-J pairs in lanes 0..DR-1 (cos on both, -sin / +sin), rolled to
            # lanes 128-DR..127; identity elsewhere; bf16 values (main's cos/sin cache dtype)
            cs = cos_sin[p.long()].to(torch.bfloat16).float()                 # [TMAX, DR]
            c = cs[:, : DR // 2].repeat_interleave(2, dim=1)
            s = torch.stack([-cs[:, DR // 2:], cs[:, DR // 2:]], -1).reshape(TMAX, DR)
            ones = torch.ones((TMAX, 128 - DR), dtype=torch.float32, device=dev)
            zeros = torch.zeros((TMAX, 128 - DR), dtype=torch.float32, device=dev)
            cos_lo, sin_lo = torch.cat([c, ones], 1), torch.cat([s, zeros], 1)
            cos_hi, sin_hi = torch.cat([ones, c], 1), torch.cat([zeros, s], 1)
            tab = torch.stack([cos_lo, sin_lo, cos_hi, sin_hi])
            tab = torch.nn.functional.pad(tab, (0, 0, 0, M - TMAX), value=0.0)
            tab[:, TMAX:, :] = torch.tensor([1.0, 0.0, 1.0, 0.0], device=dev)[:, None, None]
            h0 = self.backbone.embed_tokens(fit(ids, TMAX)).float()
            outs = op(scal, rmeta, qmeta, bt_b, rvec, h0, tab, smalls, list(weights), list(caches))
            n_c_ = len(caches)
            for dst, src in zip(caches, outs[1:1 + n_c_]):
                dst.copy_(src)
            return (outs[0][:TMAX], *outs[1 + n_c_:])

        return torch.compile(step, backend="tpu", fullgraph=True, dynamic=False)

    # ------------------------------------------------------------------- run

    def step_ok(self, runner, scheduler_output, num_chunks: int) -> bool:
        """Rank-uniform host predicate: every request in decode / verify, <= QMAX tokens each,
        <= TMAX total, one chunk, no encoder inputs."""
        cfg = self.cfg
        if self._step is None or num_chunks != 1 or scheduler_output.scheduled_encoder_inputs:
            return False
        if scheduler_output.total_num_scheduled_tokens > cfg.TMAX:
            return False
        ns = scheduler_output.num_scheduled_tokens
        if len(ns) > cfg.TMAX:
            return False
        ib = runner.input_batch
        for req_id, n in ns.items():
            i = ib.req_id_to_index[req_id]
            if n > cfg.QMAX or int(ib.num_computed_tokens_cpu[i]) < int(ib.num_prompt_tokens[i]):
                return False
        return True

    def _metadata(self) -> AttentionMetadata:
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        if isinstance(md, dict):
            md = md.get(self.mla_layer_name)
        assert isinstance(md, AttentionMetadata), type(md)
        return md

    @torch.no_grad()
    def __call__(self, runner, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self._step is None:               # capture_model was skipped (e.g. a microbenchmark)
            self.bind_caches(runner)
        md = self._metadata()
        res = self._step(input_ids, positions, md.query_start_loc, md.seq_lens, md.block_tables,
                         self.cos_sin, self.smalls, self.weights, self.caches)
        out = res[0]
        if len(res) > 1:
            self.last_dbg = tuple(res[1:])
        n = input_ids.shape[0]
        if n <= out.shape[0]:
            return out[:n]
        return torch.nn.functional.pad(out, (0, 0, 0, n - out.shape[0]))

    def warmup(self, runner) -> None:
        """Compile and run once: one request of QMAX rows at positions 0..QMAX-1 on block 0 (the
        null block; nothing is served from it before the first step)."""
        from vllm_torchtpu.utils import synchronize_tensors

        cfg = self.cfg
        dev = runner.device
        T = cfg.QMAX
        ids = torch.zeros(cfg.TMAX, dtype=torch.int32, device=dev)
        pos = torch.arange(cfg.TMAX, dtype=torch.int32, device=dev)
        qsl = torch.tensor([0] + [T] * cfg.TMAX, dtype=torch.int32, device=dev)
        seq = torch.tensor([T] + [0] * (cfg.TMAX - 1), dtype=torch.int32, device=dev)
        bt = torch.zeros(cfg.TMAX * self.max_pages, dtype=torch.int32, device=dev)
        t0 = time.monotonic()
        out = self._step(ids, pos, qsl, seq, bt, self.cos_sin, self.smalls, self.weights, self.caches)[0]
        synchronize_tensors([out])
        logger.info("GLM-5.2 TPxDCP megakernel compiled and warmed in %.1f s (%d weight operands)",
                    time.monotonic() - t0, len(self.weights))
