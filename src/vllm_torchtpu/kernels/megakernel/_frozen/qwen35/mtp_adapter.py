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
"""Megakernel driver for the Qwen3.5-397B-A17B-FP8 MTP draft layer.

The MTP draft is one full-attention decoder layer (`mtp.layers.0`) followed by
`mtp.norm`. `Qwen35MtpMegakernel` runs both through the attention-only pooled
megakernel step (`pool_io.make_flat_attn_step`, `L=1, att_every=1`) inside the
drafter's compiled programs:

- draft step 0 when every request verifies `K + 1` tokens: all `K + 1` rows of
  each request run attention and write their K/V, and only the rows gathered
  for the next draft token route through the experts;
- the fused draft loop (`mtp_fused_loop.py`, steps `1..K-1`): one row per
  request.

`qwen3_next_patch.patched_mtp_forward` hands the layer to `forward_layer` while
`mtp_megakernel_context` is active. The new K/V rows land in the draft layer's
unified pool at the slots ragged paged attention writes, so steps on either
path read each other's cache. Opt in with `VLLM_TPU_QWEN35_MTP_MK=1`.
"""

from __future__ import annotations

import dataclasses
import os
import time
import types
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from vllm.forward_context import get_forward_context

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.logger import init_logger

from . import megakernel as K
from . import pool_io
from .vllm_adapter import (
    _get_routed_experts,
    _need,
    _unwrap,
    pack_weights,
    select_bucket,
)

logger = init_logger(__name__)

# Rows per draft step: one per request in the loop, K + 1 = 4 per request at
# step 0. Larger steps keep the XLA draft layer.
_DEFAULT_LOOP_BUCKETS = (4, 8, 16)
_DEFAULT_VERIFY_BUCKETS = (4, 8, 16, 32)
_NUM_SPEC_TOKENS = 3
# `K.W_NAMES` leaves of a one-layer stack: lin_in .. sh2, e13, e2, es, smalls,
# final_norm. The experts are the layer's own parameters.
_NUM_HEAD_LEAVES = 7
_NUM_LEAVES = 12


def _parse_buckets(name: str, default: Sequence[int]) -> tuple[int, ...]:
    """Comma-separated row buckets from env `name` ('0' disables)."""
    raw = os.environ.get(name, ",".join(map(str, default)))
    if raw.strip().lower() in ("0", "none", "off", "false"):
        return ()
    buckets = sorted({int(x) for x in raw.split(",") if x.strip()})
    _need(
        all(4 <= b <= 64 and b % 4 == 0 for b in buckets),
        f"invalid {name}={raw!r}",
    )
    return tuple(buckets)


def _fit(x: jax.Array, rows: int) -> jax.Array:
    """The first `rows` rows of `x`, zero-padded when `x` is shorter."""
    if x.shape[0] >= rows:
        return x[:rows]
    return jnp.pad(x, [(0, rows - x.shape[0])] + [(0, 0)] * (x.ndim - 1))


def _step_operands(
    c: K.MKConfig,
    hidden: jax.Array,
    block_tables: jax.Array,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    gather_indices: jax.Array | None,
) -> tuple[jax.Array, ...]:
    """Per-rank `make_flat_attn_step` operands of a draft step.

    Sequence `s` is active when its `query_start_loc` span is `c.T` rows: its
    rows `s * T .. s * T + T - 1` are those hidden rows, and its tokens sit at
    `seq_lens[s] - T .. seq_lens[s] - 1`, the slots ragged paged attention
    writes. Inactive sequences write no K/V and route no experts, and their
    outputs are discarded. With `gather_indices`, only the gathered rows of
    the active requests route through the experts.
    """
    T, NS, B, M = c.T, c.NS, c.B, c.M
    knr = seq_lens.shape[0]
    pages = block_tables.reshape(knr, -1)
    live_all = (query_start_loc[1 : knr + 1] - query_start_loc[:knr]) == T
    live = _fit(live_all, NS)
    page_table = jnp.where(live[:, None], _fit(pages, NS), 0).astype(jnp.int32)
    lens = _fit(seq_lens.astype(jnp.int32), NS)
    j = jnp.arange(T, dtype=jnp.int32)
    pos = jnp.where(live[:, None], lens[:, None] - T + j, j).reshape(B)
    live_rows = jnp.repeat(live, T)
    rows = hidden.shape[0]
    hid = jnp.zeros((M, hidden.shape[1]), hidden.dtype).at[:rows].set(hidden)
    hid = jnp.where(_fit(live_rows, M)[:, None], hid, jnp.zeros_like(hid))
    if gather_indices is None:
        moe_rows = live_rows.astype(jnp.int32)
    else:
        num_reqs = jnp.sum(live_all.astype(jnp.int32))
        g = gather_indices.astype(jnp.int32)
        g = jnp.where(jnp.arange(g.shape[0]) < num_reqs, g, B)
        moe_rows = jnp.zeros((B,), jnp.int32).at[g].set(1, mode="drop")
    return pos, hid, page_table, live.astype(jnp.int32), moe_rows


class Qwen35MtpMegakernel:
    """Runs the Qwen3.5 MTP draft layer through the attention-only megakernel."""

    @classmethod
    def maybe_create(cls, runner: Any) -> Qwen35MtpMegakernel | None:
        """The driver for `runner`'s MTP drafter; None if disabled or unsupported."""
        if os.environ.get("VLLM_TPU_QWEN35_MTP_MK", "0") != "1":
            return None
        try:
            return cls(runner)
        except Exception as exc:  # noqa: BLE001 - the XLA draft layer still works
            logger.warning("Qwen3.5 MTP megakernel disabled: %s", exc, exc_info=True)
            return None

    def __init__(self, runner: Any, mesh: jax.sharding.Mesh | None = None) -> None:
        from vllm.distributed import (
            get_pp_group,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        from vllm_torchtpu.models.vllm.qwen3_next_patch import (
            is_step0_post_attn_gather_installed,
        )

        spec = getattr(runner, "speculative_config", None)
        drafter = getattr(runner, "drafter", None)
        _need(
            spec is not None and spec.method == "mtp" and drafter is not None,
            "needs an MTP drafter",
        )
        _need(
            int(spec.num_speculative_tokens) == _NUM_SPEC_TOKENS,
            f"num_speculative_tokens={spec.num_speculative_tokens} != "
            f"{_NUM_SPEC_TOKENS}",
        )
        for attr in ("_draft_replicated", "_draft_kv_shared", "_share_mtp_indices"):
            _need(not getattr(drafter, attr, False), f"drafter.{attr} is set")
        _need(not getattr(runner, "enforce_eager", False), "enforce_eager")
        _need(not envs.TPU_KERNEL_ITER_MODE, "TPU_KERNEL_ITER_MODE")
        unified = getattr(runner, "_unified_kv_layout", None)
        if unified is None:
            fn = getattr(runner, "_use_unified_kv_cache", None)
            unified = fn() if callable(fn) else True
        _need(bool(unified), "unified KV cache is not enabled")
        draft_model = drafter.draft_model
        predictor = getattr(draft_model, "model", None)
        layers = getattr(predictor, "layers", None)
        _need(
            layers is not None and len(layers) == 1 and hasattr(predictor, "norm"),
            "expected a one-layer MTP predictor",
        )
        _need(
            is_step0_post_attn_gather_installed(draft_model),
            "the patched MTP forward is not installed",
        )
        layer = layers[0]
        _need(
            getattr(layer, "layer_type", "full_attention") == "full_attention",
            f"layer_type={getattr(layer, 'layer_type', None)!r}",
        )
        _need(not getattr(layer, "layer_scale", False), "layer_scale")

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        _need(self.tp_size == K.FULL.R, f"tp_size={self.tp_size} != {K.FULL.R}")
        _need(get_pp_group().world_size == 1, "pipeline parallelism")

        base_cfg = dataclasses.replace(
            K.FULL_EP,
            L=1,
            att_every=1,
            moe_tp=False,
            stack_e=False,
            e_slots=0,
            paged_kv=True,
        )
        attn = layer.self_attn.attn
        for name in ("_k_scale_float", "_v_scale_float"):
            _need(float(getattr(attn, name, 1.0)) == 1.0, f"{name} != 1")
        impl = attn.impl
        _need(not getattr(impl, "_bm_kernel", False), "block-major KV pool")
        _need(getattr(impl, "sinks", None) is None, "attention sinks")
        _need(getattr(impl, "sliding_window", None) is None, "sliding window")
        experts = _get_routed_experts(layer.mlp)
        w13, w2 = experts.w13_weight, experts.w2_weight
        _need(
            w13.dtype == torch.float8_e4m3fn and w2.dtype == torch.float8_e4m3fn,
            f"expert weights {w13.dtype}/{w2.dtype} are not FP8",
        )
        _need(
            int(w13.shape[0]) == base_cfg.ER,
            f"{int(w13.shape[0])} expert slots per rank != {base_cfg.ER}",
        )
        for name in ("_ecopy", "_ecopy_cores"):
            _need(getattr(experts, name, None) is None, f"experts.{name} is set")
        expert_map = getattr(experts, "expert_map", None)
        if isinstance(expert_map, torch.Tensor):
            want = torch.full((base_cfg.E,), -1, dtype=torch.int64)
            lo = self.tp_rank * base_cfg.ER
            want[lo : lo + base_cfg.ER] = torch.arange(base_cfg.ER)
            _need(
                torch.equal(expert_map.detach().cpu().to(torch.int64), want),
                "experts are not placed contiguously by rank",
            )
        if getattr(layer.mlp, "_fused_gate_gu_fp8_weight", None) is None:
            gate_dtype = layer.mlp.gate.weight.dtype
            _need(
                gate_dtype in (torch.bfloat16, torch.float32),
                f"unfused router weight dtype {gate_dtype}",
            )

        if mesh is None:
            mesh = build_ep_mesh(axis_name="tp")
        if mesh is None:
            mesh = jax.sharding.Mesh(jax.devices()[: self.tp_size], ("tp",))
        self.mesh = mesh
        self.layer = layer
        self.base_cfg = base_cfg
        self.tiles = K.default_tiles(base_cfg)
        self.tk = K.plan(base_cfg, self.tiles)["tk"]
        self.loop_buckets = _parse_buckets(
            "VLLM_TPU_QWEN35_MTP_MK_BUCKETS", _DEFAULT_LOOP_BUCKETS
        )
        self.verify_buckets = _parse_buckets(
            "VLLM_TPU_QWEN35_MTP_MK_VERIFY_BUCKETS", _DEFAULT_VERIFY_BUCKETS
        )
        _need(
            bool(self.loop_buckets or self.verify_buckets),
            "both bucket lists are empty",
        )

        with torch.no_grad():
            stack = types.SimpleNamespace(layers=[layer], norm=predictor.norm)
            w_flat = pack_weights(stack, base_cfg, self.tp_rank, self.tiles)["w_flat"]
        _need(len(w_flat) == _NUM_LEAVES, f"{len(w_flat)} weight leaves")
        _need(
            w_flat[_NUM_HEAD_LEAVES].data_ptr() == w13.data_ptr()
            and w_flat[_NUM_HEAD_LEAVES + 1].data_ptr() == w2.data_ptr(),
            "expert leaves do not alias w13_weight / w2_weight",
        )
        # The expert leaves are read off the layer in `forward_layer`, so the
        # compiled programs take the parameters themselves.
        self.w_head: list[torch.Tensor] = w_flat[:_NUM_HEAD_LEAVES]
        self.w_tail: list[torch.Tensor] = w_flat[_NUM_HEAD_LEAVES + 2 :]
        packed = self.w_head + self.w_tail
        if packed[0].device.type == "tpu":
            from vllm_torchtpu.utils import synchronize_tensors

            synchronize_tensors(packed)

        self.bound = False
        self.layer_name: str = attn.layer_name
        self.num_pages = 0
        self.page_size = 0
        self._steps: dict[tuple[int, int, int], tuple[K.MKConfig, Any]] = {}
        # Registered before any draft program is built or restored from the
        # compile cache; they only run once `bind_caches` has set the pool.
        self._loop_op: Any = self._build_op(1) if self.loop_buckets else None
        self._verify_op: Any = (
            self._build_op(_NUM_SPEC_TOKENS + 1) if self.verify_buckets else None
        )
        self._step_mix: dict[str, int] = {}
        self._step_mix_t0 = time.monotonic()
        self._step_mix_period_s = float(
            os.environ.get("VLLM_TPU_QWEN35_MK_STATS_PERIOD_S", "60")
        )
        logger.info(
            "Qwen3.5 MTP megakernel initialized: loop_buckets=%s, "
            "verify_buckets=%s, %.1f MiB packed weights (tp_rank=%d/%d)",
            self.loop_buckets,
            self.verify_buckets,
            sum(t.numel() * t.element_size() for t in packed) / 2**20,
            self.tp_rank,
            self.tp_size,
        )

    def bind_caches(self, runner: Any) -> None:
        """Bind the draft layer's unified pool; until then no step runs here."""
        if self.bound:
            return
        pool = _unwrap(self.layer.self_attn.attn.kv_cache)
        _need(
            isinstance(pool, torch.Tensor) and pool.dtype == torch.float8_e4m3fn,
            f"draft KV cache {type(pool)} {getattr(pool, 'dtype', None)} is not "
            "an FP8 tensor",
        )
        _need(
            pool.dim() == 5 and tuple(pool.shape[1:4]) == (2, 64, 4),
            f"draft KV cache shape {tuple(pool.shape)} != (pages, 2, 64, 4, page)",
        )
        _need(
            pool.is_contiguous() and pool.storage_offset() == 0,
            "draft KV cache is not a base buffer",
        )
        raw = getattr(runner, "kv_cache_raw_tensors", None) or []
        _need(
            any(
                isinstance(t, torch.Tensor) and t.data_ptr() == pool.data_ptr()
                for t in raw
            ),
            "draft KV cache is not a unified raw pool",
        )
        self.num_pages = int(pool.shape[0])
        self.page_size = int(pool.shape[-1])
        self.bound = True
        logger.info(
            "Qwen3.5 MTP megakernel bound to %s (num_pages=%d, page_size=%d)",
            self.layer_name,
            self.num_pages,
            self.page_size,
        )

    def _step(self, T: int, B: int, width: int) -> tuple[K.MKConfig, Any]:
        """`(config, flat step)` of `T`-row sequences in a `B`-row bucket over
        `width`-page block tables."""
        key = (T, B, width)
        if key not in self._steps:
            c = dataclasses.replace(
                self.base_cfg,
                T=T,
                B=B,
                ctx=width * self.page_size,
                num_pages=self.num_pages,
                page_size=self.page_size,
                max_pages_per_seq=width,
            )
            tiles = K.default_tiles(c)
            _need(
                K.plan(c, tiles)["tk"] == self.tk,
                f"t{T}b{B} tiles change the packed weight tiling",
            )
            step, _, _, _ = pool_io.make_flat_attn_step(self.mesh, c, tiles)
            self._steps[key] = (c, step)
        return self._steps[key]

    def _build_op(self, T: int) -> Any:
        """Register the `sharded_jax_op` running `T`-row draft steps.

        Its JAX function sees each call's global (rank-folded) shapes and
        picks the bucket and block-table width from them.
        """
        R = self.tp_size
        A = P(self.mesh.axis_names[0])
        buckets = self.loop_buckets if T == 1 else self.verify_buckets
        gather = T > 1
        meta = ["hidden", "block_tables", "seq_lens", "query_start_loc"]
        if gather:
            meta.append("gather_indices")
        names = meta + [f"w{i}" for i in range(_NUM_LEAVES)] + ["pool"]

        def impl(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
            hidden, block_tables, seq_lens, query_start_loc = args[:4]
            gidx = args[4] if gather else None
            leaves = args[len(meta) : len(meta) + _NUM_LEAVES]
            pool = args[-1]
            rows = hidden.shape[0] // R
            knr = seq_lens.shape[0] // R
            width = block_tables.size // R // knr
            _need(width * knr * R == block_tables.size, "ragged block tables")
            c, step = self._step(T, select_bucket(rows, buckets), width)
            _need(rows <= c.M, f"{rows} rows exceed the {c.M}-row step")

            def prep(*xs: jax.Array) -> tuple[jax.Array, ...]:
                return _step_operands(c, *xs[:4], xs[4] if gather else None)

            n_in = len(meta)
            pos, hid, page_table, live, moe_rows = jax.shard_map(
                prep,
                mesh=self.mesh,
                in_specs=(A,) * n_in,
                out_specs=(A,) * 5,
                check_vma=False,
            )(*args[:n_in])
            out, new_pool = step(pos, hid, *leaves, pool, page_table, live, moe_rows)

            def post(o: jax.Array, *g: jax.Array) -> jax.Array:
                if not g:
                    return o[:rows]
                return jnp.take(o, jnp.clip(g[0].astype(jnp.int32), 0, c.M - 1), axis=0)

            post_args = (out, gidx) if gather else (out,)
            out = jax.shard_map(
                post,
                mesh=self.mesh,
                in_specs=(A,) * len(post_args),
                out_specs=A,
                check_vma=False,
            )(*post_args)
            return out, new_pool

        fn_name = f"qwen35_mtp_mk_t{T}"
        src = (
            f"def {fn_name}({', '.join(n + ': jax.Array' for n in names)})"
            " -> tuple[jax.Array, jax.Array]:\n"
            f"    return _impl({', '.join(names)})\n"
        )
        ns: dict[str, Any] = {"jax": jax, "_impl": impl}
        exec(compile(src, f"<{fn_name}>", "exec"), ns)  # noqa: S102

        def fake(hidden: torch.Tensor, *args: torch.Tensor):
            rows = args[3].shape[0] if gather else hidden.shape[0]
            return (
                hidden.new_empty((rows, hidden.shape[1])),
                torch.empty_like(args[-1]),
            )

        return sharded_jax_op(
            f"qwen35_megakernel::mtp_step_t{T}",
            ns[fn_name],
            mesh=self.mesh,
            input_partition_specs=(A,) * len(names),
            output_partition_specs=(A, A),
            donate_argnums=(len(names) - 1,),
            fake=fake,
        )

    def use_for_step(self, attn_metadata: Any, num_rows: int, *, step0: bool) -> bool:
        """Whether a draft step of `num_rows` rows runs on the megakernel.

        `step0` steps must be pure decode-verify steps (`K + 1` rows per
        request), which the caller checks. Counts the decision.
        """
        buckets = self.verify_buckets if step0 else self.loop_buckets
        md = (
            attn_metadata.get(self.layer_name)
            if isinstance(attn_metadata, dict)
            else None
        )
        ok = bool(
            self.bound
            and buckets
            and num_rows <= buckets[-1]
            and md is not None
            and md.query_start_loc.shape[0] == md.seq_lens.shape[0] + 1
            and md.block_tables.numel() % md.seq_lens.shape[0] == 0
        )
        return self.note_step(f"{'s0' if step0 else 'loop'}_n{num_rows}", ok)

    def note_step(self, key: str, run: bool) -> bool:
        """Count a draft-step decision; log the MK/XLA mix periodically."""
        key = f"{'mk' if run else 'xla'}_{key}"
        self._step_mix[key] = self._step_mix.get(key, 0) + 1
        now = time.monotonic()
        if now - self._step_mix_t0 >= self._step_mix_period_s:
            if self.tp_rank == 0:
                total = sum(self._step_mix.values())
                mk = sum(v for k, v in self._step_mix.items() if k.startswith("mk_"))
                logger.info(
                    "Qwen3.5 MTP megakernel step mix over %.0fs: %d/%d MK (%.1f%%) %s",
                    now - self._step_mix_t0,
                    mk,
                    total,
                    100.0 * mk / max(total, 1),
                    dict(sorted(self._step_mix.items())),
                )
            self._step_mix = {}
            self._step_mix_t0 = now
        return run

    def forward_layer(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        gather_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """`mtp.norm(layer(hidden_states))` for the post-`fc` rows, traced into
        the calling draft program; with `gather_indices` (draft step 0), only
        those rows are returned."""
        attn = layer.self_attn.attn
        md = get_forward_context().attn_metadata[attn.layer_name]
        pool = _unwrap(attn.kv_cache)
        experts = _get_routed_experts(layer.mlp)
        leaves = (*self.w_head, experts.w13_weight, experts.w2_weight, *self.w_tail)
        meta = (md.block_tables, md.seq_lens, md.query_start_loc)
        if gather_indices is None:
            out, new_pool = self._loop_op(hidden_states, *meta, *leaves, pool)
        else:
            out, new_pool = self._verify_op(
                hidden_states,
                *meta,
                gather_indices.to(torch.int32),
                *leaves,
                pool,
            )
        # Donated pool + copy_ writeback, as in ragged paged attention: an
        # in-place pool update in the compiled program.
        pool.copy_(new_pool)
        return out
