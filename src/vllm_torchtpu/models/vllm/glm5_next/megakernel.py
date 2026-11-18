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
"""GLM-5.3-Flash decode-owning megakernel adapter for `vllm serve` (TP=8).

vLLM runs prefill through its compiled backbone (chunked and batched); decode
steps of up to `n_slots` greedy requests run the fused 46-layer GLM-5.3-Flash
megakernel through `pallas.jax_op` (as N calls above 8 requests). Weights
exist once per worker process:

- Routed experts: vLLM's TP K-major expert tensors are moved into one stacked
  store per process (`[L, E, K, N]` for target MoE layers, plus a 1-layer store
  for the EAGLE3 MTP layer). vLLM's MoE layers read their layer in place
  through `gmm_v2(rhs_group_base=...)`; the megakernel reads the same buffers
  as `[1, L, E, K, N]`.
- Non-expert weights (~1.9 GiB per rank) are loaded once from the checkpoint in
  the megakernel's per-channel FP8 / fused-projection layout.

Per-request decode state lives in megakernel buffers indexed by a slot in
`[0, n_slots)`:
- MLA latents of all target MLA layers and the MTP layer live in
  `KernelKVCachePool` superpages; `Glm5NextMLAAttention._latent_sink` stashes
  each prefill chunk's latents and `flush_latents` writes them to the pool.
- KDA conv and recurrent states are copied once from vLLM's unified pool when a
  request's prompt completes (`take_ownership`).
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.glm53_megakernel import kernel as mk
from vllm_torchtpu.kernels.glm53_megakernel.config import Glm53MegakernelConfig
from vllm_torchtpu.kernels.glm53_megakernel.fastpath_ops import (
    copy_subpages_local,
    flush_latents_local,
    kda_state_from_pool_local,
)
from vllm_torchtpu.kernels.glm53_megakernel.load import (
    Glm53Checkpoint,
    to_per_channel_fp8,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.adapter.decode_owner import DecodeOwner, DecodeResult
from vllm_torchtpu.megakernel.core.runtime.contract import Decline
from vllm_torchtpu.utils import synchronize_device, synchronize_tensors

logger = init_logger(__name__)

MAX_SLOTS = 8
KERNEL_BATCH = 8
KV_GROUP = 8
SPEC_K = 3
VERIFY = SPEC_K + 1


def _tpu_sync() -> None:
    synchronize_tensors(wait=True)
    synchronize_device()


def _resolve_checkpoint_dir(model_ref: str) -> Path | None:
    """Resolve a local path or cached Hugging Face repo ID to a directory."""
    p = Path(model_ref)
    if p.is_dir():
        return p
    if "://" in model_ref:
        return None
    try:
        import importlib

        hf_hub = importlib.import_module("huggingface_hub")
        return Path(hf_hub.snapshot_download(model_ref, local_files_only=True))
    except Exception:  # noqa: BLE001
        return None


def decode_buckets(n_slots: int) -> tuple[int, ...]:
    """Decode batch buckets: 1, 2, 4, 8, then 16 and 32 up to `n_slots`."""
    return tuple(b for b in (1, 2, 4, 8, 16, 32) if b <= n_slots)


@dataclasses.dataclass
class ExpertStore:
    """One stacked copy of a group of MoE layers' routed experts (TP K-major)."""

    w13: torch.Tensor  # [L, E, H, 2I/W] fp8
    w13_s: torch.Tensor  # [L, E, 1, 1, 2I/W] f32
    w2: torch.Tensor  # [L, E, I/W, H] fp8
    w2_s: torch.Tensor  # [L, E, 1, 1, H] f32

    @property
    def num_layers(self) -> int:
        return int(self.w13.shape[0])


_EXPERT_PARAM_NAMES = (
    "w13_weight",
    "w13_weight_scale_inv",
    "w2_weight",
    "w2_weight_scale_inv",
)


def build_expert_store(
    routed_layers: list[Any], on_layer_moved: Any = None
) -> ExpertStore:
    """Move every layer's `w13/w2(+scales)` into one stacked store in place.

    `on_layer_moved(store, i)` runs right after layer `i` moves so the caller
    can point that layer's MoE op at the store before the next layer moves.
    Each layer's original parameter is replaced with an empty placeholder.
    """

    def put(st: jax.Array, w: jax.Array, idx: jax.Array) -> jax.Array:
        return jax.lax.dynamic_update_index_in_dim(st, w.astype(st.dtype), idx[0], 0)

    put_op = pallas.jax_op("glm53_mk::expert_store_put", put, donate_argnums=(0,))
    first = routed_layers[0]
    n = len(routed_layers)
    dev = first.w13_weight.device
    stores = {
        k: torch.empty(
            (n, *getattr(first, k).shape),
            dtype=getattr(first, k).dtype,
            device=dev,
        )
        for k in _EXPERT_PARAM_NAMES
    }
    store = ExpertStore(
        w13=stores["w13_weight"],
        w13_s=stores["w13_weight_scale_inv"],
        w2=stores["w2_weight"],
        w2_s=stores["w2_weight_scale_inv"],
    )
    for i, layer in enumerate(routed_layers):
        idx = torch.tensor([i], dtype=torch.int32, device=dev)
        for k in _EXPERT_PARAM_NAMES:
            src = getattr(layer, k)
            if (
                tuple(src.shape) != tuple(stores[k].shape[1:])
                or src.dtype != stores[k].dtype
            ):
                raise ValueError(
                    f"MoE layer {i} {k}: {tuple(src.shape)} {src.dtype} != "
                    f"store {stores[k].shape}"
                )
            new = put_op(stores[k], src.data, idx)
            stores[k].copy_(new)
            del new
            setattr(
                layer,
                k,
                torch.nn.Parameter(
                    torch.empty(0, dtype=src.dtype, device=dev),
                    requires_grad=False,
                ),
            )
            del src
        if on_layer_moved is not None:
            on_layer_moved(store, i)
        _tpu_sync()
    return store


def store_as_kernel_weights(store: ExpertStore, prefix: str) -> dict[str, torch.Tensor]:
    """View `store` as megakernel K-major expert operands `[1, L, E, ...]`."""
    s13 = store.w13_s
    s2 = store.w2_s
    return {
        f"{prefix}_expert_gate_up_q": store.w13.unsqueeze(0),
        f"{prefix}_expert_gate_up_s": s13.view(
            1, s13.shape[0], s13.shape[1], 1, s13.shape[-1]
        ),
        f"{prefix}_expert_down_q": store.w2.unsqueeze(0),
        f"{prefix}_expert_down_s": s2.view(
            1, s2.shape[0], s2.shape[1], 1, s2.shape[-1]
        ),
    }


@dataclasses.dataclass
class SlotState:
    slot: int
    latent_upto: int = 0
    kda_ready: bool = False
    owned: bool = False
    next_verify: list[int] | None = None


class _StepTimer:
    """Host-side phase timer for megakernel decode steps (`timing=1`)."""

    def __init__(self) -> None:
        import time

        self._now = time.perf_counter
        self.sums: dict[str, dict[str, float]] = {"pure": {}, "mixed": {}}
        self.counts = {"pure": 0, "mixed": 0}
        self.walls: dict[str, list[float]] = {"pure": [], "mixed": []}
        self.cur: dict[str, float] = {}
        self.t: float | None = None
        self.t0: float | None = None

    def start(self) -> None:
        self.t = self.t0 = self._now()
        self.cur = {}

    def mark(self, name: str) -> None:
        if self.t is None:
            return
        now = self._now()
        self.cur[name] = self.cur.get(name, 0.0) + now - self.t
        self.t = now

    def end(self, kind: str) -> None:
        if self.t is None or self.t0 is None:
            return
        self.mark("assemble")
        for k, v in self.cur.items():
            self.sums[kind][k] = self.sums[kind].get(k, 0.0) + v
        self.walls[kind].append(self._now() - self.t0)
        self.counts[kind] += 1
        self.t = None
        n = self.counts["pure"] + self.counts["mixed"]
        if n % 256 == 0:
            for kd in ("pure", "mixed"):
                c = self.counts[kd]
                if not c:
                    continue
                w = np.asarray(self.walls[kd][-4096:]) * 1e3
                phases = {
                    k: round(1e3 * v / c, 3) for k, v in sorted(self.sums[kd].items())
                }
                logger.info(
                    "GLM53 megakernel %s steps=%d mean phase ms %s | "
                    "wall ms mean %.2f p50 %.2f p90 %.2f max %.1f",
                    kd,
                    c,
                    phases,
                    w.mean(),
                    np.percentile(w, 50),
                    np.percentile(w, 90),
                    w.max(),
                )


def unpack_heads(flat: np.ndarray, b: int, spec: bool) -> tuple[np.ndarray, ...]:
    """Inverse of the step op's head packing."""
    k = mk.TOPK_OUT
    if spec:
        shapes = [
            (b, VERIFY, k),
            (b, VERIFY, k),
            (b, VERIFY),
            (b,),
            (b, VERIFY),
            (b,),
        ]
    else:
        shapes = [(b, k), (b, k), (b,)]
    flat = np.asarray(flat).reshape(-1)
    out: list[np.ndarray] = []
    o = 0
    for i, sh in enumerate(shapes):
        sz = int(np.prod(sh))
        part = np.ascontiguousarray(flat[o : o + sz]).reshape(sh)
        out.append(part.view(np.float32) if i == 0 else part)
        o += sz
    if o != flat.size:
        raise ValueError(f"packed head vector has {flat.size} entries, expected {o}")
    return tuple(out)


class _HostHeads:
    """Unpacked host heads of an N-call step, shaped like a device result."""

    def __init__(self, heads: tuple[np.ndarray, ...]) -> None:
        self.heads = heads

    def cpu(self) -> _HostHeads:
        return self

    def numpy(self) -> list[tuple[np.ndarray, ...]]:
        return [self.heads]


class SlotTable:
    def __init__(self, n: int) -> None:
        self.free = list(range(n - 1, -1, -1))
        self.by_req: dict[str, SlotState] = {}

    def get(self, req_id: str) -> SlotState | None:
        return self.by_req.get(req_id)

    def acquire(self, req_id: str) -> SlotState | None:
        st = self.by_req.get(req_id)
        if st is not None:
            return st
        if not self.free:
            return None
        st = SlotState(slot=self.free.pop())
        self.by_req[req_id] = st
        return st

    def release(self, req_id: str) -> SlotState | None:
        st = self.by_req.pop(req_id, None)
        if st is not None:
            self.free.append(st.slot)
        return st


def _explicit_arity(body: Any, n_in: int, n_out: int) -> Any:
    args = ", ".join(f"a{i}: jax.Array" for i in range(n_in))
    ret = (
        "jax.Array" if n_out == 1 else "tuple[" + ", ".join(["jax.Array"] * n_out) + "]"
    )
    src = (
        f"def f({args}) -> {ret}:\n"
        f"    return body({', '.join(f'a{i}' for i in range(n_in))})\n"
    )
    ns: dict[str, Any] = {"body": body, "jax": jax}
    exec(src, ns)  # noqa: S102
    return ns["f"]


def _build_sharded_step_op(
    name: str,
    fn: Any,
    *,
    mesh: Any,
    input_partition_specs: tuple[jax.sharding.PartitionSpec, ...],
    output_partition_specs: tuple[jax.sharding.PartitionSpec, ...],
    donate_argnums: tuple[int, ...],
    head_width: int,
) -> Any:
    """Register an SPMD decode-step custom op with explicit `out_shardings`.

    `torch_tpu`'s `pallas.jax_op` wraps `fn` in `jax.jit` without
    `out_shardings`. Because the decode-step wrapper concatenates head tensors
    outside `shard_map` while donating sharded state buffers (`conv`, `rec`,
    `kvq`, `kvs`), omitting `out_shardings` on the outer `jax.jit` drops both
    the `@main` result shardings (`sdy.sharding`) and XLA input-output buffer
    aliasing (`tf.aliasing_output`), causing GSPMD to all-gather and replicate
    all outputs across TP ranks. Supplying `out_shardings` on `jax.jit` and
    `output_partition_specs` on `JaxCallable` preserves both in-place state
    aliasing and per-rank output shapes.
    """
    from torch_tpu._internal.pallas import pallas as pallas_impl

    signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
    pallas_impl._verify_signature(signature)
    out_shardings = tuple(
        jax.sharding.NamedSharding(mesh, spec) for spec in output_partition_specs
    )
    jit_fn = jax.jit(
        fn,
        donate_argnums=donate_argnums,
        out_shardings=out_shardings,
        keep_unused=True,
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}",
        [],
        {
            "donate_argnums": donate_argnums,
            "output_partition_specs": str(output_partition_specs),
        },
    )
    wrapped_fn = cast(Any, pallas_impl.JaxCallable)(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=output_partition_specs,
        static_argnums=(),
        donate_argnums=list(donate_argnums),
    )
    ret_schema_str = pallas_impl._map_jax_return_to_schema_str(
        signature.return_annotation
    )
    schema_str = pallas_impl._generate_torch_schema(
        wrapped_fn.__signature__, ret_schema_str, mutates_args=()
    )
    op = cast(Any, torch.library.custom_op)(
        name, wrapped_fn, mutates_args=(), schema=schema_str
    )

    def _fake(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        conv, rec, kvq, kvs = args[-7:-3]
        return (
            conv.new_empty((1, head_width), dtype=torch.int32),
            torch.empty_like(conv),
            torch.empty_like(rec),
            torch.empty_like(kvq),
            torch.empty_like(kvs),
        )

    op.register_fake(_fake)
    return op


def _to_torch(a: Any) -> torch.Tensor:
    """Host numpy (bf16/fp8 via ml_dtypes) -> TPU torch tensor (load time)."""
    a = np.array(a, copy=True, order="C")
    name = str(a.dtype)
    if name == "bfloat16":
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16).to("tpu")
    if name == "float8_e4m3fn":
        return torch.from_numpy(a.view(np.uint8)).view(torch.float8_e4m3fn).to("tpu")
    return torch.from_numpy(a).to("tpu")


class Glm53DecodeOwner:
    """Decode-owning megakernel program for one vLLM TP=8 worker process."""

    def __init__(
        self,
        runner: Any,
        *,
        max_req: int = MAX_SLOTS,
        cache_kpages: int | None = None,
        profile_scopes: bool = False,
        debug: bool = False,
        timing: bool = False,
    ) -> None:
        from vllm.distributed import parallel_state

        from vllm_torchtpu.distributed.mesh_utils import (
            get_cp_group_layout,
            get_stage_mesh,
        )

        if max_req not in (8, 16, 32):
            raise ValueError(f"max_req={max_req}: expected 8, 16 or 32")

        self.runner = runner
        vcfg = runner.vllm_config
        self.model_dir = str(vcfg.model_config.model)
        self.tp_group = parallel_state.get_tp_group()
        self.world = int(self.tp_group.world_size)
        self.rank = int(self.tp_group.rank_in_group)
        layout = get_cp_group_layout(self.tp_group)
        self.device_ids = list(layout.device_ids)
        # `get_stage_mesh` orders `mesh.devices` by TP rank and `torch_tpu`
        # passes `mesh.devices` as the PJRT `DeviceAssignment`, so
        # `pl.DeviceIdType.MESH` coordinate `q` is already logical TP rank `q`.
        self.dev_table = list(range(self.world))
        self.mesh = get_stage_mesh(
            ("tp",), (self.world,), self.tp_group, layout.device_ids
        )
        if hasattr(vcfg, "compilation_config"):
            from vllm_torchtpu.platforms import TpuPlatform

            TpuPlatform.update_block_size_for_backend(vcfg)
        self.block_size = int(vcfg.cache_config.block_size)
        hf_cfg = getattr(vcfg.model_config, "hf_config", None)
        if hf_cfg is not None and hasattr(hf_cfg, "to_dict"):
            self.cfg = Glm53MegakernelConfig.from_dict(
                hf_cfg.to_dict(), page_size=self.block_size
            )
        else:
            ckpt_dir = _resolve_checkpoint_dir(self.model_dir)
            if ckpt_dir is None:
                raise FileNotFoundError(
                    f"Cannot resolve local checkpoint directory for {self.model_dir!r}"
                )
            self.cfg = Glm53MegakernelConfig.from_checkpoint(
                ckpt_dir, page_size=self.block_size
            )
        spec = vcfg.speculative_config
        self.spec = spec is not None and int(spec.num_speculative_tokens) > 0
        if self.spec and int(spec.num_speculative_tokens) != SPEC_K:
            raise ValueError(
                f"megakernel decode supports num_speculative_tokens={SPEC_K}"
            )
        self.drafts = bool(self.spec)
        self.max_model_len = int(vcfg.model_config.max_model_len)
        self.profile_scopes = bool(profile_scopes)
        self.debug = bool(debug)
        self.timer = _StepTimer() if timing else None
        self._chunks: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._chunk_saves: list[list[tuple[int, int]]] = []
        self._single_chunk_spec = False
        self.n_slots = int(max_req)
        self._cache_kpages_opt = cache_kpages
        self.slots = SlotTable(self.n_slots)
        self.kw: dict[str, torch.Tensor] = {}
        self._ops: dict[Any, Any] = {}
        self.ready = False
        self.num_mla = len(self.cfg.mla_layers) + (1 if self.spec else 0)
        self._mla_index = {lid: i for i, lid in enumerate(self.cfg.mla_layers)}
        if self.spec:
            self._mla_index[self.cfg.num_hidden_layers] = len(self.cfg.mla_layers)

    def owns(self, req_id: str) -> bool:
        st = self.slots.get(req_id)
        return st is not None and st.owned

    # ---------------- load time ----------------

    def _bind_latent_sinks(self, model: Any, draft_model: Any | None) -> None:
        for lid in self.cfg.mla_layers:
            attn = model.model.layers[lid].self_attn
            attn._latent_sink = functools.partial(self.stash_latent, lid)
        if self.spec and draft_model is not None:
            mtp_attn = draft_model.model._mtp_layers[0].mtp_block.self_attn
            mtp_attn._latent_sink = functools.partial(
                self.stash_latent, self.cfg.num_hidden_layers
            )

    def load(self, model: Any, draft_model: Any | None) -> None:
        """Share routed experts, load non-expert weights, allocate decode state."""
        cfg = self.cfg

        # 1. Shared routed experts (target MoE layers, then the MTP layer's store).
        moes = [
            layer.mlp
            for layer in model.model.layers
            if hasattr(layer.mlp, "_fused_moe")
        ]
        expected_moes = cfg.num_hidden_layers - cfg.first_k_dense_replace
        if len(moes) != expected_moes or any(m._fused_moe() is None for m in moes):
            raise RuntimeError(
                "megakernel decode needs the fused GLM-5.3 MoE op on every MoE layer "
                f"({expected_moes} expected)"
            )
        if any(m._fused_moe()._op_kwargs.get("use_ep", False) for m in moes):
            raise ValueError(
                "megakernel decode requires TP-sharded experts "
                "(serve without --enable-expert-parallel)"
            )
        ckpt_dir = _resolve_checkpoint_dir(self.model_dir)
        if ckpt_dir is None:
            raise FileNotFoundError(
                "megakernel decode requires a local checkpoint directory, "
                f"got {self.model_dir!r}"
            )
        self._bind_latent_sinks(model, draft_model)
        routed = [getattr(m.experts, "routed_experts", m.experts) for m in moes]
        store = build_expert_store(
            routed,
            lambda st, i: moes[i]._fused_moe().use_expert_store(st, i),
        )
        self.store = store
        self.kw.update(store_as_kernel_weights(store, "moe"))
        if self.spec:
            if draft_model is None:
                raise RuntimeError("megakernel spec decode requires a draft_model")
            mtp_layer = draft_model.model._mtp_layers[0]
            mtp_moe = mtp_layer.mtp_block.mlp
            if mtp_moe._fused_moe() is None:
                raise RuntimeError("MTP MoE layer has no fused op")
            mstore = build_expert_store(
                [getattr(mtp_moe.experts, "routed_experts", mtp_moe.experts)],
                lambda st, i: mtp_moe._fused_moe().use_expert_store(st, i),
            )
            self.mtp_store = mstore
            self.kw.update(store_as_kernel_weights(mstore, "mtp_moe"))

        # 2. Non-expert weights for this rank in megakernel layout.
        ckpt = Glm53Checkpoint(ckpt_dir)
        (w_np,) = ckpt.extract_all_ranks(
            world_size=self.world,
            include_routed_experts=False,
            only_rank=self.rank,
        )
        if self.spec:
            (mtp_np,) = ckpt.extract_mtp_ranks(
                world_size=self.world,
                include_routed_experts=False,
                only_rank=self.rank,
            )
            w_np.update(mtp_np)
        all_names = mk.KERNEL_WEIGHT_NAMES + (
            mk.KERNEL_MTP_WEIGHT_NAMES if self.spec else ()
        )
        expert_names = mk.EXPERT_NAMES + mk.MTP_EXPERT_NAMES
        names = [n for n in all_names if n not in expert_names]
        world = self.world

        def prep(w: dict[str, jax.Array]) -> dict[str, jax.Array]:
            w = to_per_channel_fp8(cfg, w)
            return mk.prepare_kernel_weights_local(cfg, w, world, experts_are_tp=True)

        cpu0 = jax.devices("cpu")[0]
        with jax.default_device(cpu0):
            out = jax.jit(prep)({k: jnp.asarray(v) for k, v in w_np.items()})
        del w_np
        for n in names:
            self.kw[n] = _to_torch(jax.device_get(out[n]))[None]
        del out

        # 3. Decode state: latent superpage pool, KDA conv / rec, page table.
        # Pad `n_pages` by the verify + MTP draft lookahead horizon (`VERIFY + SPEC_K`)
        # so positions near `max_model_len` never index past `self.kpt` or the last
        # superpage when speculative decoding advances up to `pos + 5`.
        page = cfg.page_size
        prow = -(-page // 32) * 32
        kprow = -(-(KV_GROUP * prow) // 128) * 128
        lookahead = VERIFY + SPEC_K if self.spec else 0
        self.n_pages = -(-(self.max_model_len + lookahead) // page)
        grp = world * KV_GROUP
        self.max_kp = max(1, -(-self.n_pages // grp))
        self.base_kpages = self.n_slots * self.max_kp + 2
        self.prefix_cache_enabled = bool(
            getattr(
                getattr(self.runner, "cache_config", None),
                "enable_prefix_caching",
                False,
            )
        )
        default_cache_kpages = 112 if self.n_slots <= 8 else 56
        cache_kp = (
            int(self._cache_kpages_opt)
            if self._cache_kpages_opt is not None
            else default_cache_kpages
        )
        self.cache_kpages = cache_kp if self.prefix_cache_enabled else 0
        self.num_kpages = self.base_kpages + self.cache_kpages
        self._cache_subpage_base = self.base_kpages * KV_GROUP
        self._cache_subpage_cap = self.cache_kpages * KV_GROUP
        self._block_to_subpage: dict[int, int] = {}
        self._subpage_to_block: dict[int, int] = {}
        self._cache_cursor = 0
        self._rank_cached_blocks: list[dict[int, int]] = [{} for _ in range(world)]
        self._rank_subpage_to_block: list[dict[int, int]] = [{} for _ in range(world)]
        self._rank_cache_cursors: list[int] = [0] * world
        self._mla_group_id: int | None = None
        lk = len(cfg.kda_layers)
        hk = cfg.kda_num_heads // world
        dk = cfg.kda_head_dim
        kv = cfg.kv_lora_rank
        dev = torch.device("tpu")
        self.kvq = torch.zeros(
            (1, self.num_mla, self.num_kpages, kprow // 4, 2 * kv),
            dtype=torch.int32,
            device=dev,
        )
        self.kvs = torch.zeros(
            (1, self.num_mla, self.num_kpages, 8, kprow),
            dtype=torch.float32,
            device=dev,
        )
        self.conv = torch.zeros(
            (1, lk, self.n_slots, cfg.kda_conv_kernel_size - 1, 3 * hk * dk),
            dtype=torch.bfloat16,
            device=dev,
        )
        self.rec = torch.zeros(
            (1, lk, self.n_slots, hk, dk, dk),
            dtype=torch.float32,
            device=dev,
        )
        kpt = (1 + np.arange(self.n_slots * self.max_kp, dtype=np.int32)).reshape(
            self.n_slots, self.max_kp
        )
        self.kpt = torch.from_numpy(kpt).to(dev)
        self._dev = dev
        self.rank_t = torch.tensor([self.rank], dtype=torch.int32, device=dev)
        self.dev_t = torch.tensor(self.dev_table, dtype=torch.int32, device=dev)
        self.weight_names = list(all_names)
        missing = [n for n in self.weight_names if n not in self.kw]
        if missing:
            raise RuntimeError(f"megakernel weights missing: {missing}")
        self.alloc_stash(int(self.runner.max_num_tokens))
        _tpu_sync()
        free, limit = torch.accelerator.get_memory_info(dev)
        logger.info(
            "GLM53 megakernel decode ready | rank=%d dev=%s spec=%s | "
            "HBM used %.2f GiB of %.2f GiB | latent pool %.2f GiB, kda %.2f GiB",
            self.rank,
            self.device_ids,
            self.spec,
            (limit - free) / 2**30,
            limit / 2**30,
            (self.kvq.numel() * 4 + self.kvs.numel() * 4) / 2**30,
            (self.conv.numel() * 2 + self.rec.numel() * 4) / 2**30,
        )
        self.ready = True

    # ---------------- prefill side outputs ----------------

    def alloc_stash(self, max_tokens: int) -> None:
        """Per-MLA-layer latent stash `[Lm, max_tokens, kv]` bf16."""
        self.stash_tokens = int(max_tokens)
        self.stash = torch.zeros(
            (self.num_mla, self.stash_tokens, self.cfg.kv_lora_rank),
            dtype=torch.bfloat16,
            device=torch.device("tpu"),
        )

        def put(stash: jax.Array, latent: jax.Array, idx: jax.Array) -> jax.Array:
            return jax.lax.dynamic_update_slice(
                stash, latent.astype(stash.dtype)[None], (idx[0], 0, 0)
            )

        op = pallas.jax_op("glm53_mk::stash_latent", put, donate_argnums=(0,))
        op.register_fake(lambda stash, latent, idx: torch.empty_like(stash))
        self._stash_op = op
        self._stash_idx = {
            i: torch.tensor([i], dtype=torch.int32, device=torch.device("tpu"))
            for i in range(self.num_mla)
        }

    def stash_latent(self, layer_id: int, kv_latent: torch.Tensor) -> None:
        """Called from `Glm5NextMLAAttention.forward` with `[T, kv]` of this step."""
        i = self._mla_index.get(layer_id)
        if i is None or not self.ready:
            return
        new = self._stash_op(self.stash, kv_latent.contiguous(), self._stash_idx[i])
        self.stash.copy_(new)

    def _get_flush_op(self, layers: tuple[int, ...], n_tokens: int) -> Any:
        key = ("flush", layers, n_tokens)
        op = self._ops.get(key)
        if op is not None:
            return op
        cfg, world = self.cfg, self.world

        def body(
            kvq: jax.Array,
            kvs: jax.Array,
            stash: jax.Array,
            slot_pos: jax.Array,
            kpt: jax.Array,
            rank: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return flush_latents_local(
                kvq,
                kvs,
                stash,
                slot_pos[0],
                slot_pos[1],
                kpt,
                rank,
                layers=layers,
                n_tokens=n_tokens,
                page_size=cfg.page_size,
                world=world,
                tok_idx=slot_pos[2],
            )

        op = pallas.jax_op(
            f"glm53_mk::flush_{layers[0]}_{len(layers)}_{n_tokens}",
            _explicit_arity(body, 6, 2),
            donate_argnums=(0, 1),
        )
        self._ops[key] = op
        return op

    def _flush_buckets(self) -> list[int]:
        out: list[int] = []
        b = 16
        while b < self.stash_tokens:
            out.append(b)
            if b == 1024 and self.stash_tokens > 1088:
                out.append(1088)
            b *= 2
        return out + [self.stash_tokens]

    def flush_latents(
        self,
        tok_slot: np.ndarray,
        tok_pos: np.ndarray,
        n_tokens: int,
        *,
        mtp: bool,
        all_layers: bool = False,
    ) -> None:
        """Write the stashed latents of this forward's prompt tokens into the pool."""
        valid = tok_slot[:n_tokens] >= 0
        if not np.any(valid):
            return
        page = self.cfg.page_size
        owners = (tok_pos[:n_tokens] // page) % self.world
        max_m = max(int(np.sum(valid & (owners == r))) for r in range(self.world))
        if max_m == 0:
            return
        idx = np.flatnonzero(valid & (owners == self.rank)).astype(np.int32)
        if self.debug:
            st = self.stash[:, :n_tokens].float().abs().sum(dim=(1, 2)).cpu().tolist()
            logger.info(
                "MKDBG flush mtp=%s all=%s n=%d mine=%d slots=%s pos=%s..%s "
                "stash_abs_per_layer=%s",
                mtp,
                all_layers,
                n_tokens,
                int(idx.size),
                sorted(set(tok_slot[:n_tokens].tolist())),
                int(tok_pos[0]),
                int(tok_pos[n_tokens - 1]),
                [round(x, 1) for x in st],
            )
        bucket = next(b for b in self._flush_buckets() if b >= max_m)
        sp = np.zeros((3, bucket), np.int32)
        sp[0] = -1
        if idx.size:
            sp[0, : idx.size] = tok_slot[idx]
            sp[1, : idx.size] = tok_pos[idx]
            sp[2, : idx.size] = idx
        n_tgt = len(self.cfg.mla_layers)
        if all_layers:
            layers = tuple(range(self.num_mla))
        elif mtp:
            layers = (n_tgt,)
        else:
            layers = tuple(range(n_tgt))
        op = self._get_flush_op(layers, bucket)
        self.kvq, self.kvs = op(
            self.kvq,
            self.kvs,
            self.stash,
            torch.from_numpy(sp).to(self._dev),
            self.kpt,
            self.rank_t,
        )
        # Enqueue the donated-buffer update onto PJRT without stalling the host.
        synchronize_tensors([self.kvq, self.kvs], wait=False)
        if self.debug:
            _tpu_sync()
            nz = (self.kvq != 0).sum(dim=(0, 2, 3, 4)).cpu().tolist()
            logger.info("MKDBG pool nonzero words per layer=%s", nz)

    def _get_copy_subpages_op(self, n_pages: int) -> Any:
        key = ("copy_subpages", n_pages)
        op = self._ops.get(key)
        if op is not None:
            return op

        def body(
            kvq: jax.Array, kvs: jax.Array, pairs: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return copy_subpages_local(kvq, kvs, pairs, kv_group=KV_GROUP)

        op = pallas.jax_op(
            f"glm53_mk::copy_subpages_{n_pages}",
            _explicit_arity(body, 3, 2),
            donate_argnums=(0, 1),
        )
        self._ops[key] = op
        return op

    def _invalidate_cached_block(
        self, block_id: int, *, keep_rank: int | None = None
    ) -> None:
        """Evict recycled `block_id` from per-rank latent caches."""
        if block_id <= 0 or self._cache_subpage_cap <= 0:
            return
        for r, b2s in enumerate(self._rank_cached_blocks):
            if r == keep_rank:
                continue
            c_sub = b2s.pop(block_id, None)
            if c_sub is not None:
                s2b = self._rank_subpage_to_block[r]
                if s2b.get(c_sub) == block_id:
                    s2b.pop(c_sub, None)

    def _alloc_cache_subpage_for_rank(self, owner_rank: int, block_id: int) -> int:
        b2s = self._rank_cached_blocks[owner_rank]
        c_sub = b2s.get(block_id)
        if c_sub is not None:
            return c_sub
        cur = self._rank_cache_cursors[owner_rank]
        c_sub = self._cache_subpage_base + cur
        self._rank_cache_cursors[owner_rank] = (cur + 1) % self._cache_subpage_cap
        s2b = self._rank_subpage_to_block[owner_rank]
        old_blk = s2b.get(c_sub)
        if old_blk is not None:
            b2s.pop(old_blk, None)
        b2s[block_id] = c_sub
        s2b[c_sub] = block_id
        if owner_rank == self.rank:
            self._block_to_subpage = b2s
            self._subpage_to_block = s2b
            self._cache_cursor = self._rank_cache_cursors[owner_rank]
        return c_sub

    def _can_restore_prefix(self, bt_row: Any, c0: int) -> bool:
        """Check in lockstep that every prefix block `[0, c0 // page)` is cached."""
        page = (
            self.cfg.page_size if hasattr(self, "cfg") else getattr(self, "page", 136)
        )
        if c0 <= 0:
            return True
        if c0 % page != 0 or self._cache_subpage_cap <= 0:
            return False
        return all(
            int(bt_row[lp]) in self._rank_cached_blocks[lp % self.world]
            for lp in range(c0 // page)
        )

    def copy_subpages(self, pairs: list[tuple[int, int]]) -> None:
        if not pairs or not self.prefix_cache_enabled:
            return
        dedup = list(
            {dst: (src, dst) for src, dst in pairs if src >= 0 and dst >= 0}.values()
        )
        if not dedup:
            return
        buckets = (16, 64, 128)
        for off in range(0, len(dedup), buckets[-1]):
            chunk = dedup[off : off + buckets[-1]]
            b = next(x for x in buckets if x >= len(chunk))
            arr = np.full((2, b), -1, np.int32)
            for j, (s, d) in enumerate(chunk):
                arr[0, j], arr[1, j] = s, d
            op = self._get_copy_subpages_op(b)
            self.kvq, self.kvs = op(
                self.kvq, self.kvs, torch.from_numpy(arr).to(self._dev)
            )
            # Enqueue the donated-buffer copy onto PJRT without stalling the host.
            synchronize_tensors([self.kvq, self.kvs], wait=False)

    # ---------------- KDA state handoff ----------------

    def _get_kda_op(self, n: int, pools_meta: tuple[Any, ...]) -> Any:
        key = ("kda", n, pools_meta)
        op = self._ops.get(key)
        if op is not None:
            return op
        cfg = self.cfg
        hk = cfg.kda_num_heads // self.world
        n_pools = len(self._kda_pool_tensors)
        pool_of_layer = self._kda_pool_of_layer

        def body(*a: jax.Array) -> tuple[jax.Array, jax.Array]:
            pools = list(a[:n_pools])
            block_ids, conv, rec, slots = a[n_pools:]
            return kda_state_from_pool_local(
                [pools[pool_of_layer[i]] for i in range(len(pool_of_layer))],
                block_ids,
                conv,
                rec,
                slots,
                num_heads=hk,
                head_dim=cfg.kda_head_dim,
                kernel_size=cfg.kda_conv_kernel_size,
                pool_block_tokens=self.block_size,
            )

        op = pallas.jax_op(
            f"glm53_mk::kda_handoff_{n}",
            _explicit_arity(body, n_pools + 4, 2),
            donate_argnums=(n_pools + 1, n_pools + 2),
        )
        self._ops[key] = op
        return op

    def bind_kda_pools(self, model: Any) -> None:
        """Record each KDA layer's vLLM pool tensor after `initialize_kv_cache`."""
        pools: list[torch.Tensor] = []
        pool_of_layer: list[int] = []
        names: list[str] = []
        ptrs: dict[int, int] = {}
        for lid in self.cfg.kda_layers:
            attn = model.model.layers[lid].self_attn
            kv = getattr(attn, "kv_cache", None)
            if kv is None or len(kv) != 1:
                raise RuntimeError(
                    f"KDA layer {lid}: megakernel decode needs the unified pool "
                    "(len(kv_cache) == 1)"
                )
            t = kv[0]
            p = ptrs.get(t.data_ptr())
            if p is None:
                p = len(pools)
                ptrs[t.data_ptr()] = p
                pools.append(t)
            pool_of_layer.append(p)
            names.append(
                attn.prefix
                if hasattr(attn, "prefix")
                else f"model.layers.{lid}.self_attn"
            )
        self._kda_pool_tensors = pools
        self._kda_pool_of_layer = tuple(pool_of_layer)
        self._kda_layer_names = names

    def kda_handoff(self, req_slots: list[tuple[int, list[int]]]) -> None:
        """Copy KDA states from vLLM's unified pool into megakernel slots."""
        n = len(req_slots)
        bucket = 1 << max(0, int(n - 1).bit_length())
        slots = np.full((bucket,), self.n_slots - 1, np.int32)
        ids = np.zeros((bucket, len(self.cfg.kda_layers)), np.int32)
        for i, (s, b) in enumerate(req_slots):
            slots[i], ids[i] = s, b
        for i in range(n, bucket):
            slots[i], ids[i] = slots[0], ids[0]
        op = self._get_kda_op(bucket, ())
        dev = torch.device("tpu")
        self.conv, self.rec = op(
            *self._kda_pool_tensors,
            torch.from_numpy(ids).to(dev),
            self.conv,
            self.rec,
            torch.from_numpy(slots).to(dev),
        )
        # Enqueue the donated-buffer handoff onto PJRT without stalling the host.
        synchronize_tensors([self.conv, self.rec], wait=False)
        if self.debug:
            _tpu_sync()
            logger.info(
                "MKDBG kda handoff slots=%s blocks=%s conv_abs=%s rec_abs=%s",
                slots[:n].tolist(),
                ids[:n, :3].tolist(),
                [
                    round(float(self.conv[0, :, int(x)].float().abs().sum()), 2)
                    for x in slots[:n]
                ],
                [
                    round(float(self.rec[0, :, int(x)].abs().sum()), 2)
                    for x in slots[:n]
                ],
            )

    # ---------------- decode step ----------------

    def count_custom_calls(self, batch: int) -> int:
        """`tpu_custom_call` count of the lowered decode step for `batch`."""
        make = mk.make_spec_decode_step if self.spec else mk.make_decode_step
        step = make(
            self.cfg,
            self.mesh,
            batch=batch,
            max_batch=self.n_slots,
            n_pages=self.n_pages,
            profile_scopes=getattr(self, "profile_scopes", False),
            experts_kmajor=True,
            keep_rank_axis=True,
            side_effects=False,
        )
        tp = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec("tp"))
        rep = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
        _to_jnp = {
            torch.bfloat16: jnp.bfloat16,
            torch.float32: jnp.float32,
            torch.int32: jnp.int32,
            torch.float8_e4m3fn: jnp.float8_e4m3fn,
        }

        def g(t: torch.Tensor) -> jax.ShapeDtypeStruct:
            return jax.ShapeDtypeStruct(
                (self.world, *t.shape[1:]), _to_jnp[t.dtype], sharding=tp
            )

        kw = {k: g(self.kw[k]) for k in self.weight_names}
        tok = jax.ShapeDtypeStruct(
            (batch, VERIFY) if self.spec else (batch,), jnp.int32, sharding=rep
        )
        vec = jax.ShapeDtypeStruct((batch,), jnp.int32, sharding=rep)
        kpt = jax.ShapeDtypeStruct(tuple(self.kpt.shape), jnp.int32, sharding=rep)
        rank = jax.ShapeDtypeStruct((self.world,), jnp.int32, sharding=tp)
        dev = jax.ShapeDtypeStruct((self.world,), jnp.int32, sharding=rep)
        text = step.lower(
            kw,
            tok,
            vec,
            vec,
            vec,
            g(self.conv),
            g(self.rec),
            g(self.kvq),
            g(self.kvs),
            kpt,
            rank,
            dev,
        ).as_text()
        return text.count("tpu_custom_call")

    def _get_step_op(self, batch: int) -> Any:
        """Build or return the `pallas.jax_op` for `batch` decode rows."""
        key = ("step", batch, self.spec)
        op = self._ops.get(key)
        if op is not None:
            return op

        make = mk.make_spec_decode_step if self.spec else mk.make_decode_step
        step = make(
            self.cfg,
            self.mesh,
            batch=batch,
            max_batch=self.n_slots,
            n_pages=self.n_pages,
            profile_scopes=getattr(self, "profile_scopes", False),
            experts_kmajor=True,
            keep_rank_axis=True,
            side_effects=False,
        )
        names = self.weight_names
        nw = len(names)
        n_head = 6 if self.spec else 3
        nt = batch * (VERIFY if self.spec else 1)
        spec = self.spec

        def body(*a: jax.Array) -> tuple[jax.Array, ...]:
            kw = dict(zip(names, a[:nw]))
            packed, conv, rec, kvq, kvs, kpt, rank, dev = a[nw:]
            tok = packed[:nt].reshape((batch, VERIFY) if spec else (batch,))
            pos = packed[nt : nt + batch]
            slot = packed[nt + batch : nt + 2 * batch]
            valid = packed[nt + 2 * batch : nt + 3 * batch]
            res = step(kw, tok, pos, slot, valid, conv, rec, kvq, kvs, kpt, rank, dev)
            heads = [
                jax.lax.bitcast_convert_type(res[0].astype(jnp.float32), jnp.int32)
            ] + list(res[1:n_head])
            flat = jnp.concatenate(
                [h.reshape(h.shape[0], -1).astype(jnp.int32) for h in heads],
                axis=1,
            )
            return (flat,) + tuple(res[n_head : n_head + 4])

        p = jax.sharding.PartitionSpec
        in_specs = tuple([p("tp")] * nw + [p()] + [p("tp")] * 4 + [p(), p("tp"), p()])
        out_specs = tuple([p("tp")] * 5)
        head_width = (
            batch * (VERIFY * (2 * mk.TOPK_OUT + 2) + 2)
            if spec
            else batch * (2 * mk.TOPK_OUT + 1)
        )
        op = _build_sharded_step_op(
            f"glm53_mk::{'spec' if spec else 'plain'}_b{batch}",
            _explicit_arity(body, nw + 8, 5),
            mesh=self.mesh,
            input_partition_specs=in_specs,
            output_partition_specs=out_specs,
            donate_argnums=tuple(range(nw + 1, nw + 5)),
            head_width=head_width,
        )
        self._ops[key] = op
        if not hasattr(self, "_w_list"):
            self._w_list = [self.kw[k] for k in names]
        return op

    def decode_step(
        self, tokens: np.ndarray, positions: np.ndarray, slots: np.ndarray
    ) -> tuple[Any, int]:
        """Run one megakernel step for `n` requests (bucketed to 1, 2, 4, 8)."""
        n = len(slots)
        if n > KERNEL_BATCH:
            return self._decode_step_ncall(tokens, positions, slots)
        b = 1 << max(0, int(n - 1).bit_length())
        q = VERIFY if self.spec else 1
        nt = b * q
        packed = np.zeros((nt + 3 * b,), np.int32)
        packed[: n * q] = np.asarray(tokens, np.int32).reshape(-1)
        packed[nt : nt + n] = positions
        packed[nt + b : nt + b + n] = slots
        packed[nt + 2 * b : nt + 2 * b + n] = 1
        if n < b:
            used = {int(x) for x in slots}
            spare = [x for x in range(self.n_slots) if x not in used]
            packed[nt + b + n : nt + 2 * b] = spare[: b - n]
        op = self._get_step_op(b)
        outs = op(
            *self._w_list,
            torch.from_numpy(packed).to(self._dev),
            self.conv,
            self.rec,
            self.kvq,
            self.kvs,
            self.kpt,
            self.rank_t,
            self.dev_t,
        )
        self.conv, self.rec, self.kvq, self.kvs = outs[1:5]
        return outs[0], b

    def _decode_step_ncall(
        self, tokens: np.ndarray, positions: np.ndarray, slots: np.ndarray
    ) -> tuple[Any, int]:
        """Run a batch above `KERNEL_BATCH` as consecutive `KERNEL_BATCH` calls."""
        n = len(slots)
        parts: list[tuple[np.ndarray, ...]] = []
        rows: list[int] = []
        tm = self.timer
        for g in range(0, n, KERNEL_BATCH):
            flat, b = self.decode_step(
                tokens[g : g + KERNEL_BATCH],
                positions[g : g + KERNEL_BATCH],
                slots[g : g + KERNEL_BATCH],
            )
            parts.append(self.unpack_heads(flat.cpu().numpy()[0], b))
            rows.append(min(KERNEL_BATCH, n - g))
            if tm is not None:
                tm.mark(f"call{g // KERNEL_BATCH}")
        merged = tuple(
            np.concatenate([p[i][:m] for p, m in zip(parts, rows)], axis=0)
            for i in range(len(parts[0]))
        )
        return _HostHeads(merged), n

    def unpack_heads(self, flat: Any, b: int) -> tuple[np.ndarray, ...]:
        """Split this rank's packed head vector into numpy head arrays."""
        if isinstance(flat, tuple):
            return flat
        return unpack_heads(flat, b, self.spec)

    # ---------------- runner integration (`DecodeOwner`) ----------------

    def bind_runner(self, runner: Any) -> None:
        """Bind KDA unified pools and KV cache groups after `initialize_kv_cache`."""
        self.block_size = int(runner.vllm_config.cache_config.block_size)
        if self.block_size != self.cfg.page_size:
            raise RuntimeError(
                f"vLLM block_size ({self.block_size}) != megakernel page_size "
                f"({self.cfg.page_size})"
            )
        self.bind_kda_pools(runner.model)
        self._kda_group_of: dict[str, int] = {}
        self._kda_group_block: dict[int, int] = {}
        mla0_pat = f"layers.{self.cfg.mla_layers[0]}."
        for group_list in runner.attn_groups:
            for group in group_list:
                b = group.metadata_builders[0]
                for name in group.layer_names:
                    if name in self._kda_layer_names:
                        self._kda_group_of[name] = int(b.kv_cache_group_id)
                        self._kda_group_block[int(b.kv_cache_group_id)] = int(
                            b.target_block_size
                        )
                    elif (
                        mla0_pat in name
                        and "indexer" not in name
                        and self._mla_group_id is None
                    ):
                        self._mla_group_id = int(b.kv_cache_group_id)
        missing = [n for n in self._kda_layer_names if n not in self._kda_group_of]
        if missing:
            raise RuntimeError(f"KDA layers without a KV cache group: {missing[:3]}")
        self.warmup()

    @staticmethod
    def _megakernel_eligible_request(runner: Any, req_id: str) -> bool:
        """True if `req_id` uses greedy sampling without host logits processors."""
        if req_id.startswith("__") or req_id not in runner.input_batch.greedy_reqs:
            return False
        req = runner.requests.get(req_id)
        sp = getattr(req, "sampling_params", None)
        if sp is None:
            return False
        return not (
            getattr(sp, "logits_processors", None)
            or (sp.logprobs is not None and int(sp.logprobs) > mk.TOPK_OUT)
            or sp.bad_words
            or sp.allowed_token_ids
            or sp.presence_penalty
            or sp.frequency_penalty
            or (sp.repetition_penalty not in (None, 1.0))
            or sp.logit_bias
            or (sp.min_tokens and not sp.ignore_eos)
            or getattr(sp, "structured_outputs", None)
        )

    def release(self, req_ids: Iterable[str]) -> None:
        # Sort so `SlotTable.free` stays identical across TP workers even when
        # callers pass an unsorted `set[str]` under random `PYTHONHASHSEED`.
        for r in sorted(req_ids):
            self.slots.release(r)

    def _slot_subpage(self, slot: int, lp: int) -> int:
        k = lp // self.world
        grp, g = k // KV_GROUP, k % KV_GROUP
        return (1 + slot * self.max_kp + grp) * KV_GROUP + g

    def chunk_tokens(
        self,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_tokens_padded: int,
    ) -> None:
        """Record per-token `(slot, position)` before a vLLM forward chunk."""
        ts = np.full((num_tokens_padded,), -1, np.int32)
        tp = np.zeros((num_tokens_padded,), np.int32)
        off = 0
        page = self.cfg.page_size
        use_pc = self.prefix_cache_enabled and self._mla_group_id is not None
        bt = (
            runner.input_batch.block_table[self._mla_group_id].get_cpu_tensor()
            if use_pc
            else None
        )
        restore_pairs: list[tuple[int, int]] = []
        save_pairs: list[tuple[int, int]] = []
        for i in range(start_index, start_index + num_reqs):
            rid = runner.input_batch.req_ids[i]
            n = int(scheduler_output.num_scheduled_tokens[rid])
            c0 = int(runner.input_batch.num_computed_tokens_cpu[i])
            st = self.slots.get(rid)
            if (
                st is None
                and self._megakernel_eligible_request(runner, rid)
                and (
                    c0 == 0
                    or (
                        use_pc
                        and bt is not None
                        and self._can_restore_prefix(bt[i], c0)
                    )
                )
            ):
                st = self.slots.acquire(rid)
                if st is not None and c0 > 0 and bt is not None:
                    for lp in range(c0 // page):
                        if lp % self.world == self.rank:
                            c_sub = self._rank_cached_blocks[self.rank][int(bt[i, lp])]
                            restore_pairs.append(
                                (c_sub, self._slot_subpage(st.slot, lp))
                            )
            if st is not None and not st.owned:
                ts[off : off + n] = st.slot
                tp[off : off + n] = c0 + np.arange(n, dtype=np.int32)
                if bt is not None:
                    full_end = (c0 + n) // page
                    touch_end = (c0 + n + page - 1) // page
                    for lp in range(c0 // page, full_end):
                        owner_r = lp % self.world
                        blk = int(bt[i, lp])
                        self._invalidate_cached_block(blk, keep_rank=owner_r)
                        c_sub = self._alloc_cache_subpage_for_rank(owner_r, blk)
                        if owner_r == self.rank:
                            save_pairs.append((self._slot_subpage(st.slot, lp), c_sub))
                    for lp in range(max(c0 // page, full_end), touch_end):
                        self._invalidate_cached_block(int(bt[i, lp]))
            elif bt is not None:
                for lp in range(c0 // page, (c0 + n + page - 1) // page):
                    self._invalidate_cached_block(int(bt[i, lp]))
            off += n
        if restore_pairs:
            self.copy_subpages(restore_pairs)
        self._chunks.append((ts, tp, off))
        self._chunk_saves.append(save_pairs)

    def begin_step(self, num_chunks: int = 0) -> None:
        self._chunks = []
        self._chunk_saves = []
        self._single_chunk_spec = bool(self.spec and num_chunks == 1)

    def flush_chunk(self) -> None:
        """Write prompt latents after a target forward chunk."""
        if getattr(self, "_single_chunk_spec", False):
            return
        ts, tp, n = self._chunks[-1]
        if n:
            self.flush_latents(ts, tp, n, mtp=False)
            if not self.spec and self._chunk_saves and self._chunk_saves[-1]:
                self.copy_subpages(self._chunk_saves[-1])

    def flush_draft_chunk(self, chunk_index: int) -> None:
        """Write MTP layer latents after the drafter's pass over `chunk_index`."""
        if not self.spec or chunk_index >= len(self._chunks):
            return
        ts, tp, n = self._chunks[chunk_index]
        if n:
            self.flush_latents(
                ts,
                tp,
                n,
                mtp=True,
                all_layers=getattr(self, "_single_chunk_spec", False),
            )
            if chunk_index < len(self._chunk_saves) and self._chunk_saves[chunk_index]:
                self.copy_subpages(self._chunk_saves[chunk_index])

    def take_ownership(
        self,
        runner: Any,
        scheduler_output: Any,
        sampled: Mapping[str, tuple[int, int]],
        drafts: Mapping[str, list[int]] | None,
    ) -> None:
        """Transfer completed prompts' KDA state to megakernel decode slots."""
        del scheduler_output
        todo = []
        for rid, (tok, seq_len) in sampled.items():
            st = self.slots.get(rid)
            if st is None or st.owned:
                continue
            i = runner.input_batch.req_id_to_index[rid]
            todo.append((rid, st, i, int(seq_len), tok))
        if not todo:
            return
        blocks = []
        for _rid, st, i, seq_len, _ in todo:
            row = []
            for name in self._kda_layer_names:
                gid = self._kda_group_of[name]
                bt = runner.input_batch.block_table[gid].get_cpu_tensor()
                row.append(int(bt[i, (seq_len - 1) // self._kda_group_block[gid]]))
            blocks.append((st.slot, row))
        self.kda_handoff(blocks)
        for rid, st, _i, _seq_len, tok in todo:
            st.owned = True
            st.kda_ready = True
            if self.spec:
                d = (drafts or {}).get(rid)
                if d is None or len(d) != SPEC_K:
                    raise RuntimeError(
                        f"request {rid}: expected {SPEC_K} drafts, got {d}"
                    )
                st.next_verify = [int(tok)] + [int(x) for x in d]

    def run_decode(
        self,
        runner: Any,
        indices: Sequence[int],
        scheduler_output: Any = None,
    ) -> Mapping[str, DecodeResult]:
        """Run one decode step for the owned requests at batch `indices`."""
        tm = self.timer
        if tm is not None:
            tm.start()
        req_ids = [runner.input_batch.req_ids[i] for i in indices]
        slots_list: list[int] = []
        verify_rows: list[list[int]] = []
        for r in req_ids:
            st = self.slots.get(r)
            assert st is not None
            slots_list.append(st.slot)
            if self.spec:
                assert st.next_verify is not None
                row = list(st.next_verify)
                if scheduler_output is not None:
                    sched = scheduler_output.scheduled_spec_decode_tokens.get(r, ())
                    n_sched = int(scheduler_output.num_scheduled_tokens.get(r, 0))
                    if list(sched) != row[1 : 1 + len(sched)] or n_sched != 1 + len(
                        sched
                    ):
                        logger.warning_once(
                            "GLM53 spec window mismatch req=%s scheduled=%d "
                            "sched_drafts=%s ours=%s",
                            r,
                            n_sched,
                            list(sched),
                            row,
                        )
                    # Mask any unscheduled draft positions with -1 before the step
                    # so device acceptance, KDA rollback, and MTP drafting stay in
                    # lockstep with the scheduler's scheduled window.
                    row = (
                        [row[0]]
                        + [int(x) for x in list(sched)[:SPEC_K]]
                        + [-1] * max(0, SPEC_K - len(sched))
                    )
                verify_rows.append(row)
        slots = np.asarray(slots_list, np.int32)
        pos = np.asarray(
            [runner.input_batch.num_computed_tokens_cpu[i] for i in indices],
            np.int32,
        )
        if self.spec:
            toks = np.asarray(verify_rows, np.int32)
        else:
            toks = np.asarray(
                [
                    int(runner.input_batch.token_ids_cpu[i, pos[b]])
                    for b, i in enumerate(indices)
                ],
                np.int32,
            )
        if tm is not None:
            tm.mark("prep")
        flat, b = self.decode_step(toks, pos, slots)
        if tm is not None:
            tm.mark("launch")
        host = self.unpack_heads(flat.cpu().numpy()[0], b)
        if tm is not None:
            tm.mark("kernel_d2h")
        n = len(indices)
        finite = bool(np.all(np.isfinite(host[0][:n])))
        if not finite:
            logger.warning_once(
                "GLM53 megakernel non-finite logprobs on slots=%s pos=%s",
                slots.tolist(),
                pos.tolist(),
            )
        elif self.debug:
            logger.info(
                "GLM53 megakernel step slots=%s pos=%s toks=%s top_id=%s",
                slots.tolist(),
                pos.tolist(),
                toks.tolist(),
                host[1][:n].reshape(n, -1)[:, :4].tolist(),
            )
        res: dict[str, DecodeResult] = {}
        if self.spec:
            top_lp, top_id, target, n_acc, nxt_v = (
                host[0],
                host[1],
                host[2],
                host[3],
                host[4],
            )
            for b_idx, r in enumerate(req_ids):
                st = self.slots.get(r)
                assert st is not None
                m = int(n_acc[b_idx])
                if scheduler_output is not None:
                    sched = scheduler_output.scheduled_spec_decode_tokens.get(r, ())
                    m = max(1, min(m, 1 + len(sched)))
                st.next_verify = [int(t) for t in nxt_v[b_idx]]
                res[r] = DecodeResult(
                    tokens=[int(t) for t in target[b_idx, :m]],
                    top_ids=top_id[b_idx, :m],
                    top_logprobs=top_lp[b_idx, :m],
                    drafts=[int(t) for t in nxt_v[b_idx, 1:]],
                )
        else:
            top_lp, top_id, nxt = host[0], host[1], host[2]
            for b_idx, r in enumerate(req_ids):
                res[r] = DecodeResult(
                    tokens=[int(nxt[b_idx])],
                    top_ids=top_id[b_idx : b_idx + 1],
                    top_logprobs=top_lp[b_idx : b_idx + 1],
                    drafts=None,
                )
        if tm is not None:
            total_reqs = int(
                getattr(getattr(runner, "input_batch", None), "num_reqs", n)
            )
            tm.end("pure" if n == total_reqs else "mixed")
        return res

    def warmup(self) -> None:
        """Compile and run every decode bucket, latent flush, and KDA handoff once."""
        import time

        t0 = time.perf_counter()
        for b in decode_buckets(KERNEL_BATCH):
            op = self._get_step_op(b)
            q = VERIFY if self.spec else 1
            packed = np.zeros((b * q + 3 * b,), np.int32)
            packed[b * q + b : b * q + 2 * b] = np.arange(b)
            outs = op(
                *self._w_list,
                torch.from_numpy(packed).to(self._dev),
                self.conv,
                self.rec,
                self.kvq,
                self.kvs,
                self.kpt,
                self.rank_t,
                self.dev_t,
            )
            self.conv, self.rec, self.kvq, self.kvs = outs[1:5]
            self.unpack_heads(outs[0].cpu().numpy()[0], b)
            _tpu_sync()
        n_target = len(self.cfg.mla_layers)
        layer_sets = (
            [tuple(range(self.num_mla)), tuple(range(n_target)), (n_target,)]
            if self.spec
            else [tuple(range(n_target))]
        )
        for bucket in self._flush_buckets():
            sp = np.zeros((3, bucket), np.int32)
            sp[0] = -1
            sp_t = torch.from_numpy(sp).to(self._dev)
            for layers in layer_sets:
                op = self._get_flush_op(layers, bucket)
                self.kvq, self.kvs = op(
                    self.kvq, self.kvs, self.stash, sp_t, self.kpt, self.rank_t
                )
                _tpu_sync()
        for n in decode_buckets(self.n_slots):
            idle = [(self.n_slots - 1, [0] * len(self.cfg.kda_layers))] * n
            self.kda_handoff(idle)
        if self.prefix_cache_enabled:
            for b in (16, 64, 128):
                arr = np.full((2, b), -1, np.int32)
                op = self._get_copy_subpages_op(b)
                self.kvq, self.kvs = op(
                    self.kvq, self.kvs, torch.from_numpy(arr).to(self._dev)
                )
                _tpu_sync()
        self.conv.zero_()
        self.rec.zero_()
        _tpu_sync()
        logger.info(
            "GLM53 megakernel decode warmup done in %.1f s", time.perf_counter() - t0
        )
        try:
            counts = {b: self.count_custom_calls(b) for b in (1, 8)}
            logger.info(
                "GLM53 megakernel decode step tpu_custom_call count per batch bucket: "
                "%s (spec=%s)",
                counts,
                self.spec,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("GLM53 megakernel custom call count failed: %s", e)


_VALID_PLUGIN_OPTIONS = frozenset(
    {"max_req", "cache_kpages", "profile_scopes", "debug", "timing"}
)


def create(view: Any, options: Mapping[str, Any]) -> DecodeOwner | Decline:
    """Megakernel plugin entrypoint (`glm5_next.megakernel:create`)."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel.adapter.config import (
        MegakernelConfigError,
        parse_bool,
    )

    unknown = sorted(set(options) - _VALID_PLUGIN_OPTIONS)
    if unknown:
        raise MegakernelConfigError(
            f"GLM-5.3 megakernel unknown option(s): {', '.join(unknown)} "
            f"(supported: {sorted(_VALID_PLUGIN_OPTIONS)})"
        )
    spec_tokens = int(view.speculative_tokens)
    if spec_tokens not in (0, SPEC_K):
        return Decline(
            "unsupported_spec_tokens",
            f"GLM-5.3 megakernel supports num_speculative_tokens in (0, {SPEC_K}), "
            f"got {spec_tokens}",
        )
    runner = view._runner
    vcfg = getattr(runner, "vllm_config", None)
    if spec_tokens > 0 and vcfg is not None:
        spec_cfg = getattr(vcfg, "speculative_config", None)
        method = getattr(spec_cfg, "method", None)
        if method is not None and method != "mtp":
            return Decline(
                "unsupported_spec_method",
                f"GLM-5.3 megakernel supports speculative method 'mtp', got {method!r}",
            )
    mla_impl = (envs.VLLM_TPU_GLM53_MLA_IMPL or "rpa").strip().lower()
    if mla_impl != "rpa":
        return Decline(
            "unsupported_mla_impl",
            "GLM-5.3 megakernel requires VLLM_TPU_GLM53_MLA_IMPL='rpa', "
            f"got {mla_impl!r}",
        )
    if vcfg is not None:
        load_cfg = getattr(vcfg, "load_config", None)
        if str(getattr(load_cfg, "load_format", "") or "").lower() == "dummy":
            return Decline(
                "dummy_weights_unsupported",
                "GLM-5.3 megakernel does not support --load-format dummy",
            )
        model_ref = getattr(getattr(vcfg, "model_config", None), "model", None)
        if model_ref is not None and _resolve_checkpoint_dir(str(model_ref)) is None:
            return Decline(
                "checkpoint_not_local",
                "GLM-5.3 megakernel requires a local or HF-cached checkpoint, "
                f"got {model_ref!r}",
            )
    max_req = int(options.get("max_req", MAX_SLOTS))
    if max_req not in (8, 16, 32):
        return Decline(
            "unsupported_max_req",
            f"GLM-5.3 megakernel supports max_req in (8, 16, 32), got {max_req}",
        )
    raw_cache_kp = options.get("cache_kpages")
    cache_kpages = int(raw_cache_kp) if raw_cache_kp is not None else None
    profile_scopes = parse_bool(
        options.get("profile_scopes", False), key="profile_scopes"
    )
    debug = parse_bool(options.get("debug", False), key="debug")
    timing = parse_bool(options.get("timing", False), key="timing")
    kwargs: dict[str, Any] = {
        "max_req": max_req,
        "cache_kpages": cache_kpages,
        "debug": debug,
        "timing": timing,
    }
    if profile_scopes:
        kwargs["profile_scopes"] = True
    return Glm53DecodeOwner(runner, **kwargs)
