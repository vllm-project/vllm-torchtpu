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
"""`Qwen35Program`: the Qwen3.5 megakernel as a `MegakernelProgram`.

`create` checks the checkpoint and config without loading anything; `load`
packs the weights (EP->TP reshard first when needed); `bind` aliases the
runner's unified KV+GDN pools and checks the recurrent slot layout; `build`
returns one exported-bridge step per bucket; `warmup_session` /
`warmup_inputs` run the sentinel pool-write check; `decide` / `forward` route
one decode (`T=1`) or MTP verify (`T=4`) step.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import jax
import numpy as np
import torch
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel import (
    CacheExec,
    ExecSpec,
    MegakernelLayoutError,
    ModelDef,
    Profile,
    with_cache,
)
from vllm_torchtpu.megakernel import pool as pool_pkg
from vllm_torchtpu.megakernel import runtime as mk_rt
from vllm_torchtpu.megakernel.models.qwen35 import Qwen35
from vllm_torchtpu.megakernel_runtime import (
    TorchStepMetaOps,
    build_pooled_step_op,
    execute_pooled_step_inplace,
    find_backbone,
    get_routed_experts,
    load_vllm_weights,
    reshard_layer_experts_ep_to_tp,
)
from vllm_torchtpu.megakernel_runtime.plugin import (
    Bucket,
    Decline,
    ForwardView,
    LayoutDecline,
    RunnerShape,
    RunnerView,
    WarmupCase,
)
from vllm_torchtpu.megakernel_runtime.pool_facts import (
    conv_qk_pair_layout_enabled,
    framework_pool_layout_facts,
)

logger = init_logger(__name__)

OPTION_KEYS = frozenset(
    {
        "profile",
        "mode",
        "buckets",
        "verify_buckets",
        "fused_pool_io",
        "fused_pool_io_max_b",
        "fused_pool_io_min_b_hi",
        "pool_gather",
        "compile_hlo_check",
    }
)
# The profile `create` serves when `options` names none: the candidate30 goal
# configuration (approximate numerics). `qwen35-baseline` (exact numerics) is
# one option away: `{"options": {"profile": "qwen35-baseline"}}`.
DEFAULT_PROFILE: str | None = "qwen35-candidate30-fused"
UNALIGNED_LAYOUT_MESSAGE = (
    "Megakernel requires unified KV+GDN cache layout "
    "(_unified_kv_layout=True); skipping."
)


def _need_layout(cond: bool, msg: str) -> None:
    if not cond:
        raise MegakernelLayoutError(f"Qwen35Program: {msg}")


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Qwen35Program: {msg}")


def _unwrap(x: Any) -> Any:
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def parse_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """This template's settings from the plugin `options` (unknown keys raise)."""
    if bad := sorted(set(options) - OPTION_KEYS):
        raise ValueError(
            f"Qwen35Program: unknown option {bad[0]!r}; known: {sorted(OPTION_KEYS)}"
        )
    return mk_rt.parse_runtime_config(dict(options))


_TORCH_STEP_META_OPS = TorchStepMetaOps()


def prepare_step_inputs_eager(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    s0: torch.Tensor,
    c0: torch.Tensor,
    r0: torch.Tensor,
    s1: torch.Tensor,
    c1: torch.Tensor,
    r1: torch.Tensor,
    s2: torch.Tensor,
    c2: torch.Tensor,
    r2: torch.Tensor,
    bucket_t: int,
    tokens_per_seq: int,
    max_pages_per_seq: int,
    has_roff: bool,
    use_ckpt_2d: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack `(ids_b, pos_b, pt_b, ckpt_b, roff_b)` for a bucket of `bucket_t` tokens."""
    groups = ((s0, c0, r0), (s1, c1, r1), (s2, c2, r2))
    kw = dict(
        bucket_t=bucket_t,
        tokens_per_seq=tokens_per_seq,
        max_pages_per_seq=max_pages_per_seq,
        has_roff=has_roff,
        use_ckpt_2d=use_ckpt_2d,
        ops=_TORCH_STEP_META_OPS,
    )
    return mk_rt.pack_step_meta(
        input_ids, positions, block_tables, groups, **kw
    ).as_tuple()


def _extract_step_sc(
    m_obj: Any, fb_1d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    s = getattr(m_obj, "mamba_state_indices", None)
    c = getattr(m_obj, "mamba_ckpt_indices", None)
    s = (c if c is not None else fb_1d) if s is None else s
    return s, (c if c is not None else s)


def validate_hf_config(
    hf_cfg: Any, model_def: type[ModelDef] = Qwen35
) -> mk_rt.EligibilityResult:
    """Whether the checkpoint's HF config is the model this template serves."""
    if hf_cfg is None:
        return mk_rt.EligibilityResult(True)
    return model_def.validate_hf_config(hf_cfg)


def create(view: RunnerView, options: Mapping[str, Any]) -> Qwen35Program | Decline:
    """Check the checkpoint and config; build an unloaded program, or decline.

    Runs no collective and loads no weights (the mesh doesn't exist yet).
    """
    cfg = parse_options(options)
    if cfg["profile"] is None:
        cfg["profile"] = DEFAULT_PROFILE
    mdef = Qwen35
    res = validate_hf_config(view.hf_config, mdef)
    if not res.eligible:
        return Decline(res.code, res.message, res.level)
    model = view.model
    try:
        backbone = find_backbone(model)
    except ValueError as exc:
        return Decline("backbone_not_found", f"Megakernel: {exc}; skipping.")
    d_geom = mdef.derive_geometry()
    exp_l, tot_e = int(d_geom.L), int(getattr(d_geom, "E", 0))
    r_cnt = view.tp_size
    prof = mk_rt.resolve_driver_profile(cfg["profile"], model_name=mdef.name)
    tp_sup = (
        prof.supported_tp_sizes
        if prof.supported_tp_sizes is not None
        else mdef.supported_tp_sizes
    )
    el = mk_rt.check_startup_eligibility(
        num_layers=len(getattr(backbone, "layers", ())),
        expected_num_layers=exp_l,
        unified_cache_layout=view.unified_kv_layout,
        tp_size=r_cnt,
        supported_tp_sizes=tp_sup,
    )
    if el.code == "unaligned_cache_layout":
        el = dataclasses.replace(el, message=UNALIGNED_LAYOUT_MESSAGE)
    if not el.eligible:
        return Decline(el.code, el.message, el.level)
    loaded_ep = (
        int(get_routed_experts(backbone.layers[0].mlp).w13_weight.shape[0]) < tot_e
    )
    mode = cfg["mode"] or ("ep" if loaded_ep else "tp")
    owned = mk_rt.explicit_profile_settings(cfg)
    b_policy = mk_rt.BucketPolicy.resolve(
        prof,
        owned["buckets"],
        owned["verify_buckets"],
        speculative_tokens=view.speculative_tokens,
        buckets_explicit=bool(cfg.get("buckets_explicit", False)),
        verify_buckets_explicit=bool(cfg.get("verify_buckets_explicit", False)),
    )
    el_post = mk_rt.check_post_mesh_eligibility(
        tp_size=r_cnt, device_kind=view.device_kind, bucket_policy=b_policy
    )
    if not el_post.eligible:
        return Decline(el_post.code, el_post.message, el_post.level)
    return Qwen35Program(
        model,
        model_def=mdef,
        tp_rank=view.tp_rank,
        mode=mode,
        buckets=b_policy.buckets,
        verify_buckets=b_policy.verify_buckets,
        reshard_ep_to_tp=(mode == "tp" and loaded_ep),
        profile=cfg["profile"],
        fused_pool_io=owned["fused_pool_io"],
        fused_pool_io_max_b=owned["fused_pool_io_max_b"],
        fused_pool_io_min_b_hi=owned["fused_pool_io_min_b_hi"],
        pool_gather=owned["pool_gather"],
        compile_hlo_check=bool(cfg["compile_hlo_check"]),
    )


class Qwen35Program:
    """The Qwen3.5 megakernel on one TP rank (a `MegakernelProgram`).

    Passing `mesh` loads immediately (direct use and tests); `create` leaves
    it to `load(mesh)`, after the runtime's first agreement.
    """

    def __init__(
        self,
        model: Any,
        mesh: Any = None,
        *,
        model_def: type[ModelDef] = Qwen35,
        tp_rank: int = 0,
        mode: str = "ep",
        spec: Any = None,
        geom: Any = None,
        base_exec: ExecSpec | None = None,
        weights: dict[str, Any] | None = None,
        buckets: Sequence[int] | None = None,
        verify_buckets: Sequence[int] | None = None,
        reshard_ep_to_tp: bool = False,
        expert_placement: Any = None,
        profile: str | Profile | None = None,
        fused_pool_io: bool | str | None = None,
        fused_pool_io_max_b: int | None = None,
        fused_pool_io_min_b_hi: int | None = None,
        pool_gather: bool | str | None = None,
        compile_hlo_check: bool | None = None,
    ) -> None:
        _need(mode in ("ep", "tp"), f"invalid mode={mode!r}; expected 'ep' or 'tp'")
        self.model_def: type[ModelDef] = model_def
        self.model, self.backbone = model, find_backbone(model)
        self.embed_tokens = self.backbone.embed_tokens
        self.tp_rank, self.mode = int(tp_rank), mode
        self._spec_arg, self._geom_arg = spec, geom
        self._base_exec_arg, self._weights_arg = base_exec, weights
        self._reshard_ep_to_tp, self._expert_placement = (
            reshard_ep_to_tp,
            expert_placement,
        )
        model_name = str(getattr(spec, "name", "") or model_def.name)
        self.display_name = model_def.display_name or model_name
        self._display_name = self.display_name
        p_kw = dict(
            model_name=model_name,
            fused_pool_io=fused_pool_io,
            fused_pool_io_max_b=fused_pool_io_max_b,
            fused_pool_io_min_b_hi=fused_pool_io_min_b_hi,
            pool_gather=pool_gather,
        )
        self.profile: Profile = mk_rt.resolve_driver_profile(profile, **p_kw)
        self._compile_hlo_check = bool(compile_hlo_check)
        self.fused_pool_io: bool | str = self.profile.fused_pool_io
        self.fused_pool_io_max_b = int(self.profile.fused_pool_io_max_b)
        self.fused_pool_io_min_b_hi = int(self.profile.fused_pool_io_min_b_hi)
        self.pool_gather: bool | None = self.profile.pool_gather
        self.bucket_policy = mk_rt.BucketPolicy.resolve(
            self.profile, buckets, verify_buckets
        )
        _need(
            bool(self.decode_buckets or self.verify_buckets),
            "at least one of decode buckets or verify_buckets must be non-empty",
        )
        self.mesh: Any = None
        self.raw_pools: list[torch.Tensor] = []
        self.num_pages, self.page_size, self.max_pages_per_seq = 0, 128, 0
        self.mamba_layer_names: list[str] = []
        self.attn_layer_name, self._ckpt_window = "", 1
        self._zero_roff_1d: torch.Tensor | None = None
        self._bucket_out_rows: dict[tuple[int, int], int] = {}
        self._bucket_fused_io: dict[tuple[int, int], bool] = {}
        self._bucket_gpr: dict[tuple[int, int], bool] = {}
        self._expected_buckets: list[tuple[int, int]] = []
        self._hlo_checked_buckets: set[tuple[int, int]] = set()
        self._torchtpu_ir_checked_buckets: set[tuple[int, int]] = set()
        self._torchtpu_ir_error: str | None = None
        self._prepared: tuple[Any, ...] | None = None
        self._warmup_plan: dict[tuple[int, ...], Any] | None = None
        self.view: RunnerView | None = None
        self.pool_layout_facts: Any = None
        if mesh is not None:
            self.load(mesh)

    @property
    def decode_buckets(self) -> tuple[int, ...]:
        return self.bucket_policy.buckets

    @property
    def verify_buckets(self) -> tuple[int, ...]:
        return self.bucket_policy.verify_buckets

    @property
    def max_bucket(self) -> int:
        return self.bucket_policy.max_bucket

    @property
    def max_verify_bucket(self) -> int:
        return self.bucket_policy.max_verify_bucket

    # --- load ---------------------------------------------------------------

    def _build_spec(self, mode: str, geom: Any = None) -> Any:
        if geom is None:
            e_slots = (
                int(get_routed_experts(self.backbone.layers[0].mlp).w13_weight.shape[0])
                if mode == "ep"
                else 0
            )
            geom = self.model_def.derive_geometry(
                self.model_def.default_hf_config,
                moe_tp=(mode == "tp"),
                e_slots=0 if mode == "tp" else e_slots,
            )
        return self.model_def.to_model_spec(geom)

    def load(self, mesh: Any) -> None:
        """Build the spec and pack the weights (after the runtime's agreement:
        the EP->TP reshard all-gathers across ranks)."""
        self.mesh = mesh
        r_cnt = int(mesh.devices.size)
        spec = self._spec_arg
        self.spec = (
            spec if spec is not None else self._build_spec(self.mode, self._geom_arg)
        )
        self.geom = self.spec.config
        mk_rt.check_compile_hlo_mesh(self.mesh, self._compile_hlo_check)
        base_exec = self._base_exec_arg
        if base_exec is None:
            b_max = max(self.max_bucket, self.max_verify_bucket, 8)
            base_exec = ExecSpec(
                R=r_cnt, B=b_max, T=1, stack_e=False, cache=CacheExec(paged_kv=True)
            )
        else:
            base_exec = with_cache(
                dataclasses.replace(base_exec, R=r_cnt, stack_e=False), paged_kv=True
            )
        self.base_exec = base_exec
        weights = self._weights_arg
        if weights is None:
            weights = self._prepare_and_pack_weights(
                mode=self.mode,
                reshard_ep_to_tp=self._reshard_ep_to_tp,
                expert_placement=self._expert_placement,
            )
        self.weights = weights
        self.w_flat: list[torch.Tensor] = self.weights["w_flat"]

    def _prepare_and_pack_weights(
        self, *, mode: str, reshard_ep_to_tp: bool, expert_placement: Any
    ) -> dict[str, Any]:
        if reshard_ep_to_tp and mode == "tp":
            from vllm.distributed import tensor_model_parallel_all_gather

            from vllm_torchtpu.layers.adapter.fused_moe import prebuild_fused_moe_kernel

            logger.info(
                "Resharding %s MoE experts from EP to TP across %d layers (rank=%d)...",
                self.spec.name,
                self.geom.L,
                self.tp_rank,
            )

            def _ag_fn(t: torch.Tensor) -> torch.Tensor:
                return tensor_model_parallel_all_gather(t, dim=0)

            for l_idx in range(self.geom.L):
                reshard_layer_experts_ep_to_tp(
                    self.backbone.layers[l_idx].mlp,
                    tp_rank=self.tp_rank,
                    g=self.geom,
                    ex=self.base_exec,
                    all_gather_fn=_ag_fn,
                    prebuild_moe_fn=prebuild_fused_moe_kernel,
                )
        if expert_placement is None:
            with contextlib.suppress(ImportError):
                from vllm_torchtpu import (
                    expert_placement as ep_mod,  # type: ignore[attr-defined]
                )

                expert_placement = ep_mod
        t0 = time.perf_counter()
        kw = dict(g=self.geom, tp_rank=self.tp_rank, expert_placement=expert_placement)
        weights = load_vllm_weights(self.model_def, self.backbone, self.base_exec, **kw)
        logger.info(
            "Packed %s megakernel weights (%d flat tensors, mode=%s) in %.2fs",
            self.spec.name,
            len(weights["w_flat"]),
            self.mode,
            time.perf_counter() - t0,
        )
        return weights

    # --- bind ---------------------------------------------------------------

    def bind(self, view: RunnerView) -> mk_rt.PoolLayoutFacts | None:
        """Bind the shared `float8_e4m3fn` raw pools from `backbone.layers`."""
        self.view = view
        _need_layout(
            view.unified_kv_layout,
            "runner._unified_kv_layout must be True "
            "(unified KV+GDN cache layout required)",
        )
        pinfo = mk_rt.pool_schedule_info(self.spec)
        natt, g_per_att, att_every = (
            pinfo.num_pools,
            pinfo.recurrent_slots_per_pool,
            pinfo.att_every,
        )
        _need_layout(
            g_per_att == 3,
            f"expected 3 recurrent slots per pool, got {g_per_att}",
        )
        pools: list[torch.Tensor] = []
        for r in range(natt):
            raw = _unwrap(
                self.backbone.layers[att_every * r + g_per_att].self_attn.attn.kv_cache
            )
            if isinstance(raw, torch.Tensor) and raw.ndim == 6 and raw.shape[0] == 1:
                raw = raw[0]
            pools.append(raw)

        _need_layout(len(pools) == natt, f"expected {natt} pools, got {len(pools)}")
        num_pages = int(pools[0].shape[0])
        page_size = int(pools[0].shape[-1])
        kvh_expected = max(1, self.geom.NKV // self.base_exec.R)
        expected_shape = pinfo.expected_pool_shape(
            num_pages, kvh_expected, self.geom.HD, page_size
        )
        for r, pool in enumerate(pools):
            _need_layout(
                isinstance(pool, torch.Tensor) and pool.dtype == torch.float8_e4m3fn,
                f"pool[{r}] must be float8_e4m3fn tensor, got {type(pool)}",
            )
            _need_layout(
                pool.storage_offset() == 0,
                f"pool[{r}] has non-zero storage_offset={pool.storage_offset()}",
            )
            _need_layout(
                tuple(pool.shape) == expected_shape,
                f"pool[{r}] shape {tuple(pool.shape)} != {expected_shape}",
            )
            for g_idx in range(g_per_att):
                la_cache = _unwrap(
                    self.backbone.layers[att_every * r + g_idx].linear_attn.kv_cache
                )
                _need_layout(
                    isinstance(la_cache, torch.Tensor)
                    and la_cache.numel() > 0
                    and la_cache.data_ptr() == pool.data_ptr(),
                    f"region {r} group {g_idx} linear_attn.kv_cache must "
                    f"alias pool[{r}] (unified KV+GDN cache layout is required)",
                )

        self._check_pool_layout_facts(view, pools[0])
        self.raw_pools, self.num_pages, self.page_size = (
            pools,
            num_pages,
            int(pools[0].shape[-1]),
        )
        self._zero_roff_1d = torch.zeros(
            (1,), dtype=torch.int32, device=pools[0].device
        )
        self._ckpt_window = view.ckpt_window
        self.max_pages_per_seq = view.max_pages_per_seq
        self.mamba_layer_names = [
            self.backbone.layers[g_idx].linear_attn.prefix for g_idx in range(g_per_att)
        ]
        self.attn_layer_name = self.backbone.layers[g_per_att].self_attn.attn.layer_name

        c_kw = dict(
            num_pages=self.num_pages,
            page_size=self.page_size,
            max_pages_per_seq=self.max_pages_per_seq,
        )
        self.base_exec = mk_rt.align_cache_exec(
            self.base_exec, self.profile, self.geom, **c_kw
        )
        return self.pool_layout_facts

    def _framework_slot_facts(
        self, view: RunnerView, linear_attn: Any, pool: torch.Tensor
    ) -> mk_rt.PoolLayoutFacts:
        block = view.cache_block_size
        if block is None:
            raise ValueError("runner has no cache_config.block_size")
        return framework_pool_layout_facts(
            pool=pool,
            linear_attn=linear_attn,
            block_size=block,
            conv_qk_pair=conv_qk_pair_layout_enabled(),
        )

    def _adopt_framework_conv_dtype(self, conv_dtype: str) -> None:
        """Store the conv window in the dtype the framework's pool uses.

        An f32 conv window is served on the split pool I/O path: fused pool
        I/O and in-kernel gather (GPR) implement f32 conv too but are not
        validated on TPU yet, so this turns both off, including an explicit
        `fused_pool_io` / `pool_gather` option (logged). Other dtypes are left
        to the layout comparison, which declines them.

        This replaces `self.spec` / `self.profile` before the comparison; on a
        decline the runtime replaces the whole driver, so nothing else sees
        them. `PoolBackend.with_layout` raises `ValueError` (voted
        `bind_error`) for a backend not built by `make_pool_backend`.
        """
        layout = self.spec.cache_backend.layout
        if (
            conv_dtype != "float32"
            or mk_rt.layout_facts.dtype_name(layout.conv_dtype) == "float32"
        ):
            return
        f32_layout = dataclasses.replace(layout, conv_dtype=jax.numpy.float32)
        self.spec = dataclasses.replace(
            self.spec, cache_backend=self.spec.cache_backend.with_layout(f32_layout)
        )
        cleared = [
            f"{name}={val!r}"
            for name, val in (
                ("fused_pool_io", self.fused_pool_io),
                ("pool_gather", self.profile.pool_gather),
            )
            if val is True
        ]
        self.fused_pool_io = False
        self.pool_gather = False
        self.profile = dataclasses.replace(self.profile, pool_gather=False)
        logger.info(
            "%s megakernel: the framework stores the pooled conv state as fp32; "
            "using split pool I/O with an f32 conv window (fused pool I/O and "
            "in-kernel gather are off for f32 conv until validated on TPU%s)",
            self._display_name,
            f"; cleared {', '.join(cleared)}" if cleared else "",
        )

    def _check_pool_layout_facts(self, view: RunnerView, pool: torch.Tensor) -> None:
        """Fail closed unless the framework lays out a recurrent slot exactly
        as this megakernel's `PoolLayout` expects."""
        linear_attn = next(
            (
                la
                for layer in self.backbone.layers
                if (la := getattr(layer, "linear_attn", None)) is not None
            ),
            None,
        )
        if linear_attn is None:  # no recurrent layers: no recurrent slot
            self.pool_layout_facts = None
            return
        try:
            fw_facts = self._framework_slot_facts(view, linear_attn, pool)
        except (AttributeError, TypeError, ValueError) as exc:
            msg = (
                "can't derive the framework's recurrent pool slot layout "
                f"({type(exc).__name__}: {exc}); skipping."
            )
            raise LayoutDecline(
                mk_rt.EligibilityResult(False, "pool_layout_unknown", msg)
            ) from exc
        self._adopt_framework_conv_dtype(fw_facts.conv_dtype)
        mk_facts = mk_rt.facts_from_pool_layout(
            self.spec.cache_backend.layout, int(self.base_exec.R)
        )
        res = mk_rt.compare_layout_facts(mk_facts, fw_facts)
        if not res.eligible:
            raise LayoutDecline(res)
        self.pool_layout_facts = mk_facts
        logger.info(
            "%s megakernel pool slot layout matches the framework (fingerprint %s)",
            self._display_name,
            mk_facts.fingerprint()[:16],
        )

    # --- buckets and build ----------------------------------------------------

    def buckets(self, token_paddings: Sequence[int]) -> list[Bucket]:
        """The compiled buckets: decode `(1, B)` then MTP verify `(4, B)`."""
        self.bucket_policy = mk_rt.BucketPolicy(
            self.decode_buckets,
            self.verify_buckets,
            tuple(int(x) for x in token_paddings),
        )
        for d in (self._bucket_out_rows, self._bucket_fused_io, self._bucket_gpr):
            d.clear()
        self._hlo_checked_buckets.clear()
        self._torchtpu_ir_checked_buckets.clear()
        keys = list(self.bucket_policy.iter_buckets())
        self._expected_buckets = keys
        return [Bucket(T, b, self.bucket_policy.out_rows(b)) for T, b in keys]

    def build(self, bucket: Bucket) -> Any:
        """Build the custom op and step function for `(bucket.T, bucket.B)`."""
        tokens_per_seq, bucket_t = int(bucket.T), int(bucket.B)
        b_kw = dict(
            bucket_t=bucket_t,
            tokens_per_seq=tokens_per_seq,
            fused_pool_io=self.fused_pool_io,
            fused_pool_io_max_b=self.fused_pool_io_max_b,
            fused_pool_io_min_b_hi=self.fused_pool_io_min_b_hi,
        )
        ex, use_fused, use_gpr = mk_rt.bucket_exec_spec(
            self.spec, self.base_exec, self.profile, **b_kw
        )
        bkt_key = (tokens_per_seq, bucket_t)
        if bkt_key not in self._hlo_checked_buckets:
            self._hlo_checked_buckets.add(bkt_key)
            h_kw = dict(
                max_pages_per_seq=self.max_pages_per_seq,
                compile_hlo=True if self._compile_hlo_check else None,
                ckpt_cols=2 if tokens_per_seq == 1 else tokens_per_seq,
            )
            ok_hlo, reason_hlo = mk_rt.verify_pooled_step_hlo(
                self.mesh, self.spec, ex, self.w_flat, self.raw_pools, **h_kw
            )
            if not ok_hlo:
                raise RuntimeError(
                    f"{self._display_name} megakernel HLO pool-donation check "
                    f"failed for (T={tokens_per_seq}, B={bucket_t}, "
                    f"fused_pool_io={use_fused}, pool_gather={use_gpr}): {reason_hlo}"
                )
            logger.info(
                "%s megakernel HLO pool-donation check passed for "
                "(T=%d, B=%d, fused_pool_io=%s, pool_gather=%s): %s",
                self._display_name,
                tokens_per_seq,
                bucket_t,
                use_fused,
                use_gpr,
                reason_hlo,
            )

        self._bucket_fused_io[bkt_key], self._bucket_gpr[bkt_key] = use_fused, use_gpr
        op_name = (
            f"megakernel_runtime::{self.spec.name}_t{tokens_per_seq}"
            f"_b{bucket_t}_{self.mode}"
        )
        op, _, _, _ = build_pooled_step_op(
            self.mesh, self.spec, ex, op_name=op_name, donate_pools=True
        )
        m_rows, max_pages, embed = ex.M, self.max_pages_per_seq, self.embed_tokens
        out_rows = int(bucket.out_rows) if bucket.out_rows is not None else m_rows
        self._bucket_out_rows[bkt_key] = out_rows

        def _step_impl(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            block_tables: torch.Tensor,
            s0: torch.Tensor,
            c0: torch.Tensor,
            s1: torch.Tensor,
            c1: torch.Tensor,
            s2: torch.Tensor,
            c2: torch.Tensor,
            roff: torch.Tensor,
            weights: list[torch.Tensor],
            pools: list[torch.Tensor],
        ) -> torch.Tensor:
            c0_2d = c0 if c0.ndim == 2 else c0.reshape(-1, 1)
            c1_2d = c1 if c1.ndim == 2 else c1.reshape(-1, 1)
            c2_2d = c2 if c2.ndim == 2 else c2.reshape(-1, 1)
            r_1d = roff.reshape(-1)
            p_args = (
                input_ids,
                positions,
                block_tables,
                s0,
                c0_2d,
                r_1d,
                s1,
                c1_2d,
                r_1d,
                s2,
                c2_2d,
                r_1d,
                bucket_t,
                tokens_per_seq,
                max_pages,
                True,
                c0_2d.shape[1] > 1,
            )
            ids_b, pos_b, pt_b, ckpt_b, roff_b = prepare_step_inputs_eager(*p_args)
            hid = embed(ids_b).to(torch.bfloat16)
            if bucket_t < m_rows:
                hid = F.pad(hid, (0, 0, 0, m_rows - bucket_t))
            hid_out = execute_pooled_step_inplace(
                op, pos_b, hid, pt_b, ckpt_b, roff_b, weights, pools
            )
            if hid_out.shape[0] > out_rows:
                hid_out = hid_out[:out_rows].clone()
            elif hid_out.shape[0] < out_rows:
                hid_out = F.pad(hid_out, (0, 0, 0, out_rows - hid_out.shape[0]))
            return hid_out

        if bkt_key == (self._expected_buckets or [None])[-1]:
            self._log_bound()
        return _step_impl

    def _log_bound(self) -> None:
        fused_bkts = [k for k, v in sorted(self._bucket_fused_io.items()) if v]
        gpr_bkts = [k for k, v in sorted(self._bucket_gpr.items()) if v]
        logger.info(
            "%s megakernel bound to %d unified pools (profile=%s, "
            "num_pages=%d, page_size=%d, max_pages_per_seq=%d, decode_buckets=%s, "
            "verify_buckets=%s, conv_dtype=%s, fused_pool_io=%s, fused_buckets=%s, "
            "pool_gather=%s, gpr_buckets=%s)",
            self._display_name,
            len(self.raw_pools),
            self.profile.name,
            self.num_pages,
            self.page_size,
            self.max_pages_per_seq,
            self.decode_buckets,
            self.verify_buckets,
            mk_rt.layout_facts.dtype_name(self.spec.cache_backend.layout.conv_dtype),
            self.fused_pool_io,
            fused_bkts,
            self.pool_gather,
            gpr_bkts,
        )

    def check_compiled_ir(self, debug: Any, bucket: Bucket) -> None:
        """Validate `TpuCompileDebug.stablehlo_forward_text` pool donation."""
        tokens_per_seq, bucket_t = int(bucket.T), int(bucket.B)
        fwd_texts = getattr(debug, "stablehlo_forward_text", None) or ()
        if not fwd_texts or not any(str(t).strip() for t in fwd_texts):
            msg = (
                f"{self._display_name} megakernel torch_tpu compiled IR pool-donation "
                f"check failed for (T={tokens_per_seq}, B={bucket_t}): "
                "empty stablehlo_forward_text"
            )
            self._torchtpu_ir_error = msg
            raise RuntimeError(msg)
        ir_text = "\n".join(str(t) for t in fwd_texts)
        kvh = max(1, int(self.raw_pools[0].shape[1]) // 2)
        pinfo = mk_rt.pool_schedule_info(self.spec)
        ir_kw = dict(
            num_pages=self.num_pages,
            page_size=self.page_size,
            kvh=kvh,
            hd=self.geom.HD,
            num_pools=pinfo.num_pools,
        )
        ok_ir, reason_ir = mk_rt.verify_compiled_step_ir(ir_text, **ir_kw)
        if not ok_ir:
            msg = (
                f"{self._display_name} megakernel torch_tpu compiled IR pool-donation "
                f"check failed for (T={tokens_per_seq}, B={bucket_t}): {reason_ir}"
            )
            self._torchtpu_ir_error = msg
            raise RuntimeError(msg)
        self._torchtpu_ir_checked_buckets.add((tokens_per_seq, bucket_t))

    # --- warmup -----------------------------------------------------------------

    @contextlib.contextmanager
    def warmup_session(self) -> Iterator[None]:
        """Plan the sentinel pool-write checks; on CPU, save and restore the
        pool pages warmup writes; on TPU, require IR coverage afterwards."""
        assert self.view is not None, "bind() must run before warmup"
        view = self.view
        dev = self.raw_pools[0].device
        is_eager = view.enforce_eager
        w_kw = dict(
            num_pages=self.num_pages,
            max_pages_per_seq=self.max_pages_per_seq,
            ckpt_window=self._ckpt_window,
            runner_tok_pads=view.token_paddings,
            req_block_pairs=view.req_block_pairs,
            is_eager=is_eager,
        )
        warmup_steps, warmed_buckets, n_save = mk_rt.build_warmup_plan(
            self.spec,
            self.decode_buckets,
            self.verify_buckets,
            self._bucket_fused_io,
            self._bucket_gpr,
            **w_kw,
        )
        self._warmup_plan = {
            (s.tokens_per_seq, s.bucket_t, s.tok_pad, s.req_pad, s.blk_pad): s
            for s in warmup_steps
        }
        # Note: on TPU, the warmup sentinel writes to unallocated KV/GDN pages
        # per (T, fused, gpr) kernel variant without restoring them, avoiding
        # slice-writes (`as_strided_copy_inverse`) on the 2.30 GiB pools.
        saved_page0 = (
            [p[:n_save].clone() for p in self.raw_pools] if dev.type != "tpu" else None
        )
        try:
            yield
            if dev.type == "tpu":
                mk_rt.verify_ir_coverage(
                    warmed_buckets,
                    self._bucket_fused_io,
                    self._torchtpu_ir_checked_buckets,
                    is_eager=is_eager,
                    ir_error=self._torchtpu_ir_error,
                    model_label=self._display_name,
                )
        finally:
            self._warmup_plan = None
            if saved_page0 is not None:
                for p, saved in zip(self.raw_pools, saved_page0, strict=True):
                    p[:n_save].copy_(saved)

    def warmup_inputs(self, bucket: Bucket, shape: RunnerShape) -> WarmupCase | None:
        """Inputs for one warmup call, with the sentinel check if planned."""
        assert self._warmup_plan is not None and self.view is not None
        T, b = int(bucket.T), int(bucket.B)
        s = self._warmup_plan.get((T, b, shape.tok_pad, shape.req_pad, shape.blk_pad))
        if s is None:
            return None
        tok_pad, req_pad, blk_pad, c_cols, sent = (
            s.tok_pad,
            s.req_pad,
            s.blk_pad,
            s.c_cols,
            s.sentinel,
        )
        dev = self.raw_pools[0].device
        pinfo = mk_rt.pool_schedule_info(self.spec)
        pps, rpp = pinfo.pages_per_slot, pinfo.rec_pages_per_slot
        g_cnt = pinfo.recurrent_slots_per_pool
        roff_buf = self.view.mamba_slot_read_offsets
        assert self._zero_roff_1d is not None
        r_raw = roff_buf if isinstance(roff_buf, torch.Tensor) else self._zero_roff_1d
        uses_mrope = self.view.uses_mrope
        ids_b = torch.zeros((tok_pad,), dtype=torch.int32, device=dev)
        pos_b = torch.zeros(
            (3, tok_pad) if uses_mrope else (tok_pad,),
            dtype=torch.int32,
            device=dev,
        )
        bt_b = torch.zeros((req_pad * blk_pad,), dtype=torch.int32, device=dev)
        s_list = [
            torch.zeros((req_pad,), dtype=torch.int32, device=dev) for _ in range(g_cnt)
        ]
        c_list = (
            [
                torch.zeros((req_pad, c_cols), dtype=torch.int32, device=dev)
                for _ in range(g_cnt)
            ]
            if c_cols > 1
            else list(s_list)
        )
        r_call = r_raw

        sent_kv: list[np.ndarray] | None = None
        sent_gdn: list[np.ndarray] | None = None
        if sent is not None:
            kv_page, gdn_base, gdn_p0, sub_step = (
                sent.kv_page,
                sent.gdn_base,
                sent.gdn_p0,
                sent.sub_step,
            )
            if dev.type != "tpu":
                for p in self.raw_pools:
                    p[kv_page].view(torch.uint8).fill_(0x7F)
                    p[gdn_p0 : gdn_p0 + pps].view(torch.uint8).fill_(0xFF)
            sent_kv = [
                p[kv_page].view(torch.uint8).cpu().numpy().copy()
                for p in self.raw_pools
            ]
            sent_gdn = [
                p[gdn_p0 : gdn_p0 + rpp].view(torch.uint8).cpu().numpy().copy()
                for p in self.raw_pools
            ]
            ids_b[:T], bt_b[0] = 1, kv_page
            if T > 1:
                pos_b[..., :T] = torch.arange(T, dtype=torch.int32, device=dev)
            for sub_i in range(g_cnt):
                b0 = gdn_base + sub_i * sub_step
                s_list[sub_i][0] = b0
                if c_cols > 1:
                    n_c = min(T, c_cols)
                    c_list[sub_i][0, :n_c] = torch.arange(
                        b0, b0 + n_c, dtype=torch.int32, device=dev
                    )
            r_call = torch.full_like(r_raw, -1)
            if sent.disjoint_subs and self.spec.cache_backend is not None:
                meta_s = mk_rt.pack_step_meta(
                    ids_b.cpu(),
                    pos_b.cpu(),
                    bt_b.cpu(),
                    tuple(
                        (s_t.cpu(), c_t.cpu(), r_call.cpu())
                        for s_t, c_t in zip(s_list, c_list, strict=True)
                    ),
                    bucket_t=b,
                    tokens_per_seq=T,
                    max_pages_per_seq=self.max_pages_per_seq,
                    has_roff=True,
                    use_ckpt_2d=c_cols > 1,
                    ops=TorchStepMetaOps(),
                )
                ex_s, _, _ = mk_rt.bucket_exec_spec(
                    self.spec,
                    self.base_exec,
                    self.profile,
                    bucket_t=b,
                    tokens_per_seq=T,
                    fused_pool_io=self.fused_pool_io,
                    fused_pool_io_max_b=self.fused_pool_io_max_b,
                    fused_pool_io_min_b_hi=self.fused_pool_io_min_b_hi,
                )
                pool_pkg.check_step_pool_metadata(
                    self.spec.cache_backend.layout,
                    ex_s,
                    meta_s.pt_b.numpy(),
                    meta_s.pos_b.numpy(),
                    ckpt_indices=meta_s.ckpt_b.numpy(),
                    read_offsets=meta_s.roff_b.numpy(),
                )

        sc_args = [val for pair in zip(s_list, c_list, strict=True) for val in pair]
        args = (ids_b, pos_b, bt_b, *sc_args, r_call, self.w_flat, self.raw_pools)

        def check(out_b: torch.Tensor) -> None:
            if dev.type == "tpu":
                from vllm_torchtpu.utils import synchronize_tensors

                synchronize_tensors(self.raw_pools)
            if sent is None:
                return
            assert sent_kv is not None and sent_gdn is not None
            kv_after = [
                p[sent.kv_page].view(torch.uint8).cpu().numpy() for p in self.raw_pools
            ]
            gdn_after = [
                p[sent.gdn_p0 : sent.gdn_p0 + rpp].view(torch.uint8).cpu().numpy()
                for p in self.raw_pools
            ]
            s_kw = dict(tokens_per_seq=T, bucket_t=b, variant_key=sent.variant_key)
            mk_rt.verify_sentinel_step(
                out_b[:b].detach().float().cpu().numpy(),
                sent_kv,
                kv_after,
                sent_gdn,
                gdn_after,
                **s_kw,
            )

        return WarmupCase(args, check)

    # --- per step -------------------------------------------------------------

    def decide(self, ctx: Any, fwd: ForwardView) -> mk_rt.StepDecision | None:
        """Route this step to a bucket, or decline (`None`: no metadata, stock
        without counting)."""
        self._prepared = None
        input_ids, positions = ctx.input_ids, ctx.positions
        md = fwd.attn_metadata
        if not isinstance(md, dict) or not md:
            return None
        attn_md = md.get(self.attn_layer_name)
        if attn_md is None:
            return None
        m_mds = [md.get(n) for n in self.mamba_layer_names]
        if not m_mds or any(m is None for m in m_mds):
            return None
        m0 = m_mds[0]
        if (
            getattr(m0, "mamba_state_indices", None) is None
            and getattr(m0, "mamba_ckpt_indices", None) is None
        ):
            return None

        bs = fwd.batch_shape(attn_md, m0)
        active_rows = bs.num_reqs * bs.tokens_per_seq
        mrope_ok = True
        if bs.eligible and bs.num_reqs > 0 and positions is not None:
            if positions.ndim == 2:
                if (
                    int(positions.shape[0]) != 3
                    or int(positions.shape[1]) < active_rows
                ):
                    mrope_ok = False
                else:
                    mrope_np = fwd.mrope_positions
                    if mrope_np is not None and mrope_np.shape[1] >= active_rows:
                        mrope_ok = mk_rt.check_mrope_uniform(mrope_np, active_rows)
                    elif positions.device.type == "cpu":
                        mrope_ok = mk_rt.check_mrope_uniform(
                            positions.detach().numpy(), active_rows
                        )
                    else:
                        mrope_ok = False
            elif positions.ndim > 2:
                mrope_ok = False

        decision = mk_rt.check_step_eligibility(
            bs,
            self.bucket_policy,
            fwd.steps,
            self._bucket_out_rows,
            input_rows=int(input_ids.shape[0]),
            mrope_ok=mrope_ok,
        )
        if decision.eligible:
            self._prepared = (attn_md, m_mds)
        return decision

    def forward(
        self, decision: mk_rt.StepDecision | None, ctx: Any, fwd: ForwardView
    ) -> torch.Tensor:
        """Execute one full-backbone decode (`T=1`) or MTP verify (`T=4`) step
        (the runtime calls it under `torch.no_grad()`)."""
        input_ids, positions = ctx.input_ids, ctx.positions
        prepared, self._prepared = self._prepared, None
        if decision is not None and prepared is not None:
            attn_md, m_mds = prepared
            num_reqs, tokens_per_seq = decision.num_reqs, decision.tokens_per_seq
            step_fn = fwd.steps[(tokens_per_seq, decision.bucket_t)]
        else:
            md = fwd.attn_metadata
            assert md is not None, "no forward-context attention metadata"
            attn_md = md[self.attn_layer_name]
            m_mds = [md[n] for n in self.mamba_layer_names]
            bs = fwd.batch_shape(attn_md, m_mds[0])
            num_reqs, tokens_per_seq = bs.num_reqs, (bs.tokens_per_seq or 1)
            active_rows = max(tokens_per_seq, num_reqs * tokens_per_seq)
            bkt = self.bucket_policy.select(active_rows, tokens_per_seq, clamp=True)
            step_fn = fwd.steps[(tokens_per_seq, bkt)]

        num_tok = fwd.num_tokens
        out_rows = int(num_tok if num_tok is not None else input_ids.shape[0])
        fb_1d = getattr(attn_md, "seq_lens", None)
        if fb_1d is None:
            fb_1d = torch.zeros(
                (max(1, num_reqs),), dtype=torch.int32, device=input_ids.device
            )

        sc_args = [val for m_obj in m_mds for val in _extract_step_sc(m_obj, fb_1d)]
        m0 = m_mds[0]
        roff = getattr(m0, "mamba_slot_read_offsets", None)
        if roff is None and self.view is not None:
            roff = self.view.mamba_slot_read_offsets
        if roff is None:
            assert self._zero_roff_1d is not None
            roff = self._zero_roff_1d

        active_rows = max(tokens_per_seq, num_reqs * tokens_per_seq)
        if positions is not None and (
            (positions.ndim == 2 and positions.shape[1] >= active_rows)
            or (positions.ndim == 1 and positions.numel() >= active_rows)
        ):
            pos_in = positions
        elif tokens_per_seq > 1:
            offsets = torch.arange(
                1 - tokens_per_seq, 1, dtype=torch.int32, device=fb_1d.device
            ).unsqueeze(0)
            pos_in = (fb_1d.unsqueeze(1) + offsets).clamp(min=0).reshape(-1)
        else:
            pos_in = (fb_1d - 1).clamp(min=0)

        hidden_out = step_fn(
            input_ids,
            pos_in,
            attn_md.block_tables,
            *sc_args,
            roff,
            self.w_flat,
            self.raw_pools,
        )
        if out_rows == hidden_out.shape[0]:
            return hidden_out
        if out_rows <= hidden_out.shape[0]:
            return hidden_out[:out_rows]
        return F.pad(hidden_out, (0, 0, 0, out_rows - hidden_out.shape[0]))


__all__ = [
    "DEFAULT_PROFILE",
    "OPTION_KEYS",
    "Qwen35Program",
    "create",
    "parse_options",
    "prepare_step_inputs_eager",
    "validate_hf_config",
]
