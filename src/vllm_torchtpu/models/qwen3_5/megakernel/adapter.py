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
"""`Qwen35Program`: the Qwen3.5 megakernel as a `MonoProgram`.

`create` checks the checkpoint and config without loading anything; `load`
packs the weights (EP->TP reshard first when needed); `bind` aliases the
runner's unified KV+GDN pools and checks the recurrent slot layout; `build`
returns one exported-bridge step per bucket; `warmup_session` /
`warmup_inputs` run the sentinel pool-write check; `decide` / `forward` route
one decode (`T=1`) or MTP verify (`T=4`) step. With the `mtp_draft` option,
`attach_draft` also runs the MTP draft layer on an attention-only megakernel
(`draft.py`) inside the drafter's own programs.
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
from vllm_torchtpu.megakernel.adapter import (
    TorchStepMetaOps,
    build_pooled_step_op,
    execute_pooled_step_inplace,
    find_backbone,
    get_routed_experts,
    load_vllm_weights,
    reshard_layer_experts_ep_to_tp,
)
from vllm_torchtpu.megakernel.adapter.plugin import ForwardView, RunnerView
from vllm_torchtpu.megakernel.adapter.pool_facts import (
    conv_qk_pair_layout_enabled,
    framework_pool_layout_facts,
)
from vllm_torchtpu.megakernel.core import (
    CacheExec,
    ExecSpec,
    MegakernelLayoutError,
    ModelDef,
    Profile,
    with_cache,
)
from vllm_torchtpu.megakernel.core import pool as pool_pkg
from vllm_torchtpu.megakernel.core import runtime as mk_rt
from vllm_torchtpu.megakernel.core.runtime.contract import (
    Bucket,
    Decline,
    LayoutDecline,
    RunnerShape,
    WarmupCase,
)

from . import draft as draft_mk
from . import profiles
from .model import Qwen35, register

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
        "mtp_draft",
        "mtp_draft_buckets",
        "mtp_draft_verify_buckets",
    }
)
# Options of the MTP draft megakernel (`draft.py`): `mtp_draft` turns it on
# (default off); `mtp_draft_buckets` are the row buckets of a draft loop step
# (one row per request) and `mtp_draft_verify_buckets` those of draft step 0
# (K + 1 rows per request), each a multiple of 4 in 4..64.
DRAFT_OPTION_KEYS = frozenset(
    {"mtp_draft", "mtp_draft_buckets", "mtp_draft_verify_buckets"}
)
_TRUE_STRS = frozenset({"1", "true", "on", "yes"})
_FALSE_STRS = frozenset({"0", "false", "off", "no", "none", ""})
# The profile `create` serves when `options` names none: the candidate30 goal
# configuration (approximate numerics). `qwen35-baseline` (exact numerics) is
# one option away: `{"options": {"profile": "qwen35-baseline"}}`.
DEFAULT_PROFILE: str | None = profiles.DEFAULT_PROFILE.name
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


def _parse_flag(name: str, val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    s = str(val).strip().lower()
    _need(s in _TRUE_STRS or s in _FALSE_STRS, f"invalid {name}={val!r}")
    return s in _TRUE_STRS


def parse_draft_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """The MTP draft megakernel's settings from the plugin `options`."""
    raw = options.get("mtp_draft")
    return {
        "mtp_draft": False if raw is None else _parse_flag("mtp_draft", raw),
        "mtp_draft_buckets": draft_mk.parse_draft_buckets(
            options.get("mtp_draft_buckets"), draft_mk.DEFAULT_LOOP_BUCKETS
        ),
        "mtp_draft_verify_buckets": draft_mk.parse_draft_buckets(
            options.get("mtp_draft_verify_buckets"), draft_mk.DEFAULT_VERIFY_BUCKETS
        ),
    }


def parse_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """This template's settings from the plugin `options` (unknown keys raise)."""
    if bad := sorted(options.keys() - OPTION_KEYS):
        raise ValueError(
            f"Qwen35Program: unknown option {bad[0]!r}; known: {sorted(OPTION_KEYS)}"
        )
    register()  # profile names resolve through the library registry
    cfg = mk_rt.parse_runtime_config(options)
    cfg.update(parse_draft_options(options))
    return cfg


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


def dcp_size_decline(dcp: int, tp_size: int, num_kv_heads: int) -> Decline | None:
    """Decline a DCP size the kernel can't shard, else `None`.

    vLLM builds DCP groups as contiguous runs of `dcp` ranks inside TP and
    gives TP rank `r` the KV head `r // (tp_size / num_kv_heads)`, so the
    kernel's `tp_size / num_kv_heads` ranks per KV head split into shards of
    `dcp` ranks only when `dcp` divides them (TP8 with two KV heads: dcp 1, 2
    or 4). The kernel's DCP rank is then `tp_rank % dcp`.
    """
    per_head = tp_size // num_kv_heads if num_kv_heads > 0 else 0
    if dcp == 1 or (per_head > 0 and per_head % dcp == 0):
        return None
    return Decline(
        "dcp_geometry_unsupported",
        f"Megakernel: decode_context_parallel_size={dcp} must divide the "
        f"{per_head} ranks per KV head (tp_size={tp_size}, "
        f"num_key_value_heads={num_kv_heads}); skipping.",
    )


def validate_hf_config(
    hf_cfg: Any, model_def: type[ModelDef] = Qwen35
) -> mk_rt.EligibilityResult:
    """Whether the checkpoint's HF config is the model this template serves."""
    if hf_cfg is None:
        return mk_rt.EligibilityResult(True)
    return model_def.validate_hf_config(hf_cfg)


def create(view: RunnerView, options: Mapping[str, Any]) -> Qwen35Program | Decline:
    """Check the checkpoint and config; build an unloaded program, or decline.

    Runs no collective and loads no weights (the mesh doesn't exist yet). The
    chip, TP size, parallelism, features and required server settings are the
    adapter's checks (`SPEC`), done before this runs.
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
    exp_l, tot_e = d_geom.L, d_geom.E
    dcp = view.dcp_size
    dec = dcp_size_decline(dcp, view.tp_size, d_geom.NKV)
    if dec is not None:
        return dec
    prof = mk_rt.resolve_driver_profile(cfg["profile"], model_name=mdef.name)
    el = mk_rt.check_startup_eligibility(
        num_layers=len(backbone.layers),
        expected_num_layers=exp_l,
        unified_cache_layout=view.unified_kv_layout,
    )
    if el.code == "unaligned_cache_layout":
        el = dataclasses.replace(el, message=UNALIGNED_LAYOUT_MESSAGE)
    if not el.eligible:
        return Decline(el.code, el.message, el.level)
    loaded_ep = get_routed_experts(backbone.layers[0].mlp).w13_weight.shape[0] < tot_e
    mode = cfg["mode"] or ("ep" if loaded_ep else "tp")
    owned = mk_rt.explicit_profile_settings(cfg)
    b_policy = mk_rt.BucketPolicy.resolve(
        prof,
        owned["buckets"],
        owned["verify_buckets"],
        speculative_tokens=view.speculative_tokens,
        buckets_explicit=cfg["buckets_explicit"],
        verify_buckets_explicit=cfg["verify_buckets_explicit"],
    )
    el_post = mk_rt.check_post_mesh_eligibility(
        tp_size=view.tp_size, bucket_policy=b_policy
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
        compile_hlo_check=cfg["compile_hlo_check"],
        dcp_size=dcp,
        mtp_draft=cfg["mtp_draft"],
        mtp_draft_buckets=cfg["mtp_draft_buckets"],
        mtp_draft_verify_buckets=cfg["mtp_draft_verify_buckets"],
    )


class Qwen35Program:
    """The Qwen3.5 megakernel on one TP rank (a `MonoProgram`).

    Passing `mesh` loads immediately (direct use and tests); `create` leaves
    it to `load(mesh)`, after the adapter's first agreement.

    `dcp_size` is the decode context parallel size: each of the `dcp_size`
    ranks sharing a KV head holds one shard of its pages (`pool.dcp`).

    `mtp_draft` (default off) runs the MTP draft layer on an attention-only
    megakernel: `attach_draft` builds it over `mtp_draft_buckets` (loop) and
    `mtp_draft_verify_buckets` (draft step 0) and hands it to the drafter.
    """

    # Decode context parallel size (`create` passes the runner's); 1: no DCP.
    dcp_size: int = 1

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
        dcp_size: int = 1,
        mtp_draft: bool = False,
        mtp_draft_buckets: Sequence[int] | None = None,
        mtp_draft_verify_buckets: Sequence[int] | None = None,
    ) -> None:
        register()
        _need(mode in ("ep", "tp"), f"invalid mode={mode!r}; expected 'ep' or 'tp'")
        self.model_def: type[ModelDef] = model_def
        self.model, self.backbone = model, find_backbone(model)
        self.embed_tokens = self.backbone.embed_tokens
        self.tp_rank, self.mode = tp_rank, mode
        self.dcp_size = int(dcp_size)
        self._spec_arg, self._geom_arg = spec, geom
        self._base_exec_arg, self._weights_arg = base_exec, weights
        self._reshard_ep_to_tp, self._expert_placement = (
            reshard_ep_to_tp,
            expert_placement,
        )
        model_name = getattr(spec, "name", "") or model_def.name
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
        self.fused_pool_io_max_b = self.profile.fused_pool_io_max_b
        self.fused_pool_io_min_b_hi = self.profile.fused_pool_io_min_b_hi
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
        self.mtp_draft = bool(mtp_draft)
        self.mtp_draft_buckets = tuple(
            draft_mk.DEFAULT_LOOP_BUCKETS
            if mtp_draft_buckets is None
            else mtp_draft_buckets
        )
        self.mtp_draft_verify_buckets = tuple(
            draft_mk.DEFAULT_VERIFY_BUCKETS
            if mtp_draft_verify_buckets is None
            else mtp_draft_verify_buckets
        )
        self.draft: draft_mk.Qwen35DraftProgram | None = None
        self._drafter: Any = None
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
                get_routed_experts(self.backbone.layers[0].mlp).w13_weight.shape[0]
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
        """Build the spec and pack the weights (after the adapter's agreement:
        the EP->TP reshard all-gathers across ranks)."""
        self.mesh = mesh
        r_cnt = mesh.devices.size
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
        num_pages = pools[0].shape[0]
        page_size = pools[0].shape[-1]
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
        self.raw_pools, self.num_pages, self.page_size = pools, num_pages, page_size
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
        if self.dcp_size > 1:
            c_kw["dcp"] = self.dcp_size
            logger.info(
                "%s megakernel binds under decode context parallelism: dcp=%d, "
                "dcp_rank=%d (KV pages p %% %d == dcp_rank at column p // %d)",
                self._display_name,
                self.dcp_size,
                self._dcp_rank,
                self.dcp_size,
                self.dcp_size,
            )
        self.base_exec = mk_rt.align_cache_exec(
            self.base_exec, self.profile, self.geom, **c_kw
        )
        return self.pool_layout_facts

    @property
    def _dcp_rank(self) -> int:
        """The DCP rank the kernel assumes: `tp_rank % dcp_size` (0 without
        DCP); the layout facts check it against the runner's DCP group."""
        return self.tp_rank % self.dcp_size if self.dcp_size > 1 else 0

    def _framework_slot_facts(
        self, view: RunnerView, linear_attn: Any, pool: torch.Tensor
    ) -> mk_rt.PoolLayoutFacts:
        block = view.cache_block_size
        if block is None:
            raise ValueError("runner has no cache_config.block_size")
        dcp_kw: dict[str, int] = {}
        if view.dcp_size > 1:
            dcp_kw = dict(
                dcp_size=view.dcp_size,
                dcp_rank=view.dcp_rank,
                dcp_interleave=view.dcp_interleave,
            )
        return framework_pool_layout_facts(
            pool=pool,
            linear_attn=linear_attn,
            block_size=block,
            conv_qk_pair=conv_qk_pair_layout_enabled(),
            **dcp_kw,
        )

    def _adopt_framework_conv_dtype(self, conv_dtype: str) -> None:
        """Store the conv window in the dtype the framework's pool uses.

        The split pool I/O path, fused pool I/O and in-kernel gather (GPR) all
        implement an f32 conv window, so only the pool layout changes: the
        `fused_pool_io` / `pool_gather` options (the profile's, or explicit
        ones) apply as for bf16 conv, and each bucket still resolves them
        against the layout's limits. Other dtypes are left to the layout
        comparison, which declines them.

        This replaces `self.spec` before the comparison; on a decline the
        adapter replaces the whole driver, so nothing else sees it.
        `PoolBackend.with_layout` raises `ValueError` (voted `bind_error`) for
        a backend not built by `make_pool_backend`.
        """
        layout = self.spec.cache_backend.layout
        if (
            conv_dtype != "float32"
            or mk_rt.layout_facts.dtype_name(layout.conv_dtype) == "float32"
        ):
            return
        f32_layout = dataclasses.replace(layout, conv_dtype=jax.numpy.float32)
        # `ModelSpec.with_layout` also rebinds the modules (GDN) to the layout.
        self.spec = self.spec.with_layout(f32_layout)
        logger.info(
            "%s megakernel: the framework stores the pooled conv state as fp32; "
            "using an f32 conv window (fused_pool_io=%s, pool_gather=%s)",
            self._display_name,
            self.fused_pool_io,
            self.profile.pool_gather,
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
        dcp_kw = (
            dict(dcp=self.dcp_size, dcp_rank=self._dcp_rank)
            if self.dcp_size > 1
            else {}
        )
        mk_facts = mk_rt.facts_from_pool_layout(
            self.spec.cache_backend.layout, self.base_exec.R, **dcp_kw
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
            tuple(token_paddings),
        )
        for d in (self._bucket_fused_io, self._bucket_gpr):
            d.clear()
        self._hlo_checked_buckets.clear()
        self._torchtpu_ir_checked_buckets.clear()
        keys = list(self.bucket_policy.iter_buckets())
        self._expected_buckets = keys
        return [Bucket(T, b, self.bucket_policy.out_rows(b)) for T, b in keys]

    def build(self, bucket: Bucket) -> Any:
        """Build the custom op and step function for `(bucket.T, bucket.B)`."""
        tokens_per_seq, bucket_t = bucket.T, bucket.B
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
            f"megakernel::{self.spec.name}_t{tokens_per_seq}_b{bucket_t}_{self.mode}"
        )
        op, _, _, _ = build_pooled_step_op(
            self.mesh, self.spec, ex, op_name=op_name, donate_pools=True
        )
        m_rows, max_pages, embed = ex.M, self.max_pages_per_seq, self.embed_tokens
        out_rows = bucket.out_rows if bucket.out_rows is not None else m_rows

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
            "%s megakernel bound to %d unified pools (profile=%s, numerics=%s, "
            "num_pages=%d, page_size=%d, max_pages_per_seq=%d, decode_buckets=%s, "
            "verify_buckets=%s, conv_dtype=%s, fused_pool_io=%s, fused_buckets=%s, "
            "pool_gather=%s, gpr_buckets=%s)",
            self._display_name,
            len(self.raw_pools),
            self.profile.name,
            self.profile.numerics,
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
        tokens_per_seq, bucket_t = bucket.T, bucket.B
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
        kvh = max(1, self.raw_pools[0].shape[1] // 2)
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
        T, b = bucket.T, bucket.B
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
            else s_list
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
            s_kw: dict[str, Any] = dict(
                tokens_per_seq=T, bucket_t=b, variant_key=sent.variant_key
            )
            if self.dcp_size > 1:
                # The sentinel token (position 0, page 0) is written by the DCP
                # rank holding page 0 only; the other shards hold no copy of it.
                s_kw["kv_owner"] = self._dcp_rank == 0
            mk_rt.verify_sentinel_step(
                out_b[:b].detach().float().cpu().numpy(),
                sent_kv,
                kv_after,
                sent_gdn,
                gdn_after,
                **s_kw,
            )

        return WarmupCase(args, check)

    # --- MTP draft ------------------------------------------------------------

    def _draft_decline(self, why: str) -> Decline:
        return Decline(
            "draft_declined",
            f"{self._display_name} MTP draft megakernel can't serve this drafter "
            f"({why}); skipping.",
        )

    def attach_draft(self, drafter: Any) -> bool | Decline | None:
        """Run the drafter's MTP layer on the draft megakernel (`draft.py`).

        Called by the adapter after `bind`, before its bind agreement. `None`:
        no draft requested (`mtp_draft` off, or no MTP drafter), nothing to
        agree on; `True`: attached; a `Decline` (`draft_declined`): requested
        and not servable on this rank, which declines the megakernel on every
        rank (the adapter then calls `detach_draft`). The program builds and
        binds `Qwen35DraftProgram` and hands it to the drafter's
        `attach_draft_megakernel(program) -> bool`.
        """
        if not self.mtp_draft:
            return None
        view = self.view
        assert view is not None, "bind() must run before attach_draft()"
        spec_cfg = getattr(view.vllm_config, "speculative_config", None)
        method = getattr(spec_cfg, "method", None)
        if drafter is None or method != "mtp":
            logger.warning(
                "%s megakernel: mtp_draft is set but the server has no MTP drafter "
                "(speculative method %r); no draft megakernel.",
                self._display_name,
                method,
            )
            return None
        seam = getattr(drafter, "attach_draft_megakernel", None)
        if not callable(seam):
            return self._draft_decline("no attach_draft_megakernel seam")
        if view.speculative_tokens != draft_mk.NUM_SPEC_TOKENS:
            return self._draft_decline(
                f"num_speculative_tokens={view.speculative_tokens} != "
                f"{draft_mk.NUM_SPEC_TOKENS}"
            )
        if view.enforce_eager:
            return self._draft_decline("enforce_eager")
        for attr in ("_draft_replicated", "_draft_kv_shared", "_share_mtp_indices"):
            if getattr(drafter, attr, False):
                return self._draft_decline(f"drafter.{attr} is set")
        predictor = getattr(getattr(drafter, "draft_model", None), "model", None)
        layers = getattr(predictor, "layers", None)
        if layers is None or len(layers) != 1 or not hasattr(predictor, "norm"):
            return self._draft_decline("expected a one-layer MTP predictor")
        sched = getattr(view.vllm_config, "scheduler_config", None)
        max_reqs = int(getattr(sched, "max_num_seqs", 0) or 0)
        try:
            prog = draft_mk.Qwen35DraftProgram(
                layers[0],
                predictor.norm,
                self.mesh,
                profile=self.profile,
                tp_rank=self.tp_rank,
                dcp_size=self.dcp_size,
                loop_buckets=self.mtp_draft_buckets,
                verify_buckets=self.mtp_draft_verify_buckets,
                max_reqs=max_reqs,
                token_paddings=view.token_paddings,
                model_def=self.model_def,
                geom=draft_mk.draft_geometry_of(self.geom),
                base_exec=dataclasses.replace(
                    self.base_exec, T=1, cache=CacheExec(paged_kv=True)
                ),
            )
            prog.bind()
        except ValueError as exc:  # `DraftUnsupported`, and weight packing errors
            logger.warning(
                "%s MTP draft megakernel: %s", self._display_name, exc, exc_info=True
            )
            return self._draft_decline(str(exc))
        if not seam(prog):
            return self._draft_decline("the drafter built no draft program")
        self.draft, self._drafter = prog, drafter
        return True

    def detach_draft(self) -> None:
        """Undo `attach_draft` (the adapter calls it when the megakernel
        declines at bind)."""
        drafter, self._drafter, self.draft = self._drafter, None, None
        detach = getattr(drafter, "detach_draft_megakernel", None)
        if callable(detach):
            detach()

    def on_warmup_done(self) -> None:
        """After the worker's warmup: every draft megakernel program the
        drafter built must have compiled (a cold one would trace on the first
        real draft step)."""
        if self.draft is None:
            return
        check = getattr(self._drafter, "check_draft_megakernel_warm", None)
        if callable(check):
            check()
        self.draft.reset_stats()

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
        active_rows = bs.num_reqs * bs.tokens_per_req
        mrope_ok = True
        if bs.eligible and bs.num_reqs > 0 and positions is not None:
            if positions.ndim == 2:
                if positions.shape[0] != 3 or positions.shape[1] < active_rows:
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
            input_rows=input_ids.shape[0],
            mrope_ok=mrope_ok,
        )
        if decision.eligible:
            self._prepared = (attn_md, m_mds)
        return decision

    def forward(
        self, decision: mk_rt.StepDecision | None, ctx: Any, fwd: ForwardView
    ) -> torch.Tensor:
        """Execute one full-backbone decode (`T=1`) or MTP verify (`T=4`) step
        (the adapter calls it under `torch.no_grad()`)."""
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
            # The batch shape reports any uniform step size; serve only the
            # decode / verify sizes this program compiles (else decode).
            served = bs.tokens_per_req in (1, mk_rt.MTP_VERIFY_TOKENS)
            num_reqs = bs.num_reqs
            tokens_per_seq = bs.tokens_per_req if served else 1
            active_rows = max(tokens_per_seq, num_reqs * tokens_per_seq)
            bkt = self.bucket_policy.select(active_rows, tokens_per_seq, clamp=True)
            step_fn = fwd.steps[(tokens_per_seq, bkt)]

        num_tok = fwd.num_tokens
        out_rows = num_tok if num_tok is not None else input_ids.shape[0]
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
