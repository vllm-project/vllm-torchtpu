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
"""Thin PyTorch/vLLM adapter for the `megakernel` runtime library."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import time
import types
import zlib
from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
import torch
import torch.nn.functional as F

try:
    from vllm.forward_context import get_forward_context
except ImportError:

    def get_forward_context() -> Any:  # type: ignore[misc]
        raise RuntimeError("vllm.forward_context is unavailable")


import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel import (
    CacheExec,
    ExecSpec,
    MegakernelLayoutError,
    ModelDef,
    Profile,
    find_model,
    with_cache,
)
from vllm_torchtpu.megakernel import (
    pool as pool_pkg,
)
from vllm_torchtpu.megakernel import (
    runtime as _rt,
)
from vllm_torchtpu.megakernel_runtime.exported_bridge import (
    build_pooled_step_op,
    execute_pooled_step_inplace,
)
from vllm_torchtpu.megakernel_runtime.hooks import StepContext, is_suppressed
from vllm_torchtpu.megakernel_runtime.pool_facts import (
    conv_qk_pair_layout_enabled,
    framework_pool_layout_facts,
    runner_block_size,
)
from vllm_torchtpu.megakernel_runtime.weight_collector import (
    find_backbone,
    get_routed_experts,
    load_vllm_weights,
    reshard_layer_experts_ep_to_tp,
)

logger = init_logger(__name__)

_VLLM_STARTUP_MESSAGES: dict[str, str] = {
    "unaligned_cache_layout": (
        "Megakernel requires unified KV+GDN cache layout "
        "(_unified_kv_layout=True); skipping."
    ),
    "multimodal_inputs": (
        "Megakernel requires text-only input_ids "
        "(runner.supports_mm_inputs=True passes inputs_embeds; "
        'pass --limit-mm-per-prompt \'{"image":0,"video":0}\' for '
        "text-only serving); skipping."
    ),
    "lora_enabled": "Megakernel does not support LoRA adapters; skipping.",
    "pipeline_parallel": (
        "Megakernel requires single-stage pipeline (PP=1); skipping."
    ),
    "context_parallel": (
        "Megakernel does not support prefill/decode context "
        "parallelism (PCP/DCP); skipping."
    ),
    "dp_lockstep": "Megakernel does not support EP-DP lockstep; skipping.",
    "aux_hidden_states": (
        "Megakernel does not support aux_hidden_state_layers "
        "(EAGLE-3 target capture); skipping."
    ),
}


def _need_layout(cond: bool, msg: str) -> None:
    if not cond:
        raise MegakernelLayoutError(f"MegakernelDriver: {msg}")


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"MegakernelDriver: {msg}")


def _unwrap(x: Any) -> Any:
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def _collect_runtime_config(
    runner: Any = None, *, raw: Any = None, additional_config: Any = None
) -> dict[str, Any]:
    env_val = envs.VLLM_TPU_MEGAKERNEL if raw is None else raw
    sources: list[Any] = [env_val]
    if additional_config is not None:
        sources.append(
            additional_config.get("megakernel", additional_config)
            if isinstance(additional_config, dict)
            else additional_config
        )
    r_cfg = getattr(getattr(runner, "vllm_config", None), "additional_config", None)
    if isinstance(r_cfg, dict) and "megakernel" in r_cfg:
        sources.append(r_cfg["megakernel"])
    return _rt.parse_runtime_config(*sources)


def _resolve_tp_size(runner: Any) -> int:
    p_cfg = getattr(runner, "parallel_config", None)
    tp_size = int(getattr(p_cfg, "tensor_parallel_size", 0) or 0)
    if tp_size > 0:
        return tp_size
    mesh = getattr(runner, "mesh", None)
    devs = getattr(mesh, "devices", None) if mesh is not None else None
    return int(devs.size) if devs is not None else 0


def _get_pallas_mesh_fn() -> Any:
    """Return `torch_tpu`'s `get_pallas_mesh`, or `None` without `torch_tpu`."""
    try:
        from torch_tpu._internal.pallas import get_pallas_mesh
    except ImportError:
        return None
    return get_pallas_mesh


class MeshDecline(Exception):
    """`_build_tp_rank_mesh` can't build a correctly ordered TP mesh."""

    def __init__(self, result: _rt.EligibilityResult) -> None:
        super().__init__(result.message)
        self.result = result


class LayoutDecline(MegakernelLayoutError):
    """The runner's cache layout isn't the one the megakernel was built for."""

    def __init__(self, result: _rt.EligibilityResult) -> None:
        super().__init__(f"MegakernelDriver: {result.message}")
        self.result = result


def _build_tp_rank_mesh(r_cnt: int, fallback_mesh: Any = None) -> Any:
    """Build an `r_cnt`-device JAX `Mesh` with axis `'tp'` in TP-rank order.

    With `torch_tpu` this is always its Pallas mesh: `torch_tpu` binds
    partition `p` to TP rank `p` and requires every Pallas kernel in a program
    to share one device assignment, while the runner mesh may list the devices
    in another order (e.g. ICI ring order), which would send the megakernel's
    remote DMAs, addressed by mesh position, to the wrong peers. Raises
    `MeshDecline` (`pallas_mesh_unavailable` / `pallas_mesh_mismatch`) when
    `torch_tpu` is present but its mesh is unavailable or doesn't hold `r_cnt`
    devices on axis `'tp'`. Without `torch_tpu` (CPU), the runner mesh is used,
    then `jax.devices()`.
    """
    get_pallas_mesh = _get_pallas_mesh_fn()
    if get_pallas_mesh is not None:
        try:
            pallas_mesh = get_pallas_mesh(axis_names=("tp",))
        except Exception as exc:
            msg = f"Megakernel: get_pallas_mesh failed ({exc}); skipping."
            raise MeshDecline(
                _rt.EligibilityResult(False, "pallas_mesh_unavailable", msg)
            ) from exc
        n_dev = int(pallas_mesh.devices.size)
        axes = tuple(pallas_mesh.axis_names)
        if n_dev != r_cnt or axes != ("tp",):
            msg = (
                f"Megakernel: the Pallas mesh has {n_dev} devices on axes {axes}, "
                f"expected {r_cnt} on ('tp',); skipping."
            )
            raise MeshDecline(_rt.EligibilityResult(False, "pallas_mesh_mismatch", msg))
        return pallas_mesh
    fb_devs = getattr(fallback_mesh, "devices", None)
    if fb_devs is not None and int(fb_devs.size) == r_cnt:
        if tuple(fallback_mesh.axis_names) == ("tp",):
            return fallback_mesh
        return jax.sharding.Mesh(np.asarray(fb_devs).reshape(-1), ("tp",))
    return jax.sharding.Mesh(np.asarray(jax.devices()[:r_cnt]), ("tp",))


class TorchStepMetaOps:
    """`StepMetaOps` primitives over `torch.Tensor` (Dynamo `fullgraph=True` safe)."""

    def shape(self, x: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(d) for d in x.shape)

    def ndim(self, x: torch.Tensor) -> int:
        return int(x.ndim)

    def numel(self, x: torch.Tensor) -> int:
        return int(x.numel())

    def reshape(self, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return x.reshape(shape)

    def slice_or_pad_1d(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x1 = x.reshape(-1)
        n = int(x1.shape[0])
        out = x1[:target_len] if n >= target_len else F.pad(x1, (0, target_len - n))
        return out.to(torch.int32)

    def slice_or_pad_2d(
        self, x: torch.Tensor, target_rows: int, target_cols: int
    ) -> torch.Tensor:
        r, c = int(x.shape[0]), int(x.shape[1])
        out = x[: min(r, target_rows), : min(c, target_cols)]
        pad_r, pad_c = max(0, target_rows - r), max(0, target_cols - c)
        if pad_r > 0 or pad_c > 0:
            out = F.pad(out, (0, pad_c, 0, pad_r), value=0)
        return out.to(torch.int32)

    def zeros_i32(self, shape: tuple[int, ...], *, like: torch.Tensor) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.int32, device=like.device)

    def zeros_like_i32(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def arange_i32(self, start: int, stop: int, *, like: torch.Tensor) -> torch.Tensor:
        return torch.arange(start, stop, dtype=torch.int32, device=like.device)

    def clip(self, x: torch.Tensor, lo: int, hi: int | None = None) -> torch.Tensor:
        return x.clamp(min=lo) if hi is None else x.to(torch.long).clamp(min=lo, max=hi)

    def take_1d(self, table_1d: torch.Tensor, idx_1d: torch.Tensor) -> torch.Tensor:
        return table_1d[idx_1d].to(torch.int32)

    def take_col_2d(
        self, table_2d: torch.Tensor, col_idx_1d: torch.Tensor
    ) -> torch.Tensor:
        return table_2d.gather(1, col_idx_1d.unsqueeze(1))

    def expand_1d_to_2d(
        self, x_1d: torch.Tensor, cols: int, *, contiguous: bool = True
    ) -> torch.Tensor:
        out = x_1d.unsqueeze(1).expand(int(x_1d.shape[0]), int(cols))
        return out.contiguous() if contiguous else out

    def concat_cols(self, col0: torch.Tensor, col1: torch.Tensor) -> torch.Tensor:
        return torch.cat([col0, col1], dim=1)

    def stack_dim0(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(xs), dim=0)

    def where(
        self, cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(cond, x, y)

    def gt_zero(self, x: torch.Tensor) -> torch.Tensor:
        return x > 0

    def lt_zero(self, x: torch.Tensor) -> torch.Tensor:
        return x < 0

    def logical_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a & b

    def check_distinct_cols_2d(self, ckpt_g: torch.Tensor, T: int) -> None:
        if torch.compiler.is_compiling() or ckpt_g.device.type != "cpu":
            return
        _rt.check_ckpt_write_cols_distinct(ckpt_g, T)


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
    return _rt.pack_step_meta(
        input_ids, positions, block_tables, groups, **kw
    ).as_tuple()


def _extract_step_sc(
    m_obj: Any, fb_1d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    s = getattr(m_obj, "mamba_state_indices", None)
    c = getattr(m_obj, "mamba_ckpt_indices", None)
    s = (c if c is not None else fb_1d) if s is None else s
    return s, (c if c is not None else s)


def _pcp_or_dcp_enabled(runner: Any) -> bool:
    if getattr(runner, "_pcp_enabled", False) or getattr(runner, "_dcp_enabled", False):
        return True
    p_cfg = getattr(runner, "parallel_config", None)
    if p_cfg is None:
        return False
    pcp = int(getattr(p_cfg, "prefill_context_parallel_size", 1) or 1)
    dcp = int(getattr(p_cfg, "decode_context_parallel_size", 1) or 1)
    return pcp > 1 or dcp > 1


def _data_parallel_enabled(runner: Any, tp_size: int) -> bool:
    """Whether the runner is one of several data-parallel engines, or its EP
    group spans more ranks than its TP group (the megakernel agrees on the TP
    group only). Reads config and group sizes only: no collective."""
    p_cfg = getattr(runner, "parallel_config", None)
    dp = int(os.environ.get("TORCH_TPU_DP_SIZE", "0") or 0) or int(
        getattr(p_cfg, "data_parallel_size", 1) or 1
    )
    return dp > 1 or _ep_group_size() > int(tp_size)


def _ep_group_size() -> int:
    """vLLM's expert-parallel group size (1 when EP is off or unavailable)."""
    try:
        from vllm.distributed.parallel_state import get_ep_group
    except ImportError:
        return 1
    try:
        group = get_ep_group()
    except (AssertionError, AttributeError, RuntimeError):
        return 1
    return int(getattr(group, "world_size", 1) or 1) if group is not None else 1


def _runner_req_block_pairs(runner: Any) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if runner is None:
        return pairs
    attn_shapes_fn = getattr(runner, "_attention_kernel_shapes", None)
    if callable(attn_shapes_fn):
        r_max, b_max, _ = attn_shapes_fn(True)
        pairs.append((int(r_max), int(b_max)))
        if getattr(runner, "most_model_len", None) is not None:
            r_most, b_most, _ = attn_shapes_fn(False)
            if (int(r_most), int(b_most)) not in pairs:
                pairs.append((int(r_most), int(b_most)))
        return pairs
    r_max = getattr(runner, "num_reqs_max_model_len", None)
    b_max = getattr(runner, "max_num_blocks_per_req", None)
    if r_max is not None and b_max is not None:
        pairs.append((int(r_max), int(b_max)))
    if getattr(runner, "most_model_len", None) is not None:
        r_most = getattr(runner, "num_reqs_most_model_len", None)
        b_most = getattr(runner, "num_blocks_per_most_len_req", None)
        if (
            r_most is not None
            and b_most is not None
            and (int(r_most), int(b_most)) not in pairs
        ):
            pairs.append((int(r_most), int(b_most)))
    return pairs


def _runner_hf_config(runner: Any) -> Any:
    model_cfg = getattr(runner, "model_config", None) or getattr(
        getattr(runner, "vllm_config", None), "model_config", None
    )
    return getattr(model_cfg, "hf_config", None) if model_cfg is not None else None


# Decline codes a rank can vote with in `agree_on_decline` (index + 1 is the
# vote; 0 means "can serve"). Unlisted codes vote as `other`.
DECLINE_VOTE_CODES: tuple[str, ...] = (
    "pallas_mesh_unavailable",
    "pallas_mesh_mismatch",
    "cache_layout_incompatible",
    "pool_layout_unknown",
    "pool_layout_mismatch",
    "conv_state_dtype_unsupported",
    "bind_error",
    "other",
)


def _decline_vote(result: _rt.EligibilityResult | None) -> int:
    if result is None:
        return 0
    code = result.code if result.code in DECLINE_VOTE_CODES else "other"
    return DECLINE_VOTE_CODES.index(code) + 1


def _tp_cpu_group() -> Any:
    """The TP group's CPU (gloo) process group, or `None` on a single rank."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    from vllm.distributed import get_tp_group

    tp_group = get_tp_group()
    if int(getattr(tp_group, "world_size", 1)) <= 1:
        return None
    return tp_group.cpu_group


def _all_reduce_max(t: torch.Tensor, group: Any) -> None:
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)


def buckets_token(keys: Sequence[tuple[int, int]]) -> int:
    """A rank-independent 32-bit digest of a set of `(T, B)` bucket keys."""
    return zlib.crc32(repr(sorted(keys)).encode("ascii"))


def agree_on_decline(
    local: _rt.EligibilityResult | None,
    *,
    token: int | None = None,
    group: Any = None,
) -> _rt.EligibilityResult | None:
    """Make a megakernel decline rank-uniform across the TP group.

    Every rank calls this at the same point with its own decline (`None` when
    it can serve) and, optionally, a `token` that must be equal on every rank
    (e.g. `buckets_token` of the compiled buckets). One MAX all-reduce of
    `(vote, token, -token)`: if any rank declined, every rank declines, and
    ranks that could serve get `peer_declined:<code>` with the declining
    rank's code (the highest vote when several decline); if the tokens differ,
    every rank declines with `bucket_set_mismatch`. A rank-local decline while
    the other ranks run the megakernel's collectives would hang the job.
    """
    if group is None:
        group = _tp_cpu_group()
    if group is None:
        return local
    tok = int(token or 0)
    t = torch.tensor([_decline_vote(local), tok, -tok], dtype=torch.int64)
    _all_reduce_max(t, group)
    vote, tok_max, tok_min = int(t[0]), int(t[1]), -int(t[2])
    if vote != 0:
        if local is not None:
            return local
        code = DECLINE_VOTE_CODES[vote - 1]
        msg = f"Megakernel declined on another TP rank ({code}); skipping."
        return _rt.EligibilityResult(False, f"peer_declined:{code}", msg)
    if tok_max != tok_min:
        msg = "Megakernel compiled bucket sets differ across TP ranks; skipping."
        return _rt.EligibilityResult(False, "bucket_set_mismatch", msg)
    return None


class DeclinedDriver:
    """Stands in for a megakernel the adapter enabled but had to decline.

    Every forward falls through to the stock model and is counted as
    `stock_<code>` in the same periodic step-mix stats a serving driver logs,
    so the decline shows up in serving stats and not only in the startup log.
    It holds no weights or compiled programs.
    """

    def __init__(
        self,
        result: _rt.EligibilityResult,
        *,
        tp_rank: int = 0,
        display_name: str = "megakernel",
        stats_period_s: float = 60.0,
    ) -> None:
        self.decline = result
        self.tp_rank, self._display_name = int(tp_rank), display_name
        self.step_mix_tracker = _rt.StepMixTracker(period_s=float(stats_period_s))
        self._step_key = f"stock_{result.code}"

    def bind_and_warmup(self, runner: Any) -> None:
        del runner
        # Join the bind-time agreement the serving drivers on other ranks run.
        agree_on_decline(self.decline)
        logger.warning(
            "%s megakernel declined (%s): %s",
            self._display_name,
            self.decline.code,
            self.decline.message,
        )

    def reset_stats(self) -> None:
        self.step_mix_tracker.reset()

    def try_forward(self, runner: Any, ctx: StepContext) -> tuple[bool, Any]:
        del ctx
        if is_suppressed(runner):
            return False, None
        _, summary = self.step_mix_tracker.note_step(self._step_key, False)
        if summary is not None and self.tp_rank == 0:
            elapsed, mk_cnt, total, pct, snap = summary
            logger.info(
                "%s megakernel step mix over %.0fs: %d/%d MK (%.1f%%) %s",
                self._display_name,
                elapsed,
                mk_cnt,
                total,
                pct,
                snap,
            )
        return False, None


class MegakernelDriver:
    """Model-agnostic megakernel runtime driver for `TPUModelRunner`."""

    @classmethod
    def _resolve_model_def(
        cls,
        model_def: type[ModelDef] | str | None = None,
        *,
        runner: Any = None,
        model: Any = None,
        spec: Any = None,
        geom: Any = None,
    ) -> type[ModelDef] | None:
        if model_def is not None:
            return find_model(model_def)
        if spec is not None:
            if getattr(spec, "name", None) and (m := find_model(spec.name)) is not None:
                return m
            s_cfg = getattr(spec, "config", None)
            if (m := getattr(s_cfg, "_model_def", None)) is not None:
                return m
        if geom is not None and (m := getattr(geom, "_model_def", None)) is not None:
            return m
        if runner is not None:
            hf_cfg = _runner_hf_config(runner)
            if hf_cfg is not None and (m := find_model(hf_cfg)) is not None:
                return m
            if model is None:
                model = getattr(runner, "model", None)
        if model is not None:
            m_cls = type(model).__name__
            if m_cls and m_cls != "NoneType" and (m := find_model(m_cls)) is not None:
                return m
            m_cfg = getattr(model, "config", None)
            if m_cfg is not None and (m := find_model(m_cfg)) is not None:
                return m
        return None

    def __init__(
        self,
        model: Any,
        mesh: Any,
        *,
        model_def: type[ModelDef] | str | None = None,
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
        ab: bool | None = None,
        stats_period_s: float | None = None,
        compile_hlo_check: bool | None = None,
    ) -> None:
        _need(mode in ("ep", "tp"), f"invalid mode={mode!r}; expected 'ep' or 'tp'")
        resolved_mdef = self._resolve_model_def(
            model_def, model=model, spec=spec, geom=geom
        )
        _need(
            resolved_mdef is not None,
            "could not resolve ModelDef for model; pass model_def=... explicitly",
        )
        assert resolved_mdef is not None
        self.model_def: type[ModelDef] = resolved_mdef
        self.model, self.backbone = model, find_backbone(model)
        self.embed_tokens = self.backbone.embed_tokens
        self.mesh, self.tp_rank, self.mode = mesh, int(tp_rank), mode
        r_cnt = int(mesh.devices.size)
        self.spec = spec if spec is not None else self._build_spec(mode, r_cnt, geom)
        self.geom = self.spec.config
        self._display_name = self.model_def.display_name or str(self.spec.name)

        default_cfg = _collect_runtime_config()
        prof_arg = default_cfg["profile"] if profile is None else profile
        p_kw = dict(
            model_name=self.spec.name,
            fused_pool_io=fused_pool_io,
            fused_pool_io_max_b=fused_pool_io_max_b,
            fused_pool_io_min_b_hi=fused_pool_io_min_b_hi,
            pool_gather=pool_gather,
        )
        self.profile: Profile = _rt.resolve_driver_profile(prof_arg, **p_kw)
        chk = (
            default_cfg["compile_hlo_check"]
            if compile_hlo_check is None
            else compile_hlo_check
        )
        self._compile_hlo_check = bool(chk)
        _rt.check_compile_hlo_mesh(self.mesh, self._compile_hlo_check)
        self.fused_pool_io: bool | str = self.profile.fused_pool_io
        self.fused_pool_io_max_b = int(self.profile.fused_pool_io_max_b)
        self.fused_pool_io_min_b_hi = int(self.profile.fused_pool_io_min_b_hi)
        self.pool_gather: bool | None = self.profile.pool_gather

        self.bucket_policy = _rt.BucketPolicy.resolve(
            self.profile, buckets, verify_buckets
        )
        _need(
            bool(self.buckets or self.verify_buckets),
            "at least one of decode buckets or verify_buckets must be non-empty",
        )

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

        if weights is None:
            weights = self._prepare_and_pack_weights(
                mode=mode,
                reshard_ep_to_tp=reshard_ep_to_tp,
                expert_placement=expert_placement,
            )
        self.weights = weights
        self.w_flat: list[torch.Tensor] = self.weights["w_flat"]

        self.raw_pools: list[torch.Tensor] = []
        self.num_pages, self.page_size, self.max_pages_per_seq = 0, 128, 0
        self.mamba_layer_names: list[str] = []
        self.attn_layer_name, self._ckpt_window = "", 1
        self._zero_roff_1d: torch.Tensor | None = None
        self._runner_token_paddings: tuple[int, ...] = ()
        self._bucket_steps: dict[tuple[int, int], Any] = {}
        self._bucket_ops: dict[tuple[int, int], Any] = {}
        self._bucket_out_rows: dict[tuple[int, int], int] = {}
        self._bucket_fused_io: dict[tuple[int, int], bool] = {}
        self._bucket_gpr: dict[tuple[int, int], bool] = {}
        self._bound = False
        self._hlo_checked_buckets: set[tuple[int, int]] = set()
        self._torchtpu_ir_checked_buckets: set[tuple[int, int]] = set()
        self._torchtpu_ir_error: str | None = None
        self._prepared_step: tuple[Any, ...] | None = None
        sp_s = float(
            default_cfg["stats_period_s"] if stats_period_s is None else stats_period_s
        )
        self.step_mix_tracker = _rt.StepMixTracker(period_s=sp_s)
        ab_en = bool(default_cfg["ab"] if ab is None else ab)
        self.ab_calibrator = _rt.MKABCalibrator(enabled=ab_en, period_s=sp_s)

    @property
    def buckets(self) -> tuple[int, ...]:
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

    def _build_spec(self, mode: str, r_cnt: int, geom: Any = None) -> Any:
        del r_cnt
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
                    expert_placement as _ep_mod,  # type: ignore[attr-defined]
                )

                expert_placement = _ep_mod
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

    @classmethod
    def _validate_runner_hf_config(
        cls, runner: Any, model_def: type[ModelDef] | None = None
    ) -> bool:
        hf_cfg = _runner_hf_config(runner)
        if hf_cfg is None:
            return True
        mdef = model_def or cls._resolve_model_def(runner=runner)
        if mdef is None:
            return False
        res = mdef.validate_hf_config(hf_cfg)
        if not res.eligible and res.message:
            logger.warning("%s", res.message)
        return res.eligible

    @staticmethod
    def _log_startup_ineligible(el: _rt.EligibilityResult) -> None:
        msg = _VLLM_STARTUP_MESSAGES.get(el.code, el.message)
        if msg:
            (logger.info if el.level == "info" else logger.warning)("%s", msg)

    @classmethod
    def maybe_create(
        cls, runner: Any, *, model_def: type[ModelDef] | None = None
    ) -> MegakernelDriver | DeclinedDriver | None:
        """Create a `MegakernelDriver` for `runner` if enabled and compatible.

        Returns `None` when the megakernel is off or the model/config is
        ineligible, and a `DeclinedDriver` when it is enabled and eligible but
        the runtime environment can't host it (e.g. no usable Pallas mesh).
        """
        cfg = _collect_runtime_config(runner)
        if not cfg["enabled"]:
            return None
        mdef = model_def or cls._resolve_model_def(runner=runner)
        if mdef is None or not cls._validate_runner_hf_config(runner, model_def=mdef):
            return None
        try:
            backbone = find_backbone(runner.model)
        except ValueError:
            return None
        d_geom = mdef.derive_geometry()
        exp_l, tot_e = int(d_geom.L), int(getattr(d_geom, "E", 0))
        num_layers = len(getattr(backbone, "layers", ()))
        r_cnt = _resolve_tp_size(runner)
        prof = _rt.resolve_driver_profile(cfg["profile"], model_name=mdef.name)
        tp_sup = (
            prof.supported_tp_sizes
            if prof.supported_tp_sizes is not None
            else mdef.supported_tp_sizes
        )
        dp_fn = getattr(runner, "_dp_lockstep_enabled", None)
        pp_multi = not (
            getattr(runner, "_pp_is_first", True)
            and getattr(runner, "_pp_is_last", True)
        )
        el = _rt.check_startup_eligibility(
            num_layers=num_layers,
            expected_num_layers=exp_l,
            unified_cache_layout=bool(getattr(runner, "_unified_kv_layout", False)),
            multimodal_inputs=bool(getattr(runner, "supports_mm_inputs", False)),
            lora_enabled=getattr(runner, "lora_config", None) is not None,
            pipeline_parallel=pp_multi,
            context_parallel=_pcp_or_dcp_enabled(runner),
            dp_lockstep=bool(callable(dp_fn) and dp_fn()),
            aux_hidden_states=bool(getattr(backbone, "aux_hidden_state_layers", ())),
            tp_size=r_cnt,
            supported_tp_sizes=tp_sup,
        )
        if el.eligible and _data_parallel_enabled(runner, r_cnt):
            msg = (
                "Megakernel agrees across the TP group only: data_parallel_size "
                "> 1 or an EP group wider than TP is unsupported; skipping."
            )
            el = _rt.EligibilityResult(False, "dp_unsupported", msg)
        if not el.eligible:
            cls._log_startup_ineligible(el)
            return None

        from vllm.distributed import get_tensor_model_parallel_rank

        tp_rank = int(get_tensor_model_parallel_rank())
        loaded_ep = (
            int(get_routed_experts(backbone.layers[0].mlp).w13_weight.shape[0]) < tot_e
        )
        req_mode = str(cfg["mode"]).lower()
        if req_mode not in ("", "ep", "tp"):
            raise ValueError(
                f"Invalid megakernel mode={req_mode!r}; expected 'ep' or 'tp'"
            )
        mode = req_mode if req_mode else ("ep" if loaded_ep else "tp")
        mesh, mesh_decline = None, None
        try:
            mesh = _build_tp_rank_mesh(
                r_cnt, fallback_mesh=getattr(runner, "mesh", None)
            )
        except MeshDecline as exc:
            mesh_decline = exc.result
        # Before packing weights: an EP->TP reshard all-gathers across ranks.
        mesh_decline = agree_on_decline(mesh_decline)
        if mesh_decline is not None:
            cls._log_startup_ineligible(mesh_decline)
            return DeclinedDriver(
                mesh_decline,
                tp_rank=tp_rank,
                display_name=mdef.display_name or str(mdef.name),
                stats_period_s=float(cfg["stats_period_s"]),
            )
        dev_kind = str(getattr(mesh.devices.flat[0], "device_kind", "") or "")
        owned = _rt.explicit_profile_settings(cfg)
        spec_tokens: int | None = None
        if hasattr(runner, "speculative_config"):
            spec_cfg = getattr(runner, "speculative_config", None)
            spec_tokens = int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0)
        b_policy = _rt.BucketPolicy.resolve(
            prof,
            owned["buckets"],
            owned["verify_buckets"],
            speculative_tokens=spec_tokens,
            buckets_explicit=bool(cfg.get("buckets_explicit", False)),
            verify_buckets_explicit=bool(cfg.get("verify_buckets_explicit", False)),
        )
        el_post = _rt.check_post_mesh_eligibility(
            tp_size=r_cnt,
            mesh_size=int(mesh.devices.size),
            device_kind=dev_kind,
            bucket_policy=b_policy,
        )
        if not el_post.eligible:
            cls._log_startup_ineligible(el_post)
            return None

        return cls(
            model=runner.model,
            mesh=mesh,
            model_def=mdef,
            tp_rank=tp_rank,
            mode=mode,
            buckets=b_policy.buckets,
            verify_buckets=b_policy.verify_buckets,
            reshard_ep_to_tp=(mode == "tp" and loaded_ep),
            profile=cfg["profile"],
            fused_pool_io=owned["fused_pool_io"],
            fused_pool_io_max_b=owned["fused_pool_io_max_b"],
            fused_pool_io_min_b_hi=owned["fused_pool_io_min_b_hi"],
            pool_gather=owned["pool_gather"],
            ab=bool(cfg["ab"]),
            stats_period_s=float(cfg["stats_period_s"]),
            compile_hlo_check=bool(cfg["compile_hlo_check"]),
        )

    def bind_caches(self, runner: Any) -> None:
        """Bind the shared `float8_e4m3fn` raw pools from `backbone.layers`."""
        _need_layout(
            bool(getattr(runner, "_unified_kv_layout", True)),
            "runner._unified_kv_layout must be True "
            "(unified KV+GDN cache layout required)",
        )
        pinfo = _rt.pool_schedule_info(self.spec)
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

        self._check_pool_layout_facts(runner, pools[0])
        self.raw_pools, self.num_pages, self.page_size = (
            pools,
            num_pages,
            int(pools[0].shape[-1]),
        )
        self._zero_roff_1d = torch.zeros(
            (1,), dtype=torch.int32, device=pools[0].device
        )
        spec_cfg = getattr(runner, "speculative_config", None)
        k_spec = int(spec_cfg.num_speculative_tokens) if spec_cfg is not None else 0
        self._ckpt_window = max(
            1, int(getattr(runner, "_mamba_ckpt_window", k_spec + 1))
        )
        attn_shapes_fn = getattr(runner, "_attention_kernel_shapes", None)
        if callable(attn_shapes_fn):
            self.max_pages_per_seq = int(attn_shapes_fn(True)[1])
        else:
            self.max_pages_per_seq = int(getattr(runner, "max_num_blocks_per_req", 2))
        self.mamba_layer_names = [
            self.backbone.layers[g_idx].linear_attn.prefix for g_idx in range(g_per_att)
        ]
        self.attn_layer_name = self.backbone.layers[g_per_att].self_attn.attn.layer_name

        c_kw = dict(
            num_pages=self.num_pages,
            page_size=self.page_size,
            max_pages_per_seq=self.max_pages_per_seq,
        )
        self.base_exec = _rt.align_cache_exec(
            self.base_exec, self.profile, self.geom, **c_kw
        )
        self._runner_token_paddings = tuple(
            int(x) for x in getattr(runner, "num_tokens_paddings", ())
        )
        self.bucket_policy = _rt.BucketPolicy(
            self.buckets, self.verify_buckets, self._runner_token_paddings
        )
        self._build_all_buckets()
        self._bound = True
        fused_bkts = [k for k, v in sorted(self._bucket_fused_io.items()) if v]
        gpr_bkts = [k for k, v in sorted(self._bucket_gpr.items()) if v]
        logger.info(
            "%s megakernel bound to %d unified pools (profile=%s, "
            "num_pages=%d, page_size=%d, max_pages_per_seq=%d, decode_buckets=%s, "
            "verify_buckets=%s, fused_pool_io=%s, fused_buckets=%s, "
            "pool_gather=%s, gpr_buckets=%s)",
            self._display_name,
            len(self.raw_pools),
            self.profile.name,
            self.num_pages,
            self.page_size,
            self.max_pages_per_seq,
            self.buckets,
            self.verify_buckets,
            self.fused_pool_io,
            fused_bkts,
            self.pool_gather,
            gpr_bkts,
        )

    def _framework_slot_facts(
        self, runner: Any, linear_attn: Any, pool: torch.Tensor
    ) -> _rt.PoolLayoutFacts:
        block = runner_block_size(runner)
        if block is None:
            raise ValueError("runner has no cache_config.block_size")
        return framework_pool_layout_facts(
            pool=pool,
            linear_attn=linear_attn,
            block_size=block,
            conv_qk_pair=conv_qk_pair_layout_enabled(),
        )

    def _check_pool_layout_facts(self, runner: Any, pool: torch.Tensor) -> None:
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
        mk_facts = _rt.facts_from_pool_layout(
            self.spec.cache_backend.layout, int(self.base_exec.R)
        )
        try:
            fw_facts = self._framework_slot_facts(runner, linear_attn, pool)
        except (AttributeError, TypeError, ValueError) as exc:
            msg = (
                "can't derive the framework's recurrent pool slot layout "
                f"({type(exc).__name__}: {exc}); skipping."
            )
            raise LayoutDecline(
                _rt.EligibilityResult(False, "pool_layout_unknown", msg)
            ) from exc
        res = _rt.compare_layout_facts(mk_facts, fw_facts)
        if not res.eligible:
            raise LayoutDecline(res)
        self.pool_layout_facts = mk_facts
        logger.info(
            "%s megakernel pool slot layout matches the framework (fingerprint %s)",
            self._display_name,
            mk_facts.fingerprint()[:16],
        )

    def _build_all_buckets(self) -> None:
        self._bucket_steps.clear()
        self._bucket_ops.clear()
        self._bucket_out_rows.clear()
        self._bucket_fused_io.clear()
        self._bucket_gpr.clear()
        self._hlo_checked_buckets.clear()
        self._torchtpu_ir_checked_buckets.clear()
        for T, b in self.bucket_policy.iter_buckets():
            out_rows = self.bucket_policy.out_rows(b)
            step = self._build_bucket_op(b, tokens_per_seq=T, out_pad_rows=out_rows)
            self._bucket_steps[(T, b)] = step
            self._bucket_out_rows[(T, b)] = out_rows

    def _build_bucket_op(
        self, bucket_t: int, tokens_per_seq: int = 1, out_pad_rows: int | None = None
    ) -> Any:
        """Build the custom op and step wrapper for `(tokens_per_seq, bucket_t)`."""
        b_kw = dict(
            bucket_t=bucket_t,
            tokens_per_seq=tokens_per_seq,
            fused_pool_io=self.fused_pool_io,
            fused_pool_io_max_b=self.fused_pool_io_max_b,
            fused_pool_io_min_b_hi=self.fused_pool_io_min_b_hi,
        )
        ex, use_fused, use_gpr = _rt.bucket_exec_spec(
            self.spec, self.base_exec, self.profile, **b_kw
        )
        is_tpu = self.raw_pools[0].device.type == "tpu"
        bkt_key = (tokens_per_seq, bucket_t)
        if bkt_key not in self._hlo_checked_buckets:
            self._hlo_checked_buckets.add(bkt_key)
            h_kw = dict(
                max_pages_per_seq=self.max_pages_per_seq,
                compile_hlo=True if self._compile_hlo_check else None,
                ckpt_cols=2 if tokens_per_seq == 1 else tokens_per_seq,
            )
            ok_hlo, reason_hlo = _rt.verify_pooled_step_hlo(
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
        self._bucket_ops[bkt_key] = op
        m_rows, max_pages, embed = ex.M, self.max_pages_per_seq, self.embed_tokens
        out_rows = int(out_pad_rows) if out_pad_rows is not None else m_rows

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

        # Give each bucket its own code object: nested closures share `__code__`,
        # and Dynamo caches compiled graphs per code object, so all buckets
        # would otherwise compete for one entry's `cache_size_limit`.
        bkt_name = f"step_t{tokens_per_seq}_b{bucket_t}"
        co = _step_impl.__code__.replace(co_name=bkt_name, co_qualname=bkt_name)
        _step_impl = types.FunctionType(
            co,
            _step_impl.__globals__,
            bkt_name,
            _step_impl.__defaults__,
            _step_impl.__closure__,
        )
        if is_tpu:

            def _cb(dbg: Any) -> None:
                self._verify_torchtpu_compile_debug(
                    dbg, tokens_per_seq=tokens_per_seq, bucket_t=bucket_t
                )

            opts = {"name": bkt_name, "debug_callback": _cb, "serializable": False}
            return torch.compile(
                _step_impl, backend="tpu", fullgraph=True, dynamic=False, options=opts
            )
        return _step_impl

    def _verify_torchtpu_compile_debug(
        self, debug: Any, *, tokens_per_seq: int, bucket_t: int
    ) -> None:
        """Validate `TpuCompileDebug.stablehlo_forward_text` pool donation."""
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
        pinfo = _rt.pool_schedule_info(self.spec)
        ir_kw = dict(
            num_pages=self.num_pages,
            page_size=self.page_size,
            kvh=kvh,
            hd=self.geom.HD,
            num_pools=pinfo.num_pools,
        )
        ok_ir, reason_ir = _rt.verify_compiled_step_ir(ir_text, **ir_kw)
        if not ok_ir:
            msg = (
                f"{self._display_name} megakernel torch_tpu compiled IR pool-donation "
                f"check failed for (T={tokens_per_seq}, B={bucket_t}): {reason_ir}"
            )
            self._torchtpu_ir_error = msg
            raise RuntimeError(msg)
        self._torchtpu_ir_checked_buckets.add((tokens_per_seq, bucket_t))

    def reset_stats(self) -> None:
        """Reset step-mix counters and A/B latency accumulators."""
        self.step_mix_tracker.reset()
        self.ab_calibrator.reset()

    @torch.no_grad()
    def warmup(self, runner: Any = None, *, eager_only: bool = False) -> None:
        """Warm compiled bucket steps and run the sentinel pool-write check."""
        if not self._bound:
            raise RuntimeError("bind_caches() must be called before warmup()")
        dev = self.raw_pools[0].device
        pinfo = _rt.pool_schedule_info(self.spec)
        pps, rpp = pinfo.pages_per_slot, pinfo.rec_pages_per_slot
        g_cnt = pinfo.recurrent_slots_per_pool
        runner_tok_pads = tuple(
            int(x) for x in getattr(runner, "num_tokens_paddings", ())
        )
        req_block_pairs = _runner_req_block_pairs(runner)
        is_eager = bool(eager_only or getattr(runner, "enforce_eager", False))
        w_kw = dict(
            num_pages=self.num_pages,
            max_pages_per_seq=self.max_pages_per_seq,
            ckpt_window=self._ckpt_window,
            runner_tok_pads=runner_tok_pads,
            req_block_pairs=req_block_pairs,
            is_eager=is_eager,
        )
        warmup_steps, warmed_buckets, n_save = _rt.build_warmup_plan(
            self.spec,
            self.buckets,
            self.verify_buckets,
            self._bucket_fused_io,
            self._bucket_gpr,
            **w_kw,
        )
        # Note: on TPU, the warmup sentinel writes to unallocated KV/GDN pages
        # per (T, fused, gpr) kernel variant without restoring them, avoiding
        # slice-writes (`as_strided_copy_inverse`) on the 2.30 GiB pools.
        saved_page0 = (
            [p[:n_save].clone() for p in self.raw_pools] if dev.type != "tpu" else None
        )
        roff_buf = (
            getattr(runner, "mamba_slot_read_offsets", None)
            if runner is not None
            else None
        )
        assert self._zero_roff_1d is not None
        r_raw = roff_buf if isinstance(roff_buf, torch.Tensor) else self._zero_roff_1d
        uses_mrope = bool(getattr(runner, "uses_mrope", False))
        try:
            for s in warmup_steps:
                T, b, tok_pad, req_pad, blk_pad, c_cols, sent = (
                    s.tokens_per_seq,
                    s.bucket_t,
                    s.tok_pad,
                    s.req_pad,
                    s.blk_pad,
                    s.c_cols,
                    s.sentinel,
                )
                step_fn = self._bucket_steps[(T, b)]
                ids_b = torch.zeros((tok_pad,), dtype=torch.int32, device=dev)
                pos_b = torch.zeros(
                    (3, tok_pad) if uses_mrope else (tok_pad,),
                    dtype=torch.int32,
                    device=dev,
                )
                bt_b = torch.zeros((req_pad * blk_pad,), dtype=torch.int32, device=dev)
                s_list = [
                    torch.zeros((req_pad,), dtype=torch.int32, device=dev)
                    for _ in range(g_cnt)
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
                        meta_s = _rt.pack_step_meta(
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
                        ex_s, _, _ = _rt.bucket_exec_spec(
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

                sc_args = [
                    val for pair in zip(s_list, c_list, strict=True) for val in pair
                ]
                out_b = step_fn(
                    ids_b,
                    pos_b,
                    bt_b,
                    *sc_args,
                    r_call,
                    self.w_flat,
                    self.raw_pools,
                )
                if dev.type == "tpu":
                    from vllm_torchtpu.utils import synchronize_tensors

                    synchronize_tensors([out_b, *self.raw_pools])
                if sent is not None:
                    assert sent_kv is not None and sent_gdn is not None
                    kv_after = [
                        p[sent.kv_page].view(torch.uint8).cpu().numpy()
                        for p in self.raw_pools
                    ]
                    gdn_after = [
                        p[sent.gdn_p0 : sent.gdn_p0 + rpp]
                        .view(torch.uint8)
                        .cpu()
                        .numpy()
                        for p in self.raw_pools
                    ]
                    s_kw = dict(
                        tokens_per_seq=T, bucket_t=b, variant_key=sent.variant_key
                    )
                    _rt.verify_sentinel_step(
                        out_b[:b].detach().float().cpu().numpy(),
                        sent_kv,
                        kv_after,
                        sent_gdn,
                        gdn_after,
                        **s_kw,
                    )
            if dev.type == "tpu":
                _rt.verify_ir_coverage(
                    warmed_buckets,
                    self._bucket_fused_io,
                    self._torchtpu_ir_checked_buckets,
                    is_eager=is_eager,
                    ir_error=self._torchtpu_ir_error,
                    model_label=self._display_name,
                )
            self.reset_stats()
        finally:
            if saved_page0 is not None:
                for p, saved in zip(self.raw_pools, saved_page0, strict=True):
                    p[:n_save].copy_(saved)

    def bind_and_warmup(self, runner: Any) -> None:
        """Bind runner KV+GDN cache pools and warm up compiled megakernel buckets.

        Only `MegakernelLayoutError` (an incompatible runtime cache layout)
        disables the megakernel: the runner's `_megakernel` becomes a
        `DeclinedDriver` that runs the stock path and reports the reason in the
        step-mix stats. Configuration errors (`MegakernelConfigError`) and any
        other exception propagate. Every TP rank agrees before warmup (which
        runs the megakernel's collectives): if any rank declines, all do.
        """
        precompile_timed = getattr(runner, "_precompile_timed", None)
        eager_only = bool(getattr(runner, "enforce_eager", False))
        ctx = (
            precompile_timed("megakernel")
            if callable(precompile_timed)
            else contextlib.nullcontext()
        )
        with ctx:
            decline: _rt.EligibilityResult | None = None
            try:
                self.bind_caches(runner)
            except MegakernelLayoutError as exc:
                decline = getattr(exc, "result", None) or _rt.EligibilityResult(
                    False, "cache_layout_incompatible", str(exc)
                )
            except BaseException:
                # This rank is raising: still vote, so the others decline
                # instead of waiting in the agreement.
                agree_on_decline(
                    _rt.EligibilityResult(False, "bind_error", "bind failed")
                )
                raise
            token = None if decline else buckets_token(list(self._bucket_steps))
            decline = agree_on_decline(decline, token=token)
            if decline is not None:
                logger.warning(
                    "Disabling %s megakernel due to incompatible KV cache layout: %s",
                    getattr(self, "_display_name", "megakernel"),
                    decline.message,
                )
                self._bound = False
                if runner is not None and hasattr(runner, "_megakernel"):
                    tracker = getattr(self, "step_mix_tracker", None)
                    runner._megakernel = DeclinedDriver(
                        decline,
                        tp_rank=getattr(self, "tp_rank", 0),
                        display_name=getattr(self, "_display_name", "megakernel"),
                        stats_period_s=getattr(tracker, "period_s", 60.0),
                    )
                return
            self.warmup(runner, eager_only=eager_only)

    @classmethod
    def _inspect_step_detail(
        cls, attn_md: Any, m0: Any, runner: Any = None
    ) -> _rt.BatchShape:
        """Return `BatchShape` without host-device synchronization."""
        num_reqs: int | None = None
        has_roff = False
        comb_wt: tuple[int, int] | None = None
        qsl_host: Any = None
        if runner is not None:
            bctx = getattr(runner, "_attn_metadata_builder_ctx", None)
            if bctx is not None and getattr(bctx, "num_reqs", None) is not None:
                num_reqs = int(bctx.num_reqs)
            has_roff = getattr(runner, "mamba_slot_read_offsets", None) is not None
            comb_key = getattr(runner, "_cached_comb_key", None)
            comb_cpu: Any = getattr(runner, "_combined_request_distribution_cpu", None)
            if has_roff and isinstance(comb_key, tuple) and len(comb_key) == 3:
                comb_wt = (int(comb_key[1]), int(comb_key[2]))
            elif (
                has_roff
                and isinstance(comb_cpu, torch.Tensor)
                and comb_cpu.numel() >= 6
            ):
                comb_wt = (int(comb_cpu[3].item()), int(comb_cpu[5].item()))
            qsl_host = getattr(runner, "query_start_loc_np", None)
            if qsl_host is None:
                qsl_host = getattr(runner, "query_start_loc_cpu", None)
            runner_decides = qsl_host is not None and (
                num_reqs is not None or (has_roff and comb_wt is not None)
            )
            if not runner_decides and has_roff and comb_wt is not None:
                eff_n = comb_wt[1] if num_reqs is None else num_reqs
                runner_decides = (
                    comb_wt[1] <= 0 or eff_n <= 0 or comb_wt[0] != comb_wt[1]
                )
            if runner_decides:
                return _rt.inspect_batch_shape(
                    num_reqs=num_reqs,
                    has_roff=has_roff,
                    decode_and_total_reqs=comb_wt,
                    qsl=qsl_host,
                )

        # Fallback for unit tests passing CPU metadata without a full runner instance.
        # Never read device tensors on TPU; fail closed if host/CPU qsl is absent.
        dist_dt: tuple[int, int] | None = None
        dist: Any = getattr(m0, "mamba_request_distribution", None)
        if dist is None:
            dist = getattr(attn_md, "mamba_request_distribution", None)
        if dist is None:
            dist = getattr(attn_md, "request_distribution", None)
        if (
            isinstance(dist, torch.Tensor)
            and dist.device.type == "cpu"
            and dist.numel() >= 3
        ):
            dist_dt = (int(dist[0].item()), int(dist[2].item()))

        fb_nseq: int | None = None
        ns_attr = getattr(attn_md, "__dict__", {}).get("num_seqs")
        if ns_attr is not None:
            if isinstance(ns_attr, torch.Tensor):
                fb_nseq = int(ns_attr.item()) if ns_attr.device.type == "cpu" else None
            else:
                fb_nseq = int(ns_attr)

        qsl_md = getattr(attn_md, "__dict__", {}).get("query_start_loc_cpu")
        if qsl_md is None:
            qsl_dev: Any = getattr(attn_md, "query_start_loc", None)
            if isinstance(qsl_dev, torch.Tensor) and qsl_dev.device.type == "cpu":
                qsl_md = qsl_dev

        return _rt.inspect_batch_shape(
            num_reqs=num_reqs,
            has_roff=has_roff,
            decode_and_total_reqs=comb_wt,
            qsl=qsl_host,
            fallback_decode_and_total_reqs=dist_dt,
            fallback_num_seqs=fb_nseq,
            fallback_qsl=qsl_md,
        )

    def can_run(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        runner: Any = None,
    ) -> bool:
        """Return `True` if the current forward step should run on the megakernel."""
        self._prepared_step = None
        self.ab_calibrator.last_decision = ("stock_early", "stock")
        if (
            not self._bound
            or input_ids is None
            or inputs_embeds is not None
            or intermediate_tensors is not None
            or getattr(self.backbone, "aux_hidden_state_layers", ())
        ):
            return False
        if runner is not None:
            bctx = getattr(runner, "_attn_metadata_builder_ctx", None)
            dp_fn = getattr(runner, "_dp_lockstep_enabled", None)
            if (
                getattr(bctx, "position_ids_override", None) is not None
                or not getattr(runner, "_unified_kv_layout", True)
                or getattr(runner, "supports_mm_inputs", False)
                or getattr(runner, "lora_config", None) is not None
                or not getattr(runner, "_pp_is_first", True)
                or not getattr(runner, "_pp_is_last", True)
                or _pcp_or_dcp_enabled(runner)
                or (callable(dp_fn) and dp_fn())
            ):
                return False

        md = getattr(get_forward_context(), "attn_metadata", None)
        if not isinstance(md, dict) or not md:
            return False
        attn_md = md.get(self.attn_layer_name)
        if attn_md is None:
            return False
        m_mds = [md.get(n) for n in self.mamba_layer_names]
        if not m_mds or any(m is None for m in m_mds):
            return False
        m0 = m_mds[0]
        if (
            getattr(m0, "mamba_state_indices", None) is None
            and getattr(m0, "mamba_ckpt_indices", None) is None
        ):
            return False

        bs = self._inspect_step_detail(attn_md, m0, runner=runner)
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
                    mrope_np = getattr(
                        getattr(runner, "mrope_positions", None) if runner else None,
                        "np",
                        None,
                    )
                    if mrope_np is not None and mrope_np.shape[1] >= active_rows:
                        mrope_ok = _rt.check_mrope_uniform(mrope_np, active_rows)
                    elif positions.device.type == "cpu":
                        mrope_ok = _rt.check_mrope_uniform(
                            positions.detach().numpy(), active_rows
                        )
                    else:
                        mrope_ok = False
            elif positions.ndim > 2:
                mrope_ok = False

        decision = _rt.check_step_eligibility(
            bs,
            self.bucket_policy,
            self._bucket_steps,
            self._bucket_out_rows,
            input_rows=int(input_ids.shape[0]),
            mrope_ok=mrope_ok,
        )
        if not decision.eligible:
            return self._note_step(decision.reason, False)
        T, bkt = decision.tokens_per_seq, decision.bucket_t
        step_fn = self._bucket_steps[(T, bkt)]
        self._prepared_step = (step_fn, attn_md, *m_mds, decision.num_reqs, T, bkt)
        return self._note_step(decision.reason, True)

    def _note_step(self, key: str, run: bool) -> bool:
        actual_run, summary = self.step_mix_tracker.note_step(
            key, run, self.ab_calibrator
        )
        if summary is not None and self.tp_rank == 0:
            elapsed, mk_cnt, total, pct, snap = summary
            logger.info(
                "%s megakernel step mix over %.0fs: %d/%d MK (%.1f%%) %s",
                self._display_name,
                elapsed,
                mk_cnt,
                total,
                pct,
                snap,
            )
        return actual_run

    def record_ab_step(self, dt_ms: float, padded_tokens: int | None = None) -> None:
        res = self.ab_calibrator.record(dt_ms, padded_tokens=padded_tokens)
        if res is not None and self.tp_rank == 0:
            elapsed, win_json, cum_json = res
            logger.info(
                "%s megakernel AB latency (window %.0fs): %s",
                self._display_name,
                elapsed,
                win_json,
            )
            logger.info(
                "%s megakernel AB latency (cumulative): %s",
                self._display_name,
                cum_json,
            )

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int | None = None,
        runner: Any = None,
    ) -> torch.Tensor:
        """Execute one full-backbone decode (`T=1`) or MTP verify (`T=4`) step."""
        prepared, self._prepared_step = self._prepared_step, None
        if prepared is not None:
            step_fn, attn_md, *m_mds, num_reqs, tokens_per_seq, _ = prepared
        else:
            md = get_forward_context().attn_metadata
            attn_md = md[self.attn_layer_name]
            m_mds = [md[n] for n in self.mamba_layer_names]
            bs = self._inspect_step_detail(attn_md, m_mds[0], runner=runner)
            num_reqs, tokens_per_seq = bs.num_reqs, (bs.tokens_per_seq or 1)
            active_rows = max(tokens_per_seq, num_reqs * tokens_per_seq)
            bkt = self.bucket_policy.select(active_rows, tokens_per_seq, clamp=True)
            step_fn = self._bucket_steps[(tokens_per_seq, bkt)]

        out_rows = int(num_tokens if num_tokens is not None else input_ids.shape[0])
        fb_1d = getattr(attn_md, "seq_lens", None)
        if fb_1d is None:
            fb_1d = torch.zeros(
                (max(1, num_reqs),), dtype=torch.int32, device=input_ids.device
            )

        sc_args = [val for m_obj in m_mds for val in _extract_step_sc(m_obj, fb_1d)]
        m0 = m_mds[0]
        roff = getattr(m0, "mamba_slot_read_offsets", None)
        if roff is None and runner is not None:
            roff = getattr(runner, "mamba_slot_read_offsets", None)
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

    def try_forward(self, runner: Any, ctx: StepContext) -> tuple[bool, Any]:
        """Try to execute the current forward step on the megakernel.

        Returns `(True, out)` when handled by the megakernel (or timed by the
        A/B calibrator), where `out` is what the stock model would return; or
        `(False, None)` to fall through to the runner's stock model forward.
        Always `(False, None)` inside `megakernel_runtime.suppressed(runner)`
        (dummy / precompile / profile runs), so those runs compile the stock
        graphs, and when `ctx.model_kwargs` is non-empty (this driver doesn't
        consume extra model inputs).
        """
        if is_suppressed(runner) or ctx.model_kwargs:
            self._prepared_step = None
            return False, None
        input_ids, positions = ctx.input_ids, ctx.positions
        inputs_embeds = ctx.inputs_embeds
        intermediate_tensors = ctx.intermediate_tensors
        use_mk = self.can_run(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
            runner=runner,
        )
        ab_t0 = time.perf_counter() if self.ab_calibrator.enabled else None
        if use_mk:
            assert input_ids is not None and positions is not None
            num_tok = int(positions.shape[-1])
            out = self(input_ids, positions, num_tokens=num_tok, runner=runner)
            if ab_t0 is not None:
                if getattr(getattr(runner, "device", None), "type", None) == "tpu":
                    from vllm_torchtpu.utils import synchronize_tensors

                    synchronize_tensors([out])
                self.record_ab_step(
                    (time.perf_counter() - ab_t0) * 1000.0, padded_tokens=num_tok
                )
            return True, out

        if (
            ab_t0 is not None
            and self.ab_calibrator.last_decision[0] != "stock_early"
            and positions is not None
            and intermediate_tensors is None
            and callable(getattr(runner, "model", None))
            and not getattr(self.backbone, "aux_hidden_state_layers", ())
        ):
            out = runner.model(
                input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds
            )
            if isinstance(out, (list, tuple)) and len(out) == 1:
                out = out[0]
            if getattr(
                getattr(runner, "device", None), "type", None
            ) == "tpu" and isinstance(out, torch.Tensor):
                from vllm_torchtpu.utils import synchronize_tensors

                synchronize_tensors([out])
            self.record_ab_step(
                (time.perf_counter() - ab_t0) * 1000.0,
                padded_tokens=int(positions.shape[-1]),
            )
            return True, out

        return False, None


def maybe_create(runner: Any) -> MegakernelDriver | DeclinedDriver | None:
    """Create a `MegakernelDriver` for `runner` if enabled for its model."""
    model_def = MegakernelDriver._resolve_model_def(runner=runner)
    if model_def is None:
        return None
    return MegakernelDriver.maybe_create(runner, model_def=model_def)
