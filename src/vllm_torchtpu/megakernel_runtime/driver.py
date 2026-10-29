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
import functools
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
from vllm_torchtpu import _OFFLOADING_CONNECTORS
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel import MegakernelLayoutError
from vllm_torchtpu.megakernel import runtime as _rt
from vllm_torchtpu.megakernel_runtime.config import (
    collect_config,
    import_plugin,
    resolve_callable,
    resolve_plugin_ref,
)
from vllm_torchtpu.megakernel_runtime.fingerprint import (
    agreement_token,
    plugin_fingerprint,
    runtime_src_hash,
    source_hash,
)
from vllm_torchtpu.megakernel_runtime.hooks import StepContext, is_suppressed
from vllm_torchtpu.megakernel_runtime.plugin import (
    Bucket,
    Decline,
    ForwardView,
    RunnerShape,
    RunnerView,
    StepFn,
)
from vllm_torchtpu.megakernel_runtime.weight_collector import find_backbone

logger = init_logger(__name__)

_VLLM_STARTUP_MESSAGES: dict[str, str] = {
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
    "spec_decode_unsupported": (
        "Megakernel plugin does not declare speculative decoding "
        "(capability 'spec_decode'); skipping."
    ),
    "mrope_unsupported": (
        "Megakernel plugin does not declare M-RoPE (capability 'mrope'); skipping."
    ),
    "prefix_caching_unsupported": (
        "Megakernel plugin does not declare prefix caching "
        "(capability 'prefix_caching'); skipping."
    ),
    "kv_offload_unsupported": (
        "Megakernel plugin does not declare KV offloading "
        "(capability 'kv_offload'); skipping."
    ),
    "eplb_dynamic_unsupported": (
        "Megakernel plugin does not declare dynamic EPLB "
        "(capability 'eplb_dynamic'); a static expert placement is fine; skipping."
    ),
    "sleep_mode_unsupported": (
        "Megakernel plugin does not declare sleep mode "
        "(capability 'sleep_mode'); skipping."
    ),
}


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
    "plugin_mismatch",
    "create_error",
    "draft_declined",
    "mesh_size_mismatch",
    "other",
)


def vote_codes(extra: Sequence[str] = ()) -> tuple[str, ...]:
    """The vote table: the runtime's codes, a plugin's own codes, then `other`."""
    base = DECLINE_VOTE_CODES[:-1]
    return (*base, *(c for c in extra if c not in base and c != "other"), "other")


def _decline_vote(
    result: _rt.EligibilityResult | None, codes: Sequence[str] = DECLINE_VOTE_CODES
) -> int:
    if result is None:
        return 0
    code = result.code if result.code in codes else "other"
    return list(codes).index(code) + 1


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


def buckets_token(keys: Sequence[tuple[int, int]], salt: str = "") -> int:
    """A rank-independent 32-bit digest of a set of `(T, B)` bucket keys
    (salted, e.g. with the plugin fingerprint)."""
    return zlib.crc32((salt + repr(sorted(keys))).encode("ascii"))


def agree_on_decline(
    local: _rt.EligibilityResult | None,
    *,
    token: int | None = None,
    group: Any = None,
    mismatch_code: str | None = None,
    codes: Sequence[str] = DECLINE_VOTE_CODES,
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

    With `mismatch_code` (startup agreement #1: the token is the plugin
    fingerprint), differing tokens win over votes, since the ranks may not
    even share a vote table, and every rank declines with that code.
    """
    if group is None:
        group = _tp_cpu_group()
    if group is None:
        return local
    tok = int(token or 0)
    t = torch.tensor([_decline_vote(local, codes), tok, -tok], dtype=torch.int64)
    _all_reduce_max(t, group)
    vote, tok_max, tok_min = int(t[0]), int(t[1]), -int(t[2])
    if mismatch_code is not None and tok_max != tok_min:
        msg = (
            "Megakernel plugin or runtime source differs across TP ranks "
            f"({mismatch_code}); skipping."
        )
        return _rt.EligibilityResult(False, mismatch_code, msg)
    if vote != 0:
        if local is not None:
            return local
        code = codes[vote - 1] if vote <= len(codes) else "other"
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

    serving = False

    def __init__(
        self,
        result: _rt.EligibilityResult,
        *,
        tp_rank: int = 0,
        display_name: str = "megakernel",
        stats_period_s: float = 60.0,
        codes: Sequence[str] = DECLINE_VOTE_CODES,
    ) -> None:
        self.decline = result
        self.tp_rank, self._display_name = int(tp_rank), display_name
        self._codes = tuple(codes)
        self.step_mix_tracker = _rt.StepMixTracker(period_s=float(stats_period_s))
        self._step_key = f"stock_{result.code}"

    def bind_and_warmup(self, runner: Any) -> None:
        del runner
        # Join the bind-time agreement the serving drivers on other ranks run.
        agree_on_decline(self.decline, codes=self._codes)
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


def _inspect_step_detail(attn_md: Any, m0: Any, runner: Any = None) -> _rt.BatchShape:
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
        elif has_roff and isinstance(comb_cpu, torch.Tensor) and comb_cpu.numel() >= 6:
            comb_wt = (int(comb_cpu[3].item()), int(comb_cpu[5].item()))
        qsl_host = getattr(runner, "query_start_loc_np", None)
        if qsl_host is None:
            qsl_host = getattr(runner, "query_start_loc_cpu", None)
        runner_decides = qsl_host is not None and (
            num_reqs is not None or (has_roff and comb_wt is not None)
        )
        if not runner_decides and has_roff and comb_wt is not None:
            eff_n = comb_wt[1] if num_reqs is None else num_reqs
            runner_decides = comb_wt[1] <= 0 or eff_n <= 0 or comb_wt[0] != comb_wt[1]
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


# --- Generic startup checks against a plugin's declared capabilities --------

# The offloading connectors (`OffloadingConnector`, `TPURaidenOffloadingConnector`).
_OFFLOAD_CONNECTORS = _OFFLOADING_CONNECTORS


def _runner_cfg(runner: Any, name: str) -> Any:
    v_cfg = getattr(runner, "vllm_config", None)
    cfg = getattr(v_cfg, name, None) if v_cfg is not None else None
    return cfg if cfg is not None else getattr(runner, name, None)


def kv_connector_names(kv_transfer_config: Any) -> tuple[str, ...]:
    """The KV connector names in effect (a `MultiConnector`'s members)."""
    if kv_transfer_config is None:
        return ()
    name = str(getattr(kv_transfer_config, "kv_connector", None) or "")
    if name != "MultiConnector":
        return (name,)
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    members = extra.get("connectors", ()) if isinstance(extra, dict) else ()
    names = tuple(
        str(
            (m.get("kv_connector") if isinstance(m, dict) else None)
            or getattr(m, "kv_connector", None)
            or ""
        )
        for m in members
    )
    return names or (name,)


def kv_connector_is_offloading(kv_transfer_config: Any) -> bool:
    """Offloading connectors only (a `MultiConnector` only if every member is)."""
    names = kv_connector_names(kv_transfer_config)
    return bool(names) and all(n in _OFFLOAD_CONNECTORS for n in names)


def _aux_hidden_layers(backbone: Any) -> Any:
    return getattr(backbone, "aux_hidden_state_layers", ()) if backbone else ()


def _runner_backbone(runner: Any) -> Any:
    try:
        return find_backbone(getattr(runner, "model", None))
    except ValueError:
        return None


def _decline(code: str, detail: str) -> _rt.EligibilityResult:
    return _rt.EligibilityResult(False, code, f"Megakernel {detail}; skipping.")


def check_runner_capabilities(
    runner: Any, capabilities: frozenset[str], *, tp_size: int
) -> _rt.EligibilityResult | None:
    """The first framework feature that is on but not declared, as a named
    decline (`None` if all are covered). Reads config and group sizes only,
    identical on every rank: no collective."""
    caps = frozenset(capabilities)
    pp_multi = not (
        getattr(runner, "_pp_is_first", True) and getattr(runner, "_pp_is_last", True)
    )
    p_cfg = getattr(runner, "parallel_config", None)
    pcp = (
        bool(getattr(runner, "_pcp_enabled", False))
        or int(getattr(p_cfg, "prefill_context_parallel_size", 1) or 1) > 1
    )
    dcp = (
        bool(getattr(runner, "_dcp_enabled", False))
        or int(getattr(p_cfg, "decode_context_parallel_size", 1) or 1) > 1
    )
    dp_fn = getattr(runner, "_dp_lockstep_enabled", None)
    rules: list[tuple[bool, str, str]] = [
        (
            bool(getattr(runner, "supports_mm_inputs", False))
            and "multimodal" not in caps,
            "multimodal_inputs",
            "requires text-only input_ids (multimodal inputs)",
        ),
        (
            getattr(runner, "lora_config", None) is not None and "lora" not in caps,
            "lora_enabled",
            "does not support LoRA adapters",
        ),
        (
            pp_multi and "pipeline_parallel" not in caps,
            "pipeline_parallel",
            "requires a single-stage pipeline (PP=1)",
        ),
        (
            (pcp and "pcp" not in caps) or (dcp and "dcp" not in caps),
            "context_parallel",
            "does not support prefill/decode context parallelism (PCP/DCP)",
        ),
        (
            bool(callable(dp_fn) and dp_fn()),
            "dp_lockstep",
            "does not support EP-DP lockstep",
        ),
        (
            bool(_aux_hidden_layers(_runner_backbone(runner)))
            and "aux_hidden_states" not in caps,
            "aux_hidden_states",
            "does not support aux_hidden_state_layers",
        ),
    ]
    for bad, code, detail in rules:
        if bad:
            return _decline(code, detail)
    if _data_parallel_enabled(runner, max(1, int(tp_size))):
        msg = (
            "Megakernel agrees across the TP group only: data_parallel_size "
            "> 1 or an EP group wider than TP is unsupported; skipping."
        )
        return _rt.EligibilityResult(False, "dp_unsupported", msg)
    spec_cfg = getattr(runner, "speculative_config", None)
    c_cfg = _runner_cfg(runner, "cache_config")
    kv_cfg = _runner_cfg(runner, "kv_transfer_config")
    par_cfg = _runner_cfg(runner, "parallel_config")
    m_cfg = _runner_cfg(runner, "model_config")
    offload = kv_cfg is not None and kv_connector_is_offloading(kv_cfg)
    other_kv = next(
        (n for n in kv_connector_names(kv_cfg) if n not in _OFFLOAD_CONNECTORS),
        None,
    )
    rules = [
        (
            int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0) > 0
            and "spec_decode" not in caps,
            "spec_decode_unsupported",
            "plugin does not declare speculative decoding",
        ),
        (
            bool(getattr(runner, "uses_mrope", False)) and "mrope" not in caps,
            "mrope_unsupported",
            "plugin does not declare M-RoPE",
        ),
        (
            bool(getattr(c_cfg, "enable_prefix_caching", False))
            and "prefix_caching" not in caps,
            "prefix_caching_unsupported",
            "plugin does not declare prefix caching",
        ),
        (
            offload and "kv_offload" not in caps,
            "kv_offload_unsupported",
            "plugin does not declare KV offloading",
        ),
        (
            kv_cfg is not None and not offload and "kv_transfer" not in caps,
            f"kv_connector_unsupported:{other_kv}",
            f"plugin does not declare KV transfer (connector {other_kv})",
        ),
        (
            bool(getattr(par_cfg, "enable_eplb", False)) and "eplb_dynamic" not in caps,
            "eplb_dynamic_unsupported",
            "plugin does not declare dynamic EPLB",
        ),
        # The attribute name on the pinned vLLM is checked on the serving image.
        (
            bool(getattr(m_cfg, "enable_sleep_mode", False))
            and "sleep_mode" not in caps,
            "sleep_mode_unsupported",
            "plugin does not declare sleep mode",
        ),
    ]
    for bad, code, detail in rules:
        if bad:
            return _decline(code, detail)
    return None


def _local_tp_rank() -> int:
    """This rank's TP rank (config-only, no collective)."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
    except ImportError:
        dist = torch.distributed
        ok = dist.is_available() and dist.is_initialized()
        return int(dist.get_rank()) if ok else 0
    return int(get_tensor_model_parallel_rank())


def _default_out_rows(b: int, token_paddings: Sequence[int]) -> int:
    return next(
        (int(p) for p in token_paddings if int(p) >= b), max(16, -(-b // 16) * 16)
    )


def runner_shapes(
    bucket: Bucket,
    token_paddings: Sequence[int],
    req_block_pairs: Sequence[tuple[int, int]],
    *,
    ckpt_window: int,
    max_pages_per_seq: int,
) -> list[RunnerShape]:
    """The runner input shapes a bucket is warmed up with (one graph each)."""
    T, B = int(bucket.T), int(bucket.B)
    cols = max(T, max(1, int(ckpt_window)))
    if token_paddings and req_block_pairs:
        pads = [int(p) for p in token_paddings if int(p) >= B]
        tok = pads[0] if pads else int(token_paddings[-1])
        return [RunnerShape(tok, int(r), int(k), cols) for r, k in req_block_pairs]
    return [RunnerShape(B, B // T, int(max_pages_per_seq), cols)]


def _log_step_mix(name: str, summary: Any) -> None:
    elapsed, mk_cnt, total, pct, snap = summary
    logger.info(
        "%s megakernel step mix over %.0fs: %d/%d MK (%.1f%%) %s",
        name,
        elapsed,
        mk_cnt,
        total,
        pct,
        snap,
    )


class ProgramDriver:
    """Serves one plugin's `MegakernelProgram` on this TP rank.

    Generic: startup checks against the plugin's capabilities, the two TP
    agreements (plugin fingerprint before any collective work, compiled
    bucket set before warmup), one `torch.compile`d step per bucket, warmup
    over the runner's input shapes, per-step routing, step-mix stats and the
    A/B latency calibrator. Everything model-specific is the program's.
    """

    serving = True

    def __init__(
        self,
        program: Any,
        *,
        view: RunnerView,
        tp_rank: int = 0,
        stats_period_s: float = 60.0,
        ab: bool = False,
        fingerprint: str = "",
        codes: Sequence[str] = DECLINE_VOTE_CODES,
        backbone: Any = None,
    ) -> None:
        self.program, self.view = program, view
        self.tp_rank = int(tp_rank)
        self._display_name = str(getattr(program, "display_name", "") or "megakernel")
        self.fingerprint = fingerprint
        self._codes = tuple(codes)
        self._backbone = backbone
        self._buckets: list[Bucket] = []
        self._steps: dict[tuple[int, int], StepFn] = {}
        self._bound = False
        self._pending: tuple[Any, ForwardView] | None = None
        self.pool_layout_facts: Any = None
        self.step_mix_tracker = _rt.StepMixTracker(period_s=float(stats_period_s))
        self.ab_calibrator = _rt.MKABCalibrator(
            enabled=bool(ab), period_s=float(stats_period_s)
        )

    # --- startup (steps 1-8) ------------------------------------------------

    @classmethod
    def maybe_create(cls, runner: Any) -> ProgramDriver | DeclinedDriver | None:
        """Create the configured plugin's driver for `runner`.

        `None` when the megakernel is off, no plugin serves the checkpoint's
        arch, or a generic startup check fails (identical on every rank, so
        no collective); a `DeclinedDriver` when `create`, the mesh or the
        plugin fingerprint agreement declines on any rank.
        """
        add_cfg = getattr(
            getattr(runner, "vllm_config", None), "additional_config", None
        )
        cfg = collect_config(envs.VLLM_TPU_MEGAKERNEL, add_cfg)
        if not cfg.enabled:
            return None
        view = RunnerView(runner)
        resolved = resolve_plugin_ref(cfg, view.hf_config, view.model)
        if resolved is None:
            return None
        ref, options = resolved
        plugin, module = import_plugin(ref)
        src = source_hash(module, source_root=plugin.source_root)
        fprint = plugin_fingerprint(
            module=ref.module,
            attr=ref.attr,
            plugin=plugin,
            source_hash=src,
            config={**cfg.runtime_keys(), "options": options},
            raw_env=cfg.raw_env,
        )
        token = agreement_token(fprint, runtime_src_hash())
        logger.info(
            "Megakernel plugin %s %s loaded from %s (fingerprint %s)",
            plugin.name,
            plugin.version,
            ref.module,
            fprint[:16],
        )
        tp_size = view.tp_size
        el = check_runner_capabilities(runner, plugin.capabilities, tp_size=tp_size)
        if el is not None:
            cls._log_startup_ineligible(el)
            return None
        create = resolve_callable(plugin.create, module, what="create")
        tp_rank = _local_tp_rank()
        view = RunnerView(runner, tp_rank=tp_rank)
        codes = vote_codes(plugin.decline_codes)
        agree = functools.partial(
            agree_on_decline, token=token, mismatch_code="plugin_mismatch", codes=codes
        )
        decline: _rt.EligibilityResult | None = None
        try:
            out = create(view, dict(options))
        except BaseException:
            # Still vote, so the other ranks decline instead of waiting.
            agree(_rt.EligibilityResult(False, "create_error", "create failed"))
            raise
        if isinstance(out, Decline):
            decline = out.result
        mesh = None
        if decline is None:
            try:
                mesh = _build_tp_rank_mesh(
                    tp_size, fallback_mesh=getattr(runner, "mesh", None)
                )
            except MeshDecline as exc:
                decline = exc.result
            else:
                el_mesh = _rt.check_post_mesh_eligibility(
                    tp_size=tp_size, mesh_size=int(mesh.devices.size)
                )
                decline = None if el_mesh.eligible else el_mesh
        # Agreement #1: the first collective; it also compares fingerprints.
        decline = agree(decline)
        name = str(getattr(out, "display_name", "") or plugin.name)
        if decline is not None:
            cls._log_startup_ineligible(decline)
            return DeclinedDriver(
                decline,
                tp_rank=tp_rank,
                display_name=name,
                stats_period_s=float(cfg.stats_period_s),
                codes=codes,
            )
        out.load(mesh)
        return cls(
            out,
            view=view,
            tp_rank=tp_rank,
            stats_period_s=float(cfg.stats_period_s),
            ab=bool(cfg.ab),
            fingerprint=fprint,
            codes=codes,
            backbone=_runner_backbone(runner),
        )

    @staticmethod
    def _log_startup_ineligible(el: _rt.EligibilityResult) -> None:
        msg = _VLLM_STARTUP_MESSAGES.get(el.code, el.message)
        if msg:
            (logger.info if el.level == "info" else logger.warning)("%s", msg)

    # --- bind, agreement #2, warmup (steps 10-13) ---------------------------

    def _compile(self, step: StepFn, bucket: Bucket) -> StepFn:
        if not isinstance(step, types.FunctionType):
            inner = step

            def step(*args: Any) -> Any:
                return inner(*args)

        # Give each bucket its own code object: nested closures share `__code__`,
        # and Dynamo caches compiled graphs per code object, so all buckets
        # would otherwise compete for one entry's `cache_size_limit`.
        name = f"step_t{int(bucket.T)}_b{int(bucket.B)}"
        co = step.__code__.replace(co_name=name, co_qualname=name)
        fn = types.FunctionType(
            co, step.__globals__, name, step.__defaults__, step.__closure__
        )
        if getattr(self.view.device, "type", None) != "tpu":
            return fn
        opts: dict[str, Any] = {"name": name}
        check = getattr(self.program, "check_compiled_ir", None)
        if callable(check):

            def _cb(dbg: Any) -> None:
                check(dbg, bucket)

            opts["debug_callback"] = _cb
        opts["serializable"] = False
        return torch.compile(
            fn, backend="tpu", fullgraph=True, dynamic=False, options=opts
        )

    def bind(self, runner: Any) -> None:
        """Bind the program and build one compiled step per bucket."""
        if runner is not None:
            self.view = RunnerView(runner, tp_rank=self.tp_rank)
        self.pool_layout_facts = self.program.bind(self.view)
        pads = self.view.token_paddings
        buckets = [
            Bucket(b.T, b.B, _default_out_rows(b.B, pads)) if b.out_rows is None else b
            for b in self.program.buckets(pads)
        ]
        self._steps.clear()
        for b in buckets:
            self._steps[b.key] = self._compile(self.program.build(b), b)
        self._buckets = buckets
        self._bound = True

    def bind_and_warmup(self, runner: Any) -> None:
        """Bind the program to the runner's caches, agree, and warm up.

        Only `MegakernelLayoutError` (an incompatible runtime cache layout)
        disables the megakernel: the runner's `_megakernel` becomes a
        `DeclinedDriver` that runs the stock path and reports the reason in the
        step-mix stats. Any other exception votes `bind_error` and propagates.
        Every TP rank agrees on the compiled bucket set before warmup (which
        runs the megakernel's collectives): if any rank declines, all do.
        """
        precompile_timed = getattr(runner, "_precompile_timed", None)
        ctx = (
            precompile_timed("megakernel")
            if callable(precompile_timed)
            else contextlib.nullcontext()
        )
        with ctx:
            decline: _rt.EligibilityResult | None = None
            try:
                self.bind(runner)
            except MegakernelLayoutError as exc:
                decline = getattr(exc, "result", None) or _rt.EligibilityResult(
                    False, "cache_layout_incompatible", str(exc)
                )
            except BaseException:
                # This rank is raising: still vote, so the others decline
                # instead of waiting in the agreement.
                agree_on_decline(
                    _rt.EligibilityResult(False, "bind_error", "bind failed"),
                    codes=self._codes,
                )
                raise
            token = None
            if decline is None:
                token = buckets_token(list(self._steps), salt=self.fingerprint)
            decline = agree_on_decline(decline, token=token, codes=self._codes)
            if decline is not None:
                detach = getattr(self.program, "detach_draft", None)
                if callable(detach):
                    detach()
                logger.warning(
                    "Disabling %s megakernel due to incompatible KV cache layout: %s",
                    self._display_name,
                    decline.message,
                )
                self._bound = False
                if runner is not None and hasattr(runner, "_megakernel"):
                    runner._megakernel = DeclinedDriver(
                        decline,
                        tp_rank=self.tp_rank,
                        display_name=self._display_name,
                        stats_period_s=self.step_mix_tracker.period_s,
                        codes=self._codes,
                    )
                return
            self.warmup(runner)
            done = getattr(self.program, "on_warmup_done", None)
            if callable(done):
                done()

    @torch.no_grad()
    def warmup(self, runner: Any = None) -> None:
        """Run every compiled bucket over the runner's input shapes."""
        if not self._bound:
            raise RuntimeError("the program must be bound before warmup()")
        view = self.view if runner is None else RunnerView(runner, tp_rank=self.tp_rank)
        pads, pairs = view.token_paddings, view.req_block_pairs
        ckpt_w, max_pages = view.ckpt_window, view.max_pages_per_seq
        is_tpu = getattr(view.device, "type", None) == "tpu"
        session = getattr(self.program, "warmup_session", None)
        with session() if callable(session) else contextlib.nullcontext():
            for b in self._buckets:
                for shape in runner_shapes(
                    b, pads, pairs, ckpt_window=ckpt_w, max_pages_per_seq=max_pages
                ):
                    case = self.program.warmup_inputs(b, shape)
                    if case is None:
                        continue
                    out = self._steps[b.key](*case.args)
                    if is_tpu:
                        from vllm_torchtpu.utils import synchronize_tensors

                        synchronize_tensors([out])
                    if case.check is not None:
                        case.check(out)
        self.reset_stats()

    def reset_stats(self) -> None:
        """Reset step-mix counters and A/B latency accumulators."""
        self.step_mix_tracker.reset()
        self.ab_calibrator.reset()

    # --- per step -------------------------------------------------------------

    @classmethod
    def _inspect_step_detail(
        cls, attn_md: Any, m0: Any, runner: Any = None
    ) -> _rt.BatchShape:
        """Return `BatchShape` without host-device synchronization."""
        return _inspect_step_detail(attn_md, m0, runner)

    def _forward_view(self, runner: Any, num_tokens: int | None = None) -> ForwardView:
        md = getattr(get_forward_context(), "attn_metadata", None)
        mrope = getattr(getattr(runner, "mrope_positions", None), "np", None)
        return ForwardView(
            attn_metadata=md if isinstance(md, dict) else None,
            mrope_positions=mrope if runner is not None else None,
            steps=self._steps,
            num_tokens=num_tokens,
            inspect=functools.partial(_inspect_step_detail, runner=runner),
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
        self._pending = None
        self.ab_calibrator.last_decision = ("stock_early", "stock")
        if (
            not self._bound
            or input_ids is None
            or inputs_embeds is not None
            or intermediate_tensors is not None
            or _aux_hidden_layers(self._backbone)
        ):
            return False
        bctx = getattr(runner, "_attn_metadata_builder_ctx", None)
        if getattr(bctx, "position_ids_override", None) is not None:
            return False
        ctx = StepContext(input_ids, positions, inputs_embeds, intermediate_tensors)
        fwd = self._forward_view(runner)
        decision = self.program.decide(ctx, fwd)
        if decision is None:
            return False
        if not decision.eligible:
            return self._note_step(decision.reason, False)
        self._pending = (decision, fwd)
        return self._note_step(decision.reason, True)

    def _note_step(self, key: str, run: bool) -> bool:
        actual_run, summary = self.step_mix_tracker.note_step(
            key, run, self.ab_calibrator
        )
        if summary is not None and self.tp_rank == 0:
            _log_step_mix(self._display_name, summary)
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
    ) -> Any:
        """Run the step `can_run` decided (or decide now) on the program."""
        pending, self._pending = self._pending, None
        n_tok = int(num_tokens if num_tokens is not None else input_ids.shape[0])
        if pending is None:
            decision, fwd = None, self._forward_view(runner, n_tok)
        else:
            decision, fwd = (
                pending[0],
                dataclasses.replace(pending[1], num_tokens=n_tok),
            )
        return self.program.forward(decision, StepContext(input_ids, positions), fwd)

    def try_forward(self, runner: Any, ctx: StepContext) -> tuple[bool, Any]:
        """Try to execute the current forward step on the megakernel.

        Returns `(True, out)` when handled by the megakernel (or timed by the
        A/B calibrator), where `out` is what the stock model would return; or
        `(False, None)` to fall through to the runner's stock model forward.
        Always `(False, None)` inside `megakernel_runtime.suppressed(runner)`
        (dummy / precompile / profile runs), so those runs compile the stock
        graphs, and when `ctx.model_kwargs` is non-empty (the program doesn't
        consume extra model inputs).
        """
        if is_suppressed(runner) or ctx.model_kwargs:
            self._pending = None
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
            and not _aux_hidden_layers(self._backbone)
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


def maybe_create(runner: Any) -> ProgramDriver | DeclinedDriver | None:
    """Create the configured megakernel plugin's driver for `runner`."""
    return ProgramDriver.maybe_create(runner)
