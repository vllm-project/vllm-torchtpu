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
"""Thin PyTorch/vLLM driver over the `megakernel.core` library."""

from __future__ import annotations

import contextlib
import types
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.adapter.agreement import (
    DECLINE_VOTE_CODES,
    DeclinedDriver,
    agree_on_decline,
    agreement_group,
    buckets_token,
    vote_codes,
)
from vllm_torchtpu.megakernel.adapter.config import (
    import_plugin,
    missing_template,
    resolve_callable,
    select_plugin,
)
from vllm_torchtpu.megakernel.adapter.errors import MegakernelPluginError
from vllm_torchtpu.megakernel.adapter.fingerprint import (
    agreement_token,
    plugin_fingerprint,
    runtime_src_hash,
    source_hash,
)
from vllm_torchtpu.megakernel.adapter.guards import (
    _runner_backbone,
    check_runner_features,
    check_spec,
    detect_parallelism,
    kv_connector_names,
)
from vllm_torchtpu.megakernel.adapter.hooks import StepContext
from vllm_torchtpu.megakernel.adapter.plugin import ForwardView, RunnerView
from vllm_torchtpu.megakernel.adapter.stepping import (
    StepRouting,
    _default_out_rows,
    runner_shapes,
)
from vllm_torchtpu.megakernel.core.errors import MegakernelLayoutError
from vllm_torchtpu.megakernel.core.runtime import stats as _stats
from vllm_torchtpu.megakernel.core.runtime.contract import (
    DOMAIN_TP,
    Bucket,
    Decline,
    EligibilityResult,
    LayoutDecline,
    StepFn,
)

logger = init_logger(__name__)

_VLLM_STARTUP_MESSAGES: dict[str, str] = {
    "multimodal_inputs": (
        "Megakernel serves text-only requests, and this server accepts "
        "multimodal inputs; pass --language-model-only for text-only "
        "serving; skipping."
    ),
    "lora_enabled": "Megakernel does not support LoRA adapters; skipping.",
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
    """Return `torch_tpu`'s `get_pallas_mesh` when the megakernel runs on TPU
    (JAX's default backend is `tpu`) and `torch_tpu` is installed; `None`
    otherwise. A CPU process (`JAX_PLATFORMS=cpu`, e.g. CI's unit lane) may
    have `torch_tpu` installed, but its steps run on JAX's CPU devices."""
    if jax.default_backend() != "tpu":
        return None
    try:
        from torch_tpu._internal.pallas import get_pallas_mesh
    except ImportError:
        return None
    return get_pallas_mesh


def check_mesh_size(tp_size: int, mesh_size: int) -> EligibilityResult | None:
    """`mesh_size_mismatch` when the TP-rank mesh doesn't hold `tp_size`
    devices (`None` if it does). The chip and the TP size are `check_spec`'s."""
    if mesh_size != tp_size:
        msg = (
            f"Megakernel TP-rank mesh has {mesh_size} devices, "
            f"expected {tp_size}; skipping."
        )
        return EligibilityResult(False, "mesh_size_mismatch", msg)
    return None


class MeshDecline(Exception):
    """`_build_tp_rank_mesh` can't build a correctly ordered TP mesh."""

    def __init__(self, result: EligibilityResult) -> None:
        super().__init__(result.message)
        self.result = result


def _build_tp_rank_mesh(r_cnt: int, fallback_mesh: Any = None) -> Any:
    """Build an `r_cnt`-device JAX `Mesh` with axis `'tp'` in TP-rank order.

    On TPU with `torch_tpu` this is always its Pallas mesh: `torch_tpu` binds
    partition `p` to TP rank `p` and requires every Pallas kernel in a program
    to share one device assignment, while the runner mesh may list the devices
    in another order (e.g. ICI ring order), which would send the megakernel's
    remote DMAs, addressed by mesh position, to the wrong peers. Raises
    `MeshDecline` (`pallas_mesh_unavailable` / `pallas_mesh_mismatch`) when
    `torch_tpu` is present but its mesh is unavailable or doesn't hold `r_cnt`
    devices on axis `'tp'`. Off TPU, or without `torch_tpu`, the runner mesh
    is used, then `jax.devices()`.
    """
    get_pallas_mesh = _get_pallas_mesh_fn()
    if get_pallas_mesh is not None:
        try:
            pallas_mesh = get_pallas_mesh(axis_names=("tp",))
        except Exception as exc:
            msg = f"Megakernel: get_pallas_mesh failed ({exc}); skipping."
            raise MeshDecline(
                EligibilityResult(False, "pallas_mesh_unavailable", msg)
            ) from exc
        n_dev = pallas_mesh.devices.size
        axes = tuple(pallas_mesh.axis_names)
        if n_dev != r_cnt or axes != ("tp",):
            msg = (
                f"Megakernel: the Pallas mesh has {n_dev} devices on axes {axes}, "
                f"expected {r_cnt} on ('tp',); skipping."
            )
            raise MeshDecline(EligibilityResult(False, "pallas_mesh_mismatch", msg))
        return pallas_mesh
    fb_devs = getattr(fallback_mesh, "devices", None)
    if fb_devs is not None and fb_devs.size == r_cnt:
        if tuple(fallback_mesh.axis_names) == ("tp",):
            return fallback_mesh
        return jax.sharding.Mesh(np.asarray(fb_devs).reshape(-1), ("tp",))
    return jax.sharding.Mesh(np.asarray(jax.devices()[:r_cnt]), ("tp",))


def _local_tp_rank() -> int:
    """This rank's TP rank (config-only, no collective): vLLM's TP-group rank
    once vLLM's model parallel groups exist, else the `torch.distributed` rank,
    else 0 (a single process, e.g. a CPU test with vLLM installed but no TP
    group)."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            model_parallel_is_initialized,
        )
    except ImportError:
        pass
    else:
        if model_parallel_is_initialized():
            return get_tensor_model_parallel_rank()
    dist = torch.distributed
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


_PROGRAM_METHODS = (
    "buckets",
    "load",
    "bind",
    "build",
    "warmup_inputs",
    "decide",
    "forward",
)


def loaded_numerics_note(spec_numerics: str, options: Mapping[str, Any]) -> str:
    """The numerics part of the "plugin loaded" line. A spec's `numerics` is
    its default profile's; a requested profile's own numerics are only known
    to the program, which logs them at bind."""
    note = f"spec numerics={spec_numerics} (default profile)"
    if prof := options.get("profile"):
        note += f"; requested profile={prof}, its numerics are logged at bind"
    return note


def _check_create_result(out: Any, ref: Any) -> EligibilityResult | None:
    """`create`'s decline, or `None` for a program; anything else raises."""
    if isinstance(out, Decline):
        return out.result
    missing = [m for m in _PROGRAM_METHODS if not callable(getattr(out, m, None))]
    if missing:
        raise MegakernelPluginError(
            f"megakernel plugin {ref.module}:{ref.attr} create() returned "
            f"{type(out).__name__}, which is neither a Decline nor a "
            f"MonoProgram (missing {missing})"
        )
    return None


def _vote_startup_failure(
    token: int, codes: Sequence[str], domain: str, runner: Any
) -> None:
    """Join the startup agreement as `create_error` from a rank that is
    about to raise."""
    try:
        agree_on_decline(
            EligibilityResult(False, "create_error", "startup failed"),
            group=agreement_group(domain, runner),
            token=token,
            mismatch_code="plugin_mismatch",
            codes=codes,
        )
    except Exception:  # the original error is the one to report
        logger.exception("Megakernel: voting create_error failed")


class ProgramDriver(StepRouting):
    """Serves one model folder's `MonoProgram` on this TP rank.

    Generic: startup checks against the spec's declared fields, the two TP
    agreements (plugin fingerprint before any collective work, compiled
    bucket set before warmup), one `torch.compile`d step per bucket and warmup
    over the runner's input shapes; per-step routing, step-mix stats and the
    A/B latency calibrator come from `StepRouting`. Everything model-specific
    is the program's.
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
        domain: str = DOMAIN_TP,
    ) -> None:
        self.program, self.view = program, view
        self._domain = domain
        self.tp_rank = tp_rank
        self._display_name = str(getattr(program, "display_name", "") or "megakernel")
        self.fingerprint = fingerprint
        self._codes = tuple(codes)
        self._backbone = backbone
        self._buckets: list[Bucket] = []
        self._steps: dict[tuple[int, int], StepFn] = {}
        self._bound = False
        self._warmup_done = False
        self._pending: tuple[Any, ForwardView, StepContext] | None = None
        self._inspect: Any = None
        self._inspect_runner: Any = None
        self.pool_layout_facts: Any = None
        self.step_mix_tracker = _stats.StepMixTracker(period_s=stats_period_s)
        self.ab_calibrator = _stats.MKABCalibrator(enabled=ab, period_s=stats_period_s)

    # --- startup (steps 1-8) ------------------------------------------------

    @classmethod
    def maybe_create(cls, runner: Any) -> ProgramDriver | DeclinedDriver | None:
        """Create the configured plugin's driver for `runner`.

        `None` when the megakernel is off, no plugin serves the checkpoint's
        arch (or its shorthand template isn't installed), or a generic startup
        check fails (identical on every rank, so no collective); a
        `DeclinedDriver` when `create`, the mesh or the plugin fingerprint
        agreement declines on any rank. Any rank-local
        failure from the plugin import to the mesh votes `create_error` in
        the startup agreement before it propagates, so the other ranks decline
        instead of waiting for it.
        """
        add_cfg = getattr(
            getattr(runner, "vllm_config", None), "additional_config", None
        )
        view = RunnerView(runner)
        cfg, resolved = select_plugin(envs.VLLM_TPU_MEGAKERNEL, add_cfg, view.hf_config)
        if not cfg.enabled or resolved is None:
            return None
        ref, options = resolved
        if (msg := missing_template(ref, view.hf_config)) is not None:
            logger.warning("%s", msg)
            return None
        # Until the fingerprint exists, a failing rank votes with token 0: it
        # can't match the peers' token, so they decline with plugin_mismatch.
        token, codes, domain = 0, vote_codes(), DOMAIN_TP
        mesh: Any = None
        try:
            plugin, module = import_plugin(ref)
            codes = vote_codes(plugin.decline_codes)
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
                "Megakernel plugin %s %s loaded from %s (%s, fingerprint %s)",
                plugin.name,
                plugin.version,
                ref.module,
                loaded_numerics_note(plugin.numerics, options),
                fprint[:16],
            )
            tp_size = view.tp_size
            el = check_spec(runner, plugin, view=view)
            if el is not None:
                cls._log_startup_ineligible(el)
                return None
            domain = plugin.collective_domain
            group = agreement_group(domain, runner)
            create = resolve_callable(plugin.create, module, what="create")
            tp_rank = _local_tp_rank()
            view = RunnerView(runner, tp_rank=tp_rank)
            out = create(view, dict(options))
            decline = _check_create_result(out, ref)
            if decline is None:
                try:
                    mesh = _build_tp_rank_mesh(
                        tp_size, fallback_mesh=getattr(runner, "mesh", None)
                    )
                except MeshDecline as exc:
                    decline = exc.result
                else:
                    decline = check_mesh_size(tp_size, mesh.devices.size)
        except BaseException:
            _vote_startup_failure(token, codes, domain, runner)
            raise
        # The startup agreement: the first collective; it also compares
        # fingerprints.
        decline = agree_on_decline(
            decline,
            group=group,
            token=token,
            mismatch_code="plugin_mismatch",
            codes=codes,
        )
        name = str(getattr(out, "display_name", "") or plugin.name)
        if decline is not None:
            cls._log_startup_ineligible(decline)
            return DeclinedDriver(
                decline,
                tp_rank=tp_rank,
                display_name=name,
                stats_period_s=cfg.stats_period_s,
                codes=codes,
                domain=domain,
            )
        out.load(mesh)
        return cls(
            out,
            view=view,
            tp_rank=tp_rank,
            stats_period_s=cfg.stats_period_s,
            ab=cfg.ab,
            fingerprint=fprint,
            codes=codes,
            backbone=_runner_backbone(runner),
            domain=domain,
        )

    @staticmethod
    def _log_startup_ineligible(el: EligibilityResult) -> None:
        msg = _VLLM_STARTUP_MESSAGES.get(el.code, el.message)
        if msg:
            (logger.info if el.level == "info" else logger.warning)("%s", msg)

    # --- bind, bind agreement, warmup -------------------------------------

    def _compile(self, step: StepFn, bucket: Bucket) -> StepFn:
        if not isinstance(step, types.FunctionType):
            inner = step

            def step(*args: Any) -> Any:
                return inner(*args)

        # Give each bucket its own code object: nested closures share `__code__`,
        # and Dynamo caches compiled graphs per code object, so all buckets
        # would otherwise compete for one entry's `cache_size_limit`.
        name = f"step_t{bucket.T}_b{bucket.B}"
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
        self._warmup_done = False

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
            decline: EligibilityResult | None = None
            try:
                self.bind(runner)
            except MegakernelLayoutError as exc:
                decline = (
                    exc.result
                    if isinstance(exc, LayoutDecline)
                    else EligibilityResult(False, "cache_layout_incompatible", str(exc))
                )
            except BaseException:
                # This rank is raising: still vote, so the others decline
                # instead of waiting in the agreement.
                agree_on_decline(
                    EligibilityResult(False, "bind_error", "bind failed"),
                    group=agreement_group(self._domain, runner),
                    codes=self._codes,
                )
                raise
            token = None
            if decline is None:
                token = buckets_token(self._steps, salt=self.fingerprint)
            group = agreement_group(self._domain, runner)
            decline = agree_on_decline(
                decline, group=group, token=token, codes=self._codes
            )
            if decline is not None:
                detach = getattr(self.program, "detach_draft", None)
                if callable(detach):
                    detach()
                logger.warning(
                    "Disabling %s megakernel at bind (%s): %s",
                    self._display_name,
                    decline.code,
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
                        domain=self._domain,
                    )
                return
            self.warmup(runner)

    def on_warmup_done(self, runner: Any) -> None:
        """The worker's warmup is over (after `capture_model`, i.e. after the
        stock precompile too); calls the program's optional `on_warmup_done()`
        once per bind."""
        del runner
        done = getattr(self.program, "on_warmup_done", None)
        if self._bound and not self._warmup_done and callable(done):
            done()
        self._warmup_done = self._bound

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


def maybe_create(runner: Any) -> ProgramDriver | DeclinedDriver | None:
    """Create the configured megakernel plugin's driver for `runner`."""
    return ProgramDriver.maybe_create(runner)


# The agreement and the startup guards live in `agreement` / `guards`; their
# public names stay importable from here.
__all__ = [
    "DECLINE_VOTE_CODES",
    "DeclinedDriver",
    "MeshDecline",
    "ProgramDriver",
    "agree_on_decline",
    "agreement_group",
    "buckets_token",
    "check_mesh_size",
    "check_runner_features",
    "check_spec",
    "detect_parallelism",
    "kv_connector_names",
    "loaded_numerics_note",
    "maybe_create",
    "runner_shapes",
    "vote_codes",
]
