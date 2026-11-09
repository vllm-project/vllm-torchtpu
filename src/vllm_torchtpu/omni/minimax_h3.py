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
"""TPU execution helpers for vLLM-Omni MiniMax-H3 diffusion models.

Eliminates per-step/per-block TPU host synchronizations (``.item()``,
``torch.unique``, dynamic boolean masks, and CPU-digest AdaLN caching) and
``@torch.compiler.disable`` graph breaks inside ``MiniMaxH3DiTBlock`` while
preserving numerical parity with upstream ``vllm_omni.diffusion.models.minimax_h3``.
"""

from __future__ import annotations

import math
import types
import weakref
from contextlib import nullcontext, suppress
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.omni.attention import refresh_block_sparse_step_gates

logger = init_logger(__name__)

_MINIMAX_H3_TPU_EXEC_PATCHED = "_tpu_minimax_h3_exec_patched"
_REQ_COUNTER = 0
_LAST_BRANCH_TEXT_LEN = 0


def disable_tpu_adaln_cache(model: Any) -> None:
    """Clear CPU-digest / ``.item()``-voting ``ExactProjectionCache`` on TPU DiTs.

    When AdaLN weight offload and prebuilt offline sidecars are inactive,
    computing the small ``[M <= 4, 2688] @ [2688, 96768]`` AdaLN projection
    directly on the TPU MXU takes ~0.08 ms/block with zero host syncs and
    compiles cleanly inside ``MiniMaxH3DiTBlock``.
    """
    adaln_cache = getattr(model, "adaln_cache", None)
    if (
        adaln_cache is not None
        and not getattr(model, "_adaln_offload", False)
        and getattr(adaln_cache, "sidecar", None) is None
    ):
        model.adaln_cache = None
        modules_fn = getattr(model, "modules", None)
        if callable(modules_fn):
            for submod in modules_fn():
                if getattr(submod, "_adaln_cache", None) is adaln_cache:
                    submod._adaln_cache = None


def wrap_condition_proj_cache(dit_self: Any) -> None:
    """Cache timestep-invariant ``condition_proj`` output within an active request."""
    cond_proj = getattr(dit_self, "condition_proj", None)
    if cond_proj is None or getattr(cond_proj, "_tpu_cond_cached", False):
        return
    owner_ref = weakref.ref(dit_self)
    orig_cond_fwd = cond_proj.forward

    def _cached_cond_fwd(_proj_self: Any, input_: torch.Tensor) -> Any:
        owner = owner_ref()
        req_id = (
            getattr(owner, "_tpu_active_denoise_req_id", None)
            if owner is not None
            else None
        )
        if req_id is None or input_.device.type != "tpu":
            return orig_cond_fwd(input_)
        key = (req_id, tuple(input_.shape), input_.dtype)
        cached = getattr(owner, "_tpu_cond_proj_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        out = orig_cond_fwd(input_)
        owner._tpu_cond_proj_cache = (key, out)
        return out

    cond_proj.forward = types.MethodType(_cached_cond_fwd, cond_proj)
    cond_proj._tpu_cond_cached = True


def _extract_prefix_tokens(video_layout: Any, valid_kv_length: int = 0) -> int:
    if video_layout is None:
        return 0
    prefix_tokens = 0
    spans = getattr(video_layout, "video_spans", None)
    if spans:
        for span in reversed(spans):
            if getattr(span, "role", None) == "target":
                prefix_tokens = int(span.start)
                break
    else:
        prefix_len = getattr(video_layout, "prefix_len", None)
        prefix_tokens = int(prefix_len) if prefix_len is not None else 0
    text_len = getattr(video_layout, "_tpu_text_len", None) or _LAST_BRANCH_TEXT_LEN
    if (
        valid_kv_length > 0
        and prefix_tokens > valid_kv_length // 4
        and text_len is not None
        and 0 < int(text_len) <= prefix_tokens
    ):
        return int(text_len)
    return prefix_tokens


def _resolve_tpu_turbo_fusion(od_config: Any, pipeline: Any) -> tuple[str, Any] | None:
    """Build a load-time checkpoint-stream weight fusion for LightX2V Turbo LoRAs
    when ``lora_backend == "distill"``, fusing ``(alpha / rank) * scale * (B @ A)``
    in FP32 prior to online FP8 quantization."""
    lora_backend = getattr(od_config, "lora_backend", None)
    backend_val = str(getattr(lora_backend, "value", lora_backend) or "").lower()
    if backend_val != "distill":
        return None
    lora_path = getattr(od_config, "lora_path", None)
    if isinstance(lora_path, (list, tuple)):
        if len(lora_path) != 1:
            return None
        lora_path = lora_path[0]
    if not lora_path:
        return None

    from safetensors import safe_open
    from vllm_omni.diffusion.models.minimax_h3 import fasth3
    from vllm_omni.diffusion.models.minimax_h3 import lora as h3_lora

    lora_file = h3_lora._select_turbo_file(lora_path)
    if lora_file is None:
        return None
    spec = h3_lora.parse_turbo_filename(lora_file.name)
    if spec is None:
        return None

    partition = str(getattr(pipeline, "partition", "fl2va"))
    if spec.task_family == "ref2v":
        if partition == "ref2va":
            target_prefix = "transformer."
            target_dit = getattr(pipeline, "transformer", None)
        elif hasattr(pipeline, "transformers_ref"):
            target_prefix = "transformers_ref."
            target_dit = getattr(pipeline, "transformers_ref", None)
        else:
            raise ValueError(
                f"{spec.filename} is a Ref2VA Turbo artifact, "
                f"but partition={partition!r} has no Ref2VA transformer loaded."
            )
    else:
        if partition == "ref2va":
            raise ValueError(
                f"{spec.filename} is an FL2VA/T2VA Turbo artifact, "
                f"but partition={partition!r} loads only Ref2VA."
            )
        target_prefix = "transformer."
        target_dit = getattr(pipeline, "transformer", None)

    if target_dit is None:
        return None

    with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        raw_alpha = metadata.get("alpha")
        alpha = float(raw_alpha) if raw_alpha is not None else float(spec.alpha)
        tensors = h3_lora._validate_and_convert_tensors(checkpoint)

    raw_scale = getattr(od_config, "lora_scale", 1.0)
    if isinstance(raw_scale, (list, tuple)):
        raw_scale = raw_scale[0] if raw_scale else 1.0
    lora_scale = float(1.0 if raw_scale is None else raw_scale)
    eff_scale = (alpha / float(spec.rank)) * lora_scale

    patches: dict[str, Any] = {}
    for name, tensor in tensors.items():
        split = fasth3._split_adapter_key(name)
        if split is None:
            raise ValueError(f"Unrecognized Turbo LoRA key: {name!r}")
        module, role = split
        target = fasth3._resolve_native_target(module)
        if target is None:
            raise ValueError(f"Unsupported Turbo LoRA target module: {module!r}")
        native_module, layout, _ = target
        # _validate_and_convert_tensors already swaps [value, gate] -> [gate, value]
        # on .ff.net.0.proj.lora_B.weight, so keep layout plain here to avoid
        # swapping a second time inside FastH3WeightFusion.fuse.
        eff_layout = fasth3._PLAIN if layout == fasth3._SWAP_HALVES else layout
        native_param = f"{native_module}.weight"
        patch = patches.setdefault(native_param, fasth3._ParamPatch(layout=eff_layout))
        if role == "lora_b":
            tensor = tensor.to(torch.float32) * eff_scale
        a, b = patch.low_rank.get(eff_layout, (None, None))
        patch.low_rank[eff_layout] = (tensor, b) if role == "lora_a" else (a, tensor)

    fusion = fasth3.FastH3WeightFusion(
        source=lora_file,
        patches=patches,
        head_dim=target_dit.arch.attention_head_dim,
        requires_vsa=False,
    )
    logger.info(
        "Prepared TPU load-time Turbo LoRA fusion %s -> %s "
        "(rank=%d, alpha=%g, scale=%g, eff_scale=%g, params=%d)",
        spec.filename,
        target_prefix,
        spec.rank,
        alpha,
        lora_scale,
        eff_scale,
        len(patches),
    )
    return target_prefix, fusion


def _sel_rows(
    rows: torch.Tensor, all_flag: bool, idx_dev: torch.Tensor | None
) -> torch.Tensor:
    if all_flag:
        return rows
    assert idx_dev is not None
    return rows.index_select(0, idx_dev)


def _write_rows(
    base: torch.Tensor,
    all_flag: bool,
    upd_idx: torch.Tensor | None,
    new_target: torch.Tensor,
    cond_idx: torch.Tensor | None,
    cond_anchor: torch.Tensor | None,
) -> torch.Tensor:
    if all_flag:
        return new_target.clone()
    assert upd_idx is not None
    out = base.clone()
    out.index_copy_(0, upd_idx, new_target)
    if cond_anchor is not None and cond_idx is not None:
        out.index_copy_(0, cond_idx, cond_anchor)
    return out


def tpu_rf_v_to_x0_nosync(
    xt: torch.Tensor, v: torch.Tensor, t_scalar: float
) -> torch.Tensor:
    """Convert flow velocity ``v`` to ``x0`` without per-step ``.item()`` D2H syncs."""
    if not math.isfinite(t_scalar) or t_scalar < 0.0 or t_scalar > 1.0:
        raise ValueError("timestep must be finite and in [0, 1]")
    cond_t = xt.new_tensor(t_scalar, dtype=xt.dtype)
    while cond_t.ndim < xt.ndim:
        cond_t = cond_t.unsqueeze(-1)
    sigma_t = 1.0 - cond_t
    return xt + sigma_t * v


def tpu_solver_step_nosync(
    solver: Any, state: torch.Tensor, denoised: torch.Tensor, step: int
) -> torch.Tensor:
    """Advance ``MiniMaxH3SampleSolver`` one step without per-step ``.item()`` syncs."""
    if step != solver.step_index or not 0 <= step < len(solver.coeffs):
        raise ValueError("H3 solver steps must be consumed once, in order")
    coeff = solver.coeffs[step]
    curr, nxt = solver.sigmas[step : step + 2]
    if solver.name == "euler" or coeff is None:
        if curr == 0.0:
            if nxt != 0.0:
                raise ValueError("sigma_next must be 0 when sigma_curr is 0")
            out = state
        else:
            sigma_curr_t = state.new_tensor(curr, dtype=torch.float32)
            sigma_next_t = state.new_tensor(nxt, dtype=torch.float32)
            sigma_ratio = sigma_next_t / sigma_curr_t
            out = (
                sigma_ratio * state.to(torch.float32)
                + (1.0 - sigma_ratio) * denoised.to(torch.float32)
            ).to(state.dtype)
    else:
        old = solver.old_denoised
        if old is None or old.shape != state.shape:
            raise ValueError("H3 solver state and denoised history shapes mismatch")
        ratio = state.new_tensor(nxt / curr, dtype=torch.float32)
        hb1, hb2 = coeff
        out = (
            ratio * state.to(torch.float32)
            + hb1 * denoised.to(torch.float32)
            + hb2 * old.to(torch.float32)
        ).to(state.dtype)
    if solver.name == "res_multistep":
        solver.old_denoised = (
            denoised.detach().clone() if step + 1 < len(solver.coeffs) else None
        )
    solver.step_index += 1
    return out


def install_minimax_h3_tpu_execution_patches() -> None:
    """Install TPU execution patches on ``vllm_omni.diffusion.models.minimax_h3``."""
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.models.minimax_h3 import (
        denoise_loop as dl,
    )
    from vllm_omni.diffusion.models.minimax_h3 import (
        minimax_h3_blocks as h3b,
    )
    from vllm_omni.diffusion.models.minimax_h3 import (
        minimax_h3_transformer as h3t,
    )
    from vllm_omni.diffusion.models.minimax_h3 import (
        pipeline_minimax_h3 as h3p,
    )
    from vllm_omni.diffusion.models.minimax_h3 import (
        sampling as h3s,
    )
    from vllm_omni.diffusion.models.minimax_h3 import (
        vae as h3v,
    )

    dit_cls = h3t.MiniMaxH3DiTModel
    if getattr(dit_cls.__init__, _MINIMAX_H3_TPU_EXEC_PATCHED, False):
        return

    # 1. Disable CPU-digest / TP-.item()-voting ExactProjectionCache on TPU and
    #    cache timestep-invariant prompt refinement (condition_proj + token_refiner)
    #    within an active single-request denoise loop.
    orig_dit_init = dit_cls.__init__

    def _tpu_dit_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_dit_init(self, *args, **kwargs)
        disable_tpu_adaln_cache(self)
        wrap_condition_proj_cache(self)
        if getattr(self, "token_refiner", None) is not None:
            self.token_refiner._tpu_owner_dit = weakref.ref(self)

    setattr(_tpu_dit_init, _MINIMAX_H3_TPU_EXEC_PATCHED, True)
    dit_cls.__init__ = _tpu_dit_init

    refiner_cls = h3t.MiniMaxH3TokenRefiner
    orig_refiner_fwd = refiner_cls.forward

    def _tpu_refiner_forward(
        self: Any,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        num_requests: int = 1,
    ) -> torch.Tensor:
        owner_ref = getattr(self, "_tpu_owner_dit", None)
        owner = owner_ref() if callable(owner_ref) else None
        req_id = (
            getattr(owner, "_tpu_active_denoise_req_id", None)
            if owner is not None
            else None
        )
        if req_id is None or num_requests != 1 or x.device.type != "tpu":
            return orig_refiner_fwd(
                self,
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                num_requests=num_requests,
            )
        key = (req_id, tuple(x.shape), int(max_seqlen))
        cached = getattr(owner, "_tpu_refiner_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        out = orig_refiner_fwd(
            self,
            x,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            num_requests=num_requests,
        )
        owner._tpu_refiner_cache = (key, out)
        return out

    refiner_cls.forward = _tpu_refiner_forward

    # 2. Allow single-request dense/block-sparse/Ulysses packed attention on TPU
    #    to compile straight through MiniMaxH3DiTBlock without @torch.compiler.disable
    #    splitting every block into 2 subgraphs + an eager DEFER_AND_FUSE island.
    attn_cls = h3b.MiniMaxH3Attention
    orig_run_packed_attention = attn_cls._run_packed_attention

    def _tpu_run_packed_attention(
        self: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_total: int,
        num_requests: int = 1,
        video_layout: Any = None,
        vsa_prefix_segments: tuple[int, ...] = (),
        gate_compress: torch.Tensor | None = None,
        reference_kv_state: Any = None,
        reference_kv_layer_index: int | None = None,
        reference_kv_compact: bool = False,
        vdn_window: Any = None,
    ) -> torch.Tensor:
        if (
            q.device.type == "tpu"
            and num_requests == 1
            and reference_kv_state is None
            and not reference_kv_compact
            and gate_compress is None
            and vdn_window is None
            and not h3b._ring_sequence_parallel_is_active(self.attention)
            and self.attention.attn_backend is not None
            and self.attention.attn_backend.supports_prefix_kv_slicing
            and 0 < max_seqlen <= packed_total
        ):
            used_kv = min(max_seqlen, packed_total)
            extra: dict[str, Any] = {
                "valid_kv_length": used_kv,
                "prefix_tokens": _extract_prefix_tokens(video_layout, used_kv),
            }
            vsa_sparsity = getattr(self, "vsa_sparsity", None)
            if vsa_sparsity is not None:
                extra["vsa_h3_sparsity"] = vsa_sparsity
            metadata = AttentionMetadata(
                attn_mask=None,
                packed_padding=None,
                extra=extra,
                video_layout=None,
            )
            return self.attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                metadata,
            ).squeeze(0)
        return orig_run_packed_attention(
            self,
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            packed_total=packed_total,
            num_requests=num_requests,
            video_layout=video_layout,
            vsa_prefix_segments=vsa_prefix_segments,
            gate_compress=gate_compress,
            reference_kv_state=reference_kv_state,
            reference_kv_layer_index=reference_kv_layer_index,
            reference_kv_compact=reference_kv_compact,
            vdn_window=vdn_window,
        )

    attn_cls._run_packed_attention = _tpu_run_packed_attention

    # 3. Cache static SP row-selection index tensors within the active request
    #    (scoped by _tpu_active_denoise_req_id and cleared at request exit)
    #    instead of running 4 dynamic boolean masks + torch.nonzero on TPU every step.
    orig_embed_row_indices = dit_cls._embed_row_indices

    def _tpu_embed_row_indices(
        self: Any,
        *,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        seq_len: int,
        local_span: tuple[int, int],
    ) -> Any:
        req_id = getattr(self, "_tpu_active_denoise_req_id", None)
        _, local_len = local_span
        if req_id is None or local_len == seq_len or img_pos.device.type != "tpu":
            return orig_embed_row_indices(
                self,
                img_pos=img_pos,
                audio_pos=audio_pos,
                text_pos=text_pos,
                seq_len=seq_len,
                local_span=local_span,
            )
        key = (
            req_id,
            int(img_pos.numel()),
            int(audio_pos.numel()),
            int(text_pos.numel()),
            seq_len,
            local_span,
        )
        cached = getattr(self, "_tpu_embed_row_indices_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        result = orig_embed_row_indices(
            self,
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            seq_len=seq_len,
            local_span=local_span,
        )
        self._tpu_embed_row_indices_cache = (key, result)
        return result

    dit_cls._embed_row_indices = _tpu_embed_row_indices

    # 4. Precompute static 4-class timestep inverse_indices and update/cond
    #    integer index tensors once on CPU per MiniMaxH3DenoiseBranch, avoiding
    #    per-step TPU torch.unique() D2H syncs and keeping M = len(present_classes)
    #    fixed across step 0 (where t_video == t_audio == 0) and steps 1..N-1.
    branch_cls = dl.MiniMaxH3DenoiseBranch
    orig_branch_init = branch_cls.__init__
    orig_forward_kwargs = branch_cls.forward_kwargs

    def _tpu_branch_init(self: Any, *args: Any, **kwargs: Any) -> None:
        global _LAST_BRANCH_TEXT_LEN
        orig_branch_init(self, *args, **kwargs)
        if hasattr(self, "text_len"):
            _LAST_BRANCH_TEXT_LEN = int(self.text_len)
        if getattr(self.device, "type", None) != "tpu":
            return
        cls = torch.zeros(self.seq_len, dtype=torch.long)
        img_pos_cpu = self.img_pos.view(-1).to(torch.long)
        upd_cpu = self.update_mask.view(-1).to(torch.bool)
        cls[img_pos_cpu[upd_cpu]] = 0
        cls[img_pos_cpu[~upd_cpu]] = 1
        aud_pos_cpu = self.audio_pos.view(-1).to(torch.long)
        aud_upd_cpu = self.audio_update_mask.view(-1).to(torch.bool)
        cls[aud_pos_cpu[aud_upd_cpu]] = 2
        cls[aud_pos_cpu[~aud_upd_cpu]] = 3
        present_classes, static_inv_cpu = torch.unique(
            cls, sorted=True, return_inverse=True
        )
        self._tpu_present_classes = tuple(int(c) for c in present_classes.tolist())
        self._tpu_static_inverse_dev = static_inv_cpu.to(
            device=self.device, dtype=torch.long
        )
        self._tpu_upd_all = bool(upd_cpu.all())
        self._tpu_upd_idx_dev = (
            None
            if self._tpu_upd_all
            else torch.nonzero(upd_cpu, as_tuple=False)
            .view(-1)
            .to(device=self.device, dtype=torch.long)
        )
        self._tpu_cond_idx_dev = (
            None
            if self._tpu_upd_all
            else torch.nonzero(~upd_cpu, as_tuple=False)
            .view(-1)
            .to(device=self.device, dtype=torch.long)
        )
        self._tpu_aud_upd_all = bool(aud_upd_cpu.all())
        self._tpu_aud_upd_idx_dev = (
            None
            if self._tpu_aud_upd_all
            else torch.nonzero(aud_upd_cpu, as_tuple=False)
            .view(-1)
            .to(device=self.device, dtype=torch.long)
        )
        self._tpu_aud_cond_idx_dev = (
            None
            if self._tpu_aud_upd_all
            else torch.nonzero(~aud_upd_cpu, as_tuple=False)
            .view(-1)
            .to(device=self.device, dtype=torch.long)
        )

    def _tpu_forward_kwargs(
        self: Any,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        t_video: float,
        t_audio: float,
        imgvid_cond_timestep: float,
        audio_ref_cond_timestep: float,
        video_target_timesteps: torch.Tensor | None = None,
        audio_target_timesteps: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if (
            getattr(self.device, "type", None) != "tpu"
            or video_target_timesteps is not None
            or audio_target_timesteps is not None
            or not hasattr(self, "_tpu_present_classes")
        ):
            return orig_forward_kwargs(
                self,
                video_rows=video_rows,
                audio_rows=audio_rows,
                t_video=t_video,
                t_audio=t_audio,
                imgvid_cond_timestep=imgvid_cond_timestep,
                audio_ref_cond_timestep=audio_ref_cond_timestep,
                video_target_timesteps=video_target_timesteps,
                audio_target_timesteps=audio_target_timesteps,
            )
        x = self.x_base.clone()
        x[0].index_copy_(0, self.img_pos_dev, video_rows)
        audio_x = self.audio_x_base.clone()
        audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
        eff_t_audio = float(t_audio if self.locked_audio_rows is None else 1.0)
        class_vals = (
            float(t_video),
            float(imgvid_cond_timestep),
            eff_t_audio,
            float(audio_ref_cond_timestep),
        )
        unique_timesteps = torch.tensor(
            [class_vals[c] for c in self._tpu_present_classes],
            dtype=torch.float32,
            device=self.device,
        )
        return {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": unique_timesteps,
            "inverse_indices": self._tpu_static_inverse_dev,
        }

    branch_cls.__init__ = _tpu_branch_init
    branch_cls.forward_kwargs = _tpu_forward_kwargs

    # 5. Precompute RoPE once per request on TPU, use static integer index_select /
    #    index_copy_ instead of dynamic boolean masks, and eliminate the 16 per-step
    #    `.item()` D2H syncs inside minimax_h3_rf_v_to_x0 + euler_eta0_step
    #    (checking output finiteness once at the end of the denoise loop).
    orig_denoise_loop = dl.minimax_h3_denoise_loop

    def _tpu_minimax_h3_denoise_loop(
        *,
        model: Any,
        positive: Any,
        initial_video_rows: torch.Tensor,
        initial_audio_rows: torch.Tensor,
        keyframe_cond_rows: torch.Tensor | None,
        audio_ref_rows: torch.Tensor | None = None,
        video_edit: Any = None,
        audio_edit: Any = None,
        sigmas_video: list[float],
        sigmas_audio: list[float],
        device: torch.device,
        imgvid_cond_noise_aug_for_inference: float = (
            dl.MINIMAX_H3_IMGVID_COND_TIMESTEP
        ),
        audio_cond_noise_aug_for_inference: float = (
            dl.MINIMAX_H3_AUDIO_REF_COND_TIMESTEP
        ),
        on_step: Any = None,
        step_profiler: Any = None,
        sampler: str = "euler",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            getattr(device, "type", None) != "tpu"
            or video_edit is not None
            or audio_edit is not None
            or not hasattr(positive, "_tpu_upd_all")
        ):
            return orig_denoise_loop(
                model=model,
                positive=positive,
                initial_video_rows=initial_video_rows,
                initial_audio_rows=initial_audio_rows,
                keyframe_cond_rows=keyframe_cond_rows,
                audio_ref_rows=audio_ref_rows,
                video_edit=video_edit,
                audio_edit=audio_edit,
                sigmas_video=sigmas_video,
                sigmas_audio=sigmas_audio,
                device=device,
                imgvid_cond_noise_aug_for_inference=imgvid_cond_noise_aug_for_inference,
                audio_cond_noise_aug_for_inference=audio_cond_noise_aug_for_inference,
                on_step=on_step,
                step_profiler=step_profiler,
                sampler=sampler,
            )

        if len(sigmas_video) != len(sigmas_audio):
            raise ValueError("video/audio sigma schedules must have equal length")
        if len(sigmas_video) < 2:
            raise ValueError("sigma schedules need at least 2 entries")

        raw_model = getattr(model, "_orig_mod", model)
        disable_tpu_adaln_cache(raw_model)
        wrap_condition_proj_cache(raw_model)
        if getattr(raw_model, "token_refiner", None) is not None:
            raw_model.token_refiner._tpu_owner_dit = weakref.ref(raw_model)
        positive.prepare_rope_table(model)

        video_rows = initial_video_rows.to(device=device, dtype=torch.float32).clone()
        audio_rows = initial_audio_rows.to(device=device, dtype=torch.float32).clone()
        n_cond = int((~positive.update_mask).sum())
        if keyframe_cond_rows is None:
            if n_cond != 0:
                raise ValueError(
                    f"layout has {n_cond} cond rows but keyframe_cond_rows is None"
                )
            cond_anchor = None
        else:
            if int(keyframe_cond_rows.shape[0]) != n_cond:
                raise ValueError(
                    f"keyframe_cond_rows {int(keyframe_cond_rows.shape[0])} "
                    f"!= layout cond rows {n_cond}"
                )
            cond_anchor = keyframe_cond_rows.to(device=device, dtype=torch.float32)
            if positive._tpu_cond_idx_dev is not None:
                video_rows.index_copy_(0, positive._tpu_cond_idx_dev, cond_anchor)

        n_audio_ref = int((~positive.audio_update_mask).sum())
        if audio_ref_rows is None:
            if n_audio_ref != 0:
                raise ValueError(
                    f"layout has {n_audio_ref} audio ref rows "
                    "but audio_ref_rows is None"
                )
            audio_anchor = None
        else:
            if int(audio_ref_rows.shape[0]) != n_audio_ref:
                raise ValueError(
                    f"audio_ref_rows {int(audio_ref_rows.shape[0])} "
                    f"!= layout audio ref rows {n_audio_ref}"
                )
            audio_anchor = audio_ref_rows.to(device=device, dtype=torch.float32)
            if positive._tpu_aud_cond_idx_dev is not None:
                audio_rows.index_copy_(0, positive._tpu_aud_cond_idx_dev, audio_anchor)

        video_solver = h3s.create_h3_sample_solver(sampler, sigmas_video)
        audio_solver = h3s.create_h3_sample_solver(sampler, sigmas_audio)
        num_steps = len(sigmas_video) - 1
        reference_kv_state = None
        create_reference_kv_state = getattr(model, "create_reference_kv_state", None)
        if callable(create_reference_kv_state):
            reference_kv_state = create_reference_kv_state(
                global_reference_rows=n_cond,
                device=device,
            )
        if reference_kv_state is not None:
            positive.static_kwargs["reference_kv_state"] = reference_kv_state

        global _REQ_COUNTER
        _REQ_COUNTER += 1
        raw_model._tpu_active_denoise_req_id = _REQ_COUNTER
        raw_model._tpu_embed_row_indices_cache = None
        raw_model._tpu_cond_proj_cache = None
        raw_model._tpu_refiner_cache = None

        try:
            for step in range(num_steps):
                dl.check_request_cancellation()
                step_cm = (
                    step_profiler(step) if step_profiler is not None else nullcontext()
                )
                with step_cm:
                    s_v = float(sigmas_video[step])
                    s_a = float(sigmas_audio[step])
                    if reference_kv_state is not None:
                        reference_kv_state.begin_step(step, sigma=s_v)
                    dl.minimax_h3_publish_denoise_progress(step, s_v, num_steps)
                    refresh_block_sparse_step_gates(step, num_steps)
                    t_v, t_a = 1.0 - s_v, 1.0 - s_a
                    imgvid_cond_t = max(t_v, float(imgvid_cond_noise_aug_for_inference))
                    audio_ref_cond_t = max(
                        t_a, float(audio_cond_noise_aug_for_inference)
                    )

                    fk = positive.forward_kwargs(
                        video_rows=video_rows,
                        audio_rows=audio_rows,
                        t_video=t_v,
                        t_audio=t_a,
                        imgvid_cond_timestep=imgvid_cond_t,
                        audio_ref_cond_timestep=audio_ref_cond_t,
                    )
                    if (
                        getattr(getattr(raw_model, "arch", None), "pdd_steps", None)
                        is not None
                    ):
                        fk["pdd_step"] = step
                    with torch.inference_mode():
                        v_video, v_audio = model(**fk)
                    if reference_kv_state is not None:
                        reference_kv_state.end_step()

                    mv_video_t = _sel_rows(
                        v_video.float(),
                        positive._tpu_upd_all,
                        positive._tpu_upd_idx_dev,
                    )
                    mv_audio_t = _sel_rows(
                        v_audio.float(),
                        positive._tpu_aud_upd_all,
                        positive._tpu_aud_upd_idx_dev,
                    )
                    cur_v = _sel_rows(
                        video_rows,
                        positive._tpu_upd_all,
                        positive._tpu_upd_idx_dev,
                    )
                    x0_video = tpu_rf_v_to_x0_nosync(cur_v, mv_video_t, t_v)
                    new_target = tpu_solver_step_nosync(
                        video_solver, cur_v, x0_video, step
                    )
                    video_rows = _write_rows(
                        video_rows,
                        positive._tpu_upd_all,
                        positive._tpu_upd_idx_dev,
                        new_target,
                        positive._tpu_cond_idx_dev,
                        cond_anchor,
                    )

                    cur_a = _sel_rows(
                        audio_rows,
                        positive._tpu_aud_upd_all,
                        positive._tpu_aud_upd_idx_dev,
                    )
                    x0_audio = tpu_rf_v_to_x0_nosync(cur_a, mv_audio_t, t_a)
                    new_audio = tpu_solver_step_nosync(
                        audio_solver, cur_a, x0_audio, step
                    )
                    eff_new_audio = (
                        new_audio
                        if positive.locked_audio_rows is None
                        else positive.locked_audio_rows
                    )
                    audio_rows = _write_rows(
                        audio_rows,
                        positive._tpu_aud_upd_all,
                        positive._tpu_aud_upd_idx_dev,
                        eff_new_audio,
                        positive._tpu_aud_cond_idx_dev,
                        audio_anchor,
                    )
                    if on_step is not None:
                        on_step(step, video_rows, audio_rows)
                    # check_request_cancellation(synchronize=True) calls device
                    # synchronize() only when a cancellation flag is active.
                    dl.check_request_cancellation(synchronize=True)

            # Single end-of-loop finiteness check (replaces 16 .item() syncs/step).
            if not bool(torch.isfinite(video_rows).all().item()):
                raise ValueError("v must be finite")
            if not bool(torch.isfinite(audio_rows).all().item()):
                raise ValueError("audio output must be finite")
        finally:
            raw_model._tpu_active_denoise_req_id = None
            raw_model._tpu_embed_row_indices_cache = None
            raw_model._tpu_cond_proj_cache = None
            raw_model._tpu_refiner_cache = None
            dl.minimax_h3_publish_denoise_progress(None, None, None)
            refresh_block_sparse_step_gates(None, None)
            if reference_kv_state is not None:
                positive.static_kwargs.pop("reference_kv_state", None)
                reference_kv_state.close()
        return video_rows, audio_rows

    dl.minimax_h3_denoise_loop = _tpu_minimax_h3_denoise_loop
    if getattr(h3p, "minimax_h3_denoise_loop", None) is orig_denoise_loop:
        h3p.minimax_h3_denoise_loop = _tpu_minimax_h3_denoise_loop

    # 6. Support lora_backend="distill" on MiniMaxH3Pipeline for LightX2V Turbo
    #    .safetensors adapters, fusing (alpha / rank) * scale * (B @ A) in FP32
    #    into the checkpoint stream prior to online FP8 quantization.
    #    Resolve lazily in load_weights / lora_is_fused because
    #    install_minimax_h3_tpu_execution_patches() is invoked from
    #    get_diffusion_attn_backend_cls while MiniMaxH3Pipeline.__init__ is
    #    already on the call stack.
    pipe_cls = h3p.MiniMaxH3Pipeline
    orig_pipe_load_weights = pipe_cls.load_weights

    def _ensure_tpu_turbo_fusion(self: Any) -> tuple[str, Any] | None:
        if hasattr(self, "_tpu_turbo_fusion_resolved"):
            return getattr(self, "_tpu_turbo_fusion", None)
        self._tpu_turbo_fusion_resolved = True
        self._tpu_turbo_fusion = None
        if (
            getattr(self, "_fasth3", None) is None
            and getattr(self, "_vdn", None) is None
        ):
            self._tpu_turbo_fusion = _resolve_tpu_turbo_fusion(
                getattr(self, "od_config", None), self
            )
        return self._tpu_turbo_fusion

    def _tpu_pipe_load_weights(self: Any, weights: Any) -> set[str]:
        turbo_fusion = _ensure_tpu_turbo_fusion(self)
        if turbo_fusion is None:
            return orig_pipe_load_weights(self, weights)
        target_prefix, fusion = turbo_fusion

        def _fused_stream(raw_weights: Any) -> Any:
            for name, tensor in raw_weights:
                if name.startswith(target_prefix):
                    sub_name = name[len(target_prefix) :]
                    yield name, fusion.fuse(sub_name, tensor)
                else:
                    yield name, tensor

        loaded_with_prefix = orig_pipe_load_weights(self, _fused_stream(weights))
        sub_loaded = {
            n[len(target_prefix) :]
            for n in loaded_with_prefix
            if n.startswith(target_prefix)
        }
        fusion.validate_fully_applied(sub_loaded)
        return loaded_with_prefix

    def _get_lora_is_fused(self: Any) -> bool:
        return bool(
            getattr(self, "_fasth3", None) is not None
            or getattr(self, "_vdn", None) is not None
            or _ensure_tpu_turbo_fusion(self) is not None
            or getattr(self, "_tpu_lora_is_fused_flag", False)
        )

    def _set_lora_is_fused(self: Any, value: bool) -> None:
        self._tpu_lora_is_fused_flag = bool(value)

    pipe_cls.load_weights = _tpu_pipe_load_weights
    pipe_cls.lora_is_fused = property(_get_lora_is_fused, _set_lora_is_fused)

    # 7. Wrap MiniMaxH3VideoVAE.device_module on TPU so
    #    `with self.device_module.device(device):` inside encode_image /
    #    encode_video does not invoke TorchTPU's internal-only _DeviceContext
    #    (which raises RuntimeError when called outside torch/torch_tpu).
    class _TpuSafeDeviceModuleProxy:
        def __init__(self, mod: Any) -> None:
            self._mod = mod

        def device(self, _device: Any) -> Any:
            return nullcontext()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._mod, name)

    vae_cls = h3v.MiniMaxH3VideoVAE
    orig_vae_init = vae_cls.__init__
    orig_encode_image = vae_cls.encode_image
    orig_encode_video = vae_cls.encode_video
    orig_decode_latent = vae_cls.decode_latent
    orig_decode_with_chunks = getattr(vae_cls, "decode_with_chunks", None)

    def _sync_tpu() -> None:
        if hasattr(torch, "accelerator"):
            with suppress(Exception):
                if torch.accelerator.is_available():
                    torch.accelerator.synchronize()

    def _wrap_vae_device_module(vae_self: Any) -> None:
        target = getattr(vae_self, "_device_target", None)
        if getattr(target, "type", None) != "tpu":
            return
        dev_mod = getattr(vae_self, "device_module", None)
        if dev_mod is not None and not isinstance(dev_mod, _TpuSafeDeviceModuleProxy):
            vae_self.device_module = _TpuSafeDeviceModuleProxy(dev_mod)

        model = getattr(vae_self, "model", None)
        if model is None or getattr(model, "_tpu_tile_patched", False):
            return

        orig_run_tile_tasks = getattr(model, "_run_tile_tasks", None)
        orig_tiled_encode = getattr(model, "tiled_encode", None)

        if callable(orig_run_tile_tasks):

            def _tpu_run_tile_tasks(
                model_self: Any,
                tiles: Any,
                tile_indices: Any,
                forward_fn: Any,
                stack_tiling: bool,
                cls_agg: Any = None,
            ) -> list[torch.Tensor]:
                try:
                    param_dev = next(model_self.parameters()).device
                except StopIteration:
                    param_dev = target
                if getattr(param_dev, "type", None) != "tpu":
                    return orig_run_tile_tasks(
                        tiles, tile_indices, forward_fn, stack_tiling, cls_agg
                    )
                if stack_tiling and tile_indices:
                    tiles_dev = [
                        t.cpu().contiguous().to(device=param_dev) for t in tiles
                    ]
                    _sync_tpu()
                    out = orig_run_tile_tasks(
                        tiles_dev, tile_indices, forward_fn, stack_tiling, cls_agg
                    )
                    _sync_tpu()
                    return out
                tasks: list[torch.Tensor] = []
                for idx in tile_indices:
                    tile_tpu = tiles[idx].cpu().contiguous().to(device=param_dev)
                    _sync_tpu()
                    out_tile = forward_fn(tile_tpu)
                    if cls_agg is not None:
                        cls_agg.collect()
                    _sync_tpu()
                    tasks.append(out_tile)
                return tasks

            model._run_tile_tasks = types.MethodType(_tpu_run_tile_tasks, model)

        if hasattr(model, "_all_gather_tiled_results") and hasattr(
            vae_self, "_native_parallel_state"
        ):

            def _tpu_all_gather_tiled_results(
                model_self: Any,
                tasks: list[torch.Tensor],
                num_tiles: int,
            ) -> list[torch.Tensor]:
                import torch.distributed as dist

                state = vae_self._native_parallel_state()
                group = state["sp_process_group"]
                sp_size = int(state["sp_size"])
                sp_rank = int(state["sp_rank"])
                if not tasks:
                    raise ValueError(f"Found empty tasks on sp rank {sp_rank}")
                max_tasks = -(-num_tiles // sp_size)
                if len(tasks) > max_tasks:
                    raise ValueError(
                        f"sp rank {sp_rank} holds {len(tasks)} tiles but round-robin "
                        f"ownership of {num_tiles} tiles across {sp_size} ranks allows "
                        f"at most {max_tasks}"
                    )
                if len(tasks) == max_tasks:
                    stacked = torch.stack(tasks, dim=0).contiguous()
                else:
                    stacked = tasks[0].new_zeros((max_tasks, *tasks[0].shape))
                    stacked[: len(tasks)].copy_(torch.stack(tasks, dim=0))
                buffer = stacked.new_empty((sp_size, *stacked.shape))
                _sync_tpu()
                dist.all_gather_into_tensor(buffer, stacked, group=group)
                _sync_tpu()
                results: list[torch.Tensor] = [None] * num_tiles  # type: ignore[list-item]
                for rank in range(sp_size):
                    num_rank_tasks = -(-(num_tiles - rank) // sp_size)
                    for k in range(num_rank_tasks):
                        results[k * sp_size + rank] = buffer[rank, k].clone()
                return results

            model._all_gather_tiled_results = types.MethodType(
                _tpu_all_gather_tiled_results, model
            )

        if callable(orig_tiled_encode):

            def _tpu_tiled_encode(model_self: Any, x: torch.Tensor) -> torch.Tensor:
                if x.device.type != "tpu":
                    return orig_tiled_encode(x)
                _sync_tpu()
                z_cpu = orig_tiled_encode(x.cpu())
                out = z_cpu.contiguous().to(device=x.device)
                _sync_tpu()
                return out

            model.tiled_encode = types.MethodType(_tpu_tiled_encode, model)

        model._tpu_tile_patched = True

    def _tpu_vae_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_vae_init(self, *args, **kwargs)
        _wrap_vae_device_module(self)

    def _tpu_encode_image(self: Any, *args: Any, **kwargs: Any) -> Any:
        _wrap_vae_device_module(self)
        return orig_encode_image(self, *args, **kwargs)

    def _tpu_encode_video(self: Any, *args: Any, **kwargs: Any) -> Any:
        _wrap_vae_device_module(self)
        return orig_encode_video(self, *args, **kwargs)

    def _tpu_decode_latent(self: Any, *args: Any, **kwargs: Any) -> Any:
        _wrap_vae_device_module(self)
        return orig_decode_latent(self, *args, **kwargs)

    vae_cls.__init__ = _tpu_vae_init
    vae_cls.encode_image = _tpu_encode_image
    vae_cls.encode_video = _tpu_encode_video
    vae_cls.decode_latent = _tpu_decode_latent
    if callable(orig_decode_with_chunks):

        def _tpu_decode_with_chunks(self: Any, *args: Any, **kwargs: Any) -> Any:
            _wrap_vae_device_module(self)
            return orig_decode_with_chunks(self, *args, **kwargs)

        vae_cls.decode_with_chunks = _tpu_decode_with_chunks

    audio_vae_cls = getattr(h3v, "MiniMaxH3AudioVAE", None)
    if audio_vae_cls is not None and hasattr(audio_vae_cls, "encode_waveform"):
        orig_encode_waveform = audio_vae_cls.encode_waveform

        def _tpu_encode_waveform(self: Any, *args: Any, **kwargs: Any) -> Any:
            target = getattr(self, "_device_target", None)
            model = getattr(self, "model", None)
            if getattr(target, "type", None) == "tpu" and model is not None:
                for mod_name in ("encoder", "pre_block", "mean_proj"):
                    sub = getattr(model, mod_name, None)
                    if isinstance(sub, torch.nn.Module):
                        sub.to("cpu")
            return orig_encode_waveform(self, *args, **kwargs)

        audio_vae_cls.encode_waveform = _tpu_encode_waveform

    logger.info(
        "Applied TPU patch: zero-host-sync MiniMax-H3 AdaLN, SP indices, "
        "distill LoRA fusion, VAE device_module proxy, and denoise loop."
    )
