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
"""TPU runtime and compilation optimizations for the LTX-2 3D Video VAE decoder."""

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_LTX2_VIDEO_VAE_DECODE_PATCHED = "_tpu_ltx2_video_vae_decode_patched"


def apply_ltx2_video_vae_patches() -> None:
    """Optimize and compile the LTX-2 Video VAE 3D decoder on TPU.

    Upstream ``LTX2VideoUpsampler3d.forward`` separately permutes and
    3x-flattens the 8-D residual tensor, calls ``.repeat(1, 4, 1, 1, 1)``
    (materializing a 4x larger ~761 MB tensor in ``up_blocks[2]``), and adds it
    after a second 8-D permute of ``conv_out``. Because addition commutes with
    permutation and temporal slicing, adding the unrepeated residual via
    broadcast before the 8-D permute eliminates the second 8-D permute, the 4x
    ``.repeat()`` allocation, and the post-permute 761 MB add while remaining
    algebraically identical. Combining this with ``rsqrt`` ``PerChannelRMSNorm``,
    single-reshape unpatchify, and ``torch.compile(backend="tpu", dynamic=False)``
    on ``LTX2VideoDecoder3d`` fuses the ResNet3D/upsampler stages on TPU.
    """
    from diffusers.models.autoencoders import autoencoder_kl_ltx2

    decoder_cls = autoencoder_kl_ltx2.LTX2VideoDecoder3d
    upsampler_cls = autoencoder_kl_ltx2.LTX2VideoUpsampler3d
    rms_norm_cls = autoencoder_kl_ltx2.PerChannelRMSNorm
    if getattr(decoder_cls.forward, _LTX2_VIDEO_VAE_DECODE_PATCHED, False):
        return

    def _per_channel_rms_norm_forward(
        self: Any,
        x: torch.Tensor,
        channel_dim: int | None = None,
    ) -> torch.Tensor:
        dim = self.channel_dim if channel_dim is None else channel_dim
        mean_sq = torch.mean(x * x, dim=dim, keepdim=True)
        return x * torch.rsqrt(mean_sq + self.eps)

    def _upsampler_forward(
        self: Any,
        hidden_states: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        batch_size, _, num_frames, height, width = hidden_states.shape
        s0, s1, s2 = self.stride
        conv_out = self.conv(hidden_states, causal=causal)
        if self.residual:
            repeats = (s0 * s1 * s2) // self.upscale_factor
            c_base = hidden_states.shape[1] // (s0 * s1 * s2)
            conv_out = (
                conv_out.reshape(
                    batch_size,
                    repeats,
                    c_base,
                    s0,
                    s1,
                    s2,
                    num_frames,
                    height,
                    width,
                )
                + hidden_states.reshape(
                    batch_size,
                    1,
                    c_base,
                    s0,
                    s1,
                    s2,
                    num_frames,
                    height,
                    width,
                )
            ).reshape(
                batch_size,
                repeats * c_base,
                s0,
                s1,
                s2,
                num_frames,
                height,
                width,
            )
        else:
            conv_out = conv_out.reshape(
                batch_size, -1, s0, s1, s2, num_frames, height, width
            )
        hidden_states = conv_out.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(
            batch_size, -1, num_frames * s0, height * s1, width * s2
        )[:, :, s0 - 1 :]
        return hidden_states

    def _decoder_forward_impl(
        self: Any,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        causal: bool | None = None,
    ) -> torch.Tensor:
        causal_flag = bool(causal or self.is_causal)
        hidden_states = self.conv_in(hidden_states, causal=causal_flag)
        if self.timestep_scale_multiplier is not None and temb is not None:
            temb = temb * self.timestep_scale_multiplier
        hidden_states = self.mid_block(hidden_states, temb, causal=causal_flag)
        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states, temb, causal=causal_flag)
        hidden_states = self.norm_out(hidden_states)
        if self.time_embedder is not None and temb is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1).unflatten(1, (2, -1))
            temb = temb + self.scale_shift_table[None, ..., None, None, None]
            shift, scale = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal_flag)
        p = self.patch_size
        p_t = self.patch_size_t
        batch_size, _, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, -1, p_t, p, p, num_frames, height, width
        )
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 4, 7, 3).reshape(
            batch_size, -1, num_frames * p_t, height * p, width * p
        )
        return hidden_states

    def _decoder_forward(
        self: Any,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        causal: bool | None = None,
    ) -> torch.Tensor:
        if (
            hidden_states.device.type == "tpu"
            and not getattr(self, "_tpu_disable_compile", False)
            and not torch.compiler.is_compiling()
        ):
            compiled = getattr(self, "_tpu_compiled_decoder_fn", None)
            if compiled is None:
                compiled = torch.compile(
                    _decoder_forward_impl,
                    backend="tpu",
                    dynamic=False,
                )
                self._tpu_compiled_decoder_fn = compiled
            return compiled(self, hidden_states, temb, causal)
        return _decoder_forward_impl(self, hidden_states, temb, causal)

    setattr(_decoder_forward, _LTX2_VIDEO_VAE_DECODE_PATCHED, True)
    rms_norm_cls.forward = _per_channel_rms_norm_forward
    upsampler_cls.forward = _upsampler_forward
    decoder_cls.forward = _decoder_forward
    logger.info(
        "Applied TPU patch: pre-permute LTX-2 VAE upsampler and compiled decoder."
    )
