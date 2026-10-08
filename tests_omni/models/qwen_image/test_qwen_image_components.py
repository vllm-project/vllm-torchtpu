# SPDX-License-Identifier: Apache-2.0
"""TPU component tests for Qwen-Image RoPE frequency embeddings."""

import pytest
import torch
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenEmbedLayer3DRope,
    QwenEmbedRope,
)

from vllm_torchtpu.omni.patches import apply_omni_model_specific_patches

pytestmark = [pytest.mark.omni]


class TestQwenEmbedRopeOnTpu:
    """Verify QwenEmbedRope and QwenEmbedLayer3DRope on TPU under inference_mode."""

    def test_torch_tpu_still_fails_on_pos_freqs_slice(self) -> None:
        """Canary for the bug behind _patch_qwen_image_rope_txt_freqs_clone.

        When this fails, torch_tpu handles the slice and the patch can be removed.
        """
        with torch.device("tpu"):
            embedder = QwenEmbedRope(
                theta=10000, axes_dim=[16, 56, 56], scale_rope=True
            )

        with torch.inference_mode():
            txt_slice = embedder.pos_freqs[16:80, ...]
            try:
                torch.real(txt_slice)
            except RuntimeError as e:
                assert "Cannot set version_counter for inference tensor" in str(e)
            else:
                pytest.fail(
                    "torch.real on a slice of pos_freqs now works under "
                    "torch.inference_mode() on TPU. Remove "
                    "_patch_qwen_image_rope_txt_freqs_clone and this canary."
                )

    @pytest.mark.parametrize(
        "rope_cls",
        [QwenEmbedRope, QwenEmbedLayer3DRope],
        ids=["rope_2d", "rope_3d"],
    )
    def test_qwen_image_rope_txt_freqs_under_inference_mode_on_tpu(
        self, rope_cls: type[torch.nn.Module]
    ) -> None:
        apply_omni_model_specific_patches()

        with torch.device("tpu"):
            tpu_embedder = rope_cls(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        cpu_embedder = rope_cls(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

        video_fhw = [(1, 32, 32)]
        txt_seq_lens = [64]

        # Pass pos_freqs' own device, as the transformer does through
        # hidden_states.device, so forward skips its .to(device) branch.
        device = tpu_embedder.pos_freqs.device
        with torch.inference_mode():
            tpu_vid, tpu_txt = tpu_embedder(video_fhw, txt_seq_lens, device=device)
            txt_real = torch.real(tpu_txt)
            txt_imag = torch.imag(tpu_txt)
            vid_real = torch.real(tpu_vid)
            vid_imag = torch.imag(tpu_vid)

        # pos_freqs must still be the tensor built outside inference mode;
        # otherwise this test would pass without the patch.
        assert not tpu_embedder.pos_freqs.is_inference()
        assert txt_real.device.type == "tpu"
        assert txt_imag.device.type == "tpu"

        ref_vid, ref_txt = cpu_embedder(
            video_fhw, txt_seq_lens, device=torch.device("cpu")
        )
        torch.testing.assert_close(
            txt_real.cpu(), torch.real(ref_txt), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            txt_imag.cpu(), torch.imag(ref_txt), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            vid_real.cpu(), torch.real(ref_vid), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            vid_imag.cpu(), torch.imag(ref_vid), atol=1e-5, rtol=1e-5
        )
