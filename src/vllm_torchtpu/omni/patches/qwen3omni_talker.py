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
"""TPU patches for the Qwen3-Omni Talker stage and Code Predictor.

1. Code Predictor CustomOp dispatch & autocast:
   ``qwen3_code_predictor`` defines ``_RMSNorm`` and ``_RotaryEmbedding``
   without ``forward_tpu`` implementations, and ``CodePredictorBaseModel``
   enters ``torch.amp.autocast("tpu", dtype=torch.float32)`` when given fp16
   inputs. ``maybe_patch_qwen3_omni_talker`` binds ``_RMSNorm.forward_tpu`` to
   ``forward_native``, implements ``_RotaryEmbedding.forward_tpu``, and skips
   the fp32 autocast on TPU.

2. Code Predictor 15-step residual AR loop:
   ``CodePredictorWrapper.forward`` runs 15 sequential 1-token transformer
   steps per 80 ms frame (5 layers, hidden size 1024) and calls
   ``torch.multinomial`` on every step. On TPU/XLA that unrolls 75 transformer
   layers and 15 host-syncing ``multinomial`` calls per frame; the wrapper
   runs the small residual AR loop on CPU and returns the resulting ``[B, 16]``
   codes and summed embedding on the Talker's device.

3. Talker preprocess:
   Detaches ``codes["audio"]`` to CPU during prefill so inter-stage
   serialization never touches a device tensor.

Applied by ``TPUARModelRunner.load_model()`` for
``Qwen3OmniMoeForConditionalGeneration`` only.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.omni.patches.qwen3omni_thinker import _QWEN3_OMNI_ARCH

logger = init_logger(__name__)

_TALKER_PATCHED_ATTR = "_tpu_talker_patched"


# TODO(sarthakdev): Upstream TPU CustomOp hooks and static-friendly Talker/Code
# Predictor helpers to vllm-omni and remove this monkey-patch once vllm-omni
# natively supports TPU execution for Qwen3-Omni Talker.
def maybe_patch_qwen3_omni_talker(
    architectures: list[str] | None = None,
) -> bool:
    """Apply TPU patches for the Qwen3-Omni Talker and Code Predictor.

    Applies to ``Qwen3OmniMoeForConditionalGeneration`` only, matched
    literally in ``architectures``. Returns ``True`` when the vLLM-Omni
    Talker / Code Predictor classes are newly patched, and ``False`` when
    ``architectures`` does not match or everything is already patched.
    """
    if architectures is None or _QWEN3_OMNI_ARCH not in architectures:
        return False

    from vllm_omni.model_executor.models.common import qwen3_code_predictor as cp
    from vllm_omni.model_executor.models.qwen3_omni import (
        qwen3_omni as qwen3_omni_mod,
    )

    patched = False
    if getattr(cp._RMSNorm, "forward_tpu", None) is not cp._RMSNorm.forward_native:
        cp._RMSNorm.forward_tpu = cp._RMSNorm.forward_native
        patched = True

    if not getattr(
        getattr(cp._RotaryEmbedding, "forward_tpu", None),
        _TALKER_PATCHED_ATTR,
        False,
    ):

        def _rotary_forward_tpu(
            self: Any, x: torch.Tensor, position_ids: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if self.cos_cached.device != position_ids.device:
                self.cos_cached = self.cos_cached.to(position_ids.device)
                self.sin_cached = self.sin_cached.to(position_ids.device)
            return self._lookup(x, position_ids)

        setattr(_rotary_forward_tpu, _TALKER_PATCHED_ATTR, True)
        cp._RotaryEmbedding.forward_tpu = _rotary_forward_tpu
        patched = True

    if not getattr(cp.CodePredictorBaseModel.forward, _TALKER_PATCHED_ATTR, False):
        # Adapted from CodePredictorBaseModel.forward in
        # vllm_omni/model_executor/models/common/qwen3_code_predictor.py
        # (vllm-omni==0.32.0rc1 and 0.33.0rc1), adding "tpu" alongside "cpu"
        # and "npu" in the fp32 autocast exclusion check.
        def _base_forward(
            self: Any, inputs_embeds: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
            input_dtype = inputs_embeds.dtype
            use_fp32 = (
                input_dtype == torch.float16
                and inputs_embeds.device.type not in ("cpu", "npu", "tpu")
            )
            if use_fp32:
                inputs_embeds = inputs_embeds.float()
            hidden_states = inputs_embeds
            ctx = (
                torch.amp.autocast(
                    device_type=inputs_embeds.device.type, dtype=torch.float32
                )
                if use_fp32
                else nullcontext()
            )
            with ctx:
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                for layer in self.layers:
                    hidden_states = layer(hidden_states, position_embeddings)
                hidden_states = self.norm(hidden_states)
            return hidden_states.to(input_dtype)

        setattr(_base_forward, _TALKER_PATCHED_ATTR, True)
        cp.CodePredictorBaseModel.forward = _base_forward
        patched = True

    if not getattr(cp.CodePredictorWrapper.forward, _TALKER_PATCHED_ATTR, False):
        orig_cp_forward = cp.CodePredictorWrapper.forward

        # Adapted from CodePredictorWrapper.forward in
        # vllm_omni/model_executor/models/common/qwen3_code_predictor.py
        # (vllm-omni==0.32.0rc1 and 0.33.0rc1).
        def _wrapper_forward(
            self: Any,
            layer0_code: torch.Tensor,
            layer0_embed: torch.Tensor,
            last_talker_hidden: torch.Tensor,
            do_sample: bool = True,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 1.0,
            generator: torch.Generator | None = None,
            generators: list[torch.Generator | None] | None = None,
            sample_uniforms: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            device = layer0_code.device
            if device.type in ("tpu", "xla"):
                if next(self.parameters()).device.type != "cpu":
                    self.to("cpu")
                    self._compiled_model_fwd = None
                if self._compiled_model_fwd is None:
                    self._model_dtype = next(self.model.parameters()).dtype
                    self._lm_heads_list = list(self.lm_head)
                    self._codec_embeds_list = list(self.model.codec_embedding)
                    self._compiled_model_fwd = self.model.forward
                result = orig_cp_forward(
                    self,
                    layer0_code=layer0_code.cpu(),
                    layer0_embed=layer0_embed.cpu(),
                    last_talker_hidden=last_talker_hidden.cpu(),
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator,
                    generators=generators,
                    sample_uniforms=(
                        sample_uniforms.cpu() if sample_uniforms is not None else None
                    ),
                )
                # code_predictor_forward copies `codes` into a `device`-allocated
                # `result_codes` tensor; TPUARModelRunner._get_model_inputs then
                # moves the final `code_predictor_codes` to CPU for the buffer.
                if isinstance(result, tuple):
                    codes, proj_buf = result
                    return codes.to(device=device), proj_buf.to(device=device)
                return result.to(device=device)
            return orig_cp_forward(
                self,
                layer0_code,
                layer0_embed,
                last_talker_hidden,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
                generators=generators,
                sample_uniforms=sample_uniforms,
            )

        setattr(_wrapper_forward, _TALKER_PATCHED_ATTR, True)
        cp.CodePredictorWrapper.forward = _wrapper_forward
        patched = True

    omni_cls = qwen3_omni_mod.Qwen3OmniMoeForConditionalGeneration
    if not getattr(omni_cls.talker_preprocess, _TALKER_PATCHED_ATTR, False):
        orig_talker_preprocess = omni_cls.talker_preprocess

        def _talker_preprocess(
            self: Any,
            input_ids: torch.Tensor | None,
            input_embeds: torch.Tensor | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
            ids, embeds, update_dict = orig_talker_preprocess(
                self, input_ids, input_embeds, *args, **kwargs
            )
            # Prefill populates update_dict["codes"]["audio"] with zero codes
            # on the model device; decode populates codes via talker_mtp in
            # TPUARModelRunner._get_model_inputs. Detach prefill codes to CPU
            # here so inter-stage serialization never touches a device tensor.
            codes = update_dict.get("codes") if isinstance(update_dict, dict) else None
            if isinstance(codes, dict) and isinstance(codes.get("audio"), torch.Tensor):
                codes["audio"] = codes["audio"].detach().cpu()
            return ids, embeds, update_dict

        setattr(_talker_preprocess, _TALKER_PATCHED_ATTR, True)
        omni_cls.talker_preprocess = _talker_preprocess
        patched = True

    if patched:
        logger.info("Applied TPU patch: Qwen3-Omni Talker and Code Predictor.")
    return patched
