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
"""TPU model runner for autoregressive vLLM-Omni stages."""

from collections.abc import Sequence
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.multimodal.utils import (
    copy_mm_embedding_modality,
    set_mm_embedding_modality,
)
from vllm.sequence import IntermediateTensors
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
    needs_omni_connector,
)

from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_BUCKETS,
    _VISION_PATCH_BUCKETS,
    maybe_patch_qwen3_omni_thinker,
)
from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

# Model capability flags that ``GPUARModelRunner`` reads (through
# ``_model_omni_flag``) to drive the per-step preprocess / postprocess hooks.
# vLLM-Omni models set them in ``__init__``: the Qwen3-Omni thinker sets both
# to False, the talker sets both to True.
_UNSUPPORTED_HOOK_FLAGS = ("has_preprocess", "has_postprocess")


def _slice_mm_embeds_by_row_indices(
    mm_embeds: Sequence[torch.Tensor],
    row_indices: torch.Tensor,
) -> list[torch.Tensor]:
    """Slice ``mm_embeds`` across a contiguous 1D range of concatenated rows.

    Preserves each item's feature width and ``.modality`` attribute across
    chunk boundaries without concatenating mixed-dimension tensors.
    """
    if row_indices.ndim != 1:
        raise ValueError(
            f"Expected 1D row_indices, got shape {tuple(row_indices.shape)}."
        )
    if row_indices.numel() == 0:
        return []

    total_rows = sum(int(emb.shape[0]) for emb in mm_embeds)
    start = int(row_indices[0].item())
    end = int(row_indices[-1].item()) + 1
    if start < 0 or end > total_rows or end - start != row_indices.numel():
        raise ValueError(
            f"row_indices span [{start}, {end}) (len={row_indices.numel()}) "
            f"is invalid for {total_rows} cached multimodal embedding rows."
        )
    expected = torch.arange(
        start, end, dtype=row_indices.dtype, device=row_indices.device
    )
    if not torch.equal(row_indices, expected):
        raise ValueError(
            f"row_indices must be a contiguous ascending range [{start}, {end})."
        )

    result: list[torch.Tensor] = []
    offset = 0
    for emb in mm_embeds:
        n = int(emb.shape[0])
        item_start, item_end = offset, offset + n
        offset = item_end
        if item_end <= start or item_start >= end:
            continue
        sliced = emb[max(0, start - item_start) : min(n, end - item_start)]
        copy_mm_embedding_modality(emb, sliced)
        result.append(sliced)
    return result


def _expand_mm_dummy_batches(
    modality: str, dummy_batch: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand a dummy multimodal batch to warm every static encoder bucket."""
    if (
        modality == "audio"
        and isinstance(dummy_batch.get("input_features"), torch.Tensor)
        and isinstance(dummy_batch.get("audio_feature_lengths"), torch.Tensor)
    ):
        feats = dummy_batch["input_features"]
        lens = dummy_batch["audio_feature_lengths"]
        return [
            {
                **dummy_batch,
                "input_features": feats.new_zeros((*feats.shape[:-1], b * 100)),
                "audio_feature_lengths": torch.tensor(
                    [b * 100], dtype=lens.dtype, device=lens.device
                ),
            }
            for b in _AUDIO_CHUNK_BUCKETS
        ]
    if (
        modality == "image"
        and isinstance(dummy_batch.get("pixel_values"), torch.Tensor)
        and isinstance(dummy_batch.get("image_grid_thw"), torch.Tensor)
    ):
        px = dummy_batch["pixel_values"]
        grid = dummy_batch["image_grid_thw"]
        return [
            {
                **dummy_batch,
                "pixel_values": px.new_zeros((b, *px.shape[1:])),
                "image_grid_thw": torch.tensor(
                    [[1, 64, b // 64]], dtype=grid.dtype, device=grid.device
                ),
            }
            for b in _VISION_PATCH_BUCKETS
        ]
    return [dummy_batch]


class TPUARModelRunner(OmniTPUModelRunner, OmniConnectorModelRunnerMixin):
    """TPU runner for autoregressive stages; counterpart of ``GPUARModelRunner``.

    Serves stages that vLLM's own execute-and-sample loop covers, which is
    what the Qwen3-Omni thinker-only pipeline needs. What ``GPUARModelRunner``
    adds on top of that loop is refused with ``NotImplementedError`` rather
    than approximated, each where it can first be seen:

    - at construction, a stage config that needs inter-stage connectors
      (``needs_omni_connector``): every stage of a multi-stage pipeline sends
      or receives payloads through them, and they are not wired on TPU;
    - at construction, speculative decoding (``speculative_config``): draft-model
      auxiliary hidden states are not verified or wired on ``TPUARModelRunner``;
    - at ``load_model``, a model that declares the per-step
      ``preprocess`` / ``postprocess`` hooks (``has_preprocess`` /
      ``has_postprocess``): the talker path, which runs multi-token
      prediction around every step.

    ``OmniConnectorModelRunnerMixin`` sits on this leaf, as upstream, so the
    connector methods have their home when the multi-stage work lands.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._omni_mm_embeds_list: list[torch.Tensor] | None = None
        if needs_omni_connector(self.model_config):
            raise NotImplementedError(
                f"Stage {self.stage_id} ({self.model_stage!r}) needs inter-stage Omni "
                "connectors, which are not wired on TPU yet. Run the thinker-only "
                "pipeline (qwen3_omni_moe_thinking.yaml) or another single-stage "
                "deployment."
            )
        if self.speculative_config is not None:
            raise NotImplementedError(
                f"Stage {self.stage_id} ({self.model_stage!r}): speculative decoding "
                "(speculative_config) is not supported by TPUARModelRunner yet."
            )

    def load_model(self) -> None:
        """Load the stage model, then refuse one whose hooks nobody would call."""
        maybe_patch_qwen3_omni_thinker(
            self.model_config.architectures,
            enforce_eager=self.enforce_eager,
        )
        super().load_model()
        model = self.get_model()
        has_unsupported_flags = [
            flag
            for flag in _UNSUPPORTED_HOOK_FLAGS
            if bool(getattr(model, flag, False))
        ]
        if len(has_unsupported_flags) > 0:
            raise NotImplementedError(
                f"Stage {self.stage_id} ({self.model_stage!r}): "
                f"{type(model).__name__} declares {', '.join(has_unsupported_flags)}, "
                "so it relies on the per-step preprocess/postprocess hooks (the "
                "talker's multi-token prediction), which TPUARModelRunner does not "
                "run yet."
            )

    # TODO: retire the row-index proxy (_gather_mm_embeddings / _get_model_inputs /
    # execute_model cleanup) once core TPUModelRunner.execute_model slices
    # list[torch.Tensor] per chunk without concatenating via torch.cat.
    def execute_model(
        self,
        scheduler_output: Any,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> Any:
        """Run one step and always release cached per-item multimodal embeddings."""
        try:
            return super().execute_model(
                scheduler_output, intermediate_tensors=intermediate_tensors
            )
        finally:
            self._omni_mm_embeds_list = None

    def _gather_mm_embeddings(
        self,
        scheduler_output: Any,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Gather multimodal embeddings without concatenating mixed feature widths.

        Upstream: ``GPUModelRunner._gather_mm_embeddings`` (called from
        ``TPUModelRunner.execute_model``).
        Stayed the same: delegates encoder lookup, caching, and placeholder
        masking (``mm_embeds``, ``is_mm_embed``) to ``super()._gather_mm_embeddings``.
        Adapted for TPU: caches ``mm_embeds`` (which mix DeepStack vision
        tensors and audio tensors of different widths) in
        ``self._omni_mm_embeds_list`` and returns a 1D row-index range
        ``[torch.arange(total_rows)]`` so ``TPUModelRunner.execute_model``'s
        ``torch.cat`` and per-chunk slicing operate on uniform 1D indices while
        ``_get_model_inputs`` reconstructs each item with its ``.modality`` tag.
        """
        mm_embeds, is_mm_embed = super()._gather_mm_embeddings(
            scheduler_output, shift_computed_tokens
        )
        if not mm_embeds:
            self._omni_mm_embeds_list = None
            return [], is_mm_embed
        self._omni_mm_embeds_list = list(mm_embeds)
        total_rows = sum(int(e.shape[0]) for e in mm_embeds)
        return [torch.arange(total_rows, dtype=torch.long)], is_mm_embed

    def capture_model(self) -> None:
        super().capture_model()
        if self.enforce_eager or not self.supports_mm_inputs or not self._pp_is_first:
            return
        self._precompile_mm_inputs()

    @torch.no_grad()
    def _precompile_mm_inputs(self) -> None:
        """Warm static audio/vision encoder buckets and multimodal merge graphs.

        Upstream: ``GPUModelRunner._dummy_mm_run`` (for ``_get_mm_dummy_batch``,
        ``model.embed_multimodal``, and ``mm_budget.reset_cache()``) and
        ``TPUModelRunner``'s ``num_tokens_paddings`` bucket sweep.
        Stayed the same: builds dummy multimodal inputs per active modality from
        ``self._get_mm_dummy_batch(modality, 1)``, runs ``embed_multimodal``,
        and resets ``self.mm_budget``'s dummy cache afterward.
        Adapted for TPU: sorts ``modalities`` so TP ranks run collectives in
        lockstep, skips duplicate ``"video"`` encoder runs when ``"image"``
        already warms the shared ViT, expands dummy batches across all
        ``_AUDIO_CHUNK_BUCKETS`` and ``_VISION_PATCH_BUCKETS``, warms
        ``self._get_model_inputs`` across ``self.num_tokens_paddings`` per
        modality, and clears ``deepstack_input_embeds`` before serving.
        """
        if self.mm_budget is None or not self.mm_budget.mm_max_toks_per_item:
            return
        # Sort modalities so every TP worker runs the same encoder (and its TP
        # collectives) in lockstep; MultiModalBudget builds its dict from a set
        # whose iteration order varies across processes with PYTHONHASHSEED.
        modalities = sorted(self.mm_budget.mm_max_toks_per_item)
        if "image" in modalities and "video" in modalities:
            # Image and video share the vision tower; warming "image" compiles
            # all _VISION_PATCH_BUCKETS without synthesizing a full max_model_len
            # dummy video.
            modalities.remove("video")
        with self._precompile_timed("multimodal encoders and merge"):
            warmed_embeds: list[tuple[str, torch.Tensor]] = []
            for modality in modalities:
                batched_dummy_mm_inputs = self._get_mm_dummy_batch(modality, 1)
                sample_embed: torch.Tensor | None = None
                for bucket_inputs in _expand_mm_dummy_batches(
                    modality, batched_dummy_mm_inputs
                ):
                    dummy_encoder_outputs = self.model.embed_multimodal(**bucket_inputs)
                    if not dummy_encoder_outputs:
                        continue
                    outputs_list = list(dummy_encoder_outputs)
                    synchronize_tensors(outputs_list)
                    if sample_embed is None:
                        sample_embed = outputs_list[0]
                if sample_embed is None:
                    logger.warning_once(
                        "embed_multimodal returned no embeddings for modality "
                        "%r during TPU multimodal precompilation; skipping.",
                        modality,
                    )
                    continue
                warmed_embeds.append((modality, sample_embed))
            self.mm_budget.reset_cache()
            if not warmed_embeds:
                return
            token_buckets = [n for n in self.num_tokens_paddings if n > 0]
            for modality, sample_embed in warmed_embeds:
                dummy_mm = set_mm_embedding_modality(
                    torch.zeros_like(sample_embed[:1]), modality
                )
                for num_tokens in token_buckets:
                    dummy_input_ids = torch.zeros(
                        (num_tokens,), dtype=torch.int32, device=self.device
                    )
                    is_multimodal = torch.zeros((num_tokens,), dtype=torch.bool)
                    is_multimodal[0] = True
                    _, inputs_embeds = self._get_model_inputs(
                        dummy_input_ids, ([dummy_mm], is_multimodal)
                    )
                    if inputs_embeds is not None:
                        synchronize_tensors(inputs_embeds)
            # Clear any deepstack_input_embeds state left populated by the
            # vision merge warmup above before serving starts.
            dummy_input_ids = torch.zeros(
                (token_buckets[0],), dtype=torch.int32, device=self.device
            )
            _, inputs_embeds = self._get_model_inputs(dummy_input_ids, None)
            if inputs_embeds is not None:
                synchronize_tensors(inputs_embeds)

    def _get_model_inputs(
        self,
        input_ids: torch.Tensor,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Resolve 1D row-index ranges and guard DeepStack buffers across steps."""
        if mm_embed_inputs is not None and self._omni_mm_embeds_list is not None:
            chunk_embeds, is_mm_embed = mm_embed_inputs
            if len(chunk_embeds) == 1 and chunk_embeds[0].ndim == 1:
                mm_embed_inputs = (
                    _slice_mm_embeds_by_row_indices(
                        self._omni_mm_embeds_list, chunk_embeds[0]
                    ),
                    is_mm_embed,
                )

        use_deepstack = bool(
            self.supports_mm_inputs and getattr(self.model, "use_deepstack", False)
        )
        cleared_deepstack = (
            use_deepstack and self.model.deepstack_input_embeds_num_tokens > 0
        )
        if cleared_deepstack:
            self.model._clear_deepstack_input_embeds(
                self.model.deepstack_input_embeds_num_tokens
            )
        out_ids, inputs_embeds = super()._get_model_inputs(input_ids, mm_embed_inputs)
        if (
            use_deepstack
            and (cleared_deepstack or self.model.deepstack_input_embeds_num_tokens > 0)
            and inputs_embeds is not None
            and inputs_embeds.device.type == "tpu"
        ):
            synchronize_tensors(self.model.deepstack_input_embeds, wait=False)
        return out_ids, inputs_embeds

    def forward_model(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[torch.Tensor | IntermediateTensors, torch.Tensor | None]:
        """Run the stage model and return ``(hidden_states, aux_hidden_states)``.

        vLLM-Omni's runner (``OmniGPUModelRunner._model_forward``) normalises
        whatever an Omni stage model returns through ``model.make_omni_output``
        into an ``OmniOutput``: the text hidden states to sample from plus the
        multimodal sidecar the next stage consumes. ``TPUModelRunner.forward_model``
        instead unwraps by shape, for vLLM's ``(hidden_states,
        aux_hidden_states)`` draft-model convention, which an ``OmniOutput``
        (a four-field NamedTuple) or a staged thinker's ``(hidden_states,
        captured_layers)`` would match by accident. This override applies the
        Omni contract when ``make_omni_output`` is present (pipeline-parallel
        ``IntermediateTensors`` pass through, everything else goes through
        ``make_omni_output`` and only ``text_hidden_states`` continues; the
        sidecar is dropped because ``__init__`` refuses multi-stage configs),
        and falls back to ``TPUModelRunner.forward_model``'s unwrapping for
        plain vLLM models that do not define ``make_omni_output``.
        """
        kwargs: dict[str, Any] = {}
        if intermediate_tensors is not None:
            kwargs["intermediate_tensors"] = intermediate_tensors
        out = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if isinstance(out, IntermediateTensors):
            return out, None
        # @support_torch_compile may hand a single tensor back as a 1-element
        # list or tuple; the same unwrap as TPUModelRunner.forward_model.
        if isinstance(out, (list, tuple)) and len(out) == 1:
            out = out[0]
        if isinstance(out, OmniOutput):
            return out.text_hidden_states, None
        make_omni_output = getattr(self.model, "make_omni_output", None)
        if callable(make_omni_output):
            omni_out = make_omni_output(out)
            return omni_out.text_hidden_states, None
        if isinstance(out, (list, tuple)) and len(out) == 2:
            return out[0], out[1]
        return out, None
