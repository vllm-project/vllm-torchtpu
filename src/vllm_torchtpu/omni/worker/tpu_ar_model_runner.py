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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from vllm.logger import init_logger
from vllm.multimodal.utils import (
    copy_mm_embedding_modality,
    set_mm_embedding_modality,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput
from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
    needs_omni_connector,
)

from vllm_torchtpu.omni.patches.qwen3omni_talker import (
    maybe_patch_qwen3_omni_talker,
)
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_BUCKETS,
    _VISION_PATCH_BUCKETS,
    maybe_patch_qwen3_omni_thinker,
)
from vllm_torchtpu.omni.worker.tpu_model_runner import (
    OmniTPUModelRunner,
    _unwrap_model,
)
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = init_logger(__name__)


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
        clip_lens = [b * 100 for b in _AUDIO_CHUNK_BUCKETS]
        return [
            {
                **dummy_batch,
                "input_features": feats.new_zeros((*feats.shape[:-1], sum(clip_lens))),
                "audio_feature_lengths": torch.tensor(
                    clip_lens, dtype=lens.dtype, device=lens.device
                ),
            }
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


class _OmniStageModel(Protocol):
    """The part of a vLLM-Omni stage model ``forward_model`` relies on."""

    def make_omni_output(self, model_outputs: Any, **kwargs: Any) -> OmniOutput: ...


def _decode_additional_info(info: Any) -> dict[str, Any]:
    if not info:
        return {}
    if isinstance(info, dict):
        return info
    decoded = deserialize_additional_information(info)
    return decoded if isinstance(decoded, dict) else {}


def _slice_inter_stage_payload(
    raw_mm: dict[str, Any],
    req_idx: int,
    tok_start: int | None,
    tok_end: int | None,
) -> dict[str, Any]:
    """Slice a chunk's ``multimodal_outputs`` for one request and copy to CPU.

    Note: ``.detach().cpu()`` is intentionally synchronous here because
    ``OmniConnectorModelRunnerMixin`` serializes ``inter_stage_outputs`` on the
    host immediately after ``_populate_inter_stage_outputs`` returns.
    """
    payload: dict[str, Any] = {}
    for key, val in raw_mm.items():
        if key == "hidden_states" and isinstance(val, dict) and "layers" in val:
            layers = {}
            for lk, lt in val["layers"].items():
                if isinstance(lt, torch.Tensor):
                    sliced = (
                        lt[tok_start:tok_end]
                        if tok_start is not None and tok_end is not None
                        else lt
                    )
                    layers[lk] = sliced.detach().cpu()
                else:
                    layers[lk] = lt
            payload["hidden_states"] = {"layers": layers}
        elif key == "codes" and isinstance(val, dict):
            codes = {}
            for ck, cv in val.items():
                if isinstance(cv, torch.Tensor):
                    sliced = (
                        cv[tok_start:tok_end]
                        if tok_start is not None
                        and tok_end is not None
                        and cv.shape[0] >= tok_end
                        else cv
                    )
                    codes[ck] = sliced.detach().cpu()
                else:
                    codes[ck] = cv
            payload["codes"] = codes
        elif key == "embed" and isinstance(val, dict):
            embed = {}
            for ek, ev in val.items():
                item = ev[req_idx] if isinstance(ev, list) and req_idx < len(ev) else ev
                embed[ek] = (
                    item.detach().cpu() if isinstance(item, torch.Tensor) else item
                )
            payload["embed"] = embed
        elif isinstance(val, torch.Tensor):
            payload[key] = val.detach().cpu()
        else:
            payload[key] = val
    return payload


def _merge_chunk_payload(
    existing: dict[str, Any], new_payload: dict[str, Any]
) -> dict[str, Any]:
    """Concatenate per-token payload tensors when a request spans multiple chunks."""
    merged = dict(existing)
    for key, val in new_payload.items():
        if (
            key == "hidden_states"
            and isinstance(val, dict)
            and "layers" in val
            and isinstance(merged.get("hidden_states"), dict)
            and "layers" in merged["hidden_states"]
        ):
            layers = dict(merged["hidden_states"]["layers"])
            for lk, lt in val["layers"].items():
                prev = layers.get(lk)
                if isinstance(prev, torch.Tensor) and isinstance(lt, torch.Tensor):
                    layers[lk] = torch.cat([prev, lt], dim=0)
                else:
                    layers[lk] = lt
            merged["hidden_states"] = {"layers": layers}
        elif (
            key == "codes"
            and isinstance(val, dict)
            and isinstance(merged.get("codes"), dict)
        ):
            codes = dict(merged["codes"])
            for ck, cv in val.items():
                prev = codes.get(ck)
                if isinstance(prev, torch.Tensor) and isinstance(cv, torch.Tensor):
                    codes[ck] = torch.cat([prev, cv], dim=0)
                else:
                    codes[ck] = cv
            merged["codes"] = codes
        else:
            merged[key] = val
    return merged


class TPUARModelRunner(OmniTPUModelRunner, OmniConnectorModelRunnerMixin):
    """TPU runner for autoregressive stages; counterpart of ``GPUARModelRunner``.

    Serves the Thinker and Talker stages on TPU, wiring inter-stage Omni
    connectors (with ``OmniKVTransferManager``), multimodal mixed-dimension
    chunk slicing, and the Talker's ``preprocess``, ``talker_mtp``,
    ``postprocess``, and ``make_omni_output`` hooks.
    """

    _omni_connector_initialized: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._omni_mm_embeds_list: list[torch.Tensor] | None = None
        self._omni_chunk_req_slice: tuple[int, int] | None = None
        self._omni_chunk_outputs: list[
            tuple[list[tuple[str, int, int, int]], dict[str, Any]]
        ] = []
        self.kv_transfer_manager: OmniKVTransferManager | None = None
        if needs_omni_connector(self.model_config):
            self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(
                self.vllm_config, self.model_config
            )
            self.init_omni_connectors(
                model_config=self.model_config,
                kv_transfer_manager=self.kv_transfer_manager,
            )
        if self.speculative_config is not None:
            raise NotImplementedError(
                f"Stage {self.stage_id} ({self.model_stage!r}): speculative decoding "
                "(speculative_config) is not supported by TPUARModelRunner yet."
            )

    def load_model(self) -> None:
        """Apply Qwen3-Omni stage patches and load the stage model."""
        maybe_patch_qwen3_omni_thinker(self.model_config.architectures)
        maybe_patch_qwen3_omni_talker(self.model_config.architectures)
        super().load_model()

    def _update_states(self, scheduler_output: SchedulerOutput) -> bool:
        """Release connector and buffer state for finished requests."""
        for req_id in scheduler_output.finished_req_ids:
            self.model_intermediate_buffer.pop(req_id, None)
            if self._omni_connector_initialized:
                self.cleanup_finished_request(req_id)
        res = super()._update_states(scheduler_output)
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            existing = self.model_intermediate_buffer.setdefault(req_id, {})
            self._update_intermediate_buffer(
                req_id,
                _decode_additional_info(
                    getattr(new_req, "additional_information", None)
                ),
            )
            self._update_intermediate_buffer(
                req_id,
                _decode_additional_info(
                    getattr(new_req, "model_intermediate_buffer", None)
                ),
            )
            req_state = self.requests.get(req_id)
            if req_state is not None:
                req_state.additional_information_cpu = existing
        cached_reqs = scheduler_output.scheduled_cached_reqs
        cached_infos = getattr(cached_reqs, "additional_information", None)
        if isinstance(cached_infos, dict):
            for req_id in cached_reqs.req_ids:
                self._update_intermediate_buffer(
                    req_id, _decode_additional_info(cached_infos.get(req_id))
                )
        return res

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        start_index: int,
        num_decode_reqs: int,
        num_windowed_reqs: int | None = None,
    ) -> Any:
        res = super()._prepare_inputs(
            scheduler_output, start_index, num_decode_reqs, num_windowed_reqs
        )
        self._omni_chunk_req_slice = (start_index, res[4])
        return res

    def _chunk_req_spans(self) -> list[tuple[str, int, int, int]]:
        if self._omni_chunk_req_slice is None or not hasattr(self, "input_batch"):
            return []
        input_batch = self.input_batch
        query_start_loc = self.query_start_loc_np
        num_reqs = input_batch.num_reqs
        start_idx, end_idx = self._omni_chunk_req_slice
        start_idx = min(start_idx, num_reqs)
        end_idx = min(end_idx, num_reqs)
        spans: list[tuple[str, int, int, int]] = []
        for r_i, req_id in enumerate(input_batch.req_ids[start_idx:end_idx]):
            if req_id is None or r_i + 1 >= len(query_start_loc):
                continue
            spans.append(
                (
                    req_id,
                    start_idx + r_i,
                    int(query_start_loc[r_i]),
                    int(query_start_loc[r_i + 1]),
                )
            )
        return spans

    def _gather_mm_embeddings(
        self,
        scheduler_output: Any,
        target_pad_len: int | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Gather multimodal embeddings without concatenating mixed feature widths."""
        mm_embeds, is_mm_embed = super()._gather_mm_embeddings(
            scheduler_output, target_pad_len
        )
        if not mm_embeds:
            self._omni_mm_embeds_list = None
            return [], is_mm_embed
        self._omni_mm_embeds_list = list(mm_embeds)
        total_rows = sum(int(e.shape[0]) for e in mm_embeds)
        return [torch.arange(total_rows, dtype=torch.long)], is_mm_embed

    def _slice_mm_embeds_by_row_indices(
        self, row_indices: torch.Tensor
    ) -> list[torch.Tensor]:
        """Slice cached ``self._omni_mm_embeds_list`` across ``row_indices``."""
        cached = self._omni_mm_embeds_list or []
        if row_indices.numel() == 0 or not cached:
            return []
        start = int(row_indices[0].item())
        end = int(row_indices[-1].item()) + 1
        result: list[torch.Tensor] = []
        offset = 0
        for emb in cached:
            n = int(emb.shape[0])
            item_start, item_end = offset, offset + n
            offset = item_end
            if item_end <= start or item_start >= end:
                continue
            s = max(0, start - item_start)
            e = min(n, end - item_start)
            sliced = emb[s:e]
            copy_mm_embedding_modality(emb, sliced)
            result.append(sliced)
        if end >= offset:
            self._omni_mm_embeds_list = None
        return result

    def capture_model(self) -> None:
        super().capture_model()
        if (
            self.enforce_eager
            or not self.supports_mm_inputs
            or not self._pp_is_first
            or self.model_stage not in (None, "thinker")
        ):
            return
        self._precompile_mm_inputs()

    @torch.no_grad()
    def _precompile_mm_inputs(self) -> None:
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
                    _, inputs_embeds = super()._get_model_inputs(
                        dummy_input_ids, ([dummy_mm], is_multimodal)
                    )
                    if inputs_embeds is not None:
                        synchronize_tensors(inputs_embeds)
            # Clear any deepstack_input_embeds state left populated by the
            # vision merge warmup above before serving starts.
            dummy_input_ids = torch.zeros(
                (token_buckets[0],), dtype=torch.int32, device=self.device
            )
            _, inputs_embeds = super()._get_model_inputs(dummy_input_ids, None)
            if inputs_embeds is not None:
                synchronize_tensors(inputs_embeds)

    def _get_model_inputs(
        self,
        input_ids: torch.Tensor,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Resolve 1D row-index ranges and run stage ``preprocess`` / ``talker_mtp``."""
        if mm_embed_inputs is not None:
            chunk_embeds, is_mm_embed = mm_embed_inputs
            if (
                len(chunk_embeds) == 1
                and isinstance(chunk_embeds[0], torch.Tensor)
                and chunk_embeds[0].ndim == 1
                and self._omni_mm_embeds_list is not None
            ):
                resolved = self._slice_mm_embeds_by_row_indices(chunk_embeds[0])
                mm_embed_inputs = (resolved, is_mm_embed) if resolved else None
        model_input_ids, inputs_embeds = super()._get_model_inputs(
            input_ids, mm_embed_inputs
        )
        orig_model = _unwrap_model(self.model)
        chunk_spans = self._chunk_req_spans()
        if getattr(orig_model, "has_preprocess", False) and chunk_spans:
            if self._omni_connector_initialized and not getattr(
                self.model_config, "async_chunk", False
            ):
                self._sync_local_stage_payloads()
            preprocess_ids = (
                model_input_ids if model_input_ids is not None else input_ids
            )
            if inputs_embeds is None and preprocess_ids is not None:
                embed_fn = getattr(
                    orig_model,
                    "embed_input_ids",
                    getattr(self.model, "embed_input_ids", None),
                )
                if callable(embed_fn):
                    inputs_embeds = embed_fn(preprocess_ids)
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds.clone()
            num_computed_arr = self.input_batch.num_computed_tokens_cpu
            mtp_fn = getattr(orig_model, "talker_mtp", None)
            for req_id, req_idx, s, e in chunk_spans:
                if s >= e:
                    continue
                req_info = self.model_intermediate_buffer.setdefault(req_id, {})
                num_computed = (
                    int(num_computed_arr[req_idx])
                    if req_idx < len(num_computed_arr)
                    else 0
                )
                req_state = self.requests.get(req_id)
                prompt_token_ids = (
                    getattr(req_state, "prompt_token_ids", None)
                    if req_state is not None
                    else None
                )
                prompt_len = len(prompt_token_ids) if prompt_token_ids else 0
                is_prefill = (
                    (num_computed < prompt_len)
                    if prompt_len > 0
                    else (num_computed == 0)
                )
                req_info["_omni_prompt_len"] = prompt_len
                req_info["_omni_num_computed_tokens"] = num_computed
                req_info["_omni_is_prefill"] = is_prefill
                r_ids = preprocess_ids[s:e] if preprocess_ids is not None else None
                r_embeds = inputs_embeds[s:e] if inputs_embeds is not None else None
                p_ids, p_embeds, update_dict = orig_model.preprocess(
                    input_ids=r_ids, input_embeds=r_embeds, **req_info
                )
                self._update_intermediate_buffer(req_id, update_dict)
                if isinstance(p_embeds, torch.Tensor):
                    if inputs_embeds is None:
                        total_tokens = (
                            preprocess_ids.shape[0] if preprocess_ids is not None else e
                        )
                        inputs_embeds = torch.zeros(
                            (total_tokens, p_embeds.shape[-1]),
                            dtype=p_embeds.dtype,
                            device=p_embeds.device,
                        )
                    p_len = min(e - s, p_embeds.shape[0])
                    inputs_embeds[s : s + p_len] = p_embeds[:p_len].to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    )
                mtp_inputs = req_info.pop("mtp_inputs", None)
                if (
                    not is_prefill
                    and callable(mtp_fn)
                    and mtp_inputs is not None
                    and inputs_embeds is not None
                ):
                    last_talker_hidden, text_step = mtp_inputs
                    mtp_embeds, code_predictor_codes = mtp_fn(
                        p_ids,
                        inputs_embeds[s:e],
                        last_talker_hidden.to(device=inputs_embeds.device),
                        text_step.to(device=inputs_embeds.device),
                    )
                    inputs_embeds[s:e] = mtp_embeds.reshape(-1, inputs_embeds.shape[-1])
                    if isinstance(code_predictor_codes, torch.Tensor):
                        code_predictor_codes = code_predictor_codes.detach().cpu()
                    self._update_intermediate_buffer(
                        req_id, {"codes": {"audio": code_predictor_codes}}
                    )
            if inputs_embeds is not None:
                return None, inputs_embeds
        return model_input_ids, inputs_embeds

    def forward_model(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[torch.Tensor | IntermediateTensors, None]:
        """Run the stage model, execute postprocess hooks, and record chunk outputs.

        Normalises stage outputs through ``model.make_omni_output`` into an
        ``OmniOutput`` and records per-chunk ``multimodal_outputs`` in
        ``_omni_chunk_outputs`` for active execution chunks so
        ``_populate_inter_stage_outputs`` can slice and attach inter-stage
        payloads across single-chunk or multi-chunk steps. Consumes
        ``_omni_chunk_req_slice`` before returning so warmup ``_dummy_run``
        calls never observe stale request spans.
        """
        chunk_spans = self._chunk_req_spans()
        self._omni_chunk_req_slice = None

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

        orig_model = _unwrap_model(self.model)
        has_postprocess = bool(getattr(orig_model, "has_postprocess", False))
        if has_postprocess and chunk_spans and isinstance(out, torch.Tensor):
            for req_id, _req_idx, s, e in chunk_spans:
                req_info = self.model_intermediate_buffer.get(req_id, {})
                post_kwargs = {
                    k: v for k, v in req_info.items() if k != "hidden_states"
                }
                update_dict = orig_model.postprocess(out[s:e], **post_kwargs)
                self._update_intermediate_buffer(req_id, update_dict)

        make_fn = getattr(orig_model, "make_omni_output", None)
        if isinstance(out, OmniOutput):
            omni_out: OmniOutput | None = out
            text_hidden = out.text_hidden_states
        elif callable(make_fn) and (chunk_spans or not has_postprocess):
            make_kwargs: dict[str, Any] = {}
            if chunk_spans and has_postprocess:
                make_kwargs["model_intermediate_buffer"] = [
                    self.model_intermediate_buffer.get(rid, {})
                    for rid, _, _, _ in chunk_spans
                ]
            res = cast(_OmniStageModel, orig_model).make_omni_output(out, **make_kwargs)
            omni_out = res if isinstance(res, OmniOutput) else None
            # Talker's make_omni_output slices talker_hidden[:span_len] to match
            # unpadded audio codes, whereas TPUModelRunner.execute_model indexes
            # the full padded bucket tensor at logits_indices.
            if has_postprocess and isinstance(out, torch.Tensor):
                text_hidden = out
            else:
                text_hidden = (
                    omni_out.text_hidden_states if omni_out is not None else res
                )
        else:
            omni_out = None
            text_hidden = out[0] if isinstance(out, tuple) else out

        if (
            chunk_spans
            and omni_out is not None
            and isinstance(omni_out.multimodal_outputs, dict)
            and omni_out.multimodal_outputs
        ):
            self._omni_chunk_outputs.append((chunk_spans, omni_out.multimodal_outputs))

        return text_hidden, None

    def _populate_inter_stage_outputs(self, real_output: ModelRunnerOutput) -> None:
        chunk_outputs = list(self._omni_chunk_outputs)
        self._omni_chunk_outputs.clear()
        req_ids = real_output.req_ids
        if not chunk_outputs or not req_ids:
            return
        payload_by_req_id: dict[str, dict[str, Any]] = {}
        for chunk_spans, raw_mm in chunk_outputs:
            for local_idx, (req_id, _batch_idx, tok_start, tok_end) in enumerate(
                chunk_spans
            ):
                sliced = _slice_inter_stage_payload(
                    raw_mm, local_idx, tok_start, tok_end
                )
                if req_id in payload_by_req_id:
                    payload_by_req_id[req_id] = _merge_chunk_payload(
                        payload_by_req_id[req_id], sliced
                    )
                else:
                    payload_by_req_id[req_id] = sliced
        if not payload_by_req_id:
            return
        per_req_payloads = [payload_by_req_id.get(rid, {}) for rid in req_ids]
        real_output.inter_stage_outputs = per_req_payloads  # type: ignore[attr-defined]
        if (
            self._omni_connector_initialized
            and self._should_accumulate_full_payload_output()
        ):
            for idx, rid in enumerate(req_ids):
                req_state = self.requests.get(rid)
                if req_state is not None and per_req_payloads[idx]:
                    self.accumulate_full_payload_output(
                        rid, flatten_payload(per_req_payloads[idx]), req_state
                    )

    def _with_omni_connector_output(
        self,
        output: ModelRunnerOutput | AsyncTPUModelRunnerOutput | None,
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput | None:
        """Attach inter-stage payloads and connector signals before ``get_output()``."""
        if output is None:
            return None
        if isinstance(output, AsyncTPUModelRunnerOutput):
            self._populate_inter_stage_outputs(output._model_runner_output)
            if self._omni_connector_initialized:
                output._model_runner_output = self.attach_omni_connector_output(
                    output._model_runner_output
                )
            return output
        self._populate_inter_stage_outputs(output)
        if self._omni_connector_initialized:
            return self.attach_omni_connector_output(output)
        return output

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput | None:
        self._omni_chunk_req_slice = None
        self._omni_chunk_outputs.clear()
        if self._omni_connector_initialized:
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_full_payload_inputs(scheduler_output)
            if getattr(self, "_pending_full_payload_send", None):
                flush_ids = set(getattr(scheduler_output, "finished_req_ids", set()))
                flush_ids.update(
                    {
                        rid
                        for rid in self._pending_full_payload_send
                        if rid not in self.requests
                    }
                )
                if flush_ids:
                    self.flush_full_payload_outputs(flush_ids)
        try:
            return self._with_omni_connector_output(
                super().execute_model(scheduler_output, intermediate_tensors)
            )
        finally:
            self._omni_chunk_req_slice = None

    def sample_tokens(
        self, grammar_output: GrammarOutput | None = None
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        return self._with_omni_connector_output(  # type: ignore[return-value]
            super().sample_tokens(grammar_output)
        )
