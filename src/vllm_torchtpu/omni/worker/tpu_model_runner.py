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
"""Base TPU model runner for vLLM-Omni stages."""

import contextlib
from typing import Any, cast

import torch
from vllm_omni.config import OmniModelConfig

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _unwrap_model(model: Any) -> Any:
    """Return the underlying module when ``model`` is a ``torch.compile`` wrapper."""
    return getattr(model, "_orig_mod", model)


class OmniTPUModelRunner(TPUModelRunner):
    """Shared base for every Omni runner on TPU; counterpart of ``OmniGPUModelRunner``.

    Why a base class sits between ``TPUModelRunner`` and the stage runners:
    vLLM-Omni's worker hierarchy has two runner leaves — ``OmniARModelRunner``
    for autoregressive stages (thinker, talker) and
    ``OmniGenerationModelRunner`` for non-AR generation stages (code2wav,
    diffusion, audio decoders) — both inheriting from ``OmniGPUModelRunner``.
    Keeping the same split on TPU gives ``TPUGenerationModelRunner`` a
    clean seam when it is added, and a single place to revisit if the parts
    ``TPUModelRunner`` brings in do not hold for every Omni model (for
    example Qwen2.5-Omni and GLM-Image).

    Records which pipeline stage the runner serves and owns the per-request
    ``model_intermediate_buffer`` state. vLLM-Omni hands each stage an
    ``OmniModelConfig`` carrying ``model_stage`` and ``stage_id``; a plain
    vLLM model config has neither, and the leaf runners treat that like a
    single-stage thinker. Everything stage-specific (connectors, talker
    multi-token prediction, generation outputs) belongs to the leaf classes,
    mirroring ``GPUARModelRunner`` vs ``GPUGenerationModelRunner``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_stage: str | None = None
        self.stage_id: int | None = None
        self.model_intermediate_buffer: dict[str, dict[str, Any]] = {}
        if not hasattr(self, "model"):
            self.model: Any = None
        if not hasattr(self, "requests"):
            self.requests: dict[str, Any] = {}
        if isinstance(self.model_config, OmniModelConfig):
            # The cast is for the type checker only. The full-environment check
            # runs without vllm-omni installed, so there ``OmniModelConfig``
            # resolves to ``Any`` and ``isinstance`` does not narrow
            # ``self.model_config`` away from ``ModelConfig``.
            omni_config = cast(OmniModelConfig, self.model_config)
            self.model_stage = omni_config.model_stage
            self.stage_id = omni_config.stage_id

    def _store_buffer_value(
        self, dest: dict[str, Any], key: str, value: Any, resident_keys: set[str]
    ) -> None:
        if isinstance(value, torch.Tensor):
            t = value.detach()
            dest[key] = (
                t.clone()
                if key in resident_keys
                else (
                    t.cpu().contiguous() if t.device.type != "cpu" else t.contiguous()
                )
            )
        elif isinstance(value, list):
            dest[key] = [
                item.detach().cpu().contiguous()
                if isinstance(item, torch.Tensor)
                else item
                for item in value
            ]
        else:
            dest[key] = value

    def _update_intermediate_buffer(
        self, req_id: str, update_dict: dict[str, Any] | None
    ) -> None:
        if not isinstance(update_dict, dict) or not update_dict:
            return
        orig_model = _unwrap_model(self.model)
        gpu_keys: set[tuple[str, str]] = getattr(
            orig_model, "gpu_resident_buffer_keys", set()
        )
        existing = self.model_intermediate_buffer.setdefault(req_id, {})
        for key, val in update_dict.items():
            if isinstance(val, dict):
                sub = existing.setdefault(key, {})
                sub_gpu_keys = {q for tk, q in gpu_keys if tk == key}
                for qual, sub_val in val.items():
                    self._store_buffer_value(sub, qual, sub_val, sub_gpu_keys)
            else:
                self._store_buffer_value(existing, key, val, set())
        req_state = self.requests.get(req_id)
        if req_state is not None:
            req_state.additional_information_cpu = existing

    def _replace_intermediate_buffer(
        self, req_id: str, update_dict: dict[str, Any] | None
    ) -> None:
        """Replace ``req_id``'s buffer entry while preserving ``meta`` state.

        Called by ``_sync_local_stage_payloads`` (mirroring
        ``OmniGPUModelRunner._replace_intermediate_buffer``).
        """
        if not isinstance(update_dict, dict):
            return
        previous = self.model_intermediate_buffer.get(req_id)
        previous_meta = previous.get("meta") if isinstance(previous, dict) else None
        self.model_intermediate_buffer[req_id] = {}
        self._update_intermediate_buffer(req_id, update_dict)
        if isinstance(previous_meta, dict):
            meta = self.model_intermediate_buffer.setdefault(req_id, {}).setdefault(
                "meta", {}
            )
            for key in ("num_processed_tokens", "resumable"):
                if key not in meta and key in previous_meta:
                    meta[key] = previous_meta[key]

    def _sync_local_stage_payloads(self) -> None:
        """Move received full-payload inputs into ``model_intermediate_buffer``."""
        cache = getattr(self, "_local_stage_payload_cache", None)
        if not cache:
            return
        lock = getattr(self, "_lock", None)
        ctx = lock if lock is not None else contextlib.nullcontext()
        with ctx:
            if not cache:
                return
            active_req_ids = set(self.requests)
            pending = set(
                getattr(self, "_full_payload_pending_broadcast_req_ids", set())
            )
            staged = {
                req_id: cache.pop(req_id)
                for req_id in list(cache.keys())
                if req_id in active_req_ids and req_id not in pending
            }
        for req_id, payload in staged.items():
            self._replace_intermediate_buffer(req_id, payload)
