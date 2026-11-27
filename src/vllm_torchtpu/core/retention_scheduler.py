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
"""The scheduler side of hybrid-state retention (``core/hybrid_retention.py``).

``HybridRetentionScheduler`` is the vLLM ``Scheduler`` base of
``PrefixAwareScheduler``. A subclass that builds a ``HybridStateRetention``
keeps it in ``_retention``, and the base then:

* pins the running requests' snapshots at the start of each step
  (``_retention_begin_step``, which the subclass's ``schedule()`` calls), and
  the snapshots of a finished request whose blocks the KV connector still
  holds (``_connector_finished``);
* hands a freed request's snapshots to the retention, which refreshes them
  and demotes the decode-output blocks behind them (``_free_request_blocks``);
* ends a prefill chunk at the retention's checkpoints as well as at vLLM's
  Mamba block alignment, except where the backend writes the state inside
  the prompt's last chunk (``_retention_split``);
* turns retention off after any of its calls raised, releasing the blocks it
  holds (``_retention_failed``).

The pipeline chunk scheduler keeps the same hooks in its own module; they
match these apart from its log lines.

Without a retention (``_retention`` None, the default) every hook is vLLM's
method.
"""

from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)

from vllm_torchtpu.core.hybrid_retention import HybridStateRetention
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The architectures a scheduler keeps hybrid-state snapshots for: the Qwen3.5
# MoE models, whose runs used them. Other hybrid models are scheduled without
# retention.
RETENTION_ARCHITECTURES = frozenset(
    {"Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"}
)


class HybridRetentionScheduler(Scheduler):
    # The retention the subclass built; None without one and after a
    # retention call raised.
    _retention: HybridStateRetention | None = None
    # How the retention log lines name the scheduler.
    _retention_log_name = "Scheduler"

    def _retention_failed(self) -> None:
        """Turns retention off after one of its calls raised: it releases
        the blocks it holds and its hooks fall through to vLLM's methods."""
        name = self._retention_log_name
        logger.exception("%s: hybrid state retention failed; disabling it.", name)
        retention, self._retention = self._retention, None
        if retention is None:
            return
        try:
            retention.shutdown()
        except Exception:
            # The blocks it still pins stay pinned; scheduling goes on.
            logger.exception(
                "%s: shutting hybrid state retention down failed; its pinned "
                "blocks stay pinned.",
                name,
            )

    def _retention_begin_step(self) -> None:
        """Pins the running requests' snapshots and clears the retention's
        per-step record; called at the start of ``schedule()``."""
        retention = self._retention
        if retention is None:
            return
        try:
            retention.pin_running(self.running)
        except Exception:
            self._retention_failed()
        else:
            retention.begin_step()

    def _connector_finished(self, request: Any):
        result = super()._connector_finished(request)
        if result[0] and self._retention is not None:
            try:
                self._retention.pin_running((request,))
            except Exception:
                self._retention_failed()
        return result

    def _free_request_blocks(self, request: Any):
        retention = self._retention
        if retention is None:
            return super()._free_request_blocks(request)
        try:
            attn_blocks = retention.snapshot_attention_blocks(request)
        except Exception:
            self._retention_failed()
            return super()._free_request_blocks(request)
        result = super()._free_request_blocks(request)
        try:
            retention.on_request_freed(request, attn_blocks)
        except Exception:
            self._retention_failed()
        return result

    def _retention_split(
        self,
        request: Any,
        start: int,
        num_new_tokens: int,
        num_new_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> int:
        """Ends the chunk where the GDN state must be written: at the base
        scheduler's block alignment and the retention stops. A stop where
        the backend writes the state inside the chunk (vLLM's prefill
        checkpoint blocks, ``_in_chunk_checkpoint``) does not end it."""
        retention = self._retention
        if retention is not None:
            try:
                retention.prepare_prefill_split(request, start, num_new_tokens)
            except Exception:
                self._retention_failed()
                retention = None
        aligned = super()._mamba_block_aligned_split(
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        if retention is not None:
            written = self._in_chunk_checkpoint(request, start, aligned)
            try:
                if written:
                    end = start + aligned
                    stops = [
                        s
                        for s in retention.prefill_checkpoint_stops(request, start)
                        if s < end and s != written
                    ]
                    if stops:
                        aligned = min(stops) - start
                else:
                    aligned = retention.adjust_prefill_split(request, start, aligned)
            except Exception:
                self._retention_failed()
        return aligned

    def _in_chunk_checkpoint(self, request: Any, start: int, num_tokens: int) -> int:
        """The position where the backend writes the GDN state inside the
        prefill chunk of ``num_tokens`` from ``start``, else 0.

        With prefill checkpoint blocks in every Mamba group, the chunk that
        ends the prefill also exports the state at the prompt's last
        cacheable boundary when the chunk can hold it; vLLM's split then
        lets the chunk run past that boundary. Same condition as the split's
        and the Mamba manager's.
        """
        if not self.mamba_has_prefill_checkpoint_blocks:
            return 0
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        end = start + num_tokens
        if end < prefill_end:
            return 0
        position = get_mamba_prefill_checkpoint_position(
            prefill_end,
            self.hash_block_size if self.mamba_partial_cache_hit else self.block_size,
            drop_eagle_block=self.use_eagle_block_drop,
            num_hash_lookahead_tokens=self.num_hash_lookahead_tokens,
        )
        valid = is_mamba_prefill_checkpoint_valid(
            query_start=start,
            query_end=end,
            checkpoint_position=position,
            hash_block_size=self.hash_block_size,
            mamba_block_size=self.cache_config.block_size,
            checkpoint_alignment=self.mamba_prefill_checkpoint_alignment,
        )
        return position if valid else 0
