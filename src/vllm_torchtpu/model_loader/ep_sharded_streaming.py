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
"""EP-sharded weight streaming for the Run:AI model-streamer load path.

vLLM's ``--enable-ep-weight-filter`` skips non-local expert tensors during
loading, but only in ``DefaultModelLoader`` (local files) and only for
tensor names ending in ``.weight``. Neither restriction works for TPU MoE
serving from object storage:

- ``RunaiModelStreamerLoader`` (the only ``gs://`` weight path) streams the
  FULL checkpoint into anonymous host memory on every rank. For a 1.5 TB
  MoE checkpoint at 8 ranks/host this exceeds host RAM long before the
  load finishes (Ray OOM kill at 95% of a 944 GB host).
- Compressed-tensors MXFP4 checkpoints name their expert weights
  ``.weight_packed`` / ``.weight_scale``, which the upstream filter never
  matches.

The patch below teaches the Run:AI path to build its fetch plan from the
safetensors headers and REQUEST ONLY the byte ranges of tensors this rank
keeps: dense weights, plus the packed weights AND scales of its own
experts. Skipped tensors are never fetched from storage at all.

Divergence from upstream noted: upstream deliberately keeps every expert's
scale tensors because some GPU backends reduce a global activation-scale
max across experts. The TPU compressed-tensors MoE methods consume scales
strictly per local expert (see ``compressed_tensors_moe/utils.py``), and at
896 experts the scales alone are ~85 GB per rank, so they are filtered
here like the packed weights.
"""

import os
from collections.abc import Generator
from typing import Optional

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Suffixes of per-expert tensors that are safe to skip for non-local
# experts. Anything else (dense, shared experts, gate, norms) is fetched by
# every rank.
_EXPERT_WEIGHT_SUFFIXES = (".weight", ".weight_packed", ".weight_scale")


def _should_skip(name: str, local_expert_ids: set[int]) -> bool:
    from vllm.model_executor.model_loader.ep_weight_filter import \
        parse_expert_id

    expert_id = parse_expert_id(name)
    if expert_id is None:
        return False
    if not name.endswith(_EXPERT_WEIGHT_SUFFIXES):
        return False
    return expert_id not in local_expert_ids


def _compute_local_expert_ids() -> Optional[set[int]]:
    """Local expert ids for this rank, or None when filtering is off.

    Mirrors ``DefaultModelLoader._init_ep_weight_filter`` (vLLM 0.26) so the
    fetch-time filter and the FusedMoE expert map agree by construction.
    """
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.model_loader.ep_weight_filter import \
        compute_local_expert_ids

    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    if not (model_config.is_moe and parallel_config.enable_expert_parallel
            and parallel_config.enable_ep_weight_filter):
        return None
    # EPLB redundant slots may need foreign logical experts; do not filter.
    if parallel_config.enable_eplb:
        return None
    num_experts = model_config.get_num_experts()
    if num_experts <= 0:
        return None

    from vllm.distributed import (get_dp_group, get_pcp_group,
                                  get_tensor_model_parallel_rank)
    dp_size = parallel_config.data_parallel_size
    tp_size = parallel_config.tensor_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
    tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
    pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
    ep_size = dp_size * pcp_size * tp_size
    ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

    local_ids = compute_local_expert_ids(
        num_experts,
        ep_size,
        ep_rank,
        placement=parallel_config.expert_placement_strategy,
    )
    if local_ids is not None:
        logger.info(
            "[sharded-ep-load] ep_size=%d ep_rank=%d placement=%s: "
            "this rank keeps %d/%d experts",
            ep_size, ep_rank, parallel_config.expert_placement_strategy,
            len(local_ids), num_experts)
    return local_ids


def _sharded_runai_weights_iterator(
    hf_weights_files: list[str],
    local_expert_ids: set[int],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stream only this rank's tensors from the safetensors files.

    Reads each file's safetensors header (via the Run:AI streamer, so
    ``gs://`` and local paths both work), drops non-local expert tensors
    from the request plan, and streams the surviving byte ranges as
    contiguous runs. The skipped bytes are never requested from storage.
    """
    import runai_model_streamer.safetensors_streamer.safetensors_pytorch as sp
    from runai_model_streamer.file_streamer import FileChunks
    from runai_model_streamer.safetensors_streamer.safetensors_streamer import \
        SafetensorsStreamer
    from tqdm.auto import tqdm
    from vllm.model_executor.model_loader.weight_utils import enable_tqdm

    with SafetensorsStreamer() as streamer:
        fs = streamer.file_streamer
        metas = sp.SafetensorsMetadata.from_files(fs, hf_weights_files, None)

        # Requests are indexed by their position: FileChunks.id == list index,
        # and tensors_by_request[id] lists the metadata of that run's chunks
        # in order.
        requests: list[FileChunks] = []
        tensors_by_request: list[list] = []
        # Kept tensors that occupy zero bytes never enter the fetch plan: the
        # streamer's request iterator silently drops a zero-byte chunk group
        # (and a zero-byte group at the head of its queue aborts the whole
        # remaining stream). They are yielded directly as empty tensors.
        zero_size_tensors: list = []
        kept = skipped = 0
        kept_bytes = skipped_bytes = 0

        run_start: Optional[int] = None
        run_sizes: list[int] = []
        run_tensors: list = []

        def _flush(path: str) -> None:
            nonlocal run_start, run_sizes, run_tensors
            if run_sizes:
                requests.append(
                    FileChunks(len(requests), path, run_start, run_sizes))
                tensors_by_request.append(run_tensors)
            run_start, run_sizes, run_tensors = None, [], []

        for path, meta in zip(hf_weights_files, metas):
            pos = meta.offset
            for tensor_meta, size in zip(meta.tensors_metadata,
                                         meta.read_sizes):
                if _should_skip(tensor_meta.name, local_expert_ids):
                    skipped += 1
                    skipped_bytes += size
                    # A zero-byte skip leaves the byte stream contiguous, so
                    # the current run may continue through it.
                    if size:
                        _flush(path)
                elif size == 0:
                    kept += 1
                    zero_size_tensors.append(tensor_meta)
                else:
                    kept += 1
                    kept_bytes += size
                    if run_start is None:
                        run_start = pos
                    run_sizes.append(size)
                    run_tensors.append(tensor_meta)
                pos += size
            _flush(path)

        logger.info(
            "[sharded-ep-load] fetching %d tensors (%.1f GB) in %d ranges; "
            "skipping %d non-local expert tensors (%.1f GB) before fetch",
            kept, kept_bytes / 1e9, len(requests), skipped,
            skipped_bytes / 1e9)

        fs.stream_files(requests,
                        credentials=None,
                        device="cpu",
                        is_distributed=False)

        # Under torch-tpu's deferred eager mode, weight_loader's device
        # copies queue up and keep their HOST source tensors referenced
        # until the graph flushes. Left alone, every kept tensor's host
        # clone stays resident for the whole load (~all dense bytes per
        # rank; the pod died at 8 ranks x ~110 GB within 2 minutes of
        # streaming). This iterator runs on the same thread that consumes
        # the tensors, so a periodic device sync here releases the queued
        # copies' host sources and bounds resident memory to a window.
        sync_every = int(os.getenv("TPU_SHARDED_LOAD_SYNC_EVERY", "512"))

        def _device_sync() -> None:
            try:
                torch.tpu.synchronize()
            except Exception:
                logger.warning_once(
                    "[sharded-ep-load] torch.tpu.synchronize() unavailable "
                    "during load; host memory may accumulate.")

        progress = tqdm(total=kept,
                        desc="Loading safetensors (EP-sharded Run:AI)",
                        disable=not enable_tqdm(use_tqdm_on_load),
                        mininterval=2)
        try:
            for tensor_meta in zero_size_tensors:
                # create_torch_tensor returns torch.empty for zero-element
                # metadata without touching the buffer.
                yield tensor_meta.name, sp.create_torch_tensor(
                    memoryview(b""), tensor_meta)
                progress.update(1)
            for yielded, (request_id, chunk_index,
                          buffer) in enumerate(fs.get_chunks(), start=1):
                tensor_meta = tensors_by_request[request_id][chunk_index]
                yield tensor_meta.name, sp.create_torch_tensor(
                    buffer, tensor_meta).clone()
                progress.update(1)
                if sync_every and yielded % sync_every == 0:
                    _device_sync()
        finally:
            progress.close()
        if sync_every:
            _device_sync()


def patch_runai_sharded_expert_streaming() -> None:
    """Make ``RunaiModelStreamerLoader`` fetch only this rank's tensors.

    Inert unless ``--enable-expert-parallel`` AND
    ``--enable-ep-weight-filter`` are both set (and EPLB is off); any
    failure to set up the filter falls back to the stock full-checkpoint
    streaming.
    """
    from vllm.model_executor.model_loader import runai_streamer_loader as rsl

    if getattr(rsl.RunaiModelStreamerLoader, "_tpu_sharded_ep_patch", False):
        return
    original_get_iterator = rsl.RunaiModelStreamerLoader._get_weights_iterator

    def _get_weights_iterator(self, model_or_path: str,
                              revision: Optional[str]):
        try:
            local_expert_ids = _compute_local_expert_ids()
        except AttributeError:
            # A vLLM without the EP weight filter (no
            # enable_ep_weight_filter / expert_placement_strategy fields).
            logger.info_once(
                "[sharded-ep-load] this vLLM has no EP weight filter; "
                "using stock full-checkpoint streaming.")
            local_expert_ids = None
        except Exception:
            logger.exception(
                "[sharded-ep-load] filter setup failed; falling back to "
                "full-checkpoint streaming")
            local_expert_ids = None
        if local_expert_ids is None:
            return original_get_iterator(self, model_or_path, revision)
        if getattr(self, "_is_distributed", False):
            # Per-rank fetch plans differ, so the streamer's cross-rank
            # broadcast mode would corrupt the chunk-id maps. Sharded
            # streaming already de-duplicates the heavy reads per rank.
            logger.info_once(
                "[sharded-ep-load] ignoring model_loader_extra_config "
                "'distributed': incompatible with per-rank sharded fetch "
                "plans; streaming per rank instead.")
        hf_weights_files = self._prepare_weights(model_or_path, revision)
        return _sharded_runai_weights_iterator(
            hf_weights_files, local_expert_ids,
            self.load_config.use_tqdm_on_load)

    rsl.RunaiModelStreamerLoader._get_weights_iterator = _get_weights_iterator
    rsl.RunaiModelStreamerLoader._tpu_sharded_ep_patch = True
    logger.info("Applied TPU patch: EP-sharded Run:AI weight streaming "
                "(fetch only local experts when the EP weight filter is on).")
