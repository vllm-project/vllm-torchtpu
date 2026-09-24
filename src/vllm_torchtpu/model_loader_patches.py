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
"""EP-sharded weight filtering for the Run:AI streamer and DefaultModelLoader.

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

The Run:AI patch below teaches the Run:AI path to build its fetch plan from
the safetensors headers and REQUEST ONLY the byte ranges of tensors this
rank keeps: dense weights, plus the packed weights AND scales of its own
experts. Skipped tensors are never fetched from storage at all.

The DefaultModelLoader patch below fixes the matching gap in vLLM 0.27.0's
local-files path: ``should_skip_weight`` only matches names ending in
``.weight``, so for an MXFP4 checkpoint it keeps every expert's
``.weight_packed`` and ``.weight_scale`` and the filter is a no-op for all
per-expert tensors. The patch replaces the predicate with one whose suffix
allowlist also covers ``.weight_packed`` and ``.weight_scale``.

Divergence from upstream noted: upstream deliberately keeps every expert's
scale tensors because some GPU backends reduce a global activation-scale
max across experts. The TPU compressed-tensors MoE methods consume scales
strictly per local expert (see ``compressed_tensors_moe/utils.py``), and at
896 experts the scales alone are ~85 GB per rank, so they are filtered
here like the packed weights. Per-expert activation ``.input_scale``
tensors (ModelOpt NVFP4, needed globally by FlashInfer backends) are not
in the allowlist and stay unfiltered.
"""

import functools
import gc
import os
from collections.abc import Generator

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Suffixes of per-expert tensors that are safe to skip for non-local
# experts. Anything else (dense, shared experts, gate, norms) is fetched by
# every rank.
_EXPERT_WEIGHT_SUFFIXES = (".weight", ".weight_packed", ".weight_scale")


def _should_skip(name: str, local_expert_ids: set[int]) -> bool:
    from vllm.model_executor.model_loader.ep_weight_filter import parse_expert_id

    expert_id = parse_expert_id(name)
    if expert_id is None:
        return False
    if not name.endswith(_EXPERT_WEIGHT_SUFFIXES):
        return False
    return expert_id not in local_expert_ids


def _should_skip_weight_tpu(name: str, local_expert_ids: set[int] | None) -> bool:
    """Drop-in replacement for vLLM's ``should_skip_weight``.

    Same contract as upstream (``local_expert_ids=None`` means no
    filtering), but the suffix allowlist also covers the compressed-tensors
    MXFP4 ``.weight_packed`` and ``.weight_scale`` names, so non-local
    experts are skipped before their bytes are read.
    """
    if local_expert_ids is None:
        return False
    return _should_skip(name, local_expert_ids)


def _compute_local_expert_ids() -> set[int] | None:
    """Local expert ids for this rank, or None when filtering is off.

    Mirrors ``DefaultModelLoader._init_ep_weight_filter`` (vLLM 0.26) so the
    fetch-time filter and the FusedMoE expert map agree by construction.
    """
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.model_loader import default_loader

    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    if not (
        model_config.is_moe
        and parallel_config.enable_expert_parallel
        and parallel_config.enable_ep_weight_filter
    ):
        return None
    # EPLB redundant slots may need foreign logical experts; do not filter.
    if parallel_config.enable_eplb:
        return None
    num_experts = model_config.get_num_experts()
    if num_experts <= 0:
        return None

    from vllm.distributed import (
        get_dp_group,
        get_pcp_group,
        get_tensor_model_parallel_rank,
    )

    dp_size = parallel_config.data_parallel_size
    tp_size = parallel_config.tensor_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
    tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
    pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
    ep_size = dp_size * pcp_size * tp_size
    ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

    # Use the loader's symbol instead of importing the base helper directly.
    # Hierarchical EP replaces this symbol with its chip-aware ownership map,
    # and the streaming path must fetch the same replicated expert set.
    local_ids = default_loader.compute_local_expert_ids(
        num_experts,
        ep_size,
        ep_rank,
        placement=parallel_config.expert_placement_strategy,
    )
    if local_ids is not None:
        logger.info(
            "[sharded-ep-load] ep_size=%d ep_rank=%d placement=%s: "
            "this rank keeps %d/%d experts",
            ep_size,
            ep_rank,
            parallel_config.expert_placement_strategy,
            len(local_ids),
            num_experts,
        )
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
    from runai_model_streamer.safetensors_streamer.safetensors_streamer import (
        SafetensorsStreamer,
    )
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

        run_start: int | None = None
        run_sizes: list[int] = []
        run_tensors: list = []

        def _flush(path: str) -> None:
            nonlocal run_start, run_sizes, run_tensors
            if run_sizes:
                requests.append(FileChunks(len(requests), path, run_start, run_sizes))
                tensors_by_request.append(run_tensors)
            run_start, run_sizes, run_tensors = None, [], []

        for path, meta in zip(hf_weights_files, metas):
            pos = meta.offset
            for tensor_meta, size in zip(meta.tensors_metadata, meta.read_sizes):
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
            kept,
            kept_bytes / 1e9,
            len(requests),
            skipped,
            skipped_bytes / 1e9,
        )

        fs.stream_files(requests, credentials=None, device="cpu", is_distributed=False)

        # Under torch-tpu's deferred eager mode, weight_loader's device
        # copies queue up and keep their HOST source tensors referenced
        # until the graph flushes. Left alone, every kept tensor's host
        # clone stays resident for the whole load (~all dense bytes per
        # rank; the pod died at 8 ranks x ~110 GB within 2 minutes of
        # streaming). This iterator runs on the same thread that consumes
        # the tensors, so a periodic device sync here releases the queued
        # copies' host sources and bounds resident memory to a window.
        sync_every = envs.TPU_SHARDED_LOAD_SYNC_EVERY

        def _device_sync() -> None:
            try:
                torch.accelerator.synchronize()
            except Exception:
                logger.warning_once(
                    "[sharded-ep-load] torch.accelerator.synchronize() unavailable "
                    "during load; host memory may accumulate."
                )

        progress = tqdm(
            total=kept,
            desc="Loading safetensors (EP-sharded Run:AI)",
            disable=not enable_tqdm(use_tqdm_on_load),
            mininterval=2,
        )
        try:
            for tensor_meta in zero_size_tensors:
                # create_torch_tensor returns torch.empty for zero-element
                # metadata without touching the buffer.
                yield (
                    tensor_meta.name,
                    sp.create_torch_tensor(memoryview(b""), tensor_meta),
                )
                progress.update(1)
            for yielded, (request_id, chunk_index, buffer) in enumerate(
                fs.get_chunks(), start=1
            ):
                tensor_meta = tensors_by_request[request_id][chunk_index]
                yield (
                    tensor_meta.name,
                    sp.create_torch_tensor(buffer, tensor_meta).clone(),
                )
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

    def _get_weights_iterator(self, model_or_path: str, revision: str | None):
        try:
            local_expert_ids = _compute_local_expert_ids()
        except AttributeError:
            # A vLLM without the EP weight filter (no
            # enable_ep_weight_filter / expert_placement_strategy fields).
            logger.info_once(
                "[sharded-ep-load] this vLLM has no EP weight filter; "
                "using stock full-checkpoint streaming."
            )
            local_expert_ids = None
        except Exception:
            logger.exception(
                "[sharded-ep-load] filter setup failed; falling back to "
                "full-checkpoint streaming"
            )
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
                "plans; streaming per rank instead."
            )
        hf_weights_files = self._prepare_weights(model_or_path, revision)
        return _sharded_runai_weights_iterator(
            hf_weights_files, local_expert_ids, self.load_config.use_tqdm_on_load
        )

    rsl.RunaiModelStreamerLoader._get_weights_iterator = _get_weights_iterator
    rsl.RunaiModelStreamerLoader._tpu_sharded_ep_patch = True
    logger.info(
        "Applied TPU patch: EP-sharded Run:AI weight streaming "
        "(fetch only local experts when the EP weight filter is on)."
    )


def patch_default_loader_ep_weight_filter() -> None:
    """Widen ``DefaultModelLoader``'s EP weight filter to MXFP4 names.

    vLLM 0.27.0's ``should_skip_weight`` only skips names ending in
    ``.weight``; a compressed-tensors MXFP4 checkpoint stores per-expert
    tensors as ``.weight_packed``/``.weight_scale``, so the stock filter
    reads all 896 experts on every rank. ``weight_utils`` imports the
    predicate by value, so both module references are replaced. Inert when
    the EP weight filter is off: ``safetensors_weights_iterator`` then
    passes ``local_expert_ids=None`` and the predicate keeps everything.
    """
    from vllm.model_executor.model_loader import ep_weight_filter, weight_utils

    if getattr(weight_utils, "_tpu_ep_filter_patch", False):
        return
    weight_utils.should_skip_weight = _should_skip_weight_tpu
    ep_weight_filter.should_skip_weight = _should_skip_weight_tpu
    weight_utils._tpu_ep_filter_patch = True
    logger.info(
        "Applied TPU patch: EP weight filter covers "
        ".weight_packed/.weight_scale (DefaultModelLoader)."
    )


def _evict_checkpoint_page_cache(files: list[str]) -> None:
    """Drop clean host page cache for weight files via posix_fadvise."""
    evicted_count = 0
    evicted_gb = 0.0
    for fpath in files:
        if os.path.isfile(fpath):
            try:
                file_size_gb = os.path.getsize(fpath) / (1024**3)
                with open(fpath, "rb") as f:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                evicted_gb += file_size_gb
                evicted_count += 1
                logger.info(
                    "[tpu-model-loader] Evicted %s (%.2f GB) from host page cache.",
                    fpath,
                    file_size_gb,
                )
            except Exception as e:
                logger.warning(
                    "[tpu-model-loader] Failed to evict page cache for %s: %s",
                    fpath,
                    e,
                )
    gc.collect()
    logger.info(
        "[tpu-model-loader] Evicted %d weight files (%.2f GB) from host page cache.",
        evicted_count,
        evicted_gb,
    )


def patch_default_model_loader_page_cache() -> None:
    """Evict host page cache for weight files once loaded into TPU HBM.

    Reading large models from local disk leaves significant clean page cache
    in host memory. Evicting via posix_fadvise(DONTNEED) drops this
    immediately after weights are loaded into TPU HBM, freeing host memory
    for runtime allocations such as large KV cache offloading.
    Guarded by TPU_EVICT_WEIGHTS_PAGE_CACHE (default: False).
    """
    if not envs.TPU_EVICT_WEIGHTS_PAGE_CACHE:
        return

    from vllm.config import ModelConfig
    from vllm.model_executor.model_loader import default_loader as dl

    if getattr(dl.DefaultModelLoader, "_tpu_cache_evict_patch", False):
        return

    original_prepare_weights = dl.DefaultModelLoader._prepare_weights
    original_load_weights = dl.DefaultModelLoader.load_weights

    def _prepare_weights(self, *args, **kwargs):
        res = original_prepare_weights(self, *args, **kwargs)
        if isinstance(res, tuple) and len(res) >= 2:
            self._loaded_weight_files = res[1]
        return res

    def load_weights(self, model: torch.nn.Module, model_config: ModelConfig) -> None:
        original_load_weights(self, model, model_config)

        if not envs.TPU_EVICT_WEIGHTS_PAGE_CACHE:
            return

        # Host page cache is shared across ranks; only local rank 0 issues the eviction
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        if local_rank == 0 and hasattr(self, "_loaded_weight_files"):
            try:
                _evict_checkpoint_page_cache(self._loaded_weight_files)
            except Exception as e:
                logger.warning("[tpu-model-loader] Page cache eviction skipped: %s", e)

    dl.DefaultModelLoader._prepare_weights = _prepare_weights
    dl.DefaultModelLoader.load_weights = load_weights
    dl.DefaultModelLoader._tpu_cache_evict_patch = True
    logger.info(
        "Applied TPU patch: DefaultModelLoader page cache eviction on load completion."
    )


# Shards every expert of a parameter receives, keyed by the shard written.
_SHARDS_PER_EXPERT = {"w1": 2, "w3": 2, "w2": 1}


class ExpertWriteTracker:
    """Tells when every expert of a parameter has been written.

    A w13 parameter receives a ``w1`` and a ``w3`` shard per expert, a w2
    parameter one ``w2`` shard. The shard being written says which kind the
    parameter is, so the count of writes that completes it is known from
    the first one, whatever order the shards arrive in. Writes with any
    other shard id (a loader that copies all experts at once passes None)
    and experts the loader never writes leave the parameter incomplete.
    """

    def __init__(self) -> None:
        self._seen: dict[int, set[tuple[int, str]]] = {}

    def record(self, param: torch.Tensor, expert_id: int, shard_id: str) -> bool:
        shards = _SHARDS_PER_EXPERT.get(shard_id)
        if shards is None:
            return False
        seen = self._seen.setdefault(id(param), set())
        seen.add((expert_id, shard_id))
        complete = len(seen) == param.shape[0] * shards
        if complete:
            del self._seen[id(param)]
        return complete


# Host bytes per device upload of a staged parameter.
_UPLOAD_CHUNK_BYTES = 256 << 20
# Parameters staged at once before the oldest is written early.
_MAX_STAGED = 32


def _rss_gib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


class ExpertParamStager:
    """Host stand-ins for expert parameters while their experts stream in.

    Loaders write each expert into the stand-in; the device parameter gets
    one upload when its last expert lands, or at the end of loading for
    parameters the loader never completes. Stand-in buffers are reused
    across parameters of the same shape, so host memory stays at a few
    parameters' worth however the runtime tracks upload sources.
    """

    def __init__(self) -> None:
        self._staged: dict[int, tuple[torch.nn.Parameter, torch.Tensor]] = {}
        self._pool: dict[tuple, list[torch.Tensor]] = {}
        self._tracker = ExpertWriteTracker()
        self.written_early: set[int] = set()
        self.flushed = 0

    def staging(self, param: torch.nn.Parameter) -> bool:
        """Whether writes to ``param`` still go through a stand-in.

        A parameter written early keeps its later experts on the direct
        device path, so the early write is never overwritten.
        """
        return id(param) not in self.written_early

    def host_param(self, param: torch.nn.Parameter) -> torch.Tensor:
        entry = self._staged.get(id(param))
        if entry is None:
            if len(self._staged) >= _MAX_STAGED:
                oldest = next(iter(self._staged.values()))[0]
                self.written_early.add(id(oldest))
                self.flush(oldest)
            key = (tuple(param.shape), param.dtype)
            free = self._pool.get(key)
            buffer = (
                free.pop()
                if free
                else torch.empty(param.shape, dtype=param.dtype, device="cpu")
            )
            # The stand-in keeps the parameter's class and attributes so the
            # loader's dispatch on them (is_transposed, quant flags) holds.
            host = torch.Tensor._make_subclass(type(param), buffer, False)
            host.__dict__.update(param.__dict__)
            entry = (param, host)
            self._staged[id(param)] = entry
        return entry[1]

    def written(self, param: torch.nn.Parameter, expert_id: int, shard_id: str) -> bool:
        """Note one expert write; flush the parameter if it is complete."""
        if not self._tracker.record(param, expert_id, shard_id):
            return False
        self.flush(param)
        return True

    def flush(self, param: torch.nn.Parameter) -> None:
        """Upload ``param``'s stand-in to the device in slices and join them.

        The slices bound the source copies the runtime keeps per upload. One
        concatenation joins them rather than slice writes into a
        preallocated parameter: on TPU an in-place slice write is a program
        that rewrites the whole parameter, once per slice.
        """
        entry = self._staged.pop(id(param), None)
        if entry is None:
            return
        param, host = entry
        device = param.data.device
        per_row = max(1, host[0].numel() * host.element_size())
        step = max(1, _UPLOAD_CHUNK_BYTES // per_row)
        chunks = [
            host[start : start + step].to(device)
            for start in range(0, host.shape[0], step)
        ]
        param.data = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        del chunks
        if device.type == "tpu":
            from vllm_torchtpu.utils import synchronize_tensors

            synchronize_tensors([param.data], wait=True)
        self.flushed += 1
        buffer = torch.Tensor._make_subclass(torch.Tensor, host.data, False)
        self._pool.setdefault((tuple(host.shape), host.dtype), []).append(buffer)
        logger.debug(
            "Staged expert parameter %d written: shape=%s staged=%d "
            "pooled=%d rss=%.1f GiB",
            self.flushed,
            tuple(param.shape),
            len(self._staged),
            sum(len(v) for v in self._pool.values()),
            _rss_gib(),
        )

    def flush_all(self) -> None:
        for param, _ in list(self._staged.values()):
            self.flush(param)

    def __len__(self) -> int:
        return len(self._staged)


def patch_moe_expert_write_staging() -> None:
    """Load expert parameters through host stand-ins written to the device
    once each.

    Every per-expert copy into a device parameter uploads its source and runs
    as its own program that rewrites the whole parameter; a stage holding
    whole experts (no expert parallelism) spends tens of minutes on those
    programs and holds its expert weights twice by the end of loading.
    A stand-in is written when its last expert lands, or at the end of
    loading if the loader never completes it, in slices of
    ``_UPLOAD_CHUNK_BYTES`` joined on the device. Only a parameter written
    early, to keep at most ``_MAX_STAGED`` stand-ins alive, gets further
    writes: its remaining experts go through the direct path. Experts the
    loader never writes (padding experts) stay uninitialized, as they would
    on the device.
    """
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.model_loader import base_loader

    if getattr(RoutedExperts, "_tpu_expert_staging_patch", False):
        return
    stager = ExpertParamStager()
    orig_weight_loader = RoutedExperts.weight_loader
    orig_process = base_loader.process_weights_after_loading
    calls = [0]

    # wraps keeps the attributes model loaders read off the upstream
    # loader, such as ``supports_moe_loading``.
    @functools.wraps(orig_weight_loader)
    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        # Experts of other ranks and parameters already written early take
        # the direct path, which skips or writes them as before.
        if (
            param.data.device.type != "tpu"
            or not stager.staging(param)
            or self._map_global_expert_id_to_local_expert_id(expert_id) == -1
        ):
            return orig_weight_loader(
                self,
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success=return_success,
            )
        result = orig_weight_loader(
            self,
            stager.host_param(param),
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success=return_success,
        )
        if result is not False:
            stager.written(param, expert_id, shard_id)
        calls[0] += 1
        if calls[0] % 1000 == 0:
            logger.debug(
                "Expert writes: %d, parameters staged: %d, rss=%.1f GiB",
                calls[0],
                len(stager),
                _rss_gib(),
            )
        return result

    def process_weights_after_loading(model, model_config, target_device):
        during_load = stager.flushed
        stager.flush_all()
        logger.info(
            "Expert parameters written to the device: %d as their last "
            "expert landed, %d at the end of loading.",
            during_load,
            stager.flushed - during_load,
        )
        return orig_process(model, model_config, target_device)

    RoutedExperts.weight_loader = weight_loader
    RoutedExperts._tpu_expert_staging_patch = True
    base_loader.process_weights_after_loading = process_weights_after_loading
    logger.info(
        "Applied TPU patch: expert parameters are staged on the host "
        "and written to the device once each."
    )
