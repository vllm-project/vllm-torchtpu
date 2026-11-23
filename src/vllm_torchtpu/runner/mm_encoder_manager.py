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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    SupportsEncoderCudaGraph,
    supports_encoder_cudagraph,
)
from vllm.utils.jsontree import json_map_leaves
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.encoder_cudagraph import (
    CaptureAxisKeys,
    EncoderCudaGraphManager,
)

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.mm_host_copy_patch import copy_bytes_
from vllm_torchtpu.platforms.tpu_platform import _get_linear_token_paddings
from vllm_torchtpu.runner.mm_encoder_rows import split_batch_output
from vllm_torchtpu.runner.mm_video_chunking import (
    embed_multimodal_chunked,
    get_video_chunk_patch_size,
)
from vllm_torchtpu.utils import align_to, synchronize_tensors

logger = init_logger(__name__)


# The optional model hook that builds a budget graph's inputs at the budget's
# template shapes (see `MMEncoderManager._run_budget_graph`).
BUDGET_INPUTS_HOOK = "prepare_encoder_budget_inputs"
# The optional model hook that ends an embedding merge run without a forward
# (warm-up, `TPUModelRunner._precompile_layout_free_embeds`): it handles what
# the merge staged for the forward as the forward would, then drops it.
MERGE_WARMUP_HOOK = "finish_warmup_merge"
_UNSET = object()


def _to_device(mm_kwargs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Host `mm_kwargs` on `device`, with the per-item pixel slices of a host
    selection (`_select_encoder_items`) joined as upstream's selection joins
    them."""
    joined = {
        key: torch.cat(value)
        if isinstance(value, list) and value and isinstance(value[0], torch.Tensor)
        else value
        for key, value in mm_kwargs.items()
    }
    return json_map_leaves(
        lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, joined
    )


# Host rows reach the device as ROW_CHUNKS chunks of 1/ROW_CHUNKS of a budget's
# rows (`rows_to_device`), so no device program depends on the row count. Each
# transfer holds up the H2D copies after it, so the fewest chunks win; 2 cover
# without a host copy any batch of at least half its template's rows, as an
# image batch past the next smaller budget is.
ROW_CHUNKS = 2


def rows_to_device(
    parts: Sequence[torch.Tensor],
    template: torch.Tensor,
    upload: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """The concatenated host rows of `parts`, zero-padded to `template`'s shape
    and dtype, on its device, uploaded by `upload` (see `MMEncoderManager.upload`).

    The span of host memory holding the parts uploads as ROW_CHUNKS views,
    spread evenly over it (overlapping when it is shorter than the budget),
    and a gather by a host-built index picks each part's rows in order, and
    the padding rows from a zero row. The device ops then have the budget's
    shapes only, and fuse with the budget-inputs hook's into one program per
    budget (compiled by `MMEncoderManager._capture_budget_graph`).
    """
    rows, rest = template.shape[0], tuple(template.shape[1:])
    size = -(-rows // ROW_CHUNKS)
    span, offsets = _span(parts, rows)
    if span.shape[0] < size:
        # Shorter than one chunk (the smallest budgets, or a template sized for
        # videos): pad it on the host.
        span = torch.cat([span, span.new_zeros(size - span.shape[0], *rest)])
    last = span.shape[0] - size
    starts = np.linspace(0, last, ROW_CHUNKS).round().astype(np.int64)
    chunks = [upload(span[start : start + size]) for start in starts]
    at = np.concatenate(
        [np.arange(o, o + p.shape[0]) for o, p in zip(offsets, parts, strict=True)]
    )
    chunk = np.searchsorted(starts, at, side="right") - 1
    index = np.full(rows, ROW_CHUNKS * size, dtype=np.int64)  # the zero row
    index[: len(at)] = chunk * size + at - starts[chunk]
    source = torch.cat([*chunks, chunks[0].new_zeros(1, *rest)])
    return source[upload(torch.from_numpy(index))].to(template.dtype)


def _span(parts: Sequence[torch.Tensor], rows: int) -> tuple[torch.Tensor, list[int]]:
    """A host tensor holding every part, and each part's first row in it: a
    view of the parts' own storage when they are row slices of one tensor
    within `rows` rows of each other, else a concatenation."""
    first = parts[0]
    row = first[0].numel() if first.shape[0] else 0
    if row and all(
        p.is_contiguous()
        and p.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
        and (p.storage_offset() - first.storage_offset()) % row == 0
        for p in parts
    ):
        base = min(p.storage_offset() for p in parts)
        end = max(p.storage_offset() + p.numel() for p in parts)
        if end - base <= rows * row:
            shape = ((end - base) // row, *first.shape[1:])
            span = first.as_strided(shape, first.stride(), base)
            return span, [(p.storage_offset() - base) // row for p in parts]
    starts = np.cumsum([0] + [p.shape[0] for p in parts[:-1]])
    out = first.new_empty((int(starts[-1]) + parts[-1].shape[0], *first.shape[1:]))
    flat = out.view(-1).view(torch.uint8)
    pairs, at = [], 0
    for part in parts:
        src = part.contiguous().view(-1).view(torch.uint8)
        pairs.append((flat[at : at + src.numel()], src))
        at += src.numel()
    copy_bytes_(pairs)
    return out, [int(s) for s in starts]


@contextmanager
def host_encoder_batches(runner: Any) -> Iterator[None]:
    """Hand the encoder manager this step's batches on the host.

    Its model's budget-inputs hook (`BUDGET_INPUTS_HOOK`) then returns each
    budget graph's pixels as host rows, which reach the device in chunks of
    the budget (`rows_to_device`), so no device program takes an item's patch
    count. A step with a modality the manager does not encode keeps the
    upload; so does a manager without a layout-free path (see `layout_free`).
    """
    manager = runner.encoder_cudagraph_manager
    if not isinstance(manager, MMEncoderManager) or not manager.layout_free:
        yield
        return
    upstream = gpu_model_runner.group_and_batch_mm_kwargs

    def group_and_batch(mm_kwargs, *, device=None, pin_memory=False):
        if all(manager.supports_modality(modality) for modality, _ in mm_kwargs):
            return upstream(mm_kwargs)
        return upstream(mm_kwargs, device=device, pin_memory=pin_memory)

    gpu_model_runner.group_and_batch_mm_kwargs = group_and_batch
    manager.host_batches = True
    try:
        yield
    finally:
        gpu_model_runner.group_and_batch_mm_kwargs = upstream
        manager.host_batches = False


def _pack_encoder_items(
    tokens: list[int], budgets: list[int], max_items: int
) -> list[tuple[list[int], int]]:
    batches: list[tuple[list[int], int]] = []
    indices: list[int] = []
    total = 0
    for index in sorted(range(len(tokens)), key=tokens.__getitem__):
        count = tokens[index]
        if not 0 < count <= budgets[-1]:
            raise ValueError(
                f"Coordinated encoding requires each image to fit "
                f"an encoder graph bucket; got {count} tokens, maximum {budgets[-1]}"
            )
        if indices and (len(indices) == max_items or total + count > budgets[-1]):
            batches.append((indices, next(b for b in budgets if b >= total)))
            indices, total = [], 0
        indices.append(index)
        total += count
    if indices:
        batches.append((indices, next(b for b in budgets if b >= total)))
    return batches


class MMEncoderManager(EncoderCudaGraphManager):
    """Per-budget PyTorch XLA-cache manager for the vision encoder forward.

    Vision encoders (e.g. ViTs in Qwen2.5-VL / Qwen3-VL) produce variable
    token counts based on dynamic image resolutions and aspect ratios.
    On TPU, dynamic tensor shapes trigger expensive XLA JIT recompilations
    on every distinct image size. While GPU vLLM uses CUDA Graphs with fixed
    replay buffers to avoid launch overhead, TPU requires static XLA
    compilation. ``MMEncoderManager`` solves this by mapping dynamic
    multimodal inputs into a discrete set of static token budgets
    (e.g., 256, 512, 1024, 2048), zero-padding inputs to match pre-compiled templates.

    How it is used:
        1. Initialization: Created by ``maybe_create_mm_encoder_manager`` during
           runner setup. It pre-allocates dummy CPU input templates for each token
           budget and compiles the model's ``encoder_cudagraph_forward`` via
           ``torch.compile(backend="tpu", fullgraph=True, dynamic=False)``.
        2. Warmup: During ``capture_model()``, ``precompile_vision_encoder()``
           runs a dummy forward pass for every budget in ``self.token_budgets``
           to prime the XLA binary cache before serving traffic begins.
        3. Runtime: During request execution, incoming multimodal items are
           greedily packed into the smallest fitting token budget. Dynamic inputs
           are zero-padded to the static budget template via ``_pad_to_template``,
           executed through the compiled XLA graph via ``_run_budget_graph``,
           and the resulting embeddings are cached for the decoder.
    """

    @staticmethod
    def _generate_budgets(min_budget: int, max_budget: int) -> list[int]:
        """Upstream's power-of-two budgets, plus the prompt's linear token
        buckets under ``TPU_TOKEN_BUCKET_LINEAR``: an encoder pass costs per
        patch as a prompt pass costs per token, and doubling pads it by up to
        half (a 23364-patch request ran the 32768-patch program)."""
        budgets = EncoderCudaGraphManager._generate_budgets(min_budget, max_budget)
        if not envs.TPU_TOKEN_BUCKET_LINEAR:
            return budgets
        linear = _get_linear_token_paddings(max_budget, max_budget)
        return sorted({*budgets, *(b for b in linear if b >= min_budget)})

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        vllm_model: torch.nn.Module,
    ) -> None:
        """Initialize the MMEncoderManager with budget templates and XLA cache.

        Args:
            vllm_config: The global vLLM configuration.
            device: Target torch device (TPU) where operations run.
            vllm_model: The underlying multimodal model module.
        """
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            dtype=cast(torch.dtype, vllm_config.model_config.dtype),
            # maybe_create_mm_encoder_manager checks supports_encoder_cudagraph.
            model=cast(SupportsEncoderCudaGraph, vllm_model),
        )
        if self._capture_axes:
            raise ValueError("TPU encoder budget templates do not support capture axes")

        encoder_use_dp = getattr(vllm_model, "use_data_parallel", None)
        if isinstance(encoder_use_dp, bool):
            tp_size = vllm_config.parallel_config.tensor_parallel_size
            self.use_dp = encoder_use_dp and tp_size > 1
        elif encoder_use_dp is not None:
            logger.warning("Ignoring non-boolean use_data_parallel=%r", encoder_use_dp)

        # Create dummy input templates for every budget size and path directly
        # from self.path_token_budgets prepared by EncoderCudaGraphManager.
        # Built without autograd and detached: some templates derive from or
        # return parameters directly (e.g. Kimi-K3's `pos_embeds`), and a
        # template that requires grad is a different Dynamo guard than the
        # `no_grad` replay inputs, so replays that bypass the template would
        # recompile every budget.
        with torch.no_grad():
            self.budget_templates: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
            for path, budgets in self.path_token_budgets.items():
                templates: dict[int, dict[str, torch.Tensor]] = {}
                for budget in budgets:
                    if budget <= 0:
                        continue
                    capture = self.model.prepare_encoder_cudagraph_capture_inputs(
                        budget,
                        self.max_batch_size,
                        self.max_frames_per_batch,
                        self.device,
                        self.dtype,
                        path=path,
                    )
                    templates[budget] = {
                        key: (
                            val.detach()
                            if isinstance(val, torch.Tensor) and val.requires_grad
                            else val
                        )
                        for key, val in capture.values.items()
                    }
                self.budget_templates[path] = templates
        # Whether the model builds its inputs at each budget's shapes.
        self.has_budget_inputs_hook = callable(
            getattr(type(self.model), BUDGET_INPUTS_HOOK, None)
        )
        # How the hook's host tensors reach the device. TorchTPU holds no
        # reference to a non-blocking upload's source, so this blocks unless
        # the runner installs its input staging (`stage_h2d_upload`), which
        # keeps the sources until its next fence.
        self.upload: Callable[[torch.Tensor], torch.Tensor] = self._upload_blocking
        # The layout-free path (`enable_layout_free`): encoder outputs as row
        # handles into their batch output, and this step's batches on the
        # host (`host_batches`, see `host_encoder_batches`).
        self.layout_free = False
        self.host_batches = False
        self._layout_free_off = False  # a batch on the per-item-tensor path
        # Each warmed budget's output rows: the sources row handles index.
        self.output_rows: dict[int, int] = {}
        # CUDA graph metadata owns CUDA replay buffers. Track XLA warmup
        # separately rather than storing template dicts in that upstream map.
        self.compiled_budgets: dict[str, set[int]] = {
            path: set() for path in self.path_token_budgets
        }

        # We compile the forward pass function using the TPU backend.
        # This acts as our XLA cache. We compile it once, and it will generate
        # separate XLA graphs for each unique budget size passed to it.
        self._compiled_budget_forward = torch.compile(
            self.model.encoder_cudagraph_forward,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )

        logger.info(
            "[mm_encoder_manager] Initialized XLA path_budgets=%s max_batch_size=%s",
            self.path_token_budgets,
            self.max_batch_size,
        )

    def _pad_to_template(
        self,
        replay_values: Mapping[str, torch.Tensor | list[torch.Tensor] | None],
        budget: int,
        path: str = "default",
        upload: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Zero-pads dynamic multimodal inputs up to the static budget template;
        host rows reach the device by `upload` (default `self.upload`)."""
        template = self.budget_templates[path][budget]
        padded: dict[str, torch.Tensor] = {}

        for key, tmpl in template.items():
            src = replay_values.get(key)
            if src is None or tmpl is None:
                padded[key] = tmpl
                continue

            if isinstance(src, list):
                # Host rows from the budget-inputs hook (`host_encoder_batches`).
                padded[key] = rows_to_device(src, tmpl, upload or self.upload)
                continue

            if not hasattr(src, "shape") or src.ndim == 0:
                # Only scalar values, copy directly to template
                tmpl.copy_(src.to(device=tmpl.device))
                padded[key] = tmpl
                continue

            if src.shape == tmpl.shape:
                # Already perfectly bucket-sized. Detach if needed so the
                # input matches the template's requires_grad=False guard.
                src = src.to(device=tmpl.device)
                padded[key] = src.detach() if src.requires_grad else src
                continue

            padding_logic = self.config.padding_logics.get(
                key, self._copy_padded_buffer
            )
            padding_logic(tmpl, src)
            padded[key] = tmpl

        return padded

    def _upload_blocking(self, host: torch.Tensor) -> torch.Tensor:
        return host.to(self.device)

    @torch.no_grad()
    def _capture_budget_graph(
        self,
        token_budget: int,
        path: str = "default",
        axis_keys: CaptureAxisKeys = (),
    ) -> None:
        """Primes the XLA cache for a specific budget bucket."""
        if axis_keys:
            raise ValueError("TPU encoder budget templates do not support capture axes")
        template = self.budget_templates[path][token_budget]

        # Pass a copy of template so models calling values.pop() don't mutate
        # the stored template.
        try:
            out = self._compiled_budget_forward(dict(template), path=path)
            synchronize_tensors(out, wait=True)
            if self.has_budget_inputs_hook:
                # The hook's device ops have the budget's shapes: compile them
                # now. Warm-up waits right after, so its uploads may block.
                values = getattr(self.model, BUDGET_INPUTS_HOOK)(
                    None, template, self._upload_blocking
                )
                if self.layout_free:
                    # With the upload of its host rows, as `_run_budget_graph`
                    # runs them: one program.
                    padded = self._pad_to_template(
                        values, token_budget, path=path, upload=self._upload_blocking
                    )
                    out = self._compiled_budget_forward(dict(padded), path=path)
                    synchronize_tensors(out, wait=True)
                    self.output_rows[token_budget] = out.shape[0]
                else:
                    synchronize_tensors(
                        [v for v in values.values() if isinstance(v, torch.Tensor)],
                        wait=True,
                    )
            self.compiled_budgets[path].add(token_budget)
        except Exception as e:
            logger.warning(
                "[mm_encoder_manager] Failed to precompile vision encoder "
                "for budget=%d, path='%s': %s",
                token_budget,
                path,
                e,
            )

    @torch.no_grad()
    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        path: str = "default",
        axis_keys: CaptureAxisKeys = (),
        *,
        allow_eager_fallback: bool = True,
    ) -> torch.Tensor | None:
        """Pads actual inputs and runs the compiled XLA graph."""
        if axis_keys:
            raise ValueError("TPU encoder budget templates do not support capture axes")
        num_items = len(self._get_item_specs(mm_kwargs))
        if (
            path not in self.budget_templates
            or token_budget not in self.budget_templates[path]
        ):
            self.graph_misses += num_items
            return None

        # Prepare dynamic inputs. A model with the budget-inputs hook builds its
        # layout-derived inputs at the template's shapes, so their device ops
        # are the same for every item layout that fits the budget; upstream's
        # replay buffers compile them for each new layout. The hook gets the
        # path's template, not the path: it serves one-path models (Qwen3-VL).
        if self.has_budget_inputs_hook:
            values = getattr(self.model, BUDGET_INPUTS_HOOK)(
                mm_kwargs, self.budget_templates[path][token_budget], self.upload
            )
        else:
            values = self.model.prepare_encoder_cudagraph_replay_buffers(
                mm_kwargs, self.max_batch_size, self.max_frames_per_batch, path=path
            ).values

        # Pad dynamic inputs to strictly match the static budget shape
        padded = self._pad_to_template(values, token_budget, path=path)

        # Run encoder with eager fallback on compilation/execution failure
        try:
            out_tensor = self._compiled_budget_forward(dict(padded), path=path)
            self.graph_hits += num_items
            return out_tensor
        except Exception as e:
            if not allow_eager_fallback:
                raise
            logger.warning(
                "[mm_encoder_manager] Compiled vision encoder forward failed "
                "for budget=%d, path='%s': %s. Falling back to "
                "encoder_eager_forward.",
                token_budget,
                path,
                e,
            )
            self.graph_misses += num_items
            if self.host_batches:
                mm_kwargs = _to_device(mm_kwargs, self.device)
            with torch.inference_mode():
                return self.model.encoder_eager_forward(mm_kwargs, path=path)

    def enable_layout_free(self) -> str | None:
        """Hold encoder outputs as row handles into their batch output
        (`runner.mm_encoder_rows`); None if enabled, else why this manager
        keeps vLLM's per-item tensors."""
        if not self.has_budget_inputs_hook:
            return f"{type(self.model).__name__} has no {BUDGET_INPUTS_HOOK}"
        postprocess = type(self.model).postprocess_encoder_output
        if postprocess is not SupportsEncoderCudaGraph.postprocess_encoder_output:
            owner = postprocess.__qualname__
            return f"the model post-processes encoder outputs ({owner})"
        if tuple(self.path_token_budgets) != ("default",) or self.use_dp:
            return "it has several encoder paths or encodes data-parallel"
        self.layout_free = True
        return None

    def _execute_local_layout_free(self, mm_kwargs: dict[str, Any]) -> list[Any]:
        """`_execute_local` with outputs as row handles.

        A batch whose items' total is past the largest budget (as a video to
        chunk) takes today's per-item-tensor path, on the device, instead.
        """
        item_specs = self._get_item_specs(mm_kwargs)
        max_budget = max(self.token_budgets, default=0)
        if sum(spec.output_tokens for spec in item_specs) > max_budget:
            host_batches, self.host_batches, self._layout_free_off = (
                self.host_batches,
                False,
                True,
            )
            try:
                if host_batches:
                    mm_kwargs = _to_device(mm_kwargs, self.device)
                return self._execute_local(mm_kwargs)
            finally:
                self.host_batches, self._layout_free_off = host_batches, False

        def to_rows(
            outputs: dict[str, torch.Tensor],
            indices: list[int],
            per_item_out_tokens: list[int],
            dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
            clone: bool = False,
            batch_mm_kwargs: dict[str, Any] | None = None,
        ) -> None:
            items = split_batch_output(
                outputs["default"], [per_item_out_tokens[i] for i in indices]
            )
            for index, item in zip(indices, items, strict=True):
                # Row handles stand in for vLLM's per-item tensors.
                cast(Any, dest)[index] = item

        previous = self.model.__dict__.get("postprocess_encoder_output", _UNSET)
        self.model.postprocess_encoder_output = to_rows
        try:
            return super()._execute_local(mm_kwargs)
        finally:
            if previous is _UNSET:
                self.model.__dict__.pop("postprocess_encoder_output", None)
            else:
                self.model.postprocess_encoder_output = previous

    def _execute_local(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Execute locally, temporally chunking videos that exceed max budget."""
        if self.layout_free and not self._layout_free_off:
            return self._execute_local_layout_free(mm_kwargs)
        item_specs = self._get_item_specs(mm_kwargs)
        total_tokens = sum(spec.output_tokens for spec in item_specs)
        max_budget = max(self.token_budgets, default=0)
        # When total batch tokens exceed the largest compiled bucket
        # (max_budget), embed_multimodal_chunked temporally splits oversized
        # videos so each call to upstream _execute_local fits in a compiled
        # bucket, avoiding unbucketed eager fallback, XLA recompilation, and
        # VMEM OOM on long videos.
        if total_tokens > max_budget > 0:
            total_patches = sum(spec.input_size for spec in item_specs)
            max_patches = get_video_chunk_patch_size(
                self.vllm_config.model_config,
                default_max_patches=max_budget * total_patches // total_tokens,
            )
            if max_patches > 0:
                return list(
                    embed_multimodal_chunked(
                        lambda **kw: super(MMEncoderManager, self)._execute_local(kw),
                        mm_kwargs,
                        max_patches,
                    )
                )
        return super()._execute_local(mm_kwargs)

    @torch.no_grad()
    def execute_dp_coordinated(
        self, mm_kwargs: dict[str, Any] | None, input_error: str | None = None
    ) -> list[torch.Tensor]:
        """Run encoding in lockstep across attention-DP engines sharing EP."""
        from vllm.distributed.parallel_state import get_dp_group, get_ep_group

        if tuple(self.path_token_budgets) != ("default",):
            raise ValueError(
                "Coordinated encoding requires a single default encoder path"
            )
        all_tokens: list[int] = []
        assignment: list[int] = []
        images_per_rank: list[int] = []
        tokens: list[int] = []
        batches: list[tuple[list[int], int]] = []
        try:
            if input_error is None and mm_kwargs is not None:
                if self.use_dp:
                    all_tokens = self._get_per_item_out_tokens(mm_kwargs)
                    mm_kwargs, assignment, images_per_rank, _ = self._dp_shard(
                        mm_kwargs, all_tokens
                    )
                specs = self._get_item_specs(mm_kwargs)
                tokens = [spec.output_tokens for spec in specs]
                batches = _pack_encoder_items(
                    [spec.get_path_output_tokens("default") for spec in specs],
                    self.path_token_budgets["default"],
                    self.max_batch_size,
                )
        except (KeyError, TypeError, ValueError) as exc:
            input_error = str(exc)
        group = get_ep_group() if self.use_dp else get_dp_group()
        plans: list[Any] = [None] * group.world_size
        torch.distributed.all_gather_object(
            plans,
            (input_error, [budget for _, budget in batches], sum(tokens)),
            group=group.cpu_group,
        )
        errors = [error for error, _, _ in plans if error is not None]
        if errors:
            raise ValueError(f"Invalid coordinated encoder inputs: {errors}")
        outputs: dict[int, torch.Tensor] = {}
        for step in range(max(len(plan) for _, plan, _ in plans)):
            budget = max(plan[step] if step < len(plan) else 0 for _, plan, _ in plans)
            if step < len(batches):
                assert mm_kwargs is not None
                indices, _ = batches[step]
                kwargs, axis_keys = self._select_items(mm_kwargs, indices)
                output = self._run_budget_graph(
                    kwargs, budget, axis_keys=axis_keys, allow_eager_fallback=False
                )
                assert output is not None
                self.model.postprocess_encoder_output(
                    {"default": output},
                    indices,
                    tokens,
                    outputs,
                    clone=True,
                    batch_mm_kwargs=kwargs,
                )
            else:
                output = self._compiled_budget_forward(
                    dict(self.budget_templates["default"][budget]), path="default"
                )
            synchronize_tensors(output, wait=True)
        local_outputs = [outputs[index] for index in range(len(tokens))]
        if self.use_dp:
            return self._gather_dp_outputs(
                local_outputs,
                all_tokens,
                assignment,
                images_per_rank,
                max(count for _, _, count in plans),
            )
        return local_outputs

    def _dp_gather(
        self,
        local_outputs: list[torch.Tensor],
        per_item_out_tokens: list[int],
        image_rank_assignment: list[int],
        images_per_rank: list[int],
        max_output_tokens_per_rank: int,
    ) -> list[torch.Tensor]:
        """Gather encoder DP outputs at the same point on every TP rank.

        Each rank encodes a different share of the images, and upstream's
        gather leaves its input and output deferred, so ranks can execute the
        all-gather at different points among their collectives.
        ``_gather_dp_outputs`` materializes the input before the all-gather
        and the output right after it.
        """
        return self._gather_dp_outputs(
            local_outputs,
            per_item_out_tokens,
            image_rank_assignment,
            images_per_rank,
            max_output_tokens_per_rank,
        )

    def _gather_dp_outputs(
        self,
        local_outputs: list[torch.Tensor],
        tokens: list[int],
        assignment: list[int],
        images_per_rank: list[int],
        max_tokens: int,
    ) -> list[torch.Tensor]:
        from vllm.distributed.parallel_state import get_tp_group

        # Each new all-gather size compiles on every rank; budgets bound the sizes.
        max_tokens = self._gather_slot_tokens(max_tokens)
        padded = torch.zeros(
            (max_tokens, self.config.out_hidden_size),
            device=self.device,
            dtype=self.dtype,
        )
        offset = 0
        for output in local_outputs:
            padded[offset : offset + len(output)].copy_(output)
            offset += len(output)
        synchronize_tensors(padded, wait=True)
        gathered = get_tp_group().all_gather(padded, dim=0)
        # Empty engines must materialize the collective before discarding its output.
        synchronize_tensors(gathered, wait=True)
        outputs: dict[int, torch.Tensor] = {}
        start = 0
        for rank, count in enumerate(images_per_rank):
            offset = rank * max_tokens
            for index in assignment[start : start + count]:
                outputs[index] = gathered[offset : offset + tokens[index]]
                offset += tokens[index]
            start += count
        return [outputs[index] for index in range(len(tokens))]

    def _gather_slot_tokens(self, max_tokens: int) -> int:
        """Per-rank slot size of the encoder-DP all-gather for `max_tokens`.

        The smallest token budget that fits, or, past the largest budget (one
        rank encoding several large items), the next multiple of it. Startup
        warms multiples up to ``max_step_tokens`` via
        ``precompile_vision_encoder``; any larger unscheduled multiple compiles
        on first use.
        """
        budget = self._find_smallest_fitting_budget_given_tokens(max_tokens)
        if budget is not None:
            return budget
        return align_to(max_tokens, self.token_budgets[-1])

    def precompile_vision_encoder(self, max_step_tokens: int = 0) -> None:
        """Prime the XLA compilation cache for all configured token budgets.

        Under encoder data parallelism, also warms the all-gather for every
        token budget and, when ``max_step_tokens`` exceeds the largest budget,
        each reachable multiple of the largest budget up to ``max_step_tokens``.
        """
        for path, templates in self.budget_templates.items():
            for budget in templates:
                self._capture_budget_graph(budget, path=path)
        if self.use_dp:
            largest = self.token_budgets[-1]
            slots = list(self.token_budgets)
            if max_step_tokens > largest:
                slots.extend(
                    range(
                        2 * largest,
                        align_to(max_step_tokens, largest) + 1,
                        largest,
                    )
                )
            for slot in slots:
                self._gather_dp_outputs([], [], [], [], slot)


def maybe_create_mm_encoder_manager(
    vllm_config: VllmConfig,
    device: torch.device,
    vllm_model: torch.nn.Module,
) -> MMEncoderManager | None:
    """Create an MMEncoderManager instance if enabled and supported.

    Args:
        vllm_config: The global vLLM configuration object.
        device: Target torch device (TPU) for model execution.
        vllm_model: The underlying multimodal model module.

    Returns:
        An initialized ``MMEncoderManager`` if encoder compilation is enabled
        (``cudagraph_mm_encoder=True``) and supported by the model (implements
        ``SupportsEncoderCudaGraph``). Returns ``None`` otherwise, in which case
        the runner falls back to the eager multimodal path
        (``model.embed_multimodal(**mm_kwargs)``) without budget padding.
    """
    if not vllm_config.compilation_config.cudagraph_mm_encoder:
        return None
    if not supports_encoder_cudagraph(vllm_model):
        return None
    if vllm_model.get_encoder_cudagraph_config().capture_axes:
        logger.info("Using eager encoding for a model with encoder capture axes.")
        return None
    return MMEncoderManager(vllm_config, device, vllm_model)
