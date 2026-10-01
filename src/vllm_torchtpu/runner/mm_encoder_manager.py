from typing import Any, cast

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    SupportsEncoderCudaGraph,
    supports_encoder_cudagraph,
)
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.mm_video_chunking import (
    embed_multimodal_chunked,
    get_video_chunk_patch_size,
)
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


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

        encoder_use_dp = getattr(vllm_model, "use_data_parallel", None)
        if isinstance(encoder_use_dp, bool):
            tp_size = vllm_config.parallel_config.tensor_parallel_size
            self.use_dp = encoder_use_dp and tp_size > 1
        elif encoder_use_dp is not None:
            logger.warning("Ignoring non-boolean use_data_parallel=%r", encoder_use_dp)

        # Create dummy input templates for every budget size and path directly
        # from self.path_token_budgets prepared by EncoderCudaGraphManager.
        self.budget_templates: dict[str, dict[int, dict[str, torch.Tensor]]] = {
            path: {
                budget: self.model.prepare_encoder_cudagraph_capture_inputs(
                    budget,
                    self.max_batch_size,
                    self.max_frames_per_batch,
                    self.device,
                    self.dtype,
                    path=path,
                ).values
                for budget in budgets
                if budget > 0
            }
            for path, budgets in self.path_token_budgets.items()
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
        replay_values: dict[str, torch.Tensor | None],
        budget: int,
        path: str = "default",
    ) -> dict[str, torch.Tensor]:
        """Zero-pads dynamic multimodal inputs up to the static budget template."""
        template = self.budget_templates[path][budget]
        padded: dict[str, torch.Tensor] = {}

        for key, tmpl in template.items():
            src = replay_values.get(key)
            if src is None:
                padded[key] = tmpl
                continue

            if not hasattr(src, "shape") or src.ndim == 0:
                # Only scalar values, copy directly to template
                tmpl.copy_(src.to(device=tmpl.device))
                padded[key] = tmpl
                continue

            if src.shape == tmpl.shape:
                # Already perfectly bucket-sized
                padded[key] = src.to(device=tmpl.device)
                continue

            padding_logic = self.config.padding_logics.get(
                key, self._copy_padded_buffer
            )
            padding_logic(tmpl, src)
            padded[key] = tmpl

        return padded

    @torch.no_grad()
    def _capture_budget_graph(self, token_budget: int, path: str = "default") -> None:
        """Primes the XLA cache for a specific budget bucket."""
        template = self.budget_templates[path][token_budget]

        # Pass a copy of template so models calling values.pop() don't mutate
        # the stored template.
        try:
            out = self._compiled_budget_forward(dict(template), path=path)
            synchronize_tensors(out, wait=True)
            self._get_graph_set(path)[token_budget] = template
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
        *,
        allow_eager_fallback: bool = True,
    ) -> torch.Tensor | None:
        """Pads actual inputs and runs the compiled XLA graph."""
        num_items = len(self._get_item_specs(mm_kwargs))
        if (
            path not in self.budget_templates
            or token_budget not in self.budget_templates[path]
        ):
            self.graph_misses += num_items
            return None

        # Prepare dynamic inputs
        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs, self.max_batch_size, self.max_frames_per_batch, path=path
        )

        # Pad dynamic inputs to strictly match the static budget shape
        padded = self._pad_to_template(replay.values, token_budget, path=path)

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
            with torch.inference_mode():
                return self.model.encoder_eager_forward(mm_kwargs, path=path)

    def _execute_local(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Execute locally, temporally chunking videos that exceed max budget."""
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
                kwargs = self._select_items(mm_kwargs, indices)
                output = self._run_budget_graph(
                    kwargs, budget, allow_eager_fallback=False
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

    def _gather_dp_outputs(
        self,
        local_outputs: list[torch.Tensor],
        tokens: list[int],
        assignment: list[int],
        images_per_rank: list[int],
        max_tokens: int,
    ) -> list[torch.Tensor]:
        from vllm.distributed.parallel_state import get_tp_group

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

    def precompile_vision_encoder(self) -> None:
        """Prime the XLA compilation cache for all configured token budgets."""
        for path, templates in self.budget_templates.items():
            for budget in templates:
                self._capture_budget_graph(budget, path=path)


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
    return MMEncoderManager(vllm_config, device, vllm_model)
