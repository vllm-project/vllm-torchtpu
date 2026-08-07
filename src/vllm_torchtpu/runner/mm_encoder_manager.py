import logging
from typing import Any, Dict

import torch
from torch_tpu._internal import sync
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

logger = logging.getLogger(__name__)


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
            dtype=vllm_config.model_config.dtype,
            model=vllm_model,
        )

        # Create dummy input templates for every budget size
        self.budget_templates: Dict[int, Dict[str, torch.Tensor]] = {}
        for budget in self.token_budgets:
            capture = self.model.prepare_encoder_cudagraph_capture_inputs(
                budget, self.max_batch_size, self.max_frames_per_batch,
                torch.device("cpu"), self.dtype)
            self.budget_templates[budget] = capture.values

        # We compile the forward pass function using the TPU backend.
        # This acts as our XLA cache. We compile it once, and it will generate
        # separate XLA graphs for each unique budget size passed to it.
        self._compiled_budget_forward = torch.compile(
            self.model.encoder_cudagraph_forward,
            backend="tpu",
            fullgraph=True,
            dynamic=False)

        logger.info(
            f"[mm_encoder_manager] Initialized XLA budgets={self.token_budgets} "
            f"max_batch_size={self.max_batch_size}")

    def _pad_to_template(
        self,
        replay_values: Dict[str, torch.Tensor],
        budget: int,
    ) -> Dict[str, torch.Tensor]:
        """Zero-pads dynamic multimodal inputs up to the static budget template."""
        template = self.budget_templates[budget]
        padded: Dict[str, torch.Tensor] = {}

        for key, tmpl in template.items():
            src = replay_values.get(key)
            if src is None:
                padded[key] = tmpl
                continue

            if not hasattr(src, "shape") or src.ndim == 0:
                # Scalars (like max_seqlen) pass through
                padded[key] = tmpl
                continue

            if src.shape == tmpl.shape:
                # Already perfectly bucket-sized
                padded[key] = src.to(device=tmpl.device)
                continue

            # Dynamic shape: allocate a blank static tensor and inject the data
            buf = torch.zeros_like(tmpl, device=tmpl.device, dtype=tmpl.dtype)

            # Slice-copy the dynamic data into the beginning of the static buffer
            # Assuming the sequence/token dimension is dim 0.
            n = src.shape[0]
            buf[:n] = src.to(dtype=tmpl.dtype, device=tmpl.device)
            padded[key] = buf

        return padded

    @torch.no_grad()
    def _capture_budget_graph(self, token_budget: int) -> None:
        """Primes the XLA cache for a specific budget bucket."""
        template = self.budget_templates[token_budget]

        # Move the CPU templates to TPU before capturing
        template_tpu = {
            k:
            v.to(self.device)
            if isinstance(v, torch.Tensor) and v.ndim > 0 else v
            for k, v in template.items()
        }

        # Run the compiled function once to force XLA to compile the graph
        out = self._compiled_budget_forward(template_tpu)
        sync.synchronize(out, wait=True)

        self.budget_graphs[token_budget] = template_tpu

    @torch.no_grad()
    def _run_budget_graph(
        self,
        mm_kwargs: Dict[str, Any],
        token_budget: int,
    ) -> torch.Tensor | None:
        """Pads actual inputs and runs the compiled XLA graph."""
        num_items = len(self._get_item_specs(mm_kwargs))
        if token_budget not in self.budget_templates:
            self.graph_misses += num_items
            return None

        # Prepare dynamic inputs
        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs, self.max_batch_size, self.max_frames_per_batch)

        # Pad dynamic inputs to strictly match the static budget shape
        padded_torch = self._pad_to_template(replay.values, token_budget)

        # Move the padded static-shape tensors to TPU
        padded_tpu = {
            k:
            v.to(self.device)
            if isinstance(v, torch.Tensor) and v.ndim > 0 else v
            for k, v in padded_torch.items()
        }

        # Run encoder
        out_tensor = self._compiled_budget_forward(padded_tpu)

        self.graph_hits += num_items
        return out_tensor

    def precompile_vision_encoder(self) -> None:
        """Prime the XLA compilation cache for all configured token budgets.

        Iterates through all token budgets in ``self.token_budgets`` and triggers
        a forward pass on dummy inputs to ensure the compiled XLA graphs are
        warmed up before serving traffic begins.
        """
        for budget in self.token_budgets:
            self._capture_budget_graph(budget)


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
    # TODO: dual path graph padding is not supported yet
    if getattr(vllm_model.get_encoder_cudagraph_config(),
               "enable_dual_path_graph", False):
        return None
    return MMEncoderManager(vllm_config, device, vllm_model)
