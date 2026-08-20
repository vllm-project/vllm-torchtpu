import time
from typing import Any, Dict, List, Mapping

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


def _tensors_of(values: Mapping[Any, Any]) -> List[torch.Tensor]:
    """The tensors in ``values``, for synchronization.

    Input and output dicts on the encoder path legitimately carry ``None`` for
    optional entries, and ``synchronize_tensors`` rejects the whole list if one
    slips in.
    """
    return [v for v in values.values() if isinstance(v, torch.Tensor)]


def _brief(exc: Exception, limit: int = 300) -> str:
    """An exception's message, without a full tensor repr in the log.

    A pybind arity mismatch echoes every argument it was invoked with, which on
    this path is several thousand lines of padded template.
    """
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[:limit] + " [...]"


def _drain_pending() -> None:
    """Materialize whatever the failed warmup step left in the lazy graph.

    Ops traced but never synchronized stay pending and get fused into the next
    budget's encoder program, so each successive budget compiles a strictly
    larger graph: one such failure took the boot-time encoder precompile from
    52s total to 5, then 8, then more minutes *per budget*. Dropping the
    warmup's work is fine; leaving it queued is not.
    """
    try:
        synchronize_tensors(wait=True)
    except Exception as e:
        logger.warning(
            "[mm_encoder_manager] Could not drain the pending "
            "graph after a warmup failure: %s", _brief(e))


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

        # Create dummy input templates for every budget size and path directly
        # from self.path_token_budgets prepared by EncoderCudaGraphManager.
        self.budget_templates: Dict[str, Dict[int, Dict[
            str, torch.Tensor]]] = {
                path: {
                    budget:
                    self.model.prepare_encoder_cudagraph_capture_inputs(
                        budget,
                        self.max_batch_size,
                        self.max_frames_per_batch,
                        self.device,
                        self.dtype,
                        path=path).values
                    for budget in budgets if budget > 0
                }
                for path, budgets in self.path_token_budgets.items()
            }

        # Pinned host staging buffers for _pad_to_template, keyed by
        # (path, budget, key), plus how far each was last dirtied.
        self._host_stage: Dict[tuple, torch.Tensor] = {}
        self._host_stage_dirty: Dict[tuple, int] = {}

        # We compile the forward pass function using the TPU backend.
        # This acts as our XLA cache. We compile it once, and it will generate
        # separate XLA graphs for each unique budget size passed to it.
        self._compiled_budget_forward = torch.compile(
            self.model.encoder_cudagraph_forward,
            backend="tpu",
            fullgraph=True,
            dynamic=False)

        logger.info(
            f"[mm_encoder_manager] Initialized XLA path_budgets={self.path_token_budgets} "
            f"max_batch_size={self.max_batch_size}")

    # Keys whose leading dim scales with the image and which are therefore
    # sliced/concatenated per item before any budget padding happens.
    _HOST_STAGED_KEYS = ("pixel_values", "grid_thws")

    def execute(self, mm_kwargs: Dict[str, Any]) -> Any:
        """Land the batch on the host, then run the normal budget path.

        ``_execute_local`` slices and re-concatenates ``pixel_values`` per
        item (``select_encoder_cudagraph_items``) and ``_pad_to_template``
        then pads it to a budget. On TPU a device tensor whose leading dim
        depends on the image size keys every XLA program it feeds, so doing
        either of those on-device compiles a fresh program for every new
        image shape. Moving the batch to the host once, up front, makes all
        of that plain CPU work and leaves exactly one device transfer -- the
        static budget-shaped H2D in ``_pad_to_template``.
        """
        staged = None
        for key in self._HOST_STAGED_KEYS:
            val = mm_kwargs.get(key)
            if isinstance(val, torch.Tensor) and val.device.type != "cpu":
                if staged is None:
                    staged = dict(mm_kwargs)
                staged[key] = val.cpu()
        return super().execute(staged if staged is not None else mm_kwargs)

    def _host_stage_buffer(self, path: str, budget: int, key: str,
                           tmpl: torch.Tensor) -> torch.Tensor:
        """Reusable pinned host buffer shaped like ``tmpl``.

        One per (path, budget, key); the shape is static, so allocating it
        once keeps the padding path allocation-free after the first call.
        """
        cache_key = (path, budget, key)
        buf = self._host_stage.get(cache_key)
        if buf is None:
            buf = torch.zeros(tmpl.shape, dtype=tmpl.dtype)
            try:
                buf = buf.pin_memory()
            except (RuntimeError, NotImplementedError):
                pass  # pinning is an optimization, not a requirement
            self._host_stage[cache_key] = buf
            self._host_stage_dirty[cache_key] = 0
        return buf

    def _pad_to_template(
        self,
        replay_values: Dict[str, torch.Tensor],
        budget: int,
        path: str = "default",
    ) -> Dict[str, torch.Tensor]:
        """Zero-pads dynamic multimodal inputs up to the static budget template.

        The padding is staged on the *host* rather than on the TPU. A device
        tensor whose leading dim varies with the image size keys every XLA
        program that takes it as an input, so the obvious device-side
        ``dst[:n].copy_(src)`` recompiles once per distinct image size --
        measured at 2 new XLA programs and seconds of stall per new size,
        which is the bulk of multimodal TTFT. Padding on the host means the
        only device-visible transfer happens at the static template shape, so
        the program set is keyed on the budget alone.
        """
        template = self.budget_templates[path][budget]
        padded: Dict[str, torch.Tensor] = {}

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

            padding_logic = self.config.padding_logics.get(key)
            if padding_logic is not None:
                # Model-supplied logic owns its own shape discipline.
                padding_logic(tmpl, src)
                padded[key] = tmpl
                continue

            stage = self._host_stage_buffer(path, budget, key, tmpl)
            cache_key = (path, budget, key)
            n = src.shape[0]
            # Re-zero only the region a previous call dirtied; the rest of the
            # staging buffer is already zero.
            prev = self._host_stage_dirty[cache_key]
            if prev > n:
                stage[n:prev].zero_()
            self._host_stage_dirty[cache_key] = n
            stage[:n].copy_(src if src.device.type == "cpu" else src.cpu())
            # Static-shape H2D: this is the only device op keyed on `budget`.
            # Blocking on purpose -- `stage` is reused across calls, so an
            # in-flight async copy could be overwritten by the next batch.
            padded[key] = stage.to(tmpl.device)

        return padded

    @torch.no_grad()
    def _capture_budget_graph(self,
                              token_budget: int,
                              path: str = "default") -> None:
        """Primes the XLA cache for a specific budget bucket."""
        template = self.budget_templates[path][token_budget]

        # Pass a copy of template so models calling values.pop() don't mutate the stored template.
        try:
            out = self._compiled_budget_forward(dict(template), path=path)
            synchronize_tensors(out, wait=True)
            self._get_graph_set(path)[token_budget] = template
        except Exception as e:
            logger.warning(
                "[mm_encoder_manager] Failed to precompile vision encoder "
                "for budget=%d, path='%s': %s", token_budget, path, _brief(e))
            _drain_pending()
            return

        self._capture_budget_glue(out, token_budget, path=path)

    @torch.no_grad()
    def _capture_budget_glue(self,
                             out: torch.Tensor,
                             token_budget: int,
                             path: str = "default") -> None:
        """Prime the eager ops on either side of the compiled forward.

        ``_capture_budget_graph`` hands the device-resident template straight
        to the compiled forward, so it compiles the encoder graph and nothing
        else. A real request also runs ``_pad_to_template`` before it and
        ``postprocess_encoder_output`` after it, and both of those are ordinary
        lazy-tensor ops whose programs are keyed on shapes the template alone
        never produces. Left uncovered they compile on the first request that
        reaches each shape -- 43 XLA entries over the first 40 distinct images
        on Kimi K3, none of them the encoder graph itself.

        Both shape sets are finite and known here, so they are compiled at boot
        instead.
        """
        start = time.perf_counter()
        template = self.budget_templates[path][token_budget]

        # _pad_to_template: the staged host->device copy. A src shorter than
        # the template is what routes through the staging buffer at all; the
        # transfer itself is at the template shape either way.
        replay = {
            key: tmpl[:1].cpu()
            for key, tmpl in template.items()
            if isinstance(tmpl, torch.Tensor) and tmpl.ndim >= 1
            and tmpl.shape[0] > 1
        }
        if replay:
            try:
                padded = self._pad_to_template(replay, token_budget, path=path)
                synchronize_tensors(_tensors_of(padded), wait=True)
            except Exception as e:
                logger.warning(
                    "[mm_encoder_manager] Failed to precompile the padding "
                    "path for budget=%d, path='%s': %s", token_budget, path,
                    _brief(e))
                _drain_pending()
            finally:
                # Undo the warmup's dirty rows so the first real request sees
                # the zero-filled buffer the padding contract promises.
                for key in replay:
                    cache_key = (path, token_budget, key)
                    stage = self._host_stage.get(cache_key)
                    if stage is not None:
                        stage.zero_()
                        self._host_stage_dirty[cache_key] = 0

        # postprocess_encoder_output: one index_select per item, keyed on
        # (rows in this budget's packed output, per-item bucket). Only the
        # bucketed override has a shape set small enough to enumerate; the
        # upstream exact-length slicing is unbounded and cannot be primed.
        if not (envs.TPU_MM_BUCKET_SCATTER
                and hasattr(self.model, "mm_embed_bucket")):
            return
        if not isinstance(out, torch.Tensor) or out.ndim < 1:
            return

        buckets = 0
        bucket = getattr(self.model, "MM_EMBED_BUCKET_MIN", 64)
        while bucket <= out.shape[0]:
            try:
                dest: Dict[int, torch.Tensor] = {}
                self.model.postprocess_encoder_output({path: out}, [0],
                                                      [bucket],
                                                      dest,
                                                      clone=True)
                synchronize_tensors(_tensors_of(dest), wait=True)
            except Exception as e:
                logger.warning(
                    "[mm_encoder_manager] Failed to precompile the "
                    "postprocess path for budget=%d, bucket=%d, path='%s': %s",
                    token_budget, bucket, path, _brief(e))
                _drain_pending()
                break
            buckets += 1
            bucket *= 2

        logger.info(
            "[mm_encoder_manager] Primed glue for budget=%d path='%s': "
            "%d embed bucket(s) over %d packed rows in %.2f [secs].",
            token_budget, path, buckets, out.shape[0],
            time.perf_counter() - start)

    @torch.no_grad()
    def _run_budget_graph(
        self,
        mm_kwargs: Dict[str, Any],
        token_budget: int,
        path: str = "default",
    ) -> torch.Tensor | None:
        """Pads actual inputs and runs the compiled XLA graph."""
        num_items = len(self._get_item_specs(mm_kwargs))
        if (path not in self.budget_templates
                or token_budget not in self.budget_templates[path]):
            self.graph_misses += num_items
            return None

        # Prepare dynamic inputs
        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs,
            self.max_batch_size,
            self.max_frames_per_batch,
            path=path)

        # Pad dynamic inputs to strictly match the static budget shape
        padded = self._pad_to_template(replay.values, token_budget, path=path)

        # Run encoder with eager fallback on compilation/execution failure
        try:
            out_tensor = self._compiled_budget_forward(dict(padded), path=path)
            self.graph_hits += num_items
            return out_tensor
        except Exception as e:
            logger.warning(
                "[mm_encoder_manager] Compiled vision encoder forward failed "
                "for budget=%d, path='%s': %s. Falling back to "
                "encoder_eager_forward.", token_budget, path, e)
            self.graph_misses += num_items
            with torch.inference_mode():
                return self.model.encoder_eager_forward(mm_kwargs, path=path)

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
