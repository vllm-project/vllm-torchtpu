# SPDX-License-Identifier: Apache-2.0
import contextlib
from importlib import metadata as importlib_metadata

import jax
import numpy as np
import torch
from jax.sharding import Mesh
from packaging import version
from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.rotary_embedding import (MRotaryEmbedding,
                                                         RotaryEmbedding)
from vllm.model_executor.layers.rotary_embedding.mrope_interleaved import \
    MRotaryEmbeddingInterleaved
from vllm.model_executor.model_loader import get_model_loader
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from vllm_torchtpu import envs
from vllm_torchtpu.layers.vllm import token_padding
from vllm_torchtpu.layers.vllm.quantization import get_tpu_quantization_config
from vllm_torchtpu.layers.vllm.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
from vllm_torchtpu.runner import utils as runner_utils
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.dspark import DSparkProposer
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

logger = init_logger(__name__)


def _validate_libtpu_version() -> None:
    """Validate libtpu opportunistically if it is present."""
    libtpu_version = importlib_metadata.version("libtpu")
    parsed_version = version.parse(libtpu_version)
    if parsed_version < version.parse("0.0.35"):
        raise RuntimeError(
            "Argmax is having accuracy issue with libtpu < 0.0.35")
    if parsed_version < version.parse("0.0.36"):
        logger.warning_once(
            "libtpu < 0.0.36 may enable "
            "--xla_tpu_impure_enable_large_2nd_minor_layout by default, "
            "which can hurt performance. "
            "Upgrade libtpu to >= 0.0.36, or set "
            "LIBTPU_INIT_ARGS=--xla_tpu_impure_enable_large_2nd_minor_layout=false."
        )


@contextlib.contextmanager
def _torch_tpu_wrapper():
    """Alias torch.cuda.* to torch.tpu.* during inherited GPUModelRunner init.

    Mirrors vllm/v1/worker/xpu_model_runner.py:_torch_cuda_wrapper. The parent
    GPUModelRunner constructs torch.cuda.Stream/Event for comm and async copy;
    this wrapper makes those resolve to the TPU equivalents (verified present
    on torch.tpu: Stream, Event, current_stream, default_stream, stream,
    set_stream, synchronize, set_device, current_device, device_count).

    The 5 APIs not on torch.tpu are stubbed:
      - mem_get_info -> torch.accelerator.get_memory_info (works on TPU)
      - empty_cache  -> no-op (XLA manages HBM)
      - graph / CUDAGraph / graph_pool_handle -> never reached because we
        disable cudagraphs in __init__ via compilation_config overrides.
    """
    saved = {}
    aliased = ("Stream", "Event", "current_stream", "default_stream", "stream",
               "set_stream", "synchronize", "set_device", "current_device",
               "device_count", "is_available")
    for name in aliased:
        saved[name] = getattr(torch.cuda, name, None)
        setattr(torch.cuda, name, getattr(torch.tpu, name))
    saved["mem_get_info"] = getattr(torch.cuda, "mem_get_info", None)
    torch.cuda.mem_get_info = lambda *a, **kw: torch.accelerator.get_memory_info(
    )
    saved["empty_cache"] = getattr(torch.cuda, "empty_cache", None)
    torch.cuda.empty_cache = lambda: None
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                if hasattr(torch.cuda, k):
                    delattr(torch.cuda, k)
            else:
                setattr(torch.cuda, k, v)


class ModelLoadingMixin:
    """Model load and TPU kernel initialization."""

    def _init_speculative_decoding(self) -> None:
        self.drafter = None
        self.rejection_sampler = None
        if self.speculative_config:
            self.rejection_sampler = RejectionSampler(self.speculative_config,
                                                      self.device)
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "dflash":
                self.drafter = DFlashProposer(self, self.vllm_config)
            elif self.speculative_config.method == "dspark":
                self.drafter = DSparkProposer(self, self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = Eagle3Proposer(self, self.vllm_config)
            else:
                raise NotImplementedError(
                    "Unsupported speculative decoding method: "
                    f"{self.speculative_config.method}")

    def _truncate_rope_caches(self) -> None:
        """Slice rotary cos_sin caches to max_model_len which can reduce
        the overhead of xla layout data copy. Applicable to text-only.
        """
        multimodal_config = self.model_config.multimodal_config
        if self.model_config.is_multimodal_model and not (
                multimodal_config is not None
                and multimodal_config.language_model_only):
            logger.warning(
                "TPU_ROPE_CACHE_TRUNCATE skipped, multimodal positions are "
                "not bounded by max_model_len")
            return
        max_len = self.model_config.max_model_len
        num_eligible = 0
        num_truncated = 0
        for name, module in self.model.named_modules():
            if type(module) not in (RotaryEmbedding, MRotaryEmbedding,
                                    MRotaryEmbeddingInterleaved):
                continue
            for buf_name in ("cos_sin_cache", "cos_sin_cache_bf16"):
                buf = getattr(module, buf_name, None)
                if buf is None or not isinstance(buf, torch.Tensor):
                    continue
                num_eligible += 1
                if buf.ndim >= 1 and buf.shape[0] > max_len:
                    truncated = buf[:max_len].clone(
                        memory_format=torch.contiguous_format)
                    setattr(module, buf_name, truncated)
                    num_truncated += 1
                    logger.info("Truncated rope cache %s.%s rows %d -> %d",
                                name, buf_name, buf.shape[0], max_len)
        if num_eligible:
            logger.info("Truncated %d of %d rope caches", num_truncated,
                        num_eligible)
        else:
            logger.warning(
                "TPU_ROPE_CACHE_TRUNCATE is set but the model has no "
                "position-indexed rope cache, nothing was truncated")

    def _create_mesh_for_parallelism(self) -> Mesh:
        local_devices = list(jax.local_devices())
        if not local_devices:
            raise ValueError("No TPU devices are visible to create JAX mesh.")

        # Per-worker JAX mesh is normally single-chip; vLLM native multiprocess
        # handles TP>1 outside this JAX SPMD path.
        if self.parallel_config.world_size == 1 and \
                self.parallel_config.tensor_parallel_size > 1:
            raise ValueError(
                "Single-process TPU mesh TP>1 is not supported in this path. "
                "Use vLLM multiprocess mode for --tensor-parallel-size > 1.")
        mesh_devices = np.asarray(local_devices[:1]).reshape((1, ))
        mesh = Mesh(mesh_devices, axis_names=("model", ))
        # Also report the global device view. This mesh is deliberately
        # single-device, but op-local multi-device meshes (see
        # distributed.pcp.get_or_create_pcp_mesh) are built from `jax.devices()`
        # and only work when every peer's device is visible here, so the counts
        # are worth having in every server log.
        logger.info(
            "Init mesh | tp_size=1 | device_id=%s | jax local_devices=%d "
            "global devices=%d ids=%s",
            getattr(local_devices[0], "id", str(local_devices[0])),
            len(local_devices),
            jax.device_count(),
            [getattr(d, "id", None) for d in jax.devices()],
        )
        return mesh

    def load_model(self) -> None:
        self.device_config.device = torch.device("tpu")
        self.device = self.device_config.device
        self.vllm_config.device_config = self.device_config

        # Set TPU-specific quantization config before model loading.
        # This ensures that TPU-compatible quantization methods are used
        # instead of the default GPU-based ones (e.g., MXFP4 for MoE models).
        # For unquantized models (quantization=None), we still need to apply
        # our TPU config to override vLLM's default UnquantizedFusedMoEMethod
        # which uses torch_xla.
        logger.info("Setting TPU quantization config for: %s",
                    self.model_config.quantization)
        self.vllm_config.quant_config = get_tpu_quantization_config(
            self.vllm_config)

        model_loader = get_model_loader(self.load_config)
        logger.info("Loading model from scratch...")
        with set_vllm_model_wrapper_context(mesh=self.mesh,
                                           vllm_config=self.vllm_config), \
             set_current_vllm_config(self.vllm_config):
            model = model_loader.load_model(vllm_config=self.vllm_config,
                                            model_config=self.model_config)
        self.model = model
        if envs.TPU_ROPE_CACHE_TRUNCATE:
            self._truncate_rope_caches()
        if envs.TPU_ROPE_CACHE_ROW_MAJOR:
            runner_utils.relayout_rope_caches(self.model)
        if envs.TPU_MOE_HASH_TABLE_ROW_MAJOR:
            runner_utils.relayout_hash_tables(self.model)

        # If using eagle3/mtp or dflash speculative decoding, load the draft model and
        # share the target's embeddings / LM head (if the draft requires it).
        if self._is_async_drafter:
            self.drafter.load_model(self.model)

        # Ensure attention custom ops exist before any compile/inference path,
        self._initialize_pallas_kernels()

        from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import \
            VllmCompressedTensorsW4ANMxfp4MoEMethod
        uses_mxfp4_moe = any(
            isinstance(getattr(module, "quant_method", None),
                       VllmCompressedTensorsW4ANMxfp4MoEMethod)
            for module in self.model.modules())
        if envs.TPU_MOE_SKIP_PADDED_TOKENS or uses_mxfp4_moe:
            self._token_padding_state = token_padding.TokenPaddingState.create(
                self.max_num_tokens, self.device)
            token_padding.set_padding_state(self._token_padding_state)

    def _initialize_pallas_kernels(self):
        self._initialize_attention_kernels()
        self._initialize_quantization_kernels()

    def _initialize_quantization_kernels(self):
        from vllm_torchtpu.layers.vllm.linear_common import (
            _get_quantized_matmul_fp4_op, _get_quantized_matmul_op)

        with set_current_vllm_config(self.vllm_config):
            # Pre-warm FP8 quantized-matmul lock; Dynamo can't trace Lock.
            _get_quantized_matmul_op()
            # Same for the NVFP4 W4A16 matmul op.
            _get_quantized_matmul_fp4_op()

    def _initialize_attention_kernels(self, force: bool = False) -> None:
        """Pre-build Pallas RPA and custom attention kernels before torch.compile."""
        if self._attention_kernels_initialized and not force:
            return
        from vllm_torchtpu.layers.vllm.attention import (
            _DRAFT_KV_BLOCK_CAP, PallasAttentionBackendImpl,
            _pallas_rpa_kernel_local)

        # Only draft proposers expose _draft_attn_layer_names; NgramProposer
        # operates purely on token IDs. no draft attention layers to relocate.
        spec_drafter = self.drafter
        draft_attn_names = getattr(spec_drafter, "_draft_attn_layer_names",
                                   None) or set()

        layers = get_layers_from_vllm_config(self.vllm_config,
                                             AttentionLayerBase)
        initialized_count = 0
        with set_vllm_model_wrapper_context(mesh=self.mesh,
                                            vllm_config=self.vllm_config):
            for name, attn_layer in layers.items():
                if isinstance(getattr(attn_layer, "impl", None),
                              PallasAttentionBackendImpl):
                    # Relocate a REPLICATED (tp=1) draft's attention to the LOCAL
                    # (non-shard_map) kernel.
                    if (name in draft_attn_names and
                            self.speculative_config.draft_tensor_parallel_size
                            == 1):
                        # Instance attrs shadow the ClassVars; unique prefix
                        # keeps the local kernel op out of the sharded registry.
                        attn_layer.impl._kernel_entry = _pallas_rpa_kernel_local
                        attn_layer.impl._kernel_op_prefix = (
                            "pallas::rpa_kernel_local")
                        logger.info(
                            "Draft attn %s -> LOCAL (non-shard_map) RPA kernel"
                            " | DRAFT_KV_BLOCK_CAP=%d", name,
                            _DRAFT_KV_BLOCK_CAP)
                    attn_layer.impl.initialize_kernel(attn_layer)
                    initialized_count += 1

            # Pre-build custom attention, compressor, and indexer kernels (e.g. DeepSeek-V4 SWA/CSA/HCA)
            if hasattr(self, "model") and self.model is not None:
                for module in self.model.modules():
                    if hasattr(module, "_build_attn_op") and hasattr(
                            module, "attn_op"):
                        module.__dict__.pop("_attn_op_instance", None)
                        _ = module.attn_op
                        initialized_count += 1
                    if hasattr(module, "_build_compressor_op") and hasattr(
                            module, "compressor_op"):
                        module.__dict__.pop("_compressor_op_instance", None)
                        _ = module.compressor_op
                        initialized_count += 1
                    if hasattr(module, "_build_indexer_op") and hasattr(
                            module, "indexer_op"):
                        _ = module.indexer_op
                        initialized_count += 1
                    for op_name in ("mhc_ops", "mhc_post_op", "qnorm_rope_op",
                                    "kv_rope_op", "o_proj_op"):
                        if hasattr(type(module), op_name):
                            _ = getattr(module, op_name)
                            initialized_count += 1
        logger.info(
            "Pre-built attention/indexing kernels for %d layers/modules.",
            initialized_count)
        self._attention_kernels_initialized = True
