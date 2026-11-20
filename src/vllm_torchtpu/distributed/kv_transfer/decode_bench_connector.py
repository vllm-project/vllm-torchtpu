"""Decode-only benchmarking connector that never computes or writes prompt KV.

``TPUDecodeBenchConnector`` turns a single vLLM server into the decode leg of a
prefill/decode (P/D) disaggregated deployment, without a prefill leg. Every
prompt token that the local prefix cache does not cover (except the last one)
is reported to the scheduler as already loaded from an external source, exactly
as a P/D consumer reports KV pulled from its producer. The scheduler then marks
those tokens computed and schedules only the last prompt token, which a P/D
decode leg also recomputes to produce its first sampled token.

Unlike upstream ``DecodeBenchConnector``, the worker side does not write fill
values. KV cache tensors are zero-initialized at allocation, so a claimed block
holds either zeros or the finite KV of an earlier request. Attention and the
DSA indexer run the same kernels over the same shapes either way. Writing is
also unsafe on this backend: upstream's fill assumes every cache is a single
block-indexed tensor with ``shape[0] == num_blocks``. TPU caches can be split
(``num_blocks * split`` rows), and native or indexer caches arrive as tuples,
which upstream treats as Mamba state and overwrites whole on each fill.

Attention and indexer cost depends on context length and batch size, not on
the KV values (the sparse top-k width is fixed). MoE routing of the decoded
tokens does see the dummy context through their hidden states, so expert load
balance can differ from a real decode leg. Generated text is meaningless; pair
this connector with synthetic MTP acceptance and check accuracy separately.

Config::

    --kv-transfer-config '{"kv_connector": "TPUDecodeBenchConnector",
                           "kv_role": "kv_both",
                           "kv_connector_module_path":
                               "vllm_torchtpu.distributed.kv_transfer.decode_bench_connector"}'

``kv_connector_extra_config.claim_mode``:

* ``"all"`` (default): every request is claimed.
* ``"marked"``: only requests whose ``kv_transfer_params`` carry
  ``{"decode_bench_claim": true}`` are claimed; the rest compute normally. A
  benchmark primes the prefix cache with real KV through unmarked requests,
  then marks the measured ones, so the cached history is real and only what a
  P/D prefill leg would send (the uncached suffix) is left unwritten. With all
  of a context unwritten, attention sees no real keys and decode runs ~1.5x
  slower than with real KV (GLM-5.2, TP16), so ``"all"`` understates decode.

``cp_kv_cache_interleave_size`` is kept as configured under DCP (see
``vllm_torchtpu._patch_decode_bench_keeps_cp_interleave``); the pinned vLLM
would otherwise reset it to the block size, as it does for any connector.
"""

from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
    from vllm.forward_context import ForwardContext
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

CONNECTOR_NAME = "TPUDecodeBenchConnector"
CLAIM_MARKER = "decode_bench_claim"
_CLAIM_MODES = ("all", "marked")


class TPUDecodeBenchConnector(DecodeBenchConnector):
    """Upstream DecodeBench scheduling with a no-op worker (no KV writes)."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig",
    ):
        if getattr(vllm_config.model_config, "is_hybrid", False):
            # Mamba state is per-request, not block KV: leaving it unwritten
            # would start each request from another request's state. Upstream
            # DecodeBenchConnector installs synthetic state for hybrids.
            raise ValueError(
                f"{CONNECTOR_NAME} does not support hybrid (attention + Mamba) "
                "models; use DecodeBenchConnector."
            )
        super().__init__(vllm_config, role, kv_cache_config)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._claim_mode = str(extra.get("claim_mode", "all"))
        if self._claim_mode not in _CLAIM_MODES:
            raise ValueError(
                f"{CONNECTOR_NAME}: claim_mode must be one of {_CLAIM_MODES}, "
                f"got {self._claim_mode!r}"
            )
        logger.info(
            "%s: uncached prompt tokens (all but the last) are claimed as "
            "externally loaded and never computed or written (claim_mode=%s).",
            CONNECTOR_NAME,
            self._claim_mode,
        )

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        if self._claim_mode == "marked":
            params = getattr(request, "kv_transfer_params", None) or {}
            if not params.get(CLAIM_MARKER):
                return 0, False
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def register_kv_caches(self, kv_caches: "dict[str, torch.Tensor]") -> None:
        # The worker never touches KV, so it keeps no cache references.
        return

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        # Nothing to load: claimed blocks keep their zero-initialized or prior
        # contents. The runner may pass wait_for_completion/report_completion.
        return
