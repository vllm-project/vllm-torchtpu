# SPDX-License-Identifier: Apache-2.0
"""Model-local QSA, BF16 checkpoint loading and explicit typed cache binding."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import cast

import torch
from torch import nn
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

from vllm_torchtpu import envs
from vllm_torchtpu import envs as vllm_torchtpu_envs
from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend
from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import build_qsa_op, request_shard_applies
from vllm_torchtpu.layers.adapter.custom_ops.rows_update_op import update_rows_
from vllm_torchtpu.hixglue import logical_cols as _hxg_logical_cols, pad_cols as _hxg_pad_cols  # q38 HIX-GLUE SL
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.qwen4_exp.qsa_rank_operands import rank_operand

logger = init_logger(__name__)
# q38 DS-TP (log only): one main-KV DCP engagement line per process, from its first target QSA layer.
_Q38_DCP_LOGGED = False
# q38 O1s: one bind line per process (every TP rank). Not logger.info_once: vLLM's default
# scope "local" logs on the local first rank only, and the G2 reader needs ranks 0-7.
_Q38_O1S_LOGGED = False
# q38 L2: one min-rows gate line per process (every TP rank), only when the gate is set.
_Q38_O1S_GATE_LOGGED = False
# q38 X-D14-DRAFT-1 engagement marker (_log_draft_no_index_state): once per process.
_Q38_DRAFT1_LOGGED = False


def _log_o1s_gate(min_rows: int) -> None:
    global _Q38_O1S_GATE_LOGGED
    if _Q38_O1S_GATE_LOGGED or min_rows <= 0:
        return
    _Q38_O1S_GATE_LOGGED = True
    logger.info(
        "q38 O1s min-rows gate: request-sharded selection only on graphs of >= %d rows "
        "(VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS)",
        min_rows,
    )


# q38 L2 O1s-HOIST port 2: one line per process (every TP rank) when the decode indexer runs without the cond.
_Q38_NO_EXIT_LOGGED = False


def _log_no_early_exit() -> None:
    global _Q38_NO_EXIT_LOGGED
    if _Q38_NO_EXIT_LOGGED or not envs.VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT:
        return
    _Q38_NO_EXIT_LOGGED = True
    logger.info("q38 decode indexer early-exit cond: off (VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT)")


def _log_o1s_bind(tp_rank: int, tp_size: int, dcp: int, rows: int) -> None:
    global _Q38_O1S_LOGGED
    if _Q38_O1S_LOGGED:
        return
    _Q38_O1S_LOGGED = True
    logger.info(
        "q38 O1s request-sharded decode selection: on (TP rank %d of %d, main-KV DCP %d, "
        "rows per request %d)",
        tp_rank,
        tp_size,
        dcp,
        rows,
    )


def _log_draft_no_index_state(tp_rank: int) -> None:
    """q38 X-D14-DRAFT-1 engagement marker, once per process (one line per TP rank): an MTP
    draft layer bound to the target's selection takes the shared ops without indexer state
    (VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE). Absent when the flag is off or nothing is bound."""
    global _Q38_DRAFT1_LOGGED
    if _Q38_DRAFT1_LOGGED:
        return
    _Q38_DRAFT1_LOGGED = True
    logger.info("q38 DRAFT-1 no draft index state: on (TP rank %d)", tp_rank)


@dataclass(frozen=True)
class QSAMetadata:
    """The model adapter supplies one shared ragged geometry and group-native pages.

    ``context_lens`` is the logical length BEFORE this chunk, independent of
    the RoPE ``positions`` argument.
    Padded requests use state_slots=-1 and
    repeated query offsets. Stable slots are neither batch rows nor page IDs.
    Admission establishes <=8192 tokens/step and context within the model
    position capacity. RoPE covers [0, max_model_len). All real/scratch cache
    owners must be bound before forward.
    """

    query_start_loc: torch.Tensor
    context_lens: torch.Tensor
    state_slots: torch.Tensor
    main_block_table: torch.Tensor
    compressed_block_table: torch.Tensor


def make_rope_cache(max_positions: int, *, rotary_dim: int, theta: float):
    """BF16 cos/sin rows, padded to TPU's 128-column gather alignment."""
    # Match upstream RoPE frequencies on CPU before moving the BF16 cache.
    with torch.device("cpu"):
        rotary = RotaryEmbedding(
            head_size=rotary_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_positions,
            base=theta,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        cache = rotary.cos_sin_cache
        return torch.nn.functional.pad(cache, (0, (-rotary_dim) % 128))


QSA_BLOCK_SIZE = 128


def get_qsa_cache_specs(config, tp_size, cache_dtype):
    """Pure planner contract; model admission validates this geometry once."""
    fp8 = cache_dtype in ("fp8", "fp8_e4m3")
    head_dim = int(config.head_dim)
    index_dim = int(config.indexer_head_dim)
    ratio = int(config.indexer_compress_ratio)
    ring_rows = (ratio + 7) // 8 * 8
    # 8192 B/slot covers the observed T(2,128)/T(8,128) tag layouts for
    # 2..65 slots. Retain that placement allowance and budget larger raw
    # payloads explicitly; actual TPU allocation remains independently checked.
    ring_bytes = max(8192, ring_rows * (((index_dim + 127) // 128 * 128) * 2 + 8))
    main = FullAttentionSpec(
        block_size=QSA_BLOCK_SIZE,
        num_kv_heads=max(int(config.num_key_value_heads) // tp_size, 1),
        head_size=head_dim,
        dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
        kv_quant_mode=get_kv_quant_mode(cache_dtype),
    )
    auxiliary = {
        "compressed_k": MLAAttentionSpec(
            block_size=QSA_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=index_dim,
            state_content_bytes=((index_dim + 127) // 128 * 128) * 2,
            dtype=torch.bfloat16,
            tokens_per_state=ratio,
        ),
        "raw_ring": CircularBufferSpec(
            block_size=ratio,
            num_kv_heads=1,
            head_size=index_dim + 8 // 2,
            head_size_v=0,
            dtype=torch.bfloat16,
            page_size_padded=ring_bytes,
        ),
    }
    packing = 4 if fp8 else 2
    main = replace(
        main,
        num_head_slots=(2 * main.num_kv_heads + packing - 1) // packing,
        state_content_bytes=4 * ((head_dim + 127) // 128 * 128),
    )
    return main, auxiliary


class Qwen4ExpQSABackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3"]

    @staticmethod
    def get_name():
        return "QWEN4_EXP_TPU_QSA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("QSA is constructed by its model-owned layer")

    @staticmethod
    def get_builder_cls():
        # The model adapter enriches common group metadata with stable slots
        # and the *other group's* block table before constructing QSAMetadata.
        from vllm_torchtpu.layers.core.attention_metadata import (
            AttentionMetadataBuilder,
        )

        return AttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> list[MultipleOf | int]:
        return [QSA_BLOCK_SIZE]

    @staticmethod
    def get_min_page_size(vllm_config):
        return QSA_BLOCK_SIZE

    @classmethod
    def is_sparse(cls):
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class Qwen4ExpPooledQSABackend(PallasAttentionBackend):
    """Native pool geometry; QSA maps manager pages to 4-token RPA pages."""

    @staticmethod
    def get_name():
        return "QWEN4_EXP_TPU_POOLED_QSA"

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> list[MultipleOf | int]:
        return [MultipleOf(QSA_BLOCK_SIZE)]

    @staticmethod
    def get_min_page_size(vllm_config):
        from .pool_layout import pooled_cache_specs

        return pooled_cache_specs(vllm_config)[0].block_size

    @classmethod
    def supported_kv_cache_layouts(cls):
        from vllm.v1.kv_cache_interface import KVCacheLayout

        return (KVCacheLayout.BLHNC,)


class _Indexer(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        self.index_qk_proj = ReplicatedLinear(
            config.hidden_size,
            (config.indexer_n_heads + config.indexer_kv_heads)
            * config.indexer_head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=prefix + ".index_qk_proj",
            return_bias=False,
            disable_tp=True,
        )
        self.q_layernorm = GemmaRMSNorm(
            config.indexer_head_dim, eps=config.rms_norm_eps
        )
        self.k_layernorm = GemmaRMSNorm(
            config.indexer_head_dim, eps=config.rms_norm_eps
        )


_KV_SCALE_NAMES = {
    "k_proj.k_scale": "_k_scale",
    "k_proj.output_scale": "_k_scale",
    "attn.k_scale": "_k_scale",
    "attn._k_scale": "_k_scale",
    "k_scale": "_k_scale",
    "_k_scale": "_k_scale",
    "v_proj.v_scale": "_v_scale",
    "v_proj.output_scale": "_v_scale",
    "attn.v_scale": "_v_scale",
    "attn._v_scale": "_v_scale",
    "v_scale": "_v_scale",
    "_v_scale": "_v_scale",
}


_MAIN_KV_DCP_GROUPS: dict[int, object] = {}


def main_kv_dcp_group(size: int):
    """The TP rank groups (0, 1), (2, 3), ... that share a KV head, for the
    main-KV DCP pair combine (VLLM_TPU_QSA_MAIN_KV_DCP). Built once, by every
    rank, the first time a layer binds the pool."""
    if size not in _MAIN_KV_DCP_GROUPS:
        import torch.distributed as dist
        from vllm.distributed import get_tp_group
        from vllm.distributed.parallel_state import (
            get_world_group,
            init_model_parallel_group,
        )

        tp = get_tp_group()
        ranks = list(tp.ranks)
        groups = [ranks[i : i + size] for i in range(0, len(ranks), size)]
        world = int(dist.get_world_size())
        if world > len(ranks):
            # Every TP group of the world creates its pairs (collective call).
            groups = [list(range(i, i + size)) for i in range(0, world, size)]
        _MAIN_KV_DCP_GROUPS[size] = init_model_parallel_group(
            groups,
            get_world_group().local_rank,
            dist.get_backend(tp.device_group),
            group_name=f"qsa_main_kv_dcp{size}",
        )
    return _MAIN_KV_DCP_GROUPS[size]


class Qwen4ExpQSA(nn.Module, AttentionLayerBase):
    """Config-derived gated attention with a replicated MQA indexer.

    With VLLM_TPU_QSA_INDEXER_SHARD (pooled only) each TP rank scores 1/TP of
    the compressed keys and the ranks merge their candidates.

    Checkpoint names are relative to ``...self_attn.``. No quant_config enters
    these BF16 projections. Typed cache tensors stay owned by Task4's manager;
    the custom op donates state then copy_ makes writes visible to that owner.
    Profiling/capture uses explicitly bound scratch owners and shared metadata.
    """

    supports_dcp = False
    k_proj: ColumnParallelLinear
    v_proj: ColumnParallelLinear

    def __init__(
        self,
        *,
        config,
        tp_rank: int,
        tp_size: int,
        max_model_len: int,
        device=None,
        cache_dtype: str = "auto",
        prefix: str = "",
        pooled: bool = False,
        max_query_len: int = 1,
        max_num_seqs: int = 0,
    ):
        super().__init__()
        if not 0 <= tp_rank < tp_size:
            raise ValueError("QSA requires an in-range TP rank")
        self.tp_rank, self.tp_size, self.prefix = tp_rank, tp_size, prefix
        self.pooled = pooled
        # Tokens a decoding request carries per step (num_speculative_tokens
        # + 1): the pooled op routes steps within it to the decode kernels.
        self.max_query_len = max_query_len
        self.fp8_kv = cache_dtype in ("fp8", "fp8_e4m3")
        self.cache_dtype = cache_dtype
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(config.head_dim)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(self.total_num_kv_heads // tp_size, 1)
        self.index_head_dim = int(config.indexer_head_dim)
        self.index_n_heads = int(config.indexer_n_heads)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.token_topk = int(config.indexer_budget)
        self.max_selected_tokens = self.token_topk + self.compress_ratio - 1
        self.block_size = QSA_BLOCK_SIZE
        # Eight-row alignment is the current TPU allocation policy, not the
        # logical compression ratio. Every row holds history (position p in
        # lane p % ring_size), which keeps speculative rejections from
        # evicting accepted members of the partial compression group.
        self.ring_size = (self.compress_ratio + 7) // 8 * 8
        # Set by a module whose graph is traced once for every token bucket (the
        # MTP draft): ops must not choose their structure by the token count.
        self.single_graph = False
        kv_size = min(tp_size, self.total_num_kv_heads)
        kv_rank = tp_rank // max(tp_size // self.total_num_kv_heads, 1)
        with torch.device(device or "cpu"), set_default_torch_dtype(torch.bfloat16):
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                2 * self.total_num_heads * self.head_dim,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=prefix + ".q_proj",
                return_bias=False,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            for name in ("k_proj", "v_proj"):
                setattr(
                    self,
                    name,
                    ColumnParallelLinear(
                        self.hidden_size,
                        self.total_num_kv_heads * self.head_dim,
                        bias=False,
                        params_dtype=torch.bfloat16,
                        quant_config=None,
                        prefix=prefix + "." + name,
                        return_bias=False,
                        tp_rank=kv_rank,
                        tp_size=kv_size,
                    ),
                )
            # Upstream RowParallelLinear cannot take explicit TP geometry.
            # Reuse its constituent mechanisms: local unquantized linear and
            # the same TP reduction; loader slices columns exactly once.
            self.o_proj = ReplicatedLinear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=prefix + ".o_proj",
                return_bias=False,
                disable_tp=True,
            )
            self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.indexer = _Indexer(config, prefix + ".indexer")
        self.register_buffer(
            "rope_cache",
            make_rope_cache(
                max_model_len,
                rotary_dim=int(self.head_dim * config.partial_rotary_factor),
                theta=float(config.rope_parameters["rope_theta"]),
            ).to(device=device),
            persistent=False,
        )
        self.token_shard = self._token_shard_mode(tp_size, pooled, max_num_seqs)
        # VLLM_TPU_QSA_MAIN_KV_DCP: target layers split their main KV over TP
        # rank pairs sharing a KV head; the MTP draft keeps whole pages unless
        # VLLM_TPU_QSA_MTP_KV_DCP splits it the same way.
        self.main_kv_dcp = 1
        dcp = int(envs.VLLM_TPU_QSA_MAIN_KV_DCP or 0)
        if dcp > 1 and pooled and ("mtp" not in prefix or envs.VLLM_TPU_QSA_MTP_KV_DCP):
            assert dcp in (2, 4) and tp_size % dcp == 0, "main-KV DCP groups of 2 or 4"
            self.main_kv_dcp = dcp
            self.register_buffer(
                "dcp_rank",
                # q38 L2 O1s-HOIST (MM DSTOCK-B H-A): one shared tensor per rank when
                # VLLM_TPU_QSA_SHARED_RANK_OPERANDS=1 (qsa_rank_operands.py).
                rank_operand("dcp_rank", tp_rank % dcp, device),
                persistent=False,
            )
        # q38 DS-TP (log only, no behavior change): the decode-sharding G2 reader requires this line from every rank.
        global _Q38_DCP_LOGGED
        if not _Q38_DCP_LOGGED and "mtp" not in prefix:
            _Q38_DCP_LOGGED = True
            logger.info(
                "q38 QSA main-KV DCP: size=%d tp_rank=%d tp_size=%d dcp_rank=%d pooled=%s",
                self.main_kv_dcp,
                tp_rank,
                tp_size,
                tp_rank % self.main_kv_dcp,
                bool(pooled),
            )
        if self.token_shard:
            # The rank as an operand of the token-sharded indexer, so every
            # rank traces the same program.
            self.register_buffer(
                "token_shard_index",
                rank_operand("token_shard_index", tp_rank, device),
                persistent=False,
            )
        if "mtp" not in prefix:
            _log_no_early_exit()
        # q38 O1s (VLLM_TPU_QSA_DECODE_REQUEST_SHARD): under main-KV DCP each
        # TP rank indexes its share of a decode step's requests, not all of
        # them. Target layers only (_decode_request_shard_mode).
        self.decode_request_shard = self._decode_request_shard_mode(
            self.main_kv_dcp, tp_size, prefix
        )
        if self.decode_request_shard:
            self.register_buffer(
                "request_shard_index",
                rank_operand("request_shard_index", tp_rank, device),
                persistent=False,
            )
            # q38 L2 D21 min-rows gate: read once here (never inside the
            # fullgraph Dynamo capture); 0 = every decode-shaped graph.
            self.request_shard_min_rows = int(
                envs.VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS
            )
        with torch.device(device or "cpu"):
            set_default_quant_scales(self, register_buffer=True)
        self.cache_owners: tuple[torch.Tensor, ...] = ()
        self.compressed_cache_name = prefix + ".compressed_k"
        self.main_cache_name = prefix
        self._loaded = set()
        self._op = partial(
            build_qsa_op(),
            compress_ratio=self.compress_ratio,
            token_topk=self.token_topk,
            norm_eps=float(self.q_norm.variance_epsilon),
            rotary_dim=int(self.head_dim * config.partial_rotary_factor),
        )

    @staticmethod
    def _decode_request_shard_mode(main_kv_dcp, tp_size, prefix):
        """q38 O1s: whether this layer takes the request-sharded decode
        selection. Target QSA layers under main-KV DCP with TP > 1 only. The
        MTP draft never does: it attends the target's selection (or its own
        rows indexer), and with VLLM_TPU_QSA_MTP_KV_DCP its main KV is split
        too, so without this check it would register the request-shard
        operand and evaluate the token-count routing inside its single graph,
        which is traced once for every bucket (a guard there refuses buckets
        at engine init, as shape_variants does for any untraced bucket)."""
        return bool(
            envs.VLLM_TPU_QSA_DECODE_REQUEST_SHARD
            and main_kv_dcp > 1
            and tp_size > 1
            and "mtp" not in prefix
        )

    @staticmethod
    def _token_shard_mode(tp_size, pooled, max_num_seqs):
        """Whether the SP graph uses the token-sharded indexer, for all of
        its buckets: one bucket without the all-gather would move the graph's
        collective split points, and the buckets share traced artifacts. Its
        smallest bucket's shard must keep the ragged prefill indexer (more
        rows than request slots)."""
        if not (envs.VLLM_TPU_QSA_INDEXER_TOKEN_SHARD and tp_size > 1):
            return False
        from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp

        shard = sp.min_tokens(tp_size) // tp_size
        active = pooled and sp.enabled(tp_size) and shard > max_num_seqs
        logger.info_once(
            "QSA token-sharded prefill indexer: %s (pooled=%s, prefill SP=%s, "
            "smallest SP shard %d rows, max_num_seqs %d)",
            "on" if active else "OFF",
            pooled,
            sp.enabled(tp_size),
            shard,
            max_num_seqs,
        )
        return active

    def get_attn_backend(self):
        return Qwen4ExpPooledQSABackend if self.pooled else Qwen4ExpQSABackend

    def bind_pool(
        self,
        pool,
        compressed_owner: tuple[str, int] | None = None,
        main_owner: tuple[str, int] | None = None,
    ):
        """Bind the unified pool.

        ``compressed_owner`` names the cache group whose block table addresses
        this layer's compressed keys and the layer's slot in that group's
        pages (``Qwen4ExpPoolLayout.compressed_owner``); by default the layer
        owns ``<prefix>.compressed_k`` alone. ``main_owner`` likewise names
        the group of paired main KV pages and the layer's head-pair slot
        (``Qwen4ExpPoolLayout.main_owner``); by default the layer owns whole
        rows of ``<prefix>`` pages.
        """
        from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
            build_pooled_qsa_op,
            build_pooled_qsa_rows_op,
        )

        name, slot = compressed_owner or (self.prefix + ".compressed_k", 0)
        main_name, main_slot = main_owner or (self.prefix, -1)
        self.cache_owners = (pool,)
        self.compressed_cache_name = name
        self.main_cache_name = main_name
        # q38 L2 D73MK: D73's startup line, once per process (every rank), flags on or off (log only).
        from vllm_torchtpu.kernels.qwen4_exp.fused_kv_write import log_d73_state

        log_d73_state(self.tp_rank)
        keywords = dict(self._op.keywords)
        keywords.pop("compressed_slot", None)
        keywords.pop("main_slot", None)
        keywords.pop("max_query_len", None)
        if self.max_query_len > 1:
            keywords["max_query_len"] = self.max_query_len
        if slot:
            keywords["compressed_slot"] = slot
        if main_slot >= 0:
            keywords["main_slot"] = main_slot
        self._op = partial(build_pooled_qsa_op(), **keywords)
        self._rows_op = partial(build_pooled_qsa_rows_op(), **keywords)
        self._shared_ops = None
        # q38 X-D14-DRAFT-1 (VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE): the MTP draft's shared ops
        # keep no indexer state (main K/V write only, no index_qk_proj).
        self._shared_no_index_state = self._draft_no_index_state(self.prefix)
        no_index = {"index_state": False} if self._shared_no_index_state else {}
        if vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_SHARED_SELECTION:
            from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
                build_pooled_qsa_shared_op,
            )

            # (all-token, rows) ops attending another layer's selection.
            self._shared_ops = (
                partial(build_pooled_qsa_shared_op(False, **no_index), **keywords),
                partial(build_pooled_qsa_shared_op(True, **no_index), **keywords),
            )
            if no_index and getattr(self, "_selection_source", None) is not None:
                # Bound at load (Qwen4ExpMTP.bind_shared_selection), before the pool.
                _log_draft_no_index_state(self.tp_rank)
        # The select op (`select_for_kernel`) is built here, outside the traced
        # forward (jax_op inspects the step signature, which Dynamo cannot trace),
        # and only when the megakernel is enabled: default serving never
        # registers it.
        self._select_keywords = keywords
        self._select_op = None
        raw_mk = str(vllm_torchtpu_envs.VLLM_TPU_MEGAKERNEL or "").strip().lower()
        if raw_mk and raw_mk not in ("0", "false", "off"):
            from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
                build_pooled_qsa_select_op,
            )

            self._select_op = partial(build_pooled_qsa_select_op(), **keywords)
        self.main_slot = max(main_slot, 0)
        self._shard_op = None
        self._shard_select_op = None
        self._token_shard_op = None
        if getattr(self, "token_shard", False):
            from vllm.distributed import get_tp_group

            from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
                build_pooled_qsa_token_shard_op,
            )

            group = get_tp_group()
            assert int(group.world_size) == self.tp_size
            assert int(group.rank_in_group) == self.tp_rank
            self._token_shard_op = partial(
                build_pooled_qsa_token_shard_op(group),
                kv_heads=self.num_kv_heads,
                **keywords,
            )
        self._dcp_op = None
        self._token_shard_dcp_op = None
        self._request_shard_dcp_op = None
        self._request_plain_dcp_op = None
        if self.main_kv_dcp > 1:
            from vllm.distributed import get_tp_group

            from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
                build_pooled_qsa_dcp_op,
                build_pooled_qsa_token_shard_dcp_op,
            )

            pair = main_kv_dcp_group(self.main_kv_dcp)
            self._dcp_op = partial(
                build_pooled_qsa_dcp_op(pair, self.main_kv_dcp), **keywords
            )
            compressed = bool(envs.VLLM_TPU_QSA_COMPRESSED_DCP)
            if compressed:
                from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
                    build_pooled_qsa_compressed_dcp_op,
                )

                # VLLM_TPU_QSA_COMPRESSED_DCP: owned-group scoring, pair merge.
                # The megakernel's whole-step writer still writes whole rows.
                assert self._select_op is None, (
                    "VLLM_TPU_QSA_COMPRESSED_DCP and VLLM_TPU_MEGAKERNEL are exclusive"
                )
                self._dcp_op = partial(
                    build_pooled_qsa_compressed_dcp_op(pair, self.main_kv_dcp),
                    kv_heads=self.num_kv_heads,
                    **keywords,
                )
            if "mtp" in self.prefix:
                from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
                    build_pooled_qsa_draft_dcp_op,
                )

                # VLLM_TPU_QSA_MTP_KV_DCP: the draft's rows and shared-selection
                # ops over its DCP main KV.
                def draft(shared, with_rows):
                    return partial(
                        build_pooled_qsa_draft_dcp_op(
                            pair,
                            self.main_kv_dcp,
                            shared=shared,
                            with_rows=with_rows,
                            **(no_index if shared else {}),
                        ),
                        dcp_rank=self.dcp_rank,
                        **({"compressed_dcp": True} if compressed and shared else {}),
                        **keywords,
                    )

                # Under compressed DCP the draft attends the target's selection
                # only: its own rows indexer would need the pair merge.
                assert not compressed or self._shared_ops is not None, (
                    "VLLM_TPU_QSA_COMPRESSED_DCP needs VLLM_TPU_QSA_DRAFT_SHARED_SELECTION"
                )
                self._rows_op = draft(False, True)
                if self._shared_ops is not None:
                    self._shared_ops = (draft(True, False), draft(True, True))
            if getattr(self, "token_shard", False):
                build = build_pooled_qsa_token_shard_dcp_op
                if compressed:
                    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
                        build_pooled_qsa_token_shard_compressed_dcp_op as build,
                    )
                self._token_shard_dcp_op = partial(
                    build(get_tp_group(), pair, self.main_kv_dcp),
                    kv_heads=self.num_kv_heads,
                    **keywords,
                )
            if getattr(self, "decode_request_shard", False):
                from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
                    build_pooled_qsa_request_plain_dcp_op,
                    build_pooled_qsa_request_shard_dcp_op,
                )

                # q38 O1s: the replicated indexer's selection, scored once
                # per request across the TP group instead of on every rank.
                assert not compressed, (
                    "VLLM_TPU_QSA_DECODE_REQUEST_SHARD and VLLM_TPU_QSA_COMPRESSED_DCP "
                    "are exclusive"
                )
                group = get_tp_group()
                assert int(group.world_size) == self.tp_size
                assert int(group.rank_in_group) == self.tp_rank
                self._request_shard_dcp_op = partial(
                    build_pooled_qsa_request_shard_dcp_op(group, pair, self.main_kv_dcp),
                    kv_heads=self.num_kv_heads,
                    **keywords,
                )
                # The graphs that do not take the request shard (prefill-shaped
                # steps) still take request_shard_index, so every compile range
                # of the module's graph has one input list (shape_variants).
                self._request_plain_dcp_op = partial(
                    build_pooled_qsa_request_plain_dcp_op(pair, self.main_kv_dcp),
                    kv_heads=self.num_kv_heads,
                    **keywords,
                )
                _log_o1s_bind(self.tp_rank, self.tp_size, self.main_kv_dcp, self.max_query_len)
                _log_o1s_gate(int(getattr(self, "request_shard_min_rows", 0)))
        # With VLLM_TPU_QSA_INDEXER_SHARD_SKIP_SP a single-graph module (the
        # MTP draft, traced once for every bucket including prefill chunks)
        # keeps the replicated indexer: it cannot pick per bucket.
        if (
            envs.VLLM_TPU_QSA_INDEXER_SHARD
            and self.tp_size > 1
            and not (self.single_graph and envs.VLLM_TPU_QSA_INDEXER_SHARD_SKIP_SP)
            # The key-sharded ops attend whole main-KV pages; under DCP they
            # would take a DCP draft's rows path.
            and self.main_kv_dcp == 1
        ):
            from vllm.distributed import get_tp_group

            from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
                build_pooled_qsa_shard_op,
            )

            group = get_tp_group()
            assert int(group.world_size) == self.tp_size
            assert int(group.rank_in_group) == self.tp_rank
            # The all-to-all merge is chosen by the token count while tracing;
            # a single-graph module (the MTP draft) always all-gathers.
            a2a_min_tokens = envs.VLLM_TPU_QSA_INDEXER_SHARD_A2A_MIN_TOKENS
            if self.single_graph:
                a2a_min_tokens = 0
            self._shard_op = partial(
                build_pooled_qsa_shard_op(group, a2a_min_tokens),
                kv_heads=self.num_kv_heads,
                **keywords,
            )
            if self._select_op is not None:
                # The megakernel's select op, with the indexer sharded too.
                from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import (
                    build_pooled_qsa_shard_select_op,
                )

                self._shard_select_op = partial(
                    build_pooled_qsa_shard_select_op(group, a2a_min_tokens),
                    **keywords,
                )

    @staticmethod
    def _draft_no_index_state(prefix: str) -> bool:
        """q38 X-D14-DRAFT-1: VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE takes effect on the MTP
        draft's layers under VLLM_TPU_QSA_DRAFT_SHARED_SELECTION only. The draft then
        attends the target's carried selection on every pass once bound (attend_with_indices),
        so nothing reads its compressed keys, raw ring or tags; the target keeps its own."""
        return bool(
            vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE
            and vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_SHARED_SELECTION
            and "mtp" in prefix
        )

    def get_kv_cache_spec(self, vllm_config):
        return get_qsa_cache_specs(self.config, self.tp_size, self.cache_dtype)[0]

    def get_auxiliary_cache_specs(self):
        return get_qsa_cache_specs(self.config, self.tp_size, self.cache_dtype)[1]

    def get_cache_owner_layouts(self, pages, slots):
        """Typed owner order shared by allocation, binding and the custom op."""
        auxiliary = (
            (
                "compressed_k",
                (
                    pages,
                    self.block_size // self.compress_ratio,
                    1,
                    (self.index_head_dim + 127) // 128 * 128,
                ),
                torch.bfloat16,
                0,
            ),
            (
                "raw_ring",
                (slots, self.ring_size, self.index_head_dim),
                torch.bfloat16,
                0,
            ),
            ("rope_tags", (slots, self.ring_size), torch.int64, -1),
        )
        packing = 4 if self.fp8_kv else 2
        return (
            (
                "head_packed_kv",
                (
                    pages,
                    self.block_size,
                    (2 * self.num_kv_heads + packing - 1) // packing,
                    packing,
                    (self.head_dim + 127) // 128 * 128,
                ),
                torch.float8_e4m3fn if self.fp8_kv else torch.bfloat16,
                0,
            ),
            *auxiliary,
        )

    def bind_cache_owners(self, kv_cache: Sequence[torch.Tensor]) -> None:
        if len(kv_cache) != 4:
            raise ValueError("QSA requires main KV, compressed K, ring and tags")
        layouts = self.get_cache_owner_layouts(
            kv_cache[0].shape[0], kv_cache[2].shape[0]
        )
        for owner, (role, shape, dtype, _) in zip(kv_cache, layouts):
            if owner.shape != shape or owner.dtype != dtype:
                raise ValueError(f"QSA {role} requires shape={shape}, dtype={dtype}")
        self.cache_owners = tuple(kv_cache)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        global_shapes = {
            "q_proj.weight": (
                2 * self.total_num_heads * self.head_dim,
                self.hidden_size,
            ),
            "k_proj.weight": (
                self.total_num_kv_heads * self.head_dim,
                self.hidden_size,
            ),
            "v_proj.weight": (
                self.total_num_kv_heads * self.head_dim,
                self.hidden_size,
            ),
            "o_proj.weight": (self.hidden_size, self.total_num_heads * self.head_dim),
            "q_norm.weight": (self.head_dim,),
            "k_norm.weight": (self.head_dim,),
            "indexer.index_qk_proj.weight": (
                (self.index_n_heads + 1) * self.index_head_dim,
                self.hidden_size,
            ),
            "indexer.q_layernorm.weight": (self.index_head_dim,),
            "indexer.k_layernorm.weight": (self.index_head_dim,),
        }
        loaded = set()
        for name, weight in weights:
            if name in _KV_SCALE_NAMES:
                target = _KV_SCALE_NAMES[name]
                if weight.numel() != 1:
                    raise ValueError(f"QSA {name} requires one per-tensor scale")
                value = float(weight.detach().float().cpu().item())
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"QSA {name} requires a finite positive scale")
                if target in self._loaded or target in loaded:
                    raise ValueError(f"Duplicate QSA checkpoint scale: {target}")
                # Keep the serialized buffer and host scalar consistent. The
                # compiled kernel specializes on this immutable layer scale.
                getattr(self, target).fill_(value)
                getattr(self, target + "_cpu").fill_(value)
                setattr(self, target + "_float", value)
                loaded.add(target)
                continue
            if (
                name not in global_shapes
                or weight.shape != global_shapes[name]
                or weight.dtype != torch.bfloat16
            ):
                raise ValueError(f"Invalid BF16 QSA checkpoint tensor: {name}")
            parameter = parameters[name]
            with torch.no_grad():
                if name in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                    cast(BasevLLMParameter, parameter).weight_loader(parameter, weight)
                elif name == "o_proj.weight":
                    width = self.num_heads * self.head_dim
                    self.o_proj.weight_loader(
                        parameter,
                        weight[:, self.tp_rank * width : (self.tp_rank + 1) * width],
                    )
                else:
                    parameter.copy_(weight)
            loaded.add(name)
        self._loaded.update(loaded)
        return loaded

    def finalize_weights(self):
        missing = set(dict(self.named_parameters())) - self._loaded
        if missing:
            raise ValueError(f"Missing QSA checkpoint tensors: {sorted(missing)}")

    def forward_with_indices(
        self,
        hidden,
        positions,
        metadata: QSAMetadata,
        rows=None,
        reduce=True,
        replicated_indexer=False,
        token_shard=False,
        shared=None,
    ):
        """Attention output and selected tokens.

        ``reduce=False`` returns this TP rank's partial output projection;
        the caller sums it across TP.

        ``rows`` (int, -1 for none) limits both to those token rows; the
        caches are still written for every token. Pooled QSA attends only
        the rows; the dedicated-cache path attends all tokens and picks.

        ``replicated_indexer`` (static per graph): keep the replicated indexer
        even when the TP-sharded one is built. ``token_shard`` (static per
        graph, the prefill sequence-parallel backbone): index this rank's
        token shard of the step and all-gather the selections, when built.
        """
        output, selected = self.attend_with_indices(
            hidden, positions, metadata, rows, replicated_indexer, token_shard, shared
        )
        output = self.o_proj(output)
        if reduce and self.tp_size > 1:
            # q38 AR-FIX (VLLM_TPU_DECODE_TP_AR, default off): the same kernel as sequence_parallel.reduce.
            from vllm_torchtpu.layers.adapter import tp_ar_decode

            if tp_ar_decode.use_kernel(output):
                output = tp_ar_decode.kernel_all_reduce(output)
            else:
                output = tensor_model_parallel_all_reduce(output)
        return output, selected

    def attend(self, hidden, positions, metadata: QSAMetadata, request_shard: bool = True):
        """Gated attention output before ``o_proj`` (this rank's heads).

        ``request_shard=False`` (q38 L2 O1SD14): see ``attend_with_indices``."""
        return self.attend_with_indices(
            hidden, positions, metadata, request_shard=request_shard
        )[0]

    def attend_with_indices(
        self,
        hidden,
        positions,
        metadata: QSAMetadata,
        rows=None,
        replicated_indexer=False,
        token_shard=False,
        shared=None,
        request_shard=True,
    ):
        """``forward_with_indices`` without the output projection and all-reduce.

        ``shared`` (pooled, VLLM_TPU_QSA_DRAFT_SHARED_SELECTION): another QSA
        layer's ``selected`` for the attended rows (one per ``rows`` entry);
        it replaces this layer's indexer selection.

        ``request_shard=False`` (q38 L2 O1SD14): this graph never takes the O1s
        request-sharded selection, decided by the caller as a python constant;
        the token count is not read for it (``_request_shard_applies``). The
        Qwen3.8 whole-step graph passes it: vLLM traces that graph once and
        compiles it for every whole-step bucket outside shape_variants'
        warmup, so a token-count branch there refuses every bucket the first
        trace's guards exclude (SAB-D14: ShapeSpecializationError at buckets
        48 / 64 with VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS=48).
        """
        assert len(self.cache_owners) == (1 if self.pooled else 4), (
            "Task4 must bind real or profiling scratch QSA owners"
        )
        if self.pooled:
            per_layer = cast(
                dict[str, AttentionMetadata], get_forward_context().attn_metadata
            )
            metadata = replace(
                metadata,
                state_slots=cast(
                    torch.Tensor, per_layer[self.prefix + ".raw_ring"].block_tables
                ),
                main_block_table=cast(
                    torch.Tensor, per_layer[self.main_cache_name].block_tables
                ),
                compressed_block_table=cast(
                    torch.Tensor, per_layer[self.compressed_cache_name].block_tables
                ),
            )
        pick_rows = rows is not None and not self.pooled
        op = self._op if rows is None or pick_rows else self._rows_op
        if (
            self.pooled
            and getattr(self, "_shard_op", None) is not None
            and not replicated_indexer
        ):
            # The TP-sharded indexer; it also takes an optional rows operand.
            op = self._shard_op
        token_shard_op = getattr(self, "_token_shard_op", None)
        if token_shard and rows is None and token_shard_op is not None:
            # Takes precedence over the key-sharded indexer in this graph.
            op = partial(token_shard_op, shard_index=self.token_shard_index)
        if getattr(self, "_dcp_op", None) is not None and rows is None:
            # VLLM_TPU_QSA_MAIN_KV_DCP: this rank's main-KV rows, pair-combined.
            if token_shard and self._token_shard_dcp_op is not None:
                op = partial(
                    self._token_shard_dcp_op,
                    shard_index=self.token_shard_index,
                    dcp_rank=self.dcp_rank,
                )
            elif self._request_shard_applies(hidden, metadata, request_shard):
                # q38 O1s: request-sharded selection on decode-shaped steps.
                op = partial(
                    self._request_shard_dcp_op,
                    shard_index=self.request_shard_index,
                    dcp_rank=self.dcp_rank,
                )
            elif getattr(self, "_request_plain_dcp_op", None) is not None:
                # q38 O1s, prefill-shaped steps: every row on every rank, with
                # request_shard_index an (unused) operand where the request
                # shard takes it. _precompile_backbone traces the graph once
                # per bucket group (shape_variants) and hands compiled buckets
                # between the traces only when their inputs match (G2-O1s
                # j-fa24a1ff failed engine init with ShapeSpecializationError
                # without this).
                op = partial(
                    self._request_plain_dcp_op,
                    shard_index=self.request_shard_index,
                    dcp_rank=self.dcp_rank,
                )
            else:
                op = partial(self._dcp_op, dcp_rank=self.dcp_rank)
        shared_ops = getattr(self, "_shared_ops", None)
        source = getattr(self, "_selection_source", None)
        if shared is None and source is not None and shared_ops is not None:
            # An MTP draft bound to the target (bind_shared_selection): step 0
            # attends each request's last verified row, which is that row of
            # the target's step; later steps (one row per request, in the same
            # order) reuse what step 0 took.
            carried = cast(torch.Tensor, getattr(self, "_selection_rows"))
            if rows is not None:
                shared = source[rows.clamp_min(0)]
                update_rows_(carried, shared)
            else:
                shared = carried[: hidden.shape[0]]
            shared = _hxg_logical_cols(shared, self.max_selected_tokens)  # q38 HIX-GLUE SL: no-op when widths match
        use_shared = shared is not None and shared_ops is not None and not pick_rows
        if use_shared:
            op = shared_ops[rows is not None]
        # q38 X-D14-DRAFT-1: the no-index-state shared ops ignore their index_qk operand,
        # so the draft skips index_qk_proj and passes ``hidden`` in its place.
        no_index = use_shared and getattr(self, "_shared_no_index_state", False)
        output, selected = op(
            self.q_proj(hidden),
            self.k_proj(hidden),
            self.v_proj(hidden),
            hidden if no_index else self.indexer.index_qk_proj(hidden),
            self.q_norm.weight,
            self.k_norm.weight,
            self.indexer.q_layernorm.weight,
            self.indexer.k_layernorm.weight,
            positions,
            self.rope_cache,
            *self.cache_owners,
            metadata.query_start_loc,
            metadata.context_lens,
            metadata.state_slots,
            metadata.main_block_table,
            metadata.compressed_block_table,
            *(() if rows is None or pick_rows else (rows,)),
            *((shared,) if use_shared else ()),
            **(
                {"k_scale": self._k_scale_float, "v_scale": self._v_scale_float}
                if self.fp8_kv
                else {}
            ),
        )
        if pick_rows:
            keep = (rows >= 0)[:, None]
            picked = rows.clamp_min(0)
            output = torch.where(keep, output[picked], 0)
            selected = torch.where(keep, selected[picked], -1)
        sink = getattr(self, "_selection_sink", None)
        if sink is not None and rows is None:
            # The target's selection for the MTP draft (bind_shared_selection).
            # Both writes go through the donating row update: a slice copy_
            # into the buffer copied all of it every step (~42 us at 8192 rows).
            update_rows_(sink, _hxg_pad_cols(selected, sink.shape[-1]))  # q38 HIX-GLUE SL: no-op pad when widths match
        return output, selected

    def _request_shard_applies(
        self, hidden, metadata: QSAMetadata, request_shard: bool = True
    ) -> bool:
        """q38 O1s: whether this graph takes the request-sharded selection
        (static per graph: decode-shaped, tokens divisible by TP).
        ``request_shard=False`` (q38 L2 O1SD14): never, decided before any shape
        is read, so the trace keeps no guard on the token count."""
        if getattr(self, "_request_shard_dcp_op", None) is None:
            return False
        if not request_shard:  # q38 L2 O1SD14: static, before hidden.shape
            return False
        # No import and no logging here (inside the fullgraph Dynamo capture); the op's JAX
        # step logs each traced shape.
        return request_shard_applies(
            int(hidden.shape[0]),
            int(metadata.context_lens.shape[0]),
            self.tp_size,
            self.max_query_len,
            int(getattr(self, "request_shard_min_rows", 0)),
        )

    def select_for_kernel(self, hidden, positions, metadata: QSAMetadata):
        """Cache writes and the indexer only; the attention runs in the Qwen3.8
        segment megakernel (``QSASparseAttnBlock``). Returns its extra inputs:
        ``[pool, q, gate, pages, kv_lens, head-pair offset, (k, v) scales]``."""
        if getattr(self, "_selection_sink", None) is not None:
            # The select op returns no `selected`, so the MTP draft's shared selection
            # (bind_shared_selection) would go stale on in-kernel steps. The served MK paths
            # keep this layer on `attend` (segment_call.attention_rows unset); enabling
            # in-kernel attention on the sink layer needs the select op to return `selected`.
            raise NotImplementedError(
                "select_for_kernel on the shared-selection sink layer: the draft's selection "
                "would not be written (VLLM_TPU_QSA_DRAFT_SHARED_SELECTION)"
            )
        assert self.pooled, "in-kernel QSA attention reads the unified pool"
        per_layer = cast(dict[str, AttentionMetadata], get_forward_context().attn_metadata)
        metadata = replace(
            metadata,
            state_slots=cast(torch.Tensor, per_layer[self.prefix + ".raw_ring"].block_tables),
            main_block_table=cast(torch.Tensor, per_layer[self.main_cache_name].block_tables),
            compressed_block_table=cast(
                torch.Tensor, per_layer[self.compressed_cache_name].block_tables
            ),
        )
        scales = (
            {"k_scale": self._k_scale_float, "v_scale": self._v_scale_float}
            if self.fp8_kv
            else {}
        )
        if self._select_op is None:
            raise RuntimeError(
                "select_for_kernel needs VLLM_TPU_MEGAKERNEL set at construction "
                "(the select op is not built lazily inside forward)"
            )
        select = self._select_op
        if getattr(self, "_shard_select_op", None) is not None:
            select = self._shard_select_op
        q, gate, pages, kv_lens = select(
            self.q_proj(hidden),
            self.k_proj(hidden),
            self.v_proj(hidden),
            self.indexer.index_qk_proj(hidden),
            self.q_norm.weight,
            self.k_norm.weight,
            self.indexer.q_layernorm.weight,
            self.indexer.k_layernorm.weight,
            positions,
            self.rope_cache,
            *self.cache_owners,
            metadata.query_start_loc,
            metadata.context_lens,
            metadata.state_slots,
            metadata.main_block_table,
            metadata.compressed_block_table,
            **scales,
        )
        dev = hidden.device
        offset = torch.full((1,), self.main_slot, dtype=torch.int32, device=dev)
        kv_scales = torch.tensor(
            [scales.get("k_scale", 1.0), scales.get("v_scale", 1.0)],
            dtype=torch.float32,
            device=dev,
        )
        return [self.cache_owners[0], q, gate, pages, kv_lens, offset, kv_scales]

    def forward(self, hidden, positions, metadata: QSAMetadata):
        return self.forward_with_indices(hidden, positions, metadata)[0]
