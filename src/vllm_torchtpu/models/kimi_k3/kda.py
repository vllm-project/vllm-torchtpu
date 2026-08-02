# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import os

import jax
import jax.numpy as jnp
import torch
from einops import rearrange
from torch import nn
from torch_tpu._internal import pallas
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.utils import divide
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator,
    is_conv_state_dim_first)
from vllm.model_executor.utils import set_weight_attrs

from vllm_torchtpu.kernels.kimi_k3 import chunk_kda, decode_kda, naive_kda
from vllm_torchtpu.layers.vllm.custom_ops.kda_attention_op import \
    build_kimi_sconv_op

__all__ = ["KdaMetadata", "KimiK3DeltaAttention", "GatedRMSNormSigmoid"]

_KDA_GATE_LOGBOUND_MIN = -5.0

_USE_NAIVE_KDA_ENV = "TPU_VLLM_USE_NAIVE_KDA"


def use_naive_kda() -> bool:
    """Return whether the plain-XLA recurrence is enabled."""
    return os.environ.get(_USE_NAIVE_KDA_ENV, "0") == "1"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class KdaMetadata:
    """Metadata consumed by the TPU KDA layer.

    Attributes:
        num_prefills / num_decodes: batch composition. A batch is treated as
            prefill if ``num_prefills > 0``; mixed batches are rejected, since
            resolving them needs the token-index split upstream gets from
            ``spec_token_indx`` / ``non_spec_token_indx``.
        query_start_loc: ``[num_seqs + 1]`` cu_seqlens over the token axis.
        state_indices: ``[num_seqs]`` slot per sequence in the state pool.
        has_initial_state: ``[num_seqs]`` bool -- whether the sequence carries
            recurrent/conv state in from a previous step.
        segment_ids: optional precomputed ``[1, num_tokens]`` segment IDs for
            chunked prefill. Constructing these inside the layer is a temporary
            fallback and is not graph-friendly.
        num_actual_tokens: tokens before padding.
    """
    num_prefills: int
    num_decodes: int
    query_start_loc: torch.Tensor
    state_indices: torch.Tensor
    has_initial_state: torch.Tensor
    num_actual_tokens: int
    num_spec_decodes: int = 0
    segment_ids: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Gated RMSNorm
# ---------------------------------------------------------------------------
class GatedRMSNormSigmoid(nn.Module):
    """Plain-torch equivalent of upstream ``FusedRMSNormGated``."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_f * torch.rsqrt(variance + self.eps)
        x_normed = x_normed * self.weight.float()
        g_f = g.float()
        out = x_normed * torch.sigmoid(g_f)
        return out.to(x.dtype)


def _causal_conv1d_update(
        x: torch.Tensor,  # [num_seqs, dim]  one token per sequence
        conv_state: torch.Tensor,  # [num_slots, state_len, dim]
        weight: torch.Tensor,  # [dim, kernel_size]
        bias: torch.Tensor | None,
        state_indices: torch.Tensor,  # [num_seqs]
) -> torch.Tensor:
    """Decode conv1d + SiLU: roll the window, apply, store back."""
    _dim, kernel_size = weight.shape
    slots = state_indices.to(torch.long)
    # [num_seqs, w - 1, dim] -> window of the last w tokens including this one.
    prev = conv_state[slots][:, -(kernel_size - 1):, :].to(x.dtype)
    window = torch.cat([prev, x.unsqueeze(1)], dim=1)  # [n, w, dim]
    y = (window.transpose(1, 2) * weight.unsqueeze(0)).sum(-1)  # [n, dim]
    if bias is not None:
        y = y + bias
    # Store the rolled window back into the pool.
    new_state = conv_state[slots].clone()
    new_state[:, :-1, :] = new_state[:, 1:, :].clone()
    new_state[:, -1, :] = x.to(conv_state.dtype)
    conv_state[slots] = new_state
    return torch.nn.functional.silu(y).to(x.dtype)


# ---------------------------------------------------------------------------
# JAX side: the two kernel call sites
# ---------------------------------------------------------------------------
def _chunk_kda_adapter(
    q: jax.Array,  # [1, T, H, D]
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,  # raw per-channel gate, [1, T, H, D]
    beta: jax.Array,  # [1, T, H]
    pool: jax.Array,  # [num_slots, H, K, V]
    a_log: jax.Array,  # [H]
    dt_bias: jax.Array,  # [H * K]
    segment_ids: jax.Array,  # [1, T]
    state_indices: jax.Array,  # [num_seqs]
    has_initial_state: jax.Array,  # [num_seqs]
    *,
    lower_bound: float | None,
    n_max: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunked KDA over a packed prefill batch; returns (out, updated pool).

    Layout bridge: the layer works in vLLM's ``[1, T, H, D]`` while the chunked
    kernel wants head-major ``[H, B, T, D]``.
    """

    def hm(x):  # [1, T, H, D] -> [H, 1, T, D]
        return jnp.transpose(x, (2, 0, 1, 3))

    init = jnp.take(pool, state_indices, axis=0)  # [N, H, V, K]
    init = jnp.swapaxes(init, -1, -2)  # -> [N, H, K, V]
    init = jnp.where(has_initial_state[:, None, None, None], init, 0.0)
    init = init[None]  # [1, N, H, K, V]

    out, final_state = chunk_kda(
        hm(q),
        hm(k),
        hm(v),
        hm(g),
        jnp.transpose(beta, (2, 0, 1)),  # [1,T,H] -> [H,1,T]
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        safe_gate=True,
        segment_ids=segment_ids,
        N_max=n_max,
        initial_state=init,
        output_final_state=True,
    )
    # [H, 1, T, V] -> [1, T, H, V]
    out = jnp.transpose(out, (1, 2, 0, 3))
    # Scatter the per-segment final states back, transposed into vLLM's layout.
    pool = pool.at[state_indices].set(
        jnp.swapaxes(final_state[0], -1, -2).astype(pool.dtype))
    return out, pool


def _decode_kda_adapter(
    mixed_qkv: jax.Array,  # [N, 3 * H * D], before short convolution
    conv_state: jax.Array,
    conv_weight: jax.Array,
    g: jax.Array,  # [1, N, H, D]
    beta: jax.Array,  # [1, N, H]
    pool: jax.Array,  # [num_slots, H, K, V]
    a_log: jax.Array,  # [H]
    dt_bias: jax.Array,  # [H * K]
    state_indices: jax.Array,
    has_initial_state: jax.Array,
    *,
    lower_bound: float | None,
    conv_state_dim_first: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fused decode short convolution and recurrent KDA."""
    n_v, d_k = g.shape[2], g.shape[3]
    out, conv_state, pool = decode_kda(
        mixed_qkv,
        g[0],
        beta[0],
        conv_state,
        conv_weight,
        pool,
        a_log,
        dt_bias.reshape(n_v, d_k),
        state_indices,
        has_initial_state,
        lower_bound=lower_bound,
        conv_state_dim_first=conv_state_dim_first,
        # vLLM stores recurrent state as [d_v, d_k].
        state_transposed=True,
    )
    return out[None], conv_state, pool  # [1, N, H, V]


def _naive_kda_adapter(
    mixed_qkv: jax.Array,  # [T, 3 * H * D]  post-conv, already activated
    raw_gate: jax.Array,  # [T, H * D]      raw gate (g1)
    raw_beta: jax.Array,  # [T, H]          RAW, not sigmoided
    output_gate: jax.Array,  # [T, H * D]   raw output gate (g2)
    pool: jax.Array,  # [num_slots, H, V, K]
    a_log: jax.Array,  # [H]
    dt_bias: jax.Array,  # [H * D]
    norm_weight: jax.Array,  # [D]
    query_start_loc: jax.Array,  # [num_seqs + 1]
    state_indices: jax.Array,  # [num_seqs]
    seq_lens: jax.Array,  # [num_seqs]
    *,
    lower_bound: float | None,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Naive recurrence over the whole batch; returns (out, updated pool).

    Covers prefill and decode with one call, because it scans token by token and
    reads each token's sequence from ``query_start_loc`` -- there is no chunking
    and so no reason to split the two.

    Three conventions differ from the Pallas kernels, and all three are handled
    here rather than at the call site so that ``_forward`` stays readable:

    * **beta is raw.** ``naive_kda`` applies the sigmoid itself, so this
      must NOT be handed pre-activated beta -- doing so squashes the delta-rule
      update towards sigmoid(0.5) and is nearly silent.
    * **it fuses the output norm and gate.** ``norm_weight``/``eps`` are the
      ``o_norm`` parameters, and ``output_gate`` is applied inside, so the caller
      must skip its own ``o_norm``.
    * **state orientation.** The pool stores each slot as ``[V, K]``; the naive
      recurrence works in ``[K, V]`` like the kernels do, and cannot detect the
      difference because the two dims are equal. Swap on load and on store,
      exactly as ``_chunk_kda_adapter`` does.
    """
    out, new_state = naive_kda(
        mixed_qkv,
        raw_gate,
        raw_beta,
        output_gate,
        jnp.swapaxes(pool, -1, -2),  # [slots, H, V, K] -> [slots, H, K, V]
        a_log,
        dt_bias,
        norm_weight,
        query_start_loc,
        state_indices,
        seq_lens,
        lower_bound=lower_bound,
        eps=eps,
    )
    # [T, H, D] -> [1, T, H, D], and the pool back into vLLM's orientation.
    return out[None], jnp.swapaxes(new_state, -1, -2).astype(pool.dtype)


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------
class KimiK3DeltaAttention(nn.Module):
    """KDA layer wired to the TPU kernels. See module docstring for scope."""

    def __init__(self, config, vllm_config, prefix: str = "") -> None:
        super().__init__()

        self.prefix = prefix
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = getattr(vllm_config, "quant_config", None)
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()

        kda_config = config.linear_attn_config
        assert kda_config is not None, "linear_attn_config must be set"
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)
        self.projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(self.projection_size, self.tp_size)
        self.conv_size = kda_config["short_conv_kernel_size"]
        assert kda_config.get(
            "use_full_rank_gate",
            False), ("KimiK3DeltaAttention requires a full-rank gate")

        # f_a is replicated because it is the contraction input to f_b_proj;
        # q/k/v and the gates remain column-parallel.
        self.in_proj_qkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.projection_size] * 3,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkv",
        )
        self.g_proj = ColumnParallelLinear(
            self.hidden_size,
            self.projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_proj",
        )
        # Replicated, not column-parallel: f_a feeds f_b_proj, whose input size
        # is head_dim. Every rank needs the whole vector.
        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_a_proj",
        )
        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.b_proj",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )

        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32))
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32))
        try:
            from vllm.model_executor.model_loader.weight_utils import \
                sharded_weight_loader
            set_weight_attrs(self.dt_bias,
                             {"weight_loader": sharded_weight_loader(0)})
            set_weight_attrs(self.A_log,
                             {"weight_loader": sharded_weight_loader(0)})
        except ImportError:
            pass

        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=3 * self.projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.gate_lower_bound: float | None = kda_config.get(
            "gate_lower_bound", None)
        if self.gate_lower_bound is not None:
            assert _KDA_GATE_LOGBOUND_MIN <= self.gate_lower_bound < 0, (
                "KDA gate lower bound must be in "
                f"[{_KDA_GATE_LOGBOUND_MIN}, 0). Got {self.gate_lower_bound}.")

        self.o_norm = GatedRMSNormSigmoid(self.head_dim)
        self.o_proj = RowParallelLinear(
            self.projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # (conv_state, recurrent_state); assigned by the runner, or by a test.
        self.kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._conv_state_dim_first = is_conv_state_dim_first()
        self._conv_op = None
        self._chunk_op = None
        self._decode_op = None
        self._naive_op = None

    # -- state shapes, for whoever allocates the cache -----------------------
    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=0,
        )

    # -- torch <-> jax op construction --------------------------------------
    def _get_conv_op(self):
        state_dim_first = self._conv_state_dim_first
        key = (self.conv_size, state_dim_first)
        if self._conv_op is not None and self._conv_op[0] == key:
            return self._conv_op[1]
        op = build_kimi_sconv_op(
            self.prefix,
            kernel_size=self.conv_size,
            state_dim_first=state_dim_first,
        )
        self._conv_op = (key, op)
        return op

    def _get_chunk_op(self, n_max: int):
        """Build (and cache per n_max) the prefill jax_op.

        ``N_max`` is static to the kernel, so one op is compiled per distinct
        sequence count. A real integration would bucket this the way the runner
        buckets token counts.
        """
        key = ("prefill", n_max)
        if self._chunk_op is not None and self._chunk_op[0] == key:
            return self._chunk_op[1]

        fn = functools.partial(_chunk_kda_adapter,
                               lower_bound=self.gate_lower_bound,
                               n_max=n_max)

        name = (f"pallas::kda_prefill_{self.prefix.replace('.', '_')}"
                f"_n{n_max}")
        # The caller reuses the cache tensor after copying the result back, so
        # the pool cannot be donated to this op.
        op = pallas.jax_op(name, fn)

        def _fake(q, _k, _v, _g, _beta, pool, *args, **kwargs):
            return (torch.empty_like(q), torch.empty_like(pool))

        op.register_fake(_fake)
        self._chunk_op = (key, op)
        return op

    def _get_decode_op(self):
        if self._decode_op is not None:
            return self._decode_op

        fn = functools.partial(
            _decode_kda_adapter,
            lower_bound=self.gate_lower_bound,
            conv_state_dim_first=self._conv_state_dim_first,
        )

        name = f"pallas::kda_decode_{self.prefix.replace('.', '_')}"
        # The vLLM cache owns these buffers across forwards. Pallas aliases its
        # JAX-local state operands, then the caller writes the results back into
        # the stable Torch buffers below.
        op = pallas.jax_op(name, fn)

        def _fake(mixed_qkv, conv_state, _conv_weight, g, _beta, pool,
                  *args, **kwargs):
            out = torch.empty((1, mixed_qkv.size(0), g.size(2), g.size(3)),
                              dtype=mixed_qkv.dtype,
                              device=mixed_qkv.device)
            return out, torch.empty_like(conv_state), torch.empty_like(pool)

        op.register_fake(_fake)
        self._decode_op = op
        return op

    def _get_naive_op(self):
        """Build (and cache) the naive-recurrence jax_op.

        One op for both batch kinds: unlike the prefill kernel there is no static
        ``N_max`` to specialise on, so nothing needs bucketing.
        """
        if getattr(self, "_naive_op", None) is not None:
            return self._naive_op

        fn = functools.partial(_naive_kda_adapter,
                               lower_bound=self.gate_lower_bound,
                               eps=self.o_norm.eps)

        name = f"pallas::kda_naive_{self.prefix.replace('.', '_')}"
        op = pallas.jax_op(name, fn)  # not donated; see _get_chunk_op

        def _fake(mixed_qkv, _raw_gate, _raw_beta, _output_gate, pool, *args,
                  **kwargs):
            num_heads, _v, d_k = pool.shape[1:]
            out = torch.empty((1, mixed_qkv.shape[0], num_heads, d_k),
                              dtype=mixed_qkv.dtype,
                              device=mixed_qkv.device)
            return (out, torch.empty_like(pool))

        op.register_fake(_fake)
        self._naive_op = op
        return op

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: KdaMetadata | None = None,
    ) -> torch.Tensor:
        """Step-for-step the same as upstream's ``forward``.

        ``metadata`` is an addition: upstream reads it from the global forward
        context. Passing it explicitly is what lets an accuracy test drive this
        layer without a vLLM engine; when omitted it is read from the forward
        context exactly as upstream does.
        """
        del positions  # KDA is position-free; kept for interface parity.
        num_tokens = hidden_states.size(0)
        # Upstream does this as one fused GEMM plus a split; here the projections
        # are separate so that f_a can be replicated across TP ranks. See
        # __init__ for why. Shapes are unchanged: mixed_qkv is
        # [num_tokens, 3 * local_projection_size], g_proj_states is
        # [num_tokens, local_projection_size], f_a is [num_tokens, head_dim] and
        # beta is [num_tokens, local_num_heads].
        mixed_qkv = self.in_proj_qkv(hidden_states)[0]
        g_proj_states = self.g_proj(hidden_states)[0]
        f_a = self.f_a_proj(hidden_states)[0]
        beta = self.b_proj(hidden_states)[0]

        g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
            metadata=metadata,
        )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        return self.o_proj(core_attn_out)[0]

    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        metadata: KdaMetadata | None = None,
    ) -> None:
        if metadata is None:
            raw = get_forward_context().attn_metadata
            if raw is None:
                return
            assert isinstance(raw,
                              dict), ("expected per-layer attn_metadata dict")
            metadata = raw[self.prefix]
        m = metadata

        naive = use_naive_kda()

        if m.num_spec_decodes:
            raise NotImplementedError(
                "Speculative decoding is not supported by the TPU KDA decode "
                "kernel (window_size == 1). Upstream routes these through "
                "fused_recurrent_kda with num_accepted_tokens rollback.")
        if m.num_prefills > 0 and m.num_decodes > 0 and not naive:
            # Only the two Pallas kernels have this limit. The chunked one would
            # have to pad the 1-token decode rows to chunk_size, and the decode
            # one cannot consume multi-token rows at all, so a mixed batch needs
            # the tokens split -- which upstream does with spec_token_indx /
            # non_spec_token_indx and this port does not.
            #
            # The naive recurrence has no such limit: it scans token by token and
            # reads each token's sequence from `query_start_loc`, so prefill and
            # decode rows in one batch are just rows of different lengths. The
            # conv1d ahead of it needs nothing special either -- the varlen
            # variant selected below produces bit-identical output and state to
            # the decode-update variant on a 1-token row (asserted in
            # tests/models/kimi_k3/test_kda_layer.py).
            raise NotImplementedError(
                "Mixed prefill+decode batches are not split yet; upstream uses "
                "spec_token_indx / non_spec_token_indx to separate them. Set "
                f"{_USE_NAIVE_KDA_ENV}=1 to run them through the naive "
                "recurrence, which handles them directly.")

        assert self.kv_cache is not None, "kv_cache must be assigned"
        conv_state, recurrent_state = self.kv_cache

        n = m.num_actual_tokens
        mixed_qkv = mixed_qkv[:n]
        g1 = g1[:, :n]
        beta = beta[:, :n]

        # Pallas takes activated beta; the naive recurrence activates it itself.
        if not naive:
            beta = torch.sigmoid(beta.float()).to(beta.dtype)

        # A mixed batch (naive mode only) lands here too: `num_prefills > 0`, so
        # it takes the varlen conv, which is right for both kinds of row.
        is_prefill = m.num_prefills > 0
        if is_prefill:
            q_weight, k_weight, v_weight = self.conv1d.weight.split(
                self.local_projection_size, dim=0)
            query_start_loc = m.query_start_loc.to(
                dtype=torch.int32, device=mixed_qkv.device)
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            seq_lens = query_lens + m.has_initial_state.to(torch.int32)
            conv_out = self._get_conv_op()(
                mixed_qkv,
                conv_state,
                q_weight,
                k_weight,
                v_weight,
                query_start_loc,
                m.state_indices.to(torch.int32),
                seq_lens,
            )
        elif naive:
            # conv1d weight is stored as [3 * local_proj, 1, conv_size].
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                                   self.conv1d.weight.size(2))
            if self._conv_state_dim_first:
                conv_state = conv_state.transpose(-1, -2)
            conv_out = _causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                m.state_indices,
            )

        num_seqs = m.state_indices.numel()
        has_init = m.has_initial_state.to(torch.int32)
        state_indices = m.state_indices.to(torch.int32)

        if naive:
            # The naive recurrence takes the packed post-conv q|k|v and the two
            # gates flat, and derives has_initial_state from seq_lens itself.
            qsl = m.query_start_loc.to(torch.int32).to(conv_out.device)
            query_lens = qsl[1:] - qsl[:-1]
            # Any seq_len strictly greater than query_len marks "carries state
            # in", which is the only thing the kernel reads it for; +1 where
            # has_initial_state is set reproduces this metadata's flag exactly.
            seq_lens = query_lens + has_init
            # g1 is [1, T, H, D], while g2 is [T, H, D].
            op = self._get_naive_op()
            out, new_pool = op(conv_out, g1.reshape(n,
                                                    -1), beta.reshape(n, -1),
                               g2[:n].reshape(n, -1), recurrent_state,
                               self.A_log, self.dt_bias, self.o_norm.weight,
                               qsl, state_indices, seq_lens)
            recurrent_state.copy_(new_pool)
            # The naive path already applied o_norm and the output gate, so
            # unlike the kernel paths below there is nothing left to do here.
            core_attn_out[:, :n] = out[:, :n].to(core_attn_out.dtype)
            return

        if is_prefill:
            q, k, v = (rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                       for x in conv_out.split(self.local_projection_size,
                                               dim=-1))
            segment_ids = m.segment_ids
            if segment_ids is None:
                segment_ids = self._segment_ids(m.query_start_loc, n,
                                                mixed_qkv.device)
            else:
                segment_ids = segment_ids.to(dtype=torch.int32,
                                             device=mixed_qkv.device)
                if segment_ids.shape != (1, n):
                    raise ValueError(
                        f"segment_ids must have shape {(1, n)}, got "
                        f"{tuple(segment_ids.shape)}")
            op = self._get_chunk_op(num_seqs)
            out, new_pool = op(q, k, v, g1, beta, recurrent_state, self.A_log,
                               self.dt_bias, segment_ids, state_indices,
                               has_init)
        else:
            op = self._get_decode_op()
            out, new_conv_state, new_pool = op(
                mixed_qkv,
                conv_state,
                self.conv1d.weight,
                g1,
                beta,
                recurrent_state,
                self.A_log,
                self.dt_bias,
                state_indices,
                has_init,
            )
            conv_state.copy_(new_conv_state)

        recurrent_state.copy_(new_pool)
        core_attn_out[:, :n] = out[:, :n].to(core_attn_out.dtype)
        core_attn_out.copy_(self.o_norm(core_attn_out, g2))

    @staticmethod
    def _segment_ids(query_start_loc: torch.Tensor, num_tokens: int,
                     device) -> torch.Tensor:
        """``[1, T]`` 1-indexed segment ids from cu_seqlens; 0 marks padding.

        The chunked kernel takes ``segment_ids`` where upstream's
        ``chunk_kda_with_fused_gate`` takes ``cu_seqlens``; this is the
        conversion. The Python list conversion is a temporary fallback for
        callers that have not precomputed graph inputs.
        """
        seg = torch.zeros(1, num_tokens, dtype=torch.int32, device=device)
        starts = query_start_loc.tolist()
        for i in range(len(starts) - 1):
            s, e = starts[i], min(starts[i + 1], num_tokens)
            if e > s:
                seg[0, s:e] = i + 1
        return seg
