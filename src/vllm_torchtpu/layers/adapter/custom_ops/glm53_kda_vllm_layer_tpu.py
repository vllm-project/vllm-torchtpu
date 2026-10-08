"""TPU backend for vLLM's KDA layer, via out-of-tree layer registration.

`KimiGatedDeltaNetAttention` is a `PluggableLayer`, and `PluggableLayer.__new__`
consults `op_registry_oot` by class name at every instantiation. Registering a
subclass under the same name therefore swaps the layer for the whole model with
no patching and no fork:

    import kda_vllm_layer_tpu   # registration happens on import

**Importing this module has a global side effect.** That is inherent to the
mechanism, not an oversight -- `register_oot` mutates a process-wide registry.
Import it from TPU platform setup, never as a transitive convenience.

## Why override `_forward` rather than the kernels

`_forward` imports its kernels *inside the function body*, branching on
`current_platform.is_rocm()` (`kimi_gdn_linear_attn.py:396-409`). There is no
module-level symbol to monkeypatch, and the branch has no TPU arm. Overriding
the method is the only seam that exists.

Everything outside `_forward` is reused unchanged: projections, the fused
`in_proj_qkvgfab`, weight loaders, `o_norm`, `o_proj`, and the state
shape/dtype calculators. Only the four kernel calls differ.

## Backends (`GLM53_KDA_BACKEND`)

`kimi` (default) runs the layer on the Kimi KDA ops that Kimi-Linear and
Kimi-K3 use, around the fused conv1d + GDN v3 kernel
(`kda_attention_op.py`). When the runner binds the unified block pool
(`len(kv_cache) == 1`, the default for this architecture, see
`gdn_pool_layout.POOLED_GDN_ARCHITECTURES`) the pooled op gathers and scatters
the state through the pool, which is what hybrid prefix caching and host KV
offload require. On per-layer caches (`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0`)
the dispatched op runs instead. Its ABI also takes speculative-decoding verify
windows, as Kimi-K3 uses it, but that is not yet validated for GLM-5.3. The
pooled op does not support speculative decoding at all yet.

`legacy` keeps the original GLM kernels below. They need the per-layer conv
and recurrent caches in the upstream layout, so they only run with
`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0` (the layer refuses to build otherwise),
and they do not support speculative decoding: the base class routes
multi-query spec batches to `fused_recurrent_kda`, a third kernel with its own
ragged + `num_accepted_tokens` semantics that has no TPU implementation.
Rather than silently produce wrong tokens, that path raises.
"""

from __future__ import annotations

import sys
import types
from typing import Any, NamedTuple

import numpy as np
import torch

try:
    from vllm.g3_integration import patchers

    patchers.inject_wildcard_mocks()
    patchers.patch_metadata_versions()
except ImportError:
    pass

from vllm_torchtpu.layers.adapter.custom_ops import glm53_kda_conv_tpu as kda_conv_tpu
from vllm_torchtpu.layers.adapter.custom_ops import (
    glm53_kda_vllm_decode as kda_vllm_decode,
)
from vllm_torchtpu.layers.adapter.custom_ops import (
    glm53_kda_vllm_prefill as kda_vllm_prefill,
)


def _rms_norm_gated_fn(
    x: torch.Tensor,
    g: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    is_sigmoid: bool,
) -> torch.Tensor:
    orig_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps) * weight.float()
    if g is not None:
        gf = g.float()
        if is_sigmoid:
            out = out * torch.sigmoid(gf)
        else:
            out = out * torch.nn.functional.silu(gf)
    return out.to(orig_dtype)


_COMPILED_RMS_NORM_CACHE: dict[tuple[Any, ...], Any] = {}
_COMPILED_CONV_STEP_CACHE: dict[tuple[Any, ...], Any] = {}


def _run_compiled_rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    is_sigmoid: bool,
) -> torch.Tensor:
    if x.device.type == "tpu":
        key = (tuple(x.shape), x.dtype, g is not None, is_sigmoid)
        fn = _COMPILED_RMS_NORM_CACHE.get(key)
        if fn is None:
            fn = torch.compile(_rms_norm_gated_fn, backend="tpu", dynamic=False)
            _COMPILED_RMS_NORM_CACHE[key] = fn
        return fn(x, g, weight, eps, is_sigmoid)
    return _rms_norm_gated_fn(x, g, weight, eps, is_sigmoid)


def _conv_step_fn(
    x: torch.Tensor,  # [num_tokens, dim] one decode token per sequence
    state: torch.Tensor,  # [num_tokens, dim, state_len] gathered conv window
    weight: torch.Tensor,  # [dim, width], width == state_len + 1
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure single-token causal depthwise conv + SiLU.

    Same math as `causal_conv1d_update_paged`, but returns the advanced window
    instead of scattering it, so `torch.compile` never has to propagate an
    in-place update of a (transposed) view of the paged cache.
    """
    window = torch.cat([state, x.unsqueeze(-1).to(state.dtype)], dim=-1)
    out = (window.float() * weight.float().unsqueeze(0)).sum(-1)
    if bias is not None:
        out = out + bias.float().unsqueeze(0)
    out = torch.nn.functional.silu(out)
    return out.to(x.dtype), window[..., 1:]


def _run_compiled_conv_step(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.device.type == "tpu":
        key = (tuple(x.shape), x.dtype, tuple(state.shape), bias is not None)
        fn = _COMPILED_CONV_STEP_CACHE.get(key)
        if fn is None:
            fn = torch.compile(_conv_step_fn, backend="tpu", dynamic=False)
            _COMPILED_CONV_STEP_CACHE[key] = fn
        return fn(x, state, weight, bias)
    return _conv_step_fn(x, state, weight, bias)


class FusedRMSNormGated(torch.nn.Module):
    """Strict FP32 TPU replacement for FLA's Triton FusedRMSNormGated.

    Compiled via torch.compile.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        activation: str = "sigmoid",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype or torch.float32)
        )
        self.eps = eps
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        if self.activation not in ("sigmoid", "silu", "swish"):
            raise ValueError(f"Unsupported activation: {self.activation!r}")
        return _run_compiled_rms_norm_gated(
            x, g, self.weight, self.eps, self.activation == "sigmoid"
        )


class _FlaKdaShimModule(types.ModuleType):
    """TPU shim for `vllm.third_party.flash_linear_attention.ops.kda`."""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _unimplemented_fla_op(*args, **kwargs):
            raise NotImplementedError(
                f"FLA CUDA op {name!r} is not used on the TPU KDA path."
            )

        return _unimplemented_fla_op


def _install_fla_kda_shim() -> None:
    mod_name = "vllm.third_party.flash_linear_attention.ops.kda"
    mod = _FlaKdaShimModule(mod_name)
    mod.FusedRMSNormGated = FusedRMSNormGated  # type: ignore[attr-defined]
    mod.chunk_kda_with_fused_gate = (  # type: ignore[attr-defined]
        kda_vllm_prefill.chunk_kda_with_fused_gate
    )
    mod.fused_recurrent_kda_packed_decode = (  # type: ignore[attr-defined]
        kda_vllm_decode.fused_recurrent_kda_packed_decode
    )
    sys.modules[mod_name] = mod


_install_fla_kda_shim()

# pylint: disable=g-import-not-at-top
from vllm.forward_context import get_forward_context  # noqa: E402
from vllm.logger import init_logger  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (  # noqa: E402
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    is_conv_state_dim_first,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.gdn_pool_layout import (  # noqa: E402
    unified_kv_layout_enabled_for_architecture,
)
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (  # noqa: E402
    build_kimi_dispatched_kda_op,
    build_kimi_pooled_kda_op,
)

logger = init_logger(__name__)

# Which kernels run the KDA layers; see the module docstring. Read once per
# layer at construction and by the model class's state shape/dtype
# classmethods, so the pool sizing and the layers always agree.
GLM53_KDA_BACKEND_ENV = "GLM53_KDA_BACKEND"
KDA_BACKEND_KIMI = "kimi"
KDA_BACKEND_LEGACY = "legacy"
# GLM-5.3's attention layers use the standard (RPA) paged KV cache, whose
# pool keeps one token per row; the Kimi pooled op's default assumes the
# MLA pool, which packs several.
_GLM53_POOL_TOKENS_PER_ROW = 1


def glm53_kda_backend() -> str:
    """The KDA backend selected by `GLM53_KDA_BACKEND` (default `kimi`)."""
    raw = envs.GLM53_KDA_BACKEND
    backend = raw.strip().lower() or KDA_BACKEND_KIMI
    if backend not in (KDA_BACKEND_KIMI, KDA_BACKEND_LEGACY):
        raise ValueError(
            f"{GLM53_KDA_BACKEND_ENV}={raw!r}; expected "
            f"{KDA_BACKEND_KIMI!r} or {KDA_BACKEND_LEGACY!r}."
        )
    return backend


def _require_per_layer_state_caches(vllm_config: Any) -> None:
    """Fails fast when the legacy kernels would be handed the unified pool.

    The runner binds the pool (one buffer per layer) for every architecture in
    `POOLED_GDN_ARCHITECTURES` unless `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0`,
    and the legacy kernels cannot read it. Raising at construction stops the
    server before it loads the weights rather than at its first real step.
    """
    model_config = getattr(vllm_config, "model_config", None)
    architecture = getattr(model_config, "architecture", None)
    if unified_kv_layout_enabled_for_architecture(architecture):
        raise NotImplementedError(
            f"{GLM53_KDA_BACKEND_ENV}={KDA_BACKEND_LEGACY} needs separate conv "
            "and recurrent state caches, but the unified KV pool is enabled for "
            f"{architecture}. Set TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0, or use "
            f"{GLM53_KDA_BACKEND_ENV}={KDA_BACKEND_KIMI}."
        )


def kimi_kda_state_shape(
    conv_kernel_size: int, local_num_heads: int, head_dim: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Per-slot (conv, recurrent) state shapes of the Kimi KDA ops.

    The conv state is declared in the fused kernel's `(taps, 1, dim)` slot
    layout, as Kimi-K3 does, so the dense op reads the cache in place. The
    recurrent state is `(heads, d_k, d_v)`, which is also what the pooled op's
    SSM region holds.
    """
    return (
        (conv_kernel_size - 1, 1, 3 * local_num_heads * head_dim),
        (local_num_heads, head_dim, head_dim),
    )


def kimi_kda_state_dtype() -> tuple[torch.dtype, torch.dtype]:
    """Per-slot (conv, recurrent) state dtypes of the Kimi KDA ops.

    The conv state is fp32 for the same reason as Kimi-K3's `kda_state_dtype`:
    the dense op hands the whole slot cache to the fused kernel, which widens
    anything narrower than fp32 once per layer per step. (The unified pool
    stores conv as bf16 regardless; see `gdn_pool_layout`.) The recurrent
    state must be fp32, which is the op's ABI.
    """
    return (torch.float32, torch.float32)


def _gather_initial_states_tpu(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """TPU counterpart to `gather_initial_states` (which asserts `state.is_cuda`)."""
    idx = indices.to(torch.long)
    gathered = state.index_select(0, idx)
    mask = has_initial_state.view(-1, *([1] * (state.ndim - 1)))
    return torch.where(mask, gathered, torch.zeros_like(gathered))


# Request-count buckets. Every tensor in the TPU path is shaped only by the
# padded token count, a bucketed request count and the chunk length, so the
# Pallas kernels and the compiled norm see a handful of shapes rather than one
# per distinct batch composition.
_MIN_REQ_BUCKET = 8
_MIN_PREFILL_REQ_BUCKET = 1
# Prefill chunk lengths: long prompts take fewer, longer kernel launches.
_SHORT_CHUNK_LEN = 64
_LONG_CHUNK_LEN = 256
_LONG_CHUNK_MIN_TOKENS = 512
# The state slot that padded plan entries point at. The runner reserves it
# (block 0 is the null block and is never assigned to a request), so the
# decode kernel skips entries on it instead of advancing junk into it.
_NULL_SLOT = 0
_DECODE_PATH_LOGGED = [False]


def _bucket(n: int, min_bucket: int = _MIN_REQ_BUCKET) -> int:
    b = min_bucket
    while b < n:
        b *= 2
    return b


class _TpuStepPlan(NamedTuple):
    """Gather/scatter plan for one TPU forward step, shared by all KDA layers.

    Row ids index "token tables" whose rows `[0, padded_tokens)` are the step's
    tokens and whose row `padded_tokens` is zeros. The conv table also holds
    every request's cached window after that, `state_len` rows per request.
    Padded request entries read the zero row and use the null state slot 0,
    which the runner never assigns to a request.
    """

    num_reqs: int
    num_tokens: int
    padded_tokens: int
    # Conv, for all requests at once.
    slots: torch.Tensor  # [reqs_pad] int64 state slot per request
    conv_taps: tuple[torch.Tensor, ...]  # width x [padded_tokens] int64
    new_window_rows: torch.Tensor  # [reqs_pad * state_len] int64
    # Leading single-token requests that resume from cached state.
    num_decodes: int
    decode_rows: torch.Tensor | None  # [decodes_pad] int64
    decode_slots: torch.Tensor | None  # [decodes_pad] int32
    # Every other request, through the chunked prefill kernel.
    num_prefills: int
    prefill_slots: torch.Tensor | None  # [prefills_pad] int64
    prefill_has_initial_state: torch.Tensor | None  # [prefills_pad] bool
    prefill_chunk_len: int
    prefill_chunk_rows: tuple[torch.Tensor, ...]  # [prefills_pad * chunk_len]
    prefill_chunk_valid: tuple[torch.Tensor, ...]  # [prefills_pad] int32


# One plan per forward step, shared by every KDA layer. Keyed on the forward
# context, which vLLM creates afresh for each step, so a plan never outlives
# its step; each entry also holds its metadata object so that the `id()` in
# the key cannot be recycled while cached.
_PLAN_CACHE: dict[str, Any] = {"ctx": None, "plans": {}}
_PLAN_LOG_BUDGET = [6]


def _host_request_layout(
    m: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    """Copies the step's request layout to host in one device-to-host transfer.

    vllm_torchtpu's `AttentionMetadata` declares `query_start_loc_cpu` and
    `seq_lens_cpu`, but the TPU runner never assigns them: a step is described
    only by device tensors. Returns `(query_start_loc, seq_lens, slots,
    num_reqs)`, with `num_reqs` taken from `request_distribution[2]` when set.
    """
    qsl = m.query_start_loc.reshape(-1)
    seq_lens = m.seq_lens.reshape(-1)
    slots = getattr(m, "mamba_state_indices", None)
    if slots is None:
        # Mamba shares the attention block pool: the slot is the first block.
        slots = m.block_tables.reshape(seq_lens.shape[0], -1)[:, 0]
    slots = slots.reshape(-1)
    parts = [qsl, seq_lens, slots]
    distribution = getattr(m, "request_distribution", None)
    if distribution is not None:
        parts.append(distribution.reshape(-1))
    host = torch.cat([p.to(torch.int32) for p in parts]).cpu().numpy().astype(np.int64)
    a, b, c = qsl.numel(), seq_lens.numel(), slots.numel()
    num_reqs = int(host[a + b + c + 2]) if distribution is not None else None
    return host[:a], host[a : a + b], host[a + b : a + b + c], num_reqs


def _build_tpu_step_plan(
    m: Any, padded_tokens: int, state_len: int, device: torch.device
) -> _TpuStepPlan | None:
    """Builds a step's gather plan on host; None when nothing is scheduled.

    `query_start_loc` is padded by repeating the last valid offset, `seq_lens`
    counts context plus scheduled tokens (entries past `num_reqs` are stale),
    and the runner orders decode requests first. A request resumes from cached
    state iff `seq_len > query_len`; otherwise it starts from zeros, whatever
    its (possibly reused) slot holds.
    """
    if state_len < 1:
        raise ValueError(f"KDA conv state_len must be >= 1, got {state_len}.")
    qsl, seq_lens, slots, num_reqs = _host_request_layout(m)
    q_lens_all = qsl[1:] - qsl[:-1]
    if num_reqs is None:
        num_reqs = int((q_lens_all > 0).sum())
    num_reqs = max(
        0, min(num_reqs, q_lens_all.shape[0], seq_lens.shape[0], slots.shape[0])
    )
    num_tokens = int(qsl[num_reqs]) if num_reqs else 0
    if num_tokens <= 0:
        return None
    if num_tokens > padded_tokens:
        raise ValueError(
            f"KDA step schedules {num_tokens} tokens but the batch holds only "
            f"{padded_tokens} rows."
        )
    q_lens = q_lens_all[:num_reqs]
    if (q_lens <= 0).any():
        raise ValueError(f"KDA step has an empty request: query lens {q_lens}.")
    starts = qsl[:num_reqs]
    has_init = seq_lens[:num_reqs] > q_lens
    req_slots = slots[:num_reqs]

    zero_row = padded_tokens
    reqs_pad = _bucket(num_reqs)

    def upload(a: np.ndarray, dtype: torch.dtype = torch.int64) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)

    # Conv-table row of request r's cached window entry j (oldest first); a
    # request without initial state reads the zero row instead.
    lag = np.arange(state_len)
    window_rows = np.where(
        has_init[:, None],
        zero_row + 1 + np.arange(num_reqs)[:, None] * state_len + lag[None, :],
        zero_row,
    )

    def input_rows(req: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Conv-table row of input position `pos` (negative: cached) of `req`."""
        cached = window_rows[req, np.clip(state_len + pos, 0, state_len - 1)]
        return np.where(pos >= 0, starts[req] + pos, cached)

    # Tap w of token t reads input position `pos(t) - state_len + w`.
    tok_req = np.repeat(np.arange(num_reqs), q_lens)
    tok_pos = np.arange(num_tokens) - starts[tok_req]
    conv_taps = []
    for w in range(state_len + 1):
        rows = np.full(padded_tokens, zero_row, dtype=np.int64)
        rows[:num_tokens] = input_rows(tok_req, tok_pos - state_len + w)
        conv_taps.append(upload(rows))
    # The advanced window is each request's last `state_len` inputs.
    new_rows = np.full((reqs_pad, state_len), zero_row, dtype=np.int64)
    req_grid = np.repeat(np.arange(num_reqs)[:, None], state_len, axis=1)
    new_rows[:num_reqs] = input_rows(
        req_grid, q_lens[:, None] - state_len + lag[None, :]
    )
    slots_pad = np.zeros(reqs_pad, dtype=np.int64)
    slots_pad[:num_reqs] = req_slots

    # Decode-first ordering: the leading run of 1-token requests with cached
    # state takes the decode kernel. Anything else (including a fresh 1-token
    # prompt, which must start from zeros) takes the prefill kernel.
    is_decode = (q_lens == 1) & has_init
    num_decodes = num_reqs if is_decode.all() else int(np.argmin(is_decode))
    decode_rows = decode_slots = None
    if num_decodes:
        n_pad = _bucket(num_decodes)
        rows = np.full(n_pad, zero_row, dtype=np.int64)
        rows[:num_decodes] = starts[:num_decodes]
        dslots = np.zeros(n_pad, dtype=np.int64)
        dslots[:num_decodes] = req_slots[:num_decodes]
        decode_rows, decode_slots = upload(rows), upload(dslots, torch.int32)

    num_prefills = num_reqs - num_decodes
    prefill_slots = prefill_init = None
    chunk_len = _SHORT_CHUNK_LEN
    chunk_rows, chunk_valid = [], []
    if num_prefills:
        p_lens = q_lens[num_decodes:]
        p_starts = starts[num_decodes:]
        s_pad = _MIN_PREFILL_REQ_BUCKET if num_prefills == 1 else _bucket(num_prefills)
        max_len = int(p_lens.max())
        if max_len >= _LONG_CHUNK_MIN_TOKENS:
            chunk_len = _LONG_CHUNK_LEN
        pslots = np.zeros(s_pad, dtype=np.int64)
        pslots[:num_prefills] = req_slots[num_decodes:]
        pinit = np.zeros(s_pad, dtype=bool)
        pinit[:num_prefills] = has_init[num_decodes:]
        prefill_slots, prefill_init = upload(pslots), upload(pinit, torch.bool)
        col = np.arange(chunk_len)
        for offset in range(0, max_len, chunk_len):
            valid = np.clip(p_lens - offset, 0, chunk_len)
            rows = np.full((s_pad, chunk_len), zero_row, dtype=np.int64)
            rows[:num_prefills] = np.where(
                col[None, :] < valid[:, None],
                p_starts[:, None] + offset + col[None, :],
                zero_row,
            )
            vpad = np.zeros(s_pad, dtype=np.int64)
            vpad[:num_prefills] = valid
            chunk_rows.append(upload(rows.reshape(-1)))
            chunk_valid.append(upload(vpad, torch.int32))

    plan = _TpuStepPlan(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        padded_tokens=padded_tokens,
        slots=upload(slots_pad),
        conv_taps=tuple(conv_taps),
        new_window_rows=upload(new_rows.reshape(-1)),
        num_decodes=num_decodes,
        decode_rows=decode_rows,
        decode_slots=decode_slots,
        num_prefills=num_prefills,
        prefill_slots=prefill_slots,
        prefill_has_initial_state=prefill_init,
        prefill_chunk_len=chunk_len,
        prefill_chunk_rows=tuple(chunk_rows),
        prefill_chunk_valid=tuple(chunk_valid),
    )
    # Log a few real steps (warmup runs use only the null slot 0).
    if _PLAN_LOG_BUDGET[0] > 0 and req_slots.any():
        _PLAN_LOG_BUDGET[0] -= 1
        logger.info(
            "GLM-5.3 KDA step: reqs=%d (decode=%d, prefill=%d) tokens=%d/%d "
            "prefill_chunks=%dx%d q_lens=%s has_init=%s slots=%s",
            num_reqs,
            num_decodes,
            num_prefills,
            num_tokens,
            padded_tokens,
            len(chunk_rows),
            chunk_len,
            q_lens[:8].tolist(),
            has_init[:8].tolist(),
            req_slots[:8].tolist(),
        )
    return plan


def _tpu_step_plan(
    m: Any,
    forward_context: Any,
    padded_tokens: int,
    state_len: int,
    device: torch.device,
) -> _TpuStepPlan | None:
    """Returns the step's plan, building it once per forward step."""
    if _PLAN_CACHE["ctx"] is not forward_context:
        _PLAN_CACHE["ctx"] = forward_context
        _PLAN_CACHE["plans"] = {}
    plans = _PLAN_CACHE["plans"]
    key = (id(m), padded_tokens, state_len)
    entry = plans.get(key)
    if entry is None:
        entry = (m, _build_tpu_step_plan(m, padded_tokens, state_len, device))
        plans[key] = entry
    return entry[1]


@KimiGatedDeltaNetAttention.register_oot
class TpuKimiGatedDeltaNetAttention(KimiGatedDeltaNetAttention):
    """KDA on TPU.

    Supports both GDNAttentionMetadata and vllm_torchtpu AttentionMetadata.
    """

    def __init__(
        self,
        config: Any,
        vllm_config: Any,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)
        self._kda_backend = glm53_kda_backend()
        self._conv_weight_folded = False
        if self._kda_backend != KDA_BACKEND_KIMI:
            _require_per_layer_state_caches(vllm_config)
            return
        if self.conv1d.bias is not None:
            raise NotImplementedError(
                "The Kimi KDA ops take a bias-free short convolution."
            )
        eps = float(self.o_norm.eps)
        # Same pair of ops as Kimi-K3 (models/vllm/kimi_k3/attention.py). Both
        # own the short convolution, q/k L2-norm, the gate, beta's sigmoid, the
        # recurrence and the gated output norm, so `_forward` hands them the raw
        # projections.
        self.dispatched_kda_op = build_kimi_dispatched_kda_op(
            prefix,
            lower_bound=self.gate_lower_bound,
            eps=eps,
            state_dim_first=is_conv_state_dim_first(),
            num_spec_tokens=self.num_spec,
        )
        # The manager block size is read per call inside the op; it is not
        # final at model construction.
        self.pooled_kda_op = build_kimi_pooled_kda_op(
            prefix,
            lower_bound=self.gate_lower_bound,
            eps=eps,
            vllm_config=vllm_config,
            pool_tokens_per_row=_GLM53_POOL_TOKENS_PER_ROW,
        )
        # The ops take the conv weight as [kernel_size, 3, heads, head_dim];
        # `fold_conv_weight` derives it from `conv1d.weight` after loading.
        # Non-persistent: it is not a checkpoint entry.
        self.register_buffer(
            "conv_weight_fused",
            torch.empty(
                self.conv_size,
                3,
                self.local_num_heads,
                self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        if getattr(self, "_kda_backend", KDA_BACKEND_LEGACY) == KDA_BACKEND_KIMI:
            return kimi_kda_state_dtype()
        return super().get_state_dtype()

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if getattr(self, "_kda_backend", KDA_BACKEND_LEGACY) == KDA_BACKEND_KIMI:
            return kimi_kda_state_shape(
                self.conv_size, self.local_num_heads, self.head_dim
            )
        return super().get_state_shape()

    def fold_conv_weight(self) -> None:
        """Folds `conv1d.weight` into the Kimi ops' `[K, 3, H, D]` layout.

        `conv1d.weight` is `[3 * local_proj, 1, K]` with q, k, v stacked in that
        order and each laid out as (head, head_dim), which is exactly the
        channel order of `mixed_qkv`. Idempotent; both the per-layer and the
        model-level post-load hooks call it.
        """
        if getattr(self, "_kda_backend", KDA_BACKEND_LEGACY) != KDA_BACKEND_KIMI:
            return
        weight = self.conv1d.weight
        kernel_size = weight.shape[-1]
        fused = (
            weight[:, 0, :]
            .t()
            .reshape(kernel_size, 3, self.local_num_heads, self.head_dim)
        )
        # `fused` is a strided view. A fresh row-major buffer filled by `copy_`
        # breaks the lazy chain, so torch_tpu does not re-materialize the view
        # as a per-step program at the op boundary (same fix as Kimi-K3's).
        # `empty_like` would keep the view's strides, hence `empty`.
        out = torch.empty(tuple(fused.shape), dtype=fused.dtype, device=fused.device)
        out.copy_(fused)
        self.conv_weight_fused = out
        self._conv_weight_folded = True

    def process_weights_after_loading(
        self, act_dtype: torch.dtype | None = None
    ) -> None:
        del act_dtype
        self.fold_conv_weight()

    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        is_3d_out = core_attn_out.ndim == 3
        if g1.ndim == 3:
            g1 = g1.unsqueeze(0)
        if g2.ndim == 3:
            g2 = g2.unsqueeze(0)
        if beta.ndim == 2:
            beta = beta.unsqueeze(0)

        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        kv_cache = getattr(self, "kv_cache", None)
        if (
            attn_metadata_raw is None
            or kv_cache is None
            or len(kv_cache) == 0
            or kv_cache[0].numel() == 0
        ):
            # Profiling / warmup runs before the state cache exists.
            num_toks = mixed_qkv.size(0)
            q_dummy = mixed_qkv[:, : self.local_projection_size].view(
                1, num_toks, self.local_num_heads, self.head_dim
            )
            normed_dummy = self.o_norm(q_dummy, g2[:, :num_toks])
            if is_3d_out:
                core_attn_out[:num_toks] = normed_dummy[0]
            else:
                core_attn_out[0, :num_toks] = normed_dummy[0]
            return

        if isinstance(attn_metadata_raw, dict):
            m = attn_metadata_raw.get(
                self.prefix, next(iter(attn_metadata_raw.values()))
            )
        else:
            m = attn_metadata_raw

        # Objects built without `__init__` (test doubles) keep the legacy path.
        if getattr(self, "_kda_backend", KDA_BACKEND_LEGACY) == KDA_BACKEND_KIMI:
            self._forward_kimi(m, mixed_qkv, g1, g2, beta, core_attn_out, kv_cache)
            return

        if len(kv_cache) < 2:
            raise NotImplementedError(
                "The legacy GLM 5.3 TPU KDA kernels need separate conv and "
                "recurrent state caches (compact-mamba layout); the unified mamba "
                "block pool is not supported. Use GLM53_KDA_BACKEND=kimi, or set "
                "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0."
            )

        if isinstance(m, GDNAttentionMetadata):
            self._forward_gdn_metadata(
                m, mixed_qkv, g1, g2, beta, core_attn_out, kv_cache
            )
        else:
            self._forward_tpu_metadata(
                m, forward_context, mixed_qkv, g1, g2, beta, core_attn_out, kv_cache
            )

    def _forward_kimi(
        self,
        m: Any,
        mixed_qkv: torch.Tensor,  # [T, 3 * H * D], pre-convolution
        g1: torch.Tensor,  # [1, T, H, D] raw gate
        g2: torch.Tensor,  # [1, T, H, D] raw output gate
        beta: torch.Tensor,  # [1, T, H] raw
        core_attn_out: torch.Tensor,  # [1, T, H, D] or [T, H, D]
        kv_cache: Any,
    ) -> None:
        """Kimi KDA ops over vllm_torchtpu `AttentionMetadata`.

        Routing follows Kimi-K3: the runner binds a single buffer when the
        unified block pool is on, and the conv/recurrent pair otherwise. Both
        ops return the gated, normed output, so nothing else runs here.
        """
        if isinstance(m, GDNAttentionMetadata):
            raise NotImplementedError(
                "The Kimi KDA backend runs on vllm_torchtpu AttentionMetadata; "
                "use GLM53_KDA_BACKEND=legacy for GDNAttentionMetadata."
            )
        if not self._conv_weight_folded:
            raise RuntimeError(
                f"{self.prefix}: conv_weight_fused was never folded; "
                "process_weights_after_loading did not run."
            )
        num_tokens = mixed_qkv.shape[0]
        num_heads, head_dim = self.local_num_heads, self.head_dim
        mixed_qkv = mixed_qkv.contiguous()
        raw_gate = g1.reshape(num_tokens, num_heads * head_dim).contiguous()
        beta = beta.reshape(num_tokens, num_heads).contiguous()
        output_gate = g2.reshape(num_tokens, num_heads * head_dim).contiguous()
        state_indices = m.mamba_state_indices
        if state_indices is None:
            raise RuntimeError(
                "The Kimi KDA ops require mamba_state_indices in the attention "
                "metadata."
            )

        if len(kv_cache) == 1:
            if self.num_spec > 0 or m.mamba_slot_read_offsets is not None:
                # Speculative verify needs per-window state checkpoints; the pooled
                # gather/scatter path keeps a single state per block. Checking
                # `num_spec` as well keeps a speculative step that arrives without
                # slot offsets from silently running on the pooled state.
                raise NotImplementedError(
                    "Speculative decoding is not supported with pooled KDA; set "
                    "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0 to use the dedicated "
                    "conv/recurrent state caches."
                )
            output = self.pooled_kda_op(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                kv_cache[0],
                self.conv_weight_fused,
                self.A_log,
                self.dt_bias,
                self.o_norm.weight,
                m.query_start_loc,
                state_indices,
                m.seq_lens,
                m.request_distribution,
            )
        else:
            conv_cache, recurrent_cache = kv_cache[0], kv_cache[1]
            window_distribution = m.mamba_request_distribution
            if window_distribution is None:
                window_distribution = m.request_distribution
            slot_read_offsets = m.mamba_slot_read_offsets
            if slot_read_offsets is None:
                # Non-speculative steps keep the op's ABI without allocating a
                # pool-sized offset tensor; the num_spec_tokens=0 path never reads
                # this placeholder.
                slot_read_offsets = state_indices
            output = self.dispatched_kda_op(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                conv_cache,
                recurrent_cache,
                self.conv_weight_fused,
                self.A_log,
                self.dt_bias,
                self.o_norm.weight,
                m.query_start_loc,
                state_indices,
                m.seq_lens,
                m.request_distribution,
                window_distribution,
                slot_read_offsets,
            )

        # `output` is [T, H, D], already normed and gated.
        if core_attn_out.ndim == 3:
            core_attn_out.copy_(output)
        else:
            core_attn_out.copy_(output.unsqueeze(0))

    def _forward_tpu_metadata(
        self,
        m: Any,
        forward_context: Any,
        mixed_qkv: torch.Tensor,  # [padded_tokens, conv_dim]
        g1: torch.Tensor,  # [1, padded_tokens, num_heads, head_dim]
        g2: torch.Tensor,  # [1, padded_tokens, num_heads, head_dim]
        beta: torch.Tensor,  # [1, padded_tokens, num_heads]
        core_attn_out: torch.Tensor,
        kv_cache: Any,
    ) -> None:
        """vllm_torchtpu `AttentionMetadata`: one bucketed gather plan per step."""
        device = mixed_qkv.device
        padded_tokens, dim = mixed_qkv.shape
        conv_cache, recurrent_state = kv_cache[0], kv_cache[1]
        dim_first = is_conv_state_dim_first()
        state_len = conv_cache.size(-1) if dim_first else conv_cache.size(-2)
        width = self.conv1d.weight.size(-1)
        if width != state_len + 1:
            raise ValueError(
                f"expected conv width == state_len + 1, got width={width}, "
                f"state_len={state_len}."
            )
        plan = _tpu_step_plan(m, forward_context, padded_tokens, state_len, device)
        if plan is None:
            core_attn_out.zero_()
            return

        # 1. Short causal conv for every request at once: `width` row gathers
        #    from [tokens | zero row | cached windows], accumulated in fp32.
        windows = conv_cache.index_select(0, plan.slots)
        if dim_first:
            windows = windows.transpose(1, 2)  # -> [reqs_pad, state_len, dim]
        table = torch.cat(
            [
                mixed_qkv,
                mixed_qkv.new_zeros(1, dim),
                windows.reshape(-1, dim).to(mixed_qkv.dtype),
            ],
            dim=0,
        )
        weight = self.conv1d.weight.view(dim, width).float()
        acc = None
        for w, rows in enumerate(plan.conv_taps):
            tap = table.index_select(0, rows).float() * weight[:, w]
            acc = tap if acc is None else acc + tap
        if self.conv1d.bias is not None:
            acc = acc + self.conv1d.bias.float()
        conv_out = torch.nn.functional.silu(acc).to(mixed_qkv.dtype)
        new_windows = table.index_select(0, plan.new_window_rows).view(
            -1, state_len, dim
        )
        if dim_first:
            new_windows = new_windows.transpose(1, 2)
        conv_cache.index_copy_(
            0, plan.slots, new_windows.contiguous().to(conv_cache.dtype)
        )

        # 2. Recurrence.
        num_heads, head_dim = self.local_num_heads, self.head_dim
        n_dec = plan.decode_rows.numel() if plan.num_decodes else 0
        if plan.num_decodes and not plan.num_prefills and n_dec <= padded_tokens:
            # Pure decode: request r's token is row r, so the kernel reads the
            # step's tensors directly, with no token tables, gathers or output
            # scatter. Plan entries past the real decodes point at the null slot,
            # so the kernel leaves that slot alone and their rows come out zero.
            if not _DECODE_PATH_LOGGED[0]:
                _DECODE_PATH_LOGGED[0] = True
                logger.info(
                    "GLM-5.3 KDA pure-decode fast path: in-place state=%s "
                    "(decodes padded to %d, tokens padded to %d)",
                    kda_vllm_decode.can_update_in_place(recurrent_state),
                    n_dec,
                    padded_tokens,
                )
            out, _ = kda_vllm_decode.fused_recurrent_kda_packed_decode(
                mixed_qkv=conv_out[:n_dec],
                raw_g=g1[:, :n_dec],
                raw_beta=beta[:, :n_dec],
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
                initial_state=recurrent_state,
                state_indices=plan.decode_slots,
                null_slot=_NULL_SLOT,
            )
            out_tok = out[0]
            if n_dec < padded_tokens:
                out_tok = torch.cat(
                    [
                        out_tok,
                        out_tok.new_zeros(padded_tokens - n_dec, num_heads, head_dim),
                    ],
                    dim=0,
                )
        else:
            # Row `padded_tokens` of each token table is zeros; padded plan entries
            # read it and write their discarded outputs back to it.
            qkv_rows = torch.cat([conv_out, conv_out.new_zeros(1, dim)], dim=0)
            g_rows = torch.cat(
                [g1[0, :padded_tokens], g1.new_zeros(1, num_heads, head_dim)], dim=0
            )
            beta_rows = torch.cat(
                [beta[0, :padded_tokens], beta.new_zeros(1, num_heads)], dim=0
            )
            out_rows = torch.zeros(
                (padded_tokens + 1, num_heads, head_dim),
                dtype=torch.float32,
                device=device,
            )
            if plan.num_decodes:
                rows = plan.decode_rows
                out, _ = kda_vllm_decode.fused_recurrent_kda_packed_decode(
                    mixed_qkv=qkv_rows.index_select(0, rows),
                    raw_g=g_rows.index_select(0, rows).unsqueeze(0),
                    raw_beta=beta_rows.index_select(0, rows).unsqueeze(0),
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=recurrent_state,
                    state_indices=plan.decode_slots,
                    null_slot=_NULL_SLOT,
                )
                out_rows.index_copy_(0, rows, out[0].to(out_rows.dtype))
            if plan.num_prefills:
                kda_vllm_prefill.chunk_kda_prefill_gathered(
                    qkv_rows=qkv_rows,
                    g_rows=g_rows,
                    beta_rows=beta_rows,
                    A_log=self.A_log,
                    g_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    recurrent_state=recurrent_state,
                    slots=plan.prefill_slots,
                    has_initial_state=plan.prefill_has_initial_state,
                    chunk_rows=plan.prefill_chunk_rows,
                    chunk_valid=plan.prefill_chunk_valid,
                    chunk_len=plan.prefill_chunk_len,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    out=out_rows,
                    num_prefills=plan.num_prefills,
                    num_tokens=plan.num_tokens - plan.num_decodes,
                    prefill_start=plan.num_decodes,
                )
            out_tok = out_rows[:padded_tokens]

        # 3. Gated RMSNorm over the whole padded bucket (a static shape). Rows
        #    past the scheduled tokens are zeros and stay zeros.
        normed = self.o_norm(out_tok.unsqueeze(0), g2[:, :padded_tokens])
        if core_attn_out.ndim == 3:
            core_attn_out.copy_(normed[0])
        else:
            core_attn_out.copy_(normed)

    def _forward_gdn_metadata(
        self,
        m: GDNAttentionMetadata,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        kv_cache: Any,
    ) -> None:
        """Upstream `GDNAttentionMetadata`, whose layout is already host-built."""
        if m.spec_sequence_masks is not None:
            raise NotImplementedError(
                "Speculative decoding is not supported by the TPU KDA backend."
            )
        is_3d_out = core_attn_out.ndim == 3
        device = mixed_qkv.device
        num_actual_tokens = m.num_actual_tokens
        num_prefills = m.num_prefills
        state_indices = m.non_spec_state_indices_tensor
        query_start_loc = m.non_spec_query_start_loc
        has_initial_state = m.has_initial_state
        assert state_indices is not None
        assert query_start_loc is not None

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        conv_cache, recurrent_state = kv_cache[0], kv_cache[1]
        dim_first = is_conv_state_dim_first()
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        slots = state_indices.to(torch.long)

        # This batch's conv windows as a contiguous [num_seqs, dim, state_len]
        # buffer. The advanced windows are scattered back into the paged cache in
        # its native layout below, so nothing writes through a transposed view.
        windows = conv_cache.index_select(0, slots)
        if not dim_first:
            windows = windows.transpose(-1, -2)
        windows = windows.contiguous()

        if num_prefills > 0:
            assert has_initial_state is not None
            mixed_qkv_conv = kda_conv_tpu.causal_conv1d_fn_paged(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_states=windows,
                has_initial_state=has_initial_state,
                cache_indices=torch.arange(
                    windows.shape[0], device=device, dtype=torch.int32
                ),
                query_start_loc=query_start_loc,
            ).transpose(0, 1)
            new_windows = windows
            q_ns, k_ns, v_ns = mixed_qkv_conv.split(self.local_projection_size, dim=-1)

            shape = (1, num_actual_tokens, self.local_num_heads, self.head_dim)
            initial_state = _gather_initial_states_tpu(
                recurrent_state,
                state_indices,
                has_initial_state,
            )
            out, last_recurrent_state = kda_vllm_prefill.chunk_kda_with_fused_gate(
                q=q_ns.reshape(shape),
                k=k_ns.reshape(shape),
                v=v_ns.reshape(shape),
                raw_g=g1,
                raw_beta=beta,
                A_log=self.A_log,
                g_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc,
            )
            recurrent_state.index_copy_(0, slots, last_recurrent_state)
        else:
            mixed_qkv, new_windows = _run_compiled_conv_step(
                mixed_qkv, windows, conv_weights, self.conv1d.bias
            )
            out, _ = kda_vllm_decode.fused_recurrent_kda_packed_decode(
                mixed_qkv=mixed_qkv,
                raw_g=g1,
                raw_beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
                initial_state=recurrent_state,
                state_indices=state_indices,
            )

        if not dim_first:
            new_windows = new_windows.transpose(-1, -2)
        conv_cache.index_copy_(0, slots, new_windows.contiguous().to(conv_cache.dtype))

        normed_out = self.o_norm(
            out[:, :num_actual_tokens],
            g2[:, :num_actual_tokens],
        )
        # Padded rows are left for the runner to drop; keep them finite.
        if is_3d_out:
            core_attn_out[:num_actual_tokens] = normed_out[0]
            core_attn_out[num_actual_tokens:].zero_()
        else:
            core_attn_out[0, :num_actual_tokens] = normed_out[0]
            core_attn_out[0, num_actual_tokens:].zero_()
