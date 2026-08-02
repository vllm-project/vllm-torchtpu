# SPDX-License-Identifier: Apache-2.0
"""Module-level accuracy tests for the TPU Kimi KDA layer.

Validates ``KimiDeltaAttention`` end to end -- projections, short conv1d, KDA
recurrence, gated norm, output projection and both KV-state pools -- against a
naive float32 CPU reference in this file.

Three interchangeable recurrences live behind the layer's one signature and all
three are covered here:

* ``decode_kda_op`` -- fused conv + recurrent Pallas kernel, one token per
  sequence. Selected when the batch is pure decode.
* ``chunk_kda_op`` -- chunked Pallas kernel. Selected for everything else, and
  correct for a 1-token row too (it is just a length-1 segment).
* ``kda_op`` -- the plain-XLA reference scan, selected for every batch when
  ``VLLM_TPU_USE_NAIVE_KDA=1``.

Every kernel here reads the recurrent pool as ``[num_slots, H, K, V]``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.mamba.mamba_utils import \
    is_conv_state_dim_first

import vllm_torchtpu.layers.vllm.custom_ops.kda_attention_op as kda_ops
from vllm_torchtpu.models.vllm.kimi_k3.attention import KimiDeltaAttention


def _has_tpu() -> bool:
    try:
        import torch_tpu  # noqa: F401
        return True
    except Exception:
        return False


requires_tpu = pytest.mark.skipif(not _has_tpu(),
                                  reason="needs a TPU (torch_tpu)")

# Small by design: these are accuracy tests, and the reference is a Python loop
# over every token. head_dim 64 and 128 are both exercised because 128 is the
# real checkpoint's value and the kernels tile the channel axis.
CONV_SIZE = 4
GATE_LOWER_BOUND = -5.0
O_NORM_EPS = 1e-5
L2_EPS = 1e-6

# Which recurrence a test wants. "auto" is production behaviour: the layer picks
# the decode kernel for a pure-decode batch and the chunked one otherwise.
PATH_AUTO = "auto"
PATH_NAIVE = "naive"
PATH_CHUNK = "chunk"


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------
class ReferenceKDALayer:
    """Naive float32 CPU implementation of the whole KDA module."""

    def __init__(self, weights: dict[str, torch.Tensor], num_heads: int,
                 head_dim: int, conv_size: int, lower_bound: float | None,
                 o_norm_eps: float) -> None:
        self.w = {
            k: v.detach().to(torch.float32).cpu()
            for k, v in weights.items()
        }
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_size = conv_size
        self.lower_bound = lower_bound
        self.o_norm_eps = o_norm_eps
        self.proj = num_heads * head_dim
        self.scale = head_dim**-0.5
        self.conv_state: torch.Tensor | None = None
        self.recurrent_state: torch.Tensor | None = None

    # -- pieces -------------------------------------------------------------
    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        """``x * rsqrt(sum(x^2) + eps)``, matching ``l2norm_fwd``.

        Note this is not ``F.normalize``, which clamps the denominator instead of
        adding to it. At these magnitudes the difference is negligible, but the
        kernels do it this way so the reference does too.
        """
        return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + L2_EPS)

    def _activate_gate(self, raw: torch.Tensor) -> torch.Tensor:
        """Raw gate -> log-space decay, per (head, channel).

        ``raw`` is ``[..., num_heads, head_dim]``. Mirrors
        ``kernels/kimi_k3/decode_kda.py:_activate_gate``: add ``dt_bias``, then either the
        unbounded ``-exp(A_log) * softplus(.)`` branch or the bounded
        ``lower_bound * sigmoid(exp(A_log) * .)`` one. Both are <= 0, which the
        chunked kernel's intra-chunk decay requires.
        """
        g = raw + self.w["dt_bias"].reshape(self.num_heads, self.head_dim)
        a = torch.exp(self.w["A_log"]).reshape(self.num_heads, 1)
        if self.lower_bound is None:
            return -a * torch.nn.functional.softplus(g)
        return self.lower_bound * torch.sigmoid(a * g)

    def _conv_one_sequence(self, x: torch.Tensor, slot: int,
                           has_init: bool) -> torch.Tensor:
        """Causal depthwise conv1d + SiLU over one sequence, token by token.

        ``x`` is ``[t, 3 * proj]``. Left-pads with the slot's carried state (or
        zeros), then writes the trailing ``conv_size - 1`` tokens back as the next
        step's state.
        """
        weight = self.w["conv1d"]  # [3 * proj, conv_size]
        state_len = self.conv_size - 1
        if has_init:
            history = self.conv_state[slot].clone()
        else:
            history = torch.zeros(state_len, x.shape[1])

        stream = torch.cat([history, x], dim=0)  # [state_len + t, dim]
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            window = stream[t:t + self.conv_size]  # [conv_size, dim]
            # weight[:, -1] multiplies the current token; earlier columns the
            # history, oldest first.
            acc = torch.zeros(x.shape[1])
            for j in range(self.conv_size):
                acc = acc + window[j] * weight[:, j]
            out[t] = torch.nn.functional.silu(acc)
        self.conv_state[slot] = stream[-state_len:]
        return out

    def _recurrence_one_sequence(self, q: torch.Tensor, k: torch.Tensor,
                                 v: torch.Tensor, log_g: torch.Tensor,
                                 beta: torch.Tensor, slot: int,
                                 has_init: bool) -> torch.Tensor:
        """KDA delta rule over one sequence, one token per iteration.

        ``q``/``k``/``v``/``log_g`` are ``[t, num_heads, head_dim]`` and ``beta``
        is ``[t, num_heads]``. The state ``S`` is ``[num_heads, K, V]``.
        """
        if has_init:
            state = self.recurrent_state[slot].clone()
        else:
            state = torch.zeros(self.num_heads, self.head_dim, self.head_dim)

        out = torch.zeros_like(v)
        for t in range(q.shape[0]):
            q_t = self._l2norm(q[t]) * self.scale  # [H, K]
            k_t = self._l2norm(k[t])  # [H, K]
            v_t = v[t]  # [H, V]

            # Per-channel decay: scales row k of the state, broadcast over V.
            state = state * torch.exp(log_g[t]).unsqueeze(-1)
            # Delta rule: correct the state towards v_t, at rate beta.
            v_pred = torch.einsum("hk,hkv->hv", k_t, state)
            v_new = beta[t].unsqueeze(-1) * (v_t - v_pred)
            state = state + torch.einsum("hk,hv->hkv", k_t, v_new)
            out[t] = torch.einsum("hk,hkv->hv", q_t, state)

        self.recurrent_state[slot] = state
        return out

    # -- the module ---------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor, query_start_loc: list[int],
                state_indices: list[int],
                has_initial_state: list[bool]) -> torch.Tensor:
        """One KDA module forward over a packed batch. Returns ``[T, hidden]``.

        Mutates ``self.conv_state`` / ``self.recurrent_state`` in place, as the
        layer mutates its ``kv_cache``.
        """
        assert self.conv_state is not None and self.recurrent_state is not None
        x = hidden_states.to(torch.float32).cpu()
        num_tokens = x.shape[0]

        # 1. Input projections. Separate, as in the layer.
        mixed_qkv = x @ self.w["in_proj_qkv"].T  # [T, 3 * proj]
        g2_raw = x @ self.w["g_proj"].T  # [T, proj]
        f_a = x @ self.w["f_a_proj"].T  # [T, head_dim]
        beta_raw = x @ self.w["b_proj"].T  # [T, num_heads]

        # 2. The gate goes through a second, low-rank projection.
        g1_raw = f_a @ self.w["f_b_proj"].T  # [T, proj]

        # 3. beta arrives raw from the projection; the layer activates it.
        beta = torch.sigmoid(beta_raw)  # [T, num_heads]

        # 4. Short causal conv1d + SiLU, per sequence.
        conv_out = torch.zeros_like(mixed_qkv)
        for i in range(len(state_indices)):
            s, e = query_start_loc[i], query_start_loc[i + 1]
            if e <= s:
                continue
            conv_out[s:e] = self._conv_one_sequence(mixed_qkv[s:e],
                                                    state_indices[i],
                                                    bool(has_initial_state[i]))

        # 5. Split into q/k/v and fold the head axis out.
        q, k, v = (t.reshape(num_tokens, self.num_heads, self.head_dim)
                   for t in conv_out.split(self.proj, dim=-1))
        log_g = self._activate_gate(
            g1_raw.reshape(num_tokens, self.num_heads, self.head_dim))

        # 6. The recurrence, per sequence.
        core = torch.zeros(num_tokens, self.num_heads, self.head_dim)
        for i in range(len(state_indices)):
            s, e = query_start_loc[i], query_start_loc[i + 1]
            if e <= s:
                continue
            core[s:e] = self._recurrence_one_sequence(
                q[s:e], k[s:e], v[s:e], log_g[s:e], beta[s:e],
                state_indices[i], bool(has_initial_state[i]))

        # 7. Gated RMSNorm over head_dim, gated by sigmoid(g2).
        g2 = g2_raw.reshape(num_tokens, self.num_heads, self.head_dim)
        variance = core.pow(2).mean(-1, keepdim=True)
        normed = core * torch.rsqrt(variance + self.o_norm_eps)
        normed = normed * self.w["o_norm"]
        gated = normed * torch.sigmoid(g2)

        # 8. Output projection.
        return gated.reshape(num_tokens, self.proj) @ self.w["o_proj"].T


# ---------------------------------------------------------------------------
# Building the TPU layer without a vLLM engine
# ---------------------------------------------------------------------------
class _WrappedLinear(nn.Module):
    """``nn.Linear`` whose forward returns ``(out, bias)`` like vLLM's linears."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class _ConvHolder(nn.Module):
    """Holds ``weight`` of shape ``[proj, 1, conv_size]``, like ``CausalDepthwiseConv1d``."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.unsqueeze(1))


class _NormWeight(nn.Module):
    """Stand-in for ``RMSNorm``: only ``weight`` is read, by the kernels."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)


class _Metadata:
    """The subset of ``AttentionMetadata`` the KDA layer reads.

    Constructed directly rather than through the real builder, which needs a
    runner and an initialized KV cache. ``seq_lens`` carries the carry-in flag
    the way the production metadata does: a sequence carries state iff its
    ``seq_len`` exceeds the number of tokens scheduled for it this step.
    """

    def __init__(self, *, query_start_loc: torch.Tensor,
                 mamba_state_indices: torch.Tensor, seq_lens: torch.Tensor,
                 is_decode_only: bool) -> None:
        self.query_start_loc = query_start_loc
        self.mamba_state_indices = mamba_state_indices
        self.seq_lens = seq_lens
        self.is_decode_only = is_decode_only


def _make_weights(num_heads: int, head_dim: int, hidden_size: int,
                  conv_size: int, seed: int) -> dict[str, torch.Tensor]:
    """Random float32 weights, in the reference's naming.

    Scales are small on purpose. The delta rule is a feedback loop: with
    unit-scale projections the state diverges within a few hundred tokens and the
    *reference* is the first thing to produce NaN, which would look like a kernel
    bug. The real checkpoint's weights are small for the same reason.
    """
    gen = torch.Generator().manual_seed(seed)
    proj = num_heads * head_dim

    def randn(*shape, scale: float):
        return torch.randn(*shape, generator=gen, dtype=torch.float32) * scale

    return {
        "in_proj_qkv": randn(3 * proj, hidden_size, scale=0.02),
        "g_proj": randn(proj, hidden_size, scale=0.02),
        "f_a_proj": randn(head_dim, hidden_size, scale=0.02),
        "b_proj": randn(num_heads, hidden_size, scale=0.02),
        "f_b_proj": randn(proj, head_dim, scale=0.02),
        "conv1d": randn(3 * proj, conv_size, scale=0.1),
        "dt_bias": randn(proj, scale=0.1),
        "A_log": randn(num_heads, scale=0.5),
        "o_norm": randn(head_dim, scale=0.2),
        "o_proj": randn(hidden_size, proj, scale=0.02),
    }


def _build_tpu_layer(weights: dict[str, torch.Tensor], num_heads: int,
                     head_dim: int, hidden_size: int, conv_size: int,
                     lower_bound: float | None, dtype, device, *,
                     use_naive_kda: bool):
    """Hand-build ``KimiDeltaAttention`` on the given weights.

    The real ``__init__`` builds vLLM parallel linears, which needs an
    initialized TP group, which needs a ``ModelConfig`` -- i.e. a model on disk.
    This bypasses that and substitutes plain ``nn.Linear``, which at TP=1 is
    numerically identical (``nn.Linear.weight`` is ``[out, in]``, same as
    ``ColumnParallelLinear``'s and ``ReplicatedLinear``'s). The custom ops are
    the real ones.

    ``_assert_layer_attrs`` below is the guard against this drifting from the
    real ``__init__``.
    """
    m = object.__new__(KimiDeltaAttention)
    nn.Module.__init__(m)

    proj = num_heads * head_dim
    m.prefix = "model.layers.0.linear_attn"
    m.head_dim = head_dim
    m.total_heads = num_heads
    m.num_heads = num_heads
    m.tp_size = 1
    m.projection_size = proj
    m.local_projection_size = proj
    m.conv_size = conv_size
    m.gate_lower_bound = lower_bound
    m.use_full_rank_gate = True
    m.kv_cache = None
    m.use_naive_kda = use_naive_kda

    def to_dev(t, dt):
        return t.detach().clone().to(dt).to(device)

    # The layer projects q, k and v separately and concatenates; the reference
    # keeps one [3 * proj, hidden] weight. Same GEMM, so split it three ways.
    q_w, k_w, v_w = weights["in_proj_qkv"].split(proj, dim=0)
    m.q_proj = _WrappedLinear(to_dev(q_w, dtype))
    m.k_proj = _WrappedLinear(to_dev(k_w, dtype))
    m.v_proj = _WrappedLinear(to_dev(v_w, dtype))
    m.g_proj = _WrappedLinear(to_dev(weights["g_proj"], dtype))
    m.f_a_proj = _WrappedLinear(to_dev(weights["f_a_proj"], dtype))
    m.b_proj = _WrappedLinear(to_dev(weights["b_proj"], dtype))
    m.f_b_proj = _WrappedLinear(to_dev(weights["f_b_proj"], dtype))
    m.o_proj = _WrappedLinear(to_dev(weights["o_proj"], dtype))
    # Same split for the depthwise convolution, which the layer also keeps per
    # projection. conv1d, dt_bias and A_log are float32 in the layer.
    q_c, k_c, v_c = weights["conv1d"].split(proj, dim=0)
    m.q_conv1d = _ConvHolder(to_dev(q_c, torch.float32))
    m.k_conv1d = _ConvHolder(to_dev(k_c, torch.float32))
    m.v_conv1d = _ConvHolder(to_dev(v_c, torch.float32))
    m.dt_bias = nn.Parameter(to_dev(weights["dt_bias"], torch.float32))
    m.A_log = nn.Parameter(to_dev(weights["A_log"], torch.float32))
    m.o_norm = _NormWeight(to_dev(weights["o_norm"], torch.float32))

    m.sconv_op = kda_ops.build_kimi_sconv_op(
        m.prefix,
        kernel_size=conv_size,
        state_dim_first=is_conv_state_dim_first(),
    )
    m.kda_op = kda_ops.build_kimi_kda_op(
        m.prefix,
        lower_bound=lower_bound,
        eps=O_NORM_EPS,
    )
    m.chunk_kda_op = kda_ops.build_kimi_chunk_kda_op(
        m.prefix,
        lower_bound=lower_bound,
        eps=O_NORM_EPS,
    )
    m.decode_kda_op = kda_ops.build_kimi_decode_kda_op(
        m.prefix,
        lower_bound=lower_bound,
        eps=O_NORM_EPS,
        state_dim_first=is_conv_state_dim_first(),
    )
    return m


def _assert_layer_attrs(layer) -> None:
    """Fail loudly if the layer grows a projection this harness does not set.

    The hand-build above skips ``__init__``, so a new submodule added to the
    layer would otherwise surface as an ``AttributeError`` deep inside
    ``forward`` -- or worse, as a silently stale value from a previous test.
    """
    expected = {
        "q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj",
        "f_b_proj", "o_proj", "q_conv1d", "k_conv1d", "v_conv1d", "o_norm"
    }
    actual = {name for name, _ in layer.named_children()}
    assert actual == expected, (
        f"the layer's submodules changed: missing={sorted(expected - actual)} "
        f"unexpected={sorted(actual - expected)}. Update _build_tpu_layer.")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _assert_close(name: str, ref: torch.Tensor, got: torch.Tensor,
                  rmse_ratio: float) -> None:
    """RMSE-relative comparison, the metric the KDA kernel tests use.

    Max-abs alone is a poor gate on a recurrence: one token's rounding propagates
    to every later token, so an otherwise-correct bf16 run has a few outlying
    elements. The RMSE ratio is what distinguishes that from a real error.
    """
    ref = ref.detach().float().cpu()
    got = got.detach().float().cpu()
    assert ref.shape == got.shape, f"{name}: {got.shape} != {ref.shape}"
    assert not torch.isnan(got).any(), f"{name}: NaN in the TPU result"
    assert not torch.isnan(ref).any(), f"{name}: NaN in the reference"

    max_abs = (ref - got).abs().max().item()
    rmse = (ref - got).pow(2).mean().sqrt().item()
    base = ref.pow(2).mean().sqrt().item()
    rel = rmse / (base + 1e-8)
    print(f"{name:>38} | max_abs={max_abs:.6f} | rmse_rel={rel:.6f} "
          f"| thr={rmse_ratio}")
    assert rel < rmse_ratio, (
        f"{name}: rmse_rel {rel:.6f} >= {rmse_ratio} (max_abs {max_abs:.6f})")


def _make_pools(num_slots: int, num_heads: int, head_dim: int, conv_size: int,
                seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random initial state pools.

    The conv pool is returned in the ``[num_slots, state_len, dim]`` ("SD")
    orientation the reference works in; :func:`_orient_conv` converts to whatever
    this platform actually stores. Slot 0 is left as the reserved null block;
    tests assign real sequences from slot 1 up, matching the runner's contract and
    the decode kernel's ``slot > 0`` validity check.
    """
    gen = torch.Generator().manual_seed(seed)
    conv_dim = 3 * num_heads * head_dim
    conv_state = torch.randn(
        num_slots, conv_size - 1, conv_dim, generator=gen,
        dtype=torch.float32) * 0.1
    recurrent_state = torch.randn(num_slots,
                                  num_heads,
                                  head_dim,
                                  head_dim,
                                  generator=gen,
                                  dtype=torch.float32) * 0.01
    return conv_state, recurrent_state


def _conv_state_dim_first() -> bool:
    """Whether this platform stores the conv state as ``[dim, state_len]``.

    Set by ``VLLM_SSM_CONV_STATE_LAYOUT``, not by the platform, so it can differ
    between runs. The layer normalizes internally; these tests have to hand it a
    pool in the orientation it expects.
    """
    return bool(is_conv_state_dim_first())


def _orient_conv(conv_state: torch.Tensor) -> torch.Tensor:
    """Convert between the reference's "SD" layout and this platform's.

    Its own inverse, so the same call converts reference -> layer before the
    forward and layer -> reference before comparing.
    """
    if _conv_state_dim_first():
        return conv_state.transpose(-1, -2).contiguous()
    return conv_state


def _run_pair(*,
              num_heads,
              head_dim,
              hidden_size,
              query_lens,
              has_initial_state,
              lower_bound,
              dtype,
              device,
              seed,
              path=PATH_AUTO,
              conv_state=None,
              recurrent_state=None,
              weights=None,
              reference=None,
              hidden_states=None):
    """Run the same batch through the TPU layer and the CPU reference.

    ``path`` selects the recurrence: ``PATH_AUTO`` leaves the layer to choose
    (decode kernel for a pure-decode batch, chunked otherwise), ``PATH_NAIVE``
    forces the reference scan, and ``PATH_CHUNK`` forces the chunked kernel even
    for a pure-decode batch -- which is how the 1-token-segment case gets covered.

    Returns ``(ref_out, tpu_out, reference, layer)`` where ``reference`` carries
    the updated CPU pools and ``layer.kv_cache`` the TPU ones, so a caller can
    make a second call that continues from this one.
    """
    assert path in (PATH_AUTO, PATH_NAIVE, PATH_CHUNK)
    num_seqs = len(query_lens)
    num_tokens = sum(query_lens)
    query_start_loc = [0]
    for q in query_lens:
        query_start_loc.append(query_start_loc[-1] + q)
    # Slot i+1: slot 0 is reserved.
    state_indices = list(range(1, num_seqs + 1))

    if weights is None:
        weights = _make_weights(num_heads, head_dim, hidden_size, CONV_SIZE,
                                seed)
    if conv_state is None or recurrent_state is None:
        conv_state, recurrent_state = _make_pools(num_seqs + 1, num_heads,
                                                  head_dim, CONV_SIZE,
                                                  seed + 1)

    if hidden_states is None:
        gen = torch.Generator().manual_seed(seed + 2)
        hidden_states = torch.randn(
            num_tokens, hidden_size, generator=gen, dtype=torch.float32) * 0.1
    assert hidden_states.shape == (num_tokens, hidden_size), (
        f"hidden_states {tuple(hidden_states.shape)} does not match this batch's "
        f"{(num_tokens, hidden_size)}")

    # --- reference ---
    if reference is None:
        reference = ReferenceKDALayer(weights, num_heads, head_dim, CONV_SIZE,
                                      lower_bound, O_NORM_EPS)
        reference.conv_state = conv_state.clone()
        reference.recurrent_state = recurrent_state.clone()
    ref_out = reference.forward(hidden_states, query_start_loc, state_indices,
                                has_initial_state)

    # --- TPU layer ---
    layer = _build_tpu_layer(weights,
                             num_heads,
                             head_dim,
                             hidden_size,
                             CONV_SIZE,
                             lower_bound,
                             dtype,
                             device,
                             use_naive_kda=path == PATH_NAIVE)
    _assert_layer_attrs(layer)
    layer.kv_cache = (_orient_conv(conv_state.clone()).to(device),
                      recurrent_state.clone().to(device))

    qsl = torch.tensor(query_start_loc, dtype=torch.int32)
    query_lens_t = torch.tensor(query_lens, dtype=torch.int32)
    carry = torch.tensor([1 if h else 0 for h in has_initial_state],
                         dtype=torch.int32)
    is_decode_only = all(q == 1 for q in query_lens) and path != PATH_CHUNK
    meta = _Metadata(
        query_start_loc=qsl.to(device),
        mamba_state_indices=torch.tensor(state_indices,
                                         dtype=torch.int32).to(device),
        # The ops read carry-in as seq_len > query_len, so +1 is enough to mark
        # it; the absolute sequence length is not used.
        seq_lens=(query_lens_t + carry).to(device),
        is_decode_only=is_decode_only,
    )
    layer._metadata = lambda: meta
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)
    with torch.inference_mode():
        tpu_out = layer(positions, hidden_states.to(dtype).to(device))

    return ref_out, tpu_out, reference, layer


def _final_pools(layer) -> tuple[torch.Tensor, torch.Tensor]:
    """The layer's pools on CPU, conv back in the reference's "SD" orientation."""
    conv_after, rec_after = layer.kv_cache
    return _orient_conv(conv_after.cpu()), rec_after.cpu()


def _touched(state_indices: list[int]) -> torch.Tensor:
    return torch.tensor(sorted(set(state_indices)), dtype=torch.long)


# ---------------------------------------------------------------------------
# The reference, checked against itself
# ---------------------------------------------------------------------------
def test_naive_reference_is_self_consistent():
    """Splitting a sequence in two, carrying state, must match one pass.

    This is the only guard on the reference, and it is a real one: it exercises
    both pools' carry (conv history and recurrent state) and would catch an
    off-by-one in the conv window, a wrong state orientation, or a decay applied
    at the wrong point. It runs on CPU, so it runs everywhere.
    """
    num_heads, head_dim, hidden_size = 2, 64, 128
    weights = _make_weights(num_heads,
                            head_dim,
                            hidden_size,
                            CONV_SIZE,
                            seed=0)
    conv_state, recurrent_state = _make_pools(2,
                                              num_heads,
                                              head_dim,
                                              CONV_SIZE,
                                              seed=1)
    gen = torch.Generator().manual_seed(2)
    total = 96
    x = torch.randn(total, hidden_size, generator=gen,
                    dtype=torch.float32) * 0.1

    def fresh() -> ReferenceKDALayer:
        r = ReferenceKDALayer(weights, num_heads, head_dim, CONV_SIZE,
                              GATE_LOWER_BOUND, O_NORM_EPS)
        r.conv_state = conv_state.clone()
        r.recurrent_state = recurrent_state.clone()
        return r

    whole_ref = fresh()
    whole = whole_ref.forward(x, [0, total], [1], [False])

    split_ref = fresh()
    cut = 32
    first = split_ref.forward(x[:cut], [0, cut], [1], [False])
    # Second half carries state in -- has_initial_state=True.
    second = split_ref.forward(x[cut:], [0, total - cut], [1], [True])

    _assert_close("reference split output", whole,
                  torch.cat([first, second], dim=0), 1e-6)
    _assert_close("reference split conv_state", whole_ref.conv_state,
                  split_ref.conv_state, 1e-6)
    _assert_close("reference split recurrent_state", whole_ref.recurrent_state,
                  split_ref.recurrent_state, 1e-6)


def test_naive_reference_keeps_slots_independent():
    """Two sequences in one batch must not touch each other's slots.

    Guards the per-sequence loop's slot routing: running sequence A alone must
    give the same answer as running A and B together.
    """
    num_heads, head_dim, hidden_size = 2, 64, 128
    weights = _make_weights(num_heads,
                            head_dim,
                            hidden_size,
                            CONV_SIZE,
                            seed=3)
    conv_state, recurrent_state = _make_pools(3,
                                              num_heads,
                                              head_dim,
                                              CONV_SIZE,
                                              seed=4)
    gen = torch.Generator().manual_seed(5)
    t_a, t_b = 24, 40
    x = torch.randn(t_a + t_b, hidden_size, generator=gen,
                    dtype=torch.float32) * 0.1

    def fresh() -> ReferenceKDALayer:
        r = ReferenceKDALayer(weights, num_heads, head_dim, CONV_SIZE,
                              GATE_LOWER_BOUND, O_NORM_EPS)
        r.conv_state = conv_state.clone()
        r.recurrent_state = recurrent_state.clone()
        return r

    both_ref = fresh()
    both = both_ref.forward(x, [0, t_a, t_a + t_b], [1, 2], [True, True])

    alone_ref = fresh()
    alone = alone_ref.forward(x[:t_a], [0, t_a], [1], [True])

    _assert_close("slot-1 output alone vs batched", alone, both[:t_a], 1e-6)
    _assert_close("slot-1 recurrent alone vs batched",
                  alone_ref.recurrent_state[1], both_ref.recurrent_state[1],
                  1e-6)
    # Slot 2 must be untouched in the single-sequence run.
    assert torch.equal(alone_ref.recurrent_state[2], recurrent_state[2])


# ---------------------------------------------------------------------------
# Dispatch, on the host
# ---------------------------------------------------------------------------
def _stub_dispatch_layer(*, use_naive_kda: bool):
    """A layer whose four custom ops just record that they ran."""
    layer = object.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.use_naive_kda = use_naive_kda
    layer.q_conv1d = _ConvHolder(torch.zeros(2, 3))
    layer.k_conv1d = _ConvHolder(torch.zeros(2, 3))
    layer.v_conv1d = _ConvHolder(torch.zeros(2, 3))
    layer.o_norm = _NormWeight(torch.ones(1))
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(2))

    calls: list[str] = []

    def record(name, result):

        def op(*args, **kwargs):
            calls.append(name)
            return result

        return op

    activations = torch.zeros(2, 6)
    layer.sconv_op = record("sconv", activations)
    layer.kda_op = record("naive", torch.zeros(2, 2, 1))
    layer.chunk_kda_op = record("chunk", torch.zeros(2, 2, 1))
    layer.decode_kda_op = record("decode", torch.zeros(2, 2, 1))
    return layer, calls, activations


@pytest.mark.parametrize("is_decode_only,use_naive_kda,expected", [
    (True, False, ["decode"]),
    (False, False, ["sconv", "chunk"]),
    (True, True, ["sconv", "naive"]),
    (False, True, ["sconv", "naive"]),
])
def test_core_attention_selects_the_expected_ops(is_decode_only, use_naive_kda,
                                                 expected):
    """Which of the three recurrences runs, and whether `sconv_op` runs with it.

    A host-side branch, so this runs anywhere. It is worth pinning: the decode op
    fuses the short convolution, so running `sconv_op` in front of it would
    convolve twice and quietly corrupt every decode step.
    """
    layer, calls, activations = _stub_dispatch_layer(
        use_naive_kda=use_naive_kda)
    metadata = _Metadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        is_decode_only=is_decode_only,
    )
    layer._core_attention(
        activations,
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.zeros(3, 3, 6),
        torch.zeros(3, 2, 1, 1),
        metadata,
    )
    assert calls == expected


# ---------------------------------------------------------------------------
# The layer, against the reference
# ---------------------------------------------------------------------------
# The recurrence accumulates in float32 but consumes bfloat16 projections.
PREFILL_RMSE = 2e-2
DECODE_RMSE = 2e-2
STATE_RMSE = 2e-2
# The chained case compounds prefill error through the decode state.
CHAINED_RMSE = 3e-2


@requires_tpu
@pytest.mark.parametrize("query_lens", [[64], [96], [32, 64], [17, 48, 96]])
def test_prefill_matches_naive_cpu_reference(query_lens):
    """Chunked prefill, no carried state, against the CPU reference.

    ``query_lens`` covers a single chunk-aligned sequence, a sequence spanning
    two chunks, and ragged multi-sequence batches including lengths that are not
    chunk multiples -- which is where the varlen segment handling shows up.
    """
    num_heads, head_dim, hidden_size = 2, 64, 128
    state_indices = list(range(1, len(query_lens) + 1))

    ref_out, tpu_out, reference, layer = _run_pair(
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        query_lens=query_lens,
        has_initial_state=[False] * len(query_lens),
        lower_bound=GATE_LOWER_BOUND,
        dtype=torch.bfloat16,
        device=torch.device("tpu"),
        seed=10)

    tag = f"prefill{query_lens}"
    _assert_close(f"{tag} output", ref_out, tpu_out, PREFILL_RMSE)

    conv_after, rec_after = _final_pools(layer)
    slots = _touched(state_indices)
    _assert_close(f"{tag} conv_state", reference.conv_state[slots],
                  conv_after[slots], STATE_RMSE)
    _assert_close(f"{tag} recurrent_state", reference.recurrent_state[slots],
                  rec_after[slots], STATE_RMSE)


@requires_tpu
@pytest.mark.parametrize("lower_bound", [GATE_LOWER_BOUND, None])
def test_prefill_gate_branches_match_reference(lower_bound):
    """Both gate activations.

    ``lower_bound=-5.0`` is the real config's value and selects
    ``lower_bound * sigmoid(exp(A_log) * g)``. ``None`` selects the unbounded
    ``-exp(A_log) * softplus(g)`` branch, whose decay is far more aggressive --
    getting the two confused is a plausible port error that this separates.
    """
    ref_out, tpu_out, reference, layer = _run_pair(num_heads=2,
                                                   head_dim=64,
                                                   hidden_size=128,
                                                   query_lens=[64],
                                                   has_initial_state=[False],
                                                   lower_bound=lower_bound,
                                                   dtype=torch.bfloat16,
                                                   device=torch.device("tpu"),
                                                   seed=11)

    tag = f"prefill lb={lower_bound}"
    _assert_close(f"{tag} output", ref_out, tpu_out, PREFILL_RMSE)
    _, rec_after = _final_pools(layer)
    _assert_close(f"{tag} recurrent_state", reference.recurrent_state[[1]],
                  rec_after[[1]], STATE_RMSE)


@requires_tpu
def test_prefill_head_dim_128_matches_reference():
    """The real checkpoint's ``head_dim``, which tiles the channel axis fully."""
    ref_out, tpu_out, reference, layer = _run_pair(
        num_heads=2,
        head_dim=128,
        hidden_size=256,
        query_lens=[64],
        has_initial_state=[False],
        lower_bound=GATE_LOWER_BOUND,
        dtype=torch.bfloat16,
        device=torch.device("tpu"),
        seed=12)

    _assert_close("prefill d128 output", ref_out, tpu_out, PREFILL_RMSE)
    _, rec_after = _final_pools(layer)
    _assert_close("prefill d128 recurrent_state",
                  reference.recurrent_state[[1]], rec_after[[1]], STATE_RMSE)


@requires_tpu
def test_chunk_path_leaves_rows_with_no_scheduled_tokens_alone():
    """A request scheduled zero tokens this step must keep its state.

    The chunked kernel hands back a final state for *every* segment it is given,
    including empty ones, where that state is the (zeroed) initial state. Writing
    those back would wipe a live slot, so the bridge routes them to the reserved
    null block instead. Padded request slots in a real batch are exactly this
    case, which makes it a silent-corruption bug rather than a crash.
    """
    conv_state, recurrent_state = _make_pools(3, 2, 64, CONV_SIZE, seed=51)
    _, _, _, layer = _run_pair(num_heads=2,
                               head_dim=64,
                               hidden_size=128,
                               query_lens=[32, 0],
                               has_initial_state=[False, False],
                               lower_bound=GATE_LOWER_BOUND,
                               dtype=torch.bfloat16,
                               device=torch.device("tpu"),
                               seed=50,
                               conv_state=conv_state,
                               recurrent_state=recurrent_state)

    conv_after, rec_after = _final_pools(layer)
    assert torch.equal(
        rec_after[2],
        recurrent_state[2]), ("the empty row's recurrent slot was overwritten")
    assert torch.equal(
        conv_after[2],
        conv_state[2]), ("the empty row's conv slot was overwritten")


@requires_tpu
@pytest.mark.parametrize("num_seqs", [1, 4, 8])
def test_decode_matches_naive_cpu_reference(num_seqs):
    """Pure decode -- one token per sequence through the recurrent kernel.

    Every sequence carries state in, which is the real decode case and the one
    that exercises the paged state gather inside the kernel.
    """
    state_indices = list(range(1, num_seqs + 1))
    ref_out, tpu_out, reference, layer = _run_pair(
        num_heads=2,
        head_dim=64,
        hidden_size=128,
        query_lens=[1] * num_seqs,
        has_initial_state=[True] * num_seqs,
        lower_bound=GATE_LOWER_BOUND,
        dtype=torch.bfloat16,
        device=torch.device("tpu"),
        seed=20 + num_seqs)

    tag = f"decode n{num_seqs}"
    _assert_close(f"{tag} output", ref_out, tpu_out, DECODE_RMSE)

    conv_after, rec_after = _final_pools(layer)
    slots = _touched(state_indices)
    _assert_close(f"{tag} conv_state", reference.conv_state[slots],
                  conv_after[slots], STATE_RMSE)
    _assert_close(f"{tag} recurrent_state", reference.recurrent_state[slots],
                  rec_after[slots], STATE_RMSE)


@requires_tpu
def test_decode_without_initial_state_ignores_stale_slot_contents():
    """A new sequence must start from zero even when its slot contains data."""
    ref_out, tpu_out, reference, layer = _run_pair(
        num_heads=2,
        head_dim=64,
        hidden_size=128,
        query_lens=[1],
        has_initial_state=[False],
        lower_bound=GATE_LOWER_BOUND,
        dtype=torch.bfloat16,
        device=torch.device("tpu"),
        seed=29,
    )

    _assert_close("fresh decode output", ref_out, tpu_out, DECODE_RMSE)
    conv_after, rec_after = _final_pools(layer)
    _assert_close("fresh decode conv_state", reference.conv_state[[1]],
                  conv_after[[1]], STATE_RMSE)
    _assert_close("fresh decode recurrent_state",
                  reference.recurrent_state[[1]], rec_after[[1]], STATE_RMSE)


@requires_tpu
def test_decode_leaves_untouched_slots_alone():
    """A decode batch must not disturb pool slots it was not given.

    The pools are shared across all sequences in flight, so a kernel that writes
    outside its slot corrupts other requests -- a failure that per-slot
    comparisons alone would miss.
    """
    num_seqs = 2
    num_slots = 6
    conv_state, recurrent_state = _make_pools(num_slots,
                                              2,
                                              64,
                                              CONV_SIZE,
                                              seed=31)
    _, _, _, layer = _run_pair(num_heads=2,
                               head_dim=64,
                               hidden_size=128,
                               query_lens=[1] * num_seqs,
                               has_initial_state=[True] * num_seqs,
                               lower_bound=GATE_LOWER_BOUND,
                               dtype=torch.bfloat16,
                               device=torch.device("tpu"),
                               seed=30,
                               conv_state=conv_state,
                               recurrent_state=recurrent_state)

    conv_after, rec_after = _final_pools(layer)
    untouched = list(range(num_seqs + 1, num_slots))
    assert untouched, "test needs slots beyond the batch"
    for slot in untouched:
        assert torch.equal(rec_after[slot], recurrent_state[slot]), (
            f"recurrent slot {slot} was modified but not in the batch")
        assert torch.equal(conv_after[slot], conv_state[slot]), (
            f"conv slot {slot} was modified but not in the batch")


@requires_tpu
def test_decode_continues_prefill():
    """Prefill, then decode the next token from the state prefill wrote.

    The end-to-end property that matters in serving, and the one that catches a
    state-layout mismatch *between* the two kernels: prefill writes the pool and
    decode reads it back, so a transposed or wrongly-scaled state that both
    kernels agree on internally still fails here.
    """
    num_heads, head_dim, hidden_size = 2, 64, 128
    prefill_lens = [32, 48]
    num_seqs = len(prefill_lens)
    weights = _make_weights(num_heads,
                            head_dim,
                            hidden_size,
                            CONV_SIZE,
                            seed=40)
    conv_state, recurrent_state = _make_pools(num_seqs + 1,
                                              num_heads,
                                              head_dim,
                                              CONV_SIZE,
                                              seed=41)

    ref_out, tpu_out, reference, layer = _run_pair(
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        query_lens=prefill_lens,
        has_initial_state=[False] * num_seqs,
        lower_bound=GATE_LOWER_BOUND,
        dtype=torch.bfloat16,
        device=torch.device("tpu"),
        seed=42,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        weights=weights)
    _assert_close("phase1 prefill output", ref_out, tpu_out, PREFILL_RMSE)

    # Continue: one token per sequence, from the state phase 1 just wrote.
    conv_after, rec_after = _final_pools(layer)
    ref_out2, tpu_out2, reference2, layer2 = _run_pair(
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        query_lens=[1] * num_seqs,
        has_initial_state=[True] * num_seqs,
        lower_bound=GATE_LOWER_BOUND,
        dtype=torch.bfloat16,
        device=torch.device("tpu"),
        seed=43,
        conv_state=conv_after,
        recurrent_state=rec_after,
        weights=weights,
        reference=reference)
    # The reference continues from its own phase-1 state, the layer from its own,
    # so this compares the whole two-step trajectory, not just one step.
    _assert_close("phase2 decode output", ref_out2, tpu_out2, CHAINED_RMSE)

    _, rec_after2 = _final_pools(layer2)
    slots = _touched(list(range(1, num_seqs + 1)))
    _assert_close("phase2 recurrent_state", reference2.recurrent_state[slots],
                  rec_after2[slots], CHAINED_RMSE)


# ---------------------------------------------------------------------------
# Mixed batches, and the three paths against each other
# ---------------------------------------------------------------------------
@requires_tpu
@pytest.mark.parametrize("path", [PATH_AUTO, PATH_NAIVE])
@pytest.mark.parametrize("query_lens,has_initial_state", [
    ([32, 1], [False, True]),
    ([1, 32], [True, False]),
    ([1, 48, 1, 17], [True, False, True, False]),
    ([96, 1, 1], [True, True, True]),
])
def test_mixed_batch_matches_naive_cpu_reference(query_lens, has_initial_state,
                                                 path):
    """Mixed prefill+decode against the CPU reference, on both paths.

    Neither path needs the batch split: the chunked kernel treats a 1-token row
    as a length-1 segment, and the naive scan reads each token's sequence from
    ``query_start_loc``. These cases cover both orderings (prefill-first and
    decode-first, since nothing in the layer assumes one), several decode rows
    among prefills, and prefill rows that do and do not carry state in.

    Getting the sequence attribution wrong is exactly what this catches: a decode
    row credited to the wrong ``query_start_loc`` entry would read the wrong slot
    and the per-slot state comparison below would fail.
    """
    num_seqs = len(query_lens)
    state_indices = list(range(1, num_seqs + 1))

    ref_out, tpu_out, reference, layer = _run_pair(
        num_heads=2,
        head_dim=64,
        hidden_size=128,
        query_lens=query_lens,
        has_initial_state=has_initial_state,
        lower_bound=GATE_LOWER_BOUND,
        dtype=torch.bfloat16,
        device=torch.device("tpu"),
        path=path,
        seed=80 + num_seqs)

    tag = f"mixed{query_lens} {path}"
    _assert_close(f"{tag} output", ref_out, tpu_out, PREFILL_RMSE)

    conv_after, rec_after = _final_pools(layer)
    slots = _touched(state_indices)
    _assert_close(f"{tag} conv_state", reference.conv_state[slots],
                  conv_after[slots], STATE_RMSE)
    _assert_close(f"{tag} recurrent_state", reference.recurrent_state[slots],
                  rec_after[slots], STATE_RMSE)


@requires_tpu
@pytest.mark.parametrize("num_seqs", [1, 4])
def test_decode_batch_through_the_chunked_kernel_matches_reference(num_seqs):
    """A pure-decode batch forced onto the chunked kernel.

    Not how production dispatches -- the decode kernel is faster -- but it is the
    property that lets mixed batches take the chunked path: the chunked kernel
    must be correct on a length-1 segment, where every token is simultaneously
    the first and last of its chunk.
    """
    state_indices = list(range(1, num_seqs + 1))
    ref_out, tpu_out, reference, layer = _run_pair(
        num_heads=2,
        head_dim=64,
        hidden_size=128,
        query_lens=[1] * num_seqs,
        has_initial_state=[True] * num_seqs,
        lower_bound=GATE_LOWER_BOUND,
        dtype=torch.bfloat16,
        device=torch.device("tpu"),
        path=PATH_CHUNK,
        seed=60 + num_seqs)

    tag = f"chunked decode n{num_seqs}"
    _assert_close(f"{tag} output", ref_out, tpu_out, DECODE_RMSE)
    _, rec_after = _final_pools(layer)
    slots = _touched(state_indices)
    _assert_close(f"{tag} recurrent_state", reference.recurrent_state[slots],
                  rec_after[slots], STATE_RMSE)


@requires_tpu
@pytest.mark.parametrize("query_lens,has_initial_state", [
    ([1, 1, 1], [True, True, True]),
    ([48, 1], [True, True]),
])
def test_all_three_recurrences_agree(query_lens, has_initial_state):
    """The three ops must be interchangeable on the same batch.

    Each of them handles beta's activation, the output norm, the output gate and
    the recurrent-pool layout separately, so a convention that drifts in one of
    them -- a missing sigmoid, a transposed slot -- shows up here as a
    disagreement even when every path is individually self-consistent and its own
    state comparison passes.
    """
    common = dict(num_heads=2,
                  head_dim=64,
                  hidden_size=128,
                  query_lens=query_lens,
                  has_initial_state=has_initial_state,
                  lower_bound=GATE_LOWER_BOUND,
                  dtype=torch.bfloat16,
                  device=torch.device("tpu"),
                  seed=100 + len(query_lens))

    outputs = {}
    states = {}
    for path in (PATH_AUTO, PATH_CHUNK, PATH_NAIVE):
        _, tpu_out, _, layer = _run_pair(**common, path=path)
        outputs[path] = tpu_out
        _, states[path] = _final_pools(layer)

    slots = _touched(list(range(1, len(query_lens) + 1)))
    for path in (PATH_CHUNK, PATH_NAIVE):
        _assert_close(f"output {PATH_AUTO} vs {path}", outputs[PATH_AUTO],
                      outputs[path], DECODE_RMSE)
        _assert_close(f"recurrent_state {PATH_AUTO} vs {path}",
                      states[PATH_AUTO][slots], states[path][slots],
                      STATE_RMSE)


@requires_tpu
@pytest.mark.parametrize("path", [PATH_AUTO, PATH_NAIVE])
def test_mixed_batch_row_is_unaffected_by_its_batch_mates(path):
    """A row's result must not depend on which other rows share its batch.

    Both paths thread the whole state pool through one call, so a bug in the
    per-token slot indexing could let one sequence read or clobber another's state
    and still leave every row looking plausible. This runs the *same* prefill row
    -- identical weights, identical hidden states, identical starting pools, same
    slot -- once alone and once with a decode row appended, and requires its
    output and its state slot to come out the same either way.

    Sharing the hidden states explicitly is the point: deriving them from a seed
    would give the two runs different tensors (different token counts) and the
    comparison would be meaningless.

    The bar differs by path. The naive scan is token-by-token, so adding a row
    changes nothing about the surviving row's arithmetic and the two runs come
    out bit-identical. The chunked kernel pads every segment up to a chunk
    boundary, so a second row changes the padded token count and with it the
    tiling and the summation order: the same row lands within 5e-4 rmse-relative
    (max_abs 2e-6), not on the same bits. Both bars stay far below a real
    cross-contamination bug, which moves the result by rmse-relative >1.
    """
    identical = 1e-6 if path == PATH_NAIVE else 5e-3
    num_heads, head_dim, hidden_size = 2, 64, 128
    t_prefill = 32
    weights = _make_weights(num_heads,
                            head_dim,
                            hidden_size,
                            CONV_SIZE,
                            seed=90)
    conv_state, recurrent_state = _make_pools(4,
                                              num_heads,
                                              head_dim,
                                              CONV_SIZE,
                                              seed=91)

    gen = torch.Generator().manual_seed(92)
    x = torch.randn(
        t_prefill + 1, hidden_size, generator=gen, dtype=torch.float32) * 0.1

    common = dict(num_heads=num_heads,
                  head_dim=head_dim,
                  hidden_size=hidden_size,
                  has_initial_state=None,
                  lower_bound=GATE_LOWER_BOUND,
                  dtype=torch.bfloat16,
                  device=torch.device("tpu"),
                  seed=93,
                  path=path,
                  conv_state=conv_state,
                  recurrent_state=recurrent_state,
                  weights=weights)

    # Alone: the prefill row only, on slot 1.
    _, alone_out, _, alone_layer = _run_pair(
        **{
            **common, "query_lens": [t_prefill],
            "has_initial_state": [False],
            "hidden_states": x[:t_prefill]
        })

    # Mixed: same prefill row on slot 1, plus a decode row on slot 2.
    _, mixed_out, _, mixed_layer = _run_pair(
        **{
            **common, "query_lens": [t_prefill, 1],
            "has_initial_state": [False, True],
            "hidden_states": x
        })

    _assert_close("prefill row output alone vs mixed", alone_out,
                  mixed_out[:t_prefill], identical)

    _, rec_alone = _final_pools(alone_layer)
    _, rec_mixed = _final_pools(mixed_layer)
    _assert_close("prefill row state alone vs mixed", rec_alone[1],
                  rec_mixed[1], identical)
    # Slot 3 is in neither batch, and slot 0 is the reserved null block.
    for slot in (0, 3):
        assert torch.equal(rec_mixed[slot], recurrent_state[slot]), (
            f"the mixed batch wrote to slot {slot}, which it was not given")
