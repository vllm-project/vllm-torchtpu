# KDA kernels: chunked vs recurrent-packed

## Provenance

This package holds three implementations of the same recurrence:

| file | what it is | used for |
|---|---|---|
| `chunk_kda.py` | chunked Pallas kernel, `kda_forward` | prefill |
| `decode_kda.py` | fused short-conv + recurrent Pallas kernel, `decode_kda` | decode |
| `naive_kda.py` | plain-XLA recurrence, `naive_ragged_kda` | reference |

`chunk_kda.py` is derived from **tokamax**'s experimental KDA op
(`tokamax/_src/ops/experimental/kda/`, branch `antgroup/kda-pallas-kernel`),
Apache-2.0, and carries that licence header. It has since been edited
substantially and is no longer a mechanical copy: the backward pass,
context-parallel support, the chunked gated-delta-rule implementation (which
duplicated `kernels/gdn/v3`), the unfused intra-chunk variants and the float32
fallback were all removed, so the kernel is scoped to what Kimi-K3 needs --
bfloat16, varlen `segment_ids`, `chunk_size=64`. Annotations were converted from
`jaxtyping` to this repo's `jax.Array` + shape-comment convention. It arrived here
from a former `kernels/kda/` package that split it across six modules
(`pallas_tpu_fwd.py`, `common.py`, `preprocess.py`, `utils.py`,
`pallas_tpu_types.py` and the package `__init__.py`); those are now sections of
the single file, otherwise unchanged.

`decode_kda.py` is not from tokamax: it is a TPU kernel written here, borrowing
the recurrent step and layout helpers from `kernels/gdn/v3`.

Kimi Delta Attention is a *linear* attention: it carries a recurrent state
`S` of shape `[K, V]` per head and updates it once per token with a
per-channel-gated delta rule. Written literally, the update is sequential in `T`:

```
S = S * exp(g_t)                                 # per-channel decay
S = S + beta_t * k_t ⊗ (v_t - k_tᵀ S)            # rank-1 delta-rule correction
o_t = q_t S
```

Every real implementation is one of two strategies for evaluating that
recurrence, and Kimi-K3 uses **both**, on different requests in the same batch.
This document explains the difference and what each implies for TPU.

Reference for the recurrence: `naive_recurrent_kda` in
`tests/kernels/test_kda.py` (a literal transcription, used as the correctness
oracle).

---

## The two strategies

### Chunked — used for prefill

Ported here as `chunk_kda_fwd_custom` (from tokamax); vLLM's CUDA equivalent is
`chunk_kda_with_fused_gate`.

Splits the sequence into fixed chunks (`chunk_size=64` is the only size this
kernel supports) and rewrites the recurrence so that *within* a chunk the
sequential dependency disappears. Tokamax's design doc describes four semantic
stages:

| Stage | Objective |
|---|---|
| 1 | Activate the gate; cumulative decay within each chunk (`g_cumsum`) |
| 2 | Eliminate intra-chunk sequential dependencies → `w`, `u`, `Aqk`, `Akk` |
| 3 | Propagate state along the chunk dimension |
| 4 | Combine the history read with current-chunk contributions → `o` |

Stage 2 is the interesting one: the intra-chunk delta-rule couplings become
relation matrices (`Aqk`, `Akk`) and a unit-lower-triangular solve, so the bulk
of the work turns into **matmuls**. Only the chunk-to-chunk hand-off (stage 3)
stays sequential, and there are `T / 64` of those instead of `T`.

The stages are dependency structure, not dispatches: the production path fuses
1+2 and 3+4, giving two kernel launches, because fusing cuts HBM round trips.

**Cost shape.** Compute-bound and matmul-heavy. State I/O is amortised over 64
tokens. This is what you want when `T` is large.

### Recurrent-packed — used for decode

Implemented by `decode_kda.py` for plain decode. It fuses the short causal
convolution, SiLU, and one recurrent KDA step. Speculative multi-token decode is
not implemented yet.

Evaluates the recurrence directly — no chunking, no triangular solve, because
with one token there is nothing to decouple. Parallelism comes entirely from
running many *sequences* at once: the Triton grid is
`(value_blocks, num_sequences × heads)`, and each program loads one head's
`[K, V]` state, applies one rank-1 update, writes the state back, and emits one
output token.

**Cost shape.** Bandwidth- and latency-bound. Each token moves a whole `K × V`
state per head with only a rank-1 update of arithmetic on it, so the state size
dominates and arithmetic intensity is very low. Chunking would not help: there
is only one token.

---

## Differences that matter for the port

| | chunked (prefill) | recurrent-packed (decode) |
|---|---|---|
| tokens per sequence per call | many (`T ≫ 1`) | 1 (plain) / `1 + num_spec` (spec) |
| intra-sequence parallelism | yes, within a chunk | none |
| dominant op | matmul + triangular solve | rank-1 update |
| bound by | compute | memory bandwidth |
| sequential steps | `T / 64` | 1 per call |
| state in/out | dense `[B, N, H, K, V]` at segment boundaries | in-place, indirected through a pool |
| q/k/v layout | separate tensors | compact `[qkv, head, channel]` tile |
| varlen | `segment_ids` → `cu_seqlens`, segments padded to 64 | one slot per sequence; no padding concept |

Two of these are easy to underestimate.

**State addressing.** The chunked kernel takes and returns states as a dense
array indexed by batch and segment. The decode kernel instead indirects through
a **state pool**:

```python
state_idx = tl.load(state_indices + i_n * stride_state_indices)
if state_idx <= 0:                       # null/sentinel slot
    store zeros; return
p_state = state + state_idx * stride_state_token
```

Each sequence's state lives at an arbitrary pool slot, updated **in place**, and
slot `0` is a null sentinel that yields zeros. That is the same convention the
plugin's Qwen GDN path already relies on (`mamba_state_indices` with slot 0
reserved as the null block — see `_build_mamba_state_indices` in the TPU runner).
A TPU decode kernel has to honour it, not just the maths.

**Packed inputs.** Decode reads a fused `mixed_qkv` tensor where "Q, K and V
occupy consecutive channel ranges, while the token stride may also include the
output-gate projection". Splitting it up front would cost an extra copy per
token on the hot path, so the kernel takes strides instead.

Both kernels activate the gate **in-kernel** from `raw_g` with `A_log`,
`dt_bias` and `lower_bound`, so that part of the contract is shared (see
`_activate_gate_torch` in the tests for the exact formula).

---

## Which path a request takes

`KimiK3DeltaAttention.forward` selects a kernel from the batch composition:

| condition | kernel |
|---|---|
| `num_prefills > 0` | chunked (`chunk_kda_with_fused_gate`) |
| `num_prefills == 0`, single-query | recurrent packed decode |
| `num_spec_decodes > 0` | recurrent multi-query (`fused_recurrent_kda`) |

There is also a CUDA-only `flashkda` prefill backend, which
`resolve_kda_prefill_backend` refuses unless the device is SM90/SM10x/SM12x
(plus bfloat16, `head_dim=128`, bounded gate), falling back to the chunked path
elsewhere — so it is self-excluding on TPU.

## Status on TPU

Ported and validated against the naive reference:

- Chunked prefill with varlen segment IDs and per-segment state.
- Single-token decode with fused short convolution and recurrent KDA, including
  indirect convolution/recurrent state-pool access.

Speculative multi-token decode is not ported. Mixed prefill/decode batches use
the naive fallback when explicitly enabled; the optimized paths do not yet
split and recombine mixed batches.
