# SPDX-License-Identifier: Apache-2.0
"""q38 G1LINE: the BATCHED decode tile and the G1 D1 body choice as shared pure functions.

The G1 D1 body (VLLM_TPU_GDN_G1_BODY, `compute_g1`) is chosen at trace time inside
`wrapper.fused_conv1d_gdn`, per compiled bucket: it needs the factored LOG window and a
BATCHED tile of exactly 8 sequences, and the tile follows the bucket's rows and the
static request capacity (`state_indices.size`). A trace-time line is silent when the
compile cache serves the bucket, so the engagement line (`runner/g1_engagement.py`)
evaluates THESE functions at warmup instead; the trace path calls the same functions,
and records what it chose (`record_trace`) so a cold or partly cold start can check
the prediction against the trace (the MISMATCH line).

Nothing here enters a program: `tile_sizes` and `g1_reason` are the wrapper's own
arithmetic, moved here unchanged, and the trace record is a process-level dict.
"""

import dataclasses
import inspect
from typing import Any

import jax.numpy as jnp
from jax.experimental import pallas as pl

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import config, tiling

# The BATCHED tile the G1 D1 body is written for (compute_g1.G1Geom asserts it).
G1_SEQ_TILE = 8


def tile_sizes(
    *,
    batch_size: int,
    num_seqs: int,
    padded_batch_size: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    dim: int,
    conv_state_dim_size: int,
    act_out_dtype: Any,
    recurrent_state_dtype: Any,
    num_lanes: int,
    num_spec_tokens: int,
    is_kda: bool,
    qkv_2d: bool,
    decode_tile_size: int,
    mixed_tile_size: int,
) -> tuple[int, int, bool]:
    """(decode_tile_size, mixed_tile_size, dynamic) exactly as `fused_conv1d_gdn` sizes
    its tiles: TPU_ENABLE_GDN_DYNAMIC_TILING's `tiling.get_tile_sizes` (GDN only), else
    the static VMEM budget; KDA verify keeps one sequence per tile. `dynamic` says which
    branch ran (the wrapper then applies the 2D-rows fallback and logs)."""
    decode_tile_size = min(decode_tile_size, batch_size)
    mixed_tile_size = min(mixed_tile_size, batch_size)
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes
    window_size = num_spec_tokens + 1
    decode_vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(window_size)

    # KDA stays on static tiles until the VMEM estimate models its gate.
    dynamic = bool(envs.TPU_ENABLE_GDN_DYNAMIC_TILING and not is_kda)
    if dynamic:
        decode_tile_size, mixed_tile_size = tiling.get_tile_sizes(
            batch_size=batch_size,
            num_seqs=num_seqs,
            padded_batch_size=padded_batch_size,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_state_dim_size=conv_state_dim_size,
            act_out_dtype=act_out_dtype,
            recurrent_state_dtype=recurrent_state_dtype,
            num_lanes=num_lanes,
            decode_vmem_limit_bytes=decode_vmem_limit_bytes,
            mixed_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
            window_size=window_size,
            qkv_2d=qkv_2d,
        )
    else:
        # Each sequence holds 1 input state slot plus `window` output
        # checkpoint slots per buffer in VMEM. Shrink the tile so the
        # double-buffered windows fit in roughly half the scoped-VMEM budget.
        window = num_spec_tokens + 1
        num_buffers = config.GDNConfig.__dataclass_fields__["num_buffers"].default
        state_positions = 1 + window
        recurrent_bytes_per_seq = state_positions * n_v * d_k * d_v * 4
        bytes_per_seq = (
            # Recurrent checkpoints (fp32) — the dominant term.
            recurrent_bytes_per_seq
            # Conv checkpoints (fp32).
            + state_positions * (kernel_size - 1) * dim * 4
            # qkv (fp32), b/a (fp32), out (act_out).
            + window * (dim * 4 + 2 * aligned_num_v_heads * 4 + n_v * d_v * 2)
        )
        spec_tile_budget = (decode_vmem_limit_bytes // 2) // num_buffers
        decode_tile_size = max(
            1, min(decode_tile_size, spec_tile_budget // bytes_per_seq)
        )

    if num_spec_tokens > 0 and is_kda:
        # KDA verify keeps one sequence per tile until multi-sequence KDA
        # verify tiles are sized by a KDA-aware VMEM estimate.
        decode_tile_size = 1
    return decode_tile_size, mixed_tile_size, dynamic


def g1_reason(cfg: config.GDNConfig) -> str | None:
    """None when the G1 D1 body serves this kernel (the key aside), else why not.

    The wrapper's predicate was `uses_factored_window and seq_tile_size == 8 and
    not is_kda` (behind VLLM_TPU_GDN_G1_BODY); the order here only picks the
    reason that is reported."""
    if cfg.is_kda:
        return "kda"
    if not cfg.uses_factored_window:
        return "factored_window_off"
    if cfg.seq_tile_size != G1_SEQ_TILE:
        return f"seq_tile_size={cfg.seq_tile_size}"
    return None


def wrapper_tile_defaults() -> tuple[int, int]:
    """`fused_conv1d_gdn`'s own decode / mixed tile defaults (the pooled op passes none)."""
    from vllm_torchtpu.kernels.gdn.v3 import wrapper

    params = inspect.signature(wrapper.fused_conv1d_gdn).parameters
    return params["decode_tile_size"].default, params["mixed_tile_size"].default


@dataclasses.dataclass(frozen=True)
class Prediction:
    """What the BATCHED kernel of one bucket is: its tile and name, and the reason
    the G1 D1 body is not used (None: it is)."""

    rows: int
    num_seqs: int
    window: int
    seq_tile_size: int
    kernel: str
    reason: str | None
    n_kq: int
    n_v: int
    d_k: int
    d_v: int
    dim: int

    @property
    def engaged(self) -> bool:
        return self.reason is None


def predict(
    *,
    rows: int,
    num_seqs: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    dim: int,
    num_spec_tokens: int,
    act_dtype: Any,
    window_log: bool,
    factored_window: bool,
    qkv_2d: bool,
    num_lanes: int,
) -> Prediction:
    """The pooled BATCHED kernel `fused_conv1d_gdn` builds for `rows` tokens over
    `num_seqs` request slots, from the same functions the trace path calls.

    Pooled calls (the unified block pool) keep f32 recurrent and conv state, so the
    conv state width is `dim`; f32 rows pack 1 per row, 2D bf16 rows (`qkv_2d`) 16.
    `act_dtype` is the projection dtype (the kernel's act_out)."""
    act_in_dtype = jnp.dtype(jnp.float32)
    packing = config.QKV_2D_ROW_ALIGN if qkv_2d else 4 // act_in_dtype.itemsize
    padded_batch_size = pl.cdiv(rows, packing) * packing
    decode_default, mixed_default = wrapper_tile_defaults()
    decode_tile, _, _ = tile_sizes(
        batch_size=rows,
        num_seqs=num_seqs,
        padded_batch_size=padded_batch_size,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        dim=dim,
        conv_state_dim_size=dim,
        act_out_dtype=jnp.dtype(act_dtype),
        recurrent_state_dtype=jnp.float32,
        num_lanes=num_lanes,
        num_spec_tokens=num_spec_tokens,
        is_kda=False,
        qkv_2d=qkv_2d,
        decode_tile_size=decode_default,
        mixed_tile_size=mixed_default,
    )
    window = num_spec_tokens + 1
    cfg = config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        batch_size=padded_batch_size,
        kernel_size=kernel_size,
        tile_size=decode_tile,
        window_size=window,
        dim_size=dim,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        attention_mode=config.AttentionMode.GDN,
        dtypes=config.Dtypes(
            act_in=act_in_dtype,
            act_out=jnp.dtype(act_dtype),
            compute=jnp.float32.dtype,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        checkpoint_mode=(
            config.CheckpointMode.LOG if window_log else config.CheckpointMode.SELECT
        ),
        log_positions=num_spec_tokens if window_log else 0,
        factored_window=factored_window,
    )
    reason = g1_reason(cfg)
    if reason is None:
        cfg = dataclasses.replace(cfg, g1_body=True)
    return Prediction(
        rows=rows,
        num_seqs=num_seqs,
        window=window,
        seq_tile_size=cfg.seq_tile_size,
        kernel=cfg.get_kernel_name(),
        reason=reason,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        dim=dim,
    )


# Trace-time record of the BATCHED kernels this process traced with the key on:
# (rows, num_seqs, window) -> what the trace chose. Read by the engagement line's
# MISMATCH check; empty when every bucket came from the compile cache.
_TRACED: dict[tuple[int, int, int], dict[str, Any]] = {}


def record_trace(
    *,
    rows: int,
    num_seqs: int,
    window: int,
    seq_tile_size: int,
    kernel: str,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    dim: int,
) -> None:
    _TRACED[(rows, num_seqs, window)] = dict(
        seq_tile_size=seq_tile_size,
        kernel=kernel,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        dim=dim,
    )


def traced(rows: int, num_seqs: int, window: int) -> dict[str, Any] | None:
    return _TRACED.get((rows, num_seqs, window))


def as_record(p: Prediction) -> dict[str, Any]:
    """A prediction in `record_trace`'s form, for the comparison."""
    return dict(
        seq_tile_size=p.seq_tile_size,
        kernel=p.kernel,
        n_kq=p.n_kq,
        n_v=p.n_v,
        d_k=p.d_k,
        d_v=p.d_v,
        dim=p.dim,
    )
