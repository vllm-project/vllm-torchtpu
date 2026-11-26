# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""State-storage selection and verifier rounding at decode token boundaries."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.models.kimi_k3.megakernel import backend
from vllm_torchtpu.models.kimi_k3.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import program
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import prefill_megakernel


@pytest.mark.parametrize(
    "setting,expected",
    [("auto", "float32"), ("float32", "float32"), ("bfloat16", "bfloat16")],
)
def test_engine_config_passes_state_dtype_to_decoder(monkeypatch, setting, expected):
    from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import decode, decode_ops

    monkeypatch.setattr(backend, "role", lambda _: "decode")
    monkeypatch.setattr(backend, "attention_dp", lambda _: 2)
    monkeypatch.setattr(backend, "synthetic_acceptance_lengths", lambda _: None)
    config = MegakernelEngineConfig.from_vllm_config(
        SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=32,
                decode_context_parallel_size=16,
                cp_kv_cache_interleave_size=256,
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=128, max_num_batched_tokens=128, async_scheduling=True
            ),
            model_config=SimpleNamespace(max_model_len=139264),
            cache_config=SimpleNamespace(
                block_size=256, cache_dtype="auto", mamba_ssm_cache_dtype=setting
            ),
            speculative_config=None,
        )
    )
    monkeypatch.setattr(
        decode_ops, "DecodeOps", lambda path, **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(decode, "KimiDecoder", lambda config, ops, draft, **kw: ops)
    ops = program.build(config, Path("unused"), tp_rank=0)
    assert ops.mamba_ssm_cache_dtype == setting
    assert program.recurrent_state_dtype(ops.mamba_ssm_cache_dtype) == expected


@pytest.mark.parametrize("setting", ["float16", "fp8", "invalid"])
def test_unsupported_state_dtype_rejected_before_allocation(setting):
    from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.decode_ops import DecodeOps

    with pytest.raises(ValueError, match="mamba-ssm-cache-dtype"):
        DecodeOps(Path("unused"), num_slots=2, mamba_ssm_cache_dtype=setting)


def kda_recurrence_call(projected, history, state, sequence_rows, *, interpret=True):
    """Actual KDA function with fixed projections, excluding cross-rank work."""
    from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
        decode_megakernel as kernel,
    )

    batch = projected.shape[0]

    def body(
        projected_ref,
        history_ref,
        state_ref,
        feature_ref,
        next_history,
        next_state,
        output,
    ):
        # No projection DMA: these small constant weights supply the gate and
        # normalization; the per-token input projections are supplied directly.
        no_copy = SimpleNamespace(wait=lambda: None)
        weights = SimpleNamespace(
            gate_projection=jnp.zeros((128, 128), jnp.bfloat16),
            input_projection=jnp.zeros((128, 1664), jnp.bfloat16),
            feature_gate_projection_vmem=feature_ref,
            exp_log_decay_rate=jnp.zeros((3, 1), jnp.float32),
            time_step_bias=jnp.full((3, 128), -3.0, jnp.float32),
            convolution_filter=jnp.full((4, 3, 3, 128), 0.25, jnp.bfloat16),
            output_normalization=jnp.ones((128,), jnp.bfloat16),
            make_projection_copies=lambda: SimpleNamespace(
                gate=no_copy, feature_gate=no_copy
            ),
        )
        kernel.kda(
            jnp.zeros((batch, 128), jnp.bfloat16),
            jnp.zeros((batch, 128), jnp.bfloat16),
            next_history,
            next_state,
            weights=weights,
            convolution_history_ref=history_ref,
            recurrent_state_ref=state_ref,
            communication_workspace=SimpleNamespace(
                reduction_phase=jnp.array([0], jnp.int32)
            ),
            synchronize_devices=False,
            weights_are_prefetched=True,
            sequence_rows=sequence_rows,
            shard_index=0,
            first_call=False,
            attended_out=output,
            gate_features_vmem=jnp.zeros((batch, 128), jnp.bfloat16),
            projected_rows=projected_ref[...],
        )

    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct(history.shape, history.dtype),
            jax.ShapeDtypeStruct(state.shape, state.dtype),
            jax.ShapeDtypeStruct((batch, 384), jnp.bfloat16),
        ),
        interpret=interpret,
    )(projected, history, state, jnp.zeros((128, 384), jnp.bfloat16))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("width", [4, 8])
def test_verifier_state_matches_sequential_decode(dtype, width):
    rng = np.random.default_rng(413)
    requests = 2
    projected = jnp.asarray(rng.normal(size=(requests, width, 1664)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(requests, 3, 3, 4, 128)), jnp.bfloat16)
    state = jnp.asarray(rng.normal(size=(requests, 3, 128, 128)), dtype)
    batched = kda_recurrence_call(
        projected.reshape(-1, 1664),
        jnp.repeat(history, width, axis=0),
        jnp.repeat(state, width, axis=0),
        width,
    )
    for offset in range(width):
        history, state, attended = kda_recurrence_call(
            projected[:, offset], history, state, 1
        )
        # Only the three real conv heads are written (the fourth is padding).
        np.testing.assert_array_equal(
            batched[0][offset::width, :, :, :3], history[:, :, :, :3]
        )
        np.testing.assert_array_equal(batched[1][offset::width], state)
        np.testing.assert_array_equal(batched[2][offset::width], attended)


def test_prefill_rejects_bf16_instead_of_ignoring_the_flag():
    config = MegakernelEngineConfig(
        role="prefill",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=1,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        mamba_ssm_cache_dtype="bfloat16",
    )
    with pytest.raises(NotImplementedError, match="decode-only"):
        program.check(config)


def _kda_token_step_oracle_f32(
    hidden_f32,
    gate_w_f32,
    fb_w_f32,
    proj_w_f32,
    conv_w_f32,
    alog_f32,
    dt_f32,
    norm_f32,
    rec0_f32,
    hist0_f32,
    live_len: int,
):
    seq_len = hidden_f32.shape[0]
    heads, ch = 3, 128
    width = heads * ch
    gate_feat = np.asarray(
        jnp.asarray(hidden_f32 @ gate_w_f32, jnp.bfloat16), np.float32
    )
    raw_gate = np.asarray(
        jnp.asarray(gate_feat @ fb_w_f32, jnp.bfloat16), np.float32
    ).reshape(seq_len, heads, ch)
    proj = np.asarray(jnp.asarray(hidden_f32 @ proj_w_f32, jnp.bfloat16), np.float32)
    qkv = proj[:, : 3 * width].reshape(seq_len, 3, heads, ch)
    out_gate_sig = 1.0 / (1.0 + np.exp(-proj[:, 3 * width : 4 * width]))
    beta_logits = proj[:, 4 * width : 4 * width + heads]

    window = np.concatenate([hist0_f32, qkv], axis=0)
    conv = sum(window[tap : tap + seq_len] * conv_w_f32[tap] for tap in range(4))
    filtered = conv / (1.0 + np.exp(-conv))
    a_log = np.exp(alog_f32)[None, :, :]
    decay = -5.0 / (1.0 + np.exp(-a_log * (raw_gate + dt_f32[None])))
    beta = 1.0 / (1.0 + np.exp(-beta_logits))

    q_raw, k_raw, v_raw = filtered[:, 0], filtered[:, 1], filtered[:, 2]
    q_norm = (
        q_raw
        / np.sqrt(np.sum(q_raw * q_raw, axis=-1, keepdims=True) + 1e-6)
        / np.sqrt(ch)
    )
    k_norm = k_raw / np.sqrt(np.sum(k_raw * k_raw, axis=-1, keepdims=True) + 1e-6)

    state = rec0_f32.copy()
    out = np.zeros((seq_len, width), dtype=np.float32)
    for tok in range(live_len):
        state = state * np.exp(decay[tok])[:, :, None]
        pred = np.einsum("hkv,hk->hv", state, k_norm[tok])
        delta = (v_raw[tok] - pred) * beta[tok][:, None]
        state = state + k_norm[tok][:, :, None] * delta[:, None, :]
        y_tok = np.einsum("hkv,hk->hv", state, q_norm[tok])
        rms = 1.0 / np.sqrt(np.mean(y_tok * y_tok, axis=-1, keepdims=True) + 1e-5)
        out[tok] = (y_tok * rms * norm_f32[0][None, :]).reshape(width) * out_gate_sig[
            tok
        ]
    return out, state, window[live_len : live_len + 3]


@pytest.mark.parametrize(
    ("seq_len", "tile", "live_len"),
    [
        (512, 128, 350),
        (256, 128, 0),
        (192, 64, 150),
    ],
)
def test_should_match_token_recurrence_when_kda_tiled_uses_grouped_supertiles(
    seq_len: int, tile: int, live_len: int
):
    rng = np.random.default_rng(20261112 + seq_len + live_len)
    hidden_dim = 256
    hidden_np = rng.standard_normal((seq_len, hidden_dim)).astype(np.float32) * 0.1
    if live_len < seq_len:
        hidden_np[live_len:] = 7.5
    hidden = jnp.asarray(hidden_np, jnp.bfloat16)
    gate_w = jnp.asarray(
        rng.standard_normal((hidden_dim, 128)) * 0.1, dtype=jnp.bfloat16
    )
    fb_w = jnp.asarray(rng.standard_normal((128, 384)) * 0.1, dtype=jnp.bfloat16)
    proj_w = jnp.asarray(
        rng.standard_normal((hidden_dim, 1664)) * 0.1, dtype=jnp.bfloat16
    )
    conv_w = jnp.asarray(rng.standard_normal((4, 3, 3, 128)) * 0.2, dtype=jnp.bfloat16)
    alog = jnp.asarray(rng.uniform(-1.5, -0.5, size=(3, 1)), dtype=jnp.float32)
    dt = jnp.asarray(rng.uniform(-2.0, -0.5, size=(3, 128)), dtype=jnp.float32)
    norm = jnp.asarray(rng.uniform(0.8, 1.2, size=(8, 128)), dtype=jnp.bfloat16)
    rec0 = jnp.asarray(rng.standard_normal((3, 128, 128)) * 0.05, dtype=jnp.float32)
    hist0 = jnp.asarray(rng.standard_normal((3, 3, 3, 128)) * 0.1, dtype=jnp.bfloat16)
    length = jnp.array([live_len], dtype=jnp.int32)

    fn = prefill_megakernel.make_kda_tiled(seq_len=seq_len, tile=tile, interpret=True)
    att, rec_out, hist_out = fn(
        hidden, gate_w, fb_w, proj_w, conv_w, alog, dt, norm, rec0, hist0, length
    )

    ref_att, ref_rec, ref_hist = _kda_token_step_oracle_f32(
        np.asarray(hidden, np.float32),
        np.asarray(gate_w, np.float32),
        np.asarray(fb_w, np.float32),
        np.asarray(proj_w, np.float32),
        np.asarray(conv_w, np.float32),
        np.asarray(alog, np.float32),
        np.asarray(dt, np.float32),
        np.asarray(norm, np.float32),
        np.asarray(rec0, np.float32),
        np.asarray(hist0, np.float32),
        live_len,
    )
    att_np = np.asarray(att, np.float32)
    rec_np = np.asarray(rec_out, np.float32)
    hist_np = np.asarray(hist_out, np.float32)

    if live_len < seq_len:
        assert np.max(np.abs(att_np[live_len:])) == 0.0
    if live_len == 0:
        np.testing.assert_array_equal(rec_np, np.asarray(rec0, np.float32))
        np.testing.assert_array_equal(hist_np, np.asarray(hist0, np.float32))
    else:
        rel_att = np.linalg.norm(att_np - ref_att) / (np.linalg.norm(ref_att) + 1e-8)
        rel_rec = np.linalg.norm(rec_np - ref_rec) / (np.linalg.norm(ref_rec) + 1e-8)
        assert rel_att < 0.02
        assert rel_rec < 0.02
        assert np.max(np.abs(hist_np - ref_hist)) < 1e-2
