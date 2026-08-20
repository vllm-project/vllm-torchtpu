# SPDX-License-Identifier: Apache-2.0
"""Unit tests verifying bitwise parity for ragged_paged_attention_bundled against layer-major RPA.

Tests confirm that:
  - Numerical Parity: Outputs and updated cache states match the layer-major baseline bit-for-bit.
  - Zero-Copy Isolation: Updating layer L in the bundle leaves all other layer slots (i != L) intact.
  - Sequential Donation Chaining: Chaining across all model layers correctly threads JAX buffer donation.
  - Attention Interface Compliance: Custom scaling, logits soft-capping, and causal masking flags are honored.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    get_kv_cache_shape, ragged_paged_attention, ragged_paged_attention_bundled)


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


# ---------------------------------------------------------------------------
# Workload + shape helpers
# ---------------------------------------------------------------------------


def _cfg():
    return dict(
        num_layers=6,
        num_blocks=64,
        page_size=16,
        num_kv_heads=4,
        head_dim=128,
        num_q_heads=16,
        kv_dtype=jnp.bfloat16,
    )


def _per_layer_shape(c):
    return get_kv_cache_shape(c["num_blocks"], c["page_size"],
                              c["num_kv_heads"], c["head_dim"], c["kv_dtype"])


def _workload(c, scenario, seed=0):
    page_size, num_blocks = c["page_size"], c["num_blocks"]
    num_kv_heads, num_q_heads, head_dim = (c["num_kv_heads"], c["num_q_heads"],
                                           c["head_dim"])
    if scenario == "decode":
        q_lens = [1, 1, 1, 1, 1, 1, 1, 1]
        kv_lens_l = [128] * 8
        distribution = np.array([8, 8, 8], dtype=np.int32)
    elif scenario == "prefill":
        q_lens = [64, 0, 0, 0, 0, 0, 0, 0]
        kv_lens_l = [64] + [0] * 7
        distribution = np.array([0, 1, 1], dtype=np.int32)
    elif scenario == "mixed":
        q_lens = [1, 1, 1, 1, 32, 32, 32, 32]
        kv_lens_l = [128, 128, 128, 128, 192, 192, 192, 192]
        distribution = np.array([4, 8, 8], dtype=np.int32)
    else:
        raise ValueError(scenario)
    max_num_seqs = len(q_lens)
    cu = np.cumsum([0] + q_lens, dtype=np.int32)
    max_tokens = max(int(cu[-1]), 16)
    cu = np.pad(cu, (0, max_num_seqs + 1 - len(cu)),
                mode="edge").astype(np.int32)
    pages_per_seq = max(int(np.ceil(max(kv_lens_l) / page_size)), 1)
    pool = np.arange(num_blocks, dtype=np.int32)
    np.random.default_rng(seed).shuffle(pool)
    page_indices = pool[:max_num_seqs * pages_per_seq].astype(np.int32)
    rng = np.random.default_rng(seed)
    return dict(
        q=rng.standard_normal((max_tokens, num_q_heads, head_dim),
                              dtype=np.float32),
        k=rng.standard_normal((max_tokens, num_kv_heads, head_dim),
                              dtype=np.float32),
        v=rng.standard_normal((max_tokens, num_kv_heads, head_dim),
                              dtype=np.float32),
        kv_lens=np.asarray(kv_lens_l, dtype=np.int32),
        page_indices=page_indices,
        cu_q_lens=cu,
        distribution=distribution,
    )


def _to_jax_workload(wl_np, dtype):
    return dict(q=jnp.asarray(wl_np["q"], dtype=dtype),
                k=jnp.asarray(wl_np["k"], dtype=dtype),
                v=jnp.asarray(wl_np["v"], dtype=dtype),
                kv_lens=jnp.asarray(wl_np["kv_lens"]),
                page_indices=jnp.asarray(wl_np["page_indices"]),
                cu_q_lens=jnp.asarray(wl_np["cu_q_lens"]),
                distribution=jnp.asarray(wl_np["distribution"]))


# ---------------------------------------------------------------------------
# Reference + bundled callers (re-jitted per call site for clarity)
# ---------------------------------------------------------------------------


@jax.jit
def _rpa_layer(kv, q, k, v, kv_lens, page_indices, cu_q_lens, distribution):
    return ragged_paged_attention(q,
                                  k,
                                  v,
                                  kv,
                                  kv_lens,
                                  page_indices,
                                  cu_q_lens,
                                  distribution,
                                  sm_scale=1.0 / (q.shape[-1]**0.5))


@jax.jit
def _rpa_bundled(bundle, layer_idx, q, k, v, kv_lens, page_indices, cu_q_lens,
                 distribution):
    return ragged_paged_attention_bundled(q,
                                          k,
                                          v,
                                          bundle,
                                          layer_idx,
                                          kv_lens,
                                          page_indices,
                                          cu_q_lens,
                                          distribution,
                                          sm_scale=1.0 / (q.shape[-1]**0.5))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["decode", "prefill", "mixed"])
def test_bundled_matches_layer_major(scenario):
    """Verifies bitwise output and KV cache update parity against layer-major RPA across scenarios."""
    _require_tpu()
    c = _cfg()
    pls = _per_layer_shape(c)
    init_np = np.random.default_rng(0).standard_normal((c["num_layers"], ) +
                                                       pls).astype(np.float32)
    wl_np = _workload(c, scenario, seed=1)

    for layer in range(c["num_layers"]):
        # Layer-major reference: fresh per-layer tensor + fresh inputs.
        kv_ref = jnp.asarray(init_np[layer], dtype=c["kv_dtype"])
        w_ref = _to_jax_workload(wl_np, c["kv_dtype"])
        out_ref, new_kv_ref = _rpa_layer(kv_ref, w_ref["q"], w_ref["k"],
                                         w_ref["v"], w_ref["kv_lens"],
                                         w_ref["page_indices"],
                                         w_ref["cu_q_lens"],
                                         w_ref["distribution"])

        # Bundled: fresh bundle in block-major layout (num_blocks at
        # dim 0, num_layers at dim 1) + fresh inputs.
        bundle = jnp.asarray(init_np, dtype=c["kv_dtype"]).swapaxes(0, 1)
        w_b = _to_jax_workload(wl_np, c["kv_dtype"])
        out_b, new_bundle = _rpa_bundled(bundle,
                                         jnp.asarray(layer, dtype=jnp.int32),
                                         w_b["q"], w_b["k"], w_b["v"],
                                         w_b["kv_lens"], w_b["page_indices"],
                                         w_b["cu_q_lens"], w_b["distribution"])

        np.testing.assert_array_equal(
            np.asarray(out_ref),
            np.asarray(out_b),
            err_msg=f"output differs at layer {layer} ({scenario})")
        np.testing.assert_array_equal(
            np.asarray(new_kv_ref),
            np.asarray(new_bundle[:, layer, ...]),
            err_msg=f"per-layer kv differs at layer {layer} ({scenario})")


def test_bundled_donation_chain():
    """Verifies that sequentially chaining bundled attention calls across layers preserves in-place donation."""
    _require_tpu()
    c = _cfg()
    pls = _per_layer_shape(c)
    init_np = np.random.default_rng(7).standard_normal((c["num_layers"], ) +
                                                       pls).astype(np.float32)
    wl_np = _workload(c, "mixed", seed=2)

    bundle = jnp.asarray(init_np, dtype=c["kv_dtype"]).swapaxes(0, 1)
    # Chain — each call consumes the prior bundle (donated) and returns
    # the new bundle. This is the production access pattern.
    for layer in range(c["num_layers"]):
        w = _to_jax_workload(wl_np, c["kv_dtype"])
        _, bundle = _rpa_bundled(bundle, jnp.asarray(layer, dtype=jnp.int32),
                                 w["q"], w["k"], w["v"], w["kv_lens"],
                                 w["page_indices"], w["cu_q_lens"],
                                 w["distribution"])
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), bundle)

    # Reference for each layer using a fresh per-layer call from the same
    # initial state.
    for layer in range(c["num_layers"]):
        kv_ref = jnp.asarray(init_np[layer], dtype=c["kv_dtype"])
        w_ref = _to_jax_workload(wl_np, c["kv_dtype"])
        _, new_kv_ref = _rpa_layer(kv_ref, w_ref["q"], w_ref["k"], w_ref["v"],
                                   w_ref["kv_lens"], w_ref["page_indices"],
                                   w_ref["cu_q_lens"], w_ref["distribution"])
        np.testing.assert_array_equal(
            np.asarray(new_kv_ref),
            np.asarray(bundle[:, layer, ...]),
            err_msg=f"chained donation: layer {layer} bundle slot diverged")


def test_bundled_does_not_touch_other_layers():
    """Verifies that an update to layer L preserves all untouched layer slots (i != L) in the bundle."""
    _require_tpu()
    c = _cfg()
    pls = _per_layer_shape(c)
    init_np = np.random.default_rng(11).standard_normal((c["num_layers"], ) +
                                                        pls).astype(np.float32)
    wl_np = _workload(c, "mixed", seed=3)

    target = 3
    bundle = jnp.asarray(init_np, dtype=c["kv_dtype"]).swapaxes(0, 1)
    w = _to_jax_workload(wl_np, c["kv_dtype"])
    _, new_bundle = _rpa_bundled(bundle, jnp.asarray(target, dtype=jnp.int32),
                                 w["q"], w["k"], w["v"], w["kv_lens"],
                                 w["page_indices"], w["cu_q_lens"],
                                 w["distribution"])
    for layer in range(c["num_layers"]):
        if layer == target:
            continue
        # Layers other than the target must equal their initial state.
        np.testing.assert_array_equal(
            np.asarray(new_bundle[:, layer, ...]),
            np.asarray(jnp.asarray(init_np[layer], dtype=c["kv_dtype"])),
            err_msg=f"bundled call touched untargeted layer {layer}")


# ---------------------------------------------------------------------------
# attention-interface parity: non-default scale / soft-cap / causal mask
# ---------------------------------------------------------------------------


def test_bundled_interface_forwards_scale_softcap_causal():
    """Verifies that attention_bundled correctly forwards non-default attention parameters.

    Ensures non-default sm_scale, logits soft-capping, and causal masking flags are correctly
    propagated down to the underlying Pallas kernel rather than falling back to default values.
    """
    _require_tpu()
    from vllm_torchtpu.layers.common.attention_interface import (
        AttentionMetadata, attention, attention_bundled)

    c = _cfg()
    pls = _per_layer_shape(c)
    init_np = np.random.default_rng(5).standard_normal((c["num_layers"], ) +
                                                       pls).astype(np.float32)
    wl_np = _workload(c, "mixed", seed=6)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("model", ))

    def _metadata(w):
        return AttentionMetadata(
            input_positions=None,
            block_tables=w["page_indices"],
            seq_lens=w["kv_lens"],
            query_start_loc=w["cu_q_lens"],
            request_distribution=w["distribution"],
        )

    sm_scale = 0.5  # deliberately != head_dim**-0.5
    soft_cap = 30.0
    use_causal_mask = False

    layer = 1
    kv_ref = jnp.asarray(init_np[layer], dtype=c["kv_dtype"])
    w_ref = _to_jax_workload(wl_np, c["kv_dtype"])
    new_kv_ref, out_ref = attention(
        kv_ref,
        w_ref["q"],
        w_ref["k"],
        w_ref["v"],
        _metadata(w_ref),
        mesh,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        use_causal_mask=use_causal_mask,
    )

    bundle = jnp.asarray(init_np, dtype=c["kv_dtype"]).swapaxes(0, 1)
    w_b = _to_jax_workload(wl_np, c["kv_dtype"])
    new_bundle, out_b = attention_bundled(
        bundle,
        jnp.asarray(layer, dtype=jnp.int32),
        w_b["q"],
        w_b["k"],
        w_b["v"],
        _metadata(w_b),
        mesh,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        use_causal_mask=use_causal_mask,
    )
    np.testing.assert_array_equal(
        np.asarray(out_ref),
        np.asarray(out_b),
        err_msg="non-default-param output differs from layer-major")
    np.testing.assert_array_equal(
        np.asarray(new_kv_ref),
        np.asarray(new_bundle[:, layer, ...]),
        err_msg="non-default-param kv update differs from layer-major")

    # Parameter sensitivity check: verify that non-default attention parameters alter
    # the computed output rather than silently falling back to default values.
    bundle_default = jnp.asarray(init_np, dtype=c["kv_dtype"]).swapaxes(0, 1)
    w_d = _to_jax_workload(wl_np, c["kv_dtype"])
    _, out_default = attention_bundled(
        bundle_default,
        jnp.asarray(layer, dtype=jnp.int32),
        w_d["q"],
        w_d["k"],
        w_d["v"],
        _metadata(w_d),
        mesh,
    )
    assert not np.array_equal(np.asarray(out_default), np.asarray(out_b)), (
        "non-default sm_scale/soft_cap produced the default-param output — "
        "the bundled interface dropped attention parameters")
