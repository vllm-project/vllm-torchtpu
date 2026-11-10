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
"""Which RPA kernels get a fixed-length PREFILL stage (chunk_prefill_size)."""

import inspect
from types import SimpleNamespace

import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
from vllm_torchtpu.layers.adapter import attention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


def _params(impl):
    return inspect.signature(impl._kernel_entry).parameters


def _impl(cls, *, head_size=128, chunk_prefill_size=16, kernel_entry=None):
    impl = cls.__new__(cls)
    impl.head_size = head_size
    impl.chunk_prefill_size = chunk_prefill_size
    if kernel_entry is not None:
        impl._kernel_entry = kernel_entry
    return impl


@pytest.mark.parametrize(
    "kernel_entry",
    [None, attention._pallas_rpa_kernel_local],
    ids=["v3_sharded", "v3_local"],
)
def test_rpa_v3_entries_get_the_prefill_stage(kernel_entry):
    impl = _impl(attention.PallasAttentionBackendImpl, kernel_entry=kernel_entry)
    assert impl._kernel_chunk_prefill_size(False, _params(impl)) == 16


@pytest.mark.parametrize(
    "impl",
    [
        # Batched RPA has no PREFILL stage (it folds PREFILL into MIXED).
        _impl(attention.PallasBatchedRPAAttentionBackendImpl),
        # hd64 reroutes to a kernel without the stage.
        _impl(attention.PallasAttentionBackendImpl, head_size=64),
        # Not requested.
        _impl(attention.PallasAttentionBackendImpl, chunk_prefill_size=None),
    ],
    ids=["batched", "hd64", "unset"],
)
def test_other_kernels_get_no_prefill_stage(impl):
    assert impl._kernel_chunk_prefill_size(False, _params(impl)) is None


def test_pcp_streaming_gets_no_prefill_stage():
    impl = _impl(attention.PallasAttentionBackendImpl)
    assert impl._kernel_chunk_prefill_size(True, _params(impl)) is None


def test_replicated_batched_draft_on_local_v3_gets_the_stage():
    # The runner rebinds a replicated batched-RPA draft to the local RPA v3
    # kernel, which does have the stage.
    impl = _impl(
        attention.PallasBatchedRPAAttentionBackendImpl,
        kernel_entry=attention._pallas_rpa_kernel_local,
    )
    assert impl._kernel_chunk_prefill_size(False, _params(impl)) == 16


class _FakeOp:
    def register_fake(self, fn):
        return fn


def _build_capturing(monkeypatch, impl):
    """Run the real _build_rpa_kernel, capturing each kernel it registers."""
    built = []

    def fake_jax_op(name, fn, **kwargs):
        built.append(fn)
        return _FakeOp()

    monkeypatch.setattr(attention.pallas, "jax_op", fake_jax_op)
    mesh = SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})
    with set_vllm_model_wrapper_context(mesh=mesh, vllm_config=SimpleNamespace()):
        impl._build_rpa_kernel(None, None, None)
    return built


def _real_impl(cls=attention.PallasAttentionBackendImpl, head_size=128):
    # Every attribute _build_rpa_kernel reads, set explicitly: the full
    # constructor needs a TPU vLLM config, which the CPU agent lacks.
    impl = cls.__new__(cls)
    impl.head_size = head_size
    impl.sliding_window = None
    impl.scale = 1.0
    impl.logits_soft_cap = None
    impl.use_causal_mask = False
    impl.kv_layout = KVLayout.HEAD_ALONG_SUBLANE
    impl._pool_is_seq_along_lane = False
    impl._kernel_config_cache = {}
    impl.chunk_prefill_size = None
    return impl


@pytest.fixture
def fresh_kernel_registry(monkeypatch):
    # The registry is shared across instances; isolate each test's builds.
    monkeypatch.setattr(attention.PallasAttentionBackendImpl, "_kernel_registry", {})


@pytest.mark.parametrize(
    "kernel_entry",
    [None, attention._pallas_rpa_kernel_local],
    ids=["v3_sharded", "v3_local"],
)
def test_build_binds_the_recorded_prefill_stage(
    monkeypatch, fresh_kernel_registry, kernel_entry
):
    # DFlash routes rows to the PREFILL stage on the recorded value; the kernel
    # must actually be bound with that same size, or RPA v3 skips the rows.
    impl = _real_impl()
    if kernel_entry is not None:
        impl._kernel_entry = kernel_entry
    impl.chunk_prefill_size = 16
    (fn,) = _build_capturing(monkeypatch, impl)
    assert fn.keywords["chunk_prefill_size"] == 16
    assert impl._rpa_kernel_chunk_prefill_size == 16


@pytest.mark.parametrize(
    "impl_factory",
    [
        lambda: _real_impl(attention.PallasBatchedRPAAttentionBackendImpl),
        lambda: _real_impl(head_size=64),
    ],
    ids=["batched", "hd64"],
)
def test_build_binds_no_prefill_stage_without_support(
    monkeypatch, fresh_kernel_registry, impl_factory
):
    impl = impl_factory()
    impl.chunk_prefill_size = 16
    (fn,) = _build_capturing(monkeypatch, impl)
    assert "chunk_prefill_size" not in fn.keywords
    assert impl._rpa_kernel_chunk_prefill_size is None


def test_build_keys_kernels_by_prefill_stage(monkeypatch, fresh_kernel_registry):
    impl = _real_impl()
    impl.chunk_prefill_size = 16
    first = _build_capturing(monkeypatch, impl)
    impl.chunk_prefill_size = None
    second = _build_capturing(monkeypatch, impl)
    # A kernel without the stage is a different op, and the recorded value
    # follows the latest build.
    assert len(first) == len(second) == 1
    assert "chunk_prefill_size" not in second[0].keywords
    assert impl._rpa_kernel_chunk_prefill_size is None
    impl.chunk_prefill_size = 16
    assert _build_capturing(monkeypatch, impl) == []  # reused from the cache
    assert impl._rpa_kernel_chunk_prefill_size == 16


# --- TPU: RPA v3's PREFILL stage through the real kernel entries ------------

_Q_HEADS, _KV_HEADS, _HEAD_DIM, _PAGE = 8, 2, 128, 128
# KV lengths span several 1024-token capped KV blocks on the local path.
_KV_LENS = (1300, 2600, 3100)


def _prefill_stage_inputs(block, dtype, seed=0):
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as rpa_v3

    rng = np.random.default_rng(seed)
    num_reqs = len(_KV_LENS)
    max_seqs = 8
    pages_per_seq = -(-max(_KV_LENS) // _PAGE)
    num_tokens = 128
    num_pages = max_seqs * pages_per_seq + 4

    def rand(shape):
        return jnp.asarray(rng.standard_normal(shape, dtype=np.float32), dtype)

    q = rand((num_tokens, _Q_HEADS, _HEAD_DIM))
    k = rand((num_tokens, _KV_HEADS, _HEAD_DIM))
    v = rand((num_tokens, _KV_HEADS, _HEAD_DIM))
    kv_cache = rand(
        rpa_v3.get_kv_cache_shape(num_pages, _PAGE, _KV_HEADS, _HEAD_DIM, dtype)
    )
    pages = rng.permutation(num_pages)[: max_seqs * pages_per_seq]
    kv_lens = list(_KV_LENS) + [0] * (max_seqs - num_reqs)
    cu_q = [i * block for i in range(num_reqs + 1)]
    cu_q += [cu_q[-1]] * (max_seqs - num_reqs)
    return dict(
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=jnp.asarray(kv_lens, jnp.int32),
        page_indices=jnp.asarray(pages, jnp.int32),
        cu_q_lens=jnp.asarray(cu_q, jnp.int32),
        # Every request in RPA v3's PREFILL segment, as DFlash routes them.
        distribution=jnp.asarray([0, num_reqs, num_reqs], jnp.int32),
        num_q=num_reqs * block,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("block", [5, 16])
@pytest.mark.parametrize("entry_name", ["local", "sharded"])
def test_rpa_v3_prefill_stage_matches_reference(entry_name, block, causal, dtype_name):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tests.kernels.experimental.batched_rpa import correctness_test as ref

    dtype = getattr(jnp, dtype_name)
    x = _prefill_stage_inputs(block, dtype)
    sm_scale = _HEAD_DIM**-0.5
    entry = {
        "local": attention._pallas_rpa_kernel_local,
        "sharded": attention._pallas_rpa_kernel_default,
    }[entry_name]
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))

    # RPA v3 donates q/k/v/kv_cache; the reference below needs the originals.
    new_kv, out = entry(
        x["kv_cache"].copy(),
        x["q"].copy(),
        x["k"].copy(),
        x["v"].copy(),
        x["kv_lens"],
        x["page_indices"],
        x["cu_q_lens"],
        x["distribution"],
        None,  # sinks
        None,  # q_scale
        None,  # k_scale
        None,  # v_scale
        mesh=mesh,
        sliding_window=None,
        sm_scale=sm_scale,
        soft_cap=None,
        skip_kv_update=False,
        use_causal_mask=causal,
        chunk_prefill_size=block,
    )

    # Independent reference in f32, on the f32 cache layout (1 value per slot).
    p, ps, hp, pk, d = x["kv_cache"].shape
    ref_cache = x["kv_cache"].astype(jnp.float32).reshape(p, ps, hp * pk, 1, d)
    ref_out, ref_kv = ref.ref_ragged_paged_attention(
        x["q"].astype(jnp.float32),
        x["k"].astype(jnp.float32),
        x["v"].astype(jnp.float32),
        ref_cache,
        x["kv_lens"],
        x["page_indices"],
        x["cu_q_lens"],
        x["distribution"].at[:2].set(0),  # the reference runs every request
        use_causal_mask=causal,
        sm_scale=sm_scale,
    )
    n = x["num_q"]
    got = np.asarray(out[:n], np.float32)
    want = np.asarray(ref_out[:n], np.float32)
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    # TPU matmuls run f32 at the default (bf16-pass) precision: ~2e-3 relative.
    assert rel < (5e-3 if dtype_name == "float32" else 2e-2), rel
    # The stage writes exactly the new K/V the reference writes.
    np.testing.assert_array_equal(
        np.asarray(new_kv.astype(jnp.float32).reshape(ref_kv.shape)),
        np.asarray(ref_kv),
    )
