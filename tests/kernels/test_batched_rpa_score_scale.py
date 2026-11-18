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
"""The batched RPA score scaling keeps the plain path's bits.

`flash_attention_qk_softmax` folds the static score scale into q only when
sm_scale and k_scale are each a power of two, so the fold is exact. Every other
scale (head_dim 128, a calibrated k_scale, a static q scale, per-token scales,
or a k_scale whose product with sm_scale happens to be a power of two) must
take the plain path's multiplies one factor at a time: merging the factors
first rounds differently.
"""

import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, flash_attention

_TQ, _S = 64, 1024


@pytest.fixture(autouse=True)
def tpu7x(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *args, **kwargs: info)


def _cfgs(mode, head_dim, dtype, *, scale_q=None, scale_k=None, per_token=False):
    model = configs.ModelConfigs(
        num_q_heads=1,
        num_kv_heads=1,
        head_dim=head_dim,
        mask_value=-1e9,
        sm_scale=head_dim**-0.5,
    )
    serve = configs.ServingConfigs(
        num_seqs=8,
        page_size=128,
        total_q_tokens=_TQ,
        num_page_indices=64,
        dtype_q=dtype,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=dtype,
        per_token_scale=per_token,
        per_token_scale_dtype=jnp.float32 if per_token else None,
        scale_q=scale_q,
        scale_k=scale_k,
    )
    block = configs.BlockSizes(bq_sz=_TQ, bq_c_sz=_TQ, bkv_sz=_S, batch_size=1)
    return configs.RpaConfigs(
        block=block, model=model, serve=serve, mode=mode, vmem_limit_bytes=1 << 26
    )


def _scores(cfgs, k_scale, *, diet):
    rng = np.random.default_rng(7)
    h = cfgs.model.head_dim
    q = jnp.asarray(rng.standard_normal((1, 1, _TQ, h)), cfgs.serve.dtype_q)
    k = jnp.asarray(rng.standard_normal((1, 1, _S, h)), cfgs.serve.dtype_q)
    m_prev = jnp.full((1, _TQ, 128), -jnp.inf, jnp.float32)
    l_prev = jnp.zeros((1, _TQ, 128), jnp.float32)
    is_last_k = jnp.zeros((1, 1), jnp.int32)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(flash_attention, "vpu_diet_enabled", lambda: diet)
        p, _, m_next, l_next, _, _ = flash_attention.flash_attention_qk_softmax(
            0,
            q,
            k,
            m_prev,
            l_prev,
            is_last_k,
            custom_mask=None,
            cfgs=cfgs,
            bq_start=0,
            k_scale=k_scale,
        )
    return [
        np.asarray(x.astype(jnp.float32)).view(np.uint32) for x in (p, m_next, l_next)
    ]


def _per_token_k_scale():
    rng = np.random.default_rng(3)
    return jnp.asarray(rng.uniform(0.05, 0.9, (1, 1, 1, _S)), jnp.float32)


# (head_dim, scale_q, scale_k, per-token k scales): none folds exactly, and
# with head_dim 128 (sm_scale not a power of two) merging any second factor
# into sm_scale rounds differently from multiplying by each in turn.
_INEXACT = {
    "k_scale": (128, None, 0.37, False),
    "q_scale": (128, 2.5, None, False),
    "q_and_k_scale": (128, 2.5, 0.37, False),
    "per_token": (128, None, None, True),
    # sm_scale 1/sqrt(128) times this k_scale is exactly 0.125 in double, but
    # neither factor is a power of two, so the plain path rounds twice.
    "k_scale_cancels_sm_scale": (128, None, 1.414213562373095, False),
}


@pytest.mark.parametrize("mode", [configs.RpaCase.DECODE, configs.RpaCase.MIXED])
@pytest.mark.parametrize("case", list(_INEXACT))
def test_inexact_scale_takes_the_plain_multiplies(mode, case):
    """f32 scores, so a merged multiply's rounding is not hidden by a bf16 cast."""
    head_dim, scale_q, scale_k, per_token = _INEXACT[case]
    cfgs = _cfgs(
        mode,
        head_dim,
        jnp.float32,
        scale_q=scale_q,
        scale_k=scale_k,
        per_token=per_token,
    )
    assert flash_attention.q_folded_scale(cfgs) is None
    k_scale = _per_token_k_scale() if per_token else scale_k
    diet = _scores(cfgs, k_scale, diet=True)
    plain = _scores(cfgs, k_scale, diet=False)
    for name, a, b in zip(("p", "m_next", "l_next"), diet, plain):
        np.testing.assert_array_equal(a, b, err_msg=name)


@pytest.mark.parametrize("mode", [configs.RpaCase.DECODE, configs.RpaCase.MIXED])
@pytest.mark.parametrize("scale_k", [None, 1.0])
def test_power_of_two_scale_folds_into_q_exactly(mode, scale_k):
    """head_dim 256 with k_scale 1 (the Qwen3.5 fp8 layers): sm_scale = 1/16 is
    folded into q, and the scores keep the plain path's bits."""
    cfgs = _cfgs(mode, 256, jnp.bfloat16, scale_k=scale_k)
    assert flash_attention.q_folded_scale(cfgs) == 1 / 16
    diet = _scores(cfgs, scale_k, diet=True)
    plain = _scores(cfgs, scale_k, diet=False)
    for name, a, b in zip(("p", "m_next", "l_next"), diet, plain):
        np.testing.assert_array_equal(a, b, err_msg=name)


@pytest.mark.parametrize(
    ("dtype", "folds"),
    [(jnp.bfloat16, True), (jnp.float32, True), (jnp.float16, False)],
)
def test_only_a_bf16_or_f32_q_folds(dtype, folds):
    """Scaling an fp16 q by a power of two can push it out of its normal
    range, so the fold is exact only for bf16 and f32 q."""
    cfgs = _cfgs(configs.RpaCase.DECODE, 256, dtype)
    assert (flash_attention.q_folded_scale(cfgs) is not None) == folds
