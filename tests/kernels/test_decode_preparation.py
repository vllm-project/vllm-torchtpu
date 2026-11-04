# SPDX-License-Identifier: Apache-2.0
"""CPU guards for decode preparation without benchmark dependencies."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.deepseek_v41.csa2 import cache
from vllm_torchtpu.kernels.experimental import decode_preparation as prep


class DispatchReached(Exception):
    pass


def stop(path):
    def reached(*args, **kwargs):
        raise DispatchReached(path)

    return reached


@pytest.mark.parametrize(
    "tokens,requests,fused",
    [
        (96, 16, True),
        (192, 32, True),
        (288, 32, True),
        (288, 48, True),
        (96, 32, False),
        (192, 16, False),
        (288, 16, False),
        (128, 16, False),
    ],
)
def test_swa_exact_shape_dispatch(monkeypatch, tokens, requests, fused):
    monkeypatch.setattr(cache, "prepare_swa_metadata", stop("fallback"))
    monkeypatch.setattr(prep, "_swa_capacities", stop("fused"))
    with pytest.raises(DispatchReached, match="^fused$" if fused else "^fallback$"):
        prep.swa_prepare_publish(
            (),
            jax.ShapeDtypeStruct((requests, 8), jnp.int32),
            None,
            jax.ShapeDtypeStruct((tokens,), jnp.int32),
            None,
            None,
        )


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"tokens": 287},
        {"table_shape": (48, 9)},
        {"capacity": 32},
        {"page_stride": 32},
        {"row_offset": 1},
        {"window": 256},
    ],
)
def test_state_packed_exact_envelope(change):
    args = dict(
        tokens=288,
        table_shape=(48, 8),
        capacity=16,
        page_stride=16,
        row_offset=0,
        window=128,
    )
    args.update(change)
    assert prep.supports_compressor_state_packed(**args) == (not change)
