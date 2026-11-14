# SPDX-License-Identifier: Apache-2.0
"""TPU_RELEASE_JAX_CACHES drops JAX's caches once, after warm-up."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.worker import tpu_worker


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("TPU_RELEASE_JAX_CACHES", raising=False)
    assert envs.TPU_RELEASE_JAX_CACHES is False


def test_release_empties_jit_cache_and_retrace_still_works():
    @jax.jit
    def f(x):
        return x * 2 + 1

    x = jnp.arange(8)
    f(x)
    assert f._cache_size() == 1
    tpu_worker._release_jax_caches()
    assert f._cache_size() == 0
    assert (f(x) == x * 2 + 1).all()  # re-traced on demand
    assert f._cache_size() == 1


def test_flag_ignored_by_compile_cache_key():
    from vllm_torchtpu.compilation import tpu_compiler

    assert "TPU_RELEASE_JAX_CACHES" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "TPU_HOST_HEAP_REPORT" in tpu_compiler._TPU_COMPILE_ENV_IGNORED


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
