# SPDX-License-Identifier: Apache-2.0
"""additional_config release_jax_caches drops JAX's caches after warm-up."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.opt_in_config import option


def test_key_defaults_off():
    assert option({}, "release_jax_caches") is False
    assert option(None, "release_jax_caches") is False
    assert option({"release_jax_caches": True}, "release_jax_caches") is True
    with pytest.raises(ValueError):
        option({"release_jax_caches": 1}, "release_jax_caches")


def test_release_empties_jit_cache_and_retrace_still_works():
    tpu_worker = pytest.importorskip("vllm_torchtpu.worker.tpu_worker")

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
