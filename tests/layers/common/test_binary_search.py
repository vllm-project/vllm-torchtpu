import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.common import binary_search


@pytest.mark.parametrize("compiled", [False, True])
def test_int32_search_boundaries(compiled):
    thresholds = jnp.array([[-2147483648, -100, -1], [0, 123, 2147483647]],
                           dtype=jnp.int32)

    def search():
        return binary_search.int32_bsearch(thresholds.shape,
                                           lambda x: x > thresholds)

    result = jax.jit(search)() if compiled else search()
    assert result.dtype == jnp.int32
    np.testing.assert_array_equal(result, thresholds)


@pytest.mark.parametrize("satisfied, expected", [(True, -2147483648),
                                                 (False, 2147483647)])
def test_int32_search_constant_predicate(satisfied, expected):
    result = binary_search.int32_bsearch(
        (), lambda x: jnp.full_like(x, satisfied, dtype=jnp.bool_))
    assert result.shape == ()
    assert int(result) == expected


def test_float32_search_boundaries():
    maximum = np.finfo(np.float32).max
    thresholds = jnp.array([[-maximum, -10.5, -1.0], [0.0, 3.25, maximum]],
                           dtype=jnp.float32)
    result = jax.jit(lambda t: binary_search.float32_bsearch(
        t.shape, lambda x: x > t))(thresholds)
    # XLA comparisons can flush subnormal float32 values to zero.
    np.testing.assert_allclose(result,
                               thresholds,
                               rtol=0,
                               atol=np.finfo(np.float32).tiny)
    assert result.dtype == jnp.float32


def test_float32_search_strict_threshold():
    threshold = np.float32(1.0)
    result = binary_search.float32_bsearch((), lambda x: x >= threshold)
    assert result == np.nextafter(threshold, np.float32(-np.inf))


@pytest.mark.parametrize("shape", [(5, ), (2, 5), (2, 3, 5)])
@pytest.mark.parametrize("k", [1, 2, 5])
def test_topk_matches_sorted_cutoff(shape, k):
    values = np.random.default_rng(0).permuted(np.broadcast_to(
        np.array([-4., 3., 1., 3., -2.], np.float32), shape),
                                               axis=-1)
    cutoff = np.sort(values, axis=-1)[..., -k, None]
    expected = np.where(values >= cutoff, values, -999.)
    result = jax.jit(lambda x: binary_search.topk_mask(x, k, -999.))(
        jnp.asarray(values))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(5, ), (2, 5), (2, 3, 5)])
@pytest.mark.parametrize("p, cutoff", [(0.1, .5), (0.65, .2), (1.0, .02)])
def test_topp_includes_cutoff_ties(shape, p, cutoff):
    probabilities = np.random.default_rng(0).permuted(np.broadcast_to(
        np.array([.5, .2, .2, .08, .02], np.float32), shape),
                                                      axis=-1)
    logits = np.log(probabilities)
    expected = np.where(probabilities >= cutoff, logits, -999.)
    result = jax.jit(lambda x: binary_search.topp_mask(x, p, -999.))(
        jnp.asarray(logits))
    np.testing.assert_array_equal(result, expected)
