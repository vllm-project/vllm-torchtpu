from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.megablox import common, tuned_block_sizes


@pytest.mark.parametrize("kind, generation", [("TPU v4", 4), ("TPU v5p", 5),
                                              ("TPU v5 lite", 5),
                                              ("TPU v6e", 6), ("TPU7x", 7)])
def test_tpu_identification(monkeypatch, kind, generation):
    monkeypatch.setattr(common.jax, "devices",
                        lambda: [SimpleNamespace(device_kind=kind)])
    assert common.is_tpu()
    assert common.tpu_kind() == kind
    assert common.tpu_generation() == generation


@pytest.mark.parametrize("kind", ["cpu", "NVIDIA A100", "TPU unknown"])
def test_unsupported_generation(monkeypatch, kind):
    monkeypatch.setattr(common.jax, "devices",
                        lambda: [SimpleNamespace(device_kind=kind)])
    assert common.is_tpu() == kind.startswith("TPU")
    with pytest.raises(NotImplementedError, match="only TPU"):
        common.tpu_generation()


@pytest.mark.parametrize("dtype", [
    jnp.bfloat16, jnp.float32, jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.int8,
    jnp.int4, jnp.float4_e2m1fn, jnp.uint4
])
def test_supported_dtype(dtype):
    common.assert_is_supported_dtype(dtype)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.int32, jnp.bool_])
def test_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="No support"):
        common.assert_is_supported_dtype(dtype)


@pytest.mark.parametrize("x, limit, expected", [(1, 512, 128), (128, 512, 128),
                                                (129, 512, 256),
                                                (511, 512, 512),
                                                (2048, 2048, 2048),
                                                (3072, 2048, 1536),
                                                (2051, 2048, 2048)])
def test_rounding_and_divisor_selection(x, limit, expected):
    assert tuned_block_sizes.round_up_to_multiple_of_128_within_limit(
        x, limit) == expected


@pytest.mark.parametrize("limit", [0, 127, 129])
def test_invalid_tile_limit(limit):
    with pytest.raises(AssertionError):
        tuned_block_sizes.round_up_to_multiple_of_128_within_limit(256, limit)


@pytest.mark.parametrize("m, k, n, groups, expected", [
    (320, 3072, 1000, 8, (80, 1536, 1024)),
    (64, 128, 256, 8, (64, 128, 256)),
    (4096, 4096, 6144, 8, (512, 2048, 2048)),
])
def test_default_tiles(m, k, n, groups, expected):
    tiles = tuned_block_sizes.get_default_gmm_block_sizes(m, k, n, groups)
    assert tiles == expected
    assert m % tiles[0] == 0


def test_tuned_lookup():
    assert tuned_block_sizes.get_tuned_block_sizes(128, 320, 6144, 160, 160,
                                                   'bfloat16', 'float8_e4m3fn',
                                                   320) == (128, 640, 1280)


def test_lookup_falls_back_for_untuned_shape():
    assert tuned_block_sizes.get_tuned_block_sizes(320, 3072, 1000, 16, 8,
                                                   'float32', 'float32',
                                                   0) == (80, 1536, 1024)
