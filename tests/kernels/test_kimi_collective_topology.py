# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical routing and Torch token ownership need independent permutations."""
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_torchtpu.kernels.kimi_k3.collectives import _regions, ring_tables


def test_moe_gather_preserves_bits_and_token_owners(monkeypatch):
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.kimi_k3 import collectives

    devices = jax.devices()
    mesh = Mesh(np.asarray(devices), ('k3_sp', ))
    owners = np.random.default_rng(913).permutation(len(devices))
    # NaN payloads can be canonicalized by host BF16 transport. Router
    # weights are finite; retain signed zero and infinities. CPU also checks
    # subnormals, which TPU floating-point operations flush to signed zero.
    bits = np.arange(65536, dtype=np.uint16).reshape(4096, 16)
    bits = np.where(((bits & 0x7f80) == 0x7f80) & ((bits & 0x7f) != 0), 0,
                    bits).astype(np.uint16)
    if any(device.platform == "tpu" for device in devices):
        bits = np.where((bits & 0x7f80) == 0, bits & 0x8000,
                        bits).astype(np.uint16)
    ids = np.arange(65536, dtype=np.uint16).view(np.int16).astype(
        np.int32).reshape(bits.shape)
    sharding = NamedSharding(mesh, P('k3_sp', None))
    weights = jax.device_put(bits.view(ml_dtypes.bfloat16), sharding)
    ids_device = jax.device_put(ids, sharding)
    latent_host = np.random.default_rng(914).normal(size=(4096, 384)).astype(
        ml_dtypes.bfloat16)
    latent = jax.device_put(latent_host, sharding)

    def native_gather(payload, mesh, tables, *, gather, collective_id):
        assert gather

        @jax.shard_map(mesh=mesh,
                       in_specs=P('k3_sp', None),
                       out_specs=P(),
                       check_vma=False)
        def local(x):
            full = jax.lax.all_gather(x, 'k3_sp', axis=0, tiled=False)
            return full[jnp.asarray(np.argsort(owners))].reshape(
                -1, x.shape[1])

        return local(payload)

    # Isolate packing/ownership with native collectives; test Pallas rings separately.
    monkeypatch.setattr(collectives, 'ring_collective', native_gather)
    actual_x, actual_w, actual_i = jax.jit(
        lambda x, w, i: collectives.gather_moe_inputs(
            x, w, i, mesh, None, collective_id=29004))(latent, weights,
                                                       ids_device)
    order = np.argsort(owners)
    expected_w = bits.reshape(len(devices), -1, 16)[order].reshape(-1, 16)
    expected_i = ids.reshape(len(devices), -1, 16)[order].reshape(-1, 16)
    expected_x = latent_host.reshape(len(devices), -1,
                                     384)[order].reshape(-1, 384)
    np.testing.assert_array_equal(np.asarray(actual_x), expected_x)
    assert actual_x.sharding.is_fully_replicated
    np.testing.assert_array_equal(
        np.asarray(actual_w).view(np.uint16), expected_w)
    np.testing.assert_array_equal(np.asarray(actual_i), expected_i)
    assert actual_w.sharding.is_fully_replicated
    assert actual_i.sharding.is_fully_replicated
    assert actual_i.dtype == jnp.int32


def test_ring_tables_preserve_physical_links_and_torch_token_order():
    rng = np.random.default_rng(73)
    physical = rng.permutation(32)
    owners = rng.permutation(32)
    devices = [
        SimpleNamespace(id=i,
                        coords=(p // 16, p // 8 % 2, p // 2 % 4),
                        core_on_chip=int(p % 2))
        for i, p in enumerate(physical)
    ]
    tables = ring_tables(devices, owners)
    for positions, peers, tokens in tables:
        np.testing.assert_array_equal(peers[positions], np.arange(32))
        np.testing.assert_array_equal(tokens, owners)
        for rank in range(32):
            for direction in (-1, 1):
                pos = positions[rank]
                peer = peers[(pos + direction) % 32]
                distance = sum(
                    abs(a - b) for a, b in zip(devices[rank].coords,
                                               devices[peer].coords))
                assert distance <= 1
                assert rank != peer
                destinations = [
                    tokens[peers[(pos - direction * step) % 32]]
                    for step in range(32)
                ]
                assert sorted(destinations) == list(range(32))
                assert destinations[0] == owners[rank]
                # Last RS iteration reads and emits this rank's token slice.
                assert tokens[peers[(pos - direction * 32) %
                                    32]] == owners[rank]


@pytest.mark.parametrize('rows', [128, 256])
@pytest.mark.parametrize('width', [2176, 3584, 7168])
def test_ring_regions_cover_payload_once(rows, width):
    count = np.zeros((rows, width), np.int8)
    for start, size, col, ncols in _regions(rows, width):
        assert start % 16 == size % 16 == col % 128 == ncols % 128 == 0
        count[start:start + size, col:col + ncols] += 1
    np.testing.assert_array_equal(count, np.ones_like(count))


def test_mla_chip_layout_preserves_gate_heads_and_token_owners():
    from vllm_torchtpu.kernels.kimi_k3.collectives import mla_chip_layout

    rng = np.random.default_rng(792)
    physical = rng.permutation(32)
    owners = rng.permutation(32)
    devices = [
        SimpleNamespace(coords=(p // 16, p // 8 % 2, p // 2 % 4))
        for p in physical
    ]
    pairs, parity, token_order = mla_chip_layout(devices, owners)
    assert sorted(i for pair in pairs for i in pair) == list(range(32))
    for pair in pairs:
        assert devices[pair[0]].coords == devices[pair[1]].coords
        assert owners[pair[0]] < owners[pair[1]]
    # Pack gate columns in exactly the destination order of all-to-all.
    # Every destination must receive its own heads and all owners' rows.
    pair_rows = np.asarray([[owners[i] for i in pair] for pair in pairs])
    for group in parity:
        for destination, mesh_index in enumerate(group):
            packed_heads = owners[np.asarray(group)]
            assert packed_heads[destination] == owners[mesh_index]
            received_rows = pair_rows.reshape(-1)[np.asarray(token_order)]
            np.testing.assert_array_equal(received_rows, np.arange(32))


@pytest.mark.parametrize('chip_count,cores', [(8, 2), (16, 1), (16, 3)])
def test_mla_chip_layout_rejects_incomplete_pairs(chip_count, cores):
    from vllm_torchtpu.kernels.kimi_k3.collectives import mla_chip_layout

    devices = [
        SimpleNamespace(coords=(chip, 0, 0)) for chip in range(chip_count)
        for _ in range(cores)
    ]
    with pytest.raises(ValueError, match='sixteen two-core chips'):
        mla_chip_layout(devices, tuple(range(len(devices))))
