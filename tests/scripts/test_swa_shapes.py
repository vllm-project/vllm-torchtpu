# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch regressions; stop before hardware-specific execution."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from scripts.dsv41_bench import swa_shapes


@pytest.fixture
def modules():
    return swa_shapes.load_modules()


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
def test_preparation_exact_shape_dispatch(
    modules, monkeypatch, tokens, requests, fused
):
    original, candidate, cache, _, _, _ = modules
    monkeypatch.setattr(cache, "prepare_swa_metadata", stop("fallback"))
    for module, expected in (
        (original, (tokens, requests) == (288, 48)),
        (candidate, fused),
    ):
        monkeypatch.setattr(module, "_swa_capacities", stop("fused"))
        with pytest.raises(
            DispatchReached, match="^fused$" if expected else "^fallback$"
        ):
            module.swa_prepare_publish(
                (),
                SimpleNamespace(shape=(requests, 8)),
                None,
                SimpleNamespace(shape=(tokens,)),
                None,
                None,
            )


@pytest.mark.parametrize("tokens", [95, 96, 128, 192, 288, 384])
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("prepared", [None, ()])
def test_writer_pack_dispatch_and_baseline_isolation(
    modules,
    monkeypatch,
    tokens,
    page_size,
    prepared,
):
    original, candidate, _, original_writer, writer, _ = modules
    info = SimpleNamespace(num_lanes=8, num_cores=1, num_subcores=1)
    monkeypatch.setattr(
        writer.pltpu, "get_tpu_info", lambda: SimpleNamespace(sparse_core=info)
    )
    monkeypatch.setattr(writer.plsc, "VectorSubcoreMesh", lambda **kwargs: None)
    monkeypatch.setattr(original, "swa_pack", stop("original_pack"))
    monkeypatch.setattr(candidate, "swa_pack", stop("candidate_pack"))
    monkeypatch.setattr(writer.jnp, "argsort", stop("native"))
    pages = jnp.zeros((2, page_size, 512), jnp.bfloat16)
    records = jnp.arange(tokens, dtype=jnp.int32)
    values = jnp.zeros((tokens, 512), jnp.bfloat16)
    for module, accepted, path in (
        (original_writer, (288,), "original_pack"),
        (writer, (96, 192, 288), "candidate_pack"),
    ):
        # An explicitly supplied empty tuple bypasses packing, then fails indexing.
        if prepared is not None:
            with pytest.raises(IndexError):
                module.write_swa_pages.__wrapped__(
                    pages, records, values, prepared=prepared
                )
        else:
            expected = path if tokens in accepted and page_size == 64 else "native"
            with pytest.raises(DispatchReached, match=f"^{expected}$"):
                module.write_swa_pages.__wrapped__(pages, records, values)


def test_full_region_baseline_local_writer_is_frozen(modules, monkeypatch):
    original, candidate, _, original_writer, writer, _ = modules
    monkeypatch.setattr(original_writer, "write_swa_pages", stop("original_writer"))
    monkeypatch.setattr(writer, "write_swa_pages", stop("candidate_writer"))
    for module, expected in (
        (original, "original_writer"),
        (candidate, "candidate_writer"),
    ):
        monkeypatch.setattr(module, "_swa_capacities", lambda banks: ((), 128))
        monkeypatch.setattr(
            module,
            "fuse_function",
            lambda *a, **k: lambda *xs: (
                None,
                None,
                jnp.zeros((1, 288), jnp.int32),
                (),
            ),
        )
        with pytest.raises(DispatchReached, match=f"^{expected}$"):
            module.swa_prepare_publish(
                (None,),
                jnp.zeros((48, 8), jnp.int32),
                jnp.zeros(49, jnp.int32),
                jnp.zeros(288, jnp.int32),
                jnp.zeros(48, jnp.int32),
                jnp.zeros((288, 512), jnp.bfloat16),
            )
