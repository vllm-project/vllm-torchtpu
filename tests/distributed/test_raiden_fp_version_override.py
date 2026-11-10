# SPDX-License-Identifier: Apache-2.0
"""Deviceless tests for TPU_RAIDEN_FINGERPRINT_VERSION_OVERRIDE.

The goldens are the two Stage-3 fingerprints a PCP prefill on a shadow runtime
and an R12p decode on the image runtime registered on 2026-10-30; an exact
sha256 preimage search showed they differ only in the torch_tpu / libtpu
strings. The override must make the producer hash to the consumer's value
while any real layout difference still fails closed.
"""

import inspect

import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import layout_fingerprint as rlf
from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol

from .test_seq_on_lane import (
    PRODUCER_SHAPE,
    SOL_LAYOUT,
    TP2_GDN,
    TP2_SHAPE,
    TP8_GDN,
    _manifest,
)

ENV = rlf.FINGERPRINT_VERSION_OVERRIDE_ENV
IMAGE = {"torch_tpu": "0.2.0rc1", "libtpu": "0.0.50"}
SHADOW = {"torch_tpu": "0.2.1.dev20261021050749", "libtpu": "0.0.51"}
OVERRIDE = "torch_tpu=0.2.0rc1,libtpu=0.0.50"
DECODE_FP = "a282b76caef6f4b999ae9b9d13d2bfff40fc29dd752b95778ccc767bad24dff6"
PREFILL_FP = "416d3296b323b9a361c60fd585da988f5c50f9611dc702d4003ee541a8ce5a2b"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    # Both production legs register gdn_conv_layout=qk-pair-v1.
    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")


def _producer(shape=PRODUCER_SHAPE):
    return _manifest(
        shape, TP8_GDN, num_kv_heads=2, builder=sol.build_qwen35_pool_manifest_sol
    )


def _decode():
    return _manifest(
        TP2_SHAPE, TP2_GDN, num_kv_heads=1, builder=sol.build_qwen35_pool_manifest_sol
    )


def _fp(manifest, versions, layout=SOL_LAYOUT):
    return sol.measured_sol_layout_fingerprint(
        manifest,
        layout_getter=lambda tensor: layout,
        package_version=versions.__getitem__,
    )


def test_goldens_reproduce_the_production_fingerprints():
    assert _fp(_decode(), IMAGE)[0] == DECODE_FP
    assert _fp(_producer(), SHADOW)[0] == PREFILL_FP


@pytest.mark.parametrize("value", [None, "", "   "])
def test_default_off_reports_installed_versions(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    fingerprint, payload = _fp(_producer(), SHADOW)
    assert (payload["torch_tpu"], payload["libtpu"]) == (
        SHADOW["torch_tpu"],
        SHADOW["libtpu"],
    )
    assert fingerprint == PREFILL_FP != DECODE_FP


def test_override_makes_the_producer_match_the_decode(monkeypatch):
    decode_fp, decode_payload = _fp(_decode(), IMAGE)
    monkeypatch.setenv(ENV, OVERRIDE)
    fingerprint, payload = _fp(_producer(), SHADOW)
    assert fingerprint == decode_fp == DECODE_FP
    assert payload == decode_payload


def test_override_only_replaces_the_version_strings(monkeypatch):
    _, off = _fp(_producer(), SHADOW)
    monkeypatch.setenv(ENV, OVERRIDE)
    _, on = _fp(_producer(), SHADOW)
    assert {k for k in off if off[k] != on[k]} == {"torch_tpu", "libtpu"}
    assert (on["torch_tpu"], on["libtpu"]) == (IMAGE["torch_tpu"], IMAGE["libtpu"])


def test_override_with_a_gdn_conv_layout_mutant_still_mismatches(monkeypatch):
    monkeypatch.setenv(ENV, OVERRIDE)
    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "0")
    fingerprint, payload = _fp(_producer(), SHADOW)
    assert payload["gdn_conv_layout"] == "legacy-split-qk"
    assert fingerprint != DECODE_FP


def test_override_with_a_kernel_page_mutant_still_mismatches(monkeypatch):
    monkeypatch.setenv(ENV, OVERRIDE)
    try:
        fingerprint, payload = _fp(_producer(shape=(1024, 4, 64, 4, 256)), SHADOW)
    except (ValueError, RuntimeError):
        return  # refused outright: also fail-closed
    assert payload["fa_kernel_page_tokens"] == 256
    assert fingerprint != DECODE_FP


@pytest.mark.parametrize(
    "layout",
    [
        ((3, 4, 2, 1, 0), ((4, 128), (4, 1)), 8),
        ((4, 3, 2, 1, 0), ((8, 128), (4, 1)), 8),
        ((4, 3, 2, 1, 0), ((4, 128), (4, 1)), 16),
    ],
)
def test_override_does_not_bypass_the_layout_gates(monkeypatch, layout):
    monkeypatch.setenv(ENV, OVERRIDE)
    with pytest.raises(RuntimeError):
        _fp(_producer(), SHADOW, layout=layout)


@pytest.mark.parametrize(
    "value",
    [
        "torch_tpu=0.2.0rc1",
        "libtpu=0.0.50",
        "torch_tpu=0.2.0rc1,libtpu=",
        "torch_tpu:0.2.0rc1,libtpu:0.0.50",
        "torch_tpu=0.2.0rc1,libtpu=0.0.50,schema=x",
        "torch_tpu=0.2.0rc1,torch_tpu=0.2.0,libtpu=0.0.50",
        "0.2.0rc1,0.0.50",
    ],
)
def test_malformed_override_fails_closed(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV):
        _fp(_producer(), SHADOW)


def test_override_is_logged_and_default_is_silent(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rlf,
        "_log_version_override",
        lambda override, installed: calls.append(
            (dict(override), installed("torch_tpu"), installed("libtpu"))
        ),
    )
    _fp(_producer(), SHADOW)
    assert calls == []
    monkeypatch.setenv(ENV, OVERRIDE)
    _fp(_producer(), SHADOW)
    assert calls == [(IMAGE, SHADOW["torch_tpu"], SHADOW["libtpu"])]


def test_token_major_fingerprint_honors_the_override(monkeypatch):
    version = rlf.fingerprint_package_version(SHADOW.__getitem__)
    assert version("torch_tpu") == SHADOW["torch_tpu"]
    monkeypatch.setenv(ENV, OVERRIDE)
    version = rlf.fingerprint_package_version(SHADOW.__getitem__)
    assert (version("torch_tpu"), version("libtpu")) == ("0.2.0rc1", "0.0.50")


def test_no_fingerprint_reads_package_versions_directly():
    measured = [
        rlf.measured_fa_layout_fingerprint,
        rlf.measured_kimi_k3_layout_fingerprint,
        rlf.measured_glm_layout_fingerprint,
        rlf.measured_dsv4_layout_fingerprint,
        sol.measured_sol_layout_fingerprint,
    ]
    for fn in measured:
        source = inspect.getsource(fn)
        assert "importlib.metadata" not in source, fn.__name__
        assert "fingerprint_package_version(package_version)" in source, fn.__name__
