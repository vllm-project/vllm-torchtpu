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
"""The consumer-only KV transfer feature (`kv_transfer_consumer`).

A plugin that declares it serves a KV transfer connector only on a KV
consumer engine, and only when every transfer connector's class declares
`layerwise_transfer = False` (no per-layer load or save hooks for a program
that bypasses the attention modules to skip). Producers, `kv_both`,
layer-wise connectors and connectors that declare nothing are declined.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import vllm_torchtpu.megakernel.adapter.guards as guards_mod
from vllm_torchtpu.megakernel.adapter import testing

_CONSUMER = "kv_transfer_consumer"
_CAPS = frozenset({"spec_decode", "mrope", "prefix_caching", "kv_offload", _CONSUMER})


class _RaidenLike:
    """Loads in `start_load_kv`, completes through `get_finished`; the
    layer-wise hooks are no-ops."""

    layerwise_transfer = False

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, *args: Any, **kwargs: Any) -> None:
        pass


class _Layerwise:
    """Waits per layer: a program skipping the attention modules skips this."""

    layerwise_transfer = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def wait_for_layer_load(self, layer_name: str) -> None:
        self.calls.append(layer_name)


class _Undeclared:
    pass


_CLASSES = {
    "RaidenLike": _RaidenLike,
    "LayerwiseLike": _Layerwise,
    "Undeclared": _Undeclared,
}


@pytest.fixture
def fake_connectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guards_mod, "connector_class", _CLASSES.get)


def _kv(name: str, role: Any, children: list[dict[str, Any]] | None = None) -> Any:
    extra = {"connectors": [dict(c) for c in children or []]}
    return SimpleNamespace(
        kv_connector=name, kv_role=role, kv_connector_extra_config=extra
    )


def _offload_child() -> dict[str, Any]:
    return {"kv_connector": "TPURaidenOffloadingConnector"}


@pytest.mark.parametrize(
    "kv, caps, code",
    [
        # a. a consumer of a connector with no layer-wise hooks: admitted
        (_kv("RaidenLike", "kv_consumer"), _CAPS, None),
        # b. a layer-wise connector: declined
        (_kv("LayerwiseLike", "kv_consumer"), _CAPS, "LayerwiseLike"),
        # c. a producer, kv_both or no role: declined
        (_kv("RaidenLike", "kv_producer"), _CAPS, "RaidenLike"),
        (_kv("RaidenLike", "kv_both"), _CAPS, "RaidenLike"),
        (_kv("RaidenLike", None), _CAPS, "RaidenLike"),
        # d. an offloading connector alone: kv_offload's, as before
        (_kv("OffloadingConnector", "kv_both"), _CAPS, None),
        # e. a transfer child plus an offloading child on a consumer: admitted
        (
            _kv(
                "MultiConnector",
                "kv_consumer",
                [{"kv_connector": "RaidenLike"}, _offload_child()],
            ),
            _CAPS,
            None,
        ),
        (
            _kv(
                "TPUMultiConnector",
                "kv_consumer",
                [_offload_child(), {"kv_connector": "RaidenLike"}],
            ),
            _CAPS,
            None,
        ),
        # ... but not when the transfer child says it produces, or is layer-wise
        (
            _kv(
                "MultiConnector",
                "kv_consumer",
                [
                    {"kv_connector": "RaidenLike", "kv_role": "kv_producer"},
                    _offload_child(),
                ],
            ),
            _CAPS,
            "RaidenLike",
        ),
        (
            _kv(
                "MultiConnector",
                "kv_consumer",
                [{"kv_connector": "RaidenLike"}, {"kv_connector": "LayerwiseLike"}],
            ),
            _CAPS,
            "LayerwiseLike",
        ),
        # f. a plugin without the feature: declined, as before
        (_kv("RaidenLike", "kv_consumer"), _CAPS - {_CONSUMER}, "RaidenLike"),
        # g. a connector that declares nothing, or that vLLM doesn't know: declined
        (_kv("Undeclared", "kv_consumer"), _CAPS, "Undeclared"),
        (_kv("NotRegistered", "kv_consumer"), _CAPS, "NotRegistered"),
    ],
)
def test_consumer_only_kv_transfer(
    fake_connectors: None, kv: Any, caps: frozenset[str], code: str | None
) -> None:
    runner = testing.fake_runner(kv_transfer_config=kv)
    res = guards_mod.check_runner_features(runner, caps)
    if code is None:
        assert res is None
    else:
        assert res is not None and res.code == f"kv_connector_unsupported:{code}"
        assert res.message.startswith("Megakernel ")


def test_consumer_only_kv_transfer_messages(fake_connectors: None) -> None:
    def msg(kv: Any, caps: frozenset[str] = _CAPS) -> str:
        res = guards_mod.check_runner_features(
            testing.fake_runner(kv_transfer_config=kv), caps
        )
        assert res is not None
        return res.message

    assert "layer-wise" in msg(_kv("LayerwiseLike", "kv_consumer"))
    assert "kv_role 'kv_producer'" in msg(_kv("RaidenLike", "kv_producer"))
    assert "unknown connector" in msg(_kv("NotRegistered", "kv_consumer"))
    assert "does not declare KV transfer" in msg(
        _kv("RaidenLike", "kv_consumer"), _CAPS - {_CONSUMER}
    )


@pytest.mark.parametrize("name", [None, ""])
def test_unset_connector_is_no_decline_with_the_consumer_feature(
    fake_connectors: None, name: Any
) -> None:
    """A KV transfer config that names no connector puts none in effect, so
    the consumer-only checks have nothing to decline, whatever the role."""
    for role in ("kv_consumer", "kv_producer", None):
        runner = testing.fake_runner(kv_transfer_config=_kv(name, role))
        assert guards_mod.check_runner_features(runner, _CAPS) is None


def test_offloading_without_kv_offload_is_still_declined(fake_connectors: None) -> None:
    runner = testing.fake_runner(
        kv_transfer_config=_kv("OffloadingConnector", "kv_both")
    )
    res = guards_mod.check_runner_features(runner, _CAPS - {"kv_offload"})
    assert res is not None and res.code == "kv_offload_unsupported"


def test_full_kv_transfer_still_serves_any_transfer_connector(
    fake_connectors: None,
) -> None:
    """`kv_transfer` keeps its meaning: the plugin honors any transfer
    connector itself, so the consumer-only checks don't apply to it."""
    for kv in (_kv("LayerwiseLike", "kv_producer"), _kv("NotRegistered", None)):
        runner = testing.fake_runner(kv_transfer_config=kv)
        assert (
            guards_mod.check_runner_features(runner, frozenset({"kv_transfer"})) is None
        )


def test_tpu_connectors_declare_no_layerwise_transfer() -> None:
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
        TPUConnector,
        TPURaidenConnector,
    )

    assert TPUConnector.layerwise_transfer is False
    assert TPURaidenConnector.layerwise_transfer is False


def test_connector_class_resolves_the_registered_tpu_connectors() -> None:
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnector
    from vllm_torchtpu.platforms import tpu_platform

    tpu_platform._register_tpu_kv_connectors()
    assert guards_mod.connector_class("TPURaidenConnector") is TPURaidenConnector
    assert guards_mod.connector_class("NoSuchConnector") is None
