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
"""libtpu identification when the pip package may be absent."""

import logging
from importlib import metadata as importlib_metadata

import pytest

from vllm_torchtpu import tpu_info
from vllm_torchtpu.runner import tpu_runner


def _versions(libtpu):
    def version(name):
        if name == "libtpu" and libtpu is None:
            raise importlib_metadata.PackageNotFoundError(name)
        return libtpu if name == "libtpu" else "1"

    return version


def test_get_libtpu_version_uses_package_metadata(monkeypatch):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.50"))
    monkeypatch.delenv("TPU_LIBRARY_PATH", raising=False)

    assert tpu_info.get_libtpu_version() == "0.0.50"
    assert tpu_info.get_libtpu_cache_key() == "0.0.50"


def test_get_libtpu_version_without_package(monkeypatch):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))

    assert tpu_info.get_libtpu_version() is None


def test_libtpu_cache_key_fingerprints_tpu_library_path(monkeypatch, tmp_path):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))
    first = tmp_path / "a" / "libtpu.so"
    second = tmp_path / "b" / "libtpu.so"
    for path, payload in ((first, b"build-a" * 4096), (second, b"build-b" * 4096)):
        path.parent.mkdir()
        path.write_bytes(payload)

    monkeypatch.setenv("TPU_LIBRARY_PATH", str(first))
    key_first = tpu_info.get_libtpu_cache_key()
    assert key_first.startswith("unpackaged-")
    assert tpu_info.get_libtpu_cache_key() == key_first

    monkeypatch.setenv("TPU_LIBRARY_PATH", str(second))
    assert tpu_info.get_libtpu_cache_key() != key_first


def test_libtpu_cache_key_without_package_or_library(monkeypatch, tmp_path):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(tmp_path / "missing.so"))

    assert tpu_info.get_libtpu_cache_key() == "unpackaged-unknown"


def test_libtpu_cache_key_prefers_tpu_library_path_over_package(monkeypatch, tmp_path):
    # JAX loads TPU_LIBRARY_PATH in preference to the pip package, so the key
    # must follow the library file even when libtpu==0.0.50 is installed.
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.50"))
    library = tmp_path / "libtpu.so"
    library.write_bytes(b"build-a" * 4096)
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(library))

    key = tpu_info.get_libtpu_cache_key()
    assert key.startswith("unpackaged-")
    assert key != "0.0.50"

    library.write_bytes(b"build-b" * 8192)
    assert tpu_info.get_libtpu_cache_key() not in (key, "0.0.50")


def test_libtpu_cache_key_uses_package_version_for_its_own_library(
    monkeypatch, tmp_path
):
    # `import libtpu` exports TPU_LIBRARY_PATH as the package's own library;
    # like JAX, that path is not custom, so the package version identifies it.
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.50"))
    library = tmp_path / "libtpu.so"
    library.write_bytes(b"build-a" * 4096)
    monkeypatch.setattr(tpu_info, "_pip_libtpu_library_path", lambda: str(library))
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(library))

    assert tpu_info.get_tpu_library_path() is None
    assert tpu_info.get_libtpu_cache_key() == "0.0.50"


def test_validate_libtpu_version_skips_without_package(monkeypatch, caplog):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))
    monkeypatch.delenv("TPU_LIBRARY_PATH", raising=False)

    with caplog.at_level(logging.WARNING):
        tpu_runner._validate_libtpu_version()

    assert "libtpu package metadata not found" in caplog.text
    assert "skipping the libtpu version check" in caplog.text


def test_validate_libtpu_version_skips_package_check_with_tpu_library_path(
    monkeypatch, tmp_path, caplog
):
    # An installed package older than the minimum would raise, but JAX loads
    # the TPU_LIBRARY_PATH library instead, so the package is not checked.
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.34"))
    library = tmp_path / "libtpu.so"
    library.write_bytes(b"build-a" * 4096)
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(library))

    with caplog.at_level(logging.WARNING):
        tpu_runner._validate_libtpu_version()

    assert f"TPU_LIBRARY_PATH={library}" in caplog.text
    assert "skipping the libtpu package version check" in caplog.text


def test_validate_libtpu_version_still_rejects_old_package(monkeypatch):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.34"))
    monkeypatch.delenv("TPU_LIBRARY_PATH", raising=False)

    with pytest.raises(RuntimeError, match="libtpu < 0.0.35"):
        tpu_runner._validate_libtpu_version()


def test_get_node_worker_id_prefers_tpu_worker_id_then_lws_worker_index(monkeypatch):
    monkeypatch.setattr(
        tpu_info,
        "get_tpu_metadata",
        lambda key="": pytest.fail("GCE metadata should not be queried"),
    )
    monkeypatch.setenv("TPU_WORKER_ID", "2")
    monkeypatch.setenv("LWS_WORKER_INDEX", "1")
    assert tpu_info.get_node_worker_id() == 2

    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    assert tpu_info.get_node_worker_id() == 1


def test_get_node_worker_id_falls_back_to_gce_metadata_and_zero(monkeypatch):
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    monkeypatch.delenv("LWS_WORKER_INDEX", raising=False)
    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": "3")
    assert tpu_info.get_node_worker_id() == 3

    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": None)
    assert tpu_info.get_node_worker_id() == 0


def test_get_num_chips_filters_non_tpu_vfio_and_caps_at_max_per_host(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(tpu_info.glob, "glob", lambda pattern: [])
    # Simulate 10 Google TPU VFIO groups (0..9) and 6 non-TPU VFIO groups (10..15).
    monkeypatch.setattr(
        tpu_info.os,
        "listdir",
        lambda path: [str(i) for i in range(16)] + ["vfio"]
        if path == "/dev/vfio"
        else ["0000:00:04.0"],
    )
    monkeypatch.setattr(
        tpu_info.os.path,
        "exists",
        lambda path: path == "/sys/kernel/iommu_groups",
    )

    def _fake_open(path, *args, **kwargs):
        import io

        group = int(path.split("/sys/kernel/iommu_groups/")[1].split("/")[0])
        vendor = "0x1ae0\n" if group < 10 else "0x15b3\n"
        return io.StringIO(vendor)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert tpu_info.get_num_chips() == 8

    # When only 4 VFIO groups are Google TPUs, return 4.
    def _fake_open_4(path, *args, **kwargs):
        import io

        group = int(path.split("/sys/kernel/iommu_groups/")[1].split("/")[0])
        vendor = "0x1ae0\n" if group < 4 else "0x10de\n"
        return io.StringIO(vendor)

    monkeypatch.setattr("builtins.open", _fake_open_4)
    assert tpu_info.get_num_chips() == 4


def test_sanitize_lws_multihost_tpu_env(monkeypatch):
    from vllm_torchtpu import env_override

    for key in env_override._LWS_SINGLE_HOST_DEVICE_PLUGIN_ENV_VARS:
        monkeypatch.setenv(key, "stale-value")
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    monkeypatch.setenv("LWS_WORKER_INDEX", "1")
    monkeypatch.setenv("LWS_GROUP_SIZE", "2")
    monkeypatch.setenv(
        "LWS_LEADER_ADDRESS", "ubench-lws-0.ubench-lws.default.svc.cluster.local"
    )

    env_override._sanitize_lws_multihost_tpu_env()

    assert tpu_info.os.environ.get("TPU_WORKER_ID") == "1"
    for key in env_override._LWS_SINGLE_HOST_DEVICE_PLUGIN_ENV_VARS:
        assert key not in tpu_info.os.environ
