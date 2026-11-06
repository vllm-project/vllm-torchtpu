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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.experimental.export import checkpoint

_TP2 = {"data": 1, "model": 2}


def _spec(
    name: str, shape: list[int], dtype: str, sharding: list[Any]
) -> dict[str, Any]:
    return {"name": name, "shape": shape, "dtype": dtype, "sharding": sharding}


def test_shard_shape_divides_sharded_dims() -> None:
    spec = _spec("w", [8, 6], "bf16", ["model", None])
    assert checkpoint.shard_shape(spec, _TP2) == [4, 6]
    assert checkpoint.shard_shape(_spec("s", [], "i64", []), _TP2) == []


def test_local_tensors_ties_lm_head_to_the_embedding() -> None:
    model = nn.Module()
    model.register_parameter("w", nn.Parameter(torch.zeros(2)))
    model.register_buffer("b", torch.ones(1))
    tensors = checkpoint.local_tensors(model)
    assert set(tensors) == {"w", "b"}

    tied = nn.Module()
    tied.model = nn.Module()
    tied.model.embed_tokens = nn.Embedding(4, 2)
    tensors = checkpoint.local_tensors(tied)
    assert tensors["lm_head.weight"] is tensors["model.embed_tokens.weight"]


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.int64]
)
def test_to_numpy_keeps_the_bytes(dtype: torch.dtype) -> None:
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(dtype)
    array = checkpoint.to_numpy(t)
    assert array.shape == (2, 3)
    assert array.tobytes() == t.contiguous().view(torch.uint8).numpy().tobytes()


def test_checkpoint_shards_reshapes_per_device_constants() -> None:
    inputs = [
        _spec("w", [8, 6], "bf16", ["model", None]),
        _spec("start", [2], "i64", ["model"]),
        _spec("pad", [], "i64", []),
    ]
    tensors = {
        "w": torch.ones(4, 6, dtype=torch.bfloat16),
        "start": torch.tensor(75968, dtype=torch.int64),
        "pad": torch.tensor(0, dtype=torch.int64),
    }
    shards = checkpoint.checkpoint_shards(inputs, tensors, _TP2)
    assert [s.shape for s in shards] == [(4, 6), (1,), ()]
    assert shards[1].tolist() == [75968]


def test_checkpoint_shards_rejects_mismatches() -> None:
    inputs = [_spec("w", [8, 6], "bf16", ["model", None])]
    with pytest.raises(KeyError, match="not in the loaded model"):
        checkpoint.checkpoint_shards(inputs, {}, _TP2)
    with pytest.raises(ValueError, match="dtype f32"):
        checkpoint.checkpoint_shards(inputs, {"w": torch.ones(4, 6)}, _TP2)
    with pytest.raises(ValueError, match="shard shape"):
        checkpoint.checkpoint_shards(
            inputs, {"w": torch.ones(8, 6, dtype=torch.bfloat16)}, _TP2
        )


def test_abstract_params_inputs_are_in_argument_order(tmp_path: Path) -> None:
    inputs = [
        dict(_spec("b", [2], "f32", [None]), arg_index=1),
        dict(_spec("a", [2], "f32", [None]), arg_index=0),
    ]
    meta = {"functions": {"abstract_params": {"default": {"inputs": inputs}}}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    names = [t["name"] for t in checkpoint.abstract_params_inputs(str(tmp_path))]
    assert names == ["a", "b"]


def test_check_replicas_rejects_per_rank_values_declared_replicated() -> None:
    dp2_tp2 = {"data": 2, "model": 2}
    inputs = [
        _spec("norm", [4], "bf16", [None]),
        _spec("a_log", [4], "bf16", [None]),
        _spec("wq", [8, 4], "bf16", ["model", None]),
    ]
    # Processes 0..3 sit at (data, model) = (0,0), (0,1), (1,0), (1,1).
    same = [[7, 1, 10], [7, 1, 11], [7, 1, 10], [7, 1, 11]]
    checkpoint.check_replicas(inputs, same, dp2_tp2)

    per_rank = [[7, 1, 10], [7, 2, 11], [7, 3, 10], [7, 4, 11]]
    with pytest.raises(ValueError, match=r"^1 tensors .*: a_log \[None\]$"):
        checkpoint.check_replicas(inputs, per_rank, dp2_tp2)
    # A tensor sharded over `model` is copied across `data` only.
    both = [[7, 1, 10], [7, 2, 11], [7, 3, 12], [7, 4, 11]]
    two = r"^2 tensors .*: a_log \[None\], wq \['model', None\]$"
    with pytest.raises(ValueError, match=two):
        checkpoint.check_replicas(inputs, both, dp2_tp2)


@pytest.mark.parametrize(
    "name,shape",
    [
        ("language_model.model.layers.0.linear_attn.A_log", [8]),
        ("model.layers.0.self_attn.o_proj.weight_scale", [2048]),
        ("model.layers.0.linear_attn.out_proj.weight_scale", [2048]),
        ("model.layers.0.mlp.shared_expert.down_proj.weight_scale", [1, 2048]),
    ],
)
def test_per_rank_tensors_are_sharded_over_tensor_parallel_ranks(
    name: str, shape: list[int]
) -> None:
    from vllm_torchtpu.experimental.export import weight_mapper

    sharding = weight_mapper.infer_weight_sharding(name, shape, tp_size_hint=4)
    assert sharding == ["tp"] + [None] * (len(shape) - 1)


@pytest.mark.parametrize(
    "name,shape,expected",
    [
        # The MTP predictor's fc is column-parallel; FP8 requantization drops the
        # attributes that say so, so the name has to.
        ("model.fc.weight", [4096, 2048], [None, "tp"]),
        ("model.fc.weight_scale", [2048], ["tp"]),
        ("model.layers.0.mlp.fc1.weight", [4096, 2048], [None, None]),
    ],
)
def test_the_mtp_fc_is_sharded_like_a_column_parallel_layer(
    name: str, shape: list[int], expected: list[str | None]
) -> None:
    from vllm_torchtpu.experimental.export import weight_mapper

    assert weight_mapper.infer_weight_sharding(name, shape, tp_size_hint=2) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        # Muse Spark's MoE projects to and from the expert width with column- and
        # row-parallel layers whose shard attributes processing drops.
        ("model.language_model.layers.0.mlp.pre_expert_proj.weight", [None, "tp"]),
        ("model.language_model.layers.0.mlp.post_expert_proj.weight", ["tp", None]),
    ],
)
def test_muse_expert_projections_are_sharded_like_their_parallel_layers(
    name: str, expected: list[str | None]
) -> None:
    from vllm_torchtpu.experimental.export import weight_mapper

    sharding = weight_mapper.infer_weight_sharding(name, [8192, 4096], tp_size_hint=8)
    assert sharding == expected


def test_qkvz_kernel_layout_is_rejected_when_tensor_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mapper shards in_proj_qkvz as [n_out, n_in], which the flag flips."""
    from vllm_torchtpu.experimental.export import weight_mapper

    name = "model.layers.0.linear_attn.in_proj_qkvz.weight"
    monkeypatch.delenv("TPU_GDN_QKVZ_KERNEL_LAYOUT", raising=False)
    sharding = weight_mapper.infer_weight_sharding(name, [4608, 2048], tp_size_hint=4)
    assert sharding == ["tp", None]

    monkeypatch.setenv("TPU_GDN_QKVZ_KERNEL_LAYOUT", "1")
    # Nothing is sharded at TP = 1, so the layout does not matter.
    sharding = weight_mapper.infer_weight_sharding(name, [2048, 18432])
    assert sharding == [None, None]
    with pytest.raises(NotImplementedError, match="TPU_GDN_QKVZ_KERNEL_LAYOUT"):
        weight_mapper.infer_weight_sharding(name, [2048, 4608], tp_size_hint=4)


def _ocdbt(data_dir: Path) -> dict[str, Any]:
    """The TensorStore kvstore spec of the OCDBT database in `data_dir`."""
    return {"driver": "ocdbt", "base": f"file://{data_dir}"}


def _zarray(ts: Any, data_dir: Path, i: int) -> dict[str, Any]:
    kv = ts.KvStore.open(_ocdbt(data_dir)).result()
    return json.loads(kv.read(f"array_{i}/.zarray").result().value)


def _read_array(ts: Any, data_dir: Path, i: int) -> np.ndarray:
    spec = {"driver": "zarr", "kvstore": {**_ocdbt(data_dir), "path": f"array_{i}/"}}
    return np.asarray(ts.open(spec).result().read().result())


def test_save_checkpoint_writes_zarr_v2_arrays_in_an_ocdbt_database(
    tmp_path: Path,
) -> None:
    pytest.importorskip("orbax.checkpoint")
    ts = pytest.importorskip("tensorstore")
    import jax

    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        _save_and_check(tmp_path, ts)
    finally:
        jax.config.update("jax_enable_x64", x64)


def _save_and_check(tmp_path: Path, ts: Any) -> None:
    single = {"data": 1, "model": 1}
    inputs = [
        _spec("w", [2, 3], "bf16", [None, None]),
        _spec("q", [4], "f8E4M3FN", [None]),
        _spec("i", [], "i64", []),
    ]
    tensors = {
        "w": torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16),
        "q": torch.tensor([1.0, 2.0, 0.5, -4.0]).to(torch.float8_e4m3fn),
        "i": torch.tensor(151936, dtype=torch.int64),
    }
    shards = checkpoint.checkpoint_shards(inputs, tensors, single)
    data_dir = tmp_path / "serving_checkpoint" / "data"
    checkpoint.save_checkpoint(inputs, shards, single, str(data_dir))

    # The format the artifact declares (serving_config.checkpoint_config).
    assert checkpoint.CHECKPOINT_FORMAT == "zarr2_ocdbt"
    assert (data_dir / "_CHECKPOINT_METADATA").is_file()
    assert (data_dir / "manifest.ocdbt").is_file()
    assert not (data_dir / "array_0").exists(), "chunks live in the database"
    for i, (spec, shard) in enumerate(zip(inputs, shards, strict=True)):
        header = _zarray(ts, data_dir, i)
        assert header["zarr_format"] == 2 and header["shape"] == spec["shape"]
        assert header["compressor"]["id"] == "zstd"
        value = _read_array(ts, data_dir, i)
        assert value.tobytes() == shard.tobytes(), spec["name"]
    dtypes = [_zarray(ts, data_dir, i)["dtype"] for i in range(3)]
    assert dtypes == ["bfloat16", "float8_e4m3fn", "<i8"]


def test_declare_checkpoint_format_updates_only_checkpoint_config(
    tmp_path: Path,
) -> None:
    # `--export-mode checkpoint` fills an artifact exported earlier, possibly before
    # it declared a format; the artifact must then declare what was written.
    meta = {"format_version": 2, "serving_config": {"vocab_size": 32}, "functions": {}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    checkpoint.declare_checkpoint_format(str(tmp_path))
    updated = json.loads((tmp_path / "metadata.json").read_text())
    assert updated["serving_config"] == {
        "vocab_size": 32,
        "checkpoint_config": {"format": "zarr2_ocdbt"},
    }
    assert {k: v for k, v in updated.items() if k != "serving_config"} == {
        "format_version": 2,
        "functions": {},
    }


# One tensor-parallel rank of a two-process save (see the test below).
_TWO_RANK_SAVE = r"""
import sys

import numpy as np

from vllm_torchtpu.experimental.export import checkpoint

rank, coordinator, data_dir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
checkpoint.init_jax(rank, 2, coordinator)
norm = np.arange(16, dtype=np.float32).reshape(2, 8)
weight = np.arange(256, dtype=np.float32).reshape(4, 64) * 0.5
bias = np.arange(6, dtype=np.float32)
scale = np.arange(512, dtype=np.float32)
inputs = [
    {"name": "norm", "shape": [2, 8], "dtype": "f32", "sharding": ["model", None]},
    {"name": "weight", "shape": [4, 64], "dtype": "f32", "sharding": [None, "model"]},
    {"name": "bias", "shape": [6], "dtype": "f32", "sharding": [None]},
    {"name": "scale", "shape": [512], "dtype": "f32", "sharding": [None]},
]
shards = [norm[rank : rank + 1], weight[:, 32 * rank : 32 * rank + 32], bias, scale]
checkpoint.save_checkpoint(
    inputs, shards, {"data": 1, "model": 2}, data_dir, small_array_bytes=512
)
"""


def test_small_sharded_arrays_are_saved_as_one_chunk(tmp_path: Path) -> None:
    """Two ranks save small arrays as one chunk; larger ones as before.

    Orbax writes one chunk per shard and splits a replicated array among its
    replicas; both apply only above `small_array_bytes` (512 here).
    """
    pytest.importorskip("orbax.checkpoint")
    ts = pytest.importorskip("tensorstore")
    import os
    import subprocess
    import sys

    import portpicker

    data_dir = tmp_path / "serving_checkpoint" / "data"
    coordinator = f"localhost:{portpicker.pick_unused_port()}"
    env = dict(os.environ, **checkpoint.CPU_ENV)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _TWO_RANK_SAVE, str(r), coordinator, str(data_dir)],
            env=env,
        )
        for r in range(2)
    ]
    assert [p.wait(timeout=300) for p in procs] == [0, 0]

    keys = [
        k.decode() for k in ts.KvStore.open(_ocdbt(data_dir)).result().list().result()
    ]

    def chunks(i: int) -> list[str]:
        prefix = f"array_{i}/"
        return sorted(
            k[len(prefix) :]
            for k in keys
            if k.startswith(prefix) and not k[len(prefix) :].startswith(".")
        )

    def read(i: int) -> np.ndarray:
        return _read_array(ts, data_dir, i)

    # 64 bytes in all: one chunk. 1 KiB, over the limit: one chunk per shard.
    assert chunks(0) == ["0.0"]
    assert chunks(1) == ["0.0", "0.1"]
    # Replicated, 24 bytes: one chunk, not one per replica. 2 KiB: split as before.
    assert chunks(2) == ["0"]
    assert chunks(3) == ["0", "1"]
    np.testing.assert_array_equal(
        read(0), np.arange(16, dtype=np.float32).reshape(2, 8)
    )
    np.testing.assert_array_equal(
        read(1), np.arange(256, dtype=np.float32).reshape(4, 64) * 0.5
    )
    np.testing.assert_array_equal(read(2), np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(read(3), np.arange(512, dtype=np.float32))
