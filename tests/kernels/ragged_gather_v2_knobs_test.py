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
"""ragged_gather_v2 diagnosis knobs (row_subchunks, col_size, copy_mode), on CPU.

The SparseCore kernel does not run in Pallas interpret mode, but it lowers for
TPU 7x on any host. These tests compare lowered TPU programs (source locations
excluded): equal programs compile to the same device code, so they produce
the same bits.
"""

import base64
import json
import re

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.sparse_core import ragged_gather_v2 as rg

SERVED = (6400, 2560, jnp.bfloat16, 64000)  # one 6,400-token step, top-10


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    from jax._src import tpu_info as ti
    from jax._src.pallas.mosaic import tpu_info as pti
    from jax.experimental.pallas import tpu as pltpu

    info = ti.get_tpu_info_for_chip(ti.ChipVersion.TPU_7X, 1)
    for mod in (pltpu, ti, pti):
        monkeypatch.setattr(mod, "get_tpu_info", lambda: info, raising=False)
    for mod in (ti, pti):
        monkeypatch.setattr(mod, "is_tpu_device", lambda *a, **k: True, raising=False)
    limit = jax.config.jax_traceback_in_locations_limit
    jax.config.update("jax_traceback_in_locations_limit", 0)
    yield
    jax.config.update("jax_traceback_in_locations_limit", limit)


def _lower(shape=SERVED, trim_rows=False, **kw):
    rows, hidden, dtype, n = shape
    args = (
        jax.ShapeDtypeStruct((rows, hidden), dtype),
        jax.ShapeDtypeStruct((n,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    f = jax.jit(
        lambda x, i, s, e: rg.ragged_gather_v2(x, i, s, e, trim_rows=trim_rows, **kw)
    )
    return f.trace(*args).lower(lowering_platforms=("tpu",)).as_text()


def _canonical(text):
    """The module with each serialized Mosaic kernel replaced by its
    location-free assembly and the gather kernel name replaced by KERNEL."""
    from jax._src.lib import tpu
    from jax._src.lib.mlir import ir

    def body_asm(m):
        raw = re.sub(r"\\([0-9A-Fa-f]{2})", lambda g: chr(int(g.group(1), 16)), m.group(1))
        body = base64.b64decode(json.loads(raw)["custom_call_config"]["body"])
        ctx = ir.Context()
        tpu.register_dialect(ctx)
        ctx.allow_unregistered_dialects = True
        with ctx:
            return "<<" + ir.Module.parse(body).operation.get_asm(enable_debug_info=False) + ">>"

    text = re.sub(r'backend_config = "((?:[^"\\]|\\.)*)"', body_asm, text)
    return re.sub(r"sc_ragged_gather_v2(?:-[a-z]+_?[0-9]*)*", "KERNEL", text)


def _kernel_names(text):
    return set(re.findall(r'kernel_name = "([^"]+)"', text))


def test_kernel_name_names_every_set_knob():
    assert rg.gather_kernel_name() == "sc_ragged_gather_v2"
    assert rg.gather_kernel_name(row_subchunks=8) == "sc_ragged_gather_v2-rs_8"
    assert rg.gather_kernel_name(col_size=640) == "sc_ragged_gather_v2-cs_640"
    assert rg.gather_kernel_name(copy_mode="dma") == "sc_ragged_gather_v2-dma"
    assert (
        rg.gather_kernel_name(row_subchunks=4, col_size=256, copy_mode="dma")
        == "sc_ragged_gather_v2-rs_4-cs_256-dma"
    )


def test_knobs_off_lowers_under_the_served_name():
    assert _kernel_names(_lower()) == {"sc_ragged_gather_v2"}


# sha256 of the served-shape lowered program (locations excluded) of the kernel
# before the knobs existed (60014b5e2), under jax/jaxlib 0.11.2. The knobs-off
# kernel must lower to exactly these bytes.
_PRE_KNOB_SHA256 = {
    (SERVED, False): "8790fbedc453412a",
    ((3200, 2560, jnp.bfloat16, 32000), False): "4534471715c0",
    ((6400, 1280, jnp.float32, 64000), False): "ba64803ac171",
}


@pytest.mark.skipif(
    (jax.__version__, jax.lib.__version__) != ("0.11.2", "0.11.2"),
    reason="lowered bytes are pinned for jax/jaxlib 0.11.2",
)
@pytest.mark.parametrize("key", list(_PRE_KNOB_SHA256))
def test_knobs_off_lowers_to_the_pre_knob_program(key):
    import hashlib

    shape, trim_rows = key
    got = hashlib.sha256(_lower(shape, trim_rows=trim_rows).encode()).hexdigest()
    assert got.startswith(_PRE_KNOB_SHA256[key])


@pytest.mark.parametrize(
    "shape", [SERVED, (3200, 2560, jnp.bfloat16, 32000), (512, 256, jnp.float32, 1000)]
)
def test_explicit_default_knobs_lower_to_the_same_kernel(shape):
    """The values the default path picks, passed as knobs, give the same
    program up to the kernel name: the plumbing alone changes nothing."""
    rows, hidden, dtype, n = shape
    packing = 32 // jnp.dtype(dtype).itemsize // 8
    cs = rg.calculate_col_size(hidden, packing)
    rs = max(1, min(4, -(-n // (16 * 32))))
    off = _lower(shape)
    explicit = _lower(shape, row_subchunks=rs, col_size=cs, copy_mode="vector")
    assert _kernel_names(explicit) == {rg.gather_kernel_name(rs, cs)}
    assert explicit != off
    assert _canonical(explicit) == _canonical(off)


@pytest.mark.parametrize(
    "kw",
    [
        dict(row_subchunks=1),
        dict(row_subchunks=2),
        dict(row_subchunks=8),
        dict(col_size=640),
        dict(col_size=256),
        dict(copy_mode="dma"),
        dict(copy_mode="dma", row_subchunks=8),
    ],
)
def test_each_knob_reaches_the_kernel(kw):
    text = _lower(**kw)
    assert _kernel_names(text) == {rg.gather_kernel_name(**kw)}
    assert _canonical(text) != _canonical(_lower())


def test_dma_on_the_column_packed_view_reaches_the_kernel():
    """f32 (packing 1): the dma output has the vector path's shape, so only the
    kernel body can tell the two apart."""
    shape = (6400, 1280, jnp.float32, 64000)
    assert _canonical(_lower(shape, copy_mode="dma")) != _canonical(_lower(shape))


@pytest.mark.parametrize("trim_rows", [False, True])
@pytest.mark.parametrize(
    "dtype,packing", [(jnp.bfloat16, 2), (jnp.float32, 1), (jnp.int8, 4)]
)
def test_dma_returns_the_whole_gathered_32bit_rows(dtype, packing, trim_rows):
    n, hidden = 64000, 1280
    padded = 65536  # 32 subcores x 4 subchunks x 16 rows per block
    se = jax.ShapeDtypeStruct((1,), jnp.int32)
    x = jax.ShapeDtypeStruct((6400, hidden), dtype)
    idx = jax.ShapeDtypeStruct((n,), jnp.int32)

    def shape(**kw):
        return jax.eval_shape(
            lambda x_, i_, s_, e_: rg.ragged_gather_v2(x_, i_, s_, e_, trim_rows=trim_rows, **kw),
            x, idx, se, se,
        ).shape

    rows = n if trim_rows else padded
    assert shape() == (rows, hidden)
    assert shape(copy_mode="dma") == (packing * rows, hidden)


@pytest.mark.parametrize(
    "kw,msg",
    [
        (dict(copy_mode="bogus"), "copy_mode"),
        (dict(row_subchunks=0), "row_subchunks"),
        (dict(row_subchunks=-2), "row_subchunks"),
        (dict(row_subchunks=True), "row_subchunks"),
        (dict(row_subchunks=2.0), "row_subchunks"),
        (dict(col_size=100), "multiple of 128"),
        (dict(col_size=320), "multiple of 128"),  # a quarter of 1280; divides 2560
        (dict(col_size=0), "multiple of 128"),
        (dict(col_size=True), "multiple of 128"),
        (dict(col_size=384), "divide"),
        (dict(col_size=2560), "VMEM"),
    ],
)
def test_bad_knobs_are_rejected(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _lower(**kw)
