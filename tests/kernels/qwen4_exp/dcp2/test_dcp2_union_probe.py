# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_UNION_PROBE (debug, log-only; models/vllm/qwen4_exp/union_probe.py).

CPU: the host reduction (owned union / visible per tile) against the owned
group space of ``dcp_local_selection``; ``poll``'s sampling and logging; and
the DCP prefill jax program exported (``jax.export``, the path a jax_op takes)
with the probe on, with a positive control: a host callback fails that export
("serialization of host_callbacks is not yet implemented", the probe v1 bug).
TPU (``python3 test_dcp2_union_probe.py`` in the image): the stash under
``torch.compile(backend="tpu")`` through the real ``update_rows_`` jax_op, then
``poll`` outside the compiled step, printing the probe's log line.
"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

B, RATIO = 512, 4


def _probe():
    """models.vllm.qwen4_exp.union_probe (by file when vllm itself is not installed, e.g. this VM)."""
    try:
        from vllm_torchtpu.models.vllm.qwen4_exp import union_probe
    except ImportError:
        import importlib.util

        import vllm_torchtpu

        name = "vllm_torchtpu.models.vllm.qwen4_exp.union_probe"
        if name in sys.modules:
            return sys.modules[name]
        path = os.path.join(os.path.dirname(vllm_torchtpu.__file__), "models", "vllm", "qwen4_exp", "union_probe.py")
        spec = importlib.util.spec_from_file_location(name, path)
        union_probe = importlib.util.module_from_spec(spec)
        sys.modules[name] = union_probe
        spec.loader.exec_module(union_probe)
    return union_probe


def _case(seed=0, rows=1024, budget=64):
    from dcp2_harness import random_blocks, selected_token_ids

    rng = np.random.default_rng(seed)
    lengths = np.concatenate([np.arange(1, 200), rng.integers(1, 9000, rows - 199)]).astype(np.int32)
    blocks = random_blocks(rng, lengths, RATIO, budget)
    sel = selected_token_ids(blocks, lengths, RATIO).astype(np.int32)
    return blocks, lengths, sel


def test_tile_stats_match_owned_group_space():
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_dense import dcp_local_selection
    union_probe = _probe()

    blocks, lengths, sel = _case()
    lb, ll = map(np.asarray, dcp_local_selection(jnp.asarray(blocks), jnp.asarray(lengths),
                                                 rank=jnp.asarray([0], jnp.int32), world=2, page_size=B,
                                                 ratio=RATIO))
    for tile in (128, 256):
        st = union_probe.tile_stats(sel, lengths - 1, B, tile=tile)
        assert len(st) == -(-len(lengths) // tile)
        for t in range(len(st)):
            rows = slice(t * tile, (t + 1) * tile)
            ids = set(lb[rows][lb[rows] >= 0].tolist()) | {int(x) // RATIO for x in ll[rows] if x % RATIO}
            visible = int(((ll[rows] + RATIO - 1) // RATIO).max())
            assert st[t, 0] == len(ids), (tile, t, st[t, 0], len(ids))
            assert st[t, 1] == visible, (tile, t, st[t, 1], visible)
    assert "owned_union/visible" in union_probe.summarize(sel, lengths - 1, B)


def test_poll_logs_every_nth_long_prefill(monkeypatch):
    import torch

    union_probe = _probe()

    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_PROBE", "2")
    _, lengths, sel = _case(1)
    layer = torch.nn.Module()
    layer._union_probe_selected = torch.from_numpy(sel)
    layer._union_probe_positions = torch.from_numpy((lengths - 1).reshape(-1, 1))
    logged = []
    monkeypatch.setattr(union_probe, "_PROBES", [("model.layers.3.self_attn", layer)])
    monkeypatch.setattr(union_probe, "_STATE", {"long_steps": 0, "pending": False, "page_size": B})
    monkeypatch.setattr(union_probe, "logger", types.SimpleNamespace(info=lambda msg, *a: logged.append(msg % a)))
    for tokens in (8192, 64, 8192, 8192, 100, 8192, 0):
        union_probe.poll(tokens)
    # Long steps 1 and 3 are sampled (every 2nd); each is logged at the next poll.
    assert [m.split(":")[0] for m in logged] == [
        "QSA DCP union probe long-prefill 1 model.layers.3.self_attn",
        "QSA DCP union probe long-prefill 3 model.layers.3.self_attn",
    ], logged


def _patch_tpu_info():
    import jax
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    if jax.default_backend() == "tpu":
        return
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    pltpu.get_tpu_info = lambda: info
    # Only public symbols plus the registry when it exists: the private
    # _get_tpu_info_impl is gone in newer JAX (the served image's).
    registry = getattr(tpu_info, "registry", None)
    if registry is not None:
        registry[jax.devices()[0].device_kind] = lambda: info
    if hasattr(tpu_info, "get_tpu_info"):
        tpu_info.get_tpu_info = lambda: info


def export_dcp_prefill():
    """jax.export of the DCP prefill route (union walk on) for TPU; returns the Exported."""
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

    rows, cols = 1024, 41
    args = (jax.ShapeDtypeStruct((rows, 6, 256), jnp.bfloat16), jax.ShapeDtypeStruct((rows, 512), jnp.int32),
            jax.ShapeDtypeStruct((rows,), jnp.int32), jax.ShapeDtypeStruct((rows, cols), jnp.int32),
            jax.ShapeDtypeStruct((90, B, 1, 4, 256), jnp.float8_e4m3fn), jax.ShapeDtypeStruct((2,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32))

    def f(q, b, lengths, t, c, s, r):
        return qsa_rpa._prefill_attention(q, q, b, lengths, t, c, s, kv_heads=1, ratio=RATIO, k_scale=0.375,
                                          v_scale=2.5, dcp=(r, 2), return_lse=True)

    return jax.export.export(jax.jit(f), platforms=["tpu"],
                             disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")])(*args)


def test_dcp_prefill_exports_with_probe_on(monkeypatch):
    import jax
    import jax.numpy as jnp

    for k, v in {"VLLM_TPU_QSA_PREFILL_UNION_PROBE": "4", "VLLM_TPU_QSA_PREFILL_UNION_WALK": "128",
                 "VLLM_TPU_QSA_PREFILL_UNION_DCP": "1"}.items():
        monkeypatch.setenv(k, v)
    _patch_tpu_info()
    exported = export_dcp_prefill()
    assert len(exported.mlir_module_serialized) > 0
    # Positive control: a host callback cannot be exported (probe v1's failure).
    def g(x):
        jax.debug.callback(lambda v: None, x)
        return x + 1

    try:
        jax.export.export(jax.jit(g), platforms=[jax.default_backend()])(jax.ShapeDtypeStruct((4,), jnp.float32))
    except NotImplementedError as e:
        assert "host_callbacks" in str(e), e
    else:
        raise AssertionError("a host callback exported: the positive control is broken")


def tpu_compiled_stash():
    """On TPU: the stash inside torch.compile(backend="tpu") (real update_rows_ jax_op), then poll."""
    import torch

    try:
        import torch_tpu  # noqa: F401
    except ImportError:
        pass
    union_probe = _probe()

    os.environ["VLLM_TPU_QSA_PREFILL_UNION_PROBE"] = "1"
    _, lengths, sel = _case(2, rows=4096, budget=256)
    layer = torch.nn.Module()
    layer.prefix = "model.layers.3.self_attn"
    layer.register_buffer("_union_probe_selected", torch.full((8192, sel.shape[1]), -1, dtype=torch.int32,
                                                              device="tpu"), persistent=False)
    layer.register_buffer("_union_probe_positions", torch.full((8192, 1), -1, dtype=torch.int32, device="tpu"),
                          persistent=False)
    union_probe._PROBES[:] = [(layer.prefix, layer)]
    union_probe.set_page_size(B)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def step(selected, positions):
        union_probe.stash(layer, selected, positions)
        return selected.sum()

    out = step(torch.from_numpy(sel).to("tpu"), torch.from_numpy(lengths - 1).to("tpu"))
    print("compiled step ran", int(out.cpu()))
    logged = []
    union_probe.logger = types.SimpleNamespace(info=lambda msg, *a: (logged.append(msg % a), print("LOG", msg % a)))
    union_probe.poll(4096)  # this step was a long prefill: sample it
    union_probe.poll(1)  # next step: read the buffers outside the compiled step, log
    assert logged and "owned_union/visible" in logged[0], logged
    expected = union_probe.summarize(sel, lengths - 1, B)
    assert logged[0].endswith(expected), (logged[0][-200:], expected[-200:])
    print("PASS tpu compiled stash")


if __name__ == "__main__":
    import types

    class _MP:
        def setenv(self, k, v):
            os.environ[k] = v

        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    rc = 0
    for name, fn in (("tile_stats", test_tile_stats_match_owned_group_space),
                     ("export", lambda: test_dcp_prefill_exports_with_probe_on(_MP())),
                     ("tpu_compiled_stash", tpu_compiled_stash)):
        try:
            fn()
            print("PASS", name, flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print("FAIL", name, repr(e)[:2000], flush=True)
            rc = 1
    print("PROBE_RC", rc)
    sys.exit(rc)
