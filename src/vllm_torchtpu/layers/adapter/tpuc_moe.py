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
"""TPU-C MoE backend (`TPUC_MOE=1`): Ghostfish bare-metal MoE expert FFN.

Replaces the expert computation of the plain `fused_moe_gmm` path (FP8 MoE,
collectives done by vLLM's MoE runner) with `fused_moe_tpuc`: XLA routing /
dispatch glue plus the TPU-C kernel `moe_fused_expert_ffn_gf[_h2048]`, launched
as a raw PJRT executable by the `deepsea_torch_bridge` C++ adapter.

The kernel binaries, the TPU-C serializer and the adapter sources live in the
TPU-C bundle (default `<repo>/tpuc_bundle`, override with `TPUC_BUNDLE_DIR`):

  lib/libtpuc_serializer.so      TPU-C assembly -> PJRT executable serializer
  lib/tpuc_serialize_cli.py      runs the serializer out of process
  lib/deepsea_torch_bridge.so    the adapter (built by `build_adapter.sh`)
  src/deepsea_torch_bridge_gxx.cc, src/pjrt_c_api.h   adapter sources
  runfiles/google3/torchtpu_tpuc_kernels/moe/*.s      kernel assembly
  plugins/fused_moe_tpuc_layer.py                     dispatch/combine glue

Supported MoE dims (hidden, moe_intermediate): (4096, 1024) and (2048, 512),
FP8 per-channel weights (the default MoE requantization), <= 80 local experts.

The op is not traceable (it synchronizes buffers and launches a PJRT
executable from C++), so it is registered as the custom op
`tpuc_moe::fused_moe` with a fake implementation and listed in vLLM's
`splitting_ops`: compiled pieces run before and after it and the op itself runs
eagerly between them.
"""

from __future__ import annotations

import ctypes
import glob
import importlib
import os
import subprocess
import sys
import threading
from pathlib import Path

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SPLITTING_OP = "tpuc_moe::fused_moe"
_SUPPORTED_DIMS = ((4096, 1024), (2048, 512))
_MAX_LOCAL_EXPERTS = 80

_lock = threading.Lock()
_loaded = False
_glue = None


def bundle_dir() -> Path:
    env = os.environ.get("TPUC_BUNDLE_DIR")
    if env:
        return Path(env)
    # src/vllm_torchtpu/layers/adapter/tpuc_moe.py -> <repo>/tpuc_bundle
    return Path(__file__).resolve().parents[4] / "tpuc_bundle"


def _build_adapter(bdir: Path) -> None:
    script = bdir / "build_adapter.sh"
    logger.info("[TPUC_MOE] building the deepsea_torch_bridge adapter: %s", script)
    subprocess.run(["bash", str(script)], check=True)


def _load_global(path: str) -> None:
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


def ensure_loaded():
    """Loads the adapter (registers `torch.ops.deepsea_moe.*`) and the glue."""
    global _loaded, _glue
    if _loaded:
        return _glue
    with _lock:
        if _loaded:
            return _glue
        bdir = bundle_dir()
        if not bdir.is_dir():
            raise RuntimeError(
                f"TPUC_MOE=1 but the TPU-C bundle is missing: {bdir} "
                "(set TPUC_BUNDLE_DIR)"
            )
        os.environ.setdefault("TPUC_BUNDLE_DIR", str(bdir))
        os.environ.setdefault("TEST_SRCDIR", str(bdir / "runfiles"))
        os.environ.setdefault("TPUC_SERIALIZE_PYTHON", sys.executable)
        os.environ.setdefault("TPU_MOE_SYNC_EXECUTE", "1")
        lib = bdir / "lib"
        src = bdir / "src" / "deepsea_torch_bridge_gxx.cc"
        so = lib / "deepsea_torch_bridge.so"
        if not so.exists() or (src.exists() and src.stat().st_mtime > so.stat().st_mtime):
            _build_adapter(bdir)

        # The adapter resolves c10 / libtorch / torch_tpu symbols at load time.
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        for name in ("libc10.so", "libtorch_cpu.so", "libtorch_python.so"):
            _load_global(os.path.join(torch_lib, name))
        import torch_tpu  # noqa: F401

        tpu_dir = os.path.dirname(torch_tpu.__file__)
        for path in (
            glob.glob(os.path.join(tpu_dir, "csrc", "common", "*.so"))
            + glob.glob(os.path.join(tpu_dir, "csrc", "common", "glue_*", "*.so"))
            + glob.glob(os.path.join(tpu_dir, "_C*.so"))
            + glob.glob(os.path.join(tpu_dir, "*.so"))
            + glob.glob(os.path.join(tpu_dir, "lib", "*.so"))
        ):
            if "libtpu" not in os.path.basename(path):
                _load_global(path)

        for path in (str(lib), str(bdir / "plugins")):
            if path not in sys.path:
                sys.path.insert(0, path)
        importlib.import_module("deepsea_torch_bridge")
        if not hasattr(torch.ops.deepsea_moe, "expert_ffn_out"):
            raise RuntimeError("deepsea_torch_bridge did not register deepsea_moe ops")
        _glue = importlib.import_module("fused_moe_tpuc_layer")
        _loaded = True
        logger.info("[TPUC_MOE] TPU-C MoE adapter loaded from %s", bdir)
        return _glue


def check_supported(w13: torch.Tensor, w2: torch.Tensor) -> None:
    n_exp, hidden, two_inter = (int(d) for d in w13.shape)
    inter = int(w2.shape[1])
    if (hidden, inter) not in _SUPPORTED_DIMS or two_inter != 2 * inter:
        raise ValueError(
            f"TPUC_MOE: unsupported MoE dims hidden={hidden}, intermediate={inter}"
            f" (w13={tuple(w13.shape)}, w2={tuple(w2.shape)}); supported:"
            f" {_SUPPORTED_DIMS}"
        )
    if not 1 <= n_exp <= _MAX_LOCAL_EXPERTS:
        raise ValueError(
            f"TPUC_MOE: {n_exp} local experts; the kernel supports <="
            f" {_MAX_LOCAL_EXPERTS} (use more expert parallelism)"
        )
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"TPUC_MOE: needs float8_e4m3fn MoE weights, got {w13.dtype}/{w2.dtype}"
        )


def prepare_layer(layer: torch.nn.Module) -> None:
    """Validates the layer and stores the kernel's broadcast weight scales.

    `s13_b: f32[E, 8, 2I]` / `s2_b: f32[E, 8, H]` are the per-channel FP8
    scales broadcast over 8 sublanes, registered as non-persistent buffers so
    compiled pieces see them as ordinary module state.
    """
    w13, w2 = layer.w13_weight, layer.w2_weight
    check_supported(w13, w2)
    s13 = getattr(layer, "w13_weight_scale_inv", None)
    s2 = getattr(layer, "w2_weight_scale_inv", None)
    if s13 is None or s2 is None:
        raise ValueError("TPUC_MOE: MoE layer has no per-channel weight scales")
    n_exp, hidden = int(w13.shape[0]), int(w13.shape[1])
    inter = int(w2.shape[1])
    if s13.numel() != n_exp * 2 * inter or s2.numel() != n_exp * hidden:
        raise ValueError(
            "TPUC_MOE: needs per-channel MoE scales (unset MOE_REQUANTIZE_BLOCK_SIZE);"
            f" got w13 scale {tuple(s13.shape)}, w2 scale {tuple(s2.shape)}"
        )
    with torch.no_grad():
        s13_b = (
            s13.reshape(n_exp, 1, 2 * inter).to(torch.float32)
            .expand(n_exp, 8, 2 * inter).contiguous()
        )
        s2_b = (
            s2.reshape(n_exp, 1, hidden).to(torch.float32)
            .expand(n_exp, 8, hidden).contiguous()
        )
    layer.register_buffer("_tpuc_s13_b", s13_b, persistent=False)
    layer.register_buffer("_tpuc_s2_b", s2_b, persistent=False)


_verify_calls = 0


def _verify(hidden_states, y, w13, w2, w13_scale, w2_scale, topk_weights,
            topk_ids, experts_start, top_k):
    """TPUC_MOE_VERIFY: compares `y` with `fused_moe_gmm` on the same inputs."""
    from vllm_torchtpu.layers.adapter.fused_moe import fused_moe_gmm

    ref = fused_moe_gmm(
        hidden_states, w13, w2, w13_scale, w2_scale, None, None, topk_weights,
        topk_ids, experts_start, top_k, "silu",
    )
    a = y.float().cpu()
    b = ref.float().cpu()
    x = hidden_states.float().cpu()
    # Rows of real tokens: finite and non-zero activations (bucket padding rows
    # can hold NaN / zeros).
    good = torch.isfinite(x).all(-1) & (x != 0).any(-1)
    ag, bg = a[good], b[good]
    cos = torch.nn.functional.cosine_similarity(ag.flatten(), bg.flatten(), dim=0)
    rel = (ag - bg).norm() / (bg.norm() + 1e-12)
    ids = topk_ids.cpu()
    logger.info(
        "[TPUC_MOE_VERIFY] T=%d good_rows=%d topk=%d es=%d cos=%.6f rel_l2=%.4f "
        "|y|=%.4g |ref|=%.4g nan_on_good(y/ref)=%d/%d nan_rows(x)=%d "
        "ids[min,max]=[%d,%d] w=%s/%s",
        x.shape[0], int(good.sum()), top_k, int(experts_start.cpu()), float(cos),
        float(rel), float(ag.norm()), float(bg.norm()), int(ag.isnan().sum()),
        int(bg.isnan().sum()), int((~torch.isfinite(x).all(-1)).sum()),
        int(ids.min()), int(ids.max()), str(topk_weights.dtype), str(topk_ids.dtype),
    )
    return ref


@torch.library.custom_op(SPLITTING_OP, mutates_args=())
def fused_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    s13_b: torch.Tensor,
    s2_b: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    global _verify_calls
    glue = ensure_loaded()
    y = glue.fused_moe_tpuc_prepared(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        s13_b=s13_b,
        s2_b=s2_b,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        experts_start=experts_start,
        top_k=top_k,
    )
    # Debug: TPUC_MOE_VERIFY=1 logs a comparison with fused_moe_gmm for the
    # first TPUC_MOE_VERIFY_CALLS calls; =ref also returns the GMM output.
    mode = os.environ.get("TPUC_MOE_VERIFY", "")
    if (
        mode
        and _verify_calls < int(os.environ.get("TPUC_MOE_VERIFY_CALLS", "200"))
        # Skip dummy/precompile runs (no finite, non-zero activation row).
        and bool(
            (torch.isfinite(hidden_states).all(-1) & (hidden_states != 0).any(-1))
            .any()
            .cpu()
        )
    ):
        _verify_calls += 1
        ref = _verify(hidden_states, y, w13, w2, w13_scale, w2_scale,
                      topk_weights, topk_ids, experts_start, top_k)
        if mode == "ref":
            return ref
    return y


@fused_moe.register_fake
def _(hidden_states, w13, w2, s13_b, s2_b, w13_scale, w2_scale, topk_weights,
      topk_ids, experts_start, top_k):
    return torch.empty_like(hidden_states)


def fused_moe_tpuc(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Partial MoE output of this EP shard (the runner reduces it)."""
    experts_start = getattr(layer, "_experts_start", None)
    if experts_start is None:
        experts_start = torch.zeros((), dtype=torch.int32, device=hidden_states.device)
    return torch.ops.tpuc_moe.fused_moe(
        hidden_states,
        layer.w13_weight,
        layer.w2_weight,
        layer._tpuc_s13_b,
        layer._tpuc_s2_b,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        topk_weights,
        topk_ids,
        experts_start,
        int(top_k),
    )
