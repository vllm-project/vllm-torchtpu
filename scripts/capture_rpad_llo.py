"""Dump the LLO / Mosaic of the QSA RPA decode kernel (one config) for bundle study.

Sets the libtpu dump flags before importing jax, compiles and runs one
_rpa_attention_batch call at the TP8 geometry (24 rows, FP8 paired rows,
shared/lean/slot settings), then copies the RPAd program's dump files into
$CDK_OUTPUT_DIR/llo.
"""

import os
import shutil
import sys
import tarfile
from pathlib import Path

LOCAL = "/tmp/rpad_dump"
os.makedirs(f"{LOCAL}/jf", exist_ok=True)
os.makedirs(f"{LOCAL}/mosaic", exist_ok=True)
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "")
    + f" --xla_jf_dump_to={LOCAL}/jf --xla_jf_dump_llo_text=true"
    + f" --xla_mosaic_dump_to={LOCAL}/mosaic"
    + " --xla_enable_custom_call_region_trace=true"
    + " --xla_xprof_register_llo_debug_info=true"
).strip()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import functools  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

import bench_qsa_rpa_decode as b  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa  # noqa: E402


def test_capture():
    rng = np.random.default_rng(0)
    cache = b.make_cache(pages=1024)
    q, gate, blocks, lengths, table = b.make_inputs(
        rng, requests=6, rows=24, overlap=0.0, pages=1024
    )
    attend = jax.jit(
        functools.partial(
            qsa_rpa._rpa_attention_batch, kv_heads=1, ratio=b.RATIO, k_scale=0.375,
            v_scale=2.5, lean_fetch=True, decode_only=True, slot_read=True,
            kv_head_offset=1,
        ),
        donate_argnums=(5,),
    )
    out, cache = attend(q, gate, blocks, lengths, table, cache)
    jax.block_until_ready(out)
    dest = Path(os.environ.get("CDK_OUTPUT_DIR", "/tmp")) / "llo"
    with jax.profiler.trace(str(dest / "xprof")):
        for _ in range(3):
            out, cache = attend(q, gate, blocks, lengths, table, cache)
        jax.block_until_ready(out)
    dest.mkdir(parents=True, exist_ok=True)
    jf = sorted(Path(f"{LOCAL}/jf").rglob("*"))
    print("jf files", len(jf), "mosaic files", len(list(Path(f"{LOCAL}/mosaic").rglob("*"))))
    with tarfile.open(dest / "jf.tgz", "w:gz") as tar:
        tar.add(f"{LOCAL}/jf", arcname="jf")
    with tarfile.open(dest / "mosaic.tgz", "w:gz") as tar:
        tar.add(f"{LOCAL}/mosaic", arcname="mosaic")
    for p in (dest / "jf.tgz", dest / "mosaic.tgz"):
        print(p, p.stat().st_size)
    shutil.rmtree(LOCAL, ignore_errors=True)


if __name__ == "__main__":
    test_capture()
