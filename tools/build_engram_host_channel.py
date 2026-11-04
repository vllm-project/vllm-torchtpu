# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the DeepSeek V4.1 Engram Host channel against the installed TorchTPU.

The channel uses TorchTPU's private tensor-buffer header, which is valid only
against the XLA commit named in ``torch_tpu/include/XLA_COMMIT``. Pass an
extracted source archive of exactly that commit; only the PJRT headers and the
channel are built, and the library links to the installed wheels.
"""

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import torch

from vllm_torchtpu.models.vllm.deepseek_v41.common.host_channel import (
    LIBRARY,
    SOURCE,
    build_metadata,
)

_HEADERS_BZL = """\
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
def _headers(ctx):
    return [CcInfo(compilation_context=cc_common.merge_compilation_contexts(
        compilation_contexts=[dep[CcInfo].compilation_context for dep in ctx.attr.deps],
    ))]
headers = rule(implementation=_headers, attrs={"deps": attr.label_list()})
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xla-source", type=Path, required=True)
    parser.add_argument("--bazel-root", type=Path)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()

    spec = importlib.util.find_spec("torch_tpu")
    assert spec is not None and spec.origin is not None
    tpu = Path(spec.origin).parent
    torch_root = Path(torch.__file__).parent
    metadata = build_metadata()
    xla = args.xla_source.resolve()
    if xla.name != f"xla-{metadata['xla_commit']}":
        parser.error(f"--xla-source must be xla-{metadata['xla_commit']}")

    package = xla / "engram_channel"
    package.mkdir(exist_ok=True)
    for name, target in (("torch_include", torch_root), ("tpu_include", tpu)):
        link = package / name
        link.unlink(missing_ok=True)
        link.symlink_to(target / "include", target_is_directory=True)
    shutil.copyfile(SOURCE, package / SOURCE.name)
    (package / "headers.bzl").write_text(_HEADERS_BZL)
    glue = "_".join(torch.__version__.split("+")[0].split(".")[:3])
    copts = [
        "-std=c++17",
        "-O3",
        "-fexceptions",
        "-DTORCH_TPU_USER_UNSUPPORTED_PRIVATE_HEADERS",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(metadata['cxx11_abi'])}",
    ]
    linkopts = [
        "-pthread",
        f"-L{torch_root / 'lib'}",
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
        f"-L{tpu / 'csrc/common' / ('glue_' + glue)}",
        f"-lpywrap_{glue}_common",
        f"-L{tpu / 'csrc/common'}",
        "-lxla_base",
    ]
    (package / "BUILD.bazel").write_text(f"""\
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load(":headers.bzl", "headers")
headers(name="pjrt_headers", deps=["//xla/pjrt:pjrt_client", "//xla/pjrt:raw_buffer"])
cc_library(
    name="wheel_headers",
    hdrs=glob(["torch_include/**", "tpu_include/**"]),
    includes=["torch_include", "torch_include/torch/csrc/api/include", "tpu_include"],
)
cc_binary(
    name="native.so",
    srcs=["{SOURCE.name}"],
    linkshared=True,
    copts={json.dumps(copts)},
    linkopts={json.dumps(linkopts)},
    deps=[":pjrt_headers", ":wheel_headers"],
)
""")
    bazel = ["bazel", "--nohome_rc"]
    if args.bazel_root is not None:
        bazel.append(f"--output_user_root={args.bazel_root.resolve()}")
    subprocess.run(
        [
            *bazel,
            "build",
            "//engram_channel:native.so",
            f"--jobs={args.jobs}",
            "--repo_env=USE_HERMETIC_CC_TOOLCHAIN=0",
            "--copt=-Wno-error",
        ],
        cwd=xla,
        check=True,
    )
    staging = LIBRARY.with_suffix(".tmp")
    shutil.copyfile(xla / "bazel-bin/engram_channel/native.so", staging)
    subprocess.run(["strip", "--strip-unneeded", str(staging)], check=True)
    staging.replace(LIBRARY)
    LIBRARY.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(LIBRARY)


if __name__ == "__main__":
    main()
