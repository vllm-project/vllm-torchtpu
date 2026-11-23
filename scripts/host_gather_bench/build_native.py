# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build native/{buffer,channel}.cc against the installed TorchTPU's XLA commit.

Pass an extracted xla-<XLA_COMMIT> source tree. Bazel builds only the header
generators and the shim; the result links to the installed TorchTPU wheel.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import torch

SOURCES = ("buffer.cc", "channel.cc")

HEADERS_BZL = """\
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
def _headers(ctx):
    return [CcInfo(compilation_context=cc_common.merge_compilation_contexts(
        compilation_contexts=[dep[CcInfo].compilation_context for dep in ctx.attr.deps],
    ))]
headers = rule(implementation=_headers, attrs={"deps": attr.label_list()})
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xla-source", type=Path, required=True)
    parser.add_argument("--bazel-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()

    spec = importlib.util.find_spec("torch_tpu")
    assert spec is not None and spec.origin is not None, "Install TorchTPU first"
    tpu = Path(spec.origin).parent
    torch_root = Path(torch.__file__).parent
    commit = (tpu / "include/XLA_COMMIT").read_text().strip()
    xla = args.xla_source.resolve()
    if xla.name != f"xla-{commit}":
        parser.error(f"Expected the source directory xla-{commit}")

    build = xla / "host_gather_bench"
    build.mkdir(exist_ok=True)
    for name, target in (
        ("torch_include", torch_root / "include"),
        ("tpu_include", tpu / "include"),
    ):
        if not (build / name).exists():
            (build / name).symlink_to(target, target_is_directory=True)
    native = Path(__file__).resolve().parent / "native"
    source_hash = hashlib.sha256()
    for name in SOURCES:
        source_hash.update((native / name).read_bytes())
        shutil.copyfile(native / name, build / name)

    glue = "_".join(torch.__version__.split("+")[0].split(".")[:3])
    copts = [
        "-std=c++17",
        "-O3",
        "-fexceptions",
        "-DTORCH_TPU_USER_UNSUPPORTED_PRIVATE_HEADERS",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
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
    (build / "headers.bzl").write_text(HEADERS_BZL)
    (build / "BUILD.bazel").write_text(f"""\
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
    srcs={json.dumps(list(SOURCES))},
    linkshared=True,
    copts={json.dumps(copts)},
    linkopts={json.dumps(linkopts)},
    deps=[":pjrt_headers", ":wheel_headers"],
)
""")
    subprocess.run(
        [
            "bazel",
            "--nohome_rc",
            f"--output_user_root={args.bazel_root.resolve()}",
            "build",
            "//host_gather_bench:native.so",
            f"--jobs={args.jobs}",
            "--repo_env=USE_HERMETIC_CC_TOOLCHAIN=0",
            "--copt=-Wno-error",
        ],
        cwd=xla,
        check=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(xla / "bazel-bin/host_gather_bench/native.so", args.out)
    subprocess.run(["strip", "--strip-unneeded", str(args.out)], check=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_tpu_xla_commit": commit,
                "cxx11_abi": torch.compiled_with_cxx11_abi(),
                "source_sha256": source_hash.hexdigest(),
                "library_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
