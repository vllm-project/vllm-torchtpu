load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/tokamax:license"],
    default_visibility = ["//third_party/py/tokamax/_src/ops/experimental/routing_moe:__subpackages__"],
)

pytype_strict_library(
    name = "compute",
    srcs = ["compute.py"],
    deps = [
        ":config",
        "//third_party/py/jax",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/common:layout_util",
    ],
)

pytype_strict_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:pallas",
        "//third_party/py/jax/experimental:pallas_tpu",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/common:layout_util",
    ],
)

pytype_strict_library(
    name = "memory_ref",
    srcs = ["memory_ref.py"],
    deps = [
        ":config",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:hijax",
        "//third_party/py/jax/experimental:pallas",
        "//third_party/py/jax/experimental:pallas_tpu",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/common:layout_util",
    ],
)

pytype_strict_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        ":compute",
        ":config",
        ":memory_ref",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:hijax",
        "//third_party/py/jax/experimental:pallas",
        "//third_party/py/jax/experimental:pallas_tpu",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/common:layout_util",
    ],
)

pytype_strict_library(
    name = "kernel",
    srcs = ["kernel.py"],
    deps = [
        ":config",
        ":memory_ref",
        ":pipeline",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:pallas",
        "//third_party/py/jax/experimental:pallas_tpu",
    ],
)
