load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/tokamax:license"],
    default_visibility = ["//third_party/py/tokamax/_src/ops/experimental/routing_moe:__subpackages__"],
)

pytype_strict_library(
    name = "input_pack",
    srcs = ["input_pack.py"],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/common:layout_util",
    ],
)

pytype_strict_library(
    name = "output_reduce",
    srcs = ["output_reduce.py"],
    deps = [
        "//third_party/py/jax",
    ],
)
