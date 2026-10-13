load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/tokamax:license"],
    default_visibility = ["//third_party/py/tokamax/_src/ops/experimental/routing_moe:__subpackages__"],
)

pytype_strict_library(
    name = "sharding_util",
    srcs = ["sharding_util.py"],
)

pytype_strict_library(
    name = "wrapper",
    srcs = ["wrapper.py"],
    deps = [
        ":sharding_util",
        "//third_party/py/jax",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/gmm:kernel",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/io_relayout:input_pack",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/io_relayout:output_reduce",
    ],
)
