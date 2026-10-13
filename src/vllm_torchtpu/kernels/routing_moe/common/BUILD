load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/py/tokamax:license"],
    default_visibility = ["//third_party/py/tokamax/_src/ops/experimental/routing_moe:__subpackages__"],
)

pytype_strict_library(
    name = "layout_util",
    srcs = ["layout_util.py"],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:hijax",
        "//third_party/py/jax/experimental:pallas_tpu",
    ],
)
