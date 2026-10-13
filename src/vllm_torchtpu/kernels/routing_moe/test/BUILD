load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_applicable_licenses = ["//third_party/py/tokamax:license"],
)

pytype_strict_library(
    name = "reference",
    srcs = ["reference.py"],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/tokamax/_src/ops/experimental/gmm_v2",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/layer:sharding_util",
        "//third_party/py/tokamax/_src/ops/ragged_gather:pallas_mosaic_v2_tpu_kernel",
        "//third_party/py/tokamax/_src/ops/ragged_gather_reduce:pallas_mosaic_tpu_kernel",
    ],
)

py_test(
    name = "wrapper_test",
    srcs = ["wrapper_test.py"],
    args = [
        "--xla_xprof_register_llo_debug_info=true",
        "--xla_mosaic_dump_to=sponge",
        # See http://google3/third_party/py/jax/_src/tpu_info.py;l=276;rcl=981960086
        "--xla_tpu_enable_large_2nd_minor_layout_for_x16=true",
    ],
    shard_count = 50,
    tags = [
        "notap",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
    deps = [
        ":reference",
        "//learning/brain/research/jax:tpu_support",
        "//perftools/accelerators/xprof/api/python:xprof_session",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/tokamax/_src:test_utils",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/layer:wrapper",
    ],
)

py_test(
    name = "gmm_test",
    srcs = ["gmm_test.py"],
    shard_count = 30,
    tags = [
        "notap",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
    deps = [
        "//learning/brain/research/jax:tpu_support",
        "//perftools/accelerators/xprof/api/python:xprof_session",
        "//third_party/py/absl/logging",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/tokamax/_src:benchmarking",
        "//third_party/py/tokamax/_src/ops/experimental/gmm_v2",
        "//third_party/py/tokamax/_src/ops/experimental/gmm_v2:util",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/common:layout_util",
        "//third_party/py/tokamax/_src/ops/experimental/routing_moe/gmm:kernel",
        "//third_party/py/tokamax/_src/ops/ragged_gather:pallas_mosaic_v2_tpu",
        "//third_party/py/tokamax/_src/ops/ragged_scatter:pallas_mosaic_tpu",
    ],
)
