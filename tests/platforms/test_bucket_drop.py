# SPDX-License-Identifier: Apache-2.0
import bisect

from vllm_torchtpu.platforms.bucket_drop import drop_buckets


def test_no_drop_keeps_list():
    assert drop_buckets([16, 32, 64], [], "token") == [16, 32, 64]


def test_drop_never_removes_largest():
    assert drop_buckets([16, 32, 64, 128], [32, 128], "token") == [16, 64, 128]


def test_every_size_still_pads_to_a_kept_bucket():
    kept = drop_buckets([16, 32, 64, 128, 256], [16, 64, 256], "token")
    for n in range(1, 257):
        i = bisect.bisect_left(kept, n)
        assert i < len(kept) and kept[i] >= n
