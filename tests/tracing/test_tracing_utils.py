from unittest.mock import patch

import pytest

from vllm_torchtpu.tracing.utils import (extract_kv_lens_for_tracing,
                                         extract_request_ids_for_tracing,
                                         trim_request_id_suffix)

pytestmark = pytest.mark.cpu_test


def test_trim_request_id_suffix():
    # Case A: Standard long cmpl string correctly truncates to the first 6 parts.
    assert trim_request_id_suffix("cmpl-1-2-3-4-5-6-7-8") == "cmpl-1-2-3-4-5"

    # Case B: A short string is gracefully returned untouched.
    assert trim_request_id_suffix("cmpl-1234") == "cmpl-1234"

    # Case C: Non-"cmpl" prefixed IDs are bypassed.
    assert trim_request_id_suffix(
        "custom-req-id-1-2-3-4-5-6-7") == "custom-req-id-1-2-3-4-5-6-7"


def test_extract_request_ids_for_tracing():
    req_ids = ["cmpl-1", "cmpl-2", None, "cmpl-3", "cmpl-4"]

    # Case A: Test pagination by passing start_index and num_reqs
    res = extract_request_ids_for_tracing(req_ids, start_index=1, num_reqs=2)
    # the active_ids will be `["cmpl-2", None]`
    # and the trimmed won't contain None
    assert res == {"request_id1": "cmpl-2"}

    # Check another slice
    res_2 = extract_request_ids_for_tracing(req_ids, start_index=3, num_reqs=2)
    assert res_2 == {"request_id1": "cmpl-3", "request_id2": "cmpl-4"}

    # Case C: Mock the internal components to fail and verify it swallows error
    class BadItem:

        def __str__(self):
            raise ValueError("Intentional crash")

    with patch("vllm_torchtpu.tracing.utils.logger.warning") as mock_warning:
        res_3 = extract_request_ids_for_tracing(["cmpl-1", BadItem()], 0, 2)
        assert res_3 == {
        }  # the list comprehension exception catches in outer try
        mock_warning.assert_called_once()


def test_extract_kv_lens_for_tracing():
    num_computed_tokens = [100, 200, 300, 400]

    # Case A: Test pagination by passing start_index and num_reqs
    res = extract_kv_lens_for_tracing(num_computed_tokens,
                                      start_index=1,
                                      num_reqs=2)
    assert res == {
        "kv_len1": 200,
        "kv_len2": 300,
        "min_kv_len": 200,
        "max_kv_len": 300,
        "avg_kv_len": 250,
    }

    # Case B: Default slicing (start_index=0, num_reqs=-1) uses the whole list
    res_2 = extract_kv_lens_for_tracing(num_computed_tokens)
    assert res_2 == {
        "kv_len1": 100,
        "kv_len2": 200,
        "kv_len3": 300,
        "kv_len4": 400,
        "min_kv_len": 100,
        "max_kv_len": 400,
        "avg_kv_len": 250,
    }

    # Case C: Empty slice yields no kwargs and no summary stats
    res_3 = extract_kv_lens_for_tracing(num_computed_tokens,
                                        start_index=4,
                                        num_reqs=2)
    assert res_3 == {}
