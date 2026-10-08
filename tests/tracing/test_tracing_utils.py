from vllm_torchtpu.tracing.utils import (
    extract_input_lens_for_tracing,
    extract_kv_lens_for_tracing,
    extract_request_details_for_tracing,
    trim_request_id_suffix,
)


def test_trim_request_id_suffix():
    # Case A: Standard long cmpl string correctly truncates to the first 6 parts.
    assert trim_request_id_suffix("cmpl-1-2-3-4-5-6-7-8") == "cmpl-1-2-3-4-5"

    # Case B: A short string is gracefully returned untouched.
    assert trim_request_id_suffix("cmpl-1234") == "cmpl-1234"

    # Case C: Non-"cmpl" prefixed IDs are bypassed.
    assert (
        trim_request_id_suffix("custom-req-id-1-2-3-4-5-6-7")
        == "custom-req-id-1-2-3-4-5-6-7"
    )


def test_extract_kv_lens_for_tracing():
    num_computed_tokens = [100, 200, 300, 400]

    # Case A: Test pagination by passing start_index and num_reqs
    res = extract_kv_lens_for_tracing(num_computed_tokens, start_index=1, num_reqs=2)
    assert res == {"min_kv_len": 200, "max_kv_len": 300, "avg_kv_len": 250}

    # Case B: Default slicing (start_index=0, num_reqs=-1) uses the whole list
    res_2 = extract_kv_lens_for_tracing(num_computed_tokens)
    assert res_2 == {"min_kv_len": 100, "max_kv_len": 400, "avg_kv_len": 250}

    # Case C: Empty slice yields no kwargs
    res_3 = extract_kv_lens_for_tracing(num_computed_tokens, start_index=4, num_reqs=2)
    assert res_3 == {}


def test_extract_input_lens_for_tracing():
    num_prompt_tokens = [100, 2048, 7, 31]

    assert extract_input_lens_for_tracing(num_prompt_tokens, 1, 2) == {
        "min_isl": 7,
        "max_isl": 2048,
        "avg_isl": 1027,
    }
    assert extract_input_lens_for_tracing(num_prompt_tokens, 4, 2) == {}


def test_extract_request_details_for_tracing():
    req_ids = ["cmpl-a-1-2-3-4-tail", None, "b", "x,y#z=w v/u"]
    num_prompt_tokens = [5000, 0, 900, 10]
    num_computed_tokens = [5120, 0, 0, 3]
    num_scheduled_tokens = {"cmpl-a-1-2-3-4-tail": 4, "b": 512, "x,y#z=w v/u": 1}

    res = extract_request_details_for_tracing(
        req_ids, num_prompt_tokens, num_computed_tokens, num_scheduled_tokens
    )
    # Freed slots are skipped, ids are trimmed, and characters that would
    # break the trace's `key=value,...` encoding or the entry are replaced.
    assert res == {
        "reqs_format": "req_id/isl/kv_len/new_tokens",
        "reqs": "cmpl-a-1-2-3-4/5000/5120/4 b/900/0/512 x_y_z_w_v_u/10/3/1",
    }

    # Pagination follows the chunk being traced.
    res_2 = extract_request_details_for_tracing(
        req_ids, num_prompt_tokens, num_computed_tokens, num_scheduled_tokens, 2, 1
    )
    assert res_2["reqs"] == "b/900/0/512"

    assert (
        extract_request_details_for_tracing(
            req_ids, num_prompt_tokens, num_computed_tokens, num_scheduled_tokens, 1, 1
        )
        == {}
    )


def test_extract_request_details_caps_the_list():
    req_ids = [f"r{i}" for i in range(5)]
    lens = [10, 11, 12, 13, 14]

    res = extract_request_details_for_tracing(req_ids, lens, lens, {}, max_entries=2)

    assert res == {
        "reqs_format": "req_id/isl/kv_len/new_tokens",
        "reqs_omitted": 3,
        "reqs": "r0/10/10/0 r1/11/11/0 +3 more",
    }
