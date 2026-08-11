from .profiling_api_utils import _run_online_profiler_test


def test_api_profiler_generation_online_non_phased():
    _run_online_profiler_test(is_phased=False)
