from .profiling_api_utils import _run_offline_profiler_test


def test_api_profiler_generation_offline_non_phased():
    _run_offline_profiler_test(is_phased=False)
