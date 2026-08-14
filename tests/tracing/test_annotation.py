from unittest.mock import MagicMock, patch

from vllm_torchtpu.tracing.annotation import (TraceAnnotation,
                                              is_trace_annotation_enabled)


@patch("torch.autograd._profiler_enabled", return_value=False)
def test_trace_annotation_disabled(mock_profiler_enabled):
    assert not is_trace_annotation_enabled()

    with patch("torch.profiler.record_function") as mock_rf:
        with TraceAnnotation("ModelForward"):
            pass
        mock_rf.assert_not_called()


@patch("torch.autograd._profiler_enabled", return_value=True)
def test_trace_annotation_enabled_no_kwargs(mock_profiler_enabled):
    assert is_trace_annotation_enabled()

    with patch("torch.profiler.record_function") as mock_rf:
        mock_context = MagicMock()
        mock_rf.return_value = mock_context

        with TraceAnnotation("ModelForward"):
            pass

        mock_rf.assert_called_once_with("ModelForward")
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()


@patch("torch.autograd._profiler_enabled", return_value=True)
def test_trace_annotation_enabled_with_kwargs(mock_profiler_enabled):
    with patch("torch.profiler.record_function") as mock_rf:
        mock_context = MagicMock()
        mock_rf.return_value = mock_context

        with TraceAnnotation("ModelForward",
                             num_reqs=2,
                             request_id1="cmpl-xyz"):
            pass

        # Verify the context properly appends the delimiter hash tags
        mock_rf.assert_called_once_with(
            "ModelForward#num_reqs=2,request_id1=cmpl-xyz#")
