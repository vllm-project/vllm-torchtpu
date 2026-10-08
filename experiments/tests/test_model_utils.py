"""test_model_utils.py.

Unit tests for core.model_utils.resolve_model_path:
- Local paths (absolute / relative to base_dir) are returned as-is.
- Non repo-id strings never reach the Hugging Face Hub.
- Repo-id strings that cannot be resolved raise a clear error.
"""

import os
import tempfile
import unittest
from unittest import mock

from core import model_utils
from core.model_utils import looks_like_hf_repo_id, resolve_model_path


class TestLooksLikeHfRepoId(unittest.TestCase):

    def test_repo_id_shapes(self):
        for ok in ["google/gemma-4-31B-it", "org/name.v2", "a_b/c-d"]:
            self.assertTrue(looks_like_hf_repo_id(ok), ok)
        for bad in [
                "model_A", "/mnt/models/x", "a/b/c", "org/", "/org/name",
                "org name/x", "", "./rel/path"
        ]:
            self.assertFalse(looks_like_hf_repo_id(bad), bad)


class TestResolveModelPath(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(resolve_model_path(""), "")

    def test_existing_local_path_short_circuits(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "ckpt")
            os.makedirs(sub)
            with mock.patch.object(model_utils,
                                   "snapshot_download") as dl, \
                    mock.patch.object(model_utils,
                                      "try_to_load_from_cache") as cache:
                self.assertEqual(resolve_model_path(sub), sub)
                self.assertEqual(resolve_model_path("ckpt", base_dir=d), sub)
            dl.assert_not_called()
            cache.assert_not_called()

    def test_non_repo_id_never_touches_hub(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(model_utils,
                                   "snapshot_download") as dl, \
                    mock.patch.object(model_utils,
                                      "try_to_load_from_cache") as cache:
                self.assertEqual(resolve_model_path("model_A"),
                                 os.path.abspath("model_A"))
                self.assertEqual(resolve_model_path("missing_ref", base_dir=d),
                                 os.path.join(d, "missing_ref"))
            dl.assert_not_called()
            cache.assert_not_called()

    def test_repo_id_uses_cache_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "config.json")
            open(cfg, "w").close()
            with mock.patch.object(model_utils,
                                   "try_to_load_from_cache",
                                   return_value=cfg), \
                    mock.patch.object(model_utils,
                                      "snapshot_download") as dl:
                self.assertEqual(resolve_model_path("org/name"), d)
            dl.assert_not_called()

    def test_repo_id_download_failure_raises_clear_error(self):
        with mock.patch.object(model_utils,
                               "try_to_load_from_cache",
                               return_value=None), \
                mock.patch.object(model_utils,
                                  "snapshot_download",
                                  side_effect=OSError("401 gated")):
            with self.assertRaisesRegex(FileNotFoundError,
                                        r"org/gated.*OSError.*401 gated"):
                resolve_model_path("org/gated")


if __name__ == "__main__":
    unittest.main()
