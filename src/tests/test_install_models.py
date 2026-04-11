import os
import tempfile
import unittest

from install_models import build_model_specs, configure_cache_env


class InstallModelsTests(unittest.TestCase):
    def test_build_model_specs_includes_runtime_models(self):
        specs = build_model_specs()
        repo_ids = {spec.repo_id for spec in specs}
        self.assertIn("Falconsai/medical_summarization", repo_ids)
        self.assertIn("knkarthick/MEETING_SUMMARY", repo_ids)
        self.assertIn("sentence-transformers/all-MiniLM-L6-v2", repo_ids)

    def test_configure_cache_env_sets_current_cache_vars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = configure_cache_env(tmpdir)
            self.assertTrue(os.path.isdir(resolved))
            self.assertEqual(os.environ["HF_HOME"], resolved)
            self.assertEqual(os.environ["HF_HUB_CACHE"], resolved)
            self.assertEqual(os.environ["TRANSFORMERS_CACHE"], resolved)
            self.assertEqual(os.environ["SENTENCE_TRANSFORMERS_HOME"], resolved)


if __name__ == "__main__":
    unittest.main()
