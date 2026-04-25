import unittest
from unittest.mock import patch

from scripts.model_check import _is_fully_gpu, _recreate_model, _warm_model


class ModelCheckTests(unittest.TestCase):
    def test_detects_model_loaded_fully_on_gpu(self):
        ps_output = """
NAME             ID              SIZE      PROCESSOR    UNTIL
coding-qwen-gpu  abc123          5.0 GB    100% GPU     4 minutes from now
"""

        self.assertTrue(_is_fully_gpu(ps_output, "coding-qwen-gpu"))

    def test_rejects_cpu_split(self):
        ps_output = """
NAME             ID              SIZE      PROCESSOR          UNTIL
coding-qwen-gpu  abc123          5.0 GB    63%/37% CPU/GPU    4 minutes from now
"""

        self.assertFalse(_is_fully_gpu(ps_output, "coding-qwen-gpu"))

    def test_rejects_missing_model(self):
        ps_output = """
NAME          ID        SIZE      PROCESSOR    UNTIL
other-model   abc123    5.0 GB    100% GPU     4 minutes from now
"""

        self.assertFalse(_is_fully_gpu(ps_output, "coding-qwen-gpu"))

    @patch("scripts.model_check._post_with_heartbeat")
    def test_warm_model_uses_generate_api_with_keep_alive(self, post_with_heartbeat):
        _warm_model("http://localhost:11434", "coding-qwen-gpu")

        post_with_heartbeat.assert_called_once_with(
            "http://localhost:11434/api/generate",
            {
                "model": "coding-qwen-gpu",
                "prompt": "Respond with only: ok",
                "stream": False,
                "keep_alive": -1,
                "options": {
                    "num_predict": 1,
                },
            },
            "warm model",
        )

    @patch("scripts.model_check._run_streaming")
    def test_recreate_model_streams_progress(self, run_streaming):
        _recreate_model("docker compose", "ollama", "coding-qwen-gpu", "/models/Modelfile")

        run_streaming.assert_called_once_with(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "ollama",
                "ollama",
                "create",
                "coding-qwen-gpu",
                "-f",
                "/models/Modelfile",
            ],
            "create model from /models/Modelfile",
        )


if __name__ == "__main__":
    unittest.main()
