import unittest

from scripts.model_check import _is_fully_gpu


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


if __name__ == "__main__":
    unittest.main()
