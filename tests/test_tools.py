import tempfile
import unittest
from pathlib import Path

from coding_agent.workspace_policy import WorkspacePolicy
from functions.get_file_content import get_file_content
from functions.get_files_info import get_files_info
from functions.run_python_file import run_python_file
from functions.write_file import write_file


class ToolPolicyTests(unittest.TestCase):
    def test_get_files_info_lists_workspace_entries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "example.txt").write_text("hello")

            result = get_files_info(root, ".")

        self.assertIn("- example.txt:", result)
        self.assertIn("is_dir=False", result)

    def test_get_files_info_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = get_files_info(tmp_dir, "..")

        self.assertIn("outside the permitted working directory", result)

    def test_get_file_content_reads_and_truncates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "example.txt").write_text("abcdef")
            policy = WorkspacePolicy(root=root, max_file_read_chars=3)

            result = get_file_content(root, "example.txt", policy=policy)

        self.assertIn("abc", result)
        self.assertIn("truncated at 3 characters", result)

    def test_get_file_content_rejects_large_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "large.txt").write_text("abcdef")
            policy = WorkspacePolicy(root=root, max_file_size_bytes=3)

            result = get_file_content(root, "large.txt", policy=policy)

        self.assertIn("exceeds the maximum allowed size", result)

    def test_write_file_respects_read_only_policy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            policy = WorkspacePolicy(root=tmp_dir, read_only=True)

            result = write_file(tmp_dir, "example.txt", "hello", policy=policy)

        self.assertIn("read-only", result)

    def test_write_file_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = write_file(tmp_dir, "../escape.txt", "hello")

        self.assertIn("outside the permitted working directory", result)

    def test_run_python_file_executes_allowed_script(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "script.py").write_text('print("hello")')

            result = run_python_file(root, "script.py")

        self.assertIn("STDOUT:", result)
        self.assertIn("hello", result)

    def test_run_python_file_respects_command_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "script.py").write_text('print("hello")')
            policy = WorkspacePolicy(root=root, allowed_commands=())

            result = run_python_file(root, "script.py", policy=policy)

        self.assertIn('Command "python3" is not allowed', result)

    def test_run_python_file_truncates_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "script.py").write_text('print("abcdef")')
            policy = WorkspacePolicy(root=root, max_tool_output_chars=12)

            result = run_python_file(root, "script.py", policy=policy)

        self.assertIn("truncated at 12 characters", result)


if __name__ == "__main__":
    unittest.main()

