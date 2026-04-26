import tempfile
import unittest
from pathlib import Path

from coding_agent.workspace_policy import WorkspacePolicy
from functions.append_file import append_file
from functions.get_file_content import get_file_content
from functions.get_files_info import get_files_info
from functions.grep_files import grep_files
from functions.replace_in_file import replace_in_file
from functions.run_python_file import run_python_file
from functions.search_files import search_files
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

    def test_append_file_adds_new_line_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "example.txt"
            target.write_text("hello")

            result = append_file(root, "example.txt", "world")

            content = target.read_text()

        self.assertIn("Successfully appended", result)
        self.assertEqual(content, "hello\nworld\n")

    def test_append_file_respects_read_only_policy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            policy = WorkspacePolicy(root=tmp_dir, read_only=True)

            result = append_file(tmp_dir, "example.txt", "hello", policy=policy)

        self.assertIn("read-only", result)

    def test_append_file_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = append_file(tmp_dir, "../escape.txt", "hello")

        self.assertIn("outside the permitted working directory", result)

    def test_append_file_rejects_size_overflow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "example.txt").write_text("abc")
            policy = WorkspacePolicy(root=root, max_file_size_bytes=4)

            result = append_file(root, "example.txt", "de", policy=policy)

        self.assertIn("maximum allowed size", result)

    def test_replace_in_file_replaces_exact_expected_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "example.txt"
            target.write_text("hello old")

            result = replace_in_file(root, "example.txt", "old", "new")

            content = target.read_text()

        self.assertIn("Successfully replaced 1 occurrence", result)
        self.assertEqual(content, "hello new")

    def test_replace_in_file_fails_when_match_count_differs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "example.txt"
            target.write_text("one one")

            result = replace_in_file(root, "example.txt", "one", "two")

            content = target.read_text()

        self.assertIn("Expected 1 replacement", result)
        self.assertEqual(content, "one one")

    def test_search_files_finds_names_recursively(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pkg").mkdir()
            (root / "pkg" / "example.txt").write_text("hello")

            result = search_files(root, "example")

        self.assertIn("pkg/example.txt", result)

    def test_search_files_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = search_files(tmp_dir, "example", directory="..")

        self.assertIn("outside the permitted working directory", result)

    def test_search_files_skips_generated_directories_and_truncates_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "example.pyc").write_text("hidden")
            (root / "pkg").mkdir()
            (root / "pkg" / "example_with_a_long_name.txt").write_text("visible")
            policy = WorkspacePolicy(root=root, max_tool_output_chars=45)

            result = search_files(root, "example", policy=policy)

        self.assertIn("pkg/example", result)
        self.assertNotIn("__pycache__", result)
        self.assertIn("Output truncated at 45 characters", result)

    def test_grep_files_finds_content_matches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "example.py").write_text('print("needle")\n')

            result = grep_files(root, "needle", file_pattern="*.py")

        self.assertIn("example.py:1:", result)
        self.assertIn("needle", result)

    def test_grep_files_handles_invalid_regex(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = grep_files(tmp_dir, "[")

        self.assertIn("Invalid regex pattern", result)

    def test_grep_files_skips_oversized_and_generated_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "hidden.py").write_text("needle")
            (root / "large.py").write_text("needle" * 3)
            (root / "visible.py").write_text("needle")
            policy = WorkspacePolicy(root=root, max_file_size_bytes=10)

            result = grep_files(root, "needle", file_pattern="*.py", policy=policy)

        self.assertIn("visible.py:1:", result)
        self.assertNotIn("large.py", result)
        self.assertNotIn("__pycache__", result)

    def test_grep_files_respects_file_pattern_and_truncates_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "example.py").write_text("needle\n")
            (root / "example.txt").write_text("needle\n")
            policy = WorkspacePolicy(root=root, max_tool_output_chars=15)

            result = grep_files(root, "needle", file_pattern="*.py", policy=policy)

        self.assertIn("example.py", result)
        self.assertNotIn("example.txt", result)
        self.assertIn("Output truncated at 15 characters", result)

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
