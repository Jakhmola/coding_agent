import os
import unittest
from unittest.mock import patch

from coding_agent import constants
from coding_agent.config import load_settings
from coding_agent.workspace_policy import WorkspacePolicy


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()

        self.assertEqual(settings.model_name, constants.DEFAULT_MODEL_NAME)
        self.assertEqual(settings.gemini_model, constants.DEFAULT_GEMINI_MODEL)
        self.assertEqual(settings.openai_base_url, constants.DEFAULT_OPENAI_BASE_URL)
        self.assertEqual(settings.max_iterations, constants.DEFAULT_MAX_ITERATIONS)
        self.assertFalse(settings.workspace_policy.read_only)
        self.assertEqual(settings.workspace_policy.allowed_commands, ("python3",))
        self.assertFalse(settings.opik.enabled)

    def test_env_overrides(self):
        env = {
            "MODEL_NAME": "custom-model",
            "GEMINI_MODEL": "gemini-test",
            "OPENAI_BASE_URL": "http://example.test/v1",
            "OPENAI_API_KEY": "test-key",
            "MAX_ITERATIONS": "7",
            "WORKSPACE_DIR": "./example",
            "WORKSPACE_READ_ONLY": "true",
            "ALLOWED_COMMANDS": "python3,pytest",
            "TOOL_TIMEOUT_SECONDS": "9",
            "MAX_FILE_READ_CHARS": "123",
            "MAX_FILE_SIZE_BYTES": "456",
            "MAX_TOOL_OUTPUT_CHARS": "789",
            "LOG_LEVEL": "DEBUG",
            "LOG_JSON": "false",
            "OPIK_ENABLED": "true",
            "OPIK_PROJECT_NAME": "test-project",
            "OPIK_URL": "http://opik.test",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()

        self.assertEqual(settings.model_name, "custom-model")
        self.assertEqual(settings.gemini_model, "gemini-test")
        self.assertEqual(settings.openai_api_key, "test-key")
        self.assertEqual(settings.max_iterations, 7)
        self.assertEqual(str(settings.workspace_policy.root), "example")
        self.assertTrue(settings.workspace_policy.read_only)
        self.assertEqual(settings.workspace_policy.allowed_commands, ("python3", "pytest"))
        self.assertEqual(settings.workspace_policy.tool_timeout_seconds, 9)
        self.assertEqual(settings.workspace_policy.max_file_read_chars, 123)
        self.assertEqual(settings.workspace_policy.max_file_size_bytes, 456)
        self.assertEqual(settings.workspace_policy.max_tool_output_chars, 789)
        self.assertEqual(settings.log_level, "DEBUG")
        self.assertFalse(settings.log_json)
        self.assertTrue(settings.opik.enabled)
        self.assertEqual(settings.opik.project_name, "test-project")
        self.assertEqual(settings.opik.url, "http://opik.test")

    def test_invalid_bool_fails_loudly(self):
        with patch.dict(os.environ, {"WORKSPACE_READ_ONLY": "maybe"}, clear=True):
            with self.assertRaisesRegex(ValueError, "WORKSPACE_READ_ONLY"):
                load_settings()

    def test_invalid_int_fails_loudly(self):
        with patch.dict(os.environ, {"MAX_ITERATIONS": "many"}, clear=True):
            with self.assertRaisesRegex(ValueError, "MAX_ITERATIONS"):
                load_settings()


class WorkspacePolicyTests(unittest.TestCase):
    def test_resolve_path_rejects_traversal(self):
        policy = WorkspacePolicy(root="calculator")

        with self.assertRaisesRegex(ValueError, "outside"):
            policy.resolve_path("../main.py")

    def test_read_only_write_check(self):
        policy = WorkspacePolicy(root="calculator", read_only=True)

        with self.assertRaisesRegex(PermissionError, "read-only"):
            policy.ensure_write_allowed()

    def test_command_allowlist(self):
        policy = WorkspacePolicy(root="calculator", allowed_commands=("python3",))

        policy.ensure_command_allowed("python3")
        with self.assertRaisesRegex(PermissionError, "not allowed"):
            policy.ensure_command_allowed("bash")


if __name__ == "__main__":
    unittest.main()

