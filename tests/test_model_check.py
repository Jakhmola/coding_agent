import unittest
from unittest.mock import call, patch

from scripts.model_check import (
    _check_chat_completion,
    _check_gpu_offload,
    _check_models,
    _check_tool_request_shape,
    _compose_up,
    check_model,
)


class ModelCheckTests(unittest.TestCase):
    @patch("scripts.model_check._run")
    def test_compose_up_sets_context_size_env(self, run):
        with patch("builtins.print"):
            _compose_up("docker compose", "llama-cpp", 4096)

        args, kwargs = run.call_args
        self.assertEqual(
            args[0],
            [
                "docker",
                "compose",
                "up",
                "-d",
                "--force-recreate",
                "llama-cpp",
            ],
        )
        self.assertEqual(kwargs["env"]["LLAMA_CTX_SIZE"], "4096")

    @patch("scripts.model_check._get_json")
    def test_check_models_requires_configured_alias(self, get_json):
        get_json.return_value = {"data": [{"id": "opus-ghost-codex-4b-q5"}]}

        _check_models("http://localhost:8080", "opus-ghost-codex-4b-q5")

        get_json.assert_called_once_with(
            "http://localhost:8080/v1/models",
            "list models",
        )

    @patch("scripts.model_check._get_json")
    def test_check_models_rejects_missing_alias(self, get_json):
        get_json.return_value = {"data": [{"id": "other-model"}]}

        with self.assertRaisesRegex(RuntimeError, "was not listed"):
            _check_models("http://localhost:8080", "opus-ghost-codex-4b-q5")

    @patch("scripts.model_check._post_json_with_heartbeat")
    def test_chat_completion_smoke_uses_openai_chat_endpoint(self, post_json):
        post_json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        _check_chat_completion("http://localhost:8080", "opus-ghost-codex-4b-q5")

        post_json.assert_called_once()
        url, payload, label = post_json.call_args.args
        self.assertEqual(url, "http://localhost:8080/v1/chat/completions")
        self.assertEqual(payload["model"], "opus-ghost-codex-4b-q5")
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["stream"], False)
        self.assertEqual(label, "chat completion smoke")

    @patch("scripts.model_check._post_json_with_heartbeat")
    def test_tool_request_shape_smoke_sends_tools_payload(self, post_json):
        post_json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        _check_tool_request_shape("http://localhost:8080", "opus-ghost-codex-4b-q5")

        url, payload, label = post_json.call_args.args
        self.assertEqual(url, "http://localhost:8080/v1/chat/completions")
        self.assertEqual(payload["model"], "opus-ghost-codex-4b-q5")
        self.assertEqual(payload["tools"][0]["function"]["name"], "get_files_info")
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(label, "tool request-shape smoke")

    @patch("scripts.model_check._compose_logs")
    def test_check_gpu_offload_accepts_all_layers_on_gpu(self, compose_logs):
        compose_logs.return_value = "load_tensors: offloaded 33/33 layers to GPU"

        _check_gpu_offload("docker compose", "llama-cpp")

        compose_logs.assert_called_once_with("docker compose", "llama-cpp")

    @patch("scripts.model_check._compose_logs")
    def test_check_gpu_offload_rejects_partial_gpu_layers(self, compose_logs):
        compose_logs.return_value = "load_tensors: offloaded 20/33 layers to GPU"

        with self.assertRaisesRegex(RuntimeError, "20/33"):
            _check_gpu_offload("docker compose", "llama-cpp")

    @patch("scripts.model_check._compose_logs")
    def test_check_gpu_offload_rejects_missing_gpu_report(self, compose_logs):
        compose_logs.return_value = "load_tensors: CPU buffer size = 3000 MiB"

        with self.assertRaisesRegex(RuntimeError, "did not report"):
            _check_gpu_offload("docker compose", "llama-cpp")

    @patch("scripts.model_check._run_llama_checks")
    def test_check_model_retries_once_with_reduced_context(self, run_checks):
        run_checks.side_effect = [RuntimeError("too much context"), None]

        with patch("builtins.print"):
            result = check_model(
                "docker compose",
                "llama-cpp",
                "opus-ghost-codex-4b-q5",
                "http://localhost:8080",
                4096,
                2048,
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            run_checks.call_args_list,
            [
                call(
                    "docker compose",
                    "llama-cpp",
                    "opus-ghost-codex-4b-q5",
                    "http://localhost:8080",
                    4096,
                ),
                call(
                    "docker compose",
                    "llama-cpp",
                    "opus-ghost-codex-4b-q5",
                    "http://localhost:8080",
                    2048,
                ),
            ],
        )

    @patch("scripts.model_check._run_llama_checks")
    def test_check_model_fails_after_reduced_context_failure(self, run_checks):
        run_checks.side_effect = [RuntimeError("default failed"), RuntimeError("low failed")]

        with patch("builtins.print"):
            result = check_model(
                "docker compose",
                "llama-cpp",
                "opus-ghost-codex-4b-q5",
                "http://localhost:8080",
                4096,
                2048,
            )

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
