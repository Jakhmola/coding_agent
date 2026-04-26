import sys
import unittest
from urllib.error import URLError
from unittest.mock import patch

from coding_agent.config import OpikSettings
from coding_agent.tracing import (
    NoopTracer,
    OpikTracer,
    build_tracer,
    sanitize_json,
    sanitize_tool_arguments,
)


class FakeOpikClient:
    def __init__(self):
        self.calls = []
        self.flushed = False

    def trace(self, **kwargs):
        self.calls.append(("trace", kwargs))
        return {"id": "trace_1"}

    def span(self, **kwargs):
        self.calls.append(("span", kwargs))
        return {"id": f"span_{len(self.calls)}"}

    def update_trace(self, **kwargs):
        self.calls.append(("update_trace", kwargs))

    def update_span(self, **kwargs):
        self.calls.append(("update_span", kwargs))

    def flush(self):
        self.flushed = True


class FakeOpikModule:
    def __init__(self):
        self.constructor_kwargs = None
        self.client = FakeOpikClient()

    def Opik(self, **kwargs):
        self.constructor_kwargs = kwargs
        return self.client


class TracingTests(unittest.TestCase):
    def test_disabled_mode_does_not_import_or_require_opik(self):
        with patch.dict(sys.modules, {"opik": None}):
            tracer = build_tracer(OpikSettings(enabled=False))

        self.assertIsInstance(tracer, NoopTracer)

    def test_enabled_mode_creates_trace_and_span_with_fake_opik_client(self):
        fake_module = FakeOpikModule()
        settings = OpikSettings(
            enabled=True,
            api_key="test-key",
            workspace="workspace",
            project_name="coding-agent-test",
            url="http://opik.local",
        )

        tracer = OpikTracer(settings, opik_module=fake_module)
        with tracer.trace("workflow", input={"prompt": "hello"}) as trace:
            with tracer.span(
                "model",
                span_type="llm",
                model="local-model",
                provider="llama.cpp",
                usage={"total_tokens": 12},
            ) as span:
                span.set_output({"content": "done"})
            trace.set_output({"status": "ok"})

        self.assertEqual(fake_module.constructor_kwargs["host"], "http://opik.local")
        self.assertEqual(fake_module.constructor_kwargs["api_key"], "test-key")
        self.assertFalse(fake_module.constructor_kwargs["batching"])
        call_names = [name for name, _ in fake_module.client.calls]
        self.assertEqual(call_names, ["trace", "span", "update_span", "update_trace"])
        span_call = fake_module.client.calls[1][1]
        self.assertEqual(span_call["trace_id"], "trace_1")
        self.assertEqual(span_call["type"], "llm")
        self.assertEqual(span_call["model"], "local-model")
        update_span = fake_module.client.calls[2][1]
        self.assertEqual(update_span["usage"]["total_tokens"], 12)
        self.assertEqual(update_span["output"]["content"]["preview"], "done")
        self.assertTrue(fake_module.client.flushed)

    @patch("coding_agent.tracing.urlopen", side_effect=URLError("offline"))
    def test_enabled_mode_fails_loudly_when_host_is_unreachable(self, _urlopen):
        settings = OpikSettings(enabled=True, url="http://localhost:5173")

        with self.assertRaisesRegex(RuntimeError, "Opik host is unreachable"):
            OpikTracer(settings)

    def test_sanitizers_redact_secrets_and_bound_content(self):
        source = "line\n" * 200
        sanitized = sanitize_json(
            {
                "OPENAI_API_KEY": "sk-test-secret",
                "messages": [{"role": "user", "content": source}],
                "result": source,
            }
        )
        tool_args = sanitize_tool_arguments(
            {
                "file_path": "notes.txt",
                "content": source,
                "api_key": "secret",
            }
        )

        self.assertEqual(sanitized["OPENAI_API_KEY"], "[redacted]")
        self.assertNotIn(source, str(sanitized))
        self.assertIsInstance(sanitized["result"], dict)
        self.assertEqual(tool_args["api_key"], "[redacted]")
        self.assertEqual(tool_args["file_path"], "notes.txt")
        self.assertIsInstance(tool_args["content"], dict)
        self.assertLess(len(tool_args["content"]["preview"]), len(source))


if __name__ == "__main__":
    unittest.main()
