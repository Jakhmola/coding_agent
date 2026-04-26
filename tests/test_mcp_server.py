import asyncio
import unittest

from coding_agent.config import load_settings

try:
    from coding_agent.mcp_server import FastMCP, build_mcp_server, list_server_capabilities
except RuntimeError:
    FastMCP = None


@unittest.skipIf(FastMCP is None, "mcp SDK is not installed")
class McpServerTests(unittest.TestCase):
    def test_registers_current_tools_and_prompt(self):
        async def run_test():
            capabilities = await list_server_capabilities(load_settings())
            tool_names = {tool["name"] for tool in capabilities["tools"]}
            prompt_names = {prompt["name"] for prompt in capabilities["prompts"]}

            self.assertEqual(
                {
                    "get_files_info",
                    "get_file_content",
                    "search_files",
                    "grep_files",
                    "append_file",
                    "replace_in_file",
                    "write_file",
                    "run_python_file",
                },
                tool_names,
            )
            self.assertIn("coding_agent_system_prompt", prompt_names)

        asyncio.run(run_test())

    def test_prompt_returns_system_prompt_text(self):
        async def run_test():
            server = build_mcp_server(load_settings())
            prompt_result = await server.get_prompt("coding_agent_system_prompt")
            prompt_text = prompt_result.messages[0].content.text

            self.assertIn("You are a helpful AI coding agent.", prompt_text)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
