from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


HEARTBEAT_SECONDS = 10


def _run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=True, env=env)


def _compose_command(compose: str, command: list[str]) -> list[str]:
    return [*compose.split(), *command]


def _compose_up(compose: str, service: str, ctx_size: int) -> None:
    env = {**os.environ, "LLAMA_CTX_SIZE": str(ctx_size)}
    command = _compose_command(
        compose,
        ["up", "-d", "--force-recreate", service],
    )
    print(f"starting {service} with ctx-size {ctx_size}: {' '.join(command)}", flush=True)
    _run(command, env=env)


def _compose_logs(compose: str, service: str) -> str:
    result = _run(_compose_command(compose, ["logs", "--no-color", service]))
    return result.stdout


def _get_json(url: str, label: str) -> dict[str, Any]:
    print(f"{label}: GET {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode())
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} failed: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} did not return a JSON object")
    return payload


def _post_json_with_heartbeat(
    url: str,
    payload: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"{label}: POST {url}", flush=True)
    started_at = time.monotonic()
    next_heartbeat = started_at + HEARTBEAT_SECONDS

    def send_request() -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode())
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label} failed: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"{label} did not return a JSON object")
        return payload

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(send_request)
        while not future.done():
            now = time.monotonic()
            if now >= next_heartbeat:
                elapsed = int(now - started_at)
                print(f"{label}: still running after {elapsed}s...", flush=True)
                next_heartbeat = now + HEARTBEAT_SECONDS
            time.sleep(1)
        return future.result()


def _wait_for_health(base_url: str, *, timeout_seconds: int = 300) -> None:
    health_url = f"{base_url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_seconds
    next_heartbeat = time.monotonic()
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    print("llama.cpp server is healthy", flush=True)
                    return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc

        if time.monotonic() >= next_heartbeat:
            print("waiting for llama.cpp /health...", flush=True)
            next_heartbeat = time.monotonic() + HEARTBEAT_SECONDS
        time.sleep(2)

    raise RuntimeError(f"llama.cpp server did not become healthy: {last_error}")


def _check_models(base_url: str, model: str) -> None:
    payload = _get_json(f"{base_url.rstrip('/')}/v1/models", "list models")
    models = payload.get("data")
    if not isinstance(models, list):
        raise RuntimeError("/v1/models response did not include a data list")

    model_ids = {
        item.get("id")
        for item in models
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if model not in model_ids:
        raise RuntimeError(f"{model} was not listed by /v1/models: {sorted(model_ids)}")


def _check_chat_completion(base_url: str, model: str) -> None:
    payload = _post_json_with_heartbeat(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Respond with only: ok"}],
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
        },
        "chat completion smoke",
    )
    _require_choices(payload, "chat completion smoke")


def _check_tool_request_shape(base_url: str, model: str) -> None:
    payload = _post_json_with_heartbeat(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Use the available tool to list files in the current directory.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_files_info",
                        "description": "List files in a workspace directory.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "directory": {
                                    "type": "string",
                                    "description": "Directory relative to the workspace.",
                                }
                            },
                            "required": ["directory"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 128,
            "temperature": 0,
            "stream": False,
        },
        "tool request-shape smoke",
    )
    _require_choices(payload, "tool request-shape smoke")


def _check_gpu_offload(compose: str, service: str) -> None:
    logs = _compose_logs(compose, service)
    match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", logs)
    if match is None:
        raise RuntimeError("llama.cpp logs did not report GPU layer offload")

    offloaded_layers = int(match.group(1))
    total_layers = int(match.group(2))
    if offloaded_layers != total_layers:
        raise RuntimeError(
            f"llama.cpp offloaded only {offloaded_layers}/{total_layers} layers to GPU"
        )


def _require_choices(payload: dict[str, Any], label: str) -> None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{label} did not include choices")


def _run_llama_checks(compose: str, service: str, model: str, base_url: str, ctx_size: int) -> None:
    _compose_up(compose, service, ctx_size)
    _wait_for_health(base_url)
    _check_gpu_offload(compose, service)
    _check_models(base_url, model)
    _check_chat_completion(base_url, model)
    _check_tool_request_shape(base_url, model)


def check_model(
    compose: str,
    service: str,
    model: str,
    base_url: str,
    ctx_size: int,
    lowctx_size: int,
) -> int:
    print(f"checking llama.cpp model {model} with ctx-size {ctx_size}", flush=True)
    try:
        _run_llama_checks(compose, service, model, base_url, ctx_size)
    except Exception as exc:
        print(
            f"default ctx-size {ctx_size} failed: {exc}. Retrying once with ctx-size {lowctx_size}.",
            file=sys.stderr,
            flush=True,
        )
        try:
            _run_llama_checks(compose, service, model, base_url, lowctx_size)
        except Exception as lowctx_exc:
            print(
                f"ERROR: llama.cpp model check failed at ctx-size {lowctx_size}: {lowctx_exc}",
                file=sys.stderr,
                flush=True,
            )
            return 1

        print(f"{model} passed llama.cpp checks with reduced ctx-size {lowctx_size}")
        return 0

    print(f"{model} passed llama.cpp checks with ctx-size {ctx_size}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify llama.cpp serves the local GGUF model through OpenAI-compatible endpoints."
    )
    parser.add_argument("--compose", default="docker compose")
    parser.add_argument("--service", default="llama-cpp")
    parser.add_argument("--model", default="opus-ghost-codex-4b-q5")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--lowctx-size", type=int, default=2048)
    args = parser.parse_args()

    try:
        return check_model(
            args.compose,
            args.service,
            args.model,
            args.base_url,
            args.ctx_size,
            args.lowctx_size,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stdout, end="")
        print(exc.stderr, end="", file=sys.stderr)
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
