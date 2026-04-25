from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


HEARTBEAT_SECONDS = 10


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=check)


def _run_streaming(command: list[str], label: str) -> None:
    print(f"{label}: {' '.join(command)}", flush=True)
    process = subprocess.Popen(command)
    started_at = time.monotonic()
    next_heartbeat = started_at + HEARTBEAT_SECONDS

    while process.poll() is None:
        now = time.monotonic()
        if now >= next_heartbeat:
            elapsed = int(now - started_at)
            print(f"{label}: still running after {elapsed}s...", flush=True)
            next_heartbeat = now + HEARTBEAT_SECONDS
        time.sleep(1)

    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def _compose_exec(compose: str, service: str, command: list[str]) -> list[str]:
    return [*compose.split(), "exec", "-T", service, *command]


def _post_with_heartbeat(url: str, payload: dict[str, object], label: str) -> None:
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

    def send_request() -> None:
        try:
            with urllib.request.urlopen(request) as response:
                response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{label} failed: {exc}") from exc

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(send_request)
        while not future.done():
            elapsed = int(time.monotonic() - started_at)
            if time.monotonic() >= next_heartbeat:
                print(f"{label}: still running after {elapsed}s...", flush=True)
                next_heartbeat = time.monotonic() + HEARTBEAT_SECONDS
            time.sleep(1)
        future.result()


def _warm_model(ollama_url: str, model: str) -> None:
    _post_with_heartbeat(
        f"{ollama_url.rstrip('/')}/api/generate",
        {
            "model": model,
            "prompt": "Respond with only: ok",
            "stream": False,
            "keep_alive": -1,
            "options": {
                "num_predict": 1,
            },
        },
        "warm model",
    )


def _ollama_ps(compose: str, service: str) -> str:
    result = _run(_compose_exec(compose, service, ["ollama", "ps"]))
    return result.stdout


def _is_fully_gpu(ps_output: str, model: str) -> bool:
    for line in ps_output.splitlines():
        if model in line and "100% GPU" in line:
            return True
    return False


def _recreate_model(compose: str, service: str, model: str, model_file: str) -> None:
    _run_streaming(
        _compose_exec(compose, service, ["ollama", "create", model, "-f", model_file]),
        f"create model from {model_file}",
    )


def check_model(
    compose: str,
    service: str,
    model: str,
    model_file: str,
    lowctx_model_file: str,
    ollama_url: str,
) -> int:
    print(f"checking model {model} with default context", flush=True)
    _recreate_model(compose, service, model, model_file)
    print(
        f"warming model {model}; loading weights into VRAM can take a while",
        flush=True,
    )
    _warm_model(ollama_url, model)
    print("checking Ollama residency with `ollama ps`", flush=True)
    ps_output = _ollama_ps(compose, service)
    print(ps_output)

    if _is_fully_gpu(ps_output, model):
        print(f"{model} is loaded 100% on GPU")
        return 0

    print(
        f"{model} did not load 100% on GPU. Recreating once with reduced context.",
        file=sys.stderr,
        flush=True,
    )
    _recreate_model(compose, service, model, lowctx_model_file)
    print(
        f"warming reduced-context model {model}; loading weights into VRAM can take a while",
        flush=True,
    )
    _warm_model(ollama_url, model)
    print("checking Ollama residency with `ollama ps`", flush=True)
    ps_output = _ollama_ps(compose, service)
    print(ps_output)

    if _is_fully_gpu(ps_output, model):
        print(f"{model} is loaded 100% on GPU after context reduction")
        return 0

    print(
        f"ERROR: {model} is still not 100% GPU after one context reduction. "
        "Failing loudly; choose a manual setup change before continuing.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify an Ollama model is loaded fully on GPU."
    )
    parser.add_argument("--compose", default="docker compose")
    parser.add_argument("--service", default="ollama")
    parser.add_argument("--model", default="coding-qwen-gpu")
    parser.add_argument("--model-file", default="/models/Modelfile")
    parser.add_argument("--lowctx-model-file", default="/models/Modelfile.lowctx")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    try:
        return check_model(
            args.compose,
            args.service,
            args.model,
            args.model_file,
            args.lowctx_model_file,
            args.ollama_url,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stdout, end="")
        print(exc.stderr, end="", file=sys.stderr)
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
