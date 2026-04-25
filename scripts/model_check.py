from __future__ import annotations

import argparse
import subprocess
import sys


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=check)


def _compose_exec(compose: str, service: str, command: list[str]) -> list[str]:
    return [*compose.split(), "exec", "-T", service, *command]


def _warm_model(compose: str, service: str, model: str) -> None:
    _run(
        _compose_exec(
            compose,
            service,
            ["ollama", "run", model, "Respond with only: ok"],
        )
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
    _run(_compose_exec(compose, service, ["ollama", "create", model, "-f", model_file]))


def check_model(
    compose: str,
    service: str,
    model: str,
    model_file: str,
    lowctx_model_file: str,
) -> int:
    print(f"warming model {model} with default context")
    _recreate_model(compose, service, model, model_file)
    _warm_model(compose, service, model)
    ps_output = _ollama_ps(compose, service)
    print(ps_output)

    if _is_fully_gpu(ps_output, model):
        print(f"{model} is loaded 100% on GPU")
        return 0

    print(
        f"{model} did not load 100% on GPU. Recreating once with reduced context.",
        file=sys.stderr,
    )
    _recreate_model(compose, service, model, lowctx_model_file)
    _warm_model(compose, service, model)
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
    args = parser.parse_args()

    try:
        return check_model(
            args.compose,
            args.service,
            args.model,
            args.model_file,
            args.lowctx_model_file,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stdout, end="")
        print(exc.stderr, end="", file=sys.stderr)
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
