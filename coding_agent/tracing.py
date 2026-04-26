from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from coding_agent.config import OpikSettings


JsonObject = dict[str, Any]
MAX_PREVIEW_CHARS = 240
MAX_COLLECTION_ITEMS = 20
SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
)
CONTENT_KEY_PARTS = ("content", "old_text", "new_text", "result", "prompt")


class TraceSpan(Protocol):
    def set_output(self, output: JsonObject) -> None:
        ...

    def set_metadata(self, metadata: JsonObject) -> None:
        ...

    def set_usage(self, usage: JsonObject) -> None:
        ...

    def set_error(self, message: str, *, type_: str | None = None) -> None:
        ...


class Tracer(Protocol):
    def trace(
        self,
        name: str,
        *,
        input: JsonObject | None = None,
        metadata: JsonObject | None = None,
        tags: list[str] | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        ...

    def span(
        self,
        name: str,
        *,
        span_type: str = "general",
        input: JsonObject | None = None,
        output: JsonObject | None = None,
        metadata: JsonObject | None = None,
        tags: list[str] | None = None,
        model: str | None = None,
        provider: str | None = None,
        usage: JsonObject | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        ...


@dataclass
class RecordedSpan:
    kind: str
    name: str
    span_type: str = "general"
    input: JsonObject | None = None
    output: JsonObject | None = None
    metadata: JsonObject | None = None
    tags: list[str] = field(default_factory=list)
    model: str | None = None
    provider: str | None = None
    usage: JsonObject | None = None
    error: JsonObject | None = None


class _NoopTraceSpan:
    def __enter__(self) -> TraceSpan:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def set_output(self, output: JsonObject) -> None:
        return None

    def set_metadata(self, metadata: JsonObject) -> None:
        return None

    def set_usage(self, usage: JsonObject) -> None:
        return None

    def set_error(self, message: str, *, type_: str | None = None) -> None:
        return None


class NoopTracer:
    def trace(
        self,
        name: str,
        *,
        input: JsonObject | None = None,
        metadata: JsonObject | None = None,
        tags: list[str] | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        return _NoopTraceSpan()

    def span(
        self,
        name: str,
        *,
        span_type: str = "general",
        input: JsonObject | None = None,
        output: JsonObject | None = None,
        metadata: JsonObject | None = None,
        tags: list[str] | None = None,
        model: str | None = None,
        provider: str | None = None,
        usage: JsonObject | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        return _NoopTraceSpan()


class RecordingTraceSpan:
    def __init__(self, record: RecordedSpan) -> None:
        self.record = record

    def __enter__(self) -> TraceSpan:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None:
            self.set_error(str(exc), type_=type(exc).__name__)
        return None

    def set_output(self, output: JsonObject) -> None:
        self.record.output = sanitize_json(output)

    def set_metadata(self, metadata: JsonObject) -> None:
        current = self.record.metadata or {}
        self.record.metadata = {**current, **sanitize_json(metadata)}

    def set_usage(self, usage: JsonObject) -> None:
        self.record.usage = sanitize_json(usage)

    def set_error(self, message: str, *, type_: str | None = None) -> None:
        self.record.error = sanitize_error(message, type_=type_)


class RecordingTracer:
    def __init__(self) -> None:
        self.records: list[RecordedSpan] = []

    def trace(
        self,
        name: str,
        *,
        input: JsonObject | None = None,
        metadata: JsonObject | None = None,
        tags: list[str] | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        return self._record(
            kind="trace",
            name=name,
            span_type="trace",
            input=input,
            metadata=metadata,
            tags=tags,
        )

    def span(
        self,
        name: str,
        *,
        span_type: str = "general",
        input: JsonObject | None = None,
        output: JsonObject | None = None,
        metadata: JsonObject | None = None,
        tags: list[str] | None = None,
        model: str | None = None,
        provider: str | None = None,
        usage: JsonObject | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        return self._record(
            kind="span",
            name=name,
            span_type=span_type,
            input=input,
            output=output,
            metadata=metadata,
            tags=tags,
            model=model,
            provider=provider,
            usage=usage,
        )

    def _record(
        self,
        *,
        kind: str,
        name: str,
        span_type: str,
        input: JsonObject | None = None,
        output: JsonObject | None = None,
        metadata: JsonObject | None = None,
        tags: list[str] | None = None,
        model: str | None = None,
        provider: str | None = None,
        usage: JsonObject | None = None,
    ) -> RecordingTraceSpan:
        record = RecordedSpan(
            kind=kind,
            name=name,
            span_type=span_type,
            input=sanitize_json(input),
            output=sanitize_json(output),
            metadata=sanitize_json(metadata),
            tags=list(tags or []),
            model=model,
            provider=provider,
            usage=sanitize_json(usage),
        )
        self.records.append(record)
        return RecordingTraceSpan(record)


_current_trace_id: ContextVar[str | None] = ContextVar("opik_trace_id", default=None)
_current_span_id: ContextVar[str | None] = ContextVar("opik_span_id", default=None)


class OpikTracer:
    def __init__(
        self,
        settings: OpikSettings,
        *,
        opik_module: Any | None = None,
        client: Any | None = None,
    ) -> None:
        if not settings.enabled:
            raise ValueError("OpikTracer requires enabled Opik settings")

        if client is not None:
            self._client = client
        else:
            if opik_module is None:
                _check_opik_host(settings.url)
                try:
                    import opik as opik_module
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "OPIK_ENABLED=true but the opik package is not installed"
                    ) from exc
            self._client = opik_module.Opik(
                project_name=settings.project_name,
                workspace=settings.workspace or None,
                host=settings.url,
                api_key=settings.api_key or None,
                batching=False,
            )

        self._project_name = settings.project_name

    def trace(
        self,
        name: str,
        *,
        input: JsonObject | None = None,
        metadata: JsonObject | None = None,
        tags: list[str] | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        return _OpikContext(
            client=self._client,
            project_name=self._project_name,
            kind="trace",
            name=name,
            span_type="general",
            input=input,
            metadata=metadata,
            tags=tags,
        )

    def span(
        self,
        name: str,
        *,
        span_type: str = "general",
        input: JsonObject | None = None,
        output: JsonObject | None = None,
        metadata: JsonObject | None = None,
        tags: list[str] | None = None,
        model: str | None = None,
        provider: str | None = None,
        usage: JsonObject | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        return _OpikContext(
            client=self._client,
            project_name=self._project_name,
            kind="span",
            name=name,
            span_type=span_type,
            input=input,
            output=output,
            metadata=metadata,
            tags=tags,
            model=model,
            provider=provider,
            usage=usage,
        )


class _OpikContext:
    def __init__(
        self,
        *,
        client: Any,
        project_name: str,
        kind: str,
        name: str,
        span_type: str,
        input: JsonObject | None,
        metadata: JsonObject | None,
        tags: list[str] | None,
        output: JsonObject | None = None,
        model: str | None = None,
        provider: str | None = None,
        usage: JsonObject | None = None,
    ) -> None:
        self._client = client
        self._project_name = project_name
        self._kind = kind
        self._name = name
        self._span_type = span_type
        self._input = sanitize_json(input)
        self._output = sanitize_json(output)
        self._metadata = sanitize_json(metadata) or {}
        self._tags = list(tags or [])
        self._model = model
        self._provider = provider
        self._usage = sanitize_json(usage)
        self._error: JsonObject | None = None
        self._id: str | None = None
        self._trace_id: str | None = None
        self._parent_span_id: str | None = None
        self._trace_token: Any = None
        self._span_token: Any = None

    def __enter__(self) -> TraceSpan:
        start_time = _utc_now()
        if self._kind == "trace":
            trace = self._client.trace(
                name=self._name,
                start_time=start_time,
                input=self._input,
                metadata=self._metadata,
                tags=self._tags,
                project_name=self._project_name,
            )
            self._trace_id = _object_id(trace)
            self._id = self._trace_id
            self._trace_token = _current_trace_id.set(self._trace_id)
            self._span_token = _current_span_id.set(None)
            return self

        self._trace_id = _current_trace_id.get()
        self._parent_span_id = _current_span_id.get()
        span = self._client.span(
            trace_id=self._trace_id,
            parent_span_id=self._parent_span_id,
            name=self._name,
            type=self._span_type,
            start_time=start_time,
            input=self._input,
            output=self._output,
            metadata=self._metadata,
            tags=self._tags,
            project_name=self._project_name,
            model=self._model,
            provider=self._provider,
            usage=self._usage,
        )
        self._id = _object_id(span)
        self._span_token = _current_span_id.set(self._id)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None:
            self.set_error(str(exc), type_=type(exc).__name__)

        end_time = _utc_now()
        if self._kind == "trace":
            if self._trace_id is not None:
                self._client.update_trace(
                    trace_id=self._trace_id,
                    project_name=self._project_name,
                    end_time=end_time,
                    metadata=self._metadata,
                    output=self._output,
                    error_info=self._error,
                )
            if self._span_token is not None:
                _current_span_id.reset(self._span_token)
            if self._trace_token is not None:
                _current_trace_id.reset(self._trace_token)
            flush = getattr(self._client, "flush", None)
            if callable(flush):
                flush()
            return None

        if self._id is not None:
            self._client.update_span(
                id=self._id,
                trace_id=self._trace_id,
                parent_span_id=self._parent_span_id,
                project_name=self._project_name,
                end_time=end_time,
                metadata=self._metadata,
                output=self._output,
                usage=self._usage,
                model=self._model,
                provider=self._provider,
                error_info=self._error,
            )
        if self._span_token is not None:
            _current_span_id.reset(self._span_token)
        return None

    def set_output(self, output: JsonObject) -> None:
        self._output = sanitize_json(output)

    def set_metadata(self, metadata: JsonObject) -> None:
        self._metadata = {**self._metadata, **(sanitize_json(metadata) or {})}

    def set_usage(self, usage: JsonObject) -> None:
        self._usage = sanitize_json(usage)

    def set_error(self, message: str, *, type_: str | None = None) -> None:
        self._error = sanitize_error(message, type_=type_)


def build_tracer(settings: OpikSettings) -> Tracer:
    if not settings.enabled:
        return NoopTracer()
    return OpikTracer(settings)


def _check_opik_host(url: str) -> None:
    request = Request(url, method="HEAD")
    try:
        with urlopen(request, timeout=3):
            return
    except HTTPError:
        return
    except (OSError, URLError) as exc:
        raise RuntimeError(
            f"OPIK_ENABLED=true but Opik host is unreachable: {url}"
        ) from exc


def sanitize_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value_for_key(str(key), nested_value)
            for key, nested_value in list(value.items())[:MAX_COLLECTION_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in list(value)[:MAX_COLLECTION_ITEMS]]
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, (int, float, bool)):
        return value
    return _truncate(str(value))


def sanitize_tool_arguments(arguments: JsonObject) -> JsonObject:
    sanitized: JsonObject = {}
    for key, value in arguments.items():
        key_text = str(key)
        if _is_secret_key(key_text):
            sanitized[key_text] = "[redacted]"
        elif _is_content_key(key_text) and isinstance(value, str):
            sanitized[key_text] = summarize_text(value)
        else:
            sanitized[key_text] = sanitize_json(value)
    return sanitized


def summarize_text(value: str, *, preview_chars: int = MAX_PREVIEW_CHARS) -> JsonObject:
    return {
        "length": len(value),
        "preview": _truncate(value, limit=preview_chars),
    }


def summarize_messages(messages: list[JsonObject]) -> JsonObject:
    total_chars = sum(len(str(message.get("content") or "")) for message in messages)
    roles: list[str] = []
    for message in messages[:MAX_COLLECTION_ITEMS]:
        role = message.get("role")
        roles.append(str(role) if role is not None else "unknown")
    return {
        "count": len(messages),
        "roles": roles,
        "content_chars": total_chars,
    }


def summarize_tools(tools: list[JsonObject]) -> JsonObject:
    names: list[str] = []
    for tool in tools[:MAX_COLLECTION_ITEMS]:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return {"count": len(tools), "names": names}


def summarize_tool_calls(tool_calls: list[JsonObject] | tuple[Any, ...]) -> JsonObject:
    names: list[str] = []
    for tool_call in list(tool_calls)[:MAX_COLLECTION_ITEMS]:
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
        else:
            name = getattr(tool_call, "name", None)
        if isinstance(name, str):
            names.append(name)
    return {"count": len(tool_calls), "names": names}


def sanitize_error(message: str, *, type_: str | None = None) -> JsonObject:
    error: JsonObject = {"message": _truncate(message)}
    if type_:
        error["type"] = type_
    return error


def _sanitize_value_for_key(key: str, value: Any) -> Any:
    if _is_secret_key(key):
        return "[redacted]"
    if _is_content_key(key) and isinstance(value, str):
        return summarize_text(value)
    return sanitize_json(value)


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    if lowered.endswith("_tokens") or lowered in {"tokens", "total_tokens"}:
        return False
    return any(part in lowered for part in SECRET_KEY_PARTS)


def _is_content_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in CONTENT_KEY_PARTS)


def _truncate(value: str, *, limit: int = MAX_PREVIEW_CHARS) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _object_id(value: Any) -> str | None:
    for key in ("id", "trace_id", "span_id"):
        object_value = getattr(value, key, None)
        if isinstance(object_value, str):
            return object_value
    if isinstance(value, dict):
        for key in ("id", "trace_id", "span_id"):
            object_value = value.get(key)
            if isinstance(object_value, str):
                return object_value
    return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
