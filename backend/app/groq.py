"""Groq OpenAI-compatible JSON adapter; no business or tool contract dependencies."""

import json
import logging
import os
import re
import unicodedata

import httpx

from .llm import LLMRequest, LLMResponse, ProviderFailure, TokenUsage
from .observability import annotate


_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_GPT_OSS_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
_LOGGER = logging.getLogger("novamind.groq")
_ERROR_MESSAGE_LIMIT = 512


def _log_http_error(response, *, api_key, messages):
    """Best-effort server diagnostics only; never change the public failure contract."""
    try:
        message = None
        # Never dump an HTML/proxy response, headers, or the rest of the error body.
        if len(response.content) <= 65536:
            try:
                payload = response.json()
            except (ValueError, RecursionError):
                payload = None
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                message = error["message"]
        if message:
            # Redact before truncation, including JSON-escaped echoes of inputs.
            private = [api_key, *(item["content"] for item in messages)]
            for value in sorted(private, key=len, reverse=True):
                message = message.replace(json.dumps(value)[1:-1], "[redacted]")
                message = message.replace(value, "[redacted]")
            message = unicodedata.normalize("NFKC", message)
            message = "".join(character for character in message
                              if not unicodedata.category(character).startswith("C"))
            message = message.replace(api_key, "[redacted]")
            message = re.sub(r"(?i)\b(?:bearer|basic)\s+[^\s,;\"']+", "[redacted authorization]", message)
            message = re.sub(r"(?i)\b(?:gsk_|sk-)[A-Za-z0-9_-]+", "[redacted credential]", message)
            message = re.sub(r"(?i)\b(?:api[_-]?key|authorization|access[_-]?token)\b[\"']?\s*[:=]\s*"
                             r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)", "[redacted credential]", message)
            message = " ".join(message.split())
            if len(message) > _ERROR_MESSAGE_LIMIT:
                message = message[:_ERROR_MESSAGE_LIMIT - 3] + "..."
        _LOGGER.warning("Groq HTTP %s: %s", response.status_code,
                        message or "Provider error message unavailable")
    except Exception:
        pass  # Diagnostic parsing/formatting/sink failures must not alter execution.


def _usage(payload):
    metadata = payload.get("usage")
    if not isinstance(metadata, dict):
        return None
    values = {}
    for target, source in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
                           ("total_tokens", "total_tokens")):
        value = metadata.get(source)
        values[target] = value if type(value) is int and 0 <= value <= 10**9 else None
    return TokenUsage(**values)


class GroqProvider:
    def __init__(self, *, model: str | None = None, client=None, timeout_ms: int = 30000) -> None:
        self.model = model if model is not None else os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        if not isinstance(self.model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", self.model):
            raise ValueError("Groq model must be a nonblank model ID")
        if type(timeout_ms) is not int or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        key = os.getenv("GROQ_API_KEY", "")
        if (not key or len(key) > 4096 or not key.isascii()
                or any(character.isspace() or not character.isprintable() for character in key)):
            raise ValueError("GROQ_API_KEY is required and must be a valid nonblank token")
        self._key = key
        self._timeout = timeout_ms / 1000
        self._owns_client = client is None
        # HTTPX defaults to no transport retries. Do not follow redirects with credentials.
        self._client = client if client is not None else httpx.Client(
            timeout=self._timeout, follow_redirects=False, trust_env=False)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def generate(self, request: LLMRequest, *, response_schema: dict) -> LLMResponse:
        request = LLMRequest.model_validate(request)
        # Strict Groq schemas cannot represent open-ended argument objects. JSON
        # object mode plus the ORIGINAL schema in the prompt preserves that data.
        # StructuredLLM remains the only authority for parsing/guardrails/validation.
        compatibility_instruction = (
            "Return only one JSON object matching this JSON Schema. "
            "You are a JSON decision generator, not a native tool-calling assistant. "
            "No provider-native tools or functions are available. Never emit native tool calls, "
            "function calls, or messages addressed to a tool recipient. "
            "Any instructions to use or call tools in this conversation refer ONLY to "
            "application-managed operations described as JSON data in the supplied schema. "
            "Express the selected operation and its arguments only inside that JSON object, "
            "as ordinary text in your final assistant response. "
            "The application validates the decision and controls execution; do not execute "
            "the operation or invent its result. "
            "Do not include Markdown, explanations, or extra fields. JSON Schema: "
            + json.dumps(response_schema, separators=(",", ":"))
        )
        if self.model in _GPT_OSS_MODELS:
            # Groq documentation recommends avoiding system prompts for reasoning
            # models (openai/gpt-oss-*) and placing instructions in the user message
            # instead.  Injecting the compatibility instruction as the first user
            # message avoids the "Tool choice is none, but model called a tool" HTTP
            # 400 that the system-prompt path triggers with these models.
            instruction_message = {"role": "user", "content": compatibility_instruction}
            messages = [instruction_message,
                        *(message.model_dump() for message in request.messages)]
        else:
            messages = [{"role": "system", "content": compatibility_instruction},
                        *(message.model_dump() for message in request.messages)]
        body = {"model": self.model, "messages": messages,
                "response_format": {"type": "json_object"}, "stream": False, "n": 1,
                "max_completion_tokens": request.max_output_tokens}
        # tool_choice:"none" prevents standard chat models from emitting native tool
        # calls. openai/gpt-oss-* reasoning models on Groq reject this parameter
        # (HTTP 400) when no native tools are registered, so it is omitted for them.
        # The compatibility instruction and json_object format still govern output.
        if self.model not in _GPT_OSS_MODELS:
            body["tool_choice"] = "none"
        if self.model in _GPT_OSS_MODELS:
            # reasoning_format:"hidden" is required when json_object mode is active
            # (Groq docs: format must be "parsed" or "hidden" with JSON mode/tools).
            # include_reasoning is mutually exclusive with reasoning_format per docs,
            # so it is omitted. Reasoning is suppressed via reasoning_format instead.
            body.update(reasoning_effort="low", reasoning_format="hidden")
        try:
            response = self._client.post(_ENDPOINT, json=body,
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=self._timeout, follow_redirects=False)
        except (httpx.TimeoutException, TimeoutError):
            raise ProviderFailure("TIMEOUT") from None
        except httpx.TransportError:
            raise ProviderFailure("UNAVAILABLE") from None
        if not response.is_success:
            annotate(provider_status_code=response.status_code)
            _log_http_error(response, api_key=self._key, messages=messages)
            raise ProviderFailure("TIMEOUT" if response.status_code in (408, 504) else "UNAVAILABLE") from None
        try:
            payload = response.json()
        except (ValueError, RecursionError):
            return LLMResponse(text="", finish_reason="stop")
        if not isinstance(payload, dict):
            return LLMResponse(text="", finish_reason="stop")
        usage = _usage(payload)
        invalid = LLMResponse(text="", finish_reason="stop", usage=usage)
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            return invalid
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            return LLMResponse(text="", finish_reason="length", usage=usage)
        if choice.get("finish_reason") == "content_filter":
            return LLMResponse(text="", finish_reason="refusal", usage=usage)
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return invalid
        if message.get("refusal"):
            return LLMResponse(text="", finish_reason="refusal", usage=usage)
        if (choice.get("finish_reason") != "stop" or message.get("tool_calls")
                or message.get("function_call") or not isinstance(message.get("content"), str)):
            return invalid
        # Do not repair JSON, unwrap fences, execute tools, or include message.reasoning.
        return LLMResponse(text=message["content"], finish_reason="stop", usage=usage)
