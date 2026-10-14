"""Gemini Developer API adapter; deliberately independent of business/tools/APIs."""

import os

import httpx
from google import genai
from google.genai import errors, types

from .llm import LLMRequest, LLMResponse, ProviderFailure, TokenUsage
from .observability import annotate


def _usage(response):
    """Optional SDK metadata; missing/invalid counts must not change completion behavior."""
    try:
        metadata = getattr(response, "usage_metadata", None)
        values = {}
        for target, source in (("input_tokens", "prompt_token_count"), ("output_tokens", "candidates_token_count"),
                               ("total_tokens", "total_token_count")):
            value = getattr(metadata, source, None)
            values[target] = value if type(value) is int and 0 <= value <= 10**9 else None
        return TokenUsage(**values)
    except Exception:
        return None


class GeminiProvider:
    def __init__(self, *, model: str | None = None, client=None, timeout_ms: int = 30000) -> None:
        self.model = model if model is not None else os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Gemini model must be nonblank")
        self.model = self.model.strip()
        if type(timeout_ms) is not int or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        self._owns_client = client is None
        if client is None:
            key = os.getenv("GEMINI_API_KEY", "").strip()
            if not key:
                raise ValueError("GEMINI_API_KEY is required")
            client = genai.Client(
                api_key=key, vertexai=False,
                http_options=types.HttpOptions(
                    timeout=timeout_ms,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        self._client = client

    def close(self) -> None:
        """Injected clients remain the responsibility of their owner."""
        if self._owns_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def generate(self, request: LLMRequest, *, response_schema: dict) -> LLMResponse:
        request = LLMRequest.model_validate(request)
        system = []
        contents = []
        for message in request.messages:
            if message.role == "system":
                # Hoisting a later instruction would silently change conversation semantics.
                if contents:
                    raise ValueError("Gemini requires system messages before conversation messages")
                system.append(message.content)
            else:
                contents.append(types.Content(
                    role="model" if message.role == "assistant" else "user",
                    parts=[types.Part.from_text(text=message.content)],
                ))
        if not contents:
            raise ValueError("Gemini requires at least one conversation message")
        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(system) or None,
            response_mime_type="application/json", response_json_schema=response_schema,
            max_output_tokens=request.max_output_tokens, candidate_count=1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = self._client.models.generate_content(model=self.model, contents=contents, config=config)
        except (httpx.TimeoutException, TimeoutError):
            raise ProviderFailure("TIMEOUT") from None
        except errors.APIError as error:
            annotate(provider_status_code=error.code)
            raise ProviderFailure("TIMEOUT" if error.code in (408, 504) else "UNAVAILABLE") from None
        except httpx.TransportError:
            raise ProviderFailure("UNAVAILABLE") from None

        feedback = response.prompt_feedback
        if feedback and feedback.block_reason and feedback.block_reason != "BLOCKED_REASON_UNSPECIFIED":
            return LLMResponse(text="", finish_reason="refusal", usage=_usage(response))
        candidates = response.candidates or []
        if len(candidates) != 1:
            return LLMResponse(text="", finish_reason="stop", usage=_usage(response))  # Phase 6 rejects empty JSON.
        candidate = candidates[0]
        reason = candidate.finish_reason
        if reason == "MAX_TOKENS":
            return LLMResponse(text="", finish_reason="length", usage=_usage(response))
        if reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}:
            return LLMResponse(text="", finish_reason="refusal", usage=_usage(response))
        if reason != "STOP":
            return LLMResponse(text="", finish_reason="stop", usage=_usage(response))
        parts = candidate.content.parts if candidate.content else []
        text = []
        for part in parts or []:
            if part.thought:
                continue
            # Never interpret tool calls, code, or media as a completed text response.
            fields = part.model_dump(exclude_none=True)
            if set(fields) - {"text", "thought", "thought_signature"} or part.text is None:
                return LLMResponse(text="", finish_reason="stop", usage=_usage(response))
            text.append(part.text)
        return LLMResponse(text="".join(text), finish_reason="stop", usage=_usage(response))
