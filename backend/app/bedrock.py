"""Amazon Bedrock Converse adapter for provider-neutral structured generation."""

import json

from .llm import LLMRequest, LLMResponse, ProviderFailure, TokenUsage


_MAX_TIMEOUT_MS = 120_000
_TIMEOUT_CODES = {"ModelTimeoutException", "RequestTimeout", "RequestTimeoutException"}


def _usage(response: dict) -> TokenUsage | None:
    metadata = response.get("usage")
    if not isinstance(metadata, dict):
        return None
    values = {}
    for target, source in (("input_tokens", "inputTokens"),
                           ("output_tokens", "outputTokens"),
                           ("total_tokens", "totalTokens")):
        value = metadata.get(source)
        values[target] = value if type(value) is int and 0 <= value <= 10**9 else None
    return TokenUsage(**values)


def _failure_code(error: Exception) -> str | None:
    """Classify SDK failures without exposing their possibly sensitive messages."""
    if isinstance(error, TimeoutError):
        return "TIMEOUT"
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        details = response.get("Error")
        if isinstance(details, dict) and isinstance(details.get("Code"), str):
            return "TIMEOUT" if details["Code"] in _TIMEOUT_CODES else "UNAVAILABLE"
    # Botocore transport, credentials, and endpoint errors have no service response.
    if type(error).__module__.startswith("botocore."):
        return "TIMEOUT" if type(error).__name__ in (
            "ConnectTimeoutError", "ReadTimeoutError") else "UNAVAILABLE"
    return None


class BedrockProvider:
    """Synchronous Converse client. Inject ``client`` for offline use."""

    def __init__(self, *, model_id: str, region_name: str | None = None,
                 client=None, timeout_ms: int = 30_000) -> None:
        if (not isinstance(model_id, str) or not 1 <= len(model_id) <= 2048
                or any(character.isspace() or not character.isprintable()
                       for character in model_id)):
            raise ValueError("model_id must be a nonblank Bedrock model identifier")
        if region_name is not None and (not isinstance(region_name, str)
                                        or not region_name.strip()):
            raise ValueError("region_name must be a nonblank region")
        if type(timeout_ms) is not int or not 1 <= timeout_ms <= _MAX_TIMEOUT_MS:
            raise ValueError("timeout_ms must be between 1 and 120000")
        self.model_id = model_id
        self._owns_client = client is None
        if client is None:
            # One attempt includes the initial request: no SDK replay after an
            # ambiguous timeout or transport failure. This import is intentionally
            # deferred so injected clients need neither SDK nor AWS credentials.
            import boto3
            from botocore.config import Config

            config = Config(connect_timeout=timeout_ms / 1000,
                            read_timeout=timeout_ms / 1000,
                            retries={"total_max_attempts": 1})
            self._client = boto3.client("bedrock-runtime", region_name=region_name,
                                       config=config)
        else:
            self._client = client

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def generate(self, request: LLMRequest, *, response_schema: dict) -> LLMResponse:
        request = LLMRequest.model_validate(request)
        if not isinstance(response_schema, dict):
            raise TypeError("response_schema must be a dictionary")
        instruction = (
            "Return only one JSON object matching this JSON Schema. "
            "Do not include Markdown, explanations, or extra fields. "
            "You are a JSON decision generator; no provider-native tools are available. "
            "Any tool operations mentioned in the conversation must appear only as JSON "
            "data in the requested schema. The application validates and executes them. "
            "Do not execute operations or invent their results. JSON Schema: "
            + json.dumps(response_schema, separators=(",", ":"))
        )
        system = []
        messages = []
        for message in request.messages:
            if message.role == "system":
                if messages:
                    raise ValueError("System messages must precede conversation messages")
                system.append({"text": message.content})
            else:
                messages.append({"role": message.role,
                                 "content": [{"text": message.content}]})
        if not messages:
            raise ValueError("Converse requires a user or assistant message")
        system.append({"text": instruction})
        try:
            response = self._client.converse(
                modelId=self.model_id, messages=messages, system=system,
                inferenceConfig={"maxTokens": request.max_output_tokens})
        except Exception as error:
            code = _failure_code(error)
            if code is None:
                raise
            raise ProviderFailure(code) from None

        if not isinstance(response, dict):
            return LLMResponse(text="", finish_reason="stop")
        usage = _usage(response)
        reason = response.get("stopReason")
        if reason in ("max_tokens", "model_context_window_exceeded"):
            return LLMResponse(text="", finish_reason="length", usage=usage)
        if reason in ("content_filtered", "guardrail_intervened"):
            return LLMResponse(text="", finish_reason="refusal", usage=usage)
        invalid = LLMResponse(text="", finish_reason="stop", usage=usage)
        if reason not in ("end_turn", "stop_sequence"):
            return invalid
        output = response.get("output")
        message = output.get("message") if isinstance(output, dict) else None
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return invalid
        blocks = message.get("content")
        if not isinstance(blocks, list):
            return invalid
        text = []
        for block in blocks:
            if not isinstance(block, dict):
                return invalid
            if "reasoningContent" in block and len(block) == 1:
                continue  # Never expose model reasoning as final answer text.
            if set(block) != {"text"} or not isinstance(block["text"], str):
                return invalid  # Tool use and unknown content fail closed.
            text.append(block["text"])
        return LLMResponse(text="".join(text), finish_reason="stop", usage=usage)
