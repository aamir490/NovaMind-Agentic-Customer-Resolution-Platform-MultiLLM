"""Provider-neutral structured generation. No domain, tool, or API dependencies."""

from typing import Annotated, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from .guardrails import GuardrailError, MAX_PROMPT_CHARS, validate_response_json
from .observability import annotate, annotate_schema_validation, observed


class LLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


Nonblank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class LLMMessage(LLMModel):
    role: Literal["system", "user", "assistant"]
    content: Nonblank


class LLMRequest(LLMModel):
    messages: tuple[LLMMessage, ...] = Field(min_length=1)
    max_output_tokens: int = Field(default=512, strict=True, ge=1, le=32768)


class TokenUsage(LLMModel):
    input_tokens: int | None = Field(default=None, strict=True, ge=0, le=10**9)
    output_tokens: int | None = Field(default=None, strict=True, ge=0, le=10**9)
    total_tokens: int | None = Field(default=None, strict=True, ge=0, le=10**9)


class LLMResponse(LLMModel):
    text: str
    finish_reason: Literal["stop", "length", "refusal"]
    usage: TokenUsage | None = None


FailureCode = Literal["TIMEOUT", "UNAVAILABLE", "REFUSED", "INCOMPLETE", "INVALID_RESPONSE", "INPUT_LIMIT"]


class LLMFailure(LLMModel):
    ok: Literal[False] = False
    code: FailureCode
    message: str


Output = TypeVar("Output", bound=BaseModel)


class LLMSuccess(LLMModel, Generic[Output]):
    ok: Literal[True] = True
    data: Output


class ProviderFailure(Exception):
    """Expected adapter failure; raw provider messages are not returned to callers."""

    def __init__(self, code: Literal["TIMEOUT", "UNAVAILABLE"]) -> None:
        if code not in ("TIMEOUT", "UNAVAILABLE"):
            raise ValueError("Unsupported provider failure code")
        self.code = code
        super().__init__(code)


class LLMProvider(Protocol):
    def generate(self, request: LLMRequest, *, response_schema: dict) -> LLMResponse:
        """Return a completion or raise ProviderFailure for expected transport failures."""
        ...


class StructuredLLM:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    @observed("llm")
    def generate(self, request: LLMRequest, output_model: type[Output]) -> LLMSuccess[Output] | LLMFailure:
        # Invalid caller input is a programming error, distinct from generation failure.
        request = LLMRequest.model_validate(request)
        provider_type = (type(self._provider).__module__, type(self._provider).__name__)
        provider = {("backend.app.gemini", "GeminiProvider"): "gemini",
                    (__name__, "FakeLLMProvider"): "fake"}.get(provider_type, "other")
        annotate(provider_called=False, provider=provider, output_token_limit=request.max_output_tokens,
                 input_tokens=None, output_tokens=None, total_tokens=None)
        if sum(len(message.content) for message in request.messages) > MAX_PROMPT_CHARS:
            return LLMFailure(code="INPUT_LIMIT", message="Model context exceeds the local safety limit")
        schema = output_model.model_json_schema()
        try:
            annotate(provider_called=True)
            raw = self._provider.generate(request, response_schema=schema)
        except ProviderFailure as error:
            return LLMFailure(code=error.code, message="Provider could not complete the request")
        try:
            response = LLMResponse.model_validate(raw)
        except ValidationError:
            annotate(invalid_response_stage="provider_envelope")
            return LLMFailure(code="INVALID_RESPONSE", message="Invalid provider response envelope")
        if response.usage is not None:
            annotate(**response.usage.model_dump())
        if response.finish_reason == "refusal":
            return LLMFailure(code="REFUSED", message="Provider declined the request")
        if response.finish_reason == "length":
            return LLMFailure(code="INCOMPLETE", message="Provider output was truncated")
        try:
            validate_response_json(response.text)
        except (ValidationError, GuardrailError):
            annotate(invalid_response_stage="json_guardrail")
            return LLMFailure(code="INVALID_RESPONSE", message="Output does not match the requested JSON model")
        try:
            data = output_model.model_validate_json(response.text, strict=True)
        except (ValidationError, GuardrailError) as error:
            annotate(invalid_response_stage="output_schema")
            if isinstance(error, ValidationError):
                annotate_schema_validation(error)
            return LLMFailure(code="INVALID_RESPONSE", message="Output does not match the requested JSON model")
        return LLMSuccess[output_model](data=data)


class FakeLLMProvider:
    """Test fixture only: returns the configured result on every call, ignoring prompts."""

    def __init__(self, outcome: LLMResponse | ProviderFailure) -> None:
        if not isinstance(outcome, (LLMResponse, ProviderFailure)):
            raise TypeError("Fake outcome must be an LLMResponse or ProviderFailure")
        self._outcome = outcome

    def generate(self, request: LLMRequest, *, response_schema: dict) -> LLMResponse:
        if isinstance(self._outcome, ProviderFailure):
            raise ProviderFailure(self._outcome.code)
        return self._outcome.model_copy(deep=True)
