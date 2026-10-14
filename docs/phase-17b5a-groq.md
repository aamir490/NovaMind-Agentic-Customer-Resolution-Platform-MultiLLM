# Phase 17B.5a — Groq provider compatibility

`backend/app/groq.py` implements the existing `LLMProvider` protocol through
Groq's OpenAI-compatible `https://api.groq.com/openai/v1/chat/completions` endpoint.
It reuses installed HTTPX; no new SDK, package, endpoint, or fallback routing is added.
Gemini's adapter and existing configuration behavior are unchanged. Bedrock is not implemented.

## Schema compatibility

The Groq adapter uses `response_format={"type":"json_object"}` and supplies the
original requested JSON schema in an additional system instruction. Original
messages retain their roles, content, and order. It does not send the canonical
schema as a strict Groq `json_schema` request: Groq's closed-object requirement
cannot represent NovaMind's open-ended `ToolCall.arguments: dict[str, JsonValue]`.
See [Groq's structured output and JSON object documentation](https://console.groq.com/docs/structured-outputs).

This is prompt-guided schema adherence, not provider-enforced schema compliance.
The adapter returns only assistant `message.content`, without rewriting JSON,
removing fences, coercing arguments, or executing native tool calls. The unchanged
`StructuredLLM` checks the completion envelope, JSON guardrails (including duplicate
keys, size/depth, and nonfinite numbers), and strict Pydantic validation against the
original `AgentDecision`. Invalid output fails with `INVALID_RESPONSE`. There is no
adapter retry or alternate-provider fallback. `AgentDecision`, `ToolCall`, and
`Finish` need no modification because their original JSON representation is preserved.

### Native-call runtime compatibility

A user-run workflow returned HTTP 400: `Tool choice is none, but model called a tool`.
This confirms the model attempted provider-native calling, which Groq rejected.
The likely prompt trigger is the shared agent's instruction to return a tool call
alongside tool descriptions; the previous adapter instruction requested JSON without
explicitly distinguishing application decisions from provider-native calls. JSON
object mode alone did not prevent the observed attempt.

The Groq-only system instruction now explicitly requires ordinary JSON text in the
final assistant response. References to calling tools mean describing application
operations and arguments using the supplied schema; native calls, tool recipients,
execution, and invented results are prohibited. The request explicitly sets
`tool_choice="none"`, retaining Groq's existing no-tools default
([API reference](https://console.groq.com/docs/api-reference)). No `tools`, `functions`,
or validation-bypass option is sent. The original agent messages and schema remain
unchanged, as do JSON object mode, reasoning settings, and completion budget.

Native `tool_calls`/`function_call` responses are still rejected, never translated
or executed. Valid JSON decisions still pass through `StructuredLLM` before the
application dispatcher runs. This prompt clarification is not a hard model guarantee:
mocked tests verify the request and controlled execution path, but the effect on
the real model needs another single live runtime test. None was made for this fix.

## Runtime configuration

Inject credentials through the process environment using your existing secret
mechanism; do not place keys in repository files, shell history, frontend variables,
or build arguments. Bootstrap does not load `.env` files.

- `NOVAMIND_LLM_PROVIDER=groq`: explicitly select Groq in the existing container
  bootstrap and `create_app(local_provider_factory=...)` wiring.
- `GROQ_API_KEY`: required, nonblank, at most 4,096 printable ASCII characters with
  no whitespace. Invalid or missing configuration stops startup with a fixed error.
- `GROQ_MODEL`: optional model ID, default `openai/gpt-oss-20b`. A blank value is
  invalid. Direct `GroqProvider(model=...)` overrides this environment value.
- `NOVAMIND_LLM_PROVIDER=gemini`: explicitly select existing Gemini wiring, requiring
  both `GEMINI_API_KEY` and `GEMINI_MODEL`.
- With no selector, the previous Gemini behavior remains: neither Gemini variable
  means unavailable; partial/invalid Gemini configuration fails startup. Groq
  credentials alone never auto-enable Groq. With a selector, only the selected
  provider's configuration is used, even when both sets of credentials exist.

Unknown or blank selector values stop startup. Authentication is validated first
and is still required; provider configuration grants no access. Startup makes no
inference call. One shared client is closed after existing workflow workers drain.

The optional `compose.groq.yaml` overlay sets the selector, injects the key, supplies
the default model, and attaches only the backend to the existing outbound `edge`
network, following the Gemini overlay pattern. Use one provider overlay at a time.
After securely injecting `GROQ_API_KEY` and configuring `NOVAMIND_AUTH_SOURCE`:

```powershell
docker compose --env-file .env.container.example -f compose.yaml -f compose.auth.yaml -f compose.groq.yaml config --quiet
docker compose --env-file .env.container.example -f compose.yaml -f compose.auth.yaml -f compose.groq.yaml up --build -d --wait
```

These are operator instructions, not commands executed during implementation.
Do not publish rendered Compose configuration or container inspection output that
could contain the key. Recreating the backend loses its existing in-memory state.

## Tokens, failures, and retained boundaries

`LLMRequest.max_output_tokens` maps exactly to `max_completion_tokens`; the adapter
never silently increases the caller's budget. GPT-OSS reasoning consumes that same
budget. The existing agent uses 2,048 tokens per decision. For direct development
requests, explicitly use a larger budget such as 8,192 when needed; the generic
`LLMRequest` default of 512 can be too small for reasoning workloads. GPT-OSS 20B/120B
requests set `reasoning_effort="low"` and `reasoning_format="hidden"` (required by
Groq when `response_format=json_object` is active; `reasoning_format` and
`include_reasoning` are mutually exclusive, so `include_reasoning` is omitted). Other model IDs
do not receive those model-specific parameters. Reasoning fields are never used as
the final response. See [Groq reasoning documentation](https://console.groq.com/docs/reasoning).

HTTP timeout defaults to 30 seconds per operation (`timeout_ms` constructor option).
There are no automatic retries, redirects, environment proxies, streaming, or native
tool calls. Injected HTTP clients are caller-owned; the Groq key is still required
to construct the authenticated request. All HTTP statuses outside 2xx fail closed:
408/504 and transport timeouts map to `TIMEOUT`, other API/transport errors to
`UNAVAILABLE`. Refusals map to `REFUSED`; length exhaustion maps to `INCOMPLETE`.
Raw error bodies, reasoning, and keys are not included in application errors or
telemetry. Unexpected programming defects propagate as before.

For runtime debugging, non-2xx responses also emit a warning through the backend
logger `novamind.groq`: `Groq HTTP <status>: <sanitized message>`. Inspect backend
stderr/container logs (for example, `docker compose logs backend`). Only the JSON
`error.message` field is considered, with a 64 KiB error-body parsing limit and a
512-character message limit. Configured credentials, recognizable credential
patterns, and exact request-message echoes are redacted before truncation; control
characters are removed. Missing, malformed, or oversized error bodies produce a
fixed fallback instead of raw body text. Headers and `failed_generation` are never
logged. This is a best-effort diagnostic intended for trusted backend operators;
do not publish backend logs. Logging failures cannot replace the existing provider
failure. The logging addition does not change request construction, API/telemetry
response contracts, or agent behavior.

No business rules, ownership checks, tools, graph/HITL flow, proposals, RAG, memory,
audit contracts, or customer portal code changed. Existing telemetry labels Groq
as `other`; the HTTP local-factory label remains `local_scripted`. These compatibility
labels are unchanged deliberately, and do not identify the actual provider.

## Verification and limits

Focused tests mock external HTTP with `httpx.MockTransport`, including a real
Groq adapter call through the existing authenticated run route. They cover request
construction, environment validation, explicit selection, canonical decisions,
JSON guardrails, failures, reasoning exclusion, budgets, and client cleanup.
Existing Gemini, structured LLM, and diagnostic tests provide regression coverage.

```powershell
./backend/.venv/Scripts/python.exe -B -m unittest tests.test_groq tests.test_container_groq tests.test_llm tests.test_gemini tests.test_container_gemini tests.test_llm_diagnostics tests.test_gemini_diagnostics -v
```

Focused result: **74 tests passed**, including 23 new Groq adapter/bootstrap tests.

Before the runtime-debugging follow-up, broader backend regression also passed:
**438 tests**, using:

```powershell
./backend/.venv/Scripts/python.exe -B -m unittest discover -s tests -p 'test_*.py' -q
```

`git diff --check` passed. The test runner emitted asyncio slow-task diagnostics,
but no test failures. No frontend build or unrelated package installation ran.

The runtime-debugging follow-up adds seven focused logging/redaction tests.
**81 tests passed** with the following command; broader regression was not repeated:

```powershell
./backend/.venv/Scripts/python.exe -B -m unittest tests.test_groq_diagnostics tests.test_groq tests.test_container_groq tests.test_llm tests.test_gemini tests.test_container_gemini tests.test_llm_diagnostics tests.test_gemini_diagnostics -q
```

No real Groq inference or Docker runtime test was performed by these implementation
tests. The user's reported runtime evidence is noted above. Live JSON adherence
after the compatibility fix still requires an explicit real runtime test. Mocked
integration does not establish live end-to-end verification or production readiness.

The native-call compatibility fix passed **83 focused tests** using the same focused
command above. It adds regression tests for rejecting native tool responses and
retaining sanitized handling of the exact reported HTTP 400. The authenticated
runtime test now supplies a JSON `get_order` decision followed by a final decision,
verifies execution through NovaMind's dispatcher, and checks that both Groq requests
preserve the canonical schema and agent instructions while prohibiting native calls.
