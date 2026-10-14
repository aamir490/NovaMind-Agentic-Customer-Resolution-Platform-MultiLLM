"""Groq adapter tests use HTTPX MockTransport; no API key or network required."""

import json
import os
import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from backend.app.agent import AgentConfig, AgentDecision, Finish, ToolCall, agent_instructions
from backend.app.groq import GroqProvider
from backend.app.llm import LLMMessage, LLMRequest, ProviderFailure, StructuredLLM
from backend.app.main import create_app


KEY = "groq-test-only-not-a-real-key"
FINAL = '{"decision":{"kind":"final"}}'


def completion(text=FINAL, reason="stop", **message):
    return {"choices": [{"finish_reason": reason,
        "message": {"role": "assistant", "content": text, **message}}]}


class GroqTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"GROQ_API_KEY": KEY}, clear=True))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("No network")))
        self.enterContext(patch("socket.getaddrinfo", side_effect=AssertionError("No DNS")))
        self.calls = []
        self.response = httpx.Response(200, json=completion())
        self.error = None
        self.client = self.enterContext(httpx.Client(transport=httpx.MockTransport(self.handle)))
        self.provider = GroqProvider(client=self.client)
        self.request = LLMRequest(messages=(LLMMessage(role="user", content="Review fixture"),),
                                  max_output_tokens=AgentConfig().max_output_tokens)

    def handle(self, request):
        self.calls.append(request)
        if self.error:
            raise self.error
        return self.response

    def result(self):
        return StructuredLLM(self.provider).generate(self.request, AgentDecision)

    def test_authenticated_request_json_mode_original_schema_and_message_order(self):
        self.request = LLMRequest(messages=(
            LLMMessage(role="system", content="Keep business instructions"),
            LLMMessage(role="user", content="Question"),
            LLMMessage(role="assistant", content=FINAL),
            LLMMessage(role="user", content="Followup")), max_output_tokens=8192)
        schema = AgentDecision.model_json_schema()
        original = json.dumps(schema, sort_keys=True)
        result = self.result()
        self.assertIsInstance(result.data, AgentDecision)
        self.assertIsInstance(result.data.decision, Finish)
        request, = self.calls
        self.assertEqual(str(request.url), "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.headers["Authorization"], f"Bearer {KEY}")
        self.assertEqual(request.headers["Content-Type"], "application/json")
        self.assertEqual(request.extensions["timeout"]["read"], 30)
        body = json.loads(request.content)
        self.assertEqual(body["model"], "openai/gpt-oss-20b")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        # For GPT-OSS models the compatibility instruction is delivered as the
        # FIRST USER message (not a system message) per Groq's recommendation
        # for reasoning models — this is the fix for the HTTP 400
        # "Tool choice is none, but model called a tool" error.
        first = body["messages"][0]
        self.assertEqual(first["role"], "user",
            "GPT-OSS compatibility instruction must be a user-role message, not system")
        self.assertIn("JSON Schema:", first["content"])
        self.assertIn("JSON decision generator", first["content"])
        # Caller messages follow the injected instruction, unchanged.
        self.assertEqual(body["messages"][1:], [m.model_dump() for m in self.request.messages])
        prompted_schema = json.loads(first["content"].split("JSON Schema: ", 1)[1])
        self.assertEqual(prompted_schema, schema)
        self.assertEqual(json.dumps(AgentDecision.model_json_schema(), sort_keys=True), original)
        self.assertEqual(prompted_schema["$defs"]["ToolCall"]["properties"]["arguments"]["additionalProperties"],
                         {"$ref": "#/$defs/JsonValue"})
        self.assertEqual(body["max_completion_tokens"], 8192)
        self.assertEqual(body["reasoning_effort"], "low")
        # reasoning_format:"hidden" replaces include_reasoning for JSON mode compatibility.
        self.assertEqual(body["reasoning_format"], "hidden")
        self.assertNotIn("include_reasoning", body,
            "include_reasoning must be absent — mutually exclusive with reasoning_format")
        self.assertFalse(body["stream"])
        self.assertEqual(body["n"], 1)
        # GPT-OSS: tool_choice must be ABSENT — sending it triggers a Groq HTTP 400.
        self.assertNotIn("tool_choice", body,
            "GPT-OSS must not receive tool_choice; Groq rejects it for reasoning models")
        self.assertNotIn("tools", body)
        self.assertNotIn("functions", body)
        self.assertNotIn("disable_tool_validation", body)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn(KEY, request.url.query.decode())
        self.assertNotIn(KEY, request.content.decode())

    def test_non_gpt_oss_model_uses_system_message_for_compatibility_instruction(self):
        """Non-GPT-OSS models keep the original system-message delivery path."""
        self.provider = GroqProvider(client=self.client, model="meta-llama/llama-3-70b-8192")
        result = self.result()
        self.assertTrue(result.ok)
        body = json.loads(self.calls[-1].content)
        first = body["messages"][0]
        self.assertEqual(first["role"], "system",
            "Non-GPT-OSS models must still receive the instruction as a system message")
        self.assertIn("JSON Schema:", first["content"])
        # Caller messages follow unchanged.
        self.assertEqual(body["messages"][1:], [m.model_dump() for m in self.request.messages])
        # GPT-OSS-specific options must not appear.
        self.assertNotIn("reasoning_effort", body)
        self.assertNotIn("include_reasoning", body)

    def test_preserves_caller_completion_budget_including_existing_agent_default(self):
        for limit in (1, AgentConfig().max_output_tokens, 8192, 32768):
            with self.subTest(limit=limit):
                self.request = self.request.model_copy(update={"max_output_tokens": limit})
                self.assertTrue(self.result().ok)
                self.assertEqual(json.loads(self.calls[-1].content)["max_completion_tokens"], limit)
        self.assertEqual(AgentConfig().max_output_tokens, 2048)

    def test_environment_model_override_and_owned_client_cleanup(self):
        with patch.dict(os.environ, {"GROQ_MODEL": "configured-model"}), \
                patch("backend.app.groq.httpx.Client") as constructor:
            with GroqProvider(timeout_ms=1500) as provider:
                self.assertEqual(provider.model, "configured-model")
            constructor.assert_called_once_with(timeout=1.5, follow_redirects=False, trust_env=False)
            constructor.return_value.close.assert_called_once()
            self.assertEqual(GroqProvider(client=self.client, model="override").model, "override")
        self.provider.close()
        self.assertFalse(self.client.is_closed)

    def test_other_models_do_not_receive_gpt_oss_specific_options(self):
        self.provider = GroqProvider(client=self.client, model="other-model")
        self.assertTrue(self.result().ok)
        body = json.loads(self.calls[-1].content)
        self.assertNotIn("reasoning_effort", body)
        self.assertNotIn("include_reasoning", body)

    def test_missing_or_malformed_configuration_fails_before_client_construction(self):
        invalid = ({}, {"GROQ_API_KEY": ""}, {"GROQ_API_KEY": " SECRET "},
            {"GROQ_API_KEY": "SECRET\nVALUE"}, {"GROQ_API_KEY": "SECRET\x00"},
            {"GROQ_API_KEY": "é"}, {"GROQ_API_KEY": "x" * 4097},
            {"GROQ_API_KEY": KEY, "GROQ_MODEL": ""}, {"GROQ_API_KEY": KEY, "GROQ_MODEL": " "},
            {"GROQ_API_KEY": KEY, "GROQ_MODEL": "https://SECRET.invalid"})
        with patch("backend.app.groq.httpx.Client") as constructor:
            for environment in invalid:
                with self.subTest(fields=list(environment)), patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(ValueError) as caught:
                        GroqProvider()
                    self.assertNotIn("SECRET", str(caught.exception))
                    self.assertNotIn(KEY, str(caught.exception))
            constructor.assert_not_called()
        for kwargs in ({"model": " "}, {"model": 123}, {"timeout_ms": 0}, {"timeout_ms": True}):
            with self.subTest(fields=list(kwargs)), self.assertRaises(ValueError):
                GroqProvider(client=self.client, **kwargs)

    def test_invalid_request_fails_before_http(self):
        with self.assertRaises(ValidationError):
            self.provider.generate({"messages": []}, response_schema={})
        self.assertEqual(self.calls, [])

    def test_tool_decision_retains_open_nested_json_arguments(self):
        arguments = {"case_id": "fixture-id", "nested": {"any_key": [True, None, 3, 1.5, "text"]}}
        self.response = httpx.Response(200, json=completion(json.dumps({"decision": {
            "kind": "tool", "name": "get_case", "arguments": arguments}})))
        result = self.result()
        self.assertIsInstance(result.data.decision, ToolCall)
        self.assertEqual(result.data.decision.arguments, arguments)

    def test_native_tool_response_is_not_translated_into_an_agent_decision(self):
        native = [{"id": "call_fixture", "type": "function", "function": {
            "name": "get_case", "arguments": '{"case_id":"fixture-id"}'}}]
        for reason in ("tool_calls", "stop"):
            with self.subTest(reason=reason):
                self.calls.clear()
                self.response = httpx.Response(200, json=completion(FINAL, reason=reason, tool_calls=native))
                result = self.result()
                self.assertEqual(result.code, "INVALID_RESPONSE")
                self.assertNotIn("data", result.model_dump())
                self.assertEqual(len(self.calls), 1)
                # GPT-OSS: tool_choice must be absent regardless of native tool response.
                self.assertNotIn("tool_choice", json.loads(self.calls[0].content))

    def test_canonical_validation_rejects_wrong_discriminator_arguments_and_extra_fields(self):
        for text in ('{}', '[]', 'null', '{"decision":{"kind":"finish"}}',
            '{"decision":{"kind":"final","message":"extra"}}',
            '{"decision":{"kind":"tool","name":"get_case","arguments":"{}"}}',
            '{"decision":{"kind":"tool","name":"get_case"}}',
            '{"decision":{"kind":"tool","name":2,"arguments":{}}}',
            '{"decision":{"kind":"final"},"approve":true}'):
            with self.subTest(text=text):
                self.response = httpx.Response(200, json=completion(text))
                result = self.result()
                self.assertEqual(result.code, "INVALID_RESPONSE")
                self.assertNotIn("data", result.model_dump())

    def test_json_guardrails_reject_ambiguity_nonfinite_oversize_and_no_fence_repair(self):
        for text in ('not json', '```json\n' + FINAL + '\n```',
            '{"decision":{"kind":"final","kind":"final"}}',
            '{"decision":{"kind":"tool","name":"get_case","arguments":{"x":NaN}}}',
            '{"decision":{"kind":"tool","name":"get_case","arguments":{"x":' + '[' * 20 + '0' + ']' * 20 + '}}}',
            ' ' * 65537 + FINAL):
            self.response = httpx.Response(200, json=completion(text))
            self.assertEqual(self.result().code, "INVALID_RESPONSE")

    def test_refusal_and_length_override_valid_json(self):
        for reason, message, code in (("length", {}, "INCOMPLETE"),
                ("content_filter", {}, "REFUSED"), ("stop", {"refusal": "Private refusal"}, "REFUSED")):
            with self.subTest(reason=reason):
                self.response = httpx.Response(200, json=completion(reason=reason, **message))
                result = self.result()
                self.assertEqual(result.code, code)
                self.assertNotIn("Private", result.model_dump_json())

    def test_invalid_envelopes_and_native_tool_calls_fail_closed(self):
        invalid = (None, [], {}, {"choices": []}, {"choices": [None]},
            {"choices": completion()["choices"] * 2}, completion(reason="tool_calls"),
            completion(reason="unknown"), completion(None), completion([]),
            completion(role="user"), completion(tool_calls=[{"function": {"name": "approve"}}]),
            completion(function_call={"name": "approve"}),
            {"choices": [{"finish_reason": "stop", "message": None}]})
        for payload in invalid:
            self.response = httpx.Response(200, json=payload)
            self.assertEqual(self.result().code, "INVALID_RESPONSE")
        self.response = httpx.Response(200, text="not JSON")
        self.assertEqual(self.result().code, "INVALID_RESPONSE")

    def test_only_content_returned_reasoning_never_exposed_and_usage_is_optional(self):
        payload = completion(reasoning="PRIVATE REASONING")
        payload["usage"] = {"prompt_tokens": 9, "completion_tokens": 2048, "total_tokens": 2057}
        self.response = httpx.Response(200, json=payload)
        result = self.provider.generate(self.request, response_schema=AgentDecision.model_json_schema())
        self.assertEqual(result.text, FINAL)
        self.assertEqual(result.usage.model_dump(), {"input_tokens": 9, "output_tokens": 2048, "total_tokens": 2057})
        self.assertNotIn("PRIVATE", result.model_dump_json())
        for usage in (None, [], {"prompt_tokens": True, "completion_tokens": -1, "total_tokens": "secret"}):
            payload["usage"] = usage
            self.response = httpx.Response(200, json=payload)
            self.assertTrue(self.result().ok)

    def test_http_failures_no_retry_no_redirect_no_secret_detail(self):
        for status in (301, 307, 400, 401, 403, 404, 408, 422, 429, 500, 503, 504):
            with self.subTest(status=status):
                self.calls.clear()
                self.response = httpx.Response(status, text="SECRET " + KEY,
                    headers={"Location": "https://untrusted.invalid"})
                result = self.result()
                self.assertEqual(result.code, "TIMEOUT" if status in (408, 504) else "UNAVAILABLE")
                self.assertNotIn("SECRET", result.model_dump_json())
                self.assertNotIn(KEY, result.model_dump_json())
                self.assertEqual(len(self.calls), 1)

    def test_transport_failure_sanitization_and_unexpected_bugs_propagate(self):
        for error, code in ((httpx.ReadTimeout(KEY), "TIMEOUT"), (TimeoutError(KEY), "TIMEOUT"),
                            (httpx.ConnectError(KEY), "UNAVAILABLE")):
            self.error = error
            with self.assertRaises(ProviderFailure) as caught:
                self.provider.generate(self.request, response_schema={})
            self.assertEqual(caught.exception.code, code)
            self.assertTrue(caught.exception.__suppress_context__)
            self.assertNotIn(KEY, str(caught.exception))
        self.error = RuntimeError("adapter bug")
        with self.assertRaisesRegex(RuntimeError, "adapter bug"):
            self.result()

    def test_gpt_oss_multi_turn_no_tool_choice_after_application_tool_step(self):
        """
        Regression: GPT-OSS second-call request must not contain tool_choice or
        native tool fields even when the conversation history includes an
        application-managed assistant decision with kind="tool" followed by a
        tool_result user message.

        This reproduces the exact message structure that exists on LLM call #2
        in a live agent run, confirmed from graph_agent._record_result and
        agent.py:

            LLMMessage(role="assistant",
                       content=AgentDecision(decision=call).model_dump_json())
            LLMMessage(role="user",
                       content=json.dumps({"tool_result": result.model_dump(...)}))

        The assistant content contains {"decision":{"kind":"tool","name":"...","arguments":{...}}},
        which Groq's inference may pattern-match as a prior native tool call.
        The test confirms our request body never includes tool_choice or tools,
        regardless of what the history contains.
        """
        app = create_app()
        instructions = agent_instructions(app.state.local_tools.describe())

        # Exact assistant message appended by both agent.py and graph_agent.py
        # after a successful tool step (AgentDecision.model_dump_json()).
        first_decision_json = AgentDecision(decision=ToolCall(
            kind="tool",
            name="get_case",
            arguments={"case_id": "00000000-0000-0000-0000-000000000001"},
        )).model_dump_json()

        # Exact user message appended after the tool result
        # (json.dumps({"tool_result": result.model_dump(mode="json")})).
        tool_result_json = json.dumps({
            "tool_result": {
                "ok": True,
                "tool": "get_case",
                "data": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "customer_id": "00000000-0000-0000-0000-000000000002",
                    "order_id": "00000000-0000-0000-0000-000000000003",
                    "subject": "Wrong item",
                    "description": "Please review",
                    "status": "open",
                },
            }
        })

        # Second-call LLMRequest: system instructions + case context +
        # assistant decision + tool result — exactly what the agent passes to
        # StructuredLLM on step 2+ of the reasoning loop.
        case_context_json = json.dumps({
            "request": {
                "case_id": "00000000-0000-0000-0000-000000000001",
                "message": "Please review my case",
                "conversation_id": None,
            },
            "case_result": {
                "ok": True,
                "tool": "get_case",
                "data": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "subject": "Wrong item",
                    "description": "Please review",
                },
            },
        })

        self.request = LLMRequest(
            messages=(
                LLMMessage(role="system",    content=instructions),
                LLMMessage(role="user",      content=case_context_json),
                LLMMessage(role="assistant", content=first_decision_json),
                LLMMessage(role="user",      content=tool_result_json),
            ),
            max_output_tokens=AgentConfig().max_output_tokens,
        )

        result = self.result()

        self.assertEqual(len(self.calls), 1)
        sent = json.loads(self.calls[0].content)
        messages = sent["messages"]

        # 1. tool_choice must be ABSENT for GPT-OSS regardless of history content.
        #    Any re-introduction of tool_choice for _GPT_OSS_MODELS will fail here.
        self.assertNotIn("tool_choice", sent,
            "GPT-OSS must not receive tool_choice even when history contains "
            "tool-call-shaped assistant JSON")

        # 2. No native tool registration fields — we never send a tools array.
        for field in ("tools", "functions", "function_call", "disable_tool_validation"):
            self.assertNotIn(field, sent)

        # 3. Expected message count: compatibility instruction prepended by
        #    GroqProvider (1) + the 4 caller messages = 5 total.
        self.assertEqual(len(messages), 5)

        # 4. Compatibility instruction is a user-role message at position 0
        #    (GPT-OSS path: system prompts avoided per Groq recommendation).
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("JSON decision generator", messages[0]["content"])
        self.assertIn("Never emit native tool calls", messages[0]["content"])

        # 5. Caller messages follow the compatibility instruction unchanged,
        #    in exact order: system / user / assistant / user.
        self.assertEqual(
            [m["role"] for m in messages[1:]],
            ["system", "user", "assistant", "user"],
        )

        # 6. The assistant decision is preserved as ordinary text content —
        #    not converted to a structured tool_calls object.
        assistant_msg = messages[3]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertEqual(assistant_msg["content"], first_decision_json)
        self.assertNotIn("tool_calls",    assistant_msg)
        self.assertNotIn("function_call", assistant_msg)

        # 7. The tool result is preserved as ordinary user-role text content —
        #    not a role="tool" message (OpenAI native tool result format).
        tool_result_msg = messages[4]
        self.assertEqual(tool_result_msg["role"], "user")
        self.assertEqual(tool_result_msg["content"], tool_result_json)

        # 8. GPT-OSS-specific parameters are still applied correctly.
        self.assertEqual(sent["reasoning_effort"], "low")
        self.assertEqual(sent["reasoning_format"], "hidden")
        self.assertNotIn("include_reasoning", sent,
            "include_reasoning must be absent — mutually exclusive with reasoning_format")
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        self.assertEqual(sent["model"], "openai/gpt-oss-20b")

        # 9. The full call succeeds (MockTransport returns FINAL).
        self.assertTrue(result.ok)
        self.assertIsInstance(result.data.decision, Finish)


if __name__ == "__main__":
    unittest.main()
