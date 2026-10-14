"""Phase 7A tests: injected SDK clients only, never live API requests."""

import os
import unittest
from unittest.mock import Mock, patch

import httpx
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict

from backend.app.gemini import GeminiProvider
from backend.app.llm import LLMMessage, LLMRequest, StructuredLLM


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


class GeminiTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.client = Mock()
        self.provider = GeminiProvider(client=self.client)
        self.request = LLMRequest(messages=(LLMMessage(role="user", content="Question"),))
        self.client.models.generate_content.return_value = self.response()

    @staticmethod
    def response(text='{"answer":"fixture"}', reason="STOP", parts=None):
        return types.GenerateContentResponse(candidates=[types.Candidate(
            finish_reason=reason, content=types.Content(role="model", parts=parts or [types.Part(text=text)]),
        )])

    def result(self):
        return StructuredLLM(self.provider).generate(self.request, Answer)

    def test_request_mapping_and_strict_structured_success(self):
        self.request = LLMRequest(messages=(
            LLMMessage(role="system", content="First instruction"),
            LLMMessage(role="system", content="Second instruction"),
            LLMMessage(role="user", content="Question"),
            LLMMessage(role="assistant", content="Prior answer"),
            LLMMessage(role="user", content="Followup"),
        ), max_output_tokens=123)
        self.assertEqual(self.result().data.answer, "fixture")
        kwargs = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-3.8-flash")
        self.assertEqual([c.role for c in kwargs["contents"]], ["user", "model", "user"])
        self.assertEqual(kwargs["contents"][2].parts[0].text, "Followup")
        config = kwargs["config"]
        self.assertEqual(config.system_instruction, "First instruction\n\nSecond instruction")
        self.assertEqual(config.max_output_tokens, 123)
        self.assertEqual(config.response_json_schema, Answer.model_json_schema())
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertTrue(config.automatic_function_calling.disable)
        self.assertIsNone(config.tools)

    def test_environment_key_model_timeout_and_client_cleanup(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only-placeholder", "GEMINI_MODEL": "configured-model"}, clear=True), \
                patch("backend.app.gemini.genai.Client") as constructor:
            with GeminiProvider(timeout_ms=1000) as provider:
                self.assertEqual(provider.model, "configured-model")
            kwargs = constructor.call_args.kwargs
            self.assertEqual(kwargs["api_key"], "test-only-placeholder")
            self.assertFalse(kwargs["vertexai"])
            self.assertEqual(kwargs["http_options"].timeout, 1000)
            self.assertEqual(kwargs["http_options"].retry_options.attempts, 1)
            constructor.return_value.close.assert_called_once()
            self.assertEqual(GeminiProvider(model="override", client=self.client).model, "override")
        self.provider.close()
        self.client.close.assert_not_called()

    def test_missing_key_and_invalid_configuration_fail_before_client_creation(self):
        with patch.dict(os.environ, {}, clear=True), patch("backend.app.gemini.genai.Client") as constructor:
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                GeminiProvider()
            constructor.assert_not_called()
        for kwargs in ({"model": " "}, {"timeout_ms": 0}, {"timeout_ms": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GeminiProvider(client=self.client, **kwargs)

    def test_unsupported_message_order_rejected_before_network(self):
        for messages in ((LLMMessage(role="system", content="Only system"),),
                         (*self.request.messages, LLMMessage(role="system", content="Late instruction"))):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.provider.generate(LLMRequest(messages=messages), response_schema=Answer.model_json_schema())
        self.client.models.generate_content.assert_not_called()

    def test_http_status_failures_use_phase_6_contract_without_leaking_details(self):
        for status in (400, 401, 403, 404, 408, 429, 500, 503, 504):
            with self.subTest(status=status):
                self.client.models.generate_content.side_effect = errors.APIError(status, {"message": "private detail"})
                result = self.result()
                self.assertEqual(result.code, "TIMEOUT" if status in (408, 504) else "UNAVAILABLE")
                self.assertNotIn("private detail", result.model_dump_json())

    def test_transport_failure_and_timeout(self):
        for error, code in ((httpx.ReadTimeout("private"), "TIMEOUT"),
                            (TimeoutError("private"), "TIMEOUT"),
                            (httpx.ConnectError("private"), "UNAVAILABLE")):
            with self.subTest(code=code):
                self.client.models.generate_content.side_effect = error
                self.assertEqual(self.result().code, code)

    def test_refusal_and_truncation_override_even_valid_json(self):
        for reason, code in (("MAX_TOKENS", "INCOMPLETE"), ("SAFETY", "REFUSED"),
                             ("RECITATION", "REFUSED"), ("PROHIBITED_CONTENT", "REFUSED")):
            with self.subTest(reason=reason):
                self.client.models.generate_content.return_value = self.response(reason=reason)
                self.assertEqual(self.result().code, code)
        self.client.models.generate_content.return_value = types.GenerateContentResponse(
            prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"),
        )
        self.assertEqual(self.result().code, "REFUSED")

    def test_invalid_json_schema_empty_and_unknown_finish_fail_closed(self):
        for response in (self.response("bad JSON"), self.response('{"answer":1}'),
                         self.response('{"answer":"x","approve":true}'),
                         self.response(reason="OTHER"), types.GenerateContentResponse(),
                         types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP")])):
            with self.subTest(response=response):
                self.client.models.generate_content.return_value = response
                self.assertEqual(self.result().code, "INVALID_RESPONSE")

    def test_thoughts_are_excluded_and_text_parts_combined(self):
        self.client.models.generate_content.return_value = self.response(parts=[
            types.Part(text="Internal reasoning", thought=True),
            types.Part(text='{"answer":'), types.Part(text='"fixture"}'),
        ])
        self.assertEqual(self.result().data.answer, "fixture")

    def test_function_call_is_not_executed_or_accepted_as_text(self):
        self.client.models.generate_content.return_value = self.response(parts=[
            types.Part(text='{"answer":"fixture"}'),
            types.Part(function_call=types.FunctionCall(name="approve", args={})),
        ])
        self.assertEqual(self.result().code, "INVALID_RESPONSE")
        self.assertEqual(self.client.models.generate_content.call_count, 1)

    def test_unexpected_bug_is_not_hidden(self):
        self.client.models.generate_content.side_effect = RuntimeError("bug")
        with self.assertRaises(RuntimeError):
            self.result()


if __name__ == "__main__":
    unittest.main()
