"""Server-only sanitized Groq errors; mocked HTTP, unchanged workflow failures."""

import io
import json
import logging
import os
import unittest
from unittest.mock import patch

import httpx

from backend.app.agent import AgentDecision
from backend.app.groq import GroqProvider
from backend.app.llm import LLMMessage, LLMRequest, StructuredLLM


KEY = "gsk_test_private_credential"
PROMPT = 'Private customer message with "quoted" text'


class GroqDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"GROQ_API_KEY": KEY}, clear=True))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("No network")))
        self.response = httpx.Response(400, json={"error": {"message":
            "'response_format' is not supported with 'reasoning_effort'"}})
        self.calls = []
        self.client = self.enterContext(httpx.Client(transport=httpx.MockTransport(self.handle)))
        self.provider = GroqProvider(client=self.client)
        self.request = LLMRequest(messages=(LLMMessage(role="user", content=PROMPT),), max_output_tokens=2048)
        self.output = io.StringIO()
        self.logger = logging.getLogger("novamind.groq")
        handler = logging.StreamHandler(self.output)
        self.addCleanup(self.logger.removeHandler, handler)
        self.addCleanup(self.logger.setLevel, self.logger.level)
        self.enterContext(patch.object(self.logger, "propagate", False))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)

    def handle(self, request):
        self.calls.append(request)
        return self.response

    def result(self):
        return StructuredLLM(self.provider).generate(self.request, AgentDecision)

    def test_http_message_logged_but_public_failure_and_request_unchanged(self):
        result = self.result()
        self.assertEqual(result.model_dump(), {"ok": False, "code": "UNAVAILABLE",
            "message": "Provider could not complete the request"})
        self.assertEqual(self.output.getvalue(),
            "Groq HTTP 400: 'response_format' is not supported with 'reasoning_effort'\n")
        request, = self.calls
        body = json.loads(request.content)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_completion_tokens"], 2048)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["reasoning_format"], "hidden")
        self.assertNotIn("include_reasoning", body)
        self.assertEqual(body["messages"][-1], {"role": "user", "content": PROMPT})
        self.assertEqual(request.headers["Authorization"], f"Bearer {KEY}")

    def test_only_message_is_logged_credentials_and_request_echoes_redacted(self):
        escaped = json.dumps(PROMPT)[1:-1]
        self.response = httpx.Response(400, headers={"secret-header": "PRIVATE_HEADER"}, json={
            "error": {"message": f"Invalid {KEY}; Bearer other-secret; gsk_other_secret; sk-other-secret; "
                f"api_key=another-secret; access_token: 'token-secret'; {PROMPT}; {escaped}",
                "failed_generation": "PRIVATE_MODEL_OUTPUT", "details": "PRIVATE_DETAILS"},
            "messages": "PRIVATE_BODY"})
        self.assertEqual(self.result().code, "UNAVAILABLE")
        logged = self.output.getvalue()
        self.assertIn("Groq HTTP 400: Invalid [redacted]", logged)
        for private in (KEY, "other-secret", "gsk_other_secret", "sk-other-secret", "another-secret",
                        "token-secret", PROMPT, escaped, "PRIVATE_HEADER", "PRIVATE_MODEL_OUTPUT",
                        "PRIVATE_DETAILS", "PRIVATE_BODY"):
            self.assertNotIn(private, logged)

    def test_reproduced_native_tool_error_remains_sanitized_without_retry_or_fallback(self):
        message = "Tool choice is none, but model called a tool"
        self.response = httpx.Response(400, json={"error": {
            "message": message, "failed_generation": "PRIVATE_NATIVE_TOOL_PAYLOAD"}})
        result = self.result()
        self.assertEqual(result.model_dump(), {"ok": False, "code": "UNAVAILABLE",
            "message": "Provider could not complete the request"})
        self.assertEqual(self.output.getvalue(), f"Groq HTTP 400: {message}\n")
        self.assertNotIn("PRIVATE_NATIVE_TOOL_PAYLOAD", self.output.getvalue())
        request, = self.calls
        # GPT-OSS: tool_choice must be absent — Groq rejects it for reasoning models.
        self.assertNotIn("tool_choice", json.loads(request.content))

    def test_redaction_precedes_truncation_and_control_characters_cannot_forge_lines(self):
        self.response = httpx.Response(400, json={"error": {"message":
            "x" * 495 + KEY + "\r\n\x1b[31m\u202eFORGED" + "y" * 900}})
        self.result()
        logged = self.output.getvalue()
        self.assertEqual(len(logged.splitlines()), 1)
        message = logged.removeprefix("Groq HTTP 400: ").rstrip("\n")
        self.assertEqual(len(message), 512)
        self.assertTrue(message.endswith("..."))
        self.assertNotIn(KEY[:10], logged)
        self.assertNotIn("\x1b", logged)
        self.assertNotIn("\u202e", logged)

    def test_nonjson_missing_wrong_type_and_oversized_errors_use_fixed_fallback(self):
        responses = [httpx.Response(502, text="PRIVATE_HTML"), httpx.Response(400, json=[]),
            httpx.Response(400, json={"error": "PRIVATE_ERROR"}),
            httpx.Response(400, json={"error": {"message": {"private": "PRIVATE_DATA"}}}),
            httpx.Response(400, json={"error": {"message": ""}}),
            httpx.Response(400, json={"error": {"message": "PRIVATE_OVERSIZE" * 6000}})]
        for response in responses:
            with self.subTest(status=response.status_code):
                self.output.seek(0)
                self.output.truncate()
                self.response = response
                self.assertEqual(self.result().code, "UNAVAILABLE")
                self.assertEqual(self.output.getvalue(),
                    f"Groq HTTP {response.status_code}: Provider error message unavailable\n")

    def test_all_non_success_statuses_keep_mapping_and_do_not_retry_or_redirect(self):
        for status in (301, 400, 401, 403, 404, 408, 429, 500, 503, 504):
            with self.subTest(status=status):
                self.calls.clear()
                self.response = httpx.Response(status, json={"error": {"message": "Request rejected"}},
                    headers={"Location": "https://untrusted.invalid"})
                self.assertEqual(self.result().code, "TIMEOUT" if status in (408, 504) else "UNAVAILABLE")
                self.assertEqual(len(self.calls), 1)
                self.assertIn(f"Groq HTTP {status}: Request rejected", self.output.getvalue())

    def test_diagnostic_parsing_or_logging_failure_does_not_replace_original_failure(self):
        for target in ("backend.app.groq._LOGGER.warning", "httpx.Response.json"):
            with patch(target, side_effect=RuntimeError("PRIVATE_FAILURE")):
                self.assertEqual(self.result().code, "UNAVAILABLE")
        self.assertNotIn("PRIVATE_FAILURE", self.output.getvalue())
        self.assertEqual(len(self.calls), 2)

    def test_success_and_transport_errors_do_not_log_http_error_messages(self):
        self.response = httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": '{"decision":{"kind":"final"}}'}}]})
        self.assertTrue(self.result().ok)
        with patch.object(self.client, "post", side_effect=httpx.ReadTimeout(KEY)):
            self.assertEqual(self.result().code, "TIMEOUT")
        self.assertEqual(self.output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
