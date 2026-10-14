"""Gemini status diagnostics with mocked SDK transport; no live provider calls."""

import io
import json
import logging
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

import httpx
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from backend.app.frontend_contracts import TelemetryEvent
from backend.app.gemini import GeminiProvider
from backend.app.llm import LLMMessage, LLMRequest, StructuredLLM
from backend.app.main import create_app
from backend.app.observability import LOGGER, LocalObserver, annotate, observing, span
from backend.app.security import AuthenticatedIdentity, LocalAuthenticationProvider, LocalCredential, authenticated


PRIVATE = "PRIVATE-EXCEPTION-BODY-HEADER-CUSTOMER-DATA"
PROMPT = "PRIVATE-PROMPT-MODEL-OUTPUT"
TOKEN = "PRIVATE-API-KEY-AND-TEST-CREDENTIAL"


class Answer(BaseModel):
    answer: str


class GeminiDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("No network")))
        self.observer = LocalObserver()
        self.auth = LocalAuthenticationProvider(tuple(LocalCredential(token=token, identity=AuthenticatedIdentity(
            user_id=UUID(int=index), role=role, customer_id=UUID(int=10) if role == "CUSTOMER" else None))
            for index, (token, role) in enumerate(((TOKEN, "REVIEWER"), (TOKEN + "-admin", "ADMIN"),
                                                  (TOKEN + "-customer", "CUSTOMER")), 1)))
        self.enterContext(authenticated(self.auth, TOKEN))
        self.enterContext(observing(self.observer))
        self.client = Mock()
        self.provider = GeminiProvider(client=self.client, model="gemini-test-only")
        self.request = LLMRequest(messages=(LLMMessage(role="user", content=PROMPT),))
        self.output = io.StringIO()
        handler = logging.StreamHandler(self.output)
        self.addCleanup(LOGGER.setLevel, LOGGER.level)
        self.addCleanup(LOGGER.removeHandler, handler)
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)

    def api_error(self, status):
        error = errors.APIError(503, {"message": PRIVATE, "status": PRIVATE, "details": {"api_key": TOKEN}},
                               response=httpx.Response(503, headers={"private-header": TOKEN}, text=PRIVATE))
        error.code = status
        return error

    def generate(self, error=None):
        self.client.models.generate_content.reset_mock()
        self.client.models.generate_content.side_effect = error
        result = StructuredLLM(self.provider).generate(self.request, Answer)
        self.client.models.generate_content.assert_called_once()
        for marker in (PRIVATE, PROMPT, TOKEN):
            self.assertNotIn(marker, json.dumps(self.observer.snapshot()))
            self.assertNotIn(marker, self.output.getvalue())
            if not result.ok:
                self.assertNotIn(marker, result.model_dump_json())
        return result

    def test_api_status_is_recorded_without_changing_public_failure_mapping(self):
        for status in (100, 200, 400, 401, 403, 404, 408, 429, 500, 503, 504, 599):
            with self.subTest(status=status):
                code = "TIMEOUT" if status in (408, 504) else "UNAVAILABLE"
                result = self.generate(self.api_error(status))
                self.assertEqual(result.model_dump(), {"ok": False, "code": code,
                                                       "message": "Provider could not complete the request"})
                event = self.observer.snapshot()["events"][-1]
                self.assertEqual((event["provider"], event["provider_status_code"], event["error"]),
                                 ("gemini", status, code))
                self.assertNotIn("status_code", event)
                self.assertEqual(json.loads(self.output.getvalue().splitlines()[-1]), event)

    def test_invalid_sdk_status_is_omitted_without_changing_mapping(self):
        for status in (None, True, False, "503", 503.0, 408.0, 99, 600, -1, PRIVATE, {}, []):
            with self.subTest(status_type=type(status).__name__):
                result = self.generate(self.api_error(status))
                self.assertEqual(result.code, "TIMEOUT" if status in (408, 504) else "UNAVAILABLE")
                self.assertNotIn("provider_status_code", self.observer.snapshot()["events"][-1])

    def test_transport_timeouts_success_and_refusal_do_not_inherit_provider_status(self):
        self.generate(self.api_error(429))
        for error, code in ((httpx.ReadTimeout(PRIVATE), "TIMEOUT"), (TimeoutError(PRIVATE), "TIMEOUT"),
                            (httpx.ConnectError(PRIVATE), "UNAVAILABLE")):
            self.assertEqual(self.generate(error).code, code)
        self.client.models.generate_content.return_value = types.GenerateContentResponse(candidates=[
            types.Candidate(finish_reason="STOP", content=types.Content(role="model",
                parts=[types.Part(text=json.dumps({"answer": PROMPT}))]))])
        self.assertTrue(self.generate().ok)
        self.client.models.generate_content.return_value = types.GenerateContentResponse(
            prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"))
        self.assertEqual(self.generate().code, "REFUSED")
        for event in self.observer.snapshot()["events"][1:]:
            self.assertNotIn("provider_status_code", event)

    def test_metadata_and_contract_accept_only_integer_status_in_range(self):
        for status in (None, True, False, "429", 429.0, 99, 600, PRIVATE, [], {}):
            with span("llm"):
                annotate(provider_status_code=status, response_body=PRIVATE, headers={"key": TOKEN})
            event = self.observer.snapshot()["events"][-1]
            self.assertNotIn("provider_status_code", event)
            self.assertNotIn("response_body", event)
            self.assertNotIn("headers", event)
            if status is not None:
                with self.assertRaises(ValidationError):
                    TelemetryEvent.model_validate({**event, "provider_status_code": status})
        self.assertIsNone(TelemetryEvent.model_validate(event).provider_status_code)
        self.assertIsNone(TelemetryEvent.model_validate({**event, "provider_status_code": None}).provider_status_code)
        for status in (100, 429, 599):
            self.assertEqual(TelemetryEvent.model_validate({**event, "provider_status_code": status}).provider_status_code, status)
        for field in ("message", "response_body", "headers", "prompt", "api_key"):
            with self.assertRaises(ValidationError):
                TelemetryEvent.model_validate({**event, field: PRIVATE})

    async def test_diagnostics_expose_safe_provider_status_only_to_privileged_roles(self):
        self.generate(self.api_error(503))
        self.generate(httpx.ReadTimeout(PRIVATE))
        app = create_app(auth_provider=self.auth, observer=self.observer)
        async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://local.test") as client:
            for token, expected in ((None, 401), (TOKEN + "-customer", 403), (TOKEN, 200), (TOKEN + "-admin", 200)):
                response = await client.get("/api/diagnostics", headers={"Authorization": f"Bearer {token}"} if token else {})
                self.assertEqual(response.status_code, expected)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                for marker in (PRIVATE, PROMPT, TOKEN):
                    self.assertNotIn(marker, response.text)
                if expected == 200:
                    events = [event for event in response.json()["events"] if event["operation"] == "llm"]
                    self.assertEqual([event["provider_status_code"] for event in events], [503, None])
                    self.assertEqual([event["error"] for event in events], ["UNAVAILABLE", "TIMEOUT"])
                    self.assertTrue(all(event["status_code"] is None for event in events))

    def test_logging_failure_does_not_change_error_or_retry_provider(self):
        with patch.object(LOGGER, "info", side_effect=RuntimeError(PRIVATE)):
            self.assertEqual(self.generate(self.api_error(504)).code, "TIMEOUT")
        snapshot = self.observer.snapshot()
        self.assertEqual(snapshot["events"][-1]["provider_status_code"], 504)
        self.assertEqual(snapshot["logging_failures"], 1)
        self.assertEqual(snapshot["metrics"]["llm"]["provider_calls"], 1)
