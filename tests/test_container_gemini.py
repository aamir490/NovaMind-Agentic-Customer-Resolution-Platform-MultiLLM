"""Focused container Gemini wiring checks; the SDK transport is always mocked."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from google.genai import types
import httpx
import yaml

from backend.app.gemini import GeminiProvider
from backend.app.schemas import CaseCreate, CustomerCreate, OrderCreate
from backend.container.bootstrap import create_container_app


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "container-gemini-admin-test-token"
CONFIG = {"GEMINI_API_KEY": "test-only-key", "GEMINI_MODEL": "gemini-test-model"}


class ContainerGeminiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("No network")))
        self.enterContext(patch("socket.getaddrinfo", side_effect=AssertionError("No DNS")))
        self.sdk = self.enterContext(patch("backend.app.gemini.genai.Client"))
        self.directory = Path(self.enterContext(TemporaryDirectory()))

    def auth(self):
        path = self.directory / "auth.json"
        path.write_text(json.dumps([{"token": TOKEN, "identity": {
            "user_id": str(UUID(int=17)), "role": "ADMIN"}}]), encoding="utf-8")
        os.environ["NOVAMIND_AUTH_FILE"] = str(path)

    async def request(self, app, method, path, body=None, token=TOKEN):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local.test") as client:
            return await client.request(method, path, json=body,
                                        headers={"Authorization": f"Bearer {token}"} if token else {})

    def run_body(self, app):
        service = app.state.case_service
        customer = service.create_customer(CustomerCreate(name="Wiring fixture"))
        order = service.create_order(OrderCreate(customer_id=customer.id,
            items=[{"sku": "FIXTURE", "name": "Fixture item", "quantity": 1}]))
        case = service.create_case(CaseCreate(customer_id=customer.id, order_id=order.id,
                                            subject="Fixture", description="Wiring test only"))
        return {"instance_id": str(app.state.frontend_runtime.instance_id), "request_id": str(uuid4()),
                "case_id": str(case.id), "message": "Review this case"}

    async def test_absent_configuration_keeps_provider_unavailable_and_auth_required(self):
        self.auth()
        # The SDK's other credential variables must not implicitly enable Gemini.
        os.environ["GOOGLE_API_KEY"] = "ignored-test-key"
        app = create_container_app()
        async with app.router.lifespan_context(app):
            identity = await self.request(app, "GET", "/api/identity")
            self.assertEqual(identity.json()["provider"], "unavailable")
            body = self.run_body(app)
            denied = await self.request(app, "POST", "/api/runs", body, token=None)
            self.assertEqual(denied.status_code, 401)
            unavailable = await self.request(app, "POST", "/api/runs", body)
            self.assertEqual(unavailable.status_code, 503)
            self.assertEqual(unavailable.json()["detail"], "PROVIDER_UNAVAILABLE")
        self.sdk.assert_not_called()

    async def test_valid_configuration_uses_real_adapter_for_authenticated_case_run(self):
        self.auth()
        os.environ.update(CONFIG)
        self.sdk.return_value.models.generate_content.return_value = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason="STOP", content=types.Content(role="model",
                parts=[types.Part(text='{"decision":{"kind":"final"}}')]))])
        app = create_container_app()
        runtime = app.state.frontend_runtime
        self.assertIsInstance(runtime.provider_factory(), GeminiProvider)
        self.assertIs(runtime.provider_factory(), runtime.provider_factory())
        self.sdk.assert_called_once()
        options = self.sdk.call_args.kwargs
        self.assertEqual(options["api_key"], CONFIG["GEMINI_API_KEY"])
        self.assertFalse(options["vertexai"])
        self.assertEqual(options["http_options"].retry_options.attempts, 1)
        self.assertEqual(options["http_options"].timeout, 30000)
        self.sdk.return_value.close.side_effect = lambda: self.assertTrue(runtime._closed)
        async with app.router.lifespan_context(app):
            identity = await self.request(app, "GET", "/api/identity")
            self.assertEqual(identity.status_code, 200)
            self.assertNotEqual(identity.json()["provider"], "unavailable")
            self.assertNotIn(CONFIG["GEMINI_API_KEY"], identity.text)
            self.sdk.return_value.models.generate_content.assert_not_called()
            body = self.run_body(app)
            for token in (None, "invalid-token"):
                denied = await self.request(app, "POST", "/api/runs", body, token=token)
                self.assertEqual(denied.status_code, 401)
            self.sdk.return_value.models.generate_content.assert_not_called()
            started = await self.request(app, "POST", "/api/runs", body)
            self.assertEqual(started.status_code, 202, started.text)
            self.assertEqual(started.json()["case_id"], body["case_id"])
            self.assertFalse(started.json()["actions_executed"])
        # Lifespan drains workers; no polling, sleeps, or live provider calls needed.
        snapshot = await self.request(app, "GET", f"/api/runs/{started.json()['run_id']}")
        self.assertEqual(snapshot.json()["status"], "COMPLETED", snapshot.text)
        self.assertFalse(snapshot.json()["actions_executed"])
        self.assertNotIn(CONFIG["GEMINI_API_KEY"], snapshot.text)
        self.sdk.return_value.models.generate_content.assert_called_once()
        self.assertEqual(self.sdk.return_value.models.generate_content.call_args.kwargs["model"],
                         CONFIG["GEMINI_MODEL"])
        self.sdk.return_value.close.assert_called_once()

    async def test_gemini_configuration_does_not_bootstrap_authentication(self):
        os.environ.update(CONFIG)
        app = create_container_app()
        async with app.router.lifespan_context(app):
            for path in ("/api/identity", "/api/cases"):
                self.assertEqual((await self.request(app, "GET", path)).status_code, 401)
            denied = await self.request(app, "POST", "/api/runs", self.run_body(app))
            self.assertEqual(denied.status_code, 401)
        self.sdk.return_value.models.generate_content.assert_not_called()
        self.sdk.return_value.close.assert_called_once()

    def test_partial_blank_malformed_configuration_fails_closed_without_secrets(self):
        invalid = ({"GEMINI_MODEL": "gemini-test"}, {"GEMINI_API_KEY": "SECRET"},
                   {**CONFIG, "GEMINI_API_KEY": ""}, {**CONFIG, "GEMINI_API_KEY": " SECRET "},
                   {**CONFIG, "GEMINI_API_KEY": "SECRET\nVALUE"},
                   {**CONFIG, "GEMINI_API_KEY": "x" * 4097},
                   {**CONFIG, "GEMINI_API_KEY": "SECRET\x00"},
                   {**CONFIG, "GEMINI_MODEL": ""}, {**CONFIG, "GEMINI_MODEL": " "},
                   {**CONFIG, "GEMINI_MODEL": "https://SECRET.invalid"})
        for config in invalid:
            with self.subTest(config=list(config)), patch.dict(os.environ, config, clear=True):
                with self.assertRaisesRegex(RuntimeError, "^Container Gemini configuration is invalid$") as caught:
                    create_container_app()
                self.assertTrue(caught.exception.__suppress_context__)
                self.assertNotIn("SECRET", str(caught.exception))
        self.sdk.assert_not_called()

    def test_sdk_construction_failure_is_sanitized_and_never_creates_app(self):
        os.environ.update(CONFIG)
        self.sdk.side_effect = ValueError("SECRET SDK configuration")
        with patch("backend.container.bootstrap.create_app") as factory:
            with self.assertRaisesRegex(RuntimeError, "^Container Gemini configuration is invalid$") as caught:
                create_container_app()
            self.assertTrue(caught.exception.__suppress_context__)
            factory.assert_not_called()

    def test_authentication_configuration_is_validated_before_gemini(self):
        os.environ.update(CONFIG)
        os.environ["NOVAMIND_AUTH_FILE"] = str(self.directory / "missing.json")
        with self.assertRaisesRegex(RuntimeError, "^Container authentication configuration is invalid$"):
            create_container_app()
        self.sdk.assert_not_called()

    def test_app_construction_failure_closes_provider(self):
        os.environ.update(CONFIG)
        with patch("backend.container.bootstrap.create_app", side_effect=RuntimeError("app failure")):
            with self.assertRaisesRegex(RuntimeError, "app failure"):
                create_container_app()
        self.sdk.return_value.close.assert_called_once()

    def test_gemini_compose_overlay_is_explicit_backend_only_runtime_configuration(self):
        config = yaml.safe_load((ROOT / "compose.gemini.yaml").read_text())
        self.assertEqual(set(config["services"]), {"backend"})
        backend = config["services"]["backend"]
        self.assertEqual(set(backend), {"environment", "networks"})
        self.assertEqual(backend["networks"], ["application", "edge"])
        self.assertEqual(set(backend["environment"]), {"GEMINI_API_KEY", "GEMINI_MODEL"})
        for name, value in backend["environment"].items():
            self.assertTrue(value.startswith("${" + name + ":?"))
        base = yaml.safe_load((ROOT / "compose.yaml").read_text())
        self.assertEqual(base["services"]["backend"]["networks"], ["application"])
        self.assertNotIn("GEMINI_API_KEY", base["services"]["backend"]["environment"])
