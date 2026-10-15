"""Explicit provider selection and mocked Groq transport through the existing runtime."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import yaml

from backend.app.agent import AgentDecision, ToolCall, agent_instructions
from backend.app.groq import GroqProvider
from backend.app.schemas import CaseCreate, CustomerCreate, OrderCreate
from backend.container.bootstrap import _llm_provider, create_container_app


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"NOVAMIND_LLM_PROVIDER": "groq", "GROQ_API_KEY": "test-only-groq-key"}
TOKEN = "container-groq-admin-test-token"


class ContainerGroqTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("No network")))
        self.enterContext(patch("socket.getaddrinfo", side_effect=AssertionError("No DNS")))
        self.calls = []
        self.decisions = []
        self.client = self.enterContext(httpx.Client(transport=httpx.MockTransport(self.handle)))
        self.constructor = self.enterContext(patch("backend.app.groq.httpx.Client", return_value=self.client))

    def handle(self, request):
        self.calls.append(request)
        content = self.decisions.pop(0) if self.decisions else '{"decision":{"kind":"final"}}'
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": content}}]})

    async def request(self, app, method, path, body=None, token=TOKEN):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local.test") as client:
            return await client.request(method, path, json=body,
                headers={"Authorization": f"Bearer {token}"} if token else {})

    def test_absent_selector_keeps_provider_unavailable_even_with_groq_key(self):
        self.assertIsNone(_llm_provider())
        os.environ["GROQ_API_KEY"] = CONFIG["GROQ_API_KEY"]
        self.assertIsNone(_llm_provider())
        self.assertIsNone(create_container_app().state.frontend_runtime.provider_factory)
        self.constructor.assert_not_called()

    def test_stale_gemini_configuration_fails_closed_with_or_without_groq_selection(self):
        for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_FUTURE"):
            for selected in (None, "groq"):
                with self.subTest(name=name, selected=selected):
                    config = {name: ""}
                    if selected is not None:
                        config.update(CONFIG)
                    with patch.dict(os.environ, config, clear=True):
                        with self.assertRaisesRegex(RuntimeError, "^Container LLM provider selection is invalid$"):
                            create_container_app()
        self.constructor.assert_not_called()

    def test_explicit_selection_uses_groq(self):
        os.environ.update(CONFIG)
        with _llm_provider() as provider:
            self.assertIsInstance(provider, GroqProvider)
            self.assertEqual(provider.model, "openai/gpt-oss-20b")
        self.constructor.assert_called_once()
        self.assertEqual(self.calls, [])

    def test_selected_groq_missing_configuration_fails_closed(self):
        invalid = ({"NOVAMIND_LLM_PROVIDER": "groq"},
                   {**CONFIG, "GROQ_API_KEY": ""}, {**CONFIG, "GROQ_MODEL": ""},
                   {**CONFIG, "GROQ_MODEL": "https://SECRET.invalid"})
        for config in invalid:
            with patch.dict(os.environ, config, clear=True):
                with self.assertRaisesRegex(RuntimeError, "^Container Groq configuration is invalid$") as caught:
                    create_container_app()
                self.assertTrue(caught.exception.__suppress_context__)
                self.assertNotIn("SECRET", str(caught.exception))
        self.constructor.assert_not_called()

    def test_invalid_selector_fails_closed(self):
        for selector in ("", " ", "GROQ", "gemini", "bedrock", "SECRET"):
            os.environ["NOVAMIND_LLM_PROVIDER"] = selector
            with self.assertRaisesRegex(RuntimeError, "^Container LLM provider selection is invalid$"):
                create_container_app()
        self.constructor.assert_not_called()

    def test_client_construction_failure_sanitized_and_no_fallback(self):
        os.environ.update(CONFIG)
        self.constructor.side_effect = ValueError("SECRET configuration")
        with patch("backend.container.bootstrap.create_app") as app_factory:
            with self.assertRaisesRegex(RuntimeError, "^Container Groq configuration is invalid$") as caught:
                create_container_app()
            self.assertTrue(caught.exception.__suppress_context__)
            app_factory.assert_not_called()

    def test_authentication_validated_first_and_app_failure_closes_groq(self):
        os.environ.update(CONFIG)
        os.environ["NOVAMIND_AUTH_FILE"] = "relative-invalid.json"
        with self.assertRaisesRegex(RuntimeError, "^Container authentication configuration is invalid$"):
            create_container_app()
        self.constructor.assert_not_called()
        del os.environ["NOVAMIND_AUTH_FILE"]
        with patch("backend.container.bootstrap.create_app", side_effect=RuntimeError("app failure")):
            with self.assertRaisesRegex(RuntimeError, "app failure"):
                create_container_app()
        self.assertTrue(self.client.is_closed)

    async def test_groq_configuration_grants_no_authentication(self):
        os.environ.update(CONFIG)
        app = create_container_app()
        async with app.router.lifespan_context(app):
            for path in ("/api/identity", "/api/cases"):
                self.assertEqual((await self.request(app, "GET", path)).status_code, 401)
        self.assertEqual(self.calls, [])
        self.assertTrue(self.client.is_closed)

    async def test_authenticated_runtime_uses_real_adapter_and_closes_shared_client_after_drain(self):
        os.environ.update(CONFIG)
        directory = Path(self.enterContext(TemporaryDirectory()))
        auth = directory / "auth.json"
        auth.write_text(json.dumps([{"token": TOKEN, "identity": {
            "user_id": str(UUID(int=17)), "role": "ADMIN"}}]), encoding="utf-8")
        os.environ["NOVAMIND_AUTH_FILE"] = str(auth)
        app = create_container_app()
        runtime = app.state.frontend_runtime
        self.assertIsInstance(runtime.provider_factory(), GroqProvider)
        self.assertIs(runtime.provider_factory(), runtime.provider_factory())
        self.constructor.assert_called_once()
        self.assertEqual(self.calls, [])
        invoke = self.enterContext(patch.object(app.state.local_tools, "invoke", wraps=app.state.local_tools.invoke))
        async with app.router.lifespan_context(app):
            identity = await self.request(app, "GET", "/api/identity")
            self.assertEqual(identity.json()["provider"], "groq")
            service = app.state.case_service
            customer = service.create_customer(CustomerCreate(name="Groq wiring fixture"))
            order = service.create_order(OrderCreate(customer_id=customer.id,
                items=[{"sku": "FIXTURE", "name": "Fixture", "quantity": 1}]))
            case = service.create_case(CaseCreate(customer_id=customer.id, order_id=order.id,
                subject="Fixture", description="Mocked runtime test"))
            decision = AgentDecision(decision=ToolCall(kind="tool", name="get_order",
                arguments={"order_id": str(order.id)}))
            self.decisions = [decision.model_dump_json(), '{"decision":{"kind":"final"}}']
            body = {"instance_id": str(runtime.instance_id), "request_id": str(uuid4()),
                    "case_id": str(case.id), "message": "Review fixture"}
            denied = await self.request(app, "POST", "/api/runs", body, token=None)
            self.assertEqual(denied.status_code, 401)
            self.assertEqual(self.calls, [])
            started = await self.request(app, "POST", "/api/runs", body)
            self.assertEqual(started.status_code, 202, started.text)
            self.assertEqual(started.json()["provider"], "groq")
        snapshot = await self.request(app, "GET", f"/api/runs/{started.json()['run_id']}")
        self.assertEqual(snapshot.json()["status"], "COMPLETED", snapshot.text)
        self.assertEqual(snapshot.json()["provider"], "groq")
        runs = await self.request(app, "GET", f"/api/cases/{case.id}/runs?offset=0&limit=50")
        self.assertEqual(runs.json()["items"][0]["provider"], "groq")
        self.assertFalse(snapshot.json()["actions_executed"])
        self.assertNotIn(CONFIG["GROQ_API_KEY"], snapshot.text)
        self.assertTrue(runtime._closed)
        self.assertTrue(self.client.is_closed)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual([call.args for call in invoke.call_args_list], [
            ("get_case", {"case_id": str(case.id)}), ("get_order", {"order_id": str(order.id)})])
        for request in self.calls:
            sent = json.loads(request.content)
            self.assertEqual(sent["max_completion_tokens"], 2048)
            self.assertEqual(sent["response_format"], {"type": "json_object"})
            # GPT-OSS (openai/gpt-oss-20b): tool_choice must be absent.
            self.assertNotIn("tool_choice", sent,
                "GPT-OSS must not receive tool_choice; Groq rejects it for reasoning models")
            for option in ("tools", "functions", "function_call", "disable_tool_validation"):
                self.assertNotIn(option, sent)
            boundary = sent["messages"][0]["content"]
            self.assertIn("not a native tool-calling assistant", boundary)
            self.assertIn("Never emit native tool calls", boundary)
            self.assertIn("application-managed operations described as JSON data", boundary)
            self.assertIn("ordinary text in your final assistant response", boundary)
            self.assertEqual(json.loads(boundary.split("JSON Schema: ", 1)[1]), AgentDecision.model_json_schema())
            self.assertEqual(sent["messages"][1], {"role": "system",
                "content": agent_instructions(app.state.local_tools.describe())})
        followup = json.loads(self.calls[1].content)["messages"]
        self.assertEqual(followup[-2], {"role": "assistant", "content": decision.model_dump_json()})
        tool_result = json.loads(followup[-1]["content"])["tool_result"]
        self.assertTrue(tool_result["ok"])
        self.assertEqual(tool_result["tool"], "get_order")
        self.assertEqual(tool_result["data"]["id"], str(order.id))

    def test_compose_overlay_is_explicit_backend_only_runtime_configuration(self):
        config = yaml.safe_load((ROOT / "compose.groq.yaml").read_text())
        self.assertEqual(set(config["services"]), {"backend"})
        backend = config["services"]["backend"]
        self.assertEqual(set(backend), {"environment", "networks"})
        self.assertEqual(backend["networks"], ["application", "edge"])
        self.assertEqual(backend["environment"], {
            "NOVAMIND_LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "${GROQ_API_KEY:?Inject GROQ_API_KEY into the invoking process}",
            "GROQ_MODEL": "${GROQ_MODEL:-openai/gpt-oss-20b}"})
        base = yaml.safe_load((ROOT / "compose.yaml").read_text())
        self.assertEqual(base["services"]["backend"]["networks"], ["application"])
        self.assertNotIn("GROQ_API_KEY", base["services"]["backend"]["environment"])


if __name__ == "__main__":
    unittest.main()
