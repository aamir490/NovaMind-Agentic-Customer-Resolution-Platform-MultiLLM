"""Offline admin draft-to-decision flow; no live provider, payment, or fulfillment calls."""

from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from backend.app.domain import Customer
from backend.app.llm import FakeLLMProvider, LLMResponse
from backend.app.main import create_app
from backend.app.schemas import CaseCreate, OrderCreate
from backend.app.security import AuthenticatedIdentity, LocalAuthenticationProvider, LocalCredential, Role


ADMIN_ID = UUID("00000000-0000-4000-8000-000000000091")
CUSTOMER_ID = UUID("00000000-0000-4000-8000-000000000092")
CUSTOMER_USER_ID = UUID("00000000-0000-4000-8000-000000000093")
ADMIN_TOKEN = "admin-resolution-draft-token"
CUSTOMER_TOKEN = "customer-resolution-draft-token"


class AdminResolutionDraftTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "novamind.sqlite"
        self.auth = LocalAuthenticationProvider((
            LocalCredential(token=ADMIN_TOKEN,
                            identity=AuthenticatedIdentity(user_id=ADMIN_ID, role=Role.ADMIN)),
            LocalCredential(token=CUSTOMER_TOKEN, identity=AuthenticatedIdentity(
                user_id=CUSTOMER_USER_ID, role=Role.CUSTOMER, customer_id=CUSTOMER_ID)),
        ))
        self.provider_calls = []

    def app(self):
        def factory():
            self.provider_calls.append(True)
            return FakeLLMProvider(LLMResponse(
                text='{"decision":{"kind":"final"}}', finish_reason="stop",
            ))

        return create_app(auth_provider=self.auth, sqlite_path=self.path,
                          startup_customers=(Customer(id=CUSTOMER_ID, name="Draft customer"),),
                          local_provider_factory=factory)

    @staticmethod
    def headers(token):
        return {"Authorization": f"Bearer {token}"}

    def case(self, app, subject, description):
        order = app.state.case_service.create_order(OrderCreate(
            customer_id=CUSTOMER_ID,
            items=[{"sku": "LAP-1", "name": "Laptop", "quantity": 1}],
        ))
        return app.state.case_service.create_case(CaseCreate(
            customer_id=CUSTOMER_ID, order_id=order.id, subject=subject, description=description,
        ))

    @staticmethod
    def wait_for_completion(client, run_id):
        for _ in range(100):
            snapshot = client.get(f"/api/runs/{run_id}", headers=AdminResolutionDraftTests.headers(ADMIN_TOKEN))
            if snapshot.status_code == 200 and snapshot.json()["status"] == "COMPLETED":
                return snapshot.json()
            time.sleep(0.01)
        raise AssertionError("run did not complete")

    def start(self, client, support_case, message):
        identity = client.get("/api/identity", headers=self.headers(ADMIN_TOKEN)).json()
        response = client.post("/api/runs", headers=self.headers(ADMIN_TOKEN), json={
            "instance_id": identity["instance_id"], "request_id": str(uuid4()),
            "case_id": str(support_case.id), "message": message,
        })
        self.assertEqual(response.status_code, 202, response.text)
        return self.wait_for_completion(client, response.json()["run_id"])

    def test_admin_uses_persisted_draft_once_and_customer_never_receives_staff_reason(self):
        first = self.app()
        support_case = self.case(first, "Damaged laptop", "The product arrived damaged.")
        with TestClient(first) as client:
            self.start(client, support_case, "My laptop arrived damaged.")
            denied = client.get(f"/api/cases/{support_case.id}/resolution-draft",
                                headers=self.headers(CUSTOMER_TOKEN))
            self.assertEqual(denied.status_code, 403)
            draft_response = client.get(f"/api/cases/{support_case.id}/resolution-draft",
                                        headers=self.headers(ADMIN_TOKEN))
            self.assertEqual(draft_response.status_code, 200, draft_response.text)
            draft = draft_response.json()
            self.assertEqual(draft["recommended_decision"], "REQUEST_EVIDENCE")
            self.assertFalse(draft["actions_executed"])

            body = {"action": "REQUEST_MORE_EVIDENCE", "reason": draft["decision_reason"],
                    "customer_message": draft["customer_facing_response"], "evidence_reviewed": False}
            recorded = client.post(f"/api/cases/{support_case.id}/resolution",
                                   headers=self.headers(ADMIN_TOKEN), json=body)
            self.assertEqual(recorded.status_code, 200, recorded.text)
            self.assertEqual(recorded.json()["case"]["status"], "in_review")
            duplicate = client.post(f"/api/cases/{support_case.id}/resolution",
                                    headers=self.headers(ADMIN_TOKEN), json=body)
            self.assertEqual(duplicate.status_code, 409)
            self.assertEqual(len(recorded.json()["history"]), 1)

            customer = client.get(f"/api/cases/{support_case.id}/resolution",
                                  headers=self.headers(CUSTOMER_TOKEN))
            self.assertEqual(customer.status_code, 200, customer.text)
            self.assertNotIn("reason", customer.json()["latest"])
            self.assertNotIn("history", customer.json())
            self.assertNotIn("admin_user_id", customer.json()["latest"])
        first.state.frontend_runtime.close()

        calls_before_read = len(self.provider_calls)
        restarted = self.app()
        with TestClient(restarted) as client:
            restored = client.get(f"/api/cases/{support_case.id}/resolution-draft",
                                  headers=self.headers(ADMIN_TOKEN))
            self.assertEqual(restored.status_code, 200, restored.text)
            self.assertEqual(restored.json()["issue_type"], "damaged")
            self.assertEqual(restored.json()["recommended_decision"], "REQUEST_EVIDENCE")
        self.assertEqual(len(self.provider_calls), calls_before_read)
        restarted.state.frontend_runtime.close()

    def test_escalation_draft_records_only_a_terminal_case_decision(self):
        app = self.app()
        support_case = self.case(app, "Delivery delayed", "The shipment has not arrived.")
        with TestClient(app) as client:
            self.start(client, support_case, "My delivery is late.")
            draft = client.get(f"/api/cases/{support_case.id}/resolution-draft",
                               headers=self.headers(ADMIN_TOKEN)).json()
            self.assertEqual(draft["recommended_decision"], "ESCALATE")
            recorded = client.post(f"/api/cases/{support_case.id}/resolution", headers=self.headers(ADMIN_TOKEN),
                                   json={"action": "ESCALATE", "reason": draft["decision_reason"],
                                         "customer_message": draft["customer_facing_response"],
                                         "evidence_reviewed": False})
            self.assertEqual(recorded.status_code, 200, recorded.text)
            self.assertEqual(recorded.json()["case"]["status"], "escalated")
            self.assertEqual(recorded.json()["latest"]["action"], "ESCALATE")
            self.assertNotIn("actions_executed", recorded.json())
        app.state.frontend_runtime.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
