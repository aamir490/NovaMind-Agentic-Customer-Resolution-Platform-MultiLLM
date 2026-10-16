"""Focused M3A coverage for deterministic LangGraph refund routing."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import test_agent as baseline

from backend.app.agent import AgentRequest
from backend.app.graph_agent import GraphResolutionAgent
from backend.app.hitl import HITLWorkflow
from backend.app.llm import ProviderFailure, StructuredLLM
from backend.app.operations import AuthoritativeDeliveryFact


class RefundOrchestrationTests(baseline.AgentTests):
    def refund_request(self):
        return AgentRequest(case_id=self.case.id, message="Please issue a refund for this order.")

    def ingest_delivery_fact(self, *, delivered_days_ago=10):
        now = datetime.now(timezone.utc)
        self.delivery_ingestion.ingest(self.delivery_source.id, AuthoritativeDeliveryFact(
            id=uuid4(), case_id=self.case.id, order_id=self.order.id,
            customer_id=self.case.customer_id, sku="LAP-1", quantity=1,
            delivered_at=now - timedelta(days=delivered_days_ago),
            reason="wrong_item", condition="unused", source_id=self.delivery_source.id,
            source_reference=f"m3a-delivery-{delivered_days_ago}", verified_at=now,
        ))

    def test_eligible_refund_creates_one_pending_proposal_and_pauses_for_review(self):
        self.ingest_delivery_fact()
        provider = baseline.ScriptedProvider([baseline.FINAL])
        workflow = HITLWorkflow(StructuredLLM(provider), self.tools, self.proposals)

        result = workflow.start(self.refund_request())

        self.assertEqual(result.status, "REVIEW_REQUIRED")
        self.assertEqual(result.review.action, "refund")
        self.assertEqual(len(self.proposals.list_for_case(self.case.id)), 1)
        self.assertEqual(self.proposals.list_for_case(self.case.id)[0].status, "PENDING_REVIEW")
        self.assertEqual(provider.requests, [])
        self.assertFalse(result.actions_executed)

    def test_ineligible_refund_creates_no_proposal_and_returns_verified_denial(self):
        self.ingest_delivery_fact(delivered_days_ago=31)
        provider = baseline.ScriptedProvider([baseline.FINAL])

        result = GraphResolutionAgent(StructuredLLM(provider), self.tools).run(self.refund_request())

        self.assertEqual(result.status, "INFORMATIONAL")
        self.assertIn("denied under verified policy", result.customer_response)
        self.assertIn("outside_return_window", result.customer_response)
        self.assertEqual(self.proposals.list_for_case(self.case.id), [])
        self.assertEqual(provider.requests, [])

    def test_missing_evidence_creates_no_proposal_and_returns_unresolved_outcome(self):
        provider = baseline.ScriptedProvider([baseline.FINAL])

        result = GraphResolutionAgent(StructuredLLM(provider), self.tools).run(self.refund_request())

        self.assertEqual(result.status, "INFORMATIONAL")
        self.assertIn("trusted delivery evidence is unavailable", result.customer_response)
        self.assertEqual(self.proposals.list_for_case(self.case.id), [])
        self.assertEqual(provider.requests, [])

    def test_premature_model_final_cannot_resolve_an_eligible_refund(self):
        self.ingest_delivery_fact()
        provider = baseline.ScriptedProvider([baseline.FINAL])

        result = GraphResolutionAgent(StructuredLLM(provider), self.tools).run(self.refund_request())

        self.assertEqual(result.status, "HUMAN_REVIEW_REQUIRED")
        self.assertEqual(len(result.pending_proposal_ids), 1)
        self.assertEqual(self.proposals.get(result.pending_proposal_ids[0]).action, "refund")
        self.assertEqual(provider.requests, [])

    def test_provider_failure_cannot_bypass_refund_evidence_evaluation(self):
        provider = baseline.ScriptedProvider([ProviderFailure("UNAVAILABLE")])

        result = GraphResolutionAgent(StructuredLLM(provider), self.tools).run(self.refund_request())

        self.assertEqual(result.status, "INFORMATIONAL")
        self.assertIn("trusted delivery evidence is unavailable", result.customer_response)
        self.assertEqual(self.proposals.list_for_case(self.case.id), [])
        self.assertEqual(provider.requests, [])

