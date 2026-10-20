"""Offline M5E3 durable LangGraph HITL workflow recovery tests."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import test_agent as baseline
from backend.app.agent import AgentConfig, AgentRequest
from backend.app.hitl import HITLWorkflow
from backend.app.langgraph_checkpoints import SQLiteCheckpointSaver
from backend.app.llm import StructuredLLM
from backend.app.operations import BusinessOperations
from backend.app.persistence import (
    SQLiteCaseRepository, SQLitePersistence, SQLiteProposalRepository,
    SQLiteWorkflowMetadataRepository,
)
from backend.app.proposals import ProposalService, ProposalStatus
from backend.app.schemas import CaseCreate, CustomerCreate, OrderCreate
from backend.app.security import SecurityError
from backend.app.service import CaseService
from backend.app.tools import LocalTools
from backend.app.workflow_metadata import WorkflowMetadataStatus
from security_fixtures import call_as_admin


class SQLiteHITLWorkflowTests(unittest.TestCase):
    """A durable workflow is explicitly composed; FastAPI stays in-memory."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "novamind.sqlite"
        store = SQLitePersistence(self.path, initialize=True)
        self.cases = CaseService(SQLiteCaseRepository(store))
        self.proposals = ProposalService(
            self.cases, BusinessOperations(self.cases), SQLiteProposalRepository(store),
        )
        self.tools = LocalTools(self.cases, BusinessOperations(self.cases), self.proposals)
        customer = self.cases.create_customer(CustomerCreate(name="Durable workflow customer"))
        order = self.cases.create_order(OrderCreate(
            customer_id=customer.id,
            items=[{"sku": "LAP-1", "name": "Laptop", "quantity": 1}],
        ))
        self.case = self.cases.create_case(CaseCreate(
            customer_id=customer.id, order_id=order.id, subject="Workflow recovery",
            description="Persist the interrupted human-review workflow.",
        ))
        self.request = AgentRequest(case_id=self.case.id, message="Please review my case")

    def workflow(self, script):
        store = SQLitePersistence(self.path)
        return HITLWorkflow(
            StructuredLLM(baseline.ScriptedProvider(script)), self.tools, self.proposals,
            AgentConfig(max_steps=4), checkpointer=SQLiteCheckpointSaver(store),
            workflow_metadata=SQLiteWorkflowMetadataRepository(store),
        )

    def proposal(self):
        return baseline.call("create_resolution_proposal", case_id=str(self.case.id),
                             action="replacement", rationale="A reviewer must decide.")

    @staticmethod
    def decision(paused, *, decision="APPROVE"):
        review = paused.review
        return {
            "workflow_id": str(review.workflow_id), "case_id": str(review.case_id),
            "proposal_id": str(review.proposal_id), "review_id": str(review.review_id),
            "decision": decision, "note": "A staff reviewer checked the evidence.",
        }

    def test_restart_recovers_checkpoint_review_binding_and_audit(self):
        original = self.workflow([self.proposal()])
        paused = call_as_admin(original.start, self.request)
        self.assertEqual(paused.status, "REVIEW_REQUIRED")

        # A new service/checkpointer simulates process restart without replaying the LLM.
        recovered = self.workflow([])
        self.assertEqual(call_as_admin(recovered.audit, paused.workflow_id)[-1].kind, "REVIEW_REQUIRED")
        completed = call_as_admin(recovered.resume, self.decision(paused))
        self.assertEqual(completed.status, "REVIEWED")
        self.assertEqual(self.proposals.get(paused.review.proposal_id).status, ProposalStatus.APPROVED)

        metadata = SQLiteWorkflowMetadataRepository(SQLitePersistence(self.path))
        self.assertEqual(metadata.get(paused.workflow_id).status, WorkflowMetadataStatus.REVIEWED)
        self.assertEqual(
            [event.status for event in metadata.events(paused.workflow_id)],
            [WorkflowMetadataStatus.RUNNING, WorkflowMetadataStatus.REVIEW_REQUIRED,
             WorkflowMetadataStatus.RESUMING, WorkflowMetadataStatus.REVIEWED],
        )
        proposal_events = SQLiteProposalRepository(SQLitePersistence(self.path)).audit_events(paused.workflow_id)
        self.assertEqual([event["kind"] for event in proposal_events], ["REVIEW_REQUIRED", "HUMAN_REVIEW"])

    def test_recovered_review_requires_staff_authorization(self):
        paused = call_as_admin(self.workflow([self.proposal()]).start, self.request)
        self.assertEqual(paused.status, "REVIEW_REQUIRED", paused.model_dump_json())
        recovered = self.workflow([])
        with self.assertRaises(SecurityError):
            recovered.resume(self.decision(paused))
        self.assertEqual(self.proposals.get(paused.review.proposal_id).status, ProposalStatus.PENDING_REVIEW)
        self.assertEqual(
            SQLiteWorkflowMetadataRepository(SQLitePersistence(self.path)).get(paused.workflow_id).status,
            WorkflowMetadataStatus.REVIEW_REQUIRED,
        )

    def test_concurrent_recovered_reviews_have_one_durable_winner(self):
        paused = call_as_admin(self.workflow([self.proposal()]).start, self.request)
        self.assertEqual(paused.status, "REVIEW_REQUIRED", paused.model_dump_json())
        first, second = self.workflow([]), self.workflow([])
        approve, reject = self.decision(paused), self.decision(paused, decision="REJECT")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda values: call_as_admin(values[0].resume, values[1]),
                [(first, approve), (second, reject)],
            ))
        self.assertEqual(sum(getattr(result, "status", None) == "REVIEWED" for result in results), 1)
        self.assertEqual(sum(getattr(result, "error", None) == "NOT_PAUSED" for result in results), 1)
        self.assertIn(
            self.proposals.get(paused.review.proposal_id).status,
            (ProposalStatus.APPROVED, ProposalStatus.REJECTED),
        )

    def test_duplicate_recovered_review_is_rejected_without_new_audit_write(self):
        paused = call_as_admin(self.workflow([self.proposal()]).start, self.request)
        self.assertEqual(paused.status, "REVIEW_REQUIRED", paused.model_dump_json())
        payload = self.decision(paused)
        self.assertEqual(call_as_admin(self.workflow([]).resume, payload).status, "REVIEWED")
        repository = SQLiteProposalRepository(SQLitePersistence(self.path))
        before = repository.audit_events(paused.workflow_id)
        duplicate = call_as_admin(self.workflow([]).resume, payload)
        self.assertEqual(duplicate.error, "NOT_PAUSED")
        self.assertEqual(repository.audit_events(paused.workflow_id), before)
