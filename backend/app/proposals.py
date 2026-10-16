"""Local proposal/review records. No action execution or authenticated identity."""

from datetime import datetime, timezone
from enum import StrEnum
from threading import RLock
from uuid import UUID, uuid4

from .domain import Description, Name, Record
from .operations import BusinessOperations
from .schemas import RequestModel
from .service import CaseService, ConflictError, NotFoundError
from .observability import observed


class ProposedAction(StrEnum):
    RETURN = "return"
    REFUND = "refund"
    REPLACEMENT = "replacement"


class ProposalStatus(StrEnum):
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ProposalCreate(RequestModel):
    case_id: UUID
    action: ProposedAction
    rationale: Description
    eligibility_evidence_id: UUID | None = None


class HumanReview(RequestModel):
    reviewer_name: Name
    note: Description


class ResolutionProposal(Record):
    id: UUID
    case_id: UUID
    action: ProposedAction
    rationale: Description
    eligibility_evidence_id: UUID | None = None
    status: ProposalStatus
    created_at: datetime
    updated_at: datetime
    reviewed_at: datetime | None = None
    reviewer_name: Name | None = None
    review_note: Description | None = None


class ProposalService:
    def __init__(self, cases: CaseService, operations: BusinessOperations) -> None:
        self._cases = cases
        self._operations = operations
        self._proposals: dict[UUID, ResolutionProposal] = {}
        self._lock = RLock()

    @observed("proposal.create")
    def create(self, request: ProposalCreate) -> ResolutionProposal:
        request = ProposalCreate.model_validate(request.model_dump())
        case = self._cases.get_case(request.case_id)
        if request.action == ProposedAction.REFUND:
            if request.eligibility_evidence_id is None:
                raise ConflictError("Trusted eligibility evidence is required for a refund proposal")
            try:
                evidence = self._operations.get_eligibility_evidence(request.eligibility_evidence_id)
            except NotFoundError:
                raise ConflictError("Trusted eligibility evidence is required for a refund proposal") from None
            result = evidence.result
            if (evidence.case_id != case.id
                    or evidence.order_id != case.order_id
                    or evidence.customer_id != case.customer_id
                    or result.evidence_id != evidence.id
                    or result.case_id != evidence.case_id
                    or result.order_id != evidence.order_id
                    or result.customer_id != evidence.customer_id
                    or result.sku != evidence.sku
                    or result.assessment_basis != "trusted_delivery_record"):
                raise ConflictError("Eligibility evidence does not match the selected case")
            if not result.refund_eligible or result.denial_reasons:
                raise ConflictError("Refund is not eligible under the supplied evidence")
        elif request.eligibility_evidence_id is not None:
            raise ConflictError("Eligibility evidence is only valid for refund proposals")
        with self._lock:
            now = datetime.now(timezone.utc)
            proposal = ResolutionProposal(
                id=uuid4(), **request.model_dump(), status=ProposalStatus.PENDING_REVIEW,
                created_at=now, updated_at=now,
            )
            self._proposals[proposal.id] = proposal
            return proposal

    def get(self, proposal_id: UUID) -> ResolutionProposal:
        with self._lock:
            if proposal_id not in self._proposals:
                raise NotFoundError("Proposal not found")
            return self._proposals[proposal_id]

    def list_for_case(self, case_id: UUID) -> list[ResolutionProposal]:
        self._cases.get_case(case_id)
        with self._lock:
            return [proposal for proposal in self._proposals.values() if proposal.case_id == case_id]

    @observed("proposal.review")
    def review(self, proposal_id: UUID, decision: ProposalStatus,
               review: HumanReview) -> ResolutionProposal:
        if decision not in (ProposalStatus.APPROVED, ProposalStatus.REJECTED):
            raise ConflictError("Review decision must be APPROVED or REJECTED")
        with self._lock:
            proposal = self.get(proposal_id)
            if proposal.status != ProposalStatus.PENDING_REVIEW:
                raise ConflictError("Proposal has already been reviewed")
            now = datetime.now(timezone.utc)
            updated = ResolutionProposal(**{
                **proposal.model_dump(), "status": decision,
                "updated_at": now, "reviewed_at": now,
                "reviewer_name": review.reviewer_name, "review_note": review.note,
            })
            self._proposals[proposal_id] = updated
            return updated
