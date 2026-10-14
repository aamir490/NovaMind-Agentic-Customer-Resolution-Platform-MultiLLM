"""Local proposal/review records. No action execution or authenticated identity."""

from datetime import datetime, timezone
from enum import StrEnum
from threading import RLock
from uuid import UUID, uuid4

from .domain import Description, Name, Record
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


class HumanReview(RequestModel):
    reviewer_name: Name
    note: Description


class ResolutionProposal(Record):
    id: UUID
    case_id: UUID
    action: ProposedAction
    rationale: Description
    status: ProposalStatus
    created_at: datetime
    updated_at: datetime
    reviewed_at: datetime | None = None
    reviewer_name: Name | None = None
    review_note: Description | None = None


class ProposalService:
    def __init__(self, cases: CaseService) -> None:
        self._cases = cases
        self._proposals: dict[UUID, ResolutionProposal] = {}
        self._lock = RLock()

    @observed("proposal.create")
    def create(self, request: ProposalCreate) -> ResolutionProposal:
        self._cases.get_case(request.case_id)
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
