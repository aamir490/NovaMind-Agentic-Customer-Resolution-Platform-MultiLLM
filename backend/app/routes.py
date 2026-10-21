"""Thin local HTTP routes; relationships and transitions live in the service."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request

from .domain import (
    CaseResolutionHistory, Customer, CustomerCaseResolution, CustomerCommunication,
    CustomerResolutionDecision, Order, SupportCase,
)
from .schemas import (
    CaseResolutionCreate, CustomerCommunicationCreate, CustomerCreate,
    InvestigationJobBind, OrderCreate,
    CaseCreate, CaseStatusUpdate,
)
from .service import CaseService, ConflictError, NotFoundError
from .security import Authorization, Role, current_identity, require_roles
from .workflow_metadata import WorkflowReservationResult
from .resolution_drafts import ResolutionDraft

router = APIRouter(prefix="/api")


def get_service(request: Request) -> CaseService:
    return request.app.state.case_service


Service = Annotated[CaseService, Depends(get_service)]


@router.post("/customers", response_model=Customer, status_code=201, tags=["customers"])
def create_customer(body: CustomerCreate, service: Service) -> Customer:
    require_roles(Role.ADMIN)
    return service.create_customer(body)


@router.get("/customers/{customer_id}", response_model=Customer, tags=["customers"])
def get_customer(customer_id: UUID, service: Service) -> Customer:
    Authorization.customer(customer_id)
    return service.get_customer(customer_id)


@router.post("/orders", response_model=Order, status_code=201, tags=["orders"])
def create_order(body: OrderCreate, service: Service) -> Order:
    require_roles(Role.ADMIN)
    return service.create_order(body)


@router.get("/orders/{order_id}", response_model=Order, tags=["orders"])
def get_order(order_id: UUID, service: Service) -> Order:
    Authorization(service).order(order_id)
    return service.get_order(order_id)


@router.post("/cases", response_model=SupportCase, status_code=201, tags=["cases"])
def create_case(body: CaseCreate, service: Service) -> SupportCase:
    Authorization.customer(body.customer_id)
    Authorization(service).order(body.order_id)
    return service.create_case(body)


@router.get("/cases", response_model=list[SupportCase], tags=["cases"])
def list_cases(service: Service, customer_id: UUID | None = None) -> list[SupportCase]:
    identity = current_identity()
    if customer_id is not None:
        Authorization.customer(customer_id)
    if identity.role == Role.CUSTOMER:
        customer_id = identity.customer_id
    return service.list_cases(customer_id)


@router.get("/cases/{case_id}", response_model=SupportCase, tags=["cases"])
def get_case(case_id: UUID, service: Service) -> SupportCase:
    Authorization(service).case(case_id)
    return service.get_case(case_id)


@router.post("/investigation-jobs/{job_id}/workflow-reservation",
             response_model=WorkflowReservationResult, tags=["investigation-jobs"])
def bind_investigation_job(job_id: UUID, body: InvestigationJobBind,
                           request: Request, service: Service) -> WorkflowReservationResult:
    """Staff-only durable reservation binding; it does not start a workflow."""
    identity = require_roles(Role.REVIEWER, Role.ADMIN)
    jobs = request.app.state.investigation_jobs
    if jobs is None:
        raise NotFoundError("Investigation jobs are not enabled")
    job = jobs.get(job_id)
    Authorization(service).case(job.case_id)
    runtime = request.app.state.frontend_runtime
    if runtime.agent_engine != "langgraph":
        raise ConflictError("Durable investigation reservations require LangGraph")
    return jobs.bind_claim(
        job_id, worker_id=identity.user_id, claim_id=body.claim_id,
        provider=runtime.provider_label,
    )


def _resolution_history(case_id: UUID, service: CaseService) -> CaseResolutionHistory:
    case = service.get_case(case_id)
    history = service.resolution_history(case_id)
    return CaseResolutionHistory(case=case, latest=history[-1] if history else None, history=history)


@router.get("/cases/{case_id}/resolution", response_model=CaseResolutionHistory | CustomerCaseResolution,
            tags=["cases"])
def get_resolution(case_id: UUID, service: Service) -> CaseResolutionHistory | CustomerCaseResolution:
    identity = current_identity()
    Authorization(service).case(case_id)
    resolution = _resolution_history(case_id, service)
    if identity.role != Role.CUSTOMER:
        return resolution
    latest = resolution.latest
    return CustomerCaseResolution(
        case=resolution.case,
        latest=None if latest is None else CustomerResolutionDecision(
            action=latest.action,
            customer_message=latest.customer_message, decided_at=latest.decided_at,
        ),
    )


@router.get("/cases/{case_id}/resolution-draft", response_model=ResolutionDraft | None, tags=["cases"])
def get_resolution_draft(case_id: UUID, request: Request, service: Service) -> ResolutionDraft | None:
    """Admin-only projection of the validated draft persisted in workflow state."""
    require_roles(Role.ADMIN)
    Authorization(service).case(case_id)
    return request.app.state.frontend_runtime.resolution_draft_for_case(case_id)


@router.post("/cases/{case_id}/resolution", response_model=CaseResolutionHistory, tags=["cases"])
def create_resolution(case_id: UUID, body: CaseResolutionCreate, service: Service) -> CaseResolutionHistory:
    identity = require_roles(Role.ADMIN)
    Authorization(service).case(case_id)
    service.resolve_case(
        case_id, action=body.action, reason=body.reason,
        customer_message=body.customer_message, admin_user_id=identity.user_id,
        evidence_reviewed=body.evidence_reviewed,
    )
    return _resolution_history(case_id, service)


def _customer_communication(case_id: UUID, service: CaseService) -> CustomerCommunication | None:
    communication = service.latest_customer_communication(case_id)
    if communication is None:
        return None
    return CustomerCommunication(message=communication.message, sent_at=communication.sent_at)


@router.post(
    "/cases/{case_id}/customer-communications",
    response_model=CustomerCommunication,
    status_code=201,
    tags=["cases"],
)
def send_customer_communication(
    case_id: UUID, body: CustomerCommunicationCreate, service: Service,
    idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
) -> CustomerCommunication:
    """Persist explicit ADMIN-approved guidance; never mutate case state or execute an action."""
    identity = require_roles(Role.ADMIN)
    Authorization(service).case(case_id)
    communication, _created = service.send_customer_communication(
        case_id, message=body.message, admin_user_id=identity.user_id,
        idempotency_key=idempotency_key,
    )
    return CustomerCommunication(message=communication.message, sent_at=communication.sent_at)


@router.get(
    "/cases/{case_id}/customer-communications/latest",
    response_model=CustomerCommunication | None,
    tags=["cases"],
)
def get_latest_customer_communication(
    case_id: UUID, service: Service,
) -> CustomerCommunication | None:
    """Return only customer-safe text and time after enforcing case ownership."""
    Authorization(service).case(case_id)
    return _customer_communication(case_id, service)


@router.patch("/cases/{case_id}/status", response_model=SupportCase, tags=["cases"])
def update_status(case_id: UUID, body: CaseStatusUpdate, service: Service) -> SupportCase:
    require_roles(Role.REVIEWER, Role.ADMIN)
    Authorization(service).case(case_id)
    return service.update_status(case_id, body.status)
