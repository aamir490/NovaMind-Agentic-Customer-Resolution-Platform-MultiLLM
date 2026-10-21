"""Phase 17A.1 HTTP contracts. Credentials/configuration/state never come from bodies."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from .conversations import ConversationError
from .frontend_contracts import (
    AuditPage, ConversationPage, DiagnosticsResponse, EventPage, IdentityResponse, MessagePage,
    OperationMetrics, PendingReviewPage, RunDecision, RunPage, RunSnapshot, RunStart, TelemetryEvent,
    CaseInvestigationSnapshot, InvestigationControlSnapshot, InvestigationControlUpdate,
    InvestigationModeAuditItem, InvestigationModeAuditPage,
)
from .frontend_runtime import FrontendError, FrontendRuntime, now
from .observability import OPERATIONS
from .security import Authorization, Role, current_identity, require_roles
from .domain import InvestigationJobStatus, InvestigationMode
from .service import NotFoundError
from .workflow_metadata import WorkflowMetadataStatus


class ContractRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def validated(request):
            try:
                return await handler(request)
            except RequestValidationError:
                # FastAPI's default validation response echoes rejected input.
                raise FrontendError(422, "INVALID_INPUT") from None
        return validated


router = APIRouter(prefix="/api", tags=["frontend-contracts"], route_class=ContractRoute)
Limit = Annotated[int, Query(ge=1, le=100)]
Cursor = Annotated[int, Query(ge=0, le=1000000)]


def get_runtime(request: Request) -> FrontendRuntime:
    return request.app.state.frontend_runtime


Runtime = Annotated[FrontendRuntime, Depends(get_runtime)]


def _control_repository(request: Request):
    repository = request.app.state.investigation_control
    if repository is None:
        raise FrontendError(503, "INVESTIGATION_CONTROL_UNAVAILABLE")
    return repository


def _control_snapshot(request: Request) -> InvestigationControlSnapshot:
    repository = _control_repository(request)
    control = repository.get()
    available = request.app.state.investigation_dispatcher is not None
    effective = control.mode if (control.mode is InvestigationMode.MANUAL or available) else InvestigationMode.MANUAL
    return InvestigationControlSnapshot(
        configured_mode=control.mode, effective_mode=effective,
        automatic_capability_available=available, version=control.version,
        updated_by=control.updated_by, updated_at=control.updated_at,
        counts=repository.status_counts(),
    )


def _case_investigation_snapshot(request: Request, case_id: UUID) -> CaseInvestigationSnapshot:
    Authorization(request.app.state.case_service).case(case_id)
    jobs = request.app.state.investigation_jobs
    if jobs is None:
        raise FrontendError(503, "INVESTIGATION_CONTROL_UNAVAILABLE")
    try:
        job = jobs.get_by_case(case_id)
    except NotFoundError:
        return CaseInvestigationSnapshot(
            case_id=case_id, state="NOT_AVAILABLE", execution_started=False,
            can_start=False,
        )
    workflow = None
    if job.workflow_id is not None:
        metadata = request.app.state.workflow_metadata
        if metadata is None:
            raise FrontendError(503, "INVESTIGATION_STATUS_UNAVAILABLE")
        try:
            workflow = metadata.get(job.workflow_id)
        except NotFoundError:
            raise FrontendError(503, "INVESTIGATION_STATUS_UNAVAILABLE") from None
    if job.status is InvestigationJobStatus.PENDING:
        state = "PENDING"
    elif (job.status is InvestigationJobStatus.RECOVERY_REQUIRED
          or workflow is not None and workflow.status is WorkflowMetadataStatus.RECOVERY_REQUIRED):
        state = "RECOVERY_REQUIRED"
    elif workflow is not None and workflow.status is WorkflowMetadataStatus.REVIEW_REQUIRED:
        state = "REVIEW_REQUIRED"
    elif workflow is not None and workflow.status in (
        WorkflowMetadataStatus.COMPLETED, WorkflowMetadataStatus.REVIEWED,
    ):
        state = "COMPLETED"
    elif workflow is not None and workflow.status is WorkflowMetadataStatus.FAILED:
        state = "FAILED"
    else:
        state = "RUNNING"
    identity = current_identity()
    manual_mode = _control_repository(request).get().mode is InvestigationMode.MANUAL
    return CaseInvestigationSnapshot(
        case_id=case_id, job_id=job.id, state=state, job_status=job.status,
        workflow_id=job.workflow_id,
        workflow_status=workflow.status if workflow is not None else None,
        provider=workflow.provider if workflow is not None else None,
        submission_mode=job.submission_mode, mode_version=job.mode_version,
        submitted_at=job.created_at,
        execution_started=job.execution_attempt_id is not None,
        can_start=(identity.role is Role.ADMIN
                   and job.status is InvestigationJobStatus.PENDING
                   and manual_mode
                   and request.app.state.investigation_executor is not None),
    )


@router.get("/identity", response_model=IdentityResponse)
def identity(runtime: Runtime):
    subject = current_identity()
    privileged = subject.role in (Role.REVIEWER, Role.ADMIN)
    return IdentityResponse(identity=subject, can_review=privileged, can_view_diagnostics=privileged,
        instance_id=runtime.instance_id, provider=runtime.provider_label if (runtime.provider_factory or runtime.strands_model_factory) else "unavailable",
        memory_available=runtime.memory is not None)


@router.get("/admin/investigation-control", response_model=InvestigationControlSnapshot)
def investigation_control(request: Request):
    require_roles(Role.REVIEWER, Role.ADMIN)
    return _control_snapshot(request)


@router.patch("/admin/investigation-control", response_model=InvestigationControlSnapshot)
def update_investigation_control(
        body: InvestigationControlUpdate, request: Request,
        idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
):
    identity = require_roles(Role.ADMIN)
    if body.mode is InvestigationMode.AUTO and request.app.state.investigation_dispatcher is None:
        raise FrontendError(409, "AUTO_INVESTIGATION_UNAVAILABLE")
    _control_repository(request).set_mode(
        body.mode, expected_version=body.expected_version,
        admin_user_id=identity.user_id, idempotency_key=idempotency_key,
    )
    # A mode change is control-plane only. It deliberately does not enumerate,
    # claim, submit, or replay any existing investigation job.
    return _control_snapshot(request)


@router.get("/admin/investigation-control/audit", response_model=InvestigationModeAuditPage)
def investigation_control_audit(request: Request, after: Cursor = 0, limit: Limit = 100):
    require_roles(Role.ADMIN)
    events = _control_repository(request).audit(after=after, limit=limit + 1)
    has_more = len(events) > limit
    events = events[:limit]
    items = tuple(InvestigationModeAuditItem(
        sequence=event.sequence, previous_mode=event.previous_mode, mode=event.mode,
        version=event.version, admin_user_id=event.admin_user_id,
        changed_at=event.changed_at,
    ) for event in events)
    next_after = items[-1].sequence if items else after
    return InvestigationModeAuditPage(
        items=items, next_after=next_after, has_more=has_more,
    )


@router.get("/cases/{case_id}/investigation", response_model=CaseInvestigationSnapshot)
def case_investigation(case_id: UUID, request: Request):
    require_roles(Role.REVIEWER, Role.ADMIN)
    return _case_investigation_snapshot(request, case_id)


@router.post("/cases/{case_id}/investigation", response_model=CaseInvestigationSnapshot,
             status_code=202)
def start_case_investigation(
        case_id: UUID, request: Request,
        idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
):
    identity = require_roles(Role.ADMIN)
    Authorization(request.app.state.case_service).case(case_id)
    jobs = request.app.state.investigation_jobs
    executor = request.app.state.investigation_executor
    if jobs is None or executor is None:
        raise FrontendError(503, "INVESTIGATION_EXECUTION_UNAVAILABLE")
    if _control_repository(request).get().mode is not InvestigationMode.MANUAL:
        raise FrontendError(409, "MANUAL_INVESTIGATION_DISABLED")
    try:
        job = jobs.get_by_case(case_id)
    except NotFoundError:
        raise FrontendError(409, "DURABLE_INVESTIGATION_JOB_REQUIRED") from None
    registered, _ = jobs.register_manual_start(
        job.id, admin_user_id=identity.user_id, idempotency_key=idempotency_key,
    )
    if registered.status is InvestigationJobStatus.PENDING:
        if not executor.submit_as(registered.id, worker_id=identity.user_id):
            raise FrontendError(503, "INVESTIGATION_EXECUTION_UNAVAILABLE")
    return _case_investigation_snapshot(request, case_id)


@router.get("/reviews", response_model=PendingReviewPage)
def pending_reviews(runtime: Runtime):
    return runtime.pending_reviews()


@router.post("/runs", response_model=RunSnapshot, status_code=202)
def start_run(body: RunStart, request: Request, runtime: Runtime):
    require_roles(Role.REVIEWER, Role.ADMIN)
    return runtime.start(body, trace_id=UUID(request.state.trace_id))


@router.get("/runs/{run_id}", response_model=RunSnapshot)
def get_run(run_id: UUID, runtime: Runtime):
    return runtime.snapshot(run_id)


@router.get("/runs/{run_id}/events", response_model=EventPage)
def get_events(run_id: UUID, runtime: Runtime, after: Cursor = 0, limit: Limit = 50):
    return runtime.events(run_id, after, limit)


@router.get("/cases/{case_id}/runs", response_model=RunPage)
def list_runs(case_id: UUID, runtime: Runtime, offset: Cursor = 0, limit: Limit = 50):
    return runtime.list_for_case(case_id, offset, limit)


@router.post("/runs/{run_id}/decision", response_model=RunSnapshot, status_code=202)
def decide(run_id: UUID, body: RunDecision, request: Request, runtime: Runtime):
    return runtime.decide(run_id, body, trace_id=UUID(request.state.trace_id))


def memory_for_case(request, runtime, case_id):
    Authorization(request.app.state.case_service).case(case_id)
    if runtime.memory is None:
        raise FrontendError(503, "CONVERSATION_UNAVAILABLE")
    return runtime.memory


@router.get("/cases/{case_id}/conversations", response_model=ConversationPage)
def conversations(case_id: UUID, request: Request, runtime: Runtime, offset: Cursor = 0, limit: Limit = 50):
    records = memory_for_case(request, runtime, case_id).list(case_id)
    return ConversationPage(items=records[offset:offset + limit],
        next_offset=offset + limit if offset + limit < len(records) else None)


@router.get("/cases/{case_id}/conversations/{conversation_id}", response_model=MessagePage)
def history(case_id: UUID, conversation_id: UUID, request: Request, runtime: Runtime,
            after: Cursor = 0, limit: Limit = 50):
    saved = memory_for_case(request, runtime, case_id).load(conversation_id, case_id)
    messages = tuple(message for message in saved.messages if message.sequence > after)[:limit]
    cursor = messages[-1].sequence if messages else after
    return MessagePage(conversation=saved.conversation, messages=messages, next_after=cursor,
                       has_more=cursor < saved.conversation.message_count)


@router.get("/runs/{run_id}/audit", response_model=AuditPage)
def audit(run_id: UUID, runtime: Runtime, after: Cursor = 0, limit: Limit = 50):
    return runtime.audit(run_id, after, limit)


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def diagnostics(request: Request, limit: Limit = 50):
    require_roles(Role.REVIEWER, Role.ADMIN)
    snapshot = request.app.state.observer.snapshot()
    # Observer already supplies a closed, redacted metadata schema. Explicitly
    # select fields again at the HTTP boundary; no internal object serialization.
    events = [{key: value for key, value in event.items() if key in TelemetryEvent.model_fields}
              for event in snapshot["events"][-limit:]]
    return {"snapshot_at": now(), "scope": "process_metadata_only", "events": events,
            "metrics": {operation: {key: value for key, value in metric.items() if key in OperationMetrics.model_fields}
                        for operation, metric in snapshot["metrics"].items() if operation in OPERATIONS},
            "evicted_events": snapshot["evicted_events"],
            "omitted_events": max(0, len(snapshot["events"]) - limit),
            "logging_failures": snapshot["logging_failures"]}


def conversation_status(error: ConversationError):
    return {"CONVERSATION_NOT_FOUND": 404, "CONVERSATION_CASE_NOT_FOUND": 404,
            "CONVERSATION_CASE_MISMATCH": 409, "CONVERSATION_FULL": 409}.get(error.code, 503)
