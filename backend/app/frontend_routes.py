"""Phase 17A.1 HTTP contracts. Credentials/configuration/state never come from bodies."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from .conversations import ConversationError
from .frontend_contracts import (
    AuditPage, ConversationPage, DiagnosticsResponse, EventPage, IdentityResponse, MessagePage,
    OperationMetrics, PendingReviewPage, RunDecision, RunPage, RunSnapshot, RunStart, TelemetryEvent,
)
from .frontend_runtime import FrontendError, FrontendRuntime, now
from .observability import OPERATIONS
from .security import Authorization, Role, current_identity, require_roles


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


@router.get("/identity", response_model=IdentityResponse)
def identity(runtime: Runtime):
    subject = current_identity()
    privileged = subject.role in (Role.REVIEWER, Role.ADMIN)
    return IdentityResponse(identity=subject, can_review=privileged, can_view_diagnostics=privileged,
        instance_id=runtime.instance_id, provider="local_scripted" if runtime.provider_factory else "unavailable",
        memory_available=runtime.memory is not None)


@router.get("/reviews", response_model=PendingReviewPage)
def pending_reviews(runtime: Runtime):
    return runtime.pending_reviews()


@router.post("/runs", response_model=RunSnapshot, status_code=202)
def start_run(body: RunStart, request: Request, runtime: Runtime):
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
